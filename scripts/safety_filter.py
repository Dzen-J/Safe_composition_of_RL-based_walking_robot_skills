# scripts/safety_filter.py
"""
Predictive safety filter for policy switching on Microduck — v3.

Fixes from v2:
- Shadow rollout uses a SEPARATE MjData object (self.data_shadow).
  The real self.data is never mutated by shadow.
- This eliminates state leakage (qacc_warmstart, efc_*, contact solver,
  nefc, act, etc.) that progressively diverged real trajectories between
  naive and filtered runs even when switch_step was identical.

Architecture:
  self.data        — real simulation, owned by the caller (experiment script)
  self.data_shadow — private copy for shadow rollout, never returned to caller

Shadow rollout:
  1. Copy qpos/qvel/act/time/qacc_warmstart from real → shadow
  2. mj_forward(shadow) to refresh derived quantities
  3. Swap policy.data → shadow (policy.infer reads from it)
  4. For each control step: infer, apply_action, compute PD torque,
     write to data_shadow.ctrl, mj_step(data_shadow)
  5. Swap policy.data back → real
  6. Real data unchanged. Diagnostic check: bitwise equality of real qpos.

BAM controller is deliberately NOT stepped during shadow (see comments).
Firmware-like PD approximation used instead:
  torque = kp * (q_target - q_pos) - kd * q_vel
"""
import time

import mujoco
import numpy as np


SHADOW_KP = 200.0
SHADOW_KD = 20.0


class SafetyFilter:
    def __init__(self, model, data, trunk_body_id, qpos_adr,
                 horizon=10, pred_tilt_max_deg=45.0, pred_height_min_m=0.080,
                 emergency_tilt_deg=45.0, decimation=4,
                 margin_deg=5.0,
                 verify_no_leak=True):
        self.model = model
        self.data = data
        self.trunk_body_id = trunk_body_id
        self.qpos_adr = qpos_adr
        self.horizon = horizon
        self.pred_tilt_max = pred_tilt_max_deg
        self.pred_height_min = pred_height_min_m
        self.emergency_tilt = emergency_tilt_deg
        self.decimation = decimation
        self.margin_deg = margin_deg
        self.verify_no_leak = verify_no_leak

        # Separate MjData for shadow rollout
        self.data_shadow = mujoco.MjData(model)
        self.data_shadow = mujoco.MjData(model)

        self.latencies_s = []
        self.leak_warnings = 0

    # ------------------------------------------------------------------
    # Helpers: tilt / height — now take explicit data argument
    # ------------------------------------------------------------------

    @staticmethod
    def _tilt_deg(model, data, trunk_body_id):
        R = data.xmat[trunk_body_id].reshape(3, 3)
        c = float(np.clip(R[2, 2], -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _tilt_real(self):
        return self._tilt_deg(self.model, self.data, self.trunk_body_id)

    def _tilt_shadow(self):
        return self._tilt_deg(self.model, self.data_shadow, self.trunk_body_id)

    def _height_real(self):
        return float(self.data.qpos[self.qpos_adr + 2])

    def _height_shadow(self):
        return float(self.data_shadow.qpos[self.qpos_adr + 2])

    # ------------------------------------------------------------------
    # Sync real → shadow
    # ------------------------------------------------------------------

    def _sync_shadow_from_real(self):
        """Copy the state of the real data into shadow, then refresh derived."""
        self.data_shadow.qpos[:] = self.data.qpos
        self.data_shadow.qvel[:] = self.data.qvel
        self.data_shadow.ctrl[:] = self.data.ctrl
        self.data_shadow.time = self.data.time
        if self.data_shadow.act.size and self.data.act.size:
            self.data_shadow.act[:] = self.data.act
        if hasattr(self.data_shadow, "qacc_warmstart"):
            self.data_shadow.qacc_warmstart[:] = self.data.qacc_warmstart
        mujoco.mj_forward(self.model, self.data_shadow)

    # ------------------------------------------------------------------
    # Policy state swap (session + data reference)
    # ------------------------------------------------------------------

    def _snapshot_policy(self, policy):
        return {
            "session": policy.ort_session,
            "policy": policy.current_policy,
            "vel_cmd": policy.vel_cmd.copy(),
            "last_action": policy.last_action.copy(),
            "input_name": policy.input_name,
            "output_name": policy.output_name,
            "data_ref": policy.data,  # for swapping to shadow
        }

    def _restore_policy(self, policy, psnap):
        policy.ort_session = psnap["session"]
        policy.current_policy = psnap["policy"]
        policy.vel_cmd = psnap["vel_cmd"]
        policy.last_action[:] = psnap["last_action"]
        policy.input_name = psnap["input_name"]
        policy.output_name = psnap["output_name"]
        policy.data = psnap["data_ref"]
        policy._update_command()

    # ------------------------------------------------------------------
    # PD torque helper
    # ------------------------------------------------------------------

    def _pd_torque_on_shadow(self, policy, q_target):
        q_pos = self.data_shadow.qpos[policy.joint_qpos_indices].astype(np.float32)
        q_vel = self.data_shadow.qvel[policy.joint_qvel_indices].astype(np.float32)
        return (SHADOW_KP * (q_target - q_pos) - SHADOW_KD * q_vel).astype(np.float32)

    # ------------------------------------------------------------------
    # Shadow rollout (uses data_shadow, never touches real data)
    # ------------------------------------------------------------------

    def _shadow_rollout(self, policy, session, policy_name, vel_cmd):
        """
        Run a shadow rollout on self.data_shadow. Real data is untouched.
        Returns (max_tilt_deg, min_height_m, early_stop_bool).
        """
        # Optionally verify no leak by capturing real qpos before / after
        if self.verify_no_leak:
            real_qpos_before = self.data.qpos.copy()

        # 1. Sync real → shadow
        self._sync_shadow_from_real()

        # 2. Snapshot policy and swap to shadow + target session
        # 2. Snapshot policy and swap to shadow + target session
        psnap = self._snapshot_policy(policy)
        # Snapshot bam_ctrl.q_target — it lives in Python, not in MjData,
        # and policy.apply_action() mutates it directly. Without restore,
        # the real simulation continues with shadow's q_target on the next
        # real step, causing trajectory divergence.
        bam_q_target_save = (policy.bam_ctrl.q_target.copy()
        if policy.bam_ctrl is not None else None)
        policy.data = self.data_shadow
        policy.ort_session = session
        policy.vel_cmd = np.array(vel_cmd, dtype=np.float32)
        policy.input_name = session.get_inputs()[0].name
        policy.output_name = session.get_outputs()[0].name
        policy._update_command()

        max_tilt = 0.0
        min_h = float("inf")
        early_stop = False

        try:
            for step in range(self.horizon):
                action = policy.infer()
                policy.apply_action(action)  # writes to bam_ctrl.q_target

                # Compute PD torque from q_target, step shadow physics
                if policy.bam_ctrl is not None:
                    q_target = policy.bam_ctrl.q_target
                    torque = self._pd_torque_on_shadow(policy, q_target)
                    for _ in range(self.decimation):
                        self.data_shadow.ctrl[:] = torque
                        mujoco.mj_step(self.model, self.data_shadow)
                else:
                    for _ in range(self.decimation):
                        mujoco.mj_step(self.model, self.data_shadow)

                tilt = self._tilt_shadow()
                h = self._height_shadow()
                max_tilt = max(max_tilt, tilt)
                min_h = min(min_h, h)

                if tilt > self.pred_tilt_max + 25.0 or h < self.pred_height_min - 0.025:
                    early_stop = True
                    break
        finally:
            # 3. Restore policy (including .data reference back to real)
            if policy.bam_ctrl is not None and bam_q_target_save is not None:
                policy.bam_ctrl.q_target[:] = bam_q_target_save
            self._restore_policy(policy, psnap)

        # 4. Verify real data untouched
        if self.verify_no_leak:
            diff = float(np.max(np.abs(self.data.qpos - real_qpos_before)))
            if diff > 1e-12:
                self.leak_warnings += 1
                if self.leak_warnings <= 3:  # avoid spam
                    print(f"  [WARN] shadow leaked into real qpos: max|Δ|={diff:.3e}")

        return max_tilt, min_h, early_stop

    # ------------------------------------------------------------------
    # Public: veto strategy
    # ------------------------------------------------------------------

    def check_switch(self, policy, target_session, target_policy_name,
                     target_vel_cmd):
        t0 = time.perf_counter()

        max_tilt, min_h, early = self._shadow_rollout(
            policy, target_session, target_policy_name, target_vel_cmd)

        is_safe = True
        reason = "ok"
        if early:
            is_safe = False
            reason = f"target early stop: tilt={max_tilt:.1f}° h={min_h*1000:.0f}mm"
        elif max_tilt > self.pred_tilt_max:
            is_safe = False
            reason = f"pred tilt {max_tilt:.1f}° > {self.pred_tilt_max:.0f}°"
        elif min_h < self.pred_height_min:
            is_safe = False
            reason = f"pred height {min_h*1000:.1f}mm < {self.pred_height_min*1000:.0f}mm"

        dt = time.perf_counter() - t0
        self.latencies_s.append(dt)
        return is_safe, reason, max_tilt, min_h, dt

    # ------------------------------------------------------------------
    # Public: comparative strategy
    # ------------------------------------------------------------------

    def check_switch_comparative(self, policy,
                                 target_session, target_policy_name,
                                 current_session, current_policy_name,
                                 target_vel_cmd, current_vel_cmd):
        t0 = time.perf_counter()

        cur_max_tilt, cur_min_h, cur_early = self._shadow_rollout(
            policy, current_session, current_policy_name, current_vel_cmd)

        tgt_max_tilt, tgt_min_h, tgt_early = self._shadow_rollout(
            policy, target_session, target_policy_name, target_vel_cmd)

        if tgt_early:
            should_switch = False
            reason = (f"target early stop: tilt={tgt_max_tilt:.1f}° "
                      f"h={tgt_min_h*1000:.0f}mm")
        elif (tgt_max_tilt <= self.pred_tilt_max
              and tgt_min_h >= self.pred_height_min):
            should_switch = True
            reason = (f"target safe: tgt={tgt_max_tilt:.1f}° "
                      f"cur={cur_max_tilt:.1f}°")
        elif cur_max_tilt > tgt_max_tilt + self.margin_deg:
            should_switch = True
            reason = (f"target better: tgt={tgt_max_tilt:.1f}° "
                      f"< cur={cur_max_tilt:.1f}° - {self.margin_deg:.0f}°")
        else:
            should_switch = False
            reason = (f"target not better: tgt={tgt_max_tilt:.1f}° "
                      f"vs cur={cur_max_tilt:.1f}°")

        dt = time.perf_counter() - t0
        self.latencies_s.append(dt)
        return should_switch, reason, cur_max_tilt, tgt_max_tilt, dt

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------

    def current_tilt_deg(self):
        return self._tilt_real()

    def check_emergency(self):
        return self._tilt_real() > self.emergency_tilt

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def latency_stats_ms(self):
        if not self.latencies_s:
            return None
        arr = np.array(self.latencies_s) * 1000.0
        return {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }