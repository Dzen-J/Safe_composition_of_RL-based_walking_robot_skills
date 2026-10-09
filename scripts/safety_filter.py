# scripts/safety_filter.py
"""
Predictive safety filter for policy switching.
Runs a short shadow rollout of the TARGET policy from the current state
and vetoes the switch if the predicted trajectory violates safety bounds.
"""
import copy
import numpy as np
import mujoco


class PredictiveSafetyFilter:
    def __init__(self, model, data, trunk_body_id, qpos_adr,
                 horizon=10, tilt_max_deg=45.0, height_min_m=0.080):
        """
        horizon: number of control steps to simulate ahead (10 * 20ms = 200ms)
        tilt_max_deg: max allowed tilt in predicted trajectory
        height_min_m: min allowed trunk height in predicted trajectory
        """
        self.model = model
        self.data = data
        self.trunk_body_id = trunk_body_id
        self.qpos_adr = qpos_adr
        self.horizon = horizon
        self.tilt_max = tilt_max_deg
        self.height_min = height_min_m

    def _tilt_deg(self, data):
        R = data.xmat[self.trunk_body_id].reshape(3, 3)
        c = float(np.clip(R[2, 2], -1.0, 1.0))
        return np.degrees(np.arccos(c))

    def check_switch(self, policy, target_session, target_policy_name,
                     target_vel_cmd, decimation=4):
        """
        Simulate the target policy on a COPY of the current state.
        Returns (is_safe: bool, reason: str, max_tilt, min_height).
        """
        # --- Save current state ---
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        act_save = self.data.act.copy() if self.data.act.size else None
        ctrl_save = self.data.ctrl.copy()
        time_save = self.data.time

        # --- Save policy state ---
        policy_save = {
            "session": policy.ort_session,
            "policy": policy.current_policy,
            "vel_cmd": policy.vel_cmd.copy(),
            "last_action": policy.last_action.copy(),
            "input_name": policy.input_name,
            "output_name": policy.output_name,
        }
        if policy.bam_ctrl is not None:
            bam_q_target_save = policy.bam_ctrl.q_target.copy()

        # --- Configure target policy ---
        policy.ort_session = target_session
        policy.current_policy = target_policy_name
        policy.vel_cmd = np.array(target_vel_cmd, dtype=np.float32)
        policy.input_name = target_session.get_inputs()[0].name
        policy.output_name = target_session.get_outputs()[0].name
        policy._update_command()

        # --- Shadow rollout ---
        max_tilt = 0.0
        min_height = float("inf")
        is_safe = True
        reason = "ok"

        try:
            for step in range(self.horizon):
                action = policy.infer()
                policy.apply_action(action)
                for _ in range(decimation):
                    if policy.bam_ctrl is not None:
                        policy.bam_ctrl.update()
                    mujoco.mj_step(self.model, self.data)

                tilt = self._tilt_deg(self.data)
                h = float(self.data.qpos[self.qpos_adr + 2])
                max_tilt = max(max_tilt, tilt)
                min_height = min(min_height, h)

                if tilt > self.tilt_max:
                    is_safe = False
                    reason = f"tilt {tilt:.1f}° > {self.tilt_max}° at shadow step {step}"
                    break
                if h < self.height_min:
                    is_safe = False
                    reason = f"height {h*1000:.1f}mm < {self.height_min*1000:.0f}mm at shadow step {step}"
                    break
        finally:
            # --- Restore state ---
            self.data.qpos[:] = qpos_save
            self.data.qvel[:] = qvel_save
            if act_save is not None:
                self.data.act[:] = act_save
            self.data.ctrl[:] = ctrl_save
            self.data.time = time_save
            mujoco.mj_forward(self.model, self.data)

            # --- Restore policy ---
            policy.ort_session = policy_save["session"]
            policy.current_policy = policy_save["policy"]
            policy.vel_cmd = policy_save["vel_cmd"]
            policy.last_action[:] = policy_save["last_action"]
            policy.input_name = policy_save["input_name"]
            policy.output_name = policy_save["output_name"]
            if policy.bam_ctrl is not None:
                policy.bam_ctrl.q_target[:] = bam_q_target_save
            policy._update_command()

        return is_safe, reason, max_tilt, min_height