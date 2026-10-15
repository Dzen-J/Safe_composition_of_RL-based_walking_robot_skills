# scripts/naive_switch_experiment_v3.py
"""
Naive vs filtered policy switching.

Filter modes (--filter-mode):
  veto         — shadow target only, absolute bounds.
  comparative  — shadow current + target, switch if target is safe or
                 clearly better (recommended).

Records per-episode:
- filter_checks, filter_vetoes, emergency_forced
- switch_delay (actual_step - requested_step)
- filter_latencies_ms
- veto_reasons (list of (step, reason))
- comparative_cur_tilt_at_switch, comparative_tgt_tilt_at_switch
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import (
    PolicyInference, load_bam_model, load_mujoco_with_bam,
    MICRODUCK_XML, BAM_VIN_MIN,
)
from safety_filter import SafetyFilter


DECIMATION = 4
EPISODE_STEPS = 400
FALL_HEIGHT_M = 0.060
FALL_TILT_DEG = 75.0
SWITCH_THRESHOLD = 0.05

RESET_HEIGHT_RANGE = (0.120, 0.130)
RESET_VEL_XY_RANGE = (-0.05, 0.05)
RESET_JOINT_NOISE = 0.02
RESET_YAW_RANGE = (-math.pi/6, math.pi/6)


def random_quat_from_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                    dtype=np.float64)


def reset_scene(model, data, policy, rng):
    mujoco.mj_resetData(model, data)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                           "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[fj])
    qvel_adr = int(model.jnt_dofadr[fj])

    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = rng.uniform(*RESET_HEIGHT_RANGE)
    yaw = rng.uniform(*RESET_YAW_RANGE)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = random_quat_from_yaw(yaw)

    noise = rng.normal(0.0, RESET_JOINT_NOISE, size=policy.n_joints)
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i] + noise[i]

    data.qvel[qvel_adr + 0] = rng.uniform(*RESET_VEL_XY_RANGE)
    data.qvel[qvel_adr + 1] = rng.uniform(*RESET_VEL_XY_RANGE)

    if policy.bam_ctrl is not None:
        policy.bam_ctrl.reset(data.qpos)
    policy.last_action[:] = 0.0
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return qpos_adr


def trunk_tilt_deg(data, trunk_body_id):
    R = data.xmat[trunk_body_id].reshape(3, 3)
    c = float(np.clip(R[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(c))


def force_policy(policy, name):
    if name == "walking":
        policy.current_policy = "walking"
        policy.ort_session = policy.walking_session
    elif name == "standing":
        policy.current_policy = "standing"
        policy.ort_session = policy.standing_session
    elif name == "sitstand":
        policy.current_policy = "sit"
        policy.ort_session = policy.sit_session
        policy.sit_mode = False
    elif name == "roulade":
        policy.behavior_mode = "roulade"
        policy.behavior_time_left = 1e6
        policy.current_policy = "roulade"
        policy.ort_session = policy.behavior_sessions["roulade"]
    else:
        raise ValueError(name)
    policy._update_command()


def get_target_session(policy, name):
    if name == "standing":
        return policy.standing_session
    if name == "sitstand":
        return policy.sit_session
    if name == "roulade":
        return policy.behavior_sessions["roulade"]
    raise ValueError(name)


def do_switch(policy, to_policy_name):
    force_policy(policy, to_policy_name)
    policy.vel_cmd = np.zeros(3, dtype=np.float32)
    policy._update_command()


def run_episode(model, data, policy, trunk_body_id, switch_step, walk_vel,
                from_policy, to_policy, rng,
                safety_filter=None,
                target_session=None, target_policy_name=None,
                current_session=None, current_policy_name=None,
                filter_mode="comparative"):
    qpos_adr = reset_scene(model, data, policy, rng)

    force_policy(policy, from_policy)
    if from_policy == "walking":
        policy.vel_cmd = np.array([walk_vel, 0.0, 0.0], dtype=np.float32)
    else:
        policy.vel_cmd = np.zeros(3, dtype=np.float32)
    policy._update_command()

    rec = {
        "switch_step": int(switch_step),
        "from": from_policy,
        "to": to_policy,
        "fell": False,
        "fell_at_step": None,
        "steps_after_switch": None,
        "max_torque_nm": 0.0,
        "min_height_m": float("inf"),
        "max_tilt_deg": 0.0,
        "switched": False,
        "switch_actual_step": None,
        "switch_delay": None,
        "filter_checks": 0,
        "filter_vetoes": 0,
        "emergency_forced": False,
        "pre_switch_tilt_max_deg": 0.0,
        "filter_latencies_ms": [],
        "veto_reasons": [],
        "comparative_cur_tilt_at_switch": None,
        "comparative_tgt_tilt_at_switch": None,
    }

    for step in range(EPISODE_STEPS):
        if step >= switch_step and not rec["switched"]:
            if safety_filter is None:
                do_switch(policy, to_policy)
                rec["switched"] = True
                rec["switch_actual_step"] = step
                rec["switch_delay"] = step - switch_step
            else:
                rec["filter_checks"] += 1

                if filter_mode == "veto":
                    is_safe, reason, pred_t, pred_h, dt = safety_filter.check_switch(
                        policy, target_session, target_policy_name,
                        target_vel_cmd=[0.0, 0.0, 0.0],
                    )
                    rec["filter_latencies_ms"].append(dt * 1000.0)

                    if is_safe:
                        do_switch(policy, to_policy)
                        rec["switched"] = True
                        rec["switch_actual_step"] = step
                        rec["switch_delay"] = step - switch_step
                    elif safety_filter.check_emergency():
                        do_switch(policy, to_policy)
                        rec["switched"] = True
                        rec["emergency_forced"] = True
                        rec["switch_actual_step"] = step
                        rec["switch_delay"] = step - switch_step
                    else:
                        rec["filter_vetoes"] += 1
                        rec["veto_reasons"].append((step, reason))

                else:  # comparative
                    should_switch, reason, cur_t, tgt_t, dt = (
                        safety_filter.check_switch_comparative(
                            policy,
                            target_session=target_session,
                            target_policy_name=target_policy_name,
                            current_session=current_session,
                            current_policy_name=current_policy_name,
                            target_vel_cmd=[0.0, 0.0, 0.0],
                            current_vel_cmd=[walk_vel, 0.0, 0.0],
                        )
                    )
                    rec["filter_latencies_ms"].append(dt * 1000.0)

                    if should_switch:
                        do_switch(policy, to_policy)
                        rec["switched"] = True
                        rec["switch_actual_step"] = step
                        rec["switch_delay"] = step - switch_step
                        rec["comparative_cur_tilt_at_switch"] = cur_t
                        rec["comparative_tgt_tilt_at_switch"] = tgt_t
                    elif safety_filter.check_emergency():
                        do_switch(policy, to_policy)
                        rec["switched"] = True
                        rec["emergency_forced"] = True
                        rec["switch_actual_step"] = step
                        rec["switch_delay"] = step - switch_step
                        rec["comparative_cur_tilt_at_switch"] = cur_t
                        rec["comparative_tgt_tilt_at_switch"] = tgt_t
                    else:
                        rec["filter_vetoes"] += 1
                        rec["veto_reasons"].append((step, reason))

        action = policy.infer()
        policy.apply_action(action)
        for _ in range(DECIMATION):
            if policy.bam_ctrl is not None:
                policy.bam_ctrl.update()
            mujoco.mj_step(model, data)

        h = float(data.qpos[qpos_adr + 2])
        tilt = trunk_tilt_deg(data, trunk_body_id)
        tau = float(np.max(np.abs(data.actuator_force))) if data.actuator_force.size else 0.0

        rec["min_height_m"] = min(rec["min_height_m"], h)
        rec["max_tilt_deg"] = max(rec["max_tilt_deg"], tilt)
        rec["max_torque_nm"] = max(rec["max_torque_nm"], tau)
        if not rec["switched"]:
            rec["pre_switch_tilt_max_deg"] = max(
                rec["pre_switch_tilt_max_deg"], tilt)

        if h < FALL_HEIGHT_M or tilt > FALL_TILT_DEG:
            rec["fell"] = True
            rec["fell_at_step"] = step
            if rec["switched"]:
                rec["steps_after_switch"] = step - rec["switch_actual_step"]
            break

    if not rec["fell"] and rec["switched"]:
        rec["steps_after_switch"] = (
            EPISODE_STEPS - 1 - rec["switch_actual_step"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies-dir", required=True)
    ap.add_argument("--from", dest="from_policy", default="walking",
                    choices=["walking"])
    ap.add_argument("--to", dest="to_policy", default="standing",
                    choices=["standing", "sitstand", "roulade"])
    ap.add_argument("--walk-vel", type=float, default=0.25)
    ap.add_argument("--switch-steps", type=int, nargs="+",
                    default=[20, 30, 40, 50, 60, 70, 80, 90])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--use-filter", action="store_true")
    ap.add_argument("--filter-mode", choices=["veto", "comparative"],
                    default="comparative")
    ap.add_argument("--filter-horizon", type=int, default=10)
    ap.add_argument("--filter-tilt-max", type=float, default=45.0)
    ap.add_argument("--filter-height-min", type=float, default=0.080)
    ap.add_argument("--filter-emergency-tilt", type=float, default=45.0)
    ap.add_argument("--filter-margin-deg", type=float, default=5.0)

    ap.add_argument("--out", default="naive_switch_v3.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    P = Path(args.policies_dir)

    bam_model = load_bam_model(kp_fw=200.0, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, None, BAM_VIN_MIN)

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=str(P / "alpha_walking.onnx"),
        standing_onnx_path=str(P / "alpha_stand.onnx"),
        sitstand_onnx_path=str(P / "alpha_sitstand.onnx"),
        roulade_onnx_path=str(P / "roulade.onnx"),
        roulade_duration=2.0,
        switch_threshold=SWITCH_THRESHOLD,
        new_cmd_obs=True,
    )
    assert policy.get_observations().size == 61

    trunk_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    freejoint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[freejoint_id])

    target_session = get_target_session(policy, args.to_policy)
    target_policy_name = ("standing" if args.to_policy == "standing"
                          else ("sit" if args.to_policy == "sitstand"
                                else args.to_policy))
    current_session = policy.walking_session
    current_policy_name = "walking"

    safety_filter = None
    if args.use_filter:
        safety_filter = SafetyFilter(
            model, data, trunk_body_id, qpos_adr,
            horizon=args.filter_horizon,
            pred_tilt_max_deg=args.filter_tilt_max,
            pred_height_min_m=args.filter_height_min,
            emergency_tilt_deg=args.filter_emergency_tilt,
            decimation=DECIMATION,
            margin_deg=args.filter_margin_deg,
        )
        print(f"[filter] ENABLED mode={args.filter_mode}: "
              f"horizon={args.filter_horizon} "
              f"tilt_max={args.filter_tilt_max}° "
              f"h_min={args.filter_height_min*1000:.0f}mm "
              f"emergency_tilt={args.filter_emergency_tilt}° "
              f"margin={args.filter_margin_deg}°")
    else:
        print("[filter] DISABLED (naive switching)")

    results = []
    t0 = time.time()
    for sw in args.switch_steps:
        falls = 0
        veto_total = 0
        emergency_total = 0
        for trial in range(args.trials):
            r = run_episode(
                model, data, policy, trunk_body_id, sw,
                args.walk_vel, args.from_policy, args.to_policy, rng,
                safety_filter=safety_filter,
                target_session=target_session,
                target_policy_name=target_policy_name,
                current_session=current_session,
                current_policy_name=current_policy_name,
                filter_mode=args.filter_mode,
            )
            r["trial"] = trial
            results.append(r)
            falls += int(r["fell"])
            veto_total += r["filter_vetoes"]
            emergency_total += int(r["emergency_forced"])
        n = args.trials
        print(f"switch@{sw:3d}  fall={falls}/{n} ({falls/n:.0%})  "
              f"vetoes={veto_total}  emerg={emergency_total}  "
              f"({time.time()-t0:.0f}s)")

    summary = {}
    for sw in args.switch_steps:
        rs = [r for r in results if r["switch_step"] == sw]
        sa = [r["steps_after_switch"] for r in rs
              if r["steps_after_switch"] is not None]
        delays = [r["switch_delay"] for r in rs
                  if r["switch_delay"] is not None]
        all_lat = [x for r in rs for x in r["filter_latencies_ms"]]

        entry = {
            "n": len(rs),
            "fall_rate": sum(r["fell"] for r in rs) / len(rs),
            "mean_steps_after_switch": float(np.mean(sa)) if sa else None,
            "mean_min_height_mm": float(np.mean(
                [r["min_height_m"] for r in rs])) * 1000,
            "mean_max_tilt_deg": float(np.mean(
                [r["max_tilt_deg"] for r in rs])),
            "mean_max_torque_nm": float(np.mean(
                [r["max_torque_nm"] for r in rs])),
            "mean_pre_switch_tilt_deg": float(np.mean(
                [r["pre_switch_tilt_max_deg"] for r in rs])),
            "mean_filter_checks": float(np.mean(
                [r["filter_checks"] for r in rs])),
            "mean_filter_vetoes": float(np.mean(
                [r["filter_vetoes"] for r in rs])),
            "emergency_forced_count": sum(
                int(r["emergency_forced"]) for r in rs),
            "mean_switch_delay_steps": float(np.mean(delays)) if delays else None,
        }
        if all_lat:
            arr = np.array(all_lat)
            entry["filter_latency_p50_ms"] = float(np.percentile(arr, 50))
            entry["filter_latency_p95_ms"] = float(np.percentile(arr, 95))
            entry["filter_latency_max_ms"] = float(arr.max())
        summary[str(sw)] = entry

    out_obj = {
        "config": vars(args),
        "filter_enabled": bool(args.use_filter),
        "filter_mode": args.filter_mode if args.use_filter else None,
        "summary": summary,
        "episodes": results,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"\nSaved {args.out}")

    mode_tag = (f"filtered/{args.filter_mode}" if args.use_filter else "naive")
    print(f"\n=== {args.from_policy} -> {args.to_policy} @ {args.walk_vel} m/s "
          f"({mode_tag}) ===")
    for sw, s in summary.items():
        delay = s["mean_switch_delay_steps"]
        delay_s = f"{delay:.0f}" if delay is not None else "n/a"
        print(f"switch@{sw:>3s}:  fall={s['fall_rate']:.0%}  "
              f"vetoes={s['mean_filter_vetoes']:.1f}  "
              f"delay={delay_s}  "
              f"emerg={s['emergency_forced_count']}  "
              f"tilt={s['mean_max_tilt_deg']:5.1f}°")


if __name__ == "__main__":
    main()