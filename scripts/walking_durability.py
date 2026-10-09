# scripts/walking_durability.py
"""Baseline: how long does walking policy survive without any switch?"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import (
    PolicyInference, load_bam_model, load_mujoco_with_bam,
    MICRODUCK_XML, BAM_VIN_MIN,
)

DECIMATION = 4
EPISODE_STEPS = 1500          # 30 s max
FALL_TILT_DEG = 75.0
FALL_HEIGHT_M = 0.060
RESET_YAW_RANGE = (-math.pi/6, math.pi/6)
RESET_VEL_RANGE = (-0.05, 0.05)
RESET_HEIGHT_RANGE = (0.120, 0.130)
RESET_JOINT_NOISE = 0.02

def trunk_tilt_deg(data, trunk_body_id):
    R = data.xmat[trunk_body_id].reshape(3, 3)
    c = float(np.clip(R[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(c))

def reset_scene(model, data, policy, rng):
    mujoco.mj_resetData(model, data)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[fj])
    qvel_adr = int(model.jnt_dofadr[fj])
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = rng.uniform(*RESET_HEIGHT_RANGE)
    yaw = rng.uniform(*RESET_YAW_RANGE)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    noise = rng.normal(0, RESET_JOINT_NOISE, size=policy.n_joints)
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i] + noise[i]
    data.qvel[qvel_adr + 0] = rng.uniform(*RESET_VEL_RANGE)
    data.qvel[qvel_adr + 1] = rng.uniform(*RESET_VEL_RANGE)
    if policy.bam_ctrl is not None:
        policy.bam_ctrl.reset(data.qpos)
    policy.last_action[:] = 0.0
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return qpos_adr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walking", required=True)
    ap.add_argument("--walk-vel", type=float, default=0.25)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--fall-tilt", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="walking_durability.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bam_model = load_bam_model(kp_fw=200.0, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, None, BAM_VIN_MIN)

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=args.walking,
        new_cmd_obs=True,
    )
    policy.current_policy = "walking"
    policy.ort_session = policy.walking_session
    policy.vel_cmd = np.array([args.walk_vel, 0, 0], dtype=np.float32)
    policy._update_command()

    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    results = []
    for trial in range(args.trials):
        qpos_adr = reset_scene(model, data, policy, rng)
        max_tilt = 0.0
        min_height = float("inf")
        fell_at = None
        for step in range(EPISODE_STEPS):
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(DECIMATION):
                if policy.bam_ctrl is not None:
                    policy.bam_ctrl.update()
                mujoco.mj_step(model, data)
            h = float(data.qpos[qpos_adr + 2])
            tilt = trunk_tilt_deg(data, trunk_body_id)
            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, h)
            if tilt > args.fall_tilt or h < FALL_HEIGHT_M:
                fell_at = step
                break
        results.append({
            "trial": trial,
            "fell_at_step": fell_at,
            "duration_s": (fell_at if fell_at is not None else EPISODE_STEPS) * 0.02,
            "survived_full": fell_at is None,
            "max_tilt_deg": max_tilt,
            "min_height_m": min_height,
        })
        print(f"trial {trial:2d}: "
              f"{'FELL at step ' + str(fell_at) if fell_at is not None else 'survived 30s'}  "
              f"(tilt_max={max_tilt:.1f}°, h_min={min_height*1000:.1f}mm)")

    survived = sum(1 for r in results if r["survived_full"])
    fell_steps = [r["fell_at_step"] for r in results if r["fell_at_step"] is not None]
    summary = {
        "walk_vel": args.walk_vel,
        "fall_tilt_threshold_deg": args.fall_tilt,
        "n_trials": args.trials,
        "survived_full_30s": survived,
        "survival_rate": survived / args.trials,
        "median_time_to_fall_s": float(np.median(fell_steps)) * 0.02 if fell_steps else None,
        "min_time_to_fall_s": float(np.min(fell_steps)) * 0.02 if fell_steps else None,
        "p10_time_to_fall_s": float(np.percentile(fell_steps, 10)) * 0.02 if fell_steps else None,
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)

if __name__ == "__main__":
    main()