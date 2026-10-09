"""
Naive switching experiment v2:
- Randomized reset (trunk pose, velocity, joint pose)
- Fine-grained phase sweep (every 10 control steps)
- Multiple transition pairs
- BAM always on
"""
import argparse, json, math, sys, time
from pathlib import Path
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import (
    PolicyInference, load_bam_model, load_mujoco_with_bam,
    MICRODUCK_XML, BAM_VIN_MIN,
)

DECIMATION = 4
EPISODE_STEPS = 400
FALL_HEIGHT_M = 0.06
FALL_TILT_DEG = 60.0
SWITCH_THRESHOLD = 0.05

# Randomization bounds on reset
RESET_HEIGHT_RANGE = (0.120, 0.130)      # trunk z at reset (m)
RESET_VEL_XY_RANGE = (-0.05, 0.05)       # trunk vx,vy at reset (m/s)
RESET_JOINT_NOISE = 0.02                 # rad, added to DEFAULT_POSE
RESET_YAW_RANGE = (-math.pi/6, math.pi/6)  # trunk yaw at reset (rad)

def random_quat_from_yaw(yaw):
    return np.array([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)], dtype=np.float64)

def reset_scene(model, data, policy, rng):
    mujoco.mj_resetData(model, data)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
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
    """Force current policy to a named session, bypassing auto-switch."""
    if name == "walking":
        policy.current_policy = "walking"
        policy.ort_session = policy.walking_session
    elif name == "standing":
        policy.current_policy = "standing"
        policy.ort_session = policy.standing_session
    elif name == "sitstand":
        policy.current_policy = "sit"
        policy.ort_session = policy.sit_session
        policy.sit_mode = False   # flag=0 (stand) initially
    elif name == "roulade":
        # force roulade session with zero command
        policy.behavior_mode = "roulade"
        policy.behavior_time_left = 1e6   # disable auto-return
        policy.current_policy = "roulade"
        policy.ort_session = policy.behavior_sessions["roulade"]
    else:
        raise ValueError(name)
    policy._update_command()

def run_episode(model, data, policy, trunk_body_id, switch_step, walk_vel,
                from_policy, to_policy, rng):
    qpos_adr = reset_scene(model, data, policy, rng)

    # --- warmup: 30 steps with 'from' policy and walk_vel ---
    force_policy(policy, from_policy)
    if from_policy == "walking":
        policy.vel_cmd = np.array([walk_vel, 0.0, 0.0], dtype=np.float32)
    else:
        policy.vel_cmd = np.zeros(3, dtype=np.float32)
    policy._update_command()

    rec = {
        "switch_step": int(switch_step),
        "from": from_policy, "to": to_policy,
        "fell": False, "fell_at_step": None,
        "steps_after_switch": None,
        "max_torque_nm": 0.0,
        "min_height_m": float("inf"),
        "max_tilt_deg": 0.0,
        "switched": False,
        "pre_switch_tilt_max_deg": 0.0,
    }

    for step in range(EPISODE_STEPS):
        if step == switch_step and not rec["switched"]:
            force_policy(policy, to_policy)
            policy.vel_cmd = np.zeros(3, dtype=np.float32)
            policy._update_command()
            rec["switched"] = True

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
            rec["pre_switch_tilt_max_deg"] = max(rec["pre_switch_tilt_max_deg"], tilt)

        if h < FALL_HEIGHT_M or tilt > FALL_TILT_DEG:
            rec["fell"] = True
            rec["fell_at_step"] = step
            if rec["switched"]:
                rec["steps_after_switch"] = step - switch_step
            break

    if not rec["fell"] and rec["switched"]:
        rec["steps_after_switch"] = EPISODE_STEPS - 1 - switch_step
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
                    default=list(range(20, 200, 10)))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="naive_switch_v2.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    P = Path(args.policies_dir)

    # --- model: always BAM ---
    bam_model = load_bam_model(kp_fw=200.0, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, None, BAM_VIN_MIN)

    # --- policy: load ALL sessions we might need ---
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

    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    results = []
    t0 = time.time()
    for sw in args.switch_steps:
        falls = 0
        tilt_after = []
        for trial in range(args.trials):
            r = run_episode(model, data, policy, trunk_body_id, sw,
                            args.walk_vel, args.from_policy, args.to_policy, rng)
            r["trial"] = trial
            results.append(r)
            falls += int(r["fell"])
        n = args.trials
        print(f"switch@{sw:3d}  fall={falls}/{n} ({falls/n:.0%})  "
              f"({time.time()-t0:.0f}s)")

    summary = {}
    for sw in args.switch_steps:
        rs = [r for r in results if r["switch_step"] == sw]
        sa = [r["steps_after_switch"] for r in rs if r["steps_after_switch"] is not None]
        pre_tilt = [r["pre_switch_tilt_max_deg"] for r in rs]
        summary[str(sw)] = {
            "n": len(rs),
            "fall_rate": sum(r["fell"] for r in rs) / len(rs),
            "mean_steps_after_switch": float(np.mean(sa)) if sa else None,
            "mean_min_height_mm": float(np.mean([r["min_height_m"] for r in rs]))*1000,
            "mean_max_tilt_deg": float(np.mean([r["max_tilt_deg"] for r in rs])),
            "mean_max_torque_nm": float(np.mean([r["max_torque_nm"] for r in rs])),
            "mean_pre_switch_tilt_deg": float(np.mean(pre_tilt)),
        }

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "episodes": results},
                  f, indent=2, default=str)
    print(f"\nSaved {args.out}")

    print(f"\n=== {args.from_policy} -> {args.to_policy} @ {args.walk_vel} m/s ===")
    for sw, s in summary.items():
        print(f"switch@{sw:>3s}:  fall={s['fall_rate']:.0%}  "
              f"steps_after={s['mean_steps_after_switch']}  "
              f"tilt={s['mean_max_tilt_deg']:5.1f}°  "
              f"tau={s['mean_max_torque_nm']:.2f}Nm")

if __name__ == "__main__":
    main()