#!/usr/bin/env python3
"""D322 G0a grasp-alignment probe.

This is a state-only Isaac runtime diagnostic for the grasp pivot. It does not
close the gripper, grasp, lift, render, train PPO, or alter the existing
tap/push controller. The only new variable is grasp pose geometry: base yaw
alignment plus the asymmetric fixed-jaw/moving-jaw TCP offset.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts.roarm_kinematics import clip_joints, fk_tcp, ik_dls


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d322"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _safe_float(value: Any) -> float:
    return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)


def _compute_targets(cube_local: np.ndarray, cube_size_m: float) -> dict[str, np.ndarray | float]:
    yaw = math.atan2(float(cube_local[1]), float(cube_local[0]))
    jaw_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    tcp_to_object_center_m = cube_size_m * 0.5 - 0.008
    pre_clearance_m = 0.040

    final_tcp = cube_local.astype(np.float64).copy()
    final_tcp[:2] -= jaw_axis[:2] * tcp_to_object_center_m

    pre_tcp = cube_local.astype(np.float64).copy()
    pre_tcp[:2] -= jaw_axis[:2] * (tcp_to_object_center_m + pre_clearance_m)

    # G0a is side alignment, not top tap. The target height is object center.
    final_tcp[2] = cube_local[2]
    pre_tcp[2] = cube_local[2]
    return {
        "base_yaw_rad": yaw,
        "jaw_axis": jaw_axis,
        "pre_tcp": pre_tcp,
        "final_tcp": final_tcp,
        "tcp_to_object_center_m": tcp_to_object_center_m,
        "pre_clearance_m": pre_clearance_m,
    }


def _ik_targets_for_step(
    target_tcp_local: np.ndarray,
    current_q_rad: np.ndarray,
    *,
    max_iter: int,
    tol_mm: float,
) -> tuple[np.ndarray, bool, float]:
    q0_deg = np.degrees(current_q_rad.copy())
    q_deg, converged, err_mm, _iters = ik_dls(
        target_tcp_local,
        q0_deg,
        max_iter=max_iter,
        tol_mm=tol_mm,
        damping=2.0,
        step_clip_deg=4.0,
    )
    q_deg[5] = 0.0
    q_deg = clip_joints(q_deg)
    return np.radians(q_deg), bool(converged), float(err_mm)


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "g0a_alignment_trials.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary["trial_csv"] = _rel(csv_path)
    (out_dir / "g0a_alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# D322 G0a Alignment Probe",
        "",
        "이번 case의 신규 변수: [grasp pose geometry: base yaw alignment + asymmetric TCP offset]",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- trials: `{summary['num_trials']}`",
        f"- pass_all: `{summary['pass_all_count']}/{summary['num_trials']}`",
        f"- hard_failure: `{summary['hard_failure']}`",
        f"- output CSV: `{summary['trial_csv']}`",
        "",
        "## Criteria",
        "",
        "- TCP pose error <= 5mm and base-yaw error <= 3deg.",
        "- Fixed-jaw face gap to cube face <= 3mm and no penetration.",
        "- Cube XY displacement < 5mm.",
        "- Strict pass requires all 10 trials to satisfy all criteria.",
        "",
        "## Trial Table",
        "",
        "| trial | pose err mm | yaw err deg | face gap mm | penetration mm | cube disp mm | pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {tcp_pose_error_mm:.3f} | {base_yaw_error_deg:.3f} | "
            "{fixed_jaw_face_gap_mm:.3f} | {fixed_jaw_penetration_mm:.3f} | "
            "{cube_disp_xy_mm:.3f} | {pass_all} |".format(**row)
        )
    lines.extend(["", "## Failure Counts", ""])
    for key, value in summary["failure_counts"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "g0a_alignment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32201)
    parser.add_argument("--cube_x", type=float, default=0.30)
    parser.add_argument("--cube_y", type=float, default=0.0)
    parser.add_argument("--cube_size_m", type=float, default=0.10)
    parser.add_argument("--cube_mass_kg", type=float, default=0.72)
    parser.add_argument("--static_friction", type=float, default=1.5)
    parser.add_argument("--dynamic_friction", type=float, default=1.2)
    parser.add_argument("--approach_steps", type=int, default=180)
    parser.add_argument("--hold_steps", type=int, default=80)
    parser.add_argument("--episode_length_s", type=float, default=4.0)
    parser.add_argument("--ik_tol_mm", type=float, default=1.0)
    parser.add_argument("--ik_max_iter", type=int, default=250)
    args = parser.parse_args()

    if int(args.num_trials) != 10:
        raise ValueError("D322 G0a is pre-registered for exactly 10 trials")
    if int(args.approach_steps) <= 0 or int(args.hold_steps) <= 0:
        raise ValueError("approach_steps and hold_steps must be positive")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(args.robot_usd_path)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.roarm_stack_env import TABLE_Z

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_trials)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.cube_x_min = float(args.cube_x)
    env_cfg.cube_x_max = float(args.cube_x)
    env_cfg.cube_y_min = float(args.cube_y)
    env_cfg.cube_y_max = float(args.cube_y)
    env_cfg.cube_size_x_m = float(args.cube_size_m)
    env_cfg.cube_size_y_m = float(args.cube_size_m)
    env_cfg.cube_size_z_m = float(args.cube_size_m)
    env_cfg.sponge.spawn.size = (float(args.cube_size_m), float(args.cube_size_m), float(args.cube_size_m))
    env_cfg.sponge.spawn.mass_props.mass = float(args.cube_mass_kg)
    env_cfg.sponge.spawn.physics_material.static_friction = float(args.static_friction)
    env_cfg.sponge.spawn.physics_material.dynamic_friction = float(args.dynamic_friction)
    env_cfg.sponge.init_state.pos = (
        float(args.cube_x),
        float(args.cube_y),
        TABLE_Z + 0.5 * float(args.cube_size_m),
    )
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.rl_action_mode = "joint_delta"
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    # G0b future sim contract only: measured real gripper stalls near closed.
    # Do not activate joint lower 0.09rad or effort-limit changes in G0a.

    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)

    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    inner._compute_intermediate_values()

    env_ids = torch.arange(inner.num_envs, device=device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]
    cube_start_w = inner._sponge_pos_w.detach().clone()
    cube_start_local = (cube_start_w - origins).detach().cpu().numpy()

    target_info = [_compute_targets(cube_start_local[idx], float(args.cube_size_m)) for idx in range(inner.num_envs)]
    joint_targets_full = inner._robot.data.joint_pos.detach().clone()
    q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
    q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
    q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
    ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)

    total_steps = int(args.approach_steps) + int(args.hold_steps)
    with torch.inference_mode():
        for step in range(total_steps):
            if step < int(args.approach_steps):
                alpha = (step + 1) / float(args.approach_steps)
            else:
                alpha = 1.0
            for idx, info in enumerate(target_info):
                pre = np.asarray(info["pre_tcp"], dtype=np.float64)
                final = np.asarray(info["final_tcp"], dtype=np.float64)
                target_tcp = pre + alpha * (final - pre)
                q_next, ik_ok, _err = _ik_targets_for_step(
                    target_tcp,
                    q_targets_kin[idx],
                    max_iter=int(args.ik_max_iter),
                    tol_mm=float(args.ik_tol_mm),
                )
                if not ik_ok:
                    ik_failure_counts[idx] += 1
                q_targets_kin[idx] = q_next
            targets_t = joint_targets_full.detach().clone()
            targets_t[:, inner._bc_arm_joint_ids] = torch.tensor(
                q_targets_kin[:, :5], device=device, dtype=torch.float32
            )
            targets_t[:, inner.gripper_joint_idx] = 0.0
            joint_targets_full = targets_t.detach().clone()
            inner._external_joint_targets_override = targets_t
            env.step(zero)
            inner._compute_intermediate_values()

    inner._compute_intermediate_values()
    tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
    cube_final_w = inner._sponge_pos_w.detach().clone()
    cube_disp_xy = torch.linalg.norm(cube_final_w[:, 0:2] - cube_start_w[:, 0:2], dim=-1)
    joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy()
    actual_arm = joint_pos[:, inner._bc_arm_joint_ids]
    target_arm = q_targets_kin[:, :5]
    arm_joint_err_rad = np.max(np.abs(actual_arm - target_arm), axis=1)

    rows: list[dict[str, Any]] = []
    failure_counts = {
        "tcp_pose": 0,
        "base_yaw": 0,
        "fixed_jaw_gap": 0,
        "fixed_jaw_penetration": 0,
        "cube_displacement": 0,
    }
    for idx, info in enumerate(target_info):
        jaw_axis = np.asarray(info["jaw_axis"], dtype=np.float64)
        target_tcp = np.asarray(info["final_tcp"], dtype=np.float64)
        tcp_err_m = float(np.linalg.norm(tcp_local[idx] - target_tcp))
        base_yaw = float(info["base_yaw_rad"])
        actual_base_q = float(joint_pos[idx, inner._bc_arm_joint_ids[0]])
        base_yaw_err_deg = abs(math.degrees(_wrap_pi(actual_base_q - base_yaw)))

        cube_center = cube_start_local[idx]
        cube_near_face = cube_center - jaw_axis * (float(args.cube_size_m) * 0.5)
        fixed_jaw_face = tcp_local[idx] - jaw_axis * 0.008
        signed_gap_m = float(np.dot(cube_near_face - fixed_jaw_face, jaw_axis))
        penetration_m = max(0.0, -signed_gap_m)
        disp_m = _safe_float(cube_disp_xy[idx])

        pass_tcp = tcp_err_m <= 0.005
        pass_yaw = base_yaw_err_deg <= 3.0
        pass_gap = 0.0 <= signed_gap_m <= 0.003
        pass_pen = penetration_m <= 1.0e-6
        pass_disp = disp_m < 0.005
        if not pass_tcp:
            failure_counts["tcp_pose"] += 1
        if not pass_yaw:
            failure_counts["base_yaw"] += 1
        if not pass_gap:
            failure_counts["fixed_jaw_gap"] += 1
        if not pass_pen:
            failure_counts["fixed_jaw_penetration"] += 1
        if not pass_disp:
            failure_counts["cube_displacement"] += 1
        rows.append(
            {
                "trial": idx,
                "cube_x_m": float(cube_start_local[idx, 0]),
                "cube_y_m": float(cube_start_local[idx, 1]),
                "base_yaw_target_deg": math.degrees(base_yaw),
                "tcp_target_x_m": float(target_tcp[0]),
                "tcp_target_y_m": float(target_tcp[1]),
                "tcp_target_z_m": float(target_tcp[2]),
                "tcp_actual_x_m": float(tcp_local[idx, 0]),
                "tcp_actual_y_m": float(tcp_local[idx, 1]),
                "tcp_actual_z_m": float(tcp_local[idx, 2]),
                "tcp_pose_error_mm": tcp_err_m * 1000.0,
                "base_yaw_error_deg": base_yaw_err_deg,
                "fixed_jaw_face_gap_mm": signed_gap_m * 1000.0,
                "fixed_jaw_penetration_mm": penetration_m * 1000.0,
                "cube_disp_xy_mm": disp_m * 1000.0,
                "ik_failure_steps": int(ik_failure_counts[idx]),
                "arm_joint_target_rad": [float(v) for v in target_arm[idx].tolist()],
                "arm_joint_actual_rad": [float(v) for v in actual_arm[idx].tolist()],
                "arm_joint_err_max_rad": float(arm_joint_err_rad[idx]),
                "pass_tcp_pose": bool(pass_tcp),
                "pass_base_yaw": bool(pass_yaw),
                "pass_fixed_jaw_gap": bool(pass_gap),
                "pass_no_penetration": bool(pass_pen),
                "pass_cube_displacement": bool(pass_disp),
                "pass_all": bool(pass_tcp and pass_yaw and pass_gap and pass_pen and pass_disp),
            }
        )

    pass_all_count = sum(1 for row in rows if row["pass_all"])
    hard_failure = any(count >= 3 for count in failure_counts.values())
    if pass_all_count == int(args.num_trials):
        verdict = "D322_G0A_ALIGNMENT_PASS"
    elif hard_failure:
        verdict = "D322_G0A_ALIGNMENT_FAIL"
    else:
        verdict = "D322_G0A_ALIGNMENT_PARTIAL"

    summary = {
        "artifact": "d322_g0a_grasp_alignment_probe",
        "verdict": verdict,
        "num_trials": int(args.num_trials),
        "pass_all_count": int(pass_all_count),
        "hard_failure": bool(hard_failure),
        "failure_rule": "hard fail if any alignment criterion misses in >=3/10 trials; strict pass requires 10/10 all-criteria pass",
        "failure_counts": failure_counts,
        "new_variable": "grasp pose geometry: base yaw alignment + asymmetric TCP offset",
        "non_goals": [
            "no gripper close",
            "no grasp",
            "no lift",
            "no render",
            "no RL/PPO",
            "no friction/material randomization",
            "no position randomization",
        ],
        "cube_size_m": float(args.cube_size_m),
        "cube_mass_kg": float(args.cube_mass_kg),
        "static_friction": float(args.static_friction),
        "dynamic_friction": float(args.dynamic_friction),
        "tcp_to_object_center_m": float(target_info[0]["tcp_to_object_center_m"]),
        "pre_clearance_m": float(target_info[0]["pre_clearance_m"]),
        "approach_steps": int(args.approach_steps),
        "hold_steps": int(args.hold_steps),
        "robot_usd_path": _rel(args.robot_usd_path),
        "out_dir": _rel(args.out_dir),
        "rows": rows,
    }
    _write_outputs(args.out_dir, rows, summary)
    print(
        "[d322-g0a] "
        f"verdict={verdict} pass_all={pass_all_count}/{args.num_trials} "
        f"hard_failure={hard_failure} out_dir={_rel(args.out_dir)}"
    )
    env.close()
    sim_app.close()
    return 0 if not hard_failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
