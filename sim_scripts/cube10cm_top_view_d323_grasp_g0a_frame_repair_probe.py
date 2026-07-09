#!/usr/bin/env python3
"""D323 G0a frame repair probe.

This is a repair-only diagnostic for the grasp pivot active case.  It audits the
runtime link5/TCP/gripper frames, checks whether the requested side-grasp pose
family is reachable by the 5-DOF arm, and only runs the original 10-trial G0a
alignment verdict if the pose family is feasible.  It does not close the
gripper, grasp, lift, render trajectories, train RL/PPO, or advance the
variable ladder.
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

from sim_scripts.roarm_kinematics import _CHAIN, Tmat, Trot_z, clip_joints


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d323"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)

TCP_LOCAL_OFFSET_M = np.array([0.0, 0.0, 0.115428], dtype=np.float64)
FIXED_JAW_FACE_LOCAL_M = np.array([-0.008, 0.0, 0.0], dtype=np.float64)
HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float64)
AUDIT_POSES_DEG = (
    ("home", HOME_DEG),
    ("audit_pose_a", np.array([0.0, 20.0, 120.0, 15.0, 0.0, 0.0], dtype=np.float64)),
    ("audit_pose_b", np.array([25.0, 25.0, 110.0, 20.0, -30.0, 0.0], dtype=np.float64)),
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _safe_float(value: Any) -> float:
    return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)


def _unit_xy(x: float, y: float) -> np.ndarray:
    out = np.array([x, y, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(out[:2]))
    if norm <= 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out / norm


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = np.asarray(a, dtype=np.float64)
    bn = np.asarray(b, dtype=np.float64)
    an = an / max(float(np.linalg.norm(an)), 1.0e-12)
    bn = bn / max(float(np.linalg.norm(bn)), 1.0e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(an, bn), -1.0, 1.0))))


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fk_link5_runtime(joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.radians(np.asarray(joints_deg, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    for name, xyz, rpy, joint_idx in _CHAIN:
        if name == "link5_to_tcp":
            break
        transform = transform @ Tmat(xyz, rpy)
        if joint_idx is not None:
            transform = transform @ Trot_z(float(q[joint_idx]))
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def _fk_runtime_tcp(joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    link5_pos, link5_rot = _fk_link5_runtime(joints_deg)
    tcp_pos = link5_pos + link5_rot @ TCP_LOCAL_OFFSET_M
    return tcp_pos, link5_pos, link5_rot


def _runtime_residual(
    q_arm_deg: np.ndarray,
    target_tcp: np.ndarray,
    target_x_axis: np.ndarray | None = None,
    target_z_axis: np.ndarray | None = None,
    *,
    orientation_weight_m: float = 0.05,
) -> np.ndarray:
    q_full = np.zeros(6, dtype=np.float64)
    q_full[:5] = np.asarray(q_arm_deg, dtype=np.float64)
    tcp_pos, _link5_pos, link5_rot = _fk_runtime_tcp(q_full)
    residual = [target_tcp - tcp_pos]
    if target_x_axis is not None and target_z_axis is not None:
        current_x = link5_rot[:, 0]
        current_z = link5_rot[:, 2]
        residual.append(orientation_weight_m * np.cross(current_x, target_x_axis))
        residual.append(orientation_weight_m * np.cross(current_z, target_z_axis))
    return np.concatenate(residual)


def _numeric_jacobian(
    q_arm_deg: np.ndarray,
    target_tcp: np.ndarray,
    target_x_axis: np.ndarray | None,
    target_z_axis: np.ndarray | None,
    *,
    orientation_weight_m: float,
    eps_deg: float = 0.01,
) -> np.ndarray:
    base = np.asarray(q_arm_deg, dtype=np.float64)
    r0 = _runtime_residual(
        base,
        target_tcp,
        target_x_axis,
        target_z_axis,
        orientation_weight_m=orientation_weight_m,
    )
    jac = np.zeros((r0.shape[0], 5), dtype=np.float64)
    for idx in range(5):
        qp = base.copy()
        qm = base.copy()
        qp[idx] += eps_deg
        qm[idx] -= eps_deg
        rp = _runtime_residual(
            qp,
            target_tcp,
            target_x_axis,
            target_z_axis,
            orientation_weight_m=orientation_weight_m,
        )
        rm = _runtime_residual(
            qm,
            target_tcp,
            target_x_axis,
            target_z_axis,
            orientation_weight_m=orientation_weight_m,
        )
        jac[:, idx] = (rp - rm) / (2.0 * eps_deg)
    return jac


def _solve_runtime_ik(
    target_tcp: np.ndarray,
    q0_deg: np.ndarray,
    *,
    target_x_axis: np.ndarray | None = None,
    target_z_axis: np.ndarray | None = None,
    max_iter: int = 600,
    pos_tol_mm: float = 1.0,
    axis_tol_deg: float = 3.0,
    damping: float = 0.002,
    step_clip_deg: float = 4.0,
    orientation_weight_m: float = 0.05,
) -> dict[str, Any]:
    q = np.asarray(q0_deg, dtype=np.float64).copy()
    if q.shape[0] == 6:
        q_arm = q[:5].copy()
    else:
        q_arm = q.copy()

    for it in range(max_iter + 1):
        residual = _runtime_residual(
            q_arm,
            target_tcp,
            target_x_axis,
            target_z_axis,
            orientation_weight_m=orientation_weight_m,
        )
        q_full = np.zeros(6, dtype=np.float64)
        q_full[:5] = q_arm
        tcp_pos, link5_pos, link5_rot = _fk_runtime_tcp(q_full)
        pos_err_mm = float(np.linalg.norm(target_tcp - tcp_pos) * 1000.0)
        x_err_deg = 0.0
        z_err_deg = 0.0
        if target_x_axis is not None and target_z_axis is not None:
            x_err_deg = _axis_angle_deg(link5_rot[:, 0], target_x_axis)
            z_err_deg = _axis_angle_deg(link5_rot[:, 2], target_z_axis)
        converged = pos_err_mm <= pos_tol_mm and x_err_deg <= axis_tol_deg and z_err_deg <= axis_tol_deg
        if converged or it >= max_iter:
            return {
                "q_deg": [float(v) for v in np.r_[q_arm, 0.0].tolist()],
                "converged": bool(converged),
                "iterations": int(it),
                "pos_err_mm": pos_err_mm,
                "x_axis_err_deg": float(x_err_deg),
                "z_axis_err_deg": float(z_err_deg),
                "tcp_local_m": [float(v) for v in tcp_pos.tolist()],
                "link5_local_m": [float(v) for v in link5_pos.tolist()],
                "link5_x_axis_world": [float(v) for v in link5_rot[:, 0].tolist()],
                "link5_y_axis_world": [float(v) for v in link5_rot[:, 1].tolist()],
                "link5_z_axis_world": [float(v) for v in link5_rot[:, 2].tolist()],
                "cost": float(np.linalg.norm(residual)),
            }
        jac = _numeric_jacobian(
            q_arm,
            target_tcp,
            target_x_axis,
            target_z_axis,
            orientation_weight_m=orientation_weight_m,
        )
        mat = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0], dtype=np.float64)
        try:
            # residual is target - actual.  The numerical Jacobian above is the
            # derivative of that residual, so DLS solves J*dq = -residual.
            delta = -jac.T @ np.linalg.solve(mat, residual)
        except np.linalg.LinAlgError:
            break
        max_abs = float(np.max(np.abs(delta)))
        if max_abs > step_clip_deg:
            delta *= step_clip_deg / max_abs
        q_arm = q_arm + delta
        q_arm = clip_joints(np.r_[q_arm, 0.0])[:5]

    q_full = np.zeros(6, dtype=np.float64)
    q_full[:5] = q_arm
    tcp_pos, link5_pos, link5_rot = _fk_runtime_tcp(q_full)
    return {
        "q_deg": [float(v) for v in np.r_[q_arm, 0.0].tolist()],
        "converged": False,
        "iterations": int(max_iter),
        "pos_err_mm": float(np.linalg.norm(target_tcp - tcp_pos) * 1000.0),
        "x_axis_err_deg": float(_axis_angle_deg(link5_rot[:, 0], target_x_axis))
        if target_x_axis is not None
        else 0.0,
        "z_axis_err_deg": float(_axis_angle_deg(link5_rot[:, 2], target_z_axis))
        if target_z_axis is not None
        else 0.0,
        "tcp_local_m": [float(v) for v in tcp_pos.tolist()],
        "link5_local_m": [float(v) for v in link5_pos.tolist()],
        "link5_x_axis_world": [float(v) for v in link5_rot[:, 0].tolist()],
        "link5_y_axis_world": [float(v) for v in link5_rot[:, 1].tolist()],
        "link5_z_axis_world": [float(v) for v in link5_rot[:, 2].tolist()],
        "cost": float("inf"),
    }


def _target_geometry(
    cube_local: np.ndarray,
    cube_size_m: float,
    *,
    tangent_sign: float,
    radial_tip_past_near_face_m: float,
) -> dict[str, Any]:
    radial = _unit_xy(float(cube_local[0]), float(cube_local[1]))
    tangent = np.array([-radial[1], radial[0], 0.0], dtype=np.float64) * float(tangent_sign)
    half = float(cube_size_m) * 0.5
    tangent_center_offset_m = half - 0.008
    radial_center_offset_m = half - float(radial_tip_past_near_face_m)
    final_tcp = np.asarray(cube_local, dtype=np.float64).copy()
    final_tcp -= radial * radial_center_offset_m
    final_tcp -= tangent * tangent_center_offset_m
    return {
        "radial_axis": radial,
        "tangent_axis": tangent,
        "target_x_axis": tangent,
        "target_z_axis": radial,
        "target_tcp": final_tcp,
        "tangent_center_offset_m": float(tangent_center_offset_m),
        "radial_tip_past_near_face_m": float(radial_tip_past_near_face_m),
        "radial_center_offset_m": float(radial_center_offset_m),
    }


def _seed_set() -> list[np.ndarray]:
    seeds = [
        HOME_DEG,
        np.array([0.0, 25.0, 120.0, 15.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 45.0, 110.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([-10.0, 27.0, 118.0, 14.0, 0.0, 0.0], dtype=np.float64),
        np.array([10.0, 27.0, 118.0, 14.0, 0.0, 0.0], dtype=np.float64),
        np.array([-20.0, 30.0, 125.0, -10.0, 70.0, 0.0], dtype=np.float64),
        np.array([20.0, 30.0, 125.0, -10.0, -70.0, 0.0], dtype=np.float64),
    ]
    return seeds


def _offline_feasibility(
    cube_local: np.ndarray,
    cube_size_m: float,
    radial_tip_past_near_face_m: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for tangent_sign in (1.0, -1.0):
        geom = _target_geometry(
            cube_local,
            cube_size_m,
            tangent_sign=tangent_sign,
            radial_tip_past_near_face_m=radial_tip_past_near_face_m,
        )
        for orientation_weight_m in (0.02, 0.05, 0.10):
            best: dict[str, Any] | None = None
            for seed in _seed_set():
                result = _solve_runtime_ik(
                    np.asarray(geom["target_tcp"], dtype=np.float64),
                    seed,
                    target_x_axis=np.asarray(geom["target_x_axis"], dtype=np.float64),
                    target_z_axis=np.asarray(geom["target_z_axis"], dtype=np.float64),
                    orientation_weight_m=orientation_weight_m,
                    max_iter=700,
                    pos_tol_mm=1.0,
                    axis_tol_deg=3.0,
                )
                score = (
                    float(result["pos_err_mm"])
                    + 10.0 * max(0.0, float(result["x_axis_err_deg"]) - 3.0)
                    + 10.0 * max(0.0, float(result["z_axis_err_deg"]) - 3.0)
                )
                result.update(
                    {
                        "tangent_sign": float(tangent_sign),
                        "orientation_weight_m": float(orientation_weight_m),
                        "target_tcp_m": [float(v) for v in np.asarray(geom["target_tcp"]).tolist()],
                        "target_x_axis_world": [float(v) for v in np.asarray(geom["target_x_axis"]).tolist()],
                        "target_z_axis_world": [float(v) for v in np.asarray(geom["target_z_axis"]).tolist()],
                        "score": float(score),
                    }
                )
                if best is None or score < float(best["score"]):
                    best = result
            if best is not None:
                attempts.append(best)

    attempts = sorted(attempts, key=lambda item: float(item["score"]))
    best_strict = attempts[0] if attempts else None

    reachable: list[dict[str, Any]] = []
    for tangent_sign in (1.0, -1.0):
        geom = _target_geometry(
            cube_local,
            cube_size_m,
            tangent_sign=tangent_sign,
            radial_tip_past_near_face_m=radial_tip_past_near_face_m,
        )
        best_pos: dict[str, Any] | None = None
        for seed in _seed_set():
            result = _solve_runtime_ik(
                np.asarray(geom["target_tcp"], dtype=np.float64),
                seed,
                target_x_axis=None,
                target_z_axis=None,
                max_iter=500,
                pos_tol_mm=1.0,
            )
            q = np.asarray(result["q_deg"], dtype=np.float64)
            _tcp, _link5, rot = _fk_runtime_tcp(q)
            result.update(
                {
                    "tangent_sign": float(tangent_sign),
                    "target_tcp_m": [float(v) for v in np.asarray(geom["target_tcp"]).tolist()],
                    "target_x_axis_world": [float(v) for v in np.asarray(geom["target_x_axis"]).tolist()],
                    "target_z_axis_world": [float(v) for v in np.asarray(geom["target_z_axis"]).tolist()],
                    "x_axis_err_deg": float(_axis_angle_deg(rot[:, 0], np.asarray(geom["target_x_axis"]))),
                    "z_axis_err_deg": float(_axis_angle_deg(rot[:, 2], np.asarray(geom["target_z_axis"]))),
                }
            )
            if best_pos is None or float(result["pos_err_mm"]) < float(best_pos["pos_err_mm"]):
                best_pos = result
        if best_pos is not None:
            reachable.append(best_pos)

    strict_feasible = bool(
        best_strict
        and float(best_strict["pos_err_mm"]) <= 5.0
        and float(best_strict["x_axis_err_deg"]) <= 3.0
        and float(best_strict["z_axis_err_deg"]) <= 3.0
    )
    return {
        "strict_pose_family_feasible": strict_feasible,
        "strict_thresholds": {
            "tcp_position_mm": 5.0,
            "link5_x_axis_deg": 3.0,
            "link5_z_axis_deg": 3.0,
        },
        "best_strict_attempt": best_strict,
        "strict_attempts": attempts,
        "reachable_position_only_family": reachable,
    }


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "g0a_d323_alignment_trials.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary["trial_csv"] = _rel(csv_path)
    (out_dir / "g0a_d323_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# D323 G0a Frame Repair Probe",
        "",
        "이번 case는 G0a repair이며 신규 변수나 사다리 전진은 없다.",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- trials: `{summary['num_trials']}`",
        f"- pass_all: `{summary['pass_all_count']}/{summary['num_trials']}`",
        f"- hard_failure: `{summary['hard_failure']}`",
        f"- frame audit: `{summary['frame_audit_path']}`",
        f"- output CSV: `{summary['trial_csv']}`",
    ]
    if summary.get("stop_reason"):
        lines.extend(["", "## Stop Reason", "", str(summary["stop_reason"])])
    best_attempt = summary.get("best_strict_attempt")
    if best_attempt:
        lines.extend(
            [
                "",
                "## Best Strict Attempt",
                "",
                f"- tcp pose error: `{float(best_attempt['pos_err_mm']):.3f} mm`",
                f"- link5 +x error: `{float(best_attempt['x_axis_err_deg']):.3f} deg`",
                f"- link5 +z error: `{float(best_attempt['z_axis_err_deg']):.3f} deg`",
            ]
        )
    lines.extend(
        [
            "",
            "## Criteria",
            "",
            "- TCP pose error <= 5mm and link5 axis orientation error <= 3deg.",
            "- Fixed-jaw face gap to cube face <= 3mm and no penetration.",
            "- Cube XY displacement < 5mm.",
            "- Strict pass requires all 10 trials to satisfy all criteria.",
            "",
            "## Trial Table",
            "",
            "| trial | pose err mm | orient err deg | face gap mm | penetration mm | cube disp mm | pass |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {trial} | {tcp_pose_error_mm:.3f} | {orientation_error_deg:.3f} | "
            "{fixed_jaw_face_gap_mm:.3f} | {fixed_jaw_penetration_mm:.3f} | "
            "{cube_disp_xy_mm:.3f} | {pass_all} |".format(**row)
        )
    lines.extend(["", "## Failure Counts", ""])
    for key, value in summary["failure_counts"].items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "g0a_d323_alignment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32301)
    parser.add_argument("--cube_x", type=float, default=0.30)
    parser.add_argument("--cube_y", type=float, default=0.0)
    parser.add_argument("--cube_size_m", type=float, default=0.10)
    parser.add_argument("--cube_mass_kg", type=float, default=0.72)
    parser.add_argument("--static_friction", type=float, default=1.5)
    parser.add_argument("--dynamic_friction", type=float, default=1.2)
    parser.add_argument("--orientation_steps", type=int, default=260)
    parser.add_argument("--approach_steps", type=int, default=180)
    parser.add_argument("--hold_steps", type=int, default=80)
    parser.add_argument("--pre_clearance_m", type=float, default=0.040)
    parser.add_argument("--radial_tip_past_near_face_m", type=float, default=0.010)
    parser.add_argument("--episode_length_s", type=float, default=5.5)
    args = parser.parse_args()

    if int(args.num_trials) != 10:
        raise ValueError("D323 G0a is pre-registered for exactly 10 trials")
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)

    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    inner._compute_intermediate_values()

    try:
        hand_tcp_body_idx = inner._robot.find_bodies("hand_tcp")[0][0]
        hand_tcp_body_found = True
    except Exception:
        hand_tcp_body_idx = None
        hand_tcp_body_found = False

    frame_rows: list[dict[str, Any]] = []
    for env_idx, (pose_name, q_deg) in enumerate(AUDIT_POSES_DEG):
        q_rad = torch.tensor(np.radians(q_deg), device=device, dtype=torch.float32)
        joint_pos = inner._robot.data.joint_pos.detach().clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[env_idx, :] = q_rad
        inner._robot.write_joint_state_to_sim(joint_pos[env_idx : env_idx + 1], joint_vel[env_idx : env_idx + 1], env_ids=torch.tensor([env_idx], device=device))
        inner._robot.set_joint_position_target(joint_pos[env_idx : env_idx + 1], env_ids=torch.tensor([env_idx], device=device))
        inner.robot_dof_targets[env_idx, :] = q_rad
    env.step(zero)
    inner._compute_intermediate_values()

    origins_np = inner.scene.env_origins.detach().cpu().numpy()
    body_pos_np = inner._robot.data.body_pos_w.detach().cpu().numpy()
    body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
    tcp_np = inner._tcp_pos_w.detach().cpu().numpy()
    for env_idx, (pose_name, q_deg) in enumerate(AUDIT_POSES_DEG):
        origin = origins_np[env_idx]
        link5_pos = body_pos_np[env_idx, inner.link5_idx] - origin
        link5_quat = body_quat_np[env_idx, inner.link5_idx]
        link5_rot = _quat_wxyz_to_rot(link5_quat)
        tcp_local = tcp_np[env_idx] - origin
        gripper_pos = body_pos_np[env_idx, inner.gripper_link_idx] - origin
        gripper_quat = body_quat_np[env_idx, inner.gripper_link_idx]
        gripper_rot = _quat_wxyz_to_rot(gripper_quat)
        tcp_rel_link5 = link5_rot.T @ (tcp_local - link5_pos)
        grip_rel_link5 = link5_rot.T @ (gripper_pos - link5_pos)
        hand_tcp_entry: dict[str, Any] = {
            "computed_from_link5": True,
            "pos_local_m": [float(v) for v in tcp_local.tolist()],
            "axes_world": {
                "x": [float(v) for v in link5_rot[:, 0].tolist()],
                "y": [float(v) for v in link5_rot[:, 1].tolist()],
                "z": [float(v) for v in link5_rot[:, 2].tolist()],
            },
        }
        if hand_tcp_body_found and hand_tcp_body_idx is not None:
            hand_pos = body_pos_np[env_idx, hand_tcp_body_idx] - origin
            hand_quat = body_quat_np[env_idx, hand_tcp_body_idx]
            hand_rot = _quat_wxyz_to_rot(hand_quat)
            hand_tcp_entry.update(
                {
                    "computed_from_link5": False,
                    "body_pos_local_m": [float(v) for v in hand_pos.tolist()],
                    "body_quat_wxyz": [float(v) for v in hand_quat.tolist()],
                    "body_axes_world": {
                        "x": [float(v) for v in hand_rot[:, 0].tolist()],
                        "y": [float(v) for v in hand_rot[:, 1].tolist()],
                        "z": [float(v) for v in hand_rot[:, 2].tolist()],
                    },
                }
            )
        frame_rows.append(
            {
                "pose_name": pose_name,
                "env_index": int(env_idx),
                "joint_deg": [float(v) for v in q_deg.tolist()],
                "link5": {
                    "pos_local_m": [float(v) for v in link5_pos.tolist()],
                    "quat_wxyz": [float(v) for v in link5_quat.tolist()],
                    "axes_world": {
                        "x": [float(v) for v in link5_rot[:, 0].tolist()],
                        "y": [float(v) for v in link5_rot[:, 1].tolist()],
                        "z": [float(v) for v in link5_rot[:, 2].tolist()],
                    },
                },
                "hand_tcp": hand_tcp_entry,
                "gripper_link": {
                    "pos_local_m": [float(v) for v in gripper_pos.tolist()],
                    "quat_wxyz": [float(v) for v in gripper_quat.tolist()],
                    "axes_world": {
                        "x": [float(v) for v in gripper_rot[:, 0].tolist()],
                        "y": [float(v) for v in gripper_rot[:, 1].tolist()],
                        "z": [float(v) for v in gripper_rot[:, 2].tolist()],
                    },
                },
                "relative": {
                    "tcp_in_link5_m": [float(v) for v in tcp_rel_link5.tolist()],
                    "tcp_offset_error_mm": float(np.linalg.norm(tcp_rel_link5 - TCP_LOCAL_OFFSET_M) * 1000.0),
                    "gripper_link_in_link5_m": [float(v) for v in grip_rel_link5.tolist()],
                    "fixed_jaw_face_local_m": [float(v) for v in FIXED_JAW_FACE_LOCAL_M.tolist()],
                },
            }
        )

    cube_local = np.array(
        [float(args.cube_x), float(args.cube_y), float(TABLE_Z) + 0.5 * float(args.cube_size_m)],
        dtype=np.float64,
    )
    feasibility = _offline_feasibility(
        cube_local,
        float(args.cube_size_m),
        float(args.radial_tip_past_near_face_m),
    )
    frame_audit = {
        "artifact": "d323_g0a_frame_audit",
        "active_case": "G0a",
        "repair_only": True,
        "hand_tcp_body_found": bool(hand_tcp_body_found),
        "expected_contract": {
            "tcp_from_link5_local_m": [float(v) for v in TCP_LOCAL_OFFSET_M.tolist()],
            "tool_axis": "link5 local +z",
            "jaw_separation_axis": "link5 local +x; moving jaw swings toward +x",
            "fixed_jaw_face_local_m": [float(v) for v in FIXED_JAW_FACE_LOCAL_M.tolist()],
            "radial_tip_past_cube_near_face_m": float(args.radial_tip_past_near_face_m),
            "tangent_cube_center_offset_m": float(float(args.cube_size_m) * 0.5 - 0.008),
        },
        "frame_rows": frame_rows,
        "offline_feasibility": feasibility,
    }
    frame_path = args.out_dir / "frame_audit.json"
    frame_path.write_text(json.dumps(frame_audit, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    failure_counts = {
        "tcp_pose": 0,
        "orientation": 0,
        "fixed_jaw_gap": 0,
        "fixed_jaw_penetration": 0,
        "cube_displacement": 0,
        "arm_joint_tracking": 0,
    }

    if not bool(feasibility["strict_pose_family_feasible"]):
        summary = {
            "artifact": "d323_g0a_frame_repair_probe",
            "verdict": "D323_G0A_STRICT_POSE_INFEASIBLE_STOP",
            "num_trials": 0,
            "pass_all_count": 0,
            "hard_failure": True,
            "failure_counts": failure_counts,
            "frame_audit_path": _rel(frame_path),
            "trial_csv": "",
            "active_case": "G0a",
            "repair_only": True,
            "stop_reason": "Requested link5 +z radial and link5 +x tangent pose family was not feasible within 5mm/3deg thresholds; Step 3 retrial not run by prompt stop rule.",
            "best_strict_attempt": feasibility["best_strict_attempt"],
            "reachable_position_only_family": feasibility["reachable_position_only_family"],
            "new_variable": "none; D323 is G0a repair",
            "non_goals": [
                "no ladder advance",
                "no cylinder",
                "no gripper close",
                "no grasp",
                "no lift",
                "no RL/PPO",
                "no VLA/RoArm/B200",
                "no offset tuning loop",
            ],
            "cube_size_m": float(args.cube_size_m),
            "cube_mass_kg": float(args.cube_mass_kg),
            "static_friction": float(args.static_friction),
            "dynamic_friction": float(args.dynamic_friction),
            "radial_tip_past_near_face_m": float(args.radial_tip_past_near_face_m),
        }
        _write_outputs(args.out_dir, rows, summary)
        env.close()
        sim_app.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Reset all envs before the failable retrial.
    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    inner._compute_intermediate_values()
    env_ids = torch.arange(inner.num_envs, device=device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]
    cube_start_w = inner._sponge_pos_w.detach().clone()
    cube_start_local = (cube_start_w - origins).detach().cpu().numpy()

    best = feasibility["best_strict_attempt"]
    if best is None:
        raise RuntimeError("strict pose was marked feasible without a best strict attempt")
    tangent_sign = float(best["tangent_sign"])
    target_info = [
        _target_geometry(
            cube_start_local[idx],
            float(args.cube_size_m),
            tangent_sign=tangent_sign,
            radial_tip_past_near_face_m=float(args.radial_tip_past_near_face_m),
        )
        for idx in range(inner.num_envs)
    ]

    joint_targets_full = inner._robot.data.joint_pos.detach().clone()
    q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
    q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
    q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
    ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)

    total_steps = int(args.orientation_steps) + int(args.approach_steps) + int(args.hold_steps)
    with torch.inference_mode():
        for step in range(total_steps):
            if step < int(args.orientation_steps):
                alpha = 0.0
            elif step < int(args.orientation_steps) + int(args.approach_steps):
                alpha = (step - int(args.orientation_steps) + 1) / float(args.approach_steps)
            else:
                alpha = 1.0

            for idx, info in enumerate(target_info):
                final_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                radial = np.asarray(info["radial_axis"], dtype=np.float64)
                pre_tcp = final_tcp - radial * float(args.pre_clearance_m)
                target_tcp = pre_tcp + alpha * (final_tcp - pre_tcp)
                result = _solve_runtime_ik(
                    target_tcp,
                    np.degrees(q_targets_kin[idx]),
                    target_x_axis=np.asarray(info["target_x_axis"], dtype=np.float64),
                    target_z_axis=np.asarray(info["target_z_axis"], dtype=np.float64),
                    orientation_weight_m=0.05,
                    max_iter=120,
                    pos_tol_mm=1.0,
                    axis_tol_deg=3.0,
                )
                if not bool(result["converged"]):
                    ik_failure_counts[idx] += 1
                q_targets_kin[idx] = np.radians(np.asarray(result["q_deg"], dtype=np.float64))

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
    body_pos_np = inner._robot.data.body_pos_w.detach().cpu().numpy()
    body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
    origins_np = inner.scene.env_origins.detach().cpu().numpy()
    tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
    cube_final_w = inner._sponge_pos_w.detach().clone()
    cube_disp_xy = torch.linalg.norm(cube_final_w[:, 0:2] - cube_start_w[:, 0:2], dim=-1)
    joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy()
    actual_arm = joint_pos[:, inner._bc_arm_joint_ids]
    target_arm = q_targets_kin[:, :5]
    arm_joint_err_rad = np.max(np.abs(actual_arm - target_arm), axis=1)

    for idx, info in enumerate(target_info):
        target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
        target_x = np.asarray(info["target_x_axis"], dtype=np.float64)
        target_z = np.asarray(info["target_z_axis"], dtype=np.float64)
        tangent = np.asarray(info["tangent_axis"], dtype=np.float64)

        tcp_err_m = float(np.linalg.norm(tcp_local[idx] - target_tcp))
        link5_quat = body_quat_np[idx, inner.link5_idx]
        link5_rot = _quat_wxyz_to_rot(link5_quat)
        x_err_deg = _axis_angle_deg(link5_rot[:, 0], target_x)
        z_err_deg = _axis_angle_deg(link5_rot[:, 2], target_z)
        orientation_error_deg = max(x_err_deg, z_err_deg)

        cube_center = cube_start_local[idx]
        cube_face = cube_center - tangent * (float(args.cube_size_m) * 0.5)
        fixed_jaw_face = tcp_local[idx] - tangent * 0.008
        signed_gap_m = float(np.dot(cube_face - fixed_jaw_face, tangent))
        penetration_m = max(0.0, -signed_gap_m)
        disp_m = _safe_float(cube_disp_xy[idx])

        pass_tcp = tcp_err_m <= 0.005
        pass_orient = orientation_error_deg <= 3.0
        pass_gap = 0.0 <= signed_gap_m <= 0.003
        pass_pen = penetration_m <= 1.0e-6
        pass_disp = disp_m < 0.005
        pass_tracking = float(arm_joint_err_rad[idx]) <= 0.03
        if not pass_tcp:
            failure_counts["tcp_pose"] += 1
        if not pass_orient:
            failure_counts["orientation"] += 1
        if not pass_gap:
            failure_counts["fixed_jaw_gap"] += 1
        if not pass_pen:
            failure_counts["fixed_jaw_penetration"] += 1
        if not pass_disp:
            failure_counts["cube_displacement"] += 1
        if not pass_tracking:
            failure_counts["arm_joint_tracking"] += 1

        rows.append(
            {
                "trial": idx,
                "cube_x_m": float(cube_start_local[idx, 0]),
                "cube_y_m": float(cube_start_local[idx, 1]),
                "tangent_sign": float(tangent_sign),
                "tcp_target_x_m": float(target_tcp[0]),
                "tcp_target_y_m": float(target_tcp[1]),
                "tcp_target_z_m": float(target_tcp[2]),
                "tcp_actual_x_m": float(tcp_local[idx, 0]),
                "tcp_actual_y_m": float(tcp_local[idx, 1]),
                "tcp_actual_z_m": float(tcp_local[idx, 2]),
                "tcp_pose_error_mm": tcp_err_m * 1000.0,
                "link5_x_axis_error_deg": float(x_err_deg),
                "link5_z_axis_error_deg": float(z_err_deg),
                "orientation_error_deg": float(orientation_error_deg),
                "fixed_jaw_face_gap_mm": signed_gap_m * 1000.0,
                "fixed_jaw_penetration_mm": penetration_m * 1000.0,
                "cube_disp_xy_mm": disp_m * 1000.0,
                "ik_failure_steps": int(ik_failure_counts[idx]),
                "arm_joint_err_max_rad": float(arm_joint_err_rad[idx]),
                "pass_tcp_pose": bool(pass_tcp),
                "pass_orientation": bool(pass_orient),
                "pass_fixed_jaw_gap": bool(pass_gap),
                "pass_no_penetration": bool(pass_pen),
                "pass_cube_displacement": bool(pass_disp),
                "pass_arm_joint_tracking": bool(pass_tracking),
                "pass_all": bool(pass_tcp and pass_orient and pass_gap and pass_pen and pass_disp),
            }
        )

    pass_all_count = sum(1 for row in rows if row["pass_all"])
    hard_failure = any(count >= 3 for count in failure_counts.values())
    if pass_all_count == int(args.num_trials):
        verdict = "D323_G0A_FRAME_REPAIR_PASS"
    elif hard_failure:
        verdict = "D323_G0A_FRAME_REPAIR_FAIL"
    else:
        verdict = "D323_G0A_FRAME_REPAIR_PARTIAL"

    tracking_diag = {
        "triggered": bool(failure_counts["arm_joint_tracking"] > 0),
        "threshold_rad": 0.03,
        "max_err_rad": float(max(row["arm_joint_err_max_rad"] for row in rows)) if rows else 0.0,
        "interpretation": "tracking error exceeded actuator contract threshold"
        if failure_counts["arm_joint_tracking"] > 0
        else "tracking error within actuator contract threshold",
    }
    summary = {
        "artifact": "d323_g0a_frame_repair_probe",
        "verdict": verdict,
        "num_trials": int(args.num_trials),
        "pass_all_count": int(pass_all_count),
        "hard_failure": bool(hard_failure),
        "failure_rule": "hard fail if any alignment criterion misses in >=3/10 trials; strict pass requires 10/10 all-criteria pass",
        "failure_counts": failure_counts,
        "frame_audit_path": _rel(frame_path),
        "offline_feasibility": feasibility,
        "actuator_contract_diagnosis": tracking_diag,
        "new_variable": "none; D323 is G0a repair",
        "active_case": "G0a",
        "repair_only": True,
        "cube_size_m": float(args.cube_size_m),
        "cube_mass_kg": float(args.cube_mass_kg),
        "static_friction": float(args.static_friction),
        "dynamic_friction": float(args.dynamic_friction),
        "radial_tip_past_near_face_m": float(args.radial_tip_past_near_face_m),
        "pre_clearance_m": float(args.pre_clearance_m),
        "orientation_steps": int(args.orientation_steps),
        "approach_steps": int(args.approach_steps),
        "hold_steps": int(args.hold_steps),
        "robot_usd_path": _rel(args.robot_usd_path),
        "non_goals": [
            "no ladder advance",
            "no cylinder",
            "no gripper close",
            "no grasp",
            "no lift",
            "no RL/PPO",
            "no VLA/RoArm/B200",
            "no offset tuning loop",
        ],
    }
    _write_outputs(args.out_dir, rows, summary)
    env.close()
    sim_app.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
