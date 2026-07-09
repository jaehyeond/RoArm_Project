#!/usr/bin/env python3
"""D325 G0a redefined alignment criterion probe.

This is a G0a repair run only.  It adopts the D324 position-only tangent -1
family that D323 proved reachable, then reruns the fixed 10-trial alignment
gate with the D325 pre-registered criteria.  It does not close the gripper,
spawn a cylinder, grasp, lift, train RL/PPO, render trajectories, or advance the
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

from roarm_rl.viz_debug import draw_frames, frame_from_axes, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    DEFAULT_USD,
    FIXED_JAW_FACE_LOCAL_M,
    HOME_DEG,
    _axis_angle_deg,
    _quat_wxyz_to_rot,
    _safe_float,
    _solve_runtime_ik,
    _target_geometry,
)


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d325"
ADOPTED_TANGENT_SIGN = -1.0
TCP_POS_GATE_M = 0.005
JAW_TANGENT_GATE_DEG = 15.0
GAP_GATE_M = 0.005
CUBE_DISP_GATE_M = 0.005
TOP_CLEARANCE_GATE_M = 0.015


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _unit(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _horizontal_axis_error_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    axis_h = np.asarray([float(axis[0]), float(axis[1]), 0.0], dtype=np.float64)
    target_h = np.asarray([float(target_axis[0]), float(target_axis[1]), 0.0], dtype=np.float64)
    if float(np.linalg.norm(axis_h)) <= 1.0e-9:
        return 180.0
    return _axis_angle_deg(axis_h, target_h)


def _actual_frame(name: str, tcp: np.ndarray, rot: np.ndarray, *, label: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "position": [float(v) for v in tcp.tolist()],
        "axes": {
            "x": [float(v) for v in rot[:, 0].tolist()],
            "y": [float(v) for v in rot[:, 1].tolist()],
            "z": [float(v) for v in rot[:, 2].tolist()],
        },
        "role": "actual",
    }


def _target_frame(target_tcp: np.ndarray, tangent_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "d325_target_tcp",
        target_tcp,
        x_axis=tangent_axis,
        z_axis=[0.0, 0.0, 1.0],
        role="target",
        label="target TCP / jaw tangent",
        metadata={"tool_z_policy": "free"},
    )


def _object_frame(cube_center: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "cube_object_frame",
        cube_center,
        x_axis=[1.0, 0.0, 0.0],
        z_axis=[0.0, 0.0, 1.0],
        role="object",
        label="cube",
    )


def _fixed_jaw_frame(pos: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "fixed_jaw_face",
        pos,
        x_axis=x_axis,
        z_axis=z_axis,
        role="fixed_jaw",
        label="fixed jaw face",
    )


def _contact_point_frame(pos: np.ndarray, tangent_axis: np.ndarray) -> dict[str, Any]:
    return frame_from_axes(
        "cube_side_contact_point",
        pos,
        x_axis=tangent_axis,
        z_axis=[0.0, 0.0, 1.0],
        role="cube_face",
        label="cube side contact point",
    )


def _write_snapshot(
    out_dir: Path,
    trial: int,
    *,
    cube_center: np.ndarray,
    cube_size_m: float,
    target_tcp: np.ndarray,
    tangent_axis: np.ndarray,
    actual_tcp: np.ndarray,
    link5_rot: np.ndarray,
    fixed_jaw_face: np.ndarray,
    contact_point: np.ndarray,
    row: dict[str, Any],
) -> str:
    target = _target_frame(target_tcp, tangent_axis)
    actual = _actual_frame("actual_tcp_link5", actual_tcp, link5_rot, label="actual TCP/link5")
    fixed = _fixed_jaw_frame(fixed_jaw_face, link5_rot[:, 0], link5_rot[:, 2])
    contact = _contact_point_frame(contact_point, tangent_axis)
    obj = _object_frame(cube_center)
    path = out_dir / f"d325_trial_{trial + 1:02d}_snapshot.png"
    snapshot_frame_plot(
        path,
        [target, actual, fixed, contact, obj],
        cube={"center": cube_center.tolist(), "size": float(cube_size_m)},
        title=f"D325 G0a redefined alignment trial {trial + 1}",
        annotations=[
            "D325 criterion: TCP position + jaw tangent; tool +z is free.",
            f"pos err = {row['tcp_pose_error_mm']:.3f} mm",
            f"jaw tangent err = {row['jaw_tangent_error_deg']:.3f} deg",
            f"gap = {row['fixed_jaw_face_gap_mm']:.3f} mm, penetration = {row['fixed_jaw_penetration_mm']:.3f} mm",
            f"top clearance = {row['contact_point_below_top_mm']:.3f} mm",
            f"cube disp = {row['cube_disp_xy_mm']:.3f} mm",
        ],
    )
    return _rel(path)


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "g0a_d325_alignment_trials.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary["trial_csv"] = _rel(csv_path)
    (out_dir / "g0a_d325_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# D325 G0a Redefined Alignment Probe",
        "",
        "이번 case의 신규 변수: `[]` — D325 is criterion repair inside G0a.",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- pass_all: `{summary['pass_all_count']}/{summary['num_trials']}`",
        f"- hard_failure: `{summary['hard_failure']}`",
        f"- trial CSV: `{summary['trial_csv']}`",
        "",
        "## Trial Table",
        "",
        "| trial | pos mm | tangent deg | gap mm | penetration mm | top clearance mm | cube disp mm | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {tcp_pose_error_mm:.3f} | {jaw_tangent_error_deg:.3f} | "
            "{fixed_jaw_face_gap_mm:.3f} | {fixed_jaw_penetration_mm:.3f} | "
            "{contact_point_below_top_mm:.3f} | {cube_disp_xy_mm:.3f} | {pass_all} |".format(**row)
        )
    lines.extend(["", "## Failure Counts", ""])
    for key, value in summary["failure_counts"].items():
        lines.append(f"- {key}: `{value}`")
    if summary.get("snapshot_paths"):
        lines.extend(["", "## Snapshots", ""])
        for item in summary["snapshot_paths"]:
            lines.append(f"- trial {item['trial']}: `{item['path']}`")
    if summary.get("rerun_rrd"):
        lines.extend(["", "## Rerun", "", f"- `{summary['rerun_rrd']}`"])
    (out_dir / "g0a_d325_alignment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=32501)
    parser.add_argument("--cube_x", type=float, default=0.30)
    parser.add_argument("--cube_y", type=float, default=0.0)
    parser.add_argument("--cube_size_m", type=float, default=0.10)
    parser.add_argument("--cube_mass_kg", type=float, default=0.72)
    parser.add_argument("--static_friction", type=float, default=1.5)
    parser.add_argument("--dynamic_friction", type=float, default=1.2)
    parser.add_argument("--approach_steps", type=int, default=220)
    parser.add_argument("--hold_steps", type=int, default=100)
    parser.add_argument("--pre_clearance_m", type=float, default=0.040)
    parser.add_argument("--radial_tip_past_near_face_m", type=float, default=0.010)
    parser.add_argument("--episode_length_s", type=float, default=5.5)
    args = parser.parse_args()

    if int(args.num_trials) != 10:
        raise ValueError("D325 G0a is pre-registered for exactly 10 trials")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(args.robot_usd_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    cube_local = np.asarray(
        [float(args.cube_x), float(args.cube_y), float(TABLE_Z) + 0.5 * float(args.cube_size_m)],
        dtype=np.float64,
    )
    adopted_geom = _target_geometry(
        cube_local,
        float(args.cube_size_m),
        tangent_sign=ADOPTED_TANGENT_SIGN,
        radial_tip_past_near_face_m=float(args.radial_tip_past_near_face_m),
    )
    offline = _solve_runtime_ik(np.asarray(adopted_geom["target_tcp"], dtype=np.float64), HOME_DEG)
    offline_tcp = np.asarray(offline["tcp_local_m"], dtype=np.float64)
    offline_rot_cols = [
        np.asarray(offline["link5_x_axis_world"], dtype=np.float64),
        np.asarray(offline["link5_y_axis_world"], dtype=np.float64),
        np.asarray(offline["link5_z_axis_world"], dtype=np.float64),
    ]
    offline_rot = np.column_stack(offline_rot_cols)
    offline_fixed_jaw = offline_tcp + offline_rot @ FIXED_JAW_FACE_LOCAL_M
    cube_top_z = float(cube_local[2] + 0.5 * float(args.cube_size_m))
    structural_clearance_m = cube_top_z - float(offline_fixed_jaw[2])
    structural_height_ok = structural_clearance_m >= TOP_CLEARANCE_GATE_M
    if not structural_height_ok:
        summary = {
            "artifact": "d325_g0a_redefined_alignment_probe",
            "verdict": "D325_G0A_HEIGHT_GATE_STRUCTURAL_FAIL_STOP",
            "num_trials": 0,
            "pass_all_count": 0,
            "hard_failure": True,
            "stop_reason": "D325 adopted family places the fixed-jaw contact point too near the cube top edge.",
            "structural_contact_point_below_top_mm": float(structural_clearance_m * 1000.0),
            "new_variable": "none; D325 is G0a criterion repair",
            "active_case": "G0a",
        }
        _write_outputs(args.out_dir, [], summary)
        sim_app.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

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
    env_ids = torch.arange(inner.num_envs, device=device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]
    cube_start_w = inner._sponge_pos_w.detach().clone()
    cube_start_local = (cube_start_w - origins).detach().cpu().numpy()

    target_info = [
        _target_geometry(
            cube_start_local[idx],
            float(args.cube_size_m),
            tangent_sign=ADOPTED_TANGENT_SIGN,
            radial_tip_past_near_face_m=float(args.radial_tip_past_near_face_m),
        )
        for idx in range(inner.num_envs)
    ]

    joint_targets_full = inner._robot.data.joint_pos.detach().clone()
    q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
    q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
    q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
    ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)

    total_steps = int(args.approach_steps) + int(args.hold_steps)
    with torch.inference_mode():
        for step in range(total_steps):
            alpha = min(1.0, (step + 1) / float(args.approach_steps))
            for idx, info in enumerate(target_info):
                final_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                radial = np.asarray(info["radial_axis"], dtype=np.float64)
                pre_tcp = final_tcp - radial * float(args.pre_clearance_m)
                target_tcp = pre_tcp + alpha * (final_tcp - pre_tcp)
                result = _solve_runtime_ik(
                    target_tcp,
                    np.degrees(q_targets_kin[idx]),
                    target_x_axis=None,
                    target_z_axis=None,
                    max_iter=120,
                    pos_tol_mm=1.0,
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
    body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
    origins_np = inner.scene.env_origins.detach().cpu().numpy()
    tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
    cube_final_w = inner._sponge_pos_w.detach().clone()
    cube_disp_xy = torch.linalg.norm(cube_final_w[:, 0:2] - cube_start_w[:, 0:2], dim=-1)
    joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy()
    actual_arm = joint_pos[:, inner._bc_arm_joint_ids]
    target_arm = q_targets_kin[:, :5]
    arm_joint_err_rad = np.max(np.abs(actual_arm - target_arm), axis=1)

    rows: list[dict[str, Any]] = []
    snapshot_paths: list[dict[str, Any]] = []
    failure_counts = {
        "tcp_pose": 0,
        "jaw_tangent": 0,
        "fixed_jaw_gap": 0,
        "fixed_jaw_penetration": 0,
        "contact_height": 0,
        "cube_displacement": 0,
    }
    selected_snapshot_trials = {0, 4, 9}
    marker_status: dict[str, Any] = {"ok": False, "backend": "isaac_markers", "error": "not attempted"}
    marker_frames: list[dict[str, Any]] = []

    for idx, info in enumerate(target_info):
        cube_center = cube_start_local[idx]
        target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
        tangent = _unit(np.asarray(info["target_x_axis"], dtype=np.float64))
        tcp = tcp_local[idx]
        link5_quat = body_quat_np[idx, inner.link5_idx]
        link5_rot = _quat_wxyz_to_rot(link5_quat)
        link5_x = link5_rot[:, 0]

        tcp_err_m = float(np.linalg.norm(tcp - target_tcp))
        tangent_err_deg = float(_horizontal_axis_error_deg(link5_x, tangent))
        fixed_jaw_face = tcp + link5_rot @ FIXED_JAW_FACE_LOCAL_M
        cube_side_face = cube_center - tangent * (float(args.cube_size_m) * 0.5)
        signed_gap_m = float(np.dot(cube_side_face[:2] - fixed_jaw_face[:2], tangent[:2]))
        penetration_m = max(0.0, -signed_gap_m)
        contact_point = fixed_jaw_face + tangent * signed_gap_m
        top_clearance_m = float(cube_top_z - contact_point[2])
        disp_m = _safe_float(cube_disp_xy[idx])

        pass_tcp = tcp_err_m <= TCP_POS_GATE_M
        pass_tangent = tangent_err_deg <= JAW_TANGENT_GATE_DEG
        pass_gap = 0.0 <= signed_gap_m <= GAP_GATE_M
        pass_pen = penetration_m <= 1.0e-6
        pass_height = top_clearance_m >= TOP_CLEARANCE_GATE_M
        pass_disp = disp_m < CUBE_DISP_GATE_M
        if not pass_tcp:
            failure_counts["tcp_pose"] += 1
        if not pass_tangent:
            failure_counts["jaw_tangent"] += 1
        if not pass_gap:
            failure_counts["fixed_jaw_gap"] += 1
        if not pass_pen:
            failure_counts["fixed_jaw_penetration"] += 1
        if not pass_height:
            failure_counts["contact_height"] += 1
        if not pass_disp:
            failure_counts["cube_displacement"] += 1
        pass_all = bool(pass_tcp and pass_tangent and pass_gap and pass_pen and pass_height and pass_disp)
        row = {
            "trial": idx + 1,
            "cube_x_m": float(cube_center[0]),
            "cube_y_m": float(cube_center[1]),
            "target_tcp_x_m": float(target_tcp[0]),
            "target_tcp_y_m": float(target_tcp[1]),
            "target_tcp_z_m": float(target_tcp[2]),
            "actual_tcp_x_m": float(tcp[0]),
            "actual_tcp_y_m": float(tcp[1]),
            "actual_tcp_z_m": float(tcp[2]),
            "tcp_pose_error_mm": float(tcp_err_m * 1000.0),
            "jaw_tangent_error_deg": tangent_err_deg,
            "fixed_jaw_face_gap_mm": float(signed_gap_m * 1000.0),
            "fixed_jaw_penetration_mm": float(penetration_m * 1000.0),
            "contact_point_z_m": float(contact_point[2]),
            "contact_point_below_top_mm": float(top_clearance_m * 1000.0),
            "cube_disp_xy_mm": float(disp_m * 1000.0),
            "ik_failure_steps": int(ik_failure_counts[idx]),
            "arm_joint_err_max_rad": float(arm_joint_err_rad[idx]),
            "pass_tcp_pose": bool(pass_tcp),
            "pass_jaw_tangent": bool(pass_tangent),
            "pass_fixed_jaw_gap": bool(pass_gap),
            "pass_no_penetration": bool(pass_pen),
            "pass_contact_height": bool(pass_height),
            "pass_cube_displacement": bool(pass_disp),
            "pass_all": pass_all,
        }
        rows.append(row)
        if idx in selected_snapshot_trials:
            snap_path = _write_snapshot(
                args.out_dir,
                idx,
                cube_center=cube_center,
                cube_size_m=float(args.cube_size_m),
                target_tcp=target_tcp,
                tangent_axis=tangent,
                actual_tcp=tcp,
                link5_rot=link5_rot,
                fixed_jaw_face=fixed_jaw_face,
                contact_point=contact_point,
                row=row,
            )
            snapshot_paths.append({"trial": int(idx + 1), "path": snap_path})
        if idx == 0:
            marker_frames = [
                _target_frame(target_tcp, tangent),
                _actual_frame("actual_tcp_link5", tcp, link5_rot, label="actual TCP/link5"),
                _fixed_jaw_frame(fixed_jaw_face, link5_rot[:, 0], link5_rot[:, 2]),
                _contact_point_frame(contact_point, tangent),
                _object_frame(cube_center),
            ]

    if marker_frames:
        marker_status = draw_frames(marker_frames, prim_path="/World/D325G0aFrames")

    rerun_status = log_rerun(
        args.out_dir / "d325_trial_01_frames.rrd",
        frames=marker_frames,
        joint_state={
            "trial": 1,
            "actual_arm_joint_rad": [float(v) for v in actual_arm[0].tolist()],
            "target_arm_joint_rad": [float(v) for v in target_arm[0].tolist()],
        },
        urdf_path=args.robot_usd_path,
    )

    pass_all_count = sum(1 for row in rows if row["pass_all"])
    hard_failure = any(int(count) >= 3 for count in failure_counts.values()) or pass_all_count != int(args.num_trials)
    if pass_all_count == int(args.num_trials):
        verdict = "D325_G0A_REDEFINED_ALIGNMENT_PASS"
    elif any(int(count) >= 3 for count in failure_counts.values()):
        verdict = "D325_G0A_REDEFINED_ALIGNMENT_FAIL"
    else:
        verdict = "D325_G0A_REDEFINED_ALIGNMENT_PARTIAL"

    summary = {
        "artifact": "d325_g0a_redefined_alignment_probe",
        "verdict": verdict,
        "num_trials": int(args.num_trials),
        "pass_all_count": int(pass_all_count),
        "hard_failure": bool(hard_failure),
        "failure_rule": "failure if any of the four D325 criteria misses in >=3/10 trials; pass requires 10/10 all criteria",
        "failure_counts": failure_counts,
        "new_variable": "none; D325 is G0a criterion repair based on D323/D324 evidence",
        "active_case": "G0a",
        "adopted_pose_family": {
            "source": "D324 position_only tangent -1",
            "tangent_sign": ADOPTED_TANGENT_SIGN,
            "tool_z_policy": "free; determined by reachable kinematics",
            "jaw_x_horizontal_tangent_gate_deg": JAW_TANGENT_GATE_DEG,
            "target_tcp_m": [float(v) for v in np.asarray(adopted_geom["target_tcp"]).tolist()],
            "tangent_axis_world": [float(v) for v in np.asarray(adopted_geom["target_x_axis"]).tolist()],
        },
        "criteria": {
            "tcp_position_error_mm_max": TCP_POS_GATE_M * 1000.0,
            "jaw_tangent_error_deg_max": JAW_TANGENT_GATE_DEG,
            "fixed_jaw_horizontal_gap_mm_max": GAP_GATE_M * 1000.0,
            "no_penetration": True,
            "contact_point_below_cube_top_mm_min": TOP_CLEARANCE_GATE_M * 1000.0,
            "cube_displacement_mm_max": CUBE_DISP_GATE_M * 1000.0,
        },
        "structural_contact_point_below_top_mm": float(structural_clearance_m * 1000.0),
        "snapshot_paths": snapshot_paths,
        "marker_status": marker_status,
        "rerun_status": rerun_status,
        "rerun_rrd": _rel(args.out_dir / "d325_trial_01_frames.rrd") if rerun_status.get("ok") else "",
        "cube_size_m": float(args.cube_size_m),
        "cube_mass_kg": float(args.cube_mass_kg),
        "static_friction": float(args.static_friction),
        "dynamic_friction": float(args.dynamic_friction),
        "approach_steps": int(args.approach_steps),
        "hold_steps": int(args.hold_steps),
        "robot_usd_path": _rel(args.robot_usd_path),
        "non_goals": [
            "no cylinder",
            "no gripper close",
            "no grasp",
            "no lift",
            "no RL/PPO",
            "no large render",
            "no position randomization",
            "no ladder advance",
            "no B200/RoArm/VLA",
        ],
        "rows": rows,
    }
    _write_outputs(args.out_dir, rows, summary)
    print(
        "[d325-g0a] "
        f"verdict={verdict} pass_all={pass_all_count}/{args.num_trials} "
        f"failures={failure_counts} marker_ok={marker_status.get('ok')} "
        f"rerun_ok={rerun_status.get('ok')} out_dir={_rel(args.out_dir)}"
    )
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
