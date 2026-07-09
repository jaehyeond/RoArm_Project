#!/usr/bin/env python3
"""D326 G0a execution-contract diagnosis and one-repair retest.

This is a repair-only G0a session.  It keeps the D325 alignment criterion and
the D324 position-only tangent-minus pose family fixed, diagnoses why runtime
execution missed the offline IK target, applies exactly one execution-contract
repair, and reruns the same 10-trial gate.  It does not advance to G0b, close
the gripper, grasp, lift, train RL/PPO, or generate data.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, frame_from_axes, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    FIXED_JAW_FACE_LOCAL_M,
    HOME_DEG,
    _axis_angle_deg,
    _fk_runtime_tcp,
    _quat_wxyz_to_rot,
    _safe_float,
    _solve_runtime_ik,
    _target_geometry,
)


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d326"
DEFAULT_ROBOT_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
DEFAULT_URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"

ARM_JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
]
GRIPPER_JOINT_NAME = "link5_to_gripper_link"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]

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


def _target_frame(target_tcp: np.ndarray, tangent_axis: np.ndarray, *, name: str = "target_tcp") -> dict[str, Any]:
    return frame_from_axes(
        name,
        target_tcp,
        x_axis=tangent_axis,
        z_axis=[0.0, 0.0, 1.0],
        role="target",
        label="target TCP / jaw tangent",
        metadata={"tool_z_policy": "free"},
    )


def _actual_frame(name: str, tcp: np.ndarray, rot: np.ndarray, *, label: str, role: str = "actual") -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "position": [float(v) for v in tcp.tolist()],
        "axes": {
            "x": [float(v) for v in rot[:, 0].tolist()],
            "y": [float(v) for v in rot[:, 1].tolist()],
            "z": [float(v) for v in rot[:, 2].tolist()],
        },
        "role": role,
    }


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


def _version_manifest() -> dict[str, Any]:
    mods = ["omni.kit.app", "pxr.Usd", "isaaclab", "isaacsim", "numpy", "psutil", "rerun"]
    out: dict[str, Any] = {"python": sys.version, "python_executable": sys.executable}
    for name in mods:
        try:
            mod = importlib.import_module(name)
            out[name] = {"file": getattr(mod, "__file__", None), "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            out[name] = {"error": repr(exc)}
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        out["omni.kit.app"]["kit_version"] = app.get_build_version()
    except Exception as exc:
        out.setdefault("omni.kit.app_runtime", {})["error"] = repr(exc)
    return out


def _joint_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(ALL_JOINT_NAMES)}


def _frames_for_state(
    *,
    cube_center: np.ndarray,
    target_tcp: np.ndarray,
    tangent: np.ndarray,
    actual_tcp: np.ndarray,
    actual_rot: np.ndarray,
    fixed_jaw_face: np.ndarray,
    contact_point: np.ndarray,
    commanded_tcp: np.ndarray | None = None,
    commanded_rot: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    frames = [
        _target_frame(target_tcp, tangent),
        _actual_frame("actual_tcp_link5", actual_tcp, actual_rot, label="actual TCP/link5"),
        _fixed_jaw_frame(fixed_jaw_face, actual_rot[:, 0], actual_rot[:, 2]),
        _contact_point_frame(contact_point, tangent),
        _object_frame(cube_center),
    ]
    if commanded_tcp is not None and commanded_rot is not None:
        frames.append(
            _actual_frame(
                "commanded_tcp_link5",
                commanded_tcp,
                commanded_rot,
                label="commanded TCP/link5",
                role="candidate",
            )
        )
    return frames


def _evaluate_alignment(
    *,
    trial: int,
    cube_center: np.ndarray,
    cube_size_m: float,
    target_tcp: np.ndarray,
    tangent: np.ndarray,
    actual_tcp: np.ndarray,
    link5_rot: np.ndarray,
    cube_start_w: np.ndarray,
    cube_final_w: np.ndarray,
    target_arm: np.ndarray,
    actual_arm: np.ndarray,
    ik_failure_steps: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cube_top_z = float(cube_center[2] + 0.5 * cube_size_m)
    link5_x = link5_rot[:, 0]
    tcp_err_m = float(np.linalg.norm(actual_tcp - target_tcp))
    tangent_err_deg = float(_horizontal_axis_error_deg(link5_x, tangent))
    fixed_jaw_face = actual_tcp + link5_rot @ FIXED_JAW_FACE_LOCAL_M
    cube_side_face = cube_center - tangent * (cube_size_m * 0.5)
    signed_gap_m = float(np.dot(cube_side_face[:2] - fixed_jaw_face[:2], tangent[:2]))
    penetration_m = max(0.0, -signed_gap_m)
    contact_point = fixed_jaw_face + tangent * signed_gap_m
    top_clearance_m = float(cube_top_z - contact_point[2])
    disp_m = float(np.linalg.norm(cube_final_w[:2] - cube_start_w[:2]))
    arm_joint_err_rad = float(np.max(np.abs(actual_arm - target_arm)))

    pass_tcp = tcp_err_m <= TCP_POS_GATE_M
    pass_tangent = tangent_err_deg <= JAW_TANGENT_GATE_DEG
    pass_gap = 0.0 <= signed_gap_m <= GAP_GATE_M
    pass_pen = penetration_m <= 1.0e-6
    pass_height = top_clearance_m >= TOP_CLEARANCE_GATE_M
    pass_disp = disp_m < CUBE_DISP_GATE_M
    pass_all = bool(pass_tcp and pass_tangent and pass_gap and pass_pen and pass_height and pass_disp)

    commanded_tcp, _link5_pos, commanded_rot = _fk_runtime_tcp(np.degrees(np.r_[target_arm, 0.0]))
    frames = _frames_for_state(
        cube_center=cube_center,
        target_tcp=target_tcp,
        tangent=tangent,
        actual_tcp=actual_tcp,
        actual_rot=link5_rot,
        fixed_jaw_face=fixed_jaw_face,
        contact_point=contact_point,
        commanded_tcp=commanded_tcp,
        commanded_rot=commanded_rot,
    )
    row = {
        "trial": int(trial),
        "target_tcp_x_m": float(target_tcp[0]),
        "target_tcp_y_m": float(target_tcp[1]),
        "target_tcp_z_m": float(target_tcp[2]),
        "actual_tcp_x_m": float(actual_tcp[0]),
        "actual_tcp_y_m": float(actual_tcp[1]),
        "actual_tcp_z_m": float(actual_tcp[2]),
        "commanded_tcp_x_m": float(commanded_tcp[0]),
        "commanded_tcp_y_m": float(commanded_tcp[1]),
        "commanded_tcp_z_m": float(commanded_tcp[2]),
        "tcp_pose_error_mm": float(tcp_err_m * 1000.0),
        "commanded_tcp_pose_error_mm": float(np.linalg.norm(commanded_tcp - target_tcp) * 1000.0),
        "jaw_tangent_error_deg": tangent_err_deg,
        "fixed_jaw_face_gap_mm": float(signed_gap_m * 1000.0),
        "fixed_jaw_penetration_mm": float(penetration_m * 1000.0),
        "contact_point_z_m": float(contact_point[2]),
        "contact_point_below_top_mm": float(top_clearance_m * 1000.0),
        "cube_disp_xy_mm": float(disp_m * 1000.0),
        "ik_failure_steps": int(ik_failure_steps),
        "arm_joint_err_max_rad": arm_joint_err_rad,
        "pass_tcp_pose": bool(pass_tcp),
        "pass_jaw_tangent": bool(pass_tangent),
        "pass_fixed_jaw_gap": bool(pass_gap),
        "pass_no_penetration": bool(pass_pen),
        "pass_contact_height": bool(pass_height),
        "pass_cube_displacement": bool(pass_disp),
        "pass_all": pass_all,
    }
    return row, fixed_jaw_face, contact_point, frames


def _write_snapshot(
    path: Path,
    *,
    cube_center: np.ndarray,
    cube_size_m: float,
    frames: list[dict[str, Any]],
    row: dict[str, Any],
    title: str,
) -> str:
    snapshot_frame_plot(
        path,
        frames,
        cube={"center": cube_center.tolist(), "size": float(cube_size_m)},
        title=title,
        annotations=[
            "D326 G0a: D325 criterion unchanged; tool +z free, jaw tangent -1.",
            f"pos err = {row['tcp_pose_error_mm']:.3f} mm",
            f"cmd pos err = {row['commanded_tcp_pose_error_mm']:.3f} mm",
            f"jaw tangent err = {row['jaw_tangent_error_deg']:.3f} deg",
            f"gap = {row['fixed_jaw_face_gap_mm']:.3f} mm, penetration = {row['fixed_jaw_penetration_mm']:.3f} mm",
            f"top clearance = {row['contact_point_below_top_mm']:.3f} mm",
            f"cube disp = {row['cube_disp_xy_mm']:.3f} mm",
        ],
    )
    return _rel(path)


def _configure_env_cfg(args: argparse.Namespace, num_envs: int, *, arm_effort_limit: float | None = None) -> Any:
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.roarm_stack_env import TABLE_Z

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    if arm_effort_limit is not None:
        env_cfg.robot.actuators["arm"].effort_limit_sim = float(arm_effort_limit)
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
    return env_cfg


def _make_env(args: argparse.Namespace, num_envs: int, *, arm_effort_limit: float | None = None) -> tuple[Any, Any, Any]:
    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg = _configure_env_cfg(args, num_envs, arm_effort_limit=arm_effort_limit)
    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device)
    return env, inner, zero


def _reset_env(inner: Any, env: Any, zero: Any) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    inner._compute_intermediate_values()
    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]
    cube_start_w = inner._sponge_pos_w.detach().clone()
    cube_start_local = (cube_start_w - origins).detach().cpu().numpy()
    return origins, cube_start_w, cube_start_local, cube_start_w.detach().cpu().numpy(), origins.detach().cpu().numpy()


def _target_info(cube_start_local: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        _target_geometry(
            cube_start_local[idx],
            float(args.cube_size_m),
            tangent_sign=ADOPTED_TANGENT_SIGN,
            radial_tip_past_near_face_m=float(args.radial_tip_past_near_face_m),
        )
        for idx in range(cube_start_local.shape[0])
    ]


def _state_eval_rows(
    inner: Any,
    origins: Any,
    cube_start_w: Any,
    cube_start_local: np.ndarray,
    target_info: list[dict[str, Any]],
    target_arm_by_env: np.ndarray,
    ik_failure_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    inner._compute_intermediate_values()
    body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
    tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
    cube_final_w = inner._sponge_pos_w.detach().clone()
    cube_final_w_np = cube_final_w.detach().cpu().numpy()
    cube_start_w_np = cube_start_w.detach().cpu().numpy()
    joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy()
    actual_arm = joint_pos[:, inner._bc_arm_joint_ids]

    rows: list[dict[str, Any]] = []
    frame_sets: list[list[dict[str, Any]]] = []
    for idx, info in enumerate(target_info):
        target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
        tangent = _unit(np.asarray(info["target_x_axis"], dtype=np.float64))
        link5_quat = body_quat_np[idx, inner.link5_idx]
        link5_rot = _quat_wxyz_to_rot(link5_quat)
        row, _fixed, _contact, frames = _evaluate_alignment(
            trial=idx + 1,
            cube_center=cube_start_local[idx],
            cube_size_m=float(args.cube_size_m),
            target_tcp=target_tcp,
            tangent=tangent,
            actual_tcp=tcp_local[idx],
            link5_rot=link5_rot,
            cube_start_w=cube_start_w_np[idx],
            cube_final_w=cube_final_w_np[idx],
            target_arm=target_arm_by_env[idx],
            actual_arm=actual_arm[idx],
            ik_failure_steps=int(ik_failure_counts[idx]),
        )
        rows.append(row)
        frame_sets.append(frames)
    return rows, frame_sets


def _teleport_check(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    env, inner, zero = _make_env(args, 1)
    origins, cube_start_w, cube_start_local, _cube_start_w_np, _origins_np = _reset_env(inner, env, zero)
    info = _target_info(cube_start_local, args)[0]
    result = _solve_runtime_ik(np.asarray(info["target_tcp"], dtype=np.float64), HOME_DEG)
    q_rad = np.radians(np.asarray(result["q_deg"], dtype=np.float64))

    env_ids = torch.tensor([0], device=inner.device, dtype=torch.long)
    joint_pos = inner._robot.data.joint_pos.detach().clone()
    joint_pos[0, inner._bc_arm_joint_ids] = torch.tensor(q_rad[:5], device=inner.device, dtype=torch.float32)
    joint_pos[0, inner.gripper_joint_idx] = 0.0
    joint_vel = torch.zeros_like(joint_pos)
    inner._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    inner._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    inner.robot_dof_targets[env_ids] = joint_pos[env_ids]
    inner._compute_intermediate_values()

    target_arm = np.expand_dims(q_rad[:5], axis=0)
    rows, frame_sets = _state_eval_rows(
        inner,
        origins,
        cube_start_w,
        cube_start_local,
        [info],
        target_arm,
        np.zeros(1, dtype=np.int64),
        args,
    )
    row = rows[0]
    joint_all = joint_pos[0].detach().cpu().numpy().astype(np.float64)
    rrd_path = args.out_dir / "d326_teleport_static_v2.rrd"
    rrd_status = log_rerun(
        rrd_path,
        frames=frame_sets[0],
        joint_state={"phase": "teleport_static_check", "pass": bool(row["pass_all"])},
        joint_trace=[
            {
                "step": 0,
                "phase": "teleport_static",
                "actual_joint_rad_by_name": _joint_dict(joint_all),
                "commanded_joint_rad_by_name": _joint_dict(joint_all),
                "tcp_pose_error_mm": row["tcp_pose_error_mm"],
                "commanded_tcp_pose_error_mm": row["commanded_tcp_pose_error_mm"],
                "arm_joint_err_max_rad": row["arm_joint_err_max_rad"],
                "frames": frame_sets[0],
            }
        ],
        cube={"center": cube_start_local[0].tolist(), "size": float(args.cube_size_m)},
        urdf_path=args.urdf_path,
        live_viewer=bool(args.live_viewer),
        app_id="roarm_g0a_d326_teleport_static",
    )
    snapshot_path = _write_snapshot(
        args.out_dir / "d326_teleport_static_check.png",
        cube_center=cube_start_local[0],
        cube_size_m=float(args.cube_size_m),
        frames=frame_sets[0],
        row=row,
        title="D326 teleport static check",
    )
    marker_status = draw_frames(frame_sets[0], prim_path="/World/D326TeleportFrames")
    env.close()
    return {
        "prediction": "static check should satisfy D325 criteria if offline IK is valid; otherwise stop before contract repair",
        "offline_ik": result,
        "row": row,
        "pass": bool(row["pass_all"]),
        "snapshot": snapshot_path,
        "marker_status": marker_status,
        "rrd_status": rrd_status,
        "rrd_path": _rel(rrd_path) if rrd_status.get("ok") else "",
    }


def _torque_saturation(inner: Any, env_id: int) -> dict[str, Any]:
    data = inner._robot.data
    arm_ids = inner._bc_arm_joint_ids
    out: dict[str, Any] = {"available": False}
    effort_limits = getattr(data, "joint_effort_limits", None)
    applied = getattr(data, "applied_torque", None)
    computed = getattr(data, "computed_torque", None)
    if effort_limits is None and applied is None and computed is None:
        return out
    out["available"] = True
    if effort_limits is not None:
        vals = effort_limits[env_id, arm_ids].detach().cpu().numpy().astype(np.float64)
        out["effort_limit_arm_nm"] = [float(v) for v in vals.tolist()]
    if applied is not None:
        vals = applied[env_id, arm_ids].detach().cpu().numpy().astype(np.float64)
        out["applied_torque_arm_nm"] = [float(v) for v in vals.tolist()]
    if computed is not None:
        vals = computed[env_id, arm_ids].detach().cpu().numpy().astype(np.float64)
        out["computed_torque_arm_nm"] = [float(v) for v in vals.tolist()]
    if "effort_limit_arm_nm" in out and "applied_torque_arm_nm" in out:
        lim = np.asarray(out["effort_limit_arm_nm"], dtype=np.float64)
        app = np.abs(np.asarray(out["applied_torque_arm_nm"], dtype=np.float64))
        out["saturation_rate"] = float(np.mean(app >= np.maximum(lim - 1.0e-4, 0.0)))
        out["max_applied_over_limit"] = float(np.max(app / np.maximum(lim, 1.0e-9)))
    return out


def _run_motion(
    args: argparse.Namespace,
    *,
    num_envs: int,
    approach_steps: int,
    hold_steps: int,
    trace: bool,
    label: str,
    arm_effort_limit: float | None = None,
) -> dict[str, Any]:
    import torch

    env, inner, zero = _make_env(args, num_envs, arm_effort_limit=arm_effort_limit)
    origins, cube_start_w, cube_start_local, _cube_start_w_np, _origins_np = _reset_env(inner, env, zero)
    target_info = _target_info(cube_start_local, args)

    joint_targets_full = inner._robot.data.joint_pos.detach().clone()
    q_targets_kin = np.zeros((inner.num_envs, 6), dtype=np.float64)
    q_targets_kin[:, :5] = joint_targets_full[:, inner._bc_arm_joint_ids].detach().cpu().numpy().astype(np.float64)
    q_targets_kin[:, 5] = joint_targets_full[:, inner.gripper_joint_idx].detach().cpu().numpy().astype(np.float64)
    ik_failure_counts = np.zeros(inner.num_envs, dtype=np.int64)
    trace_rows: list[dict[str, Any]] = []
    max_step_command_delta_rad = 0.0
    max_total_required_delta_rad = 0.0
    final_target_arm = q_targets_kin[:, :5].copy()

    total_steps = int(approach_steps) + int(hold_steps)
    with torch.inference_mode():
        for step in range(total_steps):
            prev_q = q_targets_kin.copy()
            alpha = min(1.0, (step + 1) / float(approach_steps))
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

            step_delta = np.max(np.abs(q_targets_kin[:, :5] - prev_q[:, :5]))
            max_step_command_delta_rad = max(max_step_command_delta_rad, float(step_delta))
            targets_t = joint_targets_full.detach().clone()
            targets_t[:, inner._bc_arm_joint_ids] = torch.tensor(
                q_targets_kin[:, :5], device=inner.device, dtype=torch.float32
            )
            targets_t[:, inner.gripper_joint_idx] = 0.0
            joint_targets_full = targets_t.detach().clone()
            inner._external_joint_targets_override = targets_t
            env.step(zero)
            inner._compute_intermediate_values()
            final_target_arm = q_targets_kin[:, :5].copy()

            if trace:
                body_quat_np = inner._robot.data.body_quat_w.detach().cpu().numpy()
                origins_np = origins.detach().cpu().numpy()
                tcp_local = (inner._tcp_pos_w - origins).detach().cpu().numpy()
                actual_all = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
                commanded_all = targets_t[0].detach().cpu().numpy().astype(np.float64)
                actual_arm = actual_all[inner._bc_arm_joint_ids]
                target_arm = commanded_all[inner._bc_arm_joint_ids]
                info = target_info[0]
                target_tcp = np.asarray(info["target_tcp"], dtype=np.float64)
                tangent = _unit(np.asarray(info["target_x_axis"], dtype=np.float64))
                link5_rot = _quat_wxyz_to_rot(body_quat_np[0, inner.link5_idx])
                row, _fixed, _contact, frames = _evaluate_alignment(
                    trial=1,
                    cube_center=cube_start_local[0],
                    cube_size_m=float(args.cube_size_m),
                    target_tcp=target_tcp,
                    tangent=tangent,
                    actual_tcp=tcp_local[0],
                    link5_rot=link5_rot,
                    cube_start_w=cube_start_w.detach().cpu().numpy()[0],
                    cube_final_w=inner._sponge_pos_w[0].detach().cpu().numpy(),
                    target_arm=target_arm,
                    actual_arm=actual_arm,
                    ik_failure_steps=int(ik_failure_counts[0]),
                )
                max_total_required_delta_rad = max(
                    max_total_required_delta_rad,
                    float(np.max(np.abs(target_arm - actual_arm))),
                )
                torque = _torque_saturation(inner, 0)
                trace_rows.append(
                    {
                        "step": int(step),
                        "phase": "approach" if step < approach_steps else "hold",
                        "alpha": float(alpha),
                        "actual_joint_rad_by_name": _joint_dict(actual_all),
                        "commanded_joint_rad_by_name": _joint_dict(commanded_all),
                        "tcp_pose_error_mm": row["tcp_pose_error_mm"],
                        "commanded_tcp_pose_error_mm": row["commanded_tcp_pose_error_mm"],
                        "arm_joint_err_max_rad": row["arm_joint_err_max_rad"],
                        "torque": torque,
                        "frames": frames,
                    }
                )

    rows, frame_sets = _state_eval_rows(
        inner,
        origins,
        cube_start_w,
        cube_start_local,
        target_info,
        final_target_arm,
        ik_failure_counts,
        args,
    )
    marker_status = draw_frames(frame_sets[0], prim_path=f"/World/D326{label}Frames") if frame_sets else {}
    snapshots: list[dict[str, Any]] = []
    for trial_idx in (0, 4, 9):
        if trial_idx < len(rows):
            path = args.out_dir / f"d326_{label}_trial_{trial_idx + 1:02d}_snapshot.png"
            snapshots.append(
                {
                    "trial": int(trial_idx + 1),
                    "path": _write_snapshot(
                        path,
                        cube_center=cube_start_local[trial_idx],
                        cube_size_m=float(args.cube_size_m),
                        frames=frame_sets[trial_idx],
                        row=rows[trial_idx],
                        title=f"D326 {label} trial {trial_idx + 1}",
                    ),
                }
            )

    rrd_status: dict[str, Any] = {"ok": False, "skipped": not trace}
    rrd_path = ""
    if trace and trace_rows:
        rrd_file = args.out_dir / f"d326_{label}_trace_v2.rrd"
        rrd_status = log_rerun(
            rrd_file,
            frames=frame_sets[0],
            joint_state={
                "label": label,
                "approach_steps": int(approach_steps),
                "hold_steps": int(hold_steps),
            },
            joint_trace=trace_rows,
            cube={"center": cube_start_local[0].tolist(), "size": float(args.cube_size_m)},
            urdf_path=args.urdf_path,
            live_viewer=bool(args.live_viewer),
            app_id=f"roarm_g0a_d326_{label}",
        )
        if rrd_status.get("ok"):
            rrd_path = _rel(rrd_file)

    failure_counts = {
        "tcp_pose": sum(1 for row in rows if not row["pass_tcp_pose"]),
        "jaw_tangent": sum(1 for row in rows if not row["pass_jaw_tangent"]),
        "fixed_jaw_gap": sum(1 for row in rows if not row["pass_fixed_jaw_gap"]),
        "fixed_jaw_penetration": sum(1 for row in rows if not row["pass_no_penetration"]),
        "contact_height": sum(1 for row in rows if not row["pass_contact_height"]),
        "cube_displacement": sum(1 for row in rows if not row["pass_cube_displacement"]),
    }
    pass_all_count = sum(1 for row in rows if row["pass_all"])
    env.close()
    return {
        "label": label,
        "num_envs": int(num_envs),
        "approach_steps": int(approach_steps),
        "hold_steps": int(hold_steps),
        "arm_effort_limit": arm_effort_limit,
        "pass_all_count": int(pass_all_count),
        "failure_counts": failure_counts,
        "rows": rows,
        "snapshots": snapshots,
        "marker_status": marker_status,
        "rrd_status": rrd_status,
        "rrd_path": rrd_path,
        "trace_stats": _trace_stats(trace_rows, max_step_command_delta_rad, max_total_required_delta_rad),
    }


def _trace_stats(trace_rows: list[dict[str, Any]], max_step_command_delta_rad: float, max_total_required_delta_rad: float) -> dict[str, Any]:
    if not trace_rows:
        return {
            "trace_steps": 0,
            "max_step_command_delta_rad": float(max_step_command_delta_rad),
            "max_total_required_delta_rad": float(max_total_required_delta_rad),
        }
    first = trace_rows[0]
    mid = trace_rows[len(trace_rows) // 2]
    last = trace_rows[-1]
    pos = np.asarray([float(row["tcp_pose_error_mm"]) for row in trace_rows], dtype=np.float64)
    joint_err = np.asarray([float(row["arm_joint_err_max_rad"]) for row in trace_rows], dtype=np.float64)
    sat_rates = []
    for row in trace_rows:
        torque = row.get("torque", {})
        if isinstance(torque, dict) and "saturation_rate" in torque:
            sat_rates.append(float(torque["saturation_rate"]))
    return {
        "trace_steps": len(trace_rows),
        "first_tcp_error_mm": float(first["tcp_pose_error_mm"]),
        "mid_tcp_error_mm": float(mid["tcp_pose_error_mm"]),
        "final_tcp_error_mm": float(last["tcp_pose_error_mm"]),
        "min_tcp_error_mm": float(np.min(pos)),
        "final_minus_mid_tcp_error_mm": float(last["tcp_pose_error_mm"] - mid["tcp_pose_error_mm"]),
        "max_joint_err_rad": float(np.max(joint_err)),
        "final_joint_err_rad": float(last["arm_joint_err_max_rad"]),
        "max_step_command_delta_rad": float(max_step_command_delta_rad),
        "max_total_required_delta_rad": float(max_total_required_delta_rad),
        "torque_saturation_rate_max": float(max(sat_rates)) if sat_rates else None,
        "torque_saturation_rate_final": float(sat_rates[-1]) if sat_rates else None,
    }


def _diagnose_and_choose_repair(baseline: dict[str, Any], x3: dict[str, Any]) -> tuple[dict[str, Any], str]:
    b = baseline["trace_stats"]
    x = x3["rows"][0] if x3.get("rows") else {}
    baseline_final = float(b.get("final_tcp_error_mm", math.inf))
    x3_final = float(x.get("tcp_pose_error_mm", math.inf))
    improvement = baseline_final - x3_final
    torque_sat = b.get("torque_saturation_rate_max")
    questions = {
        "time_shortage": {
            "prediction": "likely if baseline error is still decreasing or x3 reduces TCP error materially",
            "judgement": bool(improvement > 10.0 or x3_final <= 5.0),
            "evidence": {
                "baseline_final_tcp_error_mm": baseline_final,
                "x3_final_tcp_error_mm": x3_final,
                "improvement_mm": float(improvement),
                "baseline_final_minus_mid_mm": b.get("final_minus_mid_tcp_error_mm"),
            },
        },
        "lead_limit": {
            "prediction": "unlikely: D325 external override bypasses env joint_target_lead_limit_rad",
            "judgement": False,
            "evidence": "roarm_cube_push_env._pre_physics_step override path directly writes robot_dof_targets and zeroes lead-limit rate",
        },
        "joint_or_drive_saturation": {
            "prediction": "possible if applied torque sits at effort limit or final commanded-actual joint error remains high",
            "judgement": bool((torque_sat is not None and float(torque_sat) >= 0.4) or float(b.get("final_joint_err_rad", 0.0)) > 0.05),
            "evidence": {
                "torque_saturation_rate_max": torque_sat,
                "final_joint_err_rad": b.get("final_joint_err_rad"),
            },
        },
        "step_clip_budget": {
            "prediction": "unlikely as a 0.010rad env step-clip issue because D325 uses a custom IK-to-external-target loop",
            "judgement": False,
            "evidence": {
                "max_step_command_delta_rad": b.get("max_step_command_delta_rad"),
                "note": "D325 loop does not call env candidate6_diffik_step_clip_rad; IK solver has its own 4deg clip.",
            },
        },
    }
    if questions["time_shortage"]["judgement"]:
        repair = "approach_hold_steps_x3"
    elif questions["joint_or_drive_saturation"]["judgement"]:
        repair = "arm_effort_limit_2p5_to_8p0"
    else:
        repair = "approach_hold_steps_x3_no_better_single_candidate"
    return questions, repair


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "g0a_d326_execution_contract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    final_rows = summary.get("final_retest", {}).get("rows", [])
    if final_rows:
        csv_path = out_dir / "g0a_d326_final_retest_trials.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)
        summary["final_retest"]["trial_csv"] = _rel(csv_path)
        (out_dir / "g0a_d326_execution_contract_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    lines = [
        "# D326 G0a Execution Contract Probe",
        "",
        "이번 case의 신규 변수: `[]` — D326 is G0a execution-contract repair only.",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- teleport pass: `{summary['teleport_check']['pass']}`",
        f"- selected repair: `{summary.get('selected_repair', '')}`",
        f"- final pass_all: `{summary.get('final_retest', {}).get('pass_all_count', 0)}/10`",
        "",
        "## Diagnostic Questions",
        "",
    ]
    for name, item in summary.get("diagnostic_questions", {}).items():
        lines.append(f"- {name}: prediction={item['prediction']}; judgement=`{item['judgement']}`; evidence={item['evidence']}")
    if final_rows:
        lines.extend(
            [
                "",
                "## Final 10-Trial Table",
                "",
                "| trial | pos mm | cmd pos mm | tangent deg | gap mm | top clearance mm | cube disp mm | pass |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in final_rows:
            lines.append(
                "| {trial} | {tcp_pose_error_mm:.3f} | {commanded_tcp_pose_error_mm:.3f} | "
                "{jaw_tangent_error_deg:.3f} | {fixed_jaw_face_gap_mm:.3f} | "
                "{contact_point_below_top_mm:.3f} | {cube_disp_xy_mm:.3f} | {pass_all} |".format(**row)
            )
    lines.extend(["", "## Artifacts", ""])
    for key in ("teleport_check", "baseline_diagnostic", "x3_diagnostic", "final_retest"):
        item = summary.get(key, {})
        if item.get("snapshot"):
            lines.append(f"- {key} snapshot: `{item['snapshot']}`")
        for snap in item.get("snapshots", []):
            lines.append(f"- {key} trial {snap['trial']} snapshot: `{snap['path']}`")
        if item.get("rrd_path"):
            lines.append(f"- {key} rrd: `{item['rrd_path']}`")
    (out_dir / "g0a_d326_execution_contract_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=32601)
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
    parser.add_argument("--episode_length_s", type=float, default=16.5)
    parser.add_argument("--live_viewer", action="store_true")
    args = parser.parse_args()

    if not args.robot_usd_path.exists():
        raise FileNotFoundError(args.robot_usd_path)
    if not args.urdf_path.exists():
        raise FileNotFoundError(args.urdf_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_manifest_path = REPO / "claudedocs/env_manifest_isaaclab_d326.txt"
    env_versions = _version_manifest()
    env_versions["robot_usd_path"] = _rel(args.robot_usd_path)
    env_versions["urdf_path"] = _rel(args.urdf_path)
    env_versions["env_manifest_path"] = _rel(env_manifest_path)

    teleport = _teleport_check(args)
    if not teleport["pass"]:
        summary = {
            "artifact": "d326_g0a_execution_contract_probe",
            "verdict": "D326_G0A_TELEPORT_STATIC_FAIL_STOP",
            "new_variable": "none; D326 is G0a execution-contract repair",
            "active_case": "G0a",
            "environment": env_versions,
            "teleport_check": teleport,
            "v2_gate": {
                "completed": False,
                "reason": "Teleport static check failed before a motion trial; v2 URDF/static RRD was still written for the stop state.",
                "rrd_path": teleport.get("rrd_path", ""),
                "rrd_status": teleport.get("rrd_status", {}),
            },
            "non_goals": ["no G0b", "no cylinder", "no gripper close", "no lift", "no RL/PPO", "no B200/RoArm/VLA"],
        }
        _write_outputs(args.out_dir, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        sim_app.close()
        return 0

    baseline = _run_motion(
        args,
        num_envs=1,
        approach_steps=int(args.approach_steps),
        hold_steps=int(args.hold_steps),
        trace=True,
        label="baseline",
    )
    x3 = _run_motion(
        args,
        num_envs=1,
        approach_steps=int(args.approach_steps) * 3,
        hold_steps=int(args.hold_steps) * 3,
        trace=False,
        label="x3_diagnostic",
    )
    questions, selected_repair = _diagnose_and_choose_repair(baseline, x3)

    if selected_repair == "arm_effort_limit_2p5_to_8p0":
        final = _run_motion(
            args,
            num_envs=10,
            approach_steps=int(args.approach_steps),
            hold_steps=int(args.hold_steps),
            trace=True,
            label="final_retest",
            arm_effort_limit=8.0,
        )
        repair_contract = {"type": selected_repair, "arm_effort_limit_sim": 8.0}
    else:
        final = _run_motion(
            args,
            num_envs=10,
            approach_steps=int(args.approach_steps) * 3,
            hold_steps=int(args.hold_steps) * 3,
            trace=True,
            label="final_retest",
        )
        repair_contract = {
            "type": "approach_hold_steps_x3",
            "approach_steps": int(args.approach_steps) * 3,
            "hold_steps": int(args.hold_steps) * 3,
        }

    pass_all_count = int(final["pass_all_count"])
    failure_counts = final["failure_counts"]
    if pass_all_count == 10:
        verdict = "D326_G0A_EXECUTION_CONTRACT_PASS"
    elif any(int(v) >= 3 for v in failure_counts.values()):
        verdict = "D326_G0A_EXECUTION_CONTRACT_FAIL"
    else:
        verdict = "D326_G0A_EXECUTION_CONTRACT_PARTIAL"

    summary = {
        "artifact": "d326_g0a_execution_contract_probe",
        "verdict": verdict,
        "new_variable": "none; D326 is G0a execution-contract repair",
        "active_case": "G0a",
        "failure_rule": "failure if any D325 criterion misses in >=3/10 trials; pass requires 10/10 all criteria",
        "environment": env_versions,
        "teleport_check": teleport,
        "baseline_diagnostic": baseline,
        "x3_diagnostic": x3,
        "diagnostic_questions": questions,
        "selected_repair": selected_repair,
        "repair_contract": repair_contract,
        "final_retest": final,
        "non_goals": [
            "no pose-family change",
            "no 42mm/10mm/15deg/15mm tuning",
            "no second simultaneous contract repair",
            "no G0b/cylinder",
            "no gripper close",
            "no grasp/lift",
            "no RL/PPO",
            "no B200/RoArm/VLA",
        ],
    }
    _write_outputs(args.out_dir, summary)
    print(
        "[d326-g0a] "
        f"verdict={verdict} teleport={teleport['pass']} "
        f"repair={selected_repair} final_pass={pass_all_count}/10 "
        f"out_dir={_rel(args.out_dir)}"
    )
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
