"""Scripted positive-control sanity for the default-off 10cm tap env.

This is a tiny local IsaacLab runtime check. It is not PPO, dataset generation,
robot control, or action-teacher construction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_sanity.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_sanity_summary.out"
ENV_ID = "RoArm-CubeTap10cm-Direct-v0"
PROJECT_TABLE_Z = -0.012117


def _table_z_flat_terrain(difficulty: float, cfg: Any) -> tuple[list[Any], np.ndarray]:
    """Generate a local flat mesh at the project table height."""
    from isaaclab.terrains.trimesh.utils import make_plane

    plane_mesh = make_plane(cfg.size, PROJECT_TABLE_Z, center_zero=False)
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, PROJECT_TABLE_Z)
    return [plane_mesh], np.array(origin)


def _scalar(value: Any) -> float | int | str:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def _tensor_mean(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _update_trace_stats(stats: dict[str, dict[str, float]], key: str, value: Any) -> None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return
    entry = stats.setdefault(key, {"min": scalar, "max": scalar, "final": scalar})
    entry["min"] = min(entry["min"], scalar)
    entry["max"] = max(entry["max"], scalar)
    entry["final"] = scalar


def _tensor_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    return [float(value)]


def _bool_tensor_list(value: Any) -> list[bool]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        return [bool(item) for item in value.tolist()]
    return [bool(value)]


def _face_metrics_torch(point_w: Any, cube_w: Any, push_dir_xy: Any, cfg: Any, torch_mod: Any) -> dict[str, Any]:
    half_xy = torch_mod.tensor(
        [float(cfg.cube_size_x_m) * 0.5, float(cfg.cube_size_y_m) * 0.5],
        dtype=torch_mod.float32,
        device=point_w.device,
    )
    half_along = torch_mod.sum(torch_mod.abs(push_dir_xy) * half_xy.unsqueeze(0), dim=-1)
    rel_xy = point_w[:, 0:2] - cube_w[:, 0:2]
    along = torch_mod.sum(rel_xy * push_dir_xy, dim=-1)
    lateral = torch_mod.norm(rel_xy - along.unsqueeze(-1) * push_dir_xy, p=2, dim=-1)
    face_gap = along + half_along
    vertical = torch_mod.abs(point_w[:, 2] - cube_w[:, 2])
    inside = (
        (torch_mod.abs(face_gap) <= float(cfg.tap_contact_face_band_m))
        & (lateral <= half_along + float(cfg.tap_contact_lateral_margin_m))
        & (vertical <= float(cfg.cube_size_z_m) * 0.5 + float(cfg.tap_contact_vertical_margin_m))
    )
    return {
        "face_gap_m": face_gap,
        "lateral_m": lateral,
        "vertical_offset_m": vertical,
        "inside_contact_band": inside,
    }


def _reach_trace_from_metrics(metrics: dict[str, float], cfg: Any, args: argparse.Namespace) -> list[dict[str, Any]] | None:
    rows = metrics.pop("_reach_trace_rows", None)
    if rows is None or args.reach_trace_json is None:
        return None
    return rows


def _closed_loop_ik_joint_target(
    inner: Any, cfg: Any, args: argparse.Namespace, step: int, torch_mod: Any
) -> tuple[Any, dict[str, float]]:
    from sim_scripts.roarm_kinematics import fk_tcp, ik_dls

    inner._compute_intermediate_values()
    cube_local = (inner._cube_start_w - inner.scene.env_origins).detach().cpu().numpy()
    push_dir = inner._push_dir_xy.detach().cpu().numpy()
    current_q_rad = inner._robot.data.joint_pos.detach().cpu().numpy()
    current_tcp_local = (inner._tcp_pos_w - inner.scene.env_origins).detach().cpu().numpy()
    half_xy = np.asarray([float(cfg.cube_size_x_m) * 0.5, float(cfg.cube_size_y_m) * 0.5], dtype=np.float64)
    alpha = min(1.0, max(0.0, float(step + 1) / max(float(args.closed_loop_push_steps), 1.0)))
    joint_targets = np.zeros((int(args.num_envs), int(cfg.action_space)), dtype=np.float32)
    ok_count = 0
    err_values: list[float] = []
    target_face_gap_values: list[float] = []
    target_lateral_values: list[float] = []
    target_vertical_values: list[float] = []
    target_inside_values: list[float] = []
    target_fk_err_values: list[float] = []
    actual_fk_sim_tcp_err_values: list[float] = []
    target_delta_abs_max_values: list[float] = []
    reach_rows: list[dict[str, Any]] = []
    for env_id in range(int(args.num_envs)):
        half_along = float(np.sum(np.abs(push_dir[env_id, :2]) * half_xy))
        pre = cube_local[env_id].copy()
        through = cube_local[env_id].copy()
        pre[:2] -= push_dir[env_id] * (half_along + float(args.precontact_clearance_m))
        if args.target_path_mode == "near_face_goal":
            through[:2] += push_dir[env_id] * (float(args.goal_push_m) - half_along)
        else:
            through[:2] += push_dir[env_id] * (half_along + float(args.goal_push_m))
        side_center_z = cube_local[env_id, 2] + float(cfg.cube_size_z_m) * 0.5 + float(args.tcp_top_margin_m)
        pre[2] = side_center_z
        through[2] = side_center_z
        tcp_target = pre + alpha * (through - pre)
        q_seed_deg = np.degrees(current_q_rad[env_id])
        q_deg, converged, err_mm, _iters = ik_dls(
            tcp_target,
            q_seed_deg,
            max_iter=int(args.closed_loop_ik_max_iter),
            tol_mm=float(args.closed_loop_ik_tol_mm),
        )
        q_deg[5] = 0.0
        target_rad = np.radians(q_deg)
        target_rad[5] = 0.0
        joint_targets[env_id] = target_rad.astype(np.float32)
        ok_count += int(bool(converged))
        err_values.append(float(err_mm))
        rel_xy = tcp_target[:2] - cube_local[env_id, :2]
        target_along = float(np.dot(rel_xy, push_dir[env_id, :2]))
        target_lateral_vec = rel_xy - target_along * push_dir[env_id, :2]
        target_face_gap = target_along + half_along
        target_lateral = float(np.linalg.norm(target_lateral_vec))
        target_vertical = float(abs(tcp_target[2] - cube_local[env_id, 2]))
        target_inside = (
            abs(target_face_gap) <= float(cfg.tap_contact_face_band_m)
            and target_lateral <= half_along + float(cfg.tap_contact_lateral_margin_m)
            and target_vertical <= float(cfg.cube_size_z_m) * 0.5 + float(cfg.tap_contact_vertical_margin_m)
        )
        target_fk = fk_tcp(q_deg)
        actual_fk = fk_tcp(q_seed_deg)
        applied_rel_xy = target_fk[:2] - cube_local[env_id, :2]
        applied_along = float(np.dot(applied_rel_xy, push_dir[env_id, :2]))
        applied_lateral_vec = applied_rel_xy - applied_along * push_dir[env_id, :2]
        applied_face_gap = applied_along + half_along
        applied_lateral = float(np.linalg.norm(applied_lateral_vec))
        applied_vertical = float(abs(target_fk[2] - cube_local[env_id, 2]))
        applied_inside = (
            abs(applied_face_gap) <= float(cfg.tap_contact_face_band_m)
            and applied_lateral <= half_along + float(cfg.tap_contact_lateral_margin_m)
            and applied_vertical <= float(cfg.cube_size_z_m) * 0.5 + float(cfg.tap_contact_vertical_margin_m)
        )
        target_face_gap_values.append(float(target_face_gap))
        target_lateral_values.append(target_lateral)
        target_vertical_values.append(target_vertical)
        target_inside_values.append(float(target_inside))
        target_fk_err_values.append(float(np.linalg.norm(target_fk - tcp_target) * 1000.0))
        actual_fk_sim_tcp_err_values.append(float(np.linalg.norm(actual_fk - current_tcp_local[env_id]) * 1000.0))
        target_delta_abs_max_values.append(float(np.max(np.abs(target_rad - current_q_rad[env_id]))))
        if args.reach_trace_json is not None:
            reach_rows.append(
                {
                    "env_id": env_id,
                    "command_target_face_gap_m": float(target_face_gap),
                    "command_target_lateral_m": target_lateral,
                    "command_target_vertical_offset_m": target_vertical,
                    "command_target_inside_contact_band": bool(target_inside),
                    "applied_joint_target_fk_face_gap_m": float(applied_face_gap),
                    "applied_joint_target_fk_lateral_m": applied_lateral,
                    "applied_joint_target_fk_vertical_offset_m": applied_vertical,
                    "applied_joint_target_fk_inside_contact_band": bool(applied_inside),
                    "applied_joint_target_fk_err_mm": float(np.linalg.norm(target_fk - tcp_target) * 1000.0),
                    "joint_target_delta_abs_max_rad": float(np.max(np.abs(target_rad - current_q_rad[env_id]))),
                }
            )
    target_t = torch_mod.tensor(joint_targets, dtype=torch_mod.float32, device=inner.device)
    metrics = {
        "closed_loop_ik_ok_rate": float(ok_count) / max(float(args.num_envs), 1.0),
        "closed_loop_ik_err_mm_mean": float(np.mean(err_values)) if err_values else float("nan"),
        "closed_loop_alpha": alpha,
        "closed_loop_target_face_gap_m_mean": float(np.mean(target_face_gap_values))
        if target_face_gap_values
        else float("nan"),
        "closed_loop_target_face_gap_m_min": float(np.min(target_face_gap_values))
        if target_face_gap_values
        else float("nan"),
        "closed_loop_target_face_gap_m_max": float(np.max(target_face_gap_values))
        if target_face_gap_values
        else float("nan"),
        "closed_loop_target_lateral_m_mean": float(np.mean(target_lateral_values))
        if target_lateral_values
        else float("nan"),
        "closed_loop_target_vertical_offset_m_mean": float(np.mean(target_vertical_values))
        if target_vertical_values
        else float("nan"),
        "closed_loop_target_inside_contact_band_rate": float(np.mean(target_inside_values))
        if target_inside_values
        else float("nan"),
        "closed_loop_target_fk_err_mm_mean": float(np.mean(target_fk_err_values))
        if target_fk_err_values
        else float("nan"),
        "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean": float(np.mean(actual_fk_sim_tcp_err_values))
        if actual_fk_sim_tcp_err_values
        else float("nan"),
        "closed_loop_target_delta_from_actual_abs_max_rad_mean": float(np.mean(target_delta_abs_max_values))
        if target_delta_abs_max_values
        else float("nan"),
        "closed_loop_target_delta_from_actual_abs_max_rad_max": float(np.max(target_delta_abs_max_values))
        if target_delta_abs_max_values
        else float("nan"),
    }
    if args.reach_trace_json is not None:
        metrics["_reach_trace_rows"] = reach_rows
    return target_t, metrics


def _init_builtin_diffik_state(inner: Any, args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_rotate, subtract_frame_transforms

    arm_joint_ids, _arm_joint_names = inner._robot.find_joints(
        [
            "base_link_to_link1",
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
        ],
        preserve_order=True,
    )
    link5_body_idx = inner.link5_idx
    jacobi_body_idx = link5_body_idx - 1 if inner._robot.is_fixed_base else link5_body_idx
    jacobi_joint_ids = arm_joint_ids if inner._robot.is_fixed_base else [idx + 6 for idx in arm_joint_ids]
    diffik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": float(args.builtin_diffik_lambda)},
    )
    return {
        "diffik": DifferentialIKController(diffik_cfg, num_envs=inner.num_envs, device=inner.device),
        "arm_joint_ids": arm_joint_ids,
        "jacobi_body_idx": jacobi_body_idx,
        "jacobi_joint_ids": jacobi_joint_ids,
        "link5_body_idx": link5_body_idx,
        "matrix_from_quat": matrix_from_quat,
        "quat_inv": quat_inv,
        "quat_rotate": quat_rotate,
        "subtract_frame_transforms": subtract_frame_transforms,
        "tool_proxy_local": inner._tcp_local.unsqueeze(0).repeat(inner.num_envs, 1),
    }


def _closed_loop_builtin_diffik_joint_target(
    inner: Any,
    cfg: Any,
    args: argparse.Namespace,
    step: int,
    torch_mod: Any,
    state: dict[str, Any],
    step_clip_rad: float | None = None,
) -> tuple[Any, dict[str, float]]:
    from sim_scripts.roarm_kinematics import fk_tcp

    inner._compute_intermediate_values()
    cube_w = inner._cube_start_w
    push_dir = inner._push_dir_xy
    half_xy = torch_mod.tensor(
        [float(cfg.cube_size_x_m) * 0.5, float(cfg.cube_size_y_m) * 0.5],
        dtype=torch_mod.float32,
        device=inner.device,
    )
    half_along = torch_mod.sum(torch_mod.abs(push_dir) * half_xy.unsqueeze(0), dim=-1)
    alpha = min(1.0, max(0.0, float(step + 1) / max(float(args.closed_loop_push_steps), 1.0)))
    pre_w = cube_w.clone()
    through_w = cube_w.clone()
    pre_w[:, 0:2] = cube_w[:, 0:2] - push_dir * (half_along + float(args.precontact_clearance_m)).unsqueeze(-1)
    if args.target_path_mode == "near_face_goal":
        through_w[:, 0:2] = cube_w[:, 0:2] + push_dir * (float(args.goal_push_m) - half_along).unsqueeze(-1)
    else:
        through_w[:, 0:2] = cube_w[:, 0:2] + push_dir * (half_along + float(args.goal_push_m)).unsqueeze(-1)
    side_center_z = cube_w[:, 2] + float(cfg.cube_size_z_m) * 0.5 + float(args.tcp_top_margin_m)
    pre_w[:, 2] = side_center_z
    through_w[:, 2] = side_center_z
    tcp_target_w = pre_w + float(alpha) * (through_w - pre_w)

    root_pos_w = inner._robot.data.root_pos_w
    root_quat_w = inner._robot.data.root_quat_w
    link5_body_idx = state["link5_body_idx"]
    link5_pos_w = inner._robot.data.body_pos_w[:, link5_body_idx].clone()
    link5_quat_w = inner._robot.data.body_quat_w[:, link5_body_idx].clone()
    subtract_frame_transforms = state["subtract_frame_transforms"]
    quat_rotate = state["quat_rotate"]
    matrix_from_quat = state["matrix_from_quat"]
    quat_inv = state["quat_inv"]

    link5_pos_b, link5_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, link5_pos_w, link5_quat_w)
    tool_proxy_local = state["tool_proxy_local"]
    tool_proxy_offset_w = quat_rotate(link5_quat_w, tool_proxy_local)
    link5_target_w = tcp_target_w - tool_proxy_offset_w
    link5_target_b, _link5_quat_target_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, link5_target_w, link5_quat_w
    )

    jacobian = inner._robot.root_physx_view.get_jacobians()[:, state["jacobi_body_idx"], :, state["jacobi_joint_ids"]]
    base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian = jacobian.clone()
    jacobian[:, :3, :] = torch_mod.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch_mod.bmm(base_rot_matrix, jacobian[:, 3:, :])
    arm_joint_ids = state["arm_joint_ids"]
    joint_pos_arm = inner._robot.data.joint_pos[:, arm_joint_ids]

    diffik = state["diffik"]
    diffik.set_command(link5_target_b, ee_pos=link5_pos_b, ee_quat=link5_quat_b)
    joint_pos_des = diffik.compute(link5_pos_b, link5_quat_b, jacobian, joint_pos_arm)
    numeric_ok = torch_mod.isfinite(joint_pos_des).all(dim=-1)
    raw_delta_arm = joint_pos_des - joint_pos_arm
    if step_clip_rad is not None and float(step_clip_rad) > 0.0:
        clipped_delta_arm = torch_mod.clamp(raw_delta_arm, -float(step_clip_rad), float(step_clip_rad))
        arm_joint_target = joint_pos_arm + clipped_delta_arm
        step_clip_rate = float(
            (torch_mod.abs(raw_delta_arm) >= float(step_clip_rad) - 1.0e-9).float().mean().item()
        )
        step_clip_value = float(step_clip_rad)
    else:
        clipped_delta_arm = raw_delta_arm
        arm_joint_target = joint_pos_des
        step_clip_rate = 0.0
        step_clip_value = 0.0

    target_full = inner._robot.data.joint_pos.detach().clone()
    target_full[:, arm_joint_ids] = arm_joint_target
    target_full[:, inner.gripper_joint_idx] = 0.0
    target_full = torch_mod.clamp(target_full, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
    target_full[:, inner.gripper_joint_idx] = 0.0

    rel_xy = tcp_target_w[:, 0:2] - cube_w[:, 0:2]
    target_along = torch_mod.sum(rel_xy * push_dir, dim=-1)
    target_lateral = torch_mod.norm(rel_xy - target_along.unsqueeze(-1) * push_dir, p=2, dim=-1)
    target_face_gap = target_along + half_along
    target_vertical = torch_mod.abs(tcp_target_w[:, 2] - cube_w[:, 2])
    target_inside = (
        (torch_mod.abs(target_face_gap) <= float(cfg.tap_contact_face_band_m))
        & (target_lateral <= half_along + float(cfg.tap_contact_lateral_margin_m))
        & (target_vertical <= float(cfg.cube_size_z_m) * 0.5 + float(cfg.tap_contact_vertical_margin_m))
    )
    tool_proxy_pos_before_w = link5_pos_w + tool_proxy_offset_w
    target_tcp_err_before_m = torch_mod.norm(tool_proxy_pos_before_w - tcp_target_w, p=2, dim=-1)
    actual_fk_sim_tcp_err_mm = torch_mod.norm(tool_proxy_pos_before_w - inner._tcp_pos_w, p=2, dim=-1) * 1000.0
    target_delta_abs = torch_mod.abs(target_full - inner._robot.data.joint_pos)
    applied_target_fk_err_mm_mean = float("nan")
    reach_rows: list[dict[str, Any]] = []
    if args.reach_trace_json is not None:
        target_full_np = target_full.detach().cpu().numpy()
        cube_w_np = cube_w.detach().cpu().numpy()
        origins_np = inner.scene.env_origins.detach().cpu().numpy()
        push_np = push_dir.detach().cpu().numpy()
        half_along_np = half_along.detach().cpu().numpy()
        applied_fk_err_values: list[float] = []
        command_face = _tensor_list(target_face_gap)
        command_lateral = _tensor_list(target_lateral)
        command_vertical = _tensor_list(target_vertical)
        command_inside = _bool_tensor_list(target_inside)
        target_delta_env = _tensor_list(target_delta_abs.max(dim=-1).values)
        for env_id in range(int(args.num_envs)):
            applied_tcp_local = fk_tcp(np.degrees(target_full_np[env_id]))
            applied_tcp_w = applied_tcp_local + origins_np[env_id]
            rel_xy = applied_tcp_w[:2] - cube_w_np[env_id, :2]
            applied_along = float(np.dot(rel_xy, push_np[env_id, :2]))
            applied_lateral_vec = rel_xy - applied_along * push_np[env_id, :2]
            applied_face_gap = applied_along + float(half_along_np[env_id])
            applied_lateral = float(np.linalg.norm(applied_lateral_vec))
            applied_vertical = float(abs(applied_tcp_w[2] - cube_w_np[env_id, 2]))
            applied_inside = (
                abs(applied_face_gap) <= float(cfg.tap_contact_face_band_m)
                and applied_lateral <= float(half_along_np[env_id]) + float(cfg.tap_contact_lateral_margin_m)
                and applied_vertical <= float(cfg.cube_size_z_m) * 0.5 + float(cfg.tap_contact_vertical_margin_m)
            )
            applied_fk_err_mm = float(np.linalg.norm(applied_tcp_w - tcp_target_w.detach().cpu().numpy()[env_id]) * 1000.0)
            applied_fk_err_values.append(applied_fk_err_mm)
            reach_rows.append(
                {
                    "env_id": env_id,
                    "command_target_face_gap_m": command_face[env_id],
                    "command_target_lateral_m": command_lateral[env_id],
                    "command_target_vertical_offset_m": command_vertical[env_id],
                    "command_target_inside_contact_band": command_inside[env_id],
                    "applied_joint_target_fk_face_gap_m": float(applied_face_gap),
                    "applied_joint_target_fk_lateral_m": applied_lateral,
                    "applied_joint_target_fk_vertical_offset_m": applied_vertical,
                    "applied_joint_target_fk_inside_contact_band": bool(applied_inside),
                    "applied_joint_target_fk_err_mm": applied_fk_err_mm,
                    "joint_target_delta_abs_max_rad": target_delta_env[env_id],
                }
            )
        applied_target_fk_err_mm_mean = float(np.mean(applied_fk_err_values)) if applied_fk_err_values else float("nan")

    metrics = {
        "closed_loop_ik_ok_rate": float(numeric_ok.float().mean().item()),
        "builtin_diffik_numeric_ok_rate": float(numeric_ok.float().mean().item()),
        "builtin_diffik_live_jacobian": 1.0,
        "builtin_diffik_tool_proxy_offset": 1.0,
        "builtin_diffik_step_clipped_target_apply": 1.0 if step_clip_rad is not None else 0.0,
        "builtin_diffik_step_clip_rad": step_clip_value,
        "builtin_diffik_step_clip_rate": step_clip_rate,
        "builtin_diffik_raw_delta_abs_max_rad": float(torch_mod.abs(raw_delta_arm).max().item()),
        "builtin_diffik_clipped_delta_abs_max_rad": float(torch_mod.abs(clipped_delta_arm).max().item()),
        "closed_loop_alpha": float(alpha),
        "closed_loop_target_face_gap_m_mean": float(target_face_gap.mean().item()),
        "closed_loop_target_face_gap_m_min": float(target_face_gap.min().item()),
        "closed_loop_target_face_gap_m_max": float(target_face_gap.max().item()),
        "closed_loop_target_lateral_m_mean": float(target_lateral.mean().item()),
        "closed_loop_target_vertical_offset_m_mean": float(target_vertical.mean().item()),
        "closed_loop_target_inside_contact_band_rate": float(target_inside.float().mean().item()),
        "closed_loop_target_fk_err_mm_mean": applied_target_fk_err_mm_mean,
        "builtin_diffik_target_tcp_err_before_m_mean": float(target_tcp_err_before_m.mean().item()),
        "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean": float(actual_fk_sim_tcp_err_mm.mean().item()),
        "closed_loop_target_delta_from_actual_abs_max_rad_mean": float(target_delta_abs.max(dim=-1).values.mean().item()),
        "closed_loop_target_delta_from_actual_abs_max_rad_max": float(target_delta_abs.max().item()),
    }
    if args.reach_trace_json is not None:
        metrics["_reach_trace_rows"] = reach_rows
    return target_full, metrics


def _closed_loop_ik_action(inner: Any, cfg: Any, args: argparse.Namespace, step: int, torch_mod: Any) -> tuple[Any, dict[str, float]]:
    target_t, metrics = _closed_loop_ik_joint_target(inner, cfg, args, step, torch_mod)
    target_base = inner.robot_dof_targets.detach()
    action_t = (target_t - target_base) / max(float(cfg.action_scale), 1.0e-6)
    action_t[:, inner.gripper_joint_idx] = 0.0
    action_t = torch_mod.clamp(action_t, -1.0, 1.0)
    return action_t, metrics


def _write_result(out_json: Path, out_summary: Path, result: dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_positive_control_sanity_v1 "
        f"status={result['status']} gpu_runtime={result.get('gpu_runtime', 'UNKNOWN')} "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 env_contract "
            f"env_id={result.get('env_id', ENV_ID)} device={result.get('device', 'UNKNOWN')} "
            f"cube_size_m={result.get('cube_size_m', 'UNKNOWN')} "
            f"cube_mass_kg={result.get('cube_mass_kg', 'UNKNOWN')} "
            f"final_1cm_required={result.get('final_1cm_required', 'UNKNOWN')} "
            f"episode_length_s={result.get('episode_length_s', 'UNKNOWN')} "
            f"env_max_episode_length={result.get('env_max_episode_length', 'UNKNOWN')}"
        ),
        (
            "line3 scripted_control "
            f"num_envs={result.get('num_envs', 'NA')} max_steps={result.get('max_steps', 'NA')} "
            f"steps_executed={result.get('steps_executed', 'NA')} "
            f"cube_xy=({result.get('fixed_cube_x_m', 'NA')},{result.get('fixed_cube_y_m', 'NA')}) "
            f"push_dir=({result.get('fixed_push_dir_x', 'NA')},{result.get('fixed_push_dir_y', 'NA')}) "
            f"controller_mode={result.get('controller_mode', 'NA')} "
            f"direct_ik_joint_target_apply={result.get('direct_ik_joint_target_apply', 'NA')} "
            f"target_path_mode={result.get('target_path_mode', 'NA')} "
            f"precontact_clearance_m={result.get('precontact_clearance_m', 'NA')} "
            f"tcp_top_margin_m={result.get('tcp_top_margin_m', 'NA')} "
            f"goal_push_m={result.get('goal_push_m', 'NA')} "
            f"max_joint_delta_per_step_rad={result.get('max_joint_delta_per_step_rad', 'NA')} "
            f"joint_target_lead_limit_rad={result.get('joint_target_lead_limit_rad', 'NA')}"
        ),
        (
            "line4 reset_and_ik "
            f"ik_endpoint_reset_rate={result.get('reset_metrics', {}).get('ik_endpoint_reset_rate', 'NA')} "
            f"ik_reset_err_mm={result.get('reset_metrics', {}).get('ik_reset_err_mm', 'NA')} "
            f"teacher_goal_ok_rate={result.get('reset_metrics', {}).get('teacher_goal_ok_rate', 'NA')} "
            f"controller_goal_ok_rate={result.get('controller_goal_ok_rate', 'NA')} "
            f"initial_face_gap_m={result.get('reset_metrics', {}).get('initial_face_gap_m', 'NA')} "
            f"initial_vertical_offset_m={result.get('reset_metrics', {}).get('initial_vertical_offset_m', 'NA')} "
            f"closed_loop_ik_ok_rate={result.get('controller_metrics', {}).get('closed_loop_ik_ok_rate', 'NA')}"
        ),
        (
            "line5 tap_logs "
            f"required_log_keys_present={result.get('required_log_keys_present', 'NA')} "
            f"contact_seen={result.get('last_log', {}).get('cube_tap_contact_seen_rate', 'NA')} "
            f"reaction_signal_now={result.get('last_log', {}).get('cube_tap_reaction_signal_now_rate', 'NA')} "
            f"reaction_contact_context={result.get('last_log', {}).get('cube_tap_reaction_contact_context_rate', 'NA')} "
            f"reaction_seen={result.get('last_log', {}).get('cube_tap_reaction_seen_rate', 'NA')} "
            f"professor_physical_reaction_seen={result.get('last_log', {}).get('cube_tap_professor_physical_reaction_seen_rate', 'NA')} "
            f"overshoot_seen={result.get('last_log', {}).get('cube_tap_overshoot_seen_rate', 'NA')} "
            f"tap_success={result.get('last_log', {}).get('cube_tap_success_rate', 'NA')}"
        ),
        (
            "line6 reaction_metrics "
            f"max_disp_along_m={result.get('last_log', {}).get('cube_tap_max_disp_along_m', 'NA')} "
            f"max_z_delta_m={result.get('last_log', {}).get('cube_tap_max_z_delta_m', 'NA')} "
            f"max_speed_mps={result.get('last_log', {}).get('cube_tap_max_speed_mps', 'NA')} "
            f"terminated_count={result.get('terminated_count', 'NA')} "
            f"truncated_count={result.get('truncated_count', 'NA')}"
        ),
        (
            "line7 action_path "
            f"tcp_cube_dist_m={result.get('last_log', {}).get('cube_push_tcp_cube_dist_m', 'NA')} "
            f"joint_delta_abs_mean={result.get('last_log', {}).get('cube_push_joint_delta_abs_mean', 'NA')} "
            f"joint_delta_abs_max={result.get('last_log', {}).get('cube_push_joint_delta_abs_max', 'NA')} "
            f"joint_delta_cap_rate={result.get('last_log', {}).get('cube_push_joint_delta_cap_rate', 'NA')} "
            f"action_abs_mean={result.get('last_log', {}).get('cube_push_action_abs_mean', 'NA')} "
            f"action_abs_max={result.get('last_log', {}).get('cube_push_action_abs_max', 'NA')} "
            f"target_lead_abs_max={result.get('last_log', {}).get('cube_push_target_lead_abs_max', 'NA')} "
            f"target_lead_limit_rate={result.get('last_log', {}).get('cube_push_target_lead_limit_rate', 'NA')} "
            f"contact_slowdown_mean={result.get('last_log', {}).get('cube_push_contact_slowdown_mean', 'NA')} "
            f"teacher_blend_mean={result.get('last_log', {}).get('cube_push_teacher_blend_mean', 'NA')} "
            f"action_penalty={result.get('last_log', {}).get('action_penalty', 'NA')}"
        ),
        (
            "line8 trace_diagnostics "
            f"face_gap_min={result.get('log_trace_stats', {}).get('cube_tap_contact_face_gap_m', {}).get('min', 'NA')} "
            f"face_gap_max={result.get('log_trace_stats', {}).get('cube_tap_contact_face_gap_m', {}).get('max', 'NA')} "
            f"face_gap_final={result.get('log_trace_stats', {}).get('cube_tap_contact_face_gap_m', {}).get('final', 'NA')} "
            f"shortfall_min={result.get('log_trace_stats', {}).get('cube_tap_contact_band_shortfall_m', {}).get('min', 'NA')} "
            f"shortfall_final={result.get('log_trace_stats', {}).get('cube_tap_contact_band_shortfall_m', {}).get('final', 'NA')} "
            f"tcp_dist_min={result.get('log_trace_stats', {}).get('cube_push_tcp_cube_dist_m', {}).get('min', 'NA')} "
            f"joint_delta_abs_max={result.get('log_trace_stats', {}).get('cube_push_joint_delta_abs_max', {}).get('max', 'NA')} "
            f"joint_delta_cap_rate_max={result.get('log_trace_stats', {}).get('cube_push_joint_delta_cap_rate', {}).get('max', 'NA')} "
            f"target_lead_limit_rate_max={result.get('log_trace_stats', {}).get('cube_push_target_lead_limit_rate', {}).get('max', 'NA')}"
        ),
        (
            "line9 verdict "
            f"professor_physical_reaction_evidence={result.get('professor_physical_reaction_evidence', 'UNKNOWN')} "
            f"rl_contact_gated_positive_control={result.get('rl_contact_gated_positive_control', result.get('positive_control', 'UNKNOWN'))} "
            f"blocker={result.get('blocker', 'NONE')} "
            "professor_objective_metadata=SEPARATE_FROM_DATASET_RL "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
        (
            "line10 controller_telemetry "
            f"target_face_gap_final={result.get('controller_trace_stats', {}).get('closed_loop_target_face_gap_m_mean', {}).get('final', 'NA')} "
            f"target_inside_contact_band_final={result.get('controller_trace_stats', {}).get('closed_loop_target_inside_contact_band_rate', {}).get('final', 'NA')} "
            f"target_fk_err_mm_final={result.get('controller_trace_stats', {}).get('closed_loop_target_fk_err_mm_mean', {}).get('final', 'NA')} "
            f"actual_fk_vs_sim_tcp_err_mm_final={result.get('controller_trace_stats', {}).get('closed_loop_actual_fk_vs_sim_tcp_err_mm_mean', {}).get('final', 'NA')} "
            f"target_delta_abs_max_final={result.get('controller_trace_stats', {}).get('closed_loop_target_delta_from_actual_abs_max_rad_max', {}).get('final', 'NA')} "
            f"direct_joint_follow_abs_max_final={result.get('controller_trace_stats', {}).get('direct_joint_follow_abs_max_rad', {}).get('final', 'NA')} "
            f"direct_actual_joint_step_abs_max_final={result.get('controller_trace_stats', {}).get('direct_actual_joint_step_abs_max_rad', {}).get('final', 'NA')}"
        ),
    ]
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line, flush=True)


def _write_reach_trace(out_path: Path, result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys = (
        "artifact_type",
        "branch",
        "env_id",
        "num_envs",
        "max_steps",
        "steps_executed",
        "seed",
        "device",
        "cube_size_m",
        "cube_mass_kg",
        "episode_length_s",
        "env_max_episode_length",
        "controller_mode",
        "closed_loop_push_steps",
        "builtin_diffik_step_clip_rad",
        "direct_ik_joint_target_apply",
        "isaac_builtin_diffik_controller_apply",
        "builtin_diffik_step_clipped_target_apply",
        "rl_contact_gated_positive_control",
        "professor_physical_reaction_evidence",
    )
    artifact = {
        "artifact_type": "cube10cm_tap_rl_per_step_reach_trace_v1",
        "local_gpu_runtime_telemetry": True,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "action_teacher_dataset": False,
        "metadata": {key: result.get(key) for key in metadata_keys if key in result},
        "schema": [
            "step",
            "env_id",
            "episode_length_s",
            "cube_pos_w_xyz",
            "push_dir_xy",
            "command_target_face_gap_m",
            "command_target_lateral_m",
            "command_target_vertical_offset_m",
            "command_target_inside_contact_band",
            "applied_joint_target_fk_face_gap_m",
            "applied_joint_target_fk_lateral_m",
            "applied_joint_target_fk_vertical_offset_m",
            "applied_joint_target_fk_inside_contact_band",
            "applied_joint_target_fk_err_mm",
            "actual_tcp_face_gap_m",
            "actual_tcp_lateral_m",
            "actual_tcp_vertical_offset_m",
            "actual_contact_proxy",
            "joint_target_delta_abs_max_rad",
            "direct_joint_follow_abs_max_rad",
            "actual_joint_step_abs_max_rad",
            "cube_disp_along_m",
            "cube_speed_mps",
            "professor_physical_reaction_now",
            "professor_physical_reaction_seen",
            "tap_success_now",
            "tap_success_seen",
            "terminated",
            "truncated",
        ],
        "rows": rows,
    }
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=962)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--fixed_cube_x_m", type=float, default=0.250)
    parser.add_argument("--fixed_cube_y_m", type=float, default=0.000)
    parser.add_argument("--fixed_push_dir_x", type=float, default=1.0)
    parser.add_argument("--fixed_push_dir_y", type=float, default=0.0)
    parser.add_argument("--precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--tcp_top_margin_m", type=float, default=-0.050)
    parser.add_argument("--goal_push_m", type=float, default=0.006)
    parser.add_argument(
        "--target_path_mode",
        choices=("legacy_far_face_through", "near_face_goal"),
        default="legacy_far_face_through",
    )
    parser.add_argument("--teacher_horizon_frac", type=float, default=1.0)
    parser.add_argument("--episode_length_s", type=float, default=-1.0)
    parser.add_argument(
        "--controller_mode",
        choices=(
            "builtin_teacher",
            "external_closed_loop",
            "external_closed_loop_direct_apply",
            "isaac_builtin_diffik_direct_apply",
            "isaac_builtin_diffik_step_clipped_direct_apply",
        ),
        default="builtin_teacher",
    )
    parser.add_argument("--closed_loop_push_steps", type=int, default=72)
    parser.add_argument("--closed_loop_ik_max_iter", type=int, default=80)
    parser.add_argument("--closed_loop_ik_tol_mm", type=float, default=1.5)
    parser.add_argument("--builtin_diffik_lambda", type=float, default=0.010)
    parser.add_argument("--builtin_diffik_step_clip_rad", type=float, default=0.010)
    parser.add_argument("--action_smoothing_alpha", type=float, default=-1.0)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=-1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=-1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=-1.0)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_LOCAL_USD)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--reach_trace_json", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    sim_app = None
    env = None
    started = time.time()
    try:
        if not args.robot_usd_path.exists():
            raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
        sim_app = app_launcher.app

        import gymnasium as gym
        import torch

        import roarm_rl  # noqa: F401 - registers envs lazily
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
        from roarm_rl.roarm_cube_push_env import CUBE10CM_MASS_KG, CUBE10CM_SIZE_M, RoArmCubeTap10cmEnvCfg
        from roarm_rl.roarm_stack_env import TABLE_Z

        if abs(float(TABLE_Z) - PROJECT_TABLE_Z) > 1.0e-12:
            raise AssertionError(f"table height mismatch: env={TABLE_Z} sanity={PROJECT_TABLE_Z}")

        flat_cfg = MeshPlaneTerrainCfg(proportion=1.0)
        flat_cfg.function = _table_z_flat_terrain
        cfg = RoArmCubeTap10cmEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.seed = int(args.seed)
        cfg.sim.device = str(args.device)
        if float(args.episode_length_s) > 0.0:
            cfg.episode_length_s = float(args.episode_length_s)
        cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(2.0, 2.0),
                num_rows=1,
                num_cols=1,
                border_width=0.0,
                sub_terrains={"flat": flat_cfg},
                use_cache=False,
            ),
            env_spacing=cfg.scene.env_spacing,
            physics_material=cfg.terrain.physics_material,
            visual_material=cfg.terrain.visual_material,
        )
        cfg.robot.spawn.usd_path = str(args.robot_usd_path)
        cfg.cube_x_min = float(args.fixed_cube_x_m)
        cfg.cube_x_max = float(args.fixed_cube_x_m)
        cfg.cube_y_min = float(args.fixed_cube_y_m)
        cfg.cube_y_max = float(args.fixed_cube_y_m)
        cfg.fixed_push_dir_x = float(args.fixed_push_dir_x)
        cfg.fixed_push_dir_y = float(args.fixed_push_dir_y)
        cfg.ik_endpoint_reset = True
        cfg.ik_reset_jitter_rad = 0.0
        cfg.ik_precontact_clearance_m = float(args.precontact_clearance_m)
        cfg.ik_tcp_top_margin_m = float(args.tcp_top_margin_m)
        cfg.scripted_teacher_blend = 1.0 if args.controller_mode == "builtin_teacher" else 0.0
        cfg.scripted_teacher_horizon_frac = float(args.teacher_horizon_frac)
        cfg.scripted_teacher_goal_push_m = float(args.goal_push_m)
        if float(args.action_smoothing_alpha) >= 0.0:
            cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
        if float(args.contact_joint_delta_scale) >= 0.0:
            cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
        if float(args.max_joint_delta_per_step_rad) >= 0.0:
            cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
        if float(args.joint_target_lead_limit_rad) >= 0.0:
            cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
        if args.num_envs < 8:
            cfg.scene.clone_in_fabric = False
            cfg.scene.replicate_physics = False

        mass_kg = float(cfg.sponge.spawn.mass_props.mass)
        contract_ok = (
            abs(float(cfg.cube_size_x_m) - 0.100) <= 1.0e-12
            and abs(float(cfg.cube_size_y_m) - 0.100) <= 1.0e-12
            and abs(float(cfg.cube_size_z_m) - 0.100) <= 1.0e-12
            and abs(mass_kg - 0.720) <= 1.0e-12
            and not bool(cfg.tap_final_relocation_required)
            and str(cfg.tap_objective_name) == "tap_reaction_contact_not_final_relocation"
            and abs(float(cfg.tap_reaction_disp_m) - 0.001) <= 1.0e-12
            and abs(float(cfg.tap_overshoot_disp_m) - 0.020) <= 1.0e-12
        )
        if not contract_ok:
            raise AssertionError("10cm tap env cfg contract mismatch before env creation")

        print(
            f"[tap10cm-positive] creating {ENV_ID} num_envs={args.num_envs} "
            f"cube=({args.fixed_cube_x_m:+.3f},{args.fixed_cube_y_m:+.3f})",
            flush=True,
        )
        env = gym.make(ENV_ID, cfg=cfg)
        inner = env.unwrapped
        obs, _info = env.reset()
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        expected_shape = (args.num_envs, cfg.observation_space)
        if tuple(obs_t.shape) != expected_shape:
            raise AssertionError(f"obs shape mismatch: expected={expected_shape} actual={tuple(obs_t.shape)}")

        inner._compute_intermediate_values()
        reset_terms = inner._tap_terms()
        reset_metrics = {
            "ik_endpoint_reset_rate": _tensor_mean(inner._ik_reset_ok),
            "ik_reset_err_mm": _tensor_mean(inner._ik_reset_err_mm),
            "teacher_goal_ok_rate": _tensor_mean(inner._teacher_goal_ok),
            "initial_face_gap_m": _tensor_mean(reset_terms["tap_contact_face_gap_m"]),
            "initial_vertical_offset_m": _tensor_mean(reset_terms["tap_contact_vertical_offset_m"]),
            "initial_contact_proxy_rate": _tensor_mean(reset_terms["tap_contact_proxy"]),
        }

        rewards_all: list[float] = []
        truncated_count = 0
        terminated_count = 0
        last_log: dict[str, Any] = {}
        steps_executed = 0
        controller_metrics: dict[str, float] = {}
        controller_trace_stats: dict[str, dict[str, float]] = {}
        log_trace_stats: dict[str, dict[str, float]] = {}
        reach_trace_rows: list[dict[str, Any]] = []
        zero_action = torch.zeros((args.num_envs, cfg.action_space), device=inner.device)
        builtin_diffik_mode = args.controller_mode in (
            "isaac_builtin_diffik_direct_apply",
            "isaac_builtin_diffik_step_clipped_direct_apply",
        )
        builtin_diffik_state = (
            _init_builtin_diffik_state(inner, args)
            if builtin_diffik_mode
            else None
        )
        for step in range(int(args.steps)):
            joint_target_for_step = None
            reach_trace_for_step = None
            joint_pos_before_step = inner._robot.data.joint_pos.detach().clone()
            if args.controller_mode == "external_closed_loop":
                action, controller_metrics = _closed_loop_ik_action(inner, cfg, args, step, torch)
                reach_trace_for_step = _reach_trace_from_metrics(controller_metrics, cfg, args)
            elif args.controller_mode == "external_closed_loop_direct_apply":
                joint_target, controller_metrics = _closed_loop_ik_joint_target(inner, cfg, args, step, torch)
                reach_trace_for_step = _reach_trace_from_metrics(controller_metrics, cfg, args)
                joint_target_for_step = joint_target.detach().clone()
                inner._external_joint_targets_override = joint_target
                action = zero_action
            elif args.controller_mode == "isaac_builtin_diffik_direct_apply":
                if builtin_diffik_state is None:
                    raise AssertionError("builtin DiffIK state was not initialized")
                joint_target, controller_metrics = _closed_loop_builtin_diffik_joint_target(
                    inner, cfg, args, step, torch, builtin_diffik_state
                )
                reach_trace_for_step = _reach_trace_from_metrics(controller_metrics, cfg, args)
                joint_target_for_step = joint_target.detach().clone()
                inner._external_joint_targets_override = joint_target
                action = zero_action
            elif args.controller_mode == "isaac_builtin_diffik_step_clipped_direct_apply":
                if builtin_diffik_state is None:
                    raise AssertionError("builtin DiffIK state was not initialized")
                joint_target, controller_metrics = _closed_loop_builtin_diffik_joint_target(
                    inner,
                    cfg,
                    args,
                    step,
                    torch,
                    builtin_diffik_state,
                    step_clip_rad=float(args.builtin_diffik_step_clip_rad),
                )
                reach_trace_for_step = _reach_trace_from_metrics(controller_metrics, cfg, args)
                joint_target_for_step = joint_target.detach().clone()
                inner._external_joint_targets_override = joint_target
                action = zero_action
            else:
                action = zero_action
            obs, reward, terminated, truncated, info = env.step(action)
            steps_executed = step + 1
            if joint_target_for_step is not None:
                joint_pos_after_step = inner._robot.data.joint_pos.detach()
                follow_abs = torch.abs(joint_target_for_step - joint_pos_after_step)
                actual_step_abs = torch.abs(joint_pos_after_step - joint_pos_before_step)
                controller_metrics = dict(controller_metrics)
                controller_metrics.update(
                    {
                        "direct_joint_follow_abs_mean_rad": float(follow_abs.mean().item()),
                        "direct_joint_follow_abs_max_rad": float(follow_abs.max().item()),
                        "direct_actual_joint_step_abs_mean_rad": float(actual_step_abs.mean().item()),
                        "direct_actual_joint_step_abs_max_rad": float(actual_step_abs.max().item()),
                    }
                )
            if args.reach_trace_json is not None and reach_trace_for_step is not None:
                inner._compute_intermediate_values()
                terms = inner._tap_terms()
                actual_metrics = _face_metrics_torch(inner._tcp_pos_w, inner._sponge_pos_w, inner._push_dir_xy, cfg, torch)
                follow_max = (
                    _tensor_list(follow_abs.max(dim=-1).values)
                    if joint_target_for_step is not None
                    else [float("nan")] * int(args.num_envs)
                )
                actual_step_max = (
                    _tensor_list(actual_step_abs.max(dim=-1).values)
                    if joint_target_for_step is not None
                    else [float("nan")] * int(args.num_envs)
                )
                cube_pos_rows = inner._sponge_pos_w.detach().cpu().tolist()
                push_rows = inner._push_dir_xy.detach().cpu().tolist()
                actual_face = _tensor_list(actual_metrics["face_gap_m"])
                actual_lateral = _tensor_list(actual_metrics["lateral_m"])
                actual_vertical = _tensor_list(actual_metrics["vertical_offset_m"])
                actual_inside = _bool_tensor_list(actual_metrics["inside_contact_band"])
                cube_disp_along = _tensor_list(terms["disp_along"])
                cube_speed = _tensor_list(terms["speed"])
                professor_now = _bool_tensor_list(terms["professor_physical_reaction_now"])
                professor_seen = _bool_tensor_list(inner._professor_physical_reaction_seen)
                tap_success_now = _bool_tensor_list(terms["tap_success_now"])
                tap_success_seen = _bool_tensor_list(inner._tap_success_flag)
                terminated_rows = _bool_tensor_list(terminated)
                truncated_rows = _bool_tensor_list(truncated)
                for base_row in reach_trace_for_step:
                    env_id = int(base_row["env_id"])
                    row = dict(base_row)
                    row.update(
                        {
                            "step": int(step),
                            "episode_length_s": float(cfg.episode_length_s),
                            "cube_pos_w_xyz": [float(v) for v in cube_pos_rows[env_id]],
                            "push_dir_xy": [float(v) for v in push_rows[env_id]],
                            "actual_tcp_face_gap_m": actual_face[env_id],
                            "actual_tcp_lateral_m": actual_lateral[env_id],
                            "actual_tcp_vertical_offset_m": actual_vertical[env_id],
                            "actual_contact_proxy": actual_inside[env_id],
                            "direct_joint_follow_abs_max_rad": follow_max[env_id],
                            "actual_joint_step_abs_max_rad": actual_step_max[env_id],
                            "cube_disp_along_m": cube_disp_along[env_id],
                            "cube_speed_mps": cube_speed[env_id],
                            "professor_physical_reaction_now": professor_now[env_id],
                            "professor_physical_reaction_seen": professor_seen[env_id],
                            "tap_success_now": tap_success_now[env_id],
                            "tap_success_seen": tap_success_seen[env_id],
                            "terminated": terminated_rows[env_id],
                            "truncated": truncated_rows[env_id],
                        }
                    )
                    reach_trace_rows.append(row)
            if not torch.isfinite(reward).all():
                raise AssertionError(f"non-finite reward at step {step}")
            rewards_all.append(float(reward.mean().item()))
            truncated_count += int(truncated.sum().item())
            terminated_count += int(terminated.sum().item())
            if "log" in info:
                last_log = {key: _scalar(value) for key, value in info["log"].items()}
                for key in (
                    "cube_tap_contact_face_gap_m",
                    "cube_tap_contact_lateral_m",
                    "cube_tap_contact_vertical_offset_m",
                    "cube_push_tcp_cube_dist_m",
                    "cube_push_joint_delta_abs_mean",
                    "cube_push_joint_delta_abs_max",
                    "cube_push_joint_delta_cap_rate",
                    "cube_push_action_abs_mean",
                    "cube_push_action_abs_max",
                    "cube_push_target_lead_abs_mean",
                    "cube_push_target_lead_abs_max",
                    "cube_push_target_lead_limit_rate",
                    "cube_push_contact_slowdown_mean",
                    "cube_push_teacher_blend_mean",
                    "cube_tap_contact_seen_rate",
                    "cube_tap_reaction_contact_context_rate",
                    "cube_tap_success_rate",
                    "cube_tap_professor_physical_reaction_signal_now_rate",
                    "cube_tap_professor_physical_reaction_now_rate",
                    "cube_tap_professor_physical_reaction_seen_rate",
                    ):
                    if key in last_log:
                        _update_trace_stats(log_trace_stats, key, last_log[key])
                if "cube_tap_contact_face_gap_m" in last_log:
                    band = float(cfg.tap_contact_face_band_m)
                    face_gap = float(last_log["cube_tap_contact_face_gap_m"])
                    shortfall = max(0.0, -band - face_gap, face_gap - band)
                    _update_trace_stats(log_trace_stats, "cube_tap_contact_band_shortfall_m", shortfall)
                for key, value in controller_metrics.items():
                    _update_trace_stats(controller_trace_stats, key, value)
            if step % max(1, int(args.steps) // 6) == 0:
                print(
                    "[tap10cm-positive] "
                    f"step={step} reward_mean={reward.mean().item():+.6f} "
                    f"contact={last_log.get('cube_tap_contact_seen_rate', 'NA')} "
                    f"reaction_context={last_log.get('cube_tap_reaction_contact_context_rate', 'NA')} "
                    f"reaction_seen={last_log.get('cube_tap_reaction_seen_rate', 'NA')} "
                    f"overshoot={last_log.get('cube_tap_overshoot_seen_rate', 'NA')} "
                    f"tap_success={last_log.get('cube_tap_success_rate', 'NA')}",
                    flush=True,
                )
            if (
                float(last_log.get("cube_tap_success_rate", 0.0)) > 0.0
                and float(last_log.get("cube_tap_overshoot_seen_rate", 1.0)) == 0.0
            ):
                break

        required_log_keys = {
            "cube_tap_objective_final_relocation_required",
            "cube_tap_contact_seen_rate",
            "cube_tap_reaction_signal_now_rate",
            "cube_tap_reaction_contact_context_rate",
            "cube_tap_reaction_seen_rate",
            "cube_tap_professor_physical_reaction_signal_now_rate",
            "cube_tap_professor_physical_reaction_now_rate",
            "cube_tap_professor_physical_reaction_seen_rate",
            "cube_tap_overshoot_seen_rate",
            "cube_tap_success_rate",
            "cube_tap_max_disp_along_m",
            "cube_tap_max_z_delta_m",
            "cube_tap_max_speed_mps",
            "cube_push_tcp_cube_dist_m",
            "cube_push_joint_delta_abs_mean",
            "cube_push_joint_delta_abs_max",
            "cube_push_joint_delta_cap_rate",
            "cube_push_action_abs_mean",
            "cube_push_action_abs_max",
            "cube_push_target_lead_abs_mean",
            "cube_push_target_lead_abs_max",
            "cube_push_target_lead_limit_rate",
            "cube_push_contact_slowdown_mean",
            "cube_push_teacher_blend_mean",
            "cube_push_grasped_marker_rate",
        }
        missing_logs = sorted(required_log_keys - set(last_log))
        final_required_log = float(last_log.get("cube_tap_objective_final_relocation_required", 1.0))
        contact_seen = float(last_log.get("cube_tap_contact_seen_rate", 0.0))
        reaction_context = float(last_log.get("cube_tap_reaction_contact_context_rate", 0.0))
        reaction_seen = float(last_log.get("cube_tap_reaction_seen_rate", 0.0))
        professor_physical_reaction_seen = float(
            last_log.get("cube_tap_professor_physical_reaction_seen_rate", 0.0)
        )
        tap_success = float(last_log.get("cube_tap_success_rate", 0.0))
        overshoot_seen = float(last_log.get("cube_tap_overshoot_seen_rate", 1.0))
        controller_goal_ok_rate = (
            float(controller_metrics.get("closed_loop_ik_ok_rate", 0.0))
            if args.controller_mode != "builtin_teacher"
            else float(reset_metrics["teacher_goal_ok_rate"])
        )
        positive_control_pass = (
            not missing_logs
            and final_required_log == 0.0
            and reset_metrics["ik_endpoint_reset_rate"] > 0.0
            and controller_goal_ok_rate > 0.0
            and contact_seen > 0.0
            and reaction_context > 0.0
            and reaction_seen > 0.0
            and tap_success > 0.0
            and overshoot_seen == 0.0
            and terminated_count == 0
        )
        professor_physical_reaction_evidence_pass = (
            not missing_logs
            and final_required_log == 0.0
            and reset_metrics["ik_endpoint_reset_rate"] > 0.0
            and controller_goal_ok_rate > 0.0
            and professor_physical_reaction_seen > 0.0
            and overshoot_seen == 0.0
            and terminated_count == 0
        )
        direct_ik_joint_target_apply = args.controller_mode in (
            "external_closed_loop_direct_apply",
            "isaac_builtin_diffik_direct_apply",
            "isaac_builtin_diffik_step_clipped_direct_apply",
        )
        isaac_builtin_diffik_controller_apply = args.controller_mode in (
            "isaac_builtin_diffik_direct_apply",
            "isaac_builtin_diffik_step_clipped_direct_apply",
        )
        result = {
            "artifact_type": "cube10cm_tap_rl_positive_control_sanity_v1",
            "branch": "professor_cube10cm_tap_reaction_quality_tier",
            "status": "PASS" if positive_control_pass else "FAIL",
            "positive_control": "PASS" if positive_control_pass else "FAIL",
            "rl_contact_gated_positive_control": "PASS" if positive_control_pass else "FAIL",
            "professor_physical_reaction_evidence": "PASS"
            if professor_physical_reaction_evidence_pass
            else "FAIL",
            "professor_physical_reaction_evidence_only": (
                professor_physical_reaction_evidence_pass and not positive_control_pass
            ),
            "gpu_runtime": "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL",
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
            "b200": False,
            "track_a": False,
            "env_id": ENV_ID,
            "num_envs": int(args.num_envs),
            "max_steps": int(args.steps),
            "steps_executed": int(steps_executed),
            "seed": int(args.seed),
            "device": str(args.device),
            "robot_usd_path": str(args.robot_usd_path),
            "cube_size_m": CUBE10CM_SIZE_M,
            "cube_mass_kg": CUBE10CM_MASS_KG,
            "terrain_table_z_m": PROJECT_TABLE_Z,
            "episode_length_s": float(cfg.episode_length_s),
            "env_max_episode_length": int(inner.max_episode_length),
            "final_1cm_required": False,
            "fixed_cube_x_m": float(args.fixed_cube_x_m),
            "fixed_cube_y_m": float(args.fixed_cube_y_m),
            "fixed_push_dir_x": float(args.fixed_push_dir_x),
            "fixed_push_dir_y": float(args.fixed_push_dir_y),
            "controller_mode": str(args.controller_mode),
            "target_path_mode": str(args.target_path_mode),
            "precontact_clearance_m": float(args.precontact_clearance_m),
            "tcp_top_margin_m": float(args.tcp_top_margin_m),
            "goal_push_m": float(args.goal_push_m),
            "teacher_horizon_frac": float(args.teacher_horizon_frac),
            "closed_loop_push_steps": int(args.closed_loop_push_steps),
            "direct_ik_joint_target_apply": direct_ik_joint_target_apply,
            "isaac_builtin_diffik_controller_apply": isaac_builtin_diffik_controller_apply,
            "builtin_diffik_step_clipped_target_apply": args.controller_mode
            == "isaac_builtin_diffik_step_clipped_direct_apply",
            "builtin_diffik_lambda": float(args.builtin_diffik_lambda),
            "builtin_diffik_step_clip_rad": float(args.builtin_diffik_step_clip_rad),
            "action_smoothing_alpha": float(cfg.action_smoothing_alpha),
            "contact_joint_delta_scale": float(cfg.contact_joint_delta_scale),
            "max_joint_delta_per_step_rad": float(cfg.max_joint_delta_per_step_rad),
            "joint_target_lead_limit_rad": float(cfg.joint_target_lead_limit_rad),
            "reach_trace_enabled": args.reach_trace_json is not None,
            "reach_trace_json": str(args.reach_trace_json) if args.reach_trace_json is not None else None,
            "reach_trace_row_count": len(reach_trace_rows),
            "controller_goal_ok_rate": controller_goal_ok_rate,
            "obs_shape": list(obs_t.shape),
            "reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "reward_finite": True,
            "truncated_count": truncated_count,
            "terminated_count": terminated_count,
            "required_log_keys_present": not missing_logs,
            "missing_required_log_keys": missing_logs,
            "reset_metrics": reset_metrics,
            "controller_metrics": controller_metrics,
            "controller_trace_stats": controller_trace_stats,
            "log_trace_stats": log_trace_stats,
            "last_log": last_log,
            "blocker": (
                "NONE"
                if positive_control_pass
                else (
                    "RL_CONTACT_GATED_POSITIVE_CONTROL_GATE_FAIL_ONLY"
                    if professor_physical_reaction_evidence_pass
                    else "POSITIVE_CONTROL_AND_PROFESSOR_PHYSICAL_REACTION_GATE_FAIL"
                )
            ),
            "elapsed_s": time.time() - started,
        }
        if args.reach_trace_json is not None:
            _write_reach_trace(args.reach_trace_json, result, reach_trace_rows)
        _write_result(args.out_json, args.out_summary, result)
        return 0 if positive_control_pass else 2
    except Exception as exc:
        result = {
            "artifact_type": "cube10cm_tap_rl_positive_control_sanity_v1",
            "branch": "professor_cube10cm_tap_reaction_quality_tier",
            "status": "BLOCKED",
            "positive_control": "BLOCKED",
            "gpu_runtime": "NO_OR_FAILED_BEFORE_PASS",
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
            "b200": False,
            "track_a": False,
            "env_id": ENV_ID,
            "num_envs": int(args.num_envs),
            "max_steps": int(args.steps),
            "steps_executed": 0,
            "seed": int(args.seed),
            "device": str(args.device),
            "robot_usd_path": str(args.robot_usd_path),
            "cube_size_m": "UNKNOWN",
            "cube_mass_kg": "UNKNOWN",
            "terrain_table_z_m": PROJECT_TABLE_Z,
            "episode_length_s": float(args.episode_length_s) if float(args.episode_length_s) > 0.0 else "UNKNOWN",
            "env_max_episode_length": "UNKNOWN",
            "final_1cm_required": "UNKNOWN",
            "fixed_cube_x_m": float(args.fixed_cube_x_m),
            "fixed_cube_y_m": float(args.fixed_cube_y_m),
            "fixed_push_dir_x": float(args.fixed_push_dir_x),
            "fixed_push_dir_y": float(args.fixed_push_dir_y),
            "controller_mode": str(args.controller_mode),
            "target_path_mode": str(args.target_path_mode),
            "precontact_clearance_m": float(args.precontact_clearance_m),
            "tcp_top_margin_m": float(args.tcp_top_margin_m),
            "goal_push_m": float(args.goal_push_m),
            "teacher_horizon_frac": float(args.teacher_horizon_frac),
            "closed_loop_push_steps": int(args.closed_loop_push_steps),
            "direct_ik_joint_target_apply": False,
            "isaac_builtin_diffik_controller_apply": False,
            "builtin_diffik_step_clipped_target_apply": False,
            "builtin_diffik_lambda": float(args.builtin_diffik_lambda),
            "builtin_diffik_step_clip_rad": float(args.builtin_diffik_step_clip_rad),
            "reach_trace_enabled": args.reach_trace_json is not None,
            "reach_trace_json": str(args.reach_trace_json) if args.reach_trace_json is not None else None,
            "reach_trace_row_count": 0,
            "max_joint_delta_per_step_rad": "UNKNOWN",
            "required_log_keys_present": False,
            "reset_metrics": {},
            "controller_metrics": {},
            "last_log": {},
            "blocker": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "elapsed_s": time.time() - started,
        }
        _write_result(args.out_json, args.out_summary, result)
        return 2
    finally:
        if env is not None:
            env.close()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
