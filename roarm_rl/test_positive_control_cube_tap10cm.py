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


def _closed_loop_ik_action(inner: Any, cfg: Any, args: argparse.Namespace, step: int, torch_mod: Any) -> tuple[Any, dict[str, float]]:
    from sim_scripts.roarm_kinematics import ik_dls

    inner._compute_intermediate_values()
    cube_local = (inner._cube_start_w - inner.scene.env_origins).detach().cpu().numpy()
    push_dir = inner._push_dir_xy.detach().cpu().numpy()
    current_q_rad = inner._robot.data.joint_pos.detach().cpu().numpy()
    target_base_rad = inner.robot_dof_targets.detach().cpu().numpy()
    half_xy = np.asarray([float(cfg.cube_size_x_m) * 0.5, float(cfg.cube_size_y_m) * 0.5], dtype=np.float64)
    alpha = min(1.0, max(0.0, float(step + 1) / max(float(args.closed_loop_push_steps), 1.0)))
    actions = np.zeros((int(args.num_envs), int(cfg.action_space)), dtype=np.float32)
    ok_count = 0
    err_values: list[float] = []
    for env_id in range(int(args.num_envs)):
        half_along = float(np.sum(np.abs(push_dir[env_id, :2]) * half_xy))
        pre = cube_local[env_id].copy()
        through = cube_local[env_id].copy()
        pre[:2] -= push_dir[env_id] * (half_along + float(args.precontact_clearance_m))
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
        action = (target_rad - target_base_rad[env_id]) / max(float(cfg.action_scale), 1.0e-6)
        action[5] = 0.0
        actions[env_id] = np.clip(action, -1.0, 1.0).astype(np.float32)
        ok_count += int(bool(converged))
        err_values.append(float(err_mm))
    action_t = torch_mod.tensor(actions, dtype=torch_mod.float32, device=inner.device)
    metrics = {
        "closed_loop_ik_ok_rate": float(ok_count) / max(float(args.num_envs), 1.0),
        "closed_loop_ik_err_mm_mean": float(np.mean(err_values)) if err_values else float("nan"),
        "closed_loop_alpha": alpha,
    }
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
            f"final_1cm_required={result.get('final_1cm_required', 'UNKNOWN')}"
        ),
        (
            "line3 scripted_control "
            f"num_envs={result.get('num_envs', 'NA')} max_steps={result.get('max_steps', 'NA')} "
            f"steps_executed={result.get('steps_executed', 'NA')} "
            f"cube_xy=({result.get('fixed_cube_x_m', 'NA')},{result.get('fixed_cube_y_m', 'NA')}) "
            f"push_dir=({result.get('fixed_push_dir_x', 'NA')},{result.get('fixed_push_dir_y', 'NA')}) "
            f"controller_mode={result.get('controller_mode', 'NA')} "
            f"precontact_clearance_m={result.get('precontact_clearance_m', 'NA')} "
            f"tcp_top_margin_m={result.get('tcp_top_margin_m', 'NA')} "
            f"goal_push_m={result.get('goal_push_m', 'NA')} "
            f"max_joint_delta_per_step_rad={result.get('max_joint_delta_per_step_rad', 'NA')}"
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
            f"positive_control={result.get('positive_control', 'UNKNOWN')} "
            f"blocker={result.get('blocker', 'NONE')} "
            "unblocks=wrapper_positive_control_evidence_only "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
    ]
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line, flush=True)


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
    parser.add_argument("--teacher_horizon_frac", type=float, default=1.0)
    parser.add_argument("--controller_mode", choices=("builtin_teacher", "external_closed_loop"), default="builtin_teacher")
    parser.add_argument("--closed_loop_push_steps", type=int, default=72)
    parser.add_argument("--closed_loop_ik_max_iter", type=int, default=80)
    parser.add_argument("--closed_loop_ik_tol_mm", type=float, default=1.5)
    parser.add_argument("--action_smoothing_alpha", type=float, default=-1.0)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=-1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=-1.0)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_LOCAL_USD)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
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
        zero_action = torch.zeros((args.num_envs, cfg.action_space), device=inner.device)
        for step in range(int(args.steps)):
            if args.controller_mode == "external_closed_loop":
                action, controller_metrics = _closed_loop_ik_action(inner, cfg, args, step, torch)
            else:
                action = zero_action
            obs, reward, terminated, truncated, info = env.step(action)
            steps_executed = step + 1
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
        tap_success = float(last_log.get("cube_tap_success_rate", 0.0))
        overshoot_seen = float(last_log.get("cube_tap_overshoot_seen_rate", 1.0))
        controller_goal_ok_rate = (
            float(controller_metrics.get("closed_loop_ik_ok_rate", 0.0))
            if args.controller_mode == "external_closed_loop"
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
        result = {
            "artifact_type": "cube10cm_tap_rl_positive_control_sanity_v1",
            "branch": "professor_cube10cm_tap_reaction_quality_tier",
            "status": "PASS" if positive_control_pass else "FAIL",
            "positive_control": "PASS" if positive_control_pass else "FAIL",
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
            "final_1cm_required": False,
            "fixed_cube_x_m": float(args.fixed_cube_x_m),
            "fixed_cube_y_m": float(args.fixed_cube_y_m),
            "fixed_push_dir_x": float(args.fixed_push_dir_x),
            "fixed_push_dir_y": float(args.fixed_push_dir_y),
            "controller_mode": str(args.controller_mode),
            "precontact_clearance_m": float(args.precontact_clearance_m),
            "tcp_top_margin_m": float(args.tcp_top_margin_m),
            "goal_push_m": float(args.goal_push_m),
            "teacher_horizon_frac": float(args.teacher_horizon_frac),
            "closed_loop_push_steps": int(args.closed_loop_push_steps),
            "action_smoothing_alpha": float(cfg.action_smoothing_alpha),
            "contact_joint_delta_scale": float(cfg.contact_joint_delta_scale),
            "max_joint_delta_per_step_rad": float(cfg.max_joint_delta_per_step_rad),
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
            "blocker": "NONE" if positive_control_pass else "POSITIVE_CONTROL_GATE_FAIL",
            "elapsed_s": time.time() - started,
        }
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
            "final_1cm_required": "UNKNOWN",
            "fixed_cube_x_m": float(args.fixed_cube_x_m),
            "fixed_cube_y_m": float(args.fixed_cube_y_m),
            "fixed_push_dir_x": float(args.fixed_push_dir_x),
            "fixed_push_dir_y": float(args.fixed_push_dir_y),
            "controller_mode": str(args.controller_mode),
            "precontact_clearance_m": float(args.precontact_clearance_m),
            "tcp_top_margin_m": float(args.tcp_top_margin_m),
            "goal_push_m": float(args.goal_push_m),
            "teacher_horizon_frac": float(args.teacher_horizon_frac),
            "closed_loop_push_steps": int(args.closed_loop_push_steps),
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
