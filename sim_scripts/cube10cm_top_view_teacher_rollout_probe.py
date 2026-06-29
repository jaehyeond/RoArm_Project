#!/usr/bin/env python3
"""Teacher-only rollout and feature-alignment probe for D257/D258.

This script runs no PPO learning. It loads the D257 state-action teacher through
the existing env sidecar path, steps the env with teacher actions only, and
compares online teacher features against the D256 train-clean feature ranges.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_TEACHER_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_CHECKPOINT = (
    D242_ROOT
    / "state_action_teacher_d257"
    / "cube10cm_d257_state_action_teacher_clipped0040.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "teacher_rollout_probe_d259"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def load_train_feature_stats(csv_path: Path, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {c: [] for c in feature_columns}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        missing = [c for c in feature_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing feature columns in {csv_path}: {missing}")
        for row in reader:
            for col in feature_columns:
                values[col].append(float(row[col]))
    stats: dict[str, dict[str, float]] = {}
    for col, vals in values.items():
        vals.sort()
        stats[col] = {
            "min": float(vals[0]),
            "p01": quantile(vals, 0.01),
            "p05": quantile(vals, 0.05),
            "p50": quantile(vals, 0.50),
            "p95": quantile(vals, 0.95),
            "p99": quantile(vals, 0.99),
            "max": float(vals[-1]),
        }
    return stats


def tensor_stats(x) -> dict[str, float]:
    import torch

    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def load_d256_reset_rows(csv_path: Path, frame_index: int, num_envs: int) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    required = [
        "episode_index",
        "frame_index_t",
        "cube_local_x_m",
        "cube_local_y_m",
        "cube_local_z_m",
        "target_local_x_m",
        "target_local_y_m",
        "target_local_z_m",
        "push_dx",
        "push_dy",
        "arm_joint_0_rad",
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
        "gripper_joint_rad",
    ]
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing reset columns in {csv_path}: {missing}")
        for row in reader:
            if int(row["frame_index_t"]) != int(frame_index):
                continue
            candidates.append({c: float(row[c]) for c in required})
    if not candidates:
        raise ValueError(f"no D256 reset rows found for frame_index_t={frame_index}")

    if len(candidates) == 1:
        return [candidates[0] for _ in range(num_envs)]
    if num_envs <= len(candidates):
        idxs = [round(i * (len(candidates) - 1) / max(1, num_envs - 1)) for i in range(num_envs)]
        return [candidates[int(i)] for i in idxs]
    return [candidates[i % len(candidates)] for i in range(num_envs)]


def feature_alignment_rate(x, feature_columns: list[str], train_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    import torch

    outside_minmax_count = 0
    outside_p01p99_count = 0
    total = 0
    x_cpu = x.detach().cpu()
    for col_idx, col in enumerate(feature_columns):
        vals = x_cpu[:, col_idx]
        st = train_stats[col]
        outside_minmax_count += int(((vals < st["min"]) | (vals > st["max"])).sum().item())
        outside_p01p99_count += int(((vals < st["p01"]) | (vals > st["p99"])).sum().item())
        total += int(vals.numel())
    return {
        "outside_train_minmax_rate": float(outside_minmax_count / max(1, total)),
        "outside_train_p01p99_rate": float(outside_p01p99_count / max(1, total)),
    }


def apply_d256_pose_reset(inner, reset_rows: list[dict[str, float]]) -> dict[str, Any]:
    import torch

    device = inner.device
    num_envs = int(inner.num_envs)
    if len(reset_rows) != num_envs:
        raise ValueError(f"reset_rows length {len(reset_rows)} != num_envs {num_envs}")
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]

    joint_pos = inner._robot.data.joint_pos.detach().clone()
    arm = torch.tensor(
        [[float(row[f"arm_joint_{idx}_rad"]) for idx in range(5)] for row in reset_rows],
        device=device,
        dtype=torch.float32,
    )
    gripper = torch.tensor(
        [float(row["gripper_joint_rad"]) for row in reset_rows],
        device=device,
        dtype=torch.float32,
    )
    joint_pos[:, inner._bc_arm_joint_ids] = arm
    joint_pos[:, inner.gripper_joint_idx] = gripper
    joint_pos = torch.clamp(joint_pos, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
    joint_vel = torch.zeros_like(joint_pos)

    cube_local = torch.tensor(
        [
            [float(row["cube_local_x_m"]), float(row["cube_local_y_m"]), float(row["cube_local_z_m"])]
            for row in reset_rows
        ],
        device=device,
        dtype=torch.float32,
    )
    target_local = torch.tensor(
        [
            [float(row["target_local_x_m"]), float(row["target_local_y_m"]), float(row["target_local_z_m"])]
            for row in reset_rows
        ],
        device=device,
        dtype=torch.float32,
    )
    push_dir = torch.tensor(
        [[float(row["push_dx"]), float(row["push_dy"])] for row in reset_rows],
        device=device,
        dtype=torch.float32,
    )
    push_dir = push_dir / torch.clamp(torch.linalg.norm(push_dir, dim=-1, keepdim=True), min=1.0e-6)

    inner._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    inner._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    inner.robot_dof_targets[env_ids] = joint_pos

    cube_world = origins + cube_local
    cube_state = torch.zeros((num_envs, 13), device=device, dtype=torch.float32)
    cube_state[:, 0:3] = cube_world
    cube_state[:, 3] = 1.0
    inner._sponge.write_root_pose_to_sim(cube_state[:, 0:7], env_ids=env_ids)
    inner._sponge.write_root_velocity_to_sim(cube_state[:, 7:13], env_ids=env_ids)

    inner._target_world[env_ids] = origins + target_local
    inner._cube_start_w[env_ids] = cube_world
    inner._push_dir_xy[env_ids] = push_dir
    inner.episode_length_buf[env_ids] = 0
    inner._prev_disp_along[env_ids] = 0.0
    inner._push_success_flag[env_ids] = False
    if hasattr(inner, "_tap_contact_seen"):
        inner._tap_contact_seen[env_ids] = False
        inner._tap_reaction_seen[env_ids] = False
        inner._professor_physical_reaction_seen[env_ids] = False
        inner._tap_overshoot_seen[env_ids] = False
        inner._tap_success_flag[env_ids] = False
        inner._tap_just_succeeded_pending[env_ids] = False
        inner._tap_max_disp_along[env_ids] = 0.0
        inner._tap_max_disp_xy[env_ids] = 0.0
        inner._tap_max_z_delta[env_ids] = 0.0
        inner._tap_max_speed[env_ids] = 0.0
        inner._tap_max_tip_angle_deg[env_ids] = 0.0
        inner._tap_min_contact_vertical_offset[env_ids] = torch.inf
        inner._last_tap_stop_after_useful_hold[env_ids] = 0.0
        inner._last_tap_stop_after_disp_hold[env_ids] = 0.0
    inner._smoothed_actions[env_ids] = 0.0
    inner._last_joint_delta_abs_mean[env_ids] = 0.0
    inner._last_joint_delta_abs_max[env_ids] = 0.0
    inner._last_joint_delta_cap_rate[env_ids] = 0.0
    inner._last_action_abs_mean[env_ids] = 0.0
    inner._last_action_abs_max[env_ids] = 0.0
    inner._last_target_lead_abs_mean[env_ids] = 0.0
    inner._last_target_lead_abs_max[env_ids] = 0.0
    inner._last_target_lead_limit_rate[env_ids] = 0.0
    inner._last_contact_slowdown[env_ids] = 1.0
    inner._last_teacher_blend[env_ids] = 0.0
    inner._last_bc_teacher_blend[env_ids] = 0.0
    inner._last_bc_teacher_imitation_mse[env_ids] = 0.0
    inner._last_bc_teacher_action_abs_mean[env_ids] = 0.0
    inner._bc_prev_teacher_delta[env_ids] = 0.0
    inner._teacher_start_joints[env_ids] = joint_pos
    inner._teacher_goal_joints[env_ids] = joint_pos
    inner._teacher_goal_ok[env_ids] = False
    inner._grasped[env_ids] = False
    inner._was_grasped[env_ids] = False

    inner.scene.write_data_to_sim()
    inner.scene.update(inner.sim.get_physics_dt())
    inner._compute_intermediate_values()
    episodes = [int(row["episode_index"]) for row in reset_rows]
    return {
        "row_count": int(len(reset_rows)),
        "episode_min": int(min(episodes)),
        "episode_max": int(max(episodes)),
        "episode_unique_count": int(len(set(episodes))),
        "arm_joint_reset_mean_rad": [float(v) for v in arm.mean(dim=0).detach().cpu().tolist()],
        "cube_local_x_range_m": [float(cube_local[:, 0].min().item()), float(cube_local[:, 0].max().item())],
        "cube_local_y_range_m": [float(cube_local[:, 1].min().item()), float(cube_local[:, 1].max().item())],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_kind", choices=("push3cm", "tap10cm"), default="push3cm")
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1257)
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=580)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--artifact_tag", default="d261")
    parser.add_argument("--ik_endpoint_reset", action="store_true")
    parser.add_argument("--fixed_push_dir_x", type=float, default=None)
    parser.add_argument("--fixed_push_dir_y", type=float, default=None)
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--bc_teacher_policy_delta_clip_rad", type=float, default=0.04)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=0.8)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=0.85)
    parser.add_argument(
        "--bc_teacher_phase_timing",
        choices=("episode_scaled", "direct_steps", "linear_episode", "linear_steps"),
        default="direct_steps",
    )
    parser.add_argument("--bc_teacher_linear_phase_steps", type=int, default=579)
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument(
        "--tap_contact_proxy_mode",
        choices=("tcp_point", "link5_collision_aabb"),
        default="link5_collision_aabb",
    )
    parser.add_argument(
        "--reset_pose_source",
        choices=("env_default", "d256_initial", "env_d256_initial"),
        default="env_default",
    )
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="random")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401 - registers envs lazily after AppLauncher
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg, RoArmCubeTap10cmEnvCfg

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_columns = list(checkpoint["feature_columns"])
    train_stats = load_train_feature_stats(args.teacher_csv, feature_columns)

    if args.env_kind == "push3cm":
        env_id = "RoArm-CubePush-Direct-v0"
        env_cfg = RoArmCubePushEnvCfg()
    else:
        env_id = "RoArm-CubeTap10cm-Direct-v0"
        env_cfg = RoArmCubeTap10cmEnvCfg()

    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.ik_endpoint_reset = bool(args.ik_endpoint_reset)
    if args.fixed_push_dir_x is not None:
        env_cfg.fixed_push_dir_x = float(args.fixed_push_dir_x)
    if args.fixed_push_dir_y is not None:
        env_cfg.fixed_push_dir_y = float(args.fixed_push_dir_y)
    env_cfg.action_scale = float(args.action_scale)
    env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.bc_teacher_checkpoint_path = str(args.checkpoint)
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_policy_delta_clip_rad = float(args.bc_teacher_policy_delta_clip_rad)
    env_cfg.bc_teacher_policy_delta_scale = float(args.bc_teacher_policy_delta_scale)
    env_cfg.bc_teacher_lowx_policy_delta_scale = float(args.bc_teacher_lowx_policy_delta_scale)
    env_cfg.bc_teacher_highx_policy_delta_scale = float(args.bc_teacher_highx_policy_delta_scale)
    env_cfg.bc_teacher_delta_smoothing_alpha = float(args.bc_teacher_delta_smoothing_alpha)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)
    env_cfg.bc_teacher_linear_phase_steps = int(args.bc_teacher_linear_phase_steps)
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    if args.env_kind == "tap10cm":
        env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    if args.reset_pose_source == "env_d256_initial":
        env_cfg.d256_reset_csv_path = str(args.teacher_csv)
        env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
        env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)

    print(
        "teacher_rollout_probe line1 "
        f"env_kind={args.env_kind} env_id={env_id} num_envs={args.num_envs} "
        f"steps={args.steps} seed={args.seed} "
        f"bc_teacher_feature_target_mode={env_cfg.bc_teacher_feature_target_mode} "
        f"tap_contact_proxy_mode={getattr(env_cfg, 'tap_contact_proxy_mode', 'NA')} no_ppo_learning=YES"
    )
    print(
        "teacher_rollout_probe line2 "
        f"checkpoint={_rel(args.checkpoint)} teacher_csv={_rel(args.teacher_csv)}"
    )

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    if not getattr(inner, "_bc_teacher_ready", False):
        raise RuntimeError("BC teacher sidecar was not loaded")

    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)
    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    reset_pose_info: dict[str, Any] = {"source": str(args.reset_pose_source)}
    if args.reset_pose_source == "env_d256_initial":
        episode_idx = inner._last_d256_reset_episode_index.detach().cpu()
        valid_episode_idx = episode_idx[episode_idx >= 0]
        reset_pose_info.update(
            {
                "d256_reset_frame_index": int(args.d256_reset_frame_index),
                "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
                "env_reset_hook_active_rate": float(inner._last_d256_reset_active.mean().item()),
                "episode_min": int(valid_episode_idx.min().item()) if valid_episode_idx.numel() else -1,
                "episode_max": int(valid_episode_idx.max().item()) if valid_episode_idx.numel() else -1,
                "episode_unique_count": int(len(set(int(v) for v in valid_episode_idx.tolist()))),
            }
        )
        print(
            "teacher_rollout_probe reset_hook "
            f"source=env_d256_initial frame_index={args.d256_reset_frame_index} "
            f"sample_mode={args.d256_reset_sample_mode} "
            f"active_rate={reset_pose_info['env_reset_hook_active_rate']} "
            f"unique_episodes={reset_pose_info['episode_unique_count']} "
            f"episode_range={reset_pose_info['episode_min']}..{reset_pose_info['episode_max']}"
        )
    if args.reset_pose_source == "d256_initial":
        reset_rows = load_d256_reset_rows(args.teacher_csv, int(args.d256_reset_frame_index), int(inner.num_envs))
        reset_pose_info.update(apply_d256_pose_reset(inner, reset_rows))
        reset_pose_info["d256_reset_frame_index"] = int(args.d256_reset_frame_index)
        print(
            "teacher_rollout_probe reset_override "
            f"source=d256_initial frame_index={args.d256_reset_frame_index} "
            f"rows={reset_pose_info['row_count']} unique_episodes={reset_pose_info['episode_unique_count']} "
            f"episode_range={reset_pose_info['episode_min']}..{reset_pose_info['episode_max']}"
        )
    inner._compute_intermediate_values()
    initial_traj = inner._bc_teacher_traj()
    initial_alpha = inner._bc_teacher_phase_alpha(initial_traj)
    initial_tcp_target = inner._bc_teacher_tcp_target(initial_alpha, initial_traj)
    initial_features = inner._bc_teacher_feature_tensor(initial_alpha, initial_tcp_target)
    reset_pose_info.update(
        {
            "initial_phase_alpha": tensor_stats(initial_alpha),
            "initial_feature_alignment": feature_alignment_rate(initial_features, feature_columns, train_stats),
        }
    )

    out_dir = args.out_dir / args.env_kind
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_tag = str(args.artifact_tag)
    step_csv = out_dir / f"teacher_rollout_step_samples_{artifact_tag}.csv"
    summary_json = out_dir / f"teacher_rollout_probe_summary_{artifact_tag}.json"
    summary_md = out_dir / f"teacher_rollout_probe_summary_{artifact_tag}.md"

    min_tcp_cube_dist = torch.full((inner.num_envs,), float("inf"), device=device)
    max_disp_along = torch.full((inner.num_envs,), -float("inf"), device=device)
    max_disp_xy = torch.zeros(inner.num_envs, device=device)
    tap_overshoot_seen = torch.zeros(inner.num_envs, dtype=torch.bool, device=device)
    tap_reaction_seen = torch.zeros(inner.num_envs, dtype=torch.bool, device=device)
    min_tap_contact_vertical_offset = torch.full((inner.num_envs,), float("inf"), device=device)
    last_tap_contact_vertical_offset = torch.zeros(inner.num_envs, device=device)
    last_tap_contact_face_gap = torch.zeros(inner.num_envs, device=device)
    last_tap_contact_lateral = torch.zeros(inner.num_envs, device=device)
    first_contact_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_tcp_threshold_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_tap_useful_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_alpha_gt0_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_alpha_eq1_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    contact_threshold = float(inner.cfg.contact_slowdown_tcp_dist_m)

    outside_minmax_count = 0
    outside_p01p99_count = 0
    feature_value_count = 0
    feature_env_min = {c: float("inf") for c in feature_columns}
    feature_env_max = {c: -float("inf") for c in feature_columns}
    tracked_feature_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    raw_delta_clip_count = 0
    raw_delta_count = 0
    action_cap_count = 0
    action_count = 0
    raw_delta_abs_values = []
    clamped_delta_abs_values = []
    action_abs_values = []

    with torch.inference_mode():
        for step in range(int(args.steps)):
            inner._compute_intermediate_values()
            traj = inner._bc_teacher_traj()
            alpha = inner._bc_teacher_phase_alpha(traj)
            tcp_target = inner._bc_teacher_tcp_target(alpha, traj)
            x = inner._bc_teacher_feature_tensor(alpha, tcp_target)
            pred_n = inner._bc_teacher_model((x - inner._bc_teacher_x_mean) / inner._bc_teacher_x_std)
            raw_delta = pred_n * inner._bc_teacher_y_std + inner._bc_teacher_y_mean
            clamped_delta = torch.clamp(
                raw_delta,
                -float(inner.cfg.bc_teacher_policy_delta_clip_rad),
                float(inner.cfg.bc_teacher_policy_delta_clip_rad),
            )
            actions = inner._bc_teacher_actions().detach().clamp(-1.0, 1.0)

            raw_delta_clip_count += int((torch.abs(raw_delta) > float(inner.cfg.bc_teacher_policy_delta_clip_rad)).sum().item())
            raw_delta_count += int(raw_delta.numel())
            action_cap_count += int((torch.abs(actions) >= 0.999).sum().item())
            action_count += int(actions.numel())
            raw_delta_abs_values.append(torch.abs(raw_delta).detach().flatten().cpu())
            clamped_delta_abs_values.append(torch.abs(clamped_delta).detach().flatten().cpu())
            action_abs_values.append(torch.abs(actions).detach().flatten().cpu())

            x_cpu = x.detach().cpu()
            for col_idx, col in enumerate(feature_columns):
                vals = x_cpu[:, col_idx]
                feature_env_min[col] = min(feature_env_min[col], float(vals.min().item()))
                feature_env_max[col] = max(feature_env_max[col], float(vals.max().item()))
                st = train_stats[col]
                outside_minmax_count += int(((vals < st["min"]) | (vals > st["max"])).sum().item())
                outside_p01p99_count += int(((vals < st["p01"]) | (vals > st["p99"])).sum().item())
                feature_value_count += int(vals.numel())

            obs, rewards, dones, infos = env.step(actions)
            inner._compute_intermediate_values()
            terms = inner._tap_terms() if args.env_kind == "tap10cm" else inner._push_terms()
            tcp_dist = terms["tcp_cube_dist"].detach()
            disp_along = terms["disp_along"].detach()
            disp_xy = terms["disp_xy"].detach()
            min_tcp_cube_dist = torch.minimum(min_tcp_cube_dist, tcp_dist)
            max_disp_along = torch.maximum(max_disp_along, disp_along)
            max_disp_xy = torch.maximum(max_disp_xy, disp_xy)
            tcp_threshold_now = tcp_dist < contact_threshold
            if args.env_kind == "tap10cm":
                contact_now = terms["tap_contact_proxy"].detach()
                reaction_now = terms["tap_reaction_now"].detach()
                overshoot_now = terms["tap_overshoot_now"].detach()
                tap_useful_now = contact_now & reaction_now & ~overshoot_now
                tap_reaction_seen |= reaction_now
                tap_overshoot_seen |= overshoot_now
                vertical_offset = terms["tap_contact_vertical_offset_m"].detach()
                min_tap_contact_vertical_offset = torch.minimum(min_tap_contact_vertical_offset, vertical_offset)
                last_tap_contact_vertical_offset = vertical_offset
                last_tap_contact_face_gap = terms["tap_contact_face_gap_m"].detach()
                last_tap_contact_lateral = terms["tap_contact_lateral_m"].detach()
            else:
                contact_now = tcp_threshold_now
                tap_useful_now = torch.zeros_like(contact_now)
                reaction_now = torch.zeros_like(contact_now)
                overshoot_now = torch.zeros_like(contact_now)
            unset_contact = (first_contact_step < 0) & contact_now
            first_contact_step[unset_contact] = int(step)
            unset_tcp = (first_tcp_threshold_step < 0) & tcp_threshold_now
            first_tcp_threshold_step[unset_tcp] = int(step)
            unset_tap_useful = (first_tap_useful_step < 0) & tap_useful_now
            first_tap_useful_step[unset_tap_useful] = int(step)
            unset_alpha_gt0 = (first_alpha_gt0_step < 0) & (alpha > 1.0e-6)
            first_alpha_gt0_step[unset_alpha_gt0] = int(step)
            unset_alpha_eq1 = (first_alpha_eq1_step < 0) & (alpha >= 1.0 - 1.0e-6)
            first_alpha_eq1_step[unset_alpha_eq1] = int(step)

            if step % int(args.sample_every) == 0 or step == int(args.steps) - 1:
                row = {
                    "step": int(step),
                    "alpha_mean": float(alpha.mean().item()),
                    "alpha_min": float(alpha.min().item()),
                    "alpha_max": float(alpha.max().item()),
                    "tcp_cube_dist_mean": float(tcp_dist.mean().item()),
                    "tcp_cube_dist_min": float(tcp_dist.min().item()),
                    "disp_along_mean": float(disp_along.mean().item()),
                    "disp_along_max": float(disp_along.max().item()),
                    "disp_xy_mean": float(disp_xy.mean().item()),
                    "disp_xy_max": float(disp_xy.max().item()),
                    "contact_rate": float(contact_now.float().mean().item()),
                    "tcp_threshold_contact_rate": float(tcp_threshold_now.float().mean().item()),
                    "tap_reaction_now_rate": float(reaction_now.float().mean().item()),
                    "tap_useful_now_rate": float(tap_useful_now.float().mean().item()),
                    "tap_overshoot_now_rate": float(overshoot_now.float().mean().item()),
                    "tap_contact_vertical_offset_mean": (
                        float(last_tap_contact_vertical_offset.mean().item()) if args.env_kind == "tap10cm" else float("nan")
                    ),
                    "tap_contact_face_gap_mean": (
                        float(last_tap_contact_face_gap.mean().item()) if args.env_kind == "tap10cm" else float("nan")
                    ),
                    "tap_contact_lateral_mean": (
                        float(last_tap_contact_lateral.mean().item()) if args.env_kind == "tap10cm" else float("nan")
                    ),
                    "raw_delta_abs_mean": float(torch.abs(raw_delta).mean().item()),
                    "raw_delta_abs_max": float(torch.abs(raw_delta).max().item()),
                    "clamped_delta_abs_mean": float(torch.abs(clamped_delta).mean().item()),
                    "clamped_delta_abs_max": float(torch.abs(clamped_delta).max().item()),
                    "action_abs_mean": float(torch.abs(actions).mean().item()),
                    "action_abs_max": float(torch.abs(actions).max().item()),
                }
                step_rows.append(row)

    raw_abs = torch.cat(raw_delta_abs_values)
    clamped_abs = torch.cat(clamped_delta_abs_values)
    action_abs = torch.cat(action_abs_values)

    feature_alignment = []
    for col in feature_columns:
        st = train_stats[col]
        env_min = feature_env_min[col]
        env_max = feature_env_max[col]
        outside_minmax = env_min < st["min"] or env_max > st["max"]
        outside_p01p99 = env_min < st["p01"] or env_max > st["p99"]
        feature_alignment.append(
            {
                "feature": col,
                "train_min": st["min"],
                "train_p01": st["p01"],
                "train_p50": st["p50"],
                "train_p99": st["p99"],
                "train_max": st["max"],
                "env_min": env_min,
                "env_max": env_max,
                "outside_train_minmax": bool(outside_minmax),
                "outside_train_p01p99": bool(outside_p01p99),
            }
        )

    feature_alignment_sorted = sorted(
        feature_alignment,
        key=lambda r: (not r["outside_train_minmax"], not r["outside_train_p01p99"], r["feature"]),
    )

    summary = {
        "artifact": f"cube10cm_{artifact_tag}_teacher_rollout_probe_{args.env_kind}",
        "status": "PASS_PROBE_EXECUTED",
        "env_kind": args.env_kind,
        "env_id": env_id,
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "episode_length_s": float(args.episode_length_s),
        "ik_endpoint_reset": bool(inner.cfg.ik_endpoint_reset),
        "reset_pose_source": str(args.reset_pose_source),
        "reset_pose_info": reset_pose_info,
        "fixed_push_dir_x": float(inner.cfg.fixed_push_dir_x),
        "fixed_push_dir_y": float(inner.cfg.fixed_push_dir_y),
        "checkpoint": _rel(args.checkpoint),
        "teacher_csv": _rel(args.teacher_csv),
        "robot_usd_path": _rel(args.robot_usd_path),
        "cube_size_x_m": float(inner.cfg.cube_size_x_m),
        "cube_size_z_m": float(inner.cfg.cube_size_z_m),
        "bc_teacher_phase_timing": str(inner.cfg.bc_teacher_phase_timing),
        "bc_teacher_linear_phase_steps": int(inner.cfg.bc_teacher_linear_phase_steps),
        "bc_teacher_feature_target_mode": str(inner.cfg.bc_teacher_feature_target_mode),
        "bc_teacher_policy_delta_clip_rad": float(inner.cfg.bc_teacher_policy_delta_clip_rad),
        "tap_contact_proxy_mode": str(getattr(inner.cfg, "tap_contact_proxy_mode", "NA")),
        "contact_threshold_m": contact_threshold,
        "contact_env_count": int((first_contact_step >= 0).sum().item()),
        "contact_rate": float((first_contact_step >= 0).float().mean().item()),
        "first_contact_step_min": int(first_contact_step[first_contact_step >= 0].min().item()) if bool((first_contact_step >= 0).any()) else -1,
        "tcp_threshold_contact_env_count": int((first_tcp_threshold_step >= 0).sum().item()),
        "tcp_threshold_contact_rate": float((first_tcp_threshold_step >= 0).float().mean().item()),
        "first_tcp_threshold_step_min": int(first_tcp_threshold_step[first_tcp_threshold_step >= 0].min().item()) if bool((first_tcp_threshold_step >= 0).any()) else -1,
        "tap_useful_env_count": int((first_tap_useful_step >= 0).sum().item()),
        "tap_useful_rate": float((first_tap_useful_step >= 0).float().mean().item()),
        "first_tap_useful_step_min": int(first_tap_useful_step[first_tap_useful_step >= 0].min().item()) if bool((first_tap_useful_step >= 0).any()) else -1,
        "tap_reaction_seen_rate": float(tap_reaction_seen.float().mean().item()),
        "tap_overshoot_seen_rate": float(tap_overshoot_seen.float().mean().item()),
        "min_tap_contact_vertical_offset_m": tensor_stats(min_tap_contact_vertical_offset),
        "last_tap_contact_vertical_offset_m": tensor_stats(last_tap_contact_vertical_offset),
        "last_tap_contact_face_gap_m": tensor_stats(last_tap_contact_face_gap),
        "last_tap_contact_lateral_m": tensor_stats(last_tap_contact_lateral),
        "alpha_gt0_env_count": int((first_alpha_gt0_step >= 0).sum().item()),
        "alpha_eq1_env_count": int((first_alpha_eq1_step >= 0).sum().item()),
        "first_alpha_gt0_step_min": int(first_alpha_gt0_step[first_alpha_gt0_step >= 0].min().item()) if bool((first_alpha_gt0_step >= 0).any()) else -1,
        "first_alpha_eq1_step_min": int(first_alpha_eq1_step[first_alpha_eq1_step >= 0].min().item()) if bool((first_alpha_eq1_step >= 0).any()) else -1,
        "min_tcp_cube_dist_m": tensor_stats(min_tcp_cube_dist),
        "max_disp_along_m": tensor_stats(max_disp_along),
        "max_disp_xy_m": tensor_stats(max_disp_xy),
        "raw_delta_abs": {
            **tensor_stats(raw_abs),
            "clip_exceed_rate": float(raw_delta_clip_count / max(1, raw_delta_count)),
        },
        "clamped_delta_abs": tensor_stats(clamped_abs),
        "action_abs": {
            **tensor_stats(action_abs),
            "cap_rate": float(action_cap_count / max(1, action_count)),
        },
        "feature_outside_train_minmax_rate": float(outside_minmax_count / max(1, feature_value_count)),
        "feature_outside_train_p01p99_rate": float(outside_p01p99_count / max(1, feature_value_count)),
        "feature_alignment": feature_alignment_sorted,
        "step_samples_csv": _rel(step_csv),
        "summary_json": _rel(summary_json),
        "summary_md": _rel(summary_md),
        "interpretation": "",
    }

    if args.env_kind == "push3cm" and float(inner.cfg.cube_size_x_m) != 0.100:
        summary["interpretation"] = (
            "This reproduces the D258 env kind: it is a 3cm CubePush env, not the "
            "10cm professor cube env used by the D247-D257 data. Any feature "
            "mismatch here can explain weak D258 behavior and should not be "
            "treated as teacher failure on the intended 10cm task."
        )
    elif args.env_kind == "tap10cm":
        summary["interpretation"] = (
            "This uses the 10cm CubeTap env geometry that matches the professor "
            "dataset object size better than D258's CubePush env. Use contact and "
            "feature-alignment metrics here to decide whether the teacher itself "
            "can produce plausible contact before any longer PPO. For tap10cm, "
            "contact_rate uses tap_contact_proxy_mode, while "
            "tcp_threshold_contact_rate reports the older tcp_cube_dist threshold."
        )
    if str(inner.cfg.bc_teacher_feature_target_mode) == "env_target":
        summary["interpretation"] += (
            " The BC teacher feature target is env_target, matching the D256 "
            "visual-log target_position_world_m feature contract rather than the "
            "online TCP waypoint."
        )

    with step_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()) if step_rows else ["step"])
        writer.writeheader()
        writer.writerows(step_rows)

    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    top_features = feature_alignment_sorted[:8]
    summary_md.write_text(
        f"# {artifact_tag.upper()} Teacher Rollout Probe Summary - {args.env_kind}\n\n"
        f"- status: `{summary['status']}`\n"
        f"- env id: `{env_id}`\n"
        f"- cube size x/z m: `{summary['cube_size_x_m']}` / `{summary['cube_size_z_m']}`\n"
        f"- steps/envs: `{args.steps}` / `{args.num_envs}`\n"
        f"- ik endpoint reset: `{summary['ik_endpoint_reset']}`\n"
        f"- reset pose source: `{summary['reset_pose_source']}`\n"
        f"- initial feature outside train min/max rate: "
        f"`{summary['reset_pose_info']['initial_feature_alignment']['outside_train_minmax_rate']}`\n"
        f"- initial feature outside train p01/p99 rate: "
        f"`{summary['reset_pose_info']['initial_feature_alignment']['outside_train_p01p99_rate']}`\n"
        f"- fixed push dir x/y: `{summary['fixed_push_dir_x']}` / `{summary['fixed_push_dir_y']}`\n"
        f"- bc teacher feature target mode: `{summary['bc_teacher_feature_target_mode']}`\n"
        f"- tap contact proxy mode: `{summary['tap_contact_proxy_mode']}`\n"
        f"- contact rate: `{summary['contact_rate']}`\n"
        f"- first contact step min: `{summary['first_contact_step_min']}`\n"
        f"- TCP-threshold contact rate: `{summary['tcp_threshold_contact_rate']}`\n"
        f"- tap useful rate: `{summary['tap_useful_rate']}`\n"
        f"- tap reaction seen rate: `{summary['tap_reaction_seen_rate']}`\n"
        f"- tap overshoot seen rate: `{summary['tap_overshoot_seen_rate']}`\n"
        f"- first alpha > 0 step min: `{summary['first_alpha_gt0_step_min']}`\n"
        f"- first alpha == 1 step min: `{summary['first_alpha_eq1_step_min']}`\n"
        f"- min TCP-cube distance mean/min/max: "
        f"`{summary['min_tcp_cube_dist_m']['mean']}` / `{summary['min_tcp_cube_dist_m']['min']}` / `{summary['min_tcp_cube_dist_m']['max']}`\n"
        f"- min tap contact vertical offset mean/min/max: "
        f"`{summary['min_tap_contact_vertical_offset_m']['mean']}` / "
        f"`{summary['min_tap_contact_vertical_offset_m']['min']}` / "
        f"`{summary['min_tap_contact_vertical_offset_m']['max']}`\n"
        f"- last tap contact vertical offset mean/min/max: "
        f"`{summary['last_tap_contact_vertical_offset_m']['mean']}` / "
        f"`{summary['last_tap_contact_vertical_offset_m']['min']}` / "
        f"`{summary['last_tap_contact_vertical_offset_m']['max']}`\n"
        f"- max disp along mean/min/max: "
        f"`{summary['max_disp_along_m']['mean']}` / `{summary['max_disp_along_m']['min']}` / `{summary['max_disp_along_m']['max']}`\n"
        f"- raw delta clip exceed rate: `{summary['raw_delta_abs']['clip_exceed_rate']}`\n"
        f"- action cap rate: `{summary['action_abs']['cap_rate']}`\n"
        f"- feature outside train min/max rate: `{summary['feature_outside_train_minmax_rate']}`\n"
        f"- feature outside train p01/p99 rate: `{summary['feature_outside_train_p01p99_rate']}`\n\n"
        "Top feature alignment warnings:\n\n"
        + "\n".join(
            f"- `{r['feature']}` train [`{r['train_min']}`, `{r['train_max']}`], "
            f"env [`{r['env_min']}`, `{r['env_max']}`], "
            f"outside_minmax=`{r['outside_train_minmax']}`, outside_p01p99=`{r['outside_train_p01p99']}`"
            for r in top_features
        )
        + "\n\n"
        f"Interpretation: {summary['interpretation']}\n"
    )

    print(
        "teacher_rollout_probe result "
        f"env_kind={args.env_kind} reset_pose_source={args.reset_pose_source} "
        f"tap_contact_proxy_mode={summary['tap_contact_proxy_mode']} "
        f"contact_rate={summary['contact_rate']:.6f} "
        f"tcp_threshold_contact_rate={summary['tcp_threshold_contact_rate']:.6f} "
        f"tap_useful_rate={summary['tap_useful_rate']:.6f} "
        f"max_disp_along_mean={summary['max_disp_along_m']['mean']:.6f} "
        f"min_tcp_dist_mean={summary['min_tcp_cube_dist_m']['mean']:.6f} "
        f"outside_minmax_rate={summary['feature_outside_train_minmax_rate']:.6f} "
        f"summary={_rel(summary_json)}"
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
