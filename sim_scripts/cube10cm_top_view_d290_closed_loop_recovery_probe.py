#!/usr/bin/env python3
"""Collect closed-loop actor states with D256 recovery labels.

This is a DAgger-style diagnostic, not PPO. It executes a frozen actor under
the D256 reset contract, records the states the actor actually visits, and
labels each state with the action needed to move back toward the matching D256
recorded episode/time joint target.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_D256_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_ACTOR_CHECKPOINT = (
    RUNTIME_ROOT
    / "actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "closed_loop_recovery_d290" / "tap10cm"
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _tensor_mean(x: Any) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_max(x: Any) -> float:
    return float(x.detach().float().max().cpu().item())


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _action_metrics(torch: Any, pred: Any, target: Any) -> dict[str, float]:
    diff = pred - target
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)
    return {
        "mse": _tensor_mean(torch.mean(diff * diff, dim=-1)),
        "mae": _tensor_mean(torch.mean(torch.abs(diff), dim=-1)),
        "cosine": _tensor_mean(cosine),
        "pred_abs_mean": _tensor_mean(torch.mean(torch.abs(pred), dim=-1)),
        "pred_abs_max": _tensor_max(torch.abs(pred)),
        "target_abs_mean": _tensor_mean(torch.mean(torch.abs(target), dim=-1)),
        "target_abs_max": _tensor_max(torch.abs(target)),
    }


def _per_dim_action_metrics(torch: Any, pred: Any, target: Any, labels: list[str]) -> list[dict[str, Any]]:
    diff = pred - target
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "dim": int(idx),
                "label": str(label),
                "pred_signed_mean": _tensor_mean(pred[:, idx]),
                "target_signed_mean": _tensor_mean(target[:, idx]),
                "pred_abs_mean": _tensor_mean(torch.abs(pred[:, idx])),
                "target_abs_mean": _tensor_mean(torch.abs(target[:, idx])),
                "abs_gap_mean": _tensor_mean(torch.abs(diff[:, idx])),
                "mse": _tensor_mean(diff[:, idx] * diff[:, idx]),
                "sign_mismatch_rate": _tensor_mean(((pred[:, idx] * target[:, idx]) < 0.0).float()),
            }
        )
    return rows


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D290 Closed-Loop Recovery Probe",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- actor checkpoint: `{summary['actor_checkpoint']}`",
        f"- reset pose source: `{summary['reset_pose_source']}`",
        f"- selected episodes: `{summary['selected_episode_min']}..{summary['selected_episode_max']}` / `{summary['selected_episode_unique_count']}`",
        f"- samples: `{summary['sample_count']}`",
        f"- actor rollout contact/useful/reaction: `{summary['actor_contact_seen_rate']}` / `{summary['actor_useful_seen_rate']}` / `{summary['actor_reaction_seen_rate']}`",
        f"- actor rollout overshoot: `{summary['actor_overshoot_seen_rate']}`",
        f"- max XY mean/max: `{summary['actor_max_disp_xy_mean_m']}` / `{summary['actor_max_disp_xy_max_m']}`",
        f"- actor-vs-recovery MSE/MAE/cosine: `{summary['actor_recovery_mse_mean']}` / `{summary['actor_recovery_mae_mean']}` / `{summary['actor_recovery_cosine_mean']}`",
        f"- actor-vs-recorded MSE/MAE/cosine: `{summary['actor_recorded_metrics']['mse']}` / `{summary['actor_recorded_metrics']['mae']}` / `{summary['actor_recorded_metrics']['cosine']}`",
        f"- actor action abs mean/max: `{summary['actor_action_abs_mean']}` / `{summary['actor_action_abs_max']}`",
        f"- recovery action abs mean/max: `{summary['recovery_action_abs_mean']}` / `{summary['recovery_action_abs_max']}`",
        f"- recovery clip rate mean/max: `{summary['recovery_action_clip_rate_mean']}` / `{summary['recovery_action_clip_rate_max']}`",
        f"- dataset path: `{summary['dataset_out']}`",
        f"- per-env CSV: `{summary['out_env_csv']}`",
        f"- per-step/env action CSV: `{summary['out_step_env_csv']}`",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in summary["issues"]) if summary["issues"] else lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.",
            "A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset_out", type=Path, default=None)
    parser.add_argument("--out_env_csv", type=Path, default=None)
    parser.add_argument("--out_step_env_csv", type=Path, default=None)
    parser.add_argument("--out_reset_alignment_csv", type=Path, default=None)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29051)
    parser.add_argument("--steps", type=int, default=580)
    parser.add_argument("--hold_steps", type=int, default=3)
    parser.add_argument("--env_sample_every", type=int, default=1)
    parser.add_argument("--episode_min", type=int, default=None)
    parser.add_argument("--episode_max", type=int, default=None)
    parser.add_argument("--episode_indices", type=str, default="")
    parser.add_argument("--reset_pose_source", choices=("manual", "env_hook"), default="manual")
    parser.add_argument("--env_hook_force_second_reset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env_hook_warmup_action_source", choices=("zero", "policy"), default="zero")
    parser.add_argument("--post_reset_scene_sync", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--exec_action_clip_abs", type=float, default=1.0)
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument(
        "--rl_action_mode",
        choices=("joint_delta", "candidate6_diffik_residual_joint", "candidate8_diffik_target_residual"),
        default="joint_delta",
    )
    parser.add_argument("--policy_action_space", type=int, default=None)
    parser.add_argument("--cube_size_m", type=float, default=None)
    parser.add_argument("--cube_mass_kg", type=float, default=None)
    parser.add_argument("--cube_static_friction", type=float, default=None)
    parser.add_argument("--cube_dynamic_friction", type=float, default=None)
    parser.add_argument("--tap_stop_after_useful_seen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tap_stop_after_disp_m", type=float, default=0.0)
    parser.add_argument("--tap_contact_slowdown_use_proxy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tap_useful_terminate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tap_overshoot_terminate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--exec_source",
        choices=("actor", "zero", "tap_push_primitive", "env_tap_push_primitive"),
        default="actor",
        help=(
            "Action source for env.step. actor uses the frozen policy action. "
            "zero sends all-zero actions. "
            "tap_push_primitive ignores the actor for execution and writes a "
            "bounded DiffIK tool/object push target through the env override. "
            "env_tap_push_primitive enables the same contract inside the env "
            "with rl_action_mode=tap_push_primitive."
        ),
    )
    parser.add_argument("--primitive_goal_disp_m", type=float, default=0.003)
    parser.add_argument("--primitive_push_steps", type=int, default=220)
    parser.add_argument("--primitive_speed_stop_mps", type=float, default=0.200)
    parser.add_argument("--primitive_speed_stop_min_disp_m", type=float, default=0.001)
    parser.add_argument("--primitive_diffik_step_clip_rad", type=float, default=0.010)
    parser.add_argument("--primitive_cube_pose_noise_xy_m", type=float, default=0.0)
    parser.add_argument("--policy_cube_pose_noise_xy_m", type=float, default=0.0)
    parser.add_argument("--candidate8_diffik_target_residual_forward_m", type=float, default=None)
    parser.add_argument("--candidate8_diffik_target_residual_lateral_m", type=float, default=None)
    parser.add_argument("--candidate8_diffik_target_residual_height_m", type=float, default=None)
    parser.add_argument("--candidate8_hybrid_stop_after_useful", action="store_true")
    parser.add_argument(
        "--primitive_target_path_mode",
        choices=("near_face_goal", "legacy_far_face_through"),
        default="near_face_goal",
    )
    parser.add_argument(
        "--primitive_cube_reference_mode",
        choices=("start_pose", "current_pose"),
        default="start_pose",
    )
    parser.add_argument(
        "--primitive_target_base_mode",
        choices=("actual_joint_pos", "previous_joint_target"),
        default="actual_joint_pos",
    )
    parser.add_argument(
        "--action_governor_mode",
        choices=("off", "predict_stop", "predict_brake"),
        default="off",
        help=(
            "Default-off non-PPO action governor. predict_stop zeros actions before "
            "projected displacement exceeds the target; predict_brake applies a short "
            "opposite-action brake before holding zero."
        ),
    )
    parser.add_argument("--action_governor_target_disp_m", type=float, default=0.003)
    parser.add_argument("--action_governor_predict_horizon_s", type=float, default=0.060)
    parser.add_argument("--action_governor_speed_stop_mps", type=float, default=0.060)
    parser.add_argument("--action_governor_min_contact_steps", type=int, default=1)
    parser.add_argument("--action_governor_push_scale", type=float, default=1.0)
    parser.add_argument("--action_governor_brake_scale", type=float, default=0.35)
    parser.add_argument("--action_governor_brake_steps", type=int, default=2)
    parser.add_argument(
        "--env_action_governor_mode",
        choices=("off", "predict_stop", "predict_brake"),
        default="off",
        help=(
            "Use the env runtime action governor contract instead of the local "
            "diagnostic governor. Reuses the action_governor_* parameters."
        ),
    )
    parser.add_argument("--max_recovery_clip_rate_mean", type=float, default=0.80)
    parser.add_argument("--max_actor_recovery_mse_mean", type=float, default=0.80)
    parser.add_argument("--artifact_tag", type=str, default="d290_closed_loop_recovery_probe")
    args = parser.parse_args()

    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if int(args.hold_steps) <= 0:
        raise ValueError("--hold_steps must be positive")
    if int(args.env_sample_every) <= 0:
        raise ValueError("--env_sample_every must be positive")
    if not (0.0 < float(args.exec_action_clip_abs) <= 1.0):
        raise ValueError("--exec_action_clip_abs must be in (0, 1]")
    if args.cube_size_m is not None and float(args.cube_size_m) <= 0.0:
        raise ValueError("--cube_size_m must be positive")
    if args.cube_mass_kg is not None and float(args.cube_mass_kg) <= 0.0:
        raise ValueError("--cube_mass_kg must be positive")
    if args.cube_static_friction is not None and float(args.cube_static_friction) < 0.0:
        raise ValueError("--cube_static_friction must be non-negative")
    if args.cube_dynamic_friction is not None and float(args.cube_dynamic_friction) < 0.0:
        raise ValueError("--cube_dynamic_friction must be non-negative")
    if float(args.action_governor_target_disp_m) <= 0.0:
        raise ValueError("--action_governor_target_disp_m must be positive")
    if float(args.action_governor_predict_horizon_s) < 0.0:
        raise ValueError("--action_governor_predict_horizon_s must be non-negative")
    if float(args.action_governor_speed_stop_mps) < 0.0:
        raise ValueError("--action_governor_speed_stop_mps must be non-negative")
    if int(args.action_governor_min_contact_steps) < 0:
        raise ValueError("--action_governor_min_contact_steps must be non-negative")
    if not (0.0 <= float(args.action_governor_push_scale) <= 1.0):
        raise ValueError("--action_governor_push_scale must be in [0, 1]")
    if not (0.0 <= float(args.action_governor_brake_scale) <= 1.0):
        raise ValueError("--action_governor_brake_scale must be in [0, 1]")
    if int(args.action_governor_brake_steps) < 0:
        raise ValueError("--action_governor_brake_steps must be non-negative")
    if str(args.action_governor_mode) != "off" and str(args.env_action_governor_mode) != "off":
        raise ValueError("local and env action governors cannot be enabled together")
    if float(args.primitive_goal_disp_m) <= 0.0:
        raise ValueError("--primitive_goal_disp_m must be positive")
    if int(args.primitive_push_steps) <= 0:
        raise ValueError("--primitive_push_steps must be positive")
    if float(args.primitive_speed_stop_mps) < 0.0:
        raise ValueError("--primitive_speed_stop_mps must be non-negative")
    if float(args.primitive_speed_stop_min_disp_m) < 0.0:
        raise ValueError("--primitive_speed_stop_min_disp_m must be non-negative")
    if float(args.primitive_diffik_step_clip_rad) <= 0.0:
        raise ValueError("--primitive_diffik_step_clip_rad must be positive")
    if float(args.primitive_cube_pose_noise_xy_m) < 0.0:
        raise ValueError("--primitive_cube_pose_noise_xy_m must be non-negative")
    if float(args.policy_cube_pose_noise_xy_m) < 0.0:
        raise ValueError("--policy_cube_pose_noise_xy_m must be non-negative")
    if args.policy_action_space is not None and int(args.policy_action_space) <= 0:
        raise ValueError("--policy_action_space must be positive")
    if str(args.rl_action_mode) == "candidate8_diffik_target_residual":
        requested_action_space = 3 if args.policy_action_space is None else int(args.policy_action_space)
        if requested_action_space != 3:
            raise ValueError("candidate8_diffik_target_residual requires --policy_action_space 3")
    primitive_exec_source = str(args.exec_source) in {"tap_push_primitive", "env_tap_push_primitive"}
    if primitive_exec_source and (
        str(args.action_governor_mode) != "off" or str(args.env_action_governor_mode) != "off"
    ):
        raise ValueError("tap_push_primitive has its own stop contract; disable action governors")
    for path in (args.actor_checkpoint, args.teacher_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg, TABLE_Z
    from sim_scripts.cube10cm_top_view_d256_action_replay_probe import load_episode_rows
    from sim_scripts.cube10cm_top_view_teacher_rollout_probe import apply_d256_pose_reset

    torch.manual_seed(int(args.seed))
    episode_indices = (
        [int(part) for part in str(args.episode_indices).split(",") if part.strip()]
        if str(args.episode_indices).strip()
        else None
    )
    if args.reset_pose_source == "env_hook" and episode_indices is not None:
        raise ValueError("--reset_pose_source env_hook does not support --episode_indices; use episode_min/max")
    selected_episodes, episode_rows = load_episode_rows(
        args.teacher_csv,
        int(args.num_envs),
        args.episode_min,
        args.episode_max,
        episode_indices,
    )

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    if args.cube_size_m is not None:
        cube_size_m = float(args.cube_size_m)
        env_cfg.cube_size_x_m = cube_size_m
        env_cfg.cube_size_y_m = cube_size_m
        env_cfg.cube_size_z_m = cube_size_m
        env_cfg.sponge.spawn.size = (cube_size_m, cube_size_m, cube_size_m)
        env_cfg.sponge.init_state.pos = (0.30, 0.00, TABLE_Z + 0.5 * cube_size_m)
    if args.cube_mass_kg is not None:
        env_cfg.sponge.spawn.mass_props.mass = float(args.cube_mass_kg)
    if args.cube_static_friction is not None:
        env_cfg.sponge.spawn.physics_material.static_friction = float(args.cube_static_friction)
    if args.cube_dynamic_friction is not None:
        env_cfg.sponge.spawn.physics_material.dynamic_friction = float(args.cube_dynamic_friction)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    if args.reset_pose_source == "env_hook":
        env_cfg.d256_reset_csv_path = str(args.teacher_csv)
        env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
        env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)
        if args.episode_min is not None:
            env_cfg.d256_reset_episode_min = int(args.episode_min)
        if args.episode_max is not None:
            env_cfg.d256_reset_episode_max = int(args.episode_max)
    env_cfg.action_scale = float(args.action_scale)
    env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.rl_action_mode = str(args.rl_action_mode)
    if args.policy_action_space is not None:
        env_cfg.action_space = int(args.policy_action_space)
    elif str(args.rl_action_mode) == "candidate8_diffik_target_residual":
        env_cfg.action_space = 3
    if args.candidate8_diffik_target_residual_forward_m is not None:
        env_cfg.candidate8_diffik_target_residual_forward_m = float(
            args.candidate8_diffik_target_residual_forward_m
        )
    if args.candidate8_diffik_target_residual_lateral_m is not None:
        env_cfg.candidate8_diffik_target_residual_lateral_m = float(
            args.candidate8_diffik_target_residual_lateral_m
        )
    if args.candidate8_diffik_target_residual_height_m is not None:
        env_cfg.candidate8_diffik_target_residual_height_m = float(
            args.candidate8_diffik_target_residual_height_m
        )
    env_cfg.candidate8_hybrid_stop_after_useful = bool(args.candidate8_hybrid_stop_after_useful)
    env_cfg.tap_stop_after_useful_seen = bool(args.tap_stop_after_useful_seen)
    env_cfg.tap_stop_after_disp_m = float(args.tap_stop_after_disp_m)
    env_cfg.tap_contact_slowdown_use_proxy = bool(args.tap_contact_slowdown_use_proxy)
    env_cfg.tap_useful_terminate = bool(args.tap_useful_terminate)
    env_cfg.tap_overshoot_terminate = bool(args.tap_overshoot_terminate)
    env_cfg.tap_action_governor_mode = str(args.env_action_governor_mode)
    env_cfg.tap_action_governor_target_disp_m = float(args.action_governor_target_disp_m)
    env_cfg.tap_action_governor_predict_horizon_s = float(args.action_governor_predict_horizon_s)
    env_cfg.tap_action_governor_speed_stop_mps = float(args.action_governor_speed_stop_mps)
    env_cfg.tap_action_governor_min_contact_steps = int(args.action_governor_min_contact_steps)
    env_cfg.tap_action_governor_push_scale = float(args.action_governor_push_scale)
    env_cfg.tap_action_governor_brake_scale = float(args.action_governor_brake_scale)
    env_cfg.tap_action_governor_brake_steps = int(args.action_governor_brake_steps)
    env_cfg.policy_cube_pose_noise_xy_m = float(args.policy_cube_pose_noise_xy_m)
    if primitive_exec_source:
        env_cfg.candidate6_diffik_goal_push_m = float(args.primitive_goal_disp_m)
        env_cfg.candidate6_diffik_push_steps = int(args.primitive_push_steps)
        env_cfg.candidate6_diffik_step_clip_rad = float(args.primitive_diffik_step_clip_rad)
        env_cfg.candidate6_diffik_target_base_mode = str(args.primitive_target_base_mode)
        env_cfg.candidate6_diffik_target_path_mode = str(args.primitive_target_path_mode)
        env_cfg.candidate6_diffik_cube_reference_mode = str(args.primitive_cube_reference_mode)
        env_cfg.candidate6_diffik_cube_pose_noise_xy_m = float(args.primitive_cube_pose_noise_xy_m)
        env_cfg.candidate6_diffik_hold_after_tap_success = False
        env_cfg.tap_push_primitive_stop_disp_m = float(args.primitive_goal_disp_m)
        env_cfg.tap_push_primitive_speed_stop_mps = float(args.primitive_speed_stop_mps)
        env_cfg.tap_push_primitive_speed_stop_min_disp_m = float(args.primitive_speed_stop_min_disp_m)
        env_cfg.tap_push_primitive_stop_on_overshoot = True
    if str(args.exec_source) == "env_tap_push_primitive":
        env_cfg.rl_action_mode = "tap_push_primitive"
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)
    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[closed-loop-recovery] "
        f"env_id={env_id} training=PPO_NO actor={args.actor_checkpoint} "
        f"reset_pose_source={args.reset_pose_source} "
        f"episodes={selected_episodes[0]}..{selected_episodes[-1]} steps={args.steps}",
        flush=True,
    )

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    if int(args.steps) >= int(inner.max_episode_length) - 1:
        raise ValueError(
            f"--steps {args.steps} would hit env truncation/reset; "
            f"use <= {int(inner.max_episode_length) - 2}"
        )

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.actor_checkpoint), load_optimizer=False, map_location=inner.device)
    inference_policy = runner.get_inference_policy(device=inner.device)
    actor_critic = runner.alg.policy

    device = inner.device
    if args.reset_pose_source == "manual":
        env.reset()
        reset_info = apply_d256_pose_reset(inner, [rows[0] for rows in episode_rows])
        reset_info = {"reset_pose_source": "manual", **reset_info}
        obs = env.get_observations()
    else:
        env.reset()
        obs = env.get_observations()
        if bool(args.env_hook_force_second_reset):
            inner.episode_length_buf[:] = inner.max_episode_length
            with torch.inference_mode():
                if str(args.env_hook_warmup_action_source) == "policy":
                    warmup_actions = inference_policy(obs)
                else:
                    warmup_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
                obs, _, _, _ = env.step(warmup_actions)
        runtime_episodes = [
            int(round(float(value)))
            for value in inner._last_d256_reset_episode_index.detach().cpu().tolist()
        ]
        if any(ep < 0 for ep in runtime_episodes):
            raise RuntimeError(f"D256 env reset hook did not activate for all envs: {runtime_episodes}")
        selected_episodes, episode_rows = load_episode_rows(
            args.teacher_csv,
            int(args.num_envs),
            episode_indices=runtime_episodes,
        )
        reset_info = {
            "reset_pose_source": "env_hook",
            "row_count": int(len(runtime_episodes)),
            "episode_min": int(min(runtime_episodes)),
            "episode_max": int(max(runtime_episodes)),
            "episode_unique_count": int(len(set(runtime_episodes))),
            "runtime_episodes": runtime_episodes,
            "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
            "d256_reset_frame_index": int(args.d256_reset_frame_index),
            "force_second_reset": bool(args.env_hook_force_second_reset),
            "warmup_action_source": str(args.env_hook_warmup_action_source),
        }
        if bool(args.post_reset_scene_sync):
            inner.scene.write_data_to_sim()
            inner.scene.update(inner.sim.get_physics_dt())
            inner._compute_intermediate_values()
            obs = env.get_observations()
            reset_info["post_reset_scene_sync"] = True
        else:
            reset_info["post_reset_scene_sync"] = False
    min_len = min(len(rows) for rows in episode_rows)
    exec_clip = float(args.exec_action_clip_abs)

    inner._compute_intermediate_values()
    reset_terms = inner._tap_terms()
    env_origins = inner.scene.env_origins[torch.arange(inner.num_envs, device=device, dtype=torch.long)]
    expected_arm0 = torch.tensor(
        [[float(rows[0][f"arm_joint_{idx}_rad"]) for idx in range(5)] for rows in episode_rows],
        device=device,
        dtype=torch.float32,
    )
    expected_cube_local0 = torch.tensor(
        [
            [float(rows[0]["cube_local_x_m"]), float(rows[0]["cube_local_y_m"]), float(rows[0]["cube_local_z_m"])]
            for rows in episode_rows
        ],
        device=device,
        dtype=torch.float32,
    )
    expected_cube_world0 = env_origins + expected_cube_local0
    actual_arm0 = inner._robot.data.joint_pos[:, inner._bc_arm_joint_ids].detach()
    actual_arm_vel0 = inner._robot.data.joint_vel[:, inner._bc_arm_joint_ids].detach()
    target_arm0 = inner.robot_dof_targets[:, inner._bc_arm_joint_ids].detach()
    actual_cube0 = inner._sponge_pos_w.detach()
    actual_cube_lin_vel0 = inner._sponge.data.root_lin_vel_w.detach()
    actual_cube_ang_vel0 = inner._sponge.data.root_ang_vel_w.detach()
    cube_start0 = inner._cube_start_w.detach()
    cube_err_xy0 = torch.norm(actual_cube0[:, 0:2] - expected_cube_world0[:, 0:2], p=2, dim=-1)
    cube_start_err_xy0 = torch.norm(cube_start0[:, 0:2] - expected_cube_world0[:, 0:2], p=2, dim=-1)
    cube_actual_start_xy0 = torch.norm(actual_cube0[:, 0:2] - cube_start0[:, 0:2], p=2, dim=-1)
    arm_err0 = torch.max(torch.abs(actual_arm0 - expected_arm0), dim=-1).values
    arm_vel_abs_max0 = torch.max(torch.abs(actual_arm_vel0), dim=-1).values
    target_err0 = torch.max(torch.abs(target_arm0 - expected_arm0), dim=-1).values
    cube_lin_vel_norm0 = torch.norm(actual_cube_lin_vel0, p=2, dim=-1)
    cube_ang_vel_norm0 = torch.norm(actual_cube_ang_vel0, p=2, dim=-1)
    reset_alignment_rows: list[dict[str, Any]] = []
    for env_i, episode_index in enumerate(selected_episodes):
        reset_alignment_rows.append(
            {
                "env_id": int(env_i),
                "episode_index": int(episode_index),
                "actual_cube_x_m": float(actual_cube0[env_i, 0].detach().cpu().item()),
                "actual_cube_y_m": float(actual_cube0[env_i, 1].detach().cpu().item()),
                "expected_cube_x_m": float(expected_cube_world0[env_i, 0].detach().cpu().item()),
                "expected_cube_y_m": float(expected_cube_world0[env_i, 1].detach().cpu().item()),
                "cube_actual_expected_xy_err_m": float(cube_err_xy0[env_i].detach().cpu().item()),
                "cube_start_expected_xy_err_m": float(cube_start_err_xy0[env_i].detach().cpu().item()),
                "cube_actual_start_xy_err_m": float(cube_actual_start_xy0[env_i].detach().cpu().item()),
                "arm_actual_expected_abs_max_rad": float(arm_err0[env_i].detach().cpu().item()),
                "arm_joint_vel_abs_max_rad_s": float(arm_vel_abs_max0[env_i].detach().cpu().item()),
                "arm_target_expected_abs_max_rad": float(target_err0[env_i].detach().cpu().item()),
                "cube_lin_vel_norm_mps": float(cube_lin_vel_norm0[env_i].detach().cpu().item()),
                "cube_ang_vel_norm_rad_s": float(cube_ang_vel_norm0[env_i].detach().cpu().item()),
                "initial_disp_xy_m": float(reset_terms["disp_xy"][env_i].detach().cpu().item()),
                "initial_disp_along_m": float(reset_terms["disp_along"][env_i].detach().cpu().item()),
                "initial_lateral_abs_m": float(reset_terms["lateral_abs"][env_i].detach().cpu().item()),
                "initial_tcp_cube_dist_m": float(reset_terms["tcp_cube_dist"][env_i].detach().cpu().item()),
                "initial_tap_contact_proxy": int(
                    bool(reset_terms["tap_contact_proxy"][env_i].detach().cpu().item())
                ),
                "initial_tap_contact_face_gap_m": float(
                    reset_terms["tap_contact_face_gap_m"][env_i].detach().cpu().item()
                ),
                "initial_tap_contact_lateral_m": float(
                    reset_terms["tap_contact_lateral_m"][env_i].detach().cpu().item()
                ),
                "initial_tap_contact_vertical_offset_m": float(
                    reset_terms["tap_contact_vertical_offset_m"][env_i].detach().cpu().item()
                ),
            }
        )

    actor_obs_parts = []
    target_parts = []
    actor_action_parts = []
    recorded_action_parts = []
    episode_index_parts = []
    env_index_parts = []
    step_index_parts = []
    row_index_parts = []
    frame_index_parts = []
    step_rows: list[dict[str, Any]] = []
    step_env_rows: list[dict[str, Any]] = []
    recovery_clip_rates: list[float] = []
    recovery_abs_means: list[float] = []
    recovery_abs_maxes: list[float] = []
    actor_abs_means: list[float] = []
    actor_abs_maxes: list[float] = []
    exec_abs_means: list[float] = []
    exec_abs_maxes: list[float] = []
    actor_recovery_mses: list[float] = []
    actor_recovery_maes: list[float] = []
    actor_recovery_cosines: list[float] = []
    actor_recorded_mses: list[float] = []
    actor_recorded_maes: list[float] = []
    actor_recorded_cosines: list[float] = []
    cap_rates: list[float] = []
    max_disp_xy = torch.zeros(inner.num_envs, device=device)
    max_disp_along = torch.full((inner.num_envs,), -math.inf, device=device)
    max_lateral_disp = torch.zeros(inner.num_envs, device=device)
    action_labels = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    if len(action_labels) != int(inner.cfg.action_space):
        action_labels = [f"action_{idx}" for idx in range(int(inner.cfg.action_space))]
    env_ids_tensor = torch.arange(inner.num_envs, device=device, dtype=torch.long)
    selected_episode_tensor = torch.tensor(selected_episodes, device=device, dtype=torch.long)
    governor_enabled = str(args.action_governor_mode) != "off"
    governor_contact_age = torch.zeros(inner.num_envs, device=device, dtype=torch.long)
    governor_stop_latched = torch.zeros(inner.num_envs, device=device, dtype=torch.bool)
    governor_brake_remaining = torch.zeros(inner.num_envs, device=device, dtype=torch.long)
    governor_prev_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
    governor_brake_source_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
    governor_stop_step = torch.full((inner.num_envs,), -1, device=device, dtype=torch.long)
    governor_stop_count_trace: list[float] = []
    governor_brake_count_trace: list[float] = []
    governor_projected_disp_trace: list[float] = []
    local_primitive_enabled = str(args.exec_source) == "tap_push_primitive"
    env_primitive_enabled = str(args.exec_source) == "env_tap_push_primitive"
    primitive_stop_latched = torch.zeros(inner.num_envs, device=device, dtype=torch.bool)
    primitive_stop_step = torch.full((inner.num_envs,), -1, device=device, dtype=torch.long)
    primitive_hold_targets = inner.robot_dof_targets.detach().clone()
    primitive_target_delta_abs_mean_trace: list[float] = []
    primitive_target_delta_abs_max_trace: list[float] = []

    with torch.inference_mode():
        for step in range(int(args.steps)):
            row_idx = min(step // int(args.hold_steps), min_len - 1)
            recorded_delta = torch.tensor(
                [
                    [float(rows[row_idx][f"joint_delta_{idx}_rad"]) for idx in range(5)]
                    for rows in episode_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            target_arm = torch.tensor(
                [
                    [
                        float(rows[row_idx][f"arm_joint_{idx}_rad"])
                        + float(rows[row_idx][f"joint_delta_{idx}_rad"])
                        for idx in range(5)
                    ]
                    for rows in episode_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            current_arm = inner._robot.data.joint_pos[:, inner._bc_arm_joint_ids]
            needed_delta = target_arm - current_arm
            raw_arm_actions = needed_delta / max(float(inner.cfg.action_scale), 1.0e-6)
            recovery_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
            recorded_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
            if int(inner.cfg.action_space) >= int(inner._robot.num_joints):
                recovery_actions[:, inner._bc_arm_joint_ids] = torch.clamp(raw_arm_actions, -1.0, 1.0)
                recovery_actions[:, inner.gripper_joint_idx] = 0.0
                recorded_actions[:, inner._bc_arm_joint_ids] = torch.clamp(
                    recorded_delta / max(float(inner.cfg.action_scale), 1.0e-6),
                    -1.0,
                    1.0,
                )
                recorded_actions[:, inner.gripper_joint_idx] = 0.0
            frame_indices = torch.tensor(
                [int(float(rows[row_idx]["frame_index_t"])) for rows in episode_rows],
                device=device,
                dtype=torch.long,
            )

            actor_actions_raw = inference_policy(obs)
            actor_actions = torch.clamp(actor_actions_raw, -exec_clip, exec_clip)
            actor_obs = actor_critic.get_actor_obs(obs).detach().clone()
            actor_actions_clamped = torch.clamp(actor_actions_raw, -1.0, 1.0)

            exec_actions = actor_actions
            if str(args.exec_source) == "zero" or env_primitive_enabled:
                exec_actions = torch.zeros_like(actor_actions)
            primitive_stop_mask = torch.zeros(inner.num_envs, device=device, dtype=torch.bool)
            primitive_target_delta_abs_mean = torch.zeros(inner.num_envs, device=device)
            primitive_target_delta_abs_max = torch.zeros(inner.num_envs, device=device)
            if local_primitive_enabled:
                pre_terms = inner._tap_terms()
                primitive_stop_now = (
                    (pre_terms["disp_xy"] >= float(args.primitive_goal_disp_m))
                    | (pre_terms["speed"] >= float(args.primitive_speed_stop_mps))
                    | pre_terms["tap_overshoot_now"]
                )
                primitive_newly_stopped = primitive_stop_now & ~primitive_stop_latched
                joint_pos_now = inner._robot.data.joint_pos
                primitive_hold_targets = torch.where(
                    primitive_newly_stopped.unsqueeze(-1),
                    joint_pos_now.detach(),
                    primitive_hold_targets,
                )
                primitive_stop_step = torch.where(
                    primitive_newly_stopped,
                    torch.full_like(primitive_stop_step, int(step)),
                    primitive_stop_step,
                )
                primitive_stop_latched = primitive_stop_latched | primitive_stop_now
                primitive_targets = inner._candidate6_diffik_base_joint_target()
                primitive_targets = torch.where(
                    primitive_stop_latched.unsqueeze(-1),
                    primitive_hold_targets,
                    primitive_targets,
                )
                primitive_delta = primitive_targets - joint_pos_now
                primitive_target_delta_abs_mean = torch.mean(torch.abs(primitive_delta), dim=-1)
                primitive_target_delta_abs_max = torch.max(torch.abs(primitive_delta), dim=-1).values
                inner._external_joint_targets_override = primitive_targets.detach().clone()
                exec_actions = torch.zeros_like(actor_actions)
                primitive_stop_mask = primitive_stop_latched
            governor_projected_disp = torch.zeros(inner.num_envs, device=device)
            governor_stop_mask = torch.zeros(inner.num_envs, device=device, dtype=torch.bool)
            governor_brake_mask = torch.zeros(inner.num_envs, device=device, dtype=torch.bool)
            if governor_enabled:
                pre_terms = inner._tap_terms()
                governor_contact_age = torch.where(
                    pre_terms["tap_contact_proxy"] | inner._tap_contact_seen,
                    governor_contact_age + 1,
                    governor_contact_age,
                )
                governor_projected_disp = pre_terms["disp_xy"] + pre_terms["speed"] * float(
                    args.action_governor_predict_horizon_s
                )
                can_stop = governor_contact_age >= int(args.action_governor_min_contact_steps)
                stop_now = can_stop & (
                    (pre_terms["disp_xy"] >= float(args.action_governor_target_disp_m))
                    | (governor_projected_disp >= float(args.action_governor_target_disp_m))
                    | (pre_terms["speed"] >= float(args.action_governor_speed_stop_mps))
                )
                newly_stopped = stop_now & ~governor_stop_latched
                governor_stop_step = torch.where(
                    newly_stopped,
                    torch.full_like(governor_stop_step, int(step)),
                    governor_stop_step,
                )
                governor_stop_latched = governor_stop_latched | stop_now
                if str(args.action_governor_mode) == "predict_brake":
                    governor_brake_source_actions = torch.where(
                        newly_stopped.unsqueeze(-1),
                        governor_prev_actions,
                        governor_brake_source_actions,
                    )
                    governor_brake_remaining = torch.where(
                        newly_stopped,
                        torch.full_like(governor_brake_remaining, int(args.action_governor_brake_steps)),
                        governor_brake_remaining,
                    )
                    governor_brake_mask = governor_brake_remaining > 0
                    brake_actions = -float(args.action_governor_brake_scale) * governor_brake_source_actions
                    held_actions = torch.zeros_like(exec_actions)
                    exec_actions = torch.where(governor_stop_latched.unsqueeze(-1), held_actions, exec_actions)
                    exec_actions = torch.where(governor_brake_mask.unsqueeze(-1), brake_actions, exec_actions)
                    governor_brake_remaining = torch.clamp(governor_brake_remaining - governor_brake_mask.long(), min=0)
                else:
                    exec_actions = torch.where(governor_stop_latched.unsqueeze(-1), torch.zeros_like(exec_actions), exec_actions)
                exec_actions = exec_actions * float(args.action_governor_push_scale)
                exec_actions = torch.clamp(exec_actions, -exec_clip, exec_clip)
                governor_stop_mask = governor_stop_latched

            diff = actor_actions_clamped - recovery_actions
            mse = torch.mean(diff * diff, dim=-1)
            mae = torch.mean(torch.abs(diff), dim=-1)
            cosine = F.cosine_similarity(actor_actions_clamped, recovery_actions, dim=-1, eps=1.0e-6)
            recorded_diff = actor_actions_clamped - recorded_actions
            recorded_mse = torch.mean(recorded_diff * recorded_diff, dim=-1)
            recorded_mae = torch.mean(torch.abs(recorded_diff), dim=-1)
            recorded_cosine = F.cosine_similarity(actor_actions_clamped, recorded_actions, dim=-1, eps=1.0e-6)

            actor_obs_parts.append(actor_obs.cpu())
            target_parts.append(recovery_actions.detach().clone().cpu())
            actor_action_parts.append(actor_actions_clamped.detach().clone().cpu())
            recorded_action_parts.append(recorded_actions.detach().clone().cpu())
            episode_index_parts.append(selected_episode_tensor.detach().clone().cpu())
            env_index_parts.append(env_ids_tensor.detach().clone().cpu())
            step_index_parts.append(torch.full((inner.num_envs,), int(step), dtype=torch.long))
            row_index_parts.append(torch.full((inner.num_envs,), int(row_idx), dtype=torch.long))
            frame_index_parts.append(frame_indices.detach().clone().cpu())
            recovery_clip_rates.append(_tensor_mean((torch.abs(raw_arm_actions) >= 1.0 - 1.0e-9).float()))
            recovery_abs_means.append(_tensor_mean(torch.mean(torch.abs(recovery_actions), dim=-1)))
            recovery_abs_maxes.append(_tensor_max(torch.abs(recovery_actions)))
            actor_abs_means.append(_tensor_mean(torch.mean(torch.abs(actor_actions), dim=-1)))
            actor_abs_maxes.append(_tensor_max(torch.abs(actor_actions)))
            exec_abs_means.append(_tensor_mean(torch.mean(torch.abs(exec_actions), dim=-1)))
            exec_abs_maxes.append(_tensor_max(torch.abs(exec_actions)))
            actor_recovery_mses.append(_tensor_mean(mse))
            actor_recovery_maes.append(_tensor_mean(mae))
            actor_recovery_cosines.append(_tensor_mean(cosine))
            actor_recorded_mses.append(_tensor_mean(recorded_mse))
            actor_recorded_maes.append(_tensor_mean(recorded_mae))
            actor_recorded_cosines.append(_tensor_mean(recorded_cosine))
            governor_stop_count_trace.append(_tensor_mean(governor_stop_mask.float()))
            governor_brake_count_trace.append(_tensor_mean(governor_brake_mask.float()))
            governor_projected_disp_trace.append(_tensor_mean(governor_projected_disp))

            obs, rewards, dones, extras = env.step(exec_actions)
            if governor_enabled:
                governor_prev_actions = exec_actions.detach().clone()
            if env_primitive_enabled:
                primitive_stop_latched = inner._tap_push_primitive_stop_latched.detach().clone()
                primitive_stop_step = inner._tap_push_primitive_stop_step.detach().clone()
                primitive_stop_mask = primitive_stop_latched
                primitive_target_delta_abs_mean = inner._last_tap_push_primitive_target_delta_abs_mean.detach().clone()
                primitive_target_delta_abs_max = inner._last_tap_push_primitive_target_delta_abs_max.detach().clone()
            primitive_target_delta_abs_mean_trace.append(_tensor_mean(primitive_target_delta_abs_mean))
            primitive_target_delta_abs_max_trace.append(_tensor_max(primitive_target_delta_abs_max))
            inner._compute_intermediate_values()
            terms = inner._tap_terms()
            max_disp_xy = torch.maximum(max_disp_xy, terms["disp_xy"].detach())
            max_disp_along = torch.maximum(max_disp_along, terms["disp_along"].detach())
            lateral_disp = torch.sqrt(
                torch.clamp(
                    terms["disp_xy"].detach() ** 2 - torch.clamp(terms["disp_along"].detach(), min=0.0) ** 2,
                    min=0.0,
                )
            )
            max_lateral_disp = torch.maximum(max_lateral_disp, lateral_disp)
            cap_rates.append(_tensor_mean(inner._last_joint_delta_cap_rate))

            if args.out_step_env_csv is not None and (
                step == 0 or step % int(args.env_sample_every) == 0 or step + 1 == int(args.steps)
            ):
                useful_min_disp_m = max(float(getattr(inner.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
                useful_seen_step = (
                    inner._tap_contact_seen
                    & inner._tap_reaction_seen
                    & (inner._tap_max_disp_xy >= useful_min_disp_m)
                    & ~inner._tap_overshoot_seen
                )
                actor_cpu = actor_actions_clamped.detach().cpu()
                exec_cpu = exec_actions.detach().cpu()
                recorded_cpu = recorded_actions.detach().cpu()
                recovery_cpu = recovery_actions.detach().cpu()
                for env_i in range(int(inner.num_envs)):
                    row = {
                        "step": int(step),
                        "row_idx": int(row_idx),
                        "frame_index_t": int(frame_indices[env_i].detach().cpu().item()),
                        "env_id": int(env_i),
                        "episode_index": int(selected_episodes[env_i]),
                        "actor_recorded_mse": float(recorded_mse[env_i].detach().cpu().item()),
                        "actor_recorded_mae": float(recorded_mae[env_i].detach().cpu().item()),
                        "actor_recorded_cosine": float(recorded_cosine[env_i].detach().cpu().item()),
                        "actor_recovery_mse": float(mse[env_i].detach().cpu().item()),
                        "actor_recovery_mae": float(mae[env_i].detach().cpu().item()),
                        "actor_recovery_cosine": float(cosine[env_i].detach().cpu().item()),
                        "disp_along_m": float(terms["disp_along"][env_i].detach().cpu().item()),
                        "disp_xy_m": float(terms["disp_xy"][env_i].detach().cpu().item()),
                        "lateral_disp_m": float(lateral_disp[env_i].detach().cpu().item()),
                        "speed_mps": float(terms["speed"][env_i].detach().cpu().item()),
                        "tcp_cube_dist_m": float(terms["tcp_cube_dist"][env_i].detach().cpu().item()),
                        "tap_contact_face_gap_m": float(
                            terms["tap_contact_face_gap_m"][env_i].detach().cpu().item()
                        ),
                        "tap_contact_lateral_m": float(
                            terms["tap_contact_lateral_m"][env_i].detach().cpu().item()
                        ),
                        "tap_contact_vertical_offset_m": float(
                            terms["tap_contact_vertical_offset_m"][env_i].detach().cpu().item()
                        ),
                        "tap_contact_proxy_now": int(bool(terms["tap_contact_proxy"][env_i].detach().cpu().item())),
                        "tap_reaction_now": int(bool(terms["tap_reaction_now"][env_i].detach().cpu().item())),
                        "tap_overshoot_now": int(bool(terms["tap_overshoot_now"][env_i].detach().cpu().item())),
                        "tap_contact_seen": int(bool(inner._tap_contact_seen[env_i].detach().cpu().item())),
                        "tap_reaction_seen": int(bool(inner._tap_reaction_seen[env_i].detach().cpu().item())),
                        "tap_useful_seen": int(bool(useful_seen_step[env_i].detach().cpu().item())),
                        "tap_overshoot_seen": int(bool(inner._tap_overshoot_seen[env_i].detach().cpu().item())),
                        "joint_delta_cap_rate": float(inner._last_joint_delta_cap_rate[env_i].detach().cpu().item()),
                        "exec_action_abs_mean": float(torch.mean(torch.abs(exec_cpu[env_i])).item()),
                        "exec_action_abs_max": float(torch.max(torch.abs(exec_cpu[env_i])).item()),
                        "governor_stop_latched": int(bool(governor_stop_mask[env_i].detach().cpu().item())),
                        "governor_brake_active": int(bool(governor_brake_mask[env_i].detach().cpu().item())),
                        "governor_contact_age_steps": int(governor_contact_age[env_i].detach().cpu().item()),
                        "governor_projected_disp_m": float(governor_projected_disp[env_i].detach().cpu().item()),
                        "primitive_stop_latched": int(bool(primitive_stop_mask[env_i].detach().cpu().item())),
                        "primitive_stop_step": int(primitive_stop_step[env_i].detach().cpu().item()),
                        "primitive_target_delta_abs_mean": float(
                            primitive_target_delta_abs_mean[env_i].detach().cpu().item()
                        ),
                        "primitive_target_delta_abs_max": float(
                            primitive_target_delta_abs_max[env_i].detach().cpu().item()
                        ),
                        "env_governor_stop_latched": int(
                            bool(inner._last_tap_action_governor_stop_latched[env_i].detach().cpu().item())
                        ),
                        "env_governor_brake_active": int(
                            bool(inner._last_tap_action_governor_brake_active[env_i].detach().cpu().item())
                        ),
                        "env_governor_contact_age_steps": float(
                            inner._last_tap_action_governor_contact_age[env_i].detach().cpu().item()
                        ),
                        "env_governor_projected_disp_m": float(
                            inner._last_tap_action_governor_projected_disp[env_i].detach().cpu().item()
                        ),
                        "env_governor_stop_step": int(
                            inner._tap_action_governor_stop_step[env_i].detach().cpu().item()
                        ),
                        "candidate8_hybrid_stop_latched": int(
                            bool(inner._last_candidate8_hybrid_stop_latched[env_i].detach().cpu().item())
                        ),
                        "candidate8_hybrid_stop_step": int(
                            inner._candidate8_hybrid_stop_step[env_i].detach().cpu().item()
                        ),
                    }
                    for dim, label in enumerate(action_labels):
                        row[f"actor_{label}"] = float(actor_cpu[env_i, dim].item())
                        row[f"exec_{label}"] = float(exec_cpu[env_i, dim].item())
                        row[f"recorded_{label}"] = float(recorded_cpu[env_i, dim].item())
                        row[f"recovery_{label}"] = float(recovery_cpu[env_i, dim].item())
                        row[f"actor_minus_recorded_{label}"] = float(actor_cpu[env_i, dim].item() - recorded_cpu[env_i, dim].item())
                        row[f"actor_minus_recovery_{label}"] = float(actor_cpu[env_i, dim].item() - recovery_cpu[env_i, dim].item())
                    step_env_rows.append(row)

            if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(args.steps):
                useful_min_disp_m = max(float(getattr(inner.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
                useful_seen_step = (
                    inner._tap_contact_seen
                    & inner._tap_reaction_seen
                    & (inner._tap_max_disp_xy >= useful_min_disp_m)
                    & ~inner._tap_overshoot_seen
                )
                step_rows.append(
                    {
                        "step": int(step),
                        "row_idx": int(row_idx),
                        "actor_recovery_mse": actor_recovery_mses[-1],
                        "actor_recovery_mae": actor_recovery_maes[-1],
                        "actor_recovery_cosine": actor_recovery_cosines[-1],
                        "actor_recorded_mse": actor_recorded_mses[-1],
                        "actor_recorded_mae": actor_recorded_maes[-1],
                        "actor_recorded_cosine": actor_recorded_cosines[-1],
                        "actor_action_abs_mean": actor_abs_means[-1],
                        "actor_action_abs_max": actor_abs_maxes[-1],
                        "exec_action_abs_mean": exec_abs_means[-1],
                        "exec_action_abs_max": exec_abs_maxes[-1],
                        "recovery_action_abs_mean": recovery_abs_means[-1],
                        "recovery_action_abs_max": recovery_abs_maxes[-1],
                        "recovery_action_clip_rate": recovery_clip_rates[-1],
                        "joint_delta_cap_rate": cap_rates[-1],
                        "governor_stop_latched_rate": governor_stop_count_trace[-1],
                        "governor_brake_active_rate": governor_brake_count_trace[-1],
                        "governor_projected_disp_mean_m": governor_projected_disp_trace[-1],
                        "primitive_stop_latched_rate": _tensor_mean(primitive_stop_latched.float()),
                        "primitive_target_delta_abs_mean": primitive_target_delta_abs_mean_trace[-1],
                        "primitive_target_delta_abs_max": primitive_target_delta_abs_max_trace[-1],
                        "env_governor_stop_latched_rate": _tensor_mean(
                            inner._last_tap_action_governor_stop_latched
                        ),
                        "env_governor_brake_active_rate": _tensor_mean(
                            inner._last_tap_action_governor_brake_active
                        ),
                        "env_governor_projected_disp_mean_m": _tensor_mean(
                            inner._last_tap_action_governor_projected_disp
                        ),
                        "env_governor_contact_age_steps_mean": _tensor_mean(
                            inner._last_tap_action_governor_contact_age
                        ),
                        "contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
                        "reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
                        "useful_seen_rate": _tensor_mean(useful_seen_step.float()),
                        "overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
                        "max_disp_xy_mean_m": _tensor_mean(max_disp_xy),
                        "max_disp_xy_max_m": _tensor_max(max_disp_xy),
                    }
                )

    useful_min_disp_m = max(float(getattr(inner.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
    useful_seen = (
        inner._tap_contact_seen
        & inner._tap_reaction_seen
        & (inner._tap_max_disp_xy >= useful_min_disp_m)
        & ~inner._tap_overshoot_seen
    )
    actor_obs_all = torch.cat(actor_obs_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    actor_action_all = torch.cat(actor_action_parts, dim=0)
    recorded_action_all = torch.cat(recorded_action_parts, dim=0)
    episode_index_all = torch.cat(episode_index_parts, dim=0)
    env_index_all = torch.cat(env_index_parts, dim=0)
    step_index_all = torch.cat(step_index_parts, dim=0)
    row_index_all = torch.cat(row_index_parts, dim=0)
    frame_index_all = torch.cat(frame_index_parts, dim=0)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_out = args.dataset_out or out_dir / f"closed_loop_recovery_dataset_{args.artifact_tag}.pt"
    out_json = out_dir / f"closed_loop_recovery_summary_{args.artifact_tag}.json"
    out_md = out_dir / f"closed_loop_recovery_summary_{args.artifact_tag}.md"
    out_csv = out_dir / f"closed_loop_recovery_steps_{args.artifact_tag}.csv"

    issues: list[str] = []
    recovery_clip_mean = _safe_mean(recovery_clip_rates)
    actor_recovery_mse_mean = _safe_mean(actor_recovery_mses)
    actor_recorded_metrics = _action_metrics(torch, actor_action_all, recorded_action_all)
    actor_recovery_metrics = _action_metrics(torch, actor_action_all, target_all)
    if recovery_clip_mean > float(args.max_recovery_clip_rate_mean):
        issues.append(f"recovery clip rate too high: {recovery_clip_mean}")
    if actor_recovery_mse_mean > float(args.max_actor_recovery_mse_mean):
        issues.append(f"actor-vs-recovery MSE too high: {actor_recovery_mse_mean}")
    if (
        not torch.isfinite(actor_obs_all).all().item()
        or not torch.isfinite(target_all).all().item()
        or not torch.isfinite(actor_action_all).all().item()
        or not torch.isfinite(recorded_action_all).all().item()
    ):
        issues.append("non-finite actor_obs/actions/target_actions")

    verdict = "D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION" if not issues else "D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION"
    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "actor_checkpoint": _rel(args.actor_checkpoint),
        "teacher_csv": _rel(args.teacher_csv),
        "dataset_out": _rel(dataset_out),
        "reset_pose_source": str(args.reset_pose_source),
        "env_hook_force_second_reset": bool(args.env_hook_force_second_reset),
        "env_hook_warmup_action_source": str(args.env_hook_warmup_action_source),
        "post_reset_scene_sync": bool(args.post_reset_scene_sync),
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "hold_steps": int(args.hold_steps),
        "sample_count": int(actor_obs_all.shape[0]),
        "selected_episodes": [int(ep) for ep in selected_episodes],
        "selected_episode_min": int(min(selected_episodes)),
        "selected_episode_max": int(max(selected_episodes)),
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "episode_min_filter": int(args.episode_min) if args.episode_min is not None else None,
        "episode_max_filter": int(args.episode_max) if args.episode_max is not None else None,
        "episode_indices_filter": episode_indices,
        "reset_info": reset_info,
        "reset_alignment": {
            "cube_actual_expected_xy_err_mean_m": _tensor_mean(cube_err_xy0),
            "cube_actual_expected_xy_err_max_m": _tensor_max(cube_err_xy0),
            "cube_start_expected_xy_err_mean_m": _tensor_mean(cube_start_err_xy0),
            "cube_start_expected_xy_err_max_m": _tensor_max(cube_start_err_xy0),
            "cube_actual_start_xy_err_mean_m": _tensor_mean(cube_actual_start_xy0),
            "cube_actual_start_xy_err_max_m": _tensor_max(cube_actual_start_xy0),
            "arm_actual_expected_abs_max_mean_rad": _tensor_mean(arm_err0),
            "arm_actual_expected_abs_max_max_rad": _tensor_max(arm_err0),
            "arm_joint_vel_abs_max_mean_rad_s": _tensor_mean(arm_vel_abs_max0),
            "arm_joint_vel_abs_max_max_rad_s": _tensor_max(arm_vel_abs_max0),
            "arm_target_expected_abs_max_mean_rad": _tensor_mean(target_err0),
            "arm_target_expected_abs_max_max_rad": _tensor_max(target_err0),
            "cube_lin_vel_norm_mean_mps": _tensor_mean(cube_lin_vel_norm0),
            "cube_lin_vel_norm_max_mps": _tensor_max(cube_lin_vel_norm0),
            "cube_ang_vel_norm_mean_rad_s": _tensor_mean(cube_ang_vel_norm0),
            "cube_ang_vel_norm_max_rad_s": _tensor_max(cube_ang_vel_norm0),
            "initial_disp_xy_mean_m": _tensor_mean(reset_terms["disp_xy"]),
            "initial_disp_xy_max_m": _tensor_max(reset_terms["disp_xy"]),
            "initial_lateral_abs_mean_m": _tensor_mean(reset_terms["lateral_abs"]),
            "initial_lateral_abs_max_m": _tensor_max(reset_terms["lateral_abs"]),
            "initial_tap_contact_proxy_rate": _tensor_mean(reset_terms["tap_contact_proxy"].float()),
            "initial_tap_contact_face_gap_mean_m": _tensor_mean(reset_terms["tap_contact_face_gap_m"]),
            "initial_tap_contact_face_gap_min_m": float(
                torch.min(reset_terms["tap_contact_face_gap_m"]).detach().cpu().item()
            ),
            "initial_tap_contact_face_gap_max_m": _tensor_max(reset_terms["tap_contact_face_gap_m"]),
            "initial_tap_contact_lateral_mean_m": _tensor_mean(reset_terms["tap_contact_lateral_m"]),
            "initial_tap_contact_vertical_offset_mean_m": _tensor_mean(
                reset_terms["tap_contact_vertical_offset_m"]
            ),
        },
        "action_scale": float(inner.cfg.action_scale),
        "action_smoothing_alpha": float(inner.cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(inner.cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(inner.cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(inner.cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(inner.cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(inner.cfg.joint_delta_reference),
        "exec_action_clip_abs": float(args.exec_action_clip_abs),
        "cube_size_x_m": float(inner.cfg.cube_size_x_m),
        "cube_size_y_m": float(inner.cfg.cube_size_y_m),
        "cube_size_z_m": float(inner.cfg.cube_size_z_m),
        "cube_spawn_size_m": [float(v) for v in getattr(inner.cfg.sponge.spawn, "size", ())],
        "cube_mass_kg": float(getattr(inner.cfg.sponge.spawn.mass_props, "mass", math.nan)),
        "cube_static_friction": float(
            getattr(inner.cfg.sponge.spawn.physics_material, "static_friction", math.nan)
        ),
        "cube_dynamic_friction": float(
            getattr(inner.cfg.sponge.spawn.physics_material, "dynamic_friction", math.nan)
        ),
        "d312_perturbation_overrides": {
            "cube_size_m": float(args.cube_size_m) if args.cube_size_m is not None else None,
            "cube_mass_kg": float(args.cube_mass_kg) if args.cube_mass_kg is not None else None,
            "cube_static_friction": float(args.cube_static_friction)
            if args.cube_static_friction is not None
            else None,
            "cube_dynamic_friction": float(args.cube_dynamic_friction)
            if args.cube_dynamic_friction is not None
            else None,
        },
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "tap_stop_after_useful_seen": bool(args.tap_stop_after_useful_seen),
        "tap_stop_after_disp_m": float(args.tap_stop_after_disp_m),
        "tap_contact_slowdown_use_proxy": bool(args.tap_contact_slowdown_use_proxy),
        "tap_useful_terminate": bool(args.tap_useful_terminate),
        "tap_overshoot_terminate": bool(args.tap_overshoot_terminate),
        "exec_source": str(args.exec_source),
        "env_rl_action_mode": str(getattr(inner.cfg, "rl_action_mode", "joint_delta")),
        "policy_action_space": int(inner.cfg.action_space),
        "primitive_goal_disp_m": float(args.primitive_goal_disp_m),
        "primitive_push_steps": int(args.primitive_push_steps),
        "primitive_speed_stop_mps": float(args.primitive_speed_stop_mps),
        "primitive_speed_stop_min_disp_m": float(args.primitive_speed_stop_min_disp_m),
        "primitive_diffik_step_clip_rad": float(args.primitive_diffik_step_clip_rad),
        "primitive_cube_pose_noise_xy_m": float(args.primitive_cube_pose_noise_xy_m),
        "primitive_cube_pose_noise_abs_mean_m": _tensor_mean(
            torch.abs(inner._candidate6_cube_pose_noise_w_xy)
        ),
        "primitive_cube_pose_noise_abs_max_m": _tensor_max(
            torch.abs(inner._candidate6_cube_pose_noise_w_xy)
        ),
        "policy_cube_pose_noise_xy_m": float(args.policy_cube_pose_noise_xy_m),
        "policy_cube_pose_noise_abs_mean_m": _tensor_mean(
            torch.abs(inner._policy_cube_pose_noise_w_xy)
        ),
        "policy_cube_pose_noise_abs_max_m": _tensor_max(
            torch.abs(inner._policy_cube_pose_noise_w_xy)
        ),
        "candidate8_hybrid_stop_after_useful": bool(inner.cfg.candidate8_hybrid_stop_after_useful),
        "candidate8_hybrid_stop_latched_rate_final": _tensor_mean(
            inner._last_candidate8_hybrid_stop_latched
        ),
        "candidate8_hybrid_stop_step_min": int(
            torch.min(
                inner._candidate8_hybrid_stop_step[inner._candidate8_hybrid_stop_step >= 0]
            )
            .detach()
            .cpu()
            .item()
        )
        if bool((inner._candidate8_hybrid_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "candidate8_hybrid_stop_step_max": int(
            torch.max(
                inner._candidate8_hybrid_stop_step[inner._candidate8_hybrid_stop_step >= 0]
            )
            .detach()
            .cpu()
            .item()
        )
        if bool((inner._candidate8_hybrid_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "primitive_target_path_mode": str(args.primitive_target_path_mode),
        "primitive_cube_reference_mode": str(args.primitive_cube_reference_mode),
        "primitive_target_base_mode": str(args.primitive_target_base_mode),
        "primitive_stop_latched_rate_final": _tensor_mean(primitive_stop_latched.float()),
        "primitive_stop_step_min": int(
            torch.min(primitive_stop_step[primitive_stop_step >= 0]).detach().cpu().item()
        )
        if bool((primitive_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "primitive_stop_step_max": int(
            torch.max(primitive_stop_step[primitive_stop_step >= 0]).detach().cpu().item()
        )
        if bool((primitive_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "primitive_target_delta_abs_mean": _safe_mean(primitive_target_delta_abs_mean_trace),
        "primitive_target_delta_abs_max": max(primitive_target_delta_abs_max_trace)
        if primitive_target_delta_abs_max_trace
        else 0.0,
        "action_governor_mode": str(args.action_governor_mode),
        "action_governor_target_disp_m": float(args.action_governor_target_disp_m),
        "action_governor_predict_horizon_s": float(args.action_governor_predict_horizon_s),
        "action_governor_speed_stop_mps": float(args.action_governor_speed_stop_mps),
        "action_governor_min_contact_steps": int(args.action_governor_min_contact_steps),
        "action_governor_push_scale": float(args.action_governor_push_scale),
        "action_governor_brake_scale": float(args.action_governor_brake_scale),
        "action_governor_brake_steps": int(args.action_governor_brake_steps),
        "action_governor_stop_latched_rate_final": _tensor_mean(governor_stop_latched.float()),
        "action_governor_stop_step_min": int(torch.min(governor_stop_step[governor_stop_step >= 0]).detach().cpu().item())
        if bool((governor_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "action_governor_stop_step_max": int(torch.max(governor_stop_step[governor_stop_step >= 0]).detach().cpu().item())
        if bool((governor_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "env_action_governor_mode": str(args.env_action_governor_mode),
        "env_action_governor_stop_latched_rate_final": _tensor_mean(
            inner._last_tap_action_governor_stop_latched
        ),
        "env_action_governor_brake_active_rate_final": _tensor_mean(
            inner._last_tap_action_governor_brake_active
        ),
        "env_action_governor_projected_disp_mean_m_final": _tensor_mean(
            inner._last_tap_action_governor_projected_disp
        ),
        "env_action_governor_contact_age_steps_mean_final": _tensor_mean(
            inner._last_tap_action_governor_contact_age
        ),
        "env_action_governor_stop_step_min": int(
            torch.min(
                inner._tap_action_governor_stop_step[inner._tap_action_governor_stop_step >= 0]
            )
            .detach()
            .cpu()
            .item()
        )
        if bool((inner._tap_action_governor_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "env_action_governor_stop_step_max": int(
            torch.max(
                inner._tap_action_governor_stop_step[inner._tap_action_governor_stop_step >= 0]
            )
            .detach()
            .cpu()
            .item()
        )
        if bool((inner._tap_action_governor_stop_step >= 0).any().detach().cpu().item())
        else -1,
        "actor_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
        "actor_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
        "actor_useful_seen_rate": _tensor_mean(useful_seen.float()),
        "tap_useful_min_disp_m": useful_min_disp_m,
        "actor_overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
        "actor_max_disp_xy_mean_m": _tensor_mean(max_disp_xy),
        "actor_max_disp_xy_max_m": _tensor_max(max_disp_xy),
        "actor_max_disp_xy_ge_1mm_rate": _tensor_mean((max_disp_xy >= 0.001).float()),
        "actor_low_motion_lt_1mm_rate": _tensor_mean((max_disp_xy < 0.001).float()),
        "actor_max_disp_xy_ge_20mm_rate": _tensor_mean((max_disp_xy >= 0.020).float()),
        "actor_max_disp_along_mean_m": _tensor_mean(max_disp_along),
        "actor_max_disp_along_max_m": _tensor_max(max_disp_along),
        "actor_max_lateral_disp_mean_m": _tensor_mean(max_lateral_disp),
        "actor_max_lateral_disp_max_m": _tensor_max(max_lateral_disp),
        "actor_recovery_mse_mean": actor_recovery_mse_mean,
        "actor_recovery_mae_mean": _safe_mean(actor_recovery_maes),
        "actor_recovery_cosine_mean": _safe_mean(actor_recovery_cosines),
        "actor_recorded_mse_mean": _safe_mean(actor_recorded_mses),
        "actor_recorded_mae_mean": _safe_mean(actor_recorded_maes),
        "actor_recorded_cosine_mean": _safe_mean(actor_recorded_cosines),
        "actor_recorded_metrics": actor_recorded_metrics,
        "actor_recovery_metrics": actor_recovery_metrics,
        "actor_recorded_per_dim": _per_dim_action_metrics(torch, actor_action_all, recorded_action_all, action_labels),
        "actor_recovery_per_dim": _per_dim_action_metrics(torch, actor_action_all, target_all, action_labels),
        "actor_action_abs_mean": _safe_mean(actor_abs_means),
        "actor_action_abs_max": max(actor_abs_maxes) if actor_abs_maxes else 0.0,
        "exec_action_abs_mean": _safe_mean(exec_abs_means),
        "exec_action_abs_max": max(exec_abs_maxes) if exec_abs_maxes else 0.0,
        "recovery_action_abs_mean": _safe_mean(recovery_abs_means),
        "recovery_action_abs_max": max(recovery_abs_maxes) if recovery_abs_maxes else 0.0,
        "recovery_action_clip_rate_mean": recovery_clip_mean,
        "recovery_action_clip_rate_max": max(recovery_clip_rates) if recovery_clip_rates else 0.0,
        "joint_delta_cap_rate_mean": _safe_mean(cap_rates),
        "joint_delta_cap_rate_max": max(cap_rates) if cap_rates else 0.0,
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
        "out_env_csv": _rel(args.out_env_csv) if args.out_env_csv is not None else "",
        "out_step_env_csv": _rel(args.out_step_env_csv) if args.out_step_env_csv is not None else "",
        "out_reset_alignment_csv": _rel(args.out_reset_alignment_csv) if args.out_reset_alignment_csv is not None else "",
        "env_sample_every": int(args.env_sample_every),
    }
    torch.save(
        {
            "actor_obs": actor_obs_all,
            "target_actions": target_all,
            "recovery_actions": target_all,
            "recorded_actions": recorded_action_all,
            "actor_actions": actor_action_all,
            "episode_indices": episode_index_all,
            "env_indices": env_index_all,
            "step_indices": step_index_all,
            "row_indices": row_index_all,
            "frame_indices": frame_index_all,
            "summary": {
                **summary,
                "collection_episode_count": int(args.num_envs),
                "oracle_contact_seen_rate": summary["actor_contact_seen_rate"],
                "oracle_reaction_seen_rate": summary["actor_reaction_seen_rate"],
                "oracle_useful_seen_rate": summary["actor_useful_seen_rate"],
                "oracle_overshoot_seen_rate": summary["actor_overshoot_seen_rate"],
            },
        },
        dataset_out,
    )
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)
    with out_csv.open("w", newline="") as f:
        fieldnames = list(step_rows[0].keys()) if step_rows else ["step"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)
    if args.out_step_env_csv is not None:
        args.out_step_env_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_step_env_csv.open("w", newline="") as f:
            fieldnames = list(step_env_rows[0].keys()) if step_env_rows else ["step"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(step_env_rows)
    if args.out_reset_alignment_csv is not None:
        args.out_reset_alignment_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_reset_alignment_csv.open("w", newline="") as f:
            fieldnames = list(reset_alignment_rows[0].keys()) if reset_alignment_rows else ["env_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reset_alignment_rows)
    if args.out_env_csv is not None:
        env_rows: list[dict[str, Any]] = []
        for env_i, episode_index in enumerate(selected_episodes):
            mask = env_index_all == int(env_i)
            env_actor = actor_action_all[mask]
            env_recorded = recorded_action_all[mask]
            env_recovery = target_all[mask]
            env_rows.append(
                {
                    "env_id": int(env_i),
                    "episode_index": int(episode_index),
                    "tap_contact_seen": int(bool(inner._tap_contact_seen[env_i].detach().cpu().item())),
                    "tap_reaction_seen": int(bool(inner._tap_reaction_seen[env_i].detach().cpu().item())),
                    "tap_useful_seen": int(bool(useful_seen[env_i].detach().cpu().item())),
                    "tap_overshoot_seen": int(bool(inner._tap_overshoot_seen[env_i].detach().cpu().item())),
                    "max_disp_along_m": float(max_disp_along[env_i].detach().cpu().item()),
                    "max_disp_xy_m": float(max_disp_xy[env_i].detach().cpu().item()),
                    "max_lateral_disp_m": float(max_lateral_disp[env_i].detach().cpu().item()),
                    "final_speed_mps": float(terms["speed"][env_i].detach().cpu().item()),
                    "final_tcp_cube_dist_m": float(terms["tcp_cube_dist"][env_i].detach().cpu().item()),
                    "final_tap_contact_face_gap_m": float(
                        terms["tap_contact_face_gap_m"][env_i].detach().cpu().item()
                    ),
                    "final_tap_contact_lateral_m": float(
                        terms["tap_contact_lateral_m"][env_i].detach().cpu().item()
                    ),
                    "final_tap_contact_vertical_offset_m": float(
                        terms["tap_contact_vertical_offset_m"][env_i].detach().cpu().item()
                    ),
                    "final_tap_contact_proxy_now": int(
                        bool(terms["tap_contact_proxy"][env_i].detach().cpu().item())
                    ),
                    "final_tap_reaction_now": int(bool(terms["tap_reaction_now"][env_i].detach().cpu().item())),
                    "final_tap_overshoot_now": int(bool(terms["tap_overshoot_now"][env_i].detach().cpu().item())),
                    "actor_recorded_mse": _action_metrics(torch, env_actor, env_recorded)["mse"],
                    "actor_recorded_mae": _action_metrics(torch, env_actor, env_recorded)["mae"],
                    "actor_recorded_cosine": _action_metrics(torch, env_actor, env_recorded)["cosine"],
                    "actor_recovery_mse": _action_metrics(torch, env_actor, env_recovery)["mse"],
                    "actor_recovery_mae": _action_metrics(torch, env_actor, env_recovery)["mae"],
                    "actor_recovery_cosine": _action_metrics(torch, env_actor, env_recovery)["cosine"],
                    "primitive_stop_step": int(primitive_stop_step[env_i].detach().cpu().item()),
                    "primitive_stop_latched": int(bool(primitive_stop_latched[env_i].detach().cpu().item())),
                    "governor_stop_step": int(governor_stop_step[env_i].detach().cpu().item()),
                    "env_governor_stop_step": int(
                        inner._tap_action_governor_stop_step[env_i].detach().cpu().item()
                    ),
                    "env_governor_stop_latched": int(
                        bool(inner._tap_action_governor_stop_latched[env_i].detach().cpu().item())
                    ),
                    "env_governor_contact_age_steps": float(
                        inner._last_tap_action_governor_contact_age[env_i].detach().cpu().item()
                    ),
                    "env_governor_projected_disp_m": float(
                        inner._last_tap_action_governor_projected_disp[env_i].detach().cpu().item()
                    ),
                    "candidate8_hybrid_stop_step": int(
                        inner._candidate8_hybrid_stop_step[env_i].detach().cpu().item()
                    ),
                    "candidate8_hybrid_stop_latched": int(
                        bool(inner._last_candidate8_hybrid_stop_latched[env_i].detach().cpu().item())
                    ),
                }
            )
        args.out_env_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_env_csv.open("w", newline="") as f:
            fieldnames = list(env_rows[0].keys()) if env_rows else ["env_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(env_rows)

    print(
        "[closed-loop-recovery] SUMMARY "
        f"verdict={verdict} useful={summary['actor_useful_seen_rate']:.6f} "
        f"overshoot={summary['actor_overshoot_seen_rate']:.6f} "
        f"recovery_clip_mean={recovery_clip_mean:.6f} "
        f"actor_recovery_mse={actor_recovery_mse_mean:.6f} dataset={dataset_out}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
