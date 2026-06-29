"""PPO training entry point for the no-attach cube push/tap tasks."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_kind", choices=("push3cm", "tap10cm"), default="push3cm")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--action_smoothing_alpha", type=float, default=None)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=None)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=None)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=None)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default=None)
    parser.add_argument("--fixed_push_dir_x", type=float, default=None)
    parser.add_argument("--fixed_push_dir_y", type=float, default=None)
    parser.add_argument(
        "--tap_contact_proxy_mode",
        choices=("tcp_point", "link5_collision_aabb"),
        default="link5_collision_aabb",
    )
    parser.add_argument("--ik_precontact_clearance_m", type=float, default=None)
    parser.add_argument("--scripted_teacher_blend", type=float, default=None)
    parser.add_argument("--scripted_teacher_horizon_frac", type=float, default=None)
    parser.add_argument("--scripted_teacher_goal_push_m", type=float, default=None)
    parser.add_argument("--bc_teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--bc_teacher_blend", type=float, default=None)
    parser.add_argument("--bc_teacher_imitation_reward_scale", type=float, default=None)
    parser.add_argument("--warm_start_checkpoint_path", type=str, default=None)
    parser.add_argument("--bc_teacher_policy_delta_clip_rad", type=float, default=None)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default=None)
    parser.add_argument("--bc_teacher_posx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=None)
    parser.add_argument(
        "--bc_teacher_phase_timing",
        choices=("episode_scaled", "direct_steps", "linear_episode", "linear_steps"),
        default=None,
    )
    parser.add_argument("--bc_teacher_linear_phase_steps", type=int, default=None)
    parser.add_argument("--d256_reset_csv_path", type=str, default=None)
    parser.add_argument("--d256_reset_frame_index", type=int, default=None)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default=None)
    parser.add_argument("--d256_reset_episode_min", type=int, default=None)
    parser.add_argument("--d256_reset_episode_max", type=int, default=None)
    parser.add_argument("--ik_endpoint_reset", action="store_true")
    parser.add_argument("--cube_success_disp_m", type=float, default=None)
    parser.add_argument("--cube_success_speed_max_mps", type=float, default=None)
    parser.add_argument("--cube_push_target_disp_m", type=float, default=None)
    parser.add_argument("--push_progress_reward_scale", type=float, default=None)
    parser.add_argument("--push_displacement_reward_scale", type=float, default=None)
    parser.add_argument("--low_motion_penalty_scale", type=float, default=None)
    parser.add_argument("--controlled_bonus_scale", type=float, default=None)
    parser.add_argument("--impact_penalty_scale", type=float, default=None)
    parser.add_argument("--reverse_push_penalty_scale", type=float, default=None)
    parser.add_argument("--target_distance_penalty_scale", type=float, default=None)
    parser.add_argument("--lateral_penalty_scale", type=float, default=None)
    parser.add_argument("--overshoot_penalty_scale", type=float, default=None)
    parser.add_argument("--tap_success_terminate", action="store_true")
    parser.add_argument("--tap_overshoot_terminate", action="store_true")
    parser.add_argument("--tap_useful_terminate", action="store_true")
    parser.add_argument("--tap_stop_after_useful_seen", action="store_true")
    parser.add_argument("--tap_stop_after_disp_m", type=float, default=None)
    parser.add_argument("--tap_contact_slowdown_use_proxy", action="store_true")
    parser.add_argument("--speed_penalty_scale", type=float, default=None)
    parser.add_argument("--speed_penalty_start_mps", type=float, default=None)
    parser.add_argument("--impact_terminal_penalty", type=float, default=None)
    parser.add_argument("--success_bonus", type=float, default=None)
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--init_noise_std", type=float, default=None)
    parser.add_argument("--ppo_learning_rate", type=float, default=None)
    parser.add_argument("--ppo_num_learning_epochs", type=int, default=None)
    parser.add_argument("--ppo_num_mini_batches", type=int, default=None)
    parser.add_argument("--ppo_entropy_coef", type=float, default=None)
    parser.add_argument("--ppo_clip_param", type=float, default=None)
    parser.add_argument("--ppo_desired_kl", type=float, default=None)
    parser.add_argument("--ppo_max_grad_norm", type=float, default=None)
    parser.add_argument("--actor_preserve_blend", type=float, default=0.0)
    parser.add_argument("--no_init_at_random_ep_len", action="store_true")
    args = parser.parse_args()

    if (args.fixed_push_dir_x is None) != (args.fixed_push_dir_y is None):
        raise ValueError("--fixed_push_dir_x and --fixed_push_dir_y must be set together")
    if args.init_noise_std is not None and args.init_noise_std <= 0.0:
        raise ValueError("--init_noise_std must be positive")
    if args.ppo_learning_rate is not None and args.ppo_learning_rate <= 0.0:
        raise ValueError("--ppo_learning_rate must be positive")
    if args.ppo_num_learning_epochs is not None and args.ppo_num_learning_epochs <= 0:
        raise ValueError("--ppo_num_learning_epochs must be positive")
    if args.ppo_num_mini_batches is not None and args.ppo_num_mini_batches <= 0:
        raise ValueError("--ppo_num_mini_batches must be positive")
    if args.ppo_clip_param is not None and args.ppo_clip_param <= 0.0:
        raise ValueError("--ppo_clip_param must be positive")
    if args.ppo_desired_kl is not None and args.ppo_desired_kl <= 0.0:
        raise ValueError("--ppo_desired_kl must be positive")
    if args.ppo_max_grad_norm is not None and args.ppo_max_grad_norm <= 0.0:
        raise ValueError("--ppo_max_grad_norm must be positive")
    if not (0.0 <= args.actor_preserve_blend <= 1.0):
        raise ValueError("--actor_preserve_blend must be in [0, 1]")
    if (
        args.d256_reset_episode_min is not None
        and args.d256_reset_episode_max is not None
        and args.d256_reset_episode_min > args.d256_reset_episode_max
    ):
        raise ValueError("--d256_reset_episode_min must be <= --d256_reset_episode_max")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401 - registers envs lazily
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg, RoArmCubeTap10cmEnvCfg

    if args.env_kind == "push3cm":
        env_id = "RoArm-CubePush-Direct-v0"
        env_cfg = RoArmCubePushEnvCfg()
    else:
        env_id = "RoArm-CubeTap10cm-Direct-v0"
        env_cfg = RoArmCubeTap10cmEnvCfg()
        env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.robot.spawn.usd_path = args.robot_usd_path
    if args.ik_endpoint_reset:
        print("[cube-push-train] ik_endpoint_reset: True")
        env_cfg.ik_endpoint_reset = True
    if args.episode_length_s is not None:
        print(f"[cube-push-train] episode_length_s: {env_cfg.episode_length_s} -> {args.episode_length_s}")
        env_cfg.episode_length_s = args.episode_length_s
    if args.action_scale is not None:
        print(f"[cube-push-train] action_scale: {env_cfg.action_scale} -> {args.action_scale}")
        env_cfg.action_scale = args.action_scale
    if args.action_smoothing_alpha is not None:
        print(
            "[cube-push-train] action_smoothing_alpha: "
            f"{env_cfg.action_smoothing_alpha} -> {args.action_smoothing_alpha}"
        )
        env_cfg.action_smoothing_alpha = args.action_smoothing_alpha
    if args.max_joint_delta_per_step_rad is not None:
        print(
            "[cube-push-train] max_joint_delta_per_step_rad: "
            f"{env_cfg.max_joint_delta_per_step_rad} -> {args.max_joint_delta_per_step_rad}"
        )
        env_cfg.max_joint_delta_per_step_rad = args.max_joint_delta_per_step_rad
    if args.contact_joint_delta_scale is not None:
        print(
            "[cube-push-train] contact_joint_delta_scale: "
            f"{env_cfg.contact_joint_delta_scale} -> {args.contact_joint_delta_scale}"
        )
        env_cfg.contact_joint_delta_scale = args.contact_joint_delta_scale
    if args.fast_cube_joint_delta_scale is not None:
        print(
            "[cube-push-train] fast_cube_joint_delta_scale: "
            f"{env_cfg.fast_cube_joint_delta_scale} -> {args.fast_cube_joint_delta_scale}"
        )
        env_cfg.fast_cube_joint_delta_scale = args.fast_cube_joint_delta_scale
    if args.joint_target_lead_limit_rad is not None:
        print(
            "[cube-push-train] joint_target_lead_limit_rad: "
            f"{env_cfg.joint_target_lead_limit_rad} -> {args.joint_target_lead_limit_rad}"
        )
        env_cfg.joint_target_lead_limit_rad = args.joint_target_lead_limit_rad
    if args.joint_delta_reference is not None:
        print(
            "[cube-push-train] joint_delta_reference: "
            f"{env_cfg.joint_delta_reference} -> {args.joint_delta_reference}"
        )
        env_cfg.joint_delta_reference = args.joint_delta_reference
    if args.fixed_push_dir_x is not None and args.fixed_push_dir_y is not None:
        print(
            "[cube-push-train] fixed_push_dir_x/y: "
            f"{env_cfg.fixed_push_dir_x}/{env_cfg.fixed_push_dir_y} -> "
            f"{args.fixed_push_dir_x}/{args.fixed_push_dir_y}"
        )
        env_cfg.fixed_push_dir_x = args.fixed_push_dir_x
        env_cfg.fixed_push_dir_y = args.fixed_push_dir_y
    if args.ik_precontact_clearance_m is not None:
        print(
            "[cube-push-train] ik_precontact_clearance_m: "
            f"{env_cfg.ik_precontact_clearance_m} -> {args.ik_precontact_clearance_m}"
        )
        env_cfg.ik_precontact_clearance_m = args.ik_precontact_clearance_m
    if args.scripted_teacher_blend is not None:
        print(
            "[cube-push-train] scripted_teacher_blend: "
            f"{env_cfg.scripted_teacher_blend} -> {args.scripted_teacher_blend}"
        )
        env_cfg.scripted_teacher_blend = args.scripted_teacher_blend
    if args.scripted_teacher_horizon_frac is not None:
        print(
            "[cube-push-train] scripted_teacher_horizon_frac: "
            f"{env_cfg.scripted_teacher_horizon_frac} -> {args.scripted_teacher_horizon_frac}"
        )
        env_cfg.scripted_teacher_horizon_frac = args.scripted_teacher_horizon_frac
    if args.scripted_teacher_goal_push_m is not None:
        print(
            "[cube-push-train] scripted_teacher_goal_push_m: "
            f"{env_cfg.scripted_teacher_goal_push_m} -> {args.scripted_teacher_goal_push_m}"
        )
        env_cfg.scripted_teacher_goal_push_m = args.scripted_teacher_goal_push_m
    if args.bc_teacher_checkpoint_path is not None:
        print(
            "[cube-push-train] bc_teacher_checkpoint_path: "
            f"{env_cfg.bc_teacher_checkpoint_path} -> {args.bc_teacher_checkpoint_path}"
        )
        env_cfg.bc_teacher_checkpoint_path = args.bc_teacher_checkpoint_path
    if args.bc_teacher_blend is not None:
        print(f"[cube-push-train] bc_teacher_blend: {env_cfg.bc_teacher_blend} -> {args.bc_teacher_blend}")
        env_cfg.bc_teacher_blend = args.bc_teacher_blend
    if args.bc_teacher_imitation_reward_scale is not None:
        print(
            "[cube-push-train] bc_teacher_imitation_reward_scale: "
            f"{env_cfg.bc_teacher_imitation_reward_scale} -> {args.bc_teacher_imitation_reward_scale}"
        )
        env_cfg.bc_teacher_imitation_reward_scale = args.bc_teacher_imitation_reward_scale
    if args.warm_start_checkpoint_path is not None and not Path(args.warm_start_checkpoint_path).exists():
        raise FileNotFoundError(args.warm_start_checkpoint_path)
    if args.bc_teacher_policy_delta_clip_rad is not None:
        print(
            "[cube-push-train] bc_teacher_policy_delta_clip_rad: "
            f"{env_cfg.bc_teacher_policy_delta_clip_rad} -> {args.bc_teacher_policy_delta_clip_rad}"
        )
        env_cfg.bc_teacher_policy_delta_clip_rad = args.bc_teacher_policy_delta_clip_rad
    if args.bc_teacher_policy_delta_scale is not None:
        print(
            "[cube-push-train] bc_teacher_policy_delta_scale: "
            f"{env_cfg.bc_teacher_policy_delta_scale} -> {args.bc_teacher_policy_delta_scale}"
        )
        env_cfg.bc_teacher_policy_delta_scale = args.bc_teacher_policy_delta_scale
    if args.bc_teacher_feature_target_mode is not None:
        print(
            "[cube-push-train] bc_teacher_feature_target_mode: "
            f"{env_cfg.bc_teacher_feature_target_mode} -> {args.bc_teacher_feature_target_mode}"
        )
        env_cfg.bc_teacher_feature_target_mode = args.bc_teacher_feature_target_mode
    if args.bc_teacher_posx_policy_delta_scale is not None:
        print(
            "[cube-push-train] bc_teacher_posx_policy_delta_scale: "
            f"{env_cfg.bc_teacher_posx_policy_delta_scale} -> {args.bc_teacher_posx_policy_delta_scale}"
        )
        env_cfg.bc_teacher_posx_policy_delta_scale = args.bc_teacher_posx_policy_delta_scale
    if args.bc_teacher_lowx_policy_delta_scale is not None:
        print(
            "[cube-push-train] bc_teacher_lowx_policy_delta_scale: "
            f"{env_cfg.bc_teacher_lowx_policy_delta_scale} -> {args.bc_teacher_lowx_policy_delta_scale}"
        )
        env_cfg.bc_teacher_lowx_policy_delta_scale = args.bc_teacher_lowx_policy_delta_scale
    if args.bc_teacher_highx_policy_delta_scale is not None:
        print(
            "[cube-push-train] bc_teacher_highx_policy_delta_scale: "
            f"{env_cfg.bc_teacher_highx_policy_delta_scale} -> {args.bc_teacher_highx_policy_delta_scale}"
        )
        env_cfg.bc_teacher_highx_policy_delta_scale = args.bc_teacher_highx_policy_delta_scale
    if args.bc_teacher_delta_smoothing_alpha is not None:
        print(
            "[cube-push-train] bc_teacher_delta_smoothing_alpha: "
            f"{env_cfg.bc_teacher_delta_smoothing_alpha} -> {args.bc_teacher_delta_smoothing_alpha}"
        )
        env_cfg.bc_teacher_delta_smoothing_alpha = args.bc_teacher_delta_smoothing_alpha
    if args.bc_teacher_phase_timing is not None:
        print(
            "[cube-push-train] bc_teacher_phase_timing: "
            f"{env_cfg.bc_teacher_phase_timing} -> {args.bc_teacher_phase_timing}"
        )
        env_cfg.bc_teacher_phase_timing = args.bc_teacher_phase_timing
    if args.bc_teacher_linear_phase_steps is not None:
        print(
            "[cube-push-train] bc_teacher_linear_phase_steps: "
            f"{env_cfg.bc_teacher_linear_phase_steps} -> {args.bc_teacher_linear_phase_steps}"
        )
        env_cfg.bc_teacher_linear_phase_steps = args.bc_teacher_linear_phase_steps
    if args.d256_reset_csv_path is not None:
        print(
            "[cube-push-train] d256_reset_csv_path: "
            f"{env_cfg.d256_reset_csv_path or 'NONE'} -> {args.d256_reset_csv_path}"
        )
        env_cfg.d256_reset_csv_path = args.d256_reset_csv_path
    if args.d256_reset_frame_index is not None:
        print(
            "[cube-push-train] d256_reset_frame_index: "
            f"{env_cfg.d256_reset_frame_index} -> {args.d256_reset_frame_index}"
        )
        env_cfg.d256_reset_frame_index = args.d256_reset_frame_index
    if args.d256_reset_sample_mode is not None:
        print(
            "[cube-push-train] d256_reset_sample_mode: "
            f"{env_cfg.d256_reset_sample_mode} -> {args.d256_reset_sample_mode}"
        )
        env_cfg.d256_reset_sample_mode = args.d256_reset_sample_mode
    if args.d256_reset_episode_min is not None:
        print(
            "[cube-push-train] d256_reset_episode_min: "
            f"{env_cfg.d256_reset_episode_min} -> {args.d256_reset_episode_min}"
        )
        env_cfg.d256_reset_episode_min = args.d256_reset_episode_min
    if args.d256_reset_episode_max is not None:
        print(
            "[cube-push-train] d256_reset_episode_max: "
            f"{env_cfg.d256_reset_episode_max} -> {args.d256_reset_episode_max}"
        )
        env_cfg.d256_reset_episode_max = args.d256_reset_episode_max
    if args.cube_success_disp_m is not None:
        print(f"[cube-push-train] cube_success_disp_m: {env_cfg.cube_success_disp_m} -> {args.cube_success_disp_m}")
        env_cfg.cube_success_disp_m = args.cube_success_disp_m
    if args.cube_success_speed_max_mps is not None:
        print(
            "[cube-push-train] cube_success_speed_max_mps: "
            f"{env_cfg.cube_success_speed_max_mps} -> {args.cube_success_speed_max_mps}"
        )
        env_cfg.cube_success_speed_max_mps = args.cube_success_speed_max_mps
    if args.cube_push_target_disp_m is not None:
        print(
            "[cube-push-train] cube_push_target_disp_m: "
            f"{env_cfg.cube_push_target_disp_m} -> {args.cube_push_target_disp_m}"
        )
        env_cfg.cube_push_target_disp_m = args.cube_push_target_disp_m
    if args.push_progress_reward_scale is not None:
        print(
            "[cube-push-train] push_progress_reward_scale: "
            f"{env_cfg.push_progress_reward_scale} -> {args.push_progress_reward_scale}"
        )
        env_cfg.push_progress_reward_scale = args.push_progress_reward_scale
    if args.push_displacement_reward_scale is not None:
        print(
            "[cube-push-train] push_displacement_reward_scale: "
            f"{env_cfg.push_displacement_reward_scale} -> {args.push_displacement_reward_scale}"
        )
        env_cfg.push_displacement_reward_scale = args.push_displacement_reward_scale
    if args.low_motion_penalty_scale is not None:
        print(
            "[cube-push-train] low_motion_penalty_scale: "
            f"{env_cfg.low_motion_penalty_scale} -> {args.low_motion_penalty_scale}"
        )
        env_cfg.low_motion_penalty_scale = args.low_motion_penalty_scale
    if args.controlled_bonus_scale is not None:
        print(
            "[cube-push-train] controlled_bonus_scale: "
            f"{env_cfg.controlled_bonus_scale} -> {args.controlled_bonus_scale}"
        )
        env_cfg.controlled_bonus_scale = args.controlled_bonus_scale
    if args.impact_penalty_scale is not None:
        print(f"[cube-push-train] impact_penalty_scale: {env_cfg.impact_penalty_scale} -> {args.impact_penalty_scale}")
        env_cfg.impact_penalty_scale = args.impact_penalty_scale
    if args.reverse_push_penalty_scale is not None:
        print(
            "[cube-push-train] reverse_push_penalty_scale: "
            f"{env_cfg.reverse_push_penalty_scale} -> {args.reverse_push_penalty_scale}"
        )
        env_cfg.reverse_push_penalty_scale = args.reverse_push_penalty_scale
    if args.target_distance_penalty_scale is not None:
        print(
            "[cube-push-train] target_distance_penalty_scale: "
            f"{env_cfg.target_distance_penalty_scale} -> {args.target_distance_penalty_scale}"
        )
        env_cfg.target_distance_penalty_scale = args.target_distance_penalty_scale
    if args.lateral_penalty_scale is not None:
        print(f"[cube-push-train] lateral_penalty_scale: {env_cfg.lateral_penalty_scale} -> {args.lateral_penalty_scale}")
        env_cfg.lateral_penalty_scale = args.lateral_penalty_scale
    if args.overshoot_penalty_scale is not None:
        print(
            "[cube-push-train] overshoot_penalty_scale: "
            f"{env_cfg.overshoot_penalty_scale} -> {args.overshoot_penalty_scale}"
        )
        env_cfg.overshoot_penalty_scale = args.overshoot_penalty_scale
    if args.tap_success_terminate:
        if not hasattr(env_cfg, "tap_success_terminate"):
            raise ValueError("--tap_success_terminate is only supported for --env_kind tap10cm")
        print(f"[cube-push-train] tap_success_terminate: {env_cfg.tap_success_terminate} -> True")
        env_cfg.tap_success_terminate = True
    if args.tap_overshoot_terminate:
        if not hasattr(env_cfg, "tap_overshoot_terminate"):
            raise ValueError("--tap_overshoot_terminate is only supported for --env_kind tap10cm")
        print(f"[cube-push-train] tap_overshoot_terminate: {env_cfg.tap_overshoot_terminate} -> True")
        env_cfg.tap_overshoot_terminate = True
    if args.tap_useful_terminate:
        if not hasattr(env_cfg, "tap_useful_terminate"):
            raise ValueError("--tap_useful_terminate is only supported for --env_kind tap10cm")
        print(f"[cube-push-train] tap_useful_terminate: {env_cfg.tap_useful_terminate} -> True")
        env_cfg.tap_useful_terminate = True
    if args.tap_stop_after_useful_seen:
        if not hasattr(env_cfg, "tap_stop_after_useful_seen"):
            raise ValueError("--tap_stop_after_useful_seen is only supported for --env_kind tap10cm")
        print(f"[cube-push-train] tap_stop_after_useful_seen: {env_cfg.tap_stop_after_useful_seen} -> True")
        env_cfg.tap_stop_after_useful_seen = True
    if args.tap_stop_after_disp_m is not None:
        if not hasattr(env_cfg, "tap_stop_after_disp_m"):
            raise ValueError("--tap_stop_after_disp_m is only supported for --env_kind tap10cm")
        print(
            "[cube-push-train] tap_stop_after_disp_m: "
            f"{env_cfg.tap_stop_after_disp_m} -> {args.tap_stop_after_disp_m}"
        )
        env_cfg.tap_stop_after_disp_m = args.tap_stop_after_disp_m
    if args.tap_contact_slowdown_use_proxy:
        if not hasattr(env_cfg, "tap_contact_slowdown_use_proxy"):
            raise ValueError("--tap_contact_slowdown_use_proxy is only supported for --env_kind tap10cm")
        print(
            "[cube-push-train] tap_contact_slowdown_use_proxy: "
            f"{env_cfg.tap_contact_slowdown_use_proxy} -> True"
        )
        env_cfg.tap_contact_slowdown_use_proxy = True
    if args.speed_penalty_scale is not None:
        print(f"[cube-push-train] speed_penalty_scale: {env_cfg.speed_penalty_scale} -> {args.speed_penalty_scale}")
        env_cfg.speed_penalty_scale = args.speed_penalty_scale
    if args.speed_penalty_start_mps is not None:
        print(
            "[cube-push-train] speed_penalty_start_mps: "
            f"{env_cfg.speed_penalty_start_mps} -> {args.speed_penalty_start_mps}"
        )
        env_cfg.speed_penalty_start_mps = args.speed_penalty_start_mps
    if args.impact_terminal_penalty is not None:
        print(
            "[cube-push-train] impact_terminal_penalty: "
            f"{env_cfg.impact_terminal_penalty} -> {args.impact_terminal_penalty}"
        )
        env_cfg.impact_terminal_penalty = args.impact_terminal_penalty
    if args.success_bonus is not None:
        print(f"[cube-push-train] success_bonus: {env_cfg.success_bonus} -> {args.success_bonus}")
        env_cfg.success_bonus = args.success_bonus

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.max_iterations = args.max_iterations
    ppo_cfg.seed = args.seed
    if args.num_steps_per_env is not None:
        print(f"[cube-push-train] ppo_num_steps_per_env: {ppo_cfg.num_steps_per_env} -> {args.num_steps_per_env}")
        ppo_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        print(f"[cube-push-train] ppo_save_interval: {ppo_cfg.save_interval} -> {args.save_interval}")
        ppo_cfg.save_interval = args.save_interval
    if args.init_noise_std is not None:
        print(f"[cube-push-train] ppo_init_noise_std: {ppo_cfg.policy.init_noise_std} -> {args.init_noise_std}")
        ppo_cfg.policy.init_noise_std = args.init_noise_std
    if args.ppo_learning_rate is not None:
        print(f"[cube-push-train] ppo_learning_rate: {ppo_cfg.algorithm.learning_rate} -> {args.ppo_learning_rate}")
        ppo_cfg.algorithm.learning_rate = args.ppo_learning_rate
    if args.ppo_num_learning_epochs is not None:
        print(
            "[cube-push-train] ppo_num_learning_epochs: "
            f"{ppo_cfg.algorithm.num_learning_epochs} -> {args.ppo_num_learning_epochs}"
        )
        ppo_cfg.algorithm.num_learning_epochs = args.ppo_num_learning_epochs
    if args.ppo_num_mini_batches is not None:
        print(
            "[cube-push-train] ppo_num_mini_batches: "
            f"{ppo_cfg.algorithm.num_mini_batches} -> {args.ppo_num_mini_batches}"
        )
        ppo_cfg.algorithm.num_mini_batches = args.ppo_num_mini_batches
    if args.ppo_entropy_coef is not None:
        print(f"[cube-push-train] ppo_entropy_coef: {ppo_cfg.algorithm.entropy_coef} -> {args.ppo_entropy_coef}")
        ppo_cfg.algorithm.entropy_coef = args.ppo_entropy_coef
    if args.ppo_clip_param is not None:
        print(f"[cube-push-train] ppo_clip_param: {ppo_cfg.algorithm.clip_param} -> {args.ppo_clip_param}")
        ppo_cfg.algorithm.clip_param = args.ppo_clip_param
    if args.ppo_desired_kl is not None:
        print(f"[cube-push-train] ppo_desired_kl: {ppo_cfg.algorithm.desired_kl} -> {args.ppo_desired_kl}")
        ppo_cfg.algorithm.desired_kl = args.ppo_desired_kl
    if args.ppo_max_grad_norm is not None:
        print(f"[cube-push-train] ppo_max_grad_norm: {ppo_cfg.algorithm.max_grad_norm} -> {args.ppo_max_grad_norm}")
        ppo_cfg.algorithm.max_grad_norm = args.ppo_max_grad_norm
    ppo_cfg.experiment_name = args.experiment_name or (
        "roarm_cube_push_no_attach_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    log_root_path = args.logdir or os.path.join(
        os.environ.get("ROARM_B200_ROOT", os.getcwd()),
        "logs",
        "roarm_rl",
    )
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    log_dir = os.path.join(log_root_path, ppo_cfg.experiment_name)

    print(
        "[cube-push-train] scope=no_attach_cube_push_or_tap "
        f"env_kind={args.env_kind} env_id={env_id} training=YES dataset_generation=NO "
        "grasp_attach=NO rollout_object_posewrite=NO"
    )
    target_update = (
        f"robot_dof_targets += action_scale({env_cfg.action_scale:.3f}) * action"
        if env_cfg.joint_delta_reference == "target"
        else f"robot_dof_targets = joint_pos + action_scale({env_cfg.action_scale:.3f}) * action"
    )
    print(
        "[cube-push-train] action_semantics=normalized_joint_delta "
        f"target_update='{target_update}' action_dim=6 action_clip=[-1,1] gripper_open_hold=YES"
    )
    print(f"[cube-push-train] env: num_envs={args.num_envs} max_iterations={args.max_iterations}")
    print(
        "[cube-push-train] curriculum "
        f"ik_endpoint_reset={env_cfg.ik_endpoint_reset} "
        f"push_progress_reward_scale={env_cfg.push_progress_reward_scale} "
        f"push_displacement_reward_scale={env_cfg.push_displacement_reward_scale} "
        f"low_motion_penalty_scale={env_cfg.low_motion_penalty_scale} "
        f"controlled_bonus_scale={env_cfg.controlled_bonus_scale} "
        f"impact_penalty_scale={env_cfg.impact_penalty_scale} "
        f"target_distance_penalty_scale={env_cfg.target_distance_penalty_scale} "
        f"speed_penalty_scale={env_cfg.speed_penalty_scale} "
        f"success_target_tol_m={env_cfg.cube_success_target_tol_m} "
        f"success_speed_max_mps={env_cfg.cube_success_speed_max_mps} "
        f"action_smoothing_alpha={env_cfg.action_smoothing_alpha} "
        f"max_joint_delta_per_step_rad={env_cfg.max_joint_delta_per_step_rad} "
        f"contact_joint_delta_scale={env_cfg.contact_joint_delta_scale} "
        f"fixed_push_dir_x={env_cfg.fixed_push_dir_x} "
        f"fixed_push_dir_y={env_cfg.fixed_push_dir_y} "
        f"tap_contact_proxy_mode={getattr(env_cfg, 'tap_contact_proxy_mode', 'NA')} "
        f"tap_success_terminate={getattr(env_cfg, 'tap_success_terminate', 'NA')} "
        f"tap_overshoot_terminate={getattr(env_cfg, 'tap_overshoot_terminate', 'NA')} "
        f"tap_useful_terminate={getattr(env_cfg, 'tap_useful_terminate', 'NA')} "
        f"tap_stop_after_useful_seen={getattr(env_cfg, 'tap_stop_after_useful_seen', 'NA')} "
        f"tap_stop_after_disp_m={getattr(env_cfg, 'tap_stop_after_disp_m', 'NA')} "
        f"tap_contact_slowdown_use_proxy={getattr(env_cfg, 'tap_contact_slowdown_use_proxy', 'NA')} "
        f"joint_target_lead_limit_rad={env_cfg.joint_target_lead_limit_rad} "
        f"joint_delta_reference={env_cfg.joint_delta_reference} "
        f"ik_precontact_clearance_m={env_cfg.ik_precontact_clearance_m} "
        f"scripted_teacher_blend={env_cfg.scripted_teacher_blend} "
        f"scripted_teacher_horizon_frac={env_cfg.scripted_teacher_horizon_frac} "
        f"bc_teacher_blend={env_cfg.bc_teacher_blend} "
        f"bc_teacher_imitation_reward_scale={env_cfg.bc_teacher_imitation_reward_scale} "
        f"bc_teacher_feature_target_mode={env_cfg.bc_teacher_feature_target_mode} "
        f"bc_teacher_policy_delta_scale={env_cfg.bc_teacher_policy_delta_scale} "
        f"bc_teacher_lowx_policy_delta_scale={env_cfg.bc_teacher_lowx_policy_delta_scale} "
        f"bc_teacher_highx_policy_delta_scale={env_cfg.bc_teacher_highx_policy_delta_scale} "
        f"bc_teacher_delta_smoothing_alpha={env_cfg.bc_teacher_delta_smoothing_alpha} "
        f"bc_teacher_phase_timing={env_cfg.bc_teacher_phase_timing} "
        f"bc_teacher_linear_phase_steps={env_cfg.bc_teacher_linear_phase_steps} "
        f"d256_reset_csv_path={env_cfg.d256_reset_csv_path or 'NONE'} "
        f"d256_reset_frame_index={env_cfg.d256_reset_frame_index} "
        f"d256_reset_sample_mode={env_cfg.d256_reset_sample_mode} "
        f"d256_reset_episode_min={env_cfg.d256_reset_episode_min} "
        f"d256_reset_episode_max={env_cfg.d256_reset_episode_max}"
    )
    print(f"[cube-push-train] bc_teacher_checkpoint_path={env_cfg.bc_teacher_checkpoint_path or 'NONE'}")
    print(f"[cube-push-train] robot_usd_path={env_cfg.robot.spawn.usd_path}")
    print(f"[cube-push-train] ppo: steps_per_env={ppo_cfg.num_steps_per_env}")
    print(f"[cube-push-train] ppo: init_noise_std={ppo_cfg.policy.init_noise_std}")
    print(
        "[cube-push-train] ppo: "
        f"learning_rate={ppo_cfg.algorithm.learning_rate} "
        f"num_learning_epochs={ppo_cfg.algorithm.num_learning_epochs} "
        f"num_mini_batches={ppo_cfg.algorithm.num_mini_batches} "
        f"entropy_coef={ppo_cfg.algorithm.entropy_coef} "
        f"clip_param={ppo_cfg.algorithm.clip_param} "
        f"desired_kl={ppo_cfg.algorithm.desired_kl} "
        f"max_grad_norm={ppo_cfg.algorithm.max_grad_norm}"
    )
    print(f"[cube-push-train] ppo: actor_preserve_blend={args.actor_preserve_blend}")
    print(f"[cube-push-train] ppo: init_at_random_ep_len={not bool(args.no_init_at_random_ep_len)}")
    print(f"[cube-push-train] log_dir: {log_dir}")

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)
    if args.warm_start_checkpoint_path is not None:
        print(f"[cube-push-train] warm_start_checkpoint_path={args.warm_start_checkpoint_path}")
        runner.load(args.warm_start_checkpoint_path, load_optimizer=False, map_location=env.unwrapped.device)
        if args.init_noise_std is not None:
            policy = runner.alg.policy
            with torch.no_grad():
                if getattr(policy, "noise_std_type", "scalar") == "scalar" and hasattr(policy, "std"):
                    policy.std.fill_(float(args.init_noise_std))
                    applied_std = float(policy.std.mean().detach().cpu().item())
                elif getattr(policy, "noise_std_type", "") == "log" and hasattr(policy, "log_std"):
                    policy.log_std.fill_(float(torch.log(torch.tensor(float(args.init_noise_std))).item()))
                    applied_std = float(torch.exp(policy.log_std).mean().detach().cpu().item())
                else:
                    raise ValueError(
                        "Cannot override PPO action noise after warm start: "
                        f"unknown noise_std_type={getattr(policy, 'noise_std_type', None)}"
                    )
            print(f"[cube-push-train] ppo_init_noise_std_after_warm_start={applied_std:.6f}")
    if args.actor_preserve_blend > 0.0:
        policy = runner.alg.policy
        preserve_prefixes = ("actor.", "actor_obs_normalizer.")
        preserve_names = ("std", "log_std")
        reference_state = {
            key: value.detach().clone()
            for key, value in policy.state_dict().items()
            if key.startswith(preserve_prefixes) or key in preserve_names
        }
        if not reference_state:
            raise ValueError("actor preservation requested, but no actor/reference keys were found")
        original_update = runner.alg.update

        def _actor_preserving_update():
            loss_dict = original_update()
            current_state = policy.state_dict()
            max_pre_restore_delta = 0.0
            with torch.no_grad():
                for key, ref_value in reference_state.items():
                    cur_value = current_state[key]
                    if torch.is_floating_point(cur_value):
                        pre_delta = torch.max(torch.abs(cur_value - ref_value)).detach().cpu().item()
                        max_pre_restore_delta = max(max_pre_restore_delta, float(pre_delta))
                        cur_value.copy_((1.0 - float(args.actor_preserve_blend)) * cur_value + float(args.actor_preserve_blend) * ref_value)
                    else:
                        cur_value.copy_(ref_value)
            max_post_restore_delta = 0.0
            post_state = policy.state_dict()
            for key, ref_value in reference_state.items():
                cur_value = post_state[key]
                if torch.is_floating_point(cur_value):
                    post_delta = torch.max(torch.abs(cur_value - ref_value)).detach().cpu().item()
                    max_post_restore_delta = max(max_post_restore_delta, float(post_delta))
            print(
                "[cube-push-train] actor_preserve_after_update "
                f"blend={args.actor_preserve_blend:.6f} "
                f"keys={len(reference_state)} "
                f"max_pre_restore_delta={max_pre_restore_delta:.9f} "
                f"max_post_restore_delta={max_post_restore_delta:.9f}",
                flush=True,
            )
            return loss_dict

        runner.alg.update = _actor_preserving_update
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=not bool(args.no_init_at_random_ep_len))

    if args.env_kind == "tap10cm":
        inner = env.unwrapped
        if hasattr(inner, "_tap_contact_seen"):
            useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
            contact_reaction_seen = inner._tap_contact_seen & inner._tap_reaction_seen
            max_disp_along_ge_1mm = inner._tap_max_disp_along >= 0.001
            max_disp_xy_ge_1mm = inner._tap_max_disp_xy >= 0.001
            max_disp_along_ge_3mm = inner._tap_max_disp_along >= 0.003
            max_disp_xy_ge_3mm = inner._tap_max_disp_xy >= 0.003
            final_step = int(getattr(runner, "current_learning_iteration", max(0, int(args.max_iterations) - 1)))
            collection_final_scalars = {
                "CollectionFinal/cube_tap_contact_seen_rate": inner._tap_contact_seen.float().mean(),
                "CollectionFinal/cube_tap_reaction_seen_rate": inner._tap_reaction_seen.float().mean(),
                "CollectionFinal/cube_tap_contact_reaction_seen_rate": contact_reaction_seen.float().mean(),
                "CollectionFinal/cube_tap_useful_seen_rate": useful_seen.float().mean(),
                "CollectionFinal/cube_tap_success_rate": inner._tap_success_flag.float().mean(),
                "CollectionFinal/cube_tap_overshoot_seen_rate": inner._tap_overshoot_seen.float().mean(),
                "CollectionFinal/cube_tap_max_disp_along_m": inner._tap_max_disp_along.mean(),
                "CollectionFinal/cube_tap_max_disp_xy_m": inner._tap_max_disp_xy.mean(),
                "CollectionFinal/cube_tap_max_disp_along_max_m": inner._tap_max_disp_along.max(),
                "CollectionFinal/cube_tap_max_disp_xy_max_m": inner._tap_max_disp_xy.max(),
                "CollectionFinal/cube_tap_max_disp_along_ge_1mm_rate": max_disp_along_ge_1mm.float().mean(),
                "CollectionFinal/cube_tap_max_disp_xy_ge_1mm_rate": max_disp_xy_ge_1mm.float().mean(),
                "CollectionFinal/cube_tap_max_disp_along_ge_3mm_rate": max_disp_along_ge_3mm.float().mean(),
                "CollectionFinal/cube_tap_max_disp_xy_ge_3mm_rate": max_disp_xy_ge_3mm.float().mean(),
                "CollectionFinal/cube_tap_d256_reset_active_rate": inner._last_d256_reset_active.mean(),
                "CollectionFinal/cube_push_joint_delta_cap_rate": inner._last_joint_delta_cap_rate.mean(),
            }
            if getattr(runner, "writer", None) is not None:
                for tag, value in collection_final_scalars.items():
                    runner.writer.add_scalar(tag, float(value.detach().cpu().item()), final_step)
                runner.writer.flush()
            print(
                "[cube-push-train] collection_final "
                f"useful={float(collection_final_scalars['CollectionFinal/cube_tap_useful_seen_rate'].detach().cpu().item()):.6f} "
                f"overshoot={float(collection_final_scalars['CollectionFinal/cube_tap_overshoot_seen_rate'].detach().cpu().item()):.6f} "
                f"max_xy_mean={float(collection_final_scalars['CollectionFinal/cube_tap_max_disp_xy_m'].detach().cpu().item()):.6f} "
                f"max_xy_max={float(collection_final_scalars['CollectionFinal/cube_tap_max_disp_xy_max_m'].detach().cpu().item()):.6f}",
                flush=True,
            )
            final_terms = inner._tap_terms()

            def _tensor_list(name: str):
                return getattr(inner, name).detach().cpu().tolist()

            def _term_list(name: str):
                return final_terms[name].detach().cpu().tolist()

            trace_columns = {
                "d256_reset_active": _tensor_list("_last_d256_reset_active"),
                "d256_reset_episode_index": _tensor_list("_last_d256_reset_episode_index"),
                "contact_seen": _tensor_list("_tap_contact_seen"),
                "reaction_seen": _tensor_list("_tap_reaction_seen"),
                "useful_seen": useful_seen.detach().cpu().tolist(),
                "success": _tensor_list("_tap_success_flag"),
                "overshoot_seen": _tensor_list("_tap_overshoot_seen"),
                "max_disp_along_m": _tensor_list("_tap_max_disp_along"),
                "max_disp_xy_m": _tensor_list("_tap_max_disp_xy"),
                "max_z_delta_m": _tensor_list("_tap_max_z_delta"),
                "max_speed_mps": _tensor_list("_tap_max_speed"),
                "current_contact_proxy": _term_list("tap_contact_proxy"),
                "current_face_gap_m": _term_list("tap_contact_face_gap_m"),
                "current_lateral_m": _term_list("tap_contact_lateral_m"),
                "current_vertical_offset_m": _term_list("tap_contact_vertical_offset_m"),
                "action_abs_mean": _tensor_list("_last_action_abs_mean"),
                "action_abs_max": _tensor_list("_last_action_abs_max"),
                "joint_delta_abs_mean": _tensor_list("_last_joint_delta_abs_mean"),
                "joint_delta_abs_max": _tensor_list("_last_joint_delta_abs_max"),
                "joint_delta_cap_rate": _tensor_list("_last_joint_delta_cap_rate"),
                "bc_teacher_blend": _tensor_list("_last_bc_teacher_blend"),
                "bc_teacher_imitation_mse": _tensor_list("_last_bc_teacher_imitation_mse"),
                "bc_teacher_action_abs_mean": _tensor_list("_last_bc_teacher_action_abs_mean"),
                "tap_stop_after_useful_hold": _tensor_list("_last_tap_stop_after_useful_hold"),
                "tap_stop_after_disp_hold": _tensor_list("_last_tap_stop_after_disp_hold"),
            }
            trace_path = os.path.join(log_dir, f"collection_final_env_trace_iter_{final_step}.jsonl")
            with open(trace_path, "w", encoding="utf-8") as trace_file:
                for env_idx in range(int(inner.num_envs)):
                    row = {"env_id": env_idx, "final_step": final_step}
                    for key, values in trace_columns.items():
                        value = values[env_idx]
                        if isinstance(value, bool):
                            row[key] = value
                        elif key in {"d256_reset_episode_index"}:
                            row[key] = int(value)
                        else:
                            row[key] = float(value)
                    json.dump(row, trace_file, sort_keys=True)
                    trace_file.write("\n")
            print(f"[cube-push-train] collection_final_env_trace={trace_path}", flush=True)

    print(f"[cube-push-train] DONE checkpoints_at={log_dir}")
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
