"""PPO training entry point for the no-attach 3cm cube push task."""
from __future__ import annotations

import argparse
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
    parser.add_argument("--ik_precontact_clearance_m", type=float, default=None)
    parser.add_argument("--scripted_teacher_blend", type=float, default=None)
    parser.add_argument("--scripted_teacher_horizon_frac", type=float, default=None)
    parser.add_argument("--scripted_teacher_goal_push_m", type=float, default=None)
    parser.add_argument("--bc_teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--bc_teacher_blend", type=float, default=None)
    parser.add_argument("--bc_teacher_imitation_reward_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_policy_delta_clip_rad", type=float, default=None)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_posx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=None)
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
    parser.add_argument("--speed_penalty_scale", type=float, default=None)
    parser.add_argument("--speed_penalty_start_mps", type=float, default=None)
    parser.add_argument("--impact_terminal_penalty", type=float, default=None)
    parser.add_argument("--success_bonus", type=float, default=None)
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401 - registers envs lazily
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg

    env_cfg = RoArmCubePushEnvCfg()
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
        "[cube-push-train] scope=no_attach_cube_push "
        "env_id=RoArm-CubePush-Direct-v0 training=YES dataset_generation=NO "
        "grasp_attach=NO rollout_object_posewrite=NO"
    )
    print(
        "[cube-push-train] action_semantics=normalized_joint_delta "
        f"target_update='robot_dof_targets += action_scale({env_cfg.action_scale:.3f}) * action' "
        "action_dim=6 action_clip=[-1,1] gripper_open_hold=YES"
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
            f"joint_target_lead_limit_rad={env_cfg.joint_target_lead_limit_rad} "
            f"ik_precontact_clearance_m={env_cfg.ik_precontact_clearance_m} "
            f"scripted_teacher_blend={env_cfg.scripted_teacher_blend} "
            f"scripted_teacher_horizon_frac={env_cfg.scripted_teacher_horizon_frac} "
            f"bc_teacher_blend={env_cfg.bc_teacher_blend} "
            f"bc_teacher_imitation_reward_scale={env_cfg.bc_teacher_imitation_reward_scale} "
            f"bc_teacher_policy_delta_scale={env_cfg.bc_teacher_policy_delta_scale} "
            f"bc_teacher_lowx_policy_delta_scale={env_cfg.bc_teacher_lowx_policy_delta_scale} "
            f"bc_teacher_highx_policy_delta_scale={env_cfg.bc_teacher_highx_policy_delta_scale} "
            f"bc_teacher_delta_smoothing_alpha={env_cfg.bc_teacher_delta_smoothing_alpha}"
        )
    print(f"[cube-push-train] bc_teacher_checkpoint_path={env_cfg.bc_teacher_checkpoint_path or 'NONE'}")
    print(f"[cube-push-train] robot_usd_path={env_cfg.robot.spawn.usd_path}")
    print(f"[cube-push-train] ppo: steps_per_env={ppo_cfg.num_steps_per_env}")
    print(f"[cube-push-train] log_dir: {log_dir}")

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print(f"[cube-push-train] DONE checkpoints_at={log_dir}")
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
