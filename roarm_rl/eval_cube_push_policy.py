"""Evaluate a frozen no-attach cube-push PPO checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--ik_endpoint_reset", action="store_true")
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--action_smoothing_alpha", type=float, default=None)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=None)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=None)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=None)
    parser.add_argument("--ik_precontact_clearance_m", type=float, default=None)
    parser.add_argument("--speed_penalty_start_mps", type=float, default=None)
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
    parser.add_argument("--record_first_episode_only", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Launch Isaac Sim with a visible local GUI window.")
    parser.add_argument("--livestream", type=int, default=0, choices=(0, 1, 2), help="Enable Isaac Sim WebRTC livestream.")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera/rendering extensions.")
    parser.add_argument("--viewer_step_sleep_s", type=float, default=0.0, help="Sleep after each env step for human viewing.")
    parser.add_argument("--post_run_sleep_s", type=float, default=0.0, help="Keep the app open after rollout for viewing.")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        headless=(not args.gui) or args.livestream > 0,
        enable_cameras=args.enable_cameras,
        livestream=args.livestream,
    )
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg

    env_cfg = RoArmCubePushEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.robot.spawn.usd_path = args.robot_usd_path
    env_cfg.ik_endpoint_reset = args.ik_endpoint_reset
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    if args.action_scale is not None:
        env_cfg.action_scale = args.action_scale
    if args.action_smoothing_alpha is not None:
        env_cfg.action_smoothing_alpha = args.action_smoothing_alpha
    if args.max_joint_delta_per_step_rad is not None:
        env_cfg.max_joint_delta_per_step_rad = args.max_joint_delta_per_step_rad
    if args.contact_joint_delta_scale is not None:
        env_cfg.contact_joint_delta_scale = args.contact_joint_delta_scale
    if args.fast_cube_joint_delta_scale is not None:
        env_cfg.fast_cube_joint_delta_scale = args.fast_cube_joint_delta_scale
    if args.joint_target_lead_limit_rad is not None:
        env_cfg.joint_target_lead_limit_rad = args.joint_target_lead_limit_rad
    if args.ik_precontact_clearance_m is not None:
        env_cfg.ik_precontact_clearance_m = args.ik_precontact_clearance_m
    if args.speed_penalty_start_mps is not None:
        env_cfg.speed_penalty_start_mps = args.speed_penalty_start_mps
    if args.scripted_teacher_blend is not None:
        env_cfg.scripted_teacher_blend = args.scripted_teacher_blend
    if args.scripted_teacher_horizon_frac is not None:
        env_cfg.scripted_teacher_horizon_frac = args.scripted_teacher_horizon_frac
    if args.scripted_teacher_goal_push_m is not None:
        env_cfg.scripted_teacher_goal_push_m = args.scripted_teacher_goal_push_m
    if args.bc_teacher_checkpoint_path is not None:
        env_cfg.bc_teacher_checkpoint_path = args.bc_teacher_checkpoint_path
    if args.bc_teacher_blend is not None:
        env_cfg.bc_teacher_blend = args.bc_teacher_blend
    if args.bc_teacher_imitation_reward_scale is not None:
        env_cfg.bc_teacher_imitation_reward_scale = args.bc_teacher_imitation_reward_scale
    if args.bc_teacher_policy_delta_clip_rad is not None:
        env_cfg.bc_teacher_policy_delta_clip_rad = args.bc_teacher_policy_delta_clip_rad
    if args.bc_teacher_policy_delta_scale is not None:
        env_cfg.bc_teacher_policy_delta_scale = args.bc_teacher_policy_delta_scale
    if args.bc_teacher_posx_policy_delta_scale is not None:
        env_cfg.bc_teacher_posx_policy_delta_scale = args.bc_teacher_posx_policy_delta_scale
    if args.bc_teacher_lowx_policy_delta_scale is not None:
        env_cfg.bc_teacher_lowx_policy_delta_scale = args.bc_teacher_lowx_policy_delta_scale
    if args.bc_teacher_highx_policy_delta_scale is not None:
        env_cfg.bc_teacher_highx_policy_delta_scale = args.bc_teacher_highx_policy_delta_scale
    if args.bc_teacher_delta_smoothing_alpha is not None:
        env_cfg.bc_teacher_delta_smoothing_alpha = args.bc_teacher_delta_smoothing_alpha

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print(
        "[cube-push-eval] scope=no_attach_cube_push_eval "
        "env_id=RoArm-CubePush-Direct-v0 training=NO dataset_generation=NO "
        "grasp_attach=NO rollout_object_posewrite=NO"
    )
    print(
        "[cube-push-eval] action_semantics=policy_output_normalized_joint_delta "
        f"target_update='robot_dof_targets += action_scale({env_cfg.action_scale:.3f}) * action' "
        "action_dim=6 action_clip=[-1,1] gripper_open_hold=YES"
    )
    print(f"[cube-push-eval] checkpoint={args.checkpoint}")
    print(f"[cube-push-eval] num_envs={args.num_envs} num_rollouts={args.num_rollouts} seed={args.seed}")
    print(
        "[cube-push-eval] curriculum "
        f"ik_endpoint_reset={env_cfg.ik_endpoint_reset} "
        f"action_smoothing_alpha={env_cfg.action_smoothing_alpha} "
        f"max_joint_delta_per_step_rad={env_cfg.max_joint_delta_per_step_rad} "
        f"contact_joint_delta_scale={env_cfg.contact_joint_delta_scale} "
        f"fast_cube_joint_delta_scale={env_cfg.fast_cube_joint_delta_scale} "
        f"joint_target_lead_limit_rad={env_cfg.joint_target_lead_limit_rad} "
        f"ik_precontact_clearance_m={env_cfg.ik_precontact_clearance_m} "
        f"speed_penalty_start_mps={env_cfg.speed_penalty_start_mps} "
        f"scripted_teacher_blend={env_cfg.scripted_teacher_blend} "
        f"scripted_teacher_horizon_frac={env_cfg.scripted_teacher_horizon_frac} "
        f"bc_teacher_blend={env_cfg.bc_teacher_blend} "
        f"bc_teacher_imitation_reward_scale={env_cfg.bc_teacher_imitation_reward_scale} "
        f"bc_teacher_policy_delta_scale={env_cfg.bc_teacher_policy_delta_scale} "
        f"bc_teacher_lowx_policy_delta_scale={env_cfg.bc_teacher_lowx_policy_delta_scale} "
        f"bc_teacher_highx_policy_delta_scale={env_cfg.bc_teacher_highx_policy_delta_scale} "
        f"bc_teacher_delta_smoothing_alpha={env_cfg.bc_teacher_delta_smoothing_alpha}"
    )
    print(f"[cube-push-eval] bc_teacher_checkpoint_path={env_cfg.bc_teacher_checkpoint_path or 'NONE'}")
    print(f"[cube-push-eval] robot_usd_path={env_cfg.robot.spawn.usd_path}")

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    print(f"[cube-push-eval] max_episode_length={inner.max_episode_length}")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=inner.device)

    records: list[dict[str, float | int]] = []
    recorded_env_ids: set[int] = set()
    warmup_done = [False]
    orig_reset = inner._reset_idx

    def hooked_reset(env_ids):
        if warmup_done[0] and env_ids is not None and isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
            ids = env_ids.detach().clone()
            terms = inner._push_terms()
            for idx in ids.detach().cpu().tolist():
                if args.record_first_episode_only and idx in recorded_env_ids:
                    continue
                cube_x0 = float((inner._cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item())
                push_dx = float(inner._push_dir_xy[idx, 0].detach().cpu().item())
                push_dy = float(inner._push_dir_xy[idx, 1].detach().cpu().item())
                if int(round(push_dx)) == 1 and int(round(push_dy)) == 0:
                    if cube_x0 < 0.257:
                        posx_bucket = "low_x"
                    elif cube_x0 < 0.308:
                        posx_bucket = "mid_x"
                    else:
                        posx_bucket = "high_x"
                else:
                    posx_bucket = "not_posx"
                records.append(
                    {
                        "trial": len(records),
                        "env_id": int(idx),
                        "cube_x0_m": cube_x0,
                        "cube_y0_m": float(
                            (inner._cube_start_w[idx, 1] - inner.scene.env_origins[idx, 1]).detach().cpu().item()
                        ),
                        "push_dx": push_dx,
                        "push_dy": push_dy,
                        "posx_x_bucket": posx_bucket,
                        "disp_along_push_m": float(terms["disp_along"][idx].detach().cpu().item()),
                        "disp_xy_m": float(terms["disp_xy"][idx].detach().cpu().item()),
                        "target_xy_dist_m": float(terms["target_xy_dist"][idx].detach().cpu().item()),
                        "final_speed_mps": float(terms["speed"][idx].detach().cpu().item()),
                        "tip_angle_deg": float(terms["tip_angle_deg"][idx].detach().cpu().item()),
                        "controlled_push": int(bool(terms["controlled"][idx].detach().cpu().item())),
                        "impact_outlier": int(bool(terms["impact"][idx].detach().cpu().item())),
                        "low_motion": int(bool(terms["low_motion"][idx].detach().cpu().item())),
                        "success_marker": int(bool(inner._push_success_flag[idx].detach().cpu().item())),
                        "grasped_marker": int(bool(inner._grasped[idx].detach().cpu().item())),
                    }
                )
                recorded_env_ids.add(idx)
        orig_reset(env_ids)

    inner._reset_idx = hooked_reset
    inner.episode_length_buf[:] = inner.max_episode_length
    obs = env.get_observations()

    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if args.viewer_step_sleep_s > 0.0:
            time.sleep(args.viewer_step_sleep_s)
    warmup_done[0] = True
    print("[cube-push-eval] warmup truncation fired")

    total_steps = args.num_rollouts * inner.max_episode_length
    print(f"[cube-push-eval] running_steps={total_steps}")
    for _ in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if args.viewer_step_sleep_s > 0.0:
                time.sleep(args.viewer_step_sleep_s)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "env_id",
        "cube_x0_m",
        "cube_y0_m",
        "push_dx",
        "push_dy",
        "posx_x_bucket",
        "disp_along_push_m",
        "disp_xy_m",
        "target_xy_dist_m",
        "final_speed_mps",
        "tip_angle_deg",
        "controlled_push",
        "impact_outlier",
        "low_motion",
        "success_marker",
        "grasped_marker",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    def rate(key: str) -> float:
        return sum(int(r[key]) for r in records) / len(records) if records else 0.0

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records) if records else 0.0

    summary = {
        "controller": "rsl_rl_PPO_policy",
        "learned_policy": True,
        "diffik_controller_used": False,
        "supervised_bc_checkpoint": False,
        "checkpoint": args.checkpoint,
        "num_envs": args.num_envs,
        "num_rollouts": args.num_rollouts,
        "trials": len(records),
        "record_first_episode_only": args.record_first_episode_only,
        "unique_env_records": len(recorded_env_ids),
        "gui": args.gui,
        "livestream": args.livestream,
        "enable_cameras": args.enable_cameras,
        "viewer_step_sleep_s": args.viewer_step_sleep_s,
        "post_run_sleep_s": args.post_run_sleep_s,
        "ik_endpoint_reset": args.ik_endpoint_reset,
        "episode_length_s": env_cfg.episode_length_s,
        "action_scale": env_cfg.action_scale,
        "action_smoothing_alpha": env_cfg.action_smoothing_alpha,
        "max_joint_delta_per_step_rad": env_cfg.max_joint_delta_per_step_rad,
        "contact_joint_delta_scale": env_cfg.contact_joint_delta_scale,
        "fast_cube_joint_delta_scale": env_cfg.fast_cube_joint_delta_scale,
        "joint_target_lead_limit_rad": env_cfg.joint_target_lead_limit_rad,
        "ik_precontact_clearance_m": env_cfg.ik_precontact_clearance_m,
        "scripted_teacher_blend": env_cfg.scripted_teacher_blend,
        "scripted_teacher_horizon_frac": env_cfg.scripted_teacher_horizon_frac,
        "scripted_teacher_goal_push_m": env_cfg.scripted_teacher_goal_push_m,
        "bc_teacher_checkpoint_path": env_cfg.bc_teacher_checkpoint_path,
        "bc_teacher_blend": env_cfg.bc_teacher_blend,
        "bc_teacher_imitation_reward_scale": env_cfg.bc_teacher_imitation_reward_scale,
        "bc_teacher_policy_delta_scale": env_cfg.bc_teacher_policy_delta_scale,
        "bc_teacher_posx_policy_delta_scale": env_cfg.bc_teacher_posx_policy_delta_scale,
        "bc_teacher_lowx_policy_delta_scale": env_cfg.bc_teacher_lowx_policy_delta_scale,
        "bc_teacher_highx_policy_delta_scale": env_cfg.bc_teacher_highx_policy_delta_scale,
        "bc_teacher_delta_smoothing_alpha": env_cfg.bc_teacher_delta_smoothing_alpha,
        "speed_penalty_start_mps": env_cfg.speed_penalty_start_mps,
        "training": False,
        "dataset_generation": False,
        "grasp_attach": False,
        "rollout_object_posewrite": False,
        "reset_context_columns": ["cube_x0_m", "cube_y0_m", "push_dx", "push_dy"],
        "disp_along_push_mean_m": mean("disp_along_push_m"),
        "disp_xy_mean_m": mean("disp_xy_m"),
        "target_xy_dist_mean_m": mean("target_xy_dist_m"),
        "controlled_push_rate": rate("controlled_push"),
        "impact_outlier_rate": rate("impact_outlier"),
        "low_motion_rate": rate("low_motion"),
        "success_marker_rate": rate("success_marker"),
        "grasped_marker_rate": rate("grasped_marker"),
        "out_csv": str(out_csv),
    }
    summary_path = Path(args.summary_json)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "[cube-push-eval] SUMMARY "
        f"trials={summary['trials']} disp_along_push_mean_m={summary['disp_along_push_mean_m']:.6f} "
        f"disp_xy_mean_m={summary['disp_xy_mean_m']:.6f} controlled_push_rate={summary['controlled_push_rate']:.4f} "
        f"impact_outlier_rate={summary['impact_outlier_rate']:.4f} low_motion_rate={summary['low_motion_rate']:.4f} "
        f"success_marker_rate={summary['success_marker_rate']:.4f} grasped_marker_rate={summary['grasped_marker_rate']:.4f}"
    )
    print(f"[cube-push-eval] wrote_csv={out_csv}")
    print(f"[cube-push-eval] wrote_summary={summary_path}")

    if args.post_run_sleep_s > 0.0:
        print(f"[cube-push-eval] post_run_sleep_s={args.post_run_sleep_s}")
        time.sleep(args.post_run_sleep_s)

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
