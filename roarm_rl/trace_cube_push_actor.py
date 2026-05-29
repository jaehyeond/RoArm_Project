"""Trace a frozen cube-push actor against the BC teacher sidecar.

This is a diagnostic tool for the professor cube3cm push/tap branch. It runs a
teacher-off rollout, computes the BC teacher action on the same actor-visited
states for comparison only, and records action/target/motion statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _posx_bucket(cube_x0: float, push_dx: float, push_dy: float) -> str:
    if int(round(push_dx)) == 1 and int(round(push_dy)) == 0:
        if cube_x0 < 0.257:
            return "low_x"
        if cube_x0 < 0.308:
            return "mid_x"
        return "high_x"
    return "not_posx"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bc_teacher_checkpoint_path", required=True)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=896)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--policy_obs_target_mode", choices=("push_target", "bc_teacher_tcp_target"), default="push_target")
    parser.add_argument("--ik_precontact_clearance_m", type=float, default=0.01)
    parser.add_argument("--speed_penalty_start_mps", type=float, default=0.5)
    parser.add_argument("--bc_teacher_policy_delta_clip_rad", type=float, default=0.04)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_posx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=0.8)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=0.85)
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--trace_interval", type=int, default=20)
    parser.add_argument("--max_step_rows", type=int, default=6000)
    parser.add_argument("--out_step_csv", required=True)
    parser.add_argument("--out_env_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--livestream", type=int, default=0, choices=(0, 1, 2))
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, livestream=args.livestream)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg

    env_cfg = RoArmCubePushEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.robot.spawn.usd_path = args.robot_usd_path
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.action_scale = args.action_scale
    env_cfg.action_smoothing_alpha = args.action_smoothing_alpha
    env_cfg.max_joint_delta_per_step_rad = args.max_joint_delta_per_step_rad
    env_cfg.contact_joint_delta_scale = args.contact_joint_delta_scale
    env_cfg.fast_cube_joint_delta_scale = args.fast_cube_joint_delta_scale
    env_cfg.joint_target_lead_limit_rad = args.joint_target_lead_limit_rad
    env_cfg.joint_delta_reference = args.joint_delta_reference
    env_cfg.policy_obs_target_mode = args.policy_obs_target_mode
    env_cfg.ik_precontact_clearance_m = args.ik_precontact_clearance_m
    env_cfg.speed_penalty_start_mps = args.speed_penalty_start_mps
    env_cfg.scripted_teacher_blend = 0.0
    env_cfg.bc_teacher_checkpoint_path = args.bc_teacher_checkpoint_path
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_policy_delta_clip_rad = args.bc_teacher_policy_delta_clip_rad
    env_cfg.bc_teacher_policy_delta_scale = args.bc_teacher_policy_delta_scale
    env_cfg.bc_teacher_posx_policy_delta_scale = args.bc_teacher_posx_policy_delta_scale
    env_cfg.bc_teacher_lowx_policy_delta_scale = args.bc_teacher_lowx_policy_delta_scale
    env_cfg.bc_teacher_highx_policy_delta_scale = args.bc_teacher_highx_policy_delta_scale
    env_cfg.bc_teacher_delta_smoothing_alpha = args.bc_teacher_delta_smoothing_alpha
    env_cfg.bc_teacher_phase_timing = args.bc_teacher_phase_timing

    target_update = (
        f"robot_dof_targets += action_scale({env_cfg.action_scale:.3f}) * action"
        if env_cfg.joint_delta_reference == "target"
        else f"robot_dof_targets = joint_pos + action_scale({env_cfg.action_scale:.3f}) * action"
    )
    print(
        "[cube-push-actor-trace] scope=professor_cube3cm_push_tap "
        "training=NO dataset_generation=NO track_a_runtime=NO grasp_attach=NO rollout_object_posewrite=NO"
    )
    print(
        "[cube-push-actor-trace] mode=teacher_off_rollout_teacher_sidecar_compare "
        f"target_update='{target_update}' action_clip=[-1,1]"
    )
    print(
        "[cube-push-actor-trace] "
        f"checkpoint={args.checkpoint} bc_teacher_checkpoint_path={args.bc_teacher_checkpoint_path}"
    )
    print(
        "[cube-push-actor-trace] env "
        f"num_envs={args.num_envs} seed={args.seed} episode_length_s={env_cfg.episode_length_s} "
        f"joint_delta_reference={env_cfg.joint_delta_reference} "
        f"policy_obs_target_mode={env_cfg.policy_obs_target_mode} "
        f"bc_teacher_phase_timing={env_cfg.bc_teacher_phase_timing} "
        f"lowx_scale={env_cfg.bc_teacher_lowx_policy_delta_scale}"
    )

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    if not getattr(inner, "_bc_teacher_ready", False):
        raise RuntimeError("BC teacher sidecar was not loaded")

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    step_rows: list[dict[str, float | int | str]] = []
    env_rows: list[dict[str, float | int | str]] = []
    active = torch.zeros(inner.num_envs, dtype=torch.bool, device=device)
    active_episode_steps = torch.zeros(inner.num_envs, dtype=torch.long, device=device)
    warmup_done = [False]
    orig_reset = inner._reset_idx

    action_mse_sum = torch.zeros(inner.num_envs, device=device)
    action_mae_sum = torch.zeros(inner.num_envs, device=device)
    arm_mse_sum = torch.zeros(inner.num_envs, device=device)
    actor_abs_sum = torch.zeros(inner.num_envs, device=device)
    teacher_abs_sum = torch.zeros(inner.num_envs, device=device)
    action_cos_sum = torch.zeros(inner.num_envs, device=device)
    eff_abs_sum = torch.zeros(inner.num_envs, device=device)
    eff_vs_actor_mse_sum = torch.zeros(inner.num_envs, device=device)
    joint_move_abs_sum = torch.zeros(inner.num_envs, device=device)
    stat_count = torch.zeros(inner.num_envs, device=device)
    eff_count = torch.zeros(inner.num_envs, device=device)
    min_tcp_cube_dist = torch.full((inner.num_envs,), float("inf"), device=device)
    max_disp_along = torch.full((inner.num_envs,), -float("inf"), device=device)
    first_contact_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    contact_threshold = float(inner.cfg.contact_slowdown_tcp_dist_m)
    arm_ids = torch.as_tensor(inner._bc_arm_joint_ids, dtype=torch.long, device=device)

    phase_stats: dict[str, dict[str, float]] = {
        "approach_or_alpha0": {"n": 0.0, "mse": 0.0, "arm_mse": 0.0},
        "push_alpha01": {"n": 0.0, "mse": 0.0, "arm_mse": 0.0},
        "post_alpha1": {"n": 0.0, "mse": 0.0, "arm_mse": 0.0},
    }

    def append_env_rows(env_ids: torch.Tensor) -> None:
        inner._compute_intermediate_values()
        terms = inner._push_terms()
        for idx in env_ids.detach().cpu().tolist():
            if not bool(active[idx].detach().cpu().item()):
                continue
            cube_x0 = float((inner._cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item())
            cube_y0 = float((inner._cube_start_w[idx, 1] - inner.scene.env_origins[idx, 1]).detach().cpu().item())
            push_dx = float(inner._push_dir_xy[idx, 0].detach().cpu().item())
            push_dy = float(inner._push_dir_xy[idx, 1].detach().cpu().item())
            count = float(max(stat_count[idx].detach().cpu().item(), 1.0))
            eff_n = float(max(eff_count[idx].detach().cpu().item(), 1.0))
            env_rows.append(
                {
                    "env_id": int(idx),
                    "episode_steps": int(active_episode_steps[idx].detach().cpu().item()),
                    "cube_x0_m": cube_x0,
                    "cube_y0_m": cube_y0,
                    "push_dx": push_dx,
                    "push_dy": push_dy,
                    "posx_x_bucket": _posx_bucket(cube_x0, push_dx, push_dy),
                    "mean_action_mse": float((action_mse_sum[idx] / count).detach().cpu().item()),
                    "mean_action_mae": float((action_mae_sum[idx] / count).detach().cpu().item()),
                    "mean_arm_mse": float((arm_mse_sum[idx] / count).detach().cpu().item()),
                    "mean_actor_abs": float((actor_abs_sum[idx] / count).detach().cpu().item()),
                    "mean_teacher_abs": float((teacher_abs_sum[idx] / count).detach().cpu().item()),
                    "mean_action_cos": float((action_cos_sum[idx] / count).detach().cpu().item()),
                    "mean_effective_abs": float((eff_abs_sum[idx] / eff_n).detach().cpu().item()),
                    "mean_effective_vs_actor_mse": float((eff_vs_actor_mse_sum[idx] / eff_n).detach().cpu().item()),
                    "mean_joint_move_abs": float((joint_move_abs_sum[idx] / eff_n).detach().cpu().item()),
                    "min_tcp_cube_dist_m": float(min_tcp_cube_dist[idx].detach().cpu().item()),
                    "first_contact_step": int(first_contact_step[idx].detach().cpu().item()),
                    "max_disp_along_m": float(max_disp_along[idx].detach().cpu().item()),
                    "disp_along_push_m": float(terms["disp_along"][idx].detach().cpu().item()),
                    "disp_xy_m": float(terms["disp_xy"][idx].detach().cpu().item()),
                    "target_xy_dist_m": float(terms["target_xy_dist"][idx].detach().cpu().item()),
                    "final_tcp_cube_dist_m": float(terms["tcp_cube_dist"][idx].detach().cpu().item()),
                    "final_speed_mps": float(terms["speed"][idx].detach().cpu().item()),
                    "tip_angle_deg": float(terms["tip_angle_deg"][idx].detach().cpu().item()),
                    "controlled_push": int(bool(terms["controlled"][idx].detach().cpu().item())),
                    "impact_outlier": int(bool(terms["impact"][idx].detach().cpu().item())),
                    "low_motion": int(bool(terms["low_motion"][idx].detach().cpu().item())),
                    "success_marker": int(bool(inner._push_success_flag[idx].detach().cpu().item())),
                    "grasped_marker": int(bool(inner._grasped[idx].detach().cpu().item())),
                }
            )
            active[idx] = False

    def hooked_reset(env_ids):
        if warmup_done[0] and env_ids is not None and isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
            append_env_rows(env_ids)
        orig_reset(env_ids)

    inner._reset_idx = hooked_reset

    obs = env.get_observations()
    inner.episode_length_buf[:] = inner.max_episode_length
    with torch.inference_mode():
        warmup_actions = policy(obs)
        obs, _, _, _ = env.step(warmup_actions)
    warmup_done[0] = True
    active[:] = True
    print("[cube-push-actor-trace] warmup truncation fired")

    total_steps = int(inner.max_episode_length)
    print(f"[cube-push-actor-trace] running_steps={total_steps}")
    for step in range(total_steps):
        active_pre = active.clone()
        if not bool(active_pre.any().detach().cpu().item()):
            break
        with torch.inference_mode():
            actions = policy(obs).detach().clamp(-1.0, 1.0)
            teacher_actions = inner._bc_teacher_actions().detach().clamp(-1.0, 1.0)
            traj = inner._bc_teacher_traj()
            alpha = inner._bc_teacher_phase_alpha(traj).detach()
            joint_pos_before = inner._robot.data.joint_pos.detach().clone()
            target_before = inner.robot_dof_targets.detach().clone()
            err = actions - teacher_actions
            arm_err = actions[:, arm_ids] - teacher_actions[:, arm_ids]
            action_mse = torch.mean(err.square(), dim=-1)
            action_mae = torch.mean(torch.abs(err), dim=-1)
            arm_mse = torch.mean(arm_err.square(), dim=-1)
            actor_abs = torch.mean(torch.abs(actions), dim=-1)
            teacher_abs = torch.mean(torch.abs(teacher_actions), dim=-1)
            actor_norm = torch.linalg.norm(actions, dim=-1)
            teacher_norm = torch.linalg.norm(teacher_actions, dim=-1)
            action_cos = torch.sum(actions * teacher_actions, dim=-1) / (actor_norm * teacher_norm + 1.0e-6)

            active_ids = torch.nonzero(active_pre, as_tuple=False).squeeze(-1)
            action_mse_sum[active_ids] += action_mse[active_ids]
            action_mae_sum[active_ids] += action_mae[active_ids]
            arm_mse_sum[active_ids] += arm_mse[active_ids]
            actor_abs_sum[active_ids] += actor_abs[active_ids]
            teacher_abs_sum[active_ids] += teacher_abs[active_ids]
            action_cos_sum[active_ids] += action_cos[active_ids]
            stat_count[active_ids] += 1.0
            active_episode_steps[active_ids] += 1

            phase_masks = {
                "approach_or_alpha0": active_pre & (alpha <= 1.0e-6),
                "push_alpha01": active_pre & (alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6),
                "post_alpha1": active_pre & (alpha >= 1.0 - 1.0e-6),
            }
            for name, mask in phase_masks.items():
                if bool(mask.any().detach().cpu().item()):
                    phase_stats[name]["n"] += float(mask.sum().detach().cpu().item())
                    phase_stats[name]["mse"] += float(action_mse[mask].sum().detach().cpu().item())
                    phase_stats[name]["arm_mse"] += float(arm_mse[mask].sum().detach().cpu().item())

            obs, _, _, _ = env.step(actions)

            still_active = active_pre & active
            if bool(still_active.any().detach().cpu().item()):
                target_base = target_before
                if env_cfg.joint_delta_reference == "joint_pos":
                    target_base = joint_pos_before
                target_delta = inner.robot_dof_targets.detach() - target_base
                effective_action = target_delta / max(float(env_cfg.action_scale), 1.0e-6)
                effective_action[:, inner.gripper_joint_idx] = 0.0
                joint_move = inner._robot.data.joint_pos.detach() - joint_pos_before
                eff_vs_actor_mse = torch.mean((effective_action - actions).square(), dim=-1)
                eff_abs = torch.mean(torch.abs(effective_action), dim=-1)
                joint_move_abs = torch.mean(torch.abs(joint_move), dim=-1)
                terms = inner._push_terms()
                min_tcp_cube_dist[still_active] = torch.minimum(
                    min_tcp_cube_dist[still_active], terms["tcp_cube_dist"][still_active]
                )
                max_disp_along[still_active] = torch.maximum(max_disp_along[still_active], terms["disp_along"][still_active])
                contact_now = still_active & (terms["tcp_cube_dist"] <= contact_threshold) & (first_contact_step < 0)
                first_contact_step[contact_now] = step
                eff_ids = torch.nonzero(still_active, as_tuple=False).squeeze(-1)
                eff_abs_sum[eff_ids] += eff_abs[eff_ids]
                eff_vs_actor_mse_sum[eff_ids] += eff_vs_actor_mse[eff_ids]
                joint_move_abs_sum[eff_ids] += joint_move_abs[eff_ids]
                eff_count[eff_ids] += 1.0

                should_trace = (step % max(1, int(args.trace_interval)) == 0) or step == total_steps - 1
                if should_trace and len(step_rows) < int(args.max_step_rows):
                    remaining = int(args.max_step_rows) - len(step_rows)
                    take_ids = eff_ids[:remaining]
                    for idx in take_ids.detach().cpu().tolist():
                        cube_x0 = float(
                            (inner._cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item()
                        )
                        push_dx = float(inner._push_dir_xy[idx, 0].detach().cpu().item())
                        push_dy = float(inner._push_dir_xy[idx, 1].detach().cpu().item())
                        row: dict[str, float | int | str] = {
                            "step": int(step),
                            "env_id": int(idx),
                            "push_dx": push_dx,
                            "push_dy": push_dy,
                            "posx_x_bucket": _posx_bucket(cube_x0, push_dx, push_dy),
                            "phase_alpha": float(alpha[idx].detach().cpu().item()),
                            "action_mse": float(action_mse[idx].detach().cpu().item()),
                            "arm_mse": float(arm_mse[idx].detach().cpu().item()),
                            "action_mae": float(action_mae[idx].detach().cpu().item()),
                            "action_cos": float(action_cos[idx].detach().cpu().item()),
                            "actor_abs": float(actor_abs[idx].detach().cpu().item()),
                            "teacher_abs": float(teacher_abs[idx].detach().cpu().item()),
                            "effective_abs": float(eff_abs[idx].detach().cpu().item()),
                            "effective_vs_actor_mse": float(eff_vs_actor_mse[idx].detach().cpu().item()),
                            "joint_move_abs": float(joint_move_abs[idx].detach().cpu().item()),
                            "tcp_cube_dist_m": float(terms["tcp_cube_dist"][idx].detach().cpu().item()),
                            "disp_along_m": float(terms["disp_along"][idx].detach().cpu().item()),
                            "disp_xy_m": float(terms["disp_xy"][idx].detach().cpu().item()),
                            "target_xy_dist_m": float(terms["target_xy_dist"][idx].detach().cpu().item()),
                            "contact_slowdown": float(inner._last_contact_slowdown[idx].detach().cpu().item()),
                        }
                        for joint_i in range(actions.shape[1]):
                            row[f"actor_a{joint_i}"] = float(actions[idx, joint_i].detach().cpu().item())
                            row[f"teacher_a{joint_i}"] = float(teacher_actions[idx, joint_i].detach().cpu().item())
                            row[f"effective_a{joint_i}"] = float(effective_action[idx, joint_i].detach().cpu().item())
                        step_rows.append(row)

    remaining_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
    if remaining_ids.numel() > 0:
        append_env_rows(remaining_ids)

    out_step_csv = Path(args.out_step_csv)
    out_env_csv = Path(args.out_env_csv)
    summary_path = Path(args.summary_json)
    out_step_csv.parent.mkdir(parents=True, exist_ok=True)
    out_env_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    step_fieldnames = list(step_rows[0].keys()) if step_rows else ["step", "env_id"]
    with out_step_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    env_fieldnames = list(env_rows[0].keys()) if env_rows else ["env_id"]
    with out_env_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=env_fieldnames)
        writer.writeheader()
        writer.writerows(env_rows)

    grouped: dict[str, dict[str, float | int]] = {}
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in env_rows:
        groups[str(row["posx_x_bucket"])].append(row)
        groups[f"dir_{int(round(float(row['push_dx'])))}_{int(round(float(row['push_dy'])))}"].append(row)
    for name, rows in sorted(groups.items()):
        grouped[name] = {
            "n": len(rows),
            "success": _rate([int(r["success_marker"]) for r in rows]),
            "controlled": _rate([int(r["controlled_push"]) for r in rows]),
            "low_motion": _rate([int(r["low_motion"]) for r in rows]),
            "mean_action_mse": _mean([float(r["mean_action_mse"]) for r in rows]),
            "mean_arm_mse": _mean([float(r["mean_arm_mse"]) for r in rows]),
            "mean_actor_abs": _mean([float(r["mean_actor_abs"]) for r in rows]),
            "mean_teacher_abs": _mean([float(r["mean_teacher_abs"]) for r in rows]),
            "min_tcp_cube_dist_m": _mean([float(r["min_tcp_cube_dist_m"]) for r in rows]),
            "disp_along_push_m": _mean([float(r["disp_along_push_m"]) for r in rows]),
        }

    phase_summary = {}
    for name, stats in phase_stats.items():
        n = max(float(stats["n"]), 1.0)
        phase_summary[name] = {
            "samples": int(stats["n"]),
            "mean_action_mse": float(stats["mse"] / n),
            "mean_arm_mse": float(stats["arm_mse"] / n),
        }

    summary = {
        "controller": "rsl_rl_PPO_policy",
        "learned_policy": True,
        "trace_only": True,
        "training": False,
        "dataset_generation": False,
        "track_a_runtime": False,
        "grasp_attach": False,
        "rollout_object_posewrite": False,
        "checkpoint": args.checkpoint,
        "bc_teacher_checkpoint_path": args.bc_teacher_checkpoint_path,
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "traced_envs": len(env_rows),
        "step_rows": len(step_rows),
        "episode_length_s": float(env_cfg.episode_length_s),
        "action_scale": float(env_cfg.action_scale),
        "joint_delta_reference": str(env_cfg.joint_delta_reference),
        "policy_obs_target_mode": str(env_cfg.policy_obs_target_mode),
        "bc_teacher_phase_timing": str(env_cfg.bc_teacher_phase_timing),
        "bc_teacher_blend": float(env_cfg.bc_teacher_blend),
        "bc_teacher_imitation_reward_scale": float(env_cfg.bc_teacher_imitation_reward_scale),
        "mean_action_mse": _mean([float(r["mean_action_mse"]) for r in env_rows]),
        "mean_action_mae": _mean([float(r["mean_action_mae"]) for r in env_rows]),
        "mean_arm_mse": _mean([float(r["mean_arm_mse"]) for r in env_rows]),
        "mean_actor_abs": _mean([float(r["mean_actor_abs"]) for r in env_rows]),
        "mean_teacher_abs": _mean([float(r["mean_teacher_abs"]) for r in env_rows]),
        "mean_action_cos": _mean([float(r["mean_action_cos"]) for r in env_rows]),
        "mean_effective_abs": _mean([float(r["mean_effective_abs"]) for r in env_rows]),
        "mean_effective_vs_actor_mse": _mean([float(r["mean_effective_vs_actor_mse"]) for r in env_rows]),
        "mean_joint_move_abs": _mean([float(r["mean_joint_move_abs"]) for r in env_rows]),
        "min_tcp_cube_dist_mean_m": _mean([float(r["min_tcp_cube_dist_m"]) for r in env_rows]),
        "contact_reached_rate": _rate([1 if int(r["first_contact_step"]) >= 0 else 0 for r in env_rows]),
        "disp_along_push_mean_m": _mean([float(r["disp_along_push_m"]) for r in env_rows]),
        "disp_xy_mean_m": _mean([float(r["disp_xy_m"]) for r in env_rows]),
        "controlled_push_rate": _rate([int(r["controlled_push"]) for r in env_rows]),
        "impact_outlier_rate": _rate([int(r["impact_outlier"]) for r in env_rows]),
        "low_motion_rate": _rate([int(r["low_motion"]) for r in env_rows]),
        "success_marker_rate": _rate([int(r["success_marker"]) for r in env_rows]),
        "phase_summary": phase_summary,
        "grouped": grouped,
        "out_step_csv": str(out_step_csv),
        "out_env_csv": str(out_env_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "actor_trace line1 "
        f"traced_envs={summary['traced_envs']} step_rows={summary['step_rows']} "
        f"teacher_off=YES teacher_sidecar=COMPARE_ONLY"
    )
    print(
        "actor_trace line2 "
        f"mean_action_mse={summary['mean_action_mse']:.9f} "
        f"mean_arm_mse={summary['mean_arm_mse']:.9f} "
        f"mean_action_mae={summary['mean_action_mae']:.9f} "
        f"mean_action_cos={summary['mean_action_cos']:.9f}"
    )
    print(
        "actor_trace line3 "
        f"mean_actor_abs={summary['mean_actor_abs']:.9f} "
        f"mean_teacher_abs={summary['mean_teacher_abs']:.9f} "
        f"mean_effective_abs={summary['mean_effective_abs']:.9f} "
        f"mean_effective_vs_actor_mse={summary['mean_effective_vs_actor_mse']:.9f}"
    )
    print(
        "actor_trace line4 "
        f"mean_joint_move_abs={summary['mean_joint_move_abs']:.9f} "
        f"min_tcp_cube_dist_mean_m={summary['min_tcp_cube_dist_mean_m']:.9f} "
        f"contact_reached_rate={summary['contact_reached_rate']:.9f}"
    )
    print(
        "actor_trace line5 "
        f"controlled={summary['controlled_push_rate']:.9f} "
        f"impact={summary['impact_outlier_rate']:.9f} "
        f"low_motion={summary['low_motion_rate']:.9f} "
        f"success={summary['success_marker_rate']:.9f} "
        f"disp_along_mean_m={summary['disp_along_push_mean_m']:.9f}"
    )
    print(
        "actor_trace line6 "
        f"phase={json.dumps(phase_summary, sort_keys=True)}"
    )
    print(f"actor_trace line7 grouped={json.dumps(grouped, sort_keys=True)}")
    print(f"actor_trace line8 wrote_step_csv={out_step_csv} wrote_env_csv={out_env_csv} wrote_summary={summary_path}")

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
