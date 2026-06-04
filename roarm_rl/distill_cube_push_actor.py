"""Supervised distillation into the rsl_rl cube-push actor.

This is for the professor cube3cm push/tap branch only. It launches the
RoArm-CubePush env, rolls out the already-audited BC teacher through the
direct-step/joint-pos action loop, trains the rsl_rl actor mean to predict the
teacher's normalized joint-delta actions, and writes a normal rsl_rl checkpoint
that can be evaluated by eval_cube_push_policy.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _tensor_from_obs(obs):
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    return obs


def _make_actor(torch, obs_dim: int, action_dim: int):
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, action_dim),
    )


def _load_actor_from_checkpoint(actor, state_dict):
    actor_state = {
        "0.weight": state_dict["actor.0.weight"],
        "0.bias": state_dict["actor.0.bias"],
        "2.weight": state_dict["actor.2.weight"],
        "2.bias": state_dict["actor.2.bias"],
        "4.weight": state_dict["actor.4.weight"],
        "4.bias": state_dict["actor.4.bias"],
        "6.weight": state_dict["actor.6.weight"],
        "6.bias": state_dict["actor.6.bias"],
    }
    actor.load_state_dict(actor_state)


def _store_actor_to_checkpoint(actor, state_dict):
    actor_state = actor.state_dict()
    state_dict["actor.0.weight"] = actor_state["0.weight"].detach().cpu()
    state_dict["actor.0.bias"] = actor_state["0.bias"].detach().cpu()
    state_dict["actor.2.weight"] = actor_state["2.weight"].detach().cpu()
    state_dict["actor.2.bias"] = actor_state["2.bias"].detach().cpu()
    state_dict["actor.4.weight"] = actor_state["4.weight"].detach().cpu()
    state_dict["actor.4.bias"] = actor_state["4.bias"].detach().cpu()
    state_dict["actor.6.weight"] = actor_state["6.weight"].detach().cpu()
    state_dict["actor.6.bias"] = actor_state["6.bias"].detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--out_checkpoint", type=Path, required=True)
    parser.add_argument("--metrics_json", type=Path, required=True)
    parser.add_argument("--collection_mode", choices=("teacher", "actor"), default="teacher")
    parser.add_argument("--collection_checkpoint", type=Path, default=None)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--bc_teacher_checkpoint_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--collect_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=894)
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
    parser.add_argument("--bc_teacher_policy_delta_clip_rad", type=float, default=0.04)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_posx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=0.8)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=0.85)
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--loss_posx_low_weight", type=float, default=1.0)
    parser.add_argument("--loss_posx_mid_weight", type=float, default=1.0)
    parser.add_argument("--loss_posx_high_weight", type=float, default=1.0)
    parser.add_argument("--loss_push_phase_weight", type=float, default=1.0)
    parser.add_argument("--loss_post_phase_weight", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.10)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubePushEnvCfg

    torch.manual_seed(int(args.seed))

    env_cfg = RoArmCubePushEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.action_scale = float(args.action_scale)
    env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.policy_obs_target_mode = str(args.policy_obs_target_mode)
    env_cfg.ik_precontact_clearance_m = float(args.ik_precontact_clearance_m)
    env_cfg.bc_teacher_checkpoint_path = str(args.bc_teacher_checkpoint_path)
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_policy_delta_clip_rad = float(args.bc_teacher_policy_delta_clip_rad)
    env_cfg.bc_teacher_policy_delta_scale = float(args.bc_teacher_policy_delta_scale)
    env_cfg.bc_teacher_posx_policy_delta_scale = float(args.bc_teacher_posx_policy_delta_scale)
    env_cfg.bc_teacher_lowx_policy_delta_scale = float(args.bc_teacher_lowx_policy_delta_scale)
    env_cfg.bc_teacher_highx_policy_delta_scale = float(args.bc_teacher_highx_policy_delta_scale)
    env_cfg.bc_teacher_delta_smoothing_alpha = float(args.bc_teacher_delta_smoothing_alpha)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)

    print(
        "[cube-push-actor-distill] scope=professor_cube3cm_push_tap "
        "training=YES dataset_generation=NO track_a_runtime=NO grasp_attach=NO rollout_object_posewrite=NO"
    )
    print(
        "[cube-push-actor-distill] action_semantics=normalized_joint_delta "
        f"target_update='robot_dof_targets = joint_pos + action_scale({env_cfg.action_scale:.3f}) * action' "
        "teacher_source=bc_teacher_actions"
    )
    print(
        "[cube-push-actor-distill] env "
        f"num_envs={args.num_envs} collect_steps={args.collect_steps} seed={args.seed} "
        f"episode_length_s={env_cfg.episode_length_s} joint_delta_reference={env_cfg.joint_delta_reference} "
        f"policy_obs_target_mode={env_cfg.policy_obs_target_mode} "
        f"bc_teacher_phase_timing={env_cfg.bc_teacher_phase_timing} lowx_scale={env_cfg.bc_teacher_lowx_policy_delta_scale}"
    )
    collection_checkpoint = args.collection_checkpoint or args.base_checkpoint
    print(
        "[cube-push-actor-distill] collection "
        f"mode={args.collection_mode} collection_checkpoint={collection_checkpoint}"
    )

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    collection_policy = None
    if args.collection_mode == "actor":
        ppo_cfg = RoArmPickPPORunnerCfg()
        ppo_cfg.seed = int(args.seed)
        collection_runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=device)
        collection_runner.load(str(collection_checkpoint))
        collection_policy = collection_runner.get_inference_policy(device=device)

    obs = env.get_observations()
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)
    inner.episode_length_buf[:] = inner.max_episode_length
    obs, _, _, _ = env.step(zero)

    obs_batches = []
    action_batches = []
    weight_batches = []
    step_action_abs_batches = []
    with torch.inference_mode():
        for _ in range(int(args.collect_steps)):
            policy_obs = _tensor_from_obs(obs).detach()
            teacher_actions = inner._bc_teacher_actions().detach().clamp(-1.0, 1.0)
            step_actions = teacher_actions
            if collection_policy is not None:
                step_actions = collection_policy(obs).detach().clamp(-1.0, 1.0)
            traj = inner._bc_teacher_traj()
            alpha = inner._bc_teacher_phase_alpha(traj).detach()
            cube_x_local = inner._cube_start_w[:, 0] - inner.scene.env_origins[:, 0]
            edge0 = float(inner.cfg.bc_teacher_x_bucket_edge0_m)
            edge1 = float(inner.cfg.bc_teacher_x_bucket_edge1_m)
            posx = traj["posx"].detach()
            posx_low = posx & (cube_x_local < edge0)
            posx_mid = posx & (cube_x_local >= edge0) & (cube_x_local < edge1)
            posx_high = posx & (cube_x_local >= edge1)
            weights = torch.ones((inner.num_envs,), device=device, dtype=torch.float32)
            weights = torch.where(posx_low, weights * float(args.loss_posx_low_weight), weights)
            weights = torch.where(posx_mid, weights * float(args.loss_posx_mid_weight), weights)
            weights = torch.where(posx_high, weights * float(args.loss_posx_high_weight), weights)
            push_phase = (alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)
            post_phase = alpha >= 1.0 - 1.0e-6
            weights = torch.where(push_phase, weights * float(args.loss_push_phase_weight), weights)
            weights = torch.where(post_phase, weights * float(args.loss_post_phase_weight), weights)
            obs_batches.append(policy_obs.cpu())
            action_batches.append(teacher_actions.cpu())
            weight_batches.append(weights.cpu())
            step_action_abs_batches.append(torch.mean(torch.abs(step_actions), dim=-1).cpu())
            obs, _, _, _ = env.step(step_actions)

    obs_all = torch.cat(obs_batches, dim=0).to(device=device, dtype=torch.float32)
    actions_all = torch.cat(action_batches, dim=0).to(device=device, dtype=torch.float32)
    weights_all = torch.cat(weight_batches, dim=0).to(device=device, dtype=torch.float32).clamp_min(1.0e-6)
    sample_count = int(obs_all.shape[0])
    obs_dim = int(obs_all.shape[1])
    action_dim = int(actions_all.shape[1])
    if obs_dim != 28 or action_dim != 6:
        raise ValueError(f"unexpected rsl_rl actor dims obs={obs_dim} action={action_dim}")

    obs_mean = obs_all.mean(dim=0, keepdim=True)
    obs_std = obs_all.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    obs_var = obs_std.square()
    obs_norm = (obs_all - obs_mean) / obs_std

    order = torch.randperm(sample_count, device=device)
    val_count = max(1, int(sample_count * max(0.0, min(0.5, float(args.val_frac)))))
    val_idx = order[:val_count]
    train_idx = order[val_count:]

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    state_dict = base["model_state_dict"]
    actor = _make_actor(torch, obs_dim, action_dim).to(device=device)
    _load_actor_from_checkpoint(actor, state_dict)
    opt = torch.optim.AdamW(actor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        per_sample = torch.mean((pred - target) ** 2, dim=-1)
        return torch.sum(per_sample * weights) / torch.sum(weights)

    with torch.no_grad():
        initial_train_mse = torch.mean((actor(obs_norm[train_idx]) - actions_all[train_idx]) ** 2).item()
        initial_val_mse = torch.mean((actor(obs_norm[val_idx]) - actions_all[val_idx]) ** 2).item()
        initial_train_weighted_mse = weighted_mse(
            actor(obs_norm[train_idx]), actions_all[train_idx], weights_all[train_idx]
        ).item()
        initial_val_weighted_mse = weighted_mse(
            actor(obs_norm[val_idx]), actions_all[val_idx], weights_all[val_idx]
        ).item()

    batch_size = max(1, int(args.batch_size))
    train_count = int(train_idx.numel())
    for _ in range(int(args.epochs)):
        shuffled = train_idx[torch.randperm(train_count, device=device)]
        for start in range(0, train_count, batch_size):
            idx = shuffled[start : start + batch_size]
            pred = actor(obs_norm[idx])
            loss = weighted_mse(pred, actions_all[idx], weights_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    actor.eval()
    with torch.no_grad():
        train_pred = actor(obs_norm[train_idx])
        val_pred = actor(obs_norm[val_idx])
        final_train_mse = torch.mean((train_pred - actions_all[train_idx]) ** 2).item()
        final_val_mse = torch.mean((val_pred - actions_all[val_idx]) ** 2).item()
        final_train_weighted_mse = weighted_mse(train_pred, actions_all[train_idx], weights_all[train_idx]).item()
        final_val_weighted_mse = weighted_mse(val_pred, actions_all[val_idx], weights_all[val_idx]).item()
        final_val_mae = torch.mean(torch.abs(val_pred - actions_all[val_idx])).item()
        final_val_max_abs = torch.max(torch.abs(val_pred - actions_all[val_idx])).item()
        teacher_abs_mean = torch.mean(torch.abs(actions_all)).item()
        pred_abs_mean = torch.mean(torch.abs(actor(obs_norm))).item()
        collection_step_action_abs_mean = torch.cat(step_action_abs_batches).mean().item()
        sample_weight_mean = torch.mean(weights_all).item()
        sample_weight_max = torch.max(weights_all).item()

    _store_actor_to_checkpoint(actor, state_dict)
    state_dict["actor_obs_normalizer._mean"] = obs_mean.detach().cpu()
    state_dict["actor_obs_normalizer._std"] = obs_std.detach().cpu()
    state_dict["actor_obs_normalizer._var"] = obs_var.detach().cpu()
    state_dict["actor_obs_normalizer.count"] = torch.tensor(sample_count, dtype=torch.long)
    state_dict["critic_obs_normalizer._mean"] = obs_mean.detach().cpu()
    state_dict["critic_obs_normalizer._std"] = obs_std.detach().cpu()
    state_dict["critic_obs_normalizer._var"] = obs_var.detach().cpu()
    state_dict["critic_obs_normalizer.count"] = torch.tensor(sample_count, dtype=torch.long)

    base["model_state_dict"] = state_dict
    base["iter"] = int(base.get("iter", 0))
    base["distillation_info"] = {
        "source": "bc_teacher_actions",
        "collection_mode": str(args.collection_mode),
        "collection_checkpoint": str(collection_checkpoint),
        "sample_count": sample_count,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "base_checkpoint": str(args.base_checkpoint),
        "bc_teacher_checkpoint_path": str(args.bc_teacher_checkpoint_path),
    }
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.out_checkpoint)

    metrics = {
        "verdict": "PASS_ACTOR_DISTILLATION" if final_val_mse < initial_val_mse * 0.2 else "CHECK_ACTOR_DISTILLATION",
        "base_checkpoint": str(args.base_checkpoint),
        "out_checkpoint": str(args.out_checkpoint),
        "collection_mode": str(args.collection_mode),
        "collection_checkpoint": str(collection_checkpoint),
        "bc_teacher_checkpoint_path": str(args.bc_teacher_checkpoint_path),
        "num_envs": int(args.num_envs),
        "collect_steps": int(args.collect_steps),
        "sample_count": sample_count,
        "train_count": int(train_idx.numel()),
        "val_count": int(val_idx.numel()),
        "epochs": int(args.epochs),
        "batch_size": batch_size,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "initial_train_mse": float(initial_train_mse),
        "initial_val_mse": float(initial_val_mse),
        "initial_train_weighted_mse": float(initial_train_weighted_mse),
        "initial_val_weighted_mse": float(initial_val_weighted_mse),
        "final_train_mse": float(final_train_mse),
        "final_val_mse": float(final_val_mse),
        "final_train_weighted_mse": float(final_train_weighted_mse),
        "final_val_weighted_mse": float(final_val_weighted_mse),
        "final_val_mae": float(final_val_mae),
        "final_val_max_abs": float(final_val_max_abs),
        "teacher_action_abs_mean": float(teacher_abs_mean),
        "collection_step_action_abs_mean": float(collection_step_action_abs_mean),
        "actor_action_abs_mean": float(pred_abs_mean),
        "sample_weight_mean": float(sample_weight_mean),
        "sample_weight_max": float(sample_weight_max),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "episode_length_s": float(env_cfg.episode_length_s),
        "action_scale": float(env_cfg.action_scale),
        "action_smoothing_alpha": float(env_cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(env_cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(env_cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(env_cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(env_cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(env_cfg.joint_delta_reference),
        "policy_obs_target_mode": str(env_cfg.policy_obs_target_mode),
        "bc_teacher_lowx_policy_delta_scale": float(env_cfg.bc_teacher_lowx_policy_delta_scale),
        "bc_teacher_highx_policy_delta_scale": float(env_cfg.bc_teacher_highx_policy_delta_scale),
        "bc_teacher_delta_smoothing_alpha": float(env_cfg.bc_teacher_delta_smoothing_alpha),
        "bc_teacher_phase_timing": str(env_cfg.bc_teacher_phase_timing),
        "loss_posx_low_weight": float(args.loss_posx_low_weight),
        "loss_posx_mid_weight": float(args.loss_posx_mid_weight),
        "loss_posx_high_weight": float(args.loss_posx_high_weight),
        "loss_push_phase_weight": float(args.loss_push_phase_weight),
        "loss_post_phase_weight": float(args.loss_post_phase_weight),
        "training": True,
        "dataset_generation": False,
        "track_a_runtime": False,
        "teacher_off_rollout_validated": False,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(
        "actor_distill line1 "
        f"samples={sample_count} obs_dim={obs_dim} action_dim={action_dim} train={int(train_idx.numel())} "
        f"val={int(val_idx.numel())} collection_mode={args.collection_mode}"
    )
    print(
        "actor_distill line2 "
        f"initial_train_mse={initial_train_mse:.9f} initial_val_mse={initial_val_mse:.9f} "
        f"final_train_mse={final_train_mse:.9f} final_val_mse={final_val_mse:.9f}"
    )
    print(
        "actor_distill line2b "
        f"initial_train_weighted_mse={initial_train_weighted_mse:.9f} "
        f"initial_val_weighted_mse={initial_val_weighted_mse:.9f} "
        f"final_train_weighted_mse={final_train_weighted_mse:.9f} "
        f"final_val_weighted_mse={final_val_weighted_mse:.9f}"
    )
    print(
        "actor_distill line3 "
        f"final_val_mae={final_val_mae:.9f} final_val_max_abs={final_val_max_abs:.9f} "
        f"teacher_action_abs_mean={teacher_abs_mean:.9f} collection_step_action_abs_mean={collection_step_action_abs_mean:.9f} "
        f"actor_action_abs_mean={pred_abs_mean:.9f}"
    )
    print(
        "actor_distill line3b "
        f"sample_weight_mean={sample_weight_mean:.9f} sample_weight_max={sample_weight_max:.9f} "
        f"loss_posx_low_weight={float(args.loss_posx_low_weight):.3f} "
        f"loss_posx_mid_weight={float(args.loss_posx_mid_weight):.3f} "
        f"loss_posx_high_weight={float(args.loss_posx_high_weight):.3f} "
        f"loss_push_phase_weight={float(args.loss_push_phase_weight):.3f} "
        f"loss_post_phase_weight={float(args.loss_post_phase_weight):.3f}"
    )
    print(
        "actor_distill line4 "
        f"checkpoint={args.out_checkpoint} metrics={args.metrics_json} "
        f"verdict={metrics['verdict']} teacher_off_rollout_validated=NO"
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
