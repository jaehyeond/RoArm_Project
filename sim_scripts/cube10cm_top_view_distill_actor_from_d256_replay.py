#!/usr/bin/env python3
"""Supervised actor warm-start from D256 recorded-action oracle replay.

This is not PPO training. It collects live observations while replaying D256
recorded joint targets through the normal policy action interface, then fits the
rsl_rl actor to those oracle actions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
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
DEFAULT_D256_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_SOURCE_ACTOR_CHECKPOINT = (
    RUNTIME_ROOT
    / "actor_preserve_d285"
    / "tap10cm"
    / "ppo_actorfreeze_noise002_10_smoke"
    / "cube10cm_d285_actorfreeze_noise002_10_smoke"
    / "model_9.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "actor_d256_replay_distill_d289" / "tap10cm"
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


def _tensor_mean(x) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_max(x) -> float:
    return float(x.detach().float().max().cpu().item())


def _metrics(torch, pred, target) -> dict[str, float]:
    diff = pred - target
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)
    return {
        "mse": _tensor_mean(torch.mean(diff * diff, dim=-1)),
        "mae": _tensor_mean(torch.mean(torch.abs(diff), dim=-1)),
        "cosine": _tensor_mean(cosine),
        "pred_abs_mean": _tensor_mean(torch.mean(torch.abs(pred), dim=-1)),
        "target_abs_mean": _tensor_mean(torch.mean(torch.abs(target), dim=-1)),
        "pred_abs_max": _tensor_max(torch.abs(pred)),
        "target_abs_max": _tensor_max(torch.abs(target)),
    }


def _per_dim_metrics(torch, pred, target, labels: list[str]) -> list[dict[str, float | int | str]]:
    diff = pred - target
    out = []
    for idx, label in enumerate(labels):
        out.append(
            {
                "dim": idx,
                "label": label,
                "pred_abs_mean": _tensor_mean(torch.abs(pred[:, idx])),
                "target_abs_mean": _tensor_mean(torch.abs(target[:, idx])),
                "abs_gap_mean": _tensor_mean(torch.abs(diff[:, idx])),
                "mse": _tensor_mean(diff[:, idx] * diff[:, idx]),
                "pred_signed_mean": _tensor_mean(pred[:, idx]),
                "target_signed_mean": _tensor_mean(target[:, idx]),
            }
        )
    return out


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D289 Actor Distillation From D256 Replay",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source actor: `{summary['source_actor_checkpoint']}`",
        f"- distilled checkpoint: `{summary['distilled_actor_checkpoint']}`",
        f"- samples train/val: `{summary['train_samples']}` / `{summary['val_samples']}`",
        f"- oracle replay contact/useful/reaction: `{summary['oracle_contact_seen_rate']}` / `{summary['oracle_useful_seen_rate']}` / `{summary['oracle_reaction_seen_rate']}`",
        f"- oracle replay overshoot: `{summary['oracle_overshoot_seen_rate']}`",
        f"- oracle max XY mean/max: `{summary['oracle_max_disp_xy_mean_m']}` / `{summary['oracle_max_disp_xy_max_m']}`",
        f"- target action abs mean/max: `{summary['target_action_abs_mean']}` / `{summary['target_action_abs_max']}`",
        f"- target action clip rate mean/max: `{summary['target_action_clip_rate_mean']}` / `{summary['target_action_clip_rate_max']}`",
        f"- initial val MSE/MAE/cosine: `{summary['initial_val_metrics']['mse']}` / `{summary['initial_val_metrics']['mae']}` / `{summary['initial_val_metrics']['cosine']}`",
        f"- final val MSE/MAE/cosine: `{summary['final_val_metrics']['mse']}` / `{summary['final_val_metrics']['mae']}` / `{summary['final_val_metrics']['cosine']}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is supervised actor warm-start from D256 recorded-action oracle replay. It does not train PPO and does not use the D257 MLP teacher as an action target.",
            "Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics with the same action contract.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_actor_checkpoint", type=Path, default=DEFAULT_SOURCE_ACTOR_CHECKPOINT)
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--collection_episode_count", type=int, default=None)
    parser.add_argument("--episode_min", type=int, default=None)
    parser.add_argument("--episode_max", type=int, default=None)
    parser.add_argument("--episode_indices", type=str, default="")
    parser.add_argument("--fresh_env_per_batch", action="store_true")
    parser.add_argument("--dataset_out", type=Path, default=None)
    parser.add_argument("--dataset_only", action="store_true")
    parser.add_argument("--seed", type=int, default=28901)
    parser.add_argument("--collect_steps", type=int, default=580)
    parser.add_argument("--hold_steps", type=int, default=3)
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--max_val_mse", type=float, default=0.04)
    parser.add_argument("--min_val_cosine", type=float, default=0.80)
    parser.add_argument("--artifact_tag", type=str, default="d289_d256_replay_actor_distill")
    args = parser.parse_args()

    if int(args.collect_steps) <= 0:
        raise ValueError("--collect_steps must be positive")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if bool(args.dataset_only) and args.dataset_out is None:
        raise ValueError("--dataset_only requires --dataset_out")
    collection_episode_count = (
        int(args.collection_episode_count)
        if args.collection_episode_count is not None
        else int(args.num_envs)
    )
    if collection_episode_count <= 0:
        raise ValueError("--collection_episode_count must be positive")
    if collection_episode_count % int(args.num_envs) != 0:
        raise ValueError("--collection_episode_count must be divisible by --num_envs")
    if bool(args.fresh_env_per_batch):
        raise ValueError(
            "--fresh_env_per_batch is disabled: Isaac env close/recreate hung in D289 smoke. "
            "Use separate-process single-batch collection instead."
        )
    if collection_episode_count > int(args.num_envs):
        raise ValueError(
            "multi-batch collection inside one Isaac process is unsafe for this probe: "
            "D289 reproduced replay contamination after the first batch. "
            "Use --collection_episode_count equal to --num_envs and run separate processes per batch."
        )
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise ValueError("--train_fraction must be in (0, 1)")
    for path in (args.source_actor_checkpoint, args.teacher_csv):
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
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from sim_scripts.cube10cm_top_view_d256_action_replay_probe import load_episode_rows
    from sim_scripts.cube10cm_top_view_teacher_rollout_probe import apply_d256_pose_reset

    torch.manual_seed(int(args.seed))

    episode_indices = (
        [int(part) for part in str(args.episode_indices).split(",") if part.strip()]
        if str(args.episode_indices).strip()
        else None
    )
    selected_episodes, selected_episode_rows = load_episode_rows(
        args.teacher_csv,
        collection_episode_count,
        args.episode_min,
        args.episode_max,
        episode_indices,
    )

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.action_scale = float(args.action_scale)
    env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)

    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[d256-replay-distill] "
        f"env_id={env_id} training=PPO_NO supervised=YES num_envs={args.num_envs} "
        f"collect_steps={args.collect_steps} collection_episodes={collection_episode_count} "
        f"fresh_env_per_batch={bool(args.fresh_env_per_batch)}",
        flush=True,
    )

    def make_env_runner():
        env = gym.make(env_id, cfg=copy.deepcopy(env_cfg))
        env = RslRlVecEnvWrapper(env, clip_actions=1.0)
        inner = env.unwrapped
        if int(args.collect_steps) >= int(inner.max_episode_length) - 1:
            env.close()
            raise ValueError(
                f"--collect_steps {args.collect_steps} would hit env truncation/reset; "
                f"use <= {int(inner.max_episode_length) - 2}"
            )
        runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
        runner.load(str(args.source_actor_checkpoint), load_optimizer=False, map_location=inner.device)
        return env, inner, runner, runner.alg.policy

    shared_env = shared_inner = shared_runner = shared_policy = None
    if not bool(args.fresh_env_per_batch):
        shared_env, shared_inner, shared_runner, shared_policy = make_env_runner()
        shared_env.reset()

    obs_parts = []
    target_parts = []
    target_clip_rates = []
    target_abs_means = []
    target_abs_maxes = []
    oracle_contact_count = 0.0
    oracle_reaction_count = 0.0
    oracle_useful_count = 0.0
    oracle_overshoot_count = 0.0
    oracle_episode_count = 0
    max_disp_xy_parts = []
    max_disp_along_parts = []
    reset_infos = []
    batch_summaries = []

    with torch.inference_mode():
        for batch_start in range(0, collection_episode_count, int(args.num_envs)):
            episode_rows = selected_episode_rows[batch_start : batch_start + int(args.num_envs)]
            if len(episode_rows) != int(args.num_envs):
                raise RuntimeError("internal episode batch size mismatch")
            if bool(args.fresh_env_per_batch):
                env, inner, _runner, policy = make_env_runner()
            else:
                env = shared_env
                inner = shared_inner
                policy = shared_policy
                if env is None or inner is None or policy is None:
                    raise RuntimeError("shared env was not initialized")
            env.reset()
            device = inner.device
            reset_infos.append(apply_d256_pose_reset(inner, [rows[0] for rows in episode_rows]))
            obs = env.get_observations()
            max_disp_xy = torch.zeros(inner.num_envs, device=device)
            max_disp_along = torch.full((inner.num_envs,), -float("inf"), device=device)

            print(
                "[d256-replay-distill] collect_batch "
                f"{batch_start // int(args.num_envs) + 1}/"
                f"{collection_episode_count // int(args.num_envs)} "
                f"episode_range={selected_episodes[batch_start]}.."
                f"{selected_episodes[batch_start + int(args.num_envs) - 1]}",
                flush=True,
            )
            for step in range(int(args.collect_steps)):
                row_idx = min(
                    step // max(1, int(args.hold_steps)),
                    min(len(rows) for rows in episode_rows) - 1,
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
                target_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=device)
                raw_arm_actions = needed_delta / max(float(inner.cfg.action_scale), 1.0e-6)
                target_actions[:, inner._bc_arm_joint_ids] = torch.clamp(raw_arm_actions, -1.0, 1.0)
                target_actions[:, inner.gripper_joint_idx] = 0.0

                actor_obs = policy.get_actor_obs(obs).detach().clone()
                obs_parts.append(actor_obs.cpu())
                target_parts.append(target_actions.detach().clone().cpu())
                target_clip_rates.append(_tensor_mean((torch.abs(raw_arm_actions) >= 1.0 - 1.0e-9).float()))
                target_abs_means.append(_tensor_mean(torch.mean(torch.abs(target_actions), dim=-1)))
                target_abs_maxes.append(_tensor_max(torch.abs(target_actions)))

                obs, _, _, _ = env.step(target_actions)
                inner._compute_intermediate_values()
                terms = inner._tap_terms()
                max_disp_xy = torch.maximum(max_disp_xy, terms["disp_xy"].detach())
                max_disp_along = torch.maximum(max_disp_along, terms["disp_along"].detach())

            useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
            batch_contact = _tensor_mean(inner._tap_contact_seen.float())
            batch_reaction = _tensor_mean(inner._tap_reaction_seen.float())
            batch_useful = _tensor_mean(useful_seen.float())
            batch_overshoot = _tensor_mean(inner._tap_overshoot_seen.float())
            batch_max_xy_mean = _tensor_mean(max_disp_xy)
            batch_max_xy_max = _tensor_max(max_disp_xy)
            oracle_contact_count += float(inner._tap_contact_seen.float().sum().detach().cpu().item())
            oracle_reaction_count += float(inner._tap_reaction_seen.float().sum().detach().cpu().item())
            oracle_useful_count += float(useful_seen.float().sum().detach().cpu().item())
            oracle_overshoot_count += float(inner._tap_overshoot_seen.float().sum().detach().cpu().item())
            oracle_episode_count += int(inner.num_envs)
            max_disp_xy_parts.append(max_disp_xy.detach().clone().cpu())
            max_disp_along_parts.append(max_disp_along.detach().clone().cpu())
            batch_summaries.append(
                {
                    "batch_index": int(batch_start // int(args.num_envs)),
                    "episode_min": int(selected_episodes[batch_start]),
                    "episode_max": int(selected_episodes[batch_start + int(args.num_envs) - 1]),
                    "contact_seen_rate": batch_contact,
                    "reaction_seen_rate": batch_reaction,
                    "useful_seen_rate": batch_useful,
                    "overshoot_seen_rate": batch_overshoot,
                    "max_disp_xy_mean_m": batch_max_xy_mean,
                    "max_disp_xy_max_m": batch_max_xy_max,
                    "max_disp_along_mean_m": _tensor_mean(max_disp_along),
                    "max_disp_along_max_m": _tensor_max(max_disp_along),
                }
            )
            print(
                "[d256-replay-distill] collect_batch_summary "
                f"{batch_start // int(args.num_envs) + 1}/"
                f"{collection_episode_count // int(args.num_envs)} "
                f"useful={batch_useful:.6f} overshoot={batch_overshoot:.6f} "
                f"max_xy={batch_max_xy_max:.6f}",
                flush=True,
            )
            if bool(args.fresh_env_per_batch):
                env.close()

    if bool(args.fresh_env_per_batch):
        env, inner, runner, policy = make_env_runner()
    else:
        env = shared_env
        inner = shared_inner
        runner = shared_runner
        policy = shared_policy
        if env is None or inner is None or runner is None or policy is None:
            raise RuntimeError("shared env was not initialized for training")
    device = inner.device
    obs_all = torch.cat(obs_parts, dim=0).to(device)
    target_all = torch.cat(target_parts, dim=0).to(device)
    sample_count = int(obs_all.shape[0])
    action_labels = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    if len(action_labels) != int(target_all.shape[-1]):
        action_labels = [f"action_{idx}" for idx in range(int(target_all.shape[-1]))]

    oracle_contact = oracle_contact_count / max(float(oracle_episode_count), 1.0)
    oracle_reaction = oracle_reaction_count / max(float(oracle_episode_count), 1.0)
    oracle_useful = oracle_useful_count / max(float(oracle_episode_count), 1.0)
    oracle_overshoot = oracle_overshoot_count / max(float(oracle_episode_count), 1.0)
    max_disp_xy_all = torch.cat(max_disp_xy_parts, dim=0)
    max_disp_along_all = torch.cat(max_disp_along_parts, dim=0)

    dataset_summary = {
        "artifact_tag": str(args.artifact_tag),
        "source_actor_checkpoint": _rel(args.source_actor_checkpoint),
        "teacher_csv": _rel(args.teacher_csv),
        "num_envs": int(args.num_envs),
        "collection_episode_count": int(collection_episode_count),
        "episode_min_filter": int(args.episode_min) if args.episode_min is not None else None,
        "episode_max_filter": int(args.episode_max) if args.episode_max is not None else None,
        "episode_indices_filter": episode_indices,
        "selected_episodes": [int(ep) for ep in selected_episodes],
        "selected_episode_min": int(min(selected_episodes)),
        "selected_episode_max": int(max(selected_episodes)),
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "collect_steps": int(args.collect_steps),
        "hold_steps": int(args.hold_steps),
        "episode_length_s": float(env_cfg.episode_length_s),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "action_scale": float(inner.cfg.action_scale),
        "action_smoothing_alpha": float(inner.cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(inner.cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(inner.cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(inner.cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(inner.cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(inner.cfg.joint_delta_reference),
        "oracle_contact_seen_rate": oracle_contact,
        "oracle_reaction_seen_rate": oracle_reaction,
        "oracle_useful_seen_rate": oracle_useful,
        "oracle_overshoot_seen_rate": oracle_overshoot,
        "oracle_max_disp_xy_mean_m": _tensor_mean(max_disp_xy_all),
        "oracle_max_disp_xy_max_m": _tensor_max(max_disp_xy_all),
        "oracle_max_disp_along_mean_m": _tensor_mean(max_disp_along_all),
        "oracle_max_disp_along_max_m": _tensor_max(max_disp_along_all),
        "target_action_abs_mean": sum(target_abs_means) / len(target_abs_means),
        "target_action_abs_max": max(target_abs_maxes),
        "target_action_clip_rate_mean": sum(target_clip_rates) / len(target_clip_rates),
        "target_action_clip_rate_max": max(target_clip_rates),
        "collection_batch_summaries": batch_summaries,
    }
    if args.dataset_out is not None:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_obs": obs_all.cpu(),
                "target_actions": target_all.cpu(),
                "summary": dataset_summary,
            },
            args.dataset_out,
        )
        dataset_json = args.dataset_out.with_suffix(".json")
        dataset_json.write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n")
        print(
            "[d256-replay-distill] DATASET "
            f"path={args.dataset_out} samples={int(obs_all.shape[0])} "
            f"useful={oracle_useful:.6f} overshoot={oracle_overshoot:.6f}",
            flush=True,
        )
        if bool(args.dataset_only):
            env.close()
            sim_app.close()
            return 0

    perm = torch.randperm(sample_count, device=device)
    train_n = int(sample_count * float(args.train_fraction))
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]
    train_obs = obs_all[train_idx]
    train_target = target_all[train_idx]
    val_obs = obs_all[val_idx]
    val_target = target_all[val_idx]

    def actor_forward(raw_obs):
        return policy.actor(policy.actor_obs_normalizer(raw_obs))

    policy.eval()
    with torch.inference_mode():
        initial_train_metrics = _metrics(torch, actor_forward(train_obs), train_target)
        initial_val_pred = actor_forward(val_obs)
        initial_val_metrics = _metrics(torch, initial_val_pred, val_target)
        initial_per_dim = _per_dim_metrics(torch, initial_val_pred, val_target, action_labels)

    optimizer = torch.optim.AdamW(
        policy.actor.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_rows: list[dict[str, float | int]] = []
    policy.actor.train()
    policy.actor_obs_normalizer.eval()
    for epoch in range(int(args.epochs)):
        epoch_perm = train_idx[torch.randperm(train_n, device=device)]
        losses = []
        cosine_losses = []
        for start in range(0, train_n, int(args.batch_size)):
            batch_idx = epoch_perm[start : start + int(args.batch_size)]
            pred = actor_forward(obs_all[batch_idx])
            target = target_all[batch_idx]
            mse = F.mse_loss(pred, target)
            cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)).mean()
            loss = mse + 0.02 * cosine_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(mse.detach().cpu().item()))
            cosine_losses.append(float(cosine_loss.detach().cpu().item()))

        policy.actor.eval()
        with torch.inference_mode():
            val_pred = actor_forward(val_obs)
            val_metrics = _metrics(torch, val_pred, val_target)
        policy.actor.train()
        row = {
            "epoch": epoch,
            "train_mse_batch_mean": sum(losses) / len(losses),
            "train_cosine_loss_batch_mean": sum(cosine_losses) / len(cosine_losses),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_cosine": val_metrics["cosine"],
        }
        loss_rows.append(row)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == int(args.epochs):
            print(
                "[d256-replay-distill] "
                f"epoch={epoch + 1}/{args.epochs} val_mse={row['val_mse']:.6f} "
                f"val_cosine={row['val_cosine']:.6f}",
                flush=True,
            )

    policy.actor.eval()
    with torch.inference_mode():
        final_train_pred = actor_forward(train_obs)
        final_val_pred = actor_forward(val_obs)
        final_train_metrics = _metrics(torch, final_train_pred, train_target)
        final_val_metrics = _metrics(torch, final_val_pred, val_target)
        final_per_dim = _per_dim_metrics(torch, final_val_pred, val_target, action_labels)

    issues: list[str] = []
    if oracle_useful < 0.99:
        issues.append(f"oracle replay useful rate below 0.99: {oracle_useful}")
    if oracle_overshoot > 0.05:
        issues.append(f"oracle replay overshoot high: {oracle_overshoot}")
    if final_val_metrics["mse"] > float(args.max_val_mse):
        issues.append(f"final val MSE above threshold: {final_val_metrics['mse']}")
    if final_val_metrics["cosine"] < float(args.min_val_cosine):
        issues.append(f"final val cosine below threshold: {final_val_metrics['cosine']}")

    verdict = (
        "D289_D256_REPLAY_ACTOR_DISTILL_SUPERVISED_FIT_PASS_NEEDS_ROLLOUT_EVAL"
        if not issues
        else "D289_D256_REPLAY_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = out_dir / "model_actor_d256_replay_d289.pt"
    out_json = out_dir / "actor_d256_replay_distill_summary_d289.json"
    out_md = out_dir / "actor_d256_replay_distill_summary_d289.md"
    out_csv = out_dir / "actor_d256_replay_distill_losses_d289.csv"

    runner.current_learning_iteration = 0
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": runner.current_learning_iteration,
            "infos": {"artifact_tag": str(args.artifact_tag), "verdict": verdict},
        },
        out_checkpoint,
    )

    with out_csv.open("w", newline="") as f:
        fieldnames = list(loss_rows[0].keys()) if loss_rows else ["epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_rows)

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "source_actor_checkpoint": _rel(args.source_actor_checkpoint),
        "distilled_actor_checkpoint": _rel(out_checkpoint),
        "teacher_csv": _rel(args.teacher_csv),
        "env_id": env_id,
        "num_envs": int(args.num_envs),
        "collection_episode_count": int(collection_episode_count),
        "fresh_env_per_batch": bool(args.fresh_env_per_batch),
        "episode_min_filter": int(args.episode_min) if args.episode_min is not None else None,
        "episode_max_filter": int(args.episode_max) if args.episode_max is not None else None,
        "collect_steps": int(args.collect_steps),
        "hold_steps": int(args.hold_steps),
        "episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length": int(inner.max_episode_length),
        "seed": int(args.seed),
        "selected_episode_min": int(min(selected_episodes)),
        "selected_episode_max": int(max(selected_episodes)),
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "episode_indices_filter": episode_indices,
        "dataset_out": _rel(args.dataset_out) if args.dataset_out is not None else "",
        "collection_batch_summaries": batch_summaries,
        "sample_count": sample_count,
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "action_scale": float(inner.cfg.action_scale),
        "action_smoothing_alpha": float(inner.cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(inner.cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(inner.cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(inner.cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(inner.cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(inner.cfg.joint_delta_reference),
        "reset_pose_info_first": reset_infos[0] if reset_infos else {},
        "reset_pose_info_last": reset_infos[-1] if reset_infos else {},
        "oracle_contact_seen_rate": oracle_contact,
        "oracle_reaction_seen_rate": oracle_reaction,
        "oracle_useful_seen_rate": oracle_useful,
        "oracle_overshoot_seen_rate": oracle_overshoot,
        "oracle_max_disp_xy_mean_m": _tensor_mean(max_disp_xy_all),
        "oracle_max_disp_xy_max_m": _tensor_max(max_disp_xy_all),
        "oracle_max_disp_along_mean_m": _tensor_mean(max_disp_along_all),
        "oracle_max_disp_along_max_m": _tensor_max(max_disp_along_all),
        "target_action_abs_mean": sum(target_abs_means) / len(target_abs_means),
        "target_action_abs_max": max(target_abs_maxes),
        "target_action_clip_rate_mean": sum(target_clip_rates) / len(target_clip_rates),
        "target_action_clip_rate_max": max(target_clip_rates),
        "initial_train_metrics": initial_train_metrics,
        "initial_val_metrics": initial_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "initial_val_per_dim": initial_per_dim,
        "final_val_per_dim": final_per_dim,
        "out_checkpoint": _rel(out_checkpoint),
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)

    print(
        "[d256-replay-distill] SUMMARY "
        f"verdict={verdict} val_mse={final_val_metrics['mse']:.6f} "
        f"val_cosine={final_val_metrics['cosine']:.6f} checkpoint={out_checkpoint}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
