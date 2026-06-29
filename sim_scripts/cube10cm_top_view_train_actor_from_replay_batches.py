#!/usr/bin/env python3
"""Train an actor from separately collected D256 replay-action batches.

This is supervised learning only. It does not run PPO and does not collect
rollouts. The input batches should be produced by
cube10cm_top_view_distill_actor_from_d256_replay.py --dataset_only.
"""
from __future__ import annotations

import argparse
import csv
import glob
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
DEFAULT_SOURCE_ACTOR_CHECKPOINT = (
    RUNTIME_ROOT
    / "actor_preserve_d285"
    / "tap10cm"
    / "ppo_actorfreeze_noise002_10_smoke"
    / "cube10cm_d285_actorfreeze_noise002_10_smoke"
    / "model_9.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "actor_d256_replay_batches_d290" / "tap10cm"
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
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
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
    return rows


def _expand_dataset_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D290 Actor Training From D256 Replay Batches",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source actor: `{summary['source_actor_checkpoint']}`",
        f"- output checkpoint: `{summary['out_checkpoint']}`",
        f"- dataset count: `{summary['dataset_count']}`",
        f"- samples train/val: `{summary['train_samples']}` / `{summary['val_samples']}`",
        f"- selected episode range/count: `{summary['selected_episode_min']}..{summary['selected_episode_max']}` / `{summary['selected_episode_unique_count']}`",
        f"- aggregate oracle contact/useful/reaction: `{summary['oracle_contact_seen_rate']}` / `{summary['oracle_useful_seen_rate']}` / `{summary['oracle_reaction_seen_rate']}`",
        f"- aggregate oracle overshoot: `{summary['oracle_overshoot_seen_rate']}`",
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
            "This trains only the actor from clean separately collected replay-action batches. It is not PPO.",
            "Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_glob", action="append", required=True)
    parser.add_argument("--source_actor_checkpoint", type=Path, default=DEFAULT_SOURCE_ACTOR_CHECKPOINT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29001)
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=7.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--max_val_mse", type=float, default=0.04)
    parser.add_argument("--min_val_cosine", type=float, default=0.80)
    parser.add_argument("--artifact_tag", type=str, default="d290_d256_replay_batch_actor_train")
    args = parser.parse_args()

    if not args.source_actor_checkpoint.exists():
        raise FileNotFoundError(args.source_actor_checkpoint)
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise ValueError("--train_fraction must be in (0, 1)")

    dataset_paths = _expand_dataset_paths(args.dataset_glob)
    missing = [path for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

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

    obs_parts = []
    target_parts = []
    summaries = []
    selected_episodes: list[int] = []
    for path in dataset_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        obs_parts.append(data["actor_obs"].float())
        target_parts.append(data["target_actions"].float())
        summary = data.get("summary", {})
        summaries.append(summary)
        selected_episodes.extend(int(ep) for ep in summary.get("selected_episodes", []))

    obs_all_cpu = torch.cat(obs_parts, dim=0)
    target_all_cpu = torch.cat(target_parts, dim=0)
    if obs_all_cpu.shape[0] != target_all_cpu.shape[0]:
        raise ValueError(f"obs/target sample mismatch: {obs_all_cpu.shape} vs {target_all_cpu.shape}")

    env_cfg = RoArmCubeTap10cmEnvCfg()
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
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)
    env_id = "RoArm-CubeTap10cm-Direct-v0"
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.source_actor_checkpoint), load_optimizer=False, map_location=inner.device)
    policy = runner.alg.policy
    device = inner.device

    obs_all = obs_all_cpu.to(device)
    target_all = target_all_cpu.to(device)
    sample_count = int(obs_all.shape[0])
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

    action_labels = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    if len(action_labels) != int(target_all.shape[-1]):
        action_labels = [f"action_{idx}" for idx in range(int(target_all.shape[-1]))]

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
                "[d256-replay-batch-train] "
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

    total_episodes = sum(float(s.get("collection_episode_count", 0)) for s in summaries)

    def weighted_rate(key: str) -> float:
        if total_episodes <= 0:
            return 0.0
        return float(
            sum(float(s.get(key, 0.0)) * float(s.get("collection_episode_count", 0)) for s in summaries)
            / total_episodes
        )

    issues: list[str] = []
    oracle_useful = weighted_rate("oracle_useful_seen_rate")
    oracle_overshoot = weighted_rate("oracle_overshoot_seen_rate")
    if oracle_useful < 0.99:
        issues.append(f"aggregate oracle useful rate below 0.99: {oracle_useful}")
    if oracle_overshoot > 0.05:
        issues.append(f"aggregate oracle overshoot high: {oracle_overshoot}")
    if final_val_metrics["mse"] > float(args.max_val_mse):
        issues.append(f"final val MSE above threshold: {final_val_metrics['mse']}")
    if final_val_metrics["cosine"] < float(args.min_val_cosine):
        issues.append(f"final val cosine below threshold: {final_val_metrics['cosine']}")

    verdict = (
        "D290_D256_REPLAY_BATCH_ACTOR_TRAIN_PASS_NEEDS_ROLLOUT_EVAL"
        if not issues
        else "D290_D256_REPLAY_BATCH_ACTOR_TRAIN_WARN_NEEDS_ROLLOUT_EVAL"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = out_dir / "model_actor_d256_replay_batches_d290.pt"
    out_json = out_dir / "actor_d256_replay_batches_train_summary_d290.json"
    out_md = out_dir / "actor_d256_replay_batches_train_summary_d290.md"
    out_csv = out_dir / "actor_d256_replay_batches_train_losses_d290.csv"

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
        writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loss_rows)

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "source_actor_checkpoint": _rel(args.source_actor_checkpoint),
        "out_checkpoint": _rel(out_checkpoint),
        "dataset_paths": [_rel(path) for path in dataset_paths],
        "dataset_count": len(dataset_paths),
        "sample_count": sample_count,
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "selected_episode_min": int(min(selected_episodes)) if selected_episodes else -1,
        "selected_episode_max": int(max(selected_episodes)) if selected_episodes else -1,
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "oracle_contact_seen_rate": weighted_rate("oracle_contact_seen_rate"),
        "oracle_reaction_seen_rate": weighted_rate("oracle_reaction_seen_rate"),
        "oracle_useful_seen_rate": oracle_useful,
        "oracle_overshoot_seen_rate": oracle_overshoot,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "initial_train_metrics": initial_train_metrics,
        "initial_val_metrics": initial_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "initial_val_per_dim": initial_per_dim,
        "final_val_per_dim": final_per_dim,
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)
    print(
        "[d256-replay-batch-train] SUMMARY "
        f"verdict={verdict} val_mse={final_val_metrics['mse']:.6f} "
        f"val_cosine={final_val_metrics['cosine']:.6f} checkpoint={out_checkpoint}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
