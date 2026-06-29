#!/usr/bin/env python3
"""Build a phase-aware non-PPO actor repair dataset.

D305 showed that pure recovery targets can restore contact but create high
clip/cap pressure, while pure recorded D256 actions preserve the demonstrated
push but may not correct closed-loop drift.  This script rewrites only the
supervised target actions:

- early steps: favor recovery actions for approach/contact correction;
- later steps: favor recorded D256 actions for push/displacement preservation;
- all steps: cap and optionally smooth targets to reduce cap pressure.

It does not run Isaac Lab, PPO, rendering, or robot control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_SOURCE_DATASET = (
    RUNTIME_ROOT
    / "actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80"
    / "closed_loop_recovery_after_repair/closed_loop_recovery_dataset_d305_after_repair.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "actor_recovery_repair_d306" / "tap10cm" / "phase_target_dataset"


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    return {
        "abs_mean": float(xf.abs().mean().item()),
        "abs_max": float(xf.abs().max().item()),
        "clip_rate_ge_099": float((xf.abs() >= 0.99).float().mean().item()),
        "clip_rate_ge_090": float((xf.abs() >= 0.90).float().mean().item()),
        "clip_rate_ge_075": float((xf.abs() >= 0.75).float().mean().item()),
    }


def _comparison_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_f = pred.detach().float()
    target_f = target.detach().float()
    diff = pred_f - target_f
    cosine = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1, eps=1.0e-6)
    return {
        "mse": float(torch.mean(diff * diff).item()),
        "mae": float(torch.mean(torch.abs(diff)).item()),
        "cosine": float(torch.mean(cosine).item()),
    }


def _phase_rows(
    step_indices: torch.Tensor,
    target_actions: torch.Tensor,
    recorded_actions: torch.Tensor,
    recovery_actions: torch.Tensor,
    actor_actions: torch.Tensor | None,
    transition_start_step: int,
    transition_end_step: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = [
        ("approach", 0, transition_start_step - 1),
        ("transition", transition_start_step, transition_end_step),
        ("push", transition_end_step + 1, int(step_indices.max().item())),
    ]
    for name, lo, hi in phases:
        if hi < lo:
            continue
        mask = (step_indices >= lo) & (step_indices <= hi)
        if not bool(mask.any().item()):
            continue
        row: dict[str, Any] = {
            "phase": name,
            "step_min": int(lo),
            "step_max": int(hi),
            "sample_count": int(mask.sum().item()),
            "target": _tensor_stats(target_actions[mask]),
            "recorded": _tensor_stats(recorded_actions[mask]),
            "recovery": _tensor_stats(recovery_actions[mask]),
            "target_vs_recorded": _comparison_metrics(target_actions[mask], recorded_actions[mask]),
            "target_vs_recovery": _comparison_metrics(target_actions[mask], recovery_actions[mask]),
        }
        if actor_actions is not None:
            row["actor"] = _tensor_stats(actor_actions[mask])
            row["target_vs_actor"] = _comparison_metrics(target_actions[mask], actor_actions[mask])
        rows.append(row)
    return rows


def _smooth_by_env_step(
    target: torch.Tensor,
    env_indices: torch.Tensor,
    step_indices: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha >= 0.999999:
        return target
    if not (0.0 < alpha <= 1.0):
        raise ValueError("--smooth_alpha must be in (0, 1]")

    smoothed = target.clone()
    envs = torch.unique(env_indices).tolist()
    for env in envs:
        mask = env_indices == int(env)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        order = torch.argsort(step_indices[idx])
        ordered_idx = idx[order]
        prev = target[ordered_idx[0]].clone()
        smoothed[ordered_idx[0]] = prev
        for out_idx in ordered_idx[1:]:
            cur = target[out_idx]
            prev = float(alpha) * cur + (1.0 - float(alpha)) * prev
            smoothed[out_idx] = prev
    return smoothed


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D306 Phase-Aware Action Repair Dataset",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source dataset: `{summary['source_dataset']}`",
        f"- output dataset: `{summary['out_dataset']}`",
        f"- samples: `{summary['sample_count']}`",
        f"- recovery weight start/end: `{summary['recovery_weight_start']}` / `{summary['recovery_weight_end']}`",
        f"- transition steps: `{summary['transition_start_step']}` / `{summary['transition_end_step']}`",
        f"- target clip abs: `{summary['target_clip_abs']}`",
        f"- smooth alpha: `{summary['smooth_alpha']}`",
        f"- target abs mean/max: `{summary['target_stats']['abs_mean']}` / `{summary['target_stats']['abs_max']}`",
        f"- target clip >=0.99/0.90/0.75: `{summary['target_stats']['clip_rate_ge_099']}` / `{summary['target_stats']['clip_rate_ge_090']}` / `{summary['target_stats']['clip_rate_ge_075']}`",
        f"- target-vs-recorded MSE/cosine: `{summary['target_vs_recorded']['mse']}` / `{summary['target_vs_recorded']['cosine']}`",
        f"- target-vs-recovery MSE/cosine: `{summary['target_vs_recovery']['mse']}` / `{summary['target_vs_recovery']['cosine']}`",
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
            "This is a supervised target-rewrite dataset only. It does not prove policy success.",
            "The next required checks are offline actor-vs-target diagnostics and fresh one-bin/direct-reset rollouts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_dataset", type=Path, default=None)
    parser.add_argument("--transition_start_step", type=int, default=40)
    parser.add_argument("--transition_end_step", type=int, default=260)
    parser.add_argument("--recovery_weight_start", type=float, default=0.65)
    parser.add_argument("--recovery_weight_end", type=float, default=0.10)
    parser.add_argument("--target_clip_abs", type=float, default=0.85)
    parser.add_argument("--smooth_alpha", type=float, default=0.45)
    parser.add_argument("--artifact_tag", type=str, default="d306_phase_action_repair_dataset")
    args = parser.parse_args()

    if not args.source_dataset.exists():
        raise FileNotFoundError(args.source_dataset)
    if int(args.transition_end_step) <= int(args.transition_start_step):
        raise ValueError("--transition_end_step must be greater than --transition_start_step")
    if not (0.0 <= float(args.recovery_weight_end) <= float(args.recovery_weight_start) <= 1.0):
        raise ValueError("expected 0 <= recovery_weight_end <= recovery_weight_start <= 1")
    if not (0.0 < float(args.target_clip_abs) <= 1.0):
        raise ValueError("--target_clip_abs must be in (0, 1]")

    data = torch.load(args.source_dataset, map_location="cpu", weights_only=False)
    required = ["actor_obs", "recorded_actions", "recovery_actions", "step_indices", "env_indices"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"missing required dataset keys: {missing}")

    actor_obs = data["actor_obs"].float()
    recorded = data["recorded_actions"].float()
    recovery = data["recovery_actions"].float()
    actor_actions = data.get("actor_actions")
    if actor_actions is not None:
        actor_actions = actor_actions.float()
    step_indices = data["step_indices"].long()
    env_indices = data["env_indices"].long()
    sample_count = int(actor_obs.shape[0])
    if recorded.shape != recovery.shape:
        raise ValueError(f"recorded/recovery action shape mismatch: {recorded.shape} vs {recovery.shape}")
    if actor_obs.shape[0] != recorded.shape[0]:
        raise ValueError(f"obs/action sample mismatch: {actor_obs.shape[0]} vs {recorded.shape[0]}")

    step_f = step_indices.float()
    span = max(float(args.transition_end_step - args.transition_start_step), 1.0)
    progress = torch.clamp((step_f - float(args.transition_start_step)) / span, min=0.0, max=1.0)
    recovery_weight = float(args.recovery_weight_start) + (
        float(args.recovery_weight_end) - float(args.recovery_weight_start)
    ) * progress
    recovery_weight = recovery_weight.unsqueeze(-1)
    target = recovery_weight * recovery + (1.0 - recovery_weight) * recorded
    target = torch.clamp(target, -float(args.target_clip_abs), float(args.target_clip_abs))
    target = _smooth_by_env_step(target, env_indices, step_indices, float(args.smooth_alpha))
    target = torch.clamp(target, -float(args.target_clip_abs), float(args.target_clip_abs))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dataset = args.out_dataset or out_dir / f"phase_action_repair_dataset_{args.artifact_tag}.pt"
    out_json = out_dir / f"phase_action_repair_dataset_summary_{args.artifact_tag}.json"
    out_md = out_dir / f"phase_action_repair_dataset_summary_{args.artifact_tag}.md"

    target_stats = _tensor_stats(target)
    recorded_stats = _tensor_stats(recorded)
    recovery_stats = _tensor_stats(recovery)
    issues: list[str] = []
    if target_stats["clip_rate_ge_099"] > 0.01:
        issues.append(f"target clip >=0.99 still high: {target_stats['clip_rate_ge_099']}")
    if target_stats["abs_mean"] >= recovery_stats["abs_mean"]:
        issues.append("target abs mean did not reduce relative to recovery actions")
    if target_stats["abs_mean"] <= recorded_stats["abs_mean"] * 0.50:
        issues.append("target abs mean collapsed far below recorded actions")

    source_summary = data.get("summary", {})
    summary: dict[str, Any] = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": "D306_PHASE_ACTION_REPAIR_DATASET_READY" if not issues else "D306_PHASE_ACTION_REPAIR_DATASET_WARN_REVIEW",
        "issues": issues,
        "source_dataset": _rel(args.source_dataset),
        "out_dataset": _rel(out_dataset),
        "sample_count": sample_count,
        "transition_start_step": int(args.transition_start_step),
        "transition_end_step": int(args.transition_end_step),
        "recovery_weight_start": float(args.recovery_weight_start),
        "recovery_weight_end": float(args.recovery_weight_end),
        "target_clip_abs": float(args.target_clip_abs),
        "smooth_alpha": float(args.smooth_alpha),
        "target_stats": target_stats,
        "recorded_stats": recorded_stats,
        "recovery_stats": recovery_stats,
        "target_vs_recorded": _comparison_metrics(target, recorded),
        "target_vs_recovery": _comparison_metrics(target, recovery),
        "phase_rows": _phase_rows(
            step_indices,
            target,
            recorded,
            recovery,
            actor_actions,
            int(args.transition_start_step),
            int(args.transition_end_step),
        ),
        "source_summary": source_summary,
        "collection_episode_count": int(source_summary.get("collection_episode_count", 0)),
        "oracle_contact_seen_rate": float(source_summary.get("oracle_contact_seen_rate", source_summary.get("actor_contact_seen_rate", 0.0))),
        "oracle_reaction_seen_rate": float(source_summary.get("oracle_reaction_seen_rate", source_summary.get("actor_reaction_seen_rate", 0.0))),
        "oracle_useful_seen_rate": float(source_summary.get("oracle_useful_seen_rate", source_summary.get("actor_useful_seen_rate", 0.0))),
        "oracle_overshoot_seen_rate": float(source_summary.get("oracle_overshoot_seen_rate", source_summary.get("actor_overshoot_seen_rate", 0.0))),
        "selected_episodes": source_summary.get("selected_episodes", []),
        "selected_episode_min": source_summary.get("selected_episode_min", -1),
        "selected_episode_max": source_summary.get("selected_episode_max", -1),
        "selected_episode_unique_count": source_summary.get("selected_episode_unique_count", 0),
    }

    out_data = dict(data)
    out_data["target_actions"] = target
    out_data["phase_recovery_weight"] = recovery_weight.squeeze(-1)
    out_data["summary"] = summary
    torch.save(out_data, out_dataset)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)

    print(
        "[d306-phase-action-repair] "
        f"verdict={summary['verdict']} samples={sample_count} "
        f"target_abs_mean={target_stats['abs_mean']:.6f} "
        f"target_clip_ge099={target_stats['clip_rate_ge_099']:.6f} "
        f"out={out_dataset}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
