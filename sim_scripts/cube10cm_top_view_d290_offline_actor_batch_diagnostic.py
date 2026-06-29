#!/usr/bin/env python3
"""Offline D290 actor-vs-replay batch diagnostic.

This instantiates the same RSL actor/inference path used by rollout probes, but
does not step Isaac physics. It checks whether a checkpoint still matches the
saved D256 replay-action batches before blaming closed-loop dynamics.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
DEFAULT_ACTOR_CHECKPOINT = (
    RUNTIME_ROOT
    / "actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt"
)
DEFAULT_OUT_DIR = RUNTIME_ROOT / "actor_d256_replay_batches_d290/tap10cm_ep155/offline_batch_diagnostic"


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _expand_dataset_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        paths.extend(Path(p) for p in expanded) if expanded else paths.append(Path(pattern))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _tensor_mean(x: Any) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_max(x: Any) -> float:
    return float(x.detach().float().max().cpu().item())


def _metrics(torch: Any, pred: Any, target: Any) -> dict[str, float]:
    diff = pred - target
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)
    pred_abs = torch.abs(pred)
    target_abs = torch.abs(target)
    return {
        "mse": _tensor_mean(torch.mean(diff * diff, dim=-1)),
        "mae": _tensor_mean(torch.mean(torch.abs(diff), dim=-1)),
        "cosine": _tensor_mean(cosine),
        "pred_abs_mean": _tensor_mean(torch.mean(pred_abs, dim=-1)),
        "pred_abs_max": _tensor_max(pred_abs),
        "pred_clip_rate": _tensor_mean((pred_abs >= 1.0 - 1.0e-6).float()),
        "target_abs_mean": _tensor_mean(torch.mean(target_abs, dim=-1)),
        "target_abs_max": _tensor_max(target_abs),
        "target_clip_rate": _tensor_mean((target_abs >= 1.0 - 1.0e-6).float()),
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D290 Offline Actor Batch Diagnostic",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- actor checkpoint: `{summary['actor_checkpoint']}`",
        f"- dataset count: `{summary['dataset_count']}`",
        f"- aggregate samples: `{summary['aggregate_samples']}`",
        f"- aggregate MSE/MAE/cosine: `{summary['aggregate_metrics']['mse']}` / `{summary['aggregate_metrics']['mae']}` / `{summary['aggregate_metrics']['cosine']}`",
        "",
        "## Batch Rows",
        "",
    ]
    for row in summary["batch_rows"]:
        lines.append(
            "- "
            f"{row['dataset']}: samples `{row['samples']}`, "
            f"mse `{row['mse']}`, cosine `{row['cosine']}`, "
            f"pred_abs_max `{row['pred_abs_max']}`, target_clip `{row['target_clip_rate']}`"
        )
    lines.extend(["", "## Issues", ""])
    lines.extend(f"- {issue}" for issue in summary["issues"]) if summary["issues"] else lines.append("- none")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--dataset_glob", action="append", required=True)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29041)
    parser.add_argument("--max_mse", type=float, default=0.04)
    parser.add_argument("--min_cosine", type=float, default=0.80)
    parser.add_argument("--artifact_tag", type=str, default="d290_offline_actor_batch_diagnostic")
    args = parser.parse_args()

    if not args.actor_checkpoint.exists():
        raise FileNotFoundError(args.actor_checkpoint)
    dataset_paths = _expand_dataset_paths(args.dataset_glob)
    missing = [path for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)
    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.actor_checkpoint), load_optimizer=False, map_location=inner.device)
    inference_policy = runner.get_inference_policy(device=inner.device)

    batch_rows: list[dict[str, Any]] = []
    all_obs: list[Any] = []
    all_targets: list[Any] = []
    with torch.inference_mode():
        for path in dataset_paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            obs = data["actor_obs"].float().to(inner.device)
            target = data["target_actions"].float().to(inner.device)
            pred = inference_policy({"policy": obs})
            metrics = _metrics(torch, pred, target)
            row = {"dataset": _rel(path), "samples": int(obs.shape[0]), **metrics}
            batch_rows.append(row)
            all_obs.append(obs)
            all_targets.append(target)
        obs_all = torch.cat(all_obs, dim=0)
        target_all = torch.cat(all_targets, dim=0)
        aggregate_metrics = _metrics(torch, inference_policy({"policy": obs_all}), target_all)

    issues: list[str] = []
    if aggregate_metrics["mse"] > float(args.max_mse):
        issues.append(f"aggregate MSE above threshold: {aggregate_metrics['mse']}")
    if aggregate_metrics["cosine"] < float(args.min_cosine):
        issues.append(f"aggregate cosine below threshold: {aggregate_metrics['cosine']}")
    verdict = "D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS" if not issues else "D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_FAIL"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "offline_actor_batch_diagnostic_summary_d290.json"
    out_md = out_dir / "offline_actor_batch_diagnostic_summary_d290.md"
    out_csv = out_dir / "offline_actor_batch_diagnostic_rows_d290.csv"
    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "actor_checkpoint": _rel(args.actor_checkpoint),
        "dataset_count": len(dataset_paths),
        "aggregate_samples": int(obs_all.shape[0]),
        "aggregate_metrics": aggregate_metrics,
        "batch_rows": batch_rows,
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)
    with out_csv.open("w", newline="") as f:
        fieldnames = list(batch_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_rows)

    print(
        "[offline-actor-batch-diagnostic] SUMMARY "
        f"verdict={verdict} mse={aggregate_metrics['mse']:.6f} "
        f"cosine={aggregate_metrics['cosine']:.6f} json={out_json}",
        flush=True,
    )
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
