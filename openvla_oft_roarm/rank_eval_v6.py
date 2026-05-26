"""Rank OpenVLA-OFT v6 offline eval checkpoints and surface overfit signal.

Reads the JSON written by ``eval_offline_v6.py`` and prints:

  1. per-checkpoint table sorted by ``holdout.l2_step0_mean``
  2. train_sanity vs holdout gap per checkpoint (R-OFT-2 overfit signal)
  3. per-joint MAE trajectory across steps
  4. best ckpt + recommended ckpt for deploy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"[error] not found: {args.json_path}", file=sys.stderr)
        return 1

    payload = json.loads(args.json_path.read_text())
    pcs = payload.get("per_checkpoint", [])
    if not pcs:
        print("[error] empty per_checkpoint", file=sys.stderr)
        return 1

    print(f"# eval JSON: {args.json_path}")
    print(f"# n_checkpoints={payload.get('n_checkpoints')} ckpts_completed={len(pcs)}")
    print(f"# holdout_eps={payload.get('holdout_episodes')}")
    print(f"# train_sanity_eps={payload.get('train_sanity_episodes')}")
    std = payload.get("action_std_from_stats")
    if std:
        print(f"# action_std deg [{', '.join(f'{s:.2f}' for s in std)}]")
    print()

    print("## Table 1: per-ckpt holdout vs train_sanity (degrees)")
    print(
        f"{'step':>6}  {'h.l2_step0':>10}  {'h.l2_chunk':>10}  "
        f"{'t.l2_step0':>10}  {'t.l2_chunk':>10}  "
        f"{'gap_step0':>9}  {'gap_chunk':>9}"
    )
    rows = []
    for pc in pcs:
        step = pc["step"]
        h_step0 = pc["holdout"]["l2_step0_mean"]
        h_chunk = pc["holdout"]["l2_chunk_avg_mean"]
        t_step0 = pc["train_sanity"]["l2_step0_mean"]
        t_chunk = pc["train_sanity"]["l2_chunk_avg_mean"]
        gap_step0 = h_step0 - t_step0
        gap_chunk = h_chunk - t_chunk
        rows.append(
            dict(
                step=step,
                h_step0=h_step0,
                h_chunk=h_chunk,
                t_step0=t_step0,
                t_chunk=t_chunk,
                gap_step0=gap_step0,
                gap_chunk=gap_chunk,
                pc=pc,
            )
        )
        print(
            f"{step:>6}  {h_step0:>10.4f}  {h_chunk:>10.4f}  "
            f"{t_step0:>10.4f}  {t_chunk:>10.4f}  "
            f"{gap_step0:>9.4f}  {gap_chunk:>9.4f}"
        )
    print()

    print("## Table 2: ranked by holdout.l2_step0_mean (best = top)")
    rows_ranked = sorted(rows, key=lambda r: r["h_step0"])
    for i, r in enumerate(rows_ranked):
        marker = "  <- BEST" if i == 0 else ""
        print(
            f"  rank {i + 1:>2}  step={r['step']:>5}  "
            f"holdout.l2_step0={r['h_step0']:.4f}  gap={r['gap_step0']:.4f}{marker}"
        )
    print()

    print("## Table 3: per-joint MAE trajectory on holdout (step0)")
    print(f"{'step':>6}  " + "  ".join(f"{n:>9}" for n in JOINT_NAMES))
    for r in rows:
        mae = r["pc"]["holdout"]["mae_per_joint_step0"]
        print(f"{r['step']:>6}  " + "  ".join(f"{m:>9.3f}" for m in mae))
    print()

    print("## Table 4: per-joint MAE trajectory on train_sanity (step0)")
    print(f"{'step':>6}  " + "  ".join(f"{n:>9}" for n in JOINT_NAMES))
    for r in rows:
        mae = r["pc"]["train_sanity"]["mae_per_joint_step0"]
        print(f"{r['step']:>6}  " + "  ".join(f"{m:>9.3f}" for m in mae))
    print()

    print("## Table 5: per-joint memorization ratio (holdout/train_sanity step0)")
    print(f"{'step':>6}  " + "  ".join(f"{n:>9}" for n in JOINT_NAMES))
    for r in rows:
        h_mae = r["pc"]["holdout"]["mae_per_joint_step0"]
        t_mae = r["pc"]["train_sanity"]["mae_per_joint_step0"]
        ratios = [h / max(t, 1e-6) for h, t in zip(h_mae, t_mae)]
        print(f"{r['step']:>6}  " + "  ".join(f"{rt:>9.3f}" for rt in ratios))
    print()

    print("## Table 6: per-joint z-score on holdout (step0)")
    print(f"{'step':>6}  " + "  ".join(f"{n:>9}" for n in JOINT_NAMES))
    for r in rows:
        zs = r["pc"]["holdout"]["zscore_per_joint_step0"]
        print(f"{r['step']:>6}  " + "  ".join(f"{z:>9.4f}" for z in zs))
    print()

    best = rows_ranked[0]
    early_best = min(rows, key=lambda r: r["h_step0"] if r["step"] <= 10000 else 1e9)
    final_row = max(rows, key=lambda r: r["step"])

    print("## Summary")
    print(f"  best_overall      : step={best['step']:>5}  "
          f"holdout.l2_step0={best['h_step0']:.4f}  gap={best['gap_step0']:.4f}")
    print(f"  best_early(<=10K) : step={early_best['step']:>5}  "
          f"holdout.l2_step0={early_best['h_step0']:.4f}  gap={early_best['gap_step0']:.4f}")
    print(f"  final(highest)    : step={final_row['step']:>5}  "
          f"holdout.l2_step0={final_row['h_step0']:.4f}  gap={final_row['gap_step0']:.4f}")
    print()

    overfit_widens = final_row["gap_step0"] > rows[0]["gap_step0"] + 1.0
    holdout_increases = final_row["h_step0"] > rows[0]["h_step0"] + 1.0
    print("## Overfit signals (R-OFT-2)")
    print(f"  gap widens from step {rows[0]['step']} to {final_row['step']}: "
          f"{rows[0]['gap_step0']:.4f} -> {final_row['gap_step0']:.4f}  "
          f"({'YES' if overfit_widens else 'NO'})")
    print(f"  holdout worsens from step {rows[0]['step']} to {final_row['step']}: "
          f"{rows[0]['h_step0']:.4f} -> {final_row['h_step0']:.4f}  "
          f"({'YES' if holdout_increases else 'NO'})")
    if holdout_increases:
        print(f"  -> RECOMMEND deploy with best={best['step']} (early stop)")
    else:
        print(f"  -> RECOMMEND deploy with final={final_row['step']} (no clear overfit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
