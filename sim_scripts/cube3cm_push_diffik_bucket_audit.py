"""Bucket audit for cube3cm DiffIK/BC per-env rollout CSVs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def b(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def direction_key(row: dict[str, str]) -> tuple[int, int]:
    return (int(round(f(row, "push_dx"))), int(round(f(row, "push_dy"))))


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "controlled": 0.0,
            "impact": 0.0,
            "low_motion": 0.0,
            "success": 0.0,
            "disp_along_mean": 0.0,
            "tcp_err_mean": 0.0,
        }
    return {
        "n": n,
        "controlled": sum(b(r, "controlled_push") for r in rows) / n,
        "impact": sum(b(r, "impact_outlier") for r in rows) / n,
        "low_motion": sum(b(r, "low_motion") for r in rows) / n,
        "success": sum(b(r, "success_marker") for r in rows) / n,
        "disp_along_mean": sum(f(r, "disp_along_push_m") for r in rows) / n,
        "tcp_err_mean": sum(f(r, "final_tcp_target_err_m") for r in rows) / n,
    }


def fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"n={int(stats['n'])} controlled={stats['controlled']:.9f} "
        f"impact={stats['impact']:.9f} low_motion={stats['low_motion']:.9f} "
        f"success={stats['success']:.9f} disp_along_mean_m={stats['disp_along_mean']:.9f} "
        f"final_tcp_err_mean_m={stats['tcp_err_mean']:.9f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    ap.add_argument("--x_edges", type=float, nargs=2, default=(0.257, 0.308))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    summary = json.loads(args.summary_json.read_text())
    if not rows:
        raise ValueError(f"empty csv: {args.csv}")
    required = {
        "cube_x0_m",
        "push_dx",
        "push_dy",
        "controlled_push",
        "impact_outlier",
        "low_motion",
        "success_marker",
        "disp_along_push_m",
        "final_tcp_target_err_m",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    edge0, edge1 = float(args.x_edges[0]), float(args.x_edges[1])
    by_dir: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dir[direction_key(row)].append(row)

    print(
        "bucket_audit line1 "
        f"csv_rows={len(rows)} summary_trials={summary.get('trials')} "
        f"controller={summary.get('controller')} learned_policy={summary.get('learned_policy')} "
        f"trajectory_variant={summary.get('trajectory_variant')} x_edges=({edge0:.6f},{edge1:.6f})"
    )
    print(f"bucket_audit line2 overall {fmt_stats(metrics(rows))}")

    line = 3
    for key in sorted(by_dir):
        print(f"bucket_audit line{line} direction={key} {fmt_stats(metrics(by_dir[key]))}")
        line += 1

    posx = by_dir.get((1, 0), [])
    buckets = {
        "low_x": [r for r in posx if f(r, "cube_x0_m") < edge0],
        "mid_x": [r for r in posx if edge0 <= f(r, "cube_x0_m") < edge1],
        "high_x": [r for r in posx if f(r, "cube_x0_m") >= edge1],
    }
    for name, items in buckets.items():
        applied = 0
        if items and "v31_lowx_applied" in items[0]:
            applied = sum(b(r, "v31_lowx_applied") for r in items)
        print(
            f"bucket_audit line{line} direction=(1, 0) bucket={name} "
            f"v31_lowx_applied={applied}/{len(items)} {fmt_stats(metrics(items))}"
        )
        line += 1

    pass_bucket = True
    for name, items in buckets.items():
        stats = metrics(items)
        if stats["n"] == 0:
            pass_bucket = False
        if stats["impact"] > 0.05:
            pass_bucket = False
        if name == "low_x" and stats["success"] < 0.20:
            pass_bucket = False
    print(
        f"bucket_audit line{line} verdict="
        f"{'PASS_POSX_BUCKET_SCREEN' if pass_bucket else 'FAIL_POSX_BUCKET_SCREEN'} "
        "learned_policy=NO track_a_grasp_success=NO"
    )
    return 0 if pass_bucket else 2


if __name__ == "__main__":
    raise SystemExit(main())
