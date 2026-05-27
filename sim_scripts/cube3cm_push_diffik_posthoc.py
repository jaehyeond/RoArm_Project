"""Posthoc buckets for IsaacLab Differential IK cube-push probe CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pct(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(float(r[key])) for r in rows) / len(rows) if rows else 0.0


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(f(r, key) for r in rows) / len(rows) if rows else 0.0


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "controlled": pct(rows, "controlled_push"),
        "impact": pct(rows, "impact_outlier"),
        "low_motion": pct(rows, "low_motion"),
        "success": pct(rows, "success_marker"),
        "disp_along_mean": mean(rows, "disp_along_push_m"),
        "disp_xy_mean": mean(rows, "disp_xy_m"),
        "disp_xy_p90": quantile([f(r, "disp_xy_m") for r in rows], 0.90),
        "disp_xy_p95": quantile([f(r, "disp_xy_m") for r in rows], 0.95),
        "speed_max": max((f(r, "max_cube_speed_mps") for r in rows), default=0.0),
        "tcp_err_final_mean": mean(rows, "final_tcp_target_err_m"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    args = ap.parse_args()

    with args.csv.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    with args.summary_json.open() as fp:
        summary = json.load(fp)

    overall = summarize(rows)
    print(
        "diffik_posthoc line1 "
        f"csv_rows={len(rows)} summary_trials={summary.get('trials')} "
        f"controller={summary.get('controller')} command_type={summary.get('command_type')} "
        f"posewrite_calls={summary.get('posewrite_calls_during_rollout')} "
        f"learned_policy=NO"
    )
    print(
        "diffik_posthoc line2 overall "
        f"controlled={overall['controlled']:.9f} impact={overall['impact']:.9f} "
        f"low_motion={overall['low_motion']:.9f} success_marker={overall['success']:.9f} "
        f"disp_along_mean_m={overall['disp_along_mean']:.9f} "
        f"disp_xy_mean_m={overall['disp_xy_mean']:.9f} "
        f"disp_xy_p95_m={overall['disp_xy_p95']:.9f} "
        f"max_speed_mps={overall['speed_max']:.9f}"
    )

    dir_groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dir_groups[(int(round(f(row, "push_dx"))), int(round(f(row, "push_dy"))))].append(row)
    weakest_key = None
    weakest_controlled = 2.0
    for idx, key in enumerate(sorted(dir_groups), start=3):
        stats = summarize(dir_groups[key])
        if stats["controlled"] < weakest_controlled:
            weakest_controlled = float(stats["controlled"])
            weakest_key = key
        print(
            f"diffik_posthoc line{idx} direction={key} n={stats['n']} "
            f"controlled={stats['controlled']:.9f} impact={stats['impact']:.9f} "
            f"low_motion={stats['low_motion']:.9f} success_marker={stats['success']:.9f} "
            f"disp_along_mean_m={stats['disp_along_mean']:.9f} "
            f"disp_xy_p95_m={stats['disp_xy_p95']:.9f} "
            f"max_speed_mps={stats['speed_max']:.9f} "
            f"tcp_err_final_mean_m={stats['tcp_err_final_mean']:.9f}"
        )

    line = 3 + len(dir_groups)
    print(
        f"diffik_posthoc line{line} weakest_direction={weakest_key} "
        f"controlled={weakest_controlled:.9f}"
    )

    xs = sorted(f(r, "cube_x0_m") for r in rows)
    ys = sorted(f(r, "cube_y0_m") for r in rows)
    x_q1, x_q2 = quantile(xs, 1 / 3), quantile(xs, 2 / 3)
    y_q1, y_q2 = quantile(ys, 1 / 3), quantile(ys, 2 / 3)
    grid: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        xb = 0 if f(row, "cube_x0_m") <= x_q1 else 1 if f(row, "cube_x0_m") <= x_q2 else 2
        yb = 0 if f(row, "cube_y0_m") <= y_q1 else 1 if f(row, "cube_y0_m") <= y_q2 else 2
        grid[(xb, yb)].append(row)

    worst_grid_key = None
    worst_grid_low_or_impact = -1.0
    for key, group in sorted(grid.items()):
        stats = summarize(group)
        risk = float(stats["low_motion"]) + float(stats["impact"])
        if risk > worst_grid_low_or_impact:
            worst_grid_low_or_impact = risk
            worst_grid_key = key

    line += 1
    print(
        f"diffik_posthoc line{line} initial_grid_quantiles "
        f"x_q1={x_q1:.9f} x_q2={x_q2:.9f} y_q1={y_q1:.9f} y_q2={y_q2:.9f} "
        f"worst_grid={worst_grid_key} low_plus_impact={worst_grid_low_or_impact:.9f}"
    )
    for key, group in sorted(grid.items()):
        line += 1
        stats = summarize(group)
        print(
            f"diffik_posthoc line{line} grid={key} n={stats['n']} "
            f"controlled={stats['controlled']:.9f} impact={stats['impact']:.9f} "
            f"low_motion={stats['low_motion']:.9f} success_marker={stats['success']:.9f} "
            f"disp_along_mean_m={stats['disp_along_mean']:.9f} "
            f"max_speed_mps={stats['speed_max']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
