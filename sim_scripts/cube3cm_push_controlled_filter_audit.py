#!/usr/bin/env python3
"""Posthoc controlled-push filter audit for the 3cm cube rollout CSV.

This script does not run Isaac Lab, train a policy, or generate a dataset. It
only reads the preserved per-env rollout CSV and separates stable-looking
pushes from low-motion trials and impact outliers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

SPEED_P95_MPS = 1.302103193
SPEED_P99_MPS = 1.733444051
TIP_P95_DEG = 141.181661216
TIP_P99_DEG = 150.399799770
DISP_XY_P99_M = 0.133549188
CONTROLLED_DISP_ALONG_MIN_M = 0.001
LOW_MOTION_DISP_XY_M = 0.005
MOVED_5MM_M = 0.005


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_RUN_DIR / "per_env.csv"))
    ap.add_argument("--summary", default=str(DEFAULT_RUN_DIR / "summary.json"))
    ap.add_argument("--out", default=str(DEFAULT_RUN_DIR / "controlled_push_filter_audit.out"))
    ap.add_argument("--summary_out", default=str(DEFAULT_RUN_DIR / "controlled_push_filter_summary.json"))
    return ap.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _percentile(values: list[float], q: float) -> float:
    """Match the preserved audit's percentile convention.

    The existing rollout_stats_audit.out values match NumPy's nearest method,
    not the default linear interpolation used by summary.json.
    """
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    idx = int(round(pos))
    return xs[max(0, min(len(xs) - 1, idx))]


def _rate(rows: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float("nan")


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else float("nan"),
    }


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.9f}"


def _fmt_rate(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _dir_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(round(row["push_dx"])), int(round(row["push_dy"])))


def _valid_basic(row: dict[str, Any]) -> bool:
    return (
        row["ik_ok"] == 1
        and row["grasped_marker"] == 0
        and abs(row["action_saturation_frac"]) <= 1e-12
    )


def _low_motion(row: dict[str, Any]) -> bool:
    return row["disp_xy_m"] < LOW_MOTION_DISP_XY_M


def _controlled_push_p95(row: dict[str, Any]) -> bool:
    return (
        _valid_basic(row)
        and row["disp_along_push_m"] >= CONTROLLED_DISP_ALONG_MIN_M
        and row["max_cube_speed_mps"] <= SPEED_P95_MPS
        and row["tip_angle_deg"] <= TIP_P95_DEG
        and row["disp_xy_m"] <= DISP_XY_P99_M
    )


def _impact_outlier_p99(row: dict[str, Any]) -> bool:
    return (
        row["max_cube_speed_mps"] > SPEED_P99_MPS
        or row["disp_xy_m"] > DISP_XY_P99_M
        or row["tip_angle_deg"] > TIP_P99_DEG
    )


def _failure_not_controlled(row: dict[str, Any]) -> bool:
    return _valid_basic(row) and not _controlled_push_p95(row)


def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            parsed: dict[str, Any] = {"csv_line": line_no}
            parsed["episode"] = _int(row, "episode")
            parsed["env_id"] = _int(row, "env_id")
            parsed["ik_ok"] = _int(row, "ik_ok")
            parsed["grasped_marker"] = _int(row, "grasped_marker")
            for key in (
                "ik_err_max_mm",
                "cube_x0_m",
                "cube_y0_m",
                "push_dx",
                "push_dy",
                "tcp_z_target_m",
                "disp_x_m",
                "disp_y_m",
                "disp_z_m",
                "disp_xy_m",
                "disp_total_m",
                "disp_along_push_m",
                "max_cube_speed_mps",
                "min_tcp_cube_dist_m",
                "tip_angle_deg",
                "q_err_max_deg",
                "action_abs_mean",
                "action_saturation_frac",
            ):
                parsed[key] = _float(row, key)
            rows.append(parsed)
    return rows


def _with_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["valid_basic"] = _valid_basic(row)
        row["low_motion"] = _low_motion(row)
        row["controlled_push_p95"] = _controlled_push_p95(row)
        row["impact_outlier_p99"] = _impact_outlier_p99(row)
        row["failure_not_controlled"] = _failure_not_controlled(row)


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    disp = [r["disp_xy_m"] for r in rows]
    return {
        "n": len(rows),
        "mean_disp_xy_m": _mean(disp),
        "median_disp_xy_m": _percentile(disp, 50),
        "p90_disp_xy_m": _percentile(disp, 90),
        "p95_disp_xy_m": _percentile(disp, 95),
        "moved_5mm_rate": _rate(rows, lambda r: r["disp_xy_m"] >= MOVED_5MM_M),
        "controlled_push_rate": _rate(rows, lambda r: bool(r["controlled_push_p95"])),
        "impact_outlier_rate": _rate(rows, lambda r: bool(r["impact_outlier_p99"])),
        "low_motion_rate": _rate(rows, lambda r: bool(r["low_motion"])),
        "failure_not_controlled_rate": _rate(rows, lambda r: bool(r["failure_not_controlled"])),
    }


def _quantile_edges(values: list[float]) -> list[float]:
    return [_percentile(values, q) for q in (25, 50, 75)]


def _bin_index(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _range_for_bin(values: list[float], edges: list[float], idx: int) -> tuple[float, float]:
    lower = min(values) if idx == 0 else edges[idx - 1]
    upper = max(values) if idx == len(edges) else edges[idx]
    return (lower, upper)


def _direction_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for direction in sorted({_dir_key(r) for r in rows}):
        group = [r for r in rows if _dir_key(r) == direction]
        out[f"({direction[0]},{direction[1]})"] = _summarize_group(group)
    return out


def _grid_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    xs = [r["cube_x0_m"] for r in rows]
    ys = [r["cube_y0_m"] for r in rows]
    x_edges = _quantile_edges(xs)
    y_edges = _quantile_edges(ys)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {(ix, iy): [] for ix in range(4) for iy in range(4)}
    for row in rows:
        groups[(_bin_index(row["cube_x0_m"], x_edges), _bin_index(row["cube_y0_m"], y_edges))].append(row)

    grid: list[dict[str, Any]] = []
    for ix in range(4):
        for iy in range(4):
            group = groups[(ix, iy)]
            x_range = _range_for_bin(xs, x_edges, ix)
            y_range = _range_for_bin(ys, y_edges, iy)
            item: dict[str, Any] = {
                "x_bin": f"Q{ix + 1}",
                "y_bin": f"Q{iy + 1}",
                "x_min_m": x_range[0],
                "x_max_m": x_range[1],
                "y_min_m": y_range[0],
                "y_max_m": y_range[1],
            }
            item.update(_summarize_group(group))
            grid.append(item)
    meta = {"x_edges_m": x_edges, "y_edges_m": y_edges}
    return meta, grid


def _line_for_bucket(name: str, rows: list[dict[str, Any]], total: int, criteria: str) -> str:
    return (
        f"BUCKET name={name} count={len(rows)} rate={_fmt_rate(len(rows) / total)} "
        f"criteria=\"{criteria}\""
    )


def _write_outputs(
    rows: list[dict[str, Any]],
    csv_path: Path,
    summary_path: Path,
    out_path: Path,
    summary_out_path: Path,
) -> None:
    total = len(rows)
    summary_json: dict[str, Any] = {}
    if summary_path.exists():
        summary_json = json.loads(summary_path.read_text())

    valid_rows = [r for r in rows if r["valid_basic"]]
    low_rows = [r for r in rows if r["low_motion"]]
    controlled_rows = [r for r in rows if r["controlled_push_p95"]]
    impact_rows = [r for r in rows if r["impact_outlier_p99"]]
    non_impact_rows = [r for r in rows if not r["impact_outlier_p99"]]
    failure_rows = [r for r in rows if r["failure_not_controlled"]]

    disp_values = [r["disp_xy_m"] for r in rows]
    non_impact_disp = [r["disp_xy_m"] for r in non_impact_rows]
    controlled_disp = [r["disp_xy_m"] for r in controlled_rows]

    computed = {
        "disp_xy_p99_m": _percentile(disp_values, 99),
        "max_cube_speed_p95_mps": _percentile([r["max_cube_speed_mps"] for r in rows], 95),
        "max_cube_speed_p99_mps": _percentile([r["max_cube_speed_mps"] for r in rows], 99),
        "tip_angle_p95_deg": _percentile([r["tip_angle_deg"] for r in rows], 95),
        "tip_angle_p99_deg": _percentile([r["tip_angle_deg"] for r in rows], 99),
    }

    by_direction = _direction_summary(rows)
    grid_meta, grid = _grid_summary(rows)
    worst_grids = sorted(
        grid,
        key=lambda g: (g["failure_not_controlled_rate"], g["impact_outlier_rate"], g["low_motion_rate"]),
        reverse=True,
    )[:5]
    top_impact = sorted(
        impact_rows,
        key=lambda r: (r["disp_xy_m"], r["max_cube_speed_mps"], r["tip_angle_deg"]),
        reverse=True,
    )[:10]

    output_summary = {
        "source_csv": str(csv_path),
        "source_csv_md5": _md5(csv_path),
        "source_summary": str(summary_path),
        "source_summary_md5": _md5(summary_path) if summary_path.exists() else None,
        "local_posthoc_only": True,
        "isaac_runtime": False,
        "training": False,
        "dataset_generation": False,
        "thresholds": {
            "valid_basic": "ik_ok == 1 and grasped_marker == 0 and action_saturation_frac == 0",
            "low_motion_disp_xy_lt_m": LOW_MOTION_DISP_XY_M,
            "controlled_disp_along_push_min_m": CONTROLLED_DISP_ALONG_MIN_M,
            "controlled_speed_p95_max_mps": SPEED_P95_MPS,
            "controlled_tip_p95_max_deg": TIP_P95_DEG,
            "controlled_disp_xy_p99_max_m": DISP_XY_P99_M,
            "impact_speed_p99_gt_mps": SPEED_P99_MPS,
            "impact_disp_xy_p99_gt_m": DISP_XY_P99_M,
            "impact_tip_p99_gt_deg": TIP_P99_DEG,
        },
        "computed_percentile_crosscheck": computed,
        "counts": {
            "total": total,
            "valid_basic": len(valid_rows),
            "low_motion": len(low_rows),
            "controlled_push_p95": len(controlled_rows),
            "impact_outlier_p99": len(impact_rows),
            "non_impact": len(non_impact_rows),
            "failure_not_controlled": len(failure_rows),
        },
        "rates": {
            "valid_basic": len(valid_rows) / total,
            "low_motion": len(low_rows) / total,
            "controlled_push_p95": len(controlled_rows) / total,
            "impact_outlier_p99": len(impact_rows) / total,
            "non_impact": len(non_impact_rows) / total,
            "failure_not_controlled": len(failure_rows) / total,
        },
        "disp_xy_all": _stats(disp_values),
        "disp_xy_non_impact": _stats(non_impact_disp),
        "disp_xy_controlled_push_p95": _stats(controlled_disp),
        "by_direction": by_direction,
        "initial_position_grid_meta": grid_meta,
        "initial_position_grid": grid,
        "worst_initial_position_grid_by_failure": worst_grids,
        "top_impact_outliers": top_impact,
    }

    lines: list[str] = []
    lines.append(
        "CONTROLLED_FILTER_INPUT "
        f"rows={total} summary_total_trials={summary_json.get('total_trials', 'NA')} "
        f"csv={csv_path} csv_md5={output_summary['source_csv_md5']} "
        f"summary={summary_path} summary_md5={output_summary['source_summary_md5']}"
    )
    lines.append(
        "THRESHOLDS "
        "threshold_source=rollout_stats_audit.out_lines_5_7_9 "
        f"low_motion_disp_xy_lt_m={LOW_MOTION_DISP_XY_M:.3f} "
        f"controlled_disp_along_push_min_m={CONTROLLED_DISP_ALONG_MIN_M:.3f} "
        f"controlled_speed_max_mps={SPEED_P95_MPS:.9f} "
        f"controlled_tip_max_deg={TIP_P95_DEG:.9f} "
        f"controlled_disp_xy_max_m={DISP_XY_P99_M:.9f} "
        f"impact_speed_gt_mps={SPEED_P99_MPS:.9f} "
        f"impact_disp_xy_gt_m={DISP_XY_P99_M:.9f} "
        f"impact_tip_gt_deg={TIP_P99_DEG:.9f}"
    )
    checks = [
        ("disp_xy_p99_m", computed["disp_xy_p99_m"], DISP_XY_P99_M),
        ("max_cube_speed_p95_mps", computed["max_cube_speed_p95_mps"], SPEED_P95_MPS),
        ("max_cube_speed_p99_mps", computed["max_cube_speed_p99_mps"], SPEED_P99_MPS),
        ("tip_angle_p95_deg", computed["tip_angle_p95_deg"], TIP_P95_DEG),
        ("tip_angle_p99_deg", computed["tip_angle_p99_deg"], TIP_P99_DEG),
    ]
    for name, value, audit_value in checks:
        diff = abs(value - audit_value)
        lines.append(
            f"THRESHOLD_CHECK key={name} computed={_fmt(value)} "
            f"audit={_fmt(audit_value)} abs_diff={diff:.12f} match={'YES' if diff <= 5e-7 else 'NO'}"
        )
    lines.append(_line_for_bucket("valid_basic", valid_rows, total, "ik_ok==1 and grasped_marker==0 and action_saturation_frac==0"))
    lines.append(_line_for_bucket("low_motion", low_rows, total, "disp_xy_m < 0.005"))
    lines.append(
        _line_for_bucket(
            "controlled_push_p95",
            controlled_rows,
            total,
            "valid_basic and disp_along_push_m>=0.001 and speed<=p95 and tip<=p95 and disp_xy<=p99",
        )
    )
    lines.append(
        _line_for_bucket(
            "impact_outlier_p99",
            impact_rows,
            total,
            "speed>p99 or disp_xy>p99 or tip>p99",
        )
    )
    lines.append(_line_for_bucket("failure_not_controlled", failure_rows, total, "valid_basic and not controlled_push_p95"))

    all_stats = output_summary["disp_xy_all"]
    non_stats = output_summary["disp_xy_non_impact"]
    ctl_stats = output_summary["disp_xy_controlled_push_p95"]
    lines.append(
        "OUTLIER_REMOVAL_EFFECT "
        f"removed_count={len(impact_rows)} removed_rate={_fmt_rate(len(impact_rows) / total)} "
        f"all_mean_disp_xy_m={_fmt(all_stats['mean'])} non_impact_mean_disp_xy_m={_fmt(non_stats['mean'])} "
        f"controlled_mean_disp_xy_m={_fmt(ctl_stats['mean'])} "
        f"all_std_disp_xy_m={_fmt(all_stats['std'])} non_impact_std_disp_xy_m={_fmt(non_stats['std'])} "
        f"all_p95_disp_xy_m={_fmt(all_stats['p95'])} non_impact_p95_disp_xy_m={_fmt(non_stats['p95'])}"
    )
    for direction, item in by_direction.items():
        lines.append(
            f"BY_DIR dir={direction} n={item['n']} "
            f"mean_disp_xy_m={_fmt(item['mean_disp_xy_m'])} "
            f"median_disp_xy_m={_fmt(item['median_disp_xy_m'])} "
            f"p90_disp_xy_m={_fmt(item['p90_disp_xy_m'])} "
            f"p95_disp_xy_m={_fmt(item['p95_disp_xy_m'])} "
            f"moved_5mm_rate={_fmt_rate(item['moved_5mm_rate'])} "
            f"controlled_push_rate={_fmt_rate(item['controlled_push_rate'])} "
            f"impact_outlier_rate={_fmt_rate(item['impact_outlier_rate'])} "
            f"low_motion_rate={_fmt_rate(item['low_motion_rate'])} "
            f"failure_not_controlled_rate={_fmt_rate(item['failure_not_controlled_rate'])}"
        )
    lines.append(
        "INITIAL_GRID_EDGES "
        f"x_q25={_fmt(grid_meta['x_edges_m'][0])} x_q50={_fmt(grid_meta['x_edges_m'][1])} "
        f"x_q75={_fmt(grid_meta['x_edges_m'][2])} y_q25={_fmt(grid_meta['y_edges_m'][0])} "
        f"y_q50={_fmt(grid_meta['y_edges_m'][1])} y_q75={_fmt(grid_meta['y_edges_m'][2])}"
    )
    for item in grid:
        lines.append(
            f"GRID_INITIAL_POS x_bin={item['x_bin']} x_range=[{_fmt(item['x_min_m'])},{_fmt(item['x_max_m'])}] "
            f"y_bin={item['y_bin']} y_range=[{_fmt(item['y_min_m'])},{_fmt(item['y_max_m'])}] "
            f"n={item['n']} mean_disp_xy_m={_fmt(item['mean_disp_xy_m'])} "
            f"moved_5mm_rate={_fmt_rate(item['moved_5mm_rate'])} "
            f"controlled_push_rate={_fmt_rate(item['controlled_push_rate'])} "
            f"impact_outlier_rate={_fmt_rate(item['impact_outlier_rate'])} "
            f"low_motion_rate={_fmt_rate(item['low_motion_rate'])} "
            f"failure_not_controlled_rate={_fmt_rate(item['failure_not_controlled_rate'])}"
        )
    for rank, item in enumerate(worst_grids, start=1):
        lines.append(
            f"WORST_INITIAL_POS_BY_FAILURE rank={rank} x_bin={item['x_bin']} y_bin={item['y_bin']} "
            f"n={item['n']} failure_not_controlled_rate={_fmt_rate(item['failure_not_controlled_rate'])} "
            f"controlled_push_rate={_fmt_rate(item['controlled_push_rate'])} "
            f"low_motion_rate={_fmt_rate(item['low_motion_rate'])} "
            f"impact_outlier_rate={_fmt_rate(item['impact_outlier_rate'])}"
        )
    for rank, row in enumerate(top_impact[:5], start=1):
        lines.append(
            f"TOP_IMPACT_OUTLIER rank={rank} csv_line={row['csv_line']} ep={row['episode']} env={row['env_id']} "
            f"dir=({int(round(row['push_dx']))},{int(round(row['push_dy']))}) "
            f"disp_xy_m={_fmt(row['disp_xy_m'])} disp_push_m={_fmt(row['disp_along_push_m'])} "
            f"speed_mps={_fmt(row['max_cube_speed_mps'])} tip_deg={_fmt(row['tip_angle_deg'])}"
        )
    lines.append(f"SUMMARY_JSON path={summary_out_path}")
    lines.append("RESULT controlled_filter_audit=PASS local_posthoc_only=YES isaac_runtime=NO training=NO dataset_generation=NO")

    out_path.write_text("\n".join(lines) + "\n")
    summary_out_path.write_text(json.dumps(output_summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    out_path = Path(args.out)
    summary_out_path = Path(args.summary_out)
    rows = _read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"no rows read from {csv_path}")
    _with_flags(rows)
    _write_outputs(rows, csv_path, summary_path, out_path, summary_out_path)
    print(f"wrote {out_path}")
    print(f"wrote {summary_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
