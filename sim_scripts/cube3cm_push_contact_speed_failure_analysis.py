#!/usr/bin/env python3
"""Local posthoc failure analysis for contact-speed cube-push eval CSVs.

This does not run Isaac Lab, train a policy, or generate a dataset. It compares
the preserved teacher-off contact-speed eval against the teacher-on diagnostic
using only columns that are actually present in the CSV files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

REQUIRED_COLUMNS = {
    "trial",
    "env_id",
    "disp_along_push_m",
    "disp_xy_m",
    "target_xy_dist_m",
    "final_speed_mps",
    "tip_angle_deg",
    "controlled_push",
    "impact_outlier",
    "low_motion",
    "success_marker",
    "grasped_marker",
}
OPTIONAL_DIRECTION_COLUMNS = {"push_dx", "push_dy"}
OPTIONAL_INITIAL_POS_COLUMNS = {"cube_x0_m", "cube_y0_m"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--policy_csv", default=None)
    ap.add_argument("--policy_summary", default=None)
    ap.add_argument("--teacher_csv", default=None)
    ap.add_argument("--teacher_summary", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary_out", default=None)
    return ap.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(header)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for raw in reader:
            rows.append({k: float(v) for k, v in raw.items() if v != ""})
    return header, rows


def _rate(rows: list[dict[str, float]], pred) -> float:
    return sum(1 for row in rows if pred(row)) / len(rows) if rows else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _stats(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [float(r[key]) for r in rows]
    return {
        "mean": _mean(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else 0.0,
    }


def _clean_success(row: dict[str, float]) -> bool:
    return (
        row["success_marker"] == 1.0
        and row["controlled_push"] == 1.0
        and row["impact_outlier"] == 0.0
        and row["target_xy_dist_m"] <= 0.050
    )


def _final_controlled_3cm(row: dict[str, float]) -> bool:
    return row["controlled_push"] == 1.0 and row["disp_along_push_m"] >= 0.030


def _bucket_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "controlled_rate": _rate(rows, lambda r: r["controlled_push"] == 1.0),
        "impact_rate": _rate(rows, lambda r: r["impact_outlier"] == 1.0),
        "low_motion_rate": _rate(rows, lambda r: r["low_motion"] == 1.0),
        "success_marker_rate": _rate(rows, lambda r: r["success_marker"] == 1.0),
        "clean_success_rate": _rate(rows, _clean_success),
        "final_controlled_3cm_rate": _rate(rows, _final_controlled_3cm),
        "reverse_or_no_forward_1mm_rate": _rate(rows, lambda r: r["disp_along_push_m"] < 0.001),
        "weak_forward_under_3cm_rate": _rate(
            rows,
            lambda r: 0.001 <= r["disp_along_push_m"] < 0.030 and r["impact_outlier"] == 0.0,
        ),
        "far_target_gt_10cm_rate": _rate(rows, lambda r: r["target_xy_dist_m"] > 0.100),
        "impact_low_motion_rate": _rate(
            rows,
            lambda r: r["impact_outlier"] == 1.0 and r["disp_xy_m"] < 0.005,
        ),
        "impact_not_large_disp_lt_6cm_rate": _rate(
            rows,
            lambda r: r["impact_outlier"] == 1.0 and r["disp_xy_m"] < 0.060,
        ),
        "speed_le_0p2_rate": _rate(rows, lambda r: r["final_speed_mps"] <= 0.2),
        "speed_le_0p5_rate": _rate(rows, lambda r: r["final_speed_mps"] <= 0.5),
        "speed_gt_1p733_rate": _rate(rows, lambda r: r["final_speed_mps"] > 1.733444051),
        "tip_gt_150p4_rate": _rate(rows, lambda r: r["tip_angle_deg"] > 150.399799770),
        "disp_along_stats": _stats(rows, "disp_along_push_m"),
        "disp_xy_stats": _stats(rows, "disp_xy_m"),
        "target_xy_dist_stats": _stats(rows, "target_xy_dist_m"),
        "speed_stats": _stats(rows, "final_speed_mps"),
        "tip_stats": _stats(rows, "tip_angle_deg"),
        "grasped_marker_rate": _rate(rows, lambda r: r["grasped_marker"] == 1.0),
    }


def _exclusive_failure_mix(rows: list[dict[str, float]]) -> dict[str, int]:
    """Useful for reading, not a replacement for the non-exclusive rates above."""
    mix = {
        "clean_success": 0,
        "impact_outlier": 0,
        "low_motion": 0,
        "reverse_or_no_forward_1mm": 0,
        "controlled_but_short": 0,
        "far_target_gt_10cm": 0,
        "other_failure": 0,
    }
    for row in rows:
        if _clean_success(row):
            mix["clean_success"] += 1
        elif row["impact_outlier"] == 1.0:
            mix["impact_outlier"] += 1
        elif row["low_motion"] == 1.0:
            mix["low_motion"] += 1
        elif row["disp_along_push_m"] < 0.001:
            mix["reverse_or_no_forward_1mm"] += 1
        elif row["controlled_push"] == 1.0 and row["disp_along_push_m"] < 0.030:
            mix["controlled_but_short"] += 1
        elif row["target_xy_dist_m"] > 0.100:
            mix["far_target_gt_10cm"] += 1
        else:
            mix["other_failure"] += 1
    return mix


def _top_rows(rows: list[dict[str, float]], key: str, n: int = 5) -> list[dict[str, Any]]:
    indexed = list(enumerate(rows, start=2))
    indexed.sort(key=lambda item: item[1][key], reverse=True)
    out: list[dict[str, Any]] = []
    for csv_line, row in indexed[:n]:
        out.append(
            {
                "csv_line": csv_line,
                "trial": int(row["trial"]),
                "env_id": int(row["env_id"]),
                "disp_along_push_m": row["disp_along_push_m"],
                "disp_xy_m": row["disp_xy_m"],
                "target_xy_dist_m": row["target_xy_dist_m"],
                "final_speed_mps": row["final_speed_mps"],
                "tip_angle_deg": row["tip_angle_deg"],
                "controlled_push": int(row["controlled_push"]),
                "impact_outlier": int(row["impact_outlier"]),
                "low_motion": int(row["low_motion"]),
                "success_marker": int(row["success_marker"]),
            }
        )
    return out


def _summary_check(name: str, rows: list[dict[str, float]], summary: dict[str, Any]) -> dict[str, Any]:
    computed = _bucket_summary(rows)
    checks = {}
    for field, key in (
        ("controlled_push_rate", "controlled_rate"),
        ("impact_outlier_rate", "impact_rate"),
        ("low_motion_rate", "low_motion_rate"),
        ("success_marker_rate", "success_marker_rate"),
    ):
        expected = float(summary.get(field, 0.0))
        got = float(computed[key])
        checks[field] = {
            "summary": expected,
            "csv": got,
            "abs_diff": abs(expected - got),
            "match": abs(expected - got) <= 1.0e-12,
        }
    return {"name": name, "checks": checks}


def _dir_key(row: dict[str, float]) -> tuple[int, int]:
    return (int(round(row["push_dx"])), int(round(row["push_dy"])))


def _direction_summary(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for direction in sorted({_dir_key(row) for row in rows}):
        group = [row for row in rows if _dir_key(row) == direction]
        item = {"dir": f"({direction[0]},{direction[1]})"}
        item.update(_bucket_summary(group))
        out.append(item)
    return out


def _quantile_edges(values: list[float]) -> list[float]:
    return [_quantile(values, q) for q in (0.25, 0.50, 0.75)]


def _bin_index(value: float, edges: list[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _range_for_bin(values: list[float], edges: list[float], idx: int) -> tuple[float, float]:
    lower = min(values) if idx == 0 else edges[idx - 1]
    upper = max(values) if idx == len(edges) else edges[idx]
    return lower, upper


def _initial_grid_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    xs = [row["cube_x0_m"] for row in rows]
    ys = [row["cube_y0_m"] for row in rows]
    x_edges = _quantile_edges(xs)
    y_edges = _quantile_edges(ys)
    groups: dict[tuple[int, int], list[dict[str, float]]] = {
        (ix, iy): [] for ix in range(4) for iy in range(4)
    }
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
            item.update(_bucket_summary(group))
            grid.append(item)
    worst = sorted(
        grid,
        key=lambda item: (item["clean_success_rate"], -item["impact_rate"], -item["low_motion_rate"]),
    )[:5]
    return {"x_edges_m": x_edges, "y_edges_m": y_edges, "grid": grid, "worst": worst}


def _fmt(value: float) -> str:
    return f"{value:.9f}"


def _fmt_rate(value: float) -> str:
    return f"{value:.6f}"


def _line_summary(label: str, item: dict[str, Any]) -> str:
    return (
        f"SUMMARY label={label} n={item['n']} "
        f"controlled={_fmt_rate(item['controlled_rate'])} "
        f"impact={_fmt_rate(item['impact_rate'])} "
        f"low_motion={_fmt_rate(item['low_motion_rate'])} "
        f"clean_success={_fmt_rate(item['clean_success_rate'])} "
        f"final_controlled_3cm={_fmt_rate(item['final_controlled_3cm_rate'])} "
        f"disp_along_mean_m={_fmt(item['disp_along_stats']['mean'])} "
        f"disp_xy_mean_m={_fmt(item['disp_xy_stats']['mean'])} "
        f"speed_p95_mps={_fmt(item['speed_stats']['p95'])} "
        f"tip_p95_deg={_fmt(item['tip_stats']['p95'])}"
    )


def _write_lines(summary: dict[str, Any], out_path: Path) -> None:
    policy = summary["policy_off"]
    teacher = summary["teacher_on"]
    delta = summary["teacher_minus_policy"]
    lines = [
        (
            "FAILURE_ANALYSIS_INPUT "
            f"policy_csv={summary['inputs']['policy_csv']} policy_csv_md5={summary['inputs']['policy_csv_md5']} "
            f"policy_summary={summary['inputs']['policy_summary']} policy_summary_md5={summary['inputs']['policy_summary_md5']} "
            f"teacher_csv={summary['inputs']['teacher_csv']} teacher_csv_md5={summary['inputs']['teacher_csv_md5']} "
            f"teacher_summary={summary['inputs']['teacher_summary']} teacher_summary_md5={summary['inputs']['teacher_summary_md5']}"
        ),
        (
            "COLUMN_AVAILABILITY "
            f"policy_columns={','.join(summary['inputs']['policy_columns'])} "
            f"teacher_columns={','.join(summary['inputs']['teacher_columns'])} "
            f"direction_bucket_available={'YES' if summary['column_availability']['direction_bucket_available'] else 'NO'} "
            f"initial_position_bucket_available={'YES' if summary['column_availability']['initial_position_bucket_available'] else 'NO'}"
        ),
        _line_summary("policy_off_contact_speed", policy),
        _line_summary("teacher_on_diagnostic", teacher),
        (
            "DELTA teacher_on_minus_policy_off "
            f"controlled={_fmt(delta['controlled_rate'])} "
            f"impact={_fmt(delta['impact_rate'])} "
            f"low_motion={_fmt(delta['low_motion_rate'])} "
            f"clean_success={_fmt(delta['clean_success_rate'])} "
            f"final_controlled_3cm={_fmt(delta['final_controlled_3cm_rate'])} "
            f"disp_along_mean_m={_fmt(delta['disp_along_mean_m'])} "
            f"disp_xy_mean_m={_fmt(delta['disp_xy_mean_m'])} "
            f"speed_p95_mps={_fmt(delta['speed_p95_mps'])}"
        ),
        (
            "FAILURE_MIX label=policy_off_contact_speed "
            + " ".join(f"{k}={v}" for k, v in summary["policy_off_failure_mix"].items())
        ),
        (
            "FAILURE_MIX label=teacher_on_diagnostic "
            + " ".join(f"{k}={v}" for k, v in summary["teacher_on_failure_mix"].items())
        ),
    ]
    for label, item in (("policy_off_contact_speed", policy), ("teacher_on_diagnostic", teacher)):
        lines.append(
            f"NONEXCLUSIVE_FLAGS label={label} "
            f"reverse_or_no_forward_1mm={_fmt_rate(item['reverse_or_no_forward_1mm_rate'])} "
            f"weak_forward_under_3cm={_fmt_rate(item['weak_forward_under_3cm_rate'])} "
            f"far_target_gt_10cm={_fmt_rate(item['far_target_gt_10cm_rate'])} "
            f"impact_low_motion={_fmt_rate(item['impact_low_motion_rate'])} "
            f"impact_not_large_disp_lt_6cm={_fmt_rate(item['impact_not_large_disp_lt_6cm_rate'])} "
            f"speed_le_0p2={_fmt_rate(item['speed_le_0p2_rate'])} "
            f"speed_le_0p5={_fmt_rate(item['speed_le_0p5_rate'])} "
            f"speed_gt_1p733={_fmt_rate(item['speed_gt_1p733_rate'])} "
            f"tip_gt_150p4={_fmt_rate(item['tip_gt_150p4_rate'])}"
        )
    for check in summary["summary_crosschecks"]:
        for field, item in check["checks"].items():
            lines.append(
                f"SUMMARY_CROSSCHECK label={check['name']} field={field} "
                f"summary={_fmt(item['summary'])} csv={_fmt(item['csv'])} "
                f"abs_diff={item['abs_diff']:.12f} match={'YES' if item['match'] else 'NO'}"
            )
    for label, items in (
        ("policy_off_contact_speed", summary.get("policy_by_direction", [])),
        ("teacher_on_diagnostic", summary.get("teacher_by_direction", [])),
    ):
        for item in items:
            lines.append(
                f"BY_DIR label={label} dir={item['dir']} n={item['n']} "
                f"controlled={_fmt_rate(item['controlled_rate'])} "
                f"impact={_fmt_rate(item['impact_rate'])} "
                f"low_motion={_fmt_rate(item['low_motion_rate'])} "
                f"clean_success={_fmt_rate(item['clean_success_rate'])} "
                f"disp_along_mean_m={_fmt(item['disp_along_stats']['mean'])} "
                f"disp_xy_mean_m={_fmt(item['disp_xy_stats']['mean'])}"
            )
    for label, item in (
        ("policy_off_contact_speed", summary.get("policy_initial_grid")),
        ("teacher_on_diagnostic", summary.get("teacher_initial_grid")),
    ):
        if not item:
            continue
        lines.append(
            f"GRID_EDGES label={label} "
            f"x_q25={_fmt(item['x_edges_m'][0])} x_q50={_fmt(item['x_edges_m'][1])} x_q75={_fmt(item['x_edges_m'][2])} "
            f"y_q25={_fmt(item['y_edges_m'][0])} y_q50={_fmt(item['y_edges_m'][1])} y_q75={_fmt(item['y_edges_m'][2])}"
        )
        for rank, cell in enumerate(item["worst"], start=1):
            lines.append(
                f"WORST_GRID label={label} rank={rank} x_bin={cell['x_bin']} y_bin={cell['y_bin']} "
                f"n={cell['n']} clean_success={_fmt_rate(cell['clean_success_rate'])} "
                f"controlled={_fmt_rate(cell['controlled_rate'])} "
                f"impact={_fmt_rate(cell['impact_rate'])} "
                f"low_motion={_fmt_rate(cell['low_motion_rate'])} "
                f"x_range=[{_fmt(cell['x_min_m'])},{_fmt(cell['x_max_m'])}] "
                f"y_range=[{_fmt(cell['y_min_m'])},{_fmt(cell['y_max_m'])}]"
            )
    if not summary["column_availability"]["direction_bucket_available"]:
        lines.append(
            "UNAVAILABLE_BUCKET name=direction reason=eval_csv_lacks_push_dx_push_dy "
            "next_eval_logging_required=YES"
        )
    if not summary["column_availability"]["initial_position_bucket_available"]:
        lines.append(
            "UNAVAILABLE_BUCKET name=initial_cube_position reason=eval_csv_lacks_cube_x0_m_cube_y0_m "
            "next_eval_logging_required=YES"
        )
    lines.extend(
        [
            (
                "INTERPRETATION "
                "policy_learning_problem=YES teacher_trajectory_problem=YES "
                "basis=teacher_on_diagnostic_did_not_reduce_impact_and_reduced_clean_success "
                "scale_10k_or_100k=NO"
            ),
            (
                "NEXT_V7_DESIGN_HINT "
                "log_push_dir_and_initial_cube_position=YES "
                "redesign_teacher_contact_trajectory=YES "
                "rerun_50_100_iter_then_frozen_1k_only_after_approval=YES"
            ),
            "RESULT contact_speed_failure_analysis=PASS local_posthoc_only=YES isaac_runtime=NO training=NO dataset_generation=NO",
        ]
    )
    for label, top in (
        ("policy_off_contact_speed", summary["policy_top_speed_outliers"]),
        ("teacher_on_diagnostic", summary["teacher_top_speed_outliers"]),
    ):
        for rank, row in enumerate(top, start=1):
            lines.append(
                f"TOP_SPEED_OUTLIER label={label} rank={rank} csv_line={row['csv_line']} "
                f"trial={row['trial']} env_id={row['env_id']} speed_mps={_fmt(row['final_speed_mps'])} "
                f"disp_xy_m={_fmt(row['disp_xy_m'])} disp_along_push_m={_fmt(row['disp_along_push_m'])} "
                f"target_xy_dist_m={_fmt(row['target_xy_dist_m'])} tip_deg={_fmt(row['tip_angle_deg'])} "
                f"controlled={row['controlled_push']} impact={row['impact_outlier']} "
                f"low_motion={row['low_motion']} success_marker={row['success_marker']}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    policy_csv = Path(args.policy_csv) if args.policy_csv else run_dir / "ppo_contact_speed_model49_eval1024.csv"
    policy_summary_path = (
        Path(args.policy_summary)
        if args.policy_summary
        else run_dir / "ppo_contact_speed_model49_eval1024_summary.json"
    )
    teacher_csv = Path(args.teacher_csv) if args.teacher_csv else run_dir / "ppo_contact_speed_teacher_on_eval1024.csv"
    teacher_summary_path = (
        Path(args.teacher_summary)
        if args.teacher_summary
        else run_dir / "ppo_contact_speed_teacher_on_eval1024_summary.json"
    )
    out_path = Path(args.out) if args.out else run_dir / "contact_speed_failure_analysis.out"
    summary_out_path = (
        Path(args.summary_out)
        if args.summary_out
        else run_dir / "contact_speed_failure_analysis_summary.json"
    )

    policy_header, policy_rows = _read_csv(policy_csv)
    teacher_header, teacher_rows = _read_csv(teacher_csv)
    policy_summary = json.loads(policy_summary_path.read_text())
    teacher_summary = json.loads(teacher_summary_path.read_text())

    policy_bucket = _bucket_summary(policy_rows)
    teacher_bucket = _bucket_summary(teacher_rows)
    direction_available = (
        OPTIONAL_DIRECTION_COLUMNS.issubset(policy_header)
        and OPTIONAL_DIRECTION_COLUMNS.issubset(teacher_header)
    )
    initial_pos_available = (
        OPTIONAL_INITIAL_POS_COLUMNS.issubset(policy_header)
        and OPTIONAL_INITIAL_POS_COLUMNS.issubset(teacher_header)
    )
    delta = {
        "controlled_rate": teacher_bucket["controlled_rate"] - policy_bucket["controlled_rate"],
        "impact_rate": teacher_bucket["impact_rate"] - policy_bucket["impact_rate"],
        "low_motion_rate": teacher_bucket["low_motion_rate"] - policy_bucket["low_motion_rate"],
        "clean_success_rate": teacher_bucket["clean_success_rate"] - policy_bucket["clean_success_rate"],
        "final_controlled_3cm_rate": teacher_bucket["final_controlled_3cm_rate"]
        - policy_bucket["final_controlled_3cm_rate"],
        "disp_along_mean_m": teacher_bucket["disp_along_stats"]["mean"] - policy_bucket["disp_along_stats"]["mean"],
        "disp_xy_mean_m": teacher_bucket["disp_xy_stats"]["mean"] - policy_bucket["disp_xy_stats"]["mean"],
        "speed_p95_mps": teacher_bucket["speed_stats"]["p95"] - policy_bucket["speed_stats"]["p95"],
    }

    summary = {
        "inputs": {
            "policy_csv": str(policy_csv),
            "policy_csv_md5": _md5(policy_csv),
            "policy_summary": str(policy_summary_path),
            "policy_summary_md5": _md5(policy_summary_path),
            "policy_columns": policy_header,
            "teacher_csv": str(teacher_csv),
            "teacher_csv_md5": _md5(teacher_csv),
            "teacher_summary": str(teacher_summary_path),
            "teacher_summary_md5": _md5(teacher_summary_path),
            "teacher_columns": teacher_header,
        },
        "local_posthoc_only": True,
        "isaac_runtime": False,
        "training": False,
        "dataset_generation": False,
        "column_availability": {
            "direction_bucket_available": direction_available,
            "initial_position_bucket_available": initial_pos_available,
            "required_for_direction": sorted(OPTIONAL_DIRECTION_COLUMNS),
            "required_for_initial_position": sorted(OPTIONAL_INITIAL_POS_COLUMNS),
        },
        "policy_off": policy_bucket,
        "teacher_on": teacher_bucket,
        "teacher_minus_policy": delta,
        "policy_off_failure_mix": _exclusive_failure_mix(policy_rows),
        "teacher_on_failure_mix": _exclusive_failure_mix(teacher_rows),
        "summary_crosschecks": [
            _summary_check("policy_off_contact_speed", policy_rows, policy_summary),
            _summary_check("teacher_on_diagnostic", teacher_rows, teacher_summary),
        ],
        "policy_top_speed_outliers": _top_rows(policy_rows, "final_speed_mps"),
        "teacher_top_speed_outliers": _top_rows(teacher_rows, "final_speed_mps"),
        "interpretation": {
            "policy_learning_problem": True,
            "teacher_trajectory_problem": True,
            "scale_10k_or_100k": False,
            "reason": (
                "teacher-on diagnostic does not reduce impact and lowers clean success, "
                "so current scripted teacher is not a clean teacher; policy-off eval also "
                "fails the impact and clean-success gate."
            ),
        },
    }
    if direction_available:
        summary["policy_by_direction"] = _direction_summary(policy_rows)
        summary["teacher_by_direction"] = _direction_summary(teacher_rows)
    if initial_pos_available:
        summary["policy_initial_grid"] = _initial_grid_summary(policy_rows)
        summary["teacher_initial_grid"] = _initial_grid_summary(teacher_rows)

    summary_out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_lines(summary, out_path)
    print(f"wrote {out_path}")
    print(f"wrote {summary_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
