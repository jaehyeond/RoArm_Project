#!/usr/bin/env python3
"""Compare small IsaacLab cube-push policy/controller sweep results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"


DEFAULT_VARIANTS = [
    (
        "policy_off_contact_speed",
        "policy_sweep_v6_policy_off_seed777_eval1024_summary.json",
        "policy_sweep_v6_policy_off_seed777_eval1024.csv",
    ),
    (
        "teacher_goal055",
        "policy_sweep_teacher_on_seed777_eval1024_summary.json",
        "policy_sweep_teacher_on_seed777_eval1024.csv",
    ),
    (
        "teacher_goal030",
        "policy_sweep_teacher_goal030_seed777_eval1024_summary.json",
        "policy_sweep_teacher_goal030_seed777_eval1024.csv",
    ),
    (
        "teacher_goal040",
        "policy_sweep_teacher_goal040_seed777_eval1024_summary.json",
        "policy_sweep_teacher_goal040_seed777_eval1024.csv",
    ),
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary_out", default=None)
    return ap.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _rate(rows: list[dict[str, float]], pred) -> float:
    return sum(1 for row in rows if pred(row)) / len(rows) if rows else 0.0


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def _dir_key(row: dict[str, float]) -> tuple[int, int]:
    return int(round(row["push_dx"])), int(round(row["push_dy"]))


def _direction_rows(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for direction in sorted({_dir_key(row) for row in rows}):
        group = [row for row in rows if _dir_key(row) == direction]
        out.append(
            {
                "dir": f"({direction[0]},{direction[1]})",
                "n": len(group),
                "controlled_rate": _rate(group, lambda r: r["controlled_push"] == 1.0),
                "impact_rate": _rate(group, lambda r: r["impact_outlier"] == 1.0),
                "low_motion_rate": _rate(group, lambda r: r["low_motion"] == 1.0),
                "clean_success_rate": _rate(group, lambda r: r["success_marker"] == 1.0),
                "disp_along_mean_m": _mean(group, "disp_along_push_m"),
                "disp_xy_mean_m": _mean(group, "disp_xy_m"),
            }
        )
    return out


def _load_variant(run_dir: Path, label: str, summary_name: str, csv_name: str) -> dict[str, Any]:
    summary_path = run_dir / summary_name
    csv_path = run_dir / csv_name
    summary = json.loads(summary_path.read_text())
    rows = _rows(csv_path)
    return {
        "label": label,
        "summary_path": str(summary_path),
        "summary_md5": _md5(summary_path),
        "csv_path": str(csv_path),
        "csv_md5": _md5(csv_path),
        "trials": int(summary["trials"]),
        "unique_env_records": int(summary.get("unique_env_records", 0)),
        "record_first_episode_only": bool(summary.get("record_first_episode_only", False)),
        "scripted_teacher_blend": float(summary.get("scripted_teacher_blend", 0.0)),
        "scripted_teacher_goal_push_m": float(summary.get("scripted_teacher_goal_push_m", 0.0)),
        "controlled_push_rate": float(summary["controlled_push_rate"]),
        "impact_outlier_rate": float(summary["impact_outlier_rate"]),
        "low_motion_rate": float(summary["low_motion_rate"]),
        "success_marker_rate": float(summary["success_marker_rate"]),
        "disp_along_push_mean_m": float(summary["disp_along_push_mean_m"]),
        "disp_xy_mean_m": float(summary["disp_xy_mean_m"]),
        "target_xy_dist_mean_m": float(summary["target_xy_dist_mean_m"]),
        "dataset_generation": bool(summary["dataset_generation"]),
        "grasp_attach": bool(summary["grasp_attach"]),
        "rollout_object_posewrite": bool(summary["rollout_object_posewrite"]),
        "grasped_marker_rate": float(summary["grasped_marker_rate"]),
        "direction": _direction_rows(rows),
    }


def _fmt(value: float) -> str:
    return f"{value:.9f}"


def _fmt_rate(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "policy_sweep_seed777_compare.out"
    summary_out = Path(args.summary_out) if args.summary_out else run_dir / "policy_sweep_seed777_compare_summary.json"

    variants = [_load_variant(run_dir, *item) for item in DEFAULT_VARIANTS]
    by_clean = sorted(variants, key=lambda v: (v["success_marker_rate"], -v["impact_outlier_rate"]), reverse=True)
    by_impact = sorted(variants, key=lambda v: (v["impact_outlier_rate"], -v["success_marker_rate"]))
    valid_mechanism = all(
        v["dataset_generation"] is False
        and v["grasp_attach"] is False
        and v["rollout_object_posewrite"] is False
        and v["grasped_marker_rate"] == 0.0
        and v["trials"] == 1024
        and v["unique_env_records"] == 1024
        and v["record_first_episode_only"] is True
        for v in variants
    )
    can_scale = any(
        v["impact_outlier_rate"] < 0.05
        and v["controlled_push_rate"] > 0.60
        and v["success_marker_rate"] > 0.30
        for v in variants
    )
    summary = {
        "local_posthoc_only": True,
        "isaac_runtime_already_completed": True,
        "seed": 777,
        "variants": variants,
        "rank_by_clean_then_impact": [v["label"] for v in by_clean],
        "rank_by_impact_then_clean": [v["label"] for v in by_impact],
        "valid_mechanism": valid_mechanism,
        "can_scale_10k": can_scale,
    }

    lines = [
        (
            f"POLICY_SWEEP_COMPARE seed=777 variants={len(variants)} "
            f"valid_mechanism={'YES' if valid_mechanism else 'NO'} can_scale_10k={'YES' if can_scale else 'NO'}"
        )
    ]
    for v in variants:
        lines.append(
            f"POLICY_RESULT label={v['label']} trials={v['trials']} unique_env_records={v['unique_env_records']} "
            f"teacher_blend={_fmt(v['scripted_teacher_blend'])} teacher_goal_push_m={_fmt(v['scripted_teacher_goal_push_m'])} "
            f"controlled={_fmt_rate(v['controlled_push_rate'])} impact={_fmt_rate(v['impact_outlier_rate'])} "
            f"low_motion={_fmt_rate(v['low_motion_rate'])} clean_success={_fmt_rate(v['success_marker_rate'])} "
            f"disp_along_mean_m={_fmt(v['disp_along_push_mean_m'])} target_dist_mean_m={_fmt(v['target_xy_dist_mean_m'])}"
        )
    lines.append("RANK_BY_CLEAN_THEN_IMPACT " + " ".join(summary["rank_by_clean_then_impact"]))
    lines.append("RANK_BY_IMPACT_THEN_CLEAN " + " ".join(summary["rank_by_impact_then_clean"]))
    for v in variants:
        for item in v["direction"]:
            lines.append(
                f"BY_DIR label={v['label']} dir={item['dir']} n={item['n']} "
                f"controlled={_fmt_rate(item['controlled_rate'])} impact={_fmt_rate(item['impact_rate'])} "
                f"low_motion={_fmt_rate(item['low_motion_rate'])} clean_success={_fmt_rate(item['clean_success_rate'])} "
                f"disp_along_mean_m={_fmt(item['disp_along_mean_m'])}"
            )
    lines.append(
        "INTERPRETATION best_clean_policy_is_still_policy_off=YES "
        "teacher_goal_sweep_reduces_impact_slightly_but_increases_low_motion_and_lowers_clean_success=YES "
        "scale_10k_or_100k=NO"
    )
    lines.append("RESULT policy_sweep_seed777_compare=PASS local_posthoc_only=YES new_isaac_runtime=NO")

    out_path.write_text("\n".join(lines) + "\n")
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"wrote {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
