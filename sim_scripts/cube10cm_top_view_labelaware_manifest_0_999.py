#!/usr/bin/env python3
"""Create a label-aware 0-999 top-view manifest without rendering.

This utility writes a deterministic 1000 episode plan for the professor 10cm
cube top-view visual trajectory dataset branch. It does not run IsaacLab,
render images, build a LeRobot dataset, train, delete, archive, move, or touch
B200.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT_DIR = LOG_DIR / "cube10cm_top_view_labelaware_manifest_0_999_d241"

EXPECTED_COUNTS = {
    "debug_camera_anchor": 50,
    "clean_prior_candidate": 650,
    "transition_mixed_probe": 200,
    "overshoot_eval_candidate": 100,
}

RENDERER_REQUIRED_FIELDS = [
    "episode_index",
    "split_candidate",
    "cube_x_m",
    "cube_y_m",
    "seed",
    "sampling_rule",
    "sampling_cell_id",
    "source_decision",
    "requires_posthoc_label_validation",
]

FIELDS = [
    "episode_index",
    "split_candidate",
    "intended_sampling_bucket",
    "intended_role",
    "cube_x_m",
    "cube_y_m",
    "seed",
    "sampling_rule",
    "sampling_cell_id",
    "source_decision",
    "requires_posthoc_label_validation",
    "camera_coverage_required",
    "expected_postrender_labels",
    "label_policy",
]

FORBIDDEN_FINAL_LABEL_FIELDS = {
    "label_useful_clean_numeric",
    "label_overshoot_numeric",
    "label_camera_contract_numeric",
    "label_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2410)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * idx for idx in range(count)]


def row(
    *,
    episode_index: int,
    bucket: str,
    intended_role: str,
    cube_x_m: float,
    cube_y_m: float,
    seed: int,
    sampling_rule: str,
    sampling_cell_id: str,
    source_decision: str,
    expected_postrender_labels: str,
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "split_candidate": bucket,
        "intended_sampling_bucket": bucket,
        "intended_role": intended_role,
        "cube_x_m": round(float(cube_x_m), 6),
        "cube_y_m": round(float(cube_y_m), 6),
        "seed": int(seed),
        "sampling_rule": sampling_rule,
        "sampling_cell_id": sampling_cell_id,
        "source_decision": source_decision,
        "requires_posthoc_label_validation": True,
        "camera_coverage_required": True,
        "expected_postrender_labels": expected_postrender_labels,
        "label_policy": "post_render_numeric_validation_only",
    }


def build_debug_rows(start_episode: int, seed_base: int) -> list[dict[str, Any]]:
    anchors = [
        (0.24, 0.00),
        (0.14, -0.10),
        (0.14, 0.10),
        (0.34, -0.10),
        (0.34, 0.10),
        (0.09, 0.15),
        (0.14, 0.15),
        (0.24, 0.15),
        (0.34, 0.15),
        (0.39, 0.15),
    ]
    rows: list[dict[str, Any]] = []
    for repeat_idx in range(5):
        for anchor_idx, (x_m, y_m) in enumerate(anchors):
            episode_index = start_episode + len(rows)
            rows.append(
                row(
                    episode_index=episode_index,
                    bucket="debug_camera_anchor",
                    intended_role="camera_regression_and_meeting_png_anchor",
                    cube_x_m=x_m,
                    cube_y_m=y_m,
                    seed=seed_base + episode_index,
                    sampling_rule="repeated_debug_anchor_camera_contract_regression",
                    sampling_cell_id=f"debug_anchor_r{repeat_idx:02d}_a{anchor_idx:02d}",
                    source_decision="D233_D239_D240",
                    expected_postrender_labels="requires_validation;camera_anchor_not_final_label",
                )
            )
    return rows


def build_clean_prior_rows(start_episode: int, seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_values = linspace(0.14, 0.34, 25)
    y_values = linspace(-0.10, 0.025, 26)
    for y_idx, y_m in enumerate(y_values):
        for x_idx, x_m in enumerate(x_values):
            episode_index = start_episode + len(rows)
            rows.append(
                row(
                    episode_index=episode_index,
                    bucket="clean_prior_candidate",
                    intended_role="train_positive_candidate_after_validation",
                    cube_x_m=x_m,
                    cube_y_m=y_m,
                    seed=seed_base + episode_index,
                    sampling_rule="d241_clean_prior_y_low_negative_and_center_stratified_25x26",
                    sampling_cell_id=f"clean_prior_x{x_idx:02d}_y{y_idx:02d}",
                    source_decision="D240_LABEL_AWARE_FROM_D241_LABELS",
                    expected_postrender_labels="requires_validation;expected_clean_prior_not_final_label",
                )
            )
    return rows


def build_transition_rows(start_episode: int, seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_values = linspace(0.14, 0.34, 20)
    y_values = linspace(0.025, 0.125, 10)
    for y_idx, y_m in enumerate(y_values):
        for x_idx, x_m in enumerate(x_values):
            episode_index = start_episode + len(rows)
            rows.append(
                row(
                    episode_index=episode_index,
                    bucket="transition_mixed_probe",
                    intended_role="mixed_clean_overshoot_boundary_probe_after_validation",
                    cube_x_m=x_m,
                    cube_y_m=y_m,
                    seed=seed_base + episode_index,
                    sampling_rule="d241_transition_positive_mid_y_stratified_20x10",
                    sampling_cell_id=f"transition_x{x_idx:02d}_y{y_idx:02d}",
                    source_decision="D240_LABEL_AWARE_FROM_D241_LABELS",
                    expected_postrender_labels="requires_validation;expected_mixed_clean_or_overshoot_not_final_label",
                )
            )
    return rows


def build_overshoot_eval_rows(start_episode: int, seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x_values = linspace(0.09, 0.39, 20)
    y_values = linspace(0.125, 0.15, 5)
    for y_idx, y_m in enumerate(y_values):
        for x_idx, x_m in enumerate(x_values):
            episode_index = start_episode + len(rows)
            rows.append(
                row(
                    episode_index=episode_index,
                    bucket="overshoot_eval_candidate",
                    intended_role="camera_passing_negative_or_overshoot_eval_after_validation",
                    cube_x_m=x_m,
                    cube_y_m=y_m,
                    seed=seed_base + episode_index,
                    sampling_rule="d241_high_y_camera_covered_overshoot_eval_stratified_20x5",
                    sampling_cell_id=f"overshoot_eval_x{x_idx:02d}_y{y_idx:02d}",
                    source_decision="D240_LABEL_AWARE_FROM_D241_BOUNDARY_LABELS",
                    expected_postrender_labels="requires_validation;expected_overshoot_eval_not_final_label",
                )
            )
    return rows


def build_manifest(seed_base: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    builders = [
        build_debug_rows,
        build_clean_prior_rows,
        build_transition_rows,
        build_overshoot_eval_rows,
    ]
    for builder in builders:
        rows.extend(builder(len(rows), seed_base))
    return rows


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for manifest_row in rows:
        key = str(manifest_row[field])
        counts[key] = counts.get(key, 0) + 1
    return counts


def range_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"min": min(values), "max": max(values)}


def validate_manifest(rows: list[dict[str, Any]], seed_base: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "This is a manifest-only artifact. It is not a rendered dataset.",
        "Current cube10cm_top_view_visual_chunk_render.py is scoped to exactly 100 episodes; 0-999 render requires a separately approved renderer update or new renderer.",
        "Expected labels are intent hints only. Final labels must come from post-render numeric validation.",
    ]

    if len(rows) != 1000:
        errors.append(f"expected 1000 rows, got {len(rows)}")
    actual_ids = [int(row["episode_index"]) for row in rows]
    if actual_ids != list(range(len(rows))):
        errors.append("episode_index is not exactly contiguous 0..999")

    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        errors.append("manifest seeds are not unique")
    if seeds and min(seeds) != seed_base:
        errors.append(f"minimum seed {min(seeds)} != seed base {seed_base}")

    forbidden_present = sorted(FORBIDDEN_FINAL_LABEL_FIELDS.intersection(rows[0].keys() if rows else set()))
    if forbidden_present:
        errors.append(f"final label fields must not be present pre-render: {forbidden_present}")

    for idx, manifest_row in enumerate(rows):
        missing = [field for field in FIELDS if field not in manifest_row]
        if missing:
            errors.append(f"row {idx} missing fields {missing}")
        renderer_missing = [field for field in RENDERER_REQUIRED_FIELDS if field not in manifest_row]
        if renderer_missing:
            errors.append(f"row {idx} missing renderer-required fields {renderer_missing}")
        if manifest_row.get("requires_posthoc_label_validation") is not True:
            errors.append(f"row {idx} must require posthoc label validation")
        if manifest_row.get("split_candidate") != manifest_row.get("intended_sampling_bucket"):
            errors.append(f"row {idx} split_candidate must mirror intended_sampling_bucket")
        if any(field in manifest_row for field in FORBIDDEN_FINAL_LABEL_FIELDS):
            errors.append(f"row {idx} contains pre-render final label fields")
        if len(errors) >= 20:
            break

    bucket_counts = count_by(rows, "intended_sampling_bucket")
    if bucket_counts != EXPECTED_COUNTS:
        errors.append(f"bucket counts mismatch: {bucket_counts} != {EXPECTED_COUNTS}")

    return {
        "artifact": "cube10cm_top_view_labelaware_manifest_0_999_d241",
        "runtime": "NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING",
        "source_policy": "D240_LABEL_AWARE_DESIGN_FROM_D241_RENDER_LABELS",
        "rows": len(rows),
        "episode_index_range": [actual_ids[0], actual_ids[-1]] if actual_ids else [],
        "seed_base": seed_base,
        "seed_unique": len(seeds) == len(set(seeds)),
        "bucket_counts": bucket_counts,
        "expected_bucket_counts": EXPECTED_COUNTS,
        "x_range_m": range_summary(rows, "cube_x_m") if rows else {},
        "y_range_m": range_summary(rows, "cube_y_m") if rows else {},
        "all_requires_posthoc_label_validation": all(
            row.get("requires_posthoc_label_validation") is True for row in rows
        ),
        "renderer_required_fields": RENDERER_REQUIRED_FIELDS,
        "fields": FIELDS,
        "forbidden_final_label_fields_present": bool(forbidden_present),
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "episode_manifest.csv",
        "json": out_dir / "episode_manifest.json",
        "summary": out_dir / "manifest_summary.json",
    }
    if not args.force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"manifest outputs already exist: {existing}")

    with paths["csv"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    summary = dict(summary)
    summary.update({f"{key}_path": str(path) for key, path in paths.items()})
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_manifest(int(args.seed))
    summary = validate_manifest(rows, int(args.seed))
    write_outputs(args, rows, summary)
    print(
        "[cube10cm-labelaware-manifest-0-999] done "
        f"status={summary['status']} rows={summary['rows']} buckets={summary['bucket_counts']}",
        flush=True,
    )
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
