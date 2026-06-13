#!/usr/bin/env python3
"""Create the D236 0-99 chunk sampling manifest without rendering.

This utility writes a deterministic episode manifest for the first 100 episode
top-view visual chunk. It does not run IsaacLab, render images, build a LeRobot
dataset, train, delete, archive, move, or touch B200.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT_DIR = LOG_DIR / "cube10cm_top_view_chunk100_manifest_d236"

FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2360)
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
    split_candidate: str,
    cube_x_m: float,
    cube_y_m: float,
    seed: int,
    sampling_rule: str,
    sampling_cell_id: str,
    source_decision: str,
) -> dict[str, object]:
    return {
        "episode_index": int(episode_index),
        "split_candidate": split_candidate,
        "cube_x_m": round(float(cube_x_m), 6),
        "cube_y_m": round(float(cube_y_m), 6),
        "seed": int(seed),
        "sampling_rule": sampling_rule,
        "sampling_cell_id": sampling_cell_id,
        "source_decision": source_decision,
        "requires_posthoc_label_validation": True,
    }


def build_manifest(seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    debug_poses = [
        (0.24, 0.00),
        (0.14, -0.10),
        (0.14, +0.10),
        (0.34, -0.10),
        (0.34, +0.10),
    ]
    for idx, (x_m, y_m) in enumerate(debug_poses):
        rows.append(
            row(
                episode_index=len(rows),
                split_candidate="debug_smoke",
                cube_x_m=x_m,
                cube_y_m=y_m,
                seed=seed + idx,
                sampling_rule="d233_fixed_camera_contract_pose",
                sampling_cell_id=f"debug_smoke_{idx:02d}",
                source_decision="D233",
            )
        )

    x_values = linspace(0.14, 0.34, 13)
    y_values = linspace(-0.10, 0.10, 5)
    for y_idx, y_m in enumerate(y_values):
        for x_idx, x_m in enumerate(x_values):
            rows.append(
                row(
                    episode_index=len(rows),
                    split_candidate="train_success_candidate",
                    cube_x_m=x_m,
                    cube_y_m=y_m,
                    seed=seed + len(rows),
                    sampling_rule="xy10_stratified_13x5_visible_workspace_candidate",
                    sampling_cell_id=f"xy10_grid_x{x_idx:02d}_y{y_idx:02d}",
                    source_decision="D232_D233_D230",
                )
            )

    rng = random.Random(seed + 700)
    for idx in range(15):
        rows.append(
            row(
                episode_index=len(rows),
                split_candidate="eval_failure_candidate",
                cube_x_m=rng.uniform(0.14, 0.34),
                cube_y_m=rng.uniform(-0.10, 0.10),
                seed=seed + len(rows),
                sampling_rule="xy10_randomized_overshoot_diagnostic_candidate",
                sampling_cell_id=f"xy10_random_{idx:02d}",
                source_decision="D230",
            )
        )

    boundary_x = [
        0.09,
        0.12,
        0.14,
        0.15,
        0.16,
        0.1625,
        0.16375,
        0.165,
        0.17,
        0.18,
        0.19,
        0.21,
        0.24,
        0.30,
        0.39,
    ]
    for idx, x_m in enumerate(boundary_x):
        rows.append(
            row(
                episode_index=len(rows),
                split_candidate="eval_boundary_candidate",
                cube_x_m=x_m,
                cube_y_m=0.15,
                seed=seed + len(rows),
                sampling_rule="close_x_high_y_boundary_candidate_camera_coverage_required",
                sampling_cell_id=f"boundary_y015_x{idx:02d}",
                source_decision="D225_D227_D228",
            )
        )

    return rows


def validate_manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"expected 100 rows, got {len(rows)}")
    expected_ids = list(range(100))
    actual_ids = [int(row["episode_index"]) for row in rows]
    if actual_ids != expected_ids:
        errors.append("episode_index is not exactly 0..99")
    for idx, manifest_row in enumerate(rows):
        missing = [field for field in FIELDS if field not in manifest_row]
        if missing:
            errors.append(f"row {idx} missing fields {missing}")
        if manifest_row.get("requires_posthoc_label_validation") is not True:
            errors.append(f"row {idx} must require posthoc label validation")

    counts: dict[str, int] = {}
    for manifest_row in rows:
        split = str(manifest_row["split_candidate"])
        counts[split] = counts.get(split, 0) + 1

    expected_counts = {
        "debug_smoke": 5,
        "train_success_candidate": 65,
        "eval_failure_candidate": 15,
        "eval_boundary_candidate": 15,
    }
    if counts != expected_counts:
        errors.append(f"split counts mismatch: {counts} != {expected_counts}")

    return {
        "artifact": "cube10cm_top_view_chunk100_manifest_d236",
        "runtime": "NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING",
        "rows": len(rows),
        "split_counts": counts,
        "expected_split_counts": expected_counts,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
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
    summary = validate_manifest(rows)
    write_outputs(args, rows, summary)
    print(
        "[cube10cm-chunk100-manifest] done "
        f"status={summary['status']} rows={summary['rows']} splits={summary['split_counts']}",
        flush=True,
    )
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
