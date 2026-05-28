"""Audit a teacher-filtered DiffIK state-action dataset artifact."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(float(r[key])) for r in rows) / len(rows) if rows else 0.0


def finite_rate(rows: list[dict[str, str]], columns: list[str]) -> float:
    total = len(rows) * len(columns)
    if total == 0:
        return 0.0
    ok = 0
    for row in rows:
        for col in columns:
            try:
                ok += int(math.isfinite(float(row[col])))
            except (KeyError, ValueError):
                pass
    return ok / total


def direction_key(row: dict[str, str]) -> tuple[int, int]:
    return (int(round(float(row["push_dx"]))), int(round(float(row["push_dy"]))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=Path, required=True)
    ap.add_argument("--manifest_json", type=Path, required=True)
    ap.add_argument("--min_rows", type=int, default=30000)
    ap.add_argument("--min_envs", type=int, default=256)
    ap.add_argument("--min_per_direction", type=int, default=50)
    ap.add_argument("--min_per_posx_bucket", type=int, default=8)
    ap.add_argument("--min_frames_per_env", type=int, default=100)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.dataset_csv.open(newline="")))
    manifest = json.loads(args.manifest_json.read_text())
    fields = set(rows[0].keys()) if rows else set()
    feature_columns = list(manifest.get("feature_columns", []))
    target_columns = list(manifest.get("target_columns", []))
    label_columns = list(manifest.get("label_columns", []))
    required_columns = {
        "dataset_name",
        "split",
        "trajectory_id",
        "source_env_id",
        "frame",
        "step",
        "trajectory_variant",
        *feature_columns,
        *target_columns,
        *label_columns,
    }
    schema_ok = bool(rows) and required_columns <= fields
    row_count_ok = len(rows) == int(manifest.get("rows", -1))

    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[int(float(row["trajectory_id"]))].append(row)
    frame_counts = [len(v) for v in by_env.values()]
    frame_count_ok = bool(frame_counts) and min(frame_counts) >= int(args.min_frames_per_env)

    split_by_env: dict[int, set[str]] = defaultdict(set)
    split_rows = defaultdict(int)
    split_envs: dict[str, set[int]] = defaultdict(set)
    dir_envs: dict[tuple[int, int], set[int]] = defaultdict(set)
    posx_bucket_envs: dict[str, set[int]] = defaultdict(set)
    split_dir_envs: dict[tuple[str, tuple[int, int]], set[int]] = defaultdict(set)
    split_posx_bucket_envs: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        env_id = int(float(row["trajectory_id"]))
        split = row["split"]
        direction = direction_key(row)
        split_by_env[env_id].add(split)
        split_rows[split] += 1
        split_envs[split].add(env_id)
        dir_envs[direction].add(env_id)
        split_dir_envs[(split, direction)].add(env_id)
        if direction == (1, 0) and "posx_x_bucket" in row:
            bucket = row["posx_x_bucket"]
            posx_bucket_envs[bucket].add(env_id)
            split_posx_bucket_envs[(split, bucket)].add(env_id)
    split_leakage_ok = all(len(splits) == 1 for splits in split_by_env.values())
    split_ok = {"train", "val", "test"} <= set(split_rows)

    final_rows = [max(env_rows, key=lambda r: int(float(r["step"]))) for env_rows in by_env.values()]
    teacher_filter_ok = (
        rate(final_rows, "accepted_teacher_trajectory") == 1.0
        and rate(final_rows, "final_controlled_push") == 1.0
        and rate(final_rows, "final_impact_outlier") == 0.0
        and rate(final_rows, "final_low_motion") == 0.0
        and rate(final_rows, "final_success_marker") == 1.0
    )
    expected_dirs = {(-1, 0), (0, -1), (0, 1), (1, 0)}
    dir_counts = {str(k): len(v) for k, v in sorted(dir_envs.items())}
    direction_ok = expected_dirs <= set(dir_envs) and min((len(dir_envs[d]) for d in expected_dirs), default=0) >= int(
        args.min_per_direction
    )
    train_direction_ok = all(len(split_dir_envs[("train", d)]) > 0 for d in expected_dirs)
    val_direction_ok = all(len(split_dir_envs[("val", d)]) > 0 for d in expected_dirs)
    test_direction_ok = all(len(split_dir_envs[("test", d)]) > 0 for d in expected_dirs)
    expected_buckets = {"low_x", "mid_x", "high_x"}
    bucket_counts = {k: len(posx_bucket_envs[k]) for k in sorted(expected_buckets)}
    if manifest.get("balance_mode") == "direction_posx_bucket":
        bucket_ok = (
            expected_buckets <= set(posx_bucket_envs)
            and min(bucket_counts.values(), default=0) >= int(args.min_per_posx_bucket)
            and len(set(bucket_counts.values())) == 1
        )
        split_bucket_ok = all(
            len(split_posx_bucket_envs[(split, bucket)]) > 0
            for split in ("train", "val", "test")
            for bucket in expected_buckets
        )
    else:
        bucket_ok = True
        split_bucket_ok = True

    source = manifest.get("source_summary", {})
    mechanism_ok = (
        manifest.get("artifact_type") == "diffik_state_action_dataset_v2"
        and source.get("controller") == "IsaacLab_DifferentialIKController"
        and source.get("training") is False
        and source.get("grasp_attach") is False
        and source.get("rollout_object_posewrite") is False
        and int(source.get("posewrite_calls_during_rollout", -1)) == 0
    )
    finite_ok = finite_rate(rows, feature_columns + target_columns) == 1.0
    size_ok = len(rows) >= int(args.min_rows) and len(by_env) >= int(args.min_envs)
    manifest_candidate_ok = manifest.get("full_dataset_candidate") is True

    full_dataset_ready = all(
        [
            schema_ok,
            row_count_ok,
            frame_count_ok,
            split_leakage_ok,
            split_ok,
            teacher_filter_ok,
            direction_ok,
            train_direction_ok,
            val_direction_ok,
            test_direction_ok,
            bucket_ok,
            split_bucket_ok,
            mechanism_ok,
            finite_ok,
            size_ok,
            manifest_candidate_ok,
        ]
    )
    verdict = "PASS_FULL_STATE_ACTION_DATASET_V2" if full_dataset_ready else "FAIL_FULL_STATE_ACTION_DATASET_V2"

    print(
        "diffik_dataset_v2_audit line1 "
        f"rows={len(rows)} manifest_rows={manifest.get('rows')} row_count_ok={row_count_ok} "
        f"env_count={len(by_env)} frames_per_env_min={min(frame_counts) if frame_counts else 0} "
        f"frames_per_env_max={max(frame_counts) if frame_counts else 0}"
    )
    print(
        "diffik_dataset_v2_audit line2 "
        f"schema_ok={schema_ok} finite_ok={finite_ok} feature_count={len(feature_columns)} "
        f"target_count={len(target_columns)} label_count={len(label_columns)}"
    )
    print(
        "diffik_dataset_v2_audit line3 "
        f"split_ok={split_ok} split_leakage_ok={split_leakage_ok} "
        f"split_env_counts={{'train': {len(split_envs['train'])}, 'val': {len(split_envs['val'])}, 'test': {len(split_envs['test'])}}} "
        f"split_row_counts={dict(sorted(split_rows.items()))}"
    )
    print(
        "diffik_dataset_v2_audit line4 "
        f"direction_ok={direction_ok} train_direction_ok={train_direction_ok} "
        f"val_direction_ok={val_direction_ok} test_direction_ok={test_direction_ok} "
        f"direction_env_counts={dir_counts}"
    )
    print(
        "diffik_dataset_v2_audit line5 "
        f"final_rates controlled={rate(final_rows, 'final_controlled_push'):.9f} "
        f"impact={rate(final_rows, 'final_impact_outlier'):.9f} "
        f"low_motion={rate(final_rows, 'final_low_motion'):.9f} "
        f"success_marker={rate(final_rows, 'final_success_marker'):.9f} "
        f"teacher_filter_ok={teacher_filter_ok}"
    )
    print(
        "diffik_dataset_v2_audit line6 "
        f"mechanism_ok={mechanism_ok} source_training={source.get('training')} "
        f"source_dataset_generation={source.get('dataset_generation')} "
        f"source_posewrite_calls={source.get('posewrite_calls_during_rollout')} "
        f"size_ok={size_ok} manifest_candidate_ok={manifest_candidate_ok}"
    )
    print(
        "diffik_dataset_v2_audit line7 "
        f"balance_mode={manifest.get('balance_mode')} bucket_ok={bucket_ok} "
        f"split_bucket_ok={split_bucket_ok} posx_bucket_env_counts={bucket_counts}"
    )
    print(
        "diffik_dataset_v2_audit line8 "
        f"verdict={verdict} full_dataset_ready={'YES' if full_dataset_ready else 'NO'} "
        "learned_policy=NO"
    )
    return 0 if full_dataset_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
