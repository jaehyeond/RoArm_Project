"""Build a teacher-filtered state-action dataset from a DiffIK trace.

This turns raw step-level trace rows into a split, auditable BC dataset. It does
not train a policy and does not relabel failed teacher trajectories as success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


RAW_OBSERVATION_COLUMNS = [
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "tcp_x_m",
    "tcp_y_m",
    "tcp_z_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "env_origin_x_m",
    "env_origin_y_m",
    "env_origin_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
]
TARGET_COLUMNS = [
    "joint_delta_0_rad",
    "joint_delta_1_rad",
    "joint_delta_2_rad",
    "joint_delta_3_rad",
    "joint_delta_4_rad",
]
FEATURE_COLUMNS = [
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_local_x_m",
    "cube_local_y_m",
    "cube_local_z_m",
    "tcp_local_x_m",
    "tcp_local_y_m",
    "tcp_local_z_m",
    "target_local_x_m",
    "target_local_y_m",
    "target_local_z_m",
    "tcp_to_cube_x_m",
    "tcp_to_cube_y_m",
    "tcp_to_cube_z_m",
    "target_to_tcp_x_m",
    "target_to_tcp_y_m",
    "target_to_tcp_z_m",
    "target_to_cube_x_m",
    "target_to_cube_y_m",
    "target_to_cube_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
]
LABEL_COLUMNS = [
    "disp_along_push_m",
    "disp_xy_m",
    "lateral_abs_m",
    "target_xy_dist_m",
    "tcp_cube_dist_m",
    "cube_speed_mps",
    "tip_angle_deg",
    "controlled_push",
    "impact_outlier",
    "low_motion",
    "success_marker",
    "final_controlled_push",
    "final_impact_outlier",
    "final_low_motion",
    "final_success_marker",
    "accepted_teacher_trajectory",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int_bool(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def direction_key(row: dict[str, str]) -> tuple[int, int]:
    return (int(round(as_float(row, "push_dx"))), int(round(as_float(row, "push_dy"))))


def posx_bucket_from_initial(env_rows: list[dict[str, str]], edge0: float, edge1: float) -> str:
    first = env_rows[0]
    cube_x0 = as_float(first, "cube_x_m") - as_float(first, "env_origin_x_m")
    if cube_x0 < edge0:
        return "low_x"
    if cube_x0 < edge1:
        return "mid_x"
    return "high_x"


def split_envs(env_ids: list[int], train_frac: float, val_frac: float, rng: random.Random) -> dict[int, str]:
    shuffled = list(env_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        return {env_id: "train" for env_id in shuffled}
    test_n = max(1, int(round(n * max(0.0, 1.0 - train_frac - val_frac))))
    val_n = max(1, int(round(n * val_frac)))
    if test_n + val_n >= n:
        test_n = 1
        val_n = 1
    train_n = n - val_n - test_n
    out: dict[int, str] = {}
    for env_id in shuffled[:train_n]:
        out[env_id] = "train"
    for env_id in shuffled[train_n : train_n + val_n]:
        out[env_id] = "val"
    for env_id in shuffled[train_n + val_n :]:
        out[env_id] = "test"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--name", default="diffik_state_action_dataset_v2")
    ap.add_argument("--seed", type=int, default=779)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--min_frames_per_env", type=int, default=100)
    ap.add_argument("--min_selected_envs", type=int, default=256)
    ap.add_argument("--min_selected_per_direction", type=int, default=50)
    ap.add_argument("--max_per_direction", type=int, default=0)
    ap.add_argument("--balance_mode", choices=("direction", "direction_posx_bucket"), default="direction")
    ap.add_argument("--posx_bucket_edges", type=float, nargs=2, default=(0.257, 0.308))
    ap.add_argument("--min_selected_per_posx_bucket", type=int, default=8)
    ap.add_argument("--max_per_posx_bucket", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.trace_csv.open(newline="")))
    if not rows:
        raise ValueError(f"empty trace csv: {args.trace_csv}")
    missing = [c for c in RAW_OBSERVATION_COLUMNS + TARGET_COLUMNS + LABEL_COLUMNS[:-5] if c not in rows[0]]
    if missing:
        raise ValueError(f"trace missing required columns: {missing}")
    summary = json.loads(args.summary_json.read_text())

    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[int(float(row["env_id"]))].append(row)
    for env_rows in by_env.values():
        env_rows.sort(key=lambda r: int(float(r["step"])))

    final_by_env = {env_id: env_rows[-1] for env_id, env_rows in by_env.items() if env_rows}
    eligible_by_dir: dict[tuple[int, int], list[int]] = defaultdict(list)
    reject_reasons = defaultdict(int)
    for env_id, env_rows in by_env.items():
        final = env_rows[-1]
        if len(env_rows) < int(args.min_frames_per_env):
            reject_reasons["short_trajectory"] += 1
            continue
        if as_int_bool(final, "controlled_push") != 1:
            reject_reasons["not_controlled"] += 1
            continue
        if as_int_bool(final, "impact_outlier") != 0:
            reject_reasons["impact"] += 1
            continue
        if as_int_bool(final, "low_motion") != 0:
            reject_reasons["low_motion"] += 1
            continue
        if as_int_bool(final, "success_marker") != 1:
            reject_reasons["not_success"] += 1
            continue
        eligible_by_dir[direction_key(final)].append(env_id)

    rng = random.Random(int(args.seed))
    expected_dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    selected_envs: list[int] = []
    selected_by_dir: dict[str, int] = {}
    selected_by_posx_bucket: dict[str, int] = {}
    eligible_by_posx_bucket: dict[str, list[int]] = {"low_x": [], "mid_x": [], "high_x": []}
    posx_bucket_by_env: dict[int, str] = {}
    edge0, edge1 = float(args.posx_bucket_edges[0]), float(args.posx_bucket_edges[1])
    for env_id in eligible_by_dir[(1, 0)]:
        bucket = posx_bucket_from_initial(by_env[env_id], edge0, edge1)
        eligible_by_posx_bucket[bucket].append(env_id)
        posx_bucket_by_env[env_id] = bucket

    split_by_env: dict[int, str] = {}
    if args.balance_mode == "direction":
        min_available = min((len(eligible_by_dir[d]) for d in expected_dirs), default=0)
        if int(args.max_per_direction) > 0:
            per_dir_take = min(min_available, int(args.max_per_direction))
        else:
            per_dir_take = min_available
        for direction in expected_dirs:
            env_ids = sorted(eligible_by_dir[direction])
            rng.shuffle(env_ids)
            chosen = sorted(env_ids[:per_dir_take])
            selected_envs.extend(chosen)
            selected_by_dir[str(direction)] = len(chosen)
            split_by_env.update(split_envs(chosen, float(args.train_frac), float(args.val_frac), rng))
    else:
        min_posx_bucket = min((len(v) for v in eligible_by_posx_bucket.values()), default=0)
        if int(args.max_per_posx_bucket) > 0:
            posx_per_bucket = min(min_posx_bucket, int(args.max_per_posx_bucket))
        else:
            posx_per_bucket = min_posx_bucket
        posx_chosen: list[int] = []
        for bucket in ["low_x", "mid_x", "high_x"]:
            env_ids = sorted(eligible_by_posx_bucket[bucket])
            rng.shuffle(env_ids)
            chosen = sorted(env_ids[:posx_per_bucket])
            posx_chosen.extend(chosen)
            selected_by_posx_bucket[bucket] = len(chosen)
        selected_envs.extend(posx_chosen)
        selected_by_dir[str((1, 0))] = len(posx_chosen)
        split_by_env.update(split_envs(sorted(posx_chosen), float(args.train_frac), float(args.val_frac), rng))

        other_take = len(posx_chosen)
        if int(args.max_per_direction) > 0:
            other_take = min(other_take, int(args.max_per_direction))
        for direction in [(-1, 0), (0, -1), (0, 1)]:
            env_ids = sorted(eligible_by_dir[direction])
            rng.shuffle(env_ids)
            chosen = sorted(env_ids[:other_take])
            selected_envs.extend(chosen)
            selected_by_dir[str(direction)] = len(chosen)
            split_by_env.update(split_envs(chosen, float(args.train_frac), float(args.val_frac), rng))

    selected_set = set(selected_envs)
    out_rows: list[dict[str, str | int | float]] = []
    for env_id in sorted(selected_set):
        final = final_by_env[env_id]
        for row in by_env[env_id]:
            origin = (
                as_float(row, "env_origin_x_m"),
                as_float(row, "env_origin_y_m"),
                as_float(row, "env_origin_z_m"),
            )
            cube = (
                as_float(row, "cube_x_m") - origin[0],
                as_float(row, "cube_y_m") - origin[1],
                as_float(row, "cube_z_m") - origin[2],
            )
            tcp = (
                as_float(row, "tcp_x_m") - origin[0],
                as_float(row, "tcp_y_m") - origin[1],
                as_float(row, "tcp_z_m") - origin[2],
            )
            target = (
                as_float(row, "target_x_m") - origin[0],
                as_float(row, "target_y_m") - origin[1],
                as_float(row, "target_z_m") - origin[2],
            )
            out: dict[str, str | int | float] = {
                "dataset_name": args.name,
                "split": split_by_env[env_id],
                "trajectory_id": env_id,
                "source_env_id": env_id,
                "posx_x_bucket": posx_bucket_by_env.get(env_id, "not_posx"),
                "frame": int(float(row["frame"])),
                "step": int(float(row["step"])),
                "trajectory_variant": row["trajectory_variant"],
                "push_dx": as_float(row, "push_dx"),
                "push_dy": as_float(row, "push_dy"),
                "phase_alpha": as_float(row, "phase_alpha"),
                "cube_local_x_m": cube[0],
                "cube_local_y_m": cube[1],
                "cube_local_z_m": cube[2],
                "tcp_local_x_m": tcp[0],
                "tcp_local_y_m": tcp[1],
                "tcp_local_z_m": tcp[2],
                "target_local_x_m": target[0],
                "target_local_y_m": target[1],
                "target_local_z_m": target[2],
                "tcp_to_cube_x_m": cube[0] - tcp[0],
                "tcp_to_cube_y_m": cube[1] - tcp[1],
                "tcp_to_cube_z_m": cube[2] - tcp[2],
                "target_to_tcp_x_m": target[0] - tcp[0],
                "target_to_tcp_y_m": target[1] - tcp[1],
                "target_to_tcp_z_m": target[2] - tcp[2],
                "target_to_cube_x_m": target[0] - cube[0],
                "target_to_cube_y_m": target[1] - cube[1],
                "target_to_cube_z_m": target[2] - cube[2],
                "final_controlled_push": as_int_bool(final, "controlled_push"),
                "final_impact_outlier": as_int_bool(final, "impact_outlier"),
                "final_low_motion": as_int_bool(final, "low_motion"),
                "final_success_marker": as_int_bool(final, "success_marker"),
                "accepted_teacher_trajectory": 1,
            }
            for col in [
                "arm_joint_0_rad",
                "arm_joint_1_rad",
                "arm_joint_2_rad",
                "arm_joint_3_rad",
                "arm_joint_4_rad",
                "gripper_joint_rad",
                *TARGET_COLUMNS,
                "disp_along_push_m",
                "disp_xy_m",
                "lateral_abs_m",
                "target_xy_dist_m",
                "tcp_cube_dist_m",
                "cube_speed_mps",
                "tip_angle_deg",
                "controlled_push",
                "impact_outlier",
                "low_motion",
                "success_marker",
                "v31_lowx_applied",
            ]:
                if col in row:
                    value: str | int | float = row[col]
                    if col.endswith("_push") or col.endswith("_outlier") or col.endswith("_motion") or col.endswith("_marker"):
                        value = as_int_bool(row, col)
                    elif col == "v31_lowx_applied":
                        value = as_int_bool(row, col)
                    else:
                        value = as_float(row, col)
                    out[col] = value
            out_rows.append(out)

    split_env_counts = defaultdict(int)
    split_row_counts = defaultdict(int)
    for env_id in selected_envs:
        split_env_counts[split_by_env[env_id]] += 1
    for row in out_rows:
        split_row_counts[str(row["split"])] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = args.out_dir / f"{args.name}.csv"
    manifest_json = args.out_dir / f"{args.name}_manifest.json"
    fieldnames = [
        "dataset_name",
        "split",
        "trajectory_id",
        "source_env_id",
        "posx_x_bucket",
        "frame",
        "step",
        "trajectory_variant",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
        *LABEL_COLUMNS,
        "v31_lowx_applied",
    ]
    with dataset_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    frames_per_env = [len(v) for v in by_env.values()]
    per_dir_counts = [selected_by_dir.get(str(d), 0) for d in expected_dirs]
    if args.balance_mode == "direction":
        full_dataset_candidate = (
            len(selected_envs) >= int(args.min_selected_envs)
            and min(per_dir_counts, default=0) >= int(args.min_selected_per_direction)
            and len(set(per_dir_counts)) == 1
            and len(out_rows) > 0
        )
    else:
        posx_bucket_counts = [selected_by_posx_bucket.get(k, 0) for k in ["low_x", "mid_x", "high_x"]]
        full_dataset_candidate = (
            len(selected_envs) >= int(args.min_selected_envs)
            and min(per_dir_counts, default=0) >= int(args.min_selected_per_direction)
            and len(set(per_dir_counts)) == 1
            and min(posx_bucket_counts, default=0) >= int(args.min_selected_per_posx_bucket)
            and len(set(posx_bucket_counts)) == 1
            and len(out_rows) > 0
        )
    manifest = {
        "artifact_type": "diffik_state_action_dataset_v2",
        "dataset_name": args.name,
        "dataset_csv": str(dataset_csv),
        "dataset_csv_md5": md5(dataset_csv),
        "source_trace_csv": str(args.trace_csv),
        "source_trace_csv_md5": md5(args.trace_csv),
        "source_summary_json": str(args.summary_json),
        "source_summary_json_md5": md5(args.summary_json),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "rows": len(out_rows),
        "source_trace_rows": len(rows),
        "source_env_count": len(by_env),
        "selected_env_count": len(selected_envs),
        "eligible_env_count": sum(len(v) for v in eligible_by_dir.values()),
        "selected_per_direction": selected_by_dir,
        "eligible_per_direction": {str(k): len(v) for k, v in sorted(eligible_by_dir.items())},
        "balance_mode": args.balance_mode,
        "posx_bucket_edges": [edge0, edge1],
        "selected_per_posx_bucket": selected_by_posx_bucket,
        "eligible_per_posx_bucket": {k: len(v) for k, v in sorted(eligible_by_posx_bucket.items())},
        "rejected_env_reasons": dict(sorted(reject_reasons.items())),
        "frames_per_env_min": min(frames_per_env) if frames_per_env else 0,
        "frames_per_env_max": max(frames_per_env) if frames_per_env else 0,
        "split_env_counts": dict(sorted(split_env_counts.items())),
        "split_row_counts": dict(sorted(split_row_counts.items())),
        "teacher_filter": (
            "final_controlled=1 final_impact=0 final_low_motion=0 final_success=1 "
            f"balanced_by={args.balance_mode}"
        ),
        "min_selected_envs": int(args.min_selected_envs),
        "min_selected_per_direction": int(args.min_selected_per_direction),
        "min_selected_per_posx_bucket": int(args.min_selected_per_posx_bucket),
        "full_dataset_candidate": full_dataset_candidate,
        "learned_policy": False,
        "rollout_validated": False,
        "source_summary": {
            "controller": summary.get("controller"),
            "ik_method": summary.get("ik_method"),
            "training": summary.get("training"),
            "dataset_generation": summary.get("dataset_generation"),
            "grasp_attach": summary.get("grasp_attach"),
            "rollout_object_posewrite": summary.get("rollout_object_posewrite"),
            "posewrite_calls_during_rollout": summary.get("posewrite_calls_during_rollout"),
            "trajectory_variant": summary.get("trajectory_variant"),
            "trace_frame_count": summary.get("trace_frame_count"),
            "num_envs": summary.get("num_envs"),
            "trials": summary.get("trials"),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    verdict = "BUILT_DIFFIK_STATE_ACTION_DATASET_V2" if out_rows else "FAIL_EMPTY_DATASET_V2"
    print(
        "diffik_dataset_build line1 "
        f"source_trace_rows={len(rows)} source_env_count={len(by_env)} "
        f"frames_per_env_min={manifest['frames_per_env_min']} frames_per_env_max={manifest['frames_per_env_max']}"
    )
    print(
        "diffik_dataset_build line2 "
        f"eligible_env_count={manifest['eligible_env_count']} selected_env_count={len(selected_envs)} "
        f"selected_rows={len(out_rows)} selected_per_direction={selected_by_dir} "
        f"selected_per_posx_bucket={selected_by_posx_bucket}"
    )
    print(
        "diffik_dataset_build line3 "
        f"split_env_counts={dict(sorted(split_env_counts.items()))} "
        f"split_row_counts={dict(sorted(split_row_counts.items()))}"
    )
    print(
        "diffik_dataset_build line4 "
        f"source_controller={summary.get('controller')} source_training={summary.get('training')} "
        f"source_dataset_generation={summary.get('dataset_generation')} "
        f"source_posewrite_calls={summary.get('posewrite_calls_during_rollout')}"
    )
    print(
        "diffik_dataset_build line5 "
        f"dataset_csv={dataset_csv} manifest_json={manifest_json} dataset_md5={manifest['dataset_csv_md5']}"
    )
    print(
        "diffik_dataset_build line6 "
        f"verdict={verdict} full_dataset_candidate={'YES' if full_dataset_candidate else 'NO'} "
        "learned_policy=NO"
    )
    return 0 if out_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
