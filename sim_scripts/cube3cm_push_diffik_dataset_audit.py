"""Audit a step-level DiffIK trace as a small dataset pilot.

This checks schema and basic label distribution only. It does not certify a full
training dataset or teacher policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(float(r[key])) for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.trace_csv.open(newline="")))
    summary = json.loads(args.summary_json.read_text())
    fields = set(rows[0].keys()) if rows else set()

    observation_fields = {
        "cube_x_m",
        "cube_y_m",
        "cube_z_m",
        "cube_qw",
        "cube_qx",
        "cube_qy",
        "cube_qz",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "arm_joint_0_rad",
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
        "gripper_joint_rad",
    }
    action_fields = {
        "joint_target_0_rad",
        "joint_target_1_rad",
        "joint_target_2_rad",
        "joint_target_3_rad",
        "joint_target_4_rad",
        "joint_delta_0_rad",
        "joint_delta_1_rad",
        "joint_delta_2_rad",
        "joint_delta_3_rad",
        "joint_delta_4_rad",
        "gripper_target_rad",
    }
    label_fields = {
        "disp_along_push_m",
        "disp_xy_m",
        "target_xy_dist_m",
        "cube_speed_mps",
        "tip_angle_deg",
        "controlled_push",
        "impact_outlier",
        "low_motion",
        "success_marker",
    }
    metadata_fields = {"frame", "step", "env_id", "trajectory_variant", "push_dx", "push_dy"}

    has_observation = observation_fields <= fields
    has_action = action_fields <= fields
    has_labels = label_fields <= fields
    has_metadata = metadata_fields <= fields

    by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_env[int(row["env_id"])].append(row)
    frame_counts = [len(v) for v in by_env.values()]
    final_rows = [max(v, key=lambda r: int(float(r["step"]))) for v in by_env.values()]

    row_count_match = int(summary.get("trace_frame_count", -1)) == len(rows)
    mechanism_ok = (
        summary.get("controller") == "IsaacLab_DifferentialIKController"
        and summary.get("training") is False
        and summary.get("dataset_generation") is False
        and summary.get("grasp_attach") is False
        and summary.get("rollout_object_posewrite") is False
        and int(summary.get("posewrite_calls_during_rollout", -1)) == 0
    )
    schema_ok = bool(rows) and has_observation and has_action and has_labels and has_metadata
    frame_count_ok = bool(frame_counts) and min(frame_counts) == max(frame_counts)
    verdict = "PASS_TRACE_DATASET_PILOT_SCHEMA"
    if not (schema_ok and frame_count_ok and row_count_match and mechanism_ok):
        verdict = "FAIL_TRACE_DATASET_PILOT_SCHEMA"

    print(
        "diffik_dataset_audit line1 "
        f"trace_rows={len(rows)} summary_trace_frame_count={summary.get('trace_frame_count')} "
        f"row_count_match={row_count_match} env_count={len(by_env)} "
        f"frames_per_env_min={min(frame_counts) if frame_counts else 0} "
        f"frames_per_env_max={max(frame_counts) if frame_counts else 0}"
    )
    print(
        "diffik_dataset_audit line2 "
        f"schema_observation={has_observation} schema_action={has_action} "
        f"schema_labels={has_labels} schema_metadata={has_metadata} frame_count_ok={frame_count_ok}"
    )
    print(
        "diffik_dataset_audit line3 "
        f"mechanism_ok={mechanism_ok} training={summary.get('training')} "
        f"dataset_generation_flag={summary.get('dataset_generation')} "
        f"posewrite_calls={summary.get('posewrite_calls_during_rollout')}"
    )
    print(
        "diffik_dataset_audit line4 final_env_rates "
        f"controlled={rate(final_rows, 'controlled_push'):.9f} "
        f"impact={rate(final_rows, 'impact_outlier'):.9f} "
        f"low_motion={rate(final_rows, 'low_motion'):.9f} "
        f"success_marker={rate(final_rows, 'success_marker'):.9f}"
    )
    print(
        "diffik_dataset_audit line5 "
        f"verdict={verdict} pilot_schema_ready={'YES' if verdict.startswith('PASS') else 'NO'} "
        "full_dataset_ready=NO learned_policy=NO"
    )
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
