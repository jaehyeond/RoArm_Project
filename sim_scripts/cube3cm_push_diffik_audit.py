"""Posthoc audit for the IsaacLab built-in Differential IK cube-push probe."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(float(r[key])) for r in rows) / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary_json", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    summary_path = Path(args.summary_json)
    rows = list(csv.DictReader(csv_path.open()))
    summary = json.loads(summary_path.read_text())

    row_count_match = len(rows) == int(summary.get("trials", -1))
    mechanism_ok = (
        summary.get("controller") == "IsaacLab_DifferentialIKController"
        and summary.get("ik_method") == "dls"
        and summary.get("command_type") == "position"
        and summary.get("local_roarm_ik_dls_control_loop") is False
        and summary.get("training") is False
        and summary.get("dataset_generation") is False
        and summary.get("grasp_attach") is False
        and summary.get("rollout_object_posewrite") is False
        and int(summary.get("posewrite_calls_during_rollout", -1)) == 0
        and summary.get("env_auto_reset_disabled") is True
        and summary.get("env_joint_delta_action_loop_bypassed") is True
    )
    grasp_ok = _rate(rows, "grasped_marker") == 0.0
    controlled_rate = _rate(rows, "controlled_push")
    impact_rate = _rate(rows, "impact_outlier")
    low_motion_rate = _rate(rows, "low_motion")
    success_rate = _rate(rows, "success_marker")
    disp_xy_mean = _mean(rows, "disp_xy_m")
    disp_along_mean = _mean(rows, "disp_along_push_m")
    min_tcp_err_mean = _mean(rows, "min_tcp_target_err_m")
    final_tcp_err_mean = _mean(rows, "final_tcp_target_err_m")
    clip_rate_mean = _mean(rows, "diffik_clip_rate")
    max_speed = max((float(r["max_cube_speed_mps"]) for r in rows), default=0.0)
    max_disp = max((float(r["disp_xy_m"]) for r in rows), default=0.0)
    verdict = "PASS_MECHANISM_DIFFIK_RUNTIME_RAN"
    if not row_count_match or not mechanism_ok or not grasp_ok:
        verdict = "FAIL_MECHANISM_OR_ROWCOUNT"

    print(
        "diffik_audit line1 "
        f"csv_rows={len(rows)} summary_trials={summary.get('trials')} row_count_match={row_count_match} "
        f"controller={summary.get('controller')} ik_method={summary.get('ik_method')} "
        f"local_roarm_ik_dls_control_loop={summary.get('local_roarm_ik_dls_control_loop')}"
    )
    print(
        "diffik_audit line2 "
        f"mechanism_ok={mechanism_ok} training={summary.get('training')} "
        f"dataset_generation={summary.get('dataset_generation')} grasp_attach={summary.get('grasp_attach')} "
        f"rollout_object_posewrite={summary.get('rollout_object_posewrite')} "
        f"posewrite_calls_during_rollout={summary.get('posewrite_calls_during_rollout')} "
        f"env_auto_reset_disabled={summary.get('env_auto_reset_disabled')} "
        f"env_joint_delta_action_loop_bypassed={summary.get('env_joint_delta_action_loop_bypassed')} "
        f"grasp_ok={grasp_ok}"
    )
    print(
        "diffik_audit line3 "
        f"controlled_push_rate={controlled_rate:.9f} impact_outlier_rate={impact_rate:.9f} "
        f"low_motion_rate={low_motion_rate:.9f} success_marker_rate={success_rate:.9f}"
    )
    print(
        "diffik_audit line4 "
        f"disp_along_push_mean_m={disp_along_mean:.9f} disp_xy_mean_m={disp_xy_mean:.9f} "
        f"max_disp_xy_m={max_disp:.9f} max_cube_speed_mps={max_speed:.9f}"
    )
    print(
        "diffik_audit line5 "
        f"min_tcp_target_err_mean_m={min_tcp_err_mean:.9f} "
        f"final_tcp_target_err_mean_m={final_tcp_err_mean:.9f} "
        f"diffik_clip_rate_mean={clip_rate_mean:.9f}"
    )
    print(f"diffik_audit line6 verdict={verdict} learned_policy=NO track_a_grasp_success=NO dataset_ready=NO")

    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
