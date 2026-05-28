"""Mechanism audit for frozen no-attach cube-push PPO rollouts."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    summary = json.loads(args.summary_json.read_text())
    required = {
        "cube_x0_m",
        "push_dx",
        "push_dy",
        "controlled_push",
        "impact_outlier",
        "low_motion",
        "success_marker",
        "grasped_marker",
        "disp_along_push_m",
    }
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    row_count_match = len(rows) == int(summary.get("trials", -1))
    mechanism_ok = (
        summary.get("controller") == "rsl_rl_PPO_policy"
        and summary.get("learned_policy") is True
        and summary.get("diffik_controller_used") is False
        and summary.get("training") is False
        and summary.get("dataset_generation") is False
        and summary.get("grasp_attach") is False
        and summary.get("rollout_object_posewrite") is False
    )
    grasp_ok = bool(rows) and _rate(rows, "grasped_marker") == 0.0
    controlled_rate = _rate(rows, "controlled_push")
    impact_rate = _rate(rows, "impact_outlier")
    low_motion_rate = _rate(rows, "low_motion")
    success_rate = _rate(rows, "success_marker")
    verdict = "PASS_PPO_ROLLOUT_MECHANISM"
    if missing or not row_count_match or not mechanism_ok or not grasp_ok:
        verdict = "FAIL_PPO_ROLLOUT_MECHANISM"

    print(
        "ppo_rollout_audit line1 "
        f"csv_rows={len(rows)} summary_trials={summary.get('trials')} row_count_match={row_count_match} "
        f"controller={summary.get('controller')} learned_policy={summary.get('learned_policy')} "
        f"missing_columns={missing}"
    )
    print(
        "ppo_rollout_audit line2 "
        f"mechanism_ok={mechanism_ok} training={summary.get('training')} "
        f"dataset_generation={summary.get('dataset_generation')} grasp_attach={summary.get('grasp_attach')} "
        f"rollout_object_posewrite={summary.get('rollout_object_posewrite')} grasp_ok={grasp_ok}"
    )
    print(
        "ppo_rollout_audit line3 "
        f"controlled_push_rate={controlled_rate:.9f} impact_outlier_rate={impact_rate:.9f} "
        f"low_motion_rate={low_motion_rate:.9f} success_marker_rate={success_rate:.9f} "
        f"disp_along_push_mean_m={_mean(rows, 'disp_along_push_m'):.9f} "
        f"disp_xy_mean_m={_mean(rows, 'disp_xy_m'):.9f}"
    )
    print(
        "ppo_rollout_audit line4 "
        f"bc_teacher_blend={summary.get('bc_teacher_blend')} "
        f"bc_teacher_imitation_reward_scale={summary.get('bc_teacher_imitation_reward_scale')} "
        f"checkpoint={summary.get('checkpoint')}"
    )
    print(f"ppo_rollout_audit line5 verdict={verdict} track_a_grasp_success=NO dataset_ready=NO")
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
