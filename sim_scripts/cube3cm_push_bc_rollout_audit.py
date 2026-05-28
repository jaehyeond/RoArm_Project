"""Audit a learned BC joint-delta policy rollout for cube push."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def rate(rows: list[dict[str, str]], key: str) -> float:
    return sum(int(float(r[key])) for r in rows) / len(rows) if rows else 0.0


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def direction_key(row: dict[str, str]) -> tuple[int, int]:
    return (int(round(float(row["push_dx"]))), int(round(float(row["push_dy"]))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--summary_json", type=Path, required=True)
    ap.add_argument("--min_controlled", type=float, default=0.80)
    ap.add_argument("--max_impact", type=float, default=0.05)
    ap.add_argument("--max_low_motion", type=float, default=0.20)
    ap.add_argument("--min_success", type=float, default=0.25)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    summary = json.loads(args.summary_json.read_text())
    row_count_match = len(rows) == int(summary.get("trials", -1))
    mechanism_ok = (
        summary.get("controller") == "BC_MLP_joint_delta_policy"
        and summary.get("learned_policy") is True
        and summary.get("supervised_bc_checkpoint") is True
        and summary.get("diffik_controller_used") is False
        and summary.get("training") is False
        and summary.get("grasp_attach") is False
        and summary.get("rollout_object_posewrite") is False
        and int(summary.get("posewrite_calls_during_rollout", -1)) == 0
    )
    grasp_ok = rate(rows, "grasped_marker") == 0.0
    controlled = rate(rows, "controlled_push")
    impact = rate(rows, "impact_outlier")
    low_motion = rate(rows, "low_motion")
    success = rate(rows, "success_marker")
    performance_ok = (
        controlled >= float(args.min_controlled)
        and impact <= float(args.max_impact)
        and low_motion <= float(args.max_low_motion)
        and success >= float(args.min_success)
    )

    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[direction_key(row)].append(row)
    direction_stats = {
        str(k): {
            "n": len(v),
            "controlled": rate(v, "controlled_push"),
            "impact": rate(v, "impact_outlier"),
            "low_motion": rate(v, "low_motion"),
            "success": rate(v, "success_marker"),
        }
        for k, v in sorted(groups.items())
    }
    verdict = "PASS_LEARNED_BC_POLICY_ROLLOUT" if (
        row_count_match and mechanism_ok and grasp_ok and performance_ok
    ) else "FAIL_LEARNED_BC_POLICY_ROLLOUT"

    print(
        "bc_rollout_audit line1 "
        f"csv_rows={len(rows)} summary_trials={summary.get('trials')} row_count_match={row_count_match} "
        f"controller={summary.get('controller')} learned_policy={summary.get('learned_policy')} "
        f"diffik_controller_used={summary.get('diffik_controller_used')}"
    )
    print(
        "bc_rollout_audit line2 "
        f"mechanism_ok={mechanism_ok} training={summary.get('training')} "
        f"posewrite_calls={summary.get('posewrite_calls_during_rollout')} grasp_ok={grasp_ok}"
    )
    print(
        "bc_rollout_audit line3 "
        f"controlled_push_rate={controlled:.9f} impact_outlier_rate={impact:.9f} "
        f"low_motion_rate={low_motion:.9f} success_marker_rate={success:.9f} "
        f"performance_ok={performance_ok}"
    )
    print(
        "bc_rollout_audit line4 "
        f"disp_along_push_mean_m={mean(rows, 'disp_along_push_m'):.9f} "
        f"disp_xy_mean_m={mean(rows, 'disp_xy_m'):.9f} "
        f"final_tcp_target_err_mean_m={mean(rows, 'final_tcp_target_err_m'):.9f}"
    )
    print(f"bc_rollout_audit line5 direction_stats={direction_stats}")
    print(
        "bc_rollout_audit line6 "
        f"verdict={verdict} learned_policy_rollout={'YES' if verdict.startswith('PASS') else 'NO'} "
        "track_a_grasp_success=NO"
    )
    return 0 if verdict.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
