#!/usr/bin/env python3
"""Posthoc audit for a scripted-teacher-on cube-push diagnostic eval."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
EVAL_SUMMARY = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_summary.json"
EVAL_CSV = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024.csv"
EVAL_EXIT = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_exit_code.txt"
EVAL_STDOUT = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_stdout.out"
EVAL_STDERR = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_stderr.err"
PREV_CONTACT_EVAL_AUDIT_SUMMARY = RUN_DIR / "ppo_contact_speed_model49_eval1024_audit_summary.json"
OUT = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_audit.out"
SUMMARY_OUT = RUN_DIR / "ppo_contact_speed_teacher_on_eval1024_audit_summary.json"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _exit_code(path: Path) -> int | None:
    m = re.search(r"exit_code:(-?\d+)", path.read_text().strip())
    return int(m.group(1)) if m else None


def _rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            rows.append({k: float(v) for k, v in raw.items()})
    return rows


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


def _rate(rows: list[dict[str, float]], predicate) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def _line(path: Path, nr: int) -> str:
    for idx, text in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if idx == nr:
            return text
    return ""


def _top_rows(rows: list[dict[str, float]], key: str, n: int = 5) -> list[tuple[int, dict[str, float]]]:
    indexed = list(enumerate(rows, start=2))
    indexed.sort(key=lambda item: item[1][key], reverse=True)
    return indexed[:n]


def main() -> int:
    summary = json.loads(EVAL_SUMMARY.read_text())
    prev_contact = json.loads(PREV_CONTACT_EVAL_AUDIT_SUMMARY.read_text())
    rows = _rows(EVAL_CSV)
    code = _exit_code(EVAL_EXIT)

    disp_xy = [r["disp_xy_m"] for r in rows]
    target_dist = [r["target_xy_dist_m"] for r in rows]
    speed = [r["final_speed_mps"] for r in rows]
    tip = [r["tip_angle_deg"] for r in rows]

    final_controlled_3cm_rate = _rate(rows, lambda r: r["controlled_push"] == 1.0 and r["disp_along_push_m"] >= 0.030)
    clean_success_marker_rate = _rate(
        rows,
        lambda r: (
            r["success_marker"] == 1.0
            and r["controlled_push"] == 1.0
            and r["impact_outlier"] == 0.0
            and r["target_xy_dist_m"] <= 0.050
        ),
    )
    impact_low_motion_rate = _rate(rows, lambda r: r["impact_outlier"] == 1.0 and r["disp_xy_m"] < 0.005)
    impact_not_large_disp_rate = _rate(rows, lambda r: r["impact_outlier"] == 1.0 and r["disp_xy_m"] < 0.060)
    far_target_rate = _rate(rows, lambda r: r["target_xy_dist_m"] > 0.100)

    run_ok = (
        code == 0
        and len(rows) == int(summary["trials"])
        and summary["grasped_marker_rate"] == 0.0
        and summary["scripted_teacher_blend"] > 0.99
    )
    scripted_safe = (
        run_ok
        and summary["impact_outlier_rate"] < 0.05
        and clean_success_marker_rate > 0.30
        and summary["controlled_push_rate"] > 0.60
    )
    verdict = "TEACHER_ON_DIAGNOSTIC_BLOCKED"
    if scripted_safe:
        verdict = "TEACHER_ON_DIAGNOSTIC_SAFE_SCRIPTED_NOT_LEARNED"
    elif run_ok:
        verdict = "TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K"

    audit_summary = {
        "exit_code": code,
        "rows": len(rows),
        "summary": summary,
        "previous_contact_eval_verdict": prev_contact["verdict"],
        "final_controlled_3cm_rate": final_controlled_3cm_rate,
        "clean_success_marker_rate": clean_success_marker_rate,
        "impact_low_motion_rate": impact_low_motion_rate,
        "impact_not_large_disp_rate": impact_not_large_disp_rate,
        "far_target_rate": far_target_rate,
        "disp_xy_p95_m": _quantile(disp_xy, 0.95),
        "disp_xy_max_m": max(disp_xy) if disp_xy else 0.0,
        "target_xy_dist_p95_m": _quantile(target_dist, 0.95),
        "target_xy_dist_max_m": max(target_dist) if target_dist else 0.0,
        "speed_p95_mps": _quantile(speed, 0.95),
        "speed_max_mps": max(speed) if speed else 0.0,
        "tip_p95_deg": _quantile(tip, 0.95),
        "tip_max_deg": max(tip) if tip else 0.0,
        "verdict": verdict,
    }

    lines = [
        (
            f"TEACHER_ON_DIAGNOSTIC_INPUT exit_code={code} summary={EVAL_SUMMARY} "
            f"summary_md5={_md5(EVAL_SUMMARY)} csv={EVAL_CSV} csv_md5={_md5(EVAL_CSV)} "
            f"csv_rows={len(rows)}"
        ),
        (
            f"TEACHER_ON_DIAGNOSTIC_LOGS stdout={EVAL_STDOUT} stdout_md5={_md5(EVAL_STDOUT)} "
            f"stderr={EVAL_STDERR} stderr_md5={_md5(EVAL_STDERR)}"
        ),
        (
            f"SUMMARY_REF action_scale={summary['action_scale']:.9f} "
            f"max_joint_delta_per_step_rad={summary['max_joint_delta_per_step_rad']:.9f} "
            f"contact_joint_delta_scale={summary['contact_joint_delta_scale']:.9f} "
            f"fast_cube_joint_delta_scale={summary['fast_cube_joint_delta_scale']:.9f} "
            f"speed_penalty_start_mps={summary['speed_penalty_start_mps']:.9f} "
            f"scripted_teacher_blend={summary['scripted_teacher_blend']:.9f} "
            f"scripted_teacher_horizon_frac={summary['scripted_teacher_horizon_frac']:.9f} "
            f"scripted_teacher_goal_push_m={summary['scripted_teacher_goal_push_m']:.9f}"
        ),
        (
            f"SUMMARY_METRICS controlled_push_rate={summary['controlled_push_rate']:.9f} "
            f"disp_along_push_mean_m={summary['disp_along_push_mean_m']:.9f} "
            f"disp_xy_mean_m={summary['disp_xy_mean_m']:.9f} "
            f"impact_outlier_rate={summary['impact_outlier_rate']:.9f} "
            f"low_motion_rate={summary['low_motion_rate']:.9f} "
            f"success_marker_rate={summary['success_marker_rate']:.9f} "
            f"target_xy_dist_mean_m={summary['target_xy_dist_mean_m']:.9f} "
            f"trials={summary['trials']}"
        ),
        (
            f"CSV_DISTRIBUTION disp_xy_p95_m={audit_summary['disp_xy_p95_m']:.9f} "
            f"disp_xy_max_m={audit_summary['disp_xy_max_m']:.9f} "
            f"target_xy_dist_p95_m={audit_summary['target_xy_dist_p95_m']:.9f} "
            f"target_xy_dist_max_m={audit_summary['target_xy_dist_max_m']:.9f} "
            f"speed_p95_mps={audit_summary['speed_p95_mps']:.9f} "
            f"speed_max_mps={audit_summary['speed_max_mps']:.9f} "
            f"tip_p95_deg={audit_summary['tip_p95_deg']:.9f} "
            f"tip_max_deg={audit_summary['tip_max_deg']:.9f}"
        ),
        (
            f"CLEAN_SUCCESS_CHECK final_controlled_3cm_rate={final_controlled_3cm_rate:.9f} "
            f"clean_success_marker_rate={clean_success_marker_rate:.9f} "
            f"impact_low_motion_rate={impact_low_motion_rate:.9f} "
            f"impact_not_large_disp_rate={impact_not_large_disp_rate:.9f} "
            f"far_target_rate={far_target_rate:.9f}"
        ),
        (
            f"PREV_POLICY_EVAL_COMPARISON prev_verdict={prev_contact['verdict']} "
            f"prev_impact_rate={prev_contact['summary']['impact_outlier_rate']:.9f} "
            f"new_impact_rate={summary['impact_outlier_rate']:.9f} "
            f"prev_clean_success={prev_contact['clean_success_marker_rate']:.9f} "
            f"new_clean_success={clean_success_marker_rate:.9f}"
        ),
        (
            f"MECHANISM_CHECK training={summary['training']} dataset_generation={summary['dataset_generation']} "
            f"grasp_attach={summary['grasp_attach']} rollout_object_posewrite={summary['rollout_object_posewrite']} "
            f"grasped_marker_rate={summary['grasped_marker_rate']:.9f} scripted_teacher_blend={summary['scripted_teacher_blend']:.9f}"
        ),
        (
            "CRITICAL_INTERPRETATION "
            "teacher_on_scripted_diagnostic=YES learned_policy_success=NO "
            "this_is_not_frozen_policy_claim=YES dataset_generation=NO track_a_grasp_success=NO"
        ),
    ]
    for rank, (csv_line, row) in enumerate(_top_rows(rows, "final_speed_mps"), start=1):
        lines.append(
            f"TOP_SPEED_OUTLIER rank={rank} csv_line={csv_line} "
            f"speed_mps={row['final_speed_mps']:.9f} disp_xy_m={row['disp_xy_m']:.9f} "
            f"disp_along_push_m={row['disp_along_push_m']:.9f} target_xy_dist_m={row['target_xy_dist_m']:.9f} "
            f"tip_deg={row['tip_angle_deg']:.9f} controlled={int(row['controlled_push'])} "
            f"impact={int(row['impact_outlier'])} low_motion={int(row['low_motion'])} "
            f"success_marker={int(row['success_marker'])}"
        )
    lines.extend(
        [
            f"CSV_SAMPLE line=1 text=\"{_line(EVAL_CSV, 1)}\"",
            f"CSV_SAMPLE line=2 text=\"{_line(EVAL_CSV, 2)}\"",
            f"CSV_SAMPLE line={len(rows) + 1} text=\"{_line(EVAL_CSV, len(rows) + 1)}\"",
            f"RESULT teacher_on_diagnostic_eval1024_audit={verdict}",
        ]
    )

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY_OUT.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
