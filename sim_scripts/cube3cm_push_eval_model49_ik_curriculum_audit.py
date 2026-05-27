#!/usr/bin/env python3
"""Posthoc audit for the IK-curriculum model_49 cube-push evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
EVAL_SUMMARY = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_summary.json"
EVAL_CSV = RUN_DIR / "ppo_ik_curriculum_model49_eval1024.csv"
EVAL_EXIT = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_exit_code.txt"
EVAL_STDOUT = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_stdout.out"
EVAL_STDERR = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_stderr.err"
TRAIN_AUDIT_SUMMARY = RUN_DIR / "ppo_ik_curriculum_50iter_audit_summary.json"
OUT = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_audit.out"
SUMMARY_OUT = RUN_DIR / "ppo_ik_curriculum_model49_eval1024_audit_summary.json"


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    train_summary = json.loads(TRAIN_AUDIT_SUMMARY.read_text())
    scalars = train_summary["scalars"]
    rows = _rows(EVAL_CSV)
    code = _exit_code(EVAL_EXIT)

    disp_along = [r["disp_along_push_m"] for r in rows]
    disp_xy = [r["disp_xy_m"] for r in rows]
    target_dist = [r["target_xy_dist_m"] for r in rows]
    speed = [r["final_speed_mps"] for r in rows]
    tip = [r["tip_angle_deg"] for r in rows]

    final_aligned_3cm_rate = _rate(rows, lambda r: r["disp_along_push_m"] >= 0.030)
    final_controlled_3cm_rate = _rate(
        rows,
        lambda r: r["controlled_push"] == 1.0 and r["disp_along_push_m"] >= 0.030,
    )
    clean_success_marker_rate = _rate(
        rows,
        lambda r: (
            r["success_marker"] == 1.0
            and r["controlled_push"] == 1.0
            and r["impact_outlier"] == 0.0
            and r["target_xy_dist_m"] <= 0.050
        ),
    )
    negative_push_rate = _rate(rows, lambda r: r["disp_along_push_m"] < 0.0)
    far_target_rate = _rate(rows, lambda r: r["target_xy_dist_m"] > 0.100)

    run_ok = code == 0 and len(rows) == int(summary["trials"]) and summary["grasped_marker_rate"] == 0.0
    impact_heavy = summary["impact_outlier_rate"] > 0.10 or summary["disp_xy_mean_m"] > 0.10
    not_clean_success = clean_success_marker_rate < 0.50 or summary["impact_outlier_rate"] > 0.05
    verdict = "MODEL49_EVAL_BLOCKED"
    if run_ok and impact_heavy:
        verdict = "MODEL49_EVAL_RAN_IMPACT_HEAVY_NO_SUCCESS_CLAIM"
    elif run_ok and not_clean_success:
        verdict = "MODEL49_EVAL_RAN_WEAK_CLEAN_SUCCESS_NO_SUCCESS_CLAIM"
    elif run_ok:
        verdict = "MODEL49_EVAL_CLEAN_SIGNAL_NEEDS_LARGER_RUN"

    audit_summary = {
        "exit_code": code,
        "rows": len(rows),
        "summary": summary,
        "train_verdict": train_summary["verdict"],
        "final_aligned_3cm_rate": final_aligned_3cm_rate,
        "final_controlled_3cm_rate": final_controlled_3cm_rate,
        "clean_success_marker_rate": clean_success_marker_rate,
        "negative_push_rate": negative_push_rate,
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
            f"MODEL49_EVAL_INPUT exit_code={code} summary={EVAL_SUMMARY} "
            f"summary_md5={_md5(EVAL_SUMMARY)} csv={EVAL_CSV} csv_md5={_md5(EVAL_CSV)} "
            f"csv_rows={len(rows)}"
        ),
        (
            f"MODEL49_EVAL_LOGS stdout={EVAL_STDOUT} stdout_md5={_md5(EVAL_STDOUT)} "
            f"stderr={EVAL_STDERR} stderr_md5={_md5(EVAL_STDERR)}"
        ),
        (
            f"SUMMARY_REF line=3 controlled_push_rate={summary['controlled_push_rate']:.9f} "
            f"line=5 disp_along_push_mean_m={summary['disp_along_push_mean_m']:.9f} "
            f"line=6 disp_xy_mean_m={summary['disp_xy_mean_m']:.9f} "
            f"line=9 impact_outlier_rate={summary['impact_outlier_rate']:.9f} "
            f"line=10 low_motion_rate={summary['low_motion_rate']:.9f} "
            f"line=15 success_marker_rate={summary['success_marker_rate']:.9f} "
            f"line=16 target_xy_dist_mean_m={summary['target_xy_dist_mean_m']:.9f} "
            f"line=18 trials={summary['trials']}"
        ),
        (
            f"CSV_DISTRIBUTION disp_along_mean_m={_mean(disp_along):.9f} "
            f"disp_along_p50_m={_quantile(disp_along, 0.50):.9f} "
            f"disp_along_p95_m={_quantile(disp_along, 0.95):.9f} "
            f"disp_along_min_m={min(disp_along):.9f} disp_along_max_m={max(disp_along):.9f}"
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
            f"CLEAN_SUCCESS_CHECK final_aligned_3cm_rate={final_aligned_3cm_rate:.9f} "
            f"final_controlled_3cm_rate={final_controlled_3cm_rate:.9f} "
            f"clean_success_marker_rate={clean_success_marker_rate:.9f} "
            f"negative_push_rate={negative_push_rate:.9f} far_target_rate={far_target_rate:.9f}"
        ),
        (
            f"TRAIN_VS_EVAL train_verdict={train_summary['verdict']} "
            f"train_controlled_last={scalars['Episode/cube_push_controlled_rate']['last']:.9f} "
            f"eval_controlled={summary['controlled_push_rate']:.9f} "
            f"train_impact_last={scalars['Episode/cube_push_impact_rate']['last']:.9f} "
            f"eval_impact={summary['impact_outlier_rate']:.9f} "
            f"train_disp_along_last={scalars['Episode/cube_push_disp_along_m']['last']:.9f} "
            f"eval_disp_along_mean={summary['disp_along_push_mean_m']:.9f}"
        ),
        (
            f"MECHANISM_CHECK training={summary['training']} dataset_generation={summary['dataset_generation']} "
            f"grasp_attach={summary['grasp_attach']} rollout_object_posewrite={summary['rollout_object_posewrite']} "
            f"grasped_marker_rate={summary['grasped_marker_rate']:.9f}"
        ),
        (
            "CRITICAL_INTERPRETATION "
            "policy_rollout_ran=YES ik_endpoint_reset_eval=YES learned_clean_push_success=NO "
            "success_marker_is_contaminated_by_impact=YES dataset_generation=NO track_a_grasp_success=NO"
        ),
    ]
    for rank, (csv_line, row) in enumerate(_top_rows(rows, "disp_xy_m"), start=1):
        lines.append(
            f"TOP_DISP_OUTLIER rank={rank} csv_line={csv_line} "
            f"disp_xy_m={row['disp_xy_m']:.9f} disp_along_push_m={row['disp_along_push_m']:.9f} "
            f"target_xy_dist_m={row['target_xy_dist_m']:.9f} speed_mps={row['final_speed_mps']:.9f} "
            f"tip_deg={row['tip_angle_deg']:.9f} controlled={int(row['controlled_push'])} "
            f"impact={int(row['impact_outlier'])} success_marker={int(row['success_marker'])}"
        )
    lines.extend(
        [
            f"CSV_SAMPLE line=1 text=\"{_line(EVAL_CSV, 1)}\"",
            f"CSV_SAMPLE line=2 text=\"{_line(EVAL_CSV, 2)}\"",
            f"CSV_SAMPLE line=1025 text=\"{_line(EVAL_CSV, 1025)}\"",
            f"RESULT model49_eval1024_audit={verdict}",
        ]
    )

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY_OUT.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
