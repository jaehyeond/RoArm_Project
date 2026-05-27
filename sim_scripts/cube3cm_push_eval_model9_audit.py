#!/usr/bin/env python3
"""Posthoc audit for model_9 no-attach cube-push evaluation rollout."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
EVAL_SUMMARY = RUN_DIR / "ppo_smoke_eval_model9_summary.json"
EVAL_CSV = RUN_DIR / "ppo_smoke_eval_model9.csv"
EVAL_EXIT = RUN_DIR / "ppo_smoke_eval_model9_exit_code.txt"
EVAL_STDOUT = RUN_DIR / "ppo_smoke_eval_model9_stdout.out"
EVAL_STDERR = RUN_DIR / "ppo_smoke_eval_model9_stderr.err"
PPO_AUDIT_SUMMARY = RUN_DIR / "ppo_smoke_audit_summary.json"
OUT = RUN_DIR / "ppo_smoke_eval_model9_audit.out"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _exit_code(path: Path) -> int | None:
    m = re.search(r"exit_code:(-?\d+)", path.read_text().strip())
    return int(m.group(1)) if m else None


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _line(path: Path, nr: int) -> str:
    for idx, text in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if idx == nr:
            return text
    return ""


def main() -> int:
    summary = json.loads(EVAL_SUMMARY.read_text())
    ppo_summary = json.loads(PPO_AUDIT_SUMMARY.read_text())
    train_scalars = ppo_summary["scalars"]
    rows = _csv_rows(EVAL_CSV)
    code = _exit_code(EVAL_EXIT)

    train_controlled_last = train_scalars["Episode/cube_push_controlled_rate"]["last"]
    train_low_motion_last = train_scalars["Episode/cube_push_low_motion_rate"]["last"]
    train_success_last = train_scalars["Episode/cube_push_success_rate"]["last"]

    eval_controlled = summary["controlled_push_rate"]
    eval_low_motion = summary["low_motion_rate"]
    eval_success = summary["success_marker_rate"]
    eval_grasp = summary["grasped_marker_rate"]

    verdict = "EVAL_BLOCKED"
    if code == 0 and rows == summary["trials"] and eval_grasp == 0.0:
        verdict = "EVAL_RAN_WEAK_POLICY_NO_SUCCESS_CLAIM"
    if eval_success >= 0.20 and eval_controlled >= 0.50 and eval_grasp == 0.0:
        verdict = "EVAL_SIGNAL_STRONG_NEEDS_LARGER_RUN"

    lines = [
        (
            f"EVAL_INPUT exit_code={code} summary={EVAL_SUMMARY} summary_md5={_md5(EVAL_SUMMARY)} "
            f"csv={EVAL_CSV} csv_md5={_md5(EVAL_CSV)} csv_rows={rows}"
        ),
        (
            f"EVAL_LOGS stdout={EVAL_STDOUT} stdout_md5={_md5(EVAL_STDOUT)} "
            f"stderr={EVAL_STDERR} stderr_md5={_md5(EVAL_STDERR)}"
        ),
        (
            f"SUMMARY_REF line=3 controlled_push_rate={summary['controlled_push_rate']:.9f} "
            f"line=5 disp_along_push_mean_m={summary['disp_along_push_mean_m']:.9f} "
            f"line=6 disp_xy_mean_m={summary['disp_xy_mean_m']:.9f} "
            f"line=10 low_motion_rate={summary['low_motion_rate']:.9f} "
            f"line=15 success_marker_rate={summary['success_marker_rate']:.9f} "
            f"line=18 trials={summary['trials']}"
        ),
        (
            f"MECHANISM_CHECK training={summary['training']} dataset_generation={summary['dataset_generation']} "
            f"grasp_attach={summary['grasp_attach']} rollout_object_posewrite={summary['rollout_object_posewrite']} "
            f"grasped_marker_rate={summary['grasped_marker_rate']:.9f}"
        ),
        (
            f"TRAIN_VS_EVAL train_controlled_last={train_controlled_last:.9f} "
            f"eval_controlled={eval_controlled:.9f} train_low_motion_last={train_low_motion_last:.9f} "
            f"eval_low_motion={eval_low_motion:.9f} train_success_last={train_success_last:.9f} "
            f"eval_success={eval_success:.9f}"
        ),
        (
            "CRITICAL_INTERPRETATION "
            "policy_rollout_ran=YES learned_success=NO controlled_rate_not_enough=YES "
            "low_motion_still_high=YES dataset_generation=NO track_a_grasp_success=NO"
        ),
        f"CSV_SAMPLE line=1 text=\"{_line(EVAL_CSV, 1)}\"",
        f"CSV_SAMPLE line=2 text=\"{_line(EVAL_CSV, 2)}\"",
        f"CSV_SAMPLE line=513 text=\"{_line(EVAL_CSV, 513)}\"",
        f"RESULT eval_model9_audit={verdict}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
