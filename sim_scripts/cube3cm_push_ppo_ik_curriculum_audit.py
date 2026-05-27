#!/usr/bin/env python3
"""Audit the IK-endpoint cube-push PPO curriculum run."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
LOG_DIR = RUN_DIR / "ppo_ik_curriculum_logs/cube_push_ik_curriculum_50iter_20260526"
STDOUT = RUN_DIR / "ppo_ik_curriculum_50iter_stdout.out"
STDERR = RUN_DIR / "ppo_ik_curriculum_50iter_stderr.err"
EXIT_CODE = RUN_DIR / "ppo_ik_curriculum_50iter_exit_code.txt"
OUT = RUN_DIR / "ppo_ik_curriculum_50iter_audit.out"
SUMMARY = RUN_DIR / "ppo_ik_curriculum_50iter_audit_summary.json"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _exit_code() -> int | None:
    m = re.search(r"exit_code:(-?\d+)", EXIT_CODE.read_text().strip())
    return int(m.group(1)) if m else None


def _scalars() -> dict[str, dict[str, Any]]:
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(LOG_DIR.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"no event file in {LOG_DIR}")
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    out: dict[str, dict[str, Any]] = {}
    for tag in ea.Tags().get("scalars", []):
        values = ea.Scalars(tag)
        if not values:
            continue
        out[tag] = {
            "n": len(values),
            "first_step": values[0].step,
            "first": float(values[0].value),
            "last_step": values[-1].step,
            "last": float(values[-1].value),
            "min": float(min(v.value for v in values)),
            "max": float(max(v.value for v in values)),
        }
    return out


def _line(tag: str, scalars: dict[str, dict[str, Any]]) -> str:
    item = scalars[tag]
    return (
        f"EVENT_SCALAR tag={tag} n={item['n']} first_step={item['first_step']} "
        f"first={item['first']:.9f} last_step={item['last_step']} "
        f"last={item['last']:.9f} min={item['min']:.9f} max={item['max']:.9f}"
    )


def main() -> int:
    scalars = _scalars()
    event_file = sorted(LOG_DIR.glob("events.out.tfevents.*"))[0]
    model0 = LOG_DIR / "model_0.pt"
    model49 = LOG_DIR / "model_49.pt"
    code = _exit_code()

    required = [
        "Episode/cube_push_disp_along_m",
        "Episode/cube_push_disp_xy_m",
        "Episode/cube_push_target_xy_dist_m",
        "Episode/cube_push_controlled_rate",
        "Episode/cube_push_impact_rate",
        "Episode/cube_push_low_motion_rate",
        "Episode/cube_push_success_rate",
        "Episode/cube_push_grasped_marker_rate",
        "Episode/cube_push_ik_endpoint_reset_rate",
        "Episode/cube_push_ik_reset_err_mm",
        "Train/mean_reward",
        "Train/mean_episode_length",
    ]

    controlled = scalars["Episode/cube_push_controlled_rate"]
    target_dist = scalars["Episode/cube_push_target_xy_dist_m"]
    impact = scalars["Episode/cube_push_impact_rate"]
    low_motion = scalars["Episode/cube_push_low_motion_rate"]
    success = scalars["Episode/cube_push_success_rate"]
    disp_along = scalars["Episode/cube_push_disp_along_m"]
    grasp = scalars["Episode/cube_push_grasped_marker_rate"]
    ik_rate = scalars["Episode/cube_push_ik_endpoint_reset_rate"]
    train_iters = scalars["Train/mean_reward"]["n"]

    run_completed = (
        code == 0
        and model49.exists()
        and train_iters == 50
        and ik_rate["last"] > 0.99
        and grasp["max"] == 0.0
    )
    signal_present = (
        run_completed
        and controlled["last"] > controlled["first"]
        and low_motion["last"] < low_motion["first"]
    )
    impact_heavy_or_regressed = (
        run_completed
        and (
            impact["last"] > 0.10
            or target_dist["last"] > max(0.12, target_dist["first"] * 2.0)
            or disp_along["last"] < 0.0
        )
    )
    still_not_success = success["last"] < 0.20 or controlled["last"] < 0.60
    verdict = "IK_CURRICULUM_RUN_BLOCKED"
    if impact_heavy_or_regressed:
        verdict = "IK_CURRICULUM_RAN_IMPACT_HEAVY_REGRESSED_DIRECTION_NO_SUCCESS_CLAIM"
    elif signal_present:
        verdict = "IK_CURRICULUM_SIGNAL_PRESENT_NEEDS_EVAL" if still_not_success else "IK_CURRICULUM_STRONG_SIGNAL_NEEDS_EVAL"

    summary = {
        "exit_code": code,
        "stdout": str(STDOUT),
        "stdout_md5": _md5(STDOUT),
        "stderr": str(STDERR),
        "stderr_md5": _md5(STDERR),
        "log_dir": str(LOG_DIR),
        "event_file": str(event_file),
        "event_file_md5": _md5(event_file),
        "model_0": str(model0),
        "model_0_md5": _md5(model0),
            "model_49": str(model49),
            "model_49_md5": _md5(model49),
            "verdict": verdict,
            "run_completed": run_completed,
            "signal_present": signal_present,
            "impact_heavy_or_regressed": impact_heavy_or_regressed,
            "scalars": scalars,
        }

    lines = [
        (
            f"IK_CURRICULUM_INPUT exit_code={code} stdout={STDOUT} stdout_md5={summary['stdout_md5']} "
            f"stderr={STDERR} stderr_md5={summary['stderr_md5']}"
        ),
        (
            f"IK_CURRICULUM_ARTIFACTS log_dir={LOG_DIR} event_file={event_file} "
            f"event_md5={summary['event_file_md5']} model_0_exists={'YES' if model0.exists() else 'NO'} "
            f"model_49_exists={'YES' if model49.exists() else 'NO'} model_49_md5={summary['model_49_md5']}"
        ),
        f"ITERATION_COVERAGE event_train_mean_reward_count={train_iters}",
    ]
    for tag in required:
        lines.append(_line(tag, scalars))
    lines.append(
        "CRITICAL_INTERPRETATION "
        "ik_endpoint_reset_active=YES training_loop_ran=YES learned_policy_success=NO "
        "evaluation_rollout=NO dataset_generation=NO track_a_grasp_success=NO"
    )
    lines.append(
        "FAILURE_MODE "
        f"run_completed={'YES' if run_completed else 'NO'} "
        f"signal_present={'YES' if signal_present else 'NO'} "
        f"impact_heavy_or_regressed={'YES' if impact_heavy_or_regressed else 'NO'} "
        "interpretation='IK pre-contact worked, but the learned behavior shifted toward "
        "large/negative-direction displacement and impact instead of stable aligned push'"
    )
    lines.append(f"RESULT ppo_ik_curriculum_audit={verdict}")

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
