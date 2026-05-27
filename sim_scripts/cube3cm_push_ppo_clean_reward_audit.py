#!/usr/bin/env python3
"""Audit the clean-reward IK endpoint cube-push PPO run."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
LOG_DIR = RUN_DIR / "ppo_clean_reward_logs/cube_push_clean_reward_50iter_20260526"
STDOUT = RUN_DIR / "ppo_clean_reward_50iter_stdout.out"
STDERR = RUN_DIR / "ppo_clean_reward_50iter_stderr.err"
EXIT_CODE = RUN_DIR / "ppo_clean_reward_50iter_exit_code.txt"
PREV_AUDIT_SUMMARY = RUN_DIR / "ppo_ik_curriculum_50iter_audit_summary.json"
OUT = RUN_DIR / "ppo_clean_reward_50iter_audit.out"
SUMMARY = RUN_DIR / "ppo_clean_reward_50iter_audit_summary.json"


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
    prev = json.loads(PREV_AUDIT_SUMMARY.read_text())
    prev_scalars = prev["scalars"]
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
        "Episode/cube_push_far_target_rate",
        "Episode/cube_push_terminal_impact_rate",
        "Episode/cube_push_success_rate",
        "Episode/cube_push_grasped_marker_rate",
        "Episode/cube_push_ik_endpoint_reset_rate",
        "Episode/cube_push_ik_reset_err_mm",
        "Train/mean_reward",
        "Train/mean_episode_length",
    ]

    train_iters = scalars["Train/mean_reward"]["n"]
    controlled = scalars["Episode/cube_push_controlled_rate"]
    impact = scalars["Episode/cube_push_impact_rate"]
    low_motion = scalars["Episode/cube_push_low_motion_rate"]
    far_target = scalars["Episode/cube_push_far_target_rate"]
    terminal_impact = scalars["Episode/cube_push_terminal_impact_rate"]
    success = scalars["Episode/cube_push_success_rate"]
    disp_along = scalars["Episode/cube_push_disp_along_m"]
    disp_xy = scalars["Episode/cube_push_disp_xy_m"]
    target_dist = scalars["Episode/cube_push_target_xy_dist_m"]
    grasp = scalars["Episode/cube_push_grasped_marker_rate"]
    ik_rate = scalars["Episode/cube_push_ik_endpoint_reset_rate"]

    run_completed = (
        code == 0
        and model49.exists()
        and train_iters == 50
        and ik_rate["last"] > 0.99
        and grasp["max"] == 0.0
    )
    safety_improved_vs_prev = (
        impact["last"] < prev_scalars["Episode/cube_push_impact_rate"]["last"]
        and target_dist["last"] < prev_scalars["Episode/cube_push_target_xy_dist_m"]["last"]
        and disp_xy["last"] < prev_scalars["Episode/cube_push_disp_xy_m"]["last"]
    )
    clean_signal = (
        run_completed
        and impact["last"] < 0.05
        and far_target["last"] < 0.05
        and terminal_impact["last"] < 0.08
        and target_dist["last"] < 0.08
        and disp_along["last"] > 0.0
        and success["last"] > 0.05
    )
    verdict = "CLEAN_REWARD_RUN_BLOCKED"
    if clean_signal:
        verdict = "CLEAN_REWARD_SIGNAL_PRESENT_NEEDS_MODEL49_EVAL"
    elif run_completed and safety_improved_vs_prev:
        verdict = "CLEAN_REWARD_RAN_SAFER_BUT_WEAK_NEEDS_MODEL49_EVAL"
    elif run_completed:
        verdict = "CLEAN_REWARD_RAN_NO_SUCCESS_CLAIM"

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
        "previous_verdict": prev["verdict"],
        "run_completed": run_completed,
        "safety_improved_vs_prev": safety_improved_vs_prev,
        "clean_signal": clean_signal,
        "verdict": verdict,
        "scalars": scalars,
    }

    lines = [
        (
            f"CLEAN_REWARD_INPUT exit_code={code} stdout={STDOUT} stdout_md5={summary['stdout_md5']} "
            f"stderr={STDERR} stderr_md5={summary['stderr_md5']}"
        ),
        (
            f"CLEAN_REWARD_ARTIFACTS log_dir={LOG_DIR} event_file={event_file} "
            f"event_md5={summary['event_file_md5']} model_0_exists={'YES' if model0.exists() else 'NO'} "
            f"model_49_exists={'YES' if model49.exists() else 'NO'} model_49_md5={summary['model_49_md5']}"
        ),
        f"ITERATION_COVERAGE event_train_mean_reward_count={train_iters}",
    ]
    for tag in required:
        lines.append(_line(tag, scalars))
    lines.append(
        "PREV_IK_COMPARISON "
        f"prev_verdict={prev['verdict']} "
        f"prev_disp_xy_last={prev_scalars['Episode/cube_push_disp_xy_m']['last']:.9f} "
        f"new_disp_xy_last={disp_xy['last']:.9f} "
        f"prev_target_dist_last={prev_scalars['Episode/cube_push_target_xy_dist_m']['last']:.9f} "
        f"new_target_dist_last={target_dist['last']:.9f} "
        f"prev_impact_last={prev_scalars['Episode/cube_push_impact_rate']['last']:.9f} "
        f"new_impact_last={impact['last']:.9f}"
    )
    lines.append(
        "CRITICAL_INTERPRETATION "
        "ik_endpoint_reset_active=YES training_loop_ran=YES learned_policy_success=NO "
        "cleaner_than_prev_train_log="
        f"{'YES' if safety_improved_vs_prev else 'NO'} "
        "evaluation_rollout=NO dataset_generation=NO track_a_grasp_success=NO"
    )
    lines.append(
        "NEXT_STEP "
        "run_model49_eval1024_if_and_only_if_run_completed=YES; "
        "do_not_scale_10k_or_100k_until_eval_clean_success_beats_impact"
    )
    lines.append(f"RESULT ppo_clean_reward_audit={verdict}")

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
