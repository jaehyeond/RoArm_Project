#!/usr/bin/env python3
"""Audit the contact-speed curriculum cube-push PPO run."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
LOG_DIR = RUN_DIR / "ppo_contact_speed_logs/cube_push_contact_speed_50iter_20260526"
STDOUT = RUN_DIR / "ppo_contact_speed_50iter_stdout.out"
STDERR = RUN_DIR / "ppo_contact_speed_50iter_stderr.err"
EXIT_CODE = RUN_DIR / "ppo_contact_speed_50iter_exit_code.txt"
PREV_POLICY_AUDIT_SUMMARY = RUN_DIR / "ppo_smooth_limit_50iter_audit_summary.json"
PREV_TEACHER_AUDIT_SUMMARY = RUN_DIR / "ppo_teacher_warmstart_50iter_audit_summary.json"
OUT = RUN_DIR / "ppo_contact_speed_50iter_audit.out"
SUMMARY = RUN_DIR / "ppo_contact_speed_50iter_audit_summary.json"


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
    prev_policy = json.loads(PREV_POLICY_AUDIT_SUMMARY.read_text())
    prev_teacher = json.loads(PREV_TEACHER_AUDIT_SUMMARY.read_text())
    prev_policy_scalars = prev_policy["scalars"]
    prev_teacher_scalars = prev_teacher["scalars"]
    event_file = sorted(LOG_DIR.glob("events.out.tfevents.*"))[0]
    model0 = LOG_DIR / "model_0.pt"
    model49 = LOG_DIR / "model_49.pt"
    code = _exit_code()

    required = [
        "Episode/cube_push_disp_along_m",
        "Episode/cube_push_disp_xy_m",
        "Episode/cube_push_target_xy_dist_m",
        "Episode/cube_push_speed_mps",
        "Episode/cube_push_speed_over_0p5_rate",
        "Episode/cube_push_controlled_rate",
        "Episode/cube_push_impact_rate",
        "Episode/cube_push_low_motion_rate",
        "Episode/cube_push_far_target_rate",
        "Episode/cube_push_terminal_impact_rate",
        "Episode/cube_push_success_rate",
        "Episode/cube_push_grasped_marker_rate",
        "Episode/cube_push_ik_endpoint_reset_rate",
        "Episode/cube_push_ik_reset_err_mm",
        "Episode/cube_push_joint_delta_abs_mean",
        "Episode/cube_push_contact_slowdown_mean",
        "Episode/cube_push_teacher_blend_mean",
        "Episode/cube_push_teacher_goal_ok_rate",
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
    target_dist = scalars["Episode/cube_push_target_xy_dist_m"]
    speed = scalars["Episode/cube_push_speed_mps"]
    speed_over = scalars["Episode/cube_push_speed_over_0p5_rate"]
    joint_delta = scalars["Episode/cube_push_joint_delta_abs_mean"]
    slowdown = scalars["Episode/cube_push_contact_slowdown_mean"]
    teacher_blend = scalars["Episode/cube_push_teacher_blend_mean"]
    grasp = scalars["Episode/cube_push_grasped_marker_rate"]
    ik_rate = scalars["Episode/cube_push_ik_endpoint_reset_rate"]

    run_completed = (
        code == 0
        and model49.exists()
        and train_iters == 50
        and ik_rate["last"] > 0.99
        and grasp["max"] == 0.0
        and teacher_blend["max"] == 0.0
        and joint_delta["max"] > 0.0
        and slowdown["last"] < 0.90
    )
    train_signal = (
        run_completed
        and impact["last"] < 0.03
        and far_target["last"] < 0.03
        and terminal_impact["last"] < 0.05
        and speed_over["last"] < 0.15
        and target_dist["last"] < 0.08
        and controlled["last"] > 0.40
    )
    verdict = "CONTACT_SPEED_RUN_BLOCKED"
    if train_signal:
        verdict = "CONTACT_SPEED_TRAIN_SIGNAL_PRESENT_NEEDS_MODEL49_EVAL"
    elif run_completed:
        verdict = "CONTACT_SPEED_RAN_WEAK_NEEDS_MODEL49_EVAL"

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
        "model_0_md5": _md5(model0) if model0.exists() else None,
        "model_49": str(model49),
        "model_49_md5": _md5(model49) if model49.exists() else None,
        "previous_policy_verdict": prev_policy["verdict"],
        "previous_teacher_verdict": prev_teacher["verdict"],
        "run_completed": run_completed,
        "train_signal": train_signal,
        "verdict": verdict,
        "scalars": scalars,
    }

    lines = [
        (
            f"CONTACT_SPEED_INPUT exit_code={code} stdout={STDOUT} stdout_md5={summary['stdout_md5']} "
            f"stderr={STDERR} stderr_md5={summary['stderr_md5']}"
        ),
        (
            f"CONTACT_SPEED_ARTIFACTS log_dir={LOG_DIR} event_file={event_file} "
            f"event_md5={summary['event_file_md5']} model_0_exists={'YES' if model0.exists() else 'NO'} "
            f"model_49_exists={'YES' if model49.exists() else 'NO'} model_49_md5={summary['model_49_md5']}"
        ),
        f"ITERATION_COVERAGE event_train_mean_reward_count={train_iters}",
    ]
    for tag in required:
        lines.append(_line(tag, scalars))
    lines.append(
        "PREV_POLICY_SMOOTH_COMPARISON "
        f"prev_verdict={prev_policy['verdict']} "
        f"prev_speed_last={prev_policy_scalars['Episode/cube_push_speed_mps']['last']:.9f} "
        f"new_speed_last={speed['last']:.9f} "
        f"prev_speed_over_0p5_last={prev_policy_scalars['Episode/cube_push_speed_over_0p5_rate']['last']:.9f} "
        f"new_speed_over_0p5_last={speed_over['last']:.9f} "
        f"prev_controlled_last={prev_policy_scalars['Episode/cube_push_controlled_rate']['last']:.9f} "
        f"new_controlled_last={controlled['last']:.9f} "
        f"prev_impact_last={prev_policy_scalars['Episode/cube_push_impact_rate']['last']:.9f} "
        f"new_impact_last={impact['last']:.9f}"
    )
    lines.append(
        "PREV_TEACHER_ASSISTED_COMPARISON "
        f"prev_verdict={prev_teacher['verdict']} "
        f"prev_controlled_last={prev_teacher_scalars['Episode/cube_push_controlled_rate']['last']:.9f} "
        f"new_controlled_last={controlled['last']:.9f} "
        f"prev_impact_last={prev_teacher_scalars['Episode/cube_push_impact_rate']['last']:.9f} "
        f"new_impact_last={impact['last']:.9f} "
        f"prev_low_motion_last={prev_teacher_scalars['Episode/cube_push_low_motion_rate']['last']:.9f} "
        f"new_low_motion_last={low_motion['last']:.9f} "
        f"prev_success_last={prev_teacher_scalars['Episode/cube_push_success_rate']['last']:.9f} "
        f"new_success_last={success['last']:.9f}"
    )
    lines.append(
        "CRITICAL_INTERPRETATION "
        "policy_only_contact_speed_curriculum=YES teacher_assist=NO "
        "training_loop_ran=YES learned_policy_success=NO evaluation_rollout=NO "
        "dataset_generation=NO track_a_grasp_success=NO"
    )
    lines.append(f"RESULT ppo_contact_speed_audit={verdict}")

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
