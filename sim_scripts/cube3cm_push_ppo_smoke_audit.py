#!/usr/bin/env python3
"""Audit the no-attach cube-push PPO smoke outputs.

This is posthoc only. It reads stdout/stderr, checkpoints, and TensorBoard
events from the smoke run and writes a compact verdict.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
LOG_DIR = RUN_DIR / "ppo_smoke_logs/cube_push_no_attach_smoke_20260526"
STDOUT = RUN_DIR / "ppo_smoke_stdout.out"
STDERR = RUN_DIR / "ppo_smoke_stderr.err"
EXIT_CODE = RUN_DIR / "ppo_smoke_exit_code.txt"
OUT = RUN_DIR / "ppo_smoke_audit.out"
SUMMARY = RUN_DIR / "ppo_smoke_audit_summary.json"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_line(path: Path, needle: str) -> tuple[int, str]:
    for idx, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if needle in line:
            return idx, line.strip()
    return 0, ""


def _exit_code() -> int | None:
    text = EXIT_CODE.read_text().strip()
    m = re.search(r"exit_code:(-?\d+)", text)
    return int(m.group(1)) if m else None


def _event_scalars() -> dict[str, dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:  # pragma: no cover - depends on conda env
        return {"__tensorboard_error__": {"error": repr(exc)}}

    event_files = sorted(LOG_DIR.glob("events.out.tfevents.*"))
    if not event_files:
        return {"__event_file_error__": {"error": "no event file"}}
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
        }
    return out


def _scalar_line(tag: str, scalars: dict[str, dict[str, Any]]) -> str:
    item = scalars.get(tag)
    if not item:
        return f"EVENT_SCALAR tag={tag} missing=YES"
    return (
        f"EVENT_SCALAR tag={tag} n={item['n']} "
        f"first_step={item['first_step']} first={item['first']:.9f} "
        f"last_step={item['last_step']} last={item['last']:.9f}"
    )


def main() -> int:
    scalars = _event_scalars()
    model0 = LOG_DIR / "model_0.pt"
    model9 = LOG_DIR / "model_9.pt"
    event_files = sorted(LOG_DIR.glob("events.out.tfevents.*"))
    code = _exit_code()

    stdout_text = STDOUT.read_text(errors="replace")
    visible_iters = sorted({int(m.group(1)) for m in re.finditer(r"Learning iteration\s+(\d+)/10", stdout_text)})
    event_iteration_count = scalars.get("Train/mean_reward", {}).get("n", 0)

    controlled = scalars.get("Episode/cube_push_controlled_rate", {})
    low_motion = scalars.get("Episode/cube_push_low_motion_rate", {})
    success = scalars.get("Episode/cube_push_success_rate", {})
    impact = scalars.get("Episode/cube_push_impact_rate", {})
    grasp = scalars.get("Episode/cube_push_grasped_marker_rate", {})
    disp_along = scalars.get("Episode/cube_push_disp_along_m", {})
    target_dist = scalars.get("Episode/cube_push_target_xy_dist_m", {})

    signal_present = (
        code == 0
        and model9.exists()
        and event_iteration_count == 10
        and controlled.get("last", 0.0) > controlled.get("first", 0.0)
        and low_motion.get("last", 1.0) < low_motion.get("first", 1.0)
    )
    weak_or_unproven = (
        success.get("last", 0.0) < 0.05
        or disp_along.get("last", 0.0) < 0.005
        or target_dist.get("last", 0.0) >= target_dist.get("first", 0.0)
    )
    verdict = "RUNTIME_BLOCKED"
    if code == 0 and model9.exists() and event_iteration_count == 10:
        verdict = "SMOKE_SIGNAL_PRESENT_BUT_WEAK_NO_SUCCESS_CLAIM" if signal_present and weak_or_unproven else "TRAINING_RAN_BUT_NO_LEARNING_CLAIM"

    summary = {
        "stdout": str(STDOUT),
        "stdout_md5": _md5(STDOUT),
        "stderr": str(STDERR),
        "stderr_md5": _md5(STDERR),
        "exit_code_file": str(EXIT_CODE),
        "exit_code": code,
        "log_dir": str(LOG_DIR),
        "event_file": str(event_files[0]) if event_files else None,
        "event_file_md5": _md5(event_files[0]) if event_files else None,
        "model_0": str(model0) if model0.exists() else None,
        "model_0_md5": _md5(model0) if model0.exists() else None,
        "model_9": str(model9) if model9.exists() else None,
        "model_9_md5": _md5(model9) if model9.exists() else None,
        "visible_stdout_iterations": visible_iters,
        "event_iteration_count": event_iteration_count,
        "verdict": verdict,
        "scalars": scalars,
    }

    lines: list[str] = []
    lines.append(
        f"PPO_SMOKE_INPUT stdout={STDOUT} stdout_md5={summary['stdout_md5']} "
        f"stderr={STDERR} stderr_md5={summary['stderr_md5']} exit_code={code}"
    )
    lines.append(
        f"PPO_SMOKE_ARTIFACTS log_dir={LOG_DIR} "
        f"event_file={summary['event_file']} event_md5={summary['event_file_md5']} "
        f"model_0_exists={'YES' if model0.exists() else 'NO'} model_9_exists={'YES' if model9.exists() else 'NO'} "
        f"model_9_md5={summary['model_9_md5']}"
    )
    for needle in (
        "scope=no_attach_cube_push",
        "action_semantics=normalized_joint_delta",
        "Environment device",
        "Actor MLP",
        "Learning iteration 0/10",
        "Learning iteration 4/10",
    ):
        line_no, text = _find_line(STDOUT, needle)
        lines.append(f"STDOUT_REF needle=\"{needle}\" line={line_no} text=\"{text}\"")
    lines.append(
        f"ITERATION_COVERAGE stdout_visible={visible_iters} event_train_mean_reward_count={event_iteration_count}"
    )
    for tag in (
        "Episode/cube_push_disp_along_m",
        "Episode/cube_push_disp_xy_m",
        "Episode/cube_push_target_xy_dist_m",
        "Episode/cube_push_controlled_rate",
        "Episode/cube_push_impact_rate",
        "Episode/cube_push_low_motion_rate",
        "Episode/cube_push_success_rate",
        "Episode/cube_push_grasped_marker_rate",
        "Train/mean_reward",
        "Train/mean_episode_length",
    ):
        lines.append(_scalar_line(tag, scalars))
    lines.append(
        "CRITICAL_INTERPRETATION "
        "training_loop_ran=YES learned_policy_success=NO evaluation_rollout=NO dataset_generation=NO "
        "track_a_grasp_success=NO"
    )
    lines.append(f"RESULT ppo_smoke_audit={verdict}")

    OUT.write_text("\n".join(lines) + "\n")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
