#!/usr/bin/env python3
"""Audit the first target-path contract for cube10cm tap.

Local/posthoc only. This checks whether the commanded TCP path represents a
near-face tap/push goal or a far-face-through goal.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"
SANITY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity.json"
HARNESS = ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_target_path_first_button_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_target_path_first_button_audit_summary.out"


def _line(path: Path, needle: str, *, after: int = 0) -> int:
    for idx, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if idx <= after:
            continue
        if needle in text:
            return idx
    raise ValueError(f"needle not found: {needle!r}")


def _avg_step(rows: list[dict[str, Any]], step: int, key: str) -> float:
    vals = [float(row[key]) for row in rows if int(row["step"]) == step]
    if not vals:
        raise ValueError(f"missing step={step} key={key}")
    return mean(vals)


def _best(rows: list[dict[str, Any]], key: str) -> tuple[int, float]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(float(row[key]))
    step, val = max(((step, mean(vals)) for step, vals in by_step.items()), key=lambda item: item[1])
    return step, val


def main() -> int:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    sanity = json.loads(SANITY.read_text(encoding="utf-8"))
    rows = trace["rows"]

    cube_size = float(sanity["cube_size_m"])
    precontact = float(sanity["precontact_clearance_m"])
    goal_push = float(sanity["goal_push_m"])
    half_along = 0.5 * cube_size
    legacy_final_face_gap = cube_size + goal_push
    near_face_final_face_gap = goal_push
    legacy_path_face_gap_delta = precontact + cube_size + goal_push
    near_face_path_face_gap_delta = precontact + goal_push
    path_length_ratio = legacy_path_face_gap_delta / near_face_path_face_gap_delta

    final_step = max(int(row["step"]) for row in rows)
    command_initial = _avg_step(rows, 0, "command_target_face_gap_m")
    command_final = _avg_step(rows, final_step, "command_target_face_gap_m")
    applied_initial = _avg_step(rows, 0, "applied_joint_target_fk_face_gap_m")
    applied_final = _avg_step(rows, final_step, "applied_joint_target_fk_face_gap_m")
    actual_initial = _avg_step(rows, 0, "actual_tcp_face_gap_m")
    actual_final = _avg_step(rows, final_step, "actual_tcp_face_gap_m")
    fk_err_initial = _avg_step(rows, 0, "applied_joint_target_fk_err_mm")
    fk_err_final = _avg_step(rows, final_step, "applied_joint_target_fk_err_mm")
    applied_best_step, applied_best = _best(rows, "applied_joint_target_fk_face_gap_m")
    actual_best_step, actual_best = _best(rows, "actual_tcp_face_gap_m")
    command_inside_steps = sorted({int(row["step"]) for row in rows if bool(row["command_target_inside_contact_band"])})
    applied_inside_rows = sum(1 for row in rows if bool(row["applied_joint_target_fk_inside_contact_band"]))
    actual_inside_rows = sum(1 for row in rows if bool(row["actual_contact_proxy"]))

    external_pre_line = _line(HARNESS, "pre[:2] -= push_dir[env_id]")
    builtin_pre_line = _line(HARNESS, "pre_w[:, 0:2] = cube_w[:, 0:2]")
    line_refs = {
        "external_legacy_pre": external_pre_line,
        "external_near_face_branch": _line(HARNESS, 'if args.target_path_mode == "near_face_goal"', after=external_pre_line - 1),
        "external_legacy_through": _line(HARNESS, "through[:2] += push_dir[env_id] * (half_along", after=external_pre_line - 1),
        "builtin_legacy_pre": builtin_pre_line,
        "builtin_near_face_branch": _line(HARNESS, 'if args.target_path_mode == "near_face_goal"', after=builtin_pre_line - 1),
        "builtin_legacy_through": _line(
            HARNESS,
            "through_w[:, 0:2] = cube_w[:, 0:2] + push_dir * (half_along",
            after=builtin_pre_line - 1,
        ),
        "parser_target_path_mode": _line(HARNESS, "--target_path_mode"),
    }

    result = {
        "artifact_type": "cube10cm_tap_rl_target_path_first_button_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "trace": str(TRACE.relative_to(ROOT)),
            "sanity": str(SANITY.relative_to(ROOT)),
            "harness": str(HARNESS.relative_to(ROOT)),
        },
        "line_refs": line_refs,
        "constants": {
            "cube_size_m": cube_size,
            "half_along_m": half_along,
            "precontact_clearance_m": precontact,
            "goal_push_m": goal_push,
            "legacy_far_face_final_face_gap_m": legacy_final_face_gap,
            "near_face_goal_final_face_gap_m": near_face_final_face_gap,
            "legacy_path_face_gap_delta_m": legacy_path_face_gap_delta,
            "near_face_path_face_gap_delta_m": near_face_path_face_gap_delta,
            "legacy_over_near_face_path_ratio": path_length_ratio,
        },
        "existing_x240_trace": {
            "final_step": final_step,
            "command_initial_face_gap_m": command_initial,
            "command_final_face_gap_m": command_final,
            "command_matches_legacy_far_face": abs(command_final - legacy_final_face_gap) < 5.0e-6,
            "command_matches_near_face_goal": abs(command_final - near_face_final_face_gap) < 5.0e-6,
            "command_inside_first_step": command_inside_steps[0],
            "command_inside_last_step": command_inside_steps[-1],
            "applied_inside_rows": applied_inside_rows,
            "actual_inside_rows": actual_inside_rows,
            "applied_initial_face_gap_m": applied_initial,
            "applied_best_step": applied_best_step,
            "applied_best_face_gap_m": applied_best,
            "applied_final_face_gap_m": applied_final,
            "actual_initial_face_gap_m": actual_initial,
            "actual_best_step": actual_best_step,
            "actual_best_face_gap_m": actual_best,
            "actual_final_face_gap_m": actual_final,
            "applied_fk_err_initial_mm": fk_err_initial,
            "applied_fk_err_final_mm": fk_err_final,
        },
        "verdict": "FIRST_BUTTON_MISMATCH_LEGACY_FAR_FACE_THROUGH_TARGET_FOR_10CM_TAP",
        "next_local_step": (
            "Use the default-off target_path_mode=near_face_goal candidate for one explicit tiny runtime only "
            "after approval. Keep contact gate strict and keep dataset/RL/RoArm blocked."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = [
        "line1 artifact=cube10cm_tap_rl_target_path_first_button_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 code_contract "
            f"external_pre_line={line_refs['external_legacy_pre']} "
            f"external_near_face_branch_line={line_refs['external_near_face_branch']} "
            f"external_legacy_through_line={line_refs['external_legacy_through']} "
            f"builtin_pre_line={line_refs['builtin_legacy_pre']} "
            f"builtin_near_face_branch_line={line_refs['builtin_near_face_branch']} "
            f"builtin_legacy_through_line={line_refs['builtin_legacy_through']} "
            f"parser_target_path_mode_line={line_refs['parser_target_path_mode']}"
        ),
        (
            "line3 target_path_math "
            f"cube_size_m={cube_size:.9f} half_along_m={half_along:.9f} "
            f"precontact_m={precontact:.9f} goal_push_m={goal_push:.9f} "
            f"legacy_final_face_gap_m={legacy_final_face_gap:.9f} "
            f"near_face_goal_final_face_gap_m={near_face_final_face_gap:.9f} "
            f"legacy_path_delta_m={legacy_path_face_gap_delta:.9f} "
            f"near_face_path_delta_m={near_face_path_face_gap_delta:.9f} "
            f"path_length_ratio={path_length_ratio:.9f}"
        ),
        (
            "line4 existing_x240_command "
            f"command_initial_face_gap_m={command_initial:.9f} "
            f"command_final_face_gap_m={command_final:.9f} "
            f"matches_legacy_far_face={str(abs(command_final - legacy_final_face_gap) < 5.0e-6).upper()} "
            f"matches_near_face_goal={str(abs(command_final - near_face_final_face_gap) < 5.0e-6).upper()} "
            f"command_inside_steps={command_inside_steps[0]}..{command_inside_steps[-1]}"
        ),
        (
            "line5 consequence "
            f"applied_inside_rows={applied_inside_rows} actual_inside_rows={actual_inside_rows} "
            f"applied_best_step={applied_best_step} applied_best_face_gap_m={applied_best:.9f} "
            f"applied_final_face_gap_m={applied_final:.9f} "
            f"actual_best_step={actual_best_step} actual_best_face_gap_m={actual_best:.9f} "
            f"actual_final_face_gap_m={actual_final:.9f} "
            f"applied_fk_err_initial_mm={fk_err_initial:.9f} "
            f"applied_fk_err_final_mm={fk_err_final:.9f}"
        ),
        (
            "line6 verdict FIRST_BUTTON_MISMATCH_LEGACY_FAR_FACE_THROUGH_TARGET_FOR_10CM_TAP "
            "legacy_target_not_same_as_touch_near_face=YES contact_gate_relaxation_unblock=NO "
            "same_contract_more_steps_unblock=NO"
        ),
        (
            "line7 next DEFAULT_OFF_TARGET_PATH_MODE_NEAR_FACE_GOAL_DESIGNED "
            "runtime_requires_explicit_APPROVAL=YES dataset_rl_roarm=BLOCKED"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
