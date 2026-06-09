#!/usr/bin/env python3
"""Posthoc audit for the x=0.240 same-near-face 10cm pose runtime.

This parser compares the approved x240 reach-trace run against the previous
x250 same-center reach-trace run. It does not launch IsaacLab or change any
controller, geometry, dataset, training, or robot state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

BASE_TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_trace.json"
BASE_RESULT = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity.json"
BASE_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity_summary.out"

X240_TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"
X240_RESULT = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity.json"
X240_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity_summary.out"
DESIGN_SUMMARY = LOG_DIR / "cube10cm_tap_rl_same_center_vs_same_face_pose_audit_summary.out"

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_same_face_pose_result_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_same_face_pose_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line(path: Path, number: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[number - 1] if 1 <= number <= len(lines) else ""


def _finite(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    if not values:
        raise AssertionError(f"no finite values for {key}")
    return values


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = _finite(rows, key)
    last_step = max(int(row["step"]) for row in rows)
    finals = [float(row[key]) for row in rows if int(row["step"]) == last_step]
    return {"min": min(values), "max": max(values), "final_mean": sum(finals) / len(finals)}


def _inside(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    true_rows = [row for row in rows if bool(row.get(key))]
    true_steps = sorted({int(row["step"]) for row in true_rows})
    return {
        "true_rows": len(true_rows),
        "rate": len(true_rows) / len(rows),
        "first_step": true_steps[0] if true_steps else None,
        "last_step": true_steps[-1] if true_steps else None,
        "unique_steps": len(true_steps),
    }


def _shortfall(best_face_gap: float, band: float = 0.010) -> float:
    if abs(best_face_gap) <= band:
        return 0.0
    if best_face_gap < -band:
        return -band - best_face_gap
    return best_face_gap - band


def _run_metrics(trace_path: Path, result_path: Path) -> dict[str, Any]:
    trace = _load(trace_path)
    result = _load(result_path)
    rows = trace["rows"]
    command = _stats(rows, "command_target_face_gap_m")
    applied = _stats(rows, "applied_joint_target_fk_face_gap_m")
    actual = _stats(rows, "actual_tcp_face_gap_m")
    return {
        "result": result,
        "trace_artifact_type": trace["artifact_type"],
        "action_teacher_dataset": trace["action_teacher_dataset"],
        "row_count": len(rows),
        "step_min": min(int(row["step"]) for row in rows),
        "step_max": max(int(row["step"]) for row in rows),
        "env_ids": sorted({int(row["env_id"]) for row in rows}),
        "command": {**command, "inside": _inside(rows, "command_target_inside_contact_band")},
        "applied": {
            **applied,
            "inside": _inside(rows, "applied_joint_target_fk_inside_contact_band"),
            "shortfall_m": _shortfall(applied["max"]),
        },
        "actual": {
            **actual,
            "inside": _inside(rows, "actual_contact_proxy"),
            "shortfall_m": _shortfall(actual["max"]),
        },
        "applied_fk_err_mm": _stats(rows, "applied_joint_target_fk_err_mm"),
        "follow_rad": _stats(rows, "direct_joint_follow_abs_max_rad"),
        "actual_step_rad": _stats(rows, "actual_joint_step_abs_max_rad"),
        "cube_disp_along_m": _stats(rows, "cube_disp_along_m"),
        "professor_seen": _inside(rows, "professor_physical_reaction_seen"),
        "tap_seen": _inside(rows, "tap_success_seen"),
    }


def main() -> int:
    base = _run_metrics(BASE_TRACE, BASE_RESULT)
    x240 = _run_metrics(X240_TRACE, X240_RESULT)
    actual_shortfall_improvement = base["actual"]["shortfall_m"] - x240["actual"]["shortfall_m"]
    applied_shortfall_improvement = base["applied"]["shortfall_m"] - x240["applied"]["shortfall_m"]

    artifact = {
        "artifact_type": "cube10cm_tap_rl_same_face_pose_result_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "design_summary": str(DESIGN_SUMMARY),
            "base_trace": str(BASE_TRACE),
            "base_result": str(BASE_RESULT),
            "base_summary": str(BASE_SUMMARY),
            "x240_trace": str(X240_TRACE),
            "x240_result": str(X240_RESULT),
            "x240_summary": str(X240_SUMMARY),
        },
        "design_lines": {
            "line2": _line(DESIGN_SUMMARY, 2),
            "line3": _line(DESIGN_SUMMARY, 3),
            "line4": _line(DESIGN_SUMMARY, 4),
            "line5": _line(DESIGN_SUMMARY, 5),
        },
        "runtime_summary_lines": {
            "line3": _line(X240_SUMMARY, 3),
            "line5": _line(X240_SUMMARY, 5),
            "line8": _line(X240_SUMMARY, 8),
            "line9": _line(X240_SUMMARY, 9),
            "line10": _line(X240_SUMMARY, 10),
        },
        "base_x250": base,
        "candidate_x240": x240,
        "comparison": {
            "actual_shortfall_improvement_m": actual_shortfall_improvement,
            "applied_shortfall_improvement_m": applied_shortfall_improvement,
            "strict_contact_unblocked": x240["actual"]["inside"]["true_rows"] > 0,
            "applied_fk_contact_unblocked": x240["applied"]["inside"]["true_rows"] > 0,
            "command_target_still_crosses": x240["command"]["inside"]["true_rows"] > 0,
        },
        "outcome": {
            "verdict": "X240_POSE_IMPROVES_FACE_SHORTFALL_BUT_STILL_NO_CONTACT",
            "contact_gated_positive_control": x240["result"]["rl_contact_gated_positive_control"],
            "professor_physical_reaction_evidence": x240["result"]["professor_physical_reaction_evidence"],
            "diffik_action_dataset": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_same_face_pose_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 design_basis "
        "selected_fixed_cube_x_m=0.240 selected_fixed_cube_y_m=0.000 change=fixed_cube_x_m_only "
        "same_3cm_center_face_rejected_for_plusx=YES",
        "line3 runtime_verdict "
        f"status={x240['result']['status']} professor_physical_reaction_evidence={x240['result']['professor_physical_reaction_evidence']} "
        f"rl_contact_gated_positive_control={x240['result']['rl_contact_gated_positive_control']} "
        f"terminated_count={x240['result']['terminated_count']} truncated_count={x240['result']['truncated_count']}",
        "line4 trace_integrity "
        f"artifact_type={x240['trace_artifact_type']} action_teacher_dataset={x240['action_teacher_dataset']} "
        f"rows={x240['row_count']} steps={x240['step_min']}..{x240['step_max']} env_ids={x240['env_ids']}",
        "line5 command_target "
        f"inside_rows={x240['command']['inside']['true_rows']} inside_unique_steps={x240['command']['inside']['unique_steps']} "
        f"first_step={x240['command']['inside']['first_step']} last_step={x240['command']['inside']['last_step']} "
        f"face_gap_max={x240['command']['max']:.9f}",
        "line6 applied_joint_target_fk "
        f"inside_rows={x240['applied']['inside']['true_rows']} "
        f"x250_shortfall_m={base['applied']['shortfall_m']:.9f} x240_shortfall_m={x240['applied']['shortfall_m']:.9f} "
        f"improvement_m={applied_shortfall_improvement:.9f} "
        f"target_fk_err_mm_final_mean={x240['applied_fk_err_mm']['final_mean']:.9f}",
        "line7 actual_tcp "
        f"inside_rows={x240['actual']['inside']['true_rows']} "
        f"x250_shortfall_m={base['actual']['shortfall_m']:.9f} x240_shortfall_m={x240['actual']['shortfall_m']:.9f} "
        f"improvement_m={actual_shortfall_improvement:.9f}",
        "line8 follow_reaction "
        f"direct_joint_follow_abs_max_rad_max={x240['follow_rad']['max']:.9f} "
        f"actual_joint_step_abs_max_rad_max={x240['actual_step_rad']['max']:.9f} "
        f"cube_disp_along_max_m={x240['cube_disp_along_m']['max']:.9f} "
        f"professor_seen_rate={x240['professor_seen']['rate']:.9f} "
        f"tap_success_seen_rate={x240['tap_seen']['rate']:.9f}",
        "line9 verdict "
        "X240_POSE_IMPROVES_FACE_SHORTFALL_BUT_STILL_NO_CONTACT "
        "command_target_crossed=YES applied_joint_target_fk_crossed=NO actual_tcp_crossed=NO "
        "diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED roarm=BLOCKED",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
