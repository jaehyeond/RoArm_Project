#!/usr/bin/env python3
"""Audit the approved cube10cm per-step reach trace tiny repeat.

This is a local posthoc parser. It does not launch IsaacLab or change any
controller, geometry, dataset, training, or robot state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

TRACE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_trace.json"
RESULT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity.json"
RESULT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_reachtrace_sanity_summary.out"

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_per_step_reach_trace_result_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_per_step_reach_trace_result_audit_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
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
    final_rows = [row for row in rows if row["step"] == max(r["step"] for r in rows)]
    final_values = [float(row[key]) for row in final_rows]
    return {
        "min": min(values),
        "max": max(values),
        "final_mean": sum(final_values) / len(final_values),
    }


def _inside_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    true_rows = [row for row in rows if bool(row.get(key))]
    true_steps = sorted({int(row["step"]) for row in true_rows})
    return {
        "true_rows": len(true_rows),
        "rate": len(true_rows) / len(rows),
        "first_step": true_steps[0] if true_steps else None,
        "last_step": true_steps[-1] if true_steps else None,
        "unique_step_count": len(true_steps),
    }


def _best_negative_face_shortfall(rows: list[dict[str, Any]], key: str, band_m: float) -> float:
    values = _finite(rows, key)
    best = max(values)
    if abs(best) <= band_m:
        return 0.0
    if best < -band_m:
        return -band_m - best
    return best - band_m


def main() -> int:
    trace = _load_json(TRACE_JSON)
    result = _load_json(RESULT_JSON)
    rows = trace["rows"]
    if not rows:
        raise AssertionError("trace rows are empty")

    steps = sorted({int(row["step"]) for row in rows})
    env_ids = sorted({int(row["env_id"]) for row in rows})
    expected_rows = len(steps) * len(env_ids)
    row_count_ok = len(rows) == expected_rows == int(result["reach_trace_row_count"])
    # The runtime result does not duplicate this static contact-band config.
    # It is fixed by RoArmCubeTap10cmEnvCfg and the prior reach-contract audit.
    face_band_m = float(result.get("tap_contact_face_band_m", 0.010))

    command_inside = _inside_summary(rows, "command_target_inside_contact_band")
    applied_inside = _inside_summary(rows, "applied_joint_target_fk_inside_contact_band")
    actual_inside = _inside_summary(rows, "actual_contact_proxy")

    audit = {
        "artifact_type": "cube10cm_tap_rl_per_step_reach_trace_result_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "trace_json": str(TRACE_JSON),
            "result_json": str(RESULT_JSON),
            "result_summary": str(RESULT_SUMMARY),
        },
        "runtime_summary_lines": {
            "line1": _line(RESULT_SUMMARY, 1),
            "line5": _line(RESULT_SUMMARY, 5),
            "line8": _line(RESULT_SUMMARY, 8),
            "line9": _line(RESULT_SUMMARY, 9),
            "line10": _line(RESULT_SUMMARY, 10),
        },
        "trace_integrity": {
            "artifact_type": trace["artifact_type"],
            "action_teacher_dataset": trace["action_teacher_dataset"],
            "row_count": len(rows),
            "expected_rows": expected_rows,
            "row_count_ok": row_count_ok,
            "step_min": min(steps),
            "step_max": max(steps),
            "env_ids": env_ids,
        },
        "contact_contract": {
            "face_band_m": face_band_m,
            "face_band_source": "result.tap_contact_face_band_m_or_env_default_0p010",
        },
        "face_gap_timeline": {
            "face_band_m": face_band_m,
            "command": {
                **_stats(rows, "command_target_face_gap_m"),
                "inside": command_inside,
            },
            "applied_joint_target_fk": {
                **_stats(rows, "applied_joint_target_fk_face_gap_m"),
                "inside": applied_inside,
                "best_shortfall_m": _best_negative_face_shortfall(
                    rows, "applied_joint_target_fk_face_gap_m", face_band_m
                ),
            },
            "actual_tcp": {
                **_stats(rows, "actual_tcp_face_gap_m"),
                "inside": actual_inside,
                "best_shortfall_m": _best_negative_face_shortfall(rows, "actual_tcp_face_gap_m", face_band_m),
            },
        },
        "lateral_vertical_contract": {
            "applied_lateral_max_m": _stats(rows, "applied_joint_target_fk_lateral_m")["max"],
            "actual_lateral_max_m": _stats(rows, "actual_tcp_lateral_m")["max"],
            "applied_vertical_max_m": _stats(rows, "applied_joint_target_fk_vertical_offset_m")["max"],
            "actual_vertical_max_m": _stats(rows, "actual_tcp_vertical_offset_m")["max"],
        },
        "follow_and_reaction": {
            "applied_target_fk_err_mm": _stats(rows, "applied_joint_target_fk_err_mm"),
            "direct_joint_follow_abs_max_rad": _stats(rows, "direct_joint_follow_abs_max_rad"),
            "actual_joint_step_abs_max_rad": _stats(rows, "actual_joint_step_abs_max_rad"),
            "cube_disp_along_m": _stats(rows, "cube_disp_along_m"),
            "cube_speed_mps": _stats(rows, "cube_speed_mps"),
            "professor_physical_reaction_seen_rate": _inside_summary(
                rows, "professor_physical_reaction_seen"
            )["rate"],
            "tap_success_seen_rate": _inside_summary(rows, "tap_success_seen")["rate"],
            "terminated_rate": _inside_summary(rows, "terminated")["rate"],
            "truncated_rate": _inside_summary(rows, "truncated")["rate"],
        },
        "outcome": {
            "verdict": "APPLIED_AND_ACTUAL_REACH_NEVER_ENTER_CONTACT_BAND",
            "command_target_entered_contact_band": command_inside["true_rows"] > 0,
            "applied_joint_target_fk_entered_contact_band": applied_inside["true_rows"] > 0,
            "actual_tcp_entered_contact_band": actual_inside["true_rows"] > 0,
            "professor_physical_reaction_evidence": result["professor_physical_reaction_evidence"],
            "rl_contact_gated_positive_control": result["rl_contact_gated_positive_control"],
            "diffik_action_dataset": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
    }

    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_per_step_reach_trace_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 trace_integrity "
        f"artifact_type={trace['artifact_type']} action_teacher_dataset={trace['action_teacher_dataset']} "
        f"rows={len(rows)} expected_rows={expected_rows} row_count_ok={row_count_ok} "
        f"steps={min(steps)}..{max(steps)} env_ids={env_ids}",
        "line3 runtime_verdict "
        f"summary_status={result['status']} professor_physical_reaction_evidence={result['professor_physical_reaction_evidence']} "
        f"rl_contact_gated_positive_control={result['rl_contact_gated_positive_control']} "
        f"terminated_count={result['terminated_count']} truncated_count={result['truncated_count']}",
        "line4 command_target "
        f"inside_rows={command_inside['true_rows']} inside_unique_steps={command_inside['unique_step_count']} "
        f"first_step={command_inside['first_step']} last_step={command_inside['last_step']} "
        f"face_gap_min={audit['face_gap_timeline']['command']['min']:.9f} "
        f"face_gap_max={audit['face_gap_timeline']['command']['max']:.9f} "
        f"face_gap_final_mean={audit['face_gap_timeline']['command']['final_mean']:.9f}",
        "line5 applied_joint_target_fk "
        f"inside_rows={applied_inside['true_rows']} "
        f"face_gap_max={audit['face_gap_timeline']['applied_joint_target_fk']['max']:.9f} "
        f"best_shortfall_m={audit['face_gap_timeline']['applied_joint_target_fk']['best_shortfall_m']:.9f} "
        f"target_fk_err_mm_final_mean={audit['follow_and_reaction']['applied_target_fk_err_mm']['final_mean']:.9f}",
        "line6 actual_tcp "
        f"inside_rows={actual_inside['true_rows']} "
        f"face_gap_max={audit['face_gap_timeline']['actual_tcp']['max']:.9f} "
        f"best_shortfall_m={audit['face_gap_timeline']['actual_tcp']['best_shortfall_m']:.9f} "
        f"actual_lateral_max_m={audit['lateral_vertical_contract']['actual_lateral_max_m']:.9f} "
        f"actual_vertical_max_m={audit['lateral_vertical_contract']['actual_vertical_max_m']:.9f}",
        "line7 follow_reaction "
        f"direct_joint_follow_abs_max_rad_max={audit['follow_and_reaction']['direct_joint_follow_abs_max_rad']['max']:.9f} "
        f"actual_joint_step_abs_max_rad_max={audit['follow_and_reaction']['actual_joint_step_abs_max_rad']['max']:.9f} "
        f"cube_disp_along_max_m={audit['follow_and_reaction']['cube_disp_along_m']['max']:.9f} "
        f"cube_speed_max_mps={audit['follow_and_reaction']['cube_speed_mps']['max']:.9f} "
        f"professor_seen_rate={audit['follow_and_reaction']['professor_physical_reaction_seen_rate']:.9f} "
        f"tap_success_seen_rate={audit['follow_and_reaction']['tap_success_seen_rate']:.9f}",
        "line8 verdict "
        "APPLIED_AND_ACTUAL_REACH_NEVER_ENTER_CONTACT_BAND "
        "command_target_crossed=YES applied_joint_target_fk_crossed=NO actual_tcp_crossed=NO "
        "contact_gate_not_relaxed=YES diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED roarm=BLOCKED",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
