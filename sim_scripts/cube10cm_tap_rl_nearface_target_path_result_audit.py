#!/usr/bin/env python3
"""Posthoc audit for the cube10cm near-face target-path tiny runtime.

Compares the approved near-face run against the previous x240 legacy
far-face-through trace. Local/posthoc only; no Isaac runtime is launched here.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

LEGACY_SANITY = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_sanity.json"
LEGACY_TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"

FAILED_SANITY = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_sanity.json"
)
FAILED_STDERR = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_stderr.out"
)
NEARFACE_SANITY = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_rerun1_sanity.json"
)
NEARFACE_TRACE = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_rerun1_trace.json"
)
NEARFACE_SUMMARY = (
    LOG_DIR
    / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_reachtrace_rerun1_sanity_summary.out"
)

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_nearface_target_path_result_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_nearface_target_path_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _avg_by_step(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["step"]), []).append(float(row[key]))
    return {step: mean(vals) for step, vals in grouped.items()}


def _trace_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    command_inside_steps = sorted({int(row["step"]) for row in rows if bool(row["command_target_inside_contact_band"])})
    applied_inside_rows = sum(1 for row in rows if bool(row["applied_joint_target_fk_inside_contact_band"]))
    actual_inside_rows = sum(1 for row in rows if bool(row["actual_contact_proxy"]))
    command = _avg_by_step(rows, "command_target_face_gap_m")
    applied = _avg_by_step(rows, "applied_joint_target_fk_face_gap_m")
    actual = _avg_by_step(rows, "actual_tcp_face_gap_m")
    fk_err = _avg_by_step(rows, "applied_joint_target_fk_err_mm")
    final_step = max(command)
    applied_best_step, applied_best = max(applied.items(), key=lambda item: item[1])
    actual_best_step, actual_best = max(actual.items(), key=lambda item: item[1])
    return {
        "row_count": len(rows),
        "final_step": final_step,
        "command_initial_face_gap_m": command[0],
        "command_final_face_gap_m": command[final_step],
        "command_inside_rows": sum(1 for row in rows if bool(row["command_target_inside_contact_band"])),
        "command_inside_unique_steps": len(command_inside_steps),
        "command_inside_first_step": command_inside_steps[0] if command_inside_steps else None,
        "command_inside_last_step": command_inside_steps[-1] if command_inside_steps else None,
        "applied_inside_rows": applied_inside_rows,
        "actual_inside_rows": actual_inside_rows,
        "applied_best_step": applied_best_step,
        "applied_best_face_gap_m": applied_best,
        "applied_best_shortfall_m": max(0.0, -0.010 - applied_best),
        "applied_final_face_gap_m": applied[final_step],
        "actual_best_step": actual_best_step,
        "actual_best_face_gap_m": actual_best,
        "actual_best_shortfall_m": max(0.0, -0.010 - actual_best),
        "actual_final_face_gap_m": actual[final_step],
        "applied_fk_err_initial_mm": fk_err[0],
        "applied_fk_err_final_mm": fk_err[final_step],
    }


def _trace_stat(sanity: dict[str, Any], key: str, stat: str) -> float:
    return float(sanity["controller_trace_stats"][key][stat])


def main() -> int:
    legacy_sanity = _load(LEGACY_SANITY)
    legacy_trace = _load(LEGACY_TRACE)
    failed_sanity = _load(FAILED_SANITY) if FAILED_SANITY.exists() else {}
    near_sanity = _load(NEARFACE_SANITY)
    near_trace = _load(NEARFACE_TRACE)

    legacy_stats = _trace_stats(legacy_trace["rows"])
    near_stats = _trace_stats(near_trace["rows"])

    first_launch_blocker = failed_sanity.get("blocker", "NONE")
    first_launch_error = failed_sanity.get("error", "NONE")
    first_launch_cuda_error = "No CUDA devices found" in FAILED_STDERR.read_text(encoding="utf-8") if FAILED_STDERR.exists() else False

    result = {
        "artifact_type": "cube10cm_tap_rl_nearface_target_path_result_audit_v1",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "legacy_sanity": str(LEGACY_SANITY.relative_to(ROOT)),
            "legacy_trace": str(LEGACY_TRACE.relative_to(ROOT)),
            "first_failed_sanity": str(FAILED_SANITY.relative_to(ROOT)),
            "nearface_sanity": str(NEARFACE_SANITY.relative_to(ROOT)),
            "nearface_trace": str(NEARFACE_TRACE.relative_to(ROOT)),
            "nearface_summary": str(NEARFACE_SUMMARY.relative_to(ROOT)),
        },
        "first_launch": {
            "blocker": first_launch_blocker,
            "error": first_launch_error,
            "cuda_device_error_seen": first_launch_cuda_error,
            "interpretation": "SANDBOX_OR_PYTHONPATH_LAUNCH_FAILURE_NOT_PHYSICS_RESULT",
        },
        "nearface_runtime": {
            "status": near_sanity["status"],
            "target_path_mode": near_sanity["target_path_mode"],
            "steps_executed": int(near_sanity["steps_executed"]),
            "max_steps": int(near_sanity["max_steps"]),
            "terminated_count": int(near_sanity["terminated_count"]),
            "truncated_count": int(near_sanity["truncated_count"]),
            "reach_trace_row_count": int(near_sanity["reach_trace_row_count"]),
            "rl_contact_gated_positive_control": near_sanity["rl_contact_gated_positive_control"],
            "professor_physical_reaction_evidence": near_sanity["professor_physical_reaction_evidence"],
            "contact_seen": near_sanity["last_log"]["cube_tap_contact_seen_rate"],
            "tap_success": near_sanity["last_log"]["cube_tap_success_rate"],
            "professor_seen": near_sanity["last_log"]["cube_tap_professor_physical_reaction_seen_rate"],
            "overshoot_seen": near_sanity["last_log"]["cube_tap_overshoot_seen_rate"],
            "max_disp_along_m": near_sanity["last_log"]["cube_tap_max_disp_along_m"],
            "max_speed_mps": near_sanity["last_log"]["cube_tap_max_speed_mps"],
        },
        "legacy_trace_stats": legacy_stats,
        "nearface_trace_stats": near_stats,
        "comparison": {
            "command_final_gap_delta_m": near_stats["command_final_face_gap_m"]
            - legacy_stats["command_final_face_gap_m"],
            "applied_fk_err_final_delta_mm": near_stats["applied_fk_err_final_mm"]
            - legacy_stats["applied_fk_err_final_mm"],
            "actual_best_shortfall_delta_m": near_stats["actual_best_shortfall_m"]
            - legacy_stats["actual_best_shortfall_m"],
            "target_fk_err_final_delta_mm": _trace_stat(near_sanity, "closed_loop_target_fk_err_mm_mean", "final")
            - _trace_stat(legacy_sanity, "closed_loop_target_fk_err_mm_mean", "final"),
            "direct_follow_final_delta_rad": _trace_stat(near_sanity, "direct_joint_follow_abs_max_rad", "final")
            - _trace_stat(legacy_sanity, "direct_joint_follow_abs_max_rad", "final"),
            "actual_step_final_delta_rad": _trace_stat(near_sanity, "direct_actual_joint_step_abs_max_rad", "final")
            - _trace_stat(legacy_sanity, "direct_actual_joint_step_abs_max_rad", "final"),
        },
        "verdict": "NEAR_FACE_TARGET_PATH_APPLIED_BUT_STRICT_CONTACT_STILL_FAILS_ACTUAL_TCP_PRECONTACT",
        "next_local_step": (
            "Do not relax the contact gate or start dataset/RL/RoArm. Next local-only design should isolate why "
            "near-face command/applied target still leaves actual TCP precontact: target-base accumulation, "
            "precontact reset offset, or actuator/drive follow telemetry."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = [
        "line1 artifact=cube10cm_tap_rl_nearface_target_path_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 launch_hygiene "
            f"first_launch_blocker={first_launch_blocker} "
            f"first_launch_cuda_device_error_seen={str(first_launch_cuda_error).upper()} "
            "rerun1_unsandboxed_with_PYTHONPATH=YES "
            f"runtime_status={near_sanity['status']} steps_executed={near_sanity['steps_executed']} "
            f"truncated_count={near_sanity['truncated_count']} terminated_count={near_sanity['terminated_count']} "
            f"reach_trace_rows={near_sanity['reach_trace_row_count']}"
        ),
        (
            "line3 target_path_effect "
            f"legacy_command_final_face_gap_m={legacy_stats['command_final_face_gap_m']:.9f} "
            f"nearface_command_final_face_gap_m={near_stats['command_final_face_gap_m']:.9f} "
            f"command_final_gap_delta_m={result['comparison']['command_final_gap_delta_m']:.9f} "
            f"nearface_command_inside_steps={near_stats['command_inside_first_step']}..{near_stats['command_inside_last_step']} "
            f"nearface_command_inside_rows={near_stats['command_inside_rows']}"
        ),
        (
            "line4 gate_result "
            f"rl_contact_gated_positive_control={near_sanity['rl_contact_gated_positive_control']} "
            f"professor_physical_reaction_evidence={near_sanity['professor_physical_reaction_evidence']} "
            f"contact_seen={near_sanity['last_log']['cube_tap_contact_seen_rate']} "
            f"tap_success={near_sanity['last_log']['cube_tap_success_rate']} "
            f"professor_seen={near_sanity['last_log']['cube_tap_professor_physical_reaction_seen_rate']} "
            f"overshoot_seen={near_sanity['last_log']['cube_tap_overshoot_seen_rate']} "
            f"max_disp_along_m={near_sanity['last_log']['cube_tap_max_disp_along_m']:.9f} "
            f"max_speed_mps={near_sanity['last_log']['cube_tap_max_speed_mps']:.9f}"
        ),
        (
            "line5 reach_result "
            f"nearface_applied_inside_rows={near_stats['applied_inside_rows']} "
            f"nearface_actual_inside_rows={near_stats['actual_inside_rows']} "
            f"nearface_applied_best_face_gap_m={near_stats['applied_best_face_gap_m']:.9f} "
            f"nearface_actual_best_face_gap_m={near_stats['actual_best_face_gap_m']:.9f} "
            f"nearface_actual_best_shortfall_m={near_stats['actual_best_shortfall_m']:.9f} "
            f"nearface_actual_final_face_gap_m={near_stats['actual_final_face_gap_m']:.9f}"
        ),
        (
            "line6 comparison_to_legacy "
            f"applied_fk_err_final_mm legacy={legacy_stats['applied_fk_err_final_mm']:.9f} "
            f"nearface={near_stats['applied_fk_err_final_mm']:.9f} "
            f"delta={result['comparison']['applied_fk_err_final_delta_mm']:.9f} "
            f"target_fk_err_final_delta_mm={result['comparison']['target_fk_err_final_delta_mm']:.9f} "
            f"actual_best_shortfall_delta_m={result['comparison']['actual_best_shortfall_delta_m']:.9f}"
        ),
        (
            "line7 verdict NEAR_FACE_TARGET_PATH_APPLIED_BUT_STRICT_CONTACT_STILL_FAILS_ACTUAL_TCP_PRECONTACT "
            "contact_gate_relaxation_unblock=NO dataset_rl_roarm=BLOCKED "
            "next=TARGET_BASE_ACCUMULATION_PRECONTACT_RESET_OR_ACTUATOR_FOLLOW_TELEMETRY_DESIGN"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
