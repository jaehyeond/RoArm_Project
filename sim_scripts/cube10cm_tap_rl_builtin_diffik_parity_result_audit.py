#!/usr/bin/env python3
"""Posthoc audit for the 10cm tap built-in DifferentialIKController parity runtime.

Reads existing local runtime JSON files only. It does not launch IsaacLab/GPU
runtime, generate datasets, train, control RoArm, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_BUILTIN_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_direct_apply_sanity.json"
)
DEFAULT_BUILTIN_SUMMARY = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_direct_apply_sanity_summary.out"
)
DEFAULT_BASELINE_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
)
DEFAULT_SLOW240_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json"
)
DEFAULT_CANDIDATE_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_candidate_design.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_result_audit_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(data: dict[str, Any], section: str, key: str, field: str, default: float = 0.0) -> float:
    item = data.get(section, {}).get(key, {})
    if isinstance(item, dict):
        return _float(item, field, default)
    return default


def _runtime_metrics(data: dict[str, Any]) -> dict[str, Any]:
    last_log = data.get("last_log", {})
    controller = data.get("controller_metrics", {})
    return {
        "status": data.get("status"),
        "positive_control": data.get("positive_control"),
        "controller_mode": data.get("controller_mode"),
        "direct_ik_joint_target_apply": bool(data.get("direct_ik_joint_target_apply", False)),
        "isaac_builtin_diffik_controller_apply": bool(data.get("isaac_builtin_diffik_controller_apply", False)),
        "builtin_diffik_lambda": _float(data, "builtin_diffik_lambda"),
        "closed_loop_push_steps": int(data.get("closed_loop_push_steps", 0)),
        "steps_executed": int(data.get("steps_executed", 0)),
        "controller_goal_ok_rate": _float(data, "controller_goal_ok_rate"),
        "closed_loop_ik_ok_rate": _float(controller, "closed_loop_ik_ok_rate"),
        "builtin_diffik_numeric_ok_rate": _float(controller, "builtin_diffik_numeric_ok_rate"),
        "builtin_diffik_live_jacobian": _float(controller, "builtin_diffik_live_jacobian"),
        "builtin_diffik_tool_proxy_offset": _float(controller, "builtin_diffik_tool_proxy_offset"),
        "contact_seen": _float(last_log, "cube_tap_contact_seen_rate"),
        "reaction_signal_now": _float(last_log, "cube_tap_reaction_signal_now_rate"),
        "reaction_contact_context": _float(last_log, "cube_tap_reaction_contact_context_rate"),
        "reaction_seen": _float(last_log, "cube_tap_reaction_seen_rate"),
        "professor_physical_reaction_seen": _float(last_log, "cube_tap_professor_physical_reaction_seen_rate"),
        "overshoot_seen": _float(last_log, "cube_tap_overshoot_seen_rate"),
        "tap_success": _float(last_log, "cube_tap_success_rate"),
        "max_disp_along_m": _float(last_log, "cube_tap_max_disp_along_m"),
        "max_speed_mps": _float(last_log, "cube_tap_max_speed_mps"),
        "terminated_count": int(data.get("terminated_count", 0)),
        "truncated_count": int(data.get("truncated_count", 0)),
        "professor_physical_reaction_evidence": data.get("professor_physical_reaction_evidence"),
        "rl_contact_gated_positive_control": data.get("rl_contact_gated_positive_control"),
        "blocker": data.get("blocker"),
        "face_gap_max_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "max"),
        "face_gap_final_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "final"),
        "shortfall_min_m": _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min"),
        "shortfall_final_m": _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "final"),
        "target_inside_contact_band_rate_max": _trace(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
        ),
        "target_face_gap_final_m": _trace(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
        ),
        "target_tcp_err_before_final_m": _trace(
            data, "controller_trace_stats", "builtin_diffik_target_tcp_err_before_m_mean", "final"
        ),
        "actual_fk_vs_sim_tcp_err_max_mm": _trace(
            data,
            "controller_trace_stats",
            "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean",
            "max",
        ),
        "target_delta_abs_max_final_rad": _trace(
            data,
            "controller_trace_stats",
            "closed_loop_target_delta_from_actual_abs_max_rad_max",
            "final",
        ),
        "target_delta_abs_max_max_rad": _trace(
            data,
            "controller_trace_stats",
            "closed_loop_target_delta_from_actual_abs_max_rad_max",
            "max",
        ),
        "direct_joint_follow_abs_max_final_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"
        ),
        "direct_joint_follow_abs_max_max_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"
        ),
        "direct_actual_joint_step_abs_max_final_rad": _trace(
            data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"
        ),
        "direct_actual_joint_step_abs_max_max_rad": _trace(
            data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "max"
        ),
        "action_abs_max_trace": _trace(data, "log_trace_stats", "cube_push_action_abs_max", "max"),
        "joint_delta_cap_rate_trace": _trace(
            data, "log_trace_stats", "cube_push_joint_delta_cap_rate", "max"
        ),
        "target_lead_limit_rate_trace": _trace(
            data, "log_trace_stats", "cube_push_target_lead_limit_rate", "max"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builtin_json", type=Path, default=DEFAULT_BUILTIN_JSON)
    parser.add_argument("--builtin_summary", type=Path, default=DEFAULT_BUILTIN_SUMMARY)
    parser.add_argument("--baseline_json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--slow240_json", type=Path, default=DEFAULT_SLOW240_JSON)
    parser.add_argument("--candidate_json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    builtin_raw = _load_json(args.builtin_json)
    baseline_raw = _load_json(args.baseline_json)
    slow240_raw = _load_json(args.slow240_json)
    candidate = _load_json(args.candidate_json) if args.candidate_json.exists() else {}

    builtin = _runtime_metrics(builtin_raw)
    baseline = _runtime_metrics(baseline_raw)
    slow240 = _runtime_metrics(slow240_raw)

    target_path_ok = (
        builtin["controller_goal_ok_rate"] == 1.0
        and builtin["closed_loop_ik_ok_rate"] == 1.0
        and builtin["target_inside_contact_band_rate_max"] > 0.0
        and builtin["target_face_gap_final_m"] > 0.010
    )
    wrapper_bypassed = (
        builtin["direct_ik_joint_target_apply"]
        and builtin["action_abs_max_trace"] == 0.0
        and builtin["joint_delta_cap_rate_trace"] == 0.0
        and builtin["target_lead_limit_rate_trace"] == 0.0
    )
    strict_gate_failed = builtin["contact_seen"] == 0.0 and builtin["tap_success"] == 0.0
    professor_gate_failed_by_termination = (
        builtin["professor_physical_reaction_seen"] > 0.0
        and builtin["overshoot_seen"] == 0.0
        and builtin["terminated_count"] > 0
        and builtin["professor_physical_reaction_evidence"] == "FAIL"
    )
    follow_worse_than_baseline = (
        builtin["direct_joint_follow_abs_max_final_rad"]
        > baseline["direct_joint_follow_abs_max_final_rad"]
    )
    follow_worse_than_slow240 = (
        builtin["direct_joint_follow_abs_max_final_rad"]
        > slow240["direct_joint_follow_abs_max_final_rad"]
    )
    three_cm_target_application_gap = (
        "inner.robot_dof_targets[:] = target_full"
        if not candidate.get("candidate", {}).get("unchanged", {}).get("action_wrapper_knobs") == "3cm_step_clipped"
        else "unknown"
    )

    if target_path_ok and wrapper_bypassed and strict_gate_failed and follow_worse_than_baseline:
        primary_blocker = "BUILTIN_DIFFIK_FULL_TARGET_APPLICATION_STILL_HAS_ACTUATOR_FOLLOW_LAG"
    elif strict_gate_failed:
        primary_blocker = "BUILTIN_DIFFIK_PARITY_STILL_CONTACT_GAP_FAIL"
    else:
        primary_blocker = "RECHECK_BUILTIN_DIFFIK_PARITY_RESULT"

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_builtin_diffik_parity_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "builtin_json": str(args.builtin_json),
            "builtin_summary": str(args.builtin_summary),
            "baseline_json": str(args.baseline_json),
            "slow240_json": str(args.slow240_json),
            "candidate_json": str(args.candidate_json),
        },
        "runtime": builtin,
        "comparisons": {
            "baseline_external_closed_loop_direct_apply": baseline,
            "slow240_external_closed_loop_direct_apply": slow240,
            "follow_final_vs_baseline_delta_rad": (
                builtin["direct_joint_follow_abs_max_final_rad"]
                - baseline["direct_joint_follow_abs_max_final_rad"]
            ),
            "follow_final_vs_slow240_delta_rad": (
                builtin["direct_joint_follow_abs_max_final_rad"]
                - slow240["direct_joint_follow_abs_max_final_rad"]
            ),
            "shortfall_min_vs_baseline_delta_m": builtin["shortfall_min_m"] - baseline["shortfall_min_m"],
            "shortfall_min_vs_slow240_delta_m": builtin["shortfall_min_m"] - slow240["shortfall_min_m"],
        },
        "interpretation": {
            "target_path_ok": target_path_ok,
            "wrapper_bypassed": wrapper_bypassed,
            "strict_gate_failed": strict_gate_failed,
            "professor_gate_failed_by_termination": professor_gate_failed_by_termination,
            "follow_worse_than_baseline": follow_worse_than_baseline,
            "follow_worse_than_slow240": follow_worse_than_slow240,
            "three_cm_target_application_gap": three_cm_target_application_gap,
            "do_not_relax_contact_gate_yet": True,
            "why": (
                "Built-in DiffIK compute/parity mode still misses the strict contact gate, and this direct full "
                "joint target application is not yet the 3cm probe's step-clipped target application cadence."
            ),
        },
        "outcome": {
            "primary_blocker": primary_blocker,
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next_step": {
            "local_unblock": "design_default_off_builtin_diffik_step_clipped_target_application_parity",
            "not_dataset_or_rl": True,
            "not_contact_gate_relaxation": True,
        },
        "outputs": {"json": str(args.out_json), "summary": str(args.out_summary)},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_builtin_diffik_parity_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 runtime_outcome "
        f"status={builtin['status']} controller_mode={builtin['controller_mode']} "
        f"builtin_apply={builtin['isaac_builtin_diffik_controller_apply']} "
        f"lambda={builtin['builtin_diffik_lambda']:.3f} contact_seen={builtin['contact_seen']:.1f} "
        f"tap_success={builtin['tap_success']:.1f} professor_seen={builtin['professor_physical_reaction_seen']:.1f} "
        f"professor_evidence={builtin['professor_physical_reaction_evidence']} "
        f"terminated_count={builtin['terminated_count']}",
        "line3 target_path "
        f"target_path_ok={target_path_ok} controller_goal_ok_rate={builtin['controller_goal_ok_rate']:.1f} "
        f"closed_loop_ik_ok_rate={builtin['closed_loop_ik_ok_rate']:.1f} "
        f"numeric_ok={builtin['builtin_diffik_numeric_ok_rate']:.1f} "
        f"live_jacobian={builtin['builtin_diffik_live_jacobian']:.1f} "
        f"target_inside_max={builtin['target_inside_contact_band_rate_max']:.1f} "
        f"target_face_gap_final_m={builtin['target_face_gap_final_m']:.9f} "
        f"actual_fk_vs_sim_tcp_err_max_mm={builtin['actual_fk_vs_sim_tcp_err_max_mm']:.9f}",
        "line4 actual_follow "
        f"face_gap_max_m={builtin['face_gap_max_m']:.9f} "
        f"shortfall_min_m={builtin['shortfall_min_m']:.9f} "
        f"shortfall_final_m={builtin['shortfall_final_m']:.9f} "
        f"target_delta_final_rad={builtin['target_delta_abs_max_final_rad']:.9f} "
        f"follow_final_rad={builtin['direct_joint_follow_abs_max_final_rad']:.9f} "
        f"follow_max_rad={builtin['direct_joint_follow_abs_max_max_rad']:.9f} "
        f"actual_step_final_rad={builtin['direct_actual_joint_step_abs_max_final_rad']:.9f}",
        "line5 comparison "
        f"baseline_follow_final_rad={baseline['direct_joint_follow_abs_max_final_rad']:.9f} "
        f"slow240_follow_final_rad={slow240['direct_joint_follow_abs_max_final_rad']:.9f} "
        f"follow_worse_than_baseline={follow_worse_than_baseline} "
        f"follow_worse_than_slow240={follow_worse_than_slow240} "
        f"baseline_shortfall_min_m={baseline['shortfall_min_m']:.9f} "
        f"slow240_shortfall_min_m={slow240['shortfall_min_m']:.9f}",
        "line6 parity_gap "
        "built_in_diffik_compute=YES full_joint_pos_des_direct_apply=YES "
        "three_cm_step_clipped_target_application=NOT_YET "
        "contact_gate_relaxation=BLOCKED reason=would_hide_target_application_mismatch",
        "line7 verdict "
        f"primary_blocker={primary_blocker} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line8 next "
        "local_unblock=design_default_off_builtin_diffik_step_clipped_target_application_parity "
        "dataset_rl_roarm=NO contact_gate_relaxation=NO",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if primary_blocker.startswith("BUILTIN_DIFFIK") else 2


if __name__ == "__main__":
    raise SystemExit(main())
