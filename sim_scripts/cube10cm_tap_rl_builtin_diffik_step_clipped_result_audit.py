#!/usr/bin/env python3
"""Posthoc audit for the built-in DiffIK step-clipped 10cm tap runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_STEP_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_sanity.json"
DEFAULT_DIRECT_BUILTIN_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_direct_apply_sanity.json"
)
DEFAULT_BASELINE_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
)
DEFAULT_SLOW240_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(data: dict[str, Any], section: str, key: str, field: str, default: float = 0.0) -> float:
    item = data.get(section, {}).get(key, {})
    return _float(item, field, default) if isinstance(item, dict) else default


def _metrics(data: dict[str, Any]) -> dict[str, Any]:
    last_log = data.get("last_log", {})
    controller = data.get("controller_metrics", {})
    return {
        "status": data.get("status"),
        "controller_mode": data.get("controller_mode"),
        "positive_control": data.get("positive_control"),
        "professor_physical_reaction_evidence": data.get("professor_physical_reaction_evidence"),
        "rl_contact_gated_positive_control": data.get("rl_contact_gated_positive_control"),
        "blocker": data.get("blocker"),
        "steps_executed": int(data.get("steps_executed", 0)),
        "max_steps": int(data.get("max_steps", 0)),
        "closed_loop_push_steps": int(data.get("closed_loop_push_steps", 0)),
        "controller_goal_ok_rate": _float(data, "controller_goal_ok_rate"),
        "direct_ik_joint_target_apply": bool(data.get("direct_ik_joint_target_apply", False)),
        "isaac_builtin_diffik_controller_apply": bool(data.get("isaac_builtin_diffik_controller_apply", False)),
        "builtin_diffik_step_clipped_target_apply": bool(
            data.get("builtin_diffik_step_clipped_target_apply", False)
        ),
        "builtin_diffik_step_clip_rad": _float(data, "builtin_diffik_step_clip_rad"),
        "closed_loop_ik_ok_rate": _float(controller, "closed_loop_ik_ok_rate"),
        "builtin_diffik_step_clip_rate": _float(controller, "builtin_diffik_step_clip_rate"),
        "builtin_diffik_raw_delta_abs_max_rad": _float(controller, "builtin_diffik_raw_delta_abs_max_rad"),
        "builtin_diffik_clipped_delta_abs_max_rad": _float(
            controller, "builtin_diffik_clipped_delta_abs_max_rad"
        ),
        "contact_seen": _float(last_log, "cube_tap_contact_seen_rate"),
        "reaction_signal_now": _float(last_log, "cube_tap_reaction_signal_now_rate"),
        "reaction_contact_context": _float(last_log, "cube_tap_reaction_contact_context_rate"),
        "reaction_seen": _float(last_log, "cube_tap_reaction_seen_rate"),
        "professor_seen": _float(last_log, "cube_tap_professor_physical_reaction_seen_rate"),
        "overshoot_seen": _float(last_log, "cube_tap_overshoot_seen_rate"),
        "tap_success": _float(last_log, "cube_tap_success_rate"),
        "max_disp_along_m": _float(last_log, "cube_tap_max_disp_along_m"),
        "max_speed_mps": _float(last_log, "cube_tap_max_speed_mps"),
        "terminated_count": int(data.get("terminated_count", 0)),
        "truncated_count": int(data.get("truncated_count", 0)),
        "face_gap_max_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "max"),
        "face_gap_final_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "final"),
        "shortfall_min_m": _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min"),
        "shortfall_final_m": _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "final"),
        "target_inside_max": _trace(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
        ),
        "target_face_gap_final_m": _trace(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
        ),
        "target_delta_final_rad": _trace(
            data,
            "controller_trace_stats",
            "closed_loop_target_delta_from_actual_abs_max_rad_max",
            "final",
        ),
        "follow_final_rad": _trace(data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"),
        "follow_max_rad": _trace(data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"),
        "actual_step_final_rad": _trace(
            data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"
        ),
        "actual_step_max_rad": _trace(
            data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "max"
        ),
        "action_abs_max_trace": _trace(data, "log_trace_stats", "cube_push_action_abs_max", "max"),
        "joint_delta_cap_rate_trace": _trace(data, "log_trace_stats", "cube_push_joint_delta_cap_rate", "max"),
        "target_lead_limit_rate_trace": _trace(
            data, "log_trace_stats", "cube_push_target_lead_limit_rate", "max"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_json", type=Path, default=DEFAULT_STEP_JSON)
    parser.add_argument("--direct_builtin_json", type=Path, default=DEFAULT_DIRECT_BUILTIN_JSON)
    parser.add_argument("--baseline_json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--slow240_json", type=Path, default=DEFAULT_SLOW240_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    step = _metrics(_load(args.step_json))
    direct_builtin = _metrics(_load(args.direct_builtin_json))
    baseline = _metrics(_load(args.baseline_json))
    slow240 = _metrics(_load(args.slow240_json))

    target_path_ok = (
        step["controller_goal_ok_rate"] == 1.0
        and step["closed_loop_ik_ok_rate"] == 1.0
        and step["target_inside_max"] > 0.0
        and step["target_face_gap_final_m"] > 0.010
    )
    follow_lag_solved_vs_full_builtin = step["follow_final_rad"] < 0.05 and (
        step["follow_final_rad"] < direct_builtin["follow_final_rad"] * 0.10
    )
    strict_gate_failed = step["contact_seen"] == 0.0 and step["tap_success"] == 0.0
    horizon_limited = step["truncated_count"] > 0 and step["terminated_count"] == 0
    wrapper_bypassed = (
        step["direct_ik_joint_target_apply"]
        and step["action_abs_max_trace"] == 0.0
        and step["joint_delta_cap_rate_trace"] == 0.0
        and step["target_lead_limit_rate_trace"] == 0.0
    )

    if target_path_ok and follow_lag_solved_vs_full_builtin and strict_gate_failed and horizon_limited:
        primary_blocker = "STEP_CLIPPED_DIFFIK_TARGET_APPLICATION_HORIZON_OR_PROGRESS_TOO_SHORT"
    elif strict_gate_failed:
        primary_blocker = "STEP_CLIPPED_DIFFIK_STILL_CONTACT_GAP_FAIL"
    else:
        primary_blocker = "RECHECK_STEP_CLIPPED_DIFFIK_RESULT"

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit_v1",
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
            "step_json": str(args.step_json),
            "direct_builtin_json": str(args.direct_builtin_json),
            "baseline_json": str(args.baseline_json),
            "slow240_json": str(args.slow240_json),
        },
        "runtime": step,
        "comparison": {
            "direct_builtin_full_target": direct_builtin,
            "external_direct_apply_baseline": baseline,
            "external_direct_apply_slow240": slow240,
            "follow_final_ratio_vs_full_builtin": step["follow_final_rad"]
            / max(direct_builtin["follow_final_rad"], 1.0e-9),
            "follow_final_ratio_vs_slow240": step["follow_final_rad"] / max(slow240["follow_final_rad"], 1.0e-9),
            "shortfall_min_improvement_vs_baseline_m": baseline["shortfall_min_m"] - step["shortfall_min_m"],
            "shortfall_min_improvement_vs_slow240_m": slow240["shortfall_min_m"] - step["shortfall_min_m"],
        },
        "interpretation": {
            "target_path_ok": target_path_ok,
            "wrapper_bypassed": wrapper_bypassed,
            "follow_lag_solved_vs_full_builtin": follow_lag_solved_vs_full_builtin,
            "strict_gate_failed": strict_gate_failed,
            "horizon_limited": horizon_limited,
            "professor_evidence_preserved": step["professor_physical_reaction_evidence"] == "PASS",
            "do_not_relax_contact_gate_yet": True,
            "why": (
                "Step clipping makes the joint target followable, but the actual face gap never enters the strict "
                "contact band within 120 steps. This points to horizon/progress under step-clipped DiffIK rather "
                "than a clean contact-gate exception."
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
            "local_unblock": "design_step_clipped_diffik_horizon_progress_candidate",
            "not_dataset_or_rl": True,
            "not_contact_gate_relaxation": True,
        },
        "outputs": {"json": str(args.out_json), "summary": str(args.out_summary)},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 runtime_outcome "
        f"status={step['status']} controller_mode={step['controller_mode']} "
        f"step_clip_rad={step['builtin_diffik_step_clip_rad']:.3f} "
        f"contact_seen={step['contact_seen']:.1f} tap_success={step['tap_success']:.1f} "
        f"professor_evidence={step['professor_physical_reaction_evidence']} "
        f"terminated_count={step['terminated_count']} truncated_count={step['truncated_count']}",
        "line3 target_and_clip "
        f"target_path_ok={target_path_ok} closed_loop_ik_ok_rate={step['closed_loop_ik_ok_rate']:.1f} "
        f"target_inside_max={step['target_inside_max']:.1f} "
        f"target_face_gap_final_m={step['target_face_gap_final_m']:.9f} "
        f"raw_delta_final_rad={step['builtin_diffik_raw_delta_abs_max_rad']:.9f} "
        f"clipped_delta_final_rad={step['builtin_diffik_clipped_delta_abs_max_rad']:.9f} "
        f"clip_rate_final={step['builtin_diffik_step_clip_rate']:.9f}",
        "line4 actual_follow "
        f"face_gap_max_m={step['face_gap_max_m']:.9f} shortfall_min_m={step['shortfall_min_m']:.9f} "
        f"shortfall_final_m={step['shortfall_final_m']:.9f} "
        f"target_delta_final_rad={step['target_delta_final_rad']:.9f} "
        f"follow_final_rad={step['follow_final_rad']:.9f} follow_max_rad={step['follow_max_rad']:.9f} "
        f"actual_step_final_rad={step['actual_step_final_rad']:.9f}",
        "line5 comparison "
        f"full_builtin_follow_final_rad={direct_builtin['follow_final_rad']:.9f} "
        f"slow240_follow_final_rad={slow240['follow_final_rad']:.9f} "
        f"follow_lag_solved_vs_full_builtin={follow_lag_solved_vs_full_builtin} "
        f"baseline_shortfall_min_m={baseline['shortfall_min_m']:.9f} "
        f"slow240_shortfall_min_m={slow240['shortfall_min_m']:.9f}",
        "line6 verdict "
        f"primary_blocker={primary_blocker} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line7 next "
        "local_unblock=design_step_clipped_diffik_horizon_progress_candidate "
        "contact_gate_relaxation=NO dataset_rl_roarm=NO",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if primary_blocker.startswith("STEP_CLIPPED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
