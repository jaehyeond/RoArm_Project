"""Static local audit for the direct-IK positive-control blocker.

This audit reads existing logs and source code only. It does not launch
IsaacLab/GPU runtime, generate datasets, train, control RoArm, SSH, or touch
B200. The goal is to narrow the RL/learning blocker before any new runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_DIRECT_RUNTIME_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_sanity.json"
)
DEFAULT_DIRECT_AUDIT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit.json"
DEFAULT_HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
DEFAULT_ENV = REPO / "roarm_rl/roarm_cube_push_env.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_static_target_geometry_blocker_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_static_target_geometry_blocker_audit_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, needle: str) -> int:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return idx
    return -1


def _float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(data: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    trace = data.get("log_trace_stats", {}).get(key, {})
    if isinstance(trace, dict):
        return _float(trace, field, default)
    return default


def _controller_trace(data: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    trace = data.get("controller_trace_stats", {}).get(key, {})
    if isinstance(trace, dict):
        return _float(trace, field, default)
    return default


def _target_face_gap_plan(
    *,
    cube_size_m: float,
    precontact_clearance_m: float,
    goal_push_m: float,
    closed_loop_push_steps: int,
    total_steps: int,
    push_dir_x: float,
    push_dir_y: float,
) -> dict[str, Any]:
    norm = math.hypot(push_dir_x, push_dir_y)
    if norm <= 1.0e-9:
        raise ValueError("push direction must be nonzero")
    dx = push_dir_x / norm
    dy = push_dir_y / norm
    half_xy = (0.5 * cube_size_m, 0.5 * cube_size_m)
    half_along = abs(dx) * half_xy[0] + abs(dy) * half_xy[1]
    pre_face_gap = -float(precontact_clearance_m)
    through_face_gap = 2.0 * half_along + float(goal_push_m)
    span = through_face_gap - pre_face_gap
    target_face_gaps = []
    inside_steps_1based = []
    band = 0.010
    for step in range(int(total_steps)):
        alpha = min(1.0, max(0.0, float(step + 1) / max(float(closed_loop_push_steps), 1.0)))
        face_gap = pre_face_gap + alpha * span
        target_face_gaps.append(face_gap)
        if -band <= face_gap <= band:
            inside_steps_1based.append(step + 1)
    return {
        "half_along_m": half_along,
        "pre_face_gap_m": pre_face_gap,
        "through_face_gap_m": through_face_gap,
        "target_face_gap_span_m": span,
        "target_crosses_contact_band": any(-band <= value <= band for value in target_face_gaps),
        "target_inside_contact_band_step_count": len(inside_steps_1based),
        "target_inside_contact_band_first_step_1based": inside_steps_1based[0] if inside_steps_1based else None,
        "target_inside_contact_band_last_step_1based": inside_steps_1based[-1] if inside_steps_1based else None,
        "target_final_face_gap_m": target_face_gaps[-1] if target_face_gaps else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_runtime_json", type=Path, default=DEFAULT_DIRECT_RUNTIME_JSON)
    parser.add_argument("--direct_audit_json", type=Path, default=DEFAULT_DIRECT_AUDIT_JSON)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    direct = _load_json(args.direct_runtime_json)
    audit = _load_json(args.direct_audit_json)
    plan = _target_face_gap_plan(
        cube_size_m=float(direct.get("cube_size_m", 0.100)),
        precontact_clearance_m=float(direct.get("precontact_clearance_m", 0.020)),
        goal_push_m=float(direct.get("goal_push_m", 0.006)),
        closed_loop_push_steps=int(direct.get("closed_loop_push_steps", 72)),
        total_steps=int(direct.get("steps_executed", direct.get("max_steps", 120))),
        push_dir_x=float(direct.get("fixed_push_dir_x", 1.0)),
        push_dir_y=float(direct.get("fixed_push_dir_y", 0.0)),
    )

    observed = {
        "direct_apply_active": bool(direct.get("direct_ik_joint_target_apply", False)),
        "rl_action_path_bypassed": bool(audit.get("rl_action_path_bypassed", False)),
        "closed_loop_ik_ok_rate": _float(direct.get("controller_metrics", {}), "closed_loop_ik_ok_rate"),
        "closed_loop_ik_err_mm_mean": _float(direct.get("controller_metrics", {}), "closed_loop_ik_err_mm_mean"),
        "action_abs_max_trace": _trace(direct, "cube_push_action_abs_max", "max"),
        "joint_delta_cap_rate_trace": _trace(direct, "cube_push_joint_delta_cap_rate", "max"),
        "target_lead_limit_rate_trace": _trace(direct, "cube_push_target_lead_limit_rate", "max"),
        "initial_face_gap_m": _float(direct.get("reset_metrics", {}), "initial_face_gap_m"),
        "best_face_gap_m": float(audit.get("best_face_gap_m", _trace(direct, "cube_tap_contact_face_gap_m", "max"))),
        "final_face_gap_m": float(audit.get("final_face_gap_m", _trace(direct, "cube_tap_contact_face_gap_m", "final"))),
        "best_shortfall_m": float(
            audit.get("best_shortfall_to_contact_band_m", _trace(direct, "cube_tap_contact_band_shortfall_m", "min"))
        ),
        "final_shortfall_m": float(
            audit.get("final_shortfall_to_contact_band_m", _trace(direct, "cube_tap_contact_band_shortfall_m", "final"))
        ),
        "best_improvement_m": float(audit.get("best_improvement_from_initial_m", 0.0)),
        "contact_seen": _float(direct.get("last_log", {}), "cube_tap_contact_seen_rate"),
        "tap_success": _float(direct.get("last_log", {}), "cube_tap_success_rate"),
        "max_disp_along_m": _float(direct.get("last_log", {}), "cube_tap_max_disp_along_m"),
        "max_speed_mps": _float(direct.get("last_log", {}), "cube_tap_max_speed_mps"),
        "final_lateral_m": _float(direct.get("last_log", {}), "cube_tap_contact_lateral_m"),
        "final_vertical_offset_m": _float(direct.get("last_log", {}), "cube_tap_contact_vertical_offset_m"),
        "target_lead_abs_max_trace": _trace(direct, "cube_push_target_lead_abs_max", "max"),
    }

    has_runtime_telemetry = all(
        key in direct.get("controller_metrics", {})
        for key in (
            "closed_loop_target_face_gap_m_mean",
            "closed_loop_target_inside_contact_band_rate",
            "closed_loop_target_fk_err_mm_mean",
            "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean",
            "direct_joint_follow_abs_max_rad",
        )
    )
    instrumentation_lines = {
        "target_path_pre_xy": _line_of(args.harness, "pre[:2] -= push_dir[env_id]"),
        "target_path_through_xy": _line_of(args.harness, "through[:2] += push_dir[env_id]"),
        "target_face_gap_metric": _line_of(args.harness, "closed_loop_target_face_gap_m_mean"),
        "target_inside_metric": _line_of(args.harness, "closed_loop_target_inside_contact_band_rate"),
        "target_fk_metric": _line_of(args.harness, "closed_loop_target_fk_err_mm_mean"),
        "fk_frame_metric": _line_of(args.harness, "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean"),
        "direct_follow_metric": _line_of(args.harness, "direct_joint_follow_abs_max_rad"),
        "direct_apply_override": _line_of(args.env, "_external_joint_targets_override"),
        "direct_apply_set_target": _line_of(args.env, "self.robot_dof_targets[:] = targets"),
        "contact_face_gap_contract": _line_of(args.env, "face_gap = along + half_along"),
    }
    instrumentation_ready = all(value > 0 for value in instrumentation_lines.values())

    target_geometry_undercommand_falsified = (
        bool(plan["target_crosses_contact_band"])
        and int(plan["target_inside_contact_band_step_count"]) > 0
        and float(plan["through_face_gap_m"]) > 0.010
    )
    live_cube_motion_explains_failure = observed["max_disp_along_m"] >= observed["best_shortfall_m"]
    lateral_vertical_not_blocker = bool(audit.get("lateral_ok", False)) and bool(audit.get("vertical_ok", False))
    wrapper_cap_lead_not_primary = (
        observed["direct_apply_active"]
        and observed["rl_action_path_bypassed"]
        and observed["action_abs_max_trace"] == 0.0
        and observed["joint_delta_cap_rate_trace"] == 0.0
        and observed["target_lead_limit_rate_trace"] == 0.0
    )
    target_vs_observed_gap_delta_m = float(plan["target_final_face_gap_m"]) - observed["final_face_gap_m"]

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_static_target_geometry_blocker_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_static_existing_logs_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "target_plan": plan,
        "observed_direct_runtime": observed,
        "static_exclusions": {
            "gross_target_geometry_undercommand_falsified": target_geometry_undercommand_falsified,
            "lateral_vertical_not_blocker": lateral_vertical_not_blocker,
            "wrapper_cap_lead_not_primary": wrapper_cap_lead_not_primary,
            "live_cube_motion_explains_failure": live_cube_motion_explains_failure,
            "target_final_vs_observed_final_face_gap_delta_m": target_vs_observed_gap_delta_m,
        },
        "telemetry_state": {
            "old_runtime_has_new_telemetry": has_runtime_telemetry,
            "instrumentation_ready_in_harness": instrumentation_ready,
            "instrumentation_lines": instrumentation_lines,
        },
        "remaining_primary_split": [
            "direct_joint_follow_or_actuator_tracking_lag",
            "fk_model_vs_isaac_tcp_frame_mismatch",
            "target_application_timing_or_reset_state_mismatch",
        ],
        "still_blocked": {
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next_step": {
            "local_result": (
                "gross target path under-command is falsified; the planned direct-IK target crosses the contact band, "
                "but observed TCP face gap stays outside"
            ),
            "next_runtime_if_explicitly_allowed": "zero-control-knob direct-IK telemetry repeat",
            "not_allowed": "dataset/RL/RoArm or lead/cap/action-scale sweep before telemetry split",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_static_target_geometry_blocker_audit_v1 "
        "local_static_existing_logs_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 target_plan "
            f"pre_face_gap_m={plan['pre_face_gap_m']:.9f} "
            f"through_face_gap_m={plan['through_face_gap_m']:.9f} "
            f"target_crosses_contact_band={plan['target_crosses_contact_band']} "
            f"inside_steps={plan['target_inside_contact_band_step_count']} "
            f"first_step_1based={plan['target_inside_contact_band_first_step_1based']} "
            f"last_step_1based={plan['target_inside_contact_band_last_step_1based']}"
        ),
        (
            "line3 observed_actual "
            f"best_face_gap_m={observed['best_face_gap_m']:.9f} "
            f"final_face_gap_m={observed['final_face_gap_m']:.9f} "
            f"best_shortfall_m={observed['best_shortfall_m']:.9f} "
            f"best_improvement_m={observed['best_improvement_m']:.9f} "
            f"contact_seen={observed['contact_seen']:.1f} tap_success={observed['tap_success']:.1f}"
        ),
        (
            "line4 exclusions "
            f"gross_target_geometry_undercommand_falsified={target_geometry_undercommand_falsified} "
            f"lateral_vertical_not_blocker={lateral_vertical_not_blocker} "
            f"wrapper_cap_lead_not_primary={wrapper_cap_lead_not_primary} "
            f"live_cube_motion_explains_failure={live_cube_motion_explains_failure}"
        ),
        (
            "line5 telemetry_gap "
            f"old_runtime_has_new_telemetry={has_runtime_telemetry} "
            f"instrumentation_ready_in_harness={instrumentation_ready} "
            "remaining_split=direct_joint_follow_or_actuator_tracking_lag,"
            "fk_model_vs_isaac_tcp_frame_mismatch,target_application_timing_or_reset_state_mismatch"
        ),
        (
            "line6 verdict "
            "STATIC_TARGET_GEOMETRY_UNDERCOMMAND_FALSIFIED_NEED_ZERO_KNOB_TELEMETRY_SPLIT "
            "contact_gated_positive_control=RUN_FAILED diffik_action_dataset=BLOCKED "
            "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line7 next "
            "allowed_if_explicitly_approved=zero_control_knob_direct_ik_telemetry_repeat "
            "not_allowed=dataset_rl_roarm_or_lead_cap_action_sweep_before_telemetry"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
