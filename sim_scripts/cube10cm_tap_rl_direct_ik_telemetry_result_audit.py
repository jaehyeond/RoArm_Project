"""Posthoc audit for the direct-IK telemetry repeat.

Reads the completed telemetry runtime JSON only. It does not launch IsaacLab/GPU
runtime, generate datasets, train, control RoArm, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TELEMETRY_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
)
DEFAULT_TELEMETRY_SUMMARY = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity_summary.out"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_result_audit_summary.out"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry_json", type=Path, default=DEFAULT_TELEMETRY_JSON)
    parser.add_argument("--telemetry_summary", type=Path, default=DEFAULT_TELEMETRY_SUMMARY)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    data = _load_json(args.telemetry_json)
    controller = data.get("controller_metrics", {})
    last_log = data.get("last_log", {})

    env_step_s = 0.01
    direct_apply_active = bool(data.get("direct_ik_joint_target_apply", False))
    action_abs_max = _trace(data, "log_trace_stats", "cube_push_action_abs_max", "max")
    cap_rate = _trace(data, "log_trace_stats", "cube_push_joint_delta_cap_rate", "max")
    lead_limit_rate = _trace(data, "log_trace_stats", "cube_push_target_lead_limit_rate", "max")

    target_inside_max = _trace(
        data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
    )
    target_face_gap_min = _trace(
        data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "min"
    )
    target_face_gap_final = _trace(
        data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
    )
    target_fk_err_max_mm = _trace(
        data, "controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "max"
    )
    fk_frame_err_max_mm = _trace(
        data, "controller_trace_stats", "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean", "max"
    )
    follow_abs_max_final = _trace(
        data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"
    )
    follow_abs_max_max = _trace(
        data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"
    )
    follow_abs_mean_final = _trace(
        data, "controller_trace_stats", "direct_joint_follow_abs_mean_rad", "final"
    )
    actual_step_abs_max_final = _trace(
        data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"
    )
    actual_step_abs_mean_final = _trace(
        data, "controller_trace_stats", "direct_actual_joint_step_abs_mean_rad", "final"
    )
    target_delta_abs_max_final = _float(
        controller, "closed_loop_target_delta_from_actual_abs_max_rad_max"
    )
    target_delta_abs_mean_final = _float(
        controller, "closed_loop_target_delta_from_actual_abs_max_rad_mean"
    )
    actual_velocity_est_max_final = actual_step_abs_max_final / env_step_s
    actual_velocity_est_mean_final = actual_step_abs_mean_final / env_step_s
    velocity_limit_rad_s = 3.14

    observed_face_gap_max = _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "max")
    observed_shortfall_min = _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min")
    contact_seen = _float(last_log, "cube_tap_contact_seen_rate")
    tap_success = _float(last_log, "cube_tap_success_rate")
    professor_seen = _float(last_log, "cube_tap_professor_physical_reaction_seen_rate")

    target_path_ok = target_inside_max > 0.0 and target_face_gap_final > 0.010
    fk_frame_ok = fk_frame_err_max_mm <= 0.010
    ik_ok = _float(controller, "closed_loop_ik_ok_rate") == 1.0 and target_fk_err_max_mm <= 1.5
    wrapper_path_bypassed = direct_apply_active and action_abs_max == 0.0 and cap_rate == 0.0 and lead_limit_rate == 0.0
    follow_lag_large = follow_abs_max_final > 0.10 and follow_abs_mean_final > 0.05
    final_step_near_velocity_limit = actual_velocity_est_max_final >= 0.90 * velocity_limit_rad_s

    if target_path_ok and fk_frame_ok and ik_ok and wrapper_path_bypassed and follow_lag_large:
        primary_blocker = "DIRECT_JOINT_FOLLOW_ACTUATOR_TRACKING_LAG"
    elif not target_path_ok:
        primary_blocker = "TARGET_PATH_GEOMETRY"
    elif not fk_frame_ok:
        primary_blocker = "FK_MODEL_VS_ISAAC_TCP_FRAME_MISMATCH"
    else:
        primary_blocker = "UNRESOLVED_TARGET_APPLICATION_TIMING"

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_direct_ik_telemetry_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "telemetry_runtime": {
            "json": str(args.telemetry_json),
            "summary": str(args.telemetry_summary),
            "status": data.get("status"),
            "positive_control": data.get("positive_control"),
            "blocker": data.get("blocker"),
            "gpu_runtime": data.get("gpu_runtime"),
        },
        "target_and_frame_split": {
            "target_path_ok": target_path_ok,
            "target_inside_contact_band_rate_max": target_inside_max,
            "target_face_gap_min_m": target_face_gap_min,
            "target_face_gap_final_m": target_face_gap_final,
            "closed_loop_ik_ok_rate": _float(controller, "closed_loop_ik_ok_rate"),
            "target_fk_err_max_mm": target_fk_err_max_mm,
            "fk_frame_ok": fk_frame_ok,
            "actual_fk_vs_sim_tcp_err_max_mm": fk_frame_err_max_mm,
            "observed_actual_face_gap_max_m": observed_face_gap_max,
            "observed_actual_shortfall_min_m": observed_shortfall_min,
        },
        "action_and_follow_split": {
            "wrapper_path_bypassed": wrapper_path_bypassed,
            "direct_apply_active": direct_apply_active,
            "action_abs_max_trace": action_abs_max,
            "joint_delta_cap_rate_trace": cap_rate,
            "target_lead_limit_rate_trace": lead_limit_rate,
            "target_delta_abs_max_final_rad": target_delta_abs_max_final,
            "target_delta_abs_mean_final_rad": target_delta_abs_mean_final,
            "direct_joint_follow_abs_max_final_rad": follow_abs_max_final,
            "direct_joint_follow_abs_max_max_rad": follow_abs_max_max,
            "direct_joint_follow_abs_mean_final_rad": follow_abs_mean_final,
            "direct_actual_joint_step_abs_max_final_rad": actual_step_abs_max_final,
            "direct_actual_joint_step_abs_mean_final_rad": actual_step_abs_mean_final,
            "actual_velocity_est_max_final_rad_s": actual_velocity_est_max_final,
            "actual_velocity_est_mean_final_rad_s": actual_velocity_est_mean_final,
            "velocity_limit_rad_s": velocity_limit_rad_s,
            "final_step_near_velocity_limit": final_step_near_velocity_limit,
        },
        "outcome": {
            "primary_blocker": primary_blocker,
            "contact_seen": contact_seen,
            "tap_success": tap_success,
            "professor_physical_reaction_seen": professor_seen,
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next_step": {
            "local_design_next": "actuator_follow_time_scale_candidate_design",
            "selected_first_hypothesis": (
                "slow the direct target progression before changing geometry/action scale/cap; "
                "telemetry points to follow lag under existing actuator dynamics"
            ),
            "not_allowed": "dataset/RL/RoArm or action-teacher dataset before contact-gated positive-control passes",
        },
        "outputs": {
            "json": str(args.out_json),
            "summary": str(args.out_summary),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_direct_ik_telemetry_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_outcome "
            f"status={data.get('status')} positive_control={data.get('positive_control')} "
            f"professor_physical_reaction_seen={professor_seen:.1f} contact_seen={contact_seen:.1f} "
            f"tap_success={tap_success:.1f}"
        ),
        (
            "line3 target_frame "
            f"target_path_ok={target_path_ok} target_inside_max={target_inside_max:.1f} "
            f"target_face_gap_final_m={target_face_gap_final:.9f} "
            f"target_fk_err_max_mm={target_fk_err_max_mm:.9f} "
            f"actual_fk_vs_sim_tcp_err_max_mm={fk_frame_err_max_mm:.9f} fk_frame_ok={fk_frame_ok}"
        ),
        (
            "line4 follow_lag "
            f"target_delta_abs_max_final_rad={target_delta_abs_max_final:.9f} "
            f"direct_joint_follow_abs_max_final_rad={follow_abs_max_final:.9f} "
            f"direct_joint_follow_abs_mean_final_rad={follow_abs_mean_final:.9f} "
            f"direct_actual_joint_step_abs_max_final_rad={actual_step_abs_max_final:.9f} "
            f"actual_velocity_est_max_final_rad_s={actual_velocity_est_max_final:.9f} "
            f"near_velocity_limit={final_step_near_velocity_limit}"
        ),
        (
            "line5 exclusions "
            f"wrapper_path_bypassed={wrapper_path_bypassed} action_abs_max_trace={action_abs_max:.1f} "
            f"cap_rate={cap_rate:.1f} lead_limit_rate={lead_limit_rate:.1f} "
            f"observed_actual_face_gap_max_m={observed_face_gap_max:.9f} "
            f"observed_actual_shortfall_min_m={observed_shortfall_min:.9f}"
        ),
        (
            "line6 verdict "
            f"primary_blocker={primary_blocker} diffik_action_dataset=BLOCKED "
            "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line7 next "
            "local_design_next=actuator_follow_time_scale_candidate_design "
            "not_allowed=dataset_rl_roarm_or_action_teacher_before_contact_gated_positive_control"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
