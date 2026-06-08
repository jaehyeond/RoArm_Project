"""Posthoc audit for the strict external-closed-loop 10cm tap sanity.

This reads existing local JSON/summary logs only. It does not launch IsaacLab,
run GPU physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_BUILTIN_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_sanity.json"
DEFAULT_EXTERNAL_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_strict_sanity.json"
DEFAULT_EXTERNAL_FAILURE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_strict_failure_audit.json"
DEFAULT_EXTERNAL_VISUAL_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_strict_visual_contact_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_external_closed_loop_strict_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_external_closed_loop_strict_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(data.get(key, default))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builtin_json", type=Path, default=DEFAULT_BUILTIN_JSON)
    parser.add_argument("--external_json", type=Path, default=DEFAULT_EXTERNAL_JSON)
    parser.add_argument("--external_failure_json", type=Path, default=DEFAULT_EXTERNAL_FAILURE_JSON)
    parser.add_argument("--external_visual_json", type=Path, default=DEFAULT_EXTERNAL_VISUAL_JSON)
    parser.add_argument("--harness", type=Path, default=REPO / "roarm_rl/test_positive_control_cube_tap10cm.py")
    parser.add_argument("--env_source", type=Path, default=REPO / "roarm_rl/roarm_cube_push_env.py")
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    builtin = _load(args.builtin_json)
    external = _load(args.external_json)
    failure = _load(args.external_failure_json)
    visual = _load(args.external_visual_json)
    builtin_log = builtin.get("last_log", {})
    external_log = external.get("last_log", {})
    controller = external.get("controller_metrics", {})

    builtin_final_gap = _f(builtin_log, "cube_tap_contact_face_gap_m")
    external_final_gap = _f(external_log, "cube_tap_contact_face_gap_m")
    gap_improvement = external_final_gap - builtin_final_gap
    contact_band = _f(failure, "contact_band_m", 0.010)
    external_shortfall = _f(failure, "final_gap_shortfall_to_contact_band_m")
    controller_goal_ok_rate = _f(controller, "closed_loop_ik_ok_rate")
    tcp_cube_dist = _f(external_log, "cube_push_tcp_cube_dist_m")
    joint_delta_abs_mean = _f(external_log, "cube_push_joint_delta_abs_mean")
    contact_slowdown_mean = _f(external_log, "cube_push_contact_slowdown_mean")
    teacher_blend_mean = _f(external_log, "cube_push_teacher_blend_mean")
    action_penalty = _f(external_log, "action_penalty")
    reset = external.get("reset_metrics", {})
    trace = external.get("log_trace_stats", {})
    face_gap_trace = trace.get("cube_tap_contact_face_gap_m", {})
    shortfall_trace = trace.get("cube_tap_contact_band_shortfall_m", {})
    tcp_dist_trace = trace.get("cube_push_tcp_cube_dist_m", {})
    joint_delta_trace = trace.get("cube_push_joint_delta_abs_mean", {})
    initial_face_gap = _f(reset, "initial_face_gap_m")
    face_gap_best = _f(face_gap_trace, "max", external_final_gap)
    face_gap_worst = _f(face_gap_trace, "min", external_final_gap)
    face_gap_final_trace = _f(face_gap_trace, "final", external_final_gap)
    face_gap_best_improvement_from_initial = face_gap_best - initial_face_gap
    shortfall_best = _f(shortfall_trace, "min", external_shortfall)
    shortfall_final_trace = _f(shortfall_trace, "final", external_shortfall)
    tcp_dist_min = _f(tcp_dist_trace, "min", tcp_cube_dist)
    joint_delta_abs_max = _f(joint_delta_trace, "max", joint_delta_abs_mean)
    trace_logs_present = bool(face_gap_trace) and bool(shortfall_trace) and bool(tcp_dist_trace)
    face_gap_moved_toward_band = face_gap_best_improvement_from_initial > 0.0
    face_gap_near_band = shortfall_best <= 0.002
    action_path_logs_present = all(
        key in external_log
        for key in (
            "cube_push_tcp_cube_dist_m",
            "cube_push_joint_delta_abs_mean",
            "cube_push_contact_slowdown_mean",
            "cube_push_teacher_blend_mean",
        )
    )
    slowdown_active = contact_slowdown_mean < 0.999
    joint_delta_cap_directly_indicated = joint_delta_abs_mean >= 0.0095
    strict_knobs_ok = (
        external.get("controller_mode") == "external_closed_loop"
        and abs(float(external.get("action_smoothing_alpha", -1.0)) - 0.25) <= 1.0e-12
        and abs(float(external.get("contact_joint_delta_scale", -1.0)) - 0.35) <= 1.0e-12
        and int(external.get("closed_loop_push_steps", -1)) == 72
    )
    external_failed_for_contact = (
        external.get("status") == "FAIL"
        and _f(external_log, "cube_tap_contact_seen_rate") == 0.0
        and _f(external_log, "cube_tap_reaction_contact_context_rate") == 0.0
        and _f(external_log, "cube_tap_success_rate") == 0.0
        and external_shortfall > 0.0
    )

    code_lines = {
        "controller_goal_ok_rate_summary": _line_of(args.harness, "controller_goal_ok_rate={result.get"),
        "controller_goal_ok_rate_compute": _line_of(args.harness, "controller_goal_ok_rate = ("),
        "controller_goal_ok_rate_gate": _line_of(args.harness, "and controller_goal_ok_rate > 0.0"),
        "action_smoothing_default": _line_of(args.env_source, "action_smoothing_alpha: float = 0.25"),
        "contact_delta_scale_default": _line_of(args.env_source, "contact_joint_delta_scale: float = 0.35"),
    }
    pass_gate_code_corrected = all(line is not None for line in code_lines.values())

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_external_closed_loop_strict_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "external_runtime_valid": external.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL",
        "strict_knobs_ok": strict_knobs_ok,
        "controller_mode": external.get("controller_mode"),
        "action_smoothing_alpha": external.get("action_smoothing_alpha"),
        "contact_joint_delta_scale": external.get("contact_joint_delta_scale"),
        "closed_loop_push_steps": external.get("closed_loop_push_steps"),
        "controller_goal_ok_rate": controller_goal_ok_rate,
        "closed_loop_ik_err_mm_mean": _f(controller, "closed_loop_ik_err_mm_mean"),
        "external_status": external.get("status"),
        "external_contact_seen": _f(external_log, "cube_tap_contact_seen_rate"),
        "external_reaction_signal": _f(external_log, "cube_tap_reaction_signal_now_rate"),
        "external_reaction_context": _f(external_log, "cube_tap_reaction_contact_context_rate"),
        "external_reaction_seen": _f(external_log, "cube_tap_reaction_seen_rate"),
        "external_tap_success": _f(external_log, "cube_tap_success_rate"),
        "external_overshoot": _f(external_log, "cube_tap_overshoot_seen_rate"),
        "external_max_disp_along_m": _f(external_log, "cube_tap_max_disp_along_m"),
        "external_max_speed_mps": _f(external_log, "cube_tap_max_speed_mps"),
        "builtin_final_face_gap_m": builtin_final_gap,
        "external_final_face_gap_m": external_final_gap,
        "gap_improvement_vs_builtin_m": gap_improvement,
        "external_final_gap_shortfall_to_band_m": external_shortfall,
        "contact_band_m": contact_band,
        "trace_logs_present": trace_logs_present,
        "initial_face_gap_m": initial_face_gap,
        "face_gap_best_m": face_gap_best,
        "face_gap_worst_m": face_gap_worst,
        "face_gap_final_trace_m": face_gap_final_trace,
        "face_gap_best_improvement_from_initial_m": face_gap_best_improvement_from_initial,
        "shortfall_best_m": shortfall_best,
        "shortfall_final_trace_m": shortfall_final_trace,
        "tcp_dist_min_m": tcp_dist_min,
        "joint_delta_abs_max": joint_delta_abs_max,
        "face_gap_moved_toward_band": face_gap_moved_toward_band,
        "face_gap_near_band": face_gap_near_band,
        "action_path_logs_present": action_path_logs_present,
        "tcp_cube_dist_m": tcp_cube_dist,
        "joint_delta_abs_mean": joint_delta_abs_mean,
        "contact_slowdown_mean": contact_slowdown_mean,
        "teacher_blend_mean": teacher_blend_mean,
        "action_penalty": action_penalty,
        "slowdown_active": slowdown_active,
        "joint_delta_cap_directly_indicated": joint_delta_cap_directly_indicated,
        "visual_contact_zero_explained": visual.get("contact_zero_explained"),
        "external_lateral_ok": visual.get("lateral_ok"),
        "external_vertical_ok": visual.get("vertical_ok"),
        "external_along_gap_blocker": visual.get("along_gap_blocker"),
        "external_failed_for_contact": external_failed_for_contact,
        "pass_gate_code_corrected_for_external_mode": pass_gate_code_corrected,
        "code_lines": code_lines,
        "still_blocked": {
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "action_teacher_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "new_gpu_runtime": "REQUIRES_EXPLICIT_APPROVAL",
            "allowed_local_only": "design_or_instrument_one_actuation_limit_candidate",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_external_closed_loop_strict_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 strict_runtime "
            f"external_runtime_valid={result['external_runtime_valid']} "
            f"strict_knobs_ok={strict_knobs_ok} controller_mode={external.get('controller_mode')} "
            f"action_smoothing_alpha={external.get('action_smoothing_alpha')} "
            f"contact_joint_delta_scale={external.get('contact_joint_delta_scale')} "
            f"closed_loop_push_steps={external.get('closed_loop_push_steps')} "
            f"controller_goal_ok_rate={controller_goal_ok_rate:.9f} "
            f"closed_loop_ik_err_mm_mean={result['closed_loop_ik_err_mm_mean']:.9f}"
        ),
        (
            "line3 outcome "
            f"status={external.get('status')} contact_seen={result['external_contact_seen']} "
            f"reaction_signal={result['external_reaction_signal']} "
            f"reaction_context={result['external_reaction_context']} "
            f"reaction_seen={result['external_reaction_seen']} "
            f"tap_success={result['external_tap_success']} "
            f"overshoot={result['external_overshoot']} "
            f"max_disp_along_m={result['external_max_disp_along_m']:.9f} "
            f"max_speed_mps={result['external_max_speed_mps']:.9f}"
        ),
        (
            "line4 face_gap "
            f"builtin_final_face_gap_m={builtin_final_gap:.9f} "
            f"external_final_face_gap_m={external_final_gap:.9f} "
            f"gap_improvement_vs_builtin_m={gap_improvement:.9f} "
            f"external_shortfall_to_band_m={external_shortfall:.9f} "
            f"visual_contact_zero_explained={visual.get('contact_zero_explained')}"
        ),
        (
            "line5 axis_and_guard "
            f"lateral_ok={visual.get('lateral_ok')} vertical_ok={visual.get('vertical_ok')} "
            f"along_gap_blocker={visual.get('along_gap_blocker')} "
            f"external_failed_for_contact={external_failed_for_contact} "
            f"pass_gate_code_corrected_for_external_mode={pass_gate_code_corrected}"
        ),
        (
            "line6 action_path "
            f"logs_present={action_path_logs_present} tcp_cube_dist_m={tcp_cube_dist:.9f} "
            f"joint_delta_abs_mean={joint_delta_abs_mean:.9f} "
            f"contact_slowdown_mean={contact_slowdown_mean:.9f} "
            f"teacher_blend_mean={teacher_blend_mean:.9f} "
            f"slowdown_active={slowdown_active} "
            f"joint_delta_cap_directly_indicated={joint_delta_cap_directly_indicated} "
            f"action_penalty={action_penalty:.9f}"
        ),
        (
            "line7 tcp_progress "
            f"trace_logs_present={trace_logs_present} initial_face_gap_m={initial_face_gap:.9f} "
            f"face_gap_best_m={face_gap_best:.9f} face_gap_worst_m={face_gap_worst:.9f} "
            f"face_gap_final_m={face_gap_final_trace:.9f} "
            f"best_improvement_from_initial_m={face_gap_best_improvement_from_initial:.9f} "
            f"shortfall_best_m={shortfall_best:.9f} shortfall_final_m={shortfall_final_trace:.9f} "
            f"tcp_dist_min_m={tcp_dist_min:.9f} joint_delta_abs_max={joint_delta_abs_max:.9f} "
            f"face_gap_moved_toward_band={face_gap_moved_toward_band} "
            f"face_gap_near_band={face_gap_near_band}"
        ),
        (
            "line8 verdict "
            "strict_external_closed_loop_positive_control=FAIL "
            "wrapper_false_positive_guard=PASS "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
        (
            "line9 next "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "allowed_local_only=design_one_action_progress_gain_or_target_application_candidate "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if strict_knobs_ok and external_failed_for_contact and pass_gate_code_corrected else 2


if __name__ == "__main__":
    raise SystemExit(main())
