"""Posthoc audit for the action_smoothing_alpha=1.0 positive-control run.

This reads existing local JSON logs only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
BASELINE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_tcp_progress_sanity.json"
CURRENT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_action_smoothing1_sanity.json"
CURRENT_FAILURE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_action_smoothing1_failure_audit.json"
CURRENT_VISUAL_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_action_smoothing1_visual_contact_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_action_smoothing1_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_action_smoothing1_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace_value(run: dict[str, Any], stat_key: str, field: str, default: float = 0.0) -> float:
    return _f(run.get("log_trace_stats", {}).get(stat_key, {}), field, default)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", type=Path, default=BASELINE_JSON)
    parser.add_argument("--current_json", type=Path, default=CURRENT_JSON)
    parser.add_argument("--current_failure_json", type=Path, default=CURRENT_FAILURE_JSON)
    parser.add_argument("--current_visual_json", type=Path, default=CURRENT_VISUAL_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    baseline = _load(args.baseline_json)
    current = _load(args.current_json)
    failure = _load(args.current_failure_json)
    visual = _load(args.current_visual_json)
    base_log = baseline.get("last_log", {})
    cur_log = current.get("last_log", {})
    cur_controller = current.get("controller_metrics", {})

    base_initial_gap = _f(baseline.get("reset_metrics", {}), "initial_face_gap_m")
    cur_initial_gap = _f(current.get("reset_metrics", {}), "initial_face_gap_m")
    base_best_gap = _trace_value(baseline, "cube_tap_contact_face_gap_m", "max")
    cur_best_gap = _trace_value(current, "cube_tap_contact_face_gap_m", "max")
    base_final_gap = _trace_value(baseline, "cube_tap_contact_face_gap_m", "final", _f(base_log, "cube_tap_contact_face_gap_m"))
    cur_final_gap = _trace_value(current, "cube_tap_contact_face_gap_m", "final", _f(cur_log, "cube_tap_contact_face_gap_m"))
    base_best_shortfall = _trace_value(baseline, "cube_tap_contact_band_shortfall_m", "min")
    cur_best_shortfall = _trace_value(current, "cube_tap_contact_band_shortfall_m", "min")
    base_final_shortfall = _trace_value(baseline, "cube_tap_contact_band_shortfall_m", "final")
    cur_final_shortfall = _trace_value(current, "cube_tap_contact_band_shortfall_m", "final")

    base_best_improvement = base_best_gap - base_initial_gap
    cur_best_improvement = cur_best_gap - cur_initial_gap
    best_improvement_delta = cur_best_improvement - base_best_improvement
    best_shortfall_delta = cur_best_shortfall - base_best_shortfall
    final_gap_delta_vs_baseline = cur_final_gap - base_final_gap
    max_disp_delta = _f(cur_log, "cube_tap_max_disp_along_m") - _f(base_log, "cube_tap_max_disp_along_m")
    max_speed_delta = _f(cur_log, "cube_tap_max_speed_mps") - _f(base_log, "cube_tap_max_speed_mps")

    current_runtime_valid = (
        current.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL"
        and current.get("device") == "cuda:0"
        and current.get("dataset_generation") is False
        and current.get("training") is False
        and current.get("robot_control") is False
        and current.get("ssh") is False
        and current.get("b200") is False
        and current.get("track_a") is False
    )
    changed_knob_ok = (
        baseline.get("controller_mode") == "external_closed_loop"
        and current.get("controller_mode") == "external_closed_loop"
        and abs(float(baseline.get("action_smoothing_alpha", -1.0)) - 0.25) <= 1.0e-12
        and abs(float(current.get("action_smoothing_alpha", -1.0)) - 1.0) <= 1.0e-12
        and baseline.get("contact_joint_delta_scale") == current.get("contact_joint_delta_scale")
        and baseline.get("closed_loop_push_steps") == current.get("closed_loop_push_steps")
        and baseline.get("fixed_cube_x_m") == current.get("fixed_cube_x_m")
        and baseline.get("fixed_cube_y_m") == current.get("fixed_cube_y_m")
        and baseline.get("fixed_push_dir_x") == current.get("fixed_push_dir_x")
        and baseline.get("fixed_push_dir_y") == current.get("fixed_push_dir_y")
        and baseline.get("precontact_clearance_m") == current.get("precontact_clearance_m")
        and baseline.get("tcp_top_margin_m") == current.get("tcp_top_margin_m")
        and baseline.get("goal_push_m") == current.get("goal_push_m")
    )
    primary_pass = (
        _f(cur_log, "cube_tap_contact_seen_rate") > 0.0
        and _f(cur_log, "cube_tap_reaction_contact_context_rate") > 0.0
        and _f(cur_log, "cube_tap_reaction_seen_rate") > 0.0
        and _f(cur_log, "cube_tap_success_rate") > 0.0
        and _f(cur_log, "cube_tap_overshoot_seen_rate") == 0.0
    )
    smoothing_improved_contact_progress = (
        cur_best_shortfall < base_best_shortfall - 0.001
        or cur_best_improvement > base_best_improvement + 0.001
    )
    smoothing_hypothesis_supported = primary_pass or smoothing_improved_contact_progress

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_action_smoothing1_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "current_runtime_valid": current_runtime_valid,
        "changed_knob_ok": changed_knob_ok,
        "baseline_action_smoothing_alpha": baseline.get("action_smoothing_alpha"),
        "current_action_smoothing_alpha": current.get("action_smoothing_alpha"),
        "controller_mode": current.get("controller_mode"),
        "closed_loop_push_steps": current.get("closed_loop_push_steps"),
        "contact_joint_delta_scale": current.get("contact_joint_delta_scale"),
        "closed_loop_ik_ok_rate": _f(cur_controller, "closed_loop_ik_ok_rate"),
        "closed_loop_ik_err_mm_mean": _f(cur_controller, "closed_loop_ik_err_mm_mean"),
        "contact_seen": _f(cur_log, "cube_tap_contact_seen_rate"),
        "reaction_signal": _f(cur_log, "cube_tap_reaction_signal_now_rate"),
        "reaction_context": _f(cur_log, "cube_tap_reaction_contact_context_rate"),
        "reaction_seen": _f(cur_log, "cube_tap_reaction_seen_rate"),
        "tap_success": _f(cur_log, "cube_tap_success_rate"),
        "overshoot": _f(cur_log, "cube_tap_overshoot_seen_rate"),
        "baseline_best_improvement_m": base_best_improvement,
        "current_best_improvement_m": cur_best_improvement,
        "best_improvement_delta_m": best_improvement_delta,
        "baseline_shortfall_best_m": base_best_shortfall,
        "current_shortfall_best_m": cur_best_shortfall,
        "best_shortfall_delta_m": best_shortfall_delta,
        "baseline_final_gap_m": base_final_gap,
        "current_final_gap_m": cur_final_gap,
        "final_gap_delta_vs_baseline_m": final_gap_delta_vs_baseline,
        "current_final_shortfall_m": cur_final_shortfall,
        "baseline_joint_delta_abs_max": _trace_value(baseline, "cube_push_joint_delta_abs_mean", "max"),
        "current_joint_delta_abs_max": _trace_value(current, "cube_push_joint_delta_abs_mean", "max"),
        "current_contact_slowdown_mean": _f(cur_log, "cube_push_contact_slowdown_mean"),
        "current_tcp_dist_min_m": _trace_value(current, "cube_push_tcp_cube_dist_m", "min"),
        "max_disp_delta_m": max_disp_delta,
        "max_speed_delta_mps": max_speed_delta,
        "visual_contact_zero_explained": visual.get("contact_zero_explained"),
        "failure_is_controller_gap": failure.get("failure_is_controller_gap"),
        "primary_pass": primary_pass,
        "smoothing_improved_contact_progress": smoothing_improved_contact_progress,
        "smoothing_hypothesis_supported": smoothing_hypothesis_supported,
        "verdict": "FAIL_SMOOTHING_NOT_ROOT_CAUSE" if not smoothing_hypothesis_supported else "CHECK_PRIMARY_PASS",
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed_local_only": "target_reference_or_action_command_magnitude_design",
            "new_gpu_runtime": "REQUIRES_EXPLICIT_APPROVAL",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_action_smoothing1_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            f"current_runtime_valid={current_runtime_valid} changed_knob_ok={changed_knob_ok} "
            f"controller_mode={current.get('controller_mode')} "
            f"action_smoothing_alpha={baseline.get('action_smoothing_alpha')}->{current.get('action_smoothing_alpha')} "
            f"contact_joint_delta_scale={current.get('contact_joint_delta_scale')} "
            f"closed_loop_push_steps={current.get('closed_loop_push_steps')}"
        ),
        (
            "line3 outcome "
            f"status={current.get('status')} contact_seen={result['contact_seen']} "
            f"reaction_signal={result['reaction_signal']} reaction_context={result['reaction_context']} "
            f"reaction_seen={result['reaction_seen']} tap_success={result['tap_success']} "
            f"overshoot={result['overshoot']} primary_pass={primary_pass}"
        ),
        (
            "line4 tcp_progress_compare "
            f"baseline_best_improvement_m={base_best_improvement:.9f} "
            f"current_best_improvement_m={cur_best_improvement:.9f} "
            f"best_improvement_delta_m={best_improvement_delta:.9f} "
            f"baseline_shortfall_best_m={base_best_shortfall:.9f} "
            f"current_shortfall_best_m={cur_best_shortfall:.9f} "
            f"best_shortfall_delta_m={best_shortfall_delta:.9f}"
        ),
        (
            "line5 final_and_action_path "
            f"baseline_final_gap_m={base_final_gap:.9f} current_final_gap_m={cur_final_gap:.9f} "
            f"final_gap_delta_vs_baseline_m={final_gap_delta_vs_baseline:.9f} "
            f"current_final_shortfall_m={cur_final_shortfall:.9f} "
            f"baseline_joint_delta_abs_max={result['baseline_joint_delta_abs_max']:.9f} "
            f"current_joint_delta_abs_max={result['current_joint_delta_abs_max']:.9f} "
            f"contact_slowdown_mean={result['current_contact_slowdown_mean']:.9f}"
        ),
        (
            "line6 interpretation "
            f"smoothing_improved_contact_progress={smoothing_improved_contact_progress} "
            f"smoothing_hypothesis_supported={smoothing_hypothesis_supported} "
            f"failure_is_controller_gap={failure.get('failure_is_controller_gap')} "
            f"visual_contact_zero_explained={visual.get('contact_zero_explained')}"
        ),
        (
            "line7 verdict "
            f"{result['verdict']} diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line8 next "
            "allowed_local_only=target_reference_or_action_command_magnitude_design "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if current_runtime_valid and changed_knob_ok and not smoothing_hypothesis_supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
