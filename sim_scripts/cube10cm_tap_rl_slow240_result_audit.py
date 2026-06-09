"""Posthoc audit for the direct-IK slow240 tiny runtime.

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
DEFAULT_BASELINE_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
)
DEFAULT_SLOW240_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json"
)
DEFAULT_SLOW240_SUMMARY = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity_summary.out"
)
DEFAULT_CANDIDATE_DESIGN_JSON = LOG_DIR / "cube10cm_tap_rl_actuator_follow_time_scale_candidate_design.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_slow240_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_slow240_result_audit_summary.out"


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


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _runtime_metrics(data: dict[str, Any]) -> dict[str, Any]:
    last_log = data.get("last_log", {})
    controller = data.get("controller_metrics", {})
    return {
        "status": data.get("status"),
        "positive_control": data.get("positive_control"),
        "blocker": data.get("blocker"),
        "closed_loop_push_steps": int(data.get("closed_loop_push_steps", 0)),
        "max_steps": int(data.get("max_steps", 0)),
        "steps_executed": int(data.get("steps_executed", 0)),
        "closed_loop_alpha_final": _float(controller, "closed_loop_alpha"),
        "contact_seen": _float(last_log, "cube_tap_contact_seen_rate"),
        "reaction_signal_now": _float(last_log, "cube_tap_reaction_signal_now_rate"),
        "reaction_context": _float(last_log, "cube_tap_reaction_contact_context_rate"),
        "reaction_seen": _float(last_log, "cube_tap_reaction_seen_rate"),
        "tap_success": _float(last_log, "cube_tap_success_rate"),
        "professor_physical_reaction_seen": _float(
            last_log, "cube_tap_professor_physical_reaction_seen_rate"
        ),
        "overshoot_seen": _float(last_log, "cube_tap_overshoot_seen_rate"),
        "max_disp_along_m": _float(last_log, "cube_tap_max_disp_along_m"),
        "max_speed_mps": _float(last_log, "cube_tap_max_speed_mps"),
        "face_gap_max_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "max"),
        "face_gap_final_m": _trace(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "final"),
        "shortfall_min_m": _trace(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min"),
        "shortfall_final_m": _trace(
            data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "final"
        ),
        "target_inside_contact_band_rate_max": _trace(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
        ),
        "target_face_gap_final_m": _trace(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
        ),
        "target_fk_err_max_mm": _trace(
            data, "controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "max"
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
        "direct_joint_follow_abs_max_final_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"
        ),
        "direct_joint_follow_abs_max_max_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"
        ),
        "direct_joint_follow_abs_mean_final_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_mean_rad", "final"
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


def _target_plan(push_steps: int, max_steps: int = 120) -> dict[str, Any]:
    pre_face_gap = -0.020
    through_face_gap = 0.106
    span = through_face_gap - pre_face_gap
    band = 0.010
    inside: list[int] = []
    gaps: list[float] = []
    for step in range(max_steps):
        alpha = min(1.0, max(0.0, float(step + 1) / max(float(push_steps), 1.0)))
        face_gap = pre_face_gap + alpha * span
        gaps.append(face_gap)
        if -band <= face_gap <= band:
            inside.append(step + 1)
    return {
        "closed_loop_push_steps": push_steps,
        "max_steps": max_steps,
        "target_inside_contact_band_step_count": len(inside),
        "target_inside_contact_band_first_step_1based": inside[0] if inside else None,
        "target_inside_contact_band_last_step_1based": inside[-1] if inside else None,
        "target_final_face_gap_m": gaps[-1] if gaps else None,
        "target_face_gap_rate_m_per_step": span / max(float(push_steps), 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--slow240_json", type=Path, default=DEFAULT_SLOW240_JSON)
    parser.add_argument("--slow240_summary", type=Path, default=DEFAULT_SLOW240_SUMMARY)
    parser.add_argument("--candidate_design_json", type=Path, default=DEFAULT_CANDIDATE_DESIGN_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    baseline_raw = _load_json(args.baseline_json)
    slow240_raw = _load_json(args.slow240_json)
    candidate_design = _load_json(args.candidate_design_json) if args.candidate_design_json.exists() else {}

    baseline = _runtime_metrics(baseline_raw)
    slow240 = _runtime_metrics(slow240_raw)

    face_gap_max_gain_m = slow240["face_gap_max_m"] - baseline["face_gap_max_m"]
    shortfall_min_improvement_m = baseline["shortfall_min_m"] - slow240["shortfall_min_m"]
    follow_final_improvement_rad = (
        baseline["direct_joint_follow_abs_max_final_rad"]
        - slow240["direct_joint_follow_abs_max_final_rad"]
    )
    target_delta_final_improvement_rad = (
        baseline["target_delta_abs_max_final_rad"] - slow240["target_delta_abs_max_final_rad"]
    )
    step_final_improvement_rad = (
        baseline["direct_actual_joint_step_abs_max_final_rad"]
        - slow240["direct_actual_joint_step_abs_max_final_rad"]
    )
    follow_final_ratio = _ratio(
        slow240["direct_joint_follow_abs_max_final_rad"],
        baseline["direct_joint_follow_abs_max_final_rad"],
    )
    target_delta_final_ratio = _ratio(
        slow240["target_delta_abs_max_final_rad"],
        baseline["target_delta_abs_max_final_rad"],
    )
    target_final_face_gap_reduction_m = (
        baseline["target_face_gap_final_m"] - slow240["target_face_gap_final_m"]
    )

    target_enters_band_both = (
        baseline["target_inside_contact_band_rate_max"] > 0.0
        and slow240["target_inside_contact_band_rate_max"] > 0.0
    )
    fk_frame_ok_both = (
        baseline["actual_fk_vs_sim_tcp_err_max_mm"] <= 0.010
        and slow240["actual_fk_vs_sim_tcp_err_max_mm"] <= 0.010
    )
    wrapper_bypassed_both = (
        baseline["action_abs_max_trace"] == 0.0
        and baseline["joint_delta_cap_rate_trace"] == 0.0
        and baseline["target_lead_limit_rate_trace"] == 0.0
        and slow240["action_abs_max_trace"] == 0.0
        and slow240["joint_delta_cap_rate_trace"] == 0.0
        and slow240["target_lead_limit_rate_trace"] == 0.0
    )
    contact_gate_still_failed = slow240["contact_seen"] == 0.0 and slow240["tap_success"] == 0.0
    professor_evidence_preserved = slow240["professor_physical_reaction_seen"] > 0.0
    timing_hypothesis_supported = (
        follow_final_ratio is not None
        and follow_final_ratio < 0.60
        and shortfall_min_improvement_m > 0.0
        and face_gap_max_gain_m > 0.0
    )
    fixed_horizon_slowdown_insufficient = (
        timing_hypothesis_supported
        and contact_gate_still_failed
        and slow240["shortfall_min_m"] > 0.0
    )

    baseline_plan = _target_plan(int(baseline["closed_loop_push_steps"]), int(baseline["max_steps"]))
    slow240_plan = _target_plan(int(slow240["closed_loop_push_steps"]), int(slow240["max_steps"]))
    slow360_plan = _target_plan(360, int(slow240["max_steps"]))

    verdict = (
        "FAIL_SLOW240_IMPROVES_FOLLOW_BUT_NOT_CONTACT"
        if fixed_horizon_slowdown_insufficient
        else "RECHECK_SLOW240_RESULT"
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_slow240_result_audit_v1",
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
            "baseline_json": str(args.baseline_json),
            "slow240_json": str(args.slow240_json),
            "slow240_summary": str(args.slow240_summary),
            "candidate_design_json": str(args.candidate_design_json),
        },
        "runtime_contract": {
            "candidate_name": candidate_design.get("candidate", {}).get("name", "direct_ik_apply_slow240"),
            "changed_knobs": candidate_design.get("candidate", {}).get("changed_knobs", 1),
            "changed_knob": candidate_design.get("candidate", {}).get(
                "changed_knob", "closed_loop_push_steps 72 -> 240"
            ),
            "controller_mode": slow240_raw.get("controller_mode"),
            "num_envs": slow240_raw.get("num_envs"),
            "max_steps": slow240_raw.get("max_steps"),
            "seed": slow240_raw.get("seed"),
            "device": slow240_raw.get("device"),
            "geometry": "unchanged",
            "action_wrapper_knobs": "unchanged",
        },
        "baseline": baseline,
        "slow240": slow240,
        "comparison": {
            "face_gap_max_gain_m": face_gap_max_gain_m,
            "shortfall_min_improvement_m": shortfall_min_improvement_m,
            "follow_final_improvement_rad": follow_final_improvement_rad,
            "follow_final_ratio_slow240_over_baseline": follow_final_ratio,
            "target_delta_final_improvement_rad": target_delta_final_improvement_rad,
            "target_delta_final_ratio_slow240_over_baseline": target_delta_final_ratio,
            "actual_joint_step_final_improvement_rad": step_final_improvement_rad,
            "target_final_face_gap_reduction_m": target_final_face_gap_reduction_m,
        },
        "interpretation": {
            "target_enters_contact_band_both": target_enters_band_both,
            "fk_frame_ok_both": fk_frame_ok_both,
            "wrapper_bypassed_both": wrapper_bypassed_both,
            "timing_hypothesis_supported": timing_hypothesis_supported,
            "fixed_horizon_slowdown_insufficient": fixed_horizon_slowdown_insufficient,
            "contact_gate_still_failed": contact_gate_still_failed,
            "professor_evidence_preserved": professor_evidence_preserved,
            "primary_unresolved_blocker": "DIRECT_IK_ACTUATOR_FOLLOW_TIMING_STILL_OUTSIDE_CONTACT_BAND",
        },
        "target_timing_plans": {
            "baseline": baseline_plan,
            "slow240": slow240_plan,
            "slow360_candidate_not_run": slow360_plan,
        },
        "outcome": {
            "verdict": verdict,
            "rl_contact_gated_positive_control": "FAIL",
            "professor_physical_reaction_evidence": (
                "PASS" if professor_evidence_preserved else "FAIL"
            ),
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next_step": {
            "local_design_next": "direct_ik_apply_slow360_candidate_design_only",
            "rationale": (
                "slow240 reduced direct follow error but preserved an outside-face shortfall; "
                "a slower one-knob timing candidate can test dwell/catch-up before dataset/RL/RoArm"
            ),
            "not_runtime_approved_by_this_audit": True,
            "not_allowed": "dataset/RL/RoArm/action-teacher until contact-gated positive-control passes",
        },
        "outputs": {
            "json": str(args.out_json),
            "summary": str(args.out_summary),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_slow240_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            "candidate=direct_ik_apply_slow240 changed_knobs=1 closed_loop_push_steps=72->240 "
            f"controller_mode={slow240_raw.get('controller_mode')} num_envs={slow240_raw.get('num_envs')} "
            f"max_steps={slow240_raw.get('max_steps')} seed={slow240_raw.get('seed')} "
            f"device={slow240_raw.get('device')} geometry=unchanged action_wrapper_knobs=unchanged"
        ),
        (
            "line3 outcome "
            f"baseline_contact={baseline['contact_seen']:.1f} baseline_tap={baseline['tap_success']:.1f} "
            f"slow240_contact={slow240['contact_seen']:.1f} slow240_tap={slow240['tap_success']:.1f} "
            f"slow240_professor_seen={slow240['professor_physical_reaction_seen']:.1f} "
            f"slow240_reaction_signal_now={slow240['reaction_signal_now']:.1f}"
        ),
        (
            "line4 follow_comparison "
            f"baseline_follow_final_rad={baseline['direct_joint_follow_abs_max_final_rad']:.9f} "
            f"slow240_follow_final_rad={slow240['direct_joint_follow_abs_max_final_rad']:.9f} "
            f"follow_final_ratio={float(follow_final_ratio or 0.0):.9f} "
            f"target_delta_ratio={float(target_delta_final_ratio or 0.0):.9f} "
            f"actual_step_final_rad={slow240['direct_actual_joint_step_abs_max_final_rad']:.9f}"
        ),
        (
            "line5 contact_gap_comparison "
            f"baseline_shortfall_min_m={baseline['shortfall_min_m']:.9f} "
            f"slow240_shortfall_min_m={slow240['shortfall_min_m']:.9f} "
            f"shortfall_improvement_m={shortfall_min_improvement_m:.9f} "
            f"baseline_face_gap_max_m={baseline['face_gap_max_m']:.9f} "
            f"slow240_face_gap_max_m={slow240['face_gap_max_m']:.9f} "
            f"face_gap_gain_m={face_gap_max_gain_m:.9f}"
        ),
        (
            "line6 exclusions "
            f"target_enters_band_both={target_enters_band_both} fk_frame_ok_both={fk_frame_ok_both} "
            f"wrapper_bypassed_both={wrapper_bypassed_both} "
            f"baseline_target_final_face_gap_m={baseline['target_face_gap_final_m']:.9f} "
            f"slow240_target_final_face_gap_m={slow240['target_face_gap_final_m']:.9f} "
            f"target_final_face_gap_reduction_m={target_final_face_gap_reduction_m:.9f}"
        ),
        (
            "line7 verdict "
            f"{verdict} timing_hypothesis_supported={timing_hypothesis_supported} "
            f"fixed_horizon_slowdown_insufficient={fixed_horizon_slowdown_insufficient} "
            "rl_contact_gated_positive_control=FAIL diffik_action_dataset=BLOCKED "
            "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line8 next_local_only "
            "design_next=direct_ik_apply_slow360_candidate_design_only runtime_not_approved_by_this_audit=YES "
            f"slow360_inside_steps={slow360_plan['target_inside_contact_band_step_count']} "
            f"slow360_final_face_gap_m={float(slow360_plan['target_final_face_gap_m']):.9f} "
            "dataset_rl_roarm_action_teacher=STILL_BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if verdict == "FAIL_SLOW240_IMPROVES_FOLLOW_BUT_NOT_CONTACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
