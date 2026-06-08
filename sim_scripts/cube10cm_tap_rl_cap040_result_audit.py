"""Posthoc audit for the cap040 10cm tap positive-control runtime.

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
BASELINE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap_targetlead_diagnostic_sanity.json"
CAP040_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_sanity.json"
CAP040_FAILURE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_failure_audit.json"
CAP040_VISUAL_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_visual_contact_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_cap040_final_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_cap040_final_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(run: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    return _f(run.get("log_trace_stats", {}).get(key, {}), field, default)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", type=Path, default=BASELINE_JSON)
    parser.add_argument("--cap040_json", type=Path, default=CAP040_JSON)
    parser.add_argument("--failure_json", type=Path, default=CAP040_FAILURE_JSON)
    parser.add_argument("--visual_json", type=Path, default=CAP040_VISUAL_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    baseline = _load(args.baseline_json)
    cap040 = _load(args.cap040_json)
    failure = _load(args.failure_json)
    visual = _load(args.visual_json)
    cap040_log = cap040.get("last_log", {})

    baseline_cap_rate = _trace(baseline, "cube_push_joint_delta_cap_rate", "max")
    cap040_cap_rate = _trace(cap040, "cube_push_joint_delta_cap_rate", "max")
    baseline_delta_max = _trace(baseline, "cube_push_joint_delta_abs_max", "max")
    cap040_delta_max = _trace(cap040, "cube_push_joint_delta_abs_max", "max")
    baseline_best_shortfall = _trace(baseline, "cube_tap_contact_band_shortfall_m", "min")
    cap040_best_shortfall = _trace(cap040, "cube_tap_contact_band_shortfall_m", "min")
    baseline_final_shortfall = _trace(baseline, "cube_tap_contact_band_shortfall_m", "final")
    cap040_final_shortfall = _trace(cap040, "cube_tap_contact_band_shortfall_m", "final")
    baseline_face_best = _trace(baseline, "cube_tap_contact_face_gap_m", "max")
    cap040_face_best = _trace(cap040, "cube_tap_contact_face_gap_m", "max")
    shortfall_delta = cap040_best_shortfall - baseline_best_shortfall
    face_best_delta = cap040_face_best - baseline_face_best

    cap_override_applied = (
        abs(float(cap040.get("max_joint_delta_per_step_rad", 0.0)) - 0.040) <= 1.0e-9
        and cap040_delta_max >= 0.039
    )
    cap_no_longer_active = cap040_cap_rate == 0.0
    cap_only_did_not_improve_gap = abs(shortfall_delta) <= 1.0e-7 and abs(face_best_delta) <= 1.0e-7
    lead_limit_observed = _trace(cap040, "cube_push_target_lead_limit_rate", "max") > 0.0
    action_saturation_observed = _trace(cap040, "cube_push_action_abs_max", "max") >= 0.999
    slowdown_observed = _f(cap040_log, "cube_push_contact_slowdown_mean", 1.0) < 0.999
    cap_only_falsified_as_primary = (
        cap_override_applied
        and cap_no_longer_active
        and cap_only_did_not_improve_gap
        and _f(cap040_log, "cube_tap_contact_seen_rate") == 0.0
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_cap040_final_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "runtime_valid": cap040.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL",
        "cap040_status": cap040.get("status"),
        "device": cap040.get("device"),
        "controller_mode": cap040.get("controller_mode"),
        "max_joint_delta_per_step_rad": cap040.get("max_joint_delta_per_step_rad"),
        "contact_seen": _f(cap040_log, "cube_tap_contact_seen_rate"),
        "reaction_signal": _f(cap040_log, "cube_tap_reaction_signal_now_rate"),
        "reaction_context": _f(cap040_log, "cube_tap_reaction_contact_context_rate"),
        "tap_success": _f(cap040_log, "cube_tap_success_rate"),
        "overshoot": _f(cap040_log, "cube_tap_overshoot_seen_rate"),
        "baseline_cap_rate_max": baseline_cap_rate,
        "cap040_cap_rate_max": cap040_cap_rate,
        "baseline_joint_delta_abs_max_trace": baseline_delta_max,
        "cap040_joint_delta_abs_max_trace": cap040_delta_max,
        "baseline_best_shortfall_m": baseline_best_shortfall,
        "cap040_best_shortfall_m": cap040_best_shortfall,
        "shortfall_delta_m": shortfall_delta,
        "baseline_final_shortfall_m": baseline_final_shortfall,
        "cap040_final_shortfall_m": cap040_final_shortfall,
        "baseline_face_gap_best_m": baseline_face_best,
        "cap040_face_gap_best_m": cap040_face_best,
        "face_gap_best_delta_m": face_best_delta,
        "target_lead_abs_max_trace": _trace(cap040, "cube_push_target_lead_abs_max", "max"),
        "target_lead_limit_rate_trace": _trace(cap040, "cube_push_target_lead_limit_rate", "max"),
        "action_abs_max_trace": _trace(cap040, "cube_push_action_abs_max", "max"),
        "action_saturation_observed": action_saturation_observed,
        "lead_limit_observed": lead_limit_observed,
        "slowdown_observed": slowdown_observed,
        "failure_is_controller_gap": failure.get("failure_is_controller_gap"),
        "visual_contact_zero_explained": visual.get("contact_zero_explained"),
        "cap_override_applied": cap_override_applied,
        "cap_no_longer_active": cap_no_longer_active,
        "cap_only_did_not_improve_gap": cap_only_did_not_improve_gap,
        "cap_only_falsified_as_primary": cap_only_falsified_as_primary,
        "verdict": (
            "CAP_ONLY_FALSIFIED_AS_PRIMARY_NEXT_TARGET_APPLICATION_DIAGNOSTIC"
            if cap_only_falsified_as_primary and lead_limit_observed
            else "REQUIRES_MORE_LOCAL_REVIEW"
        ),
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed_local_only": "design_one_target_application_candidate",
            "candidate_family": "joint_target_lead_limit_or_joint_delta_reference",
            "new_gpu_runtime": "REQUIRES_EXPLICIT_APPROVAL",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_cap040_final_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            f"runtime_valid={result['runtime_valid']} status={result['cap040_status']} "
            f"device={result['device']} controller_mode={result['controller_mode']} "
            f"max_joint_delta_per_step_rad={result['max_joint_delta_per_step_rad']}"
        ),
        (
            "line3 outcome "
            f"contact_seen={result['contact_seen']} reaction_signal={result['reaction_signal']} "
            f"reaction_context={result['reaction_context']} tap_success={result['tap_success']} "
            f"overshoot={result['overshoot']}"
        ),
        (
            "line4 cap_effect "
            f"baseline_cap_rate_max={baseline_cap_rate:.9f} cap040_cap_rate_max={cap040_cap_rate:.9f} "
            f"baseline_joint_delta_abs_max_trace={baseline_delta_max:.9f} "
            f"cap040_joint_delta_abs_max_trace={cap040_delta_max:.9f} "
            f"cap_override_applied={cap_override_applied} cap_no_longer_active={cap_no_longer_active}"
        ),
        (
            "line5 contact_progress "
            f"baseline_best_shortfall_m={baseline_best_shortfall:.9f} "
            f"cap040_best_shortfall_m={cap040_best_shortfall:.9f} "
            f"shortfall_delta_m={shortfall_delta:.9f} "
            f"baseline_face_gap_best_m={baseline_face_best:.9f} "
            f"cap040_face_gap_best_m={cap040_face_best:.9f} "
            f"face_gap_best_delta_m={face_best_delta:.9f}"
        ),
        (
            "line6 target_application "
            f"action_abs_max_trace={result['action_abs_max_trace']:.9f} "
            f"target_lead_abs_max_trace={result['target_lead_abs_max_trace']:.9f} "
            f"target_lead_limit_rate_trace={result['target_lead_limit_rate_trace']:.9f} "
            f"lead_limit_observed={lead_limit_observed} slowdown_observed={slowdown_observed}"
        ),
        (
            "line7 verdict "
            f"{result['verdict']} cap_only_falsified_as_primary={cap_only_falsified_as_primary} "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line8 next "
            "allowed_local_only=design_one_target_application_candidate "
            "candidate_family=joint_target_lead_limit_or_joint_delta_reference "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if cap_only_falsified_as_primary else 2


if __name__ == "__main__":
    raise SystemExit(main())
