"""Posthoc audit for the cap/target-lead diagnostic 10cm tap runtime.

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
DIAG_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap_targetlead_diagnostic_sanity.json"
FAILURE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap_targetlead_diagnostic_failure_audit.json"
VISUAL_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap_targetlead_diagnostic_visual_contact_audit.json"
ENV_SOURCE = REPO / "roarm_rl/roarm_cube_push_env.py"
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_cap_targetlead_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_cap_targetlead_result_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(run: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    return _f(run.get("log_trace_stats", {}).get(key, {}), field, default)


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag_json", type=Path, default=DIAG_JSON)
    parser.add_argument("--failure_json", type=Path, default=FAILURE_JSON)
    parser.add_argument("--visual_json", type=Path, default=VISUAL_JSON)
    parser.add_argument("--env_source", type=Path, default=ENV_SOURCE)
    parser.add_argument("--harness", type=Path, default=HARNESS)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    diag = _load(args.diag_json)
    failure = _load(args.failure_json)
    visual = _load(args.visual_json)
    log = diag.get("last_log", {})
    controller = diag.get("controller_metrics", {})

    action_abs_max = _f(log, "cube_push_action_abs_max")
    action_abs_mean = _f(log, "cube_push_action_abs_mean")
    joint_delta_abs_max = _f(log, "cube_push_joint_delta_abs_max")
    joint_delta_abs_mean = _f(log, "cube_push_joint_delta_abs_mean")
    cap_rate = _f(log, "cube_push_joint_delta_cap_rate")
    slowdown = _f(log, "cube_push_contact_slowdown_mean")
    target_lead_abs_max_final = _f(log, "cube_push_target_lead_abs_max")
    target_lead_limit_rate_final = _f(log, "cube_push_target_lead_limit_rate")
    target_lead_abs_max_trace = _trace(diag, "cube_push_target_lead_abs_max", "max")
    target_lead_limit_rate_trace = _trace(diag, "cube_push_target_lead_limit_rate", "max")
    cap_rate_trace = _trace(diag, "cube_push_joint_delta_cap_rate", "max")
    action_abs_max_trace = _trace(diag, "cube_push_action_abs_max", "max")
    joint_delta_abs_max_trace = _trace(diag, "cube_push_joint_delta_abs_max", "max")
    shortfall_best = _trace(diag, "cube_tap_contact_band_shortfall_m", "min")
    shortfall_final = _trace(diag, "cube_tap_contact_band_shortfall_m", "final")
    face_gap_best = _trace(diag, "cube_tap_contact_face_gap_m", "max")
    face_gap_final = _trace(diag, "cube_tap_contact_face_gap_m", "final")

    action_saturation_observed = action_abs_max_trace >= 0.999
    per_joint_cap_observed = cap_rate_trace > 0.0 and joint_delta_abs_max_trace >= 0.0095
    slowdown_observed = slowdown < 0.999
    lead_limit_observed = target_lead_limit_rate_trace > 0.0
    lead_limit_primary = lead_limit_observed and not per_joint_cap_observed
    cap_is_primary_current_hypothesis = per_joint_cap_observed and action_saturation_observed and not slowdown_observed

    code_lines = {
        "action_scale_default": _line_of(args.env_source, "action_scale: float = 0.04"),
        "max_joint_delta_default": _line_of(args.env_source, "max_joint_delta_per_step_rad: float = 0.010"),
        "joint_delta_cap_rate_log": _line_of(args.env_source, "cube_push_joint_delta_cap_rate"),
        "target_lead_limit_rate_log": _line_of(args.env_source, "cube_push_target_lead_limit_rate"),
        "harness_line7_action_path": _line_of(args.harness, "joint_delta_cap_rate="),
        "harness_line8_trace": _line_of(args.harness, "target_lead_limit_rate_max="),
    }

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_cap_targetlead_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "runtime_valid": diag.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL",
        "controller_mode": diag.get("controller_mode"),
        "action_smoothing_alpha": diag.get("action_smoothing_alpha"),
        "contact_joint_delta_scale": diag.get("contact_joint_delta_scale"),
        "closed_loop_push_steps": diag.get("closed_loop_push_steps"),
        "closed_loop_ik_ok_rate": _f(controller, "closed_loop_ik_ok_rate"),
        "closed_loop_ik_err_mm_mean": _f(controller, "closed_loop_ik_err_mm_mean"),
        "contact_seen": _f(log, "cube_tap_contact_seen_rate"),
        "reaction_signal": _f(log, "cube_tap_reaction_signal_now_rate"),
        "reaction_context": _f(log, "cube_tap_reaction_contact_context_rate"),
        "tap_success": _f(log, "cube_tap_success_rate"),
        "overshoot": _f(log, "cube_tap_overshoot_seen_rate"),
        "face_gap_best_m": face_gap_best,
        "face_gap_final_m": face_gap_final,
        "shortfall_best_m": shortfall_best,
        "shortfall_final_m": shortfall_final,
        "action_abs_mean": action_abs_mean,
        "action_abs_max": action_abs_max,
        "action_abs_max_trace": action_abs_max_trace,
        "joint_delta_abs_mean": joint_delta_abs_mean,
        "joint_delta_abs_max": joint_delta_abs_max,
        "joint_delta_abs_max_trace": joint_delta_abs_max_trace,
        "joint_delta_cap_rate": cap_rate,
        "joint_delta_cap_rate_trace": cap_rate_trace,
        "contact_slowdown_mean": slowdown,
        "target_lead_abs_max_final": target_lead_abs_max_final,
        "target_lead_abs_max_trace": target_lead_abs_max_trace,
        "target_lead_limit_rate_final": target_lead_limit_rate_final,
        "target_lead_limit_rate_trace": target_lead_limit_rate_trace,
        "action_saturation_observed": action_saturation_observed,
        "per_joint_cap_observed": per_joint_cap_observed,
        "slowdown_observed": slowdown_observed,
        "lead_limit_observed": lead_limit_observed,
        "lead_limit_primary": lead_limit_primary,
        "cap_is_primary_current_hypothesis": cap_is_primary_current_hypothesis,
        "failure_is_controller_gap": failure.get("failure_is_controller_gap"),
        "visual_contact_zero_explained": visual.get("contact_zero_explained"),
        "code_lines": code_lines,
        "verdict": "CAP_ACTION_SATURATION_PRIMARY_HYPOTHESIS"
        if cap_is_primary_current_hypothesis
        else "REQUIRES_MORE_LOCAL_DIAGNOSIS",
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed_local_only": "design_one_cap_only_positive_control_candidate",
            "new_gpu_runtime": "REQUIRES_EXPLICIT_APPROVAL",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_cap_targetlead_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            f"runtime_valid={result['runtime_valid']} controller_mode={diag.get('controller_mode')} "
            f"action_smoothing_alpha={diag.get('action_smoothing_alpha')} "
            f"contact_joint_delta_scale={diag.get('contact_joint_delta_scale')} "
            f"closed_loop_push_steps={diag.get('closed_loop_push_steps')} "
            f"closed_loop_ik_ok_rate={result['closed_loop_ik_ok_rate']:.9f}"
        ),
        (
            "line3 outcome "
            f"contact_seen={result['contact_seen']} reaction_signal={result['reaction_signal']} "
            f"reaction_context={result['reaction_context']} tap_success={result['tap_success']} "
            f"overshoot={result['overshoot']} shortfall_best_m={shortfall_best:.9f} "
            f"shortfall_final_m={shortfall_final:.9f}"
        ),
        (
            "line4 cap_action "
            f"action_abs_mean={action_abs_mean:.9f} action_abs_max={action_abs_max:.9f} "
            f"action_abs_max_trace={action_abs_max_trace:.9f} "
            f"joint_delta_abs_mean={joint_delta_abs_mean:.9f} "
            f"joint_delta_abs_max={joint_delta_abs_max:.9f} "
            f"joint_delta_abs_max_trace={joint_delta_abs_max_trace:.9f} "
            f"joint_delta_cap_rate={cap_rate:.9f} joint_delta_cap_rate_trace={cap_rate_trace:.9f}"
        ),
        (
            "line5 lead_slowdown "
            f"contact_slowdown_mean={slowdown:.9f} "
            f"target_lead_abs_max_final={target_lead_abs_max_final:.9f} "
            f"target_lead_abs_max_trace={target_lead_abs_max_trace:.9f} "
            f"target_lead_limit_rate_final={target_lead_limit_rate_final:.9f} "
            f"target_lead_limit_rate_trace={target_lead_limit_rate_trace:.9f}"
        ),
        (
            "line6 interpretation "
            f"action_saturation_observed={action_saturation_observed} "
            f"per_joint_cap_observed={per_joint_cap_observed} "
            f"slowdown_observed={slowdown_observed} "
            f"lead_limit_observed={lead_limit_observed} "
            f"lead_limit_primary={lead_limit_primary} "
            f"cap_is_primary_current_hypothesis={cap_is_primary_current_hypothesis}"
        ),
        (
            "line7 verdict "
            f"{result['verdict']} diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line8 next "
            "allowed_local_only=design_one_cap_only_positive_control_candidate "
            "new_gpu_runtime=REQUIRES_EXPLICIT_APPROVAL "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if result["cap_is_primary_current_hypothesis"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
