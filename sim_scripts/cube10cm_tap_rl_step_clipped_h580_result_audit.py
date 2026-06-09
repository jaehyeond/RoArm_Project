#!/usr/bin/env python3
"""Posthoc audit for the step-clipped h580 horizon/progress runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_H580_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_sanity.json"
DEFAULT_STEP120_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_sanity.json"
DEFAULT_DESIGN_JSON = LOG_DIR / "cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_step_clipped_h580_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_step_clipped_h580_result_audit_summary.out"


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
        "max_steps": int(data.get("max_steps", 0)),
        "steps_executed": int(data.get("steps_executed", 0)),
        "closed_loop_push_steps": int(data.get("closed_loop_push_steps", 0)),
        "controller_goal_ok_rate": _float(data, "controller_goal_ok_rate"),
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
        "tap_success": _float(last_log, "cube_tap_success_rate"),
        "overshoot_seen": _float(last_log, "cube_tap_overshoot_seen_rate"),
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h580_json", type=Path, default=DEFAULT_H580_JSON)
    parser.add_argument("--step120_json", type=Path, default=DEFAULT_STEP120_JSON)
    parser.add_argument("--design_json", type=Path, default=DEFAULT_DESIGN_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    h580 = _metrics(_load(args.h580_json))
    step120 = _metrics(_load(args.step120_json))
    design = _load(args.design_json)

    contact_gate_failed = h580["contact_seen"] == 0.0 and h580["tap_success"] == 0.0
    h580_not_continuous = h580["truncated_count"] > 0
    h580_no_better_than_step120 = h580["shortfall_min_m"] >= step120["shortfall_min_m"]
    env_episode_cap_blocks_horizon_test = h580_not_continuous and contact_gate_failed

    if env_episode_cap_blocks_horizon_test:
        primary_blocker = "ENV_EPISODE_LENGTH_1P2S_TRUNCATES_H580_HORIZON_TEST"
    elif h580_no_better_than_step120:
        primary_blocker = "H580_CONTINUOUS_HORIZON_DID_NOT_IMPROVE_CONTACT"
    else:
        primary_blocker = "RECHECK_H580_RESULT"

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_step_clipped_h580_result_audit_v1",
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
            "h580_json": str(args.h580_json),
            "step120_json": str(args.step120_json),
            "design_json": str(args.design_json),
        },
        "runtime": h580,
        "baseline_step120": step120,
        "design_candidate": design.get("candidate", {}),
        "interpretation": {
            "contact_gate_failed": contact_gate_failed,
            "h580_not_continuous_due_truncation": h580_not_continuous,
            "h580_no_better_than_step120": h580_no_better_than_step120,
            "env_episode_cap_blocks_horizon_test": env_episode_cap_blocks_horizon_test,
            "do_not_claim_h580_falsified_continuous_horizon": True,
            "do_not_relax_contact_gate_yet": True,
            "why": (
                "The 580-step loop did not create one continuous 5.8s episode. The 10cm env still truncated "
                "at its 1.2s episode contract, so the horizon/progress hypothesis was not cleanly tested."
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
            "local_unblock": "design_default_off_episode_length_override_then_repeat_step_clipped_horizon",
            "not_dataset_or_rl": True,
            "not_contact_gate_relaxation": True,
        },
        "outputs": {"json": str(args.out_json), "summary": str(args.out_summary)},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_step_clipped_h580_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 runtime_outcome "
        f"status={h580['status']} max_steps={h580['max_steps']} steps_executed={h580['steps_executed']} "
        f"closed_loop_push_steps={h580['closed_loop_push_steps']} contact_seen={h580['contact_seen']:.1f} "
        f"tap_success={h580['tap_success']:.1f} professor_evidence={h580['professor_physical_reaction_evidence']} "
        f"terminated_count={h580['terminated_count']} truncated_count={h580['truncated_count']}",
        "line3 horizon_contract "
        f"designed_runtime_s=5.800 actual_loop_steps={h580['steps_executed']} "
        f"env_episode_cap_detected={h580_not_continuous} "
        "reason=truncated_count_positive_means_not_one_continuous_5p8s_episode",
        "line4 actual_contact "
        f"h580_face_gap_max_m={h580['face_gap_max_m']:.9f} "
        f"h580_shortfall_min_m={h580['shortfall_min_m']:.9f} "
        f"h580_shortfall_final_m={h580['shortfall_final_m']:.9f} "
        f"step120_shortfall_min_m={step120['shortfall_min_m']:.9f} "
        f"h580_no_better_than_step120={h580_no_better_than_step120}",
        "line5 controller "
        f"target_inside_max={h580['target_inside_max']:.1f} "
        f"target_face_gap_final_m={h580['target_face_gap_final_m']:.9f} "
        f"clip_rate_final={h580['builtin_diffik_step_clip_rate']:.9f} "
        f"follow_final_rad={h580['follow_final_rad']:.9f} "
        f"actual_step_final_rad={h580['actual_step_final_rad']:.9f}",
        "line6 verdict "
        f"primary_blocker={primary_blocker} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line7 next "
        "local_unblock=design_default_off_episode_length_override_then_repeat_step_clipped_horizon "
        "contact_gate_relaxation=NO dataset_rl_roarm=NO",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if primary_blocker.startswith("ENV_EPISODE_LENGTH") else 2


if __name__ == "__main__":
    raise SystemExit(main())
