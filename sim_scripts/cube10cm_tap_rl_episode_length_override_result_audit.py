#!/usr/bin/env python3
"""Posthoc audit for the h580 episode-length override runtime.

This is local-only. It does not launch IsaacLab/GPU runtime, generate datasets,
train, control RoArm, SSH, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_EP608_JSON = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_sanity.json"
)
DEFAULT_PREV_H580_AUDIT_JSON = LOG_DIR / "cube10cm_tap_rl_step_clipped_h580_result_audit.json"
DEFAULT_STEP120_AUDIT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit.json"
DEFAULT_DESIGN_JSON = LOG_DIR / "cube10cm_tap_rl_episode_length_override_candidate_design.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_episode_length_override_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_episode_length_override_result_audit_summary.out"


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


def _runtime_metrics(data: dict[str, Any]) -> dict[str, Any]:
    last_log = data.get("last_log", {})
    controller = data.get("controller_metrics", {})
    return {
        "status": data.get("status"),
        "controller_mode": data.get("controller_mode"),
        "episode_length_s": _float(data, "episode_length_s"),
        "env_max_episode_length": int(data.get("env_max_episode_length", 0)),
        "max_steps": int(data.get("max_steps", 0)),
        "steps_executed": int(data.get("steps_executed", 0)),
        "closed_loop_push_steps": int(data.get("closed_loop_push_steps", 0)),
        "controller_goal_ok_rate": _float(data, "controller_goal_ok_rate"),
        "closed_loop_ik_ok_rate": _float(controller, "closed_loop_ik_ok_rate"),
        "direct_ik_joint_target_apply": bool(data.get("direct_ik_joint_target_apply", False)),
        "isaac_builtin_diffik_controller_apply": bool(data.get("isaac_builtin_diffik_controller_apply", False)),
        "builtin_diffik_step_clipped_target_apply": bool(
            data.get("builtin_diffik_step_clipped_target_apply", False)
        ),
        "builtin_diffik_step_clip_rad": _float(data, "builtin_diffik_step_clip_rad"),
        "builtin_diffik_step_clip_rate_final": _trace(
            data, "controller_trace_stats", "builtin_diffik_step_clip_rate", "final"
        ),
        "builtin_diffik_raw_delta_abs_max_final_rad": _trace(
            data, "controller_trace_stats", "builtin_diffik_raw_delta_abs_max_rad", "final"
        ),
        "builtin_diffik_clipped_delta_abs_max_final_rad": _trace(
            data, "controller_trace_stats", "builtin_diffik_clipped_delta_abs_max_rad", "final"
        ),
        "contact_seen": _float(last_log, "cube_tap_contact_seen_rate"),
        "reaction_contact_context": _float(last_log, "cube_tap_reaction_contact_context_rate"),
        "reaction_seen": _float(last_log, "cube_tap_reaction_seen_rate"),
        "professor_seen": _float(last_log, "cube_tap_professor_physical_reaction_seen_rate"),
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
        "tcp_dist_min_m": _trace(data, "log_trace_stats", "cube_push_tcp_cube_dist_m", "min"),
        "target_inside_max": _trace(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
        ),
        "target_inside_final": _trace(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "final"
        ),
        "target_face_gap_min_m": _trace(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "min"
        ),
        "target_face_gap_final_m": _trace(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
        ),
        "target_fk_err_mm_final": _trace(
            data, "controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "final"
        ),
        "actual_fk_vs_sim_tcp_err_mm_final": _trace(
            data, "controller_trace_stats", "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean", "final"
        ),
        "target_delta_final_rad": _trace(
            data,
            "controller_trace_stats",
            "closed_loop_target_delta_from_actual_abs_max_rad_max",
            "final",
        ),
        "direct_joint_follow_final_rad": _trace(
            data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"
        ),
        "direct_joint_follow_max_rad": _trace(data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"),
        "actual_joint_step_final_rad": _trace(
            data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep608_json", type=Path, default=DEFAULT_EP608_JSON)
    parser.add_argument("--previous_h580_audit_json", type=Path, default=DEFAULT_PREV_H580_AUDIT_JSON)
    parser.add_argument("--step120_audit_json", type=Path, default=DEFAULT_STEP120_AUDIT_JSON)
    parser.add_argument("--design_json", type=Path, default=DEFAULT_DESIGN_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    ep608 = _runtime_metrics(_load(args.ep608_json))
    previous_h580_audit = _load(args.previous_h580_audit_json)
    step120_audit = _load(args.step120_audit_json)
    design = _load(args.design_json)

    previous_h580 = previous_h580_audit["runtime"]
    step120 = step120_audit["runtime"]
    continuous_horizon_valid = (
        abs(ep608["episode_length_s"] - 6.08) < 1.0e-6
        and ep608["env_max_episode_length"] >= 608
        and ep608["max_steps"] == 580
        and ep608["steps_executed"] == 580
        and ep608["closed_loop_push_steps"] == 580
        and ep608["terminated_count"] == 0
        and ep608["truncated_count"] == 0
    )
    episode_cap_blocker_resolved = continuous_horizon_valid and previous_h580["truncated_count"] > 0
    strict_gate_failed = ep608["contact_seen"] == 0.0 and ep608["tap_success"] == 0.0
    target_path_reaches_band = ep608["target_inside_max"] > 0.0
    actual_never_reaches_band = ep608["shortfall_min_m"] > 0.0
    no_better_than_step120 = ep608["shortfall_min_m"] >= step120["shortfall_min_m"]
    no_better_than_previous_h580 = ep608["shortfall_min_m"] >= previous_h580["shortfall_min_m"]

    if continuous_horizon_valid and strict_gate_failed and actual_never_reaches_band:
        primary_blocker = "CONTINUOUS_STEP_CLIPPED_DIFFIK_H580_STILL_OUTSIDE_STRICT_CONTACT_BAND"
    elif not continuous_horizon_valid:
        primary_blocker = "EPISODE_LENGTH_OVERRIDE_NOT_VALID_RECHECK_RUNTIME"
    else:
        primary_blocker = "RECHECK_EPISODE_LENGTH_OVERRIDE_RESULT"

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_episode_length_override_result_audit_v1",
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
            "ep608_json": str(args.ep608_json),
            "previous_h580_audit_json": str(args.previous_h580_audit_json),
            "step120_audit_json": str(args.step120_audit_json),
            "design_json": str(args.design_json),
        },
        "runtime": ep608,
        "comparison": {
            "previous_h580": previous_h580,
            "step120": step120,
            "shortfall_min_delta_vs_step120_m": ep608["shortfall_min_m"] - step120["shortfall_min_m"],
            "shortfall_min_delta_vs_previous_h580_m": ep608["shortfall_min_m"] - previous_h580["shortfall_min_m"],
            "previous_h580_primary_blocker": previous_h580_audit["outcome"]["primary_blocker"],
            "step120_primary_blocker": step120_audit["outcome"]["primary_blocker"],
        },
        "interpretation": {
            "continuous_horizon_valid": continuous_horizon_valid,
            "episode_cap_blocker_resolved": episode_cap_blocker_resolved,
            "strict_contact_gate_failed": strict_gate_failed,
            "target_path_reaches_band_at_least_once": target_path_reaches_band,
            "actual_never_reaches_strict_contact_band": actual_never_reaches_band,
            "no_better_than_step120": no_better_than_step120,
            "no_better_than_previous_h580": no_better_than_previous_h580,
            "professor_evidence_preserved": ep608["professor_physical_reaction_evidence"] == "PASS",
            "do_not_relax_contact_gate_from_this_result": True,
            "why": (
                "The default-off episode-length override made h580 a continuous episode, but strict contact/tap "
                "still stayed at zero and the actual TCP/cube face-gap never entered the strict contact band."
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
            "local_unblock": "target_actual_contact_trajectory_and_reach_contract_audit_design",
            "not_contact_gate_relaxation": True,
            "not_dataset_or_rl": True,
            "not_roarm": True,
        },
        "outputs": {"json": str(args.out_json), "summary": str(args.out_summary)},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_episode_length_override_result_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 runtime_outcome "
        f"status={ep608['status']} episode_length_s={ep608['episode_length_s']:.2f} "
        f"env_max_episode_length={ep608['env_max_episode_length']} max_steps={ep608['max_steps']} "
        f"steps_executed={ep608['steps_executed']} closed_loop_push_steps={ep608['closed_loop_push_steps']} "
        f"contact_seen={ep608['contact_seen']:.1f} tap_success={ep608['tap_success']:.1f} "
        f"professor_evidence={ep608['professor_physical_reaction_evidence']} "
        f"terminated_count={ep608['terminated_count']} truncated_count={ep608['truncated_count']}",
        "line3 horizon_contract "
        f"continuous_horizon_valid={continuous_horizon_valid} "
        f"episode_cap_blocker_resolved={episode_cap_blocker_resolved} "
        f"previous_h580_truncated_count={previous_h580['truncated_count']} "
        f"previous_primary_blocker={previous_h580_audit['outcome']['primary_blocker']}",
        "line4 actual_contact "
        f"face_gap_max_m={ep608['face_gap_max_m']:.9f} "
        f"shortfall_min_m={ep608['shortfall_min_m']:.9f} "
        f"shortfall_final_m={ep608['shortfall_final_m']:.9f} "
        f"step120_shortfall_min_m={step120['shortfall_min_m']:.9f} "
        f"previous_h580_shortfall_min_m={previous_h580['shortfall_min_m']:.9f} "
        f"no_better_than_step120={no_better_than_step120} "
        f"no_better_than_previous_h580={no_better_than_previous_h580}",
        "line5 controller "
        f"target_inside_max={ep608['target_inside_max']:.1f} "
        f"target_inside_final={ep608['target_inside_final']:.1f} "
        f"target_face_gap_min_m={ep608['target_face_gap_min_m']:.9f} "
        f"target_face_gap_final_m={ep608['target_face_gap_final_m']:.9f} "
        f"target_fk_err_mm_final={ep608['target_fk_err_mm_final']} "
        f"actual_fk_vs_sim_tcp_err_mm_final={ep608['actual_fk_vs_sim_tcp_err_mm_final']:.9f}",
        "line6 follow_and_clip "
        f"raw_delta_final_rad={ep608['builtin_diffik_raw_delta_abs_max_final_rad']:.9f} "
        f"clipped_delta_final_rad={ep608['builtin_diffik_clipped_delta_abs_max_final_rad']:.9f} "
        f"clip_rate_final={ep608['builtin_diffik_step_clip_rate_final']:.9f} "
        f"target_delta_final_rad={ep608['target_delta_final_rad']:.9f} "
        f"follow_final_rad={ep608['direct_joint_follow_final_rad']:.9f} "
        f"actual_step_final_rad={ep608['actual_joint_step_final_rad']:.9f}",
        "line7 verdict "
        f"primary_blocker={primary_blocker} contact_gated_positive_control=RUN_FAILED "
        "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line8 next "
        "local_unblock=target_actual_contact_trajectory_and_reach_contract_audit_design "
        "contact_gate_relaxation=NO dataset_rl_roarm=NO",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if primary_blocker.startswith("CONTINUOUS_STEP_CLIPPED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
