#!/usr/bin/env python3
"""Local audit/design for target-vs-actual contact trajectory reach contracts.

This uses existing JSON logs and code only. It does not launch IsaacLab/GPU
runtime, generate datasets, train, control RoArm, SSH, or touch B200.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
PATHS = {
    "tap_harness": ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py",
    "tap_env": ROOT / "roarm_rl/roarm_cube_push_env.py",
    "ep608_json": LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_sanity.json",
    "ep608_summary": (
        LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_sanity_summary.out"
    ),
    "ep608_audit_json": LOG_DIR / "cube10cm_tap_rl_episode_length_override_result_audit.json",
    "ep608_audit_summary": LOG_DIR / "cube10cm_tap_rl_episode_length_override_result_audit_summary.out",
    "step120_json": LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_sanity.json",
    "h580_json": LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_sanity.json",
    "direct_telemetry_json": (
        LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
    ),
    "slow240_json": LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json",
}
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find(path: Path, needle: str) -> dict[str, Any]:
    for idx, text in enumerate(_lines(path), start=1):
        if needle in text:
            return {"line": idx, "text": text.strip()}
    return {"line": None, "text": None}


def _regex_float(path: Path, name: str, default: float) -> float:
    pattern = re.compile(rf"{re.escape(name)}\s*:\s*float\s*=\s*([-+0-9.eE]+)")
    for text in _lines(path):
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stat(data: dict[str, Any], section: str, key: str, field: str, default: float = float("nan")) -> float:
    item = data.get(section, {}).get(key, {})
    if not isinstance(item, dict):
        return default
    return _float(item.get(field), default)


def _has_step_length_list(value: Any, steps: int) -> bool:
    if isinstance(value, list):
        if len(value) == steps:
            return True
        return any(_has_step_length_list(item, steps) for item in value)
    if isinstance(value, dict):
        return any(_has_step_length_list(item, steps) for item in value.values())
    return False


def _run_metrics(name: str, data: dict[str, Any]) -> dict[str, Any]:
    steps = int(data.get("steps_executed", 0))
    return {
        "name": name,
        "status": data.get("status"),
        "controller_mode": data.get("controller_mode"),
        "episode_length_s": data.get("episode_length_s"),
        "steps_executed": steps,
        "truncated_count": int(data.get("truncated_count", 0)),
        "terminated_count": int(data.get("terminated_count", 0)),
        "contact_seen": _float(data.get("last_log", {}).get("cube_tap_contact_seen_rate")),
        "tap_success": _float(data.get("last_log", {}).get("cube_tap_success_rate")),
        "professor_evidence": data.get("professor_physical_reaction_evidence"),
        "command_target_inside_max": _stat(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"
        ),
        "command_target_inside_final": _stat(
            data, "controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "final"
        ),
        "command_target_face_gap_min_m": _stat(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "min"
        ),
        "command_target_face_gap_final_m": _stat(
            data, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
        ),
        "applied_target_fk_err_mm_final": _stat(
            data, "controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "final"
        ),
        "actual_fk_vs_sim_tcp_err_mm_final": _stat(
            data, "controller_trace_stats", "closed_loop_actual_fk_vs_sim_tcp_err_mm_mean", "final"
        ),
        "actual_face_gap_max_m": _stat(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "max"),
        "actual_face_gap_final_m": _stat(data, "log_trace_stats", "cube_tap_contact_face_gap_m", "final"),
        "actual_shortfall_min_m": _stat(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min"),
        "actual_shortfall_final_m": _stat(data, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "final"),
        "actual_lateral_max_m": _stat(data, "log_trace_stats", "cube_tap_contact_lateral_m", "max"),
        "actual_vertical_max_m": _stat(data, "log_trace_stats", "cube_tap_contact_vertical_offset_m", "max"),
        "tcp_cube_dist_min_m": _stat(data, "log_trace_stats", "cube_push_tcp_cube_dist_m", "min"),
        "follow_final_rad": _stat(data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"),
        "follow_max_rad": _stat(data, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "max"),
        "actual_step_final_rad": _stat(data, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"),
        "raw_delta_final_rad": _stat(
            data, "controller_trace_stats", "builtin_diffik_raw_delta_abs_max_rad", "final"
        ),
        "clipped_delta_final_rad": _stat(
            data, "controller_trace_stats", "builtin_diffik_clipped_delta_abs_max_rad", "final"
        ),
        "full_step_timeline_arrays_present": _has_step_length_list(data, steps),
    }


def _fmt(value: float, precision: int = 9) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    runs = {name: _run_metrics(name, _load(path)) for name, path in PATHS.items() if name.endswith("_json")}
    ep608 = runs["ep608_json"]
    step120 = runs["step120_json"]
    h580 = runs["h580_json"]
    direct_telemetry = runs["direct_telemetry_json"]
    slow240 = runs["slow240_json"]
    ep608_audit = _load(PATHS["ep608_audit_json"])

    env_path = PATHS["tap_env"]
    harness_path = PATHS["tap_harness"]
    face_band = _regex_float(env_path, "tap_contact_face_band_m", 0.010)
    lateral_margin = _regex_float(env_path, "tap_contact_lateral_margin_m", 0.015)
    vertical_margin = _regex_float(env_path, "tap_contact_vertical_margin_m", 0.020)
    cube_size = _float(_load(PATHS["ep608_json"]).get("cube_size_m"), 0.100)
    half_along = cube_size * 0.5
    lateral_limit = half_along + lateral_margin
    vertical_limit = cube_size * 0.5 + vertical_margin

    existing_log_has_stats_only = (
        _find(harness_path, 'entry = stats.setdefault(key, {"min": scalar, "max": scalar, "final": scalar})')[
            "line"
        ]
        is not None
    )
    full_timeline_available = bool(ep608["full_step_timeline_arrays_present"])
    command_target_crosses_band = ep608["command_target_inside_max"] > 0.0
    applied_target_fk_gap_available = not math.isnan(ep608["applied_target_fk_err_mm_final"])
    actual_along_outside_band = ep608["actual_shortfall_min_m"] > 0.0
    actual_lateral_within_gate = ep608["actual_lateral_max_m"] <= lateral_limit
    actual_vertical_within_gate = ep608["actual_vertical_max_m"] <= vertical_limit
    stable_across_runs = all(
        run["actual_shortfall_min_m"] > 0.0 and run["contact_seen"] == 0.0 and run["tap_success"] == 0.0
        for run in (ep608, step120, h580, direct_telemetry, slow240)
    )

    if (
        command_target_crosses_band
        and actual_along_outside_band
        and actual_lateral_within_gate
        and actual_vertical_within_gate
        and not applied_target_fk_gap_available
        and not full_timeline_available
    ):
        verdict = "REACH_TRACE_CONTRACT_GAP_IDENTIFIED"
        next_unblock = "patch_default_off_per_step_reach_trace_then_run_one_tiny_repeat_only_after_approval"
    else:
        verdict = "RECHECK_REACH_CONTRACT_AUDIT"
        next_unblock = "manual_review_required"

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_audit_design_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {key: str(path) for key, path in PATHS.items()},
        "contact_gate_contract": {
            "face_band_m": face_band,
            "lateral_limit_m_for_fixed_x_push": lateral_limit,
            "vertical_limit_m": vertical_limit,
            "env_code_evidence": {
                "face_gap_definition": _find(env_path, "face_gap = along + half_along"),
                "contact_proxy_gate": _find(env_path, "face_gap >= -float(self.cfg.tap_contact_face_band_m)"),
                "env_mean_logs": _find(env_path, '"cube_tap_contact_face_gap_m": terms["tap_contact_face_gap_m"].mean().detach()'),
            },
        },
        "code_audit": {
            "existing_log_has_stats_only": existing_log_has_stats_only,
            "full_timeline_available_in_existing_json": full_timeline_available,
            "trace_stats_evidence": _find(
                harness_path,
                'entry = stats.setdefault(key, {"min": scalar, "max": scalar, "final": scalar})',
            ),
            "result_trace_stats_evidence": _find(harness_path, '"controller_trace_stats": controller_trace_stats'),
            "builtin_target_fk_missing_evidence": _find(
                harness_path, '"closed_loop_target_fk_err_mm_mean": float("nan")'
            ),
            "controller_command_target_evidence": _find(
                harness_path, '"closed_loop_target_face_gap_m_mean": float(target_face_gap.mean().item())'
            ),
        },
        "runs": {
            "ep608_continuous_h580": ep608,
            "step120": step120,
            "truncated_h580": h580,
            "external_direct_telemetry": direct_telemetry,
            "external_direct_slow240": slow240,
        },
        "interpretation": {
            "basis_primary_blocker": ep608_audit["outcome"]["primary_blocker"],
            "command_target_crosses_contact_band": command_target_crosses_band,
            "applied_step_clipped_joint_target_fk_gap_available": applied_target_fk_gap_available,
            "actual_along_gap_stays_outside_contact_band": actual_along_outside_band,
            "actual_lateral_within_gate": actual_lateral_within_gate,
            "actual_vertical_within_gate": actual_vertical_within_gate,
            "strict_failure_stable_across_recent_runs": stable_across_runs,
            "full_timeline_missing": not full_timeline_available,
            "critical_read": (
                "Existing logs prove the command target crosses the contact band while the actual averaged TCP "
                "stays outside along the face gap. They do not prove when the command target, step-clipped joint "
                "target FK, and actual TCP diverge because only min/max/final stats are persisted."
            ),
        },
        "required_default_off_trace_schema": [
            "step",
            "env_id",
            "episode_length_s",
            "cube_pos_w_xyz",
            "push_dir_xy",
            "command_target_face_gap_m",
            "command_target_lateral_m",
            "command_target_vertical_offset_m",
            "command_target_inside_contact_band",
            "applied_joint_target_fk_face_gap_m",
            "applied_joint_target_fk_lateral_m",
            "applied_joint_target_fk_vertical_offset_m",
            "applied_joint_target_fk_inside_contact_band",
            "actual_tcp_face_gap_m",
            "actual_tcp_lateral_m",
            "actual_tcp_vertical_offset_m",
            "actual_contact_proxy",
            "joint_target_delta_abs_max_rad",
            "direct_joint_follow_abs_max_rad",
            "actual_joint_step_abs_max_rad",
            "cube_disp_along_m",
            "cube_speed_mps",
            "professor_physical_reaction_now",
            "tap_success_now",
            "terminated",
            "truncated",
        ],
        "outcome": {
            "verdict": verdict,
            "contact_gated_positive_control": "RUN_FAILED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
            "next_local_unblock": next_unblock,
        },
        "outputs": {"json": str(OUT_JSON), "summary": str(OUT_SUMMARY)},
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit_v1 "
        "local_audit_design_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 current_basis "
        f"ep608_primary_blocker={ep608_audit['outcome']['primary_blocker']} "
        f"continuous_horizon_valid={ep608_audit['interpretation']['continuous_horizon_valid']} "
        f"contact_seen={ep608['contact_seen']:.1f} tap_success={ep608['tap_success']:.1f} "
        f"professor_evidence={ep608['professor_evidence']}",
        "line3 contact_contract "
        f"face_band_m={face_band:.3f} lateral_limit_m={lateral_limit:.3f} vertical_limit_m={vertical_limit:.3f} "
        f"actual_lateral_max_m={ep608['actual_lateral_max_m']:.9f} "
        f"actual_vertical_max_m={ep608['actual_vertical_max_m']:.9f} "
        f"lateral_within_gate={actual_lateral_within_gate} vertical_within_gate={actual_vertical_within_gate}",
        "line4 target_vs_actual_ep608 "
        f"command_target_inside_max={ep608['command_target_inside_max']:.1f} "
        f"command_target_face_gap_min_m={ep608['command_target_face_gap_min_m']:.9f} "
        f"command_target_face_gap_final_m={ep608['command_target_face_gap_final_m']:.9f} "
        f"actual_face_gap_max_m={ep608['actual_face_gap_max_m']:.9f} "
        f"actual_shortfall_min_m={ep608['actual_shortfall_min_m']:.9f} "
        f"actual_shortfall_final_m={ep608['actual_shortfall_final_m']:.9f}",
        "line5 cross_run_stability "
        f"step120_shortfall_min_m={step120['actual_shortfall_min_m']:.9f} "
        f"h580_shortfall_min_m={h580['actual_shortfall_min_m']:.9f} "
        f"direct_telemetry_shortfall_min_m={direct_telemetry['actual_shortfall_min_m']:.9f} "
        f"slow240_shortfall_min_m={slow240['actual_shortfall_min_m']:.9f} "
        f"strict_failure_stable_across_recent_runs={stable_across_runs}",
        "line6 trace_contract_gap "
        f"existing_log_has_stats_only={existing_log_has_stats_only} "
        f"full_step_timeline_available={full_timeline_available} "
        f"applied_step_clipped_joint_target_fk_gap_available={applied_target_fk_gap_available} "
        "cannot_localize_time_alignment_from_existing_json=True",
        "line7 required_next_schema "
        "default_off_per_step_reach_trace_fields=step_env_command_target_gap_applied_target_fk_gap_actual_tcp_gap_"
        "joint_follow_cube_reaction_done_flags action_fields=NO_DATASET_TEACHER_FIELDS",
        "line8 verdict "
        f"{verdict} contact_gated_positive_control=RUN_FAILED diffik_action_dataset=BLOCKED "
        "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line9 next "
        f"local_unblock={next_unblock} contact_gate_relaxation=NO dataset_rl_roarm=NO",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if verdict == "REACH_TRACE_CONTRACT_GAP_IDENTIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
