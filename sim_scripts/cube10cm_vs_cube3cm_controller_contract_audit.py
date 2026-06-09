#!/usr/bin/env python3
"""Local posthoc audit comparing 3cm DiffIK and 10cm tap controller contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

THREE_SUMMARY = LOG_DIR / "diffik_probe_v3_eval10240_seed779_summary.json"
THREE_STDOUT = LOG_DIR / "diffik_probe_v3_eval10240_seed779_stdout.out"
THREE_AUDIT = LOG_DIR / "diffik_probe_v3_eval10240_seed779_audit.out"
THREE_FILTER = LOG_DIR / "controlled_push_filter_audit.out"
THREE_CSV = LOG_DIR / "diffik_probe_v3_eval10240_seed779.csv"

TEN_BASELINE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity.json"
TEN_BASELINE_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity_summary.out"
TEN_SLOW240_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity.json"
TEN_SLOW240_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity_summary.out"
TEN_SLOW240_AUDIT = LOG_DIR / "cube10cm_tap_rl_slow240_result_audit.json"

OUT_JSON = LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit_summary.out"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def line_at(path: Path, one_based: int) -> str:
    lines = read_lines(path)
    return lines[one_based - 1] if 0 < one_based <= len(lines) else ""


def nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def trace_final(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return nested(data, ["log_trace_stats", key, "final"], default)


def trace_max(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return nested(data, ["log_trace_stats", key, "max"], default)


def ratio(numer: float, denom: float) -> float | None:
    if abs(float(denom)) <= 1.0e-12:
        return None
    return float(numer) / float(denom)


def main() -> int:
    three = read_json(THREE_SUMMARY)
    ten_base = read_json(TEN_BASELINE_JSON)
    ten_slow = read_json(TEN_SLOW240_JSON)
    slow_audit = read_json(TEN_SLOW240_AUDIT)

    ten_step_dt_s = 1.0 / 200.0 * 2.0
    ten_episode_length_s = 1.2
    three_step_dt_s = ten_step_dt_s

    three_size_m = 0.030
    three_mass_kg = 0.020
    ten_size_m = float(ten_base["cube_size_m"])
    ten_mass_kg = float(ten_base["cube_mass_kg"])
    three_steps = int(three["total_steps_per_trial"])
    ten_steps = int(ten_base["max_steps"])
    three_episode_s = float(three["episode_length_s"])

    three_csv_header = line_at(THREE_CSV, 1)
    three_existing_eval_has_follow_columns = "joint_follow_err_" in three_csv_header

    answer = {
        "ten_cm_steps_short_because": (
            "the 10cm positive-control harness defaults to --steps=120 and the 10cm env episode_length_s is 1.2 "
            "with dt=0.01s; this is not caused by random object placement"
        ),
        "ten_cm_target_random": False,
        "ten_cm_cube_position_policy": "FIXED_BY_HARNESS",
        "ten_cm_fixed_cube_x_m": float(ten_base["fixed_cube_x_m"]),
        "ten_cm_fixed_cube_y_m": float(ten_base["fixed_cube_y_m"]),
        "ten_cm_fixed_push_dir": [float(ten_base["fixed_push_dir_x"]), float(ten_base["fixed_push_dir_y"])],
        "ten_cm_env_step_dt_s": ten_step_dt_s,
        "ten_cm_episode_length_s": ten_episode_length_s,
        "ten_cm_runtime_max_steps": ten_steps,
        "ten_cm_runtime_seconds_at_dt": ten_steps * ten_step_dt_s,
        "three_cm_probe_extends_episode_for_trajectory": True,
        "three_cm_total_steps_per_trial": three_steps,
        "three_cm_episode_length_s": three_episode_s,
        "three_cm_default_position_policy": "RANDOM_UNLESS_FIXED_ARGS_ARE_PASSED",
        "three_cm_eval10240_fixed_args": "fixed_cube_x_m=None fixed_cube_y_m=None fixed_push_dir=None",
    }

    comparison_table = [
        {
            "field": "size_m",
            "cube3cm_diffik_eval10240": three_size_m,
            "cube10cm_direct_ik_baseline": ten_size_m,
            "cube10cm_direct_ik_slow240": ten_size_m,
        },
        {
            "field": "mass_kg",
            "cube3cm_diffik_eval10240": three_mass_kg,
            "cube10cm_direct_ik_baseline": ten_mass_kg,
            "cube10cm_direct_ik_slow240": ten_mass_kg,
        },
        {
            "field": "runtime_horizon_steps",
            "cube3cm_diffik_eval10240": three_steps,
            "cube10cm_direct_ik_baseline": int(ten_base["max_steps"]),
            "cube10cm_direct_ik_slow240": int(ten_slow["max_steps"]),
        },
        {
            "field": "runtime_horizon_s",
            "cube3cm_diffik_eval10240": three_episode_s,
            "cube10cm_direct_ik_baseline": float(ten_base["max_steps"]) * ten_step_dt_s,
            "cube10cm_direct_ik_slow240": float(ten_slow["max_steps"]) * ten_step_dt_s,
        },
        {
            "field": "position_policy",
            "cube3cm_diffik_eval10240": "random sampled; fixed args were None",
            "cube10cm_direct_ik_baseline": "fixed cube=(0.25,0.0), push=(1.0,0.0)",
            "cube10cm_direct_ik_slow240": "fixed cube=(0.25,0.0), push=(1.0,0.0)",
        },
        {
            "field": "controller",
            "cube3cm_diffik_eval10240": "IsaacLab_DifferentialIKController position command",
            "cube10cm_direct_ik_baseline": str(ten_base["controller_mode"]),
            "cube10cm_direct_ik_slow240": str(ten_slow["controller_mode"]),
        },
        {
            "field": "step_cap_or_follow",
            "cube3cm_diffik_eval10240": (
                "DiffIK clipped delta; max_diffik_joint_step_rad=0.035, "
                "v3_posx_max=0.020, clip_rate_mean=0.652074015"
            ),
            "cube10cm_direct_ik_baseline": (
                "direct apply; wrapper action cap inactive, direct_follow_final_rad="
                f"{nested(ten_base, ['controller_trace_stats', 'direct_joint_follow_abs_max_rad', 'final'])}"
            ),
            "cube10cm_direct_ik_slow240": (
                "direct apply; wrapper action cap inactive, direct_follow_final_rad="
                f"{nested(ten_slow, ['controller_trace_stats', 'direct_joint_follow_abs_max_rad', 'final'])}"
            ),
        },
        {
            "field": "speed_guard_or_speed_evidence",
            "cube3cm_diffik_eval10240": "speed thresholds used by controlled filter; speed_p95=1.302103193, speed_p99=1.733444051",
            "cube10cm_direct_ik_baseline": "env speed/contact slowdown exists, but direct-apply zero action bypasses wrapper path",
            "cube10cm_direct_ik_slow240": "same direct-apply bypass; max_speed_mps=0.024404075",
        },
        {
            "field": "success_metric",
            "cube3cm_diffik_eval10240": "controlled_push_rate=0.9431640625, success_marker_rate=0.59482421875",
            "cube10cm_direct_ik_baseline": "strict contact/reaction tap gate; tap_success=0.0",
            "cube10cm_direct_ik_slow240": "strict contact/reaction tap gate; tap_success=0.0",
        },
    ]

    ratios = {
        "size_ratio_10cm_over_3cm": ratio(ten_size_m, three_size_m),
        "mass_ratio_10cm_over_3cm": ratio(ten_mass_kg, three_mass_kg),
        "horizon_steps_ratio_10cm_baseline_over_3cm": ratio(ten_steps, three_steps),
        "horizon_seconds_ratio_10cm_baseline_over_3cm": ratio(ten_steps * ten_step_dt_s, three_episode_s),
        "horizon_steps_ratio_3cm_over_10cm_baseline": ratio(three_steps, ten_steps),
        "horizon_seconds_ratio_3cm_over_10cm_baseline": ratio(three_episode_s, ten_steps * ten_step_dt_s),
    }

    interpretation = {
        "ten_cm_short_horizon_is_harness_contract": True,
        "ten_cm_short_horizon_is_not_random_target_position": True,
        "ten_cm_fixed_target_and_push_dir_verified": True,
        "three_cm_also_had_speed_and_clip_issues": True,
        "three_cm_eval10240_existing_csv_has_direct_follow_columns": three_existing_eval_has_follow_columns,
        "three_cm_trace_code_can_record_follow_error_when_enabled": True,
        "primary_contract_difference": (
            "3cm DiffIK eval used a long, trajectory-sized, clipped controlled-push contract; "
            "10cm tap uses a short strict contact-gated positive-control contract with direct-apply follow telemetry"
        ),
        "slow240_result": (
            "slowing the target schedule reduced direct follow error but still failed contact/tap, so horizon/follow timing is implicated "
            "but not yet sufficient"
        ),
        "dataset_rl_roarm_status": "BLOCKED_UNTIL_CONTACT_GATED_POSITIVE_CONTROL_OR_EXPLICIT_NOISY_TIER_EXCEPTION_GATE",
    }

    artifact = {
        "artifact_type": "cube10cm_vs_cube3cm_controller_contract_audit_v1",
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
            "three_cm_summary": str(THREE_SUMMARY),
            "three_cm_stdout": str(THREE_STDOUT),
            "three_cm_audit": str(THREE_AUDIT),
            "three_cm_controlled_filter": str(THREE_FILTER),
            "three_cm_csv": str(THREE_CSV),
            "ten_cm_baseline_json": str(TEN_BASELINE_JSON),
            "ten_cm_baseline_summary": str(TEN_BASELINE_SUMMARY),
            "ten_cm_slow240_json": str(TEN_SLOW240_JSON),
            "ten_cm_slow240_summary": str(TEN_SLOW240_SUMMARY),
            "ten_cm_slow240_audit": str(TEN_SLOW240_AUDIT),
        },
        "source_line_refs": {
            "claude_current_state_protocol": ["CLAUDE.md:5", "CLAUDE.md:55"],
            "ten_cm_harness_defaults": ["roarm_rl/test_positive_control_cube_tap10cm.py:299-316"],
            "ten_cm_fixed_position_assignment": ["roarm_rl/test_positive_control_cube_tap10cm.py:377-382"],
            "ten_cm_runtime_loop_and_direct_apply": ["roarm_rl/test_positive_control_cube_tap10cm.py:449-461"],
            "ten_cm_output_contract": ["roarm_rl/test_positive_control_cube_tap10cm.py:594-638"],
            "ten_cm_env_horizon_and_tap_contract": [
                "roarm_rl/roarm_cube_push_env.py:63",
                "roarm_rl/roarm_cube_push_env.py:254-267",
                "roarm_rl/roarm_cube_push_env.py:1244-1252",
            ],
            "env_dt": ["roarm_rl/roarm_stack_env.py:108-116", "roarm_rl/roarm_stack_env.py:401"],
            "random_default_and_sampling": ["roarm_rl/roarm_cube_push_env.py:97-105", "roarm_rl/roarm_cube_push_env.py:772-784"],
            "three_cm_probe_defaults_and_episode_extension": [
                "sim_scripts/cube3cm_push_diffik_probe.py:40-47",
                "sim_scripts/cube3cm_push_diffik_probe.py:307-322",
            ],
            "three_cm_diffik_and_clip_loop": [
                "sim_scripts/cube3cm_push_diffik_probe.py:356-362",
                "sim_scripts/cube3cm_push_diffik_probe.py:587-624",
                "sim_scripts/cube3cm_push_diffik_probe.py:734-784",
            ],
            "three_cm_trace_follow_optional": [
                "sim_scripts/cube3cm_push_diffik_probe.py:134-139",
                "sim_scripts/cube3cm_push_diffik_probe.py:1015-1045",
            ],
            "three_cm_logs": [
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_stdout.out:20-21",
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/diffik_probe_v3_eval10240_seed779_audit.out:1-6",
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/controlled_push_filter_audit.out:1-13",
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/controlled_push_filter_audit.out:40-44",
            ],
            "ten_cm_logs": [
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity_summary.out:1-10",
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_slow240_sanity_summary.out:1-10",
                "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_slow240_result_audit_summary.out:1-8",
            ],
        },
        "answer": answer,
        "comparison_table": comparison_table,
        "ratios": ratios,
        "three_cm": {
            "controller": str(three["controller"]),
            "command_type": str(three["command_type"]),
            "env_joint_delta_action_loop_bypassed": bool(three["env_joint_delta_action_loop_bypassed"]),
            "episode_length_s": three_episode_s,
            "total_steps_per_trial": three_steps,
            "base_total_steps_per_trial": int(three["base_total_steps_per_trial"]),
            "v3_posx_total_steps_per_trial": int(three["v3_posx_total_steps_per_trial"]),
            "max_diffik_joint_step_rad": float(three["max_diffik_joint_step_rad"]),
            "diffik_clip_rate_mean": float(three["diffik_clip_rate_mean"]),
            "max_cube_speed_mean_mps": float(three["max_cube_speed_mean_mps"]),
            "controlled_push_rate": float(three["controlled_push_rate"]),
            "impact_outlier_rate": float(three["impact_outlier_rate"]),
            "low_motion_rate": float(three["low_motion_rate"]),
            "success_marker_rate": float(three["success_marker_rate"]),
            "final_tcp_target_err_mean_m": float(three["final_tcp_target_err_mean_m"]),
            "stdout_line20": line_at(THREE_STDOUT, 20),
            "stdout_line21": line_at(THREE_STDOUT, 21),
            "audit_lines_1_6": [line_at(THREE_AUDIT, idx) for idx in range(1, 7)],
            "controlled_filter_lines_1_13": [line_at(THREE_FILTER, idx) for idx in range(1, 14)],
            "controlled_filter_top_outlier_lines_40_44": [line_at(THREE_FILTER, idx) for idx in range(40, 45)],
            "per_env_csv_has_follow_columns": three_existing_eval_has_follow_columns,
        },
        "ten_cm_baseline": {
            "controller_mode": str(ten_base["controller_mode"]),
            "cube_size_m": ten_size_m,
            "cube_mass_kg": ten_mass_kg,
            "fixed_cube_x_m": float(ten_base["fixed_cube_x_m"]),
            "fixed_cube_y_m": float(ten_base["fixed_cube_y_m"]),
            "fixed_push_dir_x": float(ten_base["fixed_push_dir_x"]),
            "fixed_push_dir_y": float(ten_base["fixed_push_dir_y"]),
            "max_steps": int(ten_base["max_steps"]),
            "steps_executed": int(ten_base["steps_executed"]),
            "closed_loop_push_steps": int(ten_base["closed_loop_push_steps"]),
            "closed_loop_alpha_final": nested(ten_base, ["controller_trace_stats", "closed_loop_alpha", "final"]),
            "direct_joint_follow_abs_max_final_rad": nested(
                ten_base, ["controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"]
            ),
            "direct_actual_joint_step_abs_max_final_rad": nested(
                ten_base, ["controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"]
            ),
            "target_face_gap_final_m": nested(
                ten_base, ["controller_trace_stats", "closed_loop_target_face_gap_m_max", "final"]
            ),
            "target_inside_contact_band_rate_max": nested(
                ten_base, ["controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"]
            ),
            "target_fk_err_max_mm": nested(ten_base, ["controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "max"]),
            "action_abs_max_trace": trace_max(ten_base, "cube_push_action_abs_max"),
            "joint_delta_cap_rate_trace": trace_max(ten_base, "cube_push_joint_delta_cap_rate"),
            "tap_success": float(trace_final(ten_base, "cube_tap_success_rate")),
            "contact_seen": float(trace_final(ten_base, "cube_tap_contact_seen_rate")),
            "professor_physical_reaction_seen": float(
                trace_final(ten_base, "cube_tap_professor_physical_reaction_seen_rate")
            ),
            "summary_lines_1_10": [line_at(TEN_BASELINE_SUMMARY, idx) for idx in range(1, 11)],
        },
        "ten_cm_slow240": {
            "controller_mode": str(ten_slow["controller_mode"]),
            "max_steps": int(ten_slow["max_steps"]),
            "steps_executed": int(ten_slow["steps_executed"]),
            "closed_loop_push_steps": int(ten_slow["closed_loop_push_steps"]),
            "closed_loop_alpha_final": nested(ten_slow, ["controller_trace_stats", "closed_loop_alpha", "final"]),
            "direct_joint_follow_abs_max_final_rad": nested(
                ten_slow, ["controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"]
            ),
            "direct_actual_joint_step_abs_max_final_rad": nested(
                ten_slow, ["controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"]
            ),
            "target_face_gap_final_m": nested(
                ten_slow, ["controller_trace_stats", "closed_loop_target_face_gap_m_max", "final"]
            ),
            "target_inside_contact_band_rate_max": nested(
                ten_slow, ["controller_trace_stats", "closed_loop_target_inside_contact_band_rate", "max"]
            ),
            "tap_success": float(trace_final(ten_slow, "cube_tap_success_rate")),
            "contact_seen": float(trace_final(ten_slow, "cube_tap_contact_seen_rate")),
            "professor_physical_reaction_seen": float(
                trace_final(ten_slow, "cube_tap_professor_physical_reaction_seen_rate")
            ),
            "summary_lines_1_10": [line_at(TEN_SLOW240_SUMMARY, idx) for idx in range(1, 11)],
        },
        "slow240_posthoc": {
            "follow_final_ratio_slow240_over_baseline": nested(
                slow_audit, ["comparison", "follow_final_ratio_slow240_over_baseline"]
            ),
            "shortfall_min_improvement_m": nested(slow_audit, ["comparison", "shortfall_min_improvement_m"]),
            "verdict": nested(slow_audit, ["outcome", "verdict"]),
            "target_timing_plans": slow_audit.get("target_timing_plans", {}),
        },
        "interpretation": interpretation,
        "outcome": {
            "professor_physical_reaction_evidence": "PASS",
            "rl_contact_gated_positive_control": "FAIL",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "outputs": {"json": str(OUT_JSON), "summary": str(OUT_SUMMARY)},
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_vs_cube3cm_controller_contract_audit_v1 "
        "local_posthoc_audit_only=YES gpu_runtime_launched_by_this_audit=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 answer why_10cm_steps_short=HARNESS_120_STEPS_AND_ENV_1P2S_AT_DT_0P01 "
        f"ten_cm_max_steps={ten_steps} ten_cm_runtime_s={ten_steps * ten_step_dt_s:.3f} "
        f"ten_cm_episode_length_s={ten_episode_length_s:.3f} not_due_to_random_position=YES",
        "line3 target_policy ten_cm_target_random=NO "
        f"ten_cm_fixed_cube_xy=({float(ten_base['fixed_cube_x_m']):.3f},{float(ten_base['fixed_cube_y_m']):.3f}) "
        f"ten_cm_fixed_push_dir=({float(ten_base['fixed_push_dir_x']):.1f},{float(ten_base['fixed_push_dir_y']):.1f}) "
        "three_cm_default=random_unless_fixed_args three_cm_eval10240_fixed_args=None",
        "line4 horizon_mass_size "
        f"three_cm_steps={three_steps} three_cm_episode_s={three_episode_s:.3f} "
        f"ten_cm_steps={ten_steps} ten_cm_runtime_s={ten_steps * ten_step_dt_s:.3f} "
        f"size_ratio_10cm_over_3cm={ratios['size_ratio_10cm_over_3cm']:.6f} "
        f"mass_ratio_10cm_over_3cm={ratios['mass_ratio_10cm_over_3cm']:.6f} "
        f"horizon_steps_ratio_3cm_over_10cm={ratios['horizon_steps_ratio_3cm_over_10cm_baseline']:.6f} "
        f"horizon_seconds_ratio_3cm_over_10cm={ratios['horizon_seconds_ratio_3cm_over_10cm_baseline']:.6f}",
        "line5 three_cm_contract "
        f"controller={three['controller']} command_type={three['command_type']} "
        f"env_action_loop_bypassed={three['env_joint_delta_action_loop_bypassed']} "
        f"base_steps={three['base_total_steps_per_trial']} v3_posx_steps={three['v3_posx_total_steps_per_trial']} "
        f"max_diffik_joint_step_rad={three['max_diffik_joint_step_rad']} "
        f"diffik_clip_rate_mean={three['diffik_clip_rate_mean']:.9f} "
        f"existing_eval_follow_columns={three_existing_eval_has_follow_columns}",
        "line6 three_cm_outcome_and_speed "
        f"controlled_push_rate={three['controlled_push_rate']:.9f} success_marker_rate={three['success_marker_rate']:.9f} "
        f"low_motion_rate={three['low_motion_rate']:.9f} impact_outlier_rate={three['impact_outlier_rate']:.9f} "
        f"max_cube_speed_mean_mps={three['max_cube_speed_mean_mps']:.9f} "
        "speed_p95_mps=1.302103193 speed_p99_mps=1.733444051 top_impact_speed_mps_max=4.549609073",
        "line7 ten_cm_contract "
        f"baseline_closed_loop_push_steps={ten_base['closed_loop_push_steps']} slow240_closed_loop_push_steps={ten_slow['closed_loop_push_steps']} "
        f"max_steps_both={ten_steps} controller_mode={ten_base['controller_mode']} "
        f"action_wrapper_cap_inactive_by_zero_action=YES baseline_action_abs_max={trace_max(ten_base, 'cube_push_action_abs_max')} "
        f"slow240_action_abs_max={trace_max(ten_slow, 'cube_push_action_abs_max')}",
        "line8 ten_cm_follow_and_contact "
        f"baseline_follow_final_rad={nested(ten_base, ['controller_trace_stats', 'direct_joint_follow_abs_max_rad', 'final']):.9f} "
        f"slow240_follow_final_rad={nested(ten_slow, ['controller_trace_stats', 'direct_joint_follow_abs_max_rad', 'final']):.9f} "
        f"baseline_tap_success={float(trace_final(ten_base, 'cube_tap_success_rate')):.1f} "
        f"slow240_tap_success={float(trace_final(ten_slow, 'cube_tap_success_rate')):.1f} "
        f"baseline_contact_seen={float(trace_final(ten_base, 'cube_tap_contact_seen_rate')):.1f} "
        f"slow240_contact_seen={float(trace_final(ten_slow, 'cube_tap_contact_seen_rate')):.1f}",
        "line9 verdict ten_cm_short_horizon_is_harness_contract=YES "
        "object_position_primary_cause=NO three_cm_had_speed_clip_guard_issues=YES "
        "contract_difference=LONG_CONTROLLED_PUSH_FILTERED_3CM_VS_SHORT_STRICT_CONTACT_GATED_10CM",
        "line10 blocked_status rl_contact_gated_positive_control=FAIL diffik_action_dataset=BLOCKED "
        "tiny_action_dataset_dry_run=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
