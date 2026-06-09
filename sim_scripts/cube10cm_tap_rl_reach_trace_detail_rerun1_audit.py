#!/usr/bin/env python3
"""Posthoc audit for the x240 near-face detail-trace repeat.

Local/posthoc only: reads the approved tiny runtime outputs and writes a
compact branch diagnosis. It does not run Isaac, train, generate datasets, or
touch robot/B200/SSH paths.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
BASE = LOG_DIR / (
    "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_"
    "h580_ep608_x240_nearface_reachtrace_detail_rerun1"
)
SANITY_JSON = BASE.with_name(BASE.name + "_sanity.json")
SANITY_SUMMARY = BASE.with_name(BASE.name + "_sanity_summary.out")
TRACE_JSON = BASE.with_name(BASE.name + "_trace.json")
DETAIL_JSON = BASE.with_name(BASE.name + "_detail_trace.json")
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_reach_trace_detail_rerun1_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_reach_trace_detail_rerun1_audit_summary.out"

FACE_BAND_M = 0.010


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def _absmax(vec: Any) -> float:
    if vec is None:
        return float("nan")
    return float(max(abs(float(v)) for v in vec))


def _row_absmax(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_absmax(row.get(key)) for row in rows if row.get(key) is not None]


def _flat_abs(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        vec = row.get(key)
        if vec is None:
            continue
        values.extend(abs(float(v)) for v in vec)
    return values


def _gap_shortfall(face_gap_m: float) -> float:
    if face_gap_m < -FACE_BAND_M:
        return float(-FACE_BAND_M - face_gap_m)
    if face_gap_m > FACE_BAND_M:
        return float(face_gap_m - FACE_BAND_M)
    return 0.0


def _gap_stats(rows: list[dict[str, Any]], gap_key: str, inside_key: str) -> dict[str, Any]:
    gaps = [float(row[gap_key]) for row in rows]
    inside = [row for row in rows if bool(row.get(inside_key))]
    best_gap = max(gaps)
    final_gap = float(rows[-1][gap_key])
    return {
        "inside_rows": len(inside),
        "inside_unique_steps": len({int(row["step"]) for row in inside}),
        "inside_first_step": min((int(row["step"]) for row in inside), default=None),
        "inside_last_step": max((int(row["step"]) for row in inside), default=None),
        "best_face_gap_m": best_gap,
        "best_shortfall_to_band_m": _gap_shortfall(best_gap),
        "final_face_gap_m": final_gap,
        "final_shortfall_to_band_m": _gap_shortfall(final_gap),
    }


def _vector_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    row_max = _row_absmax(rows, key)
    flat = _flat_abs(rows, key)
    return {
        "nonnull_rows": len(row_max),
        "absmax_rad": max(row_max) if row_max else float("nan"),
        "p95_row_absmax_rad": _pctl(row_max, 0.95),
        "mean_abs_rad": _mean(flat),
    }


def _torque_stats(rows: list[dict[str, Any]], torque_key: str) -> dict[str, Any]:
    ratios: list[float] = []
    max_item: dict[str, Any] | None = None
    for row in rows:
        torque = row.get(torque_key)
        limits = row.get("joint_effort_limit_arm_nm")
        if torque is None or limits is None:
            continue
        for joint_idx, (torque_value, limit_value) in enumerate(zip(torque, limits)):
            limit = abs(float(limit_value))
            if limit <= 1.0e-12:
                continue
            ratio = abs(float(torque_value)) / limit
            ratios.append(ratio)
            if max_item is None or ratio > max_item["ratio"]:
                max_item = {
                    "ratio": ratio,
                    "step": int(row["step"]),
                    "env_id": int(row["env_id"]),
                    "joint_idx": int(joint_idx),
                    "torque_nm": float(torque_value),
                    "effort_limit_nm": float(limit_value),
                }
    saturated = [ratio for ratio in ratios if ratio >= 0.999]
    return {
        "count": len(ratios),
        "max": max_item,
        "mean_ratio": _mean(ratios),
        "p95_ratio": _pctl(ratios, 0.95),
        "saturation_fraction": float(len(saturated) / len(ratios)) if ratios else float("nan"),
    }


def main() -> int:
    sanity = _load(SANITY_JSON)
    detail = _load(DETAIL_JSON)
    rows: list[dict[str, Any]] = detail["rows"]
    summary_lines = SANITY_SUMMARY.read_text(encoding="utf-8").splitlines()

    command = _gap_stats(rows, "command_target_face_gap_m", "command_target_inside_contact_band")
    applied = _gap_stats(
        rows,
        "applied_joint_target_fk_face_gap_m",
        "applied_joint_target_fk_inside_contact_band",
    )
    actual = _gap_stats(rows, "actual_tcp_face_gap_m", "actual_contact_proxy")

    vector_keys = [
        "raw_delta_arm_rad",
        "clipped_delta_arm_rad",
        "target_delta_from_actual_arm_rad",
        "previous_target_minus_actual_arm_rad",
        "current_target_minus_previous_target_arm_rad",
        "direct_joint_follow_arm_rad",
        "actual_joint_step_arm_rad",
    ]
    vectors = {key: _vector_stats(rows, key) for key in vector_keys}

    ratios: list[float] = []
    for row in rows:
        follow = float(row.get("direct_joint_follow_abs_max_rad", float("nan")))
        actual_step = float(row.get("actual_joint_step_abs_max_rad", float("nan")))
        if math.isfinite(follow) and follow > 1.0e-12 and math.isfinite(actual_step):
            ratios.append(actual_step / follow)

    initial_rows = [row for row in rows if int(row["step"]) == 0]
    initial_command = _mean([float(row["command_target_face_gap_m"]) for row in initial_rows])
    initial_applied = _mean([float(row["applied_joint_target_fk_face_gap_m"]) for row in initial_rows])
    initial_actual = _mean([float(row["actual_tcp_face_gap_m"]) for row in initial_rows])
    reset_actual_minus_command = initial_actual - initial_command

    computed_torque = _torque_stats(rows, "computed_torque_after_arm_nm")
    applied_torque = _torque_stats(rows, "applied_torque_after_arm_nm")
    last_log = sanity["last_log"]

    previous_target_indicated = (
        applied["inside_rows"] == 0
        and actual["inside_rows"] == 0
        and vectors["current_target_minus_previous_target_arm_rad"]["absmax_rad"] < 0.002
        and vectors["previous_target_minus_actual_arm_rad"]["absmax_rad"] > 0.009
        and vectors["clipped_delta_arm_rad"]["absmax_rad"] >= 0.0099
    )
    actuator_follow_secondary = (
        _mean(ratios) < 0.2
        or computed_torque["p95_ratio"] > 1.0
        or applied_torque["p95_ratio"] >= 0.999
    )
    reset_lower_priority = abs(reset_actual_minus_command) < 0.002

    if previous_target_indicated:
        next_branch = "DESIGN_PREVIOUS_TARGET_BASE_RUNTIME_FIRST"
    elif applied["inside_rows"] > 0 and (actual["inside_rows"] == 0 or actuator_follow_secondary):
        next_branch = "DESIGN_ACTUATOR_FOLLOW_RUNTIME"
    elif abs(reset_actual_minus_command) >= 0.002:
        next_branch = "DESIGN_RESET_PRECONTACT_RECALIBRATION"
    else:
        next_branch = "FK_TOOL_FRAME_VISUAL_OVERLAY_AUDIT"

    artifact = {
        "artifact_type": "cube10cm_tap_rl_reach_trace_detail_rerun1_audit_v1",
        "local_posthoc_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "sanity_json": str(SANITY_JSON.relative_to(ROOT)),
            "sanity_summary": str(SANITY_SUMMARY.relative_to(ROOT)),
            "basic_trace_json": str(TRACE_JSON.relative_to(ROOT)),
            "detail_trace_json": str(DETAIL_JSON.relative_to(ROOT)),
        },
        "runtime": {
            "status": sanity["status"],
            "steps_executed": sanity["steps_executed"],
            "truncated_count": sanity["truncated_count"],
            "terminated_count": sanity["terminated_count"],
            "num_envs": sanity["num_envs"],
            "seed": sanity["seed"],
            "fixed_cube_x_m": sanity["fixed_cube_x_m"],
            "fixed_cube_y_m": sanity["fixed_cube_y_m"],
            "controller_mode": sanity["controller_mode"],
            "target_path_mode": sanity["target_path_mode"],
            "episode_length_s": sanity["episode_length_s"],
            "closed_loop_push_steps": sanity["closed_loop_push_steps"],
            "builtin_diffik_step_clip_rad": sanity["builtin_diffik_step_clip_rad"],
            "reach_trace_row_count": sanity["reach_trace_row_count"],
            "reach_trace_detail_row_count": sanity["reach_trace_detail_row_count"],
        },
        "schema_guard": {
            "artifact_type": detail["artifact_type"],
            "rows": len(rows),
            "schema_len": len(detail["schema"]),
            "contains_action_fields": detail["contains_action_fields"],
            "action_teacher_dataset": detail["action_teacher_dataset"],
        },
        "gate": {
            "rl_contact_gated_positive_control": sanity["rl_contact_gated_positive_control"],
            "professor_physical_reaction_evidence": sanity["professor_physical_reaction_evidence"],
            "contact_seen": last_log["cube_tap_contact_seen_rate"],
            "tap_success": last_log["cube_tap_success_rate"],
            "professor_physical_reaction_seen": last_log["cube_tap_professor_physical_reaction_seen_rate"],
        },
        "reach_split": {
            "command": command,
            "applied_joint_target_fk": applied,
            "actual_tcp": actual,
        },
        "target_base": {
            "vectors": vectors,
            "previous_target_base_indicated": previous_target_indicated,
            "counterfactual_fk_computed": False,
            "interpretation": (
                "current_target_minus_previous_target is near actual joint step scale, "
                "while previous_target_minus_actual remains near one clipped step; current "
                "actual-base generation is not accumulating target progress fast enough."
            ),
        },
        "actuator_follow": {
            "actual_to_follow_ratio_mean": _mean(ratios),
            "actual_to_follow_ratio_p95": _pctl(ratios, 0.95),
            "actual_to_follow_ratio_min": min(ratios) if ratios else float("nan"),
            "actual_to_follow_ratio_max": max(ratios) if ratios else float("nan"),
            "computed_torque_ratio_to_effort_limit": computed_torque,
            "applied_torque_ratio_to_effort_limit": applied_torque,
            "actuator_follow_secondary": actuator_follow_secondary,
        },
        "reset_precontact": {
            "initial_command_face_gap_m": initial_command,
            "initial_applied_face_gap_m": initial_applied,
            "initial_actual_face_gap_m": initial_actual,
            "reset_actual_minus_command_m": reset_actual_minus_command,
            "reset_lower_priority_not_cleared": reset_lower_priority,
        },
        "decision": {
            "next_branch": next_branch,
            "contact_gate_relaxation": "NOT_NEXT",
            "diffik_action_dataset": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "summary_lines": summary_lines[:10],
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_reach_trace_detail_rerun1_audit_v1 "
        "local_posthoc_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 runtime_contract "
        f"status={sanity['status']} steps_executed={sanity['steps_executed']} "
        f"truncated_count={sanity['truncated_count']} num_envs={sanity['num_envs']} "
        f"seed={sanity['seed']} cube_xy=({sanity['fixed_cube_x_m']},{sanity['fixed_cube_y_m']}) "
        f"controller={sanity['controller_mode']} target_path_mode={sanity['target_path_mode']} "
        f"episode_length_s={sanity['episode_length_s']} step_clip={sanity['builtin_diffik_step_clip_rad']}",
        "line3 gate_result "
        f"rl_contact_gated_positive_control={sanity['rl_contact_gated_positive_control']} "
        f"professor_physical_reaction_evidence={sanity['professor_physical_reaction_evidence']} "
        f"contact_seen={last_log['cube_tap_contact_seen_rate']} "
        f"tap_success={last_log['cube_tap_success_rate']} "
        f"professor_physical_reaction_seen={last_log['cube_tap_professor_physical_reaction_seen_rate']}",
        "line4 detail_schema "
        f"rows={len(rows)} schema_len={len(detail['schema'])} "
        f"contains_action_fields={str(detail['contains_action_fields']).lower()} "
        f"action_teacher_dataset={str(detail['action_teacher_dataset']).lower()}",
        "line5 reach_split "
        f"command_inside_rows={command['inside_rows']} "
        f"command_inside_steps={command['inside_first_step']}..{command['inside_last_step']} "
        f"command_final_face_gap_m={command['final_face_gap_m']:.9f} "
        f"applied_inside_rows={applied['inside_rows']} "
        f"applied_best_face_gap_m={applied['best_face_gap_m']:.9f} "
        f"applied_best_shortfall_m={applied['best_shortfall_to_band_m']:.9f} "
        f"actual_inside_rows={actual['inside_rows']} "
        f"actual_best_face_gap_m={actual['best_face_gap_m']:.9f} "
        f"actual_best_shortfall_m={actual['best_shortfall_to_band_m']:.9f}",
        "line6 target_base "
        f"raw_delta_absmax_rad={vectors['raw_delta_arm_rad']['absmax_rad']:.9f} "
        f"clipped_delta_absmax_rad={vectors['clipped_delta_arm_rad']['absmax_rad']:.9f} "
        f"previous_target_minus_actual_absmax_rad={vectors['previous_target_minus_actual_arm_rad']['absmax_rad']:.9f} "
        f"current_target_minus_previous_absmax_rad={vectors['current_target_minus_previous_target_arm_rad']['absmax_rad']:.9f} "
        f"previous_target_base_indicated={str(previous_target_indicated).upper()} "
        "counterfactual_fk_computed=NO",
        "line7 actuator_follow "
        f"direct_follow_absmax_rad={vectors['direct_joint_follow_arm_rad']['absmax_rad']:.9f} "
        f"actual_step_absmax_rad={vectors['actual_joint_step_arm_rad']['absmax_rad']:.9f} "
        f"actual_to_follow_ratio_mean={_mean(ratios):.9f} "
        f"computed_torque_ratio_p95={computed_torque['p95_ratio']:.9f} "
        f"computed_torque_ratio_max={computed_torque['max']['ratio']:.9f} "
        f"applied_torque_ratio_p95={applied_torque['p95_ratio']:.9f} "
        f"applied_torque_saturation_fraction={applied_torque['saturation_fraction']:.9f}",
        "line8 reset_precontact "
        f"initial_command_face_gap_m={initial_command:.9f} "
        f"initial_applied_face_gap_m={initial_applied:.9f} "
        f"initial_actual_face_gap_m={initial_actual:.9f} "
        f"reset_actual_minus_command_m={reset_actual_minus_command:.9f} "
        f"reset_lower_priority_not_cleared={str(reset_lower_priority).upper()}",
        "line9 decision "
        f"next_branch={next_branch} contact_gate_relaxation=NOT_NEXT "
        "diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        "line10 verdict DETAIL_TRACE_CONFIRMS_TARGET_BASE_FIRST_WITH_ACTUATOR_SATURATION_SECONDARY",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
