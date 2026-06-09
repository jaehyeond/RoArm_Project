"""Compare the first two cube10cm tap RL pass candidates.

This is local posthoc analysis only. It reads the tiny positive-control
artifacts and preserves the current blocker before designing the next pass
candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_prevtarget_pass_candidates_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_prevtarget_pass_candidates_audit_summary.out"

CANDIDATES = {
    "candidate1_prevtargetbase": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_candidate1_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_candidate1_detail_trace.json",
    },
    "candidate2_prevtargetbase_driveboost": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_driveboost_candidate2_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_driveboost_candidate2_detail_trace.json",
    },
    "candidate3_prevtargetbase_link5corner": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5corner_candidate3_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5corner_candidate3_detail_trace.json",
    },
    "candidate4_prevtargetbase_lead120": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_lead120_candidate4_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_lead120_candidate4_detail_trace.json",
    },
    "candidate5_link5aabb_pre020_degenerate": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_candidate5_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_candidate5_detail_trace.json",
    },
    "candidate6_link5aabb_pre040_nondegenerate": {
        "sanity": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_pre040_candidate6_sanity.json",
        "detail": LOG_DIR
        / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_pre040_candidate6_detail_trace.json",
    },
}

ACTION_FIELD_NAMES = {
    "action",
    "actions",
    "teacher_action",
    "policy_action",
    "action_teacher",
    "action_delta",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_abs_vector(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        vector = row.get(key)
        if isinstance(vector, list):
            values.extend(abs(_num(v)) for v in vector)
    return max(values) if values else 0.0


def _safe_stat(result: dict[str, Any], group: str, key: str, stat: str, default: float = 0.0) -> float:
    return _num(result.get(group, {}).get(key, {}).get(stat), default)


def _step_span(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    steps = sorted({int(row["step"]) for row in rows if bool(row.get(key))})
    return {
        "row_count": sum(1 for row in rows if bool(row.get(key))),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "step_count": len(steps),
    }


def _termination_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if bool(row.get("terminated")) or bool(row.get("truncated")):
            out.append(
                {
                    "step": int(row["step"]),
                    "env_id": int(row["env_id"]),
                    "terminated": bool(row.get("terminated")),
                    "truncated": bool(row.get("truncated")),
                    "cube_disp_along_m": _num(row.get("cube_disp_along_m")),
                    "cube_speed_mps": _num(row.get("cube_speed_mps")),
                    "actual_tcp_face_gap_m": _num(row.get("actual_tcp_face_gap_m")),
                    "actual_contact_proxy": bool(row.get("actual_contact_proxy")),
                    "applied_fk_inside_contact_band": bool(row.get("applied_joint_target_fk_inside_contact_band")),
                }
            )
    return out


def _first_step(rows: list[dict[str, Any]], key: str) -> int | None:
    steps = [int(row["step"]) for row in rows if bool(row.get(key))]
    return min(steps) if steps else None


def _summarize_candidate(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    result = _load(paths["sanity"])
    detail = _load(paths["detail"])
    rows = detail.get("rows", [])
    schema = detail.get("schema", [])
    action_fields = sorted(field for field in schema if field in ACTION_FIELD_NAMES or "action" in field)

    applied_face = [_num(row.get("applied_joint_target_fk_face_gap_m")) for row in rows]
    actual_face = [_num(row.get("actual_tcp_face_gap_m")) for row in rows]
    cube_disp = [_num(row.get("cube_disp_along_m")) for row in rows]
    cube_speed = [_num(row.get("cube_speed_mps")) for row in rows]
    computed_torque = _max_abs_vector(rows, "computed_torque_after_arm_nm")
    applied_torque = _max_abs_vector(rows, "applied_torque_after_arm_nm")
    effort_limit = _max_abs_vector(rows, "joint_effort_limit_arm_nm")
    torque_saturation_rows = 0
    torque_row_total = 0
    for row in rows:
        torque = row.get("applied_torque_after_arm_nm")
        limit = row.get("joint_effort_limit_arm_nm")
        if isinstance(torque, list) and isinstance(limit, list):
            for t, lim in zip(torque, limit):
                torque_row_total += 1
                if abs(_num(t)) >= abs(_num(lim)) - 1.0e-6:
                    torque_saturation_rows += 1

    return {
        "name": name,
        "sanity": str(paths["sanity"]),
        "detail": str(paths["detail"]),
        "status": result.get("status"),
        "rl_contact_gated_positive_control": result.get("rl_contact_gated_positive_control"),
        "professor_physical_reaction_evidence": result.get("professor_physical_reaction_evidence"),
        "controller_mode": result.get("controller_mode"),
        "target_base_mode": result.get("builtin_diffik_target_base_mode"),
        "tool_contact_proxy_mode": result.get("tool_contact_proxy_mode", "hand_tcp"),
        "tool_proxy_label": result.get("tool_proxy_label", "hand_tcp"),
        "arm_stiffness": result.get("arm_stiffness"),
        "arm_damping": result.get("arm_damping"),
        "arm_effort_limit": result.get("arm_effort_limit"),
        "arm_velocity_limit": result.get("arm_velocity_limit"),
        "terminated_count": int(result.get("terminated_count", 0)),
        "truncated_count": int(result.get("truncated_count", 0)),
        "reset": {
            "initial_contact_proxy_rate": _num(result.get("reset_metrics", {}).get("initial_contact_proxy_rate")),
            "initial_face_gap_m": _num(result.get("reset_metrics", {}).get("initial_face_gap_m")),
            "initial_vertical_offset_m": _num(result.get("reset_metrics", {}).get("initial_vertical_offset_m")),
        },
        "last_log": {
            "contact_seen": _num(result.get("last_log", {}).get("cube_tap_contact_seen_rate")),
            "reaction_contact_context": _num(result.get("last_log", {}).get("cube_tap_reaction_contact_context_rate")),
            "reaction_seen": _num(result.get("last_log", {}).get("cube_tap_reaction_seen_rate")),
            "professor_physical_reaction_seen": _num(
                result.get("last_log", {}).get("cube_tap_professor_physical_reaction_seen_rate")
            ),
            "overshoot_seen": _num(result.get("last_log", {}).get("cube_tap_overshoot_seen_rate")),
            "tap_success": _num(result.get("last_log", {}).get("cube_tap_success_rate")),
            "max_disp_along_m": _num(result.get("last_log", {}).get("cube_tap_max_disp_along_m")),
            "max_z_delta_m": _num(result.get("last_log", {}).get("cube_tap_max_z_delta_m")),
            "max_speed_mps": _num(result.get("last_log", {}).get("cube_tap_max_speed_mps")),
        },
        "trace_contract": {
            "detail_rows": len(rows),
            "schema_len": len(schema),
            "contains_action_fields": bool(action_fields),
            "action_fields": action_fields,
            "action_teacher_dataset": False,
            "target_base_modes": sorted({str(row.get("target_base_mode")) for row in rows}),
        },
        "inside_rows": {
            "command": _step_span(rows, "command_target_inside_contact_band"),
            "applied_fk": _step_span(rows, "applied_joint_target_fk_inside_contact_band"),
            "actual_proxy": _step_span(rows, "actual_contact_proxy"),
        },
        "face_gap_m": {
            "applied_fk_min": min(applied_face) if applied_face else None,
            "applied_fk_max": max(applied_face) if applied_face else None,
            "applied_fk_final_mean": _safe_stat(
                result, "controller_trace_stats", "closed_loop_target_face_gap_m_mean", "final"
            ),
            "actual_proxy_min": min(actual_face) if actual_face else None,
            "actual_proxy_max": max(actual_face) if actual_face else None,
            "actual_proxy_log_final": _safe_stat(
                result, "log_trace_stats", "cube_tap_contact_face_gap_m", "final"
            ),
            "actual_shortfall_min": _safe_stat(
                result, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "min"
            ),
            "actual_shortfall_final": _safe_stat(
                result, "log_trace_stats", "cube_tap_contact_band_shortfall_m", "final"
            ),
        },
        "motion": {
            "cube_disp_detail_max_m": max(cube_disp) if cube_disp else None,
            "cube_speed_detail_max_mps": max(cube_speed) if cube_speed else None,
            "first_contact_step": _first_step(rows, "actual_contact_proxy"),
            "first_tap_success_seen_step": _first_step(rows, "tap_success_seen"),
            "terminated_rows": _termination_rows(rows),
        },
        "controller_follow": {
            "target_lead_abs_max_final": _safe_stat(
                result, "log_trace_stats", "cube_push_target_lead_abs_max", "final"
            ),
            "target_lead_limit_rate_final": _safe_stat(
                result, "log_trace_stats", "cube_push_target_lead_limit_rate", "final"
            ),
            "direct_follow_abs_max_final": _safe_stat(
                result, "controller_trace_stats", "direct_joint_follow_abs_max_rad", "final"
            ),
            "actual_joint_step_abs_max_final": _safe_stat(
                result, "controller_trace_stats", "direct_actual_joint_step_abs_max_rad", "final"
            ),
            "target_base_minus_actual_abs_max_final": _safe_stat(
                result, "controller_trace_stats", "builtin_diffik_target_base_minus_actual_abs_max_rad", "final"
            ),
            "target_fk_err_mm_final": _safe_stat(
                result, "controller_trace_stats", "closed_loop_target_fk_err_mm_mean", "final"
            ),
        },
        "torque": {
            "computed_abs_max_nm": computed_torque,
            "applied_abs_max_nm": applied_torque,
            "effort_limit_abs_max_nm": effort_limit,
            "applied_limit_fraction": torque_saturation_rows / torque_row_total if torque_row_total else 0.0,
        },
    }


def main() -> int:
    candidates = {name: _summarize_candidate(name, paths) for name, paths in CANDIDATES.items()}
    c1 = candidates["candidate1_prevtargetbase"]
    c2 = candidates["candidate2_prevtargetbase_driveboost"]
    c3 = candidates["candidate3_prevtargetbase_link5corner"]
    c4 = candidates["candidate4_prevtargetbase_lead120"]
    c5 = candidates["candidate5_link5aabb_pre020_degenerate"]
    c6 = candidates["candidate6_link5aabb_pre040_nondegenerate"]
    c1_disp = _num(c1["last_log"]["max_disp_along_m"])
    c2_disp = _num(c2["last_log"]["max_disp_along_m"])
    c1_actual_inside = int(c1["inside_rows"]["actual_proxy"]["row_count"])
    c2_actual_inside = int(c2["inside_rows"]["actual_proxy"]["row_count"])
    c3_actual_inside = int(c3["inside_rows"]["actual_proxy"]["row_count"])
    c4_actual_inside = int(c4["inside_rows"]["actual_proxy"]["row_count"])
    c5_actual_inside = int(c5["inside_rows"]["actual_proxy"]["row_count"])
    c6_actual_inside = int(c6["inside_rows"]["actual_proxy"]["row_count"])
    c1_applied_inside = int(c1["inside_rows"]["applied_fk"]["row_count"])
    c2_applied_inside = int(c2["inside_rows"]["applied_fk"]["row_count"])
    c3_applied_inside = int(c3["inside_rows"]["applied_fk"]["row_count"])
    c4_applied_inside = int(c4["inside_rows"]["applied_fk"]["row_count"])
    c5_applied_inside = int(c5["inside_rows"]["applied_fk"]["row_count"])
    c6_applied_inside = int(c6["inside_rows"]["applied_fk"]["row_count"])
    candidate6_nondegenerate = (
        c6["status"] == "PASS"
        and _num(c6["reset"]["initial_contact_proxy_rate"]) == 0.0
        and c6["motion"]["first_contact_step"] is not None
        and int(c6["motion"]["first_contact_step"]) > 0
        and _num(c6["last_log"]["tap_success"]) > 0.0
        and int(c6["terminated_count"]) == 0
        and int(c6["truncated_count"]) == 0
    )

    audit = {
        "artifact_type": "cube10cm_tap_rl_prevtarget_pass_candidates_audit_v1",
        "local_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "candidates": candidates,
        "comparison": {
            "previous_target_base_fixed_applied_fk": c1_applied_inside > 0,
            "driveboost_kept_applied_fk_inside": c2_applied_inside > 0,
            "link5corner_kept_applied_fk_inside": c3_applied_inside > 0,
            "lead120_kept_applied_fk_inside": c4_applied_inside > 0,
            "strict_actual_contact_proxy_still_zero": c1_actual_inside == 0 and c2_actual_inside == 0,
            "strict_actual_contact_proxy_still_zero_after_candidate4": all(
                count == 0 for count in (c1_actual_inside, c2_actual_inside, c3_actual_inside, c4_actual_inside)
            ),
            "driveboost_disp_ratio_vs_candidate1": c2_disp / c1_disp if c1_disp else None,
            "driveboost_contact_unblock": c2_actual_inside > 0,
            "link5corner_contact_unblock": c3_actual_inside > 0,
            "lead120_contact_unblock": c4_actual_inside > 0,
            "link5aabb_pre020_contact_unblock": c5_actual_inside > 0,
            "link5aabb_pre040_contact_unblock": c6_actual_inside > 0,
            "link5corner_best_shortfall_m": c3["face_gap_m"]["actual_shortfall_min"],
            "lead120_best_shortfall_m": c4["face_gap_m"]["actual_shortfall_min"],
            "candidate5_degenerate_initial_contact": _num(c5["reset"]["initial_contact_proxy_rate"]) > 0.0,
            "candidate6_nondegenerate_positive_control_pass": candidate6_nondegenerate,
            "controller_application_mismatch_remaining": False,
            "physical_proxy_or_contact_frame_mismatch_supported": c1_applied_inside > 0
            and c2_applied_inside > 0
            and c1_actual_inside == 0
            and c2_actual_inside == 0,
            "point_proxy_model_insufficient_supported": c1_applied_inside > 0
            and c2_applied_inside > 0
            and c3_applied_inside > 0
            and c4_applied_inside > 0
            and c1_actual_inside == 0
            and c2_actual_inside == 0
            and c3_actual_inside == 0
            and c4_actual_inside == 0,
        },
        "verdict": {
            "candidate1_role": "PASS_CANDIDATE_PARTIAL: previous-target-base makes applied FK enter the strict band",
            "candidate2_role": "FAIL: stronger drive increases cube motion but does not create strict actual contact proxy",
            "candidate3_role": "FAIL: link5 corner proxy retarget still leaves actual point proxy outside the strict band",
            "candidate4_role": "FAIL: larger target lead increases motion/termination but still does not create strict actual contact proxy",
            "candidate5_role": "PASS_BUT_DEGENERATE: geometry-aware AABB contact passes at reset with precontact_clearance_m=0.020",
            "candidate6_role": "PASS_NONDEGENERATE: AABB contact starts after reset with precontact_clearance_m=0.040 and no termination",
            "not_next": "do_not_repeat_same_audit_or_more_blind_driveboost; do_not_relax_contact_band; do_not_scale_dataset_rl",
            "next_pass_candidate": (
                "freeze candidate6 as the new positive-control contract and validate before any larger IsaacLab data/RL run"
            ),
            "blocked_until": "large dataset/RL/RoArm still require promotion validation; strict tiny positive-control is unblocked by candidate6",
        },
    }

    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_prevtarget_pass_candidates_audit_v1 "
        "local_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 candidate1 "
        f"status={c1['status']} applied_inside_rows={c1_applied_inside} actual_inside_rows={c1_actual_inside} "
        f"contact_seen={c1['last_log']['contact_seen']:.9f} tap_success={c1['last_log']['tap_success']:.9f} "
        f"terminated_count={c1['terminated_count']} max_disp_along_m={c1['last_log']['max_disp_along_m']:.9f} "
        f"target_fk_err_final_mm={c1['controller_follow']['target_fk_err_mm_final']:.9f}",
        "line3 candidate2 "
        f"status={c2['status']} applied_inside_rows={c2_applied_inside} actual_inside_rows={c2_actual_inside} "
        f"contact_seen={c2['last_log']['contact_seen']:.9f} tap_success={c2['last_log']['tap_success']:.9f} "
        f"terminated_count={c2['terminated_count']} max_disp_along_m={c2['last_log']['max_disp_along_m']:.9f} "
        f"target_fk_err_final_mm={c2['controller_follow']['target_fk_err_mm_final']:.9f}",
        "line4 driveboost_effect "
        f"disp_ratio_vs_candidate1={audit['comparison']['driveboost_disp_ratio_vs_candidate1']:.9f} "
        f"candidate1_speed={c1['last_log']['max_speed_mps']:.9f} candidate2_speed={c2['last_log']['max_speed_mps']:.9f} "
        f"candidate1_actual_shortfall_min={c1['face_gap_m']['actual_shortfall_min']:.9f} "
        f"candidate2_actual_shortfall_min={c2['face_gap_m']['actual_shortfall_min']:.9f}",
        "line5 candidate3 "
        f"status={c3['status']} proxy={c3['tool_contact_proxy_mode']} applied_inside_rows={c3_applied_inside} "
        f"actual_inside_rows={c3_actual_inside} contact_seen={c3['last_log']['contact_seen']:.9f} "
        f"terminated_count={c3['terminated_count']} actual_shortfall_min={c3['face_gap_m']['actual_shortfall_min']:.9f} "
        f"max_disp_along_m={c3['last_log']['max_disp_along_m']:.9f}",
        "line6 candidate4 "
        f"status={c4['status']} proxy={c4['tool_contact_proxy_mode']} lead_limit=0.120000000 "
        f"applied_inside_rows={c4_applied_inside} actual_inside_rows={c4_actual_inside} "
        f"contact_seen={c4['last_log']['contact_seen']:.9f} terminated_count={c4['terminated_count']} "
        f"actual_shortfall_min={c4['face_gap_m']['actual_shortfall_min']:.9f} "
        f"max_disp_along_m={c4['last_log']['max_disp_along_m']:.9f}",
        "line7 torque "
        f"candidate1_applied_limit_fraction={c1['torque']['applied_limit_fraction']:.9f} "
        f"candidate2_applied_limit_fraction={c2['torque']['applied_limit_fraction']:.9f} "
        f"candidate2_applied_abs_max_nm={c2['torque']['applied_abs_max_nm']:.9f} "
        f"candidate2_effort_limit_abs_max_nm={c2['torque']['effort_limit_abs_max_nm']:.9f}",
        "line8 verdict "
        f"previous_target_base_fixed_applied_fk={audit['comparison']['previous_target_base_fixed_applied_fk']} "
        f"strict_actual_contact_proxy_still_zero_after_candidate4={audit['comparison']['strict_actual_contact_proxy_still_zero_after_candidate4']} "
        f"point_proxy_model_insufficient_supported={audit['comparison']['point_proxy_model_insufficient_supported']}",
        "line9 candidate5 "
        f"status={c5['status']} initial_contact={c5['reset']['initial_contact_proxy_rate']:.9f} "
        f"first_contact_step={c5['motion']['first_contact_step']} tap_success={c5['last_log']['tap_success']:.9f} "
        f"terminated_count={c5['terminated_count']} degenerate_initial_contact={audit['comparison']['candidate5_degenerate_initial_contact']}",
        "line10 candidate6 "
        f"status={c6['status']} initial_contact={c6['reset']['initial_contact_proxy_rate']:.9f} "
        f"first_contact_step={c6['motion']['first_contact_step']} first_success_step={c6['motion']['first_tap_success_seen_step']} "
        f"actual_inside_rows={c6_actual_inside} tap_success={c6['last_log']['tap_success']:.9f} "
        f"terminated_count={c6['terminated_count']} truncated_count={c6['truncated_count']} "
        f"nondegenerate_pass={candidate6_nondegenerate}",
        "line11 next "
        "freeze_candidate6_positive_control_contract=YES "
        "stage0_isaaclab_validation_unblocked=YES large_dataset_rl_roarm_unblocked=NO "
        "not_next=single_point_proxy_or_blind_driveboost",
        f"line12 outputs json={OUT_JSON}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
