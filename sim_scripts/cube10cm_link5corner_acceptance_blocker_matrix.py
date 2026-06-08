"""Local acceptance/blocker matrix for the cube10cm link5-corner result.

This audit decides what the latest verified link5-corner evidence permits if the
professor/user accepts a weak 1mm tap/reaction objective. It reads existing logs
only. It does not run IsaacLab, use GPU, generate data, train, control a robot,
SSH, or mutate source traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_SUMMARY_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_summary.json"
)
DEFAULT_REACTION_GATE_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_reaction_gate_audit.json"
)
DEFAULT_WINDOW_JSON = LOG_DIR / "cube10cm_reaction_window_link5corner_position_seed962_audit.json"
DEFAULT_VISUAL_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_LEGACY_BLOCKER_JSON = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5corner_acceptance_blocker_matrix.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_acceptance_blocker_matrix_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE_JSON)
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_JSON)
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL_JSON)
    parser.add_argument("--legacy_blocker_json", type=Path, default=DEFAULT_LEGACY_BLOCKER_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    summary = _load_json(args.summary_json)
    reaction = _load_json(args.reaction_gate_json)
    window = _load_json(args.reaction_window_json)
    visual = _load_json(args.visual_json)
    legacy = _load_json(args.legacy_blocker_json)

    tiers = window.get("quality_tier_counts", {})
    visual_metrics = visual.get("contact_visual_metrics", {})
    visual_verdict = visual.get("verdict", {})
    code_conflicts = legacy.get("code_conflicts", {})

    primary_gate_pass = (
        _bool(reaction.get("reaction_gate_pass"))
        and _float(reaction.get("contact_evidence_rate")) >= 1.0
        and _float(reaction.get("reaction_event_rate")) >= 1.0
        and _float(reaction.get("overshoot_rate")) == 0.0
        and _bool(reaction.get("no_posewrite"))
    )
    weak_1mm_event_verified = (
        primary_gate_pass
        and _float(summary.get("max_disp_along_push_mean_m")) >= 0.001
        and _float(summary.get("disp_threshold_rates", {}).get("1mm")) >= 1.0
        and _bool(visual_verdict.get("weak_tap_visual_mechanism_supported"))
    )
    strong_tap_requirement_documented = False
    stop_contact_geometry_gpu_tuning = weak_1mm_event_verified and not strong_tap_requirement_documented

    clean_teacher_ready = _bool(window.get("clean_diffik_teacher_window_ready"))
    clip_mean = _float(window.get("accepted_window_clip_any_rate_mean"))
    follow_p95 = _float(window.get("accepted_window_follow_p95_to_cap_p95"))
    final_tcp_err = _float(reaction.get("summary_final_tcp_target_err_mean_m"))
    action_teacher_ready = clean_teacher_ready and clip_mean <= 0.5 and final_tcp_err <= 0.03

    legacy_builder_conflict = _bool(code_conflicts.get("legacy_dataset_builder_final_success_conflict"))
    rl_env_conflict = _bool(code_conflicts.get("rl_env_3cm_relocation_conflict"))

    matrix = [
        {
            "item": "professor_tap_reaction_objective",
            "status": "ACCEPT_WEAK_1MM_EVENT_UNLESS_STRONG_TAP_EXPLICITLY_REQUIRED"
            if weak_1mm_event_verified
            else "BLOCKED",
            "current_evidence": "primary gate PASS plus local visual inspection supports weak/grazing 1mm tap",
            "unblock_or_next": "record a new 2-3mm transient target only if professor/user explicitly requires it",
            "can_proceed_local_only": True,
        },
        {
            "item": "contact_geometry_gpu_tuning",
            "status": "STOP" if stop_contact_geometry_gpu_tuning else "BLOCKED_NEEDS_EXPLICIT_STRONG_TAP_TARGET",
            "current_evidence": "link5-corner proxy is side-center height but outside/grazing and early-freeze",
            "unblock_or_next": "do not run another GPU sweep from contact geometry without explicit stronger target",
            "can_proceed_local_only": False,
        },
        {
            "item": "link5_corner_quality_tier_metadata",
            "status": "KEEP_AS_SIDE_CENTER_PROXY_EVIDENCE",
            "current_evidence": "reaction windows are accepted and all Tier B, but clean teacher is false",
            "unblock_or_next": "carry tier/clip/follow/proxy-contact fields as metadata, not action-dataset readiness",
            "can_proceed_local_only": True,
        },
        {
            "item": "event_label_dataset_or_manifest",
            "status": "READY_LOCAL_ONLY",
            "current_evidence": "event labels may use contact/reaction/no-overshoot plus quality metadata",
            "unblock_or_next": "build or update only a link5-specific event-label manifest; no action targets",
            "can_proceed_local_only": True,
        },
        {
            "item": "diffik_action_teacher_dataset",
            "status": "BLOCKED" if not action_teacher_ready else "READY",
            "current_evidence": "clean teacher false, clip/final TCP gates not clean even though follow tier improved",
            "unblock_or_next": "needs clean teacher evidence or an explicit noisy-teacher exception policy",
            "can_proceed_local_only": False,
        },
        {
            "item": "large_isaaclab_dataset",
            "status": "BLOCKED",
            "current_evidence": "action teacher is blocked and current link5 result is weak 1mm evidence only",
            "unblock_or_next": "only after action-teacher policy and dataset schema gates are resolved",
            "can_proceed_local_only": False,
        },
        {
            "item": "ten_cm_tap_rl_env_random_sanity",
            "status": "BLOCKED_BY_ENV_CONTRACT" if rl_env_conflict else "READY_FOR_LOCAL_PREFLIGHT",
            "current_evidence": "existing blocker audit says current env is 3cm relocation-oriented",
            "unblock_or_next": "write/validate 10cm/0.72kg tap reaction env random-sanity gate before training",
            "can_proceed_local_only": True,
        },
        {
            "item": "isaaclab_rl_training",
            "status": "BLOCKED",
            "current_evidence": "requires action/schema/env gates first",
            "unblock_or_next": "no PPO/RL until local preflight gates pass",
            "can_proceed_local_only": False,
        },
        {
            "item": "roarm_m3_pro_deployment",
            "status": "BLOCKED",
            "current_evidence": "no validated learned policy or hardware safety/replay gate",
            "unblock_or_next": "only after validated policy plus robot safety/replay evidence",
            "can_proceed_local_only": False,
        },
    ]

    result = {
        "artifact_type": "cube10cm_link5corner_acceptance_blocker_matrix_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_matrix_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "decision": {
            "strong_tap_required_by_current_docs": strong_tap_requirement_documented,
            "weak_1mm_tap_reaction_accepted_as_current_working_objective": weak_1mm_event_verified,
            "stop_contact_geometry_gpu_tuning": stop_contact_geometry_gpu_tuning,
            "do_not_claim_action_teacher_dataset_rl_roarm_readiness": True,
        },
        "evidence": {
            "reaction_gate_pass": _bool(reaction.get("reaction_gate_pass")),
            "contact_evidence_rate": _float(reaction.get("contact_evidence_rate")),
            "reaction_event_rate": _float(reaction.get("reaction_event_rate")),
            "overshoot_rate": _float(reaction.get("overshoot_rate")),
            "no_posewrite": _bool(reaction.get("no_posewrite")),
            "max_disp_along_push_mean_m": _float(summary.get("max_disp_along_push_mean_m")),
            "max_cube_speed_mean_mps": _float(summary.get("max_cube_speed_mean_mps")),
            "low_motion_rate": _float(summary.get("low_motion_rate")),
            "disp_1mm_rate": _float(summary.get("disp_threshold_rates", {}).get("1mm")),
            "disp_5mm_rate": _float(summary.get("disp_threshold_rates", {}).get("5mm")),
            "clean_diffik_teacher_window_ready": clean_teacher_ready,
            "accepted_window_clip_any_rate_mean": clip_mean,
            "accepted_window_follow_p95_to_cap_p95": follow_p95,
            "quality_tier_counts": tiers,
            "summary_final_tcp_target_err_mean_m": final_tcp_err,
            "proxy_side_center_z_near_5mm_rate": _float(visual_metrics.get("proxy_side_center_z_near_5mm_rate")),
            "proxy_not_top_rate": _float(visual_metrics.get("proxy_not_top_rate")),
            "proxy_gap_to_live_side_face_mean_m": _float(
                visual_metrics.get("proxy_gap_to_live_side_face_m", {}).get("mean")
            ),
            "target_gap_to_live_side_face_mean_m": _float(
                visual_metrics.get("target_gap_to_live_side_face_m", {}).get("mean")
            ),
            "contact_stop_same_as_contact_rate": _float(
                visual_metrics.get("contact_stop_same_as_contact_rate")
            ),
            "legacy_dataset_builder_final_success_conflict": legacy_builder_conflict,
            "rl_env_3cm_relocation_conflict": rl_env_conflict,
        },
        "matrix": matrix,
        "next_local_only_step": (
            "freeze 1mm reaction objective unless explicitly upgraded; update/build link5-specific "
            "event-label metadata manifest and teacher-policy matrix, not action data or RL"
        ),
        "source_files": {
            "summary_json": str(args.summary_json),
            "reaction_gate_json": str(args.reaction_gate_json),
            "reaction_window_json": str(args.reaction_window_json),
            "visual_json": str(args.visual_json),
            "legacy_blocker_json": str(args.legacy_blocker_json),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_link5corner_acceptance_blocker_matrix_v1 "
        "local_matrix_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 decision "
            f"weak_1mm_tap_reaction_accepted={weak_1mm_event_verified} "
            f"strong_tap_required_by_current_docs={strong_tap_requirement_documented} "
            f"stop_contact_geometry_gpu_tuning={stop_contact_geometry_gpu_tuning}"
        ),
        (
            "line3 event_evidence "
            f"reaction_gate_pass={reaction.get('reaction_gate_pass')} "
            f"contact={_float(reaction.get('contact_evidence_rate')):.9f} "
            f"reaction={_float(reaction.get('reaction_event_rate')):.9f} "
            f"overshoot={_float(reaction.get('overshoot_rate')):.9f} "
            f"no_posewrite={reaction.get('no_posewrite')} "
            f"max_disp={_float(summary.get('max_disp_along_push_mean_m')):.9f} "
            f"speed={_float(summary.get('max_cube_speed_mean_mps')):.9f} "
            f"low_motion={_float(summary.get('low_motion_rate')):.9f}"
        ),
        (
            "line4 visual_contact "
            f"side_center_z_rate={_float(visual_metrics.get('proxy_side_center_z_near_5mm_rate')):.9f} "
            f"proxy_not_top_rate={_float(visual_metrics.get('proxy_not_top_rate')):.9f} "
            f"proxy_gap={_float(visual_metrics.get('proxy_gap_to_live_side_face_m', {}).get('mean')):.9f} "
            f"target_gap={_float(visual_metrics.get('target_gap_to_live_side_face_m', {}).get('mean')):.9f} "
            f"contact_stop_same={_float(visual_metrics.get('contact_stop_same_as_contact_rate')):.9f}"
        ),
        (
            "line5 quality_teacher "
            f"clean_teacher={clean_teacher_ready} "
            f"tiers={tiers} "
            f"clip_mean={clip_mean:.9f} "
            f"follow_p95_to_cap={follow_p95:.9f} "
            f"final_tcp_err={final_tcp_err:.9f} "
            f"action_teacher_ready={action_teacher_ready}"
        ),
        (
            "line6 matrix "
            "event_label_manifest=READY_LOCAL_ONLY "
            "diffik_action_teacher=BLOCKED "
            "large_isaaclab_dataset=BLOCKED "
            f"ten_cm_rl_env_random_sanity={'BLOCKED_BY_ENV_CONTRACT' if rl_env_conflict else 'READY_FOR_LOCAL_PREFLIGHT'} "
            "isaaclab_rl=BLOCKED roarm_m3_pro=BLOCKED"
        ),
        (
            "line7 next_local_only "
            "build_or_update_link5_specific_event_label_metadata_manifest_and_teacher_policy_matrix; "
            "do_not_run_gpu_dataset_rl_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
