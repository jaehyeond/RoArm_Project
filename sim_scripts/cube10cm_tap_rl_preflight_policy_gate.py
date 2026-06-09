"""Local preflight/policy gate for the 10cm/0.72kg tap RL branch.

This consolidates existing evidence only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_EVENT_MANIFEST = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_TEACHER_POLICY = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate.json"
DEFAULT_VISUAL_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_RUNTIME_GATE = LOG_DIR / "cube10cm_tap_rl_env_runtime_gate_audit.json"
DEFAULT_POSITIVE_CONTROL = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_sanity.json"
)
DEFAULT_DIRECT_IK_AUDIT = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_preflight_policy_gate.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_preflight_policy_gate_summary.out"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(value: bool, pass_name: str = "PASS", fail_name: str = "BLOCKED") -> str:
    return pass_name if value else fail_name


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_manifest", type=Path, default=DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--teacher_policy", type=Path, default=DEFAULT_TEACHER_POLICY)
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL_JSON)
    parser.add_argument("--runtime_gate", type=Path, default=DEFAULT_RUNTIME_GATE)
    parser.add_argument("--positive_control_json", type=Path, default=DEFAULT_POSITIVE_CONTROL)
    parser.add_argument("--direct_ik_audit_json", type=Path, default=DEFAULT_DIRECT_IK_AUDIT)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    event_manifest = _load(args.event_manifest)
    teacher_policy = _load(args.teacher_policy)
    visual = _load(args.visual_json)
    runtime_gate = _load(args.runtime_gate)
    positive_control = _load(args.positive_control_json) if args.positive_control_json.exists() else {}
    direct_ik_audit = _load(args.direct_ik_audit_json) if args.direct_ik_audit_json.exists() else {}

    counts = event_manifest.get("counts", {})
    policy_statuses = teacher_policy.get("statuses", {})
    policy_interp = teacher_policy.get("policy_interpretation", {})
    policy_evidence = teacher_policy.get("evidence", {})
    visual_metrics = visual.get("contact_visual_metrics", {})

    event_label_ready = (
        counts.get("event_count") == 16
        and counts.get("contact_count") == 16
        and counts.get("reaction_count") == 16
        and counts.get("overshoot_count") == 0
        and counts.get("weak_1mm_count") == 16
        and counts.get("action_teacher_usable_default_count") == 0
        and policy_statuses.get("event_label_metadata_manifest") == "READY_LOCAL_ONLY"
    )
    weak_1mm_is_only_verified_objective = (
        counts.get("max_transient_ge_2mm_count") == 0
        and counts.get("max_transient_ge_3mm_count") == 0
        and bool(policy_evidence.get("weak_1mm_objective_accepted")) is True
    )
    visual_risk_preserved = (
        visual_metrics.get("proxy_outside_live_face_rate") == 1.0
        and visual_metrics.get("target_outside_live_face_rate") == 1.0
        and visual_metrics.get("contact_stop_same_as_contact_rate") == 1.0
        and bool(policy_evidence.get("clean_tap_strength_visual_verified")) is False
    )
    wrapper_preflight_ready = (
        bool(runtime_gate.get("wrapper_sanity_pass")) is True
        and runtime_gate.get("unblocked", {}).get("default_off_10cm_tap_env_wrapper_contract") is True
        and runtime_gate.get("random_contract_only") is True
        and runtime_gate.get("zero_quiet_no_action") is True
    )
    strict_action_teacher_ready = policy_statuses.get("strict_clean_action_teacher") == "READY"
    noisy_tierb_exception_recorded = bool(policy_interp.get("explicit_exception_requested")) is True
    tiny_action_dataset_allowed = strict_action_teacher_ready or noisy_tierb_exception_recorded

    professor_link5_physical_evidence_ready = (
        bool(event_manifest.get("local_manifest_only")) is True
        and int(counts.get("event_count", 0)) > 0
        and int(counts.get("reaction_count", 0)) > 0
        and int(counts.get("overshoot_count", -1)) == 0
        and policy_statuses.get("event_label_metadata_manifest") == "READY_LOCAL_ONLY"
    )
    positive_last_log = positive_control.get("last_log", {})
    positive_max_disp = _float(positive_last_log.get("cube_tap_max_disp_along_m"))
    positive_max_speed = _float(positive_last_log.get("cube_tap_max_speed_mps"))
    positive_overshoot = _float(positive_last_log.get("cube_tap_overshoot_seen_rate"), 1.0)
    positive_physical_evidence_ready = (
        positive_control.get("professor_physical_reaction_evidence") == "PASS"
        or direct_ik_audit.get("professor_physical_reaction_evidence") == "PASS"
        or ((positive_max_disp >= 0.0005 or positive_max_speed >= 0.005) and positive_overshoot == 0.0)
    )
    professor_physical_reaction_evidence_ready = (
        professor_link5_physical_evidence_ready or positive_physical_evidence_ready
    )

    positive_control_valid_runtime = (
        positive_control.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL"
        and positive_control.get("device") == "cuda:0"
        and positive_control.get("dataset_generation") is False
        and positive_control.get("training") is False
        and positive_control.get("robot_control") is False
        and positive_control.get("ssh") is False
        and positive_control.get("b200") is False
        and positive_control.get("track_a") is False
        and int(positive_control.get("steps_executed", 0)) > 0
    )
    positive_control_tap_sanity_ready = positive_control.get("positive_control") == "PASS"
    if positive_control_tap_sanity_ready:
        positive_control_status = "READY"
    elif positive_control_valid_runtime:
        positive_control_status = "RUN_FAILED"
    else:
        positive_control_status = "BLOCKED_NOT_RUN"
    ppo_preflight_ready = (
        wrapper_preflight_ready
        and event_label_ready
        and positive_control_tap_sanity_ready
        and tiny_action_dataset_allowed
    )

    gate_matrix = [
        {
            "gate": "professor_physical_reaction_evidence",
            "status": _status(
                professor_physical_reaction_evidence_ready,
                "READY_PROFESSOR_EVIDENCE_ONLY",
            ),
            "evidence": {
                "link5_event_reactions": counts.get("reaction_count"),
                "link5_overshoot": counts.get("overshoot_count"),
                "link5_event_label_ready": policy_statuses.get("event_label_metadata_manifest"),
                "direct_ik_professor_physical_reaction": direct_ik_audit.get(
                    "professor_physical_reaction_evidence"
                ),
                "direct_ik_max_disp_m": direct_ik_audit.get("max_disp_along_m"),
                "direct_ik_max_speed_mps": direct_ik_audit.get("max_speed_mps"),
            },
            "meaning": (
                "Use this for the professor/user weak physical object-reaction objective. "
                "It is separate from action-teacher, dataset, RL, and robot readiness."
            ),
        },
        {
            "gate": "event_label_quality_tier_metadata",
            "status": _status(event_label_ready, "READY_LOCAL_ONLY"),
            "evidence": {
                "events": counts.get("event_count"),
                "contact": counts.get("contact_count"),
                "reaction": counts.get("reaction_count"),
                "overshoot": counts.get("overshoot_count"),
                "weak_1mm": counts.get("weak_1mm_count"),
                "tier_counts": counts.get("quality_tier_counts", {}),
            },
            "meaning": "May be used as event-label/quality-tier metadata, not as action targets.",
        },
        {
            "gate": "default_off_10cm_tap_env_wrapper",
            "status": _status(wrapper_preflight_ready, "READY_LOCAL_PREFLIGHT_ONLY"),
            "evidence": {
                "wrapper_sanity_pass": runtime_gate.get("wrapper_sanity_pass"),
                "zero_quiet_no_action": runtime_gate.get("zero_quiet_no_action"),
                "random_contract_only": runtime_gate.get("random_contract_only"),
                "zero_metrics": runtime_gate.get("zero_metrics", {}),
                "random_metrics": runtime_gate.get("random_metrics", {}),
            },
            "meaning": "Env contract can be used for local preflight/design. It is not PPO evidence.",
        },
        {
            "gate": "strong_2_3mm_tap_requirement",
            "status": "NOT_REQUIRED_BY_CURRENT_EVIDENCE",
            "evidence": {
                "weak_1mm_is_only_verified_objective": weak_1mm_is_only_verified_objective,
                "ge_2mm_count": counts.get("max_transient_ge_2mm_count"),
                "ge_3mm_count": counts.get("max_transient_ge_3mm_count"),
            },
            "meaning": "Do not tune contact geometry for 2-3mm unless professor/user records it as explicit target.",
        },
        {
            "gate": "strict_clean_action_teacher",
            "status": "BLOCKED",
            "evidence": {
                "clean_teacher": policy_evidence.get("clean_diffik_teacher_window_ready"),
                "tier_a": policy_evidence.get("tier_a"),
                "tier_b": policy_evidence.get("tier_b"),
                "clip_mean": policy_evidence.get("accepted_window_clip_any_rate_mean"),
                "follow_p95_to_cap": policy_evidence.get("accepted_window_follow_p95_to_cap_p95"),
            },
            "meaning": "Do not use link5 DiffIK actions as default imitation targets.",
        },
        {
            "gate": "noisy_tier_b_action_teacher_exception",
            "status": "REQUIRES_EXPLICIT_USER_PROFESSOR_EXCEPTION",
            "evidence": {
                "candidate_rows": policy_evidence.get("accepted_windows"),
                "tier_b": policy_evidence.get("tier_b"),
                "visual_risk_preserved": visual_risk_preserved,
                "explicit_exception_recorded": noisy_tierb_exception_recorded,
            },
            "meaning": "If later allowed, it only permits a tiny audited dry run, not large data/RL/robot.",
        },
        {
            "gate": "positive_control_tap_sanity_in_new_wrapper",
            "status": positive_control_status,
            "evidence": {
                "zero_action_contact": runtime_gate.get("zero_metrics", {}).get("contact_seen"),
                "random_action_contact": runtime_gate.get("random_metrics", {}).get("contact_seen"),
                "random_action_tap_success": runtime_gate.get("random_metrics", {}).get("tap_success"),
                "positive_control_valid_runtime": positive_control_valid_runtime,
                "positive_control_result": positive_control.get("positive_control", "NOT_RUN"),
                "positive_contact_seen": positive_control.get("last_log", {}).get("cube_tap_contact_seen_rate"),
                "positive_reaction_signal": positive_control.get("last_log", {}).get("cube_tap_reaction_signal_now_rate"),
                "positive_reaction_context": positive_control.get("last_log", {}).get("cube_tap_reaction_contact_context_rate"),
                "positive_tap_success": positive_control.get("last_log", {}).get("cube_tap_success_rate"),
            },
            "meaning": "Before PPO, the wrapper should register contact-gated reaction under a deliberate tiny scripted policy.",
        },
        {
            "gate": "tiny_action_dataset_dry_run",
            "status": "BLOCKED_UNTIL_EXCEPTION_OR_CLEAN_TEACHER",
            "evidence": {
                "strict_action_teacher_ready": strict_action_teacher_ready,
                "noisy_tierb_exception_recorded": noisy_tierb_exception_recorded,
                "tiny_action_dataset_allowed": tiny_action_dataset_allowed,
            },
            "meaning": "This is format/filter validation only; currently not allowed.",
        },
        {
            "gate": "ppo_rl_training",
            "status": "BLOCKED",
            "evidence": {
                "wrapper_preflight_ready": wrapper_preflight_ready,
                "positive_control_tap_sanity_ready": positive_control_tap_sanity_ready,
                "teacher_or_exception_ready": tiny_action_dataset_allowed,
            },
            "meaning": "Do not start PPO from random sanity alone.",
        },
        {
            "gate": "roarm_m3_pro_deploy",
            "status": "BLOCKED",
            "evidence": {
                "ppo_policy_validated": False,
                "hardware_safety_replay_validated": False,
            },
            "meaning": "No robot step before policy validation plus safety replay.",
        },
    ]

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_preflight_policy_gate_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_preflight_policy_gate_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {
            "event_manifest": str(args.event_manifest),
            "teacher_policy": str(args.teacher_policy),
            "visual_json": str(args.visual_json),
            "runtime_gate": str(args.runtime_gate),
            "positive_control_json": str(args.positive_control_json),
            "direct_ik_audit_json": str(args.direct_ik_audit_json),
        },
        "high_level_verdict": {
            "isaac_lab_confusion_resolved": "LOCAL_GPU_WRAPPER_SANITY_ALREADY_PASSED; CPU_OR_SANDBOX_FAILURE_IS_NOT_PROMOTION_EVIDENCE",
            "professor_physical_reaction_evidence_ready": professor_physical_reaction_evidence_ready,
            "professor_evidence_separate_from_rl_positive_control": True,
            "may_move_to_local_preflight_design": wrapper_preflight_ready and event_label_ready,
            "may_move_to_ppo_training": ppo_preflight_ready,
            "may_move_to_large_dataset": False,
            "may_move_to_roarm": False,
        },
        "gate_matrix": gate_matrix,
        "next_allowed_local_only": [
            "freeze reward/done/log contract for RoArm-CubeTap10cm-Direct-v0",
            "design one tiny scripted positive-control tap sanity for the new wrapper",
            "if positive-control failed, design one revised closed-loop positive-control candidate",
            "review whether noisy Tier-B action-teacher exception is explicitly allowed",
            "prepare professor-facing physical-reaction evidence package without claiming dataset/RL/RoArm readiness",
        ],
        "not_allowed": [
            "ppo training",
            "large dataset generation",
            "RoArm deployment",
            "noisy Tier-B action teacher dataset without explicit exception",
            "2-3mm contact-geometry tuning without explicit professor/user requirement",
            "blocking professor physical-reaction evidence on RL contact-gated positive-control",
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_preflight_policy_gate_v1 "
        "local_preflight_policy_gate_only=YES gpu_runtime=NO dataset_generation=NO "
        "training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 confusion_check "
            "isaac_lab_status=OK_LOCAL_GPU_WRAPPER_SANITY_ALREADY_PASSED "
            "cpu_or_sandbox_failure_is_not_promotion_evidence=YES "
            f"wrapper_preflight_ready={wrapper_preflight_ready}"
        ),
        (
            "line3 unblocked "
            f"professor_physical_reaction_evidence={_status(professor_physical_reaction_evidence_ready, 'READY_PROFESSOR_EVIDENCE_ONLY')} "
            f"event_label_metadata={_status(event_label_ready, 'READY_LOCAL_ONLY')} "
            f"env_wrapper={_status(wrapper_preflight_ready, 'READY_LOCAL_PREFLIGHT_ONLY')} "
            f"weak_1mm_only_verified={weak_1mm_is_only_verified_objective} "
            "strong_2_3mm_required_by_current_evidence=NO "
            "professor_evidence_separate_from_rl_positive_control=YES"
        ),
        (
            "line4 teacher_policy "
            "strict_clean_action_teacher=BLOCKED "
            "noisy_tierb_exception=REQUIRES_EXPLICIT_USER_PROFESSOR_EXCEPTION "
            f"tiny_action_dataset_allowed={tiny_action_dataset_allowed} "
            f"tier_b={policy_evidence.get('tier_b')} "
            f"clip_mean={float(policy_evidence.get('accepted_window_clip_any_rate_mean', 0.0)):.9f}"
        ),
        (
            "line5 visual_risk "
            f"clean_tap_visual_verified={policy_evidence.get('clean_tap_strength_visual_verified')} "
            f"grazing_or_outside={policy_evidence.get('grazing_or_outside_face_supported')} "
            f"early_freeze={policy_evidence.get('early_freeze_supported')} "
            f"visual_risk_preserved={visual_risk_preserved}"
        ),
        (
            "line6 rl_preflight "
            f"positive_control_tap_sanity={positive_control_status} "
            f"professor_physical_reaction_evidence={_status(professor_physical_reaction_evidence_ready, 'READY_PROFESSOR_EVIDENCE_ONLY')} "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED "
            "reason=rl_contact_gated_positive_control_blocks_dataset_rl_not_professor_physical_evidence"
        ),
        (
            "line7 next "
            "allowed=professor_physical_reaction_evidence_package_or_local_rl_blocker_debug "
            "not_allowed=claiming_action_teacher_dataset_rl_roarm_readiness"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)

    return 0 if professor_physical_reaction_evidence_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
