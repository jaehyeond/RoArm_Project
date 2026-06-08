"""Next-stage promotion gate for the cube10cm link5-corner branch.

This answers what can advance now, what needs an explicit policy exception, and
what remains blocked before large dataset/RL/RoArm. It reads existing local
evidence only. It does not run IsaacLab, use GPU, generate datasets, train,
control a robot, SSH, or mutate source traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_EVENT_MANIFEST = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_POLICY_GATE = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate.json"
DEFAULT_VISUAL_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_BLOCKER_JSON = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5corner_next_stage_promotion_gate.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_next_stage_promotion_gate_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _status_is(data: dict[str, Any], path: tuple[str, ...], expected: str) -> bool:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return current == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_manifest_json", type=Path, default=DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--policy_gate_json", type=Path, default=DEFAULT_POLICY_GATE)
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL_JSON)
    parser.add_argument("--blocker_json", type=Path, default=DEFAULT_BLOCKER_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    manifest = _load_json(args.event_manifest_json)
    policy = _load_json(args.policy_gate_json)
    visual = _load_json(args.visual_json)
    blocker = _load_json(args.blocker_json)

    counts = manifest.get("counts", {})
    event_labels_ready = (
        bool(manifest.get("local_manifest_only"))
        and bool(manifest.get("not_action_teacher_dataset"))
        and _int(counts.get("event_count")) == 16
        and _int(counts.get("contact_count")) == 16
        and _int(counts.get("reaction_count")) == 16
        and _int(counts.get("overshoot_count")) == 0
        and _int(counts.get("weak_1mm_count")) == 16
    )
    visual_weak_only = (
        bool(visual.get("verdict", {}).get("side_center_proxy_visual_verified"))
        and bool(visual.get("verdict", {}).get("grazing_or_outside_face_supported"))
        and bool(visual.get("verdict", {}).get("early_freeze_supported"))
        and not bool(visual.get("verdict", {}).get("clean_tap_strength_visual_verified"))
    )
    strict_clean_blocked = _status_is(policy, ("statuses", "strict_clean_action_teacher"), "BLOCKED")
    noisy_candidate_exists = _status_is(
        policy, ("statuses", "noisy_tierb_action_teacher_candidate"), "CANDIDATE_EXISTS"
    )
    noisy_exception_missing = _status_is(
        policy,
        ("statuses", "noisy_tierb_exception"),
        "REQUIRES_EXPLICIT_USER_PROFESSOR_EXCEPTION",
    )
    rl_env_conflict = bool(
        blocker.get("code_conflicts", {}).get("rl_env_3cm_relocation_conflict")
    )
    legacy_builder_conflict = bool(
        blocker.get("code_conflicts", {}).get("legacy_dataset_builder_final_success_conflict")
    )

    stages = [
        {
            "stage": 0,
            "name": "objective_and_visual_evidence",
            "promotion": "PASS_WEAK_1MM_ONLY" if event_labels_ready and visual_weak_only else "BLOCKED",
            "evidence": "16/16 weak 1mm reaction labels and visual side-center/grazing/early-freeze review",
            "next": "do not chase 2-3mm unless explicitly required",
        },
        {
            "stage": 1,
            "name": "event_label_quality_metadata",
            "promotion": "CAN_ADVANCE_NOW_DONE" if event_labels_ready else "BLOCKED",
            "evidence": "link5 event-label metadata manifest exists and has no action targets",
            "next": "use as local metadata evidence only",
        },
        {
            "stage": 2,
            "name": "strict_clean_action_teacher",
            "promotion": "BLOCKED" if strict_clean_blocked else "READY",
            "evidence": "clean teacher false; Tier B is noisy because clip remains high",
            "next": "needs clean teacher evidence, not just accepted windows",
        },
        {
            "stage": 3,
            "name": "noisy_tierb_action_teacher_exception",
            "promotion": "WAITING_EXPLICIT_EXCEPTION"
            if noisy_candidate_exists and noisy_exception_missing
            else "READY_OR_NOT_APPLICABLE",
            "evidence": "16 Tier-B candidate windows exist, but default policy blocks action dataset",
            "next": "only user/professor can explicitly accept risky noisy teacher for a tiny dry run",
        },
        {
            "stage": 4,
            "name": "tiny_action_dataset_dryrun",
            "promotion": "BLOCKED_UNLESS_EXCEPTION_RECORDED",
            "evidence": "no explicit noisy Tier-B exception recorded",
            "next": "if exception is recorded later, dry run only; not a large dataset",
        },
        {
            "stage": 5,
            "name": "ten_cm_tap_rl_env_contract_preflight",
            "promotion": "CAN_ADVANCE_LOCAL_ONLY" if rl_env_conflict else "READY_FOR_LOCAL_PREFLIGHT",
            "evidence": "existing env is 3cm/20g relocation-oriented, so only contract/preflight work is allowed",
            "next": "write local 10cm/0.72kg tap env contract/random-sanity audit before training",
        },
        {
            "stage": 6,
            "name": "large_dataset_rl_roarm",
            "promotion": "BLOCKED",
            "evidence": "action teacher policy, 10cm env sanity, and robot safety gates are unresolved",
            "next": "no large dataset, PPO/RL training, or RoArm deployment",
        },
    ]

    result = {
        "artifact_type": "cube10cm_link5corner_next_stage_promotion_gate_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_promotion_gate_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "answer": {
            "can_move_now_to": [
                "event_label_quality_metadata_done",
                "ten_cm_tap_rl_env_contract_preflight_local_only",
            ],
            "cannot_move_yet_to": [
                "strict_clean_action_teacher_dataset",
                "tiny_action_dataset_dryrun_without_exception",
                "large_isaaclab_dataset",
                "isaaclab_rl_training",
                "roarm_m3_pro_deployment",
            ],
            "policy_exception_needed_for": "noisy_tierb_action_teacher_tiny_dryrun_only",
        },
        "evidence_flags": {
            "event_labels_ready": event_labels_ready,
            "visual_supports_weak_1mm_only": visual_weak_only,
            "strict_clean_action_teacher_blocked": strict_clean_blocked,
            "noisy_tierb_candidate_exists": noisy_candidate_exists,
            "noisy_exception_missing": noisy_exception_missing,
            "legacy_builder_final_success_conflict": legacy_builder_conflict,
            "rl_env_3cm_relocation_conflict": rl_env_conflict,
        },
        "stages": stages,
        "source_files": {
            "event_manifest_json": str(args.event_manifest_json),
            "policy_gate_json": str(args.policy_gate_json),
            "visual_json": str(args.visual_json),
            "blocker_json": str(args.blocker_json),
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_link5corner_next_stage_promotion_gate_v1 "
        "local_promotion_gate_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 move_now "
            "event_label_quality_metadata=DONE "
            f"ten_cm_tap_rl_env_contract_preflight={'READY_LOCAL_ONLY' if rl_env_conflict else 'READY'}"
        ),
        (
            "line3 weak_visual_gate "
            f"event_labels_ready={event_labels_ready} visual_supports_weak_1mm_only={visual_weak_only} "
            "strong_2_3mm_required=False"
        ),
        (
            "line4 action_teacher_gate "
            f"strict_clean_blocked={strict_clean_blocked} "
            f"noisy_tierb_candidate_exists={noisy_candidate_exists} "
            f"explicit_exception_missing={noisy_exception_missing}"
        ),
        (
            "line5 code_env_blockers "
            f"legacy_builder_final_success_conflict={legacy_builder_conflict} "
            f"rl_env_3cm_relocation_conflict={rl_env_conflict}"
        ),
        (
            "line6 cannot_move_yet "
            "tiny_action_dataset_dryrun_without_exception,large_isaaclab_dataset,"
            "isaaclab_rl_training,roarm_m3_pro_deployment"
        ),
        (
            "line7 next_step "
            "write_10cm_tap_rl_env_contract_preflight_local_only; "
            "or_record_explicit_noisy_tierb_exception_before_tiny_action_dryrun"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
