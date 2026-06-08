"""Policy gate for link5-corner noisy Tier-B action-teacher use.

The link5-corner result has 16 valid Tier-B reaction windows, but Tier B means
follow is acceptable while DiffIK clipping is still high. This script separates
event-label readiness from action-teacher permission. It records policy status
only and never builds an action dataset, runs IsaacLab/GPU, trains, controls a
robot, uses SSH, or mutates traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_MANIFEST_JSON = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_MATRIX_JSON = LOG_DIR / "cube10cm_link5corner_acceptance_blocker_matrix.json"
DEFAULT_WINDOW_JSON = LOG_DIR / "cube10cm_reaction_window_link5corner_position_seed962_audit.json"
DEFAULT_VISUAL_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate_summary.out"


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
    parser.add_argument("--event_manifest_json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--acceptance_matrix_json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_JSON)
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL_JSON)
    parser.add_argument(
        "--explicit_allow_noisy_tierb_action_teacher",
        action="store_true",
        help="Record an explicit user/professor policy exception; does not build action data.",
    )
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    manifest = _load_json(args.event_manifest_json)
    matrix = _load_json(args.acceptance_matrix_json)
    window = _load_json(args.reaction_window_json)
    visual = _load_json(args.visual_json)

    counts = manifest.get("counts", {})
    tier_counts = window.get("quality_tier_counts", {})
    event_count = int(counts.get("event_count", 0))
    tier_a = int(tier_counts.get("A_CLEAN_DIFFIK_TEACHER", 0))
    tier_b = int(tier_counts.get("B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH", 0))
    tier_c = int(tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0))
    accepted = int(window.get("accepted_window_count", 0))
    clean_ready = _bool(window.get("clean_diffik_teacher_window_ready"))
    clip_mean = _float(window.get("accepted_window_clip_any_rate_mean"))
    follow_p95 = _float(window.get("accepted_window_follow_p95_to_cap_p95"))
    weak_objective_accepted = _bool(
        matrix.get("decision", {}).get("weak_1mm_tap_reaction_accepted_as_current_working_objective")
    )
    clean_tap_visual_verified = _bool(
        visual.get("verdict", {}).get("clean_tap_strength_visual_verified")
    )
    grazing_supported = _bool(visual.get("verdict", {}).get("grazing_or_outside_face_supported"))
    early_freeze = _bool(visual.get("verdict", {}).get("early_freeze_supported"))

    event_label_ready = (
        weak_objective_accepted
        and _bool(manifest.get("local_manifest_only"))
        and _bool(manifest.get("not_action_teacher_dataset"))
        and event_count == int(counts.get("weak_1mm_count", -1))
        and int(counts.get("overshoot_count", -1)) == 0
    )
    strict_clean_ready = clean_ready and tier_a == accepted and accepted > 0 and clip_mean <= 0.5
    noisy_tierb_candidate_rows = tier_b if tier_c == 0 and tier_b == accepted else 0
    noisy_tierb_candidate_exists = noisy_tierb_candidate_rows == accepted and accepted > 0
    explicit_exception = bool(args.explicit_allow_noisy_tierb_action_teacher)
    noisy_exception_recorded = explicit_exception and noisy_tierb_candidate_exists

    if strict_clean_ready:
        action_teacher_policy = "READY_STRICT_CLEAN_TEACHER"
    elif noisy_exception_recorded:
        action_teacher_policy = "RISK_EXCEPTION_RECORDED_NOT_DATA_BUILT"
    else:
        action_teacher_policy = "BLOCKED_DEFAULT_POLICY"

    result = {
        "artifact_type": "cube10cm_link5corner_noisy_tierb_teacher_policy_gate_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_policy_gate_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "statuses": {
            "event_label_metadata_manifest": "READY_LOCAL_ONLY" if event_label_ready else "BLOCKED",
            "strict_clean_action_teacher": "READY" if strict_clean_ready else "BLOCKED",
            "noisy_tierb_action_teacher_candidate": "CANDIDATE_EXISTS"
            if noisy_tierb_candidate_exists
            else "BLOCKED",
            "noisy_tierb_exception": "RISK_EXCEPTION_RECORDED_NOT_DATA_BUILT"
            if noisy_exception_recorded
            else "REQUIRES_EXPLICIT_USER_PROFESSOR_EXCEPTION",
            "default_action_teacher_dataset_policy": action_teacher_policy,
            "tiny_action_dataset_dryrun": "BLOCKED_UNLESS_EXCEPTION_RECORDED",
            "large_isaaclab_dataset": "BLOCKED",
            "isaaclab_rl": "BLOCKED",
            "roarm_m3_pro": "BLOCKED",
        },
        "evidence": {
            "accepted_windows": accepted,
            "event_count": event_count,
            "tier_a": tier_a,
            "tier_b": tier_b,
            "tier_c": tier_c,
            "clean_diffik_teacher_window_ready": clean_ready,
            "accepted_window_clip_any_rate_mean": clip_mean,
            "accepted_window_follow_p95_to_cap_p95": follow_p95,
            "weak_1mm_objective_accepted": weak_objective_accepted,
            "clean_tap_strength_visual_verified": clean_tap_visual_verified,
            "grazing_or_outside_face_supported": grazing_supported,
            "early_freeze_supported": early_freeze,
        },
        "policy_interpretation": {
            "event_labels_are_allowed_without_action_targets": event_label_ready,
            "tier_b_means_noisy_due_high_clip": True,
            "tier_b_is_not_clean_teacher": True,
            "default_noisy_action_teacher_allowed": False,
            "explicit_exception_requested": explicit_exception,
            "quality_policy_resolved_for_training": strict_clean_ready or noisy_exception_recorded,
            "exception_does_not_unblock_large_dataset_rl_or_robot": True,
        },
        "next_gate": {
            "if_no_exception": "keep_link5_as_event_label_metadata_only",
            "if_exception_later": "record_exception_then_only_a_tiny_audited_action_dataset_dryrun",
            "before_large_dataset": "dataset_schema_plus_teacher_policy_must_be_resolved",
            "before_rl": "validate_10cm_tap_rl_env_random_sanity",
            "before_robot": "validated_policy_plus_hardware_safety_replay",
        },
        "source_files": {
            "event_manifest_json": str(args.event_manifest_json),
            "acceptance_matrix_json": str(args.acceptance_matrix_json),
            "reaction_window_json": str(args.reaction_window_json),
            "visual_json": str(args.visual_json),
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_link5corner_noisy_tierb_teacher_policy_gate_v1 "
        "local_policy_gate_only=YES action_dataset_built=NO gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 event_label_metadata "
            f"status={result['statuses']['event_label_metadata_manifest']} "
            f"events={event_count} weak_1mm={counts.get('weak_1mm_count')} overshoot={counts.get('overshoot_count')}"
        ),
        (
            "line3 strict_clean_action_teacher "
            f"status={result['statuses']['strict_clean_action_teacher']} "
            f"clean_teacher={clean_ready} tierA={tier_a} tierB={tier_b} tierC={tier_c} "
            f"clip_mean={clip_mean:.9f} follow_p95_to_cap={follow_p95:.9f}"
        ),
        (
            "line4 noisy_tierb_candidate "
            f"status={result['statuses']['noisy_tierb_action_teacher_candidate']} "
            f"candidate_rows={noisy_tierb_candidate_rows} accepted_windows={accepted} "
            "reason=TierB_follow_ok_but_clip_high"
        ),
        (
            "line5 exception_policy "
            f"status={result['statuses']['noisy_tierb_exception']} "
            f"explicit_exception_requested={explicit_exception} "
            f"default_action_teacher_dataset_policy={action_teacher_policy}"
        ),
        (
            "line6 visual_risk "
            f"clean_tap_visual_verified={clean_tap_visual_verified} "
            f"grazing_or_outside_face_supported={grazing_supported} early_freeze_supported={early_freeze}"
        ),
        (
            "line7 next_gate "
            "no_exception=event_label_metadata_only "
            "exception_path=tiny_audited_action_dataset_dryrun_only "
            "large_dataset_rl_roarm=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
