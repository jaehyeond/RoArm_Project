"""Local preflight for a cube10cm tap/reaction event-label dataset builder.

This script turns the existing seed962 event-label manifest into a tiny preview
artifact and validates the builder contract before any real dataset generation.
It intentionally produces no action-teacher dataset, no LeRobot/RLDS dataset, no
training data, no IsaacLab/GPU runtime, no robot control, no SSH, and no trace
mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_MANIFEST = LOG_DIR / "cube10cm_event_label_dataset_manifest.json"
DEFAULT_BLOCKER = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_reaction_dataset_builder_preflight.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_reaction_dataset_builder_preflight_summary.out"
DEFAULT_OUT_PREVIEW = LOG_DIR / "cube10cm_tap_reaction_dataset_builder_preflight_preview.jsonl"

REQUIRED_EVENT_FIELDS = (
    "event_id",
    "source_seed",
    "env_id",
    "reaction_window_anchor_step",
    "reaction_window_start_step",
    "reaction_window_end_step",
    "contact_evidence",
    "reaction_signal",
    "overshoot",
    "max_transient_disp_m",
    "max_transient_ge_1mm",
    "max_transient_ge_2mm",
    "max_transient_ge_3mm",
    "quality_tier",
    "joint_follow_p95_to_cap",
    "clip_any_rate",
)

FORBIDDEN_PRIMARY_GATE_FIELDS = (
    "final_1cm_relocation",
    "final_1mm_retention",
    "post_push_final_position",
    "success_marker",
    "controlled_push",
    "low_motion",
    "final_disp_m",
    "final_relocation",
    "target_xy_dist",
    "cube_success_disp_m",
)


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


def _quality_group(tier: str) -> str:
    if tier.startswith("A_"):
        return "clean_teacher"
    if tier.startswith("B_"):
        return "reaction_valid_follow_ok_clip_high"
    if tier.startswith("C_"):
        return "reaction_valid_follow_lag"
    return "unknown_or_rejected"


def _preview_row(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "cube10cm_tap_reaction_event_label_v1",
        "identity": {
            "event_id": str(event["event_id"]),
            "source_seed": int(event["source_seed"]),
            "env_id": int(event["env_id"]),
            "anchor_step": int(event["reaction_window_anchor_step"]),
            "window_start_step": int(event["reaction_window_start_step"]),
            "window_end_step": int(event["reaction_window_end_step"]),
        },
        "primary_event_labels": {
            "contact_evidence": _bool(event["contact_evidence"]),
            "reaction_signal": _bool(event["reaction_signal"]),
            "no_overshoot": not _bool(event["overshoot"]),
            "max_transient_ge_1mm": _bool(event["max_transient_ge_1mm"]),
            "max_transient_ge_2mm": _bool(event["max_transient_ge_2mm"]),
            "max_transient_ge_3mm": _bool(event["max_transient_ge_3mm"]),
        },
        "reaction_metrics": {
            "max_transient_disp_m": _float(event["max_transient_disp_m"]),
            "max_z_delta_m": _float(event.get("max_z_delta_m")),
            "max_tip_angle_deg": _float(event.get("max_tip_angle_deg")),
            "max_speed_mps": _float(event.get("max_speed_mps")),
        },
        "quality_metadata": {
            "quality_tier": str(event["quality_tier"]),
            "quality_group": _quality_group(str(event["quality_tier"])),
            "joint_follow_p95_to_cap": _float(event["joint_follow_p95_to_cap"]),
            "clip_any_rate": _float(event["clip_any_rate"]),
            "action_teacher_usable": False,
        },
    }


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            full = f"{prefix}.{key_s}" if prefix else key_s
            keys.add(key_s)
            keys.add(full)
            keys.update(_flatten_keys(child, full))
    elif isinstance(value, list):
        for child in value:
            keys.update(_flatten_keys(child, prefix))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_manifest_json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--blocker_audit_json", type=Path, default=DEFAULT_BLOCKER)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out_preview_jsonl", type=Path, default=DEFAULT_OUT_PREVIEW)
    args = parser.parse_args()

    manifest = _load_json(args.event_manifest_json)
    blocker = _load_json(args.blocker_audit_json)

    events = [row for row in manifest.get("events", []) if isinstance(row, dict)]
    missing_by_event = {
        str(row.get("event_id", f"event_{idx}")): [field for field in REQUIRED_EVENT_FIELDS if field not in row]
        for idx, row in enumerate(events)
    }
    missing_by_event = {key: value for key, value in missing_by_event.items() if value}

    preview_rows = [_preview_row(row) for row in events if str(row.get("event_id", "")) not in missing_by_event]
    preview_keys = _flatten_keys(preview_rows)
    forbidden_present = sorted(set(FORBIDDEN_PRIMARY_GATE_FIELDS).intersection(preview_keys))

    contact_count = sum(1 for row in preview_rows if row["primary_event_labels"]["contact_evidence"])
    reaction_count = sum(1 for row in preview_rows if row["primary_event_labels"]["reaction_signal"])
    overshoot_count = sum(1 for row in preview_rows if not row["primary_event_labels"]["no_overshoot"])
    ge_1mm_count = sum(1 for row in preview_rows if row["primary_event_labels"]["max_transient_ge_1mm"])
    ge_2mm_count = sum(1 for row in preview_rows if row["primary_event_labels"]["max_transient_ge_2mm"])
    ge_3mm_count = sum(1 for row in preview_rows if row["primary_event_labels"]["max_transient_ge_3mm"])
    tier_counts: dict[str, int] = {}
    for row in preview_rows:
        tier = row["quality_metadata"]["quality_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    manifest_ready = (
        bool(manifest.get("local_manifest_only"))
        and bool(manifest.get("not_action_teacher_dataset"))
        and bool(manifest.get("not_lerobot_or_rlds_dataset"))
        and len(events) > 0
    )
    event_label_rows_ready = (
        manifest_ready
        and not missing_by_event
        and not forbidden_present
        and len(preview_rows) == len(events)
        and contact_count == len(preview_rows)
        and reaction_count == len(preview_rows)
        and overshoot_count == 0
        and ge_1mm_count == len(preview_rows)
    )
    action_teacher_blocked = (
        blocker.get("statuses", {}).get("differential_ik_action_teacher_dataset") == "BLOCKED"
    )

    result = {
        "artifact_type": "cube10cm_tap_reaction_dataset_builder_preflight_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_preflight_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "input_manifest": str(args.event_manifest_json),
        "input_blocker_audit": str(args.blocker_audit_json),
        "statuses": {
            "tap_reaction_event_label_builder_preflight": "READY_LOCAL_ONLY"
            if event_label_rows_ready
            else "BLOCKED",
            "legacy_final_success_filter_bypassed_by_new_preflight": event_label_rows_ready,
            "differential_ik_action_teacher_dataset": "REMAINS_BLOCKED"
            if action_teacher_blocked
            else "RECHECK_REQUIRED",
            "large_isaaclab_dataset": "BLOCKED",
            "isaaclab_rl": "BLOCKED",
            "roarm_m3_pro": "BLOCKED",
        },
        "contract": {
            "primary_objective": "reaction_contact_no_posewrite_no_overshoot",
            "allowed_rows": "reaction_window_event_labels_with_quality_metadata",
            "not_action_teacher_dataset": True,
            "not_training_data": True,
            "forbidden_primary_gate_fields": list(FORBIDDEN_PRIMARY_GATE_FIELDS),
            "required_event_fields": list(REQUIRED_EVENT_FIELDS),
        },
        "counts": {
            "input_events": len(events),
            "preview_rows": len(preview_rows),
            "contact": contact_count,
            "reaction": reaction_count,
            "overshoot": overshoot_count,
            "max_transient_ge_1mm": ge_1mm_count,
            "max_transient_ge_2mm": ge_2mm_count,
            "max_transient_ge_3mm": ge_3mm_count,
            "quality_tier_counts": dict(sorted(tier_counts.items())),
        },
        "validation": {
            "manifest_ready": manifest_ready,
            "missing_required_fields_by_event": missing_by_event,
            "forbidden_primary_gate_fields_present_in_preview": forbidden_present,
            "preview_excludes_final_relocation_and_retention": not forbidden_present,
            "all_preview_rows_reaction_contact_no_overshoot_1mm": event_label_rows_ready,
        },
        "next_gate": {
            "action_teacher_dataset": "BLOCKED_BY_DIFFIK_TEACHER_QUALITY_POLICY",
            "quality_blockers_from_previous_audit": blocker.get("quality_evidence", {}),
            "rl_env": "BLOCKED_UNTIL_10CM_TAP_ENV_RANDOM_SANITY",
            "robot": "BLOCKED_UNTIL_VALIDATED_POLICY_AND_SAFETY_REPLAY",
        },
        "out_preview_jsonl": str(args.out_preview_jsonl),
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_preview_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in preview_rows),
        encoding="utf-8",
    )

    status = result["statuses"]
    lines = [
        "line1 artifact=cube10cm_tap_reaction_dataset_builder_preflight_v1 "
        "local_preflight_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO",
        (
            "line2 input_manifest "
            f"manifest_ready={manifest_ready} events={len(events)} preview_rows={len(preview_rows)} "
            f"contact={contact_count} reaction={reaction_count} overshoot={overshoot_count}"
        ),
        (
            "line3 transient_labels "
            f"ge_1mm={ge_1mm_count} ge_2mm={ge_2mm_count} ge_3mm={ge_3mm_count} "
            f"tiers={dict(sorted(tier_counts.items()))}"
        ),
        (
            "line4 forbidden_gate_check "
            f"pass={not forbidden_present} forbidden_present={forbidden_present} "
            "uses_final_success_filter=NO uses_final_1cm_or_retention=NO"
        ),
        (
            "line5 preflight_status "
            f"tap_reaction_event_label_builder={status['tap_reaction_event_label_builder_preflight']} "
            f"legacy_final_success_filter_bypassed={status['legacy_final_success_filter_bypassed_by_new_preflight']}"
        ),
        (
            "line6 remaining_blocks "
            f"diffik_action_teacher_dataset={status['differential_ik_action_teacher_dataset']} "
            f"large_isaaclab_dataset={status['large_isaaclab_dataset']} "
            f"isaaclab_rl={status['isaaclab_rl']} roarm_m3_pro={status['roarm_m3_pro']}"
        ),
        (
            "line7 next_unblock "
            "diffik_teacher_quality_policy_gate_before_action_dataset "
            "then_10cm_tap_rl_env_random_sanity_before_training"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
