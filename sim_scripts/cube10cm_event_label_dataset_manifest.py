"""Build a local event-label manifest for cube10cm tap/reaction windows.

This creates a schema/manifest for reaction-window event labels from existing
seed962 artifacts. It is not an action-teacher dataset, not a LeRobot/RLDS
dataset, and not training data. It performs no IsaacLab/GPU runtime, large
dataset generation, training, robot control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_WINDOW_AUDIT = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
DEFAULT_TRANSIENT_AUDIT = LOG_DIR / "cube10cm_yplus_transient_tap_strength_audit.json"
DEFAULT_READINESS_AUDIT = LOG_DIR / "cube10cm_dataset_rl_robot_readiness_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_event_label_dataset_manifest.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_event_label_dataset_manifest_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument("--transient_audit_json", type=Path, default=DEFAULT_TRANSIENT_AUDIT)
    parser.add_argument("--readiness_audit_json", type=Path, default=DEFAULT_READINESS_AUDIT)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    window = _load_json(args.reaction_window_json)
    transient = _load_json(args.transient_audit_json)
    readiness = _load_json(args.readiness_audit_json)
    if not readiness.get("gates", {}).get("event_label_dataset_ready", False):
        raise SystemExit("event label manifest blocked: readiness gate is false")

    per_window = [row for row in window.get("per_window", []) if isinstance(row, dict)]
    accepted = [row for row in per_window if bool(row.get("accepted", False))]
    tier_counts = Counter(str(row.get("quality_tier", "UNKNOWN")) for row in accepted)
    rows = []
    for row in accepted:
        max_disp = _float(row.get("max_disp_m"))
        rows.append(
            {
                "event_id": f"seed962_env{int(row.get('env_id', -1))}_window{int(row.get('local_index', 0))}",
                "source_seed": 962,
                "env_id": int(row.get("env_id", -1)),
                "reaction_window_anchor_step": int(row.get("anchor_step", -1)),
                "reaction_window_start_step": int(row.get("window_start_step", -1)),
                "reaction_window_end_step": int(row.get("window_end_step", -1)),
                "contact_evidence": bool(row.get("contact_evidence", False)),
                "reaction_signal": bool(row.get("reaction_signal", False)),
                "overshoot": bool(row.get("overshoot", False)),
                "max_transient_disp_m": max_disp,
                "max_transient_ge_1mm": max_disp >= 0.001,
                "max_transient_ge_2mm": max_disp >= 0.002,
                "max_transient_ge_3mm": max_disp >= 0.003,
                "max_z_delta_m": _float(row.get("max_z_delta_m")),
                "max_tip_angle_deg": _float(row.get("max_tip_angle_deg")),
                "max_speed_mps": _float(row.get("max_speed_mps")),
                "quality_tier": str(row.get("quality_tier", "UNKNOWN")),
                "joint_follow_p95_to_cap": _float(row.get("joint_follow_p95_to_cap")),
                "clip_any_rate": _float(row.get("clip_any_rate")),
            }
        )

    schema = {
        "identity": [
            "event_id",
            "source_seed",
            "env_id",
            "reaction_window_anchor_step",
            "reaction_window_start_step",
            "reaction_window_end_step",
        ],
        "primary_event_labels": [
            "contact_evidence",
            "reaction_signal",
            "overshoot",
            "max_transient_ge_1mm",
            "max_transient_ge_2mm",
            "max_transient_ge_3mm",
        ],
        "reaction_metrics": [
            "max_transient_disp_m",
            "max_z_delta_m",
            "max_tip_angle_deg",
            "max_speed_mps",
        ],
        "quality_metadata": [
            "quality_tier",
            "joint_follow_p95_to_cap",
            "clip_any_rate",
        ],
        "explicitly_not_in_primary_gate": [
            "final_1cm_relocation",
            "final_1mm_retention",
            "post_push_final_position",
        ],
        "not_action_dataset_columns": [
            "joint_delta_targets",
            "image_observations",
            "policy_actions",
        ],
    }

    manifest = {
        "artifact_type": "cube10cm_event_label_dataset_manifest_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_manifest_only": True,
        "not_action_teacher_dataset": True,
        "not_lerobot_or_rlds_dataset": True,
        "no_gpu_isaaclab_large_dataset_training_robot_ssh": True,
        "source_files": {
            "reaction_window_json": str(args.reaction_window_json),
            "transient_audit_json": str(args.transient_audit_json),
            "readiness_audit_json": str(args.readiness_audit_json),
        },
        "schema": schema,
        "events": rows,
        "counts": {
            "event_count": len(rows),
            "contact_evidence_count": sum(1 for row in rows if row["contact_evidence"]),
            "reaction_signal_count": sum(1 for row in rows if row["reaction_signal"]),
            "overshoot_count": sum(1 for row in rows if row["overshoot"]),
            "max_transient_ge_1mm_count": sum(1 for row in rows if row["max_transient_ge_1mm"]),
            "max_transient_ge_2mm_count": sum(1 for row in rows if row["max_transient_ge_2mm"]),
            "max_transient_ge_3mm_count": sum(1 for row in rows if row["max_transient_ge_3mm"]),
            "quality_tier_counts": dict(sorted(tier_counts.items())),
        },
        "next_gate": {
            "action_teacher_dataset": "BLOCKED_BY_QUALITY_TIER",
            "large_isaaclab_dataset": "BLOCKED_BY_QUALITY_TIER_AND_NO_10CM_DATASET_BUILDER",
            "isaaclab_rl": "BLOCKED_BY_NO_VALIDATED_10CM_RL_ENV_RANDOM_SANITY",
            "roarm_m3_pro": "BLOCKED_BY_NO_VALIDATED_POLICY_OR_HARDWARE_SAFETY_GATE",
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = manifest["counts"]
    lines = [
        "line1 artifact=cube10cm_event_label_dataset_manifest_v1 local_manifest_only=YES "
        "action_teacher_dataset=NO lerobot_rlds=NO training=NO robot_control=NO",
        (
            "line2 event_counts "
            f"events={counts['event_count']} contact={counts['contact_evidence_count']} "
            f"reaction={counts['reaction_signal_count']} overshoot={counts['overshoot_count']}"
        ),
        (
            "line3 transient_counts "
            f"ge_1mm={counts['max_transient_ge_1mm_count']} "
            f"ge_2mm={counts['max_transient_ge_2mm_count']} "
            f"ge_3mm={counts['max_transient_ge_3mm_count']}"
        ),
        f"line4 quality_tier_counts={counts['quality_tier_counts']}",
        "line5 schema_excludes=final_1cm_relocation,final_1mm_retention,post_push_final_position",
        (
            "line6 next_gate "
            "action_teacher_dataset=BLOCKED large_isaaclab_dataset=BLOCKED "
            "isaaclab_rl=BLOCKED roarm_m3_pro=BLOCKED"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
