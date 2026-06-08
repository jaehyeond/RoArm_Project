"""Build the link5-corner event-label and quality metadata manifest.

This manifest records the latest cube10cm link5-corner reaction-window evidence
as labels plus quality/proxy-contact metadata. It is not an action-teacher
dataset and does not generate training data. The script reads existing local
logs only; it does not run IsaacLab, use GPU, train, control a robot, SSH, or
mutate source traces.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
DEFAULT_MATRIX_JSON = LOG_DIR / "cube10cm_link5corner_acceptance_blocker_matrix.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_OUT_JSONL = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest_events.jsonl"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest_summary.out"


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
    parser.add_argument("--acceptance_matrix_json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_events_jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    summary = _load_json(args.summary_json)
    reaction = _load_json(args.reaction_gate_json)
    window = _load_json(args.reaction_window_json)
    visual = _load_json(args.visual_json)
    matrix = _load_json(args.acceptance_matrix_json)

    if not matrix.get("decision", {}).get("weak_1mm_tap_reaction_accepted_as_current_working_objective"):
        raise SystemExit("link5 event metadata manifest blocked: weak 1mm objective is not accepted")

    per_window = [row for row in window.get("per_window", []) if isinstance(row, dict) and row.get("accepted")]
    visual_by_env = {
        int(env.get("env_id")): env for env in visual.get("envs", []) if isinstance(env, dict) and "env_id" in env
    }
    tier_counts = Counter(str(row.get("quality_tier", "UNKNOWN")) for row in per_window)

    events: list[dict[str, Any]] = []
    for row in per_window:
        env_id = int(row.get("env_id", -1))
        detail = visual_by_env.get(env_id)
        if detail is None:
            raise SystemExit(f"missing visual detail for env_id={env_id}")
        contact = detail["contact"]
        maxdisp = detail["maxdisp"]
        max_disp = _float(row.get("max_disp_m"))
        proxy_gap = _float(contact.get("proxy_gap_to_live_side_face_m"))
        target_gap = _float(contact.get("target_gap_to_live_side_face_m"))
        event = {
            "event_id": f"link5corner_seed962_env{env_id}_window0",
            "source_runtime": "link5corner_position_seed962",
            "source_seed": int(summary.get("seed", 962)),
            "env_id": env_id,
            "reaction_window": {
                "anchor_step": int(row.get("anchor_step", -1)),
                "window_start_step": int(row.get("window_start_step", -1)),
                "window_end_step": int(row.get("window_end_step", -1)),
                "anchor_source": str(row.get("anchor_source", "")),
                "rows": int(row.get("rows", 0)),
            },
            "primary_event_labels": {
                "contact_evidence": _bool(row.get("contact_evidence")),
                "reaction_signal": _bool(row.get("reaction_signal")),
                "no_overshoot": not _bool(row.get("overshoot")),
                "weak_1mm_tap_reaction": max_disp >= 0.001,
                "max_transient_ge_2mm": max_disp >= 0.002,
                "max_transient_ge_3mm": max_disp >= 0.003,
            },
            "reaction_metrics": {
                "max_transient_disp_m": max_disp,
                "max_z_delta_m": _float(row.get("max_z_delta_m")),
                "max_tip_angle_deg": _float(row.get("max_tip_angle_deg")),
                "max_speed_mps": _float(row.get("max_speed_mps")),
                "summary_max_disp_along_push_mean_m": _float(summary.get("max_disp_along_push_mean_m")),
                "summary_max_cube_speed_mean_mps": _float(summary.get("max_cube_speed_mean_mps")),
            },
            "quality_tier_metadata": {
                "quality_tier": str(row.get("quality_tier", "UNKNOWN")),
                "clean_diffik_teacher": False,
                "action_teacher_usable_default_policy": False,
                "clip_any_rate": _float(row.get("clip_any_rate")),
                "joint_follow_p95_to_cap": _float(row.get("joint_follow_p95_to_cap")),
                "window_clip_any_rate_mean": _float(window.get("accepted_window_clip_any_rate_mean")),
                "window_follow_p95_to_cap_p95": _float(window.get("accepted_window_follow_p95_to_cap_p95")),
                "reaction_gate_teacher_quality_ready": _bool(reaction.get("teacher_quality_ready")),
                "summary_final_tcp_target_err_mean_m": _float(
                    reaction.get("summary_final_tcp_target_err_mean_m")
                ),
            },
            "proxy_contact_metadata": {
                "tool_contact_proxy_mode": str(summary.get("tool_contact_proxy_mode")),
                "tool_proxy_label": str(summary.get("tool_proxy_label")),
                "diffik_command_type": str(summary.get("command_type")),
                "side_center_proxy_visual_verified": _bool(
                    visual.get("verdict", {}).get("side_center_proxy_visual_verified")
                ),
                "top_contact_rejected_for_link5_proxy": _bool(
                    visual.get("verdict", {}).get("top_contact_rejected_for_link5_proxy")
                ),
                "grazing_or_outside_face_supported": _bool(
                    visual.get("verdict", {}).get("grazing_or_outside_face_supported")
                ),
                "early_freeze_supported": _bool(visual.get("verdict", {}).get("early_freeze_supported")),
                "clean_tap_strength_visual_verified": _bool(
                    visual.get("verdict", {}).get("clean_tap_strength_visual_verified")
                ),
                "proxy_target_err_m_at_contact": _float(contact.get("proxy_target_err_m")),
                "proxy_target_z_err_m_at_contact": _float(contact.get("proxy_target_z_err_m")),
                "proxy_minus_cube_center_z_m_at_contact": _float(
                    contact.get("proxy_minus_live_cube_center_z_m")
                ),
                "proxy_below_cube_top_m_at_contact": _float(contact.get("proxy_below_live_cube_top_m")),
                "proxy_gap_to_live_side_face_m_at_contact": proxy_gap,
                "target_gap_to_live_side_face_m_at_contact": target_gap,
                "proxy_outside_live_face_at_contact": proxy_gap < -0.001,
                "target_outside_live_face_at_contact": target_gap < -0.001,
                "contact_stop_same_as_contact_step": _bool(detail.get("contact_stop_same_rollout_step")),
                "maxdisp_proxy_gap_to_live_side_face_m": _float(
                    maxdisp.get("proxy_gap_to_live_side_face_m")
                ),
            },
        }
        events.append(event)

    counts = {
        "event_count": len(events),
        "contact_count": sum(1 for event in events if event["primary_event_labels"]["contact_evidence"]),
        "reaction_count": sum(1 for event in events if event["primary_event_labels"]["reaction_signal"]),
        "overshoot_count": sum(1 for event in events if not event["primary_event_labels"]["no_overshoot"]),
        "weak_1mm_count": sum(1 for event in events if event["primary_event_labels"]["weak_1mm_tap_reaction"]),
        "max_transient_ge_2mm_count": sum(
            1 for event in events if event["primary_event_labels"]["max_transient_ge_2mm"]
        ),
        "max_transient_ge_3mm_count": sum(
            1 for event in events if event["primary_event_labels"]["max_transient_ge_3mm"]
        ),
        "quality_tier_counts": dict(sorted(tier_counts.items())),
        "side_center_proxy_visual_count": sum(
            1 for event in events if event["proxy_contact_metadata"]["side_center_proxy_visual_verified"]
        ),
        "proxy_outside_live_face_count": sum(
            1 for event in events if event["proxy_contact_metadata"]["proxy_outside_live_face_at_contact"]
        ),
        "contact_stop_same_as_contact_count": sum(
            1 for event in events if event["proxy_contact_metadata"]["contact_stop_same_as_contact_step"]
        ),
        "action_teacher_usable_default_count": sum(
            1 for event in events if event["quality_tier_metadata"]["action_teacher_usable_default_policy"]
        ),
    }

    manifest = {
        "artifact_type": "cube10cm_link5corner_event_label_metadata_manifest_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_manifest_only": True,
        "not_action_teacher_dataset": True,
        "not_lerobot_or_rlds_dataset": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "working_objective": "weak_1mm_tap_reaction_contact_if_professor_user_accepts",
        "schema": {
            "identity": ["event_id", "source_runtime", "source_seed", "env_id", "reaction_window"],
            "primary_event_labels": [
                "contact_evidence",
                "reaction_signal",
                "no_overshoot",
                "weak_1mm_tap_reaction",
            ],
            "quality_tier_metadata": [
                "quality_tier",
                "clean_diffik_teacher",
                "action_teacher_usable_default_policy",
                "clip_any_rate",
                "joint_follow_p95_to_cap",
            ],
            "proxy_contact_metadata": [
                "side_center_proxy_visual_verified",
                "top_contact_rejected_for_link5_proxy",
                "proxy_gap_to_live_side_face_m_at_contact",
                "target_gap_to_live_side_face_m_at_contact",
                "contact_stop_same_as_contact_step",
            ],
            "explicitly_excluded": [
                "joint_delta_targets",
                "policy_actions",
                "image_observations",
                "final_1cm_relocation_primary_gate",
                "final_1mm_retention_primary_gate",
            ],
        },
        "counts": counts,
        "events": events,
        "policy_pointer": {
            "event_labels_ready_local_only": counts["event_count"] == counts["weak_1mm_count"] == 16,
            "default_action_teacher_dataset_allowed": False,
            "needs_separate_noisy_tierb_policy_gate": True,
            "large_isaaclab_dataset": "BLOCKED",
            "isaaclab_rl": "BLOCKED",
            "roarm_m3_pro": "BLOCKED",
        },
        "source_files": {
            "summary_json": str(args.summary_json),
            "reaction_gate_json": str(args.reaction_gate_json),
            "reaction_window_json": str(args.reaction_window_json),
            "visual_json": str(args.visual_json),
            "acceptance_matrix_json": str(args.acceptance_matrix_json),
        },
        "out_events_jsonl": str(args.out_events_jsonl),
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_events_jsonl.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    lines = [
        "line1 artifact=cube10cm_link5corner_event_label_metadata_manifest_v1 "
        "local_manifest_only=YES action_teacher_dataset=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 event_counts "
            f"events={counts['event_count']} contact={counts['contact_count']} "
            f"reaction={counts['reaction_count']} overshoot={counts['overshoot_count']} "
            f"weak_1mm={counts['weak_1mm_count']} ge_2mm={counts['max_transient_ge_2mm_count']} "
            f"ge_3mm={counts['max_transient_ge_3mm_count']}"
        ),
        f"line3 quality_tiers {counts['quality_tier_counts']} clean_teacher=False action_teacher_usable_default_count={counts['action_teacher_usable_default_count']}",
        (
            "line4 proxy_contact_metadata "
            f"side_center_proxy_visual_count={counts['side_center_proxy_visual_count']} "
            f"proxy_outside_live_face_count={counts['proxy_outside_live_face_count']} "
            f"contact_stop_same_as_contact_count={counts['contact_stop_same_as_contact_count']}"
        ),
        (
            "line5 policy "
            "event_labels_ready_local_only=True default_action_teacher_dataset_allowed=False "
            "needs_separate_noisy_tierb_policy_gate=True"
        ),
        "line6 next_gate large_isaaclab_dataset=BLOCKED isaaclab_rl=BLOCKED roarm_m3_pro=BLOCKED",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
