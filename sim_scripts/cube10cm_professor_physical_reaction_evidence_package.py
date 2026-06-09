"""Package professor-facing 10cm cube weak physical-reaction evidence.

This is a local packaging step over existing logs and visual artifacts only. It
does not launch IsaacLab/GPU runtime, generate datasets, train, control RoArm,
SSH, pull, or touch B200. The package intentionally excludes action payloads and
must not be interpreted as an action-teacher dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_DIRECT_AUDIT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit.json"
DEFAULT_DIRECT_AUDIT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit_summary.out"
DEFAULT_DIRECT_RUNTIME_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_sanity.json"
DEFAULT_DIRECT_RUNTIME_SUMMARY = (
    LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_sanity_summary.out"
)
DEFAULT_PREFLIGHT_JSON = LOG_DIR / "cube10cm_tap_rl_preflight_policy_gate.json"
DEFAULT_PREFLIGHT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_preflight_policy_gate_summary.out"
DEFAULT_EVENT_METADATA_JSON = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_EVENT_METADATA_SUMMARY = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest_summary.out"
DEFAULT_TEACHER_POLICY_JSON = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate.json"
DEFAULT_TEACHER_POLICY_SUMMARY = LOG_DIR / "cube10cm_link5corner_noisy_tierb_teacher_policy_gate_summary.out"
DEFAULT_VISUAL_HTML = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.html"
DEFAULT_VISUAL_SVG = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.svg"
DEFAULT_VISUAL_PNG = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.png"

DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_professor_physical_reaction_evidence_package.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_professor_physical_reaction_evidence_package_summary.out"
DEFAULT_OUT_MD = LOG_DIR / "cube10cm_professor_physical_reaction_evidence_package.md"

FORBIDDEN_ACTION_PAYLOAD_KEYS = {
    "action",
    "actions",
    "policy_actions",
    "normalized_actions",
    "joint_delta_targets",
    "joint_position_targets",
    "robot_dof_targets",
    "image_observations",
    "observations",
    "action_abs_mean",
    "action_abs_max",
    "action_abs_max_trace",
    "joint_delta_abs_mean",
    "joint_delta_abs_max",
    "joint_delta_abs_max_trace",
    "target_lead_abs_max",
    "target_lead_abs_max_trace",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(role: str, path: Path, summary_lines: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        item["sha256"] = _sha256(path)
        item["bytes"] = path.stat().st_size
    if summary_lines:
        item["summary_lines"] = summary_lines
    return item


def _gate_status(preflight: dict[str, Any], gate_name: str, default: str = "UNKNOWN") -> str:
    for gate in preflight.get("gate_matrix", []):
        if isinstance(gate, dict) and gate.get("gate") == gate_name:
            return str(gate.get("status", default))
    return default


def _gate_evidence(preflight: dict[str, Any], gate_name: str) -> dict[str, Any]:
    for gate in preflight.get("gate_matrix", []):
        if isinstance(gate, dict) and gate.get("gate") == gate_name:
            evidence = gate.get("evidence", {})
            return evidence if isinstance(evidence, dict) else {}
    return {}


def _require_no_forbidden_payload_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_ACTION_PAYLOAD_KEYS:
                dotted = ".".join((*path, key_text))
                raise SystemExit(f"forbidden action payload key in package: {dotted}")
            _require_no_forbidden_payload_keys(child, (*path, key_text))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_no_forbidden_payload_keys(child, (*path, str(index)))


def _write_markdown(package: dict[str, Any], out_md: Path) -> None:
    evidence = package["professor_physical_reaction_evidence"]
    blockers = package["blocked_gates"]
    metadata = package["event_label_quality_tier_metadata"]
    caveats = package["caveats"]
    fork = package["next_branch_decision"]

    lines = [
        "# Professor 10cm Cube Physical-Reaction Evidence Package",
        "",
        "## Scope",
        "",
        "- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier.",
        "- Local package over existing logs and visual audits only.",
        "- This is evidence of weak physical object reaction, not action-teacher, dataset, RL, or RoArm readiness.",
        "",
        "## Evidence",
        "",
        f"- Status: `{package['status']}`.",
        f"- Direct-IK professor physical evidence: `{evidence['direct_ik_professor_physical_reaction_evidence']}`.",
        f"- Max displacement along push: `{evidence['max_disp_along_m']:.9f}m`.",
        f"- Max speed: `{evidence['max_speed_mps']:.9f}m/s`.",
        f"- Overshoot: `{evidence['overshoot']}`.",
        f"- Contact/tap success remains `{evidence['contact_seen']}` / `{evidence['tap_success']}`.",
        "",
        "## Metadata Link",
        "",
        f"- Event-label metadata: `{metadata['event_label_metadata_status']}`.",
        f"- Events/contact/reaction/overshoot: `{metadata['events']}` / `{metadata['contact']}` / `{metadata['reaction']}` / `{metadata['overshoot']}`.",
        f"- Quality tiers: `{metadata['quality_tier_counts']}`.",
        "- Metadata is label/quality-tier only; no action payloads are included.",
        "",
        "## Caveats",
        "",
        f"- clean_tap_visual_verified=`{str(caveats['clean_tap_visual_verified']).lower()}`.",
        f"- grazing_or_outside_face_behavior=`{str(caveats['grazing_or_outside_face_behavior']).lower()}`.",
        f"- contact_gated_rl_success=`{caveats['contact_gated_rl_success']}`.",
        f"- action_teacher=`{blockers['diffik_action_dataset']}`.",
        f"- dataset/RL/RoArm=`{blockers['large_dataset']}` / `{blockers['ppo_rl_training']}` / `{blockers['roarm']}`.",
        "",
        "## Pipeline Position",
        "",
        f"- P0 evidence checkpoint: {fork['p0_evidence_checkpoint']}.",
        f"- P1/P2 learning and RL path: {fork['p1_p2_learning_rl_path']}.",
        "",
        "## Line Evidence",
        "",
        "- Direct audit summary: line 4 for weak reaction PASS metrics, line 8 for RL/dataset/RoArm blockers.",
        "- Preflight summary: line 3 for READY_PROFESSOR_EVIDENCE_ONLY, line 6 for RL/RoArm blockers.",
        "- Event-label metadata summary: lines 1-6 for metadata-only counts and blocked downstream gates.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_audit_json", type=Path, default=DEFAULT_DIRECT_AUDIT_JSON)
    parser.add_argument("--direct_audit_summary", type=Path, default=DEFAULT_DIRECT_AUDIT_SUMMARY)
    parser.add_argument("--direct_runtime_json", type=Path, default=DEFAULT_DIRECT_RUNTIME_JSON)
    parser.add_argument("--direct_runtime_summary", type=Path, default=DEFAULT_DIRECT_RUNTIME_SUMMARY)
    parser.add_argument("--preflight_json", type=Path, default=DEFAULT_PREFLIGHT_JSON)
    parser.add_argument("--preflight_summary", type=Path, default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--event_metadata_json", type=Path, default=DEFAULT_EVENT_METADATA_JSON)
    parser.add_argument("--event_metadata_summary", type=Path, default=DEFAULT_EVENT_METADATA_SUMMARY)
    parser.add_argument("--teacher_policy_json", type=Path, default=DEFAULT_TEACHER_POLICY_JSON)
    parser.add_argument("--teacher_policy_summary", type=Path, default=DEFAULT_TEACHER_POLICY_SUMMARY)
    parser.add_argument("--visual_html", type=Path, default=DEFAULT_VISUAL_HTML)
    parser.add_argument("--visual_svg", type=Path, default=DEFAULT_VISUAL_SVG)
    parser.add_argument("--visual_png", type=Path, default=DEFAULT_VISUAL_PNG)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out_md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    direct_audit = _load_json(args.direct_audit_json)
    preflight = _load_json(args.preflight_json)
    event_metadata = _load_json(args.event_metadata_json)
    teacher_policy = _load_json(args.teacher_policy_json)

    direct_professor_status = str(direct_audit.get("professor_physical_reaction_evidence", "UNKNOWN"))
    preflight_professor_status = _gate_status(preflight, "professor_physical_reaction_evidence")
    rl_positive_status = _gate_status(preflight, "positive_control_tap_sanity_in_new_wrapper")
    metadata_status = _gate_status(preflight, "event_label_quality_tier_metadata")
    event_counts = event_metadata.get("counts", {})
    teacher_statuses = teacher_policy.get("statuses", {})
    teacher_evidence = teacher_policy.get("evidence", {})

    package = {
        "artifact_type": "cube10cm_professor_physical_reaction_evidence_package_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "status": (
            "READY_PROFESSOR_EVIDENCE_ONLY"
            if direct_professor_status == "PASS" and preflight_professor_status == "READY_PROFESSOR_EVIDENCE_ONLY"
            else "BLOCKED_OR_INCONSISTENT"
        ),
        "package_intent": (
            "Freeze reproducible local evidence that a 10cm/0.72kg cube shows weak physical reaction; "
            "do not promote this to action-teacher, dataset, RL, or RoArm readiness."
        ),
        "local_only_contract": {
            "uses_existing_logs_only": True,
            "gpu_runtime_launched_by_this_package": False,
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
            "b200": False,
            "track_a": False,
        },
        "professor_physical_reaction_evidence": {
            "preflight_status": preflight_professor_status,
            "direct_ik_professor_physical_reaction_evidence": direct_professor_status,
            "professor_physical_reaction_evidence_only": bool(
                direct_audit.get("professor_physical_reaction_evidence_only", False)
            ),
            "max_disp_along_m": float(direct_audit.get("max_disp_along_m", 0.0)),
            "max_speed_mps": float(direct_audit.get("max_speed_mps", 0.0)),
            "overshoot": float(direct_audit.get("overshoot", 1.0)),
            "contact_seen": float(direct_audit.get("contact_seen", 0.0)),
            "reaction_context": float(direct_audit.get("reaction_context", 0.0)),
            "reaction_seen": float(direct_audit.get("reaction_seen", 0.0)),
            "tap_success": float(direct_audit.get("tap_success", 0.0)),
            "thresholds": {
                "disp_m": float(direct_audit.get("professor_physical_disp_evidence_threshold_m", 0.0005)),
                "speed_mps": float(
                    direct_audit.get("professor_physical_speed_evidence_threshold_mps", 0.005)
                ),
            },
            "verdict": str(direct_audit.get("verdict", "UNKNOWN")),
        },
        "visual_evidence_package": {
            "html": str(args.visual_html),
            "svg": str(args.visual_svg),
            "png": str(args.visual_png),
            "along_gap_blocker": bool(direct_audit.get("along_gap_blocker", True)),
            "lateral_ok": bool(direct_audit.get("lateral_ok", False)),
            "vertical_ok": bool(direct_audit.get("vertical_ok", False)),
            "face_gap_near_band": bool(direct_audit.get("face_gap_near_band", False)),
            "best_shortfall_to_contact_band_m": float(
                direct_audit.get("best_shortfall_to_contact_band_m", 0.0)
            ),
            "final_shortfall_to_contact_band_m": float(
                direct_audit.get("final_shortfall_to_contact_band_m", 0.0)
            ),
        },
        "event_label_quality_tier_metadata": {
            "linked_to_professor_evidence": True,
            "event_label_metadata_status": metadata_status,
            "local_manifest_only": bool(event_metadata.get("local_manifest_only", False)),
            "not_action_teacher_dataset": bool(event_metadata.get("not_action_teacher_dataset", True)),
            "not_lerobot_or_rlds_dataset": bool(event_metadata.get("not_lerobot_or_rlds_dataset", True)),
            "events": int(event_counts.get("event_count", 0)),
            "contact": int(event_counts.get("contact_count", 0)),
            "reaction": int(event_counts.get("reaction_count", 0)),
            "overshoot": int(event_counts.get("overshoot_count", -1)),
            "weak_1mm": int(event_counts.get("weak_1mm_count", 0)),
            "ge_2mm": int(event_counts.get("max_transient_ge_2mm_count", 0)),
            "ge_3mm": int(event_counts.get("max_transient_ge_3mm_count", 0)),
            "quality_tier_counts": event_counts.get("quality_tier_counts", {}),
            "clean_teacher": bool(teacher_evidence.get("clean_diffik_teacher_window_ready", False)),
            "default_action_teacher_dataset_allowed": False,
        },
        "caveats": {
            "clean_tap_visual_verified": False,
            "grazing_or_outside_face_behavior": True,
            "contact_gated_rl_success": "FAIL",
            "strict_clean_action_teacher": "BLOCKED",
            "noisy_tier_b_exception": str(
                teacher_statuses.get("noisy_tier_b_action_teacher_exception", "REQUIRES_EXPLICIT_USER_PROFESSOR_EXCEPTION")
            ),
            "dataset_rl_roarm_readiness": "BLOCKED",
            "weak_physical_evidence_must_not_be_promoted": True,
        },
        "blocked_gates": {
            "rl_contact_gated_positive_control": rl_positive_status,
            "diffik_action_dataset": str(
                direct_audit.get("still_blocked", {}).get("diffik_action_dataset", "BLOCKED")
            ),
            "tiny_action_dataset_dry_run": str(
                direct_audit.get("still_blocked", {}).get("tiny_action_dataset_dry_run", "BLOCKED")
            ),
            "ppo_rl_training": str(direct_audit.get("still_blocked", {}).get("ppo_rl_training", "BLOCKED")),
            "large_dataset": str(direct_audit.get("still_blocked", {}).get("large_dataset", "BLOCKED")),
            "roarm": str(direct_audit.get("still_blocked", {}).get("roarm", "BLOCKED")),
        },
        "schema_policy": {
            "actual_action_payload_included": False,
            "copied_action_payload_keys": [],
            "allowed_content": [
                "source paths and checksums",
                "event label counts",
                "quality tier counts",
                "weak physical reaction metrics",
                "visual caveats",
                "blocked readiness statuses",
            ],
            "excluded_action_payload_key_families": sorted(FORBIDDEN_ACTION_PAYLOAD_KEYS),
        },
        "next_branch_decision": {
            "p0_evidence_checkpoint": (
                "this package is one checkpoint inside the integrated professor-report pipeline, "
                "not an alternative to learning/RL"
            ),
            "p1_p2_learning_rl_path": (
                "continue by resolving the local RL/learning blockers: contact-gated positive-control, "
                "clean teacher or explicit noisy Tier-B exception gate, tiny dry run, large dataset, "
                "PPO/RL training, and only then RoArm/generalization"
            ),
        },
        "sources": [
            _source("direct_ik_result_audit_json", args.direct_audit_json),
            _source("direct_ik_result_audit_summary", args.direct_audit_summary, "4-8"),
            _source("direct_ik_runtime_json", args.direct_runtime_json),
            _source("direct_ik_runtime_summary", args.direct_runtime_summary, "1-9"),
            _source("preflight_policy_gate_json", args.preflight_json),
            _source("preflight_policy_gate_summary", args.preflight_summary, "1-7"),
            _source("event_label_quality_tier_metadata_json", args.event_metadata_json),
            _source("event_label_quality_tier_metadata_summary", args.event_metadata_summary, "1-6"),
            _source("teacher_policy_gate_json", args.teacher_policy_json),
            _source("teacher_policy_gate_summary", args.teacher_policy_summary, "1-7"),
            _source("visual_contact_audit_html", args.visual_html),
            _source("visual_contact_audit_svg", args.visual_svg),
            _source("visual_contact_audit_png", args.visual_png),
        ],
        "outputs": {
            "json": str(args.out_json),
            "summary": str(args.out_summary),
            "markdown": str(args.out_md),
        },
    }

    _require_no_forbidden_payload_keys(package)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(package, args.out_md)

    evidence = package["professor_physical_reaction_evidence"]
    metadata = package["event_label_quality_tier_metadata"]
    blockers = package["blocked_gates"]
    caveats = package["caveats"]
    summary_lines = [
        "line1 artifact=cube10cm_professor_physical_reaction_evidence_package_v1 "
        "local_existing_logs_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 professor_evidence "
            f"status={package['status']} direct_ik_professor_physical_reaction_evidence="
            f"{evidence['direct_ik_professor_physical_reaction_evidence']} "
            f"max_disp_along_m={evidence['max_disp_along_m']:.9f} "
            f"max_speed_mps={evidence['max_speed_mps']:.9f} overshoot={evidence['overshoot']:.1f}"
        ),
        (
            "line3 rl_contact_gate "
            f"rl_contact_gated_positive_control={blockers['rl_contact_gated_positive_control']} "
            f"contact_seen={evidence['contact_seen']:.1f} reaction_context={evidence['reaction_context']:.1f} "
            f"reaction_seen={evidence['reaction_seen']:.1f} tap_success={evidence['tap_success']:.1f}"
        ),
        (
            "line4 metadata_link "
            f"event_label_metadata={metadata['event_label_metadata_status']} events={metadata['events']} "
            f"contact={metadata['contact']} reaction={metadata['reaction']} overshoot={metadata['overshoot']} "
            f"quality_tiers={metadata['quality_tier_counts']}"
        ),
        (
            "line5 caveats "
            f"clean_tap_visual_verified={caveats['clean_tap_visual_verified']} "
            f"grazing_or_outside_face_behavior={caveats['grazing_or_outside_face_behavior']} "
            f"contact_gated_rl_success={caveats['contact_gated_rl_success']} "
            "action_teacher=BLOCKED dataset_rl_roarm=BLOCKED"
        ),
        (
            "line6 schema "
            "actual_action_payload_included=False copied_action_payload_keys=0 "
            "event_labels_and_quality_tiers_only=YES"
        ),
        (
            "line7 blockers "
            f"diffik_action_dataset={blockers['diffik_action_dataset']} "
            f"tiny_action_dataset_dry_run={blockers['tiny_action_dataset_dry_run']} "
            f"ppo_rl_training={blockers['ppo_rl_training']} large_dataset={blockers['large_dataset']} "
            f"roarm={blockers['roarm']}"
        ),
        (
            "line8 next "
            "p0_evidence_checkpoint=done_inside_integrated_professor_pipeline "
            "p1_p2_learning_rl_path=resolve_local_contact_teacher_dataset_training_gates_in_order"
        ),
        f"line9 outputs json={args.out_json} summary={args.out_summary} markdown={args.out_md}",
    ]
    args.out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
