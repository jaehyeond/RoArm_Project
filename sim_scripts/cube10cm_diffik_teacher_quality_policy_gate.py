"""Local policy gate for cube10cm DiffIK teacher quality.

This audit decides which dataset path is allowed after the tap/reaction
event-label builder preflight. It does not generate an action dataset, run
IsaacLab/GPU, train, control a robot, use SSH, or mutate traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_PREFLIGHT = LOG_DIR / "cube10cm_tap_reaction_dataset_builder_preflight.json"
DEFAULT_BLOCKER = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit.json"
DEFAULT_WINDOW_AUDIT = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_diffik_teacher_quality_policy_gate.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_diffik_teacher_quality_policy_gate_summary.out"


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
    parser.add_argument("--builder_preflight_json", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--blocker_audit_json", type=Path, default=DEFAULT_BLOCKER)
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument(
        "--explicit_allow_noisy_action_teacher",
        action="store_true",
        help="Record an explicit local policy exception for noisy action-teacher use; does not build data.",
    )
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    preflight = _load_json(args.builder_preflight_json)
    blocker = _load_json(args.blocker_audit_json)
    window = _load_json(args.reaction_window_json)

    counts = preflight.get("counts", {})
    tier_counts = window.get("quality_tier_counts", {})
    accepted = int(window.get("accepted_window_count", 0))
    tier_a = int(tier_counts.get("A_CLEAN_DIFFIK_TEACHER", 0))
    tier_b = int(tier_counts.get("B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH", 0))
    tier_c = int(tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0))
    clean_ready = _bool(window.get("clean_diffik_teacher_window_ready"))
    clip_mean = _float(window.get("accepted_window_clip_any_rate_mean"))
    follow_p95 = _float(window.get("accepted_window_follow_p95_to_cap_p95"))
    final_tcp = _float(
        blocker.get("quality_evidence", {}).get("summary_final_tcp_target_err_mean_m")
    )
    event_preflight_ready = (
        preflight.get("statuses", {}).get("tap_reaction_event_label_builder_preflight")
        == "READY_LOCAL_ONLY"
    )

    strict_clean_action_ready = clean_ready and tier_c == 0 and clip_mean <= 0.5 and follow_p95 <= 1.0
    tier_b_only_rows = tier_a + tier_b
    tier_b_only_ready = tier_b_only_rows >= accepted and accepted > 0
    noisy_requested = bool(args.explicit_allow_noisy_action_teacher)
    noisy_exception_recorded = noisy_requested and tier_c > 0

    if strict_clean_action_ready:
        action_policy_status = "READY_STRICT_CLEAN_TEACHER"
    elif noisy_exception_recorded:
        action_policy_status = "RISK_EXCEPTION_RECORDED_NOT_DATA_BUILT"
    else:
        action_policy_status = "BLOCKED_DEFAULT_POLICY"

    result = {
        "artifact_type": "cube10cm_diffik_teacher_quality_policy_gate_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_policy_gate_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "input_builder_preflight": str(args.builder_preflight_json),
        "input_blocker_audit": str(args.blocker_audit_json),
        "input_reaction_window": str(args.reaction_window_json),
        "statuses": {
            "event_label_dataset_path": "READY_LOCAL_ONLY" if event_preflight_ready else "BLOCKED",
            "strict_clean_diffik_action_teacher_path": "READY" if strict_clean_action_ready else "BLOCKED",
            "tier_b_only_action_teacher_path": "READY" if tier_b_only_ready else "BLOCKED_INSUFFICIENT_ROWS",
            "tier_b_c_noisy_action_teacher_path": "RISK_EXCEPTION_RECORDED_NOT_DATA_BUILT"
            if noisy_exception_recorded
            else "REQUIRES_EXPLICIT_POLICY_EXCEPTION",
            "default_action_teacher_dataset_policy": action_policy_status,
            "large_isaaclab_dataset": "BLOCKED",
            "isaaclab_rl": "BLOCKED",
            "roarm_m3_pro": "BLOCKED",
        },
        "evidence": {
            "accepted_windows": accepted,
            "tier_a": tier_a,
            "tier_b": tier_b,
            "tier_c": tier_c,
            "clean_diffik_teacher_window_ready": clean_ready,
            "accepted_window_clip_any_rate_mean": clip_mean,
            "accepted_window_follow_p95_to_cap_p95": follow_p95,
            "summary_final_tcp_target_err_mean_m": final_tcp,
            "event_preview_rows": int(counts.get("preview_rows", 0)),
            "event_preview_contact": int(counts.get("contact", 0)),
            "event_preview_reaction": int(counts.get("reaction", 0)),
            "event_preview_overshoot": int(counts.get("overshoot", 0)),
        },
        "policy_interpretation": {
            "event_labels_are_not_action_targets": True,
            "clean_diffik_teacher_is_quality_metadata_not_tap_filter": True,
            "default_noisy_action_teacher_allowed": False,
            "explicit_noisy_action_teacher_exception_requested": noisy_requested,
            "quality_policy_resolved_for_training": strict_clean_action_ready or noisy_exception_recorded,
        },
        "next_gate": {
            "if_no_exception": "improve_or_retest_teacher_quality_before_action_dataset",
            "if_exception_requested_later": "record_policy_exception_then_build_tiny_audited_action_dataset_dry_run_only",
            "before_rl": "validate_10cm_tap_rl_env_random_sanity",
            "before_robot": "validated_policy_plus_hardware_safety_replay",
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_diffik_teacher_quality_policy_gate_v1 "
        "local_policy_gate_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO",
        (
            "line2 event_label_path "
            f"status={result['statuses']['event_label_dataset_path']} "
            f"preview_rows={counts.get('preview_rows', 0)} contact={counts.get('contact', 0)} "
            f"reaction={counts.get('reaction', 0)} overshoot={counts.get('overshoot', 0)}"
        ),
        (
            "line3 strict_clean_action_teacher "
            f"status={result['statuses']['strict_clean_diffik_action_teacher_path']} "
            f"clean_teacher={clean_ready} tierA={tier_a} tierB={tier_b} tierC={tier_c} "
            f"clip_mean={clip_mean:.9f} follow_p95_to_cap={follow_p95:.9f} final_tcp_err={final_tcp:.9f}"
        ),
        (
            "line4 tier_b_only_action_teacher "
            f"status={result['statuses']['tier_b_only_action_teacher_path']} "
            f"usable_rows_without_tierC={tier_b_only_rows} accepted_windows={accepted}"
        ),
        (
            "line5 noisy_tier_bc_action_teacher "
            f"status={result['statuses']['tier_b_c_noisy_action_teacher_path']} "
            f"explicit_exception_requested={noisy_requested}"
        ),
        (
            "line6 default_policy "
            f"action_teacher_dataset={result['statuses']['default_action_teacher_dataset_policy']} "
            f"quality_policy_resolved_for_training={result['policy_interpretation']['quality_policy_resolved_for_training']}"
        ),
        (
            "line7 next_unblock "
            "no_exception=improve_or_retest_teacher_quality_before_action_dataset "
            "exception_path_requires_explicit_user_professor_policy_then_tiny_audited_dry_run_only"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
