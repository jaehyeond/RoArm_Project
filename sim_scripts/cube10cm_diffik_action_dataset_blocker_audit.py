"""Local blocker audit for cube10cm DiffIK action-dataset progression.

This audit checks whether the current professor 10cm/0.72kg tap/reaction
artifacts justify moving from event labels to a Differential-IK action teacher
dataset, then IsaacLab RL, then RoArm-M3-Pro deployment. It performs no
IsaacLab/GPU runtime, dataset generation, training, robot control, SSH, or trace
mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_REACTION_GATE = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
)
DEFAULT_WINDOW_AUDIT = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
DEFAULT_TRACE_DIAG = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace_diagnostic_summary.json"
)
DEFAULT_READINESS = LOG_DIR / "cube10cm_dataset_rl_robot_readiness_audit.json"
DEFAULT_MANIFEST = LOG_DIR / "cube10cm_event_label_dataset_manifest.json"
DEFAULT_NEXT_STEP = LOG_DIR / "cube10cm_next_research_step_seed962_pre020_audit.json"
DEFAULT_DATASET_BUILDER = REPO / "sim_scripts/cube3cm_push_diffik_build_dataset.py"
DEFAULT_RL_ENV = REPO / "roarm_rl/roarm_cube_push_env.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_diffik_action_dataset_blocker_audit_summary.out"


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


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE)
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument("--trace_diag_json", type=Path, default=DEFAULT_TRACE_DIAG)
    parser.add_argument("--readiness_json", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--event_manifest_json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--next_step_json", type=Path, default=DEFAULT_NEXT_STEP)
    parser.add_argument("--legacy_dataset_builder", type=Path, default=DEFAULT_DATASET_BUILDER)
    parser.add_argument("--rl_env_py", type=Path, default=DEFAULT_RL_ENV)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    reaction = _load_json(args.reaction_gate_json)
    window = _load_json(args.reaction_window_json)
    trace = _load_json(args.trace_diag_json)
    readiness = _load_json(args.readiness_json)
    manifest = _load_json(args.event_manifest_json)
    next_step = _load_json(args.next_step_json)

    tier_counts = window.get("quality_tier_counts", {})
    gates = readiness.get("gates", {})
    manifest_counts = manifest.get("counts", {})
    likely_modes = list(trace.get("likely_modes", []))

    primary_event_pass = (
        _bool(reaction.get("reaction_gate_pass"))
        and _float(reaction.get("contact_evidence_rate")) >= 1.0
        and _float(reaction.get("reaction_event_rate")) >= 1.0
        and _float(reaction.get("overshoot_rate")) == 0.0
        and _bool(reaction.get("no_posewrite"))
    )
    event_label_dataset_ready = (
        primary_event_pass
        and bool(gates.get("event_label_dataset_ready"))
        and bool(manifest.get("local_manifest_only"))
        and bool(manifest.get("not_action_teacher_dataset"))
    )

    clean_window_ready = _bool(window.get("clean_diffik_teacher_window_ready"))
    no_tier_c = int(tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0)) == 0
    clip_ok = _float(window.get("accepted_window_clip_any_rate_mean")) <= 0.5
    follow_ok = _float(window.get("accepted_window_follow_p95_to_cap_p95")) <= 1.0
    final_tcp_ok = _float(reaction.get("summary_final_tcp_target_err_mean_m")) <= _float(
        reaction.get("thresholds", {}).get("teacher_max_final_tcp_err_m"), 0.03
    )
    action_teacher_dataset_ready = clean_window_ready and no_tier_c and clip_ok and follow_ok and final_tcp_ok

    legacy_builder_final_success_conflict = all(
        _line_of(args.legacy_dataset_builder, pattern) is not None
        for pattern in (
            'as_int_bool(final, "controlled_push")',
            'as_int_bool(final, "low_motion")',
            'as_int_bool(final, "success_marker")',
            "final_controlled=1 final_impact=0 final_low_motion=0 final_success=1",
        )
    )
    rl_env_3cm_relocation_conflict = all(
        _line_of(args.rl_env_py, pattern) is not None
        for pattern in (
            "no-attach 3cm cube push task",
            "CUBE_SIZE_M = 0.030",
            "mass=0.020",
            "cube_success_disp_m: float = 0.030",
            'terms["disp_along"] >= self.cfg.cube_success_disp_m',
        )
    )

    next_step_blocks = set(str(x) for x in next_step.get("do_not_start", []))
    large_isaaclab_dataset_ready = action_teacher_dataset_ready and "dataset_generation" not in next_step_blocks
    isaaclab_rl_ready = (
        action_teacher_dataset_ready
        and large_isaaclab_dataset_ready
        and not rl_env_3cm_relocation_conflict
        and "PPO_RL" not in next_step_blocks
    )
    roarm_m3_pro_ready = isaaclab_rl_ready and False

    blockers = []
    if not action_teacher_dataset_ready:
        blockers.append(
            "DiffIK action teacher quality is blocked by clean_teacher/Tier-C/follow/clip/final-TCP gates"
        )
    if legacy_builder_final_success_conflict:
        blockers.append("legacy dataset builder still filters final controlled/success/low-motion markers")
    if "dataset_generation" in next_step_blocks:
        blockers.append("next-step audit explicitly blocks dataset_generation")
    if rl_env_3cm_relocation_conflict:
        blockers.append("existing RL env is 3cm relocation-oriented, not validated 10cm tap/reaction")
    if "PPO_RL" in next_step_blocks:
        blockers.append("next-step audit explicitly blocks PPO_RL")
    if not roarm_m3_pro_ready:
        blockers.append("RoArm-M3-Pro deployment lacks a validated learned policy and safety/replay gate")

    resolution_order = [
        {
            "step": 1,
            "name": "keep_1_to_2mm_tap_event_objective",
            "status": "PASS" if primary_event_pass else "BLOCKED",
            "why": "reaction/contact/no-posewrite/no-overshoot pass; final retention is not primary",
        },
        {
            "step": 2,
            "name": "event_label_manifest",
            "status": "READY_LOCAL_ONLY" if event_label_dataset_ready else "BLOCKED",
            "why": "existing manifest has reaction-window labels and quality metadata only",
        },
        {
            "step": 3,
            "name": "ten_cm_tap_specific_dataset_builder_preflight",
            "status": "NEXT_LOCAL_WORK",
            "why": "needed before any large IsaacLab dataset; it must not use final-success filters",
        },
        {
            "step": 4,
            "name": "diffik_action_teacher_dataset",
            "status": "BLOCKED" if not action_teacher_dataset_ready else "READY",
            "why": "needs clean/tier/follow/clip gate to pass or an explicit noisy-teacher policy decision",
        },
        {
            "step": 5,
            "name": "ten_cm_tap_rl_env_random_sanity",
            "status": "BLOCKED" if rl_env_3cm_relocation_conflict else "READY_FOR_LOCAL_PREFLIGHT",
            "why": "existing env hard-codes 3cm/20g relocation assumptions and success reward",
        },
        {
            "step": 6,
            "name": "isaaclab_rl_training",
            "status": "BLOCKED" if not isaaclab_rl_ready else "READY",
            "why": "requires dataset/objective/env sanity gates first",
        },
        {
            "step": 7,
            "name": "roarm_m3_pro_deployment",
            "status": "BLOCKED",
            "why": "requires validated policy plus hardware safety/replay gate",
        },
    ]

    result = {
        "artifact_type": "cube10cm_diffik_action_dataset_blocker_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_blocker_audit_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "statuses": {
            "event_label_dataset": "READY_LOCAL_ONLY" if event_label_dataset_ready else "BLOCKED",
            "differential_ik_action_teacher_dataset": "BLOCKED"
            if not action_teacher_dataset_ready
            else "READY",
            "large_isaaclab_dataset": "BLOCKED" if not large_isaaclab_dataset_ready else "READY",
            "isaaclab_rl": "BLOCKED" if not isaaclab_rl_ready else "READY",
            "roarm_m3_pro": "BLOCKED" if not roarm_m3_pro_ready else "READY",
        },
        "event_evidence": {
            "reaction_gate_pass": _bool(reaction.get("reaction_gate_pass")),
            "contact_evidence_rate": _float(reaction.get("contact_evidence_rate")),
            "reaction_event_rate": _float(reaction.get("reaction_event_rate")),
            "overshoot_rate": _float(reaction.get("overshoot_rate")),
            "no_posewrite": _bool(reaction.get("no_posewrite")),
            "manifest_event_count": int(manifest_counts.get("event_count", 0)),
            "manifest_ge_1mm": int(manifest_counts.get("max_transient_ge_1mm_count", 0)),
            "manifest_ge_2mm": int(manifest_counts.get("max_transient_ge_2mm_count", 0)),
            "manifest_ge_3mm": int(manifest_counts.get("max_transient_ge_3mm_count", 0)),
        },
        "quality_evidence": {
            "clean_diffik_teacher_window_ready": clean_window_ready,
            "quality_tier_counts": tier_counts,
            "accepted_window_clip_any_rate_mean": _float(window.get("accepted_window_clip_any_rate_mean")),
            "accepted_window_follow_p95_to_cap_p95": _float(window.get("accepted_window_follow_p95_to_cap_p95")),
            "summary_diffik_clip_rate_mean": _float(reaction.get("summary_diffik_clip_rate_mean")),
            "summary_final_tcp_target_err_mean_m": _float(reaction.get("summary_final_tcp_target_err_mean_m")),
            "likely_trace_modes": likely_modes,
            "trace_clip_any_rate": _float(trace.get("clip_any_rate")),
            "worst_follow_joint": trace.get("worst_follow_joint"),
            "worst_raw_delta_joint": trace.get("worst_raw_delta_joint"),
        },
        "code_conflicts": {
            "legacy_dataset_builder_final_success_conflict": legacy_builder_final_success_conflict,
            "legacy_dataset_builder_evidence_lines": {
                "controlled_push_filter": _line_of(args.legacy_dataset_builder, 'as_int_bool(final, "controlled_push")'),
                "low_motion_filter": _line_of(args.legacy_dataset_builder, 'as_int_bool(final, "low_motion")'),
                "success_marker_filter": _line_of(args.legacy_dataset_builder, 'as_int_bool(final, "success_marker")'),
                "teacher_filter_summary": _line_of(
                    args.legacy_dataset_builder,
                    "final_controlled=1 final_impact=0 final_low_motion=0 final_success=1",
                ),
            },
            "rl_env_3cm_relocation_conflict": rl_env_3cm_relocation_conflict,
            "rl_env_evidence_lines": {
                "doc_3cm_task": _line_of(args.rl_env_py, "no-attach 3cm cube push task"),
                "cube_size_3cm": _line_of(args.rl_env_py, "CUBE_SIZE_M = 0.030"),
                "mass_20g": _line_of(args.rl_env_py, "mass=0.020"),
                "success_disp_3cm": _line_of(args.rl_env_py, "cube_success_disp_m: float = 0.030"),
                "success_reward_final_disp": _line_of(
                    args.rl_env_py,
                    'terms["disp_along"] >= self.cfg.cube_success_disp_m',
                ),
            },
        },
        "blockers": blockers,
        "resolution_order": resolution_order,
        "source_files": {
            "reaction_gate_json": str(args.reaction_gate_json),
            "reaction_window_json": str(args.reaction_window_json),
            "trace_diag_json": str(args.trace_diag_json),
            "readiness_json": str(args.readiness_json),
            "event_manifest_json": str(args.event_manifest_json),
            "next_step_json": str(args.next_step_json),
            "legacy_dataset_builder": str(args.legacy_dataset_builder),
            "rl_env_py": str(args.rl_env_py),
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = result["statuses"]
    lines = [
        "line1 artifact=cube10cm_diffik_action_dataset_blocker_audit_v1 "
        "local_blocker_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO",
        (
            "line2 event_label_dataset "
            f"status={status['event_label_dataset']} events={manifest_counts.get('event_count', 0)} "
            f"contact={reaction.get('contact_evidence_rate')} reaction={reaction.get('reaction_event_rate')} "
            f"overshoot={reaction.get('overshoot_rate')} ge_1mm={manifest_counts.get('max_transient_ge_1mm_count', 0)} "
            f"ge_2mm={manifest_counts.get('max_transient_ge_2mm_count', 0)} ge_3mm={manifest_counts.get('max_transient_ge_3mm_count', 0)}"
        ),
        (
            "line3 diffik_action_teacher_dataset "
            f"status={status['differential_ik_action_teacher_dataset']} clean_teacher={clean_window_ready} "
            f"tiers={tier_counts} clip_mean={_float(window.get('accepted_window_clip_any_rate_mean')):.9f} "
            f"follow_p95_to_cap={_float(window.get('accepted_window_follow_p95_to_cap_p95')):.9f} "
            f"final_tcp_err={_float(reaction.get('summary_final_tcp_target_err_mean_m')):.9f}"
        ),
        (
            "line4 trace_quality "
            f"clip_any_rate={_float(trace.get('clip_any_rate')):.9f} "
            f"modes={','.join(likely_modes)} "
            f"worst_follow_joint={trace.get('worst_follow_joint', {}).get('joint')} "
            f"worst_raw_delta_joint={trace.get('worst_raw_delta_joint', {}).get('joint')}"
        ),
        (
            "line5 code_conflicts "
            f"legacy_builder_final_success_filter={legacy_builder_final_success_conflict} "
            f"rl_env_3cm_relocation={rl_env_3cm_relocation_conflict}"
        ),
        (
            "line6 pipeline "
            f"large_isaaclab_dataset={status['large_isaaclab_dataset']} "
            f"isaaclab_rl={status['isaaclab_rl']} roarm_m3_pro={status['roarm_m3_pro']}"
        ),
        (
            "line7 next_unblock_order "
            "1_event_label_manifest_ready "
            "2_write_10cm_tap_dataset_builder_preflight_no_final_success "
            "3_fix_or_explicitly_accept_diffik_teacher_quality_policy "
            "4_validate_10cm_tap_rl_env_random_sanity "
            "5_only_then_train "
            "6_only_then_robot_safety_replay"
        ),
        (
            "line8 do_not_start "
            "large_dataset,IsaacLab_RL_training,RoArm_M3_Pro_deploy,1024_10240,VLA,TrackA,B200_SSH"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
