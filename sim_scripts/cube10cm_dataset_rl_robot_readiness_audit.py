"""Local readiness audit for cube10cm tap dataset/RL/robot progression.

This audit answers whether the current professor 10cm/0.72kg tap/reaction
artifacts justify progressing to dataset generation, IsaacLab RL, and RoArm-M3-Pro
deployment. It distinguishes event-label readiness from action-teacher dataset
readiness and intentionally keeps final 1cm/final retention out of the primary
gate. It performs no IsaacLab/GPU runtime, dataset generation, training, robot
control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_TRANSIENT_AUDIT = LOG_DIR / "cube10cm_yplus_transient_tap_strength_audit.json"
DEFAULT_REACTION_GATE = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
)
DEFAULT_WINDOW_AUDIT = LOG_DIR / "cube10cm_reaction_window_seed962_audit.json"
DEFAULT_NEXT_STEP = LOG_DIR / "cube10cm_next_research_step_seed962_pre020_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_dataset_rl_robot_readiness_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_dataset_rl_robot_readiness_audit_summary.out"


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
    parser.add_argument("--transient_audit_json", type=Path, default=DEFAULT_TRANSIENT_AUDIT)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE)
    parser.add_argument("--reaction_window_json", type=Path, default=DEFAULT_WINDOW_AUDIT)
    parser.add_argument("--next_step_json", type=Path, default=DEFAULT_NEXT_STEP)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    transient = _load_json(args.transient_audit_json)
    reaction = _load_json(args.reaction_gate_json)
    window = _load_json(args.reaction_window_json)
    next_step = _load_json(args.next_step_json)

    transient_verdict = transient.get("verdict", {})
    run_summaries = transient.get("run_summaries", [])
    seed962 = next(
        row for row in run_summaries if isinstance(row, dict) and row.get("label") == "seed962_pre020"
    )
    rates = seed962.get("transient_strength_rates", {})
    tier_counts = window.get("quality_tier_counts", {})

    primary_event_ready = (
        _bool(transient_verdict.get("seed962_primary_tap_event_pass"))
        and _float(reaction.get("contact_evidence_rate")) >= 1.0
        and _float(reaction.get("overshoot_rate")) == 0.0
        and _bool(reaction.get("no_posewrite"))
    )
    one_two_mm_objective_ready = (
        primary_event_ready
        and _float(rates.get("max_ge_1mm_rate")) >= 1.0
        and _float(rates.get("max_ge_2mm_rate")) >= 0.75
    )
    quality_ready = (
        _bool(window.get("clean_diffik_teacher_window_ready"))
        and int(tier_counts.get("C_REACTION_VALID_FOLLOW_LAG", 0)) == 0
        and _float(window.get("accepted_window_clip_any_rate_mean")) <= 0.5
        and _float(window.get("accepted_window_follow_p95_to_cap_p95")) <= 1.0
    )
    existing_next_step_blocks_learning = any(
        token in set(next_step.get("do_not_start", []))
        for token in ("dataset_generation", "PPO_RL", "VLA", "1024_10k_scaleup")
    )
    legacy_dataset_builder_conflict = True
    rl_env_10cm_validated = False
    robot_deploy_validated = False

    gates = {
        "event_label_dataset_ready": primary_event_ready,
        "one_to_two_mm_tap_objective_ready": one_two_mm_objective_ready,
        "action_teacher_dataset_ready": quality_ready,
        "large_isaaclab_dataset_ready": quality_ready and not existing_next_step_blocks_learning,
        "isaaclab_rl_ready": (
            one_two_mm_objective_ready
            and quality_ready
            and rl_env_10cm_validated
            and not existing_next_step_blocks_learning
        ),
        "roarm_m3_pro_deploy_ready": (
            one_two_mm_objective_ready and quality_ready and rl_env_10cm_validated and robot_deploy_validated
        ),
    }

    blockers = []
    if not quality_ready:
        blockers.append(
            "teacher/action quality is not ready: clean_diffik_teacher_window_ready=false or Tier C/follow/clip remain"
        )
    if existing_next_step_blocks_learning:
        blockers.append("existing next-step audit still blocks dataset_generation/PPO_RL/VLA/scaleup")
    if legacy_dataset_builder_conflict:
        blockers.append(
            "legacy cube3cm dataset builder filters final_controlled/final_success, which conflicts with tap/reaction objective"
        )
    if not rl_env_10cm_validated:
        blockers.append("existing RoArmCubePushEnv is a 3cm task; 10cm/0.72kg tap RL env/random sanity is not validated")
    if not robot_deploy_validated:
        blockers.append("no learned/safe 10cm tap policy is validated for RoArm-M3-Pro deployment")

    next_sequence = [
        {
            "step": 1,
            "action": "freeze_objective_as_1_to_2mm_tap_reaction_if_accepted",
            "status": "PASS" if one_two_mm_objective_ready else "BLOCKED",
            "reason": "seed962 passes contact/reaction/no-overshoot/max1mm and majority max2mm",
        },
        {
            "step": 2,
            "action": "create_only_event_label_schema_or_manifest_from_existing_reaction_windows",
            "status": "ALLOWED_LOCAL_ONLY" if primary_event_ready else "BLOCKED",
            "reason": "labels may include contact/reaction/window/tier/transient strength; not an action teacher dataset",
        },
        {
            "step": 3,
            "action": "do_not_generate_action_teacher_dataset_yet",
            "status": "BLOCKED",
            "reason": "quality tier is 2B+14C, clip mean 1.0, follow p95/cap p95 > 1.0",
        },
        {
            "step": 4,
            "action": "do_not_start_isaaclab_rl_yet",
            "status": "BLOCKED",
            "reason": "10cm RL env/random sanity and teacher/action quality are not validated",
        },
        {
            "step": 5,
            "action": "do_not_deploy_to_roarm_m3_pro_yet",
            "status": "BLOCKED",
            "reason": "no validated learned policy, no hardware safety gate, no real-robot replay check",
        },
        {
            "step": 6,
            "action": "next_local_work",
            "status": "READY",
            "reason": "write a reaction-window event-label dataset schema/readiness manifest, not a training dataset",
        },
    ]

    result = {
        "artifact_type": "cube10cm_dataset_rl_robot_readiness_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_readiness_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "requested_pipeline": [
            "tap/reaction dataset",
            "IsaacLab dataset generation",
            "IsaacLab RL",
            "RoArm-M3-Pro deployment",
        ],
        "objective": {
            "primary": "reaction_contact_no_posewrite_no_overshoot",
            "final_1cm_or_final_retention_primary": False,
            "assumed_tap_strength": "1_to_2mm_transient",
        },
        "evidence": {
            "seed962_contact_rate": _float(reaction.get("contact_evidence_rate")),
            "seed962_reaction_rate": _float(reaction.get("reaction_event_rate")),
            "seed962_overshoot_rate": _float(reaction.get("overshoot_rate")),
            "seed962_no_posewrite": _bool(reaction.get("no_posewrite")),
            "seed962_max_ge_1mm_rate": _float(rates.get("max_ge_1mm_rate")),
            "seed962_max_ge_2mm_rate": _float(rates.get("max_ge_2mm_rate")),
            "seed962_max_ge_3mm_rate": _float(rates.get("max_ge_3mm_rate")),
            "clean_diffik_teacher_window_ready": _bool(window.get("clean_diffik_teacher_window_ready")),
            "quality_tier_counts": tier_counts,
            "accepted_window_clip_any_rate_mean": _float(window.get("accepted_window_clip_any_rate_mean")),
            "accepted_window_follow_p95_to_cap_p95": _float(
                window.get("accepted_window_follow_p95_to_cap_p95")
            ),
            "next_step_direction": next_step.get("next_direction"),
            "next_step_reasons": next_step.get("reasons", []),
        },
        "gates": gates,
        "blockers": blockers,
        "next_sequence": next_sequence,
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_dataset_rl_robot_readiness_audit_v1 local_readiness_only=YES "
        "gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO",
        (
            "line2 event_gate "
            f"primary_event_ready={primary_event_ready} "
            f"one_two_mm_objective_ready={one_two_mm_objective_ready} "
            f"contact={_float(reaction.get('contact_evidence_rate')):.9f} "
            f"reaction={_float(reaction.get('reaction_event_rate')):.9f} "
            f"overshoot={_float(reaction.get('overshoot_rate')):.9f} "
            f"max1mm={_float(rates.get('max_ge_1mm_rate')):.9f} "
            f"max2mm={_float(rates.get('max_ge_2mm_rate')):.9f}"
        ),
        (
            "line3 quality_gate "
            f"action_teacher_dataset_ready={gates['action_teacher_dataset_ready']} "
            f"clean_teacher={_bool(window.get('clean_diffik_teacher_window_ready'))} "
            f"tier_counts={tier_counts} "
            f"clip_mean={_float(window.get('accepted_window_clip_any_rate_mean')):.9f} "
            f"follow_p95_to_cap={_float(window.get('accepted_window_follow_p95_to_cap_p95')):.9f}"
        ),
        (
            "line4 pipeline_gates "
            f"event_label_dataset_ready={gates['event_label_dataset_ready']} "
            f"large_isaaclab_dataset_ready={gates['large_isaaclab_dataset_ready']} "
            f"isaaclab_rl_ready={gates['isaaclab_rl_ready']} "
            f"roarm_m3_pro_deploy_ready={gates['roarm_m3_pro_deploy_ready']}"
        ),
        (
            "line5 next_step_guard "
            f"direction={next_step.get('next_direction')} "
            "do_not_start=dataset_generation,PPO_RL,VLA,1024_10k_scaleup"
        ),
        (
            "line6 next_sequence "
            "allowed=local_event_label_schema_manifest_only "
            "blocked=action_teacher_dataset,large_isaaclab_dataset,isaaclab_rl,roarm_m3_pro_deploy"
        ),
        (
            "line7 reason "
            "legacy_dataset_builder_uses_final_success_filter=True "
            "ten_cm_rl_env_validated=False "
            "quality_still_blocks_learning=True"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
