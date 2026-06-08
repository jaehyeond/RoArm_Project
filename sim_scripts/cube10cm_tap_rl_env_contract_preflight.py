"""Local contract preflight for a future cube10cm tap/reaction RL env.

This does not modify or instantiate the RL environment. It audits the current
RoArmCubePushEnv source against the professor 10cm/0.72kg weak tap/reaction
objective and writes the contract that must be satisfied before any random
sanity, PPO/RL training, or RoArm deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_PROMOTION_JSON = LOG_DIR / "cube10cm_link5corner_next_stage_promotion_gate.json"
DEFAULT_EVENT_MANIFEST = LOG_DIR / "cube10cm_link5corner_event_label_metadata_manifest.json"
DEFAULT_RL_ENV = REPO / "roarm_rl/roarm_cube_push_env.py"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_env_contract_preflight.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_env_contract_preflight_summary.out"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_of(path: Path, pattern: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion_gate_json", type=Path, default=DEFAULT_PROMOTION_JSON)
    parser.add_argument("--event_manifest_json", type=Path, default=DEFAULT_EVENT_MANIFEST)
    parser.add_argument("--rl_env_py", type=Path, default=DEFAULT_RL_ENV)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    promotion = _load_json(args.promotion_gate_json)
    manifest = _load_json(args.event_manifest_json)
    counts = manifest.get("counts", {})

    current_conflict_patterns = {
        "doc_3cm_task": "no-attach 3cm cube push task",
        "cube_size_3cm": "CUBE_SIZE_M = 0.030",
        "mass_20g": "mass=0.020",
        "push_target_4cm": "cube_push_target_disp_m: float = 0.040",
        "success_disp_3cm": "cube_success_disp_m: float = 0.030",
        "success_target_distance_gate": "terms[\"target_xy_dist\"] <= self.cfg.cube_success_target_tol_m",
        "success_speed_gate": "terms[\"speed\"] <= self.cfg.cube_success_speed_max_mps",
        "success_bonus": "success_bonus: float = 12.0",
    }
    evidence_lines = {
        name: _line_of(args.rl_env_py, pattern) for name, pattern in current_conflict_patterns.items()
    }
    current_env_conflicts = {name: line for name, line in evidence_lines.items() if line is not None}

    event_ready = (
        bool(manifest.get("local_manifest_only"))
        and bool(manifest.get("not_action_teacher_dataset"))
        and int(counts.get("event_count", 0)) == 16
        and int(counts.get("weak_1mm_count", 0)) == 16
        and int(counts.get("overshoot_count", 1)) == 0
    )
    promotion_ready = "ten_cm_tap_rl_env_contract_preflight_local_only" in promotion.get("answer", {}).get(
        "can_move_now_to", []
    )
    contract_preflight_ready = event_ready and promotion_ready
    existing_env_compatible = not current_env_conflicts

    required_contract = {
        "object": {
            "cube_size_m": [0.1, 0.1, 0.1],
            "cube_mass_kg": 0.72,
            "density_preserving_from_professor_branch": True,
        },
        "primary_objective": {
            "name": "tap_reaction_contact_not_final_relocation",
            "requires": [
                "contact_evidence",
                "reaction_signal",
                "no_posewrite",
                "no_overshoot",
                "weak_1mm_tap_reaction_if_accepted",
            ],
            "must_not_require_by_default": [
                "final_1cm_relocation",
                "final_retention",
                "3cm_success_marker",
                "target_xy_final_distance_success",
            ],
        },
        "reward_and_done_contract": {
            "reward_should_favor": [
                "contact_event",
                "small_transient_disp_or_speed_reaction",
                "no_overshoot",
                "low_tip_or_stability_sanity",
            ],
            "done_should_not_default_to": [
                "3cm_relocation_success",
                "final_target_xy_distance_success",
            ],
            "random_sanity_before_training": [
                "reset cube is 10cm and 0.72kg",
                "zero/random policy logs contact/reaction/overshoot separately",
                "success metric is tap-event, not relocation",
                "no object posewrite or grasp attach",
            ],
        },
        "policy_boundaries": {
            "event_labels_available": event_ready,
            "action_teacher_dataset_available": False,
            "large_dataset_allowed": False,
            "rl_training_allowed": False,
            "robot_deploy_allowed": False,
        },
    }

    result = {
        "artifact_type": "cube10cm_tap_rl_env_contract_preflight_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_contract_preflight_only": True,
        "no_gpu_isaaclab_dataset_training_robot_ssh": True,
        "statuses": {
            "contract_preflight": "READY_LOCAL_ONLY" if contract_preflight_ready else "BLOCKED",
            "current_roarm_cube_push_env": "INCOMPATIBLE_WITH_10CM_TAP_CONTRACT"
            if not existing_env_compatible
            else "COMPATIBLE",
            "random_sanity_runtime": "BLOCKED_UNTIL_ENV_CONTRACT_IMPLEMENTED",
            "ppo_rl_training": "BLOCKED",
            "roarm_m3_pro_deployment": "BLOCKED",
        },
        "current_env_conflicts": current_env_conflicts,
        "required_contract": required_contract,
        "next_implementation_step": (
            "create a default-off 10cm tap env config or separate env wrapper that changes "
            "object size/mass/objective/reward/done/logging without starting training"
        ),
        "source_files": {
            "promotion_gate_json": str(args.promotion_gate_json),
            "event_manifest_json": str(args.event_manifest_json),
            "rl_env_py": str(args.rl_env_py),
        },
        "out_summary": str(args.out_summary),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_env_contract_preflight_v1 "
        "local_contract_preflight_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 promotion "
            f"contract_preflight={result['statuses']['contract_preflight']} "
            f"event_labels_ready={event_ready} promotion_ready={promotion_ready}"
        ),
        (
            "line3 current_env "
            f"status={result['statuses']['current_roarm_cube_push_env']} "
            f"conflict_count={len(current_env_conflicts)} "
            f"conflicts={','.join(sorted(current_env_conflicts))}"
        ),
        (
            "line4 required_contract "
            "cube_size=0.100m cube_mass=0.720kg objective=tap_reaction_contact_not_final_relocation "
            "default_final_1cm_required=NO"
        ),
        (
            "line5 next_allowed "
            "implement_default_off_10cm_tap_env_contract_or_wrapper=YES "
            "random_sanity_runtime=AFTER_CONTRACT_IMPLEMENTED_ONLY"
        ),
        "line6 blocked ppo_rl_training=YES large_dataset=YES roarm_m3_pro=YES",
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
