#!/usr/bin/env python3
"""Local code-review audit for 10cm tap built-in DiffIK parity before gate changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
MEMORY = Path("/home/cgxr/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md")

PATHS = {
    "claude": ROOT / "CLAUDE.md",
    "start_here": ROOT / "START_HERE.md",
    "decisions": ROOT / "claudedocs/DECISIONS.md",
    "ledger": ROOT / "claudedocs/EXPERIMENT_LEDGER.md",
    "session": ROOT / "claudedocs/session_20260608_cube10cm_tap_rl_preflight_policy_gate.md",
    "memory": MEMORY,
    "cube3cm_probe": ROOT / "sim_scripts/cube3cm_push_diffik_probe.py",
    "cube10cm_probe": ROOT / "sim_scripts/cube10cm_push_diffik_probe.py",
    "tap_harness": ROOT / "roarm_rl/test_positive_control_cube_tap10cm.py",
    "tap_env": ROOT / "roarm_rl/roarm_cube_push_env.py",
    "isaac_task_space": Path(
        "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
        "source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py"
    ),
    "isaac_diffik_test": Path(
        "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
        "source/isaaclab/test/controllers/test_differential_ik.py"
    ),
    "direct_candidate_summary": LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_candidate_design_summary.out",
    "direct_telemetry_summary": LOG_DIR
    / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_telemetry_sanity_summary.out",
    "direct_telemetry_audit": LOG_DIR / "cube10cm_tap_rl_direct_ik_telemetry_result_audit_summary.out",
    "slow240_audit": LOG_DIR / "cube10cm_tap_rl_slow240_result_audit_summary.out",
    "controller_contract_summary": LOG_DIR / "cube10cm_vs_cube3cm_controller_contract_audit_summary.out",
}

OUT_JSON = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_code_review.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_builtin_diffik_parity_code_review_summary.out"


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def line(path: Path, one_based: int) -> str:
    rows = lines(path)
    return rows[one_based - 1] if 0 < one_based <= len(rows) else ""


def find_line(path: Path, needle: str) -> int | None:
    for idx, text in enumerate(lines(path), start=1):
        if needle in text:
            return idx
    return None


def has(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def snippet(path: Path, needle: str) -> dict[str, Any]:
    idx = find_line(path, needle)
    return {"line": idx, "text": line(path, idx) if idx is not None else None}


def main() -> int:
    for key, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")

    tap_text = PATHS["tap_harness"].read_text(encoding="utf-8")
    three_text = PATHS["cube3cm_probe"].read_text(encoding="utf-8")

    original_10cm_transition_kept_builtin_diffik = (
        has(PATHS["decisions"], "Use the IsaacLab built-in `DifferentialIKController` probe path")
        and has(PATHS["cube10cm_probe"], "shared DiffIK probe engine")
        and has(PATHS["cube3cm_probe"], "DifferentialIKController")
    )
    tap_harness_has_builtin_diffik_controller = "DifferentialIKController" in tap_text
    tap_harness_uses_local_ik_dls = "from sim_scripts.roarm_kinematics import fk_tcp, ik_dls" in tap_text
    three_probe_uses_builtin_diffik = (
        "from isaaclab.controllers import DifferentialIKController" in three_text
        and "diffik.compute" in three_text
    )
    direct_apply_reason_documented = (
        has(PATHS["decisions"], "rewraps the result as normalized RL action")
        and has(PATHS["decisions"], "apply the IK joint target directly")
        and has(PATHS["direct_candidate_summary"], "separate_target_geometry_or_ik_from_RL_action_target_application")
    )

    gate_relaxation_would_hide_mismatch = (
        original_10cm_transition_kept_builtin_diffik
        and three_probe_uses_builtin_diffik
        and tap_harness_uses_local_ik_dls
        and not tap_harness_has_builtin_diffik_controller
        and direct_apply_reason_documented
    )

    artifact: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_builtin_diffik_parity_code_review_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_code_review_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "inputs": {key: str(path) for key, path in PATHS.items()},
        "memory_checked_but_repo_docs_primary": {
            "memory_line_74": line(PATHS["memory"], 74),
            "claude_repo_docs_primary": [line(PATHS["claude"], idx) for idx in (14, 15, 16, 18, 19, 23, 31, 53, 55)],
        },
        "code_review": {
            "original_10cm_transition_kept_builtin_diffik": original_10cm_transition_kept_builtin_diffik,
            "three_cm_probe_uses_builtin_diffik": three_probe_uses_builtin_diffik,
            "ten_cm_probe_is_thin_wrapper_over_three_cm_diffik_engine": has(
                PATHS["cube10cm_probe"], "from sim_scripts import cube3cm_push_diffik_probe as shared_probe"
            ),
            "tap_harness_uses_local_ik_dls": tap_harness_uses_local_ik_dls,
            "tap_harness_has_builtin_diffik_controller": tap_harness_has_builtin_diffik_controller,
            "tap_harness_controller_choices": snippet(PATHS["tap_harness"], 'choices=("builtin_teacher"'),
            "tap_harness_local_ik_import": snippet(PATHS["tap_harness"], "from sim_scripts.roarm_kinematics import fk_tcp, ik_dls"),
            "tap_harness_action_rewrap": snippet(PATHS["tap_harness"], "action_t = (target_t - target_base)"),
            "tap_harness_direct_override_write": snippet(
                PATHS["tap_harness"], "inner._external_joint_targets_override = joint_target"
            ),
            "env_direct_override_read": snippet(PATHS["tap_env"], "_external_joint_targets_override"),
            "env_action_wrapper_scale_cap_lead": [
                snippet(PATHS["tap_env"], "self._smoothed_actions"),
                snippet(PATHS["tap_env"], "raw_delta = self.cfg.action_scale * self._smoothed_actions"),
                snippet(PATHS["tap_env"], "self._last_joint_delta_cap_rate[:] ="),
                snippet(PATHS["tap_env"], "targets_unclamped = target_base + delta"),
                snippet(PATHS["tap_env"], "targets = torch.maximum(torch.minimum(targets, joint_pos + lead)"),
            ],
            "contact_gate": [
                snippet(PATHS["tap_env"], "face_gap >= -float(self.cfg.tap_contact_face_band_m)"),
                snippet(PATHS["tap_env"], "contact_context = contact_proxy | self._tap_contact_seen"),
                snippet(PATHS["tap_env"], "success_now = (contact_proxy | self._tap_contact_seen)"),
            ],
            "isaac_task_space_pattern": [
                snippet(PATHS["isaac_task_space"], "self._ik_controller = DifferentialIKController"),
                snippet(PATHS["isaac_task_space"], "joint_pos_des = self._ik_controller.compute"),
                snippet(PATHS["isaac_task_space"], "self._asset.set_joint_position_target(joint_pos_des"),
            ],
        },
        "decision_chain": {
            "d118_original_10cm_controller_contract": [
                "claudedocs/DECISIONS.md:6012-6019",
                "claudedocs/DECISIONS.md:6030-6034",
            ],
            "d122_failure_interpretation": [
                "claudedocs/DECISIONS.md:6212-6219",
                "claudedocs/DECISIONS.md:6265-6272",
            ],
            "d185_why_direct_apply_was_created": [
                "claudedocs/DECISIONS.md:10065-10079",
                "claudedocs/DECISIONS.md:10081-10101",
            ],
            "d186_direct_apply_result": [
                "claudedocs/DECISIONS.md:10129-10149",
                "claudedocs/DECISIONS.md:10181-10189",
            ],
            "session_source_cross_check": [
                "claudedocs/session_20260608_cube10cm_tap_rl_preflight_policy_gate.md:485-524",
                "claudedocs/session_20260608_cube10cm_tap_rl_preflight_policy_gate.md:528-564",
            ],
        },
        "existing_log_evidence": {
            "direct_candidate_summary_lines_1_6": [line(PATHS["direct_candidate_summary"], idx) for idx in range(1, 7)],
            "direct_telemetry_summary_lines_1_10": [line(PATHS["direct_telemetry_summary"], idx) for idx in range(1, 11)],
            "direct_telemetry_audit_lines_1_7": [line(PATHS["direct_telemetry_audit"], idx) for idx in range(1, 8)],
            "slow240_audit_lines_1_8": [line(PATHS["slow240_audit"], idx) for idx in range(1, 9)],
            "controller_contract_summary_lines_1_10": [
                line(PATHS["controller_contract_summary"], idx) for idx in range(1, 11)
            ],
        },
        "interpretation": {
            "direct_apply_was_not_the_original_10cm_controller_switch": True,
            "direct_apply_was_a_wrapper_isolation_diagnostic": direct_apply_reason_documented,
            "current_10cm_tap_positive_control_is_not_built_in_diffik_parity": (
                tap_harness_uses_local_ik_dls and not tap_harness_has_builtin_diffik_controller
            ),
            "contact_gate_relaxation_before_parity_would_hide_controller_mismatch": gate_relaxation_would_hide_mismatch,
            "recommended_next_step": (
                "design or implement a default-off 10cm tap positive-control mode using IsaacLab built-in "
                "DifferentialIKController with live Jacobian parity before any contact-band/Tier-B exception"
            ),
            "runtime_requires_explicit_approval": True,
        },
        "outcome": {
            "parity_code_review": "READY_LOCAL_ONLY",
            "contact_gate_relaxation": "BLOCKED_BEFORE_BUILTIN_DIFFIK_PARITY_REVIEWED_OR_TESTED",
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "outputs": {"json": str(OUT_JSON), "summary": str(OUT_SUMMARY)},
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines_out = [
        "line1 artifact=cube10cm_tap_rl_builtin_diffik_parity_code_review_v1 "
        "local_code_review_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 current_state memory_checked=YES repo_docs_primary=YES handoff_tasks_used=NO "
        "professor_branch=10cm_0p72kg_tap_reaction_quality_tier",
        "line3 original_10cm_transition controller_contract=BUILTIN_DIFFIK_PRESERVED "
        f"original_10cm_transition_kept_builtin_diffik={original_10cm_transition_kept_builtin_diffik} "
        f"ten_cm_probe_wraps_three_cm_engine={artifact['code_review']['ten_cm_probe_is_thin_wrapper_over_three_cm_diffik_engine']}",
        "line4 current_tap_harness parity_status=NOT_BUILTIN_DIFFIK_PARITY "
        f"uses_local_ik_dls={tap_harness_uses_local_ik_dls} "
        f"has_DifferentialIKController={tap_harness_has_builtin_diffik_controller} "
        "controller_choices=builtin_teacher_external_closed_loop_external_closed_loop_direct_apply",
        "line5 why_direct_apply_exists reason=WRAPPER_ISOLATION_DIAGNOSTIC_NOT_ORIGINAL_10CM_SWITCH "
        f"direct_apply_reason_documented={direct_apply_reason_documented} "
        "basis=IK_target_to_RL_action_path_was_suspect_vs_Isaac_set_joint_position_target_pattern",
        "line6 existing_results direct_apply_contact=0.0 direct_apply_tap=0.0 "
        "direct_apply_best_shortfall_m=0.009533616 slow240_contact=0.0 slow240_tap=0.0 "
        "slow240_shortfall_m=0.009191336",
        "line7 verdict gate_relaxation_before_parity=BLOCKED "
        f"would_hide_controller_mismatch={gate_relaxation_would_hide_mismatch} "
        "next=default_off_builtin_DifferentialIKController_parity_design_or_patch_first",
        "line8 blocked_status diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
        "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED runtime_requires_explicit_approval=YES",
    ]
    OUT_SUMMARY.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    for text in lines_out:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
