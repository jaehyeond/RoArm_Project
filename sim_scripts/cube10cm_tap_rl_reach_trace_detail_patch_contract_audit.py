#!/usr/bin/env python3
"""Static contract audit for the cube10cm reach-trace detail patch."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "roarm_rl/test_positive_control_cube_tap10cm.py"
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
BASIS_SUMMARY = LOG_DIR / "cube10cm_tap_rl_remaining_blocker_decomposition_design_summary.out"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_reach_trace_detail_patch_contract_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_reach_trace_detail_patch_contract_audit_summary.out"


def line_no(lines: list[str], pattern: str) -> int:
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return idx
    return -1


def require_line(lines: list[str], pattern: str) -> int:
    found = line_no(lines, pattern)
    if found < 0:
        raise AssertionError(f"missing pattern: {pattern}")
    return found


def main() -> int:
    harness_lines = HARNESS.read_text(encoding="utf-8").splitlines()
    basis_lines = BASIS_SUMMARY.read_text(encoding="utf-8").splitlines()

    checks = {
        "detail_arg_line": require_line(harness_lines, 'parser.add_argument("--reach_trace_detail_json"'),
        "detail_rows_init_line": require_line(harness_lines, "reach_detail_rows: list[dict[str, Any]] = []"),
        "trace_fk_enabled_line": require_line(harness_lines, "trace_fk_enabled ="),
        "previous_target_state_line": require_line(harness_lines, "\"previous_arm_joint_target\": None"),
        "previous_target_update_line": require_line(
            harness_lines, "state[\"previous_arm_joint_target\"] = arm_joint_target.detach().clone()"
        ),
        "detail_writer_line": require_line(harness_lines, "def _write_reach_trace_detail"),
        "detail_artifact_line": require_line(harness_lines, "cube10cm_tap_rl_reach_trace_detail_v1"),
        "action_teacher_false_line": require_line(harness_lines, "\"action_teacher_dataset\": False"),
        "contains_action_false_line": require_line(harness_lines, "\"contains_action_fields\": False"),
        "result_detail_enabled_line": require_line(harness_lines, "\"reach_trace_detail_enabled\""),
        "detail_write_call_line": require_line(
            harness_lines, "_write_reach_trace_detail(args.reach_trace_detail_json"
        ),
        "joint_pos_target_after_line": require_line(harness_lines, "joint_pos_target_after_arm_rad"),
        "joint_vel_after_line": require_line(harness_lines, "joint_vel_after_arm_rad"),
        "joint_acc_after_line": require_line(harness_lines, "joint_acc_after_arm_rad"),
        "computed_torque_after_line": require_line(harness_lines, "computed_torque_after_arm_nm"),
        "applied_torque_after_line": require_line(harness_lines, "applied_torque_after_arm_nm"),
        "effort_limit_line": require_line(harness_lines, "joint_effort_limit_arm_nm"),
        "velocity_limit_line": require_line(harness_lines, "joint_velocity_limit_arm_radps"),
        "command_gap_line": require_line(harness_lines, "command_target_face_gap_m"),
        "applied_gap_line": require_line(harness_lines, "applied_joint_target_fk_face_gap_m"),
        "actual_gap_line": require_line(harness_lines, "actual_tcp_face_gap_m"),
        "raw_delta_line": require_line(harness_lines, "raw_delta_arm_rad"),
        "clipped_delta_line": require_line(harness_lines, "clipped_delta_arm_rad"),
        "follow_arm_line": require_line(harness_lines, "direct_joint_follow_arm_rad"),
        "actual_step_arm_line": require_line(harness_lines, "actual_joint_step_arm_rad"),
    }

    basis = {
        "summary_line2": basis_lines[1] if len(basis_lines) >= 2 else "",
        "summary_line7": basis_lines[6] if len(basis_lines) >= 7 else "",
        "summary_line9": basis_lines[8] if len(basis_lines) >= 9 else "",
    }

    required_basis_tokens = (
        "command_final_face_gap_m=0.005999971",
        "applied_inside_rows=0",
        "actual_inside_rows=0",
        "arg=--reach_trace_detail_json",
        "behavior_absent=NO_OUTPUT_NO_CONTROL_CHANGE",
        "diffik_action_dataset=BLOCKED",
        "ppo_rl_training=BLOCKED",
        "roarm=BLOCKED",
    )
    basis_ok = all(any(token in line for line in basis_lines) for token in required_basis_tokens)
    schema_ok = all(line > 0 for line in checks.values())
    default_off_ok = "default=None" in basis["summary_line7"] and checks["detail_arg_line"] > 0
    no_action_teacher_ok = checks["action_teacher_false_line"] > 0 and checks["contains_action_false_line"] > 0
    verdict = (
        "READY_LOCAL_ONLY_DEFAULT_OFF_DETAIL_TRACE_PATCH"
        if basis_ok and schema_ok and default_off_ok and no_action_teacher_ok
        else "BLOCKED_CONTRACT_MISMATCH"
    )

    artifact = {
        "artifact_type": "cube10cm_tap_rl_reach_trace_detail_patch_contract_audit_v1",
        "local_static_audit_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "basis_summary": str(BASIS_SUMMARY.relative_to(REPO)),
        "harness": str(HARNESS.relative_to(REPO)),
        "basis": basis,
        "checks": checks,
        "basis_ok": basis_ok,
        "schema_ok": schema_ok,
        "default_off_ok": default_off_ok,
        "no_action_teacher_ok": no_action_teacher_ok,
        "verdict": verdict,
        "next_step": (
            "With explicit approval only, run one tiny repeat with the same near-face x240 h580 ep608 "
            "step-clipped built-in DiffIK contract plus reach_trace_json and reach_trace_detail_json."
        ),
    }

    OUT_JSON.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_reach_trace_detail_patch_contract_audit_v1 "
        "local_static_audit_only=YES gpu_runtime=NO dataset_generation=NO training=NO "
        "robot_control=NO ssh=NO b200=NO track_a=NO",
        "line2 basis_verified "
        f"basis_ok={basis_ok} current_state_line2_tokens=command_final_face_gap_0.005999971+applied0+actual0 "
        f"basis_summary={BASIS_SUMMARY.relative_to(REPO)}",
        "line3 default_off_contract "
        f"arg_line={checks['detail_arg_line']} default_off_ok={default_off_ok} "
        "behavior_absent=NO_OUTPUT_NO_CONTROL_CHANGE output=separate_detail_json control_change=NO",
        "line4 target_base_trace_fields "
        f"previous_state_line={checks['previous_target_state_line']} "
        f"previous_update_line={checks['previous_target_update_line']} "
        f"raw_delta_line={checks['raw_delta_line']} clipped_delta_line={checks['clipped_delta_line']} "
        f"joint_pos_target_after_line={checks['joint_pos_target_after_line']}",
        "line5 actuator_trace_fields "
        f"joint_vel_line={checks['joint_vel_after_line']} joint_acc_line={checks['joint_acc_after_line']} "
        f"computed_torque_line={checks['computed_torque_after_line']} "
        f"applied_torque_line={checks['applied_torque_after_line']} "
        f"effort_limit_line={checks['effort_limit_line']} velocity_limit_line={checks['velocity_limit_line']}",
        "line6 schema_guard "
        f"schema_ok={schema_ok} no_action_teacher_ok={no_action_teacher_ok} "
        f"writer_line={checks['detail_writer_line']} artifact_line={checks['detail_artifact_line']} "
        f"contains_action_fields=false action_teacher_dataset=false",
        "line7 next "
        "tiny_runtime_requires_explicit_approval=YES same_contract_only=nearface_x240_h580_ep608_step_clipped_builtin_diffik "
        "with_basic_and_detail_trace=YES contact_gate_relaxation=NOT_NEXT "
        "diffik_action_dataset=BLOCKED ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED",
        f"line8 verdict {verdict}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if verdict.startswith("READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
