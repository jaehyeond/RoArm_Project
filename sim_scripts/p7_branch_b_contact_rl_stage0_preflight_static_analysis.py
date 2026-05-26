#!/usr/bin/env python3
"""Static preflight for the Track A contact-RL expert-data plan.

This does not launch Isaac, train PPO, collect rollouts, generate datasets, add
constraints, use a SurfaceGripper, tune gates, or claim success. It verifies the
latest no-attach close_26 evidence and checks whether the existing PPO entry
points can produce a Track A-valid contact expert.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STDOUT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out")
DEFAULT_AUDIT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out")
EXPECTED_STDOUT_MD5 = "fe6a733727a6eeb288c6c6464c178af1"
EXPECTED_AUDIT_MD5 = "47f4ec7b78298fde0a46ac57105a6e6c"

TRAIN_PPO = REPO / "roarm_rl" / "train_ppo.py"
PICK_ENV = REPO / "roarm_rl" / "roarm_pick_env.py"
STACK_ENV = REPO / "roarm_rl" / "roarm_stack_env.py"
RUNTIME_PROBE = REPO / "sim_scripts" / "p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py"

CLOSE_TARGET_DEG = 26.0
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020


@dataclass(frozen=True)
class Evidence:
    stdout_md5: str
    audit_md5: str
    line37: str
    line391: str
    line421: str
    line423: str
    audit_line16: str
    audit_line28: str
    audit_line54: str
    final_gripper_deg: float
    final_command_deg: float
    final_target_error_m: float
    final_counter_gap_m: float
    hard_freezes: int
    close_reached: bool
    attach_calls: int
    posewrite_calls: int


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line(path: Path, line_no: int) -> str:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if idx == line_no:
            return line
    raise ValueError(f"{path} has no line {line_no}")


def _field(line: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise ValueError(f"missing field {name!r} in line: {line[:180]}")
    return match.group(1)


def _bool_field(line: str, name: str) -> bool:
    value = _field(line, name)
    if value == "YES":
        return True
    if value == "NO":
        return False
    raise ValueError(f"field {name!r} is not YES/NO: {value!r}")


def _vector_max(line: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}=\(\[([^\]]+)\]\)", line)
    if not match:
        raise ValueError(f"missing vector field {name!r} in line: {line[:180]}")
    return max(float(part.strip()) for part in match.group(1).split(","))


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def _parse_evidence(stdout: Path, audit: Path, expected_stdout_md5: str, expected_audit_md5: str) -> Evidence:
    stdout_md5 = _md5(stdout)
    audit_md5 = _md5(audit)
    if expected_stdout_md5 and stdout_md5 != expected_stdout_md5:
        raise SystemExit(f"stdout md5 mismatch: got {stdout_md5}, expected {expected_stdout_md5}")
    if expected_audit_md5 and audit_md5 != expected_audit_md5:
        raise SystemExit(f"audit md5 mismatch: got {audit_md5}, expected {expected_audit_md5}")

    line37 = _line(stdout, 37)
    line391 = _line(stdout, 391)
    line421 = _line(stdout, 421)
    line423 = _line(stdout, 423)
    audit_line16 = _line(audit, 16)
    audit_line28 = _line(audit, 28)
    audit_line54 = _line(audit, 54)
    return Evidence(
        stdout_md5=stdout_md5,
        audit_md5=audit_md5,
        line37=line37,
        line391=line391,
        line421=line421,
        line423=line423,
        audit_line16=audit_line16,
        audit_line28=audit_line28,
        audit_line54=audit_line54,
        final_gripper_deg=float(_field(line421, "gripper_q_deg")),
        final_command_deg=float(_field(line421, "gripper_command_deg")),
        final_target_error_m=float(_field(line421, "target_error_m")),
        final_counter_gap_m=_vector_max(line421, "counter_gap_obj_m"),
        hard_freezes=int(_field(line423, "target_guarded_v4_hard_safety_freezes")),
        close_reached=_bool_field(line423, "close_reached"),
        attach_calls=int(_field(line423, "attach_calls")),
        posewrite_calls=int(_field(line423, "posewrite_calls")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--expected_stdout_md5", default=EXPECTED_STDOUT_MD5)
    parser.add_argument("--expected_audit_md5", default=EXPECTED_AUDIT_MD5)
    args = parser.parse_args()

    evidence = _parse_evidence(args.stdout, args.audit, args.expected_stdout_md5, args.expected_audit_md5)
    pick_uses_attach = _contains(PICK_ENV, "kinematic attach") and _contains(PICK_ENV, "write_root_pose_to_sim")
    stack_uses_attach = _contains(STACK_ENV, "_update_grasp_attach") and _contains(STACK_ENV, "write_root_pose_to_sim")
    ppo_uses_registered_envs = _contains(TRAIN_PPO, "RoArm-Pick-Direct-v0") and _contains(
        TRAIN_PPO, "RoArm-Stack-Direct-v0"
    )
    probe_blocks_posewrite = _contains(RUNTIME_PROBE, "hidden_kinematic_posewrite_allowed=NO") and _contains(
        RUNTIME_PROBE, "attach_stats"
    )

    v4_close_blocked = (
        not evidence.close_reached
        and evidence.hard_freezes > 0
        and evidence.final_target_error_m > TARGET_ERROR_GATE_M
        and evidence.final_counter_gap_m > COUNTER_SUPPORT_BUDGET_M
        and evidence.attach_calls == 0
        and evidence.posewrite_calls == 0
    )
    existing_rl_env_track_a_valid = not (pick_uses_attach or stack_uses_attach)
    direct_b200_ppo_now_ok = existing_rl_env_track_a_valid and not v4_close_blocked
    close_remaining_deg = CLOSE_TARGET_DEG - evidence.final_gripper_deg

    print("[contact_rl_stage0_preflight] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[contact_rl_stage0_preflight] "
        "dataset_generation=NO rollout_collection=NO constraints=NO surface_gripper=NO "
        "gate_tuning=NO success_claim=NO"
    )
    print(
        "[contact_rl_stage0_preflight] verified_logs "
        f"stdout={args.stdout} md5={evidence.stdout_md5} expected_md5={args.expected_stdout_md5} "
        f"audit={args.audit} audit_md5={evidence.audit_md5} expected_audit_md5={args.expected_audit_md5}"
    )
    print(
        "[contact_rl_stage0_preflight] v4_runtime_scope "
        f"line37_diagnostic_only={'diagnostic_only=YES' in evidence.line37} "
        f"line37_close_26_only={'close_26_only=YES' in evidence.line37} "
        f"line37_training_no={'p7_training=NO' in evidence.line37} "
        f"line37_posewrite_no={'hidden_kinematic_posewrite_allowed=NO' in evidence.line37}"
    )
    print(
        "[contact_rl_stage0_preflight] v4_blocker "
        f"line391_target_error_m={float(_field(evidence.line391, 'target_error_m')):.6f} "
        f"target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"line391_counter_gap_m={_vector_max(evidence.line391, 'counter_gap_obj_m'):.6f} "
        f"support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"final_gripper_deg={evidence.final_gripper_deg:.3f} "
        f"final_command_deg={evidence.final_command_deg:.3f} "
        f"close_remaining_deg={close_remaining_deg:.3f} "
        f"hard_freezes={evidence.hard_freezes} close_reached={_yes(evidence.close_reached)}"
    )
    print(
        "[contact_rl_stage0_preflight] audit_blocker "
        f"audit_line16_close_reached_pass_no={'pass=NO' in evidence.audit_line16} "
        f"audit_line28_hard_freezes_pass_no={'pass=NO' in evidence.audit_line28} "
        f"audit_line54_pass_no={'SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO' in evidence.audit_line54}"
    )
    print(
        "[contact_rl_stage0_preflight] existing_ppo_env_semantics "
        f"train_ppo_uses_registered_envs={_yes(ppo_uses_registered_envs)} "
        f"pick_env_kinematic_attach={_yes(pick_uses_attach)} "
        f"stack_env_kinematic_attach={_yes(stack_uses_attach)} "
        f"runtime_probe_no_attach_audit_surface={_yes(probe_blocks_posewrite)} "
        f"existing_rl_env_track_a_valid={_yes(existing_rl_env_track_a_valid)}"
    )
    print(
        "[contact_rl_stage0_preflight] user_four_stage_flow "
        "rl_to_expert_to_rollout_to_dataset=VALID_AFTER_STAGE0 "
        f"direct_b200_ppo_now_ok={_yes(direct_b200_ppo_now_ok)} "
        "reason=existing_ppo_envs_are_attach_based_and_latest_no_attach_close26_is_blocked"
    )
    print(
        "[contact_rl_stage0_preflight] required_stage0_contract "
        "new_or_patched_rl_env_no_attach_posewrite=YES "
        "reward_success_must_require_close26_contact_gate=YES "
        "reward_success_must_require_hold_lift_after_close=YES "
        "success_must_not_read_kinematic_grasp_latch=YES "
        "small_random_sanity_before_ppo=YES "
        "ppo_training_requires_separate_approval=YES"
    )
    print(
        "[contact_rl_stage0_preflight] next_code_step "
        "build_no_attach_contact_rl_env_or_v5_contact_close_gate=YES "
        "do_not_use_roarm_pick_stack_default_env_as_track_a_expert=YES"
    )
    print("[contact_rl_stage0_preflight] CONTACT_RL_STAGE0_PREFLIGHT_STATIC_DONE=YES")
    return 0 if not direct_b200_ppo_now_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
