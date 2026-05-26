#!/usr/bin/env python3
"""Static v5 design contract for Track A contact-RL Stage 0.

The latest v4 runtime proves that recovery-hold-only scheduling waits until the
fixed target/support gates have already been breached. This script quantifies the
last safe pre-breach step and turns it into a falsifiable v5 contract for a
no-attach contact primitive or contact-RL preflight environment.

It does not launch Isaac, train PPO, collect rollouts, generate datasets, add
constraints, use a SurfaceGripper, tune gates, or claim success.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STDOUT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out")
EXPECTED_STDOUT_MD5 = "fe6a733727a6eeb288c6c6464c178af1"

CLOSE_TARGET_DEG = 26.0
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
PUSH_SPEED_GATE_MPS = 0.0050

PREEMPT_TARGET_MARGIN_M = 0.00020
PREEMPT_SUPPORT_MARGIN_M = 0.00010


@dataclass(frozen=True)
class CloseRow:
    line: int
    step: int
    gripper_q_deg: float
    gripper_command_deg: float
    command_backlog_deg: float
    target_error_m: float
    counter_gap_max_m: float
    object_speed_mps: float
    one_sided_push: bool
    support_horizon_active: bool
    support_budget_ok: bool
    target_nonworsening: bool
    v4_recovery_hold: bool
    v4_hard_safety_freeze: bool
    advances_total: int
    holds_total: int
    hard_freezes_total: int

    @property
    def target_margin_m(self) -> float:
        return TARGET_ERROR_GATE_M - self.target_error_m

    @property
    def support_margin_m(self) -> float:
        return COUNTER_SUPPORT_BUDGET_M - self.counter_gap_max_m


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_rows(path: Path) -> list[CloseRow]:
    rows: list[CloseRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "phase=close" not in line:
            continue
        rows.append(
            CloseRow(
                line=line_no,
                step=int(_field(line, "step")),
                gripper_q_deg=float(_field(line, "gripper_q_deg")),
                gripper_command_deg=float(_field(line, "gripper_command_deg")),
                command_backlog_deg=float(_field(line, "target_guarded_command_backlog_deg")),
                target_error_m=float(_field(line, "target_error_m")),
                counter_gap_max_m=_vector_max(line, "counter_gap_obj_m"),
                object_speed_mps=float(_field(line, "object_speed_mps")),
                one_sided_push=_bool_field(line, "one_sided_push"),
                support_horizon_active=_bool_field(line, "support_horizon_active"),
                support_budget_ok=_bool_field(line, "target_guarded_support_budget_ok"),
                target_nonworsening=_bool_field(line, "target_guarded_target_nonworsening"),
                v4_recovery_hold=_bool_field(line, "target_guarded_v4_recovery_hold"),
                v4_hard_safety_freeze=_bool_field(line, "target_guarded_v4_hard_safety_freeze"),
                advances_total=int(_field(line, "target_guarded_close_advances_total")),
                holds_total=int(_field(line, "target_guarded_close_holds_total")),
                hard_freezes_total=int(_field(line, "target_guarded_v4_hard_safety_freezes_total")),
            )
        )
    if not rows:
        raise ValueError(f"no close rows found in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--expected_stdout_md5", default=EXPECTED_STDOUT_MD5)
    args = parser.parse_args()

    stdout_md5 = _md5(args.stdout)
    if args.expected_stdout_md5 and stdout_md5 != args.expected_stdout_md5:
        raise SystemExit(f"stdout md5 mismatch: got {stdout_md5}, expected {args.expected_stdout_md5}")

    rows = _parse_rows(args.stdout)
    hard_freezes = [row for row in rows if row.v4_hard_safety_freeze]
    recovery_holds = [row for row in rows if row.v4_recovery_hold]
    first_hard = hard_freezes[0]
    pre_hard = rows[rows.index(first_hard) - 1]
    final = rows[-1]
    preempt_rows = [
        row
        for row in rows
        if (
            row.target_margin_m <= PREEMPT_TARGET_MARGIN_M
            or row.support_margin_m <= PREEMPT_SUPPORT_MARGIN_M
        )
        and not row.v4_hard_safety_freeze
    ]
    first_preempt = preempt_rows[0]
    close_remaining_deg = CLOSE_TARGET_DEG - final.gripper_q_deg

    print("[cube2cm_contact_rl_v5_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_contact_rl_v5_static] "
        "dataset_generation=NO rollout_collection=NO constraints=NO surface_gripper=NO "
        "transport_release=NO gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_contact_rl_v5_static] "
        f"stdout_log={args.stdout} md5={stdout_md5} expected_md5={args.expected_stdout_md5}"
    )
    print(
        "[cube2cm_contact_rl_v5_static] v4_last_safe_before_freeze "
        f"line={pre_hard.line} step={pre_hard.step} "
        f"target_error_m={pre_hard.target_error_m:.6f} target_margin_m={pre_hard.target_margin_m:.6f} "
        f"counter_gap_max_m={pre_hard.counter_gap_max_m:.6f} support_margin_m={pre_hard.support_margin_m:.6f} "
        f"recovery_hold={_yes(pre_hard.v4_recovery_hold)} hard_freeze={_yes(pre_hard.v4_hard_safety_freeze)} "
        f"gripper_q_deg={pre_hard.gripper_q_deg:.3f} command_deg={pre_hard.gripper_command_deg:.3f}"
    )
    print(
        "[cube2cm_contact_rl_v5_static] v4_first_freeze "
        f"line={first_hard.line} step={first_hard.step} "
        f"target_error_m={first_hard.target_error_m:.6f} target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_gap_max_m={first_hard.counter_gap_max_m:.6f} support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"object_speed_mps={first_hard.object_speed_mps:.6f} speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f} "
        f"one_sided_push={_yes(first_hard.one_sided_push)} support_horizon_active={_yes(first_hard.support_horizon_active)}"
    )
    print(
        "[cube2cm_contact_rl_v5_static] preemptive_trigger "
        f"first_preempt_line={first_preempt.line} first_preempt_step={first_preempt.step} "
        f"target_margin_m={first_preempt.target_margin_m:.6f} "
        f"support_margin_m={first_preempt.support_margin_m:.6f} "
        f"target_margin_trigger_m={PREEMPT_TARGET_MARGIN_M:.6f} "
        f"support_margin_trigger_m={PREEMPT_SUPPORT_MARGIN_M:.6f} "
        "must_recover_before_line391=YES"
    )
    print(
        "[cube2cm_contact_rl_v5_static] final_plateau "
        f"line={final.line} step={final.step} final_gripper_q_deg={final.gripper_q_deg:.3f} "
        f"final_command_deg={final.gripper_command_deg:.3f} close_remaining_deg={close_remaining_deg:.3f} "
        f"final_target_error_m={final.target_error_m:.6f} final_counter_gap_max_m={final.counter_gap_max_m:.6f} "
        f"hard_freezes_total={final.hard_freezes_total} recovery_holds={len(recovery_holds)}"
    )
    print(
        "[cube2cm_contact_rl_v5_static] v5_required_mechanism "
        "hold_only_rejected=YES "
        "preemptive_target_support_recovery_required=YES "
        "recovery_must_write_robot_joint_targets_only=YES "
        "object_posewrite_forbidden=YES attach_forbidden=YES "
        "zero_backlog_holds_forbidden=YES safety_rollbacks_forbidden=YES "
        "fixed_target_support_gates_unchanged=YES"
    )
    print(
        "[cube2cm_contact_rl_v5_static] contact_rl_stage0_gate "
        "random_action_sanity_allowed_only_after_no_attach_env_exists=YES "
        "ppo_training_blocked_until_v5_or_no_attach_env_static_readiness=YES "
        "expert_rollout_dataset_blocked_until_close26_and_hold_lift_pass=YES"
    )
    print("[cube2cm_contact_rl_v5_static] CONTACT_RL_V5_STATIC_DESIGN_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
