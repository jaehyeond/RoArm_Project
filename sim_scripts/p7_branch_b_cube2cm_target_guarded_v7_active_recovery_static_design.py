#!/usr/bin/env python3
"""Static design contract for active target/support recovery after v6 FAIL.

This script reads the locally backed-up B200 v6 stdout/audit logs, verifies their
md5s, and turns the projected-block failure into a falsifiable next-design
contract. It does not launch Isaac, train PPO, collect rollouts, generate
datasets, insert constraints, use SurfaceGripper, tune gates, posewrite the
object, or claim success.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STDOUT = (
    REPO
    / "b200_backup_20260522_final"
    / "tmp_p7"
    / "p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out"
)
DEFAULT_AUDIT = (
    REPO
    / "b200_backup_20260522_final"
    / "tmp_p7"
    / "p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out"
)
EXPECTED_STDOUT_MD5 = "9a4f8825a88ee3c9d93d83e5b9a28b41"
EXPECTED_AUDIT_MD5 = "480a3355864937763eb665e086aadbb0"

CLOSE_TARGET_DEG = 26.0
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
PUSH_SPEED_GATE_MPS = 0.0050
MAX_PLAUSIBLE_COMPRESSION_M = 0.0030


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
    virtual_support: bool
    support_horizon_active: bool
    close_advance: bool
    close_hold: bool
    zero_backlog_hold: bool
    backlog_preserved_hold: bool
    safety_rollback: bool
    hard_safety_freeze: bool
    preemptive_recovery_needed: bool
    preemptive_recovery: bool
    recovery_ik_ok: bool
    recovery_step_m: float
    projected_target_margin_m: float
    projected_support_margin_m: float
    projected_advance_ok: bool
    command_converged: bool
    support_margin_ok: bool
    support_budget_ok: bool
    target_nonworsening: bool
    v3_safety_ok: bool
    hard_safety_ok: bool
    recovery_ready: bool
    target_error_recovered: bool
    advances_total: int
    holds_total: int
    zero_backlog_holds_total: int
    backlog_preserved_holds_total: int
    safety_rollbacks_total: int
    recovery_holds_total: int
    hard_safety_freezes_total: int
    recovery_writes_total: int
    recovery_ik_failures_total: int

    @property
    def target_margin_m(self) -> float:
        return TARGET_ERROR_GATE_M - self.target_error_m

    @property
    def support_margin_m(self) -> float:
        return COUNTER_SUPPORT_BUDGET_M - self.counter_gap_max_m


@dataclass(frozen=True)
class AggregateRow:
    line: int
    close_reached: bool
    attach_calls: int
    posewrite_calls: int
    zero_backlog_holds: int
    safety_rollbacks: int
    hard_safety_freezes: int
    recovery_writes: int
    recovery_ik_failures: int
    telemetry_only: bool
    success_claim: bool


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


def _parse_close_rows(path: Path) -> list[CloseRow]:
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
                virtual_support=_bool_field(line, "virtual_support"),
                support_horizon_active=_bool_field(line, "support_horizon_active"),
                close_advance=_bool_field(line, "target_guarded_close_advance"),
                close_hold=_bool_field(line, "target_guarded_close_hold"),
                zero_backlog_hold=_bool_field(line, "target_guarded_zero_backlog_hold"),
                backlog_preserved_hold=_bool_field(line, "target_guarded_backlog_preserved_hold"),
                safety_rollback=_bool_field(line, "target_guarded_v3_safety_rollback"),
                hard_safety_freeze=_bool_field(line, "target_guarded_v4_hard_safety_freeze"),
                preemptive_recovery_needed=_bool_field(
                    line, "target_guarded_v5_preemptive_recovery_needed"
                ),
                preemptive_recovery=_bool_field(line, "target_guarded_v5_preemptive_recovery"),
                recovery_ik_ok=_bool_field(line, "target_guarded_v5_recovery_ik_ok"),
                recovery_step_m=float(_field(line, "target_guarded_v5_recovery_step_m")),
                projected_target_margin_m=float(
                    _field(line, "target_guarded_v6_projected_target_margin_m")
                ),
                projected_support_margin_m=float(
                    _field(line, "target_guarded_v6_projected_support_margin_m")
                ),
                projected_advance_ok=_bool_field(line, "target_guarded_v6_projected_advance_ok"),
                command_converged=_bool_field(line, "target_guarded_command_converged"),
                support_margin_ok=_bool_field(line, "target_guarded_support_margin_ok"),
                support_budget_ok=_bool_field(line, "target_guarded_support_budget_ok"),
                target_nonworsening=_bool_field(line, "target_guarded_target_nonworsening"),
                v3_safety_ok=_bool_field(line, "target_guarded_v3_safety_ok"),
                hard_safety_ok=_bool_field(line, "target_guarded_v4_hard_safety_ok"),
                recovery_ready=_bool_field(line, "target_guarded_v4_recovery_ready"),
                target_error_recovered=_bool_field(line, "target_guarded_v4_target_error_recovered"),
                advances_total=int(_field(line, "target_guarded_close_advances_total")),
                holds_total=int(_field(line, "target_guarded_close_holds_total")),
                zero_backlog_holds_total=int(_field(line, "target_guarded_zero_backlog_holds_total")),
                backlog_preserved_holds_total=int(
                    _field(line, "target_guarded_backlog_preserved_holds_total")
                ),
                safety_rollbacks_total=int(_field(line, "target_guarded_safety_rollbacks_total")),
                recovery_holds_total=int(_field(line, "target_guarded_v4_recovery_holds_total")),
                hard_safety_freezes_total=int(
                    _field(line, "target_guarded_v4_hard_safety_freezes_total")
                ),
                recovery_writes_total=int(_field(line, "target_guarded_v5_preemptive_recovery_writes_total")),
                recovery_ik_failures_total=int(_field(line, "target_guarded_v5_recovery_ik_failures_total")),
            )
        )
    if not rows:
        raise ValueError(f"no close rows found in {path}")
    return rows


def _parse_aggregate(path: Path) -> AggregateRow:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "[cube2cm_runtime_jaw_telemetry] aggregate " not in line:
            continue
        return AggregateRow(
            line=line_no,
            close_reached=_bool_field(line, "close_reached"),
            attach_calls=int(_field(line, "attach_calls")),
            posewrite_calls=int(_field(line, "posewrite_calls")),
            zero_backlog_holds=int(_field(line, "target_guarded_zero_backlog_holds")),
            safety_rollbacks=int(_field(line, "target_guarded_safety_rollbacks")),
            hard_safety_freezes=int(_field(line, "target_guarded_v4_hard_safety_freezes")),
            recovery_writes=int(_field(line, "target_guarded_v5_preemptive_recovery_writes")),
            recovery_ik_failures=int(_field(line, "target_guarded_v5_recovery_ik_failures")),
            telemetry_only=_bool_field(line, "telemetry_only"),
            success_claim=_bool_field(line, "success_claim"),
        )
    raise ValueError(f"aggregate line missing in {path}")


def _audit_failure_lines(path: Path) -> list[int]:
    failures: list[int] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if " pass=NO" in line or "SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO" in line:
            failures.append(line_no)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--expected_stdout_md5", default=EXPECTED_STDOUT_MD5)
    parser.add_argument("--expected_audit_md5", default=EXPECTED_AUDIT_MD5)
    args = parser.parse_args()

    stdout_md5 = _md5(args.stdout)
    audit_md5 = _md5(args.audit)
    if args.expected_stdout_md5 and stdout_md5 != args.expected_stdout_md5:
        raise SystemExit(f"stdout md5 mismatch: got {stdout_md5}, expected {args.expected_stdout_md5}")
    if args.expected_audit_md5 and audit_md5 != args.expected_audit_md5:
        raise SystemExit(f"audit md5 mismatch: got {audit_md5}, expected {args.expected_audit_md5}")

    rows = _parse_close_rows(args.stdout)
    aggregate = _parse_aggregate(args.stdout)
    audit_failures = _audit_failure_lines(args.audit)

    projected_blocks = [row for row in rows if not row.projected_advance_ok and not row.hard_safety_freeze]
    if not projected_blocks:
        raise SystemExit("no pre-freeze projected block found")
    first_projected_block = projected_blocks[0]
    hard_freezes = [row for row in rows if row.hard_safety_freeze]
    if not hard_freezes:
        raise SystemExit("no hard freeze found; this is not the v6 failure log")
    first_hard_freeze = hard_freezes[0]
    first_target_violation = next(row for row in rows if row.target_error_m > TARGET_ERROR_GATE_M)

    block_to_hard_rows = [
        row for row in rows if first_projected_block.line <= row.line < first_hard_freeze.line
    ]
    recovery_rows_before_hard = [row for row in block_to_hard_rows if row.preemptive_recovery]
    last_recovery_before_hard = recovery_rows_before_hard[-1]
    final = rows[-1]

    target_growth_during_recovery = (
        last_recovery_before_hard.target_error_m - first_projected_block.target_error_m
    )
    support_gap_growth_during_recovery = (
        last_recovery_before_hard.counter_gap_max_m - first_projected_block.counter_gap_max_m
    )
    target_growth_to_freeze = first_hard_freeze.target_error_m - first_projected_block.target_error_m
    support_gap_growth_to_freeze = first_hard_freeze.counter_gap_max_m - first_projected_block.counter_gap_max_m
    close_remaining_deg = CLOSE_TARGET_DEG - final.gripper_q_deg

    print("[cube2cm_v7_active_recovery_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_v7_active_recovery_static] "
        "dataset_generation=NO rollout_collection=NO ppo=NO hold_lift=NO "
        "constraints=NO surface_gripper=NO posewrite=NO attach=NO transport_release=NO "
        "gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_v7_active_recovery_static] evidence "
        f"stdout_log={args.stdout} md5={stdout_md5} expected_md5={args.expected_stdout_md5} "
        f"audit_log={args.audit} audit_md5={audit_md5} expected_audit_md5={args.expected_audit_md5}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v6_first_projected_block "
        f"line={first_projected_block.line} step={first_projected_block.step} "
        f"projected_target_margin_m={first_projected_block.projected_target_margin_m:.6f} "
        f"projected_support_margin_m={first_projected_block.projected_support_margin_m:.6f} "
        f"target_error_m={first_projected_block.target_error_m:.6f} "
        f"target_margin_m={first_projected_block.target_margin_m:.6f} "
        f"counter_gap_max_m={first_projected_block.counter_gap_max_m:.6f} "
        f"support_margin_m={first_projected_block.support_margin_m:.6f} "
        f"recovery_write={_yes(first_projected_block.preemptive_recovery)} "
        f"recovery_ik_ok={_yes(first_projected_block.recovery_ik_ok)}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v6_last_recovery_before_freeze "
        f"line={last_recovery_before_hard.line} step={last_recovery_before_hard.step} "
        f"projected_advance_ok={_yes(last_recovery_before_hard.projected_advance_ok)} "
        f"support_margin_ok={_yes(last_recovery_before_hard.support_margin_ok)} "
        f"target_error_recovered={_yes(last_recovery_before_hard.target_error_recovered)} "
        f"target_error_m={last_recovery_before_hard.target_error_m:.6f} "
        f"target_margin_m={last_recovery_before_hard.target_margin_m:.6f} "
        f"counter_gap_max_m={last_recovery_before_hard.counter_gap_max_m:.6f} "
        f"support_margin_m={last_recovery_before_hard.support_margin_m:.6f} "
        f"recovery_step_m={last_recovery_before_hard.recovery_step_m:.6f}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v6_first_hard_freeze "
        f"line={first_hard_freeze.line} step={first_hard_freeze.step} "
        f"target_error_m={first_hard_freeze.target_error_m:.6f} "
        f"target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_gap_max_m={first_hard_freeze.counter_gap_max_m:.6f} "
        f"support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"support_margin_m={first_hard_freeze.support_margin_m:.6f} "
        f"object_speed_mps={first_hard_freeze.object_speed_mps:.6f} "
        f"speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f} "
        f"one_sided_push={_yes(first_hard_freeze.one_sided_push)} "
        f"recovery_write={_yes(first_hard_freeze.preemptive_recovery)}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v6_first_target_violation "
        f"line={first_target_violation.line} step={first_target_violation.step} "
        f"target_error_m={first_target_violation.target_error_m:.6f} "
        f"target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_gap_max_m={first_target_violation.counter_gap_max_m:.6f}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v6_failure_mechanism "
        f"pre_freeze_recovery_rows={len(recovery_rows_before_hard)} "
        f"target_growth_during_recovery_m={target_growth_during_recovery:.6f} "
        f"support_gap_growth_during_recovery_m={support_gap_growth_during_recovery:.6f} "
        f"target_growth_projected_block_to_freeze_m={target_growth_to_freeze:.6f} "
        f"support_gap_growth_projected_block_to_freeze_m={support_gap_growth_to_freeze:.6f} "
        f"aggregate_line={aggregate.line} close_reached={_yes(aggregate.close_reached)} "
        f"zero_backlog_holds={aggregate.zero_backlog_holds} safety_rollbacks={aggregate.safety_rollbacks} "
        f"recovery_writes={aggregate.recovery_writes} recovery_ik_failures={aggregate.recovery_ik_failures} "
        f"hard_freezes={aggregate.hard_safety_freezes} attach_calls={aggregate.attach_calls} "
        f"posewrite_calls={aggregate.posewrite_calls}"
    )
    print(
        "[cube2cm_v7_active_recovery_static] audit_truth "
        f"audit_fail_lines={audit_failures} criteria_pass=NO "
        "first_support_failure_runtime_line=398 first_target_failure_runtime_line=399"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v7_required_design "
        "projected_block_must_enter_active_recovery=YES "
        "target_only_overshoot_recovery_rejected=YES "
        "candidate_selector=FINITE_DIFFERENCE_TCP_SWEEP_WITH_CURRENT_OBJECT_POSE "
        "selector_objective=maximize_min_fixed_target_and_support_margins "
        "must_reduce_counter_gap_before_next_close_advance=YES "
        "must_keep_target_error_inside_fixed_gate=YES "
        "recovery_writes_robot_joint_targets_only=YES "
        "object_posewrite_forbidden=YES attach_forbidden=YES constraints_forbidden=YES surface_gripper_forbidden=YES"
    )
    print(
        "[cube2cm_v7_active_recovery_static] v7_advance_exit_contract "
        f"fixed_target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"fixed_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"support_horizon_m={MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        "zero_backlog_holds_forbidden=YES safety_rollbacks_forbidden=YES "
        "backlog_preserved_holds_required=YES hard_freezes_must_remain_zero=YES "
        "close26_only_first=YES immediate_posthoc_audit_required=YES "
        f"final_close_remaining_deg={close_remaining_deg:.3f}"
    )
    print("[cube2cm_v7_active_recovery_static] RUNTIME_READY=NO STATIC_DESIGN_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
