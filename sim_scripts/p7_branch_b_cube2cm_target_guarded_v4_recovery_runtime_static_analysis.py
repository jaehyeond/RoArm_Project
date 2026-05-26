#!/usr/bin/env python3
"""Static posthoc attribution for target-guarded v4 recovery close_26.

This script reads the already-produced B200 stdout/audit logs and quantifies
why target-guarded v4 still failed. It does not launch Isaac, train, generate
data, insert constraints, use a SurfaceGripper, tune gates, or claim success.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STDOUT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out")
DEFAULT_AUDIT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out")
EXPECTED_STDOUT_MD5 = "fe6a733727a6eeb288c6c6464c178af1"
EXPECTED_AUDIT_MD5 = "47f4ec7b78298fde0a46ac57105a6e6c"

CLOSE_TARGET_DEG = 26.0
CLOSE_STEPS = 45
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
V4_RECOVERY_TARGET_ERROR_M = 0.0024
PUSH_SPEED_GATE_MPS = 0.0050

V3_FINAL_GRIPPER_Q_DEG = 7.144


@dataclass(frozen=True)
class CloseStep:
    line: int
    step: int
    gripper_q_deg: float
    gripper_command_deg: float
    command_backlog_deg: float
    target_error_m: float
    object_speed_mps: float
    counter_gap_max_m: float
    one_sided_push: bool
    support_horizon_active: bool
    close_advance: bool
    close_hold: bool
    zero_backlog_hold: bool
    backlog_preserved_hold: bool
    safety_rollback: bool
    v4_recovery_hold: bool
    v4_hard_safety_freeze: bool
    support_budget_ok: bool
    target_nonworsening: bool
    v4_hard_safety_ok: bool
    v4_recovery_ready: bool
    v4_target_error_recovered: bool
    advances_total: int
    holds_total: int
    zero_backlog_holds_total: int
    backlog_preserved_holds_total: int
    safety_rollbacks_total: int
    v4_recovery_holds_total: int
    v4_hard_safety_freezes_total: int


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


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


def _float_field(line: str, name: str) -> float:
    return float(_field(line, name))


def _int_field(line: str, name: str) -> int:
    return int(_field(line, name))


def _vector_max(line: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}=\(\[([^\]]+)\]\)", line)
    if not match:
        raise ValueError(f"missing vector field {name!r} in line: {line[:180]}")
    return max(float(part.strip()) for part in match.group(1).split(","))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_close_steps(path: Path) -> list[CloseStep]:
    rows: list[CloseStep] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "phase=close" not in line:
            continue
        rows.append(
            CloseStep(
                line=line_no,
                step=_int_field(line, "step"),
                gripper_q_deg=_float_field(line, "gripper_q_deg"),
                gripper_command_deg=_float_field(line, "gripper_command_deg"),
                command_backlog_deg=_float_field(line, "target_guarded_command_backlog_deg"),
                target_error_m=_float_field(line, "target_error_m"),
                object_speed_mps=_float_field(line, "object_speed_mps"),
                counter_gap_max_m=_vector_max(line, "counter_gap_obj_m"),
                one_sided_push=_bool_field(line, "one_sided_push"),
                support_horizon_active=_bool_field(line, "support_horizon_active"),
                close_advance=_bool_field(line, "target_guarded_close_advance"),
                close_hold=_bool_field(line, "target_guarded_close_hold"),
                zero_backlog_hold=_bool_field(line, "target_guarded_zero_backlog_hold"),
                backlog_preserved_hold=_bool_field(line, "target_guarded_backlog_preserved_hold"),
                safety_rollback=_bool_field(line, "target_guarded_v3_safety_rollback"),
                v4_recovery_hold=_bool_field(line, "target_guarded_v4_recovery_hold"),
                v4_hard_safety_freeze=_bool_field(line, "target_guarded_v4_hard_safety_freeze"),
                support_budget_ok=_bool_field(line, "target_guarded_support_budget_ok"),
                target_nonworsening=_bool_field(line, "target_guarded_target_nonworsening"),
                v4_hard_safety_ok=_bool_field(line, "target_guarded_v4_hard_safety_ok"),
                v4_recovery_ready=_bool_field(line, "target_guarded_v4_recovery_ready"),
                v4_target_error_recovered=_bool_field(line, "target_guarded_v4_target_error_recovered"),
                advances_total=_int_field(line, "target_guarded_close_advances_total"),
                holds_total=_int_field(line, "target_guarded_close_holds_total"),
                zero_backlog_holds_total=_int_field(line, "target_guarded_zero_backlog_holds_total"),
                backlog_preserved_holds_total=_int_field(
                    line, "target_guarded_backlog_preserved_holds_total"
                ),
                safety_rollbacks_total=_int_field(line, "target_guarded_safety_rollbacks_total"),
                v4_recovery_holds_total=_int_field(line, "target_guarded_v4_recovery_holds_total"),
                v4_hard_safety_freezes_total=_int_field(
                    line, "target_guarded_v4_hard_safety_freezes_total"
                ),
            )
        )
    if not rows:
        raise ValueError(f"no close rows found in {path}")
    return rows


def _hard_freeze_reasons(row: CloseStep) -> list[str]:
    reasons: list[str] = []
    if row.target_error_m > TARGET_ERROR_GATE_M:
        reasons.append("target_error_fixed_gate")
    if row.object_speed_mps > PUSH_SPEED_GATE_MPS:
        reasons.append("push_speed")
    if row.one_sided_push:
        reasons.append("one_sided_push")
    if not row.support_budget_ok:
        reasons.append("support_budget")
    if not row.support_horizon_active:
        reasons.append("support_horizon")
    return reasons


def _audit_fail_lines(path: Path) -> list[tuple[int, str]]:
    failures: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if " pass=NO" in line or "SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO" in line:
            failures.append((line_no, line))
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

    rows = parse_close_steps(args.stdout)
    advances = [row for row in rows if row.close_advance]
    holds = [row for row in rows if row.close_hold]
    zero_backlog_holds = [row for row in holds if row.zero_backlog_hold]
    preserved_holds = [row for row in holds if row.backlog_preserved_hold]
    safety_rollbacks = [row for row in rows if row.safety_rollback]
    recovery_holds = [row for row in rows if row.v4_recovery_hold]
    hard_freezes = [row for row in rows if row.v4_hard_safety_freeze]
    target_worsening_recovery_holds = [
        row for row in recovery_holds if not row.target_nonworsening
    ]
    first_recovery_hold = recovery_holds[0]
    first_target_worsening_recovery = target_worsening_recovery_holds[0]
    first_hard_freeze = hard_freezes[0]
    peak_target_error = max(rows, key=lambda row: row.target_error_m)
    peak_support_gap = max(rows, key=lambda row: row.counter_gap_max_m)
    final = rows[-1]
    audit_failures = _audit_fail_lines(args.audit)

    print("[cube2cm_target_guarded_v4_recovery_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_target_guarded_v4_recovery_static] "
        "dataset_generation=NO hold_lift=NO constraints=NO surface_gripper=NO "
        "transport_release=NO gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] "
        f"stdout_log={args.stdout} md5={stdout_md5} expected_md5={args.expected_stdout_md5} "
        f"audit_log={args.audit} audit_md5={audit_md5} expected_audit_md5={args.expected_audit_md5}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] schedule_totals "
        f"close_steps={len(rows)} close_steps_expected={CLOSE_STEPS} "
        f"advances={len(advances)} holds={len(holds)} zero_backlog_holds={len(zero_backlog_holds)} "
        f"backlog_preserved_holds={len(preserved_holds)} safety_rollbacks={len(safety_rollbacks)} "
        f"v4_recovery_holds={len(recovery_holds)} v4_hard_safety_freezes={len(hard_freezes)} "
        f"advance_steps={[row.step for row in advances]}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] first_recovery_hold "
        f"line={first_recovery_hold.line} step={first_recovery_hold.step} "
        f"gripper_q_deg={first_recovery_hold.gripper_q_deg:.3f} "
        f"gripper_command_deg={first_recovery_hold.gripper_command_deg:.3f} "
        f"command_backlog_deg={first_recovery_hold.command_backlog_deg:.3f} "
        f"target_error_m={first_recovery_hold.target_error_m:.6f} "
        f"counter_gap_max_m={first_recovery_hold.counter_gap_max_m:.6f}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] line385_target_worsening_recovery "
        f"line={first_target_worsening_recovery.line} step={first_target_worsening_recovery.step} "
        f"target_nonworsening={_yes(first_target_worsening_recovery.target_nonworsening)} "
        f"recovery_hold={_yes(first_target_worsening_recovery.v4_recovery_hold)} "
        f"hard_freeze={_yes(first_target_worsening_recovery.v4_hard_safety_freeze)} "
        f"target_error_m={first_target_worsening_recovery.target_error_m:.6f} "
        f"counter_gap_max_m={first_target_worsening_recovery.counter_gap_max_m:.6f}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] first_hard_safety_freeze "
        f"line={first_hard_freeze.line} step={first_hard_freeze.step} "
        f"gripper_q_deg={first_hard_freeze.gripper_q_deg:.3f} "
        f"gripper_command_deg={first_hard_freeze.gripper_command_deg:.3f} "
        f"command_backlog_deg={first_hard_freeze.command_backlog_deg:.3f} "
        f"target_error_m={first_hard_freeze.target_error_m:.6f} "
        f"target_error_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_gap_max_m={first_hard_freeze.counter_gap_max_m:.6f} "
        f"counter_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"object_speed_mps={first_hard_freeze.object_speed_mps:.6f} "
        f"one_sided_push={_yes(first_hard_freeze.one_sided_push)} "
        f"support_horizon_active={_yes(first_hard_freeze.support_horizon_active)} "
        f"reasons={'+'.join(_hard_freeze_reasons(first_hard_freeze))}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] peak_and_final "
        f"peak_target_line={peak_target_error.line} peak_target_step={peak_target_error.step} "
        f"peak_target_error_m={peak_target_error.target_error_m:.6f} "
        f"peak_support_line={peak_support_gap.line} peak_support_step={peak_support_gap.step} "
        f"peak_counter_gap_max_m={peak_support_gap.counter_gap_max_m:.6f} "
        f"final_line={final.line} final_step={final.step} "
        f"final_gripper_q_deg={final.gripper_q_deg:.3f} "
        f"final_gripper_command_deg={final.gripper_command_deg:.3f} "
        f"close_remaining_deg={CLOSE_TARGET_DEG - final.gripper_q_deg:.3f} "
        f"final_target_error_m={final.target_error_m:.6f} "
        f"final_counter_gap_max_m={final.counter_gap_max_m:.6f}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] v3_to_v4_delta "
        f"v3_final_gripper_q_deg={V3_FINAL_GRIPPER_Q_DEG:.3f} "
        f"v4_final_gripper_q_deg={final.gripper_q_deg:.3f} "
        f"delta_final_gripper_q_deg={final.gripper_q_deg - V3_FINAL_GRIPPER_Q_DEG:.3f} "
        "v3_safety_rollbacks=34 v4_safety_rollbacks=0 "
        f"v4_hard_safety_freezes={len(hard_freezes)}"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] audit_failures "
        + " | ".join(f"line={line_no}:{line.split('criterion name=')[-1]}" for line_no, line in audit_failures)
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] attribution "
        "zero_backlog_starvation=FIXED "
        "safety_rollback=FIXED "
        "primary=target_support_hard_gate_freeze_after_recovery_hold "
        "recovery_hold_alone=INSUFFICIENT "
        "fixed_gate_relaxation=REJECT "
        "structural_contact_compatible_target_support_recovery_required=YES"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] next_mechanism_requirements "
        "do_not_rerun_v4=YES "
        "preserve_fixed_close26_audit_gates=YES "
        "preserve_zero_backlog_holds_zero=YES "
        "preserve_safety_rollbacks_zero=YES "
        "must_recover_target_and_counter_support_before_hard_freeze=YES"
    )
    print("[cube2cm_target_guarded_v4_recovery_static] TARGET_GUARDED_V4_RECOVERY_RUNTIME_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
