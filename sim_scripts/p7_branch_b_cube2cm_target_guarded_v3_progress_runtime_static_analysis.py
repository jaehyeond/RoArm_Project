#!/usr/bin/env python3
"""Static posthoc attribution for target-guarded v3 close_26.

This script reads the already-produced B200 stdout/audit logs and quantifies
why target-guarded v3 still failed. It does not launch Isaac, train, generate
data, insert constraints, use a SurfaceGripper, tune gates, or claim success.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STDOUT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out")
DEFAULT_AUDIT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out")
EXPECTED_STDOUT_MD5 = "5f2d1a626edcdccce8086fafd321c9af"
EXPECTED_AUDIT_MD5 = "ca60c09b03a156c85197e34ec7b28bb5"

CLOSE_TARGET_DEG = 26.0
CLOSE_STEPS = 45
TARGET_ERROR_DESIGN_LIMIT_M = 0.0027
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
MAX_PLAUSIBLE_COMPRESSION_M = 0.0030
PUSH_SPEED_GATE_MPS = 0.0050


@dataclass(frozen=True)
class CloseStep:
    line: int
    step: int
    gripper_q_deg: float
    gripper_command_deg: float
    gripper_command_err_deg: float
    target_error_m: float
    object_speed_mps: float
    counter_gap_max_m: float
    virtual_compression_gap_max_m: float
    one_sided_push: bool
    virtual_support: bool
    support_horizon_active: bool
    close_advance: bool
    close_hold: bool
    zero_backlog_hold: bool
    backlog_preserved_hold: bool
    safety_rollback: bool
    support_budget_ok: bool
    v3_safety_ok: bool
    advances_total: int
    holds_total: int
    zero_backlog_holds_total: int
    backlog_preserved_holds_total: int
    safety_rollbacks_total: int


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
    values = [float(part.strip()) for part in match.group(1).split(",")]
    return max(values)


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
                gripper_command_err_deg=_float_field(line, "gripper_command_err_deg"),
                target_error_m=_float_field(line, "target_error_m"),
                object_speed_mps=_float_field(line, "object_speed_mps"),
                counter_gap_max_m=_vector_max(line, "counter_gap_obj_m"),
                virtual_compression_gap_max_m=_float_field(line, "virtual_compression_gap_max_m"),
                one_sided_push=_bool_field(line, "one_sided_push"),
                virtual_support=_bool_field(line, "virtual_support"),
                support_horizon_active=_bool_field(line, "support_horizon_active"),
                close_advance=_bool_field(line, "target_guarded_close_advance"),
                close_hold=_bool_field(line, "target_guarded_close_hold"),
                zero_backlog_hold=_bool_field(line, "target_guarded_zero_backlog_hold"),
                backlog_preserved_hold=_bool_field(line, "target_guarded_backlog_preserved_hold"),
                safety_rollback=_bool_field(line, "target_guarded_v3_safety_rollback"),
                support_budget_ok=_bool_field(line, "target_guarded_support_budget_ok"),
                v3_safety_ok=_bool_field(line, "target_guarded_v3_safety_ok"),
                advances_total=_int_field(line, "target_guarded_close_advances_total"),
                holds_total=_int_field(line, "target_guarded_close_holds_total"),
                zero_backlog_holds_total=_int_field(line, "target_guarded_zero_backlog_holds_total"),
                backlog_preserved_holds_total=_int_field(
                    line, "target_guarded_backlog_preserved_holds_total"
                ),
                safety_rollbacks_total=_int_field(line, "target_guarded_safety_rollbacks_total"),
            )
        )
    if not rows:
        raise ValueError(f"no close rows found in {path}")
    return rows


def _safety_reasons(row: CloseStep) -> list[str]:
    reasons: list[str] = []
    if row.target_error_m > TARGET_ERROR_DESIGN_LIMIT_M:
        reasons.append("target_error_design_limit")
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
    first_rollback = safety_rollbacks[0]
    last_advance = advances[-1]
    peak_target_error = max(rows, key=lambda row: row.target_error_m)
    peak_gripper_q = max(rows, key=lambda row: row.gripper_q_deg)
    final = rows[-1]
    audit_failures = _audit_fail_lines(args.audit)

    print("[cube2cm_target_guarded_v3_progress_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_target_guarded_v3_progress_static] "
        "dataset_generation=NO hold_lift=NO constraints=NO surface_gripper=NO "
        "transport_release=NO gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] "
        f"stdout_log={args.stdout} md5={stdout_md5} expected_md5={args.expected_stdout_md5} "
        f"audit_log={args.audit} audit_md5={audit_md5} expected_audit_md5={args.expected_audit_md5}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] schedule_totals "
        f"close_steps={len(rows)} close_steps_expected={CLOSE_STEPS} "
        f"advances={len(advances)} holds={len(holds)} zero_backlog_holds={len(zero_backlog_holds)} "
        f"backlog_preserved_holds={len(preserved_holds)} safety_rollbacks={len(safety_rollbacks)} "
        f"advance_steps={[row.step for row in advances]}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] early_progress "
        f"step3_line=379 gripper_q_deg={rows[2].gripper_q_deg:.3f} "
        f"zero_backlog_holds_total_step3={rows[2].zero_backlog_holds_total} "
        f"step6_line=382 gripper_q_deg={rows[5].gripper_q_deg:.3f} "
        f"step9_line=385 gripper_q_deg={rows[8].gripper_q_deg:.3f}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] first_safety_rollback "
        f"line={first_rollback.line} step={first_rollback.step} "
        f"gripper_q_deg={first_rollback.gripper_q_deg:.3f} "
        f"gripper_command_deg={first_rollback.gripper_command_deg:.3f} "
        f"command_backlog_deg={first_rollback.gripper_command_err_deg:.3f} "
        f"target_error_m={first_rollback.target_error_m:.6f} "
        f"target_error_design_limit_m={TARGET_ERROR_DESIGN_LIMIT_M:.6f} "
        f"object_speed_mps={first_rollback.object_speed_mps:.6f} "
        f"support_budget_ok={_yes(first_rollback.support_budget_ok)} "
        f"support_horizon_active={_yes(first_rollback.support_horizon_active)} "
        f"one_sided_push={_yes(first_rollback.one_sided_push)} "
        f"reasons={'+'.join(_safety_reasons(first_rollback))}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] last_advance "
        f"line={last_advance.line} step={last_advance.step} "
        f"gripper_q_deg={last_advance.gripper_q_deg:.3f} "
        f"target_error_m={last_advance.target_error_m:.6f} "
        f"safety_rollbacks_total_before_or_at_step={last_advance.safety_rollbacks_total}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] peak_target_and_gripper "
        f"peak_target_line={peak_target_error.line} peak_target_step={peak_target_error.step} "
        f"peak_target_error_m={peak_target_error.target_error_m:.6f} "
        f"target_error_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"support_budget_ok={_yes(peak_target_error.support_budget_ok)} "
        f"counter_gap_max_m={peak_target_error.counter_gap_max_m:.6f} "
        f"counter_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"peak_q_line={peak_gripper_q.line} peak_q_step={peak_gripper_q.step} "
        f"peak_gripper_q_deg={peak_gripper_q.gripper_q_deg:.3f}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] final_plateau "
        f"line={final.line} step={final.step} gripper_q_deg={final.gripper_q_deg:.3f} "
        f"gripper_command_deg={final.gripper_command_deg:.3f} "
        f"close_remaining_deg={CLOSE_TARGET_DEG - final.gripper_q_deg:.3f} "
        f"target_error_m={final.target_error_m:.6f} "
        f"object_speed_mps={final.object_speed_mps:.6f} "
        f"support_budget_ok={_yes(final.support_budget_ok)} "
        f"support_horizon_active={_yes(final.support_horizon_active)} "
        f"safety_rollbacks_total={final.safety_rollbacks_total}"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] audit_failures "
        + " | ".join(f"line={line_no}:{line.split('criterion name=')[-1]}" for line_no, line in audit_failures)
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] attribution "
        "zero_backlog_starvation=FIXED "
        "primary=target_pose_error_safety_rollback_after_progress "
        "secondary=support_budget_breach_after_target_error_overshoot "
        "fixed_gate_relaxation=REJECT "
        "structural_contact_compatible_close_required=YES"
    )
    print(
        "[cube2cm_target_guarded_v3_progress_static] next_mechanism_requirements "
        "preserve_fixed_close26_audit_gates=YES "
        "do_not_reintroduce_zero_backlog_holds=YES "
        "add_target_error_recovery_or_contact_compatible_close=YES "
        "prove_no_safety_rollback_before_hold_lift=YES"
    )
    print("[cube2cm_target_guarded_v3_progress_static] TARGET_GUARDED_V3_PROGRESS_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
