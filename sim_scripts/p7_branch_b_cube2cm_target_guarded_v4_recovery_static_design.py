#!/usr/bin/env python3
"""Static design check for target-guarded v4 recovery.

This script reads the already-produced v3 B200 stdout/audit logs and identifies
the earliest scheduler decision v4 must change. It does not launch Isaac, train,
generate data, insert constraints, use SurfaceGripper, tune gates, or claim
success.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STDOUT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out")
DEFAULT_AUDIT = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out")
EXPECTED_STDOUT_MD5 = "5f2d1a626edcdccce8086fafd321c9af"
EXPECTED_AUDIT_MD5 = "ca60c09b03a156c85197e34ec7b28bb5"

TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
V4_RECOVERY_TARGET_ERROR_M = 0.0024
TARGET_ERROR_GROWTH_TOLERANCE_M = 0.00025


@dataclass(frozen=True)
class CloseRow:
    line: int
    step: int
    gripper_q_deg: float
    gripper_command_deg: float
    command_backlog_deg: float
    target_error_m: float
    counter_gap_max_m: float
    target_nonworsening: bool
    support_budget_ok: bool
    support_horizon_active: bool
    one_sided_push: bool
    v3_advance: bool
    v3_safety_rollback: bool


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


def _bool(line: str, name: str) -> bool:
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
                target_nonworsening=_bool(line, "target_guarded_target_nonworsening"),
                support_budget_ok=_bool(line, "target_guarded_support_budget_ok"),
                support_horizon_active=_bool(line, "support_horizon_active"),
                one_sided_push=_bool(line, "one_sided_push"),
                v3_advance=_bool(line, "target_guarded_close_advance"),
                v3_safety_rollback=_bool(line, "target_guarded_v3_safety_rollback"),
            )
        )
    if not rows:
        raise ValueError(f"no close rows parsed from {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", type=Path, default=DEFAULT_STDOUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--expected_stdout_md5", default=EXPECTED_STDOUT_MD5)
    parser.add_argument("--expected_audit_md5", default=EXPECTED_AUDIT_MD5)
    args = parser.parse_args()

    stdout_md5 = _md5(args.stdout)
    audit_md5 = _md5(args.audit)
    if stdout_md5 != args.expected_stdout_md5:
        raise SystemExit(f"stdout md5 mismatch: got {stdout_md5}, expected {args.expected_stdout_md5}")
    if audit_md5 != args.expected_audit_md5:
        raise SystemExit(f"audit md5 mismatch: got {audit_md5}, expected {args.expected_audit_md5}")

    rows = _parse_rows(args.stdout)
    target_growth_advances = [
        row for row in rows if row.v3_advance and not row.target_nonworsening
    ]
    first_v4_block = target_growth_advances[0]
    first_rollback = next(row for row in rows if row.v3_safety_rollback)
    first_fixed_target_violation = next(row for row in rows if row.target_error_m > TARGET_ERROR_GATE_M)
    first_fixed_support_violation = next(row for row in rows if row.counter_gap_max_m > COUNTER_SUPPORT_BUDGET_M)
    prev_before_block = rows[rows.index(first_v4_block) - 1]
    target_growth_m = first_v4_block.target_error_m - prev_before_block.target_error_m

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
        "[cube2cm_target_guarded_v4_recovery_static] v4_first_intervention "
        f"line={first_v4_block.line} step={first_v4_block.step} "
        f"v3_advance=YES target_nonworsening=NO "
        f"target_error_growth_m={target_growth_m:.6f} "
        f"growth_tolerance_m={TARGET_ERROR_GROWTH_TOLERANCE_M:.6f} "
        f"gripper_q_deg={first_v4_block.gripper_q_deg:.3f} "
        f"gripper_command_deg={first_v4_block.gripper_command_deg:.3f} "
        "v4_action=RECOVERY_HOLD_PRESERVE_BACKLOG"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] v3_first_rollback_reclassified "
        f"line={first_rollback.line} step={first_rollback.step} "
        f"target_error_m={first_rollback.target_error_m:.6f} "
        f"fixed_target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_gap_max_m={first_rollback.counter_gap_max_m:.6f} "
        f"fixed_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        "v4_action=RECOVERY_HOLD_NOT_ROLLBACK"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] hard_gate_guard "
        f"first_target_violation_line={first_fixed_target_violation.line} "
        f"first_target_violation_step={first_fixed_target_violation.step} "
        f"target_error_m={first_fixed_target_violation.target_error_m:.6f} "
        f"first_support_violation_line={first_fixed_support_violation.line} "
        f"first_support_violation_step={first_fixed_support_violation.step} "
        f"counter_gap_max_m={first_fixed_support_violation.counter_gap_max_m:.6f} "
        "v4_action=HARD_SAFETY_FREEZE_AND_AUDIT_FAIL_IF_REACHED"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] design_contract "
        f"recovery_target_error_m={V4_RECOVERY_TARGET_ERROR_M:.6f} "
        f"fixed_target_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"fixed_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        "zero_backlog_holds=FORBIDDEN safety_rollbacks=FORBIDDEN "
        "recovery_holds_preserve_backlog=YES"
    )
    print(
        "[cube2cm_target_guarded_v4_recovery_static] attribution "
        "v3_zero_backlog_starvation=FIXED "
        "v3_primary_failure=target_pose_error_safety_rollback_after_progress "
        "v4_primary_change=target_nonworsening_recovery_hold_before_command_10deg "
        "fixed_gate_relaxation=REJECT"
    )
    print("[cube2cm_target_guarded_v4_recovery_static] TARGET_GUARDED_V4_RECOVERY_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
