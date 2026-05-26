#!/usr/bin/env python3
"""Static progress attribution for target-guarded v2 close_26.

This script reads the already-produced B200 stdout log and quantifies why
target-guarded v2 stopped near 6deg. It does not launch Isaac, train, generate
data, insert constraints, use a SurfaceGripper, tune gates, or claim success.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LOG = Path("/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out")
EXPECTED_MD5 = "52fa5cf2cc0cc5dbdc2f55f0d099611f"

CLOSE_TARGET_DEG = 26.0
CLOSE_STEPS = 45
MICRO_CLOSE_STEP_DEG = 2.0
COMMAND_ERROR_GATE_DEG = 0.75
ADVANCE_COUNTER_SUPPORT_MARGIN_M = 0.0015
COUNTER_SUPPORT_BUDGET_M = 0.0020
MAX_PLAUSIBLE_COMPRESSION_M = 0.0030
TARGET_ERROR_GATE_M = 0.0030
TARGET_ERROR_DESIGN_LIMIT_M = 0.0027
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
    command_backlog_deg: float
    command_converged: bool
    support_margin_ok: bool
    target_nonworsening: bool
    advances_total: int
    holds_total: int
    zero_backlog_holds_total: int


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _field(line: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}=([^ ]+)", line)
    if not match:
        raise ValueError(f"missing field {name!r} in line: {line[:160]}")
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
        raise ValueError(f"missing vector field {name!r} in line: {line[:160]}")
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
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
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
                command_backlog_deg=_float_field(line, "target_guarded_command_backlog_deg"),
                command_converged=_bool_field(line, "target_guarded_command_converged"),
                support_margin_ok=_bool_field(line, "target_guarded_support_margin_ok"),
                target_nonworsening=_bool_field(line, "target_guarded_target_nonworsening"),
                advances_total=_int_field(line, "target_guarded_close_advances_total"),
                holds_total=_int_field(line, "target_guarded_close_holds_total"),
                zero_backlog_holds_total=_int_field(line, "target_guarded_zero_backlog_holds_total"),
            )
        )
    if not rows:
        raise ValueError(f"no close rows found in {path}")
    return rows


def _hold_reasons(row: CloseStep) -> list[str]:
    reasons: list[str] = []
    if row.target_error_m > TARGET_ERROR_DESIGN_LIMIT_M:
        reasons.append("target_limit")
    if not row.command_converged:
        reasons.append("command_not_converged")
    if not row.support_margin_ok:
        reasons.append("support_margin")
    if not row.target_nonworsening:
        reasons.append("target_worsening")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--expected_md5", default=EXPECTED_MD5)
    args = parser.parse_args()

    log_md5 = _md5(args.log)
    if args.expected_md5 and log_md5 != args.expected_md5:
        raise SystemExit(f"md5 mismatch: got {log_md5}, expected {args.expected_md5}")

    rows = parse_close_steps(args.log)
    advances = [row for row in rows if row.close_advance]
    holds = [row for row in rows if row.close_hold]
    zero_backlog_holds = [row for row in holds if row.zero_backlog_hold]
    final = rows[-1]
    first_target_worsening = next(row for row in rows if not row.target_nonworsening)
    first_support_margin_block = next(row for row in rows if not row.support_margin_ok)
    first_support_margin_only_hold = next(
        row for row in holds if _hold_reasons(row) == ["support_margin"]
    )
    last_advance = advances[-1]

    advance_effects: list[float] = []
    advance_backlogs: list[float] = []
    for index, row in enumerate(rows[:-1]):
        if not row.close_advance:
            continue
        next_row = rows[index + 1]
        advance_effects.append(next_row.gripper_q_deg - row.gripper_q_deg)
        advance_backlogs.append(next_row.command_backlog_deg)

    avg_advance_effect_deg = statistics.mean(advance_effects)
    max_possible_alternating_advances = math.ceil(CLOSE_STEPS / 2)
    projected_without_margin_freeze_deg = avg_advance_effect_deg * max_possible_alternating_advances

    print("[cube2cm_target_guarded_v2_progress_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_target_guarded_v2_progress_static] "
        "dataset_generation=NO hold_lift=NO constraints=NO surface_gripper=NO "
        "transport_release=NO gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] "
        f"source_log={args.log} md5={log_md5} expected_md5={args.expected_md5}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] schedule_totals "
        f"close_steps={len(rows)} close_steps_expected={CLOSE_STEPS} "
        f"advances={len(advances)} holds={len(holds)} "
        f"zero_backlog_holds={len(zero_backlog_holds)} "
        f"advance_steps={[row.step for row in advances]}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] zero_backlog_pulse_effect "
        f"micro_close_step_deg={MICRO_CLOSE_STEP_DEG:.3f} "
        f"avg_actual_motion_after_advance_deg={avg_advance_effect_deg:.3f} "
        f"avg_next_step_backlog_before_zero_deg={statistics.mean(advance_backlogs):.3f} "
        f"command_error_gate_deg={COMMAND_ERROR_GATE_DEG:.3f} "
        f"discarded_fraction_of_micro_step={(statistics.mean(advance_backlogs) / MICRO_CLOSE_STEP_DEG):.3f}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] infeasible_with_current_pulse_schedule "
        f"max_alternating_advances_in_{CLOSE_STEPS}_steps={max_possible_alternating_advances} "
        f"projected_gripper_q_without_margin_freeze_deg={projected_without_margin_freeze_deg:.3f} "
        f"close_target_deg={CLOSE_TARGET_DEG:.3f} "
        f"projected_remaining_deg={CLOSE_TARGET_DEG - projected_without_margin_freeze_deg:.3f}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] first_target_worsening_block "
        f"line={first_target_worsening.line} step={first_target_worsening.step} "
        f"target_error_m={first_target_worsening.target_error_m:.6f} "
        f"target_error_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"command_converged={_yes(first_target_worsening.command_converged)} "
        f"support_margin_ok={_yes(first_target_worsening.support_margin_ok)} "
        f"reasons={'+'.join(_hold_reasons(first_target_worsening))}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] first_support_margin_block "
        f"line={first_support_margin_block.line} step={first_support_margin_block.step} "
        f"counter_gap_max_m={first_support_margin_block.counter_gap_max_m:.6f} "
        f"advance_support_margin_m={ADVANCE_COUNTER_SUPPORT_MARGIN_M:.6f} "
        f"margin_excess_m={first_support_margin_block.counter_gap_max_m - ADVANCE_COUNTER_SUPPORT_MARGIN_M:.6f} "
        f"counter_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f} "
        f"max_plausible_compression_m={MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        f"target_error_m={first_support_margin_block.target_error_m:.6f} "
        f"object_speed_mps={first_support_margin_block.object_speed_mps:.6f} "
        f"support_horizon_active={_yes(first_support_margin_block.support_horizon_active)} "
        f"virtual_support={_yes(first_support_margin_block.virtual_support)} "
        f"one_sided_push={_yes(first_support_margin_block.one_sided_push)} "
        f"reasons={'+'.join(_hold_reasons(first_support_margin_block))}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] first_support_margin_only_hold "
        f"line={first_support_margin_only_hold.line} step={first_support_margin_only_hold.step} "
        f"counter_gap_max_m={first_support_margin_only_hold.counter_gap_max_m:.6f} "
        f"target_error_m={first_support_margin_only_hold.target_error_m:.6f} "
        f"command_backlog_deg={first_support_margin_only_hold.command_backlog_deg:.3f} "
        f"support_horizon_active={_yes(first_support_margin_only_hold.support_horizon_active)} "
        f"virtual_support={_yes(first_support_margin_only_hold.virtual_support)}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] final_plateau "
        f"line={final.line} step={final.step} "
        f"gripper_q_deg={final.gripper_q_deg:.3f} "
        f"gripper_command_deg={final.gripper_command_deg:.3f} "
        f"close_remaining_deg={CLOSE_TARGET_DEG - final.gripper_q_deg:.3f} "
        f"target_error_m={final.target_error_m:.6f} "
        f"counter_gap_max_m={final.counter_gap_max_m:.6f} "
        f"object_speed_mps={final.object_speed_mps:.6f} "
        f"support_margin_ok={_yes(final.support_margin_ok)} "
        f"support_horizon_active={_yes(final.support_horizon_active)} "
        f"virtual_support={_yes(final.virtual_support)}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] attribution "
        "primary=zero_backlog_pulse_progress_starvation "
        "support_margin_0p0015m=STRICT_AND_SECONDARY "
        "support_margin_relaxation_alone=INSUFFICIENT "
        "fixed_gate_relaxation=REJECT "
        "structural_progress_guarantee_required=YES "
        f"last_advance_line={last_advance.line} last_advance_step={last_advance.step}"
    )
    print(
        "[cube2cm_target_guarded_v2_progress_static] next_mechanism_requirements "
        "preserve_fixed_close26_audit_gates=YES "
        "do_not_discard_micro_close_backlog_after_one_step=YES "
        "advance_or_settle_until_actual_gripper_progress=YES "
        "rollback_only_on_safety_degradation=YES "
        "separate_support_warning_margin_from_fixed_support_budget=YES"
    )
    print("[cube2cm_target_guarded_v2_progress_static] TARGET_GUARDED_V2_PROGRESS_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
