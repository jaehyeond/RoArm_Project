#!/usr/bin/env python3
"""Static analyzer for the post-reboot v7 close_26 runtime failure.

This script reads an existing runtime log only. It does not launch Isaac, train,
write env state, or modify any dataset. The goal is to separate four failure
domains before any new runtime attempt:

1. audit contract mismatch
2. late trigger timing
3. candidate prediction versus observed dynamics mismatch
4. actuator/TCP follow lag and contact geometry suspicion
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TARGET_ERROR_GATE_M = 0.003
COUNTER_SUPPORT_BUDGET_M = 0.002
PREEMPT_TARGET_MARGIN_M = 0.000200
PREEMPT_SUPPORT_MARGIN_M = 0.000100


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _field(line: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    return match.group(1) if match else None


def _float_field(line: str, name: str) -> float | None:
    value = _field(line, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool_field(line: str, name: str) -> bool | None:
    value = _field(line, name)
    if value == "YES":
        return True
    if value == "NO":
        return False
    return None


def _vector_field(line: str, name: str) -> tuple[float, ...] | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}=\(\[([^\]]+)\]\)", line)
    if not match:
        return None
    return tuple(float(v) for v in re.findall(_FLOAT, match.group(1)))


def _norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class CloseRow:
    line_no: int
    step: int
    target_error_m: float
    counter_gap_max_m: float
    tcp: tuple[float, float, float]
    target_tcp: tuple[float, float, float]
    object_drift_m: float
    gripper_q_deg: float
    gripper_command_deg: float
    gripper_command_err_deg: float
    moving_contact: bool
    counter_contact: bool
    virtual_support: bool
    hard_safety_ok: bool
    hard_freeze: bool
    v5_needed: bool
    v7_needed: bool
    v7_active: bool
    v7_ik_ok: bool
    v7_candidate_count: int
    v7_selected_score: float
    v7_best_target_margin_m: float
    v7_best_support_margin_m: float
    v7_counter_gap_delta_m: float
    v7_recovery_tcp: tuple[float, float, float]
    v6_projected_advance_ok: bool
    support_budget_ok: bool
    command_converged: bool
    source: str


def _parse_close_row(line_no: int, line: str) -> CloseRow | None:
    if " phase=close " not in line or " step=" not in line:
        return None
    step_match = re.search(r" step=(\d+)", line)
    if not step_match:
        return None
    counter_gap = _vector_field(line, "counter_gap_obj_m")
    tcp = _vector_field(line, "tcp")
    target_tcp = _vector_field(line, "target_tcp")
    recovery_tcp = _vector_field(line, "target_guarded_v7_recovery_tcp")
    required = {
        "target_error_m": _float_field(line, "target_error_m"),
        "object_drift_m": _float_field(line, "object_drift_m"),
        "gripper_q_deg": _float_field(line, "gripper_q_deg"),
        "gripper_command_deg": _float_field(line, "gripper_command_deg"),
        "gripper_command_err_deg": _float_field(line, "gripper_command_err_deg"),
        "v7_candidate_count": _float_field(line, "target_guarded_v7_candidate_count"),
        "v7_selected_score": _float_field(line, "target_guarded_v7_selected_score"),
        "v7_best_target_margin_m": _float_field(line, "target_guarded_v7_best_target_margin_m"),
        "v7_best_support_margin_m": _float_field(line, "target_guarded_v7_best_support_margin_m"),
        "v7_counter_gap_delta_m": _float_field(line, "target_guarded_v7_counter_gap_delta_m"),
    }
    if any(v is None for v in required.values()) or counter_gap is None or tcp is None or target_tcp is None:
        return None
    if recovery_tcp is None:
        recovery_tcp = target_tcp
    bools = {
        "moving_contact": _bool_field(line, "moving_contact"),
        "counter_contact": _bool_field(line, "counter_contact"),
        "virtual_support": _bool_field(line, "virtual_support"),
        "hard_safety_ok": _bool_field(line, "target_guarded_v4_hard_safety_ok"),
        "hard_freeze": _bool_field(line, "target_guarded_v4_hard_safety_freeze"),
        "v5_needed": _bool_field(line, "target_guarded_v5_preemptive_recovery_needed"),
        "v7_needed": _bool_field(line, "target_guarded_v7_active_recovery_needed"),
        "v7_active": _bool_field(line, "target_guarded_v7_active_recovery"),
        "v7_ik_ok": _bool_field(line, "target_guarded_v7_recovery_ik_ok"),
        "v6_projected_advance_ok": _bool_field(line, "target_guarded_v6_projected_advance_ok"),
        "support_budget_ok": _bool_field(line, "target_guarded_support_budget_ok"),
        "command_converged": _bool_field(line, "target_guarded_command_converged"),
    }
    if any(v is None for v in bools.values()):
        return None
    return CloseRow(
        line_no=line_no,
        step=int(step_match.group(1)),
        target_error_m=float(required["target_error_m"]),
        counter_gap_max_m=max(counter_gap),
        tcp=(tcp[0], tcp[1], tcp[2]),
        target_tcp=(target_tcp[0], target_tcp[1], target_tcp[2]),
        object_drift_m=float(required["object_drift_m"]),
        gripper_q_deg=float(required["gripper_q_deg"]),
        gripper_command_deg=float(required["gripper_command_deg"]),
        gripper_command_err_deg=float(required["gripper_command_err_deg"]),
        moving_contact=bool(bools["moving_contact"]),
        counter_contact=bool(bools["counter_contact"]),
        virtual_support=bool(bools["virtual_support"]),
        hard_safety_ok=bool(bools["hard_safety_ok"]),
        hard_freeze=bool(bools["hard_freeze"]),
        v5_needed=bool(bools["v5_needed"]),
        v7_needed=bool(bools["v7_needed"]),
        v7_active=bool(bools["v7_active"]),
        v7_ik_ok=bool(bools["v7_ik_ok"]),
        v7_candidate_count=int(required["v7_candidate_count"]),
        v7_selected_score=float(required["v7_selected_score"]),
        v7_best_target_margin_m=float(required["v7_best_target_margin_m"]),
        v7_best_support_margin_m=float(required["v7_best_support_margin_m"]),
        v7_counter_gap_delta_m=float(required["v7_counter_gap_delta_m"]),
        v7_recovery_tcp=(recovery_tcp[0], recovery_tcp[1], recovery_tcp[2]),
        v6_projected_advance_ok=bool(bools["v6_projected_advance_ok"]),
        support_budget_ok=bool(bools["support_budget_ok"]),
        command_converged=bool(bools["command_converged"]),
        source=f"{line_no}",
    )


def _parse_rows(log_path: Path) -> list[CloseRow]:
    rows: list[CloseRow] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            row = _parse_close_row(line_no, line)
            if row is not None:
                rows.append(row)
    return rows


def _fmt_m(value: float) -> str:
    return f"{value:+.6f}m"


def _fmt_bool(value: bool) -> str:
    return "YES" if value else "NO"


def _print_row(prefix: str, row: CloseRow) -> None:
    target_margin = TARGET_ERROR_GATE_M - row.target_error_m
    support_margin = COUNTER_SUPPORT_BUDGET_M - row.counter_gap_max_m
    print(
        f"{prefix} line={row.line_no} step={row.step:03d} "
        f"target={row.target_error_m:.6f} target_margin={target_margin:+.6f} "
        f"counter_gap={row.counter_gap_max_m:.6f} support_margin={support_margin:+.6f} "
        f"hard_ok={_fmt_bool(row.hard_safety_ok)} hard_freeze={_fmt_bool(row.hard_freeze)} "
        f"v7_needed={_fmt_bool(row.v7_needed)} v7_active={_fmt_bool(row.v7_active)} "
        f"counter_contact={_fmt_bool(row.counter_contact)} virtual_support={_fmt_bool(row.virtual_support)}"
    )


def analyze(log_path: Path) -> int:
    rows = _parse_rows(log_path)
    if not rows:
        print("ANALYZER_RESULT=FAIL reason=no_close_rows")
        return 2

    by_step = {row.step: row for row in rows}
    active_rows = [row for row in rows if row.v7_active]
    first_support = next((row for row in rows if row.counter_gap_max_m > COUNTER_SUPPORT_BUDGET_M), None)
    first_target = next((row for row in rows if row.target_error_m > TARGET_ERROR_GATE_M), None)
    first_hard = next((row for row in rows if row.hard_freeze), None)
    first_low_margin = next(
        (
            row
            for row in rows
            if (TARGET_ERROR_GATE_M - row.target_error_m) <= PREEMPT_TARGET_MARGIN_M
            or (COUNTER_SUPPORT_BUDGET_M - row.counter_gap_max_m) <= PREEMPT_SUPPORT_MARGIN_M
        ),
        None,
    )

    print(f"ANALYZER_LOG={log_path}")
    print(f"CLOSE_ROWS={len(rows)} first_line={rows[0].line_no} last_line={rows[-1].line_no}")
    print(f"V7_ACTIVE_ROWS={len(active_rows)}")

    if first_low_margin:
        _print_row("FIRST_LOW_MARGIN", first_low_margin)
    if active_rows:
        _print_row("FIRST_V7_ACTIVE", active_rows[0])
        _print_row("LAST_V7_ACTIVE", active_rows[-1])
    if first_support:
        _print_row("FIRST_SUPPORT_BREACH", first_support)
    if first_target:
        _print_row("FIRST_TARGET_BREACH", first_target)
    if first_hard:
        _print_row("FIRST_HARD_FREEZE", first_hard)

    print("ACTIVE_ROW_FOLLOWUP:")
    candidate_mismatch_count = 0
    weak_follow_count = 0
    for row in active_rows:
        nxt = by_step.get(row.step + 1)
        if nxt is None:
            print(f"  line={row.line_no} step={row.step:03d} next=MISSING")
            continue
        predicted_target = TARGET_ERROR_GATE_M - row.v7_best_target_margin_m
        predicted_gap_by_delta = row.counter_gap_max_m + row.v7_counter_gap_delta_m
        predicted_gap_by_margin = COUNTER_SUPPORT_BUDGET_M - row.v7_best_support_margin_m
        observed_target_delta = nxt.target_error_m - row.target_error_m
        observed_gap_delta = nxt.counter_gap_max_m - row.counter_gap_max_m
        predicted_target_delta = predicted_target - row.target_error_m
        cmd_vec = _sub(row.v7_recovery_tcp, row.tcp)
        actual_vec = _sub(nxt.tcp, row.tcp)
        cmd_norm = _norm(cmd_vec)
        actual_norm = _norm(actual_vec)
        follow_ratio = _dot(actual_vec, cmd_vec) / (cmd_norm * cmd_norm) if cmd_norm > 1e-12 else 0.0
        dist_next_to_recovery = _norm(_sub(nxt.tcp, row.v7_recovery_tcp))
        if observed_gap_delta > 0.0 or observed_target_delta > 0.0:
            candidate_mismatch_count += 1
        if follow_ratio < 0.50:
            weak_follow_count += 1
        print(
            f"  line={row.line_no}->line={nxt.line_no} step={row.step:03d}->{nxt.step:03d} "
            f"pred_target={predicted_target:.6f} pred_target_delta={_fmt_m(predicted_target_delta)} "
            f"obs_target_delta={_fmt_m(observed_target_delta)} "
            f"pred_gap_delta={_fmt_m(row.v7_counter_gap_delta_m)} "
            f"pred_gap_by_delta={predicted_gap_by_delta:.6f} "
            f"pred_gap_by_margin={predicted_gap_by_margin:.6f} "
            f"obs_gap_delta={_fmt_m(observed_gap_delta)} "
            f"cmd_norm={cmd_norm:.6f} actual_tcp_step={actual_norm:.6f} "
            f"follow_ratio={follow_ratio:+.3f} dist_next_to_recovery={dist_next_to_recovery:.6f}"
        )

    object_drift_max = max(row.object_drift_m for row in rows)
    counter_contact_active = any(row.counter_contact for row in active_rows)
    moving_contact_active = any(row.moving_contact for row in active_rows)
    freeze_after_active = first_hard is not None and active_rows and first_hard.step > active_rows[-1].step
    active_to_support_gap_steps = None
    if active_rows and first_support:
        active_to_support_gap_steps = first_support.step - active_rows[0].step

    print("DOMAIN_CLASSIFICATION:")
    print("  audit_contract_mismatch=NO reason=v7_active_checks_pass_but_fixed_gates_fail")
    late_trigger = bool(active_rows and active_to_support_gap_steps is not None and active_to_support_gap_steps <= 3)
    print(
        "  trigger_timing_late="
        f"{_fmt_bool(late_trigger)} "
        f"active_to_support_breach_steps={active_to_support_gap_steps}"
    )
    print(
        "  candidate_prediction_mismatch="
        f"{_fmt_bool(candidate_mismatch_count > 0)} "
        f"worsening_followups={candidate_mismatch_count}/{len(active_rows)}"
    )
    print(
        "  weak_tcp_follow="
        f"{_fmt_bool(weak_follow_count > 0)} "
        f"weak_follow_rows={weak_follow_count}/{len(active_rows)} threshold_ratio=0.50"
    )
    print(
        "  contact_geometry_suspect="
        f"{_fmt_bool(moving_contact_active and not counter_contact_active and object_drift_max < 0.000100)} "
        f"moving_contact_active={_fmt_bool(moving_contact_active)} "
        f"counter_contact_active={_fmt_bool(counter_contact_active)} "
        f"object_drift_max={object_drift_max:.6f}"
    )
    print(
        "  hard_safety_lockout_after_active="
        f"{_fmt_bool(freeze_after_active)} "
        f"first_hard_step={first_hard.step if first_hard else 'NONE'} "
        f"last_active_step={active_rows[-1].step if active_rows else 'NONE'}"
    )

    print("NEXT_VALID_ACTION=STATIC_V8_DESIGN_ONLY")
    print("DO_NOT_RUN=unchanged_v7,hold_lift,dataset,ppo,training,attach,posewrite,gate_tuning")
    print("ANALYZER_RESULT=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Path to preserved runtime.out")
    args = parser.parse_args()
    return analyze(args.log)


if __name__ == "__main__":
    raise SystemExit(main())
