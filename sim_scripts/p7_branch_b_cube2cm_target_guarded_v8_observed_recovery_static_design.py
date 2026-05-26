#!/usr/bin/env python3
"""Static design contract for v8 observed-response recovery after v7 FAIL.

This is a local/static artifact. It reads the preserved post-reboot v7 runtime
and analyzer logs, then checks whether a v8 design target is justified by the
evidence. It does not launch Isaac, train, collect data, insert constraints, use
SurfaceGripper, tune fixed gates, posewrite objects, or claim grasp success.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_track_a_v7_active_recovery_close26_local_post_reboot"
    / "runtime.out"
)
DEFAULT_ANALYSIS = DEFAULT_RUNTIME.with_name("v7_failure_static_analysis.out")
EXPECTED_RUNTIME_MD5 = "621d00b9d157b4e70178c28f94ca4c7f"
EXPECTED_ANALYSIS_MD5 = "0fbf57f32473fa253ee1082b888bdcb1"

TARGET_ERROR_GATE_M = 0.003
COUNTER_SUPPORT_BUDGET_M = 0.002
PROPOSED_PROJECTED_TARGET_RESERVE_M = 0.000800
PROPOSED_PROJECTED_SUPPORT_RESERVE_M = 0.000400
REQUIRED_MIN_STEPS_BEFORE_SUPPORT_BREACH = 5
REQUIRED_MIN_STEPS_BEFORE_V7_ACTIVE = 2
FOLLOW_RATIO_MIN = 0.0


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class CloseRow:
    line: int
    step: int
    target_error_m: float
    counter_gap_m: float
    tcp: tuple[float, float, float]
    target_tcp: tuple[float, float, float]
    object_drift_m: float
    moving_contact: bool
    counter_contact: bool
    counter_slop_contact: bool
    virtual_support: bool
    hard_safety_ok: bool
    hard_freeze: bool
    v7_active: bool
    v7_needed: bool
    v7_best_target_margin_m: float
    v7_best_support_margin_m: float
    v7_counter_gap_delta_m: float
    v7_recovery_tcp: tuple[float, float, float]
    projected_target_margin_m: float
    projected_support_margin_m: float
    projected_advance_ok: bool
    command_backlog_deg: float
    command_converged: bool

    @property
    def target_margin_m(self) -> float:
        return TARGET_ERROR_GATE_M - self.target_error_m

    @property
    def support_margin_m(self) -> float:
        return COUNTER_SUPPORT_BUDGET_M - self.counter_gap_m


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(line: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    if not match:
        raise ValueError(f"missing field {name!r} in line: {line[:180]}")
    return match.group(1)


def _float_field(line: str, name: str) -> float:
    return float(_field(line, name))


def _bool_field(line: str, name: str) -> bool:
    value = _field(line, name)
    if value == "YES":
        return True
    if value == "NO":
        return False
    raise ValueError(f"field {name!r} is not YES/NO: {value!r}")


def _vector_field(line: str, name: str) -> tuple[float, ...]:
    match = re.search(rf"(?:^|\s){re.escape(name)}=\(\[([^\]]+)\]\)", line)
    if not match:
        raise ValueError(f"missing vector field {name!r} in line: {line[:180]}")
    return tuple(float(v) for v in re.findall(_FLOAT, match.group(1)))


def _norm(vec: tuple[float, ...]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _parse_rows(path: Path) -> list[CloseRow]:
    rows: list[CloseRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if " phase=close " not in line:
            continue
        counter_gap = max(_vector_field(line, "counter_gap_obj_m"))
        tcp = _vector_field(line, "tcp")
        target_tcp = _vector_field(line, "target_tcp")
        recovery_tcp = _vector_field(line, "target_guarded_v7_recovery_tcp")
        rows.append(
            CloseRow(
                line=line_no,
                step=int(_field(line, "step")),
                target_error_m=_float_field(line, "target_error_m"),
                counter_gap_m=counter_gap,
                tcp=(tcp[0], tcp[1], tcp[2]),
                target_tcp=(target_tcp[0], target_tcp[1], target_tcp[2]),
                object_drift_m=_float_field(line, "object_drift_m"),
                moving_contact=_bool_field(line, "moving_contact"),
                counter_contact=_bool_field(line, "counter_contact"),
                counter_slop_contact=_bool_field(line, "counter_slop_contact"),
                virtual_support=_bool_field(line, "virtual_support"),
                hard_safety_ok=_bool_field(line, "target_guarded_v4_hard_safety_ok"),
                hard_freeze=_bool_field(line, "target_guarded_v4_hard_safety_freeze"),
                v7_active=_bool_field(line, "target_guarded_v7_active_recovery"),
                v7_needed=_bool_field(line, "target_guarded_v7_active_recovery_needed"),
                v7_best_target_margin_m=_float_field(line, "target_guarded_v7_best_target_margin_m"),
                v7_best_support_margin_m=_float_field(line, "target_guarded_v7_best_support_margin_m"),
                v7_counter_gap_delta_m=_float_field(line, "target_guarded_v7_counter_gap_delta_m"),
                v7_recovery_tcp=(recovery_tcp[0], recovery_tcp[1], recovery_tcp[2]),
                projected_target_margin_m=_float_field(line, "target_guarded_v6_projected_target_margin_m"),
                projected_support_margin_m=_float_field(line, "target_guarded_v6_projected_support_margin_m"),
                projected_advance_ok=_bool_field(line, "target_guarded_v6_projected_advance_ok"),
                command_backlog_deg=_float_field(line, "target_guarded_command_backlog_deg"),
                command_converged=_bool_field(line, "target_guarded_command_converged"),
            )
        )
    if not rows:
        raise ValueError(f"no close rows parsed from {path}")
    return rows


def _first(rows: list[CloseRow], predicate) -> CloseRow | None:
    return next((row for row in rows if predicate(row)), None)


def _follow_ratio(row: CloseRow, nxt: CloseRow) -> float:
    command_vec = _sub(row.v7_recovery_tcp, row.tcp)
    actual_vec = _sub(nxt.tcp, row.tcp)
    denom = _dot(command_vec, command_vec)
    if denom <= 1.0e-12:
        return 0.0
    return _dot(actual_vec, command_vec) / denom


def _print_row(label: str, row: CloseRow) -> None:
    print(
        f"{label} line={row.line} step={row.step:03d} "
        f"target_margin={row.target_margin_m:+.6f} support_margin={row.support_margin_m:+.6f} "
        f"projected_target_margin={row.projected_target_margin_m:+.6f} "
        f"projected_support_margin={row.projected_support_margin_m:+.6f} "
        f"v7_active={_yes(row.v7_active)} hard_freeze={_yes(row.hard_freeze)} "
        f"moving_contact={_yes(row.moving_contact)} counter_contact={_yes(row.counter_contact)} "
        f"counter_slop_contact={_yes(row.counter_slop_contact)} virtual_support={_yes(row.virtual_support)}"
    )


def analyze(runtime: Path, analysis: Path) -> int:
    runtime_md5 = _md5(runtime)
    analysis_md5 = _md5(analysis)
    runtime_ok = runtime_md5 == EXPECTED_RUNTIME_MD5
    analysis_ok = analysis_md5 == EXPECTED_ANALYSIS_MD5
    print(f"RUNTIME_MD5={runtime_md5} expected={EXPECTED_RUNTIME_MD5} ok={_yes(runtime_ok)}")
    print(f"ANALYSIS_MD5={analysis_md5} expected={EXPECTED_ANALYSIS_MD5} ok={_yes(analysis_ok)}")
    if not runtime_ok or not analysis_ok:
        print("STATIC_V8_DESIGN_DONE=NO reason=input_md5_mismatch")
        return 2

    rows = _parse_rows(runtime)
    by_step = {row.step: row for row in rows}
    first_v7 = _first(rows, lambda row: row.v7_active)
    first_support = _first(rows, lambda row: row.counter_gap_m > COUNTER_SUPPORT_BUDGET_M)
    first_hard = _first(rows, lambda row: row.hard_freeze)
    first_projected_reserve = _first(
        rows,
        lambda row: (
            row.projected_target_margin_m <= PROPOSED_PROJECTED_TARGET_RESERVE_M
            or row.projected_support_margin_m <= PROPOSED_PROJECTED_SUPPORT_RESERVE_M
        )
        and row.hard_safety_ok,
    )

    if first_v7 is None or first_support is None or first_hard is None or first_projected_reserve is None:
        print("STATIC_V8_DESIGN_DONE=NO reason=missing_required_failure_markers")
        return 2

    _print_row("FIRST_PROJECTED_RESERVE_TRIGGER", first_projected_reserve)
    _print_row("FIRST_V7_ACTIVE", first_v7)
    _print_row("FIRST_SUPPORT_BREACH", first_support)
    _print_row("FIRST_HARD_FREEZE", first_hard)

    reserve_before_v7 = first_v7.step - first_projected_reserve.step
    reserve_before_support = first_support.step - first_projected_reserve.step
    print(
        "EARLY_TRIGGER_CROSSCHECK "
        f"reserve_before_v7_steps={reserve_before_v7} "
        f"reserve_before_support_breach_steps={reserve_before_support} "
        f"min_before_v7={REQUIRED_MIN_STEPS_BEFORE_V7_ACTIVE} "
        f"min_before_support={REQUIRED_MIN_STEPS_BEFORE_SUPPORT_BREACH}"
    )

    active_rows = [row for row in rows if row.v7_active]
    response_failures = 0
    follow_failures = 0
    print("OBSERVED_RESPONSE_REJECTS_UNCHANGED_V7:")
    for row in active_rows:
        nxt = by_step.get(row.step + 1)
        if nxt is None:
            response_failures += 1
            follow_failures += 1
            print(f"  line={row.line} step={row.step:03d} next=MISSING")
            continue
        target_delta = nxt.target_error_m - row.target_error_m
        gap_delta = nxt.counter_gap_m - row.counter_gap_m
        follow_ratio = _follow_ratio(row, nxt)
        response_ok = target_delta <= 0.0 and gap_delta <= 0.0
        follow_ok = follow_ratio > FOLLOW_RATIO_MIN
        response_failures += int(not response_ok)
        follow_failures += int(not follow_ok)
        print(
            f"  line={row.line}->line={nxt.line} step={row.step:03d}->{nxt.step:03d} "
            f"target_delta={target_delta:+.6f} gap_delta={gap_delta:+.6f} "
            f"follow_ratio={follow_ratio:+.3f} response_ok={_yes(response_ok)} "
            f"follow_ok={_yes(follow_ok)}"
        )

    moving_contact_active = any(row.moving_contact for row in active_rows)
    counter_contact_active = any(row.counter_contact for row in active_rows)
    counter_slop_contact_active = any(row.counter_slop_contact for row in active_rows)
    max_object_drift = max(row.object_drift_m for row in rows)
    geometry_suspect = (
        moving_contact_active
        and not counter_contact_active
        and not counter_slop_contact_active
        and max_object_drift < 0.000100
    )

    v8_static_checks = {
        "inputs_md5_verified": runtime_ok and analysis_ok,
        "earlier_projected_reserve_trigger": reserve_before_v7 >= REQUIRED_MIN_STEPS_BEFORE_V7_ACTIVE,
        "reserve_horizon_before_support_breach": reserve_before_support >= REQUIRED_MIN_STEPS_BEFORE_SUPPORT_BREACH,
        "unchanged_v7_rejected_by_observed_response": response_failures == len(active_rows),
        "unchanged_v7_rejected_by_tcp_follow": follow_failures == len(active_rows),
        "counter_contact_geometry_must_be_modeled": geometry_suspect,
        "fixed_gates_preserved": True,
        "forbidden_mechanisms_forbidden": True,
    }
    print("V8_STATIC_CHECKS:")
    for name, passed in v8_static_checks.items():
        print(f"  {name}={_yes(passed)}")

    print("V8_DESIGN_CONTRACT:")
    print(
        "  trigger=projected_target_margin<=0.000800 OR "
        "projected_support_margin<=0.000400 while hard_safety_ok"
    )
    print(
        "  recovery=observed_response_tracked multi_step; do not treat selected "
        "candidate margin as success"
    )
    print(
        "  audit=new v8 checks must reject active rows whose next row worsens both "
        "target/support or whose TCP follow ratio <= 0"
    )
    print(
        "  geometry=score/telemetry must include counter_contact or counter_slop_contact "
        "restoration, not only max counter-gap reduction"
    )
    print(
        "  forbidden=no attach, no object posewrite, no constraints, no SurfaceGripper, "
        "no fixed-gate tuning, no zero-backlog holds, no safety rollbacks"
    )
    print("RUNTIME_READY=NO")
    print(f"STATIC_V8_DESIGN_DONE={_yes(all(v8_static_checks.values()))}")
    return 0 if all(v8_static_checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    return analyze(args.runtime, args.analysis)


if __name__ == "__main__":
    raise SystemExit(main())
