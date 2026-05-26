#!/usr/bin/env python3
"""Posthoc pass/fail audit for a future close_26 diagnostic log.

This script does not launch Isaac or change any runtime behavior. It parses a
captured ``p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`` stdout log and
applies the fixed close_26 falsification criteria selected by the static
compliance analysis. The expected mechanism metadata is explicit so a soft-
contact log cannot be passed off as a virtual compression+damping result.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "sim_scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "sim_scripts"))

from p7_branch_b_cube2cm_compliance_proxy_static_analysis import V7_CLOSE_SAMPLES  # noqa: E402


PUSH_SPEED_GATE_MPS = 0.005
TARGET_ERROR_GATE_M = 0.003
COUNTER_SUPPORT_BUDGET_M = 0.002
MAX_PLAUSIBLE_COMPRESSION_M = 0.003
TARGET_GUARDED_COMMAND_ERROR_GATE_DEG = 0.75
TARGET_GUARDED_V3_MIN_ACTUAL_PROGRESS_DEG = 0.25
REQUIRED_CLOSE_STEPS = (2, 3, 4)
TARGET_GUARDED_MECHANISM = "target_guarded_micro_close_support_horizon_diagnostic"
TARGET_GUARDED_V2_MECHANISM = "target_guarded_micro_close_v2_convergence_diagnostic"
TARGET_GUARDED_V3_MECHANISM = "target_guarded_micro_close_v3_progress_diagnostic"
TARGET_GUARDED_V4_MECHANISM = "target_guarded_micro_close_v4_recovery_diagnostic"
TARGET_GUARDED_V5_MECHANISM = "target_guarded_micro_close_v5_preemptive_recovery_diagnostic"
TARGET_GUARDED_V6_MECHANISM = "target_guarded_micro_close_v6_projected_guard_diagnostic"
TARGET_GUARDED_V7_MECHANISM = "target_guarded_micro_close_v7_active_recovery_diagnostic"
TARGET_GUARDED_MECHANISMS = (
    TARGET_GUARDED_MECHANISM,
    TARGET_GUARDED_V2_MECHANISM,
    TARGET_GUARDED_V3_MECHANISM,
    TARGET_GUARDED_V4_MECHANISM,
    TARGET_GUARDED_V5_MECHANISM,
    TARGET_GUARDED_V6_MECHANISM,
    TARGET_GUARDED_V7_MECHANISM,
)
VIRTUAL_DAMPING_MECHANISMS = (
    "virtual_compression_damping_diagnostic",
    TARGET_GUARDED_MECHANISM,
    TARGET_GUARDED_V2_MECHANISM,
    TARGET_GUARDED_V3_MECHANISM,
    TARGET_GUARDED_V4_MECHANISM,
    TARGET_GUARDED_V5_MECHANISM,
    TARGET_GUARDED_V6_MECHANISM,
    TARGET_GUARDED_V7_MECHANISM,
)


@dataclass(frozen=True)
class CloseObservation:
    source: str
    step: int
    target_error_m: float
    object_speed_mps: float
    counter_gap_max_m: float
    counter_contact: bool
    counter_slop_contact: bool
    one_sided_push: bool
    virtual_support: bool | None
    support_horizon_active: bool | None
    virtual_damping_active: bool | None
    virtual_velocity_damping_writes_total: int | None
    target_guarded_close_advance: bool | None
    target_guarded_close_hold: bool | None
    target_guarded_close_advances_total: int | None
    target_guarded_close_holds_total: int | None
    reached: bool
    early_kill: bool
    target_guarded_zero_backlog_hold: bool | None = None
    target_guarded_command_backlog_deg: float | None = None
    target_guarded_command_converged: bool | None = None
    target_guarded_support_margin_ok: bool | None = None
    target_guarded_target_nonworsening: bool | None = None
    target_guarded_zero_backlog_holds_total: int | None = None
    target_guarded_backlog_preserved_hold: bool | None = None
    target_guarded_v3_safety_rollback: bool | None = None
    target_guarded_support_budget_ok: bool | None = None
    target_guarded_v3_safety_ok: bool | None = None
    target_guarded_v3_actual_progress_deg: float | None = None
    target_guarded_v3_actual_progress_ok: bool | None = None
    target_guarded_v3_progress_gate_ok: bool | None = None
    target_guarded_v3_backlog_room_ok: bool | None = None
    target_guarded_v3_projected_backlog_after_advance_deg: float | None = None
    target_guarded_v4_recovery_hold: bool | None = None
    target_guarded_v4_hard_safety_freeze: bool | None = None
    target_guarded_v4_hard_safety_ok: bool | None = None
    target_guarded_v4_recovery_ready: bool | None = None
    target_guarded_v4_target_error_recovered: bool | None = None
    target_guarded_backlog_preserved_holds_total: int | None = None
    target_guarded_safety_rollbacks_total: int | None = None
    target_guarded_v4_recovery_holds_total: int | None = None
    target_guarded_v4_hard_safety_freezes_total: int | None = None
    target_guarded_v5_preemptive_recovery_needed: bool | None = None
    target_guarded_v5_preemptive_recovery: bool | None = None
    target_guarded_v5_recovery_ik_ok: bool | None = None
    target_guarded_v5_target_margin_m: float | None = None
    target_guarded_v5_support_margin_m: float | None = None
    target_guarded_v5_recovery_step_m: float | None = None
    target_guarded_v5_preemptive_recovery_writes_total: int | None = None
    target_guarded_v5_recovery_ik_failures_total: int | None = None
    target_guarded_v7_active_recovery_needed: bool | None = None
    target_guarded_v7_active_recovery: bool | None = None
    target_guarded_v7_recovery_ik_ok: bool | None = None
    target_guarded_v7_candidate_count: int | None = None
    target_guarded_v7_selected_score: float | None = None
    target_guarded_v7_best_target_margin_m: float | None = None
    target_guarded_v7_best_support_margin_m: float | None = None
    target_guarded_v7_counter_gap_delta_m: float | None = None
    target_guarded_v7_recovery_step_m: float | None = None
    target_guarded_v7_active_recovery_writes_total: int | None = None
    target_guarded_v7_recovery_ik_failures_total: int | None = None


@dataclass(frozen=True)
class AggregateObservation:
    source: str
    approach_ok: bool
    descend_ok: bool
    close_reached: bool
    close_early_kill: bool
    attach_calls: int
    posewrite_calls: int
    virtual_velocity_damping_writes: int | None
    target_guarded_close_advances: int | None
    target_guarded_close_holds: int | None
    telemetry_only: bool
    success_claim: bool
    target_guarded_zero_backlog_holds: int | None = None
    target_guarded_backlog_preserved_holds: int | None = None
    target_guarded_safety_rollbacks: int | None = None
    target_guarded_v4_recovery_holds: int | None = None
    target_guarded_v4_hard_safety_freezes: int | None = None
    target_guarded_v5_preemptive_recovery_writes: int | None = None
    target_guarded_v5_recovery_ik_failures: int | None = None
    target_guarded_v7_active_recovery_writes: int | None = None
    target_guarded_v7_recovery_ik_failures: int | None = None


@dataclass(frozen=True)
class RuntimeMetadata:
    source: str
    soft_contact_material_diagnostic: bool | None
    virtual_compression_damping_diagnostic: bool | None
    target_guarded_micro_close_support_horizon_diagnostic: bool | None
    object_physics_mode: str | None
    runtime_candidate_requires_separate_approval: bool | None
    target_guarded_micro_close_v2_convergence_diagnostic: bool | None = None
    target_guarded_micro_close_v3_progress_diagnostic: bool | None = None
    target_guarded_micro_close_v4_recovery_diagnostic: bool | None = None
    target_guarded_micro_close_v5_preemptive_recovery_diagnostic: bool | None = None
    target_guarded_micro_close_v6_projected_guard_diagnostic: bool | None = None
    target_guarded_micro_close_v7_active_recovery_diagnostic: bool | None = None


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _parse_bool(text: str) -> bool:
    if text == "YES":
        return True
    if text == "NO":
        return False
    raise ValueError(f"expected YES/NO, got {text!r}")


def _fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value.rstrip(",")
    return out


def _parse_xyz(text: str) -> tuple[float, float, float]:
    match = re.fullmatch(
        r"\(?\[([+-]?[0-9.]+),\s*([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]\)?",
        text,
    )
    if not match:
        raise ValueError(f"cannot parse xyz value {text!r}")
    return tuple(float(match.group(i)) for i in range(1, 4))


def _line_value(line: str, key: str) -> str:
    vector_match = re.search(rf"{re.escape(key)}=(\(?\[[^\]]+\]\)?)", line)
    if vector_match:
        return vector_match.group(1)
    value_match = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    if value_match:
        return value_match.group(1)
    raise KeyError(key)


def _parse_log(path: Path) -> tuple[dict[int, CloseObservation], AggregateObservation | None, RuntimeMetadata | None]:
    close: dict[int, CloseObservation] = {}
    aggregate: AggregateObservation | None = None
    soft_contact_material_diagnostic: bool | None = None
    virtual_compression_damping_diagnostic: bool | None = None
    target_guarded_micro_close_support_horizon_diagnostic: bool | None = None
    target_guarded_micro_close_v2_convergence_diagnostic: bool | None = None
    target_guarded_micro_close_v3_progress_diagnostic: bool | None = None
    target_guarded_micro_close_v4_recovery_diagnostic: bool | None = None
    target_guarded_micro_close_v5_preemptive_recovery_diagnostic: bool | None = None
    target_guarded_micro_close_v6_projected_guard_diagnostic: bool | None = None
    target_guarded_micro_close_v7_active_recovery_diagnostic: bool | None = None
    object_physics_mode: str | None = None
    runtime_candidate_requires_separate_approval: bool | None = None
    metadata_source = str(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "[cube2cm_runtime_jaw_telemetry] diagnostic_only=YES " in line:
            fields = _fields(line)
            if "soft_contact_material_diagnostic" in fields:
                soft_contact_material_diagnostic = _parse_bool(fields["soft_contact_material_diagnostic"])
                metadata_source = f"{path}:{line_no}"
            if "virtual_compression_damping_diagnostic" in fields:
                virtual_compression_damping_diagnostic = _parse_bool(
                    fields["virtual_compression_damping_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_support_horizon_diagnostic" in fields:
                target_guarded_micro_close_support_horizon_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_support_horizon_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v2_convergence_diagnostic" in fields:
                target_guarded_micro_close_v2_convergence_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v2_convergence_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v3_progress_diagnostic" in fields:
                target_guarded_micro_close_v3_progress_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v3_progress_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v4_recovery_diagnostic" in fields:
                target_guarded_micro_close_v4_recovery_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v4_recovery_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v5_preemptive_recovery_diagnostic" in fields:
                target_guarded_micro_close_v5_preemptive_recovery_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v5_preemptive_recovery_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v6_projected_guard_diagnostic" in fields:
                target_guarded_micro_close_v6_projected_guard_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v6_projected_guard_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
            if "target_guarded_micro_close_v7_active_recovery_diagnostic" in fields:
                target_guarded_micro_close_v7_active_recovery_diagnostic = _parse_bool(
                    fields["target_guarded_micro_close_v7_active_recovery_diagnostic"]
                )
                metadata_source = f"{path}:{line_no}"
        elif "[cube2cm_runtime_jaw_telemetry] object_physics " in line:
            fields = _fields(line)
            object_physics_mode = fields.get("mode")
            if "runtime_candidate_requires_separate_approval" in fields:
                runtime_candidate_requires_separate_approval = _parse_bool(
                    fields["runtime_candidate_requires_separate_approval"]
                )
            metadata_source = f"{path}:{line_no}"
        elif "[cube2cm_runtime_jaw_telemetry] step " in line and " phase=close " in line:
            fields = _fields(line)
            step = int(fields["step"])
            counter_gap = _parse_xyz(_line_value(line, "counter_gap_obj_m"))
            close[step] = CloseObservation(
                source=f"{path}:{line_no}",
                step=step,
                target_error_m=float(_line_value(line, "target_error_m")),
                object_speed_mps=float(_line_value(line, "object_speed_mps")),
                counter_gap_max_m=max(counter_gap),
                counter_contact=_parse_bool(_line_value(line, "counter_contact")),
                counter_slop_contact=_parse_bool(_line_value(line, "counter_slop_contact")),
                one_sided_push=_parse_bool(_line_value(line, "one_sided_push")),
                virtual_support=(
                    _parse_bool(fields["virtual_support"]) if "virtual_support" in fields else None
                ),
                support_horizon_active=(
                    _parse_bool(fields["support_horizon_active"]) if "support_horizon_active" in fields else None
                ),
                virtual_damping_active=(
                    _parse_bool(fields["virtual_damping_active"]) if "virtual_damping_active" in fields else None
                ),
                virtual_velocity_damping_writes_total=(
                    int(fields["virtual_velocity_damping_writes_total"])
                    if "virtual_velocity_damping_writes_total" in fields
                    else None
                ),
                target_guarded_close_advance=(
                    _parse_bool(fields["target_guarded_close_advance"])
                    if "target_guarded_close_advance" in fields
                    else None
                ),
                target_guarded_close_hold=(
                    _parse_bool(fields["target_guarded_close_hold"])
                    if "target_guarded_close_hold" in fields
                    else None
                ),
                target_guarded_close_advances_total=(
                    int(fields["target_guarded_close_advances_total"])
                    if "target_guarded_close_advances_total" in fields
                    else None
                ),
                target_guarded_close_holds_total=(
                    int(fields["target_guarded_close_holds_total"])
                    if "target_guarded_close_holds_total" in fields
                    else None
                ),
                reached=_parse_bool(_line_value(line, "reached")),
                early_kill=_parse_bool(_line_value(line, "early_kill")),
                target_guarded_zero_backlog_hold=(
                    _parse_bool(fields["target_guarded_zero_backlog_hold"])
                    if "target_guarded_zero_backlog_hold" in fields
                    else None
                ),
                target_guarded_command_backlog_deg=(
                    float(fields["target_guarded_command_backlog_deg"])
                    if "target_guarded_command_backlog_deg" in fields
                    else None
                ),
                target_guarded_command_converged=(
                    _parse_bool(fields["target_guarded_command_converged"])
                    if "target_guarded_command_converged" in fields
                    else None
                ),
                target_guarded_support_margin_ok=(
                    _parse_bool(fields["target_guarded_support_margin_ok"])
                    if "target_guarded_support_margin_ok" in fields
                    else None
                ),
                target_guarded_target_nonworsening=(
                    _parse_bool(fields["target_guarded_target_nonworsening"])
                    if "target_guarded_target_nonworsening" in fields
                    else None
                ),
                target_guarded_zero_backlog_holds_total=(
                    int(fields["target_guarded_zero_backlog_holds_total"])
                    if "target_guarded_zero_backlog_holds_total" in fields
                    else None
                ),
                target_guarded_backlog_preserved_hold=(
                    _parse_bool(fields["target_guarded_backlog_preserved_hold"])
                    if "target_guarded_backlog_preserved_hold" in fields
                    else None
                ),
                target_guarded_v3_safety_rollback=(
                    _parse_bool(fields["target_guarded_v3_safety_rollback"])
                    if "target_guarded_v3_safety_rollback" in fields
                    else None
                ),
                target_guarded_v4_recovery_hold=(
                    _parse_bool(fields["target_guarded_v4_recovery_hold"])
                    if "target_guarded_v4_recovery_hold" in fields
                    else None
                ),
                target_guarded_v4_hard_safety_freeze=(
                    _parse_bool(fields["target_guarded_v4_hard_safety_freeze"])
                    if "target_guarded_v4_hard_safety_freeze" in fields
                    else None
                ),
                target_guarded_support_budget_ok=(
                    _parse_bool(fields["target_guarded_support_budget_ok"])
                    if "target_guarded_support_budget_ok" in fields
                    else None
                ),
                target_guarded_v3_safety_ok=(
                    _parse_bool(fields["target_guarded_v3_safety_ok"])
                    if "target_guarded_v3_safety_ok" in fields
                    else None
                ),
                target_guarded_v3_actual_progress_deg=(
                    float(fields["target_guarded_v3_actual_progress_deg"])
                    if "target_guarded_v3_actual_progress_deg" in fields
                    else None
                ),
                target_guarded_v3_actual_progress_ok=(
                    _parse_bool(fields["target_guarded_v3_actual_progress_ok"])
                    if "target_guarded_v3_actual_progress_ok" in fields
                    else None
                ),
                target_guarded_v3_progress_gate_ok=(
                    _parse_bool(fields["target_guarded_v3_progress_gate_ok"])
                    if "target_guarded_v3_progress_gate_ok" in fields
                    else None
                ),
                target_guarded_v3_backlog_room_ok=(
                    _parse_bool(fields["target_guarded_v3_backlog_room_ok"])
                    if "target_guarded_v3_backlog_room_ok" in fields
                    else None
                ),
                target_guarded_v3_projected_backlog_after_advance_deg=(
                    float(fields["target_guarded_v3_projected_backlog_after_advance_deg"])
                    if "target_guarded_v3_projected_backlog_after_advance_deg" in fields
                    else None
                ),
                target_guarded_v4_hard_safety_ok=(
                    _parse_bool(fields["target_guarded_v4_hard_safety_ok"])
                    if "target_guarded_v4_hard_safety_ok" in fields
                    else None
                ),
                target_guarded_v4_recovery_ready=(
                    _parse_bool(fields["target_guarded_v4_recovery_ready"])
                    if "target_guarded_v4_recovery_ready" in fields
                    else None
                ),
                target_guarded_v4_target_error_recovered=(
                    _parse_bool(fields["target_guarded_v4_target_error_recovered"])
                    if "target_guarded_v4_target_error_recovered" in fields
                    else None
                ),
                target_guarded_backlog_preserved_holds_total=(
                    int(fields["target_guarded_backlog_preserved_holds_total"])
                    if "target_guarded_backlog_preserved_holds_total" in fields
                    else None
                ),
                target_guarded_safety_rollbacks_total=(
                    int(fields["target_guarded_safety_rollbacks_total"])
                    if "target_guarded_safety_rollbacks_total" in fields
                    else None
                ),
                target_guarded_v4_recovery_holds_total=(
                    int(fields["target_guarded_v4_recovery_holds_total"])
                    if "target_guarded_v4_recovery_holds_total" in fields
                    else None
                ),
                target_guarded_v4_hard_safety_freezes_total=(
                    int(fields["target_guarded_v4_hard_safety_freezes_total"])
                    if "target_guarded_v4_hard_safety_freezes_total" in fields
                    else None
                ),
                target_guarded_v5_preemptive_recovery_needed=(
                    _parse_bool(fields["target_guarded_v5_preemptive_recovery_needed"])
                    if "target_guarded_v5_preemptive_recovery_needed" in fields
                    else None
                ),
                target_guarded_v5_preemptive_recovery=(
                    _parse_bool(fields["target_guarded_v5_preemptive_recovery"])
                    if "target_guarded_v5_preemptive_recovery" in fields
                    else None
                ),
                target_guarded_v5_recovery_ik_ok=(
                    _parse_bool(fields["target_guarded_v5_recovery_ik_ok"])
                    if "target_guarded_v5_recovery_ik_ok" in fields
                    else None
                ),
                target_guarded_v5_target_margin_m=(
                    float(fields["target_guarded_v5_target_margin_m"])
                    if "target_guarded_v5_target_margin_m" in fields
                    else None
                ),
                target_guarded_v5_support_margin_m=(
                    float(fields["target_guarded_v5_support_margin_m"])
                    if "target_guarded_v5_support_margin_m" in fields
                    else None
                ),
                target_guarded_v5_recovery_step_m=(
                    float(fields["target_guarded_v5_recovery_step_m"])
                    if "target_guarded_v5_recovery_step_m" in fields
                    else None
                ),
                target_guarded_v5_preemptive_recovery_writes_total=(
                    int(fields["target_guarded_v5_preemptive_recovery_writes_total"])
                    if "target_guarded_v5_preemptive_recovery_writes_total" in fields
                    else None
                ),
                target_guarded_v5_recovery_ik_failures_total=(
                    int(fields["target_guarded_v5_recovery_ik_failures_total"])
                    if "target_guarded_v5_recovery_ik_failures_total" in fields
                    else None
                ),
                target_guarded_v7_active_recovery_needed=(
                    _parse_bool(fields["target_guarded_v7_active_recovery_needed"])
                    if "target_guarded_v7_active_recovery_needed" in fields
                    else None
                ),
                target_guarded_v7_active_recovery=(
                    _parse_bool(fields["target_guarded_v7_active_recovery"])
                    if "target_guarded_v7_active_recovery" in fields
                    else None
                ),
                target_guarded_v7_recovery_ik_ok=(
                    _parse_bool(fields["target_guarded_v7_recovery_ik_ok"])
                    if "target_guarded_v7_recovery_ik_ok" in fields
                    else None
                ),
                target_guarded_v7_candidate_count=(
                    int(fields["target_guarded_v7_candidate_count"])
                    if "target_guarded_v7_candidate_count" in fields
                    else None
                ),
                target_guarded_v7_selected_score=(
                    float(fields["target_guarded_v7_selected_score"])
                    if "target_guarded_v7_selected_score" in fields
                    else None
                ),
                target_guarded_v7_best_target_margin_m=(
                    float(fields["target_guarded_v7_best_target_margin_m"])
                    if "target_guarded_v7_best_target_margin_m" in fields
                    else None
                ),
                target_guarded_v7_best_support_margin_m=(
                    float(fields["target_guarded_v7_best_support_margin_m"])
                    if "target_guarded_v7_best_support_margin_m" in fields
                    else None
                ),
                target_guarded_v7_counter_gap_delta_m=(
                    float(fields["target_guarded_v7_counter_gap_delta_m"])
                    if "target_guarded_v7_counter_gap_delta_m" in fields
                    else None
                ),
                target_guarded_v7_recovery_step_m=(
                    float(fields["target_guarded_v7_recovery_step_m"])
                    if "target_guarded_v7_recovery_step_m" in fields
                    else None
                ),
                target_guarded_v7_active_recovery_writes_total=(
                    int(fields["target_guarded_v7_active_recovery_writes_total"])
                    if "target_guarded_v7_active_recovery_writes_total" in fields
                    else None
                ),
                target_guarded_v7_recovery_ik_failures_total=(
                    int(fields["target_guarded_v7_recovery_ik_failures_total"])
                    if "target_guarded_v7_recovery_ik_failures_total" in fields
                    else None
                ),
            )
        elif "[cube2cm_runtime_jaw_telemetry] aggregate " in line:
            fields = _fields(line)
            aggregate = AggregateObservation(
                source=f"{path}:{line_no}",
                approach_ok=_parse_bool(fields["approach_ok"]),
                descend_ok=_parse_bool(fields["descend_ok"]),
                close_reached=_parse_bool(fields["close_reached"]),
                close_early_kill=_parse_bool(fields["close_early_kill"]),
                attach_calls=int(fields["attach_calls"]),
                posewrite_calls=int(fields["posewrite_calls"]),
                virtual_velocity_damping_writes=(
                    int(fields["virtual_velocity_damping_writes"])
                    if "virtual_velocity_damping_writes" in fields
                    else None
                ),
                target_guarded_close_advances=(
                    int(fields["target_guarded_close_advances"])
                    if "target_guarded_close_advances" in fields
                    else None
                ),
                target_guarded_close_holds=(
                    int(fields["target_guarded_close_holds"])
                    if "target_guarded_close_holds" in fields
                    else None
                ),
                telemetry_only=_parse_bool(fields["telemetry_only"]),
                success_claim=_parse_bool(fields["success_claim"]),
                target_guarded_zero_backlog_holds=(
                    int(fields["target_guarded_zero_backlog_holds"])
                    if "target_guarded_zero_backlog_holds" in fields
                    else None
                ),
                target_guarded_backlog_preserved_holds=(
                    int(fields["target_guarded_backlog_preserved_holds"])
                    if "target_guarded_backlog_preserved_holds" in fields
                    else None
                ),
                target_guarded_safety_rollbacks=(
                    int(fields["target_guarded_safety_rollbacks"])
                    if "target_guarded_safety_rollbacks" in fields
                    else None
                ),
                target_guarded_v4_recovery_holds=(
                    int(fields["target_guarded_v4_recovery_holds"])
                    if "target_guarded_v4_recovery_holds" in fields
                    else None
                ),
                target_guarded_v4_hard_safety_freezes=(
                    int(fields["target_guarded_v4_hard_safety_freezes"])
                    if "target_guarded_v4_hard_safety_freezes" in fields
                    else None
                ),
                target_guarded_v5_preemptive_recovery_writes=(
                    int(fields["target_guarded_v5_preemptive_recovery_writes"])
                    if "target_guarded_v5_preemptive_recovery_writes" in fields
                    else None
                ),
                target_guarded_v5_recovery_ik_failures=(
                    int(fields["target_guarded_v5_recovery_ik_failures"])
                    if "target_guarded_v5_recovery_ik_failures" in fields
                    else None
                ),
                target_guarded_v7_active_recovery_writes=(
                    int(fields["target_guarded_v7_active_recovery_writes"])
                    if "target_guarded_v7_active_recovery_writes" in fields
                    else None
                ),
                target_guarded_v7_recovery_ik_failures=(
                    int(fields["target_guarded_v7_recovery_ik_failures"])
                    if "target_guarded_v7_recovery_ik_failures" in fields
                    else None
                ),
            )
    metadata = RuntimeMetadata(
        source=metadata_source,
        soft_contact_material_diagnostic=soft_contact_material_diagnostic,
        virtual_compression_damping_diagnostic=virtual_compression_damping_diagnostic,
        target_guarded_micro_close_support_horizon_diagnostic=(
            target_guarded_micro_close_support_horizon_diagnostic
        ),
        object_physics_mode=object_physics_mode,
        runtime_candidate_requires_separate_approval=runtime_candidate_requires_separate_approval,
        target_guarded_micro_close_v2_convergence_diagnostic=(
            target_guarded_micro_close_v2_convergence_diagnostic
        ),
        target_guarded_micro_close_v3_progress_diagnostic=(
            target_guarded_micro_close_v3_progress_diagnostic
        ),
        target_guarded_micro_close_v4_recovery_diagnostic=(
            target_guarded_micro_close_v4_recovery_diagnostic
        ),
        target_guarded_micro_close_v5_preemptive_recovery_diagnostic=(
            target_guarded_micro_close_v5_preemptive_recovery_diagnostic
        ),
        target_guarded_micro_close_v6_projected_guard_diagnostic=(
            target_guarded_micro_close_v6_projected_guard_diagnostic
        ),
        target_guarded_micro_close_v7_active_recovery_diagnostic=(
            target_guarded_micro_close_v7_active_recovery_diagnostic
        ),
    )
    return close, aggregate, metadata


def _reference_v7() -> tuple[dict[int, CloseObservation], AggregateObservation, RuntimeMetadata]:
    close = {
        sample.step: CloseObservation(
            source=sample.source,
            step=sample.step,
            target_error_m=sample.target_error_m,
            object_speed_mps=sample.object_speed_mps,
            counter_gap_max_m=max(sample.counter_gap_m),
            counter_contact=sample.logged_counter_contact,
            counter_slop_contact=sample.logged_counter_slop_contact_1mm,
            one_sided_push=sample.logged_one_sided_push,
            virtual_support=None,
            support_horizon_active=None,
            virtual_damping_active=None,
            virtual_velocity_damping_writes_total=None,
            target_guarded_close_advance=None,
            target_guarded_close_hold=None,
            target_guarded_close_advances_total=None,
            target_guarded_close_holds_total=None,
            reached=sample.reached,
            early_kill=False,
        )
        for sample in V7_CLOSE_SAMPLES
    }
    aggregate = AggregateObservation(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:420",
        approach_ok=True,
        descend_ok=True,
        close_reached=False,
        close_early_kill=False,
        attach_calls=0,
        posewrite_calls=0,
        virtual_velocity_damping_writes=None,
        target_guarded_close_advances=None,
        target_guarded_close_holds=None,
        telemetry_only=True,
        success_claim=False,
    )
    metadata = RuntimeMetadata(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38",
        soft_contact_material_diagnostic=False,
        virtual_compression_damping_diagnostic=False,
        target_guarded_micro_close_support_horizon_diagnostic=False,
        object_physics_mode="baseline",
        runtime_candidate_requires_separate_approval=False,
    )
    return close, aggregate, metadata


def _reference_synthetic_pass(expected_mechanism: str) -> tuple[dict[int, CloseObservation], AggregateObservation, RuntimeMetadata]:
    is_virtual = expected_mechanism in VIRTUAL_DAMPING_MECHANISMS
    is_target_guarded = expected_mechanism in TARGET_GUARDED_MECHANISMS
    is_target_guarded_v2 = expected_mechanism == TARGET_GUARDED_V2_MECHANISM
    is_target_guarded_v3 = expected_mechanism == TARGET_GUARDED_V3_MECHANISM
    is_target_guarded_v4 = expected_mechanism == TARGET_GUARDED_V4_MECHANISM
    is_target_guarded_v5 = expected_mechanism == TARGET_GUARDED_V5_MECHANISM
    is_target_guarded_v6 = expected_mechanism == TARGET_GUARDED_V6_MECHANISM
    is_target_guarded_v7 = expected_mechanism == TARGET_GUARDED_V7_MECHANISM
    is_v5_or_later = is_target_guarded_v5 or is_target_guarded_v6 or is_target_guarded_v7
    is_progress_preserving = is_target_guarded_v3 or is_target_guarded_v4 or is_v5_or_later
    is_v4_or_later = is_target_guarded_v4 or is_v5_or_later
    close = {
        2: CloseObservation(
            source="synthetic_pass:close_step_002",
            step=2,
            target_error_m=0.001200,
            object_speed_mps=0.001000,
            counter_gap_max_m=0.000400,
            counter_contact=False,
            counter_slop_contact=True,
            one_sided_push=False,
            virtual_support=True if is_virtual else None,
            support_horizon_active=True if is_virtual else None,
            virtual_damping_active=False if is_virtual else None,
            virtual_velocity_damping_writes_total=0 if is_virtual else None,
            target_guarded_close_advance=True if is_target_guarded else None,
            target_guarded_close_hold=False if is_target_guarded else None,
            target_guarded_close_advances_total=1 if is_target_guarded else None,
            target_guarded_close_holds_total=0 if is_target_guarded else None,
            reached=False,
            early_kill=False,
            target_guarded_zero_backlog_hold=False if is_target_guarded_v2 else None,
            target_guarded_command_backlog_deg=0.100 if is_target_guarded_v2 else None,
            target_guarded_command_converged=True if is_target_guarded_v2 else None,
            target_guarded_support_margin_ok=True if is_target_guarded_v2 else None,
            target_guarded_target_nonworsening=True if is_target_guarded_v2 else None,
            target_guarded_zero_backlog_holds_total=0 if is_target_guarded_v2 else None,
            target_guarded_backlog_preserved_hold=False if is_progress_preserving else None,
            target_guarded_v3_safety_rollback=False if is_progress_preserving else None,
            target_guarded_support_budget_ok=True if is_progress_preserving else None,
            target_guarded_v3_safety_ok=True if is_progress_preserving else None,
            target_guarded_v3_actual_progress_deg=0.000 if is_progress_preserving else None,
            target_guarded_v3_actual_progress_ok=False if is_progress_preserving else None,
            target_guarded_v3_progress_gate_ok=True if is_progress_preserving else None,
            target_guarded_v3_backlog_room_ok=True if is_progress_preserving else None,
            target_guarded_v3_projected_backlog_after_advance_deg=2.000 if is_progress_preserving else None,
            target_guarded_v4_recovery_hold=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freeze=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_ok=True if is_v4_or_later else None,
            target_guarded_v4_recovery_ready=True if is_v4_or_later else None,
            target_guarded_v4_target_error_recovered=True if is_v4_or_later else None,
            target_guarded_backlog_preserved_holds_total=0 if is_progress_preserving else None,
            target_guarded_safety_rollbacks_total=0 if is_progress_preserving else None,
            target_guarded_v4_recovery_holds_total=0 if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freezes_total=0 if is_v4_or_later else None,
            target_guarded_v5_preemptive_recovery_needed=False if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery=False if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_ok=True if is_v5_or_later else None,
            target_guarded_v5_target_margin_m=0.001800 if is_v5_or_later else None,
            target_guarded_v5_support_margin_m=0.001600 if is_v5_or_later else None,
            target_guarded_v5_recovery_step_m=0.0 if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery_writes_total=0 if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_failures_total=0 if is_v5_or_later else None,
            target_guarded_v7_active_recovery_needed=False if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery=False if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_ok=True if is_target_guarded_v7 else None,
            target_guarded_v7_candidate_count=0 if is_target_guarded_v7 else None,
            target_guarded_v7_selected_score=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_best_target_margin_m=0.001800 if is_target_guarded_v7 else None,
            target_guarded_v7_best_support_margin_m=0.001600 if is_target_guarded_v7 else None,
            target_guarded_v7_counter_gap_delta_m=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_step_m=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery_writes_total=0 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_failures_total=0 if is_target_guarded_v7 else None,
        ),
        3: CloseObservation(
            source="synthetic_pass:close_step_003",
            step=3,
            target_error_m=0.001600,
            object_speed_mps=0.004900,
            counter_gap_max_m=0.001200,
            counter_contact=False,
            counter_slop_contact=True,
            one_sided_push=False,
            virtual_support=True if is_virtual else None,
            support_horizon_active=True if is_virtual else None,
            virtual_damping_active=True if is_virtual else None,
            virtual_velocity_damping_writes_total=1 if is_virtual else None,
            target_guarded_close_advance=(False if is_progress_preserving else True) if is_target_guarded else None,
            target_guarded_close_hold=True if is_progress_preserving else (False if is_target_guarded else None),
            target_guarded_close_advances_total=(1 if is_progress_preserving else 2) if is_target_guarded else None,
            target_guarded_close_holds_total=(1 if is_progress_preserving else 0) if is_target_guarded else None,
            reached=False,
            early_kill=False,
            target_guarded_zero_backlog_hold=False if is_target_guarded_v2 else None,
            target_guarded_command_backlog_deg=0.100 if is_target_guarded_v2 else None,
            target_guarded_command_converged=True if is_target_guarded_v2 else None,
            target_guarded_support_margin_ok=True if is_target_guarded_v2 else None,
            target_guarded_target_nonworsening=True if is_target_guarded_v2 else None,
            target_guarded_zero_backlog_holds_total=0 if is_target_guarded_v2 else None,
            target_guarded_backlog_preserved_hold=True if is_progress_preserving else None,
            target_guarded_v3_safety_rollback=False if is_progress_preserving else None,
            target_guarded_support_budget_ok=True if is_progress_preserving else None,
            target_guarded_v3_safety_ok=True if is_progress_preserving else None,
            target_guarded_v3_actual_progress_deg=0.360 if is_progress_preserving else None,
            target_guarded_v3_actual_progress_ok=True if is_progress_preserving else None,
            target_guarded_v3_progress_gate_ok=True if is_progress_preserving else None,
            target_guarded_v3_backlog_room_ok=True if is_progress_preserving else None,
            target_guarded_v3_projected_backlog_after_advance_deg=3.640 if is_progress_preserving else None,
            target_guarded_v4_recovery_hold=True if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freeze=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_ok=True if is_v4_or_later else None,
            target_guarded_v4_recovery_ready=False if is_v4_or_later else None,
            target_guarded_v4_target_error_recovered=False if is_v4_or_later else None,
            target_guarded_backlog_preserved_holds_total=1 if is_progress_preserving else None,
            target_guarded_safety_rollbacks_total=0 if is_progress_preserving else None,
            target_guarded_v4_recovery_holds_total=1 if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freezes_total=0 if is_v4_or_later else None,
            target_guarded_v5_preemptive_recovery_needed=False if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery=True if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_ok=True if is_v5_or_later else None,
            target_guarded_v5_target_margin_m=0.001400 if is_v5_or_later else None,
            target_guarded_v5_support_margin_m=0.000800 if is_v5_or_later else None,
            target_guarded_v5_recovery_step_m=0.0010 if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery_writes_total=1 if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_failures_total=0 if is_v5_or_later else None,
            target_guarded_v7_active_recovery_needed=True if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery=True if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_ok=True if is_target_guarded_v7 else None,
            target_guarded_v7_candidate_count=18 if is_target_guarded_v7 else None,
            target_guarded_v7_selected_score=0.000700 if is_target_guarded_v7 else None,
            target_guarded_v7_best_target_margin_m=0.001400 if is_target_guarded_v7 else None,
            target_guarded_v7_best_support_margin_m=0.000800 if is_target_guarded_v7 else None,
            target_guarded_v7_counter_gap_delta_m=-0.000120 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_step_m=0.0010 if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery_writes_total=1 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_failures_total=0 if is_target_guarded_v7 else None,
        ),
        4: CloseObservation(
            source="synthetic_pass:close_step_004",
            step=4,
            target_error_m=0.002900,
            object_speed_mps=0.004000,
            counter_gap_max_m=0.001900,
            counter_contact=False,
            counter_slop_contact=True,
            one_sided_push=False,
            virtual_support=True if is_virtual else None,
            support_horizon_active=True if is_virtual else None,
            virtual_damping_active=True if is_virtual else None,
            virtual_velocity_damping_writes_total=2 if is_virtual else None,
            target_guarded_close_advance=True if is_target_guarded else None,
            target_guarded_close_hold=False if is_target_guarded else None,
            target_guarded_close_advances_total=(2 if is_progress_preserving else 3) if is_target_guarded else None,
            target_guarded_close_holds_total=(1 if is_progress_preserving else 0) if is_target_guarded else None,
            reached=False,
            early_kill=False,
            target_guarded_zero_backlog_hold=False if is_target_guarded_v2 else None,
            target_guarded_command_backlog_deg=0.100 if is_target_guarded_v2 else None,
            target_guarded_command_converged=True if is_target_guarded_v2 else None,
            target_guarded_support_margin_ok=True if is_target_guarded_v2 else None,
            target_guarded_target_nonworsening=True if is_target_guarded_v2 else None,
            target_guarded_zero_backlog_holds_total=0 if is_target_guarded_v2 else None,
            target_guarded_backlog_preserved_hold=False if is_progress_preserving else None,
            target_guarded_v3_safety_rollback=False if is_progress_preserving else None,
            target_guarded_support_budget_ok=True if is_progress_preserving else None,
            target_guarded_v3_safety_ok=True if is_progress_preserving else None,
            target_guarded_v3_actual_progress_deg=0.420 if is_progress_preserving else None,
            target_guarded_v3_actual_progress_ok=True if is_progress_preserving else None,
            target_guarded_v3_progress_gate_ok=True if is_progress_preserving else None,
            target_guarded_v3_backlog_room_ok=True if is_progress_preserving else None,
            target_guarded_v3_projected_backlog_after_advance_deg=4.200 if is_progress_preserving else None,
            target_guarded_v4_recovery_hold=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freeze=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_ok=True if is_v4_or_later else None,
            target_guarded_v4_recovery_ready=True if is_v4_or_later else None,
            target_guarded_v4_target_error_recovered=True if is_v4_or_later else None,
            target_guarded_backlog_preserved_holds_total=1 if is_progress_preserving else None,
            target_guarded_safety_rollbacks_total=0 if is_progress_preserving else None,
            target_guarded_v4_recovery_holds_total=1 if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freezes_total=0 if is_v4_or_later else None,
            target_guarded_v5_preemptive_recovery_needed=True if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery=True if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_ok=True if is_v5_or_later else None,
            target_guarded_v5_target_margin_m=0.000100 if is_v5_or_later else None,
            target_guarded_v5_support_margin_m=0.000100 if is_v5_or_later else None,
            target_guarded_v5_recovery_step_m=0.0012 if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery_writes_total=2 if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_failures_total=0 if is_v5_or_later else None,
            target_guarded_v7_active_recovery_needed=True if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery=True if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_ok=True if is_target_guarded_v7 else None,
            target_guarded_v7_candidate_count=18 if is_target_guarded_v7 else None,
            target_guarded_v7_selected_score=0.000080 if is_target_guarded_v7 else None,
            target_guarded_v7_best_target_margin_m=0.000100 if is_target_guarded_v7 else None,
            target_guarded_v7_best_support_margin_m=0.000100 if is_target_guarded_v7 else None,
            target_guarded_v7_counter_gap_delta_m=-0.000090 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_step_m=0.0012 if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery_writes_total=2 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_failures_total=0 if is_target_guarded_v7 else None,
        ),
        5: CloseObservation(
            source="synthetic_pass:close_step_005",
            step=5,
            target_error_m=0.002700,
            object_speed_mps=0.004200,
            counter_gap_max_m=0.001800 if is_progress_preserving else 0.002700,
            counter_contact=False,
            counter_slop_contact=False,
            one_sided_push=False,
            virtual_support=(True if is_progress_preserving else False) if is_virtual else None,
            support_horizon_active=True if is_virtual else None,
            virtual_damping_active=True if is_virtual else None,
            virtual_velocity_damping_writes_total=3 if is_virtual else None,
            target_guarded_close_advance=True if is_target_guarded else None,
            target_guarded_close_hold=False if is_target_guarded else None,
            target_guarded_close_advances_total=(3 if is_progress_preserving else 4) if is_target_guarded else None,
            target_guarded_close_holds_total=(1 if is_progress_preserving else 0) if is_target_guarded else None,
            reached=False,
            early_kill=False,
            target_guarded_zero_backlog_hold=False if is_target_guarded_v2 else None,
            target_guarded_command_backlog_deg=0.100 if is_target_guarded_v2 else None,
            target_guarded_command_converged=True if is_target_guarded_v2 else None,
            target_guarded_support_margin_ok=True if is_target_guarded_v2 else None,
            target_guarded_target_nonworsening=True if is_target_guarded_v2 else None,
            target_guarded_zero_backlog_holds_total=0 if is_target_guarded_v2 else None,
            target_guarded_backlog_preserved_hold=False if is_progress_preserving else None,
            target_guarded_v3_safety_rollback=False if is_progress_preserving else None,
            target_guarded_support_budget_ok=True if is_progress_preserving else None,
            target_guarded_v3_safety_ok=True if is_progress_preserving else None,
            target_guarded_v3_actual_progress_deg=0.390 if is_progress_preserving else None,
            target_guarded_v3_actual_progress_ok=True if is_progress_preserving else None,
            target_guarded_v3_progress_gate_ok=True if is_progress_preserving else None,
            target_guarded_v3_backlog_room_ok=True if is_progress_preserving else None,
            target_guarded_v3_projected_backlog_after_advance_deg=4.100 if is_progress_preserving else None,
            target_guarded_v4_recovery_hold=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freeze=False if is_v4_or_later else None,
            target_guarded_v4_hard_safety_ok=True if is_v4_or_later else None,
            target_guarded_v4_recovery_ready=True if is_v4_or_later else None,
            target_guarded_v4_target_error_recovered=True if is_v4_or_later else None,
            target_guarded_backlog_preserved_holds_total=1 if is_progress_preserving else None,
            target_guarded_safety_rollbacks_total=0 if is_progress_preserving else None,
            target_guarded_v4_recovery_holds_total=1 if is_v4_or_later else None,
            target_guarded_v4_hard_safety_freezes_total=0 if is_v4_or_later else None,
            target_guarded_v5_preemptive_recovery_needed=False if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery=False if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_ok=True if is_v5_or_later else None,
            target_guarded_v5_target_margin_m=0.000300 if is_v5_or_later else None,
            target_guarded_v5_support_margin_m=0.000200 if is_v5_or_later else None,
            target_guarded_v5_recovery_step_m=0.0 if is_v5_or_later else None,
            target_guarded_v5_preemptive_recovery_writes_total=2 if is_v5_or_later else None,
            target_guarded_v5_recovery_ik_failures_total=0 if is_v5_or_later else None,
            target_guarded_v7_active_recovery_needed=False if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery=False if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_ok=True if is_target_guarded_v7 else None,
            target_guarded_v7_candidate_count=0 if is_target_guarded_v7 else None,
            target_guarded_v7_selected_score=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_best_target_margin_m=0.000300 if is_target_guarded_v7 else None,
            target_guarded_v7_best_support_margin_m=0.000200 if is_target_guarded_v7 else None,
            target_guarded_v7_counter_gap_delta_m=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_step_m=0.0 if is_target_guarded_v7 else None,
            target_guarded_v7_active_recovery_writes_total=2 if is_target_guarded_v7 else None,
            target_guarded_v7_recovery_ik_failures_total=0 if is_target_guarded_v7 else None,
        ),
    }
    aggregate = AggregateObservation(
        source="synthetic_pass:aggregate",
        approach_ok=True,
        descend_ok=True,
        close_reached=True,
        close_early_kill=False,
        attach_calls=0,
        posewrite_calls=0,
        virtual_velocity_damping_writes=3 if is_virtual else None,
        target_guarded_close_advances=(3 if is_progress_preserving else 4) if is_target_guarded else None,
        target_guarded_close_holds=(1 if is_progress_preserving else 0) if is_target_guarded else None,
        telemetry_only=True,
        success_claim=False,
        target_guarded_zero_backlog_holds=0 if (is_target_guarded_v2 or is_progress_preserving) else None,
        target_guarded_backlog_preserved_holds=1 if is_progress_preserving else None,
        target_guarded_safety_rollbacks=0 if is_progress_preserving else None,
        target_guarded_v4_recovery_holds=1 if is_v4_or_later else None,
        target_guarded_v4_hard_safety_freezes=0 if is_v4_or_later else None,
        target_guarded_v5_preemptive_recovery_writes=2 if is_v5_or_later else None,
        target_guarded_v5_recovery_ik_failures=0 if is_v5_or_later else None,
        target_guarded_v7_active_recovery_writes=2 if is_target_guarded_v7 else None,
        target_guarded_v7_recovery_ik_failures=0 if is_target_guarded_v7 else None,
    )
    metadata = RuntimeMetadata(
        source="synthetic_pass:metadata",
        soft_contact_material_diagnostic=expected_mechanism == "soft_contact_material_diagnostic",
        virtual_compression_damping_diagnostic=expected_mechanism == "virtual_compression_damping_diagnostic",
        target_guarded_micro_close_support_horizon_diagnostic=expected_mechanism == TARGET_GUARDED_MECHANISM,
        object_physics_mode=expected_mechanism,
        runtime_candidate_requires_separate_approval=True,
        target_guarded_micro_close_v2_convergence_diagnostic=is_target_guarded_v2,
        target_guarded_micro_close_v3_progress_diagnostic=is_target_guarded_v3,
        target_guarded_micro_close_v4_recovery_diagnostic=is_target_guarded_v4,
        target_guarded_micro_close_v5_preemptive_recovery_diagnostic=is_target_guarded_v5,
        target_guarded_micro_close_v6_projected_guard_diagnostic=is_target_guarded_v6,
        target_guarded_micro_close_v7_active_recovery_diagnostic=is_target_guarded_v7,
    )
    return close, aggregate, metadata


def _reference_synthetic_virtual_no_damping(expected_mechanism: str) -> tuple[
    dict[int, CloseObservation], AggregateObservation, RuntimeMetadata
]:
    close, aggregate, metadata = _reference_synthetic_pass(expected_mechanism)
    close = {
        step: CloseObservation(
            source=f"synthetic_virtual_no_damping:close_step_{step:03d}",
            step=obs.step,
            target_error_m=obs.target_error_m,
            object_speed_mps=obs.object_speed_mps,
            counter_gap_max_m=obs.counter_gap_max_m,
            counter_contact=obs.counter_contact,
            counter_slop_contact=obs.counter_slop_contact,
            one_sided_push=obs.one_sided_push,
            virtual_support=obs.virtual_support,
            support_horizon_active=obs.support_horizon_active,
            virtual_damping_active=False,
            virtual_velocity_damping_writes_total=0,
            target_guarded_close_advance=obs.target_guarded_close_advance,
            target_guarded_close_hold=obs.target_guarded_close_hold,
            target_guarded_close_advances_total=obs.target_guarded_close_advances_total,
            target_guarded_close_holds_total=obs.target_guarded_close_holds_total,
            reached=obs.reached,
            early_kill=obs.early_kill,
            target_guarded_zero_backlog_hold=obs.target_guarded_zero_backlog_hold,
            target_guarded_command_backlog_deg=obs.target_guarded_command_backlog_deg,
            target_guarded_command_converged=obs.target_guarded_command_converged,
            target_guarded_support_margin_ok=obs.target_guarded_support_margin_ok,
            target_guarded_target_nonworsening=obs.target_guarded_target_nonworsening,
            target_guarded_zero_backlog_holds_total=obs.target_guarded_zero_backlog_holds_total,
            target_guarded_backlog_preserved_hold=obs.target_guarded_backlog_preserved_hold,
            target_guarded_v3_safety_rollback=obs.target_guarded_v3_safety_rollback,
            target_guarded_support_budget_ok=obs.target_guarded_support_budget_ok,
            target_guarded_v3_safety_ok=obs.target_guarded_v3_safety_ok,
            target_guarded_v3_actual_progress_deg=obs.target_guarded_v3_actual_progress_deg,
            target_guarded_v3_actual_progress_ok=obs.target_guarded_v3_actual_progress_ok,
            target_guarded_v3_progress_gate_ok=obs.target_guarded_v3_progress_gate_ok,
            target_guarded_v3_backlog_room_ok=obs.target_guarded_v3_backlog_room_ok,
            target_guarded_v3_projected_backlog_after_advance_deg=(
                obs.target_guarded_v3_projected_backlog_after_advance_deg
            ),
            target_guarded_backlog_preserved_holds_total=obs.target_guarded_backlog_preserved_holds_total,
            target_guarded_safety_rollbacks_total=obs.target_guarded_safety_rollbacks_total,
        )
        for step, obs in close.items()
    }
    aggregate = AggregateObservation(
        source="synthetic_virtual_no_damping:aggregate",
        approach_ok=aggregate.approach_ok,
        descend_ok=aggregate.descend_ok,
        close_reached=aggregate.close_reached,
        close_early_kill=aggregate.close_early_kill,
        attach_calls=aggregate.attach_calls,
        posewrite_calls=aggregate.posewrite_calls,
        virtual_velocity_damping_writes=0,
        target_guarded_close_advances=aggregate.target_guarded_close_advances,
        target_guarded_close_holds=aggregate.target_guarded_close_holds,
        telemetry_only=aggregate.telemetry_only,
        success_claim=aggregate.success_claim,
        target_guarded_zero_backlog_holds=aggregate.target_guarded_zero_backlog_holds,
        target_guarded_backlog_preserved_holds=aggregate.target_guarded_backlog_preserved_holds,
        target_guarded_safety_rollbacks=aggregate.target_guarded_safety_rollbacks,
    )
    return close, aggregate, metadata


def _reference_synthetic_v3_zero_backlog() -> tuple[
    dict[int, CloseObservation], AggregateObservation, RuntimeMetadata
]:
    close, aggregate, metadata = _reference_synthetic_pass(TARGET_GUARDED_V3_MECHANISM)
    close = dict(close)
    close[3] = replace(
        close[3],
        source="synthetic_v3_zero_backlog:close_step_003",
        target_guarded_zero_backlog_hold=True,
        target_guarded_backlog_preserved_hold=False,
    )
    aggregate = replace(
        aggregate,
        source="synthetic_v3_zero_backlog:aggregate",
        target_guarded_zero_backlog_holds=1,
        target_guarded_backlog_preserved_holds=0,
    )
    return close, aggregate, metadata


def _reference_synthetic_v4_hard_freeze() -> tuple[
    dict[int, CloseObservation], AggregateObservation, RuntimeMetadata
]:
    close, aggregate, metadata = _reference_synthetic_pass(TARGET_GUARDED_V4_MECHANISM)
    close = dict(close)
    close[4] = replace(
        close[4],
        source="synthetic_v4_hard_freeze:close_step_004",
        target_error_m=0.003100,
        counter_gap_max_m=0.002100,
        target_guarded_v4_hard_safety_freeze=True,
        target_guarded_v4_hard_safety_ok=False,
    )
    aggregate = replace(
        aggregate,
        source="synthetic_v4_hard_freeze:aggregate",
        target_guarded_v4_hard_safety_freezes=1,
    )
    return close, aggregate, metadata


def _reference_synthetic_v7_no_active_recovery() -> tuple[
    dict[int, CloseObservation], AggregateObservation, RuntimeMetadata
]:
    close, aggregate, metadata = _reference_synthetic_pass(TARGET_GUARDED_V7_MECHANISM)
    close = dict(close)
    close[3] = replace(
        close[3],
        source="synthetic_v7_no_active_recovery:close_step_003",
        target_guarded_v7_active_recovery_needed=True,
        target_guarded_v7_active_recovery=False,
        target_guarded_v7_active_recovery_writes_total=0,
        target_guarded_v7_candidate_count=18,
        target_guarded_v7_counter_gap_delta_m=0.0,
    )
    close[4] = replace(
        close[4],
        source="synthetic_v7_no_active_recovery:close_step_004",
        target_guarded_v7_active_recovery_needed=True,
        target_guarded_v7_active_recovery=False,
        target_guarded_v7_active_recovery_writes_total=0,
        target_guarded_v7_counter_gap_delta_m=0.0,
    )
    aggregate = replace(
        aggregate,
        source="synthetic_v7_no_active_recovery:aggregate",
        target_guarded_v7_active_recovery_writes=0,
    )
    return close, aggregate, metadata


def _criterion(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return name, ok, detail


def _audit(
    close: dict[int, CloseObservation],
    aggregate: AggregateObservation | None,
    metadata: RuntimeMetadata | None,
    expected_mechanism: str,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    missing = [step for step in REQUIRED_CLOSE_STEPS if step not in close]
    checks.append(_criterion("required_close_steps_present", not missing, f"missing={missing}"))

    if metadata is None:
        checks.append(_criterion("metadata_present", False, "runtime metadata missing"))
    else:
        checks.append(_criterion("metadata_present", True, f"source={metadata.source}"))
        if expected_mechanism == "soft_contact_material_diagnostic":
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_enabled",
                    metadata.soft_contact_material_diagnostic is True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism == "virtual_compression_damping_diagnostic":
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_enabled",
                    metadata.virtual_compression_damping_diagnostic is True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism == TARGET_GUARDED_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_enabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism == TARGET_GUARDED_V2_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_enabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism == TARGET_GUARDED_V3_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_enabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism == TARGET_GUARDED_V4_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v4_recovery_diagnostic_enabled",
                    metadata.target_guarded_micro_close_v4_recovery_diagnostic is True,
                    f"value={metadata.target_guarded_micro_close_v4_recovery_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
        elif expected_mechanism in (
            TARGET_GUARDED_V5_MECHANISM,
            TARGET_GUARDED_V6_MECHANISM,
            TARGET_GUARDED_V7_MECHANISM,
        ):
            if expected_mechanism == TARGET_GUARDED_V5_MECHANISM:
                checks.append(
                    _criterion(
                        "target_guarded_micro_close_v5_preemptive_recovery_diagnostic_enabled",
                        metadata.target_guarded_micro_close_v5_preemptive_recovery_diagnostic is True,
                        f"value={metadata.target_guarded_micro_close_v5_preemptive_recovery_diagnostic} source={metadata.source}",
                    )
                )
            elif expected_mechanism == TARGET_GUARDED_V6_MECHANISM:
                checks.append(
                    _criterion(
                        "target_guarded_micro_close_v6_projected_guard_diagnostic_enabled",
                        metadata.target_guarded_micro_close_v6_projected_guard_diagnostic is True,
                        f"value={metadata.target_guarded_micro_close_v6_projected_guard_diagnostic} source={metadata.source}",
                    )
                )
            else:
                checks.append(
                    _criterion(
                        "target_guarded_micro_close_v7_active_recovery_diagnostic_enabled",
                        metadata.target_guarded_micro_close_v7_active_recovery_diagnostic is True,
                        f"value={metadata.target_guarded_micro_close_v7_active_recovery_diagnostic} source={metadata.source}",
                    )
                )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_support_horizon_diagnostic_disabled",
                    metadata.target_guarded_micro_close_support_horizon_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_support_horizon_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v2_convergence_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v2_convergence_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v2_convergence_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v3_progress_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v3_progress_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v3_progress_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "soft_contact_material_diagnostic_disabled",
                    metadata.soft_contact_material_diagnostic is not True,
                    f"value={metadata.soft_contact_material_diagnostic} source={metadata.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_compression_damping_diagnostic_disabled",
                    metadata.virtual_compression_damping_diagnostic is not True,
                    f"value={metadata.virtual_compression_damping_diagnostic} source={metadata.source}",
                )
            )
        else:
            raise ValueError(f"unsupported expected mechanism {expected_mechanism!r}")
        if expected_mechanism != TARGET_GUARDED_V4_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v4_recovery_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v4_recovery_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v4_recovery_diagnostic} source={metadata.source}",
                )
            )
        if expected_mechanism != TARGET_GUARDED_V5_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v5_preemptive_recovery_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v5_preemptive_recovery_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v5_preemptive_recovery_diagnostic} source={metadata.source}",
                )
            )
        if expected_mechanism != TARGET_GUARDED_V6_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v6_projected_guard_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v6_projected_guard_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v6_projected_guard_diagnostic} source={metadata.source}",
                )
            )
        if expected_mechanism != TARGET_GUARDED_V7_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_micro_close_v7_active_recovery_diagnostic_disabled",
                    metadata.target_guarded_micro_close_v7_active_recovery_diagnostic is not True,
                    f"value={metadata.target_guarded_micro_close_v7_active_recovery_diagnostic} source={metadata.source}",
                )
            )
        checks.append(
            _criterion(
                "object_physics_mode_matches_expected",
                metadata.object_physics_mode == expected_mechanism,
                f"value={metadata.object_physics_mode} expected={expected_mechanism} source={metadata.source}",
            )
        )
        checks.append(
            _criterion(
                "runtime_candidate_marker_yes",
                metadata.runtime_candidate_requires_separate_approval is True,
                f"value={metadata.runtime_candidate_requires_separate_approval} source={metadata.source}",
            )
        )

    if aggregate is None:
        checks.append(_criterion("aggregate_present", False, "aggregate line missing"))
    else:
        checks.append(_criterion("aggregate_present", True, f"source={aggregate.source}"))
        checks.append(_criterion("approach_ok", aggregate.approach_ok, f"source={aggregate.source}"))
        checks.append(_criterion("descend_ok", aggregate.descend_ok, f"source={aggregate.source}"))
        checks.append(_criterion("close_reached", aggregate.close_reached, f"source={aggregate.source}"))
        checks.append(_criterion("close_early_kill_no", not aggregate.close_early_kill, f"source={aggregate.source}"))
        checks.append(_criterion("attach_calls_zero", aggregate.attach_calls == 0, f"attach_calls={aggregate.attach_calls}"))
        checks.append(
            _criterion("posewrite_calls_zero", aggregate.posewrite_calls == 0, f"posewrite_calls={aggregate.posewrite_calls}")
        )
        checks.append(_criterion("telemetry_only_yes", aggregate.telemetry_only, f"source={aggregate.source}"))
        checks.append(_criterion("success_claim_no", not aggregate.success_claim, f"source={aggregate.source}"))
        if expected_mechanism in VIRTUAL_DAMPING_MECHANISMS:
            checks.append(
                _criterion(
                    "virtual_velocity_damping_writes_positive",
                    aggregate.virtual_velocity_damping_writes is not None
                    and aggregate.virtual_velocity_damping_writes > 0,
                    f"value={aggregate.virtual_velocity_damping_writes} source={aggregate.source}",
                )
            )
        if expected_mechanism in TARGET_GUARDED_MECHANISMS:
            checks.append(
                _criterion(
                    "target_guarded_close_advances_positive",
                    aggregate.target_guarded_close_advances is not None
                    and aggregate.target_guarded_close_advances > 0,
                    f"value={aggregate.target_guarded_close_advances} source={aggregate.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V2_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_zero_backlog_holds_reported",
                    aggregate.target_guarded_zero_backlog_holds is not None,
                    f"value={aggregate.target_guarded_zero_backlog_holds} source={aggregate.source}",
                )
            )
        if expected_mechanism in (
            TARGET_GUARDED_V3_MECHANISM,
            TARGET_GUARDED_V4_MECHANISM,
            TARGET_GUARDED_V5_MECHANISM,
            TARGET_GUARDED_V6_MECHANISM,
            TARGET_GUARDED_V7_MECHANISM,
        ):
            checks.append(
                _criterion(
                    "target_guarded_progress_backlog_preserved_holds_positive",
                    aggregate.target_guarded_backlog_preserved_holds is not None
                    and aggregate.target_guarded_backlog_preserved_holds > 0,
                    f"value={aggregate.target_guarded_backlog_preserved_holds} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_progress_zero_backlog_holds_zero",
                    aggregate.target_guarded_zero_backlog_holds == 0,
                    f"value={aggregate.target_guarded_zero_backlog_holds} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_progress_safety_rollbacks_zero",
                    aggregate.target_guarded_safety_rollbacks == 0,
                    f"value={aggregate.target_guarded_safety_rollbacks} source={aggregate.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V4_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_v4_recovery_holds_positive",
                    aggregate.target_guarded_v4_recovery_holds is not None
                    and aggregate.target_guarded_v4_recovery_holds > 0,
                    f"value={aggregate.target_guarded_v4_recovery_holds} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v4_hard_safety_freezes_zero",
                    aggregate.target_guarded_v4_hard_safety_freezes == 0,
                    f"value={aggregate.target_guarded_v4_hard_safety_freezes} source={aggregate.source}",
                )
            )
        if expected_mechanism in (
            TARGET_GUARDED_V5_MECHANISM,
            TARGET_GUARDED_V6_MECHANISM,
            TARGET_GUARDED_V7_MECHANISM,
        ):
            checks.append(
                _criterion(
                    "target_guarded_v5_preemptive_recovery_writes_positive",
                    aggregate.target_guarded_v5_preemptive_recovery_writes is not None
                    and aggregate.target_guarded_v5_preemptive_recovery_writes > 0,
                    f"value={aggregate.target_guarded_v5_preemptive_recovery_writes} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v5_recovery_ik_failures_zero",
                    aggregate.target_guarded_v5_recovery_ik_failures == 0,
                    f"value={aggregate.target_guarded_v5_recovery_ik_failures} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v5_hard_safety_freezes_zero",
                    aggregate.target_guarded_v4_hard_safety_freezes == 0,
                    f"value={aggregate.target_guarded_v4_hard_safety_freezes} source={aggregate.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V7_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_v7_active_recovery_writes_positive",
                    aggregate.target_guarded_v7_active_recovery_writes is not None
                    and aggregate.target_guarded_v7_active_recovery_writes > 0,
                    f"value={aggregate.target_guarded_v7_active_recovery_writes} source={aggregate.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v7_recovery_ik_failures_zero",
                    aggregate.target_guarded_v7_recovery_ik_failures == 0,
                    f"value={aggregate.target_guarded_v7_recovery_ik_failures} source={aggregate.source}",
                )
            )

    step3 = close.get(3)
    if step3 is not None:
        checks.append(
            _criterion(
                "step3_speed_below_push_gate",
                step3.object_speed_mps <= PUSH_SPEED_GATE_MPS,
                f"value={step3.object_speed_mps:.6f} gate={PUSH_SPEED_GATE_MPS:.6f} source={step3.source}",
            )
        )
        if expected_mechanism in VIRTUAL_DAMPING_MECHANISMS:
            checks.append(
                _criterion(
                    "virtual_support_step3",
                    step3.virtual_support is True,
                    f"value={step3.virtual_support} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_damping_active_step3",
                    step3.virtual_damping_active is True,
                    f"value={step3.virtual_damping_active} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "virtual_damping_write_seen_by_step3",
                    step3.virtual_velocity_damping_writes_total is not None
                    and step3.virtual_velocity_damping_writes_total >= 1,
                    f"value={step3.virtual_velocity_damping_writes_total} source={step3.source}",
                )
            )
        if expected_mechanism in TARGET_GUARDED_MECHANISMS:
            checks.append(
                _criterion(
                    "support_horizon_active_step3",
                    step3.support_horizon_active is True,
                    f"value={step3.support_horizon_active} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_advance_seen_by_step3",
                    step3.target_guarded_close_advances_total is not None
                    and step3.target_guarded_close_advances_total >= 1,
                    f"value={step3.target_guarded_close_advances_total} source={step3.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V2_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_command_backlog_step3_within_gate",
                    step3.target_guarded_command_backlog_deg is not None
                    and step3.target_guarded_command_backlog_deg <= TARGET_GUARDED_COMMAND_ERROR_GATE_DEG,
                    f"value={step3.target_guarded_command_backlog_deg} gate={TARGET_GUARDED_COMMAND_ERROR_GATE_DEG:.3f} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_command_converged_step3",
                    step3.target_guarded_command_converged is True,
                    f"value={step3.target_guarded_command_converged} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_support_margin_step3",
                    step3.target_guarded_support_margin_ok is True,
                    f"value={step3.target_guarded_support_margin_ok} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_target_nonworsening_step3",
                    step3.target_guarded_target_nonworsening is True,
                    f"value={step3.target_guarded_target_nonworsening} source={step3.source}",
                )
            )
        if expected_mechanism in (
            TARGET_GUARDED_V3_MECHANISM,
            TARGET_GUARDED_V4_MECHANISM,
            TARGET_GUARDED_V5_MECHANISM,
            TARGET_GUARDED_V6_MECHANISM,
            TARGET_GUARDED_V7_MECHANISM,
        ):
            checks.append(
                _criterion(
                    "target_guarded_progress_support_budget_step3",
                    step3.target_guarded_support_budget_ok is True,
                    f"value={step3.target_guarded_support_budget_ok} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_progress_actual_progress_step3",
                    step3.target_guarded_v3_actual_progress_deg is not None
                    and step3.target_guarded_v3_actual_progress_deg >= TARGET_GUARDED_V3_MIN_ACTUAL_PROGRESS_DEG
                    and step3.target_guarded_v3_actual_progress_ok is True,
                    f"value={step3.target_guarded_v3_actual_progress_deg} ok={step3.target_guarded_v3_actual_progress_ok} gate={TARGET_GUARDED_V3_MIN_ACTUAL_PROGRESS_DEG:.3f} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_progress_backlog_room_step3",
                    step3.target_guarded_v3_backlog_room_ok is True,
                    f"value={step3.target_guarded_v3_backlog_room_ok} source={step3.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V3_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_v3_safety_step3",
                    step3.target_guarded_v3_safety_ok is True,
                    f"value={step3.target_guarded_v3_safety_ok} source={step3.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V4_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_v4_recovery_hold_seen_by_step3",
                    any(
                        obs.target_guarded_v4_recovery_hold is True
                        for step, obs in close.items()
                        if step <= 3
                    ),
                    "requires the target-error recovery path to be exercised by step3 in the synthetic/readiness contract",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v4_hard_safety_step3",
                    step3.target_guarded_v4_hard_safety_ok is True,
                    f"value={step3.target_guarded_v4_hard_safety_ok} source={step3.source}",
                )
            )
        if expected_mechanism == TARGET_GUARDED_V5_MECHANISM:
            checks.append(
                _criterion(
                    "target_guarded_v5_recovery_write_seen_by_step3",
                    step3.target_guarded_v5_preemptive_recovery_writes_total is not None
                    and step3.target_guarded_v5_preemptive_recovery_writes_total >= 1,
                    f"value={step3.target_guarded_v5_preemptive_recovery_writes_total} source={step3.source}",
                )
            )
            checks.append(
                _criterion(
                    "target_guarded_v5_recovery_ik_ok_step3",
                    step3.target_guarded_v5_recovery_ik_ok is True,
                    f"value={step3.target_guarded_v5_recovery_ik_ok} source={step3.source}",
                )
            )

    one_sided_sources = [obs.source for step, obs in close.items() if step in REQUIRED_CLOSE_STEPS and obs.one_sided_push]
    checks.append(
        _criterion("one_sided_push_steps_2_to_4_no", not one_sided_sources, f"sources={one_sided_sources}")
    )

    step4 = close.get(4)
    if step4 is not None:
        checks.append(
            _criterion(
                "counter_support_step4",
                step4.counter_gap_max_m <= COUNTER_SUPPORT_BUDGET_M,
                f"value={step4.counter_gap_max_m:.6f} budget={COUNTER_SUPPORT_BUDGET_M:.6f} source={step4.source}",
            )
        )
        checks.append(
            _criterion(
                "target_step4_within_gate",
                step4.target_error_m <= TARGET_ERROR_GATE_M,
                f"value={step4.target_error_m:.6f} gate={TARGET_ERROR_GATE_M:.6f} source={step4.source}",
            )
        )

    if expected_mechanism in TARGET_GUARDED_MECHANISMS:
        step5 = close.get(5)
        checks.append(_criterion("step5_present_for_support_horizon", step5 is not None, "required for target-guarded horizon"))
        if step5 is not None:
            checks.append(
                _criterion(
                    "support_horizon_step5",
                    step5.counter_gap_max_m <= MAX_PLAUSIBLE_COMPRESSION_M,
                    f"value={step5.counter_gap_max_m:.6f} max={MAX_PLAUSIBLE_COMPRESSION_M:.6f} source={step5.source}",
                )
            )
            checks.append(
                _criterion(
                    "support_horizon_active_step5",
                    step5.support_horizon_active is True,
                    f"value={step5.support_horizon_active} source={step5.source}",
                )
            )

    if expected_mechanism == TARGET_GUARDED_V2_MECHANISM:
        bad_zero_backlog_hold_sources = [
            obs.source
            for obs in close.values()
            if obs.target_guarded_close_hold is True and obs.target_guarded_zero_backlog_hold is not True
        ]
        checks.append(
            _criterion(
                "target_guarded_v2_zero_backlog_on_every_hold",
                not bad_zero_backlog_hold_sources,
                f"sources={bad_zero_backlog_hold_sources}",
            )
        )

    if expected_mechanism in (
        TARGET_GUARDED_V3_MECHANISM,
        TARGET_GUARDED_V4_MECHANISM,
        TARGET_GUARDED_V5_MECHANISM,
        TARGET_GUARDED_V6_MECHANISM,
        TARGET_GUARDED_V7_MECHANISM,
    ):
        zero_backlog_sources = [
            obs.source for obs in close.values() if obs.target_guarded_zero_backlog_hold is True
        ]
        unpreserved_hold_sources = [
            obs.source
            for obs in close.values()
            if obs.target_guarded_close_hold is True
            and obs.target_guarded_v3_safety_rollback is not True
            and obs.target_guarded_backlog_preserved_hold is not True
        ]
        safety_rollback_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v3_safety_rollback is True
        ]
        checks.append(
            _criterion(
                "target_guarded_progress_no_zero_backlog_holds",
                not zero_backlog_sources,
                f"sources={zero_backlog_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_progress_every_nonrollback_hold_preserves_backlog",
                not unpreserved_hold_sources,
                f"sources={unpreserved_hold_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_progress_no_safety_rollbacks",
                not safety_rollback_sources,
                f"sources={safety_rollback_sources}",
            )
        )
    if expected_mechanism in (
        TARGET_GUARDED_V4_MECHANISM,
        TARGET_GUARDED_V5_MECHANISM,
        TARGET_GUARDED_V6_MECHANISM,
        TARGET_GUARDED_V7_MECHANISM,
    ):
        hard_freeze_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v4_hard_safety_freeze is True
        ]
        fixed_target_violations = [
            obs.source for obs in close.values() if obs.target_error_m > TARGET_ERROR_GATE_M
        ]
        fixed_support_violations = [
            obs.source for obs in close.values() if obs.counter_gap_max_m > COUNTER_SUPPORT_BUDGET_M
        ]
        recovery_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v4_recovery_hold is True
        ]
        checks.append(
            _criterion(
                "target_guarded_v4_or_v5_recovery_holds_present",
                bool(recovery_sources),
                f"sources={recovery_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v4_or_v5_no_hard_safety_freezes",
                not hard_freeze_sources,
                f"sources={hard_freeze_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v4_all_close_target_within_fixed_gate",
                not fixed_target_violations,
                f"sources={fixed_target_violations}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v4_all_close_support_within_fixed_budget",
                not fixed_support_violations,
                f"sources={fixed_support_violations}",
            )
        )
    if expected_mechanism in (
        TARGET_GUARDED_V5_MECHANISM,
        TARGET_GUARDED_V6_MECHANISM,
        TARGET_GUARDED_V7_MECHANISM,
    ):
        preempt_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v5_preemptive_recovery is True
        ]
        preempt_needed_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v5_preemptive_recovery_needed is True
        ]
        ik_failure_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v5_recovery_ik_ok is False
        ]
        checks.append(
            _criterion(
                "target_guarded_v5_preemptive_recovery_present",
                bool(preempt_sources),
                f"sources={preempt_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v5_preemptive_trigger_seen",
                bool(preempt_needed_sources),
                f"sources={preempt_needed_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v5_recovery_ik_ok_all",
                not ik_failure_sources,
                f"sources={ik_failure_sources}",
            )
        )
    if expected_mechanism == TARGET_GUARDED_V7_MECHANISM:
        active_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v7_active_recovery is True
        ]
        active_needed_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v7_active_recovery_needed is True
        ]
        active_ik_failure_sources = [
            obs.source for obs in close.values() if obs.target_guarded_v7_recovery_ik_ok is False
        ]
        non_gap_reducing_sources = [
            obs.source
            for obs in close.values()
            if obs.target_guarded_v7_active_recovery is True
            and (
                obs.target_guarded_v7_counter_gap_delta_m is None
                or obs.target_guarded_v7_counter_gap_delta_m >= 0.0
            )
        ]
        invalid_margin_sources = [
            obs.source
            for obs in close.values()
            if obs.target_guarded_v7_active_recovery is True
            and (
                obs.target_guarded_v7_best_target_margin_m is None
                or obs.target_guarded_v7_best_support_margin_m is None
                or obs.target_guarded_v7_best_target_margin_m < 0.0
                or obs.target_guarded_v7_best_support_margin_m < 0.0
            )
        ]
        checks.append(
            _criterion(
                "target_guarded_v7_active_recovery_present",
                bool(active_sources),
                f"sources={active_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v7_active_recovery_trigger_seen",
                bool(active_needed_sources),
                f"sources={active_needed_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v7_recovery_ik_ok_all",
                not active_ik_failure_sources,
                f"sources={active_ik_failure_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v7_active_recovery_reduces_counter_gap",
                not non_gap_reducing_sources,
                f"sources={non_gap_reducing_sources}",
            )
        )
        checks.append(
            _criterion(
                "target_guarded_v7_active_recovery_selected_margins_valid",
                not invalid_margin_sources,
                f"sources={invalid_margin_sources}",
            )
        )

    finite_ok = all(
        math.isfinite(obs.target_error_m) and math.isfinite(obs.object_speed_mps) and math.isfinite(obs.counter_gap_max_m)
        for obs in close.values()
    )
    checks.append(_criterion("finite_close_metrics", finite_ok, "all parsed close metrics must be finite"))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--log", type=Path)
    source_group.add_argument("--use_v7_reference", action="store_true")
    source_group.add_argument("--use_synthetic_pass_reference", action="store_true")
    source_group.add_argument("--use_synthetic_virtual_no_damping_reference", action="store_true")
    source_group.add_argument("--use_synthetic_v3_zero_backlog_reference", action="store_true")
    source_group.add_argument("--use_synthetic_v4_hard_freeze_reference", action="store_true")
    source_group.add_argument("--use_synthetic_v7_no_active_recovery_reference", action="store_true")
    ap.add_argument(
        "--expected_mechanism",
        choices=[
            "soft_contact_material_diagnostic",
            "virtual_compression_damping_diagnostic",
            TARGET_GUARDED_MECHANISM,
            TARGET_GUARDED_V2_MECHANISM,
            TARGET_GUARDED_V3_MECHANISM,
            TARGET_GUARDED_V4_MECHANISM,
            TARGET_GUARDED_V5_MECHANISM,
            TARGET_GUARDED_V6_MECHANISM,
            TARGET_GUARDED_V7_MECHANISM,
        ],
        default="soft_contact_material_diagnostic",
    )
    args = ap.parse_args()

    if args.use_v7_reference:
        close, aggregate, metadata = _reference_v7()
        source = "encoded_v7_reference"
    elif args.use_synthetic_pass_reference:
        close, aggregate, metadata = _reference_synthetic_pass(args.expected_mechanism)
        source = "synthetic_pass_reference"
    elif args.use_synthetic_virtual_no_damping_reference:
        close, aggregate, metadata = _reference_synthetic_virtual_no_damping(args.expected_mechanism)
        source = "synthetic_virtual_no_damping_reference"
    elif args.use_synthetic_v3_zero_backlog_reference:
        close, aggregate, metadata = _reference_synthetic_v3_zero_backlog()
        source = "synthetic_v3_zero_backlog_reference"
    elif args.use_synthetic_v4_hard_freeze_reference:
        close, aggregate, metadata = _reference_synthetic_v4_hard_freeze()
        source = "synthetic_v4_hard_freeze_reference"
    elif args.use_synthetic_v7_no_active_recovery_reference:
        close, aggregate, metadata = _reference_synthetic_v7_no_active_recovery()
        source = "synthetic_v7_no_active_recovery_reference"
    else:
        if not args.log.exists():
            raise FileNotFoundError(args.log)
        close, aggregate, metadata = _parse_log(args.log)
        source = str(args.log)

    print("[cube2cm_soft_contact_runtime_criteria_audit] local_posthoc_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_soft_contact_runtime_criteria_audit] "
        f"source={source} expected_mechanism={args.expected_mechanism} "
        f"push_speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f} "
        f"target_error_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"counter_support_budget_m={COUNTER_SUPPORT_BUDGET_M:.6f}",
        flush=True,
    )

    checks = _audit(close, aggregate, metadata, args.expected_mechanism)
    all_ok = all(ok for _name, ok, _detail in checks)
    for name, ok, detail in checks:
        print(
            f"[cube2cm_soft_contact_runtime_criteria_audit] criterion name={name} pass={_yes(ok)} {detail}",
            flush=True,
        )
    print(f"[cube2cm_soft_contact_runtime_criteria_audit] SOFT_CONTACT_RUNTIME_CRITERIA_PASS={_yes(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
