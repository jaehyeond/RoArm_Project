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
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "sim_scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "sim_scripts"))

from p7_branch_b_cube2cm_compliance_proxy_static_analysis import V7_CLOSE_SAMPLES  # noqa: E402


PUSH_SPEED_GATE_MPS = 0.005
TARGET_ERROR_GATE_M = 0.003
COUNTER_SUPPORT_BUDGET_M = 0.002
REQUIRED_CLOSE_STEPS = (2, 3, 4)


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
    virtual_damping_active: bool | None
    virtual_velocity_damping_writes_total: int | None
    reached: bool
    early_kill: bool


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
    telemetry_only: bool
    success_claim: bool


@dataclass(frozen=True)
class RuntimeMetadata:
    source: str
    soft_contact_material_diagnostic: bool | None
    virtual_compression_damping_diagnostic: bool | None
    object_physics_mode: str | None
    runtime_candidate_requires_separate_approval: bool | None


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
                virtual_damping_active=(
                    _parse_bool(fields["virtual_damping_active"]) if "virtual_damping_active" in fields else None
                ),
                virtual_velocity_damping_writes_total=(
                    int(fields["virtual_velocity_damping_writes_total"])
                    if "virtual_velocity_damping_writes_total" in fields
                    else None
                ),
                reached=_parse_bool(_line_value(line, "reached")),
                early_kill=_parse_bool(_line_value(line, "early_kill")),
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
                telemetry_only=_parse_bool(fields["telemetry_only"]),
                success_claim=_parse_bool(fields["success_claim"]),
            )
    metadata = RuntimeMetadata(
        source=metadata_source,
        soft_contact_material_diagnostic=soft_contact_material_diagnostic,
        virtual_compression_damping_diagnostic=virtual_compression_damping_diagnostic,
        object_physics_mode=object_physics_mode,
        runtime_candidate_requires_separate_approval=runtime_candidate_requires_separate_approval,
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
            virtual_damping_active=None,
            virtual_velocity_damping_writes_total=None,
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
        telemetry_only=True,
        success_claim=False,
    )
    metadata = RuntimeMetadata(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38",
        soft_contact_material_diagnostic=False,
        virtual_compression_damping_diagnostic=False,
        object_physics_mode="baseline",
        runtime_candidate_requires_separate_approval=False,
    )
    return close, aggregate, metadata


def _reference_synthetic_pass(expected_mechanism: str) -> tuple[dict[int, CloseObservation], AggregateObservation, RuntimeMetadata]:
    is_virtual = expected_mechanism == "virtual_compression_damping_diagnostic"
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
            virtual_damping_active=False if is_virtual else None,
            virtual_velocity_damping_writes_total=0 if is_virtual else None,
            reached=False,
            early_kill=False,
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
            virtual_damping_active=True if is_virtual else None,
            virtual_velocity_damping_writes_total=1 if is_virtual else None,
            reached=False,
            early_kill=False,
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
            virtual_damping_active=True if is_virtual else None,
            virtual_velocity_damping_writes_total=2 if is_virtual else None,
            reached=False,
            early_kill=False,
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
        virtual_velocity_damping_writes=2 if is_virtual else None,
        telemetry_only=True,
        success_claim=False,
    )
    metadata = RuntimeMetadata(
        source="synthetic_pass:metadata",
        soft_contact_material_diagnostic=expected_mechanism == "soft_contact_material_diagnostic",
        virtual_compression_damping_diagnostic=expected_mechanism == "virtual_compression_damping_diagnostic",
        object_physics_mode=expected_mechanism,
        runtime_candidate_requires_separate_approval=True,
    )
    return close, aggregate, metadata


def _reference_synthetic_virtual_no_damping() -> tuple[
    dict[int, CloseObservation], AggregateObservation, RuntimeMetadata
]:
    close, aggregate, metadata = _reference_synthetic_pass("virtual_compression_damping_diagnostic")
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
            virtual_damping_active=False,
            virtual_velocity_damping_writes_total=0,
            reached=obs.reached,
            early_kill=obs.early_kill,
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
        telemetry_only=aggregate.telemetry_only,
        success_claim=aggregate.success_claim,
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
        else:
            raise ValueError(f"unsupported expected mechanism {expected_mechanism!r}")
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
        if expected_mechanism == "virtual_compression_damping_diagnostic":
            checks.append(
                _criterion(
                    "virtual_velocity_damping_writes_positive",
                    aggregate.virtual_velocity_damping_writes is not None
                    and aggregate.virtual_velocity_damping_writes > 0,
                    f"value={aggregate.virtual_velocity_damping_writes} source={aggregate.source}",
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
        if expected_mechanism == "virtual_compression_damping_diagnostic":
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
    ap.add_argument(
        "--expected_mechanism",
        choices=["soft_contact_material_diagnostic", "virtual_compression_damping_diagnostic"],
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
        close, aggregate, metadata = _reference_synthetic_virtual_no_damping()
        source = "synthetic_virtual_no_damping_reference"
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
