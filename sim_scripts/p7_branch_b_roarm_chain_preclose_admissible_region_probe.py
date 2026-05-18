#!/usr/bin/env python3
"""Diagnostic-only admissible-region check for pre-close selector logs.

This wrapper does not integrate constraints, attach SurfaceGripper, execute
transport, execute release, train, tune gates, or edit env/train/chain defaults.
It reads existing selector logs, or optionally runs the unchanged selector into
log files, then applies a conservative diagnostic-only admissible-region rule.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SELECTOR = REPO / "sim_scripts" / "p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    role: str
    log_name: str
    strategy_name: str
    segment_prefix: str
    expected: str
    side_margin_m: float | None = None
    side_top_margin_m: float | None = None


@dataclass(frozen=True)
class ParsedCase:
    spec: CaseSpec
    selection: dict[str, str]
    segment: dict[str, str]
    aggregate: dict[str, str]
    flags: dict[str, str]
    log_path: Path


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _bool(text: str | None) -> bool:
    return str(text).strip() == "YES"


def _float(text: str | None, default: float = float("nan")) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


def _parse_pairs(line: str) -> dict[str, str]:
    matches = list(re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=", line))
    pairs: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        pairs[key] = line[start:end].strip().rstrip(",")
    return pairs


def _quat_tilt_deg(quat_text: str | None) -> float:
    if not quat_text:
        return float("nan")
    values = {key: _float(value) for key, value in re.findall(r"([wxyz])=([+-]?[0-9.]+)", quat_text)}
    w = values.get("w", 1.0)
    x = values.get("x", 0.0)
    y = values.get("y", 0.0)
    z = values.get("z", 0.0)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def _inside_text(value: bool) -> str:
    return "inside" if value else "outside"


def _reason_join(reasons: list[str]) -> str:
    return ",".join(reasons) if reasons else "passes_conservative_diagnostic_filters"


def _default_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="invalid_boundary_zero_margin",
            role="side_edge_boundary",
            log_name="p7_branch_b_roarm_chain_preclose_side_margin_robustness_0p0_b200.out",
            strategy_name="candidate_side_edge_margin_0mm_top_margin_neg1p5mm",
            segment_prefix="candidate_side_edge_margin_0mm_top_margin_neg1p5mm",
            side_margin_m=0.0,
            side_top_margin_m=-0.0015,
            expected="REJECT",
        ),
        CaseSpec(
            name="minimum_positive_observed_pass_not_conservative",
            role="side_edge_boundary",
            log_name="p7_branch_b_roarm_chain_preclose_side_margin_boundary_fine_0p1_b200.out",
            strategy_name="candidate_side_edge_margin_0p1mm_top_margin_neg1p5mm",
            segment_prefix="candidate_side_edge_margin_0p1mm_top_margin_neg1p5mm",
            side_margin_m=0.0001,
            side_top_margin_m=-0.0015,
            expected="REJECT",
        ),
        CaseSpec(
            name="conservative_side_edge_depth_pass",
            role="side_edge_conservative",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out",
            strategy_name="candidate_side_edge_margin_2mm_top_margin_neg3mm",
            segment_prefix="candidate_side_edge_margin_2mm_top_margin_neg3mm",
            side_margin_m=0.0020,
            side_top_margin_m=-0.0030,
            expected="ACCEPT",
        ),
        CaseSpec(
            name="side_edge_depth_exact_fail",
            role="side_edge_depth_fail",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg4p0_b200.out",
            strategy_name="candidate_side_edge_margin_2mm_top_margin_neg4mm",
            segment_prefix="candidate_side_edge_margin_2mm_top_margin_neg4mm",
            side_margin_m=0.0020,
            side_top_margin_m=-0.0040,
            expected="REJECT",
        ),
        CaseSpec(
            name="top_tangent_control",
            role="top_or_above_control",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out",
            strategy_name="candidate_top_tangent_margin_0p5mm",
            segment_prefix="candidate_top_tangent_margin_0p5mm_down_to_top_margin_0p5mm",
            expected="ACCEPT",
        ),
        CaseSpec(
            name="above_top_control",
            role="top_or_above_control",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out",
            strategy_name="candidate_above_top_margin_1mm",
            segment_prefix="candidate_above_top_margin_1mm_down_to_above_margin_1mm",
            expected="ACCEPT",
        ),
        CaseSpec(
            name="nominal_below_top_inside_baseline",
            role="inside_footprint_invalid_control",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out",
            strategy_name="baseline_nominal_below_top_plus5deg",
            segment_prefix="baseline_nominal_below_top_plus5deg_below_top_plus5deg",
            expected="REJECT",
        ),
        CaseSpec(
            name="far_sponge_no_contact_control",
            role="far_no_contact_control",
            log_name="p7_branch_b_roarm_chain_preclose_side_top_depth_sweep_neg3p0_b200.out",
            strategy_name="far_sponge_below_top_plus5deg_control",
            segment_prefix="far_sponge_below_top_plus5deg_control_below_top_plus5deg_far_sponge",
            expected="REJECT",
        ),
    ]


def _selector_runs() -> list[tuple[str, list[str]]]:
    return [
        ("boundary_0p0_neg1p5", ["--side_margin_m", "0.0", "--side_top_margin_m", "-0.0015"]),
        ("minimum_0p1_neg1p5", ["--side_margin_m", "0.0001", "--side_top_margin_m", "-0.0015"]),
        ("conservative_2p0_neg3p0", ["--side_margin_m", "0.0020", "--side_top_margin_m", "-0.0030"]),
        ("depthfail_2p0_neg4p0", ["--side_margin_m", "0.0020", "--side_top_margin_m", "-0.0040"]),
    ]


def _run_selector_variants(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[roarm_chain_preclose_admissible] optional_selector_execution=YES unchanged_selector=YES", flush=True)
    for tag, extra_args in _selector_runs():
        out_path = output_dir / f"p7_branch_b_roarm_chain_preclose_admissible_region_{tag}_b200.out"
        err_path = output_dir / f"p7_branch_b_roarm_chain_preclose_admissible_region_{tag}_b200.err"
        cmd = [sys.executable, str(SELECTOR), *extra_args]
        print(
            f"[roarm_chain_preclose_admissible] running_selector tag={tag} cmd={' '.join(cmd)} "
            f"stdout={out_path} stderr={err_path}",
            flush=True,
        )
        with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(cmd, cwd=REPO, stdout=stdout, stderr=stderr, check=False)
        print(
            f"[roarm_chain_preclose_admissible] selector_exit tag={tag} exit_code={completed.returncode} "
            "note=nonzero_is_expected_for_intentional_negative_cases",
            flush=True,
        )


def _parse_case(spec: CaseSpec, log_dir: Path) -> ParsedCase:
    path = log_dir / spec.log_name
    if not path.exists():
        raise FileNotFoundError(f"missing required selector log for {spec.name}: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    selection: dict[str, str] | None = None
    segment: dict[str, str] | None = None
    aggregate: dict[str, str] = {}
    flags: dict[str, str] = {}
    for line in text.splitlines():
        if " candidate_selection " in line:
            pairs = _parse_pairs(line)
            if pairs.get("name") == spec.strategy_name:
                selection = pairs
        elif " segment_result " in line:
            pairs = _parse_pairs(line)
            if pairs.get("label", "").startswith(spec.segment_prefix):
                segment = pairs
        elif " aggregate " in line:
            aggregate = _parse_pairs(line)
        elif " hypothesis_flags " in line:
            flags = _parse_pairs(line)
    if selection is None:
        raise RuntimeError(f"missing candidate_selection for {spec.name} in {path}")
    if segment is None:
        raise RuntimeError(f"missing segment_result for {spec.name} in {path}")
    return ParsedCase(spec=spec, selection=selection, segment=segment, aggregate=aggregate, flags=flags, log_path=path)


def _admissible_decision(parsed: ParsedCase, min_side_margin_m: float, max_below_depth_m: float) -> tuple[str, str]:
    spec = parsed.spec
    selection = parsed.selection
    segment = parsed.segment
    planned_inside = _bool(selection.get("final_target_xy_inside_sponge_aabb"))
    final_inside = _bool(segment.get("final_target_xy_inside_sponge_aabb"))
    top_class = segment.get("final_target_top_class", "")
    exact = _bool(segment.get("exact_converged"))
    top_clamped = _bool(segment.get("top_clamped"))
    mechanically_valid = _bool(segment.get("mechanically_valid_target"))
    clean = _bool(segment.get("clean_realized_without_reduction_artifact"))
    reasons: list[str] = []

    if spec.role == "far_no_contact_control":
        reasons.append("far_sponge_below_top_is_no_contact_control")
    if top_class == "below" and final_inside:
        reasons.append("below_top_inside_footprint_final_target")
    if top_class == "below" and planned_inside:
        reasons.append("below_top_inside_footprint_planned_target")
    if spec.side_margin_m is not None:
        if spec.side_margin_m <= 0.0:
            reasons.append("zero_margin_side_edge_boundary")
        if spec.side_margin_m < min_side_margin_m:
            reasons.append(f"side_margin_below_conservative_min_{min_side_margin_m:.4f}m")
    if spec.side_top_margin_m is not None and spec.side_top_margin_m < max_below_depth_m:
        reasons.append(f"side_top_depth_deeper_than_{max_below_depth_m:.4f}m")
    if not exact:
        reasons.append("exact_converged_NO_under_unchanged_0p003m_gate")
    if top_clamped:
        reasons.append("top_clamped")
    if not mechanically_valid:
        reasons.append("mechanically_valid_target_NO")
    if spec.role != "far_no_contact_control" and not clean:
        reasons.append("clean_realized_without_reduction_artifact_NO")

    return ("REJECT", _reason_join(reasons)) if reasons else ("ACCEPT", _reason_join(reasons))


def _print_case(parsed: ParsedCase, decision: str, reason: str) -> bool:
    spec = parsed.spec
    selection = parsed.selection
    segment = parsed.segment
    aggregate = parsed.aggregate
    flags = parsed.flags
    planned_inside = _bool(selection.get("final_target_xy_inside_sponge_aabb"))
    final_inside = _bool(segment.get("final_target_xy_inside_sponge_aabb"))
    final_tilt = _quat_tilt_deg(segment.get("final_sponge_quat_wxyz"))
    expected_ok = decision == spec.expected
    print(
        f"[roarm_chain_preclose_admissible] case name={spec.name} role={spec.role} "
        f"expected={spec.expected} admissible_decision={decision} expected_match={_yes(expected_ok)} "
        f"reason={reason} selector_decision={selection.get('decision')} selector_reason={selection.get('reason')} "
        f"target_top_class_planned={selection.get('final_target_top_class')} "
        f"target_top_class_final={segment.get('final_target_top_class')} "
        f"planned_target_xy_aabb={_inside_text(planned_inside)} final_target_xy_aabb={_inside_text(final_inside)} "
        f"final_target_tcp_error_m={segment.get('final_target_tcp_error_m')} "
        f"exact_converged={segment.get('exact_converged')} reduction_gate_would_pass={segment.get('reduction_gate_would_pass')} "
        f"top_clamped={segment.get('top_clamped')} mechanically_valid_target={segment.get('mechanically_valid_target')} "
        f"clean_realized_without_reduction_artifact={segment.get('clean_realized_without_reduction_artifact')} "
        f"max_sponge_drift_m={segment.get('max_sponge_drift_m')} max_sponge_speed_mps={segment.get('max_sponge_speed_mps')} "
        f"final_sponge_tilt_deg={final_tilt:.6f} below_inside_segments_clean={aggregate.get('below_inside_segments_clean', 'UNKNOWN')} "
        f"attach_calls={aggregate.get('attach_calls', 'UNKNOWN')} far_control_is_no_contact_control={flags.get('far_control_is_no_contact_control', 'UNKNOWN')} "
        f"attach_physics_validated=NO release_physics_validated=NO transport_executed=NO log={parsed.log_path}",
        flush=True,
    )
    return expected_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    ap.add_argument("--min-side-margin-m", type=float, default=0.0020)
    ap.add_argument("--max-below-depth-m", type=float, default=-0.0030)
    ap.add_argument("--run-selector", action="store_true")
    ap.add_argument("--selector-output-dir", type=Path, default=Path("/tmp"))
    args = ap.parse_args()

    print("[roarm_chain_preclose_admissible] preclose_admissible_region_probe", flush=True)
    print(
        "[roarm_chain_preclose_admissible] "
        "diagnostic_preclose_only=YES selector_only_as_diagnostic_gate=YES "
        "constraint_prim_insertion=NO fixed_dynamic_constraint_integration=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "scripted_release_variant=NO p7_training=NO p7_tuning=NO diagnostic_gate_tuning=NO "
        "env_default_edits=NO chain_defaults_edits=NO attach_physics_validated=NO "
        "release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_admissible] rule min_side_margin_m={args.min_side_margin_m:.6f} "
        f"max_below_depth_m={args.max_below_depth_m:.6f} unchanged_exact_gate_reference_m=0.003000 "
        "reject_below_top_inside_footprint=YES reject_zero_margin_boundary=YES "
        "require_realized_final_outside_aabb_for_below_top_side_edge=YES "
        "far_sponge_below_top_is_no_contact_control=YES deploy_or_chain_rule=NO",
        flush=True,
    )

    if args.run_selector:
        _run_selector_variants(args.selector_output_dir)

    try:
        parsed_cases = [_parse_case(spec, args.log_dir) for spec in _default_cases()]
    except FileNotFoundError as exc:
        print(f"[roarm_chain_preclose_admissible] input_error {exc}", flush=True)
        return 1
    matches: list[bool] = []
    for parsed in parsed_cases:
        decision, reason = _admissible_decision(parsed, args.min_side_margin_m, args.max_below_depth_m)
        matches.append(_print_case(parsed, decision, reason))

    attach_all_zero = all(parsed.aggregate.get("attach_calls") == "0" for parsed in parsed_cases)
    below_inside_clean_all_empty = all(parsed.aggregate.get("below_inside_segments_clean") == "[]" for parsed in parsed_cases)
    success = all(matches) and attach_all_zero and below_inside_clean_all_empty
    print(
        f"[roarm_chain_preclose_admissible] aggregate cases_tested={len(parsed_cases)} "
        f"expected_matches={sum(1 for item in matches if item)}/{len(matches)} "
        f"attach_calls_all_zero={_yes(attach_all_zero)} "
        f"below_inside_segments_clean_all_empty={_yes(below_inside_clean_all_empty)} "
        "attach_physics_validated=NO release_physics_validated=NO transport_executed=NO "
        f"ROARM_PRECLOSE_ADMISSIBLE_REGION_DIAGNOSTIC_SUCCESS={_yes(success)}",
        flush=True,
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
