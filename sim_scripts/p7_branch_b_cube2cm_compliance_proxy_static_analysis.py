#!/usr/bin/env python3
"""Static compliance-proxy audit for P7 Branch B 3cm cube close telemetry.

This script is intentionally analysis-only. It does not launch Isaac, train,
generate datasets, edit defaults, insert constraints, attach SurfaceGripper,
transport, release, tune gates, or claim grasp success.

It consumes the already-verified v7 B200 close_26 telemetry facts as constants
and asks a narrower question: what bounded virtual-compression/contact envelope
would have been required to keep counter-side support during close step 2-4,
and would contact relabeling alone explain the observed one-sided-push dynamics?
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


CONTACT_EPS = 1.0e-12
OBJECT_SIZE_M = np.array([0.030, 0.030, 0.030], dtype=np.float64)


@dataclass(frozen=True)
class CloseSample:
    source: str
    step: int
    target_error_m: float
    object_drift_m: float
    object_speed_mps: float
    moving_gap_m: np.ndarray
    counter_gap_m: np.ndarray
    moving_overlap_m: np.ndarray
    counter_overlap_m: np.ndarray
    logged_moving_contact: bool
    logged_counter_contact: bool
    logged_counter_slop_contact_1mm: bool
    logged_one_sided_push: bool
    reached: bool


V7_CLOSE_SAMPLES = (
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:376",
        step=2,
        target_error_m=0.001308,
        object_drift_m=0.000000,
        object_speed_mps=0.001596,
        moving_gap_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        counter_gap_m=np.array([0.0, 0.000001, 0.0], dtype=np.float64),
        moving_overlap_m=np.array([0.000080, 0.007826, 0.002303], dtype=np.float64),
        counter_overlap_m=np.array([0.008538, 0.0, 0.008941], dtype=np.float64),
        logged_moving_contact=True,
        logged_counter_contact=False,
        logged_counter_slop_contact_1mm=True,
        logged_one_sided_push=False,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377",
        step=3,
        target_error_m=0.001442,
        object_drift_m=0.000062,
        object_speed_mps=0.061935,
        moving_gap_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        counter_gap_m=np.array([0.0, 0.000652, 0.0], dtype=np.float64),
        moving_overlap_m=np.array([0.000432, 0.007771, 0.003300], dtype=np.float64),
        counter_overlap_m=np.array([0.008550, 0.0, 0.008952], dtype=np.float64),
        logged_moving_contact=True,
        logged_counter_contact=False,
        logged_counter_slop_contact_1mm=True,
        logged_one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:378",
        step=4,
        target_error_m=0.003151,
        object_drift_m=0.000042,
        object_speed_mps=0.043783,
        moving_gap_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        counter_gap_m=np.array([0.0, 0.001813, 0.0], dtype=np.float64),
        moving_overlap_m=np.array([0.000342, 0.007715, 0.003574], dtype=np.float64),
        counter_overlap_m=np.array([0.008590, 0.0, 0.008945], dtype=np.float64),
        logged_moving_contact=True,
        logged_counter_contact=False,
        logged_counter_slop_contact_1mm=False,
        logged_one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:379",
        step=5,
        target_error_m=0.005002,
        object_drift_m=0.000060,
        object_speed_mps=0.054294,
        moving_gap_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        counter_gap_m=np.array([0.0, 0.002911, 0.0], dtype=np.float64),
        moving_overlap_m=np.array([0.000311, 0.007660, 0.003886], dtype=np.float64),
        counter_overlap_m=np.array([0.008622, 0.0, 0.008946], dtype=np.float64),
        logged_moving_contact=True,
        logged_counter_contact=False,
        logged_counter_slop_contact_1mm=False,
        logged_one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419",
        step=45,
        target_error_m=0.023422,
        object_drift_m=0.000859,
        object_speed_mps=0.051807,
        moving_gap_m=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        counter_gap_m=np.array([0.0, 0.014319, 0.0], dtype=np.float64),
        moving_overlap_m=np.array([0.000423, 0.007074, 0.004244], dtype=np.float64),
        counter_overlap_m=np.array([0.001917, 0.0, 0.005740], dtype=np.float64),
        logged_moving_contact=True,
        logged_counter_contact=False,
        logged_counter_slop_contact_1mm=False,
        logged_one_sided_push=True,
        reached=False,
    ),
)


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _fmt_m(value: float) -> str:
    return f"{value:.6f}"


def _parse_budget_list(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one compliance budget is required")
    if any(value < 0.0 for value in values):
        raise ValueError("compliance budgets must be non-negative")
    return values


def _support_under_budget(sample: CloseSample, budget_m: float) -> tuple[bool, bool]:
    moving_support = bool(np.all(sample.moving_gap_m <= budget_m + CONTACT_EPS))
    counter_support = bool(np.all(sample.counter_gap_m <= budget_m + CONTACT_EPS))
    return moving_support, counter_support


def _required_budget(sample: CloseSample) -> float:
    return float(max(np.max(sample.moving_gap_m), np.max(sample.counter_gap_m)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets_m", default="0.000,0.001,0.002,0.003,0.005,0.010,0.015")
    ap.add_argument("--plausible_budget_max_m", type=float, default=0.005)
    ap.add_argument("--push_drift_gate_m", type=float, default=0.00020)
    ap.add_argument("--push_speed_gate_mps", type=float, default=0.005)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    args = ap.parse_args()

    budgets = _parse_budget_list(args.budgets_m)
    if args.plausible_budget_max_m < 0.0:
        raise ValueError("plausible_budget_max_m must be non-negative")

    print("[cube2cm_compliance_proxy_static] local_static_only=YES isaac_run=NO training=NO dataset_generation=NO")
    print(
        "[cube2cm_compliance_proxy_static] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO transport_target=NO "
        "release_marker=NO scalar_or_gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_compliance_proxy_static] source_runtime=/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out "
        f"object_size_m=({OBJECT_SIZE_M[0]:.3f},{OBJECT_SIZE_M[1]:.3f},{OBJECT_SIZE_M[2]:.3f}) "
        f"push_drift_gate_m={args.push_drift_gate_m:.6f} push_speed_gate_mps={args.push_speed_gate_mps:.6f} "
        f"target_error_gate_m={args.target_error_gate_m:.6f} plausible_budget_max_m={args.plausible_budget_max_m:.6f}",
        flush=True,
    )

    close_2_to_4 = [sample for sample in V7_CLOSE_SAMPLES if sample.step in (2, 3, 4)]
    required_2_to_4 = max(_required_budget(sample) for sample in close_2_to_4)
    required_2_to_5 = max(_required_budget(sample) for sample in V7_CLOSE_SAMPLES if sample.step in (2, 3, 4, 5))
    required_final = max(_required_budget(sample) for sample in V7_CLOSE_SAMPLES)

    for sample in V7_CLOSE_SAMPLES:
        push_started = sample.object_drift_m > args.push_drift_gate_m or sample.object_speed_mps > args.push_speed_gate_mps
        dynamic_ok = not push_started
        required_budget = _required_budget(sample)
        print(
            f"[cube2cm_compliance_proxy_static] sample step={sample.step:03d} source={sample.source} "
            f"target_error_m={sample.target_error_m:.6f} object_drift_m={sample.object_drift_m:.6f} "
            f"object_speed_mps={sample.object_speed_mps:.6f} required_counter_budget_m={required_budget:.6f} "
            f"within_plausible_budget={_yes(required_budget <= args.plausible_budget_max_m + CONTACT_EPS)} "
            f"push_started_by_existing_gate={_yes(push_started)} dynamic_ok_without_impulse_absorption={_yes(dynamic_ok)} "
            f"logged_moving_contact={_yes(sample.logged_moving_contact)} "
            f"logged_counter_contact={_yes(sample.logged_counter_contact)} "
            f"logged_counter_slop_contact_1mm={_yes(sample.logged_counter_slop_contact_1mm)} "
            f"logged_one_sided_push={_yes(sample.logged_one_sided_push)} reached={_yes(sample.reached)}",
            flush=True,
        )

    for budget in budgets:
        supported_steps = []
        unsupported_steps = []
        dynamic_fail_steps = []
        relabel_only_false_positive_steps = []
        for sample in V7_CLOSE_SAMPLES:
            moving_support, counter_support = _support_under_budget(sample, budget)
            support_both = moving_support and counter_support
            push_started = sample.object_drift_m > args.push_drift_gate_m or sample.object_speed_mps > args.push_speed_gate_mps
            if support_both:
                supported_steps.append(sample.step)
            else:
                unsupported_steps.append(sample.step)
            if push_started:
                dynamic_fail_steps.append(sample.step)
            if support_both and push_started:
                relabel_only_false_positive_steps.append(sample.step)

        print(
            f"[cube2cm_compliance_proxy_static] budget_result budget_m={budget:.6f} "
            f"budget_mm={budget * 1000.0:.2f} within_plausible_budget={_yes(budget <= args.plausible_budget_max_m + CONTACT_EPS)} "
            f"support_both_steps={supported_steps} unsupported_steps={unsupported_steps} "
            f"dynamic_fail_steps={dynamic_fail_steps} "
            f"contact_relabel_only_would_overclaim_steps={relabel_only_false_positive_steps}",
            flush=True,
        )

    close_2_to_4_dynamic_ok = all(
        sample.object_drift_m <= args.push_drift_gate_m and sample.object_speed_mps <= args.push_speed_gate_mps
        for sample in close_2_to_4
    )
    min_budget_2_to_4_ok = required_2_to_4 <= args.plausible_budget_max_m + CONTACT_EPS
    final_budget_plausible = required_final <= args.plausible_budget_max_m + CONTACT_EPS
    contact_label_only_sufficient = min_budget_2_to_4_ok and close_2_to_4_dynamic_ok

    print(
        f"[cube2cm_compliance_proxy_static] summary required_budget_close_steps_2_to_4_m={_fmt_m(required_2_to_4)} "
        f"required_budget_close_steps_2_to_5_m={_fmt_m(required_2_to_5)} "
        f"required_budget_final_step_45_m={_fmt_m(required_final)} "
        f"close_2_to_4_budget_plausible={_yes(min_budget_2_to_4_ok)} "
        f"final_budget_plausible={_yes(final_budget_plausible)} "
        f"close_2_to_4_dynamic_ok_without_impulse_absorption={_yes(close_2_to_4_dynamic_ok)} "
        f"contact_label_only_sufficient={_yes(contact_label_only_sufficient)}",
        flush=True,
    )
    print(
        "[cube2cm_compliance_proxy_static] next_design_prediction="
        "future_close_26_runtime_must_reduce_step3_speed_and_keep_counter_support_through_step4; "
        "increasing_slop_labels_alone_is_not_a_physics_solution",
        flush=True,
    )
    print("[cube2cm_compliance_proxy_static] CUBE2CM_COMPLIANCE_PROXY_STATIC_ANALYSIS_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
