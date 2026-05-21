#!/usr/bin/env python3
"""Static-only failure analysis for the approved virtual damping runtime.

This script encodes the verified B200 close_26 virtual compression+damping
runtime lines and computes the remaining blockers. It does not launch Isaac,
change runtime behavior, tune gates, or claim success.
"""
from __future__ import annotations

from dataclasses import dataclass


PUSH_SPEED_GATE_MPS = 0.005
TARGET_ERROR_GATE_M = 0.003
SUPPORT_BUDGET_M = 0.002
MAX_PLAUSIBLE_COMPRESSION_M = 0.003
VELOCITY_DAMPING_RESIDUAL_RATIO = 0.08


@dataclass(frozen=True)
class VirtualRuntimeStep:
    step: int
    source: str
    target_error_m: float
    speed_mps: float
    pre_damping_speed_mps: float
    counter_gap_y_m: float
    virtual_support: bool
    virtual_damping_active: bool
    one_sided_push: bool


STEPS = (
    VirtualRuntimeStep(
        step=3,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:378",
        target_error_m=0.001442,
        speed_mps=0.004955,
        pre_damping_speed_mps=0.061935,
        counter_gap_y_m=0.000652,
        virtual_support=True,
        virtual_damping_active=True,
        one_sided_push=False,
    ),
    VirtualRuntimeStep(
        step=4,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:379",
        target_error_m=0.003130,
        speed_mps=0.003203,
        pre_damping_speed_mps=0.040032,
        counter_gap_y_m=0.001794,
        virtual_support=True,
        virtual_damping_active=True,
        one_sided_push=False,
    ),
    VirtualRuntimeStep(
        step=5,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:380",
        target_error_m=0.004843,
        speed_mps=0.050912,
        pre_damping_speed_mps=0.050912,
        counter_gap_y_m=0.002738,
        virtual_support=False,
        virtual_damping_active=False,
        one_sided_push=True,
    ),
)

FINAL_SOURCE = "/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:420"
FINAL_TARGET_ERROR_M = 0.022778
FINAL_COUNTER_GAP_Y_M = 0.013828


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _safe_suppression(pre_speed: float, post_speed: float) -> float:
    if pre_speed <= 0.0:
        return 0.0
    return 1.0 - post_speed / pre_speed


def main() -> int:
    print("[cube2cm_virtual_runtime_failure_static] local_static_only=YES isaac_run=NO runtime=NO")
    print(
        "[cube2cm_virtual_runtime_failure_static] training=NO dataset_generation=NO hold_lift=NO "
        "constraints=NO surface_gripper=NO transport_release=NO gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_virtual_runtime_failure_static] gates push_speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f} "
        f"target_error_gate_m={TARGET_ERROR_GATE_M:.6f} support_budget_m={SUPPORT_BUDGET_M:.6f} "
        f"max_plausible_compression_m={MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        f"velocity_damping_residual_ratio={VELOCITY_DAMPING_RESIDUAL_RATIO:.6f}",
        flush=True,
    )

    for sample in STEPS:
        damping_suppression = _safe_suppression(sample.pre_damping_speed_mps, sample.speed_mps)
        target_excess = max(sample.target_error_m - TARGET_ERROR_GATE_M, 0.0)
        support_excess = max(sample.counter_gap_y_m - SUPPORT_BUDGET_M, 0.0)
        max_plausible_margin = MAX_PLAUSIBLE_COMPRESSION_M - sample.counter_gap_y_m
        projected_if_damped = sample.pre_damping_speed_mps * VELOCITY_DAMPING_RESIDUAL_RATIO
        print(
            f"[cube2cm_virtual_runtime_failure_static] sample step={sample.step:03d} "
            f"source={sample.source} "
            f"target_error_m={sample.target_error_m:.6f} target_excess_m={target_excess:.6f} "
            f"speed_mps={sample.speed_mps:.6f} pre_damping_speed_mps={sample.pre_damping_speed_mps:.6f} "
            f"damping_speed_suppression={damping_suppression:.6f} "
            f"projected_if_damped_mps={projected_if_damped:.6f} "
            f"counter_gap_y_m={sample.counter_gap_y_m:.6f} support_excess_m={support_excess:.6f} "
            f"max_plausible_margin_m={max_plausible_margin:.6f} "
            f"virtual_support={_yes(sample.virtual_support)} virtual_damping_active={_yes(sample.virtual_damping_active)} "
            f"one_sided_push={_yes(sample.one_sided_push)}",
            flush=True,
        )

    step4 = STEPS[1]
    step5 = STEPS[2]
    step4_target_excess = step4.target_error_m - TARGET_ERROR_GATE_M
    step5_support_excess = step5.counter_gap_y_m - SUPPORT_BUDGET_M
    final_support_excess_over_max = FINAL_COUNTER_GAP_Y_M - MAX_PLAUSIBLE_COMPRESSION_M
    print(
        "[cube2cm_virtual_runtime_failure_static] remaining_blockers "
        f"step4_target_excess_m={step4_target_excess:.6f} "
        f"step4_target_excess_mm={step4_target_excess * 1000.0:.3f} "
        f"step5_support_excess_m={step5_support_excess:.6f} "
        f"step5_support_excess_mm={step5_support_excess * 1000.0:.3f} "
        f"final_target_error_m={FINAL_TARGET_ERROR_M:.6f} "
        f"final_counter_gap_y_m={FINAL_COUNTER_GAP_Y_M:.6f} "
        f"final_support_excess_over_max_m={final_support_excess_over_max:.6f}",
        flush=True,
    )
    print(
        "[cube2cm_virtual_runtime_failure_static] next_static_requirement="
        "target_error_control_below_3mm_AND_support_or_damping_horizon_beyond_step4; "
        "speed_gate_alone_is_not_success",
        flush=True,
    )
    print("[cube2cm_virtual_runtime_failure_static] CUBE2CM_VIRTUAL_RUNTIME_FAILURE_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
