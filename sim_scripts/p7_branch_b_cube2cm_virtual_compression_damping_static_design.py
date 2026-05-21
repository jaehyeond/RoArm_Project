#!/usr/bin/env python3
"""Static design budget for the next virtual compression+damping proxy.

This script uses the verified rigid-v7 and approved soft-contact close_26
telemetry as constants. It does not launch Isaac or execute a runtime. Its only
job is to quantify what the next explicit compression+damping proxy must change
before another runtime should be approved.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CloseSample:
    source: str
    step: int
    target_error_m: float
    object_speed_mps: float
    counter_gap_y_m: float
    one_sided_push: bool
    reached: bool


RIGID_V7 = (
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377",
        step=3,
        target_error_m=0.001442,
        object_speed_mps=0.061935,
        counter_gap_y_m=0.000652,
        one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:378",
        step=4,
        target_error_m=0.003151,
        object_speed_mps=0.043783,
        counter_gap_y_m=0.001813,
        one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:379",
        step=5,
        target_error_m=0.005002,
        object_speed_mps=0.054294,
        counter_gap_y_m=0.002911,
        one_sided_push=True,
        reached=False,
    ),
)


SOFT_CONTACT = (
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:377",
        step=3,
        target_error_m=0.001541,
        object_speed_mps=0.049059,
        counter_gap_y_m=0.000729,
        one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:378",
        step=4,
        target_error_m=0.003492,
        object_speed_mps=0.037702,
        counter_gap_y_m=0.001989,
        one_sided_push=True,
        reached=False,
    ),
    CloseSample(
        source="/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:379",
        step=5,
        target_error_m=0.005473,
        object_speed_mps=0.051853,
        counter_gap_y_m=0.003205,
        one_sided_push=True,
        reached=False,
    ),
)


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _required_suppression(observed_speed: float, speed_gate: float) -> float:
    if observed_speed <= 0.0:
        return 0.0
    return max(0.0, 1.0 - speed_gate / observed_speed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push_speed_gate_mps", type=float, default=0.005)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--support_budget_m", type=float, default=0.002)
    ap.add_argument("--max_plausible_compression_m", type=float, default=0.003)
    ap.add_argument("--damping_start_close_step", type=int, default=3)
    ap.add_argument("--velocity_damping_residual_ratio", type=float, default=0.08)
    args = ap.parse_args()

    if args.push_speed_gate_mps <= 0.0:
        raise ValueError("push_speed_gate_mps must be positive")
    if args.target_error_gate_m <= 0.0:
        raise ValueError("target_error_gate_m must be positive")
    if args.support_budget_m < 0.0:
        raise ValueError("support_budget_m must be non-negative")
    if args.max_plausible_compression_m < args.support_budget_m:
        raise ValueError("max_plausible_compression_m must be >= support_budget_m")
    if args.damping_start_close_step < 1:
        raise ValueError("damping_start_close_step must be >= 1")
    if not 0.0 <= args.velocity_damping_residual_ratio <= 1.0:
        raise ValueError("velocity_damping_residual_ratio must be in [0, 1]")

    rigid_by_step = {sample.step: sample for sample in RIGID_V7}
    soft_by_step = {sample.step: sample for sample in SOFT_CONTACT}

    print("[cube2cm_virtual_compression_damping_static] local_static_only=YES isaac_run=NO runtime=NO")
    print(
        "[cube2cm_virtual_compression_damping_static] training=NO dataset_generation=NO constraints=NO "
        "surface_gripper=NO transport_release=NO gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_virtual_compression_damping_static] gates push_speed_gate_mps={args.push_speed_gate_mps:.6f} "
        f"target_error_gate_m={args.target_error_gate_m:.6f} support_budget_m={args.support_budget_m:.6f} "
        f"max_plausible_compression_m={args.max_plausible_compression_m:.6f}",
        flush=True,
    )
    print(
        "[cube2cm_virtual_compression_damping_static] proposed_proxy "
        f"compression_budget_m={args.support_budget_m:.6f} "
        f"max_plausible_compression_m={args.max_plausible_compression_m:.6f} "
        f"damping_start_close_step={args.damping_start_close_step} "
        f"velocity_damping_residual_ratio={args.velocity_damping_residual_ratio:.6f} "
        "attach=NO posewrite=NO constraints=NO surface_gripper=NO transport_release=NO env_default_edits=NO",
        flush=True,
    )

    worst_soft_suppression = 0.0
    for step in (3, 4, 5):
        rigid = rigid_by_step[step]
        soft = soft_by_step[step]
        soft_vs_rigid_suppression = 1.0 - soft.object_speed_mps / rigid.object_speed_mps
        required_from_soft = _required_suppression(soft.object_speed_mps, args.push_speed_gate_mps)
        worst_soft_suppression = max(worst_soft_suppression, required_from_soft)
        support_ok = soft.counter_gap_y_m <= args.support_budget_m
        target_ok = soft.target_error_m <= args.target_error_gate_m
        projected_speed = soft.object_speed_mps * args.velocity_damping_residual_ratio
        projected_speed_ok = projected_speed <= args.push_speed_gate_mps
        print(
            f"[cube2cm_virtual_compression_damping_static] sample step={step:03d} "
            f"rigid_source={rigid.source} soft_source={soft.source} "
            f"rigid_speed_mps={rigid.object_speed_mps:.6f} soft_speed_mps={soft.object_speed_mps:.6f} "
            f"material_only_speed_suppression_vs_rigid={soft_vs_rigid_suppression:.6f} "
            f"required_extra_suppression_from_soft={required_from_soft:.6f} "
            f"projected_damped_speed_mps={projected_speed:.6f} projected_speed_ok={_yes(projected_speed_ok)} "
            f"soft_counter_gap_y_m={soft.counter_gap_y_m:.6f} support_ok={_yes(support_ok)} "
            f"soft_target_error_m={soft.target_error_m:.6f} target_ok={_yes(target_ok)} "
            f"soft_one_sided_push={_yes(soft.one_sided_push)}",
            flush=True,
        )

    step4 = soft_by_step[4]
    step5 = soft_by_step[5]
    compression_room_step4_m = max(0.0, args.max_plausible_compression_m - step4.counter_gap_y_m)
    compression_over_budget_step5_m = max(0.0, step5.counter_gap_y_m - args.support_budget_m)

    print(
        f"[cube2cm_virtual_compression_damping_static] design_requirement "
        f"worst_required_extra_speed_suppression_from_soft={worst_soft_suppression:.6f} "
        f"worst_required_extra_speed_suppression_pct={_pct(worst_soft_suppression)} "
        f"step4_compression_room_to_max_m={compression_room_step4_m:.6f} "
        f"step5_over_2mm_budget_m={compression_over_budget_step5_m:.6f}",
        flush=True,
    )
    print(
        "[cube2cm_virtual_compression_damping_static] next_proxy_prediction="
        "must_apply_velocity_damping_by_close_step3_and_bound_counter_support_through_step4; "
        "material_only_changes_are_insufficient",
        flush=True,
    )
    print(
        "[cube2cm_virtual_compression_damping_static] future_runtime_falsifier="
        "step3_speed_above_gate_OR_one_sided_push_steps_2_to_4_OR_step4_target_error_above_gate",
        flush=True,
    )
    print("[cube2cm_virtual_compression_damping_static] CUBE2CM_VIRTUAL_COMPRESSION_DAMPING_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
