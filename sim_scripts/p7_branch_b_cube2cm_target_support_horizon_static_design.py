#!/usr/bin/env python3
"""Static design for the next Track A P7/Branch B close-control mechanism.

The approved virtual compression+damping runtime showed that object-velocity
damping can fix the step-3 speed gate while active, but it did not keep the
close target within 3 mm or preserve the support/damping window after step 4.

This script is intentionally static-only. It encodes verified B200 log lines and
selects the smallest falsifiable next mechanism shape without launching Isaac,
changing defaults, tuning gates, or claiming success.
"""
from __future__ import annotations

from dataclasses import dataclass


TARGET_ERROR_GATE_M = 0.003
TARGET_ERROR_DESIGN_MARGIN_M = 0.0003
TARGET_ERROR_DESIGN_LIMIT_M = TARGET_ERROR_GATE_M - TARGET_ERROR_DESIGN_MARGIN_M
SUPPORT_BUDGET_M = 0.002
MAX_PLAUSIBLE_COMPRESSION_M = 0.003
PUSH_SPEED_GATE_MPS = 0.005
VELOCITY_DAMPING_RESIDUAL_RATIO = 0.08


@dataclass(frozen=True)
class CloseSample:
    step: int
    source: str
    gripper_q_deg: float
    gripper_err_deg: float
    target_error_m: float
    counter_gap_y_m: float
    speed_mps: float
    pre_damping_speed_mps: float
    damping_active: bool
    one_sided_push: bool


SAMPLES = (
    CloseSample(
        step=3,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:378",
        gripper_q_deg=5.398,
        gripper_err_deg=20.602,
        target_error_m=0.001442,
        counter_gap_y_m=0.000652,
        speed_mps=0.004955,
        pre_damping_speed_mps=0.061935,
        damping_active=True,
        one_sided_push=False,
    ),
    CloseSample(
        step=4,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:379",
        gripper_q_deg=7.197,
        gripper_err_deg=18.803,
        target_error_m=0.003130,
        counter_gap_y_m=0.001794,
        speed_mps=0.003203,
        pre_damping_speed_mps=0.040032,
        damping_active=True,
        one_sided_push=False,
    ),
    CloseSample(
        step=5,
        source="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:380",
        gripper_q_deg=8.996,
        gripper_err_deg=17.004,
        target_error_m=0.004843,
        counter_gap_y_m=0.002738,
        speed_mps=0.050912,
        pre_damping_speed_mps=0.050912,
        damping_active=False,
        one_sided_push=True,
    ),
)

FINAL_SOURCE = "/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out:420"
FINAL_TARGET_ERROR_M = 0.022778
FINAL_COUNTER_GAP_Y_M = 0.013828


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> int:
    print("[cube2cm_target_support_horizon_static] local_static_only=YES isaac_run=NO runtime=NO")
    print(
        "[cube2cm_target_support_horizon_static] training=NO dataset_generation=NO hold_lift=NO "
        "constraints=NO surface_gripper=NO transport_release=NO gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_target_support_horizon_static] fixed_gates target_error_gate_m={TARGET_ERROR_GATE_M:.6f} "
        f"support_budget_m={SUPPORT_BUDGET_M:.6f} max_plausible_compression_m={MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        f"push_speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_target_support_horizon_static] design_limits target_error_design_limit_m={TARGET_ERROR_DESIGN_LIMIT_M:.6f} "
        f"target_error_design_margin_m={TARGET_ERROR_DESIGN_MARGIN_M:.6f} "
        f"velocity_damping_residual_ratio={VELOCITY_DAMPING_RESIDUAL_RATIO:.6f}",
        flush=True,
    )

    for sample in SAMPLES:
        projected_if_damped = sample.pre_damping_speed_mps * VELOCITY_DAMPING_RESIDUAL_RATIO
        target_excess_gate = max(sample.target_error_m - TARGET_ERROR_GATE_M, 0.0)
        target_excess_design = max(sample.target_error_m - TARGET_ERROR_DESIGN_LIMIT_M, 0.0)
        support_excess_budget = max(sample.counter_gap_y_m - SUPPORT_BUDGET_M, 0.0)
        support_excess_max = max(sample.counter_gap_y_m - MAX_PLAUSIBLE_COMPRESSION_M, 0.0)
        print(
            f"[cube2cm_target_support_horizon_static] sample step={sample.step:03d} "
            f"source={sample.source} gripper_q_deg={sample.gripper_q_deg:.3f} "
            f"gripper_err_deg={sample.gripper_err_deg:.3f} "
            f"target_error_m={sample.target_error_m:.6f} "
            f"target_excess_gate_m={target_excess_gate:.6f} "
            f"target_excess_design_m={target_excess_design:.6f} "
            f"counter_gap_y_m={sample.counter_gap_y_m:.6f} "
            f"support_excess_budget_m={support_excess_budget:.6f} "
            f"support_excess_max_plausible_m={support_excess_max:.6f} "
            f"speed_mps={sample.speed_mps:.6f} pre_damping_speed_mps={sample.pre_damping_speed_mps:.6f} "
            f"projected_if_damped_mps={projected_if_damped:.6f} "
            f"projected_if_damped_speed_ok={_yes(projected_if_damped <= PUSH_SPEED_GATE_MPS)} "
            f"damping_active={_yes(sample.damping_active)} one_sided_push={_yes(sample.one_sided_push)}",
            flush=True,
        )

    step4 = SAMPLES[1]
    step5 = SAMPLES[2]
    print(
        "[cube2cm_target_support_horizon_static] rejection_checks "
        f"stronger_damping_only=REJECTED step4_target_excess_mm={(step4.target_error_m - TARGET_ERROR_GATE_M) * 1000.0:.3f} "
        f"step5_target_excess_mm={(step5.target_error_m - TARGET_ERROR_GATE_M) * 1000.0:.3f} "
        f"support_label_only=REJECTED final_counter_gap_y_m={FINAL_COUNTER_GAP_Y_M:.6f} "
        f"final_gap_over_max_plausible_m={FINAL_COUNTER_GAP_Y_M - MAX_PLAUSIBLE_COMPRESSION_M:.6f}",
        flush=True,
    )
    print(
        "[cube2cm_target_support_horizon_static] proposed_next_mechanism "
        "default_off_target_guarded_micro_close=YES "
        "advance_close_only_when_target_error_below_design_limit=YES "
        "support_horizon_damping_until_max_plausible_compression=YES "
        "audit_still_uses_fixed_3mm_target_gate=YES "
        "audit_still_uses_2mm_step4_support_budget=YES "
        "attach=NO posewrite=NO constraints=NO surface_gripper=NO transport_release=NO env_default_edits=NO",
        flush=True,
    )
    print(
        "[cube2cm_target_support_horizon_static] future_runtime_falsifiers "
        "step4_target_error_gt_0p003_OR_step5_support_gt_0p003_OR_close_reached_NO_OR_attach_posewrite_nonzero",
        flush=True,
    )
    print("[cube2cm_target_support_horizon_static] CUBE2CM_TARGET_SUPPORT_HORIZON_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
