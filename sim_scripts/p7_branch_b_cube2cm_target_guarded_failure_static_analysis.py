#!/usr/bin/env python3
"""Static failure attribution for the target-guarded close_26 runtime.

This script encodes the directly verified B200 stdout samples from
`/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out`.
It does not launch Isaac, train, generate data, insert constraints, use a
SurfaceGripper, tune gates, or claim grasp success.
"""
from __future__ import annotations

from dataclasses import dataclass


TARGET_ERROR_DESIGN_LIMIT_M = 0.0027
TARGET_ERROR_GATE_M = 0.0030
COUNTER_SUPPORT_BUDGET_M = 0.0020
MAX_PLAUSIBLE_COMPRESSION_M = 0.0030
CLOSE_TARGET_DEG = 26.0
PUSH_SPEED_GATE_MPS = 0.0050


@dataclass(frozen=True)
class CloseSample:
    line: int
    step: int
    target_error_m: float
    gripper_q_deg: float
    gripper_command_deg: float
    object_speed_mps: float
    object_drift_m: float
    counter_gap_max_m: float
    virtual_compression_gap_max_m: float
    virtual_support: bool
    support_horizon_active: bool
    virtual_damping_active: bool
    target_guarded_close_advance: bool
    target_guarded_close_hold: bool
    target_guarded_close_advances_total: int
    target_guarded_close_holds_total: int
    one_sided_push: bool

    @property
    def command_backlog_deg(self) -> float:
        return self.gripper_command_deg - self.gripper_q_deg

    @property
    def final_close_remaining_deg(self) -> float:
        return CLOSE_TARGET_DEG - self.gripper_q_deg


SAMPLES = [
    CloseSample(377, 1, 0.001602, 0.000, 0.000, 0.001572, 0.000000, 0.000001, 0.002533, False, True, False, True, False, 1, 0, False),
    CloseSample(378, 2, 0.001311, 0.361, 2.000, 0.001576, 0.000000, 0.000001, 0.002269, False, True, False, True, False, 2, 0, False),
    CloseSample(379, 3, 0.001095, 1.019, 4.000, 0.000126, 0.000000, 0.000001, 0.001794, True, True, True, True, False, 3, 0, False),
    CloseSample(380, 4, 0.000943, 1.918, 6.000, 0.000126, 0.000000, 0.000001, 0.001138, True, True, True, True, False, 4, 0, False),
    CloseSample(381, 5, 0.000844, 3.017, 8.000, 0.000126, 0.000000, 0.000001, 0.000330, True, True, True, True, False, 5, 0, False),
    CloseSample(382, 6, 0.000559, 4.274, 10.000, 0.001147, 0.000026, 0.000262, 0.000262, True, True, True, True, False, 6, 0, False),
    CloseSample(383, 7, 0.001558, 5.661, 12.000, 0.002205, 0.000027, 0.001141, 0.001141, True, True, True, True, False, 7, 0, False),
    CloseSample(384, 8, 0.003108, 7.158, 14.000, 0.002159, 0.000032, 0.002067, 0.002067, False, True, True, False, True, 7, 1, False),
    CloseSample(385, 9, 0.004406, 8.385, 14.000, 0.001793, 0.000033, 0.002825, 0.002825, False, True, True, False, True, 7, 2, False),
    CloseSample(386, 10, 0.005448, 9.390, 14.000, 0.033058, 0.000049, 0.003427, 0.003427, False, False, False, False, True, 7, 3, True),
    CloseSample(421, 45, 0.010622, 13.947, 14.000, 0.028852, 0.000037, 0.006551, 0.006551, False, False, False, False, True, 7, 38, True),
]


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> int:
    first_hold = next(sample for sample in SAMPLES if sample.target_guarded_close_hold)
    horizon_loss = next(sample for sample in SAMPLES if not sample.support_horizon_active)
    final = SAMPLES[-1]
    last_advance = max(
        (sample for sample in SAMPLES if sample.target_guarded_close_advance),
        key=lambda sample: sample.step,
    )

    print("[cube2cm_target_guarded_failure_static] local_static_only=YES isaac_run=NO training=NO")
    print(
        "[cube2cm_target_guarded_failure_static] "
        "dataset_generation=NO hold_lift=NO constraints=NO surface_gripper=NO "
        "transport_release=NO gate_tuning=NO success_claim=NO"
    )
    print(
        "[cube2cm_target_guarded_failure_static] source_log="
        "/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out "
        "md5=c9ae7f3af650a87c3f38ba2d8e41d5b1"
    )
    print(
        "[cube2cm_target_guarded_failure_static] first_hold "
        f"line={first_hold.line} step={first_hold.step} "
        f"target_error_m={first_hold.target_error_m:.6f} "
        f"design_limit_m={TARGET_ERROR_DESIGN_LIMIT_M:.6f} "
        f"gripper_q_deg={first_hold.gripper_q_deg:.3f} "
        f"gripper_command_deg={first_hold.gripper_command_deg:.3f} "
        f"command_backlog_deg={first_hold.command_backlog_deg:.3f} "
        f"support_horizon_active={_yes(first_hold.support_horizon_active)} "
        f"virtual_damping_active={_yes(first_hold.virtual_damping_active)}"
    )
    print(
        "[cube2cm_target_guarded_failure_static] horizon_loss "
        f"line={horizon_loss.line} step={horizon_loss.step} "
        f"counter_gap_max_m={horizon_loss.counter_gap_max_m:.6f} "
        f"max_plausible_compression_m={MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        f"horizon_excess_m={horizon_loss.counter_gap_max_m - MAX_PLAUSIBLE_COMPRESSION_M:.6f} "
        f"object_speed_mps={horizon_loss.object_speed_mps:.6f} "
        f"push_speed_gate_mps={PUSH_SPEED_GATE_MPS:.6f} "
        f"one_sided_push={_yes(horizon_loss.one_sided_push)}"
    )
    print(
        "[cube2cm_target_guarded_failure_static] final_plateau "
        f"line={final.line} step={final.step} "
        f"target_error_m={final.target_error_m:.6f} "
        f"counter_gap_max_m={final.counter_gap_max_m:.6f} "
        f"object_drift_m={final.object_drift_m:.6f} "
        f"gripper_q_deg={final.gripper_q_deg:.3f} "
        f"gripper_command_deg={final.gripper_command_deg:.3f} "
        f"final_close_remaining_deg={final.final_close_remaining_deg:.3f} "
        f"target_guarded_close_holds_total={final.target_guarded_close_holds_total}"
    )
    print(
        "[cube2cm_target_guarded_failure_static] backlog_growth_after_last_advance "
        f"last_advance_line={last_advance.line} last_advance_step={last_advance.step} "
        f"last_advance_command_backlog_deg={last_advance.command_backlog_deg:.3f} "
        f"first_hold_command_backlog_deg={first_hold.command_backlog_deg:.3f} "
        f"actual_gripper_motion_while_held_deg={final.gripper_q_deg - first_hold.gripper_q_deg:.3f} "
        f"target_error_growth_while_held_m={final.target_error_m - first_hold.target_error_m:.6f} "
        f"counter_gap_growth_while_held_m={final.counter_gap_max_m - first_hold.counter_gap_max_m:.6f}"
    )
    print(
        "[cube2cm_target_guarded_failure_static] geometry_budget "
        f"step10_extra_gap_to_keep_horizon_m={max(0.0, horizon_loss.counter_gap_max_m - MAX_PLAUSIBLE_COMPRESSION_M):.6f} "
        f"final_extra_gap_to_keep_horizon_m={max(0.0, final.counter_gap_max_m - MAX_PLAUSIBLE_COMPRESSION_M):.6f} "
        f"final_extra_gap_to_meet_support_budget_m={max(0.0, final.counter_gap_max_m - COUNTER_SUPPORT_BUDGET_M):.6f}"
    )
    print(
        "[cube2cm_target_guarded_failure_static] attribution "
        "primary=advance_scheduling_and_hold_backlog "
        "guard_release_relaxation=REJECT "
        "support_horizon_only=INSUFFICIENT "
        "jaw_geometry=SECONDARY_AFTER_BACKLOG_OR_TRUE_COMPLIANT_COUNTER"
    )
    print(
        "[cube2cm_target_guarded_failure_static] recommended_next_mechanism "
        "zero_backlog_hold=YES "
        "advance_requires_gripper_command_convergence=YES "
        "advance_requires_support_margin=YES "
        "advance_requires_nonworsening_target_error=YES "
        "do_not_expand_horizon_as_label_only=YES"
    )
    print("[cube2cm_target_guarded_failure_static] TARGET_GUARDED_FAILURE_STATIC_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
