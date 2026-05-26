#!/usr/bin/env python3
"""Static readiness check for the target-guarded v8 close_26 runtime.

This script does not launch Isaac and does not execute the runtime probe. It
checks that the default-off target-guarded micro-close v8 observed recovery
candidate and the posthoc criteria audit are wired tightly enough that a future
approved local/RunPod run can either pass all fixed criteria or fail in a way
that is immediately diagnosable.

The post-reboot v7 runtime failed posthoc, so the future command printed here
uses a new v8 mechanism. It is still only a command shape for separate runtime
approval.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME_PROBE = REPO / "sim_scripts" / "p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py"
CRITERIA_AUDIT = REPO / "sim_scripts" / "p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py"
V7_D024_USD = str(
    REPO
    / "b200_backup_20260522_final"
    / "tmp_p7"
    / "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024"
    / "roarm_m3.usd"
)
EXPECTED_MECHANISM = "target_guarded_micro_close_v8_observed_recovery_diagnostic"
OLD_V6_RUNTIME_LOG = (
    REPO
    / "b200_backup_20260522_final"
    / "tmp_p7"
    / "p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out"
)
OLD_V6_RUNTIME_MD5 = "9a4f8825a88ee3c9d93d83e5b9a28b41"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_all(name: str, text: str, needles: tuple[str, ...]) -> Check:
    missing = [needle for needle in needles if needle not in text]
    return Check(name=name, passed=not missing, detail=f"missing={missing}")


def _contains_in_block(name: str, text: str, start: str, end: str, needles: tuple[str, ...]) -> Check:
    start_idx = text.find(start)
    if start_idx < 0:
        return Check(name=name, passed=False, detail=f"missing_start={start!r}")
    end_idx = text.find(end, start_idx)
    if end_idx < 0:
        return Check(name=name, passed=False, detail=f"missing_end={end!r}")
    block = text[start_idx:end_idx]
    missing = [needle for needle in needles if needle not in block]
    return Check(name=name, passed=not missing, detail=f"missing={missing}")


def _run_expected(label: str, args: list[str], expected_code: int) -> Check:
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=False)
    summary = ""
    for line in proc.stdout.splitlines():
        if "SOFT_CONTACT_RUNTIME_CRITERIA_PASS=" in line:
            summary = line
    passed = proc.returncode == expected_code
    return Check(
        name=label,
        passed=passed,
        detail=f"returncode={proc.returncode} expected={expected_code} summary='{summary}'",
    )


def _candidate_command() -> list[str]:
    return [
        "python",
        "sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py",
        "--variant",
        "v8",
        "--robot_usd_path",
        V7_D024_USD,
        "--object_size_m",
        "0.030",
        "0.030",
        "0.030",
        "--close_deg",
        "26.0",
        "--log_every_close_step",
        "1",
        "--target_guarded_micro_close_v8_observed_recovery_diagnostic",
    ]


def _candidate_audit_command(future_stdout_log: str) -> list[str]:
    return [
        "python",
        "sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py",
        "--log",
        future_stdout_log,
        "--expected_mechanism",
        EXPECTED_MECHANISM,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--future_stdout_log",
            default="/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v8_observed_recovery_v8_close26.out",
    )
    args = ap.parse_args()

    runtime_source = _read(RUNTIME_PROBE)
    audit_source = _read(CRITERIA_AUDIT)

    checks = [
        _contains_all(
            "runtime_probe_target_guarded_micro_close_default_off_wiring",
            runtime_source,
            (
                "--target_guarded_micro_close_support_horizon_diagnostic",
                "--target_guarded_micro_close_v2_convergence_diagnostic",
                "--target_guarded_micro_close_v3_progress_diagnostic",
                "--target_guarded_micro_close_v4_recovery_diagnostic",
                "--target_guarded_micro_close_v5_preemptive_recovery_diagnostic",
                "--target_guarded_micro_close_v6_projected_guard_diagnostic",
                "--target_guarded_micro_close_v7_active_recovery_diagnostic",
                "--target_guarded_micro_close_v8_observed_recovery_diagnostic",
                "target_guarded_micro_close_support_horizon_diagnostic=",
                "target_guarded_micro_close_v2_convergence_diagnostic=",
                "target_guarded_micro_close_v3_progress_diagnostic=",
                "target_guarded_micro_close_v4_recovery_diagnostic=",
                "target_guarded_micro_close_v5_preemptive_recovery_diagnostic=",
                "target_guarded_micro_close_v6_projected_guard_diagnostic=",
                "target_guarded_micro_close_v7_active_recovery_diagnostic=",
                "target_guarded_micro_close_v8_observed_recovery_diagnostic=",
                "target_guarded_micro_close_support_horizon",
                "target_guarded_close_advance",
                "target_guarded_close_advances_total",
                "target_guarded_zero_backlog_hold",
                "target_guarded_backlog_preserved_hold",
                "target_guarded_v3_safety_rollback",
                "target_guarded_command_backlog_deg",
                "target_guarded_command_converged",
                "target_guarded_support_margin_ok",
                "target_guarded_support_budget_ok",
                "target_guarded_target_nonworsening",
                "target_guarded_v3_actual_progress_deg",
                "target_guarded_v3_actual_progress_ok",
                "target_guarded_v3_backlog_room_ok",
                "target_guarded_v4_recovery_hold",
                "target_guarded_v4_hard_safety_freeze",
                "target_guarded_v4_recovery_holds_total",
                "target_guarded_v4_hard_safety_freezes_total",
                "target_guarded_v5_preemptive_recovery_needed",
                "target_guarded_v5_preemptive_recovery",
                "target_guarded_v5_recovery_ik_ok",
                "target_guarded_v5_preemptive_recovery_writes_total",
                "target_guarded_v5_recovery_ik_failures_total",
                "target_guarded_v6_projected_advance_ok",
                "target_guarded_v6_projected_target_margin_m",
                "target_guarded_v6_projected_support_margin_m",
                "_v7_active_recovery_decision",
                "v7_finite_difference_tcp_sweep=YES",
                "v7_recovery_uses_current_object_pose=YES",
                "v7_object_posewrite=NO",
                "v7_recovery_writes_robot_joint_targets_only=YES",
                "v8_projected_reserve_trigger=YES",
                "v8_observed_response_audit=POSTHOC_ONLY",
                "v8_candidate_counter_contact_required=YES",
                "v8_object_posewrite=NO",
                "v8_recovery_writes_robot_joint_targets_only=YES",
                "target_guarded_v7_active_recovery_needed",
                "target_guarded_v7_active_recovery",
                "target_guarded_v7_recovery_ik_ok",
                "target_guarded_v7_candidate_count",
                "target_guarded_v7_counter_gap_delta_m",
                "target_guarded_v7_candidate_counter_contact",
                "target_guarded_v7_candidate_counter_slop_contact",
                "target_guarded_v8_projected_reserve_trigger",
                "target_guarded_v7_active_recovery_writes_total",
                "target_guarded_v7_recovery_ik_failures_total",
                "support_horizon_active",
                "--virtual_compression_damping_diagnostic",
                "virtual_compression_damping_diagnostic=",
                "virtual_compression_damping",
                "virtual_damping_active",
                "write_root_velocity_to_sim",
                "virtual_velocity_damping_writes",
                "runtime_candidate_requires_separate_approval=",
                "future_close26_posthoc_criteria",
                "runtime_gate=NO",
            ),
        ),
        _contains_all(
            "criteria_audit_metadata_guard",
            audit_source,
            (
                "soft_contact_material_diagnostic_enabled",
                "virtual_compression_damping_diagnostic_enabled",
                "object_physics_mode_matches_expected",
                "runtime_candidate_marker_yes",
                "virtual_damping_active_step3",
                "virtual_damping_write_seen_by_step3",
                "virtual_velocity_damping_writes_positive",
                "target_guarded_micro_close_support_horizon_diagnostic_enabled",
                "target_guarded_micro_close_v2_convergence_diagnostic_enabled",
                "target_guarded_micro_close_v3_progress_diagnostic_enabled",
                "target_guarded_micro_close_v4_recovery_diagnostic_enabled",
                "target_guarded_micro_close_v5_preemptive_recovery_diagnostic_enabled",
                "target_guarded_micro_close_v6_projected_guard_diagnostic_enabled",
                "target_guarded_micro_close_v7_active_recovery_diagnostic_enabled",
                "target_guarded_micro_close_v8_observed_recovery_diagnostic_enabled",
                "target_guarded_close_advances_positive",
                "target_guarded_command_backlog_step3_within_gate",
                "target_guarded_v2_zero_backlog_on_every_hold",
                "target_guarded_progress_backlog_preserved_holds_positive",
                "target_guarded_progress_zero_backlog_holds_zero",
                "target_guarded_progress_actual_progress_step3",
                "target_guarded_progress_no_zero_backlog_holds",
                "target_guarded_v4_recovery_holds_positive",
                "target_guarded_v4_hard_safety_freezes_zero",
                "target_guarded_v5_preemptive_recovery_writes_positive",
                "target_guarded_v5_preemptive_recovery_present",
                "target_guarded_v5_preemptive_trigger_seen",
                "target_guarded_v5_recovery_ik_ok_all",
                "target_guarded_v7_active_recovery_writes_positive",
                "target_guarded_v7_recovery_ik_failures_zero",
                "target_guarded_v7_active_recovery_present",
                "target_guarded_v7_active_recovery_trigger_seen",
                "target_guarded_v7_active_recovery_reduces_counter_gap",
                "target_guarded_v7_active_recovery_selected_margins_valid",
                "target_guarded_v8_projected_reserve_trigger_seen",
                "target_guarded_v8_observed_response_not_worsening_both",
                "target_guarded_v8_active_recovery_tcp_follow_positive",
                "target_guarded_v8_candidate_counter_contact_modeled",
                "target_guarded_v4_all_close_target_within_fixed_gate",
                "support_horizon_step5",
                "--use_synthetic_virtual_no_damping_reference",
                "--use_synthetic_v3_zero_backlog_reference",
                "--use_synthetic_v4_hard_freeze_reference",
                "--use_synthetic_v7_no_active_recovery_reference",
                "--use_synthetic_v8_worsening_response_reference",
                "--use_synthetic_v8_no_tcp_follow_reference",
                "--use_synthetic_v8_no_counter_contact_reference",
                "SOFT_CONTACT_RUNTIME_CRITERIA_PASS=",
            ),
        ),
        _contains_in_block(
            "runtime_probe_v8_inherits_virtual_damping_active",
            runtime_source,
            "virtual_damping_active = bool(",
            "target_guarded_v5_preemptive_recovery_needed = False",
            (
                "args.target_guarded_micro_close_v7_active_recovery_diagnostic",
                "args.target_guarded_micro_close_v8_observed_recovery_diagnostic",
                "step_idx >= int(args.virtual_damping_start_close_step)",
                "write_root_velocity_to_sim",
            ),
        ),
        _run_expected(
            "criteria_audit_rejects_archived_v6_runtime_as_v8",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--log",
                str(OLD_V6_RUNTIME_LOG),
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v7_reference",
            [sys.executable, str(CRITERIA_AUDIT), "--use_v7_reference", "--expected_mechanism", EXPECTED_MECHANISM],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_virtual_no_damping_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_virtual_no_damping_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v3_zero_backlog_reference_as_v8",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v3_zero_backlog_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v4_hard_freeze_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v4_hard_freeze_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v7_no_active_recovery_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v7_no_active_recovery_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v8_worsening_response_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v8_worsening_response_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v8_no_tcp_follow_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v8_no_tcp_follow_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_rejects_v8_no_counter_contact_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_v8_no_counter_contact_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            1,
        ),
        _run_expected(
            "criteria_audit_accepts_synthetic_pass_reference",
            [
                sys.executable,
                str(CRITERIA_AUDIT),
                "--use_synthetic_pass_reference",
                "--expected_mechanism",
                EXPECTED_MECHANISM,
            ],
            0,
        ),
    ]

    command = _candidate_command()
    audit_command = _candidate_audit_command(args.future_stdout_log)
    command_has_required_flags = all(
        item in command
        for item in (
            "--variant",
            "v8",
            "--close_deg",
            "26.0",
            "--target_guarded_micro_close_v8_observed_recovery_diagnostic",
        )
    )
    checks.append(
        Check(
            name="future_candidate_command_has_required_flags",
            passed=command_has_required_flags,
            detail=(
                "requires variant=v8, close_deg=26.0, and target-guarded v8 observed recovery flag; "
                f"archived v6 runtime stdout_md5={OLD_V6_RUNTIME_MD5}"
            ),
        )
    )

    print("[cube2cm_soft_contact_runtime_readiness] local_static_only=YES isaac_run=NO runtime_probe_executed=NO")
    print(
        "[cube2cm_soft_contact_runtime_readiness] "
        "training=NO dataset_generation=NO constraints=NO surface_gripper=NO object_attach=NO posewrite=NO "
        "transport_release=NO gate_tuning=NO "
        "success_claim=NO",
        flush=True,
    )
    for check in checks:
        print(
            f"[cube2cm_soft_contact_runtime_readiness] check name={check.name} pass={_yes(check.passed)} "
            f"{check.detail}",
            flush=True,
        )

    print(
        "[cube2cm_soft_contact_runtime_readiness] future_runtime_command_requires_separate_approval='"
        + shlex.join(command)
        + "'",
        flush=True,
    )
    print(
        "[cube2cm_soft_contact_runtime_readiness] "
        "future_runtime_command_status=V8_REQUIRES_SEPARATE_RUNTIME_APPROVAL",
        flush=True,
    )
    print(
        "[cube2cm_soft_contact_runtime_readiness] first_posthoc_audit_command_after_future_run='"
        + shlex.join(audit_command)
        + "'",
        flush=True,
    )
    ready = all(check.passed for check in checks)
    print(f"[cube2cm_soft_contact_runtime_readiness] READY_FOR_SEPARATE_RUNTIME_APPROVAL={_yes(ready)}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
