#!/usr/bin/env python3
"""Static readiness check for a future virtual compression+damping close_26 runtime.

This script does not launch Isaac and does not execute the runtime probe. It
checks that the default-off compression+damping candidate and the posthoc criteria
audit are wired tightly enough that a future approved run can either pass all
fixed criteria or fail in a way that is immediately diagnosable.
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
V7_D024_USD = "/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
EXPECTED_MECHANISM = "virtual_compression_damping_diagnostic"
B200_MICROMAMBA = "/NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba"
B200_ISAACSIM_ENV = "/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1"
B200_NVML_PRELOAD = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05"
B200_VK_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"


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
        "env",
        "OMNI_KIT_ACCEPT_EULA=YES",
        f"LD_PRELOAD={B200_NVML_PRELOAD}",
        f"VK_ICD_FILENAMES={B200_VK_ICD}",
        B200_MICROMAMBA,
        "run",
        "-p",
        B200_ISAACSIM_ENV,
        "python",
        "sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py",
        "--variant",
        "v7",
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
        "--virtual_compression_damping_diagnostic",
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
        default="/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out",
    )
    args = ap.parse_args()

    runtime_source = _read(RUNTIME_PROBE)
    audit_source = _read(CRITERIA_AUDIT)

    checks = [
        _contains_all(
            "runtime_probe_virtual_compression_damping_default_off_wiring",
            runtime_source,
            (
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
                "--use_synthetic_virtual_no_damping_reference",
                "SOFT_CONTACT_RUNTIME_CRITERIA_PASS=",
            ),
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
            "v7",
            "--close_deg",
            "26.0",
            "--virtual_compression_damping_diagnostic",
        )
    )
    checks.append(
        Check(
            name="future_candidate_command_has_required_flags",
            passed=command_has_required_flags,
            detail="requires variant=v7, close_deg=26.0, and virtual compression+damping flag",
        )
    )

    print("[cube2cm_soft_contact_runtime_readiness] local_static_only=YES isaac_run=NO runtime_probe_executed=NO")
    print(
        "[cube2cm_soft_contact_runtime_readiness] "
        "training=NO dataset_generation=NO constraints=NO surface_gripper=NO transport_release=NO gate_tuning=NO "
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
