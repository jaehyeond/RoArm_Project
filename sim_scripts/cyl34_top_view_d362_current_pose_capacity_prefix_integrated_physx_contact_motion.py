#!/usr/bin/env python3
"""D362: capacity/prefix-integrated current-pose PhysX contact-motion rerun.

This forward-only case changes exactly two variables relative to frozen D360:
(1) D361's version-aligned total contact capacity plus durable per-step prefix,
and (2) an interface-visible canonical-trace replay video.  The q5 perturbation,
scene, target, initial state, physics, thresholds, and actual Isaac views remain
the D360 contract.  It never classifies an exact collider face, cap/rim order,
force closure, grasp, or G0a success.

Execution is intentionally split into prepare, run, _worker, and finalize.
The run stage permits one supervised Isaac worker and no automatic retry.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import time
import textwrap
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# D351 has the frozen D348/D349/D333 import graph and is safe before AppLauncher.
from sim_scripts import cyl34_top_view_d351_zero_step_closure_geometry as d351  # noqa: E402
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d361_contact_point_capacity_and_prefix_trace_repair as d361,
)


CASE = "g0a_d362"
CASE_NAME = "current_pose_capacity_prefix_integrated_physx_contact_motion"
NEW_VARIABLES = [
    "runtime_contact_capacity_and_durable_prefix_integration",
    "interface_visible_trace_replay_video",
]
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
BASE_GIT = "68f2ff040831c13b0198fe68ef88fe84a76a9df3"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
RERUN_VERSION = "0.34.1"
SEED = 33201
BASELINE_STEPS = 200
CLOSURE_MAX_STEPS = 300
PHYSICS_DT_S = 0.005
INACTIVITY_WATCHDOG_S = 300.0
TOTAL_WATCHDOG_S = 900.0
MIN_GPU_FREE_MIB = 8192
MIN_RAM_AVAILABLE_BYTES = 10 * 1024**3
VIEWPORT_SIZE = [1280, 720]
RERUN_SCREENSHOT_SIZE = "4800x2800"
REGISTERED_TOTAL_CONTACT_CAPACITY = 33_280
VIDEO_SIZE = [1920, 1080]
VIDEO_FPS = 20
VIDEO_STRIDE = 2
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_EXPECTED_FRAME_COUNT = (BASELINE_STEPS + CLOSURE_MAX_STEPS + VIDEO_STRIDE - 1) // VIDEO_STRIDE
VIDEO_REPLAY_AUTHORITY = "canonical D362 trace + D348 live collider display copy; physics not recomputed"

Q_FROZEN_OPEN_F32 = np.asarray(
    [
        0.03750238195061684,
        0.542945146560669,
        1.9687392711639404,
        0.18299327790737152,
        0.0,
        1.5413000583648682,
    ],
    dtype=np.float32,
)
Q5_OPEN_F32 = np.float32(1.5413000583648682)
Q5_CLOSED_F32 = np.float32(0.0)
D354_LAST_CLEAR_Q5_F32 = np.float32(1.0269782543182373)
D354_FIRST_OVERLAP_Q5_F32 = np.float32(1.0269775390625)
OBJECT_POS_F32 = np.asarray([0.30000001192092896, 0.0, 0.03288299962878227], dtype=np.float32)
OBJECT_QUAT_F32 = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

CYLINDER_RADIUS_M = 0.017
CYLINDER_HEIGHT_M = 0.090
OBJECT_MASS_KG = 0.72
STATIC_FRICTION = 1.5
DYNAMIC_FRICTION = 1.2
RESTITUTION = 0.0
ACTUATOR_STIFFNESS = 80.0
ACTUATOR_DAMPING = 4.0
ACTUATOR_EFFORT_LIMIT_NM = 2.5
ACTUATOR_VELOCITY_LIMIT_RAD_S = 3.14

ROBOT_FORCE_EVENT_N = 0.1
OBJECT_XY_EVENT_M = 0.0005
OBJECT_TILT_EVENT_DEG = 1.0
CONSECUTIVE_EVENT_STEPS = 2
SUPPORT_POSITIVE_CONTROL_N = 1.0
# One Float32-near-cap telemetry band, inherited in spirit from D326's
# applied-torque saturation audit.  It is diagnostic, never a science gate.
EFFORT_SATURATION_ABS_TOL_NM = 1.0e-4

# Reuse the D351/D354 view that already showed jaw and cylinder together.
CAMERA_EYE = [0.49, -0.32, 0.28]
OPPOSITE_CAMERA_EYE = [0.49, 0.32, 0.28]
CAMERA_TARGET = [0.285, 0.0, 0.055]
PHYSX_COLLIDER_SETTING = "/persistent/physics/visualizationDisplayColliders"
PLAY_SIMULATIONS_SETTING = "/app/player/playSimulations"
WORKER_TOKEN_ENV = "D362_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D362_SUPERVISOR_PID"

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md"
D360_SESSION = REPO / "claudedocs/session_20260716_grasp_g0a_d360_current_pose_bounded_physx_contact_motion.md"
D360_HARNESS = REPO / "sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py"
D360_PREREQUISITES = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_runtime_prerequisites.json"
D360_EXCEPTION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_exception.json"
D360_SUPERVISOR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_supervisor_summary.json"
D360_WORKER_LOG = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_stdout_stderr.log"
D361_PREREGISTRATION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_preregistration.json"
D361_PREPARE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_prepare_preflight.json"
D361_CAPACITY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_contact_capacity_budget.json"
D361_PROTOCOL = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_prefix_protocol_contract.json"
D361_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_completion_summary.json"
D360_D361_EXPECTED_HASHES = {
    D360_SESSION: "97ef49eb31a754be4f12ea0c5f961ddfede4d19656df7147e3f10a63b21f9291",
    D360_HARNESS: "86bd2af855effb3bc31f067fd6cc7a4cb7088c422da3ae828d48f4e31d92fd5a",
    D360_PREREQUISITES: "bfd20dc2ab678d9929c0dd8f54323f8f4b17707c6c89d099b1374b814c467dae",
    D360_EXCEPTION: "1dfb7ffe863f77ece3dc8acafc09b134325b147f1a392907ad5692a9fd650fb1",
    D360_SUPERVISOR: "54bb8a80569048e0299183ae8ed86e81b82a527ec67329c43d4c9e8cb12f026c",
    D360_WORKER_LOG: "1bd0aa5a6060da283c8f84b83a0608156624de68aea47eba8c08b5878fc3ecf5",
    D361_PREREGISTRATION: "87250fc30146887dd4c372092bed95a9491c67e7812426834ee37467c7c6de76",
    D361_PREPARE: "b0ade378ab1b673b2dceca18141b91fdcb451b0706b06c6c4f095b3a67022c72",
    D361_CAPACITY: "ca5edc818fee321dad257dcf3d1ba4574b6c446471a7ff7abc7c8ab790bf79f5",
    D361_PROTOCOL: "dead1591fbf53f080bae0ca28643bb22147ee99a6b07f71a9285865e2c1d8e13",
    D361_COMPLETION: "3a1c9ce273182e320e3b90dd82c2ab28d8f356b892b5e4b5b346ffee8912b095",
}
D354_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_closure_geometry_measurement.json"
D354_ATTESTATION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_science_attestation.json"
D359_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d359/d359_completion_summary.json"
D359_CLARIFICATION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d359/d359_postcompletion_lineage_clarification.json"
D348_EVIDENCE = d351.D348_EVIDENCE
D334_SUMMARY = d351.D334_SUMMARY
VARIANT_ROBOT_USD = d351.VARIANT_ROBOT_USD
VARIANT_PHYSICS_USD = d351.VARIANT_PHYSICS_USD
URDF_PATH = d351.d333.DEFAULT_URDF

PREREG_PATH = OUT_DIR / "d362_preregistration.json"
PREPARE_PATH = OUT_DIR / "d362_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d362_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d362_worker_preflight.json"
PHASE_PATH = OUT_DIR / "d362_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d362_worker_stdout_stderr.log"
PREREQUISITE_PATH = OUT_DIR / "d362_runtime_prerequisites.json"
TRACE_JSON_PATH = OUT_DIR / "d362_physics_trace.json"
TRACE_CSV_PATH = OUT_DIR / "d362_physics_trace.csv"
WORKER_SUMMARY_PATH = OUT_DIR / "d362_worker_summary.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d362_worker_exception.json"
SUPERVISOR_PATH = OUT_DIR / "d362_supervisor_summary.json"
AUTOMATED_PATH = OUT_DIR / "d362_automated_summary.json"
MANUAL_PATH = OUT_DIR / "d362_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d362_completion_summary.json"
RRD_PATH = OUT_DIR / "d362_physx_contact_motion.rrd"
RBL_PATH = OUT_DIR / "d362_physx_contact_motion.rbl"
RERUN_PNG_PATH = OUT_DIR / "d362_physx_contact_motion_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d362_rerun_validation.json"
PREFIX_PATH = OUT_DIR / "d362_durable_step_prefix.jsonl"
PREFIX_AUDIT_PATH = OUT_DIR / "d362_durable_step_prefix_audit.json"
FAILURE_PREFIX_AUDIT_PATH = OUT_DIR / "d362_failure_prefix_audit.json"
SUPERVISOR_PREFIX_AUDIT_PATH = OUT_DIR / "d362_supervisor_prefix_recovery_audit.json"
CONTACT_OVERFLOW_WARNING_AUDIT_PATH = (
    OUT_DIR / "d362_contact_overflow_warning_audit.json"
)
VIDEO_PATH = OUT_DIR / "d362_interface_visible_trace_replay.mp4"
VIDEO_REPORT_PATH = OUT_DIR / "d362_interface_visible_trace_replay_report.json"
VIDEO_STORYBOARD_PATH = OUT_DIR / "d362_interface_visible_trace_replay_storyboard.png"
CAPTURE_PATHS = {
    "initial": OUT_DIR / "d362_initial_open_actual_physx_interface_primary.png",
    "start": OUT_DIR / "d362_open_precommand_actual_physx_interface.png",
    "contact": OUT_DIR / "d362_contact_confirmation_actual_physx_interface.png",
    "motion": OUT_DIR / "d362_motion_confirmation_actual_physx_interface.png",
    "final": OUT_DIR / "d362_final_actual_physx_interface.png",
}
OPPOSITE_CAPTURE_PATHS = {
    "initial": OUT_DIR / "d362_initial_open_actual_physx_interface_opposite.png",
    "start": OUT_DIR / "d362_open_precommand_actual_physx_interface_opposite.png",
    "contact": OUT_DIR / "d362_contact_confirmation_actual_physx_interface_opposite.png",
    "motion": OUT_DIR / "d362_motion_confirmation_actual_physx_interface_opposite.png",
    "final": OUT_DIR / "d362_final_actual_physx_interface_opposite.png",
}

_PHASE_SEQUENCE = 0
_CONTROLLED_STEPS = 0
_Q5_TARGET_UPDATE_COUNT = 0
_PREFIX_WRITER: Any = None
_PREFIX_LAST_STATE: dict[str, Any] | None = None
_PREFIX_HIGH_WATER = 0
_D361_ORIGINAL_VALIDATE_HEADER = d361._validate_header_payload
_D361_ORIGINAL_VALIDATE_OBSERVATION = d361._validate_observation_payload
D362_PREFIX_PROFILE = "D362_ACTUAL_PHYSX_DURABLE_PREFIX_V1"
D362_PREFIX_SUBJECT = "actual_d362_d360_inherited_physx_contact_motion_evidence"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _safe_json_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _rel(path),
        "exists": path.is_file(),
        "parse_pass": False,
        "parse_error": None,
        "payload": None,
    }
    if not result["exists"]:
        return result
    try:
        payload = _json(path)
        if not isinstance(payload, dict):
            raise TypeError("expected a top-level JSON object")
        result["payload"] = payload
        result["parse_pass"] = True
    except Exception as error:
        result["parse_error"] = f"{type(error).__name__}: {error}"
    return result


def _write_json_x(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if _json(path) != payload:
        raise RuntimeError(f"durable JSON reread mismatch: {path}")


def _marker(phase: str, event: str, details: dict[str, Any] | None = None) -> None:
    global _PHASE_SEQUENCE
    _PHASE_SEQUENCE += 1
    row = {
        "sequence": _PHASE_SEQUENCE,
        "utc": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "phase": phase,
        "event": event,
        "details": details or {},
    }
    PHASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_text(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()


def _git_head(ref: str = "HEAD") -> str:
    return _run_text(["git", "rev-parse", ref])


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.splitlines()


def _status_scope_ok(rows: list[str]) -> bool:
    allowed = (
        "START_HERE.md",
        "claudedocs/DECISIONS.md",
        "claudedocs/EXPERIMENT_LEDGER.md",
        "claudedocs/session_20260717_grasp_g0a_d362_",
        "sim_scripts/cyl34_top_view_d362_",
        "claudedocs/runtime_logs/grasp_track/g0a_d362/",
    )
    return all(any(row[3:].startswith(prefix) for prefix in allowed) for row in rows)


def _sidecar_hashes() -> dict[str, str]:
    root = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
    return {_rel(path): _sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _input_paths() -> list[Path]:
    return [
        SESSION_DOC,
        *D360_D361_EXPECTED_HASHES.keys(),
        D348_EVIDENCE,
        D334_SUMMARY,
        D354_MEASUREMENT,
        D354_ATTESTATION,
        D359_COMPLETION,
        D359_CLARIFICATION,
        VARIANT_ROBOT_USD,
        VARIANT_PHYSICS_USD,
        URDF_PATH,
        Path(d351.__file__).resolve(),
        Path(d351.d333.__file__).resolve(),
        Path(d361.__file__).resolve(),
    ]


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in _input_paths()}


def _output_file_inventory() -> list[str]:
    if not OUT_DIR.exists():
        return []
    return sorted(
        str(path.relative_to(OUT_DIR))
        for path in OUT_DIR.rglob("*")
        if path.is_file()
    )


def _inventory_hashes(names: list[str]) -> dict[str, str]:
    return {name: _sha(OUT_DIR / name) for name in names}


def _capture_inventory_names(worker: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for role in CAPTURE_PATHS:
        if worker.get("captures", {}).get(role) is not None:
            names.add(CAPTURE_PATHS[role].name)
            names.add(OPPOSITE_CAPTURE_PATHS[role].name)
    return names


def _expected_postworker_inventory(worker: dict[str, Any]) -> list[str]:
    names = {
        PREPARE_PATH.name,
        PREREG_PATH.name,
        INVOCATION_PATH.name,
        WORKER_PREFLIGHT_PATH.name,
        PHASE_PATH.name,
        WORKER_LOG_PATH.name,
        PREREQUISITE_PATH.name,
        TRACE_JSON_PATH.name,
        TRACE_CSV_PATH.name,
        WORKER_SUMMARY_PATH.name,
        RRD_PATH.name,
        RBL_PATH.name,
        RERUN_PNG_PATH.name,
        RERUN_VALIDATION_PATH.name,
        SHEET_PATH.name,
        PREFIX_PATH.name,
        PREFIX_AUDIT_PATH.name,
        SUPERVISOR_PREFIX_AUDIT_PATH.name,
        CONTACT_OVERFLOW_WARNING_AUDIT_PATH.name,
        VIDEO_PATH.name,
        VIDEO_REPORT_PATH.name,
        VIDEO_STORYBOARD_PATH.name,
        *_capture_inventory_names(worker),
    }
    return sorted(names)


def _core_postworker_inventory() -> set[str]:
    return {
        PREPARE_PATH.name,
        PREREG_PATH.name,
        INVOCATION_PATH.name,
        WORKER_PREFLIGHT_PATH.name,
        PHASE_PATH.name,
        WORKER_LOG_PATH.name,
        PREREQUISITE_PATH.name,
        TRACE_JSON_PATH.name,
        TRACE_CSV_PATH.name,
        WORKER_SUMMARY_PATH.name,
        PREFIX_PATH.name,
        PREFIX_AUDIT_PATH.name,
        SUPERVISOR_PREFIX_AUDIT_PATH.name,
        CONTACT_OVERFLOW_WARNING_AUDIT_PATH.name,
        VIDEO_REPORT_PATH.name,
    }


def _expected_precompletion_inventory(worker: dict[str, Any]) -> list[str]:
    return sorted(
        {
            *_expected_postworker_inventory(worker),
            SUPERVISOR_PATH.name,
            AUTOMATED_PATH.name,
            MANUAL_PATH.name,
        }
    )


def _phase_contract(worker: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(
        PHASE_PATH.read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise TypeError("phase row is not a JSON object")
            rows.append(parsed)
        except Exception as error:
            return {
                "checks": {"jsonl_parse": False},
                "pass": False,
                "row_count_before_error": len(rows),
                "parse_error": {
                    "line_index_zero_based": line_index,
                    "error": f"{type(error).__name__}: {error}",
                },
                "phase_sha256": _sha(PHASE_PATH),
            }

    def detail_pass(row: dict[str, Any]) -> bool:
        return row.get("details", {}).get("pass") is True

    def indices(phase: str, event: str, predicate: Any | None = None) -> list[int]:
        return [
            index
            for index, row in enumerate(rows)
            if row.get("phase") == phase
            and row.get("event") == event
            and (predicate is None or bool(predicate(row)))
        ]

    mandatory_specs: list[tuple[str, str, Any | None]] = [
        ("worker_preflight", "complete", detail_pass),
        ("AppLauncher", "start", None),
        ("AppLauncher", "complete", detail_pass),
        ("make_runtime_env", "start", None),
        ("make_runtime_env", "complete", detail_pass),
        ("reset", "start", None),
        ("reset", "complete", None),
        ("corrected_d348_audit", "start", None),
        ("corrected_d348_audit", "complete", detail_pass),
        ("live_64_plus_64_binding", "start", None),
        ("live_64_plus_64_binding", "complete", detail_pass),
        ("runtime_contact_capacity", "complete", detail_pass),
        ("durable_prefix", "initialized", detail_pass),
        (
            "viewport_capture",
            "complete",
            lambda row: row.get("details", {}).get("role") == "initial_primary",
        ),
        (
            "viewport_capture",
            "complete",
            lambda row: row.get("details", {}).get("role") == "initial_opposite",
        ),
        (
            "frozen_open_baseline",
            "progress",
            lambda row: row.get("details", {}).get("completed") == BASELINE_STEPS,
        ),
    ]
    if worker.get("baseline", {}).get("pass") is True:
        mandatory_specs.extend(
            [
                (
                    "viewport_capture",
                    "complete",
                    lambda row: row.get("details", {}).get("role")
                    == "start_primary",
                ),
                (
                    "viewport_capture",
                    "complete",
                    lambda row: row.get("details", {}).get("role")
                    == "start_opposite",
                ),
                ("q5_close_command", "target_updated_once", None),
                (
                    "q5_close_observation",
                    "progress",
                    lambda row: row.get("details", {}).get("completed")
                    == CLOSURE_MAX_STEPS,
                ),
                ("durable_prefix", "sealed", detail_pass),
            ]
        )
    else:
        mandatory_specs.extend(
            [
                ("durable_prefix", "sealed", detail_pass),
                (
                    "viewport_capture",
                    "complete",
                    lambda row: row.get("details", {}).get("role")
                    == "start_primary",
                ),
                (
                    "viewport_capture",
                    "complete",
                    lambda row: row.get("details", {}).get("role")
                    == "start_opposite",
                ),
                ("q5_close_observation", "not_run", None),
            ]
        )
    mandatory_specs.extend(
        [
            (
                "viewport_capture",
                "complete",
                lambda row: row.get("details", {}).get("role") == "final_primary",
            ),
            (
                "viewport_capture",
                "complete",
                lambda row: row.get("details", {}).get("role") == "final_opposite",
            ),
            ("durable_prefix", "audit_complete", detail_pass),
            ("rerun_recording", "start", None),
            ("rerun_recording", "rrd_finalized", None),
            ("rerun_validation", "start", None),
            ("rerun_validation", "complete", detail_pass),
            ("trace_replay_video", "start", None),
            ("trace_replay_video", "complete", detail_pass),
            ("worker_summary", "complete", detail_pass),
        ]
    )
    mandatory_indices: list[int] = []
    mandatory_exact_once = True
    for phase, event, predicate in mandatory_specs:
        found = indices(phase, event, predicate)
        mandatory_exact_once = mandatory_exact_once and len(found) == 1
        mandatory_indices.append(found[0] if len(found) == 1 else -1)
    expected_capture_roles = [
        f"{role}_{view}"
        for role in CAPTURE_PATHS
        if worker.get("captures", {}).get(role) is not None
        for view in ("primary", "opposite")
    ]
    capture_role_counts = {
        role: len(
            indices(
                "viewport_capture",
                "complete",
                lambda row, role=role: row.get("details", {}).get("role") == role,
            )
        )
        for role in expected_capture_roles
    }
    unexpected_capture_roles = [
        row.get("details", {}).get("role")
        for row in rows
        if row.get("phase") == "viewport_capture"
        and row.get("event") == "complete"
        and row.get("details", {}).get("role") not in expected_capture_roles
    ]
    capture_role_indices = {
        role: indices(
            "viewport_capture",
            "complete",
            lambda row, role=role: row.get("details", {}).get("role") == role,
        )
        for role in expected_capture_roles
    }
    all_capture_indices = [
        found[0] for found in capture_role_indices.values() if len(found) == 1
    ]
    rerun_start_indices = indices("rerun_recording", "start")
    prefix_init_indices = indices("durable_prefix", "initialized")
    prefix_seal_indices = indices("durable_prefix", "sealed")
    first_control_progress_indices = indices("frozen_open_baseline", "progress")
    final_capture_indices = [
        *indices(
            "viewport_capture",
            "complete",
            lambda row: row.get("details", {}).get("role") == "final_primary",
        ),
        *indices(
            "viewport_capture",
            "complete",
            lambda row: row.get("details", {}).get("role") == "final_opposite",
        ),
    ]
    checks = {
        "jsonl_parse": True,
        "nonempty": bool(rows),
        "sequence_exact": [row.get("sequence") for row in rows]
        == list(range(1, len(rows) + 1)),
        "monotonic_ns_nondecreasing": all(
            int(rows[index]["monotonic_ns"])
            <= int(rows[index + 1]["monotonic_ns"])
            for index in range(len(rows) - 1)
        ),
        "mandatory_exact_once": mandatory_exact_once,
        "mandatory_forward_order": mandatory_indices
        == sorted(mandatory_indices)
        and all(index >= 0 for index in mandatory_indices),
        "capture_roles_exact_once": all(
            count == 1 for count in capture_role_counts.values()
        )
        and not unexpected_capture_roles,
        "all_captures_before_rerun": len(rerun_start_indices) == 1
        and len(all_capture_indices) == len(expected_capture_roles)
        and all(index < rerun_start_indices[0] for index in all_capture_indices),
        "q5_command_count_exact": len(
            indices("q5_close_command", "target_updated_once")
        )
        == (1 if worker.get("baseline", {}).get("pass") is True else 0),
        "prefix_initialized_before_control_progress": len(prefix_init_indices) == 1
        and bool(first_control_progress_indices)
        and prefix_init_indices[0] < min(first_control_progress_indices),
        "prefix_sealed_before_final_captures": len(prefix_seal_indices) == 1
        and len(final_capture_indices) == 2
        and prefix_seal_indices[0] < min(final_capture_indices),
        "prefix_seal_reason_matches_horizon": len(prefix_seal_indices) == 1
        and prefix_seal_indices[0] >= 0
        and rows[prefix_seal_indices[0]].get("details", {}).get("reason")
        == (
            "full_500_step_horizon_complete"
            if worker.get("baseline", {}).get("pass") is True
            else "open_baseline_hard_gate_fail_stop"
        ),
        "last_marker_worker_summary": bool(rows)
        and rows[-1].get("phase") == "worker_summary"
        and rows[-1].get("event") == "complete",
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "row_count": len(rows),
        "mandatory_indices": mandatory_indices,
        "capture_role_counts": capture_role_counts,
        "capture_role_indices": capture_role_indices,
        "unexpected_capture_roles": unexpected_capture_roles,
        "phase_sha256": _sha(PHASE_PATH),
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    row: dict[str, Any] = {"command": command, "returncode": result.returncode, "stderr": result.stderr.strip()}
    if result.returncode == 0 and result.stdout.strip():
        fields = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        if len(fields) == 10:
            try:
                row.update(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "uuid": fields[2],
                        "compute_capability": fields[3],
                        "memory_total_mib": int(fields[4]),
                        "memory_used_mib": int(fields[5]),
                        "memory_free_mib": int(fields[6]),
                        "utilization_gpu_percent": int(fields[7]),
                        "temperature_c": int(fields[8]),
                        "pstate": fields[9],
                    }
                )
            except ValueError as error:
                row["parse_error"] = f"{type(error).__name__}: {error}"
    row["ram_total_bytes"] = int(psutil.virtual_memory().total)
    row["ram_available_bytes"] = int(psutil.virtual_memory().available)
    return row


def _parameter_contract() -> dict[str, Any]:
    measurement = _json(D354_MEASUREMENT)
    raw_bracket = measurement["raw_contact_order"]["first_contact_bracket"]
    live_bracket = measurement["live_contact_order"]["first_contact_bracket"]
    d359_completion = _json(D359_COMPLETION)
    d359_clarification = _json(D359_CLARIFICATION)
    d360_exception = _json(D360_EXCEPTION)
    d360_supervisor = _json(D360_SUPERVISOR)
    d360_worker_log = D360_WORKER_LOG.read_text(encoding="utf-8")
    d361_capacity = _json(D361_CAPACITY)
    d361_completion = _json(D361_COMPLETION)
    session_text = SESSION_DOC.read_text(encoding="utf-8")
    harness_sha256 = _sha(HARNESS)
    checks = {
        "seed_33201": SEED == 33201,
        "registered_base_git_exact": session_text.count(BASE_GIT) >= 1,
        "session_harness_sha_pin_exact_once": session_text.count(harness_sha256) == 1,
        "q_frozen_matches_d351_bits": Q_FROZEN_OPEN_F32.tobytes() == d351.Q_FROZEN_F32.tobytes(),
        "q5_open_exact": Q5_OPEN_F32.tobytes() == d351.Q5_OPEN_F32.tobytes(),
        "q5_closed_exact": Q5_CLOSED_F32.tobytes() == d351.Q5_CLOSED_F32.tobytes(),
        "object_position_exact": OBJECT_POS_F32.tobytes() == d351.OBJECT_POS_F32.tobytes(),
        "object_quaternion_exact": OBJECT_QUAT_F32.tobytes() == d351.OBJECT_QUAT_F32.tobytes(),
        "d354_raw_live_q5_brackets_identical": (
            np.float32(raw_bracket["q_clear_float32_rad"]).tobytes()
            == np.float32(live_bracket["q_clear_float32_rad"]).tobytes()
            and np.float32(raw_bracket["q_overlap_float32_rad"]).tobytes()
            == np.float32(live_bracket["q_overlap_float32_rad"]).tobytes()
        ),
        "d354_last_clear_exact": np.float32(raw_bracket["q_clear_float32_rad"]).tobytes()
        == D354_LAST_CLEAR_Q5_F32.tobytes(),
        "d354_first_overlap_exact": np.float32(raw_bracket["q_overlap_float32_rad"]).tobytes()
        == D354_FIRST_OVERLAP_Q5_F32.tobytes(),
        "cylinder_contract_exact": (
            CYLINDER_RADIUS_M == d351.d332.CYLINDER_RADIUS_M
            and CYLINDER_HEIGHT_M == d351.d332.CYLINDER_HEIGHT_M
            and OBJECT_MASS_KG == d351.d332.OBJECT_MASS_KG
            and STATIC_FRICTION == d351.d332.STATIC_FRICTION
            and DYNAMIC_FRICTION == d351.d332.DYNAMIC_FRICTION
        ),
        "physics_dt_exact": PHYSICS_DT_S == d351.d332.PHYSICS_DT_S,
        "thresholds_inherited_exact": (
            ROBOT_FORCE_EVENT_N == d351.d332.ROBOT_FORCE_EVENT_N
            and OBJECT_XY_EVENT_M == d351.d332.DISTURBANCE_XY_M
            and OBJECT_TILT_EVENT_DEG == d351.d332.DISTURBANCE_TILT_DEG
            and CONSECUTIVE_EVENT_STEPS == d351.d332.CONSECUTIVE_EVENT_STEPS
        ),
        "step_horizons_exact": BASELINE_STEPS == 200 and CLOSURE_MAX_STEPS == 300,
        "frozen_d360_d361_hashes_exact": all(
            path.is_file() and _sha(path) == digest
            for path, digest in D360_D361_EXPECTED_HASHES.items()
        ),
        "d360_failure_requires_exact_repair": (
            d360_exception.get("controlled_physics_steps") == 243
            and d360_exception.get("q5_target_update_count") == 1
            and "device-side assert" in str(d360_exception.get("error"))
            and d360_supervisor.get("worker_exit_code") == -9
            and d360_supervisor.get("watchdog_triggered") is False
            and d360_supervisor.get("worker_exception_exists") is True
            and d360_worker_log.count("Incomplete contact data") == 1
            and d360_worker_log.count("more contact data points") == 1
            and d360_worker_log.count("maxContactDataCount = 16") == 1
        ),
        "d361_capacity_exact_and_runtime_sufficiency_null": (
            d361_capacity.get("pass") is True
            and d361_capacity.get("formula", {}).get("direct_total_capacity")
            == REGISTERED_TOTAL_CONTACT_CAPACITY
            and d361_capacity.get("formula", {}).get("independent_sum_total_capacity")
            == REGISTERED_TOTAL_CONTACT_CAPACITY
            and d361_capacity.get("runtime_sufficiency") is None
        ),
        "d361_completion_semantics": (
            d361_completion.get("verdict")
            == "D361_CONTACT_CAPACITY_AND_PREFIX_TRACE_REPAIR_PASS_NO_PHYSICS"
            and d361_completion.get("pass") is True
            and d361_completion.get("registered_total_contact_capacity")
            == REGISTERED_TOTAL_CONTACT_CAPACITY
            and d361_completion.get("prefix_protocol_contract_pass") is True
            and d361_completion.get("failure_injection_passed") == 17
            and d361_completion.get("failure_injection_total") == 17
            and d361_completion.get("scientific_results", {}).get("g0a_pass") is False
        ),
        "d359_lineage_completion_semantics": d359_completion.get("artifact")
        == "D359_COMPLETION_SUMMARY_V1"
        and d359_completion.get("case") == "g0a_d359"
        and d359_completion.get("pass") is True
        and d359_completion.get("provenance_recovered") is True
        and d359_completion.get("verdict") == "D359_D351_HASH_PROVENANCE_RECOVERED"
        and d359_completion.get("g0a_pass") is False,
        "d359_forward_clarification_semantics": d359_clarification.get("artifact")
        == "D359_POSTCOMPLETION_LINEAGE_CLARIFICATION_V1"
        and d359_clarification.get("case") == "g0a_d359"
        and d359_clarification.get("pass") is True
        and d359_clarification.get("preserved_verdict")
        == "D359_D351_HASH_PROVENANCE_RECOVERED"
        and d359_clarification.get("d359_completion_sha256") == _sha(D359_COMPLETION)
        and d359_clarification.get("clarifications", {})
        .get("committed_d351_blob_sha256", {})
        .get("current_d351_file_byte_sha256")
        == _sha(Path(d351.__file__).resolve()),
        "camera_is_d351_verified_oblique": CAMERA_EYE == [0.49, -0.32, 0.28]
        and OPPOSITE_CAMERA_EYE == [0.49, 0.32, 0.28]
        and CAMERA_TARGET == [0.285, 0.0, 0.055],
        "exactly_two_new_variables": NEW_VARIABLES
        == [
            "runtime_contact_capacity_and_durable_prefix_integration",
            "interface_visible_trace_replay_video",
        ],
        "video_contract_exact": (
            VIDEO_SIZE == [1920, 1080]
            and VIDEO_FPS == 20
            and VIDEO_STRIDE == 2
            and VIDEO_EXPECTED_FRAME_COUNT == 250
            and "physics not recomputed" in VIDEO_REPLAY_AUTHORITY
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "baseline_semantics": "frozen q0-q4 plus q5 OPEN; never D333 HOME baseline",
        "closure_horizon_semantics": "300 steps is a contact-observation horizon, not q5=0 settle proof",
        "initial_proportional_error_torque_diagnostic_nm": float(Q5_OPEN_F32 * ACTUATOR_STIFFNESS),
        "effort_cap_nm": ACTUATOR_EFFORT_LIMIT_NM,
    }


def _offline_video_layout_contract() -> dict[str, Any]:
    """Pure CPU negative control: no Isaac import, no physics, no file output."""
    from PIL import ImageFont

    font17 = ImageFont.truetype(str(FONT_PATH), size=17)
    worst_case_lines = [
        "phase: frozen_open_baseline / step 299",
        "q5 a/t/v = -1.54130 / -1.54130 / -99.999",
        "F(N) L4/L5/grip = 999.999 / 999.999 / 999.999",
        "object XY/tiltΔ/zΔ = 999.999mm / 999.999° / -999.999mm",
        "gates: F≥0.1N×2; XY≥0.5mm or tilt≥1.0°×2",
    ]
    widths = [
        font17.getbbox(line)[2] - font17.getbbox(line)[0]
        for line in worst_case_lines
    ]
    synthetic_rows = [d361._synthetic_state_row(index) for index in range(3)]
    for index, row in enumerate(synthetic_rows):
        row["phase"] = "q5_close_observation"
        row["phase_step"] = index
        row["global_step"] = index + 201
    _annotate_event_masks(synthetic_rows)
    tetra_vertices = np.asarray(
        [
            [-0.010, -0.006, -0.006],
            [0.010, -0.006, -0.006],
            [0.0, 0.010, -0.006],
            [0.0, 0.0, 0.012],
        ],
        dtype=np.float64,
    )
    tetra_triangles = np.asarray(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
    )
    topology = {
        body: [{"_vertices": tetra_vertices, "_triangles": tetra_triangles}]
        for body in ("link5", "gripper_link")
    }
    rendered = _render_trace_replay_frame(topology, synthetic_rows, 2)
    pixels = np.asarray(rendered)
    checks = {
        "worst_case_metric_text_within_600px": max(widths) <= 600,
        "seven_metric_rows_fit_vertical_panel": 770 + 5 * 37 + 2 * 29
        <= 1038,
        "synthetic_frame_1920x1080": list(rendered.size) == VIDEO_SIZE,
        "synthetic_frame_nonblank": float(np.std(pixels)) > 5.0,
        "synthetic_frame_uint8_rgb": pixels.dtype == np.uint8
        and pixels.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3),
    }
    return {
        "authority": "offline synthetic display-layout negative control; no physics evidence",
        "worst_case_metric_lines": worst_case_lines,
        "text_width_pixels": widths,
        "synthetic_pixel_stddev": float(np.std(pixels)),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _frozen_d360_science_source_contract() -> dict[str, Any]:
    """Independent source-equality gate for every inherited science function."""
    import inspect

    from sim_scripts import (
        cyl34_top_view_d360_current_pose_bounded_physx_contact_motion as d360,
    )

    function_names = [
        "_actuator_contract",
        "_object_spawn_contract",
        "_q5_telemetry",
        "_state_row",
        "_physics_step_checked",
        "_set_closed_q5_target",
        "_instantaneous_event_masks",
        "_confirmed_robot_labels",
        "_motion_confirmed_pair",
        "_annotate_event_masks",
        "_qualifying_robot_point_contract",
        "_baseline_statistics",
        "_closure_statistics",
        "_q0_q4_drift_summary",
    ]
    rows: dict[str, Any] = {}
    for name in function_names:
        d360_source = inspect.getsource(getattr(d360, name)).replace(
            "D360", "DXXX"
        )
        d362_source = inspect.getsource(globals()[name]).replace("D362", "DXXX")
        rows[name] = {
            "normalized_source_equal": d360_source == d362_source,
            "d360_source_sha256": hashlib.sha256(
                d360_source.encode("utf-8")
            ).hexdigest(),
            "d362_source_sha256": hashlib.sha256(
                d362_source.encode("utf-8")
            ).hexdigest(),
        }
    checks = {
        "all_registered_functions_present": set(rows) == set(function_names),
        "all_normalized_sources_equal": all(
            row["normalized_source_equal"] for row in rows.values()
        ),
    }
    return {
        "authority": "inspect.getsource normalized only for D360/D362 verdict-label rename",
        "functions": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"forward-only D362 output already exists: {OUT_DIR}")
    status = _git_status()
    gpu = _gpu_snapshot()
    x_display = subprocess.run(["xdpyinfo", "-display", ":1"], text=True, capture_output=True, check=False)
    rerun_cli = subprocess.run([str(RERUN_CLI), "--version"], text=True, capture_output=True, check=False)
    try:
        import rerun as rr

        rerun_sdk_version = str(rr.__version__)
    except Exception as error:
        rerun_sdk_version = f"ERROR:{type(error).__name__}:{error}"
    try:
        import cv2
        import imageio_ffmpeg

        opencv_version = str(cv2.__version__)
        imageio_ffmpeg_version = str(imageio_ffmpeg.__version__)
        bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        ffmpeg_version = subprocess.run(
            [str(bundled_ffmpeg), "-version"], text=True, capture_output=True, check=False
        )
        ffmpeg_encoders = subprocess.run(
            [str(bundled_ffmpeg), "-hide_banner", "-encoders"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as error:
        opencv_version = f"ERROR:{type(error).__name__}:{error}"
        imageio_ffmpeg_version = opencv_version
        bundled_ffmpeg = Path("/nonexistent")
        ffmpeg_version = subprocess.CompletedProcess([], 1, "", opencv_version)
        ffmpeg_encoders = subprocess.CompletedProcess([], 1, "", opencv_version)
    parameters = _parameter_contract()
    video_layout = _offline_video_layout_contract()
    frozen_science_source = _frozen_d360_science_source_contract()
    checks = {
        "head_origin_master_and_registered_base_exact": _git_head()
        == _git_head("origin/master")
        == BASE_GIT,
        "git_scope_only_d362": _status_scope_ok(status),
        "registered_python_exact": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
        "numpy_pin_1p26p0": np.__version__ == "1.26.0",
        "psutil_pin_5p9p8": psutil.__version__ == "5.9.8",
        "display_exact_and_reachable": os.environ.get("DISPLAY") == ":1" and x_display.returncode == 0,
        "rerun_sdk_0p34p1": rerun_sdk_version == RERUN_VERSION,
        "rerun_cli_0p34p1": rerun_cli.returncode == 0 and RERUN_VERSION in rerun_cli.stdout,
        "bundled_imageio_ffmpeg_0p6p0_v7p0p2": imageio_ffmpeg_version == "0.6.0"
        and bundled_ffmpeg.is_file()
        and ffmpeg_version.returncode == 0
        and "ffmpeg version 7.0.2" in ffmpeg_version.stdout,
        "bundled_ffmpeg_libx264_encoder_available": ffmpeg_encoders.returncode
        == 0
        and "libx264" in ffmpeg_encoders.stdout,
        "opencv_decode_4p11p0": opencv_version == "4.11.0",
        "korean_font_exists": FONT_PATH.is_file(),
        "gpu_rtx4090_laptop_sm89": gpu.get("name") == "NVIDIA GeForce RTX 4090 Laptop GPU"
        and gpu.get("compute_capability") == "8.9",
        "gpu_free_at_least_8gib": int(gpu.get("memory_free_mib", 0)) >= MIN_GPU_FREE_MIB,
        "ram_available_at_least_10gib": int(gpu.get("ram_available_bytes", 0)) >= MIN_RAM_AVAILABLE_BYTES,
        "session_preregistered": SESSION_DOC.is_file()
        and "## 6. 실행·판정 순서" in SESSION_DOC.read_text(encoding="utf-8"),
        "all_inputs_exist": all(path.is_file() for path in _input_paths()),
        "parameter_contract": parameters["pass"],
        "offline_video_layout_negative_control": video_layout["pass"],
        "frozen_d360_science_source_contract": frozen_science_source["pass"],
        "d334_sidecar_present": bool(_sidecar_hashes()),
    }
    prepare = {
        "artifact": "D362_PREPARE_PREFLIGHT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "git": {"head": _git_head(), "origin_master": _git_head("origin/master"), "status": status},
        "environment": {
            "python": str(Path(sys.executable).resolve()),
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "display": os.environ.get("DISPLAY"),
            "rerun_sdk": rerun_sdk_version,
            "rerun_cli": rerun_cli.stdout.strip(),
            "opencv": opencv_version,
            "imageio_ffmpeg": imageio_ffmpeg_version,
            "bundled_ffmpeg": str(bundled_ffmpeg),
            "bundled_ffmpeg_version": ffmpeg_version.stdout.splitlines()[0]
            if ffmpeg_version.returncode == 0 and ffmpeg_version.stdout
            else ffmpeg_version.stderr,
            "bundled_ffmpeg_libx264_available": ffmpeg_encoders.returncode == 0
            and "libx264" in ffmpeg_encoders.stdout,
        },
        "gpu_and_ram": gpu,
        "input_hashes": _input_hashes(),
        "d334_sidecar_before": _sidecar_hashes(),
        "parameters": parameters,
        "offline_video_layout_negative_control": video_layout,
        "frozen_d360_science_source_contract": frozen_science_source,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not prepare["pass"]:
        print(json.dumps({"stage": "prepare", "pass": False, "checks": checks}, ensure_ascii=False))
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json_x(PREPARE_PATH, prepare)
    prereg = {
        "artifact": "D362_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "run_nonce": secrets.token_hex(16),
        "base_git": prepare["git"]["head"],
        "new_variables": NEW_VARIABLES,
        "baseline_steps": BASELINE_STEPS,
        "closure_max_steps": CLOSURE_MAX_STEPS,
        "physics_dt_s": PHYSICS_DT_S,
        "inherited_d360_contract": {
            "session_sha256": _sha(D360_SESSION),
            "harness_sha256": _sha(D360_HARNESS),
            "runtime_prerequisites_sha256": _sha(D360_PREREQUISITES),
            "worker_exception_sha256": _sha(D360_EXCEPTION),
            "supervisor_sha256": _sha(D360_SUPERVISOR),
            "worker_log_sha256": _sha(D360_WORKER_LOG),
            "registered_failure_warning": "Incomplete contact data ... maxContactDataCount = 16",
            "physical_question_changed": False,
        },
        "d361_runtime_integration": {
            "registered_total_contact_capacity": REGISTERED_TOTAL_CONTACT_CAPACITY,
            "capacity_budget_sha256": _sha(D361_CAPACITY),
            "prefix_protocol_sha256": _sha(D361_PROTOCOL),
            "completion_sha256": _sha(D361_COMPLETION),
            "step_begin_before_physics": True,
            "step_observation_fsync_exact_reread_before_optional_media": True,
            "resume_allowed": False,
            "overwrite_allowed": False,
        },
        "trace_replay_video": {
            "size": VIDEO_SIZE,
            "fps": VIDEO_FPS,
            "stride": VIDEO_STRIDE,
            "expected_frames_for_full_500_step_trace": VIDEO_EXPECTED_FRAME_COUNT,
            "codec": VIDEO_CODEC,
            "pixel_format": VIDEO_PIXEL_FORMAT,
            "authority": VIDEO_REPLAY_AUTHORITY,
            "decode_validation": "bundled FFmpeg 7.0.2 full decode plus OpenCV 4.11.0 first/middle/last frames; no standalone ffprobe or package install",
        },
        "camera": {
            "primary_eye": CAMERA_EYE,
            "opposite_eye": OPPOSITE_CAMERA_EYE,
            "target": CAMERA_TARGET,
            "source": "D351/D354 verified oblique plus symmetric anti-occlusion companion",
        },
        "thresholds": {
            "robot_force_event_n": ROBOT_FORCE_EVENT_N,
            "object_xy_event_m": OBJECT_XY_EVENT_M,
            "object_tilt_event_deg": OBJECT_TILT_EVENT_DEG,
            "consecutive_steps": CONSECUTIVE_EVENT_STEPS,
            "effort_saturation_abs_tolerance_nm_diagnostic": EFFORT_SATURATION_ABS_TOL_NM,
        },
        "actuator_frozen": {
            "stiffness": ACTUATOR_STIFFNESS,
            "damping": ACTUATOR_DAMPING,
            "effort_limit_sim_nm": ACTUATOR_EFFORT_LIMIT_NM,
            "velocity_limit_sim_rad_s": ACTUATOR_VELOCITY_LIMIT_RAD_S,
            "initial_error_times_kp_nm_diagnostic": float(Q5_OPEN_F32 * ACTUATOR_STIFFNESS),
            "authority": "ArticulationData.applied_torque; computed_torque and joint_effort_limits retained",
        },
        "input_hashes": prepare["input_hashes"],
        "d334_sidecar_before": prepare["d334_sidecar_before"],
        "harness_sha256": _sha(HARNESS),
        "prepare_sha256": _sha(PREPARE_PATH),
        "watchdogs_seconds": {"inactivity": INACTIVITY_WATCHDOG_S, "total": TOTAL_WATCHDOG_S},
        "single_invocation_no_retry": True,
        "result_boundaries": {
            "exact_manifold_or_face": None,
            "cap_rim_or_barrel_order": None,
            "force_closure": None,
            "grasp_success": None,
            "g0a_pass": False,
        },
        "prohibitions": [
            "no target/IK/path change",
            "no asset/decomposition/gate/material/mass/actuator/physics change",
            "no exact face/cap-rim/grasp/G0a classification",
            "no media-derived metric promoted over canonical JSON/prefix",
            "no retry or overwrite",
        ],
        "pass": True,
    }
    _write_json_x(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": True, "output": _rel(OUT_DIR)}, ensure_ascii=False))
    return 0


def _make_runtime_env(args: argparse.Namespace) -> Any:
    """D333 sole-support scene with only the D361 observability capacity changed."""
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import UsdPhysics
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnv

    class D362CapacityIntegratedSoleSupportCylinderEnv(RoArmCubeTap10cmEnv):
        def _setup_scene(self) -> None:
            super()._setup_scene()
            stage = self.scene.stage
            colliders_before = d351.d333._enumerate_collision_prims(
                stage, d351.d333.GROUND_ROOT_PATH
            )
            expected = stage.GetPrimAtPath(d351.d333.GROUND_COLLIDER_PATH)
            errors: list[str] = []
            if [row["path"] for row in colliders_before] != [
                d351.d333.GROUND_COLLIDER_PATH
            ]:
                errors.append(f"unexpected ground colliders: {colliders_before}")
            if not expected.IsValid() or not expected.HasAPI(UsdPhysics.CollisionAPI):
                errors.append(
                    f"missing expected ground collider: {d351.d333.GROUND_COLLIDER_PATH}"
                )
            else:
                UsdPhysics.CollisionAPI(expected).GetCollisionEnabledAttr().Set(False)
            colliders_after = d351.d333._enumerate_collision_prims(
                stage, d351.d333.GROUND_ROOT_PATH
            )
            if not (
                len(colliders_after) == 1
                and colliders_after[0]["path"] == d351.d333.GROUND_COLLIDER_PATH
                and not colliders_after[0]["collision_enabled"]
            ):
                errors.append(
                    f"ground collider did not disable pre-PLAY: {colliders_after}"
                )
            self._d333_preplay_stage_audit = {
                "expected_ground_collider_path": d351.d333.GROUND_COLLIDER_PATH,
                "ground_colliders_before_disable": colliders_before,
                "ground_colliders_after_disable": colliders_after,
                "errors": errors,
                "pass": not errors,
            }
            sensor_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Sponge",
                filter_prim_paths_expr=list(d351.d333.FILTER_PATHS),
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=REGISTERED_TOTAL_CONTACT_CAPACITY,
                force_threshold=0.0,
                debug_vis=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["d333_cylinder_contact"] = sensor
            self._d333_contact_sensor = sensor
            self._d332_contact_sensor = sensor

    env_cfg = d351.d332._configure_runtime_env(args)
    return D362CapacityIntegratedSoleSupportCylinderEnv(cfg=env_cfg)


def _configure_d361_runtime_prefix_schema() -> None:
    """Bind D361's tested wire/semantic implementation to accurate D362 labels."""
    d361.CASE = CASE
    d361.PREREG_PATH = PREREG_PATH
    d361.PREPARE_PATH = PREPARE_PATH
    d361.INVOCATION_PATH = INVOCATION_PATH
    d361.HARNESS = HARNESS
    d361.CAPACITY_PATH = D361_CAPACITY
    d361.PROTOCOL_PATH = D361_PROTOCOL

    def validate_header(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("D362 prefix header must be an object")
        if payload.get("profile") != D362_PREFIX_PROFILE:
            raise ValueError("D362 prefix profile mismatch")
        if payload.get("subject_kind") != D362_PREFIX_SUBJECT:
            raise ValueError("D362 prefix subject mismatch")
        lineage = payload.get("lineage")
        expected_lineage_keys = {
            "preregistration_sha256",
            "prepare_preflight_sha256",
            "actual_invocation_sha256",
            "harness_sha256",
            "inherited_d361_capacity_budget_sha256",
            "inherited_d361_protocol_contract_sha256",
        }
        if not isinstance(lineage, dict) or set(lineage) != expected_lineage_keys:
            raise ValueError("D362 prefix lineage keys are not exact")
        legacy = json.loads(json.dumps(payload))
        legacy["profile"] = d361.PREFIX_PROFILE
        legacy["subject_kind"] = (
            "future_actual_d360_inherited_state_requires_separate_approval"
        )
        legacy["lineage"] = {
            "preregistration_sha256": lineage["preregistration_sha256"],
            "prepare_preflight_sha256": lineage["prepare_preflight_sha256"],
            "offline_invocation_sha256": lineage["actual_invocation_sha256"],
            "harness_sha256": lineage["harness_sha256"],
            "capacity_budget_sha256": lineage[
                "inherited_d361_capacity_budget_sha256"
            ],
            "protocol_contract_sha256": lineage[
                "inherited_d361_protocol_contract_sha256"
            ],
        }
        _D361_ORIGINAL_VALIDATE_HEADER(legacy)

    def validate_observation(payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("subject_kind") != D362_PREFIX_SUBJECT:
            raise ValueError("D362 observation subject mismatch")
        legacy = json.loads(json.dumps(payload))
        legacy["subject_kind"] = (
            "future_actual_d360_inherited_state_requires_separate_approval"
        )
        _D361_ORIGINAL_VALIDATE_OBSERVATION(legacy)

    d361._validate_header_payload = validate_header
    d361._validate_observation_payload = validate_observation


def _runtime_capacity_contract(
    inner: Any,
    sensor_contract: dict[str, Any],
    topology_parts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    def enabled_collision_paths(root_path: str) -> list[str]:
        if not root_path:
            return []
        paths: list[str] = []
        for prim in Usd.PrimRange.Stage(
            inner.scene.stage, Usd.TraverseInstanceProxies()
        ):
            path = prim.GetPath().pathString
            if path != root_path and not path.startswith(root_path + "/"):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is None or bool(enabled):
                paths.append(path)
        return sorted(paths)

    sensor = inner._d333_contact_sensor
    contact_view = sensor.contact_physx_view
    probe_before = {
        "custom_step_counter": int(inner._sim_step_counter),
        "simulation_clock": d351._simulation_clock(inner),
    }
    detailed_buffers = contact_view.get_contact_data(dt=float(inner.physics_dt))
    detailed_buffer_shapes = [list(buffer.shape) for buffer in detailed_buffers]
    probe_after = {
        "custom_step_counter": int(inner._sim_step_counter),
        "simulation_clock": d351._simulation_clock(inner),
    }
    backend_capacity = int(contact_view.max_contact_data_count)
    expected_buffer_shapes = [
        [REGISTERED_TOTAL_CONTACT_CAPACITY, 1],
        [REGISTERED_TOTAL_CONTACT_CAPACITY, 3],
        [REGISTERED_TOTAL_CONTACT_CAPACITY, 3],
        [REGISTERED_TOTAL_CONTACT_CAPACITY, 1],
        [1, 4],
        [1, 4],
    ]
    resolved_paths = sensor_contract.get("resolved_filter_paths", [])
    resolved_by_label = {
        label: str(resolved_paths[index])
        for label, index in sensor_contract.get(
            "resolved_filter_index_by_label", {}
        ).items()
        if 0 <= int(index) < len(resolved_paths)
    }
    collider_paths_by_label = {
        "sensor_cylinder": enabled_collision_paths(
            "/World/envs/env_0/Sponge"
        ),
        **{
            label: enabled_collision_paths(resolved_by_label.get(label, ""))
            for label in d361.BODY_LABELS
        },
    }
    actual_shapes = {
        label: len(paths) for label, paths in collider_paths_by_label.items()
    }
    cfg_capacity = int(sensor.cfg.max_contact_data_count_per_prim)
    derived = (
        actual_shapes["sensor_cylinder"]
        * sum(actual_shapes[label] for label in d361.BODY_LABELS)
        * d361.PHYSX_CONTACTS_PER_GEOMETRY_PAIR
    )
    checks = {
        "sensor_cfg_33280": cfg_capacity == REGISTERED_TOTAL_CONTACT_CAPACITY,
        "backend_allocation_33280": backend_capacity
        == REGISTERED_TOTAL_CONTACT_CAPACITY,
        "backend_detailed_buffer_shapes_exact": detailed_buffer_shapes
        == expected_buffer_shapes,
        "paused_buffer_probe_added_zero_physics_steps": probe_before
        == probe_after,
        "sensor_body_one": actual_shapes["sensor_cylinder"] == 1,
        "environment_one": int(sensor.num_instances) == 1,
        "filter_count_four": int(sensor.contact_physx_view.filter_count) == 4,
        "resolved_filter_paths_exact": sensor_contract.get("resolved_filter_paths")
        == list(d361.RESOLVED_FILTER_PATHS),
        "resolved_filter_index_exact": sensor_contract.get(
            "resolved_filter_index_by_label"
        )
        == {label: index for index, label in enumerate(d361.BODY_LABELS)},
        "actual_shape_inventory_exact": actual_shapes
        == {
            "sensor_cylinder": 1,
            "support_table": 1,
            "link4": 1,
            "link5": 64,
            "gripper_link": 64,
        },
        "live_topology_matches_stage_link5_gripper": len(
            topology_parts["link5"]
        )
        == actual_shapes["link5"]
        and len(topology_parts["gripper_link"])
        == actual_shapes["gripper_link"],
        "derived_capacity_33280": derived == REGISTERED_TOTAL_CONTACT_CAPACITY,
        "d361_capacity_hash_exact": _sha(D361_CAPACITY)
        == D360_D361_EXPECTED_HASHES[D361_CAPACITY],
        "d361_protocol_hash_exact": _sha(D361_PROTOCOL)
        == D360_D361_EXPECTED_HASHES[D361_PROTOCOL],
    }
    return {
        "authority": "D361 installed PhysX-5.6.1 shape-pair envelope integrated at runtime",
        "sensor_cfg_max_contact_data_count_per_prim": cfg_capacity,
        "backend_max_contact_data_count": backend_capacity,
        "backend_detailed_contact_buffer_shapes": detailed_buffer_shapes,
        "expected_backend_detailed_contact_buffer_shapes": expected_buffer_shapes,
        "paused_buffer_probe": {"before": probe_before, "after": probe_after},
        "actual_shape_counts": actual_shapes,
        "enabled_collision_paths_by_label": collider_paths_by_label,
        "per_geometry_pair_contact_envelope": d361.PHYSX_CONTACTS_PER_GEOMETRY_PAIR,
        "derived_total_capacity": derived,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prefix_header() -> dict[str, Any]:
    return {
        "case": CASE,
        "scenario": "actual_frozen_d360_contract_d362_single_invocation",
        "profile": D362_PREFIX_PROFILE,
        "subject_kind": D362_PREFIX_SUBJECT,
        "registered_total_capacity": REGISTERED_TOTAL_CONTACT_CAPACITY,
        "sensor_body_names": list(d361.SENSOR_BODY_NAMES),
        "body_labels": list(d361.BODY_LABELS),
        "resolved_filter_paths": list(d361.RESOLVED_FILTER_PATHS),
        "filter_index_by_body": {
            label: index for index, label in enumerate(d361.BODY_LABELS)
        },
        "state_row_contract": {
            "authority": "D360 _state_row exact top-level field contract",
            "required_top_level_keys": list(d361.D360_STATE_ROW_KEYS),
        },
        "lineage": {
            "preregistration_sha256": _sha(PREREG_PATH),
            "prepare_preflight_sha256": _sha(PREPARE_PATH),
            "actual_invocation_sha256": _sha(INVOCATION_PATH),
            "harness_sha256": _sha(HARNESS),
            "inherited_d361_capacity_budget_sha256": _sha(D361_CAPACITY),
            "inherited_d361_protocol_contract_sha256": _sha(D361_PROTOCOL),
        },
        "execution_contract": {
            "legal_seals": {
                "full_500_step_horizon_complete": [500],
                "open_baseline_hard_gate_fail_stop": [200],
            }
        },
        "resume_allowed": False,
        "overwrite_allowed": False,
    }


def _pre_step_state(inner: Any, timeline: Any) -> dict[str, Any]:
    robot = inner._robot.data
    return {
        "custom_step_counter": int(inner._sim_step_counter),
        "simulation_clock": d351._simulation_clock(inner),
        "timeline_time_s": float(timeline.get_current_time()),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "joint_pos_float32_bits_sha256": hashlib.sha256(
            robot.joint_pos[0].detach().cpu().numpy().astype(np.float32).tobytes()
        ).hexdigest(),
        "joint_target_float32_bits_sha256": hashlib.sha256(
            robot.joint_pos_target[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .tobytes()
        ).hexdigest(),
        "object_pose_float32_bits_sha256": hashlib.sha256(
            np.concatenate(
                [
                    inner._sponge.data.root_pos_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    inner._sponge.data.root_quat_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                ]
            ).tobytes()
        ).hexdigest(),
    }


def _contact_capacity_diagnostic(inner: Any, filter_map: dict[str, int]) -> dict[str, Any]:
    global _PREFIX_HIGH_WATER
    sensor = inner._d333_contact_sensor
    _, _, _, _, raw_counts, raw_starts = sensor.contact_physx_view.get_contact_data(
        dt=float(inner.physics_dt)
    )
    counts = raw_counts.detach().cpu().numpy().reshape(-1).astype(np.int64)
    starts = raw_starts.detach().cpu().numpy().reshape(-1).astype(np.int64)
    if counts.size != 4 or starts.size != 4:
        raise RuntimeError(
            f"D362 raw contact count/start shape mismatch: {counts.shape}/{starts.shape}"
        )
    by_filter: dict[str, Any] = {}
    ranges: list[tuple[int, int, str]] = []
    for label in d361.BODY_LABELS:
        index = int(filter_map[label])
        count = int(counts[index])
        start = int(starts[index])
        end = start + count
        by_filter[label] = {
            "filter_index": index,
            "start_index": start,
            "count": count,
            "exclusive_end_index": end,
        }
        if count:
            ranges.append((start, end, label))
    count_sum = int(sum(row["count"] for row in by_filter.values()))
    max_end = int(max(row["exclusive_end_index"] for row in by_filter.values()))
    ranges.sort()
    nonoverlap = all(
        current[0] >= previous[1]
        for previous, current in zip(ranges, ranges[1:])
    )
    within = bool(
        nonoverlap
        and 0 <= count_sum <= REGISTERED_TOTAL_CONTACT_CAPACITY
        and all(
            0 <= row["start_index"] <= row["exclusive_end_index"]
            <= REGISTERED_TOTAL_CONTACT_CAPACITY
            for row in by_filter.values()
        )
    )
    _PREFIX_HIGH_WATER = max(_PREFIX_HIGH_WATER, count_sum)
    return {
        "registered_total_capacity": REGISTERED_TOTAL_CONTACT_CAPACITY,
        "reported_contact_point_count": count_sum,
        "remaining_capacity": REGISTERED_TOTAL_CONTACT_CAPACITY - count_sum,
        "high_water_mark": _PREFIX_HIGH_WATER,
        "by_filter": by_filter,
        "count_sum": count_sum,
        "max_exclusive_end_index": max_end,
        "all_ranges_within_capacity": within,
    }


def _prefix_observation_payload(
    row: dict[str, Any], diagnostic: dict[str, Any]
) -> dict[str, Any]:
    global _PREFIX_LAST_STATE
    identity = {key: row[key] for key in ("global_step", "phase", "phase_step")}
    payload = {
        "subject_kind": D362_PREFIX_SUBJECT,
        "step_identity": identity,
        "state_row": row,
        "contact_point_capacity_diagnostic": diagnostic,
        "event_projection": d361._event_projection(_PREFIX_LAST_STATE, row),
    }
    return payload


def _initialize_prefix_writer() -> None:
    global _PREFIX_WRITER
    if _PREFIX_WRITER is not None or PREFIX_PATH.exists():
        raise RuntimeError("D362 durable prefix would be initialized twice")
    _configure_d361_runtime_prefix_schema()
    _PREFIX_WRITER = d361.DurablePrefixWriter(PREFIX_PATH, _prefix_header())
    _marker(
        "durable_prefix",
        "initialized",
        {
            "path": _rel(PREFIX_PATH),
            "capacity": REGISTERED_TOTAL_CONTACT_CAPACITY,
            "header_sha256": _PREFIX_WRITER.receipts[0]["record_sha256"],
            "pass": True,
        },
    )


def _begin_prefix_step(inner: Any, timeline: Any, phase: str, phase_step: int) -> None:
    if _PREFIX_WRITER is None:
        raise RuntimeError("D362 durable prefix is not initialized")
    identity = {
        "global_step": int(inner._sim_step_counter) + 1,
        "phase": phase,
        "phase_step": int(phase_step),
    }
    _PREFIX_WRITER.begin_step(identity, _pre_step_state(inner, timeline))


def _observe_prefix_step(row: dict[str, Any], inner: Any, filter_map: dict[str, int]) -> None:
    global _PREFIX_LAST_STATE
    if _PREFIX_WRITER is None:
        raise RuntimeError("D362 durable prefix is not initialized")
    diagnostic = _contact_capacity_diagnostic(inner, filter_map)
    payload = _prefix_observation_payload(row, diagnostic)
    _PREFIX_WRITER.observe_step(payload)
    _PREFIX_LAST_STATE = json.loads(json.dumps(row))
    count = int(_PREFIX_WRITER.observation_count)
    if count == 1 or count % 10 == 0:
        _marker(
            "durable_prefix",
            "observation_durable",
            {
                "observation_count": count,
                "reported_contact_point_count": diagnostic[
                    "reported_contact_point_count"
                ],
                "high_water_mark": diagnostic["high_water_mark"],
            },
        )


def _seal_prefix(reason: str) -> None:
    if _PREFIX_WRITER is None:
        raise RuntimeError("D362 durable prefix is not initialized")
    receipt = _PREFIX_WRITER.seal(reason)
    _PREFIX_WRITER.close()
    _marker(
        "durable_prefix",
        "sealed",
        {
            "reason": reason,
            "observation_count": _PREFIX_WRITER.observation_count,
            "record_sha256": receipt["record_sha256"],
            "pass": True,
        },
    )


def _finalize_prefix_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit, observations = d361.verify_prefix(PREFIX_PATH)
    expected_states = [
        {key: row[key] for key in d361.D360_STATE_ROW_KEYS} for row in rows
    ]
    observed_states = [payload["state_row"] for payload in observations]
    first_difference = d361._first_difference(observed_states, expected_states)
    diagnostics = [
        payload["contact_point_capacity_diagnostic"] for payload in observations
    ]
    checks = {
        "wire_and_semantic_complete": audit.get("complete_pass") is True,
        "sealed": audit.get("sealed") is True,
        "no_trailing_bytes": audit.get("trailing_byte_count") == 0,
        "no_terminal_inflight_step": audit.get("terminal_inflight_step") is None,
        "observation_count_matches_trace": len(observations) == len(rows),
        "state_rows_reconcile_exact": first_difference is None,
        "all_contact_ranges_within_capacity": bool(diagnostics)
        and all(item["all_ranges_within_capacity"] for item in diagnostics),
        "high_water_below_or_equal_capacity": bool(diagnostics)
        and max(item["high_water_mark"] for item in diagnostics)
        <= REGISTERED_TOTAL_CONTACT_CAPACITY,
        "append_receipts_all_fsync_reread": _PREFIX_WRITER is not None
        and len(_PREFIX_WRITER.receipts) == 2 * len(rows) + 2
        and all(
            receipt.get("fsync_then_exact_reread_pass") is True
            for receipt in _PREFIX_WRITER.receipts
        ),
    }
    result = {
        "artifact": "D362_DURABLE_STEP_PREFIX_AUDIT_V1",
        "case": CASE,
        "prefix_path": _rel(PREFIX_PATH),
        "prefix_sha256": _sha(PREFIX_PATH),
        "inherited_protocol_sha256": _sha(D361_PROTOCOL),
        "audit": audit,
        "trace_state_reconciliation_first_difference": first_difference,
        "reported_contact_point_count_max": max(
            item["reported_contact_point_count"] for item in diagnostics
        )
        if diagnostics
        else None,
        "contact_point_high_water_mark": max(
            item["high_water_mark"] for item in diagnostics
        )
        if diagnostics
        else None,
        "append_receipt_count": len(_PREFIX_WRITER.receipts)
        if _PREFIX_WRITER is not None
        else None,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREFIX_AUDIT_PATH, result)
    _marker("durable_prefix", "audit_complete", {"pass": result["pass"]})
    return result


def _write_supervisor_prefix_recovery_audit(worker_exit_code: int) -> dict[str, Any]:
    """CPU/file-only recovery audit that also survives a hard worker exit."""
    if SUPERVISOR_PREFIX_AUDIT_PATH.exists():
        raise RuntimeError("D362 supervisor prefix recovery audit already exists")
    worker_summary_file = _safe_json_file(WORKER_SUMMARY_PATH)
    worker_prefix_audit_file = _safe_json_file(PREFIX_AUDIT_PATH)
    worker_summary_exists = bool(worker_summary_file["exists"])
    worker_prefix_audit_exists = bool(worker_prefix_audit_file["exists"])
    verification_error: str | None = None
    audit: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []
    if PREFIX_PATH.is_file():
        try:
            _configure_d361_runtime_prefix_schema()
            audit, observations = d361.verify_prefix(PREFIX_PATH)
        except Exception as error:
            verification_error = f"{type(error).__name__}: {error}"
    worker_prefix_audit = worker_prefix_audit_file["payload"]
    complete_success_path = bool(
        worker_exit_code == 0
        and worker_summary_file["parse_pass"] is True
        and worker_prefix_audit_file["parse_pass"] is True
    )
    checks = {
        "prefix_exists_if_worker_reached_summary": PREFIX_PATH.is_file()
        if worker_summary_exists
        else True,
        "worker_summary_json_parse_if_present": worker_summary_file[
            "parse_pass"
        ]
        if worker_summary_exists
        else True,
        "worker_prefix_audit_json_parse_if_present": worker_prefix_audit_file[
            "parse_pass"
        ]
        if worker_prefix_audit_exists
        else True,
        "independent_verifier_did_not_raise": verification_error is None
        if PREFIX_PATH.is_file()
        else True,
        "success_path_prefix_complete": audit.get("complete_pass") is True
        if complete_success_path and audit is not None
        else (not complete_success_path),
        "success_path_worker_audit_pass": worker_prefix_audit.get("pass") is True
        if complete_success_path and worker_prefix_audit is not None
        else (not complete_success_path),
        "success_path_prefix_hash_reconciles": bool(
            audit is not None
            and worker_prefix_audit is not None
            and audit.get("file_sha256")
            == worker_prefix_audit.get("prefix_sha256")
            == _sha(PREFIX_PATH)
        )
        if complete_success_path
        else True,
        "failure_path_recoverable_prefix_preserved": audit.get(
            "recoverable_prefix_pass"
        )
        is True
        if PREFIX_PATH.is_file() and not complete_success_path and audit is not None
        else True,
    }
    if audit is None:
        recovery_classification = (
            "PREFIX_ABSENT_BEFORE_INITIALIZATION"
            if not PREFIX_PATH.is_file()
            else "PREFIX_VERIFICATION_ERROR"
        )
    elif audit.get("complete_pass") is True:
        recovery_classification = "COMPLETE_SEALED_PREFIX"
    elif audit.get("recoverable_prefix_pass") is True:
        recovery_classification = "RECOVERABLE_UNSEALED_OR_PARTIAL_PREFIX"
    else:
        recovery_classification = "NO_RECOVERABLE_PREFIX"
    result = {
        "artifact": "D362_SUPERVISOR_PREFIX_RECOVERY_AUDIT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "worker_exit_code": worker_exit_code,
        "worker_summary_exists": worker_summary_exists,
        "worker_prefix_audit_exists": worker_prefix_audit_exists,
        "worker_summary_file": worker_summary_file,
        "worker_prefix_audit_file": worker_prefix_audit_file,
        "prefix_exists": PREFIX_PATH.is_file(),
        "prefix_path": _rel(PREFIX_PATH),
        "prefix_sha256": _sha(PREFIX_PATH) if PREFIX_PATH.is_file() else None,
        "independent_audit": audit,
        "observation_count": len(observations),
        "last_observed_step": observations[-1]["step_identity"]
        if observations
        else None,
        "verification_error": verification_error,
        "recovery_classification": recovery_classification,
        "worker_prefix_audit_sha256": _sha(PREFIX_AUDIT_PATH)
        if worker_prefix_audit_exists
        else None,
        "failure_prefix_audit_sha256": _sha(FAILURE_PREFIX_AUDIT_PATH)
        if FAILURE_PREFIX_AUDIT_PATH.is_file()
        else None,
        "checks": checks,
        "success_path_pass": complete_success_path and all(checks.values()),
        "recovery_audit_pass": all(checks.values()),
        "resume_allowed": False,
        "overwrite_allowed": False,
        "g0a_pass": False,
    }
    _write_json_x(SUPERVISOR_PREFIX_AUDIT_PATH, result)
    return result


def _write_contact_overflow_warning_audit() -> dict[str, Any]:
    """Audit the complete fsynced worker log for PhysX truncation warnings."""
    if CONTACT_OVERFLOW_WARNING_AUDIT_PATH.exists():
        raise RuntimeError("D362 contact overflow warning audit already exists")
    raw = WORKER_LOG_PATH.read_bytes()
    text_log = raw.decode("utf-8", errors="replace")
    warning_tokens = (
        "incomplete contact data",
        "more contact data points",
        "maxcontactdatacount",
    )
    matching_lines = [
        {"line_number_one_based": number, "text": line[:2000]}
        for number, line in enumerate(text_log.splitlines(), start=1)
        if any(token in line.lower() for token in warning_tokens)
    ]
    checks = {
        "worker_log_exists_and_nonempty": WORKER_LOG_PATH.is_file()
        and len(raw) > 0,
        "utf8_decoded_with_replacement_policy": True,
        "incomplete_contact_data_warning_count_zero": len(matching_lines) == 0,
    }
    result = {
        "artifact": "D362_CONTACT_OVERFLOW_WARNING_AUDIT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "worker_log_path": _rel(WORKER_LOG_PATH),
        "worker_log_sha256": hashlib.sha256(raw).hexdigest(),
        "worker_log_bytes": len(raw),
        "registered_warning_tokens_case_insensitive": list(warning_tokens),
        "matching_warning_line_count": len(matching_lines),
        "matching_warning_lines": matching_lines,
        "d361_runtime_sufficiency_component": (
            "no PhysX incomplete-contact-data warning in the complete worker log"
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(CONTACT_OVERFLOW_WARNING_AUDIT_PATH, result)
    return result


def _quat_tilt_deg(quat_wxyz: np.ndarray) -> float:
    rot = d351.d332._quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64))
    return math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))


def _body_pose(inner: Any, body: str) -> tuple[np.ndarray, np.ndarray]:
    return d351.d334._body_pose_w(inner, body)


def _runtime_snapshot(inner: Any, timeline: Any) -> dict[str, Any]:
    robot = inner._robot.data
    body_pos = robot.body_pos_w[0].detach().cpu().numpy().astype(np.float32)
    body_quat = robot.body_quat_w[0].detach().cpu().numpy().astype(np.float32)
    return {
        "counter": int(inner._sim_step_counter),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time": float(timeline.get_current_time()),
        "simulation_clock": d351._simulation_clock(inner),
        "joint_pos_bits": robot.joint_pos[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "joint_vel_bits": robot.joint_vel[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "body_pos_bits": body_pos.tobytes().hex(),
        "body_quat_bits": body_quat.tobytes().hex(),
        "object_pos_bits": inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_quat_bits": inner._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_lin_vel_bits": inner._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_ang_vel_bits": inner._sponge.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
    }


def _same_dynamic_snapshot(before: dict[str, Any], after: dict[str, Any]) -> bool:
    ignored = {"timeline_playing", "timeline_stopped"}
    return all(before[key] == after[key] for key in before if key not in ignored)


def _pause_timeline(inner: Any, timeline: Any) -> dict[str, Any]:
    before = _runtime_snapshot(inner, timeline)
    commit_count = 0
    if timeline.is_playing():
        timeline.pause()
        if timeline.is_playing() and not timeline.is_stopped():
            timeline.commit()
            commit_count = 1
    after = _runtime_snapshot(inner, timeline)
    checks = {
        "paused_not_stopped": not after["timeline_playing"] and not after["timeline_stopped"],
        "commit_at_most_once": commit_count in (0, 1),
        "counter_unchanged": before["counter"] == after["counter"],
        "clock_unchanged": before["simulation_clock"] == after["simulation_clock"],
        "state_bits_unchanged": all(
            before[key] == after[key]
            for key in before
            if key.endswith("_bits")
        ),
    }
    return {"before": before, "after": after, "commit_count": commit_count, "checks": checks, "pass": all(checks.values())}


def _resume_timeline(inner: Any, timeline: Any) -> dict[str, Any]:
    before = _runtime_snapshot(inner, timeline)
    if not timeline.is_playing():
        timeline.play()
        if not timeline.is_playing() and not timeline.is_stopped():
            timeline.commit()
    after = _runtime_snapshot(inner, timeline)
    checks = {
        "playing": after["timeline_playing"] and not after["timeline_stopped"],
        "counter_unchanged": before["counter"] == after["counter"],
        "clock_unchanged": before["simulation_clock"] == after["simulation_clock"],
        "state_bits_unchanged": all(
            before[key] == after[key] for key in before if key.endswith("_bits")
        ),
    }
    return {"before": before, "after": after, "checks": checks, "pass": all(checks.values())}


def _capture_viewport(
    path: Path,
    simulation_app: Any,
    inner: Any,
    timeline: Any,
    role: str,
    camera_eye: list[float],
) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utility

    pause = _pause_timeline(inner, timeline)
    if not pause["pass"]:
        raise RuntimeError(f"D362 pause before {role} capture failed")
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    inner.sim.set_camera_view(camera_eye, CAMERA_TARGET)
    before = _runtime_snapshot(inner, timeline)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None or not hasattr(viewport, "set_texture_resolution"):
        raise RuntimeError("D362 active viewport unavailable")
    viewport.set_texture_resolution(tuple(VIEWPORT_SIZE))
    for _ in range(12):
        simulation_app.update()
        if not _same_dynamic_snapshot(before, _runtime_snapshot(inner, timeline)):
            raise RuntimeError(f"D362 guarded render advanced physics before {role}")
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(capture.wait_for_result(completion_frames=5), run_until_complete=False)
    deadline = time.monotonic() + 30.0
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()
        if not _same_dynamic_snapshot(before, _runtime_snapshot(inner, timeline)):
            raise RuntimeError(f"D362 capture advanced physics at {role}")
    if not task.done():
        task.cancel()
        raise RuntimeError(f"D362 capture timeout: {role}")
    if not bool(task.result()):
        raise RuntimeError(f"D362 capture failed: {role}")
    for _ in range(3):
        simulation_app.update()
    after = _runtime_snapshot(inner, timeline)
    if not _same_dynamic_snapshot(before, after):
        raise RuntimeError(f"D362 post-capture state drift: {role}")
    _marker("viewport_capture", "complete", {"role": role, "path": _rel(path), "counter": after["counter"]})
    return {
        "role": role,
        "path": _rel(path),
        "camera_eye": camera_eye,
        "camera_target": CAMERA_TARGET,
        "pause_bridge": pause,
        "guard_before": before,
        "guard_after": after,
        "physics_unchanged": True,
    }


def _capture_pair(
    role: str, simulation_app: Any, inner: Any, timeline: Any
) -> dict[str, Any]:
    return {
        "primary": _capture_viewport(
            CAPTURE_PATHS[role], simulation_app, inner, timeline, f"{role}_primary", CAMERA_EYE
        ),
        "opposite": _capture_viewport(
            OPPOSITE_CAPTURE_PATHS[role],
            simulation_app,
            inner,
            timeline,
            f"{role}_opposite",
            OPPOSITE_CAMERA_EYE,
        ),
    }


def _actuator_contract(inner: Any) -> dict[str, Any]:
    configs = inner.cfg.robot.actuators
    rows: dict[str, Any] = {}
    for name in ("arm", "gripper"):
        cfg = configs[name]
        rows[name] = {
            "stiffness": float(cfg.stiffness),
            "damping": float(cfg.damping),
            "effort_limit_sim": float(cfg.effort_limit_sim),
            "velocity_limit_sim": float(cfg.velocity_limit_sim),
        }
    checks = {
        f"{name}_{field}": value == expected
        for name, row in rows.items()
        for field, value, expected in (
            ("stiffness", row["stiffness"], ACTUATOR_STIFFNESS),
            ("damping", row["damping"], ACTUATOR_DAMPING),
            ("effort_limit", row["effort_limit_sim"], ACTUATOR_EFFORT_LIMIT_NM),
            ("velocity_limit", row["velocity_limit_sim"], ACTUATOR_VELOCITY_LIMIT_RAD_S),
        )
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


def _object_spawn_contract(inner: Any) -> dict[str, Any]:
    spawn = inner.cfg.sponge.spawn
    rows = {
        "radius_m": float(spawn.radius),
        "height_m": float(spawn.height),
        "mass_kg": float(spawn.mass_props.mass),
        "static_friction": float(spawn.physics_material.static_friction),
        "dynamic_friction": float(spawn.physics_material.dynamic_friction),
        "restitution": float(spawn.physics_material.restitution),
        "physics_dt_s": float(inner.physics_dt),
    }
    checks = {
        "radius": rows["radius_m"] == CYLINDER_RADIUS_M,
        "height": rows["height_m"] == CYLINDER_HEIGHT_M,
        "mass": rows["mass_kg"] == OBJECT_MASS_KG,
        "static_friction": rows["static_friction"] == STATIC_FRICTION,
        "dynamic_friction": rows["dynamic_friction"] == DYNAMIC_FRICTION,
        "restitution": rows["restitution"] == RESTITUTION,
        "physics_dt": rows["physics_dt_s"] == PHYSICS_DT_S,
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


def _q5_telemetry(inner: Any, q5_index: int) -> dict[str, Any]:
    data = inner._robot.data

    def scalar(name: str) -> float | None:
        tensor = getattr(data, name, None)
        if tensor is None:
            return None
        return float(tensor[0, q5_index].detach().cpu().item())

    applied = scalar("applied_torque")
    computed = scalar("computed_torque")
    limit = scalar("joint_effort_limits")
    saturated = None
    ratio = None
    if applied is not None and limit is not None and limit > 0.0:
        ratio = abs(applied) / limit
        saturated = abs(applied) >= max(limit - EFFORT_SATURATION_ABS_TOL_NM, 0.0)
    return {
        "authority": "Isaac ArticulationData.applied_torque",
        "applied_torque_nm": applied,
        "computed_torque_nm": computed,
        "registered_effort_limit_nm": limit,
        "abs_applied_over_limit": ratio,
        "effort_saturated_diagnostic": saturated,
        "saturation_abs_tolerance_nm": EFFORT_SATURATION_ABS_TOL_NM,
    }


def _state_row(
    inner: Any,
    *,
    phase: str,
    phase_step: int,
    target: Any,
    reference_object_pos_w: np.ndarray,
    reference_object_quat_wxyz: np.ndarray,
    reference_root_pos_w: np.ndarray,
    reference_root_quat_wxyz: np.ndarray,
    q5_index: int,
    filter_map: dict[str, int],
) -> dict[str, Any]:
    robot = inner._robot.data
    joint_pos = robot.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    joint_vel = robot.joint_vel[0].detach().cpu().numpy().astype(np.float64)
    expected_target_f32 = target[0].detach().cpu().numpy().astype(np.float32)
    target_f32 = robot.joint_pos_target[0].detach().cpu().numpy().astype(np.float32)
    if target_f32.tobytes() != expected_target_f32.tobytes():
        raise RuntimeError("D362 articulation joint-position target buffer drifted")
    target_np = target_f32.astype(np.float64)
    object_pos = inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    object_quat = inner._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    object_lin_vel = inner._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
    object_ang_vel = inner._sponge.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
    disp = object_pos - reference_object_pos_w
    root_pos, root_quat = d351.d333._root_pose(inner)
    root_pos_drift = float(np.linalg.norm(root_pos - reference_root_pos_w))
    root_rot_drift = d351.d333._quat_delta_rad(reference_root_quat_wxyz, root_quat)
    object_rot = d351.d332._quat_wxyz_to_rot(object_quat)
    axis = object_rot[:, 2]
    reference_axis = d351.d332._quat_wxyz_to_rot(
        np.asarray(reference_object_quat_wxyz, dtype=np.float64)
    )[:, 2]
    axis_delta_deg = math.degrees(
        math.acos(float(np.clip(abs(float(np.dot(axis, reference_axis))), -1.0, 1.0)))
    )
    vertical_half_extent = (
        0.5 * CYLINDER_HEIGHT_M * abs(float(axis[2]))
        + CYLINDER_RADIUS_M * float(np.linalg.norm(axis[:2]))
    )
    object_bottom_z = float(object_pos[2] - inner.scene.env_origins[0, 2].detach().cpu().item()) - vertical_half_extent
    link5_pos, link5_quat = _body_pose(inner, "link5")
    grip_pos, grip_quat = _body_pose(inner, "gripper_link")
    contact = d351.d332._contact_state(inner._d333_contact_sensor, filter_map)
    q5_torque = _q5_telemetry(inner, q5_index)
    if any(q5_torque[key] is None for key in ("applied_torque_nm", "computed_torque_nm", "registered_effort_limit_nm")):
        raise RuntimeError(f"D362 q5 torque authority unavailable: {q5_torque}")
    clock = d351._simulation_clock(inner)
    row = {
        "global_step": int(inner._sim_step_counter),
        "phase": phase,
        "phase_step": int(phase_step),
        "physics_time_s": clock["current_time"],
        "simulation_clock": clock,
        "timeline_time_s": None,
        "actual_joint_rad": joint_pos.tolist(),
        "actual_joint_vel_rad_s": joint_vel.tolist(),
        "target_joint_rad": target_np.tolist(),
        "q5_actual_rad": float(joint_pos[q5_index]),
        "q5_velocity_rad_s": float(joint_vel[q5_index]),
        "q5_target_rad": float(target_np[q5_index]),
        "q5_error_rad": float(target_np[q5_index] - joint_pos[q5_index]),
        "q0_q4_actual_minus_frozen_rad": (joint_pos[:5] - Q_FROZEN_OPEN_F32[:5]).tolist(),
        "q0_q4_max_abs_drift_rad": float(np.max(np.abs(joint_pos[:5] - Q_FROZEN_OPEN_F32[:5]))),
        "q5_torque": q5_torque,
        "object_pos_w_m": object_pos.tolist(),
        "object_quat_wxyz": object_quat.tolist(),
        "object_lin_vel_w_mps": object_lin_vel.tolist(),
        "object_ang_vel_w_radps": object_ang_vel.tolist(),
        "object_disp_w_m": disp.tolist(),
        "object_disp_xy_mm": float(np.linalg.norm(disp[:2]) * 1000.0),
        "object_z_delta_mm": float(disp[2] * 1000.0),
        "object_tilt_deg": _quat_tilt_deg(object_quat),
        "object_tilt_delta_from_reference_deg": axis_delta_deg,
        "object_bottom_table_gap_mm": (object_bottom_z - d351.d332.TABLE_Z_M) * 1000.0,
        "robot_root_pos_w_m": root_pos.tolist(),
        "robot_root_quat_wxyz": root_quat.tolist(),
        "robot_root_position_drift_m": root_pos_drift,
        "robot_root_rotation_drift_rad": root_rot_drift,
        "link5_pos_w_m": np.asarray(link5_pos, dtype=np.float64).tolist(),
        "link5_quat_wxyz": np.asarray(link5_quat, dtype=np.float64).tolist(),
        "gripper_pos_w_m": np.asarray(grip_pos, dtype=np.float64).tolist(),
        "gripper_quat_wxyz": np.asarray(grip_quat, dtype=np.float64).tolist(),
        "contact": contact,
    }
    finite_values = [
        *joint_pos,
        *joint_vel,
        *object_pos,
        *object_quat,
        *object_lin_vel,
        *object_ang_vel,
        float(q5_torque["applied_torque_nm"]),
        float(q5_torque["computed_torque_nm"]),
        float(q5_torque["registered_effort_limit_nm"]),
    ]
    for label in ("support_table", "link4", "link5", "gripper_link"):
        finite_values.extend(contact["by_filter"][label]["force_w_n"])
        finite_values.append(contact["by_filter"][label]["force_norm_n"])
    row["finite"] = bool(np.isfinite(np.asarray(finite_values, dtype=np.float64)).all())
    return row


def _physics_step_checked(inner: Any, timeline: Any) -> dict[str, Any]:
    global _CONTROLLED_STEPS
    before_counter = int(inner._sim_step_counter)
    before_clock = d351._simulation_clock(inner)
    before_timeline = {
        "playing": bool(timeline.is_playing()),
        "stopped": bool(timeline.is_stopped()),
        "time_s": float(timeline.get_current_time()),
    }
    if not before_timeline["playing"] or before_timeline["stopped"]:
        raise RuntimeError("D362 timeline was not PLAY before controlled physics step")
    d351.d332._physics_step(inner)
    _CONTROLLED_STEPS += 1
    after_clock = d351._simulation_clock(inner)
    after_timeline = {
        "playing": bool(timeline.is_playing()),
        "stopped": bool(timeline.is_stopped()),
        "time_s": float(timeline.get_current_time()),
    }
    if int(inner._sim_step_counter) != before_counter + 1:
        raise RuntimeError("D362 controlled-step counter did not increment by one")
    if before_clock["current_time_step_index"] is not None and after_clock["current_time_step_index"] != before_clock["current_time_step_index"] + 1:
        raise RuntimeError("D362 SimulationContext step index did not increment by one")
    if before_clock["current_time"] is not None and not math.isclose(
        float(after_clock["current_time"] - before_clock["current_time"]), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("D362 SimulationContext time did not increment by 0.005 s")
    if not after_timeline["playing"] or after_timeline["stopped"]:
        raise RuntimeError("D362 timeline left PLAY during controlled physics step")
    if after_timeline["time_s"] < before_timeline["time_s"]:
        raise RuntimeError("D362 timeline time moved backward during controlled physics step")
    return {
        "before": before_timeline,
        "after": after_timeline,
        "delta_s_diagnostic": after_timeline["time_s"] - before_timeline["time_s"],
        "simulation_context_is_step_time_authority": True,
    }


def _set_closed_q5_target(inner: Any, open_target: Any, q5_index: int) -> Any:
    global _Q5_TARGET_UPDATE_COUNT
    if _Q5_TARGET_UPDATE_COUNT != 0:
        raise RuntimeError("D362 q5 target update would occur more than once")
    close_target = open_target.detach().clone()
    before_q0_q4 = close_target[:, :5].detach().cpu().numpy().astype(np.float32).tobytes()
    close_target[:, q5_index] = float(Q5_CLOSED_F32)
    after_q0_q4 = close_target[:, :5].detach().cpu().numpy().astype(np.float32).tobytes()
    if before_q0_q4 != after_q0_q4:
        raise RuntimeError("D362 q0-q4 target mutated while closing q5")
    env_ids = __import__("torch").arange(inner.num_envs, device=inner.device, dtype=__import__("torch").long)
    inner._robot.set_joint_position_target(
        close_target[:, q5_index : q5_index + 1],
        joint_ids=[q5_index],
        env_ids=env_ids,
    )
    inner.robot_dof_targets[env_ids] = close_target
    inner._external_joint_targets_override = close_target.detach().clone()
    _Q5_TARGET_UPDATE_COUNT += 1
    _marker(
        "q5_close_command",
        "target_updated_once",
        {
            "q5_target_rad": float(Q5_CLOSED_F32),
            "q0_q4_target_bits_unchanged": True,
            "initial_error_times_kp_nm_diagnostic": float(Q5_OPEN_F32 * ACTUATOR_STIFFNESS),
            "effort_limit_nm": ACTUATOR_EFFORT_LIMIT_NM,
        },
    )
    return close_target


def _first_consecutive(rows: list[dict[str, Any]], predicate: Any) -> int:
    run = 0
    for index, row in enumerate(rows):
        run = run + 1 if bool(predicate(row)) else 0
        if run >= CONSECUTIVE_EVENT_STEPS:
            return index - CONSECUTIVE_EVENT_STEPS + 1
    return -1


def _instantaneous_event_masks(row: dict[str, Any]) -> dict[str, Any]:
    body = {
        label: float(row["contact"]["by_filter"][label]["force_norm_n"])
        >= ROBOT_FORCE_EVENT_N
        for label in ("link4", "link5", "gripper_link")
    }
    motion = (
        float(row["object_disp_xy_mm"]) >= OBJECT_XY_EVENT_M * 1000.0
        or float(row["object_tilt_delta_from_reference_deg"])
        >= OBJECT_TILT_EVENT_DEG
    )
    return {"body": body, "any_robot_body": any(body.values()), "motion": motion}


def _confirmed_robot_labels(pair: list[dict[str, Any]]) -> list[str]:
    if (
        len(pair) != CONSECUTIVE_EVENT_STEPS
        or len({row.get("phase") for row in pair}) != 1
    ):
        return []
    masks = [_instantaneous_event_masks(row) for row in pair]
    return [
        label
        for label in ("link4", "link5", "gripper_link")
        if all(mask["body"][label] for mask in masks)
    ]


def _motion_confirmed_pair(pair: list[dict[str, Any]]) -> bool:
    return (
        len(pair) == CONSECUTIVE_EVENT_STEPS
        and len({row.get("phase") for row in pair}) == 1
        and all(_instantaneous_event_masks(row)["motion"] for row in pair)
    )


def _annotate_event_masks(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        instant = _instantaneous_event_masks(row)
        row["event_masks"] = {
            "instantaneous_body_force_ge_0p1n": instant["body"],
            "instantaneous_any_robot_body_force_ge_0p1n": instant[
                "any_robot_body"
            ],
            "instantaneous_object_motion_threshold": instant["motion"],
            "two_step_confirmation_end_by_filter": {
                label: False for label in ("link4", "link5", "gripper_link")
            },
            "two_step_confirmed_onset_by_filter": {
                label: False for label in ("link4", "link5", "gripper_link")
            },
            "two_step_any_robot_confirmation_end": False,
            "two_step_motion_confirmation_end": False,
            "two_step_motion_confirmed_onset": False,
        }
    for index in range(1, len(rows)):
        labels = _confirmed_robot_labels(rows[index - 1 : index + 1])
        for label in labels:
            rows[index]["event_masks"]["two_step_confirmation_end_by_filter"][
                label
            ] = True
            previous_was_run = (
                index >= 2
                and rows[index - 2].get("phase") == rows[index - 1].get("phase")
                and _instantaneous_event_masks(rows[index - 2])["body"][label]
            )
            if not previous_was_run:
                rows[index - 1]["event_masks"]["two_step_confirmed_onset_by_filter"][
                    label
                ] = True
        rows[index]["event_masks"]["two_step_any_robot_confirmation_end"] = bool(
            labels
        )
        if _motion_confirmed_pair(rows[index - 1 : index + 1]):
            rows[index]["event_masks"]["two_step_motion_confirmation_end"] = True
            previous_was_motion = (
                index >= 2
                and rows[index - 2].get("phase") == rows[index - 1].get("phase")
                and _instantaneous_event_masks(rows[index - 2])["motion"]
            )
            if not previous_was_motion:
                rows[index - 1]["event_masks"][
                    "two_step_motion_confirmed_onset"
                ] = True


def _qualifying_robot_point_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_filter: dict[str, Any] = {}
    required: list[bool] = []
    for label in ("link4", "link5", "gripper_link"):
        onsets = [
            index
            for index, row in enumerate(rows)
            if row["event_masks"]["two_step_confirmed_onset_by_filter"][label]
        ]
        event_checks = []
        for onset in onsets:
            points = [
                rows[index]["contact"]["by_filter"][label]["contact_point_w_m"]
                for index in (onset, onset + 1)
            ]
            finite = all(
                point is not None and bool(np.isfinite(np.asarray(point)).all())
                for point in points
            )
            required.append(finite)
            event_checks.append(
                {
                    "onset_global_row": onset,
                    "phase": rows[onset]["phase"],
                    "onset_phase_step": rows[onset]["phase_step"],
                    "two_event_points_finite": finite,
                }
            )
        by_filter[label] = {
            "qualifying_onset_global_row": onsets[0] if onsets else None,
            "qualifying_onset_global_rows": onsets,
            "event_point_checks": event_checks,
            "two_event_points_finite": (
                all(item["two_event_points_finite"] for item in event_checks)
                if event_checks
                else None
            ),
        }
    return {
        "by_filter": by_filter,
        "all_qualifying_robot_event_points_finite": all(required),
        "qualifying_event_count": len(required),
    }


def _baseline_statistics(
    rows: list[dict[str, Any]], *, stage_contract_pass: bool, sensor_contract_pass: bool
) -> dict[str, Any]:
    tail = rows[-50:]
    support_fz = [float(row["contact"]["by_filter"]["support_table"]["force_w_n"][2]) for row in tail]
    robot_max = {
        label: max(float(row["contact"]["by_filter"][label]["force_norm_n"]) for row in rows)
        for label in ("link4", "link5", "gripper_link")
    }
    event_by_filter = {
        label: _first_consecutive(
            rows,
            lambda row, label=label: float(
                row["contact"]["by_filter"][label]["force_norm_n"]
            )
            >= ROBOT_FORCE_EVENT_N,
        )
        for label in ("link4", "link5", "gripper_link")
    }
    first_motion = _first_consecutive(
        rows,
        lambda row: float(row["object_disp_xy_mm"])
        >= OBJECT_XY_EVENT_M * 1000.0
        or float(row["object_tilt_delta_from_reference_deg"])
        >= OBJECT_TILT_EVENT_DEG,
    )
    checks = {
        "stage_and_sensor_contracts_pass": stage_contract_pass and sensor_contract_pass,
        "exact_200_steps": len(rows) == BASELINE_STEPS,
        "frozen_open_target_every_step": all(
            np.asarray(row["target_joint_rad"], dtype=np.float32).tobytes() == Q_FROZEN_OPEN_F32.tobytes()
            for row in rows
        ),
        "support_table_fz_last50_median_gt_1n": float(np.median(support_fz)) > SUPPORT_POSITIVE_CONTROL_N,
        "first_step_abs_z_delta_le_0p5mm": abs(float(rows[0]["object_z_delta_mm"])) <= 0.5,
        "last50_bottom_table_gap_abs_max_le_0p5mm": max(
            abs(float(row["object_bottom_table_gap_mm"])) for row in tail
        ) <= 0.5,
        "robot_filters_max_lt_0p1n": max(robot_max.values()) < ROBOT_FORCE_EVENT_N,
        "robot_root_position_drift_le_1e_6m": max(
            float(row["robot_root_position_drift_m"]) for row in rows
        ) <= d351.d333.ROOT_POSITION_DRIFT_TOL_M,
        "robot_root_rotation_drift_le_1e_6rad": max(
            float(row["robot_root_rotation_drift_rad"]) for row in rows
        ) <= d351.d333.ROOT_ROTATION_DRIFT_TOL_RAD,
        "object_xy_lt_0p5mm": max(float(row["object_disp_xy_mm"]) for row in rows) < OBJECT_XY_EVENT_M * 1000.0,
        "object_tilt_lt_1deg": max(float(row["object_tilt_deg"]) for row in rows) < OBJECT_TILT_EVENT_DEG,
        "all_finite": all(row["finite"] for row in rows),
        "q5_effort_authority_every_step": all(row["q5_torque"]["applied_torque_nm"] is not None for row in rows),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "support_table_fz_last50_median_n": float(np.median(support_fz)),
        "robot_filter_max_n": robot_max,
        "first_contact_step_by_filter": event_by_filter,
        "first_object_motion_step": first_motion,
        "precommand_robot_contact_confound": any(
            step >= 0 for step in event_by_filter.values()
        ),
        "precommand_object_motion_confound": first_motion >= 0,
        "max_object_disp_xy_mm": max(float(row["object_disp_xy_mm"]) for row in rows),
        "max_object_tilt_deg": max(float(row["object_tilt_deg"]) for row in rows),
        "first_step_object_z_delta_mm": float(rows[0]["object_z_delta_mm"]),
        "last50_bottom_table_gap_abs_max_mm": max(
            abs(float(row["object_bottom_table_gap_mm"])) for row in tail
        ),
        "max_robot_root_position_drift_m": max(float(row["robot_root_position_drift_m"]) for row in rows),
        "max_robot_root_rotation_drift_rad": max(float(row["robot_root_rotation_drift_rad"]) for row in rows),
        "q5_actual_start_rad": float(rows[0]["q5_actual_rad"]),
        "q5_actual_final_rad": float(rows[-1]["q5_actual_rad"]),
    }


def _closure_statistics(
    rows: list[dict[str, Any]],
    baseline_pass: bool,
    *,
    baseline_end_q5_actual: float,
    baseline_robot_contact: bool = False,
    baseline_precommand_motion: bool = False,
) -> dict[str, Any]:
    event_by_filter: dict[str, int] = {}
    for label in ("link4", "link5", "gripper_link"):
        event_by_filter[label] = _first_consecutive(
            rows,
            lambda row, label=label: float(row["contact"]["by_filter"][label]["force_norm_n"])
            >= ROBOT_FORCE_EVENT_N,
        )
    moving_contact = event_by_filter["gripper_link"]
    fixed_contact = event_by_filter["link5"]
    jaw_steps = [
        (label, event_by_filter[label])
        for label in ("link5", "gripper_link")
        if event_by_filter[label] >= 0
    ]
    first_jaw_label, first_jaw_step = min(
        jaw_steps, key=lambda item: item[1], default=(None, -1)
    )
    if (
        moving_contact >= 0
        and fixed_contact >= 0
        and moving_contact == fixed_contact
    ):
        first_jaw_label = "link5+gripper_link_simultaneous"
    motion = _first_consecutive(
        rows,
        lambda row: float(row["object_disp_xy_mm"]) >= OBJECT_XY_EVENT_M * 1000.0
        or float(row["object_tilt_delta_from_reference_deg"]) >= OBJECT_TILT_EVENT_DEG,
    )
    q5_response_steps = [
        index
        for index, row in enumerate(rows)
        if float(row["q5_actual_rad"]) < float(baseline_end_q5_actual)
    ]
    q5_response = bool(q5_response_steps)
    bracket_reached = any(float(row["q5_actual_rad"]) <= float(D354_FIRST_OVERLAP_Q5_F32) for row in rows)
    full_horizon = len(rows) == CLOSURE_MAX_STEPS
    all_finite = all(row["finite"] for row in rows)
    max_root_position_drift_m = max(
        (float(row["robot_root_position_drift_m"]) for row in rows), default=None
    )
    max_root_rotation_drift_rad = max(
        (float(row["robot_root_rotation_drift_rad"]) for row in rows), default=None
    )
    fixed_root_contract = bool(
        max_root_position_drift_m is not None
        and max_root_rotation_drift_rad is not None
        and max_root_position_drift_m <= d351.d333.ROOT_POSITION_DRIFT_TOL_M
        and max_root_rotation_drift_rad <= d351.d333.ROOT_ROTATION_DRIFT_TOL_RAD
    )
    contact_point_finite_by_filter: dict[str, bool | None] = {}
    for label, event_index in event_by_filter.items():
        if event_index < 0:
            contact_point_finite_by_filter[label] = None
            continue
        points = [
            rows[index]["contact"]["by_filter"][label]["contact_point_w_m"]
            for index in (event_index, min(event_index + 1, len(rows) - 1))
        ]
        contact_point_finite_by_filter[label] = all(
            point is not None and bool(np.isfinite(np.asarray(point)).all())
            for point in points
        )
    contact_point_finite = False
    contact_label = None
    if moving_contact >= 0:
        contact_label = "gripper_link"
        contact_point_finite = bool(contact_point_finite_by_filter["gripper_link"])
    # The force sensor is the physical body-contact authority.  A missing
    # aggregate point is an observability defect, never a reason to erase a
    # force-positive physical witness.
    positive_contact = bool(baseline_pass and q5_response and moving_contact >= 0)
    other_steps = [step for step in (event_by_filter["link4"], fixed_contact) if step >= 0]
    first_other = min(other_steps, default=-1)
    if baseline_robot_contact or baseline_precommand_motion:
        verdict = "D362_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif (
        not baseline_pass
        or not all_finite
        or not full_horizon
        or not q5_response
        or not fixed_root_contract
    ):
        verdict = "D362_CONTROL_HORIZON_OR_BASELINE_FAIL_STOP"
    elif first_other >= 0 and (moving_contact < 0 or first_other <= moving_contact):
        verdict = "D362_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif motion >= 0 and (moving_contact < 0 or motion < moving_contact):
        verdict = "D362_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif positive_contact and motion >= moving_contact:
        verdict = "D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED"
    elif positive_contact and motion < 0:
        verdict = "D362_MOVING_JAW_CONTACT_WITHOUT_THRESHOLD_OBJECT_MOTION"
    elif moving_contact < 0 and bracket_reached:
        verdict = "D362_NO_POSITIVE_CONTACT_WITNESS_UNRESOLVED"
    else:
        verdict = "D362_CONTROL_HORIZON_OR_BASELINE_FAIL_STOP"
    saturated_rows = [row for row in rows if row["q5_torque"]["effort_saturated_diagnostic"] is True]
    peak_by_filter: dict[str, dict[str, Any]] = {}
    for label in ("link4", "link5", "gripper_link"):
        if rows:
            index = max(
                range(len(rows)),
                key=lambda i, label=label: float(
                    rows[i]["contact"]["by_filter"][label]["force_norm_n"]
                ),
            )
            peak_by_filter[label] = {
                "phase_step": index,
                "force_norm_n": float(
                    rows[index]["contact"]["by_filter"][label]["force_norm_n"]
                ),
            }
        else:
            peak_by_filter[label] = {"phase_step": None, "force_norm_n": None}
    peak_label, peak_force, peak_index = None, -math.inf, -1
    for index, row in enumerate(rows):
        for label in ("link4", "link5", "gripper_link"):
            force = float(row["contact"]["by_filter"][label]["force_norm_n"])
            if force > peak_force:
                peak_label, peak_force, peak_index = label, force, index
    return {
        "verdict": verdict,
        "physics_steps": len(rows),
        "full_300_step_observation_horizon": full_horizon,
        "fixed_base_root_contract_pass": fixed_root_contract,
        "max_robot_root_position_drift_m": max_root_position_drift_m,
        "max_robot_root_rotation_drift_rad": max_root_rotation_drift_rad,
        "open_baseline_robot_contact_confound": baseline_robot_contact,
        "open_baseline_precommand_motion_confound": baseline_precommand_motion,
        "q5_closed_direction_response": q5_response,
        "q5_actual_precommand_rad": float(baseline_end_q5_actual),
        "first_q5_closed_direction_response_step": q5_response_steps[0]
        if q5_response_steps
        else -1,
        "minimum_q5_actual_rad": min(
            (float(row["q5_actual_rad"]) for row in rows), default=None
        ),
        "d354_static_first_overlap_q5_reached_diagnostic": bracket_reached,
        "static_bracket_is_not_positive_contact_requirement": True,
        "first_contact_step_by_filter": event_by_filter,
        "first_jaw_contact_step": first_jaw_step,
        "first_moving_gripper_contact_step": moving_contact,
        "first_fixed_link5_contact_step": fixed_contact,
        "first_other_robot_body_contact_step": first_other,
        "first_jaw_contact_label": first_jaw_label,
        "moving_contact_point_label": contact_label,
        "jaw_contact_point_finite": contact_point_finite,
        "qualifying_contact_point_finite_by_filter": contact_point_finite_by_filter,
        "positive_body_level_contact_supported": positive_contact if moving_contact >= 0 else None,
        "moving_gripper_link_contact_supported": positive_contact if moving_contact >= 0 else None,
        "first_object_motion_step": motion,
        "object_motion_after_contact_supported": (
            True if positive_contact and motion >= moving_contact else False if positive_contact else None
        ),
        "peak_robot_force": {
            "phase_step": peak_index,
            "label": peak_label,
            "force_norm_n": None if peak_index < 0 else peak_force,
        },
        "peak_force_by_filter": peak_by_filter,
        "max_object_disp_xy_mm": max((float(row["object_disp_xy_mm"]) for row in rows), default=None),
        "max_object_tilt_deg": max(
            (float(row["object_tilt_delta_from_reference_deg"]) for row in rows),
            default=None,
        ),
        "max_object_absolute_tilt_deg": max(
            (float(row["object_tilt_deg"]) for row in rows), default=None
        ),
        "final_q5_actual_rad": float(rows[-1]["q5_actual_rad"]) if rows else None,
        "q5_applied_effort_saturation_fraction": len(saturated_rows) / len(rows) if rows else None,
        "q5_applied_effort_saturated_any": bool(saturated_rows),
        "body_attribution_semantics": "gripper_link is moving jaw authority; link5 is fixed-jaw/other-body evidence only",
        "zero_force_semantics": "absence of a positive filtered event does not prove no contact",
        "exact_face_or_manifold": None,
        "cap_rim_or_barrel_order": None,
        "force_closure_or_grasp": None,
        "g0a_pass": False,
    }


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "global_step": row["global_step"],
        "phase": row["phase"],
        "phase_step": row["phase_step"],
        "physics_time_s": row["physics_time_s"],
        "timeline_time_s": row["timeline_time_s"],
        "timeline_delta_s_diagnostic": row["timeline_step_contract"][
            "delta_s_diagnostic"
        ],
        "q5_actual_rad": row["q5_actual_rad"],
        "q5_velocity_rad_s": row["q5_velocity_rad_s"],
        "q5_target_rad": row["q5_target_rad"],
        "q5_error_rad": row["q5_error_rad"],
        "q0_q4_max_abs_drift_rad": row["q0_q4_max_abs_drift_rad"],
        "q5_applied_torque_nm": row["q5_torque"]["applied_torque_nm"],
        "q5_computed_torque_nm": row["q5_torque"]["computed_torque_nm"],
        "q5_effort_limit_nm": row["q5_torque"]["registered_effort_limit_nm"],
        "q5_effort_saturated": row["q5_torque"]["effort_saturated_diagnostic"],
        "object_disp_xy_mm": row["object_disp_xy_mm"],
        "object_z_delta_mm": row["object_z_delta_mm"],
        "object_tilt_deg": row["object_tilt_deg"],
        "object_tilt_delta_from_reference_deg": row[
            "object_tilt_delta_from_reference_deg"
        ],
        "object_bottom_table_gap_mm": row["object_bottom_table_gap_mm"],
        "robot_root_position_drift_m": row["robot_root_position_drift_m"],
        "robot_root_rotation_drift_rad": row["robot_root_rotation_drift_rad"],
    }
    for axis, value in zip("xyz", row["object_pos_w_m"], strict=True):
        out[f"object_pos_w_m_{axis}"] = value
    for axis, value in zip("xyz", row["object_lin_vel_w_mps"], strict=True):
        out[f"object_lin_vel_w_mps_{axis}"] = value
    for index in range(6):
        out[f"q{index}_actual_rad"] = row["actual_joint_rad"][index]
        out[f"q{index}_velocity_rad_s"] = row["actual_joint_vel_rad_s"][index]
        out[f"q{index}_target_rad"] = row["target_joint_rad"][index]
    for index in range(5):
        out[f"q{index}_actual_minus_frozen_rad"] = row["q0_q4_actual_minus_frozen_rad"][index]
    for label in ("support_table", "link4", "link5", "gripper_link"):
        item = row["contact"]["by_filter"][label]
        out[f"{label}_force_norm_n"] = item["force_norm_n"]
        for axis, value in zip("xyz", item["force_w_n"], strict=True):
            out[f"{label}_force_{axis}_n"] = value
        point = item["contact_point_w_m"] or [math.nan, math.nan, math.nan]
        for axis, value in zip("xyz", point, strict=True):
            out[f"{label}_contact_point_w_m_{axis}"] = value
    masks = row["event_masks"]
    for label in ("link4", "link5", "gripper_link"):
        out[f"{label}_force_ge_0p1n_instantaneous"] = masks[
            "instantaneous_body_force_ge_0p1n"
        ][label]
        out[f"{label}_two_step_confirmation_end"] = masks[
            "two_step_confirmation_end_by_filter"
        ][label]
        out[f"{label}_two_step_confirmed_onset"] = masks[
            "two_step_confirmed_onset_by_filter"
        ][label]
    out["any_robot_force_ge_0p1n_instantaneous"] = masks[
        "instantaneous_any_robot_body_force_ge_0p1n"
    ]
    out["any_robot_two_step_confirmation_end"] = masks[
        "two_step_any_robot_confirmation_end"
    ]
    out["object_motion_threshold_instantaneous"] = masks[
        "instantaneous_object_motion_threshold"
    ]
    out["object_motion_two_step_confirmation_end"] = masks[
        "two_step_motion_confirmation_end"
    ]
    out["object_motion_two_step_confirmed_onset"] = masks[
        "two_step_motion_confirmed_onset"
    ]
    return out


def _q0_q4_drift_summary(
    baseline_rows: list[dict[str, Any]], closure_rows: list[dict[str, Any]], closure: dict[str, Any]
) -> dict[str, Any]:
    all_rows = [*baseline_rows, *closure_rows]

    def sample(row: dict[str, Any] | None) -> Any:
        if row is None:
            return None
        return {
            "global_step": row["global_step"],
            "phase": row["phase"],
            "phase_step": row["phase_step"],
            "actual_minus_frozen_rad": row["q0_q4_actual_minus_frozen_rad"],
            "max_abs_drift_rad": row["q0_q4_max_abs_drift_rad"],
        }

    first_any_contact_row = next(
        (
            row
            for row in all_rows
            if any(
                row["event_masks"]["two_step_confirmed_onset_by_filter"].values()
            )
        ),
        None,
    )
    first_motion_row = next(
        (
            row
            for row in all_rows
            if row["event_masks"]["two_step_motion_confirmed_onset"]
        ),
        None,
    )
    moving_contact_idx = int(closure.get("first_moving_gripper_contact_step", -1))
    max_row = max(all_rows, key=lambda row: float(row["q0_q4_max_abs_drift_rad"]))
    return {
        "initial_exact_state": {
            "actual_minus_frozen_rad": [0.0] * 5,
            "max_abs_drift_rad": 0.0,
        },
        "baseline_end": sample(baseline_rows[-1]),
        "first_any_robot_contact_onset": sample(first_any_contact_row),
        "moving_gripper_contact_onset": sample(
            closure_rows[moving_contact_idx] if moving_contact_idx >= 0 else None
        ),
        "first_object_motion_onset": sample(first_motion_row),
        "final": sample(all_rows[-1]),
        "maximum_observed": sample(max_row),
        "failure_threshold": None,
        "semantics": "finite actual drift is reported, not failed; only target mutation or nonfinite state is operational failure",
    }


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_flatten_row(row) for row in rows]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
        stream.flush()
        os.fsync(stream.fileno())


def _rr_quaternion(rr: Any, quat_wxyz: list[float]) -> Any:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])


def _write_rerun(
    topology_parts: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    closure: dict[str, Any],
) -> dict[str, Any]:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    _marker("rerun_recording", "start", {"row_count": len(rows)})
    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"rerun SDK drift: {rr.__version__}")

    def eye() -> Any:
        return rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=CAMERA_EYE,
            look_target=CAMERA_TARGET,
            eye_up=[0.0, 0.0, 1.0],
        )

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/**", "/transforms/**", "/contacts/**"],
                    name="actual 64+64 colliders, cylinder, force witnesses",
                    eye_controls=eye(),
                    spatial_information=rrb.SpatialInformation(
                        target_frame="world", show_axes=True, show_bounding_box=False
                    ),
                ),
                rrb.TextLogView(origin="/events/d362", contents="/events/d362/**", name="bounded events"),
                column_shares=[0.74, 0.26],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/metrics/q5", contents="/metrics/q5/**", name="q5 target / actual / effort"),
                rrb.TimeSeriesView(origin="/metrics/force", contents="/metrics/force/**", name="filtered forces (N)"),
                rrb.TimeSeriesView(
                    origin="/metrics",
                    contents=["/metrics/motion/**", "/metrics/events/**"],
                    name="object motion / event masks",
                ),
                column_shares=[0.34, 0.33, 0.33],
            ),
            row_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    expected: set[str] = set()

    def remember(path: str) -> str:
        expected.add(path.strip("/"))
        return path

    summary_md = "\n".join(
        [
            "# D362 bounded PhysX jaw close",
            "",
            f"- baseline: {'PASS' if baseline['pass'] else 'FAIL'} ({BASELINE_STEPS} steps)",
            f"- closure verdict: `{closure['verdict']}`",
            f"- closure horizon: {closure['physics_steps']} / {CLOSURE_MAX_STEPS} steps",
            "- canonical authority: Float64 JSON/CSV; Rerun spatial values are display copies",
            "- no exact face, cap/rim, force-closure, grasp, or G0a decision",
        ]
    )
    cylinder_vertices, cylinder_triangles = d351._cylinder_mesh()
    colors = {"link5": [55, 160, 255, 145], "gripper_link": [255, 92, 80, 175]}
    with rr.RecordingStream(
        "roarm_g0a_d362_physx_contact_motion",
        recording_id="g0a_d362_physx_contact_motion",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(remember("metadata/run"), rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        recording.log(
            remember("transforms/world"),
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
                parent_frame="tf#/",
                child_frame="world",
            ),
            static=True,
        )
        for body in ("link5", "gripper_link"):
            for index, part in enumerate(topology_parts[body]):
                path = remember(f"geometry/colliders/{body}/part_{index:03d}")
                recording.log(
                    path,
                    rr.Mesh3D(
                        vertex_positions=np.asarray(part["_vertices"], dtype=np.float32),
                        triangle_indices=np.asarray(part["_triangles"], dtype=np.uint32),
                        albedo_factor=colors[body],
                    ),
                    rr.CoordinateFrame(f"actual/{body}"),
                    static=True,
                )
        recording.log(
            remember("geometry/cylinder"),
            rr.Mesh3D(
                vertex_positions=np.asarray(cylinder_vertices, dtype=np.float32),
                triangle_indices=np.asarray(cylinder_triangles, dtype=np.uint32),
                albedo_factor=[240, 176, 55, 185],
            ),
            rr.CoordinateFrame("actual/cylinder"),
            static=True,
        )
        metric_names = [
            "q5/actual_rad",
            "q5/target_rad",
            "q5/velocity_rad_s",
            "q5/applied_torque_nm",
            "q5/computed_torque_nm",
            "q5/effort_limit_nm",
            "q5/q0_q4_max_abs_drift_rad",
            "force/support_table_norm_n",
            "force/link4_norm_n",
            "force/link5_norm_n",
            "force/gripper_link_norm_n",
            "motion/object_xy_disp_mm",
            "motion/object_z_delta_mm",
            "motion/object_absolute_tilt_deg",
            "motion/object_tilt_delta_from_precommand_deg",
            "events/link4_force_ge_0p1n",
            "events/link5_force_ge_0p1n",
            "events/gripper_link_force_ge_0p1n",
            "events/any_robot_two_step_confirmation_end",
            "events/object_motion_threshold",
            "events/object_motion_two_step_confirmation_end",
        ]
        for row_index, row in enumerate(rows):
            step = int(row["global_step"])
            sim_time = float(row["physics_time_s"])
            recording.reset_time()
            recording.set_time("physics_step", sequence=step)
            recording.set_time("sim_time_s", timestamp=sim_time)
            for body in ("link5", "gripper_link"):
                recording.log(
                    remember(f"transforms/{body}"),
                    rr.Transform3D(
                        translation=row[f"{body if body == 'link5' else 'gripper'}_pos_w_m"],
                        rotation=_rr_quaternion(rr, row[f"{body if body == 'link5' else 'gripper'}_quat_wxyz"]),
                        parent_frame="world",
                        child_frame=f"actual/{body}",
                    ),
                )
            recording.log(
                remember("transforms/cylinder"),
                rr.Transform3D(
                    translation=row["object_pos_w_m"],
                    rotation=_rr_quaternion(rr, row["object_quat_wxyz"]),
                    parent_frame="world",
                    child_frame="actual/cylinder",
                ),
            )
            values = {
                "q5/actual_rad": row["q5_actual_rad"],
                "q5/target_rad": row["q5_target_rad"],
                "q5/velocity_rad_s": row["q5_velocity_rad_s"],
                "q5/applied_torque_nm": row["q5_torque"]["applied_torque_nm"],
                "q5/computed_torque_nm": row["q5_torque"]["computed_torque_nm"],
                "q5/effort_limit_nm": row["q5_torque"]["registered_effort_limit_nm"],
                "q5/q0_q4_max_abs_drift_rad": row["q0_q4_max_abs_drift_rad"],
                "force/support_table_norm_n": row["contact"]["by_filter"]["support_table"]["force_norm_n"],
                "force/link4_norm_n": row["contact"]["by_filter"]["link4"]["force_norm_n"],
                "force/link5_norm_n": row["contact"]["by_filter"]["link5"]["force_norm_n"],
                "force/gripper_link_norm_n": row["contact"]["by_filter"]["gripper_link"]["force_norm_n"],
                "motion/object_xy_disp_mm": row["object_disp_xy_mm"],
                "motion/object_z_delta_mm": row["object_z_delta_mm"],
                "motion/object_absolute_tilt_deg": row["object_tilt_deg"],
                "motion/object_tilt_delta_from_precommand_deg": row[
                    "object_tilt_delta_from_reference_deg"
                ],
                "events/link4_force_ge_0p1n": float(
                    row["event_masks"]["instantaneous_body_force_ge_0p1n"][
                        "link4"
                    ]
                ),
                "events/link5_force_ge_0p1n": float(
                    row["event_masks"]["instantaneous_body_force_ge_0p1n"][
                        "link5"
                    ]
                ),
                "events/gripper_link_force_ge_0p1n": float(
                    row["event_masks"]["instantaneous_body_force_ge_0p1n"][
                        "gripper_link"
                    ]
                ),
                "events/any_robot_two_step_confirmation_end": float(
                    row["event_masks"]["two_step_any_robot_confirmation_end"]
                ),
                "events/object_motion_threshold": float(
                    row["event_masks"]["instantaneous_object_motion_threshold"]
                ),
                "events/object_motion_two_step_confirmation_end": float(
                    row["event_masks"]["two_step_motion_confirmation_end"]
                ),
            }
            for name in metric_names:
                recording.log(remember(f"metrics/{name}"), rr.Scalars([float(values[name])]))
            for label in ("support_table", "link4", "link5", "gripper_link"):
                item = row["contact"]["by_filter"][label]
                point = item["contact_point_w_m"]
                force = np.asarray(item["force_w_n"], dtype=np.float64)
                if point is None or not np.isfinite(np.asarray(point)).all() or float(np.linalg.norm(force)) <= 0.0:
                    continue
                recording.log(
                    remember(f"contacts/{label}/point"),
                    rr.Points3D([point], radii=[0.002], labels=[f"{label} aggregated sensor point"]),
                    rr.CoordinateFrame("world"),
                )
                recording.log(
                    remember(f"contacts/{label}/force_display_scale"),
                    rr.Arrows3D(
                        origins=[point],
                        vectors=[(force * 0.005).tolist()],
                        radii=[0.0007],
                        labels=[f"{label} force, display scale 0.005 m/N"],
                    ),
                    rr.CoordinateFrame("world"),
                )
            event_text = None
            if row["phase"] == "frozen_open_baseline" and row["phase_step"] == 0:
                event_text = "OPEN frozen-pose baseline start"
            elif row["phase"] == "q5_close_observation" and row["phase_step"] == 0:
                event_text = "q5 target changed once to 0; 300-step observation horizon starts"
            if row["event_masks"]["two_step_any_robot_confirmation_end"]:
                labels = [
                    label
                    for label, value in row["event_masks"][
                        "two_step_confirmation_end_by_filter"
                    ].items()
                    if value
                ]
                event_text = (event_text + "; " if event_text else "") + (
                    f"robot contact confirmation: {labels} at "
                    f"{row['phase']} step {row['phase_step']}"
                )
            if row["event_masks"]["two_step_motion_confirmation_end"]:
                event_text = (event_text + "; " if event_text else "") + (
                    f"object motion confirmation at {row['phase']} step "
                    f"{row['phase_step']}"
                )
            if row is rows[-1]:
                event_text = (event_text + "; " if event_text else "") + f"final: {closure['verdict']}"
            if event_text:
                recording.log(remember("events/d362/run"), rr.TextLog(event_text, level=rr.TextLogLevel.INFO))
            if row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(rows):
                _marker(
                    "rerun_recording",
                    "progress",
                    {"completed": row_index + 1, "requested": len(rows)},
                )
        recording.flush(timeout_sec=30.0)
    _marker("rerun_recording", "rrd_finalized", {"rrd_bytes": RRD_PATH.stat().st_size})
    blueprint.save("roarm_g0a_d362_physx_contact_motion", RBL_PATH)
    component_contract = {
        "metadata/run": ["TextDocument:text"],
        "transforms/world": [
            "Transform3D:child_frame",
            "Transform3D:parent_frame",
            "Transform3D:quaternion",
            "Transform3D:translation",
        ],
        "geometry/cylinder": [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ],
        "metrics/q5/actual_rad": ["Scalars:scalars"],
        "metrics/force/gripper_link_norm_n": ["Scalars:scalars"],
        "metrics/motion/object_xy_disp_mm": ["Scalars:scalars"],
        "metrics/events/any_robot_two_step_confirmation_end": [
            "Scalars:scalars"
        ],
        "events/d362/run": ["TextLog:level", "TextLog:text"],
    }
    _marker("rerun_validation", "start", {"timeout_s": 180.0})
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected),
        exact_entity_paths=sorted(expected),
        expected_timeline_names=["physics_step", "sim_time_s"],
        exact_timeline_names=["blueprint", "log_time", "physics_step", "sim_time_s"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size=RERUN_SCREENSHOT_SIZE,
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    _marker("rerun_validation", "complete", {"pass": validation.get("pass")})
    return validation


def _project_world_points(
    points_w: np.ndarray,
    camera_eye: list[float],
    camera_target: list[float],
    viewport: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic display-only pinhole projection for trace replay."""
    points = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    eye = np.asarray(camera_eye, dtype=np.float64)
    target = np.asarray(camera_target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    relative = points - eye
    depth = relative @ forward
    x_camera = relative @ right
    y_camera = relative @ up
    left, top, right_px, bottom = viewport
    width = right_px - left
    height = bottom - top
    focal = 0.5 * width / math.tan(math.radians(48.0) / 2.0)
    safe_depth = np.where(depth > 1.0e-6, depth, np.nan)
    projected = np.column_stack(
        [
            left + 0.5 * width + focal * x_camera / safe_depth,
            top + 0.5 * height - focal * y_camera / safe_depth,
        ]
    )
    return projected, depth


def _transform_vertices(
    vertices_local: np.ndarray, position_w: list[float], quat_wxyz: list[float]
) -> np.ndarray:
    rotation = d351.d332._quat_wxyz_to_rot(
        np.asarray(quat_wxyz, dtype=np.float64)
    )
    return (
        np.asarray(vertices_local, dtype=np.float64) @ rotation.T
        + np.asarray(position_w, dtype=np.float64)
    )


def _draw_replay_view(
    draw: Any,
    row: dict[str, Any],
    topology_parts: dict[str, list[dict[str, Any]]],
    *,
    camera_eye: list[float],
    viewport: tuple[int, int, int, int],
    title: str,
    font: Any,
) -> None:
    left, top, right, bottom = viewport
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=18,
        fill=(17, 24, 34),
        outline=(92, 111, 137),
        width=3,
    )
    draw.text(
        (left + 18, top + 12),
        title,
        font=font(25),
        fill=(235, 240, 248),
    )

    # Ground/table grid is an orientation aid only; it is not contact evidence.
    grid_z = float(d351.d332.TABLE_Z_M)
    for x in np.linspace(0.20, 0.38, 10):
        points = np.asarray([[x, -0.10, grid_z], [x, 0.10, grid_z]])
        projected, depth = _project_world_points(
            points, camera_eye, CAMERA_TARGET, viewport
        )
        if np.all(depth > 0.0) and np.isfinite(projected).all():
            draw.line([tuple(projected[0]), tuple(projected[1])], fill=(48, 60, 73), width=1)
    for y in np.linspace(-0.10, 0.10, 11):
        points = np.asarray([[0.20, y, grid_z], [0.38, y, grid_z]])
        projected, depth = _project_world_points(
            points, camera_eye, CAMERA_TARGET, viewport
        )
        if np.all(depth > 0.0) and np.isfinite(projected).all():
            draw.line([tuple(projected[0]), tuple(projected[1])], fill=(48, 60, 73), width=1)

    triangle_draws: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], tuple[int, int, int]]] = []
    body_styles = {
        "link5": ((38, 112, 188), (119, 200, 255)),
        "gripper_link": ((185, 58, 53), (255, 145, 132)),
    }
    for body in ("link5", "gripper_link"):
        position_key = "link5_pos_w_m" if body == "link5" else "gripper_pos_w_m"
        quat_key = "link5_quat_wxyz" if body == "link5" else "gripper_quat_wxyz"
        fill, outline = body_styles[body]
        for part in topology_parts[body]:
            world = _transform_vertices(
                np.asarray(part["_vertices"]), row[position_key], row[quat_key]
            )
            projected, depth = _project_world_points(
                world, camera_eye, CAMERA_TARGET, viewport
            )
            triangles = np.asarray(part["_triangles"], dtype=np.int64).reshape(-1, 3)
            for triangle in triangles:
                tri_depth = depth[triangle]
                tri_xy = projected[triangle]
                if np.any(tri_depth <= 1.0e-6) or not np.isfinite(tri_xy).all():
                    continue
                triangle_draws.append(
                    (
                        float(np.mean(tri_depth)),
                        [tuple(point) for point in tri_xy],
                        fill,
                        outline,
                    )
                )

    cylinder_vertices, cylinder_triangles = d351._cylinder_mesh()
    cylinder_world = _transform_vertices(
        cylinder_vertices, row["object_pos_w_m"], row["object_quat_wxyz"]
    )
    cylinder_projected, cylinder_depth = _project_world_points(
        cylinder_world, camera_eye, CAMERA_TARGET, viewport
    )
    for triangle in np.asarray(cylinder_triangles, dtype=np.int64).reshape(-1, 3):
        tri_depth = cylinder_depth[triangle]
        tri_xy = cylinder_projected[triangle]
        if np.any(tri_depth <= 1.0e-6) or not np.isfinite(tri_xy).all():
            continue
        triangle_draws.append(
            (
                float(np.mean(tri_depth)),
                [tuple(point) for point in tri_xy],
                (189, 126, 34),
                (255, 207, 92),
            )
        )

    # Painter's algorithm: far triangles first, then near triangles.
    for _, polygon, fill, outline in sorted(
        triangle_draws, key=lambda item: item[0], reverse=True
    ):
        draw.polygon(polygon, fill=fill, outline=outline)

    contact_colors = {
        "support_table": (92, 220, 126),
        "link4": (210, 112, 255),
        "link5": (90, 192, 255),
        "gripper_link": (255, 103, 92),
    }
    for label in d361.BODY_LABELS:
        item = row["contact"]["by_filter"][label]
        point = item["contact_point_w_m"]
        force = np.asarray(item["force_w_n"], dtype=np.float64)
        if point is None or not np.isfinite(np.asarray(point)).all():
            continue
        start_w = np.asarray(point, dtype=np.float64)
        end_w = start_w + force * 0.005
        projected, depth = _project_world_points(
            np.vstack([start_w, end_w]), camera_eye, CAMERA_TARGET, viewport
        )
        if np.any(depth <= 0.0) or not np.isfinite(projected).all():
            continue
        color = contact_colors[label]
        x, y = projected[0]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
        draw.line([tuple(projected[0]), tuple(projected[1])], fill=color, width=4)
        draw.text(
            (x + 10, y - 20),
            f"{label} {float(item['force_norm_n']):.3f}N",
            font=font(16),
            fill=color,
        )

    draw.text(
        (left + 16, bottom - 30),
        "blue=link5 red=moving gripper orange=cylinder; link4 is sensor marker only",
        font=font(14),
        fill=(180, 194, 214),
    )


def _render_trace_replay_frame(
    topology_parts: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    row_index: int,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> Any:
        return ImageFont.truetype(str(FONT_PATH), size=size)

    row = rows[row_index]
    canvas = Image.new("RGB", tuple(VIDEO_SIZE), (9, 13, 19))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (36, 22),
        "D362 canonical trace replay — PHYSICS NOT RECOMPUTED",
        font=font(34),
        fill=(255, 222, 107),
    )
    draw.text(
        (1884, 28),
        f"row {row_index + 1}/{len(rows)}  global step {row['global_step']}",
        font=font(23),
        fill=(226, 233, 243),
        anchor="ra",
    )
    _draw_replay_view(
        draw,
        row,
        topology_parts,
        camera_eye=CAMERA_EYE,
        viewport=(34, 86, 940, 716),
        title="Primary oblique view",
        font=font,
    )
    _draw_replay_view(
        draw,
        row,
        topology_parts,
        camera_eye=OPPOSITE_CAMERA_EYE,
        viewport=(980, 86, 1886, 716),
        title="Opposite anti-occlusion view",
        font=font,
    )

    # Full-run q5 overview with a current-row cursor.
    chart = (46, 755, 1215, 1044)
    draw.rounded_rectangle(chart, radius=16, fill=(17, 24, 34), outline=(75, 92, 114), width=2)
    draw.text((66, 770), "q5 actual / target over the recorded run", font=font(20), fill=(225, 232, 241))
    x0, y0, x1, y1 = 72, 819, 1186, 998
    q_min, q_max = -0.05, float(Q5_OPEN_F32) + 0.05

    def chart_xy(index: int, value: float) -> tuple[float, float]:
        x = x0 + (x1 - x0) * index / max(len(rows) - 1, 1)
        y = y1 - (y1 - y0) * (value - q_min) / (q_max - q_min)
        return x, y

    actual_line = [chart_xy(i, float(item["q5_actual_rad"])) for i, item in enumerate(rows)]
    target_line = [chart_xy(i, float(item["q5_target_rad"])) for i, item in enumerate(rows)]
    draw.line(actual_line, fill=(92, 203, 255), width=3)
    draw.line(target_line, fill=(255, 111, 99), width=3)
    cursor_x, _ = chart_xy(row_index, float(row["q5_actual_rad"]))
    draw.line((cursor_x, y0, cursor_x, y1), fill=(255, 222, 107), width=3)
    draw.text((82, 965), "actual", font=font(16), fill=(92, 203, 255))
    draw.text((172, 965), "target", font=font(16), fill=(255, 111, 99))

    force_rows = row["contact"]["by_filter"]
    metrics = [
        f"phase: {row['phase']} / step {row['phase_step']}",
        f"q5 a/t/v = {row['q5_actual_rad']:.5f} / {row['q5_target_rad']:.5f} / {row['q5_velocity_rad_s']:.3f}",
        f"F(N) L4/L5/grip = {force_rows['link4']['force_norm_n']:.3f} / {force_rows['link5']['force_norm_n']:.3f} / {force_rows['gripper_link']['force_norm_n']:.3f}",
        f"object XY/tiltΔ/zΔ = {row['object_disp_xy_mm']:.3f}mm / {row['object_tilt_delta_from_reference_deg']:.3f}° / {row['object_z_delta_mm']:.3f}mm",
        f"gates: F≥{ROBOT_FORCE_EVENT_N:.1f}N×2; XY≥{OBJECT_XY_EVENT_M * 1000.0:.1f}mm or tilt≥{OBJECT_TILT_EVENT_DEG:.1f}°×2",
    ]
    event_labels = [
        label
        for label, value in row["event_masks"][
            "two_step_confirmation_end_by_filter"
        ].items()
        if value
    ]
    if event_labels:
        metrics.append("CONTACT CONFIRMATION: " + ", ".join(event_labels))
    if row["event_masks"]["two_step_motion_confirmation_end"]:
        metrics.append("OBJECT MOTION CONFIRMATION")
    draw.rounded_rectangle((1245, 755, 1886, 1044), radius=16, fill=(25, 31, 42), outline=(75, 92, 114), width=2)
    y = 770
    for index, line in enumerate(metrics):
        color = (255, 222, 107) if index >= 5 else (230, 236, 244)
        draw.text((1266, y), line, font=font(17 if index < 5 else 18), fill=color)
        y += 37 if index < 5 else 29
    return canvas


def _build_trace_replay_video(
    topology_parts: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    import cv2
    import imageio_ffmpeg
    from PIL import Image

    if not rows:
        raise RuntimeError("D362 video requires a nonempty canonical trace")
    for path in (VIDEO_PATH, VIDEO_REPORT_PATH, VIDEO_STORYBOARD_PATH):
        if path.exists():
            raise RuntimeError(f"D362 replay output would overwrite: {path}")
    frame_count = int(math.ceil(len(rows) / VIDEO_STRIDE))
    frame_indices = np.linspace(
        0, len(rows) - 1, frame_count, dtype=np.int64
    ).tolist()
    if len(set(frame_indices)) != frame_count:
        raise RuntimeError("D362 replay frame sampling produced duplicates")
    _marker(
        "trace_replay_video",
        "start",
        {
            "trace_rows": len(rows),
            "frame_count": frame_count,
            "size": VIDEO_SIZE,
            "fps": VIDEO_FPS,
            "authority": VIDEO_REPLAY_AUTHORITY,
        },
    )
    writer = imageio_ffmpeg.write_frames(
        str(VIDEO_PATH),
        tuple(VIDEO_SIZE),
        pix_fmt_in="rgb24",
        pix_fmt_out=VIDEO_PIXEL_FORMAT,
        fps=VIDEO_FPS,
        codec=VIDEO_CODEC,
        quality=7,
        macro_block_size=16,
        ffmpeg_log_level="warning",
        output_params=[
            "-movflags",
            "+faststart",
            "-metadata",
            "comment=D362 canonical trace replay; physics not recomputed",
        ],
    )
    writer.send(None)
    try:
        for frame_number, row_index in enumerate(frame_indices):
            image = _render_trace_replay_frame(topology_parts, rows, row_index)
            frame = np.asarray(image, dtype=np.uint8)
            if frame.shape != (VIDEO_SIZE[1], VIDEO_SIZE[0], 3):
                raise RuntimeError(f"D362 replay frame shape drift: {frame.shape}")
            writer.send(np.ascontiguousarray(frame))
            if frame_number == 0 or (frame_number + 1) % 25 == 0 or frame_number + 1 == frame_count:
                _marker(
                    "trace_replay_video",
                    "progress",
                    {
                        "completed": frame_number + 1,
                        "requested": frame_count,
                        "source_row_index": row_index,
                    },
                )
    finally:
        writer.close()

    contact_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["event_masks"]["two_step_any_robot_confirmation_end"]
        ),
        None,
    )
    motion_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row["event_masks"]["two_step_motion_confirmation_end"]
        ),
        None,
    )
    precommand_index = min(BASELINE_STEPS - 1, len(rows) - 1)
    fallback_contact_index = (
        min(BASELINE_STEPS + max((len(rows) - BASELINE_STEPS) // 2, 0), len(rows) - 1)
        if len(rows) > BASELINE_STEPS
        else len(rows) - 1
    )
    storyboard_sources = [
        (0, "initial recorded row"),
        (precommand_index, "OPEN pre-command row"),
        (
            contact_index if contact_index is not None else fallback_contact_index,
            "first 2-row contact confirmation"
            if contact_index is not None
            else "no contact confirmation — diagnostic midpoint/final",
        ),
        (
            motion_index if motion_index is not None else len(rows) - 1,
            "first 2-row motion confirmation"
            if motion_index is not None
            else "no motion confirmation — final row",
        ),
    ]
    storyboard = Image.new("RGB", tuple(VIDEO_SIZE), (9, 13, 19))
    for panel_index, (source_index, _) in enumerate(storyboard_sources):
        frame = _render_trace_replay_frame(topology_parts, rows, source_index)
        panel = frame.resize((VIDEO_SIZE[0] // 2, VIDEO_SIZE[1] // 2))
        storyboard.paste(
            panel,
            (
                (panel_index % 2) * (VIDEO_SIZE[0] // 2),
                (panel_index // 2) * (VIDEO_SIZE[1] // 2),
            ),
        )
    storyboard.save(VIDEO_STORYBOARD_PATH)

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    reported_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    reported_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    decode_indices = sorted({0, frame_count // 2, frame_count - 1})
    decode_samples: list[dict[str, Any]] = []
    for index in decode_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, decoded = capture.read()
        decode_samples.append(
            {
                "frame_index": index,
                "decode_ok": bool(ok),
                "shape": list(decoded.shape) if ok else None,
                "pixel_stddev": float(np.std(decoded)) if ok else None,
                "nonblank": bool(ok and float(np.std(decoded)) > 5.0),
            }
        )
    capture.release()
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    full_decode = subprocess.run(
        [
            str(ffmpeg_exe),
            "-hide_banner",
            "-i",
            str(VIDEO_PATH),
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=180.0,
    )
    with Image.open(VIDEO_STORYBOARD_PATH) as image:
        image.load()
        storyboard_dimensions = list(image.size)
        storyboard_stddev = float(np.std(np.asarray(image)))
    checks = {
        "video_nonempty": VIDEO_PATH.is_file() and VIDEO_PATH.stat().st_size > 0,
        "resolution_1920x1080": [reported_width, reported_height] == VIDEO_SIZE,
        "fps_20": math.isclose(reported_fps, VIDEO_FPS, rel_tol=0.0, abs_tol=0.01),
        "frame_count_exact": reported_frames == frame_count,
        "full_500_trace_has_registered_250_frames": (
            frame_count == VIDEO_EXPECTED_FRAME_COUNT
            if len(rows) == BASELINE_STEPS + CLOSURE_MAX_STEPS
            else True
        ),
        "first_middle_last_decode_nonblank": all(
            item["decode_ok"] and item["nonblank"] for item in decode_samples
        ),
        "ffmpeg_full_decode_pass": full_decode.returncode == 0,
        "encoded_codec_h264_from_libx264": "Video: h264" in full_decode.stderr,
        "encoded_pixel_format_yuv420p": "yuv420p" in full_decode.stderr,
        "storyboard_1920x1080_nonblank": storyboard_dimensions == VIDEO_SIZE
        and storyboard_stddev > 5.0,
        "source_endpoints_included": frame_indices[0] == 0
        and frame_indices[-1] == len(rows) - 1,
        "physics_not_recomputed": True,
    }
    report = {
        "artifact": "D362_INTERFACE_VISIBLE_TRACE_REPLAY_REPORT_V1",
        "case": CASE,
        "authority": VIDEO_REPLAY_AUTHORITY,
        "display_scope_limitation": "The frozen live topology binding contains link5+gripper 64+64 only; link4 is shown only by its authoritative aggregated sensor point/force, while actual Isaac captures retain the full robot view.",
        "canonical_trace_path": _rel(TRACE_JSON_PATH),
        "canonical_trace_sha256": _sha(TRACE_JSON_PATH),
        "video_path": _rel(VIDEO_PATH),
        "video_sha256": _sha(VIDEO_PATH),
        "video_bytes": VIDEO_PATH.stat().st_size,
        "storyboard_path": _rel(VIDEO_STORYBOARD_PATH),
        "storyboard_sha256": _sha(VIDEO_STORYBOARD_PATH),
        "trace_row_count": len(rows),
        "source_row_indices": frame_indices,
        "encoded_frame_count": frame_count,
        "reported_frame_count": reported_frames,
        "reported_resolution": [reported_width, reported_height],
        "reported_fps": reported_fps,
        "duration_seconds": frame_count / VIDEO_FPS,
        "decode_samples": decode_samples,
        "full_decode": {
            "ffmpeg": str(ffmpeg_exe),
            "returncode": full_decode.returncode,
            "stderr": full_decode.stderr,
        },
        "storyboard_sources": [
            {"source_row_index": index, "label": label}
            for index, label in storyboard_sources
        ],
        "storyboard_dimensions": storyboard_dimensions,
        "storyboard_pixel_stddev": storyboard_stddev,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(VIDEO_REPORT_PATH, report)
    _marker(
        "trace_replay_video",
        "complete",
        {"pass": report["pass"], "frames": frame_count},
    )
    return report


def _png_report(path: Path) -> dict[str, Any]:
    from PIL import Image

    if not path.is_file() or path.stat().st_size == 0:
        return {"path": _rel(path), "pass": False, "error": "missing or empty"}
    with Image.open(path) as image:
        image.load()
        size = list(image.size)
        mode = image.mode
    return {
        "path": _rel(path),
        "dimensions": size,
        "mode": mode,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "pass": size == VIEWPORT_SIZE,
    }


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SHEET_PATH = OUT_DIR / "d362_beginner_result_sheet_ko.png"


def _build_beginner_sheet(
    captures: dict[str, Any],
    capture_event_metadata: dict[str, Any],
    baseline: dict[str, Any],
    closure: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> Any:
        return ImageFont.truetype(str(FONT_PATH), size=size)

    contact_path = CAPTURE_PATHS["contact"] if captures.get("contact") else CAPTURE_PATHS["final"]
    decision_path = CAPTURE_PATHS["motion"] if captures.get("motion") else CAPTURE_PATHS["final"]
    panel_paths = [CAPTURE_PATHS["start"], contact_path, decision_path]
    panel_titles = [
        "A. OPEN — q5 닫기 명령 직전",
        "B. 첫 robot-body 접촉 확인" if captures.get("contact") else "B. 양성 접촉 없음 — FINAL",
        "C. 첫 물체 운동 확인" if captures.get("motion") else "C. 판정 시점 — FINAL",
    ]
    canvas = Image.new("RGB", (3840, 1720), (17, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((1920, 24), "D362 현재 자세 PhysX 죠 닫힘 — 초보자용 결과판", font=font(58), fill=(248, 250, 253), anchor="ma")
    for index, (path, title) in enumerate(zip(panel_paths, panel_titles, strict=True)):
        with Image.open(path) as source:
            panel = source.convert("RGB").resize((1280, 720))
        x = 1280 * index
        canvas.paste(panel, (x, 135))
        draw.rectangle((x, 92, x + 1279, 135), fill=(40, 53, 70))
        draw.text((x + 640, 96), title, font=font(30), fill=(255, 255, 255), anchor="ma")
    final = rows[-1]
    peak_by_filter = closure["peak_force_by_filter"]
    moving_peak = max(
        float(baseline["robot_filter_max_n"]["gripper_link"]),
        float(peak_by_filter["gripper_link"]["force_norm_n"] or 0.0),
    )
    fixed_peak = max(
        float(baseline["robot_filter_max_n"]["link5"]),
        float(peak_by_filter["link5"]["force_norm_n"] or 0.0),
    )
    link4_peak = max(
        float(baseline["robot_filter_max_n"]["link4"]),
        float(peak_by_filter["link4"]["force_norm_n"] or 0.0),
    )
    full_xy_max = max(
        float(baseline["max_object_disp_xy_mm"]),
        float(closure["max_object_disp_xy_mm"] or 0.0),
    )
    full_tilt_max = max(
        float(baseline["max_object_tilt_deg"]),
        float(closure["max_object_tilt_deg"] or 0.0),
    )
    xy_text = f"{full_xy_max:.4f} mm"
    tilt_text = f"{full_tilt_max:.4f}°"
    q5_line = (
        f"초기 q5 actual = {float(Q5_OPEN_F32):.10f} rad     "
        f"OPEN baseline 끝 q5 actual = {baseline['q5_actual_final_rad']:.10f} rad     "
        f"최종 actual/target = {final['q5_actual_rad']:.6f}/{final['q5_target_rad']:.1f} rad"
    )
    metrics = (
        f"moving gripper / fixed link5 / link4 최대 힘 = "
        f"{moving_peak:.4f} / {fixed_peak:.4f} / {link4_peak:.4f} N"
        f"     최대 XY 이동: {xy_text}     닫기 직전 대비 최대 기울기: {tilt_text}"
    )
    draw.rounded_rectangle((55, 885, 3785, 1025), radius=18, fill=(36, 47, 62), outline=(111, 135, 166), width=3)
    draw.text((1920, 910), q5_line, font=font(34), fill=(255, 214, 102), anchor="ma")
    draw.text((1920, 968), metrics, font=font(31), fill=(238, 242, 248), anchor="ma")
    contact_yes = closure["positive_body_level_contact_supported"] is True
    motion_yes = closure["object_motion_after_contact_supported"] is True
    clean_attribution = closure["verdict"] in {
        "D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED",
        "D362_MOVING_JAW_CONTACT_WITHOUT_THRESHOLD_OBJECT_MOTION",
    }
    if clean_attribution and contact_yes and motion_yes:
        proven = "moving gripper_link가 먼저 접촉했고 그 뒤 원통 운동이 관찰됨"
        box_fill, box_outline, heading_color = (29, 64, 50), (70, 207, 131), (101, 238, 157)
        heading = "선행 접촉 귀속 가능"
    elif closure["verdict"] == "D362_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP":
        proven = (
            "OPEN baseline에 이미 robot-body 접촉/물체 운동이 있었거나, "
            "closure에서 link4·link5 접촉 또는 물체 운동이 moving jaw보다 "
            "선행/동시여서 원인을 단독 귀속할 수 없음"
        )
        box_fill, box_outline, heading_color = (73, 55, 25), (244, 174, 61), (255, 204, 102)
        heading = "접촉 원인 귀속 실패"
    elif clean_attribution and contact_yes:
        proven = "moving gripper_link 선행 body 접촉은 양성, 기준 이상 원통 운동은 없음"
        box_fill, box_outline, heading_color = (29, 64, 50), (70, 207, 131), (101, 238, 157)
        heading = "선행 접촉 관찰됨"
    else:
        proven = "moving gripper_link 양성 접촉 미확정 — 접촉이 절대 없었다는 증명은 아님"
        box_fill, box_outline, heading_color = (42, 54, 70), (104, 150, 207), (151, 201, 255)
        heading = "양성 접촉 미확정"
    draw.rounded_rectangle((55, 1050, 1875, 1540), radius=22, fill=box_fill, outline=box_outline, width=5)
    draw.text((105, 1090), heading, font=font(47), fill=heading_color)
    baseline_word = "PASS" if baseline["pass"] else "FAIL — closure 미실행"
    failed_baseline = [name for name, value in baseline["checks"].items() if not value]
    baseline_check_aliases = {
        "stage_and_sensor_contracts_pass": "stage/sensor 계약",
        "exact_200_steps": "200-step 수",
        "frozen_open_target_every_step": "OPEN target 고정",
        "support_table_fz_last50_median_gt_1n": "table 지지력",
        "first_step_abs_z_delta_le_0p5mm": "첫 z 보정",
        "last50_bottom_table_gap_abs_max_le_0p5mm": "table gap",
        "robot_filters_max_lt_0p1n": "baseline robot force",
        "robot_root_position_drift_le_1e_6m": "root 위치",
        "robot_root_rotation_drift_le_1e_6rad": "root 회전",
        "object_xy_lt_0p5mm": "baseline XY",
        "object_tilt_lt_1deg": "baseline 기울기",
        "all_finite": "finite 수치",
        "q5_effort_authority_every_step": "q5 effort 기록",
    }
    failed_aliases = [baseline_check_aliases.get(name, name) for name in failed_baseline]
    shown_failures = failed_aliases[:4]
    if len(failed_aliases) > len(shown_failures):
        shown_failures.append(f"그 외 {len(failed_aliases) - len(shown_failures)}개")
    contact_meta = capture_event_metadata.get("contact")
    contact_where = (
        "없음"
        if contact_meta is None
        else f"{contact_meta['phase']} step {contact_meta['onset_phase_step']}→"
        f"{contact_meta['confirmation_phase_step']}, body="
        f"{'+'.join(contact_meta['qualifying_body_labels'])}"
    )
    left_items = [
        "핵심: " + proven,
        f"첫 robot-body 확인: {contact_where}",
        f"200-step OPEN baseline: {baseline_word}",
        "baseline 실패 gate: " + (", ".join(shown_failures) if shown_failures else "없음"),
        f"판정 코드: {closure['verdict']}",
    ]
    left_text = "\n".join(
        "• " + line
        for item in left_items
        for line in textwrap.wrap(item, width=64, subsequent_indent="  ")
    )
    left_font = font(22)
    left_spacing = 7
    left_bbox = draw.multiline_textbbox(
        (0, 0), left_text, font=left_font, spacing=left_spacing
    )
    if left_bbox[2] - left_bbox[0] > 1710 or left_bbox[3] - left_bbox[1] > 335:
        left_font = font(19)
        left_spacing = 5
        left_bbox = draw.multiline_textbbox(
            (0, 0), left_text, font=left_font, spacing=left_spacing
        )
    if left_bbox[2] - left_bbox[0] > 1710 or left_bbox[3] - left_bbox[1] > 335:
        raise RuntimeError(
            "D362 beginner-sheet left result text exceeds registered box"
        )
    draw.multiline_text(
        (105, 1165),
        left_text,
        font=left_font,
        fill=(235, 248, 240),
        spacing=left_spacing,
    )
    draw.rounded_rectangle((1965, 1050, 3785, 1540), radius=22, fill=(70, 43, 40), outline=(255, 111, 99), width=5)
    draw.text((2015, 1090), "아직 증명 아님", font=font(47), fill=(255, 142, 130))
    draw.multiline_text((2015, 1170), "• 정확한 collider face / contact manifold\n• cap·rim·barrel 선접촉 순서\n• force closure, grasp 성공, G0a PASS\n• 300 step은 q5=0 settle 보장이 아님", font=font(34), fill=(255, 239, 236), spacing=22)
    draw.text((1920, 1620), "과학 판정 원본은 d362_physics_trace.json / .csv이며, PNG와 Rerun은 사람이 보는 표시층입니다.", font=font(31), fill=(181, 197, 220), anchor="ma")
    canvas.save(SHEET_PATH)
    with Image.open(SHEET_PATH) as image:
        image.load()
        size = list(image.size)
    return {"path": _rel(SHEET_PATH), "dimensions": size, "sha256": _sha(SHEET_PATH), "pass": size == [3840, 1720]}


def _worker(args: argparse.Namespace) -> int:
    simulation_app = None
    inner = None
    settings = None
    previous_physx: Any = None
    previous_play: Any = None
    try:
        prereg = _json(PREREG_PATH)
        invocation = _json(INVOCATION_PATH)
        token = os.environ.get(WORKER_TOKEN_ENV, "")
        supervisor_pid = int(os.environ.get(SUPERVISOR_PID_ENV, "-1"))
        early_runtime_modules = sorted(
            name
            for name in sys.modules
            if name in {"pxr", "omni", "isaaclab", "isaacsim", "carb"}
            or name.startswith(("pxr.", "omni.", "isaaclab.", "isaacsim.", "carb."))
        )
        gpu = _gpu_snapshot()
        checks = {
            "prereg_pass": prereg.get("pass") is True,
            "single_invocation_marker": invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("invocation_index") == 1
            and invocation.get("automatic_retry") is False
            and invocation.get("preregistration_sha256") == _sha(PREREG_PATH),
            "registered_parent_supervisor": supervisor_pid > 0
            and os.getppid() == supervisor_pid
            and invocation.get("supervisor_pid") == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest() == invocation.get("worker_token_sha256"),
            "head_origin_and_preregistered_base_exact": _git_head()
            == _git_head("origin/master")
            == prereg.get("base_git")
            == BASE_GIT,
            "git_scope_only_d359_d362": _status_scope_ok(_git_status()),
            "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
            "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
            "registered_python": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
            "display_headless_device": os.environ.get("DISPLAY") == ":1"
            and args.headless is False
            and int(args.livestream) == 0
            and str(args.device) == "cuda:0",
            "runtime_modules_absent_before_applauncher": not early_runtime_modules,
            "gpu_resource_gate": int(gpu.get("memory_free_mib", 0)) >= MIN_GPU_FREE_MIB,
            "ram_resource_gate": int(gpu.get("ram_available_bytes", 0)) >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D362_WORKER_PREFLIGHT_V1",
            "utc": _utc_now(),
            "pid": os.getpid(),
            "early_runtime_modules": early_runtime_modules,
            "gpu_and_ram": gpu,
            "checks": checks,
            "pass": all(checks.values()),
        }
        _write_json_x(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker_preflight", "complete", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D362 worker preflight STOP: {checks}")

        from isaaclab.app import AppLauncher

        _marker("AppLauncher", "start")
        launcher = AppLauncher(
            {
                "headless": False,
                "livestream": 0,
                "enable_cameras": False,
                "xr": False,
                "device": "cuda:0",
                "experience": "",
                "kit_args": "",
                "rendering_mode": "balanced",
                "anim_recording_enabled": False,
            }
        )
        simulation_app = launcher.app
        launcher_report = d351.d350._resolved_gui_launcher(launcher)
        _marker("AppLauncher", "complete", {"pass": launcher_report.get("pass")})
        if not launcher_report.get("pass"):
            raise RuntimeError(f"D362 GUI launcher contract failed: {launcher_report}")

        import carb
        import omni.timeline

        args.robot_usd_path = VARIANT_ROBOT_USD
        _marker("make_runtime_env", "start")
        inner = _make_runtime_env(args)
        _marker("make_runtime_env", "complete", {"pass": True})
        timeline = omni.timeline.get_timeline_interface()
        reset_before = {
            "custom_step_counter": int(inner._sim_step_counter),
            "simulation_clock": d351._simulation_clock(inner),
            "timeline_time_s": float(timeline.get_current_time()),
            "timeline_playing": bool(timeline.is_playing()),
            "timeline_stopped": bool(timeline.is_stopped()),
        }
        _marker("reset", "start")
        inner.reset(seed=SEED)
        reset_after = {
            "custom_step_counter": int(inner._sim_step_counter),
            "simulation_clock": d351._simulation_clock(inner),
            "timeline_time_s": float(timeline.get_current_time()),
            "timeline_playing": bool(timeline.is_playing()),
            "timeline_stopped": bool(timeline.is_stopped()),
        }
        reset_internal_transition = {
            "before": reset_before,
            "after": reset_after,
            "controlled_d362_sample_count": 0,
            "semantics": "reset-internal work is reported separately and excluded from the 200+300 controlled rows",
        }
        _marker("reset", "complete", reset_internal_transition)
        settings = carb.settings.get_settings()
        previous_physx = settings.get(PHYSX_COLLIDER_SETTING)
        previous_play = settings.get(PLAY_SIMULATIONS_SETTING)
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
        reset_pause = _pause_timeline(inner, timeline)
        if not reset_pause["pass"]:
            raise RuntimeError("D362 reset pause/commit bridge failed")
        settings.set(PHYSX_COLLIDER_SETTING, 2)

        stage_contract = d351.d333._stage_contract(inner)
        sensor_contract, filter_map = d351.d333._sensor_contract(inner)
        actuator_contract = _actuator_contract(inner)
        object_contract = _object_spawn_contract(inner)
        _marker("corrected_d348_audit", "start")
        corrected = d351.d349._corrected_live_audit()
        _marker("corrected_d348_audit", "complete", {"pass": corrected.get("pass")})
        _marker("live_64_plus_64_binding", "start")
        topology_parts, live_binding = d351.d349._build_live_topology_parts(inner)
        _marker("live_64_plus_64_binding", "complete", {"pass": live_binding.get("pass")})
        part_counts = {body: len(topology_parts[body]) for body in ("link5", "gripper_link")}
        capacity_contract = _runtime_capacity_contract(
            inner, sensor_contract, topology_parts
        )
        _marker(
            "runtime_contact_capacity",
            "complete",
            {
                "pass": capacity_contract.get("pass"),
                "configured_capacity": capacity_contract.get(
                    "sensor_cfg_max_contact_data_count_per_prim"
                ),
                "derived_capacity": capacity_contract.get(
                    "derived_total_capacity"
                ),
            },
        )
        runtime_checks = {
            "stage_sole_support": stage_contract.get("hard_contract_pass") is True,
            "sensor_four_filters": sensor_contract.get("hard_contract_pass") is True,
            "actuators_frozen_80_4_2p5_3p14": actuator_contract["pass"],
            "object_and_dt_frozen": object_contract["pass"],
            "corrected_d348_128_of_128": corrected.get("pass") is True
            and corrected.get("checks", {}).get("all_parts_corrected_pass_128_of_128") is True,
            "live_binding_64_plus_64": live_binding.get("pass") is True
            and part_counts == {"link5": 64, "gripper_link": 64},
            "runtime_capacity_33280_exact": capacity_contract.get("pass") is True,
            "joint_order_exact": list(inner._robot.joint_names) == list(d351.d332.ALL_JOINT_NAMES),
            "counter_zero_before_physics": int(inner._sim_step_counter) == 0,
        }
        prerequisites = {
            "artifact": "D362_RUNTIME_PREREQUISITES_V1",
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "actuator_contract": actuator_contract,
            "object_contract": object_contract,
            "corrected_d348": corrected,
            "live_binding": live_binding,
            "live_part_counts": part_counts,
            "runtime_contact_capacity": capacity_contract,
            "reset_internal_transition": reset_internal_transition,
            "checks": runtime_checks,
            "pass": all(runtime_checks.values()),
        }
        _write_json_x(PREREQUISITE_PATH, prerequisites)
        if not prerequisites["pass"]:
            raise RuntimeError(f"D362 runtime prerequisites STOP: {runtime_checks}")

        open_target = d351.d332._write_exact_state(
            inner,
            Q_FROZEN_OPEN_F32.astype(np.float64),
            OBJECT_POS_F32.astype(np.float64),
        )
        q5_index = list(inner._robot.joint_names).index(d351.d332.GRIPPER_JOINT_NAME)
        initial_q = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        initial_q_vel = (
            inner._robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
        )
        initial_obj_pos, initial_obj_quat = d351.d334._object_pose_w(inner)
        initial_obj_lin_vel = (
            inner._sponge.data.root_lin_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        initial_obj_ang_vel = (
            inner._sponge.data.root_ang_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        initial_state_checks = {
            "full_q_float32_exact": initial_q.tobytes() == Q_FROZEN_OPEN_F32.tobytes(),
            "q0_q4_exact": initial_q[:5].tobytes() == Q_FROZEN_OPEN_F32[:5].tobytes(),
            "q5_open_exact": initial_q[q5_index].tobytes() == Q5_OPEN_F32.tobytes(),
            "object_position_float32_exact": np.asarray(initial_obj_pos, dtype=np.float32).tobytes()
            == OBJECT_POS_F32.tobytes(),
            "object_quaternion_float32_exact": np.asarray(initial_obj_quat, dtype=np.float32).tobytes()
            == OBJECT_QUAT_F32.tobytes(),
            "joint_velocity_float32_zero_exact": initial_q_vel.tobytes()
            == np.zeros_like(initial_q_vel).tobytes(),
            "object_linear_velocity_float32_zero_exact": initial_obj_lin_vel.tobytes()
            == np.zeros_like(initial_obj_lin_vel).tobytes(),
            "object_angular_velocity_float32_zero_exact": initial_obj_ang_vel.tobytes()
            == np.zeros_like(initial_obj_ang_vel).tobytes(),
            "counter_zero": int(inner._sim_step_counter) == 0,
        }
        if not all(initial_state_checks.values()):
            raise RuntimeError(f"D362 exact frozen OPEN state failed: {initial_state_checks}")
        _initialize_prefix_writer()
        baseline_object_ref = np.asarray(initial_obj_pos, dtype=np.float64)
        root_ref_pos, root_ref_quat = d351.d333._root_pose(inner)
        captures: dict[str, Any] = {
            "initial": _capture_pair("initial", simulation_app, inner, timeline),
            "start": None,
            "contact": None,
            "motion": None,
            "final": None,
        }
        capture_event_metadata: dict[str, Any] = {"contact": None, "motion": None}
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
        resume = _resume_timeline(inner, timeline)
        if not resume["pass"]:
            raise RuntimeError("D362 could not resume timeline for OPEN baseline")

        baseline_rows: list[dict[str, Any]] = []
        for step in range(BASELINE_STEPS):
            _begin_prefix_step(
                inner, timeline, "frozen_open_baseline", step
            )
            timeline_step = _physics_step_checked(inner, timeline)
            row = _state_row(
                inner,
                phase="frozen_open_baseline",
                phase_step=step,
                target=open_target,
                reference_object_pos_w=baseline_object_ref,
                reference_object_quat_wxyz=np.asarray(initial_obj_quat, dtype=np.float64),
                reference_root_pos_w=root_ref_pos,
                reference_root_quat_wxyz=root_ref_quat,
                q5_index=q5_index,
                filter_map=filter_map,
            )
            row["timeline_time_s"] = timeline_step["after"]["time_s"]
            _observe_prefix_step(row, inner, filter_map)
            row["timeline_step_contract"] = timeline_step
            baseline_rows.append(row)
            confirmed_labels = (
                _confirmed_robot_labels(baseline_rows[-2:])
                if len(baseline_rows) >= 2 and captures["contact"] is None
                else []
            )
            motion_confirmed = bool(
                len(baseline_rows) >= 2
                and captures["motion"] is None
                and _motion_confirmed_pair(baseline_rows[-2:])
            )
            if confirmed_labels:
                captures["contact"] = _capture_pair(
                    "contact", simulation_app, inner, timeline
                )
                capture_event_metadata["contact"] = {
                    "phase": "frozen_open_baseline",
                    "onset_phase_step": step - 1,
                    "confirmation_phase_step": step,
                    "qualifying_body_labels": confirmed_labels,
                }
            if motion_confirmed:
                captures["motion"] = _capture_pair(
                    "motion", simulation_app, inner, timeline
                )
                capture_event_metadata["motion"] = {
                    "phase": "frozen_open_baseline",
                    "onset_phase_step": step - 1,
                    "confirmation_phase_step": step,
                }
            if (confirmed_labels or motion_confirmed) and step + 1 < BASELINE_STEPS:
                inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
                resume = _resume_timeline(inner, timeline)
                if not resume["pass"]:
                    raise RuntimeError(
                        "D362 failed to resume after guarded baseline-event capture"
                    )
            if step == 0 or (step + 1) % 10 == 0:
                _marker("frozen_open_baseline", "progress", {"completed": step + 1, "requested": BASELINE_STEPS})
        baseline = _baseline_statistics(
            baseline_rows,
            stage_contract_pass=stage_contract.get("hard_contract_pass") is True,
            sensor_contract_pass=sensor_contract.get("hard_contract_pass") is True,
        )
        if not baseline["pass"]:
            _seal_prefix("open_baseline_hard_gate_fail_stop")
        captures["start"] = _capture_pair("start", simulation_app, inner, timeline)

        closure_rows: list[dict[str, Any]] = []
        close_target = open_target
        if baseline["pass"]:
            closure_object_ref = inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            closure_object_quat_ref = (
                inner._sponge.data.root_quat_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            close_target = _set_closed_q5_target(inner, open_target, q5_index)
            inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
            resume = _resume_timeline(inner, timeline)
            if not resume["pass"]:
                raise RuntimeError("D362 could not resume timeline for closure")
            for step in range(CLOSURE_MAX_STEPS):
                _begin_prefix_step(
                    inner, timeline, "q5_close_observation", step
                )
                timeline_step = _physics_step_checked(inner, timeline)
                row = _state_row(
                    inner,
                    phase="q5_close_observation",
                    phase_step=step,
                    target=close_target,
                    reference_object_pos_w=closure_object_ref,
                    reference_object_quat_wxyz=closure_object_quat_ref,
                    reference_root_pos_w=root_ref_pos,
                    reference_root_quat_wxyz=root_ref_quat,
                    q5_index=q5_index,
                    filter_map=filter_map,
                )
                row["timeline_time_s"] = timeline_step["after"]["time_s"]
                _observe_prefix_step(row, inner, filter_map)
                row["timeline_step_contract"] = timeline_step
                closure_rows.append(row)
                confirmed_labels = (
                    _confirmed_robot_labels(closure_rows[-2:])
                    if len(closure_rows) >= 2 and captures["contact"] is None
                    else []
                )
                motion_confirmed = bool(
                    len(closure_rows) >= 2
                    and captures["motion"] is None
                    and _motion_confirmed_pair(closure_rows[-2:])
                )
                if confirmed_labels:
                    captures["contact"] = _capture_pair("contact", simulation_app, inner, timeline)
                    capture_event_metadata["contact"] = {
                        "phase": "q5_close_observation",
                        "onset_phase_step": step - 1,
                        "confirmation_phase_step": step,
                        "qualifying_body_labels": confirmed_labels,
                    }
                if motion_confirmed:
                    captures["motion"] = _capture_pair("motion", simulation_app, inner, timeline)
                    capture_event_metadata["motion"] = {
                        "phase": "q5_close_observation",
                        "onset_phase_step": step - 1,
                        "confirmation_phase_step": step,
                    }
                if (confirmed_labels or motion_confirmed) and step + 1 < CLOSURE_MAX_STEPS:
                    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
                    resume = _resume_timeline(inner, timeline)
                    if not resume["pass"]:
                        raise RuntimeError("D362 failed to resume after guarded decision capture")
                if step == 0 or (step + 1) % 10 == 0:
                    _marker("q5_close_observation", "progress", {"completed": step + 1, "requested": CLOSURE_MAX_STEPS})
            _seal_prefix("full_500_step_horizon_complete")
        else:
            _marker("q5_close_observation", "not_run", {"reason": "frozen OPEN baseline hard gate failed"})

        captures["final"] = _capture_pair("final", simulation_app, inner, timeline)
        closure = _closure_statistics(
            closure_rows,
            baseline["pass"],
            baseline_end_q5_actual=float(baseline_rows[-1]["q5_actual_rad"]),
            baseline_robot_contact=baseline["precommand_robot_contact_confound"],
            baseline_precommand_motion=baseline[
                "precommand_object_motion_confound"
            ],
        )
        all_rows = [*baseline_rows, *closure_rows]
        _annotate_event_masks(all_rows)
        qualifying_point_contract = _qualifying_robot_point_contract(all_rows)
        any_robot_confirmation_in_trace = any(
            row["event_masks"]["two_step_any_robot_confirmation_end"]
            for row in all_rows
        )
        motion_confirmation_in_trace = any(
            row["event_masks"]["two_step_motion_confirmation_end"]
            for row in all_rows
        )
        q0_q4_drift = _q0_q4_drift_summary(baseline_rows, closure_rows, closure)
        prefix_audit = _finalize_prefix_audit(all_rows)
        _write_json_x(TRACE_JSON_PATH, all_rows)
        _write_trace_csv(TRACE_CSV_PATH, all_rows)
        rerun = _write_rerun(topology_parts, all_rows, baseline, closure)
        video = _build_trace_replay_video(topology_parts, all_rows)
        sheet = _build_beginner_sheet(
            captures, capture_event_metadata, baseline, closure, all_rows
        )
        capture_reports = {
            role: None
            if captures[role] is None
            else {
                "primary": _png_report(CAPTURE_PATHS[role]),
                "opposite": _png_report(OPPOSITE_CAPTURE_PATHS[role]),
            }
            for role in CAPTURE_PATHS
        }
        settings.set(PHYSX_COLLIDER_SETTING, previous_physx if previous_physx is not None else 0)
        if previous_play is None:
            settings.destroy_item(PLAY_SIMULATIONS_SETTING)
        else:
            settings.set(PLAY_SIMULATIONS_SETTING, previous_play)
        restore_checks = {
            "timeline_final_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "physx_setting_restored": settings.get(PHYSX_COLLIDER_SETTING) == previous_physx
            if previous_physx is not None
            else settings.get(PHYSX_COLLIDER_SETTING) in (None, 0),
            "play_setting_restored": settings.get(PLAY_SIMULATIONS_SETTING) == previous_play,
        }
        observability_checks = {
            "initial_start_final_two_view_actual_isaac_png": all(
                capture_reports[role][view]["pass"]
                for role in ("initial", "start", "final")
                for view in ("primary", "opposite")
            ),
            "contact_png_iff_confirmed": (captures["contact"] is not None)
            == (capture_event_metadata["contact"] is not None)
            == any_robot_confirmation_in_trace,
            "motion_png_iff_confirmed": (captures["motion"] is not None)
            == (capture_event_metadata["motion"] is not None)
            == motion_confirmation_in_trace,
            "conditional_event_capture_pairs_decode": all(
                capture_reports[role] is None
                or all(capture_reports[role][view]["pass"] for view in ("primary", "opposite"))
                for role in ("contact", "motion")
            ),
            "all_qualifying_robot_force_points_finite_for_observability": qualifying_point_contract[
                "all_qualifying_robot_event_points_finite"
            ],
            "rerun_rrd_rbl_verify_screenshot": rerun.get("pass") is True,
            "durable_prefix_exact_reconciliation": prefix_audit.get("pass") is True,
            "trace_replay_video_and_storyboard": video.get("pass") is True,
            "beginner_sheet": sheet["pass"],
        }
        operational_trace_checks = {
            "trace_exact_count": len(all_rows) == BASELINE_STEPS + len(closure_rows),
            "all_trace_rows_finite": all(row["finite"] for row in all_rows),
            "timeline_playing_not_stopped_and_nondecreasing_each_step": all(
                row["timeline_step_contract"]["before"]["playing"]
                and not row["timeline_step_contract"]["before"]["stopped"]
                and row["timeline_step_contract"]["after"]["playing"]
                and not row["timeline_step_contract"]["after"]["stopped"]
                and row["timeline_step_contract"]["delta_s_diagnostic"] >= 0.0
                for row in all_rows
            ),
            "q0_q4_target_float32_bits_frozen_every_step": all(
                np.asarray(row["target_joint_rad"][:5], dtype=np.float32).tobytes()
                == Q_FROZEN_OPEN_F32[:5].tobytes()
                for row in all_rows
            ),
            "registered_q5_effort_limit_2p5_every_step": all(
                float(row["q5_torque"]["registered_effort_limit_nm"])
                == ACTUATOR_EFFORT_LIMIT_NM
                for row in all_rows
            ),
            "controlled_steps_exact": _CONTROLLED_STEPS == len(all_rows),
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT == (1 if baseline["pass"] else 0),
            "closure_horizon_exact_if_baseline_pass": len(closure_rows) == CLOSURE_MAX_STEPS
            if baseline["pass"]
            else len(closure_rows) == 0,
            "inputs_unchanged": _input_hashes() == prereg["input_hashes"],
            "sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "settings_restored": all(restore_checks.values()),
            "runtime_contact_capacity_contract": capacity_contract.get("pass")
            is True,
            "durable_prefix_contract": prefix_audit.get("pass") is True,
        }
        summary = {
            "artifact": "D362_WORKER_SUMMARY_V1",
            "case": CASE,
            "new_variables": NEW_VARIABLES,
            "launcher": launcher_report,
            "runtime_prerequisites_path": _rel(PREREQUISITE_PATH),
            "runtime_contact_capacity": capacity_contract,
            "durable_prefix_audit": prefix_audit,
            "reset_internal_transition": reset_internal_transition,
            "initial_state_checks": initial_state_checks,
            "baseline": baseline,
            "closure": closure,
            "q0_q4_actual_drift": q0_q4_drift,
            "captures": captures,
            "capture_event_metadata": capture_event_metadata,
            "capture_reports": capture_reports,
            "qualifying_robot_point_contract": qualifying_point_contract,
            "beginner_sheet": sheet,
            "trace_replay_video": video,
            "rerun_validation_path": _rel(RERUN_VALIDATION_PATH),
            "controlled_physics_steps": _CONTROLLED_STEPS,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "restore_checks": restore_checks,
            "operational_trace_checks": operational_trace_checks,
            "observability_checks": observability_checks,
            "observability_artifact_pass": all(observability_checks.values()),
            "interface_visibility": None,
            "interface_visibility_pending_manual_original_resolution_inspection": True,
            "target_ik_path_changed": False,
            "asset_decomposition_gate_material_mass_actuator_physics_changed": False,
            "exact_manifold_or_face": None,
            "cap_rim_science": None,
            "force_closure_or_grasp": None,
            "g0a_pass": False,
            "pass": all(operational_trace_checks.values()),
        }
        _write_json_x(WORKER_SUMMARY_PATH, summary)
        _marker("worker_summary", "complete", {"pass": summary["pass"], "verdict": closure["verdict"]})
        return 0 if summary["pass"] else 2
    except Exception as error:
        failure_prefix_audit: dict[str, Any] | None = None
        if PREFIX_PATH.is_file() and not FAILURE_PREFIX_AUDIT_PATH.exists():
            try:
                _configure_d361_runtime_prefix_schema()
                raw_audit, observations = d361.verify_prefix(PREFIX_PATH)
                failure_prefix_audit = {
                    "artifact": "D362_FAILURE_PREFIX_AUDIT_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "prefix_path": _rel(PREFIX_PATH),
                    "prefix_sha256": _sha(PREFIX_PATH),
                    "audit": raw_audit,
                    "observation_count": len(observations),
                    "controlled_physics_steps": _CONTROLLED_STEPS,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "resume_allowed": False,
                    "overwrite_allowed": False,
                    "g0a_pass": False,
                }
                _write_json_x(FAILURE_PREFIX_AUDIT_PATH, failure_prefix_audit)
            except Exception as audit_error:
                failure_prefix_audit = {
                    "artifact": "D362_FAILURE_PREFIX_AUDIT_WRITE_ERROR_V1",
                    "error": f"{type(audit_error).__name__}: {audit_error}",
                }
        if not WORKER_EXCEPTION_PATH.exists():
            _write_json_x(
                WORKER_EXCEPTION_PATH,
                {
                    "artifact": "D362_WORKER_EXCEPTION_STOP_V1",
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "controlled_physics_steps": _CONTROLLED_STEPS,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "failure_prefix_audit": failure_prefix_audit,
                    "automatic_retry": False,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        if _PREFIX_WRITER is not None:
            try:
                _PREFIX_WRITER.close()
            except Exception:
                pass
        if settings is not None:
            try:
                if previous_physx is None:
                    settings.destroy_item(PHYSX_COLLIDER_SETTING)
                else:
                    settings.set(PHYSX_COLLIDER_SETTING, previous_physx)
                if previous_play is None:
                    settings.destroy_item(PLAY_SIMULATIONS_SETTING)
                else:
                    settings.set(PLAY_SIMULATIONS_SETTING, previous_play)
            except Exception:
                pass
        if inner is not None:
            try:
                inner.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass


def _run(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D362 preregistration did not pass")
    preinvocation_checks = {
        "head_origin_base_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
        "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
        "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
        "git_scope_exact": _status_scope_ok(_git_status()),
    }
    if not all(preinvocation_checks.values()):
        raise RuntimeError(
            f"D362 pre-invocation immutable contract failed: {preinvocation_checks}"
        )
    if INVOCATION_PATH.exists() or WORKER_LOG_PATH.exists():
        raise RuntimeError("D362 single invocation already consumed; no retry")
    preinvocation_inventory = _output_file_inventory()
    expected_preinvocation_inventory = sorted([PREPARE_PATH.name, PREREG_PATH.name])
    if preinvocation_inventory != expected_preinvocation_inventory:
        raise RuntimeError(
            "D362 pre-invocation inventory drift: "
            f"expected={expected_preinvocation_inventory}, actual={preinvocation_inventory}"
        )
    resources = _gpu_snapshot()
    if int(resources.get("memory_free_mib", 0)) < MIN_GPU_FREE_MIB or int(
        resources.get("ram_available_bytes", 0)
    ) < MIN_RAM_AVAILABLE_BYTES:
        print(
            json.dumps(
                {
                    "stage": "run_resource_gate_before_invocation",
                    "pass": False,
                    "invocation_marker_written": False,
                    "resources": resources,
                },
                ensure_ascii=False,
            )
        )
        return 2
    token = secrets.token_hex(32)
    invocation = {
        "artifact": "D362_SINGLE_ISAAC_INVOCATION_MARKER_V1",
        "utc": _utc_now(),
        "run_nonce": prereg["run_nonce"],
        "invocation_index": 1,
        "supervisor_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "preregistration_sha256": _sha(PREREG_PATH),
        "preinvocation_inventory": preinvocation_inventory,
        "preinvocation_checks": preinvocation_checks,
        "automatic_retry": False,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    command = [
        REGISTERED_PYTHON,
        str(HARNESS),
        "--stage",
        "_worker",
        "--out_dir",
        str(OUT_DIR),
        "--seed",
        str(SEED),
    ]
    env = os.environ.copy()
    env.update(
        {
            "DISPLAY": ":1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
            WORKER_TOKEN_ENV: token,
            SUPERVISOR_PID_ENV: str(os.getpid()),
            "PATH": f"{RERUN_CLI.parent}:{env.get('PATH', '')}",
        }
    )
    start = time.monotonic()
    last_progress = start
    last_sizes = (-1, -1, -1)
    watchdog_triggered = False
    watchdog_reason = None
    telemetry: list[dict[str, Any]] = []
    with WORKER_LOG_PATH.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            sizes = (
                WORKER_LOG_PATH.stat().st_size if WORKER_LOG_PATH.exists() else 0,
                PHASE_PATH.stat().st_size if PHASE_PATH.exists() else 0,
                PREFIX_PATH.stat().st_size if PREFIX_PATH.exists() else 0,
            )
            if sizes != last_sizes:
                last_sizes = sizes
                last_progress = time.monotonic()
            now = time.monotonic()
            elapsed = now - start
            idle = now - last_progress
            try:
                sample = _gpu_snapshot()
            except Exception as error:
                sample = {"telemetry_error": f"{type(error).__name__}: {error}"}
            telemetry.append({"elapsed_seconds": elapsed, "idle_seconds": idle, **sample})
            if elapsed > TOTAL_WATCHDOG_S or idle > INACTIVITY_WATCHDOG_S:
                watchdog_triggered = True
                watchdog_reason = "total" if elapsed > TOTAL_WATCHDOG_S else "inactivity"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10.0)
                break
            time.sleep(1.0)
        exit_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    contact_overflow_warning_audit = _write_contact_overflow_warning_audit()
    supervisor_prefix_recovery = _write_supervisor_prefix_recovery_audit(
        exit_code
    )
    worker_summary_file = _safe_json_file(WORKER_SUMMARY_PATH)
    worker_for_audit = (
        worker_summary_file["payload"]
        if worker_summary_file["parse_pass"] is True
        else None
    )
    postworker_inventory = _output_file_inventory()
    expected_postworker_inventory = (
        _expected_postworker_inventory(worker_for_audit)
        if worker_for_audit is not None
        else None
    )
    postworker_inventory_exact = bool(
        expected_postworker_inventory is not None
        and postworker_inventory == expected_postworker_inventory
    )
    postworker_missing = sorted(
        set(expected_postworker_inventory or []) - set(postworker_inventory)
    )
    postworker_unexpected = sorted(
        set(postworker_inventory) - set(expected_postworker_inventory or [])
    )
    postworker_core_missing = sorted(
        _core_postworker_inventory() - set(postworker_inventory)
    )
    postworker_inventory_integrity_pass = bool(
        expected_postworker_inventory is not None
        and not postworker_missing
        and not postworker_unexpected
        and not postworker_core_missing
    )
    postworker_hash_manifest = _inventory_hashes(postworker_inventory)
    phase_contract = (
        _phase_contract(worker_for_audit)
        if worker_for_audit is not None and PHASE_PATH.is_file()
        else {"pass": False, "error": "worker summary or phase stream missing"}
    )
    supervisor = {
        "artifact": "D362_SUPERVISOR_SUMMARY_V1",
        "case": CASE,
        "command": command,
        "worker_pid": process.pid,
        "worker_exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - start,
        "watchdog_triggered": watchdog_triggered,
        "watchdog_reason": watchdog_reason,
        "automatic_retry": False,
        "telemetry_sample_count": len(telemetry),
        "telemetry": telemetry,
        "worker_summary_exists": WORKER_SUMMARY_PATH.is_file(),
        "worker_summary_file": worker_summary_file,
        "worker_exception_exists": WORKER_EXCEPTION_PATH.is_file(),
        "supervisor_prefix_recovery_audit": supervisor_prefix_recovery,
        "supervisor_prefix_recovery_audit_sha256": _sha(
            SUPERVISOR_PREFIX_AUDIT_PATH
        ),
        "contact_overflow_warning_audit": contact_overflow_warning_audit,
        "contact_overflow_warning_audit_sha256": _sha(
            CONTACT_OVERFLOW_WARNING_AUDIT_PATH
        ),
        "expected_postworker_inventory": expected_postworker_inventory,
        "postworker_inventory": postworker_inventory,
        "postworker_inventory_exact": postworker_inventory_exact,
        "postworker_missing_expected_artifacts": postworker_missing,
        "postworker_unexpected_artifacts": postworker_unexpected,
        "postworker_core_missing": postworker_core_missing,
        "postworker_inventory_integrity_pass": postworker_inventory_integrity_pass,
        "postworker_hash_manifest": postworker_hash_manifest,
        "phase_contract": phase_contract,
        "pass": exit_code == 0
        and not watchdog_triggered
        and WORKER_SUMMARY_PATH.is_file()
        and worker_summary_file["parse_pass"] is True
        and not WORKER_EXCEPTION_PATH.exists()
        and supervisor_prefix_recovery.get("success_path_pass") is True
        and contact_overflow_warning_audit.get("pass") is True
        and postworker_inventory_integrity_pass
        and phase_contract.get("pass") is True,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    if not supervisor["pass"]:
        print(json.dumps({"stage": "run", "pass": False, "supervisor": _rel(SUPERVISOR_PATH)}, ensure_ascii=False))
        return 2
    worker = _json(WORKER_SUMMARY_PATH)
    automated_checks = {
        "supervisor_pass": supervisor["pass"],
        "worker_pass": worker.get("pass") is True,
        "worker_observability_artifact_pass": worker.get(
            "observability_artifact_pass"
        )
        is True,
        "supervisor_prefix_recovery_pass": supervisor.get(
            "supervisor_prefix_recovery_audit", {}
        ).get("success_path_pass")
        is True,
        "physx_incomplete_contact_warning_absent": supervisor.get(
            "contact_overflow_warning_audit", {}
        ).get("pass")
        is True,
        "one_invocation_no_retry": invocation["invocation_index"] == 1
        and invocation["automatic_retry"] is False,
        "manual_interface_visibility_pending": worker.get("interface_visibility") is None
        and worker.get("interface_visibility_pending_manual_original_resolution_inspection") is True,
        "g0a_false": worker.get("g0a_pass") is False,
        "sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
    }
    automated = {
        "artifact": "D362_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "operational_verdict": (
            "D362_AUTOMATED_PHYSX_CONTACT_MOTION_COMPLETE_PENDING_MANUAL_VISIBILITY"
            if all(automated_checks.values())
            else "D362_AUTOMATED_CONTRACT_FAIL_STOP"
        ),
        "physical_verdict_before_manual_visibility": worker["closure"]["verdict"],
        "baseline": worker["baseline"],
        "closure": worker["closure"],
        "controlled_physics_steps": worker["controlled_physics_steps"],
        "observability_artifact_pass_before_manual": worker.get(
            "observability_artifact_pass"
        )
        is True,
        "observability_checks": worker.get("observability_checks", {}),
        "supervisor_sha256": _sha(SUPERVISOR_PATH),
        "postworker_hash_manifest": supervisor["postworker_hash_manifest"],
        "phase_contract": supervisor["phase_contract"],
        "contact_overflow_warning_audit": supervisor[
            "contact_overflow_warning_audit"
        ],
        "interface_visibility": None,
        "manual_original_resolution_inspection_required": True,
        "target_ik_path_changed": False,
        "g0a_pass": False,
        "checks": automated_checks,
        "pass": all(automated_checks.values()),
    }
    _write_json_x(AUTOMATED_PATH, automated)
    print(
        json.dumps(
            {
                "stage": "run",
                "pass": automated["pass"],
                "physical_verdict": automated["physical_verdict_before_manual_visibility"],
                "manual_visibility_pending": True,
            },
            ensure_ascii=False,
        )
    )
    return 0 if automated["pass"] else 2


def _finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D362 completion already exists")
    automated = _json(AUTOMATED_PATH)
    worker = _json(WORKER_SUMMARY_PATH)
    manual = _json(MANUAL_PATH)
    expected_precompletion_inventory = _expected_precompletion_inventory(worker)
    precompletion_inventory = _output_file_inventory()
    precompletion_inventory_exact = (
        precompletion_inventory == expected_precompletion_inventory
    )
    precompletion_missing = sorted(
        set(expected_precompletion_inventory) - set(precompletion_inventory)
    )
    precompletion_unexpected = sorted(
        set(precompletion_inventory) - set(expected_precompletion_inventory)
    )
    precompletion_core = {
        *_core_postworker_inventory(),
        SUPERVISOR_PATH.name,
        AUTOMATED_PATH.name,
        MANUAL_PATH.name,
    }
    precompletion_core_missing = sorted(
        precompletion_core - set(precompletion_inventory)
    )
    precompletion_inventory_integrity_pass = bool(
        not precompletion_missing
        and not precompletion_unexpected
        and not precompletion_core_missing
    )
    postworker_manifest = automated.get("postworker_hash_manifest", {})
    postworker_manifest_recheck = bool(postworker_manifest) and all(
        (OUT_DIR / name).is_file() and _sha(OUT_DIR / name) == digest
        for name, digest in postworker_manifest.items()
    )
    phase_recheck = _phase_contract(worker)
    expected_paths = {
        _rel(CAPTURE_PATHS["initial"]),
        _rel(OPPOSITE_CAPTURE_PATHS["initial"]),
        _rel(CAPTURE_PATHS["start"]),
        _rel(OPPOSITE_CAPTURE_PATHS["start"]),
        _rel(CAPTURE_PATHS["final"]),
        _rel(OPPOSITE_CAPTURE_PATHS["final"]),
        _rel(RERUN_PNG_PATH),
        _rel(SHEET_PATH),
        _rel(VIDEO_PATH),
        _rel(VIDEO_REPORT_PATH),
        _rel(VIDEO_STORYBOARD_PATH),
        _rel(PREFIX_PATH),
        _rel(PREFIX_AUDIT_PATH),
        _rel(SUPERVISOR_PREFIX_AUDIT_PATH),
        _rel(CONTACT_OVERFLOW_WARNING_AUDIT_PATH),
        _rel(TRACE_JSON_PATH),
        _rel(TRACE_CSV_PATH),
        _rel(WORKER_SUMMARY_PATH),
        _rel(AUTOMATED_PATH),
        _rel(SUPERVISOR_PATH),
        _rel(PHASE_PATH),
        _rel(RERUN_VALIDATION_PATH),
    }
    for role in ("contact", "motion"):
        if worker["captures"].get(role) is not None:
            expected_paths.add(_rel(CAPTURE_PATHS[role]))
            expected_paths.add(_rel(OPPOSITE_CAPTURE_PATHS[role]))
    actual_capture_paths = {
        path
        for path in expected_paths
        if path.endswith(".png")
        and path
        not in {
            _rel(RERUN_PNG_PATH),
            _rel(SHEET_PATH),
            _rel(VIDEO_STORYBOARD_PATH),
        }
    }
    visibility_by_path = manual.get("interface_visibility_by_actual_capture", {})
    role_visibility: dict[str, dict[str, Any]] = {}
    for role in CAPTURE_PATHS:
        if worker["captures"].get(role) is None:
            continue
        primary = _rel(CAPTURE_PATHS[role])
        opposite = _rel(OPPOSITE_CAPTURE_PATHS[role])
        role_visibility[role] = {
            "primary": visibility_by_path.get(primary),
            "opposite": visibility_by_path.get(opposite),
            "at_least_one_view_visible": visibility_by_path.get(primary) is True
            or visibility_by_path.get(opposite) is True,
        }
    interface_visibility_contract = bool(
        set(visibility_by_path) == actual_capture_paths
        and all(isinstance(value, bool) for value in visibility_by_path.values())
        and all(row["at_least_one_view_visible"] for row in role_visibility.values())
    )
    manual_checks = {
        "artifact_exact": manual.get("artifact") == "D362_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "automated_summary_sha256_exact": manual.get(
            "automated_summary_sha256"
        )
        == _sha(AUTOMATED_PATH),
        "all_required_paths_inspected": set(manual.get("inspected_paths", [])) == expected_paths,
        "moving_gripper_edge_visible": manual.get("moving_gripper_edge_visible") is True,
        "cylinder_silhouette_visible": manual.get("cylinder_silhouette_visible") is True,
        "interface_visibility_entries_exact": set(visibility_by_path)
        == actual_capture_paths,
        "at_least_one_view_per_actual_state_shows_interface": interface_visibility_contract,
        "force_motion_timeline_legible": manual.get("force_motion_timeline_legible") is True,
        "no_text_overlap_on_beginner_sheet": manual.get("no_text_overlap_on_beginner_sheet") is True,
        "trace_replay_video_playback_inspected": manual.get(
            "trace_replay_video_playback_inspected"
        )
        is True,
        "video_first_middle_last_labels_legible": manual.get(
            "video_first_middle_last_labels_legible"
        )
        is True,
        "storyboard_no_text_overlap": manual.get(
            "storyboard_no_text_overlap"
        )
        is True,
        "link4_marker_only_scope_explicit": manual.get(
            "link4_marker_only_scope_explicit"
        )
        is True,
        "video_report_pass_verified": manual.get("video_report_pass_verified")
        is True,
        "durable_prefix_audits_pass_verified": manual.get(
            "durable_prefix_audits_pass_verified"
        )
        is True,
        "contact_overflow_warning_absence_verified": manual.get(
            "contact_overflow_warning_absence_verified"
        )
        is True,
        "manual_pass": manual.get("pass") is True,
    }
    visual_checks = {
        "automated_pass": automated.get("pass") is True,
        "automated_observability_artifact_pass": automated.get(
            "observability_artifact_pass_before_manual"
        )
        is True,
        "manual_original_resolution_visibility_pass": all(manual_checks.values()),
        "rrd_validation_pass": _json(RERUN_VALIDATION_PATH).get("pass") is True,
    }
    integrity_checks = {
        "precompletion_inventory_integrity_pass": precompletion_inventory_integrity_pass,
        "postworker_hash_manifest_recheck": postworker_manifest_recheck,
        "supervisor_sha256_matches_automated": automated.get("supervisor_sha256")
        == _sha(SUPERVISOR_PATH),
        "phase_contract_recheck": phase_recheck.get("pass") is True
        and phase_recheck.get("phase_sha256")
        == automated.get("phase_contract", {}).get("phase_sha256"),
        "input_hashes_unchanged": _input_hashes() == _json(PREREG_PATH)["input_hashes"],
        "d334_sidecar_unchanged": _sidecar_hashes() == _json(PREREG_PATH)["d334_sidecar_before"],
        "g0a_false": automated.get("g0a_pass") is False,
    }
    integrity_pass = all(integrity_checks.values())
    visual_pass = all(visual_checks.values())
    provisional_physical = automated["physical_verdict_before_manual_visibility"]
    if not integrity_pass:
        final_verdict = "D362_POSTRUN_INTEGRITY_OR_INVENTORY_FAIL_STOP"
        physical = None
    elif not visual_pass:
        final_verdict = "D362_OBSERVABILITY_FAIL_STOP"
        physical = provisional_physical
    else:
        final_verdict = provisional_physical
        physical = provisional_physical
    completion = {
        "artifact": "D362_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "final_verdict": final_verdict,
        "physical_verdict": physical,
        "provisional_physical_verdict_before_integrity": provisional_physical,
        "interface_visibility": True if interface_visibility_contract else False,
        "manual_checks": manual_checks,
        "interface_visibility_by_actual_capture": visibility_by_path,
        "interface_visibility_by_role": role_visibility,
        "controlled_physics_steps": automated["controlled_physics_steps"],
        "runtime_contact_capacity": worker.get("runtime_contact_capacity"),
        "durable_prefix_audit": worker.get("durable_prefix_audit"),
        "trace_replay_video": worker.get("trace_replay_video"),
        "contact_overflow_warning_audit": automated.get(
            "contact_overflow_warning_audit"
        ),
        "expected_precompletion_inventory": expected_precompletion_inventory,
        "precompletion_inventory": precompletion_inventory,
        "precompletion_inventory_exact": precompletion_inventory_exact,
        "precompletion_missing_expected_artifacts": precompletion_missing,
        "precompletion_unexpected_artifacts": precompletion_unexpected,
        "precompletion_core_missing": precompletion_core_missing,
        "precompletion_hash_manifest": _inventory_hashes(precompletion_inventory),
        "phase_contract_recheck": phase_recheck,
        "target_ik_path_changed": False,
        "exact_manifold_or_face": None,
        "cap_rim_science": None,
        "force_closure_or_grasp": None,
        "g0a_pass": False,
        "visual_checks": visual_checks,
        "integrity_checks": integrity_checks,
        "pass": integrity_pass and visual_pass,
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(json.dumps({"stage": "finalize", "pass": completion["pass"], "verdict": completion["final_verdict"]}, ensure_ascii=False))
    return 0 if completion["pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "_worker", "finalize"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D362 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D362 seed drift")
    if args.stage == "prepare":
        return _prepare(args)
    if args.stage == "run":
        return _run(args)
    if args.stage == "finalize":
        return _finalize(args)
    args.headless = False
    args.livestream = 0
    args.device = "cuda:0"
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
