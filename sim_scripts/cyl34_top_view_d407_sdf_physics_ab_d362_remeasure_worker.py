#!/usr/bin/env python3
"""D407: single-leg Isaac worker for the SDF-physics A/B re-measurement of D362.

This forward-only case changes exactly one variable relative to frozen D362:
the gripper_link collision representation (leg a = D344 attempt3 A64 64-part
control, leg b = D406 attempt1 authored derivative with link5 A64 64 kept
enabled, gripper A64 64 disabled, and one SDF res256 mesh enabled).  The q5
perturbation, scene, target, initial state, physics, thresholds, seed, and
200+300 step horizons remain the D362 contract verbatim, and the 14 inherited
science functions are byte-identical to D362 modulo the D362->D407 label
rename.  It never classifies an exact collider face, cap/rim order, force
closure, grasp, or G0a success.

This file holds ONLY the ``_worker`` stage plus its helpers.  Admission,
supervision (one worker per leg, retry 0, A before B), inter-leg policy, and
finalize are owned by the D407 controller.  The worker runs headless; the
D362 viewport-capture, MP4 trace-replay, and GUI launcher subsystems are
removed by design (session doc section 3.6.2).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import textwrap
import traceback
from pathlib import Path
from typing import Any

if not sys.dont_write_bytecode:
    raise RuntimeError(
        "D407 worker requires python -B (sys.dont_write_bytecode) before "
        "any third-party or project-local import"
    )

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


CASE = "g0a_d407"
CASE_NAME = "sdf_physics_ab_d362_remeasure"
NEW_VARIABLES = [
    "gripper_link_collision_representation_a64_to_sdf_res256_v1",
]
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
BASE_GIT = "a69a96d36219268e4bc5e25065cc234da9d99674"
EXPECTED_PREREG_SHA256 = "6deb6779a18619f547952de9119eee599ea5dd40ac466d57d6a813988afb1269"
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
RERUN_SCREENSHOT_LOGICAL_SIZE = "960x540"
RERUN_SCREENSHOT_PHYSICAL = [1920, 1080]
REPLAY_FRAME_SIZE = [1920, 1080]

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
# Both env-var names are leg-independent (design section 3.6.5: full worker
# preflight re-runs per leg in a fresh process).
WORKER_TOKEN_ENV = "D407_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D407_SUPERVISOR_PID"

ATTEMPT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d407/attempt1_sdf_physics_ab_d362_remeasure"
HARNESS = Path(__file__).resolve()
PREREG_PATH = ATTEMPT_DIR / "d407_preregistration.json"
SESSION_DOC = REPO / "claudedocs/session_20260728_grasp_g0a_d407_sdf_physics_ab_design_final_static_prep.md"
D354_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_closure_geometry_measurement.json"
D354_ATTESTATION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354/d354_zero_step_science_attestation.json"
D361_CAPACITY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_contact_capacity_budget.json"
D361_PROTOCOL = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361/d361_prefix_protocol_contract.json"
# The only two D360/D361-era pins still consumed (by _prefix_header wiring and
# _runtime_capacity_contract); every other D360/D361/D359 lineage pin retired.
D361_CAPACITY_SHA256 = "ca5edc818fee321dad257dcf3d1ba4574b6c446471a7ff7abc7c8ab790bf79f5"
D361_PROTOCOL_SHA256 = "dead1591fbf53f080bae0ca28643bb22147ee99a6b07f71a9285865e2c1d8e13"
D362_HARNESS = REPO / "sim_scripts/cyl34_top_view_d362_current_pose_capacity_prefix_integrated_physx_contact_motion.py"
D362_PREREG = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362/d362_preregistration.json"
D362_TRACE_JSON = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362/d362_physics_trace.json"
D362_WORKER_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362/d362_worker_summary.json"
D362_SUPERVISOR_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362/d362_supervisor_summary.json"
D360_HARNESS = REPO / "sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py"
ROARM_ENV_MODULE = REPO / "roarm_rl/roarm_cube_push_env.py"
RERUN_CONTRACT_MODULE = REPO / "roarm_rl/rerun_contract.py"
D348_EVIDENCE = d351.D348_EVIDENCE
D334_SUMMARY = d351.D334_SUMMARY
URDF_PATH = d351.d333.DEFAULT_URDF

SDF_SOURCE_STREAM_SHA256 = (
    "31aead25f7aa879a358a046bc01291ef2e260a2b367a990dacc255c17a2a5a31"
)
SDF_BODY_LOCAL_POINTS_F64_SHA256 = (
    "522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db"
)
SDF_BODY_LOCAL_BOUNDS_M = (
    (-0.010767397438303794, -0.009999632356670897, -0.0386173457368133),
    (0.06708260664084253, 0.015240367659608567, 0.0007502218245168529),
)

# Leg tables — single source of truth; every leg-dependent value comes from here.
LEG_A = "a"
LEG_B = "b"
LEG_ASSET_DIRS = {
    LEG_A: REPO / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts",
    LEG_B: REPO / "claudedocs/runtime_logs/grasp_track/g0a_d406/attempt1_d405_prereg_status_literal_repair/collision_asset/roarm_m3_link5_a64_gripper_sdf_res256",
}
LEG_OUT_SUBDIRS = {LEG_A: "leg_a_a64", LEG_B: "leg_b_sdf_res256"}
LEG_CAPACITY = {LEG_A: 33_280, LEG_B: 17_152}
LEG_GRIPPER_PARTS = {LEG_A: 64, LEG_B: 1}
LEG_SHAPE_INVENTORY = {
    LEG_A: {"sensor_cylinder": 1, "support_table": 1, "link4": 1, "link5": 64, "gripper_link": 64},
    LEG_B: {"sensor_cylinder": 1, "support_table": 1, "link4": 1, "link5": 64, "gripper_link": 1},
}
LEG_PREFIX_PROFILES = {LEG_A: "D407_LEGA_ACTUAL_PHYSX_DURABLE_PREFIX_V1", LEG_B: "D407_LEGB_ACTUAL_PHYSX_DURABLE_PREFIX_V1"}

# Leg-dependent module globals; every one of them is assigned by
# _configure_leg before any stage code runs (main() enforces this).
_LEG: str | None = None
OUT_DIR: Path | None = None
INVOCATION_PATH: Path | None = None
WORKER_PREFLIGHT_PATH: Path | None = None
PHASE_PATH: Path | None = None
WORKER_LOG_PATH: Path | None = None
PREREQUISITE_PATH: Path | None = None
TRACE_JSON_PATH: Path | None = None
TRACE_CSV_PATH: Path | None = None
WORKER_SUMMARY_PATH: Path | None = None
WORKER_EXCEPTION_PATH: Path | None = None
RRD_PATH: Path | None = None
RBL_PATH: Path | None = None
RERUN_PNG_PATH: Path | None = None
RERUN_VALIDATION_PATH: Path | None = None
PREFIX_PATH: Path | None = None
PREFIX_AUDIT_PATH: Path | None = None
FAILURE_PREFIX_AUDIT_PATH: Path | None = None
SUPERVISOR_PREFIX_AUDIT_PATH: Path | None = None
CONTACT_OVERFLOW_WARNING_AUDIT_PATH: Path | None = None
SHEET_PATH: Path | None = None
REGISTERED_TOTAL_CONTACT_CAPACITY: int | None = None
D407_PREFIX_PROFILE: str | None = None
D407_PREFIX_SUBJECT: str | None = None

_PHASE_SEQUENCE = 0
_CONTROLLED_STEPS = 0
_Q5_TARGET_UPDATE_COUNT = 0
_PREFIX_WRITER: Any = None
_PREFIX_LAST_STATE: dict[str, Any] | None = None
_PREFIX_HIGH_WATER = 0
_D361_ORIGINAL_VALIDATE_HEADER = d361._validate_header_payload
_D361_ORIGINAL_VALIDATE_OBSERVATION = d361._validate_observation_payload


def _configure_leg(leg: str) -> None:
    """Bind every leg-dependent module global from the leg tables."""
    global _LEG, OUT_DIR, INVOCATION_PATH, WORKER_PREFLIGHT_PATH, PHASE_PATH
    global WORKER_LOG_PATH, PREREQUISITE_PATH, TRACE_JSON_PATH, TRACE_CSV_PATH
    global WORKER_SUMMARY_PATH, WORKER_EXCEPTION_PATH, RRD_PATH, RBL_PATH
    global RERUN_PNG_PATH, RERUN_VALIDATION_PATH, PREFIX_PATH, PREFIX_AUDIT_PATH
    global FAILURE_PREFIX_AUDIT_PATH, SUPERVISOR_PREFIX_AUDIT_PATH
    global CONTACT_OVERFLOW_WARNING_AUDIT_PATH, SHEET_PATH
    global REGISTERED_TOTAL_CONTACT_CAPACITY, D407_PREFIX_PROFILE, D407_PREFIX_SUBJECT
    if leg not in (LEG_A, LEG_B):
        raise RuntimeError(f"D407 unknown leg: {leg}")
    _LEG = leg
    OUT_DIR = ATTEMPT_DIR / LEG_OUT_SUBDIRS[leg]
    INVOCATION_PATH = OUT_DIR / "d407_isaac_invocation_marker.json"
    WORKER_PREFLIGHT_PATH = OUT_DIR / "d407_worker_preflight.json"
    PHASE_PATH = OUT_DIR / "d407_phase_markers.jsonl"
    WORKER_LOG_PATH = OUT_DIR / "d407_worker_stdout_stderr.log"
    PREREQUISITE_PATH = OUT_DIR / "d407_runtime_prerequisites.json"
    TRACE_JSON_PATH = OUT_DIR / "d407_physics_trace.json"
    TRACE_CSV_PATH = OUT_DIR / "d407_physics_trace.csv"
    WORKER_SUMMARY_PATH = OUT_DIR / "d407_worker_summary.json"
    WORKER_EXCEPTION_PATH = OUT_DIR / "d407_worker_exception.json"
    RRD_PATH = OUT_DIR / "d407_physx_contact_motion.rrd"
    RBL_PATH = OUT_DIR / "d407_physx_contact_motion.rbl"
    RERUN_PNG_PATH = OUT_DIR / "d407_physx_contact_motion_rerun.png"
    RERUN_VALIDATION_PATH = OUT_DIR / "d407_rerun_validation.json"
    PREFIX_PATH = OUT_DIR / "d407_durable_step_prefix.jsonl"
    PREFIX_AUDIT_PATH = OUT_DIR / "d407_durable_step_prefix_audit.json"
    FAILURE_PREFIX_AUDIT_PATH = OUT_DIR / "d407_failure_prefix_audit.json"
    SUPERVISOR_PREFIX_AUDIT_PATH = OUT_DIR / "d407_supervisor_prefix_recovery_audit.json"
    CONTACT_OVERFLOW_WARNING_AUDIT_PATH = (
        OUT_DIR / "d407_contact_overflow_warning_audit.json"
    )
    SHEET_PATH = OUT_DIR / "d407_beginner_result_sheet_ko.png"
    REGISTERED_TOTAL_CONTACT_CAPACITY = LEG_CAPACITY[leg]
    D407_PREFIX_PROFILE = LEG_PREFIX_PROFILES[leg]
    D407_PREFIX_SUBJECT = (
        f"actual_d407_leg_{leg}_d362_inherited_physx_contact_motion_evidence"
    )


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
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()


def _status_paths() -> list[str]:
    """Exact-allowlist scope model (D401 snapshot style): every dirty path is
    compared as a whole path against the preregistered allowlist, replacing
    the D362 prefix-based scope model."""
    return [row[3:] for row in _git_status() if len(row) > 3]


def _sidecar_hashes() -> dict[str, str]:
    root = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
    return {_rel(path): _sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _leg_asset_files(leg: str) -> list[Path]:
    root = LEG_ASSET_DIRS[leg]
    return sorted(p for p in root.rglob("*") if p.is_file())


def _input_paths() -> list[Path]:
    return [
        *_leg_asset_files(LEG_A),
        *_leg_asset_files(LEG_B),
        D348_EVIDENCE,
        D334_SUMMARY,
        D354_MEASUREMENT,
        D354_ATTESTATION,
        URDF_PATH,
        D361_CAPACITY,
        D361_PROTOCOL,
        D362_HARNESS,
        D362_PREREG,
        D362_TRACE_JSON,
        D362_WORKER_SUMMARY,
        D362_SUPERVISOR_SUMMARY,
        D360_HARNESS,
        Path(d351.__file__).resolve(),
        Path(d351.d332.__file__).resolve(),
        Path(d351.d333.__file__).resolve(),
        Path(d351.d334.__file__).resolve(),
        Path(d351.d349.__file__).resolve(),
        Path(d351.d350.__file__).resolve(),
        Path(d361.__file__).resolve(),
        ROARM_ENV_MODULE,
        RERUN_CONTRACT_MODULE,
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


def _asset_dir_contract() -> dict[str, Any]:
    """Design section 3.4 gates (1)-(4): both leg asset dirs are enumerated in
    full (exactly 7 files each), every file sha256 must match the prereg
    ``leg_asset_pins`` ({"a": {relname: sha256}, "b": {...}}), the 6 non-root
    files must be bit-identical A<->B, and only the root layer may differ.
    Called in the worker preflight and re-run by the controller post-run
    (gate (5): non-tamper proof of in-place read-only consumption)."""
    prereg = _json(PREREG_PATH)
    pins = prereg.get("leg_asset_pins", {})
    observed: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for leg in (LEG_A, LEG_B):
        files = _leg_asset_files(leg)
        root = LEG_ASSET_DIRS[leg]
        observed[leg] = {
            str(path.relative_to(root)): _sha(path) for path in files
        }
        counts[leg] = len(files)
    names_a = set(observed[LEG_A])
    names_b = set(observed[LEG_B])
    root_name = "roarm_m3.usd"
    nonroot = sorted(names_a - {root_name})
    checks = {
        "each_dir_exactly_seven_files": counts[LEG_A] == 7 and counts[LEG_B] == 7,
        "relative_name_sets_equal": names_a == names_b,
        "all_fourteen_pins_exact": observed == pins,
        "six_nonroot_files_bit_identical_a_b": root_name in names_a
        and len(nonroot) == 6
        and all(observed[LEG_A][name] == observed[LEG_B][name] for name in nonroot),
        "root_layer_only_differs": root_name in names_a
        and root_name in names_b
        and observed[LEG_A][root_name] != observed[LEG_B][root_name],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "observed": observed,
        "file_counts": counts,
    }


def _expected_postworker_inventory() -> list[str]:
    """Exact per-leg (OUT_DIR) inventory after worker exit plus the two
    controller-side per-leg audits.  ATTEMPT-root artifacts (preregistration,
    attestation, tuple, supervisor/delta/completion summaries, inspection,
    comparison sheet) are never part of a leg inventory."""
    names = {
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
    }
    return sorted(names)


def _core_postworker_inventory() -> set[str]:
    return {
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
    }


def _expected_precompletion_inventory() -> list[str]:
    """Per-leg dirs gain no additional files between the post-worker audit and
    completion; supervisor/automated/manual artifacts live at ATTEMPT root."""
    return _expected_postworker_inventory()


def _phase_contract(worker: dict[str, Any]) -> dict[str, Any]:
    """D362 phase-marker ordering gate with the D407 worker's actual emitted
    phase list: viewport-capture and trace-replay-video phases are removed,
    ``live_64_plus_64_binding`` is renamed ``live_binding`` (leg B binds
    64+1), and the seal must precede the Rerun recording start."""
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
        ("live_binding", "start", None),
        ("live_binding", "complete", detail_pass),
        ("runtime_contact_capacity", "complete", detail_pass),
        ("durable_prefix", "initialized", detail_pass),
        (
            "frozen_open_baseline",
            "progress",
            lambda row: row.get("details", {}).get("completed") == BASELINE_STEPS,
        ),
    ]
    if worker.get("baseline", {}).get("pass") is True:
        mandatory_specs.extend(
            [
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
                ("q5_close_observation", "not_run", None),
            ]
        )
    mandatory_specs.extend(
        [
            ("durable_prefix", "audit_complete", detail_pass),
            ("rerun_recording", "start", None),
            ("rerun_recording", "rrd_finalized", None),
            ("rerun_validation", "start", None),
            ("rerun_validation", "complete", detail_pass),
            ("worker_summary", "complete", detail_pass),
        ]
    )
    mandatory_indices: list[int] = []
    mandatory_exact_once = True
    for phase, event, predicate in mandatory_specs:
        found = indices(phase, event, predicate)
        mandatory_exact_once = mandatory_exact_once and len(found) == 1
        mandatory_indices.append(found[0] if len(found) == 1 else -1)
    rerun_start_indices = indices("rerun_recording", "start")
    prefix_init_indices = indices("durable_prefix", "initialized")
    prefix_seal_indices = indices("durable_prefix", "sealed")
    first_control_progress_indices = indices("frozen_open_baseline", "progress")
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
        "q5_command_count_exact": len(
            indices("q5_close_command", "target_updated_once")
        )
        == (1 if worker.get("baseline", {}).get("pass") is True else 0),
        "prefix_initialized_before_control_progress": len(prefix_init_indices) == 1
        and bool(first_control_progress_indices)
        and prefix_init_indices[0] < min(first_control_progress_indices),
        "prefix_sealed_before_rerun_start": len(prefix_seal_indices) == 1
        and len(rerun_start_indices) == 1
        and prefix_seal_indices[0] < rerun_start_indices[0],
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


def _frozen_value_contract() -> dict[str, Any]:
    """Leg-independent frozen-value checks inherited from the still-relevant
    half of D362's _parameter_contract.  Both the worker preflight and the
    controller call this; D360/D361/D359 lineage checks and the video
    contract are retired with their subsystems."""
    measurement = _json(D354_MEASUREMENT)
    raw_bracket = measurement["raw_contact_order"]["first_contact_bracket"]
    live_bracket = measurement["live_contact_order"]["first_contact_bracket"]
    checks = {
        "seed_33201": SEED == 33201,
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
        "camera_is_d351_verified_oblique": CAMERA_EYE == [0.49, -0.32, 0.28]
        and OPPOSITE_CAMERA_EYE == [0.49, 0.32, 0.28]
        and CAMERA_TARGET == [0.285, 0.0, 0.055],
        "exactly_one_new_variable": NEW_VARIABLES
        == [
            "gripper_link_collision_representation_a64_to_sdf_res256_v1",
        ],
        "d361_capacity_protocol_hashes_exact": _sha(D361_CAPACITY)
        == D361_CAPACITY_SHA256
        and _sha(D361_PROTOCOL) == D361_PROTOCOL_SHA256,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "baseline_semantics": "frozen q0-q4 plus q5 OPEN; never D333 HOME baseline",
        "closure_horizon_semantics": "300 steps is a contact-observation horizon, not q5=0 settle proof",
        "initial_proportional_error_torque_diagnostic_nm": float(Q5_OPEN_F32 * ACTUATOR_STIFFNESS),
        "effort_cap_nm": ACTUATOR_EFFORT_LIMIT_NM,
    }


def _frozen_d362_science_source_contract() -> dict[str, Any]:
    """Independent source-equality gate for every inherited science function."""
    import inspect

    from sim_scripts import (
        cyl34_top_view_d362_current_pose_capacity_prefix_integrated_physx_contact_motion as d362,
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
        d362_source = inspect.getsource(getattr(d362, name)).replace(
            "D362", "DXXX"
        )
        d407_source = inspect.getsource(globals()[name]).replace("D407", "DXXX")
        rows[name] = {
            "normalized_source_equal": d362_source == d407_source,
            "d362_source_sha256": hashlib.sha256(
                d362_source.encode("utf-8")
            ).hexdigest(),
            "d407_source_sha256": hashlib.sha256(
                d407_source.encode("utf-8")
            ).hexdigest(),
        }
    checks = {
        "all_registered_functions_present": set(rows) == set(function_names),
        "all_normalized_sources_equal": all(
            row["normalized_source_equal"] for row in rows.values()
        ),
    }
    return {
        "authority": "inspect.getsource normalized only for D362/D407 verdict-label rename",
        "functions": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _make_runtime_env(args: argparse.Namespace) -> Any:
    """D333 sole-support scene with only the D361 observability capacity changed."""
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import UsdPhysics
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnv

    class D407CapacityIntegratedSoleSupportCylinderEnv(RoArmCubeTap10cmEnv):
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
    return D407CapacityIntegratedSoleSupportCylinderEnv(cfg=env_cfg)


def _configure_d361_runtime_prefix_schema() -> None:
    """Bind D361's tested wire/semantic implementation to accurate D407 labels."""
    d361.CASE = CASE
    d361.PREREG_PATH = PREREG_PATH
    d361.PREPARE_PATH = WORKER_PREFLIGHT_PATH
    d361.INVOCATION_PATH = INVOCATION_PATH
    d361.HARNESS = HARNESS
    d361.CAPACITY_PATH = D361_CAPACITY
    d361.PROTOCOL_PATH = D361_PROTOCOL
    d361.REGISTERED_TOTAL_CAPACITY = REGISTERED_TOTAL_CONTACT_CAPACITY

    def validate_header(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("D407 prefix header must be an object")
        if payload.get("profile") != D407_PREFIX_PROFILE:
            raise ValueError("D407 prefix profile mismatch")
        if payload.get("subject_kind") != D407_PREFIX_SUBJECT:
            raise ValueError("D407 prefix subject mismatch")
        lineage = payload.get("lineage")
        expected_lineage_keys = {
            "preregistration_sha256",
            "worker_preflight_sha256",
            "actual_invocation_sha256",
            "harness_sha256",
            "inherited_d361_capacity_budget_sha256",
            "inherited_d361_protocol_contract_sha256",
        }
        if not isinstance(lineage, dict) or set(lineage) != expected_lineage_keys:
            raise ValueError("D407 prefix lineage keys are not exact")
        legacy = json.loads(json.dumps(payload))
        legacy["profile"] = d361.PREFIX_PROFILE
        legacy["subject_kind"] = (
            "future_actual_d360_inherited_state_requires_separate_approval"
        )
        legacy["lineage"] = {
            "preregistration_sha256": lineage["preregistration_sha256"],
            "prepare_preflight_sha256": lineage["worker_preflight_sha256"],
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
        if not isinstance(payload, dict) or payload.get("subject_kind") != D407_PREFIX_SUBJECT:
            raise ValueError("D407 observation subject mismatch")
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
        "sensor_cfg_registered_exact": cfg_capacity == REGISTERED_TOTAL_CONTACT_CAPACITY,
        "backend_allocation_registered_exact": backend_capacity
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
        == LEG_SHAPE_INVENTORY[_LEG],
        "live_topology_matches_stage_link5_gripper": len(
            topology_parts["link5"]
        )
        == actual_shapes["link5"]
        and len(topology_parts["gripper_link"])
        == actual_shapes["gripper_link"],
        "derived_capacity_registered_exact": derived == REGISTERED_TOTAL_CONTACT_CAPACITY,
        "d361_capacity_hash_exact": _sha(D361_CAPACITY) == D361_CAPACITY_SHA256,
        "d361_protocol_hash_exact": _sha(D361_PROTOCOL) == D361_PROTOCOL_SHA256,
    }
    return {
        "authority": (
            "D361 installed PhysX-5.6.1 convex-pair evidence integrated at "
            "runtime"
            if _LEG == LEG_A
            else "project capacity assumption for the SDF pair; not a documented engine limit"
        ),
        "sdf_pair_capacity_semantics": (
            None
            if _LEG == LEG_A
            else {
                "engine_limit": False,
                "classification": "project_assumption_for_sdf_pair",
                "fail_capable_verifier": (
                    "post-worker Incomplete contact data/maxContactDataCount "
                    "overflow warning audit"
                ),
            }
        ),
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
        "scenario": "actual_frozen_d360_contract_d407_single_invocation",
        "profile": D407_PREFIX_PROFILE,
        "subject_kind": D407_PREFIX_SUBJECT,
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
            "worker_preflight_sha256": _sha(WORKER_PREFLIGHT_PATH),
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
            f"D407 raw contact count/start shape mismatch: {counts.shape}/{starts.shape}"
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
        "subject_kind": D407_PREFIX_SUBJECT,
        "step_identity": identity,
        "state_row": row,
        "contact_point_capacity_diagnostic": diagnostic,
        "event_projection": d361._event_projection(_PREFIX_LAST_STATE, row),
    }
    return payload


def _initialize_prefix_writer() -> None:
    global _PREFIX_WRITER
    if _PREFIX_WRITER is not None or PREFIX_PATH.exists():
        raise RuntimeError("D407 durable prefix would be initialized twice")
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
        raise RuntimeError("D407 durable prefix is not initialized")
    identity = {
        "global_step": int(inner._sim_step_counter) + 1,
        "phase": phase,
        "phase_step": int(phase_step),
    }
    _PREFIX_WRITER.begin_step(identity, _pre_step_state(inner, timeline))


def _observe_prefix_step(row: dict[str, Any], inner: Any, filter_map: dict[str, int]) -> None:
    global _PREFIX_LAST_STATE
    if _PREFIX_WRITER is None:
        raise RuntimeError("D407 durable prefix is not initialized")
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
        raise RuntimeError("D407 durable prefix is not initialized")
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
        "artifact": "D407_DURABLE_STEP_PREFIX_AUDIT_V1",
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
        raise RuntimeError("D407 supervisor prefix recovery audit already exists")
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
        "artifact": "D407_SUPERVISOR_PREFIX_RECOVERY_AUDIT_V1",
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
        raise RuntimeError("D407 contact overflow warning audit already exists")
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
        "artifact": "D407_CONTACT_OVERFLOW_WARNING_AUDIT_V1",
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


def _resolved_headless_launcher(launcher: Any) -> dict[str, Any]:
    """Headless counterpart of the frozen d351.d350 GUI launcher contract:
    same values dict, headless must be True, the experience must be the
    headless kit, and offscreen_render is RECORDED but not gated (design
    section 3.6.2)."""
    values = {
        "headless": bool(launcher._headless),
        "livestream": int(launcher._livestream),
        "enable_cameras": bool(launcher._enable_cameras),
        "xr": bool(launcher._xr),
        "offscreen_render": bool(launcher._offscreen_render),
        "device_id": int(launcher.device_id),
        "experience": str(launcher._sim_experience_file),
    }
    checks = {
        "headless_true": values["headless"] is True,
        "livestream_zero": values["livestream"] == 0,
        "cameras_disabled": values["enable_cameras"] is False,
        "xr_disabled": values["xr"] is False,
        "device_zero": values["device_id"] == 0,
        "headless_experience": Path(values["experience"]).name
        == "isaaclab.python.headless.kit",
    }
    return {"values": values, "checks": checks, "pass": all(checks.values())}


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
        raise RuntimeError("D407 articulation joint-position target buffer drifted")
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
        raise RuntimeError(f"D407 q5 torque authority unavailable: {q5_torque}")
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
        raise RuntimeError("D407 timeline was not PLAY before controlled physics step")
    d351.d332._physics_step(inner)
    _CONTROLLED_STEPS += 1
    after_clock = d351._simulation_clock(inner)
    after_timeline = {
        "playing": bool(timeline.is_playing()),
        "stopped": bool(timeline.is_stopped()),
        "time_s": float(timeline.get_current_time()),
    }
    if int(inner._sim_step_counter) != before_counter + 1:
        raise RuntimeError("D407 controlled-step counter did not increment by one")
    if before_clock["current_time_step_index"] is not None and after_clock["current_time_step_index"] != before_clock["current_time_step_index"] + 1:
        raise RuntimeError("D407 SimulationContext step index did not increment by one")
    if before_clock["current_time"] is not None and not math.isclose(
        float(after_clock["current_time"] - before_clock["current_time"]), PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError("D407 SimulationContext time did not increment by 0.005 s")
    if not after_timeline["playing"] or after_timeline["stopped"]:
        raise RuntimeError("D407 timeline left PLAY during controlled physics step")
    if after_timeline["time_s"] < before_timeline["time_s"]:
        raise RuntimeError("D407 timeline time moved backward during controlled physics step")
    return {
        "before": before_timeline,
        "after": after_timeline,
        "delta_s_diagnostic": after_timeline["time_s"] - before_timeline["time_s"],
        "simulation_context_is_step_time_authority": True,
    }


def _set_closed_q5_target(inner: Any, open_target: Any, q5_index: int) -> Any:
    global _Q5_TARGET_UPDATE_COUNT
    if _Q5_TARGET_UPDATE_COUNT != 0:
        raise RuntimeError("D407 q5 target update would occur more than once")
    close_target = open_target.detach().clone()
    before_q0_q4 = close_target[:, :5].detach().cpu().numpy().astype(np.float32).tobytes()
    close_target[:, q5_index] = float(Q5_CLOSED_F32)
    after_q0_q4 = close_target[:, :5].detach().cpu().numpy().astype(np.float32).tobytes()
    if before_q0_q4 != after_q0_q4:
        raise RuntimeError("D407 q0-q4 target mutated while closing q5")
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
        verdict = "D407_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif (
        not baseline_pass
        or not all_finite
        or not full_horizon
        or not q5_response
        or not fixed_root_contract
    ):
        verdict = "D407_CONTROL_HORIZON_OR_BASELINE_FAIL_STOP"
    elif first_other >= 0 and (moving_contact < 0 or first_other <= moving_contact):
        verdict = "D407_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif motion >= 0 and (moving_contact < 0 or motion < moving_contact):
        verdict = "D407_OTHER_ROBOT_BODY_CONTACT_OR_PRECONTACT_CONFOUND_FAIL_STOP"
    elif positive_contact and motion >= moving_contact:
        verdict = "D407_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED"
    elif positive_contact and motion < 0:
        verdict = "D407_MOVING_JAW_CONTACT_WITHOUT_THRESHOLD_OBJECT_MOTION"
    elif moving_contact < 0 and bracket_reached:
        verdict = "D407_NO_POSITIVE_CONTACT_WITNESS_UNRESOLVED"
    else:
        verdict = "D407_CONTROL_HORIZON_OR_BASELINE_FAIL_STOP"
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

    # Blueprint structure is inherited from D362: one Spatial3DView, one
    # TextLogView, and three TimeSeriesViews.  No TextDocumentView is used,
    # so the D405 R3 one-document-per-view constraint does not apply here.
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/**", "/transforms/**", "/contacts/**"],
                    name=(
                        "actual 64+64 colliders, cylinder, force witnesses"
                        if _LEG == LEG_A
                        else "actual link5 64 + gripper SDF mesh, cylinder, force witnesses"
                    ),
                    eye_controls=eye(),
                    spatial_information=rrb.SpatialInformation(
                        target_frame="world", show_axes=True, show_bounding_box=False
                    ),
                ),
                rrb.TextLogView(origin="/events/d407", contents="/events/d407/**", name="bounded events"),
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
            "# D407 bounded PhysX jaw close",
            "",
            f"- leg: {_LEG} ({'A64 control' if _LEG == LEG_A else 'SDF res256 treatment'})",
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
        f"roarm_g0a_d407_leg_{_LEG}_physx_contact_motion",
        recording_id=f"g0a_d407_leg_{_LEG}_physx_contact_motion",
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
        recording.log(
            remember("transforms/object_spawn_reference"),
            rr.Transform3D(
                translation=OBJECT_POS_F32.astype(np.float64).tolist(),
                rotation=_rr_quaternion(rr, OBJECT_QUAT_F32.astype(np.float64).tolist()),
                parent_frame="world",
                child_frame="reference/object_spawn",
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
                recording.log(remember("events/d407/run"), rr.TextLog(event_text, level=rr.TextLogLevel.INFO))
            if row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(rows):
                _marker(
                    "rerun_recording",
                    "progress",
                    {"completed": row_index + 1, "requested": len(rows)},
                )
        recording.flush(timeout_sec=30.0)
    _marker("rerun_recording", "rrd_finalized", {"rrd_bytes": RRD_PATH.stat().st_size})
    blueprint.save(f"roarm_g0a_d407_leg_{_LEG}_physx_contact_motion", RBL_PATH)
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
        "events/d407/run": ["TextLog:level", "TextLog:text"],
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
        screenshot_window_size=RERUN_SCREENSHOT_LOGICAL_SIZE,
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    screenshot_report = _png_report(RERUN_PNG_PATH, RERUN_SCREENSHOT_PHYSICAL)
    validation["screenshot_report"] = screenshot_report
    validation["checks"] = {
        "validation_pass_before_screenshot_size": validation.get("pass") is True,
        "screenshot_physical_1920x1080_exact": screenshot_report.get("pass") is True,
    }
    validation["pass"] = all(validation["checks"].values())
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
    canvas = Image.new("RGB", tuple(REPLAY_FRAME_SIZE), (9, 13, 19))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (36, 22),
        "D407 canonical trace replay — PHYSICS NOT RECOMPUTED",
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

    # Three full-run panels required by design section 3.6.4: q5, all three
    # robot-body forces, and object XY/tilt/z.  Contact and motion confirmation
    # markers are drawn on every panel, plus the current-row cursor.
    chart = (46, 755, 1215, 1044)
    draw.rounded_rectangle(chart, radius=16, fill=(17, 24, 34), outline=(75, 92, 114), width=2)
    event_indices = [
        index
        for index, item in enumerate(rows)
        if item["event_masks"]["two_step_any_robot_confirmation_end"]
    ]
    motion_event_indices = [
        index
        for index, item in enumerate(rows)
        if item["event_masks"]["two_step_motion_confirmation_end"]
    ]

    def plot_panel(
        top: int,
        bottom: int,
        title: str,
        series: list[tuple[str, list[float], tuple[int, int, int]]],
        fixed_range: tuple[float, float] | None = None,
    ) -> None:
        x0, x1 = 310, 1186
        y0, y1 = top + 7, bottom - 7
        values = [value for _, data, _ in series for value in data]
        low, high = (
            fixed_range
            if fixed_range is not None
            else (min(values), max(values))
        )
        if not math.isfinite(low) or not math.isfinite(high):
            raise RuntimeError(f"D407 nonfinite timeseries range: {title}")
        if high <= low:
            high = low + 1.0

        def xy(index: int, value: float) -> tuple[float, float]:
            x = x0 + (x1 - x0) * index / max(len(rows) - 1, 1)
            y = y1 - (y1 - y0) * (value - low) / (high - low)
            return x, y

        draw.text((66, top + 2), title, font=font(15), fill=(225, 232, 241))
        for event_index in event_indices:
            x, _ = xy(event_index, low)
            draw.line((x, y0, x, y1), fill=(255, 92, 80), width=1)
        for event_index in motion_event_indices:
            x, _ = xy(event_index, low)
            draw.line((x, y0, x, y1), fill=(205, 112, 255), width=1)
        cursor_x, _ = xy(row_index, low)
        draw.line((cursor_x, y0, cursor_x, y1), fill=(255, 222, 107), width=2)
        for series_index, (label, data, color) in enumerate(series):
            draw.line(
                [xy(index, value) for index, value in enumerate(data)],
                fill=color,
                width=2,
            )
            draw.text(
                (66 + 80 * series_index, bottom - 21),
                label,
                font=font(12),
                fill=color,
            )

    plot_panel(
        768,
        852,
        "q5 rad",
        [
            ("actual", [float(item["q5_actual_rad"]) for item in rows], (92, 203, 255)),
            ("target", [float(item["q5_target_rad"]) for item in rows], (255, 111, 99)),
        ],
        (-0.05, float(Q5_OPEN_F32) + 0.05),
    )
    plot_panel(
        860,
        944,
        "force N",
        [
            (
                "link4",
                [
                    float(item["contact"]["by_filter"]["link4"]["force_norm_n"])
                    for item in rows
                ],
                (210, 112, 255),
            ),
            (
                "link5",
                [
                    float(item["contact"]["by_filter"]["link5"]["force_norm_n"])
                    for item in rows
                ],
                (90, 192, 255),
            ),
            (
                "grip",
                [
                    float(
                        item["contact"]["by_filter"]["gripper_link"][
                            "force_norm_n"
                        ]
                    )
                    for item in rows
                ],
                (255, 103, 92),
            ),
        ],
        None,
    )
    plot_panel(
        952,
        1036,
        "motion",
        [
            ("XYmm", [float(item["object_disp_xy_mm"]) for item in rows], (92, 220, 126)),
            (
                "tilt°",
                [
                    float(item["object_tilt_delta_from_reference_deg"])
                    for item in rows
                ],
                (255, 198, 79),
            ),
            ("zmm", [float(item["object_z_delta_mm"]) for item in rows], (95, 150, 255)),
        ],
        None,
    )
    draw.text(
        (1005, 1030),
        "red=contact  purple=motion  yellow=current",
        font=font(11),
        fill=(190, 200, 216),
    )

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


def _png_report(path: Path, expected_size: list[int]) -> dict[str, Any]:
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
        "expected_dimensions": list(expected_size),
        "mode": mode,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "pass": size == list(expected_size),
    }


FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _build_beginner_sheet(
    capture_event_metadata: dict[str, Any],
    baseline: dict[str, Any],
    closure: dict[str, Any],
    all_rows: list[dict[str, Any]],
    topology_parts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Canonical-trace-based offline PIL sheet (design section 3.6.4): the
    D362 capture panels are replaced by three deterministic trace-replay
    frames; the text panel stays Korean and bbox-guarded."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> Any:
        return ImageFont.truetype(str(FONT_PATH), size=size)

    contact_index = next(
        (
            index
            for index, row in enumerate(all_rows)
            if row["event_masks"]["two_step_any_robot_confirmation_end"]
        ),
        None,
    )
    middle_index = (
        contact_index if contact_index is not None else min(250, len(all_rows) - 1)
    )
    frame_sources = [
        (0, "A. 시작 — row 0 (frozen OPEN)"),
        (
            middle_index,
            (
                f"B. 첫 robot-body 접촉 확인 — row {middle_index}"
                if contact_index is not None
                else f"B. 접촉 확인 없음 — 진단 row {middle_index}"
            ),
        ),
        (len(all_rows) - 1, f"C. 최종 — row {len(all_rows) - 1}"),
    ]
    leg_kind = "A64 control" if _LEG == LEG_A else "SDF res256 treatment"
    canvas = Image.new("RGB", (3840, 1720), (17, 22, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1920, 24),
        f"D407 leg {_LEG} ({leg_kind}) — 현재 자세 PhysX 죠 닫힘 초보자용 결과판",
        font=font(58),
        fill=(248, 250, 253),
        anchor="ma",
    )
    for index, (source_index, title) in enumerate(frame_sources):
        frame = _render_trace_replay_frame(topology_parts, all_rows, source_index)
        panel = frame.resize((1200, 675))
        x = 60 + index * 1260
        canvas.paste(panel, (x, 135))
        draw.rectangle((x, 92, x + 1199, 135), fill=(40, 53, 70))
        draw.text((x + 600, 96), title, font=font(30), fill=(255, 255, 255), anchor="ma")
    final = all_rows[-1]
    peak_by_filter = closure["peak_force_by_filter"]
    event_by_filter = closure["first_contact_step_by_filter"]
    contact_meta = capture_event_metadata.get("contact")
    contact_where = (
        "없음"
        if contact_meta is None
        else f"{contact_meta['phase']} step {contact_meta['onset_phase_step']}→"
        f"{contact_meta['confirmation_phase_step']}, body="
        f"{'+'.join(contact_meta['qualifying_body_labels'])}"
    )
    motion_meta = capture_event_metadata.get("motion")
    motion_where = (
        "없음"
        if motion_meta is None
        else f"{motion_meta['phase']} step {motion_meta['onset_phase_step']}→"
        f"{motion_meta['confirmation_phase_step']}"
    )
    baseline_word = "PASS" if baseline["pass"] else "FAIL — closure 미실행"
    items = [
        f"case: {CASE} / leg {_LEG} ({leg_kind}) — 신규 변수 1개: {NEW_VARIABLES[0]}",
        f"판정 코드: {closure['verdict']}     200-step OPEN baseline: {baseline_word}",
        (
            f"최종 row {len(all_rows) - 1}: XY 이동 {final['object_disp_xy_mm']:.4f} mm / "
            f"기울기Δ {final['object_tilt_delta_from_reference_deg']:.4f}° / "
            f"zΔ {final['object_z_delta_mm']:.4f} mm"
        ),
        f"첫 robot-body 확인: {contact_where}     첫 물체 운동 확인: {motion_where}",
        (
            "closure 첫 2-step 접촉 step (link4/link5/gripper) = "
            f"{event_by_filter['link4']} / {event_by_filter['link5']} / "
            f"{event_by_filter['gripper_link']}"
            f"     첫 물체 운동 step: {closure['first_object_motion_step']}"
        ),
        (
            "peak force (N) link4/link5/gripper = "
            f"{float(peak_by_filter['link4']['force_norm_n'] or 0.0):.4f} / "
            f"{float(peak_by_filter['link5']['force_norm_n'] or 0.0):.4f} / "
            f"{float(peak_by_filter['gripper_link']['force_norm_n'] or 0.0):.4f}"
        ),
        (
            "g0a_pass=false — 이 실험은 A/B 재측정 case이며 grasp 성공, force "
            "closure, cap/rim 순서, 정확한 collider face는 판정하지 않습니다."
        ),
    ]
    text = "\n".join(
        "• " + line
        for item in items
        for line in textwrap.wrap(item, width=150, subsequent_indent="  ")
    )
    draw.rounded_rectangle(
        (55, 900, 3785, 1640), radius=22, fill=(36, 47, 62), outline=(111, 135, 166), width=5
    )
    body_font = font(22)
    body_spacing = 7
    bbox = draw.multiline_textbbox((0, 0), text, font=body_font, spacing=body_spacing)
    if bbox[2] - bbox[0] > 3640 or bbox[3] - bbox[1] > 680:
        body_font = font(19)
        body_spacing = 5
        bbox = draw.multiline_textbbox(
            (0, 0), text, font=body_font, spacing=body_spacing
        )
    if bbox[2] - bbox[0] > 3640 or bbox[3] - bbox[1] > 680:
        raise RuntimeError(
            "D407 beginner-sheet left result text exceeds registered box"
        )
    draw.multiline_text(
        (105, 940), text, font=body_font, fill=(235, 248, 240), spacing=body_spacing
    )
    draw.text(
        (1920, 1670),
        "과학 판정 원본은 d407_physics_trace.json / .csv이며, 이 PNG와 Rerun은 사람이 보는 표시층입니다.",
        font=font(31),
        fill=(181, 197, 220),
        anchor="ma",
    )
    canvas.save(SHEET_PATH)
    with Image.open(SHEET_PATH) as image:
        image.load()
        size = list(image.size)
    return {
        "path": _rel(SHEET_PATH),
        "dimensions": size,
        "sha256": _sha(SHEET_PATH),
        "timeseries_panels": {
            "q5_actual_target": True,
            "three_robot_body_forces": ["link4", "link5", "gripper_link"],
            "object_motion": ["xy_mm", "tilt_delta_deg", "z_delta_mm"],
            "contact_confirmation_markers": True,
            "motion_confirmation_markers": True,
            "current_row_cursor": True,
        },
        "pass": size == [3840, 1720],
    }


def _usd_gripper_sdf_topology(
    stage: Any, robot_prefix: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Stage-independent gripper-side audit for leg B (design section 3.5 gate
    4): the 64 authored-disabled A64 parts must still exist with CollisionAPI,
    and the single SDF res256 mesh must be collision-enabled with
    PhysxSDFMeshCollisionAPI applied.  The source stream is pinned in its
    authored Float32/Int32 byte domain, then transformed mesh-local -> world ->
    gripper-body-local before it becomes a display topology part.  This avoids
    treating nested mesh-local coordinates as body-local coordinates.
    Offline-testable against the derivative USD (stage H2) because it takes
    (stage, robot_prefix) explicitly."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    collisions_root = f"{robot_prefix}/gripper_link/collisions"
    body_path = f"{robot_prefix}/gripper_link"
    part_rows: list[dict[str, Any]] = []
    for index in range(64):
        path = f"{collisions_root}/d338_convex_parts/part_{index:03d}"
        prim = stage.GetPrimAtPath(path)
        exists = bool(prim and prim.IsValid())
        has_api = bool(exists and prim.HasAPI(UsdPhysics.CollisionAPI))
        enabled = None
        if has_api:
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        part_rows.append(
            {
                "path": path,
                "exists": exists,
                "has_collision_api": has_api,
                "collision_enabled": None if enabled is None else bool(enabled),
            }
        )
    mesh_path = f"{collisions_root}/gripper_link/node_STL_BINARY_/mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    mesh_valid = bool(mesh_prim and mesh_prim.IsValid())
    applied_apis = (
        [str(name) for name in mesh_prim.GetAppliedSchemas()] if mesh_valid else []
    )
    mesh_enabled = None
    if mesh_valid and mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
        mesh_enabled = UsdPhysics.CollisionAPI(mesh_prim).GetCollisionEnabledAttr().Get()
    raw_points_f32 = None
    body_local_vertices = None
    triangles = None
    source_stream_sha256 = None
    body_local_points_sha256 = None
    body_local_bounds_m = None
    counts_all_three = False
    vertex_count = None
    triangle_count = None
    body_prim = stage.GetPrimAtPath(body_path)
    body_valid = bool(body_prim and body_prim.IsValid())
    if mesh_valid and body_valid and mesh_prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(mesh_prim)
        points_value = list(mesh.GetPointsAttr().Get() or [])
        counts_value = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices_value = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        if points_value and counts_value and indices_value:
            raw_points_f32 = np.asarray(
                [
                    [float(component) for component in point]
                    for point in points_value
                ],
                dtype="<f4",
            )
            face_counts = np.asarray(counts_value, dtype="<i4")
            flat_indices = np.asarray(indices_value, dtype="<i4")
            counts_all_three = bool(face_counts.size > 0 and np.all(face_counts == 3))
            if counts_all_three:
                triangles = flat_indices.astype(np.int64).reshape(-1, 3)
                triangle_count = int(len(triangles))
            vertex_count = int(len(raw_points_f32))
            source_stream_sha256 = hashlib.sha256(
                raw_points_f32.tobytes(order="C")
                + face_counts.tobytes(order="C")
                + flat_indices.tobytes(order="C")
            ).hexdigest()
            mesh_l2w = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            body_w2l = (
                UsdGeom.Xformable(body_prim)
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .GetInverse()
            )
            body_local_vertices = np.asarray(
                [
                    [
                        float(component)
                        for component in body_w2l.Transform(
                            mesh_l2w.Transform(
                                Gf.Vec3d(
                                    *[float(component) for component in point]
                                )
                            )
                        )
                    ]
                    for point in points_value
                ],
                dtype="<f8",
            )
            body_local_points_sha256 = hashlib.sha256(
                body_local_vertices.tobytes(order="C")
            ).hexdigest()
            body_local_bounds_m = [
                body_local_vertices.min(axis=0).tolist(),
                body_local_vertices.max(axis=0).tolist(),
            ]
    checks = {
        "gripper_a64_64_disabled": len(part_rows) == 64
        and all(
            row["exists"]
            and row["has_collision_api"]
            and row["collision_enabled"] is False
            for row in part_rows
        ),
        "gripper_sdf_mesh_1_enabled": mesh_valid
        and mesh_enabled is True
        and "PhysxSDFMeshCollisionAPI" in applied_apis,
        "sdf_mesh_all_faces_triangles": counts_all_three,
        "sdf_mesh_counts_exact": counts_all_three
        and vertex_count == 41094
        and triangle_count == 13698,
        "sdf_source_stream_sha256_exact": source_stream_sha256
        == SDF_SOURCE_STREAM_SHA256,
        "sdf_body_local_points_sha256_exact": body_local_points_sha256
        == SDF_BODY_LOCAL_POINTS_F64_SHA256,
        "sdf_body_local_bounds_exact": body_local_bounds_m
        == [list(row) for row in SDF_BODY_LOCAL_BOUNDS_M],
    }
    part = None
    if body_local_vertices is not None and triangles is not None:
        part = {
            "body": "gripper_link",
            "name": "sdf_res256_source_mesh",
            "path": mesh_path,
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "checks": checks,
            "pass": all(checks.values()),
            "_vertices": body_local_vertices,
            "_triangles": triangles,
        }
    record = {
        "robot_prefix": robot_prefix,
        "mesh_path": mesh_path,
        "mesh_collision_enabled": None if mesh_enabled is None else bool(mesh_enabled),
        "applied_api_schemas": applied_apis,
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "source_stream_sha256": source_stream_sha256,
        "expected_source_stream_sha256": SDF_SOURCE_STREAM_SHA256,
        "body_local_points_f64_sha256": body_local_points_sha256,
        "expected_body_local_points_f64_sha256": (
            SDF_BODY_LOCAL_POINTS_F64_SHA256
        ),
        "body_local_bounds_m": body_local_bounds_m,
        "expected_body_local_bounds_m": [
            list(row) for row in SDF_BODY_LOCAL_BOUNDS_M
        ],
        "a64_part_rows": part_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return part, record


def _leg_b_topology(inner: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Leg-B replacement for d351.d349._build_live_topology_parts, which is
    not separable (it asserts gripper 64 enabled).  The link5 half follows the
    d349 pattern faithfully — D348 evidence rows for link5 only, historical
    D347 path binding, live inventory presence/enabled/owner checks — and the
    gripper half is the SDF-mesh audit above."""
    d349 = d351.d349
    evidence = d349._json(d349.D348_EVIDENCE)
    historical = d349._json(d349.D347_LIVE_AUDIT)
    historical_by_name = {
        row["name"]: row for row in historical["per_body"]["link5"]["part_checks"]
    }
    inventory = d349.d334._usd_collision_inventory(inner, "link5")
    inventory_by_path = {row["path"]: row for row in inventory}
    link5_parts: list[dict[str, Any]] = []
    for source in evidence["rows"]:
        if source["body"] != "link5":
            continue
        historical_row = historical_by_name[source["name"]]
        path = historical_row["path"]
        current = inventory_by_path.get(path)
        instance = source["instance"]
        vertices = np.asarray(instance["vertices_m"], dtype=np.float64)
        triangles = np.asarray(instance["topology_triangles"], dtype=np.int64)
        checks = {
            "d348_row_pass": source["pass"] is True,
            "runtime_path_present": current is not None,
            "runtime_collision_enabled": bool(current and current["collision_enabled"]),
            "runtime_owner_matches": bool(
                current
                and current["nearest_rigid_body_ancestor"]
                == d349.d334.BODY_PATHS["link5"]
            ),
        }
        link5_parts.append(
            {
                "body": "link5",
                "name": source["name"],
                "path": path,
                "vertex_count": len(vertices),
                "triangle_count": len(triangles),
                "checks": checks,
                "pass": all(checks.values()),
                "_vertices": vertices,
                "_triangles": triangles,
            }
        )
    sdf_part, sdf_record = _usd_gripper_sdf_topology(
        inner.scene.stage, "/World/envs/env_0/Robot"
    )
    gripper_parts = [sdf_part] if sdf_part is not None else []
    checks = {
        "link5_64_enabled": len(link5_parts) == 64
        and all(part["pass"] for part in link5_parts),
        "gripper_a64_64_disabled": sdf_record["checks"]["gripper_a64_64_disabled"],
        "gripper_sdf_mesh_1_enabled": sdf_record["checks"]["gripper_sdf_mesh_1_enabled"],
        "sdf_mesh_counts_exact": sdf_record["checks"]["sdf_mesh_counts_exact"],
        "sdf_stream_and_body_local_display_exact": sdf_record["pass"] is True
        and sdf_record["checks"]["sdf_source_stream_sha256_exact"] is True
        and sdf_record["checks"]["sdf_body_local_points_sha256_exact"] is True
        and sdf_record["checks"]["sdf_body_local_bounds_exact"] is True,
    }
    live_binding = {
        "artifact": "D407_LEG_B_LIVE_TOPOLOGY_BINDING_V1",
        "authority": "leg B live stage audit + D348 frozen evidence (link5)",
        "link5_parts": [
            {key: value for key, value in part.items() if not key.startswith("_")}
            for part in link5_parts
        ],
        "gripper_sdf_record": sdf_record,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return {"link5": link5_parts, "gripper_link": gripper_parts}, live_binding


def _build_topology_parts_for_leg(
    inner: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if _LEG == LEG_A:
        return d351.d349._build_live_topology_parts(inner)
    return _leg_b_topology(inner)


def _cylinder_runtime_geometry_probe(inner: Any) -> dict[str, Any]:
    """Read-only Sponge-subtree enumeration (design section 3.5): records
    which prim carries the collider and its schema type.  Record-only —
    never a gate; identical code on both legs."""
    from pxr import Usd, UsdPhysics

    sponge_root = "/World/envs/env_0/Sponge"
    rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(
        inner.scene.stage, Usd.TraverseInstanceProxies()
    ):
        path = prim.GetPath().pathString
        if path != sponge_root and not path.startswith(sponge_root + "/"):
            continue
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        enabled = None
        if has_collision:
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        rows.append(
            {
                "path": path,
                "type_name": str(prim.GetTypeName()),
                "has_collision_api": bool(has_collision),
                "collision_enabled": None if enabled is None else bool(enabled),
                "applied_api_schemas": [str(name) for name in prim.GetAppliedSchemas()],
            }
        )
    collider_rows = [row for row in rows if row["has_collision_api"]]
    return {
        "authority": "read-only USD enumeration; record-only, never a gate",
        "sponge_root": sponge_root,
        "prims": rows,
        "collider_prims": collider_rows,
        "collider_type_names": sorted({row["type_name"] for row in collider_rows}),
    }


def _runtime_physics_settings_probe(inner: Any) -> dict[str, Any]:
    """Read-only record of previously implicit gravity/solver/offset settings.

    The design deliberately makes this a measurement rather than a gate.  The
    controller compares the canonical payload across legs and reports the
    result without changing the physical verdict.
    """
    from pxr import Usd, UsdPhysics

    def jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        try:
            return [jsonable(item) for item in value]
        except TypeError:
            return str(value)

    stage = inner.scene.stage
    scene_rows: list[dict[str, Any]] = []
    selected_attribute_rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if prim.IsA(UsdPhysics.Scene):
            scene = UsdPhysics.Scene(prim)
            direction_attr = scene.GetGravityDirectionAttr()
            magnitude_attr = scene.GetGravityMagnitudeAttr()
            scene_rows.append(
                {
                    "path": path,
                    "gravity_direction": {
                        "type": str(direction_attr.GetTypeName()),
                        "authored": bool(direction_attr.HasAuthoredValueOpinion()),
                        "value": jsonable(direction_attr.Get()),
                    },
                    "gravity_magnitude": {
                        "type": str(magnitude_attr.GetTypeName()),
                        "authored": bool(magnitude_attr.HasAuthoredValueOpinion()),
                        "value": jsonable(magnitude_attr.Get()),
                    },
                    "applied_api_schemas": [
                        str(name) for name in prim.GetAppliedSchemas()
                    ],
                }
            )
        for attribute in prim.GetAttributes():
            name = str(attribute.GetName())
            compact = name.lower().replace("_", "").replace(":", "")
            if not any(
                token in compact
                for token in ("gravity", "solver", "contactoffset", "restoffset")
            ):
                continue
            selected_attribute_rows.append(
                {
                    "path": path,
                    "name": name,
                    "type": str(attribute.GetTypeName()),
                    "authored": bool(attribute.HasAuthoredValueOpinion()),
                    "value": jsonable(attribute.Get()),
                }
            )
    canonical = {
        "physics_scene_prims": sorted(scene_rows, key=lambda row: row["path"]),
        "gravity_solver_contact_rest_offset_attributes": sorted(
            selected_attribute_rows,
            key=lambda row: (row["path"], row["name"]),
        ),
    }
    return {
        "authority": (
            "read-only live USD effective/authored value enumeration; record-only, "
            "never a runtime gate"
        ),
        "canonical_comparison_payload": canonical,
        "physics_scene_count": len(scene_rows),
        "selected_attribute_count": len(selected_attribute_rows),
    }


def _physx_cook_cache_provenance() -> dict[str, Any]:
    """Record-only PhysX cooking/cache counters (s4).  D406 observed the cook
    cache persisting across processes, so a fresh leg-B cook is not
    guaranteed; this record documents provenance and is never a gate."""
    record: dict[str, Any] = {
        "record_only": True,
        "available": False,
        "accessor": None,
        "stats": {},
        "reason": None,
    }
    try:
        import omni.physx

        interface = None
        for accessor in (
            "get_physx_cooking_interface",
            "get_physx_cooking_private_interface",
        ):
            factory = getattr(omni.physx, accessor, None)
            if callable(factory):
                try:
                    interface = factory()
                except Exception:
                    interface = None
            if interface is not None:
                record["accessor"] = accessor
                break
        if interface is None:
            record["reason"] = "no cooking interface accessor available on omni.physx"
            return record
        get_statistics = getattr(interface, "get_cooking_statistics", None)
        if callable(get_statistics):
            try:
                stats = get_statistics()
                scheduled = int(stats.total_scheduled_tasks)
                finished = int(stats.total_finished_tasks)
                record["stats"] = {
                    "total_scheduled_tasks": scheduled,
                    "total_finished_tasks": finished,
                    "total_finished_cache_hit_tasks": int(
                        stats.total_finished_cache_hit_tasks
                    ),
                    "total_finished_cache_miss_tasks": int(
                        stats.total_finished_cache_miss_tasks
                    ),
                    "running_tasks": scheduled - finished,
                }
            except Exception as error:
                record["get_cooking_statistics_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        if not record["stats"]:
            for name in dir(interface):
                if not name.startswith("get_num"):
                    continue
                method = getattr(interface, name)
                if not callable(method):
                    continue
                try:
                    record["stats"][name] = int(method())
                except Exception as error:
                    record["stats"][name] = (
                        f"ERROR:{type(error).__name__}:{error}"
                    )
        record["available"] = bool(record["stats"])
        if not record["available"]:
            record["reason"] = "cooking interface exposes no get_num* counters"
    except Exception as error:
        record["reason"] = f"{type(error).__name__}: {error}"
    return record


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
        session_text = SESSION_DOC.read_text(encoding="utf-8")
        asset_contract = _asset_dir_contract()
        frozen_values = _frozen_value_contract()
        frozen_science = _frozen_d362_science_source_contract()
        allowed_dirty = set(
            prereg.get("runtime_overlay_contract", {}).get("allowed_dirty_paths", [])
        )
        checks = {
            "prereg_sha_exact": _sha(PREREG_PATH) == EXPECTED_PREREG_SHA256,
            "prereg_status_frozen": prereg.get("status") == "PREREGISTERED_NOT_EXECUTED",
            "prereg_case_exact": prereg.get("case") == CASE
            and prereg.get("case_name") == CASE_NAME,
            "prereg_new_variables_exact": prereg.get("new_variables") == NEW_VARIABLES,
            "prereg_leg_registered": _LEG in (LEG_A, LEG_B)
            and _LEG in prereg.get("legs", {}),
            "single_invocation_marker": invocation.get("leg") == _LEG
            and invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("invocation_index") == 1
            and invocation.get("automatic_retry") is False
            and invocation.get("preregistration_sha256") == _sha(PREREG_PATH),
            "registered_parent_supervisor": supervisor_pid > 0
            and os.getppid() == supervisor_pid
            and invocation.get("supervisor_pid") == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest()
            == invocation.get("worker_token_sha256"),
            "head_origin_and_preregistered_base_exact": _git_head()
            == _git_head("origin/master")
            == prereg.get("git_baseline", {}).get("head")
            == BASE_GIT,
            "git_dirty_subset_of_allowlist": set(_status_paths()) <= allowed_dirty,
            "input_hashes_exact": _input_hashes() == prereg.get("frozen_input_hashes"),
            "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
            "leg_asset_freeze_contract": asset_contract["pass"] is True,
            "frozen_value_contract": frozen_values["pass"] is True,
            "frozen_d362_science_source_contract": frozen_science["pass"] is True,
            "registered_python": Path(sys.executable).resolve()
            == Path(REGISTERED_PYTHON).resolve(),
            "display_headless_device": args.headless is True
            and int(args.livestream) == 0
            and str(args.device) == "cuda:0",
            "registered_base_git_in_session_doc": session_text.count(BASE_GIT) >= 1,
            "session_doc_heading": "## 3. D407 확정 설계" in session_text,
            "runtime_modules_absent_before_applauncher": not early_runtime_modules,
            "gpu_resource_gate": int(gpu.get("memory_free_mib", 0)) >= MIN_GPU_FREE_MIB,
            "ram_resource_gate": int(gpu.get("ram_available_bytes", 0))
            >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D407_WORKER_PREFLIGHT_V1",
            "utc": _utc_now(),
            "pid": os.getpid(),
            "leg": _LEG,
            "early_runtime_modules": early_runtime_modules,
            "gpu_and_ram": gpu,
            "leg_asset_freeze_contract": asset_contract,
            "frozen_value_contract": frozen_values,
            "frozen_d362_science_source_contract": frozen_science,
            "checks": checks,
            "pass": all(checks.values()),
        }
        _write_json_x(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker_preflight", "complete", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D407 worker preflight STOP: {checks}")

        from isaaclab.app import AppLauncher

        _marker("AppLauncher", "start")
        launcher = AppLauncher(
            {
                "headless": True,
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
        launcher_report = _resolved_headless_launcher(launcher)
        _marker("AppLauncher", "complete", {"pass": launcher_report.get("pass")})
        if not launcher_report.get("pass"):
            raise RuntimeError(f"D407 headless launcher contract failed: {launcher_report}")

        import carb
        import omni.timeline

        args.robot_usd_path = LEG_ASSET_DIRS[_LEG] / "roarm_m3.usd"
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
            "controlled_d407_sample_count": 0,
            "semantics": "reset-internal work is reported separately and excluded from the 200+300 controlled rows",
        }
        _marker("reset", "complete", reset_internal_transition)
        settings = carb.settings.get_settings()
        previous_physx = settings.get(PHYSX_COLLIDER_SETTING)
        previous_play = settings.get(PLAY_SIMULATIONS_SETTING)
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
        reset_pause = _pause_timeline(inner, timeline)
        if not reset_pause["pass"]:
            raise RuntimeError("D407 reset pause/commit bridge failed")
        settings.set(PHYSX_COLLIDER_SETTING, 2)

        stage_contract = d351.d333._stage_contract(inner)
        sensor_contract, filter_map = d351.d333._sensor_contract(inner)
        actuator_contract = _actuator_contract(inner)
        object_contract = _object_spawn_contract(inner)
        _marker("corrected_d348_audit", "start")
        corrected = d351.d349._corrected_live_audit()
        _marker("corrected_d348_audit", "complete", {"pass": corrected.get("pass")})
        _marker("live_binding", "start")
        topology_parts, live_binding = _build_topology_parts_for_leg(inner)
        _marker("live_binding", "complete", {"pass": live_binding.get("pass")})
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
        cylinder_geometry = _cylinder_runtime_geometry_probe(inner)
        physics_settings = _runtime_physics_settings_probe(inner)
        cook_provenance = _physx_cook_cache_provenance()
        corrected_key = (
            "corrected_d348_128_of_128"
            if _LEG == LEG_A
            else "corrected_d348_128_of_128_historical_evidence_audit"
        )
        runtime_checks = {
            "stage_sole_support": stage_contract.get("hard_contract_pass") is True,
            "sensor_four_filters": sensor_contract.get("hard_contract_pass") is True,
            "actuators_frozen_80_4_2p5_3p14": actuator_contract["pass"],
            "object_and_dt_frozen": object_contract["pass"],
            corrected_key: corrected.get("pass") is True
            and corrected.get("checks", {}).get("all_parts_corrected_pass_128_of_128") is True,
            "live_binding_registered_exact": live_binding.get("pass") is True
            and part_counts == {"link5": 64, "gripper_link": LEG_GRIPPER_PARTS[_LEG]},
            "runtime_capacity_registered_exact": capacity_contract.get("pass") is True,
            "joint_order_exact": list(inner._robot.joint_names) == list(d351.d332.ALL_JOINT_NAMES),
            "counter_zero_before_physics": int(inner._sim_step_counter) == 0,
        }
        prerequisites = {
            "artifact": "D407_RUNTIME_PREREQUISITES_V1",
            "leg": _LEG,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "actuator_contract": actuator_contract,
            "object_contract": object_contract,
            "corrected_d348": corrected,
            "live_binding": live_binding,
            "live_part_counts": part_counts,
            "runtime_contact_capacity": capacity_contract,
            "cylinder_runtime_geometry": cylinder_geometry,
            "runtime_physics_settings": physics_settings,
            "physx_cook_cache_provenance": cook_provenance,
            "reset_internal_transition": reset_internal_transition,
            "checks": runtime_checks,
            "pass": all(runtime_checks.values()),
        }
        if _LEG == LEG_B:
            prerequisites["corrected_d348_authority"] = (
                "D347/D348 frozen-evidence historical audit; not a live topology proof for leg B"
            )
        _write_json_x(PREREQUISITE_PATH, prerequisites)
        if not prerequisites["pass"]:
            raise RuntimeError(f"D407 runtime prerequisites STOP: {runtime_checks}")

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
            raise RuntimeError(f"D407 exact frozen OPEN state failed: {initial_state_checks}")
        _initialize_prefix_writer()
        baseline_object_ref = np.asarray(initial_obj_pos, dtype=np.float64)
        root_ref_pos, root_ref_quat = d351.d333._root_pose(inner)
        capture_event_metadata: dict[str, Any] = {"contact": None, "motion": None}
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
        resume = _resume_timeline(inner, timeline)
        if not resume["pass"]:
            raise RuntimeError("D407 could not resume timeline for OPEN baseline")

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
                if len(baseline_rows) >= 2
                and capture_event_metadata["contact"] is None
                else []
            )
            motion_confirmed = bool(
                len(baseline_rows) >= 2
                and capture_event_metadata["motion"] is None
                and _motion_confirmed_pair(baseline_rows[-2:])
            )
            if confirmed_labels:
                capture_event_metadata["contact"] = {
                    "phase": "frozen_open_baseline",
                    "onset_phase_step": step - 1,
                    "confirmation_phase_step": step,
                    "qualifying_body_labels": confirmed_labels,
                }
            if motion_confirmed:
                capture_event_metadata["motion"] = {
                    "phase": "frozen_open_baseline",
                    "onset_phase_step": step - 1,
                    "confirmation_phase_step": step,
                }
            if step == 0 or (step + 1) % 10 == 0:
                _marker("frozen_open_baseline", "progress", {"completed": step + 1, "requested": BASELINE_STEPS})
        baseline = _baseline_statistics(
            baseline_rows,
            stage_contract_pass=stage_contract.get("hard_contract_pass") is True,
            sensor_contract_pass=sensor_contract.get("hard_contract_pass") is True,
        )
        if not baseline["pass"]:
            _seal_prefix("open_baseline_hard_gate_fail_stop")

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
                    if len(closure_rows) >= 2
                    and capture_event_metadata["contact"] is None
                    else []
                )
                motion_confirmed = bool(
                    len(closure_rows) >= 2
                    and capture_event_metadata["motion"] is None
                    and _motion_confirmed_pair(closure_rows[-2:])
                )
                if confirmed_labels:
                    capture_event_metadata["contact"] = {
                        "phase": "q5_close_observation",
                        "onset_phase_step": step - 1,
                        "confirmation_phase_step": step,
                        "qualifying_body_labels": confirmed_labels,
                    }
                if motion_confirmed:
                    capture_event_metadata["motion"] = {
                        "phase": "q5_close_observation",
                        "onset_phase_step": step - 1,
                        "confirmation_phase_step": step,
                    }
                if step == 0 or (step + 1) % 10 == 0:
                    _marker("q5_close_observation", "progress", {"completed": step + 1, "requested": CLOSURE_MAX_STEPS})
            _seal_prefix("full_500_step_horizon_complete")
        else:
            _marker("q5_close_observation", "not_run", {"reason": "frozen OPEN baseline hard gate failed"})

        # D362 paused the timeline through its final viewport capture; the
        # headless worker pauses explicitly so the inherited restore checks
        # keep their exact semantics.
        final_pause = _pause_timeline(inner, timeline)
        if not final_pause["pass"]:
            raise RuntimeError("D407 final pause bridge failed")
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
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
        q0_q4_drift = _q0_q4_drift_summary(baseline_rows, closure_rows, closure)
        prefix_audit = _finalize_prefix_audit(all_rows)
        _write_json_x(TRACE_JSON_PATH, all_rows)
        _write_trace_csv(TRACE_CSV_PATH, all_rows)
        rerun = _write_rerun(topology_parts, all_rows, baseline, closure)
        sheet = _build_beginner_sheet(
            capture_event_metadata, baseline, closure, all_rows, topology_parts
        )
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
            "all_qualifying_robot_force_points_finite_for_observability": qualifying_point_contract[
                "all_qualifying_robot_event_points_finite"
            ],
            "rerun_rrd_rbl_verify_screenshot": rerun.get("pass") is True,
            "screenshot_physical_1920x1080_exact": rerun.get("checks", {}).get(
                "screenshot_physical_1920x1080_exact"
            )
            is True,
            "durable_prefix_exact_reconciliation": prefix_audit.get("pass") is True,
            "beginner_sheet": sheet["pass"],
        }
        operational_trace_checks = {
            "frozen_open_baseline_hard_gate_pass": baseline["pass"] is True,
            "trace_exact_500_rows": len(all_rows)
            == BASELINE_STEPS + CLOSURE_MAX_STEPS,
            "final_global_step_500_exact": bool(all_rows)
            and int(all_rows[-1]["global_step"])
            == BASELINE_STEPS + CLOSURE_MAX_STEPS,
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
            "closure_horizon_exact": len(closure_rows) == CLOSURE_MAX_STEPS,
            "inputs_unchanged": _input_hashes() == prereg["frozen_input_hashes"],
            "sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "settings_restored": all(restore_checks.values()),
            "runtime_contact_capacity_contract": capacity_contract.get("pass")
            is True,
            "durable_prefix_contract": prefix_audit.get("pass") is True,
        }
        summary = {
            "artifact": "D407_WORKER_SUMMARY_V1",
            "case": CASE,
            "leg": _LEG,
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
            "capture_event_metadata": capture_event_metadata,
            "qualifying_robot_point_contract": qualifying_point_contract,
            "beginner_sheet": sheet,
            "rerun_validation_path": _rel(RERUN_VALIDATION_PATH),
            "sdf_contact_point_reporting_status": (
                "first_live_observation_no_prior_guarantee" if _LEG == LEG_B else None
            ),
            "controlled_physics_steps": _CONTROLLED_STEPS,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "restore_checks": restore_checks,
            "operational_trace_checks": operational_trace_checks,
            "observability_checks": observability_checks,
            "observability_artifact_pass": all(observability_checks.values()),
            "interface_visibility": None,
            "interface_visibility_pending_manual_original_resolution_inspection": True,
            "target_ik_path_changed": False,
            # The single registered new variable IS a gripper collision
            # representation change on leg B; stating False here (the D362
            # literal) would be untrue for that leg.  Material, mass,
            # actuator, and physics contracts remain frozen on both legs.
            "gripper_collision_representation_changed": _LEG == LEG_B,
            "material_mass_actuator_physics_changed": False,
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
                    "artifact": "D407_FAILURE_PREFIX_AUDIT_V1",
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
                    "artifact": "D407_FAILURE_PREFIX_AUDIT_WRITE_ERROR_V1",
                    "error": f"{type(audit_error).__name__}: {audit_error}",
                }
        if not WORKER_EXCEPTION_PATH.exists():
            _write_json_x(
                WORKER_EXCEPTION_PATH,
                {
                    "artifact": "D407_WORKER_EXCEPTION_STOP_V1",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("_worker",), required=True)
    parser.add_argument("--leg", choices=(LEG_A, LEG_B), required=True)
    parser.add_argument("--out_dir", type=Path, default=ATTEMPT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if Path(args.out_dir).resolve() != ATTEMPT_DIR.resolve():
        raise RuntimeError("D407 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D407 seed drift")
    if args.leg not in (LEG_A, LEG_B):
        raise RuntimeError("D407 leg drift")
    _configure_leg(args.leg)
    args.headless = True
    args.livestream = 0
    args.device = "cuda:0"
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
