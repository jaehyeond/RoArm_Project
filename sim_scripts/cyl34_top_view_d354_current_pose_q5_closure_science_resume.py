#!/usr/bin/env python3
"""D354: resume the frozen D351 current-pose q5 closure science once.

This forward-only harness combines two already-registered contracts without
changing either scientific geometry or physics configuration:

* D353's conditional main-thread Timeline.commit() zero-step PAUSE bridge; and
* D351's frozen moving-jaw OPEN-to-CLOSED q5 closure measurement/Viewer/Rerun.

The q5 evaluator remains fail closed until the new worker has reproduced the
D353 bridge and completed the corrected/live/raw representation boundaries.
No target, IK, path, asset, decomposition, gate, material, mass, actuator,
renderer, solver, or physics setting is changed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import faulthandler
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import (  # noqa: E402
    cyl34_top_view_d351_zero_step_closure_geometry as d351,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d352_d351_validate_phase_localization_watchdog as d352,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d353_timeline_pause_pending_state_commit_bridge as d353,
)


CASE = "g0a_d354"
CASE_NAME = "current_pose_q5_closure_science_resume"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354"
HARNESS = Path(__file__).resolve()
SESSION_DOC = (
    REPO
    / "claudedocs/session_20260716_grasp_g0a_d354_current_pose_q5_closure_science_resume.md"
)
START_HERE = REPO / "START_HERE.md"
EXPECTED_HEAD = "b7beb91997859a5ddb2b0407388e80aed45898dc"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
SEED = 33201

INACTIVITY_WATCHDOG_S = 120.0
TOTAL_WATCHDOG_S = 900.0
FAULT_DUMP_GRACE_S = 5.0
TERM_GRACE_S = 30.0
GPU_SAMPLE_PERIOD_S = 1.0
VIEWER_HOLD_SECONDS = 120.0
SUPERVISOR_PID_ENV = "D354_SUPERVISOR_PID"
WORKER_LAUNCH_TOKEN_ENV = "D354_WORKER_LAUNCH_TOKEN"

INHERITED_D351_SCIENTIFIC_VARIABLES = [
    "moving_jaw_actual_contact_surface_binding",
    "frozen_pose_q5_closure_sweep",
]
NEW_OPERATIONAL_VARIABLES = ["positive_asset_write_immutability_aggregation"]
NEW_SCIENTIFIC_VARIABLES: list[str] = []
NEW_PHYSICAL_VARIABLES: list[str] = []

PARAMETER_PATH = OUT_DIR / "d354_parameter_freeze_audit.json"
GPU_HARDWARE_PATH = OUT_DIR / "d354_gpu_hardware_contract.json"
PREREG_PATH = OUT_DIR / "d354_preregistration.json"
SUPERVISOR_PREFLIGHT_PATH = OUT_DIR / "d354_supervisor_preflight.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d354_validate_preflight.json"
MARKER_PATH = OUT_DIR / "d354_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d354_worker_stdout_stderr.log"
GPU_TELEMETRY_PATH = OUT_DIR / "d354_gpu_cpu_telemetry.jsonl"
FAULT_PATH = OUT_DIR / "d354_faulthandler.txt"
LIVE_BINDING_PATH = OUT_DIR / "d354_live_topology_runtime_binding.json"
RAW_CONTRACT_PATH = OUT_DIR / "d354_raw_source_contract.json"
BRIDGE_PATH = OUT_DIR / "d354_zero_step_science_arm_bridge.json"
MOVING_BINDING_PATH = OUT_DIR / "d354_moving_jaw_surface_binding.json"
MEASUREMENT_PATH = OUT_DIR / "d354_zero_step_closure_geometry_measurement.json"
SWEEP_CSV_PATH = OUT_DIR / "d354_q5_closure_sweep.csv"
OVERLAY_PATH = OUT_DIR / "d354_viewer_overlay_contract.json"
CAPTURE_PATH = OUT_DIR / "d354_viewer_capture_contract.json"
RRD_PATH = OUT_DIR / "d354_zero_step_closure_geometry.rrd"
RBL_PATH = OUT_DIR / "d354_zero_step_closure_geometry.rbl"
RERUN_PNG_PATH = OUT_DIR / "d354_zero_step_closure_geometry_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d354_rerun_validation.json"
AUTOMATED_PATH = OUT_DIR / "d354_automated_summary.json"
AUTOMATED_MD_PATH = OUT_DIR / "d354_automated_report.md"
SCIENCE_SUMMARY_PATH = OUT_DIR / "d354_science_resume_summary.json"
ATTESTATION_PATH = OUT_DIR / "d354_zero_step_science_attestation.json"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d354_runtime_exception.json"
WATCHDOG_PROC_PATH = OUT_DIR / "d354_watchdog_proc_snapshot.json"
RAW_SUPERVISOR_PATH = OUT_DIR / "d354_inherited_supervisor_raw_audit.json"
SUPERVISOR_AUDIT_PATH = OUT_DIR / "d354_supervisor_audit.json"
MANUAL_PATH = OUT_DIR / "d354_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d354_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d354_completion_summary.json"
COMPLETION_MD_PATH = OUT_DIR / "d354_completion_report.md"

VIEWER_PNGS = {
    "open_physx": OUT_DIR / "d354_open_actual_physx_colliders.png",
    "decision_physx": OUT_DIR / "d354_decision_or_open_fallback_actual_physx_colliders.png",
    "decision_colored": OUT_DIR / "d354_decision_or_open_fallback_colored_64plus64.png",
    "decision_side": OUT_DIR / "d354_decision_or_open_fallback_side_geometry.png",
}

VERDICT_INPUT = "D354_FROZEN_INPUT_OR_ZERO_STEP_CONTRACT_FAIL_STOP"
VERDICT_ORDER = "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP"
VERDICT_BINDING = "D354_MOVING_JAW_SURFACE_BINDING_FAIL_STOP"
VERDICT_ELIGIBLE = "D354_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE"
VERDICT_REPAIR = "D354_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED"
VERDICT_VISUAL = "D354_VIEWER_OR_RERUN_CONTRACT_FAIL_STOP"
VERDICT_PENDING_SUFFIX = "_MANUAL_PENDING"

USER_SIDECAR_DIR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
USER_SIDECAR_HASHES = {
    "README.md": "35e39f584737c888bcf7dfab6154c55c5d13d4154ee7f2042073e1c0a7e18783",
    "d334_collision_table_academic.html": "6d38933f959eba916208ec04a329ba25e2bd753c90720576010c222a8bda679c",
    "d334_collision_table_academic.png": "ddc9db2795f4d66b2564adf156829e6a143a599ceb72f6bb9fa28ab25e68a183",
}

D351_IMMUTABLE_HASHES = {
    "sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py": (
        "3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d"
    ),
    "claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md": (
        "20367375e05ce8cffb47f86ff0c1645a3544f5bf62516fe2e16a98919c356a06"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_parameter_freeze_audit.json": (
        "98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2"
    ),
}

D353_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d353"
D353_IMMUTABLE_HASHES = {
    "sim_scripts/cyl34_top_view_d353_timeline_pause_pending_state_commit_bridge.py": (
        "ab37141d721f5ca9571e9008a065344b3fb818ac9164fd56cda3c5617952cda9"
    ),
    "claudedocs/session_20260715_grasp_g0a_d353_timeline_pause_pending_state_commit_bridge.md": (
        "5fbdc1801b313a88d7cfa07c2d6d311198919b57d103dd042971d250c3dc4715"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_faulthandler.txt": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_gpu_cpu_telemetry.jsonl": (
        "6cfba7aa496ad8ff84a8cdb74ac39861c480092bc025d993a6e59af8f5a3c95d"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_gpu_hardware_contract.json": (
        "3bd3e947284ddd31cd39843b280cbd27ef74ddd214fe080d972b57be2237c675"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_inherited_supervisor_raw_audit.json": (
        "bf2b48cb93c09e4a66f17c6d72b60eb48686dbefcb22e7a48557f301c08f284d"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_live_topology_runtime_binding.json": (
        "b7a8897dd33cf6f455f14862d6dc38bca87fb64e6dc7d0a22dbc715188f4e206"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_parameter_freeze_audit.json": (
        "e7dc9e5a0c16e89508ca289299c6c04c195721bc3574493fac784fecce3f2be7"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_phase_markers.jsonl": (
        "f973b5797398e341f8ffdc4561422224441ae7215742364e51745c301ed4d51e"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_preregistration.json": (
        "f3534f1e9cc03b3d114b5bbef31566c8f55103e14c1df87a4a9b19dd1e116cc7"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_raw_source_contract.json": (
        "325004fdc98f01bc01e5534d96ce1e2abe410b47d21029f5961446f2b53f243b"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_supervisor_audit.json": (
        "65c57e69f017d7d7afbb5fd03b10b56e87bb1bbc442b1351a25c18a0a55a31a5"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_supervisor_preflight.json": (
        "821747fbf1d20d68a53e25e7b180888b47850b63ddf6a6cbf8b2314735041e06"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_timeline_commit_bridge_attestation.json": (
        "4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_timeline_commit_bridge_summary.json": (
        "7b8740cb176b3450936e796e6aa7dae72489fe625d08bef71da245e1b0be299a"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_timeline_commit_event_contract.json": (
        "0b1d47671fe31206398961dc18f4d66912ba3fd59cf1634059c326a5e67a0b61"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_validate_preflight.json": (
        "9e267ba0af6272fb4de976dfb05467166595d6f8ecab451db804efd062f67dc8"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_worker_stdout_stderr.log": (
        "a1a8ee026507da2aad0fd96904764ae81fa299a8b4b77672f2a172f419870586"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d353/d353_zero_step_bridge_contract.json": (
        "cfa7d4f0e28feaefab4714e8a0e1d33677ba288392a768365a311cc1c16e9b3e"
    ),
}
D353_ROOT_INVENTORY = sorted(
    path.name
    for path in [REPO / relative for relative in D353_IMMUTABLE_HASHES]
    if path.parent == D353_DIR
)

_ORIGINAL_D351_EVALUATE = d351._evaluate_q5
_ORIGINAL_D351_SET_STATE = d351._set_state_only
_ORIGINAL_D351_PUMP = d351._pump_frames
_ORIGINAL_D351_RERUN = d351._run_rerun
_ORIGINAL_D351_VIEWER = d351._run_viewer
_ORIGINAL_MAKE_RUNTIME_ENV = d351.d333._make_runtime_env
_ORIGINAL_BUILD_LIVE = d351.d349._build_live_topology_parts
_ORIGINAL_BUILD_RAW = d351.d339._build_retained_raw_shapes
_ORIGINAL_CORRECTED_AUDIT = d351.d349._corrected_live_audit
_ORIGINAL_D352_GPU = d352._gpu_hardware_contract
_ORIGINAL_D352_SAMPLE = d352._sample_gpu_cpu
_ORIGINAL_D352_SUPERVISOR = d352._run_supervisor

_PROCESS_START_MONOTONIC_NS = time.monotonic_ns()
_RUN_NONCE = "not-loaded"
_MARKER_SEQUENCE = 0
_MARKER_LOCK = threading.Lock()
_CURRENT_PHASE = "not-started"
_ACTIVE_INNER: Any | None = None
_RAW_TIMELINE: Any | None = None
_TIMELINE_PROXY: Any | None = None
_ORIGINAL_TIMELINE_GETTER: Any | None = None
_BRIDGE_EVENT: dict[str, Any] | None = None
_MATCHED_D352: dict[str, Any] | None = None
_POST_LIVE_SNAPSHOT: dict[str, Any] | None = None
_POST_RAW_SNAPSHOT: dict[str, Any] | None = None
_CORRECTED_REPORT: dict[str, Any] | None = None
_LIVE_REPORT: dict[str, Any] | None = None
_LIVE_TRACE: dict[str, Any] | None = None
_RAW_REPORT: dict[str, Any] | None = None
_SCIENCE_ARMED = False
_Q5_GATE_ATTEMPTS = 0
_Q5_INVOCATIONS = 0
_Q5_CACHE_MISSES = 0
_Q5_CACHE_MISS_SUCCESSES = 0
_PRIMARY_CACHE_ID: int | None = None
_REPEAT_CACHE_IDS: set[int] = set()
_AUX_STATE_ATTEMPTS = 0
_AUX_STATE_SUCCESSES = 0
_PUMP_UPDATES = 0
_PUMP_GUARD_FAILURES = 0
_LAST_INNER_SENTINELS: dict[str, Any] | None = None
_LAUNCHER_REPORT: dict[str, Any] | None = None
_SOURCE_BEFORE: dict[str, Any] | None = None
_INPUT_BEFORE: dict[str, str] | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _payload_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_json_exact(path: Path, payload: Any) -> None:
    _write_bytes(path, _payload_bytes(payload))


def _write_text_exact(path: Path, value: str) -> None:
    _write_bytes(path, value.encode("utf-8"))


def _append_jsonl(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, sort_keys=True, default=_json_default) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _git_head(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _status_scope_pass(status: dict[str, str]) -> bool:
    exact = {_rel(START_HERE), _rel(SESSION_DOC), _rel(HARNESS)}
    prefix = _rel(OUT_DIR) + "/"
    return all(path in exact or path.startswith(prefix) for path in status)


def _marker(actor: str, phase: str, event: str, details: Any | None = None) -> dict[str, Any]:
    global _MARKER_SEQUENCE
    with _MARKER_LOCK:
        _MARKER_SEQUENCE += 1
        row = {
            "artifact": "D354_DURABLE_PHASE_MARKER_V1",
            "run_nonce": _RUN_NONCE,
            "actor": actor,
            "pid": os.getpid(),
            "sequence": _MARKER_SEQUENCE,
            "phase": phase,
            "event": event,
            "wall_time_ns": time.time_ns(),
            "wall_time_iso": dt.datetime.now().astimezone().isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "process_elapsed_s": (
                time.monotonic_ns() - _PROCESS_START_MONOTONIC_NS
            )
            / 1.0e9,
            "timeline_commit_attempt_count": d353._COMMIT_ATTEMPT_COUNT,
            "timeline_commit_call_count": d353._COMMIT_CALL_COUNT,
            "science_armed": _SCIENCE_ARMED,
            "q5_evaluator_invocations": _Q5_INVOCATIONS,
            "q5_cache_misses": _Q5_CACHE_MISSES,
            "auxiliary_state_only_writes": _AUX_STATE_SUCCESSES,
            "details": details,
        }
        _append_jsonl(MARKER_PATH, row)
        return row


def _set_phase(value: str) -> None:
    global _CURRENT_PHASE
    _CURRENT_PHASE = value


def _user_sidecar_contract() -> dict[str, Any]:
    inventory = sorted(path.name for path in USER_SIDECAR_DIR.iterdir() if path.is_file())
    non_files = sorted(path.name for path in USER_SIDECAR_DIR.iterdir() if not path.is_file())
    rows = {}
    for name, expected in USER_SIDECAR_HASHES.items():
        path = USER_SIDECAR_DIR / name
        rows[name] = {
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "expected_sha256": expected,
        }
    checks = {
        "root_inventory_exact": inventory == sorted(USER_SIDECAR_HASHES),
        "non_file_inventory_empty": not non_files,
        "all_hashes_exact": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
    }
    return {
        "role": "user-owned non-scientific sidecar; read-only",
        "root_inventory": inventory,
        "non_file_inventory": non_files,
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _immutable_prior_contract() -> dict[str, Any]:
    rows = {}
    expected = {**D351_IMMUTABLE_HASHES, **D353_IMMUTABLE_HASHES}
    for relative, digest in expected.items():
        path = REPO / relative
        rows[relative] = {
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "expected_sha256": digest,
        }
    d353_inventory = sorted(path.name for path in D353_DIR.iterdir() if path.is_file())
    attestation = _json(D353_DIR / "d353_timeline_commit_bridge_attestation.json")
    supervisor = _json(D353_DIR / "d353_supervisor_audit.json")
    checks = {
        "all_hashes_exact": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
        "d353_root_inventory_exact_17": d353_inventory == D353_ROOT_INVENTORY,
        "d353_attestation_pass": attestation.get("pass") is True,
        "d353_attested_controlled_steps_zero": (
            attestation.get("d353_controlled_physics_steps") == 0
        ),
        "d353_attestation_no_science": (
            attestation.get("d353_q5_science_evaluation_count") == 0
            and attestation.get("scientific_verdict") is None
        ),
        "d353_supervisor_pass": supervisor.get("pass") is True,
        "d353_supervisor_verdict_exact": supervisor.get("operational_verdict")
        == "D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE",
    }
    return {
        "rows": rows,
        "d353_root_inventory": d353_inventory,
        "d353_attestation": {
            "path": _rel(D353_DIR / "d353_timeline_commit_bridge_attestation.json"),
            "sha256": _sha(D353_DIR / "d353_timeline_commit_bridge_attestation.json"),
            "pass": attestation.get("pass"),
        },
        "d353_supervisor": {
            "path": _rel(D353_DIR / "d353_supervisor_audit.json"),
            "sha256": _sha(D353_DIR / "d353_supervisor_audit.json"),
            "pass": supervisor.get("pass"),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _gpu_hardware_contract() -> dict[str, Any]:
    row = copy.deepcopy(_ORIGINAL_D352_GPU())
    row["artifact"] = "D354_GPU_HARDWARE_CONTRACT_V1"
    row["workload_interpretation"] = {
        "one_environment_zero_step_geometry_case": True,
        "gpu_telemetry_is_active_time_not_warp_occupancy": True,
        "no_batch_or_kernel_tuning_authorized": True,
        "reason": (
            "Changing environment count, kernels, solver, renderer, or physics for SM "
            "occupancy would change the frozen case; cuda:0 is used without such mutation."
        ),
    }
    return row


def _sample_gpu_cpu(worker_pid: int, sample_index: int, worker_process: Any | None) -> dict[str, Any]:
    row = copy.deepcopy(_ORIGINAL_D352_SAMPLE(worker_pid, sample_index, worker_process))
    row["artifact"] = "D354_GPU_CPU_TELEMETRY_SAMPLE_V1"
    return row


def _parameter_freeze_audit() -> dict[str, Any]:
    d351_parameter = d351._parameter_audit()
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))

    def call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    checks = {
        "head_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "new_scientific_variables_zero": NEW_SCIENTIFIC_VARIABLES == [],
        "new_physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "single_operational_bookkeeping_variable_exact": NEW_OPERATIONAL_VARIABLES
        == ["positive_asset_write_immutability_aggregation"],
        "inherited_d351_variables_exact": INHERITED_D351_SCIENTIFIC_VARIABLES
        == d351.NEW_VARIABLES,
        "d351_parameter_contract_pass": d351_parameter["pass"],
        "d352_live_trace_line_map_frozen_original_pass": d352._live_trace_line_map()[
            "pass"
        ],
        "watchdogs_exact": INACTIVITY_WATCHDOG_S == 120.0 and TOTAL_WATCHDOG_S == 900.0,
        "viewer_hold_exact_120": VIEWER_HOLD_SECONDS == 120.0,
        "d353_commit_primitive_called_once_in_source": calls.count(
            "d353._commit_pending_pause"
        )
        == 1,
        "no_direct_timeline_commit_in_d354_source": not any(
            name.endswith(".commit") for name in calls
        ),
        "no_sim_step_call_in_d354_source": not any(
            name.endswith(".step") for name in calls
        ),
        "no_target_ik_path_mutation_api": not any(
            name.rsplit(".", 1)[-1]
            in {"set_target", "solve_ik", "compute_ik", "trajectory_plan"}
            for name in calls
        ),
        "no_gpu_clock_power_or_kernel_mutation": not any(
            name.rsplit(".", 1)[-1]
            in {"set_clock", "set_power_limit", "profiler_start", "kernel_replay"}
            for name in calls
        ),
    }
    return {
        "artifact": "D354_PARAMETER_FREEZE_AUDIT_V1",
        "case": CASE,
        "new_operational_variables": NEW_OPERATIONAL_VARIABLES,
        "new_scientific_variables": NEW_SCIENTIFIC_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "inherited_d351_scientific_variables": INHERITED_D351_SCIENTIFIC_VARIABLES,
        "d351_parameter_contract": d351_parameter,
        "control_bridge": {
            "authority": "D353 attestation plus fresh conditional commit reproduction",
            "commit_policy": "one commit only if the reset PAUSE request remains pending PLAY",
        },
        "watchdog": {
            "marker_inactivity_s": INACTIVITY_WATCHDOG_S,
            "total_wall_clock_s": TOTAL_WATCHDOG_S,
            "automatic_retry": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _runtime_expected_before_final_supervisor() -> set[str]:
    return {
        path.name
        for path in (
            PARAMETER_PATH,
            GPU_HARDWARE_PATH,
            PREREG_PATH,
            SUPERVISOR_PREFLIGHT_PATH,
            WORKER_PREFLIGHT_PATH,
            MARKER_PATH,
            WORKER_LOG_PATH,
            GPU_TELEMETRY_PATH,
            FAULT_PATH,
            LIVE_BINDING_PATH,
            RAW_CONTRACT_PATH,
            BRIDGE_PATH,
            MOVING_BINDING_PATH,
            MEASUREMENT_PATH,
            SWEEP_CSV_PATH,
            OVERLAY_PATH,
            CAPTURE_PATH,
            RRD_PATH,
            RBL_PATH,
            RERUN_PNG_PATH,
            RERUN_VALIDATION_PATH,
            AUTOMATED_PATH,
            AUTOMATED_MD_PATH,
            SCIENCE_SUMMARY_PATH,
            ATTESTATION_PATH,
            RAW_SUPERVISOR_PATH,
            *VIEWER_PNGS.values(),
        )
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"forward-only D354 output already nonempty: {OUT_DIR}")
    status = d351._git_status()
    parameter = _parameter_freeze_audit()
    gpu = _gpu_hardware_contract()
    immutable = _immutable_prior_contract()
    sidecar = _user_sidecar_contract()
    environment = d351._environment_contract()
    inputs = d351._input_hashes()
    sources = d351.d349._source_inventories()
    state_hashes = {
        "start_here": _sha(START_HERE),
        "session_doc": _sha(SESSION_DOC),
    }
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d354": _status_scope_pass(status),
        "parameter_freeze_pass": parameter["pass"],
        "gpu_hardware_pass": gpu["pass"],
        "environment_pass": environment["pass"],
        "d351_input_hashes_exact": inputs == d351.EXPECTED_INPUT_HASHES,
        "prior_d351_d353_immutable": immutable["pass"],
        "user_sidecar_read_only_exact": sidecar["pass"],
        "state_docs_exist": START_HERE.is_file() and SESSION_DOC.is_file(),
        "start_here_active_d354": "D354 current-pose q5 closure-science resume"
        in START_HERE.read_text(encoding="utf-8"),
        "session_preregistered_before_run": "아직 prepare·Isaac 실행 전"
        in SESSION_DOC.read_text(encoding="utf-8"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D354 prepare STOP: {checks}")
    _write_json_exact(PARAMETER_PATH, parameter)
    _write_json_exact(GPU_HARDWARE_PATH, gpu)
    prereg = {
        "artifact": "D354_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "run_nonce": secrets.token_hex(32),
        "git_head": _git_head(),
        "origin_master": _git_head("origin/master"),
        "git_status": status,
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "harness_sha256": _sha(HARNESS),
        "state_hashes": state_hashes,
        "parameter_audit_sha256": _sha(PARAMETER_PATH),
        "gpu_hardware_sha256": _sha(GPU_HARDWARE_PATH),
        "new_operational_variables": NEW_OPERATIONAL_VARIABLES,
        "new_scientific_variables": NEW_SCIENTIFIC_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "inherited_d351_scientific_variables": INHERITED_D351_SCIENTIFIC_VARIABLES,
        "question": (
            "At the frozen D350 q0-q4/object pose, does zero-time q5 closure make "
            "the actual moving inner jaw meet the cylinder barrel first?"
        ),
        "control_authority": immutable["d353_attestation"],
        "positive_semantics": "pre-grasp barrel-closure eligibility only; never grasp/G0a",
        "watchdog": parameter["watchdog"],
        "one_validate_run_no_retry": True,
        "input_hashes": inputs,
        "source_inventories": sources,
        "environment": environment,
        "parameter_freeze": parameter,
        "gpu_hardware": gpu,
        "immutable_prior": immutable,
        "preexisting_user_files": sidecar,
        "expected_runtime_inventory_before_final_supervisor": sorted(
            _runtime_expected_before_final_supervisor()
        ),
        "prechecks": checks,
        "pass": all(checks.values()),
    }
    _write_json_exact(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": True}, sort_keys=True))
    return 0


def _supervisor_preflight(args: argparse.Namespace) -> dict[str, Any]:
    global _RUN_NONCE
    prereg = _json(PREREG_PATH)
    _RUN_NONCE = str(prereg["run_nonce"])
    existing = sorted(path.name for path in OUT_DIR.iterdir() if path.is_file())
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "fresh_supervisor_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_supervisor_nonce": prereg.get("prepare_process_identity", {}).get("nonce")
        != args.process_nonce,
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d354": _status_scope_pass(d351._git_status()),
        "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
        "state_hashes_exact": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "parameter_hash_exact": _sha(PARAMETER_PATH)
        == prereg.get("parameter_audit_sha256"),
        "gpu_hash_exact": _sha(GPU_HARDWARE_PATH) == prereg.get("gpu_hardware_sha256"),
        "input_hashes_exact": d351._input_hashes()
        == d351.EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "source_inventory_exact": d351.d349._source_inventories()
        == prereg.get("source_inventories"),
        "prior_immutable": _immutable_prior_contract()["pass"],
        "sidecar_read_only": _user_sidecar_contract() == prereg.get("preexisting_user_files"),
        "watchdogs_exact": float(args.inactivity_watchdog_s) == INACTIVITY_WATCHDOG_S
        and float(args.total_watchdog_s) == TOTAL_WATCHDOG_S,
        "display_exact_gui": os.environ.get("DISPLAY") == ":1",
        "registered_python_exact": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "environment_exact_preregistered": d351._environment_contract()
        == prereg.get("environment"),
        "prepare_inventory_exact": existing
        == sorted([PARAMETER_PATH.name, GPU_HARDWARE_PATH.name, PREREG_PATH.name]),
    }
    return {
        "artifact": "D354_SUPERVISOR_PREFLIGHT_V1",
        "case": CASE,
        "pid": os.getpid(),
        "process_nonce": args.process_nonce,
        "run_nonce": prereg.get("run_nonce"),
        "environment": {"display": os.environ.get("DISPLAY")},
        "checks": checks,
        "pass": all(checks.values()),
    }


def _worker_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    prereg = _json(PREREG_PATH)
    supervisor = _json(SUPERVISOR_PREFLIGHT_PATH)
    token = os.environ.get(WORKER_LAUNCH_TOKEN_ENV, "")
    supervisor_pid = int(os.environ.get(SUPERVISOR_PID_ENV, "-1"))
    app_checks = d351.d350._app_arg_checks(args)
    gpu_now = _gpu_hardware_contract()
    existing = {path.name for path in OUT_DIR.iterdir() if path.is_file()}
    required = {
        PARAMETER_PATH.name,
        GPU_HARDWARE_PATH.name,
        PREREG_PATH.name,
        SUPERVISOR_PREFLIGHT_PATH.name,
    }
    permitted = required | {
        MARKER_PATH.name,
        WORKER_LOG_PATH.name,
        GPU_TELEMETRY_PATH.name,
        FAULT_PATH.name,
    }
    checks = {
        "prereg_and_supervisor_pass": prereg.get("pass") is True
        and supervisor.get("pass") is True,
        "run_nonce_exact": _RUN_NONCE
        == prereg.get("run_nonce")
        == supervisor.get("run_nonce"),
        "fresh_worker_pid": os.getpid()
        not in {
            prereg.get("prepare_process_identity", {}).get("pid"),
            supervisor.get("pid"),
        },
        "parent_is_supervisor": supervisor_pid > 0
        and os.getppid() == supervisor_pid
        and supervisor.get("pid") == supervisor_pid,
        "launch_token_exact": bool(token)
        and hashlib.sha256(token.encode("utf-8")).hexdigest()
        == supervisor.get("worker_launch", {}).get("token_sha256"),
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d354": _status_scope_pass(d351._git_status()),
        "harness_and_state_hashes_exact": _sha(HARNESS) == prereg.get("harness_sha256")
        and {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "input_source_exact": d351._input_hashes() == prereg.get("input_hashes")
        and d351.d349._source_inventories() == prereg.get("source_inventories"),
        "prior_immutable": _immutable_prior_contract()["pass"],
        "sidecar_read_only": _user_sidecar_contract() == prereg.get("preexisting_user_files"),
        "headless_false": args.headless is False,
        "livestream_zero": int(args.livestream) == 0,
        "display_exact_gui": os.environ.get("DISPLAY") == ":1",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_4090_device_zero": bool(
            torch.cuda.is_available()
            and torch.cuda.current_device() == 0
            and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090 Laptop GPU"
        ),
        "gpu_contract_exact": gpu_now["pass"],
        "gpu_identity_matches_preregistered": (
            gpu_now.get("torch") == prereg.get("gpu_hardware", {}).get("torch")
            and gpu_now.get("nvidia_smi", {}).get("uuid")
            == prereg.get("gpu_hardware", {}).get("nvidia_smi", {}).get("uuid")
            and gpu_now.get("nvidia_smi", {}).get("driver_version")
            == prereg.get("gpu_hardware", {}).get("nvidia_smi", {}).get(
                "driver_version"
            )
        ),
        "environment_exact_preregistered": d351._environment_contract()
        == prereg.get("environment"),
        "app_args_exact": all(app_checks.values()),
        "viewer_hold_exact": float(args.viewer_hold_seconds) == VIEWER_HOLD_SECONDS,
        "runtime_inventory_required": required <= existing,
        "runtime_inventory_no_science_outputs_yet": existing <= permitted,
    }
    return {
        "artifact": "D354_VALIDATE_PREFLIGHT_V1",
        "case": CASE,
        "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "run_nonce": _RUN_NONCE,
        "preregistration_sha256": _sha(PREREG_PATH),
        "parameter_audit_sha256": _sha(PARAMETER_PATH),
        "harness_sha256": _sha(HARNESS),
        "environment": {
            "display": os.environ.get("DISPLAY"),
            "gpu": gpu_now,
            "app_arg_checks": app_checks,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _rewrite_d351_payload(path: Path, payload: Any) -> Any:
    row = copy.deepcopy(payload)
    if not isinstance(row, dict):
        return row
    if isinstance(row.get("artifact"), str) and row["artifact"].startswith("D351_"):
        row["artifact"] = "D354_" + row["artifact"][len("D351_") :]
    row["case"] = CASE
    if "new_variables" in row:
        row["new_variables"] = []
        row["inherited_d351_scientific_variables"] = INHERITED_D351_SCIENTIFIC_VARIABLES
    if "new_physical_variables" in row:
        row["new_physical_variables"] = []
    if path == MEASUREMENT_PATH:
        row["controlled_physics_steps"] = None
        row["controlled_physics_steps_candidate"] = 0
        row["controlled_steps_authority"] = "pending D354 fsync/reread attestation"
        row["execution_counters_at_measurement_write"] = _counter_report(row)
    if path == AUTOMATED_PATH:
        immutability = copy.deepcopy(row.get("immutability", {}))
        asset_false = immutability.pop("asset_write", None) is False
        immutability.pop("pass", None)
        if "git_scope_only_d351" in immutability:
            immutability["git_scope_only_d354"] = immutability.pop("git_scope_only_d351")
        positive_before = copy.deepcopy(immutability)
        immutability["asset_write_forbidden_and_absent"] = asset_false
        immutability["pass"] = bool(immutability) and all(
            value is True for key, value in immutability.items() if key != "pass"
        )
        repair_trigger = bool(
            asset_false
            and positive_before
            and all(value is True for value in positive_before.values())
            and row.get("immutability", {}).get("pass") is False
            and row.get("observability_pass") is False
            and row.get("automated_pass") is False
        )
        observability_pass = bool(
            row.get("overlay_pass") is True
            and row.get("rerun_pass") is True
            and row.get("viewer_capture_tokens_pass") is True
            and row.get("launcher", {}).get("pass") is True
            and immutability["pass"]
        )
        automated_pass = bool(row.get("scientific_result_recorded") and observability_pass)
        row["immutability"] = immutability
        row["observability_pass"] = observability_pass
        row["automated_pass"] = automated_pass
        row["automated_verdict"] = (
            str(row.get("scientific_verdict")) + VERDICT_PENDING_SUFFIX
            if automated_pass
            else VERDICT_VISUAL
        )
        row["aggregation_polarity_repair"] = {
            "triggered": repair_trigger,
            "frozen_bug": "D351 asset_write=false was incorrectly included in all(values)",
            "positive_key": "asset_write_forbidden_and_absent=true",
            "scientific_logic_changed": False,
            "pass": repair_trigger,
        }
        row["execution_counters"] = _counter_report(
            _json(MEASUREMENT_PATH) if MEASUREMENT_PATH.is_file() else None
        )
        row["controlled_physics_steps"] = None
        row["controlled_physics_steps_candidate"] = 0
        row["controlled_steps_authority"] = "pending D354 fsync/reread attestation"
    return row


def _write_json_router(path: Path, payload: Any) -> None:
    row = _rewrite_d351_payload(path, payload)
    if path == RUNTIME_EXCEPTION_PATH and isinstance(row, dict):
        row["artifact"] = "D354_RUNTIME_EXCEPTION_STOP_V1"
        row["case"] = CASE
        row["d354_controlled_physics_steps"] = None
        row["d354_q5_evaluator_invocations"] = _Q5_INVOCATIONS
        row["d354_q5_cache_miss_state_writes"] = _Q5_CACHE_MISS_SUCCESSES
        row["g0a_pass"] = False
    elif path == RAW_SUPERVISOR_PATH and isinstance(row, dict):
        row["artifact"] = "D354_INHERITED_D352_SUPERVISOR_RAW_V1"
        row["case"] = CASE
        row["classifier_non_authoritative_for_d354_science"] = True
    _write_json_exact(path, row)


def _write_text_router(path: Path, value: str) -> None:
    if path == AUTOMATED_MD_PATH and AUTOMATED_PATH.is_file():
        automated = _json(AUTOMATED_PATH)
        measurement = _json(MEASUREMENT_PATH)
        classification = automated["classification"]
        value = "\n".join(
            [
                "# D354 automated result",
                "",
                f"- scientific verdict: `{automated['scientific_verdict']}`",
                f"- automated pass: `{automated['automated_pass']}`",
                f"- executed zero-step q5 samples: `{measurement['execution_count']}`",
                f"- raw first-contact bracket: `{automated['raw_contact_bracket']}`",
                f"- live first-contact bracket: `{automated['live_contact_bracket']}`",
                f"- raw/live features: `{classification['raw_first_contact_feature']}` / `{classification['live_first_contact_feature']}`",
                f"- minimum table clearance: `{classification['min_gripper_table_clearance_mm']}` mm",
                "- controlled physics steps: `pending separate attestation (candidate 0)`",
                "- target/IK/path change: `false`",
                "- g0a_pass: `false`",
            ]
        ) + "\n"
    else:
        value = value.replace("D351", "D354")
    _write_text_exact(path, value)


def _counter_report(measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "evaluator_invocations_including_cache_hits": _Q5_INVOCATIONS,
        "evaluator_cache_miss_state_write_attempts": _Q5_CACHE_MISSES,
        "evaluator_cache_miss_state_write_successes": _Q5_CACHE_MISS_SUCCESSES,
        "primary_unique_rows": (
            measurement.get("execution_count") if isinstance(measurement, dict) else None
        ),
        "repeat_unique_rows": len(_REPEAT_CACHE_IDS) * 2 if _REPEAT_CACHE_IDS else 0,
        "distinct_repeat_cache_count": len(_REPEAT_CACHE_IDS),
        "auxiliary_state_only_write_attempts": _AUX_STATE_ATTEMPTS,
        "auxiliary_state_only_write_successes": _AUX_STATE_SUCCESSES,
        "viewer_ui_updates": _PUMP_UPDATES,
        "viewer_ui_guard_failures": _PUMP_GUARD_FAILURES,
    }


class _TimelineProxy:
    def __init__(self, raw: Any, inner: Any):
        self._raw = raw
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def pause(self) -> None:
        global _BRIDGE_EVENT, _MATCHED_D352
        if not self._raw.is_playing():
            return None
        if _BRIDGE_EVENT is not None:
            raise RuntimeError("D354 unexpected second PLAY/pause boundary; no extra commit allowed")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("D354 timeline PAUSE bridge is not on MainThread")
        _set_phase("timeline_pause_pending_state_commit_bridge")
        d353._TIMELINE = self._raw
        d353._TIMELINE_EVENT_ROWS.clear()
        d353._TIMELINE_EVENT_TYPE_NAMES.clear()
        import omni.timeline

        event_names = (
            "PLAY",
            "PAUSE",
            "STOP",
            "CURRENT_TIME_CHANGED",
            "CURRENT_TIME_TICKED_PERMANENT",
            "CURRENT_TIME_TICKED",
            "LOOP_MODE_CHANGED",
            "START_TIME_CHANGED",
            "END_TIME_CHANGED",
            "TIME_CODE_PER_SECOND_CHANGED",
            "AUTO_UPDATE_CHANGED",
            "PREROLLING_CHANGED",
            "TENTATIVE_TIME_CHANGED",
            "TICKS_PER_FRAME_CHANGED",
            "FAST_MODE_CHANGED",
            "PLAY_EVERY_FRAME_CHANGED",
            "TARGET_FRAMERATE_CHANGED",
            "DIRECTOR_CHANGED",
            "ZOOM_CHANGED",
        )
        for name in event_names:
            value = getattr(omni.timeline.TimelineEventType, name, None)
            if value is not None:
                d353._TIMELINE_EVENT_TYPE_NAMES[int(value)] = name
        if d353._TIMELINE_EVENT_TYPE_NAMES.get(1) != "PAUSE":
            raise RuntimeError("D354 runtime PAUSE enum drift")
        d353._TIMELINE_SUBSCRIPTION = (
            self._raw.get_timeline_event_stream().create_subscription_to_pop(
                d353._timeline_event_callback,
                name="D354 commit-only timeline observer",
            )
        )
        before = d353._detailed_snapshot(self._inner, "d354_after_reset_before_initial_pause")
        d353._assert_boundary_preconditions(before, "d354_initial_pause:before")
        callback_start = len(d353._TIMELINE_EVENT_ROWS)
        result = self._raw.pause()
        pending = d353._detailed_snapshot(
            self._inner, "d354_after_reset_pause_before_commit"
        )
        _MATCHED_D352 = d353._matched_d352_initial_snapshot_contract(pending)
        _marker(
            "worker",
            "matched_d352_initial_pending_snapshot",
            "end",
            {"pass": _MATCHED_D352["pass"]},
        )
        if not _MATCHED_D352["pass"]:
            raise RuntimeError("D354 D352/D353 reset pending-state baseline drift")
        inherited = {
            "phase": "after_reset_initial_pause",
            "playing_before": before["timeline_playing"],
            "playing_pre_commit": pending["timeline_playing"],
            "pause_interventions": 1,
            "inherited_setting_writes": 1,
        }
        _BRIDGE_EVENT = d353._commit_pending_pause(
            self._inner,
            "after_reset_initial_pause",
            before,
            pending,
            1,
            callback_start,
            inherited,
        )
        if not _BRIDGE_EVENT["pass"] or d353._TIMELINE.is_playing() is not False:
            raise RuntimeError("D354 fresh D353 commit bridge failed")
        _marker(
            "worker",
            "timeline_commit_bridge",
            "complete",
            {
                "pass": _BRIDGE_EVENT["pass"],
                "commit_count": d353._COMMIT_CALL_COUNT,
            },
        )
        return result


def _capture_inner_sentinels(inner: Any, phase: str) -> dict[str, Any]:
    timeline = _TIMELINE_PROXY
    joints = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    obj_pos, obj_quat = d351.d334._object_pose_w(inner)
    return {
        "phase": phase,
        "custom_step_counter": int(inner._sim_step_counter),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time": float(timeline.get_current_time()),
        "simulation_context_clock": d351._simulation_clock(inner),
        "joint_float32_bits": joints.tobytes().hex(),
        "object_position_float32_bits": np.asarray(obj_pos, dtype=np.float32).tobytes().hex(),
        "object_quaternion_float32_bits": np.asarray(obj_quat, dtype=np.float32).tobytes().hex(),
    }


def _wrapped_make_runtime_env(args: argparse.Namespace) -> Any:
    global _ACTIVE_INNER, _RAW_TIMELINE, _TIMELINE_PROXY, _ORIGINAL_TIMELINE_GETTER
    import omni.timeline

    _set_phase("make_runtime_env")
    _marker("worker", "_make_runtime_env", "start")
    inner = _ORIGINAL_MAKE_RUNTIME_ENV(args)
    _ACTIVE_INNER = inner
    _ORIGINAL_TIMELINE_GETTER = omni.timeline.get_timeline_interface
    _RAW_TIMELINE = _ORIGINAL_TIMELINE_GETTER()
    _TIMELINE_PROXY = _TimelineProxy(_RAW_TIMELINE, inner)
    original_reset = inner.reset
    original_close = inner.close

    def reset_wrapper(*reset_args: Any, **reset_kwargs: Any) -> Any:
        _set_phase("reset")
        _marker("worker", "reset", "start")
        omni.timeline.get_timeline_interface = _ORIGINAL_TIMELINE_GETTER
        try:
            return original_reset(*reset_args, **reset_kwargs)
        finally:
            omni.timeline.get_timeline_interface = lambda: _TIMELINE_PROXY
            _marker(
                "worker",
                "reset",
                "return_before_pause_request",
                {
                    "counter": int(inner._sim_step_counter),
                    "timeline_playing": bool(_RAW_TIMELINE.is_playing()),
                },
            )

    def close_wrapper() -> Any:
        global _ACTIVE_INNER, _LAST_INNER_SENTINELS
        if _ACTIVE_INNER is None:
            return None
        _LAST_INNER_SENTINELS = _capture_inner_sentinels(inner, "before_inner_close")
        _marker("worker", "inner.close", "start", _LAST_INNER_SENTINELS)
        try:
            return original_close()
        finally:
            _ACTIVE_INNER = None
            _marker("worker", "inner.close", "end")

    inner.reset = reset_wrapper
    inner.close = close_wrapper
    omni.timeline.get_timeline_interface = lambda: _TIMELINE_PROXY
    _marker("worker", "_make_runtime_env", "end")
    return inner


def _wrapped_corrected_audit() -> dict[str, Any]:
    global _CORRECTED_REPORT
    _set_phase("corrected_audit")
    _marker("worker", "corrected_audit", "start")
    _CORRECTED_REPORT = _ORIGINAL_CORRECTED_AUDIT()
    _marker("worker", "corrected_audit", "end", {"pass": _CORRECTED_REPORT.get("pass")})
    return _CORRECTED_REPORT


def _wrapped_build_live(inner: Any) -> tuple[Any, Any]:
    global _LIVE_REPORT, _LIVE_TRACE, _POST_LIVE_SNAPSHOT
    _set_phase("live_callback_topology_binding_parts_0_127")
    # D352's line-map verifier intentionally resolves the live builder through
    # the module global.  Expose the frozen original only for that traced call;
    # otherwise it would inspect this D354 wrapper and fail before part 0.
    installed = d351.d349._build_live_topology_parts
    d351.d349._build_live_topology_parts = _ORIGINAL_BUILD_LIVE
    try:
        parts, report, trace = d352._build_live_with_markers(inner)
    finally:
        d351.d349._build_live_topology_parts = installed
    _LIVE_REPORT = report
    _LIVE_TRACE = trace
    _POST_LIVE_SNAPSHOT = d353._detailed_snapshot(inner, "d354_after_live_binding")
    return parts, report


def _wrapped_build_raw(inner: Any, summary: dict[str, Any]) -> tuple[Any, Any]:
    global _RAW_REPORT, _POST_RAW_SNAPSHOT
    _set_phase("raw_authored_mesh_binding")
    _marker("worker", "raw_binding", "start")
    shapes, report = _ORIGINAL_BUILD_RAW(inner, summary)
    _RAW_REPORT = report
    _POST_RAW_SNAPSHOT = d353._detailed_snapshot(inner, "d354_after_raw_binding")
    _write_json_exact(RAW_CONTRACT_PATH, {**copy.deepcopy(report), "artifact": "D354_RAW_SOURCE_CONTRACT_V1", "case": CASE})
    _marker("worker", "raw_binding", "end", {"pass": report.get("pass")})
    return shapes, report


def _arm_science() -> None:
    global _SCIENCE_ARMED, _Q5_GATE_ATTEMPTS
    _Q5_GATE_ATTEMPTS += 1
    _set_phase("zero_step_bridge_science_arm")
    _marker("worker", "zero_step_bridge", "arm_attempt")
    if any(
        value is None
        for value in (
            _BRIDGE_EVENT,
            _MATCHED_D352,
            _POST_LIVE_SNAPSHOT,
            _POST_RAW_SNAPSHOT,
            _CORRECTED_REPORT,
            _LIVE_REPORT,
            _LIVE_TRACE,
            _RAW_REPORT,
        )
    ):
        raise RuntimeError("D354 science arm requested before every bridge/binding boundary")
    mutation_rows = [
        _BRIDGE_EVENT["post_commit"],
        _POST_LIVE_SNAPSHOT,
        _POST_RAW_SNAPSHOT,
    ]
    mutation = d353._mutation_checks(mutation_rows)
    prereg = _json(PREREG_PATH)
    checks = {
        "fresh_commit_event_pass": _BRIDGE_EVENT["pass"],
        "matched_d352_d353_pending_baseline": _MATCHED_D352["pass"],
        "exact_one_commit_attempt_and_call": d353._COMMIT_ATTEMPT_COUNT
        == d353._COMMIT_CALL_COUNT
        == 1,
        "single_pause_callback": len(d353._TIMELINE_EVENT_ROWS) == 1
        and d353._TIMELINE_EVENT_ROWS[0].get("event_name") == "PAUSE",
        "post_commit_live_raw_mutation_sentinels_exact": all(mutation.values()),
        "post_live_raw_pause_not_stop": all(
            row["timeline_playing"] is False and row["timeline_stopped"] is False
            for row in (_POST_LIVE_SNAPSHOT, _POST_RAW_SNAPSHOT)
        ),
        "corrected_128_of_128": _CORRECTED_REPORT.get("pass") is True,
        "live_binding_64_plus_64": _LIVE_REPORT.get("pass") is True,
        "live_trace_128_start_end": _LIVE_TRACE.get("pass") is True,
        "raw_source_contract": _RAW_REPORT.get("pass") is True,
        "live_binding_persisted": LIVE_BINDING_PATH.is_file(),
        "raw_contract_persisted": RAW_CONTRACT_PATH.is_file(),
        "launcher_gui_contract": _LAUNCHER_REPORT is not None
        and _LAUNCHER_REPORT.get("pass") is True,
        "input_hashes_exact": d351._input_hashes()
        == _INPUT_BEFORE
        == d351.EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "source_inventory_exact": d351.d349._source_inventories()
        == _SOURCE_BEFORE
        == prereg.get("source_inventories"),
        "prior_d351_d353_immutable": _immutable_prior_contract()["pass"],
        "user_sidecar_read_only": _user_sidecar_contract()
        == prereg.get("preexisting_user_files"),
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d354": _status_scope_pass(d351._git_status()),
        "both_q5_entry_aliases_same_gate": d351._evaluate_q5 is _gated_evaluate_q5
        and d352.d351a2._ORIGINAL_EVALUATE_Q5 is _gated_evaluate_q5,
        "q5_counts_zero_before_arm": _Q5_INVOCATIONS == _Q5_CACHE_MISSES == 0,
    }
    bridge = {
        "artifact": "D354_ZERO_STEP_SCIENCE_ARM_BRIDGE_V1",
        "case": CASE,
        "inherited_d353_attestation": prereg.get("control_authority"),
        "fresh_commit_event": _BRIDGE_EVENT,
        "matched_d352_initial_pending_snapshot": _MATCHED_D352,
        "post_live_snapshot": _POST_LIVE_SNAPSHOT,
        "post_raw_snapshot": _POST_RAW_SNAPSHOT,
        "mutation_checks": mutation,
        "live_trace": _LIVE_TRACE,
        "checks": checks,
        "science_armed": all(checks.values()),
        "controlled_physics_steps": None,
        "controlled_physics_steps_candidate": 0,
        "g0a_pass": False,
        "pass": all(checks.values()),
    }
    _write_json_exact(BRIDGE_PATH, bridge)
    if _json(BRIDGE_PATH) != bridge:
        raise RuntimeError("D354 bridge durable reread mismatch")
    if not bridge["pass"]:
        raise RuntimeError(f"D354 science arm bridge STOP: {checks}")
    _SCIENCE_ARMED = True
    _marker(
        "worker",
        "science_arm",
        "false_to_true",
        {"bridge_sha256": _sha(BRIDGE_PATH), "q5_count_before_arm": 0},
    )
    d353._TIMELINE_SUBSCRIPTION = None
    d353._TIMELINE = None


def _gated_evaluate_q5(*args: Any, **kwargs: Any) -> Any:
    global _Q5_INVOCATIONS, _Q5_CACHE_MISSES, _Q5_CACHE_MISS_SUCCESSES, _PRIMARY_CACHE_ID
    if not _SCIENCE_ARMED:
        _arm_science()
    if not _SCIENCE_ARMED:
        raise RuntimeError("D354 q5 evaluator gate remained closed")
    cache = args[6] if len(args) > 6 else kwargs["cache"]
    q5 = float(np.float32(args[2] if len(args) > 2 else kwargs["q5"]))
    key = d351._q5_key(q5)
    cache_id = id(cache)
    if _PRIMARY_CACHE_ID is None:
        _PRIMARY_CACHE_ID = cache_id
    elif cache_id != _PRIMARY_CACHE_ID:
        _REPEAT_CACHE_IDS.add(cache_id)
    cache_miss = key not in cache
    _Q5_INVOCATIONS += 1
    if cache_miss:
        _Q5_CACHE_MISSES += 1
    _set_phase("q5_closure_science_evaluation")
    _marker(
        "worker",
        "q5_evaluator",
        "start",
        {
            "q5_float32_rad": q5,
            "cache_miss": cache_miss,
            "cache_role": "primary" if cache_id == _PRIMARY_CACHE_ID else "repeat",
        },
    )
    result = _ORIGINAL_D351_EVALUATE(*args, **kwargs)
    if cache_miss:
        _Q5_CACHE_MISS_SUCCESSES += 1
    _marker(
        "worker",
        "q5_evaluator",
        "end",
        {"q5_float32_rad": q5, "cache_miss": cache_miss},
    )
    return result


def _wrapped_set_state_only(*args: Any, **kwargs: Any) -> Any:
    global _AUX_STATE_ATTEMPTS, _AUX_STATE_SUCCESSES
    if not _SCIENCE_ARMED:
        raise RuntimeError("D354 auxiliary state write attempted before science arm")
    _AUX_STATE_ATTEMPTS += 1
    _set_phase("auxiliary_zero_step_state_write")
    result = _ORIGINAL_D351_SET_STATE(*args, **kwargs)
    _AUX_STATE_SUCCESSES += 1
    _marker(
        "worker",
        "auxiliary_state_only_write",
        "end",
        {"attempt": _AUX_STATE_ATTEMPTS, "pass": result.get("pass")},
    )
    return result


def _wrapped_pump_frames(simulation_app: Any, inner: Any, timeline: Any, count: int) -> int:
    global _PUMP_UPDATES, _PUMP_GUARD_FAILURES
    for _ in range(count):
        before = _capture_inner_sentinels(inner, "viewer_pump_before")
        if before["timeline_playing"] or before["timeline_stopped"]:
            _PUMP_GUARD_FAILURES += 1
            raise RuntimeError("D354 Viewer pump precondition is not PAUSE/not-STOP")
        simulation_app.update()
        after = _capture_inner_sentinels(inner, "viewer_pump_after")
        invariant_keys = (
            "custom_step_counter",
            "timeline_time",
            "simulation_context_clock",
            "joint_float32_bits",
            "object_position_float32_bits",
            "object_quaternion_float32_bits",
        )
        exact = all(before[key] == after[key] for key in invariant_keys)
        paused = not after["timeline_playing"] and not after["timeline_stopped"]
        if not exact or not paused:
            _PUMP_GUARD_FAILURES += 1
            raise RuntimeError(
                f"D354 Viewer update advanced zero-step sentinel: exact={exact}, paused={paused}"
            )
        _PUMP_UPDATES += 1
        if _PUMP_UPDATES == 1 or _PUMP_UPDATES % 60 == 0:
            _marker(
                "worker",
                "viewer_ui_pump",
                "heartbeat",
                {"update_count": _PUMP_UPDATES, "zero_step_guard": True},
            )
    return 0


def _wrapped_rerun(*args: Any, **kwargs: Any) -> Any:
    _set_phase("rerun_rrd_rbl_capture")
    _marker("worker", "rerun", "start")
    result = _ORIGINAL_D351_RERUN(*args, **kwargs)
    _marker("worker", "rerun", "end", {"pass": result.get("pass")})
    return result


def _wrapped_viewer(*args: Any, **kwargs: Any) -> Any:
    _set_phase("isaac_viewer_capture_and_hold")
    _marker("worker", "viewer", "start")
    result = _ORIGINAL_D351_VIEWER(*args, **kwargs)
    _marker("worker", "viewer", "end", {"pass": result.get("pass")})
    return result


def _configure_d351_runtime_paths() -> None:
    replacements = {
        "CASE": CASE,
        "OUT_DIR": OUT_DIR,
        "PREREG_PATH": PREREG_PATH,
        "PARAMETER_PATH": PARAMETER_PATH,
        "PREFLIGHT_PATH": WORKER_PREFLIGHT_PATH,
        "LIVE_BINDING_PATH": LIVE_BINDING_PATH,
        "MOVING_BINDING_PATH": MOVING_BINDING_PATH,
        "MEASUREMENT_PATH": MEASUREMENT_PATH,
        "SWEEP_CSV_PATH": SWEEP_CSV_PATH,
        "OVERLAY_PATH": OVERLAY_PATH,
        "CAPTURE_PATH": CAPTURE_PATH,
        "RRD_PATH": RRD_PATH,
        "RBL_PATH": RBL_PATH,
        "RERUN_PNG_PATH": RERUN_PNG_PATH,
        "RERUN_VALIDATION_PATH": RERUN_VALIDATION_PATH,
        "AUTOMATED_PATH": AUTOMATED_PATH,
        "AUTOMATED_MD_PATH": AUTOMATED_MD_PATH,
        "RUNTIME_EXCEPTION_PATH": RUNTIME_EXCEPTION_PATH,
        "MANUAL_PATH": MANUAL_PATH,
        "MANUAL_MD_PATH": MANUAL_MD_PATH,
        "COMPLETION_PATH": COMPLETION_PATH,
        "COMPLETION_MD_PATH": COMPLETION_MD_PATH,
        "VIEWER_PNGS": VIEWER_PNGS,
        "SESSION_DOC": SESSION_DOC,
        "START_HERE": START_HERE,
        "HARNESS": HARNESS,
        "EXPECTED_HEAD": EXPECTED_HEAD,
        "GUIDE_ROOT": "/World/D354ViewerGuides",
        "VERDICT_INPUT": VERDICT_INPUT,
        "VERDICT_ORDER": VERDICT_ORDER,
        "VERDICT_BINDING": VERDICT_BINDING,
        "VERDICT_ELIGIBLE": VERDICT_ELIGIBLE,
        "VERDICT_REPAIR": VERDICT_REPAIR,
        "VERDICT_VISUAL": VERDICT_VISUAL,
        "VERDICT_PENDING_SUFFIX": VERDICT_PENDING_SUFFIX,
    }
    for name, value in replacements.items():
        setattr(d351, name, value)
    d351._write_json = _write_json_router
    d351._write_text = _write_text_router
    d351._status_scope_pass = _status_scope_pass
    d351._preexisting_user_untracked_contract = _user_sidecar_contract


def _configure_d352_supervisor() -> None:
    replacements = {
        "CASE": CASE,
        "CASE_NAME": CASE_NAME,
        "OUT_DIR": OUT_DIR,
        "HARNESS": HARNESS,
        "SESSION_DOC": SESSION_DOC,
        "START_HERE": START_HERE,
        "EXPECTED_HEAD": EXPECTED_HEAD,
        "REGISTERED_PYTHON": REGISTERED_PYTHON,
        "SEED": SEED,
        "INACTIVITY_WATCHDOG_S": INACTIVITY_WATCHDOG_S,
        "TOTAL_WATCHDOG_S": TOTAL_WATCHDOG_S,
        "FAULT_DUMP_GRACE_S": FAULT_DUMP_GRACE_S,
        "TERM_GRACE_S": TERM_GRACE_S,
        "GPU_SAMPLE_PERIOD_S": GPU_SAMPLE_PERIOD_S,
        "SUPERVISOR_PID_ENV": SUPERVISOR_PID_ENV,
        "WORKER_LAUNCH_TOKEN_ENV": WORKER_LAUNCH_TOKEN_ENV,
        "PARAMETER_PATH": PARAMETER_PATH,
        "GPU_HARDWARE_PATH": GPU_HARDWARE_PATH,
        "PREREG_PATH": PREREG_PATH,
        "SUPERVISOR_PREFLIGHT_PATH": SUPERVISOR_PREFLIGHT_PATH,
        "WORKER_PREFLIGHT_PATH": WORKER_PREFLIGHT_PATH,
        "MARKER_PATH": MARKER_PATH,
        "WORKER_LOG_PATH": WORKER_LOG_PATH,
        "GPU_TELEMETRY_PATH": GPU_TELEMETRY_PATH,
        "FAULT_PATH": FAULT_PATH,
        "LIVE_BINDING_PATH": LIVE_BINDING_PATH,
        "RAW_CONTRACT_PATH": RAW_CONTRACT_PATH,
        "BRIDGE_PATH": BRIDGE_PATH,
        "LOCALIZATION_PATH": SCIENCE_SUMMARY_PATH,
        "RUNTIME_EXCEPTION_PATH": RUNTIME_EXCEPTION_PATH,
        "WATCHDOG_PROC_PATH": WATCHDOG_PROC_PATH,
        "SUPERVISOR_AUDIT_PATH": RAW_SUPERVISOR_PATH,
    }
    for name, value in replacements.items():
        setattr(d352, name, value)
    d352._write_json = _write_json_router
    d352._marker = _marker
    d352._gpu_hardware_contract = _gpu_hardware_contract
    d352._sample_gpu_cpu = _sample_gpu_cpu
    d352._user_sidecar_contract = _user_sidecar_contract
    d352._immutable_d351_contract = _immutable_prior_contract
    d352._supervisor_preflight = _supervisor_preflight
    d353._marker = _marker


def _run_science(
    args: argparse.Namespace, simulation_app: Any, launcher_report: dict[str, Any]
) -> int:
    global _LAUNCHER_REPORT, _SOURCE_BEFORE, _INPUT_BEFORE
    _LAUNCHER_REPORT = launcher_report
    _SOURCE_BEFORE = d351.d349._source_inventories()
    _INPUT_BEFORE = d351._input_hashes()
    _configure_d351_runtime_paths()
    d351.d333._make_runtime_env = _wrapped_make_runtime_env
    d351.d349._corrected_live_audit = _wrapped_corrected_audit
    d351.d349._build_live_topology_parts = _wrapped_build_live
    d351.d339._build_retained_raw_shapes = _wrapped_build_raw
    d351._evaluate_q5 = _gated_evaluate_q5
    d352.d351a2._ORIGINAL_EVALUATE_Q5 = _gated_evaluate_q5
    d351._set_state_only = _wrapped_set_state_only
    d351._pump_frames = _wrapped_pump_frames
    d351._run_rerun = _wrapped_rerun
    d351._run_viewer = _wrapped_viewer
    _set_phase("d351_frozen_science_contract")
    _marker("worker", "d351_frozen_science_contract", "start")
    base_result: int | None = None
    try:
        base_result = d351._run_validate(args, simulation_app, launcher_report)
    finally:
        if _ORIGINAL_TIMELINE_GETTER is not None:
            import omni.timeline

            omni.timeline.get_timeline_interface = _ORIGINAL_TIMELINE_GETTER
    _marker(
        "worker",
        "d351_frozen_science_contract",
        "end",
        {"base_result": base_result},
    )
    automated = _json(AUTOMATED_PATH)
    measurement = _json(MEASUREMENT_PATH)
    capture = _json(CAPTURE_PATH)
    rerun = _json(RERUN_VALIDATION_PATH)
    bridge = _json(BRIDGE_PATH)
    rows = measurement.get("executed_rows", [])
    zero_step_checks = {
        "bridge_pass": bridge.get("pass") is True,
        "science_armed_once": _SCIENCE_ARMED and _Q5_GATE_ATTEMPTS == 1,
        "exact_one_commit": d353._COMMIT_ATTEMPT_COUNT == d353._COMMIT_CALL_COUNT == 1,
        "every_measurement_state_guard_pass": bool(rows)
        and all(row.get("state_guard", {}).get("pass") is True for row in rows),
        "every_measurement_counter_zero": bool(rows)
        and all(row.get("simulation_counter") == 0 for row in rows),
        "final_inner_sentinels_available": _LAST_INNER_SENTINELS is not None,
        "final_counter_zero": _LAST_INNER_SENTINELS is not None
        and _LAST_INNER_SENTINELS.get("custom_step_counter") == 0,
        "final_timeline_pause_not_stop": _LAST_INNER_SENTINELS is not None
        and _LAST_INNER_SENTINELS.get("timeline_playing") is False
        and _LAST_INNER_SENTINELS.get("timeline_stopped") is False,
        "viewer_update_guards_all_pass": _PUMP_UPDATES > 0 and _PUMP_GUARD_FAILURES == 0,
        "capture_zero_step_guards_pass": capture.get("checks", {}).get(
            "counter_zero_unchanged"
        )
        is True
        and capture.get("checks", {}).get("timeline_time_unchanged") is True
        and capture.get("checks", {}).get("simulation_context_clock_unchanged") is True,
    }
    immutable_after = _immutable_prior_contract()
    sidecar_after = _user_sidecar_contract()
    completion_checks = {
        "base_return_is_expected_d351_polarity_exit_2": base_result == 2,
        "known_polarity_repair_triggered": automated.get(
            "aggregation_polarity_repair", {}
        ).get("pass")
        is True,
        "automated_science_and_observability_pass": automated.get("automated_pass")
        is True,
        "scientific_result_recorded": automated.get("scientific_result_recorded") is True,
        "measurement_nonempty": measurement.get("execution_count", 0) > 0,
        "rerun_machine_contract_pass": rerun.get("pass") is True,
        "viewer_capture_contract_pass": capture.get("pass") is True,
        "zero_step_controls_exact": all(zero_step_checks.values()),
        "input_hashes_unchanged": d351._input_hashes()
        == _INPUT_BEFORE
        == d351.EXPECTED_INPUT_HASHES,
        "source_inventory_unchanged": d351.d349._source_inventories() == _SOURCE_BEFORE,
        "prior_d351_d353_immutable": immutable_after["pass"],
        "sidecar_read_only": sidecar_after["pass"],
        "git_scope_only_d354": _status_scope_pass(d351._git_status()),
    }
    summary = {
        "artifact": "D354_SCIENCE_RESUME_SUMMARY_V1",
        "case": CASE,
        "operational_verdict": (
            "D354_AUTOMATED_SCIENCE_COMPLETE_PENDING_ATTESTATION"
            if all(completion_checks.values())
            else "D354_SCIENCE_OR_OBSERVABILITY_CONTRACT_FAIL_STOP"
        ),
        "scientific_verdict": automated.get("scientific_verdict"),
        "automated_verdict": automated.get("automated_verdict"),
        "d354_q5_evaluator_invocations": _Q5_INVOCATIONS,
        "d354_q5_cache_miss_state_writes": _Q5_CACHE_MISS_SUCCESSES,
        "d352_q5_evaluation_count": _Q5_INVOCATIONS,
        "d352_controlled_physics_steps": None,
        "execution_counters": _counter_report(measurement),
        "zero_step_checks": zero_step_checks,
        "completion_checks": completion_checks,
        "final_inner_sentinels": _LAST_INNER_SENTINELS,
        "bridge_path": _rel(BRIDGE_PATH),
        "bridge_sha256": _sha(BRIDGE_PATH),
        "measurement_path": _rel(MEASUREMENT_PATH),
        "measurement_sha256": _sha(MEASUREMENT_PATH),
        "automated_path": _rel(AUTOMATED_PATH),
        "automated_sha256": _sha(AUTOMATED_PATH),
        "controlled_physics_steps": None,
        "controlled_physics_steps_candidate": 0
        if all(zero_step_checks.values())
        else None,
        "target_ik_path_changed": False,
        "physics_configuration_changed": False,
        "automatic_retry": False,
        "g0a_pass": False,
        "pass": all(completion_checks.values()),
    }
    _write_json_exact(SCIENCE_SUMMARY_PATH, summary)
    if _json(SCIENCE_SUMMARY_PATH) != summary:
        raise RuntimeError("D354 science summary durable reread mismatch")
    controlled_steps = 0 if all(zero_step_checks.values()) else None
    attestation = {
        "artifact": "D354_ZERO_STEP_SCIENCE_ATTESTATION_V1",
        "case": CASE,
        "summary_path": _rel(SCIENCE_SUMMARY_PATH),
        "summary_sha256": _sha(SCIENCE_SUMMARY_PATH),
        "summary_fsync_and_reread_exact": True,
        "operational_verdict": (
            "D354_ZERO_STEP_Q5_SCIENCE_ATTESTED_PENDING_MANUAL"
            if summary["pass"] and controlled_steps == 0
            else "D354_ZERO_STEP_SCIENCE_ATTESTATION_FAIL_STOP"
        ),
        "scientific_verdict": summary["scientific_verdict"],
        "d354_q5_evaluator_invocations": _Q5_INVOCATIONS,
        "d354_q5_cache_miss_state_writes": _Q5_CACHE_MISS_SUCCESSES,
        "d354_controlled_physics_steps": controlled_steps,
        "inherited_d353_controlled_physics_steps": 0,
        "inherited_d353_science_evaluation_count": 0,
        "target_ik_path_changed": False,
        "g0a_pass": False,
        "pass": bool(summary["pass"] and controlled_steps == 0),
    }
    _write_json_exact(ATTESTATION_PATH, attestation)
    if _json(ATTESTATION_PATH) != attestation:
        raise RuntimeError("D354 attestation durable reread mismatch")
    d352._BRIDGE_COMPLETE = bool(attestation["pass"])
    _marker(
        "worker",
        "science_attestation",
        "complete",
        {
            "pass": attestation["pass"],
            "controlled_physics_steps": controlled_steps,
            "scientific_verdict": summary["scientific_verdict"],
        },
    )
    return 0 if attestation["pass"] else 2


def _run_worker(args: argparse.Namespace) -> int:
    global _RUN_NONCE
    _RUN_NONCE = str(_json(PREREG_PATH)["run_nonce"])
    simulation_app = None
    fault_stream = FAULT_PATH.open("xb")
    faulthandler.register(signal.SIGUSR1, file=fault_stream, all_threads=True)
    _marker("worker", "worker_boot", "start", {"parent_pid": os.getppid()})
    try:
        preflight = _worker_preflight(args)
        _write_json_exact(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker", "worker_preflight", "end", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D354 worker preflight STOP: {preflight['checks']}")
        from isaaclab.app import AppLauncher

        _set_phase("AppLauncher")
        _marker("worker", "AppLauncher", "start")
        launcher = AppLauncher(copy.deepcopy(args))
        simulation_app = launcher.app
        launcher_report = d351.d350._resolved_gui_launcher(launcher)
        _marker("worker", "AppLauncher", "end", launcher_report)
        if not launcher_report["pass"]:
            raise RuntimeError(f"D354 GUI launcher contract STOP: {launcher_report}")
        return _run_science(args, simulation_app, launcher_report)
    except Exception as error:
        if not RUNTIME_EXCEPTION_PATH.exists():
            _write_json_exact(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D354_RUNTIME_EXCEPTION_STOP_V1",
                    "case": CASE,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "last_phase": _CURRENT_PHASE,
                    "d354_q5_evaluator_invocations": _Q5_INVOCATIONS,
                    "d354_q5_cache_miss_state_writes": _Q5_CACHE_MISS_SUCCESSES,
                    "d354_controlled_physics_steps": None,
                    "scientific_verdict": (
                        _json(AUTOMATED_PATH).get("scientific_verdict")
                        if AUTOMATED_PATH.is_file()
                        else None
                    ),
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        if _ACTIVE_INNER is not None:
            try:
                _ACTIVE_INNER.close()
            except Exception:
                pass
        if simulation_app is not None:
            _marker("worker", "SimulationApp.close", "start")
            try:
                simulation_app.close()
                _marker("worker", "SimulationApp.close", "end")
            except Exception as error:
                _marker(
                    "worker",
                    "SimulationApp.close",
                    "error",
                    {"error": f"{type(error).__name__}: {error}"},
                )
        try:
            faulthandler.unregister(signal.SIGUSR1)
            fault_stream.flush()
            os.fsync(fault_stream.fileno())
        finally:
            fault_stream.close()


def _run_supervisor(args: argparse.Namespace) -> int:
    global _RUN_NONCE
    _configure_d352_supervisor()
    raw_return = _ORIGINAL_D352_SUPERVISOR(args)
    prereg = _json(PREREG_PATH)
    _RUN_NONCE = str(prereg["run_nonce"])
    raw = _json(RAW_SUPERVISOR_PATH)
    summary = _json(SCIENCE_SUMMARY_PATH) if SCIENCE_SUMMARY_PATH.is_file() else None
    attestation = _json(ATTESTATION_PATH) if ATTESTATION_PATH.is_file() else None
    automated = _json(AUTOMATED_PATH) if AUTOMATED_PATH.is_file() else None
    markers, invalid_markers = d352._read_markers(_RUN_NONCE)
    worker = raw.get("worker", {})
    telemetry = raw.get("telemetry", {})
    inventory = sorted(path.name for path in OUT_DIR.iterdir() if path.is_file())
    expected = sorted(_runtime_expected_before_final_supervisor())
    checks = {
        "worker_exit_zero": worker.get("exit_code") == 0,
        "worker_reaped": worker.get("process_absent_after_reap") is True,
        "process_group_absent": worker.get("process_group_absent_after_cleanup") is True,
        "watchdog_not_triggered": raw.get("watchdog", {}).get("triggered") is False,
        "no_automatic_retry": raw.get("watchdog", {}).get("automatic_retry") is False,
        "markers_valid": bool(markers) and not invalid_markers,
        "runtime_exception_absent": not RUNTIME_EXCEPTION_PATH.exists(),
        "science_summary_pass": summary is not None and summary.get("pass") is True,
        "zero_step_attestation_pass": attestation is not None
        and attestation.get("pass") is True
        and attestation.get("d354_controlled_physics_steps") == 0,
        "automated_science_observability_pass": automated is not None
        and automated.get("automated_pass") is True,
        "telemetry_valid": telemetry.get("sample_count", 0) > 0
        and telemetry.get("valid_gpu_sample_count", 0) > 0
        and telemetry.get("invalid_gpu_sample_count") == 0
        and telemetry.get("uuid_mismatch_sample_count") == 0,
        "telemetry_thread_joined": raw.get("telemetry_thread_alive_after_join") is False,
        "prior_immutable": _immutable_prior_contract()["pass"],
        "sidecar_read_only": _user_sidecar_contract() == prereg.get("preexisting_user_files"),
        "harness_state_hashes_exact": _sha(HARNESS) == prereg.get("harness_sha256")
        and {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "input_source_exact": d351._input_hashes() == prereg.get("input_hashes")
        and d351.d349._source_inventories() == prereg.get("source_inventories"),
        "validate_inventory_exact_before_final_audit": inventory == expected,
    }
    verdict = (
        "D354_AUTOMATED_Q5_SCIENCE_AND_OBSERVABILITY_COMPLETE_PENDING_MANUAL"
        if all(checks.values())
        else "D354_VALIDATE_OR_SUPERVISOR_CONTRACT_FAIL_STOP"
    )
    audit = {
        "artifact": "D354_SUPERVISOR_AUDIT_V1",
        "case": CASE,
        "operational_verdict": verdict,
        "scientific_verdict": summary.get("scientific_verdict") if summary else None,
        "inherited_d352_supervisor_raw": {
            "path": _rel(RAW_SUPERVISOR_PATH),
            "sha256": _sha(RAW_SUPERVISOR_PATH),
            "return_code": raw_return,
            "raw_pass": raw.get("pass"),
            "non_authoritative_reason": (
                "D352's classifier intentionally rejects q5/Viewer outputs; D354 "
                "uses it only for process/watchdog/telemetry evidence."
            ),
        },
        "worker": worker,
        "watchdog": raw.get("watchdog"),
        "telemetry": telemetry,
        "markers": {
            "path": _rel(MARKER_PATH),
            "sha256": _sha(MARKER_PATH),
            "valid_count": len(markers),
            "invalid_rows": invalid_markers,
            "last_valid": markers[-1] if markers else None,
        },
        "science_summary": {
            "path": _rel(SCIENCE_SUMMARY_PATH) if summary else None,
            "sha256": _sha(SCIENCE_SUMMARY_PATH) if summary else None,
            "pass": summary.get("pass") if summary else None,
        },
        "zero_step_attestation": {
            "path": _rel(ATTESTATION_PATH) if attestation else None,
            "sha256": _sha(ATTESTATION_PATH) if attestation else None,
            "pass": attestation.get("pass") if attestation else None,
        },
        "output_inventory_before_final_audit": inventory,
        "expected_output_inventory_before_final_audit": expected,
        "checks": checks,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "g0a_pass": False,
        "pass": all(checks.values()),
    }
    _write_json_exact(SUPERVISOR_AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "stage": "validate",
                "operational_verdict": verdict,
                "scientific_verdict": audit["scientific_verdict"],
                "pass": audit["pass"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["pass"] else 2


def _manual_contract(manual: dict[str, Any]) -> dict[str, bool]:
    expected = {**VIEWER_PNGS, "rerun": RERUN_PNG_PATH}
    images = manual.get("images", {})
    checks: dict[str, bool] = {
        "artifact_exact": manual.get("artifact") == "D354_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "date_exact": manual.get("date") == "2026-07-16 KST",
        "method_original_resolution": manual.get("method")
        == "view_image original_resolution",
        "image_set_exact": set(images) == set(expected),
        "all_paths_exact": all(
            images.get(name, {}).get("path") == _rel(path) for name, path in expected.items()
        ),
        "all_image_hashes_exact": all(
            path.is_file() and images.get(name, {}).get("sha256") == _sha(path)
            for name, path in expected.items()
        ),
        "all_observations_nonempty": all(
            bool(images.get(name, {}).get("observation")) for name in expected
        ),
        "manual_pass_declared": manual.get("manual_pass") is True,
        "scientific_override_false": manual.get("scientific_override") is False,
        "bounded_interpretation_nonempty": bool(manual.get("bounded_interpretation")),
        "manual_markdown_exact": MANUAL_MD_PATH.is_file()
        and manual.get("manual_markdown", {}).get("path") == _rel(MANUAL_MD_PATH)
        and manual.get("manual_markdown", {}).get("sha256") == _sha(MANUAL_MD_PATH),
    }
    for name, path in expected.items():
        row = images.get(name, {})
        checks[f"{name}_bytes"] = path.is_file() and row.get("bytes") == path.stat().st_size
        checks[f"{name}_dimensions"] = row.get("raster_dimensions") == d351._png_dimensions(path)
    declared = manual.get("checks", {})
    for name in (
        "actual_isaac_open_and_decision_or_fallback_pose_visible",
        "actual_physx_colliders_visible",
        "colored_link5_64_and_gripper_64_distinguishable",
        "inner_patch_fixed_moving_chord_and_cylinder_feature_visible",
        "rerun_full_q5_dynamic_live_surface_patch_and_witness_timeline_visible",
        "no_obvious_empty_or_corrupt_panel",
    ):
        checks[name] = declared.get(name) is True
    return checks


def _run_finalize(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    supervisor = _json(SUPERVISOR_AUDIT_PATH)
    preflight = _json(WORKER_PREFLIGHT_PATH)
    automated = _json(AUTOMATED_PATH)
    summary = _json(SCIENCE_SUMMARY_PATH)
    attestation = _json(ATTESTATION_PATH)
    capture = _json(CAPTURE_PATH)
    manual = _json(MANUAL_PATH)
    manual_checks = _manual_contract(manual)
    worker_pid = supervisor.get("worker", {}).get("pid")
    process_absent_before = isinstance(worker_pid, int) and not psutil.pid_exists(worker_pid)
    pngs = {
        name: d351._stable_png(path, expected_size=d351.VIEWER_RASTER_SIZE)
        for name, path in VIEWER_PNGS.items()
    }
    pngs["rerun"] = d351._stable_png(
        RERUN_PNG_PATH, expected_size=d351.RERUN_RASTER_SIZE
    )
    process_absent_after = isinstance(worker_pid, int) and not psutil.pid_exists(worker_pid)
    viewer_hashes = [pngs[name]["samples"][-1]["sha256"] for name in VIEWER_PNGS]
    role = capture.get("display_pose_role")
    if role == "resolved_raw_last_clear_decision_pose":
        distinct = len(set(viewer_hashes)) == len(viewer_hashes) and None not in viewer_hashes
    else:
        by_name = {name: pngs[name]["samples"][-1]["sha256"] for name in VIEWER_PNGS}
        distinct = bool(
            role == "open_fallback_no_resolved_contact_bracket"
            and None not in by_name.values()
            and len(set(by_name.values())) >= 3
            and by_name["decision_colored"] != by_name["decision_side"]
            and by_name["open_physx"]
            not in {by_name["decision_colored"], by_name["decision_side"]}
            and by_name["decision_physx"]
            not in {by_name["decision_colored"], by_name["decision_side"]}
        )
    png_pass = bool(
        process_absent_before
        and process_absent_after
        and all(row["pass"] for row in pngs.values())
        and distinct
    )
    evidence_paths = {
        "preregistration": PREREG_PATH,
        "parameter_freeze": PARAMETER_PATH,
        "validate_preflight": WORKER_PREFLIGHT_PATH,
        "live_binding": LIVE_BINDING_PATH,
        "measurement": MEASUREMENT_PATH,
        "moving_binding": MOVING_BINDING_PATH,
        "sweep_csv": SWEEP_CSV_PATH,
        "overlay": OVERLAY_PATH,
        "capture": CAPTURE_PATH,
        "rerun_validation": RERUN_VALIDATION_PATH,
        "rrd": RRD_PATH,
        "rbl": RBL_PATH,
        "rerun_screenshot": RERUN_PNG_PATH,
    }
    observed = {name: _sha(path) for name, path in evidence_paths.items()}
    artifact_checks = {
        "supervisor_pass": supervisor.get("pass") is True,
        "automated_pass": automated.get("automated_pass") is True,
        "attestation_pass_steps_zero": attestation.get("pass") is True
        and attestation.get("d354_controlled_physics_steps") == 0,
        "postclose_png_pass": png_pass,
        "manual_visual_inspection_pass": all(manual_checks.values()),
        "rerun_validation_pass": _json(RERUN_VALIDATION_PATH).get("pass") is True,
        "evidence_hashes_exact": automated.get("evidence_hashes") == observed,
        "runtime_hash_chain_exact": (
            attestation.get("summary_sha256") == _sha(SCIENCE_SUMMARY_PATH)
            and summary.get("automated_sha256") == _sha(AUTOMATED_PATH)
            and summary.get("measurement_sha256") == _sha(MEASUREMENT_PATH)
            and summary.get("bridge_sha256") == _sha(BRIDGE_PATH)
            and supervisor.get("science_summary", {}).get("sha256")
            == _sha(SCIENCE_SUMMARY_PATH)
            and supervisor.get("zero_step_attestation", {}).get("sha256")
            == _sha(ATTESTATION_PATH)
        ),
        "preflight_prereg_parameter_harness_chain_exact": (
            preflight.get("preregistration_sha256") == _sha(PREREG_PATH)
            and preflight.get("parameter_audit_sha256") == _sha(PARAMETER_PATH)
            and preflight.get("harness_sha256") == _sha(HARNESS)
            and automated.get("harness_sha256") == _sha(HARNESS)
            and automated.get("state_hashes") == prereg.get("state_hashes")
        ),
        "environment_exact_preregistered": d351._environment_contract()
        == prereg.get("environment"),
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d354": _status_scope_pass(d351._git_status()),
        "harness_state_hashes_exact": _sha(HARNESS) == prereg.get("harness_sha256")
        and {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "input_source_exact": d351._input_hashes() == prereg.get("input_hashes")
        and d351.d349._source_inventories() == prereg.get("source_inventories"),
        "prior_immutable": _immutable_prior_contract()["pass"],
        "sidecar_read_only": _user_sidecar_contract() == prereg.get("preexisting_user_files"),
        "process_and_group_terminal": process_absent_after
        and supervisor.get("worker", {}).get("process_group_absent_after_cleanup") is True,
    }
    completion_pass = all(artifact_checks.values())
    final_verdict = automated.get("scientific_verdict") if completion_pass else VERDICT_VISUAL
    completion = {
        "artifact": "D354_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "completion_pass": completion_pass,
        "final_verdict": final_verdict,
        "scientific_verdict": automated.get("scientific_verdict"),
        "postclose_png_validation": {
            "worker_pid": worker_pid,
            "process_absent_before": process_absent_before,
            "process_absent_after": process_absent_after,
            "display_pose_role": role,
            "viewer_png_distinct_contract": distinct,
            "rows": pngs,
            "pass": png_pass,
        },
        "manual_visual_inspection": {
            "path": _rel(MANUAL_PATH),
            "sha256": _sha(MANUAL_PATH),
            "markdown_path": _rel(MANUAL_MD_PATH),
            "markdown_sha256": _sha(MANUAL_MD_PATH),
            "checks": manual_checks,
            "pass": all(manual_checks.values()),
        },
        "observed_evidence_hashes": observed,
        "artifact_binding_checks": artifact_checks,
        "controlled_physics_steps": attestation.get("d354_controlled_physics_steps"),
        "target_ik_path_changed": False,
        "physics_configuration_changed": False,
        "settle_authorized": False,
        "settle_executed": False,
        "ten_trial_run": False,
        "g0b_run": False,
        "rl_or_ppo_run": False,
        "vla_run": False,
        "ladder_promoted": False,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "g0a_pass": False,
    }
    _write_json_exact(COMPLETION_PATH, completion)
    classification = automated["classification"]
    _write_text_exact(
        COMPLETION_MD_PATH,
        "\n".join(
            [
                "# D354 completion",
                "",
                f"- completion pass: `{completion_pass}`",
                f"- final verdict: `{final_verdict}`",
                f"- raw first-contact: `{classification['raw_first_contact_feature']}`",
                f"- live first-contact: `{classification['live_first_contact_feature']}`",
                f"- minimum table clearance: `{classification['min_gripper_table_clearance_mm']}` mm",
                f"- controlled physics steps: `{attestation.get('d354_controlled_physics_steps')}`",
                "- target/IK/path changed: `false`",
                "- settle/trials/G0b/RL/VLA/G0a: `false`",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "stage": "finalize",
                "completion_pass": completion_pass,
                "final_verdict": final_verdict,
            },
            sort_keys=True,
        )
    )
    return 0 if completion_pass else 2


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "validate", "_worker", "finalize"), required=True
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inactivity_watchdog_s", type=float, default=INACTIVITY_WATCHDOG_S)
    parser.add_argument("--total_watchdog_s", type=float, default=TOTAL_WATCHDOG_S)
    parser.add_argument("--viewer_hold_seconds", type=float, default=VIEWER_HOLD_SECONDS)
    if stage == "_worker":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    _configure_d351_runtime_paths()
    _configure_d352_supervisor()
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--stage", choices=("prepare", "validate", "_worker", "finalize"), required=True
    )
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D354 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D354 seed drift")
    if float(args.inactivity_watchdog_s) != INACTIVITY_WATCHDOG_S:
        raise RuntimeError("D354 inactivity watchdog drift")
    if float(args.total_watchdog_s) != TOTAL_WATCHDOG_S:
        raise RuntimeError("D354 total watchdog drift")
    if float(args.viewer_hold_seconds) != VIEWER_HOLD_SECONDS:
        raise RuntimeError("D354 Viewer hold drift")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "validate":
        return _run_supervisor(args)
    if args.stage == "finalize":
        return _run_finalize(args)
    args.headless = False
    args.livestream = 0
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    if hasattr(args, "xr"):
        args.xr = False
    args.device = "cuda:0"
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
