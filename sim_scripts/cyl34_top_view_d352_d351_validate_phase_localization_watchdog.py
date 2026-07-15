#!/usr/bin/env python3
"""D352: localize the D351 attempt2 pre-science long run, once.

This forward-only operational harness reproduces the frozen D351 attempt2 path
only through its zero-step representation bridge.  Durable phase markers and
an external supervisor identify the last completed boundary.  The harness
cannot enter q5 science: the original evaluator is replaced by a fail-closed
trap and the worker returns immediately after the bridge contract.

No target/IK/path, asset, decomposition, gate, material, mass, actuator, or
physics-solver configuration is changed.  No controlled physics step, app
update, moving-surface measurement, q5 sample, Viewer, Rerun, settle, trial,
G0b, RL/PPO, VLA, or ladder action is performed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import faulthandler
import hashlib
import inspect
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
    cyl34_top_view_d351_attempt2_timeline_pause_repair as d351a2,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d351_zero_step_closure_geometry as d351,
)


CASE = "g0a_d352"
CASE_NAME = "d351_validate_phase_localization_watchdog"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d352"
HARNESS = Path(__file__).resolve()
SESSION_DOC = (
    REPO
    / "claudedocs/session_20260715_grasp_g0a_d352_d351_validate_phase_localization_watchdog.md"
)
START_HERE = REPO / "START_HERE.md"
EXPECTED_HEAD = "c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
SEED = 33201

INACTIVITY_WATCHDOG_S = 120.0
TOTAL_WATCHDOG_S = 300.0
FAULT_DUMP_GRACE_S = 5.0
TERM_GRACE_S = 30.0
GPU_SAMPLE_PERIOD_S = 1.0
SUPERVISOR_PID_ENV = "D352_SUPERVISOR_PID"
WORKER_LAUNCH_TOKEN_ENV = "D352_WORKER_LAUNCH_TOKEN"

NEW_OPERATIONAL_VARIABLES = [
    "durable_phase_marker_stream",
    "external_bounded_wall_clock_watchdog",
]
NEW_SCIENTIFIC_VARIABLES: list[str] = []
NEW_PHYSICAL_VARIABLES: list[str] = []

PARAMETER_PATH = OUT_DIR / "d352_parameter_freeze_audit.json"
GPU_HARDWARE_PATH = OUT_DIR / "d352_gpu_hardware_contract.json"
PREREG_PATH = OUT_DIR / "d352_preregistration.json"
SUPERVISOR_PREFLIGHT_PATH = OUT_DIR / "d352_supervisor_preflight.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d352_validate_preflight.json"
MARKER_PATH = OUT_DIR / "d352_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d352_worker_stdout_stderr.log"
GPU_TELEMETRY_PATH = OUT_DIR / "d352_gpu_cpu_telemetry.jsonl"
FAULT_PATH = OUT_DIR / "d352_faulthandler.txt"
LIVE_BINDING_PATH = OUT_DIR / "d352_live_topology_runtime_binding.json"
RAW_CONTRACT_PATH = OUT_DIR / "d352_raw_source_contract.json"
BRIDGE_PATH = OUT_DIR / "d352_zero_step_bridge_contract.json"
LOCALIZATION_PATH = OUT_DIR / "d352_localization_summary.json"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d352_runtime_exception.json"
WATCHDOG_PROC_PATH = OUT_DIR / "d352_watchdog_proc_snapshot.json"
SUPERVISOR_AUDIT_PATH = OUT_DIR / "d352_supervisor_audit.json"

D351_ATTEMPT1_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d351"
D351_ATTEMPT2_DIR = D351_ATTEMPT1_DIR / "attempt2_timeline_pause_repair"

IMMUTABLE_HASHES = {
    "sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py": (
        "3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d"
    ),
    "sim_scripts/cyl34_top_view_d351_attempt2_timeline_pause_repair.py": (
        "1e50334e22b780fa99b52b38d45abaef1f08ea6cf4ee6f613193f99e29df492f"
    ),
    "claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md": (
        "20367375e05ce8cffb47f86ff0c1645a3544f5bf62516fe2e16a98919c356a06"
    ),
    "claudedocs/session_20260715_grasp_g0a_d351_timeline_pause_repair.md": (
        "436192b381b563e5b3c73d6997f12ea48a9600afca530547c4ec598253708bc1"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_parameter_freeze_audit.json": (
        "98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_preregistration.json": (
        "d0639f51485b96395de88b0942ea4af13a768f31db89a400df7af97a25df1456"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_validate_preflight.json": (
        "3e3172ff595bdc48b4216ab0bbb30386a2fdf29f0786ab8d950881114d434660"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_live_topology_runtime_binding.json": (
        "9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_runtime_exception.json": (
        "138097cee4a471b84202572639fd19c0cba6103d5a628d89a2af49bcbde71914"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_parameter_freeze_audit.json": (
        "98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_preregistration.json": (
        "eb05905a683842693dd5a0f7dff717cdae9c8bc4d9d6c51a9e5e7b21eba64fc1"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_validate_preflight.json": (
        "035113da2ae94ec7d458d8f5e9a675bdac79f443fb3827f8555dfd4c37166334"
    ),
    "claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json": (
        "af17995b40d5818055388f97e38cbb50f0895f3a2aa4d2cb7f5cf1df3b6166fe"
    ),
}

ATTEMPT1_ROOT_FILES = sorted(
    [
        "d351_live_topology_runtime_binding.json",
        "d351_parameter_freeze_audit.json",
        "d351_preregistration.json",
        "d351_runtime_exception.json",
        "d351_validate_preflight.json",
    ]
)
ATTEMPT2_ROOT_FILES = sorted(
    [
        "d351_external_termination_audit.json",
        "d351_parameter_freeze_audit.json",
        "d351_preregistration.json",
        "d351_validate_preflight.json",
    ]
)

USER_SIDECAR_HASHES = {
    "claudedocs/lab_meeting/20260715/d334_collision_table/README.md": (
        "35e39f584737c888bcf7dfab6154c55c5d13d4154ee7f2042073e1c0a7e18783"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.html": (
        "6d38933f959eba916208ec04a329ba25e2bd753c90720576010c222a8bda679c"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.png": (
        "ddc9db2795f4d66b2564adf156829e6a143a599ceb72f6bb9fa28ab25e68a183"
    ),
}

_PROCESS_START_MONOTONIC_NS = time.monotonic_ns()
_RUN_NONCE = "not-loaded"
_MARKER_SEQUENCE = 0
_Q5_TRAP_INVOCATION_COUNT = 0
_BRIDGE_COMPLETE = False
_ACTIVE_INNER: Any | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, _payload_bytes(payload))


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, default=_json_default) + "\n").encode(
        "utf-8"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, raw)
        if written != len(raw):
            raise RuntimeError(f"short append: {written} != {len(raw)}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _marker(actor: str, phase: str, event: str, details: Any | None = None) -> dict[str, Any]:
    global _MARKER_SEQUENCE

    _MARKER_SEQUENCE += 1
    row = {
        "artifact": "D352_DURABLE_PHASE_MARKER_V1",
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
        "q5_trap_invocation_count": _Q5_TRAP_INVOCATION_COUNT,
        "details": details,
    }
    _append_jsonl(MARKER_PATH, row)
    return row


def _status_scope_pass(status: dict[str, str]) -> bool:
    exact = {_rel(START_HERE), _rel(SESSION_DOC), _rel(HARNESS)}
    prefix = _rel(OUT_DIR) + "/"
    return all(path in exact or path.startswith(prefix) for path in status)


def _immutable_d351_contract() -> dict[str, Any]:
    rows = {}
    for relative, expected in IMMUTABLE_HASHES.items():
        path = REPO / relative
        rows[relative] = {
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "expected_sha256": expected,
        }
    attempt1_files = sorted(
        path.name for path in D351_ATTEMPT1_DIR.iterdir() if path.is_file()
    )
    attempt2_files = sorted(
        path.name for path in D351_ATTEMPT2_DIR.iterdir() if path.is_file()
    )
    attempt2_forbidden = [
        D351_ATTEMPT2_DIR / "d351_live_topology_runtime_binding.json",
        D351_ATTEMPT2_DIR / "d351_timeline_pause_repair_contract.json",
        D351_ATTEMPT2_DIR / "d351_moving_jaw_surface_binding.json",
        D351_ATTEMPT2_DIR / "d351_zero_step_closure_geometry_measurement.json",
        D351_ATTEMPT2_DIR / "d351_q5_closure_sweep.csv",
        D351_ATTEMPT2_DIR / "d351_viewer_capture_contract.json",
        D351_ATTEMPT2_DIR / "d351_zero_step_closure_geometry.rrd",
        D351_ATTEMPT2_DIR / "d351_zero_step_closure_geometry.rbl",
    ]
    checks = {
        "all_hashes_exact": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
        "attempt1_root_inventory_exact": attempt1_files == ATTEMPT1_ROOT_FILES,
        "attempt2_root_inventory_exact": attempt2_files == ATTEMPT2_ROOT_FILES,
        "attempt2_science_bridge_viewer_outputs_absent": all(
            not path.exists() for path in attempt2_forbidden
        ),
    }
    return {
        "rows": rows,
        "attempt1_root_files": attempt1_files,
        "attempt2_root_files": attempt2_files,
        "attempt2_forbidden_outputs": [_rel(path) for path in attempt2_forbidden],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _user_sidecar_contract() -> dict[str, Any]:
    status = d351._git_status()
    rows = {}
    for relative, expected in USER_SIDECAR_HASHES.items():
        path = REPO / relative
        rows[relative] = {
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
            "expected_sha256": expected,
            "git_status": status.get(relative),
        }
    checks = {
        "all_exist": all(row["exists"] for row in rows.values()),
        "all_hashes_exact": all(
            row["sha256"] == row["expected_sha256"] for row in rows.values()
        ),
    }
    return {
        "role": "user-owned non-scientific sidecar; read-only",
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _gpu_hardware_contract() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    props = torch.cuda.get_device_properties(0) if available else None
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap,power.max_limit",
        "--format=csv,noheader,nounits",
    ]
    query_attempts = []
    values: list[str] = []
    query_row: dict[str, Any] = {
        "returncode": -1,
        "stdout": "",
        "stderr": "not attempted",
    }
    for attempt in range(1, 4):
        try:
            query = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            query_row = {
                "attempt": attempt,
                "returncode": query.returncode,
                "stdout": query.stdout.strip(),
                "stderr": query.stderr.strip(),
            }
        except Exception as error:
            query_row = {
                "attempt": attempt,
                "returncode": -1,
                "stdout": "",
                "stderr": f"{type(error).__name__}: {error}",
            }
        query_attempts.append(copy.deepcopy(query_row))
        values = [value.strip() for value in query_row["stdout"].split(",")]
        if query_row["returncode"] == 0 and len(values) == 7:
            break
        time.sleep(0.25)
    nvidia = {
        "command": command,
        "attempts": query_attempts,
        "returncode": query_row["returncode"],
        "stdout": query_row["stdout"],
        "stderr": query_row["stderr"],
        "index": values[0] if len(values) == 7 else None,
        "name": values[1] if len(values) == 7 else None,
        "uuid": values[2] if len(values) == 7 else None,
        "driver_version": values[3] if len(values) == 7 else None,
        "memory_total_mib": values[4] if len(values) == 7 else None,
        "compute_capability": values[5] if len(values) == 7 else None,
        "power_max_w": values[6] if len(values) == 7 else None,
    }
    torch_row = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": available,
        "device_index": torch.cuda.current_device() if available else None,
        "name": props.name if props is not None else None,
        "total_memory_bytes": props.total_memory if props is not None else None,
        "compute_capability": [props.major, props.minor] if props is not None else None,
        "sm_count": props.multi_processor_count if props is not None else None,
        "warp_size": props.warp_size if props is not None else None,
        "max_threads_per_sm": (
            props.max_threads_per_multi_processor if props is not None else None
        ),
    }
    warp_slots_per_sm = (
        int(props.max_threads_per_multi_processor // props.warp_size)
        if props is not None
        else None
    )
    total_warp_slots = (
        int(warp_slots_per_sm * props.multi_processor_count)
        if props is not None
        else None
    )
    checks = {
        "registered_python": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "torch_2_7_0_cu128": torch_row["torch_version"] == "2.7.0+cu128",
        "torch_cuda_12_8": torch_row["torch_cuda"] == "12.8",
        "cuda_available": available,
        "cuda_device_zero": torch_row["device_index"] == 0,
        "gpu_name_exact": torch_row["name"] == "NVIDIA GeForce RTX 4090 Laptop GPU",
        "compute_capability_8_9": torch_row["compute_capability"] == [8, 9],
        "sm_count_76": torch_row["sm_count"] == 76,
        "warp_size_32": torch_row["warp_size"] == 32,
        "max_threads_per_sm_1536": torch_row["max_threads_per_sm"] == 1536,
        "total_memory_exact": torch_row["total_memory_bytes"] == 16718168064,
        "nvidia_query_pass": query_row["returncode"] == 0 and len(values) == 7,
        "nvidia_index_zero": nvidia["index"] == "0",
        "nvidia_name_exact": nvidia["name"]
        == "NVIDIA GeForce RTX 4090 Laptop GPU",
        "nvidia_memory_total_16376_mib": nvidia["memory_total_mib"] == "16376",
        "driver_580_159_03": nvidia["driver_version"] == "580.159.03",
        "uuid_exact": nvidia["uuid"]
        == "GPU-05b1a3f8-b7cf-dc57-06aa-741fe2daa4b4",
        "nvidia_compute_capability_8_9": nvidia["compute_capability"] == "8.9",
    }
    return {
        "artifact": "D352_GPU_HARDWARE_CONTRACT_V1",
        "torch": torch_row,
        "nvidia_smi": nvidia,
        "derived_capacity_not_achieved_occupancy": {
            "maximum_resident_warp_slots_per_sm": warp_slots_per_sm,
            "maximum_resident_warp_slots_all_sms": total_warp_slots,
            "semantics": (
                "capacity only; nvidia-smi GPU utilization is active-time sampling, "
                "not achieved warp occupancy"
            ),
        },
        "no_gpu_configuration_mutation": {
            "clock_change": False,
            "power_limit_change": False,
            "persistence_mode_change": False,
            "profiler_or_kernel_replay": False,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _live_trace_line_map() -> dict[str, Any]:
    target = d351.d349._build_live_topology_parts
    lines, start = inspect.getsourcelines(target)

    def unique_line(fragment: str) -> int:
        matches = [start + idx for idx, line in enumerate(lines) if fragment in line]
        if len(matches) != 1:
            raise RuntimeError(f"D352 trace line map ambiguous for {fragment!r}: {matches}")
        return matches[0]

    result = {
        "source_path": _rel(Path(inspect.getsourcefile(target)).resolve()),
        "source_sha256": _sha(Path(inspect.getsourcefile(target)).resolve()),
        "function_first_line": target.__code__.co_firstlineno,
        "part_start_line": unique_line("body, name = source[\"body\"], source[\"name\"]"),
        "loop_exit_line": unique_line("body_checks = {}"),
        "return_line": unique_line("return parts, report"),
    }
    checks = {
        "source_hash_exact": result["source_sha256"]
        == "33a9743337fa269b71e4da3ccccfabc1d746ee29e1582a3d0f8c4764f42d68b9",
        "function_first_line_exact": result["function_first_line"] == 816,
        "part_start_line_exact": result["part_start_line"] == 835,
        "loop_exit_line_exact": result["loop_exit_line"] == 921,
        "return_line_exact": result["return_line"] == 960,
    }
    result["checks"] = checks
    result["pass"] = all(checks.values())
    return result


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _ast_scope_contract() -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"), filename=str(HARNESS))
    calls = sorted(
        _call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    )
    imports = sorted(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    imports.extend(
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    forbidden_exact = {
        "d351._evaluate_q5",
        "d351._run_validate",
        "d351a2._build_live_with_reactive_pause",
        "d351a2._run_validate_with_reactive_repairs",
        "simulation_app.update",
        "inner.step",
        "inner.sim.step",
        "d351._write_exact_state",
    }
    found = sorted(set(calls) & forbidden_exact)
    checks = {
        "forbidden_calls_absent": not found,
        "q5_grid_generation_absent": "np.linspace" not in calls,
        "moving_surface_measurement_call_absent": not any(
            "_moving_surface" in name for name in calls
        ),
        "viewer_rerun_import_absent": not any(
            name == "rerun" or name.startswith("rerun.") for name in imports
        )
        and not any(name.endswith("log_rerun") for name in calls),
        "hardware_control_absent": not any(
            name.endswith(("joints_angle_ctrl", "torque_set", "move_init"))
            for name in calls
        ),
    }
    return {
        "artifact": "D352_AST_SCOPE_CONTRACT_V1",
        "forbidden_exact": sorted(forbidden_exact),
        "found_forbidden_calls": found,
        "permitted_inherited_playback_controls": {
            "initial_base_pause_reproduction": 1,
            "attempt2_pause_without_update_calls": 2,
            "meaning": "D351/attempt2 baseline playback suppression, not a physics-solver parameter change",
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _parameter_freeze_audit() -> dict[str, Any]:
    scope = {
        "asset_write_or_recook": False,
        "decomposition_change": False,
        "fresh_property_or_cook_query": False,
        "q0_q4_or_object_pose_change": False,
        "q5_science_sample": False,
        "moving_surface_measurement": False,
        "geometry_or_current_pose_verdict": None,
        "target_ik_or_path_change": False,
        "gate_or_tolerance_change": False,
        "material_mass_actuator_physics_solver_change": False,
        "new_playback_or_timeline_control_beyond_d351_attempt2": False,
        "controlled_physics_steps_planned": 0,
        "simulation_app_update": False,
        "settle": False,
        "ten_trial": False,
        "g0b": False,
        "rl_or_ppo": False,
        "vla": False,
        "ladder_promotion": False,
        "g0a_pass": False,
    }
    checks = {
        "operational_variables_exact_two": NEW_OPERATIONAL_VARIABLES
        == ["durable_phase_marker_stream", "external_bounded_wall_clock_watchdog"],
        "scientific_variables_zero": NEW_SCIENTIFIC_VARIABLES == [],
        "physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "seed_frozen": SEED == 33201,
        "inactivity_watchdog_120s": INACTIVITY_WATCHDOG_S == 120.0,
        "total_watchdog_300s": TOTAL_WATCHDOG_S == 300.0,
        "term_grace_30s": TERM_GRACE_S == 30.0,
        "gpu_sample_nominal_1s": GPU_SAMPLE_PERIOD_S == 1.0,
        "no_true_occupancy_claim": True,
        "scope_boolean_guards_false": not any(
            value is True
            for key, value in scope.items()
            if key not in {"g0a_pass"}
        ),
        "planned_steps_zero_not_result_authority": scope[
            "controlled_physics_steps_planned"
        ]
        == 0,
    }
    return {
        "artifact": "D352_PARAMETER_FREEZE_AUDIT_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "new_operational_variables": NEW_OPERATIONAL_VARIABLES,
        "new_scientific_variables": NEW_SCIENTIFIC_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "watchdog": {
            "marker_inactivity_s": INACTIVITY_WATCHDOG_S,
            "total_worker_wall_clock_s": TOTAL_WATCHDOG_S,
            "faulthandler_signal": "SIGUSR1",
            "faulthandler_grace_s": FAULT_DUMP_GRACE_S,
            "termination_signal": "SIGTERM",
            "termination_grace_s": TERM_GRACE_S,
            "escalation_if_still_alive": "SIGKILL",
            "automatic_retry": False,
        },
        "required_localization_boundaries": [
            "source inventory and input hashes",
            "AppLauncher start/end",
            "_make_runtime_env start/end",
            "reset start/end",
            "corrected audit start/end",
            "attempt2 first passive bridge snapshot start/end",
            "live builder enter/return",
            "live part 000..127 start/end",
            "attempt2 live payload hash and zero-step pause bridge boundaries",
            "full payload deepcopy/serialization/write start/end",
            "raw binding start/end and final bridge before/after",
            "SimulationApp close start/end",
        ],
        "result_semantics": {
            "bridge_complete": "controlled_physics_steps may be reported as 0 only if exact counter/clock bridge passes",
            "bridge_incomplete": "controlled_physics_steps remains null",
            "all_outcomes": {
                "scientific_verdict": None,
                "current_pose_support_or_rejection": None,
                "target_ik_repair_justification": None,
                "g0a_pass": False,
            },
        },
        "rerun_omission": (
            "operational control-flow/file-log localization only; no spatial geometry, "
            "pose, trajectory, contact, or synchronized sensor-time verdict"
        ),
        "scope_guards": scope,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"forward-only D352 output already nonempty: {OUT_DIR}")
    status = d351._git_status()
    parameter = _parameter_freeze_audit()
    gpu = _gpu_hardware_contract()
    immutable = _immutable_d351_contract()
    sidecar = _user_sidecar_contract()
    line_map = _live_trace_line_map()
    ast_scope = _ast_scope_contract()
    inputs = d351._input_hashes()
    start_text = START_HERE.read_text(encoding="utf-8")
    session_text = SESSION_DOC.read_text(encoding="utf-8") if SESSION_DOC.is_file() else ""
    prechecks = {
        "head_exact": _git_head() == EXPECTED_HEAD,
        "origin_master_exact": _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d352": _status_scope_pass(status),
        "start_here_active_case_exact": CASE_NAME in start_text
        and "claudedocs/runtime_logs/grasp_track/g0a_d352/" in start_text,
        "session_doc_exists": SESSION_DOC.is_file(),
        "session_declares_two_operational_variables": all(
            value in session_text for value in NEW_OPERATIONAL_VARIABLES
        ),
        "d351_inputs_exact": inputs == d351.EXPECTED_INPUT_HASHES,
        "d351_attempts_immutable": immutable["pass"],
        "user_sidecar_read_only_exact": sidecar["pass"],
        "gpu_hardware": gpu["pass"],
        "parameter_freeze": parameter["pass"],
        "trace_line_map": line_map["pass"],
        "ast_scope": ast_scope["pass"],
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D352 prepare STOP: {prechecks}")

    _write_json(PARAMETER_PATH, parameter)
    _write_json(GPU_HARDWARE_PATH, gpu)
    run_nonce = secrets.token_hex(16)
    prereg = {
        "artifact": "D352_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "run_nonce": run_nonce,
        "git_head": _git_head(),
        "origin_master": _git_head("origin/master"),
        "git_status_before_prepare_outputs": status,
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "output_root": _rel(OUT_DIR),
        "harness_sha256": _sha(HARNESS),
        "state_hashes": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        },
        "parameter_path": _rel(PARAMETER_PATH),
        "parameter_sha256": _sha(PARAMETER_PATH),
        "gpu_hardware_path": _rel(GPU_HARDWARE_PATH),
        "gpu_hardware_sha256": _sha(GPU_HARDWARE_PATH),
        "input_hashes": inputs,
        "d351_immutability": immutable,
        "user_sidecar": sidecar,
        "trace_line_map": line_map,
        "ast_scope": ast_scope,
        "watchdog": parameter["watchdog"],
        "gpu_telemetry": {
            "nominal_cadence_s": GPU_SAMPLE_PERIOD_S,
            "device_metrics": [
                "gpu utilization active-time sample",
                "memory utilization",
                "framebuffer memory",
                "SM/memory clocks",
                "P-state",
                "power",
                "temperature",
            ],
            "process_metrics": ["CPU percent", "RSS", "threads", "status", "I/O"],
            "scientific_or_warp_occupancy_authority": False,
        },
        "single_effective_validate": True,
        "automatic_retry": False,
        "q5_science_authorized_inside_d352": False,
        "post_d352_q5_boundary": "requires result briefing and separate explicit confirmation",
        "prechecks": prechecks,
        "pass": all(prechecks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": True, "run_nonce": run_nonce}))
    return 0


def _runtime_paths_before_worker() -> set[Path]:
    return {PARAMETER_PATH, GPU_HARDWARE_PATH, PREREG_PATH}


def _supervisor_preflight(args: argparse.Namespace) -> dict[str, Any]:
    prereg = _json(PREREG_PATH)
    existing = {path for path in OUT_DIR.iterdir() if path.is_file()}
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "fresh_supervisor_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_supervisor_nonce": prereg.get("prepare_process_identity", {}).get("nonce")
        != args.process_nonce,
        "head_and_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d352": _status_scope_pass(d351._git_status()),
        "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
        "state_hashes_exact": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "parameter_hash_exact": _sha(PARAMETER_PATH)
        == prereg.get("parameter_sha256"),
        "gpu_hash_exact": _sha(GPU_HARDWARE_PATH)
        == prereg.get("gpu_hardware_sha256"),
        "d351_attempts_immutable": _immutable_d351_contract()["pass"],
        "user_sidecar_read_only_exact": _user_sidecar_contract()["pass"],
        "inputs_exact": d351._input_hashes()
        == d351.EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "initial_output_inventory_exact": existing == _runtime_paths_before_worker(),
        "watchdog_exact": float(args.inactivity_watchdog_s) == INACTIVITY_WATCHDOG_S
        and float(args.total_watchdog_s) == TOTAL_WATCHDOG_S,
        "display_exact": os.environ.get("DISPLAY") == ":1",
        "single_effective_validate": prereg.get("single_effective_validate") is True,
        "automatic_retry_false": prereg.get("automatic_retry") is False,
    }
    return {
        "artifact": "D352_SUPERVISOR_PREFLIGHT_V1",
        "pid": os.getpid(),
        "process_nonce": args.process_nonce,
        "run_nonce": prereg.get("run_nonce"),
        "existing_files": sorted(_rel(path) for path in existing),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _worker_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch

    prereg = _json(PREREG_PATH)
    gpu_now = _gpu_hardware_contract()
    supervisor_preflight = _json(SUPERVISOR_PREFLIGHT_PATH)
    launch_token = os.environ.get(WORKER_LAUNCH_TOKEN_ENV, "")
    supervisor_pid_raw = os.environ.get(SUPERVISOR_PID_ENV, "")
    try:
        supervisor_pid = int(supervisor_pid_raw)
    except ValueError:
        supervisor_pid = -1
    required_existing = {
        PARAMETER_PATH,
        GPU_HARDWARE_PATH,
        PREREG_PATH,
        SUPERVISOR_PREFLIGHT_PATH,
        MARKER_PATH,
        WORKER_LOG_PATH,
        FAULT_PATH,
    }
    permitted_existing = {
        PARAMETER_PATH,
        GPU_HARDWARE_PATH,
        PREREG_PATH,
        SUPERVISOR_PREFLIGHT_PATH,
        MARKER_PATH,
        WORKER_LOG_PATH,
        GPU_TELEMETRY_PATH,
        FAULT_PATH,
    }
    existing = {path for path in OUT_DIR.iterdir() if path.is_file()}
    app_checks = d351.d350._app_arg_checks(args)
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "run_nonce_exact": _RUN_NONCE == prereg.get("run_nonce"),
        "supervisor_parent_pid_exact": supervisor_pid > 0
        and os.getppid() == supervisor_pid
        and supervisor_preflight.get("pid") == supervisor_pid,
        "supervisor_launch_token_exact": bool(launch_token)
        and hashlib.sha256(launch_token.encode("utf-8")).hexdigest()
        == supervisor_preflight.get("worker_launch", {}).get("token_sha256"),
        "fresh_worker_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_d352": _status_scope_pass(d351._git_status()),
        "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
        "state_hashes_exact": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "d351_attempts_immutable": _immutable_d351_contract()["pass"],
        "user_sidecar_read_only_exact": _user_sidecar_contract()["pass"],
        "d351_inputs_exact": d351._input_hashes()
        == d351.EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "python_exact": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "numpy_1_26_0": numpy.__version__ == "1.26.0",
        "psutil_5_9_8": psutil.__version__ == "5.9.8",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_hardware_exact": gpu_now["pass"]
        and gpu_now["torch"] == _json(GPU_HARDWARE_PATH)["torch"],
        "display_exact": os.environ.get("DISPLAY") == ":1",
        "app_args_exact": all(app_checks.values()),
        "q5_fail_closed_guard_installed": d351._evaluate_q5 is _forbidden_q5,
        "q5_trap_count_zero": _Q5_TRAP_INVOCATION_COUNT == 0,
        "trace_line_map_exact": _live_trace_line_map()
        == prereg.get("trace_line_map"),
        "runtime_inventory_required_supervisor_boot_files": required_existing <= existing,
        "runtime_inventory_only_supervisor_boot_files": existing <= permitted_existing,
        "science_outputs_absent": all(
            not path.exists()
            for path in (
                LIVE_BINDING_PATH,
                RAW_CONTRACT_PATH,
                BRIDGE_PATH,
                LOCALIZATION_PATH,
                RUNTIME_EXCEPTION_PATH,
                WATCHDOG_PROC_PATH,
                SUPERVISOR_AUDIT_PATH,
            )
        ),
    }
    return {
        "artifact": "D352_VALIDATE_PREFLIGHT_V1",
        "pid": os.getpid(),
        "process_nonce": args.process_nonce,
        "run_nonce": _RUN_NONCE,
        "environment": {
            "display": os.environ.get("DISPLAY"),
            "gpu": gpu_now,
            "app_arg_checks": app_checks,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _forbidden_q5(*_args: Any, **_kwargs: Any) -> Any:
    global _Q5_TRAP_INVOCATION_COUNT

    _Q5_TRAP_INVOCATION_COUNT += 1
    _marker(
        "worker",
        "q5_fail_closed_boundary",
        "breach_trapped",
        {"message": "original D351 q5 evaluator was not called"},
    )
    raise RuntimeError("D352 localization-only q5 boundary breach trapped before science")


class _LivePartTrace:
    def __init__(self, target: Any, line_map: dict[str, Any], expected: list[tuple[str, str]]):
        self.target = target
        self.line_map = line_map
        self.expected = expected
        self.active: tuple[int, str, str] | None = None
        self.starts = 0
        self.ends = 0
        self.return_seen = False

    def global_trace(self, frame: Any, event: str, _arg: Any) -> Any:
        if event == "call" and frame.f_code is self.target.__code__:
            return self.local_trace
        return None

    def _complete_previous(self, frame: Any) -> None:
        if self.active is None:
            return
        index, body, name = self.active
        public_rows = frame.f_locals.get("public_rows", [])
        if len(public_rows) != index + 1:
            raise RuntimeError(
                f"D352 live trace completion mismatch index={index} rows={len(public_rows)}"
            )
        _marker(
            "worker",
            "live_part",
            "end",
            {"part_index": index, "body": body, "name": name},
        )
        self.ends += 1
        self.active = None

    def local_trace(self, frame: Any, event: str, _arg: Any) -> Any:
        if event == "line" and frame.f_lineno == self.line_map["part_start_line"]:
            self._complete_previous(frame)
            source = frame.f_locals.get("source")
            public_rows = frame.f_locals.get("public_rows", [])
            if not isinstance(source, dict):
                raise RuntimeError("D352 live trace source local missing")
            index = len(public_rows)
            body, name = str(source["body"]), str(source["name"])
            if index >= len(self.expected) or (body, name) != self.expected[index]:
                raise RuntimeError(
                    f"D352 live trace identity mismatch index={index} {(body, name)}"
                )
            _marker(
                "worker",
                "live_part",
                "start",
                {"part_index": index, "body": body, "name": name},
            )
            self.active = (index, body, name)
            self.starts += 1
        elif event == "line" and frame.f_lineno == self.line_map["loop_exit_line"]:
            self._complete_previous(frame)
        elif event == "return":
            self.return_seen = True
        return self.local_trace


def _build_live_with_markers(inner: Any) -> tuple[Any, Any, dict[str, Any]]:
    target = d351a2._ORIGINAL_BUILD_LIVE
    line_map = _live_trace_line_map()
    evidence = _json(d351.D348_EVIDENCE)
    expected = [(str(row["body"]), str(row["name"])) for row in evidence["rows"]]
    if len(expected) != 128:
        raise RuntimeError(f"D352 expected 128 live rows, got {len(expected)}")
    tracer = _LivePartTrace(target, line_map, expected)
    previous = sys.gettrace()
    _marker("worker", "live_builder", "start", {"expected_parts": 128})
    sys.settrace(tracer.global_trace)
    try:
        parts, report = target(inner)
    finally:
        sys.settrace(previous)
        _marker(
            "worker",
            "live_trace_restore",
            "end",
            {"restored_exact": sys.gettrace() is previous},
        )
    trace_contract = {
        "starts": tracer.starts,
        "ends": tracer.ends,
        "return_seen": tracer.return_seen,
        "expected_identity_sha256": hashlib.sha256(
            json.dumps(expected, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "checks": {
            "parts_started_128": tracer.starts == 128,
            "parts_ended_128": tracer.ends == 128,
            "target_return_seen": tracer.return_seen,
            "trace_restored": sys.gettrace() is previous,
        },
    }
    trace_contract["pass"] = all(trace_contract["checks"].values())
    _marker(
        "worker",
        "live_builder",
        "end",
        {"report_pass": report.get("pass"), "trace": trace_contract},
    )
    return parts, report, trace_contract


def _bridge_snapshot(inner: Any, phase: str) -> dict[str, Any]:
    _marker("worker", "zero_step_bridge_snapshot", "start", {"name": phase})
    row = d351a2._bridge_snapshot(inner, phase)
    _marker(
        "worker",
        "zero_step_bridge_snapshot",
        "end",
        {
            "name": phase,
            "custom_step_counter": row["custom_step_counter"],
            "timeline_playing": row["timeline_playing"],
            "timeline_time": row["timeline_time"],
            "simulation_context_clock": row["simulation_context_clock"],
        },
    )
    return row


def _run_localization(args: argparse.Namespace, launcher_report: dict[str, Any]) -> int:
    global _ACTIVE_INNER, _BRIDGE_COMPLETE

    import omni.timeline

    snapshots: list[dict[str, Any]] = []
    pause_events: list[dict[str, Any]] = []

    _marker("worker", "source_inventory", "start")
    source_before = d351.d349._source_inventories()
    _marker("worker", "source_inventory", "end")
    _marker("worker", "input_hashes", "start")
    input_before = d351._input_hashes()
    _marker("worker", "input_hashes", "end")

    args.robot_usd_path = d351.VARIANT_ROBOT_USD
    _marker("worker", "_make_runtime_env", "start")
    inner = d351.d333._make_runtime_env(args)
    _ACTIVE_INNER = inner
    _marker("worker", "_make_runtime_env", "end")
    timeline = omni.timeline.get_timeline_interface()

    _marker("worker", "reset", "start")
    inner.reset(seed=SEED)
    inner.sim.set_setting("/app/player/playSimulations", False)
    initial_pause_interventions = 0
    if timeline.is_playing():
        timeline.pause()
        initial_pause_interventions += 1
    counter_after_reset = int(inner._sim_step_counter)
    _marker(
        "worker",
        "reset",
        "end",
        {
            "counter_after_reset": counter_after_reset,
            "timeline_playing": bool(timeline.is_playing()),
            "initial_pause_interventions": initial_pause_interventions,
        },
    )

    _marker("worker", "corrected_audit", "start")
    corrected = d351.d349._corrected_live_audit()
    _marker(
        "worker",
        "corrected_audit",
        "end",
        {"pass": corrected.get("pass")},
    )

    snapshots.append(
        _bridge_snapshot(inner, "after_reset_initial_pause_before_live_binding")
    )
    topology_parts, live_binding, trace_contract = _build_live_with_markers(inner)

    _marker("worker", "live_payload_hash", "start")
    original_payload_sha256 = d351a2._payload_sha(live_binding)
    _marker(
        "worker",
        "live_payload_hash",
        "end",
        {"sha256": original_payload_sha256},
    )
    snapshots.append(_bridge_snapshot(inner, "after_live_binding_before_repause"))
    _marker("worker", "attempt2_live_repause", "start")
    event = d351a2._pause_without_update(inner, "after_live_topology_binding")
    pause_events.append(event)
    _marker("worker", "attempt2_live_repause", "end", event)
    snapshots.append(_bridge_snapshot(inner, "after_live_binding_after_repause"))
    live_binding["attempt2_reactive_timeline_pause"] = event
    live_binding["attempt2_original_live_binding_payload_sha256"] = (
        original_payload_sha256
    )
    live_binding["checks"]["attempt2_live_binding_pause_contract"] = event["pass"]
    live_binding["checks"]["attempt2_original_live_binding_payload_reproduces_attempt1"] = (
        original_payload_sha256
        == d351a2.ATTEMPT1_ROOT_HASHES["d351_live_topology_runtime_binding.json"]
    )
    live_binding["pass"] = all(live_binding["checks"].values())

    _marker("worker", "live_payload_deepcopy_serialize_write", "start")
    live_payload = _payload_bytes(copy.deepcopy(live_binding))
    _write_bytes(LIVE_BINDING_PATH, live_payload)
    _marker(
        "worker",
        "live_payload_deepcopy_serialize_write",
        "end",
        {
            "bytes": len(live_payload),
            "sha256": hashlib.sha256(live_payload).hexdigest(),
        },
    )

    _marker("worker", "raw_binding", "start")
    _raw_shapes, raw_contract = d351a2._ORIGINAL_BUILD_RAW(inner, _json(d351.D334_SUMMARY))
    _marker(
        "worker", "raw_binding", "end", {"pass": raw_contract.get("pass")}
    )
    snapshots.append(
        _bridge_snapshot(inner, "after_live_and_raw_binding_before_final_repause")
    )
    _marker("worker", "attempt2_final_repause", "start")
    event = d351a2._pause_without_update(
        inner, "after_raw_shape_binding_before_prerequisites"
    )
    pause_events.append(event)
    _marker("worker", "attempt2_final_repause", "end", event)
    snapshots.append(
        _bridge_snapshot(inner, "after_live_and_raw_binding_after_final_repause")
    )

    d351a2._BRIDGE_SNAPSHOTS.clear()
    d351a2._BRIDGE_SNAPSHOTS.extend(copy.deepcopy(snapshots))
    bridge = d351a2._bridge_contract()
    bridge["d352_scope"] = {
        "localization_only": True,
        "q5_evaluation_count": _Q5_TRAP_INVOCATION_COUNT,
        "new_playback_or_physics_setting_changes_beyond_frozen_d351_attempt2": 0,
        "inherited_playback_controls_reproduced": {
            "base_initial_playSimulations_false_write": 1,
            "base_initial_timeline_pause_interventions": initial_pause_interventions,
            "attempt2_pause_without_update_calls": len(pause_events),
            "attempt2_pause_events": pause_events,
        },
    }
    _write_json(RAW_CONTRACT_PATH, raw_contract)
    _write_json(BRIDGE_PATH, bridge)
    _BRIDGE_COMPLETE = bool(bridge["pass"])

    _marker("worker", "post_bridge_source_inventory", "start")
    source_after = d351.d349._source_inventories()
    input_after = d351._input_hashes()
    immutable_after = _immutable_d351_contract()
    _marker("worker", "post_bridge_source_inventory", "end")

    prerequisites = {
        "launcher": launcher_report["pass"],
        "counter_after_reset_zero": counter_after_reset == 0,
        "corrected_d348_128_of_128": corrected["pass"],
        "live_trace_128_start_end": trace_contract["pass"],
        "live_binding_64_plus_64": live_binding["pass"],
        "raw_source_contract": raw_contract["pass"],
        "exact_five_snapshot_zero_step_bridge": bridge["pass"],
        "source_inventory_unchanged": source_after == source_before,
        "input_hashes_unchanged": input_after == input_before,
        "d351_attempts_immutable_after": immutable_after["pass"],
        "q5_trap_invocation_count_zero": _Q5_TRAP_INVOCATION_COUNT == 0,
    }
    controlled_steps = 0 if all(prerequisites.values()) else None
    summary = {
        "artifact": "D352_LOCALIZATION_SUMMARY_V1",
        "case": CASE,
        "operational_verdict": (
            "D352_LOCALIZATION_BOUNDARY_COMPLETE_NO_SCIENCE"
            if all(prerequisites.values())
            else "D352_LOCALIZATION_CONTRACT_FAIL_STOP"
        ),
        "question": "localize D351 attempt2 pre-science long run",
        "prerequisites": prerequisites,
        "trace_contract": trace_contract,
        "pause_events": pause_events,
        "bridge_path": _rel(BRIDGE_PATH),
        "bridge_sha256": _sha(BRIDGE_PATH),
        "live_binding_path": _rel(LIVE_BINDING_PATH),
        "live_binding_sha256": _sha(LIVE_BINDING_PATH),
        "raw_contract_path": _rel(RAW_CONTRACT_PATH),
        "raw_contract_sha256": _sha(RAW_CONTRACT_PATH),
        "d352_q5_evaluation_count": _Q5_TRAP_INVOCATION_COUNT,
        "d352_controlled_physics_steps": controlled_steps,
        "inherited_d351_attempt2_q5_evaluation_count": 0,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "moving_surface_measurement": None,
        "q5_sweep": None,
        "viewer_rerun_rrd_rbl": None,
        "g0a_pass": False,
        "automatic_retry": False,
        "commit_or_push_performed": False,
        "pass": all(prerequisites.values()),
    }
    _write_json(LOCALIZATION_PATH, summary)
    _marker(
        "worker",
        "localization_boundary",
        "complete",
        {"pass": summary["pass"], "q5_count": _Q5_TRAP_INVOCATION_COUNT},
    )
    return 0 if summary["pass"] else 2


def _sample_gpu_cpu(
    worker_pid: int,
    sample_index: int,
    worker_process: psutil.Process | None,
) -> dict[str, Any]:
    query_fields = [
        "timestamp",
        "index",
        "uuid",
        "pstate",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "temperature.gpu",
    ]
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(query_fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        sample_start_wall_time_ns = time.time_ns()
        sample_start_monotonic_ns = time.monotonic_ns()
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        gpu = dict(zip(query_fields, values, strict=False))
        sample_valid = bool(
            result.returncode == 0
            and len(values) == len(query_fields)
            and gpu.get("uuid") == "GPU-05b1a3f8-b7cf-dc57-06aa-741fe2daa4b4"
        )
        gpu.update(
            {
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                "field_count_exact": len(values) == len(query_fields),
                "sample_valid": sample_valid,
            }
        )
    except Exception as error:
        sample_start_wall_time_ns = time.time_ns()
        sample_start_monotonic_ns = time.monotonic_ns()
        gpu = {
            "error": f"{type(error).__name__}: {error}",
            "field_count_exact": False,
            "sample_valid": False,
        }

    process: dict[str, Any]
    try:
        proc = worker_process if worker_process is not None else psutil.Process(worker_pid)
        memory = proc.memory_info()
        io = proc.io_counters()
        process = {
            "exists": True,
            "status": proc.status(),
            "cpu_percent": proc.cpu_percent(interval=None),
            "rss_bytes": memory.rss,
            "vms_bytes": memory.vms,
            "num_threads": proc.num_threads(),
            "read_bytes": io.read_bytes,
            "write_bytes": io.write_bytes,
        }
    except Exception as error:
        process = {"exists": False, "error": f"{type(error).__name__}: {error}"}

    markers, _invalid = _read_markers(_RUN_NONCE)
    worker_markers = [row for row in markers if row.get("actor") == "worker"]
    last = worker_markers[-1] if worker_markers else None
    return {
        "artifact": "D352_GPU_CPU_TELEMETRY_SAMPLE_V1",
        "run_nonce": _RUN_NONCE,
        "sample_index": sample_index,
        "sample_start_wall_time_ns": sample_start_wall_time_ns,
        "sample_start_monotonic_ns": sample_start_monotonic_ns,
        "sample_end_monotonic_ns": time.monotonic_ns(),
        "sample_duration_s": (time.monotonic_ns() - sample_start_monotonic_ns)
        / 1.0e9,
        "worker_pid": worker_pid,
        "last_valid_marker": None
        if last is None
        else {
            "actor": last["actor"],
            "pid": last["pid"],
            "sequence": last["sequence"],
            "phase": last["phase"],
            "event": last["event"],
            "details": last.get("details"),
        },
        "gpu": gpu,
        "process": process,
    }


def _telemetry_loop(
    worker_pid: int,
    worker_process: psutil.Process | None,
    stop_event: threading.Event,
) -> None:
    sample_index = 0
    next_sample = time.monotonic()
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_sample:
            stop_event.wait(min(0.25, next_sample - now))
            continue
        sample = _sample_gpu_cpu(worker_pid, sample_index, worker_process)
        _append_jsonl(GPU_TELEMETRY_PATH, sample)
        sample_index += 1
        next_sample += GPU_SAMPLE_PERIOD_S
        now = time.monotonic()
        if next_sample <= now:
            next_sample = now + GPU_SAMPLE_PERIOD_S


def _read_markers(expected_nonce: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not MARKER_PATH.is_file():
        return [], []
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    last_sequence: dict[tuple[str, int], int] = {}
    with MARKER_PATH.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith(b"\n"):
                invalid.append({"line": line_number, "reason": "partial_line"})
                continue
            try:
                row = json.loads(raw)
            except Exception as error:
                invalid.append(
                    {"line": line_number, "reason": f"json:{type(error).__name__}"}
                )
                continue
            try:
                key = (str(row.get("actor")), int(row.get("pid", -1)))
                sequence = int(row.get("sequence", -1))
            except (TypeError, ValueError):
                invalid.append(
                    {"line": line_number, "reason": "pid_or_sequence_type", "row": row}
                )
                continue
            reason = None
            if row.get("run_nonce") != expected_nonce:
                reason = "nonce"
            elif sequence != last_sequence.get(key, 0) + 1:
                reason = "sequence"
            if reason is not None:
                invalid.append({"line": line_number, "reason": reason, "row": row})
                continue
            last_sequence[key] = sequence
            valid.append(row)
    return valid, invalid


def _proc_snapshot(worker_pid: int) -> dict[str, Any]:
    proc_root = Path(f"/proc/{worker_pid}")
    files = {}
    for name in ("status", "stat", "wchan", "io", "cmdline"):
        path = proc_root / name
        try:
            raw = path.read_bytes()
            files[name] = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except Exception as error:
            files[name] = f"{type(error).__name__}: {error}"
    tasks = []
    task_root = proc_root / "task"
    if task_root.is_dir():
        for path in sorted(task_root.iterdir(), key=lambda item: int(item.name))[:256]:
            row = {"tid": int(path.name)}
            for name in ("comm", "wchan"):
                try:
                    row[name] = (path / name).read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                except Exception as error:
                    row[name] = f"{type(error).__name__}: {error}"
            tasks.append(row)
    try:
        process = psutil.Process(worker_pid)
        children = [
            {"pid": child.pid, "name": child.name(), "status": child.status()}
            for child in process.children(recursive=True)
        ]
    except Exception as error:
        children = [{"error": f"{type(error).__name__}: {error}"}]
    return {
        "artifact": "D352_WATCHDOG_PROC_SNAPSHOT_V1",
        "run_nonce": _RUN_NONCE,
        "wall_time_ns": time.time_ns(),
        "worker_pid": worker_pid,
        "proc_files": files,
        "tasks": tasks,
        "children": children,
    }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _telemetry_summary() -> dict[str, Any]:
    rows = []
    if GPU_TELEMETRY_PATH.is_file():
        for raw in GPU_TELEMETRY_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    fields = {
        "gpu_util_percent": "utilization.gpu",
        "memory_util_percent": "utilization.memory",
        "memory_used_mib": "memory.used",
        "sm_clock_mhz": "clocks.current.sm",
        "memory_clock_mhz": "clocks.current.memory",
        "power_w": "power.draw",
        "temperature_c": "temperature.gpu",
    }
    metrics = {}
    for label, source in fields.items():
        values = [
            value
            for row in rows
            if (value := _numeric(row.get("gpu", {}).get(source))) is not None
        ]
        metrics[label] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": float(sum(values) / len(values)) if values else None,
        }
    cpu_values = [
        value
        for row in rows
        if (value := _numeric(row.get("process", {}).get("cpu_percent"))) is not None
    ]
    valid_gpu_samples = sum(
        row.get("gpu", {}).get("sample_valid") is True for row in rows
    )
    invalid_gpu_samples = len(rows) - valid_gpu_samples
    uuid_mismatch_samples = sum(
        row.get("gpu", {}).get("uuid") not in {
            None,
            "GPU-05b1a3f8-b7cf-dc57-06aa-741fe2daa4b4",
        }
        for row in rows
    )
    sample_starts = [
        int(row["sample_start_monotonic_ns"])
        for row in rows
        if row.get("sample_start_monotonic_ns") is not None
    ]
    intervals_s = [
        (right - left) / 1.0e9
        for left, right in zip(sample_starts[:-1], sample_starts[1:], strict=True)
    ]
    return {
        "sample_count": len(rows),
        "valid_gpu_sample_count": valid_gpu_samples,
        "invalid_gpu_sample_count": invalid_gpu_samples,
        "uuid_mismatch_sample_count": uuid_mismatch_samples,
        "sample_start_interval_s": {
            "count": len(intervals_s),
            "min": min(intervals_s) if intervals_s else None,
            "max": max(intervals_s) if intervals_s else None,
            "mean": float(sum(intervals_s) / len(intervals_s))
            if intervals_s
            else None,
            "requested_nominal_s": GPU_SAMPLE_PERIOD_S,
        },
        "metrics": metrics,
        "worker_cpu_percent": {
            "count": len(cpu_values),
            "min": min(cpu_values) if cpu_values else None,
            "max": max(cpu_values) if cpu_values else None,
            "mean": float(sum(cpu_values) / len(cpu_values)) if cpu_values else None,
            "semantics": "psutil process percent; 100% is approximately one logical CPU",
        },
        "warp_occupancy_measured": False,
        "causal_bottleneck_authority": False,
    }


def _last_boundary(markers: list[dict[str, Any]]) -> str:
    worker_markers = [row for row in markers if row.get("actor") == "worker"]
    if not worker_markers:
        return "before first durable marker"
    last = worker_markers[-1]
    phase, event = last.get("phase"), last.get("event")
    details = last.get("details") or {}
    if phase == "live_builder" and event == "start":
        return "inside live builder before first completed boundary"
    if event == "start":
        if phase == "live_part":
            return f"inside live part {details.get('part_index')} {details.get('body')}/{details.get('name')}"
        return f"inside {phase}"
    return f"after {phase}:{event}"


def _run_supervisor(args: argparse.Namespace) -> int:
    global _RUN_NONCE

    preflight = _supervisor_preflight(args)
    _RUN_NONCE = str(preflight["run_nonce"])
    worker_launch_token = secrets.token_hex(32)
    preflight["worker_launch"] = {
        "supervisor_pid": os.getpid(),
        "token_sha256": hashlib.sha256(
            worker_launch_token.encode("utf-8")
        ).hexdigest(),
        "token_persisted_plaintext": False,
    }
    _write_json(SUPERVISOR_PREFLIGHT_PATH, preflight)
    if not preflight["pass"]:
        raise RuntimeError(f"D352 supervisor preflight STOP: {preflight['checks']}")

    command = [
        sys.executable,
        str(HARNESS),
        "--stage",
        "_worker",
        "--out_dir",
        str(OUT_DIR),
        "--seed",
        str(SEED),
        "--inactivity_watchdog_s",
        str(INACTIVITY_WATCHDOG_S),
        "--total_watchdog_s",
        str(TOTAL_WATCHDOG_S),
    ]
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment[SUPERVISOR_PID_ENV] = str(os.getpid())
    environment[WORKER_LAUNCH_TOKEN_ENV] = worker_launch_token
    with WORKER_LOG_PATH.open("xb") as log_stream:
        worker = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _marker(
            "supervisor",
            "worker_process",
            "spawn",
            {"worker_pid": worker.pid, "command": command},
        )
        try:
            worker_process: psutil.Process | None = psutil.Process(worker.pid)
            worker_process.cpu_percent(interval=None)
        except psutil.Error:
            worker_process = None

        start = time.monotonic()
        last_progress = start
        valid_count = 0
        timeout_reason: str | None = None
        timeout_elapsed: float | None = None
        termination_signal: str | None = None
        faulthandler_signal_sent = False
        killed = False
        telemetry_stop = threading.Event()
        telemetry_thread = threading.Thread(
            target=_telemetry_loop,
            args=(worker.pid, worker_process, telemetry_stop),
            name="d352-gpu-cpu-telemetry",
            daemon=True,
        )
        telemetry_thread.start()

        while worker.poll() is None:
            now = time.monotonic()
            markers, _invalid = _read_markers(_RUN_NONCE)
            if len(markers) > valid_count:
                valid_count = len(markers)
                last_progress = now
            total_elapsed = now - start
            inactive_elapsed = now - last_progress
            if total_elapsed >= float(args.total_watchdog_s):
                timeout_reason = "total_wall_clock"
            elif inactive_elapsed >= float(args.inactivity_watchdog_s):
                timeout_reason = "marker_inactivity"
            if timeout_reason is not None:
                markers_recheck, _ = _read_markers(_RUN_NONCE)
                if worker.poll() is not None:
                    if timeout_reason == "marker_inactivity":
                        timeout_reason = None
                    break
                if (
                    timeout_reason == "marker_inactivity"
                    and len(markers_recheck) > valid_count
                ):
                    valid_count = len(markers_recheck)
                    last_progress = time.monotonic()
                    timeout_reason = None
                    continue
                timeout_elapsed = time.monotonic() - start
                _marker(
                    "supervisor",
                    "watchdog",
                    "deadline",
                    {
                        "reason": timeout_reason,
                        "elapsed_s": timeout_elapsed,
                        "last_boundary": _last_boundary(markers_recheck),
                    },
                )
                faulthandler_ready = any(
                    row.get("actor") == "worker"
                    and row.get("phase") == "worker_boot"
                    for row in markers_recheck
                )
                if faulthandler_ready:
                    try:
                        os.kill(worker.pid, signal.SIGUSR1)
                        faulthandler_signal_sent = True
                    except ProcessLookupError:
                        pass
                fault_deadline = time.monotonic() + FAULT_DUMP_GRACE_S
                try:
                    proc_snapshot = _proc_snapshot(worker.pid)
                except Exception as error:
                    proc_snapshot = {
                        "artifact": "D352_WATCHDOG_PROC_SNAPSHOT_V1",
                        "run_nonce": _RUN_NONCE,
                        "worker_pid": worker.pid,
                        "error": f"{type(error).__name__}: {error}",
                    }
                _write_json(WATCHDOG_PROC_PATH, proc_snapshot)
                while worker.poll() is None and time.monotonic() < fault_deadline:
                    time.sleep(0.25)
                if worker.poll() is None:
                    try:
                        os.killpg(worker.pid, signal.SIGTERM)
                        termination_signal = "SIGTERM"
                    except ProcessLookupError:
                        pass
                    term_deadline = time.monotonic() + TERM_GRACE_S
                    while worker.poll() is None and time.monotonic() < term_deadline:
                        time.sleep(0.25)
                if worker.poll() is None:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                        termination_signal = "SIGKILL"
                        killed = True
                    except ProcessLookupError:
                        pass
                break
            time.sleep(0.25)

        try:
            exit_code = worker.wait(timeout=max(5.0, TERM_GRACE_S))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
                killed = True
                termination_signal = "SIGKILL"
            except ProcessLookupError:
                pass
            exit_code = worker.wait(timeout=10.0)

        telemetry_stop.set()
        telemetry_thread.join(timeout=4.0)
        telemetry_thread_alive_after_join = telemetry_thread.is_alive()
        process_group_cleanup_signal: str | None = None
        if _process_group_exists(worker.pid):
            try:
                os.killpg(worker.pid, signal.SIGTERM)
                process_group_cleanup_signal = "SIGTERM"
            except ProcessLookupError:
                pass
            group_deadline = time.monotonic() + TERM_GRACE_S
            while _process_group_exists(worker.pid) and time.monotonic() < group_deadline:
                time.sleep(0.25)
            if _process_group_exists(worker.pid):
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                    process_group_cleanup_signal = "SIGKILL"
                except ProcessLookupError:
                    pass
                kill_deadline = time.monotonic() + 5.0
                while _process_group_exists(worker.pid) and time.monotonic() < kill_deadline:
                    time.sleep(0.25)
        process_group_absent_after_cleanup = not _process_group_exists(worker.pid)

    markers, invalid_markers = _read_markers(_RUN_NONCE)
    localization = _json(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None
    runtime_exception = (
        _json(RUNTIME_EXCEPTION_PATH) if RUNTIME_EXCEPTION_PATH.is_file() else None
    )
    q5_breach_markers = sum(
        row.get("phase") == "q5_fail_closed_boundary" for row in markers
    )
    if timeout_reason is not None:
        verdict = "D352_PHASE_WATCHDOG_STOP"
    elif localization and localization.get("pass") and exit_code == 0:
        verdict = "D352_LOCALIZATION_BOUNDARY_COMPLETE_NO_SCIENCE"
    else:
        verdict = "D352_LOCALIZATION_EXCEPTION_STOP"
    q5_evaluation_count = (
        localization.get("d352_q5_evaluation_count")
        if localization is not None
        else runtime_exception.get("d352_q5_evaluation_count")
        if runtime_exception is not None
        else q5_breach_markers
    )
    controlled_steps = (
        localization.get("d352_controlled_physics_steps")
        if localization is not None
        else runtime_exception.get("d352_controlled_physics_steps")
        if runtime_exception is not None
        else None
    )
    immutable_after = _immutable_d351_contract()
    sidecar_after = _user_sidecar_contract()
    telemetry = _telemetry_summary()
    forbidden_output_names = sorted(
        path.name
        for path in OUT_DIR.iterdir()
        if any(
            token in path.name
            for token in (
                "moving_jaw_surface",
                "closure_geometry_measurement",
                "q5_closure_sweep",
                "viewer_capture",
                ".rrd",
                ".rbl",
            )
        )
    )
    audit = {
        "artifact": "D352_SUPERVISOR_AUDIT_V1",
        "case": CASE,
        "operational_verdict": verdict,
        "worker": {
            "pid": worker.pid,
            "command": command,
            "exit_code": exit_code,
            "process_absent_after_reap": not psutil.pid_exists(worker.pid),
            "process_group_cleanup_signal": process_group_cleanup_signal,
            "process_group_absent_after_cleanup": process_group_absent_after_cleanup,
        },
        "watchdog": {
            "triggered": timeout_reason is not None,
            "reason": timeout_reason,
            "elapsed_s": timeout_elapsed,
            "termination_signal": termination_signal,
            "faulthandler_signal_sent": faulthandler_signal_sent,
            "sigkill_used": killed,
            "automatic_retry": False,
        },
        "markers": {
            "path": _rel(MARKER_PATH),
            "sha256": _sha(MARKER_PATH) if MARKER_PATH.is_file() else None,
            "valid_count": len(markers),
            "invalid_rows": invalid_markers,
            "last_valid": markers[-1] if markers else None,
            "localized_boundary": _last_boundary(markers),
        },
        "telemetry": telemetry,
        "telemetry_thread_alive_after_join": telemetry_thread_alive_after_join,
        "localization_summary": {
            "path": _rel(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None,
            "sha256": _sha(LOCALIZATION_PATH) if LOCALIZATION_PATH.is_file() else None,
            "pass": localization.get("pass") if localization else None,
        },
        "runtime_exception": runtime_exception,
        "d352_q5_evaluation_count": q5_evaluation_count,
        "d352_controlled_physics_steps": controlled_steps,
        "inherited_d351_attempt2_q5_evaluation_count": 0,
        "inherited_d351_attempt2_controlled_physics_steps": None,
        "forbidden_science_or_viewer_outputs": forbidden_output_names,
        "scientific_verdict": None,
        "geometry_pass_or_fail": None,
        "current_pose_support_or_rejection": None,
        "grasp_feasibility": None,
        "target_ik_repair_justification": None,
        "g0a_pass": False,
        "d351_attempts_immutable_after": immutable_after,
        "user_sidecar_read_only_after": sidecar_after,
        "commit_or_push_performed": False,
        "pass": bool(
            verdict == "D352_LOCALIZATION_BOUNDARY_COMPLETE_NO_SCIENCE"
            and localization
            and localization.get("pass")
            and not invalid_markers
            and exit_code == 0
            and not psutil.pid_exists(worker.pid)
            and process_group_absent_after_cleanup
            and q5_evaluation_count == 0
            and controlled_steps == 0
            and not forbidden_output_names
            and immutable_after["pass"]
            and sidecar_after["pass"]
            and telemetry["sample_count"] > 0
            and telemetry["valid_gpu_sample_count"] > 0
            and telemetry["invalid_gpu_sample_count"] == 0
            and telemetry["uuid_mismatch_sample_count"] == 0
            and not telemetry_thread_alive_after_join
        ),
    }
    _write_json(SUPERVISOR_AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "stage": "validate",
                "operational_verdict": verdict,
                "pass": audit["pass"],
                "localized_boundary": audit["markers"]["localized_boundary"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["pass"] else 2


def _run_worker(args: argparse.Namespace) -> int:
    global _ACTIVE_INNER, _RUN_NONCE

    prereg = _json(PREREG_PATH)
    _RUN_NONCE = str(prereg["run_nonce"])
    d351._evaluate_q5 = _forbidden_q5
    simulation_app = None
    fault_stream = FAULT_PATH.open("xb")
    faulthandler.register(signal.SIGUSR1, file=fault_stream, all_threads=True)
    _marker("worker", "worker_boot", "start", {"parent_pid": os.getppid()})
    try:
        preflight = _worker_preflight(args)
        _write_json(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker", "worker_preflight", "end", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D352 worker preflight STOP: {preflight['checks']}")

        from isaaclab.app import AppLauncher

        _marker("worker", "AppLauncher", "start")
        launcher = AppLauncher(copy.deepcopy(args))
        simulation_app = launcher.app
        launcher_report = d351.d350._resolved_gui_launcher(launcher)
        _marker("worker", "AppLauncher", "end", launcher_report)
        if not launcher_report["pass"]:
            raise RuntimeError(f"D352 GUI launcher contract STOP: {launcher_report}")
        return _run_localization(args, launcher_report)
    except Exception as error:
        if not RUNTIME_EXCEPTION_PATH.exists():
            markers, invalid = _read_markers(_RUN_NONCE)
            _write_json(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D352_RUNTIME_EXCEPTION_STOP_V1",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "last_valid_marker": markers[-1] if markers else None,
                    "invalid_marker_rows": invalid,
                    "d352_q5_evaluation_count": _Q5_TRAP_INVOCATION_COUNT,
                    "d352_controlled_physics_steps": 0 if _BRIDGE_COMPLETE else None,
                    "scientific_verdict": None,
                    "geometry_pass_or_fail": None,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        active_exception = sys.exc_info()[1] is not None
        cleanup_error: Exception | None = None
        cleanup_traceback: str | None = None
        if _ACTIVE_INNER is not None:
            _marker("worker", "inner.close", "start")
            try:
                _ACTIVE_INNER.close()
                _marker("worker", "inner.close", "end")
            except Exception as error:
                cleanup_error = error
                cleanup_traceback = traceback.format_exc()
                _marker(
                    "worker",
                    "inner.close",
                    "error",
                    {"error": f"{type(error).__name__}: {error}"},
                )
            finally:
                _ACTIVE_INNER = None
        if simulation_app is not None:
            _marker("worker", "SimulationApp.close", "start")
            try:
                simulation_app.close()
                _marker("worker", "SimulationApp.close", "end")
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
                    cleanup_traceback = traceback.format_exc()
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
        if cleanup_error is not None and not RUNTIME_EXCEPTION_PATH.exists():
            markers, invalid = _read_markers(_RUN_NONCE)
            _write_json(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D352_RUNTIME_EXCEPTION_STOP_V1",
                    "error": (
                        f"cleanup {type(cleanup_error).__name__}: {cleanup_error}"
                    ),
                    "traceback": cleanup_traceback,
                    "last_valid_marker": markers[-1] if markers else None,
                    "invalid_marker_rows": invalid,
                    "d352_q5_evaluation_count": _Q5_TRAP_INVOCATION_COUNT,
                    "d352_controlled_physics_steps": 0 if _BRIDGE_COMPLETE else None,
                    "scientific_verdict": None,
                    "geometry_pass_or_fail": None,
                    "g0a_pass": False,
                },
            )
        if cleanup_error is not None and not active_exception:
            raise RuntimeError("D352 cleanup failed") from cleanup_error


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "validate", "_worker"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--inactivity_watchdog_s", type=float, default=INACTIVITY_WATCHDOG_S
    )
    parser.add_argument("--total_watchdog_s", type=float, default=TOTAL_WATCHDOG_S)
    if stage == "_worker":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--stage", choices=("prepare", "validate", "_worker"), required=True
    )
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D352 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D352 seed drift")
    if float(args.inactivity_watchdog_s) != INACTIVITY_WATCHDOG_S:
        raise RuntimeError("D352 inactivity watchdog drift")
    if float(args.total_watchdog_s) != TOTAL_WATCHDOG_S:
        raise RuntimeError("D352 total watchdog drift")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "validate":
        return _run_supervisor(args)

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
