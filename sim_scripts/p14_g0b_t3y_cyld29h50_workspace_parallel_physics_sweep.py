#!/usr/bin/env python3
"""p14 / t3y — broad workspace, pose-conditioned parallel PhysX grasp sweep.

This is a NEW forward-only runner.  It does not edit or execute p10/p11/p12/t3r/t3w.
It imports the SHA-pinned p10 planner as a library, consumes the exact-theta p13
physics handoff, solves IK at every object position, and executes the feasible plans
with the frozen attempt3 64+64 convex-hull jaws.  The RL kinematic attach is disabled.

The two scientific variables are object planar position and reachable grasp-form
(tool-axis theta stratum).  q5 and descend depth are deterministic, exact-theta p13
controls.  Bilateral static bite is deliberately not an admission gate: unilateral
and no-window rows are retained as labelled negative controls for PhysX.

Canonical protocol: g0b_d420/t3y_workspace1_prereg.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "sim_scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "sim_scripts"))

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
P10_PATH = REPO / "sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py"
JAW_PATH = REPO / "sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py"
KINEMATICS_PATH = REPO / "sim_scripts/roarm_kinematics.py"
ROARM_RL_INIT_PATH = REPO / "roarm_rl/__init__.py"
STACK_ENV_PATH = REPO / "roarm_rl/roarm_stack_env.py"
VIZ_DEBUG_PATH = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT_PATH = REPO / "roarm_rl/rerun_contract.py"
P13_RESULTS = CASE_DIR / "t3x_bite81_results.json"
PREREG_PATH = CASE_DIR / "t3y_workspace1_prereg.md"
PREFLIGHT2_PREREG_PATH = CASE_DIR / "t3y_workspace_preflight2_prereg.md"
P13_RESULTS_SHA256 = "d1460c9d80e5f65f4ab9d85a7851b29876ef7ae0ca9e28d34bd93ddb91f0170a"
WORKSPACE1_PREREG_SHA256 = "2191aca517bc8744713889c6179afaf2e329d244525f227ec0ee9fd9d436c3ec"
PREFLIGHT2_PREREG_SHA256 = "5c3700cc39a7fa99c7640aa234522e81d8b60ce7b98eecf34dea89b03cea7c04"

LOG = "p14_t3y"
TAG = "t3y"
RUN_OUTPUT_NAMES = (
    "results.json", "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
    "rerun_validation.json", "inspection.png", "decision_snapshot.png",
    "script.py.txt", "argv.txt", "failure.json", "exit_status.txt",
    "phase.jsonl", "preclose_sentinel.json", "terminal_attestation.json",
)
P10_SHA256 = "63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff"
JAW_SHA256 = "bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3"
ATTEMPT3_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
ATTEMPT3_PHYSICS_SHA256 = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"
PINNED_LOCAL_SOURCES: dict[str, tuple[Path, str]] = {
    "p10": (P10_PATH, P10_SHA256),
    "jaw_extractor": (JAW_PATH, JAW_SHA256),
    "roarm_kinematics": (
        KINEMATICS_PATH,
        "af4cc2d3c124ba4a8a3a6899ab1c8d676e127199bcf687b8e31fee690a011a97",
    ),
    "roarm_rl_init": (
        ROARM_RL_INIT_PATH,
        "270819a1bba4aa43723ca2257cf7ed7da160e492dcb17607e71ee7cf1a08e940",
    ),
    "roarm_stack_env": (
        STACK_ENV_PATH,
        "726a57f4be83276fda1bb5b3eaf07a17f56d5c16cdbc0441bc14fb5c794a697d",
    ),
    "viz_debug": (
        VIZ_DEBUG_PATH,
        "4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0",
    ),
    "rerun_contract": (
        RERUN_CONTRACT_PATH,
        "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e",
    ),
}
ATTEMPT3_COMPOSED_LAYER_SHA256 = {
    "roarm_m3.usd": ATTEMPT3_ROOT_SHA256,
    "configuration/roarm_m3_base.usd": (
        "ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841"
    ),
    "configuration/roarm_m3_physics.usd": ATTEMPT3_PHYSICS_SHA256,
    "configuration/roarm_m3_robot.usd": (
        "2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d"
    ),
    "configuration/roarm_m3_sensor.usd": (
        "3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f"
    ),
}
ALLOWED_BUILTIN_MDL_IDENTIFIERS = frozenset({"OmniPBR.mdl"})
EXPECTED_OMNIPBR_MDL_SOURCE_ASSET_COUNT = 8
EXPECTED_THETA = (6.0, 15.0, 24.0, 35.0, 60.0, 69.0)
FORM_BY_THETA = {
    6.0: "near_top_down", 15.0: "near_top_down",
    24.0: "oblique", 35.0: "oblique",
    60.0: "high_tilt", 69.0: "high_tilt",
}
WINDOW_KINDS = {"bilateral", "unilateral_negative_control", "no_window"}
JAW_SUPPORT_REACTIVE_BASIS = {
    "seed0_S3": {"min_table_clearance_mm": -7.470639093428849,
                 "table_penetration_count": 6198},
    "seed0_S4": {"min_table_clearance_mm": -9.528697407419322,
                 "table_penetration_count": 8134},
    "r045_dpsi0": {"min_table_clearance_mm": -9.148326941431527,
                    "table_penetration_count": 7775},
}

NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
SCIPY_PIN = "1.15.3"
ISAACSIM_PIN = "5.1.0.0"
ISAACLAB_PIN = "2.3.0"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
HULL_VIEW_STRIDE = 16
HULL_VIEW_MAX_POINTS = 20_000

OBJ_DIAM_M = 0.029
OBJ_HEIGHT_M = 0.050
OBJ_MASS_KG = 0.02483
SUPPORT_Z_M = 0.0
GRAVITY = 9.81
LIFT_GATE_MM = 6.0
TIP_HALF_ANGLE_DEG = math.degrees(math.atan(OBJ_DIAM_M / OBJ_HEIGHT_M))
CONTACT_EPS_N = 1.0e-6
JAW_LOAD_GATE_N = 0.01
PRECLOSE_GATE_N = 0.02
JAW_SUPPORT_GATE_N = 0.02
SETTLE_SUPPORT_FZ_N = OBJ_MASS_KG * GRAVITY
SETTLE_SUPPORT_TOL = 0.35

_W: dict[str, Any] = {}
_UNCAUGHT_FAILURE_CONTEXT: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably commit directory-entry changes on the local POSIX filesystem."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _durable_write_bytes_x(path: Path, payload: bytes) -> None:
    """Exclusive, forward-only write followed by file and directory fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _durable_write_json_x(path: Path, payload: dict[str, Any]) -> None:
    _durable_write_bytes_x(
        path, (json.dumps(payload, indent=2, default=_jsonable) + "\n").encode("utf-8"))


def _durable_replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably create or update the one-run failure marker without a torn final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise RuntimeError(f"STALE_DURABLE_TEMP_PATH {temp}")
    try:
        _durable_write_json_x(temp, payload)
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()
            _fsync_directory(path.parent)


def _durable_append_phase(path: Path, phase: str, **fields: Any) -> None:
    record = {
        "phase": phase,
        "pid": os.getpid(),
        "unix_time_s": time.time(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    encoded = (json.dumps(record, sort_keys=True, default=_jsonable) + "\n").encode("utf-8")
    first_write = not path.exists()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, "ab", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    if first_write:
        _fsync_directory(path.parent)


def _artifact_manifest(paths: dict[str, Path], names: tuple[str, ...]) -> dict[str, Any]:
    """Fsync and hash every finalized pre-close artifact that actually exists."""
    missing = [name for name in names if not paths[name].is_file()]
    present: dict[str, dict[str, Any]] = {}
    for name in names:
        path = paths[name]
        if not path.is_file():
            continue
        _fsync_file(path)
        present[name] = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    _fsync_directory(CASE_DIR)
    return {
        "required_names": list(names),
        "missing_names": missing,
        "complete": not missing,
        "files": present,
    }


def _record_uncaught_failure(exc: BaseException) -> None:
    """Best-effort durable marker for failures after AppLauncher has started."""
    context = _UNCAUGHT_FAILURE_CONTEXT
    if context is None:
        return
    path = Path(context["failure_path"])
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
        except Exception:
            previous = {"previous_marker_unreadable": True}
    payload = {
        **previous,
        "tool": "p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep",
        "tag": context["tag"],
        "run_profile": context["run_profile"],
        # Preserve the more precise phase already written by main's local
        # exception handler.  The outer handler records what it observed without
        # erasing that primary provenance.
        "runtime_phase": previous.get(
            "runtime_phase", context.get("runtime_phase", "uncaught_top_level")),
        "top_level_observed_phase": context.get("runtime_phase", "uncaught_top_level"),
        "primary_exception_type": previous.get(
            "primary_exception_type", previous.get("exception_type", type(exc).__name__)),
        "primary_exception_message": previous.get(
            "primary_exception_message", previous.get("exception_message", str(exc))),
        "exception_type": previous.get(
            "primary_exception_type", previous.get("exception_type", type(exc).__name__)),
        "exception_message": previous.get(
            "primary_exception_message", previous.get("exception_message", str(exc))),
        "top_level_exception_type": type(exc).__name__,
        "top_level_exception_message": str(exc),
        "executed_source_sha256": context["executed_source_sha256"],
        "protocol_path": context["protocol_path"],
        "protocol_sha256": context["protocol_sha256"],
        "unix_time_s": time.time(),
        "effective_result_must_fail_even_if_terminal_close_yields_raw_exit_zero": True,
        "external_terminal_authority_required": True,
    }
    _durable_replace_json(path, payload)
    print(f"[{LOG}] FAILURE_MARKER_DURABLE phase={payload['runtime_phase']} "
          f"path={path} error={exc}", flush=True)


def _terminal_close_after_failure(primary_exc: BaseException) -> None:
    """Attempt graceful terminal cleanup after a durable failure marker.

    A normal installed close may terminate Python with raw status zero.  That does
    not erase the failure: the external contract rejects the durable marker.  If
    close returns or raises, this helper forces a nonzero Python path.
    """
    context = _UNCAUGHT_FAILURE_CONTEXT
    if context is None or context.get("terminal_close_call_entered"):
        return
    simulation_app = context.get("simulation_app")
    if simulation_app is None:
        return
    context["runtime_phase"] = "simulation_app_terminal_close_after_failure"
    phase_path = Path(context["phase_path"])
    try:
        _durable_append_phase(
            phase_path, "simulation_app_close_start",
            failure_path=True,
            failure_marker_path=str(Path(context["failure_path"]).relative_to(REPO)),
            expected_behavior="terminal_nonreturning_framework_release",
        )
    except BaseException:
        # The primary durable failure marker remains the authority.  A journal
        # failure must not skip the best-effort graceful terminal close.
        pass
    try:
        print(f"[{LOG}] SIMULATION_APP_CLOSE_START failure_path=true ",
              f"primary={type(primary_exc).__name__}: {primary_exc}", flush=True)
    except BaseException:
        pass
    context["terminal_close_call_entered"] = True
    try:
        simulation_app.close()
    except BaseException as close_exc:
        failure = RuntimeError(
            "SIMULATION_APP_TERMINAL_CLOSE_AFTER_FAILURE_RAISED "
            f"{type(close_exc).__name__}: {close_exc}")
        _record_uncaught_failure(failure)
        raise failure from close_exc
    failure = RuntimeError("SIMULATION_APP_TERMINAL_CLOSE_AFTER_FAILURE_UNEXPECTED_RETURN")
    _record_uncaught_failure(failure)
    raise failure from primary_exc


def _run_paths(prefix: str) -> dict[str, Path]:
    return {name: CASE_DIR / f"{prefix}_{name}" for name in RUN_OUTPUT_NAMES}


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT {path}")
    return value


def _linux_pgid_members(pgid: int) -> list[int]:
    """Read-only exact Linux process-group inventory for external attestation."""
    members: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text()
            closing = raw.rfind(")")
            fields = raw[closing + 2:].split()
            # tail fields begin at proc field 3: state, ppid, pgrp.
            if closing > 0 and len(fields) >= 3 and int(fields[2]) == pgid:
                members.append(int(stat_path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return sorted(members)


def _gpu_pid_set(csv_text: str) -> set[int]:
    pids: set[int] = set()
    for line in csv_text.splitlines():
        token = line.split(",", 1)[0].strip()
        if token:
            try:
                pids.add(int(token))
            except ValueError as exc:
                raise RuntimeError(f"NVIDIA_SMI_PID_PARSE_FAIL line={line!r}") from exc
    return pids


def _external_terminal_attest(run_label: str) -> int:
    """Offline, post-reap D367/D375 authority; never launches Isaac or PhysX."""
    if run_label not in {"workspace1", "workspace_preflight2"}:
        raise RuntimeError(f"UNREGISTERED_TERMINAL_ATTEST_LABEL {run_label!r}")
    prefix = f"{TAG}_{run_label}"
    paths = _run_paths(prefix)
    external = {
        "stdout": CASE_DIR / f"{prefix}_stdout.log",
        "supervisor_pid": CASE_DIR / f"{prefix}_supervisor_pid.txt",
        "python_pid": CASE_DIR / f"{prefix}_python_pid.txt",
        "pgid": CASE_DIR / f"{prefix}_pgid.txt",
        "supervisor_contract": CASE_DIR / f"{prefix}_supervisor_contract.json",
        "gpu_before": CASE_DIR / f"{prefix}_nvidia_smi_before.csv",
        "gpu_after": CASE_DIR / f"{prefix}_nvidia_smi_after.csv",
    }
    required = {
        "results": paths["results.json"],
        "sentinel": paths["preclose_sentinel.json"],
        "phase": paths["phase.jsonl"],
        "exit_status": paths["exit_status.txt"],
        **external,
    }
    # The attestor itself creates the fresh after-GPU inventory and terminal record.
    required_before_gpu_after = {
        name: path for name, path in required.items() if name != "gpu_after"}
    missing = [f"{name}:{path}" for name, path in required_before_gpu_after.items()
               if not path.is_file()]
    if missing:
        raise RuntimeError(f"TERMINAL_ATTEST_REQUIRED_ARTIFACT_MISSING {missing}")
    if paths["terminal_attestation.json"].exists():
        raise RuntimeError(
            f"TERMINAL_ATTEST_FORWARD_ONLY_EXISTS {paths['terminal_attestation.json']}")
    if external["gpu_after"].exists():
        raise RuntimeError(f"TERMINAL_ATTEST_GPU_AFTER_EXISTS {external['gpu_after']}")
    for path in required_before_gpu_after.values():
        _fsync_file(path)
    _fsync_directory(CASE_DIR)

    result = _read_json_object(paths["results.json"])
    sentinel = _read_json_object(paths["preclose_sentinel.json"])
    phase_bytes = paths["phase.jsonl"].read_bytes()
    phase_rows = [json.loads(line) for line in phase_bytes.decode("utf-8").splitlines()
                  if line.strip()]
    exit_status_text = paths["exit_status.txt"].read_text().strip()
    supervisor_pid = int(external["supervisor_pid"].read_text().strip())
    python_pid = int(external["python_pid"].read_text().strip())
    pgid = int(external["pgid"].read_text().strip())
    stdout_text = external["stdout"].read_text(errors="replace")
    supervisor_contract = _read_json_object(external["supervisor_contract"])

    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(
            f"TERMINAL_ATTEST_NVIDIA_SMI_FAIL rc={query.returncode} stderr={query.stderr!r}")
    _durable_write_bytes_x(external["gpu_after"], query.stdout.encode("utf-8"))
    before_gpu_pids = _gpu_pid_set(external["gpu_before"].read_text())
    after_gpu_pids = _gpu_pid_set(query.stdout)

    manifest = sentinel.get("artifact_manifest")
    manifest_files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
    manifest_file_checks: dict[str, bool] = {}
    for name, record in manifest_files.items():
        try:
            artifact_path = REPO / str(record["path"])
            manifest_file_checks[name] = bool(
                artifact_path.is_file()
                and artifact_path.stat().st_size == int(record["bytes"])
                and sha256_file(artifact_path) == record["sha256"])
        except (KeyError, TypeError, ValueError):
            manifest_file_checks[name] = False

    prefix_bytes = int(sentinel.get("phase_prefix_bytes", -1))
    prefix_valid_range = 0 <= prefix_bytes <= len(phase_bytes)
    prefix_hash = (hashlib.sha256(phase_bytes[:prefix_bytes]).hexdigest()
                   if prefix_valid_range else None)
    prefix_line_count = (len([line for line in phase_bytes[:prefix_bytes].splitlines()
                              if line.strip()]) if prefix_valid_range else -1)
    phase_prefix_rows = phase_rows[:prefix_line_count] if prefix_line_count >= 0 else []
    actual_sentinel_sha256 = sha256_file(paths["preclose_sentinel.json"])
    phase_binding_checks = _terminal_phase_binding_checks(
        phase_rows, prefix_line_count, sentinel, actual_sentinel_sha256,
        python_pid, pgid)
    warning_tokens = (
        "CLEANUP_ERROR", "ISAAC_ENV_CLOSE_OR_PRECLOSE_LIFECYCLE_FAILURE",
        "FAILURE_MARKER_WRITTEN", "FAILURE_MARKER_UPDATED",
        "FAILURE_MARKER_DURABLE", "SIMULATION_APP_TERMINAL_CLOSE_RAISED",
        "SIMULATION_APP_TERMINAL_CLOSE_UNEXPECTED_RETURN", "close warning",
    )
    result_manifest = result.get("preclose_artifact_manifest")
    checks = {
        "supervisor_contract_exact": supervisor_contract == {
            "artifact": "T3Y_EXTERNAL_TIMEOUT_SUPERVISOR_V1",
            "automatic_retry_count": 0,
            "foreground": True,
            "kill_after_seconds": 20,
            "preserve_status": False,
            "term_signal": "TERM",
            "timeout_seconds": 21600 if run_label == "workspace1" else 7200,
        },
        "exit_status_exact_zero_no_timeout_or_signal": exit_status_text == "0",
        "failure_marker_absent": not paths["failure.json"].exists(),
        "supervisor_pid_reaped": not Path(f"/proc/{supervisor_pid}").exists(),
        "python_pid_reaped": not Path(f"/proc/{python_pid}").exists(),
        "recorded_pgid_empty": _linux_pgid_members(pgid) == [],
        "python_pid_absent_from_fresh_gpu_inventory": python_pid not in after_gpu_pids,
        "fresh_gpu_pid_delta_empty": not (after_gpu_pids - before_gpu_pids),
        "stdout_no_failure_or_close_warning": not any(
            token in stdout_text for token in warning_tokens),
        "stdout_exactly_one_close_start": stdout_text.count(
            "SIMULATION_APP_CLOSE_START") == 1,
        "phase_prefix_range_valid": prefix_valid_range,
        "phase_prefix_sha_exact": prefix_hash == sentinel.get("phase_prefix_sha256"),
        "phase_prefix_exact_order_and_result_binding": bool(
            [row.get("phase") for row in phase_prefix_rows]
            == ["run_claim", "results_durable"]
            and phase_prefix_rows[-1].get("results_sha256")
            == sentinel.get("results_sha256")
            and phase_prefix_rows[0].get("executed_source_sha256")
            == sentinel.get("executed_source_sha256")
            and phase_prefix_rows[0].get("protocol_sha256")
            == sentinel.get("protocol_sha256")),
        "results_path_exact": sentinel.get("results_path")
        == str(paths["results.json"].relative_to(REPO)),
        "results_sha_exact": sentinel.get("results_sha256")
        == sha256_file(paths["results.json"]),
        "results_bytes_exact": sentinel.get("results_bytes")
        == paths["results.json"].stat().st_size,
        "sentinel_manifest_equals_results": manifest == result_manifest,
        "all_manifest_files_sha_and_size_exact": bool(
            manifest_file_checks and all(manifest_file_checks.values())),
        "executed_source_still_exact": result.get("provenance", {}).get(
            "executed_source_sha256") == sha256_file(Path(__file__)),
        "frozen_script_copy_exact": result.get("provenance", {}).get(
            "executed_source_sha256") == sha256_file(paths["script.py.txt"]),
        "protocol_sha_bound": sentinel.get("protocol_sha256")
        == result.get("provenance", {}).get("prereg_sha256"),
        "p13_sha_bound": sentinel.get("p13_results_sha256")
        == result.get("provenance", {}).get("p13_results_sha256"),
        "dependency_finalize_manifest_bound": bool(
            sentinel.get("dependency_hashes_equal_start") is True
            and sentinel.get("dependency_hashes_after_results")
            == result.get("provenance", {}).get("dependency_hashes_at_finalize")
            == result.get("provenance", {}).get("dependency_expected_sha256")),
        "env_close_internal_pass": sentinel.get("env_close_internal_pass") is True,
        "sentinel_safe_to_close_app": sentinel.get("safe_to_close_app") is True,
        "simulation_app_postreturn_not_claimed": bool(
            result.get("cleanup", {}).get("simulation_app", {}).get("close_attempted") is False
            and result.get("cleanup", {}).get("simulation_app", {}).get("close_returned") is None
            and result.get("cleanup", {}).get("pass") is False),
        "preclose_verdict_exact": result.get("internal_lifecycle_verdict")
        == "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL"
        and sentinel.get("internal_lifecycle_verdict")
        == "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL",
        "scientific_verdict_not_premature": result.get("scientific_verdict") is None,
        "visual_still_requires_manual_inspection": sentinel.get(
            "manual_visual_inspection_required_after_terminal_attestation") is True,
    }
    checks.update(phase_binding_checks)
    readiness_checks = {
        "artifact_manifest_complete": bool(
            isinstance(manifest, dict) and manifest.get("complete") is True
            and not manifest.get("missing_names")),
        "rerun_technical_contract": result.get("rerun", {}).get("technical_pass") is True,
    }
    if run_label == "workspace_preflight2":
        instrumentation = result.get("instrumentation_preflight") or {}
        readiness_checks["preflight_scope_non_authoritative"] = bool(
            result.get("scientific_authoritative") is False
            and instrumentation.get("authoritative_scientific") is False)
        readiness_checks["preflight_all_internal_checks"] = bool(
            instrumentation.get("checks")
            and all(instrumentation["checks"].values()))
        readiness_checks["preflight_internal_verdict_exact"] = (
            instrumentation.get("verdict")
            == "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL")

    terminal_lifecycle_pass = all(checks.values())
    internal_and_observability_previsual_pass = all(readiness_checks.values())
    passed = bool(terminal_lifecycle_pass and internal_and_observability_previsual_pass)
    attestation = {
        "artifact": "T3Y_D367_D375_EXTERNAL_TERMINAL_ATTESTATION_V1",
        "tag": prefix,
        "run_profile": result.get("run_profile"),
        "authority_scope": (
            "separate terminal lifecycle and internal/observability previsual gates; "
            "manual visual inspection still pending"),
        "terminal_lifecycle_checks": checks,
        "internal_and_observability_previsual_checks": readiness_checks,
        "manifest_file_checks": manifest_file_checks,
        "supervisor": {
            "pid": supervisor_pid, "python_pid": python_pid, "pgid": pgid,
            "exit_status": exit_status_text,
            "timeout_or_signal_inferred": exit_status_text != "0",
            "pgid_members_after_exit": _linux_pgid_members(pgid),
        },
        "gpu": {
            "before_path": str(external["gpu_before"].relative_to(REPO)),
            "before_sha256": sha256_file(external["gpu_before"]),
            "after_path": str(external["gpu_after"].relative_to(REPO)),
            "after_sha256": sha256_file(external["gpu_after"]),
            "before_pids": sorted(before_gpu_pids),
            "after_pids": sorted(after_gpu_pids),
            "new_pids": sorted(after_gpu_pids - before_gpu_pids),
        },
        "bindings": {
            "results_path": str(paths["results.json"].relative_to(REPO)),
            "results_sha256": sha256_file(paths["results.json"]),
            "preclose_sentinel_path": str(
                paths["preclose_sentinel.json"].relative_to(REPO)),
            "preclose_sentinel_sha256": sha256_file(paths["preclose_sentinel.json"]),
            "phase_path": str(paths["phase.jsonl"].relative_to(REPO)),
            "phase_sha256": sha256_file(paths["phase.jsonl"]),
            "stdout_path": str(external["stdout"].relative_to(REPO)),
            "stdout_sha256": sha256_file(external["stdout"]),
            "exit_status_path": str(paths["exit_status.txt"].relative_to(REPO)),
            "exit_status_sha256": sha256_file(paths["exit_status.txt"]),
            "supervisor_contract_path": str(
                external["supervisor_contract"].relative_to(REPO)),
            "supervisor_contract_sha256": sha256_file(external["supervisor_contract"]),
        },
        "terminal_lifecycle_pass": terminal_lifecycle_pass,
        "internal_and_observability_previsual_pass": (
            internal_and_observability_previsual_pass),
        "pass": passed,
        "verdict": (
            "TERMINAL_ATTESTED_PENDING_MANUAL_VISUAL"
            if passed else (
                "TERMINAL_ATTESTED_INTERNAL_OR_OBSERVABILITY_FAIL"
                if terminal_lifecycle_pass else "EXTERNAL_TERMINAL_ATTESTATION_FAIL")),
        "scientific_or_instrumentation_promotion": False,
    }
    _durable_write_json_x(paths["terminal_attestation.json"], attestation)
    print(f"[{LOG}] EXTERNAL_TERMINAL_ATTESTATION pass={passed} "
          f"path={paths['terminal_attestation.json']}", flush=True)
    return 0 if passed else 1


def _verify_pinned_local_sources() -> dict[str, dict[str, str]]:
    """Hard-pin every repo-local source imported on a decision-bearing path."""
    manifest: dict[str, dict[str, str]] = {}
    failures: dict[str, Any] = {}
    for name, (path, expected) in PINNED_LOCAL_SOURCES.items():
        if not path.is_file():
            failures[name] = {"path": str(path), "error": "missing"}
            continue
        actual = sha256_file(path)
        manifest[name] = {"path": str(path), "sha256": actual}
        if actual != expected:
            failures[name] = {
                "path": str(path), "expected_sha256": expected, "actual_sha256": actual,
            }
    if failures:
        raise RuntimeError(
            f"PINNED_LOCAL_SOURCE_DRIFT {json.dumps(failures, sort_keys=True)}")
    return manifest


def _hash_named_paths(paths: dict[str, Path]) -> dict[str, str]:
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"FROZEN_DEPENDENCY_MISSING {missing}")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _asset_identifier_record(value: Any) -> dict[str, str]:
    """Return a lossless display value and a deliberately narrow identifier.

    Sdf.AssetPath exposes the authored identifier through ``path``.  The fallback
    only removes USD's outer ``@...@`` display delimiters; it does not resolve,
    basename-normalize, case-fold, URL-decode, or otherwise broaden the allowlist.
    """
    raw = str(value).strip()
    path_field = str(getattr(value, "path", "") or "").strip()
    normalized = path_field
    if not normalized:
        normalized = raw[1:-1] if len(raw) >= 2 and raw[0] == "@" and raw[-1] == "@" else raw
    return {"raw": raw, "path_field": path_field, "normalized": normalized}


def _classify_unresolved_dependencies(values: Any) -> dict[str, Any]:
    """Classify USD dependency failures without treating a built-in MDL as a file.

    Only the exact NVIDIA built-in identifier observed in preflight1 is admissible.
    Missing USD composition, another MDL, a path-qualified OmniPBR name, or any
    unknown unresolved dependency remains fatal.
    """
    records: list[dict[str, str]] = []
    for value in values:
        record = _asset_identifier_record(value)
        normalized = record["normalized"]
        if normalized in ALLOWED_BUILTIN_MDL_IDENTIFIERS:
            classification = "allowed_builtin_mdl_identifier"
        elif normalized.lower().endswith((".usd", ".usda", ".usdc")):
            classification = "fatal_unresolved_usd_composition"
        elif normalized.lower().endswith(".mdl"):
            classification = "fatal_unregistered_mdl_identifier"
        else:
            classification = "fatal_unresolved_other"
        records.append({**record, "classification": classification})
    records.sort(key=lambda row: (row["normalized"], row["raw"]))
    normalized_unique = sorted({row["normalized"] for row in records})
    fatal = [row for row in records if not row["classification"].startswith("allowed_")]
    allowed = [row for row in records if row["classification"].startswith("allowed_")]
    exact_allowed_set_pass = (
        set(normalized_unique) == set(ALLOWED_BUILTIN_MDL_IDENTIFIERS) and not fatal)
    return {
        "raw_records": records,
        "normalized_unique": normalized_unique,
        "expected_exact_allowed_set": sorted(ALLOWED_BUILTIN_MDL_IDENTIFIERS),
        "allowed_records": allowed,
        "fatal_records": fatal,
        "exact_allowed_set_pass": bool(exact_allowed_set_pass),
    }


def _mdl_unresolved_classification_regression_smoke() -> dict[str, Any]:
    """Pure regression: the MDL exception must never mask a missing USD."""
    allowed = _classify_unresolved_dependencies(["@OmniPBR.mdl@"])
    other_mdl = _classify_unresolved_dependencies(["OtherPBR.mdl"])
    path_qualified = _classify_unresolved_dependencies(["materials/OmniPBR.mdl"])
    missing_usd = _classify_unresolved_dependencies(["configuration/missing.usd"])
    mixed_missing_usd = _classify_unresolved_dependencies(
        ["OmniPBR.mdl", "configuration/missing.usdc"])
    mixed_other_mdl = _classify_unresolved_dependencies(
        ["OmniPBR.mdl", "OtherPBR.mdl"])
    passed = bool(
        allowed["exact_allowed_set_pass"]
        and not other_mdl["exact_allowed_set_pass"] and len(other_mdl["fatal_records"]) == 1
        and not path_qualified["exact_allowed_set_pass"]
        and not missing_usd["exact_allowed_set_pass"]
        and not mixed_missing_usd["exact_allowed_set_pass"]
        and any(row["classification"] == "fatal_unresolved_usd_composition"
                for row in mixed_missing_usd["fatal_records"])
        and not mixed_other_mdl["exact_allowed_set_pass"]
        and any(row["classification"] == "fatal_unregistered_mdl_identifier"
                for row in mixed_other_mdl["fatal_records"])
    )
    if not passed:
        raise RuntimeError("BUILTIN_MDL_UNRESOLVED_CLASSIFICATION_REGRESSION_FAIL")
    return {
        "pass": True,
        "allowed_exact_identifier": allowed["normalized_unique"],
        "other_mdl_fatal": other_mdl["fatal_records"],
        "path_qualified_same_basename_fatal": path_qualified["fatal_records"],
        "missing_usd_fatal": missing_usd["fatal_records"],
        "mixed_allowed_plus_missing_usd_fatal": mixed_missing_usd["fatal_records"],
        "mixed_allowed_plus_other_mdl_fatal": mixed_other_mdl["fatal_records"],
    }


def _preclose_lifecycle_contract_pass(status: dict[str, Any]) -> bool:
    """Internal authority ends at env close; terminal authority is external.

    On the installed Isaac Sim 5.1 stack ``SimulationApp.close()`` normally ends
    the Python process in framework release.  Requiring a post-return marker is
    therefore a category error (D367/D375).  A result may only attest that the
    environment closed and the terminal app-close call is the next normal action.
    """
    env_status = status["env"]
    app_status = status["simulation_app"]
    return bool(
        not status["exceptions"]
        and env_status["created"]
        and env_status["close_attempted"]
        and env_status["close_returned"]
        and app_status["created"]
        and not app_status["close_attempted"]
        and app_status["close_returned"] is None
        and status["terminal_completion"] == "PENDING_EXTERNAL_ATTESTATION")


def _preclose_lifecycle_contract_regression_smoke() -> dict[str, Any]:
    base = {
        "env": {"created": True, "close_attempted": True, "close_returned": True},
        "simulation_app": {
            "created": True, "close_attempted": False, "close_returned": None},
        "exceptions": [],
        "terminal_completion": "PENDING_EXTERNAL_ATTESTATION",
    }
    success = _preclose_lifecycle_contract_pass(base)
    env_raised = json.loads(json.dumps(base))
    env_raised["env"]["close_returned"] = False
    env_raised["exceptions"] = [{"component": "env"}]
    false_postreturn = json.loads(json.dumps(base))
    false_postreturn["simulation_app"]["close_attempted"] = True
    false_postreturn["simulation_app"]["close_returned"] = True
    app_missing = json.loads(json.dumps(base))
    app_missing["simulation_app"]["created"] = False
    terminal_claimed = json.loads(json.dumps(base))
    terminal_claimed["terminal_completion"] = "PASS"
    passed = bool(
        success and not _preclose_lifecycle_contract_pass(env_raised)
        and not _preclose_lifecycle_contract_pass(false_postreturn)
        and not _preclose_lifecycle_contract_pass(app_missing)
        and not _preclose_lifecycle_contract_pass(terminal_claimed))
    if not passed:
        raise RuntimeError("ISAAC_PRECLOSE_LIFECYCLE_CONTRACT_REGRESSION_FAIL")
    return {
        "pass": True, "env_closed_app_terminal_pending": success,
        "env_close_exception_rejected": not _preclose_lifecycle_contract_pass(env_raised),
        "false_postreturn_completion_rejected": not _preclose_lifecycle_contract_pass(
            false_postreturn),
        "missing_simulation_app_rejected": not _preclose_lifecycle_contract_pass(app_missing),
        "internal_terminal_pass_rejected": not _preclose_lifecycle_contract_pass(
            terminal_claimed),
    }


def _preclose_hash_binding_regression_smoke() -> dict[str, Any]:
    """Pure negative control for result/sentinel and artifact-manifest authority."""
    result_sha = "a" * 64
    artifact_manifest = {
        "complete": True,
        "files": {"trace.npz": {"sha256": "b" * 64, "bytes": 123}},
    }

    def gate(sentinel: dict[str, Any], observed_result_sha: str,
             observed_manifest: dict[str, Any]) -> bool:
        return bool(
            sentinel.get("results_sha256") == observed_result_sha
            and sentinel.get("artifact_manifest") == observed_manifest
            and sentinel.get("safe_to_close_app") is True
            and sentinel.get("internal_lifecycle_verdict")
            == "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL")

    base = {
        "results_sha256": result_sha,
        "artifact_manifest": artifact_manifest,
        "safe_to_close_app": True,
        "internal_lifecycle_verdict": "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL",
    }
    mutated_sha = dict(base, results_sha256="c" * 64)
    mutated_manifest = json.loads(json.dumps(base))
    mutated_manifest["artifact_manifest"]["files"]["trace.npz"]["bytes"] = 124
    false_terminal = dict(base, internal_lifecycle_verdict="TERMINAL_PASS")
    passed = bool(
        gate(base, result_sha, artifact_manifest)
        and not gate(mutated_sha, result_sha, artifact_manifest)
        and not gate(mutated_manifest, result_sha, artifact_manifest)
        and not gate(false_terminal, result_sha, artifact_manifest))
    if not passed:
        raise RuntimeError("PRECLOSE_HASH_BINDING_REGRESSION_FAIL")
    return {
        "pass": True,
        "exact_binding_accepted": True,
        "mutated_result_sha_rejected": True,
        "mutated_artifact_manifest_rejected": True,
        "premature_terminal_pass_rejected": True,
    }


def _terminal_phase_binding_checks(
    phase_rows: list[dict[str, Any]], prefix_line_count: int,
    sentinel: dict[str, Any], actual_sentinel_sha256: str,
    python_pid: int, pgid: int,
) -> dict[str, bool]:
    suffix = phase_rows[prefix_line_count:] if prefix_line_count >= 0 else []
    close_rows = [row for row in phase_rows
                  if row.get("phase") == "simulation_app_close_start"]
    sentinel_rows = [row for row in phase_rows
                     if row.get("phase") == "preclose_sentinel_durable"]
    return {
        "phase_exact_suffix_order": [row.get("phase") for row in suffix]
        == ["preclose_sentinel_durable", "simulation_app_close_start"],
        "phase_exactly_one_sentinel_durable": len(sentinel_rows) == 1,
        "phase_sentinel_row_sha_exact": bool(
            sentinel_rows
            and sentinel_rows[0].get("preclose_sentinel_sha256")
            == actual_sentinel_sha256),
        "phase_exactly_one_close_start": len(close_rows) == 1,
        "phase_close_start_bound_to_exact_sentinel": bool(
            close_rows
            and close_rows[0].get("pid") == python_pid
            and close_rows[0].get("preclose_sentinel_sha256")
            == actual_sentinel_sha256
            and close_rows[0].get("expected_behavior")
            == "terminal_nonreturning_framework_release"
            and close_rows[0].get("failure_path") is not True),
        "sentinel_process_identity_exact": bool(
            sentinel.get("pid") == python_pid and sentinel.get("pgid") == pgid),
    }


def _terminal_phase_binding_regression_smoke() -> dict[str, Any]:
    sha = "d" * 64
    pid, pgid = 123, 120
    sentinel = {"pid": pid, "pgid": pgid}
    prefix = [{"phase": "run_claim"}, {"phase": "results_durable"}]
    suffix = [
        {"phase": "preclose_sentinel_durable", "preclose_sentinel_sha256": sha},
        {"phase": "simulation_app_close_start", "pid": pid,
         "preclose_sentinel_sha256": sha,
         "expected_behavior": "terminal_nonreturning_framework_release"},
    ]
    exact = _terminal_phase_binding_checks(
        prefix + suffix, len(prefix), sentinel, sha, pid, pgid)
    wrong_sha_rows = json.loads(json.dumps(prefix + suffix))
    wrong_sha_rows[-1]["preclose_sentinel_sha256"] = "e" * 64
    wrong_sha = _terminal_phase_binding_checks(
        wrong_sha_rows, len(prefix), sentinel, sha, pid, pgid)
    swapped = _terminal_phase_binding_checks(
        prefix + list(reversed(suffix)), len(prefix), sentinel, sha, pid, pgid)
    wrong_pid = _terminal_phase_binding_checks(
        prefix + suffix, len(prefix), sentinel, sha, pid + 1, pgid)
    failure_row = json.loads(json.dumps(prefix + suffix))
    failure_row[-1]["failure_path"] = True
    failure_path = _terminal_phase_binding_checks(
        failure_row, len(prefix), sentinel, sha, pid, pgid)
    passed = bool(
        all(exact.values())
        and not all(wrong_sha.values())
        and not all(swapped.values())
        and not all(wrong_pid.values())
        and not all(failure_path.values()))
    if not passed:
        raise RuntimeError("TERMINAL_PHASE_BINDING_REGRESSION_FAIL")
    return {
        "pass": True,
        "exact_binding_accepted": True,
        "mutated_sha_rejected": True,
        "swapped_suffix_rejected": True,
        "wrong_python_pid_rejected": True,
        "failure_path_close_rejected": True,
    }


def _same_step_bilateral_summary(fixed_force_n: Any, moving_force_n: Any,
                                 gate_n: float = JAW_LOAD_GATE_N) -> dict[str, Any]:
    """Pure reference semantics for the two-jaw contact authority.

    The authoritative phase scalar is max_t(min(F_fixed(t), F_moving(t))).
    Separate per-jaw phase maxima remain useful diagnostics, but cannot establish
    that both jaws loaded the object in the same PhysX step.
    """
    fixed = np.asarray(fixed_force_n, dtype=np.float64)
    moving = np.asarray(moving_force_n, dtype=np.float64)
    if fixed.ndim != 1 or moving.ndim != 1 or fixed.shape != moving.shape or fixed.size == 0:
        raise ValueError("fixed/moving force histories must be equal, non-empty 1-D arrays")
    if not np.isfinite(fixed).all() or not np.isfinite(moving).all() or not math.isfinite(gate_n):
        raise ValueError("force histories and gate must be finite")
    authority = float(np.max(np.minimum(fixed, moving)))
    return {
        "authority": "max_over_phase(min(fixed_force_n[t],moving_force_n[t]))",
        "authority_n": authority,
        "gate_n_strict_gt": float(gate_n),
        "simultaneous_both": bool(authority > gate_n),
        "fixed_independent_max_n_diagnostic": float(np.max(fixed)),
        "moving_independent_max_n_diagnostic": float(np.max(moving)),
    }


def _simultaneous_contact_regression_smoke() -> dict[str, Any]:
    """Fail if staggered contacts can regress into a two-jaw-positive label."""
    staggered = _same_step_bilateral_summary([0.020, 0.000], [0.000, 0.030])
    simultaneous = _same_step_bilateral_summary([0.000, 0.020], [0.000, 0.030])
    if staggered["simultaneous_both"]:
        raise RuntimeError("STAGGERED_CONTACT_FALSE_POSITIVE_REGRESSION")
    if not simultaneous["simultaneous_both"]:
        raise RuntimeError("SIMULTANEOUS_CONTACT_FALSE_NEGATIVE_REGRESSION")
    if not (staggered["fixed_independent_max_n_diagnostic"] > JAW_LOAD_GATE_N
            and staggered["moving_independent_max_n_diagnostic"] > JAW_LOAD_GATE_N):
        raise RuntimeError("STAGGERED_REGRESSION_VECTOR_NO_LONGER_TESTS_OLD_MAX_AND_BUG")
    return {"pass": True, "staggered_expected_false": staggered,
            "simultaneous_expected_true": simultaneous}


def _population_validity_summary(success: Any, measurement_valid: Any,
                                 batch_positive_control: Any) -> dict[str, Any]:
    """Pure reference semantics for denominators and the global workspace claim."""
    success_arr = np.asarray(success, dtype=bool)
    valid_arr = np.asarray(measurement_valid, dtype=bool)
    batch_arr = np.asarray(batch_positive_control, dtype=bool)
    if success_arr.ndim != 1 or valid_arr.ndim != 1 or success_arr.shape != valid_arr.shape:
        raise ValueError("success/measurement_valid must be equal 1-D arrays")
    if success_arr.size == 0 or batch_arr.ndim != 1 or batch_arr.size == 0:
        raise ValueError("population and batch arrays must be non-empty")
    n_valid = int(valid_arr.sum())
    valid_success = int((success_arr & valid_arr).sum())
    return {
        "all_batches_positive_control": bool(batch_arr.all()),
        "all_trials_measurement_valid": bool(valid_arr.all()),
        "global_workspace_claim_allowed": bool(batch_arr.all() and valid_arr.all()),
        "n": int(valid_arr.size), "n_valid": n_valid,
        "n_invalid": int(valid_arr.size - n_valid),
        "valid_success": valid_success,
        "valid_success_rate": None if n_valid == 0 else float(valid_success / n_valid),
    }


def _partial_invalid_population_regression_smoke() -> dict[str, Any]:
    partial_trial = _population_validity_summary([True, False], [True, False], [True])
    partial_batch = _population_validity_summary([True, False], [True, True], [True, False])
    complete = _population_validity_summary([True, False], [True, True], [True, True])
    if partial_trial["global_workspace_claim_allowed"] or partial_trial["valid_success_rate"] != 1.0:
        raise RuntimeError("PARTIAL_INVALID_POPULATION_DENOMINATOR_REGRESSION")
    if partial_batch["global_workspace_claim_allowed"]:
        raise RuntimeError("PARTIAL_INVALID_BATCH_GLOBAL_CLAIM_REGRESSION")
    if not complete["global_workspace_claim_allowed"] or complete["valid_success_rate"] != 0.5:
        raise RuntimeError("COMPLETE_POPULATION_VALIDITY_REGRESSION")
    return {"pass": True,
            "partial_trial_invalid_expected_global_false": partial_trial,
            "partial_batch_invalid_expected_global_false": partial_batch,
            "complete_expected_global_true": complete}


def _effective_window_kind(reported_kind: str, q_margin: Any) -> str:
    """A missing exact-q5 depth is always a labelled fallback, regardless of source."""
    return ("no_depth_solution_fallback_negative_control"
            if q_margin is None else str(reported_kind))


def _fallback_label_regression_smoke() -> dict[str, Any]:
    fallback = "no_depth_solution_fallback_negative_control"
    cases = {
        "interior_target_none": _effective_window_kind("bilateral", None),
        "curve_negative_none": _effective_window_kind("no_bite_negative_control", None),
        "finite_preserves_kind": _effective_window_kind("bilateral", -0.001),
    }
    if cases["interior_target_none"] != fallback or cases["curve_negative_none"] != fallback:
        raise RuntimeError("NONE_Q_MARGIN_FALLBACK_LABEL_REGRESSION")
    if cases["finite_preserves_kind"] != "bilateral":
        raise RuntimeError("FINITE_Q_MARGIN_KIND_REGRESSION")
    return {"pass": True, "cases": cases}


def _workspace_verdict(global_population_valid: bool, replay_valid: bool,
                       replay_gate_class_equal: bool, success: int,
                       task_clear_both_close: int, task_clear_both_lift: int,
                       support_fail_with_bilateral: int,
                       support_fail_without_free_bilateral: int) -> str:
    if not global_population_valid:
        return "MEASUREMENT_INVALID_PARTIAL_BATCHES"
    if not replay_valid:
        return "MEASUREMENT_INVALID_REPRESENTATIVE_REPLAY_POSITIVE_CONTROL_FAIL"
    if not replay_gate_class_equal:
        return "REPLAY_GATE_CLASS_NONREPRODUCIBLE"
    if success > 0:
        return "PHYSICS_LIFT_SUCCESS_OBSERVED"
    if task_clear_both_close > 0:
        return "BOTH_JAWS_CONTACT_BUT_NO_VALID_LIFT"
    if task_clear_both_lift > 0:
        return "BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP"
    if support_fail_with_bilateral > 0:
        return "BILATERAL_CONTACT_COOCCURS_WITH_JAW_SUPPORT_COLLISION_NO_VALID_GRASP"
    if support_fail_without_free_bilateral > 0:
        return "JAW_SUPPORT_COLLISION_OBSERVED_NO_FREE_BILATERAL_GRASP"
    return "NO_SIMULTANEOUS_TWO_JAW_CONTACT_DURING_CLOSE_OR_LIFT_IN_SAMPLED_WORKSPACE"


def _workspace_verdict_regression_smoke() -> dict[str, Any]:
    common = (True, True, True)
    cases = {
        "close_only": _workspace_verdict(*common, 0, 1, 0, 0, 0),
        "lift_only": _workspace_verdict(*common, 0, 0, 1, 0, 0),
        "support_with_bilateral": _workspace_verdict(*common, 0, 0, 0, 1, 1),
        "support_without_bilateral": _workspace_verdict(*common, 0, 0, 0, 0, 1),
        "neither": _workspace_verdict(*common, 0, 0, 0, 0, 0),
    }
    if cases["close_only"] != "BOTH_JAWS_CONTACT_BUT_NO_VALID_LIFT":
        raise RuntimeError("CLOSE_CONTACT_VERDICT_REGRESSION")
    if cases["lift_only"] != "BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP":
        raise RuntimeError("LIFT_ONLY_CONTACT_OVERCLAIM_REGRESSION")
    if cases["support_with_bilateral"] != (
            "BILATERAL_CONTACT_COOCCURS_WITH_JAW_SUPPORT_COLLISION_NO_VALID_GRASP"):
        raise RuntimeError("SUPPORT_ASSISTED_BILATERAL_OVERCLAIM_REGRESSION")
    if cases["support_without_bilateral"] != (
            "JAW_SUPPORT_COLLISION_OBSERVED_NO_FREE_BILATERAL_GRASP"):
        raise RuntimeError("SUPPORT_COLLISION_NO_FREE_BILATERAL_REGRESSION")
    if cases["neither"] != (
            "NO_SIMULTANEOUS_TWO_JAW_CONTACT_DURING_CLOSE_OR_LIFT_IN_SAMPLED_WORKSPACE"):
        raise RuntimeError("NO_CONTACT_PHASE_SCOPE_REGRESSION")
    return {"pass": True, "cases": cases}


def _jaw_support_gate_regression_smoke() -> dict[str, Any]:
    cases = {
        "below": bool(max(0.019, 0.018) <= JAW_SUPPORT_GATE_N),
        "at_gate": bool(max(0.020, 0.020) <= JAW_SUPPORT_GATE_N),
        "above": bool(max(0.020001, 0.0) <= JAW_SUPPORT_GATE_N),
    }
    if cases != {"below": True, "at_gate": True, "above": False}:
        raise RuntimeError(f"JAW_SUPPORT_GATE_SEMANTICS_REGRESSION {cases}")
    # A reliable support collision is a measured task failure, not a sensor
    # failure: it remains in the valid denominator but can never be success.
    measured_valid = True
    success = bool(measured_valid and cases["above"])
    if not measured_valid or success:
        raise RuntimeError("JAW_SUPPORT_TASK_FAILURE_VALIDITY_REGRESSION")
    return {
        "pass": True, "gate_n_lte": JAW_SUPPORT_GATE_N, "cases": cases,
        "above_gate_measurement_valid": measured_valid,
        "above_gate_success": success,
        "above_gate_primary_label": "JAW_SUPPORT_CONTACT_FAIL",
    }


def _import_p10() -> Any:
    actual = sha256_file(P10_PATH)
    if actual != P10_SHA256:
        raise RuntimeError(f"P10_SHA256_MISMATCH expected={P10_SHA256} actual={actual}")
    spec = importlib.util.spec_from_file_location("p10_frozen_for_p14", P10_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen planner: {P10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _import_jaw_extractor_after_kit() -> Any:
    """Load the SHA-pinned read-only attempt3 extractor.

    That historical module bootstraps pxr by re-exec when run standalone.  p14
    calls it only after AppLauncher has loaded Kit/pxr, so the bootstrap sentinel
    is scoped around import to prevent an unintended process replacement.
    """
    actual = sha256_file(JAW_PATH)
    if actual != JAW_SHA256:
        raise RuntimeError(f"JAW_EXTRACTOR_SHA256_MISMATCH expected={JAW_SHA256} actual={actual}")
    flag = "G0B_JAW_AUDIT_REEXEC"
    previous = os.environ.get(flag)
    os.environ[flag] = "1"
    try:
        spec = importlib.util.spec_from_file_location("p14_pinned_jaw_extractor", JAW_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import jaw extractor: {JAW_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous
    return module


def extract_jaw_view_clouds(jaw: Any) -> dict[str, Any]:
    asset = jaw.extract_asset()
    bodies = asset["bodies"]
    report: dict[str, Any] = {
        "extractor_path": str(JAW_PATH), "extractor_sha256": JAW_SHA256,
        "surface_sample_spacing_m": float(jaw.SAMPLE_SPACING_M),
        "view_stride": HULL_VIEW_STRIDE, "view_max_points_per_body": HULL_VIEW_MAX_POINTS,
        "bodies": {},
    }
    clouds: dict[str, np.ndarray] = {}
    for body in ("link5", "gripper_link"):
        data = bodies[body]
        if len(data["parts"]) != 64 or data["approx_bad"]:
            raise RuntimeError(f"JAW_EXTRACTOR_64_PART_CONVEX_FAIL body={body}")
        if len(data["legacy"]) != 1 or data["legacy"][0][1] is not False:
            raise RuntimeError(f"JAW_EXTRACTOR_LEGACY_FAIL body={body} legacy={data['legacy']}")
        fallback_parts = sum(not bool(part["hull_ok"]) for part in data["parts"])
        if fallback_parts != 0:
            raise RuntimeError(
                f"JAW_HULL_SURFACE_SAMPLING_FALLBACK_FORBIDDEN "
                f"body={body} fallback_parts={fallback_parts}/64")
        dense = np.vstack([part["samples"] for part in data["parts"]]).astype(np.float32)
        view = dense[::HULL_VIEW_STRIDE]
        capped = len(view) > HULL_VIEW_MAX_POINTS
        if capped:
            keep = np.linspace(0, len(view)-1, HULL_VIEW_MAX_POINTS, dtype=np.int64)
            view = view[keep]
        clouds[body] = view
        report["bodies"][body] = {
            "part_count": len(data["parts"]), "dense_surface_samples": len(dense),
            "view_points": len(view), "view_cap_applied": capped,
            "hull_sample_fallback_parts": fallback_parts,
            "all_64_parts_hull_surface_sampled": True,
            "coordinate_frame": f"{body} body-local",
        }
    report["clouds"] = clouds
    return report


def _version_gate() -> dict[str, str]:
    import psutil
    import rerun as rr
    import scipy

    versions = {
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "scipy": scipy.__version__,
        "rerun-sdk": rr.__version__,
        "isaacsim": importlib.metadata.version("isaacsim"),
        "isaaclab": importlib.metadata.version("isaaclab"),
        "python": sys.version.split()[0],
    }
    expected = {
        "numpy": NUMPY_PIN, "psutil": PSUTIL_PIN, "scipy": SCIPY_PIN,
        "rerun-sdk": RERUN_VERSION,
        "isaacsim": ISAACSIM_PIN, "isaaclab": ISAACLAB_PIN,
    }
    bad = {k: {"expected": v, "actual": versions[k]}
           for k, v in expected.items() if versions[k] != v}
    if bad:
        raise RuntimeError(f"PACKAGE_PIN_MISMATCH {json.dumps(bad, sort_keys=True)}")
    return versions


def _asset_gate(p10: Any) -> dict[str, Any]:
    if p10.ATTEMPT3_ROOT_SHA256 != ATTEMPT3_ROOT_SHA256:
        raise RuntimeError("p10 attempt3 root identity constant drifted")
    if p10.ATTEMPT3_PHYSICS_SHA256 != ATTEMPT3_PHYSICS_SHA256:
        raise RuntimeError("p10 attempt3 physics identity constant drifted")
    root_path = Path(p10.ATTEMPT3_USD).resolve()
    asset_root = root_path.parent
    layer_paths = {
        f"attempt3_usd:{relative}": asset_root / relative
        for relative in ATTEMPT3_COMPOSED_LAYER_SHA256
    }
    actual = _hash_named_paths(layer_paths)
    expected = {
        f"attempt3_usd:{relative}": digest
        for relative, digest in ATTEMPT3_COMPOSED_LAYER_SHA256.items()
    }
    if actual != expected:
        raise RuntimeError(
            "ATTEMPT3_COMPOSED_LAYER_SHA256_MISMATCH "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"actual={json.dumps(actual, sort_keys=True)}")
    physics_path = asset_root / "configuration/roarm_m3_physics.usd"
    if Path(p10.ATTEMPT3_PHYSICS_LAYER).resolve() != physics_path.resolve():
        raise RuntimeError(
            "p10 attempt3 physics path drifted "
            f"p10={p10.ATTEMPT3_PHYSICS_LAYER} expected={physics_path}")
    return {
        "root_path": str(root_path), "root_sha256": actual["attempt3_usd:roarm_m3.usd"],
        "physics_path": str(physics_path),
        "physics_sha256": actual["attempt3_usd:configuration/roarm_m3_physics.usd"],
        "asset_root": str(asset_root),
        "expected_recursive_composition_layers": [
            {"relative_path": relative, "path": str(asset_root / relative),
             "sha256": ATTEMPT3_COMPOSED_LAYER_SHA256[relative]}
            for relative in sorted(ATTEMPT3_COMPOSED_LAYER_SHA256)
        ],
        "frozen_layer_hashes": actual,
    }


def _resolve_attempt3_composition_manifest(asset: dict[str, Any]) -> dict[str, Any]:
    """Resolve sublayers/references/payloads with USD itself and require an exact set.

    Hashing known filenames alone is insufficient: a root edit could silently change
    which ignored layer is composed.  This resolver runs after Kit loads pxr but before
    the stage/environment (and therefore before the first physics step) is created.
    """
    import omni.kit.app
    from pxr import Ar, Sdf, Tf, Usd, UsdShade, UsdUtils

    root_path = Path(asset["root_path"]).resolve()
    asset_root = Path(asset["asset_root"]).resolve()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(root_path)))

    unresolved_classification = _classify_unresolved_dependencies(unresolved)
    if not unresolved_classification["exact_allowed_set_pass"]:
        raise RuntimeError(
            "ATTEMPT3_USD_UNRESOLVED_DEPENDENCY_CLASSIFICATION_FAIL "
            f"{json.dumps(unresolved_classification, sort_keys=True)}")

    builtin_paths_raw = str(Tf.GetEnvSetting("OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"))
    builtin_paths = sorted({value.strip() for value in builtin_paths_raw.split(",")
                            if value.strip()})
    builtin_paths_os_raw = os.environ.get("OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS", "")
    builtin_paths_os = sorted({value.strip() for value in builtin_paths_os_raw.split(",")
                               if value.strip()})
    builtin_bypass_raw = os.environ.get("OMNI_USD_RESOLVER_MDL_BUILTIN_BYPASS", "")
    if not ALLOWED_BUILTIN_MDL_IDENTIFIERS.issubset(set(builtin_paths)):
        raise RuntimeError(
            "ATTEMPT3_BUILTIN_MDL_RUNTIME_MEMBERSHIP_FAIL "
            f"required={sorted(ALLOWED_BUILTIN_MDL_IDENTIFIERS)}")
    if builtin_bypass_raw != "1":
        raise RuntimeError(
            "ATTEMPT3_BUILTIN_MDL_BYPASS_CONFIG_NOT_EXACT_ONE "
            f"actual={builtin_bypass_raw!r}")
    if builtin_paths != builtin_paths_os:
        raise RuntimeError(
            "ATTEMPT3_BUILTIN_MDL_TF_OS_ENV_DISAGREE "
            f"tf_count={len(builtin_paths)} os_count={len(builtin_paths_os)}")
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    omni_usd_extension_id = extension_manager.get_enabled_extension_id("omni.usd")
    omni_usd_config_extension_id = extension_manager.get_enabled_extension_id("omni.usd.config")
    omni_usd_libs_extension_id = extension_manager.get_enabled_extension_id("omni.usd.libs")
    omni_usd_resolver_extension_id = extension_manager.get_enabled_extension_id(
        "omni.usd_resolver")
    if not all((omni_usd_extension_id, omni_usd_config_extension_id,
                omni_usd_libs_extension_id, omni_usd_resolver_extension_id)):
        raise RuntimeError(
            "ATTEMPT3_USD_RESOLVER_EXTENSION_ID_MISSING "
            f"omni.usd={omni_usd_extension_id!r} "
            f"omni.usd.config={omni_usd_config_extension_id!r} "
            f"omni.usd.libs={omni_usd_libs_extension_id!r} "
            f"omni.usd_resolver={omni_usd_resolver_extension_id!r}")
    extension_ids = {
        "omni.usd": str(omni_usd_extension_id),
        "omni.usd.config": str(omni_usd_config_extension_id),
        "omni.usd.libs": str(omni_usd_libs_extension_id),
        "omni.usd_resolver": str(omni_usd_resolver_extension_id),
    }
    extension_versions = {
        name: str(extension_manager.get_extension_dict(extension_id)["package"]["version"])
        for name, extension_id in extension_ids.items()
    }
    expected_extension_versions = {
        "omni.usd": "1.13.10", "omni.usd.config": "1.0.6",
        "omni.usd.libs": "1.0.1", "omni.usd_resolver": "1.0.0",
    }
    if extension_versions != expected_extension_versions:
        raise RuntimeError(
            "ATTEMPT3_USD_RESOLVER_EXTENSION_VERSION_DRIFT "
            f"expected={expected_extension_versions} actual={extension_versions}")
    resolver = Ar.GetResolver()
    usd_version = tuple(int(value) for value in Usd.GetVersion())
    if usd_version != (0, 24, 5):
        raise RuntimeError(
            f"ATTEMPT3_OPENUSD_VERSION_DRIFT expected=(0,24,5) actual={usd_version}")
    runtime_builtin_mdl = {
        "environment_variable": "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS",
        "required_identifiers": sorted(ALLOWED_BUILTIN_MDL_IDENTIFIERS),
        "required_identifiers_present": True,
        "runtime_identifier_count": len(builtin_paths),
        "runtime_identifiers_sha256": hashlib.sha256(
            "\n".join(builtin_paths).encode("utf-8")).hexdigest(),
        "builtin_bypass_config_environment_variable": (
            "OMNI_USD_RESOLVER_MDL_BUILTIN_BYPASS"),
        "builtin_bypass_config_raw": builtin_bypass_raw,
        "builtin_bypass_config_exact_one": True,
        "builtin_bypass_semantic_note": (
            "recorded exactly as installed omni.usd.config sets it; no claim is made "
            "that the name's truthiness means enable rather than disable"),
        "tf_and_os_identifier_sets_equal": True,
        "usd_version": list(usd_version),
        "expected_usd_version": [0, 24, 5],
        "ar_resolver_type": f"{type(resolver).__module__}.{type(resolver).__name__}",
        "omni_usd_extension_id": str(omni_usd_extension_id),
        "omni_usd_config_extension_id": str(omni_usd_config_extension_id),
        "omni_usd_libs_extension_id": str(omni_usd_libs_extension_id),
        "omni_usd_resolver_extension_id": str(omni_usd_resolver_extension_id),
        "extension_versions": extension_versions,
        "expected_extension_versions": expected_extension_versions,
    }

    # Treat the explicitly opened root as part of the frozen composition even if
    # a USD build returns only its dependencies in `all_layers`.
    discovered: set[Path] = {root_path}
    layer_records: list[dict[str, str]] = [
        {"identifier": str(root_path), "resolved_path": str(root_path), "role": "root"}
    ]
    recorded_layer_paths: set[Path] = {root_path}
    for layer in layers:
        candidates = [
            str(getattr(layer, "realPath", "") or ""),
            str(getattr(layer, "resolvedPath", "") or ""),
            str(getattr(layer, "identifier", "") or ""),
        ]
        resolved = next(
            (Path(value).resolve() for value in candidates
             if value and Path(value).is_file()),
            None,
        )
        if resolved is None:
            raise RuntimeError(
                f"ATTEMPT3_USD_LAYER_HAS_NO_LOCAL_FILE candidates={candidates}")
        if resolved.suffix.lower() in {".usd", ".usda", ".usdc"}:
            discovered.add(resolved)
            if resolved not in recorded_layer_paths:
                layer_records.append({
                    "identifier": str(getattr(layer, "identifier", "")),
                    "resolved_path": str(resolved),
                })
                recorded_layer_paths.add(resolved)

    asset_records: list[dict[str, str]] = []
    for item in assets:
        raw = str(getattr(item, "path", "") or str(item))
        resolved_raw = str(getattr(item, "resolvedPath", "") or "")
        candidates = [Path(resolved_raw)] if resolved_raw else []
        if raw:
            candidates.extend([Path(raw), asset_root / raw])
        resolved = next((path.resolve() for path in candidates if path.is_file()), None)
        asset_records.append({"asset_path": raw, "resolved_path": "" if resolved is None else str(resolved)})
        if resolved is not None and resolved.suffix.lower() in {".usd", ".usda", ".usdc"}:
            discovered.add(resolved)

    expected = {
        (asset_root / relative).resolve()
        for relative in ATTEMPT3_COMPOSED_LAYER_SHA256
    }
    if discovered != expected:
        raise RuntimeError(
            "ATTEMPT3_RECURSIVE_COMPOSITION_SET_MISMATCH "
            f"expected={sorted(map(str, expected))} actual={sorted(map(str, discovered))}")
    manifest = []
    for path in sorted(discovered):
        relative = path.relative_to(asset_root).as_posix()
        actual_sha = sha256_file(path)
        expected_sha = ATTEMPT3_COMPOSED_LAYER_SHA256[relative]
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"ATTEMPT3_RECURSIVE_LAYER_DRIFT path={path} "
                f"expected={expected_sha} actual={actual_sha}")
        manifest.append({"relative_path": relative, "path": str(path), "sha256": actual_sha})

    # UsdUtils.ExtractExternalReferences deliberately includes asset-valued
    # attributes, so it cannot distinguish a composition arc from an MDL shader
    # source.  Inspect Sdf's actual sublayer/reference/payload list editors instead.
    composition_arc_records: list[dict[str, str]] = []
    composition_dependency_sets: list[dict[str, Any]] = []
    mdl_asset_attributes: list[dict[str, str]] = []
    mdl_text_count_by_layer: dict[str, int] = {}
    for path in sorted(expected):
        relative = path.relative_to(asset_root).as_posix()
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f"ATTEMPT3_SDF_LAYER_OPEN_FAIL {path}")
        layer_arcs: list[dict[str, str]] = []
        for sublayer in layer.subLayerPaths:
            layer_arcs.append({
                "layer": relative, "arc_type": "sublayer", "spec_path": "",
                "list_op_bucket": "subLayerPaths", "asset_path": str(sublayer),
                "target_prim_path": "",
            })

        def visit_layer_specs(spec_path: Any) -> None:
            attribute = layer.GetAttributeAtPath(spec_path)
            if attribute is not None:
                value = attribute.default
                if isinstance(value, Sdf.AssetPath) and value.path.lower().endswith(".mdl"):
                    owner = layer.GetPrimAtPath(spec_path.GetPrimPath())
                    mdl_asset_attributes.append({
                        "layer": relative, "spec_path": str(spec_path),
                        "prim_path": str(spec_path.GetPrimPath()),
                        "attribute_name": str(attribute.name),
                        "attribute_type": str(attribute.typeName),
                        "owner_prim_type": "" if owner is None else str(owner.typeName),
                        "identifier": str(value.path),
                        "resolved_path": str(value.resolvedPath),
                    })
            if spec_path.IsPropertyPath() or spec_path == Sdf.Path.absoluteRootPath:
                return
            prim_spec = layer.GetPrimAtPath(spec_path)
            if prim_spec is None:
                return
            for arc_type, editor in (
                    ("reference", prim_spec.referenceList),
                    ("payload", prim_spec.payloadList)):
                for bucket in (
                        "explicitItems", "addedItems", "prependedItems", "appendedItems",
                        "deletedItems", "orderedItems"):
                    for item in getattr(editor, bucket):
                        layer_arcs.append({
                            "layer": relative, "arc_type": arc_type,
                            "spec_path": str(spec_path), "list_op_bucket": bucket,
                            "asset_path": str(item.assetPath),
                            "target_prim_path": str(item.primPath),
                        })

        layer.Traverse(Sdf.Path.absoluteRootPath, visit_layer_specs)
        direct_nonempty = sorted({row["asset_path"] for row in layer_arcs
                                  if row["asset_path"]})
        sdf_dependencies = sorted({str(value) for value in
                                   layer.GetCompositionAssetDependencies() if str(value)})
        if direct_nonempty != sdf_dependencies:
            raise RuntimeError(
                "ATTEMPT3_SDF_COMPOSITION_ARC_CROSSCHECK_FAIL "
                f"layer={relative} direct={direct_nonempty} sdf={sdf_dependencies}")
        composition_arc_records.extend(layer_arcs)
        composition_dependency_sets.append({
            "layer": relative, "direct_nonempty_asset_paths": direct_nonempty,
            "sdf_composition_asset_dependencies": sdf_dependencies,
            "crosscheck_pass": True,
        })
        mdl_text_count_by_layer[relative] = layer.ExportToString().count("@OmniPBR.mdl@")

    external_mdl_arc_records = [
        row for row in composition_arc_records
        if row["asset_path"].lower().endswith(".mdl")]
    if external_mdl_arc_records:
        raise RuntimeError(
            "ATTEMPT3_MDL_IDENTIFIER_USED_AS_COMPOSITION_ARC "
            f"{json.dumps(external_mdl_arc_records, sort_keys=True)}")

    allowed = set(ALLOWED_BUILTIN_MDL_IDENTIFIERS)
    authored_mdl_source_asset_pass = bool(
        len(mdl_asset_attributes) == EXPECTED_OMNIPBR_MDL_SOURCE_ASSET_COUNT
        and sum(mdl_text_count_by_layer.values()) == len(mdl_asset_attributes)
        and mdl_text_count_by_layer.get("configuration/roarm_m3_base.usd")
        == EXPECTED_OMNIPBR_MDL_SOURCE_ASSET_COUNT
        and all(
            row["layer"] == "configuration/roarm_m3_base.usd"
            and row["attribute_name"] == "info:mdl:sourceAsset"
            and row["attribute_type"] == "asset"
            and row["owner_prim_type"] == "Shader"
            and row["identifier"] in allowed
            and row["resolved_path"] == ""
            for row in mdl_asset_attributes)
    )
    if not authored_mdl_source_asset_pass:
        raise RuntimeError(
            "ATTEMPT3_BUILTIN_MDL_AUTHORED_SOURCE_ASSET_AUDIT_FAIL "
            f"records={json.dumps(mdl_asset_attributes, sort_keys=True)} "
            f"text_counts={json.dumps(mdl_text_count_by_layer, sort_keys=True)}")

    # Open each authoring layer as a stage so sibling shader specs that are not all
    # visible from the robot root composition are still checked through UsdShade.
    usdshade_source_checks: list[dict[str, Any]] = []
    stage_by_layer: dict[str, Any] = {}
    for row in mdl_asset_attributes:
        relative = row["layer"]
        if relative not in stage_by_layer:
            stage_by_layer[relative] = Usd.Stage.Open(str(asset_root / relative))
            if stage_by_layer[relative] is None:
                raise RuntimeError(
                    f"ATTEMPT3_USDSHADE_AUTHORING_LAYER_OPEN_FAIL {relative}")
        shader = UsdShade.Shader.Get(stage_by_layer[relative], Sdf.Path(row["prim_path"]))
        source_asset = shader.GetSourceAsset("mdl") if shader else None
        source_identifier = (
            _asset_identifier_record(source_asset)["normalized"]
            if isinstance(source_asset, Sdf.AssetPath) else "")
        usdshade_source_checks.append({
            "layer": relative, "prim_path": row["prim_path"],
            "shader_valid": bool(shader), "source_type": "mdl",
            "source_identifier": source_identifier,
            "matches_authored_sdf_identifier": bool(
                shader and source_identifier == row["identifier"]),
        })
    stage_by_layer.clear()
    all_authored_usdshade_pass = bool(
        len(usdshade_source_checks) == EXPECTED_OMNIPBR_MDL_SOURCE_ASSET_COUNT
        and all(row["shader_valid"] and row["matches_authored_sdf_identifier"]
                and row["source_identifier"] in allowed
                for row in usdshade_source_checks))
    if not all_authored_usdshade_pass:
        raise RuntimeError(
            "ATTEMPT3_ALL_AUTHORED_USDSHADE_SOURCE_ASSET_AUDIT_FAIL "
            f"{json.dumps(usdshade_source_checks, sort_keys=True)}")
    return {
        "resolver": "pxr.UsdUtils.ComputeAllDependencies",
        "root_path": str(root_path),
        "unresolved_dependencies": [
            row["raw"] for row in unresolved_classification["raw_records"]],
        "unresolved_dependency_classification": unresolved_classification,
        "allowed_builtin_mdl_identifiers": sorted(ALLOWED_BUILTIN_MDL_IDENTIFIERS),
        "runtime_builtin_mdl_resolver": runtime_builtin_mdl,
        "external_composition_arc_audit": {
            "extractor": (
                "Sdf.Layer.subLayerPaths plus every authored PrimSpec reference/payload "
                "list-op bucket; crosschecked by Layer.GetCompositionAssetDependencies"),
            "records": sorted(
                composition_arc_records,
                key=lambda row: (row["layer"], row["arc_type"], row["spec_path"],
                                 row["list_op_bucket"], row["asset_path"])),
            "per_layer_dependency_crosscheck": composition_dependency_sets,
            "mdl_arc_records": external_mdl_arc_records,
            "mdl_external_arc_absence_pass": True,
        },
        "usdshade_mdl_source_asset_audit": {
            "all_authored_sdf_asset_attributes": sorted(
                mdl_asset_attributes,
                key=lambda row: (row["layer"], row["spec_path"])),
            "authored_text_count_by_layer": mdl_text_count_by_layer,
            "expected_authored_count": EXPECTED_OMNIPBR_MDL_SOURCE_ASSET_COUNT,
            "all_authored_source_asset_pass": True,
            "authoring_layer_usdshade_shader_sources": sorted(
                usdshade_source_checks,
                key=lambda row: (row["layer"], row["prim_path"])),
            "all_authored_usdshade_pass": True,
            "authority_note": (
                "Sdf traversal enumerates authored occurrences; each authoring layer is "
                "also opened as a stage and every occurrence is checked with UsdShade"),
            "exclusive_source_asset_pass": True,
        },
        "layer_records": layer_records,
        "asset_records": asset_records,
        "composed_local_usd_layers": manifest,
        "exact_expected_set_pass": True,
    }


def _curve_negative_control(theta_row: dict[str, Any]) -> dict[str, Any]:
    collision = theta_row.get("collision")
    if not isinstance(collision, dict) or not isinstance(collision.get("curve"), list):
        raise RuntimeError(f"p13 theta row missing collision.curve: {theta_row.get('theta_deg')}")
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for row in collision["curve"]:
        required = {"q5_deg", "unilateral_bite_mm", "bilateral_bite_mm"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise RuntimeError(f"p13 curve row schema mismatch at theta={theta_row.get('theta_deg')}")
        q5 = float(row["q5_deg"])
        uni = max(0.0, float(row["unilateral_bite_mm"] or 0.0))
        bi = max(0.0, float(row["bilateral_bite_mm"] or 0.0))
        scored.append((max(uni, bi), -q5, row))
    if not scored:
        raise RuntimeError(f"p13 empty curve at theta={theta_row.get('theta_deg')}")
    score, _neg_q5, row = min(scored, key=lambda x: (x[0], x[1]))
    return {
        "q5_deg": float(row["q5_deg"]),
        "kind": "no_bite_negative_control" if score <= 1.0e-12 else "least_bite_negative_control",
        "unilateral_bite_mm": None if row["unilateral_bite_mm"] is None else float(row["unilateral_bite_mm"]),
        "bilateral_bite_mm": None if row["bilateral_bite_mm"] is None else float(row["bilateral_bite_mm"]),
        "curve_delta_m": row.get("delta_m"),
    }


def load_p13_handoff(path: Path, expected_sha256: str = P13_RESULTS_SHA256) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"P13_HANDOFF_MISSING {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"P13_RESULTS_SHA256_MISMATCH expected={expected_sha256} actual={actual_sha256}")
    raw = json.loads(path.read_text())
    handoff = raw.get("physics_handoff")
    if not isinstance(handoff, dict):
        raise RuntimeError("p13 results missing physics_handoff object")
    if handoff.get("schema_version") != 1 or handoff.get("source_tag") != "t3x_bite81":
        raise RuntimeError(
            "P13_HANDOFF_IDENTITY_FAIL "
            f"schema={handoff.get('schema_version')} source={handoff.get('source_tag')}"
        )
    if handoff.get("run_physx_regardless_of_bilateral") is not True:
        raise RuntimeError("p13 must explicitly set run_physx_regardless_of_bilateral=true")
    producer_gates = raw.get("gates")
    source_freeze = raw.get("source_freeze")
    n10_regression = raw.get("n10_regression")
    required_producer_gates = (
        "X1_input_sha", "X1_env_pins", "X2_asset_identity_64_plus_64",
        "X3_n10_theta29_35_regression", "X6_source_stable",
    )
    producer_failures: list[str] = []
    if raw.get("run_valid") is not True:
        producer_failures.append("run_valid_is_not_true")
    if not isinstance(source_freeze, dict) or source_freeze.get("stable_at_end") is not True:
        producer_failures.append("source_freeze.stable_at_end_is_not_true")
    if not isinstance(producer_gates, dict):
        producer_failures.append("gates_not_object")
    else:
        producer_failures.extend(
            f"gates.{name}_is_not_true" for name in required_producer_gates
            if producer_gates.get(name) is not True)
    if not isinstance(n10_regression, dict) or n10_regression.get("pass") is not True:
        producer_failures.append("n10_regression.pass_is_not_true")
    if producer_failures:
        raise RuntimeError(f"P13_PRODUCER_VALIDITY_FAIL failures={producer_failures}")
    producer_validity = {
        "run_valid": True,
        "source_freeze_stable_at_end": True,
        "required_gates": {name: True for name in required_producer_gates},
        "n10_regression_pass": True,
        "rerun_pass_not_scientific_handoff_gate": True,
    }
    controls = handoff.get("controls")
    if not isinstance(controls, list):
        raise RuntimeError("p13 physics_handoff.controls must be a list")
    per_theta = raw.get("per_theta")
    if not isinstance(per_theta, list):
        raise RuntimeError("p13 per_theta must be a list")
    candidates = handoff.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("p13 physics_handoff.candidates must be a list")
    candidate_map = {str(row.get("pose_key")): row for row in candidates}
    reactive_basis: dict[str, Any] = {}
    for pose_key, expected in JAW_SUPPORT_REACTIVE_BASIS.items():
        row = candidate_map.get(pose_key)
        if not isinstance(row, dict):
            raise RuntimeError(f"p13 jaw-support reactive-basis pose missing: {pose_key}")
        clearance = float(row.get("min_table_clearance_mm"))
        count = int(row.get("table_penetration_count"))
        if (not math.isclose(clearance, expected["min_table_clearance_mm"], abs_tol=1.0e-12)
                or count != expected["table_penetration_count"]):
            raise RuntimeError(
                f"p13 jaw-support reactive basis drift pose={pose_key} "
                f"expected={expected} actual_clearance={clearance} actual_count={count}")
        reactive_basis[pose_key] = {
            "min_table_clearance_mm": clearance, "table_penetration_count": count}

    def key(x: Any) -> float:
        return round(float(x), 6)

    control_map = {key(row["theta_deg"]): row for row in controls}
    theta_map = {key(row["theta_deg"]): row for row in per_theta}
    if set(control_map) != set(EXPECTED_THETA):
        raise RuntimeError(
            f"p13 controls theta set must be exact {EXPECTED_THETA}, got {sorted(control_map)}"
        )
    if not set(EXPECTED_THETA).issubset(theta_map):
        raise RuntimeError(f"p13 per_theta lacks required rows: {sorted(set(EXPECTED_THETA)-set(theta_map))}")

    normalized: list[dict[str, Any]] = []
    for theta in EXPECTED_THETA:
        row = control_map[theta]
        required = {
            "theta_deg", "phi_deg", "window_kind", "q5_window_deg",
            "q5_close_targets_deg", "grasp_surface_margin_m", "q_descend_deg",
            "requires_pose_specific_ik",
        }
        if not required.issubset(row):
            raise RuntimeError(f"p13 control theta={theta} missing {sorted(required-set(row))}")
        if row["window_kind"] not in WINDOW_KINDS:
            raise RuntimeError(f"invalid p13 window_kind={row['window_kind']!r} theta={theta}")
        if row["requires_pose_specific_ik"] is not True or row["q_descend_deg"] is not None:
            raise RuntimeError(f"theta={theta} must require pose-specific IK with q_descend_deg=null")
        targets = [float(v) for v in row["q5_close_targets_deg"]]
        if not targets or any(not math.isfinite(v) for v in targets):
            raise RuntimeError(f"theta={theta} has no finite q5 targets")
        margin = float(row["grasp_surface_margin_m"])
        phi = float(row["phi_deg"])
        if not math.isfinite(margin) or not math.isfinite(phi):
            raise RuntimeError(f"theta={theta} has non-finite phi/margin")
        curve = theta_map[theta]["collision"]["curve"]

        def curve_control(q: float, kind: str, source: str) -> dict[str, Any]:
            at_q = min(curve, key=lambda item: abs(float(item["q5_deg"])-q))
            residual = abs(float(at_q["q5_deg"])-q)
            if residual > 0.051:
                raise RuntimeError(
                    f"theta={theta} q5={q} has no exact 0.1-deg p13 curve row; residual={residual}")
            q_margin = at_q.get("delta_m")
            used_margin = margin if q_margin is None else float(q_margin)
            if not math.isfinite(used_margin):
                raise RuntimeError(
                    f"theta={theta} q5={q} has non-finite effective margin {used_margin}")
            effective_kind = _effective_window_kind(kind, q_margin)
            return {
                "q5_deg": q, "kind": effective_kind, "reported_curve_kind": kind,
                "source": source,
                "curve_q5_deg": float(at_q["q5_deg"]), "curve_q5_residual_deg": residual,
                "grasp_surface_margin_m": used_margin,
                "margin_source": (
                    f"p13_control_fallback:{row.get('margin_source', 'unspecified')}"
                    if q_margin is None else "p13_same_theta_q5_curve"),
                "unilateral_bite_mm": at_q.get("unilateral_bite_mm"),
                "bilateral_bite_mm": at_q.get("bilateral_bite_mm"),
            }

        q5_controls = [curve_control(q, str(row["window_kind"]), "p13_interior_target")
                       for q in targets]
        negative = _curve_negative_control(theta_map[theta])
        if all(abs(negative["q5_deg"] - item["q5_deg"]) > 1.0e-9 for item in q5_controls):
            q5_controls.append(curve_control(
                float(negative["q5_deg"]), str(negative["kind"]), "p13_same_theta_curve"))
        normalized.append({
            "theta_deg": theta,
            "form": FORM_BY_THETA[theta],
            "phi_deg": phi,
            "window_kind": str(row["window_kind"]),
            "q5_window_deg": row["q5_window_deg"],
            "grasp_surface_margin_m": margin,
            "control_margin_source": row.get("margin_source", "unspecified"),
            "q5_controls": q5_controls,
        })
    return {
        "path": str(path), "sha256": actual_sha256,
        "schema_version": 1, "source_tag": "t3x_bite81",
        "controls": normalized,
        "input_sha256": raw.get("input_sha256"),
        "producer_validity": producer_validity,
        "jaw_support_reactive_basis": reactive_basis,
    }


def build_workspace(p10: Any, grid_side: int) -> list[dict[str, Any]]:
    if grid_side < 2:
        raise ValueError("grid_side must be >=2")
    rows: list[dict[str, Any]] = []
    seeds = p10.FOUR_SPONGE_SEED0_SOURCES
    for ridx, bounds in enumerate(p10.SOURCE_REGIONS, start=1):
        x0, x1, y0, y1 = map(float, bounds)
        xs = x0 + (np.arange(grid_side) + 0.5) * ((x1 - x0) / grid_side)
        ys = y0 + (np.arange(grid_side) + 0.5) * ((y1 - y0) / grid_side)
        region: list[dict[str, Any]] = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                region.append({
                    "pose_key": f"R{ridx}_g{iy:02d}_{ix:02d}", "region": f"R{ridx}",
                    "grid_ix": int(ix), "grid_iy": int(iy),
                    "x_m": float(x), "y_m": float(y), "is_exact_seed": False,
                })
        seed_key = f"seed0_S{ridx}"
        sx, sy = map(float, seeds[seed_key])
        nearest = min(range(len(region)), key=lambda i: (region[i]["x_m"]-sx)**2 + (region[i]["y_m"]-sy)**2)
        region[nearest].update({
            "pose_key": seed_key, "x_m": sx, "y_m": sy, "is_exact_seed": True,
        })
        rows.extend(region)
    if len(rows) != 4 * grid_side * grid_side or len({r["pose_key"] for r in rows}) != len(rows):
        raise RuntimeError("workspace construction count/key invariant failed")
    return rows


class _PlanArgs:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _worker_init(plan_kw: dict[str, Any]) -> None:
    _W["p10"] = _import_p10()
    _W["plan_kw"] = plan_kw


def _actual_axis_metrics(p10: Any, q_deg: np.ndarray) -> dict[str, float]:
    tcp, link5 = p10.fk_full_5(np.asarray(q_deg[:5], dtype=np.float64))
    axis = tcp[:3, 3] - link5[:3, 3]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    theta = math.degrees(math.acos(float(np.clip(np.dot(axis, [0.0, 0.0, -1.0]), -1.0, 1.0))))
    psi = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 360.0
    return {"theta_actual_deg": theta, "psi_axis_actual_deg": psi,
            "phi_actual_deg": float(p10._phi_at(q_deg[:5]))}


def _worker_plan(job: dict[str, Any]) -> dict[str, Any]:
    p10 = _W["p10"]
    kw = dict(_W["plan_kw"])
    kw["grasp_surface_margin_m"] = float(job["grasp_surface_margin_m"])
    kw["close_deg"] = [kw["descend_open_deg"], float(job["q5_close_deg"])]
    center = np.asarray(job["center_m"], dtype=np.float64)
    p10.set_target_axis(float(job["theta_target_deg"]), float(job["psi_axis_target_deg"]))
    p10.PHI_STAR_DEG = float(job["phi_target_deg"])
    out = dict(job)
    try:
        plan = p10._build_plan_from_center(_PlanArgs(**kw), center, job["trial_id"])
    except Exception as exc:
        out.update({"feasible": False, "reason": f"planner_error:{type(exc).__name__}:{exc}"})
        return out
    ik_checks = {
        "approach": bool(plan.approach_ik_ok), "descend": bool(plan.descend_ik_ok),
        "lift": bool(plan.lift_ik_ok),
    }
    phase_q = {
        "approach": plan.q_approach_deg,
        "descend": plan.q_descend_deg,
        "lift": plan.q_lift_deg,
    }
    # p10's roll solver uses the URDF hardware envelope (+/-180 deg), whereas
    # this chain's data-distribution/V6 contract is +/-90 deg.  That fifth-joint
    # integration gate must be explicit at every executed waypoint.
    wrist_r_checks = {
        phase: bool(-90.0-1.0e-9 <= float(q[4]) <= 90.0+1.0e-9)
        for phase, q in phase_q.items()
    }
    feasible = all(ik_checks.values()) and all(wrist_r_checks.values())
    failed = [f"ik_{name}" for name, ok in ik_checks.items() if not ok]
    failed += [f"wrist_r_v6_{name}" for name, ok in wrist_r_checks.items() if not ok]
    actual_by_phase = {name: _actual_axis_metrics(p10, q) for name, q in phase_q.items()}
    out.update({
        "feasible": feasible,
        "reason": "" if feasible else "plan_gate:" + ",".join(failed),
        "ik_ok": ik_checks,
        "wrist_r_v6_ok": wrist_r_checks,
        "wrist_r_deg": {name: float(q[4]) for name, q in phase_q.items()},
        "ik_err_mm": [float(plan.approach_ik_err_mm), float(plan.descend_ik_err_mm), float(plan.lift_ik_err_mm)],
        "ik_direction_error_deg": [float(plan.approach_tilt_deg), float(plan.descend_tilt_deg), float(plan.lift_tilt_deg)],
        "actual_axis_by_phase": actual_by_phase,
        "world_grasp_m": plan.world_grasp.tolist(),
        "approach_tcp_m": plan.approach_tcp.tolist(),
        "descend_tcp_m": plan.descend_tcp.tolist(),
        "lift_tcp_m": plan.lift_tcp.tolist(),
        "q_approach_deg": plan.q_approach_deg.tolist(),
        "q_descend_deg": plan.q_descend_deg.tolist(),
        "q_lift_deg": plan.q_lift_deg.tolist(),
    })
    out.update(actual_by_phase["descend"])
    return out


def build_plan(args: argparse.Namespace, p10: Any, handoff: dict[str, Any]) -> dict[str, Any]:
    import multiprocessing as mp

    positions = build_workspace(p10, args.grid_side)
    jobs: list[dict[str, Any]] = []
    seq = 0
    for pos in positions:
        x, y = float(pos["x_m"]), float(pos["y_m"])
        psi = math.degrees(math.atan2(y, x)) % 360.0
        for control in handoff["controls"]:
            for q5 in control["q5_controls"]:
                jobs.append({
                    "trial_id": f"trial_{seq:06d}", "trial_index": seq,
                    **pos,
                    "center_m": [x, y, SUPPORT_Z_M + OBJ_HEIGHT_M / 2.0],
                    "r_m": float(math.hypot(x, y)),
                    "psi_pos_deg": psi,
                    "form": control["form"],
                    "theta_target_deg": float(control["theta_deg"]),
                    "psi_axis_target_deg": psi,
                    "phi_target_deg": float(control["phi_deg"]),
                    "window_kind": str(q5["kind"]),
                    "q5_control_source": str(q5["source"]),
                    "q5_window_deg": control["q5_window_deg"],
                    "q5_close_deg": float(q5["q5_deg"]),
                    "grasp_surface_margin_m": float(q5["grasp_surface_margin_m"]),
                    "descend_margin_source": str(q5["margin_source"]),
                    "static_unilateral_bite_mm": q5.get("unilateral_bite_mm"),
                    "static_bilateral_bite_mm": q5.get("bilateral_bite_mm"),
                    "descend_control": "p13_exact_theta_and_q5_curve_margin",
                })
                seq += 1

    plan_kw = {
        "object_size_m": np.asarray([OBJ_DIAM_M, OBJ_DIAM_M, OBJ_HEIGHT_M], dtype=np.float64),
        "grasp_surface_margin_m": 0.0,
        "approach_clearance_m": args.approach_clearance_m,
        "lift_delta_m": args.lift_delta_m,
        "descend_open_deg": args.descend_open_deg,
        "close_deg": [args.descend_open_deg, 20.0],
        "target_error_gate_m": args.target_error_gate_m,
        "plan_tilt_gate_deg": args.plan_tilt_gate_deg,
    }
    workers = max(1, min(int(args.plan_workers), len(jobs)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(plan_kw,)) as pool:
        planned = pool.map(_worker_plan, jobs, chunksize=8)

    feasible = [row for row in planned if row.get("feasible")]
    by_form: dict[str, dict[str, int]] = {}
    for form in sorted(set(row["form"] for row in planned)):
        group = [row for row in planned if row["form"] == form]
        by_form[form] = {
            "planned": len(group),
            "ik_pass": sum(bool(r.get("ik_ok")) and all(r["ik_ok"].values()) for r in group),
            "wrist_r_v6_pass": sum(
                bool(r.get("wrist_r_v6_ok")) and all(r["wrist_r_v6_ok"].values()) for r in group),
            "feasible": sum(bool(r.get("feasible")) for r in group),
        }
    reasons: dict[str, int] = {}
    for row in planned:
        if not row.get("feasible"):
            reasons[row.get("reason", "unknown")] = reasons.get(row.get("reason", "unknown"), 0) + 1
    return {
        "protocol": str(args.protocol_path_resolved.relative_to(REPO)),
        "grid_side": int(args.grid_side), "positions": positions,
        "n_positions": len(positions), "n_planned": len(planned),
        "n_feasible": len(feasible),
        "n_plan_gate_failed": len(planned)-len(feasible),
        "n_ik_failed": sum(
            not row.get("ik_ok") or not all(row["ik_ok"].values()) for row in planned),
        "n_wrist_r_v6_failed": sum(
            bool(row.get("wrist_r_v6_ok")) and not all(row["wrist_r_v6_ok"].values())
            for row in planned),
        "failure_reasons": reasons, "by_form": by_form, "trials": planned,
    }


def _find_ground_collider(stage: Any) -> str:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath("/World/ground")
    if not root.IsValid():
        raise RuntimeError("terrain root /World/ground not found")
    hits = [p.GetPath().pathString for p in Usd.PrimRange(root)
            if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not hits:
        raise RuntimeError("no collider under /World/ground")
    return sorted(hits, key=lambda s: (-s.count("/"), s))[0]


def _audit_cloned_jaw_contact_reporters(stage: Any, num_envs: int) -> dict[str, Any]:
    """Exhaustively prove threshold-zero reporters on both jaws in every clone."""
    from pxr import PhysxSchema

    failures: list[dict[str, Any]] = []
    checked = 0
    thresholds: set[float] = set()
    for env_index in range(int(num_envs)):
        for body in ("link5", "gripper_link"):
            path = f"/World/envs/env_{env_index}/Robot/{body}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                failures.append({"path": path, "reason": "prim_missing"})
                continue
            api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPath())
            threshold = None if not api else api.GetThresholdAttr().Get()
            if threshold is None or abs(float(threshold)) > 1.0e-12:
                failures.append({
                    "path": path, "reason": "threshold_not_zero", "threshold": threshold,
                })
                continue
            checked += 1
            thresholds.add(float(threshold))
    expected = int(num_envs) * 2
    if failures or checked != expected or thresholds != {0.0}:
        raise RuntimeError(
            "CLONED_JAW_CONTACT_REPORTER_AUDIT_FAIL "
            f"expected={expected} checked={checked} thresholds={sorted(thresholds)} "
            f"failures_sample={failures[:20]} failure_count={len(failures)}")
    return {
        "pass": True, "num_envs": int(num_envs), "bodies_per_env": 2,
        "expected_reporters": expected, "checked_reporters": checked,
        "thresholds": sorted(thresholds), "scope": "all cloned environment jaw bodies",
    }


def make_env(args: argparse.Namespace, p10: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import PhysxSchema
    from roarm_rl.roarm_stack_env import RoArmStackEnv, RoArmStackEnvCfg

    cfg = RoArmStackEnvCfg()
    if str(cfg.robot.spawn.usd_path) != str(p10.ATTEMPT3_USD) or "/NHNHOME" in str(cfg.robot.spawn.usd_path):
        raise RuntimeError(f"USD_GUARD_FAIL effective={cfg.robot.spawn.usd_path}")
    cfg.scene.num_envs = int(args.num_envs)
    cfg.scene.clone_in_fabric = False
    # One runner step == one 1/200 s PhysX step.  This makes the representative
    # "every step" trace genuinely substep-complete instead of observing only
    # every second PhysX step through the RL environment's default decimation=2.
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.episode_length_s = float(args.episode_length_s)
    cfg.sim.physx = sim_utils.PhysxCfg(
        gpu_found_lost_pairs_capacity=int(args.gpu_found_lost_pairs_capacity),
        gpu_total_aggregate_pairs_capacity=int(args.gpu_total_aggregate_pairs_capacity),
        gpu_collision_stack_size=int(args.gpu_collision_stack_size),
        gpu_max_rigid_contact_count=int(args.gpu_max_rigid_contact_count),
        solve_articulation_contact_last=False,
    )
    cfg.reward_phase = 6
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_attached_transport_release = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    cfg.sponge.spawn = sim_utils.CylinderCfg(
        radius=OBJ_DIAM_M / 2.0, height=OBJ_HEIGHT_M, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=1,
            max_angular_velocity=10.0, max_linear_velocity=10.0,
            max_depenetration_velocity=5.0, disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJ_MASS_KG),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args.static_friction, dynamic_friction=args.dynamic_friction,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.62, 0.38)),
    )
    cfg.sponge.init_state.pos = (0.20, -0.15, OBJ_HEIGHT_M / 2.0)
    cfg.sponge.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    base_spawn = cfg.sponge.spawn.func
    armed: dict[str, Any] = {"count": 0, "thresholds": []}

    def spawn_armed(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = base_spawn(prim_path, spawn_cfg, translation=translation,
                          orientation=orientation, **kwargs)
        sim_utils.activate_contact_sensors(prim.GetPath().pathString, threshold=0.0)
        api = PhysxSchema.PhysxContactReportAPI.Get(prim.GetStage(), prim.GetPath())
        threshold = api.GetThresholdAttr().Get()
        if threshold is None or abs(float(threshold)) > 1.0e-12:
            raise RuntimeError(f"contact threshold readback failed: {threshold}")
        armed["count"] += 1
        armed["thresholds"].append(float(threshold))
        return prim

    cfg.sponge.spawn.func = spawn_armed

    # Reactive instrumentation: p13 observed millimetre-scale jaw/table
    # penetration in S3/S4 and r=.45 candidates.  Arm the two exact jaw rigid
    # bodies at threshold 0.0 before scene cloning; do not use the spawn cfg bool,
    # which Isaac Lab 2.3 passes into the numeric threshold slot.
    cfg.robot.spawn.activate_contact_sensors = False
    robot_base_spawn = cfg.robot.spawn.func
    robot_armed: dict[str, Any] = {
        "expected_bodies": ["link5", "gripper_link"], "count": 0,
        "paths": [], "thresholds": [],
    }

    def spawn_robot_jaws_armed(prim_path, spawn_cfg, translation=None,
                               orientation=None, **kwargs):
        prim = robot_base_spawn(prim_path, spawn_cfg, translation=translation,
                                orientation=orientation, **kwargs)
        stage = prim.GetStage()
        for body in robot_armed["expected_bodies"]:
            body_path = f"{prim.GetPath().pathString}/{body}"
            body_prim = stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                raise RuntimeError(f"jaw contact reporter body missing: {body_path}")
            sim_utils.activate_contact_sensors(body_path, threshold=0.0, stage=stage)
            api = PhysxSchema.PhysxContactReportAPI.Get(stage, body_prim.GetPath())
            threshold = api.GetThresholdAttr().Get()
            if threshold is None or abs(float(threshold)) > 1.0e-12:
                raise RuntimeError(
                    f"jaw contact threshold readback failed body={body} value={threshold}")
            robot_armed["count"] += 1
            robot_armed["paths"].append(body_path)
            robot_armed["thresholds"].append(float(threshold))
        return prim

    cfg.robot.spawn.func = spawn_robot_jaws_armed
    resolved: dict[str, Any] = {}

    class P14WorkspaceEnv(RoArmStackEnv):
        def _setup_scene(self) -> None:
            super()._setup_scene()
            ground = _find_ground_collider(self.scene.stage)
            filters = [ground, "/World/envs/env_.*/Robot/link5",
                       "/World/envs/env_.*/Robot/gripper_link"]
            resolved.update({"support_plane": ground, "filter_paths": filters})
            sensor = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Sponge",
                filter_prim_paths_expr=filters,
                update_period=0.0, history_length=1,
                track_pose=False, track_contact_points=True,
                max_contact_data_count_per_prim=int(args.contact_capacity),
                force_threshold=0.0, debug_vis=False,
            ))
            self.scene.sensors["t3y_object_contact"] = sensor
            self._t3y_contact_sensor = sensor
            jaw_ground_cfg = {
                "filter_prim_paths_expr": [ground],
                "update_period": 0.0, "history_length": 1,
                "track_pose": False, "track_contact_points": False,
                "max_contact_data_count_per_prim": int(args.contact_capacity),
                "force_threshold": 0.0, "debug_vis": False,
            }
            fixed_ground = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/link5", **jaw_ground_cfg))
            moving_ground = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_link", **jaw_ground_cfg))
            self.scene.sensors["t3y_fixed_jaw_ground_contact"] = fixed_ground
            self.scene.sensors["t3y_moving_jaw_ground_contact"] = moving_ground
            self._t3y_fixed_ground_sensor = fixed_ground
            self._t3y_moving_ground_sensor = moving_ground
            resolved["jaw_ground"] = {
                "fixed_sensor_prim": "/World/envs/env_.*/Robot/link5",
                "moving_sensor_prim": "/World/envs/env_.*/Robot/gripper_link",
                "support_filter": ground,
            }

        def _apply_action(self) -> None:
            self._robot.set_joint_position_target(self.robot_dof_targets)

        def _get_rewards(self):
            import torch
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            import torch
            z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return z, z.clone()

    env = P14WorkspaceEnv(cfg=cfg)
    env._t3y_spawn_arm_report = {"object": armed, "robot_jaws": robot_armed}
    env._t3y_resolved = resolved
    return env


def resolve_filter_map(sensor: Any, expected_envs: int) -> dict[str, int]:
    raw_outer = list(sensor.contact_physx_view.filter_paths)
    raw = [str(v) for v in (list(raw_outer[0]) if raw_outer and not isinstance(raw_outer[0], (str, bytes)) else raw_outer)]
    force_shape = tuple(int(v) for v in sensor.data.force_matrix_w.shape)
    contact_shape = (None if sensor.data.contact_pos_w is None else
                     tuple(int(v) for v in sensor.data.contact_pos_w.shape))
    expected_shape = (int(expected_envs), 1, 3, 3)
    if sensor.num_bodies != 1 or force_shape != expected_shape or contact_shape != expected_shape:
        raise RuntimeError(
            "object contact sensor exact shape mismatch "
            f"expected={expected_shape} num_bodies={sensor.num_bodies} "
            f"force={force_shape} contact_pos={contact_shape} paths={raw}")
    n_cols = force_shape[2]
    mapping: dict[str, int] = {}
    for i, path in enumerate(raw[:n_cols]):
        if path.endswith("/Robot/link5"):
            label = "link5"
        elif path.endswith("/Robot/gripper_link"):
            label = "gripper_link"
        elif "/ground" in path:
            label = "support_plane"
        else:
            raise RuntimeError(f"unknown contact filter path {path!r}")
        mapping[label] = i
    if set(mapping) != {"support_plane", "link5", "gripper_link"}:
        raise RuntimeError(f"contact filter mapping failed: {mapping}")
    return mapping


def resolve_single_ground_filter(sensor: Any, sensor_label: str,
                                 expected_envs: int) -> dict[str, Any]:
    raw_outer = list(sensor.contact_physx_view.filter_paths)
    raw = [str(value) for value in (
        list(raw_outer[0])
        if raw_outer and not isinstance(raw_outer[0], (str, bytes)) else raw_outer)]
    shape = tuple(int(value) for value in sensor.data.force_matrix_w.shape)
    expected_shape = (int(expected_envs), 1, 1, 3)
    if sensor.num_bodies != 1 or shape != expected_shape:
        raise RuntimeError(
            f"{sensor_label} expected one body/one ground filter; "
            f"expected_shape={expected_shape} num_bodies={sensor.num_bodies} "
            f"force_shape={shape} paths={raw}")
    if sensor.data.contact_pos_w is not None:
        raise RuntimeError(
            f"{sensor_label} is force-only but contact_pos_w was allocated: "
            f"shape={tuple(sensor.data.contact_pos_w.shape)}")
    if len(raw) != 1 or "/ground" not in raw[0]:
        raise RuntimeError(f"{sensor_label} ground filter resolution failed: {raw}")
    return {
        "support_plane": 0, "resolved_path": raw[0],
        "force_matrix_shape": list(shape), "contact_pos_w": None,
        "num_bodies": int(sensor.num_bodies),
    }


def _tensor_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy()


def run_batch(
    args: argparse.Namespace,
    env: Any,
    samples: list[dict[str, Any]],
    *,
    active_count: int,
    capture_slots: list[int] | None = None,
) -> dict[str, Any]:
    """Execute one fixed-size batch; optionally retain every step for selected slots."""
    import torch
    from roarm_rl.roarm_stack_env import _quat_rotate

    n = env.num_envs
    if len(samples) != n or not 0 < active_count <= n:
        raise RuntimeError(f"batch shape mismatch len={len(samples)} active={active_count} envs={n}")
    captures = sorted(set(capture_slots or []))
    if any(i < 0 or i >= active_count for i in captures):
        raise RuntimeError(f"capture slot outside active population: {captures}/{active_count}")
    device = env.device
    sensor = env._t3y_contact_sensor
    fixed_ground_sensor = env._t3y_fixed_ground_sensor
    moving_ground_sensor = env._t3y_moving_ground_sensor
    fmap = resolve_filter_map(sensor, n)
    fixed_ground_map = resolve_single_ground_filter(
        fixed_ground_sensor, "fixed_jaw_ground", n)
    moving_ground_map = resolve_single_ground_filter(
        moving_ground_sensor, "moving_jaw_ground", n)

    expected_joints = ["base_link_to_link1", "link1_to_link2", "link2_to_link3",
                       "link3_to_link4", "link4_to_link5", "link5_to_gripper_link"]
    if list(env._robot.joint_names) != expected_joints or int(env.gripper_joint_idx) != 5:
        raise RuntimeError(f"joint identity mismatch names={env._robot.joint_names} q5={env.gripper_joint_idx}")

    attach_calls = {"n": 0}

    def no_attach() -> None:
        attach_calls["n"] += 1

    env._update_grasp_attach = no_attach

    def t(values: Any, dtype=torch.float32) -> Any:
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

    q_app = t([s["q_approach_deg"] for s in samples]) * math.pi / 180.0
    q_des = t([s["q_descend_deg"] for s in samples]) * math.pi / 180.0
    q_lft = t([s["q_lift_deg"] for s in samples]) * math.pi / 180.0
    q5_close = t([s["q5_close_deg"] for s in samples]) * math.pi / 180.0
    q5_open = float(args.descend_open_deg) * math.pi / 180.0
    q_home = t([[0.0, 0.0, math.pi / 2.0, 0.0, 0.0, q5_open]] * n)
    q_des_closed = q_des.clone()
    q_des_closed[:, 5] = q5_close
    q_lft_closed = q_lft.clone()
    q_lft_closed[:, 5] = q5_close

    env.reset()
    origins = env.scene.env_origins
    spawn_pos = t([s["center_m"] for s in samples]) + origins
    quat = t([[1.0, 0.0, 0.0, 0.0]] * n)
    env._sponge.write_root_pose_to_sim(torch.cat([spawn_pos, quat], dim=-1))
    env._sponge.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
    env._robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
    sensor.reset()
    fixed_ground_sensor.reset()
    moving_ground_sensor.reset()

    stiffness = torch.full((n, 6), float(args.stiffness), device=device)
    damping = torch.full((n, 6), float(args.damping), device=device)
    env._robot.write_joint_stiffness_to_sim(stiffness)
    env._robot.write_joint_damping_to_sim(damping)

    sched = [
        ("settle", args.settle_steps, q_home, q_home),
        ("approach", args.approach_steps, q_home, q_app),
        ("descend", args.descend_steps, q_app, q_des),
        ("close", args.close_steps, q_des, q_des_closed),
        ("hold", args.hold_steps, q_des_closed, q_des_closed),
        ("lift", args.lift_steps, q_des_closed, q_lft_closed),
    ]
    null_action = torch.zeros((n, 6), device=device, dtype=torch.float32)
    descend_tcp_ref = t([s["descend_tcp_m"] for s in samples])
    tcp_local = env._tcp_local

    zeros = lambda: torch.zeros(n, device=device)
    acc: dict[str, Any] = {
        "settle_support_fz": [], "preclose_fixed_max": zeros(), "preclose_moving_max": zeros(),
        "close_fixed_max": zeros(), "close_moving_max": zeros(),
        "lift_fixed_max": zeros(), "lift_moving_max": zeros(),
        "close_bilateral_min_force_max": zeros(),
        "lift_bilateral_min_force_max": zeros(),
        "close_obj_drift_max": zeros(), "max_tilt_deg": zeros(),
        "fixed_jaw_ground_max": zeros(), "moving_jaw_ground_max": zeros(),
        "fixed_jaw_ground_raw_count_max": torch.zeros(n, device=device, dtype=torch.int64),
        "moving_jaw_ground_raw_count_max": torch.zeros(n, device=device, dtype=torch.int64),
        "contact_steps": zeros(),
        "contact_data_total_peak": torch.zeros((), device=device, dtype=torch.int64),
        "fixed_ground_contact_data_total_peak": torch.zeros((), device=device, dtype=torch.int64),
        "moving_ground_contact_data_total_peak": torch.zeros((), device=device, dtype=torch.int64),
    }
    trace_lists: dict[str, list[Any]] = {name: [] for name in (
        "physics_step", "sim_time_s", "phase_id", "phase_step",
        "obj_pos_m", "obj_quat_wxyz", "tcp_pos_m", "link5_pos_m", "link5_quat_wxyz",
        "moving_pos_m", "moving_quat_wxyz", "q_actual_deg", "q_target_deg",
        "tilt_deg", "force_w_n", "contact_pos_m", "jaw_ground_force_w_n",
        "jaw_ground_raw_count",
    )}
    obj_ref = None
    step_global = 0
    tcp_at_descend_end = torch.zeros(n, device=device)
    t0 = time.time()

    for phase_id, (phase, nsteps, q_from, q_to) in enumerate(sched):
        for phase_step in range(nsteps):
            frac = (phase_step + 1) / float(nsteps)
            target = q_from + (q_to - q_from) * frac
            env.robot_dof_targets[:] = target
            env.step(null_action)

            obj_pos = env._sponge.data.root_pos_w - origins
            obj_quat = env._sponge.data.root_quat_w
            w, x, y, z = obj_quat.unbind(-1)
            tilt = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (x*x + y*y), -1.0, 1.0)))
            link5_pos_w = env._robot.data.body_pos_w[:, env.link5_idx]
            link5_quat = env._robot.data.body_quat_w[:, env.link5_idx]
            moving_pos = env._robot.data.body_pos_w[:, env.gripper_link_idx] - origins
            moving_quat = env._robot.data.body_quat_w[:, env.gripper_link_idx]
            tcp = link5_pos_w + _quat_rotate(link5_quat, tcp_local.expand(n, 3)) - origins
            link5_pos = link5_pos_w - origins

            fm = sensor.data.force_matrix_w[:, 0]
            cp = sensor.data.contact_pos_w
            if cp is None:
                raise RuntimeError("track_contact_points=True but contact_pos_w is None")
            cp_local_raw = cp[:, 0] - origins[:, None, :]
            # Contact-point columns follow the runtime-resolved filter order, which
            # is not assumed to equal cfg declaration order at N>1.  Reorder them
            # to the same canonical support/fixed/moving order as force_stack.
            cp_local = torch.stack(
                (cp_local_raw[:, fmap["support_plane"]],
                 cp_local_raw[:, fmap["link5"]],
                 cp_local_raw[:, fmap["gripper_link"]]), dim=1)
            f_ground_v = fm[:, fmap["support_plane"]]
            f_fixed_v = fm[:, fmap["link5"]]
            f_moving_v = fm[:, fmap["gripper_link"]]
            f_ground = f_ground_v.norm(dim=-1)
            f_fixed = f_fixed_v.norm(dim=-1)
            f_moving = f_moving_v.norm(dim=-1)
            fixed_ground_v = fixed_ground_sensor.data.force_matrix_w[:, 0, 0]
            moving_ground_v = moving_ground_sensor.data.force_matrix_w[:, 0, 0]
            fixed_ground_force = fixed_ground_v.norm(dim=-1)
            moving_ground_force = moving_ground_v.norm(dim=-1)
            acc["fixed_jaw_ground_max"] = torch.maximum(
                acc["fixed_jaw_ground_max"], fixed_ground_force)
            acc["moving_jaw_ground_max"] = torch.maximum(
                acc["moving_jaw_ground_max"], moving_ground_force)
            # Isaac Lab 2.3 defines max_contact_data_count_per_prim as a GLOBAL
            # capacity multiplier: cap * num_envs * num_sensor_bodies.  Preserve
            # the raw count peak so a full buffer cannot masquerade as low force.
            _bf, _bp, _bn, _bd, buffer_count, _starts = (
                sensor.contact_physx_view.get_contact_data(dt=sensor._sim_physics_dt))
            if tuple(int(v) for v in buffer_count.shape) != (n, 3):
                raise RuntimeError(
                    f"object raw contact-count shape drift expected={(n, 3)} "
                    f"actual={tuple(buffer_count.shape)}")
            acc["contact_data_total_peak"] = torch.maximum(
                acc["contact_data_total_peak"], buffer_count.sum().to(dtype=torch.int64))
            ground_count_by_jaw: list[Any] = []
            for ground_sensor, peak_name, per_env_peak_name in (
                    (fixed_ground_sensor, "fixed_ground_contact_data_total_peak",
                     "fixed_jaw_ground_raw_count_max"),
                    (moving_ground_sensor, "moving_ground_contact_data_total_peak",
                     "moving_jaw_ground_raw_count_max")):
                _gf, _gp, _gn, _gd, ground_count, _gs = (
                    ground_sensor.contact_physx_view.get_contact_data(
                        dt=ground_sensor._sim_physics_dt))
                if tuple(int(v) for v in ground_count.shape) != (n, 1):
                    raise RuntimeError(
                        f"jaw-ground raw contact-count shape drift expected={(n, 1)} "
                        f"actual={tuple(ground_count.shape)}")
                per_env_count = ground_count.reshape(n, -1).sum(dim=1).to(dtype=torch.int64)
                ground_count_by_jaw.append(per_env_count)
                acc[peak_name] = torch.maximum(
                    acc[peak_name], ground_count.sum().to(dtype=torch.int64))
                acc[per_env_peak_name] = torch.maximum(
                    acc[per_env_peak_name], per_env_count)

            if phase == "settle":
                acc["settle_support_fz"].append(f_ground_v[:, 2].clone())
            if phase in {"settle", "approach", "descend"}:
                acc["preclose_fixed_max"] = torch.maximum(acc["preclose_fixed_max"], f_fixed)
                acc["preclose_moving_max"] = torch.maximum(acc["preclose_moving_max"], f_moving)
            if phase == "close":
                if obj_ref is None:
                    raise RuntimeError("close phase began without descend-end object reference")
                acc["close_fixed_max"] = torch.maximum(acc["close_fixed_max"], f_fixed)
                acc["close_moving_max"] = torch.maximum(acc["close_moving_max"], f_moving)
                acc["close_bilateral_min_force_max"] = torch.maximum(
                    acc["close_bilateral_min_force_max"], torch.minimum(f_fixed, f_moving))
                acc["close_obj_drift_max"] = torch.maximum(
                    acc["close_obj_drift_max"], (obj_pos - obj_ref).norm(dim=-1))
            if phase == "lift":
                acc["lift_fixed_max"] = torch.maximum(acc["lift_fixed_max"], f_fixed)
                acc["lift_moving_max"] = torch.maximum(acc["lift_moving_max"], f_moving)
                acc["lift_bilateral_min_force_max"] = torch.maximum(
                    acc["lift_bilateral_min_force_max"], torch.minimum(f_fixed, f_moving))
            acc["contact_steps"] += ((f_fixed + f_moving) > CONTACT_EPS_N).float()
            acc["max_tilt_deg"] = torch.maximum(acc["max_tilt_deg"], tilt)

            if captures:
                idx = torch.as_tensor(captures, dtype=torch.long, device=device)
                force_stack = torch.stack((f_ground_v, f_fixed_v, f_moving_v), dim=1)
                jaw_ground_stack = torch.stack((fixed_ground_v, moving_ground_v), dim=1)
                jaw_ground_count_stack = torch.stack(ground_count_by_jaw, dim=1)
                # This sample is read after the just-completed PhysX step.
                trace_lists["physics_step"].append(step_global + 1)
                trace_lists["sim_time_s"].append((step_global + 1) * float(env.step_dt))
                trace_lists["phase_id"].append(phase_id)
                trace_lists["phase_step"].append(phase_step)
                for name, value in (
                    ("obj_pos_m", obj_pos), ("obj_quat_wxyz", obj_quat), ("tcp_pos_m", tcp),
                    ("link5_pos_m", link5_pos), ("link5_quat_wxyz", link5_quat),
                    ("moving_pos_m", moving_pos), ("moving_quat_wxyz", moving_quat),
                    ("q_actual_deg", torch.rad2deg(env._robot.data.joint_pos)),
                    ("q_target_deg", torch.rad2deg(target)), ("tilt_deg", tilt),
                    ("force_w_n", force_stack), ("contact_pos_m", cp_local),
                    ("jaw_ground_force_w_n", jaw_ground_stack),
                    ("jaw_ground_raw_count", jaw_ground_count_stack),
                ):
                    trace_lists[name].append(_tensor_numpy(value.index_select(0, idx)))

            if phase_step == nsteps - 1:
                if phase == "settle":
                    obj_rest_z = obj_pos[:, 2].clone()
                    obj_rest_tilt = tilt.clone()
                elif phase == "descend":
                    acc["descend_arrival_mm"] = (tcp-descend_tcp_ref).norm(dim=-1)*1000.0
                    acc["descend_q_err_deg"] = torch.rad2deg(
                        (env._robot.data.joint_pos[:, :5]-q_des[:, :5]).abs().max(dim=-1).values)
                    tcp_at_descend_end = tcp[:, 2].clone()
                    # Whole-close drift authority starts at the last descend step,
                    # before the first close command can move the cylinder.
                    obj_ref = obj_pos.clone()
                elif phase == "close":
                    acc["close_q5_err_deg"] = torch.rad2deg(
                        (env._robot.data.joint_pos[:, 5]-q5_close).abs())
                elif phase == "lift":
                    acc["lift_tcp_rise_mm"] = (tcp[:, 2]-tcp_at_descend_end)*1000.0
            step_global += 1

    obj_pos = env._sponge.data.root_pos_w - origins
    obj_quat = env._sponge.data.root_quat_w
    w, x, y, z = obj_quat.unbind(-1)
    final_tilt = torch.rad2deg(torch.acos(torch.clamp(1.0-2.0*(x*x+y*y), -1.0, 1.0)))

    def lowest(zc: Any, tilt_deg: Any) -> Any:
        tr = torch.deg2rad(tilt_deg)
        return zc - ((OBJ_HEIGHT_M/2.0)*torch.cos(tr) + (OBJ_DIAM_M/2.0)*torch.sin(tr))

    corrected_lift = (lowest(obj_pos[:, 2], final_tilt)-lowest(obj_rest_z, obj_rest_tilt))*1000.0
    raw_rise = (obj_pos[:, 2]-obj_rest_z)*1000.0
    settle_tail = torch.stack(acc["settle_support_fz"][-args.settle_stat_tail:], dim=0)
    settle_med = settle_tail.median(dim=0).values
    pc_lo = SETTLE_SUPPORT_FZ_N*(1.0-SETTLE_SUPPORT_TOL)
    pc_hi = SETTLE_SUPPORT_FZ_N*(1.0+SETTLE_SUPPORT_TOL)
    pc_per = (settle_med >= pc_lo) & (settle_med <= pc_hi)
    contact_capacity_total = int(args.contact_capacity) * n * int(sensor.num_bodies)
    contact_data_total_peak = int(acc["contact_data_total_peak"].item())
    fixed_ground_capacity_total = (
        int(args.contact_capacity) * n * int(fixed_ground_sensor.num_bodies))
    moving_ground_capacity_total = (
        int(args.contact_capacity) * n * int(moving_ground_sensor.num_bodies))
    fixed_ground_contact_data_total_peak = int(
        acc["fixed_ground_contact_data_total_peak"].item())
    moving_ground_contact_data_total_peak = int(
        acc["moving_ground_contact_data_total_peak"].item())
    object_buffer_saturated = contact_data_total_peak >= contact_capacity_total
    fixed_ground_buffer_saturated = (
        fixed_ground_contact_data_total_peak >= fixed_ground_capacity_total)
    moving_ground_buffer_saturated = (
        moving_ground_contact_data_total_peak >= moving_ground_capacity_total)
    contact_buffer_saturated = bool(
        object_buffer_saturated or fixed_ground_buffer_saturated
        or moving_ground_buffer_saturated)
    support_force_population_pass = bool(
        pc_per[:active_count].float().mean().item() >= 0.90)
    pc_batch = support_force_population_pass and not contact_buffer_saturated
    jaw_support_ok = ((acc["fixed_jaw_ground_max"] <= JAW_SUPPORT_GATE_N)
                      & (acc["moving_jaw_ground_max"] <= JAW_SUPPORT_GATE_N))
    # Positive-control/readout integrity decides whether the measurement is valid.
    # A reliable jaw/support collision is instead a measured task failure: retain it
    # in the denominator, forbid success, and classify it separately.
    measurement_valid = pc_per & pc_batch
    arrival = ((acc["descend_arrival_mm"] <= 3.0) & (acc["descend_q_err_deg"] <= 5.0)
               & (acc["close_q5_err_deg"] <= 3.0) & (acc["lift_tcp_rise_mm"] >= 15.0))
    preclose_ok = ((acc["preclose_fixed_max"] <= PRECLOSE_GATE_N)
                   & (acc["preclose_moving_max"] <= PRECLOSE_GATE_N))
    # Authority is simultaneous same-PhysX-step bilateral loading.  AND-ing two
    # independent phase maxima would falsely call staggered one-jaw contacts a
    # two-jaw grasp.
    both_close = acc["close_bilateral_min_force_max"] > JAW_LOAD_GATE_N
    both_lift = acc["lift_bilateral_min_force_max"] > JAW_LOAD_GATE_N
    success = (measurement_valid & jaw_support_ok & arrival & preclose_ok
               & both_close & both_lift
               & (corrected_lift > LIFT_GATE_MM) & (final_tilt < TIP_HALF_ANGLE_DEG))

    metrics_t = {
        "settle_support_fz_n": settle_med,
        "preclose_fixed_max_n": acc["preclose_fixed_max"],
        "preclose_moving_max_n": acc["preclose_moving_max"],
        "close_fixed_max_n": acc["close_fixed_max"],
        "close_moving_max_n": acc["close_moving_max"],
        "lift_fixed_max_n": acc["lift_fixed_max"],
        "lift_moving_max_n": acc["lift_moving_max"],
        "close_bilateral_min_force_max_n": acc["close_bilateral_min_force_max"],
        "lift_bilateral_min_force_max_n": acc["lift_bilateral_min_force_max"],
        "fixed_jaw_ground_force_max_n": acc["fixed_jaw_ground_max"],
        "moving_jaw_ground_force_max_n": acc["moving_jaw_ground_max"],
        "fixed_jaw_ground_raw_count_max": acc["fixed_jaw_ground_raw_count_max"],
        "moving_jaw_ground_raw_count_max": acc["moving_jaw_ground_raw_count_max"],
        "jaw_support_contact_pass": jaw_support_ok,
        "close_obj_drift_max_mm": acc["close_obj_drift_max"]*1000.0,
        "contact_steps": acc["contact_steps"], "max_tilt_deg": acc["max_tilt_deg"],
        "descend_arrival_mm": acc["descend_arrival_mm"],
        "descend_q_err_deg": acc["descend_q_err_deg"],
        "close_q5_err_deg": acc["close_q5_err_deg"],
        "lift_tcp_rise_mm": acc["lift_tcp_rise_mm"],
        "lift_corrected_mm": corrected_lift, "raw_rise_mm": raw_rise,
        "tilt_final_deg": final_tilt, "positive_control_env": pc_per,
        "positive_control_batch": torch.full(
            (n,), bool(pc_batch), dtype=torch.bool, device=env.device),
        "contact_buffer_saturated_batch": torch.full(
            (n,), bool(contact_buffer_saturated), dtype=torch.bool, device=env.device),
        "measurement_valid": measurement_valid,
        "arrival_pass": arrival, "preclose_pass": preclose_ok,
        "both_jaws_close": both_close, "both_jaws_lift": both_lift,
        "success": success,
    }
    metrics = {k: _tensor_numpy(v)[:active_count] for k, v in metrics_t.items()}
    labels: list[str] = []
    reason_flags: list[list[str]] = []
    for i in range(active_count):
        reasons: list[str] = []
        if not pc_batch:
            reasons.append("BATCH_POSITIVE_CONTROL_OR_CONTACT_BUFFER_INVALID")
        if not bool(metrics["positive_control_env"][i]):
            reasons.append("SETTLE_SUPPORT_FORCE_CONTROL_FAIL")
        if not bool(metrics["jaw_support_contact_pass"][i]):
            reasons.append("JAW_SUPPORT_CONTACT_OBSERVED_GT_0P02N")
        if not pc_batch or not bool(metrics["positive_control_env"][i]):
            label = "MEASUREMENT_INVALID"
        elif not bool(metrics["jaw_support_contact_pass"][i]):
            label = "JAW_SUPPORT_CONTACT_FAIL"
        elif not bool(metrics["arrival_pass"][i]):
            label = "ARRIVAL_FAIL"
        elif not bool(metrics["preclose_pass"][i]):
            label = "PRECLOSE_COLLISION"
        elif not bool(metrics["both_jaws_close"][i]):
            fixed_ever = metrics["close_fixed_max_n"][i] > JAW_LOAD_GATE_N
            moving_ever = metrics["close_moving_max_n"][i] > JAW_LOAD_GATE_N
            if fixed_ever and moving_ever:
                label = "STAGGERED_JAW_CONTACT"
            elif fixed_ever or moving_ever:
                label = "ONE_JAW_ONLY"
            else:
                label = "NO_JAW_CONTACT"
        elif not bool(metrics["both_jaws_lift"][i]):
            label = "BILATERAL_LOST_IN_LIFT"
        elif metrics["tilt_final_deg"][i] >= TIP_HALF_ANGLE_DEG:
            label = "TIPPED_NOT_LIFTED"
        elif bool(metrics["success"][i]):
            label = "PHYSICS_LIFT_SUCCESS"
        else:
            label = "BOTH_JAWS_NO_LIFT"
        labels.append(label)
        reason_flags.append(reasons)

    trace: dict[str, np.ndarray] = {}
    if captures:
        for name, values in trace_lists.items():
            trace[name] = np.asarray(values)
        trace["capture_slot"] = np.asarray(captures, dtype=np.int64)
        trace["trial_index"] = np.asarray([samples[i]["trial_index"] for i in captures], dtype=np.int64)
    return {
        "active_count": active_count, "wall_physics_s": time.time()-t0,
        "total_steps": step_global,
        "filter_map": {
            "object_contact": fmap,
            "fixed_jaw_ground": fixed_ground_map,
            "moving_jaw_ground": moving_ground_map,
        },
        "kinematic_attach_calls": attach_calls["n"],
        "positive_control": {
            "pass": pc_batch, "expected_fz_n": SETTLE_SUPPORT_FZ_N,
            "support_force_population_pass": support_force_population_pass,
            "band_n": [pc_lo, pc_hi], "envs_in_band": int(pc_per[:active_count].sum().item()),
            "envs_total": active_count,
            "median_fz_n": float(torch.median(settle_med[:active_count]).item()),
            "contact_capacity_per_prim": int(args.contact_capacity),
            "contact_capacity_total": contact_capacity_total,
            "contact_data_total_peak": contact_data_total_peak,
            "object_contact_buffer_saturated": object_buffer_saturated,
            "fixed_ground_contact_capacity_total": fixed_ground_capacity_total,
            "fixed_ground_contact_data_total_peak": fixed_ground_contact_data_total_peak,
            "fixed_ground_contact_buffer_saturated": fixed_ground_buffer_saturated,
            "moving_ground_contact_capacity_total": moving_ground_capacity_total,
            "moving_ground_contact_data_total_peak": moving_ground_contact_data_total_peak,
            "moving_ground_contact_buffer_saturated": moving_ground_buffer_saturated,
            "contact_buffer_saturated": contact_buffer_saturated,
        },
        "metrics": metrics, "mechanism": np.asarray(labels, dtype="U64"),
        "reason_flags": reason_flags, "trace": trace,
    }


def _pad_batch(rows: list[dict[str, Any]], n: int) -> tuple[list[dict[str, Any]], int]:
    if not rows:
        raise RuntimeError("cannot pad empty batch")
    active = len(rows)
    out = [dict(r) for r in rows]
    while len(out) < n:
        pad = dict(rows[(len(out)-active) % active])
        pad["_padding"] = True
        out.append(pad)
    return out, active


def _concat_metrics(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(parts[0])
    if any(set(part) != keys for part in parts):
        raise RuntimeError("batch metric schema drift")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in sorted(keys)}


def _aggregate(samples: list[dict[str, Any]], metrics: dict[str, np.ndarray],
               mechanism: np.ndarray, key: str) -> dict[str, Any]:
    values = np.asarray([str(row[key]) for row in samples])
    out: dict[str, Any] = {}
    for value in sorted(set(values.tolist())):
        mask = values == value
        valid = mask & metrics["measurement_valid"]
        task_clear = valid & metrics["jaw_support_contact_pass"]
        support_fail = valid & ~metrics["jaw_support_contact_pass"]
        support_fail_bilateral = support_fail & (
            metrics["both_jaws_close"] | metrics["both_jaws_lift"])
        validity = _population_validity_summary(
            metrics["success"][mask], metrics["measurement_valid"][mask], [True])
        n_valid = int(validity["n_valid"])
        success = int(validity["valid_success"])
        success_rate = validity["valid_success_rate"]
        lift = metrics["lift_corrected_mm"][task_clear]
        lift_valid_diagnostic = metrics["lift_corrected_mm"][valid]
        out[value] = {
            "n": int(mask.sum()), "n_valid": n_valid,
            "n_invalid": int(mask.sum()) - n_valid,
            "success": success,
            "success_rate": success_rate,
            "success_rate_denominator": "n_valid",
            "success_rate_valid_denominator": success_rate,
            "both_jaws_close": int((metrics["both_jaws_close"] & task_clear).sum()),
            "both_jaws_lift": int((metrics["both_jaws_lift"] & task_clear).sum()),
            "both_jaws_counts_scope": "measurement_valid_and_jaw_support_clear",
            "both_jaws_close_raw_diagnostic": int(metrics["both_jaws_close"][mask].sum()),
            "both_jaws_lift_raw_diagnostic": int(metrics["both_jaws_lift"][mask].sum()),
            "n_jaw_support_fail": int(support_fail.sum()),
            "n_jaw_support_observed_all_diagnostic": int(
                ((~metrics["jaw_support_contact_pass"]) & mask).sum()),
            "n_jaw_support_fail_with_bilateral": int(support_fail_bilateral.sum()),
            "task_clear": {
                "n": int(task_clear.sum()),
                "success": int((metrics["success"] & task_clear).sum()),
                "both_jaws_close": int((metrics["both_jaws_close"] & task_clear).sum()),
                "both_jaws_lift": int((metrics["both_jaws_lift"] & task_clear).sum()),
            },
            "jaw_ground_force_max_n_all_diagnostic": {
                "fixed": float(np.max(metrics["fixed_jaw_ground_force_max_n"][mask])),
                "moving": float(np.max(metrics["moving_jaw_ground_force_max_n"][mask])),
            },
            "jaw_ground_raw_count_per_env_max_all_diagnostic": {
                "fixed": int(np.max(metrics["fixed_jaw_ground_raw_count_max"][mask])),
                "moving": int(np.max(metrics["moving_jaw_ground_raw_count_max"][mask])),
            },
            "lift_corrected_mm": {
                "median": None if not len(lift) else float(np.median(lift)),
                "p90": None if not len(lift) else float(np.percentile(lift, 90)),
                "max": None if not len(lift) else float(np.max(lift)),
            },
            "lift_corrected_mm_scope": "measurement_valid_and_jaw_support_clear",
            "lift_corrected_mm_measured_valid_all_diagnostic": {
                "median": (None if not len(lift_valid_diagnostic)
                           else float(np.median(lift_valid_diagnostic))),
                "max": (None if not len(lift_valid_diagnostic)
                        else float(np.max(lift_valid_diagnostic))),
            },
            "lift_corrected_mm_raw_max_diagnostic": float(np.max(metrics["lift_corrected_mm"][mask])),
            "mechanism_counts_valid": {label: int(np.sum(mechanism[valid] == label))
                                       for label in sorted(set(mechanism[valid].tolist()))},
            "mechanism_counts_all_diagnostic": {
                label: int(np.sum(mechanism[mask] == label))
                for label in sorted(set(mechanism[mask].tolist()))},
        }
    return out


def _aggregate_theta_q5(samples: list[dict[str, Any]], metrics: dict[str, np.ndarray],
                        mechanism: np.ndarray) -> dict[str, Any]:
    """Direct, lossless grouping by the exact deterministic physics control."""
    groups: dict[tuple[float, float, str, float, str, str], list[int]] = {}
    for idx, row in enumerate(samples):
        key = (
            float(row["theta_target_deg"]), float(row["q5_close_deg"]),
            str(row["window_kind"]), float(row["grasp_surface_margin_m"]),
            str(row["descend_margin_source"]), str(row["q5_control_source"]),
        )
        if not all(math.isfinite(value) for value in (key[0], key[1], key[3])):
            raise RuntimeError(f"non-finite theta/q5/margin summary key at trial={idx}: {key}")
        groups.setdefault(key, []).append(idx)

    out: dict[str, Any] = {}
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5])):
        theta, q5, kind, margin, margin_source, q5_source = key
        indices = np.asarray(groups[key], dtype=np.int64)
        mask = np.zeros(len(samples), dtype=bool)
        mask[indices] = True
        valid = mask & metrics["measurement_valid"]
        task_clear = valid & metrics["jaw_support_contact_pass"]
        support_fail = valid & ~metrics["jaw_support_contact_pass"]
        support_fail_bilateral = support_fail & (
            metrics["both_jaws_close"] | metrics["both_jaws_lift"])
        validity = _population_validity_summary(
            metrics["success"][mask], metrics["measurement_valid"][mask], [True])

        def stats(metric_name: str, selection: np.ndarray = task_clear) -> dict[str, float | None]:
            values = np.asarray(metrics[metric_name][selection], dtype=np.float64)
            if not len(values):
                return {"min": None, "median": None, "max": None}
            return {
                "min": float(np.min(values)), "median": float(np.median(values)),
                "max": float(np.max(values)),
            }

        key_fields = {
            "theta_target_deg": theta, "theta_target_deg_float64_hex": theta.hex(),
            "q5_close_deg": q5, "q5_close_deg_float64_hex": q5.hex(),
            "window_kind": kind,
            "grasp_surface_margin_m": margin,
            "grasp_surface_margin_m_float64_hex": margin.hex(),
            "descend_margin_source": margin_source,
            "q5_control_source": q5_source,
        }
        group_key = json.dumps(key_fields, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if group_key in out:
            raise RuntimeError(f"duplicate exact theta-q5 summary key: {group_key}")
        n_valid = int(validity["n_valid"])
        out[group_key] = {
            "key": key_fields,
            "n": int(mask.sum()), "n_valid": n_valid,
            "n_invalid": int(mask.sum()) - n_valid,
            "success": int(validity["valid_success"]),
            "success_rate": validity["valid_success_rate"],
            "success_rate_denominator": "n_valid",
            "both_jaws_close": int((metrics["both_jaws_close"] & task_clear).sum()),
            "both_jaws_lift": int((metrics["both_jaws_lift"] & task_clear).sum()),
            "both_jaws_counts_scope": "measurement_valid_and_jaw_support_clear",
            "n_jaw_support_fail": int(support_fail.sum()),
            "n_jaw_support_observed_all_diagnostic": int(
                ((~metrics["jaw_support_contact_pass"]) & mask).sum()),
            "n_jaw_support_fail_with_bilateral": int(support_fail_bilateral.sum()),
            "task_clear": {
                "n": int(task_clear.sum()),
                "success": int((metrics["success"] & task_clear).sum()),
                "both_jaws_close": int((metrics["both_jaws_close"] & task_clear).sum()),
                "both_jaws_lift": int((metrics["both_jaws_lift"] & task_clear).sum()),
            },
            "same_step_minforce_authority_n": {
                "close": stats("close_bilateral_min_force_max_n"),
                "lift": stats("lift_bilateral_min_force_max_n"),
            },
            "same_step_minforce_authority_scope": (
                "measurement_valid_and_jaw_support_clear"),
            "same_step_minforce_authority_n_all_diagnostic": {
                "close": stats("close_bilateral_min_force_max_n", mask),
                "lift": stats("lift_bilateral_min_force_max_n", mask),
            },
            "jaw_ground_force_max_n_all_diagnostic": {
                "fixed": {
                    "max": float(np.max(metrics["fixed_jaw_ground_force_max_n"][mask]))},
                "moving": {
                    "max": float(np.max(metrics["moving_jaw_ground_force_max_n"][mask]))},
            },
            "jaw_ground_raw_count_per_env_max_all_diagnostic": {
                "fixed": {
                    "max": int(np.max(metrics["fixed_jaw_ground_raw_count_max"][mask]))},
                "moving": {
                    "max": int(np.max(metrics["moving_jaw_ground_raw_count_max"][mask]))},
            },
            "mechanism_counts_valid": {
                label: int(np.sum(mechanism[valid] == label))
                for label in sorted(set(mechanism[valid].tolist()))},
            "mechanism_counts_all_diagnostic": {
                label: int(np.sum(mechanism[mask] == label))
                for label in sorted(set(mechanism[mask].tolist()))},
        }
    if sum(row["n"] for row in out.values()) != len(samples):
        raise RuntimeError("theta-q5 summary partition does not cover every feasible trial exactly once")
    return out


def _theta_q5_summary_regression_smoke() -> dict[str, Any]:
    samples = [{
        "theta_target_deg": 24.0, "q5_close_deg": 19.9,
        "window_kind": "unilateral_negative_control",
        "grasp_surface_margin_m": -6.172366263008969e-06,
        "descend_margin_source": "p13_same_theta_q5_curve",
        "q5_control_source": "p13_interior_target",
    }] * 2
    metrics = {
        "success": np.asarray([True, False]),
        "measurement_valid": np.asarray([True, True]),
        "both_jaws_close": np.asarray([True, True]),
        "both_jaws_lift": np.asarray([True, False]),
        "jaw_support_contact_pass": np.asarray([True, False]),
        "close_bilateral_min_force_max_n": np.asarray([0.012, 9.0]),
        "lift_bilateral_min_force_max_n": np.asarray([0.011, 8.0]),
        "fixed_jaw_ground_force_max_n": np.asarray([0.0, 4.0]),
        "moving_jaw_ground_force_max_n": np.asarray([0.0, 5.0]),
        "fixed_jaw_ground_raw_count_max": np.asarray([0, 7]),
        "moving_jaw_ground_raw_count_max": np.asarray([0, 9]),
    }
    result = _aggregate_theta_q5(
        samples, metrics, np.asarray(["PHYSICS_LIFT_SUCCESS", "JAW_SUPPORT_CONTACT_FAIL"]))
    if len(result) != 1:
        raise RuntimeError(f"THETA_Q5_GROUP_COUNT_REGRESSION groups={len(result)}")
    row = next(iter(result.values()))
    if not (row["n"] == 2 and row["n_valid"] == 2 and row["n_invalid"] == 0
            and row["success"] == 1 and row["both_jaws_close"] == 1
            and row["n_jaw_support_fail"] == 1
            and row["n_jaw_support_fail_with_bilateral"] == 1
            and row["same_step_minforce_authority_n"]["close"]["max"] == 0.012):
        raise RuntimeError(f"THETA_Q5_SUMMARY_SEMANTICS_REGRESSION row={row}")
    return {"pass": True, "group_count": 1, "row": row}


def choose_representatives(samples: list[dict[str, Any]], metrics: dict[str, np.ndarray],
                           mechanism: np.ndarray, limit: int = 10) -> list[int]:
    chosen: list[int] = []

    def add(idx: int | None) -> None:
        if idx is not None and idx not in chosen and len(chosen) < limit:
            chosen.append(int(idx))

    indices = np.arange(len(samples))
    for form in ("near_top_down", "oblique", "high_tilt"):
        mask = np.asarray([row["form"] == form for row in samples])
        if mask.any():
            task_clear = (mask & metrics["measurement_valid"]
                          & metrics["jaw_support_contact_pass"] & metrics["arrival_pass"])
            measured = mask & metrics["measurement_valid"] & metrics["arrival_pass"]
            pool = indices[task_clear if task_clear.any() else (measured if measured.any() else mask)]
            add(int(pool[np.argmax(metrics["lift_corrected_mm"][pool])]))
    jaw_support = np.maximum(metrics["fixed_jaw_ground_force_max_n"],
                             metrics["moving_jaw_ground_force_max_n"])
    if float(np.max(jaw_support)) > JAW_SUPPORT_GATE_N:
        add(int(np.argmax(jaw_support)))
    if bool(metrics["success"].any()):
        pool = indices[metrics["success"]]
        add(int(pool[np.argmax(metrics["lift_corrected_mm"][pool])]))
    mask = (metrics["measurement_valid"] & metrics["jaw_support_contact_pass"]
            & metrics["both_jaws_close"] & ~metrics["success"])
    if mask.any():
        pool = indices[mask]
        add(int(pool[np.argmax(metrics["lift_corrected_mm"][pool])]))
    # Lift-only contact is a distinct global verdict branch; select its decision
    # subject explicitly so D341 cannot omit it while still claiming the branch.
    mask = (metrics["measurement_valid"] & metrics["jaw_support_contact_pass"]
            & metrics["both_jaws_lift"] & ~metrics["both_jaws_close"])
    if mask.any():
        pool = indices[mask]
        score = metrics["lift_bilateral_min_force_max_n"][pool]
        add(int(pool[np.argmax(score)]))
    mask = np.asarray([row["window_kind"] in {"unilateral_negative_control", "no_window"}
                       for row in samples])
    if mask.any():
        pool = indices[mask]
        add(int(pool[np.argmax(metrics["lift_corrected_mm"][pool])]))
    mask = np.asarray(["negative_control" in row["window_kind"] for row in samples])
    if mask.any():
        pool = indices[mask]
        add(int(pool[np.argmax(metrics["lift_corrected_mm"][pool])]))
    pre = metrics["preclose_fixed_max_n"] + metrics["preclose_moving_max_n"]
    add(int(np.argmax(pre)))
    return chosen


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / max(float(np.linalg.norm(q)), 1.0e-12)
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _rr_quaternion(rr: Any, q_wxyz: Any) -> Any:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return rr.Quaternion(xyzw=[float(q[1]), float(q[2]), float(q[3]), float(q[0])])


def _cylinder_mesh(segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    half = OBJ_HEIGHT_M / 2.0
    for z in (-half, half):
        for i in range(segments):
            a = 2.0*math.pi*i/segments
            vertices.append([OBJ_DIAM_M*0.5*math.cos(a), OBJ_DIAM_M*0.5*math.sin(a), z])
    vertices += [[0.0, 0.0, -half], [0.0, 0.0, half]]
    triangles: list[list[int]] = []
    for i in range(segments):
        j = (i+1) % segments
        triangles += [[i, j, segments+j], [i, segments+j, segments+i],
                      [2*segments, j, i], [2*segments+1, segments+i, segments+j]]
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)


def emit_decision_snapshot(path: Path, p10: Any, reps: list[dict[str, Any]],
                           trace: dict[str, np.ndarray], verdict: str,
                           run_profile: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl import viz_debug

    frames: list[dict[str, Any]] = []
    last = -1
    for col, row in enumerate(reps):
        d, u, v = p10.axis_frame(row["theta_target_deg"], row["psi_axis_target_deg"])
        ph = math.radians(row["phi_target_deg"])
        x_target = math.cos(ph)*u + math.sin(ph)*v
        frames.append(viz_debug.frame_from_axes(
            f"target_{col}", row["descend_tcp_m"], x_axis=x_target, z_axis=d,
            role="target", label=f"target {row['form']} θ={row['theta_target_deg']:.0f}",
        ))
        actual_quat = trace["link5_quat_wxyz"][last, col]
        frames.append({
            "name": f"actual_{col}", "label": f"actual replay {row['trial_id']}",
            "position": trace["tcp_pos_m"][last, col].tolist(),
            "quat_wxyz": actual_quat.tolist(), "role": "actual", "metadata": {},
        })
        frames.append({
            "name": f"object_{col}", "label": f"D29xH50 {row['trial_id']}",
            "position": trace["obj_pos_m"][last, col].tolist(),
            "quat_wxyz": trace["obj_quat_wxyz"][last, col].tolist(),
            "role": "object", "metadata": {},
        })
    authoritative = bool(run_profile["scientific_authoritative"])
    scope_label = ("CANONICAL_SCIENTIFIC_WORKSPACE"
                   if authoritative else "INSTRUMENTATION_PREFLIGHT_ONLY")
    status = viz_debug.snapshot(
        path, pairs=frames, prefer_viewport=False,
        title=(f"t3y {run_profile['run_label']} — {scope_label}: "
               "representative replay target vs actual"),
        annotations=[scope_label, f"diagnostic_branch={verdict}",
                     f"scientific_authoritative={str(authoritative).lower()}",
                     "D29 x H50; grasp point remains top-centre",
                     "frame positions/orientations are final replay samples"],
    )
    if not status.get("ok") or not path.is_file():
        raise RuntimeError(f"viz_debug decision snapshot failed: {status}")
    return status


def emit_rerun(paths: dict[str, Path], p10: Any, jaw_clouds: dict[str, Any],
               reps: list[dict[str, Any]],
               trace: dict[str, np.ndarray], summary: dict[str, Any],
               verdict: str, run_profile: dict[str, Any]) -> dict[str, Any]:
    """Write the full representative physics timeline and validate D341 mechanics."""
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"rerun pin mismatch {rr.__version__} != {RERUN_VERSION}")
    vertices, triangles = _cylinder_mesh()
    phase_names = ("settle", "approach", "descend", "close", "hold", "lift")
    expected: set[str] = set()

    def remember(path: str) -> str:
        expected.add(path.strip("/"))
        return path

    authoritative = bool(run_profile["scientific_authoritative"])
    scope_label = ("CANONICAL_SCIENTIFIC_WORKSPACE"
                   if authoritative else "INSTRUMENTATION_PREFLIGHT_ONLY")
    summary_md = (
        f"# p14 / t3y {run_profile['run_label']} — {scope_label}\n\n"
        f"**scientific authoritative:** `{str(authoritative).lower()}`  \n"
        f"**diagnostic workspace branch:** `{verdict}`  \n"
        f"**population trials:** {summary['all']['n']}  "
        f"**valid / invalid:** {summary['all']['n_valid']} / {summary['all']['n_invalid']}  \n"
        f"**valid successes:** {summary['all']['success']} / {summary['all']['n_valid']}  \n"
        f"**representative replays:** {len(reps)}  \n\n"
        "D29 x H50 upright cylinder; top-centre grasp point; attempt3 64+64 split "
        "convex-hull jaws; kinematic attach disabled. Native callback NPZ/JSON is "
        "the gate authority; Rerun spatial values are inspection copies. "
        f"Jaw clouds: {jaw_clouds['view_stride']}x view stride from "
        f"{jaw_clouds['surface_sample_spacing_m']*1000.0:.3f} mm hull-surface "
        f"samples, capped at {jaw_clouds['view_max_points_per_body']} points/body.\n"
    )
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/metadata", contents="/metadata/run", name="1 | run"),
                rrb.TextLogView(origin="/events", contents="/events/**", name="2 | events"),
                column_shares=[0.62, 0.38],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", contents="/replay/**", name="3 | target, object, jaws, contacts"),
                rrb.TimeSeriesView(origin="/metrics", contents="/metrics/**", name="4 | force, pose, q5"),
                column_shares=[0.55, 0.45],
            ),
            row_shares=[0.30, 0.70],
        ), auto_layout=False, auto_views=False, collapse_panels=True,
    )
    app_id = f"roarm_g0b_t3y_{run_profile['run_label']}"
    recording_id = f"g0b_d420_t3y_{run_profile['run_label']}"
    with rr.RecordingStream(app_id, recording_id=recording_id,
                            make_default=False, send_properties=True) as rec:
        rec.save(str(paths["timeline.rrd"]), write_footer=True)
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.log(remember("metadata/run"), rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        rec.log(remember("metadata/profile"), rr.TextDocument(
            json.dumps(run_profile, sort_keys=True), media_type=rr.MediaType.TEXT), static=True)
        rec.log(remember("events/run"), rr.TextLog(
            f"start; scope={scope_label}; diagnostic_branch={verdict}",
            level=rr.TextLogLevel.INFO))
        rec.log(remember("transforms/world"), rr.Transform3D(
            translation=[0.0, 0.0, 0.0], rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            parent_frame="tf#/", child_frame="world"), static=True)

        for col, row in enumerate(reps):
            rid = f"rep_{col:02d}_{row['trial_id']}"
            base = f"replay/{rid}"
            rec.log(remember(f"{base}/geometry/cylinder"), rr.Mesh3D(
                vertex_positions=vertices, triangle_indices=triangles,
                albedo_factor=[225, 165, 65, 190]),
                rr.CoordinateFrame(f"{rid}/actual/cylinder"), static=True)
            rec.log(remember(f"{base}/geometry/link5_collision"), rr.Points3D(
                jaw_clouds["clouds"]["link5"], colors=[[70, 145, 245]], radii=[0.00035]),
                rr.CoordinateFrame(f"{rid}/actual/link5"), static=True)
            rec.log(remember(f"{base}/geometry/gripper_collision"), rr.Points3D(
                jaw_clouds["clouds"]["gripper_link"], colors=[[235, 65, 105]], radii=[0.00035]),
                rr.CoordinateFrame(f"{rid}/actual/gripper_link"), static=True)
            d, u, v = p10.axis_frame(row["theta_target_deg"], row["psi_axis_target_deg"])
            target = np.asarray(row["descend_tcp_m"], dtype=np.float32)
            rec.log(remember(f"{base}/target/tcp"), rr.Points3D(
                [target], colors=[[240, 45, 35]], radii=[0.003], labels=["target TCP"]),
                rr.CoordinateFrame("world"), static=True)
            rec.log(remember(f"{base}/target/tool_axis"), rr.Arrows3D(
                origins=[target], vectors=[(np.asarray(d)*0.045).astype(np.float32)],
                colors=[[240, 45, 35]], radii=[0.0008], labels=["target link5 +z"]),
                rr.CoordinateFrame("world"), static=True)

        nsteps = int(trace["physics_step"].shape[0])
        last_phase = None
        for step_i in range(nsteps):
            step = int(trace["physics_step"][step_i])
            sim_time = float(trace["sim_time_s"][step_i])
            phase_id = int(trace["phase_id"][step_i])
            rec.reset_time()
            rec.set_time("physics_step", sequence=step)
            rec.set_time("sim_time_s", duration=sim_time)
            rec.log(remember("metrics/phase_id"), rr.Scalars([float(phase_id)]))
            if phase_id != last_phase:
                rec.log(remember("events/phase"), rr.TextLog(
                    f"phase={phase_names[phase_id]} starts at physics_step={step}",
                    level=rr.TextLogLevel.INFO))
                last_phase = phase_id
            for col, row in enumerate(reps):
                rid = f"rep_{col:02d}_{row['trial_id']}"
                base = f"replay/{rid}"
                obj_pos = trace["obj_pos_m"][step_i, col]
                obj_q = trace["obj_quat_wxyz"][step_i, col]
                rec.log(remember(f"{base}/transforms/cylinder"), rr.Transform3D(
                    translation=obj_pos, rotation=_rr_quaternion(rr, obj_q),
                    parent_frame="world", child_frame=f"{rid}/actual/cylinder"))
                rec.log(remember(f"{base}/transforms/link5"), rr.Transform3D(
                    translation=trace["link5_pos_m"][step_i, col],
                    rotation=_rr_quaternion(rr, trace["link5_quat_wxyz"][step_i, col]),
                    parent_frame="world", child_frame=f"{rid}/actual/link5"))
                rec.log(remember(f"{base}/transforms/gripper_link"), rr.Transform3D(
                    translation=trace["moving_pos_m"][step_i, col],
                    rotation=_rr_quaternion(rr, trace["moving_quat_wxyz"][step_i, col]),
                    parent_frame="world", child_frame=f"{rid}/actual/gripper_link"))
                tcp = trace["tcp_pos_m"][step_i, col]
                link_q = trace["link5_quat_wxyz"][step_i, col]
                axis = _quat_to_rot(link_q)[:, 2]
                rec.log(remember(f"{base}/actual/tcp"), rr.Points3D(
                    [tcp], colors=[[40, 165, 245]], radii=[0.0025]), rr.CoordinateFrame("world"))
                rec.log(remember(f"{base}/actual/tool_axis"), rr.Arrows3D(
                    origins=[tcp], vectors=[axis*0.045], colors=[[40, 165, 245]],
                    radii=[0.0007]), rr.CoordinateFrame("world"))
                rec.log(remember(f"{base}/actual/jaw_origins"), rr.Points3D(
                    [trace["link5_pos_m"][step_i, col], trace["moving_pos_m"][step_i, col]],
                    colors=[[205, 55, 205], [35, 210, 100]], radii=[0.002, 0.002],
                    labels=["fixed jaw body", "moving jaw body"]), rr.CoordinateFrame("world"))
                for jidx, (label, origin) in enumerate((
                        ("fixed_jaw_support", trace["link5_pos_m"][step_i, col]),
                        ("moving_jaw_support", trace["moving_pos_m"][step_i, col]))):
                    force = trace["jaw_ground_force_w_n"][step_i, col, jidx]
                    entity = remember(f"{base}/contacts/{label}/force")
                    if float(np.linalg.norm(force)) > 0.0:
                        rec.log(entity, rr.Arrows3D(
                            origins=[origin], vectors=[np.asarray(force)*0.005], radii=[0.0006],
                            labels=[f"{label} force at jaw origin x0.005 m/N"]),
                            rr.CoordinateFrame("world"))
                    else:
                        rec.log(entity, rr.Arrows3D(
                            origins=np.empty((0, 3), dtype=np.float32),
                            vectors=np.empty((0, 3), dtype=np.float32)),
                            rr.CoordinateFrame("world"))
                for fidx, label in enumerate(("support", "fixed_jaw", "moving_jaw")):
                    force = trace["force_w_n"][step_i, col, fidx]
                    point = trace["contact_pos_m"][step_i, col, fidx]
                    if np.isfinite(point).all() and float(np.linalg.norm(force)) > 0.0:
                        rec.log(remember(f"{base}/contacts/{label}/point"), rr.Points3D(
                            [point], radii=[0.0018], labels=[f"{label} averaged contact"]),
                            rr.CoordinateFrame("world"))
                        rec.log(remember(f"{base}/contacts/{label}/force"), rr.Arrows3D(
                            origins=[point], vectors=[np.asarray(force)*0.005], radii=[0.0006],
                            labels=[f"normal force x0.005 m/N"]), rr.CoordinateFrame("world"))
                    else:
                        # Empty archetypes explicitly clear latest-at state; without
                        # this, a past contact point/arrow would appear to persist
                        # visually through later no-contact physics steps.
                        rec.log(remember(f"{base}/contacts/{label}/point"), rr.Points3D(
                            np.empty((0, 3), dtype=np.float32)), rr.CoordinateFrame("world"))
                        rec.log(remember(f"{base}/contacts/{label}/force"), rr.Arrows3D(
                            origins=np.empty((0, 3), dtype=np.float32),
                            vectors=np.empty((0, 3), dtype=np.float32)), rr.CoordinateFrame("world"))
                prefix = f"metrics/{rid}"
                values = {
                    "object_z_mm": float(obj_pos[2]*1000.0),
                    "object_tilt_deg": float(trace["tilt_deg"][step_i, col]),
                    "q5_actual_deg": float(trace["q_actual_deg"][step_i, col, 5]),
                    "q5_target_deg": float(trace["q_target_deg"][step_i, col, 5]),
                    "support_force_n": float(np.linalg.norm(trace["force_w_n"][step_i, col, 0])),
                    "fixed_force_n": float(np.linalg.norm(trace["force_w_n"][step_i, col, 1])),
                    "moving_force_n": float(np.linalg.norm(trace["force_w_n"][step_i, col, 2])),
                    "fixed_jaw_support_force_n": float(
                        np.linalg.norm(trace["jaw_ground_force_w_n"][step_i, col, 0])),
                    "moving_jaw_support_force_n": float(
                        np.linalg.norm(trace["jaw_ground_force_w_n"][step_i, col, 1])),
                    "fixed_jaw_support_raw_count": float(
                        trace["jaw_ground_raw_count"][step_i, col, 0]),
                    "moving_jaw_support_raw_count": float(
                        trace["jaw_ground_raw_count"][step_i, col, 1]),
                    "tcp_target_error_mm": float(np.linalg.norm(tcp-np.asarray(row["descend_tcp_m"]))*1000.0),
                }
                for name, value in values.items():
                    rec.log(remember(f"{prefix}/{name}"), rr.Scalars([value]))
        rec.log(remember("events/run"), rr.TextLog(
            f"end; scope={scope_label}; diagnostic_branch={verdict}",
            level=rr.TextLogLevel.INFO))
        rec.flush(timeout_sec=120.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    components = {
        "metadata/run": ["TextDocument:text"],
        "metadata/profile": ["TextDocument:text"],
        "events/run": ["TextLog:level", "TextLog:text"],
        "events/phase": ["TextLog:level", "TextLog:text"],
        "transforms/world": [
            "Transform3D:translation", "Transform3D:quaternion",
            "Transform3D:parent_frame", "Transform3D:child_frame",
        ],
        "metrics/phase_id": ["Scalars:scalars"],
    }
    for col, row in enumerate(reps):
        rid = f"rep_{col:02d}_{row['trial_id']}"
        components[f"replay/{rid}/geometry/cylinder"] = ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"]
        components[f"replay/{rid}/geometry/link5_collision"] = [
            "Points3D:positions", "CoordinateFrame:frame"]
        components[f"replay/{rid}/geometry/gripper_collision"] = [
            "Points3D:positions", "CoordinateFrame:frame"]
        components[f"replay/{rid}/target/tcp"] = ["Points3D:positions"]
        components[f"replay/{rid}/target/tool_axis"] = ["Arrows3D:origins", "Arrows3D:vectors"]
        components[f"replay/{rid}/actual/tcp"] = ["Points3D:positions"]
        components[f"replay/{rid}/actual/tool_axis"] = ["Arrows3D:origins", "Arrows3D:vectors"]
        transform_components = [
            "Transform3D:translation", "Transform3D:quaternion",
            "Transform3D:parent_frame", "Transform3D:child_frame",
        ]
        components[f"replay/{rid}/transforms/cylinder"] = transform_components
        components[f"replay/{rid}/transforms/link5"] = transform_components
        components[f"replay/{rid}/transforms/gripper_link"] = transform_components
        for label in ("support", "fixed_jaw", "moving_jaw"):
            components[f"replay/{rid}/contacts/{label}/point"] = ["Points3D:positions"]
            components[f"replay/{rid}/contacts/{label}/force"] = ["Arrows3D:origins", "Arrows3D:vectors"]
        for label in ("fixed_jaw_support", "moving_jaw_support"):
            components[f"replay/{rid}/contacts/{label}/force"] = [
                "Arrows3D:origins", "Arrows3D:vectors", "CoordinateFrame:frame"]
        components[f"metrics/{rid}/q5_actual_deg"] = ["Scalars:scalars"]
        components[f"metrics/{rid}/fixed_force_n"] = ["Scalars:scalars"]
        components[f"metrics/{rid}/fixed_jaw_support_force_n"] = ["Scalars:scalars"]
        components[f"metrics/{rid}/moving_jaw_support_force_n"] = ["Scalars:scalars"]
        components[f"metrics/{rid}/fixed_jaw_support_raw_count"] = ["Scalars:scalars"]
        components[f"metrics/{rid}/moving_jaw_support_raw_count"] = ["Scalars:scalars"]

    validation = validate_rerun_artifact(
        paths["timeline.rrd"], expected_entity_paths=sorted(expected),
        exact_entity_paths=sorted(expected),
        expected_timeline_names=["physics_step", "sim_time_s"],
        exact_timeline_names=["blueprint", "log_time", "physics_step", "sim_time_s"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"], screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400", expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI, timeout_s=600.0,
    )
    hull_gate = {
        body: {
            "all_64_parts_hull_surface_sampled": bool(
                jaw_clouds["bodies"][body]["all_64_parts_hull_surface_sampled"]),
            "hull_sample_fallback_parts": int(
                jaw_clouds["bodies"][body]["hull_sample_fallback_parts"]),
        }
        for body in ("link5", "gripper_link")
    }
    validation["jaw_collision_cloud_hull_surface_provenance"] = {
        "pass": all(row["all_64_parts_hull_surface_sampled"]
                    and row["hull_sample_fallback_parts"] == 0
                    for row in hull_gate.values()),
        "bodies": hull_gate,
        "extractor_sha256": jaw_clouds["extractor_sha256"],
    }
    if not validation["jaw_collision_cloud_hull_surface_provenance"]["pass"]:
        validation["pass"] = False
        validation.setdefault("errors", []).append(
            "jaw collision cloud used a non-hull raw-point fallback")
    paths["rerun_validation.json"].write_text(json.dumps(validation, indent=2, default=_jsonable)+"\n")
    return {
        "technical_pass": bool(validation.get("pass")),
        "errors": validation.get("errors", []),
        "visual_inspection": "PENDING_SESSION_REVIEW_DO_NOT_CLAIM_COMPLETE",
        "full_timeline_steps": nsteps, "representatives": len(reps),
        "run_profile": run_profile,
        "jaw_collision_cloud_hull_surface_provenance": (
            validation["jaw_collision_cloud_hull_surface_provenance"]),
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_label", default="workspace1")
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--grid_side", type=int, default=8)
    ap.add_argument("--plan_workers", type=int, default=28)
    ap.add_argument("--handoff_results", type=Path, default=P13_RESULTS)
    ap.add_argument("--handoff_sha256", default=P13_RESULTS_SHA256)
    ap.add_argument("--protocol_path", type=Path, default=PREREG_PATH)
    ap.add_argument("--protocol_sha256", default=WORKSPACE1_PREREG_SHA256)
    ap.add_argument("--descend_open_deg", type=float, default=88.30998496351378)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--lift_delta_m", type=float, default=0.025)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--plan_tilt_gate_deg", type=float, default=5.0)
    ap.add_argument("--static_friction", type=float, default=0.40)
    ap.add_argument("--dynamic_friction", type=float, default=0.30)
    ap.add_argument("--stiffness", type=float, default=100.0)
    ap.add_argument("--damping", type=float, default=5.0)
    # decimation=1: doubled from p11 so physical phase durations stay unchanged.
    ap.add_argument("--settle_steps", type=int, default=120)
    ap.add_argument("--approach_steps", type=int, default=300)
    ap.add_argument("--descend_steps", type=int, default=500)
    ap.add_argument("--close_steps", type=int, default=300)
    ap.add_argument("--hold_steps", type=int, default=120)
    ap.add_argument("--lift_steps", type=int, default=500)
    ap.add_argument("--settle_stat_tail", type=int, default=60)
    ap.add_argument("--episode_length_s", type=float, default=120.0)
    # Each one-body sensor gets 256 * 1024 = 262,144 raw contact records.  This
    # applies independently to the object sensor and both jaw-ground sensors.
    # The gripper has 128 convex parts total; p11's 16 was not an authority.
    ap.add_argument("--contact_capacity", type=int, default=256)
    ap.add_argument("--gpu_found_lost_pairs_capacity", type=int, default=2**23)
    ap.add_argument("--gpu_total_aggregate_pairs_capacity", type=int, default=2**23)
    ap.add_argument("--gpu_collision_stack_size", type=int, default=2**28)
    ap.add_argument("--gpu_max_rigid_contact_count", type=int, default=2**23)
    return ap


def main() -> int:
    global _UNCAUGHT_FAILURE_CONTEXT
    args = build_argparser().parse_args()
    protocol_path = (args.protocol_path if args.protocol_path.is_absolute()
                     else REPO/args.protocol_path).resolve()
    args.protocol_path_resolved = protocol_path
    expected_handoff = P13_RESULTS.resolve()
    actual_handoff = (args.handoff_results if args.handoff_results.is_absolute()
                      else REPO/args.handoff_results).resolve()
    if args.run_label == "workspace1":
        run_profile = {
            "run_label": "workspace1", "profile": "canonical_workspace_scientific",
            "scientific_authoritative": True,
            "display_scope_label": "CANONICAL_SCIENTIFIC_WORKSPACE",
        }
        canonical = {
            "num_envs": 1024, "grid_side": 8, "plan_workers": 28,
            "handoff_sha256": P13_RESULTS_SHA256,
            "protocol_sha256": WORKSPACE1_PREREG_SHA256,
            "descend_open_deg": 88.30998496351378,
            "approach_clearance_m": 0.040, "lift_delta_m": 0.025,
            "target_error_gate_m": 0.003, "plan_tilt_gate_deg": 5.0,
            "static_friction": 0.40, "dynamic_friction": 0.30,
            "stiffness": 100.0, "damping": 5.0,
            "settle_steps": 120, "approach_steps": 300, "descend_steps": 500,
            "close_steps": 300, "hold_steps": 120, "lift_steps": 500,
            "settle_stat_tail": 60, "episode_length_s": 120.0, "contact_capacity": 256,
            "gpu_found_lost_pairs_capacity": 2**23,
            "gpu_total_aggregate_pairs_capacity": 2**23,
            "gpu_collision_stack_size": 2**28,
            "gpu_max_rigid_contact_count": 2**23,
        }
        drift = {name: {"expected": expected, "actual": getattr(args, name)}
                 for name, expected in canonical.items()
                 if getattr(args, name) != expected}
        if actual_handoff != expected_handoff:
            drift["handoff_results"] = {"expected": str(expected_handoff),
                                         "actual": str(actual_handoff)}
        if protocol_path != PREREG_PATH.resolve():
            drift["protocol_path"] = {"expected": str(PREREG_PATH.resolve()),
                                      "actual": str(protocol_path)}
        if drift:
            raise RuntimeError(f"CANONICAL_WORKSPACE1_ARG_DRIFT {json.dumps(drift, sort_keys=True)}")
    elif args.run_label == "workspace_preflight1":
        raise RuntimeError(
            "RETIRED_FAILED_RUN_LABEL 'workspace_preflight1'; its plan/log/PID evidence "
            "is forward-only and must not be overwritten; use workspace_preflight2")
    elif args.run_label == "workspace_preflight2":
        run_profile = {
            "run_label": "workspace_preflight2",
            "profile": "instrumentation_preflight",
            "scientific_authoritative": False,
            "display_scope_label": "INSTRUMENTATION_PREFLIGHT_ONLY",
        }
        preflight = {
            "num_envs": 128, "grid_side": 2, "plan_workers": 8,
            "handoff_sha256": P13_RESULTS_SHA256,
            "protocol_sha256": PREFLIGHT2_PREREG_SHA256,
            "descend_open_deg": 88.30998496351378,
            "approach_clearance_m": 0.040, "lift_delta_m": 0.025,
            "target_error_gate_m": 0.003, "plan_tilt_gate_deg": 5.0,
            "static_friction": 0.40, "dynamic_friction": 0.30,
            "stiffness": 100.0, "damping": 5.0,
            "settle_steps": 120, "approach_steps": 300, "descend_steps": 500,
            "close_steps": 30, "hold_steps": 20, "lift_steps": 30,
            "settle_stat_tail": 60, "episode_length_s": 120.0,
            "contact_capacity": 256,
            "gpu_found_lost_pairs_capacity": 2**23,
            "gpu_total_aggregate_pairs_capacity": 2**23,
            "gpu_collision_stack_size": 2**28,
            "gpu_max_rigid_contact_count": 2**23,
        }
        drift = {name: {"expected": expected, "actual": getattr(args, name)}
                 for name, expected in preflight.items() if getattr(args, name) != expected}
        if actual_handoff != expected_handoff:
            drift["handoff_results"] = {"expected": str(expected_handoff),
                                         "actual": str(actual_handoff)}
        if protocol_path != PREFLIGHT2_PREREG_PATH.resolve():
            drift["protocol_path"] = {"expected": str(PREFLIGHT2_PREREG_PATH.resolve()),
                                      "actual": str(protocol_path)}
        if drift:
            raise RuntimeError(f"PREFLIGHT2_ARG_DRIFT {json.dumps(drift, sort_keys=True)}")
    else:
        raise RuntimeError(
            f"UNREGISTERED_RUN_LABEL {args.run_label!r}; "
            "allowed=workspace1,workspace_preflight2; workspace_preflight1 is retired")
    if args.num_envs < 1 or args.num_envs > 1024:
        raise RuntimeError("num_envs must be in [1,1024]; larger 128-hull contact batches are unvalidated")

    prefix = f"{TAG}_{args.run_label}"
    paths = _run_paths(prefix)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        print(f"[{LOG}] G0_ARTIFACT_EXISTS_ABORT existing={existing}", flush=True)
        return 3

    start_time = time.time()
    source_start = Path(__file__).read_bytes()
    source_start_sha = hashlib.sha256(source_start).hexdigest()
    contact_semantics_smoke = _simultaneous_contact_regression_smoke()
    partial_invalid_smoke = _partial_invalid_population_regression_smoke()
    fallback_label_smoke = _fallback_label_regression_smoke()
    verdict_branch_smoke = _workspace_verdict_regression_smoke()
    jaw_support_smoke = _jaw_support_gate_regression_smoke()
    theta_q5_summary_smoke = _theta_q5_summary_regression_smoke()
    mdl_unresolved_smoke = _mdl_unresolved_classification_regression_smoke()
    preclose_lifecycle_smoke = _preclose_lifecycle_contract_regression_smoke()
    preclose_hash_binding_smoke = _preclose_hash_binding_regression_smoke()
    terminal_phase_binding_smoke = _terminal_phase_binding_regression_smoke()
    if not protocol_path.is_file():
        raise RuntimeError(f"PREREG_MISSING {protocol_path}")
    prereg_sha_start = sha256_file(protocol_path)
    if prereg_sha_start != args.protocol_sha256:
        raise RuntimeError(
            f"PROTOCOL_SHA256_MISMATCH expected={args.protocol_sha256} actual={prereg_sha_start}")
    _UNCAUGHT_FAILURE_CONTEXT = {
        "failure_path": str(paths["failure.json"]),
        "phase_path": str(paths["phase.jsonl"]),
        "tag": prefix,
        "run_profile": run_profile,
        "runtime_phase": "prelaunch_provenance_gates",
        "executed_source_sha256": source_start_sha,
        "protocol_path": str(protocol_path),
        "protocol_sha256": prereg_sha_start,
        "simulation_app": None,
        "terminal_close_call_entered": False,
    }
    _durable_append_phase(
        paths["phase.jsonl"], "run_claim",
        tag=prefix, run_profile=run_profile["profile"],
        executed_source_sha256=source_start_sha,
        protocol_sha256=prereg_sha_start,
    )
    local_source_manifest = _verify_pinned_local_sources()
    versions = _version_gate()
    p10 = _import_p10()
    jaw_source_sha = sha256_file(JAW_PATH)
    if jaw_source_sha != JAW_SHA256:
        raise RuntimeError(f"JAW_EXTRACTOR_SHA256_MISMATCH expected={JAW_SHA256} actual={jaw_source_sha}")
    asset = _asset_gate(p10)
    handoff_path = args.handoff_results if args.handoff_results.is_absolute() else REPO/args.handoff_results
    handoff = load_p13_handoff(handoff_path.resolve(), args.handoff_sha256)
    dependency_paths: dict[str, Path] = {
        "prereg": protocol_path,
        "p13_results": handoff_path.resolve(),
    }
    expected_dependency_hashes: dict[str, str] = {
        "prereg": prereg_sha_start,
        "p13_results": handoff["sha256"],
    }
    for name, (path, expected_sha) in PINNED_LOCAL_SOURCES.items():
        dependency_paths[f"local_source:{name}"] = path
        expected_dependency_hashes[f"local_source:{name}"] = expected_sha
    asset_root = Path(asset["asset_root"])
    for relative, expected_sha in ATTEMPT3_COMPOSED_LAYER_SHA256.items():
        dependency_paths[f"attempt3_usd:{relative}"] = asset_root / relative
        expected_dependency_hashes[f"attempt3_usd:{relative}"] = expected_sha
    dependency_hashes_start = _hash_named_paths(dependency_paths)
    if dependency_hashes_start != expected_dependency_hashes:
        raise RuntimeError(
            "FROZEN_DEPENDENCY_START_GATE_FAIL "
            f"expected={json.dumps(expected_dependency_hashes, sort_keys=True)} "
            f"actual={json.dumps(dependency_hashes_start, sort_keys=True)}")
    os.environ["ROARM_M3_USD_PATH"] = str(p10.ATTEMPT3_USD)
    print(f"[{LOG}] G0 inputs PASS p10={P10_SHA256[:16]} jaw={JAW_SHA256[:16]} "
          f"p13={handoff['sha256'][:16]} "
          f"asset={asset['root_sha256'][:16]}/{asset['physics_sha256'][:16]}", flush=True)

    print(f"[{LOG}] planning grid={args.grid_side}x{args.grid_side}x4 workers={args.plan_workers}", flush=True)
    plan = build_plan(args, p10, handoff)
    plan["p13_handoff"] = handoff
    feasible = [row for row in plan["trials"] if row.get("feasible")]
    print(f"[{LOG}] plan complete planned={plan['n_planned']} feasible={len(feasible)} "
          f"by_form={json.dumps(plan['by_form'])}", flush=True)
    if not feasible:
        raise RuntimeError("NO_IK_FEASIBLE_TRIAL: PhysX cannot execute a target the arm cannot reach")
    preflight_witness_trial_ids: list[str] = []
    if not run_profile["scientific_authoritative"]:
        preflight_witness_trial_ids = [
            str(row["trial_id"]) for row in feasible
            if row["pose_key"] in {"seed0_S3", "seed0_S4"}
            and row["form"] == "high_tilt"
        ]
        if not preflight_witness_trial_ids:
            raise RuntimeError(
                "PREFLIGHT_JAW_SUPPORT_WITNESS_IK_MISSING: no feasible "
                "seed0_S3/S4 high-tilt trial; silent all-zero sensor PASS forbidden")
        plan["preflight_jaw_support_witness"] = {
            "required": True,
            "selection": "exact seed0_S3/S4 and high_tilt(theta 60/69)",
            "feasible_trial_ids": preflight_witness_trial_ids,
        }
    paths["plan.json"].write_text(json.dumps(plan, indent=2, default=_jsonable)+"\n")

    simulation_app = None
    env = None
    batch_reports: list[dict[str, Any]] = []
    metric_parts: list[dict[str, np.ndarray]] = []
    mechanisms: list[np.ndarray] = []
    reason_parts: list[list[str]] = []
    replay: dict[str, Any] | None = None
    stage_audit: dict[str, Any] = {}
    jaw_clouds: dict[str, Any] | None = None
    contact_instrumentation: dict[str, Any] = {}
    usd_composition: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    runtime_phase = "app_launcher_start"
    assert _UNCAUGHT_FAILURE_CONTEXT is not None
    _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    failure_payload: dict[str, Any] | None = None
    lifecycle_errors: list[dict[str, str]] = []
    cleanup_status: dict[str, Any] = {
        "contract": "D367_D375_PRECLOSE_EXTERNAL_TERMINAL_V1",
        "env": {"created": False, "close_attempted": False, "close_returned": False},
        "simulation_app": {
            "created": False, "close_attempted": False, "close_returned": None,
            "expected_installed_behavior": (
                "terminal_nonreturning_framework_release; post-return marker not required"),
        },
        "exceptions": lifecycle_errors,
        "internal_preclose_pass": False,
        "internal_lifecycle_verdict": None,
        "terminal_completion": "PENDING_EXTERNAL_ATTESTATION",
        "terminal_authority": (
            "external supervisor: exit0, no timeout/signal, no process/PGID/GPU residue, "
            "result-sentinel-phase hash binding, then manual visual inspection"),
        # Deliberately never true inside this process.  The terminal call is expected
        # to end Python before any post-return serialization can execute.
        "pass": False,
    }
    try:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(headless=True, enable_cameras=False)
        simulation_app = app_launcher.app
        _UNCAUGHT_FAILURE_CONTEXT["simulation_app"] = simulation_app
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE: canonical parallel PhysX run requires the local GPU")
        gpu = {"torch": torch.__version__, "cuda_available": True,
               "device_name": torch.cuda.get_device_name(0),
               "device_capability": list(torch.cuda.get_device_capability(0))}
        runtime_phase = "attempt3_usd_composition_and_builtin_mdl_audit"
        usd_composition = _resolve_attempt3_composition_manifest(asset)
        runtime_phase = "jaw_hull_cloud_extraction"
        jaw = _import_jaw_extractor_after_kit()
        jaw_clouds = extract_jaw_view_clouds(jaw)
        runtime_phase = "isaaclab_environment_construction"
        env = make_env(args, p10)
        if not str(env.device).startswith("cuda"):
            raise RuntimeError(f"GPU_PHYSX_REQUIRED env.device={env.device}")
        audit_pass, body_checks = p10._audit_collision_bodies(env)
        explicit_64_plus_64 = all(
            body_checks.get(body, {}).get("enabled_total") == 64
            and body_checks[body].get("enabled_part_count") == 64
            and body_checks[body].get("part_count_64") is True
            and body_checks[body].get("enabled_only_parts") is True
            and body_checks[body].get("legacy_rows") == 1
            and body_checks[body].get("disabled_legacy_exact_one") is True
            for body in ("link5", "gripper_link")
        )
        stage_audit = {"pass": bool(audit_pass and explicit_64_plus_64),
                       "explicit_64_plus_64": explicit_64_plus_64,
                       "body_checks": body_checks}
        if not stage_audit["pass"]:
            raise RuntimeError(f"ATTEMPT3_STAGE_64_PLUS_64_AUDIT_FAIL {body_checks}")
        cloned_reporter_audit = _audit_cloned_jaw_contact_reporters(
            env.scene.stage, args.num_envs)
        stage_audit["cloned_jaw_contact_reporters"] = cloned_reporter_audit
        stage_audit["jaw_collision_cloud_hull_surface_gate"] = {
            body: {
                "all_64_parts_hull_surface_sampled": bool(
                    jaw_clouds["bodies"][body]["all_64_parts_hull_surface_sampled"]),
                "hull_sample_fallback_parts": int(
                    jaw_clouds["bodies"][body]["hull_sample_fallback_parts"]),
            }
            for body in ("link5", "gripper_link")
        }
        if env._t3y_contact_sensor.data.contact_pos_w is None:
            raise RuntimeError("contact-point sensor buffer was not allocated")
        spawn_report = env._t3y_spawn_arm_report
        object_reporter_ok = (
            spawn_report.get("object", {}).get("count") == 1
            and spawn_report["object"].get("thresholds") == [0.0])
        jaw_report = spawn_report.get("robot_jaws", {})
        jaw_reporter_ok = (
            jaw_report.get("count") == 2
            and jaw_report.get("expected_bodies") == ["link5", "gripper_link"]
            and jaw_report.get("thresholds") == [0.0, 0.0]
            and len(jaw_report.get("paths", [])) == 2)
        if not object_reporter_ok or not jaw_reporter_ok:
            raise RuntimeError(
                f"CONTACT_REPORTER_EXACT_ARM_FAIL spawn_report={spawn_report}")
        fixed_ground_resolution = resolve_single_ground_filter(
            env._t3y_fixed_ground_sensor, "fixed_jaw_ground", args.num_envs)
        moving_ground_resolution = resolve_single_ground_filter(
            env._t3y_moving_ground_sensor, "moving_jaw_ground", args.num_envs)
        object_filter_map = resolve_filter_map(env._t3y_contact_sensor, args.num_envs)
        contact_instrumentation = {
            "spawn_arm_report": spawn_report,
            "resolved_filter_paths": env._t3y_resolved,
            "object_sensor": {
                "force_matrix_shape": list(env._t3y_contact_sensor.data.force_matrix_w.shape),
                "contact_point_shape": list(env._t3y_contact_sensor.data.contact_pos_w.shape),
                "num_bodies": int(env._t3y_contact_sensor.num_bodies),
                "resolved_filter_map": object_filter_map,
                "track_contact_points": True,
            },
            "jaw_ground_force_only_sensors": {
                "fixed": fixed_ground_resolution,
                "moving": moving_ground_resolution,
                "track_contact_points": False,
                "max_contact_data_count_per_prim": int(args.contact_capacity),
                "gate_n_lte": JAW_SUPPORT_GATE_N,
                "recording_scope": {
                    "population": (
                        "read every PhysX step; reduce force and per-environment raw count "
                        "to per-row episode maxima"),
                    "representative_replay": (
                        "retain every-step world-force vectors and per-environment raw "
                        "counts in NPZ/RRD"),
                },
            },
        }

        runtime_phase = "population_physics_batches"
        for batch_id, start in enumerate(range(0, len(feasible), args.num_envs)):
            chunk = feasible[start:start+args.num_envs]
            padded, active = _pad_batch(chunk, args.num_envs)
            print(f"[{LOG}] physics batch={batch_id} active={active}/{args.num_envs}", flush=True)
            result = run_batch(args, env, padded, active_count=active)
            if result["kinematic_attach_calls"] != 0:
                raise RuntimeError(f"KINEMATIC_ATTACH_USED batch={batch_id}")
            batch_reports.append({
                "batch_id": batch_id, "start": start, "active_count": active,
                "wall_physics_s": result["wall_physics_s"],
                "positive_control": result["positive_control"],
                "kinematic_attach_calls": result["kinematic_attach_calls"],
                "filter_map": result["filter_map"],
            })
            metric_parts.append(result["metrics"])
            mechanisms.append(result["mechanism"])
            reason_parts.extend(result["reason_flags"])
            print(f"[{LOG}] batch={batch_id} pc={result['positive_control']['pass']} "
                  f"success={int(result['metrics']['success'].sum())}/{active} "
                  f"wall={result['wall_physics_s']:.1f}s", flush=True)

        population_metrics = _concat_metrics(metric_parts)
        population_mechanism = np.concatenate(mechanisms)
        if len(reason_parts) != len(feasible):
            raise RuntimeError(
                f"POPULATION_REASON_ALIGNMENT_FAIL reasons={len(reason_parts)} "
                f"feasible={len(feasible)}")
        rep_indices = choose_representatives(feasible, population_metrics, population_mechanism)
        rep_samples = [feasible[i] for i in rep_indices]
        replay_batch, replay_active = _pad_batch(rep_samples, args.num_envs)
        print(f"[{LOG}] representative replay n={replay_active} every-step timeline", flush=True)
        runtime_phase = "representative_physics_replay"
        replay = run_batch(args, env, replay_batch, active_count=replay_active,
                           capture_slots=list(range(replay_active)))
        if replay["kinematic_attach_calls"] != 0:
            raise RuntimeError("KINEMATIC_ATTACH_USED representative replay")
        runtime_phase = "physics_complete_env_close"
        _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    except Exception as exc:
        # Reactive to preflight1: SimulationApp.close may hang after a pre-env
        # exception.  Persist the cause before entering the unchanged cleanup path
        # so a detached supervisor has an auditable terminal marker even then.
        failure_payload = {
            "tool": "p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep",
            "tag": prefix,
            "run_profile": run_profile,
            "runtime_phase": runtime_phase,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "executed_source_sha256": source_start_sha,
            "protocol_path": str(protocol_path),
            "protocol_sha256": prereg_sha_start,
            "unix_time_s": time.time(),
            "cleanup_pending": True,
        }
        _durable_replace_json(paths["failure.json"], failure_payload)
        print(f"[{LOG}] FAILURE_MARKER_WRITTEN phase={runtime_phase} "
              f"path={paths['failure.json']} error={exc}", flush=True)
        raise
    finally:
        cleanup_status["env"]["created"] = env is not None
        cleanup_status["simulation_app"]["created"] = simulation_app is not None
        if env is not None:
            cleanup_status["env"]["close_attempted"] = True
            try:
                env.close()
                cleanup_status["env"]["close_returned"] = True
            except Exception as exc:
                lifecycle_errors.append({
                    "component": "env", "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                })
                print(f"[{LOG}] CLEANUP_ERROR component=env error={exc}", flush=True)
        cleanup_status["env_close_completed_unix_time_s"] = time.time()
        cleanup_status["internal_preclose_pass"] = _preclose_lifecycle_contract_pass(
            cleanup_status)
        cleanup_status["internal_lifecycle_verdict"] = (
            "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL"
            if cleanup_status["internal_preclose_pass"] else "PRECLOSE_INTERNAL_FAIL")

        if failure_payload is not None or lifecycle_errors:
            if failure_payload is None:
                failure_payload = {
                    "tool": "p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep",
                    "tag": prefix, "run_profile": run_profile,
                    "runtime_phase": "cleanup",
                    "exception_type": "IsaacCleanupError",
                    "exception_message": json.dumps(lifecycle_errors, sort_keys=True),
                    "executed_source_sha256": source_start_sha,
                    "protocol_path": str(protocol_path),
                    "protocol_sha256": prereg_sha_start,
                    "unix_time_s": time.time(),
                }
            if lifecycle_errors:
                failure_payload["runtime_phase_at_primary_exception"] = (
                    failure_payload.get("runtime_phase"))
                failure_payload["runtime_phase"] = "cleanup"
                failure_payload["failure_phases"] = sorted(set(
                    list(failure_payload.get("failure_phases", [])) + ["cleanup"]))
            failure_payload["cleanup"] = cleanup_status
            failure_payload["cleanup_pending"] = False
            failure_payload["simulation_app_terminal_close_deferred_to_top_level_failure_handler"] = (
                simulation_app is not None)
            _durable_replace_json(paths["failure.json"], failure_payload)
            print(f"[{LOG}] FAILURE_MARKER_UPDATED "
                  f"internal_preclose_pass={cleanup_status['internal_preclose_pass']} "
                  f"path={paths['failure.json']}", flush=True)

    if lifecycle_errors or not cleanup_status["internal_preclose_pass"]:
        raise RuntimeError(
            "ISAAC_ENV_CLOSE_OR_PRECLOSE_LIFECYCLE_FAILURE_BEFORE_RESULTS "
            f"{json.dumps(cleanup_status, sort_keys=True)}")

    assert replay is not None and jaw_clouds is not None
    runtime_phase = "offline_preclose_artifact_finalization"
    _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    current_source = Path(__file__).read_bytes()
    source_stable_post_physics = current_source == source_start
    _durable_write_bytes_x(paths["script.py.txt"], source_start)
    _durable_write_bytes_x(paths["argv.txt"], (" ".join(sys.argv)+"\n").encode("utf-8"))

    end_hashes = _hash_named_paths(dependency_paths)
    expected_end_hashes = dependency_hashes_start
    dependency_stable_post_physics = end_hashes == expected_end_hashes
    if not source_stable_post_physics or not dependency_stable_post_physics:
        raise RuntimeError(
            "EXECUTED_SOURCE_OR_DEPENDENCY_CHANGED_DURING_RUN; start p14 bytes frozen; "
            f"source_stable={source_stable_post_physics} "
            f"expected={expected_end_hashes} actual={end_hashes}")

    feasible_trial_index = np.asarray([row["trial_index"] for row in feasible], dtype=np.int64)
    rep_indices = choose_representatives(feasible, population_metrics, population_mechanism)
    reps = [feasible[i] for i in rep_indices]
    replay_metrics = replay["metrics"]
    replay_compare = []
    replay_gate_fields = (
        "measurement_valid", "jaw_support_contact_pass", "arrival_pass",
        "preclose_pass", "both_jaws_close", "both_jaws_lift", "success",
    )
    for col, pop_idx in enumerate(rep_indices):
        population_classes = {
            field: bool(population_metrics[field][pop_idx]) for field in replay_gate_fields}
        replay_classes = {
            field: bool(replay_metrics[field][col]) for field in replay_gate_fields}
        mechanism_population = str(population_mechanism[pop_idx])
        mechanism_replay = str(replay["mechanism"][col])
        replay_compare.append({
            "trial_id": feasible[pop_idx]["trial_id"],
            "gate_classes_population": population_classes,
            "gate_classes_replay": replay_classes,
            "mechanism_population": mechanism_population,
            "mechanism_replay": mechanism_replay,
            "gate_class_equal": bool(
                population_classes == replay_classes
                and mechanism_population == mechanism_replay),
            "lift_delta_mm": float(replay_metrics["lift_corrected_mm"][col]
                                   - population_metrics["lift_corrected_mm"][pop_idx]),
            "tilt_delta_deg": float(replay_metrics["tilt_final_deg"][col]
                                     - population_metrics["tilt_final_deg"][pop_idx]),
        })
    replay_gate_class_equal = all(row["gate_class_equal"] for row in replay_compare)

    trace_payload: dict[str, Any] = {
        "population_trial_index": feasible_trial_index,
        "population_mechanism": population_mechanism,
        "population_reason_flags_json": np.asarray(
            [json.dumps(flags, separators=(",", ":")) for flags in reason_parts], dtype="U256"),
        "replay_mechanism": replay["mechanism"],
        "replay_reason_flags_json": np.asarray(
            [json.dumps(flags, separators=(",", ":"))
             for flags in replay["reason_flags"]], dtype="U256"),
    }
    trace_payload.update({f"population_{k}": v for k, v in population_metrics.items()})
    trace_payload.update({f"replay_metric_{k}": v for k, v in replay_metrics.items()})
    trace_payload.update({f"replay_trace_{k}": v for k, v in replay["trace"].items()})
    np.savez_compressed(paths["trace.npz"], **trace_payload)

    population_valid = population_metrics["measurement_valid"].astype(bool, copy=False)
    batch_pc = np.asarray(
        [bool(report["positive_control"]["pass"]) for report in batch_reports], dtype=bool)
    population_contract = _population_validity_summary(
        population_metrics["success"], population_valid, batch_pc)
    buffers_unsaturated = all(
        not bool(report["positive_control"]["contact_buffer_saturated"])
        for report in batch_reports)
    population_contract["all_contact_buffers_unsaturated"] = buffers_unsaturated
    population_contract["global_workspace_claim_allowed"] = bool(
        population_contract["global_workspace_claim_allowed"] and buffers_unsaturated)
    task_clear_population = (
        population_valid & population_metrics["jaw_support_contact_pass"])
    support_fail_population = (
        population_valid & ~population_metrics["jaw_support_contact_pass"])
    support_fail_bilateral = support_fail_population & (
        population_metrics["both_jaws_close"] | population_metrics["both_jaws_lift"])
    valid_lift = population_metrics["lift_corrected_mm"][task_clear_population]
    measured_valid_lift_diagnostic = population_metrics["lift_corrected_mm"][population_valid]
    n_valid = int(population_contract["n_valid"])
    valid_success = int(population_contract["valid_success"])
    valid_success_rate = population_contract["valid_success_rate"]
    all_summary = {
        "n": len(feasible), "n_valid": n_valid, "n_invalid": len(feasible)-n_valid,
        "success": valid_success,
        "success_rate": valid_success_rate,
        "success_rate_denominator": "n_valid",
        "success_rate_valid_denominator": valid_success_rate,
        "both_jaws_close": int(
            (population_metrics["both_jaws_close"] & task_clear_population).sum()),
        "both_jaws_lift": int(
            (population_metrics["both_jaws_lift"] & task_clear_population).sum()),
        "both_jaws_counts_scope": "measurement_valid_and_jaw_support_clear",
        "both_jaws_close_raw_diagnostic": int(population_metrics["both_jaws_close"].sum()),
        "both_jaws_lift_raw_diagnostic": int(population_metrics["both_jaws_lift"].sum()),
        "n_jaw_support_fail": int(support_fail_population.sum()),
        "n_jaw_support_observed_all_diagnostic": int(
            (~population_metrics["jaw_support_contact_pass"]).sum()),
        "n_jaw_support_fail_with_bilateral": int(support_fail_bilateral.sum()),
        "jaw_support_fail_with_bilateral": {
            "close": int((support_fail_population
                          & population_metrics["both_jaws_close"]).sum()),
            "lift": int((support_fail_population
                         & population_metrics["both_jaws_lift"]).sum()),
            "either": int(support_fail_bilateral.sum()),
        },
        "task_clear": {
            "n": int(task_clear_population.sum()),
            "success": int((population_metrics["success"] & task_clear_population).sum()),
            "both_jaws_close": int(
                (population_metrics["both_jaws_close"] & task_clear_population).sum()),
            "both_jaws_lift": int(
                (population_metrics["both_jaws_lift"] & task_clear_population).sum()),
        },
        "jaw_ground_force_max_n_all_diagnostic": {
            "fixed": float(np.max(population_metrics["fixed_jaw_ground_force_max_n"])),
            "moving": float(np.max(population_metrics["moving_jaw_ground_force_max_n"])),
        },
        "jaw_ground_raw_count_per_env_max_all_diagnostic": {
            "fixed": int(np.max(population_metrics["fixed_jaw_ground_raw_count_max"])),
            "moving": int(np.max(population_metrics["moving_jaw_ground_raw_count_max"])),
        },
        "lift_corrected_mm": {
            "median": None if not len(valid_lift) else float(np.median(valid_lift)),
            "p90": None if not len(valid_lift) else float(np.percentile(valid_lift, 90)),
            "max": None if not len(valid_lift) else float(np.max(valid_lift)),
        },
        "lift_corrected_mm_scope": "measurement_valid_and_jaw_support_clear",
        "lift_corrected_mm_measured_valid_all_diagnostic": {
            "median": (None if not len(measured_valid_lift_diagnostic)
                       else float(np.median(measured_valid_lift_diagnostic))),
            "max": (None if not len(measured_valid_lift_diagnostic)
                    else float(np.max(measured_valid_lift_diagnostic))),
        },
        "lift_corrected_mm_raw_max_diagnostic": float(np.max(population_metrics["lift_corrected_mm"])),
        "mechanism_counts_valid": {
            label: int(np.sum(population_mechanism[population_valid] == label))
            for label in sorted(set(population_mechanism[population_valid].tolist()))},
        "mechanism_counts_all_diagnostic": {
            label: int(np.sum(population_mechanism == label))
            for label in sorted(set(population_mechanism.tolist()))},
    }
    summary = {
        "population_measurement_contract": population_contract,
        "all": all_summary,
        "by_region": _aggregate(feasible, population_metrics, population_mechanism, "region"),
        "by_form": _aggregate(feasible, population_metrics, population_mechanism, "form"),
        "by_theta": _aggregate(feasible, population_metrics, population_mechanism, "theta_target_deg"),
        "by_theta_q5": _aggregate_theta_q5(feasible, population_metrics, population_mechanism),
        "by_window_kind": _aggregate(feasible, population_metrics, population_mechanism, "window_kind"),
    }
    preflight_witness_report: dict[str, Any] | None = None
    if not run_profile["scientific_authoritative"]:
        witness_set = set(preflight_witness_trial_ids)
        witness_rows: list[dict[str, Any]] = []
        for index, row in enumerate(feasible):
            if row["trial_id"] not in witness_set:
                continue
            batch_id = index // int(args.num_envs)
            pc = batch_reports[batch_id]["positive_control"]
            fixed_force = float(population_metrics["fixed_jaw_ground_force_max_n"][index])
            moving_force = float(population_metrics["moving_jaw_ground_force_max_n"][index])
            fixed_raw = int(population_metrics["fixed_jaw_ground_raw_count_max"][index])
            moving_raw = int(population_metrics["moving_jaw_ground_raw_count_max"][index])
            raw_peak = max(fixed_raw, moving_raw)
            force_max = max(fixed_force, moving_force)
            paired_nonzero = bool(
                (fixed_force > CONTACT_EPS_N and fixed_raw > 0)
                or (moving_force > CONTACT_EPS_N and moving_raw > 0))
            paired_above_gate = bool(
                (fixed_force > JAW_SUPPORT_GATE_N and fixed_raw > 0)
                or (moving_force > JAW_SUPPORT_GATE_N and moving_raw > 0))
            witness_rows.append({
                "trial_id": row["trial_id"], "pose_key": row["pose_key"],
                "theta_target_deg": row["theta_target_deg"],
                "q5_close_deg": row["q5_close_deg"],
                "fixed_jaw_ground_force_max_n": fixed_force,
                "moving_jaw_ground_force_max_n": moving_force,
                "jaw_ground_force_max_n": force_max,
                "fixed_jaw_ground_raw_count_per_env_max": fixed_raw,
                "moving_jaw_ground_raw_count_per_env_max": moving_raw,
                "jaw_ground_raw_count_per_env_max": raw_peak,
                "batch_fixed_ground_raw_count_total_peak_saturation_diagnostic": int(
                    pc["fixed_ground_contact_data_total_peak"]),
                "batch_moving_ground_raw_count_total_peak_saturation_diagnostic": int(
                    pc["moving_ground_contact_data_total_peak"]),
                "above_0p02n_task_failure_gate_with_same_jaw_raw": paired_above_gate,
                "nonzero_force_and_raw_witness_same_jaw": paired_nonzero,
                "mechanism": str(population_mechanism[index]),
                "reason_flags": reason_parts[index],
            })
        preflight_witness_report = {
            "required": True,
            "selection": "exact seed0_S3/S4 high_tilt feasible rows",
            "n_planned_witness_rows": len(witness_rows),
            "nonzero_force_and_raw_witness_pass": any(
                row["nonzero_force_and_raw_witness_same_jaw"] for row in witness_rows),
            "above_0p02n_task_failure_witness_pass": any(
                row["above_0p02n_task_failure_gate_with_same_jaw_raw"]
                for row in witness_rows),
            "rows": witness_rows,
        }
        summary["preflight_jaw_support_witness"] = preflight_witness_report
    replay_valid = replay_metrics["measurement_valid"].astype(bool, copy=False)
    replay_measurement_complete = bool(
        replay["positive_control"]["pass"] and replay_valid.all()
        and not replay["positive_control"]["contact_buffer_saturated"])
    verdict = _workspace_verdict(
        bool(population_contract["global_workspace_claim_allowed"]),
        replay_measurement_complete, replay_gate_class_equal,
        int(all_summary["success"]), int(all_summary["both_jaws_close"]),
        int(all_summary["both_jaws_lift"]),
        int(all_summary["n_jaw_support_fail_with_bilateral"]),
        int(all_summary["n_jaw_support_fail"]),
    )

    snapshot_status = emit_decision_snapshot(
        paths["decision_snapshot.png"], p10, reps, replay["trace"], verdict, run_profile)
    try:
        rerun_status = emit_rerun(
            paths, p10, jaw_clouds, reps, replay["trace"], summary, verdict, run_profile)
    except Exception as exc:
        rerun_status = {"technical_pass": False, "errors": [f"rerun stage raised: {exc}"],
                        "visual_inspection": "NOT_AVAILABLE"}
        paths["rerun_validation.json"].write_text(json.dumps(rerun_status, indent=2)+"\n")
        print(f"[{LOG}] rerun error: {exc}", flush=True)

    # Rerun emission is still part of this process.  Recheck the script and every
    # frozen dependency immediately before the canonical result is serialized, so
    # an edit during replay/validation cannot be reported as stable-at-end.
    source_stable = Path(__file__).read_bytes() == source_start
    end_hashes = _hash_named_paths(dependency_paths)
    dependency_stable = end_hashes == expected_end_hashes
    if not source_stable or not dependency_stable:
        raise RuntimeError(
            "EXECUTED_SOURCE_OR_DEPENDENCY_CHANGED_BEFORE_FINALIZE; "
            f"source_stable={source_stable} expected={expected_end_hashes} actual={end_hashes}")

    preclose_artifact_names = (
        "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
        "rerun_validation.json", "inspection.png", "decision_snapshot.png",
        "script.py.txt", "argv.txt",
    )
    preclose_artifact_manifest = _artifact_manifest(paths, preclose_artifact_names)

    instrumentation_preflight: dict[str, Any] | None = None
    if not run_profile["scientific_authoritative"]:
        assert preflight_witness_report is not None
        instrumentation_checks = {
            "env_close_and_internal_preclose_contract": bool(
                cleanup_status["internal_preclose_pass"]
                and not cleanup_status["exceptions"]
                and cleanup_status["simulation_app"]["close_attempted"] is False
                and cleanup_status["terminal_completion"]
                == "PENDING_EXTERNAL_ATTESTATION"),
            "attempt3_exact_five_local_usd_layers": bool(
                usd_composition.get("exact_expected_set_pass")),
            "builtin_mdl_unresolved_exact_allowset": bool(
                usd_composition.get("unresolved_dependency_classification", {}).get(
                    "exact_allowed_set_pass")),
            "builtin_mdl_runtime_membership_and_config": bool(
                usd_composition.get("runtime_builtin_mdl_resolver", {}).get(
                    "required_identifiers_present")
                and usd_composition.get("runtime_builtin_mdl_resolver", {}).get(
                    "builtin_bypass_config_exact_one")),
            "builtin_mdl_external_composition_arc_absence": bool(
                usd_composition.get("external_composition_arc_audit", {}).get(
                    "mdl_external_arc_absence_pass")),
            "builtin_mdl_usdshade_source_asset_exclusive": bool(
                usd_composition.get("usdshade_mdl_source_asset_audit", {}).get(
                    "exclusive_source_asset_pass")),
            "stage_64_plus_64": bool(stage_audit.get("pass")),
            "all_cloned_jaw_reporters_threshold_zero": bool(
                stage_audit.get("cloned_jaw_contact_reporters", {}).get("pass")),
            "all_population_contact_buffers_unsaturated": all(
                not bool(report["positive_control"]["contact_buffer_saturated"])
                for report in batch_reports),
            "all_population_support_positive_controls": all(
                bool(report["positive_control"]["pass"])
                and bool(report["positive_control"]["support_force_population_pass"])
                for report in batch_reports),
            "replay_contact_buffers_unsaturated": not bool(
                replay["positive_control"]["contact_buffer_saturated"]),
            "replay_support_positive_control": bool(
                replay["positive_control"]["pass"]
                and replay["positive_control"]["support_force_population_pass"]),
            "replay_gate_class_and_mechanism_equal": bool(replay_gate_class_equal),
            "every_step_jaw_ground_trace_present": bool(
                "jaw_ground_force_w_n" in replay["trace"]
                and "jaw_ground_raw_count" in replay["trace"]
                and len(replay["trace"]["jaw_ground_force_w_n"]) == replay["total_steps"]
                and len(replay["trace"]["jaw_ground_raw_count"]) == replay["total_steps"]),
            "jaw_ground_metric_schema_present": all(
                name in population_metrics for name in (
                    "fixed_jaw_ground_force_max_n", "moving_jaw_ground_force_max_n",
                    "fixed_jaw_ground_raw_count_max", "moving_jaw_ground_raw_count_max",
                    "jaw_support_contact_pass")),
            "known_pose_nonzero_force_and_raw_witness": bool(
                preflight_witness_report["nonzero_force_and_raw_witness_pass"]),
            "known_pose_above_0p02n_task_failure_witness": bool(
                preflight_witness_report["above_0p02n_task_failure_witness_pass"]),
            "rerun_technical_contract": bool(rerun_status.get("technical_pass")),
            "preclose_artifact_manifest_complete": bool(
                preclose_artifact_manifest["complete"]),
            "source_stable": bool(source_stable),
            "dependency_manifest_stable": bool(dependency_stable),
        }
        instrumentation_runtime_pass = all(instrumentation_checks.values())
        instrumentation_preflight = {
            "authoritative_scientific": False,
            "scope": "INSTRUMENTATION_PREFLIGHT_ONLY",
            "checks": instrumentation_checks,
            "witness": preflight_witness_report,
            "verdict": (
                "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL"
                if instrumentation_runtime_pass else "INSTRUMENTATION_PREFLIGHT_FAIL"),
            "terminal_completion_authority": "external_supervisor_only",
            "workspace_branch_diagnostic_only": verdict,
        }

    output: dict[str, Any] = {
        "tool": "p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep",
        "case": "g0b_d420", "tag": prefix, "protocol": str(protocol_path.relative_to(REPO)),
        "run_profile": run_profile,
        "scientific_authoritative": bool(run_profile["scientific_authoritative"]),
        "argv": sys.argv, "versions": versions, "gpu": gpu,
        "cleanup": cleanup_status,
        "resolved_args": vars(args),
        "provenance": {
            "executed_source_sha256": source_start_sha, "executed_source_stable": source_stable,
            "executed_source_stable_post_physics": source_stable_post_physics,
            "p10_path": str(P10_PATH), "p10_sha256": P10_SHA256,
            "jaw_extractor_path": str(JAW_PATH), "jaw_extractor_sha256": JAW_SHA256,
            "p13_results_path": handoff["path"], "p13_results_sha256": handoff["sha256"],
            "p13_results_expected_sha256": args.handoff_sha256,
            "p13_input_sha256": handoff["input_sha256"], "attempt3": asset,
            "attempt3_recursive_composition_resolution": usd_composition,
            "pinned_local_source_manifest": local_source_manifest,
            "p13_producer_validity": handoff["producer_validity"],
            "p13_jaw_support_reactive_basis": handoff["jaw_support_reactive_basis"],
            "prereg_path": str(protocol_path), "prereg_sha256": prereg_sha_start,
            "prereg_expected_sha256": args.protocol_sha256,
            "end_hashes_equal_start": dependency_stable,
            "dependency_expected_sha256": expected_dependency_hashes,
            "dependency_hashes_at_start": dependency_hashes_start,
            "dependency_hashes_at_finalize": end_hashes,
        },
        "stage_collision_audit": stage_audit,
        "contact_instrumentation": contact_instrumentation,
        "rerun_collision_clouds": {k: v for k, v in jaw_clouds.items() if k != "clouds"},
        "object": {"shape": "upright_cylinder", "diameter_m": OBJ_DIAM_M,
                   "height_m": OBJ_HEIGHT_M, "mass_kg": OBJ_MASS_KG,
                   "grasp_point": "top_center_D419_unchanged"},
        "fixed_controls": {
            "object_yaw_deg": 0.0, "approach_azimuth": "per_pose_radial",
            "cylinder_material_authored_static_friction": args.static_friction,
            "cylinder_material_authored_dynamic_friction": args.dynamic_friction,
            "jaw_and_support_material": (
                "not re-authored by p14; pinned roarm_stack_env/attempt3 USD defaults"),
            "effective_pair_friction": "unmeasured_and_not_claimed",
            "stiffness": args.stiffness, "damping": args.damping,
            "contact_capacity_per_prim": args.contact_capacity,
            "solve_articulation_contact_last": False,
            "kinematic_attach_disabled": True,
            "sim_dt_s": 1.0/200.0, "decimation": 1,
            "schedule": [
                {"phase": "settle", "steps": args.settle_steps},
                {"phase": "approach", "steps": args.approach_steps},
                {"phase": "descend", "steps": args.descend_steps},
                {"phase": "close", "steps": args.close_steps},
                {"phase": "hold", "steps": args.hold_steps},
                {"phase": "lift", "steps": args.lift_steps},
            ],
        },
        "gates": {
            "ik_position_mm": args.target_error_gate_m*1000.0,
            "ik_direction_deg": args.plan_tilt_gate_deg,
            "lift_corrected_mm_strict_gt": LIFT_GATE_MM,
            "final_tilt_deg_strict_lt": TIP_HALF_ANGLE_DEG,
            "jaw_load_n_strict_gt": JAW_LOAD_GATE_N,
            "jaw_load_same_physics_step_required": True,
            "jaw_load_authority_scalar": "max_over_phase(min(fixed_force_n[t],moving_force_n[t]))",
            "preclose_force_n_lte": PRECLOSE_GATE_N,
            "jaw_support_force_n_lte": JAW_SUPPORT_GATE_N,
            "jaw_support_contact_is_measured_task_failure": True,
            "jaw_support_contact_invalidates_measurement": False,
            "measurement_validity_scope": (
                "settle positive-control and contact-buffer integrity only"),
            "bilateral_static_window_is_not_admission_gate": True,
        },
        "measurement_semantics_regression": {
            "same_step_bilateral": contact_semantics_smoke,
            "partial_invalid_population": partial_invalid_smoke,
            "none_margin_fallback_label": fallback_label_smoke,
            "phase_scoped_workspace_verdict": verdict_branch_smoke,
            "jaw_support_contact_gate": jaw_support_smoke,
            "theta_q5_direct_summary": theta_q5_summary_smoke,
        },
        "provenance_regression": {
            "builtin_mdl_unresolved_dependency_classification": mdl_unresolved_smoke,
            "d367_d375_preclose_lifecycle": preclose_lifecycle_smoke,
            "preclose_result_artifact_hash_binding": preclose_hash_binding_smoke,
            "terminal_phase_sentinel_process_binding": terminal_phase_binding_smoke,
        },
        "plan_counts": {k: plan[k] for k in (
            "n_positions", "n_planned", "n_feasible", "n_plan_gate_failed",
            "n_ik_failed", "n_wrist_r_v6_failed", "failure_reasons", "by_form")},
        "batch_reports": batch_reports,
        "population_classifications": [
            {"trial_id": row["trial_id"], "mechanism": str(population_mechanism[index]),
             "reason_flags": reason_parts[index]}
            for index, row in enumerate(feasible)
        ],
        "population_measurement_contract": population_contract,
        "summary": summary, "representative_trial_ids": [row["trial_id"] for row in reps],
        "representative_replay_positive_control": replay["positive_control"],
        "representative_replay_all_measurement_valid": replay_measurement_complete,
        "replay_compare": replay_compare,
        "replay_gate_class_and_mechanism_equal": replay_gate_class_equal,
        "decision_snapshot": snapshot_status, "rerun": rerun_status,
        "internal_lifecycle_verdict": "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL",
        "terminal_completion_authority": "external_supervisor_only",
        "scientific_verdict": None,
        "scientific_verdict_preclose_candidate": (
            verdict if run_profile["scientific_authoritative"] else None),
        "scientific_verdict_authority": (
            "not authoritative until result-sentinel binding, terminal exit/no-residue, "
            "and actual visual inspection all pass"),
        "instrumentation_preflight": instrumentation_preflight,
        "diagnostic_workspace_branch": verdict,
        "preclose_artifact_manifest": preclose_artifact_manifest,
        "preclose_sentinel_contract": {
            "path": str(paths["preclose_sentinel.json"].relative_to(REPO)),
            "must_bind_this_results_full_sha256": True,
            "simulation_app_close_is_next_normal_action_after_sentinel": True,
            "post_close_return_marker_required": False,
            "external_terminal_attestation_path": str(
                paths["terminal_attestation.json"].relative_to(REPO)),
        },
        "scope_warning": (
            ("A success is local to its measured top-centre pose/form/control. "
             "This run does not test a cylinder side-midpoint grasp and does not authorize hardware.")
            if run_profile["scientific_authoritative"] else
            ("INSTRUMENTATION_PREFLIGHT_ONLY: shortened population/schedule has no scientific "
             "workspace authority; the diagnostic workspace branch must not enter the ledger.")),
        "wall_seconds": time.time()-start_time,
    }
    output["artifacts"] = preclose_artifact_manifest["files"]
    runtime_phase = "durable_results_serialization"
    _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    _durable_write_json_x(paths["results.json"], output)
    results_sha256 = sha256_file(paths["results.json"])
    results_bytes = paths["results.json"].stat().st_size
    _durable_append_phase(
        paths["phase.jsonl"], "results_durable",
        results_sha256=results_sha256, results_bytes=results_bytes,
        internal_lifecycle_verdict="PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL",
    )

    # Fail closed if source/dependency bytes move in the narrow result->sentinel
    # window.  The already-written result alone has no authority; the external
    # supervisor requires the sentinel and this exact hash binding.
    post_result_source_stable = Path(__file__).read_bytes() == source_start
    post_result_dependency_hashes = _hash_named_paths(dependency_paths)
    post_result_dependency_stable = post_result_dependency_hashes == expected_end_hashes
    if not post_result_source_stable or not post_result_dependency_stable:
        raise RuntimeError(
            "EXECUTED_SOURCE_OR_DEPENDENCY_CHANGED_BEFORE_PRECLOSE_SENTINEL; "
            f"source_stable={post_result_source_stable} "
            f"expected={expected_end_hashes} actual={post_result_dependency_hashes}")

    phase_prefix = paths["phase.jsonl"].read_bytes()
    preclose_sentinel = {
        "artifact": "T3Y_D367_D375_PRECLOSE_SENTINEL_V1",
        "tag": prefix,
        "run_profile": run_profile,
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "results_path": str(paths["results.json"].relative_to(REPO)),
        "results_sha256": results_sha256,
        "results_bytes": results_bytes,
        "artifact_manifest": preclose_artifact_manifest,
        "executed_source_sha256": source_start_sha,
        "executed_source_stable_after_results": post_result_source_stable,
        "protocol_path": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": prereg_sha_start,
        "p13_results_path": str(handoff_path.resolve().relative_to(REPO)),
        "p13_results_sha256": handoff["sha256"],
        "dependency_hashes_after_results": post_result_dependency_hashes,
        "dependency_hashes_equal_start": post_result_dependency_stable,
        "env_close": cleanup_status["env"],
        "env_close_internal_pass": cleanup_status["internal_preclose_pass"],
        "phase_prefix_bytes": len(phase_prefix),
        "phase_prefix_sha256": hashlib.sha256(phase_prefix).hexdigest(),
        "internal_lifecycle_verdict": "PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL",
        "terminal_completion": "PENDING_EXTERNAL_ATTESTATION",
        "terminal_authority": cleanup_status["terminal_authority"],
        "safe_to_close_app": True,
        "simulation_app_close_is_next_normal_action": True,
        "simulation_app_close_postreturn_marker_required": False,
        "manual_visual_inspection_required_after_terminal_attestation": True,
    }
    runtime_phase = "durable_preclose_sentinel_serialization"
    _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    _durable_write_json_x(paths["preclose_sentinel.json"], preclose_sentinel)
    preclose_sentinel_sha256 = sha256_file(paths["preclose_sentinel.json"])
    _durable_append_phase(
        paths["phase.jsonl"], "preclose_sentinel_durable",
        preclose_sentinel_sha256=preclose_sentinel_sha256,
        results_sha256=results_sha256,
    )
    if run_profile["scientific_authoritative"]:
        print(f"[{LOG}] G0B_T3Y_WORKSPACE_PRECLOSE_CANDIDATE={verdict} "
              f"valid_success={all_summary['success']}/{all_summary['n_valid']} "
              f"invalid={all_summary['n_invalid']} "
              "lifecycle=PRECLOSE_READY_PENDING_TERMINAL_AND_VISUAL "
              f"wall={output['wall_seconds']:.1f}s", flush=True)
    else:
        assert instrumentation_preflight is not None
        print(f"[{LOG}] G0B_T3Y_INSTRUMENTATION_PREFLIGHT_PRECLOSE_VERDICT="
              f"{instrumentation_preflight['verdict']} "
              f"diagnostic_workspace_branch={verdict} "
              f"wall={output['wall_seconds']:.1f}s", flush=True)
    print(f"[{LOG}] PRECLOSE_SENTINEL_DURABLE sha256={preclose_sentinel_sha256} "
          f"results_sha256={results_sha256}", flush=True)

    # D367/D375 terminal rule: this is the literal last normal Python call.  On
    # the installed Isaac Sim 5.1 stack it normally terminates the process inside
    # framework release.  A return or raised BaseException is therefore a failure,
    # not a place from which to serialize a false cleanup PASS.
    runtime_phase = "simulation_app_terminal_close"
    _UNCAUGHT_FAILURE_CONTEXT["runtime_phase"] = runtime_phase
    _durable_append_phase(
        paths["phase.jsonl"], "simulation_app_close_start",
        preclose_sentinel_sha256=preclose_sentinel_sha256,
        expected_behavior="terminal_nonreturning_framework_release",
    )
    print(f"[{LOG}] SIMULATION_APP_CLOSE_START terminal_nonreturning_expected", flush=True)
    assert simulation_app is not None
    _UNCAUGHT_FAILURE_CONTEXT["terminal_close_call_entered"] = True
    try:
        simulation_app.close()
    except BaseException as exc:
        failure = RuntimeError(
            f"SIMULATION_APP_TERMINAL_CLOSE_RAISED {type(exc).__name__}: {exc}")
        _record_uncaught_failure(failure)
        raise failure from exc
    failure = RuntimeError("SIMULATION_APP_TERMINAL_CLOSE_UNEXPECTED_RETURN")
    _record_uncaught_failure(failure)
    raise failure


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--external_terminal_attest":
        raise SystemExit(_external_terminal_attest(sys.argv[2]))
    try:
        _exit_code = main()
    except BaseException as _top_level_exc:
        _record_uncaught_failure(_top_level_exc)
        _terminal_close_after_failure(_top_level_exc)
        raise
    if _exit_code == 0 and _UNCAUGHT_FAILURE_CONTEXT is not None:
        _unexpected = RuntimeError("MAIN_RETURNED_ZERO_WITHOUT_TERMINAL_SIMULATION_APP_CLOSE")
        _record_uncaught_failure(_unexpected)
        raise _unexpected
    raise SystemExit(_exit_code)
