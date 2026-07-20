#!/usr/bin/env python3
"""D367: localize deferred raw PLAY with one explicit main-thread commit.

This forward-only control audit reproduces the frozen D366 paused baseline,
records raw before/request/post timeline state and opaque joint/object bits, and
tests exactly one new ``Timeline.commit()`` after one raw ``timeline.play()``.
It performs no cylinder write, controlled physics step, public forward, q5
science sample/target update, contact query, render, or next-frame update.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import os
import secrets
import signal
import struct
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

# Safe before AppLauncher.  These modules intentionally import no pxr/omni at
# top level; the worker preflight verifies that invariant again at runtime.
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d353_timeline_pause_pending_state_commit_bridge as d353,
    cyl34_top_view_d366_tensor_step_fabric_visibility_commit as d366,
)


CASE = "g0a_d367"
CASE_NAME = "timeline_play_pending_state_commit_localization"
NEW_VARIABLES = ["explicit_timeline_commit_after_raw_play_request"]
BASE_GIT = "9f956a42db1bb43c817ffe435a4e9698707049f1"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
DISPLAY = ":1"
SEED = 33201
MIN_GPU_FREE_MIB = 8192
MIN_RAM_AVAILABLE_BYTES = 8 * 1024**3
TOTAL_WATCHDOG_S = 300.0
INACTIVITY_WATCHDOG_S = 90.0
TERM_GRACE_S = 15.0
KILL_GRACE_S = 10.0
PLAY_SIMULATIONS_SETTING = d366.PLAY_SIMULATIONS_SETTING

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d367"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260719_grasp_g0a_d367_timeline_play_pending_state_commit_localization.md"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
# D356 was never allocated a runtime directory.  Freeze every actually existing
# D351-D366 forward-only evidence root without inventing a missing case.
FROZEN_CASE_NUMBERS = (351, 352, 353, 354, 355, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366)
FROZEN_DIRS = {
    f"g0a_d{number}": REPO / f"claudedocs/runtime_logs/grasp_track/g0a_d{number}"
    for number in FROZEN_CASE_NUMBERS
}

D353_HARNESS = Path(d353.__file__).resolve()
D353_EVENT = FROZEN_DIRS["g0a_d353"] / "d353_timeline_commit_event_contract.json"
D353_ATTESTATION = FROZEN_DIRS["g0a_d353"] / "d353_timeline_commit_bridge_attestation.json"
D366_HARNESS = Path(d366.__file__).resolve()
D366_RUNTIME = FROZEN_DIRS["g0a_d366"] / "d366_runtime_step_fabric_attestation.json"
D366_EXCEPTION = FROZEN_DIRS["g0a_d366"] / "d366_worker_exception.json"
D366_COMPLETION = FROZEN_DIRS["g0a_d366"] / "d366_pre_step_play_guard_safe_stop_completion.json"
D366_CORRECTION = FROZEN_DIRS["g0a_d366"] / "d366_postcompletion_correction_audit.json"

FROZEN_INPUT_HASHES = {
    str(D353_HARNESS): "ab37141d721f5ca9571e9008a065344b3fb818ac9164fd56cda3c5617952cda9",
    str(D353_EVENT): "0b1d47671fe31206398961dc18f4d66912ba3fd59cf1634059c326a5e67a0b61",
    str(D353_ATTESTATION): "4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5",
    str(D366_HARNESS): "27f6c55e77d62bddca760e0309078029f213f054ee4a7b3537d798188e4a4f61",
    str(D366_RUNTIME): "e7974c5870b9e753a6668b36b78cbcd1aa13a036fd0613a7c7c875efd2ba0c2c",
    str(D366_EXCEPTION): "4ad206fdf93c4ce6bb31c3ec083419f788bb29b2f858ee19282c93a95e5c2d60",
    str(D366_COMPLETION): "cc061e07fca358d18467255a5ad02a6460513eae6bed5cccc463870b9ecf2f7d",
    str(D366_CORRECTION): "2f55c13bbd127654260f318798b2a6a76b0a8dc39cce1efe13502f0137a86312",
}

TIMELINE_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.timeline-1.0.14+69cbf6ad.lx64.r.cp311"
)
FRAME_INTEGRITY_SOURCE = TIMELINE_ROOT / "docs/FRAME_INTEGRITY.md"
TIMELINE_PYI_SOURCE = TIMELINE_ROOT / "omni/timeline/_timeline.pyi"
TIMELINE_TEST_SOURCE = TIMELINE_ROOT / "omni/timeline/tests/tests.py"
CORE_SIM_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py"
)
INSTALLED_SOURCE_HASHES = {
    str(FRAME_INTEGRITY_SOURCE): "1db84fba636fa743bcf98a38561132323ad13fcb89dc91629a06812b123b2e37",
    str(TIMELINE_PYI_SOURCE): "c5a431d83c24de23aefca0912ef819ae2f3322418264b81aba5279d4fe4ac35e",
    str(TIMELINE_TEST_SOURCE): "570b36d310e3f3a307c8a35c38ba277da051e6e1e8fc25da6889e794ad270638",
    str(CORE_SIM_SOURCE): "ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028",
}

PREREG_PATH = OUT_DIR / "d367_preregistration.json"
PREPARE_PATH = OUT_DIR / "d367_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d367_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d367_worker_preflight.json"
WORKER_PHASE_PATH = OUT_DIR / "d367_worker_phase_markers.jsonl"
SUPERVISOR_PHASE_PATH = OUT_DIR / "d367_supervisor_phase_markers.jsonl"
EVENT_ROWS_PATH = OUT_DIR / "d367_timeline_event_rows.jsonl"
CONTRACT_PATH = OUT_DIR / "d367_play_commit_contract.json"
WORKER_SUMMARY_PATH = OUT_DIR / "d367_worker_summary.json"
ZERO_STEP_ATTESTATION_PATH = OUT_DIR / "d367_zero_step_attestation.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d367_worker_exception.json"
CLEANUP_PATH = OUT_DIR / "d367_cleanup_localization.json"
CLEANUP_ENTRY_PATH = OUT_DIR / "d367_cleanup_entry_state.json"
WORKER_LOG_PATH = OUT_DIR / "d367_worker_stdout_stderr.log"
TELEMETRY_PATH = OUT_DIR / "d367_gpu_cpu_telemetry.jsonl"
WATCHDOG_PATH = OUT_DIR / "d367_watchdog_process_snapshot.json"
SUPERVISOR_PATH = OUT_DIR / "d367_supervisor_summary.json"
COMPLETION_PATH = OUT_DIR / "d367_completion_summary.json"

WORKER_TOKEN_ENV = "D367_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D367_SUPERVISOR_PID"

_WORKER_SEQUENCE = 0
_SUPERVISOR_SEQUENCE = 0
_EVENT_ROWS: list[dict[str, Any]] = []
_EVENT_NAMES: dict[int, str] = {}
_ACTIVE_EVENT_PHASE: str | None = None
_TIMELINE: Any = None

_RAW_PLAY_REQUEST_COUNT = 0
_PLAY_COMMIT_ATTEMPT_COUNT = 0
_PLAY_COMMIT_CALL_COUNT = 0
_PLAY_COMMIT_RETURN_COUNT = 0
_INHERITED_PAUSE_COMMIT_COUNT = 0
_PHYSICS_CALLBACK_REGISTER_COUNT = 0
_PHYSICS_CALLBACK_REMOVE_COUNT = 0
_PHYSICS_CALLBACK_COUNT = 0
_PHYSICS_CALLBACK_DTS: list[float] = []
_DISPLAY_STATE_WRITE_COUNT = 0
_CONTROLLED_STEP_CALL_COUNT = 0
_PUBLIC_FORWARD_COUNT = 0
_Q5_SCIENCE_SAMPLE_COUNT = 0
_Q5_TARGET_UPDATE_COUNT = 0
_CONTACT_QUERY_COUNT = 0
_APP_UPDATE_COUNT = 0
_NEXT_FRAME_COUNT = 0
_CALLBACK_NAME = "d367_zero_step_physics_guard"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_sha(path: Path, byte_count: int) -> str:
    with path.open("rb") as stream:
        payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise RuntimeError(f"short prefix read: {path}")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_x(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        payload = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _marker(
    owner: str,
    phase: str,
    event: str,
    details: dict[str, Any] | None = None,
) -> None:
    global _WORKER_SEQUENCE, _SUPERVISOR_SEQUENCE
    if owner == "worker":
        _WORKER_SEQUENCE += 1
        sequence = _WORKER_SEQUENCE
        path = WORKER_PHASE_PATH
    elif owner == "supervisor":
        _SUPERVISOR_SEQUENCE += 1
        sequence = _SUPERVISOR_SEQUENCE
        path = SUPERVISOR_PHASE_PATH
    else:
        raise ValueError(owner)
    _append_jsonl(
        path,
        {
            "sequence": sequence,
            "utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "owner": owner,
            "phase": phase,
            "event": event,
            "details": details or {},
        },
    )


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=REPO, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _git_head(ref: str = "HEAD") -> str:
    return _run_text(["git", "rev-parse", ref])


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_cached_patch() -> dict[str, Any]:
    patch = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {
        "sha256": hashlib.sha256(patch).hexdigest(),
        "byte_count": len(patch),
        "name_status": names,
    }


def _status_scope_ok(rows: list[str]) -> bool:
    allowed_exact = {
        "START_HERE.md",
        "claudedocs/DECISIONS.md",
        "claudedocs/EXPERIMENT_LEDGER.md",
        _rel(HARNESS),
        _rel(SESSION_DOC),
    }
    prefix = f"claudedocs/runtime_logs/grasp_track/{CASE}/"
    for row in rows:
        if " -> " in row:
            return False
        path = row[3:].strip()
        if path not in allowed_exact and not path.startswith(prefix):
            return False
    return True


def _tree_manifest(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(root)),
    ):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    return {
        "root": _rel(root),
        "file_count": len(rows),
        "rows": rows,
        "digest": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _frozen_manifests() -> dict[str, Any]:
    return {name: _tree_manifest(path) for name, path in FROZEN_DIRS.items()}


def _sidecar_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(D334_SIDECAR)): _sha(path)
        for path in sorted(D334_SIDECAR.rglob("*"))
        if path.is_file()
    }


def _frozen_input_hashes() -> dict[str, str]:
    return {path: _sha(Path(path)) for path in FROZEN_INPUT_HASHES}


def _float64_bits(value: Any) -> str:
    return struct.pack("<d", float(value)).hex()


def _float32_payload(values: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype=np.float32)
    payload = array.tobytes()
    return {
        "shape": list(array.shape),
        "values": array.tolist(),
        "bits_hex": payload.hex(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _play_simulations_setting() -> dict[str, Any]:
    try:
        import carb.settings

        value = carb.settings.get_settings().get(PLAY_SIMULATIONS_SETTING)
        return {
            "readable": value is not None,
            "value": None if value is None else bool(value),
            "error": None,
        }
    except Exception as error:
        return {
            "readable": False,
            "value": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _timeline_metadata(timeline: Any) -> dict[str, Any]:
    float_getters = {
        "start_time": timeline.get_start_time,
        "end_time": timeline.get_end_time,
        "tentative_time": timeline.get_tentative_time,
        "target_framerate": timeline.get_target_framerate,
        "ticks_per_second": timeline.get_ticks_per_second,
        "time_codes_per_second": timeline.get_time_codes_per_second,
        "zoom_start_time": timeline.get_zoom_start_time,
        "zoom_end_time": timeline.get_zoom_end_time,
    }
    return {
        "float64_bits": {
            name: _float64_bits(getter()) for name, getter in float_getters.items()
        },
        "current_tick": int(timeline.get_current_tick()),
        "ticks_per_frame": int(timeline.get_ticks_per_frame()),
        "fast_mode": bool(timeline.get_fast_mode()),
        "play_every_frame": bool(timeline.get_play_every_frame()),
        "auto_updating": bool(timeline.is_auto_updating()),
        "looping": bool(timeline.is_looping()),
        "prerolling": bool(timeline.is_prerolling()),
        "zoomed": bool(timeline.is_zoomed()),
        "director_none": timeline.get_director() is None,
    }


def _snapshot(inner: Any, timeline: Any, phase: str) -> dict[str, Any]:
    joints = inner._robot.data.joint_pos[0].detach().cpu().numpy()
    object_pos, object_quat = d366.d362.d351.d334._object_pose_w(inner)
    sim_clock = d366.d362.d351._simulation_clock(inner)
    timeline_time = float(timeline.get_current_time())
    return {
        "phase": phase,
        "timeline_tuple": [bool(timeline.is_playing()), bool(timeline.is_stopped())],
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time": timeline_time,
        "timeline_time_float64_bits": _float64_bits(timeline_time),
        "timeline_metadata": _timeline_metadata(timeline),
        "simulation_context_clock": sim_clock,
        "simulation_context_clock_float64_bits": {
            "current_time": _float64_bits(sim_clock["current_time"])
        },
        "custom_step_counter": int(inner._sim_step_counter),
        "physics_callback_count": _PHYSICS_CALLBACK_COUNT,
        "physics_callback_dts": list(_PHYSICS_CALLBACK_DTS),
        "joint_integrity_float32": _float32_payload(joints),
        "cylinder_position_float32": _float32_payload(object_pos),
        "cylinder_quaternion_float32": _float32_payload(object_quat),
        "play_simulations_setting": _play_simulations_setting(),
        "timeline_commit_counts": {
            "inherited_pause": _INHERITED_PAUSE_COMMIT_COUNT,
            "d367_play_attempt": _PLAY_COMMIT_ATTEMPT_COUNT,
            "d367_play_call": _PLAY_COMMIT_CALL_COUNT,
            "d367_play_return": _PLAY_COMMIT_RETURN_COUNT,
        },
        "science_and_scope_counts": {
            "cylinder_pose_write": _DISPLAY_STATE_WRITE_COUNT,
            "controlled_step_call": _CONTROLLED_STEP_CALL_COUNT,
            "public_forward": _PUBLIC_FORWARD_COUNT,
            "q5_science_sample": _Q5_SCIENCE_SAMPLE_COUNT,
            "q5_target_update": _Q5_TARGET_UPDATE_COUNT,
            "contact_query": _CONTACT_QUERY_COUNT,
            "app_update": _APP_UPDATE_COUNT,
            "next_frame": _NEXT_FRAME_COUNT,
        },
        "thread": {
            "ident": threading.get_ident(),
            "name": threading.current_thread().name,
            "is_main_thread": threading.current_thread() is threading.main_thread(),
        },
    }


def _timeline_event_callback(event: Any) -> None:
    if _TIMELINE is None:
        return
    event_type = int(event.type)
    row = {
        "sequence": len(_EVENT_ROWS) + 1,
        "phase": _ACTIVE_EVENT_PHASE,
        "event_type": event_type,
        "event_name": _EVENT_NAMES.get(event_type, f"UNKNOWN_{event_type}"),
        "callback_monotonic_ns": time.monotonic_ns(),
        "callback_thread_ident": threading.get_ident(),
        "callback_thread_name": threading.current_thread().name,
        "callback_is_main_thread": threading.current_thread()
        is threading.main_thread(),
        "timeline_tuple": [
            bool(_TIMELINE.is_playing()),
            bool(_TIMELINE.is_stopped()),
        ],
        "timeline_time_float64_bits": _float64_bits(
            _TIMELINE.get_current_time()
        ),
        "timeline_current_tick": int(_TIMELINE.get_current_tick()),
        "physics_callback_count": _PHYSICS_CALLBACK_COUNT,
    }
    _EVENT_ROWS.append(row)
    _append_jsonl(EVENT_ROWS_PATH, row)


def _physics_callback(step_size: float) -> None:
    global _PHYSICS_CALLBACK_COUNT
    _PHYSICS_CALLBACK_COUNT += 1
    _PHYSICS_CALLBACK_DTS.append(float(step_size))


def _state_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timeline_time_float64_bits": row["timeline_time_float64_bits"],
        "timeline_metadata": row["timeline_metadata"],
        "simulation_context_clock": row["simulation_context_clock"],
        "simulation_context_clock_float64_bits": row[
            "simulation_context_clock_float64_bits"
        ],
        "custom_step_counter": row["custom_step_counter"],
        "physics_callback_count": row["physics_callback_count"],
        "physics_callback_dts": row["physics_callback_dts"],
        "joint_bits": row["joint_integrity_float32"]["bits_hex"],
        "cylinder_position_bits": row["cylinder_position_float32"]["bits_hex"],
        "cylinder_quaternion_bits": row["cylinder_quaternion_float32"]["bits_hex"],
    }


def _rows_invariant(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        _state_signature(row) == _state_signature(rows[0]) for row in rows
    )


def _scope_counts_zero() -> bool:
    return all(
        count == 0
        for count in (
            _DISPLAY_STATE_WRITE_COUNT,
            _CONTROLLED_STEP_CALL_COUNT,
            _PUBLIC_FORWARD_COUNT,
            _Q5_SCIENCE_SAMPLE_COUNT,
            _Q5_TARGET_UPDATE_COUNT,
            _CONTACT_QUERY_COUNT,
            _APP_UPDATE_COUNT,
            _NEXT_FRAME_COUNT,
        )
    )


def _experimental_play_commit_once(
    inner: Any, timeline: Any
) -> dict[str, Any]:
    global _ACTIVE_EVENT_PHASE, _RAW_PLAY_REQUEST_COUNT
    global _PLAY_COMMIT_ATTEMPT_COUNT, _PLAY_COMMIT_CALL_COUNT
    global _PLAY_COMMIT_RETURN_COUNT

    before = _snapshot(inner, timeline, "before_play_request")
    callback_start = len(_EVENT_ROWS)
    play_start_ns = time.monotonic_ns()
    _RAW_PLAY_REQUEST_COUNT += 1
    play_result = timeline.play()
    play_end_ns = time.monotonic_ns()
    requested = _snapshot(inner, timeline, "post_play_request_pre_commit")
    callbacks_before_commit = copy.deepcopy(_EVENT_ROWS[callback_start:])

    precondition_checks = {
        "before_paused_not_stopped": before["timeline_tuple"] == [False, False],
        "request_remains_pending_paused_not_stopped": requested["timeline_tuple"]
        == [False, False],
        "raw_play_request_count_1": _RAW_PLAY_REQUEST_COUNT == 1,
        "play_return_none": play_result is None,
        "no_callback_before_commit": not callbacks_before_commit,
        "before_request_state_invariant": _rows_invariant([before, requested]),
        "setting_true_before_and_request": all(
            row["play_simulations_setting"]
            == {"readable": True, "value": True, "error": None}
            for row in (before, requested)
        ),
        "main_thread": before["thread"]["is_main_thread"] is True
        and requested["thread"]["is_main_thread"] is True,
        "director_none": before["timeline_metadata"]["director_none"] is True
        and requested["timeline_metadata"]["director_none"] is True,
    }
    if not all(precondition_checks.values()):
        return {
            "precondition_checks": precondition_checks,
            "before": before,
            "request": requested,
            "post": None,
            "canonical_post": None,
            "play_window": {"start_monotonic_ns": play_start_ns, "end_monotonic_ns": play_end_ns},
            "commit_window": None,
            "callbacks_before_commit": callbacks_before_commit,
            "callbacks_during_commit": [],
            "checks": {"precondition_pass": False},
            "failure_class": "D367_PLAY_PENDING_PRECONDITION_STOP",
            "pass": False,
        }

    commit_caller = threading.get_ident()
    commit_start_ns = time.monotonic_ns()
    _PLAY_COMMIT_ATTEMPT_COUNT += 1
    _PLAY_COMMIT_CALL_COUNT += 1
    _ACTIVE_EVENT_PHASE = "d367_discriminating_play_commit"
    try:
        commit_result = timeline.commit()
        _PLAY_COMMIT_RETURN_COUNT += 1
    finally:
        commit_end_ns = time.monotonic_ns()
        _ACTIVE_EVENT_PHASE = None

    post = _snapshot(inner, timeline, "post_play_commit")
    canonical_post = _snapshot(inner, timeline, "post_play_commit_canonical_reread")
    callbacks = copy.deepcopy(_EVENT_ROWS[callback_start:])
    play_event_value = next(
        (value for value, name in _EVENT_NAMES.items() if name == "PLAY"), None
    )
    event_checks = {
        "exact_one_callback": len(callbacks) == 1,
        "exact_play_type": len(callbacks) == 1
        and callbacks[0]["event_type"] == play_event_value == 0,
        "phase_exact": len(callbacks) == 1
        and callbacks[0]["phase"] == "d367_discriminating_play_commit",
        "callback_main_thread": len(callbacks) == 1
        and callbacks[0]["callback_is_main_thread"] is True,
        "callback_thread_matches_caller": len(callbacks) == 1
        and callbacks[0]["callback_thread_ident"] == commit_caller,
        "callback_inside_commit_window": len(callbacks) == 1
        and commit_start_ns
        <= callbacks[0]["callback_monotonic_ns"]
        <= commit_end_ns,
        "callback_sees_playing_not_stopped": len(callbacks) == 1
        and callbacks[0]["timeline_tuple"] == [True, False],
        "callback_time_bits_unchanged": len(callbacks) == 1
        and callbacks[0]["timeline_time_float64_bits"]
        == before["timeline_time_float64_bits"],
        "callback_tick_unchanged": len(callbacks) == 1
        and callbacks[0]["timeline_current_tick"]
        == before["timeline_metadata"]["current_tick"],
        "callback_physics_zero": len(callbacks) == 1
        and callbacks[0]["physics_callback_count"] == 0,
    }
    checks = {
        "precondition_pass": all(precondition_checks.values()),
        "post_playing_not_stopped": post["timeline_tuple"] == [True, False],
        "canonical_post_playing_not_stopped": canonical_post["timeline_tuple"]
        == [True, False],
        "commit_attempt_call_return_exact_1": _PLAY_COMMIT_ATTEMPT_COUNT
        == _PLAY_COMMIT_CALL_COUNT
        == _PLAY_COMMIT_RETURN_COUNT
        == 1,
        "commit_return_none": commit_result is None,
        "event_contract_pass": all(event_checks.values()),
        "all_four_state_signatures_invariant": _rows_invariant(
            [before, requested, post, canonical_post]
        ),
        "setting_true_all_controlled_snapshots": all(
            row["play_simulations_setting"]
            == {"readable": True, "value": True, "error": None}
            for row in (before, requested, post, canonical_post)
        ),
        "main_thread_all_snapshots": all(
            row["thread"]["is_main_thread"] is True
            for row in (before, requested, post, canonical_post)
        ),
        "director_none_all_snapshots": all(
            row["timeline_metadata"]["director_none"] is True
            for row in (before, requested, post, canonical_post)
        ),
        "physics_callback_zero": _PHYSICS_CALLBACK_COUNT == 0
        and not _PHYSICS_CALLBACK_DTS,
        "scope_counts_zero": _scope_counts_zero(),
    }
    failure_class = None
    if not checks["post_playing_not_stopped"]:
        failure_class = "D367_COMMIT_DID_NOT_APPLY_PLAY_FAIL_STOP"
    elif not checks["event_contract_pass"]:
        failure_class = "D367_EVENT_CONTRACT_FAIL_STOP"
    elif not checks["all_four_state_signatures_invariant"]:
        failure_class = "D367_ZERO_STEP_OR_STATE_MUTATION_FAIL_STOP"
    elif not checks["scope_counts_zero"]:
        failure_class = "D367_SCOPE_BREACH_FAIL_STOP"
    elif not all(checks.values()):
        failure_class = "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    return {
        "artifact": "D367_PLAY_COMMIT_CONTRACT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "precondition_checks": precondition_checks,
        "before": before,
        "request": requested,
        "post": post,
        "canonical_post": canonical_post,
        "play_window": {
            "start_monotonic_ns": play_start_ns,
            "end_monotonic_ns": play_end_ns,
        },
        "commit_window": {
            "caller_thread_ident": commit_caller,
            "start_monotonic_ns": commit_start_ns,
            "end_monotonic_ns": commit_end_ns,
        },
        "callbacks_before_commit": callbacks_before_commit,
        "callbacks_during_commit": callbacks,
        "event_checks": event_checks,
        "checks": checks,
        "failure_class": failure_class,
        "pass": all(checks.values()),
    }


def _installed_source_audit() -> dict[str, Any]:
    frame_text = FRAME_INTEGRITY_SOURCE.read_text(encoding="utf-8")
    pyi_text = TIMELINE_PYI_SOURCE.read_text(encoding="utf-8")
    tests_text = TIMELINE_TEST_SOURCE.read_text(encoding="utf-8")
    core_text = CORE_SIM_SOURCE.read_text(encoding="utf-8")
    observed = {path: _sha(Path(path)) for path in INSTALLED_SOURCE_HASHES}
    checks = {
        "hashes_exact": observed == INSTALLED_SOURCE_HASHES,
        "frame_doc_raw_play_deferred": "state change is deferred to the next frame" in frame_text
        and "timeline.play()" in frame_text,
        "frame_doc_commit_immediate": "timeline.commit()" in frame_text
        and "must be used with caution" in frame_text,
        "pyi_commit_pending_callbacks": "Applies all pending state changes and invokes all callbacks" in pyi_text,
        "pyi_commit_main_thread": "not thread-safe" in pyi_text
        and "only from the main thread" in pyi_text,
        "installed_test_play_then_commit": "self._timeline.play()" in tests_text
        and "self._assert_no_change_then_commit(self._timeline.is_playing(), False)" in tests_text
        and "self.assertFalse(self._timeline.is_stopped())" in tests_text,
        "core_public_play_banned_semantics": "def play(self) -> None:" in core_text
        and "self._timeline.play()\n        self._timeline.commit()" in core_text
        and "It does one step internally" in core_text,
    }
    return {
        "observed_hashes": observed,
        "expected_hashes": INSTALLED_SOURCE_HASHES,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: Any = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _static_source_audit() -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    active_names = {
        "_snapshot",
        "_experimental_play_commit_once",
        "_worker",
        "_run",
    }
    active_calls = [
        call
        for name in active_names
        for call in ast.walk(functions[name])
        if isinstance(call, ast.Call)
    ]
    experiment_calls = [
        _call_name(call)
        for call in ast.walk(functions["_experimental_play_commit_once"])
        if isinstance(call, ast.Call)
    ]
    worker_calls = [
        _call_name(call)
        for call in ast.walk(functions["_worker"])
        if isinstance(call, ast.Call)
    ]
    active_call_names = [_call_name(call) for call in active_calls]
    experiment_call_rows = [
        {"name": _call_name(call), "lineno": call.lineno}
        for call in ast.walk(functions["_experimental_play_commit_once"])
        if isinstance(call, ast.Call)
    ]
    phase_snapshot_lines = {
        call.args[2].value: call.lineno
        for call in ast.walk(functions["_experimental_play_commit_once"])
        if isinstance(call, ast.Call)
        and _call_name(call) == "_snapshot"
        and len(call.args) >= 3
        and isinstance(call.args[2], ast.Constant)
        and isinstance(call.args[2].value, str)
    }
    play_lines = [row["lineno"] for row in experiment_call_rows if row["name"] == "timeline.play"]
    commit_lines = [row["lineno"] for row in experiment_call_rows if row["name"] == "timeline.commit"]

    forbidden_exact = {
        "inner.step",
        "inner.sim.step",
        "inner.sim.forward",
        "inner.sim.render",
        "simulation_app.update",
        "timeline.forward_one_frame",
        "timeline.rewind_one_frame",
        "timeline.commit_silently",
        "timeline.stop",
        "timeline.pause",
        "inner._sponge.write_root_pose_to_sim",
        "inner._robot.set_joint_position_target",
        "inner._robot.write_joint_state_to_sim",
    }
    forbidden_suffixes = (
        ".next_update_async",
        ".set_world_poses",
        ".set_joint_positions",
        ".get_contact_force_matrix",
        ".get_contact_data",
    )
    forbidden_hits = [
        name
        for name in active_call_names
        if name in forbidden_exact or name.endswith(forbidden_suffixes)
    ]
    checks = {
        "experiment_raw_play_callsite_exact_1": experiment_calls.count("timeline.play") == 1,
        "experiment_direct_commit_callsite_exact_1": experiment_calls.count("timeline.commit") == 1,
        "controlled_source_order_exact": len(play_lines) == len(commit_lines) == 1
        and set(phase_snapshot_lines)
        == {
            "before_play_request",
            "post_play_request_pre_commit",
            "post_play_commit",
            "post_play_commit_canonical_reread",
        }
        and phase_snapshot_lines["before_play_request"]
        < play_lines[0]
        < phase_snapshot_lines["post_play_request_pre_commit"]
        < commit_lines[0]
        < phase_snapshot_lines["post_play_commit"]
        < phase_snapshot_lines["post_play_commit_canonical_reread"],
        "inherited_pause_helper_worker_callsite_exact_1": worker_calls.count("d366._pause_no_advance") == 1,
        "experiment_worker_callsite_exact_1": worker_calls.count("_experimental_play_commit_once") == 1,
        "physics_callback_add_exact_1": worker_calls.count("inner.sim.add_physics_callback") == 1,
        "physics_callback_remove_exact_1": worker_calls.count("inner.sim.remove_physics_callback") == 1,
        "timeline_subscription_exact_1": worker_calls.count(
            "timeline.get_timeline_event_stream"
        ) == 1,
        "timeline_unsubscribe_exact_1": worker_calls.count("timeline_subscription.unsubscribe") == 1,
        "no_active_forbidden_callsite": not forbidden_hits,
        "no_d353_d366_worker_or_supervisor_call": not any(
            name in {
                "d353._worker",
                "d353._run_worker",
                "d353._run_supervisor",
                "d366._worker",
                "d366._run",
            }
            for name in active_call_names
        ),
        "single_new_variable": NEW_VARIABLES
        == ["explicit_timeline_commit_after_raw_play_request"],
    }
    return {
        "active_functions": sorted(active_names),
        "experiment_calls": experiment_calls,
        "experiment_call_rows": experiment_call_rows,
        "phase_snapshot_lines": phase_snapshot_lines,
        "worker_calls": worker_calls,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _fixture_pass(fixture: dict[str, Any]) -> bool:
    return bool(
        fixture["before"] == [False, False]
        and fixture["request"] == [False, False]
        and fixture["post"] == [True, False]
        and fixture["play_request_count"] == 1
        and fixture["play_commit_attempt"] == 1
        and fixture["play_commit_call"] == 1
        and fixture["inherited_pause_commit"] == 1
        and fixture["total_commit"] == 2
        and fixture["event_count"] == 1
        and fixture["event_type"] == 0
        and fixture["event_phase_exact"]
        and fixture["event_main_thread"]
        and fixture["event_inside_window"]
        and fixture["event_tuple"] == [True, False]
        and fixture["timeline_invariant"]
        and fixture["simulation_clock_invariant"]
        and fixture["custom_counter_invariant"]
        and fixture["physics_callback_zero"]
        and fixture["joint_bits_invariant"]
        and fixture["object_bits_invariant"]
        and fixture["setting_true_readable"]
        and fixture["director_none"]
        and fixture["main_thread"]
        and fixture["scope_counts_zero"]
        and fixture["summary_hash_exact"]
        and fixture["cleanup_complete"]
        and fixture["watchdog_reason"] is None
        and fixture["retry_count"] == 0
        and fixture["invocation_count"] == 1
        and not fixture["process_residue"]
    )


def _negative_controls() -> dict[str, Any]:
    positive = {
        "before": [False, False],
        "request": [False, False],
        "post": [True, False],
        "play_request_count": 1,
        "play_commit_attempt": 1,
        "play_commit_call": 1,
        "inherited_pause_commit": 1,
        "total_commit": 2,
        "event_count": 1,
        "event_type": 0,
        "event_phase_exact": True,
        "event_main_thread": True,
        "event_inside_window": True,
        "event_tuple": [True, False],
        "timeline_invariant": True,
        "simulation_clock_invariant": True,
        "custom_counter_invariant": True,
        "physics_callback_zero": True,
        "joint_bits_invariant": True,
        "object_bits_invariant": True,
        "setting_true_readable": True,
        "director_none": True,
        "main_thread": True,
        "scope_counts_zero": True,
        "summary_hash_exact": True,
        "cleanup_complete": True,
        "watchdog_reason": None,
        "retry_count": 0,
        "invocation_count": 1,
        "process_residue": False,
    }
    mutations: dict[str, tuple[str, Any]] = {
        "before_already_playing": ("before", [True, False]),
        "before_stopped": ("before", [False, True]),
        "request_already_applied": ("request", [True, False]),
        "post_still_paused": ("post", [False, False]),
        "post_stopped": ("post", [False, True]),
        "play_request_zero": ("play_request_count", 0),
        "play_request_two": ("play_request_count", 2),
        "play_commit_zero": ("play_commit_call", 0),
        "play_commit_two": ("play_commit_call", 2),
        "inherited_pause_zero": ("inherited_pause_commit", 0),
        "total_commit_one_lie": ("total_commit", 1),
        "event_missing": ("event_count", 0),
        "event_duplicate": ("event_count", 2),
        "event_pause": ("event_type", 1),
        "event_wrong_phase": ("event_phase_exact", False),
        "event_wrong_thread": ("event_main_thread", False),
        "event_outside_window": ("event_inside_window", False),
        "event_wrong_tuple": ("event_tuple", [False, False]),
        "timeline_drift": ("timeline_invariant", False),
        "sim_clock_drift": ("simulation_clock_invariant", False),
        "custom_counter_drift": ("custom_counter_invariant", False),
        "physics_callback_one": ("physics_callback_zero", False),
        "joint_bit_flip": ("joint_bits_invariant", False),
        "object_bit_flip": ("object_bits_invariant", False),
        "setting_false": ("setting_true_readable", False),
        "director_present": ("director_none", False),
        "caller_not_main": ("main_thread", False),
        "forbidden_scope_count": ("scope_counts_zero", False),
        "summary_hash_tamper": ("summary_hash_exact", False),
        "cleanup_end_missing": ("cleanup_complete", False),
        "watchdog": ("watchdog_reason", "phase_inactivity"),
        "retry_one": ("retry_count", 1),
        "invocation_two": ("invocation_count", 2),
        "process_residue": ("process_residue", True),
    }
    results: dict[str, bool] = {}
    for name, (key, value) in mutations.items():
        fixture = copy.deepcopy(positive)
        fixture[key] = value
        results[name] = not _fixture_pass(fixture)
    checks = {
        "positive_fixture_passes": _fixture_pass(positive),
        "all_mutations_fail": all(results.values()),
        "at_least_30_failure_capable_mutations": len(results) >= 30,
    }
    return {
        "positive": positive,
        "negative_results": results,
        "negative_count": len(results),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _display_audit() -> dict[str, Any]:
    result = subprocess.run(
        ["xdpyinfo", "-display", DISPLAY], text=True, capture_output=True
    )
    return {
        "display": DISPLAY,
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr": result.stderr[-1000:],
        "pass": result.returncode == 0,
    }


def _prepare(_args: argparse.Namespace) -> int:
    output_absent = not OUT_DIR.exists()
    status = _git_status()
    installed = _installed_source_audit()
    static = _static_source_audit()
    negatives = _negative_controls()
    frozen_hashes = _frozen_input_hashes()
    frozen_manifests = _frozen_manifests()
    sidecar = _sidecar_hashes()
    gpu = d366._gpu_snapshot()
    display = _display_audit()
    session_bytes = SESSION_DOC.stat().st_size
    checks = {
        "output_path_absent": output_absent,
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_exact": _status_scope_ok(status),
        "cached_patch_empty": _git_cached_patch()["byte_count"] == 0,
        "session_exists_nonempty": session_bytes > 0,
        "frozen_input_hashes_exact": frozen_hashes == FROZEN_INPUT_HASHES,
        "all_frozen_dirs_exist": all(path.is_dir() for path in FROZEN_DIRS.values()),
        "d334_sidecar_present": bool(sidecar),
        "installed_source_audit": installed["pass"],
        "static_source_audit": static["pass"],
        "negative_controls": negatives["pass"],
        "registered_python": Path(sys.executable).resolve()
        == Path(REGISTERED_PYTHON).resolve(),
        "numpy_pin": np.__version__ == "1.26.0",
        "psutil_pin": psutil.__version__ == "5.9.8",
        "display_available": display["pass"],
        "gpu_is_4090_laptop": "4090" in str(gpu.get("gpu_name", ""))
        and "Laptop" in str(gpu.get("gpu_name", "")),
        "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
        "ram_free_gate": int(gpu.get("ram_available_bytes") or 0)
        >= MIN_RAM_AVAILABLE_BYTES,
    }
    if not output_absent:
        raise RuntimeError("D367 output path already exists; forward-only overwrite forbidden")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    prereg = {
        "artifact": "D367_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "utc": _utc_now(),
        "run_nonce": secrets.token_hex(16),
        "base_git": BASE_GIT,
        "new_variables": NEW_VARIABLES,
        "commit_count_contract": {
            "inherited_initial_pause_commit": 1,
            "d367_discriminating_play_commit": 1,
            "total_runtime_timeline_commit": 2,
            "terminal_cleanup_commit": 0,
        },
        "zero_scope_contract": {
            "cylinder_pose_write": 0,
            "controlled_step_call": 0,
            "public_forward": 0,
            "q5_science_sample": 0,
            "q5_target_update": 0,
            "contact_query": 0,
            "app_update": 0,
            "next_frame": 0,
        },
        "actual_worker_count": 1,
        "automatic_retry": False,
        "watchdogs_seconds": {
            "total": TOTAL_WATCHDOG_S,
            "phase_inactivity": INACTIVITY_WATCHDOG_S,
            "term_grace": TERM_GRACE_S,
            "kill_grace": KILL_GRACE_S,
        },
        "harness_sha256": _sha(HARNESS),
        "session_preregistration_bytes": session_bytes,
        "session_preregistration_prefix_sha256": _prefix_sha(
            SESSION_DOC, session_bytes
        ),
        "git_status_at_prepare": status,
        "git_cached_patch": _git_cached_patch(),
        "frozen_input_hashes": frozen_hashes,
        "frozen_manifests_before": frozen_manifests,
        "d334_sidecar_before": sidecar,
        "installed_source_audit": installed,
        "static_source_audit": static,
        "negative_controls": negatives,
        "gpu_and_ram": gpu,
        "display_audit": display,
        "rerun_omitted": True,
        "rerun_omission_reason": "pure same-frame API/event/clock/opaque-bit control audit; no spatial, contact, trajectory, or sensor-time judgment",
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    prepare = {
        "artifact": "D367_PREPARE_PREFLIGHT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "preregistration_sha256": _sha(PREREG_PATH),
        "check_count": len(checks),
        "true_count": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "pass": prereg["pass"],
    }
    _write_json_x(PREPARE_PATH, prepare)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "pass": prepare["pass"],
                "checks": f"{prepare['true_count']}/{prepare['check_count']}",
                "negative_controls": negatives["negative_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if prepare["pass"] else 2


def _worker(args: argparse.Namespace) -> int:
    global _TIMELINE, _ACTIVE_EVENT_PHASE, _INHERITED_PAUSE_COMMIT_COUNT
    global _PHYSICS_CALLBACK_REGISTER_COUNT, _PHYSICS_CALLBACK_REMOVE_COUNT

    simulation_app = None
    inner = None
    settings = None
    timeline = None
    timeline_subscription = None
    physics_callback_registered = False
    bridge_pass = False
    authoritative_zero: int | None = None
    cleanup_stages: list[dict[str, Any]] = []
    cleanup_entry: dict[str, Any] | None = None
    cleanup_after_safety: dict[str, Any] | None = None
    cleanup_commit_count_entry: int | None = None
    worker_failure_class: str | None = None

    def cleanup_stage(name: str, callable_: Any) -> None:
        row = {"phase": name, "start_utc": _utc_now(), "end_utc": None, "error": None}
        cleanup_stages.append(row)
        _marker("worker", name, "start")
        try:
            callable_()
            row["end_utc"] = _utc_now()
            _marker("worker", name, "end")
        except Exception as error:
            row["end_utc"] = _utc_now()
            row["error"] = f"{type(error).__name__}: {error}"
            _marker("worker", name, "error", {"error": row["error"]})

    try:
        prereg = _json(PREREG_PATH)
        prepare = _json(PREPARE_PATH)
        invocation = _json(INVOCATION_PATH)
        token = os.environ.get(WORKER_TOKEN_ENV, "")
        supervisor_pid = int(os.environ.get(SUPERVISOR_PID_ENV, "-1"))
        early_modules = sorted(
            name
            for name in sys.modules
            if name in {"pxr", "omni", "isaaclab", "isaacsim", "carb", "usdrt"}
            or name.startswith(
                ("pxr.", "omni.", "isaaclab.", "isaacsim.", "carb.", "usdrt.")
            )
        )
        gpu = d366._gpu_snapshot()
        session_bytes = int(prereg["session_preregistration_bytes"])
        checks = {
            "prereg_prepare_pass": prereg.get("pass") is True
            and prepare.get("pass") is True,
            "single_invocation": invocation.get("invocation_index") == 1
            and invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("automatic_retry") is False,
            "registered_parent": supervisor_pid > 0 and os.getppid() == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest()
            == invocation.get("worker_token_sha256"),
            "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
            "git_scope_exact": _status_scope_ok(_git_status()),
            "cached_patch_exact": _git_cached_patch() == prereg["git_cached_patch"],
            "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
            "session_prefix_exact": _prefix_sha(SESSION_DOC, session_bytes)
            == prereg["session_preregistration_prefix_sha256"],
            "frozen_hashes_exact": _frozen_input_hashes()
            == prereg["frozen_input_hashes"],
            "frozen_manifests_exact": _frozen_manifests()
            == prereg["frozen_manifests_before"],
            "sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "installed_source_exact": _installed_source_audit()
            == prereg["installed_source_audit"],
            "registered_python": Path(sys.executable).resolve()
            == Path(REGISTERED_PYTHON).resolve(),
            "display_device_exact": os.environ.get("DISPLAY") == DISPLAY
            and args.headless is False
            and int(args.livestream) == 0
            and str(args.device) == "cuda:0",
            "runtime_modules_absent_before_applauncher": not early_modules,
            "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
            "ram_free_gate": int(gpu.get("ram_available_bytes") or 0)
            >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D367_WORKER_PREFLIGHT_V1",
            "case": CASE,
            "utc": _utc_now(),
            "pid": os.getpid(),
            "early_runtime_modules": early_modules,
            "gpu_and_ram": gpu,
            "checks": checks,
            "pass": all(checks.values()),
        }
        _write_json_x(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker", "worker_preflight", "end", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D367 worker preflight STOP: {checks}")

        from isaaclab.app import AppLauncher

        _marker("worker", "AppLauncher", "start")
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
        launcher_report = d366.d362.d351.d350._resolved_gui_launcher(launcher)
        _marker(
            "worker", "AppLauncher", "end", {"pass": launcher_report.get("pass")}
        )
        if not launcher_report.get("pass"):
            raise RuntimeError(f"D367 launcher contract STOP: {launcher_report}")

        import carb.settings
        import omni.timeline

        args.robot_usd_path = d366.d362.VARIANT_ROBOT_USD
        _marker("worker", "make_runtime_env", "start")
        inner = d366.d362._make_runtime_env(args)
        _marker("worker", "make_runtime_env", "end")
        timeline = omni.timeline.get_timeline_interface()
        _TIMELINE = timeline
        settings = carb.settings.get_settings()

        _marker("worker", "reset", "start")
        inner.reset(seed=SEED)
        _marker(
            "worker",
            "reset",
            "end",
            {
                "simulation_clock": d366.d362.d351._simulation_clock(inner),
                "custom_step_counter": int(inner._sim_step_counter),
                "timeline_tuple": [timeline.is_playing(), timeline.is_stopped()],
            },
        )

        for name in (
            "PLAY",
            "PAUSE",
            "STOP",
            "CURRENT_TIME_CHANGED",
            "CURRENT_TIME_TICKED_PERMANENT",
            "CURRENT_TIME_TICKED",
            "LOOP_MODE_CHANGED",
            "AUTO_UPDATE_CHANGED",
            "DIRECTOR_CHANGED",
        ):
            value = getattr(omni.timeline.TimelineEventType, name, None)
            if value is not None:
                raw = value.value if hasattr(value, "value") else value
                _EVENT_NAMES[int(raw)] = name
        if _EVENT_NAMES.get(0) != "PLAY" or _EVENT_NAMES.get(1) != "PAUSE":
            raise RuntimeError(f"D367 runtime event enum drift: {_EVENT_NAMES}")

        _marker("worker", "timeline_event_subscription", "start")
        timeline_subscription = (
            timeline.get_timeline_event_stream().create_subscription_to_pop(
                _timeline_event_callback,
                name="D367 exact PLAY commit observer",
            )
        )
        _marker("worker", "timeline_event_subscription", "end")

        initial_callback_start = len(_EVENT_ROWS)
        initial_before = _snapshot(inner, timeline, "inherited_pause_before")
        _ACTIVE_EVENT_PHASE = "inherited_initial_pause_commit"
        _marker("worker", "inherited_initial_pause", "start")
        try:
            inherited_pause = d366._pause_no_advance(inner, timeline)
        finally:
            _ACTIVE_EVENT_PHASE = None
        _INHERITED_PAUSE_COMMIT_COUNT = int(inherited_pause["commit_count"])
        initial_after = _snapshot(inner, timeline, "inherited_pause_after")
        initial_callbacks = copy.deepcopy(_EVENT_ROWS[initial_callback_start:])
        inherited_checks = {
            "helper_pass": inherited_pause["pass"] is True,
            "before_playing_not_stopped": initial_before["timeline_tuple"]
            == [True, False],
            "after_paused_not_stopped": initial_after["timeline_tuple"]
            == [False, False],
            "inherited_pause_commit_exact_1": _INHERITED_PAUSE_COMMIT_COUNT == 1,
            "state_signature_invariant": _rows_invariant(
                [initial_before, initial_after]
            ),
            "exact_one_pause_callback": len(initial_callbacks) == 1
            and initial_callbacks[0]["event_type"] == 1
            and initial_callbacks[0]["event_name"] == "PAUSE"
            and initial_callbacks[0]["phase"]
            == "inherited_initial_pause_commit"
            and initial_callbacks[0]["callback_is_main_thread"] is True,
            "physics_callback_zero": _PHYSICS_CALLBACK_COUNT == 0,
            "scope_counts_zero": _scope_counts_zero(),
        }
        _marker(
            "worker",
            "inherited_initial_pause",
            "end",
            {"pass": all(inherited_checks.values())},
        )
        if not all(inherited_checks.values()):
            raise RuntimeError(f"D367 inherited PAUSE prerequisite STOP: {inherited_checks}")

        _marker("worker", "physics_callback_registration", "start")
        inner.sim.add_physics_callback(_CALLBACK_NAME, _physics_callback)
        physics_callback_registered = True
        _PHYSICS_CALLBACK_REGISTER_COUNT += 1
        _marker("worker", "physics_callback_registration", "end")
        if not inner.sim.physics_callback_exists(_CALLBACK_NAME):
            raise RuntimeError("D367 physics callback registration missing")

        setting_before = _snapshot(inner, timeline, "before_play_simulations_true")
        _marker("worker", "play_simulations_true", "start")
        settings.set(PLAY_SIMULATIONS_SETTING, True)
        _marker("worker", "play_simulations_true", "end")
        setting_after = _snapshot(inner, timeline, "after_play_simulations_true")
        setting_checks = {
            "state_signature_invariant": _rows_invariant(
                [setting_before, setting_after]
            ),
            "before_false": setting_before["play_simulations_setting"]
            == {"readable": True, "value": False, "error": None},
            "after_true": setting_after["play_simulations_setting"]
            == {"readable": True, "value": True, "error": None},
            "timeline_still_paused": setting_after["timeline_tuple"]
            == [False, False],
            "physics_callback_zero": _PHYSICS_CALLBACK_COUNT == 0,
        }
        if not all(setting_checks.values()):
            raise RuntimeError(f"D367 playSimulations setting confound STOP: {setting_checks}")

        _marker("worker", "experimental_play_commit", "start")
        contract = _experimental_play_commit_once(inner, timeline)
        _marker(
            "worker",
            "experimental_play_commit",
            "end",
            {"pass": contract["pass"], "failure_class": contract.get("failure_class")},
        )
        contract.update(
            {
                "new_variables": NEW_VARIABLES,
                "inherited_pause": {
                    "helper": inherited_pause,
                    "before": initial_before,
                    "after": initial_after,
                    "timeline_callbacks": initial_callbacks,
                    "checks": inherited_checks,
                    "pass": all(inherited_checks.values()),
                },
                "play_simulations_setting_transition": {
                    "before": setting_before,
                    "after": setting_after,
                    "checks": setting_checks,
                    "pass": all(setting_checks.values()),
                },
                "counter_contract": {
                    "inherited_pause_commit": _INHERITED_PAUSE_COMMIT_COUNT,
                    "d367_play_commit_attempt": _PLAY_COMMIT_ATTEMPT_COUNT,
                    "d367_play_commit_call": _PLAY_COMMIT_CALL_COUNT,
                    "d367_play_commit_return": _PLAY_COMMIT_RETURN_COUNT,
                    "total_runtime_timeline_commit": _INHERITED_PAUSE_COMMIT_COUNT
                    + _PLAY_COMMIT_CALL_COUNT,
                    "raw_play_request": _RAW_PLAY_REQUEST_COUNT,
                    "physics_callback_register": _PHYSICS_CALLBACK_REGISTER_COUNT,
                    "physics_callback_invocation": _PHYSICS_CALLBACK_COUNT,
                    "cylinder_pose_write": _DISPLAY_STATE_WRITE_COUNT,
                    "controlled_step_call": _CONTROLLED_STEP_CALL_COUNT,
                    "public_forward": _PUBLIC_FORWARD_COUNT,
                    "q5_science_sample": _Q5_SCIENCE_SAMPLE_COUNT,
                    "q5_target_update": _Q5_TARGET_UPDATE_COUNT,
                    "contact_query": _CONTACT_QUERY_COUNT,
                    "app_update": _APP_UPDATE_COUNT,
                    "next_frame": _NEXT_FRAME_COUNT,
                },
            }
        )
        counter_checks = {
            "commit_categories_exact_1_1_total_2": _INHERITED_PAUSE_COMMIT_COUNT
            == _PLAY_COMMIT_CALL_COUNT
            == 1
            and _INHERITED_PAUSE_COMMIT_COUNT + _PLAY_COMMIT_CALL_COUNT == 2,
            "raw_play_exact_1": _RAW_PLAY_REQUEST_COUNT == 1,
            "physics_callback_registered_once_invoked_zero": _PHYSICS_CALLBACK_REGISTER_COUNT
            == 1
            and _PHYSICS_CALLBACK_COUNT == 0
            and not _PHYSICS_CALLBACK_DTS,
            "scope_counts_zero": _scope_counts_zero(),
        }
        contract["counter_checks"] = counter_checks
        contract["pass"] = contract["pass"] and all(counter_checks.values())
        if not contract["pass"] and contract.get("failure_class") is None:
            contract["failure_class"] = "D367_SCOPE_BREACH_FAIL_STOP"
        _write_json_x(CONTRACT_PATH, contract)

        summary = {
            "artifact": "D367_WORKER_SUMMARY_V1",
            "case": CASE,
            "utc": _utc_now(),
            "new_variables": NEW_VARIABLES,
            "contract_path": _rel(CONTRACT_PATH),
            "contract_sha256": _sha(CONTRACT_PATH),
            "contract_pass": contract["pass"],
            "failure_class": contract.get("failure_class"),
            "commit_count_contract": contract["counter_contract"],
            "timeline_event_row_count": len(_EVENT_ROWS),
            "physics_callback_dts": list(_PHYSICS_CALLBACK_DTS),
            "controlled_physics_steps_candidate": 0
            if contract["pass"]
            else None,
            "controlled_physics_steps": None,
            "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "contact_query_count": _CONTACT_QUERY_COUNT,
            "physical_science_recomputed": False,
            "cap_rim_science": None,
            "grasp_or_g0a_science": None,
            "g0a_pass": False,
            "pass_before_attestation": contract["pass"],
        }
        _write_json_x(WORKER_SUMMARY_PATH, summary)
        summary_sha = _sha(WORKER_SUMMARY_PATH)
        summary_reread = _json(WORKER_SUMMARY_PATH)
        attestation_checks = {
            "contract_pass": contract["pass"] is True,
            "summary_exact_reread": summary_reread == summary,
            "summary_hash_nonempty": len(summary_sha) == 64,
            "summary_binds_contract_hash": summary_reread["contract_sha256"]
            == _sha(CONTRACT_PATH),
            "scope_counts_zero": _scope_counts_zero(),
            "physics_callback_zero": _PHYSICS_CALLBACK_COUNT == 0
            and not _PHYSICS_CALLBACK_DTS,
            "commit_counts_exact": _INHERITED_PAUSE_COMMIT_COUNT == 1
            and _PLAY_COMMIT_ATTEMPT_COUNT
            == _PLAY_COMMIT_CALL_COUNT
            == _PLAY_COMMIT_RETURN_COUNT
            == 1,
        }
        attestation = {
            "artifact": "D367_ZERO_STEP_ATTESTATION_V1",
            "case": CASE,
            "utc": _utc_now(),
            "worker_summary_path": _rel(WORKER_SUMMARY_PATH),
            "worker_summary_sha256": summary_sha,
            "worker_summary_bytes": WORKER_SUMMARY_PATH.stat().st_size,
            "contract_path": _rel(CONTRACT_PATH),
            "contract_sha256": _sha(CONTRACT_PATH),
            "checks": attestation_checks,
            "controlled_physics_steps": 0
            if all(attestation_checks.values())
            else None,
            "pass": all(attestation_checks.values()),
        }
        _write_json_x(ZERO_STEP_ATTESTATION_PATH, attestation)
        authoritative_zero = attestation["controlled_physics_steps"]
        bridge_pass = attestation["pass"]
        if not bridge_pass:
            worker_failure_class = "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    except Exception as error:
        worker_failure_class = worker_failure_class or (
            "D367_COMMIT_EXECUTION_FAIL_STOP"
            if _PLAY_COMMIT_CALL_COUNT > _PLAY_COMMIT_RETURN_COUNT
            else "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
        )
        if not WORKER_EXCEPTION_PATH.exists():
            _write_json_x(
                WORKER_EXCEPTION_PATH,
                {
                    "artifact": "D367_WORKER_EXCEPTION_STOP_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "failure_class": worker_failure_class,
                    "raw_play_request_count": _RAW_PLAY_REQUEST_COUNT,
                    "inherited_pause_commit_count": _INHERITED_PAUSE_COMMIT_COUNT,
                    "d367_play_commit_attempt_count": _PLAY_COMMIT_ATTEMPT_COUNT,
                    "d367_play_commit_call_count": _PLAY_COMMIT_CALL_COUNT,
                    "d367_play_commit_return_count": _PLAY_COMMIT_RETURN_COUNT,
                    "physics_callback_count": _PHYSICS_CALLBACK_COUNT,
                    "cylinder_pose_write_count": _DISPLAY_STATE_WRITE_COUNT,
                    "controlled_step_call_count": _CONTROLLED_STEP_CALL_COUNT,
                    "public_forward_count": _PUBLIC_FORWARD_COUNT,
                    "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "contact_query_count": _CONTACT_QUERY_COUNT,
                    "controlled_physics_steps": authoritative_zero,
                    "g0a_pass": False,
                    "pass": False,
                },
            )
        try:
            _marker(
                "worker",
                "worker_exception",
                "error",
                {"error": f"{type(error).__name__}: {error}"},
            )
        except Exception:
            pass
    finally:
        cleanup_commit_count_entry = (
            _INHERITED_PAUSE_COMMIT_COUNT + _PLAY_COMMIT_CALL_COUNT
        )
        if inner is not None and timeline is not None:
            try:
                _marker("worker", "cleanup_entry_snapshot", "start")
                cleanup_entry = _snapshot(inner, timeline, "cleanup_entry_before_safety_false")
                _marker(
                    "worker",
                    "cleanup_entry_snapshot",
                    "end",
                    {"timeline_tuple": cleanup_entry["timeline_tuple"]},
                )
            except Exception as error:
                _marker(
                    "worker",
                    "cleanup_entry_snapshot",
                    "error",
                    {"error": f"{type(error).__name__}: {error}"},
                )
        # Keep physics disabled while destroying this process.  This is a cleanup
        # safety write, not a timeline state request and not a controlled frame.
        if settings is not None:
            cleanup_stage(
                "cleanup_play_simulations_safety_false",
                lambda: settings.set(PLAY_SIMULATIONS_SETTING, False),
            )
        if inner is not None and timeline is not None and cleanup_entry is not None:
            try:
                _marker("worker", "cleanup_safety_snapshot", "start")
                cleanup_after_safety = _snapshot(
                    inner, timeline, "cleanup_after_play_simulations_safety_false"
                )
                cleanup_entry_checks = {
                    "state_signature_invariant": _rows_invariant(
                        [cleanup_entry, cleanup_after_safety]
                    ),
                    "play_simulations_false": cleanup_after_safety[
                        "play_simulations_setting"
                    ]
                    == {"readable": True, "value": False, "error": None},
                    "physics_callback_zero": _PHYSICS_CALLBACK_COUNT == 0
                    and not _PHYSICS_CALLBACK_DTS,
                    "scope_counts_zero": _scope_counts_zero(),
                }
                cleanup_entry_artifact = {
                    "artifact": "D367_CLEANUP_ENTRY_STATE_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "before_safety_false": cleanup_entry,
                    "after_safety_false": cleanup_after_safety,
                    "checks": cleanup_entry_checks,
                    "pass": all(cleanup_entry_checks.values()),
                }
                _write_json_x(CLEANUP_ENTRY_PATH, cleanup_entry_artifact)
                _marker(
                    "worker",
                    "cleanup_safety_snapshot",
                    "end",
                    {
                        "pass": cleanup_entry_artifact["pass"],
                        "timeline_tuple": cleanup_after_safety["timeline_tuple"],
                    },
                )
            except Exception as error:
                _marker(
                    "worker",
                    "cleanup_safety_snapshot",
                    "error",
                    {"error": f"{type(error).__name__}: {error}"},
                )
        if timeline_subscription is not None:
            cleanup_stage(
                "cleanup_timeline_subscription_release",
                lambda: timeline_subscription.unsubscribe(),
            )
            timeline_subscription = None
        if physics_callback_registered and inner is not None:
            def remove_callback() -> None:
                global _PHYSICS_CALLBACK_REMOVE_COUNT
                if inner.sim.physics_callback_exists(_CALLBACK_NAME):
                    inner.sim.remove_physics_callback(_CALLBACK_NAME)
                if inner.sim.physics_callback_exists(_CALLBACK_NAME):
                    raise RuntimeError("D367 physics callback still exists after removal")
                _PHYSICS_CALLBACK_REMOVE_COUNT += 1

            cleanup_stage("cleanup_physics_callback_remove", remove_callback)
        if inner is not None:
            cleanup_stage("cleanup_inner_close", inner.close)
        if simulation_app is not None:
            cleanup_stage("cleanup_simulation_app_close", simulation_app.close)

        cleanup_checks = {
            "all_started_stages_ended_without_error": bool(cleanup_stages)
            and all(row["end_utc"] is not None and row["error"] is None for row in cleanup_stages),
            "timeline_subscription_release_recorded": any(
                row["phase"] == "cleanup_timeline_subscription_release"
                and row["error"] is None
                for row in cleanup_stages
            ),
            "physics_callback_removed_once": _PHYSICS_CALLBACK_REGISTER_COUNT == 1
            and _PHYSICS_CALLBACK_REMOVE_COUNT == 1
            and any(
                row["phase"] == "cleanup_physics_callback_remove"
                and row["error"] is None
                for row in cleanup_stages
            ),
            "inner_close_completed": any(
                row["phase"] == "cleanup_inner_close" and row["error"] is None
                for row in cleanup_stages
            ),
            "simulation_app_close_completed": any(
                row["phase"] == "cleanup_simulation_app_close"
                and row["error"] is None
                for row in cleanup_stages
            ),
            "timeline_commit_count_unchanged_during_cleanup": (
                _INHERITED_PAUSE_COMMIT_COUNT + _PLAY_COMMIT_CALL_COUNT
                == cleanup_commit_count_entry
            ),
            "cleanup_entry_state_durable_pass": CLEANUP_ENTRY_PATH.is_file()
            and _json(CLEANUP_ENTRY_PATH).get("pass") is True,
        }
        cleanup = {
            "artifact": "D367_CLEANUP_LOCALIZATION_V1",
            "case": CASE,
            "utc": _utc_now(),
            "stages": cleanup_stages,
            "physics_callback_register_count": _PHYSICS_CALLBACK_REGISTER_COUNT,
            "physics_callback_remove_count": _PHYSICS_CALLBACK_REMOVE_COUNT,
            "timeline_commit_total": _INHERITED_PAUSE_COMMIT_COUNT
            + _PLAY_COMMIT_CALL_COUNT,
            "timeline_commit_count_at_cleanup_entry": cleanup_commit_count_entry,
            "checks": cleanup_checks,
            "pass": all(cleanup_checks.values()),
        }
        try:
            _write_json_x(CLEANUP_PATH, cleanup)
            _marker(
                "worker",
                "worker_finally_complete",
                "end",
                {"bridge_pass": bridge_pass, "cleanup_pass": cleanup["pass"]},
            )
        except Exception:
            pass

    cleanup_pass = CLEANUP_PATH.is_file() and _json(CLEANUP_PATH).get("pass") is True
    if bridge_pass and cleanup_pass:
        return 0
    return 2


def _process_snapshot(worker_pid: int, process_group: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "ppid", "status", "name", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            if pid == worker_pid or os.getpgid(pid) == process_group:
                rows.append(
                    {
                        "pid": pid,
                        "ppid": process.info["ppid"],
                        "status": process.info["status"],
                        "name": process.info["name"],
                        "cmdline": process.info["cmdline"],
                    }
                )
        except (psutil.NoSuchProcess, ProcessLookupError, PermissionError):
            continue
    return {
        "utc": _utc_now(),
        "worker_pid": worker_pid,
        "process_group": process_group,
        "rows": rows,
    }


def _gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    rows: list[int] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                rows.append(int(line.strip()))
            except ValueError:
                continue
    return rows


def _run(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    prepare = _json(PREPARE_PATH)
    if prereg.get("pass") is not True or prepare.get("pass") is not True:
        raise RuntimeError("D367 prepare did not pass")
    if INVOCATION_PATH.exists() or WORKER_LOG_PATH.exists():
        raise RuntimeError("D367 actual invocation already consumed; retry forbidden")
    if _git_head() != _git_head("origin/master") or _git_head() != BASE_GIT:
        raise RuntimeError("D367 Git HEAD/origin drift before invocation")
    if not _status_scope_ok(_git_status()):
        raise RuntimeError("D367 worktree scope drift before invocation")

    token = secrets.token_hex(32)
    command = [
        REGISTERED_PYTHON,
        "-B",
        str(HARNESS),
        "--stage",
        "_worker",
        "--out_dir",
        str(OUT_DIR),
        "--seed",
        str(SEED),
    ]
    invocation = {
        "artifact": "D367_ISAAC_INVOCATION_MARKER_V1",
        "case": CASE,
        "utc": _utc_now(),
        "invocation_index": 1,
        "automatic_retry": False,
        "run_nonce": prereg["run_nonce"],
        "supervisor_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "command": command,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _marker("supervisor", "worker_launch", "start", {"command": command})
    log_fd = os.open(WORKER_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    log_stream = os.fdopen(log_fd, "wb", buffering=0)
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    env[WORKER_TOKEN_ENV] = token
    env[SUPERVISOR_PID_ENV] = str(os.getpid())
    worker = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process_group = os.getpgid(worker.pid)
    _marker(
        "supervisor",
        "worker_launch",
        "end",
        {"worker_pid": worker.pid, "process_group": process_group},
    )

    start = time.monotonic()
    last_progress = start
    last_phase_bytes: int | None = None
    telemetry: list[dict[str, Any]] = []
    watchdog_reason: str | None = None
    term_sent = False
    kill_sent = False
    while worker.poll() is None:
        now = time.monotonic()
        phase_bytes = WORKER_PHASE_PATH.stat().st_size if WORKER_PHASE_PATH.exists() else 0
        log_bytes = WORKER_LOG_PATH.stat().st_size if WORKER_LOG_PATH.exists() else 0
        if phase_bytes != last_phase_bytes:
            last_phase_bytes = phase_bytes
            last_progress = now
        if not telemetry or now - telemetry[-1]["monotonic_s"] >= 2.0:
            try:
                process = psutil.Process(worker.pid)
                rss = process.memory_info().rss
                cpu = process.cpu_percent(interval=None)
            except psutil.Error:
                rss = None
                cpu = None
            row = {
                "sequence": len(telemetry) + 1,
                "utc": _utc_now(),
                "monotonic_s": now,
                "elapsed_seconds": now - start,
                "worker_pid": worker.pid,
                "worker_rss_bytes": rss,
                "worker_cpu_percent": cpu,
                "gpu": d366._gpu_snapshot(),
                "phase_bytes": phase_bytes,
                "log_bytes": log_bytes,
            }
            telemetry.append(row)
            _append_jsonl(TELEMETRY_PATH, row)
        if now - start > TOTAL_WATCHDOG_S:
            watchdog_reason = "total_runtime"
            break
        if now - last_progress > INACTIVITY_WATCHDOG_S:
            watchdog_reason = "phase_inactivity"
            break
        time.sleep(1.0)

    if watchdog_reason is not None and worker.poll() is None:
        snapshot = _process_snapshot(worker.pid, process_group)
        snapshot["watchdog_reason"] = watchdog_reason
        _write_json_x(WATCHDOG_PATH, snapshot)
        _marker(
            "supervisor",
            "watchdog_termination",
            "start",
            {"reason": watchdog_reason, "process_group": process_group},
        )
        os.killpg(process_group, signal.SIGTERM)
        term_sent = True
        try:
            worker.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(process_group, signal.SIGKILL)
            kill_sent = True
            try:
                worker.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        _marker(
            "supervisor",
            "watchdog_termination",
            "end",
            {"term_sent": term_sent, "kill_sent": kill_sent},
        )
    else:
        worker.wait()
    log_stream.close()
    elapsed = time.monotonic() - start

    time.sleep(1.0)
    residue = _process_snapshot(worker.pid, process_group)
    worker_gpu_present = worker.pid in _gpu_compute_pids()
    contract = _json(CONTRACT_PATH) if CONTRACT_PATH.is_file() else None
    summary = _json(WORKER_SUMMARY_PATH) if WORKER_SUMMARY_PATH.is_file() else None
    attestation = (
        _json(ZERO_STEP_ATTESTATION_PATH)
        if ZERO_STEP_ATTESTATION_PATH.is_file()
        else None
    )
    cleanup = _json(CLEANUP_PATH) if CLEANUP_PATH.is_file() else None
    exception = _json(WORKER_EXCEPTION_PATH) if WORKER_EXCEPTION_PATH.is_file() else None
    phase_rows = []
    if WORKER_PHASE_PATH.is_file():
        phase_rows = [
            json.loads(line)
            for line in WORKER_PHASE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    post_checks = {
        "worker_exit_zero": worker.returncode == 0,
        "watchdog_absent": watchdog_reason is None,
        "contract_pass": contract is not None and contract.get("pass") is True,
        "worker_summary_pass": summary is not None
        and summary.get("pass_before_attestation") is True,
        "zero_step_attestation_pass": attestation is not None
        and attestation.get("pass") is True
        and attestation.get("controlled_physics_steps") == 0,
        "cleanup_pass": cleanup is not None and cleanup.get("pass") is True,
        "worker_exception_absent": exception is None,
        "terminal_marker_present": any(
            row.get("phase") == "worker_finally_complete"
            and row.get("event") == "end"
            for row in phase_rows
        ),
        "process_group_absent": not residue["rows"],
        "worker_gpu_allocation_absent": not worker_gpu_present,
        "single_invocation_no_retry": _json(INVOCATION_PATH)["invocation_index"] == 1
        and _json(INVOCATION_PATH)["automatic_retry"] is False,
        "head_origin_unchanged": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_unchanged": _status_scope_ok(_git_status()),
        "cached_patch_unchanged": _git_cached_patch() == prereg["git_cached_patch"],
        "frozen_hashes_unchanged": _frozen_input_hashes()
        == prereg["frozen_input_hashes"],
        "frozen_manifests_unchanged": _frozen_manifests()
        == prereg["frozen_manifests_before"],
        "sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
    }
    supervisor = {
        "artifact": "D367_SUPERVISOR_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "command": command,
        "worker_pid": worker.pid,
        "process_group": process_group,
        "worker_exit_code": worker.returncode,
        "elapsed_seconds": elapsed,
        "watchdog_reason": watchdog_reason,
        "termination": {"sigterm_sent": term_sent, "sigkill_sent": kill_sent},
        "automatic_retry": False,
        "telemetry_samples": len(telemetry),
        "resource_summary": {
            "gpu_used_mib_max": max(
                (row["gpu"].get("memory_used_mib") or 0 for row in telemetry),
                default=None,
            ),
            "gpu_free_mib_min": min(
                (row["gpu"].get("memory_free_mib") or 10**9 for row in telemetry),
                default=None,
            ),
            "gpu_utilization_percent_max": max(
                (row["gpu"].get("utilization_gpu_percent") or 0 for row in telemetry),
                default=None,
            ),
            "worker_rss_bytes_max": max(
                (row.get("worker_rss_bytes") or 0 for row in telemetry), default=None
            ),
        },
        "postrun_process_snapshot": residue,
        "worker_gpu_allocation_present": worker_gpu_present,
        "contract_sha256": _sha(CONTRACT_PATH) if CONTRACT_PATH.is_file() else None,
        "worker_summary_sha256": _sha(WORKER_SUMMARY_PATH)
        if WORKER_SUMMARY_PATH.is_file()
        else None,
        "zero_step_attestation_sha256": _sha(ZERO_STEP_ATTESTATION_PATH)
        if ZERO_STEP_ATTESTATION_PATH.is_file()
        else None,
        "cleanup_sha256": _sha(CLEANUP_PATH) if CLEANUP_PATH.is_file() else None,
        "worker_exception": exception,
        "post_checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _marker("supervisor", "supervisor_summary", "end", {"pass": supervisor["pass"]})

    if supervisor["pass"]:
        verdict = "D367_TIMELINE_PLAY_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE"
    elif (
        attestation is not None
        and attestation.get("pass") is True
        and (
            watchdog_reason is not None
            or (cleanup is not None and cleanup.get("pass") is not True)
        )
    ):
        verdict = "D367_POST_BRIDGE_CLEANUP_FAIL_STOP"
    elif exception is not None:
        verdict = exception.get("failure_class", "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP")
    elif contract is not None:
        verdict = contract.get("failure_class") or "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    else:
        verdict = "D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    controlled_steps = (
        attestation.get("controlled_physics_steps")
        if attestation is not None and attestation.get("pass") is True
        else None
    )
    contract_counts = contract.get("counter_contract", {}) if contract else {}

    def observed_count(contract_key: str, exception_key: str) -> Any:
        if contract_key in contract_counts:
            return contract_counts[contract_key]
        if exception is not None and exception_key in exception:
            return exception[exception_key]
        return None

    artifact_hashes = {
        _rel(path): _sha(path)
        for path in sorted(OUT_DIR.iterdir())
        if path.is_file() and path != COMPLETION_PATH
    }
    completion = {
        "artifact": "D367_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "final_verdict": verdict,
        "operational_bridge_pass": supervisor["pass"],
        "controlled_physics_steps": controlled_steps,
        "raw_play_request_count": observed_count(
            "raw_play_request", "raw_play_request_count"
        ),
        "inherited_pause_commit_count": observed_count(
            "inherited_pause_commit", "inherited_pause_commit_count"
        ),
        "d367_play_commit_attempt_count": observed_count(
            "d367_play_commit_attempt", "d367_play_commit_attempt_count"
        ),
        "d367_play_commit_call_count": observed_count(
            "d367_play_commit_call", "d367_play_commit_call_count"
        ),
        "d367_play_commit_return_count": observed_count(
            "d367_play_commit_return", "d367_play_commit_return_count"
        ),
        "total_runtime_timeline_commit_count": observed_count(
            "total_runtime_timeline_commit", "total_runtime_timeline_commit_count"
        ),
        "cylinder_pose_write_count": observed_count(
            "cylinder_pose_write", "cylinder_pose_write_count"
        ),
        "controlled_step_call_count": observed_count(
            "controlled_step_call", "controlled_step_call_count"
        ),
        "public_forward_count": observed_count(
            "public_forward", "public_forward_count"
        ),
        "q5_science_sample_count": observed_count(
            "q5_science_sample", "q5_science_sample_count"
        ),
        "q5_target_update_count": observed_count(
            "q5_target_update", "q5_target_update_count"
        ),
        "contact_query_count": observed_count(
            "contact_query", "contact_query_count"
        ),
        "physical_science_recomputed": False,
        "current_pose_grasp_science": None,
        "cap_rim_science": None,
        "physx_fabric_hydra_science": None,
        "target_ik_path_changed": False,
        "g0a_pass": False,
        "supervisor_sha256": _sha(SUPERVISOR_PATH),
        "artifact_hashes_before_completion": artifact_hashes,
        "pass": supervisor["pass"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(
        json.dumps(
            {
                "stage": "run",
                "pass": completion["pass"],
                "verdict": verdict,
                "controlled_physics_steps": controlled_steps,
                "worker_exit": worker.returncode,
                "watchdog": watchdog_reason,
            },
            ensure_ascii=False,
        )
    )
    return 0 if completion["pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "run", "_worker"), required=True
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D367 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D367 seed drift")
    if args.stage == "prepare":
        return _prepare(args)
    if args.stage == "run":
        return _run(args)
    args.headless = False
    args.livestream = 0
    args.device = "cuda:0"
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
