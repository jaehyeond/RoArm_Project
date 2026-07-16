#!/usr/bin/env python3
"""D357: render the frozen D354 result without rerunning its science.

The only Isaac-side operations are one environment reset, the inherited
conditional Timeline.commit() pause bridge, three direct display-state writes,
and guarded UI/render updates.  No distance/contact query or physics step is
allowed.  The authoritative numbers are copied from immutable D354 JSON.
"""
from __future__ import annotations

import argparse
import copy
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
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# This import graph is intentionally usable before SimulationApp.  The worker
# explicitly proves that pxr/omni/isaaclab runtime modules are still absent.
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d351_zero_step_closure_geometry as d351,
)


CASE = "g0a_d357"
CASE_NAME = "d354_beginner_result_visualization_repair"
EXPECTED_HEAD = "161f6d9d185bb41eb29259349ee0fd897a3c6de8"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
SEED = 33201
INACTIVITY_WATCHDOG_S = 120.0
TOTAL_WATCHDOG_S = 300.0
VIEWER_HOLD_SECONDS = 60.0
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SIDE_CAMERA_EYE = [0.285, -0.42, 0.09]
SIDE_CAMERA_TARGET = [0.285, 0.0, 0.055]
PHYSX_COLLIDER_SETTING = "/persistent/physics/visualizationDisplayColliders"
PLAY_SIMULATIONS_SETTING = "/app/player/playSimulations"
WORKER_TOKEN_ENV = "D357_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D357_SUPERVISOR_PID"
EXPECTED_RERUN_NON_SYSTEM_ENTITY_COUNT = 62
EXPECTED_RERUN_NON_SYSTEM_ENTITY_SHA256 = (
    "2a872536c08c44ed9ba00a82f82ad72f4109039bb86df521eed0c92857866ef2"
)
EXPECTED_RERUN_TIMELINES = ["blueprint", "log_time", "step"]

Q5_STATES = [
    ("open", "OPEN / 열림", np.float32(1.5413000583648682)),
    ("last_clear", "LAST CLEAR / 마지막 비접촉", np.float32(1.0269782543182373)),
    ("first_overlap", "FIRST OVERLAP / 첫 형상 겹침", np.float32(1.0269775390625)),
]

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d357"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260716_grasp_g0a_d357_d354_beginner_result_visualization_repair.md"
START_HERE = REPO / "START_HERE.md"
D354_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d354"
D354_MEASUREMENT = D354_DIR / "d354_zero_step_closure_geometry_measurement.json"
D354_ATTESTATION = D354_DIR / "d354_zero_step_science_attestation.json"
D354_COMPLETION = D354_DIR / "d354_completion_summary.json"
D354_HASHES = {
    D354_MEASUREMENT: "fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed",
    D354_ATTESTATION: "1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20",
    D354_COMPLETION: "5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86",
}

PREREG_PATH = OUT_DIR / "d357_preregistration.json"
PREPARE_PATH = OUT_DIR / "d357_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d357_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d357_worker_preflight.json"
PHASE_PATH = OUT_DIR / "d357_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d357_worker_stdout_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d357_worker_summary.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d357_worker_exception.json"
SUPERVISOR_PATH = OUT_DIR / "d357_supervisor_summary.json"
AUTOMATED_PATH = OUT_DIR / "d357_automated_summary.json"
MANUAL_PATH = OUT_DIR / "d357_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d357_completion_summary.json"
RRD_PATH = OUT_DIR / "d357_d354_result_visualization.rrd"
RBL_PATH = OUT_DIR / "d357_d354_result_visualization.rbl"
RERUN_PNG_PATH = OUT_DIR / "d357_d354_result_visualization_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d357_rerun_validation.json"
POSTPROCESS_EXCEPTION_PATH = OUT_DIR / "d357_postprocess_exception.json"
SHEET_PATH = OUT_DIR / "d357_three_pose_beginner_sheet_ko.png"
ACTUAL_SECTION_PATH = OUT_DIR / "d357_contact_section_actual_scale_ko.png"
MAGNIFIED_SECTION_PATH = OUT_DIR / "d357_contact_section_z50000_display_only_ko.png"
OCCLUSION_ADDENDUM_PATH = OUT_DIR / "d357_isaac_camera_occlusion_addendum_ko.png"
OCCLUSION_ADDENDUM_SUMMARY_PATH = OUT_DIR / "d357_isaac_camera_occlusion_addendum_summary.json"
CAPTURE_PATHS = {
    "open": OUT_DIR / "d357_open_same_camera_actual_physx.png",
    "last_clear": OUT_DIR / "d357_last_clear_same_camera_actual_physx.png",
    "first_overlap": OUT_DIR / "d357_first_overlap_same_camera_actual_physx.png",
}

_PHASE_SEQUENCE = 0
_DISPLAY_WRITE_ATTEMPTS = 0
_DISPLAY_WRITE_SUCCESSES = 0
_UI_UPDATES = 0
_FORBIDDEN_CALL_ATTEMPTS = {
    "q5_science_evaluator": 0,
    "first_contact_certifier": 0,
    "measurement_classifier": 0,
    "moving_surface_binding": 0,
    "overlap_contact_audit": 0,
    "physics_helper": 0,
    "simulation_context_step": 0,
}


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


def _write_json_x(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
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
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
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
    return result.stdout.splitlines()


def _status_scope_ok(rows: list[str]) -> bool:
    allowed = (
        "START_HERE.md",
        "claudedocs/DECISIONS.md",
        "claudedocs/EXPERIMENT_LEDGER.md",
        "claudedocs/session_20260716_grasp_g0a_d356_",
        "claudedocs/session_20260716_grasp_g0a_d357_",
        "sim_scripts/cyl34_top_view_d357_",
        "roarm_rl/rerun_contract.py",
        "roarm_rl/viz_debug.py",
        "claudedocs/runtime_logs/grasp_track/g0a_d357/",
    )
    return all(any(row[3:].startswith(prefix) for prefix in allowed) for row in rows)


def _sidecar_hashes() -> dict[str, str]:
    root = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
    return {_rel(path): _sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    row: dict[str, Any] = {"command": command, "returncode": result.returncode, "stderr": result.stderr.strip()}
    if result.returncode == 0 and result.stdout.strip():
        fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
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


def _prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"forward-only D357 output already exists: {OUT_DIR}")
    hashes = {_rel(path): _sha(path) if path.is_file() else None for path in D354_HASHES}
    expected_hashes = {_rel(path): value for path, value in D354_HASHES.items()}
    status = _git_status()
    gpu = _gpu_snapshot()
    try:
        rerun_version = subprocess.run(
            [str(RERUN_CLI), "--version"], text=True, capture_output=True, check=False
        )
    except OSError as error:
        rerun_version = subprocess.CompletedProcess(
            [str(RERUN_CLI), "--version"],
            returncode=127,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
        )
    x_display = subprocess.run(
        ["xdpyinfo", "-display", ":1"], text=True, capture_output=True, check=False
    )
    try:
        import rerun as rr
        from PIL import ImageFont

        loaded_font = ImageFont.truetype(str(FONT_PATH), size=36)
        font_korean_bbox = list(loaded_font.getbbox("한글 시각화"))
        rerun_sdk_version = str(rr.__version__)
        font_load_error = None
    except Exception as error:
        font_korean_bbox = None
        rerun_sdk_version = None
        font_load_error = f"{type(error).__name__}: {error}"
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
        "git_scope_only_registered_state_and_d357": _status_scope_ok(status),
        "registered_python_exact": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
        "numpy_pin_1p26p0": np.__version__ == "1.26.0",
        "psutil_pin_5p9p8": psutil.__version__ == "5.9.8",
        "display_exact": os.environ.get("DISPLAY") == ":1",
        "display_x_server_reachable": x_display.returncode == 0
        and "name of display:    :1" in x_display.stdout,
        "font_exists": FONT_PATH.is_file(),
        "font_loads_and_korean_bbox_nonempty": font_load_error is None
        and font_korean_bbox is not None
        and font_korean_bbox[2] > font_korean_bbox[0]
        and font_korean_bbox[3] > font_korean_bbox[1],
        "rerun_sdk_version_0p34p1": rerun_sdk_version == "0.34.1",
        "rerun_cli_exact_path_executable": RERUN_CLI.is_file() and os.access(RERUN_CLI, os.X_OK),
        "rerun_cli_version_0p34p1": rerun_version.returncode == 0
        and "rerun-cli 0.34.1" in rerun_version.stdout,
        "session_preregistered": SESSION_DOC.is_file() and "Registered execution order" in SESSION_DOC.read_text(encoding="utf-8"),
        "d354_hashes_exact": hashes == expected_hashes,
        "d354_science_frozen_unresolved": _json(D354_MEASUREMENT).get("classification", {}).get("scientific_verdict")
        == "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
        "gpu_is_rtx4090_laptop": gpu.get("name") == "NVIDIA GeForce RTX 4090 Laptop GPU",
        "gpu_compute_capability_sm89": gpu.get("compute_capability") == "8.9",
        "gpu_free_at_least_8gib": int(gpu.get("memory_free_mib", 0)) >= 8192,
        "ram_available_at_least_12gib": int(gpu.get("ram_available_bytes", 0)) >= 12 * 1024**3,
    }
    prepare = {
        "artifact": "D357_PREPARE_PREFLIGHT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "git": {"head": _git_head(), "origin_master": _git_head("origin/master"), "status": status},
        "environment": {"python": str(Path(sys.executable).resolve()), "numpy": np.__version__, "psutil": psutil.__version__, "display": os.environ.get("DISPLAY")},
        "rerun_cli": {
            "path": str(RERUN_CLI),
            "returncode": rerun_version.returncode,
            "stdout": rerun_version.stdout.strip(),
            "stderr": rerun_version.stderr.strip(),
        },
        "postprocess_dependency_preflight": {
            "rerun_sdk_version": rerun_sdk_version,
            "font_path": str(FONT_PATH),
            "font_korean_bbox": font_korean_bbox,
            "error": font_load_error,
        },
        "x_display": {
            "command": ["xdpyinfo", "-display", ":1"],
            "returncode": x_display.returncode,
            "stderr": x_display.stderr.strip(),
            "display_line": next(
                (line for line in x_display.stdout.splitlines() if line.startswith("name of display:")),
                None,
            ),
        },
        "gpu_and_ram": gpu,
        "d354_hashes": hashes,
        "d334_sidecar_before": _sidecar_hashes(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not prepare["pass"]:
        print(json.dumps({"stage": "prepare", "pass": False, "checks": checks}, ensure_ascii=False))
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json_x(PREPARE_PATH, prepare)
    prereg = {
        "artifact": "D357_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "run_nonce": secrets.token_hex(16),
        "base_git": EXPECTED_HEAD,
        "new_variables": ["d354_beginner_result_visualization_contract"],
        "new_physical_variables": [],
        "frozen_q5_states_rad_float32": {name: float(q5) for name, _, q5 in Q5_STATES},
        "same_camera": {"eye": SIDE_CAMERA_EYE, "target": SIDE_CAMERA_TARGET},
        "exact_counters": {
            "isaac_visualization_invocations": 1,
            "display_only_q5_state_writes": 3,
            "q5_science_evaluator_invocations": 0,
            "distance_queries": 0,
            "contact_or_overlap_queries": 0,
            "new_classifications": 0,
            "controlled_physics_steps": 0,
        },
        "watchdogs_seconds": {"inactivity": INACTIVITY_WATCHDOG_S, "total": TOTAL_WATCHDOG_S},
        "viewer_hold_seconds": VIEWER_HOLD_SECONDS,
        "registered_command": [REGISTERED_PYTHON, _rel(HARNESS), "--stage", "run"],
        "worker_command_shape": [REGISTERED_PYTHON, _rel(HARNESS), "--stage", "_worker"],
        "d354_hashes": hashes,
        "harness_sha256": _sha(HARNESS),
        "session_sha256_at_prepare": _sha(SESSION_DOC),
        "prepare_sha256": _sha(PREPARE_PATH),
        "rerun_exact_inventory": {
            "non_system_entity_count": EXPECTED_RERUN_NON_SYSTEM_ENTITY_COUNT,
            "non_system_entity_paths_sha256": EXPECTED_RERUN_NON_SYSTEM_ENTITY_SHA256,
            "timeline_names": EXPECTED_RERUN_TIMELINES,
        },
        "prohibitions": [
            "no D354 evaluator or classifier",
            "no distance/contact/overlap query",
            "no physics step",
            "no target/IK/path or asset/physics configuration change",
            "no retry or overwrite",
        ],
        "pass": prepare["pass"],
    }
    _write_json_x(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"], "output": _rel(OUT_DIR)}, ensure_ascii=False))
    return 0 if prereg["pass"] else 2


def _mutation_bits(inner: Any) -> dict[str, Any]:
    joints = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    obj_pos, obj_quat = d351.d334._object_pose_w(inner)
    return {
        "joint_bits": joints.tobytes().hex(),
        "object_position_bits": np.asarray(obj_pos, dtype=np.float32).tobytes().hex(),
        "object_quaternion_bits": np.asarray(obj_quat, dtype=np.float32).tobytes().hex(),
        "object_linear_velocity_bits": inner._sponge.data.root_lin_vel_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .tobytes()
        .hex(),
        "object_angular_velocity_bits": inner._sponge.data.root_ang_vel_w[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .tobytes()
        .hex(),
    }


def _snapshot(inner: Any, timeline: Any, q5_expected: float) -> dict[str, Any]:
    joints = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    obj_pos, obj_quat = d351.d334._object_pose_w(inner)
    expected = d351.Q_FROZEN_F32.copy()
    expected[5] = np.float32(q5_expected)
    object_linear_velocity = (
        inner._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float32)
    )
    object_angular_velocity = (
        inner._sponge.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float32)
    )
    simulation_clock = d351._simulation_clock(inner)
    return {
        "counter": int(inner._sim_step_counter),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time": float(timeline.get_current_time()),
        "simulation_clock": simulation_clock,
        "joint_bits": joints.tobytes().hex(),
        "object_position_bits": np.asarray(obj_pos, dtype=np.float32).tobytes().hex(),
        "object_quaternion_bits": np.asarray(obj_quat, dtype=np.float32).tobytes().hex(),
        "object_linear_velocity_bits": object_linear_velocity.tobytes().hex(),
        "object_angular_velocity_bits": object_angular_velocity.tobytes().hex(),
        "checks": {
            "joint_exact": np.array_equal(joints, expected),
            "q0_q4_exact": np.array_equal(joints[:5], d351.Q_FROZEN_F32[:5]),
            "q5_exact": joints[5].tobytes() == np.float32(q5_expected).tobytes(),
            "object_position_exact": np.array_equal(np.asarray(obj_pos, dtype=np.float32), d351.OBJECT_POS_F32),
            "object_quaternion_exact": np.array_equal(np.asarray(obj_quat, dtype=np.float32), d351.OBJECT_QUAT_F32),
            "object_linear_velocity_zero_exact": np.array_equal(
                object_linear_velocity, np.zeros_like(object_linear_velocity)
            ),
            "object_angular_velocity_zero_exact": np.array_equal(
                object_angular_velocity, np.zeros_like(object_angular_velocity)
            ),
            "counter_zero": int(inner._sim_step_counter) == 0,
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "simulation_context_clock_available": all(
                value is not None for value in simulation_clock.values()
            ),
        },
    }


def _same_sentinel(before: dict[str, Any], after: dict[str, Any]) -> bool:
    keys = (
        "counter",
        "timeline_time",
        "simulation_clock",
        "joint_bits",
        "object_position_bits",
        "object_quaternion_bits",
        "object_linear_velocity_bits",
        "object_angular_velocity_bits",
    )
    return all(before[key] == after[key] for key in keys) and all(after["checks"].values())


def _pump_guarded(simulation_app: Any, inner: Any, timeline: Any, q5_expected: float, count: int) -> int:
    global _UI_UPDATES
    for _ in range(count):
        inner.sim.set_setting("/app/player/playSimulations", False)
        before = _snapshot(inner, timeline, q5_expected)
        if not all(before["checks"].values()):
            raise RuntimeError(f"D357 pre-update zero-step sentinel failed: {before['checks']}")
        simulation_app.update()
        after = _snapshot(inner, timeline, q5_expected)
        if not _same_sentinel(before, after):
            raise RuntimeError("D357 UI update advanced or mutated the zero-step state")
        _UI_UPDATES += 1
        if _UI_UPDATES == 1 or _UI_UPDATES % 50 == 0:
            _marker("ui_render_pump", "heartbeat", {"updates": _UI_UPDATES, "q5": float(np.float32(q5_expected))})
    return count


def _display_write(inner: Any, timeline: Any, q5: float, role: str) -> dict[str, Any]:
    global _DISPLAY_WRITE_ATTEMPTS, _DISPLAY_WRITE_SUCCESSES
    if _DISPLAY_WRITE_ATTEMPTS >= 3:
        raise RuntimeError("D357 display write count would exceed three")
    _DISPLAY_WRITE_ATTEMPTS += 1
    _marker(
        "display_state",
        "attempt",
        {"role": role, "q5": float(np.float32(q5)), "attempt_count": _DISPLAY_WRITE_ATTEMPTS},
    )
    guard = d351._set_state_only(inner, timeline, float(np.float32(q5)))
    _DISPLAY_WRITE_SUCCESSES += 1
    row = _snapshot(inner, timeline, float(np.float32(q5)))
    if not guard.get("pass") or not all(row["checks"].values()):
        raise RuntimeError(f"D357 display-only state guard failed at {role}")
    _marker(
        "display_state",
        "success",
        {
            "role": role,
            "q5": float(np.float32(q5)),
            "attempt_count": _DISPLAY_WRITE_ATTEMPTS,
            "success_count": _DISPLAY_WRITE_SUCCESSES,
            "semantics": "frozen full display state reassertion; only q5 differs",
        },
    )
    return {"role": role, "q5_rad_float32": float(np.float32(q5)), "d351_guard": guard, "snapshot": row}


def _capture_viewport(path: Path, simulation_app: Any, inner: Any, timeline: Any, q5: float) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utility

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("D357 active viewport is unavailable")
    observed_resolution = list(viewport.get_texture_resolution())
    if observed_resolution != [1280, 720]:
        raise RuntimeError(f"D357 viewport resolution drift before capture: {observed_resolution}")
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(capture.wait_for_result(completion_frames=5), run_until_complete=False)
    deadline = time.monotonic() + 30.0
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        _pump_guarded(simulation_app, inner, timeline, q5, 1)
    if not task.done():
        task.cancel()
        raise RuntimeError(f"D357 capture timeout: {path.name}")
    result = bool(task.result())
    _pump_guarded(simulation_app, inner, timeline, q5, 3)
    if not result:
        raise RuntimeError(f"D357 capture failed: {path.name}")
    return {"path": _rel(path), "capture_result": result, "postclose_decode_pending": True}


def _timeline_pause_commit_bridge(inner: Any, timeline: Any, settings: Any) -> dict[str, Any]:
    before = {
        "playing": bool(timeline.is_playing()),
        "stopped": bool(timeline.is_stopped()),
        "time": float(timeline.get_current_time()),
        "clock": d351._simulation_clock(inner),
        "counter": int(inner._sim_step_counter),
        "state_bits": _mutation_bits(inner),
    }
    pause_requested = False
    pending = copy.deepcopy(before)
    commit_count = 0
    if timeline.is_playing():
        timeline.pause()
        pause_requested = True
        pending = {
            "playing": bool(timeline.is_playing()),
            "stopped": bool(timeline.is_stopped()),
            "time": float(timeline.get_current_time()),
            "clock": d351._simulation_clock(inner),
            "counter": int(inner._sim_step_counter),
            "state_bits": _mutation_bits(inner),
        }
        if pending["playing"] and not pending["stopped"]:
            timeline.commit()
            commit_count = 1
    after = {
        "playing": bool(timeline.is_playing()),
        "stopped": bool(timeline.is_stopped()),
        "time": float(timeline.get_current_time()),
        "clock": d351._simulation_clock(inner),
        "counter": int(inner._sim_step_counter),
        "state_bits": _mutation_bits(inner),
    }
    checks = {
        "pause_or_already_paused": pause_requested or not before["playing"],
        "commit_at_most_once": commit_count in (0, 1),
        "commit_only_if_pending": commit_count == 0 or (pending["playing"] and not pending["stopped"]),
        "after_paused_not_stopped": not after["playing"] and not after["stopped"],
        "time_unchanged": before["time"] == pending["time"] == after["time"],
        "clock_unchanged": before["clock"] == pending["clock"] == after["clock"],
        "counter_zero_unchanged": before["counter"] == pending["counter"] == after["counter"] == 0,
        "joint_and_object_bits_unchanged": before["state_bits"] == pending["state_bits"] == after["state_bits"],
        "simulation_context_clock_available": all(
            value is not None
            for row in (before, pending, after)
            for value in row["clock"].values()
        ),
        "play_simulations_false_readback_before_bridge": settings.get(
            PLAY_SIMULATIONS_SETTING
        )
        is False,
    }
    report = {"before": before, "pending": pending, "after": after, "pause_requested": pause_requested, "commit_count": commit_count, "checks": checks, "pass": all(checks.values())}
    _marker("timeline_pause_commit_bridge", "complete", {"pass": report["pass"], "commit_count": commit_count})
    return report


def _worker(args: argparse.Namespace) -> int:
    simulation_app = None
    inner = None
    settings = None
    previous_physx: Any = None
    previous_play_setting: Any = None
    previous_physx_captured = False
    previous_play_captured = False
    sim_step_original: Any = None
    forbidden_originals: list[tuple[Any, str, Any]] = []
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
            "exclusive_invocation_marker_exact": invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("invocation_index") == 1
            and invocation.get("automatic_retry") is False,
            "parent_is_registered_supervisor": supervisor_pid > 0
            and os.getppid() == supervisor_pid
            and invocation.get("supervisor_pid") == supervisor_pid,
            "one_time_worker_token_exact": bool(token)
            and hashlib.sha256(token.encode("utf-8")).hexdigest()
            == invocation.get("worker_token_sha256"),
            "head_origin_exact": _git_head() == _git_head("origin/master") == EXPECTED_HEAD,
            "harness_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "registered_python_exact": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
            "display_exact": os.environ.get("DISPLAY") == ":1",
            "headless_false": args.headless is False,
            "livestream_zero": int(args.livestream) == 0,
            "device_cuda0": str(args.device) == "cuda:0",
            "no_pxr_omni_isaaclab_isaacsim_carb_runtime_before_applauncher": not early_runtime_modules,
            "d354_hashes_exact": {_rel(path): _sha(path) for path in D354_HASHES} == prereg.get("d354_hashes"),
            "gpu_free_at_least_8gib": int(gpu.get("memory_free_mib", 0)) >= 8192,
            "ram_available_at_least_10gib": int(gpu.get("ram_available_bytes", 0)) >= 10 * 1024**3,
        }
        preflight = {"artifact": "D357_WORKER_PREFLIGHT_V1", "utc": _utc_now(), "pid": os.getpid(), "early_runtime_modules": early_runtime_modules, "gpu_and_ram": gpu, "checks": checks, "pass": all(checks.values())}
        _write_json_x(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker_preflight", "complete", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D357 worker preflight STOP: {checks}")

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
            raise RuntimeError(f"D357 GUI launcher contract STOP: {launcher_report}")

        import carb
        import omni.timeline

        args.robot_usd_path = d351.VARIANT_ROBOT_USD
        _marker("make_runtime_env", "start")
        inner = d351.d333._make_runtime_env(args)
        _marker("make_runtime_env", "complete")
        timeline = omni.timeline.get_timeline_interface()
        _marker("reset", "start")
        inner.reset(seed=SEED)
        _marker("reset", "complete", {"counter": int(inner._sim_step_counter), "timeline_playing": bool(timeline.is_playing())})
        settings = carb.settings.get_settings()
        previous_play_setting = settings.get(PLAY_SIMULATIONS_SETTING)
        previous_play_captured = True
        inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
        bridge = _timeline_pause_commit_bridge(inner, timeline, settings)
        if not bridge["pass"]:
            raise RuntimeError("D357 inherited pause/commit bridge failed")

        previous_physx = settings.get(PHYSX_COLLIDER_SETTING)
        previous_physx_captured = True
        settings.set(PHYSX_COLLIDER_SETTING, 2)

        def _forbidden_trap(counter_name: str) -> Any:
            def trap(*trap_args: Any, **trap_kwargs: Any) -> Any:
                _FORBIDDEN_CALL_ATTEMPTS[counter_name] += 1
                raise RuntimeError(f"D357 forbidden runtime call attempted: {counter_name}")

            return trap

        for owner, name, counter_name in (
            (d351, "_evaluate_q5", "q5_science_evaluator"),
            (d351, "_certify_first_contact", "first_contact_certifier"),
            (d351, "_classify_measurement", "measurement_classifier"),
            (d351, "_bind_moving_surface", "moving_surface_binding"),
            (d351, "_overlap_contact_surface_audit", "overlap_contact_audit"),
            (d351.d332, "_physics_step", "physics_helper"),
        ):
            original = getattr(owner, name)
            forbidden_originals.append((owner, name, original))
            setattr(owner, name, _forbidden_trap(counter_name))
        sim_step_original = inner.sim.step
        inner.sim.step = _forbidden_trap("simulation_context_step")

        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport is None or not hasattr(viewport, "set_texture_resolution"):
            raise RuntimeError("D357 viewport resolution API unavailable")
        viewport.set_texture_resolution((1280, 720))
        states: list[dict[str, Any]] = []
        captures: dict[str, Any] = {}
        for role, label, q5 in Q5_STATES:
            states.append(_display_write(inner, timeline, float(q5), role))
            inner.sim.set_camera_view(SIDE_CAMERA_EYE, SIDE_CAMERA_TARGET)
            _pump_guarded(simulation_app, inner, timeline, float(q5), 12)
            captures[role] = _capture_viewport(CAPTURE_PATHS[role], simulation_app, inner, timeline, float(q5))
            _marker("viewport_capture", "complete", {"role": role, "label": label, "path": _rel(CAPTURE_PATHS[role])})

        hold_start = time.monotonic()
        _marker("first_overlap_inspection_hold", "start", {"seconds": VIEWER_HOLD_SECONDS, "refresh_hz": 10})
        while time.monotonic() - hold_start < VIEWER_HOLD_SECONDS and simulation_app.is_running():
            _pump_guarded(simulation_app, inner, timeline, float(Q5_STATES[-1][2]), 1)
            time.sleep(0.1)
        hold_elapsed = time.monotonic() - hold_start
        _marker("first_overlap_inspection_hold", "complete", {"elapsed_seconds": hold_elapsed})

        if previous_physx is None:
            settings.destroy_item(PHYSX_COLLIDER_SETTING)
        else:
            settings.set(PHYSX_COLLIDER_SETTING, previous_physx)
        if previous_play_setting is None:
            settings.destroy_item(PLAY_SIMULATIONS_SETTING)
        else:
            settings.set(PLAY_SIMULATIONS_SETTING, previous_play_setting)
        final = _snapshot(inner, timeline, float(Q5_STATES[-1][2]))
        counters = {
            "isaac_visualization_invocations": 1,
            "display_only_q5_state_writes": _DISPLAY_WRITE_SUCCESSES,
            "display_state_write_attempts": _DISPLAY_WRITE_ATTEMPTS,
            "q5_science_evaluator_invocations": _FORBIDDEN_CALL_ATTEMPTS["q5_science_evaluator"],
            "distance_queries": _FORBIDDEN_CALL_ATTEMPTS["first_contact_certifier"],
            "contact_or_overlap_queries": _FORBIDDEN_CALL_ATTEMPTS["overlap_contact_audit"],
            "new_classifications": _FORBIDDEN_CALL_ATTEMPTS["measurement_classifier"],
            "controlled_physics_step_attempts": _FORBIDDEN_CALL_ATTEMPTS["physics_helper"]
            + _FORBIDDEN_CALL_ATTEMPTS["simulation_context_step"],
            "controlled_physics_steps": 0,
            "ui_render_updates": _UI_UPDATES,
        }
        exact = prereg["exact_counters"]
        checks = {
            "display_write_attempts_and_successes_exact_three": _DISPLAY_WRITE_ATTEMPTS
            == _DISPLAY_WRITE_SUCCESSES
            == 3,
            "all_display_states_guarded": len(states) == 3 and all(row["d351_guard"].get("pass") for row in states),
            "all_captures_requested": set(captures) == set(CAPTURE_PATHS),
            "final_zero_step_state": all(final["checks"].values()),
            "counter_zero": final["counter"] == 0,
            "timeline_time_zero_unchanged": final["timeline_time"] == bridge["after"]["time"],
            "physics_debug_setting_restored": settings.get(PHYSX_COLLIDER_SETTING) == previous_physx,
            "play_simulations_setting_restored": settings.get(PLAY_SIMULATIONS_SETTING)
            == previous_play_setting,
            "all_forbidden_call_attempts_zero": all(value == 0 for value in _FORBIDDEN_CALL_ATTEMPTS.values()),
            "registered_zero_counters_exact": all(counters[key] == value for key, value in exact.items()),
            "sidecar_unchanged": _sidecar_hashes() == _json(PREPARE_PATH)["d334_sidecar_before"],
        }
        summary = {
            "artifact": "D357_WORKER_SUMMARY_V1",
            "case": CASE,
            "launcher": launcher_report,
            "timeline_bridge": bridge,
            "states": states,
            "captures": captures,
            "hold_elapsed_seconds": hold_elapsed,
            "final_snapshot": final,
            "counters": counters,
            "target_ik_path_changed": False,
            "asset_or_physics_configuration_changed": False,
            "scientific_verdict_inherited": "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
            "g0a_pass": False,
            "checks": checks,
            "pass": all(checks.values()),
        }
        _write_json_x(WORKER_SUMMARY_PATH, summary)
        _marker("worker_summary", "complete", {"pass": summary["pass"]})
        return 0 if summary["pass"] else 2
    except Exception as error:
        if not WORKER_EXCEPTION_PATH.exists():
            _write_json_x(
                WORKER_EXCEPTION_PATH,
                {
                    "artifact": "D357_WORKER_EXCEPTION_STOP_V1",
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "display_write_attempts": _DISPLAY_WRITE_ATTEMPTS,
                    "display_write_successes": _DISPLAY_WRITE_SUCCESSES,
                    "forbidden_call_attempts": _FORBIDDEN_CALL_ATTEMPTS,
                    "controlled_physics_steps": None,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        if sim_step_original is not None and inner is not None:
            try:
                inner.sim.step = sim_step_original
            except Exception:
                pass
        for owner, name, original in reversed(forbidden_originals):
            try:
                setattr(owner, name, original)
            except Exception:
                pass
        if settings is not None:
            try:
                if previous_physx_captured:
                    if previous_physx is None:
                        settings.destroy_item(PHYSX_COLLIDER_SETTING)
                    else:
                        settings.set(PHYSX_COLLIDER_SETTING, previous_physx)
                if previous_play_captured:
                    if previous_play_setting is None:
                        settings.destroy_item(PLAY_SIMULATIONS_SETTING)
                    else:
                        settings.set(PLAY_SIMULATIONS_SETTING, previous_play_setting)
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


def _png_report(path: Path) -> dict[str, Any]:
    from PIL import Image

    if not path.is_file() or path.stat().st_size <= 0:
        return {"path": _rel(path), "ok": False, "error": "missing or empty"}
    with Image.open(path) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    return {"path": _rel(path), "ok": dimensions == [1280, 720], "dimensions": dimensions, "mode": mode, "bytes": path.stat().st_size, "sha256": _sha(path)}


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_PATH), size=size)


def _fit_text(draw: Any, text: str, max_width: int, initial: int, minimum: int = 28) -> Any:
    size = initial
    while size > minimum:
        font = _font(size)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 2
    return _font(minimum)


def _build_sheet() -> None:
    from PIL import Image, ImageDraw

    width, height = 3840, 1500
    sheet = Image.new("RGB", (width, height), (18, 22, 29))
    draw = ImageDraw.Draw(sheet)
    title = "D354 현재 자세 q5 닫힘 결과 — 실제 Isaac, 동일 카메라"
    title_font = _fit_text(draw, title, width - 160, 66)
    draw.text((width // 2, 12), title, font=title_font, fill=(245, 248, 252), anchor="mt")
    for index, (role, label, q5) in enumerate(Q5_STATES):
        x0 = index * 1280
        with Image.open(CAPTURE_PATHS[role]) as source:
            image = source.convert("RGB")
        sheet.paste(image, (x0, 160))
        band = (38, 48, 63) if role != "first_overlap" else (91, 41, 38)
        draw.rectangle((x0, 105, x0 + 1279, 160), fill=band)
        label_font = _fit_text(draw, label, 1170, 48)
        draw.text((x0 + 640, 106), label, font=label_font, fill=(255, 255, 255), anchor="mt")
        q_text = f"q5 = {float(q5):.10f} rad"
        draw.rounded_rectangle((x0 + 35, 815, x0 + 540, 880), radius=14, fill=(0, 0, 0, 190))
        draw.text((x0 + 55, 823), q_text, font=_font(34), fill=(245, 245, 245))
        if index < 2:
            draw.line((x0 + 1279, 105, x0 + 1279, 880), fill=(210, 215, 225), width=4)

    draw.rectangle((45, 905, width - 45, 1450), outline=(105, 120, 145), width=3)
    rows = [
        (
            "실제 화면의 한계",
            "세 q5 상태는 실제 Isaac에서 저장됐지만 이 옆 카메라에서는 움직이는 죠가 원통 뒤에 가려졌습니다. 세 그림만 보고 접촉 여부를 판정할 수 없습니다.",
        ),
        (
            "D354 원 수치",
            "마지막 비접촉 점은 원통의 윗면과 옆면이 만나는 경계(rim)에 놓였고 다음 점만 옆면 안쪽이어서, ‘옆면을 먼저 접촉했다’고 확정할 수 없습니다.",
        ),
        (
            "아직 시험하지 않은 것",
            "PhysX 접촉력·마찰, 고정 죠와 움직이는 죠의 동시 접촉, 원통을 실제로 집고 버티거나 들어 올리는 동작은 이 그림에서 시험하지 않았습니다.",
        ),
    ]
    y = 945
    for heading, body in rows:
        draw.text((95, y), heading, font=_font(42), fill=(255, 203, 84))
        body_font = _fit_text(draw, body, width - 640, 36, 26)
        draw.text((520, y + 2), body, font=body_font, fill=(235, 239, 246))
        y += 145
    draw.text((width - 75, height - 8), "표시용 재현 — D354 수치 판정을 변경하지 않음", font=_font(28), fill=(165, 178, 198), anchor="rd")
    sheet.save(SHEET_PATH)


def _measurement_points() -> dict[str, Any]:
    classification = _json(D354_MEASUREMENT)["classification"]
    raw = classification["raw_first_contact_feature"]["endpoints"]
    live = classification["live_first_contact_feature"]["endpoints"]
    return {"raw": raw, "live": live}


def _build_actual_section() -> None:
    from PIL import Image, ImageDraw

    points = _measurement_points()["raw"]
    image = Image.new("RGB", (1800, 1200), (249, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((900, 45), "접촉 부근 단면 — 실제 비율", font=_font(58), fill=(25, 31, 41), anchor="ma")
    scale = 7.8  # pixels per millimetre
    cx, top = 670, 250
    radius_px = int(17.0 * scale)
    height_px = int(90.0 * scale)
    draw.rounded_rectangle((cx - radius_px, top, cx + radius_px, top + height_px), radius=radius_px, outline=(210, 129, 37), width=8, fill=(245, 186, 98))
    draw.line((cx - radius_px, top, cx + radius_px, top), fill=(122, 67, 14), width=8)
    draw.text((cx, top + height_px + 28), "원통: 지름 34 mm, 높이 90 mm", font=_font(35), fill=(45, 52, 64), anchor="ma")
    marker_x = cx - int(9.36 * scale)
    marker_y = top
    draw.ellipse((marker_x - 12, marker_y - 12, marker_x + 12, marker_y + 12), fill=(34, 102, 214), outline=(255, 255, 255), width=3)
    draw.line((marker_x, marker_y, 1080, 355), fill=(34, 102, 214), width=4)
    draw.text((1120, 305), "마지막 비접촉점", font=_font(40), fill=(34, 83, 165))
    draw.text((1120, 365), "z = +45.000000 mm (rim 경계)", font=_font(32), fill=(45, 52, 64))
    draw.line((marker_x, marker_y + 2, 1080, 535), fill=(198, 48, 43), width=4)
    draw.text((1120, 485), "첫 겹침점", font=_font(40), fill=(176, 35, 31))
    z_overlap = float(points["overlap"]["point_cylinder_local_m"][2]) * 1000.0
    draw.text((1120, 545), f"z = {z_overlap:+.9f} mm (barrel 내부)", font=_font(32), fill=(45, 52, 64))
    delta = (45.0 - z_overlap)
    draw.rounded_rectangle((980, 690, 1730, 965), radius=24, fill=(232, 237, 245), outline=(113, 128, 153), width=3)
    draw.text((1355, 735), f"두 점의 z 차이 = {delta:.9f} mm", font=_font(42), fill=(21, 28, 39), anchor="ma")
    draw.text((1355, 815), "실제 비율에서는 1픽셀보다 훨씬 작아", font=_font(35), fill=(62, 72, 89), anchor="ma")
    draw.text((1355, 865), "두 점이 사실상 같은 위치로 보이는 것이 정상입니다.", font=_font(35), fill=(62, 72, 89), anchor="ma")
    draw.text((80, 1140), "이 그림은 D354 원 JSON의 raw witness 좌표를 표시한 것이며 새 접촉 판정이 아닙니다.", font=_font(30), fill=(85, 96, 113))
    image.save(ACTUAL_SECTION_PATH)


def _build_magnified_section() -> None:
    from PIL import Image, ImageDraw

    points = _measurement_points()["raw"]
    clear_z = float(points["clear"]["point_cylinder_local_m"][2]) * 1000.0
    overlap_z = float(points["overlap"]["point_cylinder_local_m"][2]) * 1000.0
    delta = clear_z - overlap_z
    image = Image.new("RGB", (1800, 1200), (23, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1800, 145), fill=(154, 34, 31))
    draw.text((900, 34), "DISPLAY ONLY / 표시 전용 — Z축만 50,000배 확대", font=_font(54), fill=(255, 255, 255), anchor="ma")
    draw.text((900, 190), "원통 윗면–옆면 경계를 확대해서 본 개념도", font=_font(45), fill=(235, 239, 246), anchor="ma")
    x_left, x_right, y_rim = 310, 1490, 470
    draw.line((x_left, y_rim, x_right, y_rim), fill=(255, 190, 79), width=12)
    draw.line((x_left, y_rim, x_left, 1020), fill=(255, 190, 79), width=12)
    draw.text((1510, y_rim), "rim / 경계", font=_font(36), fill=(255, 209, 128), anchor="lm")
    clear = (780, y_rim)
    magnified_delta_px = int(round(delta * 7.8 * 50000.0))
    overlap = (1020, y_rim + magnified_delta_px)
    draw.ellipse((clear[0] - 22, clear[1] - 22, clear[0] + 22, clear[1] + 22), fill=(55, 143, 255))
    draw.ellipse((overlap[0] - 22, overlap[1] - 22, overlap[0] + 22, overlap[1] + 22), fill=(255, 77, 69))
    draw.line((clear[0], clear[1], 650, 330), fill=(55, 143, 255), width=5)
    draw.text((620, 280), f"마지막 비접촉\nz = {clear_z:+.9f} mm", font=_font(36), fill=(125, 190, 255), anchor="ma", align="center")
    draw.line((overlap[0], overlap[1], 1230, overlap[1] + 90), fill=(255, 77, 69), width=5)
    draw.text((1370, overlap[1] + 90), f"첫 겹침\nz = {overlap_z:+.9f} mm", font=_font(36), fill=(255, 132, 126), anchor="ma", align="center")
    draw.line((900, y_rim + 8, 900, overlap[1] - 10), fill=(220, 226, 237), width=4)
    draw.polygon([(900, overlap[1]), (884, overlap[1] - 30), (916, overlap[1] - 30)], fill=(220, 226, 237))
    draw.text((900, y_rim + magnified_delta_px // 2), f"실제 z 차이\n{delta:.9f} mm", font=_font(36), fill=(240, 243, 248), anchor="mm", align="center")
    draw.rounded_rectangle((140, 1035, 1660, 1155), radius=24, fill=(49, 58, 73), outline=(199, 64, 58), width=4)
    draw.text((900, 1070), "확대 모양은 사람이 차이를 보기 위한 것뿐이며 거리·접촉 판정에는 사용하지 않습니다.", font=_font(32), fill=(255, 242, 239), anchor="ma")
    image.save(MAGNIFIED_SECTION_PATH)


def _build_occlusion_addendum() -> None:
    """Explain the observed Isaac side-camera occlusion without new simulation."""
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (2400, 1500), (18, 22, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1200, 34),
        "왜 세 Isaac 화면이 거의 똑같아 보였나",
        font=_font(58),
        fill=(248, 250, 253),
        anchor="ma",
    )
    draw.text(
        (1200, 100),
        "카메라 가림(occlusion)을 결과와 분리해서 표시한 보충 자료",
        font=_font(34),
        fill=(178, 192, 214),
        anchor="ma",
    )

    left_box = (55, 165, 1165, 825)
    right_box = (1235, 165, 2345, 825)
    for box in (left_box, right_box):
        draw.rounded_rectangle(box, radius=22, fill=(31, 38, 49), outline=(105, 121, 148), width=4)

    draw.text((610, 188), "실제 Isaac 캡처 — FIRST OVERLAP 상태", font=_font(34), fill=(255, 255, 255), anchor="ma")
    with Image.open(CAPTURE_PATHS["first_overlap"]) as source:
        actual = ImageOps.fit(source.convert("RGB"), (1060, 596), method=Image.Resampling.LANCZOS)
    canvas.paste(actual, (80, 230))
    draw.rounded_rectangle((94, 248, 714, 337), radius=14, fill=(0, 0, 0), outline=(255, 99, 91), width=3)
    draw.text((118, 263), "움직이는 죠가 원통 뒤에 가려져", font=_font(29), fill=(255, 236, 233))
    draw.text((118, 300), "이 화면만으로 접촉을 볼 수 없음", font=_font(29), fill=(255, 236, 233))
    draw.line((650, 337, 650, 535), fill=(255, 80, 72), width=8)
    draw.polygon([(650, 568), (628, 529), (672, 529)], fill=(255, 80, 72))

    draw.text((1790, 188), "Rerun 재생 — 팔·죠·원통의 전체 배치", font=_font(34), fill=(255, 255, 255), anchor="ma")
    with Image.open(RERUN_PNG_PATH) as source:
        rerun_crop = source.convert("RGB").crop((1850, 50, 3650, 1900))
        rerun_panel = ImageOps.fit(rerun_crop, (1060, 596), method=Image.Resampling.LANCZOS)
    canvas.paste(rerun_panel, (1260, 230))
    draw.rounded_rectangle((1280, 730, 2300, 810), radius=14, fill=(0, 0, 0), outline=(91, 170, 255), width=3)
    draw.text((1790, 750), "배치 확인용 재생이며 접촉력·마찰 증거는 아님", font=_font(28), fill=(224, 239, 255), anchor="ma")

    boxes = [
        (
            (55, 875, 785, 1285),
            "화면에서 확인된 것",
            (75, 196, 132),
            "• Isaac Sim이 실제로 실행됨\n• 원통 위치까지 팔이 배치됨\n• 세 q5 표시 상태가 각각 저장됨\n• 새 physics step은 0회",
        ),
        (
            (835, 875, 1565, 1285),
            "화면에서 확인 못 한 것",
            (255, 188, 72),
            "• 죠와 원통 경계의 직접 시야\n• 어느 표면이 먼저 닿았는지\n• 양쪽 죠의 동시 접촉\n• 접촉력과 마찰",
        ),
        (
            (1615, 875, 2345, 1285),
            "아직 실행하지 않은 시험",
            (255, 103, 95),
            "• PhysX로 죠를 실제로 닫기\n• 원통이 밀리거나 회전하는지\n• 집은 뒤 버티기·들어 올리기\n• grasp 성공/실패 판정",
        ),
    ]
    for box, heading, color, body in boxes:
        draw.rounded_rectangle(box, radius=22, fill=(34, 42, 55), outline=color, width=4)
        draw.text((box[0] + 32, box[1] + 30), heading, font=_font(36), fill=color)
        draw.multiline_text((box[0] + 36, box[1] + 103), body, font=_font(29), fill=(236, 240, 246), spacing=22)

    draw.text(
        (1200, 1340),
        "D354가 멈춘 이유: 마지막 비접촉점은 z=+45.000000 mm의 rim 경계, 다음 겹침점은 z=+44.999618602 mm",
        font=_font(31),
        fill=(232, 236, 243),
        anchor="ma",
    )
    draw.text(
        (1200, 1400),
        "두 점 차이는 0.000381398 mm이므로 barrel-first를 확정하지 못함 — 새 과학 판정이 아니라 D354 원 JSON 설명",
        font=_font(29),
        fill=(181, 194, 214),
        anchor="ma",
    )
    canvas.save(OCCLUSION_ADDENDUM_PATH)


def _rerun_artifact() -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    os.environ["PATH"] = f"{RERUN_CLI.parent}:{os.environ.get('PATH', '')}"
    measurement = _json(D354_MEASUREMENT)
    points = _measurement_points()["raw"]
    joint_names = list(d351.d332.ALL_JOINT_NAMES)
    trace = []
    distances = [math.nan, 0.0010050812803802547, -0.000988475720559677]
    for step, (role, _, q5) in enumerate(Q5_STATES):
        q = d351.Q_FROZEN_F32.copy()
        q[5] = q5
        by_name = {name: float(q[index]) for index, name in enumerate(joint_names)}
        trace.append({"step": step, "phase": role, "actual_joint_rad_by_name": by_name, "commanded_joint_rad_by_name": by_name, "q5_rad": float(q5), "raw_distance_mm": distances[step]})
    cyl_vertices, cyl_triangles = d351._cylinder_mesh()
    dynamic_points = []
    for step, role in ((1, "clear"), (2, "overlap")):
        dynamic_points.append(
            {
                "entity_path": "geometry/decision/raw_cylinder_witness",
                "coordinate_frame": "world",
                "positions_m": [points[role]["point_world_m"]],
                "radii": [0.003],
                "colors": [[55, 143, 255] if role == "clear" else [255, 77, 69]],
                "sequence": {"step": step},
            }
        )
    scalars = []
    events = []
    for step, (role, label, q5) in enumerate(Q5_STATES):
        scalars.append({"entity_path": "metrics/d357/q5_rad", "value": float(q5), "sequence": {"step": step}})
        if step > 0:
            scalars.append({"entity_path": "metrics/d357/raw_distance_mm", "value": distances[step], "sequence": {"step": step}})
        events.append(
            {
                "entity_path": "events/d357/display_pose",
                "text": (
                    f"{role.upper()}; display-only replay; "
                    "no physics/contact-force/grasp test"
                ),
                "level": "INFO",
                "sequence": {"step": step},
            }
        )
    events.append(
        {
            "entity_path": "events/d357/display_pose",
            "text": (
                "EXTERNAL STOP AFTER THREE CAPTURES; no 60-second hold completion, "
                "final sentinel, or setting-restoration attestation"
            ),
            "level": "WARN",
            "sequence": {"step": 2},
        }
    )
    status = log_rerun(
        RRD_PATH,
        urdf_path=d351.d333.DEFAULT_URDF,
        joint_trace=trace,
        coordinate_frames=[
            {
                "entity_path": "frames/world",
                "frame": "world",
                "parent_frame": "tf#/",
                "translation_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            {
                "entity_path": "frames/actual_world_root",
                "frame": "actual/world",
                "parent_frame": "tf#/",
                "translation_m": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
        ],
        meshes=[{"entity_path": "geometry/target/cylinder", "coordinate_frame": "world", "vertices_m": cyl_vertices + d351.OBJECT_POS_F32.astype(np.float64), "triangles": cyl_triangles, "color_rgba": [245, 172, 52, 170], "static": True}],
        points=dynamic_points,
        scalar_trace=scalars,
        events=events,
        recording_metadata={
            "case": CASE,
            "purpose": "beginner-readable replay companion for immutable D354 result",
            "scientific_verdict_inherited": measurement["classification"]["scientific_verdict"],
            "physics": "not executed; controlled steps 0",
            "display_only": True,
            "main_user_facing_artifact": _rel(SHEET_PATH),
            "isaac_run_completion": False,
            "external_termination": "after three captures during first-overlap hold",
            "camera_limitation": "moving jaw occluded behind cylinder in the three Isaac side views",
        },
        recording_id="g0a_d357_d354_result_visualization",
        blueprint_path=RBL_PATH,
        blueprint_mode="d357_beginner_result",
        live_viewer=False,
        app_id="roarm_g0a_d357_d354_result_visualization",
    )
    if not status.get("ok"):
        return {"log_status": status, "strict_validation": {"pass": False, "errors": ["log_rerun failed"]}, "pass": False}
    observed = status["archive_validation"]["entity_path_contract"]["observed_non_system"]
    observed_sha = hashlib.sha256(
        json.dumps(observed, separators=(",", ":")).encode()
    ).hexdigest()
    inventory_precheck = {
        "entity_count_exact": len(observed) == EXPECTED_RERUN_NON_SYSTEM_ENTITY_COUNT,
        "entity_paths_sha256_exact": observed_sha
        == EXPECTED_RERUN_NON_SYSTEM_ENTITY_SHA256,
        "timelines_exact": status["archive_validation"]["timeline_contract"]["observed"]
        == EXPECTED_RERUN_TIMELINES,
    }
    if not all(inventory_precheck.values()):
        return {
            "log_status": status,
            "exact_inventory_precheck": inventory_precheck,
            "observed_non_system_entity_count": len(observed),
            "observed_non_system_entity_sha256": observed_sha,
            "strict_validation": {
                "pass": False,
                "errors": ["fixed D357 Rerun inventory drift"],
            },
            "pass": False,
        }
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=["geometry/target/cylinder", "geometry/decision/raw_cylinder_witness", "metrics/d357/q5_rad", "events/d357/display_pose"],
        expected_timeline_names=["step"],
        exact_entity_paths=observed,
        exact_timeline_names=EXPECTED_RERUN_TIMELINES,
        expected_entity_components={
            "metadata/run": ["TextDocument:text"],
            "frames/world": ["Transform3D:child_frame", "Transform3D:parent_frame", "Transform3D:quaternion", "Transform3D:translation"],
            "geometry/target/cylinder": ["CoordinateFrame:frame", "Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"],
            "geometry/decision/raw_cylinder_witness": ["CoordinateFrame:frame", "Points3D:colors", "Points3D:positions", "Points3D:radii"],
            "metrics/d357/q5_rad": ["Scalars:scalars"],
            "events/d357/display_pose": ["TextLog:level", "TextLog:text"],
        },
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="2400x1400",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
    )
    return {
        "log_status": status,
        "exact_inventory_precheck": inventory_precheck,
        "strict_validation": validation,
        "exact_non_system_entities": observed,
        "exact_non_system_entity_sha256": observed_sha,
        "pass": validation.get("pass") is True,
    }


def _postprocess(supervisor: dict[str, Any]) -> dict[str, Any]:
    capture_reports = {role: _png_report(path) for role, path in CAPTURE_PATHS.items()}
    if not all(row["ok"] for row in capture_reports.values()):
        raise RuntimeError(f"D357 captured PNG validation failed: {capture_reports}")
    _build_sheet()
    _build_actual_section()
    _build_magnified_section()
    composite_reports = {path.name: _png_report(path) for path in (SHEET_PATH, ACTUAL_SECTION_PATH, MAGNIFIED_SECTION_PATH)}
    # Composite dimensions intentionally differ from the Isaac capture size.
    for path in (SHEET_PATH, ACTUAL_SECTION_PATH, MAGNIFIED_SECTION_PATH):
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            composite_reports[path.name].update({"ok": image.width > 0 and image.height > 0, "dimensions": list(image.size), "sha256": _sha(path), "bytes": path.stat().st_size})
    rerun = _rerun_artifact()
    _write_json_x(RERUN_VALIDATION_PATH, rerun)
    worker = _json(WORKER_SUMMARY_PATH)
    checks = {
        "supervisor_worker_success": supervisor.get("worker_exit_code") == 0 and not supervisor.get("watchdog_triggered"),
        "worker_contract_pass": worker.get("pass") is True,
        "three_actual_isaac_pngs": len(capture_reports) == 3 and all(row["ok"] for row in capture_reports.values()),
        "three_beginner_visuals": len(composite_reports) == 3 and all(row["ok"] for row in composite_reports.values()),
        "rerun_rrd_rbl_screenshot_pass": rerun.get("pass") is True,
        "controlled_physics_steps_zero": worker.get("counters", {}).get("controlled_physics_steps") == 0,
        "science_queries_zero": all(worker.get("counters", {}).get(key) == 0 for key in ("q5_science_evaluator_invocations", "distance_queries", "contact_or_overlap_queries", "new_classifications")),
        "sidecar_unchanged": _sidecar_hashes() == _json(PREPARE_PATH)["d334_sidecar_before"],
    }
    report = {
        "artifact": "D357_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "operational_verdict": "D357_AUTOMATED_VISUALIZATION_COMPLETE_PENDING_MANUAL" if all(checks.values()) else "D357_VISUALIZATION_CONTRACT_FAIL_STOP",
        "scientific_verdict_inherited": "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
        "scientific_verdict_changed": False,
        "captures": capture_reports,
        "beginner_visuals": composite_reports,
        "rerun_validation_path": _rel(RERUN_VALIDATION_PATH),
        "rerun_pass": rerun.get("pass"),
        "counters": worker.get("counters"),
        "target_ik_path_changed": False,
        "asset_or_physics_configuration_changed": False,
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(AUTOMATED_PATH, report)
    return report


def _run(args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D357 preregistration did not pass")
    if INVOCATION_PATH.exists() or WORKER_LOG_PATH.exists():
        raise RuntimeError("D357 invocation already consumed; no retry permitted")
    resource_now = _gpu_snapshot()
    if int(resource_now.get("memory_free_mib", 0)) < 8192 or int(
        resource_now.get("ram_available_bytes", 0)
    ) < 12 * 1024**3:
        print(
            json.dumps(
                {
                    "stage": "run_resource_gate_before_invocation",
                    "pass": False,
                    "invocation_marker_written": False,
                    "resources": resource_now,
                },
                ensure_ascii=False,
            )
        )
        return 2
    worker_token = secrets.token_hex(32)
    invocation = {
        "artifact": "D357_SINGLE_ISAAC_INVOCATION_MARKER_V1",
        "utc": _utc_now(),
        "run_nonce": prereg["run_nonce"],
        "invocation_index": 1,
        "supervisor_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(worker_token.encode("utf-8")).hexdigest(),
        "automatic_retry": False,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    command = [REGISTERED_PYTHON, str(HARNESS), "--stage", "_worker", "--out_dir", str(OUT_DIR), "--seed", str(SEED), "--viewer_hold_seconds", str(VIEWER_HOLD_SECONDS)]
    env = os.environ.copy()
    env.update(
        {
            "DISPLAY": ":1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
            WORKER_TOKEN_ENV: worker_token,
            SUPERVISOR_PID_ENV: str(os.getpid()),
            "PATH": f"{RERUN_CLI.parent}:{env.get('PATH', '')}",
        }
    )
    start = time.monotonic()
    watchdog_triggered = False
    watchdog_reason: str | None = None
    telemetry: list[dict[str, Any]] = []
    with WORKER_LOG_PATH.open("xb") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_progress = time.monotonic()
        last_sizes = (-1, -1)
        while process.poll() is None:
            sizes = (WORKER_LOG_PATH.stat().st_size if WORKER_LOG_PATH.exists() else 0, PHASE_PATH.stat().st_size if PHASE_PATH.exists() else 0)
            if sizes != last_sizes:
                last_progress = time.monotonic()
                last_sizes = sizes
            elapsed = time.monotonic() - start
            idle = time.monotonic() - last_progress
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
    supervisor = {
        "artifact": "D357_SUPERVISOR_SUMMARY_V1",
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
        "worker_exception_exists": WORKER_EXCEPTION_PATH.is_file(),
        "pass": exit_code == 0 and not watchdog_triggered and WORKER_SUMMARY_PATH.is_file() and not WORKER_EXCEPTION_PATH.exists(),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    if not supervisor["pass"]:
        print(json.dumps({"stage": "run", "pass": False, "supervisor": _rel(SUPERVISOR_PATH)}, ensure_ascii=False))
        return 2
    try:
        automated = _postprocess(supervisor)
    except Exception as error:
        _write_json_x(
            POSTPROCESS_EXCEPTION_PATH,
            {
                "artifact": "D357_POSTPROCESS_EXCEPTION_STOP_V1",
                "utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "isaac_worker_had_succeeded": True,
                "automatic_retry": False,
                "scientific_verdict_changed": False,
                "g0a_pass": False,
            },
        )
        print(json.dumps({"stage": "postprocess", "pass": False, "exception": _rel(POSTPROCESS_EXCEPTION_PATH)}, ensure_ascii=False))
        return 2
    print(json.dumps({"stage": "run", "pass": automated["pass"], "verdict": automated["operational_verdict"]}, ensure_ascii=False))
    return 0 if automated["pass"] else 2


def _addendum(args: argparse.Namespace) -> int:
    if OCCLUSION_ADDENDUM_PATH.exists() or OCCLUSION_ADDENDUM_SUMMARY_PATH.exists():
        raise RuntimeError("D357 occlusion addendum already exists; no overwrite")
    automated = _json(AUTOMATED_PATH)
    checks_before = {
        "automated_visualization_pass": automated.get("pass") is True,
        "one_isaac_invocation_only": _json(INVOCATION_PATH).get("invocation_index") == 1,
        "three_original_captures_exist": all(path.is_file() for path in CAPTURE_PATHS.values()),
        "original_rerun_screenshot_exists": RERUN_PNG_PATH.is_file(),
        "d354_frozen_hashes_exact": all(_sha(path) == expected for path, expected in D354_HASHES.items()),
    }
    if not all(checks_before.values()):
        raise RuntimeError(f"D357 occlusion addendum preflight failed: {checks_before}")
    _build_occlusion_addendum()
    from PIL import Image

    with Image.open(OCCLUSION_ADDENDUM_PATH) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    checks = {
        **checks_before,
        "addendum_decodes_2400x1500": dimensions == [2400, 1500] and mode == "RGB",
        "no_new_isaac_invocation": _json(INVOCATION_PATH).get("invocation_index") == 1,
        "scientific_verdict_unchanged": True,
    }
    report = {
        "artifact": "D357_ISAAC_CAMERA_OCCLUSION_ADDENDUM_SUMMARY_V1",
        "case": CASE,
        "reason": "manual original-resolution inspection found the moving jaw occluded behind the cylinder in all three same-camera Isaac captures",
        "artifact_path": _rel(OCCLUSION_ADDENDUM_PATH),
        "artifact_sha256": _sha(OCCLUSION_ADDENDUM_PATH),
        "artifact_dimensions": dimensions,
        "new_isaac_invocations": 0,
        "q5_science_queries": 0,
        "physics_steps": 0,
        "new_contact_or_cap_rim_classifications": 0,
        "scientific_verdict_inherited": "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
        "scientific_verdict_changed": False,
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(OCCLUSION_ADDENDUM_SUMMARY_PATH, report)
    print(json.dumps({"stage": "addendum", "pass": report["pass"], "artifact": report["artifact_path"]}, ensure_ascii=False))
    return 0 if report["pass"] else 2


def _finalize(args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D357 completion already exists")
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    checks = {
        "automated_pass": automated.get("pass") is True,
        "manual_visual_inspection_pass": manual.get("pass") is True,
        "manual_inspected_all_eight_pngs": set(manual.get("inspected_paths", [])) == {_rel(path) for path in (*CAPTURE_PATHS.values(), SHEET_PATH, ACTUAL_SECTION_PATH, MAGNIFIED_SECTION_PATH, RERUN_PNG_PATH, OCCLUSION_ADDENDUM_PATH)},
        "camera_occlusion_addendum_pass": _json(OCCLUSION_ADDENDUM_SUMMARY_PATH).get("pass") is True,
        "frozen_science_unchanged": automated.get("scientific_verdict_inherited") == "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP" and automated.get("scientific_verdict_changed") is False,
        "controlled_physics_steps_zero": automated.get("counters", {}).get("controlled_physics_steps") == 0,
        "d334_sidecar_unchanged": _sidecar_hashes() == _json(PREPARE_PATH)["d334_sidecar_before"],
    }
    completion = {
        "artifact": "D357_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "final_verdict": "D357_D354_BEGINNER_VISUALIZATION_REPAIR_COMPLETE" if all(checks.values()) else "D357_FINALIZATION_FAIL_STOP",
        "scientific_verdict_inherited": "D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP",
        "scientific_verdict_changed": False,
        "controlled_physics_steps": 0,
        "target_ik_path_changed": False,
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(json.dumps({"stage": "finalize", "pass": completion["pass"], "verdict": completion["final_verdict"]}, ensure_ascii=False))
    return 0 if completion["pass"] else 2


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "_worker", "addendum", "finalize"),
        required=True,
    )
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--viewer_hold_seconds", type=float, default=VIEWER_HOLD_SECONDS)
    return parser


def main() -> int:
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument(
        "--stage",
        choices=("prepare", "run", "_worker", "addendum", "finalize"),
        required=True,
    )
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D357 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D357 seed drift")
    if float(args.viewer_hold_seconds) != VIEWER_HOLD_SECONDS:
        raise RuntimeError("D357 Viewer hold drift")
    if args.stage == "prepare":
        return _prepare(args)
    if args.stage == "run":
        return _run(args)
    if args.stage == "addendum":
        return _addendum(args)
    if args.stage == "finalize":
        return _finalize(args)
    args.headless = False
    args.livestream = 0
    args.device = "cuda:0"
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    if hasattr(args, "xr"):
        args.xr = False
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
