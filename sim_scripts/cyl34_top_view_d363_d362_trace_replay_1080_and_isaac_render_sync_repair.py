#!/usr/bin/env python3
"""D363: exact-1080 D362 trace replay and zero-step Fabric render-sync repair.

This forward-only observability case reads the immutable D362 trace.  It never
recomputes q5 science or advances controlled physics.  Its only runtime state
operation is a direct display-state write followed by exactly one explicit
SimulationContext.forward() at each of four registered trace rows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import locale
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

# Safe before AppLauncher: this import graph does not import pxr/omni/Isaac runtime.
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d362_current_pose_capacity_prefix_integrated_physx_contact_motion as d362,
)


CASE = "g0a_d363"
CASE_NAME = "d362_trace_replay_1080_and_isaac_render_sync_repair"
NEW_VARIABLES = [
    "exact_1920x1080_trace_replay_encoding",
    "zero_step_explicit_fabric_forward_capture_sync",
]
BASE_GIT = "f085463d2e994a633cd1bcefe0c98c0b6c19e18e"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
RERUN_VERSION = "0.34.1"
SEED = 33201
DISPLAY = ":1"
VIEWPORT_SIZE = [1280, 720]
VIDEO_SIZE = [1920, 1080]
VIDEO_FPS = 20
VIDEO_STRIDE = 2
VIDEO_FRAME_COUNT = 250
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
MIN_GPU_FREE_MIB = 8192
MIN_RAM_AVAILABLE_BYTES = 8 * 1024**3
TOTAL_WATCHDOG_S = 600.0
INACTIVITY_WATCHDOG_S = 240.0
PLAY_SIMULATIONS_SETTING = d362.PLAY_SIMULATIONS_SETTING
CAMERA_EYE = list(d362.CAMERA_EYE)
OPPOSITE_CAMERA_EYE = list(d362.OPPOSITE_CAMERA_EYE)
CAMERA_TARGET = list(d362.CAMERA_TARGET)
DECISION_ROWS = {
    "precommand": 199,
    "contact_confirmation": 232,
    "motion_confirmation": 242,
    "final": 499,
}
EXPECTED_GLOBAL_STEPS = {
    "precommand": 200,
    "contact_confirmation": 233,
    "motion_confirmation": 243,
    "final": 500,
}
MASK_HSV_LOW = np.asarray([15, 40, 80], dtype=np.uint8)
MASK_HSV_HIGH = np.asarray([45, 255, 255], dtype=np.uint8)
MASK_MIN_AREA_PX = 500
MATERIAL_CENTROID_DELTA_PX = 15.0
MATERIAL_AXIS_DELTA_DEG = 15.0
MATERIAL_IOU_MAX = 0.85
MATERIAL_MIN_CRITERIA = 2
UPRIGHT_HW_MIN = 1.5
TOPPLED_WH_MIN = 1.15

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d363"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260718_grasp_g0a_d363_trace_replay_1080_and_isaac_render_sync_repair.md"
D362_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362"
D362_TRACE = D362_DIR / "d362_physics_trace.json"
D362_SUPERVISOR = D362_DIR / "d362_supervisor_summary.json"
D362_WORKER_SUMMARY = D362_DIR / "d362_worker_summary.json"
D362_VIDEO_REPORT = D362_DIR / "d362_interface_visible_trace_replay_report.json"
D362_STALE_FINAL = D362_DIR / "d362_final_actual_physx_interface.png"
D362_HARNESS = Path(d362.__file__).resolve()
D348_EVIDENCE = Path(d362.D348_EVIDENCE).resolve()
D362_TRACE_SHA = "9483146c4941e6518614c63acbf221128a564bafa7a9928d41e633ee6e4e2044"
D362_HARNESS_SHA = "80fb5f47ec01de67c23b11f92fc6b46f3bff7063fc9474436a7863cf1c9df11c"
D362_SUPERVISOR_SHA = "bac1d037f9e0fdf9eb7efe586fc281c18508df09cd59a11c9f90995de4e14385"
D362_WORKER_SUMMARY_SHA = "10f7bd39b67f9bd254827fab580396c9a8089304f904c20dc3efd908296b217d"
D362_VIDEO_REPORT_SHA = "50e7d2fd0d03b0ece1eb8568c55ecdeab4a3305551fa56f86cba4613af57f89c"
D348_EVIDENCE_SHA = "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6"
D362_EXPECTED_FILE_COUNT = 33
D362_FILENAME_SIZE_MANIFEST_SHA = "4b14fb9bde888f5ad63f215477fc298efc300b5f63ebcac7f710b48798ec36d8"
D362_FILE_SHA_MANIFEST_SHA = "33a147c7fa2c02b90a4d972a158aba3cfbbffe0b19814535d267336a92f057be"
D362_MANIFEST_COLLATION = "en_US.UTF-8"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

PREREG_PATH = OUT_DIR / "d363_preregistration.json"
PREPARE_PATH = OUT_DIR / "d363_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d363_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d363_worker_preflight.json"
RUNTIME_PATH = OUT_DIR / "d363_runtime_prerequisites.json"
WORKER_PHASE_PATH = OUT_DIR / "d363_worker_phase_markers.jsonl"
SUPERVISOR_PHASE_PATH = OUT_DIR / "d363_supervisor_phase_markers.jsonl"
WORKER_LOG_PATH = OUT_DIR / "d363_worker_stdout_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d363_worker_summary.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d363_worker_exception.json"
POSTPROCESS_EXCEPTION_PATH = OUT_DIR / "d363_supervisor_postprocess_exception.json"
VIDEO_PATH = OUT_DIR / "d363_d362_trace_replay_exact_1920x1080.mp4"
VIDEO_STORYBOARD_PATH = OUT_DIR / "d363_d362_trace_replay_storyboard_1920x1080.png"
VIDEO_REPORT_PATH = OUT_DIR / "d363_d362_trace_replay_report.json"
SYNC_REPORT_PATH = OUT_DIR / "d363_fabric_render_sync_report.json"
PRIMARY_STORYBOARD_PATH = OUT_DIR / "d363_fabric_sync_primary_storyboard_ko.png"
OPPOSITE_STORYBOARD_PATH = OUT_DIR / "d363_fabric_sync_opposite_storyboard_ko.png"
BEGINNER_SHEET_PATH = OUT_DIR / "d363_beginner_result_sheet_ko.png"
RRD_PATH = OUT_DIR / "d363_fabric_render_sync.rrd"
RBL_PATH = OUT_DIR / "d363_fabric_render_sync.rbl"
RERUN_PNG_PATH = OUT_DIR / "d363_fabric_render_sync_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d363_rerun_validation.json"
SUPERVISOR_PATH = OUT_DIR / "d363_supervisor_summary.json"
AUTOMATED_PATH = OUT_DIR / "d363_automated_summary.json"
MANUAL_PATH = OUT_DIR / "d363_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d363_completion_summary.json"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

WORKER_TOKEN_ENV = "D363_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D363_SUPERVISOR_PID"

CAPTURE_PATHS: dict[str, dict[str, dict[str, Path]]] = {
    role: {
        timing: {
            view: OUT_DIR / f"d363_{role}_{timing}_forward_{view}.png"
            for view in ("primary", "opposite")
        }
        for timing in ("before", "after")
    }
    for role in DECISION_ROWS
}

_WORKER_SEQUENCE = 0
_SUPERVISOR_SEQUENCE = 0
_DISPLAY_STATE_WRITE_COUNT = 0
_EXPLICIT_FORWARD_COUNT = 0
_CONTROLLED_PHYSICS_STEPS = 0
_Q5_SCIENCE_SAMPLE_COUNT = 0
_Q5_TARGET_UPDATE_COUNT = 0
_CONTACT_QUERY_COUNT = 0


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
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if _json(path) != payload:
        raise RuntimeError(f"durable JSON reread mismatch: {path}")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _marker(owner: str, phase: str, event: str, details: dict[str, Any] | None = None) -> None:
    global _WORKER_SEQUENCE, _SUPERVISOR_SEQUENCE
    if owner == "worker":
        _WORKER_SEQUENCE += 1
        sequence = _WORKER_SEQUENCE
        path = WORKER_PHASE_PATH
    else:
        _SUPERVISOR_SEQUENCE += 1
        sequence = _SUPERVISOR_SEQUENCE
        path = SUPERVISOR_PHASE_PATH
    _append_jsonl(
        path,
        {
            "sequence": sequence,
            "owner": owner,
            "utc": _utc_now(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "phase": phase,
            "event": event,
            "details": details or {},
        },
    )


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
        "claudedocs/session_20260718_grasp_g0a_d363_",
        "sim_scripts/cyl34_top_view_d363_",
        "claudedocs/runtime_logs/grasp_track/g0a_d363/",
    )
    return all(len(row) >= 4 and any(row[3:].startswith(prefix) for prefix in allowed) for row in rows)


def _sidecar_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in sorted(D334_SIDECAR.rglob("*")) if path.is_file()}


def _input_hashes() -> dict[str, str]:
    paths = [
        D362_TRACE,
        D362_SUPERVISOR,
        D362_WORKER_SUMMARY,
        D362_VIDEO_REPORT,
        D362_STALE_FINAL,
        D362_HARNESS,
        D348_EVIDENCE,
        Path(d362.d351.__file__).resolve(),
        Path(d362.d351.d332.__file__).resolve(),
        Path(d362.d351.d349.__file__).resolve(),
        d362.VARIANT_ROBOT_USD,
        d362.VARIANT_PHYSICS_USD,
        d362.URDF_PATH,
    ]
    return {_rel(path): _sha(path) for path in paths}


def _frozen_display_topology_parts() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load the exact D348 callback topology only for D362's offline renderer.

    This deliberately performs no live USD/PhysX inventory or geometry query.  The
    D362 trace remains the sole temporal/scientific authority; these frozen vertices
    and triangles are display-only input to the already-frozen D362 frame renderer.
    """
    evidence = _json(D348_EVIDENCE)
    source_rows = evidence.get("rows")
    if not isinstance(source_rows, list):
        source_rows = []
    parts: dict[str, list[dict[str, Any]]] = {"link5": [], "gripper_link": []}
    row_checks: list[bool] = []
    global_indices: list[int] = []
    topology_digest = hashlib.sha256()
    body_sequence = [row.get("body") for row in source_rows if isinstance(row, dict)]
    for row in source_rows:
        if not isinstance(row, dict):
            row_checks.append(False)
            global_indices.append(-1)
            continue
        body = row.get("body")
        if body not in parts:
            row_checks.append(False)
            global_indices.append(int(row.get("global_part_idx", -1)))
            continue
        instance = row.get("instance", {})
        prototype = row.get("prototype", {})
        raw_vertices = np.asarray(instance.get("vertices_m", []), dtype=np.float64)
        raw_triangles = np.asarray(instance.get("topology_triangles", []), dtype=np.int64)
        vertex_shape_valid = raw_vertices.ndim == 2 and raw_vertices.shape[1:] == (3,)
        triangle_shape_valid = raw_triangles.ndim == 2 and raw_triangles.shape[1:] == (3,)
        vertices = raw_vertices if vertex_shape_valid else np.empty((0, 3), dtype=np.float64)
        triangles = raw_triangles if triangle_shape_valid else np.empty((0, 3), dtype=np.int64)
        global_index = int(row.get("global_part_idx", -1))
        body_index = global_index if body == "link5" else global_index - 64
        valid = bool(
            row.get("pass") is True
            and row.get("name") == f"part_{body_index:03d}"
            and 0 <= body_index < 64
            and row.get("checks", {}).get("raw_instance_prototype_payload_exact") is True
            and row.get("checks", {}).get("both_topologies_closed_and_oriented") is True
            and instance.get("payload_sha256") == prototype.get("payload_sha256")
            and vertex_shape_valid
            and triangle_shape_valid
            and len(vertices) == int(instance.get("vertex_count", -1))
            and len(triangles) == int(instance.get("triangle_count", -1))
            and len(vertices) > 0
            and len(triangles) > 0
            and np.isfinite(vertices).all()
            and int(triangles.min()) >= 0
            and int(triangles.max()) < len(vertices)
        )
        row_checks.append(valid)
        global_indices.append(global_index)
        topology_digest.update(body.encode("utf-8") + b"\0")
        topology_digest.update(str(row.get("name")).encode("utf-8") + b"\0")
        topology_digest.update(vertices.astype("<f8", copy=False).tobytes())
        topology_digest.update(triangles.astype("<i8", copy=False).tobytes())
        parts[body].append(
            {
                "body": body,
                "name": row.get("name"),
                "_vertices": vertices,
                "_triangles": triangles,
            }
        )
    counts = {body: len(rows) for body, rows in parts.items()}
    checks = {
        "evidence_sha_exact": _sha(D348_EVIDENCE) == D348_EVIDENCE_SHA,
        "evidence_artifact_exact": evidence.get("artifact") == "D348_CALLBACK_TOPOLOGY_VOLUME_EVIDENCE_V1",
        "evidence_pass": evidence.get("pass") is True,
        "source_rows_exact_128": len(source_rows) == 128,
        "body_sequence_exact_64_plus_64": body_sequence
        == ["link5"] * 64 + ["gripper_link"] * 64,
        "global_indices_exact_0_to_127": global_indices == list(range(128)),
        "part_counts_64_plus_64": counts == {"link5": 64, "gripper_link": 64},
        "all_128_rows_valid": len(row_checks) == 128 and all(row_checks),
    }
    return parts, {
        "authority": "frozen D348 callback vertices/topology; D362 renderer display-only",
        "path": _rel(D348_EVIDENCE),
        "sha256": _sha(D348_EVIDENCE),
        "part_counts": counts,
        "display_topology_stream_sha256": topology_digest.hexdigest(),
        "fresh_live_query_performed": False,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _d362_manifest() -> dict[str, Any]:
    active_collation = locale.setlocale(locale.LC_COLLATE, D362_MANIFEST_COLLATION)
    names = sorted(
        (path.name for path in D362_DIR.iterdir() if path.is_file()),
        key=locale.strxfrm,
    )
    entries: list[dict[str, Any]] = []
    filename_size_lines: list[str] = []
    file_sha_lines: list[str] = []
    for name in names:
        path = D362_DIR / name
        stat = path.stat()
        digest = _sha(path)
        entries.append(
            {
                "name": name,
                "bytes": stat.st_size,
                "sha256": digest,
                "regular_file": path.is_file(),
                "symlink": path.is_symlink(),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "link_count": stat.st_nlink,
            }
        )
        filename_size_lines.append(f"{name}\t{stat.st_size}\n")
        file_sha_lines.append(f"{digest}  {name}\n")
    return {
        "path": _rel(D362_DIR),
        "file_count": len(entries),
        "entries": entries,
        "filename_size_manifest_sha256": hashlib.sha256("".join(filename_size_lines).encode()).hexdigest(),
        "file_sha_manifest_sha256": hashlib.sha256("".join(file_sha_lines).encode()).hexdigest(),
        "collation": active_collation,
        "all_regular_no_symlink": all(row["regular_file"] and not row["symlink"] for row in entries),
    }


def _d362_manifest_pass(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest["file_count"] == D362_EXPECTED_FILE_COUNT
        and manifest["filename_size_manifest_sha256"] == D362_FILENAME_SIZE_MANIFEST_SHA
        and manifest["file_sha_manifest_sha256"] == D362_FILE_SHA_MANIFEST_SHA
        and manifest["all_regular_no_symlink"]
    )


def _d363_d362_inode_disjoint() -> bool:
    d362_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in D362_DIR.iterdir()
        if path.is_file()
    }
    if not OUT_DIR.exists():
        return True
    return all(
        (path.stat().st_dev, path.stat().st_ino) not in d362_inodes
        for path in OUT_DIR.rglob("*")
        if path.is_file()
    )


def _gpu_snapshot() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    values: list[str] = []
    if query.returncode == 0 and query.stdout.strip():
        values = [item.strip() for item in query.stdout.strip().splitlines()[0].split(",")]
    vm = psutil.virtual_memory()
    return {
        "query_returncode": query.returncode,
        "raw": query.stdout.strip(),
        "name": values[0] if len(values) == 7 else None,
        "compute_capability": values[1] if len(values) == 7 else None,
        "memory_total_mib": int(values[2]) if len(values) == 7 else None,
        "memory_used_mib": int(values[3]) if len(values) == 7 else None,
        "memory_free_mib": int(values[4]) if len(values) == 7 else None,
        "utilization_gpu_percent": int(values[5]) if len(values) == 7 else None,
        "utilization_memory_percent": int(values[6]) if len(values) == 7 else None,
        "ram_available_bytes": int(vm.available),
    }


def _residual_process_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "ppid", "status", "cmdline", "create_time"]):
        try:
            command = " ".join(proc.info.get("cmdline") or [])
            if "cyl34_top_view_d342" in command or "isaac" in command.lower() or "kit" in command.lower():
                rows.append(
                    {
                        "pid": proc.info["pid"],
                        "ppid": proc.info["ppid"],
                        "status": proc.info["status"],
                        "command": command,
                        "create_time": proc.info["create_time"],
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "rows": sorted(rows, key=lambda row: row["pid"]),
        "known_d342_pid_1729639_observed": any(row["pid"] == 1729639 for row in rows),
        "signal_sent": False,
        "gate_role": "recorded input risk; resource gates remain authoritative",
    }


def _trace_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "trace_sha_exact": _sha(D362_TRACE) == D362_TRACE_SHA,
        "row_count_500": len(rows) == 500,
    }
    selected: dict[str, Any] = {}
    for role, index in DECISION_ROWS.items():
        row = rows[index]
        selected[role] = {
            "index": index,
            "global_step": row.get("global_step"),
            "phase": row.get("phase"),
            "phase_step": row.get("phase_step"),
            "q5_actual_rad": row.get("q5_actual_rad"),
            "object_pos_w_m": row.get("object_pos_w_m"),
            "object_quat_wxyz": row.get("object_quat_wxyz"),
            "object_disp_xy_mm": row.get("object_disp_xy_mm"),
            "object_tilt_delta_from_reference_deg": row.get("object_tilt_delta_from_reference_deg"),
        }
        checks[f"{role}_global_step_exact"] = row.get("global_step") == EXPECTED_GLOBAL_STEPS[role]
        checks[f"{role}_finite_state"] = bool(
            np.isfinite(
                np.asarray(
                    [
                        *row["actual_joint_rad"],
                        *row["actual_joint_vel_rad_s"],
                        *row["object_pos_w_m"],
                        *row["object_quat_wxyz"],
                        *row["object_lin_vel_w_mps"],
                        *row["object_ang_vel_w_radps"],
                    ],
                    dtype=np.float64,
                )
            ).all()
        )
    checks["contact_confirmation_mask_exact"] = bool(
        rows[232]["event_masks"]["two_step_any_robot_confirmation_end"]
    )
    checks["motion_confirmation_mask_exact"] = bool(
        rows[242]["event_masks"]["two_step_motion_confirmation_end"]
    )
    checks["precommand_upright_trace"] = float(rows[199]["object_tilt_delta_from_reference_deg"]) < 0.01
    checks["final_toppled_trace"] = float(rows[499]["object_tilt_delta_from_reference_deg"]) > 89.0
    checks["final_motion_trace"] = float(rows[499]["object_disp_xy_mm"]) > 60.0
    return {"selected": selected, "checks": checks, "pass": all(checks.values())}


def _negative_controls() -> dict[str, Any]:
    reference_indices = np.linspace(0, 499, VIDEO_FRAME_COUNT, dtype=np.int64).tolist()

    def video_gate(width: int, height: int, indices: list[int]) -> bool:
        return bool(
            [width, height] == VIDEO_SIZE
            and indices == reference_indices
        )

    def no_advance(
        step_delta: int,
        time_delta: float,
        forward_count: int,
        q5_updates: int,
    ) -> bool:
        return (
            step_delta == 0
            and time_delta == 0.0
            and forward_count == 4
            and q5_updates == 0
        )

    wrong_middle = list(reference_indices)
    wrong_middle[125], wrong_middle[126] = wrong_middle[126], wrong_middle[125]
    wrong_rows = dict(DECISION_ROWS)
    wrong_rows["contact_confirmation"] = 233
    mutated_manifest = {
        "file_count": D362_EXPECTED_FILE_COUNT,
        "filename_size_manifest_sha256": D362_FILENAME_SIZE_MANIFEST_SHA,
        "file_sha_manifest_sha256": "0" * 64,
        "all_regular_no_symlink": True,
    }

    tests = {
        "reference_video_accept": video_gate(1920, 1080, reference_indices),
        "reject_1088": not video_gate(1920, 1088, reference_indices),
        "reject_duplicate_index": not video_gate(1920, 1080, [*reference_indices[:-1], reference_indices[-2]]),
        "reject_missing_endpoint": not video_gate(1920, 1080, [*reference_indices[:-1], 498]),
        "reject_wrong_middle_permutation": not video_gate(1920, 1080, wrong_middle),
        "reject_wrong_decision_row_index": wrong_rows != DECISION_ROWS,
        "reference_no_advance_accept": no_advance(0, 0.0, 4, 0),
        "reject_step_increment": not no_advance(1, 0.0, 4, 0),
        "reject_clock_increment": not no_advance(0, 1.0 / 60.0, 4, 0),
        "reject_fifth_forward": not no_advance(0, 0.0, 5, 0),
        "reject_q5_target_update": not no_advance(0, 0.0, 4, 1),
        "reject_d362_tree_manifest_mutation": not _d362_manifest_pass(mutated_manifest),
    }
    return {"tests": tests, "passed": sum(tests.values()), "total": len(tests), "pass": all(tests.values())}


def _static_source_audit() -> dict[str, Any]:
    import ast

    source = HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_names = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    setting_set_first_args = [
        ast.unparse(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith(".set")
        and node.args
    ]
    checks = {
        "one_explicit_forward_call_site": call_names.count("inner.sim.forward") == 1,
        "no_sim_step_call": "inner.sim.step" not in call_names,
        "no_env_step_call": "inner.step" not in call_names,
        "no_d332_physics_step_call": not any(name.endswith("._physics_step") for name in call_names),
        "no_joint_target_write": not any(name.endswith(".set_joint_position_target") for name in call_names),
        "no_scene_write_data_to_sim": not any(name.endswith(".scene.write_data_to_sim") for name in call_names),
        "no_inner_sim_render": "inner.sim.render" not in call_names,
        "no_fresh_corrected_live_audit": not any(name.endswith("._corrected_live_audit") for name in call_names),
        "no_fresh_live_topology_binding": not any(name.endswith("._build_live_topology_parts") for name in call_names),
        "no_physics_setting_mutation_in_set_calls": all(
            argument in {"PLAY_SIMULATIONS_SETTING", "cv2.CAP_PROP_POS_FRAMES"}
            for argument in setting_set_first_args
        ),
        "macro_block_size_one_literal": "macro_block_size=1" in source,
        "decision_rows_exact_literal": all(f'"{role}": {index}' in source for role, index in DECISION_ROWS.items()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D363 output already exists; overwrite forbidden: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    rows = _json(D362_TRACE)
    trace_contract = _trace_contract(rows)
    manifest = _d362_manifest()
    gpu = _gpu_snapshot()
    residual = _residual_process_audit()
    negative = _negative_controls()
    static_audit = _static_source_audit()
    _, frozen_topology = _frozen_display_topology_parts()
    ffmpeg_check = subprocess.run(
        [sys.executable, "-c", "import imageio_ffmpeg as f; print(f.get_ffmpeg_version()); print(f.get_ffmpeg_exe())"],
        text=True,
        capture_output=True,
        check=False,
    )
    rerun_check = subprocess.run([str(RERUN_CLI), "--version"], text=True, capture_output=True, check=False)
    display_check = subprocess.run(["xdpyinfo", "-display", DISPLAY], text=True, capture_output=True, check=False)
    pins = {"numpy": np.__version__, "psutil": psutil.__version__}
    prereg = {
        "artifact": "D363_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "utc": _utc_now(),
        "new_variables": NEW_VARIABLES,
        "base_git": BASE_GIT,
        "run_nonce": secrets.token_hex(24),
        "harness_path": _rel(HARNESS),
        "harness_sha256": _sha(HARNESS),
        "session_doc_path": _rel(SESSION_DOC),
        "input_hashes": _input_hashes(),
        "d362_manifest_before": manifest,
        "d334_sidecar_before": _sidecar_hashes(),
        "decision_rows": DECISION_ROWS,
        "trace_contract": trace_contract,
        "frozen_display_topology": frozen_topology,
        "encoding_contract": {
            "size": VIDEO_SIZE,
            "fps": VIDEO_FPS,
            "frames": VIDEO_FRAME_COUNT,
            "codec": VIDEO_CODEC,
            "pixel_format": VIDEO_PIXEL_FORMAT,
            "macro_block_size": 1,
            "physics_recomputed": False,
        },
        "sync_contract": {
            "direct_state_writes": 4,
            "explicit_forward_calls": 4,
            "controlled_physics_steps": 0,
            "q5_science_samples": 0,
            "q5_target_updates": 0,
            "contact_queries": 0,
            "views_per_timing": ["primary", "opposite"],
            "timings": ["before", "after"],
            "capture_count": 16,
            "mandatory_after_correspondence": True,
            "mandatory_before_after_difference": False,
        },
        "silhouette_contract": {
            "hsv_low": MASK_HSV_LOW.tolist(),
            "hsv_high": MASK_HSV_HIGH.tolist(),
            "min_area_px": MASK_MIN_AREA_PX,
            "material_centroid_delta_px": MATERIAL_CENTROID_DELTA_PX,
            "material_axis_delta_deg": MATERIAL_AXIS_DELTA_DEG,
            "material_iou_max": MATERIAL_IOU_MAX,
            "material_min_criteria": MATERIAL_MIN_CRITERIA,
            "upright_height_width_min": UPRIGHT_HW_MIN,
            "toppled_width_height_min": TOPPLED_WH_MIN,
            "relative_trace_render_correspondence": {
                "views": ["primary", "opposite"],
                "direction_cosine_min_mandatory": 0.8,
                "axis_delta_error_max_deg": 15.0,
                "observed_displacement_min_px": 15.0,
                "minimum_true_criteria": 2,
                "swapped_role_negative_must_fail": True,
                "absolute_pixel_calibration_claimed": False,
            },
        },
        "watchdogs": {"total_s": TOTAL_WATCHDOG_S, "inactivity_s": INACTIVITY_WATCHDOG_S},
        "automatic_retry": False,
        "resume": False,
        "overwrite": False,
        "target_ik_path_changed": False,
        "physics_settings_changed": False,
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "pass": True,
    }
    _write_json_x(PREREG_PATH, prereg)
    checks = {
        "head_origin_master_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_d363_only": _status_scope_ok(_git_status()),
        "registered_python": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
        "harness_hash_bound": _sha(HARNESS) == prereg["harness_sha256"],
        "input_hashes_bound": _input_hashes() == prereg["input_hashes"],
        "d362_manifest_exact": _d362_manifest_pass(manifest),
        "d362_trace_contract": trace_contract["pass"],
        "d362_harness_sha_exact": _sha(D362_HARNESS) == D362_HARNESS_SHA,
        "d362_supervisor_sha_exact": _sha(D362_SUPERVISOR) == D362_SUPERVISOR_SHA,
        "d362_worker_summary_sha_exact": _sha(D362_WORKER_SUMMARY) == D362_WORKER_SUMMARY_SHA,
        "d362_video_report_sha_exact": _sha(D362_VIDEO_REPORT) == D362_VIDEO_REPORT_SHA,
        "d348_frozen_display_topology": frozen_topology["pass"],
        "d334_sidecar_bound": bool(prereg["d334_sidecar_before"]),
        "d363_d362_inode_disjoint": _d363_d362_inode_disjoint(),
        "numpy_pin_1p26p0": pins["numpy"] == "1.26.0",
        "psutil_pin_5p9p8": pins["psutil"] == "5.9.8",
        "display_xdpyinfo": display_check.returncode == 0,
        "gpu_rtx4090_sm89": gpu.get("name") == "NVIDIA GeForce RTX 4090 Laptop GPU"
        and gpu.get("compute_capability") == "8.9",
        "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
        "ram_free_gate": int(gpu.get("ram_available_bytes") or 0) >= MIN_RAM_AVAILABLE_BYTES,
        "ffmpeg_available": ffmpeg_check.returncode == 0 and "7.0.2" in ffmpeg_check.stdout,
        "rerun_pin": rerun_check.returncode == 0 and RERUN_VERSION in (rerun_check.stdout + rerun_check.stderr),
        "negative_controls": negative["pass"],
        "static_source_audit": static_audit["pass"],
        "no_runtime_modules_before_applauncher": not any(
            name in {"pxr", "omni", "isaaclab", "isaacsim", "carb"}
            or name.startswith(("pxr.", "omni.", "isaaclab.", "isaacsim.", "carb."))
            for name in sys.modules
        ),
    }
    prepare = {
        "artifact": "D363_PREPARE_PREFLIGHT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "preregistration_sha256": _sha(PREREG_PATH),
        "git_status": _git_status(),
        "pins": pins,
        "gpu_and_ram": gpu,
        "residual_process_audit": residual,
        "ffmpeg_check": {"returncode": ffmpeg_check.returncode, "stdout": ffmpeg_check.stdout, "stderr": ffmpeg_check.stderr},
        "rerun_check": {"returncode": rerun_check.returncode, "stdout": rerun_check.stdout, "stderr": rerun_check.stderr},
        "display_check_returncode": display_check.returncode,
        "negative_controls": negative,
        "static_source_audit": static_audit,
        "frozen_display_topology": frozen_topology,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREPARE_PATH, prepare)
    print(json.dumps({"stage": "prepare", "pass": prepare["pass"], "checks": checks}, ensure_ascii=False))
    return 0 if prepare["pass"] else 2


def _physical_snapshot(inner: Any, timeline: Any) -> dict[str, Any]:
    robot = inner._robot.data
    obj = inner._sponge.data
    return {
        "custom_step_counter": int(inner._sim_step_counter),
        "simulation_clock": d362.d351._simulation_clock(inner),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time_s": float(timeline.get_current_time()),
        "joint_pos_bits": robot.joint_pos[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "joint_vel_bits": robot.joint_vel[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_pos_bits": obj.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_quat_bits": obj.root_quat_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_lin_vel_bits": obj.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_ang_vel_bits": obj.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
    }


def _snapshot_no_advance(reference: dict[str, Any], observed: dict[str, Any]) -> bool:
    return bool(
        observed["custom_step_counter"] == reference["custom_step_counter"]
        and observed["simulation_clock"] == reference["simulation_clock"]
        and observed["timeline_time_s"] == reference["timeline_time_s"]
        and not observed["timeline_playing"]
        and not observed["timeline_stopped"]
        and all(observed[key] == reference[key] for key in reference if key.endswith("_bits"))
    )


def _pause_no_advance(inner: Any, timeline: Any) -> dict[str, Any]:
    before = _physical_snapshot(inner, timeline)
    commit_count = 0
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    if timeline.is_playing():
        timeline.pause()
    if timeline.is_playing() and not timeline.is_stopped():
        timeline.commit()
        commit_count = 1
    after = _physical_snapshot(inner, timeline)
    checks = {
        "paused_not_stopped": not after["timeline_playing"] and not after["timeline_stopped"],
        "commit_at_most_once": commit_count in (0, 1),
        "counter_unchanged": before["custom_step_counter"] == after["custom_step_counter"],
        "clock_unchanged": before["simulation_clock"] == after["simulation_clock"],
        "state_bits_unchanged": all(before[key] == after[key] for key in before if key.endswith("_bits")),
    }
    return {"before": before, "after": after, "commit_count": commit_count, "checks": checks, "pass": all(checks.values())}


def _direct_write_trace_state(inner: Any, row: dict[str, Any]) -> dict[str, Any]:
    global _DISPLAY_STATE_WRITE_COUNT
    import torch

    if list(inner._robot.joint_names) != list(d362.d351.d332.ALL_JOINT_NAMES):
        raise RuntimeError("D363 joint order drift")
    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    q = torch.tensor([row["actual_joint_rad"]], device=inner.device, dtype=torch.float32)
    qd = torch.tensor([row["actual_joint_vel_rad_s"]], device=inner.device, dtype=torch.float32)
    pose = torch.tensor(
        [[*row["object_pos_w_m"], *row["object_quat_wxyz"]]],
        device=inner.device,
        dtype=torch.float32,
    )
    velocity = torch.tensor(
        [[*row["object_lin_vel_w_mps"], *row["object_ang_vel_w_radps"]]],
        device=inner.device,
        dtype=torch.float32,
    )
    inner._robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
    inner._sponge.write_root_pose_to_sim(pose, env_ids=env_ids)
    inner._sponge.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    _DISPLAY_STATE_WRITE_COUNT += 1
    observed = {
        "joint_pos_bits": inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "joint_vel_bits": inner._robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_pos_bits": inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_quat_bits": inner._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_lin_vel_bits": inner._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
        "object_ang_vel_bits": inner._sponge.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float32).tobytes().hex(),
    }
    expected = {
        "joint_pos_bits": np.asarray(row["actual_joint_rad"], dtype=np.float32).tobytes().hex(),
        "joint_vel_bits": np.asarray(row["actual_joint_vel_rad_s"], dtype=np.float32).tobytes().hex(),
        "object_pos_bits": np.asarray(row["object_pos_w_m"], dtype=np.float32).tobytes().hex(),
        "object_quat_bits": np.asarray(row["object_quat_wxyz"], dtype=np.float32).tobytes().hex(),
        "object_lin_vel_bits": np.asarray(row["object_lin_vel_w_mps"], dtype=np.float32).tobytes().hex(),
        "object_ang_vel_bits": np.asarray(row["object_ang_vel_w_radps"], dtype=np.float32).tobytes().hex(),
    }
    checks = {key: observed[key] == expected[key] for key in expected}
    return {"expected_bits": expected, "observed_bits": observed, "checks": checks, "pass": all(checks.values())}


def _capture_viewport(
    path: Path,
    simulation_app: Any,
    inner: Any,
    timeline: Any,
    camera_eye: list[float],
    role: str,
) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utility

    if timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError(f"D363 capture timeline contract failed: {role}")
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    inner.sim.set_camera_view(camera_eye, CAMERA_TARGET)
    reference = _physical_snapshot(inner, timeline)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None or not hasattr(viewport, "set_texture_resolution"):
        raise RuntimeError("D363 active viewport unavailable")
    viewport.set_texture_resolution(tuple(VIEWPORT_SIZE))
    for _ in range(8):
        simulation_app.update()
        if not _snapshot_no_advance(reference, _physical_snapshot(inner, timeline)):
            raise RuntimeError(f"D363 app update advanced state before capture: {role}")
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(capture.wait_for_result(completion_frames=5), run_until_complete=False)
    deadline = time.monotonic() + 30.0
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()
        if not _snapshot_no_advance(reference, _physical_snapshot(inner, timeline)):
            raise RuntimeError(f"D363 capture advanced state: {role}")
    if not task.done():
        task.cancel()
        raise RuntimeError(f"D363 capture timeout: {role}")
    if not bool(task.result()):
        raise RuntimeError(f"D363 capture failed: {role}")
    for _ in range(2):
        simulation_app.update()
    after = _physical_snapshot(inner, timeline)
    if not _snapshot_no_advance(reference, after):
        raise RuntimeError(f"D363 post-capture state drift: {role}")
    _marker(
        "worker",
        "viewport_capture",
        "complete",
        {"capture_role": role, "path": _rel(path), "physics_unchanged": True},
    )
    return {
        "path": _rel(path),
        "camera_eye": camera_eye,
        "camera_target": CAMERA_TARGET,
        "guard_before": reference,
        "guard_after": after,
        "physics_unchanged": True,
    }


def _build_video(topology_parts: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import imageio_ffmpeg
    from PIL import Image

    global _CONTROLLED_PHYSICS_STEPS
    _marker("worker", "trace_replay_video", "start", {"rows": len(rows), "size": VIDEO_SIZE})
    indices = np.linspace(0, len(rows) - 1, VIDEO_FRAME_COUNT, dtype=np.int64).tolist()
    if len(set(indices)) != VIDEO_FRAME_COUNT or indices[0] != 0 or indices[-1] != 499:
        raise RuntimeError("D363 video frame-index contract failed")
    writer = imageio_ffmpeg.write_frames(
        str(VIDEO_PATH),
        tuple(VIDEO_SIZE),
        pix_fmt_in="rgb24",
        pix_fmt_out=VIDEO_PIXEL_FORMAT,
        fps=VIDEO_FPS,
        codec=VIDEO_CODEC,
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=[
            "-movflags",
            "+faststart",
            "-metadata",
            "comment=D363 exact 1080 replay of immutable D362 canonical trace; physics not recomputed",
        ],
    )
    writer.send(None)
    try:
        for frame_number, row_index in enumerate(indices):
            image = d362._render_trace_replay_frame(topology_parts, rows, row_index)
            frame = np.asarray(image, dtype=np.uint8)
            if frame.shape != (1080, 1920, 3):
                raise RuntimeError(f"D363 frame shape drift: {frame.shape}")
            writer.send(np.ascontiguousarray(frame))
            if frame_number in {0, 49, 99, 149, 199, 249}:
                _marker(
                    "worker",
                    "trace_replay_video",
                    "progress",
                    {"completed": frame_number + 1, "source_row_index": row_index},
                )
    finally:
        writer.close()
    storyboard = Image.new("RGB", tuple(VIDEO_SIZE), (9, 13, 19))
    for panel_index, row_index in enumerate(DECISION_ROWS.values()):
        panel = d362._render_trace_replay_frame(topology_parts, rows, row_index).resize((960, 540))
        storyboard.paste(panel, ((panel_index % 2) * 960, (panel_index // 2) * 540))
    storyboard.save(VIDEO_STORYBOARD_PATH)
    _marker("worker", "trace_replay_video", "complete", {"frames": len(indices)})
    return {
        "video_path": _rel(VIDEO_PATH),
        "video_sha256": _sha(VIDEO_PATH),
        "storyboard_path": _rel(VIDEO_STORYBOARD_PATH),
        "storyboard_sha256": _sha(VIDEO_STORYBOARD_PATH),
        "source_indices": indices,
        "physics_recomputed": False,
        "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
    }


def _worker(args: argparse.Namespace) -> int:
    global _EXPLICIT_FORWARD_COUNT
    simulation_app = None
    inner = None
    settings = None
    previous_play: Any = None
    try:
        prereg = _json(PREREG_PATH)
        prepare = _json(PREPARE_PATH)
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
            "prereg_prepare_pass": prereg.get("pass") is True and prepare.get("pass") is True,
            "single_invocation": invocation.get("invocation_index") == 1
            and invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("automatic_retry") is False
            and invocation.get("preregistration_sha256") == _sha(PREREG_PATH)
            and invocation.get("prepare_preflight_sha256") == _sha(PREPARE_PATH),
            "registered_parent": supervisor_pid > 0
            and os.getppid() == supervisor_pid
            and invocation.get("supervisor_pid") == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest() == invocation.get("worker_token_sha256"),
            "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
            "git_scope_d363_only": _status_scope_ok(_git_status()),
            "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
            "d362_manifest_exact": _d362_manifest() == prereg.get("d362_manifest_before"),
            "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
            "registered_python": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
            "display_device_exact": os.environ.get("DISPLAY") == DISPLAY
            and args.headless is False
            and int(args.livestream) == 0
            and str(args.device) == "cuda:0",
            "runtime_modules_absent_before_applauncher": not early_runtime_modules,
            "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
            "ram_free_gate": int(gpu.get("ram_available_bytes") or 0) >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D363_WORKER_PREFLIGHT_V1",
            "utc": _utc_now(),
            "pid": os.getpid(),
            "early_runtime_modules": early_runtime_modules,
            "gpu_and_ram": gpu,
            "checks": checks,
            "pass": all(checks.values()),
        }
        _write_json_x(WORKER_PREFLIGHT_PATH, preflight)
        _marker("worker", "worker_preflight", "complete", {"pass": preflight["pass"]})
        if not preflight["pass"]:
            raise RuntimeError(f"D363 worker preflight STOP: {checks}")

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
        launcher_report = d362.d351.d350._resolved_gui_launcher(launcher)
        _marker("worker", "AppLauncher", "complete", {"pass": launcher_report.get("pass")})
        if not launcher_report.get("pass"):
            raise RuntimeError(f"D363 GUI launcher contract failed: {launcher_report}")

        import carb
        import omni.timeline

        args.robot_usd_path = d362.VARIANT_ROBOT_USD
        _marker("worker", "make_runtime_env", "start")
        inner = d362._make_runtime_env(args)
        _marker("worker", "make_runtime_env", "complete", {"pass": True})
        timeline = omni.timeline.get_timeline_interface()
        reset_before = {
            "custom_step_counter": int(inner._sim_step_counter),
            "simulation_clock": d362.d351._simulation_clock(inner),
        }
        _marker("worker", "reset", "start")
        inner.reset(seed=SEED)
        reset_after = {
            "custom_step_counter": int(inner._sim_step_counter),
            "simulation_clock": d362.d351._simulation_clock(inner),
        }
        _marker("worker", "reset", "complete", {"before": reset_before, "after": reset_after})
        settings = carb.settings.get_settings()
        previous_play = settings.get(PLAY_SIMULATIONS_SETTING)
        pause = _pause_no_advance(inner, timeline)
        if not pause["pass"]:
            raise RuntimeError(f"D363 pause bridge failed: {pause['checks']}")
        _marker("worker", "timeline_pause", "complete", {"pass": True})

        rows = _json(D362_TRACE)
        trace_contract = _trace_contract(rows)
        topology_parts, frozen_topology = _frozen_display_topology_parts()
        _marker(
            "worker",
            "frozen_display_topology",
            "complete",
            {"pass": frozen_topology.get("pass"), "fresh_live_query": False},
        )
        controlled_baseline = _physical_snapshot(inner, timeline)
        runtime_checks = {
            "trace_contract": trace_contract["pass"],
            "frozen_display_topology_64_plus_64": frozen_topology.get("pass") is True,
            "frozen_display_topology_exact_preregistered": frozen_topology
            == prereg.get("frozen_display_topology"),
            "no_fresh_live_topology_or_corrected_audit": frozen_topology.get("fresh_live_query_performed") is False,
            "joint_order_exact": list(inner._robot.joint_names) == list(d362.d351.d332.ALL_JOINT_NAMES),
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "controlled_counter_baseline_zero": int(inner._sim_step_counter) == 0,
            "simulation_clock_fields_available": all(
                controlled_baseline["simulation_clock"].get(key) is not None
                for key in ("current_time", "current_time_step_index")
            ),
        }
        runtime = {
            "artifact": "D363_RUNTIME_PREREQUISITES_V1",
            "launcher": launcher_report,
            "reset_internal_transition": {"before": reset_before, "after": reset_after, "excluded_from_d363_controlled_steps": True},
            "pause_bridge": pause,
            "controlled_zero_step_baseline": controlled_baseline,
            "frozen_display_topology": frozen_topology,
            "checks": runtime_checks,
            "pass": all(runtime_checks.values()),
        }
        _write_json_x(RUNTIME_PATH, runtime)
        if not runtime["pass"]:
            raise RuntimeError(f"D363 runtime prerequisites STOP: {runtime_checks}")

        state_reports: dict[str, Any] = {}
        for role, row_index in DECISION_ROWS.items():
            row = rows[row_index]
            _marker("worker", f"state_{role}", "direct_write_start", {"row_index": row_index})
            write = _direct_write_trace_state(inner, row)
            if not write["pass"]:
                raise RuntimeError(f"D363 exact trace-state write failed: {role}")
            state_reference = _physical_snapshot(inner, timeline)
            _marker("worker", f"state_{role}", "direct_write_complete", {"pass": True})
            before_captures = {
                "primary": _capture_viewport(
                    CAPTURE_PATHS[role]["before"]["primary"], simulation_app, inner, timeline, CAMERA_EYE, f"{role}_before_primary"
                ),
                "opposite": _capture_viewport(
                    CAPTURE_PATHS[role]["before"]["opposite"], simulation_app, inner, timeline, OPPOSITE_CAMERA_EYE, f"{role}_before_opposite"
                ),
            }
            before_forward = _physical_snapshot(inner, timeline)
            if not _snapshot_no_advance(state_reference, before_forward):
                raise RuntimeError(f"D363 before-forward guard failed: {role}")
            _marker("worker", f"state_{role}", "before_forward_captures_complete")

            forward_before = _physical_snapshot(inner, timeline)
            inner.sim.forward()
            _EXPLICIT_FORWARD_COUNT += 1
            forward_after = _physical_snapshot(inner, timeline)
            if not _snapshot_no_advance(forward_before, forward_after):
                raise RuntimeError(f"D363 explicit forward advanced physics/state: {role}")
            _marker("worker", f"state_{role}", "explicit_forward_complete", {"forward_count": _EXPLICIT_FORWARD_COUNT})

            after_captures = {
                "primary": _capture_viewport(
                    CAPTURE_PATHS[role]["after"]["primary"], simulation_app, inner, timeline, CAMERA_EYE, f"{role}_after_primary"
                ),
                "opposite": _capture_viewport(
                    CAPTURE_PATHS[role]["after"]["opposite"], simulation_app, inner, timeline, OPPOSITE_CAMERA_EYE, f"{role}_after_opposite"
                ),
            }
            terminal = _physical_snapshot(inner, timeline)
            if not _snapshot_no_advance(state_reference, terminal):
                raise RuntimeError(f"D363 state terminal guard failed: {role}")
            _marker("worker", f"state_{role}", "after_forward_captures_complete")
            state_reports[role] = {
                "row_index": row_index,
                "global_step": row["global_step"],
                "phase": row["phase"],
                "phase_step": row["phase_step"],
                "trace_q5_actual_rad": row["q5_actual_rad"],
                "trace_object_pos_w_m": row["object_pos_w_m"],
                "trace_object_quat_wxyz": row["object_quat_wxyz"],
                "trace_object_disp_xy_mm": row["object_disp_xy_mm"],
                "trace_object_tilt_delta_deg": row["object_tilt_delta_from_reference_deg"],
                "write": write,
                "state_reference": state_reference,
                "before_captures": before_captures,
                "forward_guard": {"before": forward_before, "after": forward_after, "pass": True},
                "after_captures": after_captures,
                "terminal_guard": terminal,
            }

        video = _build_video(topology_parts, rows)
        final_snapshot = _physical_snapshot(inner, timeline)
        counter_checks = {
            "display_state_writes_4": _DISPLAY_STATE_WRITE_COUNT == 4,
            "explicit_forward_calls_4": _EXPLICIT_FORWARD_COUNT == 4,
            "controlled_physics_steps_0": _CONTROLLED_PHYSICS_STEPS == 0,
            "q5_science_samples_0": _Q5_SCIENCE_SAMPLE_COUNT == 0,
            "q5_target_updates_0": _Q5_TARGET_UPDATE_COUNT == 0,
            "contact_queries_0": _CONTACT_QUERY_COUNT == 0,
            "clock_counter_unchanged_from_controlled_baseline": final_snapshot["custom_step_counter"]
            == controlled_baseline["custom_step_counter"]
            and final_snapshot["simulation_clock"] == controlled_baseline["simulation_clock"],
            "timeline_paused_not_stopped": not final_snapshot["timeline_playing"] and not final_snapshot["timeline_stopped"],
            "d362_manifest_unchanged": _d362_manifest() == prereg["d362_manifest_before"],
            "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "input_hashes_unchanged": _input_hashes() == prereg["input_hashes"],
        }
        summary = {
            "artifact": "D363_WORKER_SUMMARY_V1",
            "case": CASE,
            "utc": _utc_now(),
            "new_variables": NEW_VARIABLES,
            "state_reports": state_reports,
            "video_generation": video,
            "controlled_zero_step_baseline": controlled_baseline,
            "final_snapshot": final_snapshot,
            "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
            "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
            "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
            "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "contact_query_count": _CONTACT_QUERY_COUNT,
            "counter_checks": counter_checks,
            "target_ik_path_changed": False,
            "physics_settings_changed": False,
            "cap_rim_science": None,
            "grasp_or_g0a_science": None,
            "g0a_pass": False,
            "pass": all(counter_checks.values()),
        }
        _write_json_x(WORKER_SUMMARY_PATH, summary)
        _marker("worker", "worker_summary", "complete", {"pass": summary["pass"]})
        return 0 if summary["pass"] else 2
    except Exception as error:
        if not WORKER_EXCEPTION_PATH.exists():
            _write_json_x(
                WORKER_EXCEPTION_PATH,
                {
                    "artifact": "D363_WORKER_EXCEPTION_STOP_V1",
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
                    "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
                    "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
                    "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "contact_query_count": _CONTACT_QUERY_COUNT,
                    "automatic_retry": False,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        if settings is not None:
            try:
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


def _png_metrics(path: Path) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "path": _rel(path),
            "sha256": _sha(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
            "dimensions": None,
            "bbox_xywh": [0, 0, 0, 0],
            "centroid_xy_px": [0.0, 0.0],
            "area_px": 0,
            "pca_axis_angle_deg": 0.0,
            "height_width_ratio": 0.0,
            "width_height_ratio": 0.0,
            "component_mask_rle_not_stored": True,
            "_mask": np.zeros((VIEWPORT_SIZE[1], VIEWPORT_SIZE[0]), dtype=bool),
            "pass": False,
            "error": "decode failed",
        }
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MASK_HSV_LOW, MASK_HSV_HIGH)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return {
            "path": _rel(path),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "dimensions": [image.shape[1], image.shape[0]],
            "bbox_xywh": [0, 0, 0, 0],
            "centroid_xy_px": [0.0, 0.0],
            "area_px": 0,
            "pca_axis_angle_deg": 0.0,
            "height_width_ratio": 0.0,
            "width_height_ratio": 0.0,
            "component_mask_rle_not_stored": True,
            "_mask": np.zeros((image.shape[0], image.shape[1]), dtype=bool),
            "pass": False,
            "error": "yellow component absent",
        }
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = [int(value) for value in stats[component]]
    ys, xs = np.nonzero(labels == component)
    points = np.column_stack([xs, ys]).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "dimensions": [image.shape[1], image.shape[0]],
        "bbox_xywh": [x, y, w, h],
        "centroid_xy_px": [float(centroids[component][0]), float(centroids[component][1])],
        "area_px": area,
        "pca_axis_angle_deg": angle,
        "height_width_ratio": h / max(w, 1),
        "width_height_ratio": w / max(h, 1),
        "component_mask_rle_not_stored": True,
        "_mask": labels == component,
        "pass": [image.shape[1], image.shape[0]] == VIEWPORT_SIZE and area >= MASK_MIN_AREA_PX,
    }


def _axis_delta_deg(a: float, b: float) -> float:
    raw = abs(a - b) % 180.0
    return min(raw, 180.0 - raw)


def _compare_masks(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a.get("pass") is not True or b.get("pass") is not True:
        return {
            "centroid_delta_px": None,
            "axis_delta_deg": None,
            "mask_iou": None,
            "criteria": {
                "centroid_delta_ge_15px": False,
                "axis_delta_ge_15deg": False,
                "mask_iou_le_0p85": False,
            },
            "criterion_count": 0,
            "materially_different": False,
            "input_metrics_pass": False,
        }
    mask_a = a["_mask"]
    mask_b = b["_mask"]
    if mask_a.shape != mask_b.shape:
        return {
            "centroid_delta_px": None,
            "axis_delta_deg": None,
            "mask_iou": None,
            "criteria": {
                "centroid_delta_ge_15px": False,
                "axis_delta_ge_15deg": False,
                "mask_iou_le_0p85": False,
            },
            "criterion_count": 0,
            "materially_different": False,
            "input_metrics_pass": False,
            "error": "mask shape mismatch",
        }
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    iou = intersection / max(union, 1)
    centroid = float(np.linalg.norm(np.asarray(a["centroid_xy_px"]) - np.asarray(b["centroid_xy_px"])))
    axis = _axis_delta_deg(float(a["pca_axis_angle_deg"]), float(b["pca_axis_angle_deg"]))
    criteria = {
        "centroid_delta_ge_15px": centroid >= MATERIAL_CENTROID_DELTA_PX,
        "axis_delta_ge_15deg": axis >= MATERIAL_AXIS_DELTA_DEG,
        "mask_iou_le_0p85": iou <= MATERIAL_IOU_MAX,
    }
    return {
        "centroid_delta_px": centroid,
        "axis_delta_deg": axis,
        "mask_iou": iou,
        "criteria": criteria,
        "criterion_count": sum(criteria.values()),
        "materially_different": sum(criteria.values()) >= MATERIAL_MIN_CRITERIA,
        "input_metrics_pass": True,
    }


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "_mask"}


def _relative_trace_render_correspondence(
    worker: dict[str, Any],
    private: dict[str, dict[str, dict[str, dict[str, Any]]]],
    view: str,
    *,
    swap_expected_roles: bool = False,
) -> dict[str, Any]:
    """Compare relative trace motion/orientation to the captured yellow component.

    The inherited 48-degree D362 projection is used only for direction and axis
    deltas, never as an absolute pixel calibration of the Isaac viewport.
    """
    camera_eye = CAMERA_EYE if view == "primary" else OPPOSITE_CAMERA_EYE
    pre = worker["state_reports"]["precommand"]
    final = worker["state_reports"]["final"]
    expected_states = [pre, final]
    if swap_expected_roles:
        expected_states = [final, pre]
    centers = np.asarray(
        [state["trace_object_pos_w_m"] for state in expected_states], dtype=np.float64
    )
    projected_centers, center_depth = d362._project_world_points(
        centers, camera_eye, CAMERA_TARGET, (0, 0, VIEWPORT_SIZE[0], VIEWPORT_SIZE[1])
    )

    def projected_axis_angle(state: dict[str, Any]) -> float:
        center = np.asarray(state["trace_object_pos_w_m"], dtype=np.float64)
        rotation = _quat_rot(state["trace_object_quat_wxyz"])
        axis = rotation @ np.asarray([0.0, 0.0, 0.045], dtype=np.float64)
        points, depth = d362._project_world_points(
            np.vstack([center - axis, center + axis]),
            camera_eye,
            CAMERA_TARGET,
            (0, 0, VIEWPORT_SIZE[0], VIEWPORT_SIZE[1]),
        )
        if np.any(depth <= 0.0) or not np.isfinite(points).all():
            return math.nan
        vector = points[1] - points[0]
        return math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0

    expected_axis_angles = [projected_axis_angle(state) for state in expected_states]
    observed_pre = private["precommand"]["after"][view]
    observed_final = private["final"]["after"][view]
    valid_inputs = bool(
        observed_pre.get("pass") is True
        and observed_final.get("pass") is True
        and np.all(center_depth > 0.0)
        and np.isfinite(projected_centers).all()
        and np.isfinite(expected_axis_angles).all()
    )
    if not valid_inputs:
        return {
            "projection_role_swapped_negative_control": swap_expected_roles,
            "absolute_pixel_calibration_claimed": False,
            "valid_inputs": False,
            "direction_cosine": None,
            "expected_axis_delta_deg": None,
            "observed_axis_delta_deg": None,
            "axis_delta_error_deg": None,
            "observed_displacement_px": None,
            "criteria": {},
            "pass": False,
        }
    expected_vector = projected_centers[1] - projected_centers[0]
    observed_vector = np.asarray(observed_final["centroid_xy_px"]) - np.asarray(
        observed_pre["centroid_xy_px"]
    )
    expected_norm = float(np.linalg.norm(expected_vector))
    observed_norm = float(np.linalg.norm(observed_vector))
    direction_cosine = float(
        np.dot(expected_vector, observed_vector) / max(expected_norm * observed_norm, 1.0e-12)
    )
    expected_axis_delta = _axis_delta_deg(*expected_axis_angles)
    observed_axis_delta = _axis_delta_deg(
        float(observed_pre["pca_axis_angle_deg"]),
        float(observed_final["pca_axis_angle_deg"]),
    )
    axis_error = abs(expected_axis_delta - observed_axis_delta)
    criteria = {
        "direction_cosine_ge_0p8": direction_cosine >= 0.8,
        "axis_delta_error_le_15deg": axis_error <= 15.0,
        "observed_displacement_ge_15px": observed_norm >= MATERIAL_CENTROID_DELTA_PX,
    }
    return {
        "projection_role_swapped_negative_control": swap_expected_roles,
        "absolute_pixel_calibration_claimed": False,
        "expected_projected_centers_px_display_only": projected_centers.tolist(),
        "observed_centers_px": [
            observed_pre["centroid_xy_px"],
            observed_final["centroid_xy_px"],
        ],
        "direction_cosine": direction_cosine,
        "expected_axis_angles_deg_display_only": expected_axis_angles,
        "expected_axis_delta_deg": expected_axis_delta,
        "observed_axis_delta_deg": observed_axis_delta,
        "axis_delta_error_deg": axis_error,
        "observed_displacement_px": observed_norm,
        "criteria": criteria,
        "direction_is_mandatory": True,
        "minimum_criteria": 2,
        "valid_inputs": True,
        "pass": criteria["direction_cosine_ge_0p8"] and sum(criteria.values()) >= 2,
    }


def _verify_video() -> dict[str, Any]:
    import cv2
    import imageio_ffmpeg
    from PIL import Image

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    samples: list[dict[str, Any]] = []
    for index in (0, 125, 249):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        samples.append(
            {
                "frame_index": index,
                "decode_ok": bool(ok),
                "shape": list(frame.shape) if ok else None,
                "pixel_stddev": float(np.std(frame)) if ok else None,
                "nonblank": bool(ok and float(np.std(frame)) > 5.0),
            }
        )
    capture.release()
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    decoded = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(VIDEO_PATH), "-f", "null", "-"],
        text=True,
        capture_output=True,
        check=False,
        timeout=180.0,
    )
    with Image.open(VIDEO_STORYBOARD_PATH) as image:
        image.load()
        storyboard_dimensions = list(image.size)
        storyboard_stddev = float(np.std(np.asarray(image)))
    source_indices = _json(WORKER_SUMMARY_PATH)["video_generation"]["source_indices"]
    old_report = _json(D362_VIDEO_REPORT)
    expected_indices = np.linspace(0, 499, VIDEO_FRAME_COUNT, dtype=np.int64).tolist()
    old_indices = old_report.get("source_row_indices")
    source_indices_sha256 = hashlib.sha256(
        json.dumps(source_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_indices_sha256 = hashlib.sha256(
        json.dumps(expected_indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    worker_video = _json(WORKER_SUMMARY_PATH)["video_generation"]
    checks = {
        "video_nonempty": VIDEO_PATH.is_file() and VIDEO_PATH.stat().st_size > 0,
        "resolution_exact_1920x1080": [width, height] == VIDEO_SIZE,
        "fps_exact_20": math.isclose(fps, VIDEO_FPS, rel_tol=0.0, abs_tol=0.01),
        "frame_count_exact_250": frames == VIDEO_FRAME_COUNT,
        "first_middle_last_decode_nonblank": all(row["decode_ok"] and row["nonblank"] for row in samples),
        "ffmpeg_full_decode": decoded.returncode == 0,
        "codec_h264": "Video: h264" in decoded.stderr,
        "pixel_format_yuv420p": "yuv420p" in decoded.stderr,
        "storyboard_exact_nonblank": storyboard_dimensions == VIDEO_SIZE and storyboard_stddev > 5.0,
        "source_indices_exact_registered_sequence": source_indices == expected_indices,
        "source_indices_exact_d362_sequence": source_indices == old_indices,
        "source_indices_digest_exact": source_indices_sha256 == expected_indices_sha256,
        "worker_video_sha_current": worker_video.get("video_sha256") == _sha(VIDEO_PATH),
        "worker_storyboard_sha_current": worker_video.get("storyboard_sha256") == _sha(VIDEO_STORYBOARD_PATH),
        "old_d362_failure_stays_1920x1088": old_report.get("reported_resolution") == [1920, 1088]
        and old_report.get("pass") is False,
        "physics_not_recomputed": True,
    }
    report = {
        "artifact": "D363_D362_TRACE_REPLAY_REPORT_V1",
        "case": CASE,
        "authority": "immutable D362 canonical trace plus inherited frozen D348 64+64 display topology; no fresh live query and physics not recomputed",
        "canonical_trace_path": _rel(D362_TRACE),
        "canonical_trace_sha256": _sha(D362_TRACE),
        "video_path": _rel(VIDEO_PATH),
        "video_sha256": _sha(VIDEO_PATH),
        "video_bytes": VIDEO_PATH.stat().st_size,
        "storyboard_path": _rel(VIDEO_STORYBOARD_PATH),
        "storyboard_sha256": _sha(VIDEO_STORYBOARD_PATH),
        "reported_resolution": [width, height],
        "reported_fps": fps,
        "reported_frame_count": frames,
        "duration_seconds": frames / fps if fps else None,
        "decode_samples": samples,
        "full_decode": {"ffmpeg": str(ffmpeg), "returncode": decoded.returncode, "stderr": decoded.stderr},
        "storyboard_dimensions": storyboard_dimensions,
        "storyboard_pixel_stddev": storyboard_stddev,
        "source_indices": source_indices,
        "expected_source_indices_sha256": expected_indices_sha256,
        "observed_source_indices_sha256": source_indices_sha256,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(VIDEO_REPORT_PATH, report)
    return report


def _build_sync_report(worker: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    private: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for role in DECISION_ROWS:
        metrics[role] = {}
        private[role] = {}
        for timing in ("before", "after"):
            metrics[role][timing] = {}
            private[role][timing] = {}
            for view in ("primary", "opposite"):
                row = _png_metrics(CAPTURE_PATHS[role][timing][view])
                private[role][timing][view] = row
                metrics[role][timing][view] = _public_metrics(row)
    stale = _png_metrics(D362_STALE_FINAL)
    comparisons: dict[str, Any] = {}
    correspondence: dict[str, Any] = {}
    for view in ("primary", "opposite"):
        pre = private["precommand"]["after"][view]
        final_before = private["final"]["before"][view]
        final_after = private["final"]["after"][view]
        comparisons[view] = {
            "precommand_after_vs_final_after": _compare_masks(pre, final_after),
            "final_before_vs_final_after": _compare_masks(final_before, final_after),
            "precommand_after_upright": pre["height_width_ratio"] >= UPRIGHT_HW_MIN,
            "final_after_toppled": final_after["width_height_ratio"] >= TOPPLED_WH_MIN,
        }
        if view == "primary":
            comparisons[view]["d362_stale_final_vs_d363_final_after"] = _compare_masks(stale, final_after)
        before_diff = comparisons[view]["final_before_vs_final_after"]
        comparisons[view]["forward_effect_classification"] = (
            "stale_reproduced"
            if before_diff["materially_different"]
            else "already_synced_or_visually_ambiguous"
        )
        correspondence[view] = {
            "registered": _relative_trace_render_correspondence(worker, private, view),
            "swapped_role_negative_control": _relative_trace_render_correspondence(
                worker, private, view, swap_expected_roles=True
            ),
        }
    all_png_pass = all(
        metrics[role][timing][view]["pass"]
        for role in DECISION_ROWS
        for timing in ("before", "after")
        for view in ("primary", "opposite")
    )
    checks = {
        "all_16_png_decode_mask_pass": all_png_pass,
        "all_write_readback_bits_exact": all(row["write"]["pass"] for row in worker["state_reports"].values()),
        "worker_no_advance_counters_pass": all(worker["counter_checks"].values()),
        "primary_precommand_upright": comparisons["primary"]["precommand_after_upright"],
        "primary_final_toppled": comparisons["primary"]["final_after_toppled"],
        "primary_precommand_final_materially_different": comparisons["primary"]["precommand_after_vs_final_after"]["materially_different"],
        "opposite_precommand_final_materially_different": comparisons["opposite"]["precommand_after_vs_final_after"]["materially_different"],
        "primary_relative_trace_render_correspondence": correspondence["primary"]["registered"]["pass"],
        "opposite_relative_trace_render_correspondence": correspondence["opposite"]["registered"]["pass"],
        "primary_swapped_role_negative_rejected": not correspondence["primary"]["swapped_role_negative_control"]["pass"],
        "opposite_swapped_role_negative_rejected": not correspondence["opposite"]["swapped_role_negative_control"]["pass"],
        "new_final_differs_from_d362_stale_final": comparisons["primary"]["d362_stale_final_vs_d363_final_after"]["materially_different"],
        "trace_precommand_upright": float(worker["state_reports"]["precommand"]["trace_object_tilt_delta_deg"]) < 0.01,
        "trace_final_moved_toppled": float(worker["state_reports"]["final"]["trace_object_disp_xy_mm"]) > 60.0
        and float(worker["state_reports"]["final"]["trace_object_tilt_delta_deg"]) > 89.0,
        "d362_manifest_unchanged": _d362_manifest() == _json(PREREG_PATH)["d362_manifest_before"],
        "d363_d362_inode_disjoint": _d363_d362_inode_disjoint(),
    }
    report = {
        "artifact": "D363_FABRIC_RENDER_SYNC_REPORT_V1",
        "case": CASE,
        "authority_separation": {
            "canonical_state": _rel(D362_TRACE),
            "actual_renderer": "D363 before/after-forward Isaac viewport PNGs",
            "yellow_mask": "inspection metric only; never hashed into D362 science",
            "correspondence_scope": "relative screen-motion direction and projected-axis delta only; no absolute pixel-pose calibration claim",
        },
        "registered_thresholds": _json(PREREG_PATH)["silhouette_contract"],
        "capture_metrics": metrics,
        "d362_stale_final_metrics": _public_metrics(stale),
        "comparisons": comparisons,
        "relative_trace_render_correspondence": correspondence,
        "before_forward_difference_is_mandatory": False,
        "controlled_physics_steps": worker["controlled_physics_steps"],
        "q5_science_sample_count": worker["q5_science_sample_count"],
        "q5_target_update_count": worker["q5_target_update_count"],
        "contact_query_count": worker["contact_query_count"],
        "explicit_forward_count": worker["explicit_forward_count"],
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(SYNC_REPORT_PATH, report)
    return report, private


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_PATH), size=size)


def _build_storyboard(path: Path, view: str) -> None:
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (3840, 2160), (12, 17, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((1920, 45), f"D363 zero-step Fabric 동기화 — {view} view", font=_font(52), fill=(245, 248, 252), anchor="ma")
    roles = list(DECISION_ROWS)
    for col, role in enumerate(roles):
        x = 30 + col * 950
        draw.text((x + 455, 125), role.replace("_", " "), font=_font(30), fill=(126, 203, 255), anchor="ma")
        for row_index, timing in enumerate(("before", "after")):
            y = 175 + row_index * 860
            with Image.open(CAPTURE_PATHS[role][timing][view]) as source:
                panel = ImageOps.fit(source.convert("RGB"), (900, 506), method=Image.Resampling.LANCZOS)
            canvas.paste(panel, (x, y))
            color = (255, 190, 75) if timing == "before" else (85, 224, 145)
            draw.rectangle((x, y, x + 900, y + 506), outline=color, width=5)
            draw.text((x + 450, y + 535), f"{timing.upper()} explicit forward", font=_font(29), fill=color, anchor="ma")
            draw.text((x + 450, y + 588), "physics clock/counter 불변", font=_font(23), fill=(211, 220, 233), anchor="ma")
    draw.text(
        (1920, 2080),
        "표시 재생만 수행: controlled physics 0 · q5 science/target update 0 · 각 state explicit forward 1회",
        font=_font(30),
        fill=(186, 197, 214),
        anchor="ma",
    )
    canvas.save(path)


def _build_beginner_sheet(sync: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (3840, 2160), (10, 15, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text((1920, 40), "D363 결과: 기록된 전도 자세가 실제 Isaac 화면에 반영됐는가", font=_font(55), fill=(247, 249, 252), anchor="ma")
    panels = [
        (D362_STALE_FINAL, "D362 final — stale upright", (255, 108, 102)),
        (CAPTURE_PATHS["precommand"]["after"]["primary"], "D363 precommand after", (98, 180, 255)),
        (CAPTURE_PATHS["final"]["before"]["primary"], "D363 final before", (255, 191, 79)),
        (CAPTURE_PATHS["final"]["after"]["primary"], "D363 final after", (80, 224, 142)),
    ]
    for index, (path, label, color) in enumerate(panels):
        x = 35 + index * 950
        with Image.open(path) as source:
            panel = ImageOps.fit(source.convert("RGB"), (900, 506), method=Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, 150))
        draw.rectangle((x, 150, x + 900, 656), outline=color, width=6)
        draw.text((x + 450, 690), label, font=_font(29), fill=color, anchor="ma")
    comp = sync["comparisons"]["primary"]
    old_new = comp["d362_stale_final_vs_d363_final_after"]
    pre_final = comp["precommand_after_vs_final_after"]
    before_after = comp["final_before_vs_final_after"]
    draw.rounded_rectangle((70, 805, 1870, 1960), radius=25, fill=(27, 38, 52), outline=(80, 224, 142), width=5)
    draw.text((120, 850), "이번 case에서 실제로 확인한 것", font=_font(45), fill=(102, 238, 159))
    lines = [
        "• D362 500행 trace는 그대로 읽었고 물리를 재실행하지 않음",
        "• 새 MP4는 exact 1920×1080 / 250 frame / 20fps",
        "• 4개 기록 상태를 직접 쓰고 state마다 forward() 1회",
        "• controlled physics step 0, q5 science/target update 0",
        f"• precommand→final: centroid {pre_final['centroid_delta_px']:.1f}px, axis {pre_final['axis_delta_deg']:.1f}°, IoU {pre_final['mask_iou']:.3f}",
        f"• old stale final→new final: centroid {old_new['centroid_delta_px']:.1f}px, axis {old_new['axis_delta_deg']:.1f}°, IoU {old_new['mask_iou']:.3f}",
        f"• final before→after 분류: {comp['forward_effect_classification']}",
        f"  (centroid {before_after['centroid_delta_px']:.1f}px, axis {before_after['axis_delta_deg']:.1f}°, IoU {before_after['mask_iou']:.3f})",
    ]
    draw.multiline_text((125, 945), "\n".join(lines), font=_font(29), fill=(232, 239, 247), spacing=24)
    draw.rounded_rectangle((1970, 805, 3770, 1960), radius=25, fill=(54, 32, 34), outline=(255, 111, 103), width=5)
    draw.text((2020, 850), "이번 case로 새로 증명하지 않은 것", font=_font(45), fill=(255, 139, 132))
    draw.multiline_text(
        (2025, 965),
        "• 접촉력을 다시 계산하지 않음\n• cap/rim/barrel 중 어느 면이 먼저 닿았는지\n• 양쪽 jaw force closure 또는 안정 파지\n• target/IK/path 수리 필요성\n• G0a 성공\n\nD362의 body-level 결과만 상속:\n움직이는 jaw 접촉 뒤 원통 운동 관찰.\n최종 원통은 잡힌 것이 아니라 밀려 넘어짐.",
        font=_font(31),
        fill=(248, 231, 230),
        spacing=27,
    )
    draw.text((1920, 2070), "원자료: d363_fabric_render_sync_report.json · d362_physics_trace.json", font=_font(30), fill=(176, 192, 215), anchor="ma")
    canvas.save(BEGINNER_SHEET_PATH)


def _quat_rot(quat_wxyz: list[float]) -> np.ndarray:
    return d362.d351.d332._quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64))


def _write_rerun(sync: dict[str, Any], private: dict[str, Any], worker: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"D363 rerun version drift: {rr.__version__}")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/summary/beginner_sheet", name="D363 paired decision sheet"),
                rrb.Spatial2DView(origin="/captures/primary", name="actual Isaac primary before/after"),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/captures/opposite", name="actual Isaac opposite before/after"),
                rrb.Spatial3DView(origin="/geometry", name="canonical D362 cylinder pose"),
                rrb.TimeSeriesView(origin="/metrics", name="mask and no-advance metrics"),
                rrb.TextLogView(origin="/events", name="sync events"),
                row_shares=[2, 2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    expected_entities = {
        "/metadata/run",
        "/summary/beginner_sheet",
        "/captures/primary",
        "/captures/opposite",
        "/geometry/cylinder_expected",
        "/metrics/cylinder/centroid_x_px",
        "/metrics/cylinder/centroid_y_px",
        "/metrics/cylinder/pca_axis_angle_deg",
        "/metrics/cylinder/bbox_width_px",
        "/metrics/cylinder/bbox_height_px",
        "/metrics/cylinder/mask_area_px",
        "/metrics/trace/object_xy_disp_mm",
        "/metrics/trace/object_tilt_delta_deg",
        "/metrics/control/explicit_forward_count",
        "/metrics/control/controlled_physics_steps",
        "/metrics/control/q5_target_updates",
        "/metrics/control/q5_science_samples",
        "/metrics/control/contact_queries",
        "/metrics/control/simulation_time_delta_s",
        "/metrics/control/simulation_step_index_delta",
        "/metrics/control/timeline_time_delta_s",
        "/metrics/control/custom_step_counter_delta",
        "/events/sync",
    }
    cylinder_vertices, cylinder_triangles = d362.d351._cylinder_mesh()
    app_id = "roarm_g0a_d363_fabric_render_sync"
    with rr.RecordingStream(app_id, recording_id="g0a_d363_fabric_render_sync", make_default=False, send_properties=True) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            "metadata/run",
            rr.TextDocument(
                json.dumps(
                    {
                        "case": CASE,
                        "purpose": "actual Isaac before/after explicit Fabric-forward synchronization over immutable D362 trace",
                        "canonical_authority": _rel(D362_TRACE),
                        "physics_recomputed": False,
                        "controlled_physics_steps": 0,
                        "q5_science_samples": 0,
                        "q5_target_updates": 0,
                        "g0a_pass": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            ),
            static=True,
        )
        beginner_rgb = cv2.cvtColor(
            cv2.imread(str(BEGINNER_SHEET_PATH)), cv2.COLOR_BGR2RGB
        )
        recording.log("summary/beginner_sheet", rr.Image(beginner_rgb), static=True)
        sync_step = 0
        for role in DECISION_ROWS:
            state = worker["state_reports"][role]
            forward_before = state["forward_guard"]["before"]
            forward_after = state["forward_guard"]["after"]
            clock_before = forward_before["simulation_clock"]
            clock_after = forward_after["simulation_clock"]
            simulation_time_delta = float(
                clock_after["current_time"] - clock_before["current_time"]
            )
            simulation_step_index_delta = int(
                clock_after["current_time_step_index"]
                - clock_before["current_time_step_index"]
            )
            timeline_time_delta = float(
                forward_after["timeline_time_s"] - forward_before["timeline_time_s"]
            )
            custom_counter_delta = int(
                forward_after["custom_step_counter"]
                - forward_before["custom_step_counter"]
            )
            rot = _quat_rot(state["trace_object_quat_wxyz"])
            world_vertices = cylinder_vertices @ rot.T + np.asarray(state["trace_object_pos_w_m"], dtype=np.float64)
            for timing in ("before", "after"):
                recording.reset_time()
                recording.set_time("sync_step", sequence=sync_step)
                primary = cv2.cvtColor(cv2.imread(str(CAPTURE_PATHS[role][timing]["primary"])), cv2.COLOR_BGR2RGB)
                opposite = cv2.cvtColor(cv2.imread(str(CAPTURE_PATHS[role][timing]["opposite"])), cv2.COLOR_BGR2RGB)
                recording.log("captures/primary", rr.Image(primary))
                recording.log("captures/opposite", rr.Image(opposite))
                recording.log(
                    "geometry/cylinder_expected",
                    rr.Mesh3D(
                        vertex_positions=world_vertices.astype(np.float32),
                        triangle_indices=np.asarray(cylinder_triangles, dtype=np.uint32),
                        albedo_factor=[245, 172, 52, 190],
                    ),
                )
                metric = private[role][timing]["primary"]
                scalar_values = {
                    "cylinder/centroid_x_px": metric["centroid_xy_px"][0],
                    "cylinder/centroid_y_px": metric["centroid_xy_px"][1],
                    "cylinder/pca_axis_angle_deg": metric["pca_axis_angle_deg"],
                    "cylinder/bbox_width_px": metric["bbox_xywh"][2],
                    "cylinder/bbox_height_px": metric["bbox_xywh"][3],
                    "cylinder/mask_area_px": metric["area_px"],
                    "trace/object_xy_disp_mm": state["trace_object_disp_xy_mm"],
                    "trace/object_tilt_delta_deg": state["trace_object_tilt_delta_deg"],
                    "control/explicit_forward_count": (list(DECISION_ROWS).index(role) + 1 if timing == "after" else list(DECISION_ROWS).index(role)),
                    "control/controlled_physics_steps": 0,
                    "control/q5_target_updates": 0,
                    "control/q5_science_samples": worker["q5_science_sample_count"],
                    "control/contact_queries": worker["contact_query_count"],
                    "control/simulation_time_delta_s": simulation_time_delta,
                    "control/simulation_step_index_delta": simulation_step_index_delta,
                    "control/timeline_time_delta_s": timeline_time_delta,
                    "control/custom_step_counter_delta": custom_counter_delta,
                }
                for suffix, value in scalar_values.items():
                    recording.log(f"metrics/{suffix}", rr.Scalars([float(value)]))
                recording.log(
                    "events/sync",
                    rr.TextLog(
                        f"{role} {timing}-forward; immutable D362 row {DECISION_ROWS[role]}; physics clock unchanged",
                        level="INFO",
                    ),
                )
                sync_step += 1
        recording.flush(timeout_sec=30.0)
    blueprint.save(app_id, RBL_PATH)
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        expected_timeline_names=["sync_step"],
        exact_entity_paths=sorted(expected_entities),
        exact_timeline_names=["blueprint", "log_time", "sync_step"],
        expected_entity_components={
            "metadata/run": ["TextDocument:text"],
            "summary/beginner_sheet": ["Image:buffer", "Image:format"],
            "captures/primary": ["Image:buffer", "Image:format"],
            "captures/opposite": ["Image:buffer", "Image:format"],
            "geometry/cylinder_expected": ["Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"],
            "metrics/cylinder/centroid_x_px": ["Scalars:scalars"],
            "metrics/control/controlled_physics_steps": ["Scalars:scalars"],
            "metrics/control/simulation_time_delta_s": ["Scalars:scalars"],
            "events/sync": ["TextLog:level", "TextLog:text"],
        },
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="3200x1800",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    return {
        "rrd_path": _rel(RRD_PATH),
        "rrd_sha256": _sha(RRD_PATH),
        "rbl_path": _rel(RBL_PATH),
        "rbl_sha256": _sha(RBL_PATH),
        "screenshot_path": _rel(RERUN_PNG_PATH),
        "screenshot_sha256": _sha(RERUN_PNG_PATH) if RERUN_PNG_PATH.is_file() else None,
        "validation_path": _rel(RERUN_VALIDATION_PATH),
        "validation_sha256": _sha(RERUN_VALIDATION_PATH),
        "pass": validation.get("pass") is True,
    }


def _stable_file_report(path: Path) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for _ in range(3):
        if path.is_file():
            observations.append({"bytes": path.stat().st_size, "sha256": _sha(path)})
        else:
            observations.append({"bytes": None, "sha256": None})
        time.sleep(0.2)
    return {"path": _rel(path), "observations": observations, "stable": len({json.dumps(row, sort_keys=True) for row in observations}) == 1 and observations[0]["bytes"] not in (None, 0)}


def _phase_contract(path: Path, owner: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sequence_exact = [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    required = (
        [("worker_preflight", "complete"), ("AppLauncher", "start"), ("AppLauncher", "complete"), ("make_runtime_env", "start"), ("make_runtime_env", "complete"), ("reset", "start"), ("reset", "complete"), ("trace_replay_video", "start"), ("trace_replay_video", "complete"), ("worker_summary", "complete")]
        if owner == "worker"
        else [
            ("supervisor", "start"),
            ("worker_process", "start"),
            ("worker_process", "exit"),
            ("supervisor_summary", "complete"),
        ]
    )
    counts = {f"{phase}:{event}": sum(row["phase"] == phase and row["event"] == event for row in rows) for phase, event in required}
    checks = {
        "nonempty": bool(rows),
        "sequence_exact": sequence_exact,
        "owner_exact": all(row.get("owner") == owner for row in rows),
        "monotonic_ns": all(rows[i]["monotonic_ns"] <= rows[i + 1]["monotonic_ns"] for i in range(len(rows) - 1)),
        "required_exact_once": all(value == 1 for value in counts.values()),
    }
    details: dict[str, Any] = {}
    if owner == "worker":
        state_events = [
            "direct_write_start",
            "direct_write_complete",
            "before_forward_captures_complete",
            "explicit_forward_complete",
            "after_forward_captures_complete",
        ]
        state_positions: list[int] = []
        state_counts: dict[str, int] = {}
        for role in DECISION_ROWS:
            phase = f"state_{role}"
            for event in state_events:
                matching = [
                    index
                    for index, row in enumerate(rows)
                    if row.get("phase") == phase and row.get("event") == event
                ]
                state_counts[f"{phase}:{event}"] = len(matching)
                if len(matching) == 1:
                    state_positions.append(matching[0])
        capture_rows = [
            row
            for row in rows
            if row.get("phase") == "viewport_capture" and row.get("event") == "complete"
        ]
        expected_capture_roles = {
            f"{role}_{timing}_{view}"
            for role in DECISION_ROWS
            for timing in ("before", "after")
            for view in ("primary", "opposite")
        }
        observed_capture_roles = [row.get("details", {}).get("capture_role") for row in capture_rows]
        forward_rows = [
            row
            for row in rows
            if row.get("event") == "explicit_forward_complete"
            and str(row.get("phase", "")).startswith("state_")
        ]
        video_start_positions = [
            index
            for index, row in enumerate(rows)
            if row.get("phase") == "trace_replay_video" and row.get("event") == "start"
        ]
        worker_summary_positions = [
            index
            for index, row in enumerate(rows)
            if row.get("phase") == "worker_summary" and row.get("event") == "complete"
        ]
        checks.update(
            {
                "state_events_exact_once": all(value == 1 for value in state_counts.values()),
                "state_role_and_event_order_exact": len(state_positions) == 20
                and state_positions == sorted(state_positions),
                "capture_markers_exact_16_unique": len(capture_rows) == 16
                and len(set(observed_capture_roles)) == 16
                and set(observed_capture_roles) == expected_capture_roles,
                "explicit_forward_markers_exact_4_counts_1_to_4": len(forward_rows) == 4
                and [row.get("details", {}).get("forward_count") for row in forward_rows]
                == [1, 2, 3, 4],
                "video_after_all_state_events": len(video_start_positions) == 1
                and len(state_positions) == 20
                and video_start_positions[0] > state_positions[-1],
                "worker_summary_after_video": len(worker_summary_positions) == 1
                and len(video_start_positions) == 1
                and worker_summary_positions[0] > video_start_positions[0],
            }
        )
        details = {
            "state_event_counts": state_counts,
            "capture_roles": observed_capture_roles,
            "forward_counts": [row.get("details", {}).get("forward_count") for row in forward_rows],
        }
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "row_count": len(rows),
        "required_counts": counts,
        "details": details,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run(_args: argparse.Namespace) -> int:
    prereg = _json(PREREG_PATH)
    prepare = _json(PREPARE_PATH)
    if prereg.get("pass") is not True or prepare.get("pass") is not True:
        raise RuntimeError("D363 prepare/preregistration did not pass")
    existing_inventory = sorted(path.resolve() for path in OUT_DIR.rglob("*") if path.is_file())
    expected_prepare_inventory = sorted([PREREG_PATH.resolve(), PREPARE_PATH.resolve()])
    if existing_inventory != expected_prepare_inventory:
        raise RuntimeError(
            "D363 run/resume/overwrite forbidden; output inventory is not prepare-only: "
            f"{[_rel(path) for path in existing_inventory]}"
        )
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_d363_only": _status_scope_ok(_git_status()),
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "d362_manifest_exact": _d362_manifest() == prereg["d362_manifest_before"],
        "sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "prereg_artifact_case_exact": prereg.get("artifact") == "D363_PREREGISTRATION_V1"
        and prereg.get("case") == CASE,
        "prepare_artifact_case_exact": prepare.get("artifact") == "D363_PREPARE_PREFLIGHT_V1"
        and prepare.get("case") == CASE,
        "prepare_binds_prereg_hash": prepare.get("preregistration_sha256") == _sha(PREREG_PATH),
        "prepare_all_checks_pass": prepare.get("pass") is True
        and bool(prepare.get("checks"))
        and all(prepare["checks"].values()),
        "prepare_only_inventory_exact": existing_inventory == expected_prepare_inventory,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D363 pre-invocation STOP: {checks}")
    _marker("supervisor", "supervisor", "start", {"checks": checks})
    token = secrets.token_hex(32)
    invocation = {
        "artifact": "D363_ISAAC_INVOCATION_MARKER_V1",
        "case": CASE,
        "utc": _utc_now(),
        "run_nonce": prereg["run_nonce"],
        "invocation_index": 1,
        "supervisor_pid": os.getpid(),
        "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "preregistration_sha256": _sha(PREREG_PATH),
        "prepare_preflight_sha256": _sha(PREPARE_PATH),
        "harness_sha256": _sha(HARNESS),
        "automatic_retry": False,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    command = [REGISTERED_PYTHON, str(HARNESS), "--stage", "_worker", "--out_dir", str(OUT_DIR), "--seed", str(SEED)]
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    env[WORKER_TOKEN_ENV] = token
    env[SUPERVISOR_PID_ENV] = str(os.getpid())
    start = time.monotonic()
    telemetry: list[dict[str, Any]] = []
    watchdog_reason: str | None = None
    last_progress_mtime = start
    with WORKER_LOG_PATH.open("xb") as log_stream:
        _marker("supervisor", "worker_process", "start", {"command": command})
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ps_process = psutil.Process(process.pid)
        last_phase_size = 0
        while process.poll() is None:
            elapsed = time.monotonic() - start
            try:
                phase_size = WORKER_PHASE_PATH.stat().st_size if WORKER_PHASE_PATH.exists() else 0
                if phase_size != last_phase_size:
                    last_phase_size = phase_size
                    last_progress_mtime = time.monotonic()
                telemetry.append(
                    {
                        "elapsed_s": elapsed,
                        "worker_rss_bytes": ps_process.memory_info().rss,
                        "worker_cpu_percent": ps_process.cpu_percent(interval=None),
                        "gpu": _gpu_snapshot(),
                        "phase_bytes": phase_size,
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if elapsed > TOTAL_WATCHDOG_S:
                watchdog_reason = "total_wall_clock"
            elif time.monotonic() - last_progress_mtime > INACTIVITY_WATCHDOG_S:
                watchdog_reason = "phase_inactivity"
            if watchdog_reason is not None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(1.0)
        worker_exit = process.wait()
        log_stream.flush()
        os.fsync(log_stream.fileno())
    elapsed = time.monotonic() - start
    _marker("supervisor", "worker_process", "exit", {"exit_code": worker_exit, "elapsed_s": elapsed, "watchdog": watchdog_reason})

    stable_paths = [
        path
        for role in DECISION_ROWS
        for timing in ("before", "after")
        for path in CAPTURE_PATHS[role][timing].values()
    ]
    postprocess_errors: list[dict[str, Any]] = []

    def record_postprocess_error(stage: str, error: BaseException) -> None:
        postprocess_errors.append(
            {
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )

    stability: list[dict[str, Any]] = []
    if worker_exit == 0:
        try:
            stability = [_stable_file_report(path) for path in stable_paths]
        except Exception as error:
            record_postprocess_error("capture_stability", error)
    worker: dict[str, Any] | None = None
    if WORKER_SUMMARY_PATH.is_file():
        try:
            worker = _json(WORKER_SUMMARY_PATH)
        except Exception as error:
            record_postprocess_error("worker_summary_read", error)
    video: dict[str, Any] | None = None
    if worker is not None:
        try:
            video = _verify_video()
        except Exception as error:
            record_postprocess_error("video_verification", error)
    sync: dict[str, Any] | None = None
    rerun: dict[str, Any] | None = None
    private: dict[str, Any] | None = None
    if worker is not None:
        try:
            sync, private = _build_sync_report(worker)
        except Exception as error:
            record_postprocess_error("fabric_sync_report", error)
        for storyboard_path, view in (
            (PRIMARY_STORYBOARD_PATH, "primary"),
            (OPPOSITE_STORYBOARD_PATH, "opposite"),
        ):
            try:
                _build_storyboard(storyboard_path, view)
            except Exception as error:
                record_postprocess_error(f"storyboard_{view}", error)
        if sync is not None:
            try:
                _build_beginner_sheet(sync)
            except Exception as error:
                record_postprocess_error("beginner_sheet", error)
        if sync is not None and private is not None and BEGINNER_SHEET_PATH.is_file():
            try:
                rerun = _write_rerun(sync, private, worker)
            except Exception as error:
                record_postprocess_error("rerun", error)
    worker_phase: dict[str, Any] = {"pass": False}
    if WORKER_PHASE_PATH.is_file():
        try:
            worker_phase = _phase_contract(WORKER_PHASE_PATH, "worker")
        except Exception as error:
            record_postprocess_error("worker_phase_contract", error)
    if postprocess_errors:
        _write_json_x(
            POSTPROCESS_EXCEPTION_PATH,
            {
                "artifact": "D363_SUPERVISOR_POSTPROCESS_EXCEPTION_V1",
                "case": CASE,
                "utc": _utc_now(),
                "automatic_retry": False,
                "errors": postprocess_errors,
                "pass": False,
            },
        )
    # Supervisor phase contract is evaluated after adding this summary marker below; current required rows already exist.
    manifest_after = _d362_manifest()
    resource_summary = {
        "samples": len(telemetry),
        "gpu_used_mib_max": max((row["gpu"].get("memory_used_mib") or 0 for row in telemetry), default=None),
        "gpu_free_mib_min": min((row["gpu"].get("memory_free_mib") or 10**9 for row in telemetry), default=None),
        "gpu_utilization_percent_max": max((row["gpu"].get("utilization_gpu_percent") or 0 for row in telemetry), default=None),
        "worker_rss_bytes_max": max((row.get("worker_rss_bytes", 0) for row in telemetry), default=None),
    }
    post_checks = {
        "worker_exit_zero": worker_exit == 0,
        "watchdog_not_triggered": watchdog_reason is None,
        "worker_summary_pass": worker is not None and worker.get("pass") is True,
        "all_16_png_stable_after_close": len(stability) == 16 and all(row["stable"] for row in stability),
        "exact_1080_video_pass": video is not None and video.get("pass") is True,
        "fabric_sync_pass": sync is not None and sync.get("pass") is True,
        "rerun_pass": rerun is not None and rerun.get("pass") is True,
        "worker_phase_pass": worker_phase.get("pass") is True,
        "postprocess_exception_absent": not postprocess_errors,
        "d362_manifest_unchanged": manifest_after == prereg["d362_manifest_before"],
        "d363_d362_inode_disjoint": _d363_d362_inode_disjoint(),
        "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "input_hashes_unchanged": _input_hashes() == prereg["input_hashes"],
    }
    supervisor = {
        "artifact": "D363_SUPERVISOR_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "command": command,
        "elapsed_seconds": elapsed,
        "worker_exit_code": worker_exit,
        "watchdog_reason": watchdog_reason,
        "automatic_retry": False,
        "telemetry": telemetry,
        "resource_summary": resource_summary,
        "stable_capture_reports": stability,
        "worker_phase_contract": worker_phase,
        "postprocess_errors": postprocess_errors,
        "video_report": video,
        "sync_report": sync,
        "rerun": rerun,
        "d362_manifest_after": manifest_after,
        "post_checks": post_checks,
        "controlled_physics_steps": worker.get("controlled_physics_steps") if worker else None,
        "q5_science_sample_count": worker.get("q5_science_sample_count") if worker else None,
        "q5_target_update_count": worker.get("q5_target_update_count") if worker else None,
        "contact_query_count": worker.get("contact_query_count") if worker else None,
        "g0a_pass": False,
        "pass": all(post_checks.values()),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _marker("supervisor", "supervisor_summary", "complete", {"pass": supervisor["pass"]})
    try:
        supervisor_phase = _phase_contract(SUPERVISOR_PHASE_PATH, "supervisor")
    except Exception as error:
        record_postprocess_error("supervisor_phase_contract", error)
        supervisor_phase = {"pass": False, "error": f"{type(error).__name__}: {error}"}
        if not POSTPROCESS_EXCEPTION_PATH.exists():
            _write_json_x(
                POSTPROCESS_EXCEPTION_PATH,
                {
                    "artifact": "D363_SUPERVISOR_POSTPROCESS_EXCEPTION_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "automatic_retry": False,
                    "errors": postprocess_errors,
                    "pass": False,
                },
            )
    bound_artifact_hashes = {
        _rel(path): _sha(path)
        for path in sorted(OUT_DIR.rglob("*"))
        if path.is_file()
        and path not in {AUTOMATED_PATH, MANUAL_PATH, COMPLETION_PATH}
    }
    automated_checks = {
        "supervisor_pass": supervisor["pass"],
        "supervisor_phase_pass": supervisor_phase["pass"],
        "worker_and_reports_hash_bound": all(
            path.is_file()
            for path in (WORKER_SUMMARY_PATH, VIDEO_REPORT_PATH, SYNC_REPORT_PATH, RERUN_VALIDATION_PATH)
        ),
        "controlled_physics_zero": supervisor["controlled_physics_steps"] == 0,
        "q5_science_and_target_zero": supervisor["q5_science_sample_count"] == 0 and supervisor["q5_target_update_count"] == 0,
        "contact_query_zero": supervisor["contact_query_count"] == 0,
        "d362_immutable": _d362_manifest() == prereg["d362_manifest_before"],
        "bound_artifacts_nonempty": bool(bound_artifact_hashes)
        and all(bool(digest) for digest in bound_artifact_hashes.values()),
    }
    automated = {
        "artifact": "D363_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "supervisor_path": _rel(SUPERVISOR_PATH),
        "supervisor_sha256": _sha(SUPERVISOR_PATH),
        "worker_summary_sha256": _sha(WORKER_SUMMARY_PATH) if WORKER_SUMMARY_PATH.is_file() else None,
        "video_report_sha256": _sha(VIDEO_REPORT_PATH) if VIDEO_REPORT_PATH.is_file() else None,
        "sync_report_sha256": _sha(SYNC_REPORT_PATH) if SYNC_REPORT_PATH.is_file() else None,
        "rerun_validation_sha256": _sha(RERUN_VALIDATION_PATH) if RERUN_VALIDATION_PATH.is_file() else None,
        "bound_artifact_hashes": bound_artifact_hashes,
        "bound_artifact_inventory": sorted(bound_artifact_hashes),
        "supervisor_phase_contract": supervisor_phase,
        "checks": automated_checks,
        "manual_visual_inspection_pending": True,
        "completion_pending": True,
        "physical_science_recomputed": False,
        "inherited_d362_physical_subverdict": "D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED",
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "pass_before_manual": all(automated_checks.values()),
    }
    _write_json_x(AUTOMATED_PATH, automated)
    print(json.dumps({"stage": "run", "pass_before_manual": automated["pass_before_manual"], "worker_exit": worker_exit}, ensure_ascii=False))
    return 0 if automated["pass_before_manual"] else 2


def _all_required_visual_paths() -> list[str]:
    paths = [
        *[path for role in DECISION_ROWS for timing in ("before", "after") for path in CAPTURE_PATHS[role][timing].values()],
        VIDEO_PATH,
        VIDEO_STORYBOARD_PATH,
        PRIMARY_STORYBOARD_PATH,
        OPPOSITE_STORYBOARD_PATH,
        BEGINNER_SHEET_PATH,
        RERUN_PNG_PATH,
    ]
    return sorted(_rel(path) for path in paths)


def _finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D363 completion overwrite forbidden")
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    worker = _json(WORKER_SUMMARY_PATH)
    video_report = _json(VIDEO_REPORT_PATH)
    expected_paths = _all_required_visual_paths()
    manual_checks = {
        "artifact_exact": manual.get("artifact") == "D363_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "automated_sha_exact": manual.get("automated_summary_sha256") == _sha(AUTOMATED_PATH),
        "all_required_paths_exact": sorted(manual.get("inspected_paths", [])) == expected_paths,
        "all_path_hashes_exact": all(
            Path(REPO / path).is_file() and manual.get("inspected_sha256", {}).get(path) == _sha(REPO / path)
            for path in expected_paths
        ),
        "precommand_upright_seen": manual.get("precommand_upright_seen") is True,
        "final_moved_and_toppled_seen": manual.get("final_moved_and_toppled_seen") is True,
        "d362_stale_vs_d363_final_difference_seen": manual.get("d362_stale_vs_d363_final_difference_seen") is True,
        "before_forward_classification_recorded": manual.get("final_before_forward_classification") in {"stale_reproduced", "already_synced_or_visually_ambiguous"},
        "video_representative_frames_inspected": manual.get("video_representative_frames_inspected") is True,
        "full_video_playback_inspected": manual.get("full_video_playback_inspected") is True,
        "full_video_playback_sha_exact": manual.get("full_video_playback_sha256")
        == _sha(VIDEO_PATH),
        "full_video_playback_duration_sufficient": isinstance(
            manual.get("full_video_playback_elapsed_s"), (int, float)
        )
        and float(manual["full_video_playback_elapsed_s"])
        >= float(video_report.get("duration_seconds", math.inf)) - 0.5,
        "full_video_first_last_rows_seen": manual.get("full_video_first_source_row_seen") == 0
        and manual.get("full_video_last_source_row_seen") == 499,
        "full_video_observation_recorded_ko": len(
            str(manual.get("full_video_observation_ko", "")).strip()
        )
        >= 20,
        "video_labels_legible": manual.get("video_labels_legible") is True,
        "storyboards_no_text_overlap": manual.get("storyboards_no_text_overlap") is True,
        "rerun_actual_images_visible": manual.get("rerun_actual_images_visible") is True,
        "rerun_observation_recorded_ko": len(
            str(manual.get("rerun_observation_ko", "")).strip()
        )
        >= 20,
        "scientific_scope_limit_recorded": manual.get("no_new_contact_cap_rim_grasp_science_claim") is True,
        "manual_pass": manual.get("pass") is True,
    }
    bound_hashes = automated.get("bound_artifact_hashes", {})
    current_precompletion_inventory = sorted(
        _rel(path)
        for path in OUT_DIR.rglob("*")
        if path.is_file() and path != COMPLETION_PATH
    )
    expected_precompletion_inventory = sorted(
        [*bound_hashes, _rel(AUTOMATED_PATH), _rel(MANUAL_PATH)]
    )
    precompletion_artifact_hashes = {
        **bound_hashes,
        _rel(AUTOMATED_PATH): _sha(AUTOMATED_PATH),
        _rel(MANUAL_PATH): _sha(MANUAL_PATH),
    }
    precompletion_hash_manifest_sha256 = hashlib.sha256(
        json.dumps(
            precompletion_artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    integrity_checks = {
        "automated_pass_before_manual": automated.get("pass_before_manual") is True,
        "manual_checks": all(manual_checks.values()),
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "d362_manifest_exact": _d362_manifest() == prereg["d362_manifest_before"],
        "d363_d362_inode_disjoint": _d363_d362_inode_disjoint(),
        "d334_sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "automated_supervisor_hash_exact": automated.get("supervisor_sha256") == _sha(SUPERVISOR_PATH),
        "automated_worker_hash_exact": automated.get("worker_summary_sha256") == _sha(WORKER_SUMMARY_PATH),
        "automated_video_report_hash_exact": automated.get("video_report_sha256") == _sha(VIDEO_REPORT_PATH),
        "automated_sync_report_hash_exact": automated.get("sync_report_sha256") == _sha(SYNC_REPORT_PATH),
        "automated_rerun_validation_hash_exact": automated.get("rerun_validation_sha256") == _sha(RERUN_VALIDATION_PATH),
        "all_bound_artifact_hashes_exact": bool(bound_hashes)
        and all((REPO / path).is_file() and _sha(REPO / path) == digest for path, digest in bound_hashes.items()),
        "precompletion_inventory_exact": current_precompletion_inventory == expected_precompletion_inventory,
        "precompletion_hash_map_keys_exact": sorted(precompletion_artifact_hashes)
        == current_precompletion_inventory,
        "worker_registered_counters_exact": worker.get("controlled_physics_steps") == 0
        and worker.get("q5_science_sample_count") == 0
        and worker.get("q5_target_update_count") == 0
        and worker.get("contact_query_count") == 0
        and worker.get("explicit_forward_count") == 4,
        "video_report_pass": _json(VIDEO_REPORT_PATH).get("pass") is True,
        "sync_report_pass": _json(SYNC_REPORT_PATH).get("pass") is True,
        "rerun_validation_pass": _json(RERUN_VALIDATION_PATH).get("pass") is True,
    }
    completion = {
        "artifact": "D363_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "final_verdict": "D363_EXACT_1080_REPLAY_AND_ZERO_STEP_FABRIC_CAPTURE_CATEGORY_SYNC_COMPLETE"
        if all(integrity_checks.values())
        else "D363_OBSERVABILITY_OR_INTEGRITY_FAIL_STOP",
        "manual_checks": manual_checks,
        "integrity_checks": integrity_checks,
        "controlled_physics_steps": worker.get("controlled_physics_steps"),
        "q5_science_sample_count": worker.get("q5_science_sample_count"),
        "q5_target_update_count": worker.get("q5_target_update_count"),
        "contact_query_count": worker.get("contact_query_count"),
        "explicit_forward_count": worker.get("explicit_forward_count"),
        "counter_semantics": "registered D363 explicit-call counters; reset-internal transition excluded",
        "physical_science_recomputed": False,
        "inherited_d362_physical_subverdict": "D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED",
        "cap_rim_science": None,
        "target_ik_repair_justification": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "automated_summary_sha256": _sha(AUTOMATED_PATH),
        "manual_visual_inspection_sha256": _sha(MANUAL_PATH),
        "precompletion_artifact_hashes": precompletion_artifact_hashes,
        "precompletion_hash_manifest_sha256": precompletion_hash_manifest_sha256,
        "d362_manifest_final": _d362_manifest(),
        "output_inventory": sorted([*current_precompletion_inventory, _rel(COMPLETION_PATH)]),
        "pass": all(integrity_checks.values()),
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
        raise RuntimeError("D363 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D363 seed drift")
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
