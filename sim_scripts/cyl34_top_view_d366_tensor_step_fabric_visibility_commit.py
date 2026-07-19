#!/usr/bin/env python3
"""D366: test the one-step tensor-to-Fabric visibility commit boundary.

This forward-only observability control inherits the frozen D365 environment and
measurement layers.  With the timeline PLAYING it writes the frozen D362 row-499
cylinder pose once, performs exactly one ``sim.step(render=False)``, then exactly
one public ``SimulationContext.forward()``.  It compares the post-step PhysX pose
with same-time Fabric hierarchy/render-cache state and guarded Hydra pixels.  It
does not sample q5, write a q5 target, or query contact.
"""
from __future__ import annotations

import argparse
import ast
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

# Safe before AppLauncher: this module's top level imports no pxr/omni runtime.
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d362_current_pose_capacity_prefix_integrated_physx_contact_motion as d362,
    cyl34_top_view_d365_hierarchy_current_render_cache_propagation_localization as d365,
)


CASE = "g0a_d366"
CASE_NAME = "tensor_step_fabric_visibility_commit"
NEW_VARIABLES = [
    "timeline_play_tensor_write_contract",
    "one_controlled_physics_step_before_inherited_public_forward",
]
BASE_GIT = "ce99a2cc24bd7a3112418739edc1b4ce1c6ef8c9"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
RERUN_VERSION = "0.34.1"
DISPLAY = ":1"
SEED = 33201
VIEWPORT_SIZE = [1280, 720]
CAMERA_EYES = {
    "primary": list(d362.CAMERA_EYE),
    "opposite": list(d362.OPPOSITE_CAMERA_EYE),
}
CAMERA_TARGET = list(d362.CAMERA_TARGET)
PLAY_SIMULATIONS_SETTING = d362.PLAY_SIMULATIONS_SETTING
ROOT_PRIM_PATH = "/World/envs/env_0/Sponge"
MESH_PRIM_PATH = "/World/envs/env_0/Sponge/geometry/mesh"
TARGET_ROW_INDEX = 499
EXPECTED_GLOBAL_STEP = 500
POSITION_TOL_M = 1.0e-6
QUATERNION_TOL_DEG = 0.01
MATRIX_TOL = 1.0e-5
PHYSICS_DT_S = 0.005
TIME_TOL_S = 1.0e-9
MASK_HSV_LOW = np.asarray([15, 40, 80], dtype=np.uint8)
MASK_HSV_HIGH = np.asarray([45, 255, 255], dtype=np.uint8)
MASK_MIN_AREA_PX = 500
MATERIAL_CENTROID_DELTA_PX = 15.0
MATERIAL_AXIS_DELTA_DEG = 15.0
MATERIAL_IOU_MAX = 0.85
UPRIGHT_HW_MIN = 1.5
TOPPLED_WH_MIN = 1.15
MIN_GPU_FREE_MIB = 8192
MIN_RAM_AVAILABLE_BYTES = 8 * 1024**3
TOTAL_WATCHDOG_S = 600.0
INACTIVITY_WATCHDOG_S = 180.0
CAPTURE_TIMEOUT_S = 45.0
ISAACLAB_SIM_CONTEXT_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py"
)
ISAAC_CORE_SIM_CONTEXT_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py"
)
ISAAC_PHYSICS_CONTEXT_SOURCE = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py"
)
SOURCE_CONTRACT_HASHES = {
    str(ISAACLAB_SIM_CONTEXT_SOURCE): "340450726276d321c48b57de35f846c5a231c30a358b4922b5b7dbb8d42ec80e",
    str(ISAAC_CORE_SIM_CONTEXT_SOURCE): "ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028",
    str(ISAAC_PHYSICS_CONTEXT_SOURCE): "7486ec315d2525bdef6e1cef294d67d419511e3367e7f40001a8324e2b6a7751",
}

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d366"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260719_grasp_g0a_d366_tensor_step_fabric_visibility_commit.md"
D362_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d362"
D363_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d363"
D364_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d364"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
D362_TRACE = D362_DIR / "d362_physics_trace.json"
D362_WORKER_SUMMARY = D362_DIR / "d362_worker_summary.json"
D362_HARNESS = Path(d362.__file__).resolve()
D363_HARNESS = REPO / "sim_scripts/cyl34_top_view_d363_d362_trace_replay_1080_and_isaac_render_sync_repair.py"
D363_SESSION = REPO / "claudedocs/session_20260718_grasp_g0a_d363_trace_replay_1080_and_isaac_render_sync_repair.md"
D363_SYNC = D363_DIR / "d363_fabric_render_sync_report.json"
D363_COMPLETION = D363_DIR / "d363_completion_summary.json"
D364_HARNESS = REPO / "sim_scripts/cyl34_top_view_d364_paused_render_state_layer_localization.py"
D364_SESSION = REPO / "claudedocs/session_20260718_grasp_g0a_d364_paused_render_state_layer_localization.md"
D364_RUNTIME_SOURCE = D364_DIR / "d364_runtime_fabric_attestation.json"
D364_PREWRITE_COMPLETION = D364_DIR / "d364_prewrite_fail_completion.json"
D364_WORKER_EXCEPTION = D364_DIR / "d364_worker_exception.json"
D365_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d365"
D365_RUN_DIR = D365_DIR / "attempt2_host_access_prepare_repair"
D365_HARNESS = Path(d365.__file__).resolve()
D365_SESSION = REPO / "claudedocs/session_20260718_grasp_g0a_d365_hierarchy_current_render_cache_propagation_localization.md"
D365_RUNTIME_SOURCE = D365_RUN_DIR / "d365_runtime_fabric_attestation.json"
D365_REPORT = D365_RUN_DIR / "d365_state_layer_localization_report.json"
D365_WORKER_SUMMARY = D365_RUN_DIR / "d365_worker_summary.json"
D365_COMPLETION = D365_RUN_DIR / "d365_completion_summary.json"
D365_RRD = D365_RUN_DIR / "d365_state_layer_localization.rrd"
EXPECTED_INPUT_HASHES = {
    str(D362_TRACE.relative_to(REPO)): "9483146c4941e6518614c63acbf221128a564bafa7a9928d41e633ee6e4e2044",
    str(D362_WORKER_SUMMARY.relative_to(REPO)): "10f7bd39b67f9bd254827fab580396c9a8089304f904c20dc3efd908296b217d",
    str(D362_HARNESS.relative_to(REPO)): "80fb5f47ec01de67c23b11f92fc6b46f3bff7063fc9474436a7863cf1c9df11c",
    str(D363_HARNESS.relative_to(REPO)): "63b307137405b2a343af88e046e992ef4ee996aff3bc467e2bf58390e4e18a14",
    str(D363_SESSION.relative_to(REPO)): "888bff3bd281ad75922822757595bd9d3340f3a3eeb43a67ab9af34f897e9fe3",
    str(D363_SYNC.relative_to(REPO)): "4cd5dd401b4eaea687549c5f5279b71e0f7fb0ad67a70f4d555f94b793653b3c",
    str(D363_COMPLETION.relative_to(REPO)): "e55a155b814dabdb90ce6b219c36318431f695331342624d2d2780d7b7b4f078",
    str(D364_HARNESS.relative_to(REPO)): "4377203b9756b503b4f8a80f955db93ed9301edaa84e3113d81cfdee4c558925",
    str(D364_SESSION.relative_to(REPO)): "e931a37ced51a05c793574e0f9940534e5e2d0b0e2ca9489390c90c4bb203e69",
    str(D364_RUNTIME_SOURCE.relative_to(REPO)): "777ea962cea18fd9d9c2bbcf2306f70adf4a7f068c0439f2be4941a546703eb4",
    str(D364_PREWRITE_COMPLETION.relative_to(REPO)): "845f3b392b986f4354618f9191c953ee3919e6917ed1db1f382e7f6654810211",
    str(D364_WORKER_EXCEPTION.relative_to(REPO)): "f1a9d5334586bda31d197440ad6503dddc6f04b43bdbca7d4a2747318ed4cd6b",
    str(D365_HARNESS.relative_to(REPO)): "719011a6171e27ae6b759903ef060397c1fad10c5c822d4c24a207ccdc59d834",
    str(D365_SESSION.relative_to(REPO)): "a3e837070cfab48d1203e2a6f8c72929fb4f0ee55d20cb7e2514a72cd30d1f5f",
    str(D365_RUNTIME_SOURCE.relative_to(REPO)): "b03c724f21b70a0d85f8212efdb46c54222588be79eac03ff181a8b6a2a431ee",
    str(D365_REPORT.relative_to(REPO)): "b82065bf89930f80d2c6e4a38bdaf9a323b2604b25efb7cb840b29b0ac4c5420",
    str(D365_WORKER_SUMMARY.relative_to(REPO)): "3babf239358f48ae5d2edd2124e3bcdb29d76ef5464a82b958c9fddd3a2e8c2e",
    str(D365_COMPLETION.relative_to(REPO)): "efb2ece5bb30fa987bfa8a6ed229d282efdecff6c359beaa8034448ff1c3752d",
    str(D365_RRD.relative_to(REPO)): "73a2a1d5954e6dfadfb7e562ea2ac4de8dcd80413e94b58db8abab069378a056",
}

PREREG_PATH = OUT_DIR / "d366_preregistration.json"
PREPARE_PATH = OUT_DIR / "d366_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d366_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d366_worker_preflight.json"
RUNTIME_PATH = OUT_DIR / "d366_runtime_step_fabric_attestation.json"
WORKER_PHASE_PATH = OUT_DIR / "d366_worker_phase_markers.jsonl"
SUPERVISOR_PHASE_PATH = OUT_DIR / "d366_supervisor_phase_markers.jsonl"
LAYER_JOURNAL_PATH = OUT_DIR / "d366_layer_readback_journal.jsonl"
LAYER_AUDIT_PATH = OUT_DIR / "d366_layer_readback_journal_audit.json"
WORKER_LOG_PATH = OUT_DIR / "d366_worker_stdout_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d366_worker_summary.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d366_worker_exception.json"
REPORT_PATH = OUT_DIR / "d366_tensor_step_visibility_report.json"
SHEET_PATH = OUT_DIR / "d366_tensor_step_visibility_sheet_ko.png"
RRD_PATH = OUT_DIR / "d366_tensor_step_visibility.rrd"
RBL_PATH = OUT_DIR / "d366_tensor_step_visibility.rbl"
RERUN_PNG_PATH = OUT_DIR / "d366_tensor_step_visibility_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d366_rerun_validation.json"
SUPERVISOR_PATH = OUT_DIR / "d366_supervisor_summary.json"
AUTOMATED_PATH = OUT_DIR / "d366_automated_summary.json"
MANUAL_PATH = OUT_DIR / "d366_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d366_completion_summary.json"
POSTPROCESS_EXCEPTION_PATH = OUT_DIR / "d366_supervisor_postprocess_exception.json"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

WORKER_TOKEN_ENV = "D366_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D366_SUPERVISOR_PID"
CAPTURE_PATHS = {
    phase: {
        view: OUT_DIR / f"d366_{phase}_{view}_actual_isaac.png"
        for view in CAMERA_EYES
    }
    for phase in ("baseline", "post_step_forward")
}
CHECKPOINT_LABELS = (
    "baseline_pre_capture",
    "baseline_post_capture",
    "post_play_pre_write",
    "post_write_pre_step",
    "post_step_pre_forward",
    "post_forward_pre_pause",
    "post_pause_pre_capture",
    "post_step_forward_terminal",
)

_WORKER_SEQUENCE = 0
_SUPERVISOR_SEQUENCE = 0
_LAYER_SEQUENCE = 0
_LAYER_PREV_SHA = "0" * 64
_DISPLAY_STATE_WRITE_COUNT = 0
_EXPLICIT_FORWARD_COUNT = 0
_DISPLAY_STATE_WRITE_RETURNED = False
_EXPLICIT_FORWARD_RETURNED = False
_CONTROLLED_STEP_CALL_COUNT = 0
_CONTROLLED_STEP_RETURNED = False
_CONTROLLED_PHYSICS_STEPS: int | None = None
_PHYSICS_CALLBACK_COUNT = 0
_PHYSICS_CALLBACK_DTS: list[float] = []
_CALLBACK_NAME = "d366_exact_one_step_guard"
_Q5_SCIENCE_SAMPLE_COUNT = 0
_Q5_TARGET_UPDATE_COUNT = 0
_CONTACT_QUERY_COUNT = 0


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_x(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
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


def _marker(owner: str, phase: str, event: str, details: dict[str, Any] | None = None) -> None:
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
            "owner": owner,
            "phase": phase,
            "event": event,
            "details": details or {},
        },
    )


def _journal_layer(label: str, payload: dict[str, Any]) -> None:
    global _LAYER_SEQUENCE, _LAYER_PREV_SHA
    _LAYER_SEQUENCE += 1
    body = {
        "sequence": _LAYER_SEQUENCE,
        "label": label,
        "utc": _utc_now(),
        "previous_sha256": _LAYER_PREV_SHA,
        "payload": payload,
    }
    body["record_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    _append_jsonl(LAYER_JOURNAL_PATH, body)
    _LAYER_PREV_SHA = body["record_sha256"]


def _run_text(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()


def _git_head(ref: str = "HEAD") -> str:
    return _run_text(["git", "rev-parse", ref])


def _git_status() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_cached_patch() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
        "byte_count": len(result.stdout),
        "name_status": names,
    }


def _status_scope_ok(rows: list[str]) -> bool:
    d365_continuity = {
        _rel(D365_RUN_DIR / name)
        for name in (
            "d365_baseline_opposite_actual_isaac.png",
            "d365_baseline_primary_actual_isaac.png",
            "d365_post_forward_opposite_actual_isaac.png",
            "d365_post_forward_primary_actual_isaac.png",
            "d365_post_write_no_forward_opposite_actual_isaac.png",
            "d365_post_write_no_forward_primary_actual_isaac.png",
            "d365_state_layer_localization_rerun.png",
            "d365_state_layer_localization_sheet_ko.png",
            "d365_worker_stdout_stderr.log",
        )
    }
    allowed_exact = {
        "START_HERE.md",
        "claudedocs/DECISIONS.md",
        "claudedocs/EXPERIMENT_LEDGER.md",
        _rel(HARNESS),
        _rel(SESSION_DOC),
        *d365_continuity,
    }
    allowed_prefix = f"claudedocs/runtime_logs/grasp_track/{CASE}/"
    for row in rows:
        if " -> " in row:
            return False
        path = row[3:].strip()
        if path not in allowed_exact and not path.startswith(allowed_prefix):
            return False
    return True


def _tree_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item.relative_to(root))):
        rows.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": _sha(path)})
    return {
        "root": _rel(root),
        "file_count": len(rows),
        "rows": rows,
        "filename_size_digest": hashlib.sha256(
            _canonical_bytes([[row["path"], row["size"]] for row in rows])
        ).hexdigest(),
        "file_sha_digest": hashlib.sha256(
            _canonical_bytes([[row["path"], row["sha256"]] for row in rows])
        ).hexdigest(),
    }


def _sidecar_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(D334_SIDECAR)): _sha(path)
        for path in sorted(D334_SIDECAR.rglob("*"))
        if path.is_file()
    }


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in (
        D362_TRACE,
        D362_WORKER_SUMMARY,
        D362_HARNESS,
        D363_HARNESS,
        D363_SESSION,
        D363_SYNC,
        D363_COMPLETION,
        D364_HARNESS,
        D364_SESSION,
        D364_RUNTIME_SOURCE,
        D364_PREWRITE_COMPLETION,
        D364_WORKER_EXCEPTION,
        D365_HARNESS,
        D365_SESSION,
        D365_RUNTIME_SOURCE,
        D365_REPORT,
        D365_WORKER_SUMMARY,
        D365_COMPLETION,
        D365_RRD,
    )}


def _d365_completion_evidence_audit() -> dict[str, Any]:
    completion = _json(D365_COMPLETION)
    report = _json(D365_REPORT)
    worker = _json(D365_WORKER_SUMMARY)
    checks = {
        "completion_pass": completion.get("pass") is True,
        "verdict_exact": completion.get("final_verdict")
        == "D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED",
        "registered_zero_step_exact": worker.get("display_state_write_count") == 1
        and worker.get("explicit_forward_count") == 1
        and worker.get("controlled_physics_steps") == 0,
        "q5_contact_zero": worker.get("q5_science_sample_count") == 0
        and worker.get("q5_target_update_count") == 0
        and worker.get("contact_query_count") == 0,
        "terminal_classes_exact": report.get("terminal_classes", {}).get("physx") == "TARGET"
        and report.get("terminal_classes", {}).get("root_current") == "BASELINE"
        and report.get("terminal_classes", {}).get("mesh_current") == "BASELINE"
        and report.get("terminal_classes", {}).get("mesh_cached") == "BASELINE"
        and report.get("terminal_classes", {}).get("hydra") == "BASELINE",
        "core_hashes_exact": _sha(D365_HARNESS) == EXPECTED_INPUT_HASHES[_rel(D365_HARNESS)]
        and _sha(D365_SESSION) == EXPECTED_INPUT_HASHES[_rel(D365_SESSION)]
        and _sha(D365_RUNTIME_SOURCE) == EXPECTED_INPUT_HASHES[_rel(D365_RUNTIME_SOURCE)]
        and _sha(D365_REPORT) == EXPECTED_INPUT_HASHES[_rel(D365_REPORT)]
        and _sha(D365_WORKER_SUMMARY) == EXPECTED_INPUT_HASHES[_rel(D365_WORKER_SUMMARY)]
        and _sha(D365_COMPLETION) == EXPECTED_INPUT_HASHES[_rel(D365_COMPLETION)]
        and _sha(D365_RRD) == EXPECTED_INPUT_HASHES[_rel(D365_RRD)],
    }
    return {
        "completion_path": _rel(D365_COMPLETION),
        "report_path": _rel(D365_REPORT),
        "worker_summary_path": _rel(D365_WORKER_SUMMARY),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _installed_source_contract_audit() -> dict[str, Any]:
    lab_text = ISAACLAB_SIM_CONTEXT_SOURCE.read_text(encoding="utf-8")
    core_text = ISAAC_CORE_SIM_CONTEXT_SOURCE.read_text(encoding="utf-8")
    physics_text = ISAAC_PHYSICS_CONTEXT_SOURCE.read_text(encoding="utf-8")
    hashes = {
        str(path): _sha(path)
        for path in (
            ISAACLAB_SIM_CONTEXT_SOURCE,
            ISAAC_CORE_SIM_CONTEXT_SOURCE,
            ISAAC_PHYSICS_CONTEXT_SOURCE,
        )
    }
    checks = {
        "source_hashes_exact": hashes == SOURCE_CONTRACT_HASHES,
        "lab_step_delegates_render_argument": "super().step(render=render)" in lab_text,
        "lab_forward_calls_bound_fabric": "self._update_fabric(0.0, 0.0)" in lab_text,
        "lab_render_would_call_forward": "self.forward()" in lab_text,
        "core_render_false_steps_only_while_playing": "if self.is_playing():\n                self._physics_context._step(current_time=self.current_time, update_fabric=update_fabric)" in core_text,
        "core_default_update_fabric_false": "def step(self, render: bool = True, update_fabric: bool = False)" in core_text,
        "physics_step_simulate_then_fetch": "self._physx_sim_interface.simulate(self.get_physics_dt(), current_time)\n        self._physx_sim_interface.fetch_results()" in physics_text,
        "public_callback_api_present": "def add_physics_callback(" in core_text
        and "def remove_physics_callback(" in core_text
        and "def physics_callback_exists(" in core_text,
    }
    return {
        "paths_and_sha256": hashes,
        "semantics": {
            "controlled_step": "PLAY + sim.step(render=False) -> simulate + fetch_results, update_fabric default False",
            "public_forward": "PLAY -> articulation kinematic update + bound Fabric force_update",
            "render_forbidden": "IsaacLab render() would invoke public forward again",
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
def _d364_completion_evidence_audit() -> dict[str, Any]:
    completion = _json(D364_PREWRITE_COMPLETION)
    expected = completion.get("evidence_sha256", {})
    rows = {
        name: {
            "expected_sha256": digest,
            "actual_sha256": _sha(D364_DIR / name) if (D364_DIR / name).is_file() else None,
            "exists": (D364_DIR / name).is_file(),
        }
        for name, digest in sorted(expected.items())
    }
    checks = {
        "completion_artifact_exact": completion.get("artifact") == "D364_PREWRITE_FAIL_COMPLETION_V1",
        "evidence_map_nonempty": bool(rows),
        "all_evidence_hashes_exact": bool(rows)
        and all(row["actual_sha256"] == row["expected_sha256"] for row in rows.values()),
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


def _gpu_snapshot() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    result: dict[str, Any] = {
        "nvidia_smi_returncode": query.returncode,
        "nvidia_smi_stderr": query.stderr.strip(),
        "ram_available_bytes": psutil.virtual_memory().available,
    }
    if query.returncode == 0 and query.stdout.strip():
        fields = [item.strip() for item in query.stdout.splitlines()[0].split(",")]
        if len(fields) == 5:
            result.update(
                {
                    "gpu_name": fields[0],
                    "memory_total_mib": int(fields[1]),
                    "memory_used_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                    "utilization_gpu_percent": int(fields[4]),
                }
            )
    return result


def _pose_error(observed: list[float], expected: list[float]) -> dict[str, Any]:
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    pos_max_abs = float(np.max(np.abs(obs[:3] - exp[:3])))
    q_obs = obs[3:7] / max(float(np.linalg.norm(obs[3:7])), 1.0e-15)
    q_exp = exp[3:7] / max(float(np.linalg.norm(exp[3:7])), 1.0e-15)
    dot = float(np.clip(abs(np.dot(q_obs, q_exp)), 0.0, 1.0))
    angle_deg = float(math.degrees(2.0 * math.acos(dot)))
    return {
        "position_max_abs_error_m": pos_max_abs,
        "quaternion_angular_error_deg": angle_deg,
        "match": pos_max_abs <= POSITION_TOL_M and angle_deg <= QUATERNION_TOL_DEG,
    }


def _classify_pose(
    observed: list[float] | None,
    baseline: list[float],
    commanded: list[float],
    post_step: list[float],
) -> dict[str, Any]:
    if observed is None:
        return {"class": "UNAVAILABLE", "baseline": None, "commanded": None, "post_step": None}
    errors = {
        "baseline": _pose_error(observed, baseline),
        "commanded": _pose_error(observed, commanded),
        "post_step": _pose_error(observed, post_step),
    }
    matches = {name for name, error in errors.items() if error["match"]}
    if "post_step" in matches and "commanded" in matches:
        label = "POST_STEP_EQ_COMMANDED"
    elif "post_step" in matches:
        label = "POST_STEP"
    elif "commanded" in matches:
        label = "COMMANDED"
    elif "baseline" in matches:
        label = "BASELINE"
    else:
        label = "OTHER"
    return {"class": label, **errors}


def _classify_matrix(
    observed: list[list[float]] | None,
    baseline: list[list[float]],
    commanded: list[list[float]],
    post_step: list[list[float]],
) -> dict[str, Any]:
    if observed is None:
        return {
            "class": "UNAVAILABLE",
            "baseline_max_abs": None,
            "commanded_max_abs": None,
            "post_step_max_abs": None,
        }
    obs = np.asarray(observed, dtype=np.float64)
    references = {
        "baseline": np.asarray(baseline, dtype=np.float64),
        "commanded": np.asarray(commanded, dtype=np.float64),
        "post_step": np.asarray(post_step, dtype=np.float64),
    }
    errors = {name: float(np.max(np.abs(obs - value))) for name, value in references.items()}
    matches = {name for name, error in errors.items() if error <= MATRIX_TOL}
    if "post_step" in matches and "commanded" in matches:
        label = "POST_STEP_EQ_COMMANDED"
    elif "post_step" in matches:
        label = "POST_STEP"
    elif "commanded" in matches:
        label = "COMMANDED"
    elif "baseline" in matches:
        label = "BASELINE"
    else:
        label = "OTHER"
    return {
        "class": label,
        **{f"{name}_max_abs": error for name, error in errors.items()},
        **{f"{name}_match": name in matches for name in references},
    }


def _is_post_step(value: str) -> bool:
    return value in {"POST_STEP", "POST_STEP_EQ_COMMANDED"}


def _decision_fixture(
    root_current: str,
    mesh_current: str,
    mesh_cached: str,
    hydra: str,
    *,
    complete: bool = True,
    step_attested: bool = True,
) -> str:
    if not complete:
        return "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    if not step_attested:
        return "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    allowed = {"BASELINE", "COMMANDED", "POST_STEP", "POST_STEP_EQ_COMMANDED"}
    linear = (root_current, mesh_current, mesh_cached, hydra)
    if any(value not in allowed for value in linear):
        return "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    seen_not_post = False
    for value in linear:
        if not _is_post_step(value):
            seen_not_post = True
        elif seen_not_post:
            return "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    if not _is_post_step(root_current):
        return "D366_POST_STEP_PHYSX_TO_FABRIC_NOT_PROPAGATED"
    if not _is_post_step(mesh_current):
        return "D366_FABRIC_ROOT_TO_RENDER_MESH_NOT_PROPAGATED"
    if not _is_post_step(mesh_cached):
        return "D366_RENDER_MESH_TO_CACHE_NOT_PROPAGATED"
    if not _is_post_step(hydra):
        return "D366_FABRIC_TO_HYDRA_NOT_PROPAGATED"
    return "D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE"


def _counter_contract(
    write: int,
    step: int,
    forward: int,
    callback: int,
    q5_sample: int,
    q5_target: int,
    contact: int,
) -> bool:
    return (write, step, forward, callback, q5_sample, q5_target, contact) == (1, 1, 1, 1, 0, 0, 0)


def _required_matrix_transition_changed(pairs: list[tuple[Any, Any]]) -> bool:
    for before, after in pairs:
        if (before is None) != (after is None):
            return True
        if before is not None and after is not None:
            if float(np.max(np.abs(np.asarray(before, dtype=np.float64) - np.asarray(after, dtype=np.float64)))) > MATRIX_TOL:
                return True
    return False


def _negative_controls() -> dict[str, Any]:
    cases = {
        "fabric_break": (("BASELINE", "BASELINE", "BASELINE", "BASELINE"), "D366_POST_STEP_PHYSX_TO_FABRIC_NOT_PROPAGATED"),
        "mesh_break": (("POST_STEP", "BASELINE", "BASELINE", "BASELINE"), "D366_FABRIC_ROOT_TO_RENDER_MESH_NOT_PROPAGATED"),
        "cache_break": (("POST_STEP", "POST_STEP", "BASELINE", "BASELINE"), "D366_RENDER_MESH_TO_CACHE_NOT_PROPAGATED"),
        "hydra_break": (("POST_STEP", "POST_STEP", "POST_STEP", "BASELINE"), "D366_FABRIC_TO_HYDRA_NOT_PROPAGATED"),
        "complete": (("POST_STEP", "POST_STEP", "POST_STEP", "POST_STEP"), "D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE"),
        "commanded_alias_complete": (("POST_STEP_EQ_COMMANDED",) * 4, "D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE"),
        "missing": (("POST_STEP",) * 4, "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP", {"complete": False}),
        "step_zero_or_two": (("POST_STEP",) * 4, "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP", {"step_attested": False}),
        "root_other": (("OTHER", "POST_STEP", "POST_STEP", "POST_STEP"), "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"),
        "downstream_ahead": (("BASELINE", "POST_STEP", "POST_STEP", "POST_STEP"), "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"),
    }
    results: dict[str, Any] = {}
    for name, row in cases.items():
        inputs = row[0]
        expected = row[1]
        kwargs = row[2] if len(row) > 2 else {}
        results[name] = {"observed": _decision_fixture(*inputs, **kwargs), "expected": expected}
    q = [0.9238795325, 0.0, 0.3826834324, 0.0]
    pose = [0.31, -0.02, 0.03, *q]
    q_flip = [0.31, -0.02, 0.03, *[-item for item in q]]
    q_swap = [0.31, -0.02, 0.03, q[1], q[2], q[3], q[0]]
    pose_controls = {
        "q_sign_equivalent": _pose_error(q_flip, pose)["match"] is True,
        "xyzw_wxyz_swap_rejected": _pose_error(q_swap, pose)["match"] is False,
        "translation_10mm_rejected": _pose_error([0.32, -0.02, 0.03, *q], pose)["match"] is False,
    }
    counter_controls = {
        "exact_counter_contract_accepts": _counter_contract(1, 1, 1, 1, 0, 0, 0),
        "write_0_rejected": not _counter_contract(0, 1, 1, 1, 0, 0, 0),
        "write_2_rejected": not _counter_contract(2, 1, 1, 1, 0, 0, 0),
        "step_0_rejected": not _counter_contract(1, 0, 1, 0, 0, 0, 0),
        "step_2_rejected": not _counter_contract(1, 2, 1, 2, 0, 0, 0),
        "forward_0_rejected": not _counter_contract(1, 1, 0, 1, 0, 0, 0),
        "forward_2_rejected": not _counter_contract(1, 1, 2, 1, 0, 0, 0),
        "q5_sample_nonzero_rejected": not _counter_contract(1, 1, 1, 1, 1, 0, 0),
        "q5_target_nonzero_rejected": not _counter_contract(1, 1, 1, 1, 0, 1, 0),
        "contact_nonzero_rejected": not _counter_contract(1, 1, 1, 1, 0, 0, 1),
    }
    baseline_authority = [0.30, 0.0, 0.033, 1.0, 0.0, 0.0, 0.0]
    commanded_authority = [0.36, 0.0, 0.017, 0.70710678, 0.0, 0.70710678, 0.0]
    poststep_authority = [0.361, 0.0, 0.0169, 0.70710678, 0.0, 0.70710678, 0.0]
    authority_controls = {
        "poststep_authority_selected": _classify_pose(
            poststep_authority, baseline_authority, commanded_authority, poststep_authority
        )["class"] == "POST_STEP",
        "commanded_as_false_poststep_rejected": not _is_post_step(
            _classify_pose(
                poststep_authority, baseline_authority, poststep_authority, commanded_authority
            )["class"]
        ),
    }
    clock_reference = {
        "custom_step_counter": 0,
        "simulation_clock": {"current_time": 0.01, "current_time_step_index": 2},
        "physics_callback_count": 0,
        "physics_callback_dts": [],
        "timeline_playing": True,
        "timeline_stopped": False,
        "timeline_time_s": 0.03,
    }
    getter_mutation = {**clock_reference, "simulation_clock": {"current_time": 0.015, "current_time_step_index": 3}}
    forward_mutation = {**clock_reference, "physics_callback_count": 1, "physics_callback_dts": [PHYSICS_DT_S]}
    pause_mutation = {**clock_reference, "custom_step_counter": 1}
    clock_controls = {
        "unchanged_clock_accepts": _physics_state_no_advance(clock_reference, dict(clock_reference)),
        "getter_clock_side_effect_rejected": not _physics_state_no_advance(clock_reference, getter_mutation),
        "forward_callback_side_effect_rejected": not _physics_state_no_advance(clock_reference, forward_mutation),
        "pause_custom_counter_side_effect_rejected": not _physics_state_no_advance(clock_reference, pause_mutation),
    }
    visual_controls = {
        "two_view_agreement_accepts": _decision_fixture(
            "POST_STEP", "POST_STEP", "POST_STEP", "POST_STEP", complete=len({"POST_STEP", "POST_STEP"}) == 1
        ) == "D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE",
        "two_view_disagreement_rejected": _decision_fixture(
            "POST_STEP", "POST_STEP", "POST_STEP", "OTHER", complete=len({"POST_STEP", "BASELINE"}) == 1
        ) == "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP",
    }
    fabric_getter_controls = {
        "unchanged_required_matrices_accept": not _required_matrix_transition_changed(
            [(np.eye(4), np.eye(4)), (np.eye(4), np.eye(4))]
        ),
        "presence_side_effect_rejected": _required_matrix_transition_changed(
            [(None, np.eye(4))]
        ),
        "matrix_side_effect_rejected": _required_matrix_transition_changed(
            [(np.eye(4), np.eye(4) + np.eye(4) * (MATRIX_TOL * 2.0))]
        ),
    }
    checks = {
        **{name: row["observed"] == row["expected"] for name, row in results.items()},
        **pose_controls,
        **counter_controls,
        **authority_controls,
        **clock_controls,
        **visual_controls,
        **fabric_getter_controls,
    }
    return {
        "decision_cases": results,
        "pose_controls": pose_controls,
        "counter_controls": counter_controls,
        "authority_controls": authority_controls,
        "clock_controls": clock_controls,
        "visual_controls": visual_controls,
        "fabric_getter_controls": fabric_getter_controls,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _static_source_audit() -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    call_names: list[str] = []
    selected_calls: list[ast.Call] = []
    attribute_names: list[str] = []
    audited_functions = {
        "_worker",
        "_write_target_pose_once",
        "_read_physx_tensor_view",
        "_capture_viewport",
        "_capture_phase",
        "_play_no_advance",
        "_pause_no_advance",
        "_physics_step_callback",
    }

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    selected_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in audited_functions
    ]
    for selected in selected_nodes:
        for node in ast.walk(selected):
            if isinstance(node, ast.Call):
                call_names.append(dotted(node.func))
                selected_calls.append(node)
            if isinstance(node, ast.Attribute):
                attribute_names.append(dotted(node))

    def count_suffix(suffix: str) -> int:
        return sum(name.endswith(suffix) for name in call_names)

    forbidden_suffixes = (
        ".simulate",
        ".fetch_results",
        ".render",
        ".scene.update",
        ".scene.write_data_to_sim",
        ".write_joint_state_to_sim",
        ".write_root_velocity_to_sim",
        ".set_joint_position_target",
        ".get_contact_force_matrix",
        ".get_net_contact_forces",
        ".update_world_xforms",
        ".SetWorldXformFromUsd",
        ".SetLocalXformFromUsd",
        ".CreateWorldPositionAttr",
        ".CreateWorldOrientationAttr",
        ".CreateWorldScaleAttr",
        "._update_fabric",
    )
    forbidden_hits = sorted(name for name in call_names if any(name.endswith(suffix) for suffix in forbidden_suffixes))
    forbidden_observation_tokens = (
        ".get_contact_data",
        ".contact_forces",
        ".net_forces_w",
        ".force_matrix_w",
        ".joint_pos",
        ".joint_vel",
        ".joint_acc",
    )
    forbidden_observation_hits = sorted(
        name for name in set(attribute_names)
        if any(token in name for token in forbidden_observation_tokens)
    )
    step_calls = [call for call in selected_calls if dotted(call.func).endswith(".sim.step")]
    step_render_false_ast = len(step_calls) == 1 and any(
        keyword.arg == "render"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in step_calls[0].keywords
    )
    step_update_fabric_default_ast = len(step_calls) == 1 and not any(
        keyword.arg == "update_fabric" and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        )
        for keyword in step_calls[0].keywords
    )
    checks = {
        "audited_function_set_exact": {node.name for node in selected_nodes} == audited_functions,
        "one_root_pose_write_site": count_suffix(".write_root_pose_to_sim") == 1,
        "one_controlled_render_false_step_site": count_suffix(".sim.step") == 1,
        "one_public_forward_site": count_suffix(".sim.forward") == 1,
        "one_independent_physx_getter_site": count_suffix(".root_physx_view.get_transforms") == 1,
        "one_physics_callback_add_site": count_suffix(".sim.add_physics_callback") == 1,
        "one_physics_callback_remove_site": count_suffix(".sim.remove_physics_callback") == 1,
        "no_forbidden_mutating_call_sites": not forbidden_hits,
        "no_q5_or_contact_observation_attributes": not forbidden_observation_hits,
        "one_timeline_play_site": count_suffix(".play") == 1,
        "one_pause_commit_site_only": count_suffix(".commit") == 1,
        "no_timeline_stop_site": count_suffix(".stop") == 0,
        "active_step_render_false_ast": step_render_false_ast,
        "active_step_update_fabric_false_or_absent_ast": step_update_fabric_default_ast,
        "legacy_worker_unreachable": not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_worker_legacy_unused"
            for node in ast.walk(tree)
        ),
    }
    return {
        "call_names": sorted(set(call_names)),
        "forbidden_hits": forbidden_hits,
        "forbidden_observation_hits": forbidden_observation_hits,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D366 output already exists; overwrite forbidden: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    input_hashes = _input_hashes()
    session_preregistration_sha256 = _sha(SESSION_DOC)
    session_preregistration_bytes = SESSION_DOC.stat().st_size
    d362_manifest = _tree_manifest(D362_DIR)
    d363_manifest = _tree_manifest(D363_DIR)
    d364_manifest = _tree_manifest(D364_DIR)
    d365_manifest = _tree_manifest(D365_DIR)
    d364_evidence_audit = _d364_completion_evidence_audit()
    d365_evidence_audit = _d365_completion_evidence_audit()
    source_contract_audit = _installed_source_contract_audit()
    cached_patch = _git_cached_patch()
    expected_cached_names = {
        "M\tSTART_HERE.md",
        *{
            f"A\t{_rel(D365_RUN_DIR / name)}"
            for name in (
                "d365_baseline_opposite_actual_isaac.png",
                "d365_baseline_primary_actual_isaac.png",
                "d365_post_forward_opposite_actual_isaac.png",
                "d365_post_forward_primary_actual_isaac.png",
                "d365_post_write_no_forward_opposite_actual_isaac.png",
                "d365_post_write_no_forward_primary_actual_isaac.png",
                "d365_state_layer_localization_rerun.png",
                "d365_state_layer_localization_sheet_ko.png",
                "d365_worker_stdout_stderr.log",
            )
        },
    }
    sidecar = _sidecar_hashes()
    negative = _negative_controls()
    static = _static_source_audit()
    gpu = _gpu_snapshot()
    rerun = subprocess.run([str(RERUN_CLI), "--version"], text=True, capture_output=True, check=False)
    display = subprocess.run(["xdpyinfo", "-display", DISPLAY], text=True, capture_output=True, check=False)
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_exact_d365_continuity_and_d366": _status_scope_ok(_git_status()),
        "session_preregistered": SESSION_DOC.is_file()
        and "USER_APPROVED_D366_PREREGISTERED_NO_ISAAC_INVOCATION" in SESSION_DOC.read_text(encoding="utf-8"),
        "input_hashes_exact": input_hashes == EXPECTED_INPUT_HASHES,
        "d362_manifest_33": d362_manifest["file_count"] == 33,
        "d363_manifest_40": d363_manifest["file_count"] == 40,
        "d364_manifest_17": d364_manifest["file_count"] == 17,
        "d365_manifest_29": d365_manifest["file_count"] == 29,
        "d364_completion_evidence_map_exact": d364_evidence_audit["pass"],
        "d365_completion_evidence_exact": d365_evidence_audit["pass"],
        "installed_source_contract_exact": source_contract_audit["pass"],
        "staged_continuity_inventory_exact": set(cached_patch["name_status"])
        == expected_cached_names
        and len(cached_patch["name_status"]) == 10
        and cached_patch["byte_count"] > 0,
        "d334_sidecar_nonempty": bool(sidecar),
        "negative_controls_pass": negative["pass"],
        "static_source_audit_pass": static["pass"],
        "numpy_pin": np.__version__ == "1.26.0",
        "psutil_pin": psutil.__version__ == "5.9.8",
        "rerun_pin": rerun.returncode == 0 and RERUN_VERSION in (rerun.stdout + rerun.stderr),
        "display_available": display.returncode == 0,
        "gpu_exact": gpu.get("gpu_name") == "NVIDIA GeForce RTX 4090 Laptop GPU",
        "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
        "ram_free_gate": int(gpu.get("ram_available_bytes") or 0) >= MIN_RAM_AVAILABLE_BYTES,
    }
    prereg = {
        "artifact": "D366_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "utc": _utc_now(),
        "run_nonce": secrets.token_hex(32),
        "new_variables": NEW_VARIABLES,
        "base_git": BASE_GIT,
        "harness_sha256": _sha(HARNESS),
        "session_preregistration_sha256": session_preregistration_sha256,
        "session_preregistration_bytes": session_preregistration_bytes,
        "input_hashes": input_hashes,
        "d362_manifest_before": d362_manifest,
        "d363_manifest_before": d363_manifest,
        "d364_manifest_before": d364_manifest,
        "d365_manifest_before": d365_manifest,
        "d364_completion_evidence_audit": d364_evidence_audit,
        "d365_completion_evidence_audit": d365_evidence_audit,
        "installed_source_contract_audit": source_contract_audit,
        "git_cached_patch": cached_patch,
        "d334_sidecar_before": sidecar,
        "target_row_index": TARGET_ROW_INDEX,
        "expected_global_step": EXPECTED_GLOBAL_STEP,
        "root_prim_path": ROOT_PRIM_PATH,
        "mesh_prim_path": MESH_PRIM_PATH,
        "registered_counts": {
            "root_pose_write": 1,
            "timeline_play": 1,
            "controlled_step_call": 1,
            "physics_callback_event": 1,
            "explicit_forward": 1,
            "controlled_physics_step": 1,
            "q5_science_sample": 0,
            "q5_target_update": 0,
            "contact_query": 0,
            "worker_invocation": 1,
            "automatic_retry": 0,
        },
        "thresholds": {
            "position_tolerance_m": POSITION_TOL_M,
            "quaternion_tolerance_deg": QUATERNION_TOL_DEG,
            "matrix_tolerance": MATRIX_TOL,
            "physics_dt_s": PHYSICS_DT_S,
            "time_tolerance_s": TIME_TOL_S,
            "material_centroid_delta_px": MATERIAL_CENTROID_DELTA_PX,
            "material_axis_delta_deg": MATERIAL_AXIS_DELTA_DEG,
            "material_iou_max": MATERIAL_IOU_MAX,
        },
        "negative_controls": negative,
        "static_source_audit": static,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    prepare = {
        "artifact": "D366_PREPARE_PREFLIGHT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "preregistration_sha256": _sha(PREREG_PATH),
        "session_preregistration_sha256": session_preregistration_sha256,
        "session_preregistration_bytes": session_preregistration_bytes,
        "gpu_and_ram": gpu,
        "display_returncode": display.returncode,
        "rerun_version_output": (rerun.stdout + rerun.stderr).strip(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREPARE_PATH, prepare)
    print(json.dumps({"stage": "prepare", "pass": prepare["pass"], "checks": checks}, ensure_ascii=False))
    return 0 if prepare["pass"] else 2


def _clock_snapshot(inner: Any, timeline: Any) -> dict[str, Any]:
    return {
        "custom_step_counter": int(inner._sim_step_counter),
        "simulation_clock": d362.d351._simulation_clock(inner),
        "physics_callback_count": int(_PHYSICS_CALLBACK_COUNT),
        "physics_callback_dts": list(_PHYSICS_CALLBACK_DTS),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time_s": float(timeline.get_current_time()),
    }


def _clock_no_advance(reference: dict[str, Any], observed: dict[str, Any]) -> bool:
    return bool(
        observed["custom_step_counter"] == reference["custom_step_counter"]
        and observed["simulation_clock"] == reference["simulation_clock"]
        and observed["physics_callback_count"] == reference["physics_callback_count"]
        and observed["physics_callback_dts"] == reference["physics_callback_dts"]
        and observed["timeline_time_s"] == reference["timeline_time_s"]
        and observed["timeline_playing"] == reference["timeline_playing"]
        and observed["timeline_stopped"] == reference["timeline_stopped"]
    )


def _physics_state_no_advance(reference: dict[str, Any], observed: dict[str, Any]) -> bool:
    return bool(
        observed["custom_step_counter"] == reference["custom_step_counter"]
        and observed["simulation_clock"] == reference["simulation_clock"]
        and observed["physics_callback_count"] == reference["physics_callback_count"]
        and observed["physics_callback_dts"] == reference["physics_callback_dts"]
    )


def _pause_no_advance(inner: Any, timeline: Any) -> dict[str, Any]:
    before = _clock_snapshot(inner, timeline)
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    commit_count = 0
    if timeline.is_playing():
        timeline.pause()
    if timeline.is_playing() and not timeline.is_stopped():
        timeline.commit()
        commit_count = 1
    after = _clock_snapshot(inner, timeline)
    checks = {
        "paused_not_stopped": not after["timeline_playing"] and not after["timeline_stopped"],
        "commit_at_most_once": commit_count in (0, 1),
        "physics_state_unchanged": _physics_state_no_advance(before, after),
    }
    return {"before": before, "after": after, "commit_count": commit_count, "checks": checks, "pass": all(checks.values())}


def _play_no_advance(inner: Any, timeline: Any) -> dict[str, Any]:
    before = _clock_snapshot(inner, timeline)
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, True)
    timeline.play()
    after = _clock_snapshot(inner, timeline)
    checks = {
        "playing_not_stopped": after["timeline_playing"] and not after["timeline_stopped"],
        "play_commit_not_used": True,
        "physics_state_unchanged": _physics_state_no_advance(before, after),
    }
    return {"before": before, "after": after, "commit_count": 0, "checks": checks, "pass": all(checks.values())}


def _physics_step_callback(step_size: float) -> None:
    global _PHYSICS_CALLBACK_COUNT
    _PHYSICS_CALLBACK_COUNT += 1
    _PHYSICS_CALLBACK_DTS.append(float(step_size))


def _quat_payload(value: Any) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])]


def _matrix_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(4, 4)
    position = [float(item) for item in value.ExtractTranslation()]
    quaternion = _quat_payload(value.ExtractRotationQuat())
    return {
        "matrix_row_major": array.tolist(),
        "position_m": position,
        "quaternion_wxyz": quaternion,
    }


def _attribute_payload(attribute: Any, kind: str) -> dict[str, Any]:
    valid = bool(attribute.IsValid())
    value = attribute.Get() if valid else None
    payload: dict[str, Any] = {"valid": valid, "value_present": value is not None}
    if value is None:
        payload["value"] = None
    elif kind == "matrix":
        payload["value"] = _matrix_payload(value)
    elif kind == "vector":
        payload["value"] = [float(item) for item in value]
    elif kind == "quaternion":
        payload["value"] = _quat_payload(value)
    else:
        raise ValueError(kind)
    return payload


def _read_asset_cache(inner: Any) -> dict[str, Any]:
    buffer = inner._sponge.data._root_link_pose_w
    if buffer.data is None:
        return {
            "available": False,
            "pose_wxyz": None,
            "raw_float32_bits": None,
            "cache_timestamp": float(buffer.timestamp),
            "simulation_timestamp": float(inner._sponge.data._sim_timestamp),
        }
    array = buffer.data[0].detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "available": True,
        "pose_wxyz": array.astype(np.float64).tolist(),
        "raw_float32_bits": array.tobytes().hex(),
        "cache_timestamp": float(buffer.timestamp),
        "simulation_timestamp": float(inner._sponge.data._sim_timestamp),
        "timestamp_not_stale": bool(buffer.timestamp >= inner._sponge.data._sim_timestamp),
    }


def _read_physx_tensor_view(inner: Any) -> dict[str, Any]:
    raw_tensor = inner._sponge.root_physx_view.get_transforms().clone()
    raw = raw_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    if raw.shape != (1, 7):
        raise RuntimeError(f"D366 PhysX transform shape drift: {raw.shape}")
    pose_wxyz = np.concatenate((raw[0, :3], raw[0, 6:7], raw[0, 3:6])).astype(np.float32)
    return {
        "available": True,
        "raw_pose_xyzw": raw[0].astype(np.float64).tolist(),
        "raw_float32_bits": raw[0].tobytes().hex(),
        "pose_wxyz": pose_wxyz.astype(np.float64).tolist(),
        "pose_wxyz_float32_bits": pose_wxyz.tobytes().hex(),
        "interpretation": "independent PhysX tensor-view readback; not claimed as solver-internal committed-state getter",
    }


def _read_usd_path(usd_stage: Any, path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim = usd_stage.GetPrimAtPath(path)
    valid = bool(prim.IsValid())
    matrix = None
    xformable = UsdGeom.Xformable(prim) if valid else None
    if xformable is not None and bool(xformable):
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return {
        "prim_path": path,
        "prim_valid": valid,
        "type_name": prim.GetTypeName() if valid else None,
        "world_transform": _matrix_payload(matrix),
        "dynamic_authority": False,
        "role": "authored USD control only; Fabric-enabled runtime may leave this stale",
    }


def _read_fabric_path(rt_stage: Any, hierarchy_interface: Any, path: str) -> dict[str, Any]:
    import usdrt

    sdf_path = usdrt.Sdf.Path(path)
    prim = rt_stage.GetPrimAtPath(sdf_path)
    valid = bool(prim.IsValid())
    result: dict[str, Any] = {
        "prim_path": path,
        "prim_valid": valid,
        "type_name": prim.GetTypeName() if valid else None,
        "fallback_to_usd_used": False,
    }
    if not valid:
        result.update(
            {
                "has_world_xform": False,
                "has_local_xform": False,
                "world_position": {"valid": False, "value_present": False, "value": None},
                "world_orientation": {"valid": False, "value_present": False, "value": None},
                "world_scale": {"valid": False, "value_present": False, "value": None},
                "compat_local_matrix": {"valid": False, "value_present": False, "value": None},
                "hierarchy_local_matrix": {"valid": False, "value_present": False, "value": None},
                "hierarchy_cached_world_matrix": {"valid": False, "value_present": False, "value": None},
                "hierarchy_current_computed_world_matrix": None,
            }
        )
        return result
    xformable = usdrt.Rt.Xformable(prim)
    result.update(
        {
            "has_world_xform": bool(xformable.HasWorldXform()),
            "has_local_xform": bool(xformable.HasLocalXform()),
            "world_position": _attribute_payload(xformable.GetWorldPositionAttr(), "vector"),
            "world_orientation": _attribute_payload(xformable.GetWorldOrientationAttr(), "quaternion"),
            "world_scale": _attribute_payload(xformable.GetWorldScaleAttr(), "vector"),
            "compat_local_matrix": _attribute_payload(xformable.GetLocalMatrixAttr(), "matrix"),
            "hierarchy_local_matrix": _attribute_payload(xformable.GetFabricHierarchyLocalMatrixAttr(), "matrix"),
            "hierarchy_cached_world_matrix": _attribute_payload(xformable.GetFabricHierarchyWorldMatrixAttr(), "matrix"),
        }
    )
    computed = hierarchy_interface.get_world_xform(sdf_path)
    result["hierarchy_current_computed_world_matrix"] = _matrix_payload(computed)
    return result


def _read_passive_layers(inner: Any, timeline: Any, usd_stage: Any, rt_stage: Any, hierarchy_interface: Any) -> dict[str, Any]:
    return {
        "clock": _clock_snapshot(inner, timeline),
        "assetdata_cache": _read_asset_cache(inner),
        "authored_usd": {
            "root": _read_usd_path(usd_stage, ROOT_PRIM_PATH),
            "mesh": _read_usd_path(usd_stage, MESH_PRIM_PATH),
        },
        "fabric_usdrt": {
            "root": _read_fabric_path(rt_stage, hierarchy_interface, ROOT_PRIM_PATH),
            "mesh": _read_fabric_path(rt_stage, hierarchy_interface, MESH_PRIM_PATH),
        },
    }


def _read_checkpoint(
    label: str,
    inner: Any,
    timeline: Any,
    usd_stage: Any,
    rt_stage: Any,
    hierarchy_interface: Any,
    *,
    include_physx_getter: bool,
) -> dict[str, Any]:
    before_getter = _read_passive_layers(inner, timeline, usd_stage, rt_stage, hierarchy_interface)
    physx = _read_physx_tensor_view(inner) if include_physx_getter else None
    after_getter = _read_passive_layers(inner, timeline, usd_stage, rt_stage, hierarchy_interface)
    payload = {
        "label": label,
        "passive_before_physx_getter": before_getter,
        "independent_physx_tensor_view": physx,
        "passive_after_physx_getter": after_getter,
        "getter_requested": include_physx_getter,
        "clock_unchanged_across_getter": _clock_no_advance(before_getter["clock"], after_getter["clock"]),
    }
    _journal_layer(label, payload)
    return payload


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
        raise RuntimeError(f"D366 capture timeline contract failed: {role}")
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    inner.sim.set_camera_view(camera_eye, CAMERA_TARGET)
    reference = _clock_snapshot(inner, timeline)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None or not hasattr(viewport, "set_texture_resolution"):
        raise RuntimeError("D366 active viewport unavailable")
    viewport.set_texture_resolution(tuple(VIEWPORT_SIZE))
    app_updates = 0
    for _ in range(8):
        simulation_app.update()
        app_updates += 1
        if not _clock_no_advance(reference, _clock_snapshot(inner, timeline)):
            raise RuntimeError(f"D366 app update advanced physics clock: {role}")
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(capture.wait_for_result(completion_frames=5), run_until_complete=False)
    deadline = time.monotonic() + CAPTURE_TIMEOUT_S
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()
        app_updates += 1
        if not _clock_no_advance(reference, _clock_snapshot(inner, timeline)):
            raise RuntimeError(f"D366 capture advanced physics clock: {role}")
    if not task.done():
        task.cancel()
        raise RuntimeError(f"D366 capture timeout: {role}")
    if not bool(task.result()):
        raise RuntimeError(f"D366 capture failed: {role}")
    for _ in range(2):
        simulation_app.update()
        app_updates += 1
    terminal = _clock_snapshot(inner, timeline)
    if not _clock_no_advance(reference, terminal):
        raise RuntimeError(f"D366 post-capture physics clock drift: {role}")
    _marker("worker", "viewport_capture", "complete", {"role": role, "path": _rel(path), "app_updates": app_updates})
    return {
        "path": _rel(path),
        "camera_eye": camera_eye,
        "camera_target": CAMERA_TARGET,
        "clock_before": reference,
        "clock_after": terminal,
        "app_update_count": app_updates,
        "physics_clock_unchanged": True,
    }


def _capture_phase(
    phase: str,
    simulation_app: Any,
    inner: Any,
    timeline: Any,
) -> dict[str, Any]:
    return {
        view: _capture_viewport(
            CAPTURE_PATHS[phase][view],
            simulation_app,
            inner,
            timeline,
            eye,
            f"{phase}_{view}",
        )
        for view, eye in CAMERA_EYES.items()
    }


def _write_target_pose_once(inner: Any, target_pose_wxyz: list[float]) -> dict[str, Any]:
    global _DISPLAY_STATE_WRITE_COUNT, _DISPLAY_STATE_WRITE_RETURNED
    import torch

    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    target = torch.tensor([target_pose_wxyz], device=inner.device, dtype=torch.float32)
    _DISPLAY_STATE_WRITE_COUNT += 1
    inner._sponge.write_root_pose_to_sim(target, env_ids=env_ids)
    _DISPLAY_STATE_WRITE_RETURNED = True
    cache = _read_asset_cache(inner)
    expected = np.asarray(target_pose_wxyz, dtype=np.float32)
    return {
        "target_pose_wxyz": expected.astype(np.float64).tolist(),
        "target_float32_bits": expected.tobytes().hex(),
        "cache_after_write": cache,
        "cache_bits_exact": cache.get("raw_float32_bits") == expected.tobytes().hex(),
    }


def _fabric_attestation(inner: Any, settings: Any, rt_stage: Any) -> dict[str, Any]:
    iface = inner.sim._fabric_iface
    selected = inner.sim._update_fabric
    selected_name = getattr(selected, "__name__", None)
    return {
        "cfg_use_fabric": bool(inner.sim.cfg.use_fabric),
        "is_fabric_enabled": bool(inner.sim.is_fabric_enabled()),
        "interface_present": iface is not None,
        "interface_type": f"{type(iface).__module__}.{type(iface).__qualname__}" if iface is not None else None,
        "interface_repr": repr(iface),
        "interface_has_force_update": callable(getattr(iface, "force_update", None)) if iface is not None else False,
        "interface_has_update": callable(getattr(iface, "update", None)) if iface is not None else False,
        "selected_callable_name": selected_name,
        "selected_callable_type": f"{type(selected).__module__}.{type(selected).__qualname__}",
        "selected_callable_repr": repr(selected),
        "selected_callable_bound_to_interface": getattr(selected, "__self__", None) is iface,
        "app_use_fabric_scene_delegate": settings.get("/app/useFabricSceneDelegate"),
        "hydra_reads_transforms_from_fabric": settings.get("/rtx/hydra/readTransformsFromFabricInRenderDelegate"),
        "physics_fabric_enabled_setting": settings.get("/physics/fabricEnabled"),
        "rt_stage_fabric_id": str(rt_stage.GetFabricId()),
        "rt_stage_id": str(rt_stage.GetStageIdAsStageId()),
    }


def _worker_legacy_unused(args: argparse.Namespace) -> int:
    global _EXPLICIT_FORWARD_COUNT, _EXPLICIT_FORWARD_RETURNED
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
            name for name in sys.modules
            if name in {"pxr", "omni", "isaaclab", "isaacsim", "carb", "usdrt"}
            or name.startswith(("pxr.", "omni.", "isaaclab.", "isaacsim.", "carb.", "usdrt."))
        )
        gpu = _gpu_snapshot()
        checks = {
            "prereg_prepare_pass": prereg.get("pass") is True and prepare.get("pass") is True,
            "single_invocation": invocation.get("invocation_index") == 1
            and invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("automatic_retry") is False,
            "registered_parent": supervisor_pid > 0 and os.getppid() == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest() == invocation.get("worker_token_sha256"),
            "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
            "git_scope_d364_d366_only": _status_scope_ok(_git_status()),
            "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "session_preregistration_hash_exact": _sha(SESSION_DOC)
            == prereg.get("session_preregistration_sha256")
            and SESSION_DOC.stat().st_size == prereg.get("session_preregistration_bytes"),
            "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
            "installed_source_contract_exact": _installed_source_contract_audit()
            == prereg.get("installed_source_contract_audit"),
            "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg.get("d362_manifest_before"),
            "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg.get("d363_manifest_before"),
            "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg.get("d364_manifest_before"),
            "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
            "registered_python": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
            "display_device_exact": os.environ.get("DISPLAY") == DISPLAY
            and args.headless is False and int(args.livestream) == 0 and str(args.device) == "cuda:0",
            "runtime_modules_absent_before_applauncher": not early_runtime_modules,
            "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
            "ram_free_gate": int(gpu.get("ram_available_bytes") or 0) >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D366_WORKER_PREFLIGHT_V1",
            "case": CASE,
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
            raise RuntimeError(f"D366 worker preflight STOP: {checks}")

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
            raise RuntimeError(f"D366 GUI launcher contract failed: {launcher_report}")

        import carb
        import omni.timeline
        import omni.usd
        import usdrt
        from pxr import Usd, UsdGeom, UsdUtils
        from usdrt import hierarchy

        args.robot_usd_path = d362.VARIANT_ROBOT_USD
        _marker("worker", "make_runtime_env", "start")
        inner = d362._make_runtime_env(args)
        _marker("worker", "make_runtime_env", "complete", {"pass": True})
        timeline = omni.timeline.get_timeline_interface()
        reset_before = _clock_snapshot(inner, timeline)
        _marker("worker", "reset", "start")
        inner.reset(seed=SEED)
        reset_after = _clock_snapshot(inner, timeline)
        _marker("worker", "reset", "complete", {"before": reset_before, "after": reset_after})
        settings = carb.settings.get_settings()
        previous_play = settings.get(PLAY_SIMULATIONS_SETTING)
        pause = _pause_no_advance(inner, timeline)
        _marker("worker", "timeline_pause", "complete", {"pass": pause["pass"]})
        if not pause["pass"]:
            raise RuntimeError(f"D366 pause bridge failed: {pause['checks']}")

        rows = _json(D362_TRACE)
        if len(rows) != 500 or rows[TARGET_ROW_INDEX].get("global_step") != EXPECTED_GLOBAL_STEP:
            raise RuntimeError("D366 frozen target-row contract failed")
        target_row = rows[TARGET_ROW_INDEX]
        target_pose_wxyz = [*target_row["object_pos_w_m"], *target_row["object_quat_wxyz"]]
        controlled_baseline = _clock_snapshot(inner, timeline)

        usd_stage = omni.usd.get_context().get_stage()
        usd_stage_cache = UsdUtils.StageCache.Get()
        usd_stage_id = usd_stage_cache.GetId(usd_stage).ToLongInt()
        if usd_stage_id < 0:
            usd_stage_id = usd_stage_cache.Insert(usd_stage).ToLongInt()
        rt_stage = usdrt.Usd.Stage.Attach(usd_stage_id)
        hierarchy_interface = hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
        )
        fabric = _fabric_attestation(inner, settings, rt_stage)
        physx_prim_paths = [str(path) for path in list(inner._sponge.root_physx_view.prim_paths)]

        _marker("worker", "baseline_layers", "start")
        baseline_checkpoint = _read_checkpoint(
            "baseline_pre_capture",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )
        baseline_captures = _capture_phase("baseline", simulation_app, inner, timeline)
        baseline_after_capture = _read_checkpoint(
            "baseline_post_capture",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )
        _marker("worker", "baseline_layers", "complete")

        baseline_root_cached = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["root"]["hierarchy_cached_world_matrix"]
        baseline_mesh_cached = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["mesh"]["hierarchy_cached_world_matrix"]
        baseline_root_computed = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["root"]["hierarchy_current_computed_world_matrix"]
        baseline_mesh_computed = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["mesh"]["hierarchy_current_computed_world_matrix"]
        baseline_physx_pose = baseline_after_capture["independent_physx_tensor_view"]["pose_wxyz"]
        baseline_root_record = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["root"]
        baseline_root_position = baseline_root_record["world_position"]["value"]
        baseline_root_orientation = baseline_root_record["world_orientation"]["value"]
        baseline_root_compatibility_pose = (
            [*baseline_root_position, *baseline_root_orientation]
            if baseline_root_position is not None and baseline_root_orientation is not None
            else None
        )
        d364_runtime_source = _json(D364_RUNTIME_SOURCE)
        d364_baseline_checkpoint = d364_runtime_source["baseline_after_capture"]
        d364_baseline_passive = d364_baseline_checkpoint["passive_after_physx_getter"]
        d364_baseline_physx_pose = d364_baseline_checkpoint["independent_physx_tensor_view"]["pose_wxyz"]
        d364_baseline_root_current = d364_baseline_passive["fabric_usdrt"]["root"]["hierarchy_current_computed_world_matrix"]
        d364_baseline_mesh_current = d364_baseline_passive["fabric_usdrt"]["mesh"]["hierarchy_current_computed_world_matrix"]
        d364_baseline_mesh_cached = d364_baseline_passive["fabric_usdrt"]["mesh"]["hierarchy_cached_world_matrix"]["value"]
        target_root_matrix = usdrt.Gf.Matrix4d(1.0)
        target_root_matrix.SetRotate(
            usdrt.Gf.Quatd(
                float(target_pose_wxyz[3]),
                usdrt.Gf.Vec3d(
                    float(target_pose_wxyz[4]),
                    float(target_pose_wxyz[5]),
                    float(target_pose_wxyz[6]),
                ),
            )
        )
        target_root_matrix.SetTranslateOnly(usdrt.Gf.Vec3d(*[float(item) for item in target_pose_wxyz[:3]]))
        target_root_matrix_payload = _matrix_payload(target_root_matrix)
        baseline_root_matrix_array = np.asarray(
            baseline_root_computed["matrix_row_major"], dtype=np.float64
        )
        baseline_mesh_matrix_array = np.asarray(
            baseline_mesh_computed["matrix_row_major"], dtype=np.float64
        )
        usd_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        authored_root_prim = usd_stage.GetPrimAtPath(ROOT_PRIM_PATH)
        authored_mesh_prim = usd_stage.GetPrimAtPath(MESH_PRIM_PATH)
        authored_relative_gf, authored_relative_resets = usd_xform_cache.ComputeRelativeTransform(
            authored_mesh_prim, authored_root_prim
        )
        root_to_mesh_relative = np.asarray(authored_relative_gf, dtype=np.float64).reshape(4, 4)
        baseline_reconstructed_mesh = root_to_mesh_relative @ baseline_root_matrix_array
        baseline_reconstruction_max_abs = float(
            np.max(np.abs(baseline_reconstructed_mesh - baseline_mesh_matrix_array))
        )
        target_mesh_matrix_array = root_to_mesh_relative @ np.asarray(
            target_root_matrix_payload["matrix_row_major"], dtype=np.float64
        )
        expected_geometry = {
            "baseline_root_computed": baseline_root_computed,
            "baseline_mesh_computed": baseline_mesh_computed,
            "target_root_matrix": target_root_matrix_payload,
            "authored_root_to_mesh_relative_matrix": root_to_mesh_relative.tolist(),
            "authored_relative_resets_xform_stack": bool(authored_relative_resets),
            "target_mesh_matrix_row_major": target_mesh_matrix_array.tolist(),
            "baseline_reconstruction_max_abs": baseline_reconstruction_max_abs,
            "multiplication_order": "UsdGeom.XformCache.ComputeRelativeTransform(mesh, root) @ root_world",
            "relative_authority": "authored static root-to-mesh hierarchy only; dynamic root authority remains Fabric/PhysX",
        }
        baseline_root_local = baseline_root_record["hierarchy_local_matrix"]
        baseline_mesh_record = baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["mesh"]
        baseline_mesh_local = baseline_mesh_record["hierarchy_local_matrix"]

        def matrices_match(left: Any, right: Any) -> bool:
            if left is None or right is None:
                return False
            return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))) <= MATRIX_TOL

        compatibility_available = (
            baseline_root_record["world_position"]["valid"]
            and baseline_root_record["world_position"]["value_present"]
            and baseline_root_record["world_orientation"]["valid"]
            and baseline_root_record["world_orientation"]["value_present"]
        )
        optional_diagnostics = {
            "compatibility_world_pose_available": compatibility_available,
            "compatibility_world_pose": baseline_root_compatibility_pose,
            "baseline_physx_matches_compatibility": (
                _pose_error(baseline_root_compatibility_pose, baseline_physx_pose)["match"]
                if baseline_root_compatibility_pose is not None else None
            ),
            "root_cached_world_matrix_available": baseline_root_cached["valid"]
            and baseline_root_cached["value_present"],
            "root_cached_matches_root_current": (
                matrices_match(
                    baseline_root_cached["value"]["matrix_row_major"],
                    baseline_root_computed["matrix_row_major"],
                )
                if baseline_root_cached["valid"] and baseline_root_cached["value_present"] else None
            ),
            "compatibility_and_root_cached_are_linear_verdict_inputs": False,
        }
        runtime_checks = {
            "trace_target_row_exact": len(rows) == 500 and target_row.get("global_step") == EXPECTED_GLOBAL_STEP,
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "controlled_counter_baseline_zero": int(inner._sim_step_counter) == 0,
            "physx_prim_path_exact": physx_prim_paths == [ROOT_PRIM_PATH],
            "cfg_use_fabric_true": fabric["cfg_use_fabric"] is True,
            "fabric_interface_enabled": fabric["is_fabric_enabled"] is True and fabric["interface_present"] is True,
            "force_update_selected": fabric["interface_has_force_update"] is True
            and fabric["selected_callable_name"] == "force_update"
            and fabric["selected_callable_bound_to_interface"] is True,
            "fabric_scene_delegate_true": fabric["app_use_fabric_scene_delegate"] is True,
            "hydra_reads_fabric_true": fabric["hydra_reads_transforms_from_fabric"] is True,
            "physics_fabric_setting_not_explicit_false": fabric["physics_fabric_enabled_setting"] is not False,
            "root_and_mesh_fabric_prims_valid": baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["root"]["prim_valid"]
            and baseline_after_capture["passive_after_physx_getter"]["fabric_usdrt"]["mesh"]["prim_valid"],
            "root_and_mesh_hierarchy_local_available": baseline_root_local["valid"]
            and baseline_root_local["value_present"]
            and baseline_mesh_local["valid"]
            and baseline_mesh_local["value_present"],
            "root_and_mesh_hierarchy_current_available": baseline_root_computed is not None
            and baseline_mesh_computed is not None,
            "boundable_mesh_cached_world_matrix_available": baseline_mesh_cached["valid"]
            and baseline_mesh_cached["value_present"],
            "authored_relative_does_not_reset_stack": not bool(authored_relative_resets),
            "baseline_root_mesh_compose_independent_check": baseline_reconstruction_max_abs <= MATRIX_TOL,
            "baseline_physx_matches_fabric_hierarchy_current": _pose_error(
                [
                    *baseline_root_computed["position_m"],
                    *baseline_root_computed["quaternion_wxyz"],
                ],
                baseline_physx_pose,
            )["match"],
            "baseline_mesh_current_matches_render_cache": matrices_match(
                baseline_mesh_computed["matrix_row_major"],
                (baseline_mesh_cached.get("value") or {}).get("matrix_row_major"),
            ),
            "d364_baseline_physx_inherited": _pose_error(
                baseline_physx_pose, d364_baseline_physx_pose
            )["match"],
            "d364_baseline_root_current_inherited": matrices_match(
                baseline_root_computed["matrix_row_major"],
                d364_baseline_root_current["matrix_row_major"],
            ),
            "d364_baseline_mesh_current_inherited": matrices_match(
                baseline_mesh_computed["matrix_row_major"],
                d364_baseline_mesh_current["matrix_row_major"],
            ),
            "d364_baseline_mesh_cached_inherited": matrices_match(
                (baseline_mesh_cached.get("value") or {}).get("matrix_row_major"),
                d364_baseline_mesh_cached["matrix_row_major"],
            ),
            "baseline_getter_clock_unchanged": baseline_checkpoint["clock_unchanged_across_getter"]
            and baseline_after_capture["clock_unchanged_across_getter"],
        }
        runtime = {
            "artifact": "D366_RUNTIME_FABRIC_ATTESTATION_V1",
            "case": CASE,
            "utc": _utc_now(),
            "launcher": launcher_report,
            "reset_internal_transition": {"before": reset_before, "after": reset_after, "excluded_from_controlled_steps": True},
            "pause_bridge": pause,
            "controlled_zero_step_baseline": controlled_baseline,
            "usd_stage_id": usd_stage_id,
            "physx_prim_paths": physx_prim_paths,
            "fabric": fabric,
            "expected_geometry": expected_geometry,
            "d364_raw_baseline_authority": {
                "source": _rel(D364_RUNTIME_SOURCE),
                "source_sha256": _sha(D364_RUNTIME_SOURCE),
                "physx_pose_wxyz": d364_baseline_physx_pose,
                "root_current": d364_baseline_root_current,
                "mesh_current": d364_baseline_mesh_current,
                "mesh_cached": d364_baseline_mesh_cached,
            },
            "optional_diagnostics": optional_diagnostics,
            "baseline_checkpoint": baseline_checkpoint,
            "baseline_after_capture": baseline_after_capture,
            "required_checks": runtime_checks,
            "checks": runtime_checks,
            "pass": all(runtime_checks.values()),
        }
        _write_json_x(RUNTIME_PATH, runtime)
        if not runtime["pass"]:
            raise RuntimeError(f"D366 runtime Fabric prerequisites STOP: {runtime_checks}")

        _marker("worker", "direct_root_pose_write", "start", {"row_index": TARGET_ROW_INDEX})
        write = _write_target_pose_once(inner, target_pose_wxyz)
        _marker("worker", "direct_root_pose_write", "complete", {"pass": write["cache_bits_exact"]})

        post_write_immediate = _read_checkpoint(
            "post_write_immediate",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )
        post_write_captures = _capture_phase("post_write_no_forward", simulation_app, inner, timeline)
        post_write_after_capture = _read_checkpoint(
            "post_write_after_app_update",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )

        _marker("worker", "explicit_simulation_context_forward", "start", {"selected_callable": fabric["selected_callable_name"]})
        forward_before = _clock_snapshot(inner, timeline)
        _EXPLICIT_FORWARD_COUNT += 1
        inner.sim.forward()
        _EXPLICIT_FORWARD_RETURNED = True
        forward_after = _clock_snapshot(inner, timeline)
        if not _clock_no_advance(forward_before, forward_after):
            raise RuntimeError("D366 explicit forward advanced physics clock")
        _marker("worker", "explicit_simulation_context_forward", "complete", {"forward_count": _EXPLICIT_FORWARD_COUNT})

        post_forward_immediate = _read_checkpoint(
            "post_forward_immediate",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )
        post_forward_captures = _capture_phase("post_forward", simulation_app, inner, timeline)
        post_forward_terminal = _read_checkpoint(
            "post_forward_after_app_update",
            inner,
            timeline,
            usd_stage,
            rt_stage,
            hierarchy_interface,
            include_physx_getter=True,
        )

        final_clock = _clock_snapshot(inner, timeline)
        counter_checks = {
            "root_pose_write_1": _DISPLAY_STATE_WRITE_COUNT == 1,
            "root_pose_write_returned": _DISPLAY_STATE_WRITE_RETURNED is True,
            "explicit_forward_1": _EXPLICIT_FORWARD_COUNT == 1,
            "explicit_forward_returned": _EXPLICIT_FORWARD_RETURNED is True,
            "controlled_physics_steps_0": _CONTROLLED_PHYSICS_STEPS == 0,
            "q5_science_samples_0": _Q5_SCIENCE_SAMPLE_COUNT == 0,
            "q5_target_updates_0": _Q5_TARGET_UPDATE_COUNT == 0,
            "contact_queries_0": _CONTACT_QUERY_COUNT == 0,
            "clock_counter_unchanged": _clock_no_advance(controlled_baseline, final_clock),
            "all_getter_guards_unchanged": all(
                checkpoint["clock_unchanged_across_getter"]
                for checkpoint in (
                    baseline_checkpoint,
                    baseline_after_capture,
                    post_write_immediate,
                    post_write_after_capture,
                    post_forward_immediate,
                    post_forward_terminal,
                )
            ),
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "d362_unchanged": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
            "d363_unchanged": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
            "d364_unchanged": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
            "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "inputs_unchanged": _input_hashes() == prereg["input_hashes"],
        }
        summary = {
            "artifact": "D366_WORKER_SUMMARY_V1",
            "case": CASE,
            "utc": _utc_now(),
            "new_variables": NEW_VARIABLES,
            "target_row": {
                "row_index": TARGET_ROW_INDEX,
                "global_step": target_row["global_step"],
                "object_pos_w_m": target_row["object_pos_w_m"],
                "object_quat_wxyz": target_row["object_quat_wxyz"],
                "object_disp_xy_mm": target_row["object_disp_xy_mm"],
                "object_tilt_delta_deg": target_row["object_tilt_delta_from_reference_deg"],
            },
            "write": write,
            "controlled_zero_step_baseline": controlled_baseline,
            "fabric_attestation": fabric,
            "expected_geometry": expected_geometry,
            "checkpoints": {
                "baseline_pre_capture": baseline_checkpoint,
                "baseline_post_capture": baseline_after_capture,
                "post_write_immediate": post_write_immediate,
                "post_write_after_app_update": post_write_after_capture,
                "post_forward_immediate": post_forward_immediate,
                "post_forward_after_app_update": post_forward_terminal,
            },
            "captures": {
                "baseline": baseline_captures,
                "post_write_no_forward": post_write_captures,
                "post_forward": post_forward_captures,
            },
            "forward_guard": {"before": forward_before, "after": forward_after, "pass": True},
            "final_clock": final_clock,
            "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
            "display_state_write_returned": _DISPLAY_STATE_WRITE_RETURNED,
            "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
            "explicit_forward_returned": _EXPLICIT_FORWARD_RETURNED,
            "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
            "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "contact_query_count": _CONTACT_QUERY_COUNT,
            "counter_checks": counter_checks,
            "target_ik_path_changed": False,
            "physics_or_renderer_settings_changed": False,
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
                    "artifact": "D366_WORKER_EXCEPTION_STOP_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
                    "display_state_write_returned": _DISPLAY_STATE_WRITE_RETURNED,
                    "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
                    "explicit_forward_returned": _EXPLICIT_FORWARD_RETURNED,
                    "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
                    "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "contact_query_count": _CONTACT_QUERY_COUNT,
                    "g0a_pass": False,
                    "pass": False,
                },
            )
        try:
            _marker("worker", "worker_exception", "stop", {"error": f"{type(error).__name__}: {error}"})
        except Exception:
            pass
        return 2
    finally:
        try:
            if settings is not None and previous_play is not None:
                settings.set(PLAY_SIMULATIONS_SETTING, previous_play)
        except Exception:
            pass
        try:
            if inner is not None:
                inner.close()
        except Exception:
            pass
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass


def _worker(args: argparse.Namespace) -> int:
    global _DISPLAY_STATE_WRITE_COUNT, _DISPLAY_STATE_WRITE_RETURNED
    global _CONTROLLED_STEP_CALL_COUNT, _CONTROLLED_STEP_RETURNED, _CONTROLLED_PHYSICS_STEPS
    global _EXPLICIT_FORWARD_COUNT, _EXPLICIT_FORWARD_RETURNED
    simulation_app = None
    inner = None
    settings = None
    previous_play: Any = None
    callback_registered = False
    try:
        prereg = _json(PREREG_PATH)
        prepare = _json(PREPARE_PATH)
        invocation = _json(INVOCATION_PATH)
        token = os.environ.get(WORKER_TOKEN_ENV, "")
        supervisor_pid = int(os.environ.get(SUPERVISOR_PID_ENV, "-1"))
        early_runtime_modules = sorted(
            name for name in sys.modules
            if name in {"pxr", "omni", "isaaclab", "isaacsim", "carb", "usdrt"}
            or name.startswith(("pxr.", "omni.", "isaaclab.", "isaacsim.", "carb.", "usdrt."))
        )
        gpu = _gpu_snapshot()
        checks = {
            "prereg_prepare_pass": prereg.get("pass") is True and prepare.get("pass") is True,
            "single_invocation": invocation.get("invocation_index") == 1
            and invocation.get("run_nonce") == prereg.get("run_nonce")
            and invocation.get("automatic_retry") is False,
            "registered_parent": supervisor_pid > 0 and os.getppid() == supervisor_pid,
            "one_time_token": bool(token)
            and hashlib.sha256(token.encode()).hexdigest() == invocation.get("worker_token_sha256"),
            "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
            "git_scope_exact": _status_scope_ok(_git_status()),
            "staged_continuity_exact": _git_cached_patch() == prereg.get("git_cached_patch"),
            "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "session_preregistration_hash_exact": _sha(SESSION_DOC)
            == prereg.get("session_preregistration_sha256")
            and SESSION_DOC.stat().st_size == prereg.get("session_preregistration_bytes"),
            "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
            "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg.get("d362_manifest_before"),
            "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg.get("d363_manifest_before"),
            "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg.get("d364_manifest_before"),
            "d365_manifest_exact": _tree_manifest(D365_DIR) == prereg.get("d365_manifest_before"),
            "sidecar_exact": _sidecar_hashes() == prereg.get("d334_sidecar_before"),
            "registered_python": Path(sys.executable).resolve() == Path(REGISTERED_PYTHON).resolve(),
            "display_device_exact": os.environ.get("DISPLAY") == DISPLAY
            and args.headless is False and int(args.livestream) == 0 and str(args.device) == "cuda:0",
            "runtime_modules_absent_before_applauncher": not early_runtime_modules,
            "gpu_free_gate": int(gpu.get("memory_free_mib") or 0) >= MIN_GPU_FREE_MIB,
            "ram_free_gate": int(gpu.get("ram_available_bytes") or 0) >= MIN_RAM_AVAILABLE_BYTES,
        }
        preflight = {
            "artifact": "D366_WORKER_PREFLIGHT_V1",
            "case": CASE,
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
            raise RuntimeError(f"D366 worker preflight STOP: {checks}")

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
            raise RuntimeError(f"D366 GUI launcher contract failed: {launcher_report}")

        import carb
        import omni.timeline
        import omni.usd
        import usdrt
        from pxr import Usd, UsdGeom, UsdUtils
        from usdrt import hierarchy

        args.robot_usd_path = d362.VARIANT_ROBOT_USD
        _marker("worker", "make_runtime_env", "start")
        inner = d362._make_runtime_env(args)
        _marker("worker", "make_runtime_env", "complete", {"pass": True})
        timeline = omni.timeline.get_timeline_interface()
        reset_before = _clock_snapshot(inner, timeline)
        _marker("worker", "reset", "start")
        inner.reset(seed=SEED)
        reset_after = _clock_snapshot(inner, timeline)
        _marker("worker", "reset", "complete", {"before": reset_before, "after": reset_after})
        settings = carb.settings.get_settings()
        previous_play = settings.get(PLAY_SIMULATIONS_SETTING)
        initial_pause = _pause_no_advance(inner, timeline)
        _marker("worker", "initial_timeline_pause", "complete", {"pass": initial_pause["pass"]})
        if not initial_pause["pass"]:
            raise RuntimeError(f"D366 initial pause bridge failed: {initial_pause['checks']}")

        rows = _json(D362_TRACE)
        if len(rows) != 500 or rows[TARGET_ROW_INDEX].get("global_step") != EXPECTED_GLOBAL_STEP:
            raise RuntimeError("D366 frozen target-row contract failed")
        target_row = rows[TARGET_ROW_INDEX]
        commanded_pose_wxyz = [*target_row["object_pos_w_m"], *target_row["object_quat_wxyz"]]
        controlled_baseline = _clock_snapshot(inner, timeline)

        usd_stage = omni.usd.get_context().get_stage()
        usd_stage_cache = UsdUtils.StageCache.Get()
        usd_stage_id = usd_stage_cache.GetId(usd_stage).ToLongInt()
        if usd_stage_id < 0:
            usd_stage_id = usd_stage_cache.Insert(usd_stage).ToLongInt()
        rt_stage = usdrt.Usd.Stage.Attach(usd_stage_id)
        hierarchy_interface = hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
        )
        fabric = _fabric_attestation(inner, settings, rt_stage)
        physx_prim_paths = [str(path) for path in list(inner._sponge.root_physx_view.prim_paths)]

        _marker("worker", "baseline_layers", "start")
        baseline_checkpoint = _read_checkpoint(
            "baseline_pre_capture", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        baseline_captures = _capture_phase("baseline", simulation_app, inner, timeline)
        baseline_after_capture = _read_checkpoint(
            "baseline_post_capture", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        _marker("worker", "baseline_layers", "complete")

        def fabric_matrix(checkpoint: dict[str, Any], node: str, field: str) -> list[list[float]] | None:
            record = checkpoint["passive_after_physx_getter"]["fabric_usdrt"][node][field]
            if field == "hierarchy_current_computed_world_matrix":
                return record.get("matrix_row_major") if record is not None else None
            value = record.get("value") if isinstance(record, dict) else None
            return value.get("matrix_row_major") if value is not None else None

        def matrices_match(left: Any, right: Any) -> bool:
            if left is None or right is None:
                return False
            return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))) <= MATRIX_TOL

        baseline_pose = baseline_after_capture["independent_physx_tensor_view"]["pose_wxyz"]
        baseline_root = fabric_matrix(
            baseline_after_capture, "root", "hierarchy_current_computed_world_matrix"
        )
        baseline_mesh = fabric_matrix(
            baseline_after_capture, "mesh", "hierarchy_current_computed_world_matrix"
        )
        baseline_mesh_cached = fabric_matrix(
            baseline_after_capture, "mesh", "hierarchy_cached_world_matrix"
        )
        authored_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        authored_root_prim = usd_stage.GetPrimAtPath(ROOT_PRIM_PATH)
        authored_mesh_prim = usd_stage.GetPrimAtPath(MESH_PRIM_PATH)
        authored_relative_gf, authored_relative_resets = authored_cache.ComputeRelativeTransform(
            authored_mesh_prim, authored_root_prim
        )
        root_to_mesh_relative = np.asarray(authored_relative_gf, dtype=np.float64).reshape(4, 4)
        baseline_reconstruction_max_abs = float(
            np.max(np.abs(root_to_mesh_relative @ np.asarray(baseline_root) - np.asarray(baseline_mesh)))
        )

        def root_matrix_from_pose(pose: list[float]) -> dict[str, Any]:
            matrix = usdrt.Gf.Matrix4d(1.0)
            matrix.SetRotate(
                usdrt.Gf.Quatd(
                    float(pose[3]),
                    usdrt.Gf.Vec3d(float(pose[4]), float(pose[5]), float(pose[6])),
                )
            )
            matrix.SetTranslateOnly(usdrt.Gf.Vec3d(*[float(item) for item in pose[:3]]))
            payload = _matrix_payload(matrix)
            if payload is None:
                raise RuntimeError("D366 root matrix construction failed")
            return payload

        commanded_root = root_matrix_from_pose(commanded_pose_wxyz)
        commanded_mesh = root_to_mesh_relative @ np.asarray(
            commanded_root["matrix_row_major"], dtype=np.float64
        )
        d365_runtime = _json(D365_RUNTIME_SOURCE)
        d365_baseline = d365_runtime["baseline_after_capture"]
        d365_baseline_pose = d365_baseline["independent_physx_tensor_view"]["pose_wxyz"]
        d365_root = d365_baseline["passive_after_physx_getter"]["fabric_usdrt"]["root"][
            "hierarchy_current_computed_world_matrix"
        ]["matrix_row_major"]
        d365_mesh = d365_baseline["passive_after_physx_getter"]["fabric_usdrt"]["mesh"][
            "hierarchy_current_computed_world_matrix"
        ]["matrix_row_major"]
        d365_mesh_cached = d365_baseline["passive_after_physx_getter"]["fabric_usdrt"]["mesh"][
            "hierarchy_cached_world_matrix"
        ]["value"]["matrix_row_major"]
        runtime_checks = {
            "trace_target_row_exact": len(rows) == 500
            and target_row.get("global_step") == EXPECTED_GLOBAL_STEP,
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "controlled_counter_baseline_zero": int(inner._sim_step_counter) == 0,
            "callback_baseline_zero": _PHYSICS_CALLBACK_COUNT == 0 and not _PHYSICS_CALLBACK_DTS,
            "callback_name_absent_before_registration": not inner.sim.physics_callback_exists(_CALLBACK_NAME),
            "physics_dt_exact": abs(float(inner.sim.get_physics_dt()) - PHYSICS_DT_S) <= TIME_TOL_S,
            "physx_prim_path_exact": physx_prim_paths == [ROOT_PRIM_PATH],
            "fabric_attestation_exact": fabric["cfg_use_fabric"] is True
            and fabric["is_fabric_enabled"] is True
            and fabric["interface_present"] is True
            and fabric["selected_callable_name"] == "force_update"
            and fabric["selected_callable_bound_to_interface"] is True
            and fabric["app_use_fabric_scene_delegate"] is True
            and fabric["hydra_reads_transforms_from_fabric"] is True,
            "baseline_layers_available": baseline_root is not None
            and baseline_mesh is not None and baseline_mesh_cached is not None,
            "baseline_root_mesh_compose": baseline_reconstruction_max_abs <= MATRIX_TOL,
            "baseline_mesh_current_matches_cache": matrices_match(baseline_mesh, baseline_mesh_cached),
            "d365_baseline_physx_inherited": _pose_error(baseline_pose, d365_baseline_pose)["match"],
            "d365_baseline_root_inherited": matrices_match(baseline_root, d365_root),
            "d365_baseline_mesh_inherited": matrices_match(baseline_mesh, d365_mesh),
            "d365_baseline_mesh_cache_inherited": matrices_match(baseline_mesh_cached, d365_mesh_cached),
            "authored_relative_does_not_reset_stack": not bool(authored_relative_resets),
            "baseline_getter_guards": baseline_checkpoint["clock_unchanged_across_getter"]
            and baseline_after_capture["clock_unchanged_across_getter"],
        }
        expected_geometry: dict[str, Any] = {
            "baseline_root_matrix_row_major": baseline_root,
            "baseline_mesh_matrix_row_major": baseline_mesh,
            "commanded_root_matrix_row_major": commanded_root["matrix_row_major"],
            "commanded_mesh_matrix_row_major": commanded_mesh.tolist(),
            "authored_root_to_mesh_relative_matrix": root_to_mesh_relative.tolist(),
            "authored_relative_resets_xform_stack": bool(authored_relative_resets),
            "baseline_reconstruction_max_abs": baseline_reconstruction_max_abs,
            "multiplication_order": "root_to_mesh_relative @ root_world",
            "post_step_authority_pending": True,
        }
        runtime = {
            "artifact": "D366_RUNTIME_STEP_FABRIC_ATTESTATION_V1",
            "case": CASE,
            "utc": _utc_now(),
            "launcher": launcher_report,
            "reset_internal_transition": {
                "before": reset_before,
                "after": reset_after,
                "excluded_from_controlled_steps": True,
            },
            "initial_pause": initial_pause,
            "controlled_baseline": controlled_baseline,
            "usd_stage_id": usd_stage_id,
            "physx_prim_paths": physx_prim_paths,
            "fabric": fabric,
            "expected_geometry_pre_step": expected_geometry,
            "d365_runtime_authority": {
                "path": _rel(D365_RUNTIME_SOURCE),
                "sha256": _sha(D365_RUNTIME_SOURCE),
            },
            "baseline_checkpoint": baseline_checkpoint,
            "baseline_after_capture": baseline_after_capture,
            "required_checks": runtime_checks,
            "pass": all(runtime_checks.values()),
        }
        _write_json_x(RUNTIME_PATH, runtime)
        if not runtime["pass"]:
            raise RuntimeError(f"D366 runtime prerequisites STOP: {runtime_checks}")

        callback_before = _clock_snapshot(inner, timeline)
        inner.sim.add_physics_callback(_CALLBACK_NAME, _physics_step_callback)
        callback_registered = True
        callback_after = _clock_snapshot(inner, timeline)
        callback_registration_checks = {
            "callback_exists": inner.sim.physics_callback_exists(_CALLBACK_NAME),
            "registration_no_physics": _physics_state_no_advance(callback_before, callback_after),
            "callback_count_zero": _PHYSICS_CALLBACK_COUNT == 0 and not _PHYSICS_CALLBACK_DTS,
        }
        if not all(callback_registration_checks.values()):
            raise RuntimeError(f"D366 callback registration STOP: {callback_registration_checks}")

        _marker("worker", "timeline_play", "start")
        play_guard = _play_no_advance(inner, timeline)
        _marker("worker", "timeline_play", "complete", {"pass": play_guard["pass"]})
        if not play_guard["pass"]:
            raise RuntimeError(f"D366 PLAY transition STOP: {play_guard['checks']}")
        post_play = _read_checkpoint(
            "post_play_pre_write", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        post_play_checks = {
            "timeline_playing": timeline.is_playing() and not timeline.is_stopped(),
            "physx_unchanged": _pose_error(
                post_play["independent_physx_tensor_view"]["pose_wxyz"], baseline_pose
            )["match"],
            "root_unchanged": matrices_match(
                fabric_matrix(post_play, "root", "hierarchy_current_computed_world_matrix"), baseline_root
            ),
            "mesh_unchanged": matrices_match(
                fabric_matrix(post_play, "mesh", "hierarchy_current_computed_world_matrix"), baseline_mesh
            ),
            "mesh_cache_unchanged": matrices_match(
                fabric_matrix(post_play, "mesh", "hierarchy_cached_world_matrix"), baseline_mesh_cached
            ),
            "callback_still_zero": _PHYSICS_CALLBACK_COUNT == 0,
            "getter_no_advance": post_play["clock_unchanged_across_getter"],
        }
        if not all(post_play_checks.values()):
            raise RuntimeError(f"D366 PLAY transition confound STOP: {post_play_checks}")

        _marker("worker", "direct_root_pose_write", "start", {"row_index": TARGET_ROW_INDEX})
        write_before = _clock_snapshot(inner, timeline)
        write = _write_target_pose_once(inner, commanded_pose_wxyz)
        write_after = _clock_snapshot(inner, timeline)
        write_guard = {
            "playing_during_write": write_before["timeline_playing"] and write_after["timeline_playing"],
            "not_stopped": not write_before["timeline_stopped"] and not write_after["timeline_stopped"],
            "write_no_physics": _clock_no_advance(write_before, write_after),
            "cache_bits_exact": write["cache_bits_exact"],
        }
        _marker("worker", "direct_root_pose_write", "complete", {"pass": all(write_guard.values())})
        if not all(write_guard.values()):
            raise RuntimeError(f"D366 PLAY pose-write STOP: {write_guard}")
        post_write = _read_checkpoint(
            "post_write_pre_step", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        post_write_checks = {
            "physx_commanded": _pose_error(
                post_write["independent_physx_tensor_view"]["pose_wxyz"], commanded_pose_wxyz
            )["match"],
            "root_still_baseline": matrices_match(
                fabric_matrix(post_write, "root", "hierarchy_current_computed_world_matrix"), baseline_root
            ),
            "mesh_still_baseline": matrices_match(
                fabric_matrix(post_write, "mesh", "hierarchy_current_computed_world_matrix"), baseline_mesh
            ),
            "mesh_cache_still_baseline": matrices_match(
                fabric_matrix(post_write, "mesh", "hierarchy_cached_world_matrix"), baseline_mesh_cached
            ),
            "callback_still_zero": _PHYSICS_CALLBACK_COUNT == 0,
            "getter_no_advance": post_write["clock_unchanged_across_getter"],
        }
        post_write_required_checks = {
            "physx_commanded": post_write_checks["physx_commanded"],
            "callback_still_zero": post_write_checks["callback_still_zero"],
            "getter_no_advance": post_write_checks["getter_no_advance"],
        }
        if not all(post_write_required_checks.values()):
            raise RuntimeError(f"D366 pre-step state contract STOP: {post_write_required_checks}")

        _marker("worker", "controlled_sim_step_render_false", "start")
        step_before = _clock_snapshot(inner, timeline)
        physics_dt = float(inner.sim.get_physics_dt())
        _CONTROLLED_STEP_CALL_COUNT += 1
        inner._sim_step_counter += 1
        inner.sim.step(render=False)
        _CONTROLLED_STEP_RETURNED = True
        step_after = _clock_snapshot(inner, timeline)
        step_index_delta = int(
            step_after["simulation_clock"]["current_time_step_index"]
            - step_before["simulation_clock"]["current_time_step_index"]
        )
        step_time_delta = float(
            step_after["simulation_clock"]["current_time"]
            - step_before["simulation_clock"]["current_time"]
        )
        step_attestation_checks = {
            "step_call_1": _CONTROLLED_STEP_CALL_COUNT == 1,
            "step_returned": _CONTROLLED_STEP_RETURNED is True,
            "playing_before_after": step_before["timeline_playing"] and step_after["timeline_playing"],
            "not_stopped_before_after": not step_before["timeline_stopped"] and not step_after["timeline_stopped"],
            "custom_counter_delta_1": step_after["custom_step_counter"]
            == step_before["custom_step_counter"] + 1,
            "simulation_index_delta_1": step_index_delta == 1,
            "simulation_time_delta_dt": abs(step_time_delta - physics_dt) <= TIME_TOL_S,
            "physics_dt_registered": abs(physics_dt - PHYSICS_DT_S) <= TIME_TOL_S,
            "callback_count_1": _PHYSICS_CALLBACK_COUNT == 1,
            "callback_dt_one": len(_PHYSICS_CALLBACK_DTS) == 1
            and abs(_PHYSICS_CALLBACK_DTS[0] - physics_dt) <= TIME_TOL_S,
            "timeline_time_nondecreasing_diagnostic": step_after["timeline_time_s"]
            >= step_before["timeline_time_s"],
        }
        if all(step_attestation_checks.values()):
            _CONTROLLED_PHYSICS_STEPS = 1
        _marker(
            "worker", "controlled_sim_step_render_false", "complete",
            {"pass": _CONTROLLED_PHYSICS_STEPS == 1, "checks": step_attestation_checks},
        )
        if _CONTROLLED_PHYSICS_STEPS != 1:
            raise RuntimeError(f"D366 controlled-step attestation STOP: {step_attestation_checks}")

        post_step = _read_checkpoint(
            "post_step_pre_forward", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        post_step_pose = post_step["independent_physx_tensor_view"]["pose_wxyz"]
        post_step_array = np.asarray(post_step_pose, dtype=np.float64)
        post_step_reference_checks = {
            "seven_finite_values": post_step_array.shape == (7,) and bool(np.all(np.isfinite(post_step_array))),
            "normalized_quaternion": abs(float(np.linalg.norm(post_step_array[3:7])) - 1.0) <= 1.0e-4,
            "distinct_from_baseline": not _pose_error(post_step_pose, baseline_pose)["match"],
            "getter_no_advance": post_step["clock_unchanged_across_getter"],
            "post_step_clock_exact": _clock_no_advance(step_after, post_step["passive_after_physx_getter"]["clock"]),
        }
        if not all(post_step_reference_checks.values()):
            raise RuntimeError(f"D366 post-step reference invalid STOP: {post_step_reference_checks}")
        post_step_bits = post_step["independent_physx_tensor_view"]["pose_wxyz_float32_bits"]
        post_step_root = root_matrix_from_pose(post_step_pose)
        post_step_mesh = root_to_mesh_relative @ np.asarray(
            post_step_root["matrix_row_major"], dtype=np.float64
        )
        expected_geometry.update(
            {
                "post_step_authority_pending": False,
                "post_step_pose_wxyz": post_step_pose,
                "post_step_root_matrix_row_major": post_step_root["matrix_row_major"],
                "post_step_mesh_matrix_row_major": post_step_mesh.tolist(),
            }
        )

        _marker("worker", "explicit_simulation_context_forward", "start")
        forward_before = _clock_snapshot(inner, timeline)
        _EXPLICIT_FORWARD_COUNT += 1
        inner.sim.forward()
        _EXPLICIT_FORWARD_RETURNED = True
        forward_after = _clock_snapshot(inner, timeline)
        forward_guard = {
            "forward_count_1": _EXPLICIT_FORWARD_COUNT == 1,
            "forward_returned": _EXPLICIT_FORWARD_RETURNED is True,
            "playing_before_after": forward_before["timeline_playing"] and forward_after["timeline_playing"],
            "physics_clock_callback_unchanged": _clock_no_advance(forward_before, forward_after),
        }
        _marker("worker", "explicit_simulation_context_forward", "complete", {"pass": all(forward_guard.values())})
        if not all(forward_guard.values()):
            raise RuntimeError(f"D366 public forward side-effect STOP: {forward_guard}")
        post_forward = _read_checkpoint(
            "post_forward_pre_pause", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        post_forward_physx_stable = (
            post_forward["independent_physx_tensor_view"]["pose_wxyz_float32_bits"] == post_step_bits
        )
        if not post_forward_physx_stable:
            raise RuntimeError("D366 post-forward PhysX reference changed STOP")

        _marker("worker", "terminal_timeline_pause", "start")
        terminal_pause = _pause_no_advance(inner, timeline)
        _marker("worker", "terminal_timeline_pause", "complete", {"pass": terminal_pause["pass"]})
        if not terminal_pause["pass"]:
            raise RuntimeError(f"D366 terminal pause side-effect STOP: {terminal_pause['checks']}")
        post_pause = _read_checkpoint(
            "post_pause_pre_capture", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        post_pause_state_checks = {
            "physx_bits_unchanged": post_pause["independent_physx_tensor_view"]["pose_wxyz_float32_bits"]
            == post_step_bits,
            "root_current_unchanged": matrices_match(
                fabric_matrix(post_forward, "root", "hierarchy_current_computed_world_matrix"),
                fabric_matrix(post_pause, "root", "hierarchy_current_computed_world_matrix"),
            ),
            "mesh_current_unchanged": matrices_match(
                fabric_matrix(post_forward, "mesh", "hierarchy_current_computed_world_matrix"),
                fabric_matrix(post_pause, "mesh", "hierarchy_current_computed_world_matrix"),
            ),
            "mesh_cache_unchanged": matrices_match(
                fabric_matrix(post_forward, "mesh", "hierarchy_cached_world_matrix"),
                fabric_matrix(post_pause, "mesh", "hierarchy_cached_world_matrix"),
            ),
        }
        if not all(post_pause_state_checks.values()):
            raise RuntimeError(f"D366 post-pause state changed STOP: {post_pause_state_checks}")
        final_captures = _capture_phase("post_step_forward", simulation_app, inner, timeline)
        terminal = _read_checkpoint(
            "post_step_forward_terminal", inner, timeline, usd_stage, rt_stage,
            hierarchy_interface, include_physx_getter=True,
        )
        terminal_physx_stable = (
            terminal["independent_physx_tensor_view"]["pose_wxyz_float32_bits"] == post_step_bits
        )
        final_clock = _clock_snapshot(inner, timeline)
        counter_checks = {
            "exact_call_counter_contract": _counter_contract(
                _DISPLAY_STATE_WRITE_COUNT,
                _CONTROLLED_STEP_CALL_COUNT,
                _EXPLICIT_FORWARD_COUNT,
                _PHYSICS_CALLBACK_COUNT,
                _Q5_SCIENCE_SAMPLE_COUNT,
                _Q5_TARGET_UPDATE_COUNT,
                _CONTACT_QUERY_COUNT,
            ),
            "write_returned": _DISPLAY_STATE_WRITE_RETURNED is True,
            "step_returned_and_attested_1": _CONTROLLED_STEP_RETURNED is True
            and _CONTROLLED_PHYSICS_STEPS == 1,
            "forward_returned": _EXPLICIT_FORWARD_RETURNED is True,
            "callback_registered_terminal": callback_registered
            and inner.sim.physics_callback_exists(_CALLBACK_NAME),
            "callback_dt_exact_terminal": len(_PHYSICS_CALLBACK_DTS) == 1
            and abs(_PHYSICS_CALLBACK_DTS[0] - PHYSICS_DT_S) <= TIME_TOL_S,
            "step_attestation_pass": all(step_attestation_checks.values()),
            "forward_guard_pass": all(forward_guard.values()),
            "terminal_pause_pass": terminal_pause["pass"],
            "poststep_physx_stable": post_forward_physx_stable and terminal_physx_stable,
            "pause_state_unchanged": all(post_pause_state_checks.values()),
            "all_getter_guards": all(
                checkpoint["clock_unchanged_across_getter"]
                for checkpoint in (
                    baseline_checkpoint, baseline_after_capture, post_play, post_write,
                    post_step, post_forward, post_pause, terminal,
                )
            ),
            "final_clock_equals_post_step_except_mode": _physics_state_no_advance(step_after, final_clock),
            "timeline_paused_not_stopped": not timeline.is_playing() and not timeline.is_stopped(),
            "staged_continuity_unchanged": _git_cached_patch() == prereg["git_cached_patch"],
            "d362_unchanged": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
            "d363_unchanged": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
            "d364_unchanged": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
            "d365_unchanged": _tree_manifest(D365_DIR) == prereg["d365_manifest_before"],
            "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
            "inputs_unchanged": _input_hashes() == prereg["input_hashes"],
        }
        summary = {
            "artifact": "D366_WORKER_SUMMARY_V1",
            "case": CASE,
            "utc": _utc_now(),
            "new_variables": NEW_VARIABLES,
            "target_row": {
                "row_index": TARGET_ROW_INDEX,
                "global_step": target_row["global_step"],
                "object_pos_w_m": target_row["object_pos_w_m"],
                "object_quat_wxyz": target_row["object_quat_wxyz"],
                "object_disp_xy_mm": target_row["object_disp_xy_mm"],
                "object_tilt_delta_deg": target_row["object_tilt_delta_from_reference_deg"],
            },
            "write": write,
            "write_guard": write_guard,
            "initial_pause": initial_pause,
            "play_guard": play_guard,
            "post_play_checks": post_play_checks,
            "post_write_checks": post_write_checks,
            "post_write_required_checks": post_write_required_checks,
            "callback_registration_checks": callback_registration_checks,
            "step_attestation": {
                "before": step_before,
                "after": step_after,
                "physics_dt_s": physics_dt,
                "step_index_delta": step_index_delta,
                "step_time_delta_s": step_time_delta,
                "checks": step_attestation_checks,
                "pass": all(step_attestation_checks.values()),
            },
            "post_step_reference_checks": post_step_reference_checks,
            "forward_guard": {"before": forward_before, "after": forward_after, "checks": forward_guard, "pass": all(forward_guard.values())},
            "terminal_pause": terminal_pause,
            "post_pause_state_checks": post_pause_state_checks,
            "fabric_attestation": fabric,
            "expected_geometry": expected_geometry,
            "checkpoints": {
                "baseline_pre_capture": baseline_checkpoint,
                "baseline_post_capture": baseline_after_capture,
                "post_play_pre_write": post_play,
                "post_write_pre_step": post_write,
                "post_step_pre_forward": post_step,
                "post_forward_pre_pause": post_forward,
                "post_pause_pre_capture": post_pause,
                "post_step_forward_terminal": terminal,
            },
            "captures": {"baseline": baseline_captures, "post_step_forward": final_captures},
            "final_clock": final_clock,
            "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
            "display_state_write_returned": _DISPLAY_STATE_WRITE_RETURNED,
            "controlled_step_call_count": _CONTROLLED_STEP_CALL_COUNT,
            "controlled_step_returned": _CONTROLLED_STEP_RETURNED,
            "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
            "physics_callback_count": _PHYSICS_CALLBACK_COUNT,
            "physics_callback_dts": list(_PHYSICS_CALLBACK_DTS),
            "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
            "explicit_forward_returned": _EXPLICIT_FORWARD_RETURNED,
            "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
            "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
            "contact_query_count": _CONTACT_QUERY_COUNT,
            "counter_checks": counter_checks,
            "target_ik_path_changed": False,
            "physics_or_renderer_settings_changed": False,
            "contact_occurred_unknown_because_not_queried": True,
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
                    "artifact": "D366_WORKER_EXCEPTION_STOP_V1",
                    "case": CASE,
                    "utc": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "display_state_write_count": _DISPLAY_STATE_WRITE_COUNT,
                    "display_state_write_returned": _DISPLAY_STATE_WRITE_RETURNED,
                    "controlled_step_call_count": _CONTROLLED_STEP_CALL_COUNT,
                    "controlled_step_returned": _CONTROLLED_STEP_RETURNED,
                    "controlled_physics_steps": _CONTROLLED_PHYSICS_STEPS,
                    "physics_callback_count": _PHYSICS_CALLBACK_COUNT,
                    "physics_callback_dts": list(_PHYSICS_CALLBACK_DTS),
                    "explicit_forward_count": _EXPLICIT_FORWARD_COUNT,
                    "explicit_forward_returned": _EXPLICIT_FORWARD_RETURNED,
                    "q5_science_sample_count": _Q5_SCIENCE_SAMPLE_COUNT,
                    "q5_target_update_count": _Q5_TARGET_UPDATE_COUNT,
                    "contact_query_count": _CONTACT_QUERY_COUNT,
                    "final_verdict": "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP",
                    "g0a_pass": False,
                    "pass": False,
                },
            )
        try:
            _marker("worker", "worker_exception", "stop", {"error": f"{type(error).__name__}: {error}"})
        except Exception:
            pass
        return 2
    finally:
        try:
            if callback_registered and inner is not None and inner.sim.physics_callback_exists(_CALLBACK_NAME):
                inner.sim.remove_physics_callback(_CALLBACK_NAME)
        except Exception:
            pass
        try:
            if settings is not None and previous_play is not None:
                settings.set(PLAY_SIMULATIONS_SETTING, previous_play)
        except Exception:
            pass
        try:
            if inner is not None:
                inner.close()
        except Exception:
            pass
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass


def _audit_layer_journal() -> dict[str, Any]:
    expected_labels = list(CHECKPOINT_LABELS)
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    errors: list[str] = []
    with LAYER_JOURNAL_PATH.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: {error}")
                continue
            claimed = row.get("record_sha256")
            body = dict(row)
            body.pop("record_sha256", None)
            calculated = hashlib.sha256(_canonical_bytes(body)).hexdigest()
            if row.get("sequence") != line_number:
                errors.append(f"sequence mismatch line {line_number}")
            if row.get("previous_sha256") != previous:
                errors.append(f"previous hash mismatch line {line_number}")
            if claimed != calculated:
                errors.append(f"record hash mismatch line {line_number}")
            previous = str(claimed)
            rows.append(row)
    labels = [str(row.get("label")) for row in rows]
    checks = {
        "eight_records": len(rows) == 8,
        "labels_exact": labels == expected_labels,
        "hash_chain_exact": not errors,
        "all_getter_clock_guards_true": all(
            row.get("payload", {}).get("clock_unchanged_across_getter") is True for row in rows
        ),
    }
    audit = {
        "artifact": "D366_LAYER_READBACK_JOURNAL_AUDIT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "journal_path": _rel(LAYER_JOURNAL_PATH),
        "journal_sha256": _sha(LAYER_JOURNAL_PATH),
        "record_count": len(rows),
        "labels": labels,
        "terminal_record_sha256": previous,
        "errors": errors,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(LAYER_AUDIT_PATH, audit)
    return audit


def _journal_worker_checkpoint_match(worker: dict[str, Any]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in LAYER_JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_labels = list(CHECKPOINT_LABELS)
    observed_labels = [row.get("label") for row in rows]
    payload_matches = {
        label: (
            index < len(rows)
            and rows[index].get("label") == label
            and _canonical_bytes(rows[index].get("payload"))
            == _canonical_bytes(worker["checkpoints"][label])
        )
        for index, label in enumerate(expected_labels)
    }
    checks = {
        "labels_exact": observed_labels == expected_labels,
        "payloads_exact": bool(payload_matches) and all(payload_matches.values()),
    }
    return {
        "journal_sha256": _sha(LAYER_JOURNAL_PATH),
        "worker_summary_sha256": _sha(WORKER_SUMMARY_PATH),
        "observed_labels": observed_labels,
        "expected_labels": expected_labels,
        "payload_matches": payload_matches,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _png_metrics(path: Path) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"D366 PNG decode failed: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MASK_HSV_LOW, MASK_HSV_HIGH)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError(f"D366 yellow component absent: {path}")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    component = (labels == index).astype(np.uint8)
    y, x = np.nonzero(component)
    points = np.column_stack((x, y)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    angle_deg = float(math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0)
    bbox = [
        int(stats[index, cv2.CC_STAT_LEFT]),
        int(stats[index, cv2.CC_STAT_TOP]),
        int(stats[index, cv2.CC_STAT_WIDTH]),
        int(stats[index, cv2.CC_STAT_HEIGHT]),
    ]
    width = bbox[2]
    height = bbox[3]
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "area_px": area,
        "bbox_xywh": bbox,
        "centroid_xy_px": [float(centroids[index][0]), float(centroids[index][1])],
        "pca_axis_angle_deg": angle_deg,
        "upright": bool(height / max(width, 1) >= UPRIGHT_HW_MIN),
        "toppled": bool(width / max(height, 1) >= TOPPLED_WH_MIN),
        "component_mask": component,
        "pass": image.shape[1] == VIEWPORT_SIZE[0] and image.shape[0] == VIEWPORT_SIZE[1] and area >= MASK_MIN_AREA_PX,
    }


def _axis_delta_deg(a: float, b: float) -> float:
    raw = abs(float(a) - float(b)) % 180.0
    return min(raw, 180.0 - raw)


def _compare_masks(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    centroid = float(np.linalg.norm(np.asarray(a["centroid_xy_px"]) - np.asarray(b["centroid_xy_px"])))
    axis = _axis_delta_deg(a["pca_axis_angle_deg"], b["pca_axis_angle_deg"])
    mask_a = a["component_mask"].astype(bool)
    mask_b = b["component_mask"].astype(bool)
    union = int(np.logical_or(mask_a, mask_b).sum())
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    iou = float(intersection / union) if union else 1.0
    criteria = {
        "centroid_delta_ge_15px": centroid >= MATERIAL_CENTROID_DELTA_PX,
        "axis_delta_ge_15deg": axis >= MATERIAL_AXIS_DELTA_DEG,
        "iou_le_0p85": iou <= MATERIAL_IOU_MAX,
    }
    return {
        "centroid_delta_px": centroid,
        "axis_delta_deg": axis,
        "mask_iou": iou,
        "criteria": criteria,
        "materially_different": sum(bool(value) for value in criteria.values()) >= 2,
    }


def _public_png_metrics(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "component_mask"}


def _phase_layers_legacy_unused(
    checkpoint: dict[str, Any],
    baseline_pose: list[float],
    target_pose: list[float],
    baseline_root_matrix: list[list[float]],
    target_root_matrix: list[list[float]],
    baseline_mesh_matrix: list[list[float]],
    target_mesh_matrix: list[list[float]],
) -> dict[str, Any]:
    passive_before = checkpoint["passive_before_physx_getter"]
    passive_after = checkpoint["passive_after_physx_getter"]
    physx = checkpoint["independent_physx_tensor_view"]

    def fabric_matrix(passive: dict[str, Any], node: str, field: str) -> list[list[float]] | None:
        record = passive["fabric_usdrt"][node][field]
        if field == "hierarchy_current_computed_world_matrix":
            return record["matrix_row_major"] if record is not None else None
        value = record.get("value") if isinstance(record, dict) else None
        return value.get("matrix_row_major") if value is not None else None

    def fabric_compatibility_pose(passive: dict[str, Any], node: str) -> list[float] | None:
        record = passive["fabric_usdrt"][node]
        position = record["world_position"].get("value")
        orientation = record["world_orientation"].get("value")
        if position is None or orientation is None:
            return None
        return [*[float(item) for item in position], *[float(item) for item in orientation]]

    root_before = fabric_matrix(passive_before, "root", "hierarchy_current_computed_world_matrix")
    root_after = fabric_matrix(passive_after, "root", "hierarchy_current_computed_world_matrix")
    root_cached = fabric_matrix(passive_after, "root", "hierarchy_cached_world_matrix")
    mesh_before = fabric_matrix(passive_before, "mesh", "hierarchy_current_computed_world_matrix")
    mesh_after = fabric_matrix(passive_after, "mesh", "hierarchy_current_computed_world_matrix")
    mesh_cached_before = fabric_matrix(passive_before, "mesh", "hierarchy_cached_world_matrix")
    mesh_cached = fabric_matrix(passive_after, "mesh", "hierarchy_cached_world_matrix")
    cache_pose = passive_after["assetdata_cache"].get("pose_wxyz")
    physx_pose = physx.get("pose_wxyz") if physx is not None else None
    root_compatibility_pose = fabric_compatibility_pose(passive_after, "root")
    fabric_before_sha = hashlib.sha256(
        _canonical_bytes(passive_before["fabric_usdrt"])
    ).hexdigest()
    fabric_after_sha = hashlib.sha256(
        _canonical_bytes(passive_after["fabric_usdrt"])
    ).hexdigest()
    getter_transition = {
        "root_current_presence_changed": (root_before is None) != (root_after is None),
        "mesh_current_presence_changed": (mesh_before is None) != (mesh_after is None),
        "mesh_cached_presence_changed": (mesh_cached_before is None) != (mesh_cached is None),
        "root_current_matrix_max_abs": None if root_before is None or root_after is None else float(
            np.max(np.abs(np.asarray(root_after) - np.asarray(root_before)))
        ),
        "mesh_current_matrix_max_abs": None if mesh_before is None or mesh_after is None else float(
            np.max(np.abs(np.asarray(mesh_after) - np.asarray(mesh_before)))
        ),
        "mesh_cached_matrix_max_abs": None if mesh_cached_before is None or mesh_cached is None else float(
            np.max(np.abs(np.asarray(mesh_cached) - np.asarray(mesh_cached_before)))
        ),
        "fabric_full_snapshot_before_sha256": fabric_before_sha,
        "fabric_full_snapshot_after_sha256": fabric_after_sha,
        "optional_full_fabric_snapshot_changed": fabric_before_sha != fabric_after_sha,
    }
    required_presence_changed = any(
        value is True
        for key, value in getter_transition.items()
        if key.endswith("presence_changed")
    )
    required_matrix_changed = any(
        value is not None and value > MATRIX_TOL
        for key, value in getter_transition.items()
        if key.endswith("max_abs")
    )
    getter_transition["required_presence_changed"] = required_presence_changed
    getter_transition["required_matrix_changed"] = required_matrix_changed
    getter_transition["required_fabric_changed_across_physx_getter"] = (
        required_presence_changed or required_matrix_changed
    )
    return {
        "cache": _classify_pose(cache_pose, baseline_pose, target_pose),
        "physx_tensor_view": _classify_pose(physx_pose, baseline_pose, target_pose),
        "fabric_root_compatibility": _classify_pose(
            root_compatibility_pose, baseline_pose, target_pose
        ),
        "fabric_root_current": _classify_matrix(root_after, baseline_root_matrix, target_root_matrix),
        "fabric_root_cached_render": _classify_matrix(root_cached, baseline_root_matrix, target_root_matrix),
        "fabric_mesh_current": _classify_matrix(mesh_after, baseline_mesh_matrix, target_mesh_matrix),
        "fabric_mesh_cached_render": _classify_matrix(mesh_cached, baseline_mesh_matrix, target_mesh_matrix),
        "getter_transition": getter_transition,
        "authored_usd_control": passive_after["authored_usd"],
        "raw_fabric": passive_after["fabric_usdrt"],
    }


def _build_localization_report_legacy_unused(worker: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_checkpoint = worker["checkpoints"]["baseline_post_capture"]
    baseline_pose = baseline_checkpoint["independent_physx_tensor_view"]["pose_wxyz"]
    target_pose = worker["write"]["target_pose_wxyz"]
    geometry = worker["expected_geometry"]
    baseline_root_matrix = geometry["baseline_root_computed"]["matrix_row_major"]
    target_root_matrix = geometry["target_root_matrix"]["matrix_row_major"]
    baseline_mesh_matrix = geometry["baseline_mesh_computed"]["matrix_row_major"]
    target_mesh_matrix = geometry["target_mesh_matrix_row_major"]
    phase_layers = {
        label: _phase_layers_legacy_unused(
            checkpoint,
            baseline_pose,
            target_pose,
            baseline_root_matrix,
            target_root_matrix,
            baseline_mesh_matrix,
            target_mesh_matrix,
        )
        for label, checkpoint in worker["checkpoints"].items()
    }

    private_png: dict[str, dict[str, dict[str, Any]]] = {}
    public_png: dict[str, dict[str, dict[str, Any]]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for phase in CAPTURE_PATHS:
        private_png[phase] = {}
        public_png[phase] = {}
        for view, path in CAPTURE_PATHS[phase].items():
            metric = _png_metrics(path)
            private_png[phase][view] = metric
            public_png[phase][view] = _public_png_metrics(metric)
    for view in CAMERA_EYES:
        comparisons[view] = {
            "baseline_to_post_write_no_forward": _compare_masks(
                private_png["baseline"][view], private_png["post_write_no_forward"][view]
            ),
            "baseline_to_post_forward": _compare_masks(
                private_png["baseline"][view], private_png["post_forward"][view]
            ),
            "post_write_no_forward_to_post_forward": _compare_masks(
                private_png["post_write_no_forward"][view], private_png["post_forward"][view]
            ),
        }
    hydra_phase_views: dict[str, dict[str, dict[str, Any]]] = {}
    hydra_phase_classes: dict[str, str] = {}
    for phase in CAPTURE_PATHS:
        hydra_phase_views[phase] = {}
        for view in CAMERA_EYES:
            if phase == "baseline":
                materially_different = False
            else:
                comparison_key = (
                    "baseline_to_post_write_no_forward"
                    if phase == "post_write_no_forward"
                    else "baseline_to_post_forward"
                )
                materially_different = comparisons[view][comparison_key]["materially_different"]
            toppled = public_png[phase][view]["toppled"]
            upright = public_png[phase][view]["upright"]
            if phase != "baseline" and materially_different and toppled:
                view_class = "TARGET"
            elif not materially_different and upright:
                view_class = "BASELINE"
            else:
                view_class = "OTHER"
            hydra_phase_views[phase][view] = {
                "materially_different_from_baseline": materially_different,
                "toppled": toppled,
                "upright": upright,
                "class": view_class,
            }
        classes = [row["class"] for row in hydra_phase_views[phase].values()]
        hydra_phase_classes[phase] = (
            "TARGET" if all(value == "TARGET" for value in classes)
            else "BASELINE" if all(value == "BASELINE" for value in classes)
            else "OTHER"
        )
    hydra_views = hydra_phase_views["post_forward"]
    hydra_class = hydra_phase_classes["post_forward"]

    terminal = phase_layers["post_forward_after_app_update"]
    terminal_classes = {
        "cache": terminal["cache"]["class"],
        "physx": terminal["physx_tensor_view"]["class"],
        "root_compatibility": terminal["fabric_root_compatibility"]["class"],
        "root_current": terminal["fabric_root_current"]["class"],
        "root_cached": terminal["fabric_root_cached_render"]["class"],
        "mesh_current": terminal["fabric_mesh_current"]["class"],
        "mesh_cached": terminal["fabric_mesh_cached_render"]["class"],
        "hydra": hydra_class,
    }
    linear_terminal_names = [
        "cache",
        "physx",
        "root_current",
        "mesh_current",
        "mesh_cached",
        "hydra",
    ]
    linear_terminal_classes = {
        key: terminal_classes[key] for key in linear_terminal_names
    }
    terminal_layers_available = all(
        value != "UNAVAILABLE" for value in linear_terminal_classes.values()
    )
    terminal_classes_binary = set(linear_terminal_classes.values()) <= {"BASELINE", "TARGET"}
    phase_order = [
        "baseline_pre_capture",
        "baseline_post_capture",
        "post_write_immediate",
        "post_write_after_app_update",
        "post_forward_immediate",
        "post_forward_after_app_update",
    ]
    layer_keys = [
        "cache",
        "physx_tensor_view",
        "fabric_root_current",
        "fabric_mesh_current",
        "fabric_mesh_cached_render",
    ]
    diagnostic_layer_keys = ["fabric_root_compatibility", "fabric_root_cached_render"]
    temporal_classes = {
        layer: [phase_layers[phase][layer]["class"] for phase in phase_order]
        for layer in layer_keys
    }
    hydra_temporal_classes = [
        hydra_phase_classes[phase]
        for phase in ("baseline", "post_write_no_forward", "post_forward")
    ]
    diagnostic_temporal_classes = {
        layer: [phase_layers[phase][layer]["class"] for phase in phase_order]
        for layer in diagnostic_layer_keys
    }
    baseline_layers_exact = all(
        temporal_classes[layer][0:2] == ["BASELINE", "BASELINE"]
        for layer in layer_keys
    ) and hydra_temporal_classes[0] == "BASELINE"
    temporal_regressions: dict[str, bool] = {}
    temporal_other: dict[str, bool] = {}
    for layer, values in temporal_classes.items():
        audit = _temporal_class_audit(values)
        temporal_regressions[layer] = audit["regression"]
        temporal_other[layer] = audit["other"]
    hydra_temporal_audit = _temporal_class_audit(hydra_temporal_classes)
    temporal_regressions["hydra"] = hydra_temporal_audit["regression"]
    temporal_other["hydra"] = hydra_temporal_audit["other"]
    any_temporal_regression = any(temporal_regressions.values())
    any_temporal_other = any(temporal_other.values())
    checkpoint_hydra_phase = {
        "baseline_post_capture": "baseline",
        "post_write_after_app_update": "post_write_no_forward",
        "post_forward_after_app_update": "post_forward",
    }
    phase_linear_classes: dict[str, list[str]] = {}
    phase_downstream_ahead: dict[str, bool] = {}
    for phase in phase_order:
        values = [phase_layers[phase][layer]["class"] for layer in layer_keys]
        hydra_phase = checkpoint_hydra_phase.get(phase)
        if hydra_phase is not None:
            values.append(hydra_phase_classes[hydra_phase])
        phase_linear_classes[phase] = values
        phase_downstream_ahead[phase] = _downstream_ahead(values)
    any_phase_downstream_ahead = any(phase_downstream_ahead.values())
    getter_side_effect_detected = any(
        phase["getter_transition"]["required_fabric_changed_across_physx_getter"]
        for phase in phase_layers.values()
    )
    fabric = worker["fabric_attestation"]
    prerequisites = {
        "worker_counter_contract_pass": worker["pass"] is True,
        "fabric_enabled_and_delegate_attested": fabric["cfg_use_fabric"] is True
        and fabric["is_fabric_enabled"] is True
        and fabric["interface_present"] is True
        and fabric["app_use_fabric_scene_delegate"] is True
        and fabric["hydra_reads_transforms_from_fabric"] is True,
        "selected_force_update": fabric["selected_callable_name"] == "force_update"
        and fabric["selected_callable_bound_to_interface"] is True,
        "all_png_metrics_pass": all(
            metric["pass"] for phase in public_png.values() for metric in phase.values()
        ),
        "terminal_layers_available": terminal_layers_available,
        "no_getter_clock_advance": all(
            checkpoint["clock_unchanged_across_getter"] for checkpoint in worker["checkpoints"].values()
        ),
        "baseline_root_mesh_compose_pass": geometry["baseline_reconstruction_max_abs"] <= MATRIX_TOL,
    }

    if not all(prerequisites.values()):
        verdict = "D366_MEASUREMENT_INCOMPLETE_FAIL_STOP"
    elif not terminal_classes_binary:
        verdict = "D366_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    elif (
        getter_side_effect_detected
        or not baseline_layers_exact
        or any_temporal_regression
        or any_temporal_other
        or any_phase_downstream_ahead
    ):
        verdict = "D366_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    elif terminal_classes["cache"] != "TARGET":
        verdict = "D366_DIRECT_WRITE_OR_CACHE_FAIL"
    elif terminal_classes["physx"] != "TARGET":
        verdict = "D366_CACHE_TO_PHYSX_PENDING_OR_FAILED"
    elif terminal_classes["root_current"] != "TARGET":
        verdict = "D366_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED"
    elif terminal_classes["mesh_current"] != "TARGET":
        verdict = "D366_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED"
    elif terminal_classes["mesh_cached"] != "TARGET":
        verdict = "D366_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED"
    elif terminal_classes["hydra"] != "TARGET":
        verdict = "D366_FABRIC_TO_HYDRA_NOT_PROPAGATED"
    else:
        verdict = "D366_END_TO_END_ZERO_STEP_VISIBLE"

    nonmonotonic = any_phase_downstream_ahead

    supporting = {
        "physx_became_target_immediately_after_write": phase_layers["post_write_immediate"]["physx_tensor_view"]["class"] == "TARGET",
        "root_current_became_target_after_app_without_forward": phase_layers["post_write_after_app_update"]["fabric_root_current"]["class"] == "TARGET",
        "root_current_became_target_after_forward": phase_layers["post_forward_immediate"]["fabric_root_current"]["class"] == "TARGET",
        "root_cached_became_target_only_after_post_forward_app": phase_layers["post_forward_immediate"]["fabric_root_cached_render"]["class"] != "TARGET"
        and phase_layers["post_forward_after_app_update"]["fabric_root_cached_render"]["class"] == "TARGET",
        "optional_compatibility_terminal_class": terminal_classes["root_compatibility"],
        "optional_root_cached_terminal_class": terminal_classes["root_cached"],
        "hydra_became_target_before_public_forward": hydra_phase_classes["post_write_no_forward"] == "TARGET",
        "any_fabric_change_across_physx_getter": getter_side_effect_detected,
    }
    first_target_phase = {
        layer: next(
            (phase for phase in phase_order if phase_layers[phase][layer]["class"] == "TARGET"),
            None,
        )
        for layer in layer_keys
    }
    first_target_phase["hydra"] = next(
        (
            phase for phase in ("baseline", "post_write_no_forward", "post_forward")
            if hydra_phase_classes[phase] == "TARGET"
        ),
        None,
    )
    operational_localization_complete = verdict not in {
        "D366_MEASUREMENT_INCOMPLETE_FAIL_STOP",
        "D366_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP",
    }
    report = {
        "artifact": "D366_STATE_LAYER_LOCALIZATION_REPORT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "purpose": "localize paused zero-step cylinder pose visibility break; no physics/q5/contact science",
        "target_row_index": TARGET_ROW_INDEX,
        "target_global_step": EXPECTED_GLOBAL_STEP,
        "baseline_pose_wxyz": baseline_pose,
        "target_pose_wxyz": target_pose,
        "baseline_target_pose_separation": {
            "position_and_rotation": _pose_error(baseline_pose, target_pose),
            "trace_xy_displacement_mm": worker["target_row"]["object_disp_xy_mm"],
            "trace_tilt_delta_deg": worker["target_row"]["object_tilt_delta_deg"],
        },
        "fabric_attestation": fabric,
        "phase_layers": phase_layers,
        "pixel_metrics": public_png,
        "pixel_comparisons": comparisons,
        "hydra_views": hydra_views,
        "hydra_phase_views": hydra_phase_views,
        "hydra_phase_classes": hydra_phase_classes,
        "terminal_classes": terminal_classes,
        "linear_terminal_classes": linear_terminal_classes,
        "diagnostic_terminal_classes": {
            "root_compatibility": terminal_classes["root_compatibility"],
            "root_cached": terminal_classes["root_cached"],
        },
        "temporal_classes": temporal_classes,
        "hydra_temporal_classes": hydra_temporal_classes,
        "diagnostic_temporal_classes": diagnostic_temporal_classes,
        "diagnostic_layer_keys": diagnostic_layer_keys,
        "baseline_layers_exact": baseline_layers_exact,
        "temporal_regressions": temporal_regressions,
        "temporal_other": temporal_other,
        "getter_side_effect_detected": getter_side_effect_detected,
        "nonmonotonic_downstream_target_after_upstream_baseline": nonmonotonic,
        "phase_linear_classes": phase_linear_classes,
        "phase_downstream_ahead": phase_downstream_ahead,
        "supporting_phase_findings": supporting,
        "first_target_phase": first_target_phase,
        "prerequisites": prerequisites,
        "localization_verdict": verdict,
        "operational_localization_complete": operational_localization_complete,
        "controlled_physics_steps": worker["controlled_physics_steps"],
        "q5_science_sample_count": worker["q5_science_sample_count"],
        "q5_target_update_count": worker["q5_target_update_count"],
        "contact_query_count": worker["contact_query_count"],
        "authored_usd_is_dynamic_authority": False,
        "physx_getter_semantics": "independent tensor-view readback; solver-internal commit not separately attested",
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "pass": operational_localization_complete,
    }
    _write_json_x(REPORT_PATH, report)
    return report, private_png


def _phase_layers(
    checkpoint: dict[str, Any],
    baseline_pose: list[float],
    commanded_pose: list[float],
    post_step_pose: list[float],
    baseline_root: list[list[float]],
    commanded_root: list[list[float]],
    post_step_root: list[list[float]],
    baseline_mesh: list[list[float]],
    commanded_mesh: list[list[float]],
    post_step_mesh: list[list[float]],
) -> dict[str, Any]:
    passive_before = checkpoint["passive_before_physx_getter"]
    passive_after = checkpoint["passive_after_physx_getter"]
    physx = checkpoint["independent_physx_tensor_view"]

    def fabric_matrix(passive: dict[str, Any], node: str, field: str) -> list[list[float]] | None:
        record = passive["fabric_usdrt"][node][field]
        if field == "hierarchy_current_computed_world_matrix":
            return record.get("matrix_row_major") if record is not None else None
        value = record.get("value") if isinstance(record, dict) else None
        return value.get("matrix_row_major") if value is not None else None

    def compatibility_pose(passive: dict[str, Any]) -> list[float] | None:
        record = passive["fabric_usdrt"]["root"]
        position = record["world_position"].get("value")
        orientation = record["world_orientation"].get("value")
        if position is None or orientation is None:
            return None
        return [*[float(item) for item in position], *[float(item) for item in orientation]]

    root_before = fabric_matrix(passive_before, "root", "hierarchy_current_computed_world_matrix")
    root_after = fabric_matrix(passive_after, "root", "hierarchy_current_computed_world_matrix")
    root_cached = fabric_matrix(passive_after, "root", "hierarchy_cached_world_matrix")
    mesh_before = fabric_matrix(passive_before, "mesh", "hierarchy_current_computed_world_matrix")
    mesh_after = fabric_matrix(passive_after, "mesh", "hierarchy_current_computed_world_matrix")
    mesh_cached_before = fabric_matrix(passive_before, "mesh", "hierarchy_cached_world_matrix")
    mesh_cached = fabric_matrix(passive_after, "mesh", "hierarchy_cached_world_matrix")
    getter_transition = {
        "root_current_presence_changed": (root_before is None) != (root_after is None),
        "mesh_current_presence_changed": (mesh_before is None) != (mesh_after is None),
        "mesh_cached_presence_changed": (mesh_cached_before is None) != (mesh_cached is None),
        "root_current_max_abs": None if root_before is None or root_after is None else float(
            np.max(np.abs(np.asarray(root_before) - np.asarray(root_after)))
        ),
        "mesh_current_max_abs": None if mesh_before is None or mesh_after is None else float(
            np.max(np.abs(np.asarray(mesh_before) - np.asarray(mesh_after)))
        ),
        "mesh_cached_max_abs": None if mesh_cached_before is None or mesh_cached is None else float(
            np.max(np.abs(np.asarray(mesh_cached_before) - np.asarray(mesh_cached)))
        ),
        "fabric_before_sha256": hashlib.sha256(_canonical_bytes(passive_before["fabric_usdrt"])).hexdigest(),
        "fabric_after_sha256": hashlib.sha256(_canonical_bytes(passive_after["fabric_usdrt"])).hexdigest(),
    }
    getter_transition["required_fabric_changed"] = _required_matrix_transition_changed(
        [
            (root_before, root_after),
            (mesh_before, mesh_after),
            (mesh_cached_before, mesh_cached),
        ]
    )
    return {
        "assetdata_cache_diagnostic": _classify_pose(
            passive_after["assetdata_cache"].get("pose_wxyz"),
            baseline_pose, commanded_pose, post_step_pose,
        ),
        "physx_tensor_view": _classify_pose(
            physx.get("pose_wxyz") if physx is not None else None,
            baseline_pose, commanded_pose, post_step_pose,
        ),
        "fabric_root_compatibility_diagnostic": _classify_pose(
            compatibility_pose(passive_after), baseline_pose, commanded_pose, post_step_pose,
        ),
        "fabric_root_current": _classify_matrix(
            root_after, baseline_root, commanded_root, post_step_root,
        ),
        "fabric_root_cached_diagnostic": _classify_matrix(
            root_cached, baseline_root, commanded_root, post_step_root,
        ),
        "fabric_mesh_current": _classify_matrix(
            mesh_after, baseline_mesh, commanded_mesh, post_step_mesh,
        ),
        "fabric_mesh_cached_render": _classify_matrix(
            mesh_cached, baseline_mesh, commanded_mesh, post_step_mesh,
        ),
        "getter_transition": getter_transition,
        "authored_usd_control": passive_after["authored_usd"],
        "raw_fabric": passive_after["fabric_usdrt"],
    }


def _build_localization_report(worker: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = worker["checkpoints"]
    baseline_pose = checkpoints["baseline_post_capture"]["independent_physx_tensor_view"]["pose_wxyz"]
    commanded_pose = worker["write"]["target_pose_wxyz"]
    post_step_pose = checkpoints["post_step_pre_forward"]["independent_physx_tensor_view"]["pose_wxyz"]
    geometry = worker["expected_geometry"]
    phase_layers = {
        label: _phase_layers(
            checkpoint,
            baseline_pose,
            commanded_pose,
            post_step_pose,
            geometry["baseline_root_matrix_row_major"],
            geometry["commanded_root_matrix_row_major"],
            geometry["post_step_root_matrix_row_major"],
            geometry["baseline_mesh_matrix_row_major"],
            geometry["commanded_mesh_matrix_row_major"],
            geometry["post_step_mesh_matrix_row_major"],
        )
        for label, checkpoint in checkpoints.items()
    }

    private_png: dict[str, dict[str, dict[str, Any]]] = {}
    public_png: dict[str, dict[str, dict[str, Any]]] = {}
    for phase, views in CAPTURE_PATHS.items():
        private_png[phase] = {}
        public_png[phase] = {}
        for view, path in views.items():
            metric = _png_metrics(path)
            private_png[phase][view] = metric
            public_png[phase][view] = _public_png_metrics(metric)
    comparisons = {
        view: _compare_masks(
            private_png["baseline"][view], private_png["post_step_forward"][view]
        )
        for view in CAMERA_EYES
    }
    hydra_views: dict[str, dict[str, Any]] = {}
    for view in CAMERA_EYES:
        final_metric = public_png["post_step_forward"][view]
        comparison = comparisons[view]
        if comparison["materially_different"] and final_metric["toppled"]:
            visual_class = "POST_STEP"
        elif not comparison["materially_different"] and final_metric["upright"]:
            visual_class = "BASELINE"
        else:
            visual_class = "OTHER"
        hydra_views[view] = {
            "class": visual_class,
            "materially_different_from_baseline": comparison["materially_different"],
            "toppled": final_metric["toppled"],
            "upright": final_metric["upright"],
        }
    hydra_classes = [row["class"] for row in hydra_views.values()]
    hydra_class = hydra_classes[0] if len(set(hydra_classes)) == 1 else "OTHER"

    terminal = phase_layers["post_step_forward_terminal"]
    terminal_classes = {
        "cache_diagnostic": terminal["assetdata_cache_diagnostic"]["class"],
        "physx": terminal["physx_tensor_view"]["class"],
        "root_compatibility_diagnostic": terminal["fabric_root_compatibility_diagnostic"]["class"],
        "root_current": terminal["fabric_root_current"]["class"],
        "root_cached_diagnostic": terminal["fabric_root_cached_diagnostic"]["class"],
        "mesh_current": terminal["fabric_mesh_current"]["class"],
        "mesh_cached": terminal["fabric_mesh_cached_render"]["class"],
        "hydra": hydra_class,
    }
    post_step_phases = (
        "post_step_pre_forward",
        "post_forward_pre_pause",
        "post_pause_pre_capture",
        "post_step_forward_terminal",
    )
    later_physx_stable = all(
        _is_post_step(phase_layers[label]["physx_tensor_view"]["class"])
        for label in post_step_phases
    )
    getter_side_effect = any(
        layer["getter_transition"]["required_fabric_changed"] for layer in phase_layers.values()
    )
    baseline_expected = all(
        phase_layers[label]["physx_tensor_view"]["class"] == "BASELINE"
        and phase_layers[label]["fabric_root_current"]["class"] == "BASELINE"
        and phase_layers[label]["fabric_mesh_current"]["class"] == "BASELINE"
        and phase_layers[label]["fabric_mesh_cached_render"]["class"] == "BASELINE"
        for label in ("baseline_pre_capture", "baseline_post_capture", "post_play_pre_write")
    )
    pre_step_expected = (
        phase_layers["post_write_pre_step"]["physx_tensor_view"]["class"]
        in {"COMMANDED", "POST_STEP_EQ_COMMANDED"}
        and phase_layers["post_write_pre_step"]["fabric_root_current"]["class"] == "BASELINE"
        and phase_layers["post_write_pre_step"]["fabric_mesh_current"]["class"] == "BASELINE"
        and phase_layers["post_write_pre_step"]["fabric_mesh_cached_render"]["class"] == "BASELINE"
    )
    fabric = worker["fabric_attestation"]
    prerequisites = {
        "worker_control_contract_pass": worker.get("pass") is True,
        "step_attested_exactly_once": worker.get("step_attestation", {}).get("pass") is True
        and worker.get("controlled_physics_steps") == 1
        and worker.get("physics_callback_count") == 1,
        "write_and_forward_exactly_once": worker.get("display_state_write_count") == 1
        and worker.get("display_state_write_returned") is True
        and worker.get("explicit_forward_count") == 1
        and worker.get("explicit_forward_returned") is True,
        "q5_target_contact_zero": worker.get("q5_science_sample_count") == 0
        and worker.get("q5_target_update_count") == 0
        and worker.get("contact_query_count") == 0,
        "fabric_enabled_delegate_force_update": fabric["cfg_use_fabric"] is True
        and fabric["is_fabric_enabled"] is True
        and fabric["selected_callable_name"] == "force_update"
        and fabric["selected_callable_bound_to_interface"] is True
        and fabric["app_use_fabric_scene_delegate"] is True
        and fabric["hydra_reads_transforms_from_fabric"] is True,
        "all_four_png_metrics_pass": all(
            metric["pass"] for phase in public_png.values() for metric in phase.values()
        ),
        "both_baseline_views_upright": all(
            public_png["baseline"][view]["upright"] is True for view in CAMERA_EYES
        ),
        "two_view_visual_class_consistent": len(set(hydra_classes)) == 1
        and hydra_class in {"BASELINE", "POST_STEP"},
        "baseline_contract_exact": baseline_expected,
        "pre_step_contract_exact": pre_step_expected,
        "post_step_physx_reference_stable": later_physx_stable,
        "terminal_numeric_layers_available": all(
            terminal_classes[key] != "UNAVAILABLE"
            for key in ("physx", "root_current", "mesh_current", "mesh_cached")
        ),
        "no_getter_side_effect": not getter_side_effect,
        "all_getter_clock_guards": all(
            checkpoint["clock_unchanged_across_getter"] for checkpoint in checkpoints.values()
        ),
        "post_step_reference_valid": all(worker["post_step_reference_checks"].values()),
    }
    complete = all(prerequisites.values())
    verdict = _decision_fixture(
        terminal_classes["root_current"],
        terminal_classes["mesh_current"],
        terminal_classes["mesh_cached"],
        terminal_classes["hydra"],
        complete=complete,
        step_attested=worker.get("step_attestation", {}).get("pass") is True,
    )
    phase_order = list(CHECKPOINT_LABELS)
    phase_classes = {
        label: {
            "cache_diagnostic": phase_layers[label]["assetdata_cache_diagnostic"]["class"],
            "physx": phase_layers[label]["physx_tensor_view"]["class"],
            "root_current": phase_layers[label]["fabric_root_current"]["class"],
            "mesh_current": phase_layers[label]["fabric_mesh_current"]["class"],
            "mesh_cached": phase_layers[label]["fabric_mesh_cached_render"]["class"],
        }
        for label in phase_order
    }
    first_poststep_fabric_phase = next(
        (
            label for label in post_step_phases
            if _is_post_step(phase_layers[label]["fabric_root_current"]["class"])
        ),
        None,
    )
    operational_complete = verdict != "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP"
    report = {
        "artifact": "D366_TENSOR_STEP_VISIBILITY_REPORT_V1",
        "case": CASE,
        "utc": _utc_now(),
        "purpose": "one controlled PhysX step then one public Fabric forward; observability only",
        "baseline_pose_wxyz": baseline_pose,
        "commanded_pose_wxyz": commanded_pose,
        "post_step_physx_authority_pose_wxyz": post_step_pose,
        "commanded_vs_post_step": _pose_error(commanded_pose, post_step_pose),
        "baseline_vs_post_step": _pose_error(baseline_pose, post_step_pose),
        "phase_layers": phase_layers,
        "phase_classes": phase_classes,
        "pixel_metrics": public_png,
        "pixel_comparisons": comparisons,
        "hydra_views": hydra_views,
        "terminal_classes": terminal_classes,
        "linear_terminal_classes": {
            key: terminal_classes[key]
            for key in ("root_current", "mesh_current", "mesh_cached", "hydra")
        },
        "diagnostic_terminal_classes": {
            key: terminal_classes[key]
            for key in ("cache_diagnostic", "root_compatibility_diagnostic", "root_cached_diagnostic")
        },
        "assetdata_cache_is_linear_verdict_input": False,
        "first_poststep_fabric_phase": first_poststep_fabric_phase,
        "getter_side_effect_detected": getter_side_effect,
        "prerequisites": prerequisites,
        "localization_verdict": verdict,
        "operational_localization_complete": operational_complete,
        "controlled_physics_steps": worker["controlled_physics_steps"],
        "q5_science_sample_count": worker["q5_science_sample_count"],
        "q5_target_update_count": worker["q5_target_update_count"],
        "contact_query_count": worker["contact_query_count"],
        "hydra_semantics": "two-view baseline-change and toppled/upright inspection; not bit-exact 6-DoF authority",
        "physical_contact_occurrence": None,
        "physical_science_recomputed": False,
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "pass": operational_complete,
    }
    _write_json_x(REPORT_PATH, report)
    return report, private_png


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_PATH), size=size)


def _build_sheet_legacy_unused(report: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    width, height = 4800, 3000
    canvas = Image.new("RGB", (width, height), (15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 52), "D366 상태 전달 단절 측정 — 원본 Isaac 화면과 계층별 readback", font=_font(68), fill=(247, 249, 252), anchor="ma")
    phases = [
        ("baseline", "1. baseline"),
        ("post_write_no_forward", "2. pose write 후 · forward 전"),
        ("post_forward", "3. forward 후"),
    ]
    x_positions = [100, 1675, 3250]
    y_positions = {"primary": 180, "opposite": 1040}
    image_size = (1450, 816)
    for column, (phase, title) in enumerate(phases):
        x = x_positions[column]
        draw.text((x + image_size[0] // 2, 145), title, font=_font(40), fill=(144, 206, 255), anchor="ms")
        for view in ("primary", "opposite"):
            y = y_positions[view]
            image = Image.open(CAPTURE_PATHS[phase][view]).convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + image_size[0], y + image_size[1]), outline=(95, 112, 137), width=4)
            draw.text((x + 20, y + 20), view, font=_font(34), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

    classes = report["terminal_classes"]
    rows = [
        ("AssetData cache", classes["cache"]),
        ("독립 PhysX tensor view", classes["physx"]),
        ("compatibility _world* (선택 진단)", classes["root_compatibility"]),
        ("Fabric root current", classes["root_current"]),
        ("Fabric root cached (선택 진단)", classes["root_cached"]),
        ("Fabric mesh current", classes["mesh_current"]),
        ("Fabric mesh cached/render", classes["mesh_cached"]),
        ("Hydra pixels", classes["hydra"]),
    ]
    top = 1930
    draw.rounded_rectangle((100, top, 4700, 2890), radius=28, fill=(25, 34, 49), outline=(78, 99, 128), width=4)
    draw.text((180, top + 55), "최종 단계별 상태", font=_font(48), fill=(245, 192, 92))
    for index, (name, value) in enumerate(rows):
        column = index % 4
        row = index // 4
        x = 180 + column * 1120
        y = top + 150 + row * 150
        color = (82, 224, 143) if value == "TARGET" else (255, 126, 107) if value == "BASELINE" else (255, 201, 92)
        draw.text((x, y), name, font=_font(31), fill=(195, 207, 225))
        draw.text((x, y + 48), value, font=_font(36), fill=color)
    verdict = report["localization_verdict"]
    draw.text((180, top + 525), "판정", font=_font(44), fill=(245, 192, 92))
    draw.text((180, top + 595), verdict, font=_font(46), fill=(247, 249, 252))
    explanation = {
        "D366_DIRECT_WRITE_OR_CACHE_FAIL": "등록된 pose write 뒤 AssetData cache 자체가 final을 보존하지 못했습니다.",
        "D366_CACHE_TO_PHYSX_PENDING_OR_FAILED": "cache에는 final이 있지만 독립 PhysX tensor view에는 전달되지 않았습니다.",
        "D366_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED": "PhysX tensor view에는 final이 있지만 Fabric root hierarchy current에는 전달되지 않았습니다.",
        "D366_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED": "Fabric root는 final이지만 실제 mesh hierarchy에는 전달되지 않았습니다.",
        "D366_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED": "mesh 현재값은 final이지만 renderer cached matrix는 이전값입니다.",
        "D366_FABRIC_TO_HYDRA_NOT_PROPAGATED": "Fabric render mesh는 final이지만 실제 RTX 픽셀은 이전 자세입니다.",
        "D366_END_TO_END_ZERO_STEP_VISIBLE": "이번 1회 측정에서는 zero-step 전달 전 구간이 화면까지 보였습니다.",
    }.get(verdict, "독립 계층 값이 비단조이거나 누락되어 한 화살표로 단정할 수 없습니다.")
    draw.text((180, top + 690), explanation, font=_font(37), fill=(214, 223, 237))
    draw.text(
        (180, top + 795),
        "통제: controlled physics=0 · q5 science/target=0 · contact query=0 · pose write=1 · forward=1",
        font=_font(34),
        fill=(157, 176, 203),
    )
    draw.text((180, top + 860), "이 판정은 렌더 상태 전달 위치만 답하며 grasp/G0a를 판정하지 않습니다.", font=_font(34), fill=(157, 176, 203))
    canvas.save(SHEET_PATH)


def _build_sheet(report: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    width, height = 4800, 3600
    canvas = Image.new("RGB", (width, height), (15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 55),
        "D366 — 1 physics step 뒤 PhysX·Fabric·Hydra 동시점 검사",
        font=_font(68), fill=(247, 249, 252), anchor="ma",
    )
    columns = (("baseline", "1. step 전 baseline"), ("post_step_forward", "2. step 1회 + forward 1회 뒤"))
    x_positions = (100, 2500)
    y_positions = {"primary": 180, "opposite": 1470}
    image_size = (2200, 1238)
    for column_index, (phase, title) in enumerate(columns):
        x = x_positions[column_index]
        draw.text((x + image_size[0] // 2, 145), title, font=_font(42), fill=(144, 206, 255), anchor="ms")
        for view in ("primary", "opposite"):
            y = y_positions[view]
            image = Image.open(CAPTURE_PATHS[phase][view]).convert("RGB").resize(
                image_size, Image.Resampling.LANCZOS
            )
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + image_size[0], y + image_size[1]), outline=(95, 112, 137), width=4)
            draw.text(
                (x + 24, y + 22), view, font=_font(38), fill=(255, 255, 255),
                stroke_width=2, stroke_fill=(0, 0, 0),
            )

    classes = report["terminal_classes"]
    rows = [
        ("PhysX post-step", classes["physx"]),
        ("Fabric root current", classes["root_current"]),
        ("Fabric mesh current", classes["mesh_current"]),
        ("Mesh render cache", classes["mesh_cached"]),
        ("Hydra two-view", classes["hydra"]),
        ("AssetData (진단)", classes["cache_diagnostic"]),
    ]
    top = 2780
    draw.rounded_rectangle((100, top, 4700, 3510), radius=28, fill=(25, 34, 49), outline=(78, 99, 128), width=4)
    for index, (name, value) in enumerate(rows):
        x = 180 + (index % 3) * 1500
        y = top + 60 + (index // 3) * 135
        color = (82, 224, 143) if _is_post_step(value) else (255, 126, 107) if value == "BASELINE" else (255, 201, 92)
        draw.text((x, y), name, font=_font(31), fill=(195, 207, 225))
        draw.text((x, y + 48), value, font=_font(35), fill=color)
    verdict = report["localization_verdict"]
    explanation = {
        "D366_POST_STEP_PHYSX_TO_FABRIC_NOT_PROPAGATED": "step 뒤 PhysX 자세가 Fabric root current로 전달되지 않았습니다.",
        "D366_FABRIC_ROOT_TO_RENDER_MESH_NOT_PROPAGATED": "Fabric root는 맞지만 실제 render mesh current가 뒤따르지 않았습니다.",
        "D366_RENDER_MESH_TO_CACHE_NOT_PROPAGATED": "mesh current는 맞지만 Boundable render cache가 뒤따르지 않았습니다.",
        "D366_FABRIC_TO_HYDRA_NOT_PROPAGATED": "수치 계층은 맞지만 두 실제 RTX 화면은 baseline에 남았습니다.",
        "D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE": "한 step 뒤의 자세가 수치 계층과 두 실제 RTX 화면에 모두 보였습니다.",
    }.get(verdict, "필수 계수·동시점·두 화면 중 하나가 일치하지 않아 전달 판정을 중지했습니다.")
    draw.text((180, top + 360), verdict, font=_font(43), fill=(247, 249, 252))
    draw.text((180, top + 425), explanation, font=_font(35), fill=(214, 223, 237))
    draw.text(
        (180, top + 500),
        "통제 계수: cylinder write=1 · physics step=1 · public forward=1 · q5 sample/target/contact query=0/0/0",
        font=_font(31), fill=(157, 176, 203),
    )
    draw.text(
        (180, top + 560),
        "AssetData cache는 진단값이며, contact 발생·cap/rim·grasp·G0a는 이번 case에서 판정하지 않습니다.",
        font=_font(31), fill=(157, 176, 203),
    )
    canvas.save(SHEET_PATH)


def _quat_rotation_wxyz(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1.0e-15:
        raise RuntimeError("D366 zero-norm quaternion")
    w, x, y, z = (np.asarray([w, x, y, z], dtype=np.float64) / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _write_rerun_legacy_unused(
    report: dict[str, Any],
    private_png: dict[str, dict[str, dict[str, Any]]],
    worker: dict[str, Any],
) -> dict[str, Any]:
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"D366 rerun version drift: {rr.__version__}")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/summary/localization_sheet", name="D366 상태 전달 요약"),
                rrb.Spatial2DView(origin="/captures/primary", name="실제 Isaac primary"),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/captures/opposite", name="실제 Isaac opposite"),
                rrb.Spatial3DView(origin="/geometry", name="명령된 원통 자세"),
                rrb.TimeSeriesView(origin="/metrics/layers", name="계층별 TARGET 상태"),
                rrb.TimeSeriesView(origin="/metrics/control", name="zero-step 통제 카운터"),
                rrb.TextLogView(origin="/events", name="측정 단계"),
                row_shares=[2, 2, 2, 1, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    layer_fields = {
        "cache": "cache",
        "physx": "physx_tensor_view",
        "diagnostic_root_compatibility": "fabric_root_compatibility",
        "root_current": "fabric_root_current",
        "diagnostic_root_cached": "fabric_root_cached_render",
        "mesh_current": "fabric_mesh_current",
        "mesh_cached": "fabric_mesh_cached_render",
    }
    expected_entities = {
        "/metadata/run",
        "/summary/localization_sheet",
        "/captures/primary",
        "/captures/opposite",
        "/geometry/cylinder_commanded",
        *{f"/metrics/layers/{name}_is_target" for name in layer_fields},
        "/metrics/layers/hydra_is_target",
        "/metrics/control/display_state_writes",
        "/metrics/control/explicit_forward_calls",
        "/metrics/control/controlled_physics_steps",
        "/metrics/control/q5_science_samples",
        "/metrics/control/q5_target_updates",
        "/metrics/control/contact_queries",
        "/events/localization",
    }
    cylinder_vertices, cylinder_triangles = d362.d351._cylinder_mesh()
    phases = [
        ("baseline", "baseline_post_capture", report["baseline_pose_wxyz"]),
        ("post_write_no_forward", "post_write_after_app_update", report["target_pose_wxyz"]),
        ("post_forward", "post_forward_after_app_update", report["target_pose_wxyz"]),
    ]
    app_id = "roarm_g0a_d366_state_layer_localization"
    with rr.RecordingStream(
        app_id,
        recording_id="g0a_d366_state_layer_localization",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            "metadata/run",
            rr.TextDocument(
                json.dumps(
                    {
                        "case": CASE,
                        "purpose": "paused zero-step AssetData→PhysX→hierarchy current→mesh cache→Hydra localization",
                        "canonical_authority": _rel(REPORT_PATH),
                        "physics_recomputed": False,
                        "controlled_physics_steps": 0,
                        "q5_science_samples": 0,
                        "q5_target_updates": 0,
                        "contact_queries": 0,
                        "localization_verdict": report["localization_verdict"],
                        "g0a_pass": False,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            ),
            static=True,
        )
        sheet_bgr = cv2.imread(str(SHEET_PATH), cv2.IMREAD_COLOR)
        if sheet_bgr is None:
            raise RuntimeError("D366 localization sheet decode failed")
        recording.log(
            "summary/localization_sheet",
            rr.Image(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2RGB)),
            static=True,
        )
        for phase_index, (capture_phase, checkpoint_label, commanded_pose) in enumerate(phases):
            recording.reset_time()
            recording.set_time("localization_phase", sequence=phase_index)
            primary = cv2.imread(str(CAPTURE_PATHS[capture_phase]["primary"]), cv2.IMREAD_COLOR)
            opposite = cv2.imread(str(CAPTURE_PATHS[capture_phase]["opposite"]), cv2.IMREAD_COLOR)
            if primary is None or opposite is None:
                raise RuntimeError(f"D366 Rerun capture decode failed: {capture_phase}")
            recording.log("captures/primary", rr.Image(cv2.cvtColor(primary, cv2.COLOR_BGR2RGB)))
            recording.log("captures/opposite", rr.Image(cv2.cvtColor(opposite, cv2.COLOR_BGR2RGB)))
            pose = np.asarray(commanded_pose, dtype=np.float64)
            rotation = _quat_rotation_wxyz(pose[3:7].tolist())
            world_vertices = cylinder_vertices @ rotation.T + pose[:3]
            recording.log(
                "geometry/cylinder_commanded",
                rr.Mesh3D(
                    vertex_positions=world_vertices.astype(np.float32),
                    triangle_indices=np.asarray(cylinder_triangles, dtype=np.uint32),
                    albedo_factor=[245, 172, 52, 190],
                ),
            )
            layer = report["phase_layers"][checkpoint_label]
            for public_name, report_name in layer_fields.items():
                value = 1.0 if layer[report_name]["class"] == "TARGET" else 0.0
                recording.log(f"metrics/layers/{public_name}_is_target", rr.Scalars([value]))
            hydra_value = 1.0 if report["hydra_phase_classes"][capture_phase] == "TARGET" else 0.0
            recording.log("metrics/layers/hydra_is_target", rr.Scalars([hydra_value]))
            scalar_values = {
                "display_state_writes": 0 if phase_index == 0 else 1,
                "explicit_forward_calls": 1 if phase_index == 2 else 0,
                "controlled_physics_steps": 0,
                "q5_science_samples": 0,
                "q5_target_updates": 0,
                "contact_queries": 0,
            }
            for name, value in scalar_values.items():
                recording.log(f"metrics/control/{name}", rr.Scalars([float(value)]))
            recording.log(
                "events/localization",
                rr.TextLog(
                    f"{capture_phase}: checkpoint={checkpoint_label}; zero controlled physics; "
                    f"terminal verdict={report['localization_verdict']}",
                    level="INFO",
                ),
            )
        recording.flush(timeout_sec=30.0)
    blueprint.save(app_id, RBL_PATH)
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        expected_timeline_names=["localization_phase"],
        exact_entity_paths=sorted(expected_entities),
        exact_timeline_names=["blueprint", "localization_phase", "log_time"],
        expected_entity_components={
            "metadata/run": ["TextDocument:text"],
            "summary/localization_sheet": ["Image:buffer", "Image:format"],
            "captures/primary": ["Image:buffer", "Image:format"],
            "captures/opposite": ["Image:buffer", "Image:format"],
            "geometry/cylinder_commanded": [
                "Mesh3D:albedo_factor",
                "Mesh3D:triangle_indices",
                "Mesh3D:vertex_positions",
            ],
            "metrics/layers/cache_is_target": ["Scalars:scalars"],
            "metrics/layers/hydra_is_target": ["Scalars:scalars"],
            "metrics/control/controlled_physics_steps": ["Scalars:scalars"],
            "events/localization": ["TextLog:level", "TextLog:text"],
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


def _write_rerun(
    report: dict[str, Any],
    private_png: dict[str, dict[str, dict[str, Any]]],
    worker: dict[str, Any],
) -> dict[str, Any]:
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"D366 rerun version drift: {rr.__version__}")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/summary/visibility_sheet", name="D366 한눈 요약"),
                rrb.Spatial2DView(origin="/captures/primary", name="실제 Isaac primary"),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="/captures/opposite", name="실제 Isaac opposite"),
                rrb.Spatial3DView(origin="/geometry", name="동시점 원통 자세"),
                rrb.TimeSeriesView(origin="/metrics/layers", name="post-step 계층 일치"),
                rrb.TimeSeriesView(origin="/metrics/control", name="통제 호출 계수"),
                rrb.TextLogView(origin="/events", name="실행 단계"),
                row_shares=[2, 2, 2, 1, 1],
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    layer_fields = {
        "cache_diagnostic": "assetdata_cache_diagnostic",
        "physx": "physx_tensor_view",
        "root_current": "fabric_root_current",
        "mesh_current": "fabric_mesh_current",
        "mesh_cached": "fabric_mesh_cached_render",
    }
    expected_entities = {
        "/metadata/run",
        "/summary/visibility_sheet",
        "/captures/primary",
        "/captures/opposite",
        "/geometry/cylinder_state",
        *{f"/metrics/layers/{name}_is_poststep" for name in layer_fields},
        "/metrics/layers/hydra_is_poststep",
        "/metrics/control/cylinder_pose_writes",
        "/metrics/control/physics_steps",
        "/metrics/control/public_forward_calls",
        "/metrics/control/q5_science_samples",
        "/metrics/control/q5_target_updates",
        "/metrics/control/contact_queries",
        "/events/visibility",
    }
    cylinder_vertices, cylinder_triangles = d362.d351._cylinder_mesh()
    phases = [
        (
            checkpoint_label,
            worker["checkpoints"][checkpoint_label]["independent_physx_tensor_view"]["pose_wxyz"],
        )
        for checkpoint_label in CHECKPOINT_LABELS
    ]
    capture_for_checkpoint = {
        "baseline_post_capture": "baseline",
        "post_step_forward_terminal": "post_step_forward",
    }
    app_id = "roarm_g0a_d366_tensor_step_visibility"
    with rr.RecordingStream(
        app_id,
        recording_id="g0a_d366_tensor_step_visibility",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            "metadata/run",
            rr.TextDocument(
                json.dumps(
                    {
                        "case": CASE,
                        "purpose": "one PhysX step then one public Fabric forward visibility boundary",
                        "canonical_authority": _rel(REPORT_PATH),
                        "controlled_physics_steps": 1,
                        "cylinder_pose_writes": 1,
                        "public_forward_calls": 1,
                        "q5_science_samples": 0,
                        "q5_target_updates": 0,
                        "contact_queries": 0,
                        "localization_verdict": report["localization_verdict"],
                        "g0a_pass": False,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            ),
            static=True,
        )
        sheet_bgr = cv2.imread(str(SHEET_PATH), cv2.IMREAD_COLOR)
        if sheet_bgr is None:
            raise RuntimeError("D366 visibility sheet decode failed")
        recording.log(
            "summary/visibility_sheet",
            rr.Image(cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2RGB)),
            static=True,
        )
        for phase_index, (checkpoint_label, pose_value) in enumerate(phases):
            recording.reset_time()
            recording.set_time("visibility_phase", sequence=phase_index)
            capture_phase = capture_for_checkpoint.get(checkpoint_label)
            if capture_phase is not None:
                primary = cv2.imread(str(CAPTURE_PATHS[capture_phase]["primary"]), cv2.IMREAD_COLOR)
                opposite = cv2.imread(str(CAPTURE_PATHS[capture_phase]["opposite"]), cv2.IMREAD_COLOR)
                if primary is None or opposite is None:
                    raise RuntimeError(f"D366 Rerun capture decode failed: {capture_phase}")
                recording.log("captures/primary", rr.Image(cv2.cvtColor(primary, cv2.COLOR_BGR2RGB)))
                recording.log("captures/opposite", rr.Image(cv2.cvtColor(opposite, cv2.COLOR_BGR2RGB)))
            pose = np.asarray(pose_value, dtype=np.float64)
            rotation = _quat_rotation_wxyz(pose[3:7].tolist())
            world_vertices = cylinder_vertices @ rotation.T + pose[:3]
            recording.log(
                "geometry/cylinder_state",
                rr.Mesh3D(
                    vertex_positions=world_vertices.astype(np.float32),
                    triangle_indices=np.asarray(cylinder_triangles, dtype=np.uint32),
                    albedo_factor=[245, 172, 52, 190],
                ),
            )
            layer = report["phase_layers"][checkpoint_label]
            for public_name, report_name in layer_fields.items():
                value = 1.0 if _is_post_step(layer[report_name]["class"]) else 0.0
                recording.log(f"metrics/layers/{public_name}_is_poststep", rr.Scalars([value]))
            if checkpoint_label == "baseline_post_capture":
                recording.log("metrics/layers/hydra_is_poststep", rr.Scalars([0.0]))
            elif checkpoint_label == "post_step_forward_terminal":
                hydra_value = 1.0 if _is_post_step(report["terminal_classes"]["hydra"]) else 0.0
                recording.log("metrics/layers/hydra_is_poststep", rr.Scalars([hydra_value]))
            write_done = phase_index >= CHECKPOINT_LABELS.index("post_write_pre_step")
            step_done = phase_index >= CHECKPOINT_LABELS.index("post_step_pre_forward")
            forward_done = phase_index >= CHECKPOINT_LABELS.index("post_forward_pre_pause")
            scalar_values = {
                "cylinder_pose_writes": int(write_done),
                "physics_steps": int(step_done),
                "public_forward_calls": int(forward_done),
                "q5_science_samples": 0,
                "q5_target_updates": 0,
                "contact_queries": 0,
            }
            for name, value in scalar_values.items():
                recording.log(f"metrics/control/{name}", rr.Scalars([float(value)]))
            recording.log(
                "events/visibility",
                rr.TextLog(
                    f"checkpoint={checkpoint_label}; capture={capture_phase}; "
                    f"verdict={report['localization_verdict']}",
                    level="INFO",
                ),
            )
        recording.flush(timeout_sec=30.0)
    blueprint.save(app_id, RBL_PATH)
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        expected_timeline_names=["visibility_phase"],
        exact_entity_paths=sorted(expected_entities),
        exact_timeline_names=["blueprint", "visibility_phase", "log_time"],
        expected_entity_components={
            "metadata/run": ["TextDocument:text"],
            "summary/visibility_sheet": ["Image:buffer", "Image:format"],
            "captures/primary": ["Image:buffer", "Image:format"],
            "captures/opposite": ["Image:buffer", "Image:format"],
            "geometry/cylinder_state": [
                "Mesh3D:albedo_factor",
                "Mesh3D:triangle_indices",
                "Mesh3D:vertex_positions",
            ],
            "metrics/layers/physx_is_poststep": ["Scalars:scalars"],
            "metrics/layers/hydra_is_poststep": ["Scalars:scalars"],
            "metrics/control/physics_steps": ["Scalars:scalars"],
            "events/visibility": ["TextLog:level", "TextLog:text"],
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
    stable = (
        len({json.dumps(row, sort_keys=True) for row in observations}) == 1
        and observations[0]["bytes"] not in (None, 0)
    )
    return {"path": _rel(path), "observations": observations, "stable": stable}


def _phase_contract(path: Path, owner: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = (
        [
            ("worker_preflight", "complete"),
            ("AppLauncher", "start"),
            ("AppLauncher", "complete"),
            ("make_runtime_env", "start"),
            ("make_runtime_env", "complete"),
            ("reset", "start"),
            ("reset", "complete"),
            ("initial_timeline_pause", "complete"),
            ("baseline_layers", "start"),
            ("baseline_layers", "complete"),
            ("timeline_play", "start"),
            ("timeline_play", "complete"),
            ("direct_root_pose_write", "start"),
            ("direct_root_pose_write", "complete"),
            ("controlled_sim_step_render_false", "start"),
            ("controlled_sim_step_render_false", "complete"),
            ("explicit_simulation_context_forward", "start"),
            ("explicit_simulation_context_forward", "complete"),
            ("terminal_timeline_pause", "start"),
            ("terminal_timeline_pause", "complete"),
            ("worker_summary", "complete"),
        ]
        if owner == "worker"
        else [
            ("supervisor", "start"),
            ("worker_process", "start"),
            ("worker_process", "exit"),
            ("supervisor_summary", "complete"),
        ]
    )
    counts = {
        f"{phase}:{event}": sum(
            row.get("phase") == phase and row.get("event") == event for row in rows
        )
        for phase, event in required
    }
    required_positions = [
        next(
            (
                index
                for index, row in enumerate(rows)
                if row.get("phase") == phase and row.get("event") == event
            ),
            None,
        )
        for phase, event in required
    ]
    checks = {
        "nonempty": bool(rows),
        "sequence_exact": [row.get("sequence") for row in rows] == list(range(1, len(rows) + 1)),
        "owner_exact": all(row.get("owner") == owner for row in rows),
        "required_exact_once": all(value == 1 for value in counts.values()),
        "required_order_exact": all(position is not None for position in required_positions)
        and required_positions == sorted(required_positions),
    }
    details: dict[str, Any] = {}
    if owner == "worker":
        capture_rows = [
            row
            for row in rows
            if row.get("phase") == "viewport_capture" and row.get("event") == "complete"
        ]
        expected_roles = {
            f"{phase}_{view}"
            for phase in CAPTURE_PATHS
            for view in CAMERA_EYES
        }
        expected_role_order = [
            f"{phase}_{view}"
            for phase in CAPTURE_PATHS
            for view in CAMERA_EYES
        ]
        observed_roles = [row.get("details", {}).get("role") for row in capture_rows]
        forward_rows = [
            row
            for row in rows
            if row.get("phase") == "explicit_simulation_context_forward"
            and row.get("event") == "complete"
        ]
        checks.update(
            {
                "four_capture_markers_unique": len(capture_rows) == 4
                and set(observed_roles) == expected_roles
                and len(set(observed_roles)) == 4,
                "four_capture_marker_order_exact": observed_roles == expected_role_order,
                "one_forward_marker_count_one": len(forward_rows) == 1
                and forward_rows[0].get("details", {}).get("pass") is True,
                "one_step_complete_marker": sum(
                    row.get("phase") == "controlled_sim_step_render_false"
                    and row.get("event") == "complete"
                    and row.get("details", {}).get("pass") is True
                    for row in rows
                ) == 1,
                "terminal_worker_summary_exact": rows[-1].get("phase") == "worker_summary"
                and rows[-1].get("event") == "complete"
                and rows[-1].get("details", {}).get("pass") is True,
                "worker_exception_marker_absent": not any(
                    row.get("phase") == "worker_exception" for row in rows
                ),
            }
        )
        details = {
            "required_positions": required_positions,
            "capture_roles": observed_roles,
            "expected_capture_role_order": expected_role_order,
            "forward_rows": forward_rows,
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
        raise RuntimeError("D366 prepare/preregistration did not pass")
    existing_inventory = sorted(path.resolve() for path in OUT_DIR.rglob("*") if path.is_file())
    expected_prepare_inventory = sorted([PREREG_PATH.resolve(), PREPARE_PATH.resolve()])
    if existing_inventory != expected_prepare_inventory:
        raise RuntimeError(
            "D366 run/resume/overwrite forbidden; output inventory is not prepare-only: "
            f"{[_rel(path) for path in existing_inventory]}"
        )
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_exact": _status_scope_ok(_git_status()),
        "staged_continuity_exact": _git_cached_patch() == prereg["git_cached_patch"],
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "session_preregistration_hash_exact": _sha(SESSION_DOC)
        == prereg["session_preregistration_sha256"]
        and SESSION_DOC.stat().st_size == prereg["session_preregistration_bytes"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "installed_source_contract_exact": _installed_source_contract_audit()
        == prereg["installed_source_contract_audit"],
        "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
        "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
        "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
        "d365_manifest_exact": _tree_manifest(D365_DIR) == prereg["d365_manifest_before"],
        "sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "prereg_artifact_case_exact": prereg.get("artifact") == "D366_PREREGISTRATION_V1"
        and prereg.get("case") == CASE,
        "prepare_artifact_case_exact": prepare.get("artifact") == "D366_PREPARE_PREFLIGHT_V1"
        and prepare.get("case") == CASE,
        "prepare_binds_prereg_hash": prepare.get("preregistration_sha256") == _sha(PREREG_PATH),
        "prepare_all_checks_pass": prepare.get("pass") is True
        and bool(prepare.get("checks"))
        and all(prepare["checks"].values()),
        "prepare_only_inventory_exact": existing_inventory == expected_prepare_inventory,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D366 pre-invocation STOP: {checks}")
    _marker("supervisor", "supervisor", "start", {"checks": checks})
    token = secrets.token_hex(32)
    invocation = {
        "artifact": "D366_ISAAC_INVOCATION_MARKER_V1",
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
    env["DISPLAY"] = DISPLAY
    env[WORKER_TOKEN_ENV] = token
    env[SUPERVISOR_PID_ENV] = str(os.getpid())
    start = time.monotonic()
    last_progress = start
    last_phase_size = 0
    telemetry: list[dict[str, Any]] = []
    watchdog_reason: str | None = None
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
        while process.poll() is None:
            elapsed = time.monotonic() - start
            try:
                phase_size = WORKER_PHASE_PATH.stat().st_size if WORKER_PHASE_PATH.exists() else 0
                if phase_size != last_phase_size:
                    last_phase_size = phase_size
                    last_progress = time.monotonic()
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
            elif time.monotonic() - last_progress > INACTIVITY_WATCHDOG_S:
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
    _marker(
        "supervisor",
        "worker_process",
        "exit",
        {"exit_code": worker_exit, "elapsed_s": elapsed, "watchdog": watchdog_reason},
    )

    postprocess_errors: list[dict[str, Any]] = []

    def record_postprocess_error(stage: str, error: BaseException) -> None:
        postprocess_errors.append(
            {
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )

    stable_paths = [path for phase in CAPTURE_PATHS.values() for path in phase.values()]
    stability: list[dict[str, Any]] = []
    if worker_exit == 0 and not WORKER_EXCEPTION_PATH.exists():
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
    layer_audit: dict[str, Any] | None = None
    if LAYER_JOURNAL_PATH.is_file():
        try:
            layer_audit = _audit_layer_journal()
        except Exception as error:
            record_postprocess_error("layer_journal_audit", error)
    journal_worker_match: dict[str, Any] | None = None
    if worker is not None and LAYER_JOURNAL_PATH.is_file():
        try:
            journal_worker_match = _journal_worker_checkpoint_match(worker)
        except Exception as error:
            record_postprocess_error("journal_worker_checkpoint_match", error)
    report: dict[str, Any] | None = None
    private_png: dict[str, dict[str, dict[str, Any]]] | None = None
    rerun: dict[str, Any] | None = None
    if worker is not None and not WORKER_EXCEPTION_PATH.exists():
        try:
            report, private_png = _build_localization_report(worker)
        except Exception as error:
            record_postprocess_error("localization_report", error)
        if report is not None:
            try:
                _build_sheet(report)
            except Exception as error:
                record_postprocess_error("localization_sheet", error)
        if report is not None and private_png is not None and SHEET_PATH.is_file():
            try:
                rerun = _write_rerun(report, private_png, worker)
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
                "artifact": "D366_SUPERVISOR_POSTPROCESS_EXCEPTION_V1",
                "case": CASE,
                "utc": _utc_now(),
                "automatic_retry": False,
                "errors": postprocess_errors,
                "pass": False,
            },
        )
    resource_summary = {
        "samples": len(telemetry),
        "gpu_used_mib_max": max(
            (row["gpu"].get("memory_used_mib") or 0 for row in telemetry), default=None
        ),
        "gpu_free_mib_min": min(
            (row["gpu"].get("memory_free_mib") or 10**9 for row in telemetry), default=None
        ),
        "gpu_utilization_percent_max": max(
            (row["gpu"].get("utilization_gpu_percent") or 0 for row in telemetry), default=None
        ),
        "worker_rss_bytes_max": max(
            (row.get("worker_rss_bytes", 0) for row in telemetry), default=None
        ),
    }
    post_checks = {
        "head_origin_unchanged": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_unchanged": _status_scope_ok(_git_status()),
        "worker_exit_zero": worker_exit == 0,
        "worker_exception_absent": not WORKER_EXCEPTION_PATH.exists(),
        "watchdog_not_triggered": watchdog_reason is None,
        "worker_summary_pass": worker is not None and worker.get("pass") is True,
        "all_four_png_stable_after_close": len(stability) == 4
        and all(row["stable"] for row in stability),
        "layer_journal_audit_pass": layer_audit is not None and layer_audit.get("pass") is True,
        "journal_worker_checkpoint_exact": journal_worker_match is not None
        and journal_worker_match.get("pass") is True,
        "localization_report_concrete_pass": report is not None and report.get("pass") is True,
        "rerun_pass": rerun is not None and rerun.get("pass") is True,
        "worker_phase_pass": worker_phase.get("pass") is True,
        "postprocess_exception_absent": not postprocess_errors,
        "d362_manifest_unchanged": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
        "d363_manifest_unchanged": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
        "d364_manifest_unchanged": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
        "d365_manifest_unchanged": _tree_manifest(D365_DIR) == prereg["d365_manifest_before"],
        "staged_continuity_unchanged": _git_cached_patch() == prereg["git_cached_patch"],
        "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "input_hashes_unchanged": _input_hashes() == prereg["input_hashes"],
        "installed_source_contract_unchanged": _installed_source_contract_audit()
        == prereg["installed_source_contract_audit"],
    }
    supervisor = {
        "artifact": "D366_SUPERVISOR_SUMMARY_V1",
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
        "layer_journal_audit": layer_audit,
        "journal_worker_checkpoint_match": journal_worker_match,
        "localization_report": report,
        "rerun": rerun,
        "post_checks": post_checks,
        "display_state_write_count": worker.get("display_state_write_count") if worker else None,
        "display_state_write_returned": worker.get("display_state_write_returned") if worker else None,
        "controlled_step_call_count": worker.get("controlled_step_call_count") if worker else None,
        "controlled_step_returned": worker.get("controlled_step_returned") if worker else None,
        "physics_callback_count": worker.get("physics_callback_count") if worker else None,
        "physics_callback_dts": worker.get("physics_callback_dts") if worker else None,
        "explicit_forward_count": worker.get("explicit_forward_count") if worker else None,
        "explicit_forward_returned": worker.get("explicit_forward_returned") if worker else None,
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
    bound_artifact_hashes = {
        _rel(path): _sha(path)
        for path in sorted(OUT_DIR.rglob("*"))
        if path.is_file() and path not in {AUTOMATED_PATH, MANUAL_PATH, COMPLETION_PATH}
    }
    automated_checks = {
        "supervisor_pass": supervisor["pass"],
        "supervisor_phase_pass": supervisor_phase.get("pass") is True,
        "worker_report_and_rerun_hash_bound": all(
            path.is_file()
            for path in (
                WORKER_SUMMARY_PATH,
                LAYER_AUDIT_PATH,
                REPORT_PATH,
                RERUN_VALIDATION_PATH,
            )
        ),
        "write_step_forward_exact_one": supervisor["display_state_write_count"] == 1
        and supervisor["display_state_write_returned"] is True
        and supervisor["controlled_step_call_count"] == 1
        and supervisor["controlled_step_returned"] is True
        and supervisor["physics_callback_count"] == 1
        and len(supervisor["physics_callback_dts"] or []) == 1
        and abs((supervisor["physics_callback_dts"] or [math.inf])[0] - PHYSICS_DT_S) <= TIME_TOL_S
        and supervisor["explicit_forward_count"] == 1
        and supervisor["explicit_forward_returned"] is True,
        "controlled_physics_exactly_one": supervisor["controlled_physics_steps"] == 1,
        "q5_science_and_target_zero": supervisor["q5_science_sample_count"] == 0
        and supervisor["q5_target_update_count"] == 0,
        "contact_query_zero": supervisor["contact_query_count"] == 0,
        "d362_d363_d364_d365_immutable": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"]
        and _tree_manifest(D363_DIR) == prereg["d363_manifest_before"]
        and _tree_manifest(D364_DIR) == prereg["d364_manifest_before"]
        and _tree_manifest(D365_DIR) == prereg["d365_manifest_before"],
        "staged_continuity_unchanged": _git_cached_patch() == prereg["git_cached_patch"],
        "bound_artifacts_nonempty": bool(bound_artifact_hashes)
        and all(bool(digest) for digest in bound_artifact_hashes.values()),
    }
    automated = {
        "artifact": "D366_AUTOMATED_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "supervisor_path": _rel(SUPERVISOR_PATH),
        "supervisor_sha256": _sha(SUPERVISOR_PATH),
        "worker_summary_sha256": _sha(WORKER_SUMMARY_PATH) if WORKER_SUMMARY_PATH.is_file() else None,
        "layer_audit_sha256": _sha(LAYER_AUDIT_PATH) if LAYER_AUDIT_PATH.is_file() else None,
        "localization_report_sha256": _sha(REPORT_PATH) if REPORT_PATH.is_file() else None,
        "rerun_validation_sha256": _sha(RERUN_VALIDATION_PATH) if RERUN_VALIDATION_PATH.is_file() else None,
        "bound_artifact_hashes": bound_artifact_hashes,
        "bound_artifact_inventory": sorted(bound_artifact_hashes),
        "supervisor_phase_contract": supervisor_phase,
        "checks": automated_checks,
        "manual_visual_inspection_pending": True,
        "completion_pending": True,
        "localization_verdict": report.get("localization_verdict") if report else None,
        "physical_science_recomputed": False,
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "pass_before_manual": all(automated_checks.values()),
    }
    _write_json_x(AUTOMATED_PATH, automated)
    print(
        json.dumps(
            {
                "stage": "run",
                "pass_before_manual": automated["pass_before_manual"],
                "worker_exit": worker_exit,
                "localization_verdict": automated["localization_verdict"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if automated["pass_before_manual"] else 2


def _all_required_visual_paths() -> list[str]:
    paths = [
        *[path for phase in CAPTURE_PATHS.values() for path in phase.values()],
        SHEET_PATH,
        RERUN_PNG_PATH,
    ]
    return sorted(_rel(path) for path in paths)


def _finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists():
        raise RuntimeError("D366 completion overwrite forbidden")
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    worker = _json(WORKER_SUMMARY_PATH)
    report = _json(REPORT_PATH)
    expected_paths = _all_required_visual_paths()
    expected_manual_visual_class = {
        "POST_STEP": "post_step_toppled",
        "POST_STEP_EQ_COMMANDED": "post_step_toppled",
        "BASELINE": "baseline_upright",
        "OTHER": "visually_ambiguous",
        "UNAVAILABLE": "visually_ambiguous",
    }.get(report.get("terminal_classes", {}).get("hydra"), "visually_ambiguous")
    manual_checks = {
        "artifact_exact": manual.get("artifact") == "D366_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "automated_sha_exact": manual.get("automated_summary_sha256") == _sha(AUTOMATED_PATH),
        "all_required_paths_exact": sorted(manual.get("inspected_paths", [])) == expected_paths,
        "all_path_hashes_exact": all(
            (REPO / path).is_file()
            and manual.get("inspected_sha256", {}).get(path) == _sha(REPO / path)
            for path in expected_paths
        ),
        "original_resolution_inspected": manual.get("original_resolution_inspected") is True,
        "baseline_upright_seen": manual.get("baseline_upright_seen") is True,
        "post_step_forward_visual_class_recorded": manual.get("post_step_forward_visual_class")
        in {"baseline_upright", "post_step_toppled", "visually_ambiguous"},
        "manual_visual_class_matches_automated_hydra": manual.get(
            "post_step_forward_visual_class"
        )
        == expected_manual_visual_class,
        "two_views_and_sheet_inspected": manual.get("two_views_and_sheet_inspected") is True,
        "rerun_actual_images_visible": manual.get("rerun_actual_images_visible") is True,
        "observation_recorded_ko": len(str(manual.get("observation_ko", "")).strip()) >= 20,
        "scope_limit_recorded": manual.get("no_new_q5_contact_cap_rim_grasp_claim") is True,
        "manual_pass": manual.get("pass") is True,
    }
    bound_hashes = automated.get("bound_artifact_hashes", {})
    current_precompletion_inventory = sorted(
        _rel(path) for path in OUT_DIR.rglob("*") if path.is_file() and path != COMPLETION_PATH
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
        _canonical_bytes(precompletion_artifact_hashes)
    ).hexdigest()
    integrity_checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_exact": _status_scope_ok(_git_status()),
        "automated_pass_before_manual": automated.get("pass_before_manual") is True,
        "manual_checks": all(manual_checks.values()),
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "session_preregistration_hash_exact": _sha(SESSION_DOC)
        == prereg["session_preregistration_sha256"]
        and SESSION_DOC.stat().st_size == prereg["session_preregistration_bytes"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "installed_source_contract_exact": _installed_source_contract_audit()
        == prereg["installed_source_contract_audit"],
        "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
        "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
        "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
        "d365_manifest_exact": _tree_manifest(D365_DIR) == prereg["d365_manifest_before"],
        "staged_continuity_exact": _git_cached_patch() == prereg["git_cached_patch"],
        "d334_sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "automated_supervisor_hash_exact": automated.get("supervisor_sha256") == _sha(SUPERVISOR_PATH),
        "automated_worker_hash_exact": automated.get("worker_summary_sha256") == _sha(WORKER_SUMMARY_PATH),
        "automated_layer_audit_hash_exact": automated.get("layer_audit_sha256") == _sha(LAYER_AUDIT_PATH),
        "automated_report_hash_exact": automated.get("localization_report_sha256") == _sha(REPORT_PATH),
        "automated_rerun_validation_hash_exact": automated.get("rerun_validation_sha256")
        == _sha(RERUN_VALIDATION_PATH),
        "all_bound_artifact_hashes_exact": bool(bound_hashes)
        and all((REPO / path).is_file() and _sha(REPO / path) == digest for path, digest in bound_hashes.items()),
        "precompletion_inventory_exact": current_precompletion_inventory == expected_precompletion_inventory,
        "worker_registered_counters_exact": worker.get("display_state_write_count") == 1
        and worker.get("display_state_write_returned") is True
        and worker.get("controlled_step_call_count") == 1
        and worker.get("controlled_step_returned") is True
        and worker.get("physics_callback_count") == 1
        and len(worker.get("physics_callback_dts", [])) == 1
        and abs(worker.get("physics_callback_dts", [math.inf])[0] - PHYSICS_DT_S) <= TIME_TOL_S
        and worker.get("explicit_forward_count") == 1
        and worker.get("explicit_forward_returned") is True
        and worker.get("controlled_physics_steps") == 1
        and worker.get("q5_science_sample_count") == 0
        and worker.get("q5_target_update_count") == 0
        and worker.get("contact_query_count") == 0,
        "localization_report_concrete": report.get("pass") is True
        and report.get("localization_verdict") != "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP",
        "rerun_validation_pass": _json(RERUN_VALIDATION_PATH).get("pass") is True,
    }
    completion = {
        "artifact": "D366_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "final_verdict": report.get("localization_verdict")
        if all(integrity_checks.values())
        else "D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP",
        "manual_checks": manual_checks,
        "integrity_checks": integrity_checks,
        "terminal_classes": report.get("terminal_classes"),
        "display_state_write_count": worker.get("display_state_write_count"),
        "display_state_write_returned": worker.get("display_state_write_returned"),
        "controlled_step_call_count": worker.get("controlled_step_call_count"),
        "controlled_step_returned": worker.get("controlled_step_returned"),
        "physics_callback_count": worker.get("physics_callback_count"),
        "physics_callback_dts": worker.get("physics_callback_dts"),
        "explicit_forward_count": worker.get("explicit_forward_count"),
        "explicit_forward_returned": worker.get("explicit_forward_returned"),
        "controlled_physics_steps": worker.get("controlled_physics_steps"),
        "q5_science_sample_count": worker.get("q5_science_sample_count"),
        "q5_target_update_count": worker.get("q5_target_update_count"),
        "contact_query_count": worker.get("contact_query_count"),
        "counter_semantics": "registered D366 calls only; reset-internal transition excluded",
        "physical_science_recomputed": False,
        "cap_rim_science": None,
        "grasp_or_g0a_science": None,
        "g0a_pass": False,
        "automated_summary_sha256": _sha(AUTOMATED_PATH),
        "manual_visual_inspection_sha256": _sha(MANUAL_PATH),
        "precompletion_artifact_hashes": precompletion_artifact_hashes,
        "precompletion_hash_manifest_sha256": precompletion_hash_manifest_sha256,
        "output_inventory": sorted([*current_precompletion_inventory, _rel(COMPLETION_PATH)]),
        "pass": all(integrity_checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(
        json.dumps(
            {
                "stage": "finalize",
                "pass": completion["pass"],
                "verdict": completion["final_verdict"],
            },
            ensure_ascii=False,
        )
    )
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
        raise RuntimeError("D366 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D366 seed drift")
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
