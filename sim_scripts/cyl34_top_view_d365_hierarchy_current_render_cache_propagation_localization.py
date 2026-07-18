#!/usr/bin/env python3
"""D365: localize the hierarchy-current to render-cache propagation break.

This forward-only observability probe reads immutable D362 row 499, writes only
the cylinder root pose once, calls SimulationContext.forward() once, advances no
controlled physics, and independently observes AssetData, the PhysX tensor view,
Fabric/USDRT root and rendered-mesh transforms, and Hydra pixels. Compatibility
``_world*`` attributes and the non-Boundable root cache are diagnostic only.
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
)


CASE = "g0a_d365"
CASE_NAME = "hierarchy_current_render_cache_propagation_localization"
NEW_VARIABLES = [
    "optional_compatibility_removed_from_linear_gate",
    "physx_hierarchy_mesh_cache_hydra_single_pose_localization",
]
BASE_GIT = "94c0644ef3d4e69278bc864f0f8c2f3a40908dc8"
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

OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260718_grasp_g0a_d365_hierarchy_current_render_cache_propagation_localization.md"
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
D365_PREPARE_ATTEMPT1_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d365"
D365_PREPARE_ATTEMPT1_PREREG = D365_PREPARE_ATTEMPT1_DIR / "d365_preregistration.json"
D365_PREPARE_ATTEMPT1_PREFLIGHT = D365_PREPARE_ATTEMPT1_DIR / "d365_prepare_preflight.json"
D365_PREPARE_ATTEMPT1_INVOCATION = D365_PREPARE_ATTEMPT1_DIR / "d365_isaac_invocation_marker.json"
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
    str(D365_PREPARE_ATTEMPT1_PREREG.relative_to(REPO)): "2b81e3609bb89d64d48061235e8baa8541f787d873a319fd0dd000695803e668",
    str(D365_PREPARE_ATTEMPT1_PREFLIGHT.relative_to(REPO)): "8e676a4f1ece7b4d6714f9230fc2bf15206d6324116654a1d0fe3548e6b06301",
}

PREREG_PATH = OUT_DIR / "d365_preregistration.json"
PREPARE_PATH = OUT_DIR / "d365_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d365_isaac_invocation_marker.json"
WORKER_PREFLIGHT_PATH = OUT_DIR / "d365_worker_preflight.json"
RUNTIME_PATH = OUT_DIR / "d365_runtime_fabric_attestation.json"
WORKER_PHASE_PATH = OUT_DIR / "d365_worker_phase_markers.jsonl"
SUPERVISOR_PHASE_PATH = OUT_DIR / "d365_supervisor_phase_markers.jsonl"
LAYER_JOURNAL_PATH = OUT_DIR / "d365_layer_readback_journal.jsonl"
LAYER_AUDIT_PATH = OUT_DIR / "d365_layer_readback_journal_audit.json"
WORKER_LOG_PATH = OUT_DIR / "d365_worker_stdout_stderr.log"
WORKER_SUMMARY_PATH = OUT_DIR / "d365_worker_summary.json"
WORKER_EXCEPTION_PATH = OUT_DIR / "d365_worker_exception.json"
REPORT_PATH = OUT_DIR / "d365_state_layer_localization_report.json"
SHEET_PATH = OUT_DIR / "d365_state_layer_localization_sheet_ko.png"
RRD_PATH = OUT_DIR / "d365_state_layer_localization.rrd"
RBL_PATH = OUT_DIR / "d365_state_layer_localization.rbl"
RERUN_PNG_PATH = OUT_DIR / "d365_state_layer_localization_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d365_rerun_validation.json"
SUPERVISOR_PATH = OUT_DIR / "d365_supervisor_summary.json"
AUTOMATED_PATH = OUT_DIR / "d365_automated_summary.json"
MANUAL_PATH = OUT_DIR / "d365_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d365_completion_summary.json"
POSTPROCESS_EXCEPTION_PATH = OUT_DIR / "d365_supervisor_postprocess_exception.json"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

WORKER_TOKEN_ENV = "D365_WORKER_LAUNCH_TOKEN"
SUPERVISOR_PID_ENV = "D365_SUPERVISOR_PID"
CAPTURE_PATHS = {
    phase: {
        view: OUT_DIR / f"d365_{phase}_{view}_actual_isaac.png"
        for view in CAMERA_EYES
    }
    for phase in ("baseline", "post_write_no_forward", "post_forward")
}
CHECKPOINT_LABELS = (
    "baseline_pre_capture",
    "baseline_post_capture",
    "post_write_immediate",
    "post_write_after_app_update",
    "post_forward_immediate",
    "post_forward_after_app_update",
)

_WORKER_SEQUENCE = 0
_SUPERVISOR_SEQUENCE = 0
_LAYER_SEQUENCE = 0
_LAYER_PREV_SHA = "0" * 64
_DISPLAY_STATE_WRITE_COUNT = 0
_EXPLICIT_FORWARD_COUNT = 0
_DISPLAY_STATE_WRITE_RETURNED = False
_EXPLICIT_FORWARD_RETURNED = False
_CONTROLLED_PHYSICS_STEPS = 0
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


def _status_scope_ok(rows: list[str]) -> bool:
    allowed_exact = {
        "START_HERE.md",
        "claudedocs/DECISIONS.md",
        "claudedocs/EXPERIMENT_LEDGER.md",
        _rel(HARNESS),
        _rel(SESSION_DOC),
        _rel(D364_HARNESS),
        _rel(D364_SESSION),
    }
    allowed_prefixes = (
        f"claudedocs/runtime_logs/grasp_track/{CASE}/",
        "claudedocs/runtime_logs/grasp_track/g0a_d364/",
    )
    for row in rows:
        if " -> " in row:
            return False
        path = row[3:].strip()
        if path not in allowed_exact and not path.startswith(allowed_prefixes):
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
        D365_PREPARE_ATTEMPT1_PREREG,
        D365_PREPARE_ATTEMPT1_PREFLIGHT,
    )}


def _prepare_attempt1_access_failure_audit() -> dict[str, Any]:
    prereg = _json(D365_PREPARE_ATTEMPT1_PREREG)
    prepare = _json(D365_PREPARE_ATTEMPT1_PREFLIGHT)
    failed_checks = sorted(name for name, value in prepare.get("checks", {}).items() if value is not True)
    checks = {
        "artifacts_exact": prereg.get("artifact") == "D365_PREREGISTRATION_V1"
        and prepare.get("artifact") == "D365_PREPARE_PREFLIGHT_V1",
        "prepare_failed": prereg.get("pass") is False and prepare.get("pass") is False,
        "only_host_access_checks_failed": failed_checks == [
            "display_available",
            "gpu_exact",
            "gpu_free_gate",
        ],
        "sandbox_signature_exact": prepare.get("display_returncode") == 1
        and prepare.get("gpu_and_ram", {}).get("nvidia_smi_returncode") == 9,
        "actual_invocation_marker_absent": not D365_PREPARE_ATTEMPT1_INVOCATION.exists(),
    }
    return {
        "failed_checks": failed_checks,
        "attempt1_preregistration_sha256": _sha(D365_PREPARE_ATTEMPT1_PREREG),
        "attempt1_prepare_sha256": _sha(D365_PREPARE_ATTEMPT1_PREFLIGHT),
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


def _classify_pose(observed: list[float] | None, baseline: list[float], target: list[float]) -> dict[str, Any]:
    if observed is None:
        return {"class": "UNAVAILABLE", "baseline": None, "target": None}
    baseline_error = _pose_error(observed, baseline)
    target_error = _pose_error(observed, target)
    if target_error["match"] and not baseline_error["match"]:
        label = "TARGET"
    elif baseline_error["match"] and not target_error["match"]:
        label = "BASELINE"
    elif target_error["match"] and baseline_error["match"]:
        label = "AMBIGUOUS_BOTH"
    else:
        label = "OTHER"
    return {"class": label, "baseline": baseline_error, "target": target_error}


def _classify_matrix(observed: list[list[float]] | None, baseline: list[list[float]], target: list[list[float]]) -> dict[str, Any]:
    if observed is None:
        return {"class": "UNAVAILABLE", "baseline_max_abs": None, "target_max_abs": None}
    obs = np.asarray(observed, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    base_error = float(np.max(np.abs(obs - base)))
    target_error = float(np.max(np.abs(obs - tgt)))
    base_match = base_error <= MATRIX_TOL
    target_match = target_error <= MATRIX_TOL
    if target_match and not base_match:
        label = "TARGET"
    elif base_match and not target_match:
        label = "BASELINE"
    elif target_match and base_match:
        label = "AMBIGUOUS_BOTH"
    else:
        label = "OTHER"
    return {
        "class": label,
        "baseline_max_abs": base_error,
        "target_max_abs": target_error,
        "baseline_match": base_match,
        "target_match": target_match,
    }


def _decision_fixture(
    cache: str,
    physx: str,
    root_compatibility: str,
    root_current: str,
    root_cached: str,
    mesh_current: str,
    mesh_cached: str,
    hydra: str,
    complete: bool = True,
) -> str:
    if not complete:
        return "D365_MEASUREMENT_INCOMPLETE_FAIL_STOP"
    allowed = {"BASELINE", "TARGET"}
    linear = (cache, physx, root_current, mesh_current, mesh_cached, hydra)
    if any(value not in allowed for value in linear):
        return "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    if _downstream_ahead(linear):
        return "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    if cache != "TARGET":
        return "D365_DIRECT_WRITE_OR_CACHE_FAIL"
    if physx != "TARGET":
        return "D365_CACHE_TO_PHYSX_PENDING_OR_FAILED"
    if root_current != "TARGET":
        return "D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED"
    if mesh_current != "TARGET":
        return "D365_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED"
    if mesh_cached != "TARGET":
        return "D365_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED"
    if hydra != "TARGET":
        return "D365_FABRIC_TO_HYDRA_NOT_PROPAGATED"
    return "D365_END_TO_END_ZERO_STEP_VISIBLE"


def _temporal_class_audit(values: list[str]) -> dict[str, bool]:
    seen_target = False
    regressed = False
    for value in values:
        if value == "TARGET":
            seen_target = True
        elif seen_target and value != "TARGET":
            regressed = True
    return {
        "regression": regressed,
        "other": any(value not in {"BASELINE", "TARGET"} for value in values),
    }


def _downstream_ahead(values: tuple[str, ...] | list[str]) -> bool:
    seen_baseline = False
    for value in values:
        if value == "BASELINE":
            seen_baseline = True
        elif value == "TARGET" and seen_baseline:
            return True
    return False


def _negative_controls() -> dict[str, Any]:
    cases = {
        "cache_break": (("BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", True), "D365_DIRECT_WRITE_OR_CACHE_FAIL"),
        "physx_break": (("TARGET", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", "BASELINE", True), "D365_CACHE_TO_PHYSX_PENDING_OR_FAILED"),
        "optional_compatibility_unavailable": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "optional_compatibility_baseline": (("TARGET", "TARGET", "BASELINE", "TARGET", "BASELINE", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "optional_compatibility_other": (("TARGET", "TARGET", "OTHER", "TARGET", "OTHER", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "fabric_hierarchy_break": (("TARGET", "TARGET", "UNAVAILABLE", "BASELINE", "UNAVAILABLE", "BASELINE", "BASELINE", "BASELINE", True), "D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED"),
        "root_cached_stale_is_diagnostic_only": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "BASELINE", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "root_cached_unavailable_is_diagnostic_only": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "mesh_hierarchy_break": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "BASELINE", "BASELINE", "BASELINE", True), "D365_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED"),
        "mesh_render_cache_break": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "BASELINE", "BASELINE", True), "D365_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED"),
        "hydra_break": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "TARGET", "BASELINE", True), "D365_FABRIC_TO_HYDRA_NOT_PROPAGATED"),
        "complete": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", True), "D365_END_TO_END_ZERO_STEP_VISIBLE"),
        "missing": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", False), "D365_MEASUREMENT_INCOMPLETE_FAIL_STOP"),
        "root_other": (("TARGET", "TARGET", "UNAVAILABLE", "OTHER", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", True), "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"),
        "root_downstream_ahead": (("TARGET", "TARGET", "UNAVAILABLE", "BASELINE", "UNAVAILABLE", "TARGET", "TARGET", "TARGET", True), "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"),
        "mesh_cache_downstream_ahead": (("TARGET", "TARGET", "UNAVAILABLE", "TARGET", "UNAVAILABLE", "BASELINE", "TARGET", "TARGET", True), "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"),
    }
    results = {
        name: {"observed": _decision_fixture(*inputs), "expected": expected}
        for name, (inputs, expected) in cases.items()
    }
    q = [0.9238795325, 0.0, 0.3826834324, 0.0]
    pose = [0.31, -0.02, 0.03, *q]
    q_flip = [0.31, -0.02, 0.03, *[-item for item in q]]
    q_swap = [0.31, -0.02, 0.03, q[1], q[2], q[3], q[0]]
    pose_controls = {
        "q_sign_equivalent": _pose_error(q_flip, pose)["match"] is True,
        "xyzw_wxyz_swap_rejected": _pose_error(q_swap, pose)["match"] is False,
        "translation_10mm_rejected": _pose_error([0.32, -0.02, 0.03, *q], pose)["match"] is False,
    }
    temporal_controls = {
        "target_then_baseline_regression_detected": _temporal_class_audit(
            ["BASELINE", "TARGET", "BASELINE"]
        )["regression"] is True,
        "baseline_then_target_not_regression": _temporal_class_audit(
            ["BASELINE", "BASELINE", "TARGET"]
        )["regression"] is False,
        "temporal_other_detected": _temporal_class_audit(
            ["BASELINE", "OTHER", "TARGET"]
        )["other"] is True,
        "per_phase_downstream_ahead_detected": _downstream_ahead(
            ["TARGET", "TARGET", "BASELINE", "TARGET", "TARGET", "TARGET"]
        ) is True,
    }
    checks = {
        **{name: row["observed"] == row["expected"] for name, row in results.items()},
        **pose_controls,
        **temporal_controls,
    }
    return {
        "decision_cases": results,
        "pose_controls": pose_controls,
        "temporal_controls": temporal_controls,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _static_source_audit() -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    call_names: list[str] = []

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_names.append(dotted(node.func))

    def count_suffix(suffix: str) -> int:
        return sum(name.endswith(suffix) for name in call_names)

    forbidden_suffixes = (
        ".step",
        ".simulate",
        ".fetch_results",
        ".render",
        ".scene.update",
        ".scene.write_data_to_sim",
        ".write_joint_state_to_sim",
        ".write_root_velocity_to_sim",
        ".set_joint_position_target",
        ".update_world_xforms",
        ".SetWorldXformFromUsd",
        ".SetLocalXformFromUsd",
        ".CreateWorldPositionAttr",
        ".CreateWorldOrientationAttr",
        ".CreateWorldScaleAttr",
        "._update_fabric",
    )
    forbidden_hits = sorted(name for name in call_names if any(name.endswith(suffix) for suffix in forbidden_suffixes))
    checks = {
        "one_root_pose_write_site": count_suffix(".write_root_pose_to_sim") == 1,
        "one_public_forward_site": count_suffix(".sim.forward") == 1,
        "one_independent_physx_getter_site": count_suffix(".root_physx_view.get_transforms") == 1,
        "no_forbidden_mutating_call_sites": not forbidden_hits,
        "no_timeline_play_site": count_suffix(".play") == 0,
        "no_timeline_stop_site": count_suffix(".stop") == 0,
    }
    return {"call_names": sorted(set(call_names)), "forbidden_hits": forbidden_hits, "checks": checks, "pass": all(checks.values())}


def _prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D365 output already exists; overwrite forbidden: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    input_hashes = _input_hashes()
    session_preregistration_sha256 = _sha(SESSION_DOC)
    session_preregistration_bytes = SESSION_DOC.stat().st_size
    d362_manifest = _tree_manifest(D362_DIR)
    d363_manifest = _tree_manifest(D363_DIR)
    d364_manifest = _tree_manifest(D364_DIR)
    d364_evidence_audit = _d364_completion_evidence_audit()
    prepare_attempt1_audit = _prepare_attempt1_access_failure_audit()
    sidecar = _sidecar_hashes()
    negative = _negative_controls()
    static = _static_source_audit()
    gpu = _gpu_snapshot()
    rerun = subprocess.run([str(RERUN_CLI), "--version"], text=True, capture_output=True, check=False)
    display = subprocess.run(["xdpyinfo", "-display", DISPLAY], text=True, capture_output=True, check=False)
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_d364_d365_only": _status_scope_ok(_git_status()),
        "session_preregistered": SESSION_DOC.is_file()
        and "USER_APPROVED_IMPLEMENTATION_IN_PROGRESS_NO_D365_ISAAC_INVOCATION" in SESSION_DOC.read_text(encoding="utf-8"),
        "input_hashes_exact": input_hashes == EXPECTED_INPUT_HASHES,
        "d362_manifest_33": d362_manifest["file_count"] == 33,
        "d363_manifest_40": d363_manifest["file_count"] == 40,
        "d364_manifest_17": d364_manifest["file_count"] == 17,
        "d364_completion_evidence_map_exact": d364_evidence_audit["pass"],
        "prepare_attempt1_access_failure_exact_no_invocation": prepare_attempt1_audit["pass"],
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
        "artifact": "D365_PREREGISTRATION_V1",
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
        "d364_completion_evidence_audit": d364_evidence_audit,
        "prepare_attempt1_access_failure_audit": prepare_attempt1_audit,
        "d334_sidecar_before": sidecar,
        "target_row_index": TARGET_ROW_INDEX,
        "expected_global_step": EXPECTED_GLOBAL_STEP,
        "root_prim_path": ROOT_PRIM_PATH,
        "mesh_prim_path": MESH_PRIM_PATH,
        "registered_counts": {
            "root_pose_write": 1,
            "explicit_forward": 1,
            "controlled_physics_step": 0,
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
        "artifact": "D365_PREPARE_PREFLIGHT_V1",
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
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_stopped": bool(timeline.is_stopped()),
        "timeline_time_s": float(timeline.get_current_time()),
    }


def _clock_no_advance(reference: dict[str, Any], observed: dict[str, Any]) -> bool:
    return bool(
        observed["custom_step_counter"] == reference["custom_step_counter"]
        and observed["simulation_clock"] == reference["simulation_clock"]
        and observed["timeline_time_s"] == reference["timeline_time_s"]
        and not observed["timeline_playing"]
        and not observed["timeline_stopped"]
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
        "controlled_counter_unchanged": before["custom_step_counter"] == after["custom_step_counter"],
        "simulation_clock_unchanged": before["simulation_clock"] == after["simulation_clock"],
    }
    return {"before": before, "after": after, "commit_count": commit_count, "checks": checks, "pass": all(checks.values())}


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
        raise RuntimeError(f"D365 PhysX transform shape drift: {raw.shape}")
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
        raise RuntimeError(f"D365 capture timeline contract failed: {role}")
    inner.sim.set_setting(PLAY_SIMULATIONS_SETTING, False)
    inner.sim.set_camera_view(camera_eye, CAMERA_TARGET)
    reference = _clock_snapshot(inner, timeline)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None or not hasattr(viewport, "set_texture_resolution"):
        raise RuntimeError("D365 active viewport unavailable")
    viewport.set_texture_resolution(tuple(VIEWPORT_SIZE))
    app_updates = 0
    for _ in range(8):
        simulation_app.update()
        app_updates += 1
        if not _clock_no_advance(reference, _clock_snapshot(inner, timeline)):
            raise RuntimeError(f"D365 app update advanced physics clock: {role}")
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(capture.wait_for_result(completion_frames=5), run_until_complete=False)
    deadline = time.monotonic() + CAPTURE_TIMEOUT_S
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()
        app_updates += 1
        if not _clock_no_advance(reference, _clock_snapshot(inner, timeline)):
            raise RuntimeError(f"D365 capture advanced physics clock: {role}")
    if not task.done():
        task.cancel()
        raise RuntimeError(f"D365 capture timeout: {role}")
    if not bool(task.result()):
        raise RuntimeError(f"D365 capture failed: {role}")
    for _ in range(2):
        simulation_app.update()
        app_updates += 1
    terminal = _clock_snapshot(inner, timeline)
    if not _clock_no_advance(reference, terminal):
        raise RuntimeError(f"D365 post-capture physics clock drift: {role}")
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


def _worker(args: argparse.Namespace) -> int:
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
            "git_scope_d364_d365_only": _status_scope_ok(_git_status()),
            "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
            "session_preregistration_hash_exact": _sha(SESSION_DOC)
            == prereg.get("session_preregistration_sha256")
            and SESSION_DOC.stat().st_size == prereg.get("session_preregistration_bytes"),
            "input_hashes_exact": _input_hashes() == prereg.get("input_hashes"),
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
            "artifact": "D365_WORKER_PREFLIGHT_V1",
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
            raise RuntimeError(f"D365 worker preflight STOP: {checks}")

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
            raise RuntimeError(f"D365 GUI launcher contract failed: {launcher_report}")

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
            raise RuntimeError(f"D365 pause bridge failed: {pause['checks']}")

        rows = _json(D362_TRACE)
        if len(rows) != 500 or rows[TARGET_ROW_INDEX].get("global_step") != EXPECTED_GLOBAL_STEP:
            raise RuntimeError("D365 frozen target-row contract failed")
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
            "artifact": "D365_RUNTIME_FABRIC_ATTESTATION_V1",
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
            raise RuntimeError(f"D365 runtime Fabric prerequisites STOP: {runtime_checks}")

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
            raise RuntimeError("D365 explicit forward advanced physics clock")
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
            "artifact": "D365_WORKER_SUMMARY_V1",
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
                    "artifact": "D365_WORKER_EXCEPTION_STOP_V1",
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
        "six_records": len(rows) == 6,
        "labels_exact": labels == expected_labels,
        "hash_chain_exact": not errors,
        "all_getter_clock_guards_true": all(
            row.get("payload", {}).get("clock_unchanged_across_getter") is True for row in rows
        ),
    }
    audit = {
        "artifact": "D365_LAYER_READBACK_JOURNAL_AUDIT_V1",
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
        raise RuntimeError(f"D365 PNG decode failed: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, MASK_HSV_LOW, MASK_HSV_HIGH)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError(f"D365 yellow component absent: {path}")
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


def _phase_layers(
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


def _build_localization_report(worker: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_checkpoint = worker["checkpoints"]["baseline_post_capture"]
    baseline_pose = baseline_checkpoint["independent_physx_tensor_view"]["pose_wxyz"]
    target_pose = worker["write"]["target_pose_wxyz"]
    geometry = worker["expected_geometry"]
    baseline_root_matrix = geometry["baseline_root_computed"]["matrix_row_major"]
    target_root_matrix = geometry["target_root_matrix"]["matrix_row_major"]
    baseline_mesh_matrix = geometry["baseline_mesh_computed"]["matrix_row_major"]
    target_mesh_matrix = geometry["target_mesh_matrix_row_major"]
    phase_layers = {
        label: _phase_layers(
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
        verdict = "D365_MEASUREMENT_INCOMPLETE_FAIL_STOP"
    elif not terminal_classes_binary:
        verdict = "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    elif (
        getter_side_effect_detected
        or not baseline_layers_exact
        or any_temporal_regression
        or any_temporal_other
        or any_phase_downstream_ahead
    ):
        verdict = "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP"
    elif terminal_classes["cache"] != "TARGET":
        verdict = "D365_DIRECT_WRITE_OR_CACHE_FAIL"
    elif terminal_classes["physx"] != "TARGET":
        verdict = "D365_CACHE_TO_PHYSX_PENDING_OR_FAILED"
    elif terminal_classes["root_current"] != "TARGET":
        verdict = "D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED"
    elif terminal_classes["mesh_current"] != "TARGET":
        verdict = "D365_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED"
    elif terminal_classes["mesh_cached"] != "TARGET":
        verdict = "D365_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED"
    elif terminal_classes["hydra"] != "TARGET":
        verdict = "D365_FABRIC_TO_HYDRA_NOT_PROPAGATED"
    else:
        verdict = "D365_END_TO_END_ZERO_STEP_VISIBLE"

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
        "D365_MEASUREMENT_INCOMPLETE_FAIL_STOP",
        "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP",
    }
    report = {
        "artifact": "D365_STATE_LAYER_LOCALIZATION_REPORT_V1",
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


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_PATH), size=size)


def _build_sheet(report: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    width, height = 4800, 3000
    canvas = Image.new("RGB", (width, height), (15, 20, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 52), "D365 상태 전달 단절 측정 — 원본 Isaac 화면과 계층별 readback", font=_font(68), fill=(247, 249, 252), anchor="ma")
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
        "D365_DIRECT_WRITE_OR_CACHE_FAIL": "등록된 pose write 뒤 AssetData cache 자체가 final을 보존하지 못했습니다.",
        "D365_CACHE_TO_PHYSX_PENDING_OR_FAILED": "cache에는 final이 있지만 독립 PhysX tensor view에는 전달되지 않았습니다.",
        "D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED": "PhysX tensor view에는 final이 있지만 Fabric root hierarchy current에는 전달되지 않았습니다.",
        "D365_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED": "Fabric root는 final이지만 실제 mesh hierarchy에는 전달되지 않았습니다.",
        "D365_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED": "mesh 현재값은 final이지만 renderer cached matrix는 이전값입니다.",
        "D365_FABRIC_TO_HYDRA_NOT_PROPAGATED": "Fabric render mesh는 final이지만 실제 RTX 픽셀은 이전 자세입니다.",
        "D365_END_TO_END_ZERO_STEP_VISIBLE": "이번 1회 측정에서는 zero-step 전달 전 구간이 화면까지 보였습니다.",
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


def _quat_rotation_wxyz(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1.0e-15:
        raise RuntimeError("D365 zero-norm quaternion")
    w, x, y, z = (np.asarray([w, x, y, z], dtype=np.float64) / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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
        raise RuntimeError(f"D365 rerun version drift: {rr.__version__}")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/summary/localization_sheet", name="D365 상태 전달 요약"),
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
    app_id = "roarm_g0a_d365_state_layer_localization"
    with rr.RecordingStream(
        app_id,
        recording_id="g0a_d365_state_layer_localization",
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
            raise RuntimeError("D365 localization sheet decode failed")
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
                raise RuntimeError(f"D365 Rerun capture decode failed: {capture_phase}")
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
            ("timeline_pause", "complete"),
            ("baseline_layers", "start"),
            ("baseline_layers", "complete"),
            ("direct_root_pose_write", "start"),
            ("direct_root_pose_write", "complete"),
            ("explicit_simulation_context_forward", "start"),
            ("explicit_simulation_context_forward", "complete"),
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
                "six_capture_markers_unique": len(capture_rows) == 6
                and set(observed_roles) == expected_roles
                and len(set(observed_roles)) == 6,
                "six_capture_marker_order_exact": observed_roles == expected_role_order,
                "one_forward_marker_count_one": len(forward_rows) == 1
                and forward_rows[0].get("details", {}).get("forward_count") == 1,
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
        raise RuntimeError("D365 prepare/preregistration did not pass")
    existing_inventory = sorted(path.resolve() for path in OUT_DIR.rglob("*") if path.is_file())
    expected_prepare_inventory = sorted([PREREG_PATH.resolve(), PREPARE_PATH.resolve()])
    if existing_inventory != expected_prepare_inventory:
        raise RuntimeError(
            "D365 run/resume/overwrite forbidden; output inventory is not prepare-only: "
            f"{[_rel(path) for path in existing_inventory]}"
        )
    checks = {
        "head_origin_exact": _git_head() == _git_head("origin/master") == BASE_GIT,
        "git_scope_d364_d365_only": _status_scope_ok(_git_status()),
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "session_preregistration_hash_exact": _sha(SESSION_DOC)
        == prereg["session_preregistration_sha256"]
        and SESSION_DOC.stat().st_size == prereg["session_preregistration_bytes"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
        "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
        "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
        "sidecar_exact": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "prereg_artifact_case_exact": prereg.get("artifact") == "D365_PREREGISTRATION_V1"
        and prereg.get("case") == CASE,
        "prepare_artifact_case_exact": prepare.get("artifact") == "D365_PREPARE_PREFLIGHT_V1"
        and prepare.get("case") == CASE,
        "prepare_binds_prereg_hash": prepare.get("preregistration_sha256") == _sha(PREREG_PATH),
        "prepare_all_checks_pass": prepare.get("pass") is True
        and bool(prepare.get("checks"))
        and all(prepare["checks"].values()),
        "prepare_only_inventory_exact": existing_inventory == expected_prepare_inventory,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D365 pre-invocation STOP: {checks}")
    _marker("supervisor", "supervisor", "start", {"checks": checks})
    token = secrets.token_hex(32)
    invocation = {
        "artifact": "D365_ISAAC_INVOCATION_MARKER_V1",
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
                "artifact": "D365_SUPERVISOR_POSTPROCESS_EXCEPTION_V1",
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
        "worker_exit_zero": worker_exit == 0,
        "worker_exception_absent": not WORKER_EXCEPTION_PATH.exists(),
        "watchdog_not_triggered": watchdog_reason is None,
        "worker_summary_pass": worker is not None and worker.get("pass") is True,
        "all_six_png_stable_after_close": len(stability) == 6
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
        "d334_sidecar_unchanged": _sidecar_hashes() == prereg["d334_sidecar_before"],
        "input_hashes_unchanged": _input_hashes() == prereg["input_hashes"],
    }
    supervisor = {
        "artifact": "D365_SUPERVISOR_SUMMARY_V1",
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
        "display_write_and_forward_exact_one": supervisor["display_state_write_count"] == 1
        and supervisor["display_state_write_returned"] is True
        and supervisor["explicit_forward_count"] == 1
        and supervisor["explicit_forward_returned"] is True,
        "controlled_physics_zero": supervisor["controlled_physics_steps"] == 0,
        "q5_science_and_target_zero": supervisor["q5_science_sample_count"] == 0
        and supervisor["q5_target_update_count"] == 0,
        "contact_query_zero": supervisor["contact_query_count"] == 0,
        "d362_d363_d364_immutable": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"]
        and _tree_manifest(D363_DIR) == prereg["d363_manifest_before"]
        and _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
        "bound_artifacts_nonempty": bool(bound_artifact_hashes)
        and all(bool(digest) for digest in bound_artifact_hashes.values()),
    }
    automated = {
        "artifact": "D365_AUTOMATED_SUMMARY_V1",
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
        raise RuntimeError("D365 completion overwrite forbidden")
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    worker = _json(WORKER_SUMMARY_PATH)
    report = _json(REPORT_PATH)
    expected_paths = _all_required_visual_paths()
    expected_manual_visual_class = {
        "TARGET": "target_toppled",
        "BASELINE": "baseline_upright",
        "OTHER": "visually_ambiguous",
        "UNAVAILABLE": "visually_ambiguous",
    }.get(report.get("terminal_classes", {}).get("hydra"), "visually_ambiguous")
    manual_checks = {
        "artifact_exact": manual.get("artifact") == "D365_MANUAL_VISUAL_INSPECTION_V1",
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
        "post_forward_visual_class_recorded": manual.get("post_forward_visual_class")
        in {"baseline_upright", "target_toppled", "visually_ambiguous"},
        "manual_visual_class_matches_automated_hydra": manual.get(
            "post_forward_visual_class"
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
        "automated_pass_before_manual": automated.get("pass_before_manual") is True,
        "manual_checks": all(manual_checks.values()),
        "harness_hash_exact": _sha(HARNESS) == prereg["harness_sha256"],
        "session_preregistration_hash_exact": _sha(SESSION_DOC)
        == prereg["session_preregistration_sha256"]
        and SESSION_DOC.stat().st_size == prereg["session_preregistration_bytes"],
        "input_hashes_exact": _input_hashes() == prereg["input_hashes"],
        "d362_manifest_exact": _tree_manifest(D362_DIR) == prereg["d362_manifest_before"],
        "d363_manifest_exact": _tree_manifest(D363_DIR) == prereg["d363_manifest_before"],
        "d364_manifest_exact": _tree_manifest(D364_DIR) == prereg["d364_manifest_before"],
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
        and worker.get("explicit_forward_count") == 1
        and worker.get("explicit_forward_returned") is True
        and worker.get("controlled_physics_steps") == 0
        and worker.get("q5_science_sample_count") == 0
        and worker.get("q5_target_update_count") == 0
        and worker.get("contact_query_count") == 0,
        "localization_report_concrete": report.get("pass") is True
        and report.get("localization_verdict") not in {
            "D365_MEASUREMENT_INCOMPLETE_FAIL_STOP",
            "D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP",
        },
        "rerun_validation_pass": _json(RERUN_VALIDATION_PATH).get("pass") is True,
    }
    completion = {
        "artifact": "D365_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "utc": _utc_now(),
        "final_verdict": report.get("localization_verdict")
        if all(integrity_checks.values())
        else "D365_OBSERVABILITY_OR_INTEGRITY_FAIL_STOP",
        "manual_checks": manual_checks,
        "integrity_checks": integrity_checks,
        "terminal_classes": report.get("terminal_classes"),
        "display_state_write_count": worker.get("display_state_write_count"),
        "display_state_write_returned": worker.get("display_state_write_returned"),
        "explicit_forward_count": worker.get("explicit_forward_count"),
        "explicit_forward_returned": worker.get("explicit_forward_returned"),
        "controlled_physics_steps": worker.get("controlled_physics_steps"),
        "q5_science_sample_count": worker.get("q5_science_sample_count"),
        "q5_target_update_count": worker.get("q5_target_update_count"),
        "contact_query_count": worker.get("contact_query_count"),
        "counter_semantics": "registered D365 calls only; reset-internal transition excluded",
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
        raise RuntimeError("D365 output path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D365 seed drift")
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
