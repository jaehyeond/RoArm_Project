#!/usr/bin/env python3
"""D361 offline contact-capacity and durable prefix-trace control repair.

This module intentionally does not launch Isaac Sim/Kit, execute PhysX, sample
q5, or generate spatial/temporal media.  Its only runtime experiment is an
offline failure-injection evaluation of the evidence writer and verifier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d361"
HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260716_grasp_g0a_d361_contact_point_capacity_and_prefix_trace_repair.md"
START_HERE = REPO / "START_HERE.md"

BASE_GIT = "e7ed71ca80768df9037c16e53a12d3c032af3d5d"
CASE = "g0a_d361"
CASE_NAME = "contact_point_capacity_and_prefix_trace_repair"
NEW_VARIABLES = [
    "version_aligned_total_contact_point_capacity_budget",
    "durable_framed_step_prefix_protocol",
]

D360_SESSION = REPO / "claudedocs/session_20260716_grasp_g0a_d360_current_pose_bounded_physx_contact_motion.md"
D360_HARNESS = REPO / "sim_scripts/cyl34_top_view_d360_current_pose_bounded_physx_contact_motion.py"
D360_PREREQUISITES = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_runtime_prerequisites.json"
D360_EXCEPTION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_exception.json"
D360_SUPERVISOR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360/d360_supervisor_summary.json"
D360_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d360"
D333_HARNESS = REPO / "sim_scripts/cyl34_top_view_d333_grasp_g0a_sole_support_static_retest.py"
VARIANT_PHYSICS_USD = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/"
    "roarm_m3_fullmesh_fixed_point_parts/configuration/roarm_m3_physics.usd"
)

ISAACLAB_SENSOR_CFG = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py"
)
ISAACLAB_SENSOR_IMPL = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py"
)
ISAACLAB_SHAPE_SPAWNER = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/"
    "source/isaaclab/isaaclab/sim/spawners/shapes/shapes.py"
)
OMNI_TENSOR_API = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "omni/physics/tensors/impl/api.py"
)
PHYSX_PLUGIN = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/bin/"
    "libomni.physx.plugin.so"
)

USD_CORE_EXT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
ISAAC_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
PXR_PYTHONPATH = str(USD_CORE_EXT)
PXR_LD_LIBRARY_PATH = ":".join(("/home/cgxr/miniconda3/envs/isaaclab/lib", str(USD_CORE_EXT / "bin")))

EXPECTED_HASHES = {
    D360_SESSION: "97ef49eb31a754be4f12ea0c5f961ddfede4d19656df7147e3f10a63b21f9291",
    D360_HARNESS: "86bd2af855effb3bc31f067fd6cc7a4cb7088c422da3ae828d48f4e31d92fd5a",
    D360_PREREQUISITES: "bfd20dc2ab678d9929c0dd8f54323f8f4b17707c6c89d099b1374b814c467dae",
    D360_EXCEPTION: "1dfb7ffe863f77ece3dc8acafc09b134325b147f1a392907ad5692a9fd650fb1",
    D360_SUPERVISOR: "54bb8a80569048e0299183ae8ed86e81b82a527ec67329c43d4c9e8cb12f026c",
    D333_HARNESS: "e582f274fca44093b0e1367555459f22428c809792b6cfc3a9a336369dac68b7",
    ISAACLAB_SENSOR_CFG: "adb530a2d26ec0ca21160a20c2491c921267764915c3b29108a4ad1bd88171f8",
    ISAACLAB_SENSOR_IMPL: "c2b039eb46d55416a8699d82b2385abae563c3c4ab4404ad08fa68310ffa6c64",
    ISAACLAB_SHAPE_SPAWNER: "e39d452c3a5e5f927197a99f6ca74d95914e910f96ae1d7a1b1ef01538df6ef7",
    OMNI_TENSOR_API: "5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f",
    PHYSX_PLUGIN: "03fbf17e6f0dc3f9006c8c00aa0ca572a72fd69498874df6dd900dac726c9909",
    VARIANT_PHYSICS_USD: "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
}

D334_SIDECAR = {
    REPO / "claudedocs/lab_meeting/20260715/d334_collision_table/README.md":
        "35e39f584737c888bcf7dfab6154c55c5d13d4154ee7f2042073e1c0a7e18783",
    REPO / "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.html":
        "6d38933f959eba916208ec04a329ba25e2bd753c90720576010c222a8bda679c",
    REPO / "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.png":
        "ddc9db2795f4d66b2564adf156829e6a143a599ceb72f6bb9fa28ab25e68a183",
}

PHYSX_CONTACTS_PER_GEOMETRY_PAIR = 256
EXPECTED_SENSOR_SHAPES = 1
EXPECTED_FILTER_SHAPES = {
    "support_table": 1,
    "link4": 1,
    "link5": 64,
    "gripper_link": 64,
}
EXPECTED_FILTER_SHAPE_TOTAL = 130
REGISTERED_TOTAL_CAPACITY = 33_280
FLOAT32_BYTES = 4
DETAIL_FLOATS_PER_CONTACT = 8  # force1 + point3 + normal3 + separation1
REGISTERED_DETAIL_BYTES = 1_064_960

CONTACT_THRESHOLD_N = 0.1
MOTION_XY_THRESHOLD_MM = 0.5
MOTION_TILT_THRESHOLD_DEG = 1.0
BODY_LABELS = ("support_table", "link4", "link5", "gripper_link")
ROBOT_CONTACT_LABELS = ("link4", "link5", "gripper_link")
D360_STATE_ROW_KEYS = (
    "global_step",
    "phase",
    "phase_step",
    "physics_time_s",
    "simulation_clock",
    "timeline_time_s",
    "actual_joint_rad",
    "actual_joint_vel_rad_s",
    "target_joint_rad",
    "q5_actual_rad",
    "q5_velocity_rad_s",
    "q5_target_rad",
    "q5_error_rad",
    "q0_q4_actual_minus_frozen_rad",
    "q0_q4_max_abs_drift_rad",
    "q5_torque",
    "object_pos_w_m",
    "object_quat_wxyz",
    "object_lin_vel_w_mps",
    "object_ang_vel_w_radps",
    "object_disp_w_m",
    "object_disp_xy_mm",
    "object_z_delta_mm",
    "object_tilt_deg",
    "object_tilt_delta_from_reference_deg",
    "object_bottom_table_gap_mm",
    "robot_root_pos_w_m",
    "robot_root_quat_wxyz",
    "robot_root_position_drift_m",
    "robot_root_rotation_drift_rad",
    "link5_pos_w_m",
    "link5_quat_wxyz",
    "gripper_pos_w_m",
    "gripper_quat_wxyz",
    "contact",
    "finite",
)
D360_ARRAY_LENGTHS = {
    "actual_joint_rad": 6,
    "actual_joint_vel_rad_s": 6,
    "target_joint_rad": 6,
    "q0_q4_actual_minus_frozen_rad": 5,
    "object_pos_w_m": 3,
    "object_quat_wxyz": 4,
    "object_lin_vel_w_mps": 3,
    "object_ang_vel_w_radps": 3,
    "object_disp_w_m": 3,
    "robot_root_pos_w_m": 3,
    "robot_root_quat_wxyz": 4,
    "link5_pos_w_m": 3,
    "link5_quat_wxyz": 4,
    "gripper_pos_w_m": 3,
    "gripper_quat_wxyz": 4,
}
D360_SCALAR_FIELDS = (
    "physics_time_s",
    "q5_actual_rad",
    "q5_velocity_rad_s",
    "q5_target_rad",
    "q5_error_rad",
    "q0_q4_max_abs_drift_rad",
    "object_disp_xy_mm",
    "object_z_delta_mm",
    "object_tilt_deg",
    "object_tilt_delta_from_reference_deg",
    "object_bottom_table_gap_mm",
    "robot_root_position_drift_m",
    "robot_root_rotation_drift_rad",
)

PREFIX_SCHEMA = "D361_DURABLE_STEP_PREFIX_V1"
PREFIX_PROFILE = "D361_OFFLINE_SYNTHETIC_PREFIX_EVALUATION_V1"
ZERO_HASH = "0" * 64
FORBIDDEN_MAIN_MODULES = ("isaacsim", "isaaclab", "omni", "carb", "torch", "warp")
SENSOR_BODY_NAMES = ("Sponge",)
RESOLVED_FILTER_PATHS = (
    "/World/envs/env_0/TapTable",
    "/World/envs/env_0/Robot/link4",
    "/World/envs/env_0/Robot/link5",
    "/World/envs/env_0/Robot/gripper_link",
)

PREREG_PATH = OUT_DIR / "d361_preregistration.json"
PREPARE_PATH = OUT_DIR / "d361_prepare_preflight.json"
INVOCATION_PATH = OUT_DIR / "d361_offline_invocation_marker.json"
PHASE_PATH = OUT_DIR / "d361_phase_markers.jsonl"
CAPACITY_PATH = OUT_DIR / "d361_contact_capacity_budget.json"
PROTOCOL_PATH = OUT_DIR / "d361_prefix_protocol_contract.json"
PERTURBATION_PATH = OUT_DIR / "d361_failure_injection_results.json"
FAILURE_JOURNAL_PATH = OUT_DIR / "d361_failure_injection_results.jsonl"
COMPLETION_PATH = OUT_DIR / "d361_completion_summary.json"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d361_runtime_exception.json"
FIXTURE_DIR = OUT_DIR / "failure_injection"
EXPECTED_FIXTURE_FILENAMES = (
    "reference_sealed_prefix.jsonl",
    "exit_after_begin.jsonl",
    "exit_after_observation.jsonl",
    "partial_tail.jsonl",
    "body_force_byte_flip.jsonl",
    "record_reorder.jsonl",
    "middle_record_delete.jsonl",
    "duplicate_sequence.jsonl",
    "event_semantic_tamper_rehashed.jsonl",
    "header_semantic_tamper_rehashed.jsonl",
    "premature_seal_rehashed.jsonl",
    "state_row_semantic_tamper_rehashed.jsonl",
    "schema_reject_missing_body.jsonl",
    "nan_reject.jsonl",
)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(data):
        count = os.write(fd, view[written:])
        if count <= 0:
            raise OSError("short write made no progress")
        written += count


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_exclusive(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)
    reread = path.read_bytes()
    if reread != data:
        raise RuntimeError(f"exclusive write reread mismatch: {path}")
    return _sha256_bytes(data)


def _write_json_exclusive(path: Path, payload: Any) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    return _write_bytes_exclusive(path, data)


def _expected_artifact_paths(*, include_completion: bool) -> set[Path]:
    paths = {
        PREREG_PATH,
        PREPARE_PATH,
        INVOCATION_PATH,
        PHASE_PATH,
        CAPACITY_PATH,
        PROTOCOL_PATH,
        PERTURBATION_PATH,
        FAILURE_JOURNAL_PATH,
        *(FIXTURE_DIR / name for name in EXPECTED_FIXTURE_FILENAMES),
    }
    if include_completion:
        paths.add(COMPLETION_PATH)
    return paths


def _prefix_lineage() -> dict[str, str]:
    paths = {
        "preregistration_sha256": PREREG_PATH,
        "prepare_preflight_sha256": PREPARE_PATH,
        "offline_invocation_sha256": INVOCATION_PATH,
        "harness_sha256": HARNESS,
        "capacity_budget_sha256": CAPACITY_PATH,
        "protocol_contract_sha256": PROTOCOL_PATH,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"prefix lineage inputs are missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def _tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        _rel(path): {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    # Preserve the leading XY status column (for example `` M path``).  Using
    # strip() here deletes the first column of the first porcelain-status row.
    return proc.stdout.rstrip()


def _scope_status_ok(status: str) -> tuple[bool, list[str]]:
    allowed_exact = {
        "START_HERE.md",
        _rel(SESSION_DOC),
        _rel(HARNESS),
    }
    allowed_prefix = "claudedocs/runtime_logs/grasp_track/g0a_d361/"
    unexpected: list[str] = []
    for line in status.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed_exact and not path.startswith(allowed_prefix):
            unexpected.append(line)
    return not unexpected, unexpected


def _forbidden_loaded_modules() -> list[str]:
    return sorted(
        name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MAIN_MODULES)
    )


class DurableMarkerStream:
    """Small durable phase stream; independent of the tested prefix protocol."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC,
            0o644,
        )
        self.sequence = 0
        self.offset = 0
        _fsync_directory(path.parent)

    def append(self, phase: str, **payload: Any) -> None:
        record = {"sequence": self.sequence, "utc": _utc(), "phase": phase, **payload}
        data = _canonical(record) + b"\n"
        before = os.fstat(self.fd).st_size
        if before != self.offset:
            raise RuntimeError("phase stream offset drift")
        _write_all(self.fd, data)
        os.fsync(self.fd)
        read_fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            reread = os.pread(read_fd, len(data), self.offset)
        finally:
            os.close(read_fd)
        if reread != data:
            raise RuntimeError("phase stream append reread mismatch")
        self.offset += len(data)
        self.sequence += 1

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _validate_step_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("step_identity must be an object")
    required = ("global_step", "phase", "phase_step")
    if set(identity) != set(required):
        raise ValueError("step_identity keys are not exact")
    if not isinstance(identity["global_step"], int) or identity["global_step"] < 0:
        raise ValueError("global_step must be a non-negative integer")
    if not isinstance(identity["phase_step"], int) or identity["phase_step"] < 0:
        raise ValueError("phase_step must be a non-negative integer")
    if not isinstance(identity["phase"], str) or not identity["phase"]:
        raise ValueError("phase must be a non-empty string")
    return identity


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_header_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("header payload must be an object")
    required = {
        "case",
        "scenario",
        "profile",
        "subject_kind",
        "registered_total_capacity",
        "sensor_body_names",
        "body_labels",
        "resolved_filter_paths",
        "filter_index_by_body",
        "state_row_contract",
        "lineage",
        "execution_contract",
        "resume_allowed",
        "overwrite_allowed",
    }
    if set(payload) != required:
        raise ValueError("header payload keys are not exact")
    if payload["case"] != CASE or not isinstance(payload["scenario"], str) or not payload["scenario"]:
        raise ValueError("header case/scenario contract mismatch")
    if payload["profile"] != PREFIX_PROFILE:
        raise ValueError("header execution profile mismatch")
    if payload["subject_kind"] not in {
        "offline_synthetic_protocol_fixture_not_physics_evidence",
        "future_actual_d360_inherited_state_requires_separate_approval",
    }:
        raise ValueError("header subject kind is not registered")
    if payload["registered_total_capacity"] != REGISTERED_TOTAL_CAPACITY:
        raise ValueError("header capacity mismatch")
    if payload["sensor_body_names"] != list(SENSOR_BODY_NAMES):
        raise ValueError("header sensor-body contract mismatch")
    if payload["body_labels"] != list(BODY_LABELS):
        raise ValueError("header body label order mismatch")
    if payload["resolved_filter_paths"] != list(RESOLVED_FILTER_PATHS):
        raise ValueError("header resolved filter-path order mismatch")
    if payload["filter_index_by_body"] != {label: index for index, label in enumerate(BODY_LABELS)}:
        raise ValueError("header filter-index map mismatch")
    state_contract = payload["state_row_contract"]
    if state_contract != {
        "authority": "D360 _state_row exact top-level field contract",
        "required_top_level_keys": list(D360_STATE_ROW_KEYS),
    }:
        raise ValueError("header D360 state-row contract mismatch")
    lineage = payload["lineage"]
    if not isinstance(lineage, dict) or set(lineage) != {
        "preregistration_sha256",
        "prepare_preflight_sha256",
        "offline_invocation_sha256",
        "harness_sha256",
        "capacity_budget_sha256",
        "protocol_contract_sha256",
    }:
        raise ValueError("header lineage keys are not exact")
    if not all(_is_sha256(value) for value in lineage.values()):
        raise ValueError("header lineage contains an invalid SHA-256")
    if lineage != _prefix_lineage():
        raise ValueError("header lineage does not match the frozen D361 artifacts")
    execution = payload["execution_contract"]
    if not isinstance(execution, dict) or set(execution) != {"legal_seals"}:
        raise ValueError("header execution contract is invalid")
    legal_seals = execution["legal_seals"]
    if not isinstance(legal_seals, dict) or not legal_seals:
        raise ValueError("header legal-seal map must be non-empty")
    for reason, counts in legal_seals.items():
        if not isinstance(reason, str) or not reason or not isinstance(counts, list) or not counts:
            raise ValueError("header legal-seal entry is invalid")
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("header legal-seal counts must be non-negative integers")
    if payload["resume_allowed"] is not False or payload["overwrite_allowed"] is not False:
        raise ValueError("header must forbid resume and overwrite")


def _validate_full_d360_state_row(state: dict[str, Any]) -> None:
    if set(state) != set(D360_STATE_ROW_KEYS):
        missing = sorted(set(D360_STATE_ROW_KEYS) - set(state))
        extra = sorted(set(state) - set(D360_STATE_ROW_KEYS))
        raise ValueError(f"state_row top-level field set mismatch: missing={missing}, extra={extra}")
    for key in D360_SCALAR_FIELDS:
        _finite_number(state[key], f"state_row.{key}")
    timeline = state["timeline_time_s"]
    if timeline is not None:
        _finite_number(timeline, "state_row.timeline_time_s")
    for key, length in D360_ARRAY_LENGTHS.items():
        values = state[key]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"state_row.{key} must be a length-{length} list")
        for index, value in enumerate(values):
            _finite_number(value, f"state_row.{key}[{index}]")
    _validate_step_identity({key: state[key] for key in ("global_step", "phase", "phase_step")})
    clock = state["simulation_clock"]
    if not isinstance(clock, dict) or set(clock) != {"current_time", "current_time_step_index"}:
        raise ValueError("state_row.simulation_clock keys are not the inherited D360 contract")
    _finite_number(clock["current_time"], "state_row.simulation_clock.current_time")
    if not isinstance(clock["current_time_step_index"], int) or clock["current_time_step_index"] < 0:
        raise ValueError("state_row.simulation_clock.current_time_step_index must be a non-negative integer")
    if not math.isclose(float(state["physics_time_s"]), float(clock["current_time"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("state_row physics time and simulation clock disagree")
    torque = state["q5_torque"]
    torque_keys = {
        "authority",
        "applied_torque_nm",
        "computed_torque_nm",
        "registered_effort_limit_nm",
        "abs_applied_over_limit",
        "effort_saturated_diagnostic",
        "saturation_abs_tolerance_nm",
    }
    if not isinstance(torque, dict) or set(torque) != torque_keys or not isinstance(torque["authority"], str):
        raise ValueError("state_row.q5_torque contract mismatch")
    for key in torque_keys - {"authority", "effort_saturated_diagnostic"}:
        _finite_number(torque[key], f"state_row.q5_torque.{key}")
    if not isinstance(torque["effort_saturated_diagnostic"], bool):
        raise ValueError("state_row.q5_torque.effort_saturated_diagnostic must be boolean")
    q5_consistency = (
        math.isclose(float(state["q5_actual_rad"]), float(state["actual_joint_rad"][5]), rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            float(state["q5_velocity_rad_s"]),
            float(state["actual_joint_vel_rad_s"][5]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(float(state["q5_target_rad"]), float(state["target_joint_rad"][5]), rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            float(state["q5_error_rad"]),
            float(state["q5_target_rad"]) - float(state["q5_actual_rad"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    if not q5_consistency:
        raise ValueError("state_row q5 scalar fields disagree with the full joint vectors")
    if state["finite"] is not True:
        raise ValueError("state_row finite gate must be true")


def _validate_observation_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("observation payload must be an object")
    if set(payload) != {
        "subject_kind",
        "step_identity",
        "state_row",
        "contact_point_capacity_diagnostic",
        "event_projection",
    }:
        raise ValueError("observation payload keys are not exact")
    if payload["subject_kind"] not in {
        "offline_synthetic_protocol_fixture_not_physics_evidence",
        "future_actual_d360_inherited_state_requires_separate_approval",
    }:
        raise ValueError("observation subject kind is not registered")
    identity = _validate_step_identity(payload.get("step_identity"))
    state = payload.get("state_row")
    events = payload.get("event_projection")
    if not isinstance(state, dict) or not isinstance(events, dict):
        raise ValueError("observation requires state_row and event_projection objects")
    _validate_full_d360_state_row(state)
    if [state["global_step"], state["phase"], state["phase_step"]] != [
        identity["global_step"], identity["phase"], identity["phase_step"]
    ]:
        raise ValueError("step_identity does not match state_row")
    contact = state["contact"]
    if not isinstance(contact, dict) or set(contact) != {"by_filter", "net_force_w_n", "net_force_norm_n"}:
        raise ValueError("state_row contact keys are not exact")
    if not isinstance(contact["by_filter"], dict) or set(contact["by_filter"]) != set(BODY_LABELS):
        raise ValueError("state_row contact.by_filter body set is not exact")
    for label in BODY_LABELS:
        row = contact["by_filter"][label]
        if not isinstance(row, dict) or set(row) != {
            "filter_index", "force_w_n", "force_norm_n", "contact_point_w_m"
        }:
            raise ValueError(f"contact body {label} keys are not exact")
        if row.get("filter_index") != BODY_LABELS.index(label):
            raise ValueError(f"contact body {label} filter index mismatch")
        force = row.get("force_w_n")
        if not isinstance(force, list) or len(force) != 3:
            raise ValueError(f"contact body {label} force_w_n must have length 3")
        for index, value in enumerate(force):
            _finite_number(value, f"{label}.force_w_n[{index}]")
        force_norm = _finite_number(row.get("force_norm_n"), f"{label}.force_norm_n")
        recomputed_norm = math.sqrt(sum(float(value) ** 2 for value in force))
        if not math.isclose(force_norm, recomputed_norm, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"contact body {label} force norm does not match force vector")
        point = row.get("contact_point_w_m")
        if point is not None:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"contact body {label} contact point must be null or length 3")
            for index, value in enumerate(point):
                _finite_number(value, f"{label}.contact_point_w_m[{index}]")
    net_force = contact.get("net_force_w_n")
    if not isinstance(net_force, list) or len(net_force) != 3:
        raise ValueError("contact.net_force_w_n must have length 3")
    for index, value in enumerate(net_force):
        _finite_number(value, f"contact.net_force_w_n[{index}]")
    net_force_norm = _finite_number(contact.get("net_force_norm_n"), "contact.net_force_norm_n")
    recomputed_net_norm = math.sqrt(sum(float(value) ** 2 for value in net_force))
    if not math.isclose(net_force_norm, recomputed_net_norm, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("contact net force norm does not match net force vector")
    diagnostic = payload.get("contact_point_capacity_diagnostic")
    if not isinstance(diagnostic, dict) or set(diagnostic) != {
        "registered_total_capacity",
        "reported_contact_point_count",
        "remaining_capacity",
        "high_water_mark",
        "by_filter",
        "count_sum",
        "max_exclusive_end_index",
        "all_ranges_within_capacity",
    }:
        raise ValueError("contact_point_capacity_diagnostic keys are not exact")
    if diagnostic.get("registered_total_capacity") != REGISTERED_TOTAL_CAPACITY:
        raise ValueError("observation capacity does not match D361")
    count = diagnostic.get("reported_contact_point_count")
    if not isinstance(count, int) or not 0 <= count <= REGISTERED_TOTAL_CAPACITY:
        raise ValueError("reported contact-point count is outside registered capacity")
    if diagnostic.get("remaining_capacity") != REGISTERED_TOTAL_CAPACITY - count:
        raise ValueError("remaining contact-point capacity arithmetic mismatch")
    if not isinstance(diagnostic.get("remaining_capacity"), int):
        raise ValueError("remaining contact-point capacity must be an integer")
    high_water = diagnostic.get("high_water_mark")
    if not isinstance(high_water, int) or not count <= high_water <= REGISTERED_TOTAL_CAPACITY:
        raise ValueError("contact-point high-water mark is invalid")
    by_filter = diagnostic.get("by_filter")
    if not isinstance(by_filter, dict) or set(by_filter) != set(BODY_LABELS):
        raise ValueError("contact-point count/start body map is not exact")
    count_sum = 0
    max_end = 0
    nonempty_ranges: list[tuple[int, int, str]] = []
    for label in BODY_LABELS:
        row = by_filter[label]
        if not isinstance(row, dict) or set(row) != {
            "filter_index", "start_index", "count", "exclusive_end_index"
        }:
            raise ValueError(f"contact-point diagnostic {label} keys are not exact")
        start = row.get("start_index")
        body_count = row.get("count")
        end = row.get("exclusive_end_index")
        if row.get("filter_index") != BODY_LABELS.index(label):
            raise ValueError(f"contact-point diagnostic {label} filter index mismatch")
        if not isinstance(start, int) or not isinstance(body_count, int) or not isinstance(end, int):
            raise ValueError(f"contact-point diagnostic {label} indices must be integers")
        if start < 0 or body_count < 0 or end != start + body_count or end > REGISTERED_TOTAL_CAPACITY:
            raise ValueError(f"contact-point diagnostic {label} range is invalid")
        count_sum += body_count
        max_end = max(max_end, end)
        if body_count:
            nonempty_ranges.append((start, end, label))
    nonempty_ranges.sort()
    for previous_range, current_range in zip(nonempty_ranges, nonempty_ranges[1:]):
        if current_range[0] < previous_range[1]:
            raise ValueError(
                f"contact-point ranges overlap: {previous_range[2]} and {current_range[2]}"
            )
    if count_sum != count or diagnostic.get("count_sum") != count:
        raise ValueError("contact-point per-filter count sum mismatch")
    if not isinstance(diagnostic.get("count_sum"), int):
        raise ValueError("contact-point count_sum must be an integer")
    if diagnostic.get("max_exclusive_end_index") != max_end:
        raise ValueError("contact-point maximum end-index mismatch")
    if not isinstance(diagnostic.get("max_exclusive_end_index"), int):
        raise ValueError("contact-point maximum end-index must be an integer")
    if diagnostic.get("all_ranges_within_capacity") is not True:
        raise ValueError("contact-point range contract did not pass")
    if set(events) != {
        "thresholds",
        "instantaneous_contact_threshold_by_body",
        "two_step_confirmed_contact_bodies",
        "contact_events",
        "instantaneous_object_motion",
        "two_step_object_motion_event",
    }:
        raise ValueError("event projection keys are not exact")
    if events["thresholds"] != {
        "contact_force_n": CONTACT_THRESHOLD_N,
        "object_xy_mm": MOTION_XY_THRESHOLD_MM,
        "object_tilt_deg": MOTION_TILT_THRESHOLD_DEG,
        "confirmation_rows": 2,
    }:
        raise ValueError("event thresholds differ from the registered contract")
    if not isinstance(events["instantaneous_object_motion"], bool):
        raise ValueError("instantaneous object-motion flag must be boolean")
    instantaneous = events.get("instantaneous_contact_threshold_by_body")
    if not isinstance(instantaneous, dict) or set(instantaneous) != set(ROBOT_CONTACT_LABELS):
        raise ValueError("instantaneous contact body map is not exact")
    if any(not isinstance(value, bool) for value in instantaneous.values()):
        raise ValueError("instantaneous contact body map must contain booleans")
    confirmed = events.get("two_step_confirmed_contact_bodies")
    if not isinstance(confirmed, list) or any(label not in ROBOT_CONTACT_LABELS for label in confirmed):
        raise ValueError("confirmed contact body labels are invalid")
    contact_events = events.get("contact_events")
    if not isinstance(contact_events, dict) or set(contact_events) != set(confirmed):
        raise ValueError("contact event body/value map does not match confirmed labels")
    for label, event in contact_events.items():
        if not isinstance(event, dict) or set(event) != {
            "onset_phase_step",
            "confirmation_phase_step",
            "previous_force_norm_n",
            "current_force_norm_n",
        }:
            raise ValueError(f"contact event {label} keys are not exact")
        _finite_number(event.get("previous_force_norm_n"), f"{label}.previous_force_norm_n")
        _finite_number(event.get("current_force_norm_n"), f"{label}.current_force_norm_n")
        if event.get("onset_phase_step") != state["phase_step"] - 1:
            raise ValueError(f"contact event {label} onset step mismatch")
        if event.get("confirmation_phase_step") != state["phase_step"]:
            raise ValueError(f"contact event {label} confirmation step mismatch")
    motion_event = events.get("two_step_object_motion_event")
    if motion_event is not None:
        if not isinstance(motion_event, dict) or set(motion_event) != {
            "onset_phase_step",
            "confirmation_phase_step",
            "previous_xy_mm",
            "current_xy_mm",
            "previous_tilt_deg",
            "current_tilt_deg",
        }:
            raise ValueError("object motion event keys are not exact")
        for key in (
            "previous_xy_mm",
            "current_xy_mm",
            "previous_tilt_deg",
            "current_tilt_deg",
        ):
            _finite_number(motion_event.get(key), f"motion.{key}")
        if motion_event["onset_phase_step"] != state["phase_step"] - 1:
            raise ValueError("object motion event onset step mismatch")
        if motion_event["confirmation_phase_step"] != state["phase_step"]:
            raise ValueError("object motion event confirmation step mismatch")


class DurablePrefixWriter:
    """Exclusive append-only, fsync+reread JSONL hash-chain writer."""

    def __init__(self, path: Path, header_payload: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC,
            0o644,
        )
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.sequence = 0
        self.previous_hash = ZERO_HASH
        self.offset = 0
        self.inflight: dict[str, Any] | None = None
        self.observation_count = 0
        self.last_observation_identity: dict[str, Any] | None = None
        self.last_observation_payload: dict[str, Any] | None = None
        self.contact_point_high_water = 0
        self.sealed = False
        self.receipts: list[dict[str, Any]] = []
        try:
            _validate_header_payload(header_payload)
            self.header_payload = json.loads(_canonical(header_payload))
            self._append("header", header_payload)
        except Exception:
            os.close(self.fd)
            self.fd = -1
            raise
        _fsync_directory(path.parent)

    def _append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.sealed:
            raise RuntimeError("cannot append after seal")
        core = {
            "schema": PREFIX_SCHEMA,
            "sequence": self.sequence,
            "kind": kind,
            "previous_record_sha256": self.previous_hash,
            "payload": payload,
        }
        digest = _sha256_bytes(_canonical(core))
        wire = {**core, "record_sha256": digest}
        data = _canonical(wire) + b"\n"
        before = os.fstat(self.fd).st_size
        if before != self.offset:
            raise RuntimeError("prefix file size drift before append")
        _write_all(self.fd, data)
        os.fsync(self.fd)
        after = os.fstat(self.fd).st_size
        if after != self.offset + len(data):
            raise RuntimeError("prefix file size mismatch after append")
        read_fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            reread = os.pread(read_fd, len(data), self.offset)
        finally:
            os.close(read_fd)
        if reread != data:
            raise RuntimeError("prefix append exact reread mismatch")
        receipt = {
            "sequence": self.sequence,
            "kind": kind,
            "offset": self.offset,
            "byte_count": len(data),
            "record_sha256": digest,
            "reread_sha256": _sha256_bytes(reread),
            "fsync_then_exact_reread_pass": True,
        }
        self.receipts.append(receipt)
        self.offset = after
        self.sequence += 1
        self.previous_hash = digest
        return receipt

    def begin_step(self, identity: dict[str, Any], pre_step_state: dict[str, Any]) -> dict[str, Any]:
        if self.inflight is not None:
            raise RuntimeError("a step is already inflight")
        if not isinstance(pre_step_state, dict):
            raise ValueError("pre_step_state must be an object")
        identity = dict(_validate_step_identity(identity))
        if self.last_observation_identity is not None:
            previous = self.last_observation_identity
            if identity["global_step"] != previous["global_step"] + 1:
                raise ValueError("global_step is not contiguous")
            if identity["phase"] == previous["phase"]:
                if identity["phase_step"] != previous["phase_step"] + 1:
                    raise ValueError("same-phase phase_step is not contiguous")
            elif identity["phase_step"] != 0:
                raise ValueError("new phase must begin at phase_step zero")
        payload = {"step_identity": identity, "pre_step_state": pre_step_state}
        receipt = self._append("step_begin", payload)
        self.inflight = identity
        return receipt

    def observe_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.inflight is None:
            raise RuntimeError("step_observation has no matching step_begin")
        _validate_observation_payload(payload)
        if payload["subject_kind"] != self.header_payload["subject_kind"]:
            raise ValueError("observation subject kind differs from the header")
        if payload["step_identity"] != self.inflight:
            raise ValueError("step_observation identity differs from inflight step")
        expected_events = _event_projection(
            None if self.last_observation_payload is None else self.last_observation_payload["state_row"],
            payload["state_row"],
        )
        if payload["event_projection"] != expected_events:
            raise ValueError("event body/value projection differs from the current and previous state rows")
        diagnostic = payload["contact_point_capacity_diagnostic"]
        expected_high_water = max(self.contact_point_high_water, diagnostic["reported_contact_point_count"])
        if diagnostic["high_water_mark"] != expected_high_water:
            raise ValueError("contact-point high-water mark is not monotonic/exact")
        receipt = self._append("step_observation", payload)
        self.inflight = None
        self.observation_count += 1
        self.last_observation_identity = dict(payload["step_identity"])
        self.last_observation_payload = payload
        self.contact_point_high_water = expected_high_water
        return receipt

    def seal(self, reason: str) -> dict[str, Any]:
        if self.inflight is not None:
            raise RuntimeError("cannot seal with an inflight step")
        if not isinstance(reason, str):
            raise ValueError("seal reason must be a string")
        legal_seals = self.header_payload["execution_contract"]["legal_seals"]
        if reason not in legal_seals or self.observation_count not in legal_seals[reason]:
            raise ValueError(
                f"seal is not registered for reason/count: reason={reason!r}, "
                f"observation_count={self.observation_count}"
            )
        receipt = self._append(
            "seal",
            {
                "reason": reason,
                "observation_count": self.observation_count,
                "last_record_before_seal_sha256": self.previous_hash,
            },
        )
        self.sealed = True
        return receipt

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _semantic_record_check(
    record: dict[str, Any],
    *,
    first: bool,
    inflight: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    sealed: bool,
    header_payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool, dict[str, Any] | None]:
    kind = record["kind"]
    payload = record["payload"]
    if sealed:
        raise ValueError("record appears after seal")
    if first:
        if kind != "header" or not isinstance(payload, dict):
            raise ValueError("first record must be a header")
        _validate_header_payload(payload)
        return inflight, sealed, payload
    if kind == "header":
        raise ValueError("duplicate header")
    if kind == "step_begin":
        if inflight is not None:
            raise ValueError("nested step_begin")
        if not isinstance(payload, dict) or set(payload) != {"step_identity", "pre_step_state"}:
            raise ValueError("step_begin payload keys are not exact")
        if not isinstance(payload["pre_step_state"], dict):
            raise ValueError("step_begin pre_step_state must be an object")
        identity = dict(_validate_step_identity(payload.get("step_identity")))
        if observations:
            previous = observations[-1]["step_identity"]
            if identity["global_step"] != previous["global_step"] + 1:
                raise ValueError("global_step is not contiguous")
            if identity["phase"] == previous["phase"]:
                if identity["phase_step"] != previous["phase_step"] + 1:
                    raise ValueError("same-phase phase_step is not contiguous")
            elif identity["phase_step"] != 0:
                raise ValueError("new phase must begin at phase_step zero")
        return identity, sealed, header_payload
    if kind == "step_observation":
        if inflight is None:
            raise ValueError("observation without step_begin")
        _validate_observation_payload(payload)
        if header_payload is None or payload["subject_kind"] != header_payload["subject_kind"]:
            raise ValueError("observation subject kind differs from the header")
        if payload["step_identity"] != inflight:
            raise ValueError("observation does not match inflight step")
        expected_events = _event_projection(
            None if not observations else observations[-1]["state_row"],
            payload["state_row"],
        )
        if payload["event_projection"] != expected_events:
            raise ValueError("event body/value projection differs from the current and previous state rows")
        previous_high_water = 0 if not observations else observations[-1]["contact_point_capacity_diagnostic"]["high_water_mark"]
        diagnostic = payload["contact_point_capacity_diagnostic"]
        if diagnostic["high_water_mark"] != max(previous_high_water, diagnostic["reported_contact_point_count"]):
            raise ValueError("contact-point high-water mark is not monotonic/exact")
        observations.append(payload)
        return None, sealed, header_payload
    if kind == "seal":
        if inflight is not None:
            raise ValueError("seal with inflight step")
        if not isinstance(payload, dict) or set(payload) != {
            "reason", "observation_count", "last_record_before_seal_sha256"
        }:
            raise ValueError("seal payload keys are not exact")
        if payload.get("observation_count") != len(observations):
            raise ValueError("seal observation count mismatch")
        if payload.get("last_record_before_seal_sha256") != record["previous_record_sha256"]:
            raise ValueError("seal does not bind the preceding record hash")
        if header_payload is None:
            raise ValueError("seal has no validated header")
        legal_seals = header_payload["execution_contract"]["legal_seals"]
        reason = payload.get("reason")
        if reason not in legal_seals or len(observations) not in legal_seals[reason]:
            raise ValueError(
                f"seal is not registered for reason/count: reason={reason!r}, "
                f"observation_count={len(observations)}"
            )
        return None, True, header_payload
    raise ValueError(f"unknown record kind: {kind}")


def verify_prefix(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = path.read_bytes()
    cursor = 0
    expected_sequence = 0
    previous_hash = ZERO_HASH
    valid_bytes = 0
    valid_records = 0
    inflight: dict[str, Any] | None = None
    sealed = False
    error: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = []
    header_payload: dict[str, Any] | None = None
    last_kind: str | None = None
    last_hash: str | None = None
    while cursor < len(data):
        newline = data.find(b"\n", cursor)
        if newline < 0:
            break
        raw_line = data[cursor:newline]
        try:
            if not raw_line:
                raise ValueError("empty complete line")
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            required = {"schema", "sequence", "kind", "previous_record_sha256", "payload", "record_sha256"}
            if set(record) != required:
                raise ValueError("record keys are not exact")
            if record["schema"] != PREFIX_SCHEMA:
                raise ValueError("schema mismatch")
            if record["sequence"] != expected_sequence:
                raise ValueError("sequence mismatch")
            if record["previous_record_sha256"] != previous_hash:
                raise ValueError("previous hash mismatch")
            core = {key: record[key] for key in ("schema", "sequence", "kind", "previous_record_sha256", "payload")}
            computed = _sha256_bytes(_canonical(core))
            if record["record_sha256"] != computed:
                raise ValueError("self hash mismatch")
            inflight, sealed, header_payload = _semantic_record_check(
                record,
                first=(expected_sequence == 0),
                inflight=inflight,
                observations=observations,
                sealed=sealed,
                header_payload=header_payload,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            error = {
                "byte_offset": cursor,
                "complete_line": True,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            break
        valid_bytes = newline + 1
        valid_records += 1
        cursor = newline + 1
        expected_sequence += 1
        previous_hash = record["record_sha256"]
        last_hash = previous_hash
        last_kind = record["kind"]
    trailing = data[valid_bytes:]
    if error is None and trailing and trailing.endswith(b"\n"):
        error = {
            "byte_offset": valid_bytes,
            "complete_line": True,
            "error_type": "UnparsedCompleteData",
            "message": "complete trailing data was not validated",
        }
    audit = {
        "artifact": "D361_PREFIX_AUDIT_V1",
        "path": _rel(path),
        "file_sha256": _sha256_bytes(data),
        "file_bytes": len(data),
        "valid_prefix_byte_count": valid_bytes,
        "valid_record_count": valid_records,
        "last_valid_kind": last_kind,
        "last_record_sha256": last_hash,
        "observation_count": len(observations),
        "last_observed_step": observations[-1]["step_identity"] if observations else None,
        "terminal_inflight_step": inflight,
        "validated_header_payload": header_payload,
        "sealed": sealed,
        "trailing_byte_count": len(trailing),
        "trailing_bytes_sha256": _sha256_bytes(trailing) if trailing else None,
        "integrity_error": error,
        "chain_integrity_pass": error is None,
        "complete_pass": error is None and sealed and not trailing and inflight is None,
        "recoverable_prefix_pass": valid_records > 0,
    }
    return audit, observations


def _event_projection(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    current_force = {
        label: float(current["contact"]["by_filter"][label]["force_norm_n"]) for label in ROBOT_CONTACT_LABELS
    }
    instantaneous = {label: value >= CONTACT_THRESHOLD_N for label, value in current_force.items()}
    same_phase = previous is not None and previous["phase"] == current["phase"]
    confirmed: list[str] = []
    contact_events: dict[str, Any] = {}
    if same_phase and previous is not None:
        for label in ROBOT_CONTACT_LABELS:
            previous_value = float(previous["contact"]["by_filter"][label]["force_norm_n"])
            if previous_value >= CONTACT_THRESHOLD_N and instantaneous[label]:
                confirmed.append(label)
                contact_events[label] = {
                    "onset_phase_step": int(previous["phase_step"]),
                    "confirmation_phase_step": int(current["phase_step"]),
                    "previous_force_norm_n": previous_value,
                    "current_force_norm_n": current_force[label],
                }
    current_motion = (
        float(current["object_disp_xy_mm"]) >= MOTION_XY_THRESHOLD_MM
        or float(current["object_tilt_delta_from_reference_deg"]) >= MOTION_TILT_THRESHOLD_DEG
    )
    motion_event = None
    if same_phase and previous is not None:
        previous_motion = (
            float(previous["object_disp_xy_mm"]) >= MOTION_XY_THRESHOLD_MM
            or float(previous["object_tilt_delta_from_reference_deg"]) >= MOTION_TILT_THRESHOLD_DEG
        )
        if previous_motion and current_motion:
            motion_event = {
                "onset_phase_step": int(previous["phase_step"]),
                "confirmation_phase_step": int(current["phase_step"]),
                "previous_xy_mm": float(previous["object_disp_xy_mm"]),
                "current_xy_mm": float(current["object_disp_xy_mm"]),
                "previous_tilt_deg": float(previous["object_tilt_delta_from_reference_deg"]),
                "current_tilt_deg": float(current["object_tilt_delta_from_reference_deg"]),
            }
    return {
        "thresholds": {
            "contact_force_n": CONTACT_THRESHOLD_N,
            "object_xy_mm": MOTION_XY_THRESHOLD_MM,
            "object_tilt_deg": MOTION_TILT_THRESHOLD_DEG,
            "confirmation_rows": 2,
        },
        "instantaneous_contact_threshold_by_body": instantaneous,
        "two_step_confirmed_contact_bodies": confirmed,
        "contact_events": contact_events,
        "instantaneous_object_motion": current_motion,
        "two_step_object_motion_event": motion_event,
    }


def _synthetic_state_row(step: int) -> dict[str, Any]:
    gripper_force = (0.11, 0.12, 0.13)[step]
    xy_mm = (0.4, 0.6, 0.7)[step]
    body_forces = {
        "support_table": 7.06,
        "link4": 0.0,
        "link5": 0.02,
        "gripper_link": gripper_force,
    }
    by_filter = {}
    for index, label in enumerate(BODY_LABELS):
        value = body_forces[label]
        by_filter[label] = {
            "filter_index": index,
            "force_w_n": [value, 0.0, 0.0],
            "force_norm_n": value,
            "contact_point_w_m": [0.3, 0.0, 0.045] if value >= CONTACT_THRESHOLD_N else None,
        }
    return {
        "global_step": 201 + step,
        "phase": "synthetic_closure",
        "phase_step": step,
        "physics_time_s": 1.005 + 0.005 * step,
        "simulation_clock": {
            "current_time": 1.005 + 0.005 * step,
            "current_time_step_index": 201 + step,
        },
        "timeline_time_s": 1.005 + 0.005 * step,
        "actual_joint_rad": [0.0, 0.5, 1.9, 0.18, 0.0, 1.5 - 0.01 * step],
        "actual_joint_vel_rad_s": [0.0, 0.0, 0.0, 0.0, 0.0, -0.2],
        "target_joint_rad": [0.0, 0.5, 1.9, 0.18, 0.0, 0.0],
        "q5_actual_rad": 1.5 - 0.01 * step,
        "q5_velocity_rad_s": -0.2,
        "q5_target_rad": 0.0,
        "q5_error_rad": -(1.5 - 0.01 * step),
        "q0_q4_actual_minus_frozen_rad": [0.0] * 5,
        "q0_q4_max_abs_drift_rad": 0.0,
        "q5_torque": {
            "authority": "synthetic_offline_fixture",
            "applied_torque_nm": -2.5,
            "computed_torque_nm": -2.5,
            "registered_effort_limit_nm": 2.5,
            "abs_applied_over_limit": 1.0,
            "effort_saturated_diagnostic": True,
            "saturation_abs_tolerance_nm": 0.0001,
        },
        "object_pos_w_m": [0.3 + xy_mm / 1000.0, 0.0, 0.032883],
        "object_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "object_lin_vel_w_mps": [0.0, 0.0, 0.0],
        "object_ang_vel_w_radps": [0.0, 0.0, 0.0],
        "object_disp_w_m": [xy_mm / 1000.0, 0.0, 0.0],
        "object_disp_xy_mm": xy_mm,
        "object_z_delta_mm": 0.0,
        "object_tilt_deg": 0.0,
        "object_tilt_delta_from_reference_deg": 0.0,
        "object_bottom_table_gap_mm": 0.0,
        "robot_root_pos_w_m": [0.0, 0.0, 0.0],
        "robot_root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "robot_root_position_drift_m": 0.0,
        "robot_root_rotation_drift_rad": 0.0,
        "link5_pos_w_m": [0.3, 0.0, 0.05],
        "link5_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "gripper_pos_w_m": [0.3, 0.0, 0.05],
        "gripper_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "contact": {
            "by_filter": by_filter,
            "net_force_w_n": [sum(body_forces.values()), 0.0, 0.0],
            "net_force_norm_n": sum(body_forces.values()),
        },
        "finite": True,
    }


def _observation_payload(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    contact_count: int,
    previous_high_water: int = 0,
) -> dict[str, Any]:
    identity = {key: current[key] for key in ("global_step", "phase", "phase_step")}
    counts = {
        "support_table": max(contact_count - 9, 0),
        "link4": 1,
        "link5": 3,
        "gripper_link": 5,
    }
    if sum(counts.values()) != contact_count:
        raise ValueError("synthetic contact count fixture arithmetic failed")
    start = 0
    by_filter: dict[str, Any] = {}
    for label in BODY_LABELS:
        body_count = counts[label]
        by_filter[label] = {
            "filter_index": BODY_LABELS.index(label),
            "start_index": start,
            "count": body_count,
            "exclusive_end_index": start + body_count,
        }
        start += body_count
    return {
        "subject_kind": "offline_synthetic_protocol_fixture_not_physics_evidence",
        "step_identity": identity,
        "state_row": current,
        "contact_point_capacity_diagnostic": {
            "registered_total_capacity": REGISTERED_TOTAL_CAPACITY,
            "reported_contact_point_count": contact_count,
            "remaining_capacity": REGISTERED_TOTAL_CAPACITY - contact_count,
            "high_water_mark": max(previous_high_water, contact_count),
            "by_filter": by_filter,
            "count_sum": sum(counts.values()),
            "max_exclusive_end_index": max(row["exclusive_end_index"] for row in by_filter.values()),
            "all_ranges_within_capacity": all(
                row["exclusive_end_index"] <= REGISTERED_TOTAL_CAPACITY for row in by_filter.values()
            ),
        },
        "event_projection": _event_projection(previous, current),
    }


def _header_payload(scenario: str) -> dict[str, Any]:
    return {
        "case": CASE,
        "scenario": scenario,
        "profile": PREFIX_PROFILE,
        "subject_kind": "offline_synthetic_protocol_fixture_not_physics_evidence",
        "registered_total_capacity": REGISTERED_TOTAL_CAPACITY,
        "sensor_body_names": list(SENSOR_BODY_NAMES),
        "body_labels": list(BODY_LABELS),
        "resolved_filter_paths": list(RESOLVED_FILTER_PATHS),
        "filter_index_by_body": {label: index for index, label in enumerate(BODY_LABELS)},
        "state_row_contract": {
            "authority": "D360 _state_row exact top-level field contract",
            "required_top_level_keys": list(D360_STATE_ROW_KEYS),
        },
        "lineage": _prefix_lineage(),
        "execution_contract": {
            "legal_seals": {"offline_reference_horizon_complete": [3]},
        },
        "resume_allowed": False,
        "overwrite_allowed": False,
    }


def _write_reference_prefix(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    writer = DurablePrefixWriter(path, _header_payload("normal_sealed"))
    rows = [_synthetic_state_row(step) for step in range(3)]
    payloads: list[dict[str, Any]] = []
    previous = None
    high_water = 0
    try:
        for step, row in enumerate(rows):
            identity = {key: row[key] for key in ("global_step", "phase", "phase_step")}
            writer.begin_step(identity, {"synthetic_pre_step_counter": 200 + step})
            payload = _observation_payload(previous, row, contact_count=17 + step, previous_high_water=high_water)
            writer.observe_step(payload)
            payloads.append(payload)
            previous = row
            high_water = payload["contact_point_capacity_diagnostic"]["high_water_mark"]
        writer.seal("offline_reference_horizon_complete")
        receipts = list(writer.receipts)
    finally:
        writer.close()
    return payloads, receipts


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: key sets differ"
        for key in sorted(left):
            diff = _first_difference(left[key], right[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            diff = _first_difference(left_item, right_item, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def reconcile_prefix(path: Path, expected_projection: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile the verifier's durable observations with an external final projection."""
    audit, observed_projection = verify_prefix(path)
    difference = _first_difference(expected_projection, observed_projection)
    return {
        "audit": audit,
        "expected_observation_count": len(expected_projection),
        "observed_observation_count": len(observed_projection),
        "first_difference": difference,
        "pass": audit["complete_pass"] and difference is None,
    }


def _decode_complete_records(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line.decode("utf-8")) for line in _split_complete_lines(data)]


def _rehash_records(records: list[dict[str, Any]]) -> bytes:
    previous_hash = ZERO_HASH
    output: list[bytes] = []
    for sequence, source in enumerate(records):
        payload = json.loads(json.dumps(source["payload"], allow_nan=False))
        if source["kind"] == "seal" and isinstance(payload, dict):
            payload["last_record_before_seal_sha256"] = previous_hash
        core = {
            "schema": PREFIX_SCHEMA,
            "sequence": sequence,
            "kind": source["kind"],
            "previous_record_sha256": previous_hash,
            "payload": payload,
        }
        digest = _sha256_bytes(_canonical(core))
        output.append(_canonical({**core, "record_sha256": digest}) + b"\n")
        previous_hash = digest
    return b"".join(output)


def _wire_hash_chain_valid(data: bytes) -> bool:
    try:
        records = _decode_complete_records(data)
        previous_hash = ZERO_HASH
        for sequence, record in enumerate(records):
            if set(record) != {
                "schema", "sequence", "kind", "previous_record_sha256", "payload", "record_sha256"
            }:
                return False
            if record["schema"] != PREFIX_SCHEMA or record["sequence"] != sequence:
                return False
            if record["previous_record_sha256"] != previous_hash:
                return False
            core = {
                key: record[key]
                for key in ("schema", "sequence", "kind", "previous_record_sha256", "payload")
            }
            if record["record_sha256"] != _sha256_bytes(_canonical(core)):
                return False
            previous_hash = record["record_sha256"]
        return bool(records)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        return False


def _independent_reference_oracle(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Hard-coded oracle, deliberately independent of _event_projection()."""
    expected_events = [
        {
            "instantaneous": {"link4": False, "link5": False, "gripper_link": True},
            "confirmed": [],
            "contact_events": {},
            "instantaneous_motion": False,
            "motion_event": None,
        },
        {
            "instantaneous": {"link4": False, "link5": False, "gripper_link": True},
            "confirmed": ["gripper_link"],
            "contact_events": {
                "gripper_link": {
                    "onset_phase_step": 0,
                    "confirmation_phase_step": 1,
                    "previous_force_norm_n": 0.11,
                    "current_force_norm_n": 0.12,
                }
            },
            "instantaneous_motion": True,
            "motion_event": None,
        },
        {
            "instantaneous": {"link4": False, "link5": False, "gripper_link": True},
            "confirmed": ["gripper_link"],
            "contact_events": {
                "gripper_link": {
                    "onset_phase_step": 1,
                    "confirmation_phase_step": 2,
                    "previous_force_norm_n": 0.12,
                    "current_force_norm_n": 0.13,
                }
            },
            "instantaneous_motion": True,
            "motion_event": {
                "onset_phase_step": 1,
                "confirmation_phase_step": 2,
                "previous_xy_mm": 0.6,
                "current_xy_mm": 0.7,
                "previous_tilt_deg": 0.0,
                "current_tilt_deg": 0.0,
            },
        },
    ]
    checks: dict[str, bool] = {"three_payloads": len(payloads) == 3}
    if len(payloads) != 3:
        return {"checks": checks, "pass": False}
    for index, (payload, expected) in enumerate(zip(payloads, expected_events, strict=True)):
        events = payload["event_projection"]
        diagnostic = payload["contact_point_capacity_diagnostic"]
        checks[f"step_{index}_instantaneous_body_map"] = (
            events["instantaneous_contact_threshold_by_body"] == expected["instantaneous"]
        )
        checks[f"step_{index}_confirmed_body_order"] = (
            events["two_step_confirmed_contact_bodies"] == expected["confirmed"]
        )
        checks[f"step_{index}_contact_body_values"] = events["contact_events"] == expected["contact_events"]
        checks[f"step_{index}_instantaneous_motion"] = (
            events["instantaneous_object_motion"] is expected["instantaneous_motion"]
        )
        checks[f"step_{index}_motion_values"] = (
            events["two_step_object_motion_event"] == expected["motion_event"]
        )
        checks[f"step_{index}_capacity_values"] = (
            diagnostic["reported_contact_point_count"] == 17 + index
            and diagnostic["high_water_mark"] == 17 + index
            and diagnostic["remaining_capacity"] == REGISTERED_TOTAL_CAPACITY - (17 + index)
        )
    return {"checks": checks, "pass": all(checks.values())}


def _split_complete_lines(data: bytes) -> list[bytes]:
    if not data.endswith(b"\n"):
        raise ValueError("fixture is not newline complete")
    return [line + b"\n" for line in data[:-1].split(b"\n")]


def _run_child(path: Path, scenario: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(HARNESS), "--stage", "child", "--scenario", scenario, "--path", str(path)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _child_scenario(scenario: str, path: Path) -> None:
    if scenario not in {"exit_after_begin", "exit_after_observation", "partial_tail"}:
        raise ValueError(f"unknown child scenario {scenario}")
    writer = DurablePrefixWriter(path, _header_payload(scenario))
    row0 = _synthetic_state_row(0)
    identity0 = {key: row0[key] for key in ("global_step", "phase", "phase_step")}
    writer.begin_step(identity0, {"synthetic_pre_step_counter": 200})
    if scenario == "exit_after_begin":
        os._exit(73)
    writer.observe_step(_observation_payload(None, row0, contact_count=17))
    if scenario == "exit_after_observation":
        os._exit(74)
    writer.close()
    partial_core = {
        "schema": PREFIX_SCHEMA,
        "sequence": 3,
        "kind": "step_begin",
        "previous_record_sha256": "f" * 64,
        "payload": {"intentionally_partial_tail": True},
    }
    partial = _canonical({**partial_core, "record_sha256": _sha256_bytes(_canonical(partial_core))})
    half = partial[: len(partial) // 2]
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
    try:
        _write_all(fd, half)
        os.fsync(fd)
    finally:
        os.close(fd)
    os._exit(75)


def _offline_link4_inventory() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PXR_PYTHONPATH
    env["LD_LIBRARY_PATH"] = PXR_LD_LIBRARY_PATH
    proc = subprocess.run(
        [
            str(ISAAC_PYTHON),
            "-B",
            str(HARNESS),
            "--stage",
            "pxr_inventory",
            "--path",
            str(VARIANT_PHYSICS_USD),
        ],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"offline PXR inventory failed: {proc.stderr}")
    payload = json.loads(proc.stdout)
    payload["subprocess_returncode"] = proc.returncode
    payload["stderr"] = proc.stderr
    return payload


def _pxr_inventory(path: Path) -> None:
    from pxr import Usd, UsdPhysics  # core OpenUSD schema only; no Kit/PhysX runtime

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {path}")
    rows = []
    for prim in stage.TraverseAll():
        prim_path = prim.GetPath().pathString
        if "link4" not in prim_path or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        rows.append(
            {
                "path": prim_path,
                "type_name": prim.GetTypeName(),
                "collision_enabled": True if enabled is None else bool(enabled),
            }
        )
    output = {
        "artifact": "D361_CORE_PXR_LINK4_COLLISION_INVENTORY_V1",
        "usd_path": str(path.resolve()),
        "default_prim": stage.GetDefaultPrim().GetPath().pathString,
        "rows": rows,
        "enabled_count": sum(1 for row in rows if row["collision_enabled"]),
        "kit_or_physx_runtime_launched": False,
    }
    sys.stdout.write(json.dumps(output, sort_keys=True, allow_nan=False))


def _capacity_budget() -> dict[str, Any]:
    prerequisites = json.loads(D360_PREREQUISITES.read_text(encoding="utf-8"))
    link4_inventory = _offline_link4_inventory()
    table_rows = prerequisites["stage_contract"]["table_collision_prims"]
    actual_counts = {
        "sensor_cylinder": 1,
        "support_table": sum(1 for row in table_rows if row["collision_enabled"]),
        "link4": int(link4_inventory["enabled_count"]),
        "link5": int(prerequisites["live_part_counts"]["link5"]),
        "gripper_link": int(prerequisites["live_part_counts"]["gripper_link"]),
    }
    plugin = PHYSX_PLUGIN.read_bytes()
    version_offsets = []
    start = 0
    needle = b"5.6.1\x00"
    while True:
        offset = plugin.find(needle, start)
        if offset < 0:
            break
        version_offsets.append(offset)
        start = offset + 1
    shape_source = ISAACLAB_SHAPE_SPAWNER.read_text(encoding="utf-8")
    cylinder_static_checks = {
        "spawn_cylinder_definition_present": "def spawn_cylinder(" in shape_source,
        "single_cylinder_prim_type_call_present": shape_source.count(
            '_spawn_geom_from_prim_type(prim_path, cfg, "Cylinder", attributes, translation, orientation)'
        ) == 1,
        "one_mesh_prim_path_convention_present": 'mesh_prim_path = geom_prim_path + "/mesh"' in shape_source,
        "collision_applied_to_mesh_prim_present": "schemas.define_collision_properties(mesh_prim_path, cfg.collision_props)" in shape_source,
    }
    filter_total = sum(actual_counts[label] for label in BODY_LABELS)
    direct_capacity = actual_counts["sensor_cylinder"] * filter_total * PHYSX_CONTACTS_PER_GEOMETRY_PAIR
    per_filter_capacity = {
        label: actual_counts["sensor_cylinder"] * actual_counts[label] * PHYSX_CONTACTS_PER_GEOMETRY_PAIR
        for label in BODY_LABELS
    }
    independent_capacity = sum(per_filter_capacity.values())
    direct_bytes = direct_capacity * DETAIL_FLOATS_PER_CONTACT * FLOAT32_BYTES
    independent_bytes = direct_capacity * ((1 + 3 + 3 + 1) * FLOAT32_BYTES)
    negative_controls = {
        "legacy_16_rejected": 16 != REGISTERED_TOTAL_CAPACITY,
        "wrong_filter_count_multiplication_16x4_rejected": 16 * 4 != REGISTERED_TOTAL_CAPACITY,
        "stale_physx_5_3_limit_64_rejected": filter_total * 64 != REGISTERED_TOTAL_CAPACITY,
        "one_below_exact_boundary_rejected": REGISTERED_TOTAL_CAPACITY - 1 < direct_capacity,
        "exact_boundary_accepted": REGISTERED_TOTAL_CAPACITY >= direct_capacity,
        "link5_minus_one_shape_rejected":
            actual_counts["sensor_cylinder"] * (filter_total - 1) * PHYSX_CONTACTS_PER_GEOMETRY_PAIR
            != REGISTERED_TOTAL_CAPACITY,
        "one_hidden_extra_filter_shape_rejected":
            actual_counts["sensor_cylinder"] * (filter_total + 1) * PHYSX_CONTACTS_PER_GEOMETRY_PAIR
            != REGISTERED_TOTAL_CAPACITY,
        "two_sensor_shapes_rejected":
            2 * filter_total * PHYSX_CONTACTS_PER_GEOMETRY_PAIR != REGISTERED_TOTAL_CAPACITY,
        "filter_count_double_multiplication_rejected":
            direct_capacity * prerequisites["sensor_contract"]["filter_count"] != REGISTERED_TOTAL_CAPACITY,
    }
    checks = {
        "d360_prerequisites_pass": prerequisites.get("pass") is True,
        "d360_sensor_body_count_one": prerequisites["sensor_contract"]["num_bodies"] == 1,
        "d360_environment_count_one": prerequisites["sensor_contract"]["num_instances"] == 1,
        "d360_filter_count_four": prerequisites["sensor_contract"]["filter_count"] == 4,
        "actual_shape_inventory_exact": actual_counts == {
            "sensor_cylinder": 1,
            "support_table": 1,
            "link4": 1,
            "link5": 64,
            "gripper_link": 64,
        },
        "cylinder_static_source_contract_pass": all(cylinder_static_checks.values()),
        "physx_plugin_exact_5_6_1_string_once": version_offsets == [9_753_628],
        "filter_shape_total_130": filter_total == EXPECTED_FILTER_SHAPE_TOTAL,
        "direct_capacity_33280": direct_capacity == REGISTERED_TOTAL_CAPACITY,
        "independent_capacity_matches": independent_capacity == direct_capacity,
        "detail_bytes_1064960": direct_bytes == REGISTERED_DETAIL_BYTES,
        "independent_bytes_match": independent_bytes == direct_bytes,
        "all_negative_controls_reject": all(negative_controls.values()),
    }
    return {
        "artifact": "D361_VERSION_ALIGNED_CONTACT_CAPACITY_BUDGET_V1",
        "case": CASE,
        "scientific_or_physics_execution": False,
        "authority_scope": "future detailed-contact allocation contract only",
        "installed_runtime_identity": {
            "isaacsim_python_distribution": importlib.metadata.version("isaacsim"),
            "omni_physx_extension": "107.3.26+107.3.3",
            "physx_plugin_path": _rel(PHYSX_PLUGIN),
            "physx_plugin_sha256": _sha256_file(PHYSX_PLUGIN),
            "physx_5_6_1_null_terminated_string_offsets": version_offsets,
        },
        "physx_pair_limit_authority": {
            "value": PHYSX_CONTACTS_PER_GEOMETRY_PAIR,
            "name": "PxContactBuffer::MAX_CONTACTS",
            "version": "NVIDIA-Omniverse PhysX tag 107.3-omni-and-physx-5.6.1",
            "url": (
                "https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/"
                "107.3-omni-and-physx-5.6.1/physx/include/geomutils/PxContactBuffer.h"
            ),
            "source_declaration": "static const PxU32 MAX_CONTACTS = 256",
        },
        "actual_shape_counts": actual_counts,
        "link4_offline_core_pxr_inventory": link4_inventory,
        "cylinder_static_source_checks": cylinder_static_checks,
        "formula": {
            "sensor_shape_count": actual_counts["sensor_cylinder"],
            "filter_shape_count": filter_total,
            "per_geometry_pair_contact_envelope": PHYSX_CONTACTS_PER_GEOMETRY_PAIR,
            "direct_total_capacity": direct_capacity,
            "per_filter_capacities": per_filter_capacity,
            "independent_sum_total_capacity": independent_capacity,
            "number_of_environments": 1,
            "number_of_sensor_bodies": 1,
            "future_cfg_max_contact_data_count_per_prim": REGISTERED_TOTAL_CAPACITY,
            "derived_reported_contact_allocation_envelope": REGISTERED_TOTAL_CAPACITY,
        },
        "frontend_memory": {
            "arrays": {
                "normal_force_float32_components_per_contact": 1,
                "point_float32_components_per_contact": 3,
                "normal_float32_components_per_contact": 3,
                "separation_float32_components_per_contact": 1,
            },
            "bytes_per_contact": DETAIL_FLOATS_PER_CONTACT * FLOAT32_BYTES,
            "detail_array_bytes": direct_bytes,
            "detail_array_mib": direct_bytes / (1024 * 1024),
            "count_and_start_uint32_bytes": 1 * 4 * 2 * 4,
            "backend_internal_allocation_included": False,
        },
        "negative_controls": negative_controls,
        "checks": checks,
        "pass": all(checks.values()),
        "runtime_sufficiency": None,
        "runtime_sufficiency_reason": "D361 does not launch Isaac/PhysX; a separately approved future run must validate warning absence and observed counts",
    }


def _protocol_contract() -> dict[str, Any]:
    return {
        "artifact": "D361_DURABLE_PREFIX_PROTOCOL_CONTRACT_V1",
        "schema": PREFIX_SCHEMA,
        "file_creation_flags": ["O_WRONLY", "O_CREAT", "O_EXCL", "O_APPEND", "O_CLOEXEC"],
        "record_kinds": ["header", "step_begin", "step_observation", "seal"],
        "header_contract": {
            "profile": PREFIX_PROFILE,
            "sensor_body_names": list(SENSOR_BODY_NAMES),
            "body_labels_in_filter_order": list(BODY_LABELS),
            "resolved_filter_paths_in_order": list(RESOLVED_FILTER_PATHS),
            "filter_index_by_body": {label: index for index, label in enumerate(BODY_LABELS)},
            "full_d360_state_row_keys": list(D360_STATE_ROW_KEYS),
            "lineage_keys": sorted(
                (
                    "preregistration_sha256",
                    "prepare_preflight_sha256",
                    "offline_invocation_sha256",
                    "harness_sha256",
                    "capacity_budget_sha256",
                    "protocol_contract_sha256",
                )
            ),
            "synthetic_evaluation_legal_seals": {"offline_reference_horizon_complete": [3]},
            "future_actual_execution_contract_must_be_registered_separately": True,
        },
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "durability_order": [
            "single-writer exclusive create and nonblocking flock",
            "short-write loop",
            "file fsync for every record",
            "fresh read-only descriptor pread of exact appended byte range",
            "parent directory fsync after file creation",
        ],
        "step_order": [
            "step_begin fsync before attempted physics step",
            "physics step",
            "full inherited D360 state_row and contact-count diagnostic",
            "same-phase instantaneous/two-step event body/value projection",
            "step_observation fsync and exact reread",
            "only then memory row, screenshot, RRD, or summary",
        ],
        "observation_atomic_subjects": [
            "full D360 state_row",
            "support_table/link4/link5/gripper_link force vector, norm, contact point",
            "instantaneous threshold body map",
            "two-step confirmed body labels and previous/current force values",
            "two-step object-motion previous/current values",
            "reported contact-point count and registered total capacity",
        ],
        "recovery": {
            "trusted_prefix": "first consecutive newline+JSON+sequence+previous-hash+self-hash+schema-valid records only",
            "partial_tail": "preserve original bytes; trust only preceding complete records",
            "complete_corrupt_line": "fail at first corrupt line; never skip forward",
            "unmatched_step_begin": "report terminal inflight step separately from completed observations",
            "seal": "write before optional full JSON/CSV/RRD; absence means incomplete",
            "seal_acceptance": "reason and observation count must match the exact header legal_seals map",
            "resume_or_overwrite": False,
        },
        "scientific_authority": False,
        "future_integration_requires_separate_physics_approval": True,
        "pass": True,
    }


def _failure_injection_evaluation(
    marker: DurableMarkerStream,
    result_journal: DurableMarkerStream,
) -> dict[str, Any]:
    FIXTURE_DIR.mkdir(parents=False, exist_ok=False)
    _fsync_directory(FIXTURE_DIR.parent)
    results: dict[str, Any] = {}

    def record_result(name: str, row: dict[str, Any]) -> None:
        results[name] = row
        result_journal.append(name, result=row)
        marker.append(f"{name}_verified", pass_value=bool(row["pass"]))

    reference_path = FIXTURE_DIR / "reference_sealed_prefix.jsonl"
    expected_payloads, receipts = _write_reference_prefix(reference_path)
    reconciliation = reconcile_prefix(reference_path, expected_payloads)
    oracle = _independent_reference_oracle(expected_payloads)
    record_result(
        "normal_sealed",
        {
            "reconciliation": reconciliation,
            "independent_hard_coded_oracle": oracle,
            "receipt_count": len(receipts),
            "receipt_all_exact_reread": all(row["fsync_then_exact_reread_pass"] for row in receipts),
            "pass": (
                reconciliation["pass"]
                and oracle["pass"]
                and len(receipts) == 8
                and all(row["fsync_then_exact_reread_pass"] for row in receipts)
            ),
        },
    )

    before_hash = _sha256_file(reference_path)
    exclusive_error = None
    try:
        duplicate = DurablePrefixWriter(reference_path, _header_payload("exclusive_negative"))
        duplicate.close()
    except FileExistsError as exc:
        exclusive_error = type(exc).__name__
    after_hash = _sha256_file(reference_path)
    record_result(
        "exclusive_create_negative",
        {
            "error": exclusive_error,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "pass": exclusive_error == "FileExistsError" and before_hash == after_hash,
        },
    )

    child_expectations = {
        "exit_after_begin": {"returncode": 73, "observations": 0, "inflight": True, "trailing": 0},
        "exit_after_observation": {"returncode": 74, "observations": 1, "inflight": False, "trailing": 0},
        "partial_tail": {"returncode": 75, "observations": 1, "inflight": False, "trailing_min": 1},
    }
    for scenario, expected in child_expectations.items():
        path = FIXTURE_DIR / f"{scenario}.jsonl"
        proc = _run_child(path, scenario)
        audit, observations = verify_prefix(path)
        pass_value = (
            proc.returncode == expected["returncode"]
            and audit["observation_count"] == expected["observations"]
            and (audit["terminal_inflight_step"] is not None) == expected["inflight"]
            and audit["sealed"] is False
            and audit["chain_integrity_pass"] is True
        )
        if "trailing" in expected:
            pass_value = pass_value and audit["trailing_byte_count"] == expected["trailing"]
        if "trailing_min" in expected:
            pass_value = pass_value and audit["trailing_byte_count"] >= expected["trailing_min"]
        record_result(
            scenario,
            {
                "child_returncode": proc.returncode,
                "termination_mechanism": "os._exit abrupt process termination; not claimed as SIGKILL",
                "child_stdout": proc.stdout,
                "child_stderr": proc.stderr,
                "audit": audit,
                "recovered_observation_payloads": observations,
                "pass": pass_value,
            },
        )

    reference_data = reference_path.read_bytes()
    lines = _split_complete_lines(reference_data)
    tamper_specs: dict[str, bytes] = {}
    if b'"force_norm_n":0.12' not in reference_data:
        raise RuntimeError("reference fixture lacks registered force token")
    tamper_specs["body_force_byte_flip"] = reference_data.replace(
        b'"force_norm_n":0.12', b'"force_norm_n":9.12', 1
    )
    reordered = list(lines)
    reordered[3], reordered[4] = reordered[4], reordered[3]
    tamper_specs["record_reorder"] = b"".join(reordered)
    tamper_specs["middle_record_delete"] = b"".join(lines[:3] + lines[4:])
    tamper_specs["duplicate_sequence"] = b"".join(lines[:4] + [lines[3]] + lines[4:])
    for name, data in tamper_specs.items():
        path = FIXTURE_DIR / f"{name}.jsonl"
        _write_bytes_exclusive(path, data)
        audit, _ = verify_prefix(path)
        pass_value = audit["chain_integrity_pass"] is False and audit["complete_pass"] is False
        record_result(name, {"audit": audit, "pass": pass_value})

    reference_records = _decode_complete_records(reference_data)
    semantic_specs: dict[str, tuple[list[dict[str, Any]], str]] = {}

    event_records = json.loads(json.dumps(reference_records))
    event_records[4]["payload"]["event_projection"]["contact_events"]["gripper_link"][
        "current_force_norm_n"
    ] = 9.12
    semantic_specs["event_semantic_tamper_rehashed"] = (event_records, "event body/value projection")

    header_records = json.loads(json.dumps(reference_records))
    header_records[0]["payload"]["body_labels"] = list(reversed(BODY_LABELS))
    semantic_specs["header_semantic_tamper_rehashed"] = (header_records, "header body label order")

    premature_records = json.loads(json.dumps(reference_records[:3]))
    premature_records.append(
        {
            "kind": "seal",
            "payload": {
                "reason": "offline_reference_horizon_complete",
                "observation_count": 1,
                "last_record_before_seal_sha256": ZERO_HASH,
            },
        }
    )
    semantic_specs["premature_seal_rehashed"] = (premature_records, "seal is not registered")

    state_records = json.loads(json.dumps(reference_records))
    del state_records[4]["payload"]["state_row"]["q5_actual_rad"]
    semantic_specs["state_row_semantic_tamper_rehashed"] = (state_records, "state_row top-level")

    for name, (records, expected_message) in semantic_specs.items():
        data = _rehash_records(records)
        path = FIXTURE_DIR / f"{name}.jsonl"
        _write_bytes_exclusive(path, data)
        wire_hash_chain_valid = _wire_hash_chain_valid(data)
        audit, _ = verify_prefix(path)
        message = (audit.get("integrity_error") or {}).get("message", "")
        pass_value = (
            wire_hash_chain_valid
            and audit["complete_pass"] is False
            and audit["chain_integrity_pass"] is False
            and expected_message in message
        )
        record_result(
            name,
            {
                "wire_hash_chain_valid_before_semantic_validation": wire_hash_chain_valid,
                "expected_semantic_error_fragment": expected_message,
                "audit": audit,
                "pass": pass_value,
            },
        )

    positive_reconciliation = reconcile_prefix(reference_path, expected_payloads)
    record_result(
        "prefix_projection_reconciliation_positive",
        {"reconciliation": positive_reconciliation, "pass": positive_reconciliation["pass"]},
    )
    mismatched_projection = json.loads(json.dumps(expected_payloads))
    mismatched_projection[1]["state_row"]["contact"]["by_filter"]["gripper_link"]["force_norm_n"] = 8.12
    negative_reconciliation = reconcile_prefix(reference_path, mismatched_projection)
    mismatch = negative_reconciliation["first_difference"]
    record_result(
        "prefix_projection_reconciliation_negative",
        {
            "reconciliation": negative_reconciliation,
            "pass": (
                negative_reconciliation["audit"]["complete_pass"]
                and negative_reconciliation["pass"] is False
                and mismatch is not None
                and "gripper_link.force_norm_n" in mismatch
            ),
        },
    )

    invalid_path = FIXTURE_DIR / "schema_reject_missing_body.jsonl"
    writer = DurablePrefixWriter(invalid_path, _header_payload("schema_reject_missing_body"))
    row = _synthetic_state_row(0)
    identity = {key: row[key] for key in ("global_step", "phase", "phase_step")}
    writer.begin_step(identity, {"synthetic_pre_step_counter": 200})
    invalid_payload = _observation_payload(None, row, 17)
    del invalid_payload["state_row"]["contact"]["by_filter"]["link4"]
    schema_error = None
    try:
        writer.observe_step(invalid_payload)
    except ValueError as exc:
        schema_error = str(exc)
    finally:
        writer.close()
    invalid_audit, _ = verify_prefix(invalid_path)
    record_result(
        "schema_reject_missing_body",
        {
            "error": schema_error,
            "audit": invalid_audit,
            "pass": (
                schema_error is not None
                and "body set is not exact" in schema_error
                and invalid_audit["terminal_inflight_step"] is not None
            ),
        },
    )

    nan_path = FIXTURE_DIR / "nan_reject.jsonl"
    nan_writer_error = None
    try:
        nan_writer = DurablePrefixWriter(
            nan_path,
            {**_header_payload("nan_reject"), "bad_value": float("nan")},
        )
        nan_writer.close()
    except ValueError as exc:
        nan_writer_error = str(exc)
    nan_serialization_error = None
    try:
        _canonical({"bad_value": float("nan")})
    except ValueError as exc:
        nan_serialization_error = str(exc)
    record_result(
        "nan_reject",
        {
            "writer_schema_error": nan_writer_error,
            "canonical_json_nan_error": nan_serialization_error,
            "file_exists": nan_path.exists(),
            "file_bytes": nan_path.stat().st_size if nan_path.exists() else None,
            "pass": (
                nan_writer_error is not None
                and nan_serialization_error is not None
                and nan_path.exists()
                and nan_path.stat().st_size == 0
            ),
        },
    )

    test_passes = {name: bool(row["pass"]) for name, row in results.items()}
    return {
        "artifact": "D361_OFFLINE_PREFIX_FAILURE_INJECTION_RESULTS_V1",
        "subject_kind": "offline_synthetic_protocol_fixtures_not_physics_evidence",
        "failure_capable_perturbation_evaluation": True,
        "durable_per_test_result_journal": _rel(FAILURE_JOURNAL_PATH),
        "tests": results,
        "test_passes": test_passes,
        "passed": sum(test_passes.values()),
        "total": len(test_passes),
        "pass": all(test_passes.values()),
        "isaac_launch_count": 0,
        "physx_step_count": 0,
        "q5_sample_count": 0,
        "new_contact_media_count": 0,
    }


def _prepare() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output already exists: {OUT_DIR}")
    interpreter_exact = Path(sys.executable).resolve() == ISAAC_PYTHON.resolve()
    if not interpreter_exact:
        raise RuntimeError(
            "D361 must use the pinned isaaclab Python even though it launches no Isaac application: "
            f"expected={ISAAC_PYTHON.resolve()} actual={Path(sys.executable).resolve()}"
        )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    status = _git("status", "--short")
    scope_ok, unexpected = _scope_status_ok(status)
    actual_hashes = {_rel(path): _sha256_file(path) for path in EXPECTED_HASHES}
    input_hashes_ok = all(actual_hashes[_rel(path)] == expected for path, expected in EXPECTED_HASHES.items())
    actual_sidecar = {_rel(path): _sha256_file(path) for path in D334_SIDECAR}
    sidecar_ok = all(actual_sidecar[_rel(path)] == expected for path, expected in D334_SIDECAR.items())
    loaded = _forbidden_loaded_modules()
    checks = {
        "head_exact": head == BASE_GIT,
        "origin_master_exact": origin == BASE_GIT,
        "head_origin_equal": head == origin,
        "worktree_scope_only_d361": scope_ok,
        "all_frozen_input_hashes_exact": input_hashes_ok,
        "d334_user_sidecar_exact": sidecar_ok,
        "forbidden_runtime_modules_absent": not loaded,
        "pinned_isaaclab_python_exact": interpreter_exact,
        "new_variables_exact_two": NEW_VARIABLES == [
            "version_aligned_total_contact_point_capacity_budget",
            "durable_framed_step_prefix_protocol",
        ],
        "registered_capacity_arithmetic":
            EXPECTED_SENSOR_SHAPES * sum(EXPECTED_FILTER_SHAPES.values()) * PHYSX_CONTACTS_PER_GEOMETRY_PAIR
            == REGISTERED_TOTAL_CAPACITY,
        "session_prereg_exists": SESSION_DOC.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D361 prepare preflight failed: checks={checks}, unexpected={unexpected}")
    capacity_preflight = _capacity_budget()
    checks["capacity_budget_preflight_pass"] = capacity_preflight["pass"]
    if not checks["capacity_budget_preflight_pass"]:
        raise RuntimeError(f"D361 capacity preflight failed: {capacity_preflight['checks']}")
    OUT_DIR.mkdir(parents=False, exist_ok=False)
    _fsync_directory(OUT_DIR.parent)
    prereg = {
        "artifact": "D361_PREREGISTRATION_V1",
        "case": CASE,
        "case_name": CASE_NAME,
        "utc": _utc(),
        "base_git": BASE_GIT,
        "new_variables": NEW_VARIABLES,
        "registered_total_contact_capacity": REGISTERED_TOTAL_CAPACITY,
        "capacity_formula": {
            "sensor_shapes": EXPECTED_SENSOR_SHAPES,
            "filter_shapes": EXPECTED_FILTER_SHAPES,
            "filter_shape_total": EXPECTED_FILTER_SHAPE_TOTAL,
            "physx_contacts_per_geometry_pair": PHYSX_CONTACTS_PER_GEOMETRY_PAIR,
            "product": REGISTERED_TOTAL_CAPACITY,
        },
        "prefix_schema": PREFIX_SCHEMA,
        "harness_sha256": _sha256_file(HARNESS),
        "session_prereg_sha256": _sha256_file(SESSION_DOC),
        "input_hashes": actual_hashes,
        "d334_sidecar_before": actual_sidecar,
        "capacity_budget_preflight": capacity_preflight,
        "capacity_budget_preflight_canonical_sha256": _sha256_bytes(_canonical(capacity_preflight)),
        "expected_artifacts_before_completion": sorted(
            _rel(path) for path in _expected_artifact_paths(include_completion=False)
        ),
        "expected_artifacts_after_completion": sorted(
            _rel(path) for path in _expected_artifact_paths(include_completion=True)
        ),
        "optional_failure_only_artifact": _rel(RUNTIME_EXCEPTION_PATH),
        "prohibitions": [
            "no Isaac Sim/Kit/IsaacLab application launch",
            "no q5 command or sample",
            "no PhysX simulation, contact science query, or physics step",
            "no new contact image/video/RRD/RBL",
            "no target/IK/path or initial state change",
            "no asset/decomposition/gate/material/mass/actuator/solver/physics/renderer/dependency change",
            "no physical/contact/grasp/G0a verdict",
            "no D360 mutation/retry/finalize",
            "no D334 sidecar mutation",
            "no commit or push",
        ],
        "one_offline_failure_injection_invocation": True,
        "physics_runtime_sufficiency": None,
        "g0a_pass": False,
        "d360_tree_before": _tree_manifest(D360_DIR),
        "pass": True,
    }
    prereg_sha = _write_json_exclusive(PREREG_PATH, prereg)
    prepare = {
        "artifact": "D361_PREPARE_PREFLIGHT_V1",
        "utc": _utc(),
        "head": head,
        "origin_master": origin,
        "git_status_before_output_creation": status.splitlines(),
        "unexpected_scope": unexpected,
        "frozen_input_hashes": actual_hashes,
        "d334_sidecar": actual_sidecar,
        "forbidden_loaded_modules": loaded,
        "checks": checks,
        "preregistration_sha256": prereg_sha,
        "pass": all(checks.values()),
    }
    _write_json_exclusive(PREPARE_PATH, prepare)
    print(json.dumps({"stage": "prepare", "pass": True, "out_dir": _rel(OUT_DIR)}, sort_keys=True))


def _run() -> None:
    if not PREREG_PATH.is_file() or not PREPARE_PATH.is_file():
        raise RuntimeError("D361 prepare artifacts are missing")
    if Path(sys.executable).resolve() != ISAAC_PYTHON.resolve():
        raise RuntimeError(
            f"D361 run interpreter mismatch: expected={ISAAC_PYTHON.resolve()} "
            f"actual={Path(sys.executable).resolve()}"
        )
    pre_invocation_files = {path for path in OUT_DIR.rglob("*") if path.is_file()}
    if pre_invocation_files != {PREREG_PATH, PREPARE_PATH}:
        raise RuntimeError(
            "D361 pre-invocation output inventory is not exactly prepare+preregistration: "
            f"actual={sorted(_rel(path) for path in pre_invocation_files)}"
        )
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    prepare = json.loads(PREPARE_PATH.read_text(encoding="utf-8"))
    if prereg.get("pass") is not True or prepare.get("pass") is not True:
        raise RuntimeError("D361 prepare did not pass")
    if prepare.get("preregistration_sha256") != _sha256_file(PREREG_PATH):
        raise RuntimeError("D361 preregistration changed after prepare")
    if prereg.get("harness_sha256") != _sha256_file(HARNESS):
        raise RuntimeError("D361 harness changed after prepare")
    if prereg.get("session_prereg_sha256") != _sha256_file(SESSION_DOC):
        raise RuntimeError("D361 session preregistration changed after prepare")
    if _git("rev-parse", "HEAD") != BASE_GIT or _git("rev-parse", "origin/master") != BASE_GIT:
        raise RuntimeError("Git base changed after prepare")
    scope_ok, unexpected = _scope_status_ok(_git("status", "--short"))
    if not scope_ok:
        raise RuntimeError(f"unexpected worktree scope before D361 run: {unexpected}")
    current_frozen_hashes = {_rel(path): _sha256_file(path) for path in EXPECTED_HASHES}
    if current_frozen_hashes != prereg.get("input_hashes"):
        raise RuntimeError("one or more frozen D361 inputs changed after prepare")
    if any(current_frozen_hashes[_rel(path)] != expected for path, expected in EXPECTED_HASHES.items()):
        raise RuntimeError("one or more frozen D361 inputs differ from the registered authority hashes")
    current_sidecar = {_rel(path): _sha256_file(path) for path in D334_SIDECAR}
    if current_sidecar != prereg["d334_sidecar_before"]:
        raise RuntimeError("D334 sidecar changed before D361 run")
    if _tree_manifest(D360_DIR) != prereg["d360_tree_before"]:
        raise RuntimeError("immutable D360 output tree changed before D361 run")
    expected_before = sorted(_rel(path) for path in _expected_artifact_paths(include_completion=False))
    expected_after = sorted(_rel(path) for path in _expected_artifact_paths(include_completion=True))
    if prereg.get("expected_artifacts_before_completion") != expected_before:
        raise RuntimeError("D361 preregistered precompletion inventory differs from the harness")
    if prereg.get("expected_artifacts_after_completion") != expected_after:
        raise RuntimeError("D361 preregistered completion inventory differs from the harness")
    capacity = prereg.get("capacity_budget_preflight")
    if not isinstance(capacity, dict):
        raise RuntimeError("D361 capacity preflight payload is missing")
    if prereg.get("capacity_budget_preflight_canonical_sha256") != _sha256_bytes(_canonical(capacity)):
        raise RuntimeError("D361 capacity preflight payload digest mismatch")
    if capacity.get("pass") is not True:
        raise RuntimeError("D361 capacity preflight did not pass")
    loaded_before = _forbidden_loaded_modules()
    if loaded_before:
        raise RuntimeError(f"forbidden runtime modules loaded before D361 invocation: {loaded_before}")
    invocation = {
        "artifact": "D361_SINGLE_OFFLINE_INVOCATION_MARKER_V1",
        "case": CASE,
        "utc": _utc(),
        "pid": os.getpid(),
        "harness_sha256": _sha256_file(HARNESS),
        "isaac_or_physx_invocation": False,
        "retry_allowed": False,
    }
    _write_json_exclusive(INVOCATION_PATH, invocation)
    marker = DurableMarkerStream(PHASE_PATH)
    result_journal: DurableMarkerStream | None = None
    try:
        marker.append("offline_invocation_started")
        _write_json_exclusive(CAPACITY_PATH, capacity)
        marker.append("capacity_budget_durable", registered_total_capacity=REGISTERED_TOTAL_CAPACITY)
        if not capacity["pass"]:
            raise RuntimeError(f"capacity budget failed: {capacity['checks']}")

        protocol = _protocol_contract()
        _write_json_exclusive(PROTOCOL_PATH, protocol)
        marker.append("prefix_protocol_contract_durable")
        if not protocol["pass"]:
            raise RuntimeError("prefix protocol contract failed")

        result_journal = DurableMarkerStream(FAILURE_JOURNAL_PATH)
        perturbation = _failure_injection_evaluation(marker, result_journal)
        _write_json_exclusive(PERTURBATION_PATH, perturbation)
        result_journal.close()
        marker.append("failure_injection_results_durable", passed=perturbation["passed"], total=perturbation["total"])
        if not perturbation["pass"]:
            raise RuntimeError(f"failure injection evaluation failed: {perturbation['test_passes']}")

        sidecar_after = {_rel(path): _sha256_file(path) for path in D334_SIDECAR}
        if sidecar_after != prereg["d334_sidecar_before"]:
            raise RuntimeError("D334 sidecar changed during D361")
        forbidden_after = _forbidden_loaded_modules()
        if forbidden_after:
            raise RuntimeError(f"forbidden main-process runtime modules loaded: {forbidden_after}")
        d360_tree_after = _tree_manifest(D360_DIR)
        if d360_tree_after != prereg["d360_tree_before"]:
            raise RuntimeError("immutable D360 output tree changed during D361")
        marker.append("precompletion_integrity_pass")
        marker.close()

        artifacts_before_completion = sorted(
            path for path in OUT_DIR.rglob("*") if path.is_file() and path != COMPLETION_PATH
        )
        expected_artifact_paths = _expected_artifact_paths(include_completion=False)
        if set(artifacts_before_completion) != expected_artifact_paths:
            raise RuntimeError(
                "D361 output inventory differs from the preregistered set: "
                f"actual={[str(path) for path in artifacts_before_completion]}"
            )
        inventory = {
            _rel(path): {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in artifacts_before_completion
        }
        completion = {
            "artifact": "D361_COMPLETION_SUMMARY_V1",
            "case": CASE,
            "case_name": CASE_NAME,
            "utc": _utc(),
            "verdict": "D361_CONTACT_CAPACITY_AND_PREFIX_TRACE_REPAIR_PASS_NO_PHYSICS",
            "new_variables": NEW_VARIABLES,
            "capacity_budget_pass": capacity["pass"],
            "registered_total_contact_capacity": REGISTERED_TOTAL_CAPACITY,
            "capacity_runtime_sufficiency": None,
            "prefix_protocol_contract_pass": protocol["pass"],
            "failure_injection_pass": perturbation["pass"],
            "failure_injection_passed": perturbation["passed"],
            "failure_injection_total": perturbation["total"],
            "d334_sidecar_unchanged": sidecar_after == prereg["d334_sidecar_before"],
            "d360_tree_unchanged": d360_tree_after == prereg["d360_tree_before"],
            "forbidden_main_modules_loaded": forbidden_after,
            "scope_counts": {
                "isaac_launch": 0,
                "physx_science_run": 0,
                "physics_step": 0,
                "q5_command": 0,
                "q5_sample": 0,
                "new_contact_image": 0,
                "new_contact_video": 0,
                "rrd_or_rbl": 0,
                "target_ik_path_change": 0,
                "asset_or_physics_setting_change": 0,
            },
            "scientific_results": {
                "contacting_body": None,
                "contact_force": None,
                "object_motion": None,
                "current_pose_support_or_rejection": None,
                "grasp_feasibility": None,
                "g0a_pass": False,
            },
            "future_authorization_boundary":
                "actual q5/PhysX science rerun and new contact video require separate explicit user approval",
            "artifact_inventory_before_completion": inventory,
            "pass": True,
        }
        _write_json_exclusive(COMPLETION_PATH, completion)
        artifacts_after_completion = {path for path in OUT_DIR.rglob("*") if path.is_file()}
        if artifacts_after_completion != _expected_artifact_paths(include_completion=True):
            raise RuntimeError("D361 final output inventory differs from the preregistered exact set")
    except BaseException as exc:
        if result_journal is not None:
            result_journal.close()
        try:
            marker.append(
                "offline_invocation_failed",
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except Exception:
            pass
        marker.close()
        failure_payload = {
            "artifact": "D361_RUNTIME_EXCEPTION_V1",
            "case": CASE,
            "utc": _utc(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "artifacts_at_failure": _tree_manifest(OUT_DIR),
            "isaac_launch_count": 0,
            "physx_step_count": 0,
            "q5_sample_count": 0,
            "new_contact_media_count": 0,
            "retry_allowed": False,
        }
        if not RUNTIME_EXCEPTION_PATH.exists():
            _write_json_exclusive(RUNTIME_EXCEPTION_PATH, failure_payload)
        raise
    finally:
        if result_journal is not None:
            result_journal.close()
        marker.close()
    print(json.dumps({"stage": "run", "pass": True, "verdict": completion["verdict"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "child", "pxr_inventory"))
    parser.add_argument("--scenario")
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "run":
        _run()
    elif args.stage == "child":
        if args.scenario is None or args.path is None:
            parser.error("child stage requires --scenario and --path")
        _child_scenario(args.scenario, args.path)
    elif args.stage == "pxr_inventory":
        if args.path is None:
            parser.error("pxr_inventory stage requires --path")
        _pxr_inventory(args.path)


if __name__ == "__main__":
    main()
