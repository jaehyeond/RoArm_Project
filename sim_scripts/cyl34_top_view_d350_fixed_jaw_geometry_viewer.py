#!/usr/bin/env python3
"""D350 zero-step fixed-jaw geometry measurement and live Isaac Viewer.

This case reproduces the exact D349 Float32 target state, binds the retained
raw link5 surface component that contains D349's authoritative nearest witness,
measures its geometry, and displays the assembled 64+64 callback-topology
colliders in a real Isaac Viewer.  It never advances physics or mutates an
asset, target, tolerance, material, actuator, or physics setting.
"""
from __future__ import annotations

import argparse
import asyncio
import colorsys
import copy
import hashlib
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.rerun_contract import (  # noqa: E402
    RERUN_CONTRACT_VERSION,
    sha256_file,
    validate_rerun_artifact,
)
from roarm_rl.viz_debug import log_rerun  # noqa: E402
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d349_grasp_g0a_frozen_open_jaw_target_live_distance_gate as d349,
)


CASE = "g0a_d350"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350"
PREREG_PATH = OUT_DIR / "d350_preregistration.json"
PARAMETER_PATH = OUT_DIR / "d350_parameter_freeze_audit.json"
PREFLIGHT_PATH = OUT_DIR / "d350_validate_preflight.json"
BINDING_PATH = OUT_DIR / "d350_fixed_jaw_semantic_binding.json"
MEASUREMENT_PATH = OUT_DIR / "d350_fixed_jaw_geometry_measurement.json"
LIVE_BINDING_PATH = OUT_DIR / "d350_live_topology_runtime_binding.json"
OVERLAY_PATH = OUT_DIR / "d350_viewer_overlay_contract.json"
CAPTURE_PATH = OUT_DIR / "d350_viewer_capture_contract.json"
RRD_PATH = OUT_DIR / "d350_fixed_jaw_geometry.rrd"
RBL_PATH = OUT_DIR / "d350_fixed_jaw_geometry.rbl"
RERUN_SCREENSHOT_PATH = OUT_DIR / "d350_fixed_jaw_geometry_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d350_rerun_validation.json"
AUTOMATED_PATH = OUT_DIR / "d350_automated_summary.json"
AUTOMATED_MD_PATH = OUT_DIR / "d350_automated_report.md"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d350_runtime_exception.json"
MANUAL_PATH = OUT_DIR / "d350_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d350_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d350_completion_summary.json"
COMPLETION_MD_PATH = OUT_DIR / "d350_completion_report.md"

VIEWER_PNGS = {
    "whole_oblique_physx": OUT_DIR / "d350_whole_oblique_actual_physx_colliders.png",
    "tool_oblique_physx": OUT_DIR / "d350_tool_oblique_actual_physx_colliders.png",
    "whole_oblique_colored": OUT_DIR / "d350_whole_oblique_colored_64plus64.png",
    "tool_top": OUT_DIR / "d350_tool_top_colored_64plus64.png",
    "tool_side": OUT_DIR / "d350_tool_side_colored_64plus64.png",
    "tool_oblique": OUT_DIR / "d350_tool_oblique_colored_64plus64.png",
}

SESSION_DOC = REPO / "claudedocs/session_20260714_grasp_g0a_d350_fixed_jaw_geometry_viewer.md"
START_HERE = REPO / "START_HERE.md"
HARNESS = Path(__file__).resolve()
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"

D349_MEASUREMENT = d349.MEASUREMENT_PATH
D349_COMPLETION = d349.COMPLETION_PATH
D349_BINDING = d349.BINDING_PATH
D349_CORRECTED = d349.CORRECTED_AUDIT_PATH
D348_EVIDENCE = d349.D348_EVIDENCE
D334_SUMMARY = d349.D334_SUMMARY
VARIANT_ROBOT_USD = d349.VARIANT_ROBOT_USD
VARIANT_PHYSICS_USD = d349.VARIANT_PHYSICS_USD

EXPECTED_HEAD = "647dfe6ba8e13c781b39850bf7228010fd1683b4"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
NEW_VARIABLES = [
    "fixed_jaw_semantic_surface_binding",
    "frozen_target_fixed_jaw_centerline_measurement",
]
NEW_PHYSICAL_VARIABLES: list[str] = []
SEED = 33201
Q_TARGET_F32 = np.asarray(
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
OBJECT_POS_F32 = np.asarray(
    [0.30000001192092896, 0.0, 0.03288299962878227], dtype=np.float32
)
OBJECT_QUAT_F32 = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
BINDING_RESIDUAL_MAX_M = 1.0e-5
BINDING_FACE_TIE_M = 1.0e-9
ANCHOR_TOL_MM = 1.0e-6
CLEAR_GATE_MM = 0.1
FIDELITY_TOL_MM = 0.5
GUIDE_ROOT = "/World/D350ViewerGuides"
PHYSX_COLLIDER_SETTING = "/persistent/physics/visualizationDisplayColliders"
GUIDE_PURPOSE_SETTING = "/persistent/app/hydra/displayPurpose/guide"

EXPECTED_INPUT_HASHES = {
    "variant_robot_usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "variant_physics_usd": "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
    "d348_evidence": "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    "d349_corrected": "7e3d79f36e54fec4940bc58ecb81d4d13113329129b9d0926e0c65436cb5c079",
    "d349_binding": "9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05",
    "d349_measurement": "5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300",
    "d349_completion": "6ec883c4ebf4dd25aa2795006699b1d09e3b554412e2dcfa86277de541bd677e",
    "urdf": "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2",
}

# These unrelated files appeared after D350 approval.  They are user-owned and
# are hash-pinned so this case neither edits nor absorbs them.
EXTERNAL_DIRTY_BASELINE = {
    "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition/attempt3_collider_decomposition.rbl": {
        "bytes": 82728,
        "sha256": "e189210548688aa881e96aa2cba7661d6883fde95892639b3fa3d46291fdf292",
    },
    "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition/attempt3_collider_decomposition.rrd": {
        "bytes": 989885,
        "sha256": "718656e2fd9d7a5501a5414ddccfb13fc3e288993604f55372d443705febec1f",
    },
    "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition/attempt3_collider_decomposition_manual_inspection.md": {
        "bytes": 3275,
        "sha256": "2a65dc47c03454585ec4040f213551b94ce292afaa84d0f83c0f066e8d2d2735",
    },
    "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition/attempt3_collider_decomposition_provenance.json": {
        "bytes": 1961,
        "sha256": "7eccbe461c41a1b12baa13bebe25a86922af9a0b3b4a96ef4b2023fe480a6846",
    },
    "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition/attempt3_collider_decomposition_validation.json": {
        "bytes": 278690,
        "sha256": "f8c9b86bd3e173014b0b2b5ed6e89e5ad3abe3fbabb6fb6c277c49a45b6e192c",
    },
    "sim_scripts/cyl34_top_view_labmeeting_20260715_attempt3_collider_decomposition_viz.py": {
        "bytes": 19355,
        "sha256": "0391c808737f58482e246114253de1d1aba8a04b9758ebad4c62bee3fab814f0",
    },
}

FRAME_NAMES = ("base", "cylinder", "link5_actual", "fixed_jaw_axis")
METRIC_NAMES = (
    "fixed_axis_radial_angle_deg",
    "fixed_axis_to_link5_plus_z_deg",
    "fixed_axis_pitch_deg",
    "centerline_height_error_mm",
    "centerline_tangent_offset_mm",
    "surface_normal_to_center_angle_deg",
    "surface_normal_to_gap_angle_deg",
    "legacy_proxy_to_actual_witness_mm",
    "link5_raw_clearance_mm",
    "link5_live_clearance_mm",
    "gripper_raw_clearance_mm",
    "gripper_live_clearance_mm",
    "simulation_counter",
)
EVENT_NAMES = ("scope", "binding", "geometry", "viewer", "gate")
ARROW_NAMES = (
    "base_to_cylinder_radial",
    "cylinder_tangent",
    "fixed_jaw_centerline",
    "fixed_surface_normal",
    "raw_gap_witness",
)
EXPECTED_RERUN_COUNTS = {
    "frame_count": len(FRAME_NAMES),
    "coordinate_frame_count": 3,
    "mesh_count": 130,
    "point_entity_count": 1,
    "arrow_entity_count": len(ARROW_NAMES),
    "scalar_row_count": len(METRIC_NAMES),
    "event_row_count": len(EVENT_NAMES),
    "exact_non_system_entity_count": 296,
}

VERDICT_INPUT = "D350_FROZEN_INPUT_CONTRACT_FAIL_STOP"
VERDICT_BINDING = "D350_FIXED_JAW_SEMANTIC_BINDING_FAIL_STOP"
VERDICT_GEOMETRY = "D350_FIXED_JAW_GEOMETRY_MEASUREMENT_FAIL_STOP"
VERDICT_VISUAL = "D350_VIEWER_OR_RERUN_CONTRACT_FAIL_STOP"
VERDICT_PENDING = "D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_MANUAL_PENDING"
VERDICT_COMPLETE = "D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> dict[str, str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
    result: dict[str, str] = {}
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        status = field[:2]
        path = field[3:]
        if status[0] in "RC" and index < len(fields):
            path = fields[index]
            index += 1
        result[path] = status
    return result


def _allowed_status() -> set[str]:
    return {
        _rel(START_HERE),
        _rel(SESSION_DOC),
        _rel(HARNESS),
        *EXTERNAL_DIRTY_BASELINE,
    }


def _status_scope_pass(status: dict[str, str]) -> bool:
    prefix = _rel(OUT_DIR) + "/"
    return all(path in _allowed_status() or path.startswith(prefix) for path in status)


def _external_baseline() -> dict[str, Any]:
    status = _git_status()
    rows = []
    for relative, expected in sorted(EXTERNAL_DIRTY_BASELINE.items()):
        path = REPO / relative
        row = {
            "path": relative,
            "exists": path.is_file(),
            "status": status.get(relative),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha(path) if path.is_file() else None,
            "expected": {"status": "??", **expected},
        }
        row["pass"] = bool(
            row["exists"]
            and row["status"] == "??"
            and row["bytes"] == expected["bytes"]
            and row["sha256"] == expected["sha256"]
        )
        rows.append(row)
    return {"rows": rows, "pass": all(row["pass"] for row in rows)}


def _input_paths() -> dict[str, Path]:
    return {
        "variant_robot_usd": VARIANT_ROBOT_USD,
        "variant_physics_usd": VARIANT_PHYSICS_USD,
        "d348_evidence": D348_EVIDENCE,
        "d349_corrected": D349_CORRECTED,
        "d349_binding": D349_BINDING,
        "d349_measurement": D349_MEASUREMENT,
        "d349_completion": D349_COMPLETION,
        "urdf": URDF_PATH,
    }


def _input_hashes() -> dict[str, str]:
    return {name: _sha(path) for name, path in _input_paths().items()}


def _helper_hashes() -> dict[str, str]:
    return {
        "d332": _sha(Path(d332.__file__).resolve()),
        "d333": _sha(Path(d333.__file__).resolve()),
        "d334": _sha(Path(d334.__file__).resolve()),
        "d339": _sha(Path(d339.__file__).resolve()),
        "d349": _sha(Path(d349.__file__).resolve()),
        "viz_debug": _sha(REPO / "roarm_rl/viz_debug.py"),
        "rerun_contract": _sha(REPO / "roarm_rl/rerun_contract.py"),
    }


def _png_dimensions(path: Path) -> str | None:
    if not path.is_file():
        return None
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return f"{width}x{height}"


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities = {"metadata/run", "geometry/key_points"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "geometry/key_points": [
            "CoordinateFrame:frame",
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ],
    }
    transform = [
        "Transform3D:child_frame",
        "Transform3D:parent_frame",
        "Transform3D:quaternion",
        "Transform3D:translation",
    ]
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for body in ("link5", "gripper_link", "object"):
        path = f"coordinate_frames/{body}/body_local"
        entities.add(path)
        components[path] = transform
    for name in FRAME_NAMES:
        frame = f"frames/{name}"
        origin = f"frames/{name}/origin"
        entities.update({frame, origin})
        components[frame] = transform
        components[origin] = [
            "CoordinateFrame:frame",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ]
    mesh_paths = []
    for body in ("link5", "gripper_link"):
        mesh_paths.extend(f"geometry/live_parts/{body}/part_{idx:03d}" for idx in range(64))
    mesh_paths.extend(
        ["geometry/fixed_jaw/raw_connected_component", "geometry/target/cylinder_collider"]
    )
    for path in mesh_paths:
        metadata = f"metadata/meshes/{path.replace('/', '__')}"
        entities.update({path, metadata})
        components[path] = mesh_components
        components[metadata] = ["TextDocument:text"]
    for name in ARROW_NAMES:
        path = f"geometry/arrows/{name}"
        entities.add(path)
        components[path] = [
            "Arrows3D:colors",
            "Arrows3D:labels",
            "Arrows3D:origins",
            "Arrows3D:radii",
            "Arrows3D:vectors",
            "CoordinateFrame:frame",
        ]
    for name in METRIC_NAMES:
        path = f"metrics/d350/{name}"
        entities.add(path)
        components[path] = ["Scalars:scalars"]
    for name in EVENT_NAMES:
        path = f"events/d350/{name}"
        entities.add(path)
        components[path] = ["TextLog:level", "TextLog:text"]
    return sorted(entities), components


def _rrd_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": [
            "blueprint",
            "event_idx",
            "log_time",
            "measurement_idx",
            "part_idx",
        ],
        "required_components_by_path": components,
        "exact_observation_counts": EXPECTED_RERUN_COUNTS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parameter_audit() -> dict[str, Any]:
    d349_measurement = _json(D349_MEASUREMENT)
    completion = _json(D349_COMPLETION)
    d349_binding = _json(D349_BINDING)
    q_recorded = np.asarray(
        d349_measurement["target_state_guard"]["actual_joint_rad_float32"], dtype=np.float32
    )
    object_recorded = np.asarray(
        d349_measurement["target_state_guard"]["object_pos_w_m"], dtype=np.float32
    )
    object_quat_recorded = np.asarray(
        d349_measurement["target_state_guard"]["object_quat_wxyz"], dtype=np.float32
    )
    checks = {
        "new_variable_count_two": len(NEW_VARIABLES) == 2,
        "new_physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "expected_head_current": _git_head() == EXPECTED_HEAD,
        "input_hashes_exact": _input_hashes() == EXPECTED_INPUT_HASHES,
        "d349_complete": completion.get("final_verdict")
        == "D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED",
        "d349_g0a_false": completion.get("scope_guards", {}).get("g0a_pass") is False,
        "target_joint_float32_exact": np.array_equal(q_recorded, Q_TARGET_F32),
        "target_object_float32_exact": np.array_equal(object_recorded, OBJECT_POS_F32),
        "target_quaternion_float32_exact": np.array_equal(
            object_quat_recorded, OBJECT_QUAT_F32
        ),
        "target_contract_frozen": d349_measurement["target_contract"]
        == {
            "ik": "HOME-seeded position-only",
            "q5_rad": 1.5413,
            "radial_offset_mm": 7.0,
            "seed": 33201,
            "tangent_offset_mm": 11.0,
            "tangent_sign": -1.0,
        },
        "decomposition_64_plus_64": bool(
            d349_binding.get("checks", {}).get("part_rows_128")
            and len(d349_binding.get("parts", [])) == 128
            and all(
                d349_binding.get("body_checks", {})
                .get(body, {})
                .get("part_count_64")
                is True
                for body in ("link5", "gripper_link")
            )
        ),
        "distance_gates_frozen": CLEAR_GATE_MM == 0.1 and FIDELITY_TOL_MM == 0.5,
        "binding_residual_registered": BINDING_RESIDUAL_MAX_M == 1.0e-5,
        "alignment_success_tolerances_absent": True,
        "viewer_is_observability_not_variable": True,
        "physics_scope_zero": True,
        "rerun_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rrd_entity_count_registered": len(_expected_rrd_contract()[0])
        == EXPECTED_RERUN_COUNTS["exact_non_system_entity_count"],
    }
    return {
        "artifact": "D350_PARAMETER_FREEZE_AUDIT_V1",
        "case": CASE,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "frozen_target_joint_float32": Q_TARGET_F32.tolist(),
        "frozen_object_position_float32": OBJECT_POS_F32.tolist(),
        "frozen_object_quaternion_float32": OBJECT_QUAT_F32.tolist(),
        "binding_numerical_tolerances": {
            "seed_surface_residual_max_m": BINDING_RESIDUAL_MAX_M,
            "minimum_face_tie_m": BINDING_FACE_TIE_M,
            "d349_anchor_reproduction_mm": ANCHOR_TOL_MM,
        },
        "alignment_success_tolerances": None,
        "distance_gates": {"clear_min_mm": CLEAR_GATE_MM, "raw_live_delta_max_mm": FIDELITY_TOL_MM},
        "scope_guards": {
            "asset_write": False,
            "decomposition_change": False,
            "target_change": False,
            "tolerance_change": False,
            "material_mass_actuator_physics_change": False,
            "fresh_cook_callback_or_property_query": False,
            "controlled_physics_steps": 0,
            "settle": False,
            "ten_trial": False,
            "g0b": False,
            "rl_or_ppo": False,
            "ladder_promotion": False,
            "g0a_pass": False,
        },
        "rerun": {"sdk": "0.34.1", "counts": EXPECTED_RERUN_COUNTS, "contract_sha256": _rrd_digest()},
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_prepare(args: argparse.Namespace) -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"D350 output is forward-only and already exists: {OUT_DIR}")
    parameter = _parameter_audit()
    _write_json(PARAMETER_PATH, parameter)
    source_inventories = d349._source_inventories()
    external = _external_baseline()
    status = _git_status()
    prereg_checks = {
        "parameter_pass": parameter["pass"],
        "head_exact": _git_head() == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(status),
        "external_user_files_unchanged": external["pass"],
        "session_doc_present": SESSION_DOC.is_file(),
        "start_here_active_d350": "Active Case — D350" in START_HERE.read_text(encoding="utf-8"),
        "harness_present": HARNESS.is_file(),
        "input_hashes_exact": _input_hashes() == EXPECTED_INPUT_HASHES,
        "output_contains_only_parameter": sorted(path.name for path in OUT_DIR.iterdir())
        == [PARAMETER_PATH.name],
    }
    prereg = {
        "artifact": "D350_PREREGISTRATION_V1",
        "case": CASE,
        "git_head": _git_head(),
        "git_status": status,
        "prepare_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "state_hashes": {"start_here": _sha(START_HERE), "session_doc": _sha(SESSION_DOC)},
        "harness_sha256": _sha(HARNESS),
        "parameter_sha256": _sha(PARAMETER_PATH),
        "input_hashes": _input_hashes(),
        "helper_hashes": _helper_hashes(),
        "source_inventories": source_inventories,
        "external_dirty_baseline": external,
        "exact_target": {
            "joint_float32": Q_TARGET_F32.tolist(),
            "object_position_float32": OBJECT_POS_F32.tolist(),
            "object_quaternion_float32": OBJECT_QUAT_F32.tolist(),
        },
        "semantic_binding": {
            "seed": "D349 authoritative raw link5 nearest_point_geometry_m",
            "surface": "retained raw link5 triangle mesh",
            "component": "exact-welded vertex-connected triangle component",
            "seed_residual_max_m": BINDING_RESIDUAL_MAX_M,
            "negative_control": "gripper_link q5 child rejected as fixed-jaw owner",
        },
        "measurement_verdict_vocabulary": ["MEASURED", "FAIL_STOP"],
        "aligned_pass_forbidden": True,
        "viewer": {
            "launcher": {"headless": False, "livestream": 0, "enable_cameras": False, "xr": False, "device": "cuda:0"},
            "actual_physx_debug_and_colored_guides_separate": True,
            "colored_guides_collision_api_forbidden": True,
            "viewport_pngs": {name: _rel(path) for name, path in VIEWER_PNGS.items()},
            "ui_update_only": True,
        },
        "rerun_contract_sha256": _rrd_digest(),
        "rerun_expected_counts": EXPECTED_RERUN_COUNTS,
        "rerun_exact_entities": _expected_rrd_contract()[0],
        "scope_guards": parameter["scope_guards"],
        "checks": prereg_checks,
        "pass": all(prereg_checks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0 if prereg["pass"] else 2


def _app_arg_checks(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "headless_false": getattr(args, "headless", None) is False,
        "livestream_zero": int(getattr(args, "livestream", -1)) == 0,
        "cameras_disabled": getattr(args, "enable_cameras", None) is False,
        "xr_disabled": getattr(args, "xr", None) is False,
        "device_cuda_zero": str(getattr(args, "device", "")) == "cuda:0",
        "cpu_false": getattr(args, "cpu", None) is False,
        "experience_default": str(getattr(args, "experience", "")) == "",
        "kit_args_empty": str(getattr(args, "kit_args", "")) == "",
    }


def _resolved_gui_launcher(launcher: Any) -> dict[str, Any]:
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
        "headless_false": values["headless"] is False,
        "livestream_zero": values["livestream"] == 0,
        "cameras_disabled": values["enable_cameras"] is False,
        "xr_disabled": values["xr"] is False,
        "offscreen_false": values["offscreen_render"] is False,
        "device_zero": values["device_id"] == 0,
        "gui_experience": Path(values["experience"]).name == "isaaclab.python.kit",
    }
    return {"values": values, "checks": checks, "pass": all(checks.values())}


def _runtime_outputs() -> list[Path]:
    return [
        PREFLIGHT_PATH,
        BINDING_PATH,
        MEASUREMENT_PATH,
        LIVE_BINDING_PATH,
        OVERLAY_PATH,
        CAPTURE_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_VALIDATION_PATH,
        AUTOMATED_PATH,
        AUTOMATED_MD_PATH,
        RUNTIME_EXCEPTION_PATH,
        MANUAL_PATH,
        MANUAL_MD_PATH,
        COMPLETION_PATH,
        COMPLETION_MD_PATH,
        *VIEWER_PNGS.values(),
    ]


def _automated_evidence_paths() -> dict[str, Path]:
    return {
        "parameter": PARAMETER_PATH,
        "preregistration": PREREG_PATH,
        "preflight": PREFLIGHT_PATH,
        "semantic_binding": BINDING_PATH,
        "geometry_measurement": MEASUREMENT_PATH,
        "live_binding": LIVE_BINDING_PATH,
        "viewer_overlay": OVERLAY_PATH,
        "viewer_capture": CAPTURE_PATH,
        "rrd": RRD_PATH,
        "rbl": RBL_PATH,
        "rerun_screenshot": RERUN_SCREENSHOT_PATH,
        "rerun_validation": RERUN_VALIDATION_PATH,
    }


def _automated_evidence_hashes() -> dict[str, str]:
    paths = _automated_evidence_paths()
    if not all(path.is_file() for path in paths.values()):
        return {}
    return {name: _sha(path) for name, path in paths.items()}


def _validate_preflight(args: argparse.Namespace) -> bool:
    import torch

    prereg = _json(PREREG_PATH)
    parameter = _json(PARAMETER_PATH)
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "parameter_pass": parameter.get("pass") is True,
        "fresh_process_pid": prereg["prepare_process_identity"]["pid"] != os.getpid(),
        "fresh_process_nonce": prereg["prepare_process_identity"]["nonce"] != args.process_nonce,
        "head_exact": _git_head() == prereg["git_head"] == EXPECTED_HEAD,
        "status_scope": _status_scope_pass(_git_status()),
        "state_hashes": prereg["state_hashes"]
        == {"start_here": _sha(START_HERE), "session_doc": _sha(SESSION_DOC)},
        "harness_hash": prereg["harness_sha256"] == _sha(HARNESS),
        "parameter_hash": prereg["parameter_sha256"] == _sha(PARAMETER_PATH),
        "input_hashes": prereg["input_hashes"] == _input_hashes() == EXPECTED_INPUT_HASHES,
        "helper_hashes": prereg["helper_hashes"] == _helper_hashes(),
        "source_inventories": prereg["source_inventories"] == d349._source_inventories(),
        "external_user_files": _external_baseline()["pass"],
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "rerun_pin": str(rr.__version__) == RERUN_CONTRACT_VERSION == "0.34.1",
        "registered_python": str(Path(sys.executable).resolve()) == str(Path(REGISTERED_PYTHON).resolve()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_4090": bool(torch.cuda.is_available() and "4090" in torch.cuda.get_device_name(0)),
        "display_present": bool(os.environ.get("DISPLAY")),
        "app_args": all(_app_arg_checks(args).values()),
        "runtime_outputs_absent": all(not path.exists() for path in _runtime_outputs()),
    }
    report = {
        "artifact": "D350_VALIDATE_PREFLIGHT_V1",
        "case": CASE,
        "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "environment": {
            "python": str(Path(sys.executable).resolve()),
            "numpy": str(np.__version__),
            "psutil": str(psutil.__version__),
            "rerun": str(rr.__version__),
            "display": os.environ.get("DISPLAY"),
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "app_arg_checks": _app_arg_checks(args),
        "git_status": _git_status(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(PREFLIGHT_PATH, report)
    return bool(report["pass"])


def _quat_to_rot(quat_wxyz: Any) -> np.ndarray:
    return d332._quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64))


def _rot_to_quat_xyzw(rot: np.ndarray) -> list[float]:
    from roarm_rl.viz_debug import rot_to_quat_wxyz

    q = rot_to_quat_wxyz(rot)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _target_guard(inner: Any) -> dict[str, Any]:
    actual_q = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    obj_pos, obj_quat = d334._object_pose_w(inner)
    obj_pos32 = np.asarray(obj_pos, dtype=np.float32)
    obj_quat32 = np.asarray(obj_quat, dtype=np.float32)
    checks = {
        "joint_float32_bit_exact": np.array_equal(actual_q, Q_TARGET_F32),
        "q5_open_float32_bit_exact": actual_q[5].tobytes() == Q_TARGET_F32[5].tobytes(),
        "object_position_float32_bit_exact": np.array_equal(obj_pos32, OBJECT_POS_F32),
        "object_quaternion_float32_bit_exact": np.array_equal(obj_quat32, OBJECT_QUAT_F32),
        "counter_zero": int(inner._sim_step_counter) == 0,
    }
    return {
        "actual_joint_rad_float32": actual_q.tolist(),
        "expected_joint_rad_float32": Q_TARGET_F32.tolist(),
        "object_pos_float32": obj_pos32.tolist(),
        "expected_object_pos_float32": OBJECT_POS_F32.tolist(),
        "object_quat_float32": obj_quat32.tolist(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _closest_point_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w


def _component_digest(vertices: np.ndarray, triangles: np.ndarray, face_ids: np.ndarray) -> str:
    payload = b"".join(
        [
            np.ascontiguousarray(vertices, dtype="<f8").tobytes(),
            np.ascontiguousarray(triangles, dtype="<i8").tobytes(),
            np.ascontiguousarray(face_ids, dtype="<i8").tobytes(),
        ]
    )
    return hashlib.sha256(payload).hexdigest()


def _bind_component(
    raw_vertices: np.ndarray,
    raw_triangles: np.ndarray,
    seed_local: np.ndarray,
    *,
    reverse_order: bool,
) -> dict[str, Any]:
    vertices = np.asarray(raw_vertices, dtype=np.float64)
    triangles = np.asarray(raw_triangles, dtype=np.int64)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded = inverse[triangles]
    face_count = len(triangles)
    parent = np.arange(face_count, dtype=np.int64)
    rank = np.zeros(face_count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    owners: dict[int, int] = {}
    order = range(face_count - 1, -1, -1) if reverse_order else range(face_count)
    for face_idx in order:
        for vertex_idx in welded[face_idx]:
            key = int(vertex_idx)
            if key in owners:
                union(face_idx, owners[key])
            else:
                owners[key] = face_idx

    distances = np.empty(face_count, dtype=np.float64)
    closest = np.empty((face_count, 3), dtype=np.float64)
    for face_idx, ids in enumerate(triangles):
        a, b, c = vertices[ids]
        q = _closest_point_triangle(seed_local, a, b, c)
        closest[face_idx] = q
        distances[face_idx] = float(np.linalg.norm(seed_local - q))
    minimum = float(np.min(distances))
    tied = np.flatnonzero(distances <= minimum + BINDING_FACE_TIE_M)
    roots = sorted({find(int(face_idx)) for face_idx in tied})
    selected_root = roots[0] if len(roots) == 1 else -1
    component_faces = np.asarray(
        sorted(face_idx for face_idx in range(face_count) if find(face_idx) == selected_root),
        dtype=np.int64,
    )
    used = np.unique(welded[component_faces].reshape(-1)) if len(component_faces) else np.asarray([], dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    component_vertices = unique[used] if len(used) else np.zeros((0, 3), dtype=np.float64)
    component_triangles = (
        np.asarray([[remap[int(value)] for value in welded[idx]] for idx in component_faces], dtype=np.int64)
        if len(component_faces)
        else np.zeros((0, 3), dtype=np.int64)
    )
    seed_face = int(tied[0]) if len(tied) else -1
    digest = _component_digest(component_vertices, component_triangles, component_faces)
    return {
        "minimum_distance_m": minimum,
        "tied_face_indices": tied.tolist(),
        "tied_component_roots": roots,
        "seed_face_index": seed_face,
        "seed_face_closest_local_m": closest[seed_face].tolist() if seed_face >= 0 else None,
        "component_face_indices": component_faces,
        "component_vertices": component_vertices,
        "component_triangles": component_triangles,
        "component_digest": digest,
        "welded_unique_vertex_count": int(len(unique)),
        "component_count": int(len({find(face_idx) for face_idx in range(face_count)})),
    }


def _joint_semantics() -> dict[str, Any]:
    root = ET.parse(URDF_PATH).getroot()
    joint = next(item for item in root.findall("joint") if item.attrib.get("name") == "link5_to_gripper_link")
    parent = joint.find("parent").attrib["link"]
    child = joint.find("child").attrib["link"]
    checks = {
        "joint_revolute": joint.attrib.get("type") == "revolute",
        "selected_fixed_owner_is_urdf_parent_link5": parent == "link5",
        "moving_candidate_is_urdf_child_gripper_link": child == "gripper_link",
        "moving_child_rejected_by_parent_mismatch": child != parent,
    }
    return {"joint": joint.attrib.get("name"), "type": joint.attrib.get("type"), "parent": parent, "child": child, "checks": checks, "pass": all(checks.values())}


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1.0e-15)
    b /= max(float(np.linalg.norm(b)), 1.0e-15)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _bind_and_measure(
    inner: Any,
    raw_shapes: list[dict[str, Any]],
    raw_set: dict[str, Any],
    live_set: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    d349_measurement = _json(D349_MEASUREMENT)
    historical_raw = next(
        row for row in d349_measurement["raw_mesh"]["queries"] if row["body"] == "link5"
    )
    raw_shape = next(row for row in raw_shapes if row["body"] == "link5")
    link_pos, link_quat = d334._body_pose_w(inner, "link5")
    link_rot = _quat_to_rot(link_quat)
    seed_world = np.asarray(historical_raw["nearest_point_geometry_m"], dtype=np.float64)
    seed_local = link_rot.T @ (seed_world - link_pos)
    first = _bind_component(raw_shape["_raw_verts"], raw_shape["_triangles"], seed_local, reverse_order=False)
    repeat = _bind_component(raw_shape["_raw_verts"], raw_shape["_triangles"], seed_local, reverse_order=True)
    joint = _joint_semantics()
    component_vertices = first["component_vertices"]
    component_triangles = first["component_triangles"]
    face_ids = first["component_face_indices"]
    checks = {
        "seed_residual_le_0p01mm": first["minimum_distance_m"] <= BINDING_RESIDUAL_MAX_M,
        "tied_faces_one_component": len(first["tied_component_roots"]) == 1,
        "component_nonempty": len(component_vertices) >= 4 and len(component_triangles) >= 4,
        "repeat_digest_exact": first["component_digest"] == repeat["component_digest"],
        "repeat_minimum_exact": first["minimum_distance_m"] == repeat["minimum_distance_m"],
        "owner_link5": raw_shape["body"] == "link5" and raw_shape["owner_body_path"] == d334.BODY_PATHS["link5"],
        "q5_child_negative_control_rejected": joint["pass"],
        "historical_witness_finite": bool(np.isfinite(seed_world).all()),
    }
    public_first = {
        "minimum_distance_m": first["minimum_distance_m"],
        "tied_face_indices": first["tied_face_indices"],
        "tied_component_roots": first["tied_component_roots"],
        "seed_face_index": first["seed_face_index"],
        "seed_face_closest_local_m": first["seed_face_closest_local_m"],
        "component_face_count": int(len(component_triangles)),
        "component_unique_vertex_count": int(len(component_vertices)),
        "component_digest": first["component_digest"],
        "welded_unique_vertex_count": first["welded_unique_vertex_count"],
        "raw_connected_component_count": first["component_count"],
        "component_bounds_local_m": [component_vertices.min(axis=0).tolist(), component_vertices.max(axis=0).tolist()] if len(component_vertices) else None,
    }
    binding = {
        "artifact": "D350_FIXED_JAW_SEMANTIC_BINDING_V1",
        "case": CASE,
        "semantics": "retained raw link5 connected surface containing D349 authoritative nearest witness",
        "not_used_as_authority": ["gripper_left_link.stl mount hypothesis", "legacy TCP-minus-8mm point proxy"],
        "seed_world_m": seed_world.tolist(),
        "seed_local_m": seed_local.tolist(),
        "first_binding": public_first,
        "repeat_component_digest": repeat["component_digest"],
        "joint_owner_negative_control": joint,
        "checks": checks,
        "pass": all(checks.values()),
    }

    if not binding["pass"]:
        return binding, {}, {"vertices": component_vertices, "triangles": component_triangles}

    centroid_local = component_vertices.mean(axis=0)
    covariance = np.cov((component_vertices - centroid_local).T, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis_local = eigenvectors[:, order[0]]
    if float(np.dot(axis_local, np.asarray([0.0, 0.0, 1.0]))) < 0.0:
        axis_local = -axis_local
    axis_world = link_rot @ axis_local
    axis_world /= float(np.linalg.norm(axis_world))
    object_pos, _ = d334._object_pose_w(inner)
    base = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    radial = np.asarray([object_pos[0] - base[0], object_pos[1] - base[1], 0.0], dtype=np.float64)
    radial /= float(np.linalg.norm(radial[:2]))
    tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) * d332.ADOPTED_TANGENT_SIGN
    axis_xy = axis_world.copy()
    axis_xy[2] = 0.0
    axis_xy_norm = float(np.linalg.norm(axis_xy))
    axis_xy_unit = axis_xy / max(axis_xy_norm, 1.0e-15)
    if float(np.dot(axis_xy_unit, radial)) < 0.0:
        axis_local = -axis_local
        axis_world = -axis_world
        axis_xy = -axis_xy
        axis_xy_unit = -axis_xy_unit
    centroid_world = link_rot @ centroid_local + link_pos
    denom = float(np.dot(axis_world[:2], axis_world[:2]))
    station_parameter = float(np.dot(object_pos[:2] - centroid_world[:2], axis_world[:2]) / denom)
    centerline_station = centroid_world + station_parameter * axis_world

    seed_face = int(first["seed_face_index"])
    raw_triangle = np.asarray(raw_shape["_raw_verts"], dtype=np.float64)[
        np.asarray(raw_shape["_triangles"], dtype=np.int64)[seed_face]
    ]
    normal_local = np.cross(raw_triangle[1] - raw_triangle[0], raw_triangle[2] - raw_triangle[0])
    normal_local /= max(float(np.linalg.norm(normal_local)), 1.0e-15)
    normal_world = link_rot @ normal_local
    toward_center = np.asarray(object_pos, dtype=np.float64) - seed_world
    if float(np.dot(normal_world, toward_center)) < 0.0:
        normal_world = -normal_world
        normal_local = -normal_local
    raw_by_body = {row["body"]: row for row in raw_set["queries"]}
    live_by_body = {row["body"]: row for row in live_set["queries"]}
    raw_link = raw_by_body["link5"]
    current_witness = np.asarray(raw_link["witness_endpoint_0_m"], dtype=np.float64)
    cylinder_witness = np.asarray(raw_link["witness_endpoint_1_m"], dtype=np.float64)
    gap_vector = cylinder_witness - current_witness
    tcp_world = link_pos + link_rot @ np.asarray([0.0, 0.0, 0.115428], dtype=np.float64)
    legacy_proxy = tcp_world + link_rot @ np.asarray([-0.008, 0.0, 0.0], dtype=np.float64)
    historical_values = {
        body: {
            "raw": float(d349_measurement["distance_gate"]["per_body"][body]["raw_exact_signed_distance_mm"]),
            "live": float(d349_measurement["distance_gate"]["per_body"][body]["live_topology_exact_signed_distance_mm"]),
        }
        for body in ("link5", "gripper_link")
    }
    distance_rows = {}
    distance_checks = {}
    for body in ("link5", "gripper_link"):
        raw_value = float(raw_by_body[body]["exact_signed_distance_mm"])
        live_value = float(live_by_body[body]["exact_signed_distance_mm"])
        delta = abs(raw_value - live_value)
        body_checks = {
            "raw_clear_ge_0p1mm": raw_value >= CLEAR_GATE_MM and not raw_by_body[body]["is_collision"],
            "live_clear_ge_0p1mm": live_value >= CLEAR_GATE_MM and not live_by_body[body]["is_collision"],
            "raw_live_delta_le_0p5mm": delta <= FIDELITY_TOL_MM,
            "raw_anchor_within_1e_6mm": abs(raw_value - historical_values[body]["raw"]) <= ANCHOR_TOL_MM,
            "live_anchor_within_1e_6mm": abs(live_value - historical_values[body]["live"]) <= ANCHOR_TOL_MM,
        }
        distance_rows[body] = {
            "raw_signed_distance_mm": raw_value,
            "live_signed_distance_mm": live_value,
            "raw_live_absolute_delta_mm": delta,
            "historical_d349": historical_values[body],
            "checks": body_checks,
            "pass": all(body_checks.values()),
        }
        distance_checks[body] = distance_rows[body]["pass"]

    surface_to_center_angle = _angle_deg(normal_world, toward_center)
    surface_to_gap_angle = _angle_deg(normal_world, gap_vector)
    metrics = {
        "fixed_axis_radial_angle_deg": _angle_deg(axis_xy_unit, radial),
        "fixed_axis_to_link5_plus_z_deg": _angle_deg(
            axis_local, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        ),
        "fixed_axis_pitch_deg": math.degrees(math.atan2(float(axis_world[2]), float(np.linalg.norm(axis_world[:2])))),
        "centerline_height_error_mm": float((centerline_station[2] - object_pos[2]) * 1000.0),
        "centerline_tangent_offset_mm": float(np.dot(centerline_station - object_pos, tangent) * 1000.0),
        "centerline_radial_residual_mm": float(np.dot(centerline_station - object_pos, radial) * 1000.0),
        "surface_normal_to_center_angle_deg": surface_to_center_angle,
        "surface_normal_to_gap_angle_deg": surface_to_gap_angle,
        "legacy_proxy_to_actual_witness_mm": float(np.linalg.norm(legacy_proxy - current_witness) * 1000.0),
        "actual_witness_height_error_mm": float((current_witness[2] - object_pos[2]) * 1000.0),
    }
    finite_values = [*metrics.values(), *eigenvalues.tolist(), *axis_world.tolist(), *normal_world.tolist()]
    measurement_checks = {
        "binding_pass": binding["pass"],
        "pca_finite": bool(np.isfinite(finite_values).all()),
        "pca_axis_unique": bool(eigenvalues[0] > eigenvalues[1]),
        "horizontal_axis_nonzero": axis_xy_norm > 1.0e-9,
        "seed_triangle_nondegenerate": float(np.linalg.norm(normal_local)) > 0.99,
        "raw_witness_reproduced": float(np.linalg.norm(current_witness - seed_world)) <= 1.0e-6,
        "both_distance_gates": all(distance_checks.values()),
        "alignment_tolerance_not_applied": True,
        "values_measured_not_aligned_pass": True,
    }
    measurement = {
        "artifact": "D350_FIXED_JAW_GEOMETRY_MEASUREMENT_V1",
        "case": CASE,
        "verdict_semantics": "MEASURED, not ALIGNED_PASS",
        "coordinate_basis": {"base_world_m": base.tolist(), "cylinder_center_world_m": object_pos.tolist(), "radial": radial.tolist(), "tangent_sign_minus_one": tangent.tolist()},
        "fixed_jaw_component": {
            "centroid_local_m": centroid_local.tolist(),
            "centroid_world_m": centroid_world.tolist(),
            "pca_eigenvalues_m2": eigenvalues.tolist(),
            "principal_axis_local": axis_local.tolist(),
            "principal_axis_world": axis_world.tolist(),
            "centerline_station_world_m": centerline_station.tolist(),
            "component_digest": first["component_digest"],
        },
        "actual_surface": {
            "historical_seed_world_m": seed_world.tolist(),
            "current_raw_witness_world_m": current_witness.tolist(),
            "cylinder_witness_world_m": cylinder_witness.tolist(),
            "seed_face_index": seed_face,
            "seed_triangle_centroid_world_m": (link_rot @ raw_triangle.mean(axis=0) + link_pos).tolist(),
            "oriented_surface_normal_local": normal_local.tolist(),
            "oriented_surface_normal_world": normal_world.tolist(),
            "gap_vector_world_m": gap_vector.tolist(),
        },
        "legacy_proxy": {"tcp_world_m": tcp_world.tolist(), "proxy_world_m": legacy_proxy.tolist(), "authority": False},
        "metrics": metrics,
        "distances": distance_rows,
        "frozen_input_reproduction_pass": all(distance_checks.values()),
        "checks": measurement_checks,
        "pass": all(measurement_checks.values()),
        "aligned_pass": None,
    }
    display = {
        "vertices": component_vertices,
        "triangles": component_triangles,
        "link_pos": link_pos,
        "link_rot": link_rot,
        "object_pos": object_pos,
        "base": base,
        "radial": radial,
        "tangent": tangent,
        "axis_world": axis_world,
        "centroid_world": centroid_world,
        "centerline_station": centerline_station,
        "normal_world": normal_world,
        "current_witness": current_witness,
        "cylinder_witness": cylinder_witness,
        "legacy_proxy": legacy_proxy,
        "tcp_world": tcp_world,
    }
    return binding, measurement, display


def _cylinder_mesh(segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    radius, half = d332.CYLINDER_RADIUS_M, 0.5 * d332.CYLINDER_HEIGHT_M
    vertices = []
    for z in (-half, half):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append([radius * math.cos(angle), radius * math.sin(angle), z])
    vertices.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
    triangles = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            [
                [index, nxt, segments + nxt],
                [index, segments + nxt, segments + index],
                [2 * segments, nxt, index],
                [2 * segments + 1, segments + index, segments + nxt],
            ]
        )
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _palette(body: str, index: int) -> list[int]:
    # Coprime index permutation prevents adjacent part IDs from receiving nearly
    # identical colors while retaining a cool link5 / warm gripper family.
    permuted = ((index * 37) % 64) / 63.0
    if body == "link5":
        hue = 0.48 + 0.22 * permuted
    else:
        hue = (0.99 + 0.16 * permuted) % 1.0
    saturation = 0.58 + 0.34 * (((index * 17) % 7) / 6.0)
    value = 0.72 + 0.26 * (((index * 29) % 5) / 4.0)
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(round(channel * 255)) for channel in rgb] + [185]


def _world_vertices(vertices: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (rot @ np.asarray(vertices, dtype=np.float64).T).T + pos


def _create_viewer_guides(inner: Any, topology_parts: dict[str, list[dict[str, Any]]], display: dict[str, Any]) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    session = stage.GetSessionLayer()
    root_layer = stage.GetRootLayer()
    edit_target_before = str(stage.GetEditTarget().GetLayer().identifier)
    created_paths = []

    def colorize(gprim: Any, rgba: list[int]) -> None:
        color = Gf.Vec3f(*(value / 255.0 for value in rgba[:3]))
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([color])
        gprim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set([rgba[3] / 255.0])

    def mesh_at(path: str, vertices: np.ndarray, triangles: np.ndarray, rgba: list[int]) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*[float(value) for value in row]) for row in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * int(len(triangles)))
        mesh.CreateFaceVertexIndicesAttr([int(value) for value in triangles.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        colorize(mesh, rgba)
        created_paths.append(path)

    def line(path: str, points: list[np.ndarray], color: tuple[float, float, float], width: float) -> None:
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateBasisAttr(UsdGeom.Tokens.bezier)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*[float(value) for value in point]) for point in points])
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreatePurposeAttr(UsdGeom.Tokens.guide)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        created_paths.append(path)

    def sphere(path: str, center: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        item = UsdGeom.Sphere.Define(stage, path)
        item.CreateRadiusAttr(radius)
        item.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
        item.CreatePurposeAttr(UsdGeom.Tokens.guide)
        item.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        created_paths.append(path)

    def frame_triad(path: str, origin: np.ndarray, rotation: np.ndarray, length: float = 0.026) -> None:
        colors = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.2), (0.15, 0.35, 1.0))
        for axis_index, (axis_name, color) in enumerate(zip(("x", "y", "z"), colors, strict=True)):
            endpoint = origin + rotation[:, axis_index] * length
            line(f"{path}/{axis_name}", [origin, endpoint], color, 0.0022)

    with Usd.EditContext(stage, session):
        root = UsdGeom.Xform.Define(stage, GUIDE_ROOT)
        root.CreatePurposeAttr(UsdGeom.Tokens.guide)
        for body in ("link5", "gripper_link"):
            pos, quat = d334._body_pose_w(inner, body)
            rot = _quat_to_rot(quat)
            for index, part in enumerate(topology_parts[body]):
                path = f"{GUIDE_ROOT}/live_callback_parts/{body}/part_{index:03d}"
                mesh_at(path, _world_vertices(part["_vertices"], pos, rot), part["_triangles"], _palette(body, index))
        fixed_world = _world_vertices(display["vertices"], display["link_pos"], display["link_rot"])
        mesh_at(f"{GUIDE_ROOT}/fixed_jaw/raw_connected_component", fixed_world, display["triangles"], [50, 255, 105, 105])
        cyl_vertices, cyl_triangles = _cylinder_mesh()
        mesh_at(f"{GUIDE_ROOT}/target/cylinder_collider", cyl_vertices + display["object_pos"], cyl_triangles, [255, 190, 20, 105])
        base, center = display["base"], display["object_pos"]
        line(f"{GUIDE_ROOT}/axes/base_to_cylinder_radial", [base, center], (1.0, 0.1, 0.1), 0.0025)
        line(f"{GUIDE_ROOT}/axes/cylinder_tangent", [center - display["tangent"] * 0.06, center + display["tangent"] * 0.06], (0.1, 1.0, 0.2), 0.0022)
        line(f"{GUIDE_ROOT}/axes/fixed_jaw_centerline", [display["centerline_station"] - display["axis_world"] * 0.10, display["centerline_station"] + display["axis_world"] * 0.10], (0.1, 0.9, 1.0), 0.0030)
        line(f"{GUIDE_ROOT}/axes/fixed_surface_normal", [display["current_witness"], display["current_witness"] + display["normal_world"] * 0.045], (1.0, 0.1, 1.0), 0.0025)
        line(f"{GUIDE_ROOT}/axes/raw_gap_witness", [display["current_witness"], display["cylinder_witness"]], (1.0, 1.0, 0.1), 0.0030)
        sphere(f"{GUIDE_ROOT}/points/cylinder_center", center, 0.0040, (1.0, 0.85, 0.0))
        sphere(f"{GUIDE_ROOT}/points/actual_raw_witness", display["current_witness"], 0.0032, (1.0, 0.0, 1.0))
        sphere(f"{GUIDE_ROOT}/points/cylinder_witness", display["cylinder_witness"], 0.0032, (1.0, 1.0, 0.0))
        sphere(f"{GUIDE_ROOT}/points/legacy_proxy", display["legacy_proxy"], 0.0030, (0.9, 0.1, 0.1))
        sphere(f"{GUIDE_ROOT}/points/tcp", display["tcp_world"], 0.0030, (0.2, 0.8, 1.0))
        frame_triad(f"{GUIDE_ROOT}/frames/link5", display["link_pos"], display["link_rot"])
        frame_triad(f"{GUIDE_ROOT}/frames/tcp", display["tcp_world"], display["link_rot"])
        frame_triad(f"{GUIDE_ROOT}/frames/object", display["object_pos"], np.eye(3, dtype=np.float64))
        frame_triad(
            f"{GUIDE_ROOT}/frames/fixed_jaw_axis",
            display["centerline_station"],
            _frame_rotation_from_axis(display["axis_world"], display["normal_world"]),
        )

    edit_target_after = str(stage.GetEditTarget().GetLayer().identifier)
    all_guide_paths = [GUIDE_ROOT, *created_paths]
    session_contains_all = all(session.GetPrimAtPath(path) is not None for path in all_guide_paths)
    root_contains_none = all(root_layer.GetPrimAtPath(path) is None for path in all_guide_paths)
    no_physics = {}
    for path in created_paths:
        prim = stage.GetPrimAtPath(path)
        no_physics[path] = bool(
            prim.IsValid()
            and not prim.HasAPI(UsdPhysics.CollisionAPI)
            and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not prim.HasAPI(UsdPhysics.MassAPI)
        )
    return {
        "artifact": "D350_VIEWER_OVERLAY_CONTRACT_V1",
        "guide_root": GUIDE_ROOT,
        "edit_target_before": edit_target_before,
        "edit_target_after": edit_target_after,
        "session_layer_identifier": str(session.identifier),
        "created_prim_count": len(created_paths),
        "live_part_counts": {body: len(topology_parts[body]) for body in ("link5", "gripper_link")},
        "actual_physx_collider_role": "separate debug-render capture",
        "colored_mesh_role": "hash-bound callback-topology display copy; no physics authority",
        "all_display_prims_without_physics_api": all(no_physics.values()),
        "session_layer_contains_all_guide_prims": session_contains_all,
        "root_layer_contains_no_guide_prims": root_contains_none,
        "edit_target_restored": edit_target_after == edit_target_before,
        "failed_display_prim_paths": sorted(path for path, passed in no_physics.items() if not passed),
        "root_layer_saved_or_exported": False,
        "asset_write": False,
        "pass": bool(
            len(topology_parts["link5"]) == 64
            and len(topology_parts["gripper_link"]) == 64
            and all(no_physics.values())
            and session_contains_all
            and root_contains_none
            and edit_target_after == edit_target_before
        ),
    }


def _frame_rotation_from_axis(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(axis, dtype=np.float64)
    z_axis /= float(np.linalg.norm(z_axis))
    x_axis = np.asarray(normal, dtype=np.float64) - z_axis * float(np.dot(normal, z_axis))
    if float(np.linalg.norm(x_axis)) <= 1.0e-9:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis -= z_axis * float(np.dot(x_axis, z_axis))
    x_axis /= float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= float(np.linalg.norm(y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _rerun_rows(inner: Any, topology_parts: dict[str, list[dict[str, Any]]], measurement: dict[str, Any], display: dict[str, Any]) -> tuple[list[Any], ...]:
    coordinate_frames = []
    for body in ("link5", "gripper_link"):
        pos, quat = d334._body_pose_w(inner, body)
        coordinate_frames.append(
            {
                "frame": f"tf#/{body}",
                "parent_frame": "tf#/",
                "entity_path": f"coordinate_frames/{body}/body_local",
                "translation_m": pos.tolist(),
                "quaternion_xyzw": [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])],
            }
        )
    coordinate_frames.append(
        {
            "frame": "tf#/object",
            "parent_frame": "tf#/",
            "entity_path": "coordinate_frames/object/body_local",
            "translation_m": display["object_pos"].tolist(),
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    )
    meshes = []
    part_idx = 0
    for body in ("link5", "gripper_link"):
        for index, part in enumerate(topology_parts[body]):
            meshes.append(
                {
                    "entity_path": f"geometry/live_parts/{body}/part_{index:03d}",
                    "vertices_m": part["_vertices"],
                    "triangles": part["_triangles"],
                    "coordinate_frame": f"tf#/{body}",
                    "color_rgba": _palette(body, index),
                    "sequence": {"part_idx": part_idx},
                    "static": True,
                    "scientific_authority": False,
                    "source_path": part["path"],
                }
            )
            part_idx += 1
    meshes.append(
        {
            "entity_path": "geometry/fixed_jaw/raw_connected_component",
            "vertices_m": display["vertices"],
            "triangles": display["triangles"],
            "coordinate_frame": "tf#/link5",
            "color_rgba": [50, 255, 105, 125],
            "sequence": {"part_idx": part_idx},
            "static": True,
            "component_digest": measurement["fixed_jaw_component"]["component_digest"],
        }
    )
    part_idx += 1
    cyl_vertices, cyl_triangles = _cylinder_mesh()
    meshes.append(
        {
            "entity_path": "geometry/target/cylinder_collider",
            "vertices_m": cyl_vertices,
            "triangles": cyl_triangles,
            "coordinate_frame": "tf#/object",
            "color_rgba": [255, 190, 20, 100],
            "sequence": {"part_idx": part_idx},
            "static": True,
            "analytic_source": "Cylinder(radius=0.017m,height=0.090m,Z axis)",
        }
    )
    link_pos, link_quat = d334._body_pose_w(inner, "link5")
    frames = [
        {"name": "base", "label": "robot base", "position": display["base"].tolist(), "rotation_matrix": np.eye(3).tolist(), "role": "actual"},
        {"name": "cylinder", "label": "target cylinder", "position": display["object_pos"].tolist(), "rotation_matrix": np.eye(3).tolist(), "role": "object"},
        {"name": "link5_actual", "label": "actual link5", "position": link_pos.tolist(), "quat_wxyz": link_quat.tolist(), "role": "link5"},
        {"name": "fixed_jaw_axis", "label": "actual fixed-jaw PCA axis", "position": display["centerline_station"].tolist(), "rotation_matrix": _frame_rotation_from_axis(display["axis_world"], display["normal_world"]).tolist(), "role": "fixed_jaw"},
    ]
    points = [
        {
            "entity_path": "geometry/key_points",
            "positions_m": [
                display["base"].tolist(),
                display["object_pos"].tolist(),
                display["current_witness"].tolist(),
                display["cylinder_witness"].tolist(),
                display["legacy_proxy"].tolist(),
                display["tcp_world"].tolist(),
            ],
            "coordinate_frame": "tf#/",
            "radii": [0.004] * 6,
            "colors": [[220, 220, 220], [255, 190, 20], [255, 30, 220], [255, 245, 30], [240, 40, 40], [40, 205, 255]],
            "labels": ["base", "cylinder center", "actual fixed-jaw witness", "cylinder witness", "legacy 8mm proxy", "TCP"],
            "static": True,
        }
    ]
    arrows = [
        {"entity_path": "geometry/arrows/base_to_cylinder_radial", "origins_m": [display["base"].tolist()], "vectors_m": [(display["object_pos"] - display["base"]).tolist()], "coordinate_frame": "tf#/", "radii": [0.002], "colors": [[255, 40, 40]], "labels": ["base to cylinder radial"], "static": True},
        {"entity_path": "geometry/arrows/cylinder_tangent", "origins_m": [display["object_pos"].tolist()], "vectors_m": [(display["tangent"] * 0.08).tolist()], "coordinate_frame": "tf#/", "radii": [0.002], "colors": [[40, 255, 80]], "labels": ["adopted tangent sign -1"], "static": True},
        {"entity_path": "geometry/arrows/fixed_jaw_centerline", "origins_m": [display["centerline_station"].tolist()], "vectors_m": [(display["axis_world"] * 0.10).tolist()], "coordinate_frame": "tf#/", "radii": [0.0025], "colors": [[30, 220, 255]], "labels": ["actual fixed-jaw PCA axis"], "static": True},
        {"entity_path": "geometry/arrows/fixed_surface_normal", "origins_m": [display["current_witness"].tolist()], "vectors_m": [(display["normal_world"] * 0.045).tolist()], "coordinate_frame": "tf#/", "radii": [0.002], "colors": [[255, 40, 230]], "labels": ["actual nearest triangle normal"], "static": True},
        {"entity_path": "geometry/arrows/raw_gap_witness", "origins_m": [display["current_witness"].tolist()], "vectors_m": [(display["cylinder_witness"] - display["current_witness"]).tolist()], "coordinate_frame": "tf#/", "radii": [0.0025], "colors": [[255, 245, 30]], "labels": ["raw link5-cylinder separation"], "static": True},
    ]
    d = measurement["distances"]
    metric_values = {
        "fixed_axis_radial_angle_deg": measurement["metrics"]["fixed_axis_radial_angle_deg"],
        "fixed_axis_to_link5_plus_z_deg": measurement["metrics"][
            "fixed_axis_to_link5_plus_z_deg"
        ],
        "fixed_axis_pitch_deg": measurement["metrics"]["fixed_axis_pitch_deg"],
        "centerline_height_error_mm": measurement["metrics"]["centerline_height_error_mm"],
        "centerline_tangent_offset_mm": measurement["metrics"]["centerline_tangent_offset_mm"],
        "surface_normal_to_center_angle_deg": measurement["metrics"]["surface_normal_to_center_angle_deg"],
        "surface_normal_to_gap_angle_deg": measurement["metrics"]["surface_normal_to_gap_angle_deg"],
        "legacy_proxy_to_actual_witness_mm": measurement["metrics"]["legacy_proxy_to_actual_witness_mm"],
        "link5_raw_clearance_mm": d["link5"]["raw_signed_distance_mm"],
        "link5_live_clearance_mm": d["link5"]["live_signed_distance_mm"],
        "gripper_raw_clearance_mm": d["gripper_link"]["raw_signed_distance_mm"],
        "gripper_live_clearance_mm": d["gripper_link"]["live_signed_distance_mm"],
        "simulation_counter": float(inner._sim_step_counter),
    }
    scalars = [
        {"entity_path": f"metrics/d350/{name}", "value": float(metric_values[name]), "sequence": {"measurement_idx": index}, "static": False}
        for index, name in enumerate(METRIC_NAMES)
    ]
    events_text = {
        "scope": "D350 zero-step | D349 target exact | G0a=false | settle forbidden",
        "binding": f"fixed-jaw raw connected component digest={measurement['fixed_jaw_component']['component_digest'][:16]}",
        "geometry": f"axis-radial={measurement['metrics']['fixed_axis_radial_angle_deg']:.6f}deg height={measurement['metrics']['centerline_height_error_mm']:.6f}mm MEASURED",
        "viewer": "Isaac Viewer is primary visual; Rerun is replay observability only",
        "gate": "No centerline/height ALIGNED_PASS threshold exists in D350",
    }
    events = [
        {"entity_path": f"events/d350/{name}", "text": events_text[name], "level": "INFO", "sequence": {"event_idx": index}, "static": False}
        for index, name in enumerate(EVENT_NAMES)
    ]
    return frames, coordinate_frames, meshes, points, arrows, scalars, events


def _run_rerun(inner: Any, topology_parts: dict[str, list[dict[str, Any]]], measurement: dict[str, Any], display: dict[str, Any]) -> dict[str, Any]:
    rows = _rerun_rows(inner, topology_parts, measurement, display)
    frames, coordinate_frames, meshes, points, arrows, scalars, events = rows
    log_status = log_rerun(
        RRD_PATH,
        frames=frames,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        points=points,
        arrows=arrows,
        scalar_trace=scalars,
        events=events,
        recording_metadata={
            "case": CASE,
            "purpose": "assembled fixed-jaw geometry and 64+64 collider observability",
            "git_head": _git_head(),
            "new_variables": NEW_VARIABLES,
            "target": "exact D349 Float32 q/object; q5 OPEN 1.5413; r/t 7/11mm",
            "physics": "forbidden; controlled steps 0",
            "scientific_authority": "Float64 measurement JSON, D348 callback topology, hashes",
            "viewer_role": "Float32 one-way display copy",
        },
        recording_id="g0a_d350_fixed_jaw_geometry",
        blueprint_path=RBL_PATH,
        blueprint_mode="robot_geometry",
        live_viewer=False,
        app_id="roarm_g0a_fixed_jaw_geometry",
    )
    entities, components = _expected_rrd_contract()
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "geometry/live_parts/link5/part_000",
                "geometry/live_parts/gripper_link/part_063",
                "geometry/fixed_jaw/raw_connected_component",
                "geometry/target/cylinder_collider",
                "geometry/arrows/fixed_jaw_centerline",
            ],
            expected_timeline_names=["part_idx", "measurement_idx", "event_idx"],
            exact_entity_paths=entities,
            exact_timeline_names=["blueprint", "event_idx", "log_time", "measurement_idx", "part_idx"],
            expected_entity_components=components,
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_SCREENSHOT_PATH,
        )
        if log_status.get("ok")
        else {"pass": False, "errors": ["Rerun logging/finalization failed"]}
    )
    observed = {
        "frame_count": len(frames),
        "coordinate_frame_count": len(coordinate_frames),
        "mesh_count": len(meshes),
        "point_entity_count": len(points),
        "arrow_entity_count": len(arrows),
        "scalar_row_count": len(scalars),
        "event_row_count": len(events),
        "exact_non_system_entity_count": len(validation.get("entity_path_contract", {}).get("observed_non_system", [])),
    }
    count_checks = {name: observed.get(name) == value for name, value in EXPECTED_RERUN_COUNTS.items()}
    report = {
        "artifact": "D350_RERUN_MACHINE_VALIDATION_V1",
        "profile": "MEASURED_AUTHORITY",
        "log_status": log_status,
        "archive_validation": validation,
        "observed_counts": observed,
        "expected_counts": EXPECTED_RERUN_COUNTS,
        "count_checks": count_checks,
        "contract_sha256": _rrd_digest(),
        "pass": bool(log_status.get("ok") and validation.get("pass") and all(count_checks.values())),
    }
    _write_json(RERUN_VALIDATION_PATH, report)
    return report


def _pump_once(simulation_app: Any, inner: Any, timeline: Any) -> bool:
    inner.sim.set_setting("/app/player/playSimulations", False)
    intervened = False
    if timeline.is_playing():
        timeline.pause()
        intervened = True
    simulation_app.update()
    inner.sim.set_setting("/app/player/playSimulations", False)
    if timeline.is_playing():
        timeline.pause()
        intervened = True
    return intervened


def _pump_frames(simulation_app: Any, inner: Any, timeline: Any, count: int) -> int:
    return sum(int(_pump_once(simulation_app, inner, timeline)) for _ in range(count))


def _capture_viewport(path: Path, simulation_app: Any, inner: Any, timeline: Any) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utility

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        return {"ok": False, "error": "no active viewport"}
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(
        capture.wait_for_result(completion_frames=5), run_until_complete=False
    )
    deadline = time.monotonic() + 30.0
    interventions = 0
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        interventions += int(_pump_once(simulation_app, inner, timeline))
    if not task.done():
        task.cancel()
        return {"ok": False, "error": "viewport capture timeout", "timeline_interventions": interventions}
    result = task.result()
    _pump_frames(simulation_app, inner, timeline, 2)
    ok = bool(result and path.is_file() and path.stat().st_size > 0 and _png_dimensions(path))
    return {
        "ok": ok,
        "capture_result": bool(result),
        "path": _rel(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha(path) if path.is_file() else None,
        "dimensions": _png_dimensions(path),
        "timeline_interventions": interventions,
    }


def _set_guide_visibility(inner: Any, visible: bool) -> None:
    from pxr import Usd, UsdGeom

    stage = inner.scene.stage
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(GUIDE_ROOT))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _viewer_guide_layer_guard(inner: Any) -> dict[str, Any]:
    stage = inner.scene.stage
    session = stage.GetSessionLayer()
    root = stage.GetRootLayer()
    checks = {
        "guide_root_still_in_session": session.GetPrimAtPath(GUIDE_ROOT) is not None,
        "guide_root_absent_from_root": root.GetPrimAtPath(GUIDE_ROOT) is None,
        "edit_target_not_session": stage.GetEditTarget().GetLayer() != session,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _set_persistent_setting_exact(settings: Any, path: str, value: Any) -> None:
    if value is None:
        settings.destroy_item(path)
    else:
        settings.set(path, value)


def _run_viewer_captures(simulation_app: Any, inner: Any, timeline: Any) -> dict[str, Any]:
    import carb

    settings = carb.settings.get_settings()
    previous_physx = settings.get(PHYSX_COLLIDER_SETTING)
    previous_guide = settings.get(GUIDE_PURPOSE_SETTING)
    state_before = d334._snapshot_sim_state(inner)
    time_before = float(timeline.get_current_time())
    counter_before = int(inner._sim_step_counter)
    interventions = 0
    captures: dict[str, Any] = {}
    capture_sequence_completed = False
    cameras = {
        "whole_oblique_physx": ([0.68, -0.62, 0.52], [0.18, 0.0, 0.08]),
        "tool_oblique_physx": ([0.49, -0.32, 0.28], [0.285, 0.0, 0.055]),
        "whole_oblique_colored": ([0.68, -0.62, 0.52], [0.18, 0.0, 0.08]),
        "tool_top": ([0.285, 0.0, 0.55], [0.285, 0.0, 0.055]),
        "tool_side": ([0.285, -0.42, 0.09], [0.285, 0.0, 0.055]),
        "tool_oblique": ([0.49, -0.32, 0.28], [0.285, 0.0, 0.055]),
    }
    try:
        settings.set(GUIDE_PURPOSE_SETTING, True)
        _set_guide_visibility(inner, False)
        settings.set(PHYSX_COLLIDER_SETTING, 2)
        for name in ("whole_oblique_physx", "tool_oblique_physx"):
            eye, target = cameras[name]
            inner.sim.set_camera_view(eye, target)
            interventions += _pump_frames(simulation_app, inner, timeline, 12)
            captures[name] = _capture_viewport(
                VIEWER_PNGS[name], simulation_app, inner, timeline
            )

        settings.set(PHYSX_COLLIDER_SETTING, 0)
        _set_guide_visibility(inner, True)
        for name in ("whole_oblique_colored", "tool_top", "tool_side", "tool_oblique"):
            eye, target = cameras[name]
            inner.sim.set_camera_view(eye, target)
            interventions += _pump_frames(simulation_app, inner, timeline, 10)
            captures[name] = _capture_viewport(VIEWER_PNGS[name], simulation_app, inner, timeline)
        eye, target = cameras["tool_oblique"]
        inner.sim.set_camera_view(eye, target)
        interventions += _pump_frames(simulation_app, inner, timeline, 6)
        capture_sequence_completed = True
    finally:
        if capture_sequence_completed:
            settings.set(PHYSX_COLLIDER_SETTING, 0)
            settings.set(GUIDE_PURPOSE_SETTING, True)
            _set_guide_visibility(inner, True)
        else:
            _set_persistent_setting_exact(
                settings, PHYSX_COLLIDER_SETTING, previous_physx
            )
            _set_persistent_setting_exact(
                settings, GUIDE_PURPOSE_SETTING, previous_guide
            )
            _set_guide_visibility(inner, True)
    state_after = d334._snapshot_sim_state(inner)
    guard = d334._state_guard(state_before, state_after)
    counter_after = int(inner._sim_step_counter)
    time_after = float(timeline.get_current_time())
    checks = {
        "all_six_captures": len(captures) == 6
        and all(row.get("ok") for row in captures.values()),
        "counter_zero_unchanged": counter_before == counter_after == 0,
        "timeline_time_unchanged": time_before == time_after,
        "timeline_paused": not timeline.is_playing(),
        "state_guard": not guard["violated"],
        "guide_visible_for_interactive_view": True,
        "physx_debug_disabled_after_capture": settings.get(PHYSX_COLLIDER_SETTING) == 0,
    }
    return {
        "artifact": "D350_VIEWER_CAPTURE_CONTRACT_V1",
        "cameras": {name: {"eye": eye, "target": target} for name, (eye, target) in cameras.items()},
        "captures": captures,
        "counter": {"before": counter_before, "after": counter_after, "delta": counter_after - counter_before},
        "timeline": {"time_before": time_before, "time_after": time_after, "interventions": interventions, "playing_after": timeline.is_playing()},
        "state_guard": guard,
        "settings": {"physx_previous": previous_physx, "guide_previous": previous_guide, "physx_final_for_interactive": 0, "guide_final_for_interactive": True},
        "checks": checks,
        "pass": all(checks.values()),
    }


def _restore_viewer_settings(inner: Any, capture: dict[str, Any]) -> dict[str, Any]:
    import carb

    settings = carb.settings.get_settings()
    before = capture["settings"]

    _set_persistent_setting_exact(
        settings, PHYSX_COLLIDER_SETTING, before["physx_previous"]
    )
    _set_persistent_setting_exact(
        settings, GUIDE_PURPOSE_SETTING, before["guide_previous"]
    )
    _set_guide_visibility(inner, True)
    observed_physx = settings.get(PHYSX_COLLIDER_SETTING)
    observed_guide = settings.get(GUIDE_PURPOSE_SETTING)
    checks = {
        "physx_persistent_setting_restored": observed_physx
        == before["physx_previous"],
        "guide_persistent_setting_restored": observed_guide
        == before["guide_previous"],
    }
    return {
        "observed_physx": observed_physx,
        "observed_guide": observed_guide,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _hold_viewer(simulation_app: Any, inner: Any, timeline: Any, seconds: float) -> dict[str, Any]:
    state_before = d334._snapshot_sim_state(inner)
    counter_before = int(inner._sim_step_counter)
    time_before = float(timeline.get_current_time())
    start = time.monotonic()
    interventions = 0
    updates = 0
    drift = None
    print(
        f"D350_VIEWER_READY hold_seconds={seconds:.1f} inspect the live Viewer; "
        "it will close automatically",
        flush=True,
    )
    while simulation_app.is_running() and time.monotonic() - start < seconds:
        interventions += int(_pump_once(simulation_app, inner, timeline))
        updates += 1
        if updates % 30 == 0:
            guard = d334._state_guard(state_before, d334._snapshot_sim_state(inner))
            if guard["violated"] or int(inner._sim_step_counter) != 0:
                drift = guard
                break
        time.sleep(0.01)
    final_guard = d334._state_guard(state_before, d334._snapshot_sim_state(inner))
    counter_after = int(inner._sim_step_counter)
    time_after = float(timeline.get_current_time())
    return {
        "requested_seconds": seconds,
        "elapsed_seconds": time.monotonic() - start,
        "update_count": updates,
        "timeline_interventions": interventions,
        "viewer_closed_by_user": not simulation_app.is_running(),
        "drift_detected": drift,
        "counter_before": counter_before,
        "counter_after": counter_after,
        "timeline_time_before": time_before,
        "timeline_time_after": time_after,
        "state_guard": final_guard,
        "pass": bool(
            drift is None
            and not final_guard["violated"]
            and counter_before == counter_after == 0
            and time_before == time_after
        ),
    }


def _write_automated_report(summary: dict[str, Any]) -> None:
    metrics = summary["measurement_metrics"]
    text = (
        "# D350 automated report\n\n"
        f"- scientific verdict: `{summary['scientific_verdict']}`\n"
        f"- observability verdict: `{summary['observability_verdict']}`\n"
        f"- automated pass: `{summary['automated_pass']}`\n"
        f"- fixed axis vs radial: `{metrics.get('fixed_axis_radial_angle_deg')}` deg\n"
        f"- fixed axis vs link5 +z: `{metrics.get('fixed_axis_to_link5_plus_z_deg')}` deg\n"
        f"- fixed axis pitch: `{metrics.get('fixed_axis_pitch_deg')}` deg\n"
        f"- centerline height error: `{metrics.get('centerline_height_error_mm')}` mm\n"
        f"- legacy proxy to actual witness: `{metrics.get('legacy_proxy_to_actual_witness_mm')}` mm\n"
        f"- physics steps: `{summary['controlled_physics_steps']}`\n"
        "- `aligned_pass=null`; `g0a_pass=false`\n"
    )
    _write_text(AUTOMATED_MD_PATH, text)


def _run_validate(args: argparse.Namespace, simulation_app: Any, launcher_report: dict[str, Any]) -> int:
    import omni.timeline

    prereg = _json(PREREG_PATH)
    source_before = d349._source_inventories()
    input_before = _input_hashes()
    external_before = _external_baseline()
    args.robot_usd_path = VARIANT_ROBOT_USD
    inner = d333._make_runtime_env(args)
    timeline = omni.timeline.get_timeline_interface()
    try:
        inner.reset(seed=SEED)
        inner.sim.set_setting("/app/player/playSimulations", False)
        if timeline.is_playing():
            timeline.pause()
        counters = [{"phase": "after_reset", "counter": int(inner._sim_step_counter)}]
        d332._write_exact_state(inner, Q_TARGET_F32.astype(np.float64), OBJECT_POS_F32.astype(np.float64))
        counters.append({"phase": "after_exact_target_write", "counter": int(inner._sim_step_counter)})
        target_guard = _target_guard(inner)
        if not target_guard["pass"]:
            raise RuntimeError(f"D350 exact target guard failed: {target_guard}")
        corrected = d349._corrected_live_audit()
        topology_parts, live_binding = d349._build_live_topology_parts(inner)
        _write_json(LIVE_BINDING_PATH, live_binding)
        counters.append({"phase": "after_live_topology_binding", "counter": int(inner._sim_step_counter)})
        raw_shapes, raw_contract = d339._build_retained_raw_shapes(inner, _json(D334_SUMMARY))
        if not (corrected["pass"] and live_binding["pass"] and raw_contract["pass"]):
            raise RuntimeError("D350 immutable raw/live prerequisites failed")
        raw_parts = {body: [] for body in ("link5", "gripper_link")}
        for shape in raw_shapes:
            raw_parts[shape["body"]].append({"body": shape["body"], "name": "retained_raw_mesh", "path": shape["collider_path"], "_geometry": shape["_geom_raw"]})
        raw_set = d349._union_distances(inner, raw_parts, "_geometry", "d350_retained_raw_triangle_bvh")
        live_set = d349._union_distances(inner, topology_parts, "_geometry_topology_surface_authority", "d350_live_callback_topology_surface")
        binding, measurement, display = _bind_and_measure(inner, raw_shapes, raw_set, live_set)
        _write_json(BINDING_PATH, binding)
        if not binding["pass"]:
            raise RuntimeError(VERDICT_BINDING)
        _write_json(MEASUREMENT_PATH, {
            **measurement,
            "target_guard": target_guard,
            "execution_order_through_measurement": counters,
            "live_binding_sha256": _sha(LIVE_BINDING_PATH),
            "scope_guards": _json(PARAMETER_PATH)["scope_guards"],
        })
        counters.append({"phase": "after_fixed_jaw_geometry_measurement", "counter": int(inner._sim_step_counter)})
        if not measurement.get("frozen_input_reproduction_pass", False):
            raise RuntimeError(VERDICT_INPUT)
        if not measurement["pass"]:
            raise RuntimeError(VERDICT_GEOMETRY)
        overlay = _create_viewer_guides(inner, topology_parts, display)
        _write_json(OVERLAY_PATH, overlay)
        counters.append({"phase": "after_viewer_overlay_creation", "counter": int(inner._sim_step_counter)})
        rerun = _run_rerun(inner, topology_parts, measurement, display)
        counters.append({"phase": "after_rerun", "counter": int(inner._sim_step_counter)})
        capture = _run_viewer_captures(simulation_app, inner, timeline)
        counters.append({"phase": "after_viewer_capture", "counter": int(inner._sim_step_counter)})
        try:
            hold = _hold_viewer(
                simulation_app, inner, timeline, float(args.viewer_hold_seconds)
            )
            final_target_guard = _target_guard(inner)
        finally:
            setting_restore = _restore_viewer_settings(inner, capture)
        layer_guard_after_capture = _viewer_guide_layer_guard(inner)
        capture["interactive_hold"] = hold
        capture["target_guard_after_interactive"] = final_target_guard
        capture["persistent_setting_restore"] = setting_restore
        capture["guide_layer_guard_after_capture"] = layer_guard_after_capture
        capture["checks"]["interactive_hold_zero_step"] = hold["pass"]
        capture["checks"]["target_bits_exact_after_interactive"] = final_target_guard[
            "pass"
        ]
        capture["checks"]["persistent_settings_restored"] = setting_restore["pass"]
        capture["checks"]["session_only_guides_after_capture"] = (
            layer_guard_after_capture["pass"]
        )
        capture["pass"] = bool(
            capture["pass"]
            and hold["pass"]
            and final_target_guard["pass"]
            and setting_restore["pass"]
            and layer_guard_after_capture["pass"]
        )
        _write_json(CAPTURE_PATH, capture)
        counters.append({"phase": "final", "counter": int(inner._sim_step_counter)})
        all_counters_zero = all(row["counter"] == 0 for row in counters)
        source_after = d349._source_inventories()
        input_after = _input_hashes()
        external_after = _external_baseline()
        immutability = {
            "source_inventories_exact": source_before == source_after == prereg["source_inventories"],
            "input_hashes_exact": input_before == input_after == EXPECTED_INPUT_HASHES,
            "external_user_files_exact": external_before == external_after and external_after["pass"],
            "asset_write": False,
        }
        immutability["pass"] = all(immutability.values())
        if not final_target_guard["pass"] or not measurement.get(
            "frozen_input_reproduction_pass", False
        ):
            scientific = VERDICT_INPUT
        elif not measurement["pass"]:
            scientific = VERDICT_GEOMETRY
        else:
            scientific = VERDICT_PENDING
        visual_machine = bool(overlay["pass"] and rerun["pass"] and capture["pass"])
        observability = "D350_VIEWER_AND_RERUN_MACHINE_PASS_MANUAL_PENDING" if visual_machine else VERDICT_VISUAL
        summary = {
            "artifact": "D350_AUTOMATED_SUMMARY_V1",
            "case": CASE,
            "scientific_verdict": scientific,
            "observability_verdict": observability,
            "automated_pass": bool(
                scientific == VERDICT_PENDING
                and visual_machine
                and all_counters_zero
                and final_target_guard["pass"]
                and launcher_report["pass"]
                and immutability["pass"]
            ),
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": NEW_PHYSICAL_VARIABLES,
            "launcher": launcher_report,
            "target_guard": target_guard,
            "target_guard_after_interactive": final_target_guard,
            "semantic_binding_pass": binding["pass"],
            "measurement_pass": measurement["pass"],
            "measurement_metrics": measurement.get("metrics", {}),
            "aligned_pass": None,
            "overlay_pass": overlay["pass"],
            "rerun_pass": rerun["pass"],
            "viewer_capture_pass": capture["pass"],
            "execution_order": counters,
            "controlled_physics_steps": 0 if all_counters_zero else None,
            "immutability": immutability,
            "evidence_hashes": _automated_evidence_hashes(),
            "manual_visual_inspection_pending": True,
            "g0a_pass": False,
            "settle_executed": False,
            "ten_trial_run": False,
            "g0b_run": False,
            "rl_run": False,
            "ladder_promoted": False,
        }
        _write_json(AUTOMATED_PATH, summary)
        _write_automated_report(summary)
        print(json.dumps({"stage": "validate", "automated_pass": summary["automated_pass"], "scientific_verdict": scientific}, sort_keys=True), flush=True)
        return 0 if summary["automated_pass"] else 2
    finally:
        inner.close()


def _manual_checks(manual: dict[str, Any]) -> dict[str, bool]:
    images = manual.get("images", {})
    expected_images = {**VIEWER_PNGS, "rerun": RERUN_SCREENSHOT_PATH}
    image_checks: dict[str, bool] = {}
    for name, path in expected_images.items():
        row = images.get(name, {})
        image_checks[f"{name}_path"] = row.get("path") == _rel(path)
        image_checks[f"{name}_sha"] = bool(
            path.is_file() and row.get("sha256") == _sha(path)
        )
        image_checks[f"{name}_bytes"] = bool(
            path.is_file() and row.get("bytes") == path.stat().st_size
        )
        image_checks[f"{name}_dimensions"] = bool(
            row.get("raster_dimensions") == _png_dimensions(path)
        )
    observations = manual.get("observations", {})
    required_observations = (
        "isaac_viewer_opened_interactively",
        "whole_oblique_physx_shows_actual_collider_debug",
        "tool_oblique_physx_closeup_shows_jaw_and_target_colliders",
        "whole_oblique_colored_shows_64_plus_64",
        "tool_top_colored_geometry_nonempty",
        "tool_side_colored_geometry_nonempty",
        "tool_oblique_colored_geometry_nonempty",
        "fixed_jaw_component_visible",
        "target_cylinder_visible",
        "radial_tangent_centerline_and_witnesses_visible",
        "rerun_128_parts_visible_and_nonempty",
        "rerun_frames_arrows_and_metrics_visible",
        "required_content_not_obscured",
        "all_seven_pngs_opened_at_original_resolution",
    )
    method = str(manual.get("inspection_method", "")).lower()
    checks = {
        "artifact": manual.get("artifact")
        == "D350_VIEWER_RERUN_MANUAL_VISUAL_INSPECTION_V1",
        "case": manual.get("case") == CASE,
        "inspection_date": manual.get("inspection_date") == "2026-07-14",
        "inspection_method": "view_image" in method and "original" in method,
        "all_image_records_exact": len(images) == len(expected_images),
        "observations": all(
            observations.get(name) is True for name in required_observations
        ),
        "bounded_interpretation": len(manual.get("bounded_interpretation", [])) >= 3,
        "manual_pass": manual.get("manual_visual_inspection_pass") is True,
        "scientific_override_false": manual.get("scientific_verdict_override") is False,
        "g0a_false": manual.get("g0a_pass") is False,
        "settle_false": manual.get("settle_executed") is False,
        "markdown_nonzero": MANUAL_MD_PATH.is_file()
        and MANUAL_MD_PATH.stat().st_size > 0,
        **image_checks,
    }
    return checks


def _run_finalize(_args: argparse.Namespace) -> int:
    if COMPLETION_PATH.exists() or COMPLETION_MD_PATH.exists():
        raise RuntimeError("D350 completion already exists")
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    parameter = _json(PARAMETER_PATH)
    preflight = _json(PREFLIGHT_PATH)
    rerun_validation = _json(RERUN_VALIDATION_PATH)
    overlay = _json(OVERLAY_PATH)
    capture = _json(CAPTURE_PATH)
    manual_checks = _manual_checks(manual)
    required = [
        PARAMETER_PATH,
        PREREG_PATH,
        PREFLIGHT_PATH,
        BINDING_PATH,
        MEASUREMENT_PATH,
        LIVE_BINDING_PATH,
        OVERLAY_PATH,
        CAPTURE_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_SCREENSHOT_PATH,
        RERUN_VALIDATION_PATH,
        AUTOMATED_PATH,
        AUTOMATED_MD_PATH,
        MANUAL_PATH,
        MANUAL_MD_PATH,
        *VIEWER_PNGS.values(),
    ]
    artifact_rows = {
        _rel(path): {"bytes": path.stat().st_size, "sha256": _sha(path), "png_dimensions": _png_dimensions(path)}
        for path in required
        if path.is_file()
    }
    checks = {
        "head_exact": _git_head() == prereg.get("git_head") == EXPECTED_HEAD,
        "parameter_pass": parameter.get("pass") is True,
        "preregistration_pass": prereg.get("pass") is True,
        "preflight_pass": preflight.get("pass") is True,
        "state_hashes_exact": prereg.get("state_hashes")
        == {"start_here": _sha(START_HERE), "session_doc": _sha(SESSION_DOC)},
        "harness_hash_exact": prereg.get("harness_sha256") == _sha(HARNESS),
        "parameter_hash_exact": prereg.get("parameter_sha256") == _sha(PARAMETER_PATH),
        "input_hashes_exact": prereg.get("input_hashes")
        == _input_hashes()
        == EXPECTED_INPUT_HASHES,
        "helper_hashes_exact": prereg.get("helper_hashes") == _helper_hashes(),
        "source_inventories_exact": prereg.get("source_inventories")
        == d349._source_inventories(),
        "automated_evidence_hashes_exact": automated.get("evidence_hashes")
        == _automated_evidence_hashes(),
        "rerun_machine_contract": rerun_validation.get("pass") is True,
        "viewer_overlay_contract": overlay.get("pass") is True,
        "viewer_capture_contract": capture.get("pass") is True,
        "automated_pass": automated.get("automated_pass") is True,
        "manual_contract": all(manual_checks.values()),
        "all_required_artifacts": len(artifact_rows) == len(required),
        "g0a_false": automated.get("g0a_pass") is False,
        "physics_steps_zero": automated.get("controlled_physics_steps") == 0,
        "no_promotion": all(
            automated.get(name) is False
            for name in ("settle_executed", "ten_trial_run", "g0b_run", "rl_run", "ladder_promoted")
        ),
        "input_hashes_immutable": _input_hashes() == EXPECTED_INPUT_HASHES,
        "external_user_files_immutable": _external_baseline()["pass"],
        "status_scope": _status_scope_pass(_git_status()),
    }
    if not automated.get("measurement_pass"):
        verdict = VERDICT_GEOMETRY
    elif not (automated.get("overlay_pass") and automated.get("rerun_pass") and automated.get("viewer_capture_pass") and manual.get("manual_visual_inspection_pass")):
        verdict = VERDICT_VISUAL
    else:
        verdict = VERDICT_COMPLETE
    completion_pass = bool(verdict == VERDICT_COMPLETE and all(checks.values()))
    completion = {
        "artifact": "D350_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "final_verdict": verdict,
        "completion_pass": completion_pass,
        "scientific_scope": "actual fixed-jaw geometry MEASURED at immutable D349 target; not ALIGNED_PASS",
        "measurement_metrics": automated.get("measurement_metrics", {}),
        "artifacts": artifact_rows,
        "checks": checks,
        "manual_checks": manual_checks,
        "g0a_pass": False,
        "settle_authorized": False,
        "commit_or_push_performed": False,
    }
    _write_json(COMPLETION_PATH, completion)
    _write_text(
        COMPLETION_MD_PATH,
        "# D350 completion\n\n"
        f"- verdict: `{verdict}`\n"
        f"- completion pass: `{completion_pass}`\n"
        "- result vocabulary: MEASURED, not ALIGNED_PASS\n"
        "- physics steps: `0`\n"
        "- `g0a_pass=false`; no settle/G0b/RL promotion\n",
    )
    print(json.dumps({"stage": "finalize", "final_verdict": verdict, "pass": completion_pass}, sort_keys=True))
    return 0 if completion_pass else 2


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--viewer_hold_seconds", type=float, default=180.0)
    if stage == "validate":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D350 output path is fixed and forward-only")
    if Path(args.urdf_path).resolve() != Path(d333.DEFAULT_URDF).resolve():
        raise RuntimeError("D350 URDF path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D350 seed drift")
    if float(args.viewer_hold_seconds) < 10.0 or float(args.viewer_hold_seconds) > 600.0:
        raise RuntimeError("D350 viewer hold must be 10..600 seconds")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "finalize":
        return _run_finalize(args)

    args.headless = False
    args.livestream = 0
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    if hasattr(args, "xr"):
        args.xr = False
    args.device = "cuda:0"
    if not all(_app_arg_checks(args).values()):
        raise RuntimeError(f"D350 GUI AppLauncher drift: {_app_arg_checks(args)}")
    if not _validate_preflight(args):
        return 2

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(copy.deepcopy(args))
    simulation_app = launcher.app
    launcher_report = _resolved_gui_launcher(launcher)
    try:
        if not launcher_report["pass"]:
            raise RuntimeError(f"D350 resolved GUI launcher failed: {launcher_report}")
        return _run_validate(args, simulation_app, launcher_report)
    except Exception as error:
        if not RUNTIME_EXCEPTION_PATH.exists():
            _write_json(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D350_RUNTIME_EXCEPTION_STOP",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "controlled_physics_steps": None,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
