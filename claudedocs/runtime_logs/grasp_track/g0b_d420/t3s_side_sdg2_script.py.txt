#!/usr/bin/env python3
"""p15 / t3s — deterministic Grasping-SDG side-midpoint proposals.

This is a forward-only, instrumentation-only adapter.  It uses the installed
NVIDIA ``isaacsim.replicator.grasping`` 1.0.9 antipodal sampler on a deterministic
D29xH50 triangle proxy, but it never creates a SimulationContext, advances
physics, calls GraspingManager.evaluate_grasp_poses, or claims a grasp.

The analytic D29xH50 / 24.83 g cylinder remains the authority for p16 PhysX.
The raw SDG origin is a synthetic flying-gripper frame and is deliberately not
identified with link5, hand_tcp, or any RoArm prim.

Canonical protocol: g0b_d420/t3s_side_sdg2_prereg.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
ACTIVE_RUN_LABEL = "side_sdg2"
PREREG_PATH = CASE_DIR / "t3s_side_sdg2_prereg.md"
PREREG_SHA256 = "23acb036cd1a26f577cff8145ef4031f1c4075af3e4e60f1df28a42d86da8330"
RETIRED_FAILURE_EVIDENCE = {
    CASE_DIR / "t3s_side_sdg1_prereg.md":
        "83b3af3af1d49d67f299cdd0dbfce46c5b547e66542ce90cf589879fb0cbad13",
    CASE_DIR / "t3s_side_sdg1_failed_script.py.txt":
        "8fefc670d483f740b956649791db9650770e973263207fec52eb184370471935",
    CASE_DIR / "t3s_side_sdg1_stdout.log":
        "8752695c83c5810d4655b15f77d608a70669fda82281b73468c0d4c49ef9aef9",
    CASE_DIR / "t3s_side_sdg1_failure.json":
        "8a6d753e2ab8c962f7d627ecbe80b34e4ac5d4344a659c969f48a406210c49cc",
    CASE_DIR / "t3s_side_sdg1_argv.txt":
        "adb16cd1d3b00c1573d1cd3f6155017d5a93a7afff44e93bd62221f6e03d9663",
}
RERUN_CONTRACT_PATH = REPO / "roarm_rl/rerun_contract.py"
RERUN_CONTRACT_SHA256 = "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e"
JAW_EXTRACTOR_PATH = REPO / "sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py"
JAW_EXTRACTOR_SHA256 = "bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3"
ATTEMPT3_ROOT = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
ATTEMPT3_LAYER_SHA256 = {
    "roarm_m3.usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "configuration/roarm_m3_base.usd": "ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841",
    "configuration/roarm_m3_physics.usd": "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
    "configuration/roarm_m3_robot.usd": "2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d",
    "configuration/roarm_m3_sensor.usd": "3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f",
}

EXTENSION_ID = "isaacsim.replicator.grasping"
EXTENSION_VERSION = "1.0.9"
EXTENSION_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "exts/isaacsim.replicator.grasping"
)
EXTENSION_MANIFEST = EXTENSION_ROOT / "config/extension.toml"
EXTENSION_MANIFEST_SHA256 = "5e599aafec0d1c66776c70318535faeffc539e66070d64bf5ca15f6c5e21393a"
SAMPLER_SOURCE = (
    EXTENSION_ROOT
    / "isaacsim/replicator/grasping/sampler_utils.py"
)
SAMPLER_SOURCE_SHA256 = "613d3b41cbe0577d81bdd15a0b620a52c2516e54d80da11b6e45d1228eb1e925"

ISAACSIM_PIN = "5.1.0.0"
ISAACLAB_PIN = "2.3.0"
KIT_BUILD_PREFIX = "107.3."
NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
SCIPY_PIN = "1.15.3"
TRIMESH_PIN = "4.5.1"
RTREE_PIN = "1.3.0"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

OBJ_DIAM_M = 0.029
OBJ_RADIUS_M = OBJ_DIAM_M / 2.0
OBJ_HEIGHT_M = 0.050
OBJ_MASS_KG = 0.02483
SUPPORT_Z_M = 0.0
OBJECT_CENTER_BASE_M = np.asarray(
    [0.4235072423787768, 0.17237803311822986, 0.025], dtype=np.float64
)
OBJECT_YAW_DEG = 0.0
MESH_SEGMENTS = 256
MESH_VERTICES_SHA256 = "6cffe59dfe701358dabbddc05d04a34016b674763b761b05c7795455b0512fcb"
MESH_FACES_SHA256 = "f40e9f9fe40a882c616930a6c6436ce4d07c949367e24a31ab58c05fd5ced23b"
MESH_COMBINED_SHA256 = "871efea113d4fb4b55b33bcb87afd3b9173eed872fc39037b6a80971d9a4ae4f"

SAMPLER_CONFIG: dict[str, Any] = {
    "sampler_type": "antipodal",
    "num_candidates": 65536,
    "num_orientations": 16,
    "gripper_maximum_aperture": 0.035,
    "gripper_standoff_fingertips": 0.040,
    "gripper_approach_direction": [0.0, 0.0, 1.0],
    "grasp_align_axis": [1.0, 0.0, 0.0],
    "orientation_sample_axis": [1.0, 0.0, 0.0],
    "lateral_sigma": 0.0,
    "random_seed": 42015,
    "verbose": False,
}

MIDHEIGHT_TOL_M = 0.0025
CENTERLINE_TOL_M = 0.00025
CLOSURE_VERTICAL_TOL_DEG = 1.0
CLOSURE_TANGENT_TOL_DEG = 20.0
LINK5_Y_UP_TOL_DEG = 1.0
APPROACH_VERTICAL_TOL_DEG = 1.0
APPROACH_RADIAL_TOL_DEG = 12.0
ROT_ORTHO_TOL = 1.0e-10
ROT_DET_TOL = 1.0e-10
CANONICAL_CANDIDATE_COUNT = 8
PREGRASP_CLEARANCE_FROM_SIDE_M = 0.040
TCP_LOCAL_Z_M = 0.115428
TCP_SLAB_HALF_WIDTH_M = 0.00025
HULL_SAMPLE_TOL_M = 0.0005
EXPECTED_FIXED_INNER_X_M = -0.01002584956586361
EXPECTED_MIDPOINT_FROM_TCP_X_M = EXPECTED_FIXED_INNER_X_M + OBJ_RADIUS_M

OUTPUT_SUFFIXES = (
    "config.json",
    "mesh_proxy.json",
    "raw_candidates.json",
    "candidates.json",
    "timeline.rrd",
    "timeline.rbl",
    "rerun_validation.json",
    "inspection.png",
    "script.py.txt",
    "argv.txt",
    "failure.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_array_bytes(array: np.ndarray, dtype: str) -> bytes:
    value = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return header + b"\0" + value.tobytes(order="C")


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _retired_failure_evidence_report() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path, expected_sha in RETIRED_FAILURE_EVIDENCE.items():
        if not path.is_file():
            raise RuntimeError(f"RETIRED_SIDE_SDG1_EVIDENCE_MISSING path={path}")
        actual_sha = _sha256_file(path)
        files[path.name] = {
            "path": str(path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "pass": actual_sha == expected_sha,
        }
    if not all(row["pass"] for row in files.values()):
        raise RuntimeError(f"RETIRED_SIDE_SDG1_EVIDENCE_HASH_DRIFT files={files}")
    forbidden_success_suffixes = (
        "config.json",
        "mesh_proxy.json",
        "raw_candidates.json",
        "candidates.json",
        "timeline.rrd",
        "timeline.rbl",
        "rerun_validation.json",
        "inspection.png",
        "script.py.txt",
    )
    unexpected = [
        str(CASE_DIR / f"t3s_side_sdg1_{suffix}")
        for suffix in forbidden_success_suffixes
        if (CASE_DIR / f"t3s_side_sdg1_{suffix}").exists()
    ]
    if unexpected:
        raise RuntimeError(f"RETIRED_SIDE_SDG1_UNEXPECTED_SUCCESS_ARTIFACTS paths={unexpected}")
    return {
        "tag": "t3s_side_sdg1",
        "status": "FROZEN_FAILED__RETIRED__DO_NOT_RESUME_OR_OVERWRITE",
        "terminal_exception": "SIDE_FILTER_TOO_FEW expected_at_least=8 actual=6",
        "physics_steps": 0,
        "canonical_artifacts_emitted": 0,
        "files": files,
    }


def _write_failure_marker(
    *,
    path: Path,
    prefix: str,
    run_label: str,
    stage: str,
    error: BaseException,
    traceback_text: str,
    source_path: Path,
    source_start_bytes: bytes,
    paths: dict[str, Path],
) -> None:
    source_now = source_path.read_bytes()
    artifact_sha256 = {
        suffix: _sha256_file(artifact_path)
        for suffix, artifact_path in paths.items()
        if suffix != "failure.json" and artifact_path.is_file()
    }
    payload = {
        "schema": "g0b.t3s.side_sdg_failure.v1",
        "prefix": prefix,
        "run_label": run_label,
        "status": "FAILED_BEFORE_TERMINAL_KIT_CLOSE__NO_PHYSICS_VERDICT",
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback_text,
        "physics_steps": 0,
        "simulation_context_created": False,
        "grasping_manager_evaluate_calls": 0,
        "source_path": str(source_path),
        "source_start_sha256": _sha256_bytes(source_start_bytes),
        "source_at_failure_sha256": _sha256_bytes(source_now),
        "source_unchanged_at_failure": source_now == source_start_bytes,
        "prereg_path": str(PREREG_PATH),
        "prereg_sha256": _sha256_file(PREREG_PATH),
        "artifact_sha256_before_terminal_close": artifact_sha256,
    }
    _finite_json_tree(payload)
    _write_json(path, payload)
    _fsync_dir(path.parent)
    print(
        f"[p15_t3s] FAILURE_MARKER_WRITTEN stage={stage} "
        f"type={type(error).__name__} message={error} path={path}",
        flush=True,
    )


def _finite_json_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_json_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json_tree(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"NONFINITE_JSON_VALUE path={path} value={value!r}")


def _package_versions() -> dict[str, str]:
    versions = {
        "isaacsim": importlib.metadata.version("isaacsim"),
        "isaaclab": importlib.metadata.version("isaaclab"),
        "numpy": np.__version__,
        "psutil": importlib.metadata.version("psutil"),
        "scipy": importlib.metadata.version("scipy"),
        "trimesh": trimesh.__version__,
        "rtree": importlib.metadata.version("rtree"),
        "rerun-sdk": importlib.metadata.version("rerun-sdk"),
    }
    expected = {
        "isaacsim": ISAACSIM_PIN,
        "isaaclab": ISAACLAB_PIN,
        "numpy": NUMPY_PIN,
        "psutil": PSUTIL_PIN,
        "scipy": SCIPY_PIN,
        "trimesh": TRIMESH_PIN,
        "rtree": RTREE_PIN,
        "rerun-sdk": RERUN_VERSION,
    }
    if versions != expected:
        raise RuntimeError(f"PACKAGE_PIN_DRIFT expected={expected} actual={versions}")
    return versions


def _build_proxy_mesh() -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh, dict[str, Any]]:
    angles = np.arange(MESH_SEGMENTS, dtype=np.float64) * (2.0 * math.pi / MESH_SEGMENTS)
    ring_xy = np.column_stack([OBJ_RADIUS_M * np.cos(angles), OBJ_RADIUS_M * np.sin(angles)])
    bottom = np.column_stack([ring_xy, np.full(MESH_SEGMENTS, -OBJ_HEIGHT_M / 2.0)])
    top = np.column_stack([ring_xy, np.full(MESH_SEGMENTS, OBJ_HEIGHT_M / 2.0)])
    vertices = np.vstack([bottom, top, [[0.0, 0.0, -OBJ_HEIGHT_M / 2.0]], [[0.0, 0.0, OBJ_HEIGHT_M / 2.0]]])
    bottom_center = 2 * MESH_SEGMENTS
    top_center = bottom_center + 1
    faces: list[list[int]] = []
    for index in range(MESH_SEGMENTS):
        nxt = (index + 1) % MESH_SEGMENTS
        b0, b1 = index, nxt
        t0, t1 = MESH_SEGMENTS + index, MESH_SEGMENTS + nxt
        faces.extend([[b0, b1, t1], [b0, t1, t0]])
        faces.append([bottom_center, b1, b0])
        faces.append([top_center, t0, t1])
    vertices = np.asarray(vertices, dtype="<f8")
    face_array = np.asarray(faces, dtype="<i8")
    mesh = trimesh.Trimesh(vertices=vertices, faces=face_array, process=False, validate=False)

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    finite = bool(np.isfinite(vertices).all())
    indices_valid = bool(face_array.min() >= 0 and face_array.max() < len(vertices))
    dimensions_exact = bool(np.array_equal(extents, np.asarray([OBJ_DIAM_M, OBJ_DIAM_M, OBJ_HEIGHT_M])))
    vertex_sha = _sha256_bytes(_canonical_array_bytes(vertices, "<f8"))
    face_sha = _sha256_bytes(_canonical_array_bytes(face_array, "<i8"))
    combined_sha = _sha256_bytes(bytes.fromhex(vertex_sha) + bytes.fromhex(face_sha))
    analytic_volume = math.pi * OBJ_RADIUS_M**2 * OBJ_HEIGHT_M
    mesh_volume = float(mesh.volume)
    report = {
        "frame": "proxy_geometric_center__z_up",
        "segments": MESH_SEGMENTS,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(face_array)),
        "expected_vertex_count": 2 * MESH_SEGMENTS + 2,
        "expected_triangle_count": 4 * MESH_SEGMENTS,
        "bounds_m": bounds.tolist(),
        "extents_m": extents.tolist(),
        "diameter_m": OBJ_DIAM_M,
        "height_m": OBJ_HEIGHT_M,
        "vertices_dtype": vertices.dtype.str,
        "faces_dtype": face_array.dtype.str,
        "vertices_sha256": vertex_sha,
        "faces_sha256": face_sha,
        "combined_sha256": combined_sha,
        "expected_vertices_sha256": MESH_VERTICES_SHA256,
        "expected_faces_sha256": MESH_FACES_SHA256,
        "expected_combined_sha256": MESH_COMBINED_SHA256,
        "finite": finite,
        "indices_valid": indices_valid,
        "dimensions_exact": dimensions_exact,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "mesh_volume_m3": mesh_volume,
        "analytic_volume_m3": analytic_volume,
        "volume_relative_error": (mesh_volume - analytic_volume) / analytic_volume,
        "max_radial_chord_sagitta_mm": OBJ_RADIUS_M * (1.0 - math.cos(math.pi / MESH_SEGMENTS)) * 1000.0,
        "candidate_only": True,
        "physics_shape_authority": False,
        "mass_material_collider_authority": False,
    }
    mesh_gate = bool(
        finite
        and indices_valid
        and dimensions_exact
        and len(vertices) == 2 * MESH_SEGMENTS + 2
        and len(face_array) == 4 * MESH_SEGMENTS
        and mesh.is_watertight
        and mesh.is_winding_consistent
        and mesh_volume > 0.0
        and vertex_sha == MESH_VERTICES_SHA256
        and face_sha == MESH_FACES_SHA256
        and combined_sha == MESH_COMBINED_SHA256
    )
    report["gate_pass"] = mesh_gate
    if not mesh_gate:
        raise RuntimeError(f"MESH_PROXY_GATE_FAIL report={report}")
    return vertices, face_array, mesh, report


def _quaternion_wxyz(transform: np.ndarray) -> list[float]:
    quat = np.asarray(trimesh.transformations.quaternion_from_matrix(transform), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    first_nonzero = next((float(item) for item in quat if abs(float(item)) > 1.0e-15), 1.0)
    if first_nonzero < 0.0:
        quat = -quat
    return quat.tolist()


def _import_jaw_extractor_after_kit() -> Any:
    """Import the pinned read-only extractor without triggering its standalone re-exec."""
    module_name = "p15_pinned_attempt3_jaw_extractor"
    spec = importlib.util.spec_from_file_location(module_name, JAW_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"JAW_EXTRACTOR_SPEC_FAIL path={JAW_EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sentinel_name = "G0B_JAW_AUDIT_REEXEC"
    prior = os.environ.get(sentinel_name)
    os.environ[sentinel_name] = "1"
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            os.environ.pop(sentinel_name, None)
        else:
            os.environ[sentinel_name] = prior
    if Path(module.__file__).resolve() != JAW_EXTRACTOR_PATH.resolve():
        raise RuntimeError(f"JAW_EXTRACTOR_PATH_DRIFT actual={module.__file__}")
    return module


def _derive_midpoint_tcp_calibration(jaw: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive the asymmetric fixed-jaw midpoint offset from frozen 64-part geometry."""
    layer_manifest: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in ATTEMPT3_LAYER_SHA256.items():
        path = ATTEMPT3_ROOT.parent / relative
        actual_sha = _sha256_file(path)
        layer_manifest[relative] = {
            "path": str(path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "pass": actual_sha == expected_sha,
        }
    if not all(row["pass"] for row in layer_manifest.values()):
        raise RuntimeError(f"ATTEMPT3_LAYER_HASH_DRIFT manifest={layer_manifest}")
    asset = jaw.extract_asset()
    body_report: dict[str, Any] = {}
    for body in ("link5", "gripper_link"):
        row = asset["bodies"][body]
        active_parts = row["parts"]
        fallback = [part["name"] for part in active_parts if not part["hull_ok"]]
        legacy = [{"path": path, "enabled": bool(enabled)} for path, enabled in row["legacy"]]
        body_report[body] = {
            "active_convex_hull_part_count": len(active_parts),
            "hull_fallback_parts": fallback,
            "approximation_mismatch": row["approx_bad"],
            "legacy_colliders": legacy,
            "pass": bool(
                len(active_parts) == 64
                and not fallback
                and not row["approx_bad"]
                and len(legacy) == 1
                and legacy[0]["enabled"] is False
            ),
        }
    if not all(row["pass"] for row in body_report.values()):
        raise RuntimeError(f"ATTEMPT3_64_PLUS_64_GATE_FAIL bodies={body_report}")
    if not np.array_equal(np.asarray(jaw.TCP_LOCAL, dtype=np.float64), [0.0, 0.0, TCP_LOCAL_Z_M]):
        raise RuntimeError(f"TCP_LOCAL_DRIFT actual={np.asarray(jaw.TCP_LOCAL).tolist()}")
    if abs(float(jaw.SAMPLE_SPACING_M) - HULL_SAMPLE_TOL_M) > 1.0e-15:
        raise RuntimeError(f"JAW_SAMPLE_SPACING_DRIFT actual={jaw.SAMPLE_SPACING_M}")
    link5_points, _ = jaw.concat_parts(asset["bodies"]["link5"]["parts"])
    link5_points = np.asarray(link5_points, dtype=np.float64)
    slab_mask = (
        (np.abs(link5_points[:, 2] - TCP_LOCAL_Z_M) <= TCP_SLAB_HALF_WIDTH_M)
        & (np.abs(link5_points[:, 1]) <= OBJ_HEIGHT_M / 2.0)
    )
    slab_points = link5_points[slab_mask]
    if len(slab_points) == 0:
        raise RuntimeError("FIXED_JAW_TCP_SLAB_EMPTY")
    inner_x = float(np.max(slab_points[:, 0]))
    boundary_residual = inner_x - EXPECTED_FIXED_INNER_X_M
    if abs(boundary_residual) > HULL_SAMPLE_TOL_M:
        raise RuntimeError(
            "FIXED_JAW_INNER_BOUNDARY_DRIFT "
            f"measured={inner_x} expected={EXPECTED_FIXED_INNER_X_M} residual={boundary_residual}"
        )
    offset_x = inner_x + OBJ_RADIUS_M
    expected_offset_residual = offset_x - EXPECTED_MIDPOINT_FROM_TCP_X_M
    if abs(expected_offset_residual) > HULL_SAMPLE_TOL_M:
        raise RuntimeError(
            "MIDPOINT_TCP_OFFSET_DRIFT "
            f"measured={offset_x} expected={EXPECTED_MIDPOINT_FROM_TCP_X_M}"
        )
    offset = np.asarray([offset_x, 0.0, 0.0], dtype=np.float64)
    report = {
        "artifact": "P15_ATTEMPT3_ASYMMETRIC_JAW_MIDPOINT_TCP_CALIBRATION_V1",
        "status": "GEOMETRY_DERIVED_POSITION_MAPPING__NOT_RAW_SDG_ROOT_CALIBRATION",
        "jaw_extractor_path": str(JAW_EXTRACTOR_PATH),
        "jaw_extractor_sha256": _sha256_file(JAW_EXTRACTOR_PATH),
        "attempt3_layers": layer_manifest,
        "bodies": body_report,
        "frame": "link5",
        "tcp_local_m": [0.0, 0.0, TCP_LOCAL_Z_M],
        "slab_half_width_m": TCP_SLAB_HALF_WIDTH_M,
        "finite_cylinder_axis_in_link5": "+Y",
        "finite_cylinder_half_height_y_m": OBJ_HEIGHT_M / 2.0,
        "slab_point_count": int(len(slab_points)),
        "slab_x_min_m": float(np.min(slab_points[:, 0])),
        "slab_x_max_m_fixed_inner_boundary": inner_x,
        "expected_fixed_inner_boundary_m": EXPECTED_FIXED_INNER_X_M,
        "boundary_residual_m": boundary_residual,
        "declared_hull_surface_sampling_tolerance_m": HULL_SAMPLE_TOL_M,
        "object_radius_m": OBJ_RADIUS_M,
        "derivation": "midpoint_from_tcp_x = fixed_inner_boundary_x + object_radius",
        "midpoint_from_tcp_link5_m": offset.tolist(),
        "expected_midpoint_from_tcp_x_m": EXPECTED_MIDPOINT_FROM_TCP_X_M,
        "offset_residual_m": expected_offset_residual,
        "mapping": (
            "p_object_midpoint_base = p_tcp_base + R_base_link5 @ midpoint_from_tcp_link5; "
            "p_tcp_base = p_object_midpoint_base - R_base_link5 @ midpoint_from_tcp_link5"
        ),
        "gate_pass": True,
    }
    return offset, report


def _clamp_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _signed_axis_metrics(
    rotation: np.ndarray, radial: np.ndarray, tangent: np.ndarray
) -> dict[str, float]:
    closure = rotation[:, 0]
    link5_y = rotation[:, 1]
    approach = rotation[:, 2]
    closure_canonical = closure if float(np.dot(closure, tangent)) >= 0.0 else -closure
    closure_tangent_signed = math.degrees(
        math.atan2(float(np.dot(closure_canonical, radial)), float(np.dot(closure_canonical, tangent)))
    )
    closure_tangent_unsigned = math.degrees(
        math.acos(_clamp_unit(abs(float(np.dot(closure, tangent)))))
    )
    closure_vertical = math.degrees(math.asin(_clamp_unit(float(closure_canonical[2]))))
    approach_radial_signed = math.degrees(
        math.atan2(float(np.dot(approach, tangent)), float(np.dot(approach, radial)))
    )
    approach_vertical = math.degrees(math.asin(_clamp_unit(float(approach[2]))))
    approach_radial_3d = math.degrees(math.acos(_clamp_unit(float(np.dot(approach, radial)))))
    link5_y_world_up = math.degrees(
        math.acos(_clamp_unit(float(np.dot(link5_y, np.asarray([0.0, 0.0, 1.0])))))
    )
    return {
        "closure_tangent_signed_error_deg": closure_tangent_signed,
        "closure_tangent_unsigned_error_deg": closure_tangent_unsigned,
        "closure_vertical_signed_error_deg": closure_vertical,
        "approach_radial_signed_azimuth_error_deg": approach_radial_signed,
        "approach_vertical_signed_error_deg": approach_vertical,
        "approach_radial_3d_error_deg": approach_radial_3d,
        "link5_y_world_up_error_deg": link5_y_world_up,
    }


def _near_side_surface_point(midpoint: np.ndarray, approach: np.ndarray) -> np.ndarray:
    direction_xy = np.asarray(approach[:2], dtype=np.float64)
    direction_xy /= np.linalg.norm(direction_xy)
    point_xy = np.asarray(midpoint[:2], dtype=np.float64)
    projection = float(np.dot(point_xy, direction_xy))
    discriminant = projection**2 - (float(np.dot(point_xy, point_xy)) - OBJ_RADIUS_M**2)
    if discriminant < -1.0e-15:
        raise RuntimeError(f"SIDE_INTERSECTION_NO_REAL_ROOT discriminant={discriminant}")
    distance = projection + math.sqrt(max(discriminant, 0.0))
    result = np.asarray(midpoint, dtype=np.float64).copy()
    result[:2] = point_xy - distance * direction_xy
    radial_error = abs(float(np.linalg.norm(result[:2])) - OBJ_RADIUS_M)
    if radial_error > 1.0e-12:
        raise RuntimeError(f"SIDE_INTERSECTION_RADIUS_ERROR error={radial_error}")
    return result


def _analyze_transform(
    raw_index: int,
    transform: np.ndarray,
    radial: np.ndarray,
    tangent: np.ndarray,
) -> dict[str, Any]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError(f"RAW_TRANSFORM_INVALID index={raw_index} shape={matrix.shape}")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    approach_local = np.asarray(SAMPLER_CONFIG["gripper_approach_direction"], dtype=np.float64)
    standoff = float(SAMPLER_CONFIG["gripper_standoff_fingertips"])
    midpoint = translation + rotation @ (approach_local * standoff)
    reconstruction = midpoint - rotation @ (approach_local * standoff)
    reconstruction_error = float(np.max(np.abs(reconstruction - translation)))
    ortho_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    metrics = _signed_axis_metrics(rotation, radial, tangent)
    metrics.update(
        {
            "midpoint_height_error_mm": float(midpoint[2] * 1000.0),
            "midpoint_centerline_offset_mm": float(np.linalg.norm(midpoint[:2]) * 1000.0),
            "rotation_orthonormal_max_abs": ortho_error,
            "rotation_determinant": determinant,
            "raw_root_reconstruction_max_abs_m": reconstruction_error,
        }
    )
    checks = {
        "midheight": abs(float(midpoint[2])) <= MIDHEIGHT_TOL_M,
        "centerline": float(np.linalg.norm(midpoint[:2])) <= CENTERLINE_TOL_M,
        "closure_vertical": abs(metrics["closure_vertical_signed_error_deg"]) <= CLOSURE_VERTICAL_TOL_DEG,
        "closure_tangential": metrics["closure_tangent_unsigned_error_deg"] <= CLOSURE_TANGENT_TOL_DEG,
        "link5_y_world_up": metrics["link5_y_world_up_error_deg"] <= LINK5_Y_UP_TOL_DEG,
        "approach_vertical": abs(metrics["approach_vertical_signed_error_deg"]) <= APPROACH_VERTICAL_TOL_DEG,
        "approach_radial": abs(metrics["approach_radial_signed_azimuth_error_deg"]) <= APPROACH_RADIAL_TOL_DEG,
        "rotation_orthonormal": ortho_error <= ROT_ORTHO_TOL,
        "rotation_right_handed": abs(determinant - 1.0) <= ROT_DET_TOL,
        "raw_root_reconstruction": reconstruction_error <= 1.0e-14,
    }
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    transform_sha = _sha256_bytes(_canonical_array_bytes(matrix, "<f8"))
    base_transform = matrix.copy()
    base_transform[:3, 3] += OBJECT_CENTER_BASE_M
    midpoint_base = midpoint + OBJECT_CENTER_BASE_M
    return {
        "raw_index": raw_index,
        "raw_transform_sha256": transform_sha,
        "T_proxy_sdg_gripper": matrix.tolist(),
        "T_base_sdg_gripper": base_transform.tolist(),
        "raw_sdg_gripper_origin_proxy_m": translation.tolist(),
        "raw_sdg_gripper_origin_base_m": base_transform[:3, 3].tolist(),
        "recovered_antipodal_midpoint_proxy_m": midpoint.tolist(),
        "recovered_antipodal_midpoint_support_object_m": (midpoint + [0.0, 0.0, OBJ_HEIGHT_M / 2.0]).tolist(),
        "recovered_antipodal_midpoint_base_m": midpoint_base.tolist(),
        "orientation_quaternion_wxyz_proxy": _quaternion_wxyz(matrix),
        "axes_proxy": {
            "jaw_closure_x": rotation[:, 0].tolist(),
            "vertical_up_y": rotation[:, 1].tolist(),
            "tool_approach_z": rotation[:, 2].tolist(),
        },
        "filter_metrics": metrics,
        "filter_checks": checks,
        "filter_pass": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "raw_surface_contact_pair_endpoints": None,
        "raw_surface_contact_axis_length_m": None,
        "raw_contact_data_note": (
            "installed sample_antipodal returns transforms only; original sampled/opposing "
            "surface points and axis length are not exposed"
        ),
    }


def _candidate_from_raw(
    rank: int,
    raw: dict[str, Any],
    midpoint_from_tcp_link5: np.ndarray,
) -> dict[str, Any]:
    matrix_proxy = np.asarray(raw["T_proxy_sdg_gripper"], dtype=np.float64)
    rotation = matrix_proxy[:3, :3]
    midpoint = np.asarray(raw["recovered_antipodal_midpoint_proxy_m"], dtype=np.float64)
    approach = rotation[:, 2]
    side_surface = _near_side_surface_point(midpoint, approach)
    approach_horizontal = approach.copy()
    approach_horizontal[2] = 0.0
    approach_horizontal /= np.linalg.norm(approach_horizontal)
    pregrasp = side_surface - PREGRASP_CLEARANCE_FROM_SIDE_M * approach_horizontal
    midpoint_tf = np.eye(4, dtype=np.float64)
    midpoint_tf[:3, :3] = rotation
    midpoint_tf[:3, 3] = midpoint
    base_midpoint_tf = midpoint_tf.copy()
    base_midpoint_tf[:3, 3] += OBJECT_CENTER_BASE_M
    side_support = side_surface + np.asarray([0.0, 0.0, OBJ_HEIGHT_M / 2.0])
    side_base = side_surface + OBJECT_CENTER_BASE_M
    pregrasp_base = pregrasp + OBJECT_CENTER_BASE_M
    midpoint_base = midpoint + OBJECT_CENTER_BASE_M
    tcp_target_base = midpoint_base - rotation @ midpoint_from_tcp_link5
    link5_origin_target_base = tcp_target_base - rotation @ np.asarray(
        [0.0, 0.0, TCP_LOCAL_Z_M], dtype=np.float64
    )
    link5_target_tf = np.eye(4, dtype=np.float64)
    link5_target_tf[:3, :3] = rotation
    link5_target_tf[:3, 3] = link5_origin_target_base
    candidate_id = f"side_sdg_{rank:03d}_raw_{int(raw['raw_index']):06d}"
    return {
        "candidate_id": candidate_id,
        "candidate_rank": rank,
        "source_raw_index": int(raw["raw_index"]),
        "source_raw_transform_sha256": raw["raw_transform_sha256"],
        "candidate_frame_authority": (
            "position_is_recovered_antipodal_midpoint__orientation_is_proposed_R_base_link5"
        ),
        "candidate_frame_is_rigid_body_prim_pose": False,
        "T_proxy_candidate_midpoint": midpoint_tf.tolist(),
        "T_base_candidate_midpoint": base_midpoint_tf.tolist(),
        "R_base_link5_proposal": rotation.tolist(),
        "orientation_quaternion_wxyz_proxy": raw["orientation_quaternion_wxyz_proxy"],
        "orientation_quaternion_wxyz_base": raw["orientation_quaternion_wxyz_proxy"],
        "axes_proxy": raw["axes_proxy"],
        "axes_base": raw["axes_proxy"],
        "antipodal_midpoint_proxy_m": midpoint.tolist(),
        "antipodal_midpoint_support_object_m": (midpoint + [0.0, 0.0, OBJ_HEIGHT_M / 2.0]).tolist(),
        "antipodal_midpoint_base_m": midpoint_base.tolist(),
        "d419_side_surface_midpoint_proxy_m": side_surface.tolist(),
        "d419_side_surface_midpoint_support_object_m": side_support.tolist(),
        "d419_side_surface_midpoint_base_m": side_base.tolist(),
        "pregrasp_reference_base_m": pregrasp_base.tolist(),
        "pregrasp_clearance_from_side_m": PREGRASP_CLEARANCE_FROM_SIDE_M,
        "geometry_mapped_roarm_targets": {
            "status": "POSITION_MAPPING_DERIVED_FROM_PINNED_ATTEMPT3_GEOMETRY__IK_UNTESTED",
            "midpoint_from_tcp_link5_m": midpoint_from_tcp_link5.tolist(),
            "tcp_target_base_m": tcp_target_base.tolist(),
            "tcp_target_orientation_R_base_link5": rotation.tolist(),
            "link5_origin_target_base_m": link5_origin_target_base.tolist(),
            "T_base_link5_target_hypothesis": link5_target_tf.tolist(),
            "T_link5_tcp": {
                "rotation": np.eye(3, dtype=np.float64).tolist(),
                "translation_link5_m": [0.0, 0.0, TCP_LOCAL_Z_M],
            },
            "mapping": (
                "p_tcp_base = p_antipodal_midpoint_base - "
                "R_base_link5_proposal @ midpoint_from_tcp_link5"
            ),
        },
        "raw_sdg_gripper_origin_proxy_m": raw["raw_sdg_gripper_origin_proxy_m"],
        "raw_sdg_gripper_origin_base_m": raw["raw_sdg_gripper_origin_base_m"],
        "filter_metrics": raw["filter_metrics"],
        "filter_checks": raw["filter_checks"],
        "raw_surface_contact_pair_endpoints": None,
        "q5_control": {
            "open_deg": None,
            "close_deg": None,
            "authority": "p16_physics_harness_unassigned_by_p15",
        },
        "raw_sdg_root_calibration": {
            "gripper_frame_prim": None,
            "T_sdg_gripper_link5": None,
            "status": "UNCALIBRATED_RAW_FLYING_ROOT__DO_NOT_USE_AS_ROARM_POSE",
        },
    }


def _cylinder_lines(center: np.ndarray) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, 129)
    strips: list[np.ndarray] = []
    for z in (SUPPORT_Z_M, SUPPORT_Z_M + OBJ_HEIGHT_M):
        strips.append(
            np.column_stack(
                [
                    center[0] + OBJ_RADIUS_M * np.cos(angles),
                    center[1] + OBJ_RADIUS_M * np.sin(angles),
                    np.full_like(angles, z),
                ]
            ).astype(np.float32)
        )
    for angle in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
        strips.append(
            np.asarray(
                [
                    [center[0] + OBJ_RADIUS_M * math.cos(angle), center[1] + OBJ_RADIUS_M * math.sin(angle), SUPPORT_Z_M],
                    [center[0] + OBJ_RADIUS_M * math.cos(angle), center[1] + OBJ_RADIUS_M * math.sin(angle), SUPPORT_Z_M + OBJ_HEIGHT_M],
                ],
                dtype=np.float32,
            )
        )
    return strips


def _emit_rerun(
    paths: dict[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    raw_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    counts: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    import rerun as rr
    import rerun.blueprint as rrb

    from roarm_rl.rerun_contract import validate_rerun_artifact

    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"RERUN_PIN_DRIFT expected={RERUN_VERSION} actual={rr.__version__}")
    entity_names = [
        "metadata/run",
        "events/filter",
        "events/verdict",
        "world/support_plane",
        "world/analytic_cylinder",
        "world/proxy_mesh",
        "world/desired/radial",
        "world/desired/tangent",
        "world/desired/up",
        "world/desired/down",
        "world/raw/gripper_origins",
        "world/raw/tool_axes",
        "world/raw/closure_axes",
        "world/raw/up_axes",
        "world/accepted/midpoints",
        "world/accepted/side_surface",
        "world/accepted/pregrasp",
        "world/accepted/tcp_targets",
        "world/accepted/tool_axes",
        "world/accepted/closure_axes",
        "world/accepted/up_axes",
        "plots/filter_pass",
        "plots/midheight_error_mm",
        "plots/closure_tangent_error_deg",
        "plots/approach_radial_error_deg",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "events/filter": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"],
        "world/support_plane": ["LineStrips3D:strips"],
        "world/analytic_cylinder": ["LineStrips3D:strips"],
        "world/proxy_mesh": ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"],
        "world/desired/radial": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/desired/tangent": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/desired/up": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/desired/down": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/raw/gripper_origins": ["Points3D:positions"],
        "world/raw/tool_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/raw/closure_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/raw/up_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/accepted/midpoints": ["Points3D:positions"],
        "world/accepted/side_surface": ["Points3D:positions"],
        "world/accepted/pregrasp": ["Points3D:positions"],
        "world/accepted/tcp_targets": ["Points3D:positions"],
        "world/accepted/tool_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/accepted/closure_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "world/accepted/up_axes": ["Arrows3D:origins", "Arrows3D:vectors"],
        "plots/filter_pass": ["Scalars:scalars"],
        "plots/midheight_error_mm": ["Scalars:scalars"],
        "plots/closure_tangent_error_deg": ["Scalars:scalars"],
        "plots/approach_radial_error_deg": ["Scalars:scalars"],
    }
    radial_xy = OBJECT_CENTER_BASE_M[:2] / np.linalg.norm(OBJECT_CENTER_BASE_M[:2])
    radial = np.asarray([radial_xy[0], radial_xy[1], 0.0], dtype=np.float32)
    tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float32)
    origin = OBJECT_CENTER_BASE_M.astype(np.float32)
    summary = (
        f"# p15 / {prefix} Grasping-SDG side proposals\n\n"
        f"raw transforms: **{counts['raw_transform_count']}**  \n"
        f"side-filter pass: **{counts['filter_pass_count']}**  \n"
        f"canonical proposals: **{counts['canonical_candidate_count']}**  \n"
        "**instrumentation only — no IK, contact, lift, physics, or grasp verdict**  \n\n"
        "Raw SDG roots are flying-gripper coordinates and are not RoArm TCPs. "
        "Float64 JSON/hash data is authoritative; Rerun values are Float32 inspection copies."
    )
    plane = np.asarray(
        [[[-0.05, -0.60, 0.0], [0.60, -0.60, 0.0], [0.60, 0.60, 0.0], [-0.05, 0.60, 0.0], [-0.05, -0.60, 0.0]]],
        dtype=np.float32,
    )
    proxy_world = (vertices + OBJECT_CENTER_BASE_M).astype(np.float32)

    selected_raw_indices = {int(row["source_raw_index"]) for row in candidates}
    stride_indices = set(np.linspace(0, len(raw_rows) - 1, min(256, len(raw_rows)), dtype=int).tolist())
    view_indices = sorted(stride_indices | selected_raw_indices)
    raw_view = [raw_rows[index] for index in view_indices]
    raw_origins = np.asarray([row["raw_sdg_gripper_origin_base_m"] for row in raw_view], dtype=np.float32)
    raw_tool = np.asarray([row["axes_proxy"]["tool_approach_z"] for row in raw_view], dtype=np.float32)
    raw_closure = np.asarray([row["axes_proxy"]["jaw_closure_x"] for row in raw_view], dtype=np.float32)
    raw_up = np.asarray([row["axes_proxy"]["vertical_up_y"] for row in raw_view], dtype=np.float32)
    raw_colors = np.asarray(
        [[35, 205, 95] if row["filter_pass"] else [210, 65, 65] for row in raw_view], dtype=np.uint8
    )
    accepted_mid = np.asarray([row["antipodal_midpoint_base_m"] for row in candidates], dtype=np.float32)
    accepted_side = np.asarray([row["d419_side_surface_midpoint_base_m"] for row in candidates], dtype=np.float32)
    accepted_pre = np.asarray([row["pregrasp_reference_base_m"] for row in candidates], dtype=np.float32)
    accepted_tcp = np.asarray(
        [row["geometry_mapped_roarm_targets"]["tcp_target_base_m"] for row in candidates],
        dtype=np.float32,
    )
    accepted_tool = np.asarray([row["axes_base"]["tool_approach_z"] for row in candidates], dtype=np.float32)
    accepted_closure = np.asarray([row["axes_base"]["jaw_closure_x"] for row in candidates], dtype=np.float32)
    accepted_up = np.asarray([row["axes_base"]["vertical_up_y"] for row in candidates], dtype=np.float32)

    app_id = f"roarm_g0b_{prefix}"
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", contents="/world/**", name="1 | proxy + raw/accepted frames"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="2 | authority and counts"),
                rrb.TextLogView(origin="/events", contents="/events/**", name="3 | filters and non-verdict"),
                rrb.TimeSeriesView(origin="/plots", contents="/plots/**", name="4 | accepted candidate errors"),
            ),
            column_shares=[2, 1],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    with rr.RecordingStream(
        app_id, recording_id=f"g0b_d420_{prefix}", make_default=False, send_properties=True
    ) as recording:
        recording.save(str(paths["timeline.rrd"]), write_footer=True)
        recording.log("metadata/run", rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN), static=True)
        recording.log(
            "events/filter",
            rr.TextLog(
                f"raw={counts['raw_transform_count']} pass={counts['filter_pass_count']} selected={counts['canonical_candidate_count']} rejection_counts={counts['rejection_counts']}",
                level=rr.TextLogLevel.INFO,
            ),
            static=True,
        )
        recording.log(
            "events/verdict",
            rr.TextLog(
                "CANDIDATE_INSTRUMENTATION_PASS__NO_PHYSICS_OR_GRASP_VERDICT; p16 calibration/IK/PhysX required",
                level=rr.TextLogLevel.WARN,
            ),
            static=True,
        )
        recording.log("world/support_plane", rr.LineStrips3D(plane, colors=[105, 105, 105], radii=0.001), static=True)
        recording.log("world/analytic_cylinder", rr.LineStrips3D(_cylinder_lines(OBJECT_CENTER_BASE_M), colors=[230, 170, 55], radii=0.0015), static=True)
        recording.log("world/proxy_mesh", rr.Mesh3D(vertex_positions=proxy_world, triangle_indices=faces.astype(np.uint32), albedo_factor=[65, 140, 240, 80]), static=True)
        recording.log("world/desired/radial", rr.Arrows3D(origins=[origin], vectors=[radial * 0.07], colors=[[250, 70, 50]], radii=[0.0015], labels=["desired +Z radial"]), static=True)
        recording.log("world/desired/tangent", rr.Arrows3D(origins=[origin], vectors=[tangent * 0.07], colors=[[70, 230, 100]], radii=[0.0015], labels=["desired +X tangent"]), static=True)
        recording.log("world/desired/up", rr.Arrows3D(origins=[origin], vectors=[[0.0, 0.0, 0.06]], colors=[[60, 145, 250]], radii=[0.0015], labels=["desired +Y world up"]), static=True)
        recording.log("world/desired/down", rr.Arrows3D(origins=[origin], vectors=[[0.0, 0.0, -0.06]], colors=[[70, 120, 250]], radii=[0.0015], labels=["world down"]), static=True)
        recording.log("world/raw/gripper_origins", rr.Points3D(raw_origins, colors=raw_colors, radii=0.0012), static=True)
        recording.log("world/raw/tool_axes", rr.Arrows3D(origins=raw_origins, vectors=raw_tool * 0.012, colors=raw_colors, radii=0.00035), static=True)
        recording.log("world/raw/closure_axes", rr.Arrows3D(origins=raw_origins, vectors=raw_closure * 0.009, colors=raw_colors, radii=0.00030), static=True)
        recording.log("world/raw/up_axes", rr.Arrows3D(origins=raw_origins, vectors=raw_up * 0.009, colors=raw_colors, radii=0.00030), static=True)
        recording.log("world/accepted/midpoints", rr.Points3D(accepted_mid, colors=[255, 245, 50], radii=0.0025, labels=[row["candidate_id"] for row in candidates]), static=True)
        recording.log("world/accepted/side_surface", rr.Points3D(accepted_side, colors=[255, 120, 35], radii=0.0023), static=True)
        recording.log("world/accepted/pregrasp", rr.Points3D(accepted_pre, colors=[210, 70, 245], radii=0.0020), static=True)
        recording.log("world/accepted/tcp_targets", rr.Points3D(accepted_tcp, colors=[35, 220, 235], radii=0.0020), static=True)
        recording.log("world/accepted/tool_axes", rr.Arrows3D(origins=accepted_mid, vectors=accepted_tool * 0.055, colors=[255, 70, 45], radii=0.0012), static=True)
        recording.log("world/accepted/closure_axes", rr.Arrows3D(origins=accepted_mid, vectors=accepted_closure * 0.040, colors=[65, 235, 95], radii=0.0011), static=True)
        recording.log("world/accepted/up_axes", rr.Arrows3D(origins=accepted_mid, vectors=accepted_up * 0.040, colors=[65, 145, 250], radii=0.0011), static=True)
        for index, candidate in enumerate(candidates):
            recording.reset_time()
            recording.set_time("candidate_index", sequence=index)
            metrics = candidate["filter_metrics"]
            recording.log("plots/filter_pass", rr.Scalars(1.0))
            recording.log("plots/midheight_error_mm", rr.Scalars(float(metrics["midpoint_height_error_mm"])))
            recording.log("plots/closure_tangent_error_deg", rr.Scalars(float(metrics["closure_tangent_signed_error_deg"])))
            recording.log("plots/approach_radial_error_deg", rr.Scalars(float(metrics["approach_radial_signed_azimuth_error_deg"])))
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.flush(timeout_sec=180.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=entity_names,
        exact_entity_paths=entity_names,
        exact_timeline_names=["blueprint", "candidate_index", "log_time"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"],
        screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=600.0,
    )
    _write_json(paths["rerun_validation.json"], validation)
    return {
        "technical_pass": bool(validation.get("pass")),
        "errors": validation.get("errors", []),
        "manual_visual_inspection": "PENDING_ROOT_REVIEW__DO_NOT_CLAIM_D341_COMPLETE",
    }


def _activate_installed_sampler() -> tuple[Any, dict[str, Any]]:
    import omni.kit.app

    kit_app = omni.kit.app.get_app()
    kit_build_version = str(kit_app.get_build_version())
    if not kit_build_version.startswith(KIT_BUILD_PREFIX):
        raise RuntimeError(
            f"KIT_BUILD_VERSION_DRIFT expected_prefix={KIT_BUILD_PREFIX} actual={kit_build_version}"
        )
    manager = kit_app.get_extension_manager()
    enabled_before = bool(manager.is_extension_enabled(EXTENSION_ID))
    enable_return = manager.set_extension_enabled_immediate(EXTENSION_ID, True)
    enabled_after = bool(manager.is_extension_enabled(EXTENSION_ID))
    enabled_id = manager.get_enabled_extension_id(EXTENSION_ID)
    if not enabled_after or not enabled_id:
        raise RuntimeError(
            f"GRASPING_EXTENSION_ENABLE_FAIL before={enabled_before} return={enable_return!r} id={enabled_id!r}"
        )
    extension_info = manager.get_extension_dict(enabled_id)
    extension_version = str(extension_info["package"]["version"])
    extension_path = Path(str(manager.get_extension_path(enabled_id))).resolve()
    if extension_version != EXTENSION_VERSION or extension_path != EXTENSION_ROOT.resolve():
        raise RuntimeError(
            "GRASPING_EXTENSION_RUNTIME_DRIFT "
            f"version={extension_version} path={extension_path}"
        )
    from isaacsim.replicator.grasping import sampler_utils

    sampler_path = Path(sampler_utils.__file__).resolve()
    if sampler_path != SAMPLER_SOURCE.resolve():
        raise RuntimeError(f"SAMPLER_MODULE_PATH_DRIFT expected={SAMPLER_SOURCE} actual={sampler_path}")
    provenance = {
        "extension_id": EXTENSION_ID,
        "kit_build_version": kit_build_version,
        "enabled_extension_id": str(enabled_id),
        "enabled_before_explicit_call": enabled_before,
        "enable_return": enable_return if isinstance(enable_return, (bool, int, float, str, type(None))) else type(enable_return).__name__,
        "enabled_after": enabled_after,
        "runtime_version": extension_version,
        "runtime_extension_root": str(extension_path),
        "runtime_sampler_path": str(sampler_path),
        "extension_manifest_sha256": _sha256_file(EXTENSION_MANIFEST),
        "sampler_source_sha256": _sha256_file(sampler_path),
    }
    return sampler_utils, provenance


def _paths_for_label(run_label: str) -> tuple[str, dict[str, Path]]:
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", run_label):
        raise ValueError(f"invalid run_label={run_label!r}")
    if run_label != ACTIVE_RUN_LABEL:
        raise ValueError(
            f"retired_or_unregistered_run_label={run_label!r} active={ACTIVE_RUN_LABEL!r}"
        )
    prefix = f"t3s_{run_label}"
    return prefix, {suffix: CASE_DIR / f"{prefix}_{suffix}" for suffix in OUTPUT_SUFFIXES}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_label", default=ACTIVE_RUN_LABEL, choices=[ACTIVE_RUN_LABEL])
    parser.add_argument("--protocol_path", type=Path, default=PREREG_PATH)
    parser.add_argument("--protocol_sha256", default=PREREG_SHA256, choices=[PREREG_SHA256])
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    source_path = Path(__file__).resolve()
    source_start_bytes = source_path.read_bytes()
    source_start_sha = _sha256_bytes(source_start_bytes)
    prefix, paths = _paths_for_label(args.run_label)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"G0_OUTPUT_EXISTS prefix={prefix} paths={existing}")
    protocol_path = args.protocol_path.resolve()
    if protocol_path != PREREG_PATH.resolve():
        raise RuntimeError(f"PROTOCOL_PATH_DRIFT expected={PREREG_PATH} actual={protocol_path}")
    if _sha256_file(protocol_path) != args.protocol_sha256:
        raise RuntimeError(
            f"PROTOCOL_HASH_DRIFT expected={args.protocol_sha256} actual={_sha256_file(protocol_path)}"
        )
    if _sha256_file(EXTENSION_MANIFEST) != EXTENSION_MANIFEST_SHA256:
        raise RuntimeError("EXTENSION_MANIFEST_HASH_DRIFT")
    if _sha256_file(SAMPLER_SOURCE) != SAMPLER_SOURCE_SHA256:
        raise RuntimeError("SAMPLER_SOURCE_HASH_DRIFT")
    if _sha256_file(RERUN_CONTRACT_PATH) != RERUN_CONTRACT_SHA256:
        raise RuntimeError("RERUN_CONTRACT_HASH_DRIFT")
    if _sha256_file(JAW_EXTRACTOR_PATH) != JAW_EXTRACTOR_SHA256:
        raise RuntimeError("JAW_EXTRACTOR_HASH_DRIFT")
    versions = _package_versions()
    vertices, faces, mesh, mesh_report = _build_proxy_mesh()
    retired_failure = _retired_failure_evidence_report()
    if source_path.read_bytes() != source_start_bytes:
        raise RuntimeError("EXECUTED_SOURCE_CHANGED_BEFORE_EARLY_FREEZE")
    paths["script.py.txt"].write_bytes(source_start_bytes)
    paths["argv.txt"].write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    with paths["script.py.txt"].open("rb") as stream:
        os.fsync(stream.fileno())
    with paths["argv.txt"].open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_dir(CASE_DIR)
    print(
        f"[p15_t3s] EARLY_SOURCE_FROZEN prefix={prefix} sha256={source_start_sha} "
        f"surface_samples={SAMPLER_CONFIG['num_candidates'] // SAMPLER_CONFIG['num_orientations']}",
        flush=True,
    )

    radial_xy = OBJECT_CENTER_BASE_M[:2] / np.linalg.norm(OBJECT_CENTER_BASE_M[:2])
    radial = np.asarray([radial_xy[0], radial_xy[1], 0.0], dtype=np.float64)
    tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64)

    simulation_app = None
    failure_stage = "kit_launch"
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=False)
        simulation_app = launcher.app
        failure_stage = "sampler_extension_activation"
        sampler_utils, extension_runtime = _activate_installed_sampler()
        failure_stage = "attempt3_midpoint_tcp_calibration"
        jaw_extractor = _import_jaw_extractor_after_kit()
        midpoint_from_tcp_link5, midpoint_tcp_calibration = _derive_midpoint_tcp_calibration(
            jaw_extractor
        )
        failure_stage = "sdg_sample_replay_first"
        first = sampler_utils.sample_antipodal(mesh, **SAMPLER_CONFIG)
        failure_stage = "sdg_sample_replay_second"
        second = sampler_utils.sample_antipodal(mesh, **SAMPLER_CONFIG)
        first_array = np.asarray(first, dtype=np.float64)
        second_array = np.asarray(second, dtype=np.float64)
        deterministic = bool(
            first_array.shape == second_array.shape and np.array_equal(first_array, second_array)
        )
        if not deterministic:
            max_abs = None
            if first_array.shape == second_array.shape and first_array.size:
                max_abs = float(np.max(np.abs(first_array - second_array)))
            raise RuntimeError(
                f"SAMPLER_REPLAY_NOT_BIT_IDENTICAL first={first_array.shape} second={second_array.shape} max_abs={max_abs}"
            )
        if first_array.ndim != 3 or first_array.shape[1:] != (4, 4) or len(first_array) == 0:
            raise RuntimeError(f"SAMPLER_OUTPUT_SHAPE_INVALID shape={first_array.shape}")
        failure_stage = "raw_transform_analysis_and_side_filter"
        raw_rows = [
            _analyze_transform(index, transform, radial, tangent)
            for index, transform in enumerate(first_array)
        ]
        transform_hashes = [row["raw_transform_sha256"] for row in raw_rows]
        duplicate_count = len(transform_hashes) - len(set(transform_hashes))
        if duplicate_count:
            raise RuntimeError(f"RAW_TRANSFORM_DUPLICATES count={duplicate_count}")
        accepted = [row for row in raw_rows if row["filter_pass"]]
        accepted.sort(
            key=lambda row: (
                abs(float(row["filter_metrics"]["midpoint_height_error_mm"])),
                abs(float(row["filter_metrics"]["closure_tangent_signed_error_deg"])),
                abs(float(row["filter_metrics"]["approach_radial_signed_azimuth_error_deg"])),
                float(row["filter_metrics"]["midpoint_centerline_offset_mm"]),
                int(row["raw_index"]),
            )
        )
        failure_stage = "canonical_candidate_count_gate"
        if len(accepted) < CANONICAL_CANDIDATE_COUNT:
            raise RuntimeError(
                f"SIDE_FILTER_TOO_FEW expected_at_least={CANONICAL_CANDIDATE_COUNT} actual={len(accepted)}"
            )
        candidates = [
            _candidate_from_raw(rank, row, midpoint_from_tcp_link5)
            for rank, row in enumerate(accepted[:CANONICAL_CANDIDATE_COUNT])
        ]

        rejection_counts: dict[str, int] = {}
        for row in raw_rows:
            for reason in row["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        counts = {
            "configured_surface_sample_count": int(
                SAMPLER_CONFIG["num_candidates"] // SAMPLER_CONFIG["num_orientations"]
            ),
            "configured_candidate_attempt_count": int(SAMPLER_CONFIG["num_candidates"]),
            "raw_transform_count": len(raw_rows),
            "filter_pass_count": len(accepted),
            "filter_fail_count": len(raw_rows) - len(accepted),
            "canonical_candidate_count": len(candidates),
            "duplicate_raw_transform_count": duplicate_count,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        }

        source_end_bytes = source_path.read_bytes()
        source_stable = source_end_bytes == source_start_bytes
        if not source_stable:
            raise RuntimeError(
                f"EXECUTED_SOURCE_CHANGED start={source_start_sha} end={_sha256_bytes(source_end_bytes)}"
            )

        failure_stage = "scientific_json_artifact_write"
        mesh_payload = {
            "schema": "g0b.t3s.side_sdg_mesh_proxy.v1",
            "authority": "candidate_only_triangle_proxy__analytic_physx_cylinder_remains_authority",
            "report": mesh_report,
            "vertices_m_float64": vertices.tolist(),
            "triangles_int64": faces.tolist(),
        }
        _finite_json_tree(mesh_payload)
        _write_json(paths["mesh_proxy.json"], mesh_payload)

        frame_contract = {
            "matrix_notation": "T_A_B maps coordinates in B into A",
            "quaternion_order": "wxyz",
            "quaternion_semantics": "active local_to_parent_rotation",
            "proxy_frame": "cylinder_geometric_center__z_up",
            "support_object_frame": "cylinder_bottom_center__z_up",
            "base_frame": "fixed_roarm_base_world",
            "candidate_local_axes": {
                "+x": "jaw_closure_antipodal_axis__desired_horizontal_tangent",
                "+y": "q5_joint_axis__desired_world_up",
                "+z": "tool_approach_from_gripper_toward_object",
            },
            "right_handed_identity": "+X_cross_+Y_equals_+Z",
            "raw_sdg_transform": (
                "T_proxy_sdg_gripper=T_midpoint*R*T(-approach*standoff); synthetic "
                "flying-gripper root, not a RoArm prim or TCP"
            ),
            "raw_sdg_root_calibration": {
                "gripper_frame_prim": None,
                "T_sdg_gripper_link5": None,
                "status": "UNCALIBRATED_RAW_FLYING_ROOT__PROVENANCE_ONLY",
            },
            "candidate_frame": (
                "position is recovered antipodal midpoint; rotation is proposed "
                "R_base_link5; this mixed frame is not a rigid-body prim pose"
            ),
            "geometry_mapped_target": (
                "candidate rows separately derive TCP and link5-origin target positions "
                "from pinned attempt3 geometry; IK and collision remain untested"
            ),
            "p16_requirement": (
                "reject raw root; consume only geometry_mapped_roarm_targets and independently "
                "test parsed-URDF JOINT_LIMITS, collision, PhysX contact, and lift"
            ),
        }
        raw_payload = {
            "schema": "g0b.t3s.side_sdg_raw_candidates.v1",
            "scientific_authority": "instrumentation_only__no_physics_or_grasp_verdict",
            "reactive_predecessor": retired_failure,
            "sampler_config": SAMPLER_CONFIG,
            "frame_contract": frame_contract,
            "midpoint_tcp_position_calibration": midpoint_tcp_calibration,
            "counts": counts,
            "rows": raw_rows,
        }
        _finite_json_tree(raw_payload)
        _write_json(paths["raw_candidates.json"], raw_payload)

        candidates_payload = {
            "schema": "g0b.t3s.side_sdg_candidates.v1",
            "run_label": args.run_label,
            "instrumentation_verdict": "CANDIDATE_INSTRUMENTATION_PASS__NO_PHYSICS_OR_GRASP_VERDICT",
            "scientific_physics_verdict": None,
            "reactive_predecessor": retired_failure,
            "p16_consumption_allowed_only_if_rerun_validation_pass_and_manual_inspection": True,
            "object_contract": {
                "physics_authority": "analytic_upright_cylinder_in_p16",
                "diameter_m": OBJ_DIAM_M,
                "radius_m": OBJ_RADIUS_M,
                "height_m": OBJ_HEIGHT_M,
                "mass_kg": OBJ_MASS_KG,
                "center_base_m": OBJECT_CENTER_BASE_M.tolist(),
                "pose_source": "p10.FOUR_SPONGE_SEED0_SOURCES.seed0_S4 + support_z + H/2",
                "yaw_deg": OBJECT_YAW_DEG,
                "support_z_m": SUPPORT_Z_M,
                "grasp_point_case_exception": "D419 top_center_to_side_midpoint__sim_only_user_approved",
                "material_friction": "not_sampled_not_measured_not_claimed_by_p15",
            },
            "mesh_proxy_contract": mesh_report,
            "frame_contract": frame_contract,
            "midpoint_tcp_position_calibration": midpoint_tcp_calibration,
            "desired_axes_base": {
                "radial_tool_approach_plus_z": radial.tolist(),
                "tangential_jaw_closure_plus_x": tangent.tolist(),
                "vertical_link5_plus_y": [0.0, 0.0, 1.0],
                "world_down": [0.0, 0.0, -1.0],
            },
            "filter_contract": {
                "midheight_abs_max_m": MIDHEIGHT_TOL_M,
                "centerline_offset_max_m": CENTERLINE_TOL_M,
                "closure_vertical_abs_max_deg": CLOSURE_VERTICAL_TOL_DEG,
                "closure_tangent_unsigned_max_deg": CLOSURE_TANGENT_TOL_DEG,
                "link5_y_world_up_max_deg": LINK5_Y_UP_TOL_DEG,
                "approach_vertical_abs_max_deg": APPROACH_VERTICAL_TOL_DEG,
                "approach_radial_signed_azimuth_abs_max_deg": APPROACH_RADIAL_TOL_DEG,
                "rotation_orthonormal_max_abs": ROT_ORTHO_TOL,
                "rotation_determinant_abs_error_max": ROT_DET_TOL,
                "ordering": [
                    "abs(midpoint_height_error_mm)",
                    "abs(closure_tangent_signed_error_deg)",
                    "abs(approach_radial_signed_azimuth_error_deg)",
                    "midpoint_centerline_offset_mm",
                    "raw_index",
                ],
            },
            "sampler": {
                "extension_id": EXTENSION_ID,
                "extension_version": EXTENSION_VERSION,
                "installed_function": "sampler_utils.sample_antipodal",
                "config": SAMPLER_CONFIG,
                "determinism_replays": 2,
                "determinism_bit_identical": deterministic,
            },
            "counts": counts,
            "provenance": {
                "prereg_path": str(protocol_path),
                "prereg_sha256": args.protocol_sha256,
                "executed_source_path": str(source_path),
                "executed_source_sha256": source_start_sha,
                "executed_source_stable": source_stable,
                "frozen_source_path": str(paths["script.py.txt"]),
                "frozen_source_sha256": _sha256_file(paths["script.py.txt"]),
                "mesh_proxy_path": str(paths["mesh_proxy.json"]),
                "mesh_proxy_file_sha256": _sha256_file(paths["mesh_proxy.json"]),
                "raw_candidates_path": str(paths["raw_candidates.json"]),
                "raw_candidates_sha256": _sha256_file(paths["raw_candidates.json"]),
                "extension_runtime": extension_runtime,
                "package_versions": versions,
                "rerun_contract_path": str(RERUN_CONTRACT_PATH),
                "rerun_contract_sha256": RERUN_CONTRACT_SHA256,
                "jaw_extractor_path": str(JAW_EXTRACTOR_PATH),
                "jaw_extractor_sha256": JAW_EXTRACTOR_SHA256,
                "reactive_change": "num_candidates_16384_to_65536__all_safety_gates_unchanged",
            },
            "candidates": candidates,
        }
        _finite_json_tree(candidates_payload)
        _write_json(paths["candidates.json"], candidates_payload)

        failure_stage = "rerun_rrd_rbl_png_validation"
        rerun_report = _emit_rerun(paths, vertices, faces, raw_rows, candidates, counts, prefix)
        failure_stage = "source_finalization_and_config_write"
        source_final_bytes = source_path.read_bytes()
        if source_final_bytes != source_start_bytes:
            raise RuntimeError(
                "EXECUTED_SOURCE_CHANGED_DURING_OBSERVABILITY "
                f"start={source_start_sha} final={_sha256_bytes(source_final_bytes)}"
            )
        if paths["script.py.txt"].read_bytes() != source_start_bytes:
            raise RuntimeError("FROZEN_SOURCE_BYTES_MISMATCH")
        config_payload = {
            "schema": "g0b.t3s.side_sdg_run.v1",
            "run_label": args.run_label,
            "prefix": prefix,
            "run_valid": bool(rerun_report["technical_pass"]),
            "instrumentation_verdict": (
                "CANDIDATE_INSTRUMENTATION_PASS__NO_PHYSICS_OR_GRASP_VERDICT"
                if rerun_report["technical_pass"]
                else "OBSERVABILITY_FAIL__P16_CONSUMPTION_FORBIDDEN"
            ),
            "physics_steps": 0,
            "simulation_context_created": False,
            "grasping_manager_evaluate_calls": 0,
            "render_products": 0,
            "rerun": rerun_report,
            "reactive_predecessor": retired_failure,
            "counts": counts,
            "midpoint_tcp_position_calibration": midpoint_tcp_calibration,
            "artifact_sha256": {
                suffix: _sha256_file(path)
                for suffix, path in paths.items()
                if suffix != "config.json" and path.is_file()
            },
        }
        _finite_json_tree(config_payload)
        _write_json(paths["config.json"], config_payload)
        _fsync_dir(CASE_DIR)
        print(
            f"[p15_t3s] INSTRUMENTATION_DONE raw={len(raw_rows)} pass={len(accepted)} "
            f"selected={len(candidates)} rerun_pass={rerun_report['technical_pass']}",
            flush=True,
        )
        if not rerun_report["technical_pass"]:
            failure_stage = "rerun_contract_terminal_gate"
            raise RuntimeError(f"RERUN_CONTRACT_FAIL errors={rerun_report['errors']}")
    except BaseException as error:
        marker_traceback = traceback.format_exc()
        try:
            _write_failure_marker(
                path=paths["failure.json"],
                prefix=prefix,
                run_label=args.run_label,
                stage=failure_stage,
                error=error,
                traceback_text=marker_traceback,
                source_path=source_path,
                source_start_bytes=source_start_bytes,
                paths=paths,
            )
        except BaseException as marker_error:
            print(
                f"[p15_t3s] FAILURE_MARKER_WRITE_FAILED original={type(error).__name__}:{error} "
                f"marker={type(marker_error).__name__}:{marker_error}",
                flush=True,
            )
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
