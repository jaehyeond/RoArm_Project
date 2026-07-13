#!/usr/bin/env python3
"""D342 authored-coordinate-stream contract repair for cylinder G0a.

This case is read-only.  It reads the immutable D339 attempt2 USDC point and
face arrays before any transform, compares their ordered byte streams with the
D339 cold-cook/manifest evidence, and uses D340 body-mapped arrays only for
numeric containment/proximity checks.  It never recooks, authors attempt3,
creates a simulation context, or advances physics.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr
import scipy
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, sha256_file, validate_rerun_artifact
from roarm_rl.viz_debug import log_rerun
from sim_scripts import cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339


OUT_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d342"
PREREG_PATH = OUT_DIR / "d342_preregistration.json"
PREREG_AMENDMENT_PATH = OUT_DIR / "d342_preregistration_amendment_001.json"
PARAMETER_AUDIT_PATH = OUT_DIR / "d342_parameter_freeze_audit.json"
EVIDENCE_PATH = OUT_DIR / "d342_authored_coordinate_stream_evidence.json"
GOOD_RRD = OUT_DIR / "d342_authored_coordinate_stream.rrd"
GOOD_RBL = OUT_DIR / "d342_authored_coordinate_stream.rbl"
SCREENSHOT = OUT_DIR / "d342_authored_coordinate_stream_rerun_inspection.png"
AUTOMATED_SUMMARY = OUT_DIR / "d342_authored_coordinate_stream_automated_summary.json"
AUTOMATED_REPORT = OUT_DIR / "d342_authored_coordinate_stream_automated_report.md"

D339_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d339"
D339_ATTEMPT2 = D339_DIR / "collision_asset/attempt2"
D339_PHYSICS = (
    D339_ATTEMPT2
    / "roarm_m3_fullmesh_convex_parts/configuration/roarm_m3_physics.usd"
)
D339_MANIFEST = D339_ATTEMPT2 / "d339_hull_manifest.json"
D339_ASSET_MANIFEST = D339_ATTEMPT2 / "d339_asset_build_manifest.json"
D339_LIVE_AUDIT = D339_DIR / "d339_live_collider_audit.json"
D339_COLD1 = {
    "link5": D339_ATTEMPT2 / "d339_link5_cold1_canonical_geometry.json",
    "gripper_link": D339_ATTEMPT2 / "d339_gripper_link_cold1_canonical_geometry.json",
}
D339_COLD2 = {
    "link5": D339_ATTEMPT2 / "d339_link5_cold2_canonical_geometry.json",
    "gripper_link": D339_ATTEMPT2 / "d339_gripper_link_cold2_canonical_geometry.json",
}

D340_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d340"
D340_CANDIDATES = D340_DIR / "d340_capture_fixed_point_candidates.json"
D340_POSTRUN = D340_DIR / "d340_capture_postrun_root_cause_audit.json"
D340_SUMMARY = D340_DIR / "d340_capture_summary.json"
D340_PARAMETER_AUDIT = D340_DIR / "d340_parameter_freeze_audit.json"
D340_SESSION = REPO_ROOT / "claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md"
D342_SESSION = REPO_ROOT / "claudedocs/session_20260713_grasp_g0a_d342_authored_coordinate_stream_repair.md"
START_HERE = REPO_ROOT / "START_HERE.md"

NEW_VARIABLES = ["authored_geometry_frame_contract"]
FAILING_PARTS = {
    "link5": (
        "part_011",
        "part_018",
        "part_023",
        "part_024",
        "part_040",
        "part_041",
        "part_045",
        "part_054",
    ),
    "gripper_link": (
        "part_000",
        "part_035",
        "part_036",
        "part_048",
        "part_057",
    ),
}
PART_KEYS = tuple(
    (body, name)
    for body in ("link5", "gripper_link")
    for name in FAILING_PARTS[body]
)
PART_COUNT = len(PART_KEYS)
NUMERIC_TOL_M = 1.0e-9
NEGATIVE_PERTURBATION_M = 10.0e-6
NEGATIVE_PART = ("link5", "part_011")

METRIC_NAMES = (
    "direct_points_f32_match",
    "direct_vertex_stream_hash_match",
    "direct_topology_hash_match",
    "direct_geometry_hash_match",
    "legacy_mapped_geometry_hash_match_diagnostic",
    "prim_to_body_identity_delta_m",
    "mapped_vs_d340_x0_assignment_delta_m",
    "mapped_x0_surface_delta_m",
    "candidate_containment_violation_m",
    "x0_to_candidate_shrink_distance_m",
)

DIRECT_COLOR = [135, 135, 135, 80]
MAPPED_X0_COLOR = [35, 120, 255, 92]
CANDIDATE_COLOR = [35, 205, 90, 115]

VERDICT_NUMERIC_PASS = "D342_AUTHORED_COORDINATE_STREAM_NUMERIC_SUPPORTED_OBSERVABILITY_PENDING"
VERDICT_FAIL = "D342_AUTHORED_COORDINATE_STREAM_CONTRACT_FAIL_STOP"
VERDICT_OBSERVABILITY_FAIL = "D342_AUTHORED_COORDINATE_STREAM_OBSERVABILITY_FAIL_STOP"
VERDICT_AUTOMATED_PENDING = (
    "D342_AUTHORED_COORDINATE_STREAM_AUTOMATED_PASS_MANUAL_INSPECTION_PENDING"
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(value, encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _f32_bytes(points: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(points, dtype="<f4")).tobytes()


def _f8_bytes(points: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(points, dtype="<f8")).tobytes()


def _i8_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": _relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(payload)


def _source_hashes() -> dict[str, str]:
    paths = {
        "d342_harness": Path(__file__).resolve(),
        "viz_debug": REPO_ROOT / "roarm_rl/viz_debug.py",
        "rerun_contract": REPO_ROOT / "roarm_rl/rerun_contract.py",
        "rerun_contract_tests": REPO_ROOT / "tests/test_viz_debug_rerun_contract.py",
        "d339_harness": REPO_ROOT
        / "sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py",
        "d339_hull_manifest": D339_MANIFEST,
        "d339_asset_manifest": D339_ASSET_MANIFEST,
        "d339_live_audit": D339_LIVE_AUDIT,
        "d339_physics_usd": D339_PHYSICS,
        "d340_candidates": D340_CANDIDATES,
        "d340_postrun": D340_POSTRUN,
        "d340_summary": D340_SUMMARY,
        "d340_parameter_audit": D340_PARAMETER_AUDIT,
    }
    for body in ("link5", "gripper_link"):
        paths[f"d339_{body}_cold1"] = D339_COLD1[body]
        paths[f"d339_{body}_cold2"] = D339_COLD2[body]
    return {name: sha256_file(path) for name, path in paths.items()}


def _cold_rows(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for body, path in paths.items():
        for row in _json(path)["parts"]:
            rows[(body, str(row["name"]))] = row
    return rows


def _raw_stream_baselines() -> dict[str, str]:
    cold1 = _cold_rows(D339_COLD1)
    return {
        f"{body}/{name}": _hash_bytes(
            _f32_bytes(np.asarray(cold1[(body, name)]["vertices_m"], dtype=np.float32))
        )
        for body, name in PART_KEYS
    }


def _expected_rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities = {"metadata/run", "events/frame_contract"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/frame_contract": ["TextLog:level", "TextLog:text"],
    }
    transform_components = [
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
    for body in ("link5", "gripper_link"):
        path = f"coordinate_frames/{body}/body_local"
        entities.add(path)
        components[path] = transform_components
    for body, name in PART_KEYS:
        frame_path = f"coordinate_frames/{body}/{name}/usd_prim_local"
        entities.add(frame_path)
        components[frame_path] = transform_components
        for variant in ("direct_authored", "body_mapped_x0", "body_mapped_x1"):
            path = f"frame_contract/{variant}/{body}/parts/{name}"
            metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
            entities.update({path, metadata_path})
            components[path] = mesh_components
            components[metadata_path] = ["TextDocument:text"]
        for metric in METRIC_NAMES:
            path = f"metrics/{body}/{name}/{metric}"
            entities.add(path)
            components[path] = ["Scalars:scalars"]
        gate_path = f"gate/{body}/{name}/frame_contract_pass"
        entities.add(gate_path)
        components[gate_path] = ["Scalars:scalars"]
    return sorted(entities), components


def _rrd_contract_digest() -> str:
    entities, components = _expected_rrd_contract()
    payload = {
        "exact_non_system_entity_paths": entities,
        "exact_timeline_names": ["blueprint", "log_time", "part_idx"],
        "required_components_by_path": components,
    }
    return _hash_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _preflight() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not OUT_DIR.is_dir():
        raise RuntimeError(f"missing registered output folder {OUT_DIR}")
    allowed = {
        PREREG_PATH.name,
        PREREG_AMENDMENT_PATH.name,
        PARAMETER_AUDIT_PATH.name,
    }
    observed = {path.name for path in OUT_DIR.iterdir()}
    if observed != allowed:
        raise RuntimeError(f"D342 output folder is not pristine: {sorted(observed)}")
    base_prereg = _json(PREREG_PATH)
    amendment = _json(PREREG_AMENDMENT_PATH)
    prereg = dict(base_prereg)
    prereg["d342_session_sha256"] = amendment.get("updated_d342_session_sha256")
    prereg["start_here_sha256"] = amendment.get("updated_start_here_sha256")
    prereg["source_hashes"] = amendment.get("updated_source_hashes")
    parameter_audit = _json(PARAMETER_AUDIT_PATH)
    d340_frozen_parameters = _json(D340_PARAMETER_AUDIT)["frozen_parameters"]
    registered_decomposition = _json(D339_MANIFEST)["decomposition_parameters"]
    audited_decomposition = parameter_audit.get("frozen_parameters", {}).get(
        "decomposition", {}
    )
    audited_decomposition_subset = {
        key: audited_decomposition.get(key)
        for key in (
            "hull_vertex_limit",
            "max_convex_hulls",
            "voxel_resolution",
            "error_percentage",
            "min_thickness_m",
            "shrink_wrap",
        )
    }
    d339_before = _inventory(D339_ATTEMPT2)
    d340_before = _inventory(D340_DIR)
    exact_entities, _ = _expected_rrd_contract()
    checks = {
        "artifact": base_prereg.get("artifact")
        == "D342_AUTHORED_COORDINATE_STREAM_PREREGISTRATION_V1",
        "amendment_artifact": amendment.get("artifact")
        == "D342_PREREGISTRATION_AMENDMENT_001",
        "amendment_parent_sha256": amendment.get("parent_prereg_sha256")
        == sha256_file(PREREG_PATH),
        "amendment_launch_only_attempts": bool(
            amendment.get("scientific_execution_count_before_amendment") == 0
            and amendment.get("scientific_artifacts_created_before_amendment") == []
            and len(amendment.get("launch_only_attempts", [])) == 2
            and all(
                row.get("scientific_execution_started") is False
                and row.get("scientific_outputs_created") == []
                for row in amendment.get("launch_only_attempts", [])
            )
        ),
        "amendment_scope_only": bool(
            amendment.get("new_variables") == NEW_VARIABLES
            and amendment.get("scientific_contract_changes") == []
            and amendment.get("physical_variable_changes") == []
            and amendment.get("existing_parameter_increases") == []
            and amendment.get("existing_parameter_changes") == []
            and amendment.get("threshold_relaxations") == []
        ),
        "amendment_rrd_contract": amendment.get("rrd_contract_sha256")
        == base_prereg.get("rrd_contract_sha256")
        == _rrd_contract_digest(),
        "new_variables": prereg.get("new_variables") == NEW_VARIABLES,
        "git_head": prereg.get("git_head") == _git_head(),
        "d339_attempt2_inventory_digest": prereg.get("d339_attempt2_inventory_digest")
        == _inventory_digest(d339_before),
        "d340_inventory_digest": prereg.get("d340_inventory_digest")
        == _inventory_digest(d340_before),
        "d340_session_sha256": prereg.get("d340_session_sha256") == sha256_file(D340_SESSION),
        "d342_session_sha256": prereg.get("d342_session_sha256") == sha256_file(D342_SESSION),
        "start_here_sha256": prereg.get("start_here_sha256") == sha256_file(START_HERE),
        "parameter_audit_sha256": prereg.get("parameter_audit_sha256")
        == sha256_file(PARAMETER_AUDIT_PATH),
        "source_hashes": prereg.get("source_hashes") == _source_hashes(),
        "raw_stream_baselines": prereg.get("raw_points_f32_sha256") == _raw_stream_baselines(),
        "rerun_version": str(rr.__version__) == RERUN_CONTRACT_VERSION == prereg.get("rerun_version"),
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "scipy_pin": str(scipy.__version__) == prereg.get("scipy_version"),
        "trimesh_pin": str(trimesh.__version__) == prereg.get("trimesh_version"),
        "part_allowlist": prereg.get("part_allowlist")
        == {body: list(FAILING_PARTS[body]) for body in ("link5", "gripper_link")},
        "rrd_contract_sha256": prereg.get("rrd_contract_sha256") == _rrd_contract_digest(),
        "exact_entity_count": prereg.get("scientific_subject_counts", {}).get(
            "exact_non_system_entities"
        )
        == len(exact_entities),
        "parameter_freeze_pass": parameter_audit.get("pass") is True,
        "parameter_audit_identity": bool(
            parameter_audit.get("artifact") == "D342_PARAMETER_FREEZE_AUDIT_V1"
            and parameter_audit.get("status") == "PRE_RUNTIME_LOCKED"
            and parameter_audit.get("verdict")
            == "NO_EXISTING_PARAMETER_INCREASE_OR_CHANGE"
        ),
        "parameter_audit_variable_identity": bool(
            parameter_audit.get("new_variables") == NEW_VARIABLES
            and parameter_audit.get("measurement_only_variables") == NEW_VARIABLES
            and parameter_audit.get("variable_count") == 1
            and parameter_audit.get("physical_variables_changed") == []
        ),
        "decomposition_values_equal_d339": audited_decomposition_subset
        == registered_decomposition,
        "target_and_control_equal_d340": parameter_audit.get("frozen_parameters", {}).get(
            "target_and_control"
        )
        == d340_frozen_parameters["target_and_control"],
        "representation_gates_equal_d340": parameter_audit.get("frozen_parameters", {}).get(
            "representation_gates"
        )
        == d340_frozen_parameters["representation_gates"],
        "readback_tolerances_equal_d340": parameter_audit.get("frozen_parameters", {}).get(
            "readback_tolerances"
        )
        == d340_frozen_parameters["readback_tolerances"],
        "parameter_increase_lists_empty": all(
            parameter_audit.get(key) == []
            for key in (
                "existing_parameter_increases",
                "existing_parameter_changes",
                "decomposition_parameter_changes",
                "threshold_relaxations",
                "target_controller_solver_changes",
                "object_table_mass_changes",
                "collision_asset_writes",
                "physical_variables_changed",
            )
        ),
        "runtime_scope_zero": parameter_audit.get("runtime_scope")
        == {
            "offline_usdc_read_only": True,
            "recook_requests": 0,
            "simulation_context_created": False,
            "controlled_physics_steps": 0,
            "attempt3_authoring": False,
            "settle_or_ten_trial": False,
        },
        "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D342 preregistration gate failed: {checks}")
    return prereg, checks, d339_before, d340_before


def _read_authored_usdc() -> dict[tuple[str, str], dict[str, Any]]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(D339_PHYSICS), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open immutable D339 physics layer: {D339_PHYSICS}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for body, name in PART_KEYS:
        path = f"/colliders/{body}/d338_convex_parts/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"missing immutable authored part {path}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(
            [[float(value) for value in point] for point in list(mesh.GetPointsAttr().Get() or [])],
            dtype=np.float32,
        )
        counts = np.asarray(list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype=np.int64)
        indices = np.asarray(list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype=np.int64)
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        rows[(body, name)] = {
            "path": path,
            "points": points,
            "counts": counts,
            "indices": indices,
            "points_f32_sha256": _hash_bytes(_f32_bytes(points)),
            "counts_i8_sha256": _hash_bytes(_i8_bytes(counts)),
            "indices_i8_sha256": _hash_bytes(_i8_bytes(indices)),
            "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
            "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
            "collision_enabled": True if collision_enabled is None else bool(collision_enabled),
            "approximation": str(mesh_api.GetApproximationAttr().Get()),
            "hull_vertex_limit": int(hull_api.GetHullVertexLimitAttr().Get()),
            "min_thickness_m": float(hull_api.GetMinThicknessAttr().Get()),
        }
    return rows


def _manifest_rows(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (body, str(row["name"])): row
        for body in ("link5", "gripper_link")
        for row in manifest["parts"][body]
    }


def _candidate_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["body"]), str(row["name"])): row for row in payload["parts"]}


def _live_transform_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for body in ("link5", "gripper_link"):
        for row in payload["per_body"][body]["direct_cooks"]:
            name = Path(str(row["path"])).name
            if (body, name) in PART_KEYS:
                rows[(body, name)] = row["prim_to_body_transform"]
    return rows


def _public_convex(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertices": np.asarray(value["vertices_m"], dtype=np.float64),
        "triangles": np.asarray(value["triangles"], dtype=np.int64),
        "vertex_count": int(value["vertex_count"]),
        "triangle_count": int(value["triangle_count"]),
        "geometry_sha256": str(value["geometry_sha256"]),
        "vertex_stream_sha256": str(value["vertex_stream_sha256"]),
        "topology_sha256": str(value["topology_sha256"]),
    }


def _assignment_max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2 or aa.shape[1] != 3:
        return math.inf
    pairwise_l2 = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(pairwise_l2)
    return float(np.max(np.abs(aa[rows] - bb[cols])))


def _frame_quaternion(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
    u, _, vh = np.linalg.svd(rotation)
    rigid = u @ vh
    if np.linalg.det(rigid) < 0.0:
        u[:, -1] *= -1.0
        rigid = u @ vh
    quaternion = Rotation.from_matrix(rigid).as_quat()
    reconstructed = np.eye(4, dtype=np.float64)
    reconstructed[:3, :3] = rigid
    reconstructed[:3, 3] = matrix[:3, 3]
    residual = float(np.max(np.abs(reconstructed - matrix)))
    return quaternion, residual


def _evaluate(
    authored: dict[tuple[str, str], dict[str, Any]],
    prereg: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest_payload = _json(D339_MANIFEST)
    manifest = _manifest_rows(manifest_payload)
    cold1 = _cold_rows(D339_COLD1)
    cold2 = _cold_rows(D339_COLD2)
    candidates = _candidate_rows(_json(D340_CANDIDATES))
    transforms = _live_transform_rows(_json(D339_LIVE_AUDIT))
    part_evidence: dict[tuple[str, str], dict[str, Any]] = {}

    for part_idx, key in enumerate(PART_KEYS):
        body, name = key
        direct = authored[key]
        expected = manifest[key]
        first = cold1[key]
        second = cold2[key]
        d340_row = candidates[key]
        x0 = _public_convex(d340_row["fixed_point"]["authored_x0"])
        consensus = _public_convex(d340_row["fixed_point"]["channel_consensus"]["consensus"])
        candidate = _public_convex(d340_row["fixed_point"]["candidate_x1"])

        points = np.asarray(direct["points"], dtype=np.float32)
        counts = np.asarray(direct["counts"], dtype=np.int64)
        indices = np.asarray(direct["indices"], dtype=np.int64)
        triangles = indices.reshape((-1, 3)) if len(indices) % 3 == 0 else np.empty((0, 3), dtype=np.int64)
        cold1_points = np.asarray(first["vertices_m"], dtype=np.float32)
        cold1_triangles = np.asarray(first["triangles"], dtype=np.int64)
        cold2_points = np.asarray(second["vertices_m"], dtype=np.float32)
        cold2_triangles = np.asarray(second["triangles"], dtype=np.int64)

        vertex_bytes = _f8_bytes(points.astype(np.float64))
        topology_bytes = _i8_bytes(triangles)
        direct_vertex_hash = _hash_bytes(vertex_bytes)
        direct_topology_hash = _hash_bytes(topology_bytes)
        direct_geometry_hash = _hash_bytes(vertex_bytes + topology_bytes)
        direct_checks = {
            "direct_points_f32_array_exact_cold1": np.array_equal(points, cold1_points),
            "direct_points_f32_bytes_exact_preregistered": direct["points_f32_sha256"]
            == prereg["raw_points_f32_sha256"][f"{body}/{name}"],
            "face_counts_all_triangles": bool(len(counts) and np.all(counts == 3)),
            "face_indices_shape_valid": bool(len(indices) == 3 * len(counts)),
            "direct_triangles_exact_cold1": np.array_equal(triangles, cold1_triangles),
            "cold1_cold2_points_f32_exact": np.array_equal(cold1_points, cold2_points),
            "cold1_cold2_triangles_exact": np.array_equal(cold1_triangles, cold2_triangles),
            "vertex_count_matches_manifest": len(points) == int(expected["vertex_count"]),
            "triangle_count_matches_manifest": len(triangles) == int(expected["triangle_count"]),
            "direct_vertex_stream_hash_matches_manifest": direct_vertex_hash
            == expected["vertex_stream_sha256"],
            "direct_topology_hash_matches_manifest": direct_topology_hash
            == expected["topology_sha256"],
            "direct_geometry_hash_matches_manifest": direct_geometry_hash
            == expected["geometry_sha256"],
            "collision_enabled": bool(direct["collision_enabled"]),
            "convex_hull_approximation": direct["approximation"] == "convexHull",
            "hull_vertex_limit_frozen_64": int(direct["hull_vertex_limit"]) == 64,
            "min_thickness_frozen_1e_4m": math.isclose(
                float(direct["min_thickness_m"]), 0.0001, rel_tol=0.0, abs_tol=1.0e-12
            ),
        }

        transform = transforms[key]
        matrix = np.asarray(transform["matrix_row_major"], dtype=np.float64)
        mapped_from_direct = (matrix[:3, :3] @ points.astype(np.float64).T).T + matrix[:3, 3]
        mapped_convex = d339._canonical_convex(mapped_from_direct)
        mapped_surface = d339._convex_solid_hausdorff_m(mapped_convex, x0)
        assignment_delta = _assignment_max_abs_delta(mapped_from_direct, x0["vertices"])
        candidate_containment = d339._directed_convex_solid_distance_m(
            candidate["vertices"], x0
        )
        x0_to_candidate = d339._directed_convex_solid_distance_m(x0["vertices"], candidate)
        consensus_candidate_surface = d339._convex_solid_hausdorff_m(consensus, candidate)[
            "symmetric_m"
        ]
        quaternion, rigid_representation_residual = _frame_quaternion(matrix)
        legacy_mapped_hash_match = x0["geometry_sha256"] == expected["geometry_sha256"]
        numeric_checks = {
            "prim_to_body_transform_identity_le_1e_9m": float(
                transform["identity_max_abs_delta"]
            )
            <= NUMERIC_TOL_M,
            "rerun_rigid_frame_representation_residual_le_1e_9": rigid_representation_residual
            <= NUMERIC_TOL_M,
            "mapped_vs_d340_x0_assignment_delta_le_1e_9m": assignment_delta
            <= NUMERIC_TOL_M,
            "mapped_vs_d340_x0_surface_delta_le_1e_9m": float(mapped_surface["symmetric_m"])
            <= NUMERIC_TOL_M,
            "candidate_contained_in_mapped_x0_le_1e_9m": candidate_containment
            <= NUMERIC_TOL_M,
            "instance_prototype_consensus_exact": bool(
                d340_row["fixed_point"]["channel_consensus"]["pass"]
                and d340_row["fixed_point"]["channel_consensus"]["coordinate_max_abs_delta_m"]
                == 0.0
            ),
            "consensus_candidate_surface_le_1e_9m": consensus_candidate_surface
            <= NUMERIC_TOL_M,
            "candidate_strict_vertex_decrease": candidate["vertex_count"] < x0["vertex_count"],
            "d340_fixed_point_subevidence_pass": bool(d340_row["fixed_point"]["pass"]),
        }
        negative_domain_control = {
            "kind": "legacy post-transform exact-hash path",
            "legacy_mapped_geometry_hash_matches_manifest": legacy_mapped_hash_match,
            "expected_match": False,
            "rejected": not legacy_mapped_hash_match,
            "numeric_frame_equivalence_still_passes": all(numeric_checks.values()),
        }
        direct_pass = all(direct_checks.values())
        numeric_pass = all(numeric_checks.values())
        row_pass = bool(direct_pass and numeric_pass and negative_domain_control["rejected"])
        part_evidence[key] = {
            "part_idx": part_idx,
            "body": body,
            "name": name,
            "authored_path": direct["path"],
            "direct_stream_definition": (
                "ordered authored VtArray<Gf.Vec3f> as contiguous little-endian <f4; "
                "all-3 face counts and ordered face indices as contiguous <i8; no transform/sort/Qhull"
            ),
            "direct_hashes": {
                "points_f32_sha256": direct["points_f32_sha256"],
                "vertex_stream_f8_sha256": direct_vertex_hash,
                "topology_i8_sha256": direct_topology_hash,
                "geometry_f8_i8_sha256": direct_geometry_hash,
                "face_counts_i8_sha256": direct["counts_i8_sha256"],
                "face_indices_i8_sha256": direct["indices_i8_sha256"],
            },
            "expected_manifest_hashes": {
                "vertex_stream_sha256": expected["vertex_stream_sha256"],
                "topology_sha256": expected["topology_sha256"],
                "geometry_sha256": expected["geometry_sha256"],
            },
            "direct_checks": direct_checks,
            "direct_pass": direct_pass,
            "body_mapping": {
                "matrix_row_major": matrix.tolist(),
                "identity_max_abs_delta": float(transform["identity_max_abs_delta"]),
                "identity_tolerance_m": NUMERIC_TOL_M,
                "rerun_translation_m": matrix[:3, 3].tolist(),
                "rerun_quaternion_xyzw": quaternion.tolist(),
                "rerun_rigid_representation_residual": rigid_representation_residual,
            },
            "numeric_metrics": {
                "mapped_vs_d340_x0_assignment_max_abs_delta_m": assignment_delta,
                "mapped_vs_d340_x0_surface_delta_m": float(mapped_surface["symmetric_m"]),
                "candidate_containment_violation_m": candidate_containment,
                "x0_to_candidate_shrink_distance_m": x0_to_candidate,
                "consensus_candidate_surface_delta_m": consensus_candidate_surface,
                "initial_vertex_count": x0["vertex_count"],
                "candidate_vertex_count": candidate["vertex_count"],
            },
            "numeric_checks": numeric_checks,
            "numeric_pass": numeric_pass,
            "negative_domain_control": negative_domain_control,
            "pass": row_pass,
            "_display": {
                "direct_points": points.astype(np.float64),
                "direct_triangles": triangles,
                "mapped_x0": x0,
                "candidate_x1": candidate,
            },
        }

    negative_source = authored[NEGATIVE_PART]
    negative_points = np.asarray(negative_source["points"], dtype=np.float32).copy()
    vertex_index = int(np.argmax(negative_points[:, 0]))
    before_value = np.float32(negative_points[vertex_index, 0])
    negative_points[vertex_index, 0] = np.float32(
        negative_points[vertex_index, 0] + np.float32(NEGATIVE_PERTURBATION_M)
    )
    actual_delta = float(negative_points[vertex_index, 0] - before_value)
    negative_triangles = np.asarray(negative_source["indices"], dtype=np.int64).reshape((-1, 3))
    negative_vertex_bytes = _f8_bytes(negative_points.astype(np.float64))
    negative_topology_bytes = _i8_bytes(negative_triangles)
    negative_manifest = manifest[NEGATIVE_PART]
    original_convex = d339._canonical_convex(
        np.asarray(negative_source["points"], dtype=np.float32).astype(np.float64)
    )
    perturbed_convex = d339._canonical_convex(negative_points.astype(np.float64))
    negative_surface = d339._convex_solid_hausdorff_m(original_convex, perturbed_convex)[
        "symmetric_m"
    ]
    perturbation = {
        "kind": "in-memory-only outward +10um authored-coordinate perturbation",
        "body": NEGATIVE_PART[0],
        "name": NEGATIVE_PART[1],
        "vertex_index": vertex_index,
        "axis": "x",
        "registered_delta_m": NEGATIVE_PERTURBATION_M,
        "actual_float32_delta_m": actual_delta,
        "actual_delta_within_1pct_of_registered": bool(
            0.99 * NEGATIVE_PERTURBATION_M
            <= actual_delta
            <= 1.01 * NEGATIVE_PERTURBATION_M
        ),
        "source_file_written": False,
        "raw_points_f32_hash_changed": _hash_bytes(_f32_bytes(negative_points))
        != negative_source["points_f32_sha256"],
        "manifest_geometry_hash_rejected": _hash_bytes(
            negative_vertex_bytes + negative_topology_bytes
        )
        != negative_manifest["geometry_sha256"],
        "symmetric_surface_delta_m": float(negative_surface),
        "numeric_1e_9m_gate_rejected": float(negative_surface) > NUMERIC_TOL_M,
    }
    perturbation["pass"] = bool(
        perturbation["raw_points_f32_hash_changed"]
        and perturbation["manifest_geometry_hash_rejected"]
        and perturbation["numeric_1e_9m_gate_rejected"]
        and perturbation["actual_delta_within_1pct_of_registered"]
        and not perturbation["source_file_written"]
    )

    rows_public = []
    for key in PART_KEYS:
        row = part_evidence[key]
        rows_public.append({name: value for name, value in row.items() if name != "_display"})
    direct_pass_count = sum(row["direct_pass"] for row in part_evidence.values())
    numeric_pass_count = sum(row["numeric_pass"] for row in part_evidence.values())
    legacy_rejected_count = sum(
        row["negative_domain_control"]["rejected"] for row in part_evidence.values()
    )
    overall_pass = bool(
        len(part_evidence) == PART_COUNT
        and direct_pass_count == PART_COUNT
        and numeric_pass_count == PART_COUNT
        and legacy_rejected_count == PART_COUNT
        and perturbation["pass"]
    )
    evidence = {
        "artifact": "D342_AUTHORED_COORDINATE_STREAM_EVIDENCE_V1",
        "case": "g0a_d342",
        "new_variables": NEW_VARIABLES,
        "scientific_authority": (
            "immutable D339 USDC Vec3f/face arrays plus D339/D340 canonical JSON and hashes"
        ),
        "exact_hash_domain": "direct authored stream before any coordinate transform",
        "body_mapped_domain": "numeric containment/proximity only; exact hash is non-authoritative",
        "part_count": len(part_evidence),
        "direct_pass_count": direct_pass_count,
        "numeric_pass_count": numeric_pass_count,
        "legacy_mixed_stream_rejected_count": legacy_rejected_count,
        "parts": rows_public,
        "in_memory_perturbation": perturbation,
        "scientific_pass": overall_pass,
        "scientific_verdict": VERDICT_NUMERIC_PASS if overall_pass else VERDICT_FAIL,
        "scope_guards": {
            "collision_asset_writes": [],
            "recook_requests": 0,
            "simulation_context_created": False,
            "controlled_physics_steps": 0,
            "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
            "g0a_pass": False,
            "ladder_promoted": False,
        },
    }
    return evidence, part_evidence


def _rerun_rows(
    part_evidence: dict[tuple[str, str], dict[str, Any]],
    perturbation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = [
        {
            "frame": f"{body}_body_local",
            "parent_frame": "tf#/",
            "entity_path": f"coordinate_frames/{body}/body_local",
        }
        for body in ("link5", "gripper_link")
    ]
    meshes: list[dict[str, Any]] = []
    scalars: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        {
            "entity_path": "events/frame_contract",
            "text": "D342 reads direct authored Vec3f/face streams before any mapping",
            "level": "INFO",
            "sequence": {"part_idx": 0},
        }
    ]
    for key in PART_KEYS:
        row = part_evidence[key]
        body, name = key
        part_idx = int(row["part_idx"])
        display = row["_display"]
        frame_name = f"{body}_{name}_usd_prim_local"
        mapping = row["body_mapping"]
        frames.append(
            {
                "frame": frame_name,
                "parent_frame": f"{body}_body_local",
                "entity_path": f"coordinate_frames/{body}/{name}/usd_prim_local",
                "translation_m": mapping["rerun_translation_m"],
                "quaternion_xyzw": mapping["rerun_quaternion_xyzw"],
            }
        )
        geometry_rows = (
            (
                "direct_authored",
                display["direct_points"],
                display["direct_triangles"],
                frame_name,
                DIRECT_COLOR,
                row["direct_hashes"]["geometry_f8_i8_sha256"],
            ),
            (
                "body_mapped_x0",
                display["mapped_x0"]["vertices"],
                display["mapped_x0"]["triangles"],
                f"{body}_body_local",
                MAPPED_X0_COLOR,
                "non-authoritative-mapped-stream",
            ),
            (
                "body_mapped_x1",
                display["candidate_x1"]["vertices"],
                display["candidate_x1"]["triangles"],
                f"{body}_body_local",
                CANDIDATE_COLOR,
                "non-authoritative-candidate-stream",
            ),
        )
        for variant, vertices, triangles, frame, color, geometry_hash in geometry_rows:
            meshes.append(
                {
                    "entity_path": f"frame_contract/{variant}/{body}/parts/{name}",
                    "vertices_m": np.asarray(vertices, dtype=np.float64).copy(),
                    "triangles": np.asarray(triangles, dtype=np.int64).copy(),
                    "coordinate_frame": frame,
                    "body": body,
                    "part": name,
                    "source_kind": variant,
                    "geometry_sha256": geometry_hash,
                    "color_rgba": color,
                }
            )
        metrics = row["numeric_metrics"]
        metric_values = {
            "direct_points_f32_match": float(row["direct_checks"]["direct_points_f32_array_exact_cold1"]),
            "direct_vertex_stream_hash_match": float(
                row["direct_checks"]["direct_vertex_stream_hash_matches_manifest"]
            ),
            "direct_topology_hash_match": float(
                row["direct_checks"]["direct_topology_hash_matches_manifest"]
            ),
            "direct_geometry_hash_match": float(
                row["direct_checks"]["direct_geometry_hash_matches_manifest"]
            ),
            "legacy_mapped_geometry_hash_match_diagnostic": float(
                row["negative_domain_control"]["legacy_mapped_geometry_hash_matches_manifest"]
            ),
            "prim_to_body_identity_delta_m": float(mapping["identity_max_abs_delta"]),
            "mapped_vs_d340_x0_assignment_delta_m": float(
                metrics["mapped_vs_d340_x0_assignment_max_abs_delta_m"]
            ),
            "mapped_x0_surface_delta_m": float(metrics["mapped_vs_d340_x0_surface_delta_m"]),
            "candidate_containment_violation_m": float(metrics["candidate_containment_violation_m"]),
            "x0_to_candidate_shrink_distance_m": float(metrics["x0_to_candidate_shrink_distance_m"]),
        }
        if tuple(metric_values) != METRIC_NAMES:
            raise RuntimeError("D342 Rerun metric schema drift")
        for metric, value in metric_values.items():
            scalars.append(
                {
                    "entity_path": f"metrics/{body}/{name}/{metric}",
                    "value": value,
                    "sequence": {"part_idx": part_idx},
                }
            )
        scalars.append(
            {
                "entity_path": f"gate/{body}/{name}/frame_contract_pass",
                "value": float(row["pass"]),
                "sequence": {"part_idx": part_idx},
            }
        )
        events.append(
            {
                "entity_path": "events/frame_contract",
                "text": (
                    f"{body}/{name}: direct={row['direct_pass']} numeric={row['numeric_pass']} "
                    f"legacy_mixed_hash_rejected={row['negative_domain_control']['rejected']}"
                ),
                "level": "INFO" if row["pass"] else "WARN",
                "sequence": {"part_idx": part_idx},
            }
        )
    events.extend(
        [
            {
                "entity_path": "events/frame_contract",
                "text": (
                    "in-memory +10um negative control rejected by raw hash and 1e-9m proximity gate"
                    if perturbation["pass"]
                    else "in-memory +10um negative control FAILED to satisfy the registered rejection gate"
                ),
                "level": "INFO" if perturbation["pass"] else "WARN",
                "sequence": {"part_idx": 0},
            },
            {
                "entity_path": "events/frame_contract",
                "text": "D342 stop boundary: no attempt3, recook, asset mutation, physics, or G0a promotion",
                "level": "WARN",
                "sequence": {"part_idx": PART_COUNT - 1},
            },
        ]
    )
    return frames, meshes, scalars, events


def _immutability_report(
    d339_before: list[dict[str, Any]], d340_before: list[dict[str, Any]]
) -> dict[str, Any]:
    d339_after = _inventory(D339_ATTEMPT2)
    d340_after = _inventory(D340_DIR)
    return {
        "d339_file_count_before_after": [len(d339_before), len(d339_after)],
        "d339_digest_before_after": [
            _inventory_digest(d339_before),
            _inventory_digest(d339_after),
        ],
        "d339_exact_rows_equal": d339_before == d339_after,
        "d340_file_count_before_after": [len(d340_before), len(d340_after)],
        "d340_digest_before_after": [
            _inventory_digest(d340_before),
            _inventory_digest(d340_after),
        ],
        "d340_exact_rows_equal": d340_before == d340_after,
        "pass": d339_before == d339_after and d340_before == d340_after,
    }


def _automated_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# D342 authored-coordinate-stream automated report",
            "",
            f"- Automated verdict: `{summary['automated_verdict']}`",
            f"- Direct authored exact matches: `{summary['scientific_result']['direct_pass_count']}/13`",
            f"- Body-mapped numeric passes: `{summary['scientific_result']['numeric_pass_count']}/13`",
            f"- Legacy mixed-stream negatives rejected: `{summary['scientific_result']['legacy_mixed_stream_rejected_count']}/13`",
            f"- In-memory perturbation rejected: `{summary['scientific_result']['in_memory_perturbation_pass']}`",
            f"- Rerun archive/render gate: `{summary['rerun']['validation']['pass']}`",
            f"- D339/D340 immutable: `{summary['immutability']['pass']}`",
            "",
            "The screenshot proves renderability only. A separate actual image inspection",
            "must close the completion contract. Attempt3 remains absent and separately approval-gated.",
            "",
        ]
    )


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    prereg, prereg_checks, d339_before, d340_before = _preflight()
    authored: dict[tuple[str, str], dict[str, Any]]
    app = None
    launcher = AppLauncher(args)
    app = launcher.app

    def _close_app_at_exit() -> None:
        if app is not None:
            app.close()

    # Isaac Kit's close path may terminate the interpreter.  Keep it registered
    # for exceptional exits, but do not invoke it until every scientific and
    # observability artifact has been written and flushed.
    atexit.register(_close_app_at_exit)
    authored = _read_authored_usdc()

    evidence, part_evidence = _evaluate(authored, prereg)
    _write_json(EVIDENCE_PATH, evidence)
    frames, meshes, scalars, events = _rerun_rows(
        part_evidence, evidence["in_memory_perturbation"]
    )
    exact_entities, expected_components = _expected_rrd_contract()
    subject_counts = {
        "parts": PART_COUNT,
        "coordinate_frame_entities": len(frames),
        "mesh_entities": len(meshes),
        "metric_scalar_rows": PART_COUNT * len(METRIC_NAMES),
        "gate_scalar_rows": PART_COUNT,
        "event_rows": len(events),
        "exact_non_system_entities": len(exact_entities),
    }
    if subject_counts != prereg["scientific_subject_counts"]:
        raise RuntimeError(
            f"D342 registered subject count drift: {subject_counts} != "
            f"{prereg['scientific_subject_counts']}"
        )

    recording_metadata = {
        "case": "g0a_d342",
        "purpose": "authored-coordinate-stream contract repair",
        "git_head": _git_head(),
        "new_variables": NEW_VARIABLES,
        "scientific_authority": (
            "immutable D339 USDC authored arrays plus D339/D340 canonical JSON and hashes"
        ),
        "viewer_geometry_role": "Float32 one-way spatial observability copies only",
        "exact_hash_domain": "direct authored Vec3f/face stream before transform",
        "mapped_domain": "numeric containment/proximity only",
        "physics": "forbidden / 0 steps",
        "collision_asset_mutation": "forbidden / none",
        "attempt3": "absent / separately approval-gated",
        "q5_convention": "0=CLOSED; 1.5413rad=OPEN",
    }
    log_status = log_rerun(
        GOOD_RRD,
        coordinate_frames=frames,
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata=recording_metadata,
        recording_id="g0a_d342_authored_coordinate_stream",
        blueprint_path=GOOD_RBL,
        blueprint_mode="authored_frame_contract",
        live_viewer=False,
        app_id="roarm_g0a_authored_frame_contract",
    )
    validation = (
        validate_rerun_artifact(
            GOOD_RRD,
            expected_entity_paths=[
                "frame_contract/direct_authored/link5/parts/part_011",
                "frame_contract/body_mapped_x0/link5/parts/part_011",
                "frame_contract/body_mapped_x1/gripper_link/parts/part_057",
                "events/frame_contract",
            ],
            expected_timeline_names=["part_idx"],
            exact_entity_paths=exact_entities,
            exact_timeline_names=["blueprint", "log_time", "part_idx"],
            expected_entity_components=expected_components,
            blueprint_path=GOOD_RBL,
            screenshot_path=SCREENSHOT,
        )
        if log_status.get("ok", False)
        else {"pass": False, "errors": ["Rerun recording contract failed"]}
    )
    immutability = _immutability_report(d339_before, d340_before)
    parameter_audit = _json(PARAMETER_AUDIT_PATH)
    numeric_pass = bool(evidence["scientific_pass"])
    automated_pass = bool(
        numeric_pass
        and log_status.get("ok", False)
        and validation.get("pass", False)
        and immutability["pass"]
        and parameter_audit.get("pass") is True
        and not (D340_DIR / "collision_asset/attempt3").exists()
    )
    if automated_pass:
        verdict = VERDICT_AUTOMATED_PENDING
    elif numeric_pass:
        verdict = VERDICT_OBSERVABILITY_FAIL
    else:
        verdict = VERDICT_FAIL
    summary = {
        "artifact": "D342_AUTHORED_COORDINATE_STREAM_AUTOMATED_SUMMARY_V1",
        "case": "g0a_d342",
        "automated_verdict": verdict,
        "automated_pass": automated_pass,
        "completion_contract_pass": False,
        "manual_visual_inspection_pending": automated_pass,
        "new_variables": NEW_VARIABLES,
        "preregistration_checks": prereg_checks,
        "versions": {
            "rerun_sdk": str(rr.__version__),
            "numpy": str(np.__version__),
            "psutil": str(psutil.__version__),
            "scipy": str(scipy.__version__),
            "trimesh": str(trimesh.__version__),
        },
        "scientific_result": {
            "evidence_path": _relative(EVIDENCE_PATH),
            "evidence_sha256": sha256_file(EVIDENCE_PATH),
            "scientific_verdict": evidence["scientific_verdict"],
            "direct_pass_count": evidence["direct_pass_count"],
            "numeric_pass_count": evidence["numeric_pass_count"],
            "legacy_mixed_stream_rejected_count": evidence[
                "legacy_mixed_stream_rejected_count"
            ],
            "in_memory_perturbation_pass": evidence["in_memory_perturbation"]["pass"],
            "pass": numeric_pass,
        },
        "scientific_subject": subject_counts,
        "rerun": {
            "rrd_path": _relative(GOOD_RRD),
            "rbl_path": _relative(GOOD_RBL),
            "screenshot_path": _relative(SCREENSHOT),
            "log_status": log_status,
            "validation": validation,
        },
        "immutability": immutability,
        "parameter_audit": {
            "path": _relative(PARAMETER_AUDIT_PATH),
            "sha256": sha256_file(PARAMETER_AUDIT_PATH),
            "pass": parameter_audit.get("pass") is True,
            "existing_parameter_increases": parameter_audit.get(
                "existing_parameter_increases"
            ),
            "existing_parameter_changes": parameter_audit.get("existing_parameter_changes"),
        },
        "scope_guards": {
            "isaac_kit_app_started_for_offline_usdc_read": True,
            "simulation_context_created": False,
            "recook_requests": 0,
            "collision_asset_writes": [],
            "controlled_physics_steps": 0,
            "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
            "g0a_pass": False,
            "ladder_promoted": False,
        },
        "next_gate": (
            "Open and inspect the registered screenshot, then write a separate manual report and "
            "final completion summary. Even after PASS, attempt3 authoring/fresh validation requires "
            "a separate user-approved case."
        ),
    }
    _write_json(AUTOMATED_SUMMARY, summary)
    _write_text(AUTOMATED_REPORT, _automated_markdown(summary))
    exit_code = 0 if automated_pass else 2
    print(
        json.dumps(
            {
                "automated_verdict": verdict,
                "direct_pass_count": evidence["direct_pass_count"],
                "numeric_pass_count": evidence["numeric_pass_count"],
                "legacy_negative_count": evidence["legacy_mixed_stream_rejected_count"],
                "perturbation_pass": evidence["in_memory_perturbation"]["pass"],
                "rerun_pass": validation.get("pass", False),
                "immutable": immutability["pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    atexit.unregister(_close_app_at_exit)
    if app is not None:
        app.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "verdict": VERDICT_FAIL,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise
