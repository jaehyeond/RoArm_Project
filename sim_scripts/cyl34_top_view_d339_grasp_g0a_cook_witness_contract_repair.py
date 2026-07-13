#!/usr/bin/env python3
"""D339 callback-first cook-witness repair for cylinder G0a.

The only new variable is the measurement contract used to witness two direct
convex-decomposition requests.  Every D338 physical/decomposition setting is
frozen.  Callback evidence is persisted before classification; global cooking
statistics are informational; no physics is licensed until the two distinct
stage requests return canonically identical geometry and the derivative passes
the unchanged D338 live representation gate.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, log_rerun
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332
from sim_scripts import cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333
from sim_scripts import cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334
from sim_scripts import cyl34_top_view_d335_grasp_g0a_target_family_repair as d335
from sim_scripts import cyl34_top_view_d336_grasp_g0a_finite_grid_caveat_discriminator as d336
from sim_scripts import cyl34_top_view_d337_grasp_g0a_open_jaw_target_gate as d337
from sim_scripts import cyl34_top_view_d338_grasp_g0a_collision_representation_repair as d338


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d339"
D337_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d337/g0a_d337_open_jaw_target_gate_summary.json"
)
PIN_D337_SUMMARY_SHA256 = "80df2f0b3765faee5bbeb190ded03bc326d54602fe16bf5c8fd73513fe5500d4"
D338_SCRIPT = REPO / "sim_scripts/cyl34_top_view_d338_grasp_g0a_collision_representation_repair.py"
D338_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d338/g0a_d338_collision_representation_repair_summary.json"
)
D338_ATTEMPT1_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d338/collision_asset/attempt1"
)
PIN_D338_SCRIPT_SHA256 = "f3d330a9a5ca6f886728d0e5dc8037baa68d83a2b911aa105904d7d369ead426"
PIN_D338_SUMMARY_SHA256 = "0bda3990751253a7c50408b0106cdc9e3504a35e6ac4f72a9504de0b90aa9a1e"
PIN_D338_ATTEMPT1_SHA256 = {
    "d338_asset_build_manifest.json": "8fec513a4e344132f4e445061bbc383da2d6347f5e5883b4c53b2695da1acdda",
    "d338_invocation_abort_001.json": "f168087ac672e2bffdd9fdf29b6200afcd69e6551e4fd9e891b1e8f7695c8d42",
    "kit_20260713_114440.log": "075ce099543ae952e362c47e44894e92abb63e72610c274441e6a82690a87a6b",
}
PIN_SOURCE_LAYER_SHA256 = {
    "root": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "base": "ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841",
    "physics": "1df07d387da76dcde4cd700ee1f9546cba25965776a9700897314ef884c37ed2",
    "robot": "2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d",
    "sensor": "3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f",
}
PIN_STL_SHA256 = {
    "link5": "1d63f374a78c1419b21eec63fa8efeef40d0d42ca89c5de3ceb0d86476d9c7eb",
    "gripper_link": "7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56",
    "excluded_g2a": "bd34df3187305c3a18d572ce5c4a37e3144684cce45ee1d03ee3435b37a6d40a",
}
SOURCE_LAYER_REL = {
    "root": "roarm_m3.usd",
    "base": "configuration/roarm_m3_base.usd",
    "physics": "configuration/roarm_m3_physics.usd",
    "robot": "configuration/roarm_m3_robot.usd",
    "sensor": "configuration/roarm_m3_sensor.usd",
}
SOURCE_ASSET_PATHS = {
    "link5": {
        "body": "/roarm_m3/link5",
        "collider": "/roarm_m3/link5/collisions/link5/node_STL_BINARY_",
        "mesh": "/roarm_m3/link5/collisions/link5/node_STL_BINARY_/mesh",
        "source_spec": "/colliders/link5/link5/node_STL_BINARY_",
        "parts_parent": "/colliders/link5/d338_convex_parts",
    },
    "gripper_link": {
        "body": "/roarm_m3/gripper_link",
        "collider": "/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_",
        "mesh": "/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh",
        "source_spec": "/colliders/gripper_link/gripper_link/node_STL_BINARY_",
        "parts_parent": "/colliders/gripper_link/d338_convex_parts",
    },
}
LIVE_OLD_COLLIDER_PATHS = {
    "link5": "/World/envs/env_0/Robot/link5/collisions/link5/node_STL_BINARY_",
    "gripper_link": (
        "/World/envs/env_0/Robot/gripper_link/collisions/gripper_link/node_STL_BINARY_"
    ),
}
LIVE_PART_PARENT_PATHS = {
    "link5": "/World/envs/env_0/Robot/link5/collisions/d338_convex_parts",
    "gripper_link": (
        "/World/envs/env_0/Robot/gripper_link/collisions/d338_convex_parts"
    ),
}

Q5_OPEN_RAD = 1.5413
OLD_RADIAL_NM = 7_000_000
OLD_TANGENT_NM = 11_000_000
TARGET_SETTLE_STEPS = 200
RAW_ANCHOR_TOL_MM = 0.05
TASK_FIDELITY_TOL_MM = 0.5
CLEAR_GATE_MM = 0.1
LIVE_VOLUME_PARITY_REL_TOL = 0.005
PROPERTY_VOLUME_BINDING_REL_TOL = 0.05
LIVE_SURFACE_PARITY_TOL_M = 0.0001
COLD_COOK_COORD_TOL_M = 1.0e-9

DECOMPOSITION_PARAMS = {
    "hull_vertex_limit": 64,
    "max_convex_hulls": 64,
    "voxel_resolution": 1_000_000,
    "error_percentage": 1.0,
    "min_thickness_m": 0.0001,
    "shrink_wrap": True,
}

VERDICT_WITNESS_FAIL = "D339_G0A_COOK_WITNESS_CONTRACT_FAIL_STOP"
VERDICT_BUILD_FAIL = "D339_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP"
VERDICT_CONTRACT_FAIL = "D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP"
VERDICT_NOT_CLEAR = "D339_G0A_REPAIRED_COOKED_TARGET_NOT_CLEAR_STOP"
VERDICT_STATIC_PASS = "D339_G0A_COLLISION_REPRESENTATION_STATIC_SUPPORTED_STOP"
VERDICT_STATIC_MIXED = "D339_G0A_STATIC_RUNTIME_MIXED_STOP"
VERDICT_VIZ_FAIL = "D339_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _sha256(path: Path) -> str:
    return d332._sha256(path)


def _d338_attempt1_integrity() -> dict[str, Any]:
    inventory = sorted(
        path.name for path in D338_ATTEMPT1_DIR.iterdir()
    ) if D338_ATTEMPT1_DIR.is_dir() else []
    observed = {
        name: _sha256(D338_ATTEMPT1_DIR / name)
        for name in inventory
        if (D338_ATTEMPT1_DIR / name).is_file()
    }
    checks = {
        "directory_exists": D338_ATTEMPT1_DIR.is_dir(),
        "exact_inventory": inventory == sorted(PIN_D338_ATTEMPT1_SHA256),
        "exact_hashes": observed == PIN_D338_ATTEMPT1_SHA256,
    }
    return {
        "inventory": inventory,
        "sha256": observed,
        "expected_sha256": PIN_D338_ATTEMPT1_SHA256,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _canonical_triangles(
    vertices: np.ndarray, simplices: np.ndarray, outward_normals: np.ndarray
) -> np.ndarray:
    """Canonicalize face order while preserving an outward winding.

    Sorting the three indices of a triangle destroys its winding.  Qhull's
    facet equations provide the outward normal corresponding to each simplex,
    so orient each face against that normal, cyclically rotate it to a stable
    first index, and only then sort the *rows*.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(simplices, dtype=np.int64)
    normals = np.asarray(outward_normals, dtype=np.float64)
    if faces.shape != normals.shape or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(
            f"invalid convex facet arrays: faces={faces.shape}, normals={normals.shape}"
        )
    canonical = []
    for face, normal in zip(faces, normals, strict=True):
        row = [int(value) for value in face]
        cross = np.cross(verts[row[1]] - verts[row[0]], verts[row[2]] - verts[row[0]])
        if float(np.dot(cross, normal)) < 0.0:
            row[1], row[2] = row[2], row[1]
        first = int(np.argmin(row))
        row = row[first:] + row[:first]
        check = np.cross(verts[row[1]] - verts[row[0]], verts[row[2]] - verts[row[0]])
        if float(np.dot(check, normal)) <= 0.0:
            raise RuntimeError("failed to establish outward convex-face winding")
        canonical.append(row)
    rows = np.asarray(canonical, dtype=np.int64)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def _canonical_convex(vertices: np.ndarray) -> dict[str, Any]:
    from scipy.spatial import ConvexHull

    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 4:
        raise RuntimeError(f"invalid cooked convex vertex array: {verts.shape}")
    if not np.all(np.isfinite(verts)):
        raise RuntimeError("non-finite cooked convex vertices")
    order = np.lexsort((verts[:, 2], verts[:, 1], verts[:, 0]))
    verts = verts[order]
    keep = np.ones(len(verts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(verts, axis=0)) > 1.0e-12, axis=1)
    verts = verts[keep]
    hull = ConvexHull(verts)
    triangles = _canonical_triangles(
        verts,
        np.asarray(hull.simplices, dtype=np.int64),
        np.asarray(hull.equations[:, :3], dtype=np.float64),
    )
    vertex_bytes = np.ascontiguousarray(verts.astype("<f8")).tobytes()
    triangle_bytes = np.ascontiguousarray(triangles.astype("<i8")).tobytes()
    vertex_digest = hashlib.sha256(vertex_bytes).hexdigest()
    topology_digest = hashlib.sha256(triangle_bytes).hexdigest()
    digest = hashlib.sha256(vertex_bytes + triangle_bytes).hexdigest()
    return {
        "vertices": verts,
        "triangles": triangles,
        "vertex_count": int(len(verts)),
        "triangle_count": int(len(triangles)),
        "volume_m3": float(hull.volume),
        "bounds_m": np.vstack([verts.min(axis=0), verts.max(axis=0)]).tolist(),
        "centroid_m": verts.mean(axis=0).tolist(),
        "vertex_stream_sha256": vertex_digest,
        "topology_sha256": topology_digest,
        "geometry_sha256": digest,
    }


def _directed_convex_solid_distance_m(
    source_vertices: np.ndarray, target: dict[str, Any]
) -> float:
    """Exact-at-vertices directed distance between two convex solids.

    Distance to a convex set is convex, so its maximum over the source convex
    polytope is attained at a source vertex.  Target containment uses Qhull
    halfspaces; exterior point-to-surface distance uses triangle projection.
    """
    import trimesh
    from scipy.spatial import ConvexHull

    target_vertices = np.asarray(target["vertices"], dtype=np.float64)
    target_triangles = np.asarray(target["triangles"], dtype=np.int64)
    hull = ConvexHull(target_vertices)
    triangles = target_vertices[target_triangles]
    maximum = 0.0
    for point in np.asarray(source_vertices, dtype=np.float64):
        halfspace = hull.equations[:, :3] @ point + hull.equations[:, 3]
        if float(np.max(halfspace)) <= 1.0e-10:
            distance = 0.0
        else:
            tiled = np.repeat(point[None, :], len(triangles), axis=0)
            closest = trimesh.triangles.closest_point(triangles, tiled)
            distance = float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))
        maximum = max(maximum, distance)
    return maximum


def _convex_solid_hausdorff_m(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_to_b = _directed_convex_solid_distance_m(a["vertices"], b)
    b_to_a = _directed_convex_solid_distance_m(b["vertices"], a)
    value = max(a_to_b, b_to_a)
    return {
        "authored_to_live_m": a_to_b,
        "live_to_authored_m": b_to_a,
        "symmetric_m": value,
        "tolerance_m": LIVE_SURFACE_PARITY_TOL_M,
        "pass": value <= LIVE_SURFACE_PARITY_TOL_M,
    }


def _convex_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    centroid = np.asarray(row["centroid_m"], dtype=np.float64)
    bounds = np.asarray(row["bounds_m"], dtype=np.float64).reshape(-1)
    return (
        *np.round(centroid, 9).tolist(),
        *np.round(bounds, 9).tolist(),
        int(row["vertex_count"]),
        row["geometry_sha256"],
    )


def _callback_convex_payload(convex: Any) -> dict[str, Any]:
    vertices = [
        [float(vertex.x), float(vertex.y), float(vertex.z)]
        for vertex in list(convex.vertices)
    ]
    indices = [int(value) for value in list(convex.indices)]
    polygons = [
        {
            "index_base": int(polygon.index_base),
            "num_vertices": int(polygon.num_vertices),
            "plane": [float(value) for value in list(polygon.plane)],
        }
        for polygon in list(convex.polygons)
    ]
    return {
        "vertices": vertices,
        "indices": indices,
        "polygons": polygons,
        "vertex_count": len(vertices),
        "index_count": len(indices),
        "polygon_count": len(polygons),
    }


def _callback_payload_checks(payload: dict[str, Any]) -> dict[str, bool]:
    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    indices = [int(value) for value in payload["indices"]]
    polygons = payload["polygons"]
    vertex_count = int(payload["vertex_count"])
    spans_valid = all(
        int(row["index_base"]) >= 0
        and int(row["num_vertices"]) >= 3
        and int(row["index_base"]) + int(row["num_vertices"]) <= len(indices)
        for row in polygons
    )
    covered_positions = [
        position
        for row in polygons
        for position in range(
            int(row["index_base"]),
            int(row["index_base"]) + int(row["num_vertices"]),
        )
    ] if spans_valid else []
    referenced_indices_valid = bool(
        indices
        and all(0 <= value < vertex_count for value in indices)
    )
    plane_checks = []
    for row in polygons:
        plane = np.asarray(row["plane"], dtype=np.float64)
        plane_checks.append(
            bool(
                plane.shape == (4,)
                and np.all(np.isfinite(plane))
                and float(np.linalg.norm(plane[:3])) > 0.0
            )
        )
    return {
        "vertex_count_4_to_64": 4 <= vertex_count <= DECOMPOSITION_PARAMS["hull_vertex_limit"],
        "vertices_shape_finite": bool(
            vertices.shape == (vertex_count, 3) and np.all(np.isfinite(vertices))
        ),
        "indices_nonempty_in_range": referenced_indices_valid,
        "polygons_nonempty": bool(polygons),
        "polygon_spans_valid": spans_valid,
        "polygon_spans_exactly_cover_index_buffer": (
            sorted(covered_positions) == list(range(len(indices)))
        ),
        "polygon_planes_valid": bool(plane_checks and all(plane_checks)),
    }


def _request_convex_representation(
    stage: Any,
    prim_path: str,
    tag: str,
    witness_path: Path,
    witness_base: dict[str, Any],
) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, UsdUtils

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    invalid_before_insert = not stage_id.IsValid()
    if invalid_before_insert:
        stage_id = cache.Insert(stage)
    stage_identifier = str(stage.GetRootLayer().identifier)
    holder: dict[str, Any] = {"events": [], "raw_convexes": []}
    request_in_progress = True

    def _done(result: Any, convexes: list[Any]) -> None:
        convex_list = list(convexes)
        result_value = getattr(result, "value", None)
        if result_value is None:
            result_value = int(result)
        event = {
            "callback_ordinal": len(holder["events"]) + 1,
            "tag": tag,
            "callback_during_synchronous_request": bool(request_in_progress),
            "result_name": str(getattr(result, "name", "")),
            "result_value": int(result_value),
            "result_repr": repr(result),
            "convex_count": len(convex_list),
            "convexes": [],
            "serialization_errors": [],
        }
        holder["events"].append(event)
        holder["raw_convexes"].append(convex_list)
        for index, convex in enumerate(convex_list):
            try:
                event["convexes"].append(_callback_convex_payload(convex))
            except Exception as error:  # preserve enum/count even if one payload is malformed
                event["serialization_errors"].append(
                    {
                        "convex_index": index,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )

    request_return = None
    request_exception = None
    try:
        request_return = get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id.ToLongInt(),
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim_path),
            run_asynchronously=False,
            on_result=_done,
        )
    except Exception as error:  # noqa: BLE001 - retain any callback already received
        request_exception = {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        request_in_progress = False
    response = {
        "events": holder["events"],
        "_raw_convexes": holder["raw_convexes"],
        "callback_count": len(holder["events"]),
        "request_return_type": type(request_return).__name__,
        "request_return_repr": repr(request_return),
        "request_exception": request_exception,
        "stage_id": int(stage_id.ToLongInt()),
        "stage_id_valid": bool(stage_id.IsValid()),
        "stage_cache_id_invalid_before_insert": invalid_before_insert,
        "stage_identifier": stage_identifier,
        "result_valid_enum_name": PhysxCollisionRepresentationResult.RESULT_VALID.name,
        "result_valid_enum_value": int(PhysxCollisionRepresentationResult.RESULT_VALID.value),
    }
    _json_dump(
        witness_path,
        {
            "artifact": "D339_CALLBACK_FIRST_COOK_WITNESS",
            **witness_base,
            "request": {
                key: value for key, value in response.items() if not key.startswith("_")
            },
            "callback_evidence_persisted_inside_request_boundary": True,
            "classification_performed": False,
        },
    )
    return response


def _cooking_statistics() -> dict[str, int]:
    from omni.physx import get_physx_cooking_private_interface

    stats = get_physx_cooking_private_interface().get_cooking_statistics()
    fields = (
        "total_scheduled_tasks",
        "total_finished_tasks",
        "total_finished_cache_hit_tasks",
        "total_finished_cache_miss_tasks",
        "total_warnings_convex_polygon_limits_reached",
        "total_warnings_failed_gpu_compatibility",
    )
    return {field: int(getattr(stats, field)) for field in fields}


def _statistics_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after[key] - before[key]) for key in before}


@contextlib.contextmanager
def _isolated_cooking_settings() -> Any:
    import carb
    import omni.physx.bindings._physx as physx_bindings

    settings = carb.settings.acquire_settings_interface()
    keys = (
        physx_bindings.SETTING_UPDATE_TO_USD,
        physx_bindings.SETTING_UJITSO_COLLISION_COOKING,
        physx_bindings.SETTING_USE_LOCAL_MESH_CACHE,
    )
    saved = {key: settings.get(key) for key in keys}
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, False)
    settings.set(physx_bindings.SETTING_UJITSO_COLLISION_COOKING, False)
    settings.set(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE, False)
    evidence = {
        "settings_saved_values": saved,
        "update_to_usd_disabled": not bool(
            settings.get(physx_bindings.SETTING_UPDATE_TO_USD)
        ),
        "ujitso_collision_cooking_disabled": not bool(
            settings.get(physx_bindings.SETTING_UJITSO_COLLISION_COOKING)
        ),
        "local_mesh_cache_setting_disabled": not bool(
            settings.get(physx_bindings.SETTING_USE_LOCAL_MESH_CACHE)
        ),
        "settings_restored_after_request": False,
    }
    try:
        yield evidence
    finally:
        restore_errors = []
        for key, value in saved.items():
            try:
                settings.set(key, value)
            except Exception as error:  # settings restoration remains a hard recorded gate
                restore_errors.append(f"{key}: {type(error).__name__}: {error}")
        evidence["settings_restore_errors"] = restore_errors
        evidence["settings_restore_no_errors"] = not restore_errors
        restored_values = {}
        for key, value in saved.items():
            try:
                restored_values[key] = settings.get(key)
            except Exception as error:
                restore_errors.append(f"{key} readback: {type(error).__name__}: {error}")
        evidence["settings_restore_errors"] = restore_errors
        evidence["settings_restore_no_errors"] = not restore_errors
        evidence["settings_restored_values"] = restored_values
        evidence["settings_restored_after_request"] = bool(
            not restore_errors
            and all(restored_values.get(key) == value for key, value in saved.items())
        )


def _isolated_convex_request(
    stage: Any,
    mesh_path: str,
    tag: str,
    witness_path: Path,
    witness_base: dict[str, Any],
) -> dict[str, Any]:
    from omni.physx import (
        get_physx_cooking_interface,
        get_physx_cooking_private_interface,
    )

    with _isolated_cooking_settings() as settings_evidence:
        cooking = get_physx_cooking_interface()
        cooking_private = get_physx_cooking_private_interface()
        cooking.release_local_mesh_cache()
        cooking_private.release_runtime_mesh_cache()
        try:
            stats_before = {"available": True, "values": _cooking_statistics(), "error": None}
        except Exception as error:  # informational counters may not block the request
            stats_before = {
                "available": False,
                "values": None,
                "error": f"{type(error).__name__}: {error}",
            }
        result = _request_convex_representation(
            stage, mesh_path, tag, witness_path, witness_base
        )
        try:
            stats_after = {"available": True, "values": _cooking_statistics(), "error": None}
        except Exception as error:  # informational counters may not block the request
            stats_after = {
                "available": False,
                "values": None,
                "error": f"{type(error).__name__}: {error}",
            }
    stats_delta = (
        _statistics_delta(stats_before["values"], stats_after["values"])
        if stats_before["available"] and stats_after["available"]
        else None
    )
    return {
        "result": result,
        "settings": settings_evidence,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "stats_delta": stats_delta,
        "statistics_role": "informational_only_non_gating",
        "cache_release": {
            "local_mesh_cache_released_without_exception": True,
            "runtime_mesh_cache_released_without_exception": True,
        },
    }


def _isolated_settings_pass(evidence: dict[str, Any]) -> bool:
    return bool(
        evidence.get("update_to_usd_disabled")
        and evidence.get("ujitso_collision_cooking_disabled")
        and evidence.get("local_mesh_cache_setting_disabled")
        and evidence.get("settings_restored_after_request")
        and evidence.get("settings_restore_no_errors")
        and not evidence.get("settings_restore_errors")
    )


def _source_meshes_from_original(asset_root: Path, d334_summary: dict[str, Any]) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset_root), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open source asset {asset_root}")
    records: dict[str, Any] = {}
    for body, paths in SOURCE_ASSET_PATHS.items():
        mesh_prim = stage.GetPrimAtPath(paths["mesh"])
        body_prim = stage.GetPrimAtPath(paths["body"])
        if not mesh_prim.IsValid() or not body_prim.IsValid():
            raise RuntimeError(f"missing source mesh/body for {body}")
        mesh = UsdGeom.Mesh(mesh_prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        counts = [int(v) for v in list(mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(v) for v in list(mesh.GetFaceVertexIndicesAttr().Get() or [])]
        triangles, fan_used = d334._triangulate(counts, indices)
        mesh_l2w = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_w2l = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).GetInverse()
        verts = np.asarray(
            [
                [
                    float(x)
                    for x in body_w2l.Transform(
                        mesh_l2w.Transform(Gf.Vec3d(*[float(v) for v in point]))
                    )
                ]
                for point in points
            ],
            dtype=np.float64,
        )
        expected = next(row for row in d334_summary["shapes"] if row["body"] == body)[
            "source_mesh"
        ]
        checks = {
            "vertex_count": len(verts) == int(expected["vertex_count"]),
            "face_count": len(counts) == int(expected["face_count"]),
            "triangle_count": len(triangles) == int(expected["triangle_count"]),
            "fan_triangulated": fan_used == bool(expected["fan_triangulated"]),
            "body_local_bounds_bit_near": bool(
                np.allclose(
                    np.vstack([verts.min(axis=0), verts.max(axis=0)]),
                    np.asarray(expected["body_local_bounds_m"], dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
        }
        records[body] = {
            "checks": checks,
            "pass": all(checks.values()),
            "vertex_count": int(len(verts)),
            "face_count": int(len(counts)),
            "triangle_count": int(len(triangles)),
            "body_local_bounds_m": np.vstack([verts.min(axis=0), verts.max(axis=0)]).tolist(),
            "vertex_stream_sha256": hashlib.sha256(
                np.ascontiguousarray(verts.astype("<f8")).tobytes()
            ).hexdigest(),
            "_vertices": verts,
            "_triangles": triangles,
        }
    if not all(record["pass"] for record in records.values()):
        raise RuntimeError("source full-mesh parity failed before decomposition")
    return records


def _cold_cook_decomposition(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tag: str,
    witness_path: Path,
    canonical_path: Path,
) -> dict[str, Any]:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory(f"d339_{tag}.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    mesh_path = "/World/D339CookMesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(x) for x in row]) for row in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * int(len(triangles)))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in triangles.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_api.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)
    decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
    decomp.CreateHullVertexLimitAttr(DECOMPOSITION_PARAMS["hull_vertex_limit"])
    decomp.CreateMaxConvexHullsAttr(DECOMPOSITION_PARAMS["max_convex_hulls"])
    decomp.CreateVoxelResolutionAttr(DECOMPOSITION_PARAMS["voxel_resolution"])
    decomp.CreateErrorPercentageAttr(DECOMPOSITION_PARAMS["error_percentage"])
    decomp.CreateMinThicknessAttr(DECOMPOSITION_PARAMS["min_thickness_m"])
    decomp.CreateShrinkWrapAttr(DECOMPOSITION_PARAMS["shrink_wrap"])
    parameter_readback = {
        "hull_vertex_limit": int(decomp.GetHullVertexLimitAttr().Get()),
        "max_convex_hulls": int(decomp.GetMaxConvexHullsAttr().Get()),
        "voxel_resolution": int(decomp.GetVoxelResolutionAttr().Get()),
        "error_percentage": float(decomp.GetErrorPercentageAttr().Get()),
        "min_thickness_m": float(decomp.GetMinThicknessAttr().Get()),
        "shrink_wrap": bool(decomp.GetShrinkWrapAttr().Get()),
    }
    parameter_readback_checks = {
        "hull_vertex_limit": parameter_readback["hull_vertex_limit"]
        == DECOMPOSITION_PARAMS["hull_vertex_limit"],
        "max_convex_hulls": parameter_readback["max_convex_hulls"]
        == DECOMPOSITION_PARAMS["max_convex_hulls"],
        "voxel_resolution": parameter_readback["voxel_resolution"]
        == DECOMPOSITION_PARAMS["voxel_resolution"],
        "error_percentage": math.isclose(
            parameter_readback["error_percentage"],
            DECOMPOSITION_PARAMS["error_percentage"],
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ),
        "min_thickness_m": math.isclose(
            parameter_readback["min_thickness_m"],
            DECOMPOSITION_PARAMS["min_thickness_m"],
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ),
        "shrink_wrap": parameter_readback["shrink_wrap"]
        == DECOMPOSITION_PARAMS["shrink_wrap"],
    }
    source_payload = {
        "vertex_stream_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(vertices, dtype="<f8")).tobytes()
        ).hexdigest(),
        "triangle_stream_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(triangles, dtype="<i8")).tobytes()
        ).hexdigest(),
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
    }
    try:
        isolated = _isolated_convex_request(
            stage,
            mesh_path,
            tag,
            witness_path,
            {
                "tag": tag,
                "source_payload": source_payload,
                "parameter_readback": parameter_readback,
                "parameter_readback_checks": parameter_readback_checks,
            },
        )
    except Exception as error:  # noqa: BLE001 - preserve callback/request failure evidence
        context_failure = {
            "artifact": "D339_CALLBACK_FIRST_COOK_WITNESS",
            "tag": tag,
            "source_payload": source_payload,
            "parameter_readback": parameter_readback,
            "parameter_readback_checks": parameter_readback_checks,
            "callback_evidence_persisted_before_classification": True,
            "request_exception": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "callback_count": 0,
        }
        if witness_path.is_file():
            try:
                preserved = json.loads(witness_path.read_text(encoding="utf-8"))
                context_failure["callback_count"] = int(
                    preserved.get("request", {}).get("callback_count", 0)
                )
            except Exception as read_error:  # original witness remains untouched
                context_failure["preserved_witness_read_error"] = (
                    f"{type(read_error).__name__}: {read_error}"
                )
            context_failure["preserved_callback_witness_path"] = _rel(witness_path)
            _json_dump(
                witness_path.with_name(witness_path.stem + "_context_failure.json"),
                context_failure,
            )
        else:
            _json_dump(witness_path, context_failure)
        return {
            "result": "REQUEST_EXCEPTION",
            "stage_id": -1,
            "stage_identifier": str(stage.GetRootLayer().identifier),
            "cache_release": {},
            "isolated_cooking_settings": {},
            "cooking_statistics_before": {},
            "cooking_statistics_after": {},
            "cooking_statistics_delta": {},
            "cooking_statistics_role": "informational_only_non_gating",
            "callback_witness_path": _rel(witness_path),
            "canonical_geometry_path": None,
            "callback_checks": {"request_completed": False},
            "checks": {"request_completed": False},
            "hard_pass": False,
            "parameter_readback": parameter_readback,
            "parameter_readback_checks": parameter_readback_checks,
            "parts": [],
            "source_payload": source_payload,
            "_stage_guard": stage,
        }

    result = isolated["result"]
    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    events = public_result["events"]
    event = events[0] if len(events) == 1 else None
    payloads = event["convexes"] if event is not None else []
    payload_checks = [_callback_payload_checks(payload) for payload in payloads]
    callback_checks = {
        "request_returned_without_exception": public_result["request_exception"] is None,
        "callback_exactly_once": len(events) == 1,
        "callback_ran_inline": bool(
            event is not None and event["callback_during_synchronous_request"]
        ),
        "result_exactly_valid": bool(
            event is not None
            and event["result_name"] == public_result["result_valid_enum_name"]
            and int(event["result_value"]) == int(public_result["result_valid_enum_value"])
        ),
        "convex_count_1_to_64": bool(
            event is not None
            and 1 <= int(event["convex_count"]) <= DECOMPOSITION_PARAMS["max_convex_hulls"]
        ),
        "convex_count_matches_payload": bool(
            event is not None and int(event["convex_count"]) == len(payloads)
        ),
        "callback_serialization_no_errors": bool(
            event is not None and not event["serialization_errors"]
        ),
        "all_callback_payloads_structurally_valid": bool(
            payload_checks and all(all(checks.values()) for checks in payload_checks)
        ),
        "stage_id_valid": bool(public_result["stage_id_valid"]),
        "fresh_stage_cache_insert": bool(
            public_result["stage_cache_id_invalid_before_insert"]
        ),
    }

    parts = []
    canonical_errors = []
    for index, payload in enumerate(payloads):
        try:
            parts.append(_canonical_convex(np.asarray(payload["vertices"], dtype=np.float64)))
        except Exception as error:  # noqa: BLE001 - becomes a hard callback geometry failure
            canonical_errors.append(
                {
                    "callback_part_index": index,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
    parts.sort(key=_convex_sort_key)
    checks = {
        "parameter_readback_exact": all(parameter_readback_checks.values()),
        "isolated_settings_applied_and_restored": _isolated_settings_pass(
            isolated["settings"]
        ),
        "cache_release_calls_completed": all(isolated["cache_release"].values()),
        "callback_contract": all(callback_checks.values()),
        "canonicalization_no_errors": not canonical_errors,
        "part_count_in_range": 1 <= len(parts) <= DECOMPOSITION_PARAMS["max_convex_hulls"],
        "vertices_within_limit": all(
            4 <= part["vertex_count"] <= DECOMPOSITION_PARAMS["hull_vertex_limit"]
            for part in parts
        ),
        "positive_finite_volume": all(
            math.isfinite(part["volume_m3"]) and part["volume_m3"] > 0.0 for part in parts
        ),
    }
    canonical_payload = {
        "artifact": "D339_CANONICAL_COOK_GEOMETRY",
        "tag": tag,
        "definition": "outward-Qhull canonical solid reconstructed from each callback vertex set",
        "callback_witness_path": _rel(witness_path),
        "callback_checks": callback_checks,
        "callback_payload_checks": payload_checks,
        "canonicalization_errors": canonical_errors,
        "parts": [
            {
                **_public_part(part, index),
                "vertices_m": part["vertices"].tolist(),
                "triangles": part["triangles"].tolist(),
            }
            for index, part in enumerate(parts)
        ],
        "checks": checks,
        "pass": all(checks.values()),
    }
    _json_dump(canonical_path, canonical_payload)
    return {
        "result": event["result_name"] if event is not None else "CALLBACK_COUNT_INVALID",
        "stage_id": public_result["stage_id"],
        "stage_identifier": public_result["stage_identifier"],
        "cache_release": isolated["cache_release"],
        "isolated_cooking_settings": isolated["settings"],
        "cooking_statistics_before": isolated["stats_before"],
        "cooking_statistics_after": isolated["stats_after"],
        "cooking_statistics_delta": isolated["stats_delta"],
        "cooking_statistics_role": isolated["statistics_role"],
        "callback_witness_path": _rel(witness_path),
        "canonical_geometry_path": _rel(canonical_path),
        "callback_checks": callback_checks,
        "callback_payload_checks": payload_checks,
        "checks": checks,
        "hard_pass": all(checks.values()),
        "parameter_readback": parameter_readback,
        "parameter_readback_checks": parameter_readback_checks,
        "parts": parts,
        "source_payload": source_payload,
        "_stage_guard": stage,
    }


def _compare_cold_cooks(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    rows = []
    count_equal = len(first["parts"]) == len(second["parts"])
    if count_equal:
        for idx, (a, b) in enumerate(zip(first["parts"], second["parts"], strict=True)):
            geometry_hash_equal = a["geometry_sha256"] == b["geometry_sha256"]
            topology_equal = bool(
                a["vertex_count"] == b["vertex_count"]
                and a["triangle_count"] == b["triangle_count"]
                and np.array_equal(a["triangles"], b["triangles"])
            )
            coord_delta = (
                float(np.max(np.abs(a["vertices"] - b["vertices"])))
                if a["vertices"].shape == b["vertices"].shape
                else math.inf
            )
            rows.append(
                {
                    "part_index": idx,
                    "topology_equal": topology_equal,
                    "first_vertex_stream_sha256": a["vertex_stream_sha256"],
                    "second_vertex_stream_sha256": b["vertex_stream_sha256"],
                    "vertex_stream_hash_equal": (
                        a["vertex_stream_sha256"] == b["vertex_stream_sha256"]
                    ),
                    "first_topology_sha256": a["topology_sha256"],
                    "second_topology_sha256": b["topology_sha256"],
                    "topology_hash_equal": a["topology_sha256"] == b["topology_sha256"],
                    "first_geometry_sha256": a["geometry_sha256"],
                    "second_geometry_sha256": b["geometry_sha256"],
                    "geometry_hash_equal": geometry_hash_equal,
                    "coordinate_max_abs_delta_m": coord_delta,
                    "coordinate_tolerance_m": COLD_COOK_COORD_TOL_M,
                    "pass": (
                        topology_equal
                        and geometry_hash_equal
                        and coord_delta <= COLD_COOK_COORD_TOL_M
                    ),
                }
            )
    stage_ids_distinct = int(first["stage_id"]) != int(second["stage_id"])
    stage_identifiers_distinct = (
        str(first["stage_identifier"]) != str(second["stage_identifier"])
    )
    isolation = []
    for label, cook in (("cold1", first), ("cold2", second)):
        isolation.append(
            {
                "label": label,
                "stage_id": int(cook["stage_id"]),
                "stage_identifier": str(cook["stage_identifier"]),
                "cache_release": cook["cache_release"],
                "isolated_cooking_settings": cook["isolated_cooking_settings"],
                "callback_checks": cook["callback_checks"],
                "callback_witness_path": cook["callback_witness_path"],
                "canonical_geometry_path": cook["canonical_geometry_path"],
                "cooking_statistics_before": cook["cooking_statistics_before"],
                "cooking_statistics_after": cook["cooking_statistics_after"],
                "cooking_statistics_delta": cook["cooking_statistics_delta"],
                "cooking_statistics_role": cook["cooking_statistics_role"],
                "hard_pass": bool(cook["hard_pass"]),
            }
        )
    isolation_pass = bool(
        stage_ids_distinct
        and stage_identifiers_distinct
        and all(
            all(row["cache_release"].values())
            and _isolated_settings_pass(row["isolated_cooking_settings"])
            and all(row["callback_checks"].values())
            and row["hard_pass"]
            for row in isolation
        )
    )
    return {
        "part_count_first": len(first["parts"]),
        "part_count_second": len(second["parts"]),
        "part_count_equal": count_equal,
        "stage_ids_distinct": stage_ids_distinct,
        "stage_identifiers_distinct": stage_identifiers_distinct,
        "source_payload_equal": first["source_payload"] == second["source_payload"],
        "parameter_readback_equal": (
            first["parameter_readback"] == second["parameter_readback"]
        ),
        "isolated_cooks": isolation,
        "isolation_pass": isolation_pass,
        "per_part": rows,
        "pass": bool(
            count_equal
            and rows
            and all(row["pass"] for row in rows)
            and isolation_pass
            and first["source_payload"] == second["source_payload"]
            and first["parameter_readback"] == second["parameter_readback"]
        ),
    }


def _public_part(part: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "part_index": int(index),
        "name": f"part_{index:03d}",
        "vertex_count": int(part["vertex_count"]),
        "triangle_count": int(part["triangle_count"]),
        "volume_m3": float(part["volume_m3"]),
        "bounds_m": part["bounds_m"],
        "centroid_m": part["centroid_m"],
        "vertex_stream_sha256": part["vertex_stream_sha256"],
        "topology_sha256": part["topology_sha256"],
        "geometry_sha256": part["geometry_sha256"],
    }


def _author_frozen_parts(physics_path: Path, cooked_by_body: dict[str, dict[str, Any]]) -> None:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open copied physics layer {physics_path}")
    for body, result in cooked_by_body.items():
        source_prim = stage.GetPrimAtPath(SOURCE_ASSET_PATHS[body]["source_spec"])
        if not source_prim.IsValid() or not source_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"missing copied source collider spec for {body}")
        UsdPhysics.CollisionAPI(source_prim).CreateCollisionEnabledAttr(False)
        parent_path = SOURCE_ASSET_PATHS[body]["parts_parent"]
        UsdGeom.Xform.Define(stage, parent_path)
        for index, part in enumerate(result["parts"]):
            path = f"{parent_path}/part_{index:03d}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(
                [Gf.Vec3f(*[float(x) for x in row]) for row in part["vertices"]]
            )
            mesh.CreateFaceVertexCountsAttr([3] * int(len(part["triangles"])))
            mesh.CreateFaceVertexIndicesAttr([int(i) for i in part["triangles"].reshape(-1)])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            hull_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim())
            hull_api.CreateHullVertexLimitAttr(DECOMPOSITION_PARAMS["hull_vertex_limit"])
            hull_api.CreateMinThicknessAttr(DECOMPOSITION_PARAMS["min_thickness_m"])
    stage.GetRootLayer().Save()


def _asset_file_hashes(asset_dir: Path) -> dict[str, str]:
    return {label: _sha256(asset_dir / rel) for label, rel in SOURCE_LAYER_REL.items()}


def _tool_mass_semantics_from_stage(asset_root: Path) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(asset_root), load=Usd.Stage.LoadAll)
    rows = {}
    for body in d334.BODY_LABELS:
        prim = stage.GetPrimAtPath(f"/roarm_m3/{body}")
        api = UsdPhysics.MassAPI(prim)
        com = api.GetCenterOfMassAttr().Get()
        inertia = api.GetDiagonalInertiaAttr().Get()
        axes = api.GetPrincipalAxesAttr().Get()
        rows[body] = {
            "mass_kg": float(api.GetMassAttr().Get()),
            "center_of_mass_m": [float(v) for v in com],
            "diagonal_inertia": [float(v) for v in inertia],
            "principal_axes_wxyz": [
                float(axes.GetReal()),
                *[float(v) for v in axes.GetImaginary()],
            ],
        }
    return rows


def _stage_semantic_inventory(asset_root: Path, *, variant: bool) -> dict[str, Any]:
    """Canonical composed-stage inventory with only D339's allowlist removed."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset_root), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open semantic-inventory stage {asset_root}")
    added_prefixes = (
        "/colliders/link5/d338_convex_parts",
        "/colliders/gripper_link/d338_convex_parts",
        "/roarm_m3/link5/collisions/d338_convex_parts",
        "/roarm_m3/gripper_link/collisions/d338_convex_parts",
    )
    collision_enabled_exclusions = {
        "/colliders/link5/link5/node_STL_BINARY_",
        "/colliders/gripper_link/gripper_link/node_STL_BINARY_",
        "/roarm_m3/link5/collisions/link5/node_STL_BINARY_",
        "/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_",
    }
    rows = []
    all_paths = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        all_paths.append(path)
        if any(path == prefix or path.startswith(prefix + "/") for prefix in added_prefixes):
            continue
        attributes = []
        for attr in prim.GetAttributes():
            if path in collision_enabled_exclusions and attr.GetName() == "physics:collisionEnabled":
                continue
            attributes.append(
                (
                    attr.GetName(),
                    str(attr.GetTypeName()),
                    repr(attr.Get()),
                    tuple(str(item) for item in attr.GetConnections()),
                )
            )
        relationships = [
            (rel.GetName(), tuple(str(item) for item in rel.GetTargets()))
            for rel in prim.GetRelationships()
        ]
        metadata = [
            (name, repr(prim.GetMetadata(name)))
            for name in (
                "kind",
                "documentation",
                "customData",
                "hidden",
            )
            if prim.HasMetadata(name)
        ]
        rows.append(
            {
                "path": path,
                "type_name": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
                "instanceable": bool(prim.IsInstanceable()),
                "applied_schemas": tuple(str(item) for item in prim.GetAppliedSchemas()),
                "metadata": sorted(metadata),
                "attributes": sorted(attributes),
                "relationships": sorted(relationships),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "variant": variant,
        "row_count_excluding_allowlist": len(rows),
        "semantic_sha256_excluding_allowlist": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_paths": sorted(all_paths),
        "added_prefixes": list(added_prefixes),
    }


def _semantic_allowlist_diff(source_root: Path, variant_root: Path) -> dict[str, Any]:
    from pxr import Sdf

    source = _stage_semantic_inventory(source_root, variant=False)
    variant = _stage_semantic_inventory(variant_root, variant=True)
    source_layer = Sdf.Layer.FindOrOpen(str(source_root.parent / SOURCE_LAYER_REL["physics"]))
    variant_layer = Sdf.Layer.FindOrOpen(str(variant_root.parent / SOURCE_LAYER_REL["physics"]))

    def _sanitized_layer(layer: Any) -> dict[str, Any]:
        if layer is None:
            return {"sha256": None, "text_length": 0}
        clone = Sdf.Layer.CreateAnonymous("d339_sanitized_physics.usda")
        if not clone.ImportFromString(layer.ExportToString()):
            raise RuntimeError("failed to clone physics layer for semantic allowlist audit")
        remove_paths = (
            "/colliders/link5/d338_convex_parts",
            "/colliders/gripper_link/d338_convex_parts",
            "/colliders/link5/link5/node_STL_BINARY_.physics:collisionEnabled",
            "/colliders/gripper_link/gripper_link/node_STL_BINARY_.physics:collisionEnabled",
        )
        edits = Sdf.BatchNamespaceEdit()
        for value in remove_paths:
            path = Sdf.Path(value)
            if clone.GetObjectAtPath(path) is not None:
                edits.Add(Sdf.NamespaceEdit.Remove(path))
        if not clone.Apply(edits):
            raise RuntimeError("failed to apply semantic allowlist removals")
        text_value = clone.ExportToString()
        return {
            "sha256": hashlib.sha256(text_value.encode("utf-8")).hexdigest(),
            "text_length": len(text_value),
        }

    source_sanitized = _sanitized_layer(source_layer)
    variant_sanitized = _sanitized_layer(variant_layer)
    layer_header_equal = bool(
        source_layer is not None
        and variant_layer is not None
        and source_layer.defaultPrim == variant_layer.defaultPrim
        and list(source_layer.subLayerPaths) == list(variant_layer.subLayerPaths)
        and source_layer.customLayerData == variant_layer.customLayerData
        and source_layer.startTimeCode == variant_layer.startTimeCode
        and source_layer.endTimeCode == variant_layer.endTimeCode
        and source_layer.timeCodesPerSecond == variant_layer.timeCodesPerSecond
    )
    source_paths = set(source["all_paths"])
    variant_paths = set(variant["all_paths"])
    added = sorted(variant_paths - source_paths)
    removed = sorted(source_paths - variant_paths)
    allowed_prefixes = tuple(variant["added_prefixes"])
    checks = {
        "no_existing_prim_removed": not removed,
        "all_added_prims_under_piece_allowlist": bool(added)
        and all(any(path == prefix or path.startswith(prefix + "/") for prefix in allowed_prefixes) for path in added),
        "all_nonallowlisted_semantics_equal": (
            source["semantic_sha256_excluding_allowlist"]
            == variant["semantic_sha256_excluding_allowlist"]
        ),
        "physics_source_specs_exact_after_allowlist_removal": (
            source_sanitized == variant_sanitized
        ),
        "physics_layer_header_equal": layer_header_equal,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "source": {key: value for key, value in source.items() if key != "all_paths"},
        "variant": {key: value for key, value in variant.items() if key != "all_paths"},
        "added_prim_paths": added,
        "removed_prim_paths": removed,
        "sanitized_physics_source_layer": {
            "source": source_sanitized,
            "variant": variant_sanitized,
        },
        "allowed_existing_property_changes": [
            "/colliders/link5/link5/node_STL_BINARY_.physics:collisionEnabled=false",
            "/colliders/gripper_link/gripper_link/node_STL_BINARY_.physics:collisionEnabled=false",
        ],
    }


def _build_derivative_asset(args: argparse.Namespace, d334_summary: dict[str, Any]) -> dict[str, Any]:
    source_dir = args.source_robot_usd_path.parent
    asset_dir = args.out_dir / "collision_asset" / args.asset_attempt / "roarm_m3_fullmesh_convex_parts"
    attempt_dir = asset_dir.parent
    manifest_path = attempt_dir / "d339_asset_build_manifest.json"
    if args.asset_attempt != "attempt2":
        raise RuntimeError("D339 is registered for exact asset_attempt=attempt2 only")
    if attempt_dir.exists():
        raise RuntimeError(
            "D339 attempt2 already contains evidence; immutable attempts are never reused"
        )

    source_hashes = _asset_file_hashes(source_dir)
    source_hash_checks = {
        label: source_hashes[label] == expected for label, expected in PIN_SOURCE_LAYER_SHA256.items()
    }
    stl_paths = {
        "link5": args.urdf_path.parent / "meshes/link5.stl",
        "gripper_link": args.urdf_path.parent / "meshes/gripper_link.stl",
        "excluded_g2a": args.urdf_path.parent / "meshes/gripper_link_collision_g2a.stl",
    }
    stl_hashes = {label: _sha256(path) for label, path in stl_paths.items()}
    stl_hash_checks = {
        label: stl_hashes[label] == expected for label, expected in PIN_STL_SHA256.items()
    }
    if not all(source_hash_checks.values()) or not all(stl_hash_checks.values()):
        raise RuntimeError("source layer/STL hash contract failed before derivative build")

    sources = _source_meshes_from_original(args.source_robot_usd_path, d334_summary)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    witness_manifest_path = attempt_dir / "d339_cook_witness_manifest.json"
    witness_manifest: dict[str, Any] = {
        "artifact": "D339_COOK_WITNESS_MANIFEST",
        "asset_attempt": args.asset_attempt,
        "status": "collecting_callback_first_evidence",
        "statistics_role": "informational_only_non_gating",
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "source_meshes": {
            body: {key: value for key, value in sources[body].items() if not key.startswith("_")}
            for body in d334.BODY_LABELS
        },
        "cooks": {},
        "repeatability": {},
        "checks": {},
        "pass": False,
    }
    _json_dump(witness_manifest_path, witness_manifest)
    cooked: dict[str, dict[str, Any]] = {}
    repeatability: dict[str, Any] = {}
    retained_stage_guards: list[dict[str, Any]] = []
    for body in d334.BODY_LABELS:
        first = _cold_cook_decomposition(
            sources[body]["_vertices"],
            sources[body]["_triangles"],
            f"{body}_cold1",
            attempt_dir / f"d339_{body}_cold1_callback_witness.json",
            attempt_dir / f"d339_{body}_cold1_canonical_geometry.json",
        )
        retained_stage_guards.append(
            {
                "tag": f"{body}_cold1",
                "stage": first["_stage_guard"],
                "recorded_stage_id": int(first["stage_id"]),
                "recorded_identifier": str(first["stage_identifier"]),
            }
        )
        witness_manifest["cooks"][f"{body}_cold1"] = {
            key: value for key, value in first.items() if not key.startswith("_") and key != "parts"
        }
        _json_dump(witness_manifest_path, witness_manifest)
        second = _cold_cook_decomposition(
            sources[body]["_vertices"],
            sources[body]["_triangles"],
            f"{body}_cold2",
            attempt_dir / f"d339_{body}_cold2_callback_witness.json",
            attempt_dir / f"d339_{body}_cold2_canonical_geometry.json",
        )
        retained_stage_guards.append(
            {
                "tag": f"{body}_cold2",
                "stage": second["_stage_guard"],
                "recorded_stage_id": int(second["stage_id"]),
                "recorded_identifier": str(second["stage_identifier"]),
            }
        )
        witness_manifest["cooks"][f"{body}_cold2"] = {
            key: value for key, value in second.items() if not key.startswith("_") and key != "parts"
        }
        repeatability[body] = _compare_cold_cooks(first, second)
        witness_manifest["repeatability"][body] = repeatability[body]
        _json_dump(witness_manifest_path, witness_manifest)
        cooked[body] = first

    all_stage_ids = [
        int(cook["stage_id"])
        for body in d334.BODY_LABELS
        for cook in (
            witness_manifest["cooks"][f"{body}_cold1"],
            witness_manifest["cooks"][f"{body}_cold2"],
        )
    ]
    all_stage_identifiers = [
        str(cook["stage_identifier"])
        for body in d334.BODY_LABELS
        for cook in (
            witness_manifest["cooks"][f"{body}_cold1"],
            witness_manifest["cooks"][f"{body}_cold2"],
        )
    ]
    from pxr import Usd, UsdUtils

    stage_cache = UsdUtils.StageCache.Get()
    stage_cache_lifecycle = []
    for retained in retained_stage_guards:
        current_id = stage_cache.GetId(retained["stage"])
        current_id_value = int(current_id.ToLongInt()) if current_id.IsValid() else -1
        found_stage = (
            stage_cache.Find(Usd.StageCache.Id.FromLongInt(retained["recorded_stage_id"]))
            if retained["recorded_stage_id"] >= 0
            else None
        )
        found_identifier = (
            str(found_stage.GetRootLayer().identifier) if found_stage is not None else None
        )
        row_checks = {
            "current_id_valid": bool(current_id.IsValid()),
            "current_id_matches_recorded": (
                current_id_value == retained["recorded_stage_id"]
            ),
            "recorded_id_resolves": found_stage is not None,
            "resolved_identifier_matches_recorded": (
                found_identifier == retained["recorded_identifier"]
            ),
            "retained_stage_identifier_unchanged": (
                str(retained["stage"].GetRootLayer().identifier)
                == retained["recorded_identifier"]
            ),
        }
        stage_cache_lifecycle.append(
            {
                "tag": retained["tag"],
                "recorded_stage_id": retained["recorded_stage_id"],
                "current_stage_id": current_id_value,
                "recorded_identifier": retained["recorded_identifier"],
                "resolved_identifier": found_identifier,
                "checks": row_checks,
                "pass": all(row_checks.values()),
            }
        )

    witness_checks = {
        "four_retained_stage_objects": len(retained_stage_guards) == 4,
        "four_unique_valid_stage_ids": (
            len(all_stage_ids) == 4
            and all(stage_id >= 0 for stage_id in all_stage_ids)
            and len(set(all_stage_ids)) == 4
        ),
        "four_unique_stage_identifiers": (
            len(all_stage_identifiers) == 4 and len(set(all_stage_identifiers)) == 4
        ),
        "all_four_stage_cache_mappings_live_and_stable": bool(
            len(stage_cache_lifecycle) == 4
            and all(row["pass"] for row in stage_cache_lifecycle)
        ),
        "all_cooks_hard_pass": all(
            bool(witness_manifest["cooks"][f"{body}_{label}"]["hard_pass"])
            for body in d334.BODY_LABELS
            for label in ("cold1", "cold2")
        ),
        "both_body_repeatability_pass": all(
            bool(repeatability[body]["pass"]) for body in d334.BODY_LABELS
        ),
        "attempt1_integrity_preserved": bool(_d338_attempt1_integrity()["pass"]),
    }
    witness_manifest["checks"] = witness_checks
    witness_manifest["stage_cache_lifecycle"] = stage_cache_lifecycle
    witness_manifest["pass"] = all(witness_checks.values())
    witness_manifest["status"] = (
        "callback_first_two_cook_geometry_equality_pass"
        if witness_manifest["pass"]
        else "callback_first_two_cook_geometry_equality_fail_stop"
    )
    witness_manifest["attempt1_integrity_after_cooks"] = _d338_attempt1_integrity()
    _json_dump(witness_manifest_path, witness_manifest)
    if not witness_manifest["pass"]:
        raise RuntimeError("D339 callback-first cook-witness contract failed")

    shutil.copytree(source_dir, asset_dir)
    _author_frozen_parts(asset_dir / SOURCE_LAYER_REL["physics"], cooked)
    variant_hashes = _asset_file_hashes(asset_dir)
    source_mass_semantics = _tool_mass_semantics_from_stage(args.source_robot_usd_path)
    variant_mass_semantics = _tool_mass_semantics_from_stage(asset_dir / "roarm_m3.usd")
    semantic_diff = _semantic_allowlist_diff(
        args.source_robot_usd_path, asset_dir / "roarm_m3.usd"
    )
    copy_parity = {
        label: variant_hashes[label] == source_hashes[label]
        for label in ("root", "base", "robot", "sensor")
    }
    copy_parity["physics_is_only_changed_layer"] = (
        variant_hashes["physics"] != source_hashes["physics"] and all(copy_parity.values())
    )
    public_parts = {
        body: [_public_part(part, idx) for idx, part in enumerate(cooked[body]["parts"])]
        for body in d334.BODY_LABELS
    }
    parameter_readback = {
        body: {
            "values": cooked[body]["parameter_readback"],
            "checks": cooked[body]["parameter_readback_checks"],
        }
        for body in d334.BODY_LABELS
    }
    hull_manifest_path = attempt_dir / "d339_hull_manifest.json"
    hull_manifest = {
        "artifact": "D339_HULL_MANIFEST",
        "asset_attempt": args.asset_attempt,
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "parameter_readback": parameter_readback,
        "cold_cook_repeatability": repeatability,
        "cook_witness_manifest_path": _rel(witness_manifest_path),
        "cook_witness_manifest_sha256": _sha256(witness_manifest_path),
        "parts": public_parts,
    }
    _json_dump(hull_manifest_path, hull_manifest)
    checks = {
        "source_layer_hashes": all(source_hash_checks.values()),
        "source_stl_hashes": all(stl_hash_checks.values()),
        "source_mesh_parity": all(sources[body]["pass"] for body in d334.BODY_LABELS),
        "cold_cook_repeatability": all(repeatability[body]["pass"] for body in d334.BODY_LABELS),
        "cook_witness_contract": bool(witness_manifest["pass"]),
        "copied_nonphysics_layers_bit_exact": all(
            copy_parity[label] for label in ("root", "base", "robot", "sensor")
        ),
        "physics_is_only_changed_layer": copy_parity["physics_is_only_changed_layer"],
        "semantic_mutation_allowlist": bool(semantic_diff["pass"]),
        "tool_mass_semantics_equal": source_mass_semantics == variant_mass_semantics,
        "attempt1_integrity_after_asset_build": bool(_d338_attempt1_integrity()["pass"]),
        "d338_helper_implementation_still_pinned": _sha256(D338_SCRIPT)
        == PIN_D338_SCRIPT_SHA256,
        "hull_manifest_written_nonzero": bool(
            hull_manifest_path.is_file() and hull_manifest_path.stat().st_size > 0
        ),
    }
    manifest = {
        "artifact": "D339_ASSET_BUILD_MANIFEST",
        "checks": checks,
        "pass": all(checks.values()),
        "reused_verified_asset": False,
        "source_asset_dir": _rel(source_dir),
        "variant_asset_dir": _rel(asset_dir),
        "variant_robot_usd": _rel(asset_dir / "roarm_m3.usd"),
        "manifest_path": _rel(manifest_path),
        "hull_manifest_path": _rel(hull_manifest_path),
        "hull_manifest_sha256": _sha256(hull_manifest_path),
        "cook_witness_manifest_path": _rel(witness_manifest_path),
        "cook_witness_manifest_sha256": _sha256(witness_manifest_path),
        "asset_attempt": args.asset_attempt,
        "source_layer_sha256": source_hashes,
        "variant_layer_sha256": variant_hashes,
        "source_stl_sha256": stl_hashes,
        "excluded_moving_jaw_identity": "gripper_link_collision_g2a.stl",
        "decomposition_parameters": DECOMPOSITION_PARAMS,
        "source_meshes": {
            body: {key: value for key, value in sources[body].items() if not key.startswith("_")}
            for body in d334.BODY_LABELS
        },
        "cold_cook_repeatability": repeatability,
        "cold_cook_parameter_readback": parameter_readback,
        "semantic_mutation_allowlist": semantic_diff,
        "source_tool_mass_semantics": source_mass_semantics,
        "variant_tool_mass_semantics": variant_mass_semantics,
        "parts": public_parts,
    }
    _json_dump(manifest_path, manifest)
    if not manifest["pass"]:
        raise RuntimeError("D339 derivative build contract failed")
    return manifest


def _build_retained_raw_shapes(
    inner: Any, d334_summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import hppfcl

    shapes: list[dict[str, Any]] = []
    per_body: dict[str, Any] = {}
    for body in d334.BODY_LABELS:
        rows = d334._usd_collision_inventory(inner, body)
        source_rows = [row for row in rows if row["path"] == LIVE_OLD_COLLIDER_PATHS[body]]
        row = source_rows[0] if len(source_rows) == 1 else None
        checks = {
            "exactly_one_retained_source": len(source_rows) == 1,
            "retained_source_disabled": bool(row is not None and not row["collision_enabled"]),
            "retained_source_convex_hull_token": bool(
                row is not None and row["approximation"] == "convexHull"
            ),
            "owner_matches": bool(
                row is not None and row["nearest_rigid_body_ancestor"] == d334.BODY_PATHS[body]
            ),
        }
        source = None
        if row is not None:
            source = d334._source_mesh_body_local(inner, row, body)
            expected = next(item for item in d334_summary["shapes"] if item["body"] == body)[
                "source_mesh"
            ]
            parity = {
                "mesh_path": source["mesh_prim_path"] == expected["mesh_prim_path"],
                "vertex_count": int(source["vertex_count"]) == int(expected["vertex_count"]),
                "face_count": int(source["face_count"]) == int(expected["face_count"]),
                "triangle_count": int(source["triangle_count"]) == int(expected["triangle_count"]),
                "fan_triangulated": bool(source["fan_triangulated"])
                == bool(expected["fan_triangulated"]),
                "bounds_bit_near": bool(
                    np.allclose(
                        np.asarray(source["body_local_bounds_m"], dtype=np.float64),
                        np.asarray(expected["body_local_bounds_m"], dtype=np.float64),
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ),
            }
            checks["source_mesh_matches_d334"] = all(parity.values())
        else:
            parity = {}
            checks["source_mesh_matches_d334"] = False
        body_pass = all(checks.values())
        per_body[body] = {
            "checks": checks,
            "source_mesh_parity": parity,
            "inventory": rows,
            "pass": body_pass,
        }
        if body_pass and source is not None and row is not None:
            shapes.append(
                {
                    "body": body,
                    "collider_path": row["path"],
                    "owner_body_path": row["nearest_rigid_body_ancestor"],
                    "source_mesh": {
                        key: value for key, value in source.items() if not key.startswith("_")
                    },
                    "_raw_verts": source["_verts_body"],
                    "_triangles": source["_triangles"],
                    "_geom_raw": d332._build_raw_bvh(
                        hppfcl, source["_verts_body"], source["_triangles"]
                    ),
                }
            )
    contract = {
        "per_body": per_body,
        "body_set_exact": sorted(shape["body"] for shape in shapes) == sorted(d334.BODY_LABELS),
    }
    contract["pass"] = bool(
        contract["body_set_exact"] and all(per_body[body]["pass"] for body in d334.BODY_LABELS)
    )
    return shapes, contract


def _expected_live_part_paths(manifest: dict[str, Any], body: str) -> list[str]:
    parent = LIVE_PART_PARENT_PATHS[body]
    return [f"{parent}/{row['name']}" for row in manifest["parts"][body]]


def _prim_to_body_transform(inner: Any, prim_path: str, body: str) -> dict[str, Any]:
    """Return the composed prim-local -> rigid-body-local affine transform."""
    from pxr import Gf, Usd, UsdGeom

    stage = inner.scene.stage
    prim = stage.GetPrimAtPath(prim_path)
    body_prim = stage.GetPrimAtPath(d334.BODY_PATHS[body])
    if not prim.IsValid() or not body_prim.IsValid():
        raise RuntimeError(f"invalid prim/body transform request: {prim_path}, {body}")
    prim_l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_w2l = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()

    def _map(point: tuple[float, float, float]) -> np.ndarray:
        mapped = body_w2l.Transform(prim_l2w.Transform(Gf.Vec3d(*point)))
        return np.asarray([float(value) for value in mapped], dtype=np.float64)

    origin = _map((0.0, 0.0, 0.0))
    axes = np.column_stack(
        [_map(axis) - origin for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = axes
    matrix[:3, 3] = origin
    delta = float(np.max(np.abs(matrix - np.eye(4, dtype=np.float64))))
    return {
        "matrix": matrix,
        "matrix_row_major": matrix.tolist(),
        "identity_max_abs_delta": delta,
        "identity_tolerance": 1.0e-9,
        "identity_pass": delta <= 1.0e-9,
        "prim_l2w": prim_l2w,
        "body_w2l": body_w2l,
    }


def _direct_cook_live_piece(inner: Any, row: dict[str, Any], body: str) -> dict[str, Any]:
    from pxr import Gf, PhysicsSchemaTools

    stage = inner.scene.stage
    prim = stage.GetPrimAtPath(row["path"])
    relative = _prim_to_body_transform(inner, row["path"], body)
    candidate_prims = [prim]
    if prim.IsValid() and prim.IsInstanceProxy():
        try:
            prototype = prim.GetPrimInPrototype()
            if prototype.IsValid():
                candidate_prims.append(prototype)
        except Exception:  # pragma: no cover - API-version defensive path
            pass
    attempts = []
    chosen = None
    for candidate in candidate_prims:
        path = candidate.GetPath().pathString
        attempt: dict[str, Any] = {"prim_path": path}
        try:
            outcome = d334._request_cook(inner, PhysicsSchemaTools.sdfPathToInt(path))
            convexes = list(outcome["holder"].get("convexes", []))
            attempt.update(
                {
                    "result": str(outcome["holder"].get("result")),
                    "convex_count": len(convexes),
                    "valid": bool(outcome["valid"] and len(convexes) == 1),
                }
            )
            if attempt["valid"] and chosen is None:
                vertices_local = np.asarray(
                    [[float(v.x), float(v.y), float(v.z)] for v in convexes[0].vertices],
                    dtype=np.float64,
                )
                vertices_body = np.asarray(
                    [
                        [
                            float(value)
                            for value in relative["body_w2l"].Transform(
                                relative["prim_l2w"].Transform(Gf.Vec3d(*vertex))
                            )
                        ]
                        for vertex in vertices_local
                    ],
                    dtype=np.float64,
                )
                chosen = _canonical_convex(vertices_body)
                chosen["cook_prim_path"] = path
                chosen["coordinate_mapping_prim_path"] = row["path"]
                chosen["live_polygon_count"] = int(len(convexes[0].polygons))
        except Exception as error:  # noqa: BLE001 - evidence record
            attempt["valid"] = False
            attempt["exception"] = f"{type(error).__name__}: {error}"
        attempts.append(attempt)
    return {
        "body": body,
        "path": row["path"],
        "attempts": attempts,
        "chosen": chosen,
        "prim_to_body_transform": {
            key: value
            for key, value in relative.items()
            if key not in {"matrix", "prim_l2w", "body_w2l"}
        },
    }


def _build_live_cooked_parts(
    inner: Any,
    simulation_app: Any,
    manifest: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import hppfcl

    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    validator_extension = "omni.physx.asset_validator"
    validator_was_enabled = bool(
        extension_manager.is_extension_enabled(validator_extension)
    )
    if not validator_was_enabled:
        extension_manager.set_extension_enabled_immediate(validator_extension, True)
    validator_enabled = bool(
        extension_manager.is_extension_enabled(validator_extension)
    )
    if not validator_enabled:
        raise RuntimeError(f"failed to enable required extension {validator_extension}")
    cooked_by_body: dict[str, list[dict[str, Any]]] = {}
    from omni.physxassetvalidator import get_physx_asset_validator_interface
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

    validator = get_physx_asset_validator_interface()
    stage_id = d334._stage_id(inner)
    validator_guard_before = d334._snapshot_sim_state(inner)
    audit: dict[str, Any] = {
        "per_body": {},
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(inner.scene.stage)),
        "asset_validator_extension": {
            "name": validator_extension,
            "was_enabled": validator_was_enabled,
            "enabled_for_audit": validator_enabled,
            "pass": validator_enabled,
        },
    }
    audit["stage_meters_per_unit_one"] = math.isclose(
        audit["stage_meters_per_unit"], 1.0, rel_tol=0.0, abs_tol=1.0e-12
    )
    for body in d334.BODY_LABELS:
        usd_inventory = d334._usd_collision_inventory(inner, body)
        enabled = sorted(
            [row for row in usd_inventory if row["collision_enabled"]], key=lambda row: row["path"]
        )
        expected_paths = sorted(_expected_live_part_paths(manifest, body))
        property_query = d334._property_query_body(inner, simulation_app, body)
        property_rows = sorted(property_query["colliders"], key=lambda row: str(row["path"]))
        property_by_path = {str(row["path"]): row for row in property_rows}
        direct_rows = []
        live_parts = []
        part_checks = []
        manifest_by_name = {row["name"]: row for row in manifest["parts"][body]}
        mass_api = UsdPhysics.MassAPI(inner.scene.stage.GetPrimAtPath(d334.BODY_PATHS[body]))
        com = mass_api.GetCenterOfMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        axes = mass_api.GetPrincipalAxesAttr().Get()
        live_mass = {
            "mass_kg": float(mass_api.GetMassAttr().Get()),
            "center_of_mass_m": [float(v) for v in com],
            "diagonal_inertia": [float(v) for v in inertia],
            "principal_axes_wxyz": [
                float(axes.GetReal()),
                *[float(v) for v in axes.GetImaginary()],
            ],
        }
        expected_mass = manifest["source_tool_mass_semantics"][body]
        for row in enabled:
            direct = _direct_cook_live_piece(inner, row, body)
            direct_rows.append(
                {
                    "body": direct["body"],
                    "path": direct["path"],
                    "attempts": direct["attempts"],
                    "prim_to_body_transform": direct["prim_to_body_transform"],
                    "chosen": None
                    if direct["chosen"] is None
                    else {
                        key: value
                        for key, value in direct["chosen"].items()
                        if key not in {"vertices", "triangles"}
                    },
                }
            )
            name = Path(row["path"]).name
            expected = manifest_by_name.get(name)
            chosen = direct["chosen"]
            prop = property_by_path.get(row["path"])
            authored_source = d334._source_mesh_body_local(inner, row, body)
            authored_mesh = UsdGeom.Mesh(inner.scene.stage.GetPrimAtPath(row["path"]))
            authored = _canonical_convex(
                np.asarray(
                    [[float(v) for v in point] for point in authored_mesh.GetPointsAttr().Get()],
                    dtype=np.float64,
                )
            )
            live_prim = inner.scene.stage.GetPrimAtPath(row["path"])
            hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(live_prim)
            hull_vertex_limit = hull_api.GetHullVertexLimitAttr().Get()
            hull_min_thickness = hull_api.GetMinThicknessAttr().Get()
            gpu_compatible = bool(
                validator.convex_gpu_compatibility_is_valid(
                    stage_id, PhysicsSchemaTools.sdfPathToInt(row["path"])
                )
            )
            checks = {
                "manifest_part_found": expected is not None,
                "direct_cook_one_convex": chosen is not None,
                "property_query_path_found": prop is not None,
                "property_query_local_position_zero": bool(
                    prop is not None
                    and np.max(
                        np.abs(np.asarray(prop["local_pos_m"], dtype=np.float64))
                    )
                    <= 1.0e-9
                ),
                "property_query_local_rotation_identity": bool(
                    prop is not None
                    and min(
                        float(
                            np.linalg.norm(
                                np.asarray(prop["local_rot_xyzw"], dtype=np.float64)
                                - np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                            )
                        ),
                        float(
                            np.linalg.norm(
                                np.asarray(prop["local_rot_xyzw"], dtype=np.float64)
                                + np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                            )
                        ),
                    )
                    <= 1.0e-9
                ),
                "owner_matches": row["nearest_rigid_body_ancestor"] == d334.BODY_PATHS[body],
                "mesh_api_on_live_mesh": row["type_name"] == "Mesh"
                and row["approximation"] == "convexHull",
                "piece_to_body_transform_identity": bool(
                    direct["prim_to_body_transform"]["identity_pass"]
                ),
                "piece_body_bounds_match_manifest": bool(
                    np.allclose(
                        np.asarray(authored_source["body_local_bounds_m"], dtype=np.float64),
                        np.asarray(expected["bounds_m"], dtype=np.float64)
                        if expected is not None
                        else np.full((2, 3), np.nan),
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ),
                "authored_geometry_hash": bool(
                    expected is not None and authored["geometry_sha256"] == expected["geometry_sha256"]
                ),
                "hull_vertex_limit_readback": int(hull_vertex_limit or -1)
                == DECOMPOSITION_PARAMS["hull_vertex_limit"],
                "hull_min_thickness_readback": bool(
                    hull_min_thickness is not None
                    and math.isclose(
                        float(hull_min_thickness),
                        DECOMPOSITION_PARAMS["min_thickness_m"],
                        rel_tol=0.0,
                        abs_tol=1.0e-10,
                    )
                ),
                "physx_gpu_convex_compatible": gpu_compatible,
                "live_vertex_count_within_registered_limit": bool(
                    chosen is not None
                    and 4
                    <= int(chosen["vertex_count"])
                    <= DECOMPOSITION_PARAMS["hull_vertex_limit"]
                ),
            }
            volume_rel_manifest = math.inf
            volume_rel_property = math.inf
            if expected is not None and chosen is not None:
                volume_rel_manifest = abs(
                    float(chosen["volume_m3"]) - float(expected["volume_m3"])
                ) / max(abs(float(expected["volume_m3"])), 1.0e-12)
            if prop is not None and chosen is not None:
                volume_rel_property = abs(
                    float(chosen["volume_m3"]) - float(prop["volume_m3"])
                ) / max(abs(float(prop["volume_m3"])), 1.0e-12)
            checks["property_vs_direct_volume_binding_le_5pct"] = bool(
                math.isfinite(volume_rel_property)
                and volume_rel_property <= PROPERTY_VOLUME_BINDING_REL_TOL
            )
            volume_corroboration = {
                "authored_vs_live_volume_le_0p5pct": (
                    volume_rel_manifest <= LIVE_VOLUME_PARITY_REL_TOL
                ),
                "property_vs_live_volume_le_0p5pct": (
                    volume_rel_property <= LIVE_VOLUME_PARITY_REL_TOL
                ),
                "hard_gate": False,
            }
            surface = (
                _convex_solid_hausdorff_m(authored, chosen)
                if chosen is not None
                else {
                    "authored_to_live_m": None,
                    "live_to_authored_m": None,
                    "symmetric_m": None,
                    "tolerance_m": LIVE_SURFACE_PARITY_TOL_M,
                    "pass": False,
                }
            )
            checks["authored_vs_live_surface_le_0p1mm"] = bool(surface["pass"])
            part_checks.append(
                {
                    "path": row["path"],
                    "checks": checks,
                    "volume_corroboration_only": volume_corroboration,
                    "authored_vs_live_volume_relative_difference": volume_rel_manifest,
                    "property_vs_live_volume_relative_difference": volume_rel_property,
                    "property_vs_direct_volume_binding_tolerance": (
                        PROPERTY_VOLUME_BINDING_REL_TOL
                    ),
                    "authored_vs_live_surface": surface,
                    "hull_vertex_limit_readback": hull_vertex_limit,
                    "hull_min_thickness_readback_m": hull_min_thickness,
                    "physx_gpu_convex_compatible": gpu_compatible,
                    "pass": all(checks.values()),
                }
            )
            if all(checks.values()) and chosen is not None:
                points = d332._fcl_points(hppfcl, chosen["vertices"])
                geometry = hppfcl.Convex.convexHull(points, False, "")
                if geometry is None:
                    raise RuntimeError(f"hppfcl live convex reconstruction failed: {row['path']}")
                live_parts.append(
                    {
                        "body": body,
                        "path": row["path"],
                        "_vertices": chosen["vertices"],
                        "_triangles": chosen["triangles"],
                        "_geometry": geometry,
                    }
                )
        body_checks = {
            "old_single_hull_disabled": not any(
                row["path"] == LIVE_OLD_COLLIDER_PATHS[body] and row["collision_enabled"]
                for row in usd_inventory
            ),
            "enabled_paths_exact": [row["path"] for row in enabled] == expected_paths,
            "property_query_pass": bool(property_query["pass"]),
            "property_state_guard": not bool(property_query["state_guard"]["violated"]),
            "property_paths_exact": [str(row["path"]) for row in property_rows] == expected_paths,
            "all_parts_direct_certified": bool(part_checks)
            and all(row["pass"] for row in part_checks),
            "live_part_count_exact": len(live_parts) == len(expected_paths),
            "stage_meters_per_unit_one": audit["stage_meters_per_unit_one"],
            "live_mass_com_inertia_axes_equal": all(
                np.allclose(
                    np.asarray(live_mass[key], dtype=np.float64),
                    np.asarray(expected_mass[key], dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for key in live_mass
            ),
            "property_query_mass_equal": bool(
                property_query.get("rigid_body") is not None
                and math.isclose(
                    float(property_query["rigid_body"]["mass"]),
                    float(expected_mass["mass_kg"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ),
        }
        cooked_by_body[body] = live_parts
        audit["per_body"][body] = {
            "checks": body_checks,
            "pass": all(body_checks.values()),
            "usd_inventory": usd_inventory,
            "property_query": property_query,
            "direct_cooks": direct_rows,
            "part_checks": part_checks,
            "live_mass_semantics": live_mass,
            "expected_mass_semantics": expected_mass,
        }
    validator_guard = d334._state_guard(
        validator_guard_before, d334._snapshot_sim_state(inner)
    )
    audit["asset_validator_state_guard"] = validator_guard
    audit["asset_validator_state_guard_pass"] = not bool(validator_guard["violated"])
    audit["pass"] = bool(
        audit["stage_meters_per_unit_one"]
        and audit["asset_validator_extension"]["pass"]
        and audit["asset_validator_state_guard_pass"]
        and all(audit["per_body"][body]["pass"] for body in d334.BODY_LABELS)
    )
    return cooked_by_body, audit


def _cooked_union_distances(
    inner: Any, cooked_by_body: dict[str, list[dict[str, Any]]], pose_label: str
) -> dict[str, Any]:
    import hppfcl

    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    obj_pos, obj_quat = d334._object_pose_w(inner)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    rows = []
    body_poses = {}
    for body in d334.BODY_LABELS:
        pos, quat = d334._body_pose_w(inner, body)
        body_poses[body] = {"pos_m": pos.tolist(), "quat_wxyz": quat.tolist()}
        tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(quat), pos)
        part_queries = []
        for part in cooked_by_body[body]:
            query = d332._fcl_query(hppfcl, part["_geometry"], tf, cylinder, cyl_tf)
            epa = d336._epa_exact_contacts(
                hppfcl, part["_geometry"], tf, cylinder, cyl_tf
            )
            if bool(query["is_collision"]):
                exact = -float(epa["max_abs_depth_mm"]) if epa["num_contacts"] > 0 else None
                consistent = bool(epa["is_collision"] and epa["num_contacts"] > 0)
            else:
                exact = float(query["signed_distance_mm"])
                consistent = bool(not epa["is_collision"])
            part_queries.append(
                {
                    "path": part["path"],
                    "is_collision": bool(query["is_collision"]),
                    "bvh_separation_mm": (
                        None
                        if bool(query["is_collision"])
                        else float(query["signed_distance_mm"])
                    ),
                    "colliding_bvh_scalar_omitted_as_ranking_invalid": bool(
                        query["is_collision"]
                    ),
                    "exact_signed_distance_mm": exact,
                    "exact_consistent": consistent,
                    "epa_contact_count": int(epa["num_contacts"]),
                    "epa_cap_saturated": bool(epa["cap_saturated"]),
                    "nearest_point_geometry_m": query["nearest_point_geometry_m"],
                    "nearest_point_cylinder_m": query["nearest_point_cylinder_m"],
                }
            )
        collisions = [row for row in part_queries if row["is_collision"]]
        if collisions:
            eligible = [
                row for row in collisions if row["exact_signed_distance_mm"] is not None
            ]
            if eligible:
                witness = min(eligible, key=lambda row: float(row["exact_signed_distance_mm"]))
                exact_signed: float | None = float(witness["exact_signed_distance_mm"])
            else:
                witness = collisions[0]
                exact_signed = None
            exact_consistent = bool(
                len(eligible) == len(collisions)
                and all(row["exact_consistent"] for row in collisions)
                and all(not row["epa_cap_saturated"] for row in collisions)
            )
            state = "overlap" if exact_consistent else "borderline"
        else:
            witness = min(part_queries, key=lambda row: float(row["exact_signed_distance_mm"]))
            exact_signed = float(witness["exact_signed_distance_mm"])
            exact_consistent = bool(
                all(row["exact_consistent"] for row in part_queries)
                and all(not row["epa_cap_saturated"] for row in part_queries)
            )
            state = "clear" if exact_signed >= CLEAR_GATE_MM else "borderline"
        rows.append(
            {
                "pose": pose_label,
                "body": body,
                "representation": "live_frozen_convex_union",
                "part_count": len(part_queries),
                "is_collision": bool(collisions),
                "exact_signed_distance_mm": exact_signed,
                "exact_consistent": exact_consistent,
                "epa_cap_saturated_any": any(
                    row["epa_cap_saturated"] for row in part_queries
                ),
                "overlap_state": state,
                "clear_pass": bool(
                    not collisions
                    and exact_signed is not None
                    and exact_signed >= CLEAR_GATE_MM
                    and exact_consistent
                ),
                "nearest_point_geometry_m": witness["nearest_point_geometry_m"],
                "nearest_point_cylinder_m": witness["nearest_point_cylinder_m"],
                "witness_part_path": witness["path"],
                "parts": part_queries,
            }
        )
    return {
        "pose": pose_label,
        "object_pos_w_m": obj_pos.tolist(),
        "object_quat_wxyz": obj_quat.tolist(),
        "body_poses_w": body_poses,
        "queries": rows,
    }


def _queries_by_body(distance_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["body"]: row for row in distance_set["queries"]}


def _frozen_anchor_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "radial_offset_frozen": int(candidate["radial_offset_nm"]) == OLD_RADIAL_NM,
        "tangent_offset_frozen": int(candidate["tangent_offset_nm"]) == OLD_TANGENT_NM,
        "q5_frozen": math.isclose(float(candidate["q5_rad"]), Q5_OPEN_RAD, abs_tol=1.0e-12),
        "alignment_pass": bool(candidate["legacy_alignment_pass"]),
        "raw_candidate_pass": bool(candidate["pass"]),
        "sim_counter_unchanged": bool(candidate["sim_step_counter_unchanged"]),
        "link5_raw_anchor": abs(
            float(candidate["link5_exact_signed_distance_mm"])
            - 4.2726455336106985
        )
        <= RAW_ANCHOR_TOL_MM,
        "gripper_raw_anchor": abs(
            float(candidate["gripper_link_exact_signed_distance_mm"])
            - 11.175088374613944
        )
        <= RAW_ANCHOR_TOL_MM,
    }


def _representation_gate(
    inner: Any,
    raw_shapes: list[dict[str, Any]],
    cooked_by_body: dict[str, list[dict[str, Any]]],
    candidate: dict[str, Any],
    controls: dict[str, Any],
    live_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not live_audit.get("pass") or any(
        not cooked_by_body.get(body) for body in d334.BODY_LABELS
    ):
        raise RuntimeError("representation gate called with an uncertified live convex union")
    raw_exact = d336._exact_raw_metrics(inner, raw_shapes, "d339_frozen_target_raw")
    cooked = _cooked_union_distances(inner, cooked_by_body, "d339_frozen_target_cooked")
    raw_by_body = _queries_by_body(raw_exact)
    cooked_by_name = _queries_by_body(cooked)
    body_rows = {}
    for body in d334.BODY_LABELS:
        raw_value = raw_by_body[body]["exact_signed_distance_mm"]
        cooked_value = cooked_by_name[body]["exact_signed_distance_mm"]
        delta = (
            abs(float(raw_value) - float(cooked_value))
            if raw_value is not None and cooked_value is not None
            else math.inf
        )
        checks = {
            "raw_exact_consistent": bool(raw_by_body[body]["exact_consistent"]),
            "cooked_exact_consistent": bool(cooked_by_name[body]["exact_consistent"]),
            "raw_epa_not_saturated": not bool(raw_by_body[body]["epa_cap_saturated"]),
            "cooked_epa_not_saturated": not bool(
                cooked_by_name[body]["epa_cap_saturated_any"]
            ),
            "raw_clear_ge_0p1mm": bool(
                raw_value is not None and float(raw_value) >= CLEAR_GATE_MM
            ),
            "cooked_clear_ge_0p1mm": bool(
                cooked_value is not None and float(cooked_value) >= CLEAR_GATE_MM
            ),
            "cooked_vs_raw_delta_le_0p5mm": delta <= TASK_FIDELITY_TOL_MM,
        }
        body_rows[body] = {
            "raw_exact_signed_distance_mm": raw_value,
            "cooked_exact_signed_distance_mm": cooked_value,
            "absolute_delta_mm": delta,
            "tolerance_mm": TASK_FIDELITY_TOL_MM,
            "checks": checks,
            "pass": all(checks.values()),
        }
    anchor_checks = _frozen_anchor_checks(candidate)
    checks = {
        "d337_controls": bool(controls["pass"]),
        "live_collider_audit": bool(live_audit["pass"]),
        "frozen_anchor": all(anchor_checks.values()),
        "both_bodies_task_fidelity": all(row["pass"] for row in body_rows.values()),
    }
    gate = {
        "artifact": "D339_REPRESENTATION_GATE",
        "checks": checks,
        "anchor_checks": anchor_checks,
        "per_body": body_rows,
        "contract_pass": bool(
            checks["d337_controls"] and checks["live_collider_audit"] and checks["frozen_anchor"]
        ),
        "target_clear_and_faithful": bool(
            checks["both_bodies_task_fidelity"] and all(anchor_checks.values())
        ),
        "controlled_physics_steps": 0,
    }
    gate["physics_licensed"] = bool(gate["contract_pass"] and gate["target_clear_and_faithful"])
    return gate, raw_exact, cooked


def _combined_distance_flat(
    step: int, raw_set: dict[str, Any], cooked_set: dict[str, Any]
) -> dict[str, Any]:
    raw = _queries_by_body(raw_set)
    cooked = _queries_by_body(cooked_set)
    row: dict[str, Any] = {"step": int(step)}
    for body in d334.BODY_LABELS:
        raw_exact = raw[body].get("exact_signed_distance_mm")
        cooked_exact = cooked[body].get("exact_signed_distance_mm")
        row[f"{body}_raw_exact_signed_distance_mm"] = (
            None if raw_exact is None else float(raw_exact)
        )
        row[f"{body}_raw_clear_pass"] = bool(raw[body]["clear_pass"])
        row[f"{body}_raw_is_collision"] = bool(raw[body]["is_collision"])
        row[f"{body}_raw_exact_consistent"] = bool(raw[body]["exact_consistent"])
        row[f"{body}_raw_epa_contact_count"] = int(raw[body]["epa_contact_count"])
        row[f"{body}_raw_epa_cap_saturated"] = bool(raw[body]["epa_cap_saturated"])
        row[f"{body}_raw_nearest_point_geometry_m"] = json.dumps(
            raw[body].get("nearest_point_geometry_m")
        )
        row[f"{body}_raw_nearest_point_cylinder_m"] = json.dumps(
            raw[body].get("nearest_point_cylinder_m")
        )
        row[f"{body}_cooked_exact_signed_distance_mm"] = float(
            cooked_exact
        ) if cooked_exact is not None else None
        row[f"{body}_cooked_clear_pass"] = bool(cooked[body]["clear_pass"])
        row[f"{body}_cooked_is_collision"] = bool(cooked[body]["is_collision"])
        row[f"{body}_cooked_exact_consistent"] = bool(cooked[body]["exact_consistent"])
        row[f"{body}_cooked_epa_cap_saturated_any"] = bool(
            cooked[body]["epa_cap_saturated_any"]
        )
        row[f"{body}_cooked_overlap_state"] = cooked[body]["overlap_state"]
        row[f"{body}_cooked_witness_part_path"] = cooked[body]["witness_part_path"]
        row[f"{body}_cooked_witness_epa_contact_count"] = int(
            next(
                part["epa_contact_count"]
                for part in cooked[body]["parts"]
                if part["path"] == cooked[body]["witness_part_path"]
            )
        )
        row[f"{body}_cooked_nearest_point_geometry_m"] = json.dumps(
            cooked[body].get("nearest_point_geometry_m")
        )
        row[f"{body}_cooked_nearest_point_cylinder_m"] = json.dumps(
            cooked[body].get("nearest_point_cylinder_m")
        )
        row[f"{body}_raw_cooked_abs_delta_mm"] = (
            abs(float(raw_exact) - float(cooked_exact))
            if raw_exact is not None and cooked_exact is not None
            else None
        )
        row[f"{body}_raw_cooked_delta_le_0p5mm"] = bool(
            raw_exact is not None
            and cooked_exact is not None
            and abs(float(raw_exact) - float(cooked_exact)) <= TASK_FIDELITY_TOL_MM
        )
    return row


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_representation_figure(
    path: Path,
    title: str,
    inner: Any,
    raw_shapes: list[dict[str, Any]],
    cooked_by_body: dict[str, list[dict[str, Any]]],
    raw_set: dict[str, Any],
    cooked_set: dict[str, Any],
    canonical: dict[str, Any],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10.0, 7.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    obj_pos, obj_quat = d334._object_pose_w(inner)
    all_points = [d334._plot_cylinder(ax, obj_pos, obj_quat)]
    colors = {"link5": "tab:blue", "gripper_link": "tab:green"}
    raw_by_body = _queries_by_body(raw_set)
    cooked_queries = _queries_by_body(cooked_set) if cooked_set.get("queries") else {}
    for shape in raw_shapes:
        body = shape["body"]
        pos, quat = d334._body_pose_w(inner, body)
        rot = d332._quat_wxyz_to_rot(quat)
        raw = shape["_raw_verts"]
        stride = max(1, len(raw) // 1300)
        raw_world = (rot @ raw[::stride].T).T + pos
        ax.scatter(
            raw_world[:, 0],
            raw_world[:, 1],
            raw_world[:, 2],
            s=0.9,
            color=colors[body],
            alpha=0.42,
            label=f"{body} retained raw",
        )
        all_points.append(raw_world)
        for index, part in enumerate(cooked_by_body[body]):
            world = (rot @ part["_vertices"].T).T + pos
            ax.plot_trisurf(
                world[:, 0],
                world[:, 1],
                world[:, 2],
                triangles=part["_triangles"],
                color=colors[body],
                alpha=0.10,
                linewidth=0.12,
            )
            all_points.append(world)
    for body in d334.BODY_LABELS:
        witnesses = [(raw_by_body[body], "--", f"{body} raw witness")]
        if body in cooked_queries:
            witnesses.append((cooked_queries[body], "-", f"{body} cooked witness"))
        for query, style, label in witnesses:
            tool = np.asarray(query["nearest_point_geometry_m"], dtype=np.float64)
            cyl = np.asarray(query["nearest_point_cylinder_m"], dtype=np.float64)
            segment = np.vstack([tool, cyl])
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                linestyle=style,
                linewidth=1.8,
                color=colors[body],
                label=label,
            )
            all_points.append(segment)
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    target = origin + np.asarray(canonical["target_tcp_local_m"], dtype=np.float64)
    commanded = origin + np.asarray(canonical["commanded_tcp_local_m"], dtype=np.float64)
    actual = inner._tcp_pos_w[0].detach().cpu().numpy().astype(np.float64)
    target_rot = np.column_stack(
        [
            np.asarray(canonical["tangent_axis"], dtype=np.float64),
            np.cross([0.0, 0.0, 1.0], np.asarray(canonical["tangent_axis"], dtype=np.float64)),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ]
    )
    commanded_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    _link5_pos, actual_quat = d334._body_pose_w(inner, "link5")
    actual_rot = d332._quat_wxyz_to_rot(actual_quat)
    for point, rot, color, label, marker in (
        (target, target_rot, "red", "target TCP", "*"),
        (commanded, commanded_rot, "purple", "commanded TCP", "P"),
        (actual, actual_rot, "black", "actual TCP", "x"),
    ):
        ax.scatter([point[0]], [point[1]], [point[2]], color=color, marker=marker, s=75, label=label)
        all_points.append(point.reshape(1, 3))
        for axis_index, axis_color in enumerate(("r", "g", "b")):
            vec = rot[:, axis_index] * 0.018
            ax.quiver(
                point[0], point[1], point[2], vec[0], vec[1], vec[2],
                color=axis_color, linewidth=1.0, arrow_length_ratio=0.18,
            )
    d332._set_axes_equal(ax, np.vstack(all_points))
    metric_chunks = []
    for body in d334.BODY_LABELS:
        raw_value = raw_by_body[body].get("exact_signed_distance_mm")
        cooked_value = cooked_queries.get(body, {}).get("exact_signed_distance_mm")
        raw_text = "NA" if raw_value is None else f"{float(raw_value):+.3f}"
        cooked_text = "NA" if cooked_value is None else f"{float(cooked_value):+.3f}"
        metric_chunks.append(f"{body} raw={raw_text}mm cook={cooked_text}mm")
    metric_text = ", ".join(metric_chunks)
    ax.set_title(title + "\n" + metric_text)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return _rel(path)


def _decision_trace_row(inner: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    row = d337._decision_trace_row(inner, candidate)
    row["phase"] = "d339_prephysics_representation_decision"
    return row


def _fallback_candidate_without_raw(inner: Any, reason: str) -> dict[str, Any]:
    """Materialize frozen frames without pretending missing raw evidence exists."""
    canonical = d337._canonical_with_q5(
        OLD_RADIAL_NM / 1000.0,
        OLD_TANGENT_NM / 1000.0,
        Q5_OPEN_RAD,
    )
    counter_before = int(inner._sim_step_counter)
    command = d332._write_exact_state(
        inner,
        np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
        d332.OBJECT_CENTER_LOCAL_M,
    )
    counter_after = int(inner._sim_step_counter)
    alignment, frames = d335._alignment_at_current(inner, canonical)
    return {
        "stage": "d339_raw_source_contract_stop",
        "q5_rad": Q5_OPEN_RAD,
        "radial_offset_nm": OLD_RADIAL_NM,
        "tangent_offset_nm": OLD_TANGENT_NM,
        "radial_offset_mm": OLD_RADIAL_NM / 1.0e6,
        "tangent_offset_mm": OLD_TANGENT_NM / 1.0e6,
        "min_raw_clearance_mm": math.nan,
        "exact_min_clearance_mm": math.nan,
        "link5_exact_signed_distance_mm": None,
        "gripper_link_exact_signed_distance_mm": None,
        "legacy_alignment_pass": bool(alignment["pass_all"]),
        "raw_tool_clear_pass": False,
        "sim_step_counter_before": counter_before,
        "sim_step_counter_after": counter_after,
        "sim_step_counter_unchanged": counter_before == counter_after,
        "pass": False,
        "structured_stop_reason": reason,
        "_canonical": canonical,
        "_alignment": alignment,
        "_frames": frames,
        "_command": command,
    }


def _write_contract_stop_figure(
    path: Path,
    *,
    title: str,
    scene_checks: dict[str, bool],
    candidate: dict[str, Any],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=150)
    ax.axis("off")
    alignment = candidate.get("_alignment", {})
    lines = [
        title,
        "",
        "No partial/empty geometry union was queried.",
        "",
        *[f"{name}: {value}" for name, value in scene_checks.items()],
        "",
        f"frozen target: r=7mm, t=11mm, q5={Q5_OPEN_RAD}rad",
        f"tcp error: {alignment.get('tcp_pose_error_mm')} mm",
        f"jaw tangent error: {alignment.get('jaw_tangent_error_deg')} deg",
    ]
    ax.text(0.04, 0.95, "\n".join(lines), va="top", ha="left", family="monospace")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return _rel(path)


def _classify_static_d339(
    *,
    baseline_stats: dict[str, Any] | None,
    target_stats: dict[str, Any] | None,
    final_alignment: dict[str, Any] | None,
    raw_sets: list[dict[str, Any]],
    cooked_sets: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if baseline_stats is None or not baseline_stats["hard_gate_pass"]:
        return VERDICT_STATIC_MIXED, {
            "static_clean_pass": False,
            "interpretation": "the frozen sole-support baseline failed; target settle was not licensed",
        }
    if target_stats is None or final_alignment is None:
        return VERDICT_STATIC_MIXED, {
            "static_clean_pass": False,
            "interpretation": "the target pre-step raw/cooked contract failed; no target step ran",
        }
    _legacy_verdict, legacy = d335._classify_static(
        baseline_stats=baseline_stats,
        target_stats=target_stats,
        final_alignment=final_alignment,
        raw_sets=raw_sets,
    )
    cooked_clear_all = bool(
        cooked_sets
        and all(
            all(row["clear_pass"] and row["exact_consistent"] for row in item["queries"])
            for item in cooked_sets
        )
    )
    raw_exact_all = bool(
        raw_sets
        and all(
            all(
                row.get("exact_signed_distance_mm") is not None
                and row["exact_consistent"]
                and not row["epa_cap_saturated"]
                for row in item["queries"]
            )
            for item in raw_sets
        )
    )
    raw_cooked_fidelity_all = bool(
        len(raw_sets) == len(cooked_sets) == TARGET_SETTLE_STEPS + 1
        and all(
            all(
                _queries_by_body(raw_item)[body].get("exact_signed_distance_mm") is not None
                and _queries_by_body(cooked_item)[body].get("exact_signed_distance_mm") is not None
                and abs(
                    float(_queries_by_body(raw_item)[body]["exact_signed_distance_mm"])
                    - float(_queries_by_body(cooked_item)[body]["exact_signed_distance_mm"])
                )
                <= TASK_FIDELITY_TOL_MM
                for body in d334.BODY_LABELS
            )
            for raw_item, cooked_item in zip(raw_sets, cooked_sets, strict=True)
        )
    )
    checks = dict(legacy.get("checks", {}))
    checks["raw_exact_non_saturated_all_201_readings"] = raw_exact_all
    checks["live_cooked_tool_clear_all_201_readings"] = cooked_clear_all
    checks["raw_cooked_delta_le_0p5mm_all_201_readings"] = raw_cooked_fidelity_all
    static_clean = bool(
        legacy.get("static_clean_pass")
        and raw_exact_all
        and cooked_clear_all
        and raw_cooked_fidelity_all
    )
    verdict = VERDICT_STATIC_PASS if static_clean else VERDICT_STATIC_MIXED
    interpretation = (
        "the explicit live convex pieces retained raw/cooked clearance and every frozen static gate"
        if static_clean
        else "the conditional run failed at least one frozen raw/cooked, contact, support, root, disturbance, or alignment gate"
    )
    return verdict, {
        **legacy,
        "checks": checks,
        "static_clean_pass": static_clean,
        "interpretation": interpretation,
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    per_body = summary.get("representation_gate", {}).get("per_body", {})
    lines = [
        "# D339 cook-witness contract repair",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Gate / metric | Result |",
        "|---|---:|",
        f"| Asset build | `{summary['asset_build']['pass']}` |",
        f"| Live collider audit | `{summary.get('live_collider_audit', {}).get('pass')}` |",
        f"| D337 controls | `{summary.get('d337_controls', {}).get('pass')}` |",
        f"| Representation physics licensed | `{summary.get('representation_gate', {}).get('physics_licensed')}` |",
        f"| link5 raw / cooked / delta (mm) | `{per_body.get('link5', {}).get('raw_exact_signed_distance_mm')} / {per_body.get('link5', {}).get('cooked_exact_signed_distance_mm')} / {per_body.get('link5', {}).get('absolute_delta_mm')}` |",
        f"| gripper raw / cooked / delta (mm) | `{per_body.get('gripper_link', {}).get('raw_exact_signed_distance_mm')} / {per_body.get('gripper_link', {}).get('cooked_exact_signed_distance_mm')} / {per_body.get('gripper_link', {}).get('absolute_delta_mm')}` |",
        f"| Controlled physics steps | `{summary.get('physics', {}).get('controlled_steps_total')}` |",
        f"| Static clean | `{summary.get('classification', {}).get('static_clean_pass')}` |",
        f"| Artifact contract | `{summary.get('artifact_contract', {}).get('pass')}` |",
        "",
        summary.get("classification", {}).get("interpretation", "No classification") + ".",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    d332._runtime_versions()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical_summary = args.out_dir / "g0a_d339_cook_witness_contract_repair_summary.json"
    if canonical_summary.exists():
        raise RuntimeError(
            "a clean D339 outcome summary already exists; refuse to overwrite forward-only evidence"
        )
    d334_summary = json.loads(d337.D334_SUMMARY.read_text(encoding="utf-8"))
    d336_summary = json.loads(d337.D336_SUMMARY.read_text(encoding="utf-8"))
    d337_summary = json.loads(D337_SUMMARY.read_text(encoding="utf-8"))
    d338_summary = json.loads(D338_SUMMARY.read_text(encoding="utf-8"))
    pin = d334._pin_check()
    attempt1_integrity_before = _d338_attempt1_integrity()
    prebuild_checks = {
        "exact_output_root": args.out_dir.resolve() == DEFAULT_OUT_DIR.resolve(),
        "exact_asset_attempt2": args.asset_attempt == "attempt2",
        "attempt2_absent_before_invocation": not (
            args.out_dir / "collision_asset" / "attempt2"
        ).exists(),
        "source_robot_usd_path_frozen": (
            args.source_robot_usd_path.resolve() == d333.DEFAULT_ROBOT_USD.resolve()
        ),
        "urdf_path_frozen": args.urdf_path.resolve() == d333.DEFAULT_URDF.resolve(),
        "d338_helper_implementation_hash": _sha256(D338_SCRIPT)
        == PIN_D338_SCRIPT_SHA256,
        "d338_summary_hash": _sha256(D338_SUMMARY) == PIN_D338_SUMMARY_SHA256,
        "d338_verdict": d338_summary["verdict"]
        == "D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP",
        "d338_callback_not_recorded": not bool(
            d338_summary["registered_gate_failure"][
                "request_result_or_convex_count_recorded_before_failure"
            ]
        ),
        "d338_attempt1_integrity": bool(attempt1_integrity_before["pass"]),
        "d337_summary_hash": _sha256(D337_SUMMARY) == PIN_D337_SUMMARY_SHA256,
        "d337_verdict": d337_summary["verdict"] == "D337_G0A_STATIC_RUNTIME_MIXED_STOP",
        "d334_summary_hash": _sha256(d337.D334_SUMMARY) == d337.PIN_D334_SUMMARY_SHA256,
        "d336_summary_hash": _sha256(d337.D336_SUMMARY) == d337.PIN_D336_SUMMARY_SHA256,
        "urdf_hash": _sha256(args.urdf_path)
        == d334_summary["frozen_contract"]["urdf_sha256"],
        "seed_33201": int(args.seed) == 33201,
        "q5_open_1p5413": Q5_OPEN_RAD == d337.Q5_OPEN_RAD,
        "frozen_target_7_11": (OLD_RADIAL_NM, OLD_TANGENT_NM)
        == (d337.OLD_RADIAL_NM, d337.OLD_TANGENT_NM),
        "settle_steps_200": TARGET_SETTLE_STEPS == d337.TARGET_SETTLE_STEPS,
        "numpy_pin": bool(pin["numpy_pin_1_26_0"]),
        "psutil_pin": bool(pin["psutil_pin_5_9_8"]),
        "registered_decomposition_parameters": DECOMPOSITION_PARAMS
        == {
            "hull_vertex_limit": 64,
            "max_convex_hulls": 64,
            "voxel_resolution": 1_000_000,
            "error_percentage": 1.0,
            "min_thickness_m": 0.0001,
            "shrink_wrap": True,
        },
        "decomposition_parameters_equal_d338": DECOMPOSITION_PARAMS
        == d338.DECOMPOSITION_PARAMS,
        "collision_prim_paths_equal_d338": SOURCE_ASSET_PATHS
        == d338.SOURCE_ASSET_PATHS,
        "live_part_paths_equal_d338": LIVE_PART_PARENT_PATHS
        == d338.LIVE_PART_PARENT_PATHS,
        "task_tolerances_equal_d338": (
            TASK_FIDELITY_TOL_MM,
            CLEAR_GATE_MM,
            LIVE_VOLUME_PARITY_REL_TOL,
            PROPERTY_VOLUME_BINDING_REL_TOL,
            LIVE_SURFACE_PARITY_TOL_M,
            COLD_COOK_COORD_TOL_M,
        )
        == (
            d338.TASK_FIDELITY_TOL_MM,
            d338.CLEAR_GATE_MM,
            d338.LIVE_VOLUME_PARITY_REL_TOL,
            d338.PROPERTY_VOLUME_BINDING_REL_TOL,
            d338.LIVE_SURFACE_PARITY_TOL_M,
            d338.COLD_COOK_COORD_TOL_M,
        ),
        "registered_tolerances": (
            TASK_FIDELITY_TOL_MM,
            CLEAR_GATE_MM,
            PROPERTY_VOLUME_BINDING_REL_TOL,
        )
        == (0.5, 0.1, 0.05),
    }
    if not all(prebuild_checks.values()):
        raise RuntimeError(f"D339 prebuild frozen contract failed: {prebuild_checks}")
    asset_build = _build_derivative_asset(args, d334_summary)
    variant_robot_usd = REPO / asset_build["variant_robot_usd"]
    args.robot_usd_path = variant_robot_usd
    frozen_contract = {
        "artifact": "D339_FROZEN_CONTRACT",
        "checks": prebuild_checks,
        "pass": all(prebuild_checks.values()) and bool(asset_build["pass"]),
        "new_variable": ["cook_witness_contract"],
        "physical_variables_changed": [],
        "attempt1_integrity_before": attempt1_integrity_before,
        "q5_open_rad": Q5_OPEN_RAD,
        "radial_offset_mm": OLD_RADIAL_NM / 1.0e6,
        "tangent_offset_mm": OLD_TANGENT_NM / 1.0e6,
        "task_fidelity_tolerance_mm": TASK_FIDELITY_TOL_MM,
        "clear_gate_mm": CLEAR_GATE_MM,
        "property_volume_binding_relative_tolerance": (
            PROPERTY_VOLUME_BINDING_REL_TOL
        ),
        "environment": pin,
        "variant_robot_usd": _rel(variant_robot_usd),
        "variant_robot_usd_sha256": _sha256(variant_robot_usd),
    }
    _json_dump(args.out_dir / "d339_frozen_contract.json", frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError("D339 frozen contract failed after asset build")

    inner = d333._make_runtime_env(args)
    controlled_steps = 0
    snapshot_paths: list[Path] = []
    snapshots: list[str] = []
    try:
        inner.reset(seed=int(args.seed))
        phase_b_counter_start = int(inner._sim_step_counter)
        stage_contract = d333._stage_contract(inner)
        sensor_contract, filter_map = d333._sensor_contract(inner)
        try:
            raw_shapes, raw_source_contract = _build_retained_raw_shapes(
                inner, d334_summary
            )
        except Exception as error:  # noqa: BLE001 - registered source-contract STOP
            raw_shapes = []
            raw_source_contract = {
                "pass": False,
                "structured_exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        try:
            cooked_by_body, live_audit = _build_live_cooked_parts(
                inner, simulation_app, asset_build
            )
        except Exception as error:  # noqa: BLE001 - registered live-contract STOP
            cooked_by_body = {body: [] for body in d334.BODY_LABELS}
            live_audit = {
                "artifact": "D339_LIVE_COLLIDER_AUDIT_EXCEPTION_STOP",
                "pass": False,
                "structured_exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        scene_checks = {
            "stage_contract": bool(stage_contract["hard_contract_pass"]),
            "sensor_contract": bool(sensor_contract["hard_contract_pass"]),
            "retained_raw_source_contract": bool(raw_source_contract["pass"]),
            "live_collider_audit": bool(live_audit["pass"]),
        }
        scene_payload = {
            "artifact": "D339_PREPHYSICS_SCENE_CONTRACT",
            "checks": scene_checks,
            "pass": all(scene_checks.values()),
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
        }
        _json_dump(args.out_dir / "d339_prephysics_scene_contract.json", scene_payload)
        _json_dump(args.out_dir / "d339_live_collider_audit.json", live_audit)

        raw_prerequisites_pass = bool(
            scene_checks["stage_contract"]
            and scene_checks["sensor_contract"]
            and scene_checks["retained_raw_source_contract"]
        )
        if raw_prerequisites_pass:
            d336_rescore = d337._load_d336_rescore()
            cache = d337._Cache(inner, raw_shapes)
            controls = d337._negative_controls(
                inner, raw_shapes, d334_summary, d336_summary, d336_rescore, cache
            )
            candidate = d337._evaluate_candidate(
                inner,
                raw_shapes,
                OLD_RADIAL_NM,
                OLD_TANGENT_NM,
                Q5_OPEN_RAD,
                stage="d339_frozen_target",
            )
        else:
            controls = {
                "artifact": "D339_D337_CONTROLS_SKIPPED",
                "pass": False,
                "structured_stop_reason": (
                    "stage/sensor/retained-raw prerequisite failed; controls were not queried"
                ),
            }
            candidate = _fallback_candidate_without_raw(
                inner,
                "retained raw-source contract failed; no raw distance claim is available",
            )
        _json_dump(args.out_dir / "d339_d337_controls.json", controls)
        if all(scene_checks.values()):
            gate, decision_raw, decision_cooked = _representation_gate(
                inner, raw_shapes, cooked_by_body, candidate, controls, live_audit
            )
            gate["checks"]["scene_contract"] = True
        elif raw_prerequisites_pass:
            # A failed live audit is itself a registered clean pre-physics STOP.
            # Never query an empty or partial cooked union as if it represented
            # the intervention.
            decision_raw = d336._exact_raw_metrics(
                inner, raw_shapes, "d339_frozen_target_raw_contract_stop"
            )
            decision_cooked = {
                "pose": "d339_frozen_target_cooked_not_certified",
                "queries": [],
                "invalid_reason": "scene/live collider contract failed; partial union not queried",
            }
            anchor_checks = _frozen_anchor_checks(candidate)
            gate_checks = {
                "d337_controls": bool(controls["pass"]),
                "live_collider_audit": bool(live_audit["pass"]),
                "frozen_anchor": all(anchor_checks.values()),
                "both_bodies_task_fidelity": False,
                "scene_contract": False,
            }
            gate = {
                "artifact": "D339_REPRESENTATION_GATE",
                "checks": gate_checks,
                "anchor_checks": anchor_checks,
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "physics_licensed": False,
                "controlled_physics_steps": 0,
                "structured_stop_reason": (
                    "scene/live collider audit failed; cooked union distance was intentionally not queried"
                ),
            }
        else:
            decision_raw = {
                "pose": "d339_frozen_target_raw_not_certified",
                "queries": [],
                "invalid_reason": "retained raw-source contract failed",
            }
            decision_cooked = {
                "pose": "d339_frozen_target_cooked_not_queried",
                "queries": [],
                "invalid_reason": "raw prerequisite failed; no comparative union query",
            }
            anchor_checks = {
                "raw_source_available": False,
                "frozen_pose_materialized_without_step": bool(
                    candidate["sim_step_counter_unchanged"]
                ),
            }
            gate = {
                "artifact": "D339_REPRESENTATION_GATE",
                "checks": {
                    "d337_controls": False,
                    "live_collider_audit": bool(live_audit["pass"]),
                    "frozen_anchor": False,
                    "both_bodies_task_fidelity": False,
                    "scene_contract": False,
                },
                "anchor_checks": anchor_checks,
                "per_body": {},
                "contract_pass": False,
                "target_clear_and_faithful": False,
                "physics_licensed": False,
                "controlled_physics_steps": 0,
                "structured_stop_reason": (
                    "retained raw source unavailable; no controls or raw/cooked union query run"
                ),
            }
        phase_b_counter_end = int(inner._sim_step_counter)
        phase_b_counter_unchanged = phase_b_counter_end == phase_b_counter_start
        gate["checks"]["global_phase_b_sim_counter_unchanged"] = (
            phase_b_counter_unchanged
        )
        gate["global_phase_b_sim_counter"] = {
            "start": phase_b_counter_start,
            "end": phase_b_counter_end,
            "unchanged": phase_b_counter_unchanged,
        }
        if not phase_b_counter_unchanged:
            gate["contract_pass"] = False
            gate["physics_licensed"] = False
        _json_dump(args.out_dir / "d339_representation_gate.json", gate)

        artifact_errors: list[dict[str, str]] = []
        decision_png = args.out_dir / "d339_representation_decision.png"
        try:
            if decision_raw.get("queries"):
                snapshots.append(
                    _write_representation_figure(
                        decision_png,
                        "D339 frozen target: retained raw vs explicit live convex pieces",
                        inner,
                        raw_shapes,
                        cooked_by_body,
                        decision_raw,
                        decision_cooked,
                        candidate["_canonical"],
                    )
                )
            else:
                snapshots.append(
                    _write_contract_stop_figure(
                        decision_png,
                        title="D339 pre-physics source contract STOP",
                        scene_checks=scene_checks,
                        candidate=candidate,
                    )
                )
            snapshot_paths.append(decision_png)
        except Exception as error:  # noqa: BLE001 - retained artifact failure evidence
            artifact_errors.append(
                {
                    "artifact": "decision_png",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )

        baseline_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        raw_sets: list[dict[str, Any]] = []
        cooked_sets: list[dict[str, Any]] = []
        distance_trace: list[dict[str, Any]] = []
        baseline_stats = None
        target_stats = None
        final_alignment = None
        target_prestep_evidence = None

        if not gate["contract_pass"]:
            verdict = VERDICT_CONTRACT_FAIL
            classification = {
                "static_clean_pass": False,
                "interpretation": "a live source/owner/direct-cook/D337-control/zero-step contract failed",
            }
        elif not gate["target_clear_and_faithful"]:
            verdict = VERDICT_NOT_CLEAR
            classification = {
                "static_clean_pass": False,
                "interpretation": "the frozen target failed the registered cooked-clear or 0.5mm task-fidelity gate",
            }
        else:
            canonical = candidate["_canonical"]
            q_home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64))
            q_home[5] = Q5_OPEN_RAD
            home_target = d332._write_exact_state(inner, q_home, d332.OBJECT_CENTER_LOCAL_M)
            origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
            baseline_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
            root_pos, root_quat = d333._root_pose(inner)
            for step in range(d332.BASELINE_PHYSICS_STEPS):
                d332._physics_step(inner)
                controlled_steps += 1
                contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
                baseline_rows.append(
                    d333._state_row(
                        inner,
                        phase="d339_sole_support_baseline",
                        step=step,
                        command_target=home_target,
                        canonical=canonical,
                        object_start_w=baseline_start_w,
                        root_start_pos_w=root_pos,
                        root_start_quat_wxyz=root_quat,
                        contact=contact,
                    )
                )
            d333._write_trace_csv(args.out_dir / "d339_baseline_trace.csv", baseline_rows)
            baseline_stats = d333._baseline_statistics(
                baseline_rows, stage_contract=stage_contract, sensor_contract=sensor_contract
            )
            if baseline_stats["hard_gate_pass"]:
                command = d332._write_exact_state(
                    inner,
                    np.asarray(canonical["commanded_joint_rad"], dtype=np.float64),
                    d332.OBJECT_CENTER_LOCAL_M,
                )
                object_start_w = origin + d332.OBJECT_CENTER_LOCAL_M
                target_root_pos, target_root_quat = d333._root_pose(inner)
                pre_raw = d336._exact_raw_metrics(inner, raw_shapes, "d339_target_prestep_raw")
                pre_cooked = _cooked_union_distances(
                    inner, cooked_by_body, "d339_target_prestep_cooked"
                )
                raw_sets.append(pre_raw)
                cooked_sets.append(pre_cooked)
                distance_trace.append(_combined_distance_flat(-1, pre_raw, pre_cooked))
                pre_raw_by_body = _queries_by_body(pre_raw)
                pre_cooked_by_body = _queries_by_body(pre_cooked)
                prestep_pass = bool(
                    all(
                        pre_raw_by_body[body]["clear_pass"]
                        and pre_raw_by_body[body]["consistency_hard_check_pass"]
                        and pre_raw_by_body[body]["exact_consistent"]
                        and not pre_raw_by_body[body]["epa_cap_saturated"]
                        and pre_cooked_by_body[body]["clear_pass"]
                        and pre_cooked_by_body[body]["exact_consistent"]
                        and not pre_cooked_by_body[body]["epa_cap_saturated_any"]
                        and pre_raw_by_body[body]["exact_signed_distance_mm"] is not None
                        and pre_cooked_by_body[body]["exact_signed_distance_mm"] is not None
                        and abs(
                            float(pre_raw_by_body[body]["exact_signed_distance_mm"])
                            - float(pre_cooked_by_body[body]["exact_signed_distance_mm"])
                        )
                        <= TASK_FIDELITY_TOL_MM
                        for body in d334.BODY_LABELS
                    )
                )
                target_prestep_evidence = {
                    "pass": prestep_pass,
                    "raw": pre_raw,
                    "cooked": pre_cooked,
                    "combined_trace_row": distance_trace[-1],
                }
                _write_dict_csv(
                    args.out_dir / "d339_raw_cooked_distance_trace.csv",
                    distance_trace,
                )
                if prestep_pass:
                    for step in range(TARGET_SETTLE_STEPS):
                        d332._physics_step(inner)
                        controlled_steps += 1
                        contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
                        target_rows.append(
                            d333._state_row(
                                inner,
                                phase="d339_target_static_settle",
                                step=step,
                                command_target=command,
                                canonical=canonical,
                                object_start_w=object_start_w,
                                root_start_pos_w=target_root_pos,
                                root_start_quat_wxyz=target_root_quat,
                                contact=contact,
                            )
                        )
                        raw_now = d336._exact_raw_metrics(
                            inner, raw_shapes, f"d339_target_poststep_{step}_raw"
                        )
                        cooked_now = _cooked_union_distances(
                            inner, cooked_by_body, f"d339_target_poststep_{step}_cooked"
                        )
                        raw_sets.append(raw_now)
                        cooked_sets.append(cooked_now)
                        distance_trace.append(_combined_distance_flat(step, raw_now, cooked_now))
                    d333._write_trace_csv(
                        args.out_dir / "d339_target_static_trace.csv", target_rows
                    )
                    _write_dict_csv(
                        args.out_dir / "d339_raw_cooked_distance_trace.csv", distance_trace
                    )
                    target_stats = d333._target_statistics(target_rows)
                    target_stats["max_robot_root_position_drift_m"] = max(
                        float(row["robot_root_position_drift_m"]) for row in target_rows
                    )
                    target_stats["max_robot_root_rotation_drift_rad"] = max(
                        float(row["robot_root_rotation_drift_rad"]) for row in target_rows
                    )
                    final_alignment, _frames = d335._runtime_alignment(
                        inner, canonical, object_start_w
                    )
            verdict, classification = _classify_static_d339(
                baseline_stats=baseline_stats,
                target_stats=target_stats,
                final_alignment=final_alignment,
                raw_sets=raw_sets,
                cooked_sets=cooked_sets,
            )

        marker_frames = target_rows[-1]["frames"] if target_rows else candidate["_frames"]
        try:
            marker_status = draw_frames(
                marker_frames, prim_path="/World/D339RepresentationFrames"
            )
        except Exception as error:  # noqa: BLE001 - artifact gate must retain science
            marker_status = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            artifact_errors.append(
                {
                    "artifact": "frame_markers",
                    "error": marker_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        if target_rows:
            final_png = args.out_dir / "d339_static_final.png"
            try:
                snapshots.append(
                    _write_representation_figure(
                        final_png,
                        "D339 explicit-convex conditional static settle final",
                        inner,
                        raw_shapes,
                        cooked_by_body,
                        raw_sets[-1],
                        cooked_sets[-1],
                        candidate["_canonical"],
                    )
                )
                snapshot_paths.append(final_png)
            except Exception as error:  # noqa: BLE001
                artifact_errors.append(
                    {
                        "artifact": "final_png",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
        rrd_path = args.out_dir / "d339_collision_representation_trace.rrd"
        try:
            rrd_trace = target_rows if target_rows else [_decision_trace_row(inner, candidate)]
            rrd_status = log_rerun(
                rrd_path,
                frames=marker_frames,
                joint_state={
                    "label": "d339_collision_representation_repair",
                    "object": "cylinder_d34_h90",
                    "q5_open_rad": Q5_OPEN_RAD,
                    "controlled_physics_steps_total": controlled_steps,
                    "physics_licensed": gate["physics_licensed"],
                },
                joint_trace=rrd_trace,
                urdf_path=args.urdf_path,
                live_viewer=False,
                app_id="roarm_g0a_d339_collision_representation_repair",
            )
        except Exception as error:  # noqa: BLE001
            rrd_status = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            artifact_errors.append(
                {
                    "artifact": "rrd",
                    "error": rrd_status["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        if bool(rrd_status.get("ok")):
            rrd_status["nonzero_file"] = bool(rrd_path.is_file() and rrd_path.stat().st_size > 0)
        distance_trace_path = args.out_dir / "d339_raw_cooked_distance_trace.csv"
        artifact_checks = {
            "snapshot_count_1_to_3": 1 <= len(snapshot_paths) <= 3,
            "snapshots_exist_nonzero": all(
                path.is_file() and path.stat().st_size > 0 for path in snapshot_paths
            ),
            "marker_status_ok": bool(marker_status.get("ok")),
            "rrd_status_ok": bool(rrd_status.get("ok")),
            "rrd_nonzero": bool(rrd_status.get("nonzero_file")),
            "required_distance_trace_present": bool(
                target_prestep_evidence is None
                or (
                    distance_trace_path.is_file()
                    and distance_trace_path.stat().st_size > 0
                    and len(distance_trace) == len(raw_sets) == len(cooked_sets)
                )
            ),
            "no_artifact_exceptions": not artifact_errors,
        }
        artifact_contract = {"checks": artifact_checks, "pass": all(artifact_checks.values())}
        scientific_verdict = verdict
        final_verdict = scientific_verdict if artifact_contract["pass"] else VERDICT_VIZ_FAIL
        summary = {
            "verdict": final_verdict,
            "scientific_verdict_before_artifact_gate": scientific_verdict,
            "active_case": "G0a cylinder D34xH90 cook-witness contract repair",
            "new_variable": ["cook_witness_contract"],
            "physical_variables_changed": [],
            "frozen_contract": frozen_contract,
            "asset_build": asset_build,
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "raw_source_contract": raw_source_contract,
            "live_collider_audit": live_audit,
            "d337_controls": controls,
            "frozen_candidate": d337._candidate_public(candidate),
            "frozen_candidate_alignment": candidate["_alignment"],
            "representation_gate": gate,
            "decision_raw": decision_raw,
            "decision_cooked": decision_cooked,
            "classification": classification,
            "physics": {
                "executed": bool(gate["physics_licensed"]),
                "controlled_steps_total": controlled_steps,
                "baseline": baseline_stats,
                "target_static": target_stats,
                "final_alignment": final_alignment,
                "target_prestep": target_prestep_evidence,
                "raw_distance_sets_count": len(raw_sets),
                "cooked_distance_sets_count": len(cooked_sets),
            },
            "outcome_guards": {
                "g0a_pass": False,
                "ten_trial_run": False,
                "ladder_promoted": False,
                "canonical_asset_changed": False,
                "target_or_q5_changed": False,
                "stop_after_d339": True,
            },
            "visualization": {
                "snapshots": snapshots,
                "snapshot_count": len(snapshots),
                "marker_status": marker_status,
                "rrd_status": rrd_status,
                "artifact_errors": artifact_errors,
            },
            "artifact_contract": artifact_contract,
            "artifacts": {
                name: (_rel(args.out_dir / filename) if (args.out_dir / filename).is_file() else None)
                for name, filename in (
                    ("frozen_contract", "d339_frozen_contract.json"),
                    ("scene_contract", "d339_prephysics_scene_contract.json"),
                    ("live_collider_audit", "d339_live_collider_audit.json"),
                    ("d337_controls", "d339_d337_controls.json"),
                    ("representation_gate", "d339_representation_gate.json"),
                    ("baseline_trace", "d339_baseline_trace.csv"),
                    ("target_trace", "d339_target_static_trace.csv"),
                    ("distance_trace", "d339_raw_cooked_distance_trace.csv"),
                    ("decision_png", "d339_representation_decision.png"),
                    ("final_png", "d339_static_final.png"),
                    ("rrd", "d339_collision_representation_trace.rrd"),
                )
            },
            "non_goals_respected": [
                "no canonical USD/URDF/STL rewrite",
                "no target/q5/IK/waypoint/seed/solver change",
                "no 10-trial/G0b/RL/PPO/randomization/VLA/RoArm/B200/cube",
            ],
        }
        summary["artifacts"]["asset_build_manifest"] = asset_build.get("manifest_path")
        summary["artifacts"]["hull_manifest"] = asset_build.get("hull_manifest_path")
        summary["artifacts"]["cook_witness_manifest"] = asset_build.get(
            "cook_witness_manifest_path"
        )
        _json_dump(canonical_summary, summary)
        _write_summary_markdown(
            args.out_dir / "g0a_d339_cook_witness_contract_repair_summary.md", summary
        )
        return summary
    finally:
        inner.close()


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source_robot_usd_path", type=Path, default=d333.DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--asset_attempt", type=str, choices=("attempt2",), default="attempt2")
    parser.add_argument("--seed", type=int, default=33201)


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    attempt_dir_at_invocation_start = args.out_dir / "collision_asset" / args.asset_attempt
    attempt_existed_at_invocation_start = attempt_dir_at_invocation_start.exists()
    summary_existed_at_invocation_start = (
        args.out_dir / "g0a_d339_cook_witness_contract_repair_summary.json"
    ).exists()
    if attempt_existed_at_invocation_start or summary_existed_at_invocation_start:
        print(
            "D339 immutable-evidence refusal: attempt2 or clean summary already exists; "
            "no Isaac app was launched and no evidence was modified.",
            flush=True,
        )
        return 1
    launcher = AppLauncher(args)
    try:
        try:
            summary = _run(args, launcher.app)
            gate = summary.get("representation_gate", {})
            print(
                f"{summary['verdict']}: build={summary['asset_build']['pass']} "
                f"live={summary['live_collider_audit']['pass']} "
                f"licensed={gate.get('physics_licensed')} "
                f"steps={summary['physics']['controlled_steps_total']}",
                flush=True,
            )
            return 0 if bool(summary["artifact_contract"]["pass"]) else 1
        except Exception:
            traceback.print_exc()
            error_text = traceback.format_exc()
            if attempt_existed_at_invocation_start or summary_existed_at_invocation_start:
                print(
                    "D339 immutable-evidence refusal: invocation began with an existing "
                    "attempt2 or clean summary; no D339 evidence was modified.",
                    flush=True,
                )
                return 1
            args.out_dir.mkdir(parents=True, exist_ok=True)
            attempt_dir = args.out_dir / "collision_asset" / args.asset_attempt
            attempt_dir.mkdir(parents=True, exist_ok=True)
            build_manifest = attempt_dir / "d339_asset_build_manifest.json"
            witness_manifest = attempt_dir / "d339_cook_witness_manifest.json"
            build_pass = False
            witness_pass = None
            if witness_manifest.is_file():
                try:
                    witness_pass = bool(
                        json.loads(witness_manifest.read_text(encoding="utf-8")).get("pass")
                    )
                except Exception:
                    witness_pass = False
            if build_manifest.is_file():
                try:
                    build_pass = bool(
                        json.loads(build_manifest.read_text(encoding="utf-8")).get("pass")
                    )
                except Exception:
                    build_pass = False
            else:
                _json_dump(
                    build_manifest,
                    {
                        "artifact": "D339_ASSET_BUILD_FAILURE_MANIFEST",
                        "asset_attempt": args.asset_attempt,
                        "pass": False,
                        "variant_asset_dir": _rel(
                            attempt_dir / "roarm_m3_fullmesh_convex_parts"
                        ),
                        "interpretation": (
                            "the registered build/prebuild path failed before a passing asset manifest"
                        ),
                        "error": error_text,
                    },
                )
            if witness_pass is False:
                verdict = VERDICT_WITNESS_FAIL
            else:
                verdict = VERDICT_CONTRACT_FAIL if build_pass else VERDICT_BUILD_FAIL
            abort = {
                "verdict": verdict,
                "interpretation": "the D339 invocation aborted before a clean registered outcome",
                "asset_attempt": args.asset_attempt,
                "build_manifest": _rel(build_manifest) if build_manifest.is_file() else None,
                "cook_witness_manifest": (
                    _rel(witness_manifest) if witness_manifest.is_file() else None
                ),
                "attempt1_integrity_after_abort": _d338_attempt1_integrity(),
                "error": error_text,
            }
            # Invocation failures are immutable attempt-local evidence.  They
            # never occupy or overwrite the canonical clean-outcome summary.
            abort_index = 1
            while (attempt_dir / f"d339_invocation_abort_{abort_index:03d}.json").exists():
                abort_index += 1
            _json_dump(
                attempt_dir / f"d339_invocation_abort_{abort_index:03d}.json",
                abort,
            )
            return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
