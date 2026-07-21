#!/usr/bin/env python3
"""One-shot Isaac/PhysX worker for the D373 P34 live-asset identity preflight.

The worker clones the frozen D344 physical-reference asset once, replaces only
the two active A64 collision subtrees with the frozen D372 P34 geometry, and
performs USD readback, PhysX property queries, and synchronous prototype then
instance convex-callback requests.  It never constructs SimulationContext,
plays a timeline, advances physics, writes a cylinder pose, evaluates q5, or
queries contacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

BODIES = ("link5", "gripper_link")
EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18}
EXPECTED_HEAD = "5214721e91bd23b224998cba2b13a1f76294edad"
GEOMETRY_PATH = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)
BASE_ASSET_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/"
    "roarm_m3_fullmesh_fixed_point_parts"
)
BASE_ROOT_USD = BASE_ASSET_DIR / "roarm_m3.usd"
BASE_PHYSICS_USD = BASE_ASSET_DIR / "configuration/roarm_m3_physics.usd"
ASSET_NAME = "roarm_m3_p34_semantic_compound"
PHYSICS_REL = Path("configuration/roarm_m3_physics.usd")
CLAIM_NAME = "d373_worker_claim.json"
SUMMARY_NAME = "d373_worker_raw_summary.json"
PRECLOSE_NAME = "d373_worker_preclose_sentinel.json"
EXCEPTION_NAME = "d373_worker_exception.json"
WITNESS_DIR_NAME = "callback_witnesses"
PARTS_PARENT = {body: f"/colliders/{body}/d373_p34_parts" for body in BODIES}
OLD_PARTS_PARENT = {body: f"/colliders/{body}/d338_convex_parts" for body in BODIES}
LEGACY_SPEC = {
    "link5": "/colliders/link5/link5/node_STL_BINARY_",
    "gripper_link": "/colliders/gripper_link/gripper_link/node_STL_BINARY_",
}
LIVE_BODY_PATH = {body: f"/World/Robot/{body}" for body in BODIES}
LIVE_PARENT_PATH = {
    body: f"/World/Robot/{body}/collisions/d373_p34_parts" for body in BODIES
}
LIVE_OLD_PARENT_PATH = {
    body: f"/World/Robot/{body}/collisions/d338_convex_parts" for body in BODIES
}
LIVE_LEGACY_PATH = {
    "link5": "/World/Robot/link5/collisions/link5/node_STL_BINARY_",
    "gripper_link": "/World/Robot/gripper_link/collisions/gripper_link/node_STL_BINARY_",
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_bytes(value: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype)).tobytes(order="C")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return _rel(value)
    raise TypeError(type(value).__name__)


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(out_dir: Path, name: str, **fields: Any) -> None:
    path = out_dir / "d373_phase_markers.jsonl"
    ordinal = 1
    if path.is_file():
        ordinal = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) + 1
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _part_prim_name(index: int, part: dict[str, Any]) -> str:
    name = str(part["name"])
    if not name.replace("_", "").isalnum():
        raise RuntimeError(f"unsafe D372 part name: {name!r}")
    return f"p{index:03d}_{name}"


def _expected_f32_record(index: int, part: dict[str, Any]) -> dict[str, Any]:
    vertices_f64 = np.asarray(part["vertices"], dtype="<f8")
    triangles_i64 = np.asarray(part["triangles"], dtype="<i8")
    vertices_f32 = np.asarray(vertices_f64, dtype="<f4")
    counts_i32 = np.full(len(triangles_i64), 3, dtype="<i4")
    indices_i32 = np.asarray(triangles_i64.reshape(-1), dtype="<i4")
    f64_digest = _sha_bytes(_array_bytes(vertices_f64, "<f8") + _array_bytes(triangles_i64, "<i8"))
    payload = _array_bytes(vertices_f32, "<f4") + _array_bytes(counts_i32, "<i4") + _array_bytes(indices_i32, "<i4")
    return {
        "index": index,
        "name": str(part["name"]),
        "prim_name": _part_prim_name(index, part),
        "role": str(part["role"]),
        "source": str(part["source"]),
        "d372_payload_sha256_registered": str(part["payload_sha256"]),
        "d372_payload_sha256_recomputed": f64_digest,
        "d372_payload_exact": f64_digest == str(part["payload_sha256"]),
        "points_f32_sha256": _sha_bytes(_array_bytes(vertices_f32, "<f4")),
        "face_counts_i32_sha256": _sha_bytes(_array_bytes(counts_i32, "<i4")),
        "face_indices_i32_sha256": _sha_bytes(_array_bytes(indices_i32, "<i4")),
        "authored_f32_topology_payload_sha256": _sha_bytes(payload),
        "vertices_f32": vertices_f32,
        "face_counts_i32": counts_i32,
        "face_indices_i32": indices_i32,
        "triangles_i32": np.asarray(triangles_i64, dtype="<i4"),
        "topology_volume_m3": float(part["topology_volume_m3"]),
        "bounds_m": part["bounds_m"],
    }


def _asset_hashes(asset_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(asset_dir)): _sha(path)
        for path in sorted(item for item in asset_dir.rglob("*") if item.is_file())
    }


def _materialize_asset(out_dir: Path, geometry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    collision_root = out_dir / "collision_asset"
    variant_dir = collision_root / ASSET_NAME
    if collision_root.exists() or variant_dir.exists():
        raise RuntimeError("D373 derivative collision asset path already exists; overwrite refused")
    collision_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASE_ASSET_DIR, variant_dir)
    physics_path = variant_dir / PHYSICS_REL
    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open copied physics layer {physics_path}")

    expected: dict[str, list[dict[str, Any]]] = {body: [] for body in BODIES}
    for body in BODIES:
        legacy = stage.GetPrimAtPath(LEGACY_SPEC[body])
        if not legacy.IsValid() or not legacy.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"missing frozen disabled legacy collider {LEGACY_SPEC[body]}")
        legacy_enabled = UsdPhysics.CollisionAPI(legacy).GetCollisionEnabledAttr().Get()
        if legacy_enabled is not False:
            raise RuntimeError(f"legacy collider is not frozen disabled for {body}: {legacy_enabled!r}")
        old = stage.GetPrimAtPath(OLD_PARTS_PARENT[body])
        if not old.IsValid():
            raise RuntimeError(f"missing frozen A64 subtree {OLD_PARTS_PARENT[body]}")
        if not stage.RemovePrim(OLD_PARTS_PARENT[body]):
            raise RuntimeError(f"failed to remove copied A64 subtree {OLD_PARTS_PARENT[body]}")
        UsdGeom.Xform.Define(stage, PARTS_PARENT[body])
        parts = list(geometry["parts"][body])
        if len(parts) != EXPECTED_COUNTS[body]:
            raise RuntimeError(f"D372 {body} count changed: {len(parts)}")
        for index, part in enumerate(parts):
            record = _expected_f32_record(index, part)
            if not record["d372_payload_exact"]:
                raise RuntimeError(f"D372 Float64 payload digest mismatch for {body}/{part['name']}")
            path = f"{PARTS_PARENT[body]}/{record['prim_name']}"
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(
                [Gf.Vec3f(*[float(value) for value in row]) for row in record["vertices_f32"]]
            )
            mesh.CreateFaceVertexCountsAttr([int(value) for value in record["face_counts_i32"]])
            mesh.CreateFaceVertexIndicesAttr([int(value) for value in record["face_indices_i32"]])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            collision.CreateCollisionEnabledAttr(True)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mesh_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            hull_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim())
            hull_api.CreateHullVertexLimitAttr(64)
            hull_api.CreateMinThicknessAttr(0.0001)
            expected[body].append(
                {
                    **{key: value for key, value in record.items() if not isinstance(value, np.ndarray)},
                    "direct_path": path,
                    "live_path": f"{LIVE_PARENT_PATH[body]}/{record['prim_name']}",
                }
            )
    stage.GetRootLayer().Save()
    return variant_dir, {
        "asset_write_count": 1,
        "variant_dir": _rel(variant_dir),
        "root_usd": _rel(variant_dir / "roarm_m3.usd"),
        "physics_usd": _rel(physics_path),
        "expected": expected,
        "base_file_hashes": _asset_hashes(BASE_ASSET_DIR),
        "variant_file_hashes": _asset_hashes(variant_dir),
    }


def _readback_physics_layer(physics_path: Path, expected: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(physics_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open D373 physics layer {physics_path}")
    rows = []
    for body in BODIES:
        for exp in expected[body]:
            prim = stage.GetPrimAtPath(exp["direct_path"])
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(
                [[float(value) for value in row] for row in list(mesh.GetPointsAttr().Get() or [])],
                dtype="<f4",
            )
            counts = np.asarray(list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype="<i4")
            indices = np.asarray(list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype="<i4")
            collision = UsdPhysics.CollisionAPI(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
            observed = {
                "body": body,
                "name": exp["name"],
                "role": exp["role"],
                "prim_name": exp["prim_name"],
                "direct_path": exp["direct_path"],
                "live_path": exp["live_path"],
                "points_f32_sha256": _sha_bytes(_array_bytes(points, "<f4")),
                "face_counts_i32_sha256": _sha_bytes(_array_bytes(counts, "<i4")),
                "face_indices_i32_sha256": _sha_bytes(_array_bytes(indices, "<i4")),
                "authored_f32_topology_payload_sha256": _sha_bytes(
                    _array_bytes(points, "<f4")
                    + _array_bytes(counts, "<i4")
                    + _array_bytes(indices, "<i4")
                ),
                "points_f32": points.astype(np.float64).tolist(),
                "face_vertex_counts": counts.tolist(),
                "face_vertex_indices": indices.tolist(),
                "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
                "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
                "collision_enabled": bool(collision.GetCollisionEnabledAttr().Get()),
                "approximation": str(mesh_api.GetApproximationAttr().Get()),
                "hull_vertex_limit": int(hull_api.GetHullVertexLimitAttr().Get()),
                "min_thickness_m": float(hull_api.GetMinThicknessAttr().Get()),
            }
            checks = {
                "points_f32_exact": observed["points_f32_sha256"] == exp["points_f32_sha256"],
                "counts_i32_exact": observed["face_counts_i32_sha256"] == exp["face_counts_i32_sha256"],
                "indices_i32_exact": observed["face_indices_i32_sha256"] == exp["face_indices_i32_sha256"],
                "aggregate_payload_exact": observed["authored_f32_topology_payload_sha256"]
                == exp["authored_f32_topology_payload_sha256"],
                "subdivision_none": observed["subdivision_scheme"] == "none",
                "double_sided_true": observed["double_sided"],
                "collision_enabled_true": observed["collision_enabled"],
                "approximation_convex_hull": observed["approximation"] == "convexHull",
                "hull_vertex_limit_64": observed["hull_vertex_limit"] == 64,
                "min_thickness_frozen": abs(observed["min_thickness_m"] - 0.0001) <= 1.0e-12,
            }
            rows.append({**observed, "checks": checks, "pass": all(checks.values())})
    return {
        "rows": rows,
        "count": len(rows),
        "counts": {body: sum(row["body"] == body for row in rows) for body in BODIES},
        "pass": len(rows) == 34 and all(row["pass"] for row in rows),
    }


def _mass_row(stage: Any, path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    api = UsdPhysics.MassAPI(prim)
    com = api.GetCenterOfMassAttr().Get()
    inertia = api.GetDiagonalInertiaAttr().Get()
    axes = api.GetPrincipalAxesAttr().Get()
    return {
        "path": path,
        "mass_kg": float(api.GetMassAttr().Get()),
        "center_of_mass_m": [float(value) for value in com],
        "diagonal_inertia": [float(value) for value in inertia],
        "principal_axes_wxyz": [float(axes.GetReal()), *[float(value) for value in axes.GetImaginary()]],
    }


def _mass_api_audit(base_root: Path, variant_root: Path) -> dict[str, Any]:
    from pxr import Usd

    base_stage = Usd.Stage.Open(str(base_root), load=Usd.Stage.LoadAll)
    variant_stage = Usd.Stage.Open(str(variant_root), load=Usd.Stage.LoadAll)
    if base_stage is None or variant_stage is None:
        raise RuntimeError("failed to open base/variant root USD for MassAPI audit")
    rows = {}
    for body in BODIES:
        base = _mass_row(base_stage, f"/roarm_m3/{body}")
        variant = _mass_row(variant_stage, f"/roarm_m3/{body}")
        keys = ("mass_kg", "center_of_mass_m", "diagonal_inertia", "principal_axes_wxyz")
        deltas = {
            key: float(
                np.max(
                    np.abs(
                        np.asarray(base[key], dtype=np.float64)
                        - np.asarray(variant[key], dtype=np.float64)
                    )
                )
            )
            for key in keys
        }
        rows[body] = {
            "base": base,
            "variant": variant,
            "max_abs_delta_by_field": deltas,
            "pass": max(deltas.values()) <= 1.0e-12,
        }
    return {"bodies": rows, "tolerance": 1.0e-12, "pass": all(row["pass"] for row in rows.values())}


def _canonical_outside_subtree_diff(base_root: Path, variant_root: Path) -> dict[str, Any]:
    from sim_scripts import cyl34_top_view_d345_grasp_g0a_deterministic_usd_metadata_comparator as d345

    base_rows, base_meta = d345._stage_rows(base_root, set())
    variant_rows, variant_meta = d345._stage_rows(variant_root, set())

    def retained(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if "/d338_convex_parts" not in row["path"]
            and "/d373_p34_parts" not in row["path"]
        ]

    base_keep = retained(base_rows)
    variant_keep = retained(variant_rows)
    base_blob = json.dumps(base_keep, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    variant_blob = json.dumps(variant_keep, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    base_a64_paths = sorted(row["path"] for row in base_rows if "/d338_convex_parts" in row["path"])
    variant_a64_paths = sorted(row["path"] for row in variant_rows if "/d338_convex_parts" in row["path"])
    variant_p34_paths = sorted(row["path"] for row in variant_rows if "/d373_p34_parts" in row["path"])
    variant_p34_mesh_paths = sorted(
        row["path"]
        for row in variant_rows
        if "/d373_p34_parts/" in row["path"] and row["type_name"] == "Mesh"
    )
    return {
        "encoder": "D345 CanonicalUsdEncoder; closed typed address-free serialization",
        "base_meta": base_meta,
        "variant_meta": variant_meta,
        "base_retained_sha256": _sha_bytes(base_blob.encode("utf-8")),
        "variant_retained_sha256": _sha_bytes(variant_blob.encode("utf-8")),
        "base_retained_row_count": len(base_keep),
        "variant_retained_row_count": len(variant_keep),
        "base_a64_path_count": len(base_a64_paths),
        "variant_a64_path_count": len(variant_a64_paths),
        "variant_p34_path_count": len(variant_p34_paths),
        "variant_p34_mesh_path_count": len(variant_p34_mesh_paths),
        "base_a64_paths": base_a64_paths,
        "variant_p34_paths": variant_p34_paths,
        "checks": {
            "outside_registered_subtrees_bit_exact": base_blob == variant_blob,
            "base_contains_a64": len(base_a64_paths) > 0,
            "variant_contains_no_a64": len(variant_a64_paths) == 0,
            "variant_contains_p34": len(variant_p34_paths) > 0,
            "variant_contains_exactly_34_p34_meshes": len(variant_p34_mesh_paths)
            == 34,
            "no_runtime_address_patterns": base_meta["runtime_address_pattern_count"] == 0
            and variant_meta["runtime_address_pattern_count"] == 0,
            "no_unsupported_types": base_meta["unsupported_type_count"] == 0
            and variant_meta["unsupported_type_count"] == 0,
        },
    }


def _nonphysics_copy_audit(asset_record: dict[str, Any]) -> dict[str, Any]:
    base = dict(asset_record["base_file_hashes"])
    variant = dict(asset_record["variant_file_hashes"])
    keys = sorted(set(base) | set(variant))
    nonphysics = [key for key in keys if key != str(PHYSICS_REL)]
    checks = {key: base.get(key) == variant.get(key) for key in nonphysics}
    return {
        "paths": nonphysics,
        "hash_equal": checks,
        "physics_layer_changed": base.get(str(PHYSICS_REL)) != variant.get(str(PHYSICS_REL)),
        "pass": bool(checks and all(checks.values()) and base.get(str(PHYSICS_REL)) != variant.get(str(PHYSICS_REL))),
    }


def _make_inspection_stage(variant_root: Path) -> tuple[Any, int]:
    from pxr import Usd, UsdGeom, UsdUtils

    stage = Usd.Stage.CreateInMemory("d373_p34_identity_inspection.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference(str(variant_root.resolve()))
    robot.GetPrim().SetInstanceable(True)
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
    if not stage_id.IsValid():
        raise RuntimeError("failed to insert D373 inspection stage into StageCache")
    return stage, int(stage_id.ToLongInt())


def _owner_path(prim: Any) -> str | None:
    from pxr import UsdPhysics

    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current.GetPath().pathString
        current = current.GetParent()
    return None


def _live_inventory(stage: Any, expected: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    expected_by_path = {
        row["live_path"]: row for body in BODIES for row in expected[body]
    }
    rows = []
    all_collision_rows = []
    active_a64_paths = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        body_scope = next(
            (
                body
                for body in BODIES
                if path.startswith(LIVE_BODY_PATH[body] + "/")
            ),
            None,
        )
        if body_scope is not None and prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled_value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            all_collision_rows.append(
                {
                    "body": body_scope,
                    "path": path,
                    "collision_enabled": True
                    if enabled_value is None
                    else bool(enabled_value),
                }
            )
        if any(path.startswith(LIVE_OLD_PARENT_PATH[body] + "/") for body in BODIES):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and bool(
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            ):
                active_a64_paths.append(path)
        if path not in expected_by_path:
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        prototype = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else None
        instance_mesh = UsdGeom.Mesh(prim)
        instance_points = np.asarray(
            [
                [float(value) for value in point]
                for point in list(instance_mesh.GetPointsAttr().Get() or [])
            ],
            dtype="<f4",
        )
        instance_counts = np.asarray(
            list(instance_mesh.GetFaceVertexCountsAttr().Get() or []), dtype="<i4"
        )
        instance_indices = np.asarray(
            list(instance_mesh.GetFaceVertexIndicesAttr().Get() or []), dtype="<i4"
        )
        prototype_points_sha256 = None
        prototype_counts_sha256 = None
        prototype_indices_sha256 = None
        if prototype is not None and prototype.IsValid():
            prototype_mesh = UsdGeom.Mesh(prototype)
            prototype_points_sha256 = _sha_bytes(
                _array_bytes(
                    np.asarray(
                        [
                            [float(value) for value in point]
                            for point in list(
                                prototype_mesh.GetPointsAttr().Get() or []
                            )
                        ],
                        dtype="<f4",
                    ),
                    "<f4",
                )
            )
            prototype_counts_sha256 = _sha_bytes(
                _array_bytes(
                    np.asarray(
                        list(
                            prototype_mesh.GetFaceVertexCountsAttr().Get() or []
                        ),
                        dtype="<i4",
                    ),
                    "<i4",
                )
            )
            prototype_indices_sha256 = _sha_bytes(
                _array_bytes(
                    np.asarray(
                        list(
                            prototype_mesh.GetFaceVertexIndicesAttr().Get() or []
                        ),
                        dtype="<i4",
                    ),
                    "<i4",
                )
            )
        instance_points_sha256 = _sha_bytes(_array_bytes(instance_points, "<f4"))
        instance_counts_sha256 = _sha_bytes(_array_bytes(instance_counts, "<i4"))
        instance_indices_sha256 = _sha_bytes(_array_bytes(instance_indices, "<i4"))
        row = {
            "body": expected_by_path[path]["live_path"].split("/")[3],
            "name": expected_by_path[path]["name"],
            "role": expected_by_path[path]["role"],
            "prim_name": expected_by_path[path]["prim_name"],
            "instance_path": path,
            "owner_path": _owner_path(prim),
            "is_instance_proxy": bool(prim.IsInstanceProxy()),
            "prototype_path": (
                prototype.GetPath().pathString if prototype is not None and prototype.IsValid() else None
            ),
            "is_mesh": bool(prim.IsA(UsdGeom.Mesh)),
            "collision_enabled": bool(collision.GetCollisionEnabledAttr().Get()),
            "approximation": str(mesh_api.GetApproximationAttr().Get()),
            "hull_vertex_limit": int(hull_api.GetHullVertexLimitAttr().Get()),
            "min_thickness_m": float(hull_api.GetMinThicknessAttr().Get()),
            "instance_points_f32_sha256": instance_points_sha256,
            "instance_face_counts_i32_sha256": instance_counts_sha256,
            "instance_face_indices_i32_sha256": instance_indices_sha256,
            "prototype_points_f32_sha256": prototype_points_sha256,
            "prototype_face_counts_i32_sha256": prototype_counts_sha256,
            "prototype_face_indices_i32_sha256": prototype_indices_sha256,
        }
        checks = {
            "instance_proxy": row["is_instance_proxy"],
            "prototype_distinct_valid": row["prototype_path"] is not None
            and row["prototype_path"] != row["instance_path"],
            "owner_exact": row["owner_path"] == LIVE_BODY_PATH[row["body"]],
            "mesh": row["is_mesh"],
            "collision_enabled": row["collision_enabled"],
            "convex_hull_binding": row["approximation"] == "convexHull",
            "hull_vertex_limit_64": row["hull_vertex_limit"] == 64,
            "min_thickness_frozen": abs(row["min_thickness_m"] - 0.0001) <= 1.0e-12,
            "live_instance_authored_points_exact": instance_points_sha256
            == expected_by_path[path]["points_f32_sha256"],
            "live_instance_authored_counts_exact": instance_counts_sha256
            == expected_by_path[path]["face_counts_i32_sha256"],
            "live_instance_authored_indices_exact": instance_indices_sha256
            == expected_by_path[path]["face_indices_i32_sha256"],
            "live_prototype_authored_points_exact": prototype_points_sha256
            == expected_by_path[path]["points_f32_sha256"],
            "live_prototype_authored_counts_exact": prototype_counts_sha256
            == expected_by_path[path]["face_counts_i32_sha256"],
            "live_prototype_authored_indices_exact": prototype_indices_sha256
            == expected_by_path[path]["face_indices_i32_sha256"],
        }
        rows.append({**row, "checks": checks, "pass": all(checks.values())})
    rows.sort(key=lambda row: (row["body"], row["prim_name"]))
    legacy = {}
    for body in BODIES:
        prim = stage.GetPrimAtPath(LIVE_LEGACY_PATH[body])
        enabled = None
        if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        legacy[body] = {
            "path": LIVE_LEGACY_PATH[body],
            "valid": bool(prim.IsValid()),
            "collision_enabled": enabled,
            "pass": bool(prim.IsValid() and enabled is False),
        }
    actual_paths = {row["instance_path"] for row in rows}
    enabled_paths = {
        row["path"] for row in all_collision_rows if row["collision_enabled"]
    }
    disabled_paths = {
        row["path"] for row in all_collision_rows if not row["collision_enabled"]
    }
    checks = {
        "exact_path_bijection": actual_paths == set(expected_by_path),
        "all_enabled_collision_paths_exactly_p34": enabled_paths
        == set(expected_by_path),
        "all_disabled_collision_paths_exactly_known_legacy": disabled_paths
        == set(LIVE_LEGACY_PATH.values()),
        "exact_total_34": len(rows) == 34,
        "link5_16": sum(row["body"] == "link5" for row in rows) == 16,
        "gripper_link_18": sum(row["body"] == "gripper_link" for row in rows) == 18,
        "all_rows_pass": all(row["pass"] for row in rows),
        "active_a64_zero": not active_a64_paths,
        "legacy_disabled_known_only": all(row["pass"] for row in legacy.values()),
        "meters_per_unit_1": abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) <= 1.0e-12,
    }
    return {
        "rows": rows,
        "all_body_collision_rows": all_collision_rows,
        "enabled_collision_paths": sorted(enabled_paths),
        "disabled_collision_paths": sorted(disabled_paths),
        "active_a64_paths": active_a64_paths,
        "legacy": legacy,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _timeline_tuple() -> dict[str, Any]:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    fields: dict[str, Any] = {
        "is_playing": bool(timeline.is_playing()),
        "is_stopped": bool(timeline.is_stopped()),
        "current_time_s": float(timeline.get_current_time()),
    }
    for name in ("get_start_time", "get_end_time", "get_time_codes_per_seconds"):
        fn = getattr(timeline, name, None)
        if fn is not None:
            try:
                fields[name.removeprefix("get_")] = float(fn())
            except Exception as error:
                fields[name.removeprefix("get_") + "_error"] = f"{type(error).__name__}: {error}"
    return fields


def _int_to_path(value: int) -> str | None:
    from pxr import PhysicsSchemaTools

    fn = getattr(PhysicsSchemaTools, "intToSdfPath", None)
    if fn is None:
        return None
    try:
        return str(fn(int(value)))
    except Exception:
        return None


def _float3(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _float4_xyzw(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def _property_query(stage_id: int, body: str, timeout_s: float = 20.0) -> dict[str, Any]:
    from omni.physx import get_physx_property_query_interface
    from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
    from pxr import PhysicsSchemaTools

    holder: dict[str, Any] = {
        "finished": False,
        "rigid_body": None,
        "colliders": [],
        "errors": [],
    }

    def enum_value(value: Any) -> int:
        raw = getattr(value, "value", None)
        return int(raw) if raw is not None else int(value)

    def rigid(response: Any) -> None:
        row = {
            "result_name": str(getattr(response.result, "name", response.result)),
            "result_value": enum_value(response.result),
            "path_id": int(response.path_id),
            "path": _int_to_path(response.path_id),
            "mass_kg": float(response.mass),
            "center_of_mass_m": _float3(response.center_of_mass),
            "diagonal_inertia": _float3(response.inertia),
            "principal_axes_xyzw": _float4_xyzw(response.principal_axes),
        }
        holder["rigid_body"] = row
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(f"rigid-body query result {response.result}")

    def collider(response: Any) -> None:
        row = {
            "result_name": str(getattr(response.result, "name", response.result)),
            "result_value": enum_value(response.result),
            "path_id": int(response.path_id),
            "path": _int_to_path(response.path_id),
            "local_pos_m": _float3(response.local_pos),
            "local_rot_xyzw": _float4_xyzw(response.local_rot),
            "aabb_local_min_m": _float3(response.aabb_local_min),
            "aabb_local_max_m": _float3(response.aabb_local_max),
            "volume_m3": float(response.volume),
        }
        holder["colliders"].append(row)
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(f"collider query result {response.result} at {row['path']}")

    def finished() -> None:
        holder["finished"] = True

    start = time.monotonic()
    get_physx_property_query_interface().query_prim(
        stage_id=stage_id,
        prim_id=PhysicsSchemaTools.sdfPathToInt(LIVE_BODY_PATH[body]),
        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
        timeout_ms=int(timeout_s * 1000.0),
        finished_fn=finished,
        rigid_body_fn=rigid,
        collider_fn=collider,
    )
    while not holder["finished"] and time.monotonic() - start < timeout_s:
        time.sleep(0.005)
    holder["colliders"].sort(key=lambda row: str(row["path"]))
    holder["elapsed_s"] = time.monotonic() - start
    holder["simulation_app_update_pumps"] = 0
    holder["expected_collider_count_including_disabled_legacy"] = (
        EXPECTED_COUNTS[body] + 1
    )
    holder["body"] = body
    holder["body_path"] = LIVE_BODY_PATH[body]
    holder["pass"] = bool(
        holder["finished"]
        and holder["rigid_body"]
        and holder["colliders"]
        and len(holder["colliders"])
        == holder["expected_collider_count_including_disabled_legacy"]
        and not holder["errors"]
    )
    return holder


def _callback_payload(convex: Any) -> dict[str, Any]:
    vertices = [[float(vertex.x), float(vertex.y), float(vertex.z)] for vertex in list(convex.vertices)]
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


def _callback_request(
    *, stage_id: int, prim_path: str, body: str, part: dict[str, Any], channel: str, witness_path: Path
) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools

    holder: dict[str, Any] = {"events": []}
    request_active = True

    def done(result: Any, convexes: list[Any]) -> None:
        convex_list = list(convexes)
        result_value = getattr(result, "value", None)
        if result_value is None:
            result_value = int(result)
        event = {
            "callback_ordinal": len(holder["events"]) + 1,
            "callback_during_synchronous_request": bool(request_active),
            "result_name": str(getattr(result, "name", "")),
            "result_value": int(result_value),
            "result_repr": repr(result),
            "convex_count": len(convex_list),
            "convexes": [],
            "serialization_errors": [],
        }
        holder["events"].append(event)
        for index, convex in enumerate(convex_list):
            try:
                event["convexes"].append(_callback_payload(convex))
            except Exception as error:
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
            stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim_path),
            run_asynchronously=False,
            on_result=done,
        )
    except Exception as error:
        request_exception = {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        request_active = False
    witness = {
        "artifact": "D373_P34_CALLBACK_FIRST_RAW_WITNESS_V1",
        "body": body,
        "part_name": part["name"],
        "prim_name": part["prim_name"],
        "role": part["role"],
        "channel": channel,
        "instance_path": part["instance_path"],
        "prototype_path": part["prototype_path"],
        "cook_prim_path": prim_path,
        "registered_authored_f32_payload_sha256": part["authored_f32_topology_payload_sha256"],
        "events": holder["events"],
        "callback_count": len(holder["events"]),
        "request_return_type": type(request_return).__name__,
        "request_return_repr": repr(request_return),
        "request_exception": request_exception,
        "callback_payload_persisted_before_classification": True,
        "classification_performed": False,
        "cook_cache_or_setting_mutation_performed": False,
    }
    _write_json_x(witness_path, witness)
    event = holder["events"][0] if len(holder["events"]) == 1 else None
    checks = {
        "request_no_exception": request_exception is None,
        "callback_exactly_once": len(holder["events"]) == 1,
        "callback_inline": bool(event and event["callback_during_synchronous_request"]),
        "result_valid": bool(
            event
            and event["result_name"] == PhysxCollisionRepresentationResult.RESULT_VALID.name
            and int(event["result_value"]) == int(PhysxCollisionRepresentationResult.RESULT_VALID.value)
        ),
        "one_convex": bool(event and event["convex_count"] == 1),
        "serialization_no_errors": bool(event and not event["serialization_errors"]),
    }
    return {
        "channel": channel,
        "cook_prim_path": prim_path,
        "witness_path": _rel(witness_path),
        "witness_sha256": _sha(witness_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _inspection_mass_api(stage: Any) -> dict[str, Any]:
    rows = {body: _mass_row(stage, LIVE_BODY_PATH[body]) for body in BODIES}
    return {"bodies": rows}


def _execute(out_dir: Path, prereg_path: Path, app: Any) -> dict[str, Any]:
    from omni.physx import get_physx_simulation_interface

    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    counters = {
        "worker_invocations": 1,
        "automatic_retries": 0,
        "simulation_app_launches": 1,
        "derivative_asset_materializations": 0,
        "physx_stage_attaches": 0,
        "physx_stage_detaches": 0,
        "physx_property_queries": 0,
        "physx_callback_requests": 0,
        "simulation_context_constructions": 0,
        "resets": 0,
        "timeline_play_requests": 0,
        "timeline_commit_requests": 0,
        "physics_steps": 0,
        "public_forwards": 0,
        "q5_commands": 0,
        "q5_samples": 0,
        "contact_queries": 0,
        "cylinder_creates_or_writes": 0,
        "target_ik_path_pose_changes": 0,
        "automatic_convex_decomposition_sweeps": 0,
        "approved_collision_mesh_and_schema_authors": 34,
        "inherited_material_mass_actuator_physics_setting_changes": 0,
        "isaac_hydra_renders": 0,
        "simulation_app_update_pumps": 0,
    }
    result: dict[str, Any] = {
        "artifact": "D373_P34_LIVE_ASSET_WORKER_RAW_SUMMARY_V1",
        "case": "g0a_d373",
        "preregistration_path": _rel(prereg_path),
        "preregistration_sha256": _sha(prereg_path),
        "counters": counters,
        "timeline_before": None,
        "timeline_after": None,
        "exception": None,
    }
    stage = None
    stage_id = None
    attached = False
    _phase(out_dir, "worker_execution_start")
    try:
        if _git("rev-parse", "HEAD") != EXPECTED_HEAD:
            raise RuntimeError("HEAD changed after D373 preregistration")
        expected_hash = prereg["inputs"][_rel(GEOMETRY_PATH)]
        if _sha(GEOMETRY_PATH) != expected_hash:
            raise RuntimeError("frozen D372 geometry hash changed after preregistration")
        result["timeline_before"] = _timeline_tuple()
        if result["timeline_before"]["is_playing"] or not result["timeline_before"]["is_stopped"]:
            raise RuntimeError(f"timeline is not stopped at D373 boundary: {result['timeline_before']}")

        _phase(out_dir, "derivative_asset_materialize_start")
        variant_dir, asset_record = _materialize_asset(out_dir, geometry)
        counters["derivative_asset_materializations"] = 1
        _phase(out_dir, "derivative_asset_materialize_end", variant_root=_rel(variant_dir / "roarm_m3.usd"))
        expected = asset_record["expected"]
        result["asset"] = asset_record
        result["nonphysics_copy_audit"] = _nonphysics_copy_audit(asset_record)
        result["authored_readback"] = _readback_physics_layer(variant_dir / PHYSICS_REL, expected)
        result["mass_api_base_vs_derivative"] = _mass_api_audit(BASE_ROOT_USD, variant_dir / "roarm_m3.usd")
        result["canonical_outside_collision_subtree_diff"] = _canonical_outside_subtree_diff(
            BASE_ROOT_USD, variant_dir / "roarm_m3.usd"
        )

        _phase(out_dir, "inspection_stage_create_start")
        stage, stage_id = _make_inspection_stage(variant_dir / "roarm_m3.usd")
        result["inspection_stage"] = {
            "identifier": str(stage.GetRootLayer().identifier),
            "stage_id": stage_id,
            "root_reference": _rel(variant_dir / "roarm_m3.usd"),
            "meters_per_unit": 1.0,
        }
        result["live_inventory"] = _live_inventory(stage, expected)
        result["mass_api_inspection_stage"] = _inspection_mass_api(stage)
        _phase(out_dir, "inspection_stage_inventory_end", live_part_count=len(result["live_inventory"]["rows"]))

        simulation = get_physx_simulation_interface()
        _phase(out_dir, "physx_stage_attach_start")
        attached = bool(simulation.attach_stage(stage_id))
        counters["physx_stage_attaches"] = 1
        result["physx_stage_attach_return"] = attached
        _phase(out_dir, "physx_stage_attach_end", attached=attached)
        if not attached:
            raise RuntimeError("PhysX attach_stage returned false; callback/property audit stops")

        witness_dir = out_dir / WITNESS_DIR_NAME
        witness_dir.mkdir(parents=True, exist_ok=False)
        live_by_key = {
            (row["body"], row["prim_name"]): row for row in result["live_inventory"]["rows"]
        }
        callback_rows = []
        progress = 0
        for body in BODIES:
            for exp in expected[body]:
                live = live_by_key[(body, exp["prim_name"])]
                part = {**exp, **live}
                channels = {}
                for channel, prim_path in (
                    ("prototype", live["prototype_path"]),
                    ("instance", live["instance_path"]),
                ):
                    witness_path = witness_dir / f"{body}_{exp['prim_name']}_{channel}.json"
                    channels[channel] = _callback_request(
                        stage_id=stage_id,
                        prim_path=prim_path,
                        body=body,
                        part=part,
                        channel=channel,
                        witness_path=witness_path,
                    )
                    counters["physx_callback_requests"] += 1
                    _phase(
                        out_dir,
                        "callback_progress",
                        part_progress=progress,
                        part_total=34,
                        body=body,
                        prim_name=exp["prim_name"],
                        channel=channel,
                        request_ordinal=counters["physx_callback_requests"],
                        passed=channels[channel]["pass"],
                    )
                callback_rows.append(
                    {
                        "body": body,
                        "name": exp["name"],
                        "role": exp["role"],
                        "prim_name": exp["prim_name"],
                        "instance_path": live["instance_path"],
                        "prototype_path": live["prototype_path"],
                        "authored_f32_topology_payload_sha256": exp[
                            "authored_f32_topology_payload_sha256"
                        ],
                        "channels": channels,
                        "protocol_pass": all(row["pass"] for row in channels.values()),
                    }
                )
                progress += 1
        result["callback_rows"] = callback_rows

        # Synchronous prototype/instance callback cooking above makes all 34
        # meshes cache-ready before property query.  This preserves the
        # registered zero Kit-update-pump boundary for the fresh derivative.
        properties = {}
        for body in BODIES:
            _phase(out_dir, "property_query_start", body=body)
            properties[body] = _property_query(stage_id, body)
            counters["physx_property_queries"] += 1
            _phase(
                out_dir,
                "property_query_end",
                body=body,
                collider_count=len(properties[body]["colliders"]),
                passed=properties[body]["pass"],
            )
        result["property_queries"] = properties
        result["timeline_after"] = _timeline_tuple()
    except Exception as error:
        result["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json_x(
                out_dir / EXCEPTION_NAME,
                {
                    "artifact": "D373_WORKER_EXCEPTION_V1",
                    **result["exception"],
                    "counters_at_exception": counters,
                },
            )
        except FileExistsError:
            pass
    finally:
        _phase(out_dir, "worker_cleanup_start", physx_stage_attached=attached)
        if attached:
            try:
                get_physx_simulation_interface().detach_stage()
                counters["physx_stage_detaches"] = 1
            except Exception as error:
                result["detach_exception"] = f"{type(error).__name__}: {error}"
        if result.get("timeline_after") is None:
            try:
                result["timeline_after"] = _timeline_tuple()
            except Exception as error:
                result["timeline_after_error"] = f"{type(error).__name__}: {error}"
        result["phase_scope"] = {
            "physics_authority": "asset load/cook/readback only",
            "no_simulation_context_or_step": True,
            "callback_order": "prototype_then_instance_per_part",
            "callback_raw_polygon_authority": True,
        }
        result["worker_protocol_pass"] = bool(
            result["exception"] is None
            and result.get("physx_stage_attach_return") is True
            and result.get("nonphysics_copy_audit", {}).get("pass") is True
            and result.get("authored_readback", {}).get("pass") is True
            and result.get("mass_api_base_vs_derivative", {}).get("pass") is True
            and all(
                result.get("canonical_outside_collision_subtree_diff", {})
                .get("checks", {})
                .values()
            )
            and result.get("live_inventory", {}).get("pass") is True
            and all(row.get("pass") for row in result.get("property_queries", {}).values())
            and len(result.get("callback_rows", [])) == 34
            and all(row.get("protocol_pass") for row in result.get("callback_rows", []))
            and counters["physx_callback_requests"] == 68
            and counters["worker_invocations"] == 1
            and counters["simulation_app_launches"] == 1
            and counters["derivative_asset_materializations"] == 1
            and counters["physx_stage_attaches"] == 1
            and counters["physx_stage_detaches"] == 1
            and counters["physx_property_queries"] == 2
            and all(
                counters[key] == 0
                for key in (
                    "automatic_retries",
                    "simulation_context_constructions",
                    "resets",
                    "timeline_play_requests",
                    "timeline_commit_requests",
                    "physics_steps",
                    "public_forwards",
                    "q5_commands",
                    "q5_samples",
                    "contact_queries",
                    "cylinder_creates_or_writes",
                    "target_ik_path_pose_changes",
                    "automatic_convex_decomposition_sweeps",
                    "inherited_material_mass_actuator_physics_setting_changes",
                    "isaac_hydra_renders",
                    "simulation_app_update_pumps",
                )
            )
            and result.get("timeline_before", {}).get("is_stopped") is True
            and result.get("timeline_after", {}).get("is_stopped") is True
            and result.get("timeline_before") == result.get("timeline_after")
        )
        _write_json_x(out_dir / SUMMARY_NAME, result)
        _write_json_x(
            out_dir / PRECLOSE_NAME,
            {
                "artifact": "D373_WORKER_PRECLOSE_SENTINEL_V1",
                "summary_path": _rel(out_dir / SUMMARY_NAME),
                "summary_sha256": _sha(out_dir / SUMMARY_NAME),
                "counters": counters,
                "timeline_after": result.get("timeline_after"),
                "worker_protocol_pass": result["worker_protocol_pass"],
                "safe_to_close_app": True,
            },
        )
        _phase(out_dir, "worker_cleanup_end", worker_protocol_pass=result["worker_protocol_pass"])
    return result


def _exclusive_out_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise RuntimeError("D373 --out-dir must remain inside the repository") from error
    if not resolved.is_dir():
        raise RuntimeError("D373 controller must create the forward-only output directory first")
    owned = [resolved / CLAIM_NAME, resolved / SUMMARY_NAME, resolved / PRECLOSE_NAME]
    existing = [_rel(path) for path in owned if path.exists()]
    if existing:
        raise RuntimeError(f"D373 worker-owned outputs already exist: {existing}")
    return resolved


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    out_dir = _exclusive_out_dir(args.out_dir)
    prereg = args.prereg.resolve()
    if not prereg.is_file():
        raise RuntimeError(f"missing D373 preregistration: {prereg}")
    _write_json_x(
        out_dir / CLAIM_NAME,
        {
            "artifact": "D373_SINGLE_WORKER_EXCLUSIVE_CLAIM_V1",
            "pid": os.getpid(),
            "monotonic_ns": time.monotonic_ns(),
            "worker_invocation_count": 1,
            "automatic_retry_count": 0,
        },
    )
    launcher = None
    _phase(out_dir, "simulation_app_launch_start")
    try:
        launcher = AppLauncher(args)
        _phase(out_dir, "simulation_app_launch_end", headless=True)
        result = _execute(out_dir, prereg, launcher.app)
        print(
            json.dumps(
                {
                    "artifact": "D373_WORKER_EXIT",
                    "summary": _rel(out_dir / SUMMARY_NAME),
                    "pass": result["worker_protocol_pass"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["worker_protocol_pass"] else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
