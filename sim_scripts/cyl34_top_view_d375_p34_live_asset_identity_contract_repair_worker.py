#!/usr/bin/env python3
"""One-shot Isaac/PhysX worker for D375 P34 live-identity repair.

The frozen D373 P34 asset is referenced without making the whole robot
instanceable.  The worker records direct/live Float32 identity, one callback
for each of the 34 live collider paths, and two property queries.  It never
constructs SimulationContext, plays the timeline, advances physics, evaluates
q5, creates a cylinder, or queries contacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
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

D373_WORKER_PATH = REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_worker.py"
D373_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d373/"
    "attempt1_p34_live_asset_identity_preflight"
)
D373_RAW_PATH = D373_DIR / "d373_worker_raw_summary.json"
D373_ASSET_DIR = D373_DIR / "collision_asset/roarm_m3_p34_semantic_compound"
D373_ROOT_USD = D373_ASSET_DIR / "roarm_m3.usd"
D373_PHYSICS_USD = D373_ASSET_DIR / "configuration/roarm_m3_physics.usd"
D372_GEOMETRY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair/"
    "d372_professor_semantic_candidate_geometry.json"
)

BODIES = ("link5", "gripper_link")
EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18}
EXPECTED_HEAD = "3d71aac219ba16f3262dc94b1898a459eaa534e7"
EXPECTED_MIN_THICKNESS_BITS = 0x38D1B717
EXPECTED_MIN_THICKNESS_BYTES = "17b7d138"
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
DIRECT_PARENT_PATH = {
    body: f"/colliders/{body}/d373_p34_parts" for body in BODIES
}

CLAIM_NAME = "d375_worker_claim.json"
SUMMARY_NAME = "d375_worker_raw_summary.json"
PRECLOSE_NAME = "d375_worker_preclose_sentinel.json"
EXCEPTION_NAME = "d375_worker_exception.json"
WITNESS_DIR_NAME = "callback_witnesses"
PHASE_NAME = "d375_phase_markers.jsonl"


def _load_frozen_d373() -> Any:
    spec = importlib.util.spec_from_file_location("d373_worker_frozen_for_d375", D373_WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D373 worker module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D373 = _load_frozen_d373()


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
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(out_dir: Path, name: str, **fields: Any) -> None:
    path = out_dir / PHASE_NAME
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


def _typed_float_record(value: Any, type_name: str) -> dict[str, Any]:
    value_f32 = np.float32(float(value))
    packed = struct.pack("<f", float(value_f32))
    bits = struct.unpack("<I", packed)[0]
    checks = {
        "usd_type_name_float": type_name == "float",
        "observed_bits_exact_d343": bits == EXPECTED_MIN_THICKNESS_BITS,
        "observed_le_bytes_exact_d343": packed.hex() == EXPECTED_MIN_THICKNESS_BYTES,
    }
    return {
        "value_m": float(value_f32),
        "type_name": type_name,
        "bits_uint32": bits,
        "bits_hex": f"0x{bits:08x}",
        "little_endian_bytes_hex": packed.hex(),
        "inherited_d343_expected_bits_hex": f"0x{EXPECTED_MIN_THICKNESS_BITS:08x}",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _expected_records(geometry: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    expected: dict[str, list[dict[str, Any]]] = {body: [] for body in BODIES}
    for body in BODIES:
        parts = list(geometry["parts"][body])
        if len(parts) != EXPECTED_COUNTS[body]:
            raise RuntimeError(f"frozen D372 count changed for {body}: {len(parts)}")
        for index, part in enumerate(parts):
            record = D373._expected_f32_record(index, part)
            if not record["d372_payload_exact"]:
                raise RuntimeError(f"D372 payload digest mismatch for {body}/{part['name']}")
            expected[body].append(
                {
                    **{key: value for key, value in record.items() if not isinstance(value, np.ndarray)},
                    "direct_path": f"{DIRECT_PARENT_PATH[body]}/{record['prim_name']}",
                    "live_path": f"{LIVE_PARENT_PATH[body]}/{record['prim_name']}",
                }
            )
    return expected


def _corrected_direct_readback(expected: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from pxr import Usd

    inherited = D373._readback_physics_layer(D373_PHYSICS_USD, expected)
    stage = Usd.Stage.Open(str(D373_PHYSICS_USD), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError("failed to open frozen D373 physics layer")
    expected_by_path = {
        row["direct_path"]: row for body in BODIES for row in expected[body]
    }
    for row in inherited["rows"]:
        prim = stage.GetPrimAtPath(row["direct_path"])
        attr = prim.GetAttribute("physxConvexHullCollision:minThickness")
        typed = _typed_float_record(attr.Get(), str(attr.GetTypeName()))
        row["typed_min_thickness"] = typed
        row["checks"].pop("min_thickness_frozen", None)
        row["checks"]["min_thickness_inherits_d343_typed_bits"] = typed["pass"]
        row["checks"]["registered_direct_path_exact"] = row["direct_path"] in expected_by_path
        row["pass"] = all(row["checks"].values())
    inherited["counts"] = {
        body: sum(row["body"] == body for row in inherited["rows"]) for body in BODIES
    }
    inherited["typed_float32_contract_retests"] = 0
    inherited["d343_contract_inherited"] = True
    inherited["pass"] = (
        len(inherited["rows"]) == 34
        and inherited["counts"] == EXPECTED_COUNTS
        and all(row["pass"] for row in inherited["rows"])
    )
    return inherited


def _make_noninstance_stage(variant_root: Path) -> tuple[Any, int, dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.CreateInMemory("d375_p34_noninstance_identity_inspection.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference(str(variant_root.resolve()))
    # D375's only stage-structure repair: deliberately do not call
    # SetInstanceable(True) on /World/Robot.
    owners = {}
    for path in ("/World/Robot", LIVE_BODY_PATH["link5"], LIVE_BODY_PATH["gripper_link"]):
        prim = stage.GetPrimAtPath(path)
        is_body = path in LIVE_BODY_PATH.values()
        checks = {
            "prim_valid": bool(prim.IsValid()),
            "not_instanceable": bool(prim.IsValid() and not prim.IsInstanceable()),
            "not_instance": bool(prim.IsValid() and not prim.IsInstance()),
            "not_instance_proxy": bool(prim.IsValid() and not prim.IsInstanceProxy()),
            "rigid_body_api_valid_when_owner": bool(
                not is_body or (prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI))
            ),
        }
        owners[path] = {
            "path": path,
            "is_articulation_rigid_body_owner": is_body,
            "checks": checks,
            "pass": all(checks.values()),
        }
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
    if not stage_id.IsValid():
        raise RuntimeError("failed to insert D375 stage into StageCache")
    audit = {
        "root_reference": _rel(variant_root),
        "whole_robot_set_instanceable_true_calls": 0,
        "owners": owners,
        "pass": all(row["pass"] for row in owners.values()),
    }
    return stage, int(stage_id.ToLongInt()), audit


def _nearest_rigid_owner(prim: Any) -> Any:
    from pxr import UsdPhysics

    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _live_inventory(
    stage: Any,
    expected: dict[str, list[dict[str, Any]]],
    owner_structure: dict[str, Any],
) -> dict[str, Any]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    expected_by_path = {
        row["live_path"]: row for body in BODIES for row in expected[body]
    }
    default_paths = sorted(
        prim.GetPath().pathString
        for prim in Usd.PrimRange.Stage(stage)
        if prim.GetPath().pathString in expected_by_path
    )
    rows = []
    collision_rows = []
    active_a64_paths = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        body_scope = next(
            (body for body in BODIES if path.startswith(LIVE_BODY_PATH[body] + "/")),
            None,
        )
        if body_scope is not None and prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            collision_rows.append(
                {
                    "body": body_scope,
                    "path": path,
                    "collision_enabled": True if enabled is None else bool(enabled),
                }
            )
        if any(path.startswith(LIVE_OLD_PARENT_PATH[body] + "/") for body in BODIES):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and bool(
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            ):
                active_a64_paths.append(path)
        if path not in expected_by_path:
            continue
        exp = expected_by_path[path]
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(
            [[float(value) for value in point] for point in list(mesh.GetPointsAttr().Get() or [])],
            dtype="<f4",
        )
        counts = np.asarray(list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype="<i4")
        indices = np.asarray(list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype="<i4")
        collision = UsdPhysics.CollisionAPI(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        hull_api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        min_attr = prim.GetAttribute("physxConvexHullCollision:minThickness")
        typed = _typed_float_record(min_attr.Get(), str(min_attr.GetTypeName()))
        owner = _nearest_rigid_owner(prim)
        owner_path = owner.GetPath().pathString if owner is not None and owner.IsValid() else None
        prototype = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else None
        row = {
            "body": exp["live_path"].split("/")[3],
            "name": exp["name"],
            "role": exp["role"],
            "prim_name": exp["prim_name"],
            "live_path": path,
            "owner_path": owner_path,
            "owner_is_instance": bool(owner and owner.IsValid() and owner.IsInstance()),
            "owner_is_instance_proxy": bool(owner and owner.IsValid() and owner.IsInstanceProxy()),
            "owner_is_instanceable": bool(owner and owner.IsValid() and owner.IsInstanceable()),
            "leaf_is_instance": bool(prim.IsInstance()),
            "leaf_is_instance_proxy": bool(prim.IsInstanceProxy()),
            "prototype_path_diagnostic": (
                prototype.GetPath().pathString if prototype is not None and prototype.IsValid() else None
            ),
            "is_mesh": bool(prim.IsA(UsdGeom.Mesh)),
            "collision_enabled": bool(collision.GetCollisionEnabledAttr().Get()),
            "approximation": str(mesh_api.GetApproximationAttr().Get()),
            "hull_vertex_limit": int(hull_api.GetHullVertexLimitAttr().Get()),
            "typed_min_thickness": typed,
            "points_f32_sha256": _sha_bytes(_array_bytes(points, "<f4")),
            "face_counts_i32_sha256": _sha_bytes(_array_bytes(counts, "<i4")),
            "face_indices_i32_sha256": _sha_bytes(_array_bytes(indices, "<i4")),
        }
        owner_gate = owner_structure["owners"][LIVE_BODY_PATH[row["body"]]]["pass"]
        checks = {
            "owner_path_exact": owner_path == LIVE_BODY_PATH[row["body"]],
            "owner_noninstance_gate": owner_gate
            and not row["owner_is_instance"]
            and not row["owner_is_instance_proxy"]
            and not row["owner_is_instanceable"],
            "mesh": row["is_mesh"],
            "collision_enabled": row["collision_enabled"],
            "convex_hull_binding": row["approximation"] == "convexHull",
            "hull_vertex_limit_64": row["hull_vertex_limit"] == 64,
            "min_thickness_inherits_d343_typed_bits": typed["pass"],
            "live_points_exact": row["points_f32_sha256"] == exp["points_f32_sha256"],
            "live_counts_exact": row["face_counts_i32_sha256"] == exp["face_counts_i32_sha256"],
            "live_indices_exact": row["face_indices_i32_sha256"] == exp["face_indices_i32_sha256"],
        }
        rows.append({**row, "checks": checks, "pass": all(checks.values())})
    rows.sort(key=lambda row: (row["body"], row["prim_name"]))
    actual_paths = {row["live_path"] for row in rows}
    enabled_paths = {row["path"] for row in collision_rows if row["collision_enabled"]}
    disabled_paths = {row["path"] for row in collision_rows if not row["collision_enabled"]}
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
    checks = {
        "owner_structure_pass": owner_structure["pass"],
        "proxy_aware_exact_path_bijection": actual_paths == set(expected_by_path),
        "proxy_aware_exact_total_34": len(rows) == 34,
        "link5_16": sum(row["body"] == "link5" for row in rows) == 16,
        "gripper_link_18": sum(row["body"] == "gripper_link" for row in rows) == 18,
        "all_rows_pass": all(row["pass"] for row in rows),
        "all_enabled_collision_paths_exactly_p34": enabled_paths == set(expected_by_path),
        "all_disabled_paths_exactly_known_legacy": disabled_paths == set(LIVE_LEGACY_PATH.values()),
        "known_legacy_disabled": all(row["pass"] for row in legacy.values()),
        "active_a64_zero": not active_a64_paths,
        "meters_per_unit_1": abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) <= 1.0e-12,
    }
    return {
        "rows": rows,
        "default_traversal_paths_diagnostic_only": default_paths,
        "default_traversal_count_diagnostic_only": len(default_paths),
        "proxy_aware_count": len(rows),
        "leaf_instance_proxy_count_diagnostic_only": sum(row["leaf_is_instance_proxy"] for row in rows),
        "legacy": legacy,
        "active_a64_paths": sorted(active_a64_paths),
        "all_collision_rows": sorted(collision_rows, key=lambda row: row["path"]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _callback_payload(convex: Any) -> dict[str, Any]:
    return D373._callback_payload(convex)


def _callback_request(
    *, stage_id: int, prim_path: str, body: str, part: dict[str, Any], witness_path: Path
) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools

    holder: dict[str, Any] = {"events": []}
    request_active = True

    def done(result: Any, convexes: list[Any]) -> None:
        convex_list = list(convexes)
        raw_value = getattr(result, "value", None)
        event = {
            "callback_ordinal": len(holder["events"]) + 1,
            "callback_during_synchronous_request": bool(request_active),
            "result_name": str(getattr(result, "name", "")),
            "result_value": int(raw_value) if raw_value is not None else int(result),
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
        "artifact": "D375_P34_DIRECT_LIVE_CALLBACK_FIRST_RAW_WITNESS_V1",
        "body": body,
        "part_name": part["name"],
        "prim_name": part["prim_name"],
        "role": part["role"],
        "channel": "live_direct_path",
        "live_path": prim_path,
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
        "channel": "live_direct_path",
        "cook_prim_path": prim_path,
        "witness_path": _rel(witness_path),
        "witness_sha256": _sha(witness_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _execute(out_dir: Path, prereg_path: Path) -> dict[str, Any]:
    from omni.physx import get_physx_simulation_interface

    geometry = json.loads(D372_GEOMETRY.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    counters = {
        "worker_invocations": 1,
        "automatic_retries": 0,
        "simulation_app_launches": 1,
        "derivative_asset_materializations": 0,
        "usd_stage_file_writes": 0,
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
        "material_mass_actuator_physics_setting_changes": 0,
        "isaac_hydra_renders": 0,
        "simulation_app_update_pumps": 0,
        "d343_contract_retests": 0,
    }
    result: dict[str, Any] = {
        "artifact": "D375_P34_LIVE_ASSET_IDENTITY_REPAIR_WORKER_RAW_V1",
        "case": "g0a_d375",
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
            raise RuntimeError("HEAD changed after D375 preregistration")
        if _sha(D372_GEOMETRY) != prereg["inputs"][_rel(D372_GEOMETRY)]:
            raise RuntimeError("frozen D372 geometry changed after preregistration")
        if _sha(D373_RAW_PATH) != prereg["inputs"][_rel(D373_RAW_PATH)]:
            raise RuntimeError("frozen D373 raw summary changed after preregistration")
        expected_asset_hashes = prereg["d373_asset_file_hashes"]
        asset_hashes_before = D373._asset_hashes(D373_ASSET_DIR)
        if asset_hashes_before != expected_asset_hashes:
            raise RuntimeError("frozen D373 P34 asset hash inventory changed")
        result["asset_reuse"] = {
            "variant_dir": _rel(D373_ASSET_DIR),
            "root_usd": _rel(D373_ROOT_USD),
            "physics_usd": _rel(D373_PHYSICS_USD),
            "file_hashes_before": asset_hashes_before,
            "materialization_count": 0,
            "usd_stage_file_write_count": 0,
        }
        result["timeline_before"] = D373._timeline_tuple()
        if result["timeline_before"]["is_playing"] or not result["timeline_before"]["is_stopped"]:
            raise RuntimeError(f"timeline not stopped at D375 boundary: {result['timeline_before']}")

        expected = _expected_records(geometry)
        result["authored_readback"] = _corrected_direct_readback(expected)
        result["mass_api_base_vs_derivative"] = D373._mass_api_audit(
            D373.BASE_ROOT_USD, D373_ROOT_USD
        )
        result["canonical_outside_collision_subtree_diff"] = D373._canonical_outside_subtree_diff(
            D373.BASE_ROOT_USD, D373_ROOT_USD
        )

        _phase(out_dir, "noninstance_inspection_stage_create_start")
        stage, stage_id, owner_structure = _make_noninstance_stage(D373_ROOT_USD)
        result["inspection_stage"] = {
            "identifier": str(stage.GetRootLayer().identifier),
            "stage_id": stage_id,
            "root_reference": _rel(D373_ROOT_USD),
            "meters_per_unit": 1.0,
            "persisted_usd_write": False,
        }
        result["owner_structure"] = owner_structure
        result["live_inventory"] = _live_inventory(stage, expected, owner_structure)
        result["mass_api_inspection_stage"] = D373._inspection_mass_api(stage)
        _phase(
            out_dir,
            "noninstance_owner_and_inventory_gate_end",
            owners_pass=owner_structure["pass"],
            live_parts=len(result["live_inventory"]["rows"]),
            inventory_pass=result["live_inventory"]["pass"],
        )
        if not owner_structure["pass"] or not result["live_inventory"]["pass"]:
            raise RuntimeError("D375 non-instance owner/live inventory pre-attach gate failed")

        simulation = get_physx_simulation_interface()
        _phase(out_dir, "physx_stage_attach_start")
        attached = bool(simulation.attach_stage(stage_id))
        counters["physx_stage_attaches"] = 1
        result["physx_stage_attach_return"] = attached
        _phase(out_dir, "physx_stage_attach_end", attached=attached)
        if not attached:
            raise RuntimeError("PhysX attach_stage returned false")

        witness_dir = out_dir / WITNESS_DIR_NAME
        witness_dir.mkdir(parents=True, exist_ok=False)
        live_by_key = {
            (row["body"], row["prim_name"]): row for row in result["live_inventory"]["rows"]
        }
        callback_rows = []
        part_ordinal = 0
        for body in BODIES:
            for exp in expected[body]:
                live = live_by_key[(body, exp["prim_name"])]
                part = {**exp, **live}
                witness_path = witness_dir / f"{body}_{exp['prim_name']}_live.json"
                callback = _callback_request(
                    stage_id=stage_id,
                    prim_path=live["live_path"],
                    body=body,
                    part=part,
                    witness_path=witness_path,
                )
                counters["physx_callback_requests"] += 1
                callback_rows.append(
                    {
                        "body": body,
                        "name": exp["name"],
                        "role": exp["role"],
                        "prim_name": exp["prim_name"],
                        "live_path": live["live_path"],
                        "authored_f32_topology_payload_sha256": exp[
                            "authored_f32_topology_payload_sha256"
                        ],
                        "callback": callback,
                        "protocol_pass": callback["pass"],
                    }
                )
                part_ordinal += 1
                _phase(
                    out_dir,
                    "callback_progress",
                    part_progress=part_ordinal,
                    part_total=34,
                    request_ordinal=counters["physx_callback_requests"],
                    body=body,
                    prim_name=exp["prim_name"],
                    passed=callback["pass"],
                )
        result["callback_rows"] = callback_rows

        properties = {}
        for body in BODIES:
            _phase(out_dir, "property_query_start", body=body)
            properties[body] = D373._property_query(stage_id, body)
            counters["physx_property_queries"] += 1
            _phase(
                out_dir,
                "property_query_end",
                body=body,
                collider_count=len(properties[body]["colliders"]),
                passed=properties[body]["pass"],
            )
        result["property_queries"] = properties
        result["timeline_after"] = D373._timeline_tuple()
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
                    "artifact": "D375_WORKER_EXCEPTION_V1",
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
                result["timeline_after"] = D373._timeline_tuple()
            except Exception as error:
                result["timeline_after_error"] = f"{type(error).__name__}: {error}"
        try:
            hashes_after = D373._asset_hashes(D373_ASSET_DIR)
            result.setdefault("asset_reuse", {})["file_hashes_after"] = hashes_after
            result["asset_reuse"]["immutable_before_after"] = (
                result["asset_reuse"].get("file_hashes_before") == hashes_after
            )
        except Exception as error:
            result["asset_reuse_after_error"] = f"{type(error).__name__}: {error}"
        canonical_checks = result.get("canonical_outside_collision_subtree_diff", {}).get(
            "checks", {}
        )
        canonical_required = all(
            canonical_checks.get(key) is True
            for key in (
                "outside_registered_subtrees_bit_exact",
                "no_runtime_address_patterns",
                "no_unsupported_types",
                "variant_contains_no_a64",
            )
        )
        zero_keys = (
            "automatic_retries",
            "derivative_asset_materializations",
            "usd_stage_file_writes",
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
            "material_mass_actuator_physics_setting_changes",
            "isaac_hydra_renders",
            "simulation_app_update_pumps",
            "d343_contract_retests",
        )
        result["phase_scope"] = {
            "authority": "asset load/cook/readback only",
            "whole_robot_instanceability_removed": True,
            "callback_channel": "one request per actual live collider path",
            "callback_raw_polygon_authority": True,
            "physics_or_q5_science": False,
        }
        result["worker_protocol_pass"] = bool(
            result["exception"] is None
            and result.get("physx_stage_attach_return") is True
            and result.get("authored_readback", {}).get("pass") is True
            and result.get("mass_api_base_vs_derivative", {}).get("pass") is True
            and canonical_required
            and result.get("owner_structure", {}).get("pass") is True
            and result.get("live_inventory", {}).get("pass") is True
            and all(row.get("pass") for row in result.get("property_queries", {}).values())
            and len(result.get("callback_rows", [])) == 34
            and all(row.get("protocol_pass") for row in result.get("callback_rows", []))
            and counters["worker_invocations"] == 1
            and counters["simulation_app_launches"] == 1
            and counters["physx_stage_attaches"] == 1
            and counters["physx_stage_detaches"] == 1
            and counters["physx_property_queries"] == 2
            and counters["physx_callback_requests"] == 34
            and all(counters[key] == 0 for key in zero_keys)
            and result.get("asset_reuse", {}).get("immutable_before_after") is True
            and result.get("timeline_before", {}).get("is_stopped") is True
            and result.get("timeline_after", {}).get("is_stopped") is True
            and result.get("timeline_before") == result.get("timeline_after")
        )
        _write_json_x(out_dir / SUMMARY_NAME, result)
        _write_json_x(
            out_dir / PRECLOSE_NAME,
            {
                "artifact": "D375_WORKER_PRECLOSE_SENTINEL_V1",
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
        raise RuntimeError("D375 output directory must remain inside repository") from error
    if not resolved.is_dir():
        raise RuntimeError("D375 controller must create output directory before worker")
    owned = [resolved / CLAIM_NAME, resolved / SUMMARY_NAME, resolved / PRECLOSE_NAME]
    existing = [_rel(path) for path in owned if path.exists()]
    if existing:
        raise RuntimeError(f"D375 worker-owned outputs already exist: {existing}")
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
        raise RuntimeError(f"missing D375 preregistration: {prereg}")
    _write_json_x(
        out_dir / CLAIM_NAME,
        {
            "artifact": "D375_SINGLE_WORKER_EXCLUSIVE_CLAIM_V1",
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
        result = _execute(out_dir, prereg)
        print(
            json.dumps(
                {
                    "artifact": "D375_WORKER_EXIT",
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
