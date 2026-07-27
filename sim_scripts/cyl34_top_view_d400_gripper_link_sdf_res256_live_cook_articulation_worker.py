#!/usr/bin/env python3
"""One-shot D400 Isaac/PhysX worker.

This file is intentionally inert when merely parsed or imported.  Its runtime
entry point is guarded by ``if __name__ == "__main__"`` and is meant to be
invoked only by the hash-gated D400 controller after a separate user approval.

At runtime it copies the frozen D344 asset once, keeps link5 A64 unchanged,
disables the 64 gripper_link A64 colliders, and authors one SDF-res256
collision representation on the existing full gripper Mesh.  It then performs
zero-step load/cook/readback and owner/property-query checks.  It never creates
SimulationContext, plays or commits the timeline, advances physics, evaluates
q5, queries contacts, creates a cylinder, or renders through Hydra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WORKER_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_preflight.py"
)
PREREG_PATH = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d400/"
    "attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/"
    "d400_preregistration.json"
)
OUT_DIR = PREREG_PATH.parent
ATTESTATION_PATH = OUT_DIR / "d400_reviewed_script_attestation.json"
TUPLE_PATH = OUT_DIR / "d400_proposed_runtime_hash_tuple.json"
RUNTIME_MANIFEST_PATH = OUT_DIR / "d400_runtime_freeze_manifest.json"
INVOCATION_PATH = OUT_DIR / "d400_worker_invocation.json"
ISAAC_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
TUPLE_FIELDS = (
    "preregistration_sha256",
    "reviewed_script_attestation_sha256",
    "controller_script_sha256",
    "worker_script_sha256",
)
EXPECTED_PREREG_SHA256 = (
    "fc689cb1afd6108a326a73f22b8117dfdefc0bb4d8caee5bcb7470c362e96c93"
)
BASE_ASSET_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/"
    "roarm_m3_fullmesh_fixed_point_parts"
)
BASE_ROOT_USD = BASE_ASSET_DIR / "roarm_m3.usd"
ASSET_NAME = "roarm_m3_link5_a64_gripper_sdf_res256"
PHYSX_NATIVE_PLUGIN = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/"
    "omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
    "bin/libomni.physx.plugin.so"
)
EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256 = (
    "03fbf17e6f0dc3f9006c8c00aa0ca572a72fd69498874df6dd900dac726c9909"
)

CLAIM_NAME = "d400_worker_claim.json"
RAW_SUMMARY_NAME = "d400_worker_raw_summary.json"
PRECLOSE_NAME = "d400_worker_preclose_sentinel.json"
PHASE_NAME = "d400_phase_markers.jsonl"
OWNER_EVIDENCE_NAME = "d400_live_configuration_owner_evidence.json"

SOURCE_MESH_PATH = (
    "/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh"
)
LIVE_MESH_PATH = (
    "/World/Robot/gripper_link/collisions/gripper_link/node_STL_BINARY_/mesh"
)
SOURCE_BODY_PATHS = {
    "link5": "/roarm_m3/link5",
    "gripper_link": "/roarm_m3/gripper_link",
}
LIVE_BODY_PATHS = {
    "link5": "/World/Robot/link5",
    "gripper_link": "/World/Robot/gripper_link",
}
SOURCE_COLLISION_SCOPES = (
    "/roarm_m3/link5/collisions",
    "/roarm_m3/gripper_link/collisions",
)
LIVE_REQUIRED_NONINSTANCE_PATHS = (
    "/World/Robot",
    "/World/Robot/link5",
    "/World/Robot/gripper_link",
    LIVE_MESH_PATH,
)
SOURCE_GRIPPER_A64_PATHS = tuple(
    f"/roarm_m3/gripper_link/collisions/d338_convex_parts/part_{index:03d}"
    for index in range(64)
)
LIVE_LINK5_A64_PATHS = tuple(
    f"/World/Robot/link5/collisions/d338_convex_parts/part_{index:03d}"
    for index in range(64)
)
LIVE_GRIPPER_A64_PATHS = tuple(
    f"/World/Robot/gripper_link/collisions/d338_convex_parts/part_{index:03d}"
    for index in range(64)
)
LIVE_LEGACY_PATHS = {
    "link5": "/World/Robot/link5/collisions/link5/node_STL_BINARY_",
    "gripper_link": (
        "/World/Robot/gripper_link/collisions/gripper_link/node_STL_BINARY_"
    ),
}
EXPECTED_QUERY_PATH_SHA256 = {
    "link5": "54dd23cb24d9c85d505fd9c44708248e8715b904f4ea4d275d7992ab21ef7a5a",
    "gripper_link": (
        "7b23094e24a7f574e3d0cdc7057f6510817ba6cd47ccc3ba319b27dce4fe2821"
    ),
}
EXPECTED_QUERY_COUNTS = {"link5": 65, "gripper_link": 66}

SOURCE_STREAM_HASHES = {
    "points_f32": "b89c67e99bd253ae710e6b0a2fcacd0b27263d6ede29fe6f6334ed70247895ed",
    "counts_i32": "efbe0858156ea6b81fd25df456a53df63c700d0f8dd5ddc5b6c453f04d50185a",
    "indices_i32": "c42fe919568d56e3ae85cebbef5b42cf04ca973c78db538359f587deda401d84",
    "combined": "31aead25f7aa879a358a046bc01291ef2e260a2b367a990dacc255c17a2a5a31",
    "body_local_points_f64": (
        "522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db"
    ),
    "body_local_counts_i64": (
        "f17eac58b9b109f98f7a69efcc3b1e64b632d805ccca8cc8883cf0349e07cb6c"
    ),
    "body_local_indices_i64": (
        "205a08458b895d96c6eb9593d1f04a8815629f7f972a889cce683b86955f2545"
    ),
}
EXPECTED_SOURCE_COUNTS = {
    "vertices": 41094,
    "faces": 13698,
    "triangles": 13698,
}
EXPECTED_SOURCE_BOUNDS_M = (
    (-0.010767397438303794, -0.009999632356670897, -0.0386173457368133),
    (0.06708260664084253, 0.015240367659608567, 0.0007502218245168529),
)
EXPECTED_SOURCE_MATERIAL_TARGET = (
    "/roarm_m3/gripper_link/collisions/gripper_link/"
    "node_STL_BINARY_/Looks/DefaultMaterial"
)
EXPECTED_LIVE_MATERIAL_TARGET = (
    "/World/Robot/gripper_link/collisions/gripper_link/"
    "node_STL_BINARY_/Looks/DefaultMaterial"
)

SDF_ATTRIBUTE_SPECS = {
    "physxSDFMeshCollision:sdfResolution": ("int", 256, None),
    "physxSDFMeshCollision:sdfSubgridResolution": ("int", 6, None),
    "physxSDFMeshCollision:sdfBitsPerSubgridPixel": (
        "token",
        "BitsPerPixel16",
        None,
    ),
    "physxSDFMeshCollision:sdfNarrowBandThickness": (
        "float",
        0.01,
        "0x3c23d70a",
    ),
    "physxSDFMeshCollision:sdfMargin": ("float", 0.01, "0x3c23d70a"),
    "physxSDFMeshCollision:sdfEnableRemeshing": ("bool", False, None),
    "physxSDFMeshCollision:sdfTriangleCountReductionFactor": (
        "float",
        1.0,
        "0x3f800000",
    ),
}
SEMANTIC_ALLOWED_MESH_ATTRIBUTES = (
    "physics:collisionEnabled",
    "physics:approximation",
    *tuple(SDF_ATTRIBUTE_SPECS),
)
REQUIRED_APPLIED_APIS = (
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxSDFMeshCollisionAPI",
)

EXPECTED_TIMELINE = {
    "is_playing": False,
    "is_stopped": True,
    "current_time_s": 0.0,
    "start_time": 0.0,
    "end_time": 1.6666666666666667,
    "time_codes_per_seconds": 60.0,
}
COOK_WAIT_TIMEOUT_S = 180.0
PROPERTY_QUERY_TIMEOUT_S = 30.0
MAX_APP_UPDATE_PUMPS = 300000

MASS_BASELINE = {
    "gripper_link": {
        "mass_kg": 0.0028707999736070633,
        "center_of_mass_m": [
            0.027000000700354576,
            0.0027000000700354576,
            -0.018935000523924828,
        ],
        "diagonal_inertia": [
            5.232159878687526e-7,
            0.0000018207101675216109,
            0.0000016023100215534214,
        ],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
    "link5": {
        "mass_kg": 0.015392799861729145,
        "center_of_mass_m": [
            -0.007799999788403511,
            0.0,
            0.05949999764561653,
        ],
        "diagonal_inertia": [
            0.000020286899598431773,
            0.00002144270001736004,
            0.0000043949498831352685,
        ],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
    },
}
MASS_TOLERANCES = {
    "mass_kg": (1.0e-12, 1.0e-7),
    "center_of_mass_m": (1.0e-12, 1.0e-7),
    "diagonal_inertia": (5.0e-13, 1.0e-7),
    "principal_axes_wxyz": (1.0e-12, 1.0e-7),
}

EXACT_COUNTER_KEYS = (
    "actual_worker_invocations",
    "automatic_retries",
    "simulation_app_launches",
    "derivative_asset_materializations",
    "collision_scope_instanceable_false_writes",
    "gripper_a64_collision_enable_changes",
    "gripper_sdf_mesh_collision_enable_true_writes",
    "gripper_sdf_api_apply_sets",
    "sdf_parameter_attr_writes",
    "link5_collision_representation_changes",
    "source_geometry_stream_changes",
    "p34_or_d397_geometry_reads_for_materialization",
    "automatic_decomposition_sweeps",
    "sdf_resolution_sweeps",
    "sdf_remesh_operations",
    "simulation_context_constructions",
    "resets",
    "timeline_play_requests",
    "timeline_commit_requests",
    "timeline_raw_stop_time_zero_checks",
    "physx_stage_attaches",
    "physx_stage_detaches",
    "physx_property_queries",
    "stagecache_erase_calls",
    "simulation_app_update_pumps",
    "controlled_physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "unregistered_nonrepresentation_material_mass_actuator_scene_solver_setting_changes",
    "physx_convex_callback_requests",
    "sdf_tensor_or_distance_queries",
    "isaac_hydra_renders",
)
EXACT_PASS_COUNTERS = {
    "actual_worker_invocations": 1,
    "automatic_retries": 0,
    "simulation_app_launches": 1,
    "derivative_asset_materializations": 1,
    "collision_scope_instanceable_false_writes": 2,
    "gripper_a64_collision_enable_changes": 64,
    "gripper_sdf_mesh_collision_enable_true_writes": 1,
    "gripper_sdf_api_apply_sets": 1,
    "sdf_parameter_attr_writes": 7,
    "link5_collision_representation_changes": 0,
    "source_geometry_stream_changes": 0,
    "p34_or_d397_geometry_reads_for_materialization": 0,
    "automatic_decomposition_sweeps": 0,
    "sdf_resolution_sweeps": 0,
    "sdf_remesh_operations": 0,
    "simulation_context_constructions": 0,
    "resets": 0,
    "timeline_play_requests": 0,
    "timeline_commit_requests": 0,
    "timeline_raw_stop_time_zero_checks": 2,
    "physx_stage_attaches": 1,
    "physx_stage_detaches": 1,
    "physx_property_queries": 2,
    "stagecache_erase_calls": 1,
    "controlled_physics_steps": 0,
    "public_forwards": 0,
    "q5_commands": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "cylinder_creates_or_writes": 0,
    "target_ik_path_pose_changes": 0,
    "unregistered_nonrepresentation_material_mass_actuator_scene_solver_setting_changes": 0,
    "physx_convex_callback_requests": 0,
    "sdf_tensor_or_distance_queries": 0,
    "isaac_hydra_renders": 0,
}


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


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_no_duplicates(text: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject)


def _read_json(path: Path) -> dict[str, Any]:
    value = _json_no_duplicates(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
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


def _runtime_stack_probe() -> dict[str, Any]:
    import importlib.metadata
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    physx_extension_id = manager.get_extension_id_by_module("omni.physx")
    physx_extension = (
        manager.get_extension_dict(physx_extension_id)
        if physx_extension_id
        else {}
    )
    physx_extension_root = (
        Path(manager.get_extension_path(physx_extension_id)).resolve()
        if physx_extension_id
        else None
    )
    expected_extension_root = PHYSX_NATIVE_PLUGIN.parent.parent.resolve()
    active_extension_plugin = (
        physx_extension_root / "bin/libomni.physx.plugin.so"
        if physx_extension_root is not None
        else None
    )
    physx_extension_version = (
        physx_extension.get("package", {}).get("version")
        if isinstance(physx_extension, dict)
        else None
    )
    active_plugin_sha256 = (
        _sha(active_extension_plugin)
        if active_extension_plugin is not None
        and active_extension_plugin.is_file()
        else None
    )
    checks = {
        "isaac_sim_distribution_exact": importlib.metadata.version(
            "isaacsim"
        )
        == "5.1.0.0",
        "isaac_lab_distribution_exact": importlib.metadata.version(
            "isaaclab"
        )
        == "2.3.0",
        "omni_physx_extension_id_resolved": bool(physx_extension_id),
        "omni_physx_extension_version_exact": str(
            physx_extension_version
        )
        == "107.3.26",
        "active_extension_root_exact": physx_extension_root
        == expected_extension_root,
        "active_extension_native_plugin_hash_exact": active_plugin_sha256
        == EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256,
    }
    return {
        "supported_runtime_probe": {
            "method": (
                "Kit extension manager get_extension_id_by_module"
                "('omni.physx') then package.version"
            ),
            "extension_id": str(physx_extension_id),
            "extension_root": (
                str(physx_extension_root)
                if physx_extension_root is not None
                else None
            ),
            "omni_physx_extension_version": (
                str(physx_extension_version)
                if physx_extension_version is not None
                else None
            ),
        },
        "physx_sdk_engine_provenance": {
            "classification": (
                "native_plugin_hash_bound_embedded_evidence; "
                "not a public runtime version getter"
            ),
            "native_plugin_path": str(PHYSX_NATIVE_PLUGIN),
            "active_extension_native_plugin_path": (
                str(active_extension_plugin)
                if active_extension_plugin is not None
                else None
            ),
            "active_extension_native_plugin_sha256": (
                active_plugin_sha256
            ),
            "expected_native_plugin_sha256": (
                EXPECTED_PHYSX_NATIVE_PLUGIN_SHA256
            ),
            "physx_sdk_engine_version_inference": "5.6.1",
            "omni_physx_extension_version_is_not_engine_version": True,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _phase(out_dir: Path, name: str, **fields: Any) -> None:
    path = out_dir / PHASE_NAME
    ordinal = 1
    if path.is_file():
        ordinal = (
            sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            + 1
        )
    row = {
        "ordinal": ordinal,
        "phase": name,
        "owner": "worker",
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    encoded = json.dumps(
        row, sort_keys=True, ensure_ascii=False, default=_json_default
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"D400_PHASE {encoded}", flush=True)


def _heartbeat(label: str, **fields: Any) -> None:
    row = {
        "label": label,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    print(
        "D400_HEARTBEAT "
        + json.dumps(row, sort_keys=True, ensure_ascii=False, default=_json_default),
        flush=True,
    )


def _counter_template() -> dict[str, int]:
    counters = {key: 0 for key in EXACT_COUNTER_KEYS}
    counters["actual_worker_invocations"] = 1
    counters["simulation_app_launches"] = 1
    return counters


def _array_bytes(value: Any, dtype: str) -> bytes:
    import numpy as np

    return np.ascontiguousarray(np.asarray(value, dtype=dtype)).tobytes(order="C")


def _float32_bits_hex(value: Any) -> str:
    packed = struct.pack("<f", float(value))
    return f"0x{struct.unpack('<I', packed)[0]:08x}"


def _path_set_sha(paths: list[str] | tuple[str, ...]) -> str:
    return _sha_bytes(("".join(f"{path}\n" for path in sorted(paths))).encode("utf-8"))


def _asset_hashes(asset_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(asset_dir)): _sha(path)
        for path in sorted(item for item in asset_dir.rglob("*") if item.is_file())
    }


def _timeline_tuple() -> dict[str, Any]:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    return {
        "is_playing": bool(timeline.is_playing()),
        "is_stopped": bool(timeline.is_stopped()),
        "current_time_s": float(timeline.get_current_time()),
        "start_time": float(timeline.get_start_time()),
        "end_time": float(timeline.get_end_time()),
        "time_codes_per_seconds": float(
            timeline.get_time_codes_per_seconds()
        ),
    }


def _timeline_gate(snapshot: dict[str, Any]) -> dict[str, Any]:
    checks = {
        key: type(snapshot.get(key)) is type(expected)
        and snapshot.get(key) == expected
        for key, expected in EXPECTED_TIMELINE.items()
    }
    return {
        "observed": snapshot,
        "expected": dict(EXPECTED_TIMELINE),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _mesh_stream_record(stage: Any, mesh_path: str, body_path: str) -> dict[str, Any]:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    body_prim = stage.GetPrimAtPath(body_path)
    if not prim.IsValid() or not body_prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing required Mesh/body: {mesh_path}, {body_path}")
    mesh = UsdGeom.Mesh(prim)
    points_value = list(mesh.GetPointsAttr().Get() or [])
    counts_value = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices_value = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    points = np.asarray(
        [[float(component) for component in row] for row in points_value],
        dtype="<f4",
    )
    counts_i32 = np.asarray(counts_value, dtype="<i4")
    indices_i32 = np.asarray(indices_value, dtype="<i4")
    counts_i64 = np.asarray(counts_value, dtype="<i8")
    indices_i64 = np.asarray(indices_value, dtype="<i8")
    mesh_l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_w2l = (
        UsdGeom.Xformable(body_prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .GetInverse()
    )
    body_local_points = np.asarray(
        [
            [
                float(component)
                for component in body_w2l.Transform(
                    mesh_l2w.Transform(
                        Gf.Vec3d(*[float(component) for component in row])
                    )
                )
            ]
            for row in points_value
        ],
        dtype="<f8",
    )
    relationships = prim.GetRelationships()
    material_targets = []
    for relationship in relationships:
        if relationship.GetName() == "material:binding":
            material_targets.extend(
                target.pathString for target in relationship.GetTargets()
            )
    orientation_attr = mesh.GetOrientationAttr()
    subdivision_attr = mesh.GetSubdivisionSchemeAttr()
    holes_attr = mesh.GetHoleIndicesAttr()
    stream_hashes = {
        "points_f32": _sha_bytes(_array_bytes(points, "<f4")),
        "counts_i32": _sha_bytes(_array_bytes(counts_i32, "<i4")),
        "indices_i32": _sha_bytes(_array_bytes(indices_i32, "<i4")),
        "combined": _sha_bytes(
            _array_bytes(points, "<f4")
            + _array_bytes(counts_i32, "<i4")
            + _array_bytes(indices_i32, "<i4")
        ),
        "body_local_points_f64": _sha_bytes(
            _array_bytes(body_local_points, "<f8")
        ),
        "body_local_counts_i64": _sha_bytes(_array_bytes(counts_i64, "<i8")),
        "body_local_indices_i64": _sha_bytes(_array_bytes(indices_i64, "<i8")),
    }
    return {
        "mesh_path": mesh_path,
        "body_path": body_path,
        "points_type": str(mesh.GetPointsAttr().GetTypeName()),
        "counts_type": str(mesh.GetFaceVertexCountsAttr().GetTypeName()),
        "indices_type": str(mesh.GetFaceVertexIndicesAttr().GetTypeName()),
        "vertex_count": int(points.shape[0]),
        "face_count": int(counts_i32.size),
        "triangle_count": int(counts_i32.size)
        if bool(np.all(counts_i32 == 3))
        else None,
        "stream_hashes": stream_hashes,
        "body_local_bounds_m": [
            body_local_points.min(axis=0).tolist(),
            body_local_points.max(axis=0).tolist(),
        ],
        "orientation": {
            "authored": bool(orientation_attr.HasAuthoredValueOpinion()),
            "effective": str(orientation_attr.Get()),
        },
        "subdivision_scheme": {
            "authored": bool(subdivision_attr.HasAuthoredValueOpinion()),
            "effective": str(subdivision_attr.Get()),
        },
        "hole_indices": {
            "authored": bool(holes_attr.HasAuthoredValueOpinion()),
            "effective": [int(value) for value in list(holes_attr.Get() or [])],
        },
        "material_binding_targets": material_targets,
        "_points": points,
        "_counts_i32": counts_i32,
        "_indices_i32": indices_i32,
        "_body_local_points": body_local_points,
    }


def _public_mesh_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _stream_gate(record: dict[str, Any], expected_material: str) -> dict[str, Any]:
    import numpy as np

    bounds = np.asarray(record["body_local_bounds_m"], dtype=np.float64)
    expected_bounds = np.asarray(EXPECTED_SOURCE_BOUNDS_M, dtype=np.float64)
    checks = {
        "points_type_point3f_array": record["points_type"] == "point3f[]",
        "counts_type_int_array": record["counts_type"] == "int[]",
        "indices_type_int_array": record["indices_type"] == "int[]",
        "vertex_count_exact": record["vertex_count"]
        == EXPECTED_SOURCE_COUNTS["vertices"],
        "face_count_exact": record["face_count"] == EXPECTED_SOURCE_COUNTS["faces"],
        "triangle_count_exact": record["triangle_count"]
        == EXPECTED_SOURCE_COUNTS["triangles"],
        "all_stream_hashes_exact": record["stream_hashes"] == SOURCE_STREAM_HASHES,
        "body_local_bounds_exact": bool(np.array_equal(bounds, expected_bounds)),
        "orientation_authored_false": record["orientation"]["authored"] is False,
        "orientation_right_handed": record["orientation"]["effective"]
        == "rightHanded",
        "subdivision_authored_true": record["subdivision_scheme"]["authored"]
        is True,
        "subdivision_none": record["subdivision_scheme"]["effective"] == "none",
        "hole_indices_authored_false": record["hole_indices"]["authored"] is False,
        "hole_indices_empty": record["hole_indices"]["effective"] == [],
        "one_material_binding_exact": record["material_binding_targets"]
        == [expected_material],
    }
    return {"checks": checks, "pass": all(checks.values())}


def _mass_row(stage: Any, path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError(f"missing MassAPI owner {path}")
    api = UsdPhysics.MassAPI(prim)
    mass_attr = api.GetMassAttr()
    com_attr = api.GetCenterOfMassAttr()
    inertia_attr = api.GetDiagonalInertiaAttr()
    axes_attr = api.GetPrincipalAxesAttr()
    com = com_attr.Get()
    inertia = inertia_attr.Get()
    axes = axes_attr.Get()
    return {
        "path": path,
        "usd_types": {
            "mass": str(mass_attr.GetTypeName()),
            "center_of_mass": str(com_attr.GetTypeName()),
            "diagonal_inertia": str(inertia_attr.GetTypeName()),
            "principal_axes": str(axes_attr.GetTypeName()),
        },
        "mass_kg": float(mass_attr.Get()),
        "center_of_mass_m": [float(value) for value in com],
        "diagonal_inertia": [float(value) for value in inertia],
        "principal_axes_wxyz": [
            float(axes.GetReal()),
            *[float(value) for value in axes.GetImaginary()],
        ],
    }


def _authored_mass_gate(base_stage: Any, derivative_stage: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for body in ("link5", "gripper_link"):
        base = _mass_row(base_stage, SOURCE_BODY_PATHS[body])
        derivative = _mass_row(derivative_stage, SOURCE_BODY_PATHS[body])
        checks = {
            "usd_types_exact": base["usd_types"] == derivative["usd_types"],
            "mass_exact": base["mass_kg"] == derivative["mass_kg"],
            "center_of_mass_exact": base["center_of_mass_m"]
            == derivative["center_of_mass_m"],
            "diagonal_inertia_exact": base["diagonal_inertia"]
            == derivative["diagonal_inertia"],
            "principal_axes_exact": base["principal_axes_wxyz"]
            == derivative["principal_axes_wxyz"],
        }
        rows[body] = {
            "base": base,
            "derivative": derivative,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"bodies": rows, "pass": all(row["pass"] for row in rows.values())}


def _canonical_usd_value(value: Any) -> Any:
    """Encode a composed USD value without repr or process-local addresses."""

    from pxr import Sdf

    type_name = f"{type(value).__module__}.{type(value).__name__}"
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {
            "type": "float64",
            "bits_le": struct.pack("<d", float(value)).hex(),
            "finite": math.isfinite(float(value)),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, Sdf.Path):
        return {"type": "pxr.Sdf.Path", "path": value.pathString}
    if isinstance(value, Sdf.AssetPath):
        return {
            "type": "pxr.Sdf.AssetPath",
            "authored_path": value.path,
            "resolved_path": value.resolvedPath,
        }
    if isinstance(value, Sdf.LayerOffset):
        return {
            "type": "pxr.Sdf.LayerOffset",
            "offset": _canonical_usd_value(float(value.offset)),
            "scale": _canonical_usd_value(float(value.scale)),
        }
    if isinstance(value, Sdf.Reference):
        return {
            "type": "pxr.Sdf.Reference",
            "asset_path": value.assetPath,
            "prim_path": value.primPath.pathString,
            "layer_offset": _canonical_usd_value(value.layerOffset),
            "custom_data": _canonical_usd_value(value.customData),
        }
    if isinstance(value, Sdf.Payload):
        return {
            "type": "pxr.Sdf.Payload",
            "asset_path": value.assetPath,
            "prim_path": value.primPath.pathString,
            "layer_offset": _canonical_usd_value(value.layerOffset),
        }
    if isinstance(value, Sdf.ValueTypeName):
        return {
            "type": "pxr.Sdf.ValueTypeName",
            "name": str(value),
            "aliases": list(value.aliasesAsStrings),
            "cpp_type_name": value.cppTypeName,
            "is_array": bool(value.isArray),
            "is_scalar": bool(value.isScalar),
            "role": value.role,
        }
    if isinstance(value, (Sdf.Specifier, Sdf.Variability)):
        return {
            "type": type_name,
            "name": value.name,
            "value": str(value.value),
        }
    if isinstance(value, Sdf.ValueBlock):
        return {"type": "pxr.Sdf.ValueBlock"}
    if type(value).__module__ == "pxr.Sdf" and type(value).__name__.endswith(
        "ListOp"
    ):
        required = (
            "isExplicit",
            "explicitItems",
            "addedItems",
            "prependedItems",
            "appendedItems",
            "deletedItems",
            "orderedItems",
        )
        if not all(hasattr(value, field) for field in required):
            raise TypeError(f"incomplete Sdf ListOp binding: {type_name}")
        return {
            "type": type_name,
            "is_explicit": bool(value.isExplicit),
            "explicit_items": [
                _canonical_usd_value(item) for item in value.explicitItems
            ],
            "added_items": [
                _canonical_usd_value(item) for item in value.addedItems
            ],
            "prepended_items": [
                _canonical_usd_value(item) for item in value.prependedItems
            ],
            "appended_items": [
                _canonical_usd_value(item) for item in value.appendedItems
            ],
            "deleted_items": [
                _canonical_usd_value(item) for item in value.deletedItems
            ],
            "ordered_items": [
                _canonical_usd_value(item) for item in value.orderedItems
            ],
            "applied_items": [
                _canonical_usd_value(item) for item in value.GetAppliedItems()
            ],
        }
    module_name = type(value).__module__
    class_name = type(value).__name__
    if module_name == "pxr.Gf" and class_name.startswith("Quat"):
        width = {"h": 16, "f": 32, "d": 64}.get(class_name[-1])
        if width is None:
            raise TypeError(f"unsupported quaternion type: {type_name}")
        formatter = {16: "<e", 32: "<f", 64: "<d"}[width]
        return {
            "type": type_name,
            "real_bits_le": struct.pack(
                formatter, float(value.GetReal())
            ).hex(),
            "imaginary_bits_le": [
                struct.pack(formatter, float(component)).hex()
                for component in value.GetImaginary()
            ],
        }
    if module_name == "pxr.Gf" and class_name.startswith("Vec"):
        suffix = class_name[-1]
        if suffix in {"h", "f", "d"}:
            formatter = {"h": "<e", "f": "<f", "d": "<d"}[suffix]
            items = [
                {"float_bits_le": struct.pack(formatter, float(item)).hex()}
                for item in value
            ]
        elif suffix in {"i", "l"}:
            items = [{"int": str(int(item))} for item in value]
        else:
            raise TypeError(f"unsupported vector type: {type_name}")
        return {"type": type_name, "items": items}
    if module_name == "pxr.Gf" and class_name.startswith("Matrix"):
        return {
            "type": type_name,
            "rows": [_canonical_usd_value(row) for row in value],
        }
    if module_name == "pxr.Vt" and class_name.endswith("Array"):
        return {
            "type": type_name,
            "length": len(value),
            "items": [_canonical_usd_value(item) for item in value],
        }
    if isinstance(value, dict):
        rows = [
            [_canonical_usd_value(key), _canonical_usd_value(item)]
            for key, item in value.items()
        ]
        rows.sort(
            key=lambda row: json.dumps(
                row[0], sort_keys=True, separators=(",", ":")
            )
        )
        return {"type": "dict", "items": rows}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_canonical_usd_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_canonical_usd_value(item) for item in value],
        }
    raise TypeError(f"unsupported USD/PXR value type: {type_name}")


def _canonical_metadata(metadata: dict[str, Any]) -> list[list[Any]]:
    return [
        [key, _canonical_usd_value(metadata[key])]
        for key in sorted(metadata)
    ]


def _composed_stage_rows(stage: Any) -> list[dict[str, Any]]:
    from pxr import Usd

    rows = []
    prims = sorted(
        Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()),
        key=lambda prim: prim.GetPath().pathString,
    )
    for prim in prims:
        attributes = []
        for attr in sorted(prim.GetAttributes(), key=lambda item: item.GetName()):
            attributes.append(
                {
                    "name": attr.GetName(),
                    "type_name": str(attr.GetTypeName()),
                    "value": _canonical_usd_value(attr.Get()),
                    "time_samples": [
                        {
                            "time_bits_le": struct.pack(
                                "<d", float(time_code)
                            ).hex(),
                            "value": _canonical_usd_value(attr.Get(time_code)),
                        }
                        for time_code in attr.GetTimeSamples()
                    ],
                    "connections": [
                        _canonical_usd_value(item)
                        for item in attr.GetConnections()
                    ],
                    "metadata": _canonical_metadata(attr.GetAllMetadata()),
                }
            )
        relationships = []
        for relationship in sorted(
            prim.GetRelationships(), key=lambda item: item.GetName()
        ):
            relationships.append(
                {
                    "name": relationship.GetName(),
                    "targets": [
                        _canonical_usd_value(item)
                        for item in relationship.GetTargets()
                    ],
                    "metadata": _canonical_metadata(
                        relationship.GetAllMetadata()
                    ),
                }
            )
        rows.append(
            {
                "path": prim.GetPath().pathString,
                "type_name": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
                "instanceable": bool(prim.IsInstanceable()),
                "applied_schemas": [
                    str(item) for item in prim.GetAppliedSchemas()
                ],
                "metadata": _canonical_metadata(prim.GetAllMetadata()),
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _replace_asset_root(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(root, "$ASSET_ROOT")
    if isinstance(value, list):
        return [_replace_asset_root(item, root) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_asset_root(item, root)
            for key, item in value.items()
        }
    return value


def _drop_metadata(metadata: list[list[Any]], key: str) -> list[list[Any]]:
    return [row for row in metadata if row[0] != key]


def _mask_metadata_value(
    metadata: list[list[Any]], key: str, marker: str
) -> list[list[Any]]:
    return [
        [row[0], marker if row[0] == key else row[1]]
        for row in metadata
    ]


def _normalize_allowlisted_semantics(
    rows: list[dict[str, Any]], asset_root: Path
) -> list[dict[str, Any]]:
    normalized = _replace_asset_root(rows, str(asset_root.resolve()))
    for row in normalized:
        path = row["path"]
        if path in SOURCE_COLLISION_SCOPES:
            row["instanceable"] = "$ALLOWLIST_INSTANCEABLE"
            row["metadata"] = _mask_metadata_value(
                row["metadata"],
                "instanceable",
                "$ALLOWLIST_INSTANCEABLE",
            )
        if path in SOURCE_GRIPPER_A64_PATHS:
            for attr in row["attributes"]:
                if attr["name"] == "physics:collisionEnabled":
                    attr["value"] = "$ALLOWLIST_COLLISION_ENABLED_VALUE"
        if path == SOURCE_MESH_PATH:
            row["applied_schemas"] = [
                schema
                for schema in row["applied_schemas"]
                if schema not in REQUIRED_APPLIED_APIS
            ]
            row["metadata"] = _drop_metadata(row["metadata"], "apiSchemas")
            row["attributes"] = [
                attr
                for attr in row["attributes"]
                if attr["name"] not in SEMANTIC_ALLOWED_MESH_ATTRIBUTES
            ]
    return normalized


def _rows_digest(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    row_hashes = [
        _sha_bytes(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        for row in rows
    ]
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha_bytes(payload), row_hashes


def _composed_semantic_diff_gate(
    base_stage: Any,
    derivative_stage: Any,
    derivative_dir: Path,
) -> dict[str, Any]:
    base_rows = _composed_stage_rows(base_stage)
    derivative_rows = _composed_stage_rows(derivative_stage)
    normalized_base = _normalize_allowlisted_semantics(
        base_rows, BASE_ASSET_DIR
    )
    normalized_derivative = _normalize_allowlisted_semantics(
        derivative_rows, derivative_dir
    )
    base_digest, base_row_hashes = _rows_digest(normalized_base)
    derivative_digest, derivative_row_hashes = _rows_digest(
        normalized_derivative
    )
    base_paths = [row["path"] for row in normalized_base]
    derivative_paths = [row["path"] for row in normalized_derivative]
    mismatches = [
        {
            "index": index,
            "base_path": (
                base_paths[index] if index < len(base_paths) else None
            ),
            "derivative_path": (
                derivative_paths[index]
                if index < len(derivative_paths)
                else None
            ),
            "base_row_sha256": (
                base_row_hashes[index]
                if index < len(base_row_hashes)
                else None
            ),
            "derivative_row_sha256": (
                derivative_row_hashes[index]
                if index < len(derivative_row_hashes)
                else None
            ),
        }
        for index in range(max(len(base_row_hashes), len(derivative_row_hashes)))
        if (
            index >= len(base_row_hashes)
            or index >= len(derivative_row_hashes)
            or base_row_hashes[index] != derivative_row_hashes[index]
        )
    ]
    checks = {
        "composed_prim_path_sequence_exact": base_paths == derivative_paths,
        "normalized_nonallowlisted_row_hashes_exact": (
            base_row_hashes == derivative_row_hashes
        ),
        "normalized_nonallowlisted_digest_exact": (
            base_digest == derivative_digest
        ),
        "normalized_nonallowlisted_diff_count_zero": not mismatches,
    }
    return {
        "method": (
            "closed_typed_composed_prim_attribute_relationship_encoder_then_"
            "exact_registered_allowlist_normalization"
        ),
        "allowlist": {
            "instanceable_paths": list(SOURCE_COLLISION_SCOPES),
            "collision_enabled_false_paths": list(SOURCE_GRIPPER_A64_PATHS),
            "sdf_mesh_path": SOURCE_MESH_PATH,
            "sdf_mesh_allowed_attributes": list(
                SEMANTIC_ALLOWED_MESH_ATTRIBUTES
            ),
            "sdf_mesh_allowed_applied_apis": list(REQUIRED_APPLIED_APIS),
        },
        "base_row_count": len(base_rows),
        "derivative_row_count": len(derivative_rows),
        "base_normalized_sha256": base_digest,
        "derivative_normalized_sha256": derivative_digest,
        "nonallowlisted_mismatches": mismatches,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _author_sdf_derivative(
    out_dir: Path, counters: dict[str, int]
) -> tuple[Path, dict[str, Any]]:
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    collision_root = out_dir / "collision_asset"
    derivative_dir = collision_root / ASSET_NAME
    if collision_root.exists() or derivative_dir.exists():
        raise RuntimeError("D400 derivative path already exists; overwrite refused")
    _phase(out_dir, "derivative_copy_start", source=_rel(BASE_ASSET_DIR))
    collision_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASE_ASSET_DIR, derivative_dir)
    counters["derivative_asset_materializations"] += 1
    before_hashes = _asset_hashes(BASE_ASSET_DIR)
    copied_hashes = _asset_hashes(derivative_dir)
    if copied_hashes != before_hashes:
        raise RuntimeError("D400 copy is not bit-exact before authoring")

    root_path = derivative_dir / "roarm_m3.usd"
    stage = Usd.Stage.Open(str(root_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open copied D400 root stage {root_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    source_before = _mesh_stream_record(
        stage, SOURCE_MESH_PATH, SOURCE_BODY_PATHS["gripper_link"]
    )
    source_before_gate = _stream_gate(
        source_before, EXPECTED_SOURCE_MATERIAL_TARGET
    )
    if not source_before_gate["pass"]:
        raise RuntimeError(f"frozen source stream gate failed: {source_before_gate}")

    instanceability_before = {}
    for path in SOURCE_COLLISION_SCOPES:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing collision scope {path}")
        instanceability_before[path] = bool(prim.IsInstanceable())
        if not prim.SetInstanceable(False):
            raise RuntimeError(f"failed to author instanceable=false at {path}")
        counters["collision_scope_instanceable_false_writes"] += 1

    disabled_rows = []
    for path in SOURCE_GRIPPER_A64_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"missing frozen gripper A64 collider {path}")
        api = UsdPhysics.CollisionAPI(prim)
        before = api.GetCollisionEnabledAttr().Get()
        api.CreateCollisionEnabledAttr(False)
        after = api.GetCollisionEnabledAttr().Get()
        counters["gripper_a64_collision_enable_changes"] += 1
        disabled_rows.append(
            {
                "path": path,
                "before": True if before is None else bool(before),
                "after": bool(after),
            }
        )

    mesh_prim = stage.GetPrimAtPath(SOURCE_MESH_PATH)
    if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing full gripper Mesh {SOURCE_MESH_PATH}")
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    if not collision_api or not mesh_api or not sdf_api:
        raise RuntimeError("failed to apply the three registered SDF API schemas")
    counters["gripper_sdf_api_apply_sets"] += 1
    collision_api.CreateCollisionEnabledAttr(True)
    counters["gripper_sdf_mesh_collision_enable_true_writes"] += 1
    mesh_api.CreateApproximationAttr("sdf")

    value_types = {
        "int": Sdf.ValueTypeNames.Int,
        "token": Sdf.ValueTypeNames.Token,
        "float": Sdf.ValueTypeNames.Float,
        "bool": Sdf.ValueTypeNames.Bool,
    }
    for name, (type_name, value, _bits) in SDF_ATTRIBUTE_SPECS.items():
        attr = mesh_prim.CreateAttribute(
            name,
            value_types[type_name],
            custom=False,
            variability=Sdf.VariabilityUniform,
        )
        if not attr.Set(value):
            raise RuntimeError(f"failed to set registered SDF attribute {name}")
        counters["sdf_parameter_attr_writes"] += 1
    _phase(
        out_dir,
        "derivative_sdf_opinion_write_end",
        instanceable_false_writes=counters[
            "collision_scope_instanceable_false_writes"
        ],
        gripper_a64_disable_writes=counters[
            "gripper_a64_collision_enable_changes"
        ],
        sdf_attr_writes=counters["sdf_parameter_attr_writes"],
    )
    stage.GetRootLayer().Save()

    derivative_stage = Usd.Stage.Open(str(root_path), load=Usd.Stage.LoadAll)
    base_stage = Usd.Stage.Open(str(BASE_ROOT_USD), load=Usd.Stage.LoadAll)
    if derivative_stage is None or base_stage is None:
        raise RuntimeError("failed to reopen base/derivative stage")
    source_after = _mesh_stream_record(
        derivative_stage, SOURCE_MESH_PATH, SOURCE_BODY_PATHS["gripper_link"]
    )
    source_after_gate = _stream_gate(
        source_after, EXPECTED_SOURCE_MATERIAL_TARGET
    )
    sdf_readback = _sdf_prim_readback(
        derivative_stage, SOURCE_MESH_PATH, expected_live=False
    )
    mass_gate = _authored_mass_gate(base_stage, derivative_stage)
    semantic_diff_gate = _composed_semantic_diff_gate(
        base_stage, derivative_stage, derivative_dir
    )
    after_hashes = _asset_hashes(derivative_dir)
    changed_files = sorted(
        path
        for path in set(before_hashes) | set(after_hashes)
        if before_hashes.get(path) != after_hashes.get(path)
    )
    copy_and_opinion_checks = {
        "copy_bit_exact_before_authoring": copied_hashes == before_hashes,
        "only_root_layer_file_changed": changed_files == ["roarm_m3.usd"],
        "instanceable_false_writes_exactly_two": counters[
            "collision_scope_instanceable_false_writes"
        ]
        == 2,
        "both_scopes_were_instanceable_before": all(
            instanceability_before.get(path) is True
            for path in SOURCE_COLLISION_SCOPES
        ),
        "all_scopes_now_noninstanceable": all(
            not derivative_stage.GetPrimAtPath(path).IsInstanceable()
            for path in SOURCE_COLLISION_SCOPES
        ),
        "gripper_a64_disable_writes_exactly_64": counters[
            "gripper_a64_collision_enable_changes"
        ]
        == 64,
        "all_gripper_a64_before_enabled": all(
            row["before"] for row in disabled_rows
        ),
        "all_gripper_a64_after_disabled": all(
            not row["after"] for row in disabled_rows
        ),
        "source_stream_before_after_exact": source_before["stream_hashes"]
        == source_after["stream_hashes"],
        "source_stream_after_registered_exact": source_after_gate["pass"],
        "sdf_typed_readback_pass": sdf_readback["pass"],
        "authored_mass_com_inertia_exact": mass_gate["pass"],
        "composed_semantic_diff_allowlist_exact": semantic_diff_gate["pass"],
    }
    record = {
        "base_asset_dir": _rel(BASE_ASSET_DIR),
        "derivative_asset_dir": _rel(derivative_dir),
        "root_usd": _rel(root_path),
        "base_hashes": before_hashes,
        "copied_hashes_before_authoring": copied_hashes,
        "derivative_hashes_after_authoring": after_hashes,
        "changed_files": changed_files,
        "instanceability_before": instanceability_before,
        "disabled_gripper_a64": disabled_rows,
        "source_before": _public_mesh_record(source_before),
        "source_before_gate": source_before_gate,
        "source_after": _public_mesh_record(source_after),
        "source_after_gate": source_after_gate,
        "sdf_readback": sdf_readback,
        "authored_mass_gate": mass_gate,
        "composed_semantic_diff_gate": semantic_diff_gate,
        "checks": copy_and_opinion_checks,
        "pass": all(copy_and_opinion_checks.values()),
    }
    _phase(
        out_dir,
        "typed_authored_readback_gate_end",
        passed=record["pass"],
        source_combined_sha256=source_after["stream_hashes"]["combined"],
    )
    if not record["pass"]:
        raise RuntimeError(f"D400 authored derivative gate failed: {record['checks']}")
    return derivative_dir, record


def _sdf_prim_readback(
    stage: Any, path: str, *, expected_live: bool
) -> dict[str, Any]:
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return {"path": path, "pass": False, "error": "invalid prim"}
    applied = [str(value) for value in prim.GetAppliedSchemas()]
    collision_api = UsdPhysics.CollisionAPI(prim)
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
    collision_attr = collision_api.GetCollisionEnabledAttr()
    approximation_attr = mesh_api.GetApproximationAttr()

    def attribute_shape(attribute: Any) -> dict[str, Any]:
        return {
            "valid": bool(attribute.IsValid()),
            "authored": bool(
                attribute.IsValid()
                and attribute.HasAuthoredValueOpinion()
            ),
            "custom": bool(attribute.IsValid() and attribute.IsCustom()),
            "variability": (
                str(attribute.GetVariability())
                if attribute.IsValid()
                else None
            ),
            "uniform": bool(
                attribute.IsValid()
                and attribute.GetVariability() == Sdf.VariabilityUniform
            ),
            "time_samples": (
                [float(value) for value in attribute.GetTimeSamples()]
                if attribute.IsValid()
                else []
            ),
            "connections": (
                [
                    value.pathString
                    for value in attribute.GetConnections()
                ]
                if attribute.IsValid()
                else []
            ),
        }

    collision_shape = attribute_shape(collision_attr)
    approximation_shape = attribute_shape(approximation_attr)
    attrs: dict[str, Any] = {}
    attr_checks: dict[str, bool] = {}
    for name, (expected_type, expected_value, expected_bits) in SDF_ATTRIBUTE_SPECS.items():
        attr = prim.GetAttribute(name)
        value = attr.Get() if attr.IsValid() else None
        row = {
            **attribute_shape(attr),
            "usd_type": str(attr.GetTypeName()) if attr.IsValid() else None,
            "value": value,
            "float32_bits_hex": (
                _float32_bits_hex(value)
                if expected_type == "float" and value is not None
                else None
            ),
        }
        attrs[name] = row
        attr_checks[name] = bool(
            row["valid"]
            and row["authored"]
            and row["custom"] is False
            and row["uniform"] is True
            and row["time_samples"] == []
            and row["connections"] == []
            and row["usd_type"] == expected_type
            and row["value"] == expected_value
            and (
                expected_bits is None
                or row["float32_bits_hex"] == expected_bits
            )
        )
    material_targets = []
    for relationship in prim.GetRelationships():
        if relationship.GetName() == "material:binding":
            material_targets.extend(
                target.pathString for target in relationship.GetTargets()
            )
    expected_material = (
        EXPECTED_LIVE_MATERIAL_TARGET
        if expected_live
        else EXPECTED_SOURCE_MATERIAL_TARGET
    )
    required_instance_state = not expected_live or (
        not prim.IsInstanceable()
        and not prim.IsInstance()
        and not prim.IsInstanceProxy()
    )
    checks = {
        "is_mesh": bool(prim.IsA(UsdGeom.Mesh)),
        "collision_enabled_true": bool(collision_attr.Get()),
        "collision_enabled_authored_uniform_noncustom_default_only": bool(
            collision_shape["valid"]
            and collision_shape["authored"]
            and collision_shape["custom"] is False
            and collision_shape["uniform"] is True
            and collision_shape["time_samples"] == []
            and collision_shape["connections"] == []
        ),
        "approximation_sdf": str(approximation_attr.Get()) == "sdf",
        "approximation_authored_uniform_noncustom_default_only": bool(
            approximation_shape["valid"]
            and approximation_shape["authored"]
            and approximation_shape["custom"] is False
            and approximation_shape["uniform"] is True
            and approximation_shape["time_samples"] == []
            and approximation_shape["connections"] == []
        ),
        "required_apis_applied": all(api in applied for api in REQUIRED_APPLIED_APIS),
        "all_seven_attrs_exact": all(attr_checks.values())
        and len(attrs) == 7,
        "material_binding_exact": material_targets == [expected_material],
        "noninstance_state_when_live": required_instance_state,
    }
    return {
        "path": path,
        "applied_schemas": applied,
        "collision_enabled": collision_api.GetCollisionEnabledAttr().Get(),
        "collision_enabled_shape": collision_shape,
        "approximation": str(mesh_api.GetApproximationAttr().Get()),
        "approximation_shape": approximation_shape,
        "attributes": attrs,
        "attribute_checks": attr_checks,
        "material_binding_targets": material_targets,
        "instance_flags": {
            "is_instanceable": bool(prim.IsInstanceable()),
            "is_instance": bool(prim.IsInstance()),
            "is_instance_proxy": bool(prim.IsInstanceProxy()),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _make_live_stage(derivative_root: Path) -> tuple[Any, int, dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdUtils

    stage = Usd.Stage.CreateInMemory("d400_sdf_zero_step_preflight.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference(str(derivative_root.resolve()))
    if robot.GetPrim().IsInstanceable():
        raise RuntimeError("D400 live /World/Robot unexpectedly instanceable")
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    inserted = False
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
        inserted = True
    if not stage_id.IsValid():
        raise RuntimeError("failed to insert D400 live stage into StageCache")
    record = {
        "identifier": str(stage.GetRootLayer().identifier),
        "root_reference": _rel(derivative_root),
        "stage_id": int(stage_id.ToLongInt()),
        "stagecache_inserted": inserted,
        "persisted_usd_write": False,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
    }
    return stage, int(stage_id.ToLongInt()), record


def _nearest_rigid_owner(prim: Any) -> Any:
    from pxr import UsdPhysics

    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _live_inventory(stage: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    rows = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        body = next(
            (
                name
                for name, body_path in LIVE_BODY_PATHS.items()
                if path.startswith(body_path + "/")
            ),
            None,
        )
        if body is None or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled_value = (
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        )
        owner = _nearest_rigid_owner(prim)
        rows.append(
            {
                "body": body,
                "path": path,
                "collision_enabled": (
                    True if enabled_value is None else bool(enabled_value)
                ),
                "owner_path": (
                    owner.GetPath().pathString
                    if owner is not None and owner.IsValid()
                    else None
                ),
                "is_mesh": bool(prim.IsA(UsdGeom.Mesh)),
                "is_instanceable": bool(prim.IsInstanceable()),
                "is_instance": bool(prim.IsInstance()),
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "prototype_path_diagnostic": (
                    prim.GetPrimInPrototype().GetPath().pathString
                    if prim.IsInstanceProxy()
                    and prim.GetPrimInPrototype().IsValid()
                    else None
                ),
            }
        )
    rows.sort(key=lambda row: row["path"])
    enabled = {
        body: sorted(
            row["path"]
            for row in rows
            if row["body"] == body and row["collision_enabled"]
        )
        for body in LIVE_BODY_PATHS
    }
    disabled = {
        body: sorted(
            row["path"]
            for row in rows
            if row["body"] == body and not row["collision_enabled"]
        )
        for body in LIVE_BODY_PATHS
    }
    structure_rows = {}
    for path in LIVE_REQUIRED_NONINSTANCE_PATHS:
        prim = stage.GetPrimAtPath(path)
        structure_rows[path] = {
            "valid": bool(prim.IsValid()),
            "is_instanceable": bool(prim.IsValid() and prim.IsInstanceable()),
            "is_instance": bool(prim.IsValid() and prim.IsInstance()),
            "is_instance_proxy": bool(prim.IsValid() and prim.IsInstanceProxy()),
        }
    sdf_readback = _sdf_prim_readback(stage, LIVE_MESH_PATH, expected_live=True)
    stream = _mesh_stream_record(
        stage, LIVE_MESH_PATH, LIVE_BODY_PATHS["gripper_link"]
    )
    stream_gate = _stream_gate(stream, EXPECTED_LIVE_MATERIAL_TARGET)
    expected_link5_enabled = set(LIVE_LINK5_A64_PATHS)
    expected_gripper_enabled = {LIVE_MESH_PATH}
    expected_link5_disabled = {LIVE_LEGACY_PATHS["link5"]}
    expected_gripper_disabled = set(LIVE_GRIPPER_A64_PATHS) | {
        LIVE_LEGACY_PATHS["gripper_link"]
    }
    checks = {
        "required_structure_noninstance_nonproxy": all(
            row["valid"]
            and not row["is_instanceable"]
            and not row["is_instance"]
            and not row["is_instance_proxy"]
            for row in structure_rows.values()
        ),
        "no_collision_leaf_is_proxy": all(
            not row["is_instance"] and not row["is_instance_proxy"] for row in rows
        ),
        "all_owners_exact": all(
            row["owner_path"] == LIVE_BODY_PATHS[row["body"]] for row in rows
        ),
        "link5_enabled_exact_64": set(enabled["link5"])
        == expected_link5_enabled,
        "link5_disabled_exact_legacy": set(disabled["link5"])
        == expected_link5_disabled,
        "gripper_enabled_exact_sdf_one": set(enabled["gripper_link"])
        == expected_gripper_enabled,
        "gripper_disabled_exact_65": set(disabled["gripper_link"])
        == expected_gripper_disabled,
        "sdf_readback_exact": sdf_readback["pass"],
        "live_stream_exact": stream_gate["pass"],
    }
    return {
        "rows": rows,
        "enabled_paths": enabled,
        "disabled_paths": disabled,
        "structure": structure_rows,
        "sdf_readback": sdf_readback,
        "live_mesh_stream": _public_mesh_record(stream),
        "live_mesh_stream_gate": stream_gate,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _body_local_geometry(stage: Any, mesh_path: str, body_path: str) -> dict[str, Any]:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(mesh_path)
    body = stage.GetPrimAtPath(body_path)
    if not prim.IsValid() or not body.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing inspection geometry {mesh_path}")
    mesh = UsdGeom.Mesh(prim)
    points = list(mesh.GetPointsAttr().Get() or [])
    counts = np.asarray(
        list(mesh.GetFaceVertexCountsAttr().Get() or []), dtype=np.int64
    )
    indices = np.asarray(
        list(mesh.GetFaceVertexIndicesAttr().Get() or []), dtype=np.int64
    )
    if not bool(np.all(counts == 3)) or indices.size != counts.size * 3:
        raise RuntimeError(f"inspection subject is not triangular: {mesh_path}")
    mesh_l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_w2l = (
        UsdGeom.Xformable(body)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .GetInverse()
    )
    vertices = np.asarray(
        [
            [
                float(component)
                for component in body_w2l.Transform(
                    mesh_l2w.Transform(
                        Gf.Vec3d(*[float(component) for component in point])
                    )
                )
            ]
            for point in points
        ],
        dtype="<f8",
    )
    triangles = indices.reshape(-1, 3).astype("<i8", copy=False)
    return {
        "mesh_path": mesh_path,
        "body_path": body_path,
        "vertices_m": vertices.tolist(),
        "triangles": triangles.tolist(),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "vertices_f64_sha256": _sha_bytes(_array_bytes(vertices, "<f8")),
        "triangles_i64_sha256": _sha_bytes(_array_bytes(triangles, "<i8")),
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def _inspection_geometry_evidence(
    *,
    base_root: Path,
    live_stage: Any,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Usd

    base_stage = Usd.Stage.Open(str(base_root), load=Usd.Stage.LoadAll)
    if base_stage is None:
        raise RuntimeError("failed to open D344 source stage for inspection evidence")
    source = _body_local_geometry(
        base_stage, SOURCE_MESH_PATH, SOURCE_BODY_PATHS["gripper_link"]
    )
    live = _body_local_geometry(
        live_stage, LIVE_MESH_PATH, LIVE_BODY_PATHS["gripper_link"]
    )
    link5 = [
        _body_local_geometry(
            live_stage, path, LIVE_BODY_PATHS["link5"]
        )
        for path in LIVE_LINK5_A64_PATHS
    ]
    checks = {
        "source_vertex_count_exact": source["vertex_count"]
        == EXPECTED_SOURCE_COUNTS["vertices"],
        "source_triangle_count_exact": source["triangle_count"]
        == EXPECTED_SOURCE_COUNTS["triangles"],
        "source_live_vertices_exact": source["vertices_f64_sha256"]
        == live["vertices_f64_sha256"],
        "source_live_triangles_exact": source["triangles_i64_sha256"]
        == live["triangles_i64_sha256"],
        "link5_a64_exactly_64": len(link5) == 64,
        "link5_paths_exact": [row["mesh_path"] for row in link5]
        == list(LIVE_LINK5_A64_PATHS),
        "inventory_pass": inventory["pass"] is True,
    }
    return {
        "artifact": "D400_RERUN_INPUT_GEOMETRY_AND_OWNER_EVIDENCE_V1",
        "authority_guard": (
            "Original USD Float32 streams and property-query JSON decide gates; "
            "these Float64 arrays are controller-side inspection inputs only."
        ),
        "source_gripper_mesh": source,
        "live_sdf_input_mesh": live,
        "link5_a64": link5,
        "active_inventory": {
            "enabled_paths": inventory["enabled_paths"],
            "disabled_paths": inventory["disabled_paths"],
            "structure": inventory["structure"],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _cooking_stats() -> dict[str, int]:
    from omni.physx import get_physx_cooking_private_interface

    stats = get_physx_cooking_private_interface().get_cooking_statistics()
    scheduled = int(stats.total_scheduled_tasks)
    finished = int(stats.total_finished_tasks)
    return {
        "total_scheduled_tasks": scheduled,
        "total_finished_tasks": finished,
        "total_finished_cache_hit_tasks": int(
            stats.total_finished_cache_hit_tasks
        ),
        "total_finished_cache_miss_tasks": int(
            stats.total_finished_cache_miss_tasks
        ),
        "running_tasks": scheduled - finished,
    }


def _pump_app(app: Any, counters: dict[str, int]) -> None:
    if counters["simulation_app_update_pumps"] >= MAX_APP_UPDATE_PUMPS:
        raise RuntimeError("D400 bounded app-update pump budget exhausted")
    app.update()
    counters["simulation_app_update_pumps"] += 1


def _wait_for_cooking(
    app: Any,
    counters: dict[str, int],
    baseline: dict[str, int],
) -> dict[str, Any]:
    start = time.monotonic()
    samples = []
    last_heartbeat = start
    _pump_app(app, counters)
    while True:
        current = _cooking_stats()
        delta = {
            "scheduled": current["total_scheduled_tasks"]
            - baseline["total_scheduled_tasks"],
            "finished": current["total_finished_tasks"]
            - baseline["total_finished_tasks"],
            "cache_hits": current["total_finished_cache_hit_tasks"]
            - baseline["total_finished_cache_hit_tasks"],
            "cache_misses": current["total_finished_cache_miss_tasks"]
            - baseline["total_finished_cache_miss_tasks"],
        }
        if not samples or current != samples[-1]["statistics"]:
            samples.append(
                {
                    "elapsed_s": time.monotonic() - start,
                    "statistics": current,
                    "delta": delta,
                }
            )
        drained = bool(
            delta["scheduled"] > 0
            and delta["finished"] == delta["scheduled"]
            and current["running_tasks"] == 0
        )
        if drained:
            return {
                "baseline": baseline,
                "final": current,
                "delta": delta,
                "samples": samples,
                "elapsed_s": time.monotonic() - start,
                "release_local_mesh_cache_calls": 0,
                "checks": {
                    "baseline_running_zero": baseline["running_tasks"] == 0,
                    "scheduled_delta_positive": delta["scheduled"] > 0,
                    "finished_delta_equals_scheduled": delta["finished"]
                    == delta["scheduled"],
                    "final_running_zero": current["running_tasks"] == 0,
                    "at_least_one_update_pump": counters[
                        "simulation_app_update_pumps"
                    ]
                    >= 1,
                    "local_mesh_cache_not_cleared": True,
                },
                "pass": True,
            }
        now = time.monotonic()
        if now - start >= COOK_WAIT_TIMEOUT_S:
            return {
                "baseline": baseline,
                "final": current,
                "delta": delta,
                "samples": samples,
                "elapsed_s": now - start,
                "release_local_mesh_cache_calls": 0,
                "checks": {
                    "baseline_running_zero": baseline["running_tasks"] == 0,
                    "scheduled_delta_positive": delta["scheduled"] > 0,
                    "finished_delta_equals_scheduled": delta["finished"]
                    == delta["scheduled"],
                    "final_running_zero": current["running_tasks"] == 0,
                    "at_least_one_update_pump": counters[
                        "simulation_app_update_pumps"
                    ]
                    >= 1,
                    "local_mesh_cache_not_cleared": True,
                },
                "pass": False,
                "timeout": True,
            }
        if now - last_heartbeat >= 5.0:
            _heartbeat(
                "sdf_cooking_wait",
                elapsed_s=now - start,
                running_tasks=current["running_tasks"],
                scheduled_delta=delta["scheduled"],
                finished_delta=delta["finished"],
                app_update_pumps=counters["simulation_app_update_pumps"],
            )
            last_heartbeat = now
        _pump_app(app, counters)


def _int_to_path(value: int) -> str | None:
    from pxr import PhysicsSchemaTools

    function = getattr(PhysicsSchemaTools, "intToSdfPath", None)
    if function is None:
        return None
    try:
        return str(function(int(value)))
    except Exception:
        return None


def _float3(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _float4_xyzw(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z), float(value.w)]


def _query_properties(
    *,
    app: Any,
    counters: dict[str, int],
    stage_id: int,
    body: str,
) -> dict[str, Any]:
    from omni.physx import get_physx_property_query_interface
    from omni.physx.bindings._physx import (
        PhysxPropertyQueryMode,
        PhysxPropertyQueryResult,
    )
    from pxr import PhysicsSchemaTools

    holder: dict[str, Any] = {
        "rigid_body_callbacks": [],
        "collider_callbacks": [],
        "completed_callback_count": 0,
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
        holder["rigid_body_callbacks"].append(row)
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(
                f"rigid result not VALID: {response.result!r}"
            )

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
        holder["collider_callbacks"].append(row)
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(
                f"collider result not VALID at {row['path']}: {response.result!r}"
            )

    def finished() -> None:
        holder["completed_callback_count"] += 1

    start = time.monotonic()
    get_physx_property_query_interface().query_prim(
        stage_id=stage_id,
        prim_id=PhysicsSchemaTools.sdfPathToInt(LIVE_BODY_PATHS[body]),
        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
        timeout_ms=int(PROPERTY_QUERY_TIMEOUT_S * 1000.0),
        finished_fn=finished,
        rigid_body_fn=rigid,
        collider_fn=collider,
    )
    last_heartbeat = start
    while (
        holder["completed_callback_count"] == 0
        and time.monotonic() - start < PROPERTY_QUERY_TIMEOUT_S
    ):
        _pump_app(app, counters)
        now = time.monotonic()
        if now - last_heartbeat >= 5.0:
            _heartbeat(
                f"property_query_{body}",
                elapsed_s=now - start,
                rigid_callbacks=len(holder["rigid_body_callbacks"]),
                collider_callbacks=len(holder["collider_callbacks"]),
            )
            last_heartbeat = now
    holder["collider_callbacks"].sort(key=lambda row: str(row["path"]))
    paths = [row["path"] for row in holder["collider_callbacks"]]
    all_paths_resolved = all(
        isinstance(path, str) and bool(path) for path in paths
    )
    path_set_sha256 = _path_set_sha(paths) if all_paths_resolved else None
    rigid_path_exact = bool(
        len(holder["rigid_body_callbacks"]) == 1
        and holder["rigid_body_callbacks"][0]["path"]
        == LIVE_BODY_PATHS[body]
    )
    sdf_row = next(
        (
            row
            for row in holder["collider_callbacks"]
            if row["path"] == LIVE_MESH_PATH
        ),
        None,
    )
    finite_sdf = bool(
        sdf_row
        and all(
            math.isfinite(float(value))
            for key in ("aabb_local_min_m", "aabb_local_max_m")
            for value in sdf_row[key]
        )
        and math.isfinite(float(sdf_row["volume_m3"]))
        and float(sdf_row["volume_m3"]) > 0.0
    )
    checks = {
        "rigid_callback_exactly_one": len(holder["rigid_body_callbacks"]) == 1,
        "rigid_body_callback_path_exact": rigid_path_exact,
        "completed_callback_exactly_one": holder["completed_callback_count"] == 1,
        "collider_count_exact": len(holder["collider_callbacks"])
        == EXPECTED_QUERY_COUNTS[body],
        "path_set_hash_exact": path_set_sha256
        == EXPECTED_QUERY_PATH_SHA256[body],
        "all_paths_resolved": all_paths_resolved,
        "all_results_valid": not holder["errors"],
        "sdf_leaf_sanity_when_gripper": body != "gripper_link" or finite_sdf,
    }
    holder.update(
        {
            "body": body,
            "body_path": LIVE_BODY_PATHS[body],
            "elapsed_s": time.monotonic() - start,
            "sorted_path_set_sha256": path_set_sha256,
            "expected_sorted_path_set_sha256": EXPECTED_QUERY_PATH_SHA256[
                body
            ],
            "expected_collider_count": EXPECTED_QUERY_COUNTS[body],
            "enabled_sdf_row": sdf_row,
            "enabled_sdf_row_finite_positive_sanity": finite_sdf,
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    counters["physx_property_queries"] += 1
    return holder


def _post_query_cooking_gate(
    baseline: dict[str, int],
    initial_drain: dict[str, Any],
) -> dict[str, Any]:
    final = _cooking_stats()
    total_delta = {
        "scheduled": final["total_scheduled_tasks"]
        - baseline["total_scheduled_tasks"],
        "finished": final["total_finished_tasks"]
        - baseline["total_finished_tasks"],
        "cache_hits": final["total_finished_cache_hit_tasks"]
        - baseline["total_finished_cache_hit_tasks"],
        "cache_misses": final["total_finished_cache_miss_tasks"]
        - baseline["total_finished_cache_miss_tasks"],
    }
    initial_final = initial_drain["final"]
    checks = {
        "initial_drain_pass": initial_drain.get("pass") is True,
        "scheduled_never_decreased": final["total_scheduled_tasks"]
        >= initial_final["total_scheduled_tasks"],
        "finished_never_decreased": final["total_finished_tasks"]
        >= initial_final["total_finished_tasks"],
        "total_scheduled_delta_positive": total_delta["scheduled"] > 0,
        "total_finished_delta_equals_scheduled": total_delta["finished"]
        == total_delta["scheduled"],
        "post_query_running_zero": final["running_tasks"] == 0,
    }
    return {
        "baseline": baseline,
        "initial_drain_final": initial_final,
        "post_query_final": final,
        "total_delta_from_baseline": total_delta,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _canonical_quaternion_wxyz(values: list[float]) -> list[float]:
    result = [float(value) for value in values]
    for value in result:
        if abs(value) > 1.0e-15:
            if value < 0.0:
                result = [-item for item in result]
            break
    return result


def _live_mass_gate(queries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rows = {}
    for body in ("link5", "gripper_link"):
        callbacks = queries[body]["rigid_body_callbacks"]
        observed = callbacks[0] if len(callbacks) == 1 else {}
        xyzw = observed.get("principal_axes_xyzw", [])
        wxyz = (
            _canonical_quaternion_wxyz(
                [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
            )
            if len(xyzw) == 4
            else []
        )
        normalized = {
            "mass_kg": observed.get("mass_kg"),
            "center_of_mass_m": observed.get("center_of_mass_m"),
            "diagonal_inertia": observed.get("diagonal_inertia"),
            "principal_axes_wxyz": wxyz,
        }
        expected = MASS_BASELINE[body]
        fields = {}
        for field, (absolute, relative) in MASS_TOLERANCES.items():
            actual_array = np.asarray(normalized[field], dtype=np.float64)
            expected_array = np.asarray(expected[field], dtype=np.float64)
            close = bool(
                actual_array.shape == expected_array.shape
                and np.allclose(
                    actual_array,
                    expected_array,
                    rtol=relative,
                    atol=absolute,
                )
            )
            fields[field] = {
                "observed": normalized[field],
                "expected": expected[field],
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "pass": close,
            }
        rows[body] = {
            "fields": fields,
            "callback_xyzw": xyzw,
            "canonical_wxyz": wxyz,
            "pass": all(row["pass"] for row in fields.values()),
        }
    return {"bodies": rows, "pass": all(row["pass"] for row in rows.values())}


def _stagecache_erase(stage: Any, stage_id: int) -> dict[str, Any]:
    from pxr import UsdUtils

    cache = UsdUtils.StageCache.Get()
    before_id = cache.GetId(stage)
    found_before = cache.Find(before_id) if before_id.IsValid() else None
    before = {
        "contains_stage": bool(cache.Contains(stage)),
        "id_valid": bool(before_id.IsValid()),
        "id_int": int(before_id.ToLongInt()) if before_id.IsValid() else None,
        "id_matches_registered_stage_id": bool(
            before_id.IsValid()
            and int(before_id.ToLongInt()) == int(stage_id)
        ),
        "find_old_id_present": found_before is not None,
        "find_old_id_matches_stage": bool(found_before == stage),
    }
    erase_return = bool(cache.Erase(stage))
    after_id = cache.GetId(stage)
    found_after = cache.Find(before_id) if before_id.IsValid() else None
    after = {
        "contains_stage": bool(cache.Contains(stage)),
        "id_valid": bool(after_id.IsValid()),
        "id_int": int(after_id.ToLongInt()) if after_id.IsValid() else None,
        "find_old_id_present": found_after is not None,
    }
    checks = {
        "before_contains_true": before["contains_stage"] is True,
        "before_id_valid": before["id_valid"] is True,
        "before_id_matches_registered": before[
            "id_matches_registered_stage_id"
        ]
        is True,
        "before_find_matches_stage": before["find_old_id_matches_stage"] is True,
        "erase_return_true": erase_return is True,
        "after_contains_false": after["contains_stage"] is False,
        "after_id_invalid": after["id_valid"] is False,
        "after_find_old_id_absent": after["find_old_id_present"] is False,
        "python_stage_reference_retained": stage is not None,
    }
    return {
        "api": "UsdUtils.StageCache.Get().Erase(stage)",
        "before": before,
        "erase_return": erase_return,
        "after": after,
        "python_stage_reference_retained": True,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _counter_gate(counters: dict[str, int]) -> dict[str, Any]:
    exact_checks = {
        key: counters.get(key) == expected
        for key, expected in EXACT_PASS_COUNTERS.items()
    }
    pump_check = (
        1
        <= counters.get("simulation_app_update_pumps", 0)
        <= MAX_APP_UPDATE_PUMPS
    )
    keys_exact = tuple(counters) == EXACT_COUNTER_KEYS
    return {
        "keys_exact_in_registered_order": keys_exact,
        "exact_checks": exact_checks,
        "simulation_app_update_pumps_in_range": pump_check,
        "pass": keys_exact and all(exact_checks.values()) and pump_check,
    }


def _expected_worker_command(args: Any) -> list[str]:
    return [
        str(ISAAC_PYTHON),
        "-B",
        str(WORKER_PATH),
        "--out-dir",
        str(OUT_DIR),
        "--prereg",
        str(PREREG_PATH),
        "--invocation",
        str(INVOCATION_PATH),
        "--controller-pid",
        str(args.controller_pid),
        "--approved-tuple-sha256",
        args.approved_tuple_sha256,
        "--one-shot-nonce",
        args.one_shot_nonce,
        "--headless",
    ]


def _prelaunch_authority_gate(args: Any) -> dict[str, Any]:
    required = (
        PREREG_PATH,
        ATTESTATION_PATH,
        TUPLE_PATH,
        RUNTIME_MANIFEST_PATH,
        INVOCATION_PATH,
        CONTROLLER_PATH,
        WORKER_PATH,
    )
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"D400 prelaunch authority files missing: {missing}"
        )
    invocation_sha_from_controller = os.environ.get(
        "D400_INVOCATION_SHA256"
    )
    invocation_sha_current = _sha(INVOCATION_PATH)
    invocation = _read_json(INVOCATION_PATH)
    manifest = _read_json(RUNTIME_MANIFEST_PATH)
    tuple_value = _read_json(TUPLE_PATH)
    expected_command = _expected_worker_command(args)
    current_hashes = {
        "preregistration_sha256": _sha(PREREG_PATH),
        "reviewed_script_attestation_sha256": _sha(ATTESTATION_PATH),
        "controller_script_sha256": _sha(CONTROLLER_PATH),
        "worker_script_sha256": _sha(WORKER_PATH),
    }
    parent_pid = os.getppid()
    parent_cwd = None
    parent_cmdline: list[str] = []
    parent_probe_error = None
    try:
        parent_cwd = str(
            Path(f"/proc/{parent_pid}/cwd").resolve(strict=True)
        )
        parent_cmdline = [
            item.decode("utf-8", errors="strict")
            for item in Path(
                f"/proc/{parent_pid}/cmdline"
            ).read_bytes().split(b"\0")
            if item
        ]
    except Exception as error:
        parent_probe_error = f"{type(error).__name__}: {error}"
    parent_controller_tokens = []
    if parent_cwd is not None:
        for token in parent_cmdline:
            if Path(token).name != CONTROLLER_PATH.name:
                continue
            token_path = Path(token)
            if not token_path.is_absolute():
                token_path = Path(parent_cwd) / token_path
            parent_controller_tokens.append(
                str(token_path.resolve(strict=False))
            )
    parent_approval_bound = any(
        left == "--approved-tuple-sha256"
        and right == args.approved_tuple_sha256
        for left, right in zip(parent_cmdline, parent_cmdline[1:])
    )
    bindings = invocation.get("runtime_authority_bindings", {})
    manifest_authority = manifest.get(
        "worker_launch_authority", {}
    )
    checks = {
        "out_dir_exact": args.out_dir.resolve() == OUT_DIR.resolve(),
        "prereg_path_exact": args.prereg.resolve()
        == PREREG_PATH.resolve(),
        "invocation_path_exact": args.invocation.resolve()
        == INVOCATION_PATH.resolve(),
        "controller_pid_positive_integer": (
            type(args.controller_pid) is int and args.controller_pid > 0
        ),
        "direct_parent_is_registered_controller_pid": (
            parent_pid == args.controller_pid
        ),
        "parent_proc_probe_succeeded": parent_probe_error is None,
        "worker_cwd_exact_repo": Path.cwd().resolve()
        == REPO.resolve(),
        "parent_cmdline_contains_exact_controller": (
            str(CONTROLLER_PATH.resolve())
            in parent_controller_tokens
        ),
        "parent_cmdline_contains_approved_tuple": parent_approval_bound,
        "nonce_is_lowercase_hex_64": (
            isinstance(args.one_shot_nonce, str)
            and len(args.one_shot_nonce) == 64
            and all(
                character in "0123456789abcdef"
                for character in args.one_shot_nonce
            )
        ),
        "tuple_field_order_exact": tuple(tuple_value) == TUPLE_FIELDS,
        "tuple_member_hashes_current": tuple_value == current_hashes,
        "approved_tuple_file_sha_exact": (
            args.approved_tuple_sha256 == _sha(TUPLE_PATH)
        ),
        "preregistration_sha_exact": (
            current_hashes["preregistration_sha256"]
            == EXPECTED_PREREG_SHA256
        ),
        "invocation_digest_environment_present": (
            isinstance(invocation_sha_from_controller, str)
            and len(invocation_sha_from_controller) == 64
        ),
        "invocation_digest_environment_exact": (
            invocation_sha_from_controller
            == invocation_sha_current
        ),
        "invocation_artifact_exact": invocation.get("artifact")
        == "D400_SINGLE_WORKER_SPAWN_REQUEST_V1",
        "invocation_command_exact": invocation.get("command")
        == expected_command,
        "actual_sys_argv_exact": sys.argv == expected_command[2:],
        "invocation_cwd_exact": invocation.get("cwd")
        == str(REPO),
        "invocation_controller_pid_exact": invocation.get(
            "controller_pid"
        )
        == args.controller_pid,
        "invocation_nonce_exact": invocation.get("one_shot_nonce")
        == args.one_shot_nonce,
        "invocation_tuple_sha_exact": invocation.get(
            "approved_tuple_sha256"
        )
        == args.approved_tuple_sha256,
        "invocation_prereg_hash_exact": invocation.get(
            "preregistration_sha256"
        )
        == current_hashes["preregistration_sha256"],
        "invocation_controller_hash_exact": invocation.get(
            "controller_sha256"
        )
        == current_hashes["controller_script_sha256"],
        "invocation_worker_hash_exact": invocation.get(
            "worker_sha256"
        )
        == current_hashes["worker_script_sha256"],
        "invocation_one_spawn_no_retry_exact": (
            invocation.get("worker_spawn_request_budget") == 1
            and invocation.get(
                "actual_worker_invocations_before_popen"
            )
            == 0
            and invocation.get("automatic_retries") == 0
        ),
        "manifest_pass_true": manifest.get("pass") is True,
        "manifest_command_exact": manifest.get("worker_command")
        == expected_command,
        "manifest_output_root_exact": manifest.get("output_root")
        == _rel(OUT_DIR),
        "manifest_controller_pid_exact": manifest_authority.get(
            "controller_pid"
        )
        == args.controller_pid,
        "manifest_tuple_sha_exact": manifest_authority.get(
            "approved_tuple_sha256"
        )
        == args.approved_tuple_sha256,
        "manifest_nonce_exact": manifest_authority.get(
            "one_shot_nonce"
        )
        == args.one_shot_nonce,
        "invocation_manifest_hash_exact": bindings.get(
            "runtime_manifest_sha256"
        )
        == _sha(RUNTIME_MANIFEST_PATH),
        "invocation_tuple_members_exact": bindings.get(
            "approved_tuple_members"
        )
        == tuple_value,
        "invocation_tuple_binding_exact": bindings.get(
            "approved_tuple_sha256"
        )
        == args.approved_tuple_sha256,
    }
    record = {
        "artifact": "D400_WORKER_PRELAUNCH_CONTROLLER_AUTHORITY_GATE_V1",
        "parent_pid": parent_pid,
        "parent_cwd": parent_cwd,
        "parent_cmdline": parent_cmdline,
        "parent_probe_error": parent_probe_error,
        "invocation_sha256_from_controller_environment": (
            invocation_sha_from_controller
        ),
        "invocation_sha256_current": invocation_sha_current,
        "runtime_manifest_sha256_current": _sha(
            RUNTIME_MANIFEST_PATH
        ),
        "approved_tuple_sha256_current": _sha(TUPLE_PATH),
        "current_tuple_member_hashes": current_hashes,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not record["pass"]:
        raise RuntimeError(
            "D400 worker refused pre-AppLauncher direct or unbound "
            f"invocation: {checks}"
        )
    return record


def _exclusive_out_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise RuntimeError("D400 output directory must remain inside repo") from error
    if not resolved.is_dir():
        raise RuntimeError("D400 controller must create/use the registered output dir")
    owned = [
        resolved / CLAIM_NAME,
        resolved / RAW_SUMMARY_NAME,
        resolved / PRECLOSE_NAME,
    ]
    existing = [_rel(path) for path in owned if path.exists()]
    if existing:
        raise RuntimeError(f"D400 worker-owned outputs already exist: {existing}")
    return resolved


def _execute(app: Any, out_dir: Path, prereg_path: Path) -> dict[str, Any]:
    from omni.physx import get_physx_simulation_interface

    counters = _counter_template()
    result: dict[str, Any] = {
        "artifact": "D400_GRIPPER_LINK_SDF_RES256_WORKER_RAW_V1",
        "case": "g0a_d400",
        "preregistration_path": _rel(prereg_path),
        "preregistration_sha256": _sha(prereg_path),
        "counters": counters,
        "exception": None,
        "worker_protocol_pass": False,
        "scientific_or_physics_verdict": None,
        "g0a_pass": False,
    }
    stage = None
    stage_id = None
    attached = False
    detached = False
    erase_record: dict[str, Any] | None = None
    try:
        if _sha(prereg_path) != EXPECTED_PREREG_SHA256:
            raise RuntimeError("D400 preregistration hash drift")
        prereg = _read_json(prereg_path)
        if prereg.get("status") != "PREREGISTERED_NOT_EXECUTED":
            raise RuntimeError("D400 preregistration status is not frozen")
        runtime_stack = _runtime_stack_probe()
        result["runtime_stack_probe"] = runtime_stack
        if not runtime_stack["pass"]:
            raise RuntimeError(
                "D400 launched runtime stack probe failed: "
                f"{runtime_stack['checks']}"
            )
        derivative_dir, derivative_record = _author_sdf_derivative(
            out_dir, counters
        )
        result["derivative_asset"] = derivative_record

        stage, stage_id, live_stage_record = _make_live_stage(
            derivative_dir / "roarm_m3.usd"
        )
        result["live_stage"] = live_stage_record
        _phase(
            out_dir,
            "live_stage_create_end",
            stage_id=stage_id,
            root_reference=live_stage_record["root_reference"],
        )
        inventory = _live_inventory(stage)
        result["live_inventory"] = inventory
        _phase(
            out_dir,
            "live_owner_inventory_gate_end",
            passed=inventory["pass"],
            link5_active=len(inventory["enabled_paths"]["link5"]),
            gripper_active=len(inventory["enabled_paths"]["gripper_link"]),
        )
        if not inventory["pass"]:
            raise RuntimeError(f"D400 live inventory gate failed: {inventory['checks']}")

        timeline_before = _timeline_tuple()
        result["timeline_before_attach"] = _timeline_gate(timeline_before)
        counters["timeline_raw_stop_time_zero_checks"] += 1
        if not result["timeline_before_attach"]["pass"]:
            raise RuntimeError(
                f"D400 pre-attach timeline mismatch: {timeline_before}"
            )

        _phase(out_dir, "cooking_baseline_capture_start")
        cooking_baseline = _cooking_stats()
        result["cooking_baseline"] = cooking_baseline
        _phase(
            out_dir,
            "cooking_baseline_capture_end",
            **cooking_baseline,
        )
        if cooking_baseline["running_tasks"] != 0:
            raise RuntimeError(
                f"D400 cooking baseline has running tasks: {cooking_baseline}"
            )

        simulation = get_physx_simulation_interface()
        _phase(
            out_dir,
            "physx_stage_attach_start",
            stage_id=stage_id,
            case_output_path=_rel(out_dir),
            link5_rigid_prim=LIVE_BODY_PATHS["link5"],
            gripper_rigid_prim=LIVE_BODY_PATHS["gripper_link"],
            sdf_mesh_prim=LIVE_MESH_PATH,
        )
        attached = bool(simulation.attach_stage(stage_id))
        counters["physx_stage_attaches"] += 1
        result["physx_stage_attach_return"] = attached
        _phase(out_dir, "physx_stage_attach_end", attached=attached)
        if not attached:
            raise RuntimeError("D400 PhysX attach_stage returned false")

        _phase(out_dir, "sdf_cooking_wait_start")
        cooking = _wait_for_cooking(app, counters, cooking_baseline)
        result["cooking"] = cooking
        _phase(
            out_dir,
            "sdf_cooking_wait_end",
            passed=cooking["pass"],
            scheduled_delta=cooking["delta"]["scheduled"],
            finished_delta=cooking["delta"]["finished"],
            running_tasks=cooking["final"]["running_tasks"],
        )
        if not cooking["pass"]:
            raise RuntimeError(f"D400 SDF cooking drain gate failed: {cooking}")

        queries = {}
        link5_query = _query_properties(
            app=app,
            counters=counters,
            stage_id=stage_id,
            body="link5",
        )
        queries["link5"] = link5_query
        _phase(
            out_dir,
            "property_query_link5_end",
            passed=link5_query["pass"],
            collider_count=len(link5_query["collider_callbacks"]),
            completed_callback_count=link5_query["completed_callback_count"],
            path_set_sha256=link5_query["sorted_path_set_sha256"],
        )
        if not link5_query["pass"]:
            raise RuntimeError(
                f"D400 property query failed for link5: {link5_query['checks']}"
            )
        gripper_query = _query_properties(
            app=app,
            counters=counters,
            stage_id=stage_id,
            body="gripper_link",
        )
        queries["gripper_link"] = gripper_query
        _phase(
            out_dir,
            "property_query_gripper_link_end",
            passed=gripper_query["pass"],
            collider_count=len(gripper_query["collider_callbacks"]),
            completed_callback_count=gripper_query["completed_callback_count"],
            path_set_sha256=gripper_query["sorted_path_set_sha256"],
        )
        if not gripper_query["pass"]:
            raise RuntimeError(
                "D400 property query failed for gripper_link: "
                f"{gripper_query['checks']}"
            )
        result["property_queries"] = queries
        post_query_cooking = _post_query_cooking_gate(
            cooking_baseline, cooking
        )
        result["post_query_cooking_gate"] = post_query_cooking
        if not post_query_cooking["pass"]:
            raise RuntimeError(
                "D400 post-query global cooking queue did not remain drained: "
                f"{post_query_cooking['checks']}"
            )
        live_mass = _live_mass_gate(queries)
        result["live_mass_com_inertia_gate"] = live_mass
        _phase(
            out_dir,
            "mass_invariance_gate_end",
            authored_pass=result["derivative_asset"]["authored_mass_gate"]["pass"],
            live_pass=live_mass["pass"],
            post_query_cook_pass=post_query_cooking["pass"],
        )
        if not live_mass["pass"]:
            raise RuntimeError("D400 live mass/COM/inertia invariance gate failed")

        owner_evidence = _inspection_geometry_evidence(
            base_root=BASE_ROOT_USD,
            live_stage=stage,
            inventory=inventory,
        )
        if not owner_evidence["pass"]:
            raise RuntimeError(
                f"D400 inspection geometry evidence failed: {owner_evidence['checks']}"
            )
        _write_json_x(out_dir / OWNER_EVIDENCE_NAME, owner_evidence)
        result["owner_evidence"] = {
            "path": _rel(out_dir / OWNER_EVIDENCE_NAME),
            "sha256": _sha(out_dir / OWNER_EVIDENCE_NAME),
            "pass": True,
        }

        timeline_after = _timeline_tuple()
        result["timeline_after_queries"] = _timeline_gate(timeline_after)
        counters["timeline_raw_stop_time_zero_checks"] += 1
        if not result["timeline_after_queries"]["pass"]:
            raise RuntimeError(
                f"D400 post-query timeline mismatch: {timeline_after}"
            )
    except Exception as error:
        result["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        _phase(out_dir, "cleanup_start", attached=attached)
        if attached:
            _phase(out_dir, "physx_stage_detach_start", stage_id=stage_id)
            try:
                get_physx_simulation_interface().detach_stage()
                counters["physx_stage_detaches"] += 1
                detached = True
                _phase(
                    out_dir,
                    "physx_stage_detach_end",
                    stage_id=stage_id,
                    detached=True,
                )
            except Exception as error:
                result["detach_exception"] = (
                    f"{type(error).__name__}: {error}"
                )
                _phase(
                    out_dir,
                    "physx_stage_detach_end",
                    stage_id=stage_id,
                    detached=False,
                    error=result["detach_exception"],
                )
        if stage is not None and stage_id is not None:
            try:
                counters["stagecache_erase_calls"] += 1
                erase_record = _stagecache_erase(stage, stage_id)
                result["stagecache_erase"] = erase_record
                _phase(
                    out_dir,
                    "stagecache_erase_end",
                    passed=erase_record["pass"],
                    erase_return=erase_record["erase_return"],
                    after_contains=erase_record["after"]["contains_stage"],
                )
            except Exception as error:
                result["stagecache_erase_exception"] = (
                    f"{type(error).__name__}: {error}"
                )
        safe_to_close = bool(
            attached
            and detached
            and erase_record is not None
            and erase_record.get("pass") is True
        )
        result["safe_to_close_app"] = safe_to_close
        _phase(
            out_dir,
            "safe_to_close_app",
            value=safe_to_close,
            detached=detached,
            stagecache_nonmembership=bool(
                erase_record and erase_record.get("pass")
            ),
        )
        counter_gate = _counter_gate(counters)
        result["counter_gate"] = counter_gate
        result["worker_protocol_pass"] = bool(
            result["exception"] is None
            and result.get("runtime_stack_probe", {}).get("pass") is True
            and result.get("derivative_asset", {}).get("pass") is True
            and result.get("live_inventory", {}).get("pass") is True
            and result.get("timeline_before_attach", {}).get("pass") is True
            and result.get("physx_stage_attach_return") is True
            and result.get("cooking", {}).get("pass") is True
            and all(
                query.get("pass") is True
                for query in result.get("property_queries", {}).values()
            )
            and result.get("post_query_cooking_gate", {}).get("pass") is True
            and result.get("live_mass_com_inertia_gate", {}).get("pass") is True
            and result.get("owner_evidence", {}).get("pass") is True
            and result.get("timeline_after_queries", {}).get("pass") is True
            and safe_to_close
            and counter_gate["pass"]
        )
        _write_json_x(out_dir / RAW_SUMMARY_NAME, result)
        _phase(
            out_dir,
            "worker_raw_summary_written",
            sha256=_sha(out_dir / RAW_SUMMARY_NAME),
            worker_protocol_pass=result["worker_protocol_pass"],
        )
        phase_payload = (out_dir / PHASE_NAME).read_bytes()
        preclose = {
            "artifact": "D400_WORKER_PRECLOSE_SENTINEL_V1",
            "raw_summary_path": _rel(out_dir / RAW_SUMMARY_NAME),
            "summary_sha256": _sha(out_dir / RAW_SUMMARY_NAME),
            "counters": dict(counters),
            "counter_gate": counter_gate,
            "stagecache_erase": erase_record,
            "safe_to_close_app": safe_to_close,
            "phase_prefix_bytes": len(phase_payload),
            "phase_prefix_sha256": _sha_bytes(phase_payload),
            "worker_protocol_pass": result["worker_protocol_pass"],
            "scientific_or_physics_verdict": None,
            "g0a_pass": False,
        }
        _write_json_x(out_dir / PRECLOSE_NAME, preclose)
        _phase(
            out_dir,
            "worker_preclose_sentinel_written",
            sha256=_sha(out_dir / PRECLOSE_NAME),
            worker_protocol_pass=result["worker_protocol_pass"],
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument(
        "--approved-tuple-sha256", required=True
    )
    parser.add_argument("--one-shot-nonce", required=True)
    prelaunch_args, _unknown = parser.parse_known_args()
    authority_gate = _prelaunch_authority_gate(prelaunch_args)
    out_dir = _exclusive_out_dir(prelaunch_args.out_dir)
    _write_json_x(
        out_dir / CLAIM_NAME,
        {
            "artifact": "D400_SINGLE_WORKER_EXCLUSIVE_CLAIM_V1",
            "pid": os.getpid(),
            "parent_controller_pid": os.getppid(),
            "monotonic_ns": time.monotonic_ns(),
            "worker_invocation_count": 1,
            "automatic_retry_count": 0,
            "headless": True,
            "prelaunch_controller_authority_gate": authority_gate,
        },
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    prereg = args.prereg.resolve()
    if prereg != PREREG_PATH.resolve() or not prereg.is_file():
        raise RuntimeError("D400 worker received a non-registered preregistration")
    launcher = None
    _phase(out_dir, "simulation_app_launch_start", headless=True)
    try:
        launcher = AppLauncher(args)
        _phase(out_dir, "simulation_app_launch_end", headless=True)
        result = _execute(launcher.app, out_dir, prereg)
        print(
            json.dumps(
                {
                    "artifact": "D400_WORKER_EXIT",
                    "raw_summary": _rel(out_dir / RAW_SUMMARY_NAME),
                    "worker_protocol_pass": result["worker_protocol_pass"],
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
            _phase(out_dir, "simulation_app_close_start")
            launcher.app.close()
            _phase(out_dir, "simulation_app_close_returned")


if __name__ == "__main__":
    raise SystemExit(main())
