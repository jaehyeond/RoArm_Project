#!/usr/bin/env python3
"""D334 frozen-pose live collision shape / ownership audit for grasp G0a.

Zero new physical variables. Physics steps are exact replays of D333: the
200-step sole-support baseline (hard gate re-applied) plus one target step.
The audit adds live rigid-body collider enumeration (PhysX property query),
per-collider cook parity gates, non-AABB signed distances of cooked and raw
tool shapes against the analytic cylinder at the frozen pre-step and replayed
post-step-0 poses, and mapping of the recorded D333 gripper contact point to
candidate shapes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, log_rerun
from sim_scripts import cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332
from sim_scripts import cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333

DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d334"
D333_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d333"
D333_SUMMARY = D333_DIR / "g0a_d333_sole_support_static_summary.json"
D333_TARGET_CSV = D333_DIR / "d333_teleport_settle_trace.csv"

BODY_LABELS = ("link5", "gripper_link")
BODY_PATHS = {
    "link5": "/World/envs/env_0/Robot/link5",
    "gripper_link": "/World/envs/env_0/Robot/gripper_link",
}

VOLUME_PARITY_REL_TOL = 0.005
AABB_PARITY_TOL_M = 5.0e-4
REPLAY_OBJECT_POS_TOL_M = 5.0e-5
REPLAY_TCP_TOL_M = 5.0e-5
REPLAY_FORCE_REL_TOL = 0.01
REPLAY_CONTACT_POINT_TOL_M = 1.0e-3
CONTACT_ON_SURFACE_TOL_M = 1.0e-3
STATE_GUARD_ATOL = 1.0e-12
QUERY_TIMEOUT_S = 30.0
POINT_PROBE_RADIUS_M = 1.0e-6


def _rel(path: Path) -> str:
    return d332._rel(path)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    d332._json_dump(path, payload)


def _load_d333_row0() -> dict[str, float | str]:
    with D333_TARGET_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        values = next(reader)
    row: dict[str, float | str] = {}
    for key, value in zip(columns, values, strict=True):
        try:
            row[key] = float(value)
        except ValueError:
            row[key] = value
    return row


def _snapshot_sim_state(inner: Any) -> dict[str, np.ndarray]:
    return {
        "joint_pos": inner._robot.data.joint_pos.detach().cpu().numpy().astype(np.float64),
        "joint_vel": inner._robot.data.joint_vel.detach().cpu().numpy().astype(np.float64),
        "object_root": inner._sponge.data.root_state_w.detach().cpu().numpy().astype(np.float64),
    }


def _state_guard(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, Any]:
    deltas = {
        key: float(np.max(np.abs(after[key] - before[key]))) if after[key].size else 0.0
        for key in before
    }
    max_delta = max(deltas.values()) if deltas else 0.0
    return {
        "max_abs_delta": max_delta,
        "per_field_max_abs_delta": deltas,
        "tolerance": STATE_GUARD_ATOL,
        "violated": bool(max_delta > STATE_GUARD_ATOL),
    }


def _int_to_sdf_path(value: int) -> str | None:
    from pxr import PhysicsSchemaTools

    fn = getattr(PhysicsSchemaTools, "intToSdfPath", None)
    if fn is None:
        return None
    try:
        return str(fn(int(value)))
    except Exception:
        return None


def _stage_id(inner: Any) -> int:
    from pxr import UsdUtils

    return UsdUtils.StageCache.Get().GetId(inner.scene.stage).ToLongInt()


def _property_query_body(inner: Any, simulation_app: Any, body_label: str) -> dict[str, Any]:
    from omni.physx import get_physx_property_query_interface
    from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
    from pxr import PhysicsSchemaTools

    body_path = BODY_PATHS[body_label]
    holder: dict[str, Any] = {"finished": False, "rigid_body": None, "colliders": [], "errors": []}

    def _float3(value: Any) -> list[float]:
        return [float(value.x), float(value.y), float(value.z)]

    def _rigid_body_fn(response: Any) -> None:
        row = {"result": str(response.result)}
        for name in ("mass", "type"):
            if hasattr(response, name):
                value = getattr(response, name)
                row[name] = float(value) if name == "mass" else str(value)
        holder["rigid_body"] = row
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(f"rigid body query result {response.result}")

    def _collider_fn(response: Any) -> None:
        quat = response.local_rot
        row = {
            "result": str(response.result),
            "path_id": int(response.path_id),
            "path": _int_to_sdf_path(response.path_id),
            "local_pos_m": _float3(response.local_pos),
            "local_rot_xyzw": [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
            "aabb_local_min_m": _float3(response.aabb_local_min),
            "aabb_local_max_m": _float3(response.aabb_local_max),
            "volume_m3": float(response.volume),
        }
        holder["colliders"].append(row)
        if response.result != PhysxPropertyQueryResult.VALID:
            holder["errors"].append(f"collider query result {response.result} for {row['path']}")

    def _finished_fn() -> None:
        holder["finished"] = True

    guard_before = _snapshot_sim_state(inner)
    get_physx_property_query_interface().query_prim(
        stage_id=_stage_id(inner),
        prim_id=PhysicsSchemaTools.sdfPathToInt(body_path),
        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
        timeout_ms=int(QUERY_TIMEOUT_S * 1000.0),
        finished_fn=_finished_fn,
        rigid_body_fn=_rigid_body_fn,
        collider_fn=_collider_fn,
    )
    start = time.monotonic()
    deadline = start + QUERY_TIMEOUT_S
    pump_after = start + 2.0
    pumped = 0
    # Kit app updates would advance physics while the timeline plays; disable
    # simulation playback for the whole wait so pumps cannot step the scene.
    inner.sim.set_setting("/app/player/playSimulations", False)
    try:
        while not holder["finished"] and time.monotonic() < deadline:
            time.sleep(0.005)
            if not holder["finished"] and time.monotonic() >= pump_after and pumped < 400:
                simulation_app.update()
                pumped += 1
    finally:
        inner.sim.set_setting("/app/player/playSimulations", True)
    guard = _state_guard(guard_before, _snapshot_sim_state(inner))
    return {
        "body": body_label,
        "body_path": body_path,
        "finished": bool(holder["finished"]),
        "rigid_body": holder["rigid_body"],
        "colliders": holder["colliders"],
        "app_update_pumps": pumped,
        "state_guard": guard,
        "errors": holder["errors"],
        "pass": bool(holder["finished"] and not holder["errors"] and holder["colliders"]),
    }


def _usd_collision_inventory(inner: Any, body_label: str) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    body_path = BODY_PATHS[body_label]
    rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if path != body_path and not path.startswith(body_path + "/"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        approximation = None
        api_prim = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            api_prim = prim
        mesh_prims = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        if api_prim is None:
            for mesh in mesh_prims:
                if mesh.HasAPI(UsdPhysics.MeshCollisionAPI):
                    api_prim = mesh
                    break
        if api_prim is not None:
            approximation = UsdPhysics.MeshCollisionAPI(api_prim).GetApproximationAttr().Get()
        ancestor = prim
        owner = None
        while ancestor.IsValid():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = ancestor.GetPath().pathString
                break
            ancestor = ancestor.GetParent()
        rows.append(
            {
                "path": path,
                "type_name": str(prim.GetTypeName()),
                "collision_enabled": True if enabled is None else bool(enabled),
                "is_instance_proxy": bool(prim.IsInstanceProxy()),
                "approximation": None if approximation is None else str(approximation),
                "mesh_prim_paths": [p.GetPath().pathString for p in mesh_prims],
                "nearest_rigid_body_ancestor": owner,
            }
        )
    return rows


def _paths_related(a: str, b: str) -> bool:
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _ownership_parity(
    physx_rows: dict[str, dict[str, Any]], usd_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    per_body: dict[str, Any] = {}
    cross: list[dict[str, Any]] = []
    for body in BODY_LABELS:
        physx_paths = [row["path"] for row in physx_rows[body]["colliders"] if row["path"]]
        usd_paths = [row["path"] for row in usd_rows[body] if row["collision_enabled"]]
        matched = []
        physx_unmatched = list(physx_paths)
        usd_unmatched = list(usd_paths)
        for p_path in physx_paths:
            for u_path in usd_paths:
                if _paths_related(p_path, u_path):
                    matched.append({"physx": p_path, "usd": u_path})
                    if p_path in physx_unmatched:
                        physx_unmatched.remove(p_path)
                    if u_path in usd_unmatched:
                        usd_unmatched.remove(u_path)
                    break
        for other in BODY_LABELS:
            if other == body:
                continue
            for p_path in physx_paths:
                if _paths_related(p_path, BODY_PATHS[other]):
                    cross.append({"attached_to_body": body, "collider_path": p_path, "path_under": other})
        per_body[body] = {
            "physx_collider_paths": physx_paths,
            "usd_enabled_collision_paths": usd_paths,
            "matched": matched,
            "physx_unmatched": physx_unmatched,
            "usd_unmatched": usd_unmatched,
            "parity_pass": bool(not physx_unmatched and not usd_unmatched and matched),
        }
    return {
        "per_body": per_body,
        "cross_body_attachments": cross,
        "parity_pass_all": bool(all(per_body[b]["parity_pass"] for b in BODY_LABELS)),
    }


def _triangulate(face_counts: list[int], face_indices: list[int]) -> tuple[np.ndarray, bool]:
    triangles: list[list[int]] = []
    cursor = 0
    fan_used = False
    for count in face_counts:
        poly = face_indices[cursor : cursor + count]
        cursor += count
        if count == 3:
            triangles.append(list(poly))
        else:
            fan_used = True
            for k in range(1, count - 1):
                triangles.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(triangles, dtype=np.int64), fan_used


def _source_mesh_body_local(inner: Any, usd_row: dict[str, Any], body_label: str) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom

    stage = inner.scene.stage
    mesh_paths = usd_row["mesh_prim_paths"]
    if len(mesh_paths) != 1:
        raise RuntimeError(f"expected one mesh under {usd_row['path']}, got {mesh_paths}")
    mesh_prim = stage.GetPrimAtPath(mesh_paths[0])
    mesh_geom = UsdGeom.Mesh(mesh_prim)
    points = list(mesh_geom.GetPointsAttr().Get() or [])
    face_counts = [int(v) for v in list(mesh_geom.GetFaceVertexCountsAttr().Get() or [])]
    face_indices = [int(v) for v in list(mesh_geom.GetFaceVertexIndicesAttr().Get() or [])]
    if not points or not face_counts or not face_indices:
        raise RuntimeError(f"empty collision mesh topology at {mesh_paths[0]}")
    mesh_l2w = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_prim = stage.GetPrimAtPath(BODY_PATHS[body_label])
    body_w2l = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    verts_body = np.asarray(
        [
            [float(v) for v in body_w2l.Transform(mesh_l2w.Transform(Gf.Vec3d(*[float(x) for x in p])))]
            for p in points
        ],
        dtype=np.float64,
    )
    triangles, fan_used = _triangulate(face_counts, face_indices)
    return {
        "mesh_prim_path": mesh_paths[0],
        "vertex_count": int(len(verts_body)),
        "face_count": int(len(face_counts)),
        "triangle_count": int(len(triangles)),
        "fan_triangulated": fan_used,
        "body_local_bounds_m": np.vstack([verts_body.min(axis=0), verts_body.max(axis=0)]).tolist(),
        "_verts_body": verts_body,
        "_triangles": triangles,
        "_face_counts": face_counts,
        "_face_indices": face_indices,
    }


def _request_cook(inner: Any, prim_id: int) -> dict[str, Any]:
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    holder: dict[str, Any] = {}

    def _on_result(result: Any, convexes: list[Any]) -> None:
        holder["result"] = result
        holder["convexes"] = list(convexes)

    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=_stage_id(inner),
        collision_prim_id=prim_id,
        run_asynchronously=False,
        on_result=_on_result,
    )
    valid = holder.get("result") == PhysxCollisionRepresentationResult.RESULT_VALID
    return {"holder": holder, "valid": bool(valid)}


def _convex_to_rows(convex: Any) -> dict[str, Any]:
    vertices = np.asarray(
        [[float(v.x), float(v.y), float(v.z)] for v in convex.vertices], dtype=np.float64
    )
    indices = [int(i) for i in convex.indices]
    polygons = []
    for polygon in convex.polygons:
        start = int(polygon.index_base)
        count = int(polygon.num_vertices)
        polygons.append(indices[start : start + count])
    return {"vertices": vertices, "polygons": polygons}


def _attempt_direct_live_cook(inner: Any, usd_row: dict[str, Any]) -> dict[str, Any]:
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom

    stage = inner.scene.stage
    attempts = []
    chosen = None
    for candidate in [usd_row["path"], *usd_row["mesh_prim_paths"]]:
        record: dict[str, Any] = {"prim_path": candidate}
        try:
            outcome = _request_cook(inner, PhysicsSchemaTools.sdfPathToInt(candidate))
            record["result"] = str(outcome["holder"].get("result"))
            convexes = outcome["holder"].get("convexes", [])
            record["convex_count"] = len(convexes)
            record["valid"] = bool(outcome["valid"] and len(convexes) == 1)
            if record["valid"] and chosen is None:
                rows = _convex_to_rows(convexes[0])
                prim = stage.GetPrimAtPath(candidate)
                l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                body_prim = stage.GetPrimAtPath(usd_row["_body_path"])
                w2l = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ).GetInverse()
                verts_body = np.asarray(
                    [
                        [float(x) for x in w2l.Transform(l2w.Transform(Gf.Vec3d(*v)))]
                        for v in rows["vertices"].tolist()
                    ],
                    dtype=np.float64,
                )
                chosen = {"prim_path": candidate, "vertices_body": verts_body, "polygons": rows["polygons"]}
        except Exception as error:  # noqa: BLE001 - recorded as evidence
            record["valid"] = False
            record["exception"] = f"{type(error).__name__}: {error}"
        attempts.append(record)
    return {"attempts": attempts, "chosen": chosen}


def _mirror_cook(inner: Any, verts_body: np.ndarray, triangles: np.ndarray, tag: str) -> dict[str, Any]:
    from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    mirror_root = "/World/D334CookMirror"
    mirror_path = f"{mirror_root}/{tag}"
    if stage.GetPrimAtPath(mirror_root).IsValid():
        stage.RemovePrim(mirror_root)
    mirror = UsdGeom.Mesh.Define(stage, mirror_path)
    mirror.CreatePointsAttr([Gf.Vec3f(*[float(x) for x in vert]) for vert in verts_body])
    mirror.CreateFaceVertexCountsAttr([3] * int(len(triangles)))
    mirror.CreateFaceVertexIndicesAttr([int(i) for i in triangles.reshape(-1)])
    mirror.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mirror.GetPrim())
    api = UsdPhysics.MeshCollisionAPI.Apply(mirror.GetPrim())
    api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    try:
        outcome = _request_cook(inner, PhysicsSchemaTools.sdfPathToInt(mirror_path))
    finally:
        stage.RemovePrim(mirror_root)
    if not outcome["valid"]:
        raise RuntimeError(f"mirror cook failed for {tag}: {outcome['holder']}")
    convexes = outcome["holder"]["convexes"]
    if len(convexes) != 1:
        raise RuntimeError(f"mirror cook returned {len(convexes)} convex parts for {tag}")
    rows = _convex_to_rows(convexes[0])
    return {
        "vertices_body": rows["vertices"],
        "polygons": rows["polygons"],
        "mirror_removed": bool(not stage.GetPrimAtPath(mirror_root).IsValid()),
    }


def _aabb_extent_comparison(physx_row: dict[str, Any], cooked_verts_body: np.ndarray) -> dict[str, Any]:
    """Informational only — AABB is never gate evidence (frame/scale conventions
    of the property-query AABB are recorded raw so they can be audited post-run)."""
    lo = np.asarray(physx_row["aabb_local_min_m"], dtype=np.float64)
    hi = np.asarray(physx_row["aabb_local_max_m"], dtype=np.float64)
    physx_extents = hi - lo
    qx, qy, qz, qw = physx_row["local_rot_xyzw"]
    rot = d332._quat_wxyz_to_rot(np.asarray([qw, qx, qy, qz], dtype=np.float64))
    pos = np.asarray(physx_row["local_pos_m"], dtype=np.float64)
    verts_shape = (rot.T @ (cooked_verts_body - pos).T).T
    cooked_extents = verts_shape.max(axis=0) - verts_shape.min(axis=0)
    per_axis = np.abs(physx_extents - cooked_extents)
    sorted_delta = np.abs(np.sort(physx_extents) - np.sort(cooked_extents))
    return {
        "physx_aabb_local_min_m": lo.tolist(),
        "physx_aabb_local_max_m": hi.tolist(),
        "physx_extents_m": physx_extents.tolist(),
        "cooked_extents_shape_frame_m": cooked_extents.tolist(),
        "per_axis_abs_delta_m": per_axis.tolist(),
        "sorted_extent_abs_delta_m": sorted_delta.tolist(),
        "per_axis_within_0p5mm": bool(np.max(per_axis) <= AABB_PARITY_TOL_M),
        "sorted_within_0p5mm": bool(np.max(sorted_delta) <= AABB_PARITY_TOL_M),
        "note": "informational corroboration only - never gate evidence",
    }


def _build_shape_records(
    inner: Any, physx_inventory: dict[str, dict[str, Any]], usd_inventory: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    import hppfcl
    from scipy.spatial import ConvexHull

    shapes: list[dict[str, Any]] = []
    for body in BODY_LABELS:
        for usd_row in usd_inventory[body]:
            if not usd_row["collision_enabled"]:
                continue
            usd_row = dict(usd_row)
            usd_row["_body_path"] = BODY_PATHS[body]
            source = _source_mesh_body_local(inner, usd_row, body)
            guard_before = _snapshot_sim_state(inner)
            direct = _attempt_direct_live_cook(inner, usd_row)
            if direct["chosen"] is not None:
                cooked_verts = direct["chosen"]["vertices_body"]
                cooked_polygons = direct["chosen"]["polygons"]
                cook_source = f"direct_live_cook:{direct['chosen']['prim_path']}"
                mirror_info = None
            else:
                mirror = _mirror_cook(
                    inner, source["_verts_body"], source["_triangles"], f"{body}_{len(shapes)}"
                )
                cooked_verts = mirror["vertices_body"]
                cooked_polygons = mirror["polygons"]
                cook_source = "mirror_cook_of_live_stage_mesh"
                mirror_info = {"mirror_removed": mirror["mirror_removed"]}
            guard = _state_guard(guard_before, _snapshot_sim_state(inner))

            physx_match = None
            live_owner_body = None
            for owner in BODY_LABELS:
                for row in physx_inventory[owner]["colliders"]:
                    if row["path"] and _paths_related(row["path"], usd_row["path"]):
                        physx_match = row
                        live_owner_body = owner
                        break
                if physx_match is not None:
                    break

            hull = ConvexHull(cooked_verts)
            cooked_volume = float(hull.volume)
            volume_check: dict[str, Any] = {"physx_volume_m3": None, "pass": False}
            aabb_check: dict[str, Any] | None = None
            if physx_match is not None:
                physx_volume = float(physx_match["volume_m3"])
                rel = abs(cooked_volume - physx_volume) / max(abs(physx_volume), 1.0e-12)
                volume_check = {
                    "physx_volume_m3": physx_volume,
                    "cooked_volume_m3": cooked_volume,
                    "relative_difference": rel,
                    "tolerance": VOLUME_PARITY_REL_TOL,
                    "pass": bool(rel <= VOLUME_PARITY_REL_TOL),
                }
                aabb_check = _aabb_extent_comparison(physx_match, cooked_verts)
            direct_used = direct["chosen"] is not None
            # Pre-registered gate: non-AABB volume parity against the live
            # PhysX collider is required for certification with no direct-cook
            # exemption; AABB comparison is recorded corroboration only.
            parity_pass = bool(volume_check["pass"])
            certified = parity_pass

            points = d332._fcl_points(hppfcl, cooked_verts)
            convex_geom = hppfcl.Convex.convexHull(points, False, "")
            if convex_geom is None:
                raise RuntimeError(f"hppfcl convex reconstruction failed for {usd_row['path']}")
            raw_bvh = d332._build_raw_bvh(hppfcl, source["_verts_body"], source["_triangles"])

            shapes.append(
                {
                    "body": body,
                    "collider_path": usd_row["path"],
                    "usd_row": {k: v for k, v in usd_row.items() if not k.startswith("_")},
                    "physx_collider": physx_match,
                    "physx_collider_matched": physx_match is not None,
                    "live_owner_body": live_owner_body,
                    "usd_parent_body": body,
                    "ownership_crossed": bool(
                        live_owner_body is not None and live_owner_body != body
                    ),
                    "source_mesh": {k: v for k, v in source.items() if not k.startswith("_")},
                    "cook_source": cook_source,
                    "direct_cook": {"attempts": direct["attempts"], "used": direct_used},
                    "mirror_info": mirror_info,
                    "cook_state_guard": guard,
                    "cooked_vertex_count": int(len(cooked_verts)),
                    "cooked_polygon_count": int(len(cooked_polygons)),
                    "cooked_vertices_body_local_m": cooked_verts.tolist(),
                    "volume_parity": volume_check,
                    "aabb_comparison_informational": aabb_check,
                    "parity_pass": parity_pass,
                    "certified": certified,
                    "_geom_cooked": convex_geom,
                    "_geom_raw": raw_bvh,
                    "_cooked_verts": cooked_verts,
                    "_raw_verts": source["_verts_body"],
                    "_hull_simplices": np.asarray(hull.simplices, dtype=np.int64),
                }
            )
    return shapes


def _body_pose_w(inner: Any, body_label: str) -> tuple[np.ndarray, np.ndarray]:
    name = body_label
    ids, names = inner._robot.find_bodies(name)
    if len(ids) != 1 or list(names) != [name]:
        raise RuntimeError(f"body lookup failed for {name}: ids={ids}, names={names}")
    data = inner._robot.data
    pos_attr = getattr(data, "body_pos_w", None)
    quat_attr = getattr(data, "body_quat_w", None)
    if pos_attr is None or quat_attr is None:
        pos_attr = data.body_link_pos_w
        quat_attr = data.body_link_quat_w
    pos = pos_attr[0, int(ids[0])].detach().cpu().numpy().astype(np.float64)
    quat = quat_attr[0, int(ids[0])].detach().cpu().numpy().astype(np.float64)
    return pos, quat


def _object_pose_w(inner: Any) -> tuple[np.ndarray, np.ndarray]:
    pos = inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = inner._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    return pos, quat


def _distance_matrix(inner: Any, shapes: list[dict[str, Any]], pose_label: str) -> dict[str, Any]:
    import hppfcl

    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    obj_pos, obj_quat = _object_pose_w(inner)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    body_poses = {}
    rows = []
    for shape in shapes:
        body = shape["body"]
        if body not in body_poses:
            body_poses[body] = _body_pose_w(inner, body)
        pos, quat = body_poses[body]
        tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(quat), pos)
        for rep in ("cooked", "raw"):
            query = d332._fcl_query(hppfcl, shape[f"_geom_{rep}"], tf, cylinder, cyl_tf)
            query["verdict"] = d332._signed_distance_verdict(float(query["signed_distance_m"]))
            depth = None if query["contact"] is None else float(query["contact"]["penetration_depth_m"])
            query["penetration_depth_m"] = depth
            if rep == "raw":
                # BVH distance is not a signed distance; consistency here means
                # a collision must come with an EPA contact record.
                query["sign_consistent"] = bool(
                    (not query["is_collision"]) or query["contact"] is not None
                )
                query["gjk_epa_depth_consistent_within_0p1mm"] = None
            else:
                query["sign_consistent"] = bool(
                    (float(query["signed_distance_m"]) < 0.0) == bool(query["is_collision"])
                )
                query["gjk_epa_depth_consistent_within_0p1mm"] = bool(
                    depth is None
                    or abs(depth + float(query["signed_distance_m"]))
                    <= d332.SIGNED_DISTANCE_BORDER_M
                )
            query["consistency_hard_check_pass"] = bool(
                query["sign_consistent"]
                and query["gjk_epa_depth_consistent_within_0p1mm"] is not False
            )
            rows.append(
                {
                    "pose": pose_label,
                    "body": body,
                    "collider_path": shape["collider_path"],
                    "representation": rep,
                    "certified": bool(shape["certified"]) if rep == "cooked" else None,
                    **query,
                }
            )
    return {
        "pose": pose_label,
        "object_pos_w_m": obj_pos.tolist(),
        "object_quat_wxyz": obj_quat.tolist(),
        "body_poses_w": {
            body: {"pos_m": pose[0].tolist(), "quat_wxyz": pose[1].tolist()}
            for body, pose in body_poses.items()
        },
        "queries": rows,
    }


def _point_probe(inner: Any, shapes: list[dict[str, Any]], point_w: np.ndarray, tag: str) -> dict[str, Any]:
    import hppfcl

    sphere = hppfcl.Sphere(POINT_PROBE_RADIUS_M)
    tf_point = hppfcl.Transform3f(np.eye(3, dtype=np.float64), np.asarray(point_w, dtype=np.float64))
    obj_pos, obj_quat = _object_pose_w(inner)
    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    cyl_tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(obj_quat), obj_pos)
    rows = []
    for shape in shapes:
        pos, quat = _body_pose_w(inner, shape["body"])
        tf = hppfcl.Transform3f(d332._quat_wxyz_to_rot(quat), pos)
        for rep in ("cooked", "raw"):
            query = d332._fcl_query(hppfcl, shape[f"_geom_{rep}"], tf, sphere, tf_point)
            surface_distance = float(query["signed_distance_m"]) + POINT_PROBE_RADIUS_M
            rows.append(
                {
                    "tag": tag,
                    "body": shape["body"],
                    "collider_path": shape["collider_path"],
                    "representation": rep,
                    "surface_distance_m": surface_distance,
                    "surface_distance_mm": surface_distance * 1000.0,
                    "is_collision": bool(query["is_collision"]),
                    "on_surface_within_1mm": bool(
                        abs(surface_distance) <= CONTACT_ON_SURFACE_TOL_M
                    ),
                }
            )
    cyl_query = d332._fcl_query(hppfcl, cylinder, cyl_tf, sphere, tf_point)
    cyl_distance = float(cyl_query["signed_distance_m"]) + POINT_PROBE_RADIUS_M
    return {
        "tag": tag,
        "point_w_m": np.asarray(point_w, dtype=np.float64).tolist(),
        "cylinder_surface_distance_mm": cyl_distance * 1000.0,
        "cylinder_is_collision": bool(cyl_query["is_collision"]),
        "shapes": rows,
    }


def _replay_parity(flat_row: dict[str, Any], row0: dict[str, float | str]) -> dict[str, Any]:
    def _vec(source: dict[str, Any], prefix: str, axes: tuple[str, ...]) -> np.ndarray:
        return np.asarray([float(source[f"{prefix}{axis}"]) for axis in axes], dtype=np.float64)

    object_delta = float(
        np.linalg.norm(
            _vec(flat_row, "object_pos_local_m_", ("x", "y", "z"))
            - _vec(row0, "object_pos_local_m_", ("x", "y", "z"))
        )
    )
    tcp_delta = float(
        np.linalg.norm(
            _vec(flat_row, "actual_tcp_local_m_", ("x", "y", "z"))
            - _vec(row0, "actual_tcp_local_m_", ("x", "y", "z"))
        )
    )
    force_ref = float(row0["gripper_link_force_norm_n"])
    force_new = float(flat_row["gripper_link_force_norm_n"])
    force_rel = abs(force_new - force_ref) / max(abs(force_ref), 1.0e-9)
    contact_delta = float(
        np.linalg.norm(
            np.asarray(
                [
                    float(flat_row[f"gripper_link_contact_point_{axis}_w_m"])
                    - float(row0[f"gripper_link_contact_point_{axis}_w_m"])
                    for axis in ("x", "y", "z")
                ],
                dtype=np.float64,
            )
        )
    )
    checks = {
        "object_pos_delta_le_0p05mm": object_delta <= REPLAY_OBJECT_POS_TOL_M,
        "tcp_delta_le_0p05mm": tcp_delta <= REPLAY_TCP_TOL_M,
        "gripper_force_rel_delta_le_1pct": force_rel <= REPLAY_FORCE_REL_TOL,
        "gripper_contact_point_delta_le_1mm": contact_delta <= REPLAY_CONTACT_POINT_TOL_M,
    }
    return {
        "object_pos_delta_mm": object_delta * 1000.0,
        "tcp_delta_mm": tcp_delta * 1000.0,
        "gripper_force_ref_n": force_ref,
        "gripper_force_replay_n": force_new,
        "gripper_force_relative_delta": force_rel,
        "gripper_contact_point_delta_mm": contact_delta * 1000.0,
        "tolerances": {
            "object_pos_m": REPLAY_OBJECT_POS_TOL_M,
            "tcp_m": REPLAY_TCP_TOL_M,
            "force_rel": REPLAY_FORCE_REL_TOL,
            "contact_point_m": REPLAY_CONTACT_POINT_TOL_M,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _overlap_state(query: dict[str, Any]) -> str:
    # Pre-registered hard check: an inconsistent GJK/EPA query is unusable as
    # decision evidence and can only be borderline (drives MIXED).
    if not query.get("consistency_hard_check_pass", True):
        return "borderline"
    distance = float(query["signed_distance_m"])
    depth = query.get("penetration_depth_m")
    if query["representation"] == "raw":
        if bool(query["is_collision"]) and depth is not None and abs(depth) >= d332.SIGNED_DISTANCE_BORDER_M:
            return "overlap"
        if not bool(query["is_collision"]) and distance >= d332.SIGNED_DISTANCE_BORDER_M:
            return "clear"
        return "borderline"
    if distance <= -d332.SIGNED_DISTANCE_BORDER_M:
        return "overlap"
    if distance >= d332.SIGNED_DISTANCE_BORDER_M:
        return "clear"
    return "borderline"


def _classify(
    replay: dict[str, Any] | None,
    distance_sets: list[dict[str, Any]],
    point_probes: list[dict[str, Any]],
    shapes: list[dict[str, Any]],
    ownership: dict[str, Any],
) -> dict[str, Any]:
    if replay is None or not replay["pass"]:
        return {
            "verdict": "D334_G0A_REPLAY_PARITY_FAIL_STOP",
            "interpretation": "the D333 step-0 replay did not reproduce the recorded row within "
            "pre-registered tolerances; frozen-pose geometry conclusions are not licensed",
        }
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for dist_set in distance_sets:
        for query in dist_set["queries"]:
            key = (query["pose"], query["collider_path"], query["representation"])
            by_key[key] = {"state": _overlap_state(query), "certified": query.get("certified")}

    recorded_probe = next((p for p in point_probes if p["tag"] == "recorded_d333_step0"), None)
    on_surface: dict[str, set[str]] = {"cooked": set(), "raw": set()}
    if recorded_probe is not None:
        for row in recorded_probe.get("shapes", []):
            if row["on_surface_within_1mm"]:
                on_surface[row["representation"]].add(row["collider_path"])

    raw_overlap = []
    cook_artifact = []
    for shape in shapes:
        path = shape["collider_path"]
        attributed = shape.get("live_owner_body") == "gripper_link"
        for pose in ("pose_b_poststep0", "pose_a_prestep"):
            raw_state = by_key.get((pose, path, "raw"), {}).get("state")
            cooked_state = by_key.get((pose, path, "cooked"), {}).get("state")
            if raw_state == "overlap" and attributed:
                raw_overlap.append({"pose": pose, "collider_path": path})
            if (
                cooked_state == "overlap"
                and raw_state == "clear"
                and shape["certified"]
                and attributed
                and path in on_surface["cooked"]
            ):
                cook_artifact.append({"pose": pose, "collider_path": path})

    if raw_overlap:
        return {
            "verdict": "D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED",
            "interpretation": "the raw tool mesh itself overlaps the cylinder at the frozen pose "
            "with the recorded gripper_link attribution; a target-family repair is the candidate",
            "evidence": raw_overlap,
        }
    if cook_artifact:
        return {
            "verdict": "D334_G0A_COOK_ARTIFACT_SUPPORTED",
            "interpretation": "a certified cooked collision shape overlaps the cylinder while its raw "
            "mesh is clear, consistent with the recorded contact; a collision-representation repair "
            "is the candidate",
            "evidence": cook_artifact,
        }
    return {
        "verdict": "D334_G0A_SHAPE_OWNERSHIP_PARITY_UNRESOLVED_MIXED_STOP",
        "interpretation": "no certified shape cleanly explains the recorded contact under the "
        "pre-registered overlap/clear/ownership rules",
        "ownership_parity_pass_all": bool(ownership["parity_pass_all"]),
    }


def _plot_cylinder(ax: Any, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rot = d332._quat_wxyz_to_rot(np.asarray(quat, dtype=np.float64))
    theta = np.linspace(0.0, 2.0 * math.pi, 40)
    circle = np.stack(
        [d332.CYLINDER_RADIUS_M * np.cos(theta), d332.CYLINDER_RADIUS_M * np.sin(theta)], axis=1
    )
    points = []
    for z in (-0.5 * d332.CYLINDER_HEIGHT_M, 0.5 * d332.CYLINDER_HEIGHT_M):
        ring = np.column_stack([circle, np.full(len(theta), z)])
        world = (rot @ ring.T).T + pos
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="tab:orange", linewidth=1.2)
        points.append(world)
    for k in range(0, len(theta), 5):
        seg = np.stack([points[0][k], points[1][k]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="tab:orange", linewidth=0.6, alpha=0.7)
    return np.vstack(points)


def _write_shape_figure(
    path: Path,
    title: str,
    inner: Any,
    shapes: list[dict[str, Any]],
    contact_points: dict[str, np.ndarray],
    zoom_center: np.ndarray | None = None,
    zoom_radius_m: float = 0.03,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.0, 7.0), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    obj_pos, obj_quat = _object_pose_w(inner)
    all_points.append(_plot_cylinder(ax, obj_pos, obj_quat))
    colors = {"link5": "tab:blue", "gripper_link": "tab:green"}
    for shape in shapes:
        pos, quat = _body_pose_w(inner, shape["body"])
        rot = d332._quat_wxyz_to_rot(quat)
        cooked_world = (rot @ shape["_cooked_verts"].T).T + pos
        ax.plot_trisurf(
            cooked_world[:, 0],
            cooked_world[:, 1],
            cooked_world[:, 2],
            triangles=shape["_hull_simplices"],
            color=colors[shape["body"]],
            alpha=0.30,
            linewidth=0.1,
        )
        raw = shape["_raw_verts"]
        stride = max(1, len(raw) // 800)
        raw_world = (rot @ raw[::stride].T).T + pos
        ax.scatter(
            raw_world[:, 0],
            raw_world[:, 1],
            raw_world[:, 2],
            s=1.0,
            color=colors[shape["body"]],
            alpha=0.45,
        )
        all_points.extend([cooked_world, raw_world])
    markers = {"recorded_d333_step0": ("r", "*"), "replayed_step0": ("m", "P")}
    for tag, point in contact_points.items():
        color, marker = markers.get(tag, ("k", "x"))
        ax.scatter([point[0]], [point[1]], [point[2]], color=color, marker=marker, s=90, label=tag)
        all_points.append(point.reshape(1, 3))
    if zoom_center is not None and np.all(np.isfinite(zoom_center)):
        center = np.asarray(zoom_center, dtype=np.float64)
        ax.set_xlim(center[0] - zoom_radius_m, center[0] + zoom_radius_m)
        ax.set_ylim(center[1] - zoom_radius_m, center[1] + zoom_radius_m)
        ax.set_zlim(center[2] - zoom_radius_m, center[2] + zoom_radius_m)
    else:
        d332._set_axes_equal(ax, np.vstack(all_points))
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return _rel(path)


def _strip_private(shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in shape.items() if not k.startswith("_")} for shape in shapes]


def _pin_check() -> dict[str, Any]:
    try:
        versions = d332._runtime_versions()
    except RuntimeError as error:
        versions = {"pin_error": str(error)}
    packages = versions.get("packages", versions)
    numpy_version = str(packages.get("numpy", np.__version__))
    try:
        import psutil

        psutil_version = str(psutil.__version__)
    except Exception:  # noqa: BLE001
        psutil_version = "unavailable"
    return {
        "versions": versions,
        "numpy_pin_1_26_0": numpy_version == "1.26.0",
        "psutil_pin_5_9_8": psutil_version == "5.9.8",
        "numpy_version": numpy_version,
        "psutil_version": psutil_version,
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D334 live collision shape / ownership audit",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "| Gate | Result |",
        "|---|---:|",
        f"| Frozen invariant contract | `{summary['frozen_contract']['pass']}` |",
        f"| Stage/sensor contracts | `{summary['runtime']['stage_contract']['hard_contract_pass']}` / "
        f"`{summary['runtime']['sensor_contract']['hard_contract_pass']}` |",
        f"| Baseline replay hard gate | `{summary['runtime']['baseline_replay']['hard_gate_pass']}` |",
        f"| Ownership parity (all) | `{summary['ownership']['parity_pass_all']}` |",
        f"| Step-0 replay parity | `{summary['replay_parity']['pass'] if summary['replay_parity'] else None}` |",
    ]
    for shape in summary["shapes"]:
        lines.append(
            f"| {shape['body']} `{shape['collider_path'].split('/')[-1]}` cook parity "
            f"(certified) | `{shape['parity_pass']}` (`{shape['certified']}`) |"
        )
    lines += ["", "## Signed distances (mm)", "", "| Pose | Body | Rep | Signed dist | State |", "|---|---|---|---:|---|"]
    for dist_set in summary["distance_sets"]:
        for query in dist_set["queries"]:
            lines.append(
                f"| {query['pose']} | {query['body']} | {query['representation']} | "
                f"`{query['signed_distance_mm']:.6f}` | {_overlap_state(query)} |"
            )
    lines += ["", summary["classification"]["interpretation"] + ".", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_runtime(args: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    d332._runtime_versions()  # D326 pin gate: abort before scene creation on pin drift
    source = json.loads(D333_SUMMARY.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    row0 = _load_d333_row0()

    canonical = dict(source["frozen_d332_contract"]["canonical"])
    recomputed = d332._canonical_contract()
    if not np.allclose(
        np.asarray(canonical["commanded_joint_rad"]),
        np.asarray(recomputed["commanded_joint_rad"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("D333 frozen canonical joint target changed")

    d333_frozen = source["frozen_d332_contract"]["invariant_hard_contract"]
    frozen_checks = {
        "seed_same_as_d333": int(args.seed) == int(source["runtime"]["seed"]),
        "robot_usd_sha256_same_as_d333": d332._sha256(args.robot_usd_path)
        == d333_frozen["robot_usd_sha256"],
        "urdf_sha256_same_as_d333": d332._sha256(args.urdf_path) == d333_frozen["urdf_sha256"],
        "physics_dt_same_as_d333": math.isclose(
            float(source["runtime"]["physics_dt_s"]), d332.PHYSICS_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "baseline_steps_same_as_d333": int(source["runtime"]["baseline_physics_steps"])
        == d332.BASELINE_PHYSICS_STEPS,
        "d333_verdict_is_mixed_stop": source["verdict"]
        == "D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP",
        "d333_target_csv_exists": D333_TARGET_CSV.is_file(),
    }
    frozen_contract = {
        "checks": frozen_checks,
        "pass": all(frozen_checks.values()),
        "seed": int(args.seed),
        "robot_usd_sha256": d332._sha256(args.robot_usd_path),
        "urdf_sha256": d332._sha256(args.urdf_path),
        "source_summary": _rel(D333_SUMMARY),
        "source_summary_sha256": d332._sha256(D333_SUMMARY),
        "source_target_csv": _rel(D333_TARGET_CSV),
        "source_target_csv_sha256": d332._sha256(D333_TARGET_CSV),
    }
    frozen_path = args.out_dir / "d334_frozen_invariant_contract.json"
    _json_dump(frozen_path, frozen_contract)
    if not frozen_contract["pass"]:
        raise RuntimeError(f"D334 frozen invariant contract failed; see {frozen_path}")

    inner = d333._make_runtime_env(args)
    try:
        inner.reset(seed=int(args.seed))
        stage_contract = d333._stage_contract(inner)
        sensor_contract, filter_map = d333._sensor_contract(inner)
        _json_dump(
            args.out_dir / "d334_prebaseline_contract.json",
            {
                "artifact": "D334_PREBASELINE_CONTRACT",
                "stage_contract": stage_contract,
                "sensor_contract": sensor_contract,
            },
        )
        if not stage_contract["hard_contract_pass"] or not sensor_contract["hard_contract_pass"]:
            raise RuntimeError("D334 prebaseline contract failed")

        physx_inventory = {body: _property_query_body(inner, simulation_app, body) for body in BODY_LABELS}
        usd_inventory = {body: _usd_collision_inventory(inner, body) for body in BODY_LABELS}
        ownership = _ownership_parity(physx_inventory, usd_inventory)
        inventory_payload = {
            "artifact": "D334_LIVE_COLLIDER_INVENTORY",
            "physx_property_query": physx_inventory,
            "usd_instance_proxy_traversal": usd_inventory,
            "ownership": ownership,
        }
        _json_dump(args.out_dir / "d334_live_collider_inventory.json", inventory_payload)
        for body in BODY_LABELS:
            if not physx_inventory[body]["pass"]:
                raise RuntimeError(f"live property query failed for {body}: {physx_inventory[body]}")

        shapes = _build_shape_records(inner, physx_inventory, usd_inventory)
        _json_dump(
            args.out_dir / "d334_cook_parity.json",
            {"artifact": "D334_COOK_PARITY", "shapes": _strip_private(shapes)},
        )

        origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
        object_start_w = origin + d332.OBJECT_CENTER_LOCAL_M

        q_home = np.radians(np.asarray(d332.HOME_DEG, dtype=np.float64))
        q_home[5] = 0.0
        home_target = d332._write_exact_state(inner, q_home, d332.OBJECT_CENTER_LOCAL_M)
        base_root_pos, base_root_quat = d333._root_pose(inner)
        baseline_rows = []
        for step in range(d332.BASELINE_PHYSICS_STEPS):
            d332._physics_step(inner)
            contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
            baseline_rows.append(
                d333._state_row(
                    inner,
                    phase="d334_baseline_replay",
                    step=step,
                    command_target=home_target,
                    canonical=canonical,
                    object_start_w=object_start_w,
                    root_start_pos_w=base_root_pos,
                    root_start_quat_wxyz=base_root_quat,
                    contact=contact,
                )
            )
        baseline_stats = d333._baseline_statistics(
            baseline_rows, stage_contract=stage_contract, sensor_contract=sensor_contract
        )
        d333._write_trace_csv(args.out_dir / "d334_baseline_replay_trace.csv", baseline_rows)
        baseline_stats["d333_reference"] = {
            "first_step_z_delta_mm": source["runtime"]["baseline"]["first_step_z_delta_mm"],
            "support_table_force_z_last50_median_n": source["runtime"]["baseline"][
                "support_table_force_z_last50_median_n"
            ],
        }
        if not baseline_stats["hard_gate_pass"]:
            fail_rows = [baseline_rows[-1]]
            for frame in fail_rows[0]["frames"]:
                frame["name"] = str(frame["name"]).replace("d333_", "d334_")
            summary = _finalize(
                args,
                inner,
                verdict="D334_G0A_AUDIT_CONTRACT_FAIL_STOP",
                classification={
                    "verdict": "D334_G0A_AUDIT_CONTRACT_FAIL_STOP",
                    "interpretation": "the replayed sole-support baseline failed the D333 hard gate",
                },
                frozen_contract=frozen_contract,
                stage_contract=stage_contract,
                sensor_contract=sensor_contract,
                baseline_stats=baseline_stats,
                ownership=ownership,
                shapes=shapes,
                distance_sets=[],
                point_probes=[],
                replay=None,
                audit_rows=fail_rows,
                canonical=canonical,
                pose_a_fk=None,
            )
            return summary

        target = d332._write_exact_state(
            inner, np.asarray(canonical["commanded_joint_rad"], dtype=np.float64), d332.OBJECT_CENTER_LOCAL_M
        )
        pose_a_root_pos, pose_a_root_quat = d333._root_pose(inner)
        pose_a_contact_error = None
        try:
            pose_a_contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
        except Exception as error:  # noqa: BLE001 - sensor buffers may be freshly reset
            pose_a_contact = None
            pose_a_contact_error = f"{type(error).__name__}: {error}"
        link5_pos_w, _ = _body_pose_w(inner, "link5")
        pose_a_fk = {
            "commanded_link5_pos_local_m": canonical["commanded_link5_pos_local_m"],
            "live_link5_pos_local_m": (link5_pos_w - origin).tolist(),
            "fk_vs_live_link5_error_mm": float(
                np.linalg.norm(
                    link5_pos_w - origin - np.asarray(canonical["commanded_link5_pos_local_m"])
                )
                * 1000.0
            ),
            "pose_a_contact_error": pose_a_contact_error,
            "pose_a_row_present": pose_a_contact is not None,
        }
        audit_rows = []
        if pose_a_contact is not None:
            pose_a_row = d333._state_row(
                inner,
                phase="d334_pose_a_frozen_prestep",
                step=0,
                command_target=target,
                canonical=canonical,
                object_start_w=object_start_w,
                root_start_pos_w=pose_a_root_pos,
                root_start_quat_wxyz=pose_a_root_quat,
                contact=pose_a_contact,
            )
            audit_rows.append(pose_a_row)
        pose_a_distances = _distance_matrix(inner, shapes, "pose_a_prestep")
        pose_a_png = args.out_dir / "d334_pose_a_shapes.png"
        pose_a_png_rel = _write_shape_figure(
            pose_a_png, "D334 pose A (frozen pre-step) live shapes", inner, shapes, {}
        )

        d332._physics_step(inner)
        replay_contact = d332._contact_state(inner._d333_contact_sensor, filter_map)
        replay_row = d333._state_row(
            inner,
            phase="d334_step0_replay",
            step=0,
            command_target=target,
            canonical=canonical,
            object_start_w=object_start_w,
            root_start_pos_w=pose_a_root_pos,
            root_start_quat_wxyz=pose_a_root_quat,
            contact=replay_contact,
        )
        audit_rows.append(replay_row)
        for row in audit_rows:
            for frame in row["frames"]:
                frame["name"] = str(frame["name"]).replace("d333_", "d334_")
        d333._write_trace_csv(args.out_dir / "d334_audit_trace.csv", audit_rows)
        replay_flat = d333._flatten_trace_row(replay_row)
        replay = _replay_parity(replay_flat, row0)
        _json_dump(
            args.out_dir / "d334_step0_replay_parity.json",
            {"artifact": "D334_STEP0_REPLAY_PARITY", "parity": replay},
        )

        pose_b_distances = _distance_matrix(inner, shapes, "pose_b_poststep0")
        recorded_point = np.asarray(
            [float(row0[f"gripper_link_contact_point_{axis}_w_m"]) for axis in ("x", "y", "z")],
            dtype=np.float64,
        )
        replayed_point = np.asarray(
            [float(replay_flat[f"gripper_link_contact_point_{axis}_w_m"]) for axis in ("x", "y", "z")],
            dtype=np.float64,
        )
        point_probes = []
        for tag, point in (
            ("recorded_d333_step0", recorded_point),
            ("replayed_step0", replayed_point),
        ):
            if np.all(np.isfinite(point)):
                point_probes.append(_point_probe(inner, shapes, point, tag))
            else:
                point_probes.append(
                    {"tag": tag, "skipped": "contact point not finite", "point_w_m": point.tolist()}
                )
        distance_sets = [pose_a_distances, pose_b_distances]
        _json_dump(
            args.out_dir / "d334_signed_distance_matrix.json",
            {
                "artifact": "D334_SIGNED_DISTANCE_MATRIX",
                "border_mm": d332.SIGNED_DISTANCE_BORDER_M * 1000.0,
                "distance_sets": distance_sets,
                "point_probes": point_probes,
            },
        )

        classification = _classify(replay, distance_sets, point_probes, shapes, ownership)
        summary = _finalize(
            args,
            inner,
            verdict=classification["verdict"],
            classification=classification,
            frozen_contract=frozen_contract,
            stage_contract=stage_contract,
            sensor_contract=sensor_contract,
            baseline_stats=baseline_stats,
            ownership=ownership,
            shapes=shapes,
            distance_sets=distance_sets,
            point_probes=point_probes,
            replay=replay,
            audit_rows=audit_rows,
            canonical=canonical,
            pose_a_fk=pose_a_fk,
            contact_points={
                "recorded_d333_step0": recorded_point,
                "replayed_step0": replayed_point,
            },
            extra_snapshots=[(pose_a_png, pose_a_png_rel)],
        )
        return summary
    finally:
        inner.close()


def _finalize(
    args: argparse.Namespace,
    inner: Any,
    *,
    verdict: str,
    classification: dict[str, Any],
    frozen_contract: dict[str, Any],
    stage_contract: dict[str, Any],
    sensor_contract: dict[str, Any],
    baseline_stats: dict[str, Any],
    ownership: dict[str, Any],
    shapes: list[dict[str, Any]],
    distance_sets: list[dict[str, Any]],
    point_probes: list[dict[str, Any]],
    replay: dict[str, Any] | None,
    audit_rows: list[dict[str, Any]],
    canonical: dict[str, Any],
    pose_a_fk: dict[str, Any] | None,
    contact_points: dict[str, np.ndarray] | None = None,
    extra_snapshots: list[tuple[Path, str]] | None = None,
) -> dict[str, Any]:
    snapshots = []
    snapshot_paths: list[Path] = []
    for extra_path, extra_rel in extra_snapshots or []:
        snapshot_paths.append(extra_path)
        snapshots.append(extra_rel)
    points = contact_points or {}
    fig_b = args.out_dir / "d334_pose_b_shapes.png"
    snapshot_paths.append(fig_b)
    snapshots.append(
        _write_shape_figure(fig_b, "D334 pose B (replayed post-step-0) live shapes", inner, shapes, points)
    )
    if replay is not None:
        fig_contact = args.out_dir / "d334_contact_map.png"
        snapshot_paths.append(fig_contact)
        snapshots.append(
            _write_shape_figure(
                fig_contact,
                "D334 contact-point mapping (pose B, zoomed)",
                inner,
                shapes,
                points,
                zoom_center=points.get("recorded_d333_step0"),
            )
        )

    marker_frames = audit_rows[-1]["frames"] if audit_rows else []
    marker_status = (
        draw_frames(marker_frames, prim_path="/World/D334AuditFrames")
        if marker_frames
        else {"ok": False, "error": "no audit rows"}
    )
    rrd_path = args.out_dir / "d334_live_collision_audit_trace.rrd"
    rrd_status = log_rerun(
        rrd_path,
        frames=marker_frames,
        joint_state={
            "label": "d334_live_collision_shape_ownership_audit",
            "replayed_baseline_steps": d332.BASELINE_PHYSICS_STEPS,
            "replayed_target_steps": 1 if replay is not None else 0,
            "trace_rows": len(audit_rows),
            "physics_dt_s": d332.PHYSICS_DT_S,
            "object": "cylinder_d34_h90",
        },
        joint_trace=audit_rows if audit_rows else None,
        urdf_path=args.urdf_path,
        live_viewer=False,
        app_id="roarm_g0a_d334_live_collision_audit",
    )
    if bool(rrd_status.get("ok")):
        rrd_status["nonzero_file"] = bool(rrd_path.is_file() and rrd_path.stat().st_size > 0)

    artifact_checks = {
        "snapshot_count_between_1_and_3": 1 <= len(snapshot_paths) <= 3,
        "snapshots_exist_and_nonzero": all(
            path.is_file() and path.stat().st_size > 0 for path in snapshot_paths
        ),
        "marker_status_ok": bool(marker_status.get("ok")),
        "rrd_status_ok": bool(rrd_status.get("ok")),
        "rrd_nonzero_file": bool(rrd_status.get("nonzero_file")),
    }
    artifact_contract = {"checks": artifact_checks, "pass": all(artifact_checks.values())}
    final_verdict = (
        verdict if artifact_contract["pass"] else "D334_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP"
    )

    summary = {
        "verdict": final_verdict,
        "scientific_verdict_before_artifact_gate": verdict,
        "active_case": "G0a cylinder D34xH90 frozen-pose live collision shape / ownership audit",
        "new_variable": [],
        "replayed_physics_steps": {
            "baseline": d332.BASELINE_PHYSICS_STEPS,
            "target": 1 if replay is not None else 0,
            "all_steps_are_d333_replays": True,
        },
        "frozen_contract": frozen_contract,
        "runtime": {
            "seed": int(args.seed),
            "num_envs": 1,
            "physics_dt_s": float(inner.physics_dt),
            "stage_contract": stage_contract,
            "sensor_contract": sensor_contract,
            "baseline_replay": baseline_stats,
        },
        "ownership": ownership,
        "shapes": _strip_private(shapes),
        "pose_a_fk_parity": pose_a_fk,
        "replay_parity": replay,
        "distance_sets": distance_sets,
        "point_probes": point_probes,
        "classification": classification,
        "canonical": canonical,
        "outcome_guards": {
            "g0a_pass": False,
            "alignment_ladder_promoted": False,
            "collision_repair_authorized": False,
            "target_repair_authorized": False,
            "mesh_rewritten": False,
            "stop_after_d334": True,
        },
        "visualization": {
            "snapshots": snapshots,
            "snapshot_count": len(snapshots),
            "marker_status": marker_status,
            "rrd_status": rrd_status,
        },
        "artifact_contract": artifact_contract,
        "artifacts": {
            name: (_rel(args.out_dir / filename) if (args.out_dir / filename).is_file() else None)
            for name, filename in (
                ("frozen_invariant_contract_json", "d334_frozen_invariant_contract.json"),
                ("prebaseline_contract_json", "d334_prebaseline_contract.json"),
                ("live_collider_inventory_json", "d334_live_collider_inventory.json"),
                ("cook_parity_json", "d334_cook_parity.json"),
                ("signed_distance_matrix_json", "d334_signed_distance_matrix.json"),
                ("step0_replay_parity_json", "d334_step0_replay_parity.json"),
                ("baseline_replay_trace_csv", "d334_baseline_replay_trace.csv"),
                ("audit_trace_csv", "d334_audit_trace.csv"),
                ("rrd", "d334_live_collision_audit_trace.rrd"),
            )
        }
        | {
            "summary_json": _rel(args.out_dir / "g0a_d334_live_collision_audit_summary.json"),
            "summary_markdown": _rel(args.out_dir / "g0a_d334_live_collision_audit_summary.md"),
        },
        "environment": _pin_check(),
        "non_goals_respected": [
            "no mesh rewrite or collision-approximation change",
            "no target/gate/offset/standoff tuning",
            "no ownership scan beyond link5/gripper_link",
            "no new physics beyond the 200+1 D333 replay",
            "no gripper close/grasp/lift/G0b, no RL/PPO/randomization",
            "no render beyond at most three diagnostic PNGs; no video",
        ],
    }
    _json_dump(args.out_dir / "g0a_d334_live_collision_audit_summary.json", summary)
    _write_summary_markdown(args.out_dir / "g0a_d334_live_collision_audit_summary.md", summary)
    return summary


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=d333.DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
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
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        try:
            summary = _run_runtime(args, simulation_app)
            detail = []
            for dist_set in summary["distance_sets"]:
                for query in dist_set["queries"]:
                    if query["pose"] == "pose_b_poststep0":
                        detail.append(
                            f"{query['body']}/{query['representation']}="
                            f"{query['signed_distance_mm']:.3f}mm"
                        )
            print(f"{summary['verdict']}: {' '.join(detail) if detail else 'no distance rows'}", flush=True)
            return 0 if bool(summary["artifact_contract"]["pass"]) else 1
        except Exception:
            import traceback

            traceback.print_exc()
            try:
                summary_path = args.out_dir / "g0a_d334_live_collision_audit_summary.json"
                if not summary_path.is_file():
                    args.out_dir.mkdir(parents=True, exist_ok=True)
                    _json_dump(
                        summary_path,
                        {
                            "verdict": "D334_G0A_AUDIT_CONTRACT_FAIL_STOP",
                            "interpretation": "run aborted before a full summary was written",
                            "error": traceback.format_exc(),
                        },
                    )
            except Exception:  # noqa: BLE001 - best-effort fail artifact
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
