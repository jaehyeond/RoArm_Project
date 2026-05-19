#!/usr/bin/env python3
"""Static gripper/cube geometry probe for P7 Branch B.

This is numpy-only and does not launch Isaac. It checks whether the current
2cm-cube grasp plan places the authored gripper collision mesh in a plausible
contact/enclosure relationship with the cube. It does not validate physics,
friction, force closure, constraints, SurfaceGripper, transport, or release.
"""
from __future__ import annotations

import argparse
import itertools
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    NAMED_GRASPS,
    _build_plan,
    _build_plan_from_center,
    _fmt_xyz,
    _norm,
    _rot_z,
    _yes,
)
from roarm_kinematics import _CHAIN, Tmat, Trot_z, fk_tcp  # noqa: E402


GRASP_GRIPPER_THRESH_RAD = 0.4
GRIPPER_JOINT_ORIGIN_XYZ = np.array([0.0, 0.018821, 0.052035], dtype=np.float64)
GRIPPER_JOINT_ORIGIN_RPY = np.array([-math.pi / 2.0, -math.pi / 2.0, 0.0], dtype=np.float64)


def _load_stl_vertices_m(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5].lower() == b"solid":
        verts = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            raise ValueError(f"ASCII STL has no vertices: {path}")
        return np.asarray(verts, dtype=np.float64) * 0.001
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    n_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + 50 * n_tri
    if expected > len(data):
        raise ValueError(f"STL size mismatch: {path} expected {expected}, got {len(data)}")
    verts = []
    off = 84
    for _ in range(n_tri):
        vals = struct.unpack("<12fH", data[off:off + 50])
        verts.extend((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    return np.asarray(verts, dtype=np.float64) * 0.001


def _candidate_mesh_paths() -> list[Path]:
    return [
        REPO / "local_assets/roarm_m3/urdf/meshes/gripper_link_collision_g2a.stl",
        REPO / "local_assets/roarm_m3/urdf/meshes/gripper_link.stl",
        REPO / "assets/roarm_m3/urdf/meshes/gripper_link.stl",
    ]


def _load_gripper_mesh(path_arg: str | None) -> tuple[np.ndarray, Path]:
    paths = [Path(path_arg)] if path_arg else _candidate_mesh_paths()
    for path in paths:
        if path.exists():
            return _load_stl_vertices_m(path), path
    raise FileNotFoundError("no gripper mesh found; pass --gripper_mesh_stl")


def _link5_transform(q_deg: np.ndarray) -> np.ndarray:
    q_rad = np.radians(q_deg)
    transform = np.eye(4, dtype=np.float64)
    for name, xyz, rpy, qi in _CHAIN:
        if name == "link5_to_tcp":
            break
        transform = transform @ Tmat(xyz, rpy)
        if qi is not None:
            transform = transform @ Trot_z(q_rad[qi])
    return transform


def _gripper_transform(q_deg: np.ndarray) -> np.ndarray:
    return _link5_transform(q_deg) @ Tmat(GRIPPER_JOINT_ORIGIN_XYZ, GRIPPER_JOINT_ORIGIN_RPY) @ Trot_z(math.radians(q_deg[5]))


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3]


def _box_vertices(center: np.ndarray, size: np.ndarray, yaw_deg: float) -> np.ndarray:
    rot = _rot_z(yaw_deg)
    half = size / 2.0
    corners = np.asarray(list(itertools.product([-1.0, 1.0], repeat=3)), dtype=np.float64) * half
    return center + corners @ rot.T


def _aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _aabb_overlap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))


def _aabb_signed_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    gap = np.zeros(3, dtype=np.float64)
    for idx in range(3):
        if a_max[idx] < b_min[idx]:
            gap[idx] = b_min[idx] - a_max[idx]
        elif b_max[idx] < a_min[idx]:
            gap[idx] = a_min[idx] - b_max[idx]
        else:
            gap[idx] = -min(a_max[idx], b_max[idx]) + max(a_min[idx], b_min[idx])
    return gap


def _points_inside_aabb(points: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> int:
    inside = np.all((points >= box_min) & (points <= box_max), axis=1)
    return int(np.count_nonzero(inside))


def _min_point_aabb_distance(points: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> float:
    below = np.maximum(box_min - points, 0.0)
    above = np.maximum(points - box_max, 0.0)
    return float(np.linalg.norm(below + above, axis=1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--pose_label", default="seed0_S1")
    ap.add_argument("--object_xy", nargs=2, type=float, default=None)
    ap.add_argument(
        "--object_center_z",
        type=float,
        default=None,
        help="Override cube center z for settled-pose static checks; default uses table_z + size_z/2.",
    )
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--grasp_name", choices=sorted(NAMED_GRASPS), default="top_center")
    ap.add_argument("--normalized_grasp", nargs=3, type=float, default=None)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0, 35.0, 40.0, 45.84])
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--gripper_mesh_stl", default=None)
    args = ap.parse_args()

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")

    mesh_local, mesh_path = _load_gripper_mesh(args.gripper_mesh_stl)
    plan = _build_plan(args)
    if args.object_center_z is not None:
        center = plan.center.copy()
        center[2] = float(args.object_center_z)
        plan = _build_plan_from_center(args, center, f"{plan.label}_zoverride")
    cube_vertices = _box_vertices(plan.center, object_size, plan.yaw_deg)
    cube_min, cube_max = _aabb(cube_vertices)

    print("[cube2cm_gripper_static_geom] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_gripper_static_geom] "
        "diagnostic_only=YES no env/chain default edits, no p7 training, no constraints, "
        "no SurfaceGripper, no transport, no release"
    )
    print(
        f"[cube2cm_gripper_static_geom] mesh={mesh_path} vertices={len(mesh_local)} "
        f"local_bbox_min={_fmt_xyz(mesh_local.min(axis=0))} local_bbox_max={_fmt_xyz(mesh_local.max(axis=0))}"
    )
    print(
        f"[cube2cm_gripper_static_geom] selected pose={plan.label} center={_fmt_xyz(plan.center)} "
        f"yaw_deg={plan.yaw_deg:.1f} grasp={plan.grasp_name} world_grasp={_fmt_xyz(plan.world_grasp)} "
        f"descend_tcp={_fmt_xyz(plan.descend_tcp)} ik_ok={_yes(plan.descend_ik_ok)} "
        f"descend_fk_error_m={_norm(fk_tcp(plan.q_descend_deg) - plan.descend_tcp):.6f}"
    )
    print(
        f"[cube2cm_gripper_static_geom] cube_aabb_min={_fmt_xyz(cube_min)} "
        f"cube_aabb_max={_fmt_xyz(cube_max)} grasp_distance_thresh_m=0.025000 "
        f"grasp_gripper_thresh_deg={math.degrees(GRASP_GRIPPER_THRESH_RAD):.3f}"
    )

    any_overlap = False
    any_mesh_vertex_inside_cube = False
    for close_deg in args.close_deg:
        q = plan.q_descend_deg.copy()
        q[5] = close_deg
        mesh_world = _transform_points(_gripper_transform(q), mesh_local)
        mesh_min, mesh_max = _aabb(mesh_world)
        overlap = _aabb_overlap(mesh_min, mesh_max, cube_min, cube_max)
        signed_gap = _aabb_signed_gap(mesh_min, mesh_max, cube_min, cube_max)
        mesh_inside_count = _points_inside_aabb(mesh_world, cube_min, cube_max)
        cube_inside_gripper_bbox = _points_inside_aabb(cube_vertices, mesh_min, mesh_max)
        min_mesh_to_cube = _min_point_aabb_distance(mesh_world, cube_min, cube_max)
        tcp_to_cube_center = _norm(fk_tcp(q) - plan.center)
        latch_cond = tcp_to_cube_center < 0.025 and math.radians(close_deg) >= GRASP_GRIPPER_THRESH_RAD
        any_overlap = any_overlap or bool(np.all(overlap > 0.0))
        any_mesh_vertex_inside_cube = any_mesh_vertex_inside_cube or mesh_inside_count > 0
        print(
            f"[cube2cm_gripper_static_geom] sample close_deg={close_deg:.2f} "
            f"tcp_to_cube_center_m={tcp_to_cube_center:.6f} env_latch_condition={_yes(latch_cond)} "
            f"mesh_aabb_min={_fmt_xyz(mesh_min)} mesh_aabb_max={_fmt_xyz(mesh_max)} "
            f"aabb_overlap_m={_fmt_xyz(overlap)} signed_gap_m={_fmt_xyz(signed_gap)} "
            f"mesh_vertices_inside_cube_aabb={mesh_inside_count} cube_vertices_inside_mesh_aabb={cube_inside_gripper_bbox} "
            f"min_mesh_vertex_to_cube_aabb_m={min_mesh_to_cube:.6f}"
        )

    if any_mesh_vertex_inside_cube:
        verdict = "STATIC_MESH_PENETRATION_OR_CONTACT_POSSIBLE"
    elif any_overlap:
        verdict = "STATIC_AABB_OVERLAP_ONLY_AMBIGUOUS"
    else:
        verdict = "STATIC_NO_GRIPPER_CUBE_AABB_OVERLAP"
    print(f"[cube2cm_gripper_static_geom] verdict={verdict}")
    print("[cube2cm_gripper_static_geom] CUBE2CM_GRIPPER_STATIC_GEOMETRY_PROBE_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
