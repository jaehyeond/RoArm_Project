#!/usr/bin/env python3
"""Prepare a diagnostic RoArm URDF with an explicit opposing jaw contact pair.

This is a local/static asset-prep utility. It does not launch Isaac, does not
edit the repo's default RoArm assets, and does not add constraints,
SurfaceGripper, transport, release, training, or chain-default behavior.

The output URDF is intended for a later, separately approved USD conversion and
local grasp/close/hold/tiny-lift diagnostic only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from p7_branch_b_cube2cm_gripper_static_geometry_probe import (  # noqa: E402
    _aabb,
    _aabb_overlap,
    _box_vertices,
    _fmt_xyz,
    _gripper_transform,
    _link5_transform,
    _points_inside_aabb,
    _transform_points,
)
from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    _build_plan,
    _build_plan_from_center,
    _yes,
)


DEFAULT_SOURCE_URDF_DIR = REPO / "local_assets/roarm_m3/urdf"
DEFAULT_OUTPUT_URDF_DIR = Path("/tmp/p7_branch_b_cube2cm_opposing_jaw_collision_urdf")
MOVING_MESH_NAME = "gripper_link_collision_cube2cm_opposing_moving_jaw_v1.stl"
COUNTER_MESH_NAME = "cube2cm_counter_jaw_v1.stl"
COUNTER_LINK_NAME = "cube2cm_counter_jaw_link"
COUNTER_JOINT_NAME = "link5_to_cube2cm_counter_jaw_link"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _box_stl_text(name: str, center_m: np.ndarray, size_m: np.ndarray) -> str:
    half = size_m / 2.0
    corners_m = np.asarray(
        [
            [center_m[0] - half[0], center_m[1] - half[1], center_m[2] - half[2]],
            [center_m[0] + half[0], center_m[1] - half[1], center_m[2] - half[2]],
            [center_m[0] + half[0], center_m[1] + half[1], center_m[2] - half[2]],
            [center_m[0] - half[0], center_m[1] + half[1], center_m[2] - half[2]],
            [center_m[0] - half[0], center_m[1] - half[1], center_m[2] + half[2]],
            [center_m[0] + half[0], center_m[1] - half[1], center_m[2] + half[2]],
            [center_m[0] + half[0], center_m[1] + half[1], center_m[2] + half[2]],
            [center_m[0] - half[0], center_m[1] + half[1], center_m[2] + half[2]],
        ],
        dtype=np.float64,
    )
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 4, 5), (0, 5, 1),  # -y
        (1, 5, 6), (1, 6, 2),  # +x
        (2, 6, 7), (2, 7, 3),  # +y
        (3, 7, 4), (3, 4, 0),  # -x
    ]
    lines = [f"solid {name}"]
    for face in faces:
        a, b, c = (corners_m[i] for i in face)
        normal = np.cross(b - a, c - a)
        n = float(np.linalg.norm(normal))
        if n > 0.0:
            normal = normal / n
        lines.append(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
        lines.append("    outer loop")
        for v in (a, b, c):
            mm = v * 1000.0
            lines.append(f"      vertex {mm[0]:.9g} {mm[1]:.9g} {mm[2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    lines.append("")
    return "\n".join(lines)


def _load_ascii_stl_vertices_m(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"no vertices found in generated STL: {path}")
    return np.asarray(verts, dtype=np.float64) * 0.001


def _format_xyz_attr(value: np.ndarray) -> str:
    return f"{value[0]:.9f} {value[1]:.9f} {value[2]:.9f}"


def _insert_counter_link_and_joint(text: str, counter_mesh_rel: str, origin_link5_m: np.ndarray) -> str:
    if COUNTER_LINK_NAME in text or COUNTER_JOINT_NAME in text:
        raise RuntimeError("counter jaw link/joint already present in URDF copy")
    link_block = f"""
<!-- Diagnostic-only fixed opposing jaw for 2cm cube local grasp tests. -->
<link name="{COUNTER_LINK_NAME}">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="0.001"/>
    <inertia ixx="1e-08" ixy="0" ixz="0" iyy="1e-08" iyz="0" izz="1e-08"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="{counter_mesh_rel}" scale="0.001 0.001 0.001"/>
    </geometry>
    <material name="silver"/>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="{counter_mesh_rel}" scale="0.001 0.001 0.001"/>
    </geometry>
  </collision>
</link>
"""
    joint_block = f"""
<!-- Diagnostic-only fixed opposing jaw mount; not a chain-default constraint. -->
<joint name="{COUNTER_JOINT_NAME}" type="fixed">
  <origin xyz="{_format_xyz_attr(origin_link5_m)}" rpy="0 0 0"/>
  <parent link="link5"/>
  <child link="{COUNTER_LINK_NAME}"/>
</joint>
"""
    link_anchor = "<!-- Hand TCP (Tool Center Point) -->"
    joint_anchor = "<!-- Fixed: wrist to hand TCP -->"
    if link_anchor not in text:
        raise RuntimeError("URDF link insertion anchor not found")
    if joint_anchor not in text:
        raise RuntimeError("URDF joint insertion anchor not found")
    text = text.replace(link_anchor, link_block + "\n" + link_anchor, 1)
    text = text.replace(joint_anchor, joint_block + "\n" + joint_anchor, 1)
    return text


def _print_static_sample(
    close_deg: float,
    q_descend_deg: np.ndarray,
    moving_mesh_m: np.ndarray,
    counter_mesh_m: np.ndarray,
    counter_origin_link5_m: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
    plan_center: np.ndarray,
) -> tuple[bool, bool]:
    q = q_descend_deg.copy()
    q[5] = close_deg
    moving_world = _transform_points(_gripper_transform(q), moving_mesh_m)
    counter_world = _transform_points(_link5_transform(q) @ _translation(counter_origin_link5_m), counter_mesh_m)
    moving_min, moving_max = _aabb(moving_world)
    counter_min, counter_max = _aabb(counter_world)
    moving_overlap = _aabb_overlap(moving_min, moving_max, cube_min, cube_max)
    counter_overlap = _aabb_overlap(counter_min, counter_max, cube_min, cube_max)
    moving_inside = _points_inside_aabb(moving_world, cube_min, cube_max)
    counter_inside = _points_inside_aabb(counter_world, cube_min, cube_max)
    moving_center = 0.5 * (moving_min + moving_max)
    counter_center = 0.5 * (counter_min + counter_max)
    axis_sep_y = abs(float(moving_center[1] - counter_center[1]))
    cube_between_y = (
        min(moving_center[1], counter_center[1]) <= plan_center[1] <= max(moving_center[1], counter_center[1])
    )
    moving_ok = bool(np.any(moving_overlap > 0.0) and moving_inside > 0)
    counter_ok = bool(np.any(counter_overlap > 0.0) and counter_inside > 0)
    print(
        f"[cube2cm_opposing_jaw_urdf] static_sample close_deg={close_deg:.2f} "
        f"moving_aabb_min={_fmt_xyz(moving_min)} moving_aabb_max={_fmt_xyz(moving_max)} "
        f"moving_overlap_m={_fmt_xyz(moving_overlap)} moving_vertices_inside_cube={moving_inside} "
        f"counter_aabb_min={_fmt_xyz(counter_min)} counter_aabb_max={_fmt_xyz(counter_max)} "
        f"counter_overlap_m={_fmt_xyz(counter_overlap)} counter_vertices_inside_cube={counter_inside} "
        f"jaw_center_sep_y_m={axis_sep_y:.6f} cube_center_between_jaws_y={_yes(cube_between_y)}",
        flush=True,
    )
    return moving_ok, counter_ok


def _translation(xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    return transform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_urdf_dir", type=Path, default=DEFAULT_SOURCE_URDF_DIR)
    ap.add_argument("--output_urdf_dir", type=Path, default=DEFAULT_OUTPUT_URDF_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--pose_label", default="seed0_S1")
    ap.add_argument("--object_xy", nargs=2, type=float, default=None)
    ap.add_argument("--object_center_z", type=float, default=0.010)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--grasp_name", default="top_center")
    ap.add_argument("--normalized_grasp", nargs=3, type=float, default=None)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0])
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.006, 0.002, 0.012])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.006, 0.002, 0.012])
    ap.add_argument("--jaw_overlap_m", type=float, default=0.001)
    ap.add_argument("--jaw_center_z_m", type=float, default=0.011)
    args = ap.parse_args()

    src = args.source_urdf_dir.expanduser().resolve()
    dst = args.output_urdf_dir.expanduser().resolve()
    src_urdf = src / "roarm_m3.urdf"
    dst_urdf = dst / "roarm_m3.urdf"
    if not src_urdf.exists():
        raise FileNotFoundError(f"source URDF not found: {src_urdf}")
    if dst.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to replace: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    if np.any(moving_size <= 0.0) or np.any(counter_size <= 0.0):
        raise ValueError("jaw sizes must be positive")
    if args.jaw_overlap_m <= 0.0:
        raise ValueError("jaw_overlap_m must be positive")

    plan = _build_plan(args)
    if args.object_center_z is not None:
        center = plan.center.copy()
        center[2] = float(args.object_center_z)
        plan = _build_plan_from_center(args, center, f"{plan.label}_zoverride")
    if not plan.descend_ik_ok:
        raise RuntimeError("descend IK did not converge for diagnostic pose")

    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)

    cube_min, cube_max = _aabb(_box_vertices(plan.center, object_size, plan.yaw_deg))
    moving_world_center = np.array(
        [
            plan.center[0],
            plan.center[1] + object_size[1] / 2.0 - args.jaw_overlap_m,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )
    counter_world_center = np.array(
        [
            plan.center[0],
            plan.center[1] - object_size[1] / 2.0 + args.jaw_overlap_m,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )

    moving_local_center = np.linalg.inv(_gripper_transform(q_design)) @ np.array(
        [moving_world_center[0], moving_world_center[1], moving_world_center[2], 1.0],
        dtype=np.float64,
    )
    counter_origin_link5 = np.linalg.inv(_link5_transform(q_design)) @ np.array(
        [counter_world_center[0], counter_world_center[1], counter_world_center[2], 1.0],
        dtype=np.float64,
    )
    moving_local_center_m = moving_local_center[:3]
    counter_origin_link5_m = counter_origin_link5[:3]

    meshes_dir = dst / "meshes"
    moving_mesh = meshes_dir / MOVING_MESH_NAME
    counter_mesh = meshes_dir / COUNTER_MESH_NAME
    moving_mesh.write_text(
        _box_stl_text("cube2cm_opposing_moving_jaw", moving_local_center_m, moving_size),
        encoding="utf-8",
    )
    counter_mesh.write_text(
        _box_stl_text("cube2cm_counter_jaw", np.zeros(3, dtype=np.float64), counter_size),
        encoding="utf-8",
    )

    text = dst_urdf.read_text(encoding="utf-8")
    old = 'filename="meshes/gripper_link_collision_g2a.stl"'
    new = f'filename="meshes/{MOVING_MESH_NAME}"'
    if old not in text:
        raise RuntimeError(f"expected collision mesh reference not found in {dst_urdf}")
    text = text.replace(old, new, 1)
    text = _insert_counter_link_and_joint(text, f"meshes/{COUNTER_MESH_NAME}", counter_origin_link5_m)
    dst_urdf.write_text(text, encoding="utf-8")

    moving_vertices = _load_ascii_stl_vertices_m(moving_mesh)
    counter_vertices = _load_ascii_stl_vertices_m(counter_mesh)
    print("[cube2cm_opposing_jaw_urdf] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_opposing_jaw_urdf] diagnostic_only=YES env_default_edits=NO "
        "chain_defaults_edits=NO p7_training=NO constraint_prim_insertion=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "default_local_assets_edited=NO",
        flush=True,
    )
    print(f"[cube2cm_opposing_jaw_urdf] source_urdf={src_urdf}")
    print(f"[cube2cm_opposing_jaw_urdf] output_urdf={dst_urdf}")
    print(
        f"[cube2cm_opposing_jaw_urdf] selected pose={plan.label} center={_fmt_xyz(plan.center)} "
        f"cube_aabb_min={_fmt_xyz(cube_min)} cube_aabb_max={_fmt_xyz(cube_max)} "
        f"design_close_deg={args.design_close_deg:.2f} jaw_overlap_m={args.jaw_overlap_m:.6f} "
        f"jaw_center_z_m={args.jaw_center_z_m:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_urdf] moving_jaw desired_world_center={_fmt_xyz(moving_world_center)} "
        f"gripper_link_local_center={_fmt_xyz(moving_local_center_m)} "
        f"local_bbox_min={_fmt_xyz(moving_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(moving_vertices.max(axis=0))}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_urdf] counter_jaw desired_world_center={_fmt_xyz(counter_world_center)} "
        f"link5_fixed_origin={_fmt_xyz(counter_origin_link5_m)} "
        f"local_bbox_min={_fmt_xyz(counter_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(counter_vertices.max(axis=0))}",
        flush=True,
    )

    moving_ok_any = False
    counter_ok_any = False
    for close_deg in args.close_deg:
        moving_ok, counter_ok = _print_static_sample(
            float(close_deg),
            plan.q_descend_deg,
            moving_vertices,
            counter_vertices,
            counter_origin_link5_m,
            cube_min,
            cube_max,
            plan.center,
        )
        moving_ok_any = moving_ok_any or moving_ok
        counter_ok_any = counter_ok_any or counter_ok

    success = moving_ok_any and counter_ok_any
    print(
        f"[cube2cm_opposing_jaw_urdf] md5 output_urdf={_md5(dst_urdf)} "
        f"moving_mesh={_md5(moving_mesh)} counter_mesh={_md5(counter_mesh)}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_urdf] static_opposing_pair_plausible={_yes(success)} "
        "NEXT_CONVERSION_REQUIRED=YES NEXT_ISAAC_RUN_REQUIRES_EXPLICIT_APPROVAL=YES",
        flush=True,
    )
    print("[cube2cm_opposing_jaw_urdf] CUBE2CM_OPPOSING_JAW_URDF_PREP_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
