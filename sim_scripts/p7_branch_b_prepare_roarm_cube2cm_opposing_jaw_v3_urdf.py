#!/usr/bin/env python3
"""Prepare diagnostic RoArm opposing-jaw v3 assets for 2cm cube close tests.

This is a local/static asset-prep utility. It does not launch Isaac, train, edit
repo default assets, insert constraints, attach SurfaceGripper, transport, or
release.

v3 changes one thing relative to v2: the counter jaw is fixed under
``gripper_link`` instead of ``link5``. That makes the counter close-dependent
with the gripper joint, so static checks can ask whether open descent stays clear
while the counter becomes an opposing contact candidate near the design close.
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
    _points_inside_aabb,
    _transform_points,
)
from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    HOME_DEG,
    GRIPPER_OPEN_DEG,
    _build_plan,
    _build_plan_from_center,
    _norm,
    _solve_q,
    _yes,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import (  # noqa: E402
    _box_stl_text,
    _load_ascii_stl_vertices_m,
    _translation,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v2_urdf import _open_descent_waypoints  # noqa: E402


DEFAULT_SOURCE_URDF_DIR = REPO / "local_assets/roarm_m3/urdf"
DEFAULT_OUTPUT_URDF_DIR = Path("/tmp/p7_branch_b_cube2cm_opposing_jaw_v3_collision_urdf")
MOVING_MESH_NAME = "gripper_link_collision_cube2cm_opposing_moving_jaw_v3.stl"
COUNTER_MESH_NAME = "cube2cm_counter_jaw_v3.stl"
COUNTER_LINK_NAME = "cube2cm_counter_jaw_v3_link"
COUNTER_JOINT_NAME = "gripper_link_to_cube2cm_counter_jaw_v3_link"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_xyz_attr(value: np.ndarray) -> str:
    return f"{value[0]:.9f} {value[1]:.9f} {value[2]:.9f}"


def _axis_gap(mesh_min: np.ndarray, mesh_max: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(cube_min - mesh_max, mesh_min - cube_max), 0.0)


def _aabb_contact(overlap: np.ndarray) -> bool:
    return bool(np.all(np.asarray(overlap, dtype=np.float64) > 0.0))


def _separating_axis_gap(axis_gap: np.ndarray) -> float:
    positives = np.asarray(axis_gap, dtype=np.float64)
    positives = positives[positives > 0.0]
    if positives.size == 0:
        return 0.0
    return float(np.min(positives))


def _insert_counter_link_and_joint(text: str, counter_mesh_rel: str, origin_gripper_m: np.ndarray) -> str:
    if COUNTER_LINK_NAME in text or COUNTER_JOINT_NAME in text:
        raise RuntimeError("counter jaw v3 link/joint already present in URDF copy")
    link_block = f"""
<!-- Diagnostic-only gripper-mounted opposing counter jaw v3 for 2cm cube local grasp tests. -->
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
<!-- Diagnostic-only gripper-mounted opposing counter jaw v3; not a chain-default constraint. -->
<joint name="{COUNTER_JOINT_NAME}" type="fixed">
  <origin xyz="{_format_xyz_attr(origin_gripper_m)}" rpy="0 0 0"/>
  <parent link="gripper_link"/>
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


def _mesh_stats(
    label: str,
    mesh_name: str,
    world_vertices: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
) -> dict[str, object]:
    mesh_min, mesh_max = _aabb(world_vertices)
    overlap = _aabb_overlap(mesh_min, mesh_max, cube_min, cube_max)
    axis_gap = _axis_gap(mesh_min, mesh_max, cube_min, cube_max)
    contact = _aabb_contact(overlap)
    vertices_inside = _points_inside_aabb(world_vertices, cube_min, cube_max)
    sep_gap = _separating_axis_gap(axis_gap)
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] static_sample label={label} mesh={mesh_name} "
        f"aabb_min={_fmt_xyz(mesh_min)} aabb_max={_fmt_xyz(mesh_max)} "
        f"overlap_m={_fmt_xyz(overlap)} axis_gap_m={_fmt_xyz(axis_gap)} "
        f"separating_axis_gap_m={sep_gap:.6f} aabb_contact={_yes(contact)} "
        f"vertices_inside_cube={vertices_inside}",
        flush=True,
    )
    return {
        "contact": contact,
        "vertices_inside": int(vertices_inside),
        "separating_axis_gap_m": sep_gap,
    }


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
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.800)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--moving_close_overlap_m", type=float, default=0.0005)
    ap.add_argument("--counter_open_clearance_m", type=float, default=0.0025)
    ap.add_argument("--jaw_center_z_m", type=float, default=0.012)
    ap.add_argument("--open_clearance_gate_m", type=float, default=0.0005)
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
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    if np.any(object_size <= 0.0) or np.any(moving_size <= 0.0) or np.any(counter_size <= 0.0):
        raise ValueError("object and jaw sizes must be positive")
    if args.moving_close_overlap_m <= 0.0:
        raise ValueError("moving_close_overlap_m must be positive")
    if args.counter_open_clearance_m <= 0.0:
        raise ValueError("counter_open_clearance_m must be positive")

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
            plan.center[1] + object_size[1] / 2.0 - args.moving_close_overlap_m,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )
    counter_world_center = np.array(
        [
            plan.center[0],
            plan.center[1] - object_size[1] / 2.0 - args.counter_open_clearance_m - counter_size[1] / 2.0,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )

    moving_local_center = np.linalg.inv(_gripper_transform(q_design)) @ np.array(
        [moving_world_center[0], moving_world_center[1], moving_world_center[2], 1.0],
        dtype=np.float64,
    )
    counter_origin_gripper = np.linalg.inv(_gripper_transform(q_design)) @ np.array(
        [counter_world_center[0], counter_world_center[1], counter_world_center[2], 1.0],
        dtype=np.float64,
    )
    moving_local_center_m = moving_local_center[:3]
    counter_origin_gripper_m = counter_origin_gripper[:3]

    meshes_dir = dst / "meshes"
    moving_mesh = meshes_dir / MOVING_MESH_NAME
    counter_mesh = meshes_dir / COUNTER_MESH_NAME
    moving_mesh.write_text(
        _box_stl_text("cube2cm_opposing_moving_jaw_v3", moving_local_center_m, moving_size),
        encoding="utf-8",
    )
    counter_mesh.write_text(
        _box_stl_text("cube2cm_counter_jaw_v3", np.zeros(3, dtype=np.float64), counter_size),
        encoding="utf-8",
    )

    text = dst_urdf.read_text(encoding="utf-8")
    old = 'filename="meshes/gripper_link_collision_g2a.stl"'
    new = f'filename="meshes/{MOVING_MESH_NAME}"'
    if old not in text:
        raise RuntimeError(f"expected collision mesh reference not found in {dst_urdf}")
    text = text.replace(old, new, 1)
    text = _insert_counter_link_and_joint(text, f"meshes/{COUNTER_MESH_NAME}", counter_origin_gripper_m)
    dst_urdf.write_text(text, encoding="utf-8")

    moving_vertices = _load_ascii_stl_vertices_m(moving_mesh)
    counter_vertices = _load_ascii_stl_vertices_m(counter_mesh)

    print("[cube2cm_opposing_jaw_v3_urdf] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_opposing_jaw_v3_urdf] diagnostic_only=YES env_default_edits=NO "
        "chain_defaults_edits=NO p7_training=NO constraint_prim_insertion=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "default_local_assets_edited=NO counter_mount=gripper_link close_dependent_counter=YES",
        flush=True,
    )
    print(f"[cube2cm_opposing_jaw_v3_urdf] source_urdf={src_urdf}")
    print(f"[cube2cm_opposing_jaw_v3_urdf] output_urdf={dst_urdf}")
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] selected pose={plan.label} center={_fmt_xyz(plan.center)} "
        f"cube_aabb_min={_fmt_xyz(cube_min)} cube_aabb_max={_fmt_xyz(cube_max)} "
        f"design_close_deg={args.design_close_deg:.2f} moving_close_overlap_m={args.moving_close_overlap_m:.6f} "
        f"counter_open_clearance_m={args.counter_open_clearance_m:.6f} jaw_center_z_m={args.jaw_center_z_m:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] moving_jaw desired_world_center={_fmt_xyz(moving_world_center)} "
        f"gripper_link_local_center={_fmt_xyz(moving_local_center_m)} "
        f"local_bbox_min={_fmt_xyz(moving_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(moving_vertices.max(axis=0))}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] counter_jaw desired_world_center={_fmt_xyz(counter_world_center)} "
        f"gripper_link_fixed_origin={_fmt_xyz(counter_origin_gripper_m)} "
        f"local_bbox_min={_fmt_xyz(counter_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(counter_vertices.max(axis=0))}",
        flush=True,
    )

    open_contacts = 0
    open_clearance_bad = 0
    q_seed = HOME_DEG.copy()
    open_waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] open_descent_static_check waypoints={len(open_waypoints)} "
        f"open_clearance_gate_m={args.open_clearance_gate_m:.6f}",
        flush=True,
    )
    for idx, waypoint in enumerate(open_waypoints, start=1):
        q_open, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        moving_world = _transform_points(_gripper_transform(q_open), moving_vertices)
        counter_world = _transform_points(_gripper_transform(q_open) @ _translation(counter_origin_gripper_m), counter_vertices)
        print(
            f"[cube2cm_opposing_jaw_v3_urdf] open_waypoint index={idx:03d}/{len(open_waypoints):03d} "
            f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f}",
            flush=True,
        )
        for mesh_name, vertices in (("moving", moving_world), ("counter", counter_world)):
            stats = _mesh_stats(f"open_wp{idx:03d}", mesh_name, vertices, cube_min, cube_max)
            if stats["contact"] or stats["vertices_inside"] > 0:
                open_contacts += 1
            elif stats["separating_axis_gap_m"] < args.open_clearance_gate_m:
                open_clearance_bad += 1

    moving_contact_any = False
    counter_contact_at_design = False
    counter_contact_any = False
    for close_deg in args.close_deg:
        q_close = plan.q_descend_deg.copy()
        q_close[5] = float(close_deg)
        moving_world = _transform_points(_gripper_transform(q_close), moving_vertices)
        counter_world = _transform_points(_gripper_transform(q_close) @ _translation(counter_origin_gripper_m), counter_vertices)
        print(
            f"[cube2cm_opposing_jaw_v3_urdf] close_static_check close_deg={close_deg:.2f}",
            flush=True,
        )
        moving_stats = _mesh_stats(f"close_{close_deg:.2f}", "moving", moving_world, cube_min, cube_max)
        counter_stats = _mesh_stats(f"close_{close_deg:.2f}", "counter", counter_world, cube_min, cube_max)
        moving_contact_any = moving_contact_any or bool(moving_stats["contact"])
        counter_contact_any = counter_contact_any or bool(counter_stats["contact"])
        if abs(float(close_deg) - float(args.design_close_deg)) < 1e-6:
            counter_contact_at_design = bool(counter_stats["contact"])

    open_descent_clearance = open_contacts == 0 and open_clearance_bad == 0
    close_moving_contact = moving_contact_any
    success = open_descent_clearance and close_moving_contact and counter_contact_at_design

    print(
        f"[cube2cm_opposing_jaw_v3_urdf] md5 output_urdf={_md5(dst_urdf)} "
        f"moving_mesh={_md5(moving_mesh)} counter_mesh={_md5(counter_mesh)}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v3_urdf] gates open_descent_clearance={_yes(open_descent_clearance)} "
        f"open_contacts={open_contacts} open_clearance_bad={open_clearance_bad} "
        f"close_moving_contact_candidate={_yes(close_moving_contact)} "
        f"counter_contact_any={_yes(counter_contact_any)} "
        f"counter_contact_at_design_close={_yes(counter_contact_at_design)} "
        f"static_opposing_pair_plausible={_yes(success)} "
        "NEXT_CONVERSION_REQUIRED=YES NEXT_ISAAC_RUN_REQUIRES_EXPLICIT_APPROVAL=YES",
        flush=True,
    )
    print("[cube2cm_opposing_jaw_v3_urdf] CUBE2CM_OPPOSING_JAW_V3_URDF_PREP_DONE=YES")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
