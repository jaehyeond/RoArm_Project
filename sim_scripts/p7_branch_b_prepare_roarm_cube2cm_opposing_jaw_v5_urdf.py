#!/usr/bin/env python3
"""Prepare diagnostic RoArm opposing-jaw v5 assets for 2cm cube close tests.

This is a local/static asset-prep utility. It does not launch Isaac, train, edit
repo default assets, insert constraints, attach SurfaceGripper, transport, or
release.

v5 differs from v4 by authoring the jaw pair in the cube object frame and by
choosing the canonical yaw/TCP offset from a fixture-frame normal audit. The
candidate is still static-only; physics validation requires a separate explicit
approval and run.
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
    _transform_points,
)
from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    GRIPPER_OPEN_DEG,
    HOME_DEG,
    _build_plan_from_center,
    _norm,
    _rot_z,
    _solve_q,
    _yes,
)
from p7_branch_b_cube2cm_v5_fixture_frame_static_audit import _contact, _to_object_frame  # noqa: E402
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import (  # noqa: E402
    _box_stl_text,
    _load_ascii_stl_vertices_m,
    _translation,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v2_urdf import _open_descent_waypoints  # noqa: E402


DEFAULT_SOURCE_URDF_DIR = REPO / "local_assets/roarm_m3/urdf"
DEFAULT_OUTPUT_URDF_DIR = Path("/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_collision_urdf")
MOVING_MESH_NAME = "gripper_link_collision_cube2cm_opposing_moving_jaw_v5.stl"
COUNTER_MESH_NAME = "cube2cm_counter_jaw_v5.stl"
COUNTER_LINK_NAME = "cube2cm_counter_jaw_v5_link"
COUNTER_JOINT_NAME = "gripper_link_to_cube2cm_counter_jaw_v5_link"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_xyz_attr(value: np.ndarray) -> str:
    return f"{value[0]:.9f} {value[1]:.9f} {value[2]:.9f}"


def _insert_counter_link_and_joint(text: str, counter_mesh_rel: str, origin_gripper_m: np.ndarray) -> str:
    if COUNTER_LINK_NAME in text or COUNTER_JOINT_NAME in text:
        raise RuntimeError("counter jaw v5 link/joint already present in URDF copy")
    link_block = f"""
<!-- Diagnostic-only gripper-mounted opposing counter jaw v5 for object-frame 2cm cube tests. -->
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
<!-- Diagnostic-only gripper-mounted opposing counter jaw v5; not a chain-default constraint. -->
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


def _plan_args(args: argparse.Namespace) -> argparse.Namespace:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    return argparse.Namespace(
        object_size_m=args.object_size_m,
        pose_label="v5_fixture_frame",
        object_xy=None,
        object_center_z=float(args.object_center_m[2]),
        yaw_deg=float(args.yaw_deg),
        grasp_name="top_center",
        normalized_grasp=[
            float(args.tcp_offset_obj_mm[0]) / (object_size[0] * 1000.0),
            float(args.tcp_offset_obj_mm[1]) / (object_size[1] * 1000.0),
            0.5,
        ],
        approach_clearance_m=args.approach_clearance_m,
        grasp_surface_margin_m=args.grasp_surface_margin_m,
        lift_delta_m=args.lift_delta_m,
        close_deg=args.close_deg,
        target_error_gate_m=args.target_error_gate_m,
        ik_tol_mm=args.ik_tol_mm,
        ik_max_iter=args.ik_max_iter,
        max_tcp_step_m=args.max_tcp_step_m,
        command_resample_fraction=args.command_resample_fraction,
    )


def _candidate_centers(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    cube_half_y = object_size[1] * 0.5
    moving_half_y = moving_size[1] * 0.5
    counter_half_y = counter_size[1] * 0.5
    penetration = float(args.design_penetration_m)
    jaw_x = float(args.jaw_center_obj_m[0])
    jaw_z = float(args.jaw_center_obj_m[2])
    moving_obj_center = np.array([jaw_x, cube_half_y + moving_half_y - penetration, jaw_z], dtype=np.float64)
    counter_obj_center = np.array([jaw_x, -cube_half_y - counter_half_y + penetration, jaw_z], dtype=np.float64)
    return moving_obj_center, counter_obj_center


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_urdf_dir", type=Path, default=DEFAULT_SOURCE_URDF_DIR)
    ap.add_argument("--output_urdf_dir", type=Path, default=DEFAULT_OUTPUT_URDF_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--object_center_m", nargs=3, type=float, default=[0.21369617, -0.19571920, 0.010000])
    ap.add_argument("--yaw_deg", type=float, default=50.0)
    ap.add_argument("--tcp_offset_obj_mm", nargs=2, type=float, default=[3.0, -3.0])
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
    ap.add_argument("--jaw_center_obj_m", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--design_penetration_m", type=float, default=0.0015)
    ap.add_argument("--open_clearance_gate_m", type=float, default=0.0005)
    ap.add_argument("--normal_alignment_gate", type=float, default=0.98)
    ap.add_argument("--tangential_slip_gate", type=float, default=0.10)
    ap.add_argument("--vertical_component_gate", type=float, default=0.10)
    ap.add_argument("--design_balance_gate_m", type=float, default=0.00025)
    ap.add_argument("--design_min_overlap_y_m", type=float, default=0.0010)
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
    center = np.asarray(args.object_center_m, dtype=np.float64)
    plan = _build_plan_from_center(_plan_args(args), center, "v5_fixture_frame_zoverride")
    if not plan.descend_ik_ok:
        raise RuntimeError("descend IK did not converge for v5 diagnostic pose")

    rot = _rot_z(float(args.yaw_deg))
    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    moving_obj_center, counter_obj_center = _candidate_centers(args)
    moving_world_center = center + rot @ moving_obj_center
    counter_world_center = center + rot @ counter_obj_center

    inv_gripper = np.linalg.inv(_gripper_transform(q_design))
    moving_local_center_m = (inv_gripper @ np.array([*moving_world_center, 1.0], dtype=np.float64))[:3]
    counter_origin_gripper_m = (inv_gripper @ np.array([*counter_world_center, 1.0], dtype=np.float64))[:3]

    meshes_dir = dst / "meshes"
    moving_mesh = meshes_dir / MOVING_MESH_NAME
    counter_mesh = meshes_dir / COUNTER_MESH_NAME
    moving_mesh.write_text(
        _box_stl_text("cube2cm_opposing_moving_jaw_v5", moving_local_center_m, moving_size),
        encoding="utf-8",
    )
    counter_mesh.write_text(
        _box_stl_text("cube2cm_counter_jaw_v5", np.zeros(3, dtype=np.float64), counter_size),
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
    cube_min_obj = -0.5 * object_size
    cube_max_obj = 0.5 * object_size

    gripper = _gripper_transform(q_design)
    closing_axis_world = gripper[:3, 1]
    closing_xy = closing_axis_world.copy()
    closing_xy[2] = 0.0
    closing_xy /= max(_norm(closing_xy), 1.0e-12)
    object_y_world = rot[:, 1]
    object_x_world = rot[:, 0]
    normal_alignment = abs(float(np.dot(closing_xy, object_y_world)))
    tangential_slip = abs(float(np.dot(closing_xy, object_x_world)))
    vertical_component = abs(float(closing_axis_world[2]))

    print("[cube2cm_opposing_jaw_v5_urdf] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_opposing_jaw_v5_urdf] diagnostic_only=YES env_default_edits=NO "
        "chain_defaults_edits=NO p7_training=NO constraint_prim_insertion=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "default_local_assets_edited=NO counter_mount=gripper_link close_dependent_counter=YES "
        "object_frame_jaw_authoring=YES",
        flush=True,
    )
    print(f"[cube2cm_opposing_jaw_v5_urdf] source_urdf={src_urdf}")
    print(f"[cube2cm_opposing_jaw_v5_urdf] output_urdf={dst_urdf}")
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] selected center={_fmt_xyz(center)} yaw_deg={args.yaw_deg:.2f} "
        f"tcp_offset_obj_mm=([{args.tcp_offset_obj_mm[0]:+.3f}, {args.tcp_offset_obj_mm[1]:+.3f}]) "
        f"world_grasp={_fmt_xyz(plan.world_grasp)} descend_tcp={_fmt_xyz(plan.descend_tcp)} "
        f"moving_obj_center={_fmt_xyz(moving_obj_center)} counter_obj_center={_fmt_xyz(counter_obj_center)} "
        f"design_penetration_m={args.design_penetration_m:.6f}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] fixture_normal normal_alignment={normal_alignment:.6f} "
        f"tangential_slip={tangential_slip:.6f} vertical_component={vertical_component:.6f} "
        f"closing_axis_world={_fmt_xyz(closing_axis_world)} object_y_world={_fmt_xyz(object_y_world)}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] moving_jaw desired_world_center={_fmt_xyz(moving_world_center)} "
        f"gripper_link_local_center={_fmt_xyz(moving_local_center_m)} "
        f"local_bbox_min={_fmt_xyz(moving_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(moving_vertices.max(axis=0))}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] counter_jaw desired_world_center={_fmt_xyz(counter_world_center)} "
        f"gripper_link_fixed_origin={_fmt_xyz(counter_origin_gripper_m)} "
        f"local_bbox_min={_fmt_xyz(counter_vertices.min(axis=0))} "
        f"local_bbox_max={_fmt_xyz(counter_vertices.max(axis=0))}",
        flush=True,
    )

    open_contacts = 0
    open_clearance_bad = 0
    q_seed = HOME_DEG.copy()
    open_waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    for idx, waypoint in enumerate(open_waypoints, start=1):
        q_open, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        moving_world = _transform_points(_gripper_transform(q_open), moving_vertices)
        counter_world = _transform_points(_gripper_transform(q_open) @ _translation(counter_origin_gripper_m), counter_vertices)
        print(
            f"[cube2cm_opposing_jaw_v5_urdf] open_waypoint index={idx:03d}/{len(open_waypoints):03d} "
            f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f}",
            flush=True,
        )
        for mesh_name, vertices in (("moving", moving_world), ("counter", counter_world)):
            stats = _contact(_to_object_frame(vertices, center, float(args.yaw_deg)), cube_min_obj, cube_max_obj)
            gap = np.asarray(stats["gap"], dtype=np.float64)
            positive = gap[gap > 0.0]
            sep_gap = 0.0 if positive.size == 0 else float(positive.min())
            print(
                f"[cube2cm_opposing_jaw_v5_urdf] static_sample label=open_wp{idx:03d} mesh={mesh_name} "
                f"center_obj={_fmt_xyz(np.asarray(stats['center']))} overlap_obj_m={_fmt_xyz(np.asarray(stats['overlap']))} "
                f"gap_obj_m={_fmt_xyz(gap)} separating_axis_gap_m={sep_gap:.6f} "
                f"obb_contact={_yes(bool(stats['contact']))}",
                flush=True,
            )
            if bool(stats["contact"]):
                open_contacts += 1
            elif sep_gap < args.open_clearance_gate_m:
                open_clearance_bad += 1

    moving_contact_at_design = False
    counter_contact_at_design = False
    design_moving_overlap_y_m = 0.0
    design_counter_overlap_y_m = 0.0
    for close_deg in args.close_deg:
        q_close = plan.q_descend_deg.copy()
        q_close[5] = float(close_deg)
        moving_world = _transform_points(_gripper_transform(q_close), moving_vertices)
        counter_world = _transform_points(_gripper_transform(q_close) @ _translation(counter_origin_gripper_m), counter_vertices)
        moving_stats = _contact(_to_object_frame(moving_world, center, float(args.yaw_deg)), cube_min_obj, cube_max_obj)
        counter_stats = _contact(_to_object_frame(counter_world, center, float(args.yaw_deg)), cube_min_obj, cube_max_obj)
        print(
            f"[cube2cm_opposing_jaw_v5_urdf] close_static_check close_deg={close_deg:.2f} "
            f"moving_contact={_yes(bool(moving_stats['contact']))} counter_contact={_yes(bool(counter_stats['contact']))} "
            f"moving_center_obj={_fmt_xyz(np.asarray(moving_stats['center']))} "
            f"counter_center_obj={_fmt_xyz(np.asarray(counter_stats['center']))} "
            f"moving_overlap_obj_m={_fmt_xyz(np.asarray(moving_stats['overlap']))} "
            f"counter_overlap_obj_m={_fmt_xyz(np.asarray(counter_stats['overlap']))}",
            flush=True,
        )
        if abs(float(close_deg) - float(args.design_close_deg)) < 1e-6:
            moving_contact_at_design = bool(moving_stats["contact"])
            counter_contact_at_design = bool(counter_stats["contact"])
            design_moving_overlap_y_m = float(np.asarray(moving_stats["overlap"], dtype=np.float64)[1])
            design_counter_overlap_y_m = float(np.asarray(counter_stats["overlap"], dtype=np.float64)[1])

    open_descent_clearance = open_contacts == 0 and open_clearance_bad == 0
    design_balance_abs_m = abs(design_moving_overlap_y_m - design_counter_overlap_y_m)
    design_balance_ok = design_balance_abs_m <= args.design_balance_gate_m
    design_min_overlap_ok = (
        design_moving_overlap_y_m >= args.design_min_overlap_y_m
        and design_counter_overlap_y_m >= args.design_min_overlap_y_m
    )
    normal_ok = (
        normal_alignment >= args.normal_alignment_gate
        and tangential_slip <= args.tangential_slip_gate
        and vertical_component <= args.vertical_component_gate
    )
    success = (
        open_descent_clearance
        and normal_ok
        and moving_contact_at_design
        and counter_contact_at_design
        and design_balance_ok
        and design_min_overlap_ok
    )
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] md5 output_urdf={_md5(dst_urdf)} "
        f"moving_mesh={_md5(moving_mesh)} counter_mesh={_md5(counter_mesh)}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v5_urdf] gates open_descent_clearance={_yes(open_descent_clearance)} "
        f"open_contacts={open_contacts} open_clearance_bad={open_clearance_bad} "
        f"fixture_normal_ok={_yes(normal_ok)} normal_alignment={normal_alignment:.6f} "
        f"tangential_slip={tangential_slip:.6f} vertical_component={vertical_component:.6f} "
        f"moving_contact_at_design_close={_yes(moving_contact_at_design)} "
        f"counter_contact_at_design_close={_yes(counter_contact_at_design)} "
        f"design_moving_overlap_y_m={design_moving_overlap_y_m:.6f} "
        f"design_counter_overlap_y_m={design_counter_overlap_y_m:.6f} "
        f"design_balance_abs_m={design_balance_abs_m:.6f} design_balance_ok={_yes(design_balance_ok)} "
        f"design_min_overlap_ok={_yes(design_min_overlap_ok)} "
        f"static_fixture_pair_plausible={_yes(success)} "
        "NEXT_CONVERSION_REQUIRES_EXPLICIT_APPROVAL=YES NEXT_ISAAC_RUN_REQUIRES_EXPLICIT_APPROVAL=YES",
        flush=True,
    )
    print("[cube2cm_opposing_jaw_v5_urdf] CUBE2CM_OPPOSING_JAW_V5_URDF_PREP_DONE=YES")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
