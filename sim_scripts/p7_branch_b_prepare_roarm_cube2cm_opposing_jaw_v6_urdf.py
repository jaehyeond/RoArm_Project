#!/usr/bin/env python3
"""Prepare diagnostic RoArm opposing-jaw v6 assets for the 3cm cube candidate.

This is a local/static asset-prep utility. It does not launch Isaac, train, edit
repo default assets, insert constraints, attach SurfaceGripper, transport, or
release. v6 is the gated 3cm candidate from the static/runtime audit:
v4-like top-center grasp, counter jaw shifted toward runtime contact, and a
thicker counter collision proxy.
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
    GRIPPER_OPEN_DEG,
    HOME_DEG,
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
DEFAULT_OUTPUT_URDF_DIR = Path("/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_collision_urdf")
MOVING_MESH_NAME = "gripper_link_collision_cube2cm_opposing_moving_jaw_v6.stl"
COUNTER_MESH_NAME = "cube2cm_counter_jaw_v6.stl"
COUNTER_LINK_NAME = "cube2cm_counter_jaw_v6_link"
COUNTER_JOINT_NAME = "gripper_link_to_cube2cm_counter_jaw_v6_link"


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


def _separating_axis_gap(axis_gap: np.ndarray) -> float:
    positives = np.asarray(axis_gap, dtype=np.float64)
    positives = positives[positives > 0.0]
    if positives.size == 0:
        return 0.0
    return float(np.min(positives))


def _insert_counter_link_and_joint(text: str, counter_mesh_rel: str, origin_gripper_m: np.ndarray) -> str:
    if COUNTER_LINK_NAME in text or COUNTER_JOINT_NAME in text:
        raise RuntimeError("counter jaw v6 link/joint already present in URDF copy")
    link_block = f"""
<!-- Diagnostic-only gripper-mounted opposing counter jaw v6 for 3cm cube tests. -->
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
<!-- Diagnostic-only gripper-mounted opposing counter jaw v6; not a chain-default constraint. -->
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


def _mesh_stats(label: str, mesh_name: str, vertices: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray) -> dict[str, object]:
    mesh_min, mesh_max = _aabb(vertices)
    overlap = _aabb_overlap(mesh_min, mesh_max, cube_min, cube_max)
    axis_gap = _axis_gap(mesh_min, mesh_max, cube_min, cube_max)
    contact = bool(np.all(overlap > 0.0))
    vertices_inside = _points_inside_aabb(vertices, cube_min, cube_max)
    sep_gap = _separating_axis_gap(axis_gap)
    print(
        f"[cube2cm_opposing_jaw_v6_urdf] static_sample label={label} mesh={mesh_name} "
        f"aabb_min={_fmt_xyz(mesh_min)} aabb_max={_fmt_xyz(mesh_max)} "
        f"overlap_m={_fmt_xyz(overlap)} axis_gap_m={_fmt_xyz(axis_gap)} "
        f"separating_axis_gap_m={sep_gap:.6f} aabb_contact={_yes(contact)} "
        f"vertices_inside_cube={vertices_inside}",
        flush=True,
    )
    return {"contact": contact, "overlap": overlap, "vertices_inside": int(vertices_inside), "separating_axis_gap_m": sep_gap}


def _plan_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        object_size_m=args.object_size_m,
        pose_label="v6_3cm_candidate",
        object_xy=None,
        yaw_deg=float(args.yaw_deg),
        grasp_name="top_center",
        normalized_grasp=[0.0, 0.0, 0.5],
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_urdf_dir", type=Path, default=DEFAULT_SOURCE_URDF_DIR)
    ap.add_argument("--output_urdf_dir", type=Path, default=DEFAULT_OUTPUT_URDF_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.030, 0.030, 0.030])
    ap.add_argument("--object_center_m", nargs=3, type=float, default=[0.21369617, -0.19571920, 0.015000])
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[26.0])
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.800)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0050, 0.008])
    ap.add_argument("--moving_close_overlap_m", type=float, default=-0.0015)
    ap.add_argument("--counter_open_clearance_m", type=float, default=0.00075)
    ap.add_argument("--counter_x_shift_mm", type=float, default=1.0)
    ap.add_argument("--counter_y_shift_mm", type=float, default=5.0)
    ap.add_argument("--jaw_center_z_m", type=float, default=0.017)
    ap.add_argument("--open_clearance_gate_m", type=float, default=0.0005)
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
    if not np.allclose(object_size, [0.030, 0.030, 0.030], atol=1.0e-12):
        raise ValueError("v6 prep is intentionally gated to object_size_m=(0.030,0.030,0.030)")
    if abs(float(args.yaw_deg)) > 1.0e-12:
        raise ValueError("v6 prep is intentionally gated to yaw_deg=0")
    if np.any(moving_size <= 0.0) or np.any(counter_size <= 0.0):
        raise ValueError("jaw sizes must be positive")

    plan = _build_plan_from_center(_plan_args(args), center, "v6_3cm_candidate")
    if not plan.descend_ik_ok:
        raise RuntimeError("descend IK did not converge for v6 diagnostic pose")
    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)

    cube_min, cube_max = _aabb(_box_vertices(center, object_size, float(args.yaw_deg)))
    cube_half_y = object_size[1] * 0.5
    base_counter_y = -cube_half_y - float(args.counter_open_clearance_m) - 0.0015 * 0.5
    moving_obj_center = np.array([0.0, cube_half_y - float(args.moving_close_overlap_m), 0.0020], dtype=np.float64)
    counter_obj_center = np.array(
        [
            float(args.counter_x_shift_mm) / 1000.0,
            base_counter_y + float(args.counter_y_shift_mm) / 1000.0,
            0.0020,
        ],
        dtype=np.float64,
    )
    moving_world_center = center + moving_obj_center
    counter_world_center = center + counter_obj_center

    inv_gripper = np.linalg.inv(_gripper_transform(q_design))
    moving_local_center_m = (inv_gripper @ np.array([*moving_world_center, 1.0], dtype=np.float64))[:3]
    counter_origin_gripper_m = (inv_gripper @ np.array([*counter_world_center, 1.0], dtype=np.float64))[:3]

    meshes_dir = dst / "meshes"
    moving_mesh = meshes_dir / MOVING_MESH_NAME
    counter_mesh = meshes_dir / COUNTER_MESH_NAME
    moving_mesh.write_text(_box_stl_text("cube2cm_opposing_moving_jaw_v6", moving_local_center_m, moving_size), encoding="utf-8")
    counter_mesh.write_text(_box_stl_text("cube2cm_counter_jaw_v6", np.zeros(3, dtype=np.float64), counter_size), encoding="utf-8")

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

    print("[cube2cm_opposing_jaw_v6_urdf] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_opposing_jaw_v6_urdf] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "p7_training=NO dataset_generation=NO constraint_prim_insertion=NO surface_gripper=NO "
        "attached_transport=NO transport_target=NO release_marker=NO default_local_assets_edited=NO "
        "counter_mount=gripper_link close_dependent_counter=YES object_size_priority=0.030_first",
        flush=True,
    )
    print(f"[cube2cm_opposing_jaw_v6_urdf] source_urdf={src_urdf}")
    print(f"[cube2cm_opposing_jaw_v6_urdf] output_urdf={dst_urdf}")
    print(
        f"[cube2cm_opposing_jaw_v6_urdf] selected center={_fmt_xyz(center)} yaw_deg={args.yaw_deg:.2f} "
        f"normalized_grasp=([+0.000,+0.000,+0.500]) object_size_m={_fmt_xyz(object_size)} "
        f"world_grasp={_fmt_xyz(plan.world_grasp)} descend_tcp={_fmt_xyz(plan.descend_tcp)} "
        f"counter_x_shift_mm={args.counter_x_shift_mm:+.3f} counter_y_shift_mm={args.counter_y_shift_mm:+.3f} "
        f"counter_thickness_y_mm={counter_size[1] * 1000.0:.3f} candidate_patch_margin_m=0.0030",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v6_urdf] moving_jaw desired_world_center={_fmt_xyz(moving_world_center)} "
        f"moving_obj_center={_fmt_xyz(moving_obj_center)} gripper_link_local_center={_fmt_xyz(moving_local_center_m)} "
        f"local_bbox_min={_fmt_xyz(moving_vertices.min(axis=0))} local_bbox_max={_fmt_xyz(moving_vertices.max(axis=0))}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v6_urdf] counter_jaw desired_world_center={_fmt_xyz(counter_world_center)} "
        f"counter_obj_center={_fmt_xyz(counter_obj_center)} gripper_link_fixed_origin={_fmt_xyz(counter_origin_gripper_m)} "
        f"local_bbox_min={_fmt_xyz(counter_vertices.min(axis=0))} local_bbox_max={_fmt_xyz(counter_vertices.max(axis=0))}",
        flush=True,
    )

    open_contacts = 0
    open_clearance_bad = 0
    q_seed = HOME_DEG.copy()
    open_waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    for idx, waypoint in enumerate(open_waypoints, start=1):
        q_open, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        print(
            f"[cube2cm_opposing_jaw_v6_urdf] open_waypoint index={idx:03d}/{len(open_waypoints):03d} "
            f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f}",
            flush=True,
        )
        moving_world = _transform_points(_gripper_transform(q_open), moving_vertices)
        counter_world = _transform_points(_gripper_transform(q_open) @ _translation(counter_origin_gripper_m), counter_vertices)
        for mesh_name, vertices in (("moving", moving_world), ("counter", counter_world)):
            stats = _mesh_stats(f"open_wp{idx:03d}", mesh_name, vertices, cube_min, cube_max)
            if stats["contact"] or stats["vertices_inside"] > 0:
                open_contacts += 1
            elif stats["separating_axis_gap_m"] < args.open_clearance_gate_m:
                open_clearance_bad += 1

    q_close = plan.q_descend_deg.copy()
    q_close[5] = float(args.design_close_deg)
    moving_close = _transform_points(_gripper_transform(q_close), moving_vertices)
    counter_close = _transform_points(_gripper_transform(q_close) @ _translation(counter_origin_gripper_m), counter_vertices)
    moving_stats = _mesh_stats("close_26.00", "moving", moving_close, cube_min, cube_max)
    counter_stats = _mesh_stats("close_26.00", "counter", counter_close, cube_min, cube_max)

    open_descent_clearance = open_contacts == 0 and open_clearance_bad == 0
    design_moving_overlap_y_m = float(np.asarray(moving_stats["overlap"], dtype=np.float64)[1])
    design_counter_overlap_y_m = float(np.asarray(counter_stats["overlap"], dtype=np.float64)[1])
    design_min_overlap_ok = (
        design_moving_overlap_y_m >= float(args.design_min_overlap_y_m)
        and design_counter_overlap_y_m >= float(args.design_min_overlap_y_m)
    )
    success = open_descent_clearance and bool(moving_stats["contact"]) and bool(counter_stats["contact"]) and design_min_overlap_ok

    print(
        f"[cube2cm_opposing_jaw_v6_urdf] md5 output_urdf={_md5(dst_urdf)} "
        f"moving_mesh={_md5(moving_mesh)} counter_mesh={_md5(counter_mesh)}",
        flush=True,
    )
    print(
        f"[cube2cm_opposing_jaw_v6_urdf] gates open_descent_clearance={_yes(open_descent_clearance)} "
        f"open_contacts={open_contacts} open_clearance_bad={open_clearance_bad} "
        f"moving_contact_at_design_close={_yes(bool(moving_stats['contact']))} "
        f"counter_contact_at_design_close={_yes(bool(counter_stats['contact']))} "
        f"design_moving_overlap_y_m={design_moving_overlap_y_m:.6f} "
        f"design_counter_overlap_y_m={design_counter_overlap_y_m:.6f} "
        f"design_min_overlap_ok={_yes(design_min_overlap_ok)} "
        f"v6_candidate_static_plausible={_yes(success)} "
        "NEXT_CONVERSION_APPROVED=YES NEXT_ISAAC_RUNTIME_REQUIRES_SEPARATE_APPROVAL=YES",
        flush=True,
    )
    print("[cube2cm_opposing_jaw_v6_urdf] CUBE2CM_OPPOSING_JAW_V6_URDF_PREP_DONE=YES")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
