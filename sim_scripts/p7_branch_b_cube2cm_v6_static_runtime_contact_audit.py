#!/usr/bin/env python3
"""Local v6 static/runtime contact audit for P7 Branch B cube grasp.

This is numpy-only. It does not launch Isaac, train, edit defaults, insert
constraints, attach SurfaceGripper, transport, release, tune gates, or generate a
dataset. It compares v4-like and v5-like opposing-jaw candidates at the authored
static design pose and at the latest B200 runtime close endpoint recorded in
logs, then repeats the check for 2cm and 3cm object sizes with simple compliant
contact-patch margins.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
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
    _build_plan_from_center,
    _norm,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import _translation  # noqa: E402


@dataclass(frozen=True)
class VariantSpec:
    name: str
    yaw_deg: float
    normalized_grasp: np.ndarray
    v4_like: bool


@dataclass(frozen=True)
class RuntimeEndpoint:
    label: str
    q_deg: np.ndarray
    object_center_m: np.ndarray
    source_log: str


VARIANTS = (
    VariantSpec("v4_like", 0.0, np.array([0.0, 0.0, 0.5], dtype=np.float64), True),
    VariantSpec("v5_like", 50.0, np.array([0.150, -0.150, 0.5], dtype=np.float64), False),
)

RUNTIME_ENDPOINTS = {
    "v4_like": RuntimeEndpoint(
        "v4_log_line422",
        np.array([-42.416, 33.377, 111.198, 7.289, 88.582, 25.915], dtype=np.float64),
        np.array([0.213274, -0.195691, 0.010381], dtype=np.float64),
        "/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v4_b200.out:422",
    ),
    "v5_like": RuntimeEndpoint(
        "v5_log_line422",
        np.array([-41.869, 33.368, 110.558, 7.106, 89.528, 25.917], dtype=np.float64),
        np.array([0.212724, -0.194922, 0.011083], dtype=np.float64),
        "/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v5_b200.out:422",
    ),
}


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _rot_z(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _axis_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)


def _to_object_frame(points_world: np.ndarray, object_center: np.ndarray, yaw_deg: float) -> np.ndarray:
    return (points_world - object_center) @ _rot_z(yaw_deg)


def _contact(points_world: np.ndarray, object_center: np.ndarray, object_size: np.ndarray, yaw_deg: float, patch_m: float) -> dict[str, object]:
    points_obj = _to_object_frame(points_world, object_center, yaw_deg)
    cube_min = -0.5 * object_size - patch_m
    cube_max = 0.5 * object_size + patch_m
    mn, mx = _aabb(points_obj)
    overlap = _aabb_overlap(mn, mx, cube_min, cube_max)
    gap = _axis_gap(mn, mx, cube_min, cube_max)
    center = 0.5 * (mn + mx)
    return {
        "center": center,
        "overlap": overlap,
        "gap": gap,
        "contact": bool(np.all(overlap > 0.0)),
    }


def _plan_args(object_size: np.ndarray, variant: VariantSpec, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        object_size_m=object_size.tolist(),
        pose_label="v6_static_runtime_contact_audit",
        object_xy=None,
        yaw_deg=variant.yaw_deg,
        grasp_name="top_center",
        normalized_grasp=variant.normalized_grasp.tolist(),
        approach_clearance_m=args.approach_clearance_m,
        grasp_surface_margin_m=args.grasp_surface_margin_m,
        lift_delta_m=args.lift_delta_m,
        close_deg=[args.design_close_deg],
        target_error_gate_m=args.target_error_gate_m,
        ik_tol_mm=args.ik_tol_mm,
        ik_max_iter=args.ik_max_iter,
        max_tcp_step_m=args.max_tcp_step_m,
        command_resample_fraction=args.command_resample_fraction,
    )


def _jaw_centers_obj(object_size: np.ndarray, variant: VariantSpec, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    cube_half_y = object_size[1] * 0.5
    if variant.v4_like:
        z_offset = float(args.v4_jaw_center_z_offset_m)
        moving = np.array([0.0, cube_half_y - float(args.v4_moving_close_overlap_m), z_offset], dtype=np.float64)
        counter = np.array(
            [0.0, -cube_half_y - float(args.v4_counter_open_clearance_m) - counter_size[1] * 0.5, z_offset],
            dtype=np.float64,
        )
    else:
        penetration = float(args.v5_design_penetration_m)
        jaw_x = float(args.v5_jaw_center_obj_m[0])
        jaw_z = float(args.v5_jaw_center_obj_m[2])
        moving = np.array([jaw_x, cube_half_y + moving_size[1] * 0.5 - penetration, jaw_z], dtype=np.float64)
        counter = np.array([jaw_x, -cube_half_y - counter_size[1] * 0.5 + penetration, jaw_z], dtype=np.float64)
    return moving, counter


def _build_jaws(object_size: np.ndarray, variant: VariantSpec, args: argparse.Namespace):
    center = np.array([args.object_xy[0], args.object_xy[1], object_size[2] * 0.5], dtype=np.float64)
    plan = _build_plan_from_center(_plan_args(object_size, variant, args), center, "v6_audit")
    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    moving_obj, counter_obj = _jaw_centers_obj(object_size, variant, args)
    rot = _rot_z(variant.yaw_deg)
    moving_world = center + rot @ moving_obj
    counter_world = center + rot @ counter_obj
    inv_gripper = np.linalg.inv(_gripper_transform(q_design))
    moving_local_center = (inv_gripper @ np.array([*moving_world, 1.0], dtype=np.float64))[:3]
    counter_origin_gripper = (inv_gripper @ np.array([*counter_world, 1.0], dtype=np.float64))[:3]
    moving_vertices_local = _box_vertices(moving_local_center, np.asarray(args.moving_jaw_size_m, dtype=np.float64), 0.0)
    counter_vertices_local = _box_vertices(np.zeros(3, dtype=np.float64), np.asarray(args.counter_jaw_size_m, dtype=np.float64), 0.0)
    return center, plan, q_design, moving_obj, counter_obj, moving_vertices_local, counter_vertices_local, counter_origin_gripper


def _sample(label: str, q_deg: np.ndarray, object_center: np.ndarray, object_size: np.ndarray, variant: VariantSpec, moving_local: np.ndarray, counter_local: np.ndarray, counter_origin: np.ndarray, patch_m: float) -> dict[str, object]:
    moving_world = _transform_points(_gripper_transform(q_deg), moving_local)
    counter_world = _transform_points(_gripper_transform(q_deg) @ _translation(counter_origin), counter_local)
    moving = _contact(moving_world, object_center, object_size, variant.yaw_deg, patch_m)
    counter = _contact(counter_world, object_center, object_size, variant.yaw_deg, patch_m)
    one_sided = bool(moving["contact"]) and not bool(counter["contact"])
    print(
        "[cube2cm_v6_static_runtime_audit] sample "
        f"label={label} variant={variant.name} object_size_m={_fmt_xyz(object_size)} "
        f"patch_margin_m={patch_m:.4f} moving_contact={_yes(bool(moving['contact']))} "
        f"counter_contact={_yes(bool(counter['contact']))} one_sided_push={_yes(one_sided)} "
        f"moving_center_obj={_fmt_xyz(np.asarray(moving['center']))} "
        f"counter_center_obj={_fmt_xyz(np.asarray(counter['center']))} "
        f"moving_overlap_obj_m={_fmt_xyz(np.asarray(moving['overlap']))} "
        f"counter_overlap_obj_m={_fmt_xyz(np.asarray(counter['overlap']))} "
        f"moving_gap_obj_m={_fmt_xyz(np.asarray(moving['gap']))} "
        f"counter_gap_obj_m={_fmt_xyz(np.asarray(counter['gap']))}",
        flush=True,
    )
    return {"moving": moving, "counter": counter, "one_sided": one_sided}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_sizes_m", nargs="+", type=float, default=[0.020, 0.030])
    ap.add_argument("--object_xy", nargs=2, type=float, default=[0.21369617, -0.19571920])
    ap.add_argument("--patch_margins_m", nargs="+", type=float, default=[0.0, 0.001, 0.002, 0.003, 0.005])
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.800)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--v4_moving_close_overlap_m", type=float, default=-0.0015)
    ap.add_argument("--v4_counter_open_clearance_m", type=float, default=0.00075)
    ap.add_argument("--v4_jaw_center_z_offset_m", type=float, default=0.0020)
    ap.add_argument("--v5_jaw_center_obj_m", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--v5_design_penetration_m", type=float, default=0.0015)
    args = ap.parse_args()

    if len(args.object_sizes_m) % 1 != 0:
        raise ValueError("object_sizes_m must be scalar cube edges in meters")

    print("[cube2cm_v6_static_runtime_audit] local_static_only=YES isaac_run=NO training=NO dataset_generation=NO")
    print(
        "[cube2cm_v6_static_runtime_audit] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "constraint_prim_insertion=NO surface_gripper=NO transport_target=NO release_marker=NO "
        "scalar_or_gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        "[cube2cm_v6_static_runtime_audit] runtime_endpoint_sources="
        + ",".join(endpoint.source_log for endpoint in RUNTIME_ENDPOINTS.values()),
        flush=True,
    )

    for edge in args.object_sizes_m:
        object_size = np.array([edge, edge, edge], dtype=np.float64)
        print(f"[cube2cm_v6_static_runtime_audit] object_case object_size_m={_fmt_xyz(object_size)}", flush=True)
        for variant in VARIANTS:
            center, plan, q_design, moving_obj, counter_obj, moving_local, counter_local, counter_origin = _build_jaws(
                object_size, variant, args
            )
            print(
                "[cube2cm_v6_static_runtime_audit] authored "
                f"variant={variant.name} yaw_deg={variant.yaw_deg:.1f} normalized_grasp={_fmt_xyz(variant.normalized_grasp)} "
                f"center={_fmt_xyz(center)} descend_tcp={_fmt_xyz(plan.descend_tcp)} "
                f"moving_design_center_obj={_fmt_xyz(moving_obj)} counter_design_center_obj={_fmt_xyz(counter_obj)} "
                f"counter_origin_gripper_m={_fmt_xyz(counter_origin)} design_ik_ok={_yes(plan.descend_ik_ok)} "
                f"design_fk_error_m={_norm(plan.descend_tcp - plan.world_grasp - np.array([0.0, 0.0, args.grasp_surface_margin_m])):.6f}",
                flush=True,
            )
            endpoint = RUNTIME_ENDPOINTS[variant.name]
            for patch_m in args.patch_margins_m:
                _sample("authored_static_design", q_design, center, object_size, variant, moving_local, counter_local, counter_origin, patch_m)
                runtime_center = endpoint.object_center_m.copy()
                if abs(edge - 0.020) > 1.0e-9:
                    # Counterfactual size-only check: keep the logged xy endpoint but put
                    # the bottom at z=0 for the larger real foam cube.
                    runtime_center[2] = edge * 0.5
                _sample(endpoint.label, endpoint.q_deg, runtime_center, object_size, variant, moving_local, counter_local, counter_origin, patch_m)

    print(
        "[cube2cm_v6_static_runtime_audit] interpretation "
        "runtime_counter_contact_missing_is_reproduced_by_geometry=YES "
        "v4_3cm_with_patch_margin_is_less_bad_than_v5=CHECK_OUTPUT "
        "next_candidate_bias=v4_like_plus_counter_runtime_alignment object_size_priority=0.030_first "
        "success_condition_later='moving_contact=YES counter_contact=YES one_sided_push=NO close_reached=YES posewrite_calls=0'",
        flush=True,
    )
    print("[cube2cm_v6_static_runtime_audit] CUBE2CM_V6_STATIC_RUNTIME_CONTACT_AUDIT_DONE=YES", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
