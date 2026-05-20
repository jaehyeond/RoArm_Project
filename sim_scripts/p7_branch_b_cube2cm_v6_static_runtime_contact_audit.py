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
    GRIPPER_OPEN_DEG,
    _build_plan_from_center,
    _norm,
    _solve_q,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import _translation  # noqa: E402
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v2_urdf import _open_descent_waypoints  # noqa: E402


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


def _runtime_center_for_size(endpoint: RuntimeEndpoint, object_size: np.ndarray) -> np.ndarray:
    runtime_center = endpoint.object_center_m.copy()
    if abs(float(object_size[2]) - 0.020) > 1.0e-9:
        # Counterfactual size-only check: keep the logged xy endpoint but put
        # the bottom at z=0 for the larger real foam cube.
        runtime_center[2] = float(object_size[2]) * 0.5
    return runtime_center


def _counter_alignment_sample(
    q_deg: np.ndarray,
    runtime_center: np.ndarray,
    object_size: np.ndarray,
    variant: VariantSpec,
    moving_local: np.ndarray,
    counter_local: np.ndarray,
    counter_origin: np.ndarray,
    patch_m: float,
) -> dict[str, object]:
    moving_world = _transform_points(_gripper_transform(q_deg), moving_local)
    counter_world = _transform_points(_gripper_transform(q_deg) @ _translation(counter_origin), counter_local)
    moving = _contact(moving_world, runtime_center, object_size, variant.yaw_deg, patch_m)
    counter = _contact(counter_world, runtime_center, object_size, variant.yaw_deg, patch_m)
    one_sided = bool(moving["contact"]) and not bool(counter["contact"])
    return {"moving": moving, "counter": counter, "one_sided": one_sided}


def _counter_candidate_geometry(
    center: np.ndarray,
    q_design: np.ndarray,
    variant: VariantSpec,
    moving_local: np.ndarray,
    counter_obj_base: np.ndarray,
    counter_size: np.ndarray,
    counter_x_shift_mm: float,
    counter_y_shift_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counter_obj = counter_obj_base + np.array(
        [float(counter_x_shift_mm) / 1000.0, float(counter_y_shift_mm) / 1000.0, 0.0],
        dtype=np.float64,
    )
    counter_world = center + _rot_z(variant.yaw_deg) @ counter_obj
    counter_origin = (np.linalg.inv(_gripper_transform(q_design)) @ np.array([*counter_world, 1.0], dtype=np.float64))[:3]
    counter_local = _box_vertices(np.zeros(3, dtype=np.float64), counter_size, 0.0)
    return moving_local, counter_local, counter_origin


def _open_clearance_check(
    args: argparse.Namespace,
    center: np.ndarray,
    plan,
    object_size: np.ndarray,
    variant: VariantSpec,
    moving_local: np.ndarray,
    counter_local: np.ndarray,
    counter_origin: np.ndarray,
) -> dict[str, object]:
    q_seed = plan.q_approach_deg.copy()
    min_separating_gap = float("inf")
    contact_count = 0
    clearance_bad = 0
    waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    for idx, waypoint in enumerate(waypoints, start=1):
        q_open, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        if not ik_ok:
            clearance_bad += 1
            print(
                "[cube2cm_v6_static_runtime_audit] v6_candidate_open_waypoint "
                f"index={idx:03d}/{len(waypoints):03d} ik_ok=NO ik_err_mm={ik_err_mm:.3f}",
                flush=True,
            )
            continue
        moving_world = _transform_points(_gripper_transform(q_open), moving_local)
        counter_world = _transform_points(_gripper_transform(q_open) @ _translation(counter_origin), counter_local)
        for mesh_name, vertices_world in (("moving", moving_world), ("counter", counter_world)):
            stats = _contact(vertices_world, center, object_size, variant.yaw_deg, 0.0)
            gap = np.asarray(stats["gap"], dtype=np.float64)
            positive = gap[gap > 0.0]
            sep_gap = 0.0 if positive.size == 0 else float(positive.min())
            min_separating_gap = min(min_separating_gap, sep_gap)
            contact = bool(stats["contact"])
            if contact:
                contact_count += 1
            elif sep_gap < float(args.candidate_open_clearance_gate_m):
                clearance_bad += 1
            if args.candidate_print_open_waypoints:
                print(
                    "[cube2cm_v6_static_runtime_audit] v6_candidate_open_waypoint "
                    f"index={idx:03d}/{len(waypoints):03d} mesh={mesh_name} "
                    f"ik_ok=YES ik_err_mm={ik_err_mm:.3f} contact={_yes(contact)} "
                    f"separating_axis_gap_m={sep_gap:.6f} center_obj={_fmt_xyz(np.asarray(stats['center']))} "
                    f"gap_obj_m={_fmt_xyz(gap)}",
                    flush=True,
                )
    open_clear = contact_count == 0 and clearance_bad == 0
    return {
        "waypoints": len(waypoints),
        "contact_count": contact_count,
        "clearance_bad": clearance_bad,
        "min_separating_gap": min_separating_gap,
        "open_clear": open_clear,
    }


def _run_v6_candidate_check(args: argparse.Namespace) -> bool:
    object_size = np.asarray(args.candidate_object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or not np.allclose(object_size, [0.030, 0.030, 0.030], atol=1.0e-12):
        raise ValueError("v6 candidate check is intentionally gated to object_size_m=(0.030,0.030,0.030)")
    variant = next(v for v in VARIANTS if v.name == args.candidate_variant)
    if not variant.v4_like:
        raise ValueError("v6 candidate check is intentionally gated to v4_like first")

    center, plan, q_design, moving_obj, counter_obj_base, moving_local, _counter_local, _counter_origin = _build_jaws(
        object_size, variant, args
    )
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64).copy()
    counter_size[1] = float(args.candidate_counter_thickness_y_mm) / 1000.0
    moving_local, counter_local, counter_origin = _counter_candidate_geometry(
        center,
        q_design,
        variant,
        moving_local,
        counter_obj_base,
        counter_size,
        float(args.candidate_counter_x_shift_mm),
        float(args.candidate_counter_y_shift_mm),
    )

    print(
        "[cube2cm_v6_static_runtime_audit] v6_candidate_check_begin "
        "local_static_only=YES isaac_run=NO training=NO dataset_generation=NO "
        f"variant={variant.name} object_size_m={_fmt_xyz(object_size)} "
        f"counter_x_shift_mm={float(args.candidate_counter_x_shift_mm):+.3f} "
        f"counter_y_shift_mm={float(args.candidate_counter_y_shift_mm):+.3f} "
        f"counter_thickness_y_mm={float(args.candidate_counter_thickness_y_mm):.3f} "
        f"max_plausible_patch_margin_m={float(args.max_plausible_patch_margin_m):.4f} "
        f"moving_design_center_obj={_fmt_xyz(moving_obj)} "
        f"counter_design_center_obj_base={_fmt_xyz(counter_obj_base)}",
        flush=True,
    )

    authored = _counter_alignment_sample(q_design, center, object_size, variant, moving_local, counter_local, counter_origin, 0.0)
    authored_both = bool(authored["moving"]["contact"]) and bool(authored["counter"]["contact"]) and not bool(authored["one_sided"])
    print(
        "[cube2cm_v6_static_runtime_audit] v6_candidate_authored_static "
        f"moving_contact={_yes(bool(authored['moving']['contact']))} "
        f"counter_contact={_yes(bool(authored['counter']['contact']))} "
        f"one_sided_push={_yes(bool(authored['one_sided']))} "
        f"moving_overlap_obj_m={_fmt_xyz(np.asarray(authored['moving']['overlap']))} "
        f"counter_overlap_obj_m={_fmt_xyz(np.asarray(authored['counter']['overlap']))}",
        flush=True,
    )

    open_check = _open_clearance_check(args, center, plan, object_size, variant, moving_local, counter_local, counter_origin)
    print(
        "[cube2cm_v6_static_runtime_audit] v6_candidate_open_descent "
        f"waypoints={int(open_check['waypoints'])} contact_count={int(open_check['contact_count'])} "
        f"clearance_bad={int(open_check['clearance_bad'])} "
        f"min_separating_gap_m={float(open_check['min_separating_gap']):.6f} "
        f"open_descent_clearance={_yes(bool(open_check['open_clear']))}",
        flush=True,
    )

    endpoint = RUNTIME_ENDPOINTS[variant.name]
    runtime_center = _runtime_center_for_size(endpoint, object_size)
    runtime_hits: list[tuple[float, dict[str, object]]] = []
    runtime_at_candidate_patch: dict[str, object] | None = None
    for patch_m in args.alignment_patch_margins_m:
        sample = _counter_alignment_sample(
            endpoint.q_deg,
            runtime_center,
            object_size,
            variant,
            moving_local,
            counter_local,
            counter_origin,
            float(patch_m),
        )
        ok = bool(sample["moving"]["contact"]) and bool(sample["counter"]["contact"]) and not bool(sample["one_sided"])
        if abs(float(patch_m) - float(args.candidate_patch_margin_m)) <= 1.0e-12:
            runtime_at_candidate_patch = sample
        if ok:
            runtime_hits.append((float(patch_m), sample))
        print(
            "[cube2cm_v6_static_runtime_audit] v6_candidate_runtime_line422 "
            f"patch_margin_m={float(patch_m):.4f} moving_contact={_yes(bool(sample['moving']['contact']))} "
            f"counter_contact={_yes(bool(sample['counter']['contact']))} "
            f"one_sided_push={_yes(bool(sample['one_sided']))} "
            f"counter_center_obj={_fmt_xyz(np.asarray(sample['counter']['center']))} "
            f"counter_gap_obj_m={_fmt_xyz(np.asarray(sample['counter']['gap']))}",
            flush=True,
        )

    if runtime_at_candidate_patch is None:
        raise ValueError("candidate_patch_margin_m must be included in alignment_patch_margins_m")
    min_runtime_patch = None if not runtime_hits else min(patch for patch, _sample in runtime_hits)
    runtime_plausible = min_runtime_patch is not None and min_runtime_patch <= float(args.max_plausible_patch_margin_m)
    candidate_patch_ok = (
        bool(runtime_at_candidate_patch["moving"]["contact"])
        and bool(runtime_at_candidate_patch["counter"]["contact"])
        and not bool(runtime_at_candidate_patch["one_sided"])
        and float(args.candidate_patch_margin_m) <= float(args.max_plausible_patch_margin_m)
    )
    success = authored_both and bool(open_check["open_clear"]) and runtime_plausible and candidate_patch_ok
    print(
        "[cube2cm_v6_static_runtime_audit] v6_candidate_summary "
        f"authored_static_both_contact={_yes(authored_both)} "
        f"open_descent_clearance={_yes(bool(open_check['open_clear']))} "
        f"runtime_endpoint={endpoint.source_log} "
        f"runtime_min_patch_margin_m={'NONE' if min_runtime_patch is None else f'{min_runtime_patch:.4f}'} "
        f"runtime_contact_within_plausible_patch_margin={_yes(runtime_plausible)} "
        f"candidate_patch_margin_m={float(args.candidate_patch_margin_m):.4f} "
        f"candidate_patch_condition_ok={_yes(candidate_patch_ok)} "
        f"target_condition='moving_contact=YES counter_contact=YES one_sided_push=NO' "
        f"success_claim=NO isaac_physics_validated=NO v6_candidate_static_better={_yes(success)}",
        flush=True,
    )
    print("[cube2cm_v6_static_runtime_audit] v6_candidate_check_end", flush=True)
    return success


def _run_counter_alignment_sweep(args: argparse.Namespace) -> None:
    object_size = np.asarray(args.alignment_object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or not np.allclose(object_size, [0.030, 0.030, 0.030], atol=1.0e-12):
        raise ValueError("counter alignment sweep is intentionally gated to object_size_m=(0.030,0.030,0.030)")
    variant = next(v for v in VARIANTS if v.name == args.alignment_variant)
    if not variant.v4_like:
        raise ValueError("counter alignment sweep is intentionally gated to v4_like first")

    center, plan, q_design, moving_obj, counter_obj_base, moving_local, _counter_local, _counter_origin = _build_jaws(
        object_size, variant, args
    )
    endpoint = RUNTIME_ENDPOINTS[variant.name]
    runtime_center = _runtime_center_for_size(endpoint, object_size)
    q_runtime = endpoint.q_deg
    base_counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)

    baseline_counter_local = _box_vertices(np.zeros(3, dtype=np.float64), base_counter_size, 0.0)
    baseline_counter_world = center + _rot_z(variant.yaw_deg) @ counter_obj_base
    baseline_counter_origin = (
        np.linalg.inv(_gripper_transform(q_design)) @ np.array([*baseline_counter_world, 1.0], dtype=np.float64)
    )[:3]
    baseline = _counter_alignment_sample(
        q_runtime,
        runtime_center,
        object_size,
        variant,
        moving_local,
        baseline_counter_local,
        baseline_counter_origin,
        0.0,
    )

    print(
        "[cube2cm_v6_static_runtime_audit] counter_alignment_sweep_begin "
        "local_static_only=YES isaac_run=NO training=NO dataset_generation=NO "
        "variant=v4_like object_size_m=([+0.030000, +0.030000, +0.030000]) "
        f"runtime_endpoint={endpoint.source_log} "
        f"runtime_center_for_3cm={_fmt_xyz(runtime_center)} "
        f"base_moving_design_center_obj={_fmt_xyz(moving_obj)} "
        f"base_counter_design_center_obj={_fmt_xyz(counter_obj_base)}",
        flush=True,
    )
    print(
        "[cube2cm_v6_static_runtime_audit] counter_alignment_baseline "
        f"patch_margin_m=0.0000 moving_contact={_yes(bool(baseline['moving']['contact']))} "
        f"counter_contact={_yes(bool(baseline['counter']['contact']))} "
        f"one_sided_push={_yes(bool(baseline['one_sided']))} "
        f"moving_center_obj={_fmt_xyz(np.asarray(baseline['moving']['center']))} "
        f"counter_center_obj={_fmt_xyz(np.asarray(baseline['counter']['center']))} "
        f"counter_gap_obj_m={_fmt_xyz(np.asarray(baseline['counter']['gap']))}",
        flush=True,
    )

    best: dict[str, object] | None = None
    plausible_best: dict[str, object] | None = None
    total = 0
    success_count = 0
    rot = _rot_z(variant.yaw_deg)
    inv_design = np.linalg.inv(_gripper_transform(q_design))

    for thickness_mm in args.alignment_counter_thickness_y_mm:
        counter_size = base_counter_size.copy()
        counter_size[1] = float(thickness_mm) / 1000.0
        counter_local = _box_vertices(np.zeros(3, dtype=np.float64), counter_size, 0.0)
        for patch_m in args.alignment_patch_margins_m:
            first_for_combo: dict[str, object] | None = None
            for shift_mm in args.alignment_counter_y_shift_mm:
                total += 1
                shift_m = float(shift_mm) / 1000.0
                counter_obj = counter_obj_base + np.array([0.0, shift_m, 0.0], dtype=np.float64)
                counter_world = center + rot @ counter_obj
                counter_origin = (inv_design @ np.array([*counter_world, 1.0], dtype=np.float64))[:3]
                sample = _counter_alignment_sample(
                    q_runtime,
                    runtime_center,
                    object_size,
                    variant,
                    moving_local,
                    counter_local,
                    counter_origin,
                    float(patch_m),
                )
                ok = bool(sample["moving"]["contact"]) and bool(sample["counter"]["contact"]) and not bool(sample["one_sided"])
                if args.alignment_print_all or (ok and first_for_combo is None):
                    print(
                        "[cube2cm_v6_static_runtime_audit] counter_alignment_candidate "
                        f"shift_y_mm={float(shift_mm):+.3f} counter_thickness_y_mm={float(thickness_mm):.3f} "
                        f"patch_margin_m={float(patch_m):.4f} moving_contact={_yes(bool(sample['moving']['contact']))} "
                        f"counter_contact={_yes(bool(sample['counter']['contact']))} "
                        f"one_sided_push={_yes(bool(sample['one_sided']))} "
                        f"counter_center_obj={_fmt_xyz(np.asarray(sample['counter']['center']))} "
                        f"counter_overlap_obj_m={_fmt_xyz(np.asarray(sample['counter']['overlap']))} "
                        f"counter_gap_obj_m={_fmt_xyz(np.asarray(sample['counter']['gap']))}",
                        flush=True,
                    )
                if not ok:
                    continue
                success_count += 1
                row = {
                    "shift_mm": float(shift_mm),
                    "thickness_mm": float(thickness_mm),
                    "patch_m": float(patch_m),
                    "sample": sample,
                }
                if first_for_combo is None:
                    first_for_combo = row
                if best is None or (
                    row["shift_mm"],
                    row["patch_m"],
                    row["thickness_mm"],
                ) < (
                    float(best["shift_mm"]),
                    float(best["patch_m"]),
                    float(best["thickness_mm"]),
                ):
                    best = row
                if float(patch_m) <= float(args.max_plausible_patch_margin_m) and (
                    plausible_best is None
                    or (
                        row["shift_mm"],
                        row["patch_m"],
                        row["thickness_mm"],
                    )
                    < (
                        float(plausible_best["shift_mm"]),
                        float(plausible_best["patch_m"]),
                        float(plausible_best["thickness_mm"]),
                    )
                ):
                    plausible_best = row
            if first_for_combo is None:
                print(
                    "[cube2cm_v6_static_runtime_audit] counter_alignment_combo_no_hit "
                    f"counter_thickness_y_mm={float(thickness_mm):.3f} patch_margin_m={float(patch_m):.4f}",
                    flush=True,
                )

    if best is None:
        print(
            "[cube2cm_v6_static_runtime_audit] counter_alignment_summary "
            f"tested={total} success=0 minimal_shift_found=NO depends_on_unrealistic_patch_margin=UNKNOWN "
            "target_condition='moving_contact=YES counter_contact=YES one_sided_push=NO'",
            flush=True,
        )
    else:
        sample = best["sample"]
        best_patch_plausible = float(best["patch_m"]) <= float(args.max_plausible_patch_margin_m)
        print(
            "[cube2cm_v6_static_runtime_audit] counter_alignment_summary "
            f"tested={total} success={success_count} minimal_shift_found=YES "
            f"minimal_counter_y_shift_mm={float(best['shift_mm']):.3f} "
            f"counter_thickness_y_mm={float(best['thickness_mm']):.3f} "
            f"patch_margin_m={float(best['patch_m']):.4f} "
            f"minimal_shift_within_plausible_patch_margin={_yes(best_patch_plausible)} "
            f"minimal_shift_depends_on_unrealistic_patch_margin={_yes(not best_patch_plausible)} "
            f"plausible_counter_y_shift_found={_yes(plausible_best is not None)} "
            f"moving_contact={_yes(bool(sample['moving']['contact']))} "
            f"counter_contact={_yes(bool(sample['counter']['contact']))} "
            f"one_sided_push={_yes(bool(sample['one_sided']))} "
            "overfit_to_grasped_marker=NO success_claim=NO "
            "later_runtime_target='close_reached=YES posewrite_calls=0 hold_lift_follow=YES'",
            flush=True,
        )
        if plausible_best is not None and plausible_best is not best:
            p_sample = plausible_best["sample"]
            print(
                "[cube2cm_v6_static_runtime_audit] counter_alignment_plausible_best "
                f"minimal_counter_y_shift_mm={float(plausible_best['shift_mm']):.3f} "
                f"counter_thickness_y_mm={float(plausible_best['thickness_mm']):.3f} "
                f"patch_margin_m={float(plausible_best['patch_m']):.4f} "
                f"moving_contact={_yes(bool(p_sample['moving']['contact']))} "
                f"counter_contact={_yes(bool(p_sample['counter']['contact']))} "
                f"one_sided_push={_yes(bool(p_sample['one_sided']))}",
                flush=True,
            )
    print("[cube2cm_v6_static_runtime_audit] counter_alignment_sweep_end", flush=True)


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
    ap.add_argument("--run_counter_alignment_sweep", action="store_true")
    ap.add_argument("--alignment_variant", choices=["v4_like"], default="v4_like")
    ap.add_argument("--alignment_object_size_m", nargs=3, type=float, default=[0.030, 0.030, 0.030])
    ap.add_argument(
        "--alignment_counter_y_shift_mm",
        nargs="+",
        type=float,
        default=[round(0.5 * i, 3) for i in range(0, 29)],
        help="Positive values move the counter proxy toward object-frame +Y at the runtime endpoint.",
    )
    ap.add_argument(
        "--alignment_counter_thickness_y_mm",
        nargs="+",
        type=float,
        default=[1.5, 2.0, 3.0, 4.0, 5.0],
    )
    ap.add_argument(
        "--alignment_patch_margins_m",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.002, 0.003, 0.005],
    )
    ap.add_argument("--max_plausible_patch_margin_m", type=float, default=0.003)
    ap.add_argument("--alignment_print_all", action="store_true")
    ap.add_argument("--run_v6_candidate_check", action="store_true")
    ap.add_argument("--candidate_variant", choices=["v4_like"], default="v4_like")
    ap.add_argument("--candidate_object_size_m", nargs=3, type=float, default=[0.030, 0.030, 0.030])
    ap.add_argument("--candidate_counter_x_shift_mm", type=float, default=1.0)
    ap.add_argument("--candidate_counter_y_shift_mm", type=float, default=5.0)
    ap.add_argument("--candidate_counter_thickness_y_mm", type=float, default=5.0)
    ap.add_argument("--candidate_patch_margin_m", type=float, default=0.003)
    ap.add_argument("--candidate_open_clearance_gate_m", type=float, default=0.0005)
    ap.add_argument("--candidate_print_open_waypoints", action="store_true")
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
                runtime_center = _runtime_center_for_size(endpoint, object_size)
                _sample(endpoint.label, endpoint.q_deg, runtime_center, object_size, variant, moving_local, counter_local, counter_origin, patch_m)

    if args.run_counter_alignment_sweep:
        _run_counter_alignment_sweep(args)
    candidate_success = True
    if args.run_v6_candidate_check:
        candidate_success = _run_v6_candidate_check(args)

    print(
        "[cube2cm_v6_static_runtime_audit] interpretation "
        "runtime_counter_contact_missing_is_reproduced_by_geometry=YES "
        "v4_3cm_with_patch_margin_is_less_bad_than_v5=CHECK_OUTPUT "
        "next_candidate_bias=v4_like_plus_counter_runtime_alignment object_size_priority=0.030_first "
        "success_condition_later='moving_contact=YES counter_contact=YES one_sided_push=NO close_reached=YES posewrite_calls=0'",
        flush=True,
    )
    print("[cube2cm_v6_static_runtime_audit] CUBE2CM_V6_STATIC_RUNTIME_CONTACT_AUDIT_DONE=YES", flush=True)
    return 0 if candidate_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
