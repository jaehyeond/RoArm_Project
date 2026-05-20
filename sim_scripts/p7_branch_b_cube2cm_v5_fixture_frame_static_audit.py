#!/usr/bin/env python3
"""Fixture-frame static audit for a v5 2cm cube grasp candidate.

This is numpy-only. It does not launch Isaac, train, edit defaults, insert
constraints, attach SurfaceGripper, transport, or release. The goal is to stop
judging candidates by world-axis AABB balance alone and instead score whether
the jaw fixture closing normal is aligned with the cube object frame.
"""
from __future__ import annotations

import argparse
import math
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
    _build_plan_from_center,
    _norm,
    _rot_z,
    _solve_q,
    _yes,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import _translation  # noqa: E402
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v2_urdf import _open_descent_waypoints  # noqa: E402


def _axis_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)


def _contact(vertices_obj: np.ndarray, cube_min_obj: np.ndarray, cube_max_obj: np.ndarray) -> dict[str, object]:
    mn, mx = _aabb(vertices_obj)
    overlap = _aabb_overlap(mn, mx, cube_min_obj, cube_max_obj)
    gap = _axis_gap(mn, mx, cube_min_obj, cube_max_obj)
    return {
        "center": 0.5 * (mn + mx),
        "min": mn,
        "max": mx,
        "overlap": overlap,
        "gap": gap,
        "contact": bool(np.all(overlap > 0.0)),
    }


def _to_object_frame(points_world: np.ndarray, center: np.ndarray, yaw_deg: float) -> np.ndarray:
    rot = _rot_z(yaw_deg)
    return (points_world - center) @ rot


def _fmt_mm(value_m: np.ndarray) -> str:
    return _fmt_xyz(np.asarray(value_m, dtype=np.float64) * 1000.0)


def _values_from_range(spec: str) -> list[float]:
    if ":" not in spec:
        return [float(v) for v in spec.split(",") if v]
    start, stop, step = (float(v) for v in spec.split(":"))
    values = []
    cur = start
    eps = abs(step) * 1.0e-6
    if step == 0.0:
        raise ValueError("range step must be non-zero")
    if step > 0:
        while cur <= stop + eps:
            values.append(round(cur, 10))
            cur += step
    else:
        while cur >= stop - eps:
            values.append(round(cur, 10))
            cur += step
    return values


def _build_plan(args: argparse.Namespace, yaw_deg: float, tcp_x_mm: float, tcp_y_mm: float):
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    center = np.asarray(args.object_center_m, dtype=np.float64)
    normalized = np.array(
        [
            tcp_x_mm / (object_size[0] * 1000.0),
            tcp_y_mm / (object_size[1] * 1000.0),
            0.5,
        ],
        dtype=np.float64,
    )
    plan_args = argparse.Namespace(**vars(args))
    plan_args.yaw_deg = yaw_deg
    plan_args.normalized_grasp = normalized
    plan_args.grasp_name = "top_center"
    return _build_plan_from_center(plan_args, center, "v5_fixture_frame")


def _candidate_vertices(
    args: argparse.Namespace,
    plan,
    yaw_deg: float,
    jaw_x_mm: float,
    jaw_z_mm: float,
    penetration_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    rot = _rot_z(yaw_deg)
    cube_half_y = object_size[1] * 0.5
    moving_half_y = moving_size[1] * 0.5
    counter_half_y = counter_size[1] * 0.5
    penetration_m = penetration_mm / 1000.0
    jaw_x_m = jaw_x_mm / 1000.0
    jaw_z_m = jaw_z_mm / 1000.0
    moving_obj_center = np.array(
        [jaw_x_m, cube_half_y + moving_half_y - penetration_m, jaw_z_m],
        dtype=np.float64,
    )
    counter_obj_center = np.array(
        [jaw_x_m, -cube_half_y - counter_half_y + penetration_m, jaw_z_m],
        dtype=np.float64,
    )
    moving_world_center = plan.center + rot @ moving_obj_center
    counter_world_center = plan.center + rot @ counter_obj_center

    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    inv_gripper = np.linalg.inv(_gripper_transform(q_design))
    moving_local_center = (inv_gripper @ np.array([*moving_world_center, 1.0], dtype=np.float64))[:3]
    counter_origin_gripper = (inv_gripper @ np.array([*counter_world_center, 1.0], dtype=np.float64))[:3]
    moving_vertices = _box_vertices(moving_local_center, moving_size, 0.0)
    counter_vertices = _box_vertices(np.zeros(3, dtype=np.float64), counter_size, 0.0)
    return moving_vertices, counter_vertices, counter_origin_gripper


def _evaluate_candidate(
    args: argparse.Namespace,
    yaw_deg: float,
    tcp_x_mm: float,
    tcp_y_mm: float,
    jaw_x_mm: float,
    jaw_z_mm: float,
    penetration_mm: float,
) -> dict[str, object]:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    cube_min_obj = -0.5 * object_size
    cube_max_obj = 0.5 * object_size
    plan = _build_plan(args, yaw_deg, tcp_x_mm, tcp_y_mm)
    moving_vertices, counter_vertices, counter_origin_gripper = _candidate_vertices(
        args,
        plan,
        yaw_deg,
        jaw_x_mm,
        jaw_z_mm,
        penetration_mm,
    )

    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    gripper = _gripper_transform(q_design)
    closing_axis_world = gripper[:3, 1]
    object_y_world = _rot_z(yaw_deg)[:, 1]
    object_x_world = _rot_z(yaw_deg)[:, 0]
    closing_xy = closing_axis_world.copy()
    closing_xy[2] = 0.0
    closing_xy_norm = _norm(closing_xy)
    if closing_xy_norm > 1.0e-12:
        closing_xy /= closing_xy_norm
    normal_alignment = abs(float(np.dot(closing_xy, object_y_world)))
    tangential_slip = abs(float(np.dot(closing_xy, object_x_world)))
    vertical_component = abs(float(closing_axis_world[2]))

    open_contacts = 0
    open_clearance_bad = 0
    open_waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    q_seed = plan.q_descend_deg.copy()
    for waypoint in open_waypoints:
        q_open, _ik_ok, _ik_err = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        moving_open = _transform_points(_gripper_transform(q_open), moving_vertices)
        counter_open = _transform_points(
            _gripper_transform(q_open) @ _translation(counter_origin_gripper),
            counter_vertices,
        )
        for vertices_world in (moving_open, counter_open):
            stats = _contact(_to_object_frame(vertices_world, plan.center, yaw_deg), cube_min_obj, cube_max_obj)
            if bool(stats["contact"]):
                open_contacts += 1
            else:
                gap = np.asarray(stats["gap"], dtype=np.float64)
                positive = gap[gap > 0.0]
                if positive.size and float(positive.min()) < args.open_clearance_gate_m:
                    open_clearance_bad += 1

    close = {}
    q_seed = plan.q_descend_deg.copy()
    for angle in args.close_deg:
        q_close, _ik_ok, _ik_err = _solve_q(plan.descend_tcp, q_seed, float(angle), args)
        q_seed = q_close
        moving_world = _transform_points(_gripper_transform(q_close), moving_vertices)
        counter_world = _transform_points(
            _gripper_transform(q_close) @ _translation(counter_origin_gripper),
            counter_vertices,
        )
        close[float(angle)] = {
            "moving": _contact(_to_object_frame(moving_world, plan.center, yaw_deg), cube_min_obj, cube_max_obj),
            "counter": _contact(_to_object_frame(counter_world, plan.center, yaw_deg), cube_min_obj, cube_max_obj),
        }

    design = close[float(args.design_close_deg)]
    moving_overlap = np.asarray(design["moving"]["overlap"], dtype=np.float64)
    counter_overlap = np.asarray(design["counter"]["overlap"], dtype=np.float64)
    moving_y_mm = float(moving_overlap[1] * 1000.0)
    counter_y_mm = float(counter_overlap[1] * 1000.0)
    moving_center_obj = np.asarray(design["moving"]["center"], dtype=np.float64)
    counter_center_obj = np.asarray(design["counter"]["center"], dtype=np.float64)
    both_contact = bool(design["moving"]["contact"]) and bool(design["counter"]["contact"])
    open_clear = open_contacts == 0 and open_clearance_bad == 0
    z_balance_mm = abs(float((moving_center_obj[2] - counter_center_obj[2]) * 1000.0))
    x_balance_mm = abs(float((moving_center_obj[0] - counter_center_obj[0]) * 1000.0))
    score = (
        (1.0 - normal_alignment) * 100.0
        + tangential_slip * 25.0
        + vertical_component * 15.0
        + abs(moving_y_mm - counter_y_mm) * 1.5
        + max(1.0 - moving_y_mm, 0.0) * 4.0
        + max(1.0 - counter_y_mm, 0.0) * 4.0
        + x_balance_mm * 0.5
        + z_balance_mm * 0.5
        + (0.0 if both_contact else 100.0)
        + (0.0 if open_clear else 100.0)
    )
    return {
        "score": score,
        "yaw_deg": yaw_deg,
        "tcp_x_mm": tcp_x_mm,
        "tcp_y_mm": tcp_y_mm,
        "jaw_x_mm": jaw_x_mm,
        "jaw_z_mm": jaw_z_mm,
        "penetration_mm": penetration_mm,
        "normal_alignment": normal_alignment,
        "tangential_slip": tangential_slip,
        "vertical_component": vertical_component,
        "moving_y_mm": moving_y_mm,
        "counter_y_mm": counter_y_mm,
        "moving_center_obj": moving_center_obj,
        "counter_center_obj": counter_center_obj,
        "both_contact": both_contact,
        "open_clear": open_clear,
        "open_contacts": open_contacts,
        "open_clearance_bad": open_clearance_bad,
        "close": close,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--object_center_m", nargs=3, type=float, default=[0.21369617, -0.19571920, 0.010000])
    ap.add_argument("--yaw_deg_values", default="-75:75:5")
    ap.add_argument("--tcp_x_mm_values", default="-6:6:3")
    ap.add_argument("--tcp_y_mm_values", default="-6:6:3")
    ap.add_argument("--jaw_x_mm_values", default="-3:3:3")
    ap.add_argument("--jaw_z_mm_values", default="-2:4:2")
    ap.add_argument("--penetration_mm_values", default="0.75,1.0,1.5,2.0")
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.800)
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0])
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--open_clearance_gate_m", type=float, default=0.0005)
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    print("[cube2cm_v5_fixture_static] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_v5_fixture_static] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "p7_training=NO constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO "
        "transport_target=NO release_marker=NO",
        flush=True,
    )
    yaw_values = _values_from_range(args.yaw_deg_values)
    tcp_x_values = _values_from_range(args.tcp_x_mm_values)
    tcp_y_values = _values_from_range(args.tcp_y_mm_values)
    jaw_x_values = _values_from_range(args.jaw_x_mm_values)
    jaw_z_values = _values_from_range(args.jaw_z_mm_values)
    penetration_values = _values_from_range(args.penetration_mm_values)
    candidates = []
    for yaw_deg in yaw_values:
        for tcp_x_mm in tcp_x_values:
            for tcp_y_mm in tcp_y_values:
                for jaw_x_mm in jaw_x_values:
                    for jaw_z_mm in jaw_z_values:
                        for penetration_mm in penetration_values:
                            candidates.append(
                                _evaluate_candidate(
                                    args,
                                    yaw_deg,
                                    tcp_x_mm,
                                    tcp_y_mm,
                                    jaw_x_mm,
                                    jaw_z_mm,
                                    penetration_mm,
                                )
                            )
    candidates.sort(key=lambda item: float(item["score"]))
    print(
        f"[cube2cm_v5_fixture_static] sweep_count={len(candidates)} "
        f"yaw_values={len(yaw_values)} tcp_x_values={len(tcp_x_values)} tcp_y_values={len(tcp_y_values)} "
        f"jaw_x_values={len(jaw_x_values)} jaw_z_values={len(jaw_z_values)} penetration_values={len(penetration_values)}",
        flush=True,
    )
    for rank, item in enumerate(candidates[: args.top_k], start=1):
        angle_parts = []
        for angle in args.close_deg:
            sample = item["close"][float(angle)]
            moving_y = float(np.asarray(sample["moving"]["overlap"])[1] * 1000.0)
            counter_y = float(np.asarray(sample["counter"]["overlap"])[1] * 1000.0)
            angle_parts.append(
                f"{float(angle):.0f}:moving_y={moving_y:.3f},counter_y={counter_y:.3f},"
                f"moving_contact={_yes(bool(sample['moving']['contact']))},"
                f"counter_contact={_yes(bool(sample['counter']['contact']))}"
            )
        print(
            f"[cube2cm_v5_fixture_static] candidate rank={rank:02d} score={float(item['score']):.3f} "
            f"yaw_deg={float(item['yaw_deg']):+.1f} tcp_obj_mm=({float(item['tcp_x_mm']):+.1f},{float(item['tcp_y_mm']):+.1f}) "
            f"jaw_obj_mm=({float(item['jaw_x_mm']):+.1f},z={float(item['jaw_z_mm']):+.1f}) "
            f"penetration_mm={float(item['penetration_mm']):.3f} "
            f"normal_alignment={float(item['normal_alignment']):.4f} tangential_slip={float(item['tangential_slip']):.4f} "
            f"vertical_component={float(item['vertical_component']):.4f} "
            f"open_clearance={_yes(bool(item['open_clear']))} open_contacts={int(item['open_contacts'])} "
            f"open_clearance_bad={int(item['open_clearance_bad'])} both_contact_design={_yes(bool(item['both_contact']))} "
            f"design26_moving_y_mm={float(item['moving_y_mm']):.3f} "
            f"design26_counter_y_mm={float(item['counter_y_mm']):.3f} "
            f"moving_center_obj_mm={_fmt_mm(np.asarray(item['moving_center_obj']))} "
            f"counter_center_obj_mm={_fmt_mm(np.asarray(item['counter_center_obj']))} "
            f"close_overlap_by_angle=[{'; '.join(angle_parts)}]",
            flush=True,
        )
    best = candidates[0]
    print(
        f"[cube2cm_v5_fixture_static] best_interpretation yaw_deg={float(best['yaw_deg']):+.1f} "
        f"normal_alignment={float(best['normal_alignment']):.4f} "
        f"v4_world_y_aabb_balance_is_not_sufficient=YES",
        flush=True,
    )
    print("[cube2cm_v5_fixture_static] CUBE2CM_V5_FIXTURE_FRAME_STATIC_AUDIT_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
