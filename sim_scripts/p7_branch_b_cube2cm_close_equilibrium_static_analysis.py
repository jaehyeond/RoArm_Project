#!/usr/bin/env python3
"""Static close-equilibrium analysis for the cube2cm opposing-jaw v2 failure.

This script is diagnostic-only. It does not launch Isaac, train, edit defaults,
insert constraints, attach SurfaceGripper, transport, or release. It compares the
planned v2 jaw/cube geometry against the already-observed B200 close equilibrium
TCP offsets from the latchstop26 continue-close run.
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
    _link5_transform,
    _points_inside_aabb,
    _transform_points,
)
from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    GRIPPER_OPEN_DEG,
    _build_plan,
    _build_plan_from_center,
    _norm,
    _solve_q,
    _yes,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import _translation  # noqa: E402
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v2_urdf import _open_descent_waypoints  # noqa: E402


def _box_local_vertices(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    return _box_vertices(center, size, 0.0)


def _axis_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)


def _signed_center_delta_mm(center: np.ndarray, cube_center: np.ndarray) -> np.ndarray:
    return (np.asarray(center, dtype=np.float64) - np.asarray(cube_center, dtype=np.float64)) * 1000.0


def _contact_stats(
    label: str,
    mesh_name: str,
    vertices: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
    cube_center: np.ndarray,
) -> dict[str, object]:
    mesh_min, mesh_max = _aabb(vertices)
    overlap = _aabb_overlap(mesh_min, mesh_max, cube_min, cube_max)
    gap = _axis_gap(mesh_min, mesh_max, cube_min, cube_max)
    center = 0.5 * (mesh_min + mesh_max)
    inside = _points_inside_aabb(vertices, cube_min, cube_max)
    contact = bool(np.all(overlap > 0.0))
    print(
        f"[cube2cm_close_equilibrium_static] sample label={label} mesh={mesh_name} "
        f"center={_fmt_xyz(center)} center_minus_cube_mm={_fmt_xyz(_signed_center_delta_mm(center, cube_center))} "
        f"aabb_min={_fmt_xyz(mesh_min)} aabb_max={_fmt_xyz(mesh_max)} "
        f"overlap_m={_fmt_xyz(overlap)} axis_gap_m={_fmt_xyz(gap)} "
        f"aabb_contact={_yes(contact)} vertices_inside_cube={inside}",
        flush=True,
    )
    return {
        "contact": contact,
        "center": center,
        "overlap": overlap,
        "gap": gap,
        "inside": inside,
    }


def _contact_stats_silent(
    vertices: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
) -> dict[str, object]:
    mesh_min, mesh_max = _aabb(vertices)
    overlap = _aabb_overlap(mesh_min, mesh_max, cube_min, cube_max)
    gap = _axis_gap(mesh_min, mesh_max, cube_min, cube_max)
    inside = _points_inside_aabb(vertices, cube_min, cube_max)
    return {
        "contact": bool(np.all(overlap > 0.0)),
        "overlap": overlap,
        "gap": gap,
        "inside": int(inside),
    }


def _fmt_vec_mm(vec_m: np.ndarray) -> str:
    return _fmt_xyz(np.asarray(vec_m, dtype=np.float64) * 1000.0)


def _min_positive_gap_m(gap: np.ndarray) -> float:
    positive_gap = np.asarray(gap, dtype=np.float64)
    positive_gap = positive_gap[positive_gap > 0.0]
    if positive_gap.size == 0:
        return 0.0
    return float(positive_gap.min())


def _build_jaw_vertices(
    args: argparse.Namespace,
    plan,
    object_size: np.ndarray,
    moving_size: np.ndarray,
    counter_size: np.ndarray,
    moving_close_overlap_m: float,
    counter_open_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    moving_world_center = np.array(
        [
            plan.center[0],
            plan.center[1] + object_size[1] / 2.0 - moving_close_overlap_m,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )
    counter_world_center = np.array(
        [
            plan.center[0],
            plan.center[1] - object_size[1] / 2.0 - counter_open_clearance_m - counter_size[1] / 2.0,
            args.jaw_center_z_m,
        ],
        dtype=np.float64,
    )
    moving_local_center = (
        np.linalg.inv(_gripper_transform(q_design))
        @ np.array([*moving_world_center, 1.0], dtype=np.float64)
    )[:3]
    if args.counter_mount == "link5":
        counter_local = (
            np.linalg.inv(_link5_transform(q_design))
            @ np.array([*counter_world_center, 1.0], dtype=np.float64)
        )[:3]
    else:
        counter_local = (
            np.linalg.inv(_gripper_transform(q_design))
            @ np.array([*counter_world_center, 1.0], dtype=np.float64)
        )[:3]
    return (
        _box_local_vertices(moving_local_center, moving_size),
        _box_local_vertices(counter_local, counter_size),
    )


def _evaluate_candidate(
    args: argparse.Namespace,
    plan,
    object_size: np.ndarray,
    moving_size: np.ndarray,
    counter_size: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
    open_waypoints: list[np.ndarray],
    close_angles: list[float],
    moving_close_overlap_m: float,
    counter_open_clearance_m: float,
) -> dict[str, object]:
    moving_vertices, counter_vertices = _build_jaw_vertices(
        args,
        plan,
        object_size,
        moving_size,
        counter_size,
        moving_close_overlap_m,
        counter_open_clearance_m,
    )
    open_contacts = 0
    open_clearance_bad = 0
    q_seed = plan.q_descend_deg.copy()
    for waypoint in open_waypoints:
        q_open, _ik_ok, _ik_err = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        moving_open = _transform_points(_gripper_transform(q_open), moving_vertices)
        if args.counter_mount == "link5":
            counter_open = _transform_points(_link5_transform(q_open), counter_vertices)
        else:
            counter_open = _transform_points(_gripper_transform(q_open), counter_vertices)
        for vertices in (moving_open, counter_open):
            stats = _contact_stats_silent(vertices, cube_min, cube_max)
            if bool(stats["contact"]) or int(stats["inside"]) > 0:
                open_contacts += 1
            elif _min_positive_gap_m(np.asarray(stats["gap"], dtype=np.float64)) < args.open_clearance_gate_m:
                open_clearance_bad += 1

    close: dict[float, dict[str, dict[str, object]]] = {}
    q_seed = plan.q_descend_deg.copy()
    for angle in close_angles:
        q_close, _ik_ok, _ik_err = _solve_q(plan.descend_tcp, q_seed, float(angle), args)
        q_seed = q_close
        moving_world = _transform_points(_gripper_transform(q_close), moving_vertices)
        if args.counter_mount == "link5":
            counter_world = _transform_points(_link5_transform(q_close), counter_vertices)
        else:
            counter_world = _transform_points(_gripper_transform(q_close), counter_vertices)
        close[float(angle)] = {
            "moving": _contact_stats_silent(moving_world, cube_min, cube_max),
            "counter": _contact_stats_silent(counter_world, cube_min, cube_max),
        }

    design = close[float(args.design_close_deg)]
    moving_y_mm = float(np.asarray(design["moving"]["overlap"])[1] * 1000.0)
    counter_y_mm = float(np.asarray(design["counter"]["overlap"])[1] * 1000.0)
    both_contact = bool(design["moving"]["contact"]) and bool(design["counter"]["contact"])
    open_clear = open_contacts == 0 and open_clearance_bad == 0
    balance_abs_mm = abs(moving_y_mm - counter_y_mm)
    score = (
        balance_abs_mm
        + 0.35 * max(moving_y_mm - 3.0, 0.0)
        + 0.35 * max(counter_y_mm - 3.5, 0.0)
        + 0.75 * max(1.0 - moving_y_mm, 0.0)
        + 0.75 * max(1.0 - counter_y_mm, 0.0)
        + (0.0 if both_contact else 100.0)
        + (0.0 if open_clear else 100.0)
    )
    return {
        "moving_close_overlap_m": moving_close_overlap_m,
        "counter_open_clearance_m": counter_open_clearance_m,
        "open_contacts": open_contacts,
        "open_clearance_bad": open_clearance_bad,
        "open_clear": open_clear,
        "both_contact": both_contact,
        "moving_y_mm": moving_y_mm,
        "counter_y_mm": counter_y_mm,
        "balance_abs_mm": balance_abs_mm,
        "score": score,
        "close": close,
    }


def _print_candidate_sweep(
    args: argparse.Namespace,
    plan,
    object_size: np.ndarray,
    moving_size: np.ndarray,
    counter_size: np.ndarray,
    cube_min: np.ndarray,
    cube_max: np.ndarray,
    open_waypoints: list[np.ndarray],
) -> None:
    print(
        "[cube2cm_close_equilibrium_static] v4_sweep static_only=YES isaac_run=NO "
        "objective=balance_moving_counter_overlap_preserve_open_descent_clearance",
        flush=True,
    )
    candidates = []
    for moving_mm in args.v4_moving_overlap_mm:
        for clearance_mm in args.v4_counter_clearance_mm:
            candidates.append(
                _evaluate_candidate(
                    args,
                    plan,
                    object_size,
                    moving_size,
                    counter_size,
                    cube_min,
                    cube_max,
                    open_waypoints,
                    [float(a) for a in args.close_deg],
                    moving_mm / 1000.0,
                    clearance_mm / 1000.0,
                )
            )
    candidates.sort(key=lambda item: (float(item["score"]), float(item["balance_abs_mm"])))
    for rank, item in enumerate(candidates[: args.v4_top_k], start=1):
        close = item["close"]
        angle_parts = []
        for angle in args.close_deg:
            sample = close[float(angle)]
            moving_y = float(np.asarray(sample["moving"]["overlap"])[1] * 1000.0)
            counter_y = float(np.asarray(sample["counter"]["overlap"])[1] * 1000.0)
            angle_parts.append(
                f"{float(angle):.0f}:moving_y={moving_y:.3f},counter_y={counter_y:.3f},"
                f"moving_contact={_yes(bool(sample['moving']['contact']))},"
                f"counter_contact={_yes(bool(sample['counter']['contact']))}"
            )
        print(
            f"[cube2cm_close_equilibrium_static] v4_candidate rank={rank:02d} "
            f"moving_close_overlap_mm={float(item['moving_close_overlap_m']) * 1000.0:+.3f} "
            f"counter_open_clearance_mm={float(item['counter_open_clearance_m']) * 1000.0:+.3f} "
            f"open_descent_clearance={_yes(bool(item['open_clear']))} "
            f"open_contacts={int(item['open_contacts'])} open_clearance_bad={int(item['open_clearance_bad'])} "
            f"design26_moving_y_mm={float(item['moving_y_mm']):.3f} "
            f"design26_counter_y_mm={float(item['counter_y_mm']):.3f} "
            f"design26_balance_abs_mm={float(item['balance_abs_mm']):.3f} "
            f"score={float(item['score']):.3f} close_overlap_by_angle=[{'; '.join(angle_parts)}]",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.800)
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0])
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--moving_close_overlap_m", type=float, default=0.0005)
    ap.add_argument("--counter_open_clearance_m", type=float, default=0.0030)
    ap.add_argument("--counter_mount", choices=["link5", "gripper"], default="link5")
    ap.add_argument("--jaw_center_z_m", type=float, default=0.012)
    ap.add_argument("--open_clearance_gate_m", type=float, default=0.0005)
    ap.add_argument("--target_tcp", nargs=3, type=float, default=[0.213690, -0.195703, 0.020500])
    ap.add_argument("--fresh_tcp_23", nargs=3, type=float, default=[0.223354, -0.203888, 0.027362])
    ap.add_argument("--fresh_tcp_26", nargs=3, type=float, default=[0.225327, -0.205801, 0.028875])
    ap.add_argument("--sweep_v4_candidates", action="store_true")
    ap.add_argument("--v4_moving_overlap_mm", nargs="+", type=float, default=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    ap.add_argument("--v4_counter_clearance_mm", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--v4_top_k", type=int, default=10)
    args = ap.parse_args()

    print("[cube2cm_close_equilibrium_static] static_only=YES isaac_run=NO physics_grasp_validated=NO")
    print(
        "[cube2cm_close_equilibrium_static] diagnostic_only=YES env_default_edits=NO "
        "chain_defaults_edits=NO p7_training=NO constraint_prim_insertion=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "uses_existing_b200_observed_tcp=YES",
        flush=True,
    )

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    plan = _build_plan(args)
    if args.object_center_z is not None:
        center = plan.center.copy()
        center[2] = float(args.object_center_z)
        plan = _build_plan_from_center(args, center, f"{plan.label}_zoverride")

    cube_min, cube_max = _aabb(_box_vertices(plan.center, object_size, plan.yaw_deg))
    target_tcp = np.asarray(args.target_tcp, dtype=np.float64)
    fresh_tcp = {
        23.0: np.asarray(args.fresh_tcp_23, dtype=np.float64),
        26.0: np.asarray(args.fresh_tcp_26, dtype=np.float64),
    }

    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
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
    moving_local_center = (
        np.linalg.inv(_gripper_transform(q_design))
        @ np.array([*moving_world_center, 1.0], dtype=np.float64)
    )[:3]
    if args.counter_mount == "link5":
        counter_local = (
            np.linalg.inv(_link5_transform(q_design))
            @ np.array([*counter_world_center, 1.0], dtype=np.float64)
        )[:3]
    else:
        counter_local = (
            np.linalg.inv(_gripper_transform(q_design))
            @ np.array([*counter_world_center, 1.0], dtype=np.float64)
        )[:3]
    moving_vertices = _box_local_vertices(moving_local_center, moving_size)
    counter_vertices = _box_local_vertices(np.zeros(3, dtype=np.float64), counter_size)

    print(
        f"[cube2cm_close_equilibrium_static] cube_center={_fmt_xyz(plan.center)} "
        f"cube_aabb_min={_fmt_xyz(cube_min)} cube_aabb_max={_fmt_xyz(cube_max)} "
        f"target_tcp={_fmt_xyz(target_tcp)} design_close_deg={args.design_close_deg:.2f}",
        flush=True,
    )
    print(
        f"[cube2cm_close_equilibrium_static] moving_local_center={_fmt_xyz(moving_local_center)} "
        f"counter_local={_fmt_xyz(counter_local)} counter_mount={args.counter_mount} "
        f"counter_is_close_dependent={_yes(args.counter_mount == 'gripper')} "
        f"separate_articulated_opposing_jaw=NO",
        flush=True,
    )

    offsets = {}
    for angle, tcp in fresh_tcp.items():
        delta = tcp - target_tcp
        offsets[angle] = delta
        print(
            f"[cube2cm_close_equilibrium_static] observed angle_deg={angle:.2f} "
            f"fresh_tcp={_fmt_xyz(tcp)} target_error_vec_mm={_fmt_vec_mm(delta)} "
            f"target_error_norm_m={_norm(delta):.6f} target_error_gate_m={args.target_error_gate_m:.6f} "
            f"reached={_yes(_norm(delta) <= args.target_error_gate_m)}",
            flush=True,
        )
    delta_worsen = offsets[26.0] - offsets[23.0]
    print(
        f"[cube2cm_close_equilibrium_static] observed_worsening_23_to_26_mm={_fmt_vec_mm(delta_worsen)} "
        f"norm_increase_m={_norm(offsets[26.0]) - _norm(offsets[23.0]):.6f}",
        flush=True,
    )

    open_contacts = 0
    open_clearance_bad = 0
    q_seed = plan.q_descend_deg.copy()
    open_waypoints = _open_descent_waypoints(args, plan.approach_tcp, plan.descend_tcp)
    print(
        f"[cube2cm_close_equilibrium_static] open_descent_static_check waypoints={len(open_waypoints)} "
        f"open_clearance_gate_m={args.open_clearance_gate_m:.6f}",
        flush=True,
    )
    for idx, waypoint in enumerate(open_waypoints, start=1):
        q_open, ik_ok, ik_err = _solve_q(waypoint, q_seed, GRIPPER_OPEN_DEG, args)
        q_seed = q_open
        moving_open = _transform_points(_gripper_transform(q_open), moving_vertices)
        if args.counter_mount == "link5":
            counter_open = _transform_points(_link5_transform(q_open) @ _translation(counter_local), counter_vertices)
        else:
            counter_open = _transform_points(_gripper_transform(q_open) @ _translation(counter_local), counter_vertices)
        print(
            f"[cube2cm_close_equilibrium_static] open_waypoint index={idx:03d}/{len(open_waypoints):03d} "
            f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err:.3f}",
            flush=True,
        )
        for mesh_name, vertices in (("moving", moving_open), ("counter", counter_open)):
            stats = _contact_stats(f"open_wp{idx:03d}", mesh_name, vertices, cube_min, cube_max, plan.center)
            if stats["contact"] or int(stats["inside"]) > 0:
                open_contacts += 1
            else:
                gap = np.asarray(stats["gap"], dtype=np.float64)
                positive_gap = gap[gap > 0.0]
                sep_gap = 0.0 if positive_gap.size == 0 else float(positive_gap.min())
                if sep_gap < args.open_clearance_gate_m:
                    open_clearance_bad += 1
    print(
        f"[cube2cm_close_equilibrium_static] open_descent_summary open_contacts={open_contacts} "
        f"open_clearance_bad={open_clearance_bad} open_descent_clearance={_yes(open_contacts == 0 and open_clearance_bad == 0)}",
        flush=True,
    )

    q_seed = plan.q_descend_deg.copy()
    for angle in args.close_deg:
        q_close, ik_ok, ik_err = _solve_q(plan.descend_tcp, q_seed, float(angle), args)
        q_seed = q_close
        moving_world = _transform_points(_gripper_transform(q_close), moving_vertices)
        if args.counter_mount == "link5":
            counter_world = _transform_points(_link5_transform(q_close) @ _translation(counter_local), counter_vertices)
        else:
            counter_world = _transform_points(_gripper_transform(q_close) @ _translation(counter_local), counter_vertices)
        print(
            f"[cube2cm_close_equilibrium_static] planned_close angle_deg={angle:.2f} "
            f"ik_ok={_yes(ik_ok)} ik_err_mm={ik_err:.3f}",
            flush=True,
        )
        _contact_stats(f"planned_close_{angle:.2f}", "moving", moving_world, cube_min, cube_max, plan.center)
        _contact_stats(f"planned_close_{angle:.2f}", "counter", counter_world, cube_min, cube_max, plan.center)
        if angle in offsets:
            shift = offsets[angle]
            print(
                f"[cube2cm_close_equilibrium_static] shifted_by_observed_tcp angle_deg={angle:.2f} "
                f"rigid_shift_mm={_fmt_vec_mm(shift)} approximate_only=YES",
                flush=True,
            )
            _contact_stats(
                f"observed_shifted_close_{angle:.2f}",
                "moving",
                moving_world + shift,
                cube_min,
                cube_max,
                plan.center,
            )
            _contact_stats(
                f"observed_shifted_close_{angle:.2f}",
                "counter",
                counter_world + shift,
                cube_min,
                cube_max,
                plan.center,
            )

    if args.sweep_v4_candidates:
        _print_candidate_sweep(
            args,
            plan,
            object_size,
            moving_size,
            counter_size,
            cube_min,
            cube_max,
            open_waypoints,
        )

    if args.counter_mount == "link5":
        hint = (
            "fixed_link5_counter_remains_non_contact_in_planned_close; observed shift moves the fixed counter "
            "farther from cube; moving contact relocates toward cube +x/+z edge instead of forming a two-sided pinch"
        )
    else:
        hint = (
            "gripper_mounted_counter_is_close_dependent_static_candidate; judge it by open_descent_summary and "
            "planned_close counter contact above; this is still not physics grasp validation"
        )
    print(
        f"[cube2cm_close_equilibrium_static] interpretation_hint={hint}; "
        "_grasped_marker alone is insufficient because reached gate fails.",
        flush=True,
    )
    print("[cube2cm_close_equilibrium_static] CUBE2CM_CLOSE_EQUILIBRIUM_STATIC_ANALYSIS_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
