#!/usr/bin/env python3
"""Local v7 object-frame fixed-jaw offset analysis for P7 Branch B cube grasp.

This script is static/read-only. It does not launch Isaac, train, generate a
dataset, insert constraints, attach SurfaceGripper, transport, release, or edit
defaults. It models the professor-style fixed-jaw object-frame primitive and
compares it against the latest v6 authored/static and runtime telemetry facts.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np


V6_DESIGN_MOVING_CENTER_OBJ = np.array([0.000000, 0.016500, 0.002000], dtype=np.float64)
V6_DESIGN_COUNTER_CENTER_OBJ = np.array([0.001000, -0.011500, 0.002000], dtype=np.float64)
V6_RUNTIME_MOVING_CENTER_OBJ = np.array([0.017977, 0.001313, 0.011828], dtype=np.float64)
V6_RUNTIME_COUNTER_CENTER_OBJ = np.array([0.019788, -0.026548, 0.014168], dtype=np.float64)
V6_RUNTIME_COUNTER_GAP_OBJ = np.array([0.000260, 0.006757, 0.000000], dtype=np.float64)
V6_RUNTIME_TARGET_ERROR_M = 0.024584
CONTACT_EPS = 1.0e-9


def _fmt_xyz(values: np.ndarray) -> str:
    return "([" + ", ".join(f"{float(v):+.6f}" for v in values) + "])"


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _box_min_max(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * size
    return center - half, center + half


def _axis_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)


def _overlap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))


def _contact_stats(center: np.ndarray, size: np.ndarray, object_size: np.ndarray, slop_m: float) -> dict[str, object]:
    jaw_min, jaw_max = _box_min_max(center, size)
    cube_min = -0.5 * object_size
    cube_max = 0.5 * object_size
    strict_overlap = _overlap(jaw_min, jaw_max, cube_min, cube_max)
    strict_gap = _axis_gap(jaw_min, jaw_max, cube_min, cube_max)
    slop_min = cube_min - float(slop_m)
    slop_max = cube_max + float(slop_m)
    slop_overlap = _overlap(jaw_min, jaw_max, slop_min, slop_max)
    slop_gap = _axis_gap(jaw_min, jaw_max, slop_min, slop_max)
    return {
        "strict_contact": bool(np.all(strict_overlap > CONTACT_EPS)),
        "slop_contact": bool(np.all(slop_overlap > CONTACT_EPS)),
        "strict_overlap": strict_overlap,
        "strict_gap": strict_gap,
        "slop_gap": slop_gap,
    }


@dataclass(frozen=True)
class ModelSample:
    fixed_center: np.ndarray
    moving_open_center: np.ndarray
    moving_close_center: np.ndarray
    fixed_open: dict[str, object]
    moving_open: dict[str, object]
    fixed_close: dict[str, object]
    moving_close: dict[str, object]

    @property
    def open_descent_clearance(self) -> bool:
        return not bool(self.fixed_open["strict_contact"]) and not bool(self.moving_open["strict_contact"])

    @property
    def close_both_contact_with_slop(self) -> bool:
        return bool(self.fixed_close["slop_contact"]) and bool(self.moving_close["slop_contact"])

    @property
    def close_both_strict_contact(self) -> bool:
        return bool(self.fixed_close["strict_contact"]) and bool(self.moving_close["strict_contact"])


def _side_sign(name: str) -> float:
    if name == "neg_y":
        return -1.0
    if name == "pos_y":
        return 1.0
    raise ValueError(f"unsupported fixed_jaw_side: {name}")


def _sample_model(
    object_size: np.ndarray,
    fixed_side: str,
    fixed_alpha_m: float,
    moving_alpha_m: float,
    fixed_counter_y_offset_m: float,
    fixed_counter_x_offset_m: float,
    moving_x_offset_m: float,
    z_contact_m: float,
    moving_jaw_size: np.ndarray,
    counter_jaw_size: np.ndarray,
    moving_open_clearance_m: float,
    contact_slop_m: float,
) -> ModelSample:
    sign = _side_sign(fixed_side)
    moving_sign = -sign
    half_y = 0.5 * object_size[1]
    fixed_center = np.array(
        [
            fixed_counter_x_offset_m,
            sign * (half_y + 0.5 * counter_jaw_size[1] - fixed_alpha_m) + fixed_counter_y_offset_m,
            z_contact_m,
        ],
        dtype=np.float64,
    )
    moving_open_center = np.array(
        [
            moving_x_offset_m,
            moving_sign * (half_y + 0.5 * moving_jaw_size[1] + moving_open_clearance_m),
            z_contact_m,
        ],
        dtype=np.float64,
    )
    moving_close_center = np.array(
        [
            moving_x_offset_m,
            moving_sign * (half_y + 0.5 * moving_jaw_size[1] - moving_alpha_m),
            z_contact_m,
        ],
        dtype=np.float64,
    )
    return ModelSample(
        fixed_center=fixed_center,
        moving_open_center=moving_open_center,
        moving_close_center=moving_close_center,
        fixed_open=_contact_stats(fixed_center, counter_jaw_size, object_size, contact_slop_m),
        moving_open=_contact_stats(moving_open_center, moving_jaw_size, object_size, contact_slop_m),
        fixed_close=_contact_stats(fixed_center, counter_jaw_size, object_size, contact_slop_m),
        moving_close=_contact_stats(moving_close_center, moving_jaw_size, object_size, contact_slop_m),
    )


def _implied_fixed_alpha(center_y: float, object_b: float, thickness_y: float, fixed_side: str) -> float:
    sign = _side_sign(fixed_side)
    return 0.5 * object_b + 0.5 * thickness_y - sign * center_y


def _implied_moving_alpha(center_y: float, object_b: float, thickness_y: float, fixed_side: str) -> float:
    moving_sign = -_side_sign(fixed_side)
    return 0.5 * object_b + 0.5 * thickness_y - moving_sign * center_y


def _print_sample(prefix: str, sample: ModelSample) -> None:
    print(
        f"[cube2cm_v7_object_frame_static] {prefix} "
        f"fixed_center_obj={_fmt_xyz(sample.fixed_center)} "
        f"moving_open_center_obj={_fmt_xyz(sample.moving_open_center)} "
        f"moving_close26_center_obj={_fmt_xyz(sample.moving_close_center)} "
        f"open_descent_clearance={_yes(sample.open_descent_clearance)} "
        f"fixed_open_strict_contact={_yes(bool(sample.fixed_open['strict_contact']))} "
        f"moving_open_strict_contact={_yes(bool(sample.moving_open['strict_contact']))} "
        f"fixed_close_strict_contact={_yes(bool(sample.fixed_close['strict_contact']))} "
        f"moving_close_strict_contact={_yes(bool(sample.moving_close['strict_contact']))} "
        f"fixed_close_slop_contact={_yes(bool(sample.fixed_close['slop_contact']))} "
        f"moving_close_slop_contact={_yes(bool(sample.moving_close['slop_contact']))} "
        f"fixed_close_gap_m={_fmt_xyz(np.asarray(sample.fixed_close['strict_gap']))} "
        f"moving_close_gap_m={_fmt_xyz(np.asarray(sample.moving_close['strict_gap']))}",
        flush=True,
    )


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.030, 0.030, 0.030])
    ap.add_argument("--fixed_jaw_side", choices=["neg_y", "pos_y"], default="neg_y")
    ap.add_argument("--fixed_alpha_m", type=float, default=0.0)
    ap.add_argument("--moving_alpha_m", type=float, default=0.0015)
    ap.add_argument("--fixed_counter_y_offset_m", type=float, default=0.0)
    ap.add_argument("--fixed_counter_x_offset_m", type=float, default=0.0010)
    ap.add_argument("--moving_x_offset_m", type=float, default=0.0)
    ap.add_argument("--z_contact_m", type=float, default=0.0020)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0050, 0.008])
    ap.add_argument("--moving_open_clearance_m", type=float, default=0.0060)
    ap.add_argument("--contact_slop_m", type=float, default=0.0010)
    ap.add_argument("--alpha_sweep_m", default="-0.001,0.000,0.001,0.002,0.003,0.004,0.006")
    ap.add_argument("--counter_y_sweep_m", default="-0.004,-0.002,0.000,0.002,0.004,0.006,0.008,0.010")
    ap.add_argument("--counter_thickness_y_sweep_m", default="0.003,0.005,0.007")
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--print_all", action="store_true")
    args = ap.parse_args()

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    if object_size.shape != (3,) or moving_size.shape != (3,) or counter_size.shape != (3,):
        raise ValueError("object and jaw sizes must be 3-vectors")
    if np.any(object_size <= 0.0) or np.any(moving_size <= 0.0) or np.any(counter_size <= 0.0):
        raise ValueError("object and jaw sizes must be positive")
    if abs(float(args.yaw_deg)) > 1.0e-12:
        raise ValueError("v7 first-pass object-frame analysis is intentionally yaw=0 first")

    alpha_values = _parse_float_list(args.alpha_sweep_m)
    counter_y_values = _parse_float_list(args.counter_y_sweep_m)
    thickness_values = _parse_float_list(args.counter_thickness_y_sweep_m)

    print("[cube2cm_v7_object_frame_static] local_static_only=YES isaac_run=NO training=NO dataset_generation=NO")
    print(
        "[cube2cm_v7_object_frame_static] diagnostic_only=YES env_default_edits=NO chain_defaults_edits=NO "
        "constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO transport_target=NO "
        "release_marker=NO scalar_or_gate_tuning=NO success_claim=NO",
        flush=True,
    )
    print(
        f"[cube2cm_v7_object_frame_static] selected object_size_m={_fmt_xyz(object_size)} "
        f"fixed_jaw_side={args.fixed_jaw_side} yaw_deg={args.yaw_deg:.1f} "
        f"z_contact_m={float(args.z_contact_m):+.6f} contact_slop_m={float(args.contact_slop_m):.6f}",
        flush=True,
    )

    v6_fixed_alpha = _implied_fixed_alpha(
        V6_DESIGN_COUNTER_CENTER_OBJ[1], object_size[1], counter_size[1], args.fixed_jaw_side
    )
    v6_moving_alpha = _implied_moving_alpha(
        V6_DESIGN_MOVING_CENTER_OBJ[1], object_size[1], moving_size[1], args.fixed_jaw_side
    )
    runtime_fixed_alpha = _implied_fixed_alpha(
        V6_RUNTIME_COUNTER_CENTER_OBJ[1], object_size[1], counter_size[1], args.fixed_jaw_side
    )
    runtime_moving_alpha = _implied_moving_alpha(
        V6_RUNTIME_MOVING_CENTER_OBJ[1], object_size[1], moving_size[1], args.fixed_jaw_side
    )
    fixed_tangent_y = _side_sign(args.fixed_jaw_side) * (0.5 * object_size[1] + 0.5 * counter_size[1])
    required_runtime_counter_shift_y = fixed_tangent_y - V6_RUNTIME_COUNTER_CENTER_OBJ[1]
    required_runtime_counter_shift_x = 0.0 - V6_RUNTIME_COUNTER_CENTER_OBJ[0]

    print(
        "[cube2cm_v7_object_frame_static] v6_compare "
        "source_design='runtime telemetry line 66 / v6 prep lines 23-26' "
        "source_runtime='runtime telemetry line 417' "
        f"v6_design_moving_center_obj={_fmt_xyz(V6_DESIGN_MOVING_CENTER_OBJ)} "
        f"v6_design_counter_center_obj={_fmt_xyz(V6_DESIGN_COUNTER_CENTER_OBJ)} "
        f"v6_runtime_moving_center_obj={_fmt_xyz(V6_RUNTIME_MOVING_CENTER_OBJ)} "
        f"v6_runtime_counter_center_obj={_fmt_xyz(V6_RUNTIME_COUNTER_CENTER_OBJ)} "
        f"v6_runtime_counter_gap_obj_m={_fmt_xyz(V6_RUNTIME_COUNTER_GAP_OBJ)} "
        f"v6_runtime_target_error_m={V6_RUNTIME_TARGET_ERROR_M:.6f}",
        flush=True,
    )
    print(
        "[cube2cm_v7_object_frame_static] v6_implied_offsets "
        f"design_fixed_alpha_m={v6_fixed_alpha:+.6f} "
        f"design_moving_alpha_m={v6_moving_alpha:+.6f} "
        f"runtime_fixed_alpha_m={runtime_fixed_alpha:+.6f} "
        f"runtime_moving_alpha_m={runtime_moving_alpha:+.6f} "
        f"runtime_counter_shift_to_fixed_tangent_m=(["
        f"{required_runtime_counter_shift_x:+.6f}, {required_runtime_counter_shift_y:+.6f}, +0.000000])",
        flush=True,
    )

    baseline = _sample_model(
        object_size,
        args.fixed_jaw_side,
        float(args.fixed_alpha_m),
        float(args.moving_alpha_m),
        float(args.fixed_counter_y_offset_m),
        float(args.fixed_counter_x_offset_m),
        float(args.moving_x_offset_m),
        float(args.z_contact_m),
        moving_size,
        counter_size,
        float(args.moving_open_clearance_m),
        float(args.contact_slop_m),
    )
    _print_sample("baseline", baseline)

    hit_count = 0
    strict_hit_count = 0
    open_clear_count = 0
    best: tuple[float, float, float, ModelSample] | None = None
    for alpha_m, counter_y_m, thickness_y_m in itertools.product(alpha_values, counter_y_values, thickness_values):
        counter_sweep_size = counter_size.copy()
        counter_sweep_size[1] = float(thickness_y_m)
        sample = _sample_model(
            object_size,
            args.fixed_jaw_side,
            float(alpha_m),
            float(args.moving_alpha_m),
            float(counter_y_m),
            float(args.fixed_counter_x_offset_m),
            float(args.moving_x_offset_m),
            float(args.z_contact_m),
            moving_size,
            counter_sweep_size,
            float(args.moving_open_clearance_m),
            float(args.contact_slop_m),
        )
        open_clear_count += int(sample.open_descent_clearance)
        hit = sample.open_descent_clearance and sample.close_both_contact_with_slop
        strict_hit = sample.open_descent_clearance and sample.close_both_strict_contact
        hit_count += int(hit)
        strict_hit_count += int(strict_hit)
        if hit and best is None:
            best = (float(alpha_m), float(counter_y_m), float(thickness_y_m), sample)
        if args.print_all or hit or strict_hit:
            print(
                "[cube2cm_v7_object_frame_static] sensitivity "
                f"fixed_alpha_m={float(alpha_m):+.6f} "
                f"counter_y_offset_m={float(counter_y_m):+.6f} "
                f"counter_thickness_y_m={float(thickness_y_m):.6f} "
                f"open_clearance={_yes(sample.open_descent_clearance)} "
                f"close_both_slop_contact={_yes(hit)} "
                f"close_both_strict_contact={_yes(strict_hit)} "
                f"fixed_center_obj={_fmt_xyz(sample.fixed_center)} "
                f"moving_close26_center_obj={_fmt_xyz(sample.moving_close_center)} "
                f"fixed_gap_m={_fmt_xyz(np.asarray(sample.fixed_close['strict_gap']))} "
                f"moving_gap_m={_fmt_xyz(np.asarray(sample.moving_close['strict_gap']))}",
                flush=True,
            )

    if best is not None:
        alpha_m, counter_y_m, thickness_y_m, sample = best
        print(
            "[cube2cm_v7_object_frame_static] best_first_hit "
            f"fixed_alpha_m={alpha_m:+.6f} counter_y_offset_m={counter_y_m:+.6f} "
            f"counter_thickness_y_m={thickness_y_m:.6f} "
            f"fixed_center_obj={_fmt_xyz(sample.fixed_center)} "
            f"moving_close26_center_obj={_fmt_xyz(sample.moving_close_center)} "
            f"open_clearance={_yes(sample.open_descent_clearance)} "
            f"close_both_slop_contact={_yes(sample.close_both_contact_with_slop)} "
            f"close_both_strict_contact={_yes(sample.close_both_strict_contact)}",
            flush=True,
        )
    else:
        print("[cube2cm_v7_object_frame_static] best_first_hit NONE", flush=True)

    dynamic_push_issue = True
    authoring_offset_issue = bool(required_runtime_counter_shift_y > 0.003 or required_runtime_counter_shift_x < -0.003)
    if authoring_offset_issue and dynamic_push_issue:
        next_candidate = "C_BOTH_authoring_offset_correction_and_dynamic_push_issue"
    elif authoring_offset_issue:
        next_candidate = "A_authoring_offset_correction"
    else:
        next_candidate = "B_dynamic_push_issue"
    print(
        "[cube2cm_v7_object_frame_static] summary "
        f"sweep_total={len(alpha_values) * len(counter_y_values) * len(thickness_values)} "
        f"open_clear_count={open_clear_count} close_both_slop_hits={hit_count} "
        f"close_both_strict_hits={strict_hit_count} "
        "v6_static_authoring_looked_plausible=YES "
        "v6_runtime_close26_failed=YES "
        f"authoring_offset_issue={_yes(authoring_offset_issue)} "
        f"dynamic_push_issue={_yes(dynamic_push_issue)} "
        f"next_candidate={next_candidate} "
        "physics_success_claim=NO",
        flush=True,
    )
    print("[cube2cm_v7_object_frame_static] CUBE2CM_V7_OBJECT_FRAME_STATIC_ANALYSIS_DONE=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
