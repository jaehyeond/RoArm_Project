#!/usr/bin/env python3
"""2cm cube normalized grasp static probe for P7 Branch B.

This is a local, numpy-only diagnostic. It does not launch Isaac, does not train,
does not edit env/train/chain defaults, does not insert constraints, does not use
SurfaceGripper, and does not claim physical grasp success. It audits whether a
2cm cube normalized grasp generator produces RoArm TCP targets that are reachable
by the existing position-only IK, across the current workspace source regions.
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

from roarm_kinematics import JOINT_LIMITS_DEG, clip_joints, fk_tcp, ik_dls  # noqa: E402


# Mirrors the current env/source constants for this static diagnostic only.
TABLE_Z = -0.012117
HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float64)
PICK_WRIST_R_DEG = 90.0
GRIPPER_OPEN_DEG = 0.0
GRIPPER_LATCH_DEG = 26.0
GRIPPER_CLOSE_DEG = 45.84
GRASP_DISTANCE_THRESH_M = 0.025
GRASP_GRIPPER_THRESH_RAD = 0.4

SOURCE_REGIONS = (
    (0.150, 0.250, -0.220, -0.130),
    (0.150, 0.250, +0.070, +0.200),
    (0.330, 0.430, -0.220, -0.100),
    (0.330, 0.430, +0.050, +0.200),
)

FOUR_SPONGE_SEED0_SOURCES = (
    (+0.21369616873214542, -0.19571919576125169),
    (+0.15165276355285290, +0.17572513109603544),
    (+0.39066357757671800, -0.13246041268192021),
    (+0.42350724237877680, +0.17237803311822986),
)


@dataclass(frozen=True)
class ObjectPose:
    label: str
    center: np.ndarray
    yaw_deg: float


@dataclass(frozen=True)
class GraspSpec:
    name: str
    normalized: np.ndarray


@dataclass(frozen=True)
class IkStage:
    name: str
    target: np.ndarray
    q_deg: np.ndarray
    ik_converged: bool
    ik_err_mm: float
    ik_iter: int
    fk_error_m: float
    joint_margin_deg: float


@dataclass(frozen=True)
class CandidateResult:
    pose: ObjectPose
    grasp: GraspSpec
    world_grasp: np.ndarray
    stages: tuple[IkStage, IkStage, IkStage, IkStage]
    tcp_to_object_center_m: float
    latch_geometry_possible: bool
    min_latch_close_deg: float | None
    close_candidates_ok: tuple[float, ...]
    max_fk_error_m: float
    max_tcp_gap_m: float
    resample_needed: bool
    static_reach_ok: bool
    verdict: str
    reason: str


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _fmt_xyz(value: np.ndarray) -> str:
    return f"([{value[0]:+.6f}, {value[1]:+.6f}, {value[2]:+.6f}])"


def _fmt_norm(value: np.ndarray) -> str:
    return f"([{value[0]:+.3f}, {value[1]:+.3f}, {value[2]:+.3f}])"


def _rot_z(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _drive_joint_margin_deg(q_deg: np.ndarray) -> float:
    margins: list[float] = []
    # Existing top-down pick convention intentionally forces wrist_r to +90 deg,
    # which is exactly the configured wrist_r high limit. Treating that as a
    # reachability failure would reject the current known pick convention itself.
    for idx, name in enumerate(["base", "shoulder", "elbow", "wrist_p"]):
        lo, hi = JOINT_LIMITS_DEG[name]
        margins.append(float(min(q_deg[idx] - lo, hi - q_deg[idx])))
    return min(margins)


def _solve_stage(name: str, target: np.ndarray, seed_q: np.ndarray, gripper_deg: float, args: argparse.Namespace) -> IkStage:
    q, converged, err_mm, n_iter = ik_dls(
        target,
        seed_q,
        max_iter=args.ik_max_iter,
        tol_mm=args.ik_tol_mm,
    )
    q = clip_joints(q)
    q[4] = PICK_WRIST_R_DEG
    q[5] = gripper_deg
    fk_error = _norm(fk_tcp(q) - target)
    return IkStage(
        name=name,
        target=target.copy(),
        q_deg=q.copy(),
        ik_converged=bool(converged),
        ik_err_mm=float(err_mm),
        ik_iter=int(n_iter),
        fk_error_m=fk_error,
        joint_margin_deg=_drive_joint_margin_deg(q),
    )


def _grasp_specs(kind: str) -> list[GraspSpec]:
    core = [
        GraspSpec("top_center", np.array([0.0, 0.0, 0.5], dtype=np.float64)),
        GraspSpec("top_pos_x", np.array([0.35, 0.0, 0.5], dtype=np.float64)),
        GraspSpec("top_neg_x", np.array([-0.35, 0.0, 0.5], dtype=np.float64)),
        GraspSpec("top_pos_y", np.array([0.0, 0.35, 0.5], dtype=np.float64)),
        GraspSpec("top_neg_y", np.array([0.0, -0.35, 0.5], dtype=np.float64)),
    ]
    if kind == "core":
        return core
    corners = [
        GraspSpec("top_pos_x_pos_y", np.array([0.35, 0.35, 0.5], dtype=np.float64)),
        GraspSpec("top_pos_x_neg_y", np.array([0.35, -0.35, 0.5], dtype=np.float64)),
        GraspSpec("top_neg_x_pos_y", np.array([-0.35, 0.35, 0.5], dtype=np.float64)),
        GraspSpec("top_neg_x_neg_y", np.array([-0.35, -0.35, 0.5], dtype=np.float64)),
    ]
    return core + corners


def _sample_workspace(grid_per_region: int, include_seed0_sources: bool) -> list[tuple[str, float, float]]:
    samples: list[tuple[str, float, float]] = []
    for region_idx, (x_min, x_max, y_min, y_max) in enumerate(SOURCE_REGIONS, start=1):
        xs = np.linspace(x_min, x_max, grid_per_region)
        ys = np.linspace(y_min, y_max, grid_per_region)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                samples.append((f"R{region_idx}_gx{ix}_gy{iy}", float(x), float(y)))
    if include_seed0_sources:
        for idx, (x, y) in enumerate(FOUR_SPONGE_SEED0_SOURCES, start=1):
            samples.append((f"seed0_S{idx}", float(x), float(y)))

    deduped: list[tuple[str, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for label, x, y in samples:
        key = (round(x * 1_000_000), round(y * 1_000_000))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, x, y))
    return deduped


def _candidate_result(pose: ObjectPose, grasp: GraspSpec, object_size: np.ndarray, args: argparse.Namespace) -> CandidateResult:
    rot = _rot_z(pose.yaw_deg)
    local_grasp = grasp.normalized * object_size
    world_grasp = pose.center + rot @ local_grasp

    approach = world_grasp + np.array([0.0, 0.0, args.approach_clearance_m], dtype=np.float64)
    pregrasp = world_grasp + np.array([0.0, 0.0, args.pregrasp_clearance_m], dtype=np.float64)
    final = world_grasp + np.array([0.0, 0.0, args.grasp_surface_margin_m], dtype=np.float64)
    lift = final + np.array([0.0, 0.0, args.static_lift_probe_m], dtype=np.float64)

    q_seed = HOME_DEG.copy()
    q_seed[5] = GRIPPER_OPEN_DEG
    approach_stage = _solve_stage("approach", approach, q_seed, GRIPPER_OPEN_DEG, args)
    pregrasp_stage = _solve_stage("pregrasp", pregrasp, approach_stage.q_deg, GRIPPER_OPEN_DEG, args)
    grasp_stage = _solve_stage("grasp", final, pregrasp_stage.q_deg, GRIPPER_OPEN_DEG, args)
    lift_seed = grasp_stage.q_deg.copy()
    lift_seed[5] = GRIPPER_LATCH_DEG
    lift_stage = _solve_stage("static_lift_probe", lift, lift_seed, GRIPPER_LATCH_DEG, args)
    stages = (approach_stage, pregrasp_stage, grasp_stage, lift_stage)

    tcp_to_object_center = _norm(final - pose.center)
    latch_geometry_possible = tcp_to_object_center < GRASP_DISTANCE_THRESH_M
    close_candidates_ok = tuple(
        close_deg
        for close_deg in args.close_deg
        if math.radians(close_deg) >= GRASP_GRIPPER_THRESH_RAD and latch_geometry_possible
    )
    min_latch_close_deg = close_candidates_ok[0] if close_candidates_ok else None
    max_fk_error = max(stage.fk_error_m for stage in stages)
    tcp_points = [stage.target for stage in stages]
    max_gap = max(_norm(tcp_points[i] - tcp_points[i - 1]) for i in range(1, len(tcp_points)))
    resample_needed = max_gap > args.max_tcp_step_m
    stage_ok = all(stage.ik_converged and stage.fk_error_m <= args.fk_error_gate_m for stage in stages)
    margin_ok = all(stage.joint_margin_deg >= args.joint_margin_gate_deg for stage in stages)
    static_reach_ok = stage_ok and margin_ok and latch_geometry_possible and bool(close_candidates_ok)

    failed = []
    for stage in stages:
        if not stage.ik_converged:
            failed.append(f"{stage.name}_ik")
        if stage.fk_error_m > args.fk_error_gate_m:
            failed.append(f"{stage.name}_fk")
        if stage.joint_margin_deg < args.joint_margin_gate_deg:
            failed.append(f"{stage.name}_joint_margin")
    if not latch_geometry_possible:
        failed.append("latch_geometry")
    if not close_candidates_ok:
        failed.append("close_sweep_below_latch_threshold")
    verdict = "STATIC_REACH_PASS" if static_reach_ok else "STATIC_REACH_FAIL"
    reason = "ok" if static_reach_ok else ",".join(failed)

    return CandidateResult(
        pose=pose,
        grasp=grasp,
        world_grasp=world_grasp,
        stages=stages,
        tcp_to_object_center_m=tcp_to_object_center,
        latch_geometry_possible=latch_geometry_possible,
        min_latch_close_deg=min_latch_close_deg,
        close_candidates_ok=close_candidates_ok,
        max_fk_error_m=max_fk_error,
        max_tcp_gap_m=max_gap,
        resample_needed=resample_needed,
        static_reach_ok=static_reach_ok,
        verdict=verdict,
        reason=reason,
    )


def _print_candidate(result: CandidateResult) -> None:
    stages = {stage.name: stage for stage in result.stages}
    close_ok = "/".join(f"{x:.2f}" for x in result.close_candidates_ok) if result.close_candidates_ok else "NONE"
    print(
        "[cube2cm_static] candidate "
        f"pose={result.pose.label} center={_fmt_xyz(result.pose.center)} yaw_deg={result.pose.yaw_deg:.1f} "
        f"grasp={result.grasp.name} norm={_fmt_norm(result.grasp.normalized)} "
        f"world_grasp={_fmt_xyz(result.world_grasp)} "
        f"approach_fk_m={stages['approach'].fk_error_m:.6f} "
        f"pregrasp_fk_m={stages['pregrasp'].fk_error_m:.6f} "
        f"grasp_fk_m={stages['grasp'].fk_error_m:.6f} "
        f"lift_fk_m={stages['static_lift_probe'].fk_error_m:.6f} "
        f"min_joint_margin_deg={min(stage.joint_margin_deg for stage in result.stages):.3f} "
        f"tcp_to_object_center_m={result.tcp_to_object_center_m:.6f} "
        f"latch_geometry_possible={_yes(result.latch_geometry_possible)} "
        f"min_latch_close_deg={'NONE' if result.min_latch_close_deg is None else f'{result.min_latch_close_deg:.2f}'} "
        f"close_candidates_ok={close_ok} "
        f"max_tcp_gap_m={result.max_tcp_gap_m:.6f} resample_needed={_yes(result.resample_needed)} "
        f"verdict={result.verdict} reason={result.reason}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--grid_per_region", type=int, default=3)
    ap.add_argument("--yaw_deg", nargs="+", type=float, default=[0.0, 45.0, 90.0, 135.0])
    ap.add_argument("--grasp_set", choices=["core", "dense"], default="core")
    ap.add_argument("--include_seed0_sources", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--pregrasp_clearance_m", type=float, default=0.015)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--static_lift_probe_m", type=float, default=0.010)
    ap.add_argument("--fk_error_gate_m", type=float, default=0.003)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--joint_margin_gate_deg", type=float, default=0.25)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0, 35.0, 40.0, 45.84])
    ap.add_argument("--print_all", action="store_true")
    ap.add_argument("--max_failure_rows", type=int, default=80)
    args = ap.parse_args()

    if args.grid_per_region < 1:
        raise ValueError("grid_per_region must be >= 1")
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")
    if args.pregrasp_clearance_m <= args.grasp_surface_margin_m:
        raise ValueError("pregrasp_clearance_m must be above grasp_surface_margin_m")
    if args.approach_clearance_m <= args.pregrasp_clearance_m:
        raise ValueError("approach_clearance_m must be above pregrasp_clearance_m")
    if args.static_lift_probe_m <= 0.0:
        raise ValueError("static_lift_probe_m must be positive")
    if sorted(args.close_deg) != list(args.close_deg):
        raise ValueError("close_deg values must be sorted ascending")

    object_center_z = TABLE_Z + float(object_size[2]) / 2.0
    top_z = object_center_z + float(object_size[2]) / 2.0
    latch_threshold_deg = math.degrees(GRASP_GRIPPER_THRESH_RAD)
    workspace = _sample_workspace(args.grid_per_region, args.include_seed0_sources)
    grasps = _grasp_specs(args.grasp_set)

    print("[cube2cm_static] 2cm cube normalized grasp static probe", flush=True)
    print(
        "[cube2cm_static] "
        "static_only=YES isaac_run=NO physics_grasp_validated=NO constraint_prim_insertion=NO "
        "surface_gripper=NO transport_target=NO release_marker=NO p7_training=NO "
        "env_default_edits=NO chain_defaults_edits=NO",
        flush=True,
    )
    print(
        f"[cube2cm_static] geometry object_size_m={_fmt_xyz(object_size)} table_z_m={TABLE_Z:.6f} "
        f"object_center_z_m={object_center_z:.6f} object_top_z_m={top_z:.6f} "
        "height_axis=z normalized_grasp_range=[-0.5,+0.5]^3 "
        f"workspace_points={len(workspace)} yaw_count={len(args.yaw_deg)} grasp_count={len(grasps)}",
        flush=True,
    )
    print(
        f"[cube2cm_static] gates fk_error_gate_m={args.fk_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} joint_margin_gate_deg={args.joint_margin_gate_deg:.3f} "
        f"grasp_distance_thresh_m={GRASP_DISTANCE_THRESH_M:.6f} "
        f"grasp_gripper_thresh_rad={GRASP_GRIPPER_THRESH_RAD:.6f} "
        f"grasp_gripper_thresh_deg={latch_threshold_deg:.3f} "
        f"close_sweep_deg={','.join(f'{x:.2f}' for x in args.close_deg)}",
        flush=True,
    )

    results: list[CandidateResult] = []
    by_pose: dict[str, list[CandidateResult]] = {}
    failure_rows_printed = 0
    for label, x, y in workspace:
        for yaw in args.yaw_deg:
            pose = ObjectPose(
                label=f"{label}_yaw{str(yaw).replace('.', 'p')}",
                center=np.array([x, y, object_center_z], dtype=np.float64),
                yaw_deg=float(yaw),
            )
            for grasp in grasps:
                result = _candidate_result(pose, grasp, object_size, args)
                results.append(result)
                by_pose.setdefault(pose.label, []).append(result)
                should_print = args.print_all or (
                    not result.static_reach_ok and failure_rows_printed < args.max_failure_rows
                )
                if should_print:
                    _print_candidate(result)
                    if not result.static_reach_ok:
                        failure_rows_printed += 1

    total = len(results)
    passed = sum(1 for result in results if result.static_reach_ok)
    pose_total = len(by_pose)
    pose_any_pass = sum(1 for pose_results in by_pose.values() if any(r.static_reach_ok for r in pose_results))
    pose_all_pass = sum(1 for pose_results in by_pose.values() if all(r.static_reach_ok for r in pose_results))
    resample_needed = sum(1 for result in results if result.resample_needed)
    min_close_values = [r.min_latch_close_deg for r in results if r.min_latch_close_deg is not None]
    global_min_close = min(min_close_values) if min_close_values else float("nan")
    max_fk_error = max((r.max_fk_error_m for r in results), default=float("inf"))
    max_tcp_gap = max((r.max_tcp_gap_m for r in results), default=float("inf"))

    print(
        f"[cube2cm_static] aggregate candidates={total} pass={passed} fail={total - passed} "
        f"pass_rate={passed / max(total, 1):.6f} pose_cells={pose_total} "
        f"pose_any_pass={pose_any_pass} pose_all_pass={pose_all_pass} "
        f"resample_needed_candidates={resample_needed} "
        f"global_min_latch_close_deg={global_min_close:.3f} "
        f"max_fk_error_m={max_fk_error:.6f} max_tcp_gap_m={max_tcp_gap:.6f}",
        flush=True,
    )

    print("[cube2cm_static] pose_summary_begin", flush=True)
    for pose_label, pose_results in sorted(by_pose.items()):
        pose_pass = sum(1 for r in pose_results if r.static_reach_ok)
        best = min(pose_results, key=lambda r: (not r.static_reach_ok, r.max_fk_error_m))
        print(
            f"[cube2cm_static] pose_summary pose={pose_label} pass={pose_pass}/{len(pose_results)} "
            f"best_grasp={best.grasp.name} best_verdict={best.verdict} "
            f"best_max_fk_error_m={best.max_fk_error_m:.6f} "
            f"best_min_latch_close_deg={'NONE' if best.min_latch_close_deg is None else f'{best.min_latch_close_deg:.2f}'} "
            f"best_reason={best.reason}",
            flush=True,
        )
    print("[cube2cm_static] pose_summary_end", flush=True)

    success = passed > 0 and pose_any_pass == pose_total
    print(
        f"[cube2cm_static] CUBE2CM_NORMALIZED_GRASP_STATIC_REACHABILITY_SUCCESS={_yes(success)}",
        flush=True,
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
