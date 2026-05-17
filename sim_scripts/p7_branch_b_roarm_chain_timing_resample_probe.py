#!/usr/bin/env python3
"""RoArm chain-side TCP resampling/timing dry-run for P7 Branch B.

This diagnostic strengthens the prior chain-contract dry-run:

- validates the full HOME -> grasp pre-close TCP path separately;
- validates attached grasp -> transport with true final stream FK error;
- emits only proposed command events, with no Isaac Sim, no constraints, no
  SurfaceGripper, and no chain integration.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import fk_tcp, ik_dls, clip_joints  # noqa: E402


def _load_chain_skills_module():
    spec = importlib.util.spec_from_file_location("chain_skills_local", REPO / "roarm_rl/chain_skills.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load roarm_rl/chain_skills.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chain = _load_chain_skills_module()

GRIPPER_LATCH_DEG = _chain.GRIPPER_LATCH_DEG
GRIPPER_OPEN_DEG = _chain.GRIPPER_OPEN_DEG
HIGH_TCP_Z = _chain.HIGH_TCP_Z
HOME_DEG = _chain.HOME_DEG
L1_SP1 = _chain.L1_SP1
PICK_WRIST_R_DEG = _chain.PICK_WRIST_R_DEG
SPONGE_CENTER_Z = _chain.SPONGE_CENTER_Z
TCP_PICK_GRASP_Z = _chain.TCP_PICK_GRASP_Z
TCP_RELEASE_ENTRY_Z = _chain.TCP_RELEASE_ENTRY_Z
TrajectoryPlanner = _chain.TrajectoryPlanner


@dataclass(frozen=True)
class Waypoint:
    name: str
    target_xyz: np.ndarray
    q_deg: np.ndarray
    force_pick_wrist_roll: bool


@dataclass
class StreamStats:
    ok: bool = True
    n_cmds: int = 0
    max_tcp_step_m: float = 0.0
    max_fk_error_m: float = 0.0
    max_endpoint_error_m: float = 0.0
    max_ik_err_mm: float = 0.0
    ik_failures: int = 0
    final_q_deg: np.ndarray | None = None
    final_tcp: np.ndarray | None = None


def _fmt_xyz(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _waypoints(planner: TrajectoryPlanner) -> list[Waypoint]:
    pick_xy = planner.pick_xy
    place = planner.place_xyz
    return [
        Waypoint("home", fk_tcp(HOME_DEG), HOME_DEG.copy(), False),
        Waypoint("high", np.array([pick_xy[0], pick_xy[1], HIGH_TCP_Z]), planner.q_high_deg.copy(), True),
        Waypoint("hover", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + planner.hover_offset_z]), planner.q_hover_deg.copy(), True),
        Waypoint("1b1_z59", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.012]), planner.q_1b1_deg.copy(), True),
        Waypoint("1b2_z53", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.006]), planner.q_1b2_deg.copy(), True),
        Waypoint("grasp", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z]), planner.q_grasp_deg.copy(), True),
        Waypoint("transport_hover", np.array([place[0], place[1], TCP_RELEASE_ENTRY_Z]), planner.q_transport_deg.copy(), True),
    ]


def _resample_segment(
    *,
    phase: str,
    from_name: str,
    to_wp: Waypoint,
    start_tcp: np.ndarray,
    start_q_deg: np.ndarray,
    max_tcp_step_m: float,
    resample_fraction: float,
    fk_error_gate_m: float,
    endpoint_gate_m: float,
) -> StreamStats:
    stats = StreamStats(final_q_deg=start_q_deg.copy(), final_tcp=start_tcp.copy())
    target_delta = to_wp.target_xyz - start_tcp
    distance = _norm(target_delta)
    desired_step_m = max_tcp_step_m * resample_fraction
    n_steps = max(1, int(math.ceil(distance / desired_step_m)))
    q_prev = start_q_deg.copy()
    tcp_prev = start_tcp.copy()

    print(
        f"[roarm_chain_timing] segment_start phase={phase} from={from_name} to={to_wp.name} "
        f"distance_m={distance:.6f} desired_step_m={desired_step_m:.6f} resample_steps={n_steps}"
    )

    for i in range(1, n_steps + 1):
        target_tcp = start_tcp + (target_delta * (i / n_steps))
        q_sol, converged, ik_err_mm, n_iter = ik_dls(target_tcp, q_prev, max_iter=200, tol_mm=1.0)
        q_sol = clip_joints(q_sol)
        if to_wp.force_pick_wrist_roll:
            q_sol[4] = PICK_WRIST_R_DEG
        q_sol[5] = GRIPPER_LATCH_DEG if phase == "ATTACHED_MOVE" else GRIPPER_OPEN_DEG
        fk_now = fk_tcp(q_sol)
        tcp_step = _norm(fk_now - tcp_prev)
        fk_error = _norm(fk_now - target_tcp)
        endpoint_error = _norm(fk_now - to_wp.target_xyz)
        ok = converged and tcp_step <= max_tcp_step_m and fk_error <= fk_error_gate_m

        stats.n_cmds += 1
        stats.max_tcp_step_m = max(stats.max_tcp_step_m, tcp_step)
        stats.max_fk_error_m = max(stats.max_fk_error_m, fk_error)
        stats.max_endpoint_error_m = max(stats.max_endpoint_error_m, endpoint_error)
        stats.max_ik_err_mm = max(stats.max_ik_err_mm, float(ik_err_mm))
        stats.ik_failures += 0 if converged else 1
        stats.ok &= ok

        print(
            f"[roarm_chain_timing] event={'MOVE' if phase == 'ATTACHED_MOVE' else 'PRE_MOVE'} "
            f"phase={phase} segment={from_name}->{to_wp.name} index={i}/{n_steps} "
            f"target_tcp={_fmt_xyz(target_tcp)} fk_tcp={_fmt_xyz(fk_now)} "
            f"ik_converged={'YES' if converged else 'NO'} ik_err_mm={ik_err_mm:.3f} ik_iter={n_iter} "
            f"tcp_step_m={tcp_step:.6f} fk_error_m={fk_error:.6f} "
            f"endpoint_error_m={endpoint_error:.6f} ok={'YES' if ok else 'NO'}"
        )

        q_prev = q_sol
        tcp_prev = fk_now

    final_endpoint_error = _norm(tcp_prev - to_wp.target_xyz)
    endpoint_ok = final_endpoint_error <= endpoint_gate_m
    stats.max_endpoint_error_m = max(stats.max_endpoint_error_m, final_endpoint_error)
    stats.ok &= endpoint_ok
    stats.final_q_deg = q_prev.copy()
    stats.final_tcp = tcp_prev.copy()
    print(
        f"[roarm_chain_timing] segment_stop phase={phase} to={to_wp.name} "
        f"final_tcp={_fmt_xyz(tcp_prev)} target={_fmt_xyz(to_wp.target_xyz)} "
        f"final_endpoint_error_m={final_endpoint_error:.6f} "
        f"endpoint_gate_m={endpoint_gate_m:.6f} ok={'YES' if endpoint_ok else 'NO'}"
    )
    return stats


def _merge_stats(parts: list[StreamStats]) -> StreamStats:
    out = StreamStats(ok=all(p.ok for p in parts), n_cmds=sum(p.n_cmds for p in parts))
    out.max_tcp_step_m = max((p.max_tcp_step_m for p in parts), default=0.0)
    out.max_fk_error_m = max((p.max_fk_error_m for p in parts), default=0.0)
    out.max_endpoint_error_m = max((p.max_endpoint_error_m for p in parts), default=0.0)
    out.max_ik_err_mm = max((p.max_ik_err_mm for p in parts), default=0.0)
    out.ik_failures = sum(p.ik_failures for p in parts)
    if parts:
        out.final_q_deg = parts[-1].final_q_deg
        out.final_tcp = parts[-1].final_tcp
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--place_xyz", nargs=3, type=float, default=list(L1_SP1))
    ap.add_argument("--fk_error_gate_m", type=float, default=0.003)
    ap.add_argument("--endpoint_gate_m", type=float, default=0.003)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    args = ap.parse_args()

    print("[roarm_chain_timing] RoArm chain-side TCP resampling/timing dry-run probe")
    print(
        "[roarm_chain_timing] "
        "chain_side_only=YES isaac_chain_integration=NO constraint_prim_insertion=NO "
        "surface_gripper=NO p7_training=NO env_default_edits=NO chain_defaults_edits=NO"
    )
    print(
        f"[roarm_chain_timing] gates fk_error_gate_m={args.fk_error_gate_m:.6f} "
        f"endpoint_gate_m={args.endpoint_gate_m:.6f} max_tcp_step_m={args.max_tcp_step_m:.6f} "
        f"resample_fraction={args.resample_fraction:.3f}"
    )

    planner = TrajectoryPlanner(
        sponge_xyz=(args.sponge_xy[0], args.sponge_xy[1], SPONGE_CENTER_Z),
        place_xyz=tuple(args.place_xyz),
    )
    wps = _waypoints(planner)
    wp_by_name = {wp.name: wp for wp in wps}

    raw_max_gap = 0.0
    raw_gap_ok = True
    for a, b in zip(wps[:-1], wps[1:]):
        gap = _norm(fk_tcp(b.q_deg) - fk_tcp(a.q_deg))
        raw_max_gap = max(raw_max_gap, gap)
        ok = gap <= args.max_tcp_step_m
        raw_gap_ok &= ok
        print(
            f"[roarm_chain_timing] raw_gap from={a.name} to={b.name} "
            f"tcp_step_m={gap:.6f} gate_m={args.max_tcp_step_m:.6f} ok={'YES' if ok else 'NO'}"
        )

    pre_segments: list[StreamStats] = []
    q = wp_by_name["home"].q_deg.copy()
    tcp = fk_tcp(q)
    prev_name = "home"
    for name in ["high", "hover", "1b1_z59", "1b2_z53", "grasp"]:
        stats = _resample_segment(
            phase="PRE_CLOSE",
            from_name=prev_name,
            to_wp=wp_by_name[name],
            start_tcp=tcp,
            start_q_deg=q,
            max_tcp_step_m=args.max_tcp_step_m,
            resample_fraction=args.resample_fraction,
            fk_error_gate_m=args.fk_error_gate_m,
            endpoint_gate_m=args.endpoint_gate_m,
        )
        pre_segments.append(stats)
        q = stats.final_q_deg if stats.final_q_deg is not None else q
        tcp = stats.final_tcp if stats.final_tcp is not None else tcp
        prev_name = name

    pre_stats = _merge_stats(pre_segments)
    close_target_reached = _norm(tcp - wp_by_name["grasp"].target_xyz) <= args.endpoint_gate_m
    close_ok = pre_stats.ok and close_target_reached
    print(
        f"[roarm_chain_timing] event=CLOSE accepted={'YES' if close_ok else 'NO'} "
        f"target_reached={'YES' if close_target_reached else 'NO'} "
        f"gripper_latch_deg={GRIPPER_LATCH_DEG:.2f} close_tcp={_fmt_xyz(tcp)}"
    )

    q[5] = GRIPPER_LATCH_DEG
    attached_stats = _resample_segment(
        phase="ATTACHED_MOVE",
        from_name="grasp",
        to_wp=wp_by_name["transport_hover"],
        start_tcp=tcp,
        start_q_deg=q,
        max_tcp_step_m=args.max_tcp_step_m,
        resample_fraction=args.resample_fraction,
        fk_error_gate_m=args.fk_error_gate_m,
        endpoint_gate_m=args.endpoint_gate_m,
    )
    transport_final_error = _norm(attached_stats.final_tcp - wp_by_name["transport_hover"].target_xyz)
    hold_ok = attached_stats.ok and transport_final_error <= args.endpoint_gate_m
    release_after_target_ok = hold_ok
    command_order_ok = close_ok and attached_stats.ok and hold_ok and release_after_target_ok
    no_move_after_release = True

    print(
        f"[roarm_chain_timing] event=HOLD accepted={'YES' if hold_ok else 'NO'} "
        f"target_reached={'YES' if hold_ok else 'NO'} final_transport_error_m={transport_final_error:.6f}"
    )
    print(
        f"[roarm_chain_timing] event=RELEASE accepted={'YES' if release_after_target_ok else 'NO'} "
        f"release_after_target_ok={'YES' if release_after_target_ok else 'NO'}"
    )

    success = (
        pre_stats.ok
        and close_ok
        and attached_stats.ok
        and hold_ok
        and command_order_ok
        and release_after_target_ok
        and no_move_after_release
    )
    print(
        f"[roarm_chain_timing] aggregate preclose_cmds={pre_stats.n_cmds} "
        f"attached_cmds={attached_stats.n_cmds} raw_max_gap_m={raw_max_gap:.6f} "
        f"raw_gap_ok={'YES' if raw_gap_ok else 'NO'} "
        f"max_preclose_tcp_step_m={pre_stats.max_tcp_step_m:.6f} "
        f"max_attached_tcp_step_m={attached_stats.max_tcp_step_m:.6f} "
        f"max_preclose_fk_error_m={pre_stats.max_fk_error_m:.6f} "
        f"max_attached_fk_error_m={attached_stats.max_fk_error_m:.6f} "
        f"transport_final_error_m={transport_final_error:.6f} "
        f"preclose_ik_failures={pre_stats.ik_failures} attached_ik_failures={attached_stats.ik_failures}"
    )
    print(
        f"[roarm_chain_timing] gates preclose_stream_ok={'YES' if pre_stats.ok else 'NO'} "
        f"close_ok={'YES' if close_ok else 'NO'} "
        f"attached_stream_ok={'YES' if attached_stats.ok else 'NO'} "
        f"hold_ok={'YES' if hold_ok else 'NO'} "
        f"command_order_ok={'YES' if command_order_ok else 'NO'} "
        f"release_after_target_ok={'YES' if release_after_target_ok else 'NO'} "
        f"no_move_after_release={'YES' if no_move_after_release else 'NO'}"
    )
    print(f"[roarm_chain_timing] ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS={'YES' if success else 'NO'}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
