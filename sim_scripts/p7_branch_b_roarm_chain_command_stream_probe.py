#!/usr/bin/env python3
"""RoArm chain-side command-stream abstraction probe for P7 Branch B.

This is a local/numpy-only pre-integration diagnostic. It converts the existing
TrajectoryPlanner waypoints into an explicit TCP command stream:

    PRE_MOVE* -> CLOSE -> MOVE* -> HOLD -> RELEASE

It validates only chain-side command timing and FK/IK realization. It does not
run Isaac, insert constraint prims, use SurfaceGripper, change env/train/chain
defaults, or integrate any fixed/dynamic constraint into the RoArm chain.
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

from roarm_kinematics import clip_joints, fk_tcp, ik_dls  # noqa: E402


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
    target_tcp: np.ndarray
    q_deg: np.ndarray
    force_pick_wrist_roll: bool


@dataclass(frozen=True)
class CommandEvent:
    index: int
    kind: str
    phase: str
    segment: str
    target_tcp: np.ndarray
    realized_tcp: np.ndarray
    q_deg: np.ndarray
    tcp_step_m: float
    fk_error_m: float
    endpoint_error_m: float
    ik_converged: bool
    ik_err_mm: float
    ik_iter: int
    accepted: bool


@dataclass
class StreamStats:
    ok: bool = True
    max_tcp_step_m: float = 0.0
    max_fk_error_m: float = 0.0
    max_endpoint_error_m: float = 0.0
    ik_failures: int = 0
    pre_move_count: int = 0
    move_count: int = 0
    final_q_deg: np.ndarray | None = None
    final_tcp: np.ndarray | None = None


class CommandState:
    def __init__(self) -> None:
        self.attached = False
        self.released = False
        self.target_ok = False
        self.command_order_ok = True
        self.release_after_target_ok = True
        self.no_move_after_release = True

    def accept_pre_move(self, target_ok: bool) -> bool:
        accepted = (not self.attached) and (not self.released)
        self.command_order_ok &= accepted
        self.target_ok = target_ok
        return accepted

    def accept_close(self, target_ok: bool) -> bool:
        accepted = (not self.attached) and (not self.released) and target_ok
        self.command_order_ok &= accepted
        if accepted:
            self.attached = True
            self.target_ok = target_ok
        return accepted

    def accept_move(self, target_ok: bool) -> bool:
        accepted = self.attached and (not self.released)
        self.command_order_ok &= accepted
        if self.released:
            self.no_move_after_release = False
        self.target_ok = target_ok
        return accepted

    def accept_hold(self, target_ok: bool) -> bool:
        accepted = self.attached and (not self.released) and target_ok
        self.command_order_ok &= accepted
        if accepted:
            self.target_ok = True
        return accepted

    def accept_release(self) -> bool:
        accepted = self.attached and (not self.released) and self.target_ok
        self.command_order_ok &= accepted
        self.release_after_target_ok &= self.target_ok
        if accepted:
            self.attached = False
            self.released = True
        return accepted


def _fmt_xyz(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _make_waypoints(planner: TrajectoryPlanner) -> list[Waypoint]:
    pick_xy = planner.pick_xy
    place = planner.place_xyz
    return [
        Waypoint("home", fk_tcp(HOME_DEG), HOME_DEG.copy(), False),
        Waypoint("high", np.array([pick_xy[0], pick_xy[1], HIGH_TCP_Z]), planner.q_high_deg.copy(), True),
        Waypoint(
            "hover",
            np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + planner.hover_offset_z]),
            planner.q_hover_deg.copy(),
            True,
        ),
        Waypoint("1b1_z59", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.012]), planner.q_1b1_deg.copy(), True),
        Waypoint("1b2_z53", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.006]), planner.q_1b2_deg.copy(), True),
        Waypoint("grasp", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z]), planner.q_grasp_deg.copy(), True),
        Waypoint("transport_hover", np.array([place[0], place[1], TCP_RELEASE_ENTRY_Z]), planner.q_transport_deg.copy(), True),
    ]


def _print_waypoints(waypoints: list[Waypoint]) -> None:
    for wp in waypoints:
        tcp = fk_tcp(wp.q_deg)
        print(
            f"[roarm_chain_cmd_stream] waypoint name={wp.name} target_tcp={_fmt_xyz(wp.target_tcp)} "
            f"fk_tcp={_fmt_xyz(tcp)} fk_error_m={_norm(tcp - wp.target_tcp):.6f} "
            f"q_deg={[round(float(x), 3) for x in wp.q_deg[:5]]}"
        )


def _audit_raw_gaps(waypoints: list[Waypoint], max_tcp_step_m: float) -> tuple[bool, float]:
    raw_ok = True
    max_gap = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        gap = _norm(fk_tcp(b.q_deg) - fk_tcp(a.q_deg))
        max_gap = max(max_gap, gap)
        ok = gap <= max_tcp_step_m
        raw_ok &= ok
        print(
            f"[roarm_chain_cmd_stream] raw_gap from={a.name} to={b.name} "
            f"tcp_step_m={gap:.6f} gate_m={max_tcp_step_m:.6f} ok={'YES' if ok else 'NO'}"
        )
    return raw_ok, max_gap


def _solve_segment(
    *,
    start_name: str,
    end_wp: Waypoint,
    phase: str,
    event_kind: str,
    start_q_deg: np.ndarray,
    start_tcp: np.ndarray,
    max_tcp_step_m: float,
    resample_fraction: float,
    fk_error_gate_m: float,
    endpoint_gate_m: float,
    next_index: int,
) -> tuple[list[CommandEvent], StreamStats]:
    target_delta = end_wp.target_tcp - start_tcp
    distance = _norm(target_delta)
    desired_step_m = max_tcp_step_m * resample_fraction
    n_steps = max(1, int(math.ceil(distance / desired_step_m)))
    q_prev = start_q_deg.copy()
    tcp_prev = start_tcp.copy()
    events: list[CommandEvent] = []
    stats = StreamStats(final_q_deg=q_prev.copy(), final_tcp=tcp_prev.copy())

    print(
        f"[roarm_chain_cmd_stream] segment_start phase={phase} from={start_name} to={end_wp.name} "
        f"distance_m={distance:.6f} desired_step_m={desired_step_m:.6f} resample_steps={n_steps}"
    )

    for i in range(1, n_steps + 1):
        target_tcp = start_tcp + target_delta * (i / n_steps)
        q_sol, converged, ik_err_mm, n_iter = ik_dls(target_tcp, q_prev, max_iter=200, tol_mm=1.0)
        q_sol = clip_joints(q_sol)
        if end_wp.force_pick_wrist_roll:
            q_sol[4] = PICK_WRIST_R_DEG
        q_sol[5] = GRIPPER_LATCH_DEG if event_kind == "MOVE" else GRIPPER_OPEN_DEG
        realized_tcp = fk_tcp(q_sol)
        tcp_step = _norm(realized_tcp - tcp_prev)
        fk_error = _norm(realized_tcp - target_tcp)
        endpoint_error = _norm(realized_tcp - end_wp.target_tcp)
        accepted = converged and tcp_step <= max_tcp_step_m and fk_error <= fk_error_gate_m

        event = CommandEvent(
            index=next_index + len(events),
            kind=event_kind,
            phase=phase,
            segment=f"{start_name}->{end_wp.name}",
            target_tcp=target_tcp,
            realized_tcp=realized_tcp,
            q_deg=q_sol.copy(),
            tcp_step_m=tcp_step,
            fk_error_m=fk_error,
            endpoint_error_m=endpoint_error,
            ik_converged=bool(converged),
            ik_err_mm=float(ik_err_mm),
            ik_iter=int(n_iter),
            accepted=bool(accepted),
        )
        events.append(event)

        stats.ok &= accepted
        stats.max_tcp_step_m = max(stats.max_tcp_step_m, tcp_step)
        stats.max_fk_error_m = max(stats.max_fk_error_m, fk_error)
        stats.max_endpoint_error_m = max(stats.max_endpoint_error_m, endpoint_error)
        stats.ik_failures += 0 if converged else 1
        if event_kind == "PRE_MOVE":
            stats.pre_move_count += 1
        else:
            stats.move_count += 1

        print(
            f"[roarm_chain_cmd_stream] event_index={event.index:03d} event={event.kind} phase={event.phase} "
            f"segment={event.segment} target_tcp={_fmt_xyz(event.target_tcp)} "
            f"realized_tcp={_fmt_xyz(event.realized_tcp)} ik_converged={'YES' if event.ik_converged else 'NO'} "
            f"ik_err_mm={event.ik_err_mm:.3f} ik_iter={event.ik_iter} "
            f"tcp_step_m={event.tcp_step_m:.6f} fk_error_m={event.fk_error_m:.6f} "
            f"endpoint_error_m={event.endpoint_error_m:.6f} accepted={'YES' if event.accepted else 'NO'}"
        )

        q_prev = q_sol
        tcp_prev = realized_tcp

    final_endpoint_error = _norm(tcp_prev - end_wp.target_tcp)
    endpoint_ok = final_endpoint_error <= endpoint_gate_m
    stats.ok &= endpoint_ok
    stats.max_endpoint_error_m = max(stats.max_endpoint_error_m, final_endpoint_error)
    stats.final_q_deg = q_prev.copy()
    stats.final_tcp = tcp_prev.copy()
    print(
        f"[roarm_chain_cmd_stream] segment_stop phase={phase} to={end_wp.name} "
        f"final_tcp={_fmt_xyz(tcp_prev)} target_tcp={_fmt_xyz(end_wp.target_tcp)} "
        f"final_endpoint_error_m={final_endpoint_error:.6f} endpoint_gate_m={endpoint_gate_m:.6f} "
        f"ok={'YES' if endpoint_ok else 'NO'}"
    )
    return events, stats


def _merge_stats(parts: list[StreamStats]) -> StreamStats:
    out = StreamStats(ok=all(p.ok for p in parts))
    out.max_tcp_step_m = max((p.max_tcp_step_m for p in parts), default=0.0)
    out.max_fk_error_m = max((p.max_fk_error_m for p in parts), default=0.0)
    out.max_endpoint_error_m = max((p.max_endpoint_error_m for p in parts), default=0.0)
    out.ik_failures = sum(p.ik_failures for p in parts)
    out.pre_move_count = sum(p.pre_move_count for p in parts)
    out.move_count = sum(p.move_count for p in parts)
    if parts:
        out.final_q_deg = parts[-1].final_q_deg
        out.final_tcp = parts[-1].final_tcp
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    parser.add_argument("--place_xyz", nargs=3, type=float, default=list(L1_SP1))
    parser.add_argument("--fk_error_gate_m", type=float, default=0.003)
    parser.add_argument("--endpoint_gate_m", type=float, default=0.003)
    parser.add_argument("--max_tcp_step_m", type=float, default=0.010)
    parser.add_argument("--resample_fraction", type=float, default=0.90)
    args = parser.parse_args()

    if args.resample_fraction <= 0.0 or args.resample_fraction > 1.0:
        raise ValueError("resample_fraction must be in (0, 1]")

    print("[roarm_chain_cmd_stream] RoArm chain-side command-stream abstraction probe")
    print(
        "[roarm_chain_cmd_stream] "
        "command_stream_only=YES chain_side_only=YES isaac_chain_integration=NO "
        "constraint_prim_insertion=NO surface_gripper=NO p7_training=NO "
        "env_default_edits=NO chain_defaults_edits=NO"
    )
    print(
        f"[roarm_chain_cmd_stream] gates fk_error_gate_m={args.fk_error_gate_m:.6f} "
        f"endpoint_gate_m={args.endpoint_gate_m:.6f} max_tcp_step_m={args.max_tcp_step_m:.6f} "
        f"resample_fraction={args.resample_fraction:.3f}"
    )
    print("[roarm_chain_cmd_stream] schema PRE_MOVE* CLOSE MOVE* HOLD RELEASE")

    planner = TrajectoryPlanner(
        sponge_xyz=(args.sponge_xy[0], args.sponge_xy[1], SPONGE_CENTER_Z),
        place_xyz=tuple(args.place_xyz),
    )
    waypoints = _make_waypoints(planner)
    wp = {item.name: item for item in waypoints}
    _print_waypoints(waypoints)
    raw_gap_ok, raw_max_gap = _audit_raw_gaps(waypoints, args.max_tcp_step_m)

    events: list[CommandEvent] = []
    pre_stats_parts: list[StreamStats] = []
    q = wp["home"].q_deg.copy()
    tcp = fk_tcp(q)
    prev_name = "home"
    next_index = 1
    for name in ["high", "hover", "1b1_z59", "1b2_z53", "grasp"]:
        segment_events, segment_stats = _solve_segment(
            start_name=prev_name,
            end_wp=wp[name],
            phase="PRE_CLOSE",
            event_kind="PRE_MOVE",
            start_q_deg=q,
            start_tcp=tcp,
            max_tcp_step_m=args.max_tcp_step_m,
            resample_fraction=args.resample_fraction,
            fk_error_gate_m=args.fk_error_gate_m,
            endpoint_gate_m=args.endpoint_gate_m,
            next_index=next_index,
        )
        events.extend(segment_events)
        pre_stats_parts.append(segment_stats)
        next_index += len(segment_events)
        q = segment_stats.final_q_deg if segment_stats.final_q_deg is not None else q
        tcp = segment_stats.final_tcp if segment_stats.final_tcp is not None else tcp
        prev_name = name

    pre_stats = _merge_stats(pre_stats_parts)
    state = CommandState()
    for event in events:
        target_ok = event.endpoint_error_m <= args.endpoint_gate_m if event.segment.endswith("->grasp") else False
        state.accept_pre_move(target_ok)

    close_target_error = _norm(tcp - wp["grasp"].target_tcp)
    close_target_ok = close_target_error <= args.endpoint_gate_m
    close_accepted = state.accept_close(close_target_ok) and pre_stats.ok
    close_index = next_index
    print(
        f"[roarm_chain_cmd_stream] event_index={close_index:03d} event=CLOSE accepted={'YES' if close_accepted else 'NO'} "
        f"target_reached={'YES' if close_target_ok else 'NO'} target_error_m={close_target_error:.6f} "
        f"gripper_latch_deg={GRIPPER_LATCH_DEG:.2f} close_tcp={_fmt_xyz(tcp)}"
    )
    next_index += 1

    q[5] = GRIPPER_LATCH_DEG
    move_events, attached_stats = _solve_segment(
        start_name="grasp",
        end_wp=wp["transport_hover"],
        phase="ATTACHED_MOVE",
        event_kind="MOVE",
        start_q_deg=q,
        start_tcp=tcp,
        max_tcp_step_m=args.max_tcp_step_m,
        resample_fraction=args.resample_fraction,
        fk_error_gate_m=args.fk_error_gate_m,
        endpoint_gate_m=args.endpoint_gate_m,
        next_index=next_index,
    )
    events.extend(move_events)
    next_index += len(move_events)
    for event in move_events:
        target_ok = event.endpoint_error_m <= args.endpoint_gate_m
        accepted_by_state = state.accept_move(target_ok)
        if not accepted_by_state:
            attached_stats.ok = False

    final_tcp = attached_stats.final_tcp if attached_stats.final_tcp is not None else tcp
    transport_final_error = _norm(final_tcp - wp["transport_hover"].target_tcp)
    transport_target_ok = attached_stats.ok and transport_final_error <= args.endpoint_gate_m
    hold_accepted = state.accept_hold(transport_target_ok)
    print(
        f"[roarm_chain_cmd_stream] event_index={next_index:03d} event=HOLD accepted={'YES' if hold_accepted else 'NO'} "
        f"target_reached={'YES' if transport_target_ok else 'NO'} final_transport_error_m={transport_final_error:.6f}"
    )
    next_index += 1

    release_accepted = state.accept_release()
    print(
        f"[roarm_chain_cmd_stream] event_index={next_index:03d} event=RELEASE accepted={'YES' if release_accepted else 'NO'} "
        f"release_after_target_ok={'YES' if state.release_after_target_ok and release_accepted else 'NO'}"
    )

    event_kinds = [event.kind for event in events]
    stream_shape_ok = (
        event_kinds[: pre_stats.pre_move_count] == ["PRE_MOVE"] * pre_stats.pre_move_count
        and event_kinds[pre_stats.pre_move_count :] == ["MOVE"] * attached_stats.move_count
    )
    command_order_ok = state.command_order_ok and close_accepted and hold_accepted and release_accepted and stream_shape_ok
    release_after_target_ok = state.release_after_target_ok and release_accepted and transport_target_ok
    no_move_after_release = state.no_move_after_release
    success = (
        pre_stats.ok
        and close_accepted
        and attached_stats.ok
        and hold_accepted
        and release_after_target_ok
        and no_move_after_release
        and command_order_ok
        and pre_stats.max_tcp_step_m <= args.max_tcp_step_m
        and attached_stats.max_tcp_step_m <= args.max_tcp_step_m
        and pre_stats.max_fk_error_m <= args.fk_error_gate_m
        and attached_stats.max_fk_error_m <= args.fk_error_gate_m
        and transport_final_error <= args.endpoint_gate_m
    )

    print(
        f"[roarm_chain_cmd_stream] aggregate events_total={pre_stats.pre_move_count + attached_stats.move_count + 3} "
        f"pre_move_cmds={pre_stats.pre_move_count} move_cmds={attached_stats.move_count} "
        f"raw_max_gap_m={raw_max_gap:.6f} raw_gap_ok={'YES' if raw_gap_ok else 'NO'} "
        f"max_pre_move_tcp_step_m={pre_stats.max_tcp_step_m:.6f} "
        f"max_move_tcp_step_m={attached_stats.max_tcp_step_m:.6f} "
        f"max_pre_move_fk_error_m={pre_stats.max_fk_error_m:.6f} "
        f"max_move_fk_error_m={attached_stats.max_fk_error_m:.6f} "
        f"transport_final_error_m={transport_final_error:.6f} "
        f"pre_move_ik_failures={pre_stats.ik_failures} move_ik_failures={attached_stats.ik_failures}"
    )
    print(
        f"[roarm_chain_cmd_stream] gates pre_move_stream_ok={'YES' if pre_stats.ok else 'NO'} "
        f"close_ok={'YES' if close_accepted else 'NO'} "
        f"move_stream_ok={'YES' if attached_stats.ok else 'NO'} "
        f"hold_ok={'YES' if hold_accepted else 'NO'} "
        f"command_order_ok={'YES' if command_order_ok else 'NO'} "
        f"release_after_target_ok={'YES' if release_after_target_ok else 'NO'} "
        f"no_move_after_release={'YES' if no_move_after_release else 'NO'} "
        f"stream_shape_ok={'YES' if stream_shape_ok else 'NO'}"
    )
    print(f"[roarm_chain_cmd_stream] ROARM_CHAIN_COMMAND_STREAM_SUCCESS={'YES' if success else 'NO'}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
