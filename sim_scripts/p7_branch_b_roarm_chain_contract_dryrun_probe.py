#!/usr/bin/env python3
"""RoArm chain-side dry-run contract probe for P7 Branch B.

This is a local, numpy-only diagnostic. It inspects the existing
TrajectoryPlanner waypoints and kinematics, then emits a proposed
CLOSE/MOVE/HOLD/RELEASE command stream without Isaac Sim, physics constraints,
SurfaceGripper, or RoArm chain integration.
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import fk_tcp, ik_dls, clip_joints  # noqa: E402


def _load_chain_skills_module():
    spec = importlib.util.spec_from_file_location("chain_skills_local", REPO / "roarm_rl/chain_skills.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load roarm_rl/chain_skills.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chain_skills = _load_chain_skills_module()
GRIPPER_LATCH_DEG = _chain_skills.GRIPPER_LATCH_DEG
GRIPPER_OPEN_DEG = _chain_skills.GRIPPER_OPEN_DEG
HIGH_TCP_Z = _chain_skills.HIGH_TCP_Z
HOME_DEG = _chain_skills.HOME_DEG
L1_SP1 = _chain_skills.L1_SP1
PICK_WRIST_R_DEG = _chain_skills.PICK_WRIST_R_DEG
SPONGE_CENTER_Z = _chain_skills.SPONGE_CENTER_Z
TCP_PICK_GRASP_Z = _chain_skills.TCP_PICK_GRASP_Z
TCP_RELEASE_ENTRY_Z = _chain_skills.TCP_RELEASE_ENTRY_Z
TrajectoryPlanner = _chain_skills.TrajectoryPlanner


@dataclass(frozen=True)
class Waypoint:
    name: str
    target_xyz: np.ndarray
    q_deg: np.ndarray


def _fmt_xyz_m(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))


class ContractState:
    def __init__(self) -> None:
        self.attached = False
        self.released = False
        self.target_reached = False
        self.order_ok = True
        self.no_move_after_release = True
        self.release_after_target_ok = True

    def close(self, target_reached: bool) -> bool:
        accepted = (not self.attached) and (not self.released) and target_reached
        self.order_ok &= accepted
        if accepted:
            self.attached = True
            self.target_reached = target_reached
        return accepted

    def move(self) -> bool:
        accepted = self.attached and (not self.released)
        self.order_ok &= accepted
        if self.released:
            self.no_move_after_release = False
        self.target_reached = False
        return accepted

    def hold(self, target_reached: bool) -> bool:
        accepted = self.attached and (not self.released) and target_reached
        self.order_ok &= accepted
        if accepted:
            self.target_reached = True
        return accepted

    def release(self) -> bool:
        accepted = self.attached and (not self.released) and self.target_reached
        self.order_ok &= accepted
        self.release_after_target_ok &= self.target_reached
        if accepted:
            self.attached = False
            self.released = True
        return accepted


def _make_waypoints(planner: TrajectoryPlanner) -> list[Waypoint]:
    pick_xy = planner.pick_xy
    place_xyz = planner.place_xyz
    return [
        Waypoint("high", np.array([pick_xy[0], pick_xy[1], HIGH_TCP_Z]), planner.q_high_deg.copy()),
        Waypoint("hover", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + planner.hover_offset_z]), planner.q_hover_deg.copy()),
        Waypoint("1b1_z59", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.012]), planner.q_1b1_deg.copy()),
        Waypoint("1b2_z53", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z + 0.006]), planner.q_1b2_deg.copy()),
        Waypoint("grasp", np.array([pick_xy[0], pick_xy[1], TCP_PICK_GRASP_Z]), planner.q_grasp_deg.copy()),
        Waypoint("transport_hover", np.array([place_xyz[0], place_xyz[1], TCP_RELEASE_ENTRY_Z]), planner.q_transport_deg.copy()),
    ]


def _audit_waypoint_ik(waypoints: list[Waypoint], fk_error_limit_m: float) -> tuple[bool, float]:
    max_fk_error = 0.0
    all_ok = True
    print("[roarm_chain_contract_dryrun] waypoint_fk_audit")
    for wp in waypoints:
        tcp = fk_tcp(wp.q_deg)
        fk_error = _norm(tcp - wp.target_xyz)
        max_fk_error = max(max_fk_error, fk_error)
        ok = fk_error <= fk_error_limit_m
        all_ok &= ok
        print(
            f"[roarm_chain_contract_dryrun] waypoint name={wp.name} "
            f"target={_fmt_xyz_m(wp.target_xyz)} fk_tcp={_fmt_xyz_m(tcp)} "
            f"fk_error_m={fk_error:.6f} gate_m={fk_error_limit_m:.6f} ok={'YES' if ok else 'NO'} "
            f"q_deg={[round(float(x), 3) for x in wp.q_deg[:5]]}"
        )
    return all_ok, max_fk_error


def _raw_gap_audit(waypoints: list[Waypoint], max_tcp_step_m: float) -> tuple[bool, float]:
    max_gap = 0.0
    all_ok = True
    print("[roarm_chain_contract_dryrun] raw_planner_gap_audit")
    prev_tcp = fk_tcp(waypoints[0].q_deg)
    prev_name = waypoints[0].name
    for wp in waypoints[1:]:
        tcp = fk_tcp(wp.q_deg)
        gap = _norm(tcp - prev_tcp)
        max_gap = max(max_gap, gap)
        ok = gap <= max_tcp_step_m
        all_ok &= ok
        print(
            f"[roarm_chain_contract_dryrun] raw_gap from={prev_name} to={wp.name} "
            f"tcp_step_m={gap:.6f} gate_m={max_tcp_step_m:.6f} ok={'YES' if ok else 'NO'}"
        )
        prev_tcp = tcp
        prev_name = wp.name
    return all_ok, max_gap


def _solve_contract_stream(
    start_q_deg: np.ndarray,
    start_tcp: np.ndarray,
    final_tcp: np.ndarray,
    max_tcp_step_m: float,
    fk_error_limit_m: float,
) -> tuple[bool, float, float, int]:
    delta = final_tcp - start_tcp
    distance = _norm(delta)
    n_steps = max(1, int(math.ceil(distance / max_tcp_step_m)))
    q_prev = start_q_deg.copy()
    q_prev[5] = GRIPPER_LATCH_DEG
    max_step = 0.0
    max_fk_error = 0.0
    all_ok = True
    prev_tcp = start_tcp.copy()

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        target_tcp = start_tcp + alpha * delta
        q_sol, converged, ik_err_mm, n_iter = ik_dls(target_tcp, q_prev, max_iter=200, tol_mm=1.0)
        q_sol = clip_joints(q_sol)
        q_sol[4] = PICK_WRIST_R_DEG
        q_sol[5] = GRIPPER_LATCH_DEG
        fk_now = fk_tcp(q_sol)
        tcp_step = _norm(fk_now - prev_tcp)
        fk_error = _norm(fk_now - target_tcp)
        max_step = max(max_step, tcp_step)
        max_fk_error = max(max_fk_error, fk_error)
        ok = converged and tcp_step <= max_tcp_step_m and fk_error <= fk_error_limit_m
        all_ok &= ok
        print(
            f"[roarm_chain_contract_dryrun] event=MOVE index={i} accepted=YES "
            f"target_tcp={_fmt_xyz_m(target_tcp)} fk_tcp={_fmt_xyz_m(fk_now)} "
            f"ik_converged={'YES' if converged else 'NO'} ik_err_mm={ik_err_mm:.3f} ik_iter={n_iter} "
            f"tcp_step_m={tcp_step:.6f} fk_error_m={fk_error:.6f} ok={'YES' if ok else 'NO'}"
        )
        prev_tcp = fk_now
        q_prev = q_sol

    return all_ok, max_step, max_fk_error, n_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--place_xyz", nargs=3, type=float, default=list(L1_SP1))
    ap.add_argument("--fk_error_gate_m", type=float, default=0.003)
    ap.add_argument("--target_gate_m", type=float, default=0.003)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    args = ap.parse_args()

    print("[roarm_chain_contract_dryrun] RoArm chain-side contract dry-run probe")
    print(
        "[roarm_chain_contract_dryrun] "
        "chain_side_only=YES isaac_chain_integration=NO constraint_prim_insertion=NO "
        "surface_gripper=NO p7_training=NO env_default_edits=NO chain_defaults_edits=NO"
    )
    print(
        f"[roarm_chain_contract_dryrun] gates fk_error_gate_m={args.fk_error_gate_m:.6f} "
        f"target_gate_m={args.target_gate_m:.6f} max_tcp_step_m={args.max_tcp_step_m:.6f}"
    )

    sponge_xyz = (args.sponge_xy[0], args.sponge_xy[1], SPONGE_CENTER_Z)
    place_xyz = tuple(args.place_xyz)
    planner = TrajectoryPlanner(sponge_xyz=sponge_xyz, place_xyz=place_xyz)
    waypoints = _make_waypoints(planner)

    waypoint_ok, max_waypoint_fk_error = _audit_waypoint_ik(waypoints, args.fk_error_gate_m)
    raw_gap_ok, max_raw_gap = _raw_gap_audit(waypoints, args.max_tcp_step_m)

    state = ContractState()
    grasp_wp = next(wp for wp in waypoints if wp.name == "grasp")
    transport_wp = next(wp for wp in waypoints if wp.name == "transport_hover")
    grasp_tcp = fk_tcp(grasp_wp.q_deg)
    transport_tcp = fk_tcp(transport_wp.q_deg)
    close_target_reached = _norm(grasp_tcp - grasp_wp.target_xyz) <= args.target_gate_m
    close_ok = state.close(close_target_reached)
    print(
        f"[roarm_chain_contract_dryrun] event=CLOSE accepted={'YES' if close_ok else 'NO'} "
        f"target_reached={'YES' if close_target_reached else 'NO'} "
        f"gripper_open_deg={GRIPPER_OPEN_DEG:.2f} gripper_latch_deg={GRIPPER_LATCH_DEG:.2f} "
        f"tcp={_fmt_xyz_m(grasp_tcp)}"
    )

    move_accepted = state.move()
    stream_ok, max_stream_step, max_stream_fk_error, stream_steps = _solve_contract_stream(
        start_q_deg=grasp_wp.q_deg,
        start_tcp=grasp_tcp,
        final_tcp=transport_tcp,
        max_tcp_step_m=args.max_tcp_step_m,
        fk_error_limit_m=args.fk_error_gate_m,
    )
    final_target_error = _norm(transport_tcp - transport_wp.target_xyz)
    hold_target_reached = final_target_error <= args.target_gate_m
    hold_ok = state.hold(hold_target_reached)
    release_ok = state.release()

    print(
        f"[roarm_chain_contract_dryrun] event=HOLD accepted={'YES' if hold_ok else 'NO'} "
        f"target_reached={'YES' if hold_target_reached else 'NO'} "
        f"final_target_error_m={final_target_error:.6f}"
    )
    print(
        f"[roarm_chain_contract_dryrun] event=RELEASE accepted={'YES' if release_ok else 'NO'} "
        f"release_after_target_ok={'YES' if state.release_after_target_ok else 'NO'}"
    )

    command_order_ok = state.order_ok and move_accepted and close_ok and hold_ok and release_ok
    no_move_after_release = state.no_move_after_release
    release_after_target_ok = state.release_after_target_ok and release_ok
    contract_stream_ok = (
        waypoint_ok
        and stream_ok
        and command_order_ok
        and release_after_target_ok
        and no_move_after_release
        and max_stream_step <= args.max_tcp_step_m
        and max_stream_fk_error <= args.fk_error_gate_m
        and final_target_error <= args.target_gate_m
    )

    print(
        f"[roarm_chain_contract_dryrun] aggregate waypoints={len(waypoints)} "
        f"contract_move_steps={stream_steps} max_waypoint_fk_error_m={max_waypoint_fk_error:.6f} "
        f"max_raw_planner_gap_m={max_raw_gap:.6f} raw_planner_gap_ok={'YES' if raw_gap_ok else 'NO'} "
        f"max_contract_tcp_step_m={max_stream_step:.6f} max_contract_fk_error_m={max_stream_fk_error:.6f} "
        f"final_transport_target_error_m={final_target_error:.6f}"
    )
    print(
        f"[roarm_chain_contract_dryrun] gates waypoint_fk_ok={'YES' if waypoint_ok else 'NO'} "
        f"raw_planner_gap_ok={'YES' if raw_gap_ok else 'NO'} "
        f"contract_stream_step_ok={'YES' if max_stream_step <= args.max_tcp_step_m else 'NO'} "
        f"contract_stream_fk_ok={'YES' if max_stream_fk_error <= args.fk_error_gate_m else 'NO'} "
        f"command_order_ok={'YES' if command_order_ok else 'NO'} "
        f"release_after_target_ok={'YES' if release_after_target_ok else 'NO'} "
        f"no_move_after_release={'YES' if no_move_after_release else 'NO'}"
    )
    print(
        "[roarm_chain_contract_dryrun] "
        f"ROARM_CHAIN_CONTRACT_DRYRUN_SUCCESS={'YES' if contract_stream_ok else 'NO'}"
    )
    return 0 if contract_stream_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
