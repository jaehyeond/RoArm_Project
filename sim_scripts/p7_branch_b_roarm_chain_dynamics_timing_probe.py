#!/usr/bin/env python3
"""Isaac/RoArm articulation timing probe for P7 Branch B.

This is a narrow pre-integration diagnostic. It executes the conservative
RoArm TCP command stream on the real Isaac articulation/controller, but it does
not insert constraint prims, use SurfaceGripper, attach fixed/dynamic constraints
to the chain, change env/train/chain defaults, or run P7 training.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
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
    expected_tcp: np.ndarray
    q_deg: np.ndarray


@dataclass
class EventResult:
    event: CommandEvent
    reached: bool
    steps: int
    first_step_requested_error_m: float
    final_requested_error_m: float
    final_expected_error_m: float
    max_sim_tcp_step_m: float
    max_cache_fresh_delta_m: float
    max_joint_err_rad: float
    gripper_q_rad: float
    grasped: bool


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


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


def _resample_segment(
    *,
    start_name: str,
    end_wp: Waypoint,
    phase: str,
    kind: str,
    start_q_deg: np.ndarray,
    start_tcp: np.ndarray,
    max_tcp_step_m: float,
    resample_fraction: float,
    next_index: int,
) -> tuple[list[CommandEvent], np.ndarray, np.ndarray]:
    target_delta = end_wp.target_tcp - start_tcp
    desired_step_m = max_tcp_step_m * resample_fraction
    n_steps = max(1, int(math.ceil(_norm(target_delta) / desired_step_m)))
    q_prev = start_q_deg.copy()
    tcp_prev = start_tcp.copy()
    events: list[CommandEvent] = []
    for i in range(1, n_steps + 1):
        target_tcp = start_tcp + target_delta * (i / n_steps)
        q_sol, _converged, _ik_err_mm, _n_iter = ik_dls(target_tcp, q_prev, max_iter=200, tol_mm=1.0)
        q_sol = clip_joints(q_sol)
        if end_wp.force_pick_wrist_roll:
            q_sol[4] = PICK_WRIST_R_DEG
        q_sol[5] = GRIPPER_LATCH_DEG if kind == "MOVE" else GRIPPER_OPEN_DEG
        expected_tcp = fk_tcp(q_sol)
        events.append(
            CommandEvent(
                index=next_index + len(events),
                kind=kind,
                phase=phase,
                segment=f"{start_name}->{end_wp.name}",
                target_tcp=target_tcp.copy(),
                expected_tcp=expected_tcp.copy(),
                q_deg=q_sol.copy(),
            )
        )
        q_prev = q_sol
        tcp_prev = expected_tcp
    return events, q_prev, tcp_prev


def build_command_events(args: argparse.Namespace) -> tuple[list[CommandEvent], dict[str, float | int]]:
    with contextlib.redirect_stdout(io.StringIO()):
        planner = TrajectoryPlanner(
            sponge_xyz=(args.sponge_xy[0], args.sponge_xy[1], SPONGE_CENTER_Z),
            place_xyz=tuple(args.place_xyz),
        )
    waypoints = _make_waypoints(planner)
    wp = {item.name: item for item in waypoints}

    raw_max_gap = 0.0
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        raw_max_gap = max(raw_max_gap, _norm(fk_tcp(b.q_deg) - fk_tcp(a.q_deg)))

    events: list[CommandEvent] = []
    q = wp["home"].q_deg.copy()
    tcp = fk_tcp(q)
    next_index = 1
    prev_name = "home"
    for name in ["high", "hover", "1b1_z59", "1b2_z53", "grasp"]:
        segment, q, tcp = _resample_segment(
            start_name=prev_name,
            end_wp=wp[name],
            phase="PRE_CLOSE",
            kind="PRE_MOVE",
            start_q_deg=q,
            start_tcp=tcp,
            max_tcp_step_m=args.max_tcp_step_m,
            resample_fraction=args.resample_fraction,
            next_index=next_index,
        )
        events.extend(segment)
        next_index += len(segment)
        prev_name = name

    close_q = q.copy()
    close_q[5] = GRIPPER_LATCH_DEG
    close_event = CommandEvent(
        index=next_index,
        kind="CLOSE",
        phase="CLOSE",
        segment="grasp",
        target_tcp=wp["grasp"].target_tcp.copy(),
        expected_tcp=fk_tcp(close_q),
        q_deg=close_q.copy(),
    )
    events.append(close_event)
    next_index += 1

    q = close_q.copy()
    attached, q, tcp = _resample_segment(
        start_name="grasp",
        end_wp=wp["transport_hover"],
        phase="ATTACHED_MOVE_MARKER_ONLY",
        kind="MOVE",
        start_q_deg=q,
        start_tcp=tcp,
        max_tcp_step_m=args.max_tcp_step_m,
        resample_fraction=args.resample_fraction,
        next_index=next_index,
    )
    events.extend(attached)
    next_index += len(attached)

    hold_q = q.copy()
    events.append(
        CommandEvent(
            index=next_index,
            kind="HOLD",
            phase="HOLD",
            segment="transport_hover",
            target_tcp=wp["transport_hover"].target_tcp.copy(),
            expected_tcp=fk_tcp(hold_q),
            q_deg=hold_q.copy(),
        )
    )
    next_index += 1

    release_q = hold_q.copy()
    release_q[5] = GRIPPER_OPEN_DEG
    events.append(
        CommandEvent(
            index=next_index,
            kind="RELEASE",
            phase="RELEASE_MARKER_ONLY",
            segment="transport_hover",
            target_tcp=wp["transport_hover"].target_tcp.copy(),
            expected_tcp=fk_tcp(release_q),
            q_deg=release_q.copy(),
        )
    )

    meta = {
        "raw_max_gap_m": raw_max_gap,
        "pre_move_cmds": sum(1 for e in events if e.kind == "PRE_MOVE"),
        "move_cmds": sum(1 for e in events if e.kind == "MOVE"),
        "events_total": len(events),
    }
    return events, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--place_xyz", nargs=3, type=float, default=list(L1_SP1))
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--cache_delta_gate_m", type=float, default=0.002)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--hold_steps", type=int, default=5)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    args = ap.parse_args()

    if args.resample_fraction <= 0.0 or args.resample_fraction > 1.0:
        raise ValueError("resample_fraction must be in (0, 1]")

    events, meta = build_command_events(args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z
    from roarm_rl.roarm_stack_env import _quat_rotate

    print("[roarm_chain_dyn_timing] Isaac/RoArm articulation dynamics timing probe", flush=True)
    print(
        "[roarm_chain_dyn_timing] "
        "chain_side_articulation_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO p7_training=NO env_default_edits=NO "
        "chain_defaults_edits=NO release_marker_only=YES",
        flush=True,
    )
    print(
        f"[roarm_chain_dyn_timing] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} cache_delta_gate_m={args.cache_delta_gate_m:.6f} "
        f"home_fk_gate_m={args.home_fk_gate_m:.6f} resample_fraction={args.resample_fraction:.3f}",
        flush=True,
    )
    print(
        f"[roarm_chain_dyn_timing] stream events_total={meta['events_total']} "
        f"pre_move_cmds={meta['pre_move_cmds']} move_cmds={meta['move_cmds']} "
        f"raw_max_gap_m={meta['raw_max_gap_m']:.6f} raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.reward_phase = 6
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_attached_transport_release = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    cfg.target_pos = tuple(args.place_xyz)
    cfg.episode_length_s = args.episode_length_s

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    obs, _info = env.reset()

    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env._robot.set_joint_position_target(home_rad)
    base_env.robot_dof_targets[:] = home_rad

    # Keep the mandatory env sponge far away so CLOSE cannot trigger the existing
    # kinematic grasp path. This probe is about articulation/controller timing only.
    sponge_pose = torch.tensor(
        [[0.60, 0.40, ENV_SPONGE_CENTER_Z, 1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    sponge_pose[:, 0:3] += base_env.scene.env_origins[:1]
    base_env._sponge.write_root_pose_to_sim(sponge_pose)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    base_env._grasped[:] = False
    base_env._was_grasped[:] = False

    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)

    def step_once():
        out = env.step(null_action)
        if len(out) == 5:
            _obs, _rew, terminated, truncated, _extras = out
            done = bool((terminated | truncated).any().item())
        else:
            _obs, _rew, dones, _extras = out
            done = bool(dones.any().item())
        return done

    for _ in range(3):
        step_once()

    def fresh_tcp_local() -> np.ndarray:
        link5_pos = base_env._robot.data.body_pos_w[:1, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[:1, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(1, 3))
        tcp = link5_pos + tcp_offset_world
        return (tcp[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def cached_tcp_local() -> np.ndarray:
        return (base_env._tcp_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    home_fresh = fresh_tcp_local()
    home_expected = fk_tcp(HOME_DEG)
    home_fk_error = _norm(home_fresh - home_expected)
    print(
        f"[roarm_chain_dyn_timing] initial home_fresh_tcp={_fmt_xyz(home_fresh)} "
        f"home_expected_tcp={_fmt_xyz(home_expected)} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)}",
        flush=True,
    )

    results: list[EventResult] = []
    total_sim_steps = 0
    timeout = False
    nan_seen = False
    episode_done = False
    one_step_target_ok = True
    grasped_seen = False
    release_gripper_open_ok = False

    def execute_event(event: CommandEvent) -> EventResult:
        nonlocal total_sim_steps, timeout, nan_seen, episode_done, one_step_target_ok, grasped_seen

        target_q = torch.tensor(np.radians(event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        settle_count = 0
        steps_used = 0
        first_step_requested_error = float("inf")
        max_sim_tcp_step = 0.0
        max_cache_delta = 0.0
        max_joint_err = 0.0
        prev_tcp = fresh_tcp_local()

        max_steps = args.hold_steps if event.kind == "HOLD" else args.max_steps_per_event
        for step_idx in range(1, max_steps + 1):
            base_env.robot_dof_targets[:] = target_q
            done = step_once()
            total_sim_steps += 1
            steps_used = step_idx

            fresh = fresh_tcp_local()
            cached = cached_tcp_local()
            sim_step = _norm(fresh - prev_tcp)
            cache_delta = _norm(cached - fresh)
            requested_error = _norm(fresh - event.target_tcp)
            expected_error = _norm(fresh - event.expected_tcp)
            joint_err = float(
                torch.max(torch.abs(base_env._robot.data.joint_pos[0] - target_q[0])).detach().cpu().item()
            )
            gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
            grasped = bool(base_env._grasped[0].detach().cpu().item())

            if not np.isfinite(fresh).all() or not math.isfinite(requested_error):
                nan_seen = True
            if step_idx == 1:
                first_step_requested_error = requested_error
                one_step_target_ok &= requested_error <= args.target_error_gate_m
            max_sim_tcp_step = max(max_sim_tcp_step, sim_step)
            max_cache_delta = max(max_cache_delta, cache_delta)
            max_joint_err = max(max_joint_err, joint_err)
            grasped_seen |= grasped

            if event.kind == "CLOSE":
                event_reached = requested_error <= args.target_error_gate_m and gripper_q >= base_env.cfg.grasp_gripper_thresh
            elif event.kind == "RELEASE":
                event_reached = requested_error <= args.target_error_gate_m and gripper_q < base_env.cfg.grasp_gripper_thresh
            elif event.kind == "HOLD":
                event_reached = requested_error <= args.target_error_gate_m
            else:
                event_reached = requested_error <= args.target_error_gate_m

            settle_count = settle_count + 1 if event_reached else 0
            prev_tcp = fresh
            episode_done |= done
            if event_reached and settle_count >= args.settle_steps:
                break

        final_fresh = fresh_tcp_local()
        final_requested_error = _norm(final_fresh - event.target_tcp)
        final_expected_error = _norm(final_fresh - event.expected_tcp)
        final_gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
        final_grasped = bool(base_env._grasped[0].detach().cpu().item())
        if steps_used >= max_steps and final_requested_error > args.target_error_gate_m:
            timeout = True

        reached = final_requested_error <= args.target_error_gate_m
        if event.kind == "CLOSE":
            reached = reached and final_gripper_q >= base_env.cfg.grasp_gripper_thresh
        if event.kind == "RELEASE":
            reached = reached and final_gripper_q < base_env.cfg.grasp_gripper_thresh

        result = EventResult(
            event=event,
            reached=reached,
            steps=steps_used,
            first_step_requested_error_m=first_step_requested_error,
            final_requested_error_m=final_requested_error,
            final_expected_error_m=final_expected_error,
            max_sim_tcp_step_m=max_sim_tcp_step,
            max_cache_fresh_delta_m=max_cache_delta,
            max_joint_err_rad=max_joint_err,
            gripper_q_rad=final_gripper_q,
            grasped=final_grasped,
        )
        return result

    for event in events:
        result = execute_event(event)
        results.append(result)
        if event.index <= 5 or event.kind != "PRE_MOVE" or event.index % args.log_every_event == 0 or not result.reached:
            print(
                f"[roarm_chain_dyn_timing] event_index={event.index:03d} event={event.kind} "
                f"phase={event.phase} segment={event.segment} steps={result.steps} "
                f"first_step_target_error_m={result.first_step_requested_error_m:.6f} "
                f"final_target_error_m={result.final_requested_error_m:.6f} "
                f"final_expected_error_m={result.final_expected_error_m:.6f} "
                f"max_sim_tcp_step_m={result.max_sim_tcp_step_m:.6f} "
                f"max_cache_fresh_delta_m={result.max_cache_fresh_delta_m:.6f} "
                f"gripper_q_deg={math.degrees(result.gripper_q_rad):+.2f} "
                f"grasped={_yes(result.grasped)} reached={_yes(result.reached)}",
                flush=True,
            )

    release_results = [r for r in results if r.event.kind == "RELEASE"]
    if release_results:
        release_gripper_open_ok = release_results[-1].gripper_q_rad < base_env.cfg.grasp_gripper_thresh

    max_final_target_error = max((r.final_requested_error_m for r in results), default=float("inf"))
    max_final_expected_error = max((r.final_expected_error_m for r in results), default=float("inf"))
    max_first_step_error = max((r.first_step_requested_error_m for r in results), default=float("inf"))
    max_sim_tcp_step = max((r.max_sim_tcp_step_m for r in results), default=float("inf"))
    max_cache_delta = max((r.max_cache_fresh_delta_m for r in results), default=float("inf"))
    max_event_steps = max((r.steps for r in results), default=0)
    reached_all = all(r.reached for r in results)
    event_timeouts = sum(1 for r in results if not r.reached)

    controller_latency_ok = reached_all and not timeout and not episode_done
    target_error_ok = max_final_target_error <= args.target_error_gate_m
    sim_step_ok = max_sim_tcp_step <= args.max_tcp_step_m
    cache_timing_ok = max_cache_delta <= args.cache_delta_gate_m
    no_env_grasp_attach_ok = not grasped_seen
    release_marker_ok = release_gripper_open_ok and no_env_grasp_attach_ok
    success = (
        home_fk_error <= args.home_fk_gate_m
        and controller_latency_ok
        and target_error_ok
        and sim_step_ok
        and cache_timing_ok
        and no_env_grasp_attach_ok
        and release_marker_ok
        and not nan_seen
    )

    print(
        f"[roarm_chain_dyn_timing] aggregate events_total={len(results)} total_sim_steps={total_sim_steps} "
        f"max_event_steps={max_event_steps} event_timeouts={event_timeouts} "
        f"max_first_step_target_error_m={max_first_step_error:.6f} "
        f"one_step_target_ok={_yes(one_step_target_ok)} "
        f"max_final_target_error_m={max_final_target_error:.6f} "
        f"max_final_expected_error_m={max_final_expected_error:.6f} "
        f"max_sim_tcp_step_m={max_sim_tcp_step:.6f} "
        f"max_cache_fresh_delta_m={max_cache_delta:.6f} "
        f"grasped_seen={_yes(grasped_seen)} release_gripper_open_ok={_yes(release_gripper_open_ok)}",
        flush=True,
    )
    print(
        f"[roarm_chain_dyn_timing] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"controller_latency_ok={_yes(controller_latency_ok)} target_error_ok={_yes(target_error_ok)} "
        f"sim_step_ok={_yes(sim_step_ok)} cache_timing_ok={_yes(cache_timing_ok)} "
        f"one_step_target_ok={_yes(one_step_target_ok)} no_env_grasp_attach_ok={_yes(no_env_grasp_attach_ok)} "
        f"release_marker_ok={_yes(release_marker_ok)} nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_dyn_timing] ROARM_CHAIN_DYNAMICS_TIMING_SUCCESS={_yes(success)}", flush=True)

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
