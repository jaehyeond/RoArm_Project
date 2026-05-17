#!/usr/bin/env python3
"""Passive-contact / close-timing probe for P7 Branch B.

This is a narrow pre-integration diagnostic. It executes the conservative
RoArm PRE_MOVE stream on the real Isaac articulation with the sponge at the
nominal pick location, then observes CLOSE timing. It does not insert
constraint prims, use SurfaceGripper, execute attached transport, or claim
attach/release physics.
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

from p7_branch_b_roarm_chain_dynamics_timing_probe import (  # noqa: E402
    GRIPPER_LATCH_DEG,
    GRIPPER_OPEN_DEG,
    HOME_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)


@dataclass
class ContactEventResult:
    index: int
    kind: str
    phase: str
    segment: str
    reached: bool
    steps: int
    final_target_error_m: float
    max_sim_tcp_step_m: float
    gripper_q_deg: float
    d_tcp_sponge_m: float
    sponge_xy_drift_m: float
    sponge_z_delta_m: float
    max_sponge_speed_mps: float
    min_upright_z: float
    latch_seen: bool
    latch_step: int


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _fmt_xyz(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--preclose_drift_gate_m", type=float, default=0.005)
    ap.add_argument("--close_drift_gate_m", type=float, default=0.020)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.90)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    args = ap.parse_args()

    if args.resample_fraction <= 0.0 or args.resample_fraction > 1.0:
        raise ValueError("resample_fraction must be in (0, 1]")

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    close_index = next(event.index for event in events if event.kind == "CLOSE")
    contact_events = [event for event in events if event.kind in ("PRE_MOVE", "CLOSE")]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z
    from roarm_rl.roarm_stack_env import _quat_rotate

    print("[roarm_chain_contact_close] RoArm passive-contact / close-timing probe", flush=True)
    print(
        "[roarm_chain_contact_close] "
        "passive_contact_close_timing_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO attached_transport=NO release_marker=NO "
        "p7_training=NO env_default_edits=NO chain_defaults_edits=NO "
        "kinematic_env_latch_is_marker_only=YES",
        flush=True,
    )
    print(
        f"[roarm_chain_contact_close] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"preclose_drift_gate_m={args.preclose_drift_gate_m:.6f} "
        f"close_drift_gate_m={args.close_drift_gate_m:.6f} "
        f"min_upright_z_gate={args.min_upright_z_gate:.3f} "
        f"resample_fraction={args.resample_fraction:.3f}",
        flush=True,
    )
    print(
        f"[roarm_chain_contact_close] stream source_events_total={meta['events_total']} "
        f"executed_events={len(contact_events)} pre_move_cmds={meta['pre_move_cmds']} "
        f"close_index={close_index} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
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
    cfg.episode_length_s = args.episode_length_s

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    env.reset()

    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env._robot.set_joint_position_target(home_rad)
    base_env.robot_dof_targets[:] = home_rad

    sponge_pose = torch.tensor(
        [[args.sponge_xy[0], args.sponge_xy[1], ENV_SPONGE_CENTER_Z, 1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    sponge_pose[:, 0:3] += base_env.scene.env_origins[:1]
    base_env._sponge.write_root_pose_to_sim(sponge_pose)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    base_env._grasped[:] = False
    base_env._was_grasped[:] = False

    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)

    def step_once() -> bool:
        out = env.step(null_action)
        if len(out) == 5:
            _obs, _rew, terminated, truncated, _extras = out
            return bool((terminated | truncated).any().item())
        _obs, _rew, dones, _extras = out
        return bool(dones.any().item())

    def fresh_tcp_local() -> np.ndarray:
        link5_pos = base_env._robot.data.body_pos_w[:1, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[:1, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(1, 3))
        tcp = link5_pos + tcp_offset_world
        return (tcp[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def sponge_pos_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def sponge_lin_vel() -> np.ndarray:
        return base_env._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_upright_z() -> float:
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        world_z = _quat_rotate(base_env._sponge.data.root_quat_w[:1], z_axis)
        return float(world_z[0, 2].detach().cpu().item())

    for _ in range(args.initial_settle_steps):
        step_once()
    settled_sponge = sponge_pos_local()
    settled_upright = sponge_upright_z()
    home_fresh = fresh_tcp_local()
    home_expected = fk_tcp(HOME_DEG)
    home_fk_error = _norm(home_fresh - home_expected)
    print(
        f"[roarm_chain_contact_close] initial home_fresh_tcp={_fmt_xyz(home_fresh)} "
        f"home_expected_tcp={_fmt_xyz(home_expected)} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"settled_sponge_pos={_fmt_xyz(settled_sponge)} settled_upright_z={settled_upright:.6f}",
        flush=True,
    )

    total_sim_steps = 0
    timeout = False
    nan_seen = False
    episode_done = False
    latch_seen = False
    latch_event_index = -1
    latch_step = -1
    gripper_threshold_step = -1
    preclose_latch_seen = False
    results: list[ContactEventResult] = []

    def execute_event(event) -> ContactEventResult:
        nonlocal total_sim_steps, timeout, nan_seen, episode_done
        nonlocal latch_seen, latch_event_index, latch_step, gripper_threshold_step, preclose_latch_seen

        target_q = torch.tensor(np.radians(event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        settle_count = 0
        steps_used = 0
        max_sim_tcp_step = 0.0
        max_sponge_speed = 0.0
        min_upright = sponge_upright_z()
        local_latch_step = -1
        prev_tcp = fresh_tcp_local()

        for step_idx in range(1, args.max_steps_per_event + 1):
            base_env.robot_dof_targets[:] = target_q
            done = step_once()
            total_sim_steps += 1
            steps_used = step_idx

            fresh = fresh_tcp_local()
            sponge = sponge_pos_local()
            speed = _norm(sponge_lin_vel())
            upright = sponge_upright_z()
            target_error = _norm(fresh - event.target_tcp)
            gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
            grasped = bool(base_env._grasped[0].detach().cpu().item())

            if not np.isfinite(fresh).all() or not np.isfinite(sponge).all() or not math.isfinite(target_error):
                nan_seen = True
            max_sim_tcp_step = max(max_sim_tcp_step, _norm(fresh - prev_tcp))
            max_sponge_speed = max(max_sponge_speed, speed)
            min_upright = min(min_upright, upright)
            episode_done |= done

            if gripper_threshold_step < 0 and gripper_q >= base_env.cfg.grasp_gripper_thresh:
                gripper_threshold_step = total_sim_steps
            if grasped and not latch_seen:
                latch_seen = True
                latch_event_index = event.index
                latch_step = total_sim_steps
                local_latch_step = step_idx
                preclose_latch_seen = event.kind == "PRE_MOVE"
                break

            if event.kind == "CLOSE":
                reached = target_error <= args.target_error_gate_m and gripper_q >= base_env.cfg.grasp_gripper_thresh
            else:
                reached = target_error <= args.target_error_gate_m
            settle_count = settle_count + 1 if reached else 0
            prev_tcp = fresh
            if reached and settle_count >= args.settle_steps:
                break

        final_tcp = fresh_tcp_local()
        final_sponge = sponge_pos_local()
        final_target_error = _norm(final_tcp - event.target_tcp)
        final_gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
        final_d_tcp_sponge = _norm(final_sponge - final_tcp)
        drift_xy = _norm(final_sponge[:2] - settled_sponge[:2])
        z_delta = float(final_sponge[2] - settled_sponge[2])
        reached = final_target_error <= args.target_error_gate_m
        if event.kind == "CLOSE":
            reached = reached and final_gripper_q >= base_env.cfg.grasp_gripper_thresh
        if steps_used >= args.max_steps_per_event and not reached:
            timeout = True

        return ContactEventResult(
            index=event.index,
            kind=event.kind,
            phase=event.phase,
            segment=event.segment,
            reached=reached,
            steps=steps_used,
            final_target_error_m=final_target_error,
            max_sim_tcp_step_m=max_sim_tcp_step,
            gripper_q_deg=math.degrees(final_gripper_q),
            d_tcp_sponge_m=final_d_tcp_sponge,
            sponge_xy_drift_m=drift_xy,
            sponge_z_delta_m=z_delta,
            max_sponge_speed_mps=max_sponge_speed,
            min_upright_z=min_upright,
            latch_seen=latch_seen,
            latch_step=local_latch_step,
        )

    for event in contact_events:
        result = execute_event(event)
        results.append(result)
        should_log = (
            event.index <= 5
            or event.kind == "CLOSE"
            or event.index % args.log_every_event == 0
            or not result.reached
            or result.sponge_xy_drift_m > 0.001
            or result.latch_step > 0
        )
        if should_log:
            print(
                f"[roarm_chain_contact_close] event_index={result.index:03d} event={result.kind} "
                f"phase={result.phase} segment={result.segment} steps={result.steps} "
                f"final_target_error_m={result.final_target_error_m:.6f} "
                f"max_sim_tcp_step_m={result.max_sim_tcp_step_m:.6f} "
                f"gripper_q_deg={result.gripper_q_deg:+.2f} "
                f"d_tcp_sponge_m={result.d_tcp_sponge_m:.6f} "
                f"sponge_xy_drift_m={result.sponge_xy_drift_m:.6f} "
                f"sponge_z_delta_m={result.sponge_z_delta_m:+.6f} "
                f"max_sponge_speed_mps={result.max_sponge_speed_mps:.6f} "
                f"min_upright_z={result.min_upright_z:.6f} "
                f"latch_seen={_yes(result.latch_seen)} latch_step={result.latch_step} "
                f"reached={_yes(result.reached)}",
                flush=True,
            )
        if latch_seen:
            break

    max_final_target_error = max((r.final_target_error_m for r in results), default=float("inf"))
    max_sim_tcp_step = max((r.max_sim_tcp_step_m for r in results), default=float("inf"))
    max_preclose_xy_drift = max((r.sponge_xy_drift_m for r in results if r.kind == "PRE_MOVE"), default=0.0)
    max_close_xy_drift = max((r.sponge_xy_drift_m for r in results), default=0.0)
    max_sponge_speed = max((r.max_sponge_speed_mps for r in results), default=0.0)
    min_upright = min((r.min_upright_z for r in results), default=1.0)
    reached_all = all(r.reached for r in results)
    latch_after_threshold_ok = (not latch_seen) or (gripper_threshold_step >= 0 and latch_step >= gripper_threshold_step)

    controller_latency_ok = reached_all and not timeout and not episode_done
    target_error_ok = max_final_target_error <= args.target_error_gate_m
    sim_step_ok = max_sim_tcp_step <= args.max_tcp_step_m
    preclose_passive_ok = max_preclose_xy_drift <= args.preclose_drift_gate_m and not preclose_latch_seen
    close_motion_ok = max_close_xy_drift <= args.close_drift_gate_m and min_upright >= args.min_upright_z_gate
    close_timing_ok = latch_after_threshold_ok
    success = (
        home_fk_error <= args.home_fk_gate_m
        and controller_latency_ok
        and target_error_ok
        and sim_step_ok
        and preclose_passive_ok
        and close_motion_ok
        and close_timing_ok
        and not nan_seen
    )

    print(
        f"[roarm_chain_contact_close] aggregate executed_events={len(results)} total_sim_steps={total_sim_steps} "
        f"max_final_target_error_m={max_final_target_error:.6f} "
        f"max_sim_tcp_step_m={max_sim_tcp_step:.6f} "
        f"max_preclose_sponge_xy_drift_m={max_preclose_xy_drift:.6f} "
        f"max_close_sponge_xy_drift_m={max_close_xy_drift:.6f} "
        f"max_sponge_speed_mps={max_sponge_speed:.6f} "
        f"min_upright_z={min_upright:.6f} "
        f"latch_seen={_yes(latch_seen)} latch_event_index={latch_event_index} "
        f"latch_global_step={latch_step} gripper_threshold_global_step={gripper_threshold_step} "
        f"preclose_latch_seen={_yes(preclose_latch_seen)} "
        f"kinematic_env_latch_is_marker_only=YES",
        flush=True,
    )
    print(
        f"[roarm_chain_contact_close] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"controller_latency_ok={_yes(controller_latency_ok)} "
        f"target_error_ok={_yes(target_error_ok)} sim_step_ok={_yes(sim_step_ok)} "
        f"preclose_passive_ok={_yes(preclose_passive_ok)} "
        f"close_motion_ok={_yes(close_motion_ok)} "
        f"close_timing_ok={_yes(close_timing_ok)} "
        f"attach_physics_validated=NO release_physics_validated=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_contact_close] ROARM_PASSIVE_CONTACT_CLOSE_TIMING_SUCCESS={_yes(success)}", flush=True)

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
