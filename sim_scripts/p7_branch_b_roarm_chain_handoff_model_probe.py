#!/usr/bin/env python3
"""Post-close handoff-model probe for P7 Branch B.

This is a narrow pre-integration diagnostic. It executes the conservative
RoArm PRE_MOVE stream and CLOSE on the real Isaac articulation with the sponge
at the nominal pick location, then holds the same grasp pose for a short
stationary post-close window. It compares local CLOSE handoff alternatives
around the existing env `_grasped` marker only. It does not insert constraint
prims, use SurfaceGripper, execute attached transport, run release, or claim
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
    HOME_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)


@dataclass
class SimState:
    global_step: int
    tcp: np.ndarray
    sponge_pos: np.ndarray
    sponge_quat: np.ndarray
    sponge_lin_vel: np.ndarray
    sponge_ang_vel: np.ndarray
    upright_z: float
    gripper_q_rad: float
    grasped: bool

    @property
    def d_tcp_sponge_m(self) -> float:
        return _norm(self.sponge_pos - self.tcp)

    @property
    def sponge_speed_mps(self) -> float:
        return _norm(self.sponge_lin_vel)

    @property
    def sponge_ang_speed_rps(self) -> float:
        return _norm(self.sponge_ang_vel)


@dataclass
class EventResult:
    index: int
    kind: str
    phase: str
    segment: str
    reached: bool
    steps: int
    final_target_error_m: float
    max_sim_tcp_step_m: float
    sponge_xy_drift_m: float
    sponge_z_delta_m: float
    max_sponge_speed_mps: float
    min_upright_z: float
    latch_seen: bool
    latch_step_local: int


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _fmt_xyz(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa = qa / max(_norm(qa), 1.0e-12)
    qb = qb / max(_norm(qb), 1.0e-12)
    dot = float(abs(np.dot(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--preclose_drift_gate_m", type=float, default=0.005)
    ap.add_argument("--close_drift_gate_m", type=float, default=0.020)
    ap.add_argument("--post_latch_pose_jump_gate_m", type=float, default=0.005)
    ap.add_argument("--post_latch_hold_drift_gate_m", type=float, default=0.005)
    ap.add_argument("--post_latch_d_tcp_jump_gate_m", type=float, default=0.010)
    ap.add_argument("--post_latch_speed_gate_mps", type=float, default=0.050)
    ap.add_argument("--post_latch_quat_angle_gate_deg", type=float, default=5.0)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.90)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--post_latch_hold_steps", type=int, default=20)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    ap.add_argument(
        "--handoff_model",
        choices=[
            "posewrite_tcp",
            "marker_only",
            "delayed_posewrite",
            "oneshot_align",
            "offset_preserve_posewrite",
        ],
        default="posewrite_tcp",
        help=(
            "Diagnostic-local handoff model after CLOSE. posewrite_tcp is the current env baseline; "
            "marker_only disables pose-write; delayed_posewrite waits N post-latch env steps; "
            "oneshot_align writes once then disables; offset_preserve_posewrite keeps the latch TCP->sponge offset."
        ),
    )
    ap.add_argument("--delay_posewrite_steps", type=int, default=3)
    ap.add_argument("--attach_quat_mode", choices=["preserve", "identity"], default="preserve")
    ap.add_argument("--attach_velocity_mode", choices=["zero", "keep"], default="zero")
    ap.add_argument(
        "--disable_attach_posewrite",
        action="store_true",
        help="Compatibility alias for --handoff_model marker_only.",
    )
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
    close_event = next(event for event in events if event.kind == "CLOSE")
    close_index = close_event.index
    contact_events = [event for event in events if event.kind in ("PRE_MOVE", "CLOSE")]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z
    from roarm_rl.roarm_stack_env import _quat_rotate

    if args.disable_attach_posewrite:
        args.handoff_model = "marker_only"
    if args.delay_posewrite_steps < 0:
        raise ValueError("delay_posewrite_steps must be >= 0")

    print("[roarm_chain_handoff] RoArm post-close handoff-model probe", flush=True)
    print(
        "[roarm_chain_handoff] "
        "post_close_handoff_model_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO attached_transport=NO release_marker=NO "
        "p7_training=NO env_default_edits=NO chain_defaults_edits=NO "
        "kinematic_env_latch_only=YES attach_physics_validated=NO release_physics_validated=NO "
        "claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_handoff] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"preclose_drift_gate_m={args.preclose_drift_gate_m:.6f} "
        f"close_drift_gate_m={args.close_drift_gate_m:.6f} "
        f"post_latch_pose_jump_gate_m={args.post_latch_pose_jump_gate_m:.6f} "
        f"post_latch_hold_drift_gate_m={args.post_latch_hold_drift_gate_m:.6f} "
        f"post_latch_d_tcp_jump_gate_m={args.post_latch_d_tcp_jump_gate_m:.6f} "
        f"post_latch_speed_gate_mps={args.post_latch_speed_gate_mps:.6f} "
        f"post_latch_quat_angle_gate_deg={args.post_latch_quat_angle_gate_deg:.2f} "
        f"min_upright_z_gate={args.min_upright_z_gate:.3f} "
        f"post_latch_hold_steps={args.post_latch_hold_steps} "
        f"resample_fraction={args.resample_fraction:.3f} "
        f"handoff_model={args.handoff_model} delay_posewrite_steps={args.delay_posewrite_steps} "
        f"attach_quat_mode={args.attach_quat_mode} "
        f"attach_velocity_mode={args.attach_velocity_mode} "
        f"continuous_posewrite_enabled={_yes(args.handoff_model in ('posewrite_tcp', 'offset_preserve_posewrite'))}",
        flush=True,
    )
    print(
        f"[roarm_chain_handoff] stream source_events_total={meta['events_total']} "
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
    cfg.attach_quat_mode = args.attach_quat_mode
    cfg.attach_velocity_mode = args.attach_velocity_mode

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    original_update_grasp_attach = base_env._update_grasp_attach
    handoff_stats = {
        "attach_calls": 0,
        "posewrite_calls": 0,
        "offset_initialized": 0,
        "delay_env_steps_seen": 0,
    }
    delay_step_tracker = {"last_total_sim_steps": None}
    preserved_tcp_to_sponge: torch.Tensor | None = None

    def _posewrite_status() -> str:
        if args.handoff_model == "marker_only":
            return "OFF"
        if args.handoff_model == "delayed_posewrite":
            return "DELAYED"
        if args.handoff_model == "oneshot_align":
            return "ONESHOT"
        if args.handoff_model == "offset_preserve_posewrite":
            return "OFFSET_PRESERVE"
        return "TCP_SNAP"

    def _current_tcp_world(env_ids: torch.Tensor) -> torch.Tensor:
        link5_pos = base_env._robot.data.body_pos_w[env_ids, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[env_ids, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(link5_pos.shape[0], 3))
        return link5_pos + tcp_offset_world

    def _write_sponge_pose(env_ids: torch.Tensor, pos_w: torch.Tensor) -> None:
        pose7 = torch.zeros((len(env_ids), 7), device=device)
        pose7[:, 0:3] = pos_w
        if base_env.cfg.attach_quat_mode == "preserve":
            pose7[:, 3:7] = base_env._sponge.data.root_quat_w[env_ids]
        elif base_env.cfg.attach_quat_mode == "identity":
            pose7[:, 3] = 1.0
        else:
            raise RuntimeError(f"Unexpected attach_quat_mode={base_env.cfg.attach_quat_mode!r}")
        base_env._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        if base_env.cfg.attach_velocity_mode == "zero":
            base_env._sponge.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids=env_ids)

    def _diagnostic_handoff_attach() -> None:
        nonlocal preserved_tcp_to_sponge
        env_ids = torch.where(base_env._grasped)[0]
        if len(env_ids) == 0:
            return
        handoff_stats["attach_calls"] += 1

        if args.handoff_model == "marker_only":
            return

        if args.handoff_model == "delayed_posewrite":
            # Isaac may call _apply_action more than once inside one env.step.
            # Count the stationary hold boundary in env-step units, not raw
            # internal attach-call units.
            if delay_step_tracker["last_total_sim_steps"] != total_sim_steps:
                delay_step_tracker["last_total_sim_steps"] = total_sim_steps
                handoff_stats["delay_env_steps_seen"] += 1
            if handoff_stats["delay_env_steps_seen"] <= args.delay_posewrite_steps:
                return

        if args.handoff_model == "oneshot_align" and handoff_stats["posewrite_calls"] > 0:
            return

        if args.handoff_model == "offset_preserve_posewrite":
            tcp_pos = _current_tcp_world(env_ids)
            if preserved_tcp_to_sponge is None:
                preserved_tcp_to_sponge = base_env._sponge.data.root_pos_w[env_ids].clone() - tcp_pos
                handoff_stats["offset_initialized"] = 1
            _write_sponge_pose(env_ids, tcp_pos + preserved_tcp_to_sponge)
            handoff_stats["posewrite_calls"] += 1
            return

        original_update_grasp_attach()
        handoff_stats["posewrite_calls"] += 1

    base_env._update_grasp_attach = _diagnostic_handoff_attach
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

    def sponge_quat() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_vel6() -> tuple[np.ndarray, np.ndarray]:
        root_vel = base_env._sponge.data.root_vel_w[0].detach().cpu().numpy().astype(np.float64)
        return root_vel[0:3], root_vel[3:6]

    def sponge_upright_z() -> float:
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        world_z = _quat_rotate(base_env._sponge.data.root_quat_w[:1], z_axis)
        return float(world_z[0, 2].detach().cpu().item())

    total_sim_steps = 0

    def capture_state() -> SimState:
        lin, ang = sponge_vel6()
        return SimState(
            global_step=total_sim_steps,
            tcp=fresh_tcp_local(),
            sponge_pos=sponge_pos_local(),
            sponge_quat=sponge_quat(),
            sponge_lin_vel=lin,
            sponge_ang_vel=ang,
            upright_z=sponge_upright_z(),
            gripper_q_rad=float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item()),
            grasped=bool(base_env._grasped[0].detach().cpu().item()),
        )

    for _ in range(args.initial_settle_steps):
        step_once()
    settled_state = capture_state()
    home_expected = fk_tcp(HOME_DEG)
    home_fk_error = _norm(settled_state.tcp - home_expected)
    print(
        f"[roarm_chain_handoff] initial home_fresh_tcp={_fmt_xyz(settled_state.tcp)} "
        f"home_expected_tcp={_fmt_xyz(home_expected)} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"settled_sponge_pos={_fmt_xyz(settled_state.sponge_pos)} "
        f"settled_upright_z={settled_state.upright_z:.6f} "
        f"handoff_model={args.handoff_model} "
        f"attach_quat_mode={base_env.cfg.attach_quat_mode} "
        f"attach_velocity_mode={base_env.cfg.attach_velocity_mode} "
        f"posewrite_status={_posewrite_status()}",
        flush=True,
    )

    timeout = False
    nan_seen = False
    episode_done = False
    latch_seen = False
    latch_event_index = -1
    latch_global_step = -1
    latch_local_step = -1
    gripper_threshold_global_step = -1
    preclose_latch_seen = False
    pre_latch_state: SimState | None = None
    latch_state: SimState | None = None
    results: list[EventResult] = []

    def execute_event(event) -> EventResult:
        nonlocal total_sim_steps, timeout, nan_seen, episode_done
        nonlocal latch_seen, latch_event_index, latch_global_step, latch_local_step
        nonlocal gripper_threshold_global_step, preclose_latch_seen, pre_latch_state, latch_state

        target_q = torch.tensor(np.radians(event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        settle_count = 0
        steps_used = 0
        max_sim_tcp_step = 0.0
        max_sponge_speed = 0.0
        min_upright = capture_state().upright_z
        local_latch_step = -1
        prev_state = capture_state()

        for step_idx in range(1, args.max_steps_per_event + 1):
            base_env.robot_dof_targets[:] = target_q
            done = step_once()
            total_sim_steps += 1
            steps_used = step_idx
            state = capture_state()

            target_error = _norm(state.tcp - event.target_tcp)
            if not np.isfinite(state.tcp).all() or not np.isfinite(state.sponge_pos).all() or not math.isfinite(target_error):
                nan_seen = True
            max_sim_tcp_step = max(max_sim_tcp_step, _norm(state.tcp - prev_state.tcp))
            max_sponge_speed = max(max_sponge_speed, state.sponge_speed_mps)
            min_upright = min(min_upright, state.upright_z)
            episode_done |= done

            if gripper_threshold_global_step < 0 and state.gripper_q_rad >= base_env.cfg.grasp_gripper_thresh:
                gripper_threshold_global_step = total_sim_steps
            if state.grasped and not latch_seen:
                latch_seen = True
                latch_event_index = event.index
                latch_global_step = total_sim_steps
                latch_local_step = step_idx
                local_latch_step = step_idx
                pre_latch_state = prev_state
                latch_state = state
                preclose_latch_seen = event.kind == "PRE_MOVE"

            if event.kind == "CLOSE":
                reached = target_error <= args.target_error_gate_m and state.gripper_q_rad >= base_env.cfg.grasp_gripper_thresh
            else:
                reached = target_error <= args.target_error_gate_m
            settle_count = settle_count + 1 if reached else 0
            prev_state = state
            if reached and settle_count >= args.settle_steps:
                break
            if latch_seen and event.kind == "CLOSE":
                break

        final_state = capture_state()
        final_target_error = _norm(final_state.tcp - event.target_tcp)
        sponge_xy_drift = _norm(final_state.sponge_pos[:2] - settled_state.sponge_pos[:2])
        sponge_z_delta = float(final_state.sponge_pos[2] - settled_state.sponge_pos[2])
        reached = final_target_error <= args.target_error_gate_m
        if event.kind == "CLOSE":
            reached = reached and final_state.gripper_q_rad >= base_env.cfg.grasp_gripper_thresh
        if steps_used >= args.max_steps_per_event and not reached:
            timeout = True

        return EventResult(
            index=event.index,
            kind=event.kind,
            phase=event.phase,
            segment=event.segment,
            reached=reached,
            steps=steps_used,
            final_target_error_m=final_target_error,
            max_sim_tcp_step_m=max_sim_tcp_step,
            sponge_xy_drift_m=sponge_xy_drift,
            sponge_z_delta_m=sponge_z_delta,
            max_sponge_speed_mps=max_sponge_speed,
            min_upright_z=min_upright,
            latch_seen=latch_seen,
            latch_step_local=local_latch_step,
        )

    for event in contact_events:
        result = execute_event(event)
        results.append(result)
        state = capture_state()
        should_log = (
            event.index <= 5
            or event.kind == "CLOSE"
            or event.index % args.log_every_event == 0
            or not result.reached
            or result.sponge_xy_drift_m > 0.001
            or result.latch_step_local > 0
        )
        if should_log:
            print(
                f"[roarm_chain_handoff] event_index={result.index:03d} event={result.kind} "
                f"phase={result.phase} segment={result.segment} steps={result.steps} "
                f"final_target_error_m={result.final_target_error_m:.6f} "
                f"max_sim_tcp_step_m={result.max_sim_tcp_step_m:.6f} "
                f"gripper_q_deg={math.degrees(state.gripper_q_rad):+.2f} "
                f"d_tcp_sponge_m={state.d_tcp_sponge_m:.6f} "
                f"sponge_xy_drift_m={result.sponge_xy_drift_m:.6f} "
                f"sponge_z_delta_m={result.sponge_z_delta_m:+.6f} "
                f"max_sponge_speed_mps={result.max_sponge_speed_mps:.6f} "
                f"min_upright_z={result.min_upright_z:.6f} "
                f"latch_seen={_yes(result.latch_seen)} latch_step={result.latch_step_local} "
                f"reached={_yes(result.reached)}",
                flush=True,
            )
        if latch_seen:
            break

    if pre_latch_state is None:
        pre_latch_state = capture_state()
    if latch_state is None:
        latch_state = capture_state()

    latch_pose_jump = _norm(latch_state.sponge_pos - pre_latch_state.sponge_pos)
    latch_xy_jump = _norm(latch_state.sponge_pos[:2] - pre_latch_state.sponge_pos[:2])
    latch_z_jump = float(latch_state.sponge_pos[2] - pre_latch_state.sponge_pos[2])
    latch_tcp_step = _norm(latch_state.tcp - pre_latch_state.tcp)
    latch_d_tcp_sponge_jump = abs(latch_state.d_tcp_sponge_m - pre_latch_state.d_tcp_sponge_m)
    latch_quat_angle = _quat_angle_deg(latch_state.sponge_quat, pre_latch_state.sponge_quat)
    latch_speed_delta = latch_state.sponge_speed_mps - pre_latch_state.sponge_speed_mps

    print(
        f"[roarm_chain_handoff] latch_boundary pre_step={pre_latch_state.global_step} "
        f"latch_step={latch_state.global_step} latch_event_index={latch_event_index} "
        f"latch_local_step={latch_local_step} gripper_threshold_global_step={gripper_threshold_global_step} "
        f"pre_gripper_q_deg={math.degrees(pre_latch_state.gripper_q_rad):+.2f} "
        f"latch_gripper_q_deg={math.degrees(latch_state.gripper_q_rad):+.2f} "
        f"pre_d_tcp_sponge_m={pre_latch_state.d_tcp_sponge_m:.6f} "
        f"latch_d_tcp_sponge_m={latch_state.d_tcp_sponge_m:.6f} "
        f"pose_jump_m={latch_pose_jump:.6f} xy_jump_m={latch_xy_jump:.6f} z_jump_m={latch_z_jump:+.6f} "
        f"tcp_step_m={latch_tcp_step:.6f} d_tcp_sponge_jump_m={latch_d_tcp_sponge_jump:.6f} "
        f"quat_angle_deg={latch_quat_angle:.3f} upright_pre={pre_latch_state.upright_z:.6f} "
        f"upright_latch={latch_state.upright_z:.6f} pre_speed_mps={pre_latch_state.sponge_speed_mps:.6f} "
        f"latch_speed_mps={latch_state.sponge_speed_mps:.6f} speed_delta_mps={latch_speed_delta:+.6f}",
        flush=True,
    )

    target_q = torch.tensor(np.radians(close_event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
    hold_reference = latch_state
    hold_reference_tcp_to_sponge = hold_reference.sponge_pos - hold_reference.tcp
    hold_max_pose_drift = 0.0
    hold_max_xy_drift = 0.0
    hold_max_d_tcp_sponge_jump = 0.0
    hold_max_offset_error = 0.0
    hold_max_speed = latch_state.sponge_speed_mps
    hold_max_ang_speed = latch_state.sponge_ang_speed_rps
    hold_max_quat_angle = 0.0
    hold_min_upright = latch_state.upright_z
    hold_max_target_error = 0.0
    hold_max_tcp_step = 0.0
    hold_early_kill = False
    hold_steps_done = 0
    prev_hold_state = latch_state

    for step_idx in range(1, args.post_latch_hold_steps + 1):
        base_env.robot_dof_targets[:] = target_q
        done = step_once()
        total_sim_steps += 1
        hold_steps_done = step_idx
        state = capture_state()
        target_error = _norm(state.tcp - close_event.target_tcp)
        pose_drift = _norm(state.sponge_pos - hold_reference.sponge_pos)
        xy_drift = _norm(state.sponge_pos[:2] - hold_reference.sponge_pos[:2])
        d_tcp_jump = abs(state.d_tcp_sponge_m - hold_reference.d_tcp_sponge_m)
        offset_error = _norm((state.sponge_pos - state.tcp) - hold_reference_tcp_to_sponge)
        quat_angle = _quat_angle_deg(state.sponge_quat, hold_reference.sponge_quat)
        tcp_step = _norm(state.tcp - prev_hold_state.tcp)

        if not np.isfinite(state.tcp).all() or not np.isfinite(state.sponge_pos).all() or not math.isfinite(target_error):
            nan_seen = True
        episode_done |= done
        hold_max_pose_drift = max(hold_max_pose_drift, pose_drift)
        hold_max_xy_drift = max(hold_max_xy_drift, xy_drift)
        hold_max_d_tcp_sponge_jump = max(hold_max_d_tcp_sponge_jump, d_tcp_jump)
        hold_max_offset_error = max(hold_max_offset_error, offset_error)
        hold_max_speed = max(hold_max_speed, state.sponge_speed_mps)
        hold_max_ang_speed = max(hold_max_ang_speed, state.sponge_ang_speed_rps)
        hold_max_quat_angle = max(hold_max_quat_angle, quat_angle)
        hold_min_upright = min(hold_min_upright, state.upright_z)
        hold_max_target_error = max(hold_max_target_error, target_error)
        hold_max_tcp_step = max(hold_max_tcp_step, tcp_step)
        prev_hold_state = state

        hold_early_kill = (
            pose_drift > args.post_latch_hold_drift_gate_m
            or d_tcp_jump > args.post_latch_d_tcp_jump_gate_m
            or offset_error > args.post_latch_d_tcp_jump_gate_m
            or state.sponge_speed_mps > args.post_latch_speed_gate_mps
            or quat_angle > args.post_latch_quat_angle_gate_deg
            or state.upright_z < args.min_upright_z_gate
            or target_error > args.target_error_gate_m
            or tcp_step > args.max_tcp_step_m
            or not state.grasped
            or done
            or nan_seen
        )
        if step_idx <= 5 or step_idx == args.post_latch_hold_steps or hold_early_kill:
            print(
                f"[roarm_chain_handoff] post_latch_hold step={step_idx:03d} "
                f"target_error_m={target_error:.6f} tcp_step_m={tcp_step:.6f} "
                f"pose_drift_m={pose_drift:.6f} xy_drift_m={xy_drift:.6f} "
                f"d_tcp_sponge_m={state.d_tcp_sponge_m:.6f} d_tcp_sponge_jump_m={d_tcp_jump:.6f} "
                f"offset_error_m={offset_error:.6f} "
                f"sponge_speed_mps={state.sponge_speed_mps:.6f} sponge_ang_speed_rps={state.sponge_ang_speed_rps:.6f} "
                f"quat_angle_deg={quat_angle:.3f} upright_z={state.upright_z:.6f} "
                f"grasped={_yes(state.grasped)} early_kill={_yes(hold_early_kill)}",
                flush=True,
            )
        if hold_early_kill:
            break

    max_final_target_error = max((r.final_target_error_m for r in results), default=float("inf"))
    max_sim_tcp_step = max((r.max_sim_tcp_step_m for r in results), default=float("inf"))
    max_preclose_xy_drift = max((r.sponge_xy_drift_m for r in results if r.kind == "PRE_MOVE"), default=0.0)
    max_close_xy_drift = max((r.sponge_xy_drift_m for r in results), default=0.0)
    max_sponge_speed = max((r.max_sponge_speed_mps for r in results), default=0.0)
    min_upright = min((r.min_upright_z for r in results), default=1.0)
    final_state = capture_state()
    max_close_xy_drift = max(max_close_xy_drift, _norm(final_state.sponge_pos[:2] - settled_state.sponge_pos[:2]))

    latch_after_threshold_ok = (not latch_seen) or (
        gripper_threshold_global_step >= 0 and latch_global_step >= gripper_threshold_global_step
    )
    latch_boundary_ok = (
        latch_seen
        and not preclose_latch_seen
        and latch_after_threshold_ok
        and latch_pose_jump <= args.post_latch_pose_jump_gate_m
        and latch_d_tcp_sponge_jump <= args.post_latch_d_tcp_jump_gate_m
        and latch_quat_angle <= args.post_latch_quat_angle_gate_deg
        and latch_state.upright_z >= args.min_upright_z_gate
        and latch_state.sponge_speed_mps <= args.post_latch_speed_gate_mps
    )
    hold_ok = (
        hold_steps_done == args.post_latch_hold_steps
        and not hold_early_kill
        and hold_max_pose_drift <= args.post_latch_hold_drift_gate_m
        and hold_max_d_tcp_sponge_jump <= args.post_latch_d_tcp_jump_gate_m
        and hold_max_offset_error <= args.post_latch_d_tcp_jump_gate_m
        and hold_max_speed <= args.post_latch_speed_gate_mps
        and hold_max_quat_angle <= args.post_latch_quat_angle_gate_deg
        and hold_min_upright >= args.min_upright_z_gate
        and hold_max_target_error <= args.target_error_gate_m
        and hold_max_tcp_step <= args.max_tcp_step_m
    )
    reached_all = all(r.reached for r in results)
    controller_latency_ok = reached_all and not timeout and not episode_done
    target_error_ok = max_final_target_error <= args.target_error_gate_m and hold_max_target_error <= args.target_error_gate_m
    sim_step_ok = max(max_sim_tcp_step, hold_max_tcp_step) <= args.max_tcp_step_m
    preclose_passive_ok = (not preclose_latch_seen) and max_preclose_xy_drift <= args.preclose_drift_gate_m
    close_motion_ok = max_close_xy_drift <= args.close_drift_gate_m and min(min_upright, hold_min_upright) >= args.min_upright_z_gate
    success = (
        home_fk_error <= args.home_fk_gate_m
        and controller_latency_ok
        and target_error_ok
        and sim_step_ok
        and preclose_passive_ok
        and close_motion_ok
        and latch_boundary_ok
        and hold_ok
        and not nan_seen
    )

    print(
        f"[roarm_chain_handoff] aggregate executed_events={len(results)} total_sim_steps={total_sim_steps} "
        f"post_latch_hold_steps_done={hold_steps_done} max_final_target_error_m={max_final_target_error:.6f} "
        f"hold_max_target_error_m={hold_max_target_error:.6f} max_sim_tcp_step_m={max(max_sim_tcp_step, hold_max_tcp_step):.6f} "
        f"max_preclose_sponge_xy_drift_m={max_preclose_xy_drift:.6f} "
        f"max_close_sponge_xy_drift_m={max_close_xy_drift:.6f} "
        f"latch_pose_jump_m={latch_pose_jump:.6f} latch_d_tcp_sponge_jump_m={latch_d_tcp_sponge_jump:.6f} "
        f"latch_quat_angle_deg={latch_quat_angle:.3f} hold_max_pose_drift_m={hold_max_pose_drift:.6f} "
        f"hold_max_xy_drift_m={hold_max_xy_drift:.6f} hold_max_d_tcp_sponge_jump_m={hold_max_d_tcp_sponge_jump:.6f} "
        f"hold_max_offset_error_m={hold_max_offset_error:.6f} "
        f"hold_max_speed_mps={hold_max_speed:.6f} hold_max_ang_speed_rps={hold_max_ang_speed:.6f} "
        f"hold_max_quat_angle_deg={hold_max_quat_angle:.3f} hold_min_upright_z={hold_min_upright:.6f} "
        f"latch_seen={_yes(latch_seen)} latch_event_index={latch_event_index} latch_global_step={latch_global_step} "
        f"gripper_threshold_global_step={gripper_threshold_global_step} preclose_latch_seen={_yes(preclose_latch_seen)} "
        f"hold_early_kill={_yes(hold_early_kill)} kinematic_env_latch_only=YES "
        f"handoff_model={args.handoff_model} attach_calls={handoff_stats['attach_calls']} "
        f"posewrite_calls={handoff_stats['posewrite_calls']} "
        f"delay_env_steps_seen={handoff_stats['delay_env_steps_seen']} "
        f"offset_initialized={_yes(bool(handoff_stats['offset_initialized']))} "
        f"attach_quat_mode={args.attach_quat_mode} attach_velocity_mode={args.attach_velocity_mode} "
        f"posewrite_status={_posewrite_status()}",
        flush=True,
    )
    print(
        f"[roarm_chain_handoff] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"controller_latency_ok={_yes(controller_latency_ok)} target_error_ok={_yes(target_error_ok)} "
        f"sim_step_ok={_yes(sim_step_ok)} preclose_passive_ok={_yes(preclose_passive_ok)} "
        f"close_motion_ok={_yes(close_motion_ok)} latch_boundary_ok={_yes(latch_boundary_ok)} "
        f"post_latch_hold_ok={_yes(hold_ok)} attach_physics_validated=NO release_physics_validated=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_handoff] ROARM_POST_CLOSE_HANDOFF_MODEL_SUCCESS={_yes(success)}", flush=True)

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
