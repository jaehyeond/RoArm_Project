#!/usr/bin/env python3
"""Post-latch micro-command executor instrumentation for P7 Branch B.

This narrow pre-integration diagnostic stops at local CLOSE handoff and then
tests whether a tiny post-latch TCP or joint perturbation is actually delivered
to the RoArm articulation. It does not insert constraints, use SurfaceGripper,
go to the transport target, run release, run P7 training, or edit defaults.
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

from p7_branch_b_roarm_chain_dynamics_timing_probe import (  # noqa: E402
    HOME_DEG,
    PICK_WRIST_R_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)
from p7_branch_b_roarm_chain_handoff_micro_motion_probe import (  # noqa: E402
    _fmt_xyz,
    _norm,
    _quat_angle_deg,
    _yes,
)
from roarm_kinematics import clip_joints, ik_dls  # noqa: E402


def _fmt_deg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _fmt_rad(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.6f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--post_latch_hold_steps", type=int, default=5)
    ap.add_argument("--executor_steps", type=int, default=20)
    ap.add_argument("--micro_delta_m", type=float, default=0.004)
    ap.add_argument("--joint_nudge_index", type=int, default=1)
    ap.add_argument("--joint_nudge_deg", type=float, default=1.0)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    ap.add_argument("--micro_mode", choices=["tcp_micro", "joint_nudge"], default="tcp_micro")
    ap.add_argument(
        "--handoff_model",
        choices=["marker_only", "offset_preserve_posewrite"],
        default="marker_only",
        help="Keep marker-only by default so executor debugging is not mistaken for attach physics.",
    )
    args = ap.parse_args()

    if args.resample_fraction <= 0.0 or args.resample_fraction > 1.0:
        raise ValueError("resample_fraction must be in (0, 1]")
    if args.executor_steps <= 0:
        raise ValueError("executor_steps must be positive")
    if args.joint_nudge_index < 0 or args.joint_nudge_index >= 6:
        raise ValueError("joint_nudge_index must be in [0, 5]")

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    close_event = next(event for event in events if event.kind == "CLOSE")
    contact_events = [event for event in events if event.kind in ("PRE_MOVE", "CLOSE")]

    if args.micro_mode == "tcp_micro":
        micro_target_tcp = close_event.target_tcp + np.array([args.micro_delta_m, 0.0, 0.0], dtype=np.float64)
        target_q_deg, ik_converged, ik_err_mm, ik_iter = ik_dls(
            micro_target_tcp, close_event.q_deg, max_iter=200, tol_mm=1.0
        )
        target_q_deg = clip_joints(target_q_deg)
        target_q_deg[4] = PICK_WRIST_R_DEG
        target_q_deg[5] = close_event.q_deg[5]
        micro_label = "tcp_plus_x"
    else:
        target_q_deg = close_event.q_deg.copy()
        target_q_deg[args.joint_nudge_index] += args.joint_nudge_deg
        target_q_deg = clip_joints(target_q_deg)
        micro_target_tcp = fk_tcp(target_q_deg)
        ik_converged = True
        ik_err_mm = 0.0
        ik_iter = 0
        micro_label = f"joint{args.joint_nudge_index}_nudge"
    expected_tcp = fk_tcp(target_q_deg)
    close_expected_tcp = fk_tcp(close_event.q_deg)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z
    from roarm_rl.roarm_stack_env import _quat_rotate

    print("[roarm_chain_micro_executor] post_latch_micro_executor_probe", flush=True)
    print(
        "[roarm_chain_micro_executor] "
        "executor_instrumentation_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO attached_transport=NO transport_target=NO "
        "release_marker=NO p7_training=NO env_default_edits=NO chain_defaults_edits=NO "
        "kinematic_env_latch_only=YES micro_motion_not_transport=YES "
        "attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_micro_executor] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"post_latch_hold_steps={args.post_latch_hold_steps} executor_steps={args.executor_steps} "
        f"micro_mode={args.micro_mode} micro_delta_m={args.micro_delta_m:.6f} "
        f"joint_nudge_index={args.joint_nudge_index} joint_nudge_deg={args.joint_nudge_deg:.3f} "
        f"resample_fraction={args.resample_fraction:.3f} handoff_model={args.handoff_model}",
        flush=True,
    )
    print(
        f"[roarm_chain_micro_executor] stream source_events_total={meta['events_total']} "
        f"executed_events={len(contact_events)} pre_move_cmds={meta['pre_move_cmds']} "
        f"close_index={close_event.index} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
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
    cfg.attach_quat_mode = "preserve"
    cfg.attach_velocity_mode = "zero"

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)
    handoff_stats = {"attach_calls": 0, "posewrite_calls": 0, "offset_initialized": 0}
    preserved_tcp_to_sponge: torch.Tensor | None = None

    def _current_tcp_world(env_ids: torch.Tensor) -> torch.Tensor:
        link5_pos = base_env._robot.data.body_pos_w[env_ids, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[env_ids, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(link5_pos.shape[0], 3))
        return link5_pos + tcp_offset_world

    def _diagnostic_handoff_attach() -> None:
        nonlocal preserved_tcp_to_sponge
        env_ids = torch.where(base_env._grasped)[0]
        if len(env_ids) == 0:
            return
        handoff_stats["attach_calls"] += 1
        if args.handoff_model == "marker_only":
            return
        tcp_pos = _current_tcp_world(env_ids)
        if preserved_tcp_to_sponge is None:
            preserved_tcp_to_sponge = base_env._sponge.data.root_pos_w[env_ids].clone() - tcp_pos
            handoff_stats["offset_initialized"] = 1
        pose7 = torch.zeros((len(env_ids), 7), device=device)
        pose7[:, 0:3] = tcp_pos + preserved_tcp_to_sponge
        pose7[:, 3:7] = base_env._sponge.data.root_quat_w[env_ids]
        base_env._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        base_env._sponge.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=device), env_ids=env_ids)
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

    total_sim_steps = 0
    episode_done = False
    nan_seen = False
    latch_seen = False
    latch_global_step = -1
    latch_event_index = -1

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

    def current_q_rad() -> np.ndarray:
        return base_env._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)

    def targets_rad() -> np.ndarray:
        return base_env.robot_dof_targets[0].detach().cpu().numpy().astype(np.float64)

    def sponge_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy()
        quat = base_env._sponge.data.root_quat_w[0].detach().cpu().numpy()
        vel = base_env._sponge.data.root_vel_w[0].detach().cpu().numpy()
        return pos.astype(np.float64), quat.astype(np.float64), vel.astype(np.float64)

    for _ in range(args.initial_settle_steps):
        step_once()
        total_sim_steps += 1

    home_tcp = fresh_tcp_local()
    home_fk_error = _norm(home_tcp - fk_tcp(HOME_DEG))
    print(
        f"[roarm_chain_micro_executor] initial home_fresh_tcp={_fmt_xyz(home_tcp)} "
        f"home_expected_tcp={_fmt_xyz(fk_tcp(HOME_DEG))} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)}",
        flush=True,
    )

    def execute_contact_event(event) -> tuple[bool, int, float]:
        nonlocal total_sim_steps, episode_done, latch_seen, latch_global_step, latch_event_index, nan_seen
        target_q = torch.tensor(np.radians(event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        settle_count = 0
        final_error = float("inf")
        steps_used = 0
        for step_idx in range(1, args.max_steps_per_event + 1):
            base_env.robot_dof_targets[:] = target_q
            done = step_once()
            total_sim_steps += 1
            episode_done |= done
            steps_used = step_idx
            tcp = fresh_tcp_local()
            final_error = _norm(tcp - event.target_tcp)
            if not np.isfinite(tcp).all() or not math.isfinite(final_error):
                nan_seen = True
            gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
            if bool(base_env._grasped[0].detach().cpu().item()) and not latch_seen:
                latch_seen = True
                latch_global_step = total_sim_steps
                latch_event_index = event.index
            reached = final_error <= args.target_error_gate_m
            if event.kind == "CLOSE":
                reached = reached and gripper_q >= base_env.cfg.grasp_gripper_thresh
            settle_count = settle_count + 1 if reached else 0
            if settle_count >= args.settle_steps or (event.kind == "CLOSE" and latch_seen):
                break
        return final_error <= args.target_error_gate_m, steps_used, final_error

    for event in contact_events:
        reached, steps, final_error = execute_contact_event(event)
        if event.index <= 5 or event.index % args.log_every_event == 0 or event.kind == "CLOSE" or not reached:
            print(
                f"[roarm_chain_micro_executor] event_index={event.index:03d} event={event.kind} "
                f"phase={event.phase} segment={event.segment} steps={steps} "
                f"final_target_error_m={final_error:.6f} reached={_yes(reached)} "
                f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
                flush=True,
            )
        if latch_seen:
            break

    close_tcp = fresh_tcp_local()
    close_q_rad = current_q_rad()
    close_q_deg_actual = np.degrees(close_q_rad)
    close_q_deg_cmd = close_event.q_deg
    close_sponge_pos, close_sponge_quat, close_sponge_vel = sponge_state()
    close_tcp_to_sponge = close_sponge_pos - close_tcp

    close_target_q_rad = torch.tensor(np.radians(close_event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
    for hold_idx in range(1, args.post_latch_hold_steps + 1):
        base_env.robot_dof_targets[:] = close_target_q_rad
        done = step_once()
        total_sim_steps += 1
        episode_done |= done
        tcp = fresh_tcp_local()
        sponge_pos, sponge_quat, sponge_vel = sponge_state()
        target_error = _norm(tcp - close_event.target_tcp)
        pose_drift = _norm(sponge_pos - close_sponge_pos)
        offset_error = _norm((sponge_pos - tcp) - close_tcp_to_sponge)
        quat_angle = _quat_angle_deg(sponge_quat, close_sponge_quat)
        print(
            f"[roarm_chain_micro_executor] post_latch_hold step={hold_idx:03d} "
            f"target_error_m={target_error:.6f} pose_drift_m={pose_drift:.6f} "
            f"offset_error_m={offset_error:.6f} sponge_speed_mps={_norm(sponge_vel[0:3]):.6f} "
            f"quat_angle_deg={quat_angle:.3f} grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
            flush=True,
        )

    target_q_rad_np = np.radians(target_q_deg)
    target_q_rad = torch.tensor(target_q_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
    close_to_target_q_deg = target_q_deg - close_q_deg_cmd
    close_to_target_fk = _norm(expected_tcp - close_expected_tcp)
    print(
        f"[roarm_chain_micro_executor] micro_plan label={micro_label} ik_converged={_yes(bool(ik_converged))} "
        f"ik_err_mm={float(ik_err_mm):.3f} ik_iter={int(ik_iter)} "
        f"close_cmd_q_deg={_fmt_deg(close_q_deg_cmd)} target_q_deg={_fmt_deg(target_q_deg)} "
        f"delta_q_deg={_fmt_deg(close_to_target_q_deg)} delta_q_norm_deg={_norm(close_to_target_q_deg):.6f} "
        f"delta_q_max_abs_deg={float(np.max(np.abs(close_to_target_q_deg))):.6f} "
        f"close_expected_tcp={_fmt_xyz(close_expected_tcp)} target_tcp={_fmt_xyz(micro_target_tcp)} "
        f"expected_tcp={_fmt_xyz(expected_tcp)} expected_tcp_delta_m={close_to_target_fk:.6f}",
        flush=True,
    )

    overwrite_seen = False
    reached_seen = False
    max_realized_tcp_delta = 0.0
    max_expected_tcp_error = 0.0
    max_target_tcp_error = 0.0
    min_joint_error_max_deg = float("inf")
    prev_tcp = fresh_tcp_local()

    for step_idx in range(1, args.executor_steps + 1):
        before_targets = targets_rad()
        before_q = current_q_rad()
        before_tcp = fresh_tcp_local()
        base_env.robot_dof_targets[:] = target_q_rad
        commanded_targets = targets_rad()
        done = step_once()
        total_sim_steps += 1
        episode_done |= done
        after_targets = targets_rad()
        after_q = current_q_rad()
        after_tcp = fresh_tcp_local()
        sponge_pos, sponge_quat, sponge_vel = sponge_state()

        target_overwrite_rad = after_targets - target_q_rad_np
        overwrite_step = float(np.max(np.abs(target_overwrite_rad))) > 1.0e-5
        overwrite_seen |= overwrite_step
        joint_error_rad = target_q_rad_np - after_q
        joint_error_deg = np.degrees(joint_error_rad)
        joint_error_max_deg = float(np.max(np.abs(joint_error_deg)))
        min_joint_error_max_deg = min(min_joint_error_max_deg, joint_error_max_deg)
        realized_tcp_delta = _norm(after_tcp - close_tcp)
        step_tcp_delta = _norm(after_tcp - prev_tcp)
        expected_tcp_error = _norm(after_tcp - expected_tcp)
        target_tcp_error = _norm(after_tcp - micro_target_tcp)
        max_realized_tcp_delta = max(max_realized_tcp_delta, realized_tcp_delta)
        max_expected_tcp_error = max(max_expected_tcp_error, expected_tcp_error)
        max_target_tcp_error = max(max_target_tcp_error, target_tcp_error)
        reached = target_tcp_error <= args.target_error_gate_m
        reached_seen |= reached
        offset_error = _norm((sponge_pos - after_tcp) - close_tcp_to_sponge)
        quat_angle = _quat_angle_deg(sponge_quat, close_sponge_quat)
        if not np.isfinite(after_tcp).all() or not math.isfinite(target_tcp_error):
            nan_seen = True

        should_log = step_idx <= 5 or step_idx == args.executor_steps or reached or overwrite_step or done
        if should_log:
            print(
                f"[roarm_chain_micro_executor] executor_step={step_idx:03d} "
                f"before_targets_rad={_fmt_rad(before_targets)} commanded_targets_rad={_fmt_rad(commanded_targets)} "
                f"after_targets_rad={_fmt_rad(after_targets)} target_overwrite_max_rad={float(np.max(np.abs(target_overwrite_rad))):.8f} "
                f"overwrite_after_step={_yes(overwrite_step)} current_q_deg={_fmt_deg(np.degrees(after_q))} "
                f"joint_error_deg={_fmt_deg(joint_error_deg)} joint_error_max_deg={joint_error_max_deg:.6f} "
                f"expected_tcp={_fmt_xyz(expected_tcp)} fresh_tcp={_fmt_xyz(after_tcp)} "
                f"expected_tcp_error_m={expected_tcp_error:.6f} target_tcp_error_m={target_tcp_error:.6f} "
                f"realized_tcp_delta_from_close_m={realized_tcp_delta:.6f} step_tcp_delta_m={step_tcp_delta:.6f} "
                f"before_tcp_delta_m={_norm(before_tcp - close_tcp):.6f} "
                f"offset_error_m={offset_error:.6f} sponge_speed_mps={_norm(sponge_vel[0:3]):.6f} "
                f"quat_angle_deg={quat_angle:.3f} grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
                f"reached={_yes(reached)} done={_yes(done)}",
                flush=True,
            )
        prev_tcp = after_tcp
        if reached:
            break

    executor_target_q_distinct = float(np.max(np.abs(close_to_target_q_deg))) > 1.0e-4
    expected_motion_nonzero = close_to_target_fk > 1.0e-4
    targets_not_overwritten = not overwrite_seen
    realized_motion_seen = max_realized_tcp_delta > max(0.25 * args.micro_delta_m, 0.0005)
    executor_reached = reached_seen

    print(
        f"[roarm_chain_micro_executor] aggregate executed_events={latch_event_index if latch_event_index > 0 else len(contact_events)} "
        f"total_sim_steps={total_sim_steps} latch_seen={_yes(latch_seen)} latch_event_index={latch_event_index} "
        f"latch_global_step={latch_global_step} micro_mode={args.micro_mode} "
        f"target_q_distinct={_yes(executor_target_q_distinct)} expected_motion_nonzero={_yes(expected_motion_nonzero)} "
        f"targets_not_overwritten={_yes(targets_not_overwritten)} realized_motion_seen={_yes(realized_motion_seen)} "
        f"executor_reached={_yes(executor_reached)} max_realized_tcp_delta_m={max_realized_tcp_delta:.6f} "
        f"max_expected_tcp_error_m={max_expected_tcp_error:.6f} max_target_tcp_error_m={max_target_tcp_error:.6f} "
        f"min_joint_error_max_deg={min_joint_error_max_deg:.6f} attach_calls={handoff_stats['attach_calls']} "
        f"posewrite_calls={handoff_stats['posewrite_calls']} offset_initialized={_yes(bool(handoff_stats['offset_initialized']))} "
        f"handoff_model={args.handoff_model} action_scale={base_env.cfg.action_scale:.6f} null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f} "
        f"kinematic_env_latch_only=YES micro_motion_not_transport=YES",
        flush=True,
    )

    success = (
        home_fk_error <= args.home_fk_gate_m
        and latch_seen
        and executor_target_q_distinct
        and expected_motion_nonzero
        and targets_not_overwritten
        and realized_motion_seen
        and executor_reached
        and not episode_done
        and not nan_seen
    )
    print(
        f"[roarm_chain_micro_executor] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"latch_seen_ok={_yes(latch_seen)} target_q_distinct_ok={_yes(executor_target_q_distinct)} "
        f"expected_motion_ok={_yes(expected_motion_nonzero)} targets_not_overwritten_ok={_yes(targets_not_overwritten)} "
        f"realized_motion_ok={_yes(realized_motion_seen)} executor_reached_ok={_yes(executor_reached)} "
        f"attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_micro_executor] ROARM_POST_LATCH_MICRO_EXECUTOR_SUCCESS={_yes(success)}", flush=True)

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
