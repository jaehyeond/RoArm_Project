#!/usr/bin/env python3
"""Top-tangent local constraint handoff smoke for P7 Branch B.

This is a diagnostic-only bridge between the verified RoArm top-tangent local
signal and the isolated dynamic-anchor fixed-joint contract. It stops before
MOVE transport and before release. It does not use SurfaceGripper, does not
edit env/train/chain defaults, and does not use env TCP-center pose-write as
success evidence.
"""
from __future__ import annotations

import argparse
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

from p7_branch_b_roarm_chain_close_near_local_signal_probe import (  # noqa: E402
    LocalEvent,
    _build_safe_events,
    _fmt_quat,
    _fmt_xyz,
    _norm,
    _quat_angle_deg,
    _target_geometry,
    _yes,
)
from p7_branch_b_roarm_chain_dynamics_timing_probe import (  # noqa: E402
    GRIPPER_OPEN_DEG,
    HOME_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)


@dataclass
class HandoffResult:
    label: str
    phase: str
    reached: bool
    steps: int
    final_target_error_m: float
    final_anchor_target_error_m: float
    final_object_target_error_m: float
    max_tcp_step_m: float
    max_anchor_target_error_m: float
    max_object_target_error_m: float
    max_tcp_anchor_offset_error_m: float
    max_anchor_object_offset_error_m: float
    max_sponge_drift_m: float
    max_sponge_speed_mps: float
    max_quat_angle_deg: float
    min_upright_z: float
    set_target_seen: bool
    max_set_diff_rad: float
    early_kill: bool


def _tensor_to_np(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float64)


def _tensor_to_list(value) -> list[float]:
    return [float(x) for x in value.detach().cpu().flatten().tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--reassert_sponge_z_m", type=float, default=0.0235)
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--clearance_margin_m", type=float, default=0.024)
    ap.add_argument("--top_margin_m", type=float, default=0.0005)
    ap.add_argument("--micro_delta_m", type=float, default=0.004)
    ap.add_argument("--stationary_hold_steps", type=int, default=5)
    ap.add_argument("--steps_per_local_event", type=int, default=60)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--preclose_drift_gate_m", type=float, default=0.005)
    ap.add_argument("--stationary_speed_gate_mps", type=float, default=0.050)
    ap.add_argument("--micro_speed_gate_mps", type=float, default=0.500)
    ap.add_argument("--quat_angle_gate_deg", type=float, default=5.0)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.90)
    ap.add_argument("--tcp_to_anchor_offset", nargs=3, type=float, default=[0.015, 0.0, -0.010])
    ap.add_argument("--tcp_anchor_offset_gate_m", type=float, default=1.0e-6)
    ap.add_argument("--anchor_object_offset_gate_m", type=float, default=0.003)
    ap.add_argument("--anchor_mass", type=float, default=100.0)
    ap.add_argument("--target_kp", type=float, default=8.0)
    ap.add_argument("--max_cmd_speed", type=float, default=0.080)
    ap.add_argument("--stop_target_error_m", type=float, default=0.0015)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    args = ap.parse_args()

    if args.micro_delta_m <= 0.0:
        raise ValueError("micro_delta_m must be positive")

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    pre_move_events = [event for event in events if event.kind == "PRE_MOVE"]
    base_q = pre_move_events[-1].q_deg.copy()

    local_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        reassert_sponge_z_m=args.reassert_sponge_z_m,
        geometry="top_tangent",
        clearance_margin_m=args.clearance_margin_m,
        top_margin_m=args.top_margin_m,
        above_margin_m=0.0010,
        side_margin_m=0.0020,
        side_top_margin_m=-0.0030,
        micro_delta_m=args.micro_delta_m,
    )
    local_events, ik_records = _build_safe_events(local_args, base_q, GRIPPER_OPEN_DEG)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from pxr import Gf, Sdf, UsdPhysics
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    print("[roarm_chain_top_tangent_handoff] top-tangent local constraint handoff probe", flush=True)
    print(
        "[roarm_chain_top_tangent_handoff] "
        "local_constraint_handoff_only=YES geometry=top_tangent "
        "constraint_prim_insertion=YES fixed_dynamic_constraint_integration=DIAGNOSTIC_LOCAL_ONLY "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "scripted_release_variant=NO p7_training=NO p7_tuning=NO diagnostic_gate_tuning=NO "
        "env_default_edits=NO chain_defaults_edits=NO claim_transport_success=NO "
        "release_physics_validated=NO claim_p7_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_top_tangent_handoff] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"preclose_drift_gate_m={args.preclose_drift_gate_m:.6f} "
        f"stationary_speed_gate_mps={args.stationary_speed_gate_mps:.6f} "
        f"micro_speed_gate_mps={args.micro_speed_gate_mps:.6f} "
        f"quat_angle_gate_deg={args.quat_angle_gate_deg:.2f} min_upright_z_gate={args.min_upright_z_gate:.3f} "
        f"tcp_anchor_offset_gate_m={args.tcp_anchor_offset_gate_m:.8f} "
        f"anchor_object_offset_gate_m={args.anchor_object_offset_gate_m:.6f} "
        f"micro_delta_m={args.micro_delta_m:.6f}",
        flush=True,
    )
    print(
        f"[roarm_chain_top_tangent_handoff] stream source_events_total={meta['events_total']} "
        f"pre_move_cmds={meta['pre_move_cmds']} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )
    for key, value in ik_records.items():
        print(f"[roarm_chain_top_tangent_handoff] ik_record {key} {value}", flush=True)

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
    stage = base_env.sim.stage

    anchor_path = "/World/envs/env_0/BranchBTopTangentAnchor"
    sponge_path = "/World/envs/env_0/Sponge"
    joint_path = "/World/envs/env_0/BranchBTopTangentLocalFixedJoint"
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
    if stage.GetPrimAtPath(anchor_path).IsValid():
        stage.RemovePrim(anchor_path)

    anchor = RigidObject(
        RigidObjectCfg(
            prim_path=anchor_path,
            spawn=sim_utils.CuboidCfg(
                size=(0.012, 0.012, 0.012),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=args.anchor_mass),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.20)),
        )
    )

    original_set_joint_position_target = base_env._robot.set_joint_position_target
    attach_stats = {"attach_calls": 0, "posewrite_calls": 0}
    watch = {"active": False, "target": None, "calls": 0, "max_diff": 0.0}

    def marker_only_attach() -> None:
        attach_stats["attach_calls"] += 1
        return

    def watched_set_joint_position_target(target, *a, **kw):
        if watch["active"] and watch["target"] is not None:
            target_np = target.detach().cpu().numpy().astype(np.float64)
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], float(np.max(np.abs(target_np - watch["target"]))))
        return original_set_joint_position_target(target, *a, **kw)

    base_env._update_grasp_attach = marker_only_attach
    base_env._robot.set_joint_position_target = watched_set_joint_position_target

    env.reset()
    anchor.reset()
    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env._robot.set_joint_position_target(home_rad)
    base_env.robot_dof_targets[:] = home_rad
    sponge_pose = torch.tensor(
        [[args.sponge_xy[0], args.sponge_xy[1], args.reassert_sponge_z_m, 1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    sponge_pose[:, 0:3] += base_env.scene.env_origins[:1]
    base_env._sponge.write_root_pose_to_sim(sponge_pose)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    base_env._grasped[:] = False
    base_env._was_grasped[:] = False
    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)
    zero_anchor_vel = torch.zeros((1, 6), device=device, dtype=torch.float32)

    def step_once() -> bool:
        out = env.step(null_action)
        if hasattr(anchor, "update"):
            anchor.update(base_env.sim.get_physics_dt())
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

    def sponge_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def sponge_quat() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_vel6() -> np.ndarray:
        return base_env._sponge.data.root_vel_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_upright_z() -> float:
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        world_z = _quat_rotate(base_env._sponge.data.root_quat_w[:1], z_axis)
        return float(world_z[0, 2].detach().cpu().item())

    def anchor_local() -> np.ndarray:
        return _tensor_to_np(anchor.data.root_pos_w[0] - base_env.scene.env_origins[0])

    def write_anchor_pose_local(pos_local: np.ndarray) -> None:
        pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
        pose[:, 0:3] = torch.tensor(pos_local, device=device, dtype=torch.float32).unsqueeze(0) + base_env.scene.env_origins[:1]
        pose[:, 3] = 1.0
        anchor.write_root_pose_to_sim(pose)
        anchor.write_root_velocity_to_sim(zero_anchor_vel)

    def write_anchor_velocity_to_target(target_local: np.ndarray) -> tuple[float, np.ndarray]:
        current = anchor_local()
        remaining = np.asarray(target_local, dtype=np.float64) - current
        remaining_norm = _norm(remaining)
        vel_np = np.zeros(3, dtype=np.float64)
        if remaining_norm > args.stop_target_error_m:
            speed = min(args.max_cmd_speed, args.target_kp * remaining_norm)
            vel_np = remaining / max(remaining_norm, 1.0e-9) * speed
        vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
        vel[:, 0:3] = torch.tensor(vel_np, device=device, dtype=torch.float32).unsqueeze(0)
        anchor.write_root_velocity_to_sim(vel)
        return remaining_norm, vel_np

    total_sim_steps = 0
    episode_done = False
    nan_seen = False
    handoff_created = False
    joint_created = False
    handoff_anchor_object_offset: np.ndarray | None = None
    handoff_tcp_anchor_offset = np.asarray(args.tcp_to_anchor_offset, dtype=np.float64)
    post_handoff_sponge_posewrite_calls = 0

    for _ in range(args.initial_settle_steps):
        episode_done |= step_once()
        total_sim_steps += 1

    settled_tcp = fresh_tcp_local()
    settled_sponge = sponge_local()
    settled_quat = sponge_quat()
    home_fk_error = _norm(settled_tcp - fk_tcp(HOME_DEG))
    print(
        f"[roarm_chain_top_tangent_handoff] initial home_fresh_tcp={_fmt_xyz(settled_tcp)} "
        f"home_expected_tcp={_fmt_xyz(fk_tcp(HOME_DEG))} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"settled_sponge_pos={_fmt_xyz(settled_sponge)} settled_sponge_quat_wxyz={_fmt_quat(settled_quat)} "
        f"settled_upright_z={sponge_upright_z():.6f}",
        flush=True,
    )

    def create_handoff_at_signal_pose(signal_event: LocalEvent) -> bool:
        nonlocal handoff_created, joint_created, handoff_anchor_object_offset, total_sim_steps, episode_done
        tcp = fresh_tcp_local()
        sponge = sponge_local()
        anchor_pos = tcp + handoff_tcp_anchor_offset
        write_anchor_pose_local(anchor_pos)
        anchor.update(base_env.sim.get_physics_dt())
        handoff_anchor_object_offset = sponge - anchor_pos
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(sponge_path)])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(
                float(handoff_anchor_object_offset[0]),
                float(handoff_anchor_object_offset[1]),
                float(handoff_anchor_object_offset[2]),
            )
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.GetJointEnabledAttr().Set(True)
        for _ in range(args.settle_steps):
            anchor.write_root_velocity_to_sim(zero_anchor_vel)
            done = step_once()
            total_sim_steps += 1
            episode_done |= done
            if done:
                break
        joint_created = stage.GetPrimAtPath(joint_path).IsValid()
        handoff_created = joint_created
        print(
            "[roarm_chain_top_tangent_handoff] handoff_create "
            f"label={signal_event.label} constraint_prim_insertion={_yes(joint_created)} "
            "fixed_dynamic_constraint_integration=DIAGNOSTIC_LOCAL_ONLY "
            f"anchor_path={anchor_path} sponge_path={sponge_path} joint_path={joint_path} "
            f"tcp_at_handoff={_fmt_xyz(tcp)} anchor_at_handoff={_fmt_xyz(anchor_pos)} "
            f"sponge_at_handoff={_fmt_xyz(sponge)} "
            f"tcp_anchor_offset={_fmt_xyz(handoff_tcp_anchor_offset)} "
            f"anchor_object_offset={_fmt_xyz(handoff_anchor_object_offset)} "
            "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO",
            flush=True,
        )
        return handoff_created

    def run_to_event(event: LocalEvent, max_steps: int, phase: str, drive_anchor: bool) -> HandoffResult:
        nonlocal total_sim_steps, episode_done, nan_seen, post_handoff_sponge_posewrite_calls
        target_rad_np = np.radians(event.q_deg)
        target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
        target_anchor = event.target_tcp + handoff_tcp_anchor_offset
        anchor_object_offset = (
            np.zeros(3, dtype=np.float64)
            if handoff_anchor_object_offset is None
            else handoff_anchor_object_offset.copy()
        )
        target_object = target_anchor + anchor_object_offset
        start_sponge = sponge_local()
        start_quat = sponge_quat()
        prev_tcp = fresh_tcp_local()
        settle_count = 0
        steps_used = 0
        reached = False
        early_kill = False
        max_tcp_step = 0.0
        max_anchor_target_error = 0.0
        max_object_target_error = 0.0
        max_tcp_anchor_offset_error = 0.0
        max_anchor_object_offset_error = 0.0
        max_sponge_drift = 0.0
        max_sponge_speed = 0.0
        max_quat_angle = 0.0
        min_upright = sponge_upright_z()
        final_error = float("inf")
        final_anchor_error = float("inf")
        final_object_error = float("inf")
        watch["active"] = True
        watch["target"] = target_rad_np
        watch["calls"] = 0
        watch["max_diff"] = 0.0
        for step_idx in range(1, max_steps + 1):
            base_env.robot_dof_targets[:] = target_rad
            if drive_anchor:
                write_anchor_velocity_to_target(target_anchor)
            else:
                anchor.write_root_velocity_to_sim(zero_anchor_vel)
            done = step_once()
            total_sim_steps += 1
            steps_used = step_idx
            tcp = fresh_tcp_local()
            anchor_pos = anchor_local()
            sponge = sponge_local()
            quat = sponge_quat()
            vel = sponge_vel6()
            target_error = _norm(tcp - event.target_tcp)
            anchor_target_error = _norm(anchor_pos - target_anchor)
            object_target_error = _norm(sponge - target_object)
            tcp_step = _norm(tcp - prev_tcp)
            tcp_anchor_offset_error = _norm((anchor_pos - tcp) - handoff_tcp_anchor_offset)
            anchor_object_offset_error = _norm((sponge - anchor_pos) - anchor_object_offset)
            sponge_drift = _norm(sponge - start_sponge)
            sponge_speed = _norm(vel[0:3])
            quat_angle = _quat_angle_deg(quat, start_quat)
            upright = sponge_upright_z()
            final_error = target_error
            final_anchor_error = anchor_target_error
            final_object_error = object_target_error
            max_tcp_step = max(max_tcp_step, tcp_step)
            max_anchor_target_error = max(max_anchor_target_error, anchor_target_error)
            max_object_target_error = max(max_object_target_error, object_target_error)
            max_tcp_anchor_offset_error = max(max_tcp_anchor_offset_error, tcp_anchor_offset_error)
            max_anchor_object_offset_error = max(max_anchor_object_offset_error, anchor_object_offset_error)
            max_sponge_drift = max(max_sponge_drift, sponge_drift)
            max_sponge_speed = max(max_sponge_speed, sponge_speed)
            max_quat_angle = max(max_quat_angle, quat_angle)
            min_upright = min(min_upright, upright)
            if not np.isfinite(tcp).all() or not np.isfinite(sponge).all() or not math.isfinite(target_error):
                nan_seen = True
            episode_done |= done
            speed_gate = args.stationary_speed_gate_mps if phase == "stationary_hold" else args.micro_speed_gate_mps
            early_kill = (
                tcp_step > args.max_tcp_step_m
                or sponge_drift > args.preclose_drift_gate_m
                or sponge_speed > speed_gate
                or quat_angle > args.quat_angle_gate_deg
                or upright < args.min_upright_z_gate
                or tcp_anchor_offset_error > args.tcp_anchor_offset_gate_m
                or anchor_object_offset_error > args.anchor_object_offset_gate_m
                or (drive_anchor and anchor_target_error > args.target_error_gate_m and step_idx == max_steps)
                or (drive_anchor and object_target_error > args.target_error_gate_m and step_idx == max_steps)
                or (phase == "stationary_hold" and target_error > args.target_error_gate_m)
                or done
                or nan_seen
            )
            reached = (
                target_error <= args.target_error_gate_m
                and (not drive_anchor or anchor_target_error <= args.target_error_gate_m)
                and (not drive_anchor or object_target_error <= args.target_error_gate_m)
            )
            settle_count = settle_count + 1 if reached else 0
            if step_idx <= 3 or reached or early_kill or step_idx == max_steps:
                geom = _target_geometry(event.target_tcp, sponge, quat)
                print(
                    f"[roarm_chain_top_tangent_handoff] event label={event.label} phase={phase} "
                    f"role={event.role} drive_anchor={_yes(drive_anchor)} step={step_idx:03d} "
                    f"target_tcp={_fmt_xyz(event.target_tcp)} fresh_tcp={_fmt_xyz(tcp)} "
                    f"target_error_m={target_error:.6f} anchor={_fmt_xyz(anchor_pos)} "
                    f"anchor_target_error_m={anchor_target_error:.6f} "
                    f"object_target_error_m={object_target_error:.6f} tcp_step_m={tcp_step:.6f} "
                    f"tcp_anchor_offset_error_m={tcp_anchor_offset_error:.8f} "
                    f"anchor_object_offset_error_m={anchor_object_offset_error:.6f} "
                    f"sponge_drift_m={sponge_drift:.6f} sponge_speed_mps={sponge_speed:.6f} "
                    f"quat_angle_deg={quat_angle:.3f} upright_z={upright:.6f} "
                    f"target_top_class={geom['target_top_class']} "
                    f"target_xy_inside_sponge_aabb={_yes(bool(geom['target_xy_inside_sponge_aabb']))} "
                    f"target_tcp_minus_sponge_oriented_top_m={float(geom['target_tcp_minus_sponge_oriented_top_m']):+.6f} "
                    f"set_calls={watch['calls']} set_max_diff_rad={watch['max_diff']:.8f} "
                    f"grasped_marker={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
                    f"reached={_yes(reached)} early_kill={_yes(early_kill)}",
                    flush=True,
                )
            prev_tcp = tcp
            if early_kill or (phase != "stationary_hold" and settle_count >= args.settle_steps):
                break
        watch["active"] = False
        if drive_anchor and attach_stats["posewrite_calls"] > post_handoff_sponge_posewrite_calls:
            post_handoff_sponge_posewrite_calls = attach_stats["posewrite_calls"]
        if not reached and not early_kill:
            early_kill = True
        return HandoffResult(
            label=event.label,
            phase=phase,
            reached=reached,
            steps=steps_used,
            final_target_error_m=final_error,
            final_anchor_target_error_m=final_anchor_error,
            final_object_target_error_m=final_object_error,
            max_tcp_step_m=max_tcp_step,
            max_anchor_target_error_m=max_anchor_target_error,
            max_object_target_error_m=max_object_target_error,
            max_tcp_anchor_offset_error_m=max_tcp_anchor_offset_error,
            max_anchor_object_offset_error_m=max_anchor_object_offset_error,
            max_sponge_drift_m=max_sponge_drift,
            max_sponge_speed_mps=max_sponge_speed,
            max_quat_angle_deg=max_quat_angle,
            min_upright_z=min_upright,
            set_target_seen=watch["calls"] > 0 and watch["max_diff"] <= 1.0e-5,
            max_set_diff_rad=float(watch["max_diff"]),
            early_kill=early_kill,
        )

    prep_results: list[HandoffResult] = []
    for event in pre_move_events:
        result = run_to_event(
            LocalEvent(event.index, f"prep_{event.index:03d}_{event.segment}", event.q_deg, event.target_tcp, "source_pre_move"),
            args.steps_per_local_event,
            "prep",
            drive_anchor=False,
        )
        prep_results.append(result)
        if event.index <= 5 or event.index % args.log_every_event == 0 or not result.reached:
            print(
                f"[roarm_chain_top_tangent_handoff] prep_result index={event.index:03d} "
                f"label={result.label} reached={_yes(result.reached)} "
                f"final_target_error_m={result.final_target_error_m:.6f} "
                f"max_tcp_step_m={result.max_tcp_step_m:.6f} "
                f"set_target_seen={_yes(result.set_target_seen)} early_kill={_yes(result.early_kill)}",
                flush=True,
            )
        if not result.reached or result.early_kill:
            break

    local_results: list[HandoffResult] = []
    handoff_ok = False
    if prep_results and prep_results[-1].reached and not prep_results[-1].early_kill:
        for idx, event in enumerate(local_events):
            phase = "stationary_reach" if idx < 2 else "micro"
            drive_anchor = handoff_created
            result = run_to_event(event, args.steps_per_local_event, phase, drive_anchor=drive_anchor)
            local_results.append(result)
            print(
                f"[roarm_chain_top_tangent_handoff] local_result label={result.label} phase={phase} "
                f"reached={_yes(result.reached)} steps={result.steps} "
                f"final_target_error_m={result.final_target_error_m:.6f} "
                f"final_anchor_target_error_m={result.final_anchor_target_error_m:.6f} "
                f"final_object_target_error_m={result.final_object_target_error_m:.6f} "
                f"max_tcp_anchor_offset_error_m={result.max_tcp_anchor_offset_error_m:.8f} "
                f"max_anchor_object_offset_error_m={result.max_anchor_object_offset_error_m:.6f} "
                f"set_target_seen={_yes(result.set_target_seen)} early_kill={_yes(result.early_kill)}",
                flush=True,
            )
            if result.early_kill:
                break
            if "stationary_signal_pose" in event.label:
                handoff_ok = create_handoff_at_signal_pose(event)
                hold_event = LocalEvent(
                    event.index,
                    f"{event.label}_handoff_hold",
                    event.q_deg.copy(),
                    event.target_tcp.copy(),
                    "stationary_hold_after_constraint_handoff",
                )
                hold_result = run_to_event(hold_event, args.stationary_hold_steps, "stationary_hold", drive_anchor=True)
                local_results.append(hold_result)
                print(
                    f"[roarm_chain_top_tangent_handoff] local_result label={hold_result.label} "
                    f"phase=stationary_hold reached={_yes(hold_result.reached)} steps={hold_result.steps} "
                    f"final_target_error_m={hold_result.final_target_error_m:.6f} "
                    f"final_anchor_target_error_m={hold_result.final_anchor_target_error_m:.6f} "
                    f"final_object_target_error_m={hold_result.final_object_target_error_m:.6f} "
                    f"max_tcp_anchor_offset_error_m={hold_result.max_tcp_anchor_offset_error_m:.8f} "
                    f"max_anchor_object_offset_error_m={hold_result.max_anchor_object_offset_error_m:.6f} "
                    f"set_target_seen={_yes(hold_result.set_target_seen)} early_kill={_yes(hold_result.early_kill)}",
                    flush=True,
                )
                if not handoff_ok or hold_result.early_kill:
                    break
    else:
        print("[roarm_chain_top_tangent_handoff] local_handoff skipped=YES reason=prep_not_ok", flush=True)

    prep_ok = len(prep_results) == len(pre_move_events) and all(r.reached and not r.early_kill for r in prep_results)
    stationary_results = [r for r in local_results if r.phase == "stationary_hold"]
    micro_plus_results = [r for r in local_results if r.label == "micro_plus_x"]
    micro_return_results = [r for r in local_results if r.label == "micro_return_x"]
    handoff_creation_ok = handoff_ok and joint_created
    stationary_hold_ok = bool(stationary_results) and all(
        r.reached
        and not r.early_kill
        and r.final_target_error_m <= args.target_error_gate_m
        and r.final_anchor_target_error_m <= args.target_error_gate_m
        and r.final_object_target_error_m <= args.target_error_gate_m
        and r.max_tcp_step_m <= args.max_tcp_step_m
        and r.max_sponge_speed_mps <= args.stationary_speed_gate_mps
        and r.max_quat_angle_deg <= args.quat_angle_gate_deg
        and r.min_upright_z >= args.min_upright_z_gate
        for r in stationary_results
    )
    micro_plus_ok = len(micro_plus_results) == 1 and all(
        r.reached
        and not r.early_kill
        and r.final_target_error_m <= args.target_error_gate_m
        and r.final_anchor_target_error_m <= args.target_error_gate_m
        and r.final_object_target_error_m <= args.target_error_gate_m
        and r.max_sponge_speed_mps <= args.micro_speed_gate_mps
        for r in micro_plus_results
    )
    return_ok = len(micro_return_results) == 1 and all(
        r.reached
        and not r.early_kill
        and r.final_target_error_m <= args.target_error_gate_m
        and r.final_anchor_target_error_m <= args.target_error_gate_m
        and r.final_object_target_error_m <= args.target_error_gate_m
        for r in micro_return_results
    )
    handoff_results = [r for r in local_results if r.phase in ("stationary_hold", "micro")]
    relative_transform_ok = bool(handoff_results) and all(
        r.max_tcp_anchor_offset_error_m <= args.tcp_anchor_offset_gate_m
        and r.max_anchor_object_offset_error_m <= args.anchor_object_offset_gate_m
        for r in handoff_results
    )
    upright_ok = bool(handoff_results) and all(
        r.min_upright_z >= args.min_upright_z_gate and r.max_quat_angle_deg <= args.quat_angle_gate_deg
        for r in handoff_results
    )
    no_hidden_posewrite_artifact = (
        attach_stats["attach_calls"] == 0
        and post_handoff_sponge_posewrite_calls == 0
        and all(r.set_target_seen for r in local_results)
    )
    target_error_ok = bool(handoff_results) and max(
        max(r.final_target_error_m, r.final_anchor_target_error_m, r.final_object_target_error_m)
        for r in handoff_results
    ) <= args.target_error_gate_m
    sim_step_ok = bool(local_results) and max(r.max_tcp_step_m for r in local_results) <= args.max_tcp_step_m
    transport_target_no = True
    release_marker_no = True
    surface_gripper_no = True
    success = (
        home_fk_error <= args.home_fk_gate_m
        and prep_ok
        and handoff_creation_ok
        and stationary_hold_ok
        and micro_plus_ok
        and return_ok
        and relative_transform_ok
        and upright_ok
        and no_hidden_posewrite_artifact
        and target_error_ok
        and sim_step_ok
        and transport_target_no
        and release_marker_no
        and surface_gripper_no
        and not episode_done
        and not nan_seen
    )
    all_results = prep_results + local_results
    print(
        f"[roarm_chain_top_tangent_handoff] aggregate total_sim_steps={total_sim_steps} "
        f"prep_events_done={len(prep_results)} prep_events_planned={len(pre_move_events)} "
        f"local_events_done={len(local_results)} local_events_planned={len(local_events) + 1} "
        f"max_final_target_error_m={max((r.final_target_error_m for r in all_results), default=float('inf')):.6f} "
        f"max_final_anchor_target_error_m={max((r.final_anchor_target_error_m for r in handoff_results), default=0.0):.6f} "
        f"max_final_object_target_error_m={max((r.final_object_target_error_m for r in handoff_results), default=0.0):.6f} "
        f"max_tcp_step_m={max((r.max_tcp_step_m for r in all_results), default=float('inf')):.6f} "
        f"max_tcp_anchor_offset_error_m={max((r.max_tcp_anchor_offset_error_m for r in handoff_results), default=0.0):.8f} "
        f"max_anchor_object_offset_error_m={max((r.max_anchor_object_offset_error_m for r in handoff_results), default=0.0):.6f} "
        f"max_sponge_drift_m={max((r.max_sponge_drift_m for r in handoff_results), default=0.0):.6f} "
        f"max_sponge_speed_mps={max((r.max_sponge_speed_mps for r in handoff_results), default=0.0):.6f} "
        f"max_quat_angle_deg={max((r.max_quat_angle_deg for r in handoff_results), default=0.0):.3f} "
        f"min_upright_z={min((r.min_upright_z for r in handoff_results), default=1.0):.6f} "
        f"attach_calls={attach_stats['attach_calls']} posewrite_calls={attach_stats['posewrite_calls']} "
        f"post_handoff_sponge_posewrite_calls={post_handoff_sponge_posewrite_calls} "
        "constraint_prim_insertion=YES fixed_dynamic_constraint_integration=DIAGNOSTIC_LOCAL_ONLY "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "p7_training=NO env_default_edits=NO chain_defaults_edits=NO "
        "claim_transport_success=NO release_physics_validated=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_top_tangent_handoff] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"prep_ok={_yes(prep_ok)} top_tangent_handoff_creation_ok={_yes(handoff_creation_ok)} "
        f"stationary_hold_ok={_yes(stationary_hold_ok)} micro_plus_x_ok={_yes(micro_plus_ok)} "
        f"return_ok={_yes(return_ok)} relative_tcp_anchor_object_transform_ok={_yes(relative_transform_ok)} "
        f"upright_preservation_ok={_yes(upright_ok)} "
        f"no_hidden_kinematic_posewrite_artifact={_yes(no_hidden_posewrite_artifact)} "
        f"target_error_ok={_yes(target_error_ok)} sim_step_ok={_yes(sim_step_ok)} "
        f"transport_target={_yes(not transport_target_no)} release_marker={_yes(not release_marker_no)} "
        f"surface_gripper={_yes(not surface_gripper_no)} "
        "attach_physics_validated=DIAGNOSTIC_LOCAL_CONSTRAINT_ONLY "
        "claim_transport_success=NO release_physics_validated=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(
        f"[roarm_chain_top_tangent_handoff] ROARM_TOP_TANGENT_LOCAL_CONSTRAINT_HANDOFF_SUCCESS={_yes(success)}",
        flush=True,
    )
    env.close()
    sim_app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if success else 2)


if __name__ == "__main__":
    raise SystemExit(main())
