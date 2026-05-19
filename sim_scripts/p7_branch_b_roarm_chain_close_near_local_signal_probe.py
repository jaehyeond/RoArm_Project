#!/usr/bin/env python3
"""Close-near local signal probe for P7 Branch B.

This is a narrow diagnostic for the next Branch B gate. It asks whether the real
RoArm articulation can realize a tiny CLOSE-near local TCP signal under admissible
pre-close geometry, while a dynamic-anchor-style TCP->anchor mapping remains
well-defined. The carrier is virtual in this probe: no constraint prim is
inserted, no SurfaceGripper is attached, no transport target is visited, no
release is executed, and no attach success is claimed.
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
    PICK_WRIST_R_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)
from roarm_kinematics import clip_joints, ik_dls  # noqa: E402

SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022
SPONGE_HEIGHT_EDGE = 0.047


@dataclass(frozen=True)
class LocalEvent:
    index: int
    label: str
    q_deg: np.ndarray
    target_tcp: np.ndarray
    role: str


@dataclass
class RunResult:
    label: str
    reached: bool
    steps: int
    final_target_error_m: float
    max_tcp_step_m: float
    max_virtual_anchor_target_error_m: float
    max_tcp_anchor_offset_error_m: float
    max_sponge_drift_m: float
    max_sponge_speed_mps: float
    max_quat_angle_deg: float
    min_upright_z: float
    set_target_seen: bool
    max_set_diff_rad: float
    early_kill: bool


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _fmt_xyz(v: np.ndarray) -> str:
    return f"([{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}])"


def _fmt_deg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _fmt_quat(v: np.ndarray) -> str:
    q = np.asarray(v, dtype=np.float64)
    return f"[w={q[0]:+.6f}, x={q[1]:+.6f}, y={q[2]:+.6f}, z={q[3]:+.6f}]"


def _mm_tag(value_m: float) -> str:
    text = f"{value_m * 1000.0:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p") + "mm"


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa = qa / max(_norm(qa), 1.0e-12)
    qb = qb / max(_norm(qb), 1.0e-12)
    dot = max(-1.0, min(1.0, float(abs(np.dot(qa, qb)))))
    return math.degrees(2.0 * math.acos(dot))


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _sponge_pose_metrics(pos: np.ndarray, quat: np.ndarray) -> dict[str, float]:
    rot = _quat_wxyz_to_rot(quat)
    half_extents = np.array([SPONGE_LEN_LONG / 2.0, SPONGE_WIDTH / 2.0, SPONGE_HEIGHT_EDGE / 2.0])
    oriented_half_height = float(np.dot(np.abs(rot[2, :]), half_extents))
    up_z = float(rot[2, 2])
    return {
        "up_z": up_z,
        "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, up_z)))),
        "oriented_top_z_m": float(pos[2] + oriented_half_height),
    }


def _target_geometry(target_tcp: np.ndarray, sponge_pos: np.ndarray, sponge_quat: np.ndarray) -> dict[str, float | bool | str]:
    pose = _sponge_pose_metrics(sponge_pos, sponge_quat)
    target_minus_top = float(target_tcp[2] - pose["oriented_top_z_m"])
    if target_minus_top > 0.001:
        top_class = "above"
    elif target_minus_top < -0.001:
        top_class = "below"
    else:
        top_class = "tangent"
    dx = float(target_tcp[0] - sponge_pos[0])
    dy = float(target_tcp[1] - sponge_pos[1])
    inside = abs(dx) <= SPONGE_LEN_LONG / 2.0 and abs(dy) <= SPONGE_WIDTH / 2.0
    return {
        "target_tcp_minus_sponge_oriented_top_m": target_minus_top,
        "target_top_class": top_class,
        "target_dx_sponge_m": dx,
        "target_dy_sponge_m": dy,
        "target_xy_inside_sponge_aabb": inside,
        "sponge_up_z": float(pose["up_z"]),
        "sponge_tilt_deg": float(pose["tilt_deg"]),
    }


def _ik_pose(target_tcp: np.ndarray, seed_q: np.ndarray, gripper_deg: float) -> tuple[np.ndarray, bool, float, int]:
    q, converged, err_mm, n_iter = ik_dls(target_tcp, seed_q, max_iter=240, tol_mm=0.75)
    q = clip_joints(q)
    q[4] = PICK_WRIST_R_DEG
    q[5] = gripper_deg
    return q, bool(converged), float(err_mm), int(n_iter)


def _build_safe_events(args: argparse.Namespace, base_q: np.ndarray, gripper_deg: float) -> tuple[list[LocalEvent], dict[str, str]]:
    top_z = args.reassert_sponge_z_m + SPONGE_HEIGHT_EDGE / 2.0
    base_tcp = fk_tcp(base_q)
    q_seed = base_q.copy()
    q_seed[5] = gripper_deg

    records: dict[str, str] = {}
    clear_tcp = np.array([base_tcp[0], base_tcp[1], top_z + args.clearance_margin_m], dtype=np.float64)
    q_clear, clear_conv, clear_err_mm, clear_iter = _ik_pose(clear_tcp, q_seed, gripper_deg)

    if args.geometry == "top_tangent":
        final_tcp = np.array([base_tcp[0], base_tcp[1], top_z + args.top_margin_m], dtype=np.float64)
        label = f"top_tangent_margin_{_mm_tag(args.top_margin_m)}"
        role = "above_or_tangent_inside_footprint"
    elif args.geometry == "above_top":
        final_tcp = np.array([base_tcp[0], base_tcp[1], top_z + args.above_margin_m], dtype=np.float64)
        label = f"above_top_margin_{_mm_tag(args.above_margin_m)}"
        role = "above_inside_footprint"
    elif args.geometry == "side_edge":
        side_y = float(args.sponge_xy[1] + SPONGE_WIDTH / 2.0 + args.side_margin_m)
        clear_tcp = np.array([base_tcp[0], side_y, top_z + args.clearance_margin_m], dtype=np.float64)
        q_clear, clear_conv, clear_err_mm, clear_iter = _ik_pose(clear_tcp, q_seed, gripper_deg)
        final_tcp = np.array([base_tcp[0], side_y, top_z + args.side_top_margin_m], dtype=np.float64)
        label = f"side_edge_margin_{_mm_tag(args.side_margin_m)}_top_{_mm_tag(args.side_top_margin_m)}"
        role = "conservative_side_edge_outside_aabb"
    else:
        raise RuntimeError(f"unexpected geometry={args.geometry!r}")

    q_final, final_conv, final_err_mm, final_iter = _ik_pose(final_tcp, q_clear, gripper_deg)
    records["ik"] = (
        f"clear={_yes(clear_conv)}/{clear_err_mm:.3f}mm/{clear_iter}iter "
        f"final={_yes(final_conv)}/{final_err_mm:.3f}mm/{final_iter}iter"
    )

    base_for_micro = final_tcp.copy()
    q_prev = q_final.copy()
    micro_targets = [
        ("micro_plus_x", base_for_micro + np.array([args.micro_delta_m, 0.0, 0.0], dtype=np.float64)),
        ("micro_return_x", base_for_micro.copy()),
    ]
    out = [
        LocalEvent(1, f"{label}_clearance", q_clear.copy(), clear_tcp.copy(), "safe_clearance"),
        LocalEvent(2, f"{label}_stationary_signal_pose", q_final.copy(), final_tcp.copy(), role),
    ]
    for idx, (micro_label, tcp) in enumerate(micro_targets, start=3):
        q_micro, conv, err_mm, n_iter = _ik_pose(tcp, q_prev, gripper_deg)
        records[micro_label] = f"ik={_yes(conv)}/{err_mm:.3f}mm/{n_iter}iter target_tcp={_fmt_xyz(tcp)}"
        out.append(LocalEvent(idx, micro_label, q_micro.copy(), tcp.copy(), "tiny_local_micro_motion"))
        q_prev = q_micro
    return out, records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--reassert_sponge_z_m", type=float, default=0.0235)
    ap.add_argument("--geometry", choices=["top_tangent", "above_top", "side_edge"], default="top_tangent")
    ap.add_argument("--signal_stage", choices=["just_before_close", "post_close_marker"], default="just_before_close")
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--clearance_margin_m", type=float, default=0.024)
    ap.add_argument("--top_margin_m", type=float, default=0.0005)
    ap.add_argument("--above_margin_m", type=float, default=0.0010)
    ap.add_argument("--side_margin_m", type=float, default=0.0020)
    ap.add_argument("--side_top_margin_m", type=float, default=-0.0030)
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
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    args = ap.parse_args()

    if args.geometry == "side_edge":
        if args.side_margin_m < 0.0020:
            raise ValueError("side_edge geometry must keep side_margin_m >= 0.0020 for this conservative diagnostic")
        if args.side_top_margin_m < -0.0030:
            raise ValueError("side_edge geometry must keep side_top_margin_m >= -0.0030 for this conservative diagnostic")
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
    gripper_deg = GRIPPER_OPEN_DEG if args.signal_stage == "just_before_close" else GRIPPER_LATCH_DEG
    local_events, ik_records = _build_safe_events(args, base_q, gripper_deg)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from roarm_rl.roarm_stack_env import _quat_rotate

    print("[roarm_chain_close_near_signal] RoArm close-near local signal probe", flush=True)
    print(
        "[roarm_chain_close_near_signal] "
        "close_near_local_signal_only=YES virtual_dynamic_anchor_style_carrier=YES "
        "virtual_carrier_only=YES constraint_prim_insertion=NO fixed_dynamic_constraint_integration=NO "
        "surface_gripper=NO surface_gripper_chain_attachment=NO attached_transport=NO "
        "transport_target=NO release_marker=NO scripted_release_variant=NO p7_training=NO "
        "p7_tuning=NO diagnostic_gate_tuning=NO env_default_edits=NO chain_defaults_edits=NO "
        "attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_close_near_signal] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"preclose_drift_gate_m={args.preclose_drift_gate_m:.6f} "
        f"stationary_speed_gate_mps={args.stationary_speed_gate_mps:.6f} "
        f"micro_speed_gate_mps={args.micro_speed_gate_mps:.6f} "
        f"quat_angle_gate_deg={args.quat_angle_gate_deg:.2f} min_upright_z_gate={args.min_upright_z_gate:.3f} "
        f"tcp_anchor_offset_gate_m={args.tcp_anchor_offset_gate_m:.8f} "
        f"geometry={args.geometry} signal_stage={args.signal_stage} micro_delta_m={args.micro_delta_m:.6f}",
        flush=True,
    )
    print(
        f"[roarm_chain_close_near_signal] stream source_events_total={meta['events_total']} "
        f"pre_move_cmds={meta['pre_move_cmds']} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )
    for key, value in ik_records.items():
        print(f"[roarm_chain_close_near_signal] ik_record {key} {value}", flush=True)

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

    total_sim_steps = 0
    episode_done = False
    nan_seen = False
    for _ in range(args.initial_settle_steps):
        episode_done |= step_once()
        total_sim_steps += 1

    settled_tcp = fresh_tcp_local()
    settled_sponge = sponge_local()
    settled_quat = sponge_quat()
    home_fk_error = _norm(settled_tcp - fk_tcp(HOME_DEG))
    print(
        f"[roarm_chain_close_near_signal] initial home_fresh_tcp={_fmt_xyz(settled_tcp)} "
        f"home_expected_tcp={_fmt_xyz(fk_tcp(HOME_DEG))} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"settled_sponge_pos={_fmt_xyz(settled_sponge)} settled_sponge_quat_wxyz={_fmt_quat(settled_quat)} "
        f"settled_upright_z={sponge_upright_z():.6f}",
        flush=True,
    )

    def virtual_anchor(tcp: np.ndarray) -> np.ndarray:
        return tcp + np.asarray(args.tcp_to_anchor_offset, dtype=np.float64)

    def run_to_local_event(event: LocalEvent, max_steps: int, phase: str) -> RunResult:
        nonlocal total_sim_steps, episode_done, nan_seen
        target_rad_np = np.radians(event.q_deg)
        target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
        target_anchor = virtual_anchor(event.target_tcp)
        ref_offset = np.asarray(args.tcp_to_anchor_offset, dtype=np.float64)
        start_sponge = sponge_local()
        start_quat = sponge_quat()
        prev_tcp = fresh_tcp_local()
        settle_count = 0
        steps_used = 0
        reached = False
        early_kill = False
        max_tcp_step = 0.0
        max_anchor_target_error = 0.0
        max_offset_error = 0.0
        max_sponge_drift = 0.0
        max_sponge_speed = 0.0
        max_quat_angle = 0.0
        min_upright = sponge_upright_z()
        final_error = float("inf")
        watch["active"] = True
        watch["target"] = target_rad_np
        watch["calls"] = 0
        watch["max_diff"] = 0.0
        for step_idx in range(1, max_steps + 1):
            base_env.robot_dof_targets[:] = target_rad
            done = step_once()
            total_sim_steps += 1
            steps_used = step_idx
            tcp = fresh_tcp_local()
            sponge = sponge_local()
            quat = sponge_quat()
            vel = sponge_vel6()
            anchor = virtual_anchor(tcp)
            target_error = _norm(tcp - event.target_tcp)
            final_error = target_error
            tcp_step = _norm(tcp - prev_tcp)
            anchor_target_error = _norm(anchor - target_anchor)
            offset_error = _norm((anchor - tcp) - ref_offset)
            sponge_drift = _norm(sponge - start_sponge)
            sponge_speed = _norm(vel[0:3])
            quat_angle = _quat_angle_deg(quat, start_quat)
            upright = sponge_upright_z()
            max_tcp_step = max(max_tcp_step, tcp_step)
            max_anchor_target_error = max(max_anchor_target_error, anchor_target_error)
            max_offset_error = max(max_offset_error, offset_error)
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
                or offset_error > args.tcp_anchor_offset_gate_m
                or (phase == "stationary_hold" and target_error > args.target_error_gate_m)
                or done
                or nan_seen
            )
            reached = target_error <= args.target_error_gate_m
            settle_count = settle_count + 1 if reached else 0
            if step_idx <= 3 or reached or early_kill or step_idx == max_steps:
                geom = _target_geometry(event.target_tcp, sponge, quat)
                print(
                    f"[roarm_chain_close_near_signal] local_event label={event.label} phase={phase} "
                    f"role={event.role} step={step_idx:03d} target_tcp={_fmt_xyz(event.target_tcp)} "
                    f"fresh_tcp={_fmt_xyz(tcp)} target_error_m={target_error:.6f} "
                    f"tcp_step_m={tcp_step:.6f} virtual_anchor={_fmt_xyz(anchor)} "
                    f"virtual_anchor_target_error_m={anchor_target_error:.6f} "
                    f"tcp_anchor_offset_error_m={offset_error:.8f} sponge_drift_m={sponge_drift:.6f} "
                    f"sponge_speed_mps={sponge_speed:.6f} quat_angle_deg={quat_angle:.3f} "
                    f"upright_z={upright:.6f} target_top_class={geom['target_top_class']} "
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
        if not reached and not early_kill:
            early_kill = True
        return RunResult(
            label=event.label,
            reached=reached,
            steps=steps_used,
            final_target_error_m=final_error,
            max_tcp_step_m=max_tcp_step,
            max_virtual_anchor_target_error_m=max_anchor_target_error,
            max_tcp_anchor_offset_error_m=max_offset_error,
            max_sponge_drift_m=max_sponge_drift,
            max_sponge_speed_mps=max_sponge_speed,
            max_quat_angle_deg=max_quat_angle,
            min_upright_z=min_upright,
            set_target_seen=watch["calls"] > 0 and watch["max_diff"] <= 1.0e-5,
            max_set_diff_rad=float(watch["max_diff"]),
            early_kill=early_kill,
        )

    prep_results: list[RunResult] = []
    for event in pre_move_events:
        result = run_to_local_event(
            LocalEvent(event.index, f"prep_{event.index:03d}_{event.segment}", event.q_deg, event.target_tcp, "source_pre_move"),
            args.steps_per_local_event,
            "prep",
        )
        prep_results.append(result)
        if event.index <= 5 or event.index % args.log_every_event == 0 or not result.reached:
            print(
                f"[roarm_chain_close_near_signal] prep_result index={event.index:03d} "
                f"label={result.label} reached={_yes(result.reached)} "
                f"final_target_error_m={result.final_target_error_m:.6f} max_tcp_step_m={result.max_tcp_step_m:.6f} "
                f"set_target_seen={_yes(result.set_target_seen)} early_kill={_yes(result.early_kill)}",
                flush=True,
            )
        if not result.reached or result.early_kill:
            break

    close_marker_results: list[RunResult] = []
    close_marker_ok = True
    if args.signal_stage == "post_close_marker" and prep_results and prep_results[-1].reached:
        close_q = base_q.copy()
        close_q[5] = GRIPPER_LATCH_DEG
        close_event = LocalEvent(999, "close_marker_at_nominal_grasp_q", close_q, fk_tcp(close_q), "close_marker_only_no_posewrite")
        close_result = run_to_local_event(close_event, args.steps_per_local_event, "close_marker")
        close_marker_results.append(close_result)
        close_marker_ok = close_result.reached and not close_result.early_kill
        print(
            f"[roarm_chain_close_near_signal] close_marker_result reached={_yes(close_result.reached)} "
            f"final_target_error_m={close_result.final_target_error_m:.6f} "
            f"attach_calls={attach_stats['attach_calls']} posewrite_calls={attach_stats['posewrite_calls']} "
            f"claim_attach_success=NO",
            flush=True,
        )
    elif args.signal_stage == "post_close_marker":
        close_marker_ok = False

    local_results: list[RunResult] = []
    if prep_results and prep_results[-1].reached and not prep_results[-1].early_kill and close_marker_ok:
        for idx, event in enumerate(local_events):
            phase = "stationary_reach" if idx < 2 else "micro"
            max_steps = args.steps_per_local_event
            result = run_to_local_event(event, max_steps, phase)
            local_results.append(result)
            print(
                f"[roarm_chain_close_near_signal] local_result label={result.label} phase={phase} "
                f"reached={_yes(result.reached)} steps={result.steps} "
                f"final_target_error_m={result.final_target_error_m:.6f} "
                f"max_virtual_anchor_target_error_m={result.max_virtual_anchor_target_error_m:.6f} "
                f"max_tcp_anchor_offset_error_m={result.max_tcp_anchor_offset_error_m:.8f} "
                f"set_target_seen={_yes(result.set_target_seen)} early_kill={_yes(result.early_kill)}",
                flush=True,
            )
            if result.early_kill:
                break
            if "stationary_signal_pose" in event.label:
                hold_event = LocalEvent(
                    event.index,
                    f"{event.label}_hold",
                    event.q_deg.copy(),
                    event.target_tcp.copy(),
                    "stationary_hold_after_safe_signal_pose",
                )
                hold_result = run_to_local_event(hold_event, args.stationary_hold_steps, "stationary_hold")
                local_results.append(hold_result)
                print(
                    f"[roarm_chain_close_near_signal] local_result label={hold_result.label} "
                    f"phase=stationary_hold reached={_yes(hold_result.reached)} steps={hold_result.steps} "
                    f"final_target_error_m={hold_result.final_target_error_m:.6f} "
                    f"max_virtual_anchor_target_error_m={hold_result.max_virtual_anchor_target_error_m:.6f} "
                    f"max_tcp_anchor_offset_error_m={hold_result.max_tcp_anchor_offset_error_m:.8f} "
                    f"set_target_seen={_yes(hold_result.set_target_seen)} early_kill={_yes(hold_result.early_kill)}",
                    flush=True,
                )
                if hold_result.early_kill:
                    break
    else:
        print("[roarm_chain_close_near_signal] local_signal skipped=YES reason=prep_or_close_marker_not_ok", flush=True)

    prep_ok = len(prep_results) == len(pre_move_events) and all(r.reached and not r.early_kill for r in prep_results)
    stationary_results = [r for r in local_results if r.label.endswith("_hold")]
    micro_results = [r for r in local_results if "micro_" in r.label]
    stationary_hold_ok = bool(stationary_results) and all(
        r.reached
        and not r.early_kill
        and r.final_target_error_m <= args.target_error_gate_m
        and r.max_tcp_step_m <= args.max_tcp_step_m
        and r.max_sponge_speed_mps <= args.stationary_speed_gate_mps
        and r.max_quat_angle_deg <= args.quat_angle_gate_deg
        and r.min_upright_z >= args.min_upright_z_gate
        for r in stationary_results
    )
    micro_motion_ok = len(micro_results) == 2 and all(
        r.reached
        and not r.early_kill
        and r.final_target_error_m <= args.target_error_gate_m
        and r.max_tcp_step_m <= args.max_tcp_step_m
        and r.max_sponge_speed_mps <= args.micro_speed_gate_mps
        for r in micro_results
    )
    relative_transform_ok = bool(local_results) and all(
        r.max_tcp_anchor_offset_error_m <= args.tcp_anchor_offset_gate_m for r in local_results
    )
    upright_ok = bool(local_results) and all(
        r.min_upright_z >= args.min_upright_z_gate and r.max_quat_angle_deg <= args.quat_angle_gate_deg for r in local_results
    )
    no_hidden_posewrite_artifact = attach_stats["posewrite_calls"] == 0 and all(r.set_target_seen for r in local_results)
    no_overclaim = True
    target_error_ok = bool(local_results) and max(r.final_target_error_m for r in local_results) <= args.target_error_gate_m
    sim_step_ok = bool(local_results) and max(r.max_tcp_step_m for r in local_results) <= args.max_tcp_step_m
    success = (
        home_fk_error <= args.home_fk_gate_m
        and prep_ok
        and close_marker_ok
        and stationary_hold_ok
        and micro_motion_ok
        and relative_transform_ok
        and upright_ok
        and no_hidden_posewrite_artifact
        and no_overclaim
        and target_error_ok
        and sim_step_ok
        and not episode_done
        and not nan_seen
    )
    all_results = prep_results + close_marker_results + local_results
    print(
        f"[roarm_chain_close_near_signal] aggregate total_sim_steps={total_sim_steps} "
        f"prep_events_done={len(prep_results)} prep_events_planned={len(pre_move_events)} "
        f"local_events_done={len(local_results)} local_events_planned={len(local_events)} "
        f"max_final_target_error_m={max((r.final_target_error_m for r in all_results), default=float('inf')):.6f} "
        f"max_tcp_step_m={max((r.max_tcp_step_m for r in all_results), default=float('inf')):.6f} "
        f"max_virtual_anchor_target_error_m={max((r.max_virtual_anchor_target_error_m for r in local_results), default=0.0):.6f} "
        f"max_tcp_anchor_offset_error_m={max((r.max_tcp_anchor_offset_error_m for r in local_results), default=0.0):.8f} "
        f"max_sponge_drift_m={max((r.max_sponge_drift_m for r in all_results), default=0.0):.6f} "
        f"max_sponge_speed_mps={max((r.max_sponge_speed_mps for r in all_results), default=0.0):.6f} "
        f"max_quat_angle_deg={max((r.max_quat_angle_deg for r in all_results), default=0.0):.3f} "
        f"min_upright_z={min((r.min_upright_z for r in all_results), default=1.0):.6f} "
        f"attach_calls={attach_stats['attach_calls']} posewrite_calls={attach_stats['posewrite_calls']} "
        f"virtual_carrier_only=YES transport_target=NO release_marker=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_close_near_signal] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"prep_ok={_yes(prep_ok)} close_marker_ok={_yes(close_marker_ok)} stationary_hold_ok={_yes(stationary_hold_ok)} "
        f"micro_motion_realized_ok={_yes(micro_motion_ok)} "
        f"relative_tcp_anchor_transform_ok={_yes(relative_transform_ok)} "
        f"upright_preservation_ok={_yes(upright_ok)} "
        f"no_hidden_kinematic_posewrite_artifact={_yes(no_hidden_posewrite_artifact)} "
        f"no_attach_release_transport_overclaim={_yes(no_overclaim)} "
        f"target_error_ok={_yes(target_error_ok)} sim_step_ok={_yes(sim_step_ok)} "
        f"attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_close_near_signal] ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS={_yes(success)}", flush=True)
    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
