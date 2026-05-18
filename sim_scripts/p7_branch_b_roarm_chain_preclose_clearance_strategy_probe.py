#!/usr/bin/env python3
"""Diagnostic-only pre-close clearance/grasp-posture strategy probe.

This stays before Branch-B integration. It does not train, insert constraints,
attach SurfaceGripper, execute transport, execute release, or edit env/train/
chain defaults. The probe compares the known nominal below-top top-clamp
baseline against mechanically safer pre-close targets that stay above or tangent
to the sponge top, plus a far-sponge control.
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
    GRIPPER_OPEN_DEG,
    HOME_DEG,
    PICK_WRIST_R_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)
from p7_branch_b_roarm_chain_handoff_micro_motion_probe import _fmt_xyz, _norm, _yes  # noqa: E402
from roarm_kinematics import JOINT_LIMITS_DEG, clip_joints, ik_dls  # noqa: E402


def _fmt_deg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _fmt_rad(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.6f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _fmt_quat(v: np.ndarray) -> str:
    q = np.asarray(v, dtype=np.float64)
    return f"[w={q[0]:+.6f}, x={q[1]:+.6f}, y={q[2]:+.6f}, z={q[3]:+.6f}]"


def _as_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy().astype(np.float64)
    return np.asarray(v, dtype=np.float64)


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _analytic_limits_ok(q_deg: np.ndarray) -> bool:
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    return all(JOINT_LIMITS_DEG[name][0] - 1.0e-6 <= q_deg[i] <= JOINT_LIMITS_DEG[name][1] + 1.0e-6 for i, name in enumerate(names))


def _soft_limit_ok(q_deg: np.ndarray, lower_rad: np.ndarray, upper_rad: np.ndarray) -> tuple[bool, float, float]:
    q_rad = np.radians(q_deg)
    lower_margin = q_rad - lower_rad
    upper_margin = upper_rad - q_rad
    return (
        bool(np.all(lower_margin >= -1.0e-5) and np.all(upper_margin >= -1.0e-5)),
        float(np.min(lower_margin)),
        float(np.min(upper_margin)),
    )


@dataclass(frozen=True)
class Segment:
    name: str
    q_deg: np.ndarray
    note: str


@dataclass(frozen=True)
class Strategy:
    name: str
    sponge_xy: tuple[float, float]
    segments: tuple[Segment, ...]
    hypothesis: str


def _q_with_shoulder(base_q: np.ndarray, delta_deg: float) -> np.ndarray:
    q = base_q.copy()
    q[1] += delta_deg
    q[4] = PICK_WRIST_R_DEG
    q[5] = GRIPPER_OPEN_DEG
    return q


def _ik_pose(target_tcp: np.ndarray, seed_q: np.ndarray) -> tuple[np.ndarray, bool, float, int]:
    q, converged, err_mm, n_iter = ik_dls(target_tcp, seed_q, max_iter=240, tol_mm=0.75)
    q = clip_joints(q)
    q[4] = PICK_WRIST_R_DEG
    q[5] = GRIPPER_OPEN_DEG
    return q, bool(converged), float(err_mm), int(n_iter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--far_sponge_xy", nargs=2, type=float, default=[0.80, 0.40])
    ap.add_argument("--reassert_sponge_z_m", type=float, default=0.0235)
    ap.add_argument("--top_tangent_margin_m", type=float, default=0.0005)
    ap.add_argument("--above_margin_m", type=float, default=0.0120)
    ap.add_argument("--clearance_margin_m", type=float, default=0.0240)
    ap.add_argument("--side_margin_m", type=float, default=0.0120)
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--segment_steps", type=int, default=45)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    args = ap.parse_args()

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    pre_move_events = [event for event in events if event.kind == "PRE_MOVE"]
    base_q = pre_move_events[-1].q_deg.copy()
    base_q[5] = GRIPPER_OPEN_DEG
    base_tcp = fk_tcp(base_q)
    top_z = float(args.reassert_sponge_z_m + 0.047 / 2.0)

    q_up = _q_with_shoulder(base_q, -5.0)
    q_above = _q_with_shoulder(base_q, -2.5)
    q_tangent = base_q.copy()
    q_tangent[4] = PICK_WRIST_R_DEG
    q_tangent[5] = GRIPPER_OPEN_DEG
    q_below_2p5 = _q_with_shoulder(base_q, +2.5)
    q_below_5 = _q_with_shoulder(base_q, +5.0)

    side_y = float(args.sponge_xy[1] + 0.047 * 0.0 + 0.022 / 2.0 + args.side_margin_m)
    side_clear_tcp = np.array([base_tcp[0], side_y, top_z + args.clearance_margin_m], dtype=np.float64)
    side_tangent_tcp = np.array([base_tcp[0], side_y, top_z + args.top_tangent_margin_m], dtype=np.float64)
    q_side_clear, side_clear_conv, side_clear_err_mm, side_clear_iter = _ik_pose(side_clear_tcp, q_up)
    q_side_tangent, side_tangent_conv, side_tangent_err_mm, side_tangent_iter = _ik_pose(side_tangent_tcp, q_side_clear)

    strategies = [
        Strategy(
            name="baseline_nominal_below_top_plus5deg",
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            segments=(Segment("below_top_plus5deg", q_below_5, "known top-clamp baseline: target is below top inside footprint"),),
            hypothesis="below-top command should clamp near top, not count as command convergence",
        ),
        Strategy(
            name="far_sponge_below_top_plus5deg_control",
            sponge_xy=(args.far_sponge_xy[0], args.far_sponge_xy[1]),
            segments=(Segment("below_top_plus5deg_far_sponge", q_below_5, "no-contact/far-sponge control for same q target"),),
            hypothesis="if this realizes while nominal fails, contact/proximity remains primary",
        ),
        Strategy(
            name="upward_first_then_above_top",
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            segments=(
                Segment("clearance_up_minus5deg", q_up, "move upward to clearance"),
                Segment("down_to_above_top_minus2p5deg", q_above, "downward approach remains above top"),
            ),
            hypothesis="above-top downward staging should realize without top clamp",
        ),
        Strategy(
            name="upward_first_then_top_tangent",
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            segments=(
                Segment("clearance_up_minus5deg", q_up, "move upward to clearance"),
                Segment("down_to_top_tangent_nominal", q_tangent, "downward approach stops at top-tangent nominal pre-close"),
            ),
            hypothesis="tangent pre-close should avoid through-top command",
        ),
        Strategy(
            name="upward_first_then_below_top_kill_control",
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            segments=(
                Segment("clearance_up_minus5deg", q_up, "move upward to clearance"),
                Segment("down_to_below_top_plus2p5deg", q_below_2p5, "negative control: later command crosses below top"),
            ),
            hypothesis="if this reclamps, upward-first alone is insufficient for below-top targets",
        ),
        Strategy(
            name="side_edge_tangent_approach",
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            segments=(
                Segment("side_edge_clearance", q_side_clear, "IK side/edge clearance outside sponge footprint"),
                Segment("side_edge_tangent", q_side_tangent, "IK side/edge tangent target outside sponge footprint"),
            ),
            hypothesis="outside-footprint tangent approach should realize if top contact is the blocker",
        ),
    ]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401 registers env
    import torch
    from roarm_rl.roarm_stack_env import (
        SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z,
        SPONGE_HEIGHT_EDGE,
        SPONGE_LEN_LONG,
        SPONGE_WIDTH,
        RoArmStackEnvCfg,
        _quat_rotate,
    )

    print("[roarm_chain_preclose_clearance] preclose_clearance_strategy_probe", flush=True)
    print(
        "[roarm_chain_preclose_clearance] "
        "diagnostic_preclose_only=YES constraint_prim_insertion=NO fixed_dynamic_constraint_integration=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "scripted_release_variant=NO p7_training=NO p7_tuning=NO diagnostic_gate_tuning=NO "
        "env_default_edits=NO chain_defaults_edits=NO attach_physics_validated=NO "
        "release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} segment_steps={args.segment_steps} "
        f"settle_steps={args.settle_steps} reassert_sponge_z_m={args.reassert_sponge_z_m:.6f} "
        f"nominal_top_z_m={top_z:.6f} reduction_gate_reference_only=YES",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] stream source_events_total={meta['events_total']} "
        f"executed_pre_moves={len(pre_move_events)} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] side_ik side_clear_converged={_yes(side_clear_conv)} "
        f"side_clear_err_mm={side_clear_err_mm:.3f} side_clear_iter={side_clear_iter} "
        f"side_tangent_converged={_yes(side_tangent_conv)} side_tangent_err_mm={side_tangent_err_mm:.3f} "
        f"side_tangent_iter={side_tangent_iter} side_clear_tcp={_fmt_xyz(side_clear_tcp)} "
        f"side_tangent_tcp={_fmt_xyz(side_tangent_tcp)}",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] contact_sensor_available=NO "
        f"proximity_metrics=distance_aabb_top_proxy sponge_size_m=({SPONGE_LEN_LONG:.6f},{SPONGE_WIDTH:.6f},{SPONGE_HEIGHT_EDGE:.6f})",
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

    attach_stats = {"attach_calls": 0}

    def _marker_only_attach() -> None:
        env_ids = torch.where(base_env._grasped)[0]
        if len(env_ids) > 0:
            attach_stats["attach_calls"] += 1

    base_env._update_grasp_attach = _marker_only_attach

    watch = {"active": False, "label": "none", "target": None, "calls": 0, "max_diff": 0.0}
    original_set_joint_position_target = base_env._robot.set_joint_position_target

    def _wrapped_set_joint_position_target(target, *set_args, **set_kwargs):
        if watch["active"] and watch["target"] is not None:
            arr = _as_np(target)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            tgt = watch["target"]
            diff = float(np.max(np.abs(arr[0, : tgt.shape[0]] - tgt)))
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], diff)
            if watch["calls"] <= 2:
                print(
                    f"[roarm_chain_preclose_clearance] set_joint_position_target_call "
                    f"label={watch['label']} call={watch['calls']} max_diff_to_watch_rad={diff:.8f} "
                    f"target_rad={_fmt_rad(arr[0])}",
                    flush=True,
                )
        return original_set_joint_position_target(target, *set_args, **set_kwargs)

    base_env._robot.set_joint_position_target = _wrapped_set_joint_position_target

    total_sim_steps = 0
    nan_seen = False
    episode_done = False

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

    def data_joint_pos_target_rad() -> np.ndarray | None:
        value = getattr(base_env._robot.data, "joint_pos_target", None)
        if value is None or not hasattr(value, "shape"):
            return None
        arr = _as_np(value)
        if arr.size < 6:
            return None
        return arr.reshape(-1, arr.shape[-1])[0, :6].astype(np.float64)

    def data_target_snapshot(label: str, target_rad_np: np.ndarray) -> float:
        names: list[str] = []
        best = float("inf")
        for name in dir(base_env._robot.data):
            if "target" not in name.lower():
                continue
            try:
                value = getattr(base_env._robot.data, name)
            except Exception:
                continue
            if not hasattr(value, "shape"):
                continue
            arr = _as_np(value)
            if arr.size < target_rad_np.size:
                continue
            names.append(name)
            flat = arr.reshape(-1, arr.shape[-1])
            if flat.shape[-1] >= target_rad_np.size:
                best = min(best, float(np.max(np.abs(flat[0, : target_rad_np.size] - target_rad_np))))
        best_text = "nan" if not math.isfinite(best) else f"{best:.8f}"
        print(
            f"[roarm_chain_preclose_clearance] data_target_snapshot label={label} "
            f"target_attrs={names} best_attr_diff_rad={best_text}",
            flush=True,
        )
        return best

    def sponge_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def sponge_quat_wxyz() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_speed() -> float:
        return float(torch.linalg.norm(base_env._sponge.data.root_vel_w[0]).detach().cpu().item())

    def sponge_pose_metrics(pos: np.ndarray, quat: np.ndarray) -> dict[str, float]:
        rot = _quat_wxyz_to_rot(quat)
        half_extents = np.array([SPONGE_LEN_LONG / 2.0, SPONGE_WIDTH / 2.0, SPONGE_HEIGHT_EDGE / 2.0], dtype=np.float64)
        oriented_half_height = float(np.dot(np.abs(rot[2, :]), half_extents))
        up_z = float(rot[2, 2])
        tilt_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, up_z)))))
        return {
            "up_z": up_z,
            "tilt_deg": tilt_deg,
            "axis_x_z_abs": float(abs(rot[2, 0])),
            "axis_y_z_abs": float(abs(rot[2, 1])),
            "axis_z_z_abs": float(abs(rot[2, 2])),
            "upright_top_z_m": float(pos[2] + SPONGE_HEIGHT_EDGE / 2.0),
            "oriented_top_z_m": float(pos[2] + oriented_half_height),
        }

    def proximity_metrics(tcp: np.ndarray, target_tcp: np.ndarray, sponge: np.ndarray, quat: np.ndarray) -> dict[str, float | bool | str]:
        half_long = SPONGE_LEN_LONG / 2.0
        half_width = SPONGE_WIDTH / 2.0
        pose = sponge_pose_metrics(sponge, quat)
        oriented_top = float(pose["oriented_top_z_m"])
        target_minus_top = float(target_tcp[2] - oriented_top)
        final_minus_top = float(tcp[2] - oriented_top)
        if target_minus_top > 0.001:
            top_class = "above"
        elif target_minus_top < -0.001:
            top_class = "below"
        else:
            top_class = "tangent"
        dx = abs(float(tcp[0] - sponge[0]))
        dy = abs(float(tcp[1] - sponge[1]))
        target_dx = abs(float(target_tcp[0] - sponge[0]))
        target_dy = abs(float(target_tcp[1] - sponge[1]))
        return {
            "d_tcp_sponge_m": _norm(tcp - sponge),
            "d_target_tcp_sponge_m": _norm(target_tcp - sponge),
            "target_dx_sponge_m": float(target_tcp[0] - sponge[0]),
            "target_dy_sponge_m": float(target_tcp[1] - sponge[1]),
            "sponge_up_z": float(pose["up_z"]),
            "sponge_tilt_deg": float(pose["tilt_deg"]),
            "sponge_axis_x_z_abs": float(pose["axis_x_z_abs"]),
            "sponge_axis_y_z_abs": float(pose["axis_y_z_abs"]),
            "sponge_axis_z_z_abs": float(pose["axis_z_z_abs"]),
            "sponge_upright_top_z_m": float(pose["upright_top_z_m"]),
            "sponge_oriented_top_z_m": oriented_top,
            "tcp_minus_sponge_oriented_top_m": final_minus_top,
            "target_tcp_minus_sponge_oriented_top_m": target_minus_top,
            "target_top_class": top_class,
            "tcp_xy_inside_sponge_aabb": dx <= half_long and dy <= half_width,
            "target_xy_inside_sponge_aabb": target_dx <= half_long and target_dy <= half_width,
        }

    def reassert_sponge_pose(strategy: Strategy, label: str) -> None:
        pose = torch.tensor(
            [[strategy.sponge_xy[0], strategy.sponge_xy[1], args.reassert_sponge_z_m, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        pose[:, 0:3] += base_env.scene.env_origins[:1]
        base_env._sponge.write_root_pose_to_sim(pose)
        base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        base_env.scene.write_data_to_sim()
        base_env.scene.update(base_env.sim.get_physics_dt())
        pos = sponge_local()
        quat = sponge_quat_wxyz()
        metrics = sponge_pose_metrics(pos, quat)
        print(
            f"[roarm_chain_preclose_clearance] sponge_pose_reassert label={label} "
            f"requested_xyz=({strategy.sponge_xy[0]:+.6f},{strategy.sponge_xy[1]:+.6f},{args.reassert_sponge_z_m:+.6f}) "
            f"actual_xyz={_fmt_xyz(pos)} quat_wxyz={_fmt_quat(quat)} up_z={metrics['up_z']:.6f} "
            f"tilt_deg={metrics['tilt_deg']:.6f} upright_top_z_m={metrics['upright_top_z_m']:.6f} "
            f"oriented_top_z_m={metrics['oriented_top_z_m']:.6f}",
            flush=True,
        )

    def run_to_q(label: str, q_deg: np.ndarray, steps: int, quiet: bool = False) -> tuple[bool, int, float]:
        nonlocal total_sim_steps, episode_done, nan_seen
        target_rad = torch.tensor(np.radians(q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        target_tcp = fk_tcp(q_deg)
        settle_count = 0
        final_error = float("inf")
        steps_used = 0
        for step_idx in range(1, steps + 1):
            base_env.robot_dof_targets[:] = target_rad
            done = step_once()
            total_sim_steps += 1
            episode_done |= done
            steps_used = step_idx
            tcp = fresh_tcp_local()
            final_error = _norm(tcp - target_tcp)
            if not np.isfinite(tcp).all() or not math.isfinite(final_error):
                nan_seen = True
            reached = final_error <= args.target_error_gate_m
            settle_count = settle_count + 1 if reached else 0
            if settle_count >= args.settle_steps:
                break
        if not quiet:
            print(
                f"[roarm_chain_preclose_clearance] restore_or_advance label={label} steps={steps_used} "
                f"final_target_error_m={final_error:.6f} reached={_yes(final_error <= args.target_error_gate_m)} "
                f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
                flush=True,
            )
        return final_error <= args.target_error_gate_m, steps_used, final_error

    def reset_and_prepare(strategy: Strategy) -> tuple[bool, float]:
        nonlocal total_sim_steps, episode_done, nan_seen
        env.reset()
        home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
        base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
        base_env._robot.set_joint_position_target(home_rad)
        base_env.robot_dof_targets[:] = home_rad
        reassert_sponge_pose(strategy, f"{strategy.name}_reset")
        base_env._grasped[:] = False
        base_env._was_grasped[:] = False
        for _ in range(args.initial_settle_steps):
            episode_done |= step_once()
            total_sim_steps += 1
        home_error = _norm(fresh_tcp_local() - fk_tcp(HOME_DEG))
        if not math.isfinite(home_error):
            nan_seen = True
        for event in pre_move_events:
            reached, _steps, _err = run_to_q(f"{strategy.name}_pre_{event.index:03d}", event.q_deg, args.max_steps_per_event, quiet=True)
            if not reached:
                print(
                    f"[roarm_chain_preclose_clearance] prepare_pre_move_failed strategy={strategy.name} "
                    f"event_index={event.index:03d} final_error_m={_err:.6f}",
                    flush=True,
                )
                break
        reached, _steps, base_err = run_to_q(f"{strategy.name}_nominal_start_pose", base_q, args.max_steps_per_event, quiet=False)
        return reached and home_error <= args.target_error_gate_m, base_err

    def run_segment(strategy_name: str, segment: Segment) -> dict[str, float | bool | str]:
        nonlocal total_sim_steps, episode_done, nan_seen
        target_rad_np = np.radians(segment.q_deg)
        target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
        expected_tcp = fk_tcp(segment.q_deg)
        start_tcp = fresh_tcp_local()
        start_q = current_q_rad()
        start_sponge = sponge_local()
        start_quat = sponge_quat_wxyz()
        start_prox = proximity_metrics(start_tcp, expected_tcp, start_sponge, start_quat)
        start_target_error = _norm(start_tcp - expected_tcp)
        start_joint_error_max_deg = float(np.max(np.abs(np.degrees(target_rad_np - start_q))))
        start_shoulder_error_deg = float(abs(np.degrees(target_rad_np[1] - start_q[1])))
        label = f"{strategy_name}_{segment.name}"
        watch["active"] = True
        watch["label"] = label
        watch["target"] = target_rad_np
        watch["calls"] = 0
        watch["max_diff"] = 0.0
        max_realized_tcp_delta = 0.0
        max_step_tcp_delta = 0.0
        max_sponge_drift = 0.0
        max_sponge_speed = 0.0
        max_joint_vel_abs_deg_s = 0.0
        min_joint_error_max_deg = float("inf")
        best_attr_diff = float("inf")
        final_target_tcp_error = float("inf")
        final_shoulder_error_deg = float("inf")
        final_joint_error_max_deg = float("inf")
        prev_tcp = start_tcp
        prev_q = start_q
        for step_idx in range(1, args.segment_steps + 1):
            base_env.robot_dof_targets[:] = target_rad
            done = step_once()
            total_sim_steps += 1
            episode_done |= done
            after_tcp = fresh_tcp_local()
            after_q = current_q_rad()
            after_sponge = sponge_local()
            after_quat = sponge_quat_wxyz()
            physics_dt = float(base_env.sim.get_physics_dt())
            step_tcp_delta = _norm(after_tcp - prev_tcp)
            joint_vel_deg_s = np.degrees((after_q - prev_q) / max(physics_dt, 1.0e-9))
            realized_tcp_delta = _norm(after_tcp - start_tcp)
            target_tcp_error = _norm(after_tcp - expected_tcp)
            joint_error_deg = np.degrees(target_rad_np - after_q)
            joint_error_max_deg = float(np.max(np.abs(joint_error_deg)))
            shoulder_error_deg = float(abs(joint_error_deg[1]))
            elbow_error_deg = float(joint_error_deg[2])
            wrist_p_error_deg = float(joint_error_deg[3])
            wrist_r_error_deg = float(joint_error_deg[4])
            sponge_drift = _norm(after_sponge - start_sponge)
            max_realized_tcp_delta = max(max_realized_tcp_delta, realized_tcp_delta)
            max_step_tcp_delta = max(max_step_tcp_delta, step_tcp_delta)
            max_sponge_drift = max(max_sponge_drift, sponge_drift)
            max_sponge_speed = max(max_sponge_speed, sponge_speed())
            max_joint_vel_abs_deg_s = max(max_joint_vel_abs_deg_s, float(np.max(np.abs(joint_vel_deg_s))))
            min_joint_error_max_deg = min(min_joint_error_max_deg, joint_error_max_deg)
            final_target_tcp_error = target_tcp_error
            final_shoulder_error_deg = shoulder_error_deg
            final_joint_error_max_deg = joint_error_max_deg
            attr_diff = data_target_snapshot(f"{label}_step{step_idx:03d}", target_rad_np)
            best_attr_diff = min(best_attr_diff, attr_diff)
            data_target_rad = data_joint_pos_target_rad()
            data_target_diff_rad = float("nan")
            data_target_deg_text = "NA"
            if data_target_rad is not None:
                data_target_diff_rad = float(np.max(np.abs(data_target_rad - target_rad_np)))
                data_target_deg_text = _fmt_deg(np.degrees(data_target_rad))
            prox = proximity_metrics(after_tcp, expected_tcp, after_sponge, after_quat)
            gripper_q_deg = float(np.degrees(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item()))
            print(
                f"[roarm_chain_preclose_clearance] segment_step label={label} step={step_idx:03d} "
                f"note={segment.note!r} set_calls={watch['calls']} set_max_diff_rad={watch['max_diff']:.8f} "
                f"robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np))):.8f} "
                f"robot_dof_targets_deg={_fmt_deg(np.degrees(targets_rad()))} "
                f"data_joint_pos_target_diff_rad={'nan' if not math.isfinite(data_target_diff_rad) else f'{data_target_diff_rad:.8f}'} "
                f"data_joint_pos_target_deg={data_target_deg_text} current_q_deg={_fmt_deg(np.degrees(after_q))} "
                f"joint_error_deg={_fmt_deg(joint_error_deg)} joint_vel_deg_s={_fmt_deg(joint_vel_deg_s)} "
                f"joint_error_max_deg={joint_error_max_deg:.6f} shoulder_error_deg={shoulder_error_deg:.6f} "
                f"elbow_error_deg={elbow_error_deg:+.6f} wrist_p_error_deg={wrist_p_error_deg:+.6f} "
                f"wrist_r_error_deg={wrist_r_error_deg:+.6f} "
                f"fresh_tcp={_fmt_xyz(after_tcp)} expected_tcp={_fmt_xyz(expected_tcp)} "
                f"tcp_z_m={after_tcp[2]:+.6f} expected_tcp_z_m={expected_tcp[2]:+.6f} "
                f"target_tcp_error_m={target_tcp_error:.6f} "
                f"realized_tcp_delta_m={realized_tcp_delta:.6f} step_tcp_delta_m={step_tcp_delta:.6f} "
                f"sponge_quat_wxyz={_fmt_quat(after_quat)} sponge_up_z={prox['sponge_up_z']:.6f} "
                f"sponge_tilt_deg={prox['sponge_tilt_deg']:.6f} "
                f"sponge_axis_z_abs=({prox['sponge_axis_x_z_abs']:.6f},{prox['sponge_axis_y_z_abs']:.6f},{prox['sponge_axis_z_z_abs']:.6f}) "
                f"sponge_upright_top_z_m={prox['sponge_upright_top_z_m']:.6f} "
                f"sponge_oriented_top_z_m={prox['sponge_oriented_top_z_m']:.6f} "
                f"tcp_minus_sponge_oriented_top_m={prox['tcp_minus_sponge_oriented_top_m']:.6f} "
                f"target_tcp_minus_sponge_oriented_top_m={prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
                f"target_top_class={prox['target_top_class']} "
                f"target_xy_inside_sponge_aabb={_yes(bool(prox['target_xy_inside_sponge_aabb']))} "
                f"tcp_xy_inside_sponge_aabb={_yes(bool(prox['tcp_xy_inside_sponge_aabb']))} "
                f"d_tcp_sponge_m={prox['d_tcp_sponge_m']:.6f} d_target_tcp_sponge_m={prox['d_target_tcp_sponge_m']:.6f} "
                f"target_dx_sponge_m={prox['target_dx_sponge_m']:.6f} target_dy_sponge_m={prox['target_dy_sponge_m']:.6f} "
                f"sponge_drift_m={sponge_drift:.6f} sponge_speed_mps={sponge_speed():.6f} "
                f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))} gripper_q_deg={gripper_q_deg:+.3f} "
                f"done={_yes(done)}",
                flush=True,
            )
            if not np.isfinite(after_tcp).all() or not math.isfinite(target_tcp_error):
                nan_seen = True
            prev_tcp = after_tcp
            prev_q = after_q
        watch["active"] = False
        final_tcp = fresh_tcp_local()
        final_q = current_q_rad()
        final_sponge = sponge_local()
        final_quat = sponge_quat_wxyz()
        final_prox = proximity_metrics(final_tcp, expected_tcp, final_sponge, final_quat)
        exact_converged = final_target_tcp_error <= args.target_error_gate_m
        reduction_gate_would_pass = (
            final_target_tcp_error < max(args.target_error_gate_m, 0.75 * start_target_error)
            and final_shoulder_error_deg < max(1.0, 0.5 * start_shoulder_error_deg)
        )
        top_clamped = (
            str(final_prox["target_top_class"]) == "below"
            and abs(float(final_prox["tcp_minus_sponge_oriented_top_m"])) <= 0.0015
            and bool(final_prox["tcp_xy_inside_sponge_aabb"])
        )
        mechanically_valid_target = str(final_prox["target_top_class"]) in ("above", "tangent") or not bool(final_prox["target_xy_inside_sponge_aabb"])
        clean_realized = exact_converged and mechanically_valid_target and not top_clamped
        print(
            f"[roarm_chain_preclose_clearance] segment_result label={label} "
            f"start_q_deg={_fmt_deg(np.degrees(start_q))} target_q_deg={_fmt_deg(np.degrees(target_rad_np))} "
            f"final_q_deg={_fmt_deg(np.degrees(final_q))} expected_tcp={_fmt_xyz(expected_tcp)} final_tcp={_fmt_xyz(final_tcp)} "
            f"start_target_error_m={start_target_error:.6f} final_target_tcp_error_m={final_target_tcp_error:.6f} "
            f"start_joint_error_max_deg={start_joint_error_max_deg:.6f} final_joint_error_max_deg={final_joint_error_max_deg:.6f} "
            f"start_shoulder_error_deg={start_shoulder_error_deg:.6f} final_shoulder_error_deg={final_shoulder_error_deg:.6f} "
            f"min_joint_error_max_deg={min_joint_error_max_deg:.6f} max_realized_tcp_delta_m={max_realized_tcp_delta:.6f} "
            f"max_step_tcp_delta_m={max_step_tcp_delta:.6f} max_joint_vel_abs_deg_s={max_joint_vel_abs_deg_s:.6f} "
            f"set_target_seen={_yes(watch['calls'] > 0 and watch['max_diff'] <= 1.0e-5)} "
            f"set_max_diff_rad={watch['max_diff']:.8f} robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np))):.8f} "
            f"best_data_target_attr_diff_rad={'nan' if not math.isfinite(best_attr_diff) else f'{best_attr_diff:.8f}'} "
            f"start_sponge_xyz={_fmt_xyz(start_sponge)} start_sponge_quat_wxyz={_fmt_quat(start_quat)} "
            f"start_target_tcp_minus_sponge_oriented_top_m={start_prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
            f"start_target_top_class={start_prox['target_top_class']} "
            f"start_target_xy_inside_sponge_aabb={_yes(bool(start_prox['target_xy_inside_sponge_aabb']))} "
            f"final_sponge_xyz={_fmt_xyz(final_sponge)} final_sponge_quat_wxyz={_fmt_quat(final_quat)} "
            f"final_tcp_minus_sponge_oriented_top_m={final_prox['tcp_minus_sponge_oriented_top_m']:.6f} "
            f"final_target_tcp_minus_sponge_oriented_top_m={final_prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
            f"final_target_top_class={final_prox['target_top_class']} "
            f"final_target_xy_inside_sponge_aabb={_yes(bool(final_prox['target_xy_inside_sponge_aabb']))} "
            f"final_tcp_xy_inside_sponge_aabb={_yes(bool(final_prox['tcp_xy_inside_sponge_aabb']))} "
            f"max_sponge_drift_m={max_sponge_drift:.6f} max_sponge_speed_mps={max_sponge_speed:.6f} "
            f"exact_converged={_yes(exact_converged)} reduction_gate_would_pass={_yes(reduction_gate_would_pass)} "
            f"top_clamped={_yes(top_clamped)} mechanically_valid_target={_yes(mechanically_valid_target)} "
            f"clean_realized_without_reduction_artifact={_yes(clean_realized)} "
            f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
            flush=True,
        )
        return {
            "label": label,
            "exact_converged": exact_converged,
            "reduction_gate_would_pass": reduction_gate_would_pass,
            "top_clamped": top_clamped,
            "target_top_class": str(final_prox["target_top_class"]),
            "mechanically_valid_target": mechanically_valid_target,
            "clean_realized": clean_realized,
            "target_xy_inside": bool(final_prox["target_xy_inside_sponge_aabb"]),
            "final_error": final_target_tcp_error,
        }

    lo = base_env.robot_dof_lower_limits.detach().cpu().numpy().astype(np.float64)
    hi = base_env.robot_dof_upper_limits.detach().cpu().numpy().astype(np.float64)
    actuator_parts = []
    for name, actuator in base_env.cfg.robot.actuators.items():
        actuator_parts.append(
            f"{name}:stiffness={actuator.stiffness},damping={actuator.damping},"
            f"effort_limit_sim={actuator.effort_limit_sim},velocity_limit_sim={actuator.velocity_limit_sim}"
        )
    print(
        f"[roarm_chain_preclose_clearance] controller_config action_scale={base_env.cfg.action_scale:.6f} "
        f"null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f} "
        f"soft_lower_limits_deg={_fmt_deg(np.degrees(lo))} soft_upper_limits_deg={_fmt_deg(np.degrees(hi))} "
        f"actuators={'|'.join(actuator_parts)}",
        flush=True,
    )

    strategy_results: list[tuple[str, bool, list[dict[str, float | bool | str]]]] = []
    for strategy in strategies:
        print(
            f"[roarm_chain_preclose_clearance] strategy_plan name={strategy.name} hypothesis={strategy.hypothesis!r} "
            f"sponge_xy=({strategy.sponge_xy[0]:+.6f},{strategy.sponge_xy[1]:+.6f}) segment_count={len(strategy.segments)}",
            flush=True,
        )
        prepared, prep_error = reset_and_prepare(strategy)
        print(
            f"[roarm_chain_preclose_clearance] strategy_prepare_result name={strategy.name} "
            f"prepared={_yes(prepared)} prep_error_m={prep_error:.6f}",
            flush=True,
        )
        reassert_sponge_pose(strategy, f"{strategy.name}_before_segments")
        segment_results: list[dict[str, float | bool | str]] = []
        for segment in strategy.segments:
            soft_ok, soft_lower_margin, soft_upper_margin = _soft_limit_ok(segment.q_deg, lo, hi)
            target_tcp = fk_tcp(segment.q_deg)
            target_minus_top = target_tcp[2] - top_z
            print(
                f"[roarm_chain_preclose_clearance] segment_plan strategy={strategy.name} segment={segment.name} "
                f"note={segment.note!r} target_q_deg={_fmt_deg(segment.q_deg)} target_tcp={_fmt_xyz(target_tcp)} "
                f"target_tcp_minus_nominal_top_m={target_minus_top:.6f} "
                f"soft_lower_margin_min_rad={soft_lower_margin:.6f} soft_upper_margin_min_rad={soft_upper_margin:.6f} "
                f"soft_limits_ok={_yes(soft_ok)} analytic_joint_limits_ok={_yes(_analytic_limits_ok(segment.q_deg))}",
                flush=True,
            )
            segment_results.append(run_segment(strategy.name, segment))
        strategy_clean = all(bool(item["clean_realized"]) for item in segment_results)
        strategy_results.append((strategy.name, strategy_clean, segment_results))
        print(
            f"[roarm_chain_preclose_clearance] strategy_result name={strategy.name} "
            f"all_segments_clean_realized={_yes(strategy_clean)} "
            f"segment_clean_flags={[bool(item['clean_realized']) for item in segment_results]} "
            f"segment_top_classes={[item['target_top_class'] for item in segment_results]} "
            f"segment_top_clamped={[bool(item['top_clamped']) for item in segment_results]}",
            flush=True,
        )

    clean_strategies = [name for name, clean, _segments in strategy_results if clean]
    clamped_segments = [
        item["label"]
        for _name, _clean, segments in strategy_results
        for item in segments
        if bool(item["top_clamped"])
    ]
    below_segments_clean = [
        item["label"]
        for _name, _clean, segments in strategy_results
        for item in segments
        if item["target_top_class"] == "below" and bool(item["clean_realized"])
    ]
    above_or_tangent_clean = [
        item["label"]
        for _name, _clean, segments in strategy_results
        for item in segments
        if item["target_top_class"] in ("above", "tangent") and bool(item["clean_realized"])
    ]
    far_control_clean = any(name == "far_sponge_below_top_plus5deg_control" and clean for name, clean, _segments in strategy_results)
    nominal_baseline_clean = any(name == "baseline_nominal_below_top_plus5deg" and clean for name, clean, _segments in strategy_results)
    success = not nan_seen and not episode_done and len(strategy_results) == len(strategies)
    print(
        f"[roarm_chain_preclose_clearance] aggregate total_sim_steps={total_sim_steps} strategies_tested={len(strategy_results)} "
        f"clean_strategies={clean_strategies} clamped_segments={clamped_segments} "
        f"below_segments_clean={below_segments_clean} above_or_tangent_clean={above_or_tangent_clean} "
        f"nominal_baseline_clean={_yes(nominal_baseline_clean)} far_control_clean={_yes(far_control_clean)} "
        f"attach_calls={attach_stats['attach_calls']} action_scale={base_env.cfg.action_scale:.6f} "
        f"null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f}",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] hypothesis_flags "
        f"below_top_nominal_invalid={_yes((not nominal_baseline_clean) and len(clamped_segments) > 0)} "
        f"far_sponge_realizes_below_top_nominal_fails={_yes(far_control_clean and not nominal_baseline_clean)} "
        f"above_or_tangent_targets_realize_cleanly={_yes(len(above_or_tangent_clean) > 0)} "
        f"below_top_targets_realize_cleanly={_yes(len(below_segments_clean) > 0)} "
        f"safe_strategy_candidates={clean_strategies} "
        f"attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(
        f"[roarm_chain_preclose_clearance] ROARM_PRECLOSE_CLEARANCE_STRATEGY_DIAGNOSTIC_SUCCESS={_yes(success)}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
