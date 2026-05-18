#!/usr/bin/env python3
"""Local grasp-pose command-realization dead-zone diagnostic for P7 Branch B.

This stays pre-integration: no constraint prim insertion, no SurfaceGripper,
no transport target, no release, no P7 training, and no env/train/chain default
edits. It compares the same shoulder nudge around the local grasp pose while
changing only diagnostic-local sponge proximity, pre-grasp height, and open vs
sub-threshold partial gripper closure.
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


def _offset_tag_mm(offset_mm: float) -> str:
    sign = "plus" if offset_mm >= 0.0 else "minus"
    mag = f"{abs(offset_mm):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{sign}_{mag}mm"


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


@dataclass(frozen=True)
class Condition:
    name: str
    q_deg: np.ndarray
    sponge_xy: tuple[float, float]
    gripper_deg: float
    z_offset_m: float
    ik_converged: bool
    ik_err_mm: float
    note: str


def _soft_limit_ok(q_deg: np.ndarray, lower_rad: np.ndarray, upper_rad: np.ndarray) -> tuple[bool, float, float]:
    q_rad = np.radians(q_deg)
    lower_margin = q_rad - lower_rad
    upper_margin = upper_rad - q_rad
    return (
        bool(np.all(lower_margin >= -1.0e-5) and np.all(upper_margin >= -1.0e-5)),
        float(np.min(lower_margin)),
        float(np.min(upper_margin)),
    )


def _analytic_limits_ok(q_deg: np.ndarray) -> bool:
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    return all(JOINT_LIMITS_DEG[name][0] - 1.0e-6 <= q_deg[i] <= JOINT_LIMITS_DEG[name][1] + 1.0e-6 for i, name in enumerate(names))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--far_sponge_xy", nargs=2, type=float, default=[0.80, 0.40])
    ap.add_argument("--z_offsets_mm", nargs="*", type=float, default=[3.0, 6.0, 12.0])
    ap.add_argument("--sponge_x_offsets_mm", nargs="*", type=float, default=[])
    ap.add_argument("--sponge_y_offsets_mm", nargs="*", type=float, default=[])
    ap.add_argument("--partial_gripper_deg", type=float, default=20.0)
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--delivery_steps", type=int, default=40)
    ap.add_argument("--direct_steps", type=int, default=20)
    ap.add_argument("--restore_steps", type=int, default=40)
    ap.add_argument("--joint_nudge_index", type=int, default=1)
    ap.add_argument("--joint_nudge_deg", type=float, default=5.0)
    ap.add_argument("--joint_nudge_degs", nargs="*", type=float, default=None)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=16)
    ap.add_argument("--trace_every_step", action="store_true")
    ap.add_argument("--reassert_sponge_before_delivery", action="store_true")
    ap.add_argument("--reassert_sponge_z_m", type=float, default=None)
    args = ap.parse_args()

    if args.joint_nudge_index != 1:
        raise ValueError("this diagnostic is intentionally scoped to shoulder joint index 1")
    joint_nudge_degs = args.joint_nudge_degs if args.joint_nudge_degs is not None else [args.joint_nudge_deg]
    if len(joint_nudge_degs) < 1:
        raise ValueError("at least one joint nudge magnitude is required")

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    pre_move_events = [event for event in events if event.kind == "PRE_MOVE"]
    grasp_event = pre_move_events[-1]
    base_grasp_q = grasp_event.q_deg.copy()
    base_grasp_q[5] = GRIPPER_OPEN_DEG

    conditions: list[Condition] = [
        Condition(
            name="nominal_sponge_open",
            q_deg=base_grasp_q.copy(),
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            gripper_deg=GRIPPER_OPEN_DEG,
            z_offset_m=0.0,
            ik_converged=True,
            ik_err_mm=0.0,
            note="nominal grasp pose, nominal sponge, gripper open",
        ),
        Condition(
            name="far_sponge_open",
            q_deg=base_grasp_q.copy(),
            sponge_xy=(args.far_sponge_xy[0], args.far_sponge_xy[1]),
            gripper_deg=GRIPPER_OPEN_DEG,
            z_offset_m=0.0,
            ik_converged=True,
            ik_err_mm=0.0,
            note="same robot q, sponge moved far for no-contact comparison",
        ),
    ]

    for axis, offset_values in (("x", args.sponge_x_offsets_mm), ("y", args.sponge_y_offsets_mm)):
        for offset_mm in offset_values:
            offset_m = offset_mm / 1000.0
            sponge_xy = [float(args.sponge_xy[0]), float(args.sponge_xy[1])]
            idx = 0 if axis == "x" else 1
            sponge_xy[idx] += offset_m
            conditions.append(
                Condition(
                    name=f"sponge_{axis}_{_offset_tag_mm(offset_mm)}_open",
                    q_deg=base_grasp_q.copy(),
                    sponge_xy=(sponge_xy[0], sponge_xy[1]),
                    gripper_deg=GRIPPER_OPEN_DEG,
                    z_offset_m=0.0,
                    ik_converged=True,
                    ik_err_mm=0.0,
                    note="same robot q/target, sponge shifted horizontally for proximity boundary sweep",
                )
            )

    base_tcp = fk_tcp(base_grasp_q)
    q_seed = base_grasp_q.copy()
    for offset_mm in args.z_offsets_mm:
        offset_m = offset_mm / 1000.0
        target_tcp = base_tcp + np.array([0.0, 0.0, offset_m], dtype=np.float64)
        q_z, conv, err_mm, _n_iter = ik_dls(target_tcp, q_seed, max_iter=200, tol_mm=1.0)
        q_z = clip_joints(q_z)
        q_z[4] = PICK_WRIST_R_DEG
        q_z[5] = GRIPPER_OPEN_DEG
        conditions.append(
            Condition(
                name=f"nominal_sponge_z_{_offset_tag_mm(offset_mm)}_open",
                q_deg=q_z.copy(),
                sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
                gripper_deg=GRIPPER_OPEN_DEG,
                z_offset_m=offset_m,
                ik_converged=bool(conv),
                ik_err_mm=float(err_mm),
                note="diagnostic-local higher pre-grasp z offset, nominal sponge, gripper open",
            )
        )

    partial_q = base_grasp_q.copy()
    partial_q[5] = args.partial_gripper_deg
    conditions.append(
        Condition(
            name="nominal_sponge_partial_close",
            q_deg=partial_q,
            sponge_xy=(args.sponge_xy[0], args.sponge_xy[1]),
            gripper_deg=args.partial_gripper_deg,
            z_offset_m=0.0,
            ik_converged=True,
            ik_err_mm=0.0,
            note="same grasp pose with sub-threshold partial gripper closure",
        )
    )

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

    print("[roarm_chain_grasp_deadzone] grasp_pose_deadzone_probe", flush=True)
    print(
        "[roarm_chain_grasp_deadzone] "
        "local_grasp_pose_command_realization_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO attached_transport=NO transport_target=NO "
        "release_marker=NO p7_training=NO p7_tuning=NO env_default_edits=NO chain_defaults_edits=NO "
        "kinematic_env_latch_only=YES attach_physics_validated=NO release_physics_validated=NO "
        "claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"delivery_steps={args.delivery_steps} direct_steps={args.direct_steps} "
        f"joint_nudge_index={args.joint_nudge_index} joint_nudge_degs={joint_nudge_degs} "
        f"partial_gripper_deg={args.partial_gripper_deg:.3f} resample_fraction={args.resample_fraction:.3f} "
        f"trace_every_step={_yes(args.trace_every_step)} "
        f"reassert_sponge_before_delivery={_yes(args.reassert_sponge_before_delivery)} "
        f"reassert_sponge_z_m={args.reassert_sponge_z_m}",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] local_variations "
        f"z_offsets_mm={args.z_offsets_mm} sponge_x_offsets_mm={args.sponge_x_offsets_mm} "
        f"sponge_y_offsets_mm={args.sponge_y_offsets_mm}",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] stream source_events_total={meta['events_total']} "
        f"executed_pre_moves={len(pre_move_events)} move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] contact_sensor_available=NO "
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
                    f"[roarm_chain_grasp_deadzone] set_joint_position_target_call "
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

    def direct_step_once() -> None:
        base_env.scene.write_data_to_sim()
        try:
            base_env.sim.step(render=False)
        except TypeError:
            base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())

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

    def sponge_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def sponge_quat_wxyz() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def sponge_speed() -> float:
        return float(torch.linalg.norm(base_env._sponge.data.root_vel_w[0]).detach().cpu().item())

    def soft_limits_rad() -> tuple[np.ndarray, np.ndarray]:
        lo = base_env.robot_dof_lower_limits.detach().cpu().numpy().astype(np.float64)
        hi = base_env.robot_dof_upper_limits.detach().cpu().numpy().astype(np.float64)
        return lo, hi

    def data_target_snapshot(label: str, target_rad_np: np.ndarray) -> tuple[str, float]:
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
            f"[roarm_chain_grasp_deadzone] data_target_snapshot label={label} "
            f"target_attrs={names} best_attr_diff_rad={best_text}",
            flush=True,
        )
        return ",".join(names), best

    def data_joint_pos_target_rad() -> np.ndarray | None:
        value = getattr(base_env._robot.data, "joint_pos_target", None)
        if value is None or not hasattr(value, "shape"):
            return None
        arr = _as_np(value)
        if arr.size < 6:
            return None
        return arr.reshape(-1, arr.shape[-1])[0, :6].astype(np.float64)

    def sponge_pose_metrics(pos: np.ndarray, quat: np.ndarray) -> dict[str, float]:
        rot = _quat_wxyz_to_rot(quat)
        half_extents = np.array(
            [SPONGE_LEN_LONG / 2.0, SPONGE_WIDTH / 2.0, SPONGE_HEIGHT_EDGE / 2.0],
            dtype=np.float64,
        )
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

    def proximity_metrics(tcp: np.ndarray, target_tcp: np.ndarray, sponge: np.ndarray, sponge_quat: np.ndarray) -> dict[str, float | bool]:
        half_long = SPONGE_LEN_LONG / 2.0
        half_width = SPONGE_WIDTH / 2.0
        pose = sponge_pose_metrics(sponge, sponge_quat)
        upright_top_z = float(pose["upright_top_z_m"])
        oriented_top_z = float(pose["oriented_top_z_m"])
        dx = abs(float(tcp[0] - sponge[0]))
        dy = abs(float(tcp[1] - sponge[1]))
        target_dx = abs(float(target_tcp[0] - sponge[0]))
        target_dy = abs(float(target_tcp[1] - sponge[1]))
        return {
            "d_tcp_sponge_m": _norm(tcp - sponge),
            "d_target_tcp_sponge_m": _norm(target_tcp - sponge),
            "tcp_dx_sponge_m": float(tcp[0] - sponge[0]),
            "tcp_dy_sponge_m": float(tcp[1] - sponge[1]),
            "target_dx_sponge_m": float(target_tcp[0] - sponge[0]),
            "target_dy_sponge_m": float(target_tcp[1] - sponge[1]),
            "sponge_upright_top_z_m": upright_top_z,
            "sponge_oriented_top_z_m": oriented_top_z,
            "sponge_up_z": float(pose["up_z"]),
            "sponge_tilt_deg": float(pose["tilt_deg"]),
            "sponge_axis_x_z_abs": float(pose["axis_x_z_abs"]),
            "sponge_axis_y_z_abs": float(pose["axis_y_z_abs"]),
            "sponge_axis_z_z_abs": float(pose["axis_z_z_abs"]),
            "tcp_minus_sponge_upright_top_m": float(tcp[2] - upright_top_z),
            "target_tcp_minus_sponge_upright_top_m": float(target_tcp[2] - upright_top_z),
            "tcp_minus_sponge_oriented_top_m": float(tcp[2] - oriented_top_z),
            "target_tcp_minus_sponge_oriented_top_m": float(target_tcp[2] - oriented_top_z),
            "tcp_xy_inside_sponge_aabb": dx <= half_long and dy <= half_width,
            "target_xy_inside_sponge_aabb": target_dx <= half_long and target_dy <= half_width,
        }

    def reassert_sponge_pose(cond: Condition, label: str) -> None:
        z_m = ENV_SPONGE_CENTER_Z if args.reassert_sponge_z_m is None else float(args.reassert_sponge_z_m)
        pose = torch.tensor(
            [[cond.sponge_xy[0], cond.sponge_xy[1], z_m, 1.0, 0.0, 0.0, 0.0]],
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
            f"[roarm_chain_grasp_deadzone] sponge_pose_reassert label={label} "
            f"requested_xyz=({cond.sponge_xy[0]:+.6f},{cond.sponge_xy[1]:+.6f},{z_m:+.6f}) "
            f"actual_xyz={_fmt_xyz(pos)} quat_wxyz={_fmt_quat(quat)} "
            f"up_z={metrics['up_z']:.6f} tilt_deg={metrics['tilt_deg']:.6f} "
            f"upright_top_z_m={metrics['upright_top_z_m']:.6f} "
            f"oriented_top_z_m={metrics['oriented_top_z_m']:.6f}",
            flush=True,
        )

    def print_controller_config() -> None:
        lo, hi = soft_limits_rad()
        actuator_parts = []
        for name, actuator in base_env.cfg.robot.actuators.items():
            actuator_parts.append(
                f"{name}:stiffness={actuator.stiffness},damping={actuator.damping},"
                f"effort_limit_sim={actuator.effort_limit_sim},velocity_limit_sim={actuator.velocity_limit_sim}"
            )
        print(
            f"[roarm_chain_grasp_deadzone] controller_config action_scale={base_env.cfg.action_scale:.6f} "
            f"null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f} "
            f"soft_lower_limits_deg={_fmt_deg(np.degrees(lo))} soft_upper_limits_deg={_fmt_deg(np.degrees(hi))} "
            f"actuators={'|'.join(actuator_parts)}",
            flush=True,
        )

    def reset_and_prepare(cond: Condition) -> tuple[bool, float]:
        nonlocal total_sim_steps, episode_done, nan_seen
        env.reset()
        home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
        base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
        base_env._robot.set_joint_position_target(home_rad)
        base_env.robot_dof_targets[:] = home_rad
        sponge_pose = torch.tensor(
            [[cond.sponge_xy[0], cond.sponge_xy[1], ENV_SPONGE_CENTER_Z, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        sponge_pose[:, 0:3] += base_env.scene.env_origins[:1]
        base_env._sponge.write_root_pose_to_sim(sponge_pose)
        base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        base_env._grasped[:] = False
        base_env._was_grasped[:] = False
        for _ in range(args.initial_settle_steps):
            episode_done |= step_once()
            total_sim_steps += 1
        home_error = _norm(fresh_tcp_local() - fk_tcp(HOME_DEG))
        if not math.isfinite(home_error):
            nan_seen = True

        # Replay the conservative pre-close path first to preserve controller/path context.
        for event in pre_move_events:
            reached, _steps, _err = run_to_q(f"{cond.name}_pre_{event.index:03d}", event.q_deg, args.max_steps_per_event, quiet=True)
            if not reached:
                print(
                    f"[roarm_chain_grasp_deadzone] prepare_pre_move_failed condition={cond.name} "
                    f"event_index={event.index:03d} final_error_m={_err:.6f}",
                    flush=True,
                )
                break
        reached, _steps, final_err = run_to_q(f"{cond.name}_condition_pose", cond.q_deg, args.restore_steps, quiet=False)
        return reached and home_error <= args.home_fk_gate_m, final_err

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
                f"[roarm_chain_grasp_deadzone] restore_or_advance label={label} steps={steps_used} "
                f"final_target_error_m={final_error:.6f} reached={_yes(final_error <= args.target_error_gate_m)} "
                f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
                flush=True,
            )
        return final_error <= args.target_error_gate_m, steps_used, final_error

    def run_delivery(label: str, mode: str, target_rad, target_rad_np: np.ndarray, steps: int):
        nonlocal total_sim_steps, episode_done, nan_seen
        start_tcp = fresh_tcp_local()
        start_q = current_q_rad()
        start_sponge = sponge_local()
        start_sponge_quat = sponge_quat_wxyz()
        expected_tcp = fk_tcp(np.degrees(target_rad_np))
        expected_motion_from_start = _norm(expected_tcp - start_tcp)
        start_joint_error_max_deg = float(np.max(np.abs(np.degrees(target_rad_np - start_q))))
        start_shoulder_error_deg = float(abs(np.degrees(target_rad_np[1] - start_q[1])))
        start_prox = proximity_metrics(start_tcp, expected_tcp, start_sponge, start_sponge_quat)
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
        final_target_tcp_error = float("inf")
        final_joint_error_max_deg = float("inf")
        final_shoulder_error_deg = float("inf")
        prev_tcp = start_tcp
        prev_q = start_q
        best_attr_diff = float("inf")
        for step_idx in range(1, steps + 1):
            if mode == "env_step":
                base_env.robot_dof_targets[:] = target_rad
                done = step_once()
                total_sim_steps += 1
                episode_done |= done
            elif mode == "direct":
                base_env._robot.set_joint_position_target(target_rad)
                direct_step_once()
                total_sim_steps += 1
                done = False
            else:
                raise ValueError(f"unknown delivery mode {mode}")
            after_tcp = fresh_tcp_local()
            after_q = current_q_rad()
            after_sponge = sponge_local()
            after_sponge_quat = sponge_quat_wxyz()
            physics_dt = float(base_env.sim.get_physics_dt())
            step_tcp_delta = _norm(after_tcp - prev_tcp)
            joint_vel_deg_s = np.degrees((after_q - prev_q) / max(physics_dt, 1.0e-9))
            max_joint_vel_abs_deg_s = max(max_joint_vel_abs_deg_s, float(np.max(np.abs(joint_vel_deg_s))))
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
            min_joint_error_max_deg = min(min_joint_error_max_deg, joint_error_max_deg)
            final_target_tcp_error = target_tcp_error
            final_joint_error_max_deg = joint_error_max_deg
            final_shoulder_error_deg = shoulder_error_deg
            if args.trace_every_step or step_idx in (1, 2, 3, steps):
                _attrs, diff = data_target_snapshot(f"{label}_step{step_idx:03d}", target_rad_np)
                best_attr_diff = min(best_attr_diff, diff)
                gripper_q_deg = float(np.degrees(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item()))
                prox = proximity_metrics(after_tcp, expected_tcp, after_sponge, after_sponge_quat)
                data_target_rad = data_joint_pos_target_rad()
                data_target_diff_rad = float("nan")
                data_target_deg_text = "NA"
                if data_target_rad is not None:
                    data_target_diff_rad = float(np.max(np.abs(data_target_rad - target_rad_np)))
                    data_target_deg_text = _fmt_deg(np.degrees(data_target_rad))
                print(
                    f"[roarm_chain_grasp_deadzone] delivery_step label={label} mode={mode} "
                    f"step={step_idx:03d} set_calls={watch['calls']} set_max_diff_rad={watch['max_diff']:.8f} "
                    f"robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np))):.8f} "
                    f"robot_dof_targets_deg={_fmt_deg(np.degrees(targets_rad()))} "
                    f"data_joint_pos_target_diff_rad={'nan' if not math.isfinite(data_target_diff_rad) else f'{data_target_diff_rad:.8f}'} "
                    f"data_joint_pos_target_deg={data_target_deg_text} "
                    f"current_q_deg={_fmt_deg(np.degrees(after_q))} joint_error_deg={_fmt_deg(joint_error_deg)} "
                    f"joint_vel_deg_s={_fmt_deg(joint_vel_deg_s)} joint_error_max_deg={joint_error_max_deg:.6f} "
                    f"shoulder_error_deg={shoulder_error_deg:.6f} elbow_error_deg={elbow_error_deg:+.6f} "
                    f"wrist_p_error_deg={wrist_p_error_deg:+.6f} wrist_r_error_deg={wrist_r_error_deg:+.6f} "
                    f"shoulder_error_reduction_deg={start_shoulder_error_deg - shoulder_error_deg:+.6f} "
                    f"fresh_tcp={_fmt_xyz(after_tcp)} expected_tcp={_fmt_xyz(expected_tcp)} "
                    f"tcp_z_m={after_tcp[2]:+.6f} expected_tcp_z_m={expected_tcp[2]:+.6f} "
                    f"target_tcp_error_m={target_tcp_error:.6f} "
                    f"target_tcp_error_reduction_m={expected_motion_from_start - target_tcp_error:+.6f} "
                    f"realized_tcp_delta_m={realized_tcp_delta:.6f} step_tcp_delta_m={step_tcp_delta:.6f} "
                    f"d_tcp_sponge_m={prox['d_tcp_sponge_m']:.6f} d_target_tcp_sponge_m={prox['d_target_tcp_sponge_m']:.6f} "
                    f"target_dx_sponge_m={prox['target_dx_sponge_m']:.6f} "
                    f"target_dy_sponge_m={prox['target_dy_sponge_m']:.6f} "
                    f"sponge_quat_wxyz={_fmt_quat(after_sponge_quat)} "
                    f"sponge_up_z={prox['sponge_up_z']:.6f} sponge_tilt_deg={prox['sponge_tilt_deg']:.6f} "
                    f"sponge_axis_z_abs=({prox['sponge_axis_x_z_abs']:.6f},{prox['sponge_axis_y_z_abs']:.6f},{prox['sponge_axis_z_z_abs']:.6f}) "
                    f"sponge_upright_top_z_m={prox['sponge_upright_top_z_m']:.6f} "
                    f"sponge_oriented_top_z_m={prox['sponge_oriented_top_z_m']:.6f} "
                    f"tcp_minus_sponge_oriented_top_m={prox['tcp_minus_sponge_oriented_top_m']:.6f} "
                    f"target_tcp_minus_sponge_oriented_top_m={prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
                    f"target_tcp_minus_sponge_upright_top_m={prox['target_tcp_minus_sponge_upright_top_m']:.6f} "
                    f"tcp_xy_inside_sponge_aabb={_yes(bool(prox['tcp_xy_inside_sponge_aabb']))} "
                    f"target_xy_inside_sponge_aabb={_yes(bool(prox['target_xy_inside_sponge_aabb']))} "
                    f"sponge_drift_m={sponge_drift:.6f} sponge_speed_mps={sponge_speed():.6f} "
                    f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
                    f"gripper_q_deg={gripper_q_deg:+.3f} done={_yes(done)}",
                    flush=True,
                )
            if not np.isfinite(after_tcp).all() or not math.isfinite(target_tcp_error):
                nan_seen = True
            prev_tcp = after_tcp
            prev_q = after_q
        watch["active"] = False
        joint_error_reduced = final_joint_error_max_deg < max(1.0, 0.5 * start_joint_error_max_deg)
        shoulder_error_reduced = final_shoulder_error_deg < max(1.0, 0.5 * start_shoulder_error_deg)
        tcp_target_reduced = final_target_tcp_error < max(args.target_error_gate_m, 0.75 * expected_motion_from_start)
        target_realized = shoulder_error_reduced and tcp_target_reduced
        final_tcp = fresh_tcp_local()
        final_q = current_q_rad()
        final_sponge = sponge_local()
        final_sponge_quat = sponge_quat_wxyz()
        final_prox = proximity_metrics(final_tcp, expected_tcp, final_sponge, final_sponge_quat)
        print(
            f"[roarm_chain_grasp_deadzone] delivery_result label={label} mode={mode} "
            f"steps={steps} start_q_deg={_fmt_deg(np.degrees(start_q))} "
            f"target_q_deg={_fmt_deg(np.degrees(target_rad_np))} "
            f"final_q_deg={_fmt_deg(np.degrees(final_q))} "
            f"expected_motion_from_start_m={expected_motion_from_start:.6f} "
            f"start_joint_error_max_deg={start_joint_error_max_deg:.6f} "
            f"start_shoulder_error_deg={start_shoulder_error_deg:.6f} "
            f"set_calls={watch['calls']} set_target_seen={_yes(watch['calls'] > 0 and watch['max_diff'] <= 1.0e-5)} "
            f"set_max_diff_rad={watch['max_diff']:.8f} "
            f"robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np))):.8f} "
            f"best_data_target_attr_diff_rad={'nan' if not math.isfinite(best_attr_diff) else f'{best_attr_diff:.8f}'} "
            f"max_realized_tcp_delta_m={max_realized_tcp_delta:.6f} max_step_tcp_delta_m={max_step_tcp_delta:.6f} "
            f"max_joint_vel_abs_deg_s={max_joint_vel_abs_deg_s:.6f} "
            f"final_target_tcp_error_m={final_target_tcp_error:.6f} min_joint_error_max_deg={min_joint_error_max_deg:.6f} "
            f"final_joint_error_max_deg={final_joint_error_max_deg:.6f} "
            f"final_shoulder_error_deg={final_shoulder_error_deg:.6f} "
            f"tcp_target_reduced={_yes(tcp_target_reduced)} joint_error_reduced={_yes(joint_error_reduced)} "
            f"shoulder_error_reduced={_yes(shoulder_error_reduced)} target_realized={_yes(target_realized)} "
            f"start_sponge_xyz={_fmt_xyz(start_sponge)} start_sponge_quat_wxyz={_fmt_quat(start_sponge_quat)} "
            f"start_d_tcp_sponge_m={start_prox['d_tcp_sponge_m']:.6f} "
            f"start_target_dx_sponge_m={start_prox['target_dx_sponge_m']:.6f} "
            f"start_target_dy_sponge_m={start_prox['target_dy_sponge_m']:.6f} "
            f"start_sponge_up_z={start_prox['sponge_up_z']:.6f} "
            f"start_sponge_tilt_deg={start_prox['sponge_tilt_deg']:.6f} "
            f"start_sponge_upright_top_z_m={start_prox['sponge_upright_top_z_m']:.6f} "
            f"start_sponge_oriented_top_z_m={start_prox['sponge_oriented_top_z_m']:.6f} "
            f"start_target_tcp_minus_sponge_oriented_top_m={start_prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
            f"start_target_tcp_minus_sponge_upright_top_m={start_prox['target_tcp_minus_sponge_upright_top_m']:.6f} "
            f"start_target_xy_inside_sponge_aabb={_yes(bool(start_prox['target_xy_inside_sponge_aabb']))} "
            f"final_tcp={_fmt_xyz(final_tcp)} final_sponge_xyz={_fmt_xyz(final_sponge)} "
            f"final_tcp_minus_sponge_oriented_top_m={final_prox['tcp_minus_sponge_oriented_top_m']:.6f} "
            f"final_target_tcp_minus_sponge_oriented_top_m={final_prox['target_tcp_minus_sponge_oriented_top_m']:.6f} "
            f"final_tcp_xy_inside_sponge_aabb={_yes(bool(final_prox['tcp_xy_inside_sponge_aabb']))} "
            f"final_target_xy_inside_sponge_aabb={_yes(bool(final_prox['target_xy_inside_sponge_aabb']))} "
            f"max_sponge_drift_m={max_sponge_drift:.6f} max_sponge_speed_mps={max_sponge_speed:.6f} "
            f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
            flush=True,
        )
        return {
            "set_seen": watch["calls"] > 0 and watch["max_diff"] <= 1.0e-5,
            "target_realized": target_realized,
            "tcp_target_reduced": tcp_target_reduced,
            "shoulder_error_reduced": shoulder_error_reduced,
            "final_shoulder_error_deg": final_shoulder_error_deg,
            "final_target_tcp_error": final_target_tcp_error,
            "max_realized_tcp_delta": max_realized_tcp_delta,
            "max_sponge_drift": max_sponge_drift,
            "max_sponge_speed": max_sponge_speed,
            "best_attr_diff": best_attr_diff,
        }

    print_controller_config()
    condition_results: list[tuple[str, dict, dict, bool]] = []
    for cond in conditions:
        for nudge_deg in joint_nudge_degs:
            nudge_tag = _offset_tag_mm(nudge_deg).replace("mm", "deg")
            result_name = cond.name if len(joint_nudge_degs) == 1 else f"{cond.name}_nudge_{nudge_tag}"
            target_q_deg = cond.q_deg.copy()
            target_q_deg[args.joint_nudge_index] += nudge_deg
            target_q_deg[4] = PICK_WRIST_R_DEG
            target_q_deg[5] = cond.gripper_deg
            lo, hi = soft_limits_rad()
            soft_ok, soft_lower_margin, soft_upper_margin = _soft_limit_ok(target_q_deg, lo, hi)
            base_tcp_cond = fk_tcp(cond.q_deg)
            target_tcp_cond = fk_tcp(target_q_deg)
            print(
                f"[roarm_chain_grasp_deadzone] condition_plan condition={result_name} base_condition={cond.name} note={cond.note!r} "
                f"z_offset_m={cond.z_offset_m:.6f} ik_converged={_yes(cond.ik_converged)} ik_err_mm={cond.ik_err_mm:.3f} "
                f"sponge_xy=({cond.sponge_xy[0]:+.6f},{cond.sponge_xy[1]:+.6f}) "
                f"joint_nudge_deg={nudge_deg:+.6f} "
                f"base_q_deg={_fmt_deg(cond.q_deg)} target_q_deg={_fmt_deg(target_q_deg)} "
                f"delta_q_deg={_fmt_deg(target_q_deg - cond.q_deg)} "
                f"base_tcp={_fmt_xyz(base_tcp_cond)} target_tcp={_fmt_xyz(target_tcp_cond)} "
                f"expected_tcp_delta_m={_norm(target_tcp_cond - base_tcp_cond):.6f} "
                f"soft_lower_margin_min_rad={soft_lower_margin:.6f} soft_upper_margin_min_rad={soft_upper_margin:.6f} "
                f"soft_limits_ok={_yes(soft_ok)} analytic_joint_limits_ok={_yes(_analytic_limits_ok(target_q_deg))}",
                flush=True,
            )
            prepared, prep_error = reset_and_prepare(cond)
            target_rad_np = np.radians(target_q_deg)
            target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
            if args.reassert_sponge_before_delivery:
                reassert_sponge_pose(cond, f"{result_name}_before_env")
            env_result = run_delivery(f"{result_name}_joint_nudge_env", "env_step", target_rad, target_rad_np, args.delivery_steps)
            # Compare direct set from the same local pose, not from the env-step aftermath.
            run_to_q(f"{result_name}_restore_before_direct", cond.q_deg, args.restore_steps, quiet=False)
            if args.reassert_sponge_before_delivery:
                reassert_sponge_pose(cond, f"{result_name}_before_direct")
            direct_result = run_delivery(f"{result_name}_joint_nudge_direct", "direct", target_rad, target_rad_np, args.direct_steps)
            condition_results.append((result_name, env_result, direct_result, prepared and prep_error <= args.target_error_gate_m))

    env_realized = [name for name, env_result, _direct, _prepared in condition_results if env_result["target_realized"]]
    env_failed = [name for name, env_result, _direct, _prepared in condition_results if not env_result["target_realized"]]
    direct_realized = [name for name, _env_result, direct_result, _prepared in condition_results if direct_result["target_realized"]]
    direct_rescue = [
        name
        for name, env_result, direct_result, _prepared in condition_results
        if (not env_result["target_realized"]) and direct_result["target_realized"]
    ]
    all_set_seen = all(env_result["set_seen"] and direct_result["set_seen"] for _name, env_result, direct_result, _prepared in condition_results)
    prepared_all = all(prepared for _name, _env_result, _direct_result, prepared in condition_results)
    nominal_failed = any(name.startswith("nominal_sponge_open") for name in env_failed)
    far_realized = any(name.startswith("far_sponge_open") for name in env_realized)
    higher_realized = [name for name in env_realized if "_z_plus_" in name]
    xy_realized = [name for name in env_realized if name.startswith("sponge_x_") or name.startswith("sponge_y_")]
    success = all_set_seen and not nan_seen and not episode_done
    print(
        f"[roarm_chain_grasp_deadzone] aggregate total_sim_steps={total_sim_steps} conditions_tested={len(condition_results)} "
        f"prepared_all={_yes(prepared_all)} env_realized_conditions={env_realized} env_failed_conditions={env_failed} "
        f"direct_realized_conditions={direct_realized} direct_rescue_conditions={direct_rescue} "
        f"all_set_seen={_yes(all_set_seen)} nominal_failed={_yes(nominal_failed)} "
        f"far_sponge_realized={_yes(far_realized)} higher_z_realized_conditions={higher_realized} "
        f"horizontal_sponge_realized_conditions={xy_realized} "
        f"attach_calls={attach_stats['attach_calls']} action_scale={base_env.cfg.action_scale:.6f} "
        f"null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f}",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] hypothesis_flags "
        f"sponge_far_realizes_nominal_fails={_yes(nominal_failed and far_realized)} "
        f"higher_z_realizes_nominal_fails={_yes(nominal_failed and len(higher_realized) > 0)} "
        f"horizontal_sponge_shift_realizes_nominal_fails={_yes(nominal_failed and len(xy_realized) > 0)} "
        f"all_grasp_local_variants_fail={_yes(nominal_failed and len(env_realized) == 0)} "
        f"env_step_direct_split_seen={_yes(len(direct_rescue) > 0)} "
        f"direct_set_also_fails_nominal={_yes(not any(name.startswith('nominal_sponge_open') for name in direct_realized))} "
        f"attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(
        f"[roarm_chain_grasp_deadzone] ROARM_GRASP_POSE_DEADZONE_DIAGNOSTIC_SUCCESS={_yes(success)}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
