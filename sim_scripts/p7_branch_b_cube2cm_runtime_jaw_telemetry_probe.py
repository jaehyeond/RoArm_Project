#!/usr/bin/env python3
"""Runtime jaw telemetry probe for P7 Branch B 2cm cube close diagnostics.

Diagnostic-only Isaac script. It does not train, insert constraints, attach a
SurfaceGripper, execute transport/release, tune gates, or edit env/chain
defaults. The env's hidden pose-write attach path is monkey-patched to a marker
counter so the log can inspect close-time jaw/object geometry without claiming a
physical grasp.
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

from p7_branch_b_cube2cm_gripper_static_geometry_probe import (  # noqa: E402
    _aabb,
    _aabb_overlap,
    _box_vertices,
    _fmt_xyz,
    _gripper_transform,
    _link5_transform,
    _transform_points,
)
from p7_branch_b_cube2cm_local_grasp_close_sweep_probe import (  # noqa: E402
    FOUR_SPONGE_SEED0_SOURCES,
    GRASP_GRIPPER_THRESH_RAD,
    GRIPPER_OPEN_DEG,
    HOME_DEG,
    NAMED_GRASPS,
    PICK_WRIST_R_DEG,
    TABLE_Z,
    _build_plan,
    _build_plan_from_center,
    fk_tcp,
    _norm,
    _quat_wxyz_to_rot,
    _solve_q,
    _yaw_quat_wxyz,
    _yes,
)
from p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_urdf import _translation  # noqa: E402


V4_USD_PATH = "/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/roarm_m3.usd"
V5_USD_PATH = "/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_collision_usd/roarm_m3.usd"
V6_USD_PATH = "/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_collision_usd/roarm_m3.usd"
V7_USD_PATH = "/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"


@dataclass(frozen=True)
class JawGeometry:
    moving_vertices_local: np.ndarray
    counter_vertices_local: np.ndarray
    counter_origin_parent_m: np.ndarray
    counter_parent: str
    moving_size_m: np.ndarray
    counter_size_m: np.ndarray
    design_moving_center_ref: np.ndarray
    design_counter_center_ref: np.ndarray


def _axis_gap(mesh_min: np.ndarray, mesh_max: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(cube_min - mesh_max, mesh_min - cube_max), 0.0)


def _fmt_deg(values: np.ndarray) -> str:
    return "[" + ",".join(f"{v:+.3f}" for v in values) + "]"


def _homogeneous(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def _to_object_frame(points_world: np.ndarray, object_pos: np.ndarray, object_quat: np.ndarray) -> np.ndarray:
    rot = _quat_wxyz_to_rot(object_quat)
    return (points_world - object_pos) @ rot


def _contact_stats(
    points_world: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    object_size: np.ndarray,
    slop_m: float = 0.0,
) -> dict[str, object]:
    points_obj = _to_object_frame(points_world, object_pos, object_quat)
    cube_min = -0.5 * object_size
    cube_max = 0.5 * object_size
    mn, mx = _aabb(points_obj)
    overlap = _aabb_overlap(mn, mx, cube_min, cube_max)
    slop_overlap = _aabb_overlap(mn, mx, cube_min - slop_m, cube_max + slop_m)
    gap = _axis_gap(mn, mx, cube_min, cube_max)
    center = 0.5 * (mn + mx)
    return {
        "center_obj": center,
        "overlap_obj": overlap,
        "slop_overlap_obj": slop_overlap,
        "gap_obj": gap,
        "contact": bool(np.all(overlap > 0.0)),
        "slop_contact": bool(np.all(slop_overlap > 0.0)),
    }


def _candidate_centers_v5(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)
    cube_half_y = object_size[1] * 0.5
    moving_half_y = moving_size[1] * 0.5
    counter_half_y = counter_size[1] * 0.5
    penetration = float(args.design_penetration_m)
    jaw_center = np.asarray(args.jaw_center_obj_m, dtype=np.float64)
    moving = np.array([jaw_center[0], cube_half_y + moving_half_y - penetration, jaw_center[2]], dtype=np.float64)
    counter = np.array([jaw_center[0], -cube_half_y - counter_half_y + penetration, jaw_center[2]], dtype=np.float64)
    return moving, counter


def _jaw_geometry(args: argparse.Namespace, plan) -> JawGeometry:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    moving_size = np.asarray(args.moving_jaw_size_m, dtype=np.float64)
    counter_size = np.asarray(args.counter_jaw_size_m, dtype=np.float64)

    q_design = plan.q_descend_deg.copy()
    q_design[5] = float(args.design_close_deg)
    inv_gripper = np.linalg.inv(_gripper_transform(q_design))
    inv_link5 = np.linalg.inv(_link5_transform(q_design))
    counter_parent = "gripper_link"

    if args.variant == "v5":
        moving_obj, counter_obj = _candidate_centers_v5(args)
        rot = np.array(
            [
                [math.cos(math.radians(plan.yaw_deg)), -math.sin(math.radians(plan.yaw_deg)), 0.0],
                [math.sin(math.radians(plan.yaw_deg)), math.cos(math.radians(plan.yaw_deg)), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        moving_world = plan.center + rot @ moving_obj
        counter_world = plan.center + rot @ counter_obj
        design_moving_ref = moving_obj
        design_counter_ref = counter_obj
    elif args.variant == "v6":
        cube_half_y = object_size[1] * 0.5
        moving_obj = np.array(
            [0.0, cube_half_y - float(args.moving_close_overlap_m), 0.0020],
            dtype=np.float64,
        )
        base_counter_y = -cube_half_y - float(args.counter_open_clearance_m) - 0.0015 * 0.5
        counter_obj = np.array(
            [
                float(args.counter_x_shift_mm) / 1000.0,
                base_counter_y + float(args.counter_y_shift_mm) / 1000.0,
                0.0020,
            ],
            dtype=np.float64,
        )
        rot = np.array(
            [
                [math.cos(math.radians(plan.yaw_deg)), -math.sin(math.radians(plan.yaw_deg)), 0.0],
                [math.sin(math.radians(plan.yaw_deg)), math.cos(math.radians(plan.yaw_deg)), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        moving_world = plan.center + rot @ moving_obj
        counter_world = plan.center + rot @ counter_obj
        design_moving_ref = moving_obj
        design_counter_ref = counter_obj
    elif args.variant == "v7":
        cube_half_y = object_size[1] * 0.5
        moving_obj = np.array(
            [
                0.0,
                cube_half_y + moving_size[1] * 0.5 - float(args.moving_close_overlap_m),
                float(args.jaw_center_z_m),
            ],
            dtype=np.float64,
        )
        counter_obj = np.array(
            [
                float(args.fixed_counter_x_offset_m),
                -cube_half_y - counter_size[1] * 0.5 - float(args.fixed_counter_clearance_m),
                float(args.jaw_center_z_m),
            ],
            dtype=np.float64,
        )
        rot = np.array(
            [
                [math.cos(math.radians(plan.yaw_deg)), -math.sin(math.radians(plan.yaw_deg)), 0.0],
                [math.sin(math.radians(plan.yaw_deg)), math.cos(math.radians(plan.yaw_deg)), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        moving_world = plan.center + rot @ moving_obj
        counter_world = plan.center + rot @ counter_obj
        design_moving_ref = moving_obj
        design_counter_ref = counter_obj
        counter_parent = "link5"
    else:
        moving_world = np.array(
            [
                plan.center[0],
                plan.center[1] + object_size[1] / 2.0 - float(args.moving_close_overlap_m),
                float(args.jaw_center_z_m),
            ],
            dtype=np.float64,
        )
        counter_world = np.array(
            [
                plan.center[0],
                plan.center[1] - object_size[1] / 2.0 - float(args.counter_open_clearance_m) - counter_size[1] / 2.0,
                float(args.jaw_center_z_m),
            ],
            dtype=np.float64,
        )
        design_moving_ref = moving_world - plan.center
        design_counter_ref = counter_world - plan.center

    moving_local_center = (inv_gripper @ np.array([*moving_world, 1.0], dtype=np.float64))[:3]
    if counter_parent == "link5":
        counter_origin_parent = (inv_link5 @ np.array([*counter_world, 1.0], dtype=np.float64))[:3]
    else:
        counter_origin_parent = (inv_gripper @ np.array([*counter_world, 1.0], dtype=np.float64))[:3]

    return JawGeometry(
        moving_vertices_local=_box_vertices(moving_local_center, moving_size, 0.0),
        counter_vertices_local=_box_vertices(np.zeros(3, dtype=np.float64), counter_size, 0.0),
        counter_origin_parent_m=counter_origin_parent,
        counter_parent=counter_parent,
        moving_size_m=moving_size,
        counter_size_m=counter_size,
        design_moving_center_ref=design_moving_ref,
        design_counter_center_ref=design_counter_ref,
    )


def _apply_variant_defaults(args: argparse.Namespace) -> None:
    if args.yaw_deg is None:
        args.yaw_deg = 50.0 if args.variant == "v5" else 0.0
    if args.normalized_grasp is None and args.variant == "v5":
        args.normalized_grasp = [0.150, -0.150, 0.500]
    if args.normalized_grasp is None and args.variant == "v6":
        args.normalized_grasp = [0.000, 0.000, 0.500]
    if args.normalized_grasp is None and args.variant == "v7":
        args.normalized_grasp = [0.000, 0.000, 0.500]
    if args.variant == "v6":
        if np.allclose(args.object_size_m, [0.020, 0.020, 0.020], atol=1.0e-12):
            args.object_size_m = [0.030, 0.030, 0.030]
        if np.allclose(args.counter_jaw_size_m, [0.004, 0.0015, 0.008], atol=1.0e-12):
            args.counter_jaw_size_m = [0.004, 0.0050, 0.008]
        if abs(float(args.jaw_center_z_m) - 0.012) <= 1.0e-12:
            args.jaw_center_z_m = 0.017
    if args.variant == "v7":
        if np.allclose(args.object_size_m, [0.020, 0.020, 0.020], atol=1.0e-12):
            args.object_size_m = [0.030, 0.030, 0.030]
        if np.allclose(args.counter_jaw_size_m, [0.004, 0.0015, 0.008], atol=1.0e-12):
            args.counter_jaw_size_m = [0.004, 0.0050, 0.008]
        if abs(float(args.moving_close_overlap_m) - (-0.0015)) <= 1.0e-12:
            args.moving_close_overlap_m = 0.0015
        if abs(float(args.jaw_center_z_m) - 0.012) <= 1.0e-12:
            args.jaw_center_z_m = 0.002
        if abs(float(args.counter_contact_slop_m)) <= 1.0e-12:
            args.counter_contact_slop_m = 0.0010
    if args.robot_usd_path is None:
        if args.variant == "v5":
            args.robot_usd_path = V5_USD_PATH
        elif args.variant == "v6":
            args.robot_usd_path = V6_USD_PATH
        elif args.variant == "v7":
            args.robot_usd_path = V7_USD_PATH
        else:
            args.robot_usd_path = V4_USD_PATH


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v4", "v5", "v6", "v7"], default="v5")
    ap.add_argument("--robot_usd_path", default=None)
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--object_mass_kg", type=float, default=0.02)
    ap.add_argument("--pose_label", default="seed0_S1")
    ap.add_argument("--object_xy", nargs=2, type=float, default=None)
    ap.add_argument("--yaw_deg", type=float, default=None)
    ap.add_argument("--grasp_name", choices=sorted(NAMED_GRASPS), default="top_center")
    ap.add_argument("--normalized_grasp", nargs=3, type=float, default=None)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[26.0])
    ap.add_argument("--design_close_deg", type=float, default=26.0)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.80)
    ap.add_argument("--substep_steps", type=int, default=60)
    ap.add_argument("--close_steps", type=int, default=45)
    ap.add_argument("--initial_settle_steps", type=int, default=30)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--gripper_error_gate_deg", type=float, default=0.75)
    ap.add_argument("--object_drift_gate_m", type=float, default=0.006)
    ap.add_argument("--object_speed_gate_mps", type=float, default=0.080)
    ap.add_argument("--tilt_gate_deg", type=float, default=12.0)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.95)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--log_every_close_step", type=int, default=1)
    ap.add_argument("--push_drift_gate_m", type=float, default=0.00020)
    ap.add_argument("--push_speed_gate_mps", type=float, default=0.005)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--moving_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--counter_jaw_size_m", nargs=3, type=float, default=[0.004, 0.0015, 0.008])
    ap.add_argument("--moving_close_overlap_m", type=float, default=-0.0015)
    ap.add_argument("--counter_open_clearance_m", type=float, default=0.00075)
    ap.add_argument("--jaw_center_z_m", type=float, default=0.012)
    ap.add_argument("--jaw_center_obj_m", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--design_penetration_m", type=float, default=0.0015)
    ap.add_argument("--counter_x_shift_mm", type=float, default=1.0)
    ap.add_argument("--counter_y_shift_mm", type=float, default=5.0)
    ap.add_argument("--fixed_counter_clearance_m", type=float, default=0.0021)
    ap.add_argument("--fixed_counter_x_offset_m", type=float, default=0.0)
    ap.add_argument("--counter_contact_slop_m", type=float, default=0.0)
    args = ap.parse_args()
    _apply_variant_defaults(args)

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")
    if len(args.close_deg) != 1:
        raise ValueError("runtime telemetry is close_26-only; pass exactly one --close_deg value")
    if math.radians(args.close_deg[0]) < GRASP_GRIPPER_THRESH_RAD:
        raise ValueError("close_deg must be at or above grasp_gripper_thresh")
    if args.approach_clearance_m <= args.grasp_surface_margin_m:
        raise ValueError("approach_clearance_m must be above grasp_surface_margin_m")
    if args.command_resample_fraction <= 0.0 or args.command_resample_fraction > 1.0:
        raise ValueError("command_resample_fraction must be in (0, 1]")
    return args


def main() -> int:
    args = _parse_args()
    os.environ["ROARM_M3_USD_PATH"] = str(args.robot_usd_path)
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    plan = _build_plan(args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    print("[cube2cm_runtime_jaw_telemetry] runtime jaw telemetry probe", flush=True)
    print(
        "[cube2cm_runtime_jaw_telemetry] "
        f"diagnostic_only=YES isaac_run=YES variant={args.variant} robot_usd_path={args.robot_usd_path} "
        "env_default_edits=NO chain_defaults_edits=NO p7_training=NO constraint_prim_insertion=NO "
        "surface_gripper=NO attached_transport=NO transport_target=NO release_marker=NO "
        "scripted_release_variant=NO gate_tuning=NO close_26_only=YES "
        "hidden_kinematic_posewrite_allowed=NO claim_p7_success=NO",
        flush=True,
    )
    print(
        f"[cube2cm_runtime_jaw_telemetry] selected pose={plan.label} center={_fmt_xyz(plan.center)} "
        f"object_size_m={_fmt_xyz(object_size)} yaw_deg={plan.yaw_deg:.1f} "
        f"normalized=([{plan.normalized_grasp[0]:+.3f},{plan.normalized_grasp[1]:+.3f},{plan.normalized_grasp[2]:+.3f}]) "
        f"approach_tcp={_fmt_xyz(plan.approach_tcp)} descend_tcp={_fmt_xyz(plan.descend_tcp)} "
        f"close_deg={args.close_deg[0]:.2f} ik_ok={_yes(plan.approach_ik_ok and plan.descend_ik_ok)} "
        f"ik_err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f}) "
        f"max_fk_error_m={plan.max_fk_error_m:.6f}",
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
    cfg.sponge.spawn = sim_utils.CuboidCfg(
        size=tuple(float(x) for x in object_size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.2, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.70, 1.00), metallic=0.0),
    )
    cfg.sponge.init_state.pos = tuple(float(x) for x in plan.center)
    cfg.sponge.init_state.rot = tuple(float(x) for x in _yaw_quat_wxyz(plan.yaw_deg))

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)

    attach_stats = {"attach_calls": 0, "posewrite_calls": 0}
    original_set_joint_position_target = base_env._robot.set_joint_position_target
    original_write_root_pose_to_sim = base_env._sponge.write_root_pose_to_sim
    watch = {"active": False, "target": None, "calls": 0, "max_diff": 0.0}
    posewrite_watch = {"active": False}

    def marker_only_attach() -> None:
        attach_stats["attach_calls"] += 1

    def watched_set_joint_position_target(target, *a, **kw):
        if watch["active"] and watch["target"] is not None:
            arr = target.detach().cpu().numpy().astype(np.float64)
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], float(np.max(np.abs(arr - watch["target"]))))
        return original_set_joint_position_target(target, *a, **kw)

    def watched_write_root_pose_to_sim(*a, **kw):
        if posewrite_watch["active"]:
            attach_stats["posewrite_calls"] += 1
        return original_write_root_pose_to_sim(*a, **kw)

    base_env._update_grasp_attach = marker_only_attach
    base_env._robot.set_joint_position_target = watched_set_joint_position_target
    base_env._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

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

    def object_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def object_quat() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def object_vel6() -> np.ndarray:
        return base_env._sponge.data.root_vel_w[0].detach().cpu().numpy().astype(np.float64)

    def gripper_link_transform_local() -> np.ndarray:
        pos = (base_env._robot.data.body_pos_w[0, base_env.gripper_link_idx] - base_env.scene.env_origins[0]).detach().cpu().numpy()
        quat = base_env._robot.data.body_quat_w[0, base_env.gripper_link_idx].detach().cpu().numpy()
        return _homogeneous(_quat_wxyz_to_rot(quat.astype(np.float64)), pos.astype(np.float64))

    def link5_transform_local() -> np.ndarray:
        pos = (base_env._robot.data.body_pos_w[0, base_env.link5_idx] - base_env.scene.env_origins[0]).detach().cpu().numpy()
        quat = base_env._robot.data.body_quat_w[0, base_env.link5_idx].detach().cpu().numpy()
        return _homogeneous(_quat_wxyz_to_rot(quat.astype(np.float64)), pos.astype(np.float64))

    def write_object_pose() -> None:
        yaw_q = _yaw_quat_wxyz(plan.yaw_deg)
        pose = torch.tensor(
            [[plan.center[0], plan.center[1], plan.center[2], yaw_q[0], yaw_q[1], yaw_q[2], yaw_q[3]]],
            device=device,
            dtype=torch.float32,
        )
        pose[:, 0:3] += base_env.scene.env_origins[:1]
        original_write_root_pose_to_sim(pose)
        base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        base_env.scene.write_data_to_sim()
        base_env.scene.update(base_env.sim.get_physics_dt())

    env.reset()
    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env._robot.set_joint_position_target(home_rad)
    base_env.robot_dof_targets[:] = home_rad
    base_env._grasped[:] = False
    base_env._was_grasped[:] = False
    write_object_pose()

    episode_done = False
    nan_seen = False
    for _ in range(args.initial_settle_steps):
        episode_done |= step_once()

    initial_object = object_local()
    if _norm(initial_object - plan.center) > args.target_error_gate_m:
        old_center = plan.center.copy()
        plan = _build_plan_from_center(args, initial_object.copy(), f"{plan.label}_settled_pose")
        print(
            f"[cube2cm_runtime_jaw_telemetry] settled_pose_replan=YES requested_center={_fmt_xyz(old_center)} "
            f"settled_center={_fmt_xyz(initial_object)} updated_descend_tcp={_fmt_xyz(plan.descend_tcp)}",
            flush=True,
        )
    else:
        print("[cube2cm_runtime_jaw_telemetry] settled_pose_replan=NO", flush=True)

    jaw = _jaw_geometry(args, plan)
    print(
        f"[cube2cm_runtime_jaw_telemetry] authored_jaw_geometry variant={args.variant} "
        f"moving_size_m={_fmt_xyz(jaw.moving_size_m)} counter_size_m={_fmt_xyz(jaw.counter_size_m)} "
        f"design_moving_center_ref={_fmt_xyz(jaw.design_moving_center_ref)} "
        f"design_counter_center_ref={_fmt_xyz(jaw.design_counter_center_ref)} "
        f"counter_parent={jaw.counter_parent} counter_origin_parent_m={_fmt_xyz(jaw.counter_origin_parent_m)} "
        f"counter_contact_slop_m={float(args.counter_contact_slop_m):.6f}",
        flush=True,
    )
    posewrite_watch["active"] = True

    def run_to_q(label: str, q_deg: np.ndarray, target_tcp: np.ndarray, max_steps: int, phase: str) -> tuple[bool, bool]:
        nonlocal episode_done, nan_seen
        target_rad_np = np.radians(q_deg)
        target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
        start_object = object_local()
        prev_tcp = fresh_tcp_local()
        settle_count = 0
        reached = False
        early_kill = False
        watch["active"] = True
        watch["target"] = target_rad_np
        watch["calls"] = 0
        watch["max_diff"] = 0.0
        for step_idx in range(1, max_steps + 1):
            base_env.robot_dof_targets[:] = target_rad
            done = step_once()
            episode_done |= done
            tcp = fresh_tcp_local()
            obj = object_local()
            quat = object_quat()
            vel = object_vel6()
            q_actual_rad = base_env._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
            q_actual_deg = np.degrees(q_actual_rad)
            gripper_q = float(q_actual_rad[base_env.gripper_joint_idx])
            gripper_err = abs(gripper_q - float(target_rad_np[5]))
            target_error = _norm(tcp - target_tcp)
            tcp_step = _norm(tcp - prev_tcp)
            drift_vec = obj - start_object
            drift = _norm(drift_vec)
            speed = _norm(vel[:3])

            gripper_tf = gripper_link_transform_local()
            counter_parent_tf = link5_transform_local() if jaw.counter_parent == "link5" else gripper_tf
            moving_world = _transform_points(gripper_tf, jaw.moving_vertices_local)
            counter_world = _transform_points(
                counter_parent_tf @ _translation(jaw.counter_origin_parent_m),
                jaw.counter_vertices_local,
            )
            moving = _contact_stats(moving_world, obj, quat, object_size, slop_m=float(args.counter_contact_slop_m))
            counter = _contact_stats(counter_world, obj, quat, object_size, slop_m=float(args.counter_contact_slop_m))
            moving_overlap = np.asarray(moving["overlap_obj"], dtype=np.float64)
            counter_overlap = np.asarray(counter["overlap_obj"], dtype=np.float64)
            moving_slop_overlap = np.asarray(moving["slop_overlap_obj"], dtype=np.float64)
            counter_slop_overlap = np.asarray(counter["slop_overlap_obj"], dtype=np.float64)
            moving_gap = np.asarray(moving["gap_obj"], dtype=np.float64)
            counter_gap = np.asarray(counter["gap_obj"], dtype=np.float64)
            moving_center = np.asarray(moving["center_obj"], dtype=np.float64)
            counter_center = np.asarray(counter["center_obj"], dtype=np.float64)
            contact_count = int(bool(moving["contact"])) + int(bool(counter["contact"]))
            one_sided_contact = contact_count == 1
            push_started = drift > args.push_drift_gate_m or speed > args.push_speed_gate_mps
            one_sided_push = push_started and one_sided_contact

            early_kill = (
                tcp_step > args.max_tcp_step_m
                or speed > args.object_speed_gate_mps
                or drift > args.object_drift_gate_m
                or done
                or not np.isfinite(tcp).all()
                or not np.isfinite(obj).all()
                or not math.isfinite(target_error)
            )
            nan_seen |= not np.isfinite(tcp).all() or not np.isfinite(obj).all() or not math.isfinite(target_error)
            reached = target_error <= args.target_error_gate_m
            if phase == "close":
                reached = reached and gripper_err <= math.radians(args.gripper_error_gate_deg)
            settle_count = settle_count + 1 if reached else 0

            should_log = (
                phase == "close"
                and (args.log_every_close_step <= 1 or step_idx % args.log_every_close_step == 0)
            ) or step_idx <= 3 or reached or early_kill
            if should_log:
                print(
                    f"[cube2cm_runtime_jaw_telemetry] step label={label} phase={phase} step={step_idx:03d} "
                    f"target_tcp={_fmt_xyz(target_tcp)} tcp={_fmt_xyz(tcp)} target_error_m={target_error:.6f} "
                    f"tcp_step_m={tcp_step:.6f} q_deg={_fmt_deg(q_actual_deg)} "
                    f"gripper_q_deg={math.degrees(gripper_q):.3f} gripper_err_deg={math.degrees(gripper_err):.3f} "
                    f"object_pos={_fmt_xyz(obj)} object_drift_vec_m={_fmt_xyz(drift_vec)} object_drift_m={drift:.6f} "
                    f"object_speed_mps={speed:.6f} moving_center_obj={_fmt_xyz(moving_center)} "
                    f"counter_center_obj={_fmt_xyz(counter_center)} moving_overlap_obj_m={_fmt_xyz(moving_overlap)} "
                    f"counter_overlap_obj_m={_fmt_xyz(counter_overlap)} moving_gap_obj_m={_fmt_xyz(moving_gap)} "
                    f"counter_gap_obj_m={_fmt_xyz(counter_gap)} moving_slop_overlap_obj_m={_fmt_xyz(moving_slop_overlap)} "
                    f"counter_slop_overlap_obj_m={_fmt_xyz(counter_slop_overlap)} "
                    f"moving_contact={_yes(bool(moving['contact']))} counter_contact={_yes(bool(counter['contact']))} "
                    f"moving_slop_contact={_yes(bool(moving['slop_contact']))} "
                    f"counter_slop_contact={_yes(bool(counter['slop_contact']))} "
                    f"one_sided_push={_yes(one_sided_push)} "
                    f"_grasped_marker={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
                    f"attach_calls_total={attach_stats['attach_calls']} posewrite_calls_total={attach_stats['posewrite_calls']} "
                    f"set_target_seen={_yes(watch['calls'] > 0 and watch['max_diff'] <= 1.0e-5)} "
                    f"set_max_diff_rad={watch['max_diff']:.8f} reached={_yes(reached)} early_kill={_yes(early_kill)}",
                    flush=True,
                )
            prev_tcp = tcp
            if early_kill or settle_count >= args.settle_steps:
                break
        watch["active"] = False
        return reached, early_kill

    def resampled_waypoints(start_tcp: np.ndarray, end_tcp: np.ndarray) -> list[np.ndarray]:
        delta = np.asarray(end_tcp, dtype=np.float64) - np.asarray(start_tcp, dtype=np.float64)
        gap = _norm(delta)
        max_cmd_gap = args.max_tcp_step_m * args.command_resample_fraction
        count = max(1, int(math.ceil(gap / max_cmd_gap)))
        return [np.asarray(start_tcp, dtype=np.float64) + delta * (i / count) for i in range(1, count + 1)]

    def run_path(label: str, end_tcp: np.ndarray, seed_q: np.ndarray, gripper_deg: float, phase: str) -> tuple[np.ndarray, bool]:
        q_seed = seed_q.copy()
        for idx, waypoint in enumerate(resampled_waypoints(fresh_tcp_local(), end_tcp), start=1):
            q_step, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, gripper_deg, args)
            print(
                f"[cube2cm_runtime_jaw_telemetry] path_waypoint label={label} index={idx:03d} "
                f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f}",
                flush=True,
            )
            if not ik_ok or _norm(fk_tcp(q_step) - waypoint) > args.target_error_gate_m:
                return q_seed, False
            reached, early_kill = run_to_q(f"{label}_wp{idx:03d}", q_step, waypoint, args.substep_steps, phase)
            q_seed = q_step
            if not reached or early_kill:
                return q_seed, False
        return q_seed, True

    current_seed_q = HOME_DEG.copy()
    if not (plan.approach_ik_ok and plan.descend_ik_ok and plan.max_fk_error_m <= args.target_error_gate_m):
        print("[cube2cm_runtime_jaw_telemetry] TELEMETRY_ABORT=REACH_PLAN_FAIL", flush=True)
        env.close()
        sim_app.close()
        return 2

    current_seed_q, approach_ok = run_path("approach_open", plan.approach_tcp, current_seed_q, GRIPPER_OPEN_DEG, "approach")
    descend_ok = False
    if approach_ok:
        current_seed_q, descend_ok = run_path("descend_open", plan.descend_tcp, current_seed_q, GRIPPER_OPEN_DEG, "descend")

    close_reached = False
    close_early_kill = False
    if descend_ok:
        q_close = current_seed_q.copy()
        q_close[4] = PICK_WRIST_R_DEG
        q_close[5] = float(args.close_deg[0])
        close_reached, close_early_kill = run_to_q(
            f"close_{args.close_deg[0]:.2f}deg_runtime_jaw_telemetry",
            q_close,
            plan.descend_tcp,
            args.close_steps,
            "close",
        )

    print(
        f"[cube2cm_runtime_jaw_telemetry] aggregate variant={args.variant} approach_ok={_yes(approach_ok)} "
        f"descend_ok={_yes(descend_ok)} close_reached={_yes(close_reached)} close_early_kill={_yes(close_early_kill)} "
        f"grasped_seen={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
        f"attach_calls={attach_stats['attach_calls']} posewrite_calls={attach_stats['posewrite_calls']} "
        f"episode_done={_yes(episode_done)} nan_seen={_yes(nan_seen)} "
        "telemetry_only=YES success_claim=NO",
        flush=True,
    )
    print("[cube2cm_runtime_jaw_telemetry] CUBE2CM_RUNTIME_JAW_TELEMETRY_DONE=YES", flush=True)
    env.close()
    sim_app.close()
    return 0 if not nan_seen else 2


if __name__ == "__main__":
    raise SystemExit(main())
