#!/usr/bin/env python3
"""2cm cube local grasp/close sweep probe for P7 Branch B.

Diagnostic-only Isaac script. It changes only this script's env config instance to
spawn a 2cm cube, runs no training, inserts no constraints, uses no SurfaceGripper,
executes no transport target or release, and refuses to count the env's hidden
kinematic pose-write attach path as physical grasp evidence.
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

from roarm_kinematics import clip_joints, fk_tcp, ik_dls  # noqa: E402

TABLE_Z = -0.012117
HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float64)
PICK_WRIST_R_DEG = 90.0
GRIPPER_OPEN_DEG = 0.0
GRASP_GRIPPER_THRESH_RAD = 0.4

SOURCE_REGIONS = (
    (0.150, 0.250, -0.220, -0.130),
    (0.150, 0.250, +0.070, +0.200),
    (0.330, 0.430, -0.220, -0.100),
    (0.330, 0.430, +0.050, +0.200),
)
FOUR_SPONGE_SEED0_SOURCES = {
    "seed0_S1": (+0.21369616873214542, -0.19571919576125169),
    "seed0_S2": (+0.15165276355285290, +0.17572513109603544),
    "seed0_S3": (+0.39066357757671800, -0.13246041268192021),
    "seed0_S4": (+0.42350724237877680, +0.17237803311822986),
}
NAMED_GRASPS = {
    "top_center": np.array([0.0, 0.0, 0.5], dtype=np.float64),
    "top_pos_x": np.array([0.35, 0.0, 0.5], dtype=np.float64),
    "top_neg_x": np.array([-0.35, 0.0, 0.5], dtype=np.float64),
    "top_pos_y": np.array([0.0, 0.35, 0.5], dtype=np.float64),
    "top_neg_y": np.array([0.0, -0.35, 0.5], dtype=np.float64),
}


@dataclass(frozen=True)
class PosePlan:
    label: str
    center: np.ndarray
    yaw_deg: float
    grasp_name: str
    normalized_grasp: np.ndarray
    world_grasp: np.ndarray
    approach_tcp: np.ndarray
    descend_tcp: np.ndarray
    lift_tcp: np.ndarray
    q_approach_deg: np.ndarray
    q_descend_deg: np.ndarray
    q_lift_deg: np.ndarray
    approach_ik_ok: bool
    descend_ik_ok: bool
    lift_ik_ok: bool
    approach_ik_err_mm: float
    descend_ik_err_mm: float
    lift_ik_err_mm: float
    max_fk_error_m: float
    max_raw_tcp_gap_m: float


@dataclass
class StepResult:
    label: str
    reached: bool
    steps: int
    final_target_error_m: float
    max_tcp_step_m: float
    max_object_drift_m: float
    max_object_speed_mps: float
    max_tilt_deg: float
    min_upright_z: float
    object_follow_delta_m: float
    grasped_seen: bool
    attach_calls: int
    posewrite_calls: int
    early_kill: bool


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _fmt_xyz(value: np.ndarray) -> str:
    return f"([{value[0]:+.6f}, {value[1]:+.6f}, {value[2]:+.6f}])"


def _fmt_quat(value: np.ndarray) -> str:
    q = np.asarray(value, dtype=np.float64)
    return f"[w={q[0]:+.6f}, x={q[1]:+.6f}, y={q[2]:+.6f}, z={q[3]:+.6f}]"


def _rot_z(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _yaw_quat_wxyz(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


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


def _object_pose_metrics(pos: np.ndarray, quat: np.ndarray, object_size: np.ndarray) -> dict[str, float]:
    rot = _quat_wxyz_to_rot(quat)
    half_extents = object_size / 2.0
    oriented_half_height = float(np.dot(np.abs(rot[2, :]), half_extents))
    up_z = float(rot[2, 2])
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    return {
        "up_z": up_z,
        "tilt_deg": tilt_deg,
        "oriented_top_z_m": float(pos[2] + oriented_half_height),
        "center_z_m": float(pos[2]),
    }


def _solve_q(target_tcp: np.ndarray, seed_q: np.ndarray, gripper_deg: float, args: argparse.Namespace) -> tuple[np.ndarray, bool, float]:
    q, converged, err_mm, _n_iter = ik_dls(
        target_tcp,
        seed_q,
        max_iter=args.ik_max_iter,
        tol_mm=args.ik_tol_mm,
    )
    q = clip_joints(q)
    q[4] = PICK_WRIST_R_DEG
    q[5] = gripper_deg
    return q, bool(converged), float(err_mm)


def _workspace_xy_from_label(label: str) -> tuple[float, float]:
    if label in FOUR_SPONGE_SEED0_SOURCES:
        return FOUR_SPONGE_SEED0_SOURCES[label]
    if label.startswith("R") and "_center" in label:
        region_idx = int(label[1]) - 1
        x_min, x_max, y_min, y_max = SOURCE_REGIONS[region_idx]
        return (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    raise ValueError(f"unknown pose_label={label!r}; use seed0_S1..seed0_S4 or R1_center..R4_center")


def _build_plan_from_center(args: argparse.Namespace, center: np.ndarray, label: str) -> PosePlan:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if args.normalized_grasp is None:
        normalized = NAMED_GRASPS[args.grasp_name].copy()
        grasp_name = args.grasp_name
    else:
        normalized = np.asarray(args.normalized_grasp, dtype=np.float64)
        grasp_name = "custom_normalized"
    if normalized.shape != (3,) or np.any(normalized < -0.5) or np.any(normalized > 0.5):
        raise ValueError("normalized_grasp must be three values in [-0.5, +0.5]")

    world_grasp = center + _rot_z(args.yaw_deg) @ (normalized * object_size)
    approach_tcp = world_grasp + np.array([0.0, 0.0, args.approach_clearance_m], dtype=np.float64)
    descend_tcp = world_grasp + np.array([0.0, 0.0, args.grasp_surface_margin_m], dtype=np.float64)
    lift_tcp = descend_tcp + np.array([0.0, 0.0, args.lift_delta_m], dtype=np.float64)

    seed = HOME_DEG.copy()
    seed[5] = GRIPPER_OPEN_DEG
    q_approach, approach_ok, approach_err = _solve_q(approach_tcp, seed, GRIPPER_OPEN_DEG, args)
    q_descend, descend_ok, descend_err = _solve_q(descend_tcp, q_approach, GRIPPER_OPEN_DEG, args)
    q_lift, lift_ok, lift_err = _solve_q(lift_tcp, q_descend, args.close_deg[-1], args)
    max_fk_error = max(
        _norm(fk_tcp(q_approach) - approach_tcp),
        _norm(fk_tcp(q_descend) - descend_tcp),
        _norm(fk_tcp(q_lift) - lift_tcp),
    )
    max_gap = max(_norm(approach_tcp - descend_tcp), _norm(lift_tcp - descend_tcp))

    return PosePlan(
        label=label,
        center=center,
        yaw_deg=float(args.yaw_deg),
        grasp_name=grasp_name,
        normalized_grasp=normalized,
        world_grasp=world_grasp,
        approach_tcp=approach_tcp,
        descend_tcp=descend_tcp,
        lift_tcp=lift_tcp,
        q_approach_deg=q_approach,
        q_descend_deg=q_descend,
        q_lift_deg=q_lift,
        approach_ik_ok=approach_ok,
        descend_ik_ok=descend_ok,
        lift_ik_ok=lift_ok,
        approach_ik_err_mm=approach_err,
        descend_ik_err_mm=descend_err,
        lift_ik_err_mm=lift_err,
        max_fk_error_m=max_fk_error,
        max_raw_tcp_gap_m=max_gap,
    )


def _build_plan(args: argparse.Namespace) -> PosePlan:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if args.object_xy is None:
        x, y = _workspace_xy_from_label(args.pose_label)
        label = args.pose_label
    else:
        x, y = args.object_xy
        label = "custom_xy"
    center = np.array([x, y, TABLE_Z + object_size[2] / 2.0], dtype=np.float64)
    return _build_plan_from_center(args, center, label)


def _verdict(
    plan: PosePlan,
    approach: StepResult | None,
    descend: StepResult | None,
    latch: StepResult | None,
    hold: StepResult | None,
    lift: StepResult | None,
    args: argparse.Namespace,
) -> str:
    if (
        not plan.approach_ik_ok
        or not plan.descend_ik_ok
        or not plan.lift_ik_ok
        or plan.max_fk_error_m > args.target_error_gate_m
    ):
        return "REACH_FAIL"
    if approach is None or not approach.reached or approach.early_kill:
        return "APPROACH_FAIL"
    if descend is None or not descend.reached or descend.early_kill:
        return "APPROACH_FAIL"
    if latch is None or not latch.reached or latch.early_kill or not latch.grasped_seen:
        return "LATCH_FAIL"
    if hold is None or not hold.reached or hold.early_kill:
        return "HOLD_FAIL"
    if lift is None or not lift.reached or lift.early_kill or lift.object_follow_delta_m < args.min_lift_follow_m:
        return "LIFT_FAIL"
    return "GRASP_PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.020, 0.020, 0.020])
    ap.add_argument("--object_mass_kg", type=float, default=0.02)
    ap.add_argument("--pose_label", default="seed0_S1")
    ap.add_argument("--object_xy", nargs=2, type=float, default=None)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    ap.add_argument("--grasp_name", choices=sorted(NAMED_GRASPS), default="top_center")
    ap.add_argument("--normalized_grasp", nargs=3, type=float, default=None)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument("--close_deg", nargs="+", type=float, default=[23.0, 26.0, 30.0, 35.0, 40.0, 45.84])
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--command_resample_fraction", type=float, default=0.80)
    ap.add_argument("--substep_steps", type=int, default=60)
    ap.add_argument("--object_drift_gate_m", type=float, default=0.006)
    ap.add_argument("--object_speed_gate_mps", type=float, default=0.080)
    ap.add_argument("--lift_speed_gate_mps", type=float, default=0.250)
    ap.add_argument("--gripper_error_gate_deg", type=float, default=0.75)
    ap.add_argument("--tilt_gate_deg", type=float, default=12.0)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.95)
    ap.add_argument("--min_lift_follow_m", type=float, default=0.006)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=30)
    ap.add_argument("--approach_steps", type=int, default=80)
    ap.add_argument("--descend_steps", type=int, default=80)
    ap.add_argument("--close_steps_per_angle", type=int, default=45)
    ap.add_argument("--hold_steps", type=int, default=30)
    ap.add_argument("--lift_steps", type=int, default=80)
    ap.add_argument(
        "--continue_close_after_grasped_until_angles_done",
        action="store_true",
        help=(
            "Diagnostic-only: keep logging later close_deg values after _grasped_marker "
            "appears. Early-kill still stops immediately, and hold/lift gates are unchanged."
        ),
    )
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--ik_tol_mm", type=float, default=0.75)
    ap.add_argument("--ik_max_iter", type=int, default=240)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")
    if sorted(args.close_deg) != list(args.close_deg):
        raise ValueError("close_deg values must be sorted ascending")
    if math.radians(args.close_deg[0]) < GRASP_GRIPPER_THRESH_RAD:
        raise ValueError("first close angle must be at or above grasp_gripper_thresh")
    if args.approach_clearance_m <= args.grasp_surface_margin_m:
        raise ValueError("approach_clearance_m must be above grasp_surface_margin_m")
    if args.lift_delta_m <= 0.0:
        raise ValueError("lift_delta_m must be positive")
    if args.command_resample_fraction <= 0.0 or args.command_resample_fraction > 1.0:
        raise ValueError("command_resample_fraction must be in (0, 1]")

    plan = _build_plan(args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    print("[cube2cm_local_grasp] 2cm cube Isaac local grasp/close sweep probe", flush=True)
    print(
        "[cube2cm_local_grasp] "
        "diagnostic_only=YES isaac_run=YES object_size_m="
        f"({_fmt_xyz(object_size)}) env_default_edits=NO chain_defaults_edits=NO p7_training=NO "
        "constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO "
        "transport_target=NO release_marker=NO scripted_release_variant=NO "
        "hidden_kinematic_posewrite_allowed=NO claim_p7_success=NO",
        flush=True,
    )
    print(
        f"[cube2cm_local_grasp] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} object_drift_gate_m={args.object_drift_gate_m:.6f} "
        f"command_resample_fraction={args.command_resample_fraction:.3f} substep_steps={args.substep_steps} "
        f"object_speed_gate_mps={args.object_speed_gate_mps:.6f} lift_speed_gate_mps={args.lift_speed_gate_mps:.6f} "
        f"gripper_error_gate_deg={args.gripper_error_gate_deg:.3f} "
        f"tilt_gate_deg={args.tilt_gate_deg:.2f} min_upright_z_gate={args.min_upright_z_gate:.3f} "
        f"min_lift_follow_m={args.min_lift_follow_m:.6f} close_sweep_deg={','.join(f'{x:.2f}' for x in args.close_deg)} "
        f"continue_close_after_grasped_until_angles_done={_yes(args.continue_close_after_grasped_until_angles_done)}",
        flush=True,
    )
    print(
        f"[cube2cm_local_grasp] selected pose={plan.label} center={_fmt_xyz(plan.center)} yaw_deg={plan.yaw_deg:.1f} "
        f"grasp={plan.grasp_name} normalized=([{plan.normalized_grasp[0]:+.3f},{plan.normalized_grasp[1]:+.3f},{plan.normalized_grasp[2]:+.3f}]) "
        f"world_grasp={_fmt_xyz(plan.world_grasp)} approach_tcp={_fmt_xyz(plan.approach_tcp)} "
        f"descend_tcp={_fmt_xyz(plan.descend_tcp)} lift_tcp={_fmt_xyz(plan.lift_tcp)} "
        f"ik_ok={_yes(plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok)} "
        f"ik_err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
        f"max_fk_error_m={plan.max_fk_error_m:.6f} max_raw_tcp_gap_m={plan.max_raw_tcp_gap_m:.6f} "
        f"resample_needed={_yes(plan.max_raw_tcp_gap_m > args.max_tcp_step_m)}",
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
    watch = {"active": False, "target": None, "calls": 0, "max_diff": 0.0}

    def marker_only_attach() -> None:
        attach_stats["attach_calls"] += 1

    def watched_set_joint_position_target(target, *a, **kw):
        if watch["active"] and watch["target"] is not None:
            arr = target.detach().cpu().numpy().astype(np.float64)
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], float(np.max(np.abs(arr - watch["target"]))))
        return original_set_joint_position_target(target, *a, **kw)

    base_env._update_grasp_attach = marker_only_attach
    base_env._robot.set_joint_position_target = watched_set_joint_position_target

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

    def write_object_pose() -> None:
        yaw_q = _yaw_quat_wxyz(plan.yaw_deg)
        pose = torch.tensor(
            [[plan.center[0], plan.center[1], plan.center[2], yaw_q[0], yaw_q[1], yaw_q[2], yaw_q[3]]],
            device=device,
            dtype=torch.float32,
        )
        pose[:, 0:3] += base_env.scene.env_origins[:1]
        base_env._sponge.write_root_pose_to_sim(pose)
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

    original_write_root_pose_to_sim = base_env._sponge.write_root_pose_to_sim
    posewrite_watch = {"active": False}

    def watched_write_root_pose_to_sim(*a, **kw):
        if posewrite_watch["active"]:
            attach_stats["posewrite_calls"] += 1
        return original_write_root_pose_to_sim(*a, **kw)

    base_env._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

    total_sim_steps = 0
    episode_done = False
    nan_seen = False
    for _ in range(args.initial_settle_steps):
        episode_done |= step_once()
        total_sim_steps += 1

    initial_object = object_local()
    initial_quat = object_quat()
    initial_metrics = _object_pose_metrics(initial_object, initial_quat, object_size)
    print(
        f"[cube2cm_local_grasp] initial home_tcp={_fmt_xyz(fresh_tcp_local())} object_pos={_fmt_xyz(initial_object)} "
        f"object_quat_wxyz={_fmt_quat(initial_quat)} object_top_z_m={initial_metrics['oriented_top_z_m']:.6f} "
        f"upright_z={initial_metrics['up_z']:.6f} tilt_deg={initial_metrics['tilt_deg']:.3f}",
        flush=True,
    )
    if _norm(initial_object - plan.center) > args.target_error_gate_m:
        old_center = plan.center.copy()
        plan = _build_plan_from_center(args, initial_object.copy(), f"{plan.label}_settled_pose")
        print(
            f"[cube2cm_local_grasp] settled_pose_replan=YES requested_center={_fmt_xyz(old_center)} "
            f"settled_center={_fmt_xyz(initial_object)} settled_top_z_m={initial_metrics['oriented_top_z_m']:.6f} "
            f"updated_world_grasp={_fmt_xyz(plan.world_grasp)} updated_approach_tcp={_fmt_xyz(plan.approach_tcp)} "
            f"updated_descend_tcp={_fmt_xyz(plan.descend_tcp)} updated_lift_tcp={_fmt_xyz(plan.lift_tcp)} "
            f"updated_ik_ok={_yes(plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok)} "
            f"updated_ik_err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
            f"updated_max_fk_error_m={plan.max_fk_error_m:.6f}",
            flush=True,
        )
    else:
        print("[cube2cm_local_grasp] settled_pose_replan=NO", flush=True)
    posewrite_watch["active"] = True

    def run_to_q(label: str, q_deg: np.ndarray, target_tcp: np.ndarray, max_steps: int, phase: str) -> StepResult:
        nonlocal total_sim_steps, episode_done, nan_seen
        target_rad_np = np.radians(q_deg)
        target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
        start_object = object_local()
        start_lift_ref = start_object.copy()
        prev_tcp = fresh_tcp_local()
        settle_count = 0
        reached = False
        early_kill = False
        steps_used = 0
        final_error = float("inf")
        max_tcp_step = 0.0
        max_drift = 0.0
        max_speed = 0.0
        max_tilt = 0.0
        min_upright = 1.0
        grasped_seen = bool(base_env._grasped[0].detach().cpu().item())
        attach_start = attach_stats["attach_calls"]
        posewrite_start = attach_stats["posewrite_calls"]
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
            obj = object_local()
            quat = object_quat()
            vel = object_vel6()
            metrics = _object_pose_metrics(obj, quat, object_size)
            gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())
            gripper_err = abs(gripper_q - float(target_rad_np[5]))
            target_error = _norm(tcp - target_tcp)
            tcp_step = _norm(tcp - prev_tcp)
            drift = _norm(obj - start_object)
            speed = _norm(vel[:3])
            tilt = float(metrics["tilt_deg"])
            upright = float(metrics["up_z"])
            max_tcp_step = max(max_tcp_step, tcp_step)
            max_drift = max(max_drift, drift)
            max_speed = max(max_speed, speed)
            max_tilt = max(max_tilt, tilt)
            min_upright = min(min_upright, upright)
            final_error = target_error
            grasped_seen = grasped_seen or bool(base_env._grasped[0].detach().cpu().item())
            if not np.isfinite(tcp).all() or not np.isfinite(obj).all() or not math.isfinite(target_error):
                nan_seen = True
            episode_done |= done
            speed_gate = args.lift_speed_gate_mps if phase == "lift" else args.object_speed_gate_mps
            drift_gate = args.object_drift_gate_m
            if phase == "lift":
                drift_gate = max(args.object_drift_gate_m, args.lift_delta_m + 0.010)
            early_kill = (
                tcp_step > args.max_tcp_step_m
                or speed > speed_gate
                or tilt > args.tilt_gate_deg
                or upright < args.min_upright_z_gate
                or drift > drift_gate
                or done
                or nan_seen
            )
            reached = target_error <= args.target_error_gate_m
            if phase == "close":
                reached = reached and gripper_err <= math.radians(args.gripper_error_gate_deg)
            settle_count = settle_count + 1 if reached else 0
            if step_idx <= 3 or step_idx == max_steps or reached or early_kill or (args.log_every > 0 and step_idx % args.log_every == 0):
                print(
                    f"[cube2cm_local_grasp] event label={label} phase={phase} step={step_idx:03d} "
                    f"target_tcp={_fmt_xyz(target_tcp)} fresh_tcp={_fmt_xyz(tcp)} target_error_m={target_error:.6f} "
                    f"tcp_step_m={tcp_step:.6f} object_pos={_fmt_xyz(obj)} object_drift_m={drift:.6f} "
                    f"object_speed_mps={speed:.6f} upright_z={upright:.6f} tilt_deg={tilt:.3f} "
                    f"gripper_q_deg={math.degrees(gripper_q):.3f} gripper_err_deg={math.degrees(gripper_err):.3f} "
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
        obj_end = object_local()
        return StepResult(
            label=label,
            reached=reached,
            steps=steps_used,
            final_target_error_m=final_error,
            max_tcp_step_m=max_tcp_step,
            max_object_drift_m=max_drift,
            max_object_speed_mps=max_speed,
            max_tilt_deg=max_tilt,
            min_upright_z=min_upright,
            object_follow_delta_m=float(obj_end[2] - start_lift_ref[2]),
            grasped_seen=grasped_seen,
            attach_calls=attach_stats["attach_calls"] - attach_start,
            posewrite_calls=attach_stats["posewrite_calls"] - posewrite_start,
            early_kill=early_kill,
        )

    def resampled_waypoints(start_tcp: np.ndarray, end_tcp: np.ndarray) -> list[np.ndarray]:
        delta = np.asarray(end_tcp, dtype=np.float64) - np.asarray(start_tcp, dtype=np.float64)
        gap = _norm(delta)
        max_cmd_gap = args.max_tcp_step_m * args.command_resample_fraction
        count = max(1, int(math.ceil(gap / max_cmd_gap)))
        return [np.asarray(start_tcp, dtype=np.float64) + delta * (i / count) for i in range(1, count + 1)]

    def run_resampled_path(
        label: str,
        start_tcp: np.ndarray,
        end_tcp: np.ndarray,
        seed_q: np.ndarray,
        gripper_deg: float,
        phase: str,
        max_steps: int,
    ) -> tuple[StepResult | None, np.ndarray, bool]:
        q_seed = seed_q.copy()
        final_result: StepResult | None = None
        waypoints = resampled_waypoints(start_tcp, end_tcp)
        print(
            f"[cube2cm_local_grasp] path_plan label={label} start_tcp={_fmt_xyz(start_tcp)} "
            f"end_tcp={_fmt_xyz(end_tcp)} waypoints={len(waypoints)} "
            f"max_command_gap_m={args.max_tcp_step_m * args.command_resample_fraction:.6f}",
            flush=True,
        )
        for idx, waypoint in enumerate(waypoints, start=1):
            q_step, ik_ok, ik_err_mm = _solve_q(waypoint, q_seed, gripper_deg, args)
            print(
                f"[cube2cm_local_grasp] path_waypoint label={label} index={idx:03d}/{len(waypoints):03d} "
                f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f}",
                flush=True,
            )
            if not ik_ok or _norm(fk_tcp(q_step) - waypoint) > args.target_error_gate_m:
                return final_result, q_seed, False
            final_result = run_to_q(f"{label}_wp{idx:03d}", q_step, waypoint, max_steps, phase)
            q_seed = q_step
            if not final_result.reached or final_result.early_kill:
                return final_result, q_seed, False
        return final_result, q_seed, True

    approach_result: StepResult | None = None
    descend_result: StepResult | None = None
    latch_result: StepResult | None = None
    hold_result: StepResult | None = None
    lift_result: StepResult | None = None

    current_seed_q = HOME_DEG.copy()
    if plan.approach_ik_ok and plan.max_fk_error_m <= args.target_error_gate_m:
        approach_result, current_seed_q, approach_path_ok = run_resampled_path(
            "approach_open",
            fresh_tcp_local(),
            plan.approach_tcp,
            current_seed_q,
            GRIPPER_OPEN_DEG,
            "approach",
            args.substep_steps,
        )
    else:
        approach_path_ok = False
    if approach_path_ok and approach_result and approach_result.reached and not approach_result.early_kill:
        descend_result, current_seed_q, descend_path_ok = run_resampled_path(
            "descend_open",
            fresh_tcp_local(),
            plan.descend_tcp,
            current_seed_q,
            GRIPPER_OPEN_DEG,
            "descend",
            args.substep_steps,
        )
    else:
        descend_path_ok = False
    if descend_result and descend_result.reached and not descend_result.early_kill:
        q_close = current_seed_q.copy()
        for close_deg in args.close_deg:
            q_close[5] = close_deg
            result = run_to_q(f"close_{close_deg:.2f}deg", q_close, plan.descend_tcp, args.close_steps_per_angle, "close")
            print(
                f"[cube2cm_local_grasp] close_result angle_deg={close_deg:.2f} reached={_yes(result.reached)} "
                f"grasped_seen={_yes(result.grasped_seen)} final_target_error_m={result.final_target_error_m:.6f} "
                f"object_drift_m={result.max_object_drift_m:.6f} object_speed_mps={result.max_object_speed_mps:.6f} "
                f"tilt_deg={result.max_tilt_deg:.3f} attach_calls={result.attach_calls} posewrite_calls={result.posewrite_calls} "
                f"early_kill={_yes(result.early_kill)}",
                flush=True,
            )
            latch_result = result
            if result.early_kill:
                break
            if result.grasped_seen and not args.continue_close_after_grasped_until_angles_done:
                break
    if latch_result and latch_result.reached and latch_result.grasped_seen and not latch_result.early_kill:
        q_hold = q_close.copy()
        q_hold[5] = args.close_deg[-1]
        hold_result = run_to_q("stationary_hold_closed", q_hold, plan.descend_tcp, args.hold_steps, "hold")
    if hold_result and hold_result.reached and not hold_result.early_kill:
        lift_result, current_seed_q, _lift_path_ok = run_resampled_path(
            "tiny_lift_closed_10mm",
            fresh_tcp_local(),
            plan.lift_tcp,
            q_hold,
            args.close_deg[-1],
            "lift",
            args.substep_steps,
        )

    results = [r for r in [approach_result, descend_result, latch_result, hold_result, lift_result] if r is not None]
    max_target_error = max((r.final_target_error_m for r in results), default=float("inf"))
    max_tcp_step = max((r.max_tcp_step_m for r in results), default=float("inf"))
    max_drift = max((r.max_object_drift_m for r in results), default=float("inf"))
    max_speed = max((r.max_object_speed_mps for r in results), default=float("inf"))
    max_tilt = max((r.max_tilt_deg for r in results), default=float("inf"))
    min_upright = min((r.min_upright_z for r in results), default=float("nan"))
    total_attach_calls = attach_stats["attach_calls"]
    total_posewrite_calls = attach_stats["posewrite_calls"]
    lift_follow = 0.0 if lift_result is None else lift_result.object_follow_delta_m
    hidden_posewrite_ok = total_posewrite_calls == 0
    verdict = _verdict(plan, approach_result, descend_result, latch_result, hold_result, lift_result, args)

    print(
        f"[cube2cm_local_grasp] aggregate verdict={verdict} events_done={len(results)}/5 "
        f"max_target_error_m={max_target_error:.6f} max_tcp_step_m={max_tcp_step:.6f} "
        f"max_object_drift_m={max_drift:.6f} max_object_speed_mps={max_speed:.6f} "
        f"max_tilt_deg={max_tilt:.3f} min_upright_z={min_upright:.6f} "
        f"lift_follow_delta_m={lift_follow:.6f} attach_calls={total_attach_calls} "
        f"posewrite_calls={total_posewrite_calls} hidden_kinematic_posewrite_artifact={_yes(not hidden_posewrite_ok)} "
        f"episode_done={_yes(episode_done)} nan_seen={_yes(nan_seen)}",
        flush=True,
    )
    print(f"[cube2cm_local_grasp] CUBE2CM_LOCAL_GRASP_CLOSE_SWEEP_VERDICT={verdict}", flush=True)
    env.close()
    sim_app.close()
    return 0 if verdict == "GRASP_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
