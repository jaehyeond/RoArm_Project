#!/usr/bin/env python3
"""Parallel 3cm cube push/tap rollout probe for RoArm in Isaac Lab.

This is not a grasp, hold-lift, dataset, PPO, or VLA run.  It reuses the local
RoArm Isaac Lab scene to answer a narrower question: if the robot only moves
near a 3cm cube and pushes through/near it with joint-target commands, what
state/action/result distributions come out?

Object pose writes are used only for reset/randomization.  During rollout the
cube must move only through physics, and hidden grasp attach is monkey-patched
to a counter-only no-op.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SIM_SCRIPTS = REPO / "sim_scripts"
if str(SIM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SIM_SCRIPTS))

from roarm_kinematics import fk_tcp, ik_dls, clip_joints  # noqa: E402


TABLE_Z = -0.012117
HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final"
    / "tmp_p7"
    / "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024"
    / "roarm_m3.usd"
)


@dataclass(frozen=True)
class RolloutConfig:
    num_envs: int
    episodes: int
    approach_steps: int
    precontact_steps: int
    push_steps: int
    post_steps: int
    seed: int
    cube_size_m: tuple[float, float, float]
    cube_mass_kg: float
    cube_xy_x_range_m: tuple[float, float]
    cube_xy_y_range_m: tuple[float, float]
    approach_clearance_m: float
    precontact_clearance_m: float
    push_through_m: float
    tcp_top_margin_m: float
    action_scale_rad: float


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _quat_angle_from_identity_wxyz(q: np.ndarray) -> np.ndarray:
    w = np.clip(np.abs(q[:, 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)


def _push_dirs(num_envs: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "random":
        theta = rng.uniform(-math.pi, math.pi, size=num_envs)
        return np.stack([np.cos(theta), np.sin(theta), np.zeros(num_envs)], axis=1)
    dirs = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return dirs[rng.integers(0, len(dirs), size=num_envs)]


def _solve_stage_targets(
    cube_pos_local: np.ndarray,
    push_dir: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return IK targets in degrees plus IK ok mask.

    Shape:
      q_targets: (num_envs, 3, 6) for approach, pre-contact, push-end.
      tcp_targets: (num_envs, 3, 3)
      ik_ok: (num_envs,)
    """
    n = cube_pos_local.shape[0]
    cube_size = np.asarray(args.cube_size_m, dtype=np.float64)
    half_xy = 0.5 * max(cube_size[0], cube_size[1])
    tcp_z = cube_pos_local[:, 2] + 0.5 * cube_size[2] + float(args.tcp_top_margin_m)

    approach_tcp = cube_pos_local.copy()
    approach_tcp[:, :2] -= push_dir[:, :2] * (half_xy + float(args.approach_clearance_m))
    approach_tcp[:, 2] = tcp_z

    pre_tcp = cube_pos_local.copy()
    pre_tcp[:, :2] -= push_dir[:, :2] * (half_xy + float(args.precontact_clearance_m))
    pre_tcp[:, 2] = tcp_z

    end_tcp = cube_pos_local.copy()
    end_tcp[:, :2] += push_dir[:, :2] * (half_xy + float(args.push_through_m))
    end_tcp[:, 2] = tcp_z

    q_targets = np.zeros((n, 3, 6), dtype=np.float64)
    tcp_targets = np.stack([approach_tcp, pre_tcp, end_tcp], axis=1)
    ik_ok = np.ones(n, dtype=bool)
    ik_err_mm = np.zeros((n, 3), dtype=np.float64)

    for env_id in range(n):
        q_seed = HOME_DEG.copy()
        for stage_idx in range(3):
            q, converged, err_mm, _iters = ik_dls(
                tcp_targets[env_id, stage_idx],
                q_seed,
                max_iter=int(args.ik_max_iter),
                tol_mm=float(args.ik_tol_mm),
                damping=float(args.ik_damping),
            )
            q[5] = 0.0  # gripper open; no grasp latch.
            q = clip_joints(q)
            q_targets[env_id, stage_idx] = q
            ik_err_mm[env_id, stage_idx] = err_mm
            if not converged or np.linalg.norm(fk_tcp(q) - tcp_targets[env_id, stage_idx]) > float(args.ik_accept_m):
                ik_ok[env_id] = False
            q_seed = q

    return q_targets, tcp_targets, ik_ok, ik_err_mm, tcp_z


def _stage_alpha(step: int, start: int, length: int) -> float:
    if length <= 1:
        return 1.0
    return min(1.0, max(0.0, (step - start + 1) / float(length)))


def _target_q_for_step(q_targets: np.ndarray, step: int, args: argparse.Namespace) -> np.ndarray:
    a0 = int(args.approach_steps)
    a1 = a0 + int(args.precontact_steps)
    a2 = a1 + int(args.push_steps)
    if step < a0:
        alpha = _stage_alpha(step, 0, a0)
        return HOME_DEG[None, :] * (1.0 - alpha) + q_targets[:, 0, :] * alpha
    if step < a1:
        alpha = _stage_alpha(step, a0, int(args.precontact_steps))
        return q_targets[:, 0, :] * (1.0 - alpha) + q_targets[:, 1, :] * alpha
    if step < a2:
        alpha = _stage_alpha(step, a1, int(args.push_steps))
        return q_targets[:, 1, :] * (1.0 - alpha) + q_targets[:, 2, :] * alpha
    return q_targets[:, 2, :]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot_usd_path", default=str(DEFAULT_LOCAL_USD))
    ap.add_argument("--out_dir", default="claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe")
    ap.add_argument("--cube_size_m", nargs=3, type=float, default=[0.030, 0.030, 0.030])
    ap.add_argument("--cube_mass_kg", type=float, default=0.020)
    ap.add_argument("--cube_x_min", type=float, default=0.205)
    ap.add_argument("--cube_x_max", type=float, default=0.360)
    ap.add_argument("--cube_y_min", type=float, default=-0.125)
    ap.add_argument("--cube_y_max", type=float, default=0.125)
    ap.add_argument("--push_dir_mode", choices=["cardinal", "random"], default="cardinal")
    ap.add_argument("--approach_clearance_m", type=float, default=0.060)
    ap.add_argument("--precontact_clearance_m", type=float, default=0.004)
    ap.add_argument("--push_through_m", type=float, default=0.040)
    ap.add_argument("--tcp_top_margin_m", type=float, default=0.001)
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--approach_steps", type=int, default=40)
    ap.add_argument("--precontact_steps", type=int, default=20)
    ap.add_argument("--push_steps", type=int, default=30)
    ap.add_argument("--post_steps", type=int, default=20)
    ap.add_argument("--ik_tol_mm", type=float, default=1.5)
    ap.add_argument("--ik_accept_m", type=float, default=0.004)
    ap.add_argument("--ik_max_iter", type=int, default=250)
    ap.add_argument("--ik_damping", type=float, default=2.0)
    ap.add_argument("--save_arrays", action="store_true")
    ap.add_argument("--sample_envs_to_print", type=int, default=5)
    args = ap.parse_args()

    if args.num_envs <= 0 or args.episodes <= 0:
        raise ValueError("num_envs and episodes must be positive")
    if any(v <= 0.0 for v in args.cube_size_m):
        raise ValueError("cube_size_m entries must be positive")
    if args.push_through_m <= 0.0:
        raise ValueError("push_through_m must be positive")
    for name in ("settle_steps", "approach_steps", "precontact_steps", "push_steps", "post_steps"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ROARM_M3_USD_PATH"] = str(args.robot_usd_path)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import isaaclab.sim as sim_utils
    import roarm_rl  # noqa: F401
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    cfg_obj = RolloutConfig(
        num_envs=int(args.num_envs),
        episodes=int(args.episodes),
        approach_steps=int(args.approach_steps),
        precontact_steps=int(args.precontact_steps),
        push_steps=int(args.push_steps),
        post_steps=int(args.post_steps),
        seed=int(args.seed),
        cube_size_m=tuple(float(x) for x in args.cube_size_m),
        cube_mass_kg=float(args.cube_mass_kg),
        cube_xy_x_range_m=(float(args.cube_x_min), float(args.cube_x_max)),
        cube_xy_y_range_m=(float(args.cube_y_min), float(args.cube_y_max)),
        approach_clearance_m=float(args.approach_clearance_m),
        precontact_clearance_m=float(args.precontact_clearance_m),
        push_through_m=float(args.push_through_m),
        tcp_top_margin_m=float(args.tcp_top_margin_m),
        action_scale_rad=0.1,
    )

    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.episode_length_s = 20.0
    env_cfg.reward_phase = 6
    env_cfg.curriculum_pregrasp = False
    env_cfg.curriculum_pregrasp_hover = False
    env_cfg.curriculum_attached_transport_release = False
    env_cfg.curriculum_post_grasp_cap = False
    env_cfg.curriculum_disable_nearzone_cap = False
    env_cfg.curriculum_spawn_min_r = 0.0
    env_cfg.curriculum_spawn_max_r = 0.0
    env_cfg.seed = int(args.seed)
    env_cfg.sponge.spawn = sim_utils.CuboidCfg(
        size=tuple(float(x) for x in args.cube_size_m),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(args.cube_mass_kg)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.70, 1.00), metallic=0.0),
    )
    cube_center_z = TABLE_Z + 0.5 * float(args.cube_size_m[2])
    env_cfg.sponge.init_state.pos = (0.280, 0.000, cube_center_z)

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    n = inner.num_envs
    action_scale = float(inner.cfg.action_scale)
    total_rollout_steps = int(args.approach_steps + args.precontact_steps + args.push_steps + args.post_steps)

    original_attach = inner._update_grasp_attach
    original_write_pose = inner._sponge.write_root_pose_to_sim
    counters = {"attach_calls": 0, "posewrite_calls_during_rollout": 0}
    posewrite_watch = {"active": False}

    def marker_only_attach() -> None:
        counters["attach_calls"] += int(inner._grasped.sum().detach().cpu().item())

    def watched_write_root_pose_to_sim(*a, **kw):
        if posewrite_watch["active"]:
            counters["posewrite_calls_during_rollout"] += 1
        return original_write_pose(*a, **kw)

    inner._update_grasp_attach = marker_only_attach
    inner._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

    print(
        "[cube3cm_push_rollout_probe] "
        f"isaac_run=YES num_envs={n} episodes={args.episodes} total_trials={n * args.episodes} "
        f"robot_usd_path={args.robot_usd_path} cube_size_m={tuple(args.cube_size_m)} "
        "grasp=NO attach_posewrite=NO rollout_object_posewrite=NO training=NO dataset_generation=NO",
        flush=True,
    )
    print(
        "[cube3cm_push_rollout_probe] "
        "action_semantics=normalized_joint_delta "
        f"target_update='robot_dof_targets += action_scale({action_scale:.3f}) * action' "
        "action_dim=6 action_clip=[-1,1] gripper_action_target=open_0rad",
        flush=True,
    )

    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    per_env_rows: list[dict[str, float | int | str]] = []
    arrays: dict[str, list[np.ndarray]] = {"actions": [], "cube_pos": [], "joint_pos": []}
    t0 = time.time()

    for episode in range(int(args.episodes)):
        rng = np.random.default_rng(int(args.seed) + episode)
        env.reset()
        counters["attach_calls"] = 0
        counters["posewrite_calls_during_rollout"] = 0
        posewrite_watch["active"] = False

        cube_pos_local = np.zeros((n, 3), dtype=np.float64)
        cube_pos_local[:, 0] = rng.uniform(float(args.cube_x_min), float(args.cube_x_max), size=n)
        cube_pos_local[:, 1] = rng.uniform(float(args.cube_y_min), float(args.cube_y_max), size=n)
        cube_pos_local[:, 2] = cube_center_z
        push_dir = _push_dirs(n, rng, str(args.push_dir_mode))

        q_targets_deg, tcp_targets, ik_ok, ik_err_mm, tcp_z = _solve_stage_targets(cube_pos_local, push_dir, args)

        cube_pose = torch.zeros((n, 7), device=device, dtype=torch.float32)
        cube_pose[:, 0:3] = torch.tensor(cube_pos_local, device=device, dtype=torch.float32) + inner.scene.env_origins
        cube_pose[:, 3] = 1.0
        original_write_pose(cube_pose, env_ids=torch.arange(n, device=device))
        inner._sponge.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
        inner._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
        inner._robot.set_joint_position_target(home_rad)
        inner.robot_dof_targets[:] = home_rad
        inner._grasped[:] = False
        inner._was_grasped[:] = False
        inner.scene.write_data_to_sim()
        inner.scene.update(inner.sim.get_physics_dt())

        zero_action = torch.zeros((n, 6), device=device, dtype=torch.float32)
        for _ in range(int(args.settle_steps)):
            env.step(zero_action)

        cube_start = (inner._sponge.data.root_pos_w - inner.scene.env_origins).detach().cpu().numpy().astype(np.float64)
        max_speed = np.zeros(n, dtype=np.float64)
        min_tcp_cube_dist = np.full(n, np.inf, dtype=np.float64)
        action_abs_sum = np.zeros(n, dtype=np.float64)
        action_sat_count = np.zeros(n, dtype=np.int64)
        action_count = 0
        posewrite_watch["active"] = True
        ep_actions = []
        ep_cube_pos = []
        ep_joint_pos = []

        for step in range(total_rollout_steps):
            target_q_deg = _target_q_for_step(q_targets_deg, step, args)
            target_q_rad = torch.tensor(np.radians(target_q_deg), device=device, dtype=torch.float32)
            delta = (target_q_rad - inner.robot_dof_targets) / action_scale
            actions = torch.clamp(delta, -1.0, 1.0)
            out = env.step(actions)
            if len(out) == 5:
                _obs, _rew, terminated, truncated, _extras = out
                done_any = bool((terminated | truncated).any().detach().cpu().item())
            else:
                _obs, _rew, dones, _extras = out
                done_any = bool(dones.any().detach().cpu().item())
            if done_any:
                print(f"[cube3cm_push_rollout_probe] WARNING done_seen=YES episode={episode} step={step}", flush=True)

            link5_pos = inner._robot.data.body_pos_w[:, inner.link5_idx]
            link5_quat = inner._robot.data.body_quat_w[:, inner.link5_idx]
            tcp_offset = _quat_rotate(link5_quat, inner._tcp_local.expand(n, 3))
            tcp_local = (link5_pos + tcp_offset - inner.scene.env_origins).detach().cpu().numpy().astype(np.float64)
            cube_now = (inner._sponge.data.root_pos_w - inner.scene.env_origins).detach().cpu().numpy().astype(np.float64)
            cube_vel = inner._sponge.data.root_vel_w[:, 0:3].detach().cpu().numpy().astype(np.float64)
            speed = np.linalg.norm(cube_vel, axis=1)
            max_speed = np.maximum(max_speed, speed)
            min_tcp_cube_dist = np.minimum(min_tcp_cube_dist, np.linalg.norm(tcp_local - cube_now, axis=1))

            act_np = actions.detach().cpu().numpy().astype(np.float32)
            action_abs_sum += np.mean(np.abs(act_np), axis=1)
            action_sat_count += np.any(np.abs(act_np) >= 0.999, axis=1)
            action_count += 1
            if args.save_arrays:
                ep_actions.append(act_np)
                ep_cube_pos.append(cube_now.astype(np.float32))
                ep_joint_pos.append(inner._robot.data.joint_pos.detach().cpu().numpy().astype(np.float32))

        posewrite_watch["active"] = False

        cube_final = (inner._sponge.data.root_pos_w - inner.scene.env_origins).detach().cpu().numpy().astype(np.float64)
        cube_quat = inner._sponge.data.root_quat_w.detach().cpu().numpy().astype(np.float64)
        joint_pos = inner._robot.data.joint_pos.detach().cpu().numpy().astype(np.float64)
        target_pos = inner.robot_dof_targets.detach().cpu().numpy().astype(np.float64)
        disp = cube_final - cube_start
        disp_xy = np.linalg.norm(disp[:, :2], axis=1)
        disp_total = np.linalg.norm(disp, axis=1)
        disp_push = np.sum(disp[:, :2] * push_dir[:, :2], axis=1)
        tip_angle = _quat_angle_from_identity_wxyz(cube_quat)
        q_err_deg = np.degrees(np.max(np.abs(joint_pos - target_pos), axis=1))
        grasped = inner._grasped.detach().cpu().numpy().astype(bool)

        for env_id in range(n):
            per_env_rows.append(
                {
                    "episode": episode,
                    "env_id": env_id,
                    "ik_ok": int(bool(ik_ok[env_id])),
                    "ik_err_max_mm": float(np.max(ik_err_mm[env_id])),
                    "cube_x0_m": float(cube_start[env_id, 0]),
                    "cube_y0_m": float(cube_start[env_id, 1]),
                    "push_dx": float(push_dir[env_id, 0]),
                    "push_dy": float(push_dir[env_id, 1]),
                    "tcp_z_target_m": float(tcp_z[env_id]),
                    "disp_x_m": float(disp[env_id, 0]),
                    "disp_y_m": float(disp[env_id, 1]),
                    "disp_z_m": float(disp[env_id, 2]),
                    "disp_xy_m": float(disp_xy[env_id]),
                    "disp_total_m": float(disp_total[env_id]),
                    "disp_along_push_m": float(disp_push[env_id]),
                    "max_cube_speed_mps": float(max_speed[env_id]),
                    "min_tcp_cube_dist_m": float(min_tcp_cube_dist[env_id]),
                    "tip_angle_deg": float(math.degrees(tip_angle[env_id])),
                    "q_err_max_deg": float(q_err_deg[env_id]),
                    "action_abs_mean": float(action_abs_sum[env_id] / max(1, action_count)),
                    "action_saturation_frac": float(action_sat_count[env_id] / max(1, action_count)),
                    "grasped_marker": int(bool(grasped[env_id])),
                }
            )

        if args.save_arrays:
            arrays["actions"].append(np.stack(ep_actions, axis=0))
            arrays["cube_pos"].append(np.stack(ep_cube_pos, axis=0))
            arrays["joint_pos"].append(np.stack(ep_joint_pos, axis=0))

        moved_1mm = float(np.mean(disp_xy >= 0.001))
        moved_5mm = float(np.mean(disp_xy >= 0.005))
        moved_10mm = float(np.mean(disp_xy >= 0.010))
        push_positive_1mm = float(np.mean(disp_push >= 0.001))
        print(
            "[cube3cm_push_rollout_probe] "
            f"episode={episode} ik_ok_rate={np.mean(ik_ok):.4f} "
            f"disp_xy_mean_m={np.mean(disp_xy):.6f} disp_xy_p95_m={np.percentile(disp_xy, 95):.6f} "
            f"moved_1mm_rate={moved_1mm:.4f} moved_5mm_rate={moved_5mm:.4f} moved_10mm_rate={moved_10mm:.4f} "
            f"push_positive_1mm_rate={push_positive_1mm:.4f} "
            f"max_speed_mean_mps={np.mean(max_speed):.6f} "
            f"action_abs_mean={np.mean(action_abs_sum / max(1, action_count)):.6f} "
            f"action_sat_frac_mean={np.mean(action_sat_count / max(1, action_count)):.6f} "
            f"grasped_marker_rate={np.mean(grasped):.4f} attach_calls={counters['attach_calls']} "
            f"posewrite_calls_during_rollout={counters['posewrite_calls_during_rollout']}",
            flush=True,
        )

    rows_path = out_dir / "per_env.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_env_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_env_rows)

    disp_xy_all = np.asarray([float(r["disp_xy_m"]) for r in per_env_rows], dtype=np.float64)
    disp_push_all = np.asarray([float(r["disp_along_push_m"]) for r in per_env_rows], dtype=np.float64)
    action_abs_all = np.asarray([float(r["action_abs_mean"]) for r in per_env_rows], dtype=np.float64)
    action_sat_all = np.asarray([float(r["action_saturation_frac"]) for r in per_env_rows], dtype=np.float64)
    ik_ok_all = np.asarray([int(r["ik_ok"]) for r in per_env_rows], dtype=np.float64)
    grasped_all = np.asarray([int(r["grasped_marker"]) for r in per_env_rows], dtype=np.float64)

    summary = {
        "probe": "cube3cm_push_rollout_probe",
        "local_only": True,
        "isaac_run": True,
        "training": False,
        "dataset_generation": False,
        "grasp": False,
        "attach_posewrite": False,
        "rollout_object_posewrite": False,
        "robot_usd_path": str(args.robot_usd_path),
        "config": asdict(cfg_obj),
        "total_trials": len(per_env_rows),
        "elapsed_s": time.time() - t0,
        "ik_ok_rate": float(np.mean(ik_ok_all)),
        "disp_xy_mean_m": float(np.mean(disp_xy_all)),
        "disp_xy_std_m": float(np.std(disp_xy_all)),
        "disp_xy_p50_m": float(np.percentile(disp_xy_all, 50)),
        "disp_xy_p90_m": float(np.percentile(disp_xy_all, 90)),
        "disp_xy_p95_m": float(np.percentile(disp_xy_all, 95)),
        "disp_xy_max_m": float(np.max(disp_xy_all)),
        "moved_1mm_rate": float(np.mean(disp_xy_all >= 0.001)),
        "moved_5mm_rate": float(np.mean(disp_xy_all >= 0.005)),
        "moved_10mm_rate": float(np.mean(disp_xy_all >= 0.010)),
        "push_positive_1mm_rate": float(np.mean(disp_push_all >= 0.001)),
        "push_negative_1mm_rate": float(np.mean(disp_push_all <= -0.001)),
        "action_abs_mean": float(np.mean(action_abs_all)),
        "action_saturation_frac_mean": float(np.mean(action_sat_all)),
        "grasped_marker_rate": float(np.mean(grasped_all)),
        "attach_calls": int(counters["attach_calls"]),
        "posewrite_calls_during_rollout": int(counters["posewrite_calls_during_rollout"]),
        "outputs": {
            "per_env_csv": str(rows_path),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.save_arrays:
        arrays_path = out_dir / "rollout_arrays.npz"
        np.savez_compressed(
            arrays_path,
            actions=np.stack(arrays["actions"], axis=0),
            cube_pos=np.stack(arrays["cube_pos"], axis=0),
            joint_pos=np.stack(arrays["joint_pos"], axis=0),
        )
        summary["outputs"]["rollout_arrays_npz"] = str(arrays_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[cube3cm_push_rollout_probe] SUMMARY "
        f"total_trials={summary['total_trials']} ik_ok_rate={summary['ik_ok_rate']:.4f} "
        f"disp_xy_mean_m={summary['disp_xy_mean_m']:.6f} disp_xy_p95_m={summary['disp_xy_p95_m']:.6f} "
        f"moved_1mm_rate={summary['moved_1mm_rate']:.4f} moved_5mm_rate={summary['moved_5mm_rate']:.4f} "
        f"push_positive_1mm_rate={summary['push_positive_1mm_rate']:.4f} "
        f"action_abs_mean={summary['action_abs_mean']:.6f} "
        f"action_saturation_frac_mean={summary['action_saturation_frac_mean']:.6f} "
        f"grasped_marker_rate={summary['grasped_marker_rate']:.4f} "
        f"attach_calls={summary['attach_calls']} posewrite_calls_during_rollout={summary['posewrite_calls_during_rollout']}",
        flush=True,
    )
    print(f"[cube3cm_push_rollout_probe] wrote summary={summary_path}", flush=True)
    print(f"[cube3cm_push_rollout_probe] wrote per_env={rows_path}", flush=True)
    for row in per_env_rows[: max(0, int(args.sample_envs_to_print))]:
        print(
            "[cube3cm_push_rollout_probe] sample "
            f"episode={row['episode']} env_id={row['env_id']} ik_ok={row['ik_ok']} "
            f"cube_xy=({row['cube_x0_m']:.4f},{row['cube_y0_m']:.4f}) "
            f"push_dir=({row['push_dx']:.1f},{row['push_dy']:.1f}) "
            f"disp_xy_m={row['disp_xy_m']:.6f} disp_push_m={row['disp_along_push_m']:.6f} "
            f"action_abs_mean={row['action_abs_mean']:.6f} sat_frac={row['action_saturation_frac']:.6f}",
            flush=True,
        )

    inner._update_grasp_attach = original_attach
    inner._sponge.write_root_pose_to_sim = original_write_pose
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
