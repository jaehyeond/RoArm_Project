"""Evaluate a learned BC joint-delta policy in IsaacLab cube-push physics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"


def make_model(torch, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
    layers = []
    last = in_dim
    for _ in range(int(hidden_layers)):
        layers.append(torch.nn.Linear(last, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        last = int(hidden_dim)
    layers.append(torch.nn.Linear(last, out_dim))
    return torch.nn.Sequential(*layers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=881)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--tcp_top_margin_m", type=float, default=0.003)
    parser.add_argument("--push_through_m", type=float, default=0.030)
    parser.add_argument("--approach_steps", type=int, default=220)
    parser.add_argument("--push_steps", type=int, default=90)
    parser.add_argument("--post_steps", type=int, default=40)
    parser.add_argument("--settle_steps", type=int, default=8)
    parser.add_argument("--trajectory_variant", choices=("v3_1",), default="v3_1")
    parser.add_argument("--v31_posx_precontact_clearance_m", type=float, default=0.014)
    parser.add_argument("--v31_posx_push_through_m", type=float, default=0.020)
    parser.add_argument("--v31_posx_tcp_top_margin_m", type=float, default=-0.011)
    parser.add_argument("--v31_posx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v31_posx_approach_steps", type=int, default=300)
    parser.add_argument("--v31_posx_push_steps", type=int, default=220)
    parser.add_argument("--v31_posx_post_steps", type=int, default=60)
    parser.add_argument("--v31_lowx_threshold_m", type=float, default=0.240)
    parser.add_argument("--v31_lowx_precontact_clearance_m", type=float, default=0.020)
    parser.add_argument("--v31_lowx_push_through_m", type=float, default=0.030)
    parser.add_argument("--v31_lowx_tcp_top_margin_m", type=float, default=0.003)
    parser.add_argument("--v31_lowx_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--v31_lowx_approach_steps", type=int, default=300)
    parser.add_argument("--v31_lowx_push_steps", type=int, default=220)
    parser.add_argument("--v31_lowx_post_steps", type=int, default=60)
    parser.add_argument("--policy_delta_clip_rad", type=float, default=0.040)
    parser.add_argument("--x_bucket_edges", type=float, nargs=2, default=(0.257, 0.308))
    parser.add_argument("--policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--posx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--lowx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--highx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--policy_delta_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_rollout_steps", type=int, default=0)
    parser.add_argument("--progress_log", type=Path, default=None)
    parser.add_argument("--skip_sim_close", action="store_true")
    parser.add_argument("--out_csv", type=Path, default=LOG_DIR / "bc_policy_rollout_eval.csv")
    parser.add_argument("--summary_json", type=Path, default=LOG_DIR / "bc_policy_rollout_eval_summary.json")
    args = parser.parse_args()

    def progress(message: str) -> None:
        if args.progress_log is None:
            return
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a") as fp:
            fp.write(f"{time.time():.3f} {message}\n")
            fp.flush()

    from isaaclab.app import AppLauncher

    progress("line1 parsed_args importing_app_launcher_done")
    app_launcher = AppLauncher(headless=True)
    sim_app = app_launcher.app
    progress("line2 app_launcher_ready")

    import gymnasium as gym
    import torch

    import roarm_rl  # noqa: F401 - registers envs
    from roarm_rl.roarm_cube_push_env import (
        AUDIT_DISP_XY_P99_M,
        AUDIT_SPEED_P95_MPS,
        AUDIT_SPEED_P99_MPS,
        AUDIT_TIP_P95_DEG,
        AUDIT_TIP_P99_DEG,
        CUBE_SIZE_M,
        RoArmCubePushEnvCfg,
    )

    env_cfg = RoArmCubePushEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.ik_endpoint_reset = False
    env_cfg.scripted_teacher_blend = 0.0
    env_cfg.action_scale = 0.0
    base_total_steps = int(args.approach_steps + args.push_steps + args.post_steps)
    v31_posx_total_steps = int(args.v31_posx_approach_steps + args.v31_posx_push_steps + args.v31_posx_post_steps)
    v31_lowx_total_steps = int(args.v31_lowx_approach_steps + args.v31_lowx_push_steps + args.v31_lowx_post_steps)
    total_steps = max(base_total_steps, v31_posx_total_steps, v31_lowx_total_steps)
    step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    min_episode_s = (int(args.settle_steps) + total_steps + 20) * step_dt
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), min_episode_s)

    env = gym.make("RoArm-CubePush-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    n = inner.num_envs
    progress(f"line3 env_ready num_envs={n} total_steps={total_steps} device={device}")

    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
    feature_columns = list(checkpoint["feature_columns"])
    target_columns = list(checkpoint["target_columns"])
    model = make_model(
        torch,
        len(feature_columns),
        len(target_columns),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x_mean = checkpoint["x_mean"].to(device=device, dtype=torch.float32).view(1, -1)
    x_std = checkpoint["x_std"].to(device=device, dtype=torch.float32).view(1, -1)
    y_mean = checkpoint["y_mean"].to(device=device, dtype=torch.float32).view(1, -1)
    y_std = checkpoint["y_std"].to(device=device, dtype=torch.float32).view(1, -1)
    progress(f"line4 checkpoint_ready feature_count={len(feature_columns)}")

    def no_dones() -> tuple[torch.Tensor, torch.Tensor]:
        done = torch.zeros(n, dtype=torch.bool, device=device)
        return done, done

    def zero_pre_physics_step(actions: torch.Tensor) -> None:
        inner.actions = torch.zeros_like(actions)
        inner._last_teacher_blend.zero_()
        inner._last_joint_delta_abs_mean.zero_()
        inner._last_contact_slowdown.fill_(1.0)

    inner._get_dones = no_dones
    inner._pre_physics_step = zero_pre_physics_step

    arm_joint_ids, arm_joint_names = inner._robot.find_joints(
        [
            "base_link_to_link1",
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
        ],
        preserve_order=True,
    )
    zero_action = torch.zeros((n, inner.cfg.action_space), device=device, dtype=torch.float32)
    half = 0.5 * float(CUBE_SIZE_M)

    original_write_pose = inner._sponge.write_root_pose_to_sim
    counters = {"posewrite_calls_during_rollout": 0}
    posewrite_watch = {"active": False}

    def watched_write_root_pose_to_sim(*a, **kw):
        if posewrite_watch["active"]:
            counters["posewrite_calls_during_rollout"] += 1
        return original_write_pose(*a, **kw)

    inner._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

    print(
        "[cube3cm_push_bc_policy_rollout] "
        f"isaac_run=YES num_envs={n} episodes={args.episodes} total_trials={n * args.episodes} "
        "controller=BC_MLP_joint_delta_policy learned_policy=YES training=NO dataset_generation=NO "
        "diffik_controller_used=NO grasp=NO attach_posewrite=NO rollout_object_posewrite=NO",
        flush=True,
    )
    print(
        "[cube3cm_push_bc_policy_rollout] "
        f"model_path={args.model_path} checkpoint_verdict={checkpoint.get('verdict')} "
        f"feature_count={len(feature_columns)} target_columns={target_columns} "
        f"trajectory_variant={args.trajectory_variant} arm_joint_names={arm_joint_names}",
        flush=True,
    )

    records: list[dict[str, float | int | str]] = []
    t0 = time.time()
    posx_trial_count = 0
    lowx_trial_count = 0

    def build_trajectory_tensors() -> dict[str, torch.Tensor]:
        cube = inner._cube_start_w
        push_dir = inner._push_dir_xy
        posx = (push_dir[:, 0] > 0.5) & (torch.abs(push_dir[:, 1]) < 0.5)
        cube_x_local = cube[:, 0] - inner.scene.env_origins[:, 0]
        edge0, edge1 = float(args.x_bucket_edges[0]), float(args.x_bucket_edges[1])
        posx_low_bucket = posx & (cube_x_local < edge0)
        posx_mid_bucket = posx & (cube_x_local >= edge0) & (cube_x_local < edge1)
        posx_high_bucket = posx & (cube_x_local >= edge1)
        lowx = posx & (cube_x_local <= float(args.v31_lowx_threshold_m))
        approach_steps = torch.full((n,), int(args.approach_steps), dtype=torch.long, device=device)
        push_steps = torch.full((n,), int(args.push_steps), dtype=torch.long, device=device)
        post_steps = torch.full((n,), int(args.post_steps), dtype=torch.long, device=device)
        precontact = torch.full((n,), float(args.precontact_clearance_m), dtype=torch.float32, device=device)
        push_through = torch.full((n,), float(args.push_through_m), dtype=torch.float32, device=device)
        tcp_top_margin = torch.full((n,), float(args.tcp_top_margin_m), dtype=torch.float32, device=device)
        lateral_offset = torch.zeros((n,), dtype=torch.float32, device=device)
        approach_steps[posx] = int(args.v31_posx_approach_steps)
        push_steps[posx] = int(args.v31_posx_push_steps)
        post_steps[posx] = int(args.v31_posx_post_steps)
        precontact[posx] = float(args.v31_posx_precontact_clearance_m)
        push_through[posx] = float(args.v31_posx_push_through_m)
        tcp_top_margin[posx] = float(args.v31_posx_tcp_top_margin_m)
        lateral_offset[posx] = float(args.v31_posx_lateral_offset_m)
        approach_steps[lowx] = int(args.v31_lowx_approach_steps)
        push_steps[lowx] = int(args.v31_lowx_push_steps)
        post_steps[lowx] = int(args.v31_lowx_post_steps)
        precontact[lowx] = float(args.v31_lowx_precontact_clearance_m)
        push_through[lowx] = float(args.v31_lowx_push_through_m)
        tcp_top_margin[lowx] = float(args.v31_lowx_tcp_top_margin_m)
        lateral_offset[lowx] = float(args.v31_lowx_lateral_offset_m)
        return {
            "posx": posx,
            "posx_low_bucket": posx_low_bucket,
            "posx_mid_bucket": posx_mid_bucket,
            "posx_high_bucket": posx_high_bucket,
            "lowx": lowx,
            "approach_steps": approach_steps,
            "push_steps": push_steps,
            "post_steps": post_steps,
            "precontact": precontact,
            "push_through": push_through,
            "tcp_top_margin": tcp_top_margin,
            "lateral_offset": lateral_offset,
        }

    def compute_alpha(step: int, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        step_v = torch.full((n,), int(step), dtype=torch.float32, device=device)
        approach = traj["approach_steps"].to(dtype=torch.float32)
        push = torch.clamp(traj["push_steps"].to(dtype=torch.float32), min=1.0)
        raw_alpha = (step_v - approach + 1.0) / push
        alpha = torch.where(step_v < approach, torch.zeros_like(raw_alpha), raw_alpha)
        alpha = torch.where(step_v >= approach + push, torch.ones_like(alpha), alpha)
        return torch.clamp(alpha, min=0.0, max=1.0)

    def compute_tcp_targets(alpha: torch.Tensor, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        cube = inner._cube_start_w
        push_dir = inner._push_dir_xy
        lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=-1)
        pre = cube.clone()
        through = cube.clone()
        z = cube[:, 2] + half + traj["tcp_top_margin"]
        lateral = lateral_dir * traj["lateral_offset"].unsqueeze(-1)
        pre[:, 0:2] = cube[:, 0:2] - push_dir * (half + traj["precontact"]).unsqueeze(-1) + lateral
        through[:, 0:2] = cube[:, 0:2] + push_dir * (half + traj["push_through"]).unsqueeze(-1) + lateral
        pre[:, 2] = z
        through[:, 2] = z
        return pre + alpha.unsqueeze(-1) * (through - pre)

    def build_feature_tensor(alpha: torch.Tensor, tcp_target_w: torch.Tensor) -> torch.Tensor:
        origin = inner.scene.env_origins
        cube_local = inner._sponge_pos_w - origin
        tcp_local = inner._tcp_pos_w - origin
        target_local = tcp_target_w - origin
        tcp_to_cube = cube_local - tcp_local
        target_to_tcp = target_local - tcp_local
        target_to_cube = target_local - cube_local
        joints = inner._robot.data.joint_pos[:, arm_joint_ids]
        gripper = inner._robot.data.joint_pos[:, inner.gripper_joint_idx]
        values = {
            "push_dx": inner._push_dir_xy[:, 0],
            "push_dy": inner._push_dir_xy[:, 1],
            "phase_alpha": alpha,
            "cube_local_x_m": cube_local[:, 0],
            "cube_local_y_m": cube_local[:, 1],
            "cube_local_z_m": cube_local[:, 2],
            "tcp_local_x_m": tcp_local[:, 0],
            "tcp_local_y_m": tcp_local[:, 1],
            "tcp_local_z_m": tcp_local[:, 2],
            "target_local_x_m": target_local[:, 0],
            "target_local_y_m": target_local[:, 1],
            "target_local_z_m": target_local[:, 2],
            "tcp_to_cube_x_m": tcp_to_cube[:, 0],
            "tcp_to_cube_y_m": tcp_to_cube[:, 1],
            "tcp_to_cube_z_m": tcp_to_cube[:, 2],
            "target_to_tcp_x_m": target_to_tcp[:, 0],
            "target_to_tcp_y_m": target_to_tcp[:, 1],
            "target_to_tcp_z_m": target_to_tcp[:, 2],
            "target_to_cube_x_m": target_to_cube[:, 0],
            "target_to_cube_y_m": target_to_cube[:, 1],
            "target_to_cube_z_m": target_to_cube[:, 2],
            "arm_joint_0_rad": joints[:, 0],
            "arm_joint_1_rad": joints[:, 1],
            "arm_joint_2_rad": joints[:, 2],
            "arm_joint_3_rad": joints[:, 3],
            "arm_joint_4_rad": joints[:, 4],
            "gripper_joint_rad": gripper,
        }
        missing = [col for col in feature_columns if col not in values]
        if missing:
            raise KeyError(f"unsupported checkpoint feature columns: {missing}")
        return torch.stack([values[col] for col in feature_columns], dim=-1).to(dtype=torch.float32)

    try:
        for episode in range(int(args.episodes)):
            progress(f"line5 episode_start episode={episode}")
            env.reset()
            posewrite_watch["active"] = False
            inner._grasped[:] = False
            inner._was_grasped[:] = False
            for _ in range(int(args.settle_steps)):
                env.step(zero_action)
            progress(f"line6 settle_done episode={episode}")
            inner._compute_intermediate_values()
            cube_start_w = inner._cube_start_w.clone()
            push_dir = inner._push_dir_xy.clone()
            min_tcp_cube_dist = torch.full((n,), float("inf"), device=device)
            min_tcp_target_err = torch.full((n,), float("inf"), device=device)
            final_tcp_target_err = torch.zeros((n,), device=device)
            max_cube_speed = torch.zeros((n,), device=device)
            max_joint_delta_abs = torch.zeros((n,), device=device)
            traj = build_trajectory_tensors()
            prev_policy_delta = torch.zeros((n, len(target_columns)), device=device, dtype=torch.float32)
            posx_trial_count += int(traj["posx"].sum().detach().cpu().item())
            lowx_trial_count += int(traj["lowx"].sum().detach().cpu().item())
            posewrite_watch["active"] = True
            rollout_steps = total_steps
            if int(args.max_rollout_steps) > 0:
                rollout_steps = min(total_steps, int(args.max_rollout_steps))
            for step in range(rollout_steps):
                if args.progress_log is not None and step % 25 == 0:
                    progress(f"line7 rollout_step episode={episode} step={step}/{rollout_steps}")
                alpha = compute_alpha(step, traj)
                inner._compute_intermediate_values()
                tcp_target_w = compute_tcp_targets(alpha, traj)
                with torch.no_grad():
                    x = build_feature_tensor(alpha, tcp_target_w)
                    pred_n = model((x - x_mean) / x_std)
                    clipped_delta = pred_n * y_std + y_mean
                    clipped_delta = torch.clamp(
                        clipped_delta,
                        -float(args.policy_delta_clip_rad),
                        float(args.policy_delta_clip_rad),
                    )
                    delta_scale = torch.full((n,), float(args.policy_delta_scale), dtype=torch.float32, device=device)
                    delta_scale = torch.where(
                        traj["posx"],
                        delta_scale * float(args.posx_policy_delta_scale),
                        delta_scale,
                    )
                    delta_scale = torch.where(
                        traj["posx_low_bucket"],
                        delta_scale * float(args.lowx_policy_delta_scale),
                        delta_scale,
                    )
                    delta_scale = torch.where(
                        traj["posx_high_bucket"],
                        delta_scale * float(args.highx_policy_delta_scale),
                        delta_scale,
                    )
                    clipped_delta = clipped_delta * delta_scale.unsqueeze(-1)
                    smooth_alpha = max(0.0, min(1.0, float(args.policy_delta_smoothing_alpha)))
                    if smooth_alpha < 1.0:
                        clipped_delta = smooth_alpha * clipped_delta + (1.0 - smooth_alpha) * prev_policy_delta
                    prev_policy_delta = clipped_delta.detach()
                joint_pos_arm = inner._robot.data.joint_pos[:, arm_joint_ids]
                max_joint_delta_abs = torch.maximum(
                    max_joint_delta_abs, torch.max(torch.abs(clipped_delta), dim=-1).values
                )
                target_full = inner.robot_dof_targets.clone()
                target_full[:, arm_joint_ids] = joint_pos_arm + clipped_delta
                target_full[:, inner.gripper_joint_idx] = 0.0
                target_full = torch.clamp(target_full, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
                target_full[:, inner.gripper_joint_idx] = 0.0
                inner.robot_dof_targets[:] = target_full

                env.step(zero_action)
                inner._compute_intermediate_values()
                tcp_err = torch.norm(inner._tcp_pos_w - tcp_target_w, p=2, dim=-1)
                final_tcp_target_err = tcp_err
                min_tcp_target_err = torch.minimum(min_tcp_target_err, tcp_err)
                min_tcp_cube_dist = torch.minimum(
                    min_tcp_cube_dist,
                    torch.norm(inner._tcp_pos_w - inner._sponge_pos_w, p=2, dim=-1),
                )
                max_cube_speed = torch.maximum(max_cube_speed, torch.norm(inner._sponge.data.root_lin_vel_w, p=2, dim=-1))

            posewrite_watch["active"] = False
            progress(f"line8 rollout_done episode={episode}")
            inner._compute_intermediate_values()
            terms = inner._push_terms()
            disp_xy_vec = inner._sponge_pos_w[:, 0:2] - cube_start_w[:, 0:2]
            lateral_vec = disp_xy_vec - terms["disp_along"].unsqueeze(-1) * push_dir
            for idx in range(n):
                records.append(
                    {
                        "trial": len(records),
                        "episode": episode,
                        "env_id": idx,
                        "cube_x0_m": float((cube_start_w[idx, 0] - inner.scene.env_origins[idx, 0]).detach().cpu().item()),
                        "cube_y0_m": float((cube_start_w[idx, 1] - inner.scene.env_origins[idx, 1]).detach().cpu().item()),
                        "push_dx": float(push_dir[idx, 0].detach().cpu().item()),
                        "push_dy": float(push_dir[idx, 1].detach().cpu().item()),
                        "disp_along_push_m": float(terms["disp_along"][idx].detach().cpu().item()),
                        "disp_xy_m": float(terms["disp_xy"][idx].detach().cpu().item()),
                        "lateral_abs_m": float(torch.norm(lateral_vec[idx], p=2).detach().cpu().item()),
                        "target_xy_dist_m": float(terms["target_xy_dist"][idx].detach().cpu().item()),
                        "max_cube_speed_mps": float(max_cube_speed[idx].detach().cpu().item()),
                        "final_speed_mps": float(terms["speed"][idx].detach().cpu().item()),
                        "tip_angle_deg": float(terms["tip_angle_deg"][idx].detach().cpu().item()),
                        "controlled_push": int(bool(terms["controlled"][idx].detach().cpu().item())),
                        "impact_outlier": int(bool(terms["impact"][idx].detach().cpu().item())),
                        "low_motion": int(bool(terms["low_motion"][idx].detach().cpu().item())),
                        "success_marker": int(bool(inner._push_success_flag[idx].detach().cpu().item())),
                        "grasped_marker": int(bool(inner._grasped[idx].detach().cpu().item())),
                        "trajectory_variant": args.trajectory_variant,
                        "v31_posx_applied": int(bool(traj["posx"][idx].detach().cpu().item())),
                        "v31_lowx_applied": int(bool(traj["lowx"][idx].detach().cpu().item())),
                        "posx_x_bucket": (
                            "low_x"
                            if bool(traj["posx_low_bucket"][idx].detach().cpu().item())
                            else "mid_x"
                            if bool(traj["posx_mid_bucket"][idx].detach().cpu().item())
                            else "high_x"
                            if bool(traj["posx_high_bucket"][idx].detach().cpu().item())
                            else "not_posx"
                        ),
                        "min_tcp_cube_dist_m": float(min_tcp_cube_dist[idx].detach().cpu().item()),
                        "min_tcp_target_err_m": float(min_tcp_target_err[idx].detach().cpu().item()),
                        "final_tcp_target_err_m": float(final_tcp_target_err[idx].detach().cpu().item()),
                        "max_joint_delta_abs_rad": float(max_joint_delta_abs[idx].detach().cpu().item()),
                    }
                )
    finally:
        posewrite_watch["active"] = False

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with args.out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / len(records) if records else 0.0

    def rate(key: str) -> float:
        return sum(int(r[key]) for r in records) / len(records) if records else 0.0

    summary = {
        "controller": "BC_MLP_joint_delta_policy",
        "learned_policy": True,
        "supervised_bc_checkpoint": True,
        "diffik_controller_used": False,
        "training": False,
        "dataset_generation": False,
        "grasp_attach": False,
        "rollout_object_posewrite": False,
        "posewrite_calls_during_rollout": counters["posewrite_calls_during_rollout"],
        "env_auto_reset_disabled": True,
        "env_joint_delta_action_loop_bypassed": True,
        "model_path": str(args.model_path),
        "model_checkpoint_verdict": checkpoint.get("verdict"),
        "model_dataset_csv": checkpoint.get("dataset_csv"),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "trajectory_variant": args.trajectory_variant,
        "num_envs": n,
        "episodes": int(args.episodes),
        "trials": len(records),
        "total_steps_per_trial": total_steps,
        "executed_steps_per_trial": min(total_steps, int(args.max_rollout_steps)) if int(args.max_rollout_steps) > 0 else total_steps,
        "rollout_truncated": int(args.max_rollout_steps) > 0 and int(args.max_rollout_steps) < total_steps,
        "policy_delta_clip_rad": float(args.policy_delta_clip_rad),
        "x_bucket_edges": [float(args.x_bucket_edges[0]), float(args.x_bucket_edges[1])],
        "policy_delta_scale": float(args.policy_delta_scale),
        "posx_policy_delta_scale": float(args.posx_policy_delta_scale),
        "lowx_policy_delta_scale": float(args.lowx_policy_delta_scale),
        "highx_policy_delta_scale": float(args.highx_policy_delta_scale),
        "policy_delta_smoothing_alpha": float(args.policy_delta_smoothing_alpha),
        "model_safety_config": checkpoint.get("safety_config", {}),
        "v31_posx_trial_count": posx_trial_count,
        "v31_lowx_trial_count": lowx_trial_count,
        "controlled_push_rate": rate("controlled_push"),
        "impact_outlier_rate": rate("impact_outlier"),
        "low_motion_rate": rate("low_motion"),
        "success_marker_rate": rate("success_marker"),
        "grasped_marker_rate": rate("grasped_marker"),
        "disp_along_push_mean_m": mean("disp_along_push_m"),
        "disp_xy_mean_m": mean("disp_xy_m"),
        "max_cube_speed_mean_mps": mean("max_cube_speed_mps"),
        "min_tcp_cube_dist_mean_m": mean("min_tcp_cube_dist_m"),
        "min_tcp_target_err_mean_m": mean("min_tcp_target_err_m"),
        "final_tcp_target_err_mean_m": mean("final_tcp_target_err_m"),
        "audit_thresholds": {
            "speed_p95_mps": AUDIT_SPEED_P95_MPS,
            "speed_p99_mps": AUDIT_SPEED_P99_MPS,
            "tip_p95_deg": AUDIT_TIP_P95_DEG,
            "tip_p99_deg": AUDIT_TIP_P99_DEG,
            "disp_xy_p99_m": AUDIT_DISP_XY_P99_M,
        },
        "out_csv": str(args.out_csv),
        "elapsed_s": time.time() - t0,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "[cube3cm_push_bc_policy_rollout] "
        f"summary trials={summary['trials']} controlled_push_rate={summary['controlled_push_rate']:.6f} "
        f"impact_outlier_rate={summary['impact_outlier_rate']:.6f} low_motion_rate={summary['low_motion_rate']:.6f} "
        f"success_marker_rate={summary['success_marker_rate']:.6f} "
        f"posewrite_calls_during_rollout={summary['posewrite_calls_during_rollout']} "
        f"learned_policy=YES diffik_controller_used=NO "
        f"policy_delta_scale={float(args.policy_delta_scale):.6f} "
        f"posx_policy_delta_scale={float(args.posx_policy_delta_scale):.6f} "
        f"lowx_policy_delta_scale={float(args.lowx_policy_delta_scale):.6f} "
        f"highx_policy_delta_scale={float(args.highx_policy_delta_scale):.6f} "
        f"policy_delta_smoothing_alpha={float(args.policy_delta_smoothing_alpha):.6f}",
        flush=True,
    )
    progress("line9 artifacts_written")
    if bool(args.skip_sim_close):
        progress("line10 skip_sim_close")
    else:
        progress("line10 closing_env")
        env.close()
        progress("line11 closing_sim_app")
        sim_app.close()
        progress("line12 sim_app_closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
