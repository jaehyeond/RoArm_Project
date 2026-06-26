#!/usr/bin/env python3
"""Replay D256 recorded states in the live 10cm env and measure geometry.

This is not a policy, teacher, or PPO probe. It writes the recorded D256 arm
joint state and cube pose for each frame, then checks the current env contact
proxy. The goal is to separate controller/action replay mismatch from contact
metric or geometry mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sim_scripts.cube10cm_top_view_d256_action_replay_probe import (
    DEFAULT_TEACHER_CSV,
    DEFAULT_USD,
    REPO,
    RUNTIME_ROOT,
    _rel,
    load_episode_rows,
    tensor_stats,
)


DEFAULT_OUT_DIR = RUNTIME_ROOT / "d256_state_sequence_probe_d266"


def write_d256_recorded_state(inner, rows: list[dict[str, float]]) -> dict[str, Any]:
    import torch

    device = inner.device
    num_envs = int(inner.num_envs)
    if len(rows) != num_envs:
        raise ValueError(f"rows length {len(rows)} != num_envs {num_envs}")
    env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    origins = inner.scene.env_origins[env_ids]

    joint_pos = inner._robot.data.joint_pos.detach().clone()
    arm = torch.tensor(
        [[float(row[f"arm_joint_{idx}_rad"]) for idx in range(5)] for row in rows],
        device=device,
        dtype=torch.float32,
    )
    gripper = torch.tensor(
        [float(row["gripper_joint_rad"]) for row in rows],
        device=device,
        dtype=torch.float32,
    )
    joint_pos[:, inner._bc_arm_joint_ids] = arm
    joint_pos[:, inner.gripper_joint_idx] = gripper
    joint_pos = torch.clamp(joint_pos, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
    joint_vel = torch.zeros_like(joint_pos)

    cube_local = torch.tensor(
        [[float(row["cube_local_x_m"]), float(row["cube_local_y_m"]), float(row["cube_local_z_m"])] for row in rows],
        device=device,
        dtype=torch.float32,
    )
    target_local = torch.tensor(
        [
            [float(row["target_local_x_m"]), float(row["target_local_y_m"]), float(row["target_local_z_m"])]
            for row in rows
        ],
        device=device,
        dtype=torch.float32,
    )
    push_dir = torch.tensor(
        [[float(row["push_dx"]), float(row["push_dy"])] for row in rows],
        device=device,
        dtype=torch.float32,
    )
    push_dir = push_dir / torch.clamp(torch.linalg.norm(push_dir, dim=-1, keepdim=True), min=1.0e-6)

    inner._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    inner._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    inner.robot_dof_targets[env_ids] = joint_pos

    cube_world = origins + cube_local
    cube_state = torch.zeros((num_envs, 13), device=device, dtype=torch.float32)
    cube_state[:, 0:3] = cube_world
    cube_state[:, 3] = 1.0
    inner._sponge.write_root_pose_to_sim(cube_state[:, 0:7], env_ids=env_ids)
    inner._sponge.write_root_velocity_to_sim(cube_state[:, 7:13], env_ids=env_ids)

    inner._target_world[env_ids] = origins + target_local
    inner._push_dir_xy[env_ids] = push_dir
    inner._compute_intermediate_values()
    return {
        "cube_local_x_range_m": [float(cube_local[:, 0].min().item()), float(cube_local[:, 0].max().item())],
        "cube_local_y_range_m": [float(cube_local[:, 1].min().item()), float(cube_local[:, 1].max().item())],
        "arm_joint_mean_rad": [float(v) for v in arm.mean(dim=0).detach().cpu().tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1266)
    parser.add_argument("--steps", type=int, default=580)
    parser.add_argument("--hold_steps", type=int, default=1)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument(
        "--tap_contact_proxy_mode",
        choices=("tcp_point", "link5_collision_aabb"),
        default="link5_collision_aabb",
    )
    parser.add_argument("--artifact_tag", type=str, default="d266")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from sim_scripts.cube10cm_top_view_teacher_rollout_probe import apply_d256_pose_reset

    selected_episodes, episode_rows = load_episode_rows(args.teacher_csv, int(args.num_envs))
    min_episode_len = min(len(rows) for rows in episode_rows)

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_feature_target_mode = "env_target"
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)

    env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    device = inner.device
    zero = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)
    inner.episode_length_buf[:] = inner.max_episode_length
    env.step(zero)
    reset_info = apply_d256_pose_reset(inner, [rows[0] for rows in episode_rows])

    out_dir = args.out_dir / "tap10cm"
    out_dir.mkdir(parents=True, exist_ok=True)
    step_csv = out_dir / f"d256_state_sequence_step_samples_{args.artifact_tag}.csv"
    summary_json = out_dir / f"d256_state_sequence_summary_{args.artifact_tag}.json"
    summary_md = out_dir / f"d256_state_sequence_summary_{args.artifact_tag}.md"

    min_tcp_cube_dist = torch.full((inner.num_envs,), float("inf"), device=device)
    max_disp_along = torch.full((inner.num_envs,), -float("inf"), device=device)
    max_disp_xy = torch.zeros(inner.num_envs, device=device)
    first_contact_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_tcp_threshold_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_useful_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    contact_threshold = float(inner.cfg.contact_slowdown_tcp_dist_m)
    min_contact_face_gap_abs = torch.full((inner.num_envs,), float("inf"), device=device)
    min_contact_lateral = torch.full((inner.num_envs,), float("inf"), device=device)
    min_contact_vertical_offset = torch.full((inner.num_envs,), float("inf"), device=device)
    max_tap_contact_proximity = torch.zeros(inner.num_envs, device=device)
    step_rows: list[dict[str, Any]] = []
    last_state_info: dict[str, Any] = {}

    with torch.inference_mode():
        for step in range(int(args.steps)):
            row_idx = min(step // max(1, int(args.hold_steps)), min_episode_len - 1)
            last_state_info = write_d256_recorded_state(inner, [rows[row_idx] for rows in episode_rows])
            targets = inner._robot.data.joint_pos.detach().clone()
            inner._external_joint_targets_override = targets
            env.step(zero)
            inner._compute_intermediate_values()
            terms = inner._tap_terms()
            tcp_dist = terms["tcp_cube_dist"].detach()
            disp_along = terms["disp_along"].detach()
            disp_xy = terms["disp_xy"].detach()
            contact_proxy = terms["tap_contact_proxy"].detach()
            useful_now = contact_proxy & terms["tap_reaction_now"].detach() & ~terms["tap_overshoot_now"].detach()
            min_tcp_cube_dist = torch.minimum(min_tcp_cube_dist, tcp_dist)
            max_disp_along = torch.maximum(max_disp_along, disp_along)
            max_disp_xy = torch.maximum(max_disp_xy, disp_xy)
            min_contact_face_gap_abs = torch.minimum(min_contact_face_gap_abs, torch.abs(terms["tap_contact_face_gap_m"].detach()))
            min_contact_lateral = torch.minimum(min_contact_lateral, terms["tap_contact_lateral_m"].detach())
            min_contact_vertical_offset = torch.minimum(
                min_contact_vertical_offset,
                terms["tap_contact_vertical_offset_m"].detach(),
            )
            max_tap_contact_proximity = torch.maximum(max_tap_contact_proximity, terms["tap_contact_proximity"].detach())
            unset_contact = (first_contact_step < 0) & contact_proxy
            first_contact_step[unset_contact] = int(step)
            tcp_threshold_now = tcp_dist < contact_threshold
            unset_tcp = (first_tcp_threshold_step < 0) & tcp_threshold_now
            first_tcp_threshold_step[unset_tcp] = int(step)
            unset_useful = (first_useful_step < 0) & useful_now
            first_useful_step[unset_useful] = int(step)

            if step % int(args.sample_every) == 0 or step == int(args.steps) - 1:
                step_rows.append(
                    {
                        "step": int(step),
                        "row_idx": int(row_idx),
                        "tcp_cube_dist_mean": float(tcp_dist.mean().item()),
                        "tcp_cube_dist_min": float(tcp_dist.min().item()),
                        "disp_along_mean": float(disp_along.mean().item()),
                        "disp_along_max": float(disp_along.max().item()),
                        "disp_xy_mean": float(disp_xy.mean().item()),
                        "disp_xy_max": float(disp_xy.max().item()),
                        "tap_contact_proxy_rate": float(contact_proxy.float().mean().item()),
                        "tcp_threshold_contact_rate": float(tcp_threshold_now.float().mean().item()),
                        "tap_useful_now_rate": float(useful_now.float().mean().item()),
                        "tap_contact_face_gap_abs_min": float(torch.abs(terms["tap_contact_face_gap_m"]).min().item()),
                        "tap_contact_lateral_min": float(terms["tap_contact_lateral_m"].min().item()),
                        "tap_contact_vertical_offset_min": float(terms["tap_contact_vertical_offset_m"].min().item()),
                    }
                )

    summary = {
        "artifact": f"cube10cm_{args.artifact_tag}_d256_state_sequence_probe",
        "status": "PASS_PROBE_EXECUTED",
        "no_ppo_learning": True,
        "teacher_used": False,
        "env_id": "RoArm-CubeTap10cm-Direct-v0",
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "hold_steps": int(args.hold_steps),
        "selected_episode_min": int(min(selected_episodes)),
        "selected_episode_max": int(max(selected_episodes)),
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "reset_pose_info": reset_info,
        "last_state_info": last_state_info,
        "contact_threshold_m": contact_threshold,
        "contact_env_count": int((first_contact_step >= 0).sum().item()),
        "contact_rate": float((first_contact_step >= 0).float().mean().item()),
        "first_contact_step_min": int(first_contact_step[first_contact_step >= 0].min().item()) if bool((first_contact_step >= 0).any()) else -1,
        "tcp_threshold_contact_env_count": int((first_tcp_threshold_step >= 0).sum().item()),
        "tcp_threshold_contact_rate": float((first_tcp_threshold_step >= 0).float().mean().item()),
        "first_tcp_threshold_step_min": int(first_tcp_threshold_step[first_tcp_threshold_step >= 0].min().item()) if bool((first_tcp_threshold_step >= 0).any()) else -1,
        "tap_useful_env_count": int((first_useful_step >= 0).sum().item()),
        "tap_useful_rate": float((first_useful_step >= 0).float().mean().item()),
        "first_tap_useful_step_min": int(first_useful_step[first_useful_step >= 0].min().item()) if bool((first_useful_step >= 0).any()) else -1,
        "min_tcp_cube_dist_m": tensor_stats(min_tcp_cube_dist),
        "min_tap_contact_face_gap_abs_m": tensor_stats(min_contact_face_gap_abs),
        "min_tap_contact_lateral_m": tensor_stats(min_contact_lateral),
        "min_tap_contact_vertical_offset_m": tensor_stats(min_contact_vertical_offset),
        "max_tap_contact_proximity": tensor_stats(max_tap_contact_proximity),
        "max_disp_along_m": tensor_stats(max_disp_along),
        "max_disp_xy_m": tensor_stats(max_disp_xy),
        "teacher_csv": _rel(args.teacher_csv),
        "robot_usd_path": _rel(args.robot_usd_path),
        "step_samples_csv": _rel(step_csv),
        "summary_json": _rel(summary_json),
        "summary_md": _rel(summary_md),
        "interpretation": (
            "This writes D256 recorded arm/cube states into the live 10cm env and "
            "measures tap contact with the selected tap_contact_proxy_mode, while "
            "also logging the older tcp_cube_dist threshold. The D247/D256 visual "
            "dataset was rendered through the Candidate6 contract, which uses "
            "link5_collision_aabb rather than raw tcp_point."
        ),
    }

    with step_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()) if step_rows else ["step"])
        writer.writeheader()
        writer.writerows(step_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_md.write_text(
        "# D266 D256 State Sequence Probe\n\n"
        f"- status: `{summary['status']}`\n"
        f"- teacher used: `{summary['teacher_used']}`\n"
        f"- tap contact proxy mode: `{summary['tap_contact_proxy_mode']}`\n"
        f"- steps/envs/hold_steps: `{args.steps}` / `{args.num_envs}` / `{args.hold_steps}`\n"
        f"- contact rate: `{summary['contact_rate']}`\n"
        f"- first contact step min: `{summary['first_contact_step_min']}`\n"
        f"- TCP-threshold contact rate: `{summary['tcp_threshold_contact_rate']}`\n"
        f"- tap useful rate: `{summary['tap_useful_rate']}`\n"
        f"- min TCP-cube distance mean/min/max: "
        f"`{summary['min_tcp_cube_dist_m']['mean']}` / `{summary['min_tcp_cube_dist_m']['min']}` / `{summary['min_tcp_cube_dist_m']['max']}`\n"
        f"- min contact face-gap abs mean/min/max: "
        f"`{summary['min_tap_contact_face_gap_abs_m']['mean']}` / `{summary['min_tap_contact_face_gap_abs_m']['min']}` / `{summary['min_tap_contact_face_gap_abs_m']['max']}`\n"
        f"- max disp along mean/min/max: "
        f"`{summary['max_disp_along_m']['mean']}` / `{summary['max_disp_along_m']['min']}` / `{summary['max_disp_along_m']['max']}`\n\n"
        f"Interpretation: {summary['interpretation']}\n"
    )

    print(
        "d256_state_sequence_probe result "
        f"proxy={summary['tap_contact_proxy_mode']} "
        f"contact_rate={summary['contact_rate']:.6f} "
        f"tcp_threshold_contact_rate={summary['tcp_threshold_contact_rate']:.6f} "
        f"tap_useful_rate={summary['tap_useful_rate']:.6f} "
        f"min_tcp_dist_mean={summary['min_tcp_cube_dist_m']['mean']:.6f} "
        f"min_tcp_dist_min={summary['min_tcp_cube_dist_m']['min']:.6f} "
        f"summary={_rel(summary_json)}"
    )
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
