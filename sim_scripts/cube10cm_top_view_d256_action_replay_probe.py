#!/usr/bin/env python3
"""Replay D256 train-clean joint targets in the live 10cm env, no PPO/teacher."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_TEACHER_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_TEACHER_CHECKPOINT = D242_ROOT / "state_action_teacher_d257" / "cube10cm_d257_state_action_teacher_clipped0040.pt"
DEFAULT_OUT_DIR = RUNTIME_ROOT / "d256_action_replay_probe_d264"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def tensor_stats(x) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(x.mean().item()), "min": float(x.min().item()), "max": float(x.max().item())}


def cosine_mean(torch, a, b) -> float:
    denom = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
    valid = denom > 1.0e-8
    if not bool(valid.any().item()):
        return 0.0
    cos = torch.sum(a[valid] * b[valid], dim=-1) / denom[valid]
    return float(cos.mean().item())


def load_episode_rows(
    csv_path: Path,
    num_envs: int,
    episode_min: int | None = None,
    episode_max: int | None = None,
    episode_indices: list[int] | None = None,
) -> tuple[list[int], list[list[dict[str, float]]]]:
    required = [
        "episode_index",
        "frame_index_t",
        "cube_local_x_m",
        "cube_local_y_m",
        "cube_local_z_m",
        "target_local_x_m",
        "target_local_y_m",
        "target_local_z_m",
        "push_dx",
        "push_dy",
        "arm_joint_0_rad",
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
        "gripper_joint_rad",
        "joint_delta_0_rad",
        "joint_delta_1_rad",
        "joint_delta_2_rad",
        "joint_delta_3_rad",
        "joint_delta_4_rad",
    ]
    episodes: dict[int, list[dict[str, float]]] = {}
    episode_index_set = set(episode_indices) if episode_indices is not None else None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing columns in {csv_path}: {missing}")
        for row in reader:
            ep = int(row["episode_index"])
            if episode_index_set is not None and ep not in episode_index_set:
                continue
            if episode_min is not None and ep < int(episode_min):
                continue
            if episode_max is not None and ep > int(episode_max):
                continue
            episodes.setdefault(ep, []).append({c: float(row[c]) for c in required})

    valid = []
    for ep, rows in episodes.items():
        rows.sort(key=lambda r: int(r["frame_index_t"]))
        if rows and int(rows[0]["frame_index_t"]) == 0:
            valid.append(ep)
    if episode_indices is not None:
        valid = [ep for ep in episode_indices if ep in episodes]
    else:
        valid.sort()
    if not valid:
        raise ValueError(
            f"no valid episodes in {csv_path} for episode_min={episode_min} "
            f"episode_max={episode_max} episode_indices={episode_indices}"
        )

    if num_envs <= len(valid):
        idxs = [round(i * (len(valid) - 1) / max(1, num_envs - 1)) for i in range(num_envs)]
        selected = [valid[int(i)] for i in idxs]
    else:
        selected = [valid[i % len(valid)] for i in range(num_envs)]
    return selected, [episodes[ep] for ep in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1264)
    parser.add_argument("--steps", type=int, default=580)
    parser.add_argument("--hold_steps", type=int, default=3)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--episode_min", type=int, default=None)
    parser.add_argument("--episode_max", type=int, default=None)
    parser.add_argument("--episode_indices", type=str, default="")
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--execution_mode", choices=("direct_targets", "policy_actions"), default="direct_targets")
    parser.add_argument(
        "--tap_contact_proxy_mode",
        choices=("tcp_point", "link5_collision_aabb"),
        default="link5_collision_aabb",
    )
    parser.add_argument("--compare_teacher", action="store_true")
    parser.add_argument("--teacher_checkpoint_path", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--action_scale", type=float, default=0.04)
    parser.add_argument("--action_smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=0.04)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.06)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument(
        "--bc_teacher_phase_timing",
        choices=("episode_scaled", "direct_steps", "linear_episode", "linear_steps"),
        default="linear_steps",
    )
    parser.add_argument("--bc_teacher_linear_phase_steps", type=int, default=579)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=1.0)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=0.8)
    parser.add_argument("--bc_teacher_delta_smoothing_alpha", type=float, default=0.85)
    parser.add_argument("--artifact_tag", type=str, default="d264")
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

    episode_indices = (
        [int(part) for part in str(args.episode_indices).split(",") if part.strip()]
        if str(args.episode_indices).strip()
        else None
    )
    selected_episodes, episode_rows = load_episode_rows(
        args.teacher_csv,
        int(args.num_envs),
        args.episode_min,
        args.episode_max,
        episode_indices,
    )

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.action_scale = float(args.action_scale)
    env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.bc_teacher_checkpoint_path = str(args.teacher_checkpoint_path) if bool(args.compare_teacher) else ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_feature_target_mode = "env_target"
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)
    env_cfg.bc_teacher_linear_phase_steps = int(args.bc_teacher_linear_phase_steps)
    env_cfg.bc_teacher_policy_delta_scale = float(args.bc_teacher_policy_delta_scale)
    env_cfg.bc_teacher_lowx_policy_delta_scale = float(args.bc_teacher_lowx_policy_delta_scale)
    env_cfg.bc_teacher_highx_policy_delta_scale = float(args.bc_teacher_highx_policy_delta_scale)
    env_cfg.bc_teacher_delta_smoothing_alpha = float(args.bc_teacher_delta_smoothing_alpha)
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
    step_csv = out_dir / f"d256_action_replay_step_samples_{args.artifact_tag}.csv"
    summary_json = out_dir / f"d256_action_replay_summary_{args.artifact_tag}.json"
    summary_md = out_dir / f"d256_action_replay_summary_{args.artifact_tag}.md"

    min_tcp_cube_dist = torch.full((inner.num_envs,), float("inf"), device=device)
    max_disp_along = torch.full((inner.num_envs,), -float("inf"), device=device)
    max_disp_xy = torch.zeros(inner.num_envs, device=device)
    first_contact_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_tcp_threshold_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    first_tap_useful_step = torch.full((inner.num_envs,), -1, dtype=torch.long, device=device)
    contact_threshold = float(inner.cfg.contact_slowdown_tcp_dist_m)
    step_rows: list[dict[str, Any]] = []
    max_target_jump = torch.zeros(inner.num_envs, device=device)
    compare_teacher = bool(args.compare_teacher)
    if compare_teacher and not getattr(inner, "_bc_teacher_ready", False):
        raise RuntimeError("compare_teacher requested but BC teacher did not load")
    teacher_recorded_mse_sum = 0.0
    teacher_needed_mse_sum = 0.0
    teacher_recorded_mae_sum = 0.0
    teacher_needed_mae_sum = 0.0
    teacher_recorded_cos_sum = 0.0
    teacher_needed_cos_sum = 0.0
    teacher_action_abs_mean_sum = 0.0
    teacher_action_abs_max = 0.0
    teacher_compare_steps = 0

    with torch.inference_mode():
        for step in range(int(args.steps)):
            row_idx = min(step // max(1, int(args.hold_steps)), min(len(rows) for rows in episode_rows) - 1)
            targets = inner._robot.data.joint_pos.detach().clone()
            recorded_delta = torch.tensor(
                [
                    [float(rows[row_idx][f"joint_delta_{idx}_rad"]) for idx in range(5)]
                    for rows in episode_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            target_arm = torch.tensor(
                [
                    [
                        float(rows[row_idx][f"arm_joint_{idx}_rad"]) + float(rows[row_idx][f"joint_delta_{idx}_rad"])
                        for idx in range(5)
                    ]
                    for rows in episode_rows
                ],
                device=device,
                dtype=torch.float32,
            )
            targets[:, inner._bc_arm_joint_ids] = target_arm
            targets[:, inner.gripper_joint_idx] = 0.0
            targets = torch.clamp(targets, inner.robot_dof_lower_limits, inner.robot_dof_upper_limits)
            current_arm = inner._robot.data.joint_pos[:, inner._bc_arm_joint_ids]
            needed_delta = target_arm - current_arm
            teacher_recorded_mse = teacher_needed_mse = teacher_recorded_mae = teacher_needed_mae = 0.0
            teacher_recorded_cos = teacher_needed_cos = teacher_action_abs_mean = teacher_action_abs_max_step = 0.0
            if compare_teacher:
                teacher_actions = inner._bc_teacher_actions().detach().clamp(-1.0, 1.0)
                teacher_delta = teacher_actions[:, inner._bc_arm_joint_ids] * float(inner.cfg.action_scale)
                teacher_recorded_err = teacher_delta - recorded_delta
                teacher_needed_err = teacher_delta - needed_delta
                teacher_recorded_mse = float(torch.mean(teacher_recorded_err**2).item())
                teacher_needed_mse = float(torch.mean(teacher_needed_err**2).item())
                teacher_recorded_mae = float(torch.mean(torch.abs(teacher_recorded_err)).item())
                teacher_needed_mae = float(torch.mean(torch.abs(teacher_needed_err)).item())
                teacher_recorded_cos = cosine_mean(torch, teacher_delta, recorded_delta)
                teacher_needed_cos = cosine_mean(torch, teacher_delta, needed_delta)
                teacher_action_abs_mean = float(torch.mean(torch.abs(teacher_actions)).item())
                teacher_action_abs_max_step = float(torch.max(torch.abs(teacher_actions)).item())
                teacher_recorded_mse_sum += teacher_recorded_mse
                teacher_needed_mse_sum += teacher_needed_mse
                teacher_recorded_mae_sum += teacher_recorded_mae
                teacher_needed_mae_sum += teacher_needed_mae
                teacher_recorded_cos_sum += teacher_recorded_cos
                teacher_needed_cos_sum += teacher_needed_cos
                teacher_action_abs_mean_sum += teacher_action_abs_mean
                teacher_action_abs_max = max(teacher_action_abs_max, teacher_action_abs_max_step)
                teacher_compare_steps += 1
            max_target_jump = torch.maximum(max_target_jump, torch.max(torch.abs(target_arm - current_arm), dim=-1).values)
            if str(args.execution_mode) == "direct_targets":
                exec_actions = zero
                inner._external_joint_targets_override = targets
            else:
                exec_actions = torch.zeros((inner.num_envs, inner.cfg.action_space), device=device)
                exec_actions[:, inner._bc_arm_joint_ids] = torch.clamp(
                    needed_delta / max(float(inner.cfg.action_scale), 1.0e-6),
                    -1.0,
                    1.0,
                )
                exec_actions[:, inner.gripper_joint_idx] = 0.0
            obs, rewards, dones, infos = env.step(exec_actions)

            inner._compute_intermediate_values()
            terms = inner._tap_terms()
            tcp_dist = terms["tcp_cube_dist"].detach()
            disp_along = terms["disp_along"].detach()
            disp_xy = terms["disp_xy"].detach()
            contact_proxy = terms["tap_contact_proxy"].detach()
            tap_useful_now = contact_proxy & terms["tap_reaction_now"].detach() & ~terms["tap_overshoot_now"].detach()
            min_tcp_cube_dist = torch.minimum(min_tcp_cube_dist, tcp_dist)
            max_disp_along = torch.maximum(max_disp_along, disp_along)
            max_disp_xy = torch.maximum(max_disp_xy, disp_xy)
            tcp_threshold_now = tcp_dist < contact_threshold
            unset_contact = (first_contact_step < 0) & contact_proxy
            first_contact_step[unset_contact] = int(step)
            unset_tcp = (first_tcp_threshold_step < 0) & tcp_threshold_now
            first_tcp_threshold_step[unset_tcp] = int(step)
            unset_useful = (first_tap_useful_step < 0) & tap_useful_now
            first_tap_useful_step[unset_useful] = int(step)

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
                        "tap_useful_now_rate": float(tap_useful_now.float().mean().item()),
                        "target_jump_abs_max_mean": float(max_target_jump.mean().item()),
                        "target_jump_abs_max_max": float(max_target_jump.max().item()),
                        "teacher_recorded_delta_mse": teacher_recorded_mse,
                        "teacher_needed_delta_mse": teacher_needed_mse,
                        "teacher_recorded_delta_mae": teacher_recorded_mae,
                        "teacher_needed_delta_mae": teacher_needed_mae,
                        "teacher_recorded_delta_cosine": teacher_recorded_cos,
                        "teacher_needed_delta_cosine": teacher_needed_cos,
                        "teacher_action_abs_mean": teacher_action_abs_mean,
                        "teacher_action_abs_max": teacher_action_abs_max_step,
                    }
                )

    summary = {
        "artifact": f"cube10cm_{args.artifact_tag}_d256_action_replay_probe",
        "status": "PASS_PROBE_EXECUTED",
        "no_ppo_learning": True,
        "teacher_used": False,
        "teacher_compared": compare_teacher,
        "env_id": "RoArm-CubeTap10cm-Direct-v0",
        "execution_mode": str(args.execution_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "action_scale": float(inner.cfg.action_scale),
        "action_smoothing_alpha": float(inner.cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(inner.cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(inner.cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(inner.cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(inner.cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(inner.cfg.joint_delta_reference),
        "bc_teacher_phase_timing": str(inner.cfg.bc_teacher_phase_timing),
        "bc_teacher_linear_phase_steps": int(inner.cfg.bc_teacher_linear_phase_steps),
        "bc_teacher_feature_target_mode": str(inner.cfg.bc_teacher_feature_target_mode),
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "hold_steps": int(args.hold_steps),
        "episode_min_filter": int(args.episode_min) if args.episode_min is not None else None,
        "episode_max_filter": int(args.episode_max) if args.episode_max is not None else None,
        "episode_indices_filter": episode_indices,
        "selected_episode_min": int(min(selected_episodes)),
        "selected_episode_max": int(max(selected_episodes)),
        "selected_episode_unique_count": int(len(set(selected_episodes))),
        "reset_pose_info": reset_info,
        "contact_threshold_m": contact_threshold,
        "contact_env_count": int((first_contact_step >= 0).sum().item()),
        "contact_rate": float((first_contact_step >= 0).float().mean().item()),
        "first_contact_step_min": int(first_contact_step[first_contact_step >= 0].min().item()) if bool((first_contact_step >= 0).any()) else -1,
        "tcp_threshold_contact_env_count": int((first_tcp_threshold_step >= 0).sum().item()),
        "tcp_threshold_contact_rate": float((first_tcp_threshold_step >= 0).float().mean().item()),
        "first_tcp_threshold_step_min": int(first_tcp_threshold_step[first_tcp_threshold_step >= 0].min().item()) if bool((first_tcp_threshold_step >= 0).any()) else -1,
        "tap_useful_env_count": int((first_tap_useful_step >= 0).sum().item()),
        "tap_useful_rate": float((first_tap_useful_step >= 0).float().mean().item()),
        "first_tap_useful_step_min": int(first_tap_useful_step[first_tap_useful_step >= 0].min().item()) if bool((first_tap_useful_step >= 0).any()) else -1,
        "min_tcp_cube_dist_m": tensor_stats(min_tcp_cube_dist),
        "max_disp_along_m": tensor_stats(max_disp_along),
        "max_disp_xy_m": tensor_stats(max_disp_xy),
        "max_target_jump_abs_rad": tensor_stats(max_target_jump),
        "teacher_recorded_delta_mse_mean": (
            float(teacher_recorded_mse_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_needed_delta_mse_mean": (
            float(teacher_needed_mse_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_recorded_delta_mae_mean": (
            float(teacher_recorded_mae_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_needed_delta_mae_mean": (
            float(teacher_needed_mae_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_recorded_delta_cosine_mean": (
            float(teacher_recorded_cos_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_needed_delta_cosine_mean": (
            float(teacher_needed_cos_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_action_abs_mean": (
            float(teacher_action_abs_mean_sum / teacher_compare_steps) if teacher_compare_steps else None
        ),
        "teacher_action_abs_max": float(teacher_action_abs_max) if teacher_compare_steps else None,
        "teacher_compare_steps": int(teacher_compare_steps),
        "teacher_csv": _rel(args.teacher_csv),
        "teacher_checkpoint_path": _rel(args.teacher_checkpoint_path) if compare_teacher else "",
        "robot_usd_path": _rel(args.robot_usd_path),
        "step_samples_csv": _rel(step_csv),
        "summary_json": _rel(summary_json),
        "summary_md": _rel(summary_md),
        "interpretation": (
            "This replays D256 state+joint_delta targets directly in the live 10cm env. "
            "For tap10cm, contact_rate uses tap_contact_proxy_mode and "
            "tcp_threshold_contact_rate reports the older tcp_cube_dist threshold."
        ),
    }

    with step_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()) if step_rows else ["step"])
        writer.writeheader()
        writer.writerows(step_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary_md.write_text(
        "# D264 D256 Action Replay Probe\n\n"
        f"- status: `{summary['status']}`\n"
        f"- teacher used: `{summary['teacher_used']}`\n"
        f"- teacher compared: `{summary['teacher_compared']}`\n"
        f"- execution mode: `{summary['execution_mode']}`\n"
        f"- tap contact proxy mode: `{summary['tap_contact_proxy_mode']}`\n"
        f"- action scale/max delta: `{summary['action_scale']}` / `{summary['max_joint_delta_per_step_rad']}`\n"
        f"- action smoothing/contact scales: `{summary['action_smoothing_alpha']}` / `{summary['contact_joint_delta_scale']}` / `{summary['fast_cube_joint_delta_scale']}`\n"
        f"- joint delta reference: `{summary['joint_delta_reference']}`\n"
        f"- bc teacher phase timing: `{summary['bc_teacher_phase_timing']}`\n"
        f"- steps/envs/hold_steps: `{args.steps}` / `{args.num_envs}` / `{args.hold_steps}`\n"
        f"- episode filter: `{summary['episode_min_filter']}`..`{summary['episode_max_filter']}`\n"
        f"- selected episode range/count: `{summary['selected_episode_min']}..{summary['selected_episode_max']}` / `{summary['selected_episode_unique_count']}`\n"
        f"- contact rate: `{summary['contact_rate']}`\n"
        f"- first contact step min: `{summary['first_contact_step_min']}`\n"
        f"- TCP-threshold contact rate: `{summary['tcp_threshold_contact_rate']}`\n"
        f"- tap useful rate: `{summary['tap_useful_rate']}`\n"
        f"- min TCP-cube distance mean/min/max: "
        f"`{summary['min_tcp_cube_dist_m']['mean']}` / `{summary['min_tcp_cube_dist_m']['min']}` / `{summary['min_tcp_cube_dist_m']['max']}`\n"
        f"- max disp along mean/min/max: "
        f"`{summary['max_disp_along_m']['mean']}` / `{summary['max_disp_along_m']['min']}` / `{summary['max_disp_along_m']['max']}`\n"
        f"- max target jump abs mean/max: "
        f"`{summary['max_target_jump_abs_rad']['mean']}` / `{summary['max_target_jump_abs_rad']['max']}`\n\n"
        f"- teacher recorded delta MSE/MAE/cosine mean: "
        f"`{summary['teacher_recorded_delta_mse_mean']}` / "
        f"`{summary['teacher_recorded_delta_mae_mean']}` / "
        f"`{summary['teacher_recorded_delta_cosine_mean']}`\n"
        f"- teacher needed delta MSE/MAE/cosine mean: "
        f"`{summary['teacher_needed_delta_mse_mean']}` / "
        f"`{summary['teacher_needed_delta_mae_mean']}` / "
        f"`{summary['teacher_needed_delta_cosine_mean']}`\n"
        f"- teacher action abs mean/max: "
        f"`{summary['teacher_action_abs_mean']}` / `{summary['teacher_action_abs_max']}`\n\n"
        f"Interpretation: {summary['interpretation']}\n"
    )

    print(
        "d256_action_replay_probe result "
        f"proxy={summary['tap_contact_proxy_mode']} "
        f"contact_rate={summary['contact_rate']:.6f} "
        f"tcp_threshold_contact_rate={summary['tcp_threshold_contact_rate']:.6f} "
        f"tap_useful_rate={summary['tap_useful_rate']:.6f} "
        f"min_tcp_dist_mean={summary['min_tcp_cube_dist_m']['mean']:.6f} "
        f"max_disp_along_mean={summary['max_disp_along_m']['mean']:.6f} "
        f"summary={_rel(summary_json)}"
    )
    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
