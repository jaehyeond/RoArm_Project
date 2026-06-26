"""Trace D277 frozen actor actions against the D257 teacher sidecar.

This is a diagnostic, not training.  The BC teacher checkpoint is loaded only to
compute comparison actions; `bc_teacher_blend` and imitation reward remain zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
DEFAULT_D256_CSV = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/"
    "ppo_actor_prior_teacher_rows_d256.csv"
)
DEFAULT_TEACHER_CHECKPOINT = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/"
    "cube10cm_d257_state_action_teacher_clipped0040.pt"
)
DEFAULT_ACTOR_CHECKPOINT = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/"
    "model_0.pt"
)
DEFAULT_OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "actor_teacher_trace_d279/tap10cm"
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _obs_is_finite(torch, obs: Any) -> bool:
    if isinstance(obs, dict):
        return all(_obs_is_finite(torch, value) for value in obs.values())
    if "TensorDict" in type(obs).__name__ and hasattr(obs, "values"):
        return all(_obs_is_finite(torch, value) for value in obs.values())
    return bool(torch.isfinite(obs).all().detach().cpu().item())


def _tensor_mean(x) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_min(x) -> float:
    return float(x.detach().float().min().cpu().item())


def _tensor_max(x) -> float:
    return float(x.detach().float().max().cpu().item())


def _finite_mean(torch, x) -> float | None:
    mask = torch.isfinite(x)
    if not bool(mask.any().detach().cpu().item()):
        return None
    return float(x[mask].detach().float().mean().cpu().item())


def _finite_min(torch, x) -> float | None:
    mask = torch.isfinite(x)
    if not bool(mask.any().detach().cpu().item()):
        return None
    return float(x[mask].detach().float().min().cpu().item())


def _finite_max(torch, x) -> float | None:
    mask = torch.isfinite(x)
    if not bool(mask.any().detach().cpu().item()):
        return None
    return float(x[mask].detach().float().max().cpu().item())


def _group_stats(torch, name: str, mask, env_metrics: dict[str, Any]) -> dict[str, float | int | str | None]:
    count = int(mask.detach().cpu().sum().item())
    out: dict[str, float | int | str | None] = {"group": name, "count": count}
    keys = (
        "actor_teacher_mse_mean",
        "actor_teacher_mae_mean",
        "actor_teacher_cosine_mean",
        "actor_clip_abs_mean",
        "teacher_abs_mean",
        "actor_raw_clip_exceed_rate",
        "tap_max_disp_along_m",
        "tap_max_disp_xy_m",
        "max_vertical_offset_m",
        "min_contact_vertical_offset_m",
    )
    for key in keys:
        values = env_metrics[key]
        if count == 0:
            out[key] = None
        elif key == "min_contact_vertical_offset_m":
            out[key] = _finite_mean(torch, values[mask])
        else:
            out[key] = _tensor_mean(values[mask])
    return out


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# D279 Actor-vs-Teacher Trace",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- diagnostic class: `{summary['diagnostic_class']}`",
        f"- actor checkpoint: `{summary['actor_checkpoint']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- env id: `{summary['env_id']}`",
        f"- steps/envs: `{summary['eval_steps']}` / `{summary['num_envs']}`",
        f"- action scale/max joint delta: `{summary['action_scale']}` / `{summary['max_joint_delta_per_step_rad']}`",
        f"- env stop/useful terminate: `{summary['tap_stop_after_useful_seen']}` / `{summary['tap_useful_terminate']}`",
        f"- env useful hold rate last/max: `{summary['env_stop_after_useful_hold_rate_last']}` / `{summary['env_stop_after_useful_hold_rate_max_trace']}`",
        f"- vertical gate mode/value: `{summary['vertical_gate_mode']}` / `{summary['vertical_gate_value_m']}`",
        f"- D256 reset active rate: `{summary['d256_reset_active_rate']}`",
        f"- BC blend last: `{summary['bc_teacher_blend_mean_last']}`",
        f"- actor-teacher MSE/MAE/cosine: `{summary['actor_teacher_mse_mean']}` / `{summary['actor_teacher_mae_mean']}` / `{summary['actor_teacher_cosine_mean']}`",
        f"- actor clipped abs mean/max trace: `{summary['actor_clip_abs_mean_trace_mean']}` / `{summary['actor_clip_abs_max_trace_max']}`",
        f"- teacher abs mean/max trace: `{summary['teacher_abs_mean_trace_mean']}` / `{summary['teacher_abs_max_trace_max']}`",
        f"- actor raw clip exceed rate/max: `{summary['actor_raw_clip_exceed_rate_mean']}` / `{summary['actor_raw_clip_exceed_rate_max_trace']}`",
        f"- contact/useful/reaction seen: `{summary['tap_contact_seen_rate']}` / `{summary['tap_useful_seen_rate']}` / `{summary['tap_reaction_seen_rate']}`",
        f"- success/overshoot seen: `{summary['tap_success_rate']}` / `{summary['tap_overshoot_seen_rate']}`",
        f"- max disp along mean/max: `{summary['tap_max_disp_along_mean_m']}` / `{summary['tap_max_disp_along_max_m']}`",
        f"- max disp xy mean/max: `{summary['tap_max_disp_xy_mean_m']}` / `{summary['tap_max_disp_xy_max_m']}`",
        f"- max vertical offset mean/max: `{summary['max_vertical_offset_mean_m']}` / `{summary['max_vertical_offset_max_m']}`",
        f"- min contact vertical offset mean/min/max: `{summary['min_contact_vertical_offset_mean_m']}` / `{summary['min_contact_vertical_offset_min_m']}` / `{summary['min_contact_vertical_offset_max_m']}`",
        f"- joint delta cap last/max: `{summary['joint_delta_cap_rate_mean_last']}` / `{summary['joint_delta_cap_rate_max_trace']}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(["", "## Groups", ""])
    for group in summary["groups"]:
        lines.append(
            "- "
            f"{group['group']}: count `{group['count']}`, "
            f"mse `{group['actor_teacher_mse_mean']}`, "
            f"actor abs `{group['actor_clip_abs_mean']}`, "
            f"teacher abs `{group['teacher_abs_mean']}`, "
            f"max disp xy `{group['tap_max_disp_xy_m']}`, "
            f"max vertical `{group['max_vertical_offset_m']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.",
            "AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=580)
    parser.add_argument("--seed", type=int, default=27901)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--d256_reset_csv_path", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument("--fixed_push_dir_x", type=float, default=1.0)
    parser.add_argument("--fixed_push_dir_y", type=float, default=0.0)
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--tap_stop_after_useful_seen", action="store_true")
    parser.add_argument("--tap_useful_terminate", action="store_true")
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--vertical_gate_mode", choices=("max", "min_contact"), default="max")
    parser.add_argument("--max_actor_teacher_mse_mean", type=float, default=0.10)
    parser.add_argument("--min_actor_teacher_cosine_mean", type=float, default=0.50)
    parser.add_argument("--max_overshoot_seen_rate", type=float, default=0.05)
    parser.add_argument("--max_vertical_offset_m", type=float, default=0.08)
    parser.add_argument("--max_joint_delta_cap_rate", type=float, default=0.25)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact_tag", type=str, default="d279_actor_teacher_trace")
    args = parser.parse_args()

    if int(args.eval_steps) <= 0:
        raise ValueError("--eval_steps must be positive")
    if not args.actor_checkpoint.exists():
        raise FileNotFoundError(args.actor_checkpoint)
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(args.teacher_checkpoint)
    if not args.d256_reset_csv_path.exists():
        raise FileNotFoundError(args.d256_reset_csv_path)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    if args.action_scale is not None:
        env_cfg.action_scale = float(args.action_scale)
    if args.max_joint_delta_per_step_rad is not None:
        env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    env_cfg.fixed_push_dir_x = float(args.fixed_push_dir_x)
    env_cfg.fixed_push_dir_y = float(args.fixed_push_dir_y)
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.tap_stop_after_useful_seen = bool(args.tap_stop_after_useful_seen)
    env_cfg.tap_useful_terminate = bool(args.tap_useful_terminate)
    env_cfg.d256_reset_csv_path = str(args.d256_reset_csv_path)
    env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
    env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)
    env_cfg.bc_teacher_checkpoint_path = str(args.teacher_checkpoint)
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)

    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[actor-teacher-trace] scope=cube10cm_top_view_d279_actor_teacher_trace "
        f"env_id={env_id} training=NO bc_teacher_blend={env_cfg.bc_teacher_blend} "
        f"actor={args.actor_checkpoint} teacher={args.teacher_checkpoint}",
        flush=True,
    )

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    if int(args.eval_steps) >= int(inner.max_episode_length) - 1:
        raise ValueError(
            f"--eval_steps {args.eval_steps} would hit env truncation/reset; "
            f"use <= {int(inner.max_episode_length) - 2}"
        )
    if not getattr(inner, "_bc_teacher_ready", False):
        raise RuntimeError("BC teacher sidecar did not load")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.actor_checkpoint), load_optimizer=False, map_location=inner.device)
    policy = runner.get_inference_policy(device=inner.device)

    inner.episode_length_buf[:] = inner.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        warmup_actions = policy(obs)
        obs, _, _, _ = env.step(warmup_actions)
    print("[actor-teacher-trace] warmup_reset_done", flush=True)

    n = int(inner.num_envs)
    action_dim = int(inner.cfg.action_space)
    action_labels = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    if len(action_labels) != action_dim:
        action_labels = [f"action_{idx}" for idx in range(action_dim)]

    inf = torch.full((n,), math.inf, device=inner.device)
    neg_one = torch.full((n,), -1, dtype=torch.long, device=inner.device)
    min_contact_vertical = inf.clone()
    max_vertical = torch.zeros(n, device=inner.device)
    tcp_threshold_seen = torch.zeros(n, dtype=torch.bool, device=inner.device)
    first_contact_step = neg_one.clone()
    first_reaction_step = neg_one.clone()
    first_useful_step = neg_one.clone()
    first_success_step = neg_one.clone()
    first_overshoot_step = neg_one.clone()
    first_vertical_over_step = neg_one.clone()
    first_contact_vertical_over_step = neg_one.clone()

    env_sum_mse = torch.zeros(n, device=inner.device)
    env_sum_mae = torch.zeros(n, device=inner.device)
    env_sum_cos = torch.zeros(n, device=inner.device)
    env_sum_actor_abs = torch.zeros(n, device=inner.device)
    env_sum_teacher_abs = torch.zeros(n, device=inner.device)
    env_sum_raw_clip_exceed = torch.zeros(n, device=inner.device)
    env_max_actor_abs = torch.zeros(n, device=inner.device)
    env_max_teacher_abs = torch.zeros(n, device=inner.device)

    dim_sum_actor = torch.zeros(action_dim, device=inner.device)
    dim_sum_teacher = torch.zeros(action_dim, device=inner.device)
    dim_sum_gap_abs = torch.zeros(action_dim, device=inner.device)
    dim_sum_gap_sq = torch.zeros(action_dim, device=inner.device)
    dim_sum_actor_signed = torch.zeros(action_dim, device=inner.device)
    dim_sum_teacher_signed = torch.zeros(action_dim, device=inner.device)

    reward_finite_all = True
    obs_finite_all = _obs_is_finite(torch, obs)
    action_finite_all = True
    step_rows: list[dict[str, float | int]] = []
    cap_trace: list[float] = []
    actor_clip_abs_mean_trace: list[float] = []
    actor_clip_abs_max_trace: list[float] = []
    teacher_abs_mean_trace: list[float] = []
    teacher_abs_max_trace: list[float] = []
    raw_clip_exceed_trace: list[float] = []
    mse_trace: list[float] = []
    cosine_trace: list[float] = []
    env_stop_after_useful_hold_rate_trace: list[float] = []

    for step in range(int(args.eval_steps)):
        with torch.inference_mode():
            actor_raw = policy(obs)
            actor_clip = torch.clamp(actor_raw, -1.0, 1.0)
            traj = inner._bc_teacher_traj()
            phase_alpha = inner._bc_teacher_phase_alpha(traj)
            teacher_raw = inner._bc_teacher_actions()
            teacher_clip = torch.clamp(teacher_raw, -1.0, 1.0)

            action_finite_all = action_finite_all and bool(torch.isfinite(actor_raw).all().detach().cpu().item())
            action_finite_all = action_finite_all and bool(torch.isfinite(teacher_raw).all().detach().cpu().item())

            diff = actor_clip - teacher_clip
            mse = torch.mean(diff * diff, dim=-1)
            mae = torch.mean(torch.abs(diff), dim=-1)
            cosine = F.cosine_similarity(actor_clip, teacher_clip, dim=-1, eps=1.0e-6)
            actor_abs = torch.mean(torch.abs(actor_clip), dim=-1)
            teacher_abs = torch.mean(torch.abs(teacher_clip), dim=-1)
            raw_clip_exceed = torch.mean((torch.abs(actor_raw) > 1.0).float(), dim=-1)

            obs, rewards, _, _ = env.step(actor_raw)

        reward_finite_all = reward_finite_all and bool(torch.isfinite(rewards).all().detach().cpu().item())
        obs_finite_all = obs_finite_all and _obs_is_finite(torch, obs)

        env_sum_mse += mse
        env_sum_mae += mae
        env_sum_cos += cosine
        env_sum_actor_abs += actor_abs
        env_sum_teacher_abs += teacher_abs
        env_sum_raw_clip_exceed += raw_clip_exceed
        env_max_actor_abs = torch.maximum(env_max_actor_abs, torch.max(torch.abs(actor_clip), dim=-1).values)
        env_max_teacher_abs = torch.maximum(env_max_teacher_abs, torch.max(torch.abs(teacher_clip), dim=-1).values)
        dim_sum_actor += torch.sum(torch.abs(actor_clip), dim=0)
        dim_sum_teacher += torch.sum(torch.abs(teacher_clip), dim=0)
        dim_sum_gap_abs += torch.sum(torch.abs(diff), dim=0)
        dim_sum_gap_sq += torch.sum(diff * diff, dim=0)
        dim_sum_actor_signed += torch.sum(actor_clip, dim=0)
        dim_sum_teacher_signed += torch.sum(teacher_clip, dim=0)

        terms = inner._tap_terms()
        contact = terms["tap_contact_proxy"]
        useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
        max_vertical = torch.maximum(max_vertical, terms["tap_contact_vertical_offset_m"])
        min_contact_vertical = torch.where(
            contact,
            torch.minimum(min_contact_vertical, terms["tap_contact_vertical_offset_m"]),
            min_contact_vertical,
        )
        vertical_over = terms["tap_contact_vertical_offset_m"] > float(args.max_vertical_offset_m)
        contact_vertical_over = contact & vertical_over
        tcp_threshold_seen |= terms["tcp_cube_dist"] < float(inner.cfg.contact_slowdown_tcp_dist_m)

        first_contact_step = torch.where(
            (first_contact_step < 0) & inner._tap_contact_seen,
            torch.full_like(first_contact_step, step),
            first_contact_step,
        )
        first_reaction_step = torch.where(
            (first_reaction_step < 0) & inner._tap_reaction_seen,
            torch.full_like(first_reaction_step, step),
            first_reaction_step,
        )
        first_useful_step = torch.where(
            (first_useful_step < 0) & useful_seen,
            torch.full_like(first_useful_step, step),
            first_useful_step,
        )
        first_success_step = torch.where(
            (first_success_step < 0) & inner._tap_success_flag,
            torch.full_like(first_success_step, step),
            first_success_step,
        )
        first_overshoot_step = torch.where(
            (first_overshoot_step < 0) & inner._tap_overshoot_seen,
            torch.full_like(first_overshoot_step, step),
            first_overshoot_step,
        )
        first_vertical_over_step = torch.where(
            (first_vertical_over_step < 0) & vertical_over,
            torch.full_like(first_vertical_over_step, step),
            first_vertical_over_step,
        )
        first_contact_vertical_over_step = torch.where(
            (first_contact_vertical_over_step < 0) & contact_vertical_over,
            torch.full_like(first_contact_vertical_over_step, step),
            first_contact_vertical_over_step,
        )

        cap_mean = _tensor_mean(inner._last_joint_delta_cap_rate)
        env_hold = _tensor_mean(getattr(inner, "_last_tap_stop_after_useful_hold", torch.zeros(n, device=inner.device)))
        actor_clip_abs_mean = _tensor_mean(actor_abs)
        actor_clip_abs_max = _tensor_max(torch.abs(actor_clip))
        teacher_abs_mean = _tensor_mean(teacher_abs)
        teacher_abs_max = _tensor_max(torch.abs(teacher_clip))
        raw_clip_exceed_mean = _tensor_mean(raw_clip_exceed)
        mse_mean = _tensor_mean(mse)
        cosine_mean = _tensor_mean(cosine)
        cap_trace.append(cap_mean)
        env_stop_after_useful_hold_rate_trace.append(env_hold)
        actor_clip_abs_mean_trace.append(actor_clip_abs_mean)
        actor_clip_abs_max_trace.append(actor_clip_abs_max)
        teacher_abs_mean_trace.append(teacher_abs_mean)
        teacher_abs_max_trace.append(teacher_abs_max)
        raw_clip_exceed_trace.append(raw_clip_exceed_mean)
        mse_trace.append(mse_mean)
        cosine_trace.append(cosine_mean)

        row: dict[str, float | int] = {
            "step": step,
            "phase_alpha_mean": _tensor_mean(phase_alpha),
            "phase_alpha_min": _tensor_min(phase_alpha),
            "phase_alpha_max": _tensor_max(phase_alpha),
            "actor_teacher_mse_mean": mse_mean,
            "actor_teacher_mse_max": _tensor_max(mse),
            "actor_teacher_mae_mean": _tensor_mean(mae),
            "actor_teacher_cosine_mean": cosine_mean,
            "actor_clip_abs_mean": actor_clip_abs_mean,
            "actor_clip_abs_max": actor_clip_abs_max,
            "teacher_abs_mean": teacher_abs_mean,
            "teacher_abs_max": teacher_abs_max,
            "actor_raw_abs_mean": _tensor_mean(torch.abs(actor_raw).mean(dim=-1)),
            "actor_raw_abs_max": _tensor_max(torch.abs(actor_raw)),
            "actor_raw_clip_exceed_rate": raw_clip_exceed_mean,
            "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
            "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
            "tap_useful_seen_rate": _tensor_mean(useful_seen.float()),
            "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
            "tap_overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
            "tap_contact_proxy_rate": _tensor_mean(contact.float()),
            "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
            "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
            "tap_contact_vertical_offset_mean_m": _tensor_mean(terms["tap_contact_vertical_offset_m"]),
            "tap_contact_vertical_offset_max_m": _tensor_max(terms["tap_contact_vertical_offset_m"]),
            "tap_contact_face_gap_mean_m": _tensor_mean(terms["tap_contact_face_gap_m"]),
            "tap_contact_lateral_mean_m": _tensor_mean(terms["tap_contact_lateral_m"]),
            "joint_delta_cap_rate_mean": cap_mean,
            "env_stop_after_useful_hold_rate": env_hold,
            "bc_teacher_blend_mean": _tensor_mean(inner._last_bc_teacher_blend),
            "d256_reset_active_rate": _tensor_mean(inner._last_d256_reset_active),
        }
        for idx, label in enumerate(action_labels):
            row[f"{label}_actor_mean"] = _tensor_mean(actor_clip[:, idx])
            row[f"{label}_teacher_mean"] = _tensor_mean(teacher_clip[:, idx])
            row[f"{label}_abs_gap_mean"] = _tensor_mean(torch.abs(diff[:, idx]))
        step_rows.append(row)

    final_terms = inner._tap_terms()
    useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
    steps_f = float(args.eval_steps)
    env_metrics = {
        "actor_teacher_mse_mean": env_sum_mse / steps_f,
        "actor_teacher_mae_mean": env_sum_mae / steps_f,
        "actor_teacher_cosine_mean": env_sum_cos / steps_f,
        "actor_clip_abs_mean": env_sum_actor_abs / steps_f,
        "teacher_abs_mean": env_sum_teacher_abs / steps_f,
        "actor_raw_clip_exceed_rate": env_sum_raw_clip_exceed / steps_f,
        "actor_clip_abs_max": env_max_actor_abs,
        "teacher_abs_max": env_max_teacher_abs,
        "tap_max_disp_along_m": inner._tap_max_disp_along,
        "tap_max_disp_xy_m": inner._tap_max_disp_xy,
        "max_vertical_offset_m": max_vertical,
        "min_contact_vertical_offset_m": min_contact_vertical,
    }

    d256_active = _tensor_mean(inner._last_d256_reset_active)
    bc_blend_last = _tensor_mean(inner._last_bc_teacher_blend)
    actor_teacher_mse_mean = _tensor_mean(env_metrics["actor_teacher_mse_mean"])
    actor_teacher_cosine_mean = _tensor_mean(env_metrics["actor_teacher_cosine_mean"])
    overshoot_rate = _tensor_mean(inner._tap_overshoot_seen.float())
    vertical_max = _tensor_max(max_vertical)
    min_contact_vertical_max = _finite_max(torch, min_contact_vertical)
    if str(args.vertical_gate_mode) == "min_contact":
        vertical_gate_value = min_contact_vertical_max if min_contact_vertical_max is not None else math.inf
    else:
        vertical_gate_value = vertical_max
    cap_rate_last = _tensor_mean(inner._last_joint_delta_cap_rate)
    cap_rate_max = max(cap_trace) if cap_trace else 0.0

    issues: list[str] = []
    if d256_active < 0.99:
        issues.append(f"D256 reset hook inactive: active_rate={d256_active}")
    if abs(bc_blend_last) > 1.0e-6:
        issues.append(f"BC teacher blend must stay zero during trace: {bc_blend_last}")
    if not reward_finite_all or not obs_finite_all or not action_finite_all:
        issues.append("non-finite reward/obs/action observed")
    if actor_teacher_mse_mean > float(args.max_actor_teacher_mse_mean):
        issues.append(f"actor-teacher action MSE above diagnostic threshold: {actor_teacher_mse_mean}")
    if actor_teacher_cosine_mean < float(args.min_actor_teacher_cosine_mean):
        issues.append(f"actor-teacher action cosine below diagnostic threshold: {actor_teacher_cosine_mean}")
    if overshoot_rate > float(args.max_overshoot_seen_rate):
        issues.append(f"tap overshoot seen rate too high: {overshoot_rate}")
    if vertical_gate_value > float(args.max_vertical_offset_m):
        issues.append(
            f"tap vertical offset too high: mode={args.vertical_gate_mode} max={vertical_gate_value}"
        )
    if cap_rate_max > float(args.max_joint_delta_cap_rate):
        issues.append(f"joint delta cap rate too high: max_trace={cap_rate_max}")

    high_action_gap = (
        actor_teacher_mse_mean > float(args.max_actor_teacher_mse_mean)
        or actor_teacher_cosine_mean < float(args.min_actor_teacher_cosine_mean)
    )
    unsafe_physics = (
        overshoot_rate > float(args.max_overshoot_seen_rate)
        or vertical_gate_value > float(args.max_vertical_offset_m)
    )
    if high_action_gap and unsafe_physics:
        diagnostic_class = "actor_teacher_mismatch_plus_unsafe_physics"
    elif high_action_gap:
        diagnostic_class = "actor_teacher_mismatch"
    elif unsafe_physics:
        diagnostic_class = "teacher_like_action_but_unsafe_physics"
    else:
        diagnostic_class = "no_major_trace_blocker"

    if issues:
        verdict = "D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION"
    else:
        verdict = "D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW"

    total_samples = float(int(args.eval_steps) * n)
    per_dim = []
    for idx, label in enumerate(action_labels):
        per_dim.append(
            {
                "dim": idx,
                "label": label,
                "actor_abs_mean": float((dim_sum_actor[idx] / total_samples).detach().cpu().item()),
                "teacher_abs_mean": float((dim_sum_teacher[idx] / total_samples).detach().cpu().item()),
                "abs_gap_mean": float((dim_sum_gap_abs[idx] / total_samples).detach().cpu().item()),
                "mse": float((dim_sum_gap_sq[idx] / total_samples).detach().cpu().item()),
                "actor_signed_mean": float((dim_sum_actor_signed[idx] / total_samples).detach().cpu().item()),
                "teacher_signed_mean": float((dim_sum_teacher_signed[idx] / total_samples).detach().cpu().item()),
            }
        )

    groups = [
        _group_stats(torch, "all", torch.ones(n, dtype=torch.bool, device=inner.device), env_metrics),
        _group_stats(torch, "overshoot", inner._tap_overshoot_seen, env_metrics),
        _group_stats(torch, "no_overshoot", ~inner._tap_overshoot_seen, env_metrics),
        _group_stats(torch, "useful", useful_seen, env_metrics),
        _group_stats(torch, "not_useful", ~useful_seen, env_metrics),
        _group_stats(torch, "vertical_over_threshold", max_vertical > float(args.max_vertical_offset_m), env_metrics),
        _group_stats(torch, "vertical_ok", max_vertical <= float(args.max_vertical_offset_m), env_metrics),
    ]

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "diagnostic_class": diagnostic_class,
        "issues": issues,
        "actor_checkpoint": _rel(args.actor_checkpoint),
        "teacher_checkpoint": _rel(args.teacher_checkpoint),
        "env_id": env_id,
        "num_envs": n,
        "eval_steps": int(args.eval_steps),
        "episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length": int(inner.max_episode_length),
        "action_scale": float(env_cfg.action_scale),
        "max_joint_delta_per_step_rad": float(env_cfg.max_joint_delta_per_step_rad),
        "seed": int(args.seed),
        "d256_reset_csv_path": _rel(args.d256_reset_csv_path),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "tap_stop_after_useful_seen": bool(args.tap_stop_after_useful_seen),
        "tap_useful_terminate": bool(args.tap_useful_terminate),
        "vertical_gate_mode": str(args.vertical_gate_mode),
        "vertical_gate_value_m": vertical_gate_value,
        "env_stop_after_useful_hold_rate_last": (
            env_stop_after_useful_hold_rate_trace[-1] if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "env_stop_after_useful_hold_rate_max_trace": (
            max(env_stop_after_useful_hold_rate_trace) if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "bc_teacher_feature_target_mode": str(args.bc_teacher_feature_target_mode),
        "bc_teacher_phase_timing": str(args.bc_teacher_phase_timing),
        "bc_teacher_blend": 0.0,
        "bc_teacher_imitation_reward_scale": 0.0,
        "bc_teacher_blend_mean_last": bc_blend_last,
        "d256_reset_active_rate": d256_active,
        "d256_reset_episode_index_mean": _tensor_mean(inner._last_d256_reset_episode_index),
        "d256_reset_episode_index_min": _tensor_min(inner._last_d256_reset_episode_index),
        "d256_reset_episode_index_max": _tensor_max(inner._last_d256_reset_episode_index),
        "actor_teacher_mse_mean": actor_teacher_mse_mean,
        "actor_teacher_mse_max_env_mean": _tensor_max(env_metrics["actor_teacher_mse_mean"]),
        "actor_teacher_mae_mean": _tensor_mean(env_metrics["actor_teacher_mae_mean"]),
        "actor_teacher_cosine_mean": actor_teacher_cosine_mean,
        "actor_teacher_cosine_min_env_mean": _tensor_min(env_metrics["actor_teacher_cosine_mean"]),
        "actor_clip_abs_mean_trace_mean": sum(actor_clip_abs_mean_trace) / len(actor_clip_abs_mean_trace),
        "actor_clip_abs_max_trace_max": max(actor_clip_abs_max_trace) if actor_clip_abs_max_trace else 0.0,
        "teacher_abs_mean_trace_mean": sum(teacher_abs_mean_trace) / len(teacher_abs_mean_trace),
        "teacher_abs_max_trace_max": max(teacher_abs_max_trace) if teacher_abs_max_trace else 0.0,
        "actor_raw_clip_exceed_rate_mean": sum(raw_clip_exceed_trace) / len(raw_clip_exceed_trace),
        "actor_raw_clip_exceed_rate_max_trace": max(raw_clip_exceed_trace) if raw_clip_exceed_trace else 0.0,
        "mse_trace_first": mse_trace[0] if mse_trace else None,
        "mse_trace_last": mse_trace[-1] if mse_trace else None,
        "cosine_trace_first": cosine_trace[0] if cosine_trace else None,
        "cosine_trace_last": cosine_trace[-1] if cosine_trace else None,
        "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
        "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
        "tap_useful_seen_rate": _tensor_mean(useful_seen.float()),
        "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
        "tap_overshoot_seen_rate": overshoot_rate,
        "tap_contact_proxy_rate_last": _tensor_mean(final_terms["tap_contact_proxy"].float()),
        "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
        "tap_max_disp_along_max_m": _tensor_max(inner._tap_max_disp_along),
        "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
        "tap_max_disp_xy_max_m": _tensor_max(inner._tap_max_disp_xy),
        "max_vertical_offset_mean_m": _tensor_mean(max_vertical),
        "max_vertical_offset_max_m": vertical_max,
        "min_contact_vertical_offset_mean_m": _finite_mean(torch, min_contact_vertical),
        "min_contact_vertical_offset_min_m": _finite_min(torch, min_contact_vertical),
        "min_contact_vertical_offset_max_m": _finite_max(torch, min_contact_vertical),
        "last_contact_vertical_offset_mean_m": _tensor_mean(final_terms["tap_contact_vertical_offset_m"]),
        "last_contact_vertical_offset_max_m": _tensor_max(final_terms["tap_contact_vertical_offset_m"]),
        "last_contact_face_gap_mean_m": _tensor_mean(final_terms["tap_contact_face_gap_m"]),
        "last_contact_lateral_mean_m": _tensor_mean(final_terms["tap_contact_lateral_m"]),
        "tcp_threshold_contact_seen_rate": _tensor_mean(tcp_threshold_seen.float()),
        "joint_delta_cap_rate_mean_last": cap_rate_last,
        "joint_delta_cap_rate_max_trace": cap_rate_max,
        "reward_finite_all": reward_finite_all,
        "obs_finite_all": obs_finite_all,
        "action_finite_all": action_finite_all,
        "per_dim_action_metrics": per_dim,
        "groups": groups,
    }

    out_dir = args.out_dir
    out_json = out_dir / "actor_teacher_trace_summary_d279.json"
    out_md = out_dir / "actor_teacher_trace_summary_d279.md"
    out_steps_csv = out_dir / "actor_teacher_trace_steps_d279.csv"
    out_env_csv = out_dir / "actor_teacher_trace_envs_d279.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_steps_csv.open("w", newline="") as f:
        fieldnames = list(step_rows[0].keys()) if step_rows else ["step"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    env_rows: list[dict[str, float | int]] = []
    for env_idx in range(n):
        row = {
            "env_idx": env_idx,
            "d256_episode_index": float(inner._last_d256_reset_episode_index[env_idx].detach().cpu().item()),
            "contact_seen": int(inner._tap_contact_seen[env_idx].detach().cpu().item()),
            "reaction_seen": int(inner._tap_reaction_seen[env_idx].detach().cpu().item()),
            "useful_seen": int(useful_seen[env_idx].detach().cpu().item()),
            "success_seen": int(inner._tap_success_flag[env_idx].detach().cpu().item()),
            "overshoot_seen": int(inner._tap_overshoot_seen[env_idx].detach().cpu().item()),
            "first_contact_step": int(first_contact_step[env_idx].detach().cpu().item()),
            "first_reaction_step": int(first_reaction_step[env_idx].detach().cpu().item()),
            "first_useful_step": int(first_useful_step[env_idx].detach().cpu().item()),
            "first_success_step": int(first_success_step[env_idx].detach().cpu().item()),
            "first_overshoot_step": int(first_overshoot_step[env_idx].detach().cpu().item()),
            "first_vertical_over_step": int(first_vertical_over_step[env_idx].detach().cpu().item()),
            "first_contact_vertical_over_step": int(first_contact_vertical_over_step[env_idx].detach().cpu().item()),
            "actor_teacher_mse_mean": float(env_metrics["actor_teacher_mse_mean"][env_idx].detach().cpu().item()),
            "actor_teacher_mae_mean": float(env_metrics["actor_teacher_mae_mean"][env_idx].detach().cpu().item()),
            "actor_teacher_cosine_mean": float(env_metrics["actor_teacher_cosine_mean"][env_idx].detach().cpu().item()),
            "actor_clip_abs_mean": float(env_metrics["actor_clip_abs_mean"][env_idx].detach().cpu().item()),
            "teacher_abs_mean": float(env_metrics["teacher_abs_mean"][env_idx].detach().cpu().item()),
            "actor_raw_clip_exceed_rate": float(env_metrics["actor_raw_clip_exceed_rate"][env_idx].detach().cpu().item()),
            "actor_clip_abs_max": float(env_metrics["actor_clip_abs_max"][env_idx].detach().cpu().item()),
            "teacher_abs_max": float(env_metrics["teacher_abs_max"][env_idx].detach().cpu().item()),
            "tap_max_disp_along_m": float(inner._tap_max_disp_along[env_idx].detach().cpu().item()),
            "tap_max_disp_xy_m": float(inner._tap_max_disp_xy[env_idx].detach().cpu().item()),
            "max_vertical_offset_m": float(max_vertical[env_idx].detach().cpu().item()),
            "min_contact_vertical_offset_m": float(min_contact_vertical[env_idx].detach().cpu().item()),
        }
        env_rows.append(row)
    with out_env_csv.open("w", newline="") as f:
        fieldnames = list(env_rows[0].keys()) if env_rows else ["env_idx"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(env_rows)

    summary["out_json"] = _rel(out_json)
    summary["out_md"] = _rel(out_md)
    summary["out_steps_csv"] = _rel(out_steps_csv)
    summary["out_env_csv"] = _rel(out_env_csv)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)

    print(
        "[actor-teacher-trace] SUMMARY "
        f"verdict={verdict} class={diagnostic_class} "
        f"mse={actor_teacher_mse_mean:.6f} cosine={actor_teacher_cosine_mean:.6f} "
        f"useful={summary['tap_useful_seen_rate']:.6f} overshoot={overshoot_rate:.6f} "
        f"vertical_max={vertical_max:.6f} json={out_json}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
