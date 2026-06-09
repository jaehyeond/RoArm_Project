#!/usr/bin/env python3
"""Tiny PPO smoke runner for the 10 cm cube tap Candidate6 contract.

This script is intentionally not a dataset generator.  It only checks that the
D207 Candidate6 tap contract can be instantiated as the RL task and, when
requested, runs a small local PPO smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = (
    REPO_ROOT
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
DEFAULT_USD = (
    REPO_ROOT
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate6 fixed-contract 10cm cube tap PPO smoke"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=966)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=0)
    parser.add_argument("--num_steps_per_env", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=580)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    parser.add_argument("--runtime_dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_RUNTIME_DIR
        / "cube10cm_tap_rl_candidate6_pilot_ppo_smoke_summary.json",
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=DEFAULT_RUNTIME_DIR
        / "cube10cm_tap_rl_candidate6_pilot_ppo_smoke_summary.out",
    )
    parser.add_argument(
        "--experiment_name",
        default="cube10cm_tap_rl_candidate6_pilot_ppo_smoke",
    )
    parser.add_argument("--fixed_cube_x_m", type=float, default=0.240)
    parser.add_argument("--fixed_cube_y_m", type=float, default=0.000)
    parser.add_argument("--cube_randomization_half_extent_x_m", type=float, default=0.0)
    parser.add_argument("--cube_randomization_half_extent_y_m", type=float, default=0.0)
    parser.add_argument("--policy_target_disp_m", type=float, default=0.006)
    parser.add_argument("--precontact_clearance_m", type=float, default=0.040)
    parser.add_argument("--episode_length_s", type=float, default=6.08)
    parser.add_argument("--step_clip_rad", type=float, default=0.010)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=0.060)
    parser.add_argument("--action_scale", type=float, default=0.050)
    parser.add_argument(
        "--rl_action_mode",
        choices=(
            "joint_delta",
            "candidate6_diffik_residual_joint",
            "candidate8_diffik_target_residual",
        ),
        default="joint_delta",
    )
    parser.add_argument("--candidate6_diffik_push_steps", type=int, default=580)
    parser.add_argument("--candidate6_diffik_residual_scale_rad", type=float, default=0.002)
    parser.add_argument("--candidate6_diffik_lambda", type=float, default=0.010)
    parser.add_argument("--candidate8_diffik_target_residual_forward_m", type=float, default=0.004)
    parser.add_argument("--candidate8_diffik_target_residual_lateral_m", type=float, default=0.012)
    parser.add_argument("--candidate8_diffik_target_residual_height_m", type=float, default=0.004)
    parser.add_argument("--tap_success_terminate", action="store_true")
    parser.add_argument(
        "--candidate6_diffik_no_hold_after_tap_success",
        action="store_true",
        help="Disable Candidate6 base-target hold after tap success.",
    )
    parser.add_argument(
        "--candidate6_diffik_target_base_mode",
        choices=("previous_joint_target", "actual_joint_pos"),
        default="previous_joint_target",
    )
    parser.add_argument(
        "--candidate6_diffik_target_path_mode",
        choices=("near_face_goal", "legacy_far_face_through"),
        default="near_face_goal",
    )
    parser.add_argument(
        "--candidate6_diffik_cube_reference_mode",
        choices=("start_pose", "current_pose"),
        default="start_pose",
    )
    parser.add_argument("--init_at_random_ep_len", action="store_true")
    parser.add_argument(
        "--load_checkpoint",
        type=Path,
        default=None,
        help="Load an existing PPO checkpoint for eval or resume smoke.",
    )
    parser.add_argument(
        "--initial_policy_eval",
        action="store_true",
        help="Evaluate the untrained PPO policy before learning.",
    )
    parser.add_argument(
        "--ppo_init_noise_std",
        type=float,
        default=None,
        help="Optional PPO actor init_noise_std override; default preserves the shared runner cfg.",
    )
    parser.add_argument(
        "--tap_transient_disp_reward_scale",
        type=float,
        default=None,
        help="Optional tap transient displacement reward override; default preserves env cfg.",
    )
    parser.add_argument(
        "--tap_overshoot_penalty_scale",
        type=float,
        default=None,
        help="Optional tap overshoot penalty override; default preserves env cfg.",
    )
    parser.add_argument(
        "--action_penalty_scale",
        type=float,
        default=None,
        help="Optional action penalty override; default preserves env cfg.",
    )
    return parser.parse_args()


def _scalar(value: Any) -> float | bool | int | str | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.float().mean()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (float, int, bool, str)):
        return value
    return None


def _mean_float(value: Any, default: float = 0.0) -> float:
    result = _scalar(value)
    if isinstance(result, bool):
        return float(result)
    if isinstance(result, (float, int)):
        return float(result)
    return default


def _sum_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "sum"):
        value = value.sum()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return default


def _maybe_float(value: Any) -> float | None:
    result = _scalar(value)
    if isinstance(result, bool):
        return float(result)
    if isinstance(result, (float, int)):
        return float(result)
    return None


def _update_min(metrics: dict[str, Any], key: str, value: float | None) -> None:
    if value is None:
        return
    current = metrics.get(key)
    metrics[key] = value if current is None else min(float(current), value)


def _update_max(metrics: dict[str, Any], key: str, value: float | None) -> None:
    if value is None:
        return
    current = metrics.get(key)
    metrics[key] = value if current is None else max(float(current), value)


def _finite_bool(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict) or hasattr(value, "items"):
        try:
            return all(_finite_bool(item) for _, item in value.items())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return all(_finite_bool(item) for item in value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
    except Exception:
        pass
    try:
        return bool(value == value)
    except Exception:
        return False


def _extract_log(extras: dict[str, Any], inner_env: Any) -> dict[str, Any]:
    log = extras.get("log", {})
    if not log:
        log = getattr(inner_env, "extras", {}).get("log", {})
    return log or {}


def _force_reset(env: Any, inner_env: Any, torch: Any) -> Any:
    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()
    action_dim = int(inner_env.cfg.action_space)
    zero_actions = torch.zeros(
        (inner_env.num_envs, action_dim), device=inner_env.device
    )
    obs, _, _, _ = env.step(zero_actions)
    return obs


def _rollout(
    env: Any,
    inner_env: Any,
    torch: Any,
    *,
    steps: int,
    policy: Any | None,
    label: str,
) -> dict[str, Any]:
    obs = _force_reset(env, inner_env, torch)
    action_dim = int(inner_env.cfg.action_space)
    metrics: dict[str, Any] = {
        "label": label,
        "steps_requested": int(steps),
        "steps_executed": 0,
        "reward_mean_sum": 0.0,
        "reward_finite_all": True,
        "obs_finite_all": True,
        "action_finite_all": True,
        "action_abs_max": 0.0,
        "action_abs_mean_max": 0.0,
        "done_count": 0.0,
        "timeout_count": 0.0,
        "tap_contact_seen_max": 0.0,
        "tap_success_max": 0.0,
        "tap_success_event_count": 0.0,
        "tap_success_event_rate_per_env": 0.0,
        "tap_success_episode_rate": None,
        "tap_overshoot_max": 0.0,
        "reaction_seen_max": 0.0,
        "tap_disp_max": 0.0,
        "tap_speed_max": 0.0,
        "tap_contact_proxy_rate_max": 0.0,
        "tap_contact_face_gap_m_final": None,
        "tap_contact_face_gap_abs_min": None,
        "tap_contact_lateral_m_final": None,
        "tap_contact_lateral_m_min": None,
        "tap_contact_vertical_offset_m_final": None,
        "tap_contact_vertical_offset_m_min": None,
        "tcp_cube_dist_m_final": None,
        "tcp_cube_dist_m_min": None,
        "target_lead_limit_rate_max": 0.0,
        "target_lead_abs_max_max": 0.0,
        "joint_delta_abs_max_max": 0.0,
        "joint_delta_cap_rate_max": 0.0,
        "ik_endpoint_reset_rate_min": None,
        "ik_endpoint_reset_rate_max": 0.0,
        "ik_reset_err_mm_max": 0.0,
        "candidate6_diffik_active_rate_max": 0.0,
        "candidate6_diffik_numeric_ok_rate_min": None,
        "candidate6_diffik_numeric_ok_rate_max": 0.0,
        "candidate6_diffik_raw_delta_abs_max_max": 0.0,
        "candidate6_diffik_clipped_delta_abs_max_max": 0.0,
        "candidate6_diffik_step_clip_rate_max": 0.0,
        "candidate6_diffik_residual_abs_max_max": 0.0,
        "candidate6_diffik_hold_success_rate_max": 0.0,
        "candidate8_diffik_target_residual_abs_max_max": 0.0,
        "candidate8_diffik_target_residual_forward_abs_max": 0.0,
        "candidate8_diffik_target_residual_lateral_abs_max": 0.0,
        "candidate8_diffik_target_residual_height_abs_max": 0.0,
    }

    with torch.inference_mode():
        for _ in range(steps):
            if policy is None:
                actions = torch.zeros(
                    (inner_env.num_envs, action_dim), device=inner_env.device
                )
            else:
                actions = policy(obs)
            metrics["action_finite_all"] = bool(
                metrics["action_finite_all"] and _finite_bool(actions)
            )
            metrics["action_abs_max"] = max(
                float(metrics["action_abs_max"]),
                float(actions.detach().abs().max().cpu().item()),
            )
            metrics["action_abs_mean_max"] = max(
                float(metrics["action_abs_mean_max"]),
                float(actions.detach().abs().mean().cpu().item()),
            )
            obs, rewards, dones, extras = env.step(actions)
            metrics["steps_executed"] += 1
            metrics["reward_finite_all"] = bool(
                metrics["reward_finite_all"] and _finite_bool(rewards)
            )
            metrics["obs_finite_all"] = bool(
                metrics["obs_finite_all"] and _finite_bool(obs)
            )
            metrics["reward_mean_sum"] += _mean_float(rewards)
            metrics["done_count"] += _sum_float(dones)
            metrics["timeout_count"] += _sum_float(extras.get("time_outs"))

            log = _extract_log(extras, inner_env)
            metrics["tap_contact_seen_max"] = max(
                float(metrics["tap_contact_seen_max"]),
                _mean_float(log.get("cube_tap_contact_seen_rate")),
            )
            metrics["tap_success_max"] = max(
                float(metrics["tap_success_max"]),
                _mean_float(log.get("cube_tap_success_rate")),
            )
            success_event_count = _maybe_float(log.get("cube_tap_just_succeeded_count"))
            if success_event_count is None:
                success_event_rate = _maybe_float(log.get("cube_tap_just_succeeded_rate"))
                if success_event_rate is not None:
                    success_event_count = success_event_rate * float(inner_env.num_envs)
            if success_event_count is not None:
                metrics["tap_success_event_count"] += float(success_event_count)
            metrics["tap_overshoot_max"] = max(
                float(metrics["tap_overshoot_max"]),
                _mean_float(log.get("cube_tap_overshoot_seen_rate")),
            )
            metrics["reaction_seen_max"] = max(
                float(metrics["reaction_seen_max"]),
                _mean_float(log.get("cube_tap_reaction_seen_rate")),
            )
            metrics["tap_disp_max"] = max(
                float(metrics["tap_disp_max"]),
                _mean_float(log.get("cube_tap_max_disp_along_m")),
            )
            metrics["tap_speed_max"] = max(
                float(metrics["tap_speed_max"]),
                _mean_float(log.get("cube_tap_speed_mps")),
            )
            _update_max(
                metrics,
                "tap_contact_proxy_rate_max",
                _maybe_float(log.get("cube_tap_contact_proxy_rate")),
            )
            face_gap = _maybe_float(log.get("cube_tap_contact_face_gap_m"))
            metrics["tap_contact_face_gap_m_final"] = face_gap
            _update_min(
                metrics,
                "tap_contact_face_gap_abs_min",
                abs(face_gap) if face_gap is not None else None,
            )
            lateral = _maybe_float(log.get("cube_tap_contact_lateral_m"))
            metrics["tap_contact_lateral_m_final"] = lateral
            _update_min(metrics, "tap_contact_lateral_m_min", lateral)
            vertical = _maybe_float(log.get("cube_tap_contact_vertical_offset_m"))
            metrics["tap_contact_vertical_offset_m_final"] = vertical
            _update_min(metrics, "tap_contact_vertical_offset_m_min", vertical)
            tcp_dist = _maybe_float(log.get("cube_push_tcp_cube_dist_m"))
            metrics["tcp_cube_dist_m_final"] = tcp_dist
            _update_min(metrics, "tcp_cube_dist_m_min", tcp_dist)
            _update_max(
                metrics,
                "target_lead_limit_rate_max",
                _maybe_float(log.get("cube_push_target_lead_limit_rate")),
            )
            _update_max(
                metrics,
                "target_lead_abs_max_max",
                _maybe_float(log.get("cube_push_target_lead_abs_max")),
            )
            _update_max(
                metrics,
                "joint_delta_abs_max_max",
                _maybe_float(log.get("cube_push_joint_delta_abs_max")),
            )
            _update_max(
                metrics,
                "joint_delta_cap_rate_max",
                _maybe_float(log.get("cube_push_joint_delta_cap_rate")),
            )
            ik_reset_rate = _maybe_float(log.get("cube_push_ik_endpoint_reset_rate"))
            _update_min(metrics, "ik_endpoint_reset_rate_min", ik_reset_rate)
            _update_max(metrics, "ik_endpoint_reset_rate_max", ik_reset_rate)
            _update_max(
                metrics,
                "ik_reset_err_mm_max",
                _maybe_float(log.get("cube_push_ik_reset_err_mm")),
            )
            active_rate = _maybe_float(log.get("cube_push_candidate6_diffik_active_rate"))
            numeric_ok_rate = _maybe_float(log.get("cube_push_candidate6_diffik_numeric_ok_rate"))
            _update_max(metrics, "candidate6_diffik_active_rate_max", active_rate)
            _update_max(metrics, "candidate6_diffik_numeric_ok_rate_max", numeric_ok_rate)
            if active_rate is not None and active_rate > 0.0:
                _update_min(metrics, "candidate6_diffik_numeric_ok_rate_min", numeric_ok_rate)
            _update_max(
                metrics,
                "candidate6_diffik_raw_delta_abs_max_max",
                _maybe_float(log.get("cube_push_candidate6_diffik_raw_delta_abs_max")),
            )
            _update_max(
                metrics,
                "candidate6_diffik_clipped_delta_abs_max_max",
                _maybe_float(log.get("cube_push_candidate6_diffik_clipped_delta_abs_max")),
            )
            _update_max(
                metrics,
                "candidate6_diffik_step_clip_rate_max",
                _maybe_float(log.get("cube_push_candidate6_diffik_step_clip_rate")),
            )
            _update_max(
                metrics,
                "candidate6_diffik_residual_abs_max_max",
                _maybe_float(log.get("cube_push_candidate6_diffik_residual_abs_max")),
            )
            _update_max(
                metrics,
                "candidate6_diffik_hold_success_rate_max",
                _maybe_float(log.get("cube_push_candidate6_diffik_hold_success_rate")),
            )
            _update_max(
                metrics,
                "candidate8_diffik_target_residual_abs_max_max",
                _maybe_float(log.get("cube_push_candidate8_diffik_target_residual_abs_max")),
            )
            _update_max(
                metrics,
                "candidate8_diffik_target_residual_forward_abs_max",
                _maybe_float(log.get("cube_push_candidate8_diffik_target_residual_forward_abs")),
            )
            _update_max(
                metrics,
                "candidate8_diffik_target_residual_lateral_abs_max",
                _maybe_float(log.get("cube_push_candidate8_diffik_target_residual_lateral_abs")),
            )
            _update_max(
                metrics,
                "candidate8_diffik_target_residual_height_abs_max",
                _maybe_float(log.get("cube_push_candidate8_diffik_target_residual_height_abs")),
            )

    metrics["reward_mean_per_step"] = (
        float(metrics["reward_mean_sum"]) / max(1, int(metrics["steps_executed"]))
    )
    metrics["tap_success_event_rate_per_env"] = min(
        1.0,
        float(metrics["tap_success_event_count"]) / max(1.0, float(inner_env.num_envs)),
    )
    if float(metrics["done_count"]) > 0.0:
        metrics["tap_success_episode_rate"] = min(
            1.0,
            float(metrics["tap_success_event_count"]) / float(metrics["done_count"]),
        )
    return metrics


def _has_task_success(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return bool(
        float(metrics.get("tap_success_max") or 0.0) > 0.0
        or float(metrics.get("tap_success_event_count") or 0.0) > 0.0
    )


def _success_episode_metric(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    value = metrics.get("tap_success_episode_rate")
    if value is None:
        value = metrics.get("tap_success_event_rate_per_env")
    if value is None:
        return None
    return float(value)


def _apply_candidate6_contract(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    cfg.scene.num_envs = int(args.num_envs)
    cfg.seed = int(args.seed)
    cfg.sim.device = args.device
    cfg.robot.spawn.usd_path = str(args.robot_usd_path)

    rand_x = float(args.cube_randomization_half_extent_x_m)
    rand_y = float(args.cube_randomization_half_extent_y_m)
    if rand_x < 0.0 or rand_y < 0.0:
        raise ValueError("cube randomization half extents must be non-negative")
    cfg.cube_x_min = float(args.fixed_cube_x_m) - rand_x
    cfg.cube_x_max = float(args.fixed_cube_x_m) + rand_x
    cfg.cube_y_min = float(args.fixed_cube_y_m) - rand_y
    cfg.cube_y_max = float(args.fixed_cube_y_m) + rand_y
    cfg.fixed_push_dir_x = 1.0
    cfg.fixed_push_dir_y = 0.0
    cfg.cube_push_target_disp_m = float(args.policy_target_disp_m)

    cfg.tap_contact_proxy_mode = "link5_collision_aabb"
    cfg.tool_contact_proxy_mode = "hand_tcp"
    cfg.tap_success_terminate = bool(args.tap_success_terminate)
    cfg.tap_overshoot_terminate = True
    if args.tap_transient_disp_reward_scale is not None:
        cfg.tap_transient_disp_reward_scale = float(args.tap_transient_disp_reward_scale)
    if args.tap_overshoot_penalty_scale is not None:
        cfg.tap_overshoot_penalty_scale = float(args.tap_overshoot_penalty_scale)
    if args.action_penalty_scale is not None:
        cfg.action_penalty_scale = float(args.action_penalty_scale)

    cfg.episode_length_s = float(args.episode_length_s)
    cfg.ik_endpoint_reset = True
    cfg.ik_reset_jitter_rad = 0.0
    cfg.ik_precontact_clearance_m = float(args.precontact_clearance_m)
    cfg.ik_tcp_top_margin_m = -0.050
    cfg.scripted_teacher_blend = 0.0
    cfg.scripted_teacher_horizon_frac = 1.0
    cfg.scripted_teacher_goal_push_m = float(args.policy_target_disp_m)

    cfg.rl_action_mode = str(args.rl_action_mode)
    if cfg.rl_action_mode == "candidate8_diffik_target_residual":
        cfg.action_space = 3
    cfg.candidate6_diffik_goal_push_m = float(args.policy_target_disp_m)
    cfg.candidate6_diffik_push_steps = int(args.candidate6_diffik_push_steps)
    cfg.candidate6_diffik_step_clip_rad = float(args.step_clip_rad)
    cfg.candidate6_diffik_lambda = float(args.candidate6_diffik_lambda)
    cfg.candidate6_diffik_residual_scale_rad = float(args.candidate6_diffik_residual_scale_rad)
    cfg.candidate6_diffik_hold_after_tap_success = not bool(args.candidate6_diffik_no_hold_after_tap_success)
    cfg.candidate6_diffik_target_base_mode = str(args.candidate6_diffik_target_base_mode)
    cfg.candidate6_diffik_target_path_mode = str(args.candidate6_diffik_target_path_mode)
    cfg.candidate6_diffik_cube_reference_mode = str(args.candidate6_diffik_cube_reference_mode)
    cfg.candidate8_diffik_target_residual_forward_m = float(args.candidate8_diffik_target_residual_forward_m)
    cfg.candidate8_diffik_target_residual_lateral_m = float(args.candidate8_diffik_target_residual_lateral_m)
    cfg.candidate8_diffik_target_residual_height_m = float(args.candidate8_diffik_target_residual_height_m)
    cfg.candidate8_diffik_target_residual_zero_after_contact = False
    cfg.candidate8_diffik_target_residual_zero_after_reaction = False
    cfg.candidate8_diffik_target_residual_zero_after_disp_m = 0.0
    cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    cfg.action_scale = float(args.action_scale)
    if int(args.num_envs) < 8:
        cfg.scene.clone_in_fabric = False

    return {
        "cube_xy_m": [float(args.fixed_cube_x_m), float(args.fixed_cube_y_m)],
        "cube_x_range_m": [cfg.cube_x_min, cfg.cube_x_max],
        "cube_y_range_m": [cfg.cube_y_min, cfg.cube_y_max],
        "cube_randomization_half_extent_m": [rand_x, rand_y],
        "fixed_push_dir": [cfg.fixed_push_dir_x, cfg.fixed_push_dir_y],
        "policy_target_disp_m": cfg.cube_push_target_disp_m,
        "tap_contact_proxy_mode": cfg.tap_contact_proxy_mode,
        "tool_contact_proxy_mode": cfg.tool_contact_proxy_mode,
        "tap_success_terminate": cfg.tap_success_terminate,
        "tap_overshoot_terminate": cfg.tap_overshoot_terminate,
        "tap_transient_disp_reward_scale": cfg.tap_transient_disp_reward_scale,
        "tap_overshoot_penalty_scale": cfg.tap_overshoot_penalty_scale,
        "action_penalty_scale": cfg.action_penalty_scale,
        "episode_length_s": cfg.episode_length_s,
        "precontact_clearance_m": cfg.ik_precontact_clearance_m,
        "rl_action_mode": cfg.rl_action_mode,
        "policy_action_space": int(cfg.action_space),
        "step_clip_rad": cfg.candidate6_diffik_step_clip_rad,
        "candidate6_diffik_goal_push_m": cfg.candidate6_diffik_goal_push_m,
        "candidate6_diffik_push_steps": cfg.candidate6_diffik_push_steps,
        "candidate6_diffik_lambda": cfg.candidate6_diffik_lambda,
        "candidate6_diffik_residual_scale_rad": cfg.candidate6_diffik_residual_scale_rad,
        "candidate6_diffik_hold_after_tap_success": cfg.candidate6_diffik_hold_after_tap_success,
        "candidate6_diffik_target_base_mode": cfg.candidate6_diffik_target_base_mode,
        "candidate6_diffik_target_path_mode": cfg.candidate6_diffik_target_path_mode,
        "candidate6_diffik_cube_reference_mode": cfg.candidate6_diffik_cube_reference_mode,
        "candidate8_diffik_target_residual_forward_m": cfg.candidate8_diffik_target_residual_forward_m,
        "candidate8_diffik_target_residual_lateral_m": cfg.candidate8_diffik_target_residual_lateral_m,
        "candidate8_diffik_target_residual_height_m": cfg.candidate8_diffik_target_residual_height_m,
        "joint_target_lead_limit_rad": cfg.joint_target_lead_limit_rad,
        "action_scale": cfg.action_scale,
        "scripted_teacher_blend": cfg.scripted_teacher_blend,
        "robot_usd_path": cfg.robot.spawn.usd_path,
    }


def _contract_violations(contract: dict[str, Any], args: argparse.Namespace) -> list[str]:
    expected = {
        "cube_xy_m": [float(args.fixed_cube_x_m), float(args.fixed_cube_y_m)],
        "fixed_push_dir": [1.0, 0.0],
        "policy_target_disp_m": float(args.policy_target_disp_m),
        "tap_contact_proxy_mode": "link5_collision_aabb",
        "tool_contact_proxy_mode": "hand_tcp",
        "tap_success_terminate": bool(args.tap_success_terminate),
        "tap_overshoot_terminate": True,
        "tap_transient_disp_reward_scale": (
            40.0
            if args.tap_transient_disp_reward_scale is None
            else float(args.tap_transient_disp_reward_scale)
        ),
        "tap_overshoot_penalty_scale": (
            12.0
            if args.tap_overshoot_penalty_scale is None
            else float(args.tap_overshoot_penalty_scale)
        ),
        "action_penalty_scale": 0.005 if args.action_penalty_scale is None else float(args.action_penalty_scale),
        "precontact_clearance_m": 0.040,
        "step_clip_rad": 0.010,
        "joint_target_lead_limit_rad": 0.060,
        "scripted_teacher_blend": 0.0,
        "rl_action_mode": str(args.rl_action_mode),
        "policy_action_space": 3 if str(args.rl_action_mode) == "candidate8_diffik_target_residual" else 6,
        "candidate6_diffik_goal_push_m": float(args.policy_target_disp_m),
        "candidate6_diffik_push_steps": int(args.candidate6_diffik_push_steps),
        "candidate6_diffik_residual_scale_rad": float(args.candidate6_diffik_residual_scale_rad),
        "candidate6_diffik_hold_after_tap_success": not bool(args.candidate6_diffik_no_hold_after_tap_success),
        "candidate6_diffik_target_base_mode": str(args.candidate6_diffik_target_base_mode),
        "candidate6_diffik_target_path_mode": str(args.candidate6_diffik_target_path_mode),
        "candidate6_diffik_cube_reference_mode": str(args.candidate6_diffik_cube_reference_mode),
        "candidate8_diffik_target_residual_forward_m": float(
            args.candidate8_diffik_target_residual_forward_m
        ),
        "candidate8_diffik_target_residual_lateral_m": float(
            args.candidate8_diffik_target_residual_lateral_m
        ),
        "candidate8_diffik_target_residual_height_m": float(
            args.candidate8_diffik_target_residual_height_m
        ),
    }
    violations: list[str] = []
    for key, expected_value in expected.items():
        actual = contract.get(key)
        if isinstance(expected_value, list):
            if any(abs(float(a) - float(e)) > 1e-9 for a, e in zip(actual, expected_value)):
                violations.append(f"{key}={actual} expected={expected_value}")
        elif isinstance(expected_value, float):
            if abs(float(actual) - expected_value) > 1e-9:
                violations.append(f"{key}={actual} expected={expected_value}")
        elif actual != expected_value:
            violations.append(f"{key}={actual} expected={expected_value}")
    return violations


def _write_summary(summary: dict[str, Any], summary_json: Path, summary_out: Path) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    pre = summary.get("pre_eval") or {}
    initial = summary.get("initial_policy_eval") or {}
    post = summary.get("post_eval") or {}
    candidate8_base_relative = summary.get("candidate8_base_relative") or {}
    lines = [
        "candidate6_pilot_ppo_smoke_audit=v2 "
        f"max_iterations={summary['max_iterations']} "
        f"num_envs={summary['num_envs']} seed={summary['seed']} "
        f"device={summary['device']}",
        "ppo_config "
        f"init_noise_std={summary.get('ppo_config', {}).get('init_noise_std')}",
        "contract "
        f"cube_xy_m={summary['contract']['cube_xy_m']} "
        f"cube_x_range_m={summary['contract']['cube_x_range_m']} "
        f"cube_y_range_m={summary['contract']['cube_y_range_m']} "
        f"cube_randomization_half_extent_m={summary['contract']['cube_randomization_half_extent_m']} "
        f"push_dir={summary['contract']['fixed_push_dir']} "
        f"proxy={summary['contract']['tap_contact_proxy_mode']} "
        f"rl_action_mode={summary['contract']['rl_action_mode']} "
        f"policy_action_space={summary['contract']['policy_action_space']} "
        f"tap_success_terminate={summary['contract']['tap_success_terminate']} "
        f"tap_transient_disp_reward_scale={summary['contract']['tap_transient_disp_reward_scale']} "
        f"tap_overshoot_penalty_scale={summary['contract']['tap_overshoot_penalty_scale']} "
        f"action_penalty_scale={summary['contract']['action_penalty_scale']} "
        f"policy_target_disp_m={summary['contract']['policy_target_disp_m']} "
        f"step_clip_rad={summary['contract']['step_clip_rad']} "
        f"lead_limit_rad={summary['contract']['joint_target_lead_limit_rad']} "
        f"residual_scale_rad={summary['contract']['candidate6_diffik_residual_scale_rad']} "
        f"hold_after_success={summary['contract']['candidate6_diffik_hold_after_tap_success']} "
        f"target_base_mode={summary['contract']['candidate6_diffik_target_base_mode']} "
        f"target_path_mode={summary['contract']['candidate6_diffik_target_path_mode']} "
        f"cube_reference_mode={summary['contract']['candidate6_diffik_cube_reference_mode']} "
        f"candidate8_forward_m={summary['contract']['candidate8_diffik_target_residual_forward_m']} "
        f"candidate8_lateral_m={summary['contract']['candidate8_diffik_target_residual_lateral_m']} "
        f"candidate8_height_m={summary['contract']['candidate8_diffik_target_residual_height_m']} "
        f"teacher_blend={summary['contract']['scripted_teacher_blend']} "
        f"violations={len(summary['contract_violations'])}",
        "zero_policy_pre_eval "
        f"finite={pre.get('reward_finite_all')} "
        f"tap_success_max={pre.get('tap_success_max')} "
        f"success_event_count={pre.get('tap_success_event_count')} "
        f"success_event_rate_per_env={pre.get('tap_success_event_rate_per_env')} "
        f"success_episode_rate={pre.get('tap_success_episode_rate')} "
        f"contact_seen_max={pre.get('tap_contact_seen_max')} "
        f"reaction_seen_max={pre.get('reaction_seen_max')} "
        f"overshoot_max={pre.get('tap_overshoot_max')} "
        f"reward_mean_per_step={pre.get('reward_mean_per_step')} "
        f"face_gap_final_m={pre.get('tap_contact_face_gap_m_final')} "
        f"tcp_dist_min_m={pre.get('tcp_cube_dist_m_min')} "
        f"ik_reset_rate_min={pre.get('ik_endpoint_reset_rate_min')} "
        f"ik_reset_err_mm_max={pre.get('ik_reset_err_mm_max')} "
        f"candidate6_active_rate_max={pre.get('candidate6_diffik_active_rate_max')} "
        f"candidate6_numeric_ok_rate_min={pre.get('candidate6_diffik_numeric_ok_rate_min')} "
        f"candidate6_hold_success_rate_max={pre.get('candidate6_diffik_hold_success_rate_max')}",
        "initial_ppo_policy_eval "
        f"enabled={bool(initial)} "
        f"finite={initial.get('reward_finite_all')} "
        f"tap_success_max={initial.get('tap_success_max')} "
        f"success_event_count={initial.get('tap_success_event_count')} "
        f"success_event_rate_per_env={initial.get('tap_success_event_rate_per_env')} "
        f"success_episode_rate={initial.get('tap_success_episode_rate')} "
        f"contact_seen_max={initial.get('tap_contact_seen_max')} "
        f"reaction_seen_max={initial.get('reaction_seen_max')} "
        f"overshoot_max={initial.get('tap_overshoot_max')} "
        f"face_gap_final_m={initial.get('tap_contact_face_gap_m_final')} "
        f"tcp_dist_min_m={initial.get('tcp_cube_dist_m_min')} "
        f"ik_reset_rate_min={initial.get('ik_endpoint_reset_rate_min')} "
        f"ik_reset_err_mm_max={initial.get('ik_reset_err_mm_max')} "
        f"candidate6_active_rate_max={initial.get('candidate6_diffik_active_rate_max')} "
        f"candidate6_numeric_ok_rate_min={initial.get('candidate6_diffik_numeric_ok_rate_min')} "
        f"candidate6_hold_success_rate_max={initial.get('candidate6_diffik_hold_success_rate_max')}",
        "training "
        f"ran={summary['training_ran']} "
        f"checkpoint_exists={summary['checkpoint_exists']} "
        f"log_dir={summary.get('log_dir')}",
        "post_eval "
        f"enabled={bool(post)} "
        f"finite={post.get('reward_finite_all')} "
        f"tap_success_max={post.get('tap_success_max')} "
        f"success_event_count={post.get('tap_success_event_count')} "
        f"success_event_rate_per_env={post.get('tap_success_event_rate_per_env')} "
        f"success_episode_rate={post.get('tap_success_episode_rate')} "
        f"contact_seen_max={post.get('tap_contact_seen_max')} "
        f"reaction_seen_max={post.get('reaction_seen_max')} "
        f"overshoot_max={post.get('tap_overshoot_max')} "
        f"reward_mean_per_step={post.get('reward_mean_per_step')} "
        f"face_gap_final_m={post.get('tap_contact_face_gap_m_final')} "
        f"tcp_dist_min_m={post.get('tcp_cube_dist_m_min')} "
        f"lead_limit_rate_max={post.get('target_lead_limit_rate_max')} "
        f"joint_delta_cap_rate_max={post.get('joint_delta_cap_rate_max')} "
        f"ik_reset_rate_min={post.get('ik_endpoint_reset_rate_min')} "
        f"ik_reset_err_mm_max={post.get('ik_reset_err_mm_max')} "
        f"candidate6_active_rate_max={post.get('candidate6_diffik_active_rate_max')} "
        f"candidate6_numeric_ok_rate_min={post.get('candidate6_diffik_numeric_ok_rate_min')} "
        f"candidate6_step_clip_rate_max={post.get('candidate6_diffik_step_clip_rate_max')} "
        f"candidate6_residual_abs_max_max={post.get('candidate6_diffik_residual_abs_max_max')} "
        f"candidate8_target_residual_abs_max_max={post.get('candidate8_diffik_target_residual_abs_max_max')} "
        f"candidate8_forward_abs_max={post.get('candidate8_diffik_target_residual_forward_abs_max')} "
        f"candidate8_lateral_abs_max={post.get('candidate8_diffik_target_residual_lateral_abs_max')} "
        f"candidate8_height_abs_max={post.get('candidate8_diffik_target_residual_height_abs_max')} "
        f"candidate6_hold_success_rate_max={post.get('candidate6_diffik_hold_success_rate_max')}",
        "verdict "
        f"preflight_pass={summary['preflight_pass']} "
        f"bridge_preflight_pass={summary['bridge_preflight_pass']} "
        f"zero_policy_task_pass={summary['zero_policy_task_pass']} "
        f"training_smoke_pass={summary['training_smoke_pass']} "
        f"policy_task_pass={summary['policy_task_pass']} "
        "large_dataset_rl_roarm_unblocked=NO "
        "action_teacher_dataset=NO",
        "candidate8_base_relative "
        f"enabled={bool(candidate8_base_relative)} "
        f"pre_success_episode_rate={candidate8_base_relative.get('pre_success_episode_rate')} "
        f"post_success_episode_rate={candidate8_base_relative.get('post_success_episode_rate')} "
        f"success_episode_delta={candidate8_base_relative.get('success_episode_delta')} "
        f"pre_overshoot_max={candidate8_base_relative.get('pre_overshoot_max')} "
        f"post_overshoot_max={candidate8_base_relative.get('post_overshoot_max')} "
        f"overshoot_delta={candidate8_base_relative.get('overshoot_delta')} "
        f"target_residual_abs_max_max={candidate8_base_relative.get('target_residual_abs_max_max')} "
        f"target_residual_forward_abs_max={candidate8_base_relative.get('target_residual_forward_abs_max')} "
        f"target_residual_lateral_abs_max={candidate8_base_relative.get('target_residual_lateral_abs_max')} "
        f"target_residual_height_abs_max={candidate8_base_relative.get('target_residual_height_abs_max')} "
        f"signal_seen={candidate8_base_relative.get('signal_seen')} "
        f"l1_health_pass={candidate8_base_relative.get('l1_health_pass')} "
        f"l2_scale_candidate={candidate8_base_relative.get('l2_scale_candidate')}",
        f"outputs summary_json={summary_json} summary_out={summary_out}",
    ]
    summary_out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")
    print(
        "[candidate6-ppo-smoke] start "
        f"seed={args.seed} num_envs={args.num_envs} "
        f"max_iterations={args.max_iterations} robot_usd_path={args.robot_usd_path}",
        flush=True,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
    simulation_app = app_launcher.app
    print("[candidate6-ppo-smoke] app_launched", flush=True)

    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner

        import roarm_rl  # noqa: F401 - registers Gym environments
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
        from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg

        env_cfg = RoArmCubeTap10cmEnvCfg()
        contract = _apply_candidate6_contract(env_cfg, args)
        contract_violations = _contract_violations(contract, args)
        print(
            "[candidate6-ppo-smoke] contract_applied "
            f"violations={len(contract_violations)}",
            flush=True,
        )

        env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
        print("[candidate6-ppo-smoke] gym_env_created", flush=True)
        env = RslRlVecEnvWrapper(env)
        inner_env = env.unwrapped
        print("[candidate6-ppo-smoke] rsl_wrapper_created", flush=True)

        ppo_cfg = RoArmPickPPORunnerCfg()
        if args.ppo_init_noise_std is not None:
            ppo_cfg.policy.init_noise_std = float(args.ppo_init_noise_std)
        ppo_cfg.max_iterations = int(args.max_iterations)
        ppo_cfg.num_steps_per_env = int(args.num_steps_per_env)
        ppo_cfg.save_interval = int(args.save_interval)
        ppo_cfg.experiment_name = args.experiment_name
        ppo_cfg.run_name = f"seed{args.seed}_env{args.num_envs}_it{args.max_iterations}"

        log_dir_path = None
        if int(args.max_iterations) > 0:
            log_dir_path = (
                args.runtime_dir
                / "ppo_runs"
                / args.experiment_name
                / ppo_cfg.run_name
            )
            log_dir_path.mkdir(parents=True, exist_ok=True)

        runner = OnPolicyRunner(
            env,
            ppo_cfg.to_dict(),
            log_dir=str(log_dir_path) if log_dir_path else None,
            device=inner_env.device,
        )
        runner.add_git_repo_to_log(__file__)
        print("[candidate6-ppo-smoke] runner_created", flush=True)
        checkpoint_loaded = False
        if args.load_checkpoint is not None:
            if not args.load_checkpoint.exists():
                raise FileNotFoundError(f"checkpoint missing: {args.load_checkpoint}")
            runner.load(
                str(args.load_checkpoint),
                load_optimizer=bool(int(args.max_iterations) > 0),
                map_location=inner_env.device,
            )
            checkpoint_loaded = True
            print(
                f"[candidate6-ppo-smoke] checkpoint_loaded {args.load_checkpoint}",
                flush=True,
            )

        pre_eval = _rollout(
            env,
            inner_env,
            torch,
            steps=int(args.eval_steps),
            policy=None,
            label="zero_policy_pre_eval",
        )
        print("[candidate6-ppo-smoke] zero_policy_pre_eval_done", flush=True)

        initial_policy_eval = None
        if args.initial_policy_eval:
            initial_policy = runner.get_inference_policy(device=inner_env.device)
            initial_policy_eval = _rollout(
                env,
                inner_env,
                torch,
                steps=int(args.eval_steps),
                policy=initial_policy,
                label="initial_ppo_policy_eval",
            )
            print("[candidate6-ppo-smoke] initial_policy_eval_done", flush=True)

        checkpoint_path = args.load_checkpoint if checkpoint_loaded else None
        training_ran = int(args.max_iterations) > 0
        if training_ran:
            print("[candidate6-ppo-smoke] learn_start", flush=True)
            runner.learn(
                num_learning_iterations=int(args.max_iterations),
                init_at_random_ep_len=bool(args.init_at_random_ep_len),
            )
            checkpoint_path = log_dir_path / f"model_{runner.current_learning_iteration}.pt"
            print(
                "[candidate6-ppo-smoke] learn_done "
                f"checkpoint_path={checkpoint_path}",
                flush=True,
            )

        post_eval = None
        if training_ran:
            learned_policy = runner.get_inference_policy(device=inner_env.device)
            post_eval = _rollout(
                env,
                inner_env,
                torch,
                steps=int(args.eval_steps),
                policy=learned_policy,
                label="post_training_policy_eval",
            )
            print("[candidate6-ppo-smoke] post_eval_done", flush=True)
        elif checkpoint_loaded:
            loaded_policy = runner.get_inference_policy(device=inner_env.device)
            post_eval = _rollout(
                env,
                inner_env,
                torch,
                steps=int(args.eval_steps),
                policy=loaded_policy,
                label="loaded_checkpoint_policy_eval",
            )
            print("[candidate6-ppo-smoke] loaded_checkpoint_eval_done", flush=True)

        checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())
        bridge_preflight_pass = True
        if str(args.rl_action_mode) in (
            "candidate6_diffik_residual_joint",
            "candidate8_diffik_target_residual",
        ):
            bridge_preflight_pass = bool(
                float(pre_eval["candidate6_diffik_active_rate_max"]) > 0.0
                and pre_eval["candidate6_diffik_numeric_ok_rate_min"] is not None
                and float(pre_eval["candidate6_diffik_numeric_ok_rate_min"]) >= 0.999
            )
        zero_policy_task_pass = bool(
            _has_task_success(pre_eval)
            and float(pre_eval["tap_overshoot_max"]) <= 0.0
        )
        preflight_pass = bool(
            not contract_violations
            and pre_eval["reward_finite_all"]
            and pre_eval["obs_finite_all"]
            and pre_eval["action_finite_all"]
            and bridge_preflight_pass
        )
        training_smoke_pass = None
        policy_task_pass = None
        if training_ran:
            training_smoke_pass = bool(
                checkpoint_exists
                and post_eval
                and post_eval["reward_finite_all"]
                and post_eval["obs_finite_all"]
                and post_eval["action_finite_all"]
                and float(post_eval["tap_overshoot_max"]) <= 0.0
            )
            policy_task_pass = bool(
                post_eval
                and _has_task_success(post_eval)
                and float(post_eval["tap_overshoot_max"]) <= 0.0
            )
        elif checkpoint_loaded:
            policy_task_pass = bool(
                post_eval
                and _has_task_success(post_eval)
                and float(post_eval["tap_overshoot_max"]) <= 0.0
            )

        candidate8_base_relative = None
        if str(args.rl_action_mode) == "candidate8_diffik_target_residual" and post_eval:
            pre_success = _success_episode_metric(pre_eval)
            post_success = _success_episode_metric(post_eval)
            pre_overshoot = float(pre_eval.get("tap_overshoot_max") or 0.0)
            post_overshoot = float(post_eval.get("tap_overshoot_max") or 0.0)
            target_residual_abs_max = float(
                post_eval.get("candidate8_diffik_target_residual_abs_max_max") or 0.0
            )
            target_residual_forward_abs_max = float(
                post_eval.get("candidate8_diffik_target_residual_forward_abs_max") or 0.0
            )
            target_residual_lateral_abs_max = float(
                post_eval.get("candidate8_diffik_target_residual_lateral_abs_max") or 0.0
            )
            target_residual_height_abs_max = float(
                post_eval.get("candidate8_diffik_target_residual_height_abs_max") or 0.0
            )
            signal_seen = target_residual_abs_max > 1.0e-6
            success_delta = (
                None if pre_success is None or post_success is None else post_success - pre_success
            )
            overshoot_delta = post_overshoot - pre_overshoot
            l1_health_pass = bool(
                post_eval["reward_finite_all"]
                and post_eval["obs_finite_all"]
                and post_eval["action_finite_all"]
                and signal_seen
                and success_delta is not None
                and success_delta >= -0.10
                and overshoot_delta <= 0.10
            )
            l2_scale_candidate = bool(
                success_delta is not None
                and success_delta >= 0.0
                and overshoot_delta <= 0.0
            )
            candidate8_base_relative = {
                "pre_success_episode_rate": pre_success,
                "post_success_episode_rate": post_success,
                "success_episode_delta": success_delta,
                "pre_overshoot_max": pre_overshoot,
                "post_overshoot_max": post_overshoot,
                "overshoot_delta": overshoot_delta,
                "target_residual_abs_max_max": target_residual_abs_max,
                "target_residual_forward_abs_max": target_residual_forward_abs_max,
                "target_residual_lateral_abs_max": target_residual_lateral_abs_max,
                "target_residual_height_abs_max": target_residual_height_abs_max,
                "signal_seen": signal_seen,
                "l1_health_pass": l1_health_pass,
                "l2_scale_candidate": l2_scale_candidate,
            }

        summary = {
            "audit_version": "candidate6_pilot_ppo_smoke_v2",
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "device": args.device,
            "max_iterations": int(args.max_iterations),
            "num_steps_per_env": int(args.num_steps_per_env),
            "eval_steps": int(args.eval_steps),
            "init_at_random_ep_len": bool(args.init_at_random_ep_len),
            "ppo_config": {
                "init_noise_std": float(ppo_cfg.policy.init_noise_std),
            },
            "contract": contract,
            "contract_violations": contract_violations,
            "pre_eval": pre_eval,
            "initial_policy_eval": initial_policy_eval,
            "training_ran": training_ran,
            "checkpoint_loaded": checkpoint_loaded,
            "training_smoke_pass": training_smoke_pass,
            "policy_task_pass": policy_task_pass,
            "preflight_pass": preflight_pass,
            "bridge_preflight_pass": bridge_preflight_pass,
            "zero_policy_task_pass": zero_policy_task_pass,
            "candidate8_base_relative": candidate8_base_relative,
            "post_eval": post_eval,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_exists": checkpoint_exists,
            "log_dir": str(log_dir_path) if log_dir_path else None,
            "large_dataset_rl_roarm_unblocked": False,
            "action_teacher_dataset": False,
        }
        _write_summary(summary, args.summary_json, args.summary_out)
        print(f"[candidate6-ppo-smoke] summary_written {args.summary_out}", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True))

        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
