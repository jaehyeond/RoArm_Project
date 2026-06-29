"""Evaluate a frozen PPO actor on the cube10cm tap D256-reset contract."""
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


def _tensor_max(x) -> float:
    return float(x.detach().float().max().cpu().item())


def _tensor_min(x) -> float:
    return float(x.detach().float().min().cpu().item())


def _scalar_log_value(torch, value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {summary['artifact_tag'].upper()} Teacher-Off Frozen Policy Eval",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- env id: `{summary['env_id']}`",
        f"- steps/envs: `{summary['eval_steps']}` / `{summary['num_envs']}`",
        f"- action scale/max joint delta: `{summary['action_scale']}` / `{summary['max_joint_delta_per_step_rad']}`",
        f"- action smoothing/contact scales: `{summary['action_smoothing_alpha']}` / `{summary['contact_joint_delta_scale']}` / `{summary['fast_cube_joint_delta_scale']}`",
        f"- joint delta reference: `{summary['joint_delta_reference']}`",
        f"- d256 reset active rate: `{summary['d256_reset_active_rate']}`",
        f"- bc teacher blend mean last: `{summary['bc_teacher_blend_mean_last']}`",
        f"- action mode: `{summary['action_mode']}`",
        f"- vertical gate mode/value: `{summary['vertical_gate_mode']}` / `{summary['vertical_gate_value_m']}`",
        f"- env stop/success/useful terminate: `{summary['tap_stop_after_useful_seen']}` / `{summary['tap_success_terminate']}` / `{summary['tap_useful_terminate']}`",
        f"- D256 reset warmup mode: `{summary['d256_reset_warmup_mode']}`",
        f"- env stop after displacement m: `{summary['tap_stop_after_disp_m']}`",
        f"- env contact slowdown uses proxy: `{summary['tap_contact_slowdown_use_proxy']}`",
        f"- done rate mean/max/total: `{summary['done_rate_mean_trace']}` / `{summary['done_rate_max_trace']}` / `{summary['done_count_total']}`",
        f"- RSL-like log contact/useful/success/overshoot mean: `{summary['rsl_log_cube_tap_contact_seen_rate_mean']}` / `{summary['rsl_log_cube_tap_useful_seen_rate_mean']}` / `{summary['rsl_log_cube_tap_success_rate_mean']}` / `{summary['rsl_log_cube_tap_overshoot_seen_rate_mean']}`",
        f"- RSL-like log max disp along/xy mean: `{summary['rsl_log_cube_tap_max_disp_along_m_mean']}` / `{summary['rsl_log_cube_tap_max_disp_xy_m_mean']}`",
        f"- env useful hold rate last/max: `{summary['env_stop_after_useful_hold_rate_last']}` / `{summary['env_stop_after_useful_hold_rate_max_trace']}`",
        f"- env displacement hold rate last/max: `{summary['env_stop_after_disp_hold_rate_last']}` / `{summary['env_stop_after_disp_hold_rate_max_trace']}`",
        f"- zero actions after useful seen: `{summary['zero_actions_after_useful_seen']}`",
        f"- exec action clip abs: `{summary['exec_action_clip_abs']}`",
        f"- useful action hold rate last/max: `{summary['useful_action_hold_rate_last']}` / `{summary['useful_action_hold_rate_max_trace']}`",
        f"- contact/useful/reaction seen: `{summary['tap_contact_seen_rate']}` / `{summary['tap_useful_seen_rate']}` / `{summary['tap_reaction_seen_rate']}`",
        f"- success rate: `{summary['tap_success_rate']}`",
        f"- overshoot seen rate: `{summary['tap_overshoot_seen_rate']}`",
        f"- max disp along mean/max: `{summary['tap_max_disp_along_mean_m']}` / `{summary['tap_max_disp_along_max_m']}`",
        f"- max disp xy mean/max: `{summary['tap_max_disp_xy_mean_m']}` / `{summary['tap_max_disp_xy_max_m']}`",
        f"- max disp along >=1mm/>=3mm rate: `{summary['tap_max_disp_along_ge_1mm_rate']}` / `{summary['tap_max_disp_along_ge_3mm_rate']}`",
        f"- max disp xy >=1mm/>=3mm rate: `{summary['tap_max_disp_xy_ge_1mm_rate']}` / `{summary['tap_max_disp_xy_ge_3mm_rate']}`",
        f"- displacement gate mean/max along: `{summary['min_mean_disp_along_m']}` / `{summary['min_max_disp_along_m']}`",
        f"- displacement gate mean/max xy: `{summary['min_mean_disp_xy_m']}` / `{summary['min_max_disp_xy_m']}`",
        f"- displacement gate >=1mm rate along/xy: `{summary['min_disp_along_ge_1mm_rate']}` / `{summary['min_disp_xy_ge_1mm_rate']}`",
        f"- min contact vertical offset mean/min/max: `{summary['min_contact_vertical_offset_mean_m']}` / `{summary['min_contact_vertical_offset_min_m']}` / `{summary['min_contact_vertical_offset_max_m']}`",
        f"- last contact vertical offset mean/max: `{summary['last_contact_vertical_offset_mean_m']}` / `{summary['last_contact_vertical_offset_max_m']}`",
        f"- raw TCP-threshold contact seen rate: `{summary['tcp_threshold_contact_seen_rate']}`",
        f"- joint delta cap rate mean/max trace: `{summary['joint_delta_cap_rate_mean_last']}` / `{summary['joint_delta_cap_rate_max_trace']}`",
        f"- policy action abs mean/max trace: `{summary['policy_action_abs_mean_trace_mean']}` / `{summary['policy_action_abs_max_trace_max']}`",
        f"- reward finite all: `{summary['reward_finite_all']}`",
        f"- obs finite all: `{summary['obs_finite_all']}`",
        f"- action finite all: `{summary['action_finite_all']}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.",
            "For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=580)
    parser.add_argument("--seed", type=int, default=27701)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--action_smoothing_alpha", type=float, default=None)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=None)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=None)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=None)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default=None)
    parser.add_argument("--d256_reset_csv_path", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument(
        "--d256_reset_warmup_mode",
        choices=("direct_reset", "force_step_zero", "force_step_policy"),
        default="direct_reset",
    )
    parser.add_argument("--fixed_push_dir_x", type=float, default=1.0)
    parser.add_argument("--fixed_push_dir_y", type=float, default=0.0)
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--bc_teacher_checkpoint_path", type=Path, default=None)
    parser.add_argument("--bc_teacher_blend", type=float, default=0.0)
    parser.add_argument("--bc_teacher_imitation_reward_scale", type=float, default=0.0)
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--tap_stop_after_useful_seen", action="store_true")
    parser.add_argument("--tap_stop_after_disp_m", type=float, default=0.0)
    parser.add_argument("--tap_contact_slowdown_use_proxy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tap_success_terminate", action="store_true")
    parser.add_argument("--tap_useful_terminate", action="store_true")
    parser.add_argument("--zero_actions_after_useful_seen", action="store_true")
    parser.add_argument("--action_mode", choices=("inference", "ppo_stochastic"), default="inference")
    parser.add_argument("--require_rsl_log_gate", action="store_true")
    parser.add_argument("--exec_action_clip_abs", type=float, default=1.0)
    parser.add_argument("--min_useful_seen_rate", type=float, default=0.01)
    parser.add_argument("--max_overshoot_seen_rate", type=float, default=0.05)
    parser.add_argument("--max_joint_delta_cap_rate", type=float, default=0.25)
    parser.add_argument("--max_vertical_offset_m", type=float, default=0.08)
    parser.add_argument("--min_mean_disp_along_m", type=float, default=0.0)
    parser.add_argument("--min_max_disp_along_m", type=float, default=0.0)
    parser.add_argument("--min_mean_disp_xy_m", type=float, default=0.0)
    parser.add_argument("--min_max_disp_xy_m", type=float, default=0.0)
    parser.add_argument("--min_disp_along_ge_1mm_rate", type=float, default=0.0)
    parser.add_argument("--min_disp_xy_ge_1mm_rate", type=float, default=0.0)
    parser.add_argument("--vertical_gate_mode", choices=("last", "min_contact"), default="last")
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_env_csv", type=Path, default=None)
    parser.add_argument("--out_env_step_csv", type=Path, default=None)
    parser.add_argument("--artifact_tag", type=str, default="d278_teacher_off_frozen_eval")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg

    if int(args.eval_steps) <= 0:
        raise ValueError("--eval_steps must be positive")
    if float(args.bc_teacher_blend) != 0.0:
        raise ValueError("teacher-off eval requires --bc_teacher_blend 0.0")
    if not (0.0 < float(args.exec_action_clip_abs) <= 1.0):
        raise ValueError("--exec_action_clip_abs must be in (0, 1]")
    if float(args.tap_stop_after_disp_m) < 0.0:
        raise ValueError("--tap_stop_after_disp_m must be non-negative")
    for name in (
        "min_mean_disp_along_m",
        "min_max_disp_along_m",
        "min_mean_disp_xy_m",
        "min_max_disp_xy_m",
        "min_disp_along_ge_1mm_rate",
        "min_disp_xy_ge_1mm_rate",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    for name in ("min_disp_along_ge_1mm_rate", "min_disp_xy_ge_1mm_rate"):
        if float(getattr(args, name)) > 1.0:
            raise ValueError(f"--{name} must be <= 1.0")

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    if args.action_scale is not None:
        env_cfg.action_scale = float(args.action_scale)
    if args.action_smoothing_alpha is not None:
        env_cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)
    if args.max_joint_delta_per_step_rad is not None:
        env_cfg.max_joint_delta_per_step_rad = float(args.max_joint_delta_per_step_rad)
    if args.contact_joint_delta_scale is not None:
        env_cfg.contact_joint_delta_scale = float(args.contact_joint_delta_scale)
    if args.fast_cube_joint_delta_scale is not None:
        env_cfg.fast_cube_joint_delta_scale = float(args.fast_cube_joint_delta_scale)
    if args.joint_target_lead_limit_rad is not None:
        env_cfg.joint_target_lead_limit_rad = float(args.joint_target_lead_limit_rad)
    if args.joint_delta_reference is not None:
        env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.fixed_push_dir_x = float(args.fixed_push_dir_x)
    env_cfg.fixed_push_dir_y = float(args.fixed_push_dir_y)
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.tap_stop_after_useful_seen = bool(args.tap_stop_after_useful_seen)
    env_cfg.tap_stop_after_disp_m = float(args.tap_stop_after_disp_m)
    env_cfg.tap_contact_slowdown_use_proxy = bool(args.tap_contact_slowdown_use_proxy)
    env_cfg.tap_success_terminate = bool(args.tap_success_terminate)
    env_cfg.tap_useful_terminate = bool(args.tap_useful_terminate)
    env_cfg.d256_reset_csv_path = str(args.d256_reset_csv_path)
    env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
    env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)
    env_cfg.bc_teacher_blend = float(args.bc_teacher_blend)
    env_cfg.bc_teacher_imitation_reward_scale = float(args.bc_teacher_imitation_reward_scale)
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)
    if args.bc_teacher_checkpoint_path is not None:
        env_cfg.bc_teacher_checkpoint_path = str(args.bc_teacher_checkpoint_path)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)
    ppo_cfg.num_steps_per_env = int(args.eval_steps)

    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[teacher-off-eval] scope=cube10cm_top_view_teacher_off_frozen_eval "
        f"env_id={env_id} training=NO bc_teacher_blend={env_cfg.bc_teacher_blend} "
        f"d256_reset_csv_path={env_cfg.d256_reset_csv_path}"
    )

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    if int(args.eval_steps) >= int(inner.max_episode_length) - 1:
        raise ValueError(
            f"--eval_steps {args.eval_steps} would hit env truncation/reset; "
            f"use <= {int(inner.max_episode_length) - 2}"
        )

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.checkpoint), load_optimizer=False, map_location=inner.device)
    if str(args.action_mode) == "ppo_stochastic":
        runner.train_mode()
    else:
        policy = runner.get_inference_policy(device=inner.device)

    env.reset()
    obs = env.get_observations()
    if str(args.d256_reset_warmup_mode) != "direct_reset":
        inner.episode_length_buf[:] = inner.max_episode_length
        with torch.inference_mode():
            if str(args.d256_reset_warmup_mode) == "force_step_policy":
                if str(args.action_mode) == "ppo_stochastic":
                    actions = runner.alg.act(obs)
                else:
                    actions = policy(obs)
            else:
                actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=inner.device)
            obs, _, _, _ = env.step(actions)
    print(f"[teacher-off-eval] reset_done mode={args.d256_reset_warmup_mode}", flush=True)

    inf = torch.full((inner.num_envs,), math.inf, device=inner.device)
    min_contact_vertical = inf.clone()
    tcp_threshold_seen = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    reward_finite_all = True
    obs_finite_all = _obs_is_finite(torch, obs)
    action_finite_all = True
    cap_trace: list[float] = []
    action_abs_mean_trace: list[float] = []
    action_abs_max_trace: list[float] = []
    action_abs_mean_per_env_sum = torch.zeros(inner.num_envs, device=inner.device)
    action_abs_max_per_env_trace = torch.zeros(inner.num_envs, device=inner.device)
    action_abs_mean_per_env_last = torch.zeros(inner.num_envs, device=inner.device)
    action_abs_max_per_env_last = torch.zeros(inner.num_envs, device=inner.device)
    cap_rate_max_per_env_trace = torch.zeros(inner.num_envs, device=inner.device)
    useful_action_hold_rate_trace: list[float] = []
    env_stop_after_useful_hold_rate_trace: list[float] = []
    env_stop_after_disp_hold_rate_trace: list[float] = []
    done_rate_trace: list[float] = []
    done_count_total = 0
    rsl_log_sums: dict[str, float] = {}
    rsl_log_counts: dict[str, int] = {}
    rsl_log_last: dict[str, float] = {}
    step_rows: list[dict[str, float | int]] = []
    env_step_rows: list[dict[str, float | int]] = []

    for step in range(int(args.eval_steps)):
        with torch.inference_mode():
            if str(args.action_mode) == "ppo_stochastic":
                actions = runner.alg.act(obs)
            else:
                actions = policy(obs)
            exec_clip = float(args.exec_action_clip_abs)
            actions = torch.clamp(actions, -exec_clip, exec_clip)
            hold_mask = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
            if bool(args.zero_actions_after_useful_seen):
                hold_mask = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
                actions = torch.where(hold_mask.unsqueeze(-1), torch.zeros_like(actions), actions)
            action_finite_all = action_finite_all and bool(torch.isfinite(actions).all().detach().cpu().item())
            obs, rewards, dones, extras = env.step(actions)
            if str(args.action_mode) == "ppo_stochastic":
                runner.alg.process_env_step(obs, rewards, dones, extras)
        useful_action_hold_rate = _tensor_mean(hold_mask.float())
        reward_finite_all = reward_finite_all and bool(torch.isfinite(rewards).all().detach().cpu().item())
        obs_finite_all = obs_finite_all and _obs_is_finite(torch, obs)
        done_rate = _tensor_mean(dones.float())
        done_count_total += int(dones.detach().long().sum().cpu().item())
        done_rate_trace.append(done_rate)

        extras_log = extras.get("log", {}) if isinstance(extras, dict) else {}
        scalar_extras: dict[str, float] = {}
        for key, value in extras_log.items():
            scalar = _scalar_log_value(torch, value)
            if scalar is None or not math.isfinite(scalar):
                continue
            scalar_extras[key] = scalar
            rsl_log_sums[key] = rsl_log_sums.get(key, 0.0) + scalar
            rsl_log_counts[key] = rsl_log_counts.get(key, 0) + 1
            rsl_log_last[key] = scalar

        terms = inner._tap_terms()
        contact = terms["tap_contact_proxy"]
        min_contact_vertical = torch.where(
            contact,
            torch.minimum(min_contact_vertical, terms["tap_contact_vertical_offset_m"]),
            min_contact_vertical,
        )
        tcp_threshold_seen |= terms["tcp_cube_dist"] < float(inner.cfg.contact_slowdown_tcp_dist_m)
        useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
        cap_mean = _tensor_mean(inner._last_joint_delta_cap_rate)
        action_abs_mean_per_env = torch.abs(actions).mean(dim=-1)
        action_abs_max_per_env = torch.abs(actions).max(dim=-1).values
        action_abs_mean = _tensor_mean(torch.abs(actions).mean(dim=-1))
        action_abs_max = _tensor_max(torch.abs(actions))
        env_hold = _tensor_mean(getattr(inner, "_last_tap_stop_after_useful_hold", torch.zeros(inner.num_envs, device=inner.device)))
        disp_hold = _tensor_mean(getattr(inner, "_last_tap_stop_after_disp_hold", torch.zeros(inner.num_envs, device=inner.device)))
        action_abs_mean_per_env_sum += action_abs_mean_per_env
        action_abs_max_per_env_trace = torch.maximum(action_abs_max_per_env_trace, action_abs_max_per_env)
        action_abs_mean_per_env_last = action_abs_mean_per_env
        action_abs_max_per_env_last = action_abs_max_per_env
        cap_rate_max_per_env_trace = torch.maximum(cap_rate_max_per_env_trace, inner._last_joint_delta_cap_rate)
        cap_trace.append(cap_mean)
        action_abs_mean_trace.append(action_abs_mean)
        action_abs_max_trace.append(action_abs_max)
        useful_action_hold_rate_trace.append(useful_action_hold_rate)
        env_stop_after_useful_hold_rate_trace.append(env_hold)
        env_stop_after_disp_hold_rate_trace.append(disp_hold)
        step_rows.append(
            {
                "step": step,
                "done_rate": done_rate,
                "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
                "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
                "tap_useful_seen_rate": _tensor_mean(useful_seen.float()),
                "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
                "tap_overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
                "tap_contact_proxy_rate": _tensor_mean(contact.float()),
                "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
                "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
                "tap_max_disp_along_ge_1mm_rate": _tensor_mean(
                    (inner._tap_max_disp_along >= 0.001).float()
                ),
                "tap_max_disp_xy_ge_1mm_rate": _tensor_mean((inner._tap_max_disp_xy >= 0.001).float()),
                "tap_contact_vertical_offset_mean_m": _tensor_mean(terms["tap_contact_vertical_offset_m"]),
                "tcp_cube_dist_mean_m": _tensor_mean(terms["tcp_cube_dist"]),
                "joint_delta_cap_rate_mean": cap_mean,
                "policy_action_abs_mean": action_abs_mean,
                "policy_action_abs_max": action_abs_max,
                "useful_action_hold_rate": useful_action_hold_rate,
                "env_stop_after_useful_hold_rate": env_hold,
                "env_stop_after_disp_hold_rate": disp_hold,
                "rsl_log_cube_tap_contact_seen_rate": scalar_extras.get("cube_tap_contact_seen_rate", math.nan),
                "rsl_log_cube_tap_reaction_seen_rate": scalar_extras.get("cube_tap_reaction_seen_rate", math.nan),
                "rsl_log_cube_tap_useful_seen_rate": scalar_extras.get("cube_tap_useful_seen_rate", math.nan),
                "rsl_log_cube_tap_success_rate": scalar_extras.get("cube_tap_success_rate", math.nan),
                "rsl_log_cube_tap_overshoot_seen_rate": scalar_extras.get("cube_tap_overshoot_seen_rate", math.nan),
                "rsl_log_cube_tap_max_disp_along_m": scalar_extras.get("cube_tap_max_disp_along_m", math.nan),
                "rsl_log_cube_tap_max_disp_xy_m": scalar_extras.get("cube_tap_max_disp_xy_m", math.nan),
                "bc_teacher_blend_mean": _tensor_mean(inner._last_bc_teacher_blend),
                "d256_reset_active_rate": _tensor_mean(inner._last_d256_reset_active),
            }
        )
        if args.out_env_step_csv is not None:
            stop_after_disp_hold = getattr(
                inner,
                "_last_tap_stop_after_disp_hold",
                torch.zeros(inner.num_envs, device=inner.device),
            )
            for env_i in range(int(inner.num_envs)):
                env_step_rows.append(
                    {
                        "step": step,
                        "env_id": env_i,
                        "d256_reset_episode_index": float(
                            inner._last_d256_reset_episode_index[env_i].detach().cpu().item()
                        ),
                        "tap_contact_seen": int(bool(inner._tap_contact_seen[env_i].detach().cpu().item())),
                        "tap_reaction_seen": int(bool(inner._tap_reaction_seen[env_i].detach().cpu().item())),
                        "tap_useful_seen": int(bool(useful_seen[env_i].detach().cpu().item())),
                        "tap_success": int(bool(inner._tap_success_flag[env_i].detach().cpu().item())),
                        "tap_overshoot_seen": int(bool(inner._tap_overshoot_seen[env_i].detach().cpu().item())),
                        "tap_contact_proxy_now": int(bool(contact[env_i].detach().cpu().item())),
                        "tap_reaction_now": int(bool(terms["tap_reaction_now"][env_i].detach().cpu().item())),
                        "tap_target_band_now": int(bool(terms["tap_target_band_now"][env_i].detach().cpu().item())),
                        "tap_overshoot_now": int(bool(terms["tap_overshoot_now"][env_i].detach().cpu().item())),
                        "tap_success_now": int(bool(terms["tap_success_now"][env_i].detach().cpu().item())),
                        "tap_disp_along_m": float(terms["disp_along"][env_i].detach().cpu().item()),
                        "tap_disp_xy_m": float(terms["disp_xy"][env_i].detach().cpu().item()),
                        "tap_max_disp_xy_m": float(inner._tap_max_disp_xy[env_i].detach().cpu().item()),
                        "tap_contact_face_gap_m": float(
                            terms["tap_contact_face_gap_m"][env_i].detach().cpu().item()
                        ),
                        "tap_contact_lateral_m": float(
                            terms["tap_contact_lateral_m"][env_i].detach().cpu().item()
                        ),
                        "tap_contact_vertical_offset_m": float(
                            terms["tap_contact_vertical_offset_m"][env_i].detach().cpu().item()
                        ),
                        "tcp_cube_dist_m": float(terms["tcp_cube_dist"][env_i].detach().cpu().item()),
                        "speed_mps": float(terms["speed"][env_i].detach().cpu().item()),
                        "tip_angle_deg": float(terms["tip_angle_deg"][env_i].detach().cpu().item()),
                        "action_abs_mean": float(action_abs_mean_per_env[env_i].detach().cpu().item()),
                        "action_abs_max": float(action_abs_max_per_env[env_i].detach().cpu().item()),
                        "joint_delta_cap_rate": float(
                            inner._last_joint_delta_cap_rate[env_i].detach().cpu().item()
                        ),
                        "stop_after_disp_hold": float(stop_after_disp_hold[env_i].detach().cpu().item()),
                    }
                )

    final_terms = inner._tap_terms()
    rsl_log_means = {
        key: rsl_log_sums[key] / float(rsl_log_counts[key])
        for key in sorted(rsl_log_sums)
        if rsl_log_counts.get(key, 0) > 0
    }
    def _rsl_mean(key: str) -> float | None:
        return rsl_log_means.get(key)

    useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
    issues: list[str] = []
    d256_active = _tensor_mean(inner._last_d256_reset_active)
    bc_blend_last = _tensor_mean(inner._last_bc_teacher_blend)
    useful_rate = _tensor_mean(useful_seen.float())
    overshoot_rate = _tensor_mean(inner._tap_overshoot_seen.float())
    cap_rate_last = _tensor_mean(inner._last_joint_delta_cap_rate)
    cap_rate_max = max(cap_trace) if cap_trace else 0.0
    tap_max_disp_along_mean = _tensor_mean(inner._tap_max_disp_along)
    tap_max_disp_along_max = _tensor_max(inner._tap_max_disp_along)
    tap_max_disp_xy_mean = _tensor_mean(inner._tap_max_disp_xy)
    tap_max_disp_xy_max = _tensor_max(inner._tap_max_disp_xy)
    tap_max_disp_along_ge_1mm_rate = _tensor_mean((inner._tap_max_disp_along >= 0.001).float())
    tap_max_disp_xy_ge_1mm_rate = _tensor_mean((inner._tap_max_disp_xy >= 0.001).float())
    tap_max_disp_along_ge_3mm_rate = _tensor_mean((inner._tap_max_disp_along >= 0.003).float())
    tap_max_disp_xy_ge_3mm_rate = _tensor_mean((inner._tap_max_disp_xy >= 0.003).float())
    last_vertical_max = _tensor_max(final_terms["tap_contact_vertical_offset_m"])
    min_contact_vertical_max = _finite_max(torch, min_contact_vertical)
    if str(args.vertical_gate_mode) == "min_contact":
        vertical_gate_value = min_contact_vertical_max if min_contact_vertical_max is not None else math.inf
    else:
        vertical_gate_value = last_vertical_max

    if d256_active < 0.99:
        issues.append(f"D256 reset hook inactive: active_rate={d256_active}")
    if abs(bc_blend_last) > 1.0e-6:
        issues.append(f"BC teacher blend is nonzero in teacher-off eval: {bc_blend_last}")
    if not reward_finite_all or not obs_finite_all or not action_finite_all:
        issues.append("non-finite reward/obs/action observed")
    if useful_rate < float(args.min_useful_seen_rate):
        issues.append(f"tap useful seen rate below threshold: {useful_rate}")
    if overshoot_rate > float(args.max_overshoot_seen_rate):
        issues.append(f"tap overshoot seen rate too high: {overshoot_rate}")
    if cap_rate_max > float(args.max_joint_delta_cap_rate):
        issues.append(f"joint delta cap rate too high: max_trace={cap_rate_max}")
    if vertical_gate_value > float(args.max_vertical_offset_m):
        issues.append(
            f"tap contact vertical offset too high: mode={args.vertical_gate_mode} max={vertical_gate_value}"
        )
    if tap_max_disp_along_mean < float(args.min_mean_disp_along_m):
        issues.append(
            f"tap mean max along displacement below threshold: {tap_max_disp_along_mean}"
        )
    if tap_max_disp_along_max < float(args.min_max_disp_along_m):
        issues.append(
            f"tap max along displacement below threshold: {tap_max_disp_along_max}"
        )
    if tap_max_disp_xy_mean < float(args.min_mean_disp_xy_m):
        issues.append(f"tap mean max xy displacement below threshold: {tap_max_disp_xy_mean}")
    if tap_max_disp_xy_max < float(args.min_max_disp_xy_m):
        issues.append(f"tap max xy displacement below threshold: {tap_max_disp_xy_max}")
    if tap_max_disp_along_ge_1mm_rate < float(args.min_disp_along_ge_1mm_rate):
        issues.append(
            f"tap along displacement >=1mm rate below threshold: {tap_max_disp_along_ge_1mm_rate}"
        )
    if tap_max_disp_xy_ge_1mm_rate < float(args.min_disp_xy_ge_1mm_rate):
        issues.append(f"tap XY displacement >=1mm rate below threshold: {tap_max_disp_xy_ge_1mm_rate}")
    if bool(args.require_rsl_log_gate):
        rsl_useful = _rsl_mean("cube_tap_useful_seen_rate")
        rsl_overshoot = _rsl_mean("cube_tap_overshoot_seen_rate")
        if rsl_useful is None:
            issues.append("RSL-like useful TensorBoard log is missing")
        elif rsl_useful < float(args.min_useful_seen_rate):
            issues.append(f"RSL-like useful seen rate below threshold: {rsl_useful}")
        if rsl_overshoot is None:
            issues.append("RSL-like overshoot TensorBoard log is missing")
        elif rsl_overshoot > float(args.max_overshoot_seen_rate):
            issues.append(f"RSL-like overshoot seen rate too high: {rsl_overshoot}")

    if issues:
        verdict = "TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM"
    else:
        verdict = "TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE"

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "checkpoint": _rel(args.checkpoint),
        "env_id": env_id,
        "num_envs": int(args.num_envs),
        "eval_steps": int(args.eval_steps),
        "episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length": int(inner.max_episode_length),
        "action_scale": float(env_cfg.action_scale),
        "action_smoothing_alpha": float(env_cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(env_cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(env_cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(env_cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(env_cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(env_cfg.joint_delta_reference),
        "seed": int(args.seed),
        "d256_reset_csv_path": _rel(args.d256_reset_csv_path),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "d256_reset_warmup_mode": str(args.d256_reset_warmup_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "bc_teacher_checkpoint_path": _rel(args.bc_teacher_checkpoint_path) if args.bc_teacher_checkpoint_path else "",
        "bc_teacher_blend": float(args.bc_teacher_blend),
        "bc_teacher_imitation_reward_scale": float(args.bc_teacher_imitation_reward_scale),
        "bc_teacher_blend_mean_last": bc_blend_last,
        "action_mode": str(args.action_mode),
        "vertical_gate_mode": str(args.vertical_gate_mode),
        "vertical_gate_value_m": vertical_gate_value,
        "tap_stop_after_useful_seen": bool(args.tap_stop_after_useful_seen),
        "tap_success_terminate": bool(args.tap_success_terminate),
        "tap_stop_after_disp_m": float(args.tap_stop_after_disp_m),
        "tap_contact_slowdown_use_proxy": bool(args.tap_contact_slowdown_use_proxy),
        "tap_useful_terminate": bool(args.tap_useful_terminate),
        "done_rate_mean_trace": sum(done_rate_trace) / len(done_rate_trace) if done_rate_trace else 0.0,
        "done_rate_max_trace": max(done_rate_trace) if done_rate_trace else 0.0,
        "done_count_total": int(done_count_total),
        "rsl_log_means": rsl_log_means,
        "rsl_log_last": rsl_log_last,
        "require_rsl_log_gate": bool(args.require_rsl_log_gate),
        "rsl_log_cube_tap_contact_seen_rate_mean": _rsl_mean("cube_tap_contact_seen_rate"),
        "rsl_log_cube_tap_useful_seen_rate_mean": _rsl_mean("cube_tap_useful_seen_rate"),
        "rsl_log_cube_tap_success_rate_mean": _rsl_mean("cube_tap_success_rate"),
        "rsl_log_cube_tap_overshoot_seen_rate_mean": _rsl_mean("cube_tap_overshoot_seen_rate"),
        "rsl_log_cube_tap_max_disp_along_m_mean": _rsl_mean("cube_tap_max_disp_along_m"),
        "rsl_log_cube_tap_max_disp_xy_m_mean": _rsl_mean("cube_tap_max_disp_xy_m"),
        "env_stop_after_useful_hold_rate_last": (
            env_stop_after_useful_hold_rate_trace[-1] if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "env_stop_after_useful_hold_rate_max_trace": (
            max(env_stop_after_useful_hold_rate_trace) if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "env_stop_after_disp_hold_rate_last": (
            env_stop_after_disp_hold_rate_trace[-1] if env_stop_after_disp_hold_rate_trace else 0.0
        ),
        "env_stop_after_disp_hold_rate_max_trace": (
            max(env_stop_after_disp_hold_rate_trace) if env_stop_after_disp_hold_rate_trace else 0.0
        ),
        "zero_actions_after_useful_seen": bool(args.zero_actions_after_useful_seen),
        "exec_action_clip_abs": float(args.exec_action_clip_abs),
        "useful_action_hold_rate_last": useful_action_hold_rate_trace[-1] if useful_action_hold_rate_trace else 0.0,
        "useful_action_hold_rate_max_trace": max(useful_action_hold_rate_trace) if useful_action_hold_rate_trace else 0.0,
        "d256_reset_active_rate": d256_active,
        "d256_reset_episode_index_mean": _tensor_mean(inner._last_d256_reset_episode_index),
        "d256_reset_episode_index_min": _tensor_min(inner._last_d256_reset_episode_index),
        "d256_reset_episode_index_max": _tensor_max(inner._last_d256_reset_episode_index),
        "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
        "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
        "tap_useful_seen_rate": useful_rate,
        "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
        "tap_overshoot_seen_rate": overshoot_rate,
        "tap_contact_proxy_rate_last": _tensor_mean(final_terms["tap_contact_proxy"].float()),
        "tap_max_disp_along_mean_m": tap_max_disp_along_mean,
        "tap_max_disp_along_max_m": tap_max_disp_along_max,
        "tap_max_disp_xy_mean_m": tap_max_disp_xy_mean,
        "tap_max_disp_xy_max_m": tap_max_disp_xy_max,
        "tap_max_disp_along_ge_1mm_rate": tap_max_disp_along_ge_1mm_rate,
        "tap_max_disp_xy_ge_1mm_rate": tap_max_disp_xy_ge_1mm_rate,
        "tap_max_disp_along_ge_3mm_rate": tap_max_disp_along_ge_3mm_rate,
        "tap_max_disp_xy_ge_3mm_rate": tap_max_disp_xy_ge_3mm_rate,
        "min_mean_disp_along_m": float(args.min_mean_disp_along_m),
        "min_max_disp_along_m": float(args.min_max_disp_along_m),
        "min_mean_disp_xy_m": float(args.min_mean_disp_xy_m),
        "min_max_disp_xy_m": float(args.min_max_disp_xy_m),
        "min_disp_along_ge_1mm_rate": float(args.min_disp_along_ge_1mm_rate),
        "min_disp_xy_ge_1mm_rate": float(args.min_disp_xy_ge_1mm_rate),
        "min_contact_vertical_offset_mean_m": _finite_mean(torch, min_contact_vertical),
        "min_contact_vertical_offset_min_m": _finite_min(torch, min_contact_vertical),
        "min_contact_vertical_offset_max_m": _finite_max(torch, min_contact_vertical),
        "last_contact_vertical_offset_mean_m": _tensor_mean(final_terms["tap_contact_vertical_offset_m"]),
        "last_contact_vertical_offset_max_m": last_vertical_max,
        "last_contact_face_gap_mean_m": _tensor_mean(final_terms["tap_contact_face_gap_m"]),
        "last_contact_lateral_mean_m": _tensor_mean(final_terms["tap_contact_lateral_m"]),
        "tcp_threshold_contact_seen_rate": _tensor_mean(tcp_threshold_seen.float()),
        "joint_delta_cap_rate_mean_last": cap_rate_last,
        "joint_delta_cap_rate_max_trace": cap_rate_max,
        "policy_action_abs_mean_trace_mean": sum(action_abs_mean_trace) / len(action_abs_mean_trace),
        "policy_action_abs_max_trace_max": max(action_abs_max_trace) if action_abs_max_trace else 0.0,
        "reward_finite_all": reward_finite_all,
        "obs_finite_all": obs_finite_all,
        "action_finite_all": action_finite_all,
        "out_csv": _rel(args.out_csv),
        "out_env_csv": _rel(args.out_env_csv) if args.out_env_csv is not None else "",
        "out_env_step_csv": _rel(args.out_env_step_csv) if args.out_env_step_csv is not None else "",
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        fieldnames = list(step_rows[0].keys()) if step_rows else ["step"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(args.out_md, summary)

    if args.out_env_csv is not None:
        args.out_env_csv.parent.mkdir(parents=True, exist_ok=True)
        per_env_fields = [
            "env_id",
            "d256_reset_episode_index",
            "d256_reset_active",
            "tap_contact_seen",
            "tap_reaction_seen",
            "tap_useful_seen",
            "tap_success",
            "tap_overshoot_seen",
            "tap_contact_proxy_last",
            "tap_reaction_now_last",
            "tap_target_band_now_last",
            "tap_overshoot_now_last",
            "tap_success_now_last",
            "tap_disp_along_last_m",
            "tap_disp_xy_last_m",
            "tap_max_disp_along_m",
            "tap_max_disp_xy_m",
            "tap_max_disp_along_ge_1mm",
            "tap_max_disp_xy_ge_1mm",
            "tap_max_disp_along_ge_3mm",
            "tap_max_disp_xy_ge_3mm",
            "min_contact_vertical_offset_m",
            "last_contact_face_gap_m",
            "last_contact_lateral_m",
            "last_contact_vertical_offset_m",
            "last_tcp_cube_dist_m",
            "last_speed_mps",
            "last_tip_angle_deg",
            "last_target_disp_error_m",
            "last_target_excess_m",
            "action_abs_mean_trace",
            "action_abs_max_trace",
            "action_abs_mean_last",
            "action_abs_max_last",
            "joint_delta_cap_rate_max_trace",
            "stop_after_disp_hold_last",
            "failure_reason",
        ]
        per_env_rows = []
        action_abs_mean_per_env_trace = action_abs_mean_per_env_sum / float(args.eval_steps)
        for env_i in range(int(inner.num_envs)):
            contact_seen_i = bool(inner._tap_contact_seen[env_i].detach().cpu().item())
            reaction_seen_i = bool(inner._tap_reaction_seen[env_i].detach().cpu().item())
            overshoot_seen_i = bool(inner._tap_overshoot_seen[env_i].detach().cpu().item())
            useful_seen_i = bool(useful_seen[env_i].detach().cpu().item())
            success_i = bool(inner._tap_success_flag[env_i].detach().cpu().item())
            contact_proxy_last_i = bool(final_terms["tap_contact_proxy"][env_i].detach().cpu().item())
            reaction_now_last_i = bool(final_terms["tap_reaction_now"][env_i].detach().cpu().item())
            target_band_now_last_i = bool(final_terms["tap_target_band_now"][env_i].detach().cpu().item())
            overshoot_now_last_i = bool(final_terms["tap_overshoot_now"][env_i].detach().cpu().item())
            success_now_last_i = bool(final_terms["tap_success_now"][env_i].detach().cpu().item())
            if success_i:
                failure_reason = "success"
            elif overshoot_seen_i:
                failure_reason = "overshoot_seen"
            elif not contact_seen_i:
                failure_reason = "no_contact_seen"
            elif not reaction_seen_i:
                failure_reason = "contact_without_reaction"
            elif not target_band_now_last_i:
                failure_reason = "reaction_outside_target_band"
            elif not contact_proxy_last_i:
                failure_reason = "lost_contact_at_final"
            elif not useful_seen_i:
                failure_reason = "not_useful_unknown"
            else:
                failure_reason = "useful_not_success"
            per_env_rows.append(
                {
                    "env_id": env_i,
                    "d256_reset_episode_index": float(inner._last_d256_reset_episode_index[env_i].detach().cpu().item()),
                    "d256_reset_active": float(inner._last_d256_reset_active[env_i].detach().cpu().item()),
                    "tap_contact_seen": int(contact_seen_i),
                    "tap_reaction_seen": int(reaction_seen_i),
                    "tap_useful_seen": int(useful_seen_i),
                    "tap_success": int(success_i),
                    "tap_overshoot_seen": int(overshoot_seen_i),
                    "tap_contact_proxy_last": int(contact_proxy_last_i),
                    "tap_reaction_now_last": int(reaction_now_last_i),
                    "tap_target_band_now_last": int(target_band_now_last_i),
                    "tap_overshoot_now_last": int(overshoot_now_last_i),
                    "tap_success_now_last": int(success_now_last_i),
                    "tap_disp_along_last_m": float(final_terms["disp_along"][env_i].detach().cpu().item()),
                    "tap_disp_xy_last_m": float(final_terms["disp_xy"][env_i].detach().cpu().item()),
                    "tap_max_disp_along_m": float(inner._tap_max_disp_along[env_i].detach().cpu().item()),
                    "tap_max_disp_xy_m": float(inner._tap_max_disp_xy[env_i].detach().cpu().item()),
                    "tap_max_disp_along_ge_1mm": int(bool((inner._tap_max_disp_along[env_i] >= 0.001).detach().cpu().item())),
                    "tap_max_disp_xy_ge_1mm": int(bool((inner._tap_max_disp_xy[env_i] >= 0.001).detach().cpu().item())),
                    "tap_max_disp_along_ge_3mm": int(bool((inner._tap_max_disp_along[env_i] >= 0.003).detach().cpu().item())),
                    "tap_max_disp_xy_ge_3mm": int(bool((inner._tap_max_disp_xy[env_i] >= 0.003).detach().cpu().item())),
                    "min_contact_vertical_offset_m": float(min_contact_vertical[env_i].detach().cpu().item()),
                    "last_contact_face_gap_m": float(final_terms["tap_contact_face_gap_m"][env_i].detach().cpu().item()),
                    "last_contact_lateral_m": float(final_terms["tap_contact_lateral_m"][env_i].detach().cpu().item()),
                    "last_contact_vertical_offset_m": float(final_terms["tap_contact_vertical_offset_m"][env_i].detach().cpu().item()),
                    "last_tcp_cube_dist_m": float(final_terms["tcp_cube_dist"][env_i].detach().cpu().item()),
                    "last_speed_mps": float(final_terms["speed"][env_i].detach().cpu().item()),
                    "last_tip_angle_deg": float(final_terms["tip_angle_deg"][env_i].detach().cpu().item()),
                    "last_target_disp_error_m": float(final_terms["tap_target_disp_error_m"][env_i].detach().cpu().item()),
                    "last_target_excess_m": float(final_terms["tap_target_excess_m"][env_i].detach().cpu().item()),
                    "action_abs_mean_trace": float(action_abs_mean_per_env_trace[env_i].detach().cpu().item()),
                    "action_abs_max_trace": float(action_abs_max_per_env_trace[env_i].detach().cpu().item()),
                    "action_abs_mean_last": float(action_abs_mean_per_env_last[env_i].detach().cpu().item()),
                    "action_abs_max_last": float(action_abs_max_per_env_last[env_i].detach().cpu().item()),
                    "joint_delta_cap_rate_max_trace": float(cap_rate_max_per_env_trace[env_i].detach().cpu().item()),
                    "stop_after_disp_hold_last": float(
                        getattr(
                            inner,
                            "_last_tap_stop_after_disp_hold",
                            torch.zeros(inner.num_envs, device=inner.device),
                        )[env_i]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "failure_reason": failure_reason,
                }
            )
        with args.out_env_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_env_fields)
            writer.writeheader()
            writer.writerows(per_env_rows)

    if args.out_env_step_csv is not None:
        args.out_env_step_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(env_step_rows[0].keys()) if env_step_rows else ["step", "env_id"]
        with args.out_env_step_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(env_step_rows)

    print(
        "[teacher-off-eval] SUMMARY "
        f"verdict={verdict} useful={useful_rate:.6f} overshoot={overshoot_rate:.6f} "
        f"d256_reset_active={d256_active:.6f} bc_blend_last={bc_blend_last:.6f} "
        f"json={args.out_json}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
