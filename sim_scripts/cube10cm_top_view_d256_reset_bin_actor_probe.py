"""Probe D256 reset episode bins with a frozen PPO actor.

This is a diagnostic, not PPO training.  It reuses one Isaac Lab app, changes
the D256 reset episode filter per bin, and records action/cap/contact metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
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
    "actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/"
    "cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt"
)
DEFAULT_OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "d256_reset_bin_actor_probe_d286/tap10cm"
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _tensor_mean(x: Any) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_min(x: Any) -> float:
    return float(x.detach().float().min().cpu().item())


def _tensor_max(x: Any) -> float:
    return float(x.detach().float().max().cpu().item())


def _obs_is_finite(torch: Any, obs: Any) -> bool:
    if isinstance(obs, dict):
        return all(_obs_is_finite(torch, value) for value in obs.values())
    if "TensorDict" in type(obs).__name__ and hasattr(obs, "values"):
        return all(_obs_is_finite(torch, value) for value in obs.values())
    return bool(torch.isfinite(obs).all().detach().cpu().item())


def _log_scalar(log: dict[str, Any], key: str) -> float | None:
    value = log.get(key)
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return float(value.detach().float().cpu().item())
        return float(value)
    except Exception:
        return None


def _max_with_log(trace: list[float], log_trace: list[float | None]) -> float:
    values = list(trace)
    values.extend(value for value in log_trace if value is not None)
    return max(values) if values else 0.0


def _frame0_rows(csv_path: Path, frame_index: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["frame_index_t"])) == int(frame_index):
                rows.append(row)
    if not rows:
        raise ValueError(f"no frame_index_t={frame_index} rows in {csv_path}")
    return rows


def _episode_bins(rows: list[dict[str, str]], bin_count: int) -> list[tuple[int, int]]:
    episodes = sorted({int(float(row["episode_index"])) for row in rows})
    if not episodes:
        raise ValueError("empty episode list")
    if int(bin_count) <= 0:
        raise ValueError("--bin_count must be positive")
    bins: list[tuple[int, int]] = []
    n = len(episodes)
    for idx in range(int(bin_count)):
        start_i = int(round(idx * n / int(bin_count)))
        end_i = int(round((idx + 1) * n / int(bin_count))) - 1
        start_i = max(0, min(start_i, n - 1))
        end_i = max(start_i, min(end_i, n - 1))
        bins.append((episodes[start_i], episodes[end_i]))
    return bins


def _parse_episode_ranges(values: list[str] | None) -> list[tuple[int, int]] | None:
    if not values:
        return None
    ranges: list[tuple[int, int]] = []
    for value in values:
        text = value.strip()
        if not text:
            continue
        if ":" in text:
            left, right = text.split(":", 1)
            episode_min = int(left)
            episode_max = int(right)
        else:
            episode_min = int(text)
            episode_max = episode_min
        if episode_min > episode_max:
            raise ValueError(f"invalid --episode_range {value!r}: min > max")
        ranges.append((episode_min, episode_max))
    if not ranges:
        raise ValueError("--episode_range was provided but no valid ranges were parsed")
    return ranges


def _static_bin_stats(rows: list[dict[str, str]], episode_min: int, episode_max: int) -> dict[str, float | int]:
    subset = [
        row for row in rows
        if int(float(row["episode_index"])) >= episode_min and int(float(row["episode_index"])) <= episode_max
    ]
    if not subset:
        raise ValueError(f"empty static bin {episode_min}-{episode_max}")

    def values(col: str) -> list[float]:
        return [float(row[col]) for row in subset]

    def mean(col: str) -> float:
        vals = values(col)
        return sum(vals) / len(vals)

    def max_abs(prefix: str, count: int) -> float:
        return max(max(abs(float(row[f"{prefix}_{idx}_rad"])) for idx in range(count)) for row in subset)

    return {
        "d256_frame0_rows": len(subset),
        "d256_cube_x_mean_m": mean("cube_local_x_m"),
        "d256_cube_y_mean_m": mean("cube_local_y_m"),
        "d256_cube_z_mean_m": mean("cube_local_z_m"),
        "d256_tcp_to_cube_x_mean_m": mean("tcp_to_cube_x_m"),
        "d256_tcp_to_cube_y_mean_m": mean("tcp_to_cube_y_m"),
        "d256_tcp_to_cube_z_mean_m": mean("tcp_to_cube_z_m"),
        "d256_target_to_cube_x_mean_m": mean("target_to_cube_x_m"),
        "d256_target_to_cube_y_mean_m": mean("target_to_cube_y_m"),
        "d256_arm_joint_abs_max_rad": max_abs("arm_joint", 5),
        "d256_joint_delta_abs_max_rad": max_abs("joint_delta", 5),
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D256 Reset Bin Actor Probe",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- diagnostic class: `{summary['diagnostic_class']}`",
        f"- actor checkpoint: `{summary['actor_checkpoint']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- exec source: `{summary['exec_source']}`",
        f"- exec teacher blend: `{summary['exec_teacher_blend']}`",
        f"- exec action clip abs: `{summary['exec_action_clip_abs']}`",
        f"- warmup action source: `{summary['warmup_action_source']}`",
        f"- joint delta reference: `{summary['joint_delta_reference']}`",
        f"- bc teacher delta scale: `{summary['bc_teacher_policy_delta_scale']}`",
        f"- tap stop after disp m: `{summary['tap_stop_after_disp_m']}`",
        f"- tap contact slowdown use proxy: `{summary['tap_contact_slowdown_use_proxy']}`",
        f"- bins/envs/steps: `{summary['bin_count']}` / `{summary['num_envs']}` / `{summary['eval_steps']}`",
        f"- action noise std: `{summary['action_noise_std']}`",
        f"- cap action threshold abs: `{summary['cap_action_threshold_abs']}`",
        f"- safe bins: `{summary['safe_bins']}`",
        "",
        "## Bin Rows",
        "",
    ]
    for row in summary["bins"]:
        lines.append(
            "- "
            f"{row['episode_min']}-{row['episode_max']}: "
            f"cap max `{row['joint_delta_cap_rate_max_trace']}`, "
            f"action max `{row['action_abs_max_trace_max']}`, "
            f"useful max `{row['tap_useful_seen_rate_max_trace']}`, "
            f"contact max `{row['tap_contact_seen_rate_max_trace']}`, "
            f"overshoot max `{row['tap_overshoot_seen_rate_max_trace']}`, "
            f"mse `{row['actor_teacher_mse_mean_trace_mean']}`, "
            f"cube_y `{row['d256_cube_y_mean_m']}`"
        )
    lines.extend(["", "## Issues", ""])
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.",
            "A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.",
            "Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT)
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--d256_reset_csv_path", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--bin_count", type=int, default=5)
    parser.add_argument(
        "--episode_range",
        action="append",
        default=None,
        help="Explicit D256 episode range to probe, as EP or MIN:MAX. Repeat for multiple bins.",
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=72)
    parser.add_argument("--seed", type=int, default=28601)
    parser.add_argument("--action_noise_std", type=float, default=0.02)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--action_smoothing_alpha", type=float, default=None)
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--contact_joint_delta_scale", type=float, default=None)
    parser.add_argument("--fast_cube_joint_delta_scale", type=float, default=None)
    parser.add_argument("--joint_target_lead_limit_rad", type=float, default=None)
    parser.add_argument("--joint_delta_reference", choices=("target", "joint_pos"), default="joint_pos")
    parser.add_argument("--exec_source", choices=("actor", "teacher", "blend"), default="actor")
    parser.add_argument("--exec_teacher_blend", type=float, default=0.5)
    parser.add_argument("--exec_action_clip_abs", type=float, default=1.0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument("--reset_warmup_mode", choices=("direct_reset", "force_step_zero"), default="direct_reset")
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--tap_stop_after_useful_seen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tap_stop_after_disp_m", type=float, default=0.0)
    parser.add_argument("--tap_contact_slowdown_use_proxy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tap_useful_terminate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tap_overshoot_terminate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument(
        "--bc_teacher_phase_timing",
        choices=("episode_scaled", "direct_steps", "linear_episode", "linear_steps"),
        default="direct_steps",
    )
    parser.add_argument("--bc_teacher_linear_phase_steps", type=int, default=579)
    parser.add_argument("--bc_teacher_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_lowx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--bc_teacher_highx_policy_delta_scale", type=float, default=None)
    parser.add_argument("--max_cap_rate_for_safe_bin", type=float, default=0.25)
    parser.add_argument("--max_overshoot_rate_for_safe_bin", type=float, default=0.05)
    parser.add_argument("--min_useful_rate_for_safe_bin", type=float, default=0.01)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact_tag", type=str, default="d286_d256_reset_bin_actor_probe")
    args = parser.parse_args()

    if int(args.eval_steps) <= 0:
        raise ValueError("--eval_steps must be positive")
    if float(args.action_noise_std) < 0.0:
        raise ValueError("--action_noise_std must be non-negative")
    if not (0.0 <= float(args.exec_teacher_blend) <= 1.0):
        raise ValueError("--exec_teacher_blend must be in [0, 1]")
    if not (0.0 < float(args.exec_action_clip_abs) <= 1.0):
        raise ValueError("--exec_action_clip_abs must be in (0, 1]")
    if not args.actor_checkpoint.exists():
        raise FileNotFoundError(args.actor_checkpoint)
    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(args.teacher_checkpoint)
    if not args.d256_reset_csv_path.exists():
        raise FileNotFoundError(args.d256_reset_csv_path)

    rows = _frame0_rows(args.d256_reset_csv_path, int(args.d256_reset_frame_index))
    explicit_bins = _parse_episode_ranges(args.episode_range)
    bins = explicit_bins if explicit_bins is not None else _episode_bins(rows, int(args.bin_count))

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

    torch.manual_seed(int(args.seed))

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
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
    env_cfg.joint_delta_reference = str(args.joint_delta_reference)
    env_cfg.d256_reset_csv_path = str(args.d256_reset_csv_path)
    env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
    env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.tap_stop_after_useful_seen = bool(args.tap_stop_after_useful_seen)
    env_cfg.tap_stop_after_disp_m = float(args.tap_stop_after_disp_m)
    env_cfg.tap_contact_slowdown_use_proxy = bool(args.tap_contact_slowdown_use_proxy)
    env_cfg.tap_useful_terminate = bool(args.tap_useful_terminate)
    env_cfg.tap_overshoot_terminate = bool(args.tap_overshoot_terminate)
    env_cfg.bc_teacher_checkpoint_path = str(args.teacher_checkpoint)
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)
    env_cfg.bc_teacher_linear_phase_steps = int(args.bc_teacher_linear_phase_steps)
    if args.bc_teacher_policy_delta_scale is not None:
        env_cfg.bc_teacher_policy_delta_scale = float(args.bc_teacher_policy_delta_scale)
    if args.bc_teacher_lowx_policy_delta_scale is not None:
        env_cfg.bc_teacher_lowx_policy_delta_scale = float(args.bc_teacher_lowx_policy_delta_scale)
    if args.bc_teacher_highx_policy_delta_scale is not None:
        env_cfg.bc_teacher_highx_policy_delta_scale = float(args.bc_teacher_highx_policy_delta_scale)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)

    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[d256-bin-probe] scope=cube10cm_top_view_d286_reset_bin_actor_probe "
        f"env_id={env_id} training=NO bins={len(bins)} actor={args.actor_checkpoint}",
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

    cap_action_threshold_abs = float(inner.cfg.max_joint_delta_per_step_rad) / max(float(inner.cfg.action_scale), 1.0e-9)
    bin_rows: list[dict[str, Any]] = []
    reward_finite_all = True
    obs_finite_all = True
    action_finite_all = True

    for bin_idx, (episode_min, episode_max) in enumerate(bins):
        inner.cfg.d256_reset_episode_min = int(episode_min)
        inner.cfg.d256_reset_episode_max = int(episode_max)
        if hasattr(inner, "_d256_reset_table"):
            delattr(inner, "_d256_reset_table")

        if str(args.reset_warmup_mode) == "direct_reset":
            env.reset()
            obs = env.get_observations()
        else:
            inner.episode_length_buf[:] = inner.max_episode_length
            obs = env.get_observations()
            with torch.inference_mode():
                warmup_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=inner.device)
            obs, _, _, _ = env.step(warmup_actions.detach().clone())

        cap_trace: list[float] = []
        action_abs_mean_trace: list[float] = []
        action_abs_max_trace: list[float] = []
        policy_abs_mean_trace: list[float] = []
        policy_abs_max_trace: list[float] = []
        teacher_abs_mean_trace: list[float] = []
        actor_teacher_mse_trace: list[float] = []
        actor_teacher_cosine_trace: list[float] = []
        reset_episode_mean_trace: list[float] = []
        contact_seen_trace: list[float] = []
        useful_seen_trace: list[float] = []
        overshoot_seen_trace: list[float] = []
        contact_proxy_trace: list[float] = []
        max_disp_along_trace: list[float] = []
        max_disp_xy_trace: list[float] = []
        log_contact_seen_trace: list[float | None] = []
        log_reaction_seen_trace: list[float | None] = []
        log_useful_seen_trace: list[float | None] = []
        log_overshoot_seen_trace: list[float | None] = []
        log_contact_proxy_trace: list[float | None] = []
        log_max_disp_along_trace: list[float | None] = []
        log_max_disp_xy_trace: list[float | None] = []

        for _step in range(int(args.eval_steps)):
            with torch.no_grad():
                policy_actions = policy(obs)
                teacher_actions = inner._bc_teacher_actions()
                if args.exec_source == "actor":
                    exec_actions = policy_actions
                elif args.exec_source == "teacher":
                    exec_actions = teacher_actions
                else:
                    blend = float(args.exec_teacher_blend)
                    exec_actions = (1.0 - blend) * policy_actions + blend * teacher_actions
                if float(args.action_noise_std) > 0.0:
                    exec_actions = exec_actions + torch.randn_like(exec_actions) * float(args.action_noise_std)
                exec_clip = float(args.exec_action_clip_abs)
                exec_actions = torch.clamp(exec_actions, -exec_clip, exec_clip)
                diff = torch.clamp(policy_actions, -1.0, 1.0) - torch.clamp(teacher_actions, -1.0, 1.0)
                mse = torch.mean(diff * diff, dim=-1)
                cosine = F.cosine_similarity(
                    torch.clamp(policy_actions, -1.0, 1.0),
                    torch.clamp(teacher_actions, -1.0, 1.0),
                    dim=-1,
                    eps=1.0e-6,
                )
            obs, rewards, _, _ = env.step(exec_actions.detach().clone())

            step_log = getattr(inner, "extras", {}).get("log", {})
            reward_finite_all = reward_finite_all and bool(torch.isfinite(rewards).all().detach().cpu().item())
            obs_finite_all = obs_finite_all and _obs_is_finite(torch, obs)
            action_finite_all = action_finite_all and bool(torch.isfinite(exec_actions).all().detach().cpu().item())
            step_terms = inner._tap_terms()
            useful_seen_step = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen

            cap_trace.append(_tensor_mean(inner._last_joint_delta_cap_rate))
            action_abs_mean_trace.append(_tensor_mean(torch.abs(exec_actions).mean(dim=-1)))
            action_abs_max_trace.append(_tensor_max(torch.abs(exec_actions)))
            policy_abs_mean_trace.append(_tensor_mean(torch.abs(policy_actions).mean(dim=-1)))
            policy_abs_max_trace.append(_tensor_max(torch.abs(policy_actions)))
            teacher_abs_mean_trace.append(_tensor_mean(torch.abs(teacher_actions).mean(dim=-1)))
            actor_teacher_mse_trace.append(_tensor_mean(mse))
            actor_teacher_cosine_trace.append(_tensor_mean(cosine))
            reset_episode_mean_trace.append(_tensor_mean(inner._last_d256_reset_episode_index))
            contact_seen_trace.append(_tensor_mean(inner._tap_contact_seen.float()))
            useful_seen_trace.append(_tensor_mean(useful_seen_step.float()))
            overshoot_seen_trace.append(_tensor_mean(inner._tap_overshoot_seen.float()))
            contact_proxy_trace.append(_tensor_mean(step_terms["tap_contact_proxy"].float()))
            max_disp_along_trace.append(_tensor_mean(inner._tap_max_disp_along))
            max_disp_xy_trace.append(_tensor_mean(inner._tap_max_disp_xy))
            log_contact_seen_trace.append(_log_scalar(step_log, "cube_tap_contact_seen_rate"))
            log_reaction_seen_trace.append(_log_scalar(step_log, "cube_tap_reaction_seen_rate"))
            log_useful_seen_trace.append(_log_scalar(step_log, "cube_tap_useful_seen_rate"))
            log_overshoot_seen_trace.append(_log_scalar(step_log, "cube_tap_overshoot_seen_rate"))
            log_contact_proxy_trace.append(_log_scalar(step_log, "cube_tap_contact_proxy_rate"))
            log_max_disp_along_trace.append(_log_scalar(step_log, "cube_tap_max_disp_along_m"))
            log_max_disp_xy_trace.append(_log_scalar(step_log, "cube_tap_max_disp_xy_m"))

        terms = inner._tap_terms()
        useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
        static_stats = _static_bin_stats(rows, episode_min, episode_max)
        contact_seen_max = _max_with_log(contact_seen_trace, log_contact_seen_trace)
        reaction_seen_max = _max_with_log([], log_reaction_seen_trace)
        useful_seen_max = _max_with_log(useful_seen_trace, log_useful_seen_trace)
        overshoot_seen_max = _max_with_log(overshoot_seen_trace, log_overshoot_seen_trace)
        contact_proxy_max = _max_with_log(contact_proxy_trace, log_contact_proxy_trace)
        max_disp_along_max = _max_with_log(max_disp_along_trace, log_max_disp_along_trace)
        max_disp_xy_max = _max_with_log(max_disp_xy_trace, log_max_disp_xy_trace)
        bin_summary: dict[str, Any] = {
            "bin_idx": bin_idx,
            "episode_min": int(episode_min),
            "episode_max": int(episode_max),
            "runtime_episode_min": _tensor_min(inner._last_d256_reset_episode_index),
            "runtime_episode_mean": _tensor_mean(inner._last_d256_reset_episode_index),
            "runtime_episode_max": _tensor_max(inner._last_d256_reset_episode_index),
            "runtime_episode_mean_first": reset_episode_mean_trace[0] if reset_episode_mean_trace else None,
            "runtime_episode_mean_last": reset_episode_mean_trace[-1] if reset_episode_mean_trace else None,
            "joint_delta_cap_rate_mean_trace": sum(cap_trace) / len(cap_trace),
            "joint_delta_cap_rate_max_trace": max(cap_trace),
            "action_abs_mean_trace_mean": sum(action_abs_mean_trace) / len(action_abs_mean_trace),
            "action_abs_max_trace_max": max(action_abs_max_trace),
            "policy_abs_mean_trace_mean": sum(policy_abs_mean_trace) / len(policy_abs_mean_trace),
            "policy_abs_max_trace_max": max(policy_abs_max_trace),
            "teacher_abs_mean_trace_mean": sum(teacher_abs_mean_trace) / len(teacher_abs_mean_trace),
            "actor_teacher_mse_mean_trace_mean": sum(actor_teacher_mse_trace) / len(actor_teacher_mse_trace),
            "actor_teacher_cosine_mean_trace_mean": sum(actor_teacher_cosine_trace) / len(actor_teacher_cosine_trace),
            "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
            "tap_contact_seen_rate_max_trace": contact_seen_max,
            "tap_contact_seen_rate_post_step_buffer_max_trace": max(contact_seen_trace) if contact_seen_trace else 0.0,
            "tap_contact_seen_rate_log_max_trace": _max_with_log([], log_contact_seen_trace),
            "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
            "tap_reaction_seen_rate_max_trace": reaction_seen_max,
            "tap_reaction_seen_rate_log_max_trace": _max_with_log([], log_reaction_seen_trace),
            "tap_useful_seen_rate": _tensor_mean(useful_seen.float()),
            "tap_useful_seen_rate_max_trace": useful_seen_max,
            "tap_useful_seen_rate_post_step_buffer_max_trace": max(useful_seen_trace) if useful_seen_trace else 0.0,
            "tap_useful_seen_rate_log_max_trace": _max_with_log([], log_useful_seen_trace),
            "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
            "tap_overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
            "tap_overshoot_seen_rate_max_trace": overshoot_seen_max,
            "tap_overshoot_seen_rate_post_step_buffer_max_trace": max(overshoot_seen_trace) if overshoot_seen_trace else 0.0,
            "tap_overshoot_seen_rate_log_max_trace": _max_with_log([], log_overshoot_seen_trace),
            "tap_contact_proxy_rate_last": _tensor_mean(terms["tap_contact_proxy"].float()),
            "tap_contact_proxy_rate_max_trace": contact_proxy_max,
            "tap_contact_proxy_rate_post_step_max_trace": max(contact_proxy_trace) if contact_proxy_trace else 0.0,
            "tap_contact_proxy_rate_log_max_trace": _max_with_log([], log_contact_proxy_trace),
            "tap_contact_face_gap_mean_last_m": _tensor_mean(terms["tap_contact_face_gap_m"]),
            "tap_contact_face_gap_min_last_m": _tensor_min(terms["tap_contact_face_gap_m"]),
            "tap_contact_face_gap_max_last_m": _tensor_max(terms["tap_contact_face_gap_m"]),
            "tap_contact_lateral_mean_last_m": _tensor_mean(terms["tap_contact_lateral_m"]),
            "tap_contact_vertical_offset_mean_last_m": _tensor_mean(terms["tap_contact_vertical_offset_m"]),
            "tcp_cube_dist_mean_last_m": _tensor_mean(terms["tcp_cube_dist"]),
            "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
            "tap_max_disp_along_mean_max_trace_m": max_disp_along_max,
            "tap_max_disp_along_mean_post_step_max_trace_m": max(max_disp_along_trace) if max_disp_along_trace else 0.0,
            "tap_max_disp_along_mean_log_max_trace_m": _max_with_log([], log_max_disp_along_trace),
            "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
            "tap_max_disp_xy_mean_max_trace_m": max_disp_xy_max,
            "tap_max_disp_xy_mean_post_step_max_trace_m": max(max_disp_xy_trace) if max_disp_xy_trace else 0.0,
            "tap_max_disp_xy_mean_log_max_trace_m": _max_with_log([], log_max_disp_xy_trace),
            "tap_contact_vertical_offset_mean_m": _tensor_mean(terms["tap_contact_vertical_offset_m"]),
            "d256_reset_active_rate": _tensor_mean(inner._last_d256_reset_active),
        }
        bin_summary.update(static_stats)
        bin_summary["cap_safe"] = (
            float(bin_summary["joint_delta_cap_rate_max_trace"]) <= float(args.max_cap_rate_for_safe_bin)
        )
        bin_summary["overshoot_safe"] = (
            float(bin_summary["tap_overshoot_seen_rate_max_trace"]) <= float(args.max_overshoot_rate_for_safe_bin)
        )
        bin_summary["useful_enough"] = (
            float(bin_summary["tap_useful_seen_rate_max_trace"]) >= float(args.min_useful_rate_for_safe_bin)
        )
        bin_summary["safe_for_next_smoke_candidate"] = bool(
            bin_summary["cap_safe"] and bin_summary["overshoot_safe"] and bin_summary["useful_enough"]
        )
        bin_rows.append(bin_summary)
        print(
            "[d256-bin-probe] BIN "
            f"{bin_idx} ep={episode_min}-{episode_max} "
            f"cap_max={bin_summary['joint_delta_cap_rate_max_trace']:.6f} "
            f"action_max={bin_summary['action_abs_max_trace_max']:.6f} "
            f"useful_max={bin_summary['tap_useful_seen_rate_max_trace']:.6f} "
            f"overshoot_max={bin_summary['tap_overshoot_seen_rate_max_trace']:.6f}",
            flush=True,
        )

    safe_bins = [
        [int(row["episode_min"]), int(row["episode_max"])]
        for row in bin_rows
        if bool(row["safe_for_next_smoke_candidate"])
    ]
    issues: list[str] = []
    if not reward_finite_all or not obs_finite_all or not action_finite_all:
        issues.append("non-finite reward/obs/action observed")
    if not safe_bins:
        issues.append("no bin met cap/overshoot/useful thresholds")

    highest_cap = max(float(row["joint_delta_cap_rate_max_trace"]) for row in bin_rows)
    lowest_cap = min(float(row["joint_delta_cap_rate_max_trace"]) for row in bin_rows)
    if highest_cap - lowest_cap >= 0.10:
        diagnostic_class = "reset_episode_bin_dependent_action_cap_pressure"
    else:
        diagnostic_class = "cap_pressure_not_strongly_episode_bin_dependent"

    verdict = "D286_D256_RESET_BIN_ACTOR_PROBE_PASS_HAS_SAFE_BIN" if safe_bins and not issues else (
        "D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN"
        if not safe_bins
        else "D286_D256_RESET_BIN_ACTOR_PROBE_WARN"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "d256_reset_bin_actor_probe_summary_d286.json"
    out_md = out_dir / "d256_reset_bin_actor_probe_summary_d286.md"
    out_csv = out_dir / "d256_reset_bin_actor_probe_bins_d286.csv"

    if bin_rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bin_rows)

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "diagnostic_class": diagnostic_class,
        "issues": issues,
        "actor_checkpoint": _rel(args.actor_checkpoint),
        "teacher_checkpoint": _rel(args.teacher_checkpoint),
        "d256_reset_csv_path": _rel(args.d256_reset_csv_path),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "bin_count": int(args.bin_count),
        "episode_ranges": [[int(a), int(b)] for a, b in bins],
        "num_envs": int(args.num_envs),
        "eval_steps": int(args.eval_steps),
        "seed": int(args.seed),
        "action_noise_std": float(args.action_noise_std),
        "warmup_action_source": "zero",
        "exec_source": str(args.exec_source),
        "exec_teacher_blend": float(args.exec_teacher_blend),
        "exec_action_clip_abs": float(args.exec_action_clip_abs),
        "action_scale": float(inner.cfg.action_scale),
        "action_smoothing_alpha": float(inner.cfg.action_smoothing_alpha),
        "max_joint_delta_per_step_rad": float(inner.cfg.max_joint_delta_per_step_rad),
        "contact_joint_delta_scale": float(inner.cfg.contact_joint_delta_scale),
        "fast_cube_joint_delta_scale": float(inner.cfg.fast_cube_joint_delta_scale),
        "joint_target_lead_limit_rad": float(inner.cfg.joint_target_lead_limit_rad),
        "joint_delta_reference": str(inner.cfg.joint_delta_reference),
        "cap_action_threshold_abs": cap_action_threshold_abs,
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "reset_warmup_mode": str(args.reset_warmup_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "tap_stop_after_useful_seen": bool(args.tap_stop_after_useful_seen),
        "tap_stop_after_disp_m": float(args.tap_stop_after_disp_m),
        "tap_contact_slowdown_use_proxy": bool(args.tap_contact_slowdown_use_proxy),
        "tap_useful_terminate": bool(args.tap_useful_terminate),
        "tap_overshoot_terminate": bool(args.tap_overshoot_terminate),
        "bc_teacher_feature_target_mode": str(inner.cfg.bc_teacher_feature_target_mode),
        "bc_teacher_phase_timing": str(inner.cfg.bc_teacher_phase_timing),
        "bc_teacher_linear_phase_steps": int(inner.cfg.bc_teacher_linear_phase_steps),
        "bc_teacher_policy_delta_scale": float(inner.cfg.bc_teacher_policy_delta_scale),
        "bc_teacher_lowx_policy_delta_scale": float(inner.cfg.bc_teacher_lowx_policy_delta_scale),
        "bc_teacher_highx_policy_delta_scale": float(inner.cfg.bc_teacher_highx_policy_delta_scale),
        "max_cap_rate_for_safe_bin": float(args.max_cap_rate_for_safe_bin),
        "max_overshoot_rate_for_safe_bin": float(args.max_overshoot_rate_for_safe_bin),
        "min_useful_rate_for_safe_bin": float(args.min_useful_rate_for_safe_bin),
        "safe_bins": safe_bins,
        "bins": bin_rows,
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)

    print(
        "[d256-bin-probe] SUMMARY "
        f"verdict={verdict} class={diagnostic_class} safe_bins={safe_bins} json={out_json}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
