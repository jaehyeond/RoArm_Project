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
        f"- d256 reset active rate: `{summary['d256_reset_active_rate']}`",
        f"- bc teacher blend mean last: `{summary['bc_teacher_blend_mean_last']}`",
        f"- vertical gate mode/value: `{summary['vertical_gate_mode']}` / `{summary['vertical_gate_value_m']}`",
        f"- env stop/useful terminate: `{summary['tap_stop_after_useful_seen']}` / `{summary['tap_useful_terminate']}`",
        f"- env useful hold rate last/max: `{summary['env_stop_after_useful_hold_rate_last']}` / `{summary['env_stop_after_useful_hold_rate_max_trace']}`",
        f"- zero actions after useful seen: `{summary['zero_actions_after_useful_seen']}`",
        f"- useful action hold rate last/max: `{summary['useful_action_hold_rate_last']}` / `{summary['useful_action_hold_rate_max_trace']}`",
        f"- contact/useful/reaction seen: `{summary['tap_contact_seen_rate']}` / `{summary['tap_useful_seen_rate']}` / `{summary['tap_reaction_seen_rate']}`",
        f"- success rate: `{summary['tap_success_rate']}`",
        f"- overshoot seen rate: `{summary['tap_overshoot_seen_rate']}`",
        f"- max disp along mean/max: `{summary['tap_max_disp_along_mean_m']}` / `{summary['tap_max_disp_along_max_m']}`",
        f"- max disp xy mean/max: `{summary['tap_max_disp_xy_mean_m']}` / `{summary['tap_max_disp_xy_max_m']}`",
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
    parser.add_argument("--max_joint_delta_per_step_rad", type=float, default=None)
    parser.add_argument("--d256_reset_csv_path", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument("--fixed_push_dir_x", type=float, default=1.0)
    parser.add_argument("--fixed_push_dir_y", type=float, default=0.0)
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--bc_teacher_checkpoint_path", type=Path, default=None)
    parser.add_argument("--bc_teacher_blend", type=float, default=0.0)
    parser.add_argument("--bc_teacher_imitation_reward_scale", type=float, default=0.0)
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--tap_stop_after_useful_seen", action="store_true")
    parser.add_argument("--tap_useful_terminate", action="store_true")
    parser.add_argument("--zero_actions_after_useful_seen", action="store_true")
    parser.add_argument("--min_useful_seen_rate", type=float, default=0.01)
    parser.add_argument("--max_overshoot_seen_rate", type=float, default=0.05)
    parser.add_argument("--max_joint_delta_cap_rate", type=float, default=0.25)
    parser.add_argument("--max_vertical_offset_m", type=float, default=0.08)
    parser.add_argument("--vertical_gate_mode", choices=("last", "min_contact"), default="last")
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
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

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    if args.episode_length_s is not None:
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
    env_cfg.bc_teacher_blend = float(args.bc_teacher_blend)
    env_cfg.bc_teacher_imitation_reward_scale = float(args.bc_teacher_imitation_reward_scale)
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)
    if args.bc_teacher_checkpoint_path is not None:
        env_cfg.bc_teacher_checkpoint_path = str(args.bc_teacher_checkpoint_path)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)

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
    policy = runner.get_inference_policy(device=inner.device)

    inner.episode_length_buf[:] = inner.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    print("[teacher-off-eval] warmup_reset_done", flush=True)

    inf = torch.full((inner.num_envs,), math.inf, device=inner.device)
    min_contact_vertical = inf.clone()
    tcp_threshold_seen = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    reward_finite_all = True
    obs_finite_all = _obs_is_finite(torch, obs)
    action_finite_all = True
    cap_trace: list[float] = []
    action_abs_mean_trace: list[float] = []
    action_abs_max_trace: list[float] = []
    useful_action_hold_rate_trace: list[float] = []
    env_stop_after_useful_hold_rate_trace: list[float] = []
    step_rows: list[dict[str, float | int]] = []

    for step in range(int(args.eval_steps)):
        with torch.inference_mode():
            actions = policy(obs)
            hold_mask = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
            if bool(args.zero_actions_after_useful_seen):
                hold_mask = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
                actions = torch.where(hold_mask.unsqueeze(-1), torch.zeros_like(actions), actions)
            action_finite_all = action_finite_all and bool(torch.isfinite(actions).all().detach().cpu().item())
            obs, rewards, _, _ = env.step(actions)
        useful_action_hold_rate = _tensor_mean(hold_mask.float())
        reward_finite_all = reward_finite_all and bool(torch.isfinite(rewards).all().detach().cpu().item())
        obs_finite_all = obs_finite_all and _obs_is_finite(torch, obs)

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
        action_abs_mean = _tensor_mean(torch.abs(actions).mean(dim=-1))
        action_abs_max = _tensor_max(torch.abs(actions))
        env_hold = _tensor_mean(getattr(inner, "_last_tap_stop_after_useful_hold", torch.zeros(inner.num_envs, device=inner.device)))
        cap_trace.append(cap_mean)
        action_abs_mean_trace.append(action_abs_mean)
        action_abs_max_trace.append(action_abs_max)
        useful_action_hold_rate_trace.append(useful_action_hold_rate)
        env_stop_after_useful_hold_rate_trace.append(env_hold)
        step_rows.append(
            {
                "step": step,
                "tap_contact_seen_rate": _tensor_mean(inner._tap_contact_seen.float()),
                "tap_reaction_seen_rate": _tensor_mean(inner._tap_reaction_seen.float()),
                "tap_useful_seen_rate": _tensor_mean(useful_seen.float()),
                "tap_success_rate": _tensor_mean(inner._tap_success_flag.float()),
                "tap_overshoot_seen_rate": _tensor_mean(inner._tap_overshoot_seen.float()),
                "tap_contact_proxy_rate": _tensor_mean(contact.float()),
                "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
                "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
                "tap_contact_vertical_offset_mean_m": _tensor_mean(terms["tap_contact_vertical_offset_m"]),
                "tcp_cube_dist_mean_m": _tensor_mean(terms["tcp_cube_dist"]),
                "joint_delta_cap_rate_mean": cap_mean,
                "policy_action_abs_mean": action_abs_mean,
                "policy_action_abs_max": action_abs_max,
                "useful_action_hold_rate": useful_action_hold_rate,
                "env_stop_after_useful_hold_rate": env_hold,
                "bc_teacher_blend_mean": _tensor_mean(inner._last_bc_teacher_blend),
                "d256_reset_active_rate": _tensor_mean(inner._last_d256_reset_active),
            }
        )

    final_terms = inner._tap_terms()
    useful_seen = inner._tap_contact_seen & inner._tap_reaction_seen & ~inner._tap_overshoot_seen
    issues: list[str] = []
    d256_active = _tensor_mean(inner._last_d256_reset_active)
    bc_blend_last = _tensor_mean(inner._last_bc_teacher_blend)
    useful_rate = _tensor_mean(useful_seen.float())
    overshoot_rate = _tensor_mean(inner._tap_overshoot_seen.float())
    cap_rate_last = _tensor_mean(inner._last_joint_delta_cap_rate)
    cap_rate_max = max(cap_trace) if cap_trace else 0.0
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
        "max_joint_delta_per_step_rad": float(env_cfg.max_joint_delta_per_step_rad),
        "seed": int(args.seed),
        "d256_reset_csv_path": _rel(args.d256_reset_csv_path),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "bc_teacher_checkpoint_path": _rel(args.bc_teacher_checkpoint_path) if args.bc_teacher_checkpoint_path else "",
        "bc_teacher_blend": float(args.bc_teacher_blend),
        "bc_teacher_imitation_reward_scale": float(args.bc_teacher_imitation_reward_scale),
        "bc_teacher_blend_mean_last": bc_blend_last,
        "vertical_gate_mode": str(args.vertical_gate_mode),
        "vertical_gate_value_m": vertical_gate_value,
        "tap_stop_after_useful_seen": bool(args.tap_stop_after_useful_seen),
        "tap_useful_terminate": bool(args.tap_useful_terminate),
        "env_stop_after_useful_hold_rate_last": (
            env_stop_after_useful_hold_rate_trace[-1] if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "env_stop_after_useful_hold_rate_max_trace": (
            max(env_stop_after_useful_hold_rate_trace) if env_stop_after_useful_hold_rate_trace else 0.0
        ),
        "zero_actions_after_useful_seen": bool(args.zero_actions_after_useful_seen),
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
        "tap_max_disp_along_mean_m": _tensor_mean(inner._tap_max_disp_along),
        "tap_max_disp_along_max_m": _tensor_max(inner._tap_max_disp_along),
        "tap_max_disp_xy_mean_m": _tensor_mean(inner._tap_max_disp_xy),
        "tap_max_disp_xy_max_m": _tensor_max(inner._tap_max_disp_xy),
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
