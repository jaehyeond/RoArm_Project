#!/usr/bin/env python3
"""Base-only pose-bin sweep for the 10cm cube useful-tap objective.

This script is diagnosis only.  It does not train PPO and does not change the
env reward/control contract.  It reuses the Candidate6/Candidate8 base contract
from the smoke runner, then evaluates zero policy on fixed cube pose bins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RUNTIME_DIR = (
    REPO_ROOT
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
DEFAULT_SUMMARY_JSON = (
    DEFAULT_RUNTIME_DIR / "cube10cm_tap_useful_posebin_sweep_summary.json"
)
DEFAULT_SUMMARY_OUT = (
    DEFAULT_RUNTIME_DIR / "cube10cm_tap_useful_posebin_sweep_summary.out"
)
DEFAULT_USD = (
    REPO_ROOT
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def _float_list(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return [float(value) for value in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base-only useful-tap fixed pose-bin sweep"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1030)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=580)
    parser.add_argument("--x_values", default="0.09,0.14,0.24,0.34,0.39")
    parser.add_argument("--y_values", default="-0.15,-0.10,0.0,0.10,0.15")
    parser.add_argument("--runtime_dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary_out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_USD)
    return parser.parse_args()


def _base_args(args: argparse.Namespace, fixed_x: float, fixed_y: float) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        max_iterations=0,
        num_steps_per_env=64,
        eval_steps=int(args.eval_steps),
        save_interval=1,
        robot_usd_path=args.robot_usd_path,
        runtime_dir=args.runtime_dir,
        summary_json=args.summary_json,
        summary_out=args.summary_out,
        experiment_name="cube10cm_tap_useful_posebin_sweep",
        fixed_cube_x_m=float(fixed_x),
        fixed_cube_y_m=float(fixed_y),
        cube_randomization_half_extent_x_m=0.0,
        cube_randomization_half_extent_y_m=0.0,
        policy_target_disp_m=0.006,
        precontact_clearance_m=0.040,
        episode_length_s=6.08,
        step_clip_rad=0.010,
        joint_target_lead_limit_rad=0.060,
        action_scale=0.050,
        rl_action_mode="candidate8_diffik_target_residual",
        candidate6_diffik_push_steps=580,
        candidate6_diffik_residual_scale_rad=0.002,
        candidate6_diffik_lambda=0.010,
        candidate8_diffik_target_residual_forward_m=0.004,
        candidate8_diffik_target_residual_lateral_m=0.012,
        candidate8_diffik_target_residual_height_m=0.004,
        tap_success_terminate=True,
        candidate6_diffik_no_hold_after_tap_success=False,
        candidate6_diffik_target_base_mode="previous_joint_target",
        candidate6_diffik_target_path_mode="near_face_goal",
        candidate6_diffik_cube_reference_mode="current_pose",
        init_at_random_ep_len=False,
        load_checkpoint=None,
        initial_policy_eval=False,
        constant_policy_action=None,
        ppo_init_noise_std=0.2,
        tap_transient_disp_reward_scale=None,
        tap_overshoot_penalty_scale=None,
        action_penalty_scale=None,
    )


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _useful_fail_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    useful = _metric(metrics, "tap_useful_seen_max")
    contact_reaction = _metric(metrics, "tap_contact_reaction_seen_max")
    no_overshoot = _metric(metrics, "tap_no_overshoot_seen_min")
    if useful is None or useful < 0.999:
        reasons.append(f"useful_seen={useful}")
    if contact_reaction is None or contact_reaction < 0.999:
        reasons.append(f"contact_reaction_seen={contact_reaction}")
    if no_overshoot is None or no_overshoot < 0.999:
        reasons.append(f"no_overshoot_seen={no_overshoot}")
    return reasons


def _bin_line(row: dict[str, Any]) -> str:
    metrics = row["metrics"]
    reasons = row["useful_fail_reasons"]
    return (
        "bin "
        f"x={row['x']:.5f} y={row['y']:.5f} "
        f"useful_seen={metrics.get('tap_useful_seen_max')} "
        f"contact_reaction_seen={metrics.get('tap_contact_reaction_seen_max')} "
        f"no_overshoot_seen={metrics.get('tap_no_overshoot_seen_min')} "
        f"contact_seen={metrics.get('tap_contact_seen_max')} "
        f"reaction_seen={metrics.get('reaction_seen_max')} "
        f"overshoot_seen={metrics.get('tap_overshoot_max')} "
        f"target_band={metrics.get('tap_target_band_max')} "
        f"event_rate={metrics.get('tap_success_event_rate_per_env')} "
        f"tap_success_episode={metrics.get('tap_success_episode_rate')} "
        f"preflight_pass={row['preflight_pass']} "
        f"violations={len(row['contract_violations'])} "
        f"useful_fail={bool(reasons)} "
        f"reasons={';'.join(reasons) if reasons else 'none'}"
    )


def main() -> None:
    args = _parse_args()
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

    x_values = _float_list(args.x_values)
    y_values = _float_list(args.y_values)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
    simulation_app = app_launcher.app
    rows: list[dict[str, Any]] = []

    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import roarm_rl  # noqa: F401 - registers Gym environments
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.train_cube_tap10cm_ppo_smoke import (
        _apply_candidate6_contract,
        _contract_violations,
        _rollout,
    )

    for fixed_y in y_values:
        for fixed_x in x_values:
            bin_args = _base_args(args, fixed_x=fixed_x, fixed_y=fixed_y)
            env_cfg = RoArmCubeTap10cmEnvCfg()
            contract = _apply_candidate6_contract(env_cfg, bin_args)
            violations = _contract_violations(contract, bin_args)
            env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg)
            wrapped_env = RslRlVecEnvWrapper(env)
            inner_env = wrapped_env.unwrapped
            try:
                metrics = _rollout(
                    wrapped_env,
                    inner_env,
                    torch,
                    steps=int(args.eval_steps),
                    policy=None,
                    label="zero_policy_pre_eval",
                )
            finally:
                wrapped_env.close()
            bridge_preflight_pass = bool(
                float(metrics["candidate6_diffik_active_rate_max"]) > 0.0
                and metrics["candidate6_diffik_numeric_ok_rate_min"] is not None
                and float(metrics["candidate6_diffik_numeric_ok_rate_min"]) >= 0.999
            )
            preflight_pass = bool(
                not violations
                and metrics["reward_finite_all"]
                and metrics["obs_finite_all"]
                and metrics["action_finite_all"]
                and bridge_preflight_pass
            )
            row = {
                "x": float(fixed_x),
                "y": float(fixed_y),
                "seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "eval_steps": int(args.eval_steps),
                "contract": contract,
                "contract_violations": violations,
                "preflight_pass": preflight_pass,
                "metrics": metrics,
            }
            row["useful_fail_reasons"] = _useful_fail_reasons(metrics)
            rows.append(row)
            print(_bin_line(row), flush=True)

    useful_fail_rows = [row for row in rows if row["useful_fail_reasons"]]
    summary = {
        "audit": "cube10cm_tap_useful_posebin_sweep",
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "eval_steps": int(args.eval_steps),
        "x_values": x_values,
        "y_values": y_values,
        "useful_definition": "contact_seen && reaction_seen && !overshoot_seen",
        "policy_target_disp_m": 0.006,
        "target_band_is_quality_tier": True,
        "rows": rows,
        "useful_fail_count": len(useful_fail_rows),
        "useful_fail_bins": [
            {
                "x": row["x"],
                "y": row["y"],
                "reasons": row["useful_fail_reasons"],
                "metrics": row["metrics"],
            }
            for row in useful_fail_rows
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "cube10cm_tap_useful_posebin_sweep "
        f"seed={args.seed} num_envs={args.num_envs} eval_steps={args.eval_steps} "
        f"x_values={x_values} y_values={y_values} "
        f"useful_fail_count={len(useful_fail_rows)}",
        "definition useful_tap=contact_seen_and_reaction_seen_and_no_overshoot "
        "target_band=quality_tier_only ppo_training=NO large_ppo=NO dataset=NO",
    ]
    lines.extend(_bin_line(row) for row in rows)
    lines.append(f"outputs summary_json={args.summary_json} summary_out={args.summary_out}")
    args.summary_out.write_text("\n".join(lines) + "\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
