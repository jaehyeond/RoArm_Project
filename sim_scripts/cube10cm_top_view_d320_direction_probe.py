#!/usr/bin/env python3
"""Probe whether candidate8 can execute non-+x fixed push directions.

This is a small non-render runtime probe for D320 Step 4. It uses the existing
fixed_push_dir_x/y env contract, candidate8 zero residual actions, and hybrid
stop after useful. It does not train PPO and does not add controller conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT = RUNTIME_ROOT / "data_conveyor_d320" / "direction_probe" / "direction_probe_summary.json"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


DIRECTIONS = {
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3204)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=580)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--robot-usd-path", type=Path, default=DEFAULT_USD)
    return parser.parse_args()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def tensor_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return [float(x) for x in value.tolist()]


def make_contract_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        robot_usd_path=args.robot_usd_path,
        runtime_dir=args.out_json.parent,
        summary_json=args.out_json.parent / "unused_summary.json",
        summary_out=args.out_json.parent / "unused_summary.out",
        experiment_name="cube10cm_top_view_d320_direction_probe",
        fixed_cube_x_m=0.24,
        fixed_cube_y_m=0.0,
        cube_randomization_half_extent_x_m=0.04,
        cube_randomization_half_extent_y_m=0.04,
        policy_target_disp_m=0.006,
        precontact_clearance_m=0.040,
        episode_length_s=6.0,
        step_clip_rad=0.010,
        joint_target_lead_limit_rad=0.060,
        action_scale=0.050,
        rl_action_mode="candidate8_diffik_target_residual",
        candidate6_diffik_push_steps=int(args.steps),
        candidate6_diffik_residual_scale_rad=0.002,
        candidate6_diffik_lambda=0.010,
        candidate8_diffik_target_residual_forward_m=0.004,
        candidate8_diffik_target_residual_lateral_m=0.012,
        candidate8_diffik_target_residual_height_m=0.004,
        tap_success_terminate=False,
        disable_tap_overshoot_terminate=True,
        candidate6_diffik_no_hold_after_tap_success=False,
        candidate6_diffik_target_base_mode="previous_joint_target",
        candidate6_diffik_target_path_mode="near_face_goal",
        candidate6_diffik_cube_reference_mode="current_pose",
        tap_transient_disp_reward_scale=None,
        tap_overshoot_penalty_scale=None,
        action_penalty_scale=None,
    )


def run_direction(
    *,
    args: argparse.Namespace,
    env: Any,
    inner: Any,
    torch: Any,
    label: str,
    direction: tuple[float, float],
) -> dict[str, Any]:
    inner.cfg.fixed_push_dir_x = float(direction[0])
    inner.cfg.fixed_push_dir_y = float(direction[1])
    with torch.inference_mode():
        env.reset()
        actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=inner.device)
        for _ in range(int(args.steps)):
            env.step(actions)
        inner._compute_intermediate_values()
        terms = inner._tap_terms()
        contact = inner._tap_contact_seen.detach().cpu()
        reaction = inner._tap_reaction_seen.detach().cpu()
        overshoot = inner._tap_overshoot_seen.detach().cpu()
        max_disp = inner._tap_max_disp_xy.detach().cpu()
        push_dir = inner._push_dir_xy.detach().cpu()
        hybrid_step = inner._candidate8_hybrid_stop_step.detach().cpu()
        useful = (contact > 0.5) & (reaction > 0.5) & (max_disp >= 0.001) & ~(overshoot > 0.5)
        env_rows = []
        for idx in range(inner.num_envs):
            env_rows.append(
                {
                    "env_id": int(idx),
                    "push_dir_xy": [float(push_dir[idx, 0]), float(push_dir[idx, 1])],
                    "contact": int(contact[idx].item() > 0.5),
                    "reaction": int(reaction[idx].item() > 0.5),
                    "useful_filter": int(useful[idx].item()),
                    "overshoot": int(overshoot[idx].item() > 0.5),
                    "max_disp_xy_m": float(max_disp[idx].item()),
                    "disp_along_m_final": float(terms["disp_along"][idx].detach().cpu().item()),
                    "hybrid_stop_step": int(hybrid_step[idx].item()),
                }
            )
    return {
        "direction": label,
        "requested_push_dir_xy": [float(direction[0]), float(direction[1])],
        "num_envs": int(args.num_envs),
        "contact_count": sum(row["contact"] for row in env_rows),
        "reaction_count": sum(row["reaction"] for row in env_rows),
        "useful_filter_count": sum(row["useful_filter"] for row in env_rows),
        "overshoot_count": sum(row["overshoot"] for row in env_rows),
        "max_disp_xy_m_mean": float(sum(row["max_disp_xy_m"] for row in env_rows) / max(1, len(env_rows))),
        "max_disp_xy_m_max": float(max((row["max_disp_xy_m"] for row in env_rows), default=math.nan)),
        "env_rows": env_rows,
    }


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
    sim_app = app_launcher.app
    try:
        import gymnasium as gym
        import torch

        import roarm_rl  # noqa: F401
        from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
        from roarm_rl.train_cube_tap10cm_ppo_smoke import _apply_candidate6_contract

        cfg = RoArmCubeTap10cmEnvCfg()
        contract_args = make_contract_args(args)
        contract = _apply_candidate6_contract(cfg, contract_args)
        cfg.candidate8_hybrid_stop_after_useful = True
        cfg.tap_success_terminate = False
        cfg.tap_overshoot_terminate = False
        cfg.scene.clone_in_fabric = False
        env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        results = [
            run_direction(args=args, env=env, inner=inner, torch=torch, label=label, direction=direction)
            for label, direction in DIRECTIONS.items()
        ]
        summary = {
            "artifact": "d320_candidate8_direction_probe",
            "runtime": "DIRECTION_PROBE_NO_TRAINING",
            "code_contract": {
                "env_fixed_push_dir_fields": "roarm_rl/roarm_cube_push_env.py:1593-1605",
                "candidate8_uses_push_dir_and_lateral_dir": "roarm_rl/roarm_cube_push_env.py:1226-1238",
            },
            "steps": int(args.steps),
            "num_envs_per_direction": int(args.num_envs),
            "base_contract": contract,
            "results": results,
        }
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            "[d320-direction-probe] done "
            + " ".join(
                f"{item['direction']}:useful={item['useful_filter_count']}/{item['num_envs']} "
                f"overshoot={item['overshoot_count']}/{item['num_envs']}"
                for item in results
            )
            + f" out={args.out_json}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
