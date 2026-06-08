"""Random-action sanity for the default-off 10cm/0.72kg cube tap env.

This is a tiny env-contract runtime check, not PPO, dataset generation, robot
control, or action-teacher construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_env_random_sanity.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_env_random_sanity_summary.out"
ENV_ID = "RoArm-CubeTap10cm-Direct-v0"
PROJECT_TABLE_Z = -0.012117


def _table_z_flat_terrain(difficulty: float, cfg: Any) -> tuple[list[Any], np.ndarray]:
    """Generate a local flat mesh at the project table height."""
    from isaaclab.terrains.trimesh.utils import make_plane

    plane_mesh = make_plane(cfg.size, PROJECT_TABLE_Z, center_zero=False)
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, PROJECT_TABLE_Z)
    return [plane_mesh], np.array(origin)


def _scalar(value: Any) -> float | int | str:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def _write_result(
    out_json: Path,
    out_summary: Path,
    result: dict[str, Any],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "line1 artifact=cube10cm_tap_rl_env_random_sanity_v1 "
        f"status={result['status']} gpu_runtime={result.get('gpu_runtime', 'UNKNOWN')} "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 env_contract "
            f"env_id={result.get('env_id', ENV_ID)} "
            f"cube_size_m={result.get('cube_size_m', 'UNKNOWN')} "
            f"cube_mass_kg={result.get('cube_mass_kg', 'UNKNOWN')} "
            f"terrain_table_z_m={result.get('terrain_table_z_m', 'UNKNOWN')} "
            f"final_1cm_required={result.get('final_1cm_required', 'UNKNOWN')}"
        ),
        (
            "line3 rollout "
            f"num_envs={result.get('num_envs', 'NA')} steps={result.get('steps', 'NA')} "
            f"obs_shape={result.get('obs_shape', 'NA')} "
            f"reward_finite={result.get('reward_finite', 'NA')} "
            f"truncated_count={result.get('truncated_count', 'NA')} "
            f"terminated_count={result.get('terminated_count', 'NA')}"
        ),
        (
            "line4 tap_logs "
            f"required_log_keys_present={result.get('required_log_keys_present', 'NA')} "
            f"contact_seen={result.get('last_log', {}).get('cube_tap_contact_seen_rate', 'NA')} "
            f"reaction_signal_now={result.get('last_log', {}).get('cube_tap_reaction_signal_now_rate', 'NA')} "
            f"reaction_contact_context={result.get('last_log', {}).get('cube_tap_reaction_contact_context_rate', 'NA')} "
            f"reaction_seen={result.get('last_log', {}).get('cube_tap_reaction_seen_rate', 'NA')} "
            f"overshoot_seen={result.get('last_log', {}).get('cube_tap_overshoot_seen_rate', 'NA')} "
            f"tap_success={result.get('last_log', {}).get('cube_tap_success_rate', 'NA')}"
        ),
        (
            "line5 verdict "
            f"random_sanity={result['status']} "
            f"blocker={result.get('blocker', 'NONE')} "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
    ]
    out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--random_action_abs", type=float, default=0.2)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_LOCAL_USD)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    sim_app = None
    env = None
    started = time.time()
    try:
        if not args.robot_usd_path.exists():
            raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
        sim_app = app_launcher.app

        import gymnasium as gym
        import torch

        import roarm_rl  # noqa: F401 - registers envs lazily
        from roarm_rl.roarm_cube_push_env import CUBE10CM_MASS_KG, CUBE10CM_SIZE_M, RoArmCubeTap10cmEnvCfg
        from roarm_rl.roarm_stack_env import TABLE_Z
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

        if abs(float(TABLE_Z) - PROJECT_TABLE_Z) > 1.0e-12:
            raise AssertionError(f"table height mismatch: env={TABLE_Z} sanity={PROJECT_TABLE_Z}")

        flat_cfg = MeshPlaneTerrainCfg(proportion=1.0)
        flat_cfg.function = _table_z_flat_terrain
        cfg = RoArmCubeTap10cmEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.seed = int(args.seed)
        cfg.sim.device = str(args.device)
        cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(2.0, 2.0),
                num_rows=1,
                num_cols=1,
                border_width=0.0,
                sub_terrains={"flat": flat_cfg},
                use_cache=False,
            ),
            env_spacing=cfg.scene.env_spacing,
            physics_material=cfg.terrain.physics_material,
            visual_material=cfg.terrain.visual_material,
        )
        cfg.robot.spawn.usd_path = str(args.robot_usd_path)
        if args.num_envs < 8:
            cfg.scene.clone_in_fabric = False
            cfg.scene.replicate_physics = False

        mass_kg = float(cfg.sponge.spawn.mass_props.mass)
        contract_ok = (
            abs(float(cfg.cube_size_x_m) - 0.100) <= 1.0e-12
            and abs(float(cfg.cube_size_y_m) - 0.100) <= 1.0e-12
            and abs(float(cfg.cube_size_z_m) - 0.100) <= 1.0e-12
            and abs(mass_kg - 0.720) <= 1.0e-12
            and not bool(cfg.tap_final_relocation_required)
            and str(cfg.tap_objective_name) == "tap_reaction_contact_not_final_relocation"
            and abs(float(cfg.tap_reaction_disp_m) - 0.001) <= 1.0e-12
            and abs(float(cfg.tap_overshoot_disp_m) - 0.020) <= 1.0e-12
        )
        if not contract_ok:
            raise AssertionError("10cm tap env cfg contract mismatch before env creation")

        print(f"[tap10cm-sanity] creating {ENV_ID} num_envs={args.num_envs}", flush=True)
        env = gym.make(ENV_ID, cfg=cfg)
        inner = env.unwrapped
        obs, _info = env.reset()
        obs_t = obs["policy"] if isinstance(obs, dict) else obs
        expected_shape = (args.num_envs, cfg.observation_space)
        if tuple(obs_t.shape) != expected_shape:
            raise AssertionError(f"obs shape mismatch: expected={expected_shape} actual={tuple(obs_t.shape)}")

        rewards_all: list[float] = []
        truncated_count = 0
        terminated_count = 0
        last_log: dict[str, Any] = {}
        for step in range(int(args.steps)):
            action = (torch.rand((args.num_envs, cfg.action_space), device=inner.device) - 0.5) * (
                2.0 * float(args.random_action_abs)
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if not torch.isfinite(reward).all():
                raise AssertionError(f"non-finite reward at step {step}")
            rewards_all.append(float(reward.mean().item()))
            truncated_count += int(truncated.sum().item())
            terminated_count += int(terminated.sum().item())
            if "log" in info:
                last_log = {key: _scalar(value) for key, value in info["log"].items()}
            if step % max(1, int(args.steps) // 4) == 0:
                print(
                    "[tap10cm-sanity] "
                    f"step={step} reward_mean={reward.mean().item():+.6f} "
                    f"contact_seen={last_log.get('cube_tap_contact_seen_rate', 'NA')} "
                    f"reaction_seen={last_log.get('cube_tap_reaction_seen_rate', 'NA')} "
                    f"overshoot_seen={last_log.get('cube_tap_overshoot_seen_rate', 'NA')}",
                    flush=True,
                )

        required_log_keys = {
            "cube_tap_objective_final_relocation_required",
            "cube_tap_contact_seen_rate",
            "cube_tap_reaction_signal_now_rate",
            "cube_tap_reaction_contact_context_rate",
            "cube_tap_reaction_seen_rate",
            "cube_tap_overshoot_seen_rate",
            "cube_tap_success_rate",
            "cube_tap_max_disp_along_m",
            "cube_push_grasped_marker_rate",
        }
        missing_logs = sorted(required_log_keys - set(last_log))
        final_required_log = float(last_log.get("cube_tap_objective_final_relocation_required", 1.0))
        if missing_logs:
            raise AssertionError(f"missing required tap log keys: {missing_logs}")
        if final_required_log != 0.0:
            raise AssertionError(f"final relocation flag must be 0, got {final_required_log}")

        result = {
            "artifact_type": "cube10cm_tap_rl_env_random_sanity_v1",
            "branch": "professor_cube10cm_tap_reaction_quality_tier",
            "status": "PASS",
            "gpu_runtime": "YES_LOCAL_TINY_ISAACLAB_RANDOM_SANITY",
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
            "env_id": ENV_ID,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "device": str(args.device),
            "robot_usd_path": str(args.robot_usd_path),
            "cube_size_m": CUBE10CM_SIZE_M,
            "cube_mass_kg": CUBE10CM_MASS_KG,
            "terrain_table_z_m": PROJECT_TABLE_Z,
            "final_1cm_required": False,
            "obs_shape": list(obs_t.shape),
            "reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "reward_finite": True,
            "truncated_count": truncated_count,
            "terminated_count": terminated_count,
            "required_log_keys_present": True,
            "missing_required_log_keys": [],
            "last_log": last_log,
            "elapsed_s": time.time() - started,
        }
        _write_result(args.out_json, args.out_summary, result)
        return 0
    except Exception as exc:
        result = {
            "artifact_type": "cube10cm_tap_rl_env_random_sanity_v1",
            "branch": "professor_cube10cm_tap_reaction_quality_tier",
            "status": "BLOCKED",
            "gpu_runtime": "NO_OR_FAILED_BEFORE_PASS",
            "dataset_generation": False,
            "training": False,
            "robot_control": False,
            "ssh": False,
            "env_id": ENV_ID,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "device": str(args.device),
            "robot_usd_path": str(args.robot_usd_path),
            "cube_size_m": "UNKNOWN",
            "cube_mass_kg": "UNKNOWN",
            "terrain_table_z_m": PROJECT_TABLE_Z,
            "final_1cm_required": "UNKNOWN",
            "required_log_keys_present": False,
            "blocker": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "elapsed_s": time.time() - started,
        }
        _write_result(args.out_json, args.out_summary, result)
        return 2
    finally:
        if env is not None:
            env.close()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
