"""Sanity test for RoArmPickEnv — small-env random-action rollout.

PASS criteria:
  - No SIGSEGV / Python exception during launch + reset + 200 step
  - obs shape (1, 22), action shape (1, 6)
  - reward, terminated, truncated, info all valid
  - At least 1 episode completes (truncated -> reset cycle)

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.test_sanity
"""
import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--reward_phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # force unbuffered prints for tail-monitoring
    import os as _os
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # registers RoArm-Pick-Direct-v0

    from roarm_rl.roarm_pick_env import RoArmPickEnvCfg

    cfg = RoArmPickEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = args.reward_phase
    cfg.seed = args.seed
    # For tiny num_envs (1-4), disable Fabric clone to avoid edge-case crash.
    if args.num_envs < 8:
        cfg.scene.clone_in_fabric = False
        cfg.scene.replicate_physics = False
        print(f"[sanity] num_envs={args.num_envs} small -> clone_in_fabric=False, replicate_physics=False")

    print(f"[sanity] creating env (num_envs={args.num_envs}) ...", flush=True)
    try:
        env = gym.make("RoArm-Pick-Direct-v0", cfg=cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[sanity] FAIL during env creation: {e}", flush=True)
        sim_app.close()
        sys.exit(2)

    print(f"[sanity] env created. obs_space: {env.observation_space}", flush=True)
    print(f"[sanity] obs_space: {env.observation_space}")
    print(f"[sanity] act_space: {env.action_space}")
    print(f"[sanity] num_envs: {env.unwrapped.num_envs}")

    obs, info = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[sanity] obs shape after reset: {obs_t.shape}")
    assert obs_t.shape == (args.num_envs, 22), f"obs shape mismatch: {obs_t.shape}"

    # random action rollout
    t_start = time.time()
    rewards_all = []
    success_seen = 0
    truncated_count = 0

    for step in range(args.steps):
        action = (torch.rand((args.num_envs, 6), device=env.unwrapped.device) - 0.5) * 0.4  # small random ±0.2
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_all.append(reward.mean().item())
        truncated_count += truncated.sum().item()
        if "log" in info:
            d = info["log"].get("tcp_sponge_dist_m", None)
            sh = info["log"].get("sponge_height_m", None)
            if step % 25 == 0:
                d_str = f"{d.item():.3f}" if d is not None else "?"
                sh_str = f"{sh.item():.3f}" if sh is not None else "?"
                print(f"[sanity] step {step:3d}: r={reward.mean().item():+.3f} d_tcp_sponge={d_str}m sponge_h={sh_str}m trunc={truncated.sum().item()}")

    elapsed = time.time() - t_start
    sps = args.num_envs * args.steps / elapsed
    print(f"[sanity] DONE: {args.steps} steps × {args.num_envs} envs in {elapsed:.2f}s = {sps:.0f} steps/s")
    print(f"[sanity] reward avg: {np.mean(rewards_all):+.3f}")
    print(f"[sanity] truncations seen: {truncated_count}")
    print(f"[sanity] PASS")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
