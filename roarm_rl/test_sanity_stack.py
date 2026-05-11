"""Sanity test for RoArmStackEnv (Phase 1.B-alpha).

PASS criteria:
  - No SIGSEGV / Python exception during launch + reset + N step
  - obs shape (num_envs, 28), action shape (num_envs, 6)
  - reward, terminated (always False), truncated, info all valid
  - At least 1 episode truncates (max_episode_length boundary)
  - target_pos in obs is the configured L1.spot1 (env-local coord)

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.test_sanity_stack --num_envs 4
"""
import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)  # 1 full episode
    parser.add_argument("--reward_phase", type=int, default=4, choices=[4, 5, 6])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import os as _os
    _os.environ.setdefault("PYTHONUNBUFFERED", "1")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # registers RoArm-Stack-Direct-v0

    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, TARGET_L1_SPOT1

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = args.reward_phase
    cfg.seed = args.seed
    if args.num_envs < 8:
        cfg.scene.clone_in_fabric = False
        cfg.scene.replicate_physics = False
        print(f"[sanity-stack] num_envs={args.num_envs} small -> clone_in_fabric=False, replicate_physics=False")

    print(f"[sanity-stack] creating env (num_envs={args.num_envs}, reward_phase={args.reward_phase}) ...", flush=True)
    try:
        env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[sanity-stack] FAIL env creation: {e}", flush=True)
        sim_app.close()
        sys.exit(2)

    inner = env.unwrapped
    print(f"[sanity-stack] env created.", flush=True)
    print(f"[sanity-stack] obs_space: {env.observation_space}")
    print(f"[sanity-stack] act_space: {env.action_space}")
    print(f"[sanity-stack] num_envs: {inner.num_envs}")
    print(f"[sanity-stack] target_pos (cfg): {TARGET_L1_SPOT1}")
    print(f"[sanity-stack] max_episode_length: {inner.max_episode_length}")

    obs, info = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[sanity-stack] obs shape after reset: {obs_t.shape}")
    assert obs_t.shape == (args.num_envs, 28), \
        f"FAIL: obs shape expected ({args.num_envs}, 28), got {obs_t.shape}"

    # Verify target_pos in obs.
    # 28-dim layout: [0-5 dof_pos] [6-11 joint_vel] [12-14 sponge_pos_local]
    #                [15-18 sponge_quat] [19-21 tcp_to_sponge]
    #                [22-24 target_pos_local] [25-27 sponge_to_target]
    # _target_world is set as env_origin + TARGET_L1_SPOT1 (per-env), so
    # target_local = target_world - env_origin = TARGET_L1_SPOT1 for ALL envs.
    target_in_obs = obs_t[0, 22:25].cpu().numpy()
    expected = np.clip(np.array(TARGET_L1_SPOT1), -5.0, 5.0)  # obs is clamped to [-5,5]
    diff = np.abs(target_in_obs - expected).max()
    print(f"[sanity-stack] target_in_obs[0]: {target_in_obs}")
    print(f"[sanity-stack] expected (TARGET_L1_SPOT1): {expected}")
    print(f"[sanity-stack] diff_max       : {diff:.6f}")
    assert diff < 1e-3, f"FAIL: target_pos in obs mismatch (diff {diff})"

    # Also verify env 1 (if exists) has same local target -> per-env replication works
    if args.num_envs > 1:
        target_env1 = obs_t[1, 22:25].cpu().numpy()
        diff1 = np.abs(target_env1 - expected).max()
        print(f"[sanity-stack] target_in_obs[1]: {target_env1}, diff: {diff1:.6f}")
        assert diff1 < 1e-3, f"FAIL: env 1 target_pos mismatch — per-env target broken"

    # Rollout with small random actions
    t_start = time.time()
    rewards_all = []
    truncated_count = 0
    terminated_count = 0
    last_log = {}
    for step in range(args.steps):
        action = (torch.rand((args.num_envs, 6), device=inner.device) - 0.5) * 0.4
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_all.append(reward.mean().item())
        truncated_count += truncated.sum().item()
        terminated_count += terminated.sum().item()
        if "log" in info:
            last_log = {k: (v.item() if hasattr(v, "item") else v) for k, v in info["log"].items()}
        if step % 50 == 0:
            d_t = last_log.get("tcp_sponge_dist_m", "?")
            d_st = last_log.get("sponge_target_dist_m", "?")
            sh = last_log.get("sponge_height_m", "?")
            ls = last_log.get("lift_success_rate", "?")
            ps = last_log.get("place_success_rate", "?")
            print(f"[sanity-stack] step {step:3d}: r={reward.mean().item():+.3f} "
                  f"d_tcp_sponge={d_t} d_sponge_target={d_st} h={sh} "
                  f"lift_s={ls} place_s={ps} trunc={truncated.sum().item()}")

    elapsed = time.time() - t_start
    sps = args.num_envs * args.steps / elapsed
    print(f"[sanity-stack] DONE: {args.steps} steps x {args.num_envs} envs in {elapsed:.2f}s = {sps:.0f} steps/s")
    print(f"[sanity-stack] reward avg: {np.mean(rewards_all):+.3f}")
    print(f"[sanity-stack] truncations: {truncated_count}, terminations: {terminated_count}")

    # Hard checks
    assert terminated_count == 0, f"FAIL: terminated should always be False (got {terminated_count})"
    assert truncated_count > 0, f"FAIL: at least 1 truncation expected after {args.steps} steps"
    assert np.all(np.isfinite(rewards_all)), "FAIL: NaN/Inf reward detected"

    print(f"[sanity-stack] PASS")
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
