"""Eval frozen rsl_rl policy on RoArm Pick env.

Strategy:
  - Avoid `env.reset()` (Isaac Lab joint_acc inference-tensor bug after `runner.load`).
  - Hook `_reset_idx` to capture per-env stats AT episode-end before reset clears flags.
  - Force a "warmup" episode-end on first step (so spawns are properly randomized) by
    setting episode_length_buf to max-1 before the first step.

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.eval_policy \
      --checkpoint $ROARM_B200_ROOT/logs/roarm_rl/roarm_pick_p1_500iter_seed0/model_499.pt \
      --reward_phase 1 --num_envs 256 --num_rollouts 4
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_rollouts", type=int, default=4,
                        help="Number of episodes per env (after warmup).")
    parser.add_argument("--reward_phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import numpy as np
    import roarm_rl  # noqa: F401  registers env
    from roarm_rl.roarm_pick_env import RoArmPickEnvCfg, TABLE_Z
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = RoArmPickEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = args.reward_phase
    env_cfg.seed = args.seed

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print(f"[eval] checkpoint: {args.checkpoint}")
    print(f"[eval] num_envs={args.num_envs} num_rollouts={args.num_rollouts} reward_phase={args.reward_phase}")

    env = gym.make("RoArm-Pick-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    print(f"[eval] max_episode_length={inner_env.max_episode_length}")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=inner_env.device)

    # Hook _reset_idx to capture per-env stats at episode-end.
    trial_records: list[tuple] = []
    warmup_done = [False]

    orig_reset = inner_env._reset_idx

    def hooked_reset(env_ids):
        if warmup_done[0] and env_ids is not None:
            if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
                ids = env_ids
                success = inner_env._success_flag[ids].detach().cpu().clone()
                d = torch.norm(
                    inner_env._sponge_pos_w[ids] - inner_env._tcp_pos_w[ids],
                    p=2, dim=-1,
                ).detach().cpu().clone()
                h = (inner_env._sponge_pos_w[ids, 2] - TABLE_Z).detach().cpu().clone()
                grasped = inner_env._grasped[ids].detach().cpu().clone()
                trial_records.append((success, d, h, grasped))
        orig_reset(env_ids)

    inner_env._reset_idx = hooked_reset

    # Force first-step truncation for all envs → clean random spawn before warmup_done flips.
    inner_env.episode_length_buf[:] = inner_env.max_episode_length

    # Initial obs (without env.reset). Compute via wrapper.
    obs = env.get_observations()

    # First step: trunc fires for all envs, warmup reset (NOT logged).
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    warmup_done[0] = True
    print("[eval] warmup truncation fired — all envs randomized.")

    # Now run num_rollouts full episodes per env. Reset hook logs each completed episode.
    total_steps = args.num_rollouts * inner_env.max_episode_length
    print(f"[eval] running {total_steps} steps for {args.num_rollouts} episodes/env...")
    for t in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)

    # Aggregate trial records (each entry = stats for one batch of envs that reset together)
    if not trial_records:
        print("[eval] WARNING: no trial records captured (no resets fired).")
        env.close()
        sim_app.close()
        return

    success = torch.cat([r[0] for r in trial_records]).numpy()
    dist = torch.cat([r[1] for r in trial_records]).numpy()
    height = torch.cat([r[2] for r in trial_records]).numpy()
    grasped = torch.cat([r[3] for r in trial_records]).numpy()

    print()
    print(f"=== EVAL SUMMARY (n={len(success)} trials) ===")
    print(f"checkpoint        : {args.checkpoint}")
    print(f"reward_phase      : {args.reward_phase}")
    print(f"success_rate      : {success.mean()*100:.2f}%   "
          f"(threshold: sponge_h>{inner_env.cfg.success_height*1000:.0f}mm "
          f"× {inner_env.cfg.success_steps_required} consec steps)")
    print(f"final TCP-sponge  : mean={dist.mean()*1000:.2f}mm  std={dist.std()*1000:.2f}  "
          f"min={dist.min()*1000:.2f}  max={dist.max()*1000:.2f}")
    print(f"final sponge_h    : mean={height.mean()*1000:.2f}mm std={height.std()*1000:.2f}  "
          f"max={height.max()*1000:.2f}")
    print(f"grasped@reset     : {grasped.mean()*100:.2f}%")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
