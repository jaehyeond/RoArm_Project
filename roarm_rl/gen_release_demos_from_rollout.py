"""Path D Phase D.1 — Generate release demos from P6v14a rollout.

Strategy:
  - Init each env via curriculum_pregrasp (TCP +5cm above target, sponge attached,
    gripper closed q=0.8). Matches P6v14a's training distribution exactly.
  - Roll out P6v14a deterministic policy for `--num_episodes` full episodes per env.
  - Capture (obs, action) per step in [T_max, B, dim] buffers.
  - Track per-env `success_step` = first step where `_place_success_flag` fires.
  - Save trajectories for envs with success_step >= 0 (success_flag latches True).

Avoids env.reset() (Isaac Lab joint_acc inference-tensor bug after runner.load).
Uses warmup-truncation pattern from eval_policy.py.

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.gen_release_demos_from_rollout \
      --checkpoint $ROARM_B200_ROOT/logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt \
      --num_envs 256 --num_episodes 1 \
      --output $ROARM_B200_ROOT/data/release_demos_v1.pt
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Episodes per env. Total demos ~ num_envs * num_episodes * success_rate.")
    parser.add_argument("--output", type=str, default="release_demos_v1.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum_xy_thresh", type=float, default=0.05,
                        help="Match P6v14a training (0.05).")
    parser.add_argument("--curriculum_z_thresh", type=float, default=0.04,
                        help="Match P6v14a training (0.04).")
    parser.add_argument("--episode_length_s", type=float, default=2.0,
                        help="Match P6v14a training (2.0s).")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = 6
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.curriculum_pregrasp = True
    env_cfg.curriculum_xy_thresh = args.curriculum_xy_thresh
    env_cfg.curriculum_z_thresh = args.curriculum_z_thresh

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print(f"[demo_gen] checkpoint        : {args.checkpoint}")
    print(f"[demo_gen] num_envs          : {args.num_envs}")
    print(f"[demo_gen] num_episodes      : {args.num_episodes}")
    print(f"[demo_gen] curriculum_pregrasp: True")
    print(f"[demo_gen] curriculum_xy/z   : {args.curriculum_xy_thresh}/{args.curriculum_z_thresh}")
    print(f"[demo_gen] episode_length_s  : {args.episode_length_s}")
    print(f"[demo_gen] output            : {args.output}")

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    T_max = int(inner_env.max_episode_length)
    print(f"[demo_gen] max_episode_length: {T_max}")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=inner_env.device)

    def _obs_tensor(o):
        # rsl_rl >=3.x returns a TensorDict from env.step() / env.get_observations().
        # Policy accepts TensorDict; for buffer storage we need the flat tensor under
        # the 'policy' key (28-dim observation).
        if isinstance(o, torch.Tensor):
            return o
        return o["policy"]

    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, _, _ = env.step(actions)
    obs_t = _obs_tensor(obs)
    print(f"[demo_gen] warmup truncation fired. obs_t={tuple(obs_t.shape)} act={tuple(actions.shape)}")

    B = args.num_envs
    obs_dim = obs_t.shape[-1]
    act_dim = actions.shape[-1]
    device = inner_env.device

    all_obs_list, all_act_list, all_succ_step = [], [], []
    total_trials = 0

    for ep in range(args.num_episodes):
        obs_buf = torch.zeros(T_max, B, obs_dim, device=device)
        act_buf = torch.zeros(T_max, B, act_dim, device=device)
        success_step = torch.full((B,), -1, dtype=torch.long, device=device)

        for t in range(T_max):
            obs_buf[t] = _obs_tensor(obs)
            with torch.inference_mode():
                actions = policy(obs)
            act_buf[t] = actions
            obs, _, _, _ = env.step(actions)
            fired = inner_env._place_success_flag & (success_step == -1)
            success_step = torch.where(fired, torch.full_like(success_step, t), success_step)

        succ_mask = (success_step >= 0)
        n_succ = int(succ_mask.sum().item())
        mean_step = (success_step[succ_mask].float().mean().item()
                     if n_succ > 0 else float("nan"))
        print(f"[demo_gen] ep={ep}  success={n_succ}/{B} ({100*n_succ/B:.2f}%)  "
              f"mean_step={mean_step:.1f}")

        if n_succ > 0:
            ids = succ_mask.nonzero(as_tuple=False).flatten()
            all_obs_list.append(obs_buf[:, ids].permute(1, 0, 2).cpu().clone())
            all_act_list.append(act_buf[:, ids].permute(1, 0, 2).cpu().clone())
            all_succ_step.append(success_step[ids].cpu().clone())
        total_trials += B

    n_demos = sum(t.shape[0] for t in all_obs_list)
    if n_demos == 0:
        print(f"[demo_gen] WARNING: 0 demos from {total_trials} trials. Exiting without save.")
        env.close()
        sim_app.close()
        return

    obs_tensor = torch.cat(all_obs_list, dim=0)
    act_tensor = torch.cat(all_act_list, dim=0)
    succ_tensor = torch.cat(all_succ_step, dim=0)

    meta = {
        "checkpoint": args.checkpoint,
        "num_envs": args.num_envs,
        "num_episodes": args.num_episodes,
        "total_trials": total_trials,
        "success_rate": n_demos / total_trials,
        "max_episode_length": T_max,
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "curriculum_pregrasp": True,
        "curriculum_xy_thresh": args.curriculum_xy_thresh,
        "curriculum_z_thresh": args.curriculum_z_thresh,
        "episode_length_s": args.episode_length_s,
        "reward_phase": 6,
        "seed": args.seed,
    }
    torch.save(
        {"obs": obs_tensor, "action": act_tensor, "success_step": succ_tensor, "meta": meta},
        args.output,
    )

    print()
    print(f"=== DEMO GEN SUMMARY ===")
    print(f"trials       : {total_trials}")
    print(f"demos        : {n_demos} ({100*n_demos/total_trials:.2f}%)")
    print(f"obs.shape    : {tuple(obs_tensor.shape)}")
    print(f"action.shape : {tuple(act_tensor.shape)}")
    print(f"success_step : mean={succ_tensor.float().mean().item():.1f}  "
          f"median={int(succ_tensor.median().item())}  "
          f"min={int(succ_tensor.min().item())}  "
          f"max={int(succ_tensor.max().item())}")
    print(f"saved to     : {args.output}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
