"""PPO training for RoArm Pick env (rsl_rl 3.1.2).

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.train_ppo --num_envs 4096 --max_iterations 500 --reward_phase 1
"""
import argparse
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pick", choices=["pick", "stack"])
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--reward_phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from (e.g. logs/.../model_100.pt)")
    parser.add_argument("--reset_std", type=float, default=None,
                        help="Force-overwrite policy std after resume (e.g. 1.5). "
                             "Phase 1.B-α P6 v2: counters monotonic std divergence "
                             "(P3=2.7 -> P6=7.4) caused by weak reward signal vs entropy push.")
    parser.add_argument("--entropy_coef", type=float, default=None,
                        help="Override PPO entropy_coef (default 0.005). "
                             "Lower (0.001) suppresses log_std positive gradient -> std stops diverging.")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # registers env
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    if args.task == "pick":
        from roarm_rl.roarm_pick_env import RoArmPickEnvCfg
        env_cfg = RoArmPickEnvCfg()
        env_id = "RoArm-Pick-Direct-v0"
        assert args.reward_phase in (1, 2, 3), \
            f"task=pick supports reward_phase 1/2/3 (got {args.reward_phase})"
    else:  # stack
        from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
        env_cfg = RoArmStackEnvCfg()
        env_id = "RoArm-Stack-Direct-v0"
        assert args.reward_phase in (4, 5, 6), \
            f"task=stack supports reward_phase 4/5/6 (got {args.reward_phase})"

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = args.reward_phase
    env_cfg.seed = args.seed

    # ppo cfg
    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.max_iterations = args.max_iterations
    ppo_cfg.seed = args.seed
    if args.entropy_coef is not None:
        print(f"[train] entropy_coef override: {ppo_cfg.algorithm.entropy_coef} -> {args.entropy_coef}")
        ppo_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.experiment_name:
        ppo_cfg.experiment_name = args.experiment_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ppo_cfg.experiment_name = f"roarm_{args.task}_p{args.reward_phase}_{ts}"

    # log dir
    if args.logdir:
        log_root_path = args.logdir
    else:
        log_root_path = os.path.join(
            os.environ.get("ROARM_B200_ROOT", os.getcwd()),
            "logs", "roarm_rl",
        )
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    log_dir = os.path.join(log_root_path, ppo_cfg.experiment_name)

    print(f"[train] env: num_envs={args.num_envs} reward_phase={args.reward_phase}")
    print(f"[train] ppo: max_iter={args.max_iterations} steps_per_env={ppo_cfg.num_steps_per_env}")
    print(f"[train] log_dir: {log_dir}")

    # create env
    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # runner
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)

    if args.resume:
        print(f"[train] resume from (model only, fresh optimizer): {args.resume}")
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

        # Force-reset policy std BEFORE load (Phase 1.B-α P6 v2 std-divergence fix)
        if args.reset_std is not None and "std" in sd:
            old_std = sd["std"].clone()
            sd["std"] = torch.full_like(sd["std"], args.reset_std)
            print(f"[train] reset_std: ckpt std {old_std.tolist()} -> {sd['std'].tolist()}")

        if hasattr(runner.alg, "policy"):
            target = runner.alg.policy
        elif hasattr(runner.alg, "actor_critic"):
            target = runner.alg.actor_critic
        else:
            raise RuntimeError(f"runner.alg has no policy/actor_critic attr")
        ret = target.load_state_dict(sd, strict=False)
        if isinstance(ret, tuple):
            print(f"[train] resume missing={list(ret[0])} unexpected={list(ret[1])}")
        else:
            print(f"[train] resume load_state_dict returned: {ret}")

        # Verify std was reset post-load
        if args.reset_std is not None and hasattr(target, "std"):
            print(f"[train] post-load policy.std = {target.std.data.tolist()}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print(f"[train] DONE. checkpoints at: {log_dir}")
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
