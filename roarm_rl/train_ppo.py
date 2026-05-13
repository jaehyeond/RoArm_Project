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
    parser.add_argument("--reset_actor_bias_idx", type=int, default=None,
                        help="Zero out actor's LAST-layer bias at this output dimension after resume. "
                             "Phase 1.B-α P6 v5 (5/11): counters PPO entropy collapse on a single "
                             "action dim (e.g. gripper joint idx 5: actor.6.bias[5]=+0.84 at P6v4 "
                             "iter 999 → P(close)=74%/step → release never explored). Reset to 0 "
                             "restores 50/50 prior over open/close so PPO can re-learn signed dir.")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="Override env episode_length_s (Phase 1.B-α P6 v8 α, 5/14: 2.0 = 200 step "
                             "at 100Hz; default 4.0 = 400 step). Shorter episodes reduce stage 3 hover "
                             "incentive (cumulative reward 1976 → 988) and give stage 4 transition "
                             "relatively higher gain. ManiSkill StackCube uses 50 step convention.")
    # P6v14 (5/12) Curriculum CLI overrides (Option B: bootstrap stage-4 release signal).
    # Phase 0 example: --curriculum_spawn_min_r 0.08 --curriculum_spawn_max_r 0.15 \
    #                  --curriculum_xy_thresh 0.05 --curriculum_z_thresh 0.04 \
    #                  --curriculum_disable_nearzone_cap
    parser.add_argument("--curriculum_spawn_min_r", type=float, default=None,
                        help="Sponge spawn annulus min radius (m) around target_xy. "
                             "Phase 0 = 0.08 (avoid spawn-at-target trivial jackpot).")
    parser.add_argument("--curriculum_spawn_max_r", type=float, default=None,
                        help="Sponge spawn annulus max radius (m). 0 = legacy R1-R4. "
                             "Phase 0 = 0.15 / Phase 1 = 0.22 / Phase 2 = 0.30 (~full WS).")
    parser.add_argument("--curriculum_xy_thresh", type=float, default=None,
                        help="on_target_xy_thresh override (m). 0 = production (0.030). "
                             "Phase 0 = 0.05 (random π release-signal feasibility).")
    parser.add_argument("--curriculum_z_thresh", type=float, default=None,
                        help="on_target_z_thresh override (m). 0 = production (0.025). "
                             "Phase 0 = 0.04.")
    parser.add_argument("--curriculum_disable_nearzone_cap", action="store_true",
                        help="Disable stage 2 d<0.1 cap (P6v12 anti-hover fix). Required "
                             "for Phase 0 short-transport (curriculum spawn often d<0.1).")
    parser.add_argument("--curriculum_pregrasp", action="store_true",
                        help="P6v14a Phase 0a: pre-grasp init (Option α). Robot starts at IK "
                             "pose with TCP +5cm above target, gripper closed (q=0.8>0.4 thresh), "
                             "sponge attached + _grasped/_was_grasped latched True. Agent's only "
                             "task: open gripper → sponge falls 5cm → stage 4 fires. Bootstrap "
                             "signal guaranteed (release path 1525 vs hover 400, +281% margin "
                             "with near-zone cap KEPT).")
    parser.add_argument("--curriculum_pregrasp_hover", action="store_true",
                        help="P6v14c Phase 0a': pre-grasp HOVER. Robot at IK pose (TCP +5cm "
                             "above target) but gripper OPEN (q=0.0). Sponge spawned on table "
                             "near target via curriculum_spawn_max_r annulus (recommend 0.05-0.07). "
                             "_grasped=False, _was_grasped=False. Agent task: descend (5cm) → "
                             "close gripper → grasp → release at target. Bridges P6v14a release-"
                             "only ↔ P6v14b cold-start full chain. Pair with --curriculum_post_grasp_cap.")
    parser.add_argument("--curriculum_post_grasp_cap", action="store_true",
                        help="P6v14c Phase 0a': force stage 2 r = post_grasp_cap_value (default 3.0) "
                             "ALWAYS when is_grasped (any d). Overrides nearzone_cap. Kills P6v14b's "
                             "8th 'grasp + move away' farming (where agent moved sponge to d>0.1 to "
                             "earn 5/step). cap=3.0 > stage 1 max (2.0) → PPO grasp gradient preserved.")
    parser.add_argument("--curriculum_post_grasp_cap_value", type=float, default=None,
                        help="Override curriculum_post_grasp_cap value (default 3.0). Must >2.0.")
    # P6v16 Path B (5/14) — Residual Policy Learning (Silver 2018) for catastrophic
    # forgetting fix. BC base (frozen) + trainable residual MLP; PPO trains residual only.
    parser.add_argument("--residual_mode", action="store_true",
                        help="Enable Residual Policy Learning. Requires --residual_bc_ckpt. "
                             "actor = ResidualMLPWrapper(bc_actor_frozen, residual_mlp, alpha). "
                             "BC params requires_grad=False -> zero forgetting by construction.")
    parser.add_argument("--residual_bc_ckpt", type=str, default=None,
                        help="Path to BC actor checkpoint (.pt). Loaded as frozen base in residual mode.")
    parser.add_argument("--residual_alpha", type=float, default=0.3,
                        help="Residual scale (default 0.3). final = bc(x) + alpha*residual(x).")
    parser.add_argument("--residual_hidden", type=str, default="64,32",
                        help="Residual MLP hidden dims (comma-separated). Default '64,32'.")
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
    if args.episode_length_s is not None:
        print(f"[train] episode_length_s override: {env_cfg.episode_length_s} -> {args.episode_length_s}")
        env_cfg.episode_length_s = args.episode_length_s
    # P6v14 Curriculum CLI overrides
    if args.curriculum_spawn_min_r is not None:
        print(f"[train] curriculum_spawn_min_r: {env_cfg.curriculum_spawn_min_r} -> {args.curriculum_spawn_min_r}")
        env_cfg.curriculum_spawn_min_r = args.curriculum_spawn_min_r
    if args.curriculum_spawn_max_r is not None:
        print(f"[train] curriculum_spawn_max_r: {env_cfg.curriculum_spawn_max_r} -> {args.curriculum_spawn_max_r}")
        env_cfg.curriculum_spawn_max_r = args.curriculum_spawn_max_r
    if args.curriculum_xy_thresh is not None:
        print(f"[train] curriculum_xy_thresh: {env_cfg.curriculum_xy_thresh} -> {args.curriculum_xy_thresh}")
        env_cfg.curriculum_xy_thresh = args.curriculum_xy_thresh
    if args.curriculum_z_thresh is not None:
        print(f"[train] curriculum_z_thresh: {env_cfg.curriculum_z_thresh} -> {args.curriculum_z_thresh}")
        env_cfg.curriculum_z_thresh = args.curriculum_z_thresh
    if args.curriculum_disable_nearzone_cap:
        print(f"[train] curriculum_disable_nearzone_cap: True")
        env_cfg.curriculum_disable_nearzone_cap = True
    if args.curriculum_pregrasp:
        print(f"[train] curriculum_pregrasp: True  (pregrasp_joints_rad={env_cfg.pregrasp_joints_rad})")
        env_cfg.curriculum_pregrasp = True
    if args.curriculum_pregrasp_hover:
        print(f"[train] curriculum_pregrasp_hover: True  (TCP at pregrasp pose, gripper OVERRIDE to OPEN q=0.0)")
        env_cfg.curriculum_pregrasp_hover = True
    if args.curriculum_post_grasp_cap:
        cap_val = args.curriculum_post_grasp_cap_value if args.curriculum_post_grasp_cap_value is not None else env_cfg.curriculum_post_grasp_cap_value
        print(f"[train] curriculum_post_grasp_cap: True (cap value = {cap_val})")
        env_cfg.curriculum_post_grasp_cap = True
        if args.curriculum_post_grasp_cap_value is not None:
            env_cfg.curriculum_post_grasp_cap_value = args.curriculum_post_grasp_cap_value
    # Mutual exclusion guard: pregrasp and pregrasp_hover are mutually exclusive (different env init).
    if args.curriculum_pregrasp and args.curriculum_pregrasp_hover:
        raise ValueError("--curriculum_pregrasp and --curriculum_pregrasp_hover are mutually exclusive")

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

    # Residual mode setup (Path B P6v16, 5/14): replace actor with frozen-BC + residual.
    # Must happen BEFORE --resume logic (resume + residual_mode are mutually exclusive).
    if args.residual_mode:
        if args.residual_bc_ckpt is None:
            raise ValueError("--residual_mode requires --residual_bc_ckpt")
        if args.resume is not None:
            raise ValueError("--residual_mode and --resume are mutually exclusive (BC is the base)")
        from roarm_rl.policies.residual_actor import install_residual_actor
        bc_state = torch.load(args.residual_bc_ckpt, map_location="cpu", weights_only=False)
        bc_sd = bc_state["model_state_dict"] if isinstance(bc_state, dict) and "model_state_dict" in bc_state else bc_state
        target_pre = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        residual_hidden = tuple(int(x) for x in args.residual_hidden.split(","))
        install_residual_actor(target_pre, bc_sd,
                               alpha=args.residual_alpha,
                               residual_hidden=residual_hidden)
        # Rebuild optimizer to include only trainable params (residual + critic + std).
        # rsl_rl's algorithm holds optimizer; rebuild with .parameters() (BC has requires_grad=False
        # so torch optim auto-skips, but we re-create cleanly).
        import torch.optim as optim
        trainable = [p for p in target_pre.parameters() if p.requires_grad]
        bc_count = sum(1 for p in target_pre.parameters() if not p.requires_grad)
        print(f"[residual] trainable_params={sum(p.numel() for p in trainable)} bc_frozen_modules={bc_count}")
        # Reuse existing optimizer config from rsl_rl algo
        old_opt = runner.alg.optimizer
        old_lr = old_opt.param_groups[0]["lr"]
        runner.alg.optimizer = optim.Adam(trainable, lr=old_lr)
        print(f"[residual] optimizer rebuilt: Adam lr={old_lr} over {len(trainable)} trainable tensors")

    if args.resume:
        print(f"[train] resume from (model only, fresh optimizer): {args.resume}")
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

        # Force-reset policy std BEFORE load (Phase 1.B-α P6 v2 std-divergence fix)
        if args.reset_std is not None and "std" in sd:
            old_std = sd["std"].clone()
            sd["std"] = torch.full_like(sd["std"], args.reset_std)
            print(f"[train] reset_std: ckpt std {old_std.tolist()} -> {sd['std'].tolist()}")

        # Reset actor's LAST-layer bias at one output dim (Phase 1.B-α P6 v5 entropy-collapse fix)
        if args.reset_actor_bias_idx is not None:
            actor_bias_keys = sorted(
                [k for k in sd.keys() if k.startswith("actor.") and k.endswith(".bias")]
            )
            if not actor_bias_keys:
                raise RuntimeError("--reset_actor_bias_idx: no actor.*.bias found in state_dict")
            last_bias_key = actor_bias_keys[-1]  # e.g. actor.6.bias for [256,128,64,6] MLP
            bias_vec = sd[last_bias_key]
            idx = args.reset_actor_bias_idx
            if not (0 <= idx < bias_vec.shape[0]):
                raise RuntimeError(
                    f"--reset_actor_bias_idx={idx} out of range for {last_bias_key} "
                    f"shape={tuple(bias_vec.shape)}"
                )
            old_val = bias_vec[idx].item()
            sd[last_bias_key][idx] = 0.0
            print(
                f"[train] reset_actor_bias: {last_bias_key}[{idx}]: "
                f"{old_val:+.4f} -> 0.0 (other dims kept: {bias_vec.tolist()})"
            )

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

        # Verify actor bias was reset post-load
        if args.reset_actor_bias_idx is not None:
            # Find the matching nn.Linear in the loaded policy and check its bias.
            last_linear = None
            for m in target.actor.modules() if hasattr(target, "actor") else target.modules():
                if isinstance(m, torch.nn.Linear):
                    last_linear = m
            if last_linear is not None and last_linear.bias is not None:
                print(f"[train] post-load actor last-bias = {last_linear.bias.data.tolist()}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print(f"[train] DONE. checkpoints at: {log_dir}")
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
