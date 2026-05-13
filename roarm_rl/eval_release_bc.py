"""Path D Phase D.3 v1 — Eval release_bc alone vs P6v14a baseline.

Same env config as demo gen (curriculum_pregrasp): TCP +5cm above target, sponge
attached, gripper closed. For this init, state-machine handoff trigger is satisfied
at t=0 → state-machine reduces to BC alone. True state-machine value emerges only
for non-pregrasp init (later phase).

Metrics:
  - stage4_success_rate = fraction of envs where _place_success_flag ever fires
  - success_step distribution (gripper-open clean fires only — filtered post-hoc)
  - gripper_open_rate (per-step avg, across episode)

Run on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.eval_release_bc \
      --bc_ckpt $ROARM_B200_ROOT/data/release_bc.pt --num_envs 256
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum_xy_thresh", type=float, default=0.05)
    parser.add_argument("--curriculum_z_thresh", type=float, default=0.04)
    parser.add_argument("--episode_length_s", type=float, default=2.0)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save eval metrics .pt for later analysis.")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import torch.nn as nn
    import gymnasium as gym
    import roarm_rl  # noqa: F401
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = 6
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.curriculum_pregrasp = True
    env_cfg.curriculum_xy_thresh = args.curriculum_xy_thresh
    env_cfg.curriculum_z_thresh = args.curriculum_z_thresh

    print(f"[eval_bc] bc_ckpt        : {args.bc_ckpt}")
    print(f"[eval_bc] num_envs       : {args.num_envs}")
    print(f"[eval_bc] curriculum_xy/z: {args.curriculum_xy_thresh}/{args.curriculum_z_thresh}")

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    T_max = int(inner_env.max_episode_length)
    device = inner_env.device
    print(f"[eval_bc] max_episode_length: {T_max}  device: {device}")

    ckpt = torch.load(args.bc_ckpt, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden = ckpt["hidden"]
    print(f"[eval_bc] BC arch: Linear({obs_dim}→{hidden}) ELU Linear({hidden}→{act_dim}) Tanh")
    print(f"[eval_bc] BC best_val_loss: {ckpt['best_val_loss']:.5f} @ epoch {ckpt['best_epoch']}")

    model = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ELU(),
        nn.Linear(hidden, act_dim),
        nn.Tanh(),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def _obs_tensor(o):
        if isinstance(o, torch.Tensor):
            return o
        return o["policy"]

    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = model(_obs_tensor(obs))
    obs, _, _, _ = env.step(actions)
    print(f"[eval_bc] warmup truncation fired.")

    B = args.num_envs
    grasp_thresh_rad = inner_env.cfg.grasp_gripper_thresh
    CLEAN_GRIPPER_THRESH_RAD = 0.4  # user-specified clean filter threshold

    all_succ_step, all_gripper_open_rate, all_gq_at_s = [], [], []

    for ep in range(args.num_episodes):
        success_step = torch.full((B,), -1, dtype=torch.long, device=device)
        # NaN sentinel = never fired. Captured at the exact step `_place_success_flag` rises.
        gripper_q_at_success = torch.full((B,), float("nan"), device=device)
        gripper_open_cumcount = torch.zeros(B, device=device)

        for t in range(T_max):
            with torch.inference_mode():
                actions = model(_obs_tensor(obs))
            obs, _, _, _ = env.step(actions)
            fired = inner_env._place_success_flag & (success_step == -1)
            success_step = torch.where(fired, torch.full_like(success_step, t), success_step)
            gripper_q = inner_env._robot.data.joint_pos[:, inner_env.gripper_joint_idx]
            gripper_q_at_success = torch.where(fired, gripper_q, gripper_q_at_success)
            gripper_open_cumcount += (gripper_q < grasp_thresh_rad).float()

        succ_mask = (success_step >= 0)
        n_succ = int(succ_mask.sum().item())
        succ_step_mean = (success_step[succ_mask].float().mean().item()
                          if n_succ > 0 else float("nan"))

        # Direct-path "clean" filter: gripper actually open at the success step.
        gq_succ = gripper_q_at_success[succ_mask]
        clean_mask = succ_mask & (gripper_q_at_success < CLEAN_GRIPPER_THRESH_RAD)
        n_clean = int(clean_mask.sum().item())

        gor_per_env = (gripper_open_cumcount / T_max).cpu()
        print(f"[eval_bc] ep={ep}  success={n_succ}/{B} ({100*n_succ/B:.2f}%)  "
              f"clean_at_s(<{CLEAN_GRIPPER_THRESH_RAD:.2f}rad)={n_clean}/{n_succ} "
              f"({100*n_clean/max(n_succ,1):.1f}%)  "
              f"mean_succ_step={succ_step_mean:.1f}  "
              f"gq_at_s mean={gq_succ.mean().item() if n_succ>0 else float('nan'):.3f}  "
              f"gripper_open_rate_mean={gor_per_env.mean().item():.3f}")

        all_succ_step.append(success_step.cpu().clone())
        all_gripper_open_rate.append(gor_per_env)
        all_gq_at_s.append(gripper_q_at_success.cpu().clone())

    succ_all = torch.cat(all_succ_step, dim=0)
    gor_all = torch.cat(all_gripper_open_rate, dim=0)
    gq_at_s_all = torch.cat(all_gq_at_s, dim=0)
    n_succ_total = int((succ_all >= 0).sum().item())
    n_total = succ_all.shape[0]

    succ_mask_all = (succ_all >= 0)
    clean_mask_all = succ_mask_all & (gq_at_s_all < CLEAN_GRIPPER_THRESH_RAD)
    n_clean_total = int(clean_mask_all.sum().item())

    print()
    print("=== EVAL SUMMARY ===")
    print(f"trials              : {n_total}")
    print(f"nominal successes   : {n_succ_total}/{n_total} ({100*n_succ_total/n_total:.2f}%)")
    print(f"CLEAN successes     : {n_clean_total}/{n_total} ({100*n_clean_total/n_total:.2f}%)  "
          f"[gripper_q@s < {CLEAN_GRIPPER_THRESH_RAD:.2f} rad]")
    if n_succ_total > 0:
        ss = succ_all[succ_mask_all]
        print(f"success_step        : mean={ss.float().mean().item():.1f}  "
              f"median={int(ss.median().item())}  "
              f"min={int(ss.min().item())}  max={int(ss.max().item())}")
        gq_succ_all = gq_at_s_all[succ_mask_all]
        print(f"gripper_q@s (succ)  : mean={gq_succ_all.mean().item():.3f}  "
              f"median={gq_succ_all.median().item():.3f}  "
              f"min={gq_succ_all.min().item():.3f}  max={gq_succ_all.max().item():.3f}  "
              f"(grasp_thresh={grasp_thresh_rad:.3f} rad)")
    print(f"gripper_open_rate   : mean={gor_all.mean().item():.3f}  "
          f"median={gor_all.median().item():.3f}  "
          f"std={gor_all.std().item():.3f}")
    print()
    print("=== BASELINE COMPARE (curriculum_pregrasp init) ===")
    print(f"P6v14a alone (prior measure): 20/256 = 7.81%")
    print(f"release_bc nominal (this)   : {n_succ_total}/{n_total} = {100*n_succ_total/n_total:.2f}%")
    print(f"release_bc CLEAN (this)     : {n_clean_total}/{n_total} = {100*n_clean_total/n_total:.2f}%")
    print(f"Δ nominal vs baseline       : {100*(n_succ_total/n_total - 0.0781):+.2f} pp")
    print(f"Δ CLEAN  vs baseline        : {100*(n_clean_total/n_total - 0.0781):+.2f} pp")
    print()
    print("=== PASS GATE (CLEAN rate, design doc Path D.3) ===")
    rate = n_clean_total / n_total
    if rate >= 0.50:
        print(f"≥50% CLEAN — publishable result")
    elif rate >= 0.30:
        print(f"≥30% CLEAN — proceed to subskill expansion")
    elif rate >= 0.10:
        print(f"10-30% CLEAN — BC capacity 확장 또는 demo source 개선 (procedural)")
    else:
        print(f"<10% CLEAN — PATH D FAIL → SkillGen/MimicGen procedural release pivot")

    if args.output:
        torch.save({
            "success_step": succ_all,
            "gripper_q_at_success": gq_at_s_all,
            "gripper_open_rate": gor_all,
            "n_total": n_total,
            "n_succ": n_succ_total,
            "n_clean": n_clean_total,
            "clean_thresh_rad": CLEAN_GRIPPER_THRESH_RAD,
            "grasp_thresh_rad": grasp_thresh_rad,
            "bc_ckpt": args.bc_ckpt,
        }, args.output)
        print(f"saved metrics to    : {args.output}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
