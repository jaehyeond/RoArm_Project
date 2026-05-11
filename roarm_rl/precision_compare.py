"""Paired precision comparison of P3 checkpoints.

Runs 4 seed levels × N ckpts in a single Isaac Sim session.
For each (ckpt, seed): reseed torch RNG → reset all envs → 1 episode rollout
→ record per-env (success, dist, height, grasped).

Pairing: same seed → same spawn distribution across ckpts (env RNG is reseeded
per (ckpt, seed) so spawns are deterministic and shared).

Output: per-ckpt aggregate + per-seed paired diff (1100 - 1050, new - 1100, etc.)
        + 2-sample / paired t-stat for success rate.

Usage on B200:
  source $ROARM_B200_ROOT/env.sh
  micromamba activate $ROARM_B200_ROOT/envs/isaacsim_5_1
  export OMNI_KIT_ACCEPT_EULA=YES
  python -m roarm_rl.precision_compare \\
    --num_envs 4096 \\
    --reward_phase 3 \\
    --seeds 42 43 44 45 \\
    --ckpts /path/A.pt /path/B.pt /path/C.pt \\
    --labels A B C \\
    --out_json /tmp/precision_compare.json
"""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--reward_phase", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--out_json", type=str, default="/tmp/precision_compare.json")
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.labels), "ckpts and labels length must match"

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import math
    import numpy as np
    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    from roarm_rl.roarm_pick_env import RoArmPickEnvCfg, TABLE_Z
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = RoArmPickEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = args.reward_phase
    env_cfg.seed = 0  # base seed; reseeded per (ckpt, seed) below

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = 0

    print(f"[compare] num_envs={args.num_envs} seeds={args.seeds}")
    print(f"[compare] ckpts:")
    for label, ckpt in zip(args.labels, args.ckpts):
        print(f"    {label}: {ckpt}")

    # Single env / single Sim launch.
    env = gym.make("RoArm-Pick-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    max_ep_len = inner_env.max_episode_length
    print(f"[compare] max_episode_length={max_ep_len}")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)

    # Hook _reset_idx to capture per-env stats AT episode-end (before reset clears flags).
    # This is the same trick as eval_policy.py.
    captured: list[dict] = []  # each entry = {"label": ..., "seed": ..., success/dist/h/grasped tensors}
    current_label = [None]
    current_seed = [None]
    capture_enabled = [False]

    orig_reset = inner_env._reset_idx

    def hooked_reset(env_ids):
        if capture_enabled[0] and env_ids is not None:
            if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
                ids = env_ids
                success = inner_env._success_flag[ids].detach().cpu().clone()
                d = torch.norm(
                    inner_env._sponge_pos_w[ids] - inner_env._tcp_pos_w[ids],
                    p=2, dim=-1,
                ).detach().cpu().clone()
                h = (inner_env._sponge_pos_w[ids, 2] - TABLE_Z).detach().cpu().clone()
                grasped = inner_env._grasped[ids].detach().cpu().clone()
                captured.append({
                    "label": current_label[0],
                    "seed": current_seed[0],
                    "success": success.numpy(),
                    "dist_m": d.numpy(),
                    "height_m": h.numpy(),
                    "grasped": grasped.numpy(),
                })
        orig_reset(env_ids)

    inner_env._reset_idx = hooked_reset

    # Helper: run one episode of rollout for given (ckpt_path, seed).
    def run_one(ckpt_path: str, label: str, seed: int):
        # Re-seed all RNGs the env uses for spawn sampling.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Force a clean episode boundary on next step — set buf to max so truncation fires.
        # But disable capture for this WARMUP truncation (random pre-eval state).
        capture_enabled[0] = False
        inner_env.episode_length_buf[:] = max_ep_len

        # Load ckpt now (can be re-loaded between runs).
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=inner_env.device)

        obs = env.get_observations()
        # First step: truncates everywhere → re-spawn under the new seed. NOT logged.
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Now enable capture and run exactly one full episode (max_ep_len steps).
        current_label[0] = label
        current_seed[0] = seed
        capture_enabled[0] = True
        for _ in range(max_ep_len):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
        capture_enabled[0] = False

    # Initial obs (no env.reset() to avoid Isaac Lab joint_acc bug).
    inner_env.episode_length_buf[:] = max_ep_len  # force first-step truncate
    _ = env.get_observations()

    # Run all (ckpt × seed) combinations.
    for ckpt_path, label in zip(args.ckpts, args.labels):
        for seed in args.seeds:
            print(f"[compare] running label={label} seed={seed} ...")
            run_one(ckpt_path, label, seed)
            # Quick per-run summary
            last = captured[-1]
            print(f"    n={len(last['success'])}  success={last['success'].mean()*100:.2f}%  "
                  f"mean_dist={last['dist_m'].mean()*1000:.2f}mm  "
                  f"mean_h={last['height_m'].mean()*1000:.2f}mm  "
                  f"grasped={last['grasped'].mean()*100:.2f}%")

    # ============================================
    # Aggregate
    # ============================================
    print()
    print("=" * 70)
    print("AGGREGATE — per-ckpt summary (across all seeds)")
    print("=" * 70)

    per_ckpt_summary = {}
    for label in args.labels:
        per_seed = [c for c in captured if c["label"] == label]
        all_success = np.concatenate([c["success"] for c in per_seed])
        all_dist = np.concatenate([c["dist_m"] for c in per_seed])
        all_h = np.concatenate([c["height_m"] for c in per_seed])
        all_grasped = np.concatenate([c["grasped"] for c in per_seed])
        n = len(all_success)
        sr = all_success.mean()
        se = float(np.sqrt(sr * (1 - sr) / n))
        per_ckpt_summary[label] = {
            "n": int(n),
            "success_rate": float(sr),
            "se": se,
            "mean_dist_mm": float(all_dist.mean() * 1000),
            "mean_h_mm": float(all_h.mean() * 1000),
            "grasped_pct": float(all_grasped.mean() * 100),
            "per_seed_success": [float(c["success"].mean()) for c in per_seed],
            "seeds": [int(c["seed"]) for c in per_seed],
        }
        print(f"  [{label}] n={n}  success={sr*100:.3f}% (SE={se*100:.3f}%)  "
              f"dist={all_dist.mean()*1000:.2f}mm  h={all_h.mean()*1000:.2f}mm  "
              f"grasp={all_grasped.mean()*100:.2f}%")
        print(f"      per-seed success: {[f'{v*100:.2f}%' for v in per_ckpt_summary[label]['per_seed_success']]}")

    # ============================================
    # Paired comparisons (success rate)
    # ============================================
    print()
    print("=" * 70)
    print("PAIRED PER-SEED DIFFS (success rate B - A, paired t on per-seed means)")
    print("=" * 70)

    pair_results = []
    for i, label_a in enumerate(args.labels):
        for label_b in args.labels[i + 1:]:
            sr_a = np.array(per_ckpt_summary[label_a]["per_seed_success"])
            sr_b = np.array(per_ckpt_summary[label_b]["per_seed_success"])
            assert len(sr_a) == len(sr_b), "seed counts must match"
            diff = sr_b - sr_a
            mean_diff = diff.mean()
            sd_diff = diff.std(ddof=1) if len(diff) > 1 else float("nan")
            se_diff = sd_diff / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
            t_stat = mean_diff / se_diff if se_diff > 0 else float("nan")
            pair_results.append({
                "from": label_a, "to": label_b,
                "mean_diff_pp": float(mean_diff * 100),
                "sd_diff_pp": float(sd_diff * 100),
                "se_diff_pp": float(se_diff * 100),
                "t_stat": float(t_stat),
                "per_seed_diff_pp": [float(v * 100) for v in diff],
            })
            print(f"  {label_b} - {label_a}:  mean_diff={mean_diff*100:+.3f}pp  "
                  f"SD={sd_diff*100:.3f}pp  SE={se_diff*100:.3f}pp  t={t_stat:.2f}  "
                  f"per-seed: {[f'{v*100:+.2f}' for v in diff]}")

    # ============================================
    # Save
    # ============================================
    out = {
        "args": vars(args),
        "per_ckpt": per_ckpt_summary,
        "pair_results": pair_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[compare] saved -> {args.out_json}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
