"""sim_rollout_p6v12.py — P6v12 policy rollout (state-only) + trajectory plot.

Phase 1.B-α P6v12 (η fix) model_999.pt rollout on local 4090.
State-only: no Annotator/RTX camera (HARD RULE #17 compliant).
Physics-only Isaac Lab headless (enable_cameras=False).

Usage (conda env isaaclab):
  # 1. scp model from B200 first:
  #    scp JHPark:/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/logs/roarm_rl/p6v12_eta_stage2cap_stage3transient_resumeP6v11/model_999.pt /tmp/p6v12_model_999.pt
  #    scp -r JHPark:/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/assets/roarm_m3/usd/ /tmp/roarm_m3_usd/

  # 2. Set USD path env var:
  #    export ROARM_M3_USD_PATH=/tmp/roarm_m3_usd/roarm_m3.usd

  # 3. Run rollout:
  #    conda run -n isaaclab python sim_rollout_p6v12.py \
  #        --checkpoint /tmp/p6v12_model_999.pt \
  #        --num_envs 8 \
  #        --num_episodes 3 \
  #        --out_dir claudedocs/figures

Output files:
  claudedocs/figures/p6v12_trajectory_top.png    -- XY plane: TCP + sponge + target
  claudedocs/figures/p6v12_trajectory_side.png   -- XZ plane: height evolution
  claudedocs/figures/p6v12_failure_mode_snapshot.png -- Reward/gripper timeline
  claudedocs/figures/p6v12_rollout_stats.txt     -- Numeric summary for lab meeting
"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/tmp/p6v12_model_999.pt")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of parallel envs (4090 safe: 8-16)")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Episodes to record per env")
    parser.add_argument("--reward_phase", type=int, default=6)
    parser.add_argument("--out_dir", default="claudedocs/figures")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Isaac Lab launch (physics-only, HARD RULE #17 compliant) ──────────
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import numpy as np
    import gymnasium as gym
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    import roarm_rl  # registers envs
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, TARGET_L1_SPOT1, TABLE_Z
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    print(f"[rollout] checkpoint: {args.checkpoint}")
    print(f"[rollout] num_envs={args.num_envs} episodes={args.num_episodes} phase={args.reward_phase}")
    print(f"[rollout] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Env setup ─────────────────────────────────────────────────────────
    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_phase = args.reward_phase
    env_cfg.seed = args.seed
    # Reduce episode length for faster rollout (still enough to see failure)
    env_cfg.episode_length_s = 2.0   # 200 steps at 100Hz

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped

    max_ep_len = inner_env.max_episode_length
    print(f"[rollout] max_episode_length={max_ep_len}")

    # ── Load policy ───────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if hasattr(runner.alg, "policy"):
        target = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        target = runner.alg.actor_critic
    else:
        raise RuntimeError("No policy/actor_critic attr on runner.alg")
    ret = target.load_state_dict(sd, strict=False)
    print(f"[rollout] load_state_dict: {ret}")
    policy = runner.get_inference_policy(device=inner_env.device)
    print("[rollout] policy loaded OK")

    # ── Per-episode trajectory buffers ────────────────────────────────────
    # We record env=0 only (first env) for clean single-episode visualization
    TARGET = torch.tensor(TARGET_L1_SPOT1, device=inner_env.device)

    # Storage: lists per episode
    all_episodes = []   # list of dicts

    ep_buf: dict = None

    def start_episode():
        return {
            "tcp_x": [], "tcp_y": [], "tcp_z": [],
            "sponge_x": [], "sponge_y": [], "sponge_z": [],
            "gripper_cmd": [],   # action[5]
            "gripper_open_flag": [],  # inner_env._gripper_open[0]
            "grasped": [],
            "reward": [],
            "stage": [],         # 1-4 inferred from reward flags
            "sponge_to_target_dist": [],
        }

    def flush_episode(buf):
        if len(buf["tcp_x"]) > 5:
            all_episodes.append({k: np.array(v) for k, v in buf.items()})

    # Hook _reset_idx to detect episode boundaries for env 0
    orig_reset = inner_env._reset_idx
    in_warmup = [True]
    current_ep = [start_episode()]

    def hooked_reset(env_ids):
        nonlocal ep_buf
        if env_ids is not None and isinstance(env_ids, torch.Tensor):
            if 0 in env_ids and not in_warmup[0]:
                flush_episode(current_ep[0])
                current_ep[0] = start_episode()
        orig_reset(env_ids)

    inner_env._reset_idx = hooked_reset

    # Warmup: force first truncation
    inner_env.episode_length_buf[:] = max_ep_len
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    in_warmup[0] = False
    print("[rollout] warmup done — envs randomized")

    # ── Main rollout loop ─────────────────────────────────────────────────
    total_steps = args.num_episodes * max_ep_len
    print(f"[rollout] running {total_steps} steps ({args.num_episodes} ep × {max_ep_len} steps)...")

    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, info = env.step(actions)

        # Record env=0 state
        buf = current_ep[0]
        tcp = inner_env._tcp_pos_w[0].cpu()
        sp  = inner_env._sponge_pos_w[0].cpu()
        buf["tcp_x"].append(tcp[0].item())
        buf["tcp_y"].append(tcp[1].item())
        buf["tcp_z"].append(tcp[2].item())
        buf["sponge_x"].append(sp[0].item())
        buf["sponge_y"].append(sp[1].item())
        buf["sponge_z"].append(sp[2].item())
        buf["gripper_cmd"].append(actions[0, 5].item())
        buf["reward"].append(rew[0].item())

        # Gripper open flag (if available)
        if hasattr(inner_env, "_gripper_open"):
            buf["gripper_open_flag"].append(inner_env._gripper_open[0].item())
        else:
            buf["gripper_open_flag"].append(float(actions[0, 5].item() > 0))

        # Grasped flag
        if hasattr(inner_env, "_grasped"):
            buf["grasped"].append(inner_env._grasped[0].item())
        else:
            buf["grasped"].append(0.0)

        # Sponge-to-target distance
        dist = torch.norm(sp - TARGET.cpu()).item()
        buf["sponge_to_target_dist"].append(dist)

    flush_episode(current_ep[0])

    env.close()
    sim_app.close()

    n_ep = len(all_episodes)
    if n_ep == 0:
        print("[rollout] ERROR: no episodes recorded. Exiting.")
        sys.exit(1)
    print(f"[rollout] recorded {n_ep} episodes")

    # ── Plotting ──────────────────────────────────────────────────────────
    TARGET_XY = (TARGET_L1_SPOT1[0], TARGET_L1_SPOT1[1])
    TARGET_Z  = TARGET_L1_SPOT1[2]

    COLORS = plt.cm.tab10.colors

    # --- Figure 1: XY top-down trajectory ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("P6v12 η — XY Trajectory (Top View)\nTCP path + Sponge path per episode", fontsize=13)
    for i, ep in enumerate(all_episodes[:3]):
        c = COLORS[i]
        ax.plot(ep["tcp_x"], ep["tcp_y"], color=c, lw=1.2, alpha=0.8, label=f"Ep{i+1} TCP")
        ax.plot(ep["sponge_x"], ep["sponge_y"], color=c, lw=1.2, alpha=0.5, ls="--", label=f"Ep{i+1} Sponge")
        ax.scatter(ep["tcp_x"][0], ep["tcp_y"][0], color=c, marker="o", s=60, zorder=5)
        ax.scatter(ep["tcp_x"][-1], ep["tcp_y"][-1], color=c, marker="x", s=80, zorder=5)
    # Target marker
    ax.scatter(*TARGET_XY, color="red", marker="*", s=200, zorder=10, label="Target (L1.spot1)")
    ax.add_patch(mpatches.Circle(TARGET_XY, 0.05, fill=False, color="red", ls="--", alpha=0.5))
    ax.set_xlabel("X (m) [forward]")
    ax.set_ylabel("Y (m) [lateral]")
    ax.legend(fontsize=8, ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Annotate failure mode
    ax.text(0.02, 0.02, "Failure: sponge reaches zone but gripper stays CLOSED",
            transform=ax.transAxes, fontsize=8, color="darkred",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    out1 = os.path.join(args.out_dir, "p6v12_trajectory_top.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rollout] saved: {out1}")

    # --- Figure 2: Height (Z) and Dist-to-target over time ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle("P6v12 η — Side View: Z Height + Sponge-to-Target Distance", fontsize=12)
    for i, ep in enumerate(all_episodes[:3]):
        c = COLORS[i]
        T = np.arange(len(ep["tcp_z"]))
        axes[0].plot(T, ep["tcp_z"], color=c, lw=1.2, label=f"Ep{i+1} TCP-Z")
        axes[0].plot(T, ep["sponge_z"], color=c, lw=1.2, ls="--", alpha=0.6, label=f"Ep{i+1} Sponge-Z")
    axes[0].axhline(TARGET_Z, color="red", ls=":", lw=1.5, label=f"Target Z = {TARGET_Z*1000:.1f}mm")
    axes[0].axhline(TABLE_Z, color="gray", ls=":", lw=1, label=f"Table Z = {TABLE_Z*1000:.1f}mm")
    axes[0].set_ylabel("Z (m)")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)

    for i, ep in enumerate(all_episodes[:3]):
        c = COLORS[i]
        T = np.arange(len(ep["sponge_to_target_dist"]))
        axes[1].plot(T, np.array(ep["sponge_to_target_dist"])*1000, color=c, lw=1.2, label=f"Ep{i+1}")
    axes[1].axhline(50, color="red", ls="--", lw=1.5, label="50mm zone threshold")
    axes[1].set_ylabel("Sponge-to-Target dist (mm)")
    axes[1].set_xlabel("Step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    out2 = os.path.join(args.out_dir, "p6v12_trajectory_side.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rollout] saved: {out2}")

    # --- Figure 3: Failure mode snapshot — gripper + reward timeline ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "P6v12 η — Failure Mode: 'Grasps, reaches zone, never opens gripper'\n"
        "gripper_open_rate 6.4% FLAT | is_on_target 40.6% | stage4_success 0.02%",
        fontsize=11, color="darkred"
    )
    ep = all_episodes[0]
    T = np.arange(len(ep["reward"]))

    # Panel 1: Gripper command (action[5])
    axes[0].plot(T, ep["gripper_cmd"], color="steelblue", lw=1.0)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].fill_between(T, 0, ep["gripper_cmd"],
                          where=np.array(ep["gripper_cmd"]) > 0, alpha=0.3, color="green", label="open cmd")
    axes[0].fill_between(T, 0, ep["gripper_cmd"],
                          where=np.array(ep["gripper_cmd"]) < 0, alpha=0.3, color="red", label="close cmd")
    axes[0].set_ylabel("Gripper action cmd")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Gripper command (action[5]): positive=open, negative=close", fontsize=9)

    # Panel 2: Grasped flag
    axes[1].fill_between(T, 0, ep["grasped"], step="post", color="orange", alpha=0.7, label="grasped")
    axes[1].fill_between(T, 0, ep["gripper_open_flag"], step="post",
                          color="green", alpha=0.5, label="gripper_open")
    axes[1].set_ylabel("Boolean flag")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.1, 1.3)

    # Panel 3: Sponge-to-target distance
    d = np.array(ep["sponge_to_target_dist"]) * 1000
    axes[2].plot(T, d, color="purple", lw=1.2)
    axes[2].axhline(50, color="red", ls="--", lw=1.5, label="50mm zone")
    axes[2].fill_between(T, 0, 50, alpha=0.1, color="green", label="success zone")
    axes[2].set_ylabel("Dist to target (mm)")
    axes[2].set_xlabel("Step")
    axes[2].legend(fontsize=8)

    # Annotate key moments
    in_zone_steps = np.where(d < 50)[0]
    if len(in_zone_steps) > 0:
        first_in = in_zone_steps[0]
        axes[2].axvline(first_in, color="green", ls=":", lw=1.5)
        axes[2].text(first_in + 2, d[first_in] + 10,
                     f"First in zone\n(step {first_in})", fontsize=7, color="green")

    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "p6v12_failure_mode_snapshot.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rollout] saved: {out3}")

    # --- Stats summary ---
    stats_lines = [
        "P6v12 η Rollout Summary (Local 4090, state-only)",
        f"Checkpoint: {args.checkpoint}",
        f"Envs: {args.num_envs} | Episodes recorded: {n_ep}",
        "",
        "Per-episode stats (env=0):",
    ]
    for i, ep in enumerate(all_episodes):
        mean_dist = np.mean(ep["sponge_to_target_dist"]) * 1000
        min_dist  = np.min(ep["sponge_to_target_dist"]) * 1000
        mean_gr_open = np.mean(ep["gripper_open_flag"]) * 100
        mean_grasped = np.mean(ep["grasped"]) * 100
        in_zone_frac = np.mean(np.array(ep["sponge_to_target_dist"]) < 0.05) * 100
        stats_lines.append(
            f"  Ep{i+1}: dist mean={mean_dist:.1f}mm min={min_dist:.1f}mm "
            f"| in_zone={in_zone_frac:.1f}% | grasped={mean_grasped:.1f}% "
            f"| gripper_open={mean_gr_open:.1f}%"
        )

    stats_lines += [
        "",
        "Known B200 metrics (P6v12 model_999, 4096 envs):",
        "  gripper_open_rate: 6.4% (FLAT, failure mode)",
        "  is_on_target (strict): 40.6%",
        "  stage4_success: 0.02%",
        "  sponge_target_dist: ~0.079m (mean)",
        "  is_success_zone (50mm): 54.1%",
        "",
        "Failure diagnosis:",
        "  Policy grasps sponge reliably (grasped ~86%)",
        "  Sponge transported to zone (50mm) ~54% of time",
        "  Gripper stays CLOSED → stage4_success ~0%",
        "  Root cause: stage3 hover reward (stay closed, hold target)",
        "    dominates over stage4 release (open, +10 transient bonus)",
        "  η fix reduced stage2 cap (2.0) + added transient +10,",
        "    but 1-step close advantage persists → release never explored",
    ]

    out4 = os.path.join(args.out_dir, "p6v12_rollout_stats.txt")
    with open(out4, "w") as f:
        f.write("\n".join(stats_lines))
    print(f"[rollout] saved: {out4}")

    print("\n=== ROLLOUT COMPLETE ===")
    print(f"Figures: {args.out_dir}/")
    print("  p6v12_trajectory_top.png")
    print("  p6v12_trajectory_side.png")
    print("  p6v12_failure_mode_snapshot.png")
    print("  p6v12_rollout_stats.txt")


if __name__ == "__main__":
    main()
