"""Extract P6v12 policy trajectory (state-only, no rendering).

eval_policy.py 패턴 차용. enable_cameras=False로 안정성/속도 확보.
1 episode (200 step) 굴리고 매 frame state를 CSV로 저장.

Output CSV columns:
    t, joint_deg_0..5, sponge_x, sponge_y, sponge_z, grasped, tcp_x, tcp_y, tcp_z

Run:
    conda run -n isaaclab python -u -m scripts.extract_p6v12_trajectory \
        --checkpoint local_ckpts/p6v12_model_999.pt \
        --out claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"[extract] checkpoint: {args.checkpoint}", flush=True)
    print(f"[extract] out       : {out_path}", flush=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import numpy as np
    import gymnasium as gym
    import roarm_rl  # noqa: F401  registers env
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.reward_phase = 6
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 2.0
    # Override B200-hardcoded USD path -> local copy (visual mesh 깨졌어도 articulation은 정상)
    LOCAL_USD = "/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd/roarm_m3.usd"
    if Path(LOCAL_USD).exists():
        env_cfg.robot.spawn.usd_path = LOCAL_USD
        print(f"[extract] USD override -> {LOCAL_USD}", flush=True)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    print(f"[extract] max_episode_length={inner_env.max_episode_length}", flush=True)

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=inner_env.device)
    print(f"[extract] policy loaded", flush=True)

    # Force first-step truncation → clean random spawn (eval_policy.py 패턴)
    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()

    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    print(f"[extract] warmup truncation fired", flush=True)

    total_steps = inner_env.max_episode_length
    rows = []
    for t in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)

        joint_rad = inner_env._robot.data.joint_pos[0].detach().cpu().numpy()
        joint_deg = np.degrees(joint_rad)
        sponge_w = inner_env._sponge_pos_w[0].detach().cpu().numpy()
        tcp_w = inner_env._tcp_pos_w[0].detach().cpu().numpy()
        grasped = bool(inner_env._grasped[0].item())

        row = [t] + joint_deg.tolist() + sponge_w.tolist() + [int(grasped)] + tcp_w.tolist()
        rows.append(row)

        if t < 3 or t % 25 == 0 or t == total_steps - 1:
            print(f"  t={t:3d}  joint_deg={[f'{x:+.1f}' for x in joint_deg]}  "
                  f"sponge=({sponge_w[0]*1000:+.0f},{sponge_w[1]*1000:+.0f},{sponge_w[2]*1000:+.0f})mm  "
                  f"grasped={grasped}", flush=True)

    print(f"[extract] saving {len(rows)} rows -> {out_path}", flush=True)
    header = "t,j0_deg,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,sponge_x,sponge_y,sponge_z,grasped,tcp_x,tcp_y,tcp_z"
    np.savetxt(out_path, np.asarray(rows), delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.6f"]*6 + ["%.6f"]*3 + ["%d"] + ["%.6f"]*3)
    print(f"[extract] CSV saved.", flush=True)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
