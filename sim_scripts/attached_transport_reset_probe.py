"""Probe P6v17 attached transport/release curriculum reset.

This is a short sanity diagnostic, not a training job. It verifies that the
G2-A seed0 attached-start curriculum initializes:

- _grasped=True and _was_grasped=True
- sponge at the attached TCP handoff table
- target sampled from the intended L1 spot table
- non-trivial source-to-target distance before policy action
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False

    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = 7
    cfg.episode_length_s = 0.2
    cfg.curriculum_attached_transport_release = True
    cfg.curriculum_attached_start_jitter_rad = 0.0

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    obs, _ = env.reset()
    base_env._compute_intermediate_values()

    env_origins = base_env.scene.env_origins
    sponge_local = base_env._sponge_pos_w - env_origins
    tcp_local = base_env._tcp_pos_w - env_origins
    target_local = base_env._target_world - env_origins
    d_sponge_tcp = torch.norm(sponge_local - tcp_local, dim=-1)
    d_sponge_target_xy = torch.norm(sponge_local[:, :2] - target_local[:, :2], dim=-1)
    d_sponge_target_z = torch.abs(sponge_local[:, 2] - target_local[:, 2])

    print("[attached_probe] reset complete", flush=True)
    print(f"[attached_probe] obs_shape={tuple(obs['policy'].shape)}", flush=True)
    print(f"[attached_probe] grasped_frac={base_env._grasped.float().mean().item():.3f}", flush=True)
    print(f"[attached_probe] was_grasped_frac={base_env._was_grasped.float().mean().item():.3f}", flush=True)
    print(f"[attached_probe] d_sponge_tcp_mean_mm={d_sponge_tcp.mean().item() * 1000.0:.2f}", flush=True)
    print(f"[attached_probe] d_xy_mean_mm={d_sponge_target_xy.mean().item() * 1000.0:.2f}", flush=True)
    print(f"[attached_probe] d_z_mean_mm={d_sponge_target_z.mean().item() * 1000.0:.2f}", flush=True)
    for i in range(min(args.num_envs, 8)):
        print(
            "[attached_probe] env"
            f"{i:02d} sponge=({sponge_local[i,0].item():+.4f},"
            f"{sponge_local[i,1].item():+.4f},{sponge_local[i,2].item():+.4f}) "
            f"target=({target_local[i,0].item():+.4f},"
            f"{target_local[i,1].item():+.4f},{target_local[i,2].item():+.4f}) "
            f"d_xy={d_sponge_target_xy[i].item()*1000.0:.1f}mm",
            flush=True,
        )

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
