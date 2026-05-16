"""State-only probe for env-level attached orientation semantics.

This verifies the gated config path in ``RoArmStackEnv`` without loading a
policy or monkey-patching ``_update_grasp_attach``.  It starts from the P7
attached curriculum, injects a tipped sponge quaternion while ``_grasped=True``,
calls the real env attach update, and checks whether the configured semantics
preserve or reset the attached orientation.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--quat_mode", choices=("preserve", "identity"), default="identity")
    parser.add_argument("--velocity_mode", choices=("zero", "keep"), default="keep")
    parser.add_argument("--tilt_deg", type=float, default=60.0)
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
    cfg.attach_quat_mode = args.quat_mode
    cfg.attach_velocity_mode = args.velocity_mode

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    env.reset()
    base_env._compute_intermediate_values()

    env_ids = torch.arange(args.num_envs, device=base_env.device, dtype=torch.long)
    before_pos = base_env._sponge.data.root_pos_w[env_ids].clone()
    tcp_before = base_env._tcp_pos_w[env_ids].clone()
    initial_d_tcp = torch.norm(before_pos - tcp_before, dim=-1)

    tilt = math.radians(args.tilt_deg)
    tipped_quat = torch.zeros((args.num_envs, 4), device=base_env.device)
    tipped_quat[:, 0] = math.cos(tilt / 2.0)
    tipped_quat[:, 1] = math.sin(tilt / 2.0)

    pose7 = torch.zeros((args.num_envs, 7), device=base_env.device)
    pose7[:, 0:3] = before_pos
    pose7[:, 3:7] = tipped_quat
    base_env._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)

    vel6 = torch.zeros((args.num_envs, 6), device=base_env.device)
    vel6[:, 0] = 0.11
    vel6[:, 5] = 3.0
    base_env._sponge.write_root_velocity_to_sim(vel6, env_ids=env_ids)

    base_env._update_grasp_attach()
    base_env._compute_intermediate_values()

    quat = base_env._sponge.data.root_quat_w[env_ids]
    qx = quat[:, 1]
    qy = quat[:, 2]
    sz = 1.0 - 2.0 * (qx * qx + qy * qy)
    d_tcp = torch.norm(base_env._sponge.data.root_pos_w[env_ids] - base_env._tcp_pos_w[env_ids], dim=-1)
    lin_vel = base_env._sponge.data.root_lin_vel_w[env_ids]
    ang_vel = base_env._sponge.data.root_ang_vel_w[env_ids]
    vel_norm = torch.norm(torch.cat((lin_vel, ang_vel), dim=-1), dim=-1)

    print("[attach_semantics_probe] env-level attach semantics probe", flush=True)
    print(
        f"[attach_semantics_probe] attach_quat_mode={base_env.cfg.attach_quat_mode} "
        f"attach_velocity_mode={base_env.cfg.attach_velocity_mode} "
        f"num_envs={args.num_envs} tilt_deg={args.tilt_deg:.1f}",
        flush=True,
    )
    print(
        f"[attach_semantics_probe] reset grasped_frac={base_env._grasped.float().mean().item():.3f} "
        f"initial_d_tcp_mean={initial_d_tcp.mean().item():.6f}",
        flush=True,
    )
    print(
        f"[attach_semantics_probe] after_attach sz_mean={sz.mean().item():.4f} "
        f"sz_min={sz.min().item():.4f} d_tcp_mean={d_tcp.mean().item():.6f} "
        f"vel_norm_mean={vel_norm.mean().item():.4f} vel_norm_max={vel_norm.max().item():.4f}",
        flush=True,
    )
    for i in range(min(args.num_envs, 4)):
        print(
            f"[attach_semantics_probe] env{i:02d} quat="
            f"({quat[i,0].item():+.4f},{quat[i,1].item():+.4f},"
            f"{quat[i,2].item():+.4f},{quat[i,3].item():+.4f}) "
            f"sz={sz[i].item():+.4f} d_tcp={d_tcp[i].item():.6f} "
            f"vel_norm={vel_norm[i].item():.4f}",
            flush=True,
        )

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
