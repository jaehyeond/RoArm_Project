"""B200 smoke for P7 structured near-target release curriculum.

This is a policy-free mechanics probe.  It verifies that the default-off
structured release curriculum reset is active, then applies only a gripper-open
command while holding the arm still.  The goal is to falsify the branch before
long PPO if near-target identity+keep release cannot settle upright.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--hold_steps", type=int, default=5)
    parser.add_argument("--open_steps", type=int, default=12)
    parser.add_argument("--settle_steps", type=int, default=60)
    parser.add_argument("--xy_jitter", type=float, default=0.0)
    parser.add_argument("--z_jitter", type=float, default=0.0)
    parser.add_argument("--attach_quat_mode", choices=("preserve", "identity"), default="identity")
    parser.add_argument("--attach_velocity_mode", choices=("zero", "keep"), default="keep")
    parser.add_argument("--upright_thresh", type=float, default=0.90)
    parser.add_argument("--release_xy_kill", type=float, default=0.060)
    parser.add_argument("--final_sz_kill", type=float, default=0.90)
    parser.add_argument("--final_success_rate_kill", type=float, default=0.90)

    from isaaclab.app import AppLauncher

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
    cfg.episode_length_s = max(1.0, (args.hold_steps + args.open_steps + args.settle_steps + 20) / 100.0)
    cfg.curriculum_attached_transport_release = True
    cfg.curriculum_attached_start_jitter_rad = 0.0
    cfg.p7_structured_release_curriculum = True
    cfg.p7_structured_release_xy_jitter = args.xy_jitter
    cfg.p7_structured_release_z_jitter = args.z_jitter
    cfg.attach_quat_mode = args.attach_quat_mode
    cfg.attach_velocity_mode = args.attach_velocity_mode

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    env.reset()

    all_ids = torch.arange(args.num_envs, device=base_env.device, dtype=torch.long)
    gripper_idx = int(base_env.gripper_joint_idx)

    def step(actions):
        out = env.step(actions)
        if len(out) == 5:
            return out[0], out[1], out[2] | out[3], out[4]
        return out

    def snapshot():
        base_env._compute_intermediate_values(all_ids)
        origins = base_env.scene.env_origins[all_ids]
        tcp = base_env._tcp_pos_w[all_ids] - origins
        sponge = base_env._sponge_pos_w[all_ids] - origins
        target = base_env._target_world[all_ids] - origins
        quat = base_env._sponge_quat_w[all_ids]
        qx = quat[:, 1]
        qy = quat[:, 2]
        sz = 1.0 - 2.0 * (qx * qx + qy * qy)
        d_xy = torch.norm(sponge[:, :2] - target[:, :2], dim=-1)
        rel_z_signed = (target[:, 2] + 0.029) - sponge[:, 2]
        rel_z_abs = torch.abs(rel_z_signed)
        settled_z_abs = torch.abs(target[:, 2] - sponge[:, 2])
        d_tcp = torch.norm(sponge - tcp, dim=-1)
        gripper_q = base_env._robot.data.joint_pos[all_ids, gripper_idx]
        lin_vel = base_env._sponge.data.root_lin_vel_w[all_ids]
        vel = torch.norm(lin_vel, dim=-1)
        open_mask = gripper_q < base_env.cfg.grasp_gripper_thresh
        upright = sz > args.upright_thresh
        success = (d_xy < 0.060) & (settled_z_abs < 0.030) & upright & open_mask & (~base_env._grasped[all_ids])
        return {
            "tcp": tcp.detach().clone(),
            "sponge": sponge.detach().clone(),
            "target": target.detach().clone(),
            "sz": sz.detach().clone(),
            "d_xy": d_xy.detach().clone(),
            "rel_z_abs": rel_z_abs.detach().clone(),
            "rel_z_signed": rel_z_signed.detach().clone(),
            "settled_z_abs": settled_z_abs.detach().clone(),
            "d_tcp": d_tcp.detach().clone(),
            "gripper_q": gripper_q.detach().clone(),
            "open": open_mask.detach().clone(),
            "grasped": base_env._grasped[all_ids].detach().clone(),
            "vel": vel.detach().clone(),
            "success": success.detach().clone(),
        }

    def mean(vals, mask=None):
        if mask is None:
            return float(vals.float().mean().item())
        if int(mask.sum().item()) == 0:
            return float("nan")
        return float(vals[mask].float().mean().item())

    print("[p7_structured_probe] structured release curriculum smoke", flush=True)
    print(
        f"[p7_structured_probe] attach_quat_mode={base_env.cfg.attach_quat_mode} "
        f"attach_velocity_mode={base_env.cfg.attach_velocity_mode} "
        f"structured_release={base_env.cfg.p7_structured_release_curriculum} "
        f"num_envs={args.num_envs} xy_jitter={args.xy_jitter:.4f} z_jitter={args.z_jitter:.4f}",
        flush=True,
    )

    actions = torch.zeros((args.num_envs, base_env.cfg.action_space), device=base_env.device)
    reset = snapshot()
    print(
        f"[p7_structured_probe] reset d_xy={mean(reset['d_xy']):.4f} "
        f"rel_z_abs={mean(reset['rel_z_abs']):.4f} sz={mean(reset['sz']):.4f} "
        f"d_tcp={mean(reset['d_tcp']):.5f} grasped={mean(reset['grasped'].float()):.3f} "
        f"open={mean(reset['open'].float()):.3f}",
        flush=True,
    )

    prev = reset
    max_tcp_delta = torch.zeros(args.num_envs, device=base_env.device)
    first_open = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    release_step = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_tip_grasped = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    release_sz = torch.full((args.num_envs,), float("nan"), device=base_env.device)
    release_d_xy = torch.full((args.num_envs,), float("nan"), device=base_env.device)

    total_steps = args.hold_steps + args.open_steps + args.settle_steps
    for step_idx in range(1, total_steps + 1):
        actions.zero_()
        if step_idx > args.hold_steps and step_idx <= args.hold_steps + args.open_steps:
            actions[:, gripper_idx] = -1.0
        step(actions)
        snap = snapshot()

        tcp_delta = torch.norm(snap["tcp"] - prev["tcp"], dim=-1)
        max_tcp_delta = torch.maximum(max_tcp_delta, tcp_delta)
        open_now = snap["open"] & (first_open < 0)
        released_now = ((~snap["grasped"]) | snap["open"]) & (release_step < 0)
        tipped_grasped = (snap["sz"] < args.upright_thresh) & snap["grasped"] & (first_tip_grasped < 0)
        first_open[open_now] = step_idx
        release_step[released_now] = step_idx
        first_tip_grasped[tipped_grasped] = step_idx
        release_sz[released_now] = snap["sz"][released_now]
        release_d_xy[released_now] = snap["d_xy"][released_now]
        prev = snap

        if step_idx in (args.hold_steps, args.hold_steps + 1, args.hold_steps + args.open_steps):
            print(
                f"[p7_structured_probe] step={step_idx:03d} d_xy={mean(snap['d_xy']):.4f} "
                f"rel_z_abs={mean(snap['rel_z_abs']):.4f} sz={mean(snap['sz']):.4f} "
                f"gripper_q={mean(snap['gripper_q']):+.4f} open={mean(snap['open'].float()):.3f} "
                f"grasped={mean(snap['grasped'].float()):.3f}",
                flush=True,
            )

    final = snapshot()
    release_mask = release_step >= 0
    open_mask = first_open >= 0
    tip_grasped_mask = first_tip_grasped >= 0

    print("[p7_structured_probe] aggregate_transition_counts", flush=True)
    print(f"[p7_structured_probe]   first_open={int(open_mask.sum().item())}/{args.num_envs}", flush=True)
    print(f"[p7_structured_probe]   release_or_open={int(release_mask.sum().item())}/{args.num_envs}", flush=True)
    print(
        f"[p7_structured_probe]   tip_while_grasped_before_release={int(tip_grasped_mask.sum().item())}/{args.num_envs}",
        flush=True,
    )
    print("[p7_structured_probe] aggregate_pose_means", flush=True)
    print(
        f"[p7_structured_probe]   release sz={mean(release_sz, release_mask):.4f} "
        f"d_xy={mean(release_d_xy, release_mask):.4f}",
        flush=True,
    )
    print(
        f"[p7_structured_probe]   final d_xy={mean(final['d_xy']):.4f} "
        f"rel_z_abs={mean(final['rel_z_abs']):.4f} "
        f"settled_z_abs={mean(final['settled_z_abs']):.4f} "
        f"sz={mean(final['sz']):.4f} success_rate={mean(final['success'].float()):.4f} "
        f"max_tcp_delta_mean={mean(max_tcp_delta):.4f} max_tcp_delta_max={float(max_tcp_delta.max().item()):.4f}",
        flush=True,
    )

    mechanism_active = (
        base_env.cfg.p7_structured_release_curriculum
        and base_env.cfg.attach_quat_mode == "identity"
        and base_env.cfg.attach_velocity_mode == "keep"
        and mean(reset["d_xy"]) <= 0.040
        and mean(reset["rel_z_abs"]) <= 0.020
        and mean(reset["sz"]) >= args.upright_thresh
        and mean(reset["grasped"].float()) >= 0.999
    )
    kill = (
        mean(release_d_xy, release_mask) > args.release_xy_kill
        or mean(final["sz"]) < args.final_sz_kill
        or mean(final["success"].float()) < args.final_success_rate_kill
    )
    print(f"[p7_structured_probe] MECHANISM_ACTIVE={'YES' if mechanism_active else 'NO'}", flush=True)
    print(f"[p7_structured_probe] EARLY_KILL={'YES' if kill else 'NO'}", flush=True)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
