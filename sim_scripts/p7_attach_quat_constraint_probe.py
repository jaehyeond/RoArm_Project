"""Probe attached-transport behavior under diagnostic quaternion constraints.

This script monkey-patches `_update_grasp_attach` at runtime only. It does not
edit the environment, reward, training code, chain skills, or asset files.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace_envs", type=int, default=4)
    parser.add_argument("--attached_start_jitter_rad", type=float, default=0.0)
    parser.add_argument("--upright_thresh", type=float, default=0.90)
    parser.add_argument("--large_tcp_jump_m", type=float, default=0.030)
    parser.add_argument(
        "--quat_mode",
        choices=("preserve", "identity", "reset"),
        default="identity",
        help="Attached quaternion diagnostic mode.",
    )
    parser.add_argument(
        "--velocity_mode",
        choices=("zero", "keep"),
        default="zero",
        help="Whether patched attach zeroes sponge velocity each attached step.",
    )

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
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = 7
    cfg.seed = args.seed
    cfg.curriculum_attached_transport_release = True
    cfg.curriculum_attached_start_jitter_rad = args.attached_start_jitter_rad

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print("[p7_attach_probe] runtime attach quaternion constraint probe", flush=True)
    print(f"[p7_attach_probe] checkpoint={args.checkpoint}", flush=True)
    print(
        f"[p7_attach_probe] num_envs={args.num_envs} max_steps={args.max_steps} "
        f"trace_envs={args.trace_envs} seed={args.seed}",
        flush=True,
    )
    print(
        f"[p7_attach_probe] quat_mode={args.quat_mode} velocity_mode={args.velocity_mode} "
        f"attached_start_jitter_rad={args.attached_start_jitter_rad}",
        flush=True,
    )

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    base_env = env.unwrapped

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=base_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    dt = float(getattr(base_env, "step_dt", 0.01))
    all_ids = torch.arange(args.num_envs, device=base_env.device, dtype=torch.long)
    trace_ids = torch.arange(min(args.trace_envs, args.num_envs), device=base_env.device, dtype=torch.long)
    gripper_idx = int(base_env.gripper_joint_idx)
    attach_ref_quat = torch.zeros((args.num_envs, 4), device=base_env.device)
    attach_ref_quat[:, 0] = 1.0

    print(f"[p7_attach_probe] max_episode_length={base_env.max_episode_length}", flush=True)
    print(f"[p7_attach_probe] step_dt={dt:.4f}", flush=True)
    print(f"[p7_attach_probe] grasp_gripper_thresh={base_env.cfg.grasp_gripper_thresh:.4f}rad", flush=True)
    print(
        "[p7_attach_probe] original semantics roarm_stack_env.py:1096-1110 "
        "preserve current quat and zero velocity",
        flush=True,
    )

    def patched_update_grasp_attach(self):
        env_ids = torch.where(self._grasped)[0]
        if len(env_ids) == 0:
            return
        link5_pos = self._robot.data.body_pos_w[env_ids, self.link5_idx]
        link5_quat = self._robot.data.body_quat_w[env_ids, self.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, self._tcp_local.expand(link5_pos.shape[0], 3))
        tcp_pos = link5_pos + tcp_offset_world

        pose7 = torch.zeros((len(env_ids), 7), device=self.device)
        pose7[:, 0:3] = tcp_pos
        if args.quat_mode == "preserve":
            pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]
        elif args.quat_mode == "identity":
            pose7[:, 3] = 1.0
        elif args.quat_mode == "reset":
            pose7[:, 3:7] = attach_ref_quat[env_ids]
        else:
            raise RuntimeError(args.quat_mode)
        self._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        if args.velocity_mode == "zero":
            zeros = torch.zeros((len(env_ids), 6), device=self.device)
            self._sponge.write_root_velocity_to_sim(zeros, env_ids=env_ids)

    if args.quat_mode != "preserve" or args.velocity_mode != "zero":
        base_env._update_grasp_attach = types.MethodType(patched_update_grasp_attach, base_env)
        print("[p7_attach_probe] PATCHED _update_grasp_attach at runtime only", flush=True)
    else:
        print("[p7_attach_probe] using original _update_grasp_attach", flush=True)

    def snapshot(ids: torch.Tensor) -> dict:
        base_env._compute_intermediate_values(ids)
        origins = base_env.scene.env_origins[ids]
        tcp = base_env._tcp_pos_w[ids] - origins
        sponge = base_env._sponge_pos_w[ids] - origins
        target = base_env._target_world[ids] - origins
        quat = base_env._sponge_quat_w[ids]
        qx = quat[:, 1]
        qy = quat[:, 2]
        sz = 1.0 - 2.0 * (qx * qx + qy * qy)
        d_xy = torch.norm(sponge[:, :2] - target[:, :2], dim=-1)
        rel_z_signed = (target[:, 2] + 0.029) - sponge[:, 2]
        rel_z_abs = torch.abs(rel_z_signed)
        settled_z_abs = torch.abs(target[:, 2] - sponge[:, 2])
        d_sponge_tcp = torch.norm(sponge - tcp, dim=-1)
        gripper_q = base_env._robot.data.joint_pos[ids, gripper_idx]
        lin_vel = base_env._sponge.data.root_lin_vel_w[ids]
        ang_vel = base_env._sponge.data.root_ang_vel_w[ids]
        return {
            "tcp": tcp.detach().clone(),
            "sponge": sponge.detach().clone(),
            "quat": quat.detach().clone(),
            "sz": sz.detach().clone(),
            "d_xy": d_xy.detach().clone(),
            "rel_z_abs": rel_z_abs.detach().clone(),
            "rel_z_signed": rel_z_signed.detach().clone(),
            "settled_z_abs": settled_z_abs.detach().clone(),
            "d_sponge_tcp": d_sponge_tcp.detach().clone(),
            "gripper_q": gripper_q.detach().clone(),
            "grasped": base_env._grasped[ids].detach().clone(),
            "was_grasped": base_env._was_grasped[ids].detach().clone(),
            "lin_vel": lin_vel.detach().clone(),
            "ang_vel": ang_vel.detach().clone(),
        }

    def mean_valid(vals: torch.Tensor, mask: torch.Tensor) -> float:
        if int(mask.sum().item()) == 0:
            return float("nan")
        return float(vals[mask].float().mean().item())

    base_env.episode_length_buf[:] = base_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs).clamp(-1.0, 1.0)
        obs, _, _, _ = env.step(actions)

    prev = snapshot(all_ids)
    attach_ref_quat[:] = prev["quat"]
    print("[p7_attach_probe] warmup reset complete", flush=True)
    print(
        f"[p7_attach_probe] reset_mean d_xy={float(prev['d_xy'].mean()):.4f} "
        f"sz={float(prev['sz'].mean()):.4f} d_sponge_tcp={float(prev['d_sponge_tcp'].mean()):.5f} "
        f"grasped={float(prev['grasped'].float().mean()):.3f}",
        flush=True,
    )

    first_open = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_grasp_false = torch.full_like(first_open, -1)
    first_tip_any = torch.full_like(first_open, -1)
    first_tip_grasped = torch.full_like(first_open, -1)
    first_large_tcp_jump = torch.full_like(first_open, -1)
    release_step = torch.full_like(first_open, -1)
    release_sz = torch.full((args.num_envs,), float("nan"), device=base_env.device)
    release_d_xy = torch.full_like(release_sz, float("nan"))
    release_rel_z_abs = torch.full_like(release_sz, float("nan"))
    max_tcp_delta = torch.zeros(args.num_envs, device=base_env.device)
    max_abs_action = torch.zeros(args.num_envs, device=base_env.device)
    max_ang_vel_grasped = torch.zeros(args.num_envs, device=base_env.device)

    for step in range(1, args.max_steps + 1):
        with torch.inference_mode():
            actions = policy(obs).clamp(-1.0, 1.0)
            obs, _, _, _ = env.step(actions)

        snap = snapshot(all_ids)
        tcp_delta = torch.norm(snap["tcp"] - prev["tcp"], dim=-1)
        max_tcp_delta = torch.maximum(max_tcp_delta, tcp_delta)
        max_abs_action = torch.maximum(max_abs_action, torch.max(torch.abs(actions.detach()), dim=-1).values)
        ang_norm = torch.norm(snap["ang_vel"], dim=-1)
        max_ang_vel_grasped = torch.maximum(max_ang_vel_grasped, torch.where(snap["grasped"], ang_norm, max_ang_vel_grasped))

        gripper_open = snap["gripper_q"] < base_env.cfg.grasp_gripper_thresh
        release_now = (~snap["grasped"]) | gripper_open
        large_jump = tcp_delta > args.large_tcp_jump_m
        tip_any = snap["sz"] < args.upright_thresh
        tip_grasped = tip_any & snap["grasped"]

        first_open[(first_open < 0) & gripper_open] = step
        first_grasp_false[(first_grasp_false < 0) & (~snap["grasped"])] = step
        first_tip_any[(first_tip_any < 0) & tip_any] = step
        first_tip_grasped[(first_tip_grasped < 0) & tip_grasped] = step
        first_large_tcp_jump[(first_large_tcp_jump < 0) & large_jump] = step
        newly_release = (release_step < 0) & release_now
        release_step[newly_release] = step
        release_sz[newly_release] = snap["sz"][newly_release]
        release_d_xy[newly_release] = snap["d_xy"][newly_release]
        release_rel_z_abs[newly_release] = snap["rel_z_abs"][newly_release]

        if step <= 5 or step in (10, 15, 20, 25, 30):
            trace = snapshot(trace_ids)
            actions_cpu = actions.detach().cpu()
            for j, env_id in enumerate(trace_ids.detach().cpu().tolist()):
                action = actions_cpu[env_id]
                print(
                    f"[p7_attach_probe] env={env_id:03d} step={step:03d} "
                    f"act0_5=({','.join(f'{float(x):+.3f}' for x in action.tolist())}) "
                    f"grip_q={float(trace['gripper_q'][j]):+.4f} "
                    f"open={int(float(trace['gripper_q'][j]) < base_env.cfg.grasp_gripper_thresh)} "
                    f"grasped={int(bool(trace['grasped'][j]))} "
                    f"tcp=({','.join(f'{float(x):+.4f}' for x in trace['tcp'][j].tolist())}) "
                    f"sponge=({','.join(f'{float(x):+.4f}' for x in trace['sponge'][j].tolist())}) "
                    f"quat=({','.join(f'{float(x):+.4f}' for x in trace['quat'][j].tolist())}) "
                    f"sz={float(trace['sz'][j]):+.4f} d_xy={float(trace['d_xy'][j]):.4f} "
                    f"rel_z_abs={float(trace['rel_z_abs'][j]):.4f} "
                    f"d_sponge_tcp={float(trace['d_sponge_tcp'][j]):.4f} "
                    f"ang_vel_norm={float(torch.norm(trace['ang_vel'][j])):.4f}",
                    flush=True,
                )

        prev = snap
        if torch.all(release_step >= 0) and step >= int(torch.max(release_step).item()) + 5:
            print(f"[p7_attach_probe] early_stop step={step} all envs released/opened plus 5 steps", flush=True)
            break

    final = snapshot(all_ids)
    open_mask = first_open >= 0
    grasp_false_mask = first_grasp_false >= 0
    release_mask = release_step >= 0
    tip_any_mask = first_tip_any >= 0
    tip_grasped_mask = first_tip_grasped >= 0
    large_jump_mask = first_large_tcp_jump >= 0
    tip_before_open = tip_any_mask & ((first_open < 0) | (first_tip_any <= first_open))
    tip_grasped_before_release = tip_grasped_mask & ((release_step < 0) | (first_tip_grasped <= release_step))

    print("[p7_attach_probe] aggregate_transition_counts", flush=True)
    print(f"[p7_attach_probe]   first_open={int(open_mask.sum())}/{args.num_envs}", flush=True)
    print(f"[p7_attach_probe]   first_grasp_false={int(grasp_false_mask.sum())}/{args.num_envs}", flush=True)
    print(f"[p7_attach_probe]   release_or_open={int(release_mask.sum())}/{args.num_envs}", flush=True)
    print(f"[p7_attach_probe]   first_tip_any={int(tip_any_mask.sum())}/{args.num_envs}", flush=True)
    print(f"[p7_attach_probe]   first_tip_while_grasped={int(tip_grasped_mask.sum())}/{args.num_envs}", flush=True)
    print(f"[p7_attach_probe]   tip_before_or_at_open={int(tip_before_open.sum())}/{args.num_envs}", flush=True)
    print(
        f"[p7_attach_probe]   tip_while_grasped_before_or_at_release="
        f"{int(tip_grasped_before_release.sum())}/{args.num_envs}",
        flush=True,
    )
    print(
        f"[p7_attach_probe]   first_large_tcp_jump>{args.large_tcp_jump_m:.3f}m="
        f"{int(large_jump_mask.sum())}/{args.num_envs}",
        flush=True,
    )

    print("[p7_attach_probe] aggregate_transition_steps", flush=True)
    print(f"[p7_attach_probe]   mean_first_open={mean_valid(first_open, open_mask):.2f}", flush=True)
    print(f"[p7_attach_probe]   mean_first_grasp_false={mean_valid(first_grasp_false, grasp_false_mask):.2f}", flush=True)
    print(f"[p7_attach_probe]   mean_release_or_open={mean_valid(release_step, release_mask):.2f}", flush=True)
    print(f"[p7_attach_probe]   mean_first_tip_any={mean_valid(first_tip_any, tip_any_mask):.2f}", flush=True)
    print(f"[p7_attach_probe]   mean_first_tip_while_grasped={mean_valid(first_tip_grasped, tip_grasped_mask):.2f}", flush=True)
    print(f"[p7_attach_probe]   mean_first_large_tcp_jump={mean_valid(first_large_tcp_jump, large_jump_mask):.2f}", flush=True)

    print("[p7_attach_probe] aggregate_pose_means", flush=True)
    print(
        f"[p7_attach_probe]   release sz={mean_valid(release_sz, release_mask):.4f} "
        f"d_xy={mean_valid(release_d_xy, release_mask):.4f} "
        f"rel_z_abs={mean_valid(release_rel_z_abs, release_mask):.4f}",
        flush=True,
    )
    print(
        f"[p7_attach_probe]   final d_xy={float(final['d_xy'].mean()):.4f} "
        f"rel_z_abs={float(final['rel_z_abs'].mean()):.4f} "
        f"settled_z_abs={float(final['settled_z_abs'].mean()):.4f} "
        f"sz={float(final['sz'].mean()):.4f} "
        f"d_sponge_tcp={float(final['d_sponge_tcp'].mean()):.4f}",
        flush=True,
    )
    print(
        f"[p7_attach_probe]   max_tcp_delta_mean={float(max_tcp_delta.mean()):.4f} "
        f"max_tcp_delta_max={float(max_tcp_delta.max()):.4f} "
        f"max_abs_action_mean={float(max_abs_action.mean()):.4f} "
        f"max_ang_vel_grasped_mean={float(max_ang_vel_grasped.mean()):.4f} "
        f"max_ang_vel_grasped_max={float(max_ang_vel_grasped.max()):.4f}",
        flush=True,
    )

    print("[p7_attach_probe] env_transition_samples", flush=True)
    for env_id in range(min(args.trace_envs, args.num_envs)):
        print(
            f"[p7_attach_probe] env={env_id:03d} "
            f"first_open={int(first_open[env_id])} "
            f"first_grasp_false={int(first_grasp_false[env_id])} "
            f"release_or_open={int(release_step[env_id])} "
            f"first_tip_any={int(first_tip_any[env_id])} "
            f"first_tip_grasped={int(first_tip_grasped[env_id])} "
            f"first_large_tcp_jump={int(first_large_tcp_jump[env_id])} "
            f"release_sz={float(release_sz[env_id]):.4f} "
            f"final_sz={float(final['sz'][env_id]):.4f} "
            f"max_tcp_delta={float(max_tcp_delta[env_id]):.4f} "
            f"max_ang_vel_grasped={float(max_ang_vel_grasped[env_id]):.4f}",
            flush=True,
        )

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
