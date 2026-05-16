"""Step trace for P7 attached transport/release action, TCP, and quaternion.

Evaluation-only diagnostic. It does not change reward, curriculum, scripted
release, chain skills, or grasp attach semantics.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _fmt_vec(vals, precision: int = 3) -> str:
    return "(" + ",".join(f"{float(v):+.{precision}f}" for v in vals) + ")"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace_envs", type=int, default=4)
    parser.add_argument("--trace_every", type=int, default=1)
    parser.add_argument("--attached_start_jitter_rad", type=float, default=0.0)
    parser.add_argument("--upright_thresh", type=float, default=0.90)
    parser.add_argument("--large_tcp_jump_m", type=float, default=0.030)
    parser.add_argument("--attach_quat_mode", choices=("preserve", "identity"), default="preserve")
    parser.add_argument("--attach_velocity_mode", choices=("zero", "keep"), default="zero")

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
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = 7
    cfg.seed = args.seed
    cfg.curriculum_attached_transport_release = True
    cfg.curriculum_attached_start_jitter_rad = args.attached_start_jitter_rad
    cfg.attach_quat_mode = args.attach_quat_mode
    cfg.attach_velocity_mode = args.attach_velocity_mode

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print("[p7_trace] action/tcp/quaternion step trace", flush=True)
    print(f"[p7_trace] checkpoint={args.checkpoint}", flush=True)
    print(
        f"[p7_trace] num_envs={args.num_envs} max_steps={args.max_steps} "
        f"trace_envs={args.trace_envs} seed={args.seed}",
        flush=True,
    )
    print(
        f"[p7_trace] attach_quat_mode={args.attach_quat_mode} "
        f"attach_velocity_mode={args.attach_velocity_mode} "
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
    trace_ids = torch.arange(min(args.trace_envs, args.num_envs), device=base_env.device, dtype=torch.long)
    all_ids = torch.arange(args.num_envs, device=base_env.device, dtype=torch.long)
    gripper_idx = int(base_env.gripper_joint_idx)

    print(f"[p7_trace] max_episode_length={base_env.max_episode_length}", flush=True)
    print(f"[p7_trace] step_dt={dt:.4f}", flush=True)
    print(f"[p7_trace] grasp_gripper_thresh={base_env.cfg.grasp_gripper_thresh:.4f}rad", flush=True)
    print(f"[p7_trace] gripper_joint_idx={gripper_idx}", flush=True)
    print(
        "[p7_trace] attach_semantics roarm_stack_env.py _update_grasp_attach "
        f"quat_mode={base_env.cfg.attach_quat_mode} velocity_mode={base_env.cfg.attach_velocity_mode}",
        flush=True,
    )

    def snapshot(ids: torch.Tensor) -> dict:
        base_env._compute_intermediate_values(ids)
        origins = base_env.scene.env_origins[ids]
        tcp = base_env._tcp_pos_w[ids] - origins
        sponge = base_env._sponge_pos_w[ids] - origins
        target = base_env._target_world[ids] - origins
        quat = base_env._sponge_quat_w[ids]
        qx = quat[:, 1]
        qy = quat[:, 2]
        sz_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        d_xy = torch.norm(sponge[:, :2] - target[:, :2], dim=-1)
        release_z_signed = (target[:, 2] + 0.029) - sponge[:, 2]
        release_z_abs = torch.abs(release_z_signed)
        settled_z_abs = torch.abs(target[:, 2] - sponge[:, 2])
        d_sponge_tcp = torch.norm(sponge - tcp, dim=-1)
        gripper_q = base_env._robot.data.joint_pos[ids, gripper_idx]
        lin_vel = base_env._sponge.data.root_lin_vel_w[ids]
        ang_vel = base_env._sponge.data.root_ang_vel_w[ids]
        return {
            "tcp": tcp.detach().clone(),
            "sponge": sponge.detach().clone(),
            "target": target.detach().clone(),
            "quat": quat.detach().clone(),
            "sz": sz_world_z.detach().clone(),
            "d_xy": d_xy.detach().clone(),
            "release_z_signed": release_z_signed.detach().clone(),
            "release_z_abs": release_z_abs.detach().clone(),
            "settled_z_abs": settled_z_abs.detach().clone(),
            "d_sponge_tcp": d_sponge_tcp.detach().clone(),
            "gripper_q": gripper_q.detach().clone(),
            "grasped": base_env._grasped[ids].detach().clone(),
            "was_grasped": base_env._was_grasped[ids].detach().clone(),
            "lin_vel": lin_vel.detach().clone(),
            "ang_vel": ang_vel.detach().clone(),
        }

    def print_trace(step: int, env_id: int, j: int, actions_cpu, snap_cpu, prev_cpu):
        action = actions_cpu[env_id]
        tcp = snap_cpu["tcp"][j]
        sponge = snap_cpu["sponge"][j]
        quat = snap_cpu["quat"][j]
        tcp_delta = tcp - prev_cpu["tcp"][j]
        tcp_vel = tcp_delta / max(dt, 1e-6)
        quat_delta = torch.norm(quat - prev_cpu["quat"][j]).item()
        gripper_q = float(snap_cpu["gripper_q"][j])
        gripper_open = gripper_q < base_env.cfg.grasp_gripper_thresh
        print(
            f"[p7_trace] env={env_id:03d} step={step:03d} "
            f"act={_fmt_vec(action.tolist(), 3)} grip_act={float(action[gripper_idx]):+.3f} "
            f"grip_q={gripper_q:+.4f} open={int(gripper_open)} "
            f"tcp={_fmt_vec(tcp.tolist(), 4)} dtcp={_fmt_vec(tcp_delta.tolist(), 4)} "
            f"tcp_vel={_fmt_vec(tcp_vel.tolist(), 3)} "
            f"sponge={_fmt_vec(sponge.tolist(), 4)} quat={_fmt_vec(quat.tolist(), 4)} "
            f"q_delta={quat_delta:.5f} sz={float(snap_cpu['sz'][j]):+.4f} "
            f"grasped={int(bool(snap_cpu['grasped'][j]))} was={int(bool(snap_cpu['was_grasped'][j]))} "
            f"d_xy={float(snap_cpu['d_xy'][j]):.4f} "
            f"rel_z_signed={float(snap_cpu['release_z_signed'][j]):+.4f} "
            f"rel_z_abs={float(snap_cpu['release_z_abs'][j]):.4f} "
            f"settled_z_abs={float(snap_cpu['settled_z_abs'][j]):.4f} "
            f"d_sponge_tcp={float(snap_cpu['d_sponge_tcp'][j]):.4f} "
            f"lin_vel_norm={float(torch.norm(snap_cpu['lin_vel'][j])):.4f} "
            f"ang_vel_norm={float(torch.norm(snap_cpu['ang_vel'][j])):.4f}",
            flush=True,
        )

    # Force one truncation/reset so initial records match the P7 attached-start curriculum.
    base_env.episode_length_buf[:] = base_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    prev_all = snapshot(all_ids)
    print("[p7_trace] warmup reset complete", flush=True)
    print(
        f"[p7_trace] reset_mean d_xy={float(prev_all['d_xy'].mean()):.4f} "
        f"sz={float(prev_all['sz'].mean()):.4f} "
        f"d_sponge_tcp={float(prev_all['d_sponge_tcp'].mean()):.5f} "
        f"grasped={float(prev_all['grasped'].float().mean()):.3f}",
        flush=True,
    )

    first_grasp_false = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_open = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_tip_any = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_tip_grasped = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    first_large_tcp_jump = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    max_tcp_delta = torch.zeros(args.num_envs, device=base_env.device)
    max_abs_action = torch.zeros(args.num_envs, device=base_env.device)
    release_step = torch.full((args.num_envs,), -1, device=base_env.device, dtype=torch.long)
    release_sz = torch.full((args.num_envs,), float("nan"), device=base_env.device)
    release_d_xy = torch.full((args.num_envs,), float("nan"), device=base_env.device)
    release_rel_z_abs = torch.full((args.num_envs,), float("nan"), device=base_env.device)
    release_rel_z_signed = torch.full((args.num_envs,), float("nan"), device=base_env.device)

    for step in range(1, args.max_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            actions = actions.clamp(-1.0, 1.0)
            obs, _, dones, _ = env.step(actions)

        snap_all = snapshot(all_ids)
        tcp_delta_all = torch.norm(snap_all["tcp"] - prev_all["tcp"], dim=-1)
        max_tcp_delta = torch.maximum(max_tcp_delta, tcp_delta_all)
        max_abs_action = torch.maximum(max_abs_action, torch.max(torch.abs(actions.detach()), dim=-1).values)

        gripper_open = snap_all["gripper_q"] < base_env.cfg.grasp_gripper_thresh
        became_grasp_false = (~snap_all["grasped"]) & (first_grasp_false < 0)
        became_open = gripper_open & (first_open < 0)
        tipped_any = (snap_all["sz"] < args.upright_thresh) & (first_tip_any < 0)
        tipped_grasped = (snap_all["sz"] < args.upright_thresh) & snap_all["grasped"] & (first_tip_grasped < 0)
        large_jump = (tcp_delta_all > args.large_tcp_jump_m) & (first_large_tcp_jump < 0)
        released = ((~snap_all["grasped"]) | gripper_open) & (release_step < 0)

        first_grasp_false[became_grasp_false] = step
        first_open[became_open] = step
        first_tip_any[tipped_any] = step
        first_tip_grasped[tipped_grasped] = step
        first_large_tcp_jump[large_jump] = step
        release_step[released] = step
        release_sz[released] = snap_all["sz"][released]
        release_d_xy[released] = snap_all["d_xy"][released]
        release_rel_z_abs[released] = snap_all["release_z_abs"][released]
        release_rel_z_signed[released] = snap_all["release_z_signed"][released]

        if step == 1 or step % max(args.trace_every, 1) == 0:
            snap_trace = snapshot(trace_ids)
            prev_trace = {k: v[trace_ids].detach().cpu() if torch.is_tensor(v) else v for k, v in prev_all.items()}
            snap_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in snap_trace.items()}
            actions_cpu = actions.detach().cpu()
            for j, env_id in enumerate(trace_ids.detach().cpu().tolist()):
                print_trace(step, env_id, j, actions_cpu, snap_cpu, prev_trace)

        prev_all = snap_all
        if torch.all(release_step >= 0) and step >= int(torch.max(release_step).item()) + 5:
            print(f"[p7_trace] early_stop step={step} all envs released/opened plus 5 steps", flush=True)
            break

    final = snapshot(all_ids)

    def _count(mask) -> int:
        return int(mask.sum().item())

    def _mean_valid(vals, mask):
        if _count(mask) == 0:
            return float("nan")
        return float(vals[mask].float().mean().item())

    release_mask = release_step >= 0
    tip_any_mask = first_tip_any >= 0
    tip_grasped_mask = first_tip_grasped >= 0
    open_mask = first_open >= 0
    grasp_false_mask = first_grasp_false >= 0
    large_jump_mask = first_large_tcp_jump >= 0
    tip_before_open = tip_any_mask & ((first_open < 0) | (first_tip_any <= first_open))
    tip_while_grasped_before_release = tip_grasped_mask & (
        (release_step < 0) | (first_tip_grasped <= release_step)
    )

    print("[p7_trace] aggregate_transition_counts", flush=True)
    print(f"[p7_trace]   first_open={_count(open_mask)}/{args.num_envs}", flush=True)
    print(f"[p7_trace]   first_grasp_false={_count(grasp_false_mask)}/{args.num_envs}", flush=True)
    print(f"[p7_trace]   release_or_open={_count(release_mask)}/{args.num_envs}", flush=True)
    print(f"[p7_trace]   first_tip_any={_count(tip_any_mask)}/{args.num_envs}", flush=True)
    print(f"[p7_trace]   first_tip_while_grasped={_count(tip_grasped_mask)}/{args.num_envs}", flush=True)
    print(f"[p7_trace]   tip_before_or_at_open={_count(tip_before_open)}/{args.num_envs}", flush=True)
    print(
        f"[p7_trace]   tip_while_grasped_before_or_at_release={_count(tip_while_grasped_before_release)}/{args.num_envs}",
        flush=True,
    )
    print(
        f"[p7_trace]   first_large_tcp_jump>{args.large_tcp_jump_m:.3f}m="
        f"{_count(large_jump_mask)}/{args.num_envs}",
        flush=True,
    )

    print("[p7_trace] aggregate_transition_steps", flush=True)
    print(f"[p7_trace]   mean_first_open={_mean_valid(first_open, open_mask):.2f}", flush=True)
    print(f"[p7_trace]   mean_first_grasp_false={_mean_valid(first_grasp_false, grasp_false_mask):.2f}", flush=True)
    print(f"[p7_trace]   mean_release_or_open={_mean_valid(release_step, release_mask):.2f}", flush=True)
    print(f"[p7_trace]   mean_first_tip_any={_mean_valid(first_tip_any, tip_any_mask):.2f}", flush=True)
    print(f"[p7_trace]   mean_first_tip_while_grasped={_mean_valid(first_tip_grasped, tip_grasped_mask):.2f}", flush=True)
    print(f"[p7_trace]   mean_first_large_tcp_jump={_mean_valid(first_large_tcp_jump, large_jump_mask):.2f}", flush=True)

    print("[p7_trace] aggregate_pose_means", flush=True)
    print(
        f"[p7_trace]   release sz={_mean_valid(release_sz, release_mask):.4f} "
        f"d_xy={_mean_valid(release_d_xy, release_mask):.4f} "
        f"rel_z_abs={_mean_valid(release_rel_z_abs, release_mask):.4f} "
        f"rel_z_signed={_mean_valid(release_rel_z_signed, release_mask):+.4f}",
        flush=True,
    )
    print(
        f"[p7_trace]   final d_xy={float(final['d_xy'].mean()):.4f} "
        f"rel_z_abs={float(final['release_z_abs'].mean()):.4f} "
        f"rel_z_signed={float(final['release_z_signed'].mean()):+.4f} "
        f"settled_z_abs={float(final['settled_z_abs'].mean()):.4f} "
        f"sz={float(final['sz'].mean()):.4f} "
        f"d_sponge_tcp={float(final['d_sponge_tcp'].mean()):.4f}",
        flush=True,
    )
    print(
        f"[p7_trace]   max_tcp_delta_mean={float(max_tcp_delta.mean()):.4f} "
        f"max_tcp_delta_max={float(max_tcp_delta.max()):.4f} "
        f"max_abs_action_mean={float(max_abs_action.mean()):.4f}",
        flush=True,
    )

    print("[p7_trace] env_transition_samples", flush=True)
    for env_id in range(min(args.trace_envs, args.num_envs)):
        print(
            f"[p7_trace] env={env_id:03d} "
            f"first_open={int(first_open[env_id].item())} "
            f"first_grasp_false={int(first_grasp_false[env_id].item())} "
            f"release_or_open={int(release_step[env_id].item())} "
            f"first_tip_any={int(first_tip_any[env_id].item())} "
            f"first_tip_grasped={int(first_tip_grasped[env_id].item())} "
            f"first_large_tcp_jump={int(first_large_tcp_jump[env_id].item())} "
            f"release_sz={float(release_sz[env_id]):.4f} "
            f"final_sz={float(final['sz'][env_id]):.4f} "
            f"max_tcp_delta={float(max_tcp_delta[env_id]):.4f}",
            flush=True,
        )

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
