"""Rollout failure diagnostic for P7 attached transport/release policies.

This is an evaluation script only. It does not change reward, curriculum, or
scripted release logic. It starts from the P7 G2-A attached-start curriculum,
loads an rsl_rl checkpoint, and classifies rollout failures from state traces.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=2,
                        help="Episodes per env after one warmup reset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--sample_print", type=int, default=16)
    parser.add_argument("--attached_start_jitter_rad", type=float, default=0.0)

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
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, TABLE_Z

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_phase = 7
    cfg.seed = args.seed
    cfg.curriculum_attached_transport_release = True
    cfg.curriculum_attached_start_jitter_rad = args.attached_start_jitter_rad

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    print("[p7_diag] state-only rollout diagnostic", flush=True)
    print(f"[p7_diag] checkpoint={args.checkpoint}", flush=True)
    print(f"[p7_diag] num_envs={args.num_envs} episodes={args.episodes} seed={args.seed}", flush=True)
    print(f"[p7_diag] attached_start_jitter_rad={args.attached_start_jitter_rad}", flush=True)

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    base_env = env.unwrapped

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=base_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)
    print(f"[p7_diag] max_episode_length={base_env.max_episode_length}", flush=True)
    print(f"[p7_diag] grasp_gripper_thresh={base_env.cfg.grasp_gripper_thresh:.4f}rad", flush=True)

    current: list[dict | None] = [None] * args.num_envs
    completed: list[dict] = []
    expected_episodes = args.num_envs * args.episodes
    warmup_done = [False]
    ep_serial = [0]
    step_in_ep = torch.zeros(args.num_envs, dtype=torch.long, device=base_env.device)

    def snapshot(ids: torch.Tensor, label: str) -> dict:
        base_env._compute_intermediate_values(ids)
        env_origins = base_env.scene.env_origins[ids]
        sponge = base_env._sponge_pos_w[ids] - env_origins
        tcp = base_env._tcp_pos_w[ids] - env_origins
        target = base_env._target_world[ids] - env_origins
        quat = base_env._sponge_quat_w[ids]
        qx = quat[:, 1]
        qy = quat[:, 2]
        sz_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        d_xy = torch.norm(sponge[:, :2] - target[:, :2], dim=-1)
        release_z = torch.abs((target[:, 2] + 0.029) - sponge[:, 2])
        settled_z = torch.abs(target[:, 2] - sponge[:, 2])
        d_sponge_tcp = torch.norm(sponge - tcp, dim=-1)
        vel = torch.norm(base_env._sponge.data.root_lin_vel_w[ids], dim=-1)
        gripper_q = base_env._robot.data.joint_pos[ids, base_env.gripper_joint_idx]
        return {
            "label": label,
            "sponge": sponge.detach().cpu(),
            "tcp": tcp.detach().cpu(),
            "target": target.detach().cpu(),
            "quat": quat.detach().cpu(),
            "sz_world_z": sz_world_z.detach().cpu(),
            "d_xy": d_xy.detach().cpu(),
            "release_z_offset": release_z.detach().cpu(),
            "settled_z_offset": settled_z.detach().cpu(),
            "d_sponge_tcp": d_sponge_tcp.detach().cpu(),
            "vel": vel.detach().cpu(),
            "gripper_q": gripper_q.detach().cpu(),
            "grasped": base_env._grasped[ids].detach().cpu(),
            "was_grasped": base_env._was_grasped[ids].detach().cpu(),
        }

    def init_records(ids: torch.Tensor):
        if ids.numel() == 0:
            return
        snap = snapshot(ids, "reset")
        for j, env_id_t in enumerate(ids.detach().cpu().tolist()):
            current[env_id_t] = {
                "episode_id": ep_serial[0],
                "env_id": env_id_t,
                "reset": {k: (v[j].clone() if torch.is_tensor(v) and v.ndim > 0 else v)
                          for k, v in snap.items()},
                "pre_release": None,
                "release": None,
                "post_settle": None,
                "final": None,
                "release_step": -1,
                "release_reason": "none",
                "prev": None,
            }
            ep_serial[0] += 1
            step_in_ep[env_id_t] = 0

    def attach_snapshot(rec: dict, name: str, snap: dict, j: int, step: int):
        one = {}
        for k, v in snap.items():
            one[k] = v[j].clone() if torch.is_tensor(v) and v.ndim > 0 else v
        one["step"] = step
        rec[name] = one

    def classify(rec: dict) -> str:
        release = rec.get("release")
        pre = rec.get("pre_release")
        post = rec.get("post_settle") or rec.get("final")
        final = rec.get("final")
        reset = rec.get("reset")

        if reset is not None:
            if float(reset["d_sponge_tcp"]) > 0.010 or float(reset["sz_world_z"]) < 0.90:
                return "F_pose_write_nonphysical_start"
        if release is None:
            return "E_never_releases"
        if rec["release_step"] > int(0.75 * base_env.max_episode_length):
            return "E_releases_too_late"
        if pre is not None and float(pre["sz_world_z"]) < 0.90:
            return "C_tips_during_attached_transport"
        if float(release["sz_world_z"]) < 0.90:
            return "C_tips_during_attached_transport"
        if float(release["d_xy"]) > 0.080:
            return "A_releases_too_early_before_xy"
        if float(release["d_xy"]) <= 0.040 and float(release["release_z_offset"]) > 0.025:
            return "B_reaches_xy_wrong_release_height"
        if post is not None and float(release["sz_world_z"]) >= 0.90 and float(post["sz_world_z"]) < 0.90:
            return "D_tips_after_release_settle"
        if final is not None and float(final["d_xy"]) > 0.030:
            return "A_releases_with_residual_xy_error"
        if final is not None and float(final["settled_z_offset"]) > 0.025:
            return "B_wrong_settled_height"
        return "other_unsuccessful_or_success_like"

    orig_reset = base_env._reset_idx

    def hooked_reset(env_ids):
        ids = env_ids
        if ids is None:
            ids = base_env._robot._ALL_INDICES
        if warmup_done[0] and isinstance(ids, torch.Tensor) and ids.numel() > 0:
            snap = snapshot(ids, "final")
            for j, env_id in enumerate(ids.detach().cpu().tolist()):
                if len(completed) >= expected_episodes:
                    break
                rec = current[env_id]
                if rec is None:
                    continue
                attach_snapshot(rec, "final", snap, j, int(step_in_ep[env_id].item()))
                rec["bucket"] = classify(rec)
                completed.append(rec)
                current[env_id] = None
        orig_reset(env_ids)
        if (
            warmup_done[0]
            and isinstance(ids, torch.Tensor)
            and ids.numel() > 0
            and len(completed) < expected_episodes
        ):
            init_records(ids)

    base_env._reset_idx = hooked_reset

    base_env.episode_length_buf[:] = base_env.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    warmup_done[0] = True
    all_ids = torch.arange(args.num_envs, device=base_env.device)
    init_records(all_ids)
    print("[p7_diag] warmup reset complete", flush=True)

    total_steps = args.episodes * base_env.max_episode_length
    for _ in range(total_steps):
        live_ids = torch.tensor(
            [i for i, rec in enumerate(current) if rec is not None],
            device=base_env.device,
            dtype=torch.long,
        )
        if live_ids.numel() > 0:
            prev_snap = snapshot(live_ids, "prev")
            for j, env_id in enumerate(live_ids.detach().cpu().tolist()):
                rec = current[env_id]
                if rec is not None:
                    rec["prev"] = {k: (v[j].clone() if torch.is_tensor(v) and v.ndim > 0 else v)
                                   for k, v in prev_snap.items()}

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        done_mask = dones.detach().to(torch.bool) if torch.is_tensor(dones) else torch.zeros(args.num_envs, dtype=torch.bool, device=base_env.device)
        active_ids = torch.where(~done_mask)[0]
        step_in_ep[active_ids] += 1

        ids = torch.tensor(
            [i for i in active_ids.detach().cpu().tolist() if current[i] is not None],
            device=base_env.device,
            dtype=torch.long,
        )
        if ids.numel() == 0:
            continue
        snap = snapshot(ids, "step")
        gripper_open = snap["gripper_q"] < base_env.cfg.grasp_gripper_thresh
        for j, env_id in enumerate(ids.detach().cpu().tolist()):
            rec = current[env_id]
            if rec is None:
                continue
            step = int(step_in_ep[env_id].item())
            release_now = (not bool(snap["grasped"][j].item())) or bool(gripper_open[j].item())
            if rec["release"] is None and release_now:
                rec["pre_release"] = rec.get("prev")
                attach_snapshot(rec, "release", snap, j, step)
                rec["release_step"] = step
                rec["release_reason"] = (
                    "grasped_false+gripper_open"
                    if (not bool(snap["grasped"][j].item()) and bool(gripper_open[j].item()))
                    else ("grasped_false" if not bool(snap["grasped"][j].item()) else "gripper_open")
                )
            if rec["release"] is not None and rec["post_settle"] is None:
                if step >= rec["release_step"] + args.settle_steps:
                    attach_snapshot(rec, "post_settle", snap, j, step)

    # In case the loop ended before an automatic truncation, close out live records.
    live_ids = torch.tensor(
        [i for i, rec in enumerate(current) if rec is not None],
        device=base_env.device,
        dtype=torch.long,
    )
    if live_ids.numel() > 0 and len(completed) < expected_episodes:
        snap = snapshot(live_ids, "final_loop_end")
        for j, env_id in enumerate(live_ids.detach().cpu().tolist()):
            if len(completed) >= expected_episodes:
                break
            rec = current[env_id]
            if rec is None:
                continue
            attach_snapshot(rec, "final", snap, j, int(step_in_ep[env_id].item()))
            rec["bucket"] = classify(rec)
            completed.append(rec)
            current[env_id] = None

    counts = Counter(r["bucket"] for r in completed)
    n = len(completed)
    print(f"[p7_diag] completed_episodes={n}", flush=True)
    print("[p7_diag] bucket_counts", flush=True)
    for key, val in counts.most_common():
        print(f"[p7_diag]   {key}: {val} ({val / max(n, 1):.3f})", flush=True)

    def mean_of(name: str, snap_name: str) -> float:
        vals = [float(r[snap_name][name]) for r in completed if r.get(snap_name) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("[p7_diag] aggregate_means_m", flush=True)
    for snap_name in ("reset", "pre_release", "release", "post_settle", "final"):
        print(
            f"[p7_diag]   {snap_name}: "
            f"d_xy={mean_of('d_xy', snap_name):.4f} "
            f"release_z_offset={mean_of('release_z_offset', snap_name):.4f} "
            f"settled_z_offset={mean_of('settled_z_offset', snap_name):.4f} "
            f"sz_world_z={mean_of('sz_world_z', snap_name):.4f}",
            flush=True,
        )

    print("[p7_diag] episode_samples", flush=True)
    for rec in completed[:args.sample_print]:
        release = rec.get("release")
        final = rec.get("final")
        post = rec.get("post_settle")
        print(
            f"[p7_diag] ep={rec['episode_id']} env={rec['env_id']} "
            f"bucket={rec['bucket']} release_step={rec['release_step']} "
            f"reason={rec['release_reason']} "
            f"release_d_xy={(float(release['d_xy']) if release else float('nan')):.4f} "
            f"release_z={(float(release['release_z_offset']) if release else float('nan')):.4f} "
            f"post_sz={(float(post['sz_world_z']) if post else float('nan')):.4f} "
            f"final_d_xy={(float(final['d_xy']) if final else float('nan')):.4f} "
            f"final_settled_z={(float(final['settled_z_offset']) if final else float('nan')):.4f} "
            f"final_sz={(float(final['sz_world_z']) if final else float('nan')):.4f}",
            flush=True,
        )

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
