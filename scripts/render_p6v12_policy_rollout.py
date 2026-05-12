"""Render P6v12 policy rollout to MP4 for lab meeting.

Shows the failure mode: robot grasps sponge -> transports to zone -> gripper stays closed.

Run (local 4090 only — HARD RULE #17 narrow):
    conda run -n isaaclab python scripts/render_p6v12_policy_rollout.py \
        --checkpoint local_ckpts/p6v12_model_999.pt \
        --num_episodes 3 \
        --out_dir claudedocs/figures/p6v12_rollout

Output:
    <out_dir>/frame_NNNN.png  — viewport screenshots per step
    <out_dir>/p6v12_rollout.mp4  — encoded video (ffmpeg, 30fps)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--out_dir", default="claudedocs/figures/p6v12_rollout")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Force line buffering (conda run pipe → block-buffered by default)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print(f"[render] checkpoint: {args.checkpoint}", flush=True)
    print(f"[render] out_dir   : {out_dir}", flush=True)
    print(f"[render] episodes  : {args.num_episodes}", flush=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl  # noqa: F401 registers env
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    LOCAL_USD = "/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd/roarm_m3.usd"
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = RoArmStackEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.reward_phase = 6
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 2.0
    # Override B200-hardcoded USD path -> local copy
    if Path(LOCAL_USD).exists():
        env_cfg.robot.spawn.usd_path = LOCAL_USD
        print(f"[render] USD_PATH override -> {LOCAL_USD}")

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = args.seed

    env = gym.make("RoArm-Stack-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner_env = env.unwrapped
    print(f"[render] max_episode_length={inner_env.max_episode_length}")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner_env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=inner_env.device)
    print(f"[render] policy loaded")

    # ====== Camera setup (USD camera prim looking at robot + target) ======
    import omni.usd
    from pxr import UsdGeom, UsdLux, Gf

    stage = omni.usd.get_context().get_stage()

    # Bright dome light
    dome_path = "/World/RenderDomeLight"
    if not stage.GetPrimAtPath(dome_path):
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(3000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.85, 0.85, 0.85))

    # Render camera — angled view of workspace
    cam_path = "/World/RenderCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(11.787)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    # Position: front-right elevated, looking at workspace center (~0.28, 0, 0)
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(0.55, -0.45, 0.45))
    cam_xf.AddRotateXYZOp().Set(Gf.Vec3d(-50.0, 0.0, 50.0))

    # ====== Replicator render product + BasicWriter (proven async streaming) ======
    import omni.replicator.core as rep
    render_product = rep.create.render_product(cam_path, (1280, 720))

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frames_dir), rgb=True)
    writer.attach([render_product])
    print(f"[render] BasicWriter attached -> {frames_dir}", flush=True)

    # ====== Rollout loop ======
    # Force first-step truncation (clean spawn) — same trick as eval_policy.py
    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()

    # Warmup step
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    print("[render] warmup truncation fired", flush=True)

    total_steps = args.num_episodes * inner_env.max_episode_length
    print(f"[render] running {total_steps} steps...", flush=True)

    for t in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)
        # Drive replicator render once per env.step (writer writes async)
        rep.orchestrator.step(rt_subframes=1)

        if t < 5 or t % 25 == 0:
            print(f"  step {t}/{total_steps}  ep={t // inner_env.max_episode_length}", flush=True)

    print(f"[render] rollout done. Frames in {frames_dir}", flush=True)

    env.close()

    # ====== Post-process: sample N target frames evenly (BasicWriter overproduces) ======
    import glob
    import shutil
    pngs = sorted(glob.glob(str(frames_dir / "**/rgb_*.png"), recursive=True))
    if not pngs:
        pngs = sorted(glob.glob(str(frames_dir / "rgb_*.png")))
    print(f"[render] BasicWriter produced {len(pngs)} PNGs", flush=True)

    # Target: total_steps frames (1 ep = 200, 3 ep = 600), evenly sampled
    target_n = total_steps
    if len(pngs) > target_n:
        import numpy as np
        idx = np.linspace(0, len(pngs) - 1, target_n).astype(int)
        sampled = [pngs[i] for i in idx]
        print(f"[render] sampling {target_n} evenly from {len(pngs)} -> {target_n} frames", flush=True)
        # Copy sampled frames to sampled/ subdir for clean MP4 input
        sampled_dir = frames_dir / "sampled"
        sampled_dir.mkdir(exist_ok=True)
        for i, src in enumerate(sampled):
            dst = sampled_dir / f"rgb_{i:04d}.png"
            if not dst.exists():
                shutil.copy(src, dst)
        encode_pngs = sorted(glob.glob(str(sampled_dir / "rgb_*.png")))
    else:
        encode_pngs = pngs

    # ====== ffmpeg encode ======
    mp4_path = out_dir / "p6v12_rollout.mp4"
    print(f"[render] encoding MP4 -> {mp4_path}  ({len(encode_pngs)} frames @ {args.fps}fps)", flush=True)
    import subprocess
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if encode_pngs:
        list_path = out_dir / "_concat.txt"
        with open(list_path, "w") as f:
            for p in encode_pngs:
                f.write(f"file '{p}'\nduration {1.0/args.fps}\n")
            f.write(f"file '{encode_pngs[-1]}'\n")
        rc = subprocess.call([
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", f"fps={args.fps}",
            "-crf", "23",
            str(mp4_path),
        ])
        if rc == 0:
            print(f"[render] MP4 OK: {mp4_path}", flush=True)
        else:
            print(f"[render] ffmpeg FAIL rc={rc}  — PNGs preserved", flush=True)
    else:
        print(f"[render] NO PNGs found — check render product setup", flush=True)

    sim_app.close()


if __name__ == "__main__":
    main()
