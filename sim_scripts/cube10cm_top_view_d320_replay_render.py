#!/usr/bin/env python3
"""Replay-render a small D319 trajectory smoke set for D320.

The replay contract is D319 baseline v2:
- D256 reset row selected by episode_index/frame_index 0
- friction override from the D319 conveyor manifest
- candidate8_diffik_target_residual with zero 3D residual actions
- candidate8_hybrid_stop_after_useful enabled

This script renders only a tiny smoke set. It does not train PPO and does not
scale data generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT = RUNTIME_ROOT / "data_conveyor_d320" / "replay_smoke" / "render_d319_replay_smoke"
DEFAULT_MANIFEST = RUNTIME_ROOT / "data_conveyor_d320" / "replay_smoke" / "d320_replay_smoke_manifest.csv"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3200)
    parser.add_argument("--steps-per-episode", type=int, default=580)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-usd-path", type=Path, default=DEFAULT_USD)
    parser.add_argument(
        "--close-sim-app",
        action="store_true",
        help="Explicitly close Isaac app. Default exits directly because local Kit close can hang.",
    )
    return parser.parse_args()


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty replay manifest: {path}")
    rows.sort(key=lambda row: int(row["d320_episode_id"]))
    if len(rows) > 10:
        raise ValueError("D320 replay render is limited to <=10 episodes")
    return rows


def base_contract_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=int(args.seed),
        num_envs=1,
        max_iterations=0,
        num_steps_per_env=64,
        eval_steps=int(args.steps_per_episode),
        save_interval=1,
        robot_usd_path=args.robot_usd_path,
        runtime_dir=args.out_dir,
        summary_json=args.out_dir / "unused_summary.json",
        summary_out=args.out_dir / "unused_summary.out",
        experiment_name="cube10cm_top_view_d320_replay_render",
        fixed_cube_x_m=0.24,
        fixed_cube_y_m=0.0,
        cube_randomization_half_extent_x_m=0.0,
        cube_randomization_half_extent_y_m=0.0,
        policy_target_disp_m=0.006,
        precontact_clearance_m=0.040,
        episode_length_s=6.0,
        step_clip_rad=0.010,
        joint_target_lead_limit_rad=0.060,
        action_scale=0.050,
        rl_action_mode="candidate8_diffik_target_residual",
        candidate6_diffik_push_steps=int(args.steps_per_episode),
        candidate6_diffik_residual_scale_rad=0.002,
        candidate6_diffik_lambda=0.010,
        candidate8_diffik_target_residual_forward_m=0.004,
        candidate8_diffik_target_residual_lateral_m=0.012,
        candidate8_diffik_target_residual_height_m=0.004,
        tap_success_terminate=False,
        disable_tap_overshoot_terminate=True,
        candidate6_diffik_no_hold_after_tap_success=False,
        candidate6_diffik_target_base_mode="previous_joint_target",
        candidate6_diffik_target_path_mode="near_face_goal",
        candidate6_diffik_cube_reference_mode="current_pose",
        init_at_random_ep_len=False,
        load_checkpoint=None,
        initial_policy_eval=False,
        constant_policy_action=None,
        ppo_init_noise_std=0.2,
        tap_transient_disp_reward_scale=None,
        tap_overshoot_penalty_scale=None,
        action_penalty_scale=None,
    )


def make_env_cfg(args: argparse.Namespace, row: dict[str, str]) -> tuple[Any, dict[str, Any]]:
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.train_cube_tap10cm_ppo_smoke import _apply_candidate6_contract, _contract_violations
    from sim_scripts.cube10cm_top_view_d290_closed_loop_recovery_probe import DEFAULT_D256_CSV

    contract_args = base_contract_args(args)
    env_cfg = RoArmCubeTap10cmEnvCfg()
    contract = _apply_candidate6_contract(env_cfg, contract_args)
    violations = _contract_violations(contract, contract_args)
    if violations:
        raise RuntimeError(f"candidate contract violations: {violations}")

    env_cfg.viewer.cam_prim_path = "/World/Cube10cmTopViewCamera"
    env_cfg.viewer.resolution = (int(args.width), int(args.height))
    env_cfg.d256_reset_csv_path = str(DEFAULT_D256_CSV)
    env_cfg.d256_reset_frame_index = 0
    env_cfg.d256_reset_sample_mode = "linspace"
    env_cfg.d256_reset_episode_min = int(row["episode_index"])
    env_cfg.d256_reset_episode_max = int(row["episode_index"])
    env_cfg.candidate8_hybrid_stop_after_useful = True
    env_cfg.tap_success_terminate = False
    env_cfg.tap_overshoot_terminate = False
    env_cfg.scene.clone_in_fabric = False
    env_cfg.sponge.spawn.physics_material.static_friction = float(row["static_friction"])
    env_cfg.sponge.spawn.physics_material.dynamic_friction = float(row["dynamic_friction"])
    env_cfg.cube_friction_randomize_min = float(row["static_friction"])
    env_cfg.cube_friction_randomize_max = float(row["static_friction"]) + 1.0e-9
    env_cfg.cube_dynamic_friction_ratio = float(row["dynamic_friction"]) / max(float(row["static_friction"]), 1.0e-9)
    return env_cfg, contract


def row_provenance(row: dict[str, str]) -> dict[str, Any]:
    return {
        "d319_bin": row["bin"],
        "d319_chunk": row["chunk"],
        "d319_artifact_tag": row["artifact_tag"],
        "d319_env_id": int(row["env_id"]),
        "d319_episode_index": int(row["episode_index"]),
        "d319_static_friction": float(row["static_friction"]),
        "d319_dynamic_friction": float(row["dynamic_friction"]),
        "d319_contact": int(row["contact"]),
        "d319_reaction": int(row["reaction"]),
        "d319_useful": int(row["useful"]),
        "d319_overshoot": int(row["overshoot"]),
        "d319_accepted": int(row["accepted"]),
        "d319_max_disp_xy_m": float(row["max_disp_xy_m"]),
        "d319_max_disp_along_m": float(row["max_disp_along_m"]),
        "d319_max_lateral_disp_m": float(row["max_lateral_disp_m"]),
        "d319_final_proxy": int(row["final_proxy"]),
        "d319_hybrid_latched": int(row["hybrid_latched"]),
        "d319_hybrid_stop_step": int(row["hybrid_stop_step"]),
        "source_role": row["source_role"],
        "selection_reason": row["selection_reason"],
    }


def make_frame_row(
    *,
    smoke_helpers: Any,
    inner_env: Any,
    episode_id: int,
    frame_id: int,
    sim_step: int,
    action3: Any,
    fps: int,
    manifest_row: dict[str, str],
) -> dict[str, Any]:
    row = smoke_helpers.frame_metadata(
        inner_env=inner_env,
        episode_id=episode_id,
        frame_id=frame_id,
        sim_step=sim_step,
        action3=action3,
        fps=fps,
    )
    row.update(row_provenance(manifest_row))
    row["d256_reset_active"] = float(inner_env._last_d256_reset_active[0].detach().cpu().item())
    row["d256_reset_episode_index_runtime"] = int(
        round(float(inner_env._last_d256_reset_episode_index[0].detach().cpu().item()))
    )
    row["candidate8_hybrid_stop_after_useful"] = bool(
        getattr(inner_env.cfg, "candidate8_hybrid_stop_after_useful", False)
    )
    row["candidate8_hybrid_stop_latched"] = int(
        bool(inner_env._last_candidate8_hybrid_stop_latched[0].detach().cpu().item())
    )
    row["candidate8_hybrid_stop_step"] = int(inner_env._candidate8_hybrid_stop_step[0].detach().cpu().item())
    return row


def summarize(rows: list[dict[str, Any]], out_dir: Path, elapsed_s: float, manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    png_bytes = sum((REPO / row["source_png"]).stat().st_size for row in rows)
    episodes = sorted({int(row["episode_id"]) for row in rows})
    final_rows = [max((row for row in rows if int(row["episode_id"]) == ep), key=lambda item: int(item["sim_step"])) for ep in episodes]
    by_role: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        by_role[row["source_role"]] += 1
    replay_metrics = []
    for row in final_rows:
        replay_metrics.append(
            {
                "episode_id": int(row["episode_id"]),
                "source_role": row["source_role"],
                "d319_episode_index": int(row["d319_episode_index"]),
                "d319_bin": row["d319_bin"],
                "d319_max_disp_xy_m": float(row["d319_max_disp_xy_m"]),
                "replay_tap_disp_xy_m": float(row["tap_disp_xy_m"]),
                "replay_contact_seen": int(float(row["tap_contact_seen"]) >= 0.5),
                "replay_reaction_seen": int(float(row["tap_reaction_seen"]) >= 0.5),
                "replay_overshoot_seen": int(float(row["tap_overshoot_seen"]) >= 0.5),
                "replay_hybrid_stop_step": int(row["candidate8_hybrid_stop_step"]),
                "runtime_d256_episode_index": int(row["d256_reset_episode_index_runtime"]),
            }
        )
    return {
        "artifact": "d320_d319_replay_smoke_render",
        "runtime": "D319_REPLAY_RENDER_ONLY_NO_TRAINING_NO_SCALEUP",
        "episodes": len(episodes),
        "frames": len(rows),
        "elapsed_s": float(elapsed_s),
        "effective_render_fps": float(len(rows) / max(elapsed_s, 1.0e-9)),
        "png_bytes_total": int(png_bytes),
        "png_mb_per_episode": float((png_bytes / 1_000_000.0) / max(1, len(episodes))),
        "role_counts": dict(by_role),
        "replay_episode_metrics": replay_metrics,
        "out_dir": rel(out_dir),
    }


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if int(args.capture_stride) < 1:
        raise ValueError("--capture-stride must be >= 1")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")
    manifest_rows = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_frames_dir = args.out_dir / "raw_env_render_frames"
    if raw_frames_dir.exists() and any(raw_frames_dir.glob("*.png")):
        raise FileExistsError(f"{raw_frames_dir} already contains PNG files; choose a new --out-dir")
    raw_frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[d320-replay-render] start "
        f"episodes={len(manifest_rows)} steps={args.steps_per_episode} stride={args.capture_stride} "
        f"manifest={args.manifest} out={args.out_dir}",
        flush=True,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
    sim_app = app_launcher.app

    try:
        import gymnasium as gym
        import omni.usd
        import torch
        from PIL import Image

        import roarm_rl  # noqa: F401
        from sim_scripts import cube10cm_top_view_visual_smoke_render as smoke_helpers

        stage = None
        rows: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        overall_t0 = time.time()
        env_cfg, contract = make_env_cfg(args, manifest_rows[0])
        env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
        inner_env = env.unwrapped
        print(
            "[d320-replay-render] env_ready "
            f"episodes={len(manifest_rows)} max_episode_length={inner_env.max_episode_length} "
            "friction_override=reset_time_material_randomization",
            flush=True,
        )
        stage = omni.usd.get_context().get_stage()
        smoke_helpers.install_camera(stage, int(args.width), int(args.height))
        for _ in range(10):
            sim_app.update()

        with torch.inference_mode():
            for manifest_row in manifest_rows:
                episode_id = int(manifest_row["d320_episode_id"])
                ep_index = int(manifest_row["episode_index"])
                static_friction = float(manifest_row["static_friction"])
                dynamic_friction = float(manifest_row["dynamic_friction"])
                inner_env.cfg.d256_reset_episode_min = ep_index
                inner_env.cfg.d256_reset_episode_max = ep_index
                inner_env.cfg.cube_friction_randomize_min = static_friction
                inner_env.cfg.cube_friction_randomize_max = static_friction + 1.0e-9
                inner_env.cfg.cube_dynamic_friction_ratio = dynamic_friction / max(static_friction, 1.0e-9)
                if hasattr(inner_env, "_d256_reset_table"):
                    delattr(inner_env, "_d256_reset_table")
                env.reset()
                runtime_ep = int(round(float(inner_env._last_d256_reset_episode_index[0].detach().cpu().item())))
                if runtime_ep != ep_index:
                    raise RuntimeError(f"D256 replay episode mismatch: expected={ep_index} runtime={runtime_ep}")

                episode_t0 = time.time()
                frame_id = 0
                action3 = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)

                rgb = smoke_helpers.capture_rgb_from_env(env)
                frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                Image.fromarray(rgb).save(frame_path)
                frame_paths.append(frame_path)
                rows.append(
                    make_frame_row(
                        smoke_helpers=smoke_helpers,
                        inner_env=inner_env,
                        episode_id=episode_id,
                        frame_id=frame_id,
                        sim_step=0,
                        action3=action3,
                        fps=int(args.fps),
                        manifest_row=manifest_row,
                    )
                )

                for sim_step in range(1, int(args.steps_per_episode) + 1):
                    action3 = torch.zeros(
                        (inner_env.num_envs, int(inner_env.cfg.action_space)),
                        device=inner_env.device,
                    )
                    env.step(action3)
                    if sim_step % int(args.capture_stride) != 0 and sim_step != int(args.steps_per_episode):
                        continue
                    frame_id += 1
                    rgb = smoke_helpers.capture_rgb_from_env(env)
                    frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                    Image.fromarray(rgb).save(frame_path)
                    frame_paths.append(frame_path)
                    rows.append(
                        make_frame_row(
                            smoke_helpers=smoke_helpers,
                            inner_env=inner_env,
                            episode_id=episode_id,
                            frame_id=frame_id,
                            sim_step=sim_step,
                            action3=action3,
                            fps=int(args.fps),
                            manifest_row=manifest_row,
                        )
                    )

                print(
                    "[d320-replay-render] episode "
                    f"d320={episode_id} d319_ep={ep_index} role={manifest_row['source_role']} "
                    f"bin={manifest_row['bin']} friction=({static_friction:.2f},{dynamic_friction:.2f}) "
                    f"frames={frame_id + 1} elapsed_s={time.time() - episode_t0:.1f}",
                    flush=True,
                )

        elapsed_s = time.time() - overall_t0
        smoke_helpers.attach_source_pngs_and_visibility(rows, frame_paths)
        frames_jsonl = args.out_dir / "frames.jsonl"
        with frames_jsonl.open("w") as fh:
            for row in sorted(rows, key=lambda item: (int(item["episode_id"]), int(item["frame_id"]))):
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        summary = summarize(rows, args.out_dir, elapsed_s, manifest_rows)
        summary["manifest"] = rel(args.manifest)
        summary["frames_jsonl"] = rel(frames_jsonl)
        summary["contract"] = contract
        summary["friction_replay_method"] = "reset_time_cube_friction_randomization_min_static_max_static_plus_1e-9"
        summary_path = args.out_dir / "render_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            "[d320-replay-render] done "
            f"frames={summary['frames']} episodes={summary['episodes']} "
            f"effective_fps={summary['effective_render_fps']:.3f} summary={summary_path}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        if not args.close_sim_app:
            os._exit(0)
        if args.close_sim_app:
            env.close()
    finally:
        if args.close_sim_app:
            sim_app.close()


if __name__ == "__main__":
    main()
