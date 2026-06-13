#!/usr/bin/env python3
"""Render a manifest-fed 0-99 top-view visual chunk for the 10cm cube task.

This is the scale-up renderer for the D236/D237 manifest path. It deliberately
does not replace or weaken the D233 smoke renderer, which remains capped at
1-10 episodes. This script requires an explicit episode manifest and refuses to
run without one.

Do not run this script without explicit runtime approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import cube10cm_top_view_visual_smoke_render as smoke


LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_MANIFEST = LOG_DIR / "cube10cm_top_view_chunk100_manifest_d236" / "episode_manifest.csv"
DEFAULT_OUT = LOG_DIR / "cube10cm_top_view_visual_chunk100_d235"
ARTIFACT = "cube10cm_top_view_visual_chunk100_d235_render"

MANIFEST_FIELDS = [
    "episode_index",
    "split_candidate",
    "cube_x_m",
    "cube_y_m",
    "seed",
    "sampling_rule",
    "sampling_cell_id",
    "source_decision",
    "requires_posthoc_label_validation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2370)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=580)
    parser.add_argument("--capture-stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=smoke.WIDTH)
    parser.add_argument("--height", type=int, default=smoke.HEIGHT)
    parser.add_argument("--fps", type=int, default=smoke.FPS)
    parser.add_argument("--robot-usd-path", type=Path, default=smoke.DEFAULT_USD)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and arguments, then exit before starting IsaacLab.",
    )
    parser.add_argument(
        "--close-sim-app",
        action="store_true",
        help="Explicitly close Isaac app at the end. Default skips close because local Kit close can hang.",
    )
    return parser.parse_args()


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path, expected_episodes: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    with path.open(newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if len(rows) != int(expected_episodes):
        raise ValueError(f"manifest rows {len(rows)} != expected {expected_episodes}")
    for idx, row in enumerate(rows):
        missing = [field for field in MANIFEST_FIELDS if field not in row]
        if missing:
            raise ValueError(f"manifest row {idx} missing fields {missing}")
        if int(row["episode_index"]) != idx:
            raise ValueError(f"manifest episode_index mismatch at row {idx}: {row['episode_index']}")
        row["episode_index"] = int(row["episode_index"])
        row["cube_x_m"] = float(row["cube_x_m"])
        row["cube_y_m"] = float(row["cube_y_m"])
        row["seed"] = int(row["seed"])
        row["requires_posthoc_label_validation"] = truthy(str(row["requires_posthoc_label_validation"]))
        if row["requires_posthoc_label_validation"] is not True:
            raise ValueError(f"manifest row {idx} must require posthoc label validation")
    return rows


def split_counts(manifest: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest:
        split = str(row["split_candidate"])
        counts[split] = counts.get(split, 0) + 1
    return counts


def attach_manifest_fields(frame_row: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, Any]:
    frame_row.update(
        {
            "split_candidate": str(manifest_row["split_candidate"]),
            "manifest_seed": int(manifest_row["seed"]),
            "sampling_rule": str(manifest_row["sampling_rule"]),
            "sampling_cell_id": str(manifest_row["sampling_cell_id"]),
            "source_decision": str(manifest_row["source_decision"]),
            "requires_posthoc_label_validation": bool(manifest_row["requires_posthoc_label_validation"]),
        }
    )
    return frame_row


def ensure_fresh_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; choose a fresh --out-dir")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.robot_usd_path = args.robot_usd_path.resolve()

    if int(args.expected_episodes) != 100:
        raise ValueError("This chunk renderer is intentionally scoped to exactly 100 episodes")
    if int(args.capture_stride) < 1:
        raise ValueError("--capture-stride must be >= 1")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

    manifest = read_manifest(args.manifest, int(args.expected_episodes))
    manifest_counts = split_counts(manifest)
    if args.validate_only:
        print(
            "[cube10cm-top-view-chunk-render] validate_only "
            f"episodes={len(manifest)} manifest={args.manifest} splits={manifest_counts}",
            flush=True,
        )
        return

    ensure_fresh_out_dir(args.out_dir)
    shutil.copy2(args.manifest, args.out_dir / "episode_manifest.csv")

    episodes_dir = args.out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[cube10cm-top-view-chunk-render] start "
        f"episodes={len(manifest)} steps={args.steps_per_episode} stride={args.capture_stride} "
        f"manifest={args.manifest} out={args.out_dir} splits={manifest_counts}",
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
        from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
        from roarm_rl.train_cube_tap10cm_ppo_smoke import (
            _apply_candidate6_contract,
            _contract_violations,
        )

        first = manifest[0]
        first_args = smoke.base_contract_args(args, float(first["cube_x_m"]), float(first["cube_y_m"]))
        env_cfg = RoArmCubeTap10cmEnvCfg()
        contract = _apply_candidate6_contract(env_cfg, first_args)
        violations = _contract_violations(contract, first_args)
        if violations:
            raise RuntimeError(f"candidate contract violations: {violations}")
        env_cfg.viewer.cam_prim_path = smoke.CAMERA_PATH
        env_cfg.viewer.resolution = (int(args.width), int(args.height))

        env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
        inner_env = env.unwrapped
        print(
            "[cube10cm-top-view-chunk-render] env_ready "
            f"max_episode_length={inner_env.max_episode_length} action_space={inner_env.cfg.action_space}",
            flush=True,
        )

        stage = omni.usd.get_context().get_stage()
        smoke.install_camera(stage, int(args.width), int(args.height))
        raw_frames_dir = args.out_dir / "raw_env_render_frames"
        if raw_frames_dir.exists() and any(raw_frames_dir.glob("*.png")):
            raise FileExistsError(f"{raw_frames_dir} already contains PNG files; choose a new --out-dir")
        raw_frames_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            sim_app.update()

        rows: list[dict[str, Any]] = []
        overall_t0 = time.time()
        with torch.inference_mode():
            for manifest_row in manifest:
                episode_id = int(manifest_row["episode_index"])
                fixed_x = float(manifest_row["cube_x_m"])
                fixed_y = float(manifest_row["cube_y_m"])
                inner_env.cfg.cube_x_min = fixed_x
                inner_env.cfg.cube_x_max = fixed_x
                inner_env.cfg.cube_y_min = fixed_y
                inner_env.cfg.cube_y_max = fixed_y
                env.reset()
                ep_dir = episodes_dir / f"episode_{episode_id:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                episode_t0 = time.time()
                frame_id = 0
                action3 = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)

                rgb = smoke.capture_rgb_from_env(env)
                frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                Image.fromarray(rgb).save(frame_path)
                rows.append(
                    attach_manifest_fields(
                        smoke.frame_metadata(
                            inner_env=inner_env,
                            episode_id=episode_id,
                            frame_id=frame_id,
                            sim_step=0,
                            action3=action3,
                            fps=int(args.fps),
                        ),
                        manifest_row,
                    )
                )

                for sim_step in range(1, int(args.steps_per_episode) + 1):
                    action3 = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)
                    env.step(action3)
                    if sim_step % int(args.capture_stride) != 0 and sim_step != int(args.steps_per_episode):
                        continue
                    frame_id += 1
                    rgb = smoke.capture_rgb_from_env(env)
                    frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                    Image.fromarray(rgb).save(frame_path)
                    rows.append(
                        attach_manifest_fields(
                            smoke.frame_metadata(
                                inner_env=inner_env,
                                episode_id=episode_id,
                                frame_id=frame_id,
                                sim_step=sim_step,
                                action3=action3,
                                fps=int(args.fps),
                            ),
                            manifest_row,
                        )
                    )

                print(
                    "[cube10cm-top-view-chunk-render] episode "
                    f"{episode_id:03d} split={manifest_row['split_candidate']} "
                    f"xy=({fixed_x:.3f},{fixed_y:.3f}) frames={frame_id + 1} "
                    f"elapsed_s={time.time() - episode_t0:.1f}",
                    flush=True,
                )

        elapsed_s = time.time() - overall_t0
        frame_paths = sorted(raw_frames_dir.glob("*.png"))
        smoke.attach_source_pngs_and_visibility(rows, frame_paths)
        frames_jsonl = args.out_dir / "frames.jsonl"
        with frames_jsonl.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        poses = [(float(row["cube_x_m"]), float(row["cube_y_m"])) for row in manifest]
        summary = smoke.summarize(rows, args.out_dir, elapsed_s, poses)
        summary["artifact"] = ARTIFACT
        summary["runtime"] = "ISAAC_RENDER_ONLY_MANIFEST_FED_NO_TRAINING"
        summary["manifest_csv"] = str((args.out_dir / "episode_manifest.csv").relative_to(REPO))
        summary["manifest_split_counts"] = manifest_counts
        summary["contract"] = contract
        summary["contract_violations"] = violations
        summary["frames_jsonl"] = str(frames_jsonl.relative_to(REPO))
        summary_path = args.out_dir / "render_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        print(
            "[cube10cm-top-view-chunk-render] done "
            f"frames={summary['frames']} effective_fps={summary['effective_render_fps']:.3f} "
            f"png_mb_per_ep={summary['debug_png_mb_per_episode']:.2f} summary={summary_path}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        if args.close_sim_app:
            sim_app.close()
        else:
            print(
                "[cube10cm-top-view-chunk-render] skip sim_app.close() "
                "because local Kit close may hang; process exit will release it",
                flush=True,
            )


if __name__ == "__main__":
    main()
