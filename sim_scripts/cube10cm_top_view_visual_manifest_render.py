#!/usr/bin/env python3
"""Render a manifest-fed top-view visual dataset chunk for the 10cm cube task.

This is the general manifest renderer prepared after the D241 label-aware
0-999 manifest. It supports validate-only checks without launching IsaacLab.
Actual rendering requires both omitting --validate-only and passing
--render-approved.
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
DEFAULT_MANIFEST = LOG_DIR / "cube10cm_top_view_labelaware_manifest_0_999_d241" / "episode_manifest.csv"
DEFAULT_OUT = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_VALIDATION_SUMMARY = LOG_DIR / "cube10cm_top_view_visual_manifest_render_validate_d242.json"
ARTIFACT = "cube10cm_top_view_visual_manifest_render_d242"

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

INTENT_FIELDS = [
    "intended_sampling_bucket",
    "intended_role",
    "camera_coverage_required",
    "expected_postrender_labels",
    "label_policy",
]

FORBIDDEN_FINAL_LABEL_FIELDS = {
    "label_useful_clean_numeric",
    "label_overshoot_numeric",
    "label_camera_contract_numeric",
    "label_status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2420)
    parser.add_argument("--expected-episodes", type=int, default=1000)
    parser.add_argument("--steps-per-episode", type=int, default=580)
    parser.add_argument("--capture-stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=smoke.WIDTH)
    parser.add_argument("--height", type=int, default=smoke.HEIGHT)
    parser.add_argument("--fps", type=int, default=smoke.FPS)
    parser.add_argument("--robot-usd-path", type=Path, default=smoke.DEFAULT_USD)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and arguments, write summary, then exit before IsaacLab.",
    )
    parser.add_argument(
        "--render-approved",
        action="store_true",
        help="Required for actual rendering. This does not bypass validate-only.",
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
        forbidden = sorted(FORBIDDEN_FINAL_LABEL_FIELDS.intersection(row.keys()))
        if forbidden:
            raise ValueError(f"manifest row {idx} contains final label fields {forbidden}")
        if int(row["episode_index"]) != idx:
            raise ValueError(f"manifest episode_index mismatch at row {idx}: {row['episode_index']}")
        row["episode_index"] = int(row["episode_index"])
        row["cube_x_m"] = float(row["cube_x_m"])
        row["cube_y_m"] = float(row["cube_y_m"])
        row["seed"] = int(row["seed"])
        row["requires_posthoc_label_validation"] = truthy(str(row["requires_posthoc_label_validation"]))
        if row["requires_posthoc_label_validation"] is not True:
            raise ValueError(f"manifest row {idx} must require posthoc label validation")
        if "camera_coverage_required" in row:
            row["camera_coverage_required"] = truthy(str(row["camera_coverage_required"]))
    return rows


def counts_for(manifest: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest:
        if field not in row:
            continue
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return counts


def range_summary(manifest: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in manifest]
    return {"min": min(values), "max": max(values)}


def out_dir_empty(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def validation_summary(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(row["seed"]) for row in manifest]
    camera_flags = [
        bool(row.get("camera_coverage_required"))
        for row in manifest
        if "camera_coverage_required" in row
    ]
    return {
        "artifact": ARTIFACT,
        "runtime": "VALIDATE_ONLY_NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING",
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "expected_episodes": int(args.expected_episodes),
        "rows": len(manifest),
        "episode_index_range": [manifest[0]["episode_index"], manifest[-1]["episode_index"]],
        "split_counts": counts_for(manifest, "split_candidate"),
        "intended_bucket_counts": counts_for(manifest, "intended_sampling_bucket"),
        "seed_unique": len(seeds) == len(set(seeds)),
        "all_requires_posthoc_label_validation": all(
            row["requires_posthoc_label_validation"] is True for row in manifest
        ),
        "all_camera_coverage_required": all(camera_flags) if camera_flags else None,
        "x_range_m": range_summary(manifest, "cube_x_m"),
        "y_range_m": range_summary(manifest, "cube_y_m"),
        "manifest_fields_required": MANIFEST_FIELDS,
        "intent_fields_supported": INTENT_FIELDS,
        "forbidden_final_label_fields": sorted(FORBIDDEN_FINAL_LABEL_FIELDS),
        "robot_usd_path": str(args.robot_usd_path),
        "robot_usd_exists": args.robot_usd_path.exists(),
        "out_dir_exists": args.out_dir.exists(),
        "out_dir_empty": out_dir_empty(args.out_dir),
        "steps_per_episode": int(args.steps_per_episode),
        "capture_stride": int(args.capture_stride),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "render_approved": bool(args.render_approved),
        "status": "PASS",
        "warnings": [
            "validate-only does not launch IsaacLab and does not render images",
            "actual render still requires explicit runtime approval, disk/output-root preflight, and --render-approved",
            "final clean/overshoot/camera labels must be assigned only by post-render numeric validation",
        ],
    }


def write_validation_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
    for field in INTENT_FIELDS:
        if field in manifest_row:
            frame_row[field] = manifest_row[field]
    return frame_row


def ensure_fresh_out_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; choose a fresh --out-dir")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.validation_summary = args.validation_summary.resolve()
    args.robot_usd_path = args.robot_usd_path.resolve()

    if int(args.expected_episodes) < 1:
        raise ValueError("--expected-episodes must be >= 1")
    if int(args.capture_stride) < 1:
        raise ValueError("--capture-stride must be >= 1")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

    manifest = read_manifest(args.manifest, int(args.expected_episodes))
    summary = validation_summary(args, manifest)
    write_validation_summary(args.validation_summary, summary)

    if args.validate_only:
        print(
            "[cube10cm-top-view-manifest-render] validate_only "
            f"status={summary['status']} episodes={len(manifest)} manifest={args.manifest} "
            f"summary={args.validation_summary}",
            flush=True,
        )
        return

    if not args.render_approved:
        raise RuntimeError("actual render requires --render-approved plus explicit runtime approval")

    ensure_fresh_out_dir(args.out_dir)
    shutil.copy2(args.manifest, args.out_dir / "episode_manifest.csv")
    episodes_dir = args.out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[cube10cm-top-view-manifest-render] start "
        f"episodes={len(manifest)} steps={args.steps_per_episode} stride={args.capture_stride} "
        f"manifest={args.manifest} out={args.out_dir} splits={summary['split_counts']}",
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
            "[cube10cm-top-view-manifest-render] env_ready "
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
                    "[cube10cm-top-view-manifest-render] episode "
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
        render_summary = smoke.summarize(rows, args.out_dir, elapsed_s, poses)
        render_summary["artifact"] = ARTIFACT
        render_summary["runtime"] = "ISAAC_RENDER_ONLY_MANIFEST_FED_NO_TRAINING"
        render_summary["manifest_csv"] = str((args.out_dir / "episode_manifest.csv").relative_to(REPO))
        render_summary["manifest_split_counts"] = summary["split_counts"]
        render_summary["manifest_intended_bucket_counts"] = summary["intended_bucket_counts"]
        render_summary["contract"] = contract
        render_summary["contract_violations"] = violations
        render_summary["frames_jsonl"] = str(frames_jsonl.relative_to(REPO))
        summary_path = args.out_dir / "render_summary.json"
        summary_path.write_text(json.dumps(render_summary, indent=2, sort_keys=True) + "\n")

        print(
            "[cube10cm-top-view-manifest-render] done "
            f"frames={render_summary['frames']} effective_fps={render_summary['effective_render_fps']:.3f} "
            f"png_mb_per_ep={render_summary['debug_png_mb_per_episode']:.2f} summary={summary_path}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        if args.close_sim_app:
            sim_app.close()
        else:
            print(
                "[cube10cm-top-view-manifest-render] skip sim_app.close() "
                "because local Kit close may hang; process exit will release it",
                flush=True,
            )


if __name__ == "__main__":
    main()
