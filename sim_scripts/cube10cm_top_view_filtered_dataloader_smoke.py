#!/usr/bin/env python3
"""Smoke-test filtered LeRobot views without training.

This verifies that the frozen split frame indices can be used to read the
LeRobot dataset through the same decoding path that training would rely on.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"
VIDEO_KEY = "observation.images.top"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--filtered-views-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_writable_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/roarm_hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/roarm_hf_datasets_cache")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def read_indices(path: Path) -> list[int]:
    with path.open() as f:
        return [int(line.strip()) for line in f if line.strip()]


def read_view_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(row["global_index"]): row for row in rows}


def sample_indices(indices: list[int]) -> list[int]:
    if not indices:
        return []
    picks = [0, len(indices) // 2, len(indices) - 1]
    return [indices[pos] for pos in sorted(set(picks))]


def shape_of(value: Any) -> list[int]:
    return [int(dim) for dim in getattr(value, "shape", [])]


def scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def main() -> None:
    args = parse_args()
    ensure_writable_hf_cache()

    render_dir = args.render_dir
    filtered_dir = args.filtered_views_dir or (render_dir / "filtered_views_d250")
    out_dir = args.out_dir or (filtered_dir / "dataloader_smoke_d251")
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} exists; use --force or another --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    lerobot_validation = json.loads((render_dir / "lerobot_validation_summary.json").read_text())
    filtered_summary = json.loads((filtered_dir / "filtered_views_summary.json").read_text())
    repo_id = lerobot_validation["repo_id"]
    dataset_root = REPO / filtered_summary["lerobot_root"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=str(dataset_root), video_backend=args.video_backend)
    if int(ds.meta.total_frames) != int(lerobot_validation["total_frames"]):
        raise RuntimeError("LeRobot total frame count mismatch")
    if int(ds.meta.total_episodes) != int(lerobot_validation["total_episodes"]):
        raise RuntimeError("LeRobot total episode count mismatch")

    results: dict[str, Any] = {
        "artifact": "cube10cm_top_view_filtered_dataloader_smoke_d251",
        "runtime": "NO_TRAINING_FILTERED_DATALOADER_DECODE_ONLY",
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "video_backend": args.video_backend,
        "total_frames": int(ds.meta.total_frames),
        "total_episodes": int(ds.meta.total_episodes),
        "splits": {},
        "status": "PASS",
    }

    total_filtered_frames = 0
    for split_name, split_info in sorted(filtered_summary["views"].items()):
        indices = read_indices(Path(split_info["frame_indices_txt"]))
        view_rows = read_view_rows(Path(split_info["frame_view_csv"]))
        total_filtered_frames += len(indices)
        samples = []
        for global_index in sample_indices(indices):
            expected = view_rows[global_index]
            t0 = time.time()
            item = ds[global_index]
            decode_s = time.time() - t0
            image_shape = shape_of(item[VIDEO_KEY])
            state_shape = shape_of(item["observation.state"])
            action_shape = shape_of(item["action"])
            item_episode = scalar_int(item["episode_index"])
            item_frame = scalar_int(item["frame_index"])
            if item_episode != int(expected["episode_index"]):
                raise RuntimeError(f"{split_name} episode mismatch at {global_index}")
            if item_frame != int(expected["frame_index"]):
                raise RuntimeError(f"{split_name} frame mismatch at {global_index}")
            if state_shape != [6] or action_shape != [6]:
                raise RuntimeError(f"{split_name} state/action shape mismatch at {global_index}")
            if image_shape not in ([3, 720, 1280], [720, 1280, 3]):
                raise RuntimeError(f"{split_name} image shape mismatch at {global_index}: {image_shape}")
            samples.append(
                {
                    "global_index": int(global_index),
                    "episode_index": item_episode,
                    "frame_index": item_frame,
                    "decode_s": decode_s,
                    "image_shape": image_shape,
                    "state_shape": state_shape,
                    "action_shape": action_shape,
                    "label_status": expected["label_status"],
                    "package_subsplit": expected["package_subsplit"],
                }
            )
        results["splits"][split_name] = {
            "korean_definition": split_info["korean_definition"],
            "episodes": int(split_info["episodes"]),
            "frames": len(indices),
            "samples": samples,
            "sample_count": len(samples),
            "avg_sample_decode_s": sum(sample["decode_s"] for sample in samples) / max(1, len(samples)),
            "max_sample_decode_s": max((sample["decode_s"] for sample in samples), default=0.0),
        }

    if total_filtered_frames != int(ds.meta.total_frames):
        raise RuntimeError(f"filtered views do not cover all frames exactly once: {total_filtered_frames}")

    out_json = out_dir / "filtered_dataloader_smoke_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        "[cube10cm-filtered-dataloader-smoke] done "
        f"status=PASS splits={len(results['splits'])} frames={total_filtered_frames} out={out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
