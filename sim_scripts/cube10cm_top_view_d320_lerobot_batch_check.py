#!/usr/bin/env python3
"""Load one DataLoader batch from the D320 LeRobot smoke dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_ROOT = (
    RUNTIME_ROOT
    / "data_conveyor_d320/replay_smoke/render_d319_replay_smoke/lerobot_dataset"
)
DEFAULT_OUT = (
    RUNTIME_ROOT
    / "data_conveyor_d320/replay_smoke/render_d319_replay_smoke/dataloader_batch_validation_d320.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default="roarm_cube10cm_top_view_d320_replay_smoke")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def shape_of(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return [int(dim) for dim in shape]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/tmp/roarm_hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/roarm_hf_datasets_cache")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    ds = LeRobotDataset(args.repo_id, root=str(args.dataset_root), video_backend=args.video_backend)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    summary = {
        "artifact": "d320_lerobot_dataloader_batch_validation",
        "runtime": "LEROBOT_DATALOADER_ONE_BATCH",
        "dataset_root": str(args.dataset_root),
        "repo_id": str(args.repo_id),
        "video_backend": str(args.video_backend),
        "total_frames": int(ds.meta.total_frames),
        "total_episodes": int(ds.meta.total_episodes),
        "batch_size": int(args.batch_size),
        "batch_keys": sorted(str(key) for key in batch.keys()),
        "batch_shapes": {str(key): shape_of(value) for key, value in batch.items()},
        "status": "PASS",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "[d320-lerobot-batch-check] done "
        f"frames={summary['total_frames']} episodes={summary['total_episodes']} "
        f"keys={summary['batch_keys']} out={args.out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
