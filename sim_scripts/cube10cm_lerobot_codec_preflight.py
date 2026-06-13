#!/usr/bin/env python3
"""Read-only LeRobot video codec preflight using an existing dataset.

This script does not render, generate a dataset, train, or modify the input
dataset. It verifies whether the installed LeRobot data path can decode existing
video frames through `LeRobotDataset`, independent of helper extraction tools.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO / "lerobot_dataset_v6")
    parser.add_argument("--repo-id", default="roarm_m3_pick_v6_codec_preflight")
    parser.add_argument("--video-key", default="observation.images.top")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=LOG_DIR / "cube10cm_lerobot_codec_preflight_v6_d232.json",
    )
    return parser.parse_args()


def _shape(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return [int(x) for x in shape]


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", type(value).__name__))


def _minmax(value: Any) -> list[float] | None:
    try:
        return [float(value.min()), float(value.max())]
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    info_path = args.dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feature_info = info["features"][args.video_key].get("info", {})

    result: dict[str, Any] = {
        "artifact": "cube10cm_lerobot_codec_preflight_v6_d232",
        "runtime": "NO_RENDER_NO_DATASET_NO_TRAINING",
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "video_key": args.video_key,
        "info_codec": feature_info.get("video.codec"),
        "info_pix_fmt": feature_info.get("video.pix_fmt"),
        "info_fps": feature_info.get("video.fps"),
        "info_shape": info["features"][args.video_key].get("shape"),
        "status": "UNKNOWN",
        "samples": [],
    }

    t0 = time.time()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, root=str(args.dataset_root))
        total = int(ds.meta.total_frames)
        sample_count = max(1, min(int(args.samples), total))
        indices = sorted({round(i * (total - 1) / max(1, sample_count - 1)) for i in range(sample_count)})
        result["total_frames"] = total
        result["total_episodes"] = int(ds.meta.total_episodes)

        decode_times = []
        for idx in indices:
            ts = time.time()
            item = ds[int(idx)]
            elapsed = time.time() - ts
            image = item[args.video_key]
            sample = {
                "index": int(idx),
                "decode_s": elapsed,
                "image_shape": _shape(image),
                "image_dtype": _dtype(image),
                "image_minmax": _minmax(image),
                "state_shape": _shape(item.get("observation.state")),
                "action_shape": _shape(item.get("action")),
            }
            decode_times.append(elapsed)
            result["samples"].append(sample)

        result["avg_decode_s"] = sum(decode_times) / len(decode_times)
        result["max_decode_s"] = max(decode_times)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["elapsed_s"] = time.time() - t0
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n")

    print("line1 artifact=cube10cm_lerobot_codec_preflight_v6_d232 runtime=NO_RENDER_NO_DATASET_NO_TRAINING")
    print(
        "line2 dataset "
        f"root={args.dataset_root} codec={result.get('info_codec')} "
        f"pix_fmt={result.get('info_pix_fmt')} fps={result.get('info_fps')} "
        f"shape={result.get('info_shape')}"
    )
    print(
        "line3 verdict "
        f"status={result['status']} avg_decode_s={result.get('avg_decode_s')} "
        f"max_decode_s={result.get('max_decode_s')} samples={len(result.get('samples', []))} "
        f"error={result.get('error_type')}"
    )
    print(f"line4 out_json={args.out_json}")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
