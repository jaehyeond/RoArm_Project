#!/usr/bin/env python3
"""Convert D232 top-view smoke PNG/JSONL output to a LeRobot video dataset.

This script is for the approved small smoke only. It validates:
- LeRobotDataset load/decode through the training data path
- source PNG vs decoded video pixel differences
- frame/episode counts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_smoke_d232"
VIDEO_KEY = "observation.images.top"
WIDTH = 1280
HEIGHT = 720
FPS = 30
TASK = "Tap or push the 10cm 0.72kg cube with the RoArm from the top-view camera."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RENDER_DIR / "lerobot_dataset")
    parser.add_argument("--repo-id", default="roarm_cube10cm_top_view_smoke_d232")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--video-backend", default=None, help="LeRobot video decode backend, e.g. pyav.")
    parser.add_argument("--quality-samples", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing LeRobot dataset without rebuilding it.")
    return parser.parse_args()


def ensure_writable_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/roarm_hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/roarm_hf_datasets_cache")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def read_rows(render_dir: Path) -> list[dict[str, Any]]:
    frames_jsonl = render_dir / "frames.jsonl"
    if not frames_jsonl.exists():
        raise FileNotFoundError(frames_jsonl)
    rows = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (int(row["episode_id"]), int(row["frame_id"])))
    for idx, row in enumerate(rows):
        row["global_index"] = idx
    return rows


def sample_indices(total: int, count: int) -> list[int]:
    count = max(1, min(int(count), int(total)))
    return sorted({round(i * (total - 1) / max(1, count - 1)) for i in range(count)})


def source_png(row: dict[str, Any]) -> Path:
    return REPO / row["source_png"]


def ffprobe_frames(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    text = proc.stdout.strip()
    return int(text) if text.isdigit() else None


def tensor_image_to_uint8(image: Any) -> Any:
    import numpy as np

    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype.kind == "f":
        arr = np.clip(arr * 255.0, 0.0, 255.0)
    return arr.astype(np.uint8)


def build_dataset(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_writable_hf_cache()
    if args.out_dir.exists():
        raise FileExistsError(f"{args.out_dir} exists; choose a new --out-dir")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    import numpy as np

    features = {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": [f"joint_{i}" for i in range(6)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": [f"joint_target_{i}" for i in range(6)]},
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),
        root=str(args.out_dir),
        features=features,
        robot_type="roarm_m3",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        vcodec=str(args.vcodec),
    )

    t0 = time.time()
    episodes = sorted({int(row["episode_id"]) for row in rows})
    for episode_id in episodes:
        ep_rows = [row for row in rows if int(row["episode_id"]) == episode_id]
        for row in ep_rows:
            image = np.array(Image.open(source_png(row)).convert("RGB"))
            ds.add_frame(
                {
                    VIDEO_KEY: image,
                    "observation.state": np.asarray(row["observation_state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": TASK,
                }
            )
        ds.save_episode()
        print(
            f"[cube10cm-smoke-to-lerobot] saved episode={episode_id} frames={len(ep_rows)}",
            flush=True,
        )
    if hasattr(ds, "finalize"):
        ds.finalize()
    return {
        "build_elapsed_s": time.time() - t0,
        "episodes": len(episodes),
        "source_frames": len(rows),
    }


def validate_dataset(args: argparse.Namespace, rows: list[dict[str, Any]], build_info: dict[str, Any]) -> dict[str, Any]:
    ensure_writable_hf_cache()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    import numpy as np

    t0 = time.time()
    ds = LeRobotDataset(args.repo_id, root=str(args.out_dir), video_backend=args.video_backend)
    total_frames = int(ds.meta.total_frames)
    total_episodes = int(ds.meta.total_episodes)
    indices = sample_indices(total_frames, int(args.quality_samples))
    samples: list[dict[str, Any]] = []
    decode_times = []
    for idx in indices:
        ts = time.time()
        item = ds[int(idx)]
        decode_s = time.time() - ts
        decode_times.append(decode_s)
        decoded = tensor_image_to_uint8(item[VIDEO_KEY])
        src = np.array(Image.open(source_png(rows[int(idx)])).convert("RGB"), dtype=np.uint8)
        if decoded.shape != src.shape:
            raise RuntimeError(f"decoded/source shape mismatch at {idx}: {decoded.shape} != {src.shape}")
        diff = np.abs(decoded.astype(np.int16) - src.astype(np.int16))
        samples.append(
            {
                "index": int(idx),
                "episode_id": int(rows[int(idx)]["episode_id"]),
                "frame_id": int(rows[int(idx)]["frame_id"]),
                "decode_s": decode_s,
                "image_shape": list(decoded.shape),
                "state_shape": list(item["observation.state"].shape),
                "action_shape": list(item["action"].shape),
                "pixel_abs_mean": float(diff.mean()),
                "pixel_abs_max": int(diff.max()),
                "source_png": rows[int(idx)]["source_png"],
            }
        )

    video_files = sorted(args.out_dir.glob("videos/**/*.mp4"))
    video_sizes = {str(path.relative_to(args.out_dir)): path.stat().st_size for path in video_files}
    video_frame_counts = {str(path.relative_to(args.out_dir)): ffprobe_frames(path) for path in video_files}
    info_path = args.out_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    feature_info = info.get("features", {}).get(VIDEO_KEY, {}).get("info", {})
    video_bytes = sum(video_sizes.values())
    mb_per_episode = (video_bytes / 1_000_000.0) / max(1, total_episodes)

    result = {
        "artifact": "cube10cm_top_view_visual_smoke_d232_lerobot",
        "runtime": "LEROBOT_CONVERSION_AND_DATALOADER_VALIDATION_ONLY",
        "repo_id": args.repo_id,
        "dataset_root": str(args.out_dir),
        "video_key": VIDEO_KEY,
        "requested_vcodec": str(args.vcodec),
        "video_backend": args.video_backend,
        "info_codec": feature_info.get("video.codec"),
        "info_pix_fmt": feature_info.get("video.pix_fmt"),
        "info_fps": feature_info.get("video.fps"),
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "source_frames": len(rows),
        "frame_count_match": bool(total_frames == len(rows)),
        "video_files": video_sizes,
        "video_frame_counts": video_frame_counts,
        "video_bytes_total": int(video_bytes),
        "video_mb_per_episode": mb_per_episode,
        "video_projected_gb": {
            "100_ep": mb_per_episode * 100.0 / 1000.0,
            "1000_ep": mb_per_episode * 1000.0 / 1000.0,
            "10000_ep": mb_per_episode * 10000.0 / 1000.0,
        },
        "samples": samples,
        "avg_decode_s": float(sum(decode_times) / max(1, len(decode_times))),
        "max_decode_s": float(max(decode_times) if decode_times else 0.0),
        "pixel_abs_mean_max": float(max((sample["pixel_abs_mean"] for sample in samples), default=0.0)),
        "pixel_abs_max_max": int(max((sample["pixel_abs_max"] for sample in samples), default=0)),
        "build_info": build_info,
        "validation_elapsed_s": time.time() - t0,
        "status": "PASS",
    }
    if not result["frame_count_match"]:
        result["status"] = "FAIL"
    return result


def main() -> None:
    args = parse_args()
    rows = read_rows(args.render_dir)
    print(
        "[cube10cm-smoke-to-lerobot] start "
        f"rows={len(rows)} render_dir={args.render_dir} out_dir={args.out_dir} vcodec={args.vcodec}",
        flush=True,
    )
    if args.validate_only:
        if not args.out_dir.exists():
            raise FileNotFoundError(args.out_dir)
        build_info = {
            "build_skipped": True,
            "reason": "validate_only_existing_lerobot_dataset",
            "episodes": len({int(row["episode_id"]) for row in rows}),
            "source_frames": len(rows),
        }
    else:
        build_info = build_dataset(args, rows)
    result = validate_dataset(args, rows, build_info)
    out_json = args.render_dir / "lerobot_validation_summary.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "[cube10cm-smoke-to-lerobot] done "
        f"status={result['status']} frames={result['total_frames']} "
        f"episodes={result['total_episodes']} codec={result.get('info_codec')} "
        f"pixel_mean_max={result['pixel_abs_mean_max']:.3f} "
        f"pixel_abs_max={result['pixel_abs_max_max']} out={out_json}",
        flush=True,
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
