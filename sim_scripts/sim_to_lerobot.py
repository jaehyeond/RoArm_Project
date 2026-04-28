#!/usr/bin/env python3
"""Step E — Convert sim_renders_v2 PNGs into a LeRobot v3 dataset (sim_v1).

Strategy:
  - meta/info.json, meta/tasks.parquet, data/chunk-000/file-000.parquet: copy from v6 (replay)
  - videos/observation.images.top/chunk-000/file-000.mp4: build from sim PNGs via av1_nvenc
  - meta/episodes/chunk-000/file-000.parquet: copy + overwrite stats/observation.images.top/* per-ep
  - meta/stats.json: copy + overwrite observation.images.top (aggregate)

Run:
  conda run -n roarm python sim_scripts/sim_to_lerobot.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
V6 = REPO_ROOT / "lerobot_dataset_v6"
RENDERS = REPO_ROOT / "sim_renders_v2"
SIM = REPO_ROOT / "sim_v1"
REPO_ID = "roarm_m3_pick_sim"

VIDEO_KEY = "observation.images.top"
MP4_REL = Path("videos") / VIDEO_KEY / "chunk-000" / "file-000.mp4"
EPISODES_REL = Path("meta/episodes/chunk-000/file-000.parquet")
DATA_REL = Path("data/chunk-000/file-000.parquet")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copy_static_files():
    (SIM / "meta").mkdir(parents=True, exist_ok=True)
    (SIM / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for rel in ["meta/info.json", "meta/tasks.parquet", str(DATA_REL)]:
        src, dst = V6 / rel, SIM / rel
        shutil.copy2(src, dst)
        log(f"copied {rel}  ({dst.stat().st_size} bytes)")


def encode_mp4(framerate: int = 30, bitrate: str = "5M") -> Path:
    out = SIM / MP4_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log(f"MP4 exists, skipping encode: {out} ({out.stat().st_size} bytes)")
        return out

    # Glob pattern: lex-order of episode_NNN/frame_NNNN guarantees correct concatenation
    # (verified: episode_000..episode_049, frame_0000..frame_NNNN zero-padded)
    pattern = str(RENDERS / "episode_*" / "frame_*.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-pattern_type", "glob",
        "-i", pattern,
        "-c:v", "av1_nvenc",
        "-b:v", bitrate,
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    log("encoding MP4: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    log(f"MP4 encoded in {time.time()-t0:.1f}s → {out} ({out.stat().st_size/1e6:.1f} MB)")

    # Verify frame count via ffprobe
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", str(out)],
            capture_output=True, text=True, check=True,
        )
        n = int(r.stdout.strip())
        log(f"ffprobe nb_read_frames={n} (expected 6942)")
        if n != 6942:
            raise RuntimeError(f"MP4 frame count mismatch: {n} != 6942")
    except FileNotFoundError:
        log("WARN: ffprobe not found, skipping frame-count verification")
    return out


def compute_per_ep_image_stats():
    from lerobot.datasets.compute_stats import compute_episode_stats
    features = {VIDEO_KEY: {"dtype": "image"}}
    per_ep = []
    t0 = time.time()
    for ep in range(50):
        render_dir = RENDERS / f"episode_{ep:03d}"
        paths = sorted(render_dir.glob("frame_*.png"))
        if not paths:
            raise RuntimeError(f"No PNGs in {render_dir}")
        stats = compute_episode_stats({VIDEO_KEY: [str(p) for p in paths]}, features)
        per_ep.append(stats)
        if (ep + 1) % 10 == 0:
            log(f"  ep stats computed: {ep+1}/50  ({time.time()-t0:.1f}s elapsed)")
    log(f"all per-ep image stats done in {time.time()-t0:.1f}s")
    return per_ep


def update_episodes_parquet(per_ep_stats: list[dict]):
    src = V6 / EPISODES_REL
    dst = SIM / EPISODES_REL
    dst.parent.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(src)
    cols = t.to_pydict()

    img_keys = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]
    for key in img_keys:
        col_name = f"stats/{VIDEO_KEY}/{key}"
        new_vals = []
        for ep in range(50):
            arr = per_ep_stats[ep][VIDEO_KEY][key]  # shape (3, 1, 1)
            if arr.shape != (3, 1, 1):
                raise RuntimeError(f"ep{ep} {key}: shape {arr.shape} != (3,1,1)")
            new_vals.append(arr.tolist())
        cols[col_name] = new_vals

    count_col = f"stats/{VIDEO_KEY}/count"
    cols[count_col] = [
        per_ep_stats[ep][VIDEO_KEY]["count"].tolist() for ep in range(50)
    ]

    new_t = pa.Table.from_pydict(cols, schema=t.schema)
    pq.write_table(new_t, dst)
    log(f"episodes parquet written: {dst} ({dst.stat().st_size} bytes)")


def update_stats_json(per_ep_stats: list[dict]):
    from lerobot.datasets.compute_stats import aggregate_stats

    agg = aggregate_stats(per_ep_stats)

    with open(V6 / "meta/stats.json") as f:
        stats_json = json.load(f)

    agg_img = agg[VIDEO_KEY]
    img_out = {}
    for k, v in agg_img.items():
        if isinstance(v, np.ndarray):
            img_out[k] = v.tolist()
        else:
            img_out[k] = v
    stats_json[VIDEO_KEY] = img_out

    dst = SIM / "meta/stats.json"
    with open(dst, "w") as f:
        json.dump(stats_json, f, indent=4)
    log(f"stats.json written: {dst} ({dst.stat().st_size} bytes)")


def validate():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_ID, root=str(SIM))
    log(f"LeRobotDataset loaded: n_eps={ds.meta.total_episodes}, n_frames={ds.meta.total_frames}")
    item = ds[0]
    log(f"item[0] keys: {list(item.keys())}")
    img = item[VIDEO_KEY]
    state = item["observation.state"]
    action = item["action"]
    log(f"  {VIDEO_KEY}: shape={tuple(img.shape)} dtype={img.dtype} "
        f"range=[{img.min():.4f}, {img.max():.4f}]")
    log(f"  observation.state: shape={tuple(state.shape)} dtype={state.dtype}")
    log(f"  action: shape={tuple(action.shape)} dtype={action.dtype}")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-encode", action="store_true",
                        help="Skip MP4 encoding (use existing file)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip LeRobotDataset validation")
    args = parser.parse_args()

    log(f"REPO_ROOT = {REPO_ROOT}")
    log(f"V6 = {V6} (exists={V6.exists()})")
    log(f"RENDERS = {RENDERS} (exists={RENDERS.exists()})")
    log(f"SIM = {SIM}")

    log("=== Step E.1: copy static files ===")
    copy_static_files()

    log("=== Step E.2: encode MP4 (av1_nvenc) ===")
    if not args.skip_encode:
        encode_mp4()

    log("=== Step E.3: compute per-episode sim image stats ===")
    per_ep = compute_per_ep_image_stats()

    log("=== Step E.4: update episodes parquet ===")
    update_episodes_parquet(per_ep)

    log("=== Step E.5: update stats.json ===")
    update_stats_json(per_ep)

    if not args.skip_validate:
        log("=== Step E.6: validate sim_v1 ===")
        validate()

    log("=== DONE ===")


if __name__ == "__main__":
    main()
