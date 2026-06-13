#!/usr/bin/env python3
"""Extract one PNG from a LeRobot video dataset by episode_id and frame_id."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--video-key", default="observation.images.top")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    return parser.parse_args()


def load_episode_row(dataset: Path, episode_id: int):
    import pandas as pd

    episode_files = sorted((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        return None
    rows = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
    match = rows[rows["episode_index"] == episode_id]
    if match.empty:
        raise SystemExit(f"episode_id {episode_id} not found in {dataset / 'meta/episodes'}")
    return match.iloc[0]


def resolve_video(dataset: Path, video_key: str, episode_id: int, frame_id: int) -> tuple[Path, int, int]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    fps = int(round(float(info.get("fps", 30))))
    pattern = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    row = load_episode_row(dataset, episode_id)
    if row is None or "{episode_index" in pattern:
        chunk_size = int(info.get("chunks_size", 1000))
        rel = pattern.format(
            video_key=video_key,
            episode_index=episode_id,
            episode_chunk=episode_id // chunk_size,
        )
        return dataset / rel, frame_id, fps

    length = int(row["length"])
    if frame_id < 0 or frame_id >= length:
        raise SystemExit(f"frame_id {frame_id} outside episode {episode_id} length {length}")

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_ts_col = f"videos/{video_key}/from_timestamp"
    rel = pattern.format(
        video_key=video_key,
        chunk_index=int(row[chunk_col]),
        file_index=int(row[file_col]),
    )
    local_frame = int(round(float(row[from_ts_col]) * fps)) + frame_id
    return dataset / rel, local_frame, fps


def main() -> None:
    args = parse_args()
    video_path, local_frame, _fps = resolve_video(args.dataset, args.video_key, args.episode_id, args.frame_id)
    out = args.out or Path("extracted_frames") / f"episode_{args.episode_id:06d}_frame_{args.frame_id:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found; install ffmpeg or imageio-ffmpeg")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{local_frame})",
        "-frames:v",
        "1",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {out} from {video_path} local_frame={local_frame}")


if __name__ == "__main__":
    main()
