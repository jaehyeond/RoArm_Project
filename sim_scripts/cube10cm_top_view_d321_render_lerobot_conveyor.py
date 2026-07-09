#!/usr/bin/env python3
"""Chunk-render D321 accepted rows and append them into one LeRobot dataset.

Run this from the `lerobot` environment. It calls the D320 Isaac replay renderer
as a subprocess for each chunk, appends the rendered frames into one LeRobot v3
dataset, validates the dataset through DataLoader, and deletes raw PNG frames
after each chunk passes append checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_ACCEPTED = RUNTIME_ROOT / "data_conveyor_d321" / "audit" / "d321_accepted_env_rows.csv"
DEFAULT_OUT = RUNTIME_ROOT / "data_conveyor_d321" / "render_lerobot_v1"
RENDERER = REPO / "sim_scripts" / "cube10cm_top_view_d320_replay_render.py"
VIDEO_KEY = "observation.images.top"
WIDTH = 1280
HEIGHT = 720
FPS = 30
TASK = "Tap or push the 10cm 0.72kg cube with the RoArm from the top-view camera."


# Measured in D320: 344,057,130 raw PNG bytes / 9 episodes.
RAW_BYTES_PER_EP_EST = 38_230_000
# Measured in D320 LeRobot AV1 conversion: about 0.418 MB / episode.
VIDEO_BYTES_PER_EP_EST = 420_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-csv", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", default="roarm_cube10cm_top_view_d321_script_v2_low_mid")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-total-episodes", type=int, default=None)
    parser.add_argument("--free-margin", type=float, default=0.20)
    parser.add_argument("--render-conda-env", default="isaaclab")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/roarm_hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/roarm_hf_datasets_cache")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": rel(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "total_gb": usage.total / 1_000_000_000.0,
        "used_gb": usage.used / 1_000_000_000.0,
        "free_gb": usage.free / 1_000_000_000.0,
        "free_rate": usage.free / max(1, usage.total),
    }


def adjust_plan(args: argparse.Namespace, total_rows: int) -> dict[str, Any]:
    requested_chunk = max(1, int(args.chunk_size))
    requested_total = total_rows if args.max_total_episodes is None else min(total_rows, int(args.max_total_episodes))
    disk = disk_snapshot(REPO)
    video_total = requested_total * VIDEO_BYTES_PER_EP_EST
    raw_chunk = requested_chunk * RAW_BYTES_PER_EP_EST
    required_peak = video_total + raw_chunk
    required_with_margin = required_peak * (1.0 + float(args.free_margin))
    adjusted_chunk = requested_chunk
    adjusted_total = requested_total
    reason = "no_scale_down"

    if disk["free_bytes"] < required_with_margin:
        available_for_raw = disk["free_bytes"] / (1.0 + float(args.free_margin)) - video_total
        adjusted_chunk = max(10, int(available_for_raw // RAW_BYTES_PER_EP_EST))
        if adjusted_chunk < 10 or adjusted_chunk > requested_chunk:
            adjusted_chunk = max(10, min(requested_chunk, adjusted_chunk))
        required_peak = video_total + adjusted_chunk * RAW_BYTES_PER_EP_EST
        required_with_margin = required_peak * (1.0 + float(args.free_margin))
        reason = "chunk_size_scaled_for_disk_margin"

    if disk["free_bytes"] < required_with_margin:
        available_for_video = disk["free_bytes"] / (1.0 + float(args.free_margin)) - adjusted_chunk * RAW_BYTES_PER_EP_EST
        adjusted_total = max(0, int(available_for_video // VIDEO_BYTES_PER_EP_EST))
        adjusted_total = min(adjusted_total, requested_total)
        reason = "total_episodes_scaled_for_disk_margin"
        required_peak = adjusted_total * VIDEO_BYTES_PER_EP_EST + adjusted_chunk * RAW_BYTES_PER_EP_EST
        required_with_margin = required_peak * (1.0 + float(args.free_margin))

    if adjusted_total <= 0:
        raise RuntimeError(
            "disk free space cannot satisfy even the minimum D321 render chunk "
            f"with margin={args.free_margin}: {disk}"
        )

    return {
        "disk_before": disk,
        "requested_total_episodes": requested_total,
        "planned_total_episodes": adjusted_total,
        "requested_chunk_size": requested_chunk,
        "planned_chunk_size": adjusted_chunk,
        "estimated_video_total_bytes": int(adjusted_total * VIDEO_BYTES_PER_EP_EST),
        "estimated_raw_chunk_bytes": int(adjusted_chunk * RAW_BYTES_PER_EP_EST),
        "estimated_peak_bytes": int(required_peak),
        "estimated_required_with_margin_bytes": int(required_with_margin),
        "scale_reason": reason,
    }


def source_png(row: dict[str, Any]) -> Path:
    return REPO / str(row["source_png"])


def read_frame_rows(render_dir: Path) -> list[dict[str, Any]]:
    frames_jsonl = render_dir / "frames.jsonl"
    if not frames_jsonl.exists():
        raise FileNotFoundError(frames_jsonl)
    rows = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda item: (int(item["episode_id"]), int(item["frame_id"])))
    return rows


def make_features() -> dict[str, Any]:
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": [f"joint_{idx}" for idx in range(6)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": [f"joint_target_{idx}" for idx in range(6)]},
        },
    }


def shape_of(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return [int(dim) for dim in shape]


def render_chunk(args: argparse.Namespace, manifest_path: Path, render_dir: Path, chunk_len: int) -> dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "-n",
        str(args.render_conda_env),
        "--no-capture-output",
        "python",
        str(RENDERER),
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(render_dir),
        "--capture-stride",
        str(args.capture_stride),
        "--fps",
        str(args.fps),
        "--max-episodes",
        str(chunk_len),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    t0 = time.time()
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)
    render_summary = json.loads((render_dir / "render_summary.json").read_text())
    render_summary["subprocess_elapsed_s"] = time.time() - t0
    return render_summary


def append_render_to_dataset(ds: Any, render_dir: Path) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    rows = read_frame_rows(render_dir)
    episodes = sorted({int(row["episode_id"]) for row in rows})
    frame_count = 0
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
            frame_count += 1
        ds.save_episode()
    return {"episodes": len(episodes), "frames": frame_count}


def dataset_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def validate_dataset(args: argparse.Namespace, dataset_root: Path, expected_episodes: int, expected_frames: int) -> dict[str, Any]:
    ensure_hf_cache()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    ds = LeRobotDataset(args.repo_id, root=str(dataset_root), video_backend=args.video_backend)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    total_frames = int(ds.meta.total_frames)
    total_episodes = int(ds.meta.total_episodes)
    if total_episodes != int(expected_episodes):
        raise RuntimeError(f"episode count mismatch: {total_episodes} != {expected_episodes}")
    if total_frames != int(expected_frames):
        raise RuntimeError(f"frame count mismatch: {total_frames} != {expected_frames}")
    return {
        "status": "PASS",
        "repo_id": str(args.repo_id),
        "dataset_root": rel(dataset_root),
        "video_backend": str(args.video_backend),
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "dataset_bytes": int(dataset_bytes(dataset_root)),
        "batch_keys": sorted(str(key) for key in batch.keys()),
        "batch_shapes": {str(key): shape_of(value) for key, value in batch.items()},
    }


def main() -> None:
    args = parse_args()
    ensure_hf_cache()
    if int(args.chunk_size) <= 0:
        raise ValueError("--chunk-size must be positive")
    if int(args.capture_stride) <= 0:
        raise ValueError("--capture-stride must be positive")
    if not RENDERER.exists():
        raise FileNotFoundError(RENDERER)

    args.out_dir = args.out_dir.resolve()
    dataset_root = (args.dataset_root or (args.out_dir / "lerobot_dataset")).resolve()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{args.out_dir} exists and is non-empty; use --force or choose another out-dir")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in read_csv(args.accepted_csv) if int(float(row.get("accepted", "0"))) == 1]
    rows.sort(
        key=lambda row: (
            str(row["bin"]),
            str(row["chunk"]),
            int(float(row["env_id"])),
            int(float(row["episode_index"])),
        )
    )
    plan = adjust_plan(args, len(rows))
    rows = rows[: int(plan["planned_total_episodes"])]

    manifests_dir = args.out_dir / "manifests"
    renders_dir = args.out_dir / "renders"
    chunks_jsonl = args.out_dir / "d321_chunk_summaries.jsonl"
    disk_jsonl = args.out_dir / "d321_disk_checks.jsonl"
    plan_path = args.out_dir / "d321_render_lerobot_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_jsonl(disk_jsonl, {"event": "before_start", **disk_snapshot(REPO)})

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),
        root=str(dataset_root),
        features=make_features(),
        robot_type="roarm_m3",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        vcodec=str(args.vcodec),
    )

    total_frames = 0
    total_episodes = 0
    chunk_size = int(plan["planned_chunk_size"])
    overall_t0 = time.time()
    for start in range(0, len(rows), chunk_size):
        chunk_index = start // chunk_size
        chunk_rows = rows[start : start + chunk_size]
        manifest_rows: list[dict[str, Any]] = []
        for offset, row in enumerate(chunk_rows):
            out_row: dict[str, Any] = {
                "d320_episode_id": start + offset,
                "source_role": f"d321_{row['bin']}_accepted",
                "selection_reason": "d321_low_mid_production_accepted_physicality_gate",
            }
            out_row.update(row)
            manifest_rows.append(out_row)

        disk_before = disk_snapshot(REPO)
        chunk_peak = len(chunk_rows) * RAW_BYTES_PER_EP_EST + len(rows) * VIDEO_BYTES_PER_EP_EST
        if disk_before["free_bytes"] < chunk_peak * (1.0 + float(args.free_margin)):
            raise RuntimeError(
                "disk margin fell below D321 threshold before chunk "
                f"{chunk_index}: free={disk_before['free_bytes']} required={chunk_peak}"
            )

        manifest_path = manifests_dir / f"d321_replay_manifest_chunk_{chunk_index:03d}.csv"
        render_dir = renders_dir / f"render_chunk_{chunk_index:03d}"
        write_csv(manifest_path, manifest_rows)
        render_summary = render_chunk(args, manifest_path, render_dir, len(chunk_rows))
        append_info = append_render_to_dataset(ds, render_dir)
        total_frames += int(append_info["frames"])
        total_episodes += int(append_info["episodes"])

        raw_dir = render_dir / "raw_env_render_frames"
        raw_deleted = False
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            raw_deleted = True

        disk_after = disk_snapshot(REPO)
        chunk_summary = {
            "chunk_index": chunk_index,
            "manifest": rel(manifest_path),
            "render_dir": rel(render_dir),
            "episodes": len(chunk_rows),
            "append_info": append_info,
            "render_summary": render_summary,
            "raw_frames_deleted": raw_deleted,
            "disk_before": disk_before,
            "disk_after": disk_after,
        }
        write_jsonl(chunks_jsonl, chunk_summary)
        write_jsonl(disk_jsonl, {"event": f"after_chunk_{chunk_index:03d}", **disk_after})
        print(
            "[d321-render-lerobot] chunk "
            f"{chunk_index:03d} episodes={len(chunk_rows)} frames={append_info['frames']} "
            f"disk_free_gb={disk_after['free_gb']:.2f}",
            flush=True,
        )

    if hasattr(ds, "finalize"):
        ds.finalize()
    del ds

    validation = validate_dataset(args, dataset_root, total_episodes, total_frames)
    final_disk = disk_snapshot(REPO)
    write_jsonl(disk_jsonl, {"event": "after_validation", **final_disk})
    summary = {
        "artifact": "d321_render_lerobot_conveyor",
        "runtime": "CHUNK_RENDER_APPEND_VALIDATE_RAW_DELETE",
        "accepted_csv": rel(args.accepted_csv),
        "out_dir": rel(args.out_dir),
        "dataset_root": rel(dataset_root),
        "repo_id": str(args.repo_id),
        "plan": plan,
        "episodes": int(total_episodes),
        "frames": int(total_frames),
        "chunks": int((len(rows) + chunk_size - 1) // chunk_size),
        "elapsed_s": time.time() - overall_t0,
        "validation": validation,
        "disk_after": final_disk,
        "chunks_jsonl": rel(chunks_jsonl),
        "disk_jsonl": rel(disk_jsonl),
        "status": "PASS",
    }
    summary_path = args.out_dir / "d321_render_lerobot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "[d321-render-lerobot] done "
        f"episodes={total_episodes} frames={total_frames} dataset_bytes={validation['dataset_bytes']} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
