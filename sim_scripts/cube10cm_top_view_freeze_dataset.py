#!/usr/bin/env python3
"""Freeze the cube10cm top-view D247/D248 dataset version.

This creates a dataset card and checksums for the already-rendered and already
converted artifacts. It does not render, train, delete, move, archive, or modify
the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"
FREEZE_ID = "cube10cm_top_view_0_999_v0_1_d249"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--freeze-id", default=FREEZE_ID)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def line_count(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def read_id_file(path: Path) -> list[int]:
    with path.open() as f:
        return [int(line.strip()) for line in f if line.strip()]


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_files(render_dir: Path) -> list[Path]:
    lerobot_dir = render_dir / "lerobot_dataset_av1_d247"
    label_dir = render_dir / "label_package_d248"
    metadata_dir = render_dir / "metadata_companion_d247"
    files = [
        render_dir / "render_summary.json",
        render_dir / "frames.jsonl",
        render_dir / "postrender_label_validation_d246" / "label_validation_summary.json",
        render_dir / "postrender_label_validation_d246" / "episode_labels.csv",
        render_dir / "lerobot_validation_summary.json",
        render_dir / "lerobot_video_frame_counts_pyav_d247.json",
        metadata_dir / "metadata_validation_summary.json",
        metadata_dir / "metadata_schema.json",
        label_dir / "split_package_summary.json",
        label_dir / "episode_split_manifest.csv",
        label_dir / "train_clean_positive_episode_ids.txt",
        label_dir / "eval_clean_holdout_episode_ids.txt",
        label_dir / "eval_overshoot_diagnostic_episode_ids.txt",
        label_dir / "quarantine_camera_fail_episode_ids.txt",
        label_dir / "camera_fail_details.csv",
        label_dir / "camera_fail_contact_sheet.png",
    ]
    files.extend(sorted((lerobot_dir / "data").glob("**/*")))
    files.extend(sorted((lerobot_dir / "meta").glob("**/*")))
    files.extend(sorted((lerobot_dir / "videos").glob("**/*.mp4")))
    return [path for path in files if path.is_file()]


def validate_inputs(render_dir: Path) -> dict[str, Any]:
    label_dir = render_dir / "label_package_d248"
    render_summary = load_json(render_dir / "render_summary.json")
    label_summary = load_json(render_dir / "postrender_label_validation_d246" / "label_validation_summary.json")
    lerobot_summary = load_json(render_dir / "lerobot_validation_summary.json")
    frame_count_summary = load_json(render_dir / "lerobot_video_frame_counts_pyav_d247.json")
    metadata_summary = load_json(render_dir / "metadata_companion_d247" / "metadata_validation_summary.json")
    package_summary = load_json(label_dir / "split_package_summary.json")

    ids = {
        "train_clean_positive": read_id_file(label_dir / "train_clean_positive_episode_ids.txt"),
        "eval_clean_holdout": read_id_file(label_dir / "eval_clean_holdout_episode_ids.txt"),
        "eval_overshoot_diagnostic": read_id_file(label_dir / "eval_overshoot_diagnostic_episode_ids.txt"),
        "quarantine_camera_fail": read_id_file(label_dir / "quarantine_camera_fail_episode_ids.txt"),
    }
    flat_ids = [episode for values in ids.values() for episode in values]
    manifest_rows = read_manifest_rows(label_dir / "episode_split_manifest.csv")

    checks = {
        "render_episodes_1000": int(render_summary["num_episodes"]) == 1000,
        "render_frames_195000": int(render_summary["frames"]) == 195000,
        "label_episodes_1000": int(label_summary["actual_episodes"]) == 1000,
        "label_frames_195000": int(label_summary["actual_frames"]) == 195000,
        "lerobot_status_pass": lerobot_summary.get("status") == "PASS",
        "lerobot_frames_195000": int(lerobot_summary["total_frames"]) == 195000,
        "lerobot_episodes_1000": int(lerobot_summary["total_episodes"]) == 1000,
        "video_frame_count_pass": frame_count_summary.get("status") == "PASS",
        "metadata_status_pass": metadata_summary.get("status") == "PASS",
        "package_status_pass": package_summary.get("status") == "PASS",
        "package_total_1000": int(package_summary["counts"]["total"]) == 1000,
        "id_union_1000": len(flat_ids) == 1000,
        "id_union_unique_1000": len(set(flat_ids)) == 1000,
        "manifest_rows_1000": len(manifest_rows) == 1000,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"freeze input validation failed: {failed}")

    return {
        "checks": checks,
        "render": {
            "episodes": int(render_summary["num_episodes"]),
            "frames": int(render_summary["frames"]),
            "resolution": "1280x720",
            "target_fps": render_summary.get("target_fps", 30),
            "raw_png_bytes_total": int(render_summary["png_bytes_total"]),
            "raw_png_mb_per_episode": float(render_summary["debug_png_mb_per_episode"]),
        },
        "labels": {
            "camera_contract_pass": int(label_summary["camera_contract_pass_count"]),
            "counts": label_summary["label_status_counts"],
            "reprojection_max_gate_px": float(label_summary["reprojection_max_gate_px"]),
        },
        "lerobot": {
            "root": rel(render_dir / "lerobot_dataset_av1_d247"),
            "total_frames": int(lerobot_summary["total_frames"]),
            "total_episodes": int(lerobot_summary["total_episodes"]),
            "codec": lerobot_summary["info_codec"],
            "pixel_format": lerobot_summary["info_pix_fmt"],
            "fps": int(lerobot_summary["info_fps"]),
            "video_backend_validated": lerobot_summary.get("video_backend"),
            "video_bytes_total": int(lerobot_summary["video_bytes_total"]),
            "video_mb_per_episode": float(lerobot_summary["video_mb_per_episode"]),
        },
        "package": {
            "root": rel(label_dir),
            "counts": package_summary["counts"],
            "episode_id_counts": {key: len(value) for key, value in ids.items()},
        },
    }


def write_hash_manifest(out_path: Path, files: list[Path]) -> dict[str, Any]:
    rows = []
    total_bytes = 0
    for path in sorted(files, key=rel):
        size = path.stat().st_size
        total_bytes += size
        rows.append(
            {
                "sha256": sha256_file(path),
                "bytes": size,
                "path": rel(path),
            }
        )
    with out_path.open("w") as f:
        f.write("sha256\tbytes\tpath\n")
        for row in rows:
            f.write(f"{row['sha256']}\t{row['bytes']}\t{row['path']}\n")
    return {"file_count": len(rows), "total_bytes": total_bytes, "rows": rows}


def write_dataset_card(path: Path, *, freeze_id: str, manifest: dict[str, Any]) -> None:
    package = manifest["package"]["counts"]
    text = f"""# Dataset Card - {freeze_id}

## What This Freezes

This freezes the professor cube10cm top-view visual trajectory corpus at the
D249 dataset-freeze stage. It is a dataset artifact, not a training result.

Primary dataset:

```text
{manifest['lerobot']['root']}
```

Split package:

```text
{manifest['package']['root']}
```

## Counts

- Total episodes: `{manifest['render']['episodes']}`
- Total frames: `{manifest['render']['frames']}`
- Resolution: `{manifest['render']['resolution']}`
- LeRobot codec: `{manifest['lerobot']['codec']}` / `{manifest['lerobot']['pixel_format']}` / `{manifest['lerobot']['fps']}fps`
- Local validated video backend: `{manifest['lerobot']['video_backend_validated']}`
- Video bytes: `{manifest['lerobot']['video_bytes_total']}`
- Raw PNG bytes: `{manifest['render']['raw_png_bytes_total']}`

## Splits

- `train_clean_positive`: 학습용 정상 성공 예시, `{package['by_subsplit']['train_clean_positive']}` episodes.
  카메라 검증을 통과했고, 접촉과 큐브 반응이 있으며, overshoot가 없는 데이터만 포함한다.
- `eval_clean_holdout`: 평가용 정상 보류 예시, `{package['by_subsplit']['eval_clean_holdout']}` episodes.
  학습에 넣을 수 있는 정상 성공 데이터 중 일부를 일부러 빼둔 시험 문제다.
- `eval_overshoot_diagnostic`: 과하게 민 케이스 진단용 평가 데이터, `{package['by_subsplit']['eval_overshoot_diagnostic']}` episodes.
  접촉과 반응은 있지만 큐브를 과하게 밀었으므로 기본 positive behavior-cloning 학습에는 넣지 않는다.
- `quarantine_camera_fail`: 카메라 기준 실패 격리 데이터, `{package['by_subsplit']['quarantine_camera_fail']}` episodes.
  카메라 투영/coverage 기준을 통과하지 못했으므로 train/eval에서 제외한다.

## Default Use

- Default train set: `train_clean_positive`.
- Default clean evaluation set: `eval_clean_holdout`.
- Failure/overshoot diagnostic set: `eval_overshoot_diagnostic`.
- Excluded set: `quarantine_camera_fail`.

## Caveats

- This does not prove model performance.
- This does not run PPO, VLA/SmolVLA fine-tuning, action-teacher, or RoArm deployment.
- Local LeRobot validation should explicitly use `video_backend=pyav` unless the
  local torchcodec/FFmpeg stack is repaired.
- Raw PNGs remain preserved and dominate disk usage; no cleanup is implied by
  this freeze.

## Reproducibility Files

- `dataset_freeze_manifest_d249.json`
- `sha256_manifest_d249.tsv`
- Source split package: `label_package_d248`
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    render_dir = args.render_dir
    out_dir = args.out_dir or (render_dir / "dataset_freeze_d249")
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} exists; use --force or another --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_inputs(render_dir)
    files = collect_files(render_dir)
    hash_info = write_hash_manifest(out_dir / "sha256_manifest_d249.tsv", files)
    manifest = {
        "artifact": "cube10cm_top_view_dataset_freeze_d249",
        "freeze_id": args.freeze_id,
        "runtime": "NO_RENDER_NO_TRAINING_NO_DELETE_DATASET_FREEZE_ONLY",
        "render_dir": rel(render_dir),
        "out_dir": rel(out_dir),
        "validation": validation["checks"],
        "render": validation["render"],
        "labels": validation["labels"],
        "lerobot": validation["lerobot"],
        "package": validation["package"],
        "hash_manifest": {
            "path": rel(out_dir / "sha256_manifest_d249.tsv"),
            "file_count": hash_info["file_count"],
            "total_bytes": hash_info["total_bytes"],
        },
        "raw_png_policy": {
            "raw_pngs_preserved": True,
            "raw_pngs_individual_sha256": False,
            "reason": "primary frozen artifact is LeRobot MP4+parquet plus split manifests; raw PNGs are large debug/source frames and remain preserved without per-PNG hashing",
        },
        "status": "PASS",
    }
    manifest_path = out_dir / "dataset_freeze_manifest_d249.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_dataset_card(out_dir / "dataset_card_d249.md", freeze_id=args.freeze_id, manifest=manifest)
    print(
        "[cube10cm-dataset-freeze] done "
        f"status=PASS freeze_id={args.freeze_id} files={hash_info['file_count']} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
