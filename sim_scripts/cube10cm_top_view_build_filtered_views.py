#!/usr/bin/env python3
"""Build filtered index views for the frozen cube10cm top-view LeRobot dataset.

The output is a set of episode/frame index files that lets later loaders use
only approved train/eval subsets without copying the LeRobot dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--label-package-dir", type=Path, default=None)
    parser.add_argument("--freeze-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ints(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(str(value) for value in values) + "\n")


def int_field(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def make_view_rows(rows: list[dict[str, str]], subsplit: str) -> tuple[list[dict[str, Any]], list[int]]:
    episode_rows = [row for row in rows if row["package_subsplit"] == subsplit]
    episode_rows.sort(key=lambda row: int_field(row, "episode_index"))
    view_rows: list[dict[str, Any]] = []
    frame_indices: list[int] = []
    for row in episode_rows:
        first = int_field(row, "first_global_index")
        last = int_field(row, "last_global_index")
        episode_index = int_field(row, "episode_index")
        if last < first:
            raise RuntimeError(f"bad global range for episode {episode_index}: {first}>{last}")
        for local_frame, global_index in enumerate(range(first, last + 1)):
            frame_indices.append(global_index)
            view_rows.append(
                {
                    "global_index": global_index,
                    "episode_index": episode_index,
                    "frame_index": local_frame,
                    "package_subsplit": subsplit,
                    "label_status": row["label_status"],
                    "source_first_global_index": first,
                    "source_last_global_index": last,
                }
            )
    return view_rows, frame_indices


def main() -> None:
    args = parse_args()
    render_dir = args.render_dir
    label_package_dir = args.label_package_dir or (render_dir / "label_package_d248")
    freeze_dir = args.freeze_dir or (render_dir / "dataset_freeze_d249")
    out_dir = args.out_dir or (render_dir / "filtered_views_d250")
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} exists; use --force or another --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    package_summary = json.loads((label_package_dir / "split_package_summary.json").read_text())
    freeze_manifest = json.loads((freeze_dir / "dataset_freeze_manifest_d249.json").read_text())
    rows = read_rows(label_package_dir / "episode_split_manifest.csv")

    views = {
        "train_clean_positive": "학습용 정상 성공 예시",
        "eval_clean_holdout": "평가용 정상 보류 예시",
        "eval_overshoot_diagnostic": "과하게 민 케이스 진단용 평가 데이터",
        "quarantine_camera_fail": "카메라 기준 실패 격리 데이터",
    }

    summary: dict[str, Any] = {
        "artifact": "cube10cm_top_view_filtered_views_d250",
        "runtime": "NO_RENDER_NO_TRAINING_NO_DELETE_FILTERED_INDEX_VIEW_ONLY",
        "render_dir": str(render_dir),
        "label_package_dir": str(label_package_dir),
        "freeze_id": freeze_manifest["freeze_id"],
        "lerobot_root": freeze_manifest["lerobot"]["root"],
        "views": {},
        "status": "PASS",
    }

    total_frame_indices = 0
    for subsplit, korean_definition in views.items():
        view_rows, frame_indices = make_view_rows(rows, subsplit)
        episode_count = len({row["episode_index"] for row in view_rows})
        expected_episode_count = int(package_summary["counts"]["by_subsplit"][subsplit])
        expected_frames = expected_episode_count * 195
        if episode_count != expected_episode_count:
            raise RuntimeError(f"{subsplit} episode count mismatch: {episode_count} != {expected_episode_count}")
        if len(frame_indices) != expected_frames:
            raise RuntimeError(f"{subsplit} frame count mismatch: {len(frame_indices)} != {expected_frames}")
        write_csv(
            out_dir / f"{subsplit}_frame_view.csv",
            view_rows,
            [
                "global_index",
                "episode_index",
                "frame_index",
                "package_subsplit",
                "label_status",
                "source_first_global_index",
                "source_last_global_index",
            ],
        )
        write_ints(out_dir / f"{subsplit}_frame_indices.txt", frame_indices)
        total_frame_indices += len(frame_indices)
        summary["views"][subsplit] = {
            "korean_definition": korean_definition,
            "episodes": episode_count,
            "frames": len(frame_indices),
            "frame_view_csv": str(out_dir / f"{subsplit}_frame_view.csv"),
            "frame_indices_txt": str(out_dir / f"{subsplit}_frame_indices.txt"),
        }

    if total_frame_indices != int(freeze_manifest["lerobot"]["total_frames"]):
        raise RuntimeError(f"total frame coverage mismatch: {total_frame_indices}")

    (out_dir / "filtered_views_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        "[cube10cm-filtered-views] done "
        f"status=PASS views={len(views)} frame_indices={total_frame_indices} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
