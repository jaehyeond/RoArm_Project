#!/usr/bin/env python3
"""Build companion metadata parquet for cube10cm top-view visual datasets.

This is a non-render utility. It reads an existing render `frames.jsonl` and,
optionally, an existing LeRobot dataset root, then writes companion metadata
tables keyed by the same frame indices. It does not run IsaacLab, train, delete,
archive, move, or modify the LeRobot core dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_smoke_d232"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument(
        "--lerobot-dir",
        type=Path,
        default=None,
        help="Optional LeRobot dataset root. Defaults to <render-dir>/lerobot_dataset_av1 when present.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output companion metadata dir. Defaults to <render-dir>/metadata_companion_d235.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing companion files.")
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def vec(row: dict[str, Any], key: str, size: int) -> list[float]:
    raw = row.get(key, [])
    if not isinstance(raw, list):
        raw = []
    values = [finite_float(raw[i]) if i < len(raw) else math.nan for i in range(size)]
    return values


def bbox_values(projection: dict[str, Any]) -> list[float]:
    bbox = projection.get("bbox") if isinstance(projection, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return [math.nan, math.nan, math.nan, math.nan]
    return [finite_float(value) for value in bbox]


def center_values(projection: dict[str, Any]) -> list[float]:
    center = projection.get("uv_center") if isinstance(projection, dict) else None
    if not isinstance(center, list) or len(center) != 2:
        return [math.nan, math.nan]
    return [finite_float(value) for value in center]


def read_rows(render_dir: Path) -> list[dict[str, Any]]:
    frames_jsonl = render_dir / "frames.jsonl"
    if not frames_jsonl.exists():
        raise FileNotFoundError(frames_jsonl)
    rows = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (int(row["episode_id"]), int(row["frame_id"])))
    for idx, row in enumerate(rows):
        row["global_index"] = idx
    return rows


def flatten_frame(row: dict[str, Any]) -> dict[str, Any]:
    cam = vec(row, "camera_center_world_m", 3)
    cube = vec(row, "cube_position_world_m", 3)
    quat = vec(row, "cube_quat_wxyz", 4)
    cube_vel = vec(row, "cube_linear_velocity_mps", 3)
    tcp = vec(row, "tcp_position_world_m", 3)
    target = vec(row, "target_position_world_m", 3)
    push = vec(row, "push_dir_xy", 2)
    projection = row.get("projection", {})
    center_u, center_v = center_values(projection)
    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox_values(projection)

    return {
        "global_index": int(row["global_index"]),
        "episode_index": int(row["episode_id"]),
        "frame_index": int(row["frame_id"]),
        "sim_step": int(row.get("sim_step", -1)),
        "timestamp_s": finite_float(row.get("timestamp_s")),
        "sim_time_s": finite_float(row.get("sim_time_s")),
        "source_png": str(row.get("source_png", "")),
        "camera_contract_id": str(row.get("camera_contract_id", "")),
        "camera_path": str(row.get("camera_path", "")),
        "camera_center_world_x": cam[0],
        "camera_center_world_y": cam[1],
        "camera_center_world_z": cam[2],
        "camera_height_above_table_m": finite_float(row.get("camera_height_above_table_m")),
        "image_width": int(row.get("image_width", 0)),
        "image_height": int(row.get("image_height", 0)),
        "image_convention": str(row.get("image_convention", "")),
        "cube_position_world_x": cube[0],
        "cube_position_world_y": cube[1],
        "cube_position_world_z": cube[2],
        "cube_quat_w": quat[0],
        "cube_quat_x": quat[1],
        "cube_quat_y": quat[2],
        "cube_quat_z": quat[3],
        "cube_linear_velocity_x": cube_vel[0],
        "cube_linear_velocity_y": cube_vel[1],
        "cube_linear_velocity_z": cube_vel[2],
        "tcp_position_world_x": tcp[0],
        "tcp_position_world_y": tcp[1],
        "tcp_position_world_z": tcp[2],
        "target_position_world_x": target[0],
        "target_position_world_y": target[1],
        "target_position_world_z": target[2],
        "push_dir_x": push[0],
        "push_dir_y": push[1],
        "tap_contact_proxy": finite_float(row.get("tap_contact_proxy")),
        "tap_contact_seen": finite_float(row.get("tap_contact_seen")),
        "tap_reaction_seen": finite_float(row.get("tap_reaction_seen")),
        "tap_overshoot_seen": finite_float(row.get("tap_overshoot_seen")),
        "tap_success_flag": finite_float(row.get("tap_success_flag")),
        "tap_disp_along_m": finite_float(row.get("tap_disp_along_m")),
        "tap_disp_xy_m": finite_float(row.get("tap_disp_xy_m")),
        "tap_speed_mps": finite_float(row.get("tap_speed_mps")),
        "cube_visibility": str(row.get("cube_visibility", "")),
        "blue_coverage": finite_float(row.get("blue_coverage")),
        "blue_pixels": int(row.get("blue_pixels", 0)),
        "bbox_area": int(row.get("bbox_area", 0)),
        "centroid_error_px": finite_float(row.get("centroid_error_px")),
        "projection_center_u": center_u,
        "projection_center_v": center_v,
        "projection_bbox_x0": bbox_x0,
        "projection_bbox_y0": bbox_y0,
        "projection_bbox_x1": bbox_x1,
        "projection_bbox_y1": bbox_y1,
        "projection_uv_corners_json": json.dumps(projection.get("uv_corners", []), sort_keys=True),
    }


def episode_metadata(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        by_episode.setdefault(int(frame["episode_index"]), []).append(frame)

    out: list[dict[str, Any]] = []
    for episode_index, ep_frames in sorted(by_episode.items()):
        ep_frames.sort(key=lambda frame: int(frame["frame_index"]))
        first = ep_frames[0]
        last = ep_frames[-1]
        out.append(
            {
                "episode_index": episode_index,
                "num_frames": len(ep_frames),
                "first_global_index": int(first["global_index"]),
                "last_global_index": int(last["global_index"]),
                "first_sim_step": int(first["sim_step"]),
                "last_sim_step": int(last["sim_step"]),
                "first_cube_x": float(first["cube_position_world_x"]),
                "first_cube_y": float(first["cube_position_world_y"]),
                "first_cube_z": float(first["cube_position_world_z"]),
                "last_cube_x": float(last["cube_position_world_x"]),
                "last_cube_y": float(last["cube_position_world_y"]),
                "last_cube_z": float(last["cube_position_world_z"]),
                "full_visibility_frames": sum(
                    1 for frame in ep_frames if frame["cube_visibility"] == "cube_visible_full"
                ),
                "partial_visibility_frames": sum(
                    1 for frame in ep_frames if frame["cube_visibility"] == "cube_visible_partial"
                ),
                "full_occlusion_frames": sum(
                    1 for frame in ep_frames if frame["cube_visibility"] == "cube_occluded_full"
                ),
                "contact_seen_any": bool(max(float(frame["tap_contact_seen"]) for frame in ep_frames)),
                "reaction_seen_any": bool(max(float(frame["tap_reaction_seen"]) for frame in ep_frames)),
                "overshoot_seen_any": bool(max(float(frame["tap_overshoot_seen"]) for frame in ep_frames)),
            }
        )
    return out


def validate_against_lerobot(lerobot_dir: Path | None, frames: list[dict[str, Any]]) -> dict[str, Any]:
    if lerobot_dir is None or not lerobot_dir.exists():
        return {"checked": False, "reason": "lerobot_dir_missing"}

    import pandas as pd

    info_path = lerobot_dir / "meta" / "info.json"
    data_files = sorted((lerobot_dir / "data").glob("chunk-*/file-*.parquet"))
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not data_files:
        raise FileNotFoundError(lerobot_dir / "data")

    info = json.loads(info_path.read_text())
    data = pd.concat([pd.read_parquet(path) for path in data_files], ignore_index=True)
    expected_rows = len(frames)
    if int(info["total_frames"]) != expected_rows:
        raise RuntimeError(f"LeRobot total_frames {info['total_frames']} != metadata rows {expected_rows}")
    if len(data) != expected_rows:
        raise RuntimeError(f"LeRobot parquet rows {len(data)} != metadata rows {expected_rows}")

    mismatches = []
    for idx, frame in enumerate(frames):
        data_row = data.iloc[idx]
        if int(data_row["index"]) != int(frame["global_index"]):
            mismatches.append(("index", idx, int(data_row["index"]), int(frame["global_index"])))
        if int(data_row["episode_index"]) != int(frame["episode_index"]):
            mismatches.append(("episode_index", idx, int(data_row["episode_index"]), int(frame["episode_index"])))
        if int(data_row["frame_index"]) != int(frame["frame_index"]):
            mismatches.append(("frame_index", idx, int(data_row["frame_index"]), int(frame["frame_index"])))
        if len(mismatches) >= 5:
            break
    if mismatches:
        raise RuntimeError(f"LeRobot/core index mismatch examples: {mismatches}")

    return {
        "checked": True,
        "lerobot_dir": str(lerobot_dir),
        "total_frames": int(info["total_frames"]),
        "total_episodes": int(info["total_episodes"]),
        "data_files": [str(path) for path in data_files],
        "data_columns": list(data.columns),
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    out_dir = args.out_dir or (args.render_dir / "metadata_companion_d235")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "per_frame": out_dir / "per_frame_metadata.parquet",
        "episode": out_dir / "episode_metadata.parquet",
        "schema": out_dir / "metadata_schema.json",
        "summary": out_dir / "metadata_validation_summary.json",
    }
    if not args.force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"companion outputs already exist: {existing}")

    frames = [flatten_frame(row) for row in rows]
    episodes = episode_metadata(frames)
    lerobot_dir = args.lerobot_dir
    if lerobot_dir is None:
        candidate = args.render_dir / "lerobot_dataset_av1"
        lerobot_dir = candidate if candidate.exists() else None
    lerobot_validation = validate_against_lerobot(lerobot_dir, frames)

    frame_df = pd.DataFrame(frames)
    episode_df = pd.DataFrame(episodes)
    frame_df.to_parquet(paths["per_frame"], index=False)
    episode_df.to_parquet(paths["episode"], index=False)

    schema = {
        "artifact": "cube10cm_top_view_visual_metadata_companion_d235",
        "policy": "standard_lerobot_core_plus_companion_metadata",
        "join_keys": ["global_index", "episode_index", "frame_index"],
        "per_frame_columns": list(frame_df.columns),
        "episode_columns": list(episode_df.columns),
    }
    paths["schema"].write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")

    summary = {
        "artifact": "cube10cm_top_view_visual_metadata_companion_d235",
        "runtime": "NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING",
        "render_dir": str(args.render_dir),
        "out_dir": str(out_dir),
        "rows": len(frames),
        "episodes": len(episodes),
        "per_frame_path": str(paths["per_frame"]),
        "episode_path": str(paths["episode"]),
        "schema_path": str(paths["schema"]),
        "lerobot_validation": lerobot_validation,
        "status": "PASS",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    args.render_dir = args.render_dir.resolve()
    if args.lerobot_dir is not None:
        args.lerobot_dir = args.lerobot_dir.resolve()
    if args.out_dir is not None:
        args.out_dir = args.out_dir.resolve()

    rows = read_rows(args.render_dir)
    summary = write_outputs(args, rows)
    print(
        "[cube10cm-metadata-companion] done "
        f"status={summary['status']} rows={summary['rows']} episodes={summary['episodes']} "
        f"out={summary['out_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
