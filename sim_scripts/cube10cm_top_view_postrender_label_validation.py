#!/usr/bin/env python3
"""Post-render numeric label validation for cube10cm top-view chunks.

This utility reads an existing render `frames.jsonl` and writes episode-level
numeric labels. It does not run IsaacLab, convert to LeRobot, train, delete,
move, or modify the render dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_chunk100_d241"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--expected-frames-per-episode", type=int, default=195)
    parser.add_argument("--reprojection-max-gate-px", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bool_num(value: bool) -> int:
    return 1 if value else 0


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def read_rows(render_dir: Path) -> list[dict[str, Any]]:
    frames_jsonl = render_dir / "frames.jsonl"
    if not frames_jsonl.exists():
        raise FileNotFoundError(frames_jsonl)
    rows = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (finite_int(row.get("episode_id"), -1), finite_int(row.get("frame_id"), -1)))
    for index, row in enumerate(rows):
        row["global_index"] = index
    return rows


def first_event(frames: list[dict[str, Any]], key: str) -> tuple[int, float]:
    for row in frames:
        if finite_float(row.get(key), 0.0) > 0.0:
            return finite_int(row.get("frame_id"), -1), finite_float(row.get("timestamp_s"))
    return -1, math.nan


def vec(row: dict[str, Any], key: str, size: int) -> list[float]:
    raw = row.get(key)
    if not isinstance(raw, list):
        raw = []
    return [finite_float(raw[i]) if i < len(raw) else math.nan for i in range(size)]


def label_status(
    *,
    frame_count_ok: bool,
    camera_contract_pass: bool,
    contact_seen: bool,
    reaction_seen: bool,
    overshoot_seen: bool,
) -> str:
    if not frame_count_ok:
        return "frame_count_fail"
    if not camera_contract_pass:
        return "camera_quality_fail"
    if contact_seen and reaction_seen and not overshoot_seen:
        return "clean_useful_tap"
    if contact_seen and reaction_seen and overshoot_seen:
        return "contact_reaction_with_overshoot"
    if contact_seen and not reaction_seen:
        return "contact_without_reaction"
    return "missing_contact"


def episode_label(
    episode_id: int,
    frames: list[dict[str, Any]],
    *,
    expected_frames_per_episode: int,
    reprojection_max_gate_px: float,
) -> dict[str, Any]:
    frames.sort(key=lambda row: finite_int(row.get("frame_id"), -1))
    first = frames[0]
    last = frames[-1]

    visibility = [str(row.get("cube_visibility", "")) for row in frames]
    centroid_errors = [finite_float(row.get("centroid_error_px")) for row in frames]
    blue_coverages = [finite_float(row.get("blue_coverage")) for row in frames]
    bbox_areas = [finite_float(row.get("bbox_area")) for row in frames]
    projection_inside = [
        bool(row.get("projection", {}).get("inside", False)) if isinstance(row.get("projection"), dict) else False
        for row in frames
    ]

    num_frames = len(frames)
    full_visibility_frames = sum(1 for value in visibility if value == "cube_visible_full")
    partial_visibility_frames = sum(1 for value in visibility if value == "cube_visible_partial")
    full_occlusion_frames = sum(1 for value in visibility if value == "cube_occluded_full")
    projection_inside_frames = sum(1 for value in projection_inside if value)

    contact_seen = any(finite_float(row.get("tap_contact_seen"), 0.0) > 0.0 for row in frames)
    reaction_seen = any(finite_float(row.get("tap_reaction_seen"), 0.0) > 0.0 for row in frames)
    overshoot_seen = any(finite_float(row.get("tap_overshoot_seen"), 0.0) > 0.0 for row in frames)
    legacy_success_seen = any(finite_float(row.get("tap_success_flag"), 0.0) > 0.0 for row in frames)

    contact_frame, contact_time = first_event(frames, "tap_contact_seen")
    reaction_frame, reaction_time = first_event(frames, "tap_reaction_seen")
    overshoot_frame, overshoot_time = first_event(frames, "tap_overshoot_seen")
    legacy_success_frame, legacy_success_time = first_event(frames, "tap_success_flag")
    overshoot_before_contact = overshoot_frame >= 0 and (contact_frame < 0 or overshoot_frame < contact_frame)

    frame_count_ok = num_frames == expected_frames_per_episode
    full_visibility_ok = full_visibility_frames == num_frames
    projection_inside_ok = projection_inside_frames == num_frames
    centroid_error_max = max((value for value in centroid_errors if math.isfinite(value)), default=math.nan)
    reprojection_gate_ok = math.isfinite(centroid_error_max) and centroid_error_max <= reprojection_max_gate_px
    camera_contract_pass = full_visibility_ok and projection_inside_ok and reprojection_gate_ok

    first_cube = vec(first, "cube_position_world_m", 3)
    last_cube = vec(last, "cube_position_world_m", 3)
    target = vec(first, "target_position_world_m", 3)
    initial_xy = first_cube[:2]
    final_xy = last_cube[:2]
    final_dx = final_xy[0] - initial_xy[0]
    final_dy = final_xy[1] - initial_xy[1]

    status = label_status(
        frame_count_ok=frame_count_ok,
        camera_contract_pass=camera_contract_pass,
        contact_seen=contact_seen,
        reaction_seen=reaction_seen,
        overshoot_seen=overshoot_seen,
    )

    return {
        "episode_index": episode_id,
        "split_candidate": str(first.get("split_candidate", "")),
        "sampling_cell_id": str(first.get("sampling_cell_id", "")),
        "sampling_rule": str(first.get("sampling_rule", "")),
        "source_decision": str(first.get("source_decision", "")),
        "manifest_seed": finite_int(first.get("manifest_seed"), -1),
        "requires_posthoc_label_validation": bool_num(bool(first.get("requires_posthoc_label_validation", False))),
        "num_frames": num_frames,
        "frame_count_ok": bool_num(frame_count_ok),
        "first_global_index": finite_int(first.get("global_index"), -1),
        "last_global_index": finite_int(last.get("global_index"), -1),
        "first_frame_index": finite_int(first.get("frame_id"), -1),
        "last_frame_index": finite_int(last.get("frame_id"), -1),
        "first_sim_step": finite_int(first.get("sim_step"), -1),
        "last_sim_step": finite_int(last.get("sim_step"), -1),
        "duration_s": finite_float(last.get("timestamp_s"), 0.0) - finite_float(first.get("timestamp_s"), 0.0),
        "initial_cube_x_m": initial_xy[0],
        "initial_cube_y_m": initial_xy[1],
        "final_cube_x_m": final_xy[0],
        "final_cube_y_m": final_xy[1],
        "target_cube_x_m": target[0],
        "target_cube_y_m": target[1],
        "final_dx_m": final_dx,
        "final_dy_m": final_dy,
        "final_tap_disp_along_m": finite_float(last.get("tap_disp_along_m")),
        "max_tap_disp_along_m": max(finite_float(row.get("tap_disp_along_m")) for row in frames),
        "final_tap_disp_xy_m": finite_float(last.get("tap_disp_xy_m")),
        "max_tap_disp_xy_m": max(finite_float(row.get("tap_disp_xy_m")) for row in frames),
        "max_tap_speed_mps": max(finite_float(row.get("tap_speed_mps")) for row in frames),
        "contact_seen_any": bool_num(contact_seen),
        "reaction_seen_any": bool_num(reaction_seen),
        "overshoot_seen_any": bool_num(overshoot_seen),
        "legacy_target_band_success_any": bool_num(legacy_success_seen),
        "contact_first_frame": contact_frame,
        "contact_first_time_s": contact_time,
        "reaction_first_frame": reaction_frame,
        "reaction_first_time_s": reaction_time,
        "overshoot_first_frame": overshoot_frame,
        "overshoot_first_time_s": overshoot_time,
        "overshoot_before_contact": bool_num(overshoot_before_contact),
        "legacy_success_first_frame": legacy_success_frame,
        "legacy_success_first_time_s": legacy_success_time,
        "full_visibility_frames": full_visibility_frames,
        "partial_visibility_frames": partial_visibility_frames,
        "full_occlusion_frames": full_occlusion_frames,
        "projection_inside_frames": projection_inside_frames,
        "full_visibility_ok": bool_num(full_visibility_ok),
        "projection_inside_ok": bool_num(projection_inside_ok),
        "centroid_error_px_median": median([v for v in centroid_errors if math.isfinite(v)]),
        "centroid_error_px_p95": percentile(centroid_errors, 0.95),
        "centroid_error_px_max": centroid_error_max,
        "reprojection_max_gate_px": reprojection_max_gate_px,
        "reprojection_gate_ok": bool_num(reprojection_gate_ok),
        "blue_coverage_min": min((value for value in blue_coverages if math.isfinite(value)), default=math.nan),
        "blue_coverage_median": median([v for v in blue_coverages if math.isfinite(v)]),
        "bbox_area_min": min((value for value in bbox_areas if math.isfinite(value)), default=math.nan),
        "bbox_area_max": max((value for value in bbox_areas if math.isfinite(value)), default=math.nan),
        "camera_contract_pass": bool_num(camera_contract_pass),
        "label_useful_clean_numeric": bool_num(contact_seen and reaction_seen and not overshoot_seen),
        "label_overshoot_numeric": bool_num(overshoot_seen),
        "label_missing_contact_or_reaction_numeric": bool_num(not (contact_seen and reaction_seen)),
        "label_legacy_target_band_numeric": bool_num(legacy_success_seen),
        "label_camera_contract_numeric": bool_num(camera_contract_pass),
        "label_status": status,
    }


def build_labels(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(args.render_dir)
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(finite_int(row.get("episode_id"), -1), []).append(row)

    labels = [
        episode_label(
            episode_id,
            frames,
            expected_frames_per_episode=args.expected_frames_per_episode,
            reprojection_max_gate_px=args.reprojection_max_gate_px,
        )
        for episode_id, frames in sorted(by_episode.items())
    ]

    split_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in labels:
        split_counts[row["split_candidate"]] = split_counts.get(row["split_candidate"], 0) + 1
        status_counts[row["label_status"]] = status_counts.get(row["label_status"], 0) + 1

    summary = {
        "artifact": "cube10cm_top_view_postrender_label_validation",
        "render_dir": str(args.render_dir),
        "expected_episodes": args.expected_episodes,
        "expected_frames_per_episode": args.expected_frames_per_episode,
        "actual_episodes": len(labels),
        "actual_frames": len(rows),
        "episode_count_ok": len(labels) == args.expected_episodes,
        "frame_count_ok": all(row["frame_count_ok"] == 1 for row in labels),
        "split_counts": split_counts,
        "label_status_counts": status_counts,
        "useful_clean_count": sum(row["label_useful_clean_numeric"] for row in labels),
        "overshoot_count": sum(row["label_overshoot_numeric"] for row in labels),
        "missing_contact_or_reaction_count": sum(row["label_missing_contact_or_reaction_numeric"] for row in labels),
        "legacy_target_band_count": sum(row["label_legacy_target_band_numeric"] for row in labels),
        "camera_contract_pass_count": sum(row["label_camera_contract_numeric"] for row in labels),
        "reprojection_max_gate_px": args.reprojection_max_gate_px,
        "centroid_error_px_max_over_episodes": max(row["centroid_error_px_max"] for row in labels),
        "centroid_error_px_median_over_episodes": median(row["centroid_error_px_median"] for row in labels),
    }
    return labels, summary


def write_outputs(args: argparse.Namespace, labels: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    out_dir = args.out_dir or (args.render_dir / "postrender_label_validation_d241")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "episode_csv": out_dir / "episode_labels.csv",
        "episode_json": out_dir / "episode_labels.json",
        "summary": out_dir / "label_validation_summary.json",
    }
    if not args.force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"label outputs already exist: {existing}")

    with paths["episode_csv"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
        writer.writeheader()
        writer.writerows(labels)
    paths["episode_json"].write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    labels, summary = build_labels(args)
    paths = write_outputs(args, labels, summary)
    print(json.dumps({"summary": summary, "paths": paths}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
