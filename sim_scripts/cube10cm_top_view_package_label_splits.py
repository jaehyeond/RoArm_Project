#!/usr/bin/env python3
"""Package D246 cube10cm top-view episode labels into train/eval/quarantine lists.

This reads existing render and post-render label artifacts only. It does not run
IsaacLab, convert videos, train, delete, move, or modify the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_LABEL_CSV = DEFAULT_RENDER_DIR / "postrender_label_validation_d246" / "episode_labels.csv"

CORE_FIELDS = [
    "episode_index",
    "package_split",
    "package_subsplit",
    "include_in_positive_bc_train",
    "include_in_eval",
    "quarantine",
    "package_reason",
    "label_status",
    "split_candidate",
    "sampling_cell_id",
    "initial_cube_x_m",
    "initial_cube_y_m",
    "final_tap_disp_along_m",
    "max_tap_disp_along_m",
    "contact_seen_any",
    "reaction_seen_any",
    "overshoot_seen_any",
    "camera_contract_pass",
    "projection_inside_frames",
    "full_visibility_frames",
    "reprojection_gate_ok",
    "centroid_error_px_max",
    "reprojection_max_gate_px",
    "label_useful_clean_numeric",
    "label_overshoot_numeric",
]

FAIL_FIELDS = [
    "episode_index",
    "package_subsplit",
    "camera_fail_reason",
    "split_candidate",
    "sampling_cell_id",
    "initial_cube_x_m",
    "initial_cube_y_m",
    "contact_seen_any",
    "reaction_seen_any",
    "overshoot_seen_any",
    "label_useful_clean_numeric",
    "label_overshoot_numeric",
    "num_frames",
    "full_visibility_frames",
    "projection_inside_frames",
    "centroid_error_px_max",
    "reprojection_max_gate_px",
    "blue_coverage_min",
    "bbox_area_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--clean-holdout-frac", type=float, default=0.10)
    parser.add_argument("--holdout-salt", default="cube10cm_top_view_d248_clean_holdout_v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    return str(value).strip() in {"1", "1.0", "true", "True", "yes", "YES"}


def stable_score(episode_index: int, salt: str) -> str:
    raw = f"{salt}:{episode_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def camera_fail_reason(row: dict[str, str]) -> str:
    reasons: list[str] = []
    if not truthy(row.get("frame_count_ok")):
        reasons.append("frame_count")
    if not truthy(row.get("full_visibility_ok")):
        reasons.append("full_visibility")
    if not truthy(row.get("projection_inside_ok")):
        reasons.append("projection_outside")
    if not truthy(row.get("reprojection_gate_ok")):
        reasons.append("reprojection_error_gt_gate")
    return "+".join(reasons) if reasons else "camera_contract_fail_unspecified"


def classify(rows: list[dict[str, str]], *, holdout_frac: float, holdout_salt: str) -> list[dict[str, Any]]:
    clean_rows = [r for r in rows if r.get("label_status") == "clean_useful_tap"]
    holdout_n = round(len(clean_rows) * holdout_frac)
    clean_holdout_ids = {
        finite_int(r["episode_index"])
        for r in sorted(clean_rows, key=lambda r: (stable_score(finite_int(r["episode_index"]), holdout_salt), finite_int(r["episode_index"])))[:holdout_n]
    }

    packaged: list[dict[str, Any]] = []
    for row in rows:
        episode_index = finite_int(row["episode_index"])
        status = row.get("label_status", "")
        out = dict(row)
        out["include_in_positive_bc_train"] = 0
        out["include_in_eval"] = 0
        out["quarantine"] = 0

        if status == "clean_useful_tap":
            if episode_index in clean_holdout_ids:
                out["package_split"] = "eval"
                out["package_subsplit"] = "eval_clean_holdout"
                out["include_in_eval"] = 1
                out["package_reason"] = (
                    "camera-pass clean useful tap; deterministic 10pct clean holdout for future eval"
                )
            else:
                out["package_split"] = "train"
                out["package_subsplit"] = "train_clean_positive"
                out["include_in_positive_bc_train"] = 1
                out["package_reason"] = (
                    "camera-pass clean useful tap; contact and reaction seen; overshoot absent"
                )
        elif status == "contact_reaction_with_overshoot":
            out["package_split"] = "eval"
            out["package_subsplit"] = "eval_overshoot_diagnostic"
            out["include_in_eval"] = 1
            out["package_reason"] = (
                "camera-pass contact/reaction with overshoot; exclude from positive BC train"
            )
        elif status == "camera_quality_fail":
            out["package_split"] = "quarantine"
            out["package_subsplit"] = "quarantine_camera_fail"
            out["quarantine"] = 1
            out["camera_fail_reason"] = camera_fail_reason(row)
            out["package_reason"] = (
                "camera contract failed; exclude until camera coverage/reprojection is inspected"
            )
        else:
            out["package_split"] = "quarantine"
            out["package_subsplit"] = "quarantine_other_label_fail"
            out["quarantine"] = 1
            out["camera_fail_reason"] = ""
            out["package_reason"] = f"non-trainable label status: {status}"

        out.setdefault("camera_fail_reason", "")
        packaged.append(out)

    return sorted(packaged, key=lambda r: finite_int(r["episode_index"]))


def read_frames(render_dir: Path) -> dict[int, list[dict[str, Any]]]:
    frames_jsonl = render_dir / "frames.jsonl"
    by_episode: dict[int, list[dict[str, Any]]] = {}
    with frames_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_episode.setdefault(finite_int(row.get("episode_id"), -1), []).append(row)
    for frames in by_episode.values():
        frames.sort(key=lambda r: finite_int(r.get("frame_id"), -1))
    return by_episode


def max_error_frame(frames: list[dict[str, Any]]) -> int:
    best = max(frames, key=lambda r: finite_float(r.get("centroid_error_px"), -1.0))
    return finite_int(best.get("frame_id"), 0)


def frame_row(frames: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    by_id = {finite_int(row.get("frame_id"), -1): row for row in frames}
    if frame_id in by_id:
        return by_id[frame_id]
    return frames[min(max(frame_id, 0), len(frames) - 1)]


def create_contact_sheet(render_dir: Path, fail_rows: list[dict[str, Any]], out_path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    frames_by_episode = read_frames(render_dir)
    thumb_w, thumb_h = 256, 144
    label_w = 500
    header_h = 56
    row_h = thumb_h + 44
    cols = [
        ("first", lambda r, frames: finite_int(r["first_frame_index"], 0)),
        ("contact", lambda r, frames: finite_int(r["contact_first_frame"], 0)),
        ("reaction", lambda r, frames: finite_int(r["reaction_first_frame"], 0)),
        ("max_err", lambda r, frames: max_error_frame(frames)),
        ("last", lambda r, frames: finite_int(r["last_frame_index"], 194)),
    ]
    sheet = Image.new("RGB", (label_w + thumb_w * len(cols), header_h + row_h * len(fail_rows)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "D248 camera-fail quarantine visual audit", fill=(0, 0, 0))
    for col_idx, (name, _) in enumerate(cols):
        draw.text((label_w + col_idx * thumb_w + 8, 34), name, fill=(0, 0, 0))

    for row_idx, row in enumerate(fail_rows):
        y0 = header_h + row_idx * row_h
        ep = finite_int(row["episode_index"], -1)
        draw.text(
            (8, y0 + 8),
            (
                f"ep {ep} | {row.get('camera_fail_reason', '')}\n"
                f"{row.get('sampling_cell_id', '')} x={float(row['initial_cube_x_m']):.3f} "
                f"y={float(row['initial_cube_y_m']):.3f}\n"
                f"proj={row['projection_inside_frames']}/{row['num_frames']} "
                f"max_err={float(row['centroid_error_px_max']):.2f}px "
                f"clean={row['label_useful_clean_numeric']} over={row['label_overshoot_numeric']}"
            ),
            fill=(0, 0, 0),
        )

        frames = frames_by_episode[ep]
        for col_idx, (name, getter) in enumerate(cols):
            frame_id = getter(row, frames)
            frow = frame_row(frames, frame_id)
            img_path = REPO / frow["source_png"]
            img = Image.open(img_path).convert("RGB").resize((thumb_w, thumb_h))
            cell_x = label_w + col_idx * thumb_w
            sheet.paste(img, (cell_x, y0))
            cell_draw = ImageDraw.Draw(sheet)
            projection = frow.get("projection") if isinstance(frow.get("projection"), dict) else {}
            bbox = projection.get("bbox") if isinstance(projection, dict) else None
            if isinstance(bbox, list) and len(bbox) == 4:
                sx = thumb_w / 1280.0
                sy = thumb_h / 720.0
                rect = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
                rect = [
                    max(0, min(thumb_w - 1, rect[0])),
                    max(0, min(thumb_h - 1, rect[1])),
                    max(0, min(thumb_w - 1, rect[2])),
                    max(0, min(thumb_h - 1, rect[3])),
                ]
                color = (0, 200, 0) if projection.get("inside") else (220, 0, 0)
                cell_draw.rectangle([cell_x + rect[0], y0 + rect[1], cell_x + rect[2], y0 + rect[3]], outline=color, width=2)
            cell_draw.text(
                (cell_x + 6, y0 + thumb_h + 4),
                f"f={frame_id} err={finite_float(frow.get('centroid_error_px'), 0.0):.1f}",
                fill=(0, 0, 0),
            )
    sheet.save(out_path)
    return True


def write_id_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(str(finite_int(row["episode_index"])) for row in rows) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.render_dir / "label_package_d248")
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} exists; use --force or a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.label_csv)
    packaged = classify(rows, holdout_frac=args.clean_holdout_frac, holdout_salt=args.holdout_salt)

    split_manifest = out_dir / "episode_split_manifest.csv"
    fields = CORE_FIELDS + [field for field in rows[0].keys() if field not in CORE_FIELDS]
    write_csv(split_manifest, packaged, fields)

    by_subsplit: dict[str, list[dict[str, Any]]] = {}
    for row in packaged:
        by_subsplit.setdefault(str(row["package_subsplit"]), []).append(row)

    for subsplit, subset_rows in sorted(by_subsplit.items()):
        write_id_list(out_dir / f"{subsplit}_episode_ids.txt", subset_rows)
        write_csv(out_dir / f"{subsplit}.csv", subset_rows, fields)

    fail_rows = by_subsplit.get("quarantine_camera_fail", [])
    write_csv(out_dir / "camera_fail_details.csv", fail_rows, FAIL_FIELDS)
    contact_sheet_path = out_dir / "camera_fail_contact_sheet.png"
    contact_sheet_created = create_contact_sheet(args.render_dir, fail_rows, contact_sheet_path) if fail_rows else False

    summary = {
        "artifact": "cube10cm_top_view_label_package_d248",
        "runtime": "NO_RENDER_NO_TRAINING_NO_DELETE_LABEL_PACKAGING_ONLY",
        "render_dir": str(args.render_dir),
        "label_csv": str(args.label_csv),
        "out_dir": str(out_dir),
        "clean_holdout_frac": args.clean_holdout_frac,
        "clean_holdout_salt": args.holdout_salt,
        "policy": {
            "train_clean_positive": "camera-pass clean useful taps only",
            "eval_clean_holdout": "deterministic 10pct holdout from clean useful taps",
            "eval_overshoot_diagnostic": "camera-pass overshoot episodes excluded from positive BC train",
            "quarantine_camera_fail": "camera contract failures excluded from train/eval",
        },
        "counts": {
            "total": len(packaged),
            "train": sum(1 for r in packaged if r["package_split"] == "train"),
            "eval": sum(1 for r in packaged if r["package_split"] == "eval"),
            "quarantine": sum(1 for r in packaged if r["package_split"] == "quarantine"),
            "by_subsplit": {key: len(value) for key, value in sorted(by_subsplit.items())},
            "by_label_status": {},
        },
        "camera_fail_contact_sheet": str(contact_sheet_path) if contact_sheet_created else None,
        "outputs": {
            "episode_split_manifest": str(split_manifest),
            "camera_fail_details": str(out_dir / "camera_fail_details.csv"),
        },
        "status": "PASS",
    }
    for row in packaged:
        status = str(row["label_status"])
        summary["counts"]["by_label_status"][status] = summary["counts"]["by_label_status"].get(status, 0) + 1

    (out_dir / "split_package_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "[cube10cm-label-package] done "
        f"status=PASS total={summary['counts']['total']} train={summary['counts']['train']} "
        f"eval={summary['counts']['eval']} quarantine={summary['counts']['quarantine']} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
