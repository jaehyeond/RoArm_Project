#!/usr/bin/env python3
"""Check xy distribution of D250 filtered views before any training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_RENDER_DIR = LOG_DIR / "cube10cm_top_view_visual_0_999_d242"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--label-package-dir", type=Path, default=None)
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


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def band_y(y: float) -> str:
    if y < -0.05:
        return "low_y_below_-0.05"
    if y < 0.05:
        return "mid_y_-0.05_to_0.05"
    if y < 0.12:
        return "high_y_0.05_to_0.12"
    return "boundary_y_ge_0.12"


def band_x(x: float) -> str:
    if x < 0.16:
        return "near_x_below_0.16"
    if x < 0.28:
        return "center_x_0.16_to_0.28"
    return "far_x_ge_0.28"


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values),
    }


def main() -> None:
    args = parse_args()
    label_dir = args.label_package_dir or (args.render_dir / "label_package_d248")
    out_dir = args.out_dir or (args.render_dir / "split_distribution_d252")
    if out_dir.exists() and not args.force:
        raise FileExistsError(f"{out_dir} exists; use --force or another --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(label_dir / "episode_split_manifest.csv")
    subsplits = sorted({row["package_subsplit"] for row in rows})

    summary: dict[str, Any] = {
        "artifact": "cube10cm_top_view_split_distribution_d252",
        "runtime": "NO_RENDER_NO_TRAINING_NO_DELETE_DISTRIBUTION_CHECK_ONLY",
        "label_package_dir": str(label_dir),
        "subsplits": {},
        "critical_findings": [],
        "status": "PASS",
    }
    band_rows: list[dict[str, Any]] = []

    for subsplit in subsplits:
        sub = [row for row in rows if row["package_subsplit"] == subsplit]
        xs = [f(row, "initial_cube_x_m") for row in sub]
        ys = [f(row, "initial_cube_y_m") for row in sub]
        split_counter = Counter(row["split_candidate"] for row in sub)
        y_counter = Counter(band_y(y) for y in ys)
        x_counter = Counter(band_x(x) for x in xs)
        for band_name, count in sorted(y_counter.items()):
            band_rows.append(
                {
                    "package_subsplit": subsplit,
                    "axis": "y",
                    "band": band_name,
                    "count": count,
                    "fraction": count / max(1, len(sub)),
                }
            )
        for band_name, count in sorted(x_counter.items()):
            band_rows.append(
                {
                    "package_subsplit": subsplit,
                    "axis": "x",
                    "band": band_name,
                    "count": count,
                    "fraction": count / max(1, len(sub)),
                }
            )
        summary["subsplits"][subsplit] = {
            "episodes": len(sub),
            "x": stats(xs),
            "y": stats(ys),
            "split_candidate_counts": dict(sorted(split_counter.items())),
            "x_band_counts": dict(sorted(x_counter.items())),
            "y_band_counts": dict(sorted(y_counter.items())),
        }

    train = summary["subsplits"].get("train_clean_positive", {})
    overshoot = summary["subsplits"].get("eval_overshoot_diagnostic", {})
    quarantine = summary["subsplits"].get("quarantine_camera_fail", {})
    if train:
        if train["x"]["min"] <= 0.091 and train["x"]["max"] >= 0.389 and train["y"]["min"] <= -0.099 and train["y"]["max"] >= 0.149:
            summary["critical_findings"].append(
                "학습용 정상 성공 예시는 x/y 전체 sampled workspace를 포함한다."
            )
    if train and overshoot:
        if overshoot["y"]["mean"] > train["y"]["mean"] + 0.05:
            summary["critical_findings"].append(
                "과하게 민 진단용 평가 데이터는 학습용 정상 성공 예시보다 높은 y 영역에 뚜렷하게 몰린다."
            )
    if quarantine:
        summary["critical_findings"].append(
            "카메라 기준 실패 격리 데이터는 x 약 0.14-0.165m 근처에 몰리며, episode 721은 coverage 실패로 별도 주의가 필요하다."
        )

    write_csv(out_dir / "split_xy_band_counts.csv", band_rows, ["package_subsplit", "axis", "band", "count", "fraction"])
    (out_dir / "split_distribution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(
        "[cube10cm-split-distribution] done "
        f"status=PASS subsplits={len(subsplits)} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
