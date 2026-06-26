#!/usr/bin/env python3
"""Visualize D256 train-clean teacher feature distributions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
)
D242_ROOT = RUNTIME_ROOT / "cube10cm_top_view_visual_0_999_d242"
DEFAULT_CSV = D242_ROOT / "rl_transition_preflight_d256" / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_OUT_DIR = RUNTIME_ROOT / "d256_feature_distribution_viz_d262"
D261_NOIK = (
    RUNTIME_ROOT
    / "teacher_rollout_probe_d261_envtarget_posx"
    / "tap10cm"
    / "teacher_rollout_probe_summary_d261_envtarget_posx.json"
)
D261_IK = (
    RUNTIME_ROOT
    / "teacher_rollout_probe_d261_envtarget_posx_ik"
    / "tap10cm"
    / "teacher_rollout_probe_summary_d261_envtarget_posx_ik.json"
)

FEATURE_COLUMNS = [
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_local_x_m",
    "cube_local_y_m",
    "cube_local_z_m",
    "tcp_local_x_m",
    "tcp_local_y_m",
    "tcp_local_z_m",
    "target_local_x_m",
    "target_local_y_m",
    "target_local_z_m",
    "tcp_to_cube_x_m",
    "tcp_to_cube_y_m",
    "tcp_to_cube_z_m",
    "target_to_tcp_x_m",
    "target_to_tcp_y_m",
    "target_to_tcp_z_m",
    "target_to_cube_x_m",
    "target_to_cube_y_m",
    "target_to_cube_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
]
TARGET_COLUMNS = [
    "joint_delta_0_rad",
    "joint_delta_1_rad",
    "joint_delta_2_rad",
    "joint_delta_3_rad",
    "joint_delta_4_rad",
]

SELECTED_OVERLAY_FEATURES = [
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "tcp_local_z_m",
    "target_to_tcp_x_m",
    "target_to_tcp_y_m",
    "target_to_tcp_z_m",
    "tcp_to_cube_x_m",
    "tcp_to_cube_y_m",
    "tcp_to_cube_z_m",
    "target_local_z_m",
]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _float(row: dict[str, str], col: str) -> float:
    value = row[col]
    return float(value) if value else float("nan")


def load_table(csv_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    columns = FEATURE_COLUMNS + TARGET_COLUMNS
    values: dict[str, list[float]] = {col: [] for col in columns}
    episode_ids: set[int] = set()
    subsplits: Counter[str] = Counter()
    labels: Counter[str] = Counter()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        missing = [col for col in columns if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing columns in {csv_path}: {missing}")
        for row in reader:
            episode_ids.add(int(row["episode_index"]))
            subsplits[row.get("package_subsplit", "")] += 1
            labels[row.get("label_status", "")] += 1
            for col in columns:
                values[col].append(_float(row, col))

    arrays = {col: np.asarray(vals, dtype=np.float64) for col, vals in values.items()}
    meta = {
        "rows": int(len(next(iter(arrays.values())))),
        "episodes": int(len(episode_ids)),
        "package_subsplit_counts": dict(subsplits),
        "label_status_counts": dict(labels),
    }
    return arrays, meta


def stats_for(arr: np.ndarray) -> dict[str, float]:
    qs = np.nanquantile(arr, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p50": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def load_env_ranges(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    ranges = {}
    for item in data.get("feature_alignment", []):
        ranges[str(item["feature"])] = (float(item["env_min"]), float(item["env_max"]))
    return ranges


def save_workspace_xy(arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "d256_workspace_xy_distribution.png"
    rows = len(arrays["cube_local_x_m"])
    max_points = min(rows, 30000)
    idx = np.linspace(0, rows - 1, max_points, dtype=np.int64)
    phase = arrays["phase_alpha"][idx]

    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    sc = ax.scatter(
        arrays["tcp_local_x_m"][idx],
        arrays["tcp_local_y_m"][idx],
        c=phase,
        s=4,
        alpha=0.25,
        cmap="viridis",
        label="TCP samples",
        rasterized=True,
    )
    ax.scatter(
        arrays["cube_local_x_m"][idx],
        arrays["cube_local_y_m"][idx],
        s=6,
        alpha=0.12,
        color="#2f6fef",
        label="Cube samples",
        rasterized=True,
    )
    ax.scatter(
        arrays["target_local_x_m"][idx],
        arrays["target_local_y_m"][idx],
        s=6,
        alpha=0.12,
        color="#d62728",
        label="Target samples",
        rasterized=True,
    )
    ax.set_title("D256 train-clean XY distribution: cube / target / TCP")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", markerscale=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("phase_alpha")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_joint_state_hist(arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "d256_arm_joint_state_distribution.png"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.ravel()
    for i in range(5):
        col = f"arm_joint_{i}_rad"
        ax = axes_flat[i]
        arr = arrays[col]
        st = stats_for(arr)
        ax.hist(arr, bins=90, color="#4c78a8", alpha=0.82)
        ax.axvline(st["p01"], color="#222222", linestyle="--", linewidth=1.1, label="p01/p99" if i == 0 else None)
        ax.axvline(st["p99"], color="#222222", linestyle="--", linewidth=1.1)
        ax.axvline(st["p50"], color="#f58518", linewidth=1.3, label="median" if i == 0 else None)
        ax.set_title(col)
        ax.set_xlabel("rad")
        ax.grid(True, alpha=0.2)
    axes_flat[-1].axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle("D256 train-clean arm joint state distribution", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_joint_delta_hist(arrays: dict[str, np.ndarray], out_dir: Path, clip_rad: float) -> Path:
    path = out_dir / "d256_joint_delta_distribution.png"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes_flat = axes.ravel()
    for i in range(5):
        col = f"joint_delta_{i}_rad"
        ax = axes_flat[i]
        arr = arrays[col]
        st = stats_for(arr)
        exceed = float(np.mean(np.abs(arr) > clip_rad))
        ax.hist(arr, bins=100, color="#54a24b", alpha=0.82)
        ax.axvline(-clip_rad, color="#d62728", linestyle="--", linewidth=1.2, label="+/-0.04 cap" if i == 0 else None)
        ax.axvline(clip_rad, color="#d62728", linestyle="--", linewidth=1.2)
        ax.axvline(st["p50"], color="#f58518", linewidth=1.3, label="median" if i == 0 else None)
        ax.set_title(f"{col} | abs>cap {exceed:.3f}")
        ax.set_xlabel("rad")
        ax.grid(True, alpha=0.2)
    axes_flat[-1].axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle("D256 train-clean target joint-delta distribution", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_relative_geometry_hist(arrays: dict[str, np.ndarray], out_dir: Path) -> Path:
    path = out_dir / "d256_tcp_target_relative_geometry_distribution.png"
    cols = [
        "tcp_to_cube_x_m",
        "tcp_to_cube_y_m",
        "tcp_to_cube_z_m",
        "target_to_tcp_x_m",
        "target_to_tcp_y_m",
        "target_to_tcp_z_m",
        "tcp_local_z_m",
        "target_local_z_m",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, col in zip(axes.ravel(), cols):
        arr = arrays[col]
        st = stats_for(arr)
        ax.hist(arr, bins=90, color="#b279a2", alpha=0.82)
        ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.axvline(st["p01"], color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(st["p99"], color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(st["p50"], color="#f58518", linewidth=1.2)
        ax.set_title(col)
        ax.set_xlabel("m")
        ax.grid(True, alpha=0.2)
    fig.suptitle("D256 train-clean TCP / target relative geometry", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_d261_overlay(
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    noik_ranges: dict[str, tuple[float, float]],
    ik_ranges: dict[str, tuple[float, float]],
) -> Path:
    path = out_dir / "d256_hist_with_d261_env_range_overlay.png"
    fig, axes = plt.subplots(4, 4, figsize=(17, 13), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, col in zip(axes_flat, SELECTED_OVERLAY_FEATURES):
        arr = arrays[col]
        st = stats_for(arr)
        extra_vals = [st["min"], st["max"]]
        if col in noik_ranges:
            extra_vals.extend(noik_ranges[col])
        if col in ik_ranges:
            extra_vals.extend(ik_ranges[col])
        x_min = min(extra_vals)
        x_max = max(extra_vals)
        pad = max((x_max - x_min) * 0.05, 1.0e-6)
        ax.hist(arr, bins=90, color="#9ecae9", alpha=0.85, density=True, label="D256 train-clean")
        ax.axvspan(st["p01"], st["p99"], color="#2f6fef", alpha=0.14, label="D256 p01-p99")
        if col in noik_ranges:
            lo, hi = noik_ranges[col]
            ax.axvspan(lo, hi, color="#ff7f0e", alpha=0.22, label="D261 no-IK env range")
        if col in ik_ranges:
            lo, hi = ik_ranges[col]
            ax.axvspan(lo, hi, color="#d62728", alpha=0.18, label="D261 IK env range")
        ax.axvline(st["p50"], color="#111111", linewidth=1.0)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.2)
    for ax in axes_flat[len(SELECTED_OVERLAY_FEATURES) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("D256 feature histograms with D261 live-env range overlay", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_normalized_support_bars(
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    noik_ranges: dict[str, tuple[float, float]],
    ik_ranges: dict[str, tuple[float, float]],
) -> Path:
    path = out_dir / "d256_vs_d261_normalized_support_bars.png"
    features = [
        "arm_joint_0_rad",
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
        "tcp_local_z_m",
        "target_to_tcp_x_m",
        "target_to_tcp_y_m",
        "target_to_tcp_z_m",
        "tcp_to_cube_x_m",
        "tcp_to_cube_y_m",
        "tcp_to_cube_z_m",
    ]
    y = np.arange(len(features), dtype=np.float64)
    x_min, x_max = -1.5, 3.5

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.axvspan(0.0, 1.0, color="#d9d9d9", alpha=0.35, label="D256 min-max support")
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.axvline(1.0, color="#555555", linewidth=1.0)

    def norm_pair(col: str, pair: tuple[float, float]) -> tuple[float, float] | None:
        st = stats_for(arrays[col])
        denom = st["max"] - st["min"]
        if abs(denom) < 1.0e-12:
            return None
        return ((pair[0] - st["min"]) / denom, (pair[1] - st["min"]) / denom)

    for row, col in enumerate(features):
        st = stats_for(arrays[col])
        denom = st["max"] - st["min"]
        if abs(denom) < 1.0e-12:
            continue
        p01 = (st["p01"] - st["min"]) / denom
        p99 = (st["p99"] - st["min"]) / denom
        ax.plot([p01, p99], [row, row], color="#2f6fef", linewidth=7, alpha=0.78)

        if col in noik_ranges:
            noik = norm_pair(col, noik_ranges[col])
            if noik is not None:
                lo, hi = noik
                ax.plot(
                    [max(x_min, lo), min(x_max, hi)],
                    [row + 0.18, row + 0.18],
                    color="#ff7f0e",
                    linewidth=4,
                    solid_capstyle="round",
                    label="D261 no-IK range" if row == 0 else None,
                )
                if lo < x_min or hi > x_max:
                    ax.scatter(
                        [x_min if lo < x_min else x_max],
                        [row + 0.18],
                        color="#ff7f0e",
                        marker="<" if lo < x_min else ">",
                        s=36,
                    )

        if col in ik_ranges:
            ik = norm_pair(col, ik_ranges[col])
            if ik is not None:
                lo, hi = ik
                ax.plot(
                    [max(x_min, lo), min(x_max, hi)],
                    [row - 0.18, row - 0.18],
                    color="#d62728",
                    linewidth=4,
                    solid_capstyle="round",
                    label="D261 IK range" if row == 0 else None,
                )
                if lo < x_min or hi > x_max:
                    ax.scatter(
                        [x_min if lo < x_min else x_max],
                        [row - 0.18],
                        color="#d62728",
                        marker="<" if lo < x_min else ">",
                        s=36,
                    )

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("normalized to D256 train-clean min-max: 0=min, 1=max")
    ax.set_title("D261 live-env feature ranges vs D256 support (normalized, clipped at plot edges)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_summary(
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
    out_dir: Path,
    paths: dict[str, Path],
    clip_rad: float,
    noik_ranges: dict[str, tuple[float, float]],
    ik_ranges: dict[str, tuple[float, float]],
) -> tuple[Path, Path]:
    stats = {col: stats_for(arrays[col]) for col in FEATURE_COLUMNS + TARGET_COLUMNS}
    raw_abs = np.concatenate([np.abs(arrays[col]) for col in TARGET_COLUMNS])
    clip_exceed_rate = float(np.mean(raw_abs > clip_rad))
    payload = {
        "artifact": "d256_feature_distribution_viz_d262",
        "source_csv": _rel(DEFAULT_CSV),
        "rows": meta["rows"],
        "episodes": meta["episodes"],
        "package_subsplit_counts": meta["package_subsplit_counts"],
        "label_status_counts": meta["label_status_counts"],
        "target_clip_rad_reference": float(clip_rad),
        "joint_delta_abs_clip_exceed_rate": clip_exceed_rate,
        "feature_stats": {col: stats[col] for col in FEATURE_COLUMNS},
        "target_stats": {col: stats[col] for col in TARGET_COLUMNS},
        "d261_overlay_available": {
            "no_ik": bool(noik_ranges),
            "ik_reset": bool(ik_ranges),
        },
        "plots": {name: _rel(path) for name, path in paths.items()},
    }
    json_path = out_dir / "d256_feature_distribution_summary_d262.json"
    md_path = out_dir / "d256_feature_distribution_summary_d262.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    key_cols = [
        "cube_local_x_m",
        "cube_local_y_m",
        "tcp_local_z_m",
        "target_to_tcp_x_m",
        "target_to_tcp_y_m",
        "target_to_tcp_z_m",
        "arm_joint_0_rad",
        "arm_joint_1_rad",
        "arm_joint_2_rad",
        "arm_joint_3_rad",
        "arm_joint_4_rad",
    ]
    lines = [
        "# D256 Feature Distribution Visualization D262",
        "",
        f"- source csv: `{_rel(DEFAULT_CSV)}`",
        f"- rows / episodes: `{meta['rows']}` / `{meta['episodes']}`",
        f"- label counts: `{meta['label_status_counts']}`",
        f"- joint delta abs > `{clip_rad}` rate: `{clip_exceed_rate}`",
        "",
        "## Plots",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{name}`: `{_rel(path)}`")
    lines.extend(["", "## Key Feature Quantiles", ""])
    lines.append("| feature | min | p01 | p50 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for col in key_cols:
        st = stats[col]
        lines.append(
            f"| `{col}` | `{st['min']:.6g}` | `{st['p01']:.6g}` | "
            f"`{st['p50']:.6g}` | `{st['p99']:.6g}` | `{st['max']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Reading The Overlay",
            "",
            "- Blue histogram: D256 train-clean teacher feature distribution.",
            "- Blue band: D256 p01-p99 support.",
            "- Orange band: D261 live env range without IK reset.",
            "- Red band: D261 live env range with IK reset.",
            "- When orange/red bands sit outside the blue mass, the D257 MLP teacher is extrapolating.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--d261_noik_summary", type=Path, default=D261_NOIK)
    parser.add_argument("--d261_ik_summary", type=Path, default=D261_IK)
    parser.add_argument("--target_clip_rad", type=float, default=0.04)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays, meta = load_table(args.csv)
    noik_ranges = load_env_ranges(args.d261_noik_summary)
    ik_ranges = load_env_ranges(args.d261_ik_summary)

    paths = {
        "workspace_xy": save_workspace_xy(arrays, out_dir),
        "arm_joint_state": save_joint_state_hist(arrays, out_dir),
        "joint_delta": save_joint_delta_hist(arrays, out_dir, float(args.target_clip_rad)),
        "relative_geometry": save_relative_geometry_hist(arrays, out_dir),
        "d261_overlay": save_d261_overlay(arrays, out_dir, noik_ranges, ik_ranges),
        "d261_normalized_support": save_normalized_support_bars(arrays, out_dir, noik_ranges, ik_ranges),
    }
    json_path, md_path = save_summary(
        arrays,
        meta,
        out_dir,
        paths,
        float(args.target_clip_rad),
        noik_ranges,
        ik_ranges,
    )

    print(
        "d256_feature_distribution_viz "
        f"rows={meta['rows']} episodes={meta['episodes']} "
        f"out_dir={_rel(out_dir)} summary={_rel(json_path)} brief={_rel(md_path)}"
    )
    for name, path in paths.items():
        print(f"plot {name}: {_rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
