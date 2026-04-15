"""Estimate per-episode sponge initial pose from v6 collection data.

Per Plan Step 2 (Real→Sim replay). For each of the 50 v6 episodes we use the
frame-0 RGB+depth from `collected_data_v6/episode_XXXX/` to localise the pink
sponge in the robot base frame.

Pipeline
--------
1. HSV mask the pink sponge in `rgb_0000.jpg`.
2. Back-project masked pixels through Kinect color intrinsics using
   `depth_0000.npy` (transformed_depth, uint16 mm).
3. Median point -> centroid in camera frame.
4. Rotate+translate by Kinect->robot extrinsics (from --calib YAML; falls
   back to a placeholder until Step 4 measurement is available).
5. PCA on the mask points -> principal-axis angle to vertical; warn if >25°.
6. Fallback (no pink detection): use ESP32 FK at grip_close_frame (TCP XYZ
   is a lower-bound proxy for sponge centroid - lifts ~62mm up).

Output
------
`sim_scripts/sponge_poses.json`::

    {
      "meta": {"source": "depth+color", "calib": "approx"|"measured", ...},
      "episodes": {
        "0000": {"pos_m": [x, y, z], "rot_quat_wxyz": [w, x, y, z],
                  "method": "depth", "axis_deg_from_up": 3.4, ...},
        ...
      }
    }

Coordinate convention: robot base (URDF world) frame. +Z up, +X forward from
robot toward the workspace. All positions in metres.

Run
---
    python sim_scripts/compute_sponge_poses.py \\
        --collected collected_data_v6 \\
        --calib sim_scripts/kinect_calib.yaml \\
        --out sim_scripts/sponge_poses.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure repo root import (for fk_roarm_m3 fallback).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data_z_vs_elbow_analysis import fk_roarm_m3  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (Plan Step 2 assumptions; override via CLI/yaml).
# ---------------------------------------------------------------------------

# Azure Kinect 720p color intrinsics (approximate, from calibrate_azure_kinect.py).
DEFAULT_INTRINSICS = {
    "fx": 607.0,
    "fy": 607.0,
    "cx": 638.0,
    "cy": 367.0,
    "width": 1280,
    "height": 720,
}

# Placeholder Kinect->robot-base extrinsics. Camera sits roughly front-right of
# the base, looking down-forward at the table. These numbers are ONLY a stand-in
# until Step 4 measurement (cv2.calibrateCamera + solvePnP on chessboard).
# All sponge poses tagged "approx" should be recomputed once kinect_calib.yaml
# lands.
DEFAULT_EXTRINSICS = {
    "rotation_matrix": [
        # Rotate camera -Z (optical forward) to robot +X, camera +X to -Y, camera +Y to -Z.
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    "translation_m": [0.0, 0.5, 0.3],  # camera ~50 cm to the right, 30 cm up.
    "source": "placeholder",
}

# Pink sponge HSV bounds (OpenCV H: 0-180). Tuned on ep0 inspection;
# widen if detection rate <90%.
PINK_HSV_LOW_1 = np.array([155, 60, 80], dtype=np.uint8)
PINK_HSV_HIGH_1 = np.array([180, 255, 255], dtype=np.uint8)
PINK_HSV_LOW_2 = np.array([0, 60, 80], dtype=np.uint8)   # wrap-around
PINK_HSV_HIGH_2 = np.array([12, 255, 255], dtype=np.uint8)

SPONGE_VERTICAL_Z_M = 0.125  # tallest sponge dimension when upright.
MIN_MASK_PIXELS = 80         # below this -> fall back.
MAX_DEPTH_MM = 1200          # workspace max ~1m; anything further is background.
MIN_DEPTH_MM = 150           # ignore too-close noise.
MAX_AXIS_TILT_DEG = 25.0     # warn threshold.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Extrinsics:
    R: np.ndarray  # 3x3
    t: np.ndarray  # (3,)
    source: str

    @classmethod
    def from_dict(cls, d: dict) -> "Extrinsics":
        R = np.asarray(d["rotation_matrix"], dtype=np.float64)
        t = np.asarray(d["translation_m"], dtype=np.float64)
        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError("extrinsics must be 3x3 R + length-3 t")
        return cls(R=R, t=t, source=d.get("source", "measured"))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_calib(calib_path: Path | None) -> tuple[Intrinsics, Extrinsics]:
    if calib_path and calib_path.exists():
        with calib_path.open() as f:
            data = yaml.safe_load(f)
        intr = Intrinsics(**{**DEFAULT_INTRINSICS, **data.get("intrinsics", {})})
        extr = Extrinsics.from_dict(data.get("extrinsics", DEFAULT_EXTRINSICS))
    else:
        intr = Intrinsics(**DEFAULT_INTRINSICS)
        extr = Extrinsics.from_dict(DEFAULT_EXTRINSICS)
    return intr, extr


def pink_mask(rgb: np.ndarray, depth_mm: np.ndarray | None = None) -> np.ndarray:
    """Return a bool HxW mask of pink sponge pixels.

    If ``depth_mm`` is provided, connected components are ranked by
    median depth (closest first) rather than area — this suppresses
    background pink objects (walls, cloth) which otherwise win by
    pixel count in ~12% of v6 episodes.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, PINK_HSV_LOW_1, PINK_HSV_HIGH_1)
    m2 = cv2.inRange(hsv, PINK_HSV_LOW_2, PINK_HSV_HIGH_2)
    mask = (m1 | m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_MASK_PIXELS // 2]
    if not candidates:
        candidates = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]
    if depth_mm is not None and len(candidates) > 1:
        def median_depth(idx: int) -> float:
            ys, xs = np.nonzero(labels == idx)
            z = depth_mm[ys, xs]
            z = z[(z >= MIN_DEPTH_MM) & (z <= MAX_DEPTH_MM)]
            return float(np.median(z)) if z.size else float("inf")
        keep = min(candidates, key=median_depth)
    else:
        keep = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    return labels == keep


def backproject(mask: np.ndarray, depth_mm: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Return Nx3 (x,y,z) camera-frame points (metres) for masked pixels with valid depth."""
    ys, xs = np.nonzero(mask)
    z_mm = depth_mm[ys, xs]
    valid = (z_mm >= MIN_DEPTH_MM) & (z_mm <= MAX_DEPTH_MM)
    xs, ys, z_mm = xs[valid], ys[valid], z_mm[valid]
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    z_m = z_mm.astype(np.float64) / 1000.0
    x_m = (xs - intr.cx) * z_m / intr.fx
    y_m = (ys - intr.cy) * z_m / intr.fy
    return np.stack([x_m, y_m, z_m], axis=1)


def camera_to_world(pts_cam: np.ndarray, extr: Extrinsics) -> np.ndarray:
    return (extr.R @ pts_cam.T).T + extr.t


def axis_tilt_deg(pts_world: np.ndarray) -> tuple[float, np.ndarray]:
    """PCA principal axis; returns (angle-from-vertical-deg, axis-unit-vec-in-world)."""
    if pts_world.shape[0] < 10:
        return math.nan, np.array([0.0, 0.0, 1.0])
    centered = pts_world - pts_world.mean(axis=0)
    # SVD for PCA principal direction.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[2] < 0:
        axis = -axis
    cos_angle = float(np.clip(axis[2], -1.0, 1.0))
    return math.degrees(math.acos(cos_angle)), axis


def upright_quat_wxyz() -> list[float]:
    """Sponge standing upright with long axis == world +Z."""
    return [1.0, 0.0, 0.0, 0.0]


def fk_fallback_pose(metadata: dict) -> tuple[np.ndarray, str]:
    """When depth segmentation fails, estimate sponge pose from the frame
    at which the gripper closes (TCP ~ top of sponge). Return (pos_m, note).
    The TCP sits roughly at the top of the sponge; we subtract half the sponge
    height to approximate its centroid.
    """
    close_idx = metadata.get("grip_close_frame")
    frames = metadata.get("frames", [])
    if close_idx is None or close_idx >= len(frames):
        close_idx = metadata.get("grip_open_frame")
    if close_idx is None or close_idx >= len(frames):
        return np.full(3, np.nan), "no_grip_frame"
    angles = frames[close_idx]["angles"]
    tcp_mm = fk_roarm_m3(angles)
    pos_m = np.asarray(tcp_mm, dtype=np.float64) / 1000.0
    # Assume sponge upright; centroid is ~half-height below TCP tip.
    pos_m[2] -= SPONGE_VERTICAL_Z_M / 2.0
    return pos_m, f"fk@frame{close_idx}"


def estimate_episode(
    ep_dir: Path, intr: Intrinsics, extr: Extrinsics
) -> dict:
    meta_path = ep_dir / "metadata.json"
    rgb_path = ep_dir / "rgb_0000.jpg"
    depth_path = ep_dir / "depth_0000.npy"
    with meta_path.open() as f:
        metadata = json.load(f)

    ep_id = f"{metadata.get('episode_id', int(ep_dir.name.split('_')[-1])):04d}"
    zone = metadata.get("zone", "?")

    result: dict = {
        "ep_id": ep_id,
        "zone": zone,
        "method": "depth",
        "notes": [],
    }

    # Depth-based.
    if rgb_path.exists() and depth_path.exists():
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        depth_mm = np.load(depth_path).astype(np.uint16)
        mask = pink_mask(rgb, depth_mm)
        n_mask = int(mask.sum())
        result["mask_pixels"] = n_mask
        if n_mask >= MIN_MASK_PIXELS:
            pts_cam = backproject(mask, depth_mm, intr)
            if pts_cam.shape[0] >= MIN_MASK_PIXELS // 2:
                pts_world = camera_to_world(pts_cam, extr)
                centroid = np.median(pts_world, axis=0)
                tilt_deg, axis = axis_tilt_deg(pts_world)
                result["pos_m"] = centroid.tolist()
                result["rot_quat_wxyz"] = upright_quat_wxyz()
                result["axis_deg_from_up"] = tilt_deg if not math.isnan(tilt_deg) else None
                result["pca_axis"] = axis.tolist()
                result["n_depth_pts"] = int(pts_cam.shape[0])
                if not math.isnan(tilt_deg) and tilt_deg > MAX_AXIS_TILT_DEG:
                    result["notes"].append(f"tilt {tilt_deg:.1f}deg > {MAX_AXIS_TILT_DEG}")
                return result
            result["notes"].append("no_valid_depth_in_mask")
        else:
            result["notes"].append(f"mask_pixels={n_mask} below {MIN_MASK_PIXELS}")
    else:
        result["notes"].append("missing_rgb_or_depth")

    # FK fallback.
    pos_m, note = fk_fallback_pose(metadata)
    result["method"] = "fk_fallback"
    result["pos_m"] = pos_m.tolist()
    result["rot_quat_wxyz"] = upright_quat_wxyz()
    result["notes"].append(note)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", type=Path, default=REPO_ROOT / "collected_data_v6")
    ap.add_argument("--calib", type=Path, default=REPO_ROOT / "sim_scripts" / "kinect_calib.yaml")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "sim_scripts" / "sponge_poses.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    args = ap.parse_args()

    intr, extr = load_calib(args.calib)
    print(
        f"[calib] intrinsics fx={intr.fx:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} | "
        f"extrinsics source={extr.source}"
    )
    if extr.source == "placeholder":
        print("[calib] WARNING: placeholder extrinsics - re-run after Step 4 Kinect calibration.")

    ep_dirs = sorted(args.collected.glob("episode_*"))
    if args.limit > 0:
        ep_dirs = ep_dirs[: args.limit]
    if not ep_dirs:
        print(f"[error] no episodes found under {args.collected}")
        return 1

    episodes: dict[str, dict] = {}
    depth_count = 0
    fk_count = 0
    for ep_dir in ep_dirs:
        r = estimate_episode(ep_dir, intr, extr)
        episodes[r["ep_id"]] = r
        if r["method"] == "depth":
            depth_count += 1
        else:
            fk_count += 1
        tilt = r.get("axis_deg_from_up")
        tilt_str = f"tilt={tilt:.1f}" if isinstance(tilt, float) else "tilt=?"
        print(
            f"  ep{r['ep_id']} zone={r['zone']:<9} method={r['method']:<12} "
            f"pos={tuple(round(v, 3) for v in r['pos_m'])} {tilt_str} "
            f"notes={r['notes']}"
        )

    # Cross-episode sanity stats.
    poses = np.array([
        ep["pos_m"] for ep in episodes.values() if not any(math.isnan(v) for v in ep["pos_m"])
    ])
    stats = {}
    if poses.size:
        stats = {
            "n": int(poses.shape[0]),
            "mean_m": poses.mean(axis=0).tolist(),
            "std_m": poses.std(axis=0).tolist(),
            "min_m": poses.min(axis=0).tolist(),
            "max_m": poses.max(axis=0).tolist(),
        }

    out = {
        "meta": {
            "source_dir": str(args.collected),
            "calib_extrinsics_source": extr.source,
            "intrinsics": {
                "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                "width": intr.width, "height": intr.height,
            },
            "extrinsics_R": extr.R.tolist(),
            "extrinsics_t_m": extr.t.tolist(),
            "counts": {"depth": depth_count, "fk_fallback": fk_count},
            "stats": stats,
        },
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} ({depth_count} depth, {fk_count} fk_fallback)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
