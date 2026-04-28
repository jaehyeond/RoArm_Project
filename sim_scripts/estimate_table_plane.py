"""Estimate table surface Z in URDF world frame from v6 depth archives.

Uses the calibrated Kinect extrinsics from `sim_scripts/kinect_calib.yaml` to
back-project table-only depth pixels to URDF world, then RANSAC-fits a plane.

Table Z is needed for Isaac Sim Step 5 replay (place table below sponges
without falling through / floating).

No live hardware required — uses depth NPY from collected_data_v6.

Run:
    conda run -n roarm python sim_scripts/estimate_table_plane.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
V6_DIR = REPO / "collected_data_v6"
CALIB = Path(__file__).resolve().parent / "kinect_calib.yaml"

# Pink HSV mask (same as compute_sponge_poses.py).
PINK_LO1, PINK_HI1 = np.array([0, 80, 60]), np.array([12, 255, 255])
PINK_LO2, PINK_HI2 = np.array([165, 80, 60]), np.array([180, 255, 255])

# Crop region: keep central table strip. Avoid far background + robot base.
ROI_Y_LO, ROI_Y_HI = 280, 660  # row indices (720 tall)
ROI_X_LO, ROI_X_HI = 200, 1100  # col indices (1280 wide)

# Depth range (mm) for table surface (~60-150cm from Kinect).
DEPTH_LO_MM, DEPTH_HI_MM = 400, 1400

# Limit on URDF-world-Z for table (exclude robot arm, which reaches high Z).
WORLD_Z_LO_M, WORLD_Z_HI_M = -0.2, 0.1


def load_calib():
    with open(CALIB) as f:
        c = yaml.safe_load(f)
    intr = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return intr, R, t


def backproject(u, v, depth_mm, intr):
    """Pixels + depth (mm) -> 3D points in camera frame (m)."""
    z = depth_mm.astype(np.float64) / 1000.0
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=-1)


def ransac_plane(pts, n_iter=200, thresh_m=0.003, seed=0):
    """RANSAC 3-point plane fit. Returns (normal[3], d, inlier_mask, rmse_m)."""
    rng = np.random.default_rng(seed)
    N = len(pts)
    best_inliers, best_normal, best_d = None, None, None
    best_count = -1
    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p0
        dist = np.abs(pts @ n + d)
        inliers = dist < thresh_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = n
            best_d = d
    # Refine with SVD on inliers.
    ins = pts[best_inliers]
    centroid = ins.mean(axis=0)
    U, S, Vt = np.linalg.svd(ins - centroid, full_matrices=False)
    normal_ref = Vt[-1]
    # Keep normal pointing up (+Z in world after transform — but here still cam frame, handle later).
    d_ref = -normal_ref @ centroid
    dist_ref = np.abs(ins @ normal_ref + d_ref)
    rmse = float(np.sqrt((dist_ref**2).mean()))
    return normal_ref, d_ref, best_inliers, rmse, centroid


def process_episode(ep_dir, intr, R, t):
    """Return (table_points_world (N,3), diag dict)."""
    rgb_path = ep_dir / "rgb_0000.jpg"
    depth_path = ep_dir / "depth_0000.npy"
    if not (rgb_path.exists() and depth_path.exists()):
        return None, {"skip": "missing files"}

    rgb = cv2.imread(str(rgb_path))
    depth = np.load(depth_path)
    if rgb is None or depth is None:
        return None, {"skip": "read failed"}
    if depth.shape[:2] != rgb.shape[:2]:
        return None, {"skip": f"shape mismatch {depth.shape} vs {rgb.shape[:2]}"}

    H, W = depth.shape

    # Build table mask.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, PINK_LO1, PINK_HI1) | cv2.inRange(hsv, PINK_LO2, PINK_HI2)
    # Dilate pink mask to exclude sponge edge pixels.
    pink_dil = cv2.dilate(pink, np.ones((15, 15), np.uint8), iterations=1)

    roi = np.zeros((H, W), dtype=bool)
    roi[ROI_Y_LO:ROI_Y_HI, ROI_X_LO:ROI_X_HI] = True

    depth_valid = (depth >= DEPTH_LO_MM) & (depth <= DEPTH_HI_MM)
    mask = roi & depth_valid & (pink_dil == 0)

    if mask.sum() < 500:
        return None, {"skip": f"too few pts ({int(mask.sum())})"}

    v_idx, u_idx = np.where(mask)
    # Subsample to speed up.
    if len(u_idx) > 20000:
        sel = np.random.default_rng(0).choice(len(u_idx), 20000, replace=False)
        u_idx, v_idx = u_idx[sel], v_idx[sel]

    p_cam = backproject(u_idx, v_idx, depth[v_idx, u_idx], intr)  # (N,3)
    p_world = (R @ p_cam.T).T + t  # (N,3) in URDF world frame

    # Pre-filter by plausible world Z (table can't be above robot base).
    keep = (p_world[:, 2] >= WORLD_Z_LO_M) & (p_world[:, 2] <= WORLD_Z_HI_M)
    p_world = p_world[keep]
    if len(p_world) < 500:
        return None, {"skip": f"after Z filter only {len(p_world)}"}

    return p_world, {"n_pts": len(p_world)}


def main():
    intr, R, t = load_calib()
    print(f"Calib: fx={intr['fx']:.1f} cx={intr['cx']:.1f} t={t.tolist()}")

    all_pts = []
    per_ep = []
    eps_to_use = [f"episode_{i:04d}" for i in range(0, 50, 2)]  # 25 eps (even-indexed)
    for ep_name in eps_to_use:
        ep_dir = V6_DIR / ep_name
        if not ep_dir.exists():
            continue
        pts, diag = process_episode(ep_dir, intr, R, t)
        if pts is None:
            print(f"  {ep_name}: skip ({diag})")
            continue

        # Per-episode RANSAC to reject robot-arm pixels in that frame.
        normal, d, inliers, rmse, centroid = ransac_plane(pts, thresh_m=0.005)
        ins_pts = pts[inliers]
        per_ep.append({
            "ep": ep_name,
            "n_pts": int(pts.shape[0]),
            "n_inliers": int(inliers.sum()),
            "normal": normal.tolist(),
            "d": float(d),
            "rmse_mm": rmse * 1000,
            "z_median_m": float(np.median(ins_pts[:, 2])),
            "z_mean_m": float(ins_pts[:, 2].mean()),
        })
        all_pts.append(ins_pts)

    if not all_pts:
        print("No usable episodes.")
        return

    # Global plane from pooled inliers.
    pooled = np.concatenate(all_pts, axis=0)
    normal, d, inliers, rmse, centroid = ransac_plane(pooled, thresh_m=0.004, n_iter=400)
    ins = pooled[inliers]
    z_median = float(np.median(ins[:, 2]))
    # Flip normal to +Z if needed.
    if normal[2] < 0:
        normal = -normal
        d = -d
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))

    print()
    print("=" * 70)
    print(f"Global pooled plane fit (N={len(pooled)} pts, {len(ins)} inliers)")
    print(f"  normal  = [{normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f}]")
    print(f"  tilt from +Z (world up) = {tilt_deg:.2f} deg")
    print(f"  RMSE      = {rmse*1000:.2f} mm")
    print(f"  z_median  = {z_median*1000:+.2f} mm   (URDF world frame)")
    print(f"  z_mean    = {ins[:,2].mean()*1000:+.2f} mm")
    print(f"  z_std     = {ins[:,2].std()*1000:.2f} mm")

    # Per-episode z consistency.
    zs = [p["z_median_m"] for p in per_ep]
    print()
    print(f"Per-episode table z_median (N={len(per_ep)}):")
    print(f"  mean={np.mean(zs)*1000:+.2f}mm  std={np.std(zs)*1000:.2f}mm")
    print(f"  range=[{min(zs)*1000:+.1f}, {max(zs)*1000:+.1f}]mm")

    # Save result.
    out_path = Path(__file__).resolve().parent / "table_plane.json"
    result = {
        "table_z_urdf_world_m": z_median,
        "table_z_urdf_world_mm": z_median * 1000,
        "normal_world": normal.tolist(),
        "tilt_deg_from_up": tilt_deg,
        "rmse_mm": rmse * 1000,
        "n_pooled_pts": int(len(pooled)),
        "n_inliers": int(len(ins)),
        "n_episodes": len(per_ep),
        "per_episode": per_ep,
        "note": "Run estimate_table_plane.py. Table surface Z in URDF world frame from v6 depth archives.",
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
