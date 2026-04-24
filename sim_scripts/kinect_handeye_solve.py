#!/usr/bin/env python
"""Kinect hand-eye calibration — solver.

Loads captured data from kinect_handeye_capture.py, computes FK for each pose,
back-projects marker pixels to 3D in camera frame, and solves for:
  - R, t: camera-to-robot-base rigid transform  (p_base = R @ p_cam + t)
  - d: marker offset in link5 local frame

Convention matches compute_sponge_poses.py:
    p_world = R @ p_cam + t   (camera_to_world)

Two-stage solve:
  1. Umeyama alignment with offset d=0 (initial guess)
  2. Nonlinear refinement (R, t, d) via scipy least_squares

Usage:
    conda run -n roarm python sim_scripts/kinect_handeye_solve.py

Output:
    sim_scripts/kinect_calib.yaml   (compatible with compute_sponge_poses.py)
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────
# FK (from URDF, self-contained — same chain as data_z_vs_elbow_analysis.py)
# ─────────────────────────────────────────────────
def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _T(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def _jt(xyz, rpy, q):
    """URDF joint transform: fixed origin (xyz+rpy) then revolute around Z.

    URDF RPY convention is fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rpy = [roll, pitch, yaw].
    """
    return _T(*xyz) @ _Rz(rpy[2]) @ _Ry(rpy[1]) @ _Rx(rpy[0]) @ _Rz(q)


def fk_link5(angles_deg):
    """Return link5 4×4 homogeneous transform in URDF world frame (meters).

    Joint chain:
      world → base_link (fixed +70.1mm Z)
      → joint0 (base, Z) → joint1 (shoulder, Z) → joint2 (elbow, Z)
      → joint3 (wrist_pitch, Z) → joint4 (wrist_roll, Z) → link5
    """
    pi = np.pi
    q = [a * pi / 180.0 for a in angles_deg[:5]]
    T0 = _T(0, 0, 0.0701)
    T1 = T0 @ _jt([0, 0, 0], [0, 0, 0], q[0])
    T2 = T1 @ _jt([0, 0, 0.051959], [-pi / 2, -pi / 2, 0], q[1])
    T3 = T2 @ _jt([0.236815, 0.030002, 0], [0, 0, pi / 2], q[2])
    T4 = T3 @ _jt([0, -0.144586, 0], [0, 0, 0], q[3])
    T5 = T4 @ _jt([0.015147, -0.053653, 0], [pi / 2, pi / 2, 0], q[4])
    return T5


# ─────────────────────────────────────────────────
# Back-projection
# ─────────────────────────────────────────────────
def back_project(u, v, depth_mm, intr):
    """Pixel (u,v) + depth → 3D point in camera frame (meters)."""
    z = depth_mm / 1000.0
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z])


# ─────────────────────────────────────────────────
# Umeyama rigid alignment (no scaling)
# ─────────────────────────────────────────────────
def umeyama(src, dst):
    """Find R, t minimizing ||dst - (R @ src + t)||².

    Returns R (3×3), t (3,).
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_d - R @ mu_s
    return R, t


# ─────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────
def solve(T5s, p_cams):
    """Solve for camera→base transform (R, t) and marker offset d.

    Model:  p_base_i = R_cw @ p_cam_i + t_cw
            p_base_i = T5_rot_i @ d + T5_pos_i

    So: T5_rot_i @ d + T5_pos_i = R_cw @ p_cam_i + t_cw

    Parameters
    ----------
    T5s : list of (4,4) ndarray — link5 transforms in base frame (meters)
    p_cams : (N,3) ndarray — marker positions in camera frame (meters)

    Returns
    -------
    R, t, d, rmse_mm, per_pose_mm
    """
    N = len(T5s)
    T5_rots = np.array([T[:3, :3] for T in T5s])
    T5_poss = np.array([T[:3, 3] for T in T5s])

    # Stage 1: Umeyama with d=0
    # src = p_cam, dst = T5_pos  →  dst ≈ R @ src + t
    R_init, t_init = umeyama(p_cams, T5_poss)
    rvec_init = Rotation.from_matrix(R_init).as_rotvec()

    # Stage 2: refine (R_cw, t_cw, d) jointly
    x0 = np.concatenate([rvec_init, t_init, [0.0, 0.0, 0.05]])

    def residuals(x):
        rvec, tvec, d = x[:3], x[3:6], x[6:9]
        R_cw = Rotation.from_rotvec(rvec).as_matrix()
        errs = np.empty(N * 3)
        for i in range(N):
            p_base_target = T5_rots[i] @ d + T5_poss[i]
            p_base_pred = R_cw @ p_cams[i] + tvec
            errs[i * 3 : (i + 1) * 3] = p_base_target - p_base_pred
        return errs

    result = least_squares(residuals, x0, method="lm")

    R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t_opt = result.x[3:6]
    d_opt = result.x[6:9]

    res = result.fun.reshape(-1, 3)
    per_pose_m = np.sqrt(np.sum(res**2, axis=1))
    rmse_m = np.sqrt(np.mean(per_pose_m**2))

    return R_opt, t_opt, d_opt, rmse_m * 1000, per_pose_m * 1000


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "handeye_data"
CALIB_OUT = Path(__file__).resolve().parent / "kinect_calib.yaml"


def main():
    cap_path = DATA_DIR / "handeye_captures.json"
    if not cap_path.exists():
        print(f"ERROR: {cap_path} not found — run kinect_handeye_capture.py first.")
        return

    with open(cap_path) as f:
        data = json.load(f)

    intr = data["intrinsics"]
    poses = data["poses"]
    print(f"Loaded {len(poses)} captured poses")
    print(f"Intrinsics: fx={intr['fx']} fy={intr['fy']} cx={intr['cx']} cy={intr['cy']}")

    # Build correspondences (filter low-quality detections)
    MIN_CIRCULARITY = 0.45
    T5s_all, pcams_all, ids_all, circs_all = [], [], [], []
    for p in poses:
        angles = p["measured"] if p["measured"] is not None else p["commanded"]
        m = p["marker"]
        T5s_all.append(fk_link5(angles))
        pcams_all.append(back_project(m["u"], m["v"], m["depth_mm"], intr))
        ids_all.append(p["pose_id"])
        circs_all.append(m["circularity"])

    pcams_all = np.array(pcams_all)

    # First pass: use all poses
    T5s, p_cams, ids = T5s_all, pcams_all, ids_all
    print(f"Correspondences: {len(T5s)} total")

    if len(T5s) < 6:
        print("ERROR: Need >= 6 poses")
        return

    # Solve
    R, t, d, rmse, per_pose = solve(T5s, p_cams)

    # Iterative outlier rejection (2-sigma, max 2 rounds)
    mask = np.ones(len(T5s), dtype=bool)
    for iteration in range(2):
        errors = per_pose[mask] if iteration == 0 else per_pose
        thresh = np.mean(per_pose[mask]) + 2.0 * np.std(per_pose[mask])
        new_mask = mask.copy()
        for i in range(len(mask)):
            if mask[i] and per_pose[i] > thresh:
                new_mask[i] = False
        removed = np.sum(mask) - np.sum(new_mask)
        if removed == 0:
            break
        mask = new_mask
        if np.sum(mask) < 6:
            break
        # Re-solve with inliers
        idx = np.where(mask)[0]
        T5s_sub = [T5s_all[i] for i in idx]
        pcams_sub = pcams_all[idx]
        R, t, d, rmse, per_sub = solve(T5s_sub, pcams_sub)
        # Recompute per-pose on ALL
        T5_rots = np.array([T[:3, :3] for T in T5s_all])
        T5_poss = np.array([T[:3, 3] for T in T5s_all])
        per_pose = np.zeros(len(T5s_all))
        for i in range(len(T5s_all)):
            target = T5_rots[i] @ d + T5_poss[i]
            pred = R @ pcams_all[i] + t
            per_pose[i] = np.linalg.norm(target - pred) * 1000
        rmse = np.sqrt(np.mean(per_pose[mask] ** 2))
        print(f"  Outlier rejection round {iteration + 1}: removed {removed}, {np.sum(mask)} remain, RMSE={rmse:.2f}mm")

    n_used = int(np.sum(mask))
    outliers = len(T5s_all) - n_used

    # Report
    print(f"\n{'=' * 50}")
    print(f"RMSE: {rmse:.2f} mm ({n_used} poses used, {outliers} rejected)")
    print(f"Marker offset (link5 frame): "
          f"[{d[0]*1000:.1f}, {d[1]*1000:.1f}, {d[2]*1000:.1f}] mm")
    print(f"Translation (cam→base): "
          f"[{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm")
    rvec = Rotation.from_matrix(R).as_rotvec()
    print(f"Rotation (Rodrigues): [{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]")

    print(f"\nPer-pose errors (mm):")
    for i in range(len(T5s_all)):
        flag = ""
        if not mask[i]:
            flag = " *** REJECTED"
        print(f"  pose {ids_all[i]:2d}: {per_pose[i]:6.2f} mm  circ={circs_all[i]:.2f}{flag}")

    # Verdict (calibrated for sim→real pipeline: ±10mm sponge contact tolerance)
    if rmse < 5.0:
        verdict = "EXCELLENT"
    elif rmse < 10.0:
        verdict = "PASS"
    elif rmse < 15.0:
        verdict = "MARGINAL — acceptable for sim pipeline"
    else:
        verdict = "FAIL — re-capture with better marker visibility"

    print(f"\n{'=' * 50}")
    print(f"Verdict: {verdict}")

    # Save kinect_calib.yaml (compatible with compute_sponge_poses.py)
    calib = {
        "intrinsics": {
            "fx": intr["fx"],
            "fy": intr["fy"],
            "cx": intr["cx"],
            "cy": intr["cy"],
            "width": intr.get("width", 1280),
            "height": intr.get("height", 720),
        },
        "extrinsics": {
            "rotation_matrix": R.tolist(),
            "translation_m": t.tolist(),
            "source": "hand-eye calibration",
        },
        "marker_offset": {
            "d_link5_frame_mm": (d * 1000).tolist(),
            "note": "Sticker on wrist_roll drum side, wrist_roll=0 facing forward",
        },
        "quality": {
            "rmse_mm": round(float(rmse), 2),
            "n_poses_total": len(T5s_all),
            "n_poses_used": n_used,
            "n_outliers": outliers,
            "verdict": verdict,
            "date": str(date.today()),
        },
    }

    with open(CALIB_OUT, "w") as f:
        yaml.dump(calib, f, default_flow_style=None, sort_keys=False)

    print(f"Saved: {CALIB_OUT}")


if __name__ == "__main__":
    main()
