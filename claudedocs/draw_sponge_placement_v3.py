"""Overlay recommended 4-sponge placement onto a Kinect frame.

World frame (RoArm M3 base):
  X = forward (away from robot base)
  Y = lateral (sign per kinect_calib.yaml hand-eye)
  Z = up (table top ~ z=0)

Calibration: P_base = R @ P_cam + t  ->  P_cam = R^T @ (P_base - t)
Then pinhole: u = fx*P_cam.x/P_cam.z + cx, v = fy*P_cam.y/P_cam.z + cy
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def load_calib():
    with open(REPO / "sim_scripts/kinect_calib.yaml") as f:
        c = yaml.safe_load(f)
    K = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return K, R, t


def world_to_pixel(P_w, K, R, t):
    """P_w: (3,) world-frame point in meters."""
    P_c = R.T @ (np.asarray(P_w, dtype=np.float64) - t)
    if P_c[2] <= 0:
        return None
    u = K["fx"] * P_c[0] / P_c[2] + K["cx"]
    v = K["fy"] * P_c[1] / P_c[2] + K["cy"]
    return int(round(u)), int(round(v))


def draw_polygon(img, pts_world, K, R, t, color, thickness=2, z=0.0):
    pts = []
    for x, y in pts_world:
        p = world_to_pixel((x, y, z), K, R, t)
        if p is None:
            return
        pts.append(p)
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness)


def draw_sponge_footprint(img, cx_m, cy_m, orient, K, R, t, label):
    """orient = 'X' (long along X axis, 125mm) or 'Y' (long along Y axis)."""
    if orient == "X":
        dx, dy = 0.125 / 2, 0.022 / 2
    else:
        dx, dy = 0.022 / 2, 0.125 / 2
    corners = [
        (cx_m - dx, cy_m - dy),
        (cx_m + dx, cy_m - dy),
        (cx_m + dx, cy_m + dy),
        (cx_m - dx, cy_m + dy),
    ]
    draw_polygon(img, corners, K, R, t, color=(0, 255, 0), thickness=3, z=0.0)
    cen = world_to_pixel((cx_m, cy_m, 0.0), K, R, t)
    if cen is not None:
        cv2.drawMarker(img, cen, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(img, label, (cen[0] + 10, cen[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(img, label, (cen[0] + 10, cen[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        sorted((REPO / "claudedocs").glob("sponge_check_*.png"))[-1]
    img = cv2.imread(str(src))
    if img is None:
        print(f"FAIL: cannot read {src}")
        return 1
    print(f"Source image: {src}")

    K, R, t = load_calib()

    # Tower destination (HASH1_CENTER + L1/L2 footprints)
    HASH1 = (0.280, 0.000)
    DY_L1 = 0.0435  # L1 sp1/sp2 along Y
    DX_L2 = 0.0335  # L2 sp3/sp4 along X
    # EXCLUSION (zone where # tower will build — must be empty)
    EX_X = (0.2125, 0.3475)
    EX_Y = (-0.0675, +0.0675)
    excl_corners = [
        (EX_X[0], EX_Y[0]), (EX_X[1], EX_Y[0]),
        (EX_X[1], EX_Y[1]), (EX_X[0], EX_Y[1]),
    ]
    draw_polygon(img, excl_corners, K, R, t, color=(0, 255, 255), thickness=2)
    cen = world_to_pixel((HASH1[0], HASH1[1], 0.0), K, R, t)
    if cen is not None:
        cv2.putText(img, "EXCLUSION (# tower zone)", (cen[0] - 100, cen[1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.drawMarker(img, cen, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, "(+280, 0)", (cen[0] + 8, cen[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Recommended 4 source positions (X mm, Y mm), diagonal orient diversity
    SOURCES = [
        ("R1 near-Y-",  +0.200, -0.175, "Y", "(+200,-175) Y"),
        ("R2 near-Y+",  +0.200, +0.135, "X", "(+200,+135) X"),
        ("R3 far-Y-",   +0.380, -0.160, "X", "(+380,-160) X"),
        ("R4 far-Y+",   +0.380, +0.125, "Y", "(+380,+125) Y"),
    ]
    for tag, x, y, orient, lbl in SOURCES:
        draw_sponge_footprint(img, x, y, orient, K, R, t, lbl)
    cv2.putText(img, "PLACE 4 SPONGES AT PINK RECTS (edge-stand 47mm tall)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Cyan = # tower destination zone (KEEP EMPTY)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    out = src.with_name(src.stem + "_PLACEMENT.png")
    cv2.imwrite(str(out), img)
    print(f"OK: {out}")

    # Pairwise distance & EXCLUSION sanity check
    print("\nGeometry checks:")
    pts = [(s[1], s[2]) for s in SOURCES]
    for i in range(4):
        for j in range(i + 1, 4):
            d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
            ok = "OK" if d >= 0.15 else "FAIL"
            print(f"  {SOURCES[i][0]} <-> {SOURCES[j][0]}: {d*1000:.1f} mm  [{ok}]")
    print("\nEXCLUSION zone overlap check:")
    for tag, x, y, orient, _ in SOURCES:
        if orient == "X":
            sp_x = (x - 0.0625, x + 0.0625)
            sp_y = (y - 0.011, y + 0.011)
        else:
            sp_x = (x - 0.011, x + 0.011)
            sp_y = (y - 0.0625, y + 0.0625)
        overlap = (sp_x[0] < EX_X[1] and sp_x[1] > EX_X[0] and
                   sp_y[0] < EX_Y[1] and sp_y[1] > EX_Y[0])
        verdict = "FAIL (in zone)" if overlap else "OK (outside)"
        print(f"  {tag} sponge bbox X={sp_x} Y={sp_y}: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
