#!/usr/bin/env python
"""Kinect hand-eye calibration — data capture.

Moves the follower arm through a grid of poses, captures Azure Kinect
RGB + depth at each, detects the red marker sticker on the wrist_roll drum,
and saves joint angles + marker pixel + depth for each successful detection.

Wrist_roll is set to -base_angle to compensate for base rotation and keep
the side-mounted marker facing the Kinect.

Usage:
    conda run -n roarm python sim_scripts/kinect_handeye_capture.py

Output:
    sim_scripts/handeye_data/handeye_captures.json
    sim_scripts/handeye_data/pose_NN.jpg  (annotated RGB per pose)
"""

import json
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A

# ── SDK setup (suppress print spam + background decode errors) ──
# WARNING: do NOT replace `_process_received` with `lambda *a, **k: None` —
# it parses data['x'/'y'/'z'] + handle_m3_feedback(); a no-op makes
# joints_angle_get() return None → subscript errors downstream.
logging.getLogger("roarm_sdk").setLevel(logging.CRITICAL)
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback  # noqa: E402


def _silent_process(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data = [data['x'], data['y'], data['z']]
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res


DataProcessor._process_received = _silent_process
from roarm_sdk.roarm import roarm  # noqa: E402

# ── Constants ──
FOLLOWER_PORT = "/dev/ttyUSB1"
OUT_DIR = Path(__file__).resolve().parent / "handeye_data"

# Joint limits (CLAUDE.md hardware table)
JOINT_LIMITS = [
    (-190, 190),  # 0 base
    (-110, 110),  # 1 shoulder
    (-70, 190),   # 2 elbow
    (-110, 110),  # 3 wrist_pitch
    (-190, 190),  # 4 wrist_roll
    (-10, 100),   # 5 gripper
]

# Pose grid values (degrees)
BASES = [-60, -30, 0, 30, 60]
SHOULDERS = [0, 25, 50]
ELBOWS = [30, 70]
PITCHES = [-15, 15]

# Motion
MOVE_SPEED = 300
MOVE_ACC = 100
STABILIZE_S = 2.5
ANGLE_READ_RETRIES = 5

# HSV red marker detection (hue wraps around 0/180)
HSV_RANGES = [
    (np.array([0, 80, 60]), np.array([12, 255, 255])),
    (np.array([165, 80, 60]), np.array([180, 255, 255])),
]
MARKER_AREA_MIN = 40
MARKER_AREA_MAX = 5000
MARKER_DEPTH_MIN = 200   # mm
MARKER_DEPTH_MAX = 1200  # mm


# ── Marker detection ──
def _robust_depth(tdepth, u, v):
    """Depth at integer pixel (u,v); falls back to 5×5 median if center is 0."""
    h, w = tdepth.shape
    if 0 <= v < h and 0 <= u < w and tdepth[v, u] > 0:
        return int(tdepth[v, u])
    y0, y1 = max(0, v - 2), min(h, v + 3)
    x0, x1 = max(0, u - 2), min(w, u + 3)
    valid = tdepth[y0:y1, x0:x1]
    valid = valid[valid > 0]
    return int(np.median(valid)) if len(valid) > 0 else 0


def detect_marker(rgb_bgr, tdepth, fx):
    """Detect the red circular marker. Returns info dict or None."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for c in contours:
        a = cv2.contourArea(c)
        if MARKER_AREA_MIN <= a <= MARKER_AREA_MAX and a > best_area:
            best, best_area = c, a

    if best is None:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]

    depth_mm = _robust_depth(tdepth, int(round(u)), int(round(v)))
    if not (MARKER_DEPTH_MIN <= depth_mm <= MARKER_DEPTH_MAX):
        return None

    _, r = cv2.minEnclosingCircle(best)
    perim = cv2.arcLength(best, True)
    circ = 4 * np.pi * best_area / (perim * perim) if perim > 0 else 0

    return {
        "u": round(float(u), 2),
        "v": round(float(v), 2),
        "depth_mm": depth_mm,
        "area_px": int(best_area),
        "circularity": round(float(circ), 3),
        "diameter_mm": round(float(2 * r * depth_mm / fx), 1),
    }


# ── Arm helpers ──
def safe_angle_read(arm):
    """Read joint angles with retry. Returns list[6] or None."""
    for _ in range(ANGLE_READ_RETRIES):
        try:
            a = arm.joints_angle_get()
            if a and len(a) == 6:
                return [round(float(x), 2) for x in a]
        except Exception:
            pass
        time.sleep(0.3)
    return None


def generate_poses():
    """Build pose grid: base × shoulder × elbow × pitch, with wrist_roll = -base."""
    poses = [[0, 0, 0, 0, 0, 0]]  # HOME first (reference)
    for b in BASES:
        wr = max(-190, min(190, -b))
        for s in SHOULDERS:
            for e in ELBOWS:
                for p in PITCHES:
                    angles = [b, s, e, p, wr, 0]
                    if all(
                        JOINT_LIMITS[i][0] <= angles[i] <= JOINT_LIMITS[i][1]
                        for i in range(6)
                    ):
                        poses.append(angles)
    return poses


# ── Main ──
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    poses = generate_poses()
    print(f"Pose grid: {len(poses)} candidates (incl. HOME)")

    # Connect arm
    arm = roarm("roarm_m3", FOLLOWER_PORT, 115200)
    time.sleep(1)
    arm.torque_set(cmd=1)
    time.sleep(0.3)

    # Start Kinect
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1)
    for _ in range(15):
        k4a.get_capture()
        time.sleep(0.05)

    # Intrinsics
    ci = k4a.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    intrinsics = {
        "fx": round(float(ci[0, 0]), 2),
        "fy": round(float(ci[1, 1]), 2),
        "cx": round(float(ci[0, 2]), 2),
        "cy": round(float(ci[1, 2]), 2),
        "width": 1280,
        "height": 720,
    }
    print(f"Intrinsics: fx={intrinsics['fx']} fy={intrinsics['fy']}")

    # Ctrl+C → HOME + cleanup
    def on_interrupt(*_):
        print("\nInterrupted — returning to HOME...")
        try:
            arm.move_init()
            time.sleep(2)
        except Exception:
            pass
        k4a.stop()
        arm.disconnect()
        sys.exit(1)

    signal.signal(signal.SIGINT, on_interrupt)

    # Capture loop
    results = []
    skipped = 0

    for idx, target in enumerate(poses):
        tag = f"[{idx + 1}/{len(poses)}]"
        print(f"{tag} cmd={target[:5]}", end=" → ", flush=True)

        try:
            arm.joints_angle_ctrl(angles=target, speed=MOVE_SPEED, acc=MOVE_ACC)
        except Exception as e:
            print(f"REJECTED ({e})")
            skipped += 1
            continue

        time.sleep(STABILIZE_S)

        measured = safe_angle_read(arm)
        angle_source = "measured" if measured is not None else "commanded"

        # Flush stale frames, then capture
        for _ in range(5):
            k4a.get_capture()
            time.sleep(0.05)
        cap = k4a.get_capture()
        rgb = cap.color[:, :, :3]
        tdepth = cap.transformed_depth

        det = detect_marker(rgb, tdepth, intrinsics["fx"])
        if det is None:
            print("no marker")
            skipped += 1
            continue

        print(
            f"OK u={det['u']:.0f} v={det['v']:.0f} d={det['depth_mm']}mm "
            f"dia={det['diameter_mm']:.0f}mm circ={det['circularity']:.2f} "
            f"[{angle_source}]"
        )

        # Save annotated image
        img_name = f"pose_{len(results):02d}.jpg"
        ann = rgb.copy()
        cv2.circle(ann, (int(det["u"]), int(det["v"])), 5, (0, 255, 0), -1)
        cv2.imwrite(str(OUT_DIR / img_name), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "pose_id": len(results),
            "commanded": target,
            "measured": measured,
            "angle_source": angle_source,
            "marker": det,
            "image": img_name,
        })

    # Return HOME
    print("\nReturning to HOME...")
    arm.move_init()
    time.sleep(2)
    k4a.stop()
    arm.disconnect()

    # Save JSON
    data = {
        "intrinsics": intrinsics,
        "n_attempted": len(poses),
        "n_captured": len(results),
        "n_skipped": skipped,
        "poses": results,
    }
    out_json = OUT_DIR / "handeye_captures.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Captured: {len(results)}/{len(poses)} poses ({skipped} skipped)")
    print(f"Saved: {out_json}")

    if len(results) < 6:
        print("FAIL: Need >= 6 poses for calibration")
    elif len(results) < 15:
        print("WARN: Marginal — 15+ recommended for ±3mm accuracy")
    else:
        print(f"PASS: {len(results)} poses — sufficient")


if __name__ == "__main__":
    main()
