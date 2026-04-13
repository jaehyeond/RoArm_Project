#!/usr/bin/env python3
"""IMX335 wrist camera focus test — Laplacian variance at multiple distances.

Usage:
  1) Live preview (confirm camera works):
     python test_imx335_focus.py --preview

  2) Capture at each distance (press SPACE to capture, Q to quit):
     python test_imx335_focus.py --capture

  3) Analyze saved images:
     python test_imx335_focus.py --analyze
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path

DEVICE = "/dev/video4"
OUTPUT_DIR = Path("hw_imx335_test_output")
DISTANCES_CM = [5, 10, 15, 20, 30, 50]


def laplacian_variance(img_gray):
    """Laplacian variance — higher = sharper."""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return lap.var()


def open_camera(device=DEVICE, width=1280, height=720):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        # fallback: try index
        idx = int(device.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {device}")
        return None

    # Try MJPG for better FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read actual settings
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {w}x{h} @ {fps:.0f}fps")
    return cap


def mode_preview():
    """Live preview with real-time Laplacian display."""
    print("=== LIVE PREVIEW ===")
    print("Press Q to quit")
    cap = open_camera()
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = laplacian_variance(gray)

        # Draw info
        color = (0, 255, 0) if lap > 100 else (0, 255, 255) if lap > 50 else (0, 0, 255)
        cv2.putText(frame, f"Laplacian: {lap:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if lap > 100:
            verdict = "SHARP"
        elif lap > 50:
            verdict = "OK"
        else:
            verdict = "BLURRY"
        cv2.putText(frame, verdict, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Center crop 224x224 preview (what SmolVLA sees)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        crop_size = min(h, w, 224)
        crop = frame[cy - crop_size//2:cy + crop_size//2,
                      cx - crop_size//2:cx + crop_size//2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_lap = laplacian_variance(crop_gray)
        cv2.putText(frame, f"Center224 Lap: {crop_lap:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("IMX335 Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def mode_capture():
    """Interactive capture at each distance."""
    print("=== DISTANCE CAPTURE MODE ===")
    print("Place sponge at each distance, press SPACE to capture, Q to quit")
    OUTPUT_DIR.mkdir(exist_ok=True)

    cap = open_camera()
    if cap is None:
        return

    distance_idx = 0
    results = []

    while distance_idx < len(DISTANCES_CM):
        dist = DISTANCES_CM[distance_idx]
        print(f"\n>>> Place sponge at {dist}cm from camera lens. Press SPACE to capture.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap = laplacian_variance(gray)

            display = frame.copy()
            color = (0, 255, 0) if lap > 100 else (0, 255, 255) if lap > 50 else (0, 0, 255)
            cv2.putText(display, f"Distance: {dist}cm | Lap: {lap:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, "SPACE=capture  Q=quit  S=skip", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("IMX335 Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Save raw + analyze
                fname = OUTPUT_DIR / f"dist_{dist:03d}cm.png"
                cv2.imwrite(str(fname), frame)

                # Center crop analysis
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                s = 224
                crop = gray[max(0, cy-s//2):cy+s//2, max(0, cx-s//2):cx+s//2]
                crop_lap = laplacian_variance(crop)

                # Save crop
                crop_fname = OUTPUT_DIR / f"dist_{dist:03d}cm_center224.png"
                cv2.imwrite(str(crop_fname), crop)

                result = {
                    "distance_cm": dist,
                    "laplacian_full": lap,
                    "laplacian_center224": crop_lap,
                    "resolution": f"{w}x{h}",
                    "file": str(fname),
                }
                results.append(result)
                print(f"  ✅ {dist}cm: Full Lap={lap:.1f}, Center224 Lap={crop_lap:.1f}")
                distance_idx += 1
                break
            elif key == ord('q'):
                distance_idx = len(DISTANCES_CM)
                break
            elif key == ord('s'):
                print(f"  ⏭ Skipped {dist}cm")
                distance_idx += 1
                break

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("IMX335 Focus Test Results")
        print("=" * 60)
        print(f"{'Dist(cm)':>10} {'Full Lap':>12} {'Center224':>12} {'Verdict':>10}")
        print("-" * 50)
        for r in results:
            lap = r["laplacian_full"]
            clap = r["laplacian_center224"]
            if clap > 100:
                v = "✅ SHARP"
            elif clap > 50:
                v = "⚠️ OK"
            else:
                v = "❌ BLURRY"
            print(f"{r['distance_cm']:>10} {lap:>12.1f} {clap:>12.1f} {v:>10}")

        print("-" * 50)
        # Grasp range verdict
        grasp_results = [r for r in results if 5 <= r["distance_cm"] <= 15]
        if grasp_results:
            min_clap = min(r["laplacian_center224"] for r in grasp_results)
            if min_clap > 100:
                print("🎯 GRASP RANGE (5-15cm): SHARP — wrist camera usable!")
            elif min_clap > 50:
                print("⚠️ GRASP RANGE (5-15cm): Marginal — M12 lens swap recommended")
            else:
                print("❌ GRASP RANGE (5-15cm): BLURRY — fixed focus too far, M12 lens swap required")
        print(f"\nImages saved to: {OUTPUT_DIR}/")


def mode_analyze():
    """Analyze previously saved images."""
    if not OUTPUT_DIR.exists():
        print(f"No images found in {OUTPUT_DIR}/")
        return

    print("=" * 60)
    print("IMX335 Focus Analysis (saved images)")
    print("=" * 60)

    for f in sorted(OUTPUT_DIR.glob("dist_*cm.png")):
        if "center224" in f.name:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = laplacian_variance(gray)

        h, w = gray.shape
        cx, cy = w // 2, h // 2
        s = 224
        crop = gray[max(0, cy-s//2):cy+s//2, max(0, cx-s//2):cx+s//2]
        clap = laplacian_variance(crop)

        dist = f.stem.replace("dist_", "").replace("cm", "")
        verdict = "SHARP" if clap > 100 else "OK" if clap > 50 else "BLURRY"
        print(f"  {dist}cm: Full={lap:.1f}, Center224={clap:.1f} → {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMX335 wrist camera focus test")
    parser.add_argument("--preview", action="store_true", help="Live preview with Laplacian")
    parser.add_argument("--capture", action="store_true", help="Interactive capture at distances")
    parser.add_argument("--analyze", action="store_true", help="Analyze saved images")
    parser.add_argument("--device", default=DEVICE, help="Video device path")
    args = parser.parse_args()

    if args.device != DEVICE:
        DEVICE = args.device

    if args.preview:
        mode_preview()
    elif args.capture:
        mode_capture()
    elif args.analyze:
        mode_analyze()
    else:
        # Default: preview first, then capture
        print("Usage:")
        print("  1) python test_imx335_focus.py --preview   (live view)")
        print("  2) python test_imx335_focus.py --capture   (distance test)")
        print("  3) python test_imx335_focus.py --analyze   (re-analyze)")
        print()
        print("Quick start: run --preview first to confirm camera works")
