"""Phase ST-C v3 진단 #2 - Real Kinect layout capture for vision-blind γ test.

목적: 사용자가 sponge 재배치 → Enter 누름 → 1 frame 캡처 → episode_NNN/frame_0000.png.
diagnostic script (--renders-dir) 호환 format.

Usage:
    conda activate roarm
    python capture_real_layouts_for_gamma.py [--out data/real_layouts_<ts>] [--n 15]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None,
                   help="output dir; default data/real_layouts_<ts>")
    p.add_argument("--n", type=int, default=15, help="target number of layouts")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else Path("data") / f"real_layouts_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")
    print(f"Target N: {args.n}")

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1.0)
    for _ in range(5):
        k4a.get_capture()
    print("Kinect warmed up.\n")

    print("=" * 60)
    print("INSTRUCTIONS")
    print("  1. Place 4 sponges (edge-stand 47mm tall) per layout")
    print("  2. Vary positions: S1/S2/S3/S4 quadrants + asymmetric")
    print("  3. Press [ENTER] to capture, type 'q' + ENTER to quit")
    print("  4. Capture moves arm out of frame? Press ENTER again to retry")
    print("=" * 60)
    print()

    saved = []
    i = 0
    while i < args.n:
        prompt = f"[{i+1}/{args.n}] Rearrange sponges, then ENTER (or 'q' to quit, 'r' to retry last): "
        try:
            cmd = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            break

        if cmd == "q":
            print("Quit by user.")
            break
        if cmd == "r" and i > 0:
            i -= 1
            print(f"  retry: overwriting episode_{i:03d}")

        # flush stale frames + capture fresh
        for _ in range(3):
            k4a.get_capture()
        cap = k4a.get_capture()
        if cap.color is None:
            print("  FAIL: no color frame, retry...")
            continue
        bgr = np.ascontiguousarray(cap.color[:, :, :3])

        ep_dir = out_root / f"episode_{i:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        out_path = ep_dir / "frame_0000.png"
        cv2.imwrite(str(out_path), bgr)
        saved.append(out_path)
        print(f"  saved: {out_path} ({bgr.shape})")
        i += 1

    k4a.stop()

    # Build a quick collage for visual review.
    if saved:
        try:
            collage_path = out_root / "_collage.png"
            cells = []
            for p in saved:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h_target = 200
                w_target = int(img.shape[1] * h_target / img.shape[0])
                cells.append(cv2.resize(img, (w_target, h_target)))
            if cells:
                # 5 columns
                cols = 5
                rows = (len(cells) + cols - 1) // cols
                cell_w = max(c.shape[1] for c in cells)
                cell_h = cells[0].shape[0]
                canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
                for k, c in enumerate(cells):
                    r, col = divmod(k, cols)
                    canvas[r*cell_h:(r+1)*cell_h, col*cell_w:col*cell_w + c.shape[1]] = c
                    cv2.putText(canvas, f"{k:02d}", (col*cell_w+5, r*cell_h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(collage_path), canvas)
                print(f"\nCollage saved: {collage_path}")
        except Exception as e:
            print(f"Collage failed: {e}")

    print(f"\nTotal captured: {len(saved)}/{args.n}")
    print(f"Use with diagnostic: --renders-dir {out_root} --n-images {len(saved)}")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
