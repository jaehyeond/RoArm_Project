"""
1-frame Kinect 캡처 → sim_renders_v4와 side-by-side 비교용 PNG 저장.

Usage:
    conda activate roarm
    python capture_kinect_for_sponge_check.py [output_path]

Default output: claudedocs/sponge_check_YYYYMMDD_HHMMSS.png
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("claudedocs") / f"sponge_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    time.sleep(1.0)
    for _ in range(5):
        k4a.get_capture()
    cap = k4a.get_capture()
    if cap.color is None:
        print("FAIL: no color frame")
        k4a.stop()
        return 1
    bgr = np.ascontiguousarray(cap.color[:, :, :3])
    cv2.imwrite(str(out_path), bgr)
    print(f"OK: {out_path} ({bgr.shape})")

    sim_ref = Path("sim_renders_v5/stacking_initial_seed0_v3.png")
    if sim_ref.exists():
        ref = cv2.imread(str(sim_ref))
        if ref is not None:
            h = 480
            r1 = cv2.resize(bgr, (int(bgr.shape[1] * h / bgr.shape[0]), h))
            r2 = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h))
            cv2.putText(r1, "REAL Kinect", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(r2, "SIM seed=0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            side = np.hstack([r1, r2])
            cmp_path = out_path.with_name(out_path.stem + "_vs_sim.png")
            cv2.imwrite(str(cmp_path), side)
            print(f"OK: {cmp_path}")

    k4a.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
