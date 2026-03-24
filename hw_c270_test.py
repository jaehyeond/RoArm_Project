#!/usr/bin/env python3
"""C270 HD Webcam quality test for wrist camera feasibility.

Tests:
1. Basic capture: resolution, FPS, color format
2. Color fidelity: RGB histogram, saturation, white balance
3. Fixed-focus sharpness at ~40cm (eye-in-hand distance)
4. SmolVLA input resize (224x224) quality
5. Comparison with Azure Kinect reference (if available)
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path

OUTPUT_DIR = Path("hw_c270_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

C270_DEVICE = 4  # /dev/video4


def test_basic_capture(cap):
    """Test 1: Basic capture capabilities."""
    print("\n" + "="*60)
    print("TEST 1: Basic Capture")
    print("="*60)

    # Query actual resolution
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])

    print(f"  Resolution: {w}x{h}")
    print(f"  Reported FPS: {fps}")
    print(f"  FourCC: {fourcc_str}")

    # Actual FPS measurement (30 frames)
    frames = []
    t0 = time.time()
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    elapsed = time.time() - t0
    actual_fps = len(frames) / elapsed if elapsed > 0 else 0

    print(f"  Actual FPS (30-frame avg): {actual_fps:.1f}")
    print(f"  Frame shape: {frames[0].shape if frames else 'NO FRAMES'}")

    if not frames:
        print("  [FAIL] No frames captured!")
        return None

    # Save sample frame
    sample = frames[-1]
    cv2.imwrite(str(OUTPUT_DIR / "c270_raw_sample.png"), sample)
    print(f"  Saved: {OUTPUT_DIR / 'c270_raw_sample.png'}")

    return sample


def test_color_quality(frame):
    """Test 2: Color fidelity analysis."""
    print("\n" + "="*60)
    print("TEST 2: Color Quality")
    print("="*60)

    # BGR stats
    for i, ch_name in enumerate(["Blue", "Green", "Red"]):
        ch = frame[:, :, i]
        print(f"  {ch_name}: mean={ch.mean():.1f}, std={ch.std():.1f}, "
              f"min={ch.min()}, max={ch.max()}")

    # Check green bias (Xitech had severe green tint)
    b_mean = frame[:, :, 0].mean()
    g_mean = frame[:, :, 1].mean()
    r_mean = frame[:, :, 2].mean()
    green_ratio = g_mean / ((b_mean + r_mean) / 2 + 1e-6)
    print(f"\n  Green ratio (G / avg(B,R)): {green_ratio:.3f}")
    if green_ratio > 1.3:
        print("  [WARNING] Green bias detected (>1.3) — similar to Xitech issue")
    elif green_ratio > 1.1:
        print("  [CAUTION] Slight green bias (>1.1)")
    else:
        print("  [OK] No significant color bias")

    # HSV analysis for saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    print(f"\n  Saturation: mean={sat.mean():.1f}, std={sat.std():.1f}")
    print(f"  Value/Brightness: mean={val.mean():.1f}, std={val.std():.1f}")

    # Dynamic range
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dr = gray.max() - gray.min()
    print(f"  Dynamic range (gray): {dr} (max-min)")
    print(f"  Contrast (gray std): {gray.std():.1f}")

    # Save histogram
    fig_path = OUTPUT_DIR / "c270_color_histogram.png"
    # Create histogram image manually (no matplotlib dependency)
    hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 280, cv2.NORM_MINMAX)
        for x in range(1, 256):
            cv2.line(hist_img,
                     (2*(x-1), 300 - int(hist[x-1])),
                     (2*x, 300 - int(hist[x])),
                     color, 1)
    cv2.imwrite(str(fig_path), hist_img)
    print(f"  Saved histogram: {fig_path}")

    return {
        "green_ratio": green_ratio,
        "saturation_mean": sat.mean(),
        "dynamic_range": dr,
        "contrast": gray.std(),
    }


def test_sharpness(frame):
    """Test 3: Sharpness/focus quality at current distance."""
    print("\n" + "="*60)
    print("TEST 3: Sharpness (Fixed Focus)")
    print("="*60)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (standard sharpness metric)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    # Sobel gradient magnitude
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobx**2 + soby**2).mean()

    # Tenengrad (gradient-based focus measure)
    tenengrad = (sobx**2 + soby**2).mean()

    print(f"  Laplacian variance: {lap_var:.1f}")
    print(f"  Sobel mean magnitude: {sobel_mag:.1f}")
    print(f"  Tenengrad: {tenengrad:.1f}")

    # Sharpness interpretation
    # C270 fixed focus ~40cm. Typical Laplacian variance benchmarks:
    # < 50: very blurry, 50-200: soft, 200-500: acceptable, > 500: sharp
    if lap_var < 50:
        verdict = "[FAIL] Very blurry — focus distance mismatch"
    elif lap_var < 200:
        verdict = "[MARGINAL] Soft — may be usable but not ideal"
    elif lap_var < 500:
        verdict = "[OK] Acceptable sharpness"
    else:
        verdict = "[GOOD] Sharp image"
    print(f"  Verdict: {verdict}")

    # Save Laplacian visualization
    lap_vis = np.abs(lap)
    lap_vis = (lap_vis / lap_vis.max() * 255).astype(np.uint8) if lap_vis.max() > 0 else lap_vis.astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "c270_laplacian.png"), lap_vis)

    return {"laplacian_var": lap_var, "sobel_mag": sobel_mag, "tenengrad": tenengrad}


def test_smolvla_resize(frame):
    """Test 4: Quality after resize to SmolVLA input (224x224)."""
    print("\n" + "="*60)
    print("TEST 4: SmolVLA Resize (224x224)")
    print("="*60)

    h, w = frame.shape[:2]
    print(f"  Original: {w}x{h}")

    # Resize to 224x224 (SmolVLA input)
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    print(f"  Resized: {resized.shape[1]}x{resized.shape[0]}")

    # Sharpness after resize
    gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray_resized, cv2.CV_64F)
    lap_var = lap.var()
    print(f"  Laplacian variance (resized): {lap_var:.1f}")

    # Compare with original resized sharpness ratio
    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_orig = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
    ratio = lap_var / (lap_orig + 1e-6)
    print(f"  Sharpness retention ratio: {ratio:.3f}")

    # SNR estimate
    signal = gray_resized.mean()
    noise = gray_resized.std()
    snr = signal / (noise + 1e-6)
    print(f"  SNR estimate: {snr:.1f}")

    # Color preservation
    for i, ch_name in enumerate(["B", "G", "R"]):
        orig_mean = frame[:, :, i].mean()
        resized_mean = resized[:, :, i].mean()
        diff = abs(orig_mean - resized_mean)
        print(f"  {ch_name} mean shift: {diff:.2f} ({orig_mean:.1f} → {resized_mean:.1f})")

    cv2.imwrite(str(OUTPUT_DIR / "c270_224x224.png"), resized)
    print(f"  Saved: {OUTPUT_DIR / 'c270_224x224.png'}")

    return {"lap_var_224": lap_var, "sharpness_retention": ratio, "snr": snr}


def test_multi_frame_consistency(cap, n_frames=10):
    """Test 5: Frame-to-frame consistency (temporal noise)."""
    print("\n" + "="*60)
    print("TEST 5: Temporal Consistency")
    print("="*60)

    frames = []
    for _ in range(n_frames):
        ret, f = cap.read()
        if ret:
            frames.append(f.astype(np.float32))

    if len(frames) < 2:
        print("  [FAIL] Not enough frames")
        return {}

    # Compute frame-to-frame difference
    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i] - frames[i-1]).mean()
        diffs.append(diff)

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    print(f"  Frames captured: {len(frames)}")
    print(f"  Mean frame-to-frame diff: {mean_diff:.2f}")
    print(f"  Std frame-to-frame diff: {std_diff:.2f}")

    # Temporal noise (std across time per pixel)
    stack = np.stack(frames, axis=0)
    temporal_std = stack.std(axis=0).mean()
    print(f"  Temporal noise (pixel std over time): {temporal_std:.2f}")

    if temporal_std > 10:
        print("  [WARNING] High temporal noise — may cause jittery VLA predictions")
    elif temporal_std > 5:
        print("  [CAUTION] Moderate temporal noise")
    else:
        print("  [OK] Low temporal noise")

    return {"temporal_noise": temporal_std, "frame_diff_mean": mean_diff}


def main():
    print("C270 HD Webcam Quality Test")
    print(f"Device: /dev/video{C270_DEVICE}")
    print(f"Output: {OUTPUT_DIR.absolute()}")

    # Open camera
    cap = cv2.VideoCapture(C270_DEVICE)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open /dev/video{C270_DEVICE}")
        return

    # Try to set 720p (C270 max)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm-up (auto-exposure settle)
    print("Warming up (2 sec)...")
    t0 = time.time()
    while time.time() - t0 < 2:
        cap.read()

    # Run tests
    sample = test_basic_capture(cap)
    if sample is None:
        cap.release()
        return

    color_results = test_color_quality(sample)
    sharpness_results = test_sharpness(sample)
    resize_results = test_smolvla_resize(sample)
    temporal_results = test_multi_frame_consistency(cap)

    cap.release()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    issues = []
    if color_results["green_ratio"] > 1.3:
        issues.append("Severe green bias")
    if color_results["green_ratio"] > 1.1:
        issues.append("Slight green bias")
    if sharpness_results["laplacian_var"] < 50:
        issues.append("Very blurry (focus mismatch)")
    elif sharpness_results["laplacian_var"] < 200:
        issues.append("Soft focus")
    if temporal_results.get("temporal_noise", 0) > 10:
        issues.append("High temporal noise")
    if resize_results["sharpness_retention"] < 0.3:
        issues.append("Poor sharpness after 224x224 resize")

    if not issues:
        print("  [PASS] No critical issues detected")
        print("  C270 appears viable for wrist camera use")
    else:
        print(f"  [ISSUES FOUND] {len(issues)}:")
        for issue in issues:
            print(f"    - {issue}")

    print(f"\n  Key metrics:")
    print(f"    Green ratio:       {color_results['green_ratio']:.3f} (<1.1 good)")
    print(f"    Laplacian var:     {sharpness_results['laplacian_var']:.1f} (>200 good)")
    print(f"    Temporal noise:    {temporal_results.get('temporal_noise', 'N/A')}")
    print(f"    224x224 sharpness: {resize_results['lap_var_224']:.1f}")
    print(f"    Sharpness retain:  {resize_results['sharpness_retention']:.3f}")

    print(f"\n  Output images saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
