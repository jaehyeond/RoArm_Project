#!/usr/bin/env python3
"""
data_new_episodes_analysis.py
Comprehensive analysis of 31 newly collected sponge pick-up episodes.
Data Agent - RoArm M3 SmolVLA Pipeline
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

COLLECTED_DATA_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_pitch", "Wrist_roll", "Gripper"]

# Thresholds from MEMORY.md (ESP32 FK Z calibration)
Z_DEEP_THRESH = 80      # mm: gripper at/below object surface
Z_APPROACH_THRESH = 160  # mm: arm descending
SHOULDER_DEEP_THRESH = 50  # degrees
GRIPPER_OPEN_THRESH = 20   # degrees
GRIPPER_CLOSED_THRESH = 5  # degrees
GRIPPER_MAX_EXPECTED = 55  # degrees (hardware max seen)

def load_all_episodes():
    """Load metadata for all 31 episodes."""
    episodes = []
    ep_dirs = sorted(COLLECTED_DATA_DIR.iterdir())
    for ep_dir in ep_dirs:
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = str(ep_dir)
        meta["_name"] = ep_dir.name
        episodes.append(meta)
    return episodes

def extract_frame_arrays(episode):
    """Extract per-frame numpy arrays from episode frames."""
    frames = episode["frames"]
    n = len(frames)
    angles = np.array([f["angles"] for f in frames])  # (N, 6)
    # Some frames may not have pose
    has_pose = "pose" in frames[0] and frames[0]["pose"] is not None
    if has_pose:
        poses = np.array([f.get("pose", [np.nan, np.nan, np.nan]) for f in frames])  # (N, 3)
    else:
        poses = None
    return angles, poses, n

def classify_z(min_z):
    """Classify episode grasp depth by min_z (ESP32 FK convention)."""
    if min_z is None or np.isnan(min_z):
        return "UNKNOWN"
    if min_z < Z_DEEP_THRESH:
        return "DEEP"
    elif min_z < Z_APPROACH_THRESH:
        return "APPROACH"
    else:
        return "SHALLOW"

def detect_position_group(base_angles):
    """Classify base angle into spatial position group."""
    mean_base = np.mean(base_angles)
    if abs(mean_base) < 10:
        return "CENTER"
    elif mean_base > 20:
        return "RIGHT"
    elif mean_base < -20:
        return "LEFT"
    elif mean_base > 0:
        return "SLIGHT_RIGHT"
    else:
        return "SLIGHT_LEFT"

def analyze_gripper_timing(angles, n):
    """
    Detect when gripper opens and closes.
    Returns: (open_frame, close_frame, open_pct, timing_label)
    """
    gripper = angles[:, 5]
    # Find first frame where gripper > OPEN_THRESH
    open_frames = np.where(gripper > GRIPPER_OPEN_THRESH)[0]
    if len(open_frames) == 0:
        return None, None, None, "NEVER_OPENS"
    open_frame = open_frames[0]
    open_pct = open_frame / n * 100

    # Find last frame where gripper > OPEN_THRESH (close after peak)
    close_frames = np.where(gripper[open_frame:] < GRIPPER_CLOSED_THRESH)[0]
    if len(close_frames) == 0:
        close_frame = None
    else:
        close_frame = open_frame + close_frames[0]

    # Timing label
    if open_pct < 25:
        timing = "VERY_EARLY (<25%)"
    elif open_pct < 40:
        timing = "EARLY (25-40%)"
    elif open_pct < 60:
        timing = "MIDDLE (40-60%)"
    elif open_pct < 75:
        timing = "LATE (60-75%)"
    else:
        timing = "VERY_LATE (>75%)"

    return open_frame, close_frame, open_pct, timing

def detect_static_frames(angles, threshold=0.5):
    """Count frames where no joint moves more than threshold degrees."""
    if len(angles) < 2:
        return 0
    deltas = np.abs(np.diff(angles, axis=0))
    static_transitions = np.all(deltas < threshold, axis=1)
    return int(np.sum(static_transitions))

def main():
    print("=" * 70)
    print("DATA ANALYSIS: 31 New Sponge Pick-up Episodes")
    print("RoArm M3 SmolVLA - Data Agent Report")
    print("=" * 70)

    episodes = load_all_episodes()
    print(f"\nLoaded {len(episodes)} episodes from {COLLECTED_DATA_DIR}\n")

    # -------------------------------------------------------------------------
    # 1. PER-EPISODE SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("1. PER-EPISODE SUMMARY")
    print("=" * 70)
    header = f"{'EP':>6} {'FRAMES':>6} {'BASE_MEAN':>9} {'SHOULDER_MAX':>12} {'MIN_Z':>7} {'DEPTH_CLASS':>11} {'GRIP_MAX':>8} {'OPEN@%':>7} {'TIMING'}"
    print(header)
    print("-" * 100)

    ep_data = []  # collect for aggregate analysis

    for ep in episodes:
        angles, poses, n = extract_frame_arrays(ep)
        name = ep["_name"]
        ep_id = int(name.split("_")[-1])

        base_angles = angles[:, 0]
        shoulder_angles = angles[:, 1]
        gripper_angles = angles[:, 5]

        base_mean = np.mean(base_angles)
        shoulder_max = np.max(shoulder_angles)
        gripper_max_val = np.max(gripper_angles)

        # Z from metadata (min_z is the deepest point reached)
        min_z = ep.get("min_z", None)
        depth_class = classify_z(min_z)

        open_frame, close_frame, open_pct, timing = analyze_gripper_timing(angles, n)

        min_z_str = f"{min_z:.1f}" if min_z is not None else "N/A"
        open_pct_str = f"{open_pct:.0f}%" if open_pct is not None else "N/A"

        print(f"{name:>12} {n:>6} {base_mean:>+9.1f}° {shoulder_max:>11.1f}° {min_z_str:>7} {depth_class:>11} {gripper_max_val:>7.1f}° {open_pct_str:>7}  {timing}")

        static_n = detect_static_frames(angles)
        ep_data.append({
            "name": name,
            "ep_id": ep_id,
            "n_frames": n,
            "angles": angles,
            "poses": poses,
            "base_mean": base_mean,
            "base_std": np.std(base_angles),
            "shoulder_max": shoulder_max,
            "elbow_min": float(np.min(angles[:, 2])),
            "elbow_max": float(np.max(angles[:, 2])),
            "gripper_max": gripper_max_val,
            "min_z": min_z,
            "max_z": ep.get("max_z", None),
            "depth_class": depth_class,
            "open_frame": open_frame,
            "close_frame": close_frame,
            "open_pct": open_pct,
            "timing": timing,
            "static_frames": static_n,
            "static_pct": static_n / max(n-1, 1) * 100,
        })

    # -------------------------------------------------------------------------
    # 2. JOINT ANGLE DISTRIBUTIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. JOINT ANGLE DISTRIBUTIONS (across ALL frames, all episodes)")
    print("=" * 70)

    all_angles = np.vstack([d["angles"] for d in ep_data])
    print(f"Total frames analyzed: {len(all_angles)}")
    print()
    print(f"{'Joint':<14} {'Min':>8} {'P10':>8} {'P25':>8} {'Median':>8} {'P75':>8} {'P90':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 90)
    for i, name in enumerate(JOINT_NAMES):
        col = all_angles[:, i]
        print(f"{name:<14} {col.min():>8.2f} {np.percentile(col,10):>8.2f} {np.percentile(col,25):>8.2f} "
              f"{np.median(col):>8.2f} {np.percentile(col,75):>8.2f} {np.percentile(col,90):>8.2f} "
              f"{col.max():>8.2f} {col.mean():>8.2f} {col.std():>8.2f}")

    # -------------------------------------------------------------------------
    # 3. GRIPPER PATTERNS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. GRIPPER PATTERNS")
    print("=" * 70)

    gripper_all = all_angles[:, 5]
    frames_open = np.sum(gripper_all > GRIPPER_OPEN_THRESH)
    frames_closed = np.sum(gripper_all < GRIPPER_CLOSED_THRESH)
    frames_wide = np.sum(gripper_all > 50)
    print(f"Frames with gripper > 20° (open):    {frames_open:>6} ({frames_open/len(gripper_all)*100:.1f}%)")
    print(f"Frames with gripper < 5°  (closed):  {frames_closed:>6} ({frames_closed/len(gripper_all)*100:.1f}%)")
    print(f"Frames with gripper > 50° (wide):    {frames_wide:>6} ({frames_wide/len(gripper_all)*100:.1f}%)")
    print()

    # Open angle stats
    open_angles = gripper_all[gripper_all > GRIPPER_OPEN_THRESH]
    if len(open_angles) > 0:
        print(f"Open-phase gripper angle:  mean={np.mean(open_angles):.1f}°  std={np.std(open_angles):.1f}°  "
              f"max={np.max(open_angles):.1f}°  min={np.min(open_angles):.1f}°")

    # Timing distribution
    print("\nGripper opening timing distribution:")
    timing_counts = defaultdict(int)
    for d in ep_data:
        timing_counts[d["timing"]] += 1
    for timing, cnt in sorted(timing_counts.items()):
        pct = cnt / len(ep_data) * 100
        bar = "#" * cnt
        print(f"  {timing:<25} {cnt:>3} ep ({pct:>5.1f}%) {bar}")

    # Episodes that never open gripper
    never_open = [d["name"] for d in ep_data if d["timing"] == "NEVER_OPENS"]
    if never_open:
        print(f"\nWARNING: Episodes with NEVER_OPENS gripper: {never_open}")

    # Open percentage stats
    open_pcts = [d["open_pct"] for d in ep_data if d["open_pct"] is not None]
    if open_pcts:
        print(f"\nGripper open timing: mean={np.mean(open_pcts):.1f}%  "
              f"std={np.std(open_pcts):.1f}%  min={np.min(open_pcts):.1f}%  max={np.max(open_pcts):.1f}%")

    # -------------------------------------------------------------------------
    # 4. POSITION DIVERSITY (BASE ANGLE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. POSITION DIVERSITY (Base Angle Analysis)")
    print("=" * 70)

    base_means = [d["base_mean"] for d in ep_data]
    base_means_arr = np.array(base_means)

    print(f"Base angle across episodes:  mean={np.mean(base_means_arr):.1f}°  "
          f"std={np.std(base_means_arr):.1f}°  min={np.min(base_means_arr):.1f}°  max={np.max(base_means_arr):.1f}°")

    # Bucket analysis
    print("\nBase angle distribution (per-episode mean):")
    buckets = [
        ("<-30°  LEFT_FAR",        base_means_arr < -30),
        ("-30 to -15°  LEFT",      (base_means_arr >= -30) & (base_means_arr < -15)),
        ("-15 to -5°  SLIGHT_LEFT",(base_means_arr >= -15) & (base_means_arr < -5)),
        ("-5 to +5°   CENTER",     (base_means_arr >= -5) & (base_means_arr <= 5)),
        ("+5 to +15°  SLIGHT_RIGHT",(base_means_arr > 5) & (base_means_arr <= 15)),
        ("+15 to +30° RIGHT",      (base_means_arr > 15) & (base_means_arr <= 30)),
        (">+30°  RIGHT_FAR",       base_means_arr > 30),
    ]
    for label, mask in buckets:
        cnt = int(np.sum(mask))
        bar = "#" * cnt
        print(f"  {label:<30} {cnt:>3} ep  {bar}")

    # -------------------------------------------------------------------------
    # 5. EPISODE DURATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. EPISODE DURATION (Frame Count)")
    print("=" * 70)

    frame_counts = [d["n_frames"] for d in ep_data]
    fc_arr = np.array(frame_counts)
    fps = 30

    print(f"Frame count: mean={np.mean(fc_arr):.1f}  std={np.std(fc_arr):.1f}  "
          f"min={np.min(fc_arr)}  max={np.max(fc_arr)}")
    print(f"Duration:    mean={np.mean(fc_arr)/fps:.1f}s  std={np.std(fc_arr)/fps:.1f}s  "
          f"min={np.min(fc_arr)/fps:.1f}s  max={np.max(fc_arr)/fps:.1f}s")
    print()

    too_short = [(d["name"], d["n_frames"]) for d in ep_data if d["n_frames"] < 100]
    too_long = [(d["name"], d["n_frames"]) for d in ep_data if d["n_frames"] > 500]
    print(f"Episodes < 100 frames (too short, <3.3s): {len(too_short)}")
    for name, n in too_short:
        print(f"  {name}: {n} frames ({n/fps:.1f}s)")
    print(f"Episodes > 500 frames (very long, >16.7s): {len(too_long)}")
    for name, n in too_long:
        print(f"  {name}: {n} frames ({n/fps:.1f}s)")

    # Frame count histogram
    print("\nFrame count distribution:")
    bins = [0, 100, 150, 200, 250, 300, 400, 600]
    bin_labels = ["<100", "100-150", "150-200", "200-250", "250-300", "300-400", ">400"]
    for i, label in enumerate(bin_labels):
        lo, hi = bins[i], bins[i+1]
        cnt = int(np.sum((fc_arr >= lo) & (fc_arr < hi)))
        bar = "#" * cnt
        print(f"  {label:<12} {cnt:>3} ep  {bar}")

    # -------------------------------------------------------------------------
    # 6. SHOULDER DEPTH ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. SHOULDER DEPTH ANALYSIS (Shoulder is the correct depth proxy)")
    print("=" * 70)

    shoulder_maxes = [d["shoulder_max"] for d in ep_data]
    sh_arr = np.array(shoulder_maxes)

    print(f"Max shoulder per episode: mean={np.mean(sh_arr):.1f}°  std={np.std(sh_arr):.1f}°  "
          f"min={np.min(sh_arr):.1f}°  max={np.max(sh_arr):.1f}°")

    deep_sh = sum(1 for v in shoulder_maxes if v >= SHOULDER_DEEP_THRESH)
    approach_sh = sum(1 for v in shoulder_maxes if 30 <= v < SHOULDER_DEEP_THRESH)
    shallow_sh = sum(1 for v in shoulder_maxes if v < 30)
    print(f"\nBy max shoulder angle:")
    print(f"  DEEP     (>50°):  {deep_sh:>3} ep ({deep_sh/len(ep_data)*100:.0f}%)")
    print(f"  APPROACH (30-50°):{approach_sh:>3} ep ({approach_sh/len(ep_data)*100:.0f}%)")
    print(f"  SHALLOW  (<30°):  {shallow_sh:>3} ep ({shallow_sh/len(ep_data)*100:.0f}%)")

    print("\nShoulder max distribution:")
    sh_bins = [0, 30, 40, 50, 60, 70, 80, 120]
    sh_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70-80", ">80"]
    for i, label in enumerate(sh_labels):
        lo, hi = sh_bins[i], sh_bins[i+1]
        cnt = int(np.sum((sh_arr >= lo) & (sh_arr < hi)))
        bar = "#" * cnt
        print(f"  sh_max {label:<8}° {cnt:>3} ep  {bar}")

    # -------------------------------------------------------------------------
    # 7. Z-HEIGHT PATTERNS (grasp depth via ESP32 FK)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. Z-HEIGHT PATTERNS (ESP32 FK: lower Z = deeper grasp)")
    print("=" * 70)
    print("  Z thresholds: DEEP < 80mm, APPROACH 80-160mm, SHALLOW > 160mm")
    print("  (Z=30mm: arm at table surface, Z=80mm: object grasp, Z=230mm: home)")
    print()

    min_zs = [d["min_z"] for d in ep_data if d["min_z"] is not None]
    z_arr = np.array(min_zs)

    print(f"Min Z per episode: mean={np.mean(z_arr):.1f}mm  std={np.std(z_arr):.1f}mm  "
          f"min={np.min(z_arr):.1f}mm  max={np.max(z_arr):.1f}mm")

    deep_z = sum(1 for d in ep_data if d["depth_class"] == "DEEP")
    approach_z = sum(1 for d in ep_data if d["depth_class"] == "APPROACH")
    shallow_z = sum(1 for d in ep_data if d["depth_class"] == "SHALLOW")
    unknown_z = sum(1 for d in ep_data if d["depth_class"] == "UNKNOWN")
    print(f"\nDepth classification (min_z):")
    print(f"  DEEP     (Z < 80mm):   {deep_z:>3} ep ({deep_z/len(ep_data)*100:.0f}%)  {'#'*deep_z}")
    print(f"  APPROACH (80-160mm):   {approach_z:>3} ep ({approach_z/len(ep_data)*100:.0f}%)  {'#'*approach_z}")
    print(f"  SHALLOW  (Z > 160mm):  {shallow_z:>3} ep ({shallow_z/len(ep_data)*100:.0f}%)  {'#'*shallow_z}")
    if unknown_z:
        print(f"  UNKNOWN (no Z data):   {unknown_z:>3} ep")

    print("\nMin-Z distribution (mm):")
    z_bins = [(-200, 0), (0, 40), (40, 80), (80, 120), (120, 160), (160, 200), (200, 300)]
    z_labels = ["<0 (VERY DEEP)", "0-40 (TABLE)", "40-80 (DEEP)", "80-120 (APPROACH-DEEP)", "120-160 (APPROACH)", "160-200 (SHALLOW)", ">200 (HOME)"]
    for (lo, hi), label in zip(z_bins, z_labels):
        cnt = int(np.sum((z_arr >= lo) & (z_arr < hi)))
        bar = "#" * cnt
        print(f"  {label:<25} {cnt:>3} ep  {bar}")

    # -------------------------------------------------------------------------
    # 8. TEMPORAL PATTERNS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. TEMPORAL PATTERNS")
    print("=" * 70)

    # Static frame analysis
    static_pcts = [d["static_pct"] for d in ep_data]
    print(f"Static frame fraction (no joint moves >0.5°):")
    print(f"  Mean: {np.mean(static_pcts):.1f}%  Std: {np.std(static_pcts):.1f}%  "
          f"Min: {np.min(static_pcts):.1f}%  Max: {np.max(static_pcts):.1f}%")

    high_static = [(d["name"], d["static_pct"]) for d in ep_data if d["static_pct"] > 40]
    if high_static:
        print(f"\nHigh static episodes (>40% static frames):")
        for name, pct in sorted(high_static, key=lambda x: -x[1]):
            print(f"  {name}: {pct:.1f}%")

    # Gripper open timing relative to episode
    if open_pcts:
        print(f"\nGripper open timing (% into episode when first opens):")
        print(f"  Mean: {np.mean(open_pcts):.1f}%  Std: {np.std(open_pcts):.1f}%  "
              f"Min: {np.min(open_pcts):.1f}%  Max: {np.max(open_pcts):.1f}%")

    # Episode phase analysis: gripper open/close synchronization with shoulder
    print("\nGripper-Shoulder synchronization per episode:")
    sync_stats = []
    for d in ep_data:
        angles = d["angles"]
        n = d["n_frames"]
        shoulder = angles[:, 1]
        gripper = angles[:, 5]

        max_sh_frame = int(np.argmax(shoulder))
        max_sh_pct = max_sh_frame / n * 100

        open_pct_ep = d["open_pct"] if d["open_pct"] is not None else -1

        if open_pct_ep < 0:
            sync = "NO_OPEN"
        else:
            delta = open_pct_ep - max_sh_pct
            if abs(delta) < 10:
                sync = "SYNCED"
            elif delta < 0:
                sync = "EARLY_OPEN"
            else:
                sync = "LATE_OPEN"
        sync_stats.append(sync)

    sync_counts = defaultdict(int)
    for s in sync_stats:
        sync_counts[s] += 1
    for s, c in sorted(sync_counts.items()):
        print(f"  {s:<15}: {c} ep ({c/len(ep_data)*100:.0f}%)")

    # -------------------------------------------------------------------------
    # 9. QUALITY FLAGS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. QUALITY FLAGS (Per Episode)")
    print("=" * 70)

    quality_issues = []
    for d in ep_data:
        issues = []
        # Too short
        if d["n_frames"] < 100:
            issues.append("SHORT(<100fr)")
        # Never opens gripper
        if d["timing"] == "NEVER_OPENS":
            issues.append("NO_GRIP_OPEN")
        # Opens too early (< 20% into episode)
        if d["open_pct"] is not None and d["open_pct"] < 20:
            issues.append("GRIP_VERY_EARLY")
        # Opens too late (> 80% into episode)
        if d["open_pct"] is not None and d["open_pct"] > 80:
            issues.append("GRIP_VERY_LATE")
        # Very shallow (never descended)
        if d["depth_class"] == "SHALLOW":
            issues.append("SHALLOW_GRASP")
        # Gripper barely opens
        if d["gripper_max"] < 15:
            issues.append("GRIP_TOO_SMALL")
        # Too many static frames
        if d["static_pct"] > 50:
            issues.append("HIGH_STATIC")
        # Very long episode
        if d["n_frames"] > 450:
            issues.append("VERY_LONG(>15s)")

        if issues:
            quality_issues.append((d["name"], issues))
            print(f"  {d['name']}: {', '.join(issues)}")
        else:
            print(f"  {d['name']}: OK")

    print(f"\nEpisodes with quality issues: {len(quality_issues)}/{len(ep_data)}")

    # -------------------------------------------------------------------------
    # 10. AGGREGATE SUMMARY & RECOMMENDATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("10. AGGREGATE SUMMARY")
    print("=" * 70)

    total_frames = sum(d["n_frames"] for d in ep_data)
    print(f"Total episodes:      {len(ep_data)}")
    print(f"Total frames:        {total_frames}")
    print(f"Total duration:      {total_frames/fps:.0f}s ({total_frames/fps/60:.1f} min)")
    print(f"Mean ep duration:    {np.mean(frame_counts)/fps:.1f}s ({np.mean(frame_counts):.0f} frames)")
    print()
    print(f"Depth coverage:")
    print(f"  DEEP (Z<80mm):     {deep_z}/{len(ep_data)} ({deep_z/len(ep_data)*100:.0f}%)")
    print(f"  APPROACH (80-160): {approach_z}/{len(ep_data)} ({approach_z/len(ep_data)*100:.0f}%)")
    print(f"  SHALLOW (>160):    {shallow_z}/{len(ep_data)} ({shallow_z/len(ep_data)*100:.0f}%)")
    print()
    print(f"Shoulder DEEP (>50°):{deep_sh}/{len(ep_data)} ({deep_sh/len(ep_data)*100:.0f}%)")
    print()
    print(f"Gripper quality:")
    print(f"  Episodes with good open (>20°): {sum(1 for d in ep_data if d['gripper_max'] > 20)}/{len(ep_data)}")
    print(f"  Max gripper mean across ep:     {np.mean([d['gripper_max'] for d in ep_data]):.1f}°")
    print(f"  Frames open (>20°):             {frames_open/len(all_angles)*100:.1f}%")
    print(f"  Frames closed (<5°):            {frames_closed/len(all_angles)*100:.1f}%")
    print()
    print(f"Position diversity:")
    print(f"  Base angle std (ep means):      {np.std(base_means_arr):.1f}°")
    print(f"  Base range:                     {np.min(base_means_arr):.1f}° to {np.max(base_means_arr):.1f}°")
    print()
    print(f"Quality issues:      {len(quality_issues)}/{len(ep_data)} episodes flagged")
    print(f"Static frame mean:   {np.mean(static_pcts):.1f}%")

    # -------------------------------------------------------------------------
    # 11. RECOMMENDATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("11. RECOMMENDATIONS")
    print("=" * 70)

    # Episodes to exclude
    bad_episodes = [name for name, issues in quality_issues
                    if any(i in ["NO_GRIP_OPEN", "SHORT(<100fr)", "GRIP_TOO_SMALL"] for i in issues)]
    print(f"\nEpisodes to EXCLUDE from training ({len(bad_episodes)}):")
    if bad_episodes:
        for name in bad_episodes:
            issues = next(iss for n, iss in quality_issues if n == name)
            print(f"  {name}: {', '.join(issues)}")
    else:
        print("  None - all episodes appear usable!")

    # Position gaps
    print(f"\nPosition coverage gaps:")
    right_far = sum(1 for m in base_means if m > 30)
    left_far = sum(1 for m in base_means if m < -30)
    center = sum(1 for m in base_means if abs(m) < 5)
    print(f"  CENTER (|base|<5°):  {center} ep {'OK' if center >= 5 else 'NEEDS_MORE'}")
    print(f"  LEFT_FAR (base<-30°): {left_far} ep {'OK' if left_far >= 5 else 'NEEDS_MORE'}")
    print(f"  RIGHT_FAR (base>30°): {right_far} ep {'OK' if right_far >= 5 else 'NEEDS_MORE'}")

    # How many more needed
    usable = len(ep_data) - len(bad_episodes)
    target = 100
    needed = max(0, target - usable)
    print(f"\nEpisode count analysis:")
    print(f"  Current total:    {len(ep_data)}")
    print(f"  Usable (no flags):{usable}")
    print(f"  Target (SmolVLA): {target}")
    print(f"  Still needed:     {needed}")

    # Depth gap
    target_deep = 40  # aim for 40% DEEP
    target_deep_ep = int(target * 0.40)
    needed_deep = max(0, target_deep_ep - deep_z)
    print(f"\nDepth balance analysis (target: 40% DEEP):")
    print(f"  Current DEEP:     {deep_z}/{len(ep_data)} ({deep_z/len(ep_data)*100:.0f}%)")
    print(f"  Target DEEP:      {target_deep_ep}/{target} (40%)")
    print(f"  Additional DEEP needed: ~{needed_deep}")

    print("\n" + "=" * 70)
    print("ACTION PLAN")
    print("=" * 70)

    action_plan = [
        f"1. Collect {needed} more episodes (minimum) to reach {target} total usable",
        f"2. Of those, prioritize {needed_deep}+ DEEP grasps (arm fully down, Z < 80mm)",
        f"3. Ensure right-far and left-far positions if under-represented",
        f"4. Gripper timing: open at 40-60% of episode (not too early, not too late)",
        f"5. Episode length: aim for 150-300 frames (5-10 seconds at 30fps)",
        f"6. After collecting, re-run this analysis to verify balance before training",
    ]
    for line in action_plan:
        print(f"  {line}")

    print("\n" + "=" * 70)
    print("TRAINING READINESS ASSESSMENT")
    print("=" * 70)
    score = 0
    max_score = 6
    checks = [
        (len(ep_data) >= 50, f"Episode count >= 50: {len(ep_data)}"),
        (deep_z / len(ep_data) >= 0.30, f"DEEP coverage >= 30%: {deep_z/len(ep_data)*100:.0f}%"),
        (np.std(base_means_arr) >= 10, f"Position diversity (base_std >= 10°): {np.std(base_means_arr):.1f}°"),
        (frames_open / len(all_angles) >= 0.15, f"Gripper open frames >= 15%: {frames_open/len(all_angles)*100:.1f}%"),
        (len(bad_episodes) / len(ep_data) < 0.15, f"Bad episode ratio < 15%: {len(bad_episodes)/len(ep_data)*100:.0f}%"),
        (np.mean(frame_counts) >= 150, f"Mean episode duration >= 150 frames: {np.mean(frame_counts):.0f}"),
    ]
    for passed, desc in checks:
        status = "PASS" if passed else "FAIL"
        mark = "+" if passed else "-"
        if passed:
            score += 1
        print(f"  [{mark}] {status}: {desc}")

    print(f"\nOverall readiness: {score}/{max_score}")
    if score >= 5:
        print("  READY FOR TRAINING (collect more for better coverage)")
    elif score >= 3:
        print("  PARTIALLY READY (collect more before training)")
    else:
        print("  NOT READY (significant data gaps, collect more first)")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
