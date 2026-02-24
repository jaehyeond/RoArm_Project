#!/usr/bin/env python3
"""
Full comprehensive dataset analysis - v4 (51 episodes, 2026-02-24).

Covers all 8+ analysis dimensions:
  1. Basic Stats
  2. Position Diversity (base angle zones)
  3. Grasp Depth (Z-height classification)
  4. Gripper Phase Analysis (open/close timing and pattern)
  5. Joint Diversity (all 6 joints, frame-level)
  6. Episode Quality Flags (SHORT, STATIC, NO_GRIP, INCOMPLETE)
  7. Temporal Quality (static frames)
  8. Comparison with v1 dataset (hardcoded reference stats)
  9. Shoulder Distribution
 10. Per-episode summary table
 11. Training Readiness Assessment + Recommendations
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_pitch", "Wrist_roll", "Gripper"]

# Joint hardware limits (from CLAUDE.md spec)
JOINT_LIMITS = {
    "Base":        (-190, 190),
    "Shoulder":    (-110, 110),
    "Elbow":       (-70, 190),
    "Wrist_pitch": (-110, 110),
    "Wrist_roll":  (-190, 190),
    "Gripper":     (-10, 100),
}

# Position zones by base angle
POSITION_ZONES = {
    "LEFT_FAR":  (-90, -30),
    "LEFT":      (-30, -10),
    "CENTER":    (-10, 10),
    "RIGHT":     (10, 30),
    "RIGHT_FAR": (30, 90),
}
ZONE_ORDER = ["LEFT_FAR", "LEFT", "CENTER", "RIGHT", "RIGHT_FAR"]

# Z-height grasp depth thresholds (ESP32 FK Z in mm)
# Z=30mm ~ table surface, Z=80mm ~ object top, Z=160mm ~ approach, Z=220mm ~ home
Z_DEEP = 80       # below 80mm = DEEP grasp
Z_APPROACH = 160  # 80-160mm = APPROACH

# Gripper thresholds
GRIP_OPEN_THRESH  = 30  # >30 deg = open
GRIP_CLOSED_THRESH = 15 # <15 deg = closed
GRIP_PARTIAL_THRESH = 20 # sponge grasped at ~24 deg

# Episode quality thresholds
MIN_FRAMES = 30          # below = SHORT
MAX_FRAMES = 300         # above = LONG
MAX_STATIC_PCT = 60      # above = STATIC episode
MIN_GRIPPER_MAX = 20     # below = NO_GRIP

# V1 dataset reference stats (50 episodes, old collection, 68% SHALLOW)
V1_STATS = {
    "n_episodes": 50,
    "total_frames": 10803,
    "mean_frames": 216.1,
    "deep_pct": 18.0,   # 9/50
    "approach_pct": 14.0,  # 7/50
    "shallow_pct": 68.0,   # 34/50
    "base_range": (-11.0, 16.0),
    "base_std": 5.2,
    "gripper_open_pct": 21.0,  # % frames > 30 deg
    "action_mean": [2.71, 40.31, 13.04, 62.75, -2.65, 9.61],
    "action_std":  [15.2, 22.1, 18.8, 28.3, 14.9, 19.5],
    "notes": "Elbow-based depth (DEEP = elbow < -30 deg), gripper bias toward closed",
}


# ============================================================
# HELPERS
# ============================================================

def load_episode(ep_dir):
    """Load metadata.json from an episode directory. Returns dict or None."""
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def classify_position(base_angle):
    """Classify base angle into position zone string."""
    if base_angle < -30:
        return "LEFT_FAR"
    elif base_angle < -10:
        return "LEFT"
    elif base_angle < 10:
        return "CENTER"
    elif base_angle < 30:
        return "RIGHT"
    else:
        return "RIGHT_FAR"


def classify_depth(min_z):
    """Classify Z-height into depth category."""
    if min_z is None:
        return "UNKNOWN"
    if min_z < Z_DEEP:
        return "DEEP"
    elif min_z < Z_APPROACH:
        return "APPROACH"
    else:
        return "SHALLOW"


def classify_gripper_pattern(ep):
    """
    Classify gripper trajectory pattern for an episode.

    Returns one of:
      OPEN_THEN_CLOSE  - gripper opens above GRIP_OPEN_THRESH then closes below GRIP_CLOSED_THRESH
      OPENS_STAYS_OPEN - gripper opens but settles at partial (sponge grasp)
      NO_OPEN          - gripper never exceeds GRIP_OPEN_THRESH
    """
    frames = ep.get("frames", [])
    if not frames:
        return "NO_DATA"

    grip_vals = np.array([f["angles"][5] for f in frames])
    g_max = np.max(grip_vals)

    if g_max < GRIP_OPEN_THRESH:
        return "NO_OPEN"

    # Find peak
    peak_idx = int(np.argmax(grip_vals))
    post_peak = grip_vals[peak_idx:]

    if len(post_peak) > 3 and np.min(post_peak) < GRIP_CLOSED_THRESH:
        return "OPEN_THEN_CLOSE"
    else:
        return "OPENS_STAYS_OPEN"


def get_episode_flags(ep, angles_arr, static_pct):
    """
    Return list of quality flag strings for an episode.

    FLAGS:
      SHORT      - fewer than MIN_FRAMES frames
      LONG       - more than MAX_FRAMES frames
      STATIC     - static_pct > MAX_STATIC_PCT
      NO_GRIP    - gripper never opens (max < MIN_GRIPPER_MAX)
      INCOMPLETE - no z data or gripper never opens and z is shallow
    """
    flags = []
    nf = ep["num_frames"]

    if nf < MIN_FRAMES:
        flags.append("SHORT")
    if nf > MAX_FRAMES:
        flags.append("LONG")

    if static_pct > MAX_STATIC_PCT:
        flags.append("STATIC")

    g_max = ep.get("gripper_max", 0)
    if g_max < MIN_GRIPPER_MAX:
        flags.append("NO_GRIP")

    min_z = ep.get("min_z", None)
    if min_z is None:
        flags.append("NO_Z_DATA")
    elif min_z >= Z_APPROACH and g_max < MIN_GRIPPER_MAX:
        flags.append("INCOMPLETE")

    return flags


def sep(n=80):
    return "=" * n


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print(sep())
    print("COMPREHENSIVE DATASET ANALYSIS - v4 (51 episodes, 2026-02-24)")
    print(sep())
    print(f"  Data directory: {DATA_DIR}")
    print()

    # ---- Discover and load all episodes ----
    ep_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    ])

    episodes = []
    for ep_dir in ep_dirs:
        meta = load_episode(ep_dir)
        if meta is None:
            print(f"  WARNING: No metadata.json in {ep_dir.name}")
            continue
        meta["_dir_name"] = ep_dir.name
        episodes.append(meta)

    print(f"  Episode directories found: {len(ep_dirs)}")
    print(f"  Episodes with valid metadata: {len(episodes)}")
    if not episodes:
        print("  ERROR: No valid episodes!")
        return

    # ---- Pre-compute frame-level arrays ----
    all_angles = []
    all_poses = []
    ep_frame_angles = {}   # ep_id -> np.array shape (N, 6)
    ep_base_means = []
    ep_static_pcts = []
    ep_gripper_patterns = []
    ep_flags = []

    for ep in episodes:
        ep_id = ep["episode_id"]
        frames = ep.get("frames", [])
        if not frames:
            ep_frame_angles[ep_id] = np.zeros((0, 6))
            ep_base_means.append(0.0)
            ep_static_pcts.append(0.0)
            ep_gripper_patterns.append("NO_DATA")
            ep_flags.append(["NO_FRAMES"])
            continue

        angles_arr = np.array([f["angles"] for f in frames])
        all_angles.append(angles_arr)
        ep_frame_angles[ep_id] = angles_arr

        if "pose" in frames[0] and frames[0]["pose"] is not None:
            poses_arr = np.array([f["pose"] for f in frames if f.get("pose") is not None])
            if len(poses_arr) > 0:
                all_poses.append(poses_arr)

        ep_base_means.append(float(np.mean(angles_arr[:, 0])))

        # Static pct
        if len(angles_arr) >= 2:
            diffs = np.abs(np.diff(angles_arr, axis=0))
            max_diff = np.max(diffs, axis=1)
            sp = 100.0 * float(np.sum(max_diff < 0.5)) / len(max_diff)
        else:
            sp = 0.0
        ep_static_pcts.append(sp)

        # Gripper pattern
        ep_gripper_patterns.append(classify_gripper_pattern(ep))

        # Episode flags
        ep_flags.append(get_episode_flags(ep, angles_arr, sp))

    all_angles_flat = np.vstack(all_angles) if all_angles else np.zeros((0, 6))
    n_eps = len(episodes)
    num_frames_list = [ep["num_frames"] for ep in episodes]
    fps_list = [ep.get("fps", 30) for ep in episodes]
    total_frames = sum(num_frames_list)

    # ============================================================
    # 1. BASIC STATS
    # ============================================================
    print()
    print(sep())
    print("1. BASIC STATS")
    print(sep())

    durations_s = [nf / fp for nf, fp in zip(num_frames_list, fps_list)]
    total_dur = sum(durations_s)

    print(f"  Total episodes:    {n_eps}")
    print(f"  Total frames:      {total_frames}")
    print(f"  Total duration:    {total_dur:.1f}s  ({total_dur/60:.1f} min)")
    print(f"  Mean frames/ep:    {np.mean(num_frames_list):.1f}")
    print(f"  Median frames/ep:  {np.median(num_frames_list):.1f}")
    print(f"  Std frames/ep:     {np.std(num_frames_list):.1f}")
    print(f"  Min frames/ep:     {min(num_frames_list)}  (ep {episodes[num_frames_list.index(min(num_frames_list))]['episode_id']:04d})")
    print(f"  Max frames/ep:     {max(num_frames_list)}  (ep {episodes[num_frames_list.index(max(num_frames_list))]['episode_id']:04d})")
    print(f"  Mean duration:     {np.mean(durations_s):.2f}s")
    print(f"  Min duration:      {np.min(durations_s):.2f}s")
    print(f"  Max duration:      {np.max(durations_s):.2f}s")
    print(f"  FPS:               {fps_list[0]} (all consistent: {'YES' if len(set(fps_list))==1 else 'NO - VARIES!'})")

    # Duration histogram
    print(f"\n  Duration histogram (seconds):")
    bins = [0, 2, 3, 4, 5, 6, 7, 8, 10, 15]
    for i in range(len(bins)-1):
        c = sum(1 for d in durations_s if bins[i] <= d < bins[i+1])
        bar = "#" * c
        print(f"    {bins[i]:2d}-{bins[i+1]:2d}s: {c:3d}  {bar}")
    c = sum(1 for d in durations_s if d >= bins[-1])
    if c:
        print(f"    >={bins[-1]:2d}s: {c:3d}  {'#'*c}")

    # ============================================================
    # 2. POSITION DIVERSITY
    # ============================================================
    print()
    print(sep())
    print("2. POSITION DIVERSITY (Base Angle)")
    print(sep())

    base_all = all_angles_flat[:, 0] if len(all_angles_flat) > 0 else np.array([])
    print(f"  Frame-level base angle:")
    if len(base_all):
        print(f"    Min:  {np.min(base_all):.1f} deg")
        print(f"    Max:  {np.max(base_all):.1f} deg")
        print(f"    Mean: {np.mean(base_all):.1f} deg")
        print(f"    Std:  {np.std(base_all):.1f} deg")

    print(f"\n  Episode-level base angle (mean per episode):")
    print(f"    Min:  {np.min(ep_base_means):.1f} deg")
    print(f"    Max:  {np.max(ep_base_means):.1f} deg")
    print(f"    Mean: {np.mean(ep_base_means):.1f} deg")
    print(f"    Std:  {np.std(ep_base_means):.1f} deg")

    # Zone classification
    ep_zones = defaultdict(list)
    for i, ep in enumerate(episodes):
        zone = classify_position(ep_base_means[i])
        ep_zones[zone].append(ep["_dir_name"])

    print(f"\n  Zone distribution (by episode mean base angle):")
    print(f"  {'Zone':12s} {'Count':>6s} {'Pct':>6s}  {'Episodes (if <= 8)':s}")
    print(f"  {'-'*12} {'-'*6} {'-'*6}  {'-'*40}")
    for zone in ZONE_ORDER:
        eps_in = ep_zones.get(zone, [])
        pct = 100.0 * len(eps_in) / n_eps
        ep_names = ", ".join(e.replace("episode_", "ep") for e in eps_in[:8])
        extra = f"  (+{len(eps_in)-8} more)" if len(eps_in) > 8 else ""
        print(f"  {zone:12s} {len(eps_in):6d} {pct:5.1f}%  {ep_names}{extra}")

    missing = [z for z in ZONE_ORDER if not ep_zones.get(z)]
    weak    = [z for z in ZONE_ORDER if 0 < len(ep_zones.get(z, [])) < 3]
    if missing:
        print(f"\n  *** MISSING ZONES: {', '.join(missing)} ***")
    if weak:
        print(f"  *** WEAK ZONES (<3 eps): {', '.join(weak)} ***")

    # V1 comparison note
    print(f"\n  V1 comparison: base range was [-11.0, +16.0] deg, std ~5.2 deg (CENTER-heavy)")
    print(f"  V4 current:    base range [{np.min(ep_base_means):.1f}, {np.max(ep_base_means):.1f}] deg, std {np.std(ep_base_means):.1f} deg")
    improvement = np.std(ep_base_means) / 5.2 if 5.2 > 0 else 0
    print(f"  Improvement:   {improvement:.1f}x more diverse in base angle")

    # ============================================================
    # 3. GRASP DEPTH (Z-height from ESP32 FK)
    # ============================================================
    print()
    print(sep())
    print("3. GRASP DEPTH (Z-height from ESP32 FK)")
    print(sep())

    min_z_list = [ep["min_z"] for ep in episodes if ep.get("min_z") is not None]
    max_z_list = [ep["max_z"] for ep in episodes if ep.get("max_z") is not None]
    z_at_grip_list = [ep["z_at_grip_close"] for ep in episodes if ep.get("z_at_grip_close") is not None]

    depth_classes = {"DEEP": [], "APPROACH": [], "SHALLOW": [], "UNKNOWN": []}
    for ep in episodes:
        cat = classify_depth(ep.get("min_z"))
        depth_classes[cat].append(ep["_dir_name"])

    if min_z_list:
        print(f"  Min Z (deepest point reached per episode):")
        print(f"    Mean:   {np.mean(min_z_list):.1f} mm")
        print(f"    Std:    {np.std(min_z_list):.1f} mm")
        print(f"    Min:    {np.min(min_z_list):.1f} mm  (deepest)")
        print(f"    Max:    {np.max(min_z_list):.1f} mm  (shallowest)")

    if max_z_list:
        print(f"\n  Max Z (home/retract position):")
        print(f"    Mean: {np.mean(max_z_list):.1f} mm,  Std: {np.std(max_z_list):.1f} mm")

    print(f"\n  Depth classification (by min_z, threshold DEEP<{Z_DEEP}mm, APPROACH<{Z_APPROACH}mm):")
    print(f"    DEEP (<{Z_DEEP}mm):        {len(depth_classes['DEEP']):3d}/{n_eps}  ({100*len(depth_classes['DEEP'])/n_eps:.1f}%)")
    print(f"    APPROACH ({Z_DEEP}-{Z_APPROACH}mm): {len(depth_classes['APPROACH']):3d}/{n_eps}  ({100*len(depth_classes['APPROACH'])/n_eps:.1f}%)")
    print(f"    SHALLOW (>{Z_APPROACH}mm):   {len(depth_classes['SHALLOW']):3d}/{n_eps}  ({100*len(depth_classes['SHALLOW'])/n_eps:.1f}%)")
    if depth_classes["UNKNOWN"]:
        print(f"    UNKNOWN:              {len(depth_classes['UNKNOWN']):3d}/{n_eps}  (missing z data)")

    if z_at_grip_list:
        print(f"\n  Z at grip close ({len(z_at_grip_list)} episodes with grip_close_frame):")
        print(f"    Mean: {np.mean(z_at_grip_list):.1f} mm")
        print(f"    Std:  {np.std(z_at_grip_list):.1f} mm")
        print(f"    Min:  {np.min(z_at_grip_list):.1f} mm")
        print(f"    Max:  {np.max(z_at_grip_list):.1f} mm")
    else:
        print(f"\n  Z at grip close: no episodes with grip_close_frame set")

    print(f"\n  V1 comparison: DEEP={V1_STATS['deep_pct']:.0f}%, APPROACH={V1_STATS['approach_pct']:.0f}%, SHALLOW={V1_STATS['shallow_pct']:.0f}%")
    print(f"  V4 current:    DEEP={100*len(depth_classes['DEEP'])/n_eps:.0f}%, APPROACH={100*len(depth_classes['APPROACH'])/n_eps:.0f}%, SHALLOW={100*len(depth_classes['SHALLOW'])/n_eps:.0f}%")

    # ============================================================
    # 4. GRIPPER PHASE ANALYSIS
    # ============================================================
    print()
    print(sep())
    print("4. GRIPPER PHASE ANALYSIS")
    print(sep())

    gripper_all = all_angles_flat[:, 5] if len(all_angles_flat) > 0 else np.array([])

    if len(gripper_all):
        frames_open   = int(np.sum(gripper_all > GRIP_OPEN_THRESH))
        frames_closed = int(np.sum(gripper_all < GRIP_CLOSED_THRESH))
        frames_partial = int(np.sum((gripper_all >= GRIP_CLOSED_THRESH) & (gripper_all <= GRIP_OPEN_THRESH)))

        print(f"  Frame-level gripper distribution (total {len(gripper_all)} frames):")
        print(f"    Open (>{GRIP_OPEN_THRESH} deg):      {frames_open:6d}  ({100*frames_open/len(gripper_all):.1f}%)")
        print(f"    Partial ({GRIP_CLOSED_THRESH}-{GRIP_OPEN_THRESH} deg): {frames_partial:6d}  ({100*frames_partial/len(gripper_all):.1f}%)")
        print(f"    Closed (<{GRIP_CLOSED_THRESH} deg):   {frames_closed:6d}  ({100*frames_closed/len(gripper_all):.1f}%)")
        print(f"    Mean: {np.mean(gripper_all):.1f} deg,  Std: {np.std(gripper_all):.1f} deg")

    # Episode-level gripper stats
    grip_max_all = [ep.get("gripper_max") for ep in episodes if ep.get("gripper_max") is not None]
    grip_range_all = [ep.get("gripper_range") for ep in episodes if ep.get("gripper_range") is not None]

    if grip_max_all:
        print(f"\n  Episode-level gripper max:")
        print(f"    Mean: {np.mean(grip_max_all):.1f} deg")
        print(f"    Min:  {np.min(grip_max_all):.1f} deg")
        print(f"    Max:  {np.max(grip_max_all):.1f} deg")
        print(f"    Histogram:")
        gbins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        for i in range(len(gbins)-1):
            c = sum(1 for g in grip_max_all if gbins[i] <= g < gbins[i+1])
            bar = "#" * c
            print(f"      {gbins[i]:3d}-{gbins[i+1]:3d} deg: {c:3d} {bar}")

    # Grip open timing
    open_frames_list = [ep.get("grip_open_frame") for ep in episodes if ep.get("grip_open_frame") is not None]
    close_frames_list = [ep.get("grip_close_frame") for ep in episodes if ep.get("grip_close_frame") is not None]

    if open_frames_list:
        # Normalize as % of episode
        open_pcts = []
        for ep in episodes:
            gof = ep.get("grip_open_frame")
            if gof is not None:
                open_pcts.append(100.0 * gof / ep["num_frames"])
        print(f"\n  Grip open frame (gripper first > {GRIP_OPEN_THRESH} deg):")
        print(f"    Episodes with grip_open: {len(open_frames_list)}/{n_eps}")
        print(f"    Mean % into episode: {np.mean(open_pcts):.1f}%  (gripper opens at this point)")
        print(f"    Std:  {np.std(open_pcts):.1f}%")
        print(f"    Range: [{np.min(open_pcts):.1f}%, {np.max(open_pcts):.1f}%]")

    if close_frames_list:
        close_pcts = []
        for ep in episodes:
            gcf = ep.get("grip_close_frame")
            if gcf is not None:
                close_pcts.append(100.0 * gcf / ep["num_frames"])
        print(f"\n  Grip close frame (gripper first drops back < {GRIP_CLOSED_THRESH} deg after peak):")
        print(f"    Episodes with grip_close: {len(close_frames_list)}/{n_eps}")
        print(f"    Mean % into episode: {np.mean(close_pcts):.1f}%")
        print(f"    Std:  {np.std(close_pcts):.1f}%")
    else:
        print(f"\n  Grip close frame: {len(close_frames_list)}/{n_eps} episodes")
        print(f"    (Most episodes end with gripper in partial/sponge-grasped position)")

    # Gripper pattern classification
    pattern_counts = defaultdict(int)
    pattern_eps = defaultdict(list)
    for i, (ep, pat) in enumerate(zip(episodes, ep_gripper_patterns)):
        pattern_counts[pat] += 1
        pattern_eps[pat].append(ep["_dir_name"])

    print(f"\n  Gripper trajectory patterns:")
    for pat in ["OPEN_THEN_CLOSE", "OPENS_STAYS_OPEN", "NO_OPEN", "NO_DATA"]:
        c = pattern_counts.get(pat, 0)
        pct = 100.0 * c / n_eps
        desc = {
            "OPEN_THEN_CLOSE":  "open → approach → CLOSE below 15 deg (full grasp)",
            "OPENS_STAYS_OPEN": "open → settles at partial/sponge grip (~24 deg)",
            "NO_OPEN":          "gripper never exceeds 30 deg (possible bad episode)",
            "NO_DATA":          "no frame data",
        }.get(pat, "")
        print(f"    {pat:20s}: {c:3d} ({pct:5.1f}%)  -- {desc}")

    # Episodes with NO_OPEN are suspect
    no_open_eps = pattern_eps.get("NO_OPEN", [])
    if no_open_eps:
        print(f"\n  *** WARNING: {len(no_open_eps)} episodes where gripper never opens: ***")
        for e in no_open_eps:
            ep_obj = next(x for x in episodes if x["_dir_name"] == e)
            print(f"      {e}: gripper_max={ep_obj.get('gripper_max', '?'):.1f} deg")

    # Phase pattern description
    print(f"\n  Note on sponge grasping:")
    print(f"    OPENS_STAYS_OPEN = gripper opens to ~60-108 deg, then settles at ~24 deg = sponge gripped")
    print(f"    This is CORRECT behavior for a soft/compressible sponge object.")
    print(f"    ~24 deg at end = sponge compressed in gripper, NOT incomplete grasp.")

    # V1 comparison
    if len(gripper_all):
        grip_open_pct = 100 * frames_open / len(gripper_all)
        print(f"\n  V1 comparison: gripper open ratio ~{V1_STATS['gripper_open_pct']:.0f}%")
        print(f"  V4 current:    gripper open ratio  {grip_open_pct:.1f}%")

    # ============================================================
    # 5. JOINT DIVERSITY
    # ============================================================
    print()
    print(sep())
    print("5. JOINT DIVERSITY (frame-level across all episodes)")
    print(sep())

    if len(all_angles_flat) > 0:
        print(f"  {'Joint':15s} {'Min':>8s} {'Max':>8s} {'Range':>8s} {'Mean':>8s} {'Std':>8s}  {'HW_Limits':s}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

        for j, jname in enumerate(JOINT_NAMES):
            jdata = all_angles_flat[:, j]
            jmin  = float(np.min(jdata))
            jmax  = float(np.max(jdata))
            jrng  = jmax - jmin
            jmean = float(np.mean(jdata))
            jstd  = float(np.std(jdata))
            hw_lo, hw_hi = JOINT_LIMITS[jname]
            hw_range = hw_hi - hw_lo
            coverage = 100.0 * jrng / hw_range if hw_range > 0 else 0
            print(f"  {jname:15s} {jmin:8.1f} {jmax:8.1f} {jrng:8.1f} {jmean:8.1f} {jstd:8.1f}  [{hw_lo},{hw_hi}] {coverage:.0f}% covered")

        # Action stats for training config
        action_mean = np.mean(all_angles_flat, axis=0)
        action_std  = np.std(all_angles_flat, axis=0)
        print(f"\n  Action normalization stats (ALL {n_eps} episodes):")
        print(f"    Mean: [{', '.join(f'{v:.2f}' for v in action_mean)}]")
        print(f"    Std:  [{', '.join(f'{v:.2f}' for v in action_std)}]")

        # V1 comparison
        print(f"\n  V1 action mean: {V1_STATS['action_mean']}")
        print(f"  V4 action mean: [{', '.join(f'{v:.2f}' for v in action_mean)}]")

        # Check for suspiciously low diversity
        print(f"\n  Diversity check (std < 5 deg = suspect):")
        for j, jname in enumerate(JOINT_NAMES):
            jstd = float(np.std(all_angles_flat[:, j]))
            if jstd < 5.0:
                print(f"    *** {jname}: std={jstd:.2f} deg -- VERY LOW DIVERSITY ***")
        any_low = any(np.std(all_angles_flat[:, j]) < 5.0 for j in range(6))
        if not any_low:
            print(f"    All joints have std >= 5 deg -- OK")

    # ============================================================
    # 6. EPISODE QUALITY FLAGS
    # ============================================================
    print()
    print(sep())
    print("6. EPISODE QUALITY FLAGS")
    print(sep())

    # Summarize flags
    all_flags_flat = [f for ep_fl in ep_flags for f in ep_fl]
    flag_counts = defaultdict(int)
    for f in all_flags_flat:
        flag_counts[f] += 1

    print(f"  Flag summary:")
    for flag, count in sorted(flag_counts.items()):
        pct = 100.0 * count / n_eps
        print(f"    {flag:15s}: {count:3d} episodes ({pct:.1f}%)")

    if not all_flags_flat:
        print(f"    No quality issues detected -- all {n_eps} episodes pass basic checks")

    # List flagged episodes
    flagged_eps = [(ep["_dir_name"], ep_flags[i]) for i, ep in enumerate(episodes) if ep_flags[i]]
    if flagged_eps:
        print(f"\n  Flagged episodes ({len(flagged_eps)} total):")
        print(f"  {'Episode':15s} {'Frames':>6s} {'Flags':s}")
        print(f"  {'-'*15} {'-'*6} {'-'*40}")
        for ep_name, flags in flagged_eps:
            ep_obj = next(x for x in episodes if x["_dir_name"] == ep_name)
            flag_str = " | ".join(flags)
            print(f"  {ep_name:15s} {ep_obj['num_frames']:6d}  {flag_str}")
    else:
        print(f"\n  No episodes flagged!")

    # ============================================================
    # 7. TEMPORAL QUALITY (Static frames)
    # ============================================================
    print()
    print(sep())
    print("7. TEMPORAL QUALITY (Static Frames)")
    print(sep())

    # Compute global static stats
    total_trans = 0
    total_static = 0
    for ep_id, angles_arr in ep_frame_angles.items():
        if len(angles_arr) < 2:
            continue
        diffs = np.abs(np.diff(angles_arr, axis=0))
        max_diff = np.max(diffs, axis=1)
        total_trans  += len(max_diff)
        total_static += int(np.sum(max_diff < 0.5))

    global_static_pct = 100.0 * total_static / total_trans if total_trans > 0 else 0

    print(f"  Global static frame ratio (max joint change < 0.5 deg):")
    print(f"    Total transitions: {total_trans}")
    print(f"    Static:            {total_static}  ({global_static_pct:.1f}%)")
    print(f"\n  Episode-level static %:")
    if ep_static_pcts:
        print(f"    Mean: {np.mean(ep_static_pcts):.1f}%")
        print(f"    Min:  {np.min(ep_static_pcts):.1f}%")
        print(f"    Max:  {np.max(ep_static_pcts):.1f}%")

    high_static = [(episodes[i]["_dir_name"], ep_static_pcts[i])
                   for i in range(len(ep_static_pcts)) if ep_static_pcts[i] > MAX_STATIC_PCT]
    if high_static:
        print(f"\n  *** HIGH STATIC EPISODES (>{MAX_STATIC_PCT}%): {len(high_static)} ***")
        for name, pct in high_static:
            print(f"      {name}: {pct:.1f}% static")

    print(f"\n  Static frame histogram (episode-level %):")
    sbins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    for i in range(len(sbins)-1):
        c = sum(1 for sp in ep_static_pcts if sbins[i] <= sp < sbins[i+1])
        bar = "#" * c
        print(f"    {sbins[i]:3d}-{sbins[i+1]:3d}%: {c:3d}  {bar}")

    # ============================================================
    # 8. OLD vs NEW EPISODE COMPARISON + V1 REFERENCE
    # ============================================================
    print()
    print(sep())
    print("8. OLD vs NEW EPISODE COMPARISON (+ V1 dataset reference)")
    print(sep())

    old_eps = [ep for ep in episodes if ep["episode_id"] <= 30]   # ep 0-30
    new_eps = [ep for ep in episodes if ep["episode_id"] > 30]    # ep 31-50

    def ep_stats(ep_list, ep_bm_map):
        """Compute summary stats for a list of episodes."""
        if not ep_list:
            return {}
        ids = [ep["episode_id"] for ep in ep_list]
        frames = [ep["num_frames"] for ep in ep_list]
        bm = [ep_bm_map[ep["episode_id"]] for ep in ep_list if ep["episode_id"] in ep_bm_map]
        min_z = [ep["min_z"] for ep in ep_list if ep.get("min_z") is not None]
        grip_max = [ep.get("gripper_max", 0) for ep in ep_list]
        deep_c = sum(1 for ep in ep_list if classify_depth(ep.get("min_z")) == "DEEP")
        return {
            "n": len(ep_list),
            "mean_frames": np.mean(frames),
            "base_mean": np.mean(bm) if bm else float("nan"),
            "base_std": np.std(bm) if bm else float("nan"),
            "base_min": np.min(bm) if bm else float("nan"),
            "base_max": np.max(bm) if bm else float("nan"),
            "min_z_mean": np.mean(min_z) if min_z else float("nan"),
            "min_z_std": np.std(min_z) if min_z else float("nan"),
            "grip_max_mean": np.mean(grip_max),
            "deep_pct": 100.0 * deep_c / len(ep_list) if ep_list else 0,
        }

    ep_bm_map = {ep["episode_id"]: ep_base_means[i] for i, ep in enumerate(episodes)}

    old_s = ep_stats(old_eps, ep_bm_map)
    new_s = ep_stats(new_eps, ep_bm_map)

    print(f"  {'Metric':30s} {'V1 (50ep)':>15s} {'OLD (ep0-30)':>15s} {'NEW (ep31-50)':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")

    def fmt(v, fmt_str=".1f"):
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        return format(v, fmt_str)

    rows = [
        ("Episodes",           f"{V1_STATS['n_episodes']}",       f"{old_s['n']}",           f"{new_s['n']}"),
        ("Mean frames/ep",     f"{V1_STATS['mean_frames']:.1f}",  fmt(old_s['mean_frames']), fmt(new_s['mean_frames'])),
        ("Base std (deg)",     f"{V1_STATS['base_std']:.1f}",     fmt(old_s['base_std']),    fmt(new_s['base_std'])),
        ("Base range (deg)",   f"[-11,+16]",   f"[{fmt(old_s['base_min'])},{fmt(old_s['base_max'])}]",
                                               f"[{fmt(new_s['base_min'])},{fmt(new_s['base_max'])}]"),
        ("Min Z mean (mm)",    "N/A",          fmt(old_s['min_z_mean']),  fmt(new_s['min_z_mean'])),
        ("DEEP grasp %",       f"{V1_STATS['deep_pct']:.0f}%",   f"{old_s['deep_pct']:.0f}%", f"{new_s['deep_pct']:.0f}%"),
        ("Grip max mean (deg)","N/A",          fmt(old_s['grip_max_mean']),fmt(new_s['grip_max_mean'])),
    ]
    for label, v1, old, new in rows:
        print(f"  {label:30s} {v1:>15s} {old:>15s} {new:>15s}")

    # ============================================================
    # 9. SHOULDER DISTRIBUTION
    # ============================================================
    print()
    print(sep())
    print("9. SHOULDER DISTRIBUTION (depth indicator)")
    print(sep())

    if len(all_angles_flat) > 0:
        shoulder_all = all_angles_flat[:, 1]
        print(f"  Frame-level shoulder:")
        print(f"    Min: {np.min(shoulder_all):.1f}  Max: {np.max(shoulder_all):.1f}  Mean: {np.mean(shoulder_all):.1f}  Std: {np.std(shoulder_all):.1f} deg")

        max_sh_list = [ep.get("max_shoulder") for ep in episodes if ep.get("max_shoulder") is not None]
        if max_sh_list:
            print(f"\n  Episode max shoulder (proxy for grasp depth):")
            print(f"    Mean: {np.mean(max_sh_list):.1f}  Min: {np.min(max_sh_list):.1f}  Max: {np.max(max_sh_list):.1f} deg")
            deep_sh   = sum(1 for s in max_sh_list if s > 60)
            appr_sh   = sum(1 for s in max_sh_list if 40 <= s <= 60)
            shal_sh   = sum(1 for s in max_sh_list if s < 40)
            print(f"    DEEP (>60 deg):   {deep_sh:3d} ({100*deep_sh/len(max_sh_list):.1f}%)")
            print(f"    APPROACH (40-60): {appr_sh:3d} ({100*appr_sh/len(max_sh_list):.1f}%)")
            print(f"    SHALLOW (<40):    {shal_sh:3d} ({100*shal_sh/len(max_sh_list):.1f}%)")

        print(f"\n  Shoulder angle histogram (frame-level):")
        sh_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110]
        for i in range(len(sh_bins)-1):
            c = int(np.sum((shoulder_all >= sh_bins[i]) & (shoulder_all < sh_bins[i+1])))
            pct = 100.0 * c / len(shoulder_all)
            bar = "#" * int(pct)
            print(f"    {sh_bins[i]:3d}-{sh_bins[i+1]:3d} deg: {c:6d} ({pct:5.1f}%) {bar}")

    # ============================================================
    # 10. PER-EPISODE SUMMARY TABLE
    # ============================================================
    print()
    print(sep())
    print("10. PER-EPISODE SUMMARY TABLE")
    print(sep())

    hdr = f"  {'Ep':>4s}  {'Frames':>6s}  {'Dur(s)':>6s}  {'BaseMn':>7s}  {'ShMax':>6s}  {'MinZ':>7s}  {'GrpMax':>7s}  {'Zone':>10s}  {'Depth':>8s}  {'Pattern':>18s}  Flags"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        nf    = ep["num_frames"]
        dur   = nf / ep.get("fps", 30)
        base_mn = ep_base_means[i] if i < len(ep_base_means) else 0.0
        sh_max  = ep.get("max_shoulder", 0.0)
        min_z   = ep.get("min_z")
        gr_max  = ep.get("gripper_max", 0.0)
        zone    = classify_position(base_mn)
        depth   = classify_depth(min_z)
        pattern = ep_gripper_patterns[i] if i < len(ep_gripper_patterns) else "?"
        flags   = ep_flags[i] if i < len(ep_flags) else []
        flag_str = " ".join(flags) if flags else "-"
        min_z_str = f"{min_z:7.1f}" if min_z is not None else "    N/A"

        # Shorten pattern for table
        pat_short = {"OPEN_THEN_CLOSE": "OPEN>CLOSE", "OPENS_STAYS_OPEN": "OPEN>STAY", "NO_OPEN": "NO_OPEN"}.get(pattern, pattern)

        print(f"  {ep_id:4d}  {nf:6d}  {dur:6.1f}  {base_mn:7.1f}  {sh_max:6.1f}  {min_z_str}  {gr_max:7.1f}  {zone:>10s}  {depth:>8s}  {pat_short:>18s}  {flag_str}")

    # ============================================================
    # 11. TRAINING READINESS ASSESSMENT
    # ============================================================
    print()
    print(sep())
    print("11. TRAINING READINESS ASSESSMENT")
    print(sep())

    score = 0.0
    max_score = 10.0
    issues = []
    strengths = []

    # 1. Episode count
    if n_eps >= 100:
        score += 2
        strengths.append(f"Episode count: {n_eps} >= 100 (TARGET MET)")
    elif n_eps >= 70:
        score += 1.5
        issues.append(f"Episode count: {n_eps} (target 100+, good progress)")
    elif n_eps >= 50:
        score += 1.0
        issues.append(f"Episode count: {n_eps} (target 100+, below target)")
    else:
        issues.append(f"Episode count: {n_eps} (BELOW MINIMUM 50)")

    # 2. Grasp depth
    deep_pct_now = 100.0 * len(depth_classes["DEEP"]) / n_eps
    if deep_pct_now >= 80:
        score += 2
        strengths.append(f"Grasp depth: {deep_pct_now:.0f}% DEEP (excellent)")
    elif deep_pct_now >= 60:
        score += 1.5
        strengths.append(f"Grasp depth: {deep_pct_now:.0f}% DEEP (good)")
    elif deep_pct_now >= 40:
        score += 1
        issues.append(f"Grasp depth: {deep_pct_now:.0f}% DEEP (want 80%+)")
    else:
        issues.append(f"Grasp depth: {deep_pct_now:.0f}% DEEP (critical gap)")

    # 3. Position diversity
    filled_zones = sum(1 for z in ZONE_ORDER if len(ep_zones.get(z, [])) > 0)
    zones_with_3plus = sum(1 for z in ZONE_ORDER if len(ep_zones.get(z, [])) >= 3)
    if filled_zones >= 5 and zones_with_3plus >= 4:
        score += 2
        strengths.append(f"Position diversity: all 5 zones filled, {zones_with_3plus}/5 with 3+ eps")
    elif filled_zones >= 4 and zones_with_3plus >= 3:
        score += 1.5
        issues.append(f"Position diversity: {filled_zones}/5 zones filled, weak zones need more eps")
    elif filled_zones >= 3:
        score += 1
        issues.append(f"Position diversity: only {filled_zones}/5 zones filled")
    else:
        issues.append(f"Position diversity: critically low ({filled_zones}/5 zones)")

    # 4. Gripper coverage
    if len(gripper_all) > 0:
        gop = 100.0 * frames_open / len(gripper_all)
        if gop >= 30:
            score += 1
            strengths.append(f"Gripper open ratio: {gop:.1f}% (>30%)")
        elif gop >= 20:
            score += 0.5
            issues.append(f"Gripper open ratio: {gop:.1f}% (want 30%+)")
        else:
            issues.append(f"Gripper open ratio: {gop:.1f}% (too low, want 30%+)")

    # 5. Static frame quality
    if global_static_pct < 25:
        score += 1
        strengths.append(f"Temporal quality: {global_static_pct:.1f}% static (good, <25%)")
    elif global_static_pct < 40:
        score += 0.5
        issues.append(f"Temporal quality: {global_static_pct:.1f}% static (moderate, want <25%)")
    else:
        issues.append(f"Temporal quality: {global_static_pct:.1f}% static (high, want <25%)")

    # 6. Duration consistency
    short_count = sum(1 for nf in num_frames_list if nf < MIN_FRAMES)
    long_count  = sum(1 for nf in num_frames_list if nf > MAX_FRAMES)
    if short_count == 0 and long_count == 0:
        score += 1
        strengths.append(f"Duration: all episodes in normal range ({MIN_FRAMES}-{MAX_FRAMES} frames)")
    elif short_count == 0:
        score += 0.5
        issues.append(f"Duration: {long_count} too-long episodes (>300 frames)")
    else:
        issues.append(f"Duration: {short_count} too-short episodes")

    # 7. No-grip episodes
    no_grip_count = sum(1 for p in ep_gripper_patterns if p == "NO_OPEN")
    if no_grip_count == 0:
        score += 1
        strengths.append("All episodes have gripper opening")
    elif no_grip_count <= 2:
        score += 0.5
        issues.append(f"{no_grip_count} episodes where gripper never opens")
    else:
        issues.append(f"{no_grip_count} episodes where gripper never opens (remove these)")

    print(f"\n  SCORE: {score:.1f} / {max_score:.0f}")
    print(f"\n  STRENGTHS:")
    for s in strengths:
        print(f"    [+] {s}")
    print(f"\n  ISSUES:")
    for iss in issues:
        print(f"    [-] {iss}")

    if score >= 8:
        verdict = "READY FOR TRAINING"
    elif score >= 6:
        verdict = "PARTIALLY READY - address issues first"
    elif score >= 4:
        verdict = "NEEDS MORE WORK - significant gaps"
    else:
        verdict = "NOT READY - major quality issues"

    print(f"\n  VERDICT: {verdict}")

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================
    print()
    print(sep())
    print("RECOMMENDATIONS")
    print(sep())

    rec_n = 1

    if n_eps < 100:
        need = 100 - n_eps
        print(f"\n  {rec_n}. COLLECT {need} MORE EPISODES to reach 100+ target")
        print(f"     Priority zones for new episodes:")
        for zone in ZONE_ORDER:
            c = len(ep_zones.get(zone, []))
            if c < 5:
                lo, hi = POSITION_ZONES[zone]
                needed = max(5, 15) - c
                print(f"       {zone}: has {c}, need ~{needed} more (place sponge so base={lo} to {hi} deg)")
        rec_n += 1

    if missing:
        print(f"\n  {rec_n}. FILL MISSING ZONES: {', '.join(missing)}")
        rec_n += 1

    weak_zones_full = [z for z in ZONE_ORDER if 0 < len(ep_zones.get(z, [])) < 5]
    if weak_zones_full:
        print(f"\n  {rec_n}. STRENGTHEN WEAK ZONES (< 5 episodes):")
        for z in weak_zones_full:
            lo, hi = POSITION_ZONES[z]
            print(f"     {z}: {len(ep_zones[z])} eps, need {5 - len(ep_zones[z])} more (base={lo} to {hi} deg)")
        rec_n += 1

    if no_grip_count > 0:
        print(f"\n  {rec_n}. REVIEW/REMOVE {no_grip_count} NO_GRIP episodes:")
        for e in no_open_eps:
            print(f"     {e}")
        rec_n += 1

    if global_static_pct > 35:
        print(f"\n  {rec_n}. REDUCE STATIC FRAMES ({global_static_pct:.1f}% is high):")
        print(f"     - Demonstrate at 20-30% faster pace")
        print(f"     - Or apply frame deduplication in convert script")
        rec_n += 1

    if len(gripper_all) > 0 and (100.0 * frames_open / len(gripper_all)) < 25:
        print(f"\n  {rec_n}. IMPROVE GRIPPER OPEN COVERAGE:")
        print(f"     - Hold gripper open longer during approach phase")
        print(f"     - Current: {100.0*frames_open/len(gripper_all):.1f}% open frames, want 25%+")
        rec_n += 1

    # Final action norm stats for training config
    if len(all_angles_flat) > 0:
        action_mean = np.mean(all_angles_flat, axis=0)
        action_std  = np.std(all_angles_flat, axis=0)
        print(f"\n  UPDATED NORMALIZATION STATS (use these in training config):")
        print(f"     action.mean: [{', '.join(f'{v:.2f}' for v in action_mean)}]")
        print(f"     action.std:  [{', '.join(f'{v:.2f}' for v in action_std)}]")

    print()
    print(sep())
    print("END OF ANALYSIS")
    print(sep())


if __name__ == "__main__":
    main()
