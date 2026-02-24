#!/usr/bin/env python3
"""
Comprehensive analysis of 50+ episodes in collected_data/.
Covers: basic stats, position diversity, grasp depth, gripper behavior,
joint diversity, episode duration, temporal quality, old vs new comparison,
shoulder distribution, and training readiness.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_pitch", "Wrist_roll", "Gripper"]

# Position zones by base angle
POSITION_ZONES = {
    "LEFT_FAR":  (-90, -30),
    "LEFT":      (-30, -10),
    "CENTER":    (-10, 10),
    "RIGHT":     (10, 30),
    "RIGHT_FAR": (30, 90),
}

# Z-height grasp depth thresholds (ESP32 FK Z in mm)
# Z=30mm = table, Z=80mm = object top, Z=160mm = approach, Z=230mm = home
Z_DEEP = 80
Z_APPROACH = 160

def load_episode(ep_dir):
    """Load metadata.json from an episode directory."""
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)

def classify_position(base_angle):
    """Classify base angle into position zone."""
    for zone, (lo, hi) in POSITION_ZONES.items():
        if lo <= base_angle < hi:
            return zone
    if base_angle < -90:
        return "LEFT_FAR"
    if base_angle >= 90:
        return "RIGHT_FAR"
    # Edge cases
    if base_angle < -30:
        return "LEFT_FAR"
    if base_angle >= 30:
        return "RIGHT_FAR"
    return "CENTER"

def main():
    # Discover all episodes
    ep_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"{'='*80}")
    print(f"COMPREHENSIVE DATASET ANALYSIS - collected_data/")
    print(f"{'='*80}\n")

    # Load all episodes
    episodes = []
    for ep_dir in ep_dirs:
        meta = load_episode(ep_dir)
        if meta is None:
            print(f"  WARNING: No metadata.json in {ep_dir.name}")
            continue
        meta["_dir_name"] = ep_dir.name
        episodes.append(meta)

    print(f"Total episode directories: {len(ep_dirs)}")
    print(f"Episodes with valid metadata: {len(episodes)}")

    if not episodes:
        print("ERROR: No valid episodes found!")
        return

    # =========================================================================
    # 1. BASIC STATS
    # =========================================================================
    print(f"\n{'='*80}")
    print("1. BASIC STATS")
    print(f"{'='*80}")

    num_frames_list = [ep["num_frames"] for ep in episodes]
    total_frames = sum(num_frames_list)
    fps_list = [ep.get("fps", 30) for ep in episodes]

    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames:   {total_frames}")
    print(f"  Mean frames/ep: {np.mean(num_frames_list):.1f}")
    print(f"  Median frames:  {np.median(num_frames_list):.1f}")
    print(f"  Std frames:     {np.std(num_frames_list):.1f}")
    print(f"  Min frames:     {min(num_frames_list)} ({episodes[num_frames_list.index(min(num_frames_list))]['_dir_name']})")
    print(f"  Max frames:     {max(num_frames_list)} ({episodes[num_frames_list.index(max(num_frames_list))]['_dir_name']})")
    print(f"  FPS consistency: all={fps_list[0]} => {'YES' if len(set(fps_list))==1 else 'NO - VARIES!'}")

    total_duration_s = sum(nf / fp for nf, fp in zip(num_frames_list, fps_list))
    print(f"  Total duration:  {total_duration_s:.1f}s ({total_duration_s/60:.1f} min)")
    print(f"  Mean ep duration: {total_duration_s/len(episodes):.1f}s")

    # =========================================================================
    # 2. POSITION DIVERSITY (Base angle distribution)
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. POSITION DIVERSITY (Base Angle)")
    print(f"{'='*80}")

    # Collect all frame-level data
    all_angles = []  # shape: (N, 6)
    all_poses = []   # shape: (N, 3) - x, y, z
    ep_base_means = []
    ep_base_mins = []
    ep_base_maxs = []
    ep_frame_angles = {}  # ep_id -> list of angle arrays

    for ep in episodes:
        ep_id = ep["episode_id"]
        frames = ep.get("frames", [])
        if not frames:
            continue

        angles_arr = np.array([f["angles"] for f in frames])
        all_angles.append(angles_arr)
        ep_frame_angles[ep_id] = angles_arr

        # Pose data (x, y, z)
        if "pose" in frames[0] and frames[0]["pose"] is not None:
            poses_arr = np.array([f["pose"] for f in frames if f.get("pose") is not None])
            if len(poses_arr) > 0:
                all_poses.append(poses_arr)

        base_angles = angles_arr[:, 0]
        ep_base_means.append(np.mean(base_angles))
        ep_base_mins.append(np.min(base_angles))
        ep_base_maxs.append(np.max(base_angles))

    all_angles_flat = np.vstack(all_angles)
    base_all = all_angles_flat[:, 0]

    print(f"  Frame-level base angle:")
    print(f"    Min:  {np.min(base_all):.1f} deg")
    print(f"    Max:  {np.max(base_all):.1f} deg")
    print(f"    Mean: {np.mean(base_all):.1f} deg")
    print(f"    Std:  {np.std(base_all):.1f} deg")

    print(f"\n  Episode-level base angle (mean per episode):")
    print(f"    Min:  {np.min(ep_base_means):.1f} deg")
    print(f"    Max:  {np.max(ep_base_means):.1f} deg")
    print(f"    Mean: {np.mean(ep_base_means):.1f} deg")
    print(f"    Std:  {np.std(ep_base_means):.1f} deg")

    # Zone classification by episode
    ep_zones = defaultdict(list)
    for i, ep in enumerate(episodes):
        zone = classify_position(ep_base_means[i] if i < len(ep_base_means) else 0)
        ep_zones[zone].append(ep["_dir_name"])

    print(f"\n  Position zone distribution (by episode mean base angle):")
    zone_order = ["LEFT_FAR", "LEFT", "CENTER", "RIGHT", "RIGHT_FAR"]
    for zone in zone_order:
        eps_in_zone = ep_zones.get(zone, [])
        pct = 100 * len(eps_in_zone) / len(episodes) if episodes else 0
        print(f"    {zone:12s}: {len(eps_in_zone):3d} episodes ({pct:5.1f}%)")
        if len(eps_in_zone) <= 8:
            print(f"                 Episodes: {', '.join(eps_in_zone)}")

    # Check if any zone is completely missing
    missing_zones = [z for z in zone_order if z not in ep_zones or len(ep_zones[z]) == 0]
    if missing_zones:
        print(f"\n  *** MISSING ZONES: {', '.join(missing_zones)} ***")

    # =========================================================================
    # 3. GRASP DEPTH (Z-height)
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. GRASP DEPTH (Z-height from ESP32 FK)")
    print(f"{'='*80}")

    min_z_list = []
    max_z_list = []
    z_at_grip_list = []
    depth_classes = {"DEEP": [], "APPROACH": [], "SHALLOW": [], "UNKNOWN": []}

    for ep in episodes:
        min_z = ep.get("min_z")
        max_z = ep.get("max_z")
        z_at_grip = ep.get("z_at_grip_close")

        if min_z is not None:
            min_z_list.append(min_z)
        if max_z is not None:
            max_z_list.append(max_z)
        if z_at_grip is not None:
            z_at_grip_list.append(z_at_grip)

        # Classify by min_z
        if min_z is not None:
            if min_z < Z_DEEP:
                depth_classes["DEEP"].append(ep["_dir_name"])
            elif min_z < Z_APPROACH:
                depth_classes["APPROACH"].append(ep["_dir_name"])
            else:
                depth_classes["SHALLOW"].append(ep["_dir_name"])
        else:
            depth_classes["UNKNOWN"].append(ep["_dir_name"])

    if min_z_list:
        print(f"  Min Z across all episodes:")
        print(f"    Lowest:  {np.min(min_z_list):.1f} mm")
        print(f"    Highest: {np.max(min_z_list):.1f} mm")
        print(f"    Mean:    {np.mean(min_z_list):.1f} mm")
        print(f"    Std:     {np.std(min_z_list):.1f} mm")

    if max_z_list:
        print(f"\n  Max Z (home position):")
        print(f"    Mean: {np.mean(max_z_list):.1f} mm, Std: {np.std(max_z_list):.1f} mm")

    print(f"\n  Depth classification (by min_z):")
    print(f"    DEEP (<{Z_DEEP}mm):      {len(depth_classes['DEEP']):3d} episodes ({100*len(depth_classes['DEEP'])/len(episodes):.1f}%)")
    print(f"    APPROACH ({Z_DEEP}-{Z_APPROACH}mm): {len(depth_classes['APPROACH']):3d} episodes ({100*len(depth_classes['APPROACH'])/len(episodes):.1f}%)")
    print(f"    SHALLOW (>{Z_APPROACH}mm):  {len(depth_classes['SHALLOW']):3d} episodes ({100*len(depth_classes['SHALLOW'])/len(episodes):.1f}%)")
    if depth_classes["UNKNOWN"]:
        print(f"    UNKNOWN:          {len(depth_classes['UNKNOWN']):3d} episodes")

    if z_at_grip_list:
        print(f"\n  Z at grip close (actual grasp height):")
        print(f"    Mean: {np.mean(z_at_grip_list):.1f} mm")
        print(f"    Std:  {np.std(z_at_grip_list):.1f} mm")
        print(f"    Min:  {np.min(z_at_grip_list):.1f} mm")
        print(f"    Max:  {np.max(z_at_grip_list):.1f} mm")

    # =========================================================================
    # 4. GRIPPER BEHAVIOR
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. GRIPPER BEHAVIOR")
    print(f"{'='*80}")

    gripper_all = all_angles_flat[:, 5]

    # Thresholds
    GRIP_OPEN_THRESH = 30  # degrees
    GRIP_CLOSED_THRESH = 15

    frames_open = np.sum(gripper_all > GRIP_OPEN_THRESH)
    frames_closed = np.sum(gripper_all < GRIP_CLOSED_THRESH)
    frames_mid = total_frames - frames_open - frames_closed

    print(f"  Frame-level gripper distribution:")
    print(f"    Open (>{GRIP_OPEN_THRESH} deg):   {frames_open:6d} frames ({100*frames_open/len(gripper_all):.1f}%)")
    print(f"    Mid ({GRIP_CLOSED_THRESH}-{GRIP_OPEN_THRESH} deg):    {frames_mid:6d} frames ({100*frames_mid/len(gripper_all):.1f}%)")
    print(f"    Closed (<{GRIP_CLOSED_THRESH} deg): {frames_closed:6d} frames ({100*frames_closed/len(gripper_all):.1f}%)")
    print(f"    Overall mean:  {np.mean(gripper_all):.1f} deg")
    print(f"    Overall std:   {np.std(gripper_all):.1f} deg")

    # Episode-level gripper stats
    grip_max_list = [ep.get("gripper_max", None) for ep in episodes]
    grip_max_list = [g for g in grip_max_list if g is not None]
    grip_min_list = [ep.get("gripper_min", None) for ep in episodes]
    grip_min_list = [g for g in grip_min_list if g is not None]
    grip_range_list = [ep.get("gripper_range", None) for ep in episodes]
    grip_range_list = [g for g in grip_range_list if g is not None]

    if grip_max_list:
        print(f"\n  Episode-level gripper max:")
        print(f"    Mean: {np.mean(grip_max_list):.1f} deg")
        print(f"    Min:  {np.min(grip_max_list):.1f} deg")
        print(f"    Max:  {np.max(grip_max_list):.1f} deg")

    if grip_range_list:
        print(f"\n  Episode-level gripper range (max-min):")
        print(f"    Mean: {np.mean(grip_range_list):.1f} deg")
        print(f"    Min:  {np.min(grip_range_list):.1f} deg")
        print(f"    Max:  {np.max(grip_range_list):.1f} deg")

    # Grip open/close frame analysis
    grip_open_frames = [ep.get("grip_open_frame") for ep in episodes if ep.get("grip_open_frame") is not None]
    grip_close_frames = [ep.get("grip_close_frame") for ep in episodes if ep.get("grip_close_frame") is not None]

    print(f"\n  Grip open timing (frame index where gripper first opens >30 deg):")
    if grip_open_frames:
        # Normalize by episode length
        grip_open_pcts = []
        for ep in episodes:
            gof = ep.get("grip_open_frame")
            if gof is not None:
                pct = 100.0 * gof / ep["num_frames"]
                grip_open_pcts.append(pct)
        print(f"    Mean frame: {np.mean(grip_open_frames):.0f}")
        print(f"    Mean % into episode: {np.mean(grip_open_pcts):.1f}%")
        print(f"    Std % into episode:  {np.std(grip_open_pcts):.1f}%")
        print(f"    Episodes with grip_open: {len(grip_open_frames)}/{len(episodes)}")

    print(f"  Grip close timing:")
    if grip_close_frames:
        grip_close_pcts = []
        for ep in episodes:
            gcf = ep.get("grip_close_frame")
            if gcf is not None:
                pct = 100.0 * gcf / ep["num_frames"]
                grip_close_pcts.append(pct)
        print(f"    Mean frame: {np.mean(grip_close_frames):.0f}")
        print(f"    Mean % into episode: {np.mean(grip_close_pcts):.1f}%")
        print(f"    Episodes with grip_close: {len(grip_close_frames)}/{len(episodes)}")
    else:
        print(f"    No episodes have grip_close_frame set")

    # Check for episodes where gripper never opens significantly
    no_open_eps = [ep["_dir_name"] for ep in episodes if ep.get("gripper_max", 0) < 20]
    if no_open_eps:
        print(f"\n  *** EPISODES WHERE GRIPPER NEVER OPENS (max < 20 deg): {len(no_open_eps)} ***")
        for e in no_open_eps:
            print(f"      {e}")

    # =========================================================================
    # 5. JOINT DIVERSITY
    # =========================================================================
    print(f"\n{'='*80}")
    print("5. JOINT DIVERSITY (frame-level across all episodes)")
    print(f"{'='*80}")

    print(f"  {'Joint':15s} {'Min':>8s} {'Max':>8s} {'Range':>8s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for j in range(6):
        j_data = all_angles_flat[:, j]
        j_min = np.min(j_data)
        j_max = np.max(j_data)
        j_range = j_max - j_min
        j_mean = np.mean(j_data)
        j_std = np.std(j_data)
        print(f"  {JOINT_NAMES[j]:15s} {j_min:8.1f} {j_max:8.1f} {j_range:8.1f} {j_mean:8.1f} {j_std:8.1f}")

    # Action stats for training
    print(f"\n  Action stats (for normalization):")
    action_mean = np.mean(all_angles_flat, axis=0)
    action_std = np.std(all_angles_flat, axis=0)
    print(f"    Mean: [{', '.join(f'{v:.2f}' for v in action_mean)}]")
    print(f"    Std:  [{', '.join(f'{v:.2f}' for v in action_std)}]")

    # =========================================================================
    # 6. EPISODE DURATION
    # =========================================================================
    print(f"\n{'='*80}")
    print("6. EPISODE DURATION")
    print(f"{'='*80}")

    short_eps = [(ep["_dir_name"], ep["num_frames"]) for ep in episodes if ep["num_frames"] < 30]
    long_eps = [(ep["_dir_name"], ep["num_frames"]) for ep in episodes if ep["num_frames"] > 300]
    normal_eps = [ep for ep in episodes if 30 <= ep["num_frames"] <= 300]

    print(f"  Too short (<30 frames): {len(short_eps)}")
    for name, nf in short_eps:
        print(f"    {name}: {nf} frames")

    print(f"  Normal (30-300 frames): {len(normal_eps)}")

    print(f"  Too long (>300 frames): {len(long_eps)}")
    for name, nf in long_eps:
        print(f"    {name}: {nf} frames")

    # Duration histogram (in buckets of 30 frames = 1 second)
    print(f"\n  Duration distribution (seconds):")
    durations_s = [ep["num_frames"] / ep.get("fps", 30) for ep in episodes]
    bins = [0, 3, 4, 5, 6, 7, 8, 10, 15, 30]
    for i in range(len(bins)-1):
        count = sum(1 for d in durations_s if bins[i] <= d < bins[i+1])
        bar = "#" * count
        print(f"    {bins[i]:3d}-{bins[i+1]:3d}s: {count:3d} {bar}")
    count_over = sum(1 for d in durations_s if d >= bins[-1])
    if count_over:
        print(f"    >{bins[-1]:3d}s:  {count_over:3d} {'#' * count_over}")

    # =========================================================================
    # 7. TEMPORAL QUALITY (Static frames)
    # =========================================================================
    print(f"\n{'='*80}")
    print("7. TEMPORAL QUALITY")
    print(f"{'='*80}")

    total_transitions = 0
    total_static = 0
    ep_static_pcts = []

    for ep_id, angles_arr in ep_frame_angles.items():
        if len(angles_arr) < 2:
            continue
        diffs = np.abs(np.diff(angles_arr, axis=0))
        max_diff_per_transition = np.max(diffs, axis=1)
        n_trans = len(max_diff_per_transition)
        n_static = np.sum(max_diff_per_transition < 0.5)
        total_transitions += n_trans
        total_static += n_static
        ep_static_pcts.append(100.0 * n_static / n_trans if n_trans > 0 else 0)

    print(f"  Static frame transitions (max joint change < 0.5 deg):")
    print(f"    Total transitions: {total_transitions}")
    print(f"    Static transitions: {total_static} ({100*total_static/total_transitions:.1f}%)")
    print(f"    Episode-level static %:")
    print(f"      Mean: {np.mean(ep_static_pcts):.1f}%")
    print(f"      Min:  {np.min(ep_static_pcts):.1f}%")
    print(f"      Max:  {np.max(ep_static_pcts):.1f}%")

    high_static_eps = [(episodes[i]["_dir_name"], ep_static_pcts[i])
                       for i in range(len(ep_static_pcts)) if ep_static_pcts[i] > 50]
    if high_static_eps:
        print(f"\n  *** HIGH STATIC EPISODES (>50% static): {len(high_static_eps)} ***")
        for name, pct in high_static_eps:
            print(f"      {name}: {pct:.1f}%")

    # =========================================================================
    # 8. OLD vs NEW COMPARISON
    # =========================================================================
    print(f"\n{'='*80}")
    print("8. OLD vs NEW EPISODE COMPARISON")
    print(f"{'='*80}")

    old_eps = [ep for ep in episodes if ep["episode_id"] <= 30]
    new_eps = [ep for ep in episodes if ep["episode_id"] > 30]

    print(f"  Old episodes (0-30): {len(old_eps)}")
    print(f"  New episodes (31-50): {len(new_eps)}")

    if old_eps and new_eps:
        # Base angle comparison
        old_base_means = []
        new_base_means = []
        for ep in old_eps:
            frames = ep.get("frames", [])
            if frames:
                base_vals = [f["angles"][0] for f in frames]
                old_base_means.append(np.mean(base_vals))
        for ep in new_eps:
            frames = ep.get("frames", [])
            if frames:
                base_vals = [f["angles"][0] for f in frames]
                new_base_means.append(np.mean(base_vals))

        print(f"\n  Base angle (mean per episode):")
        print(f"    OLD: mean={np.mean(old_base_means):.1f}, std={np.std(old_base_means):.1f}, range=[{np.min(old_base_means):.1f}, {np.max(old_base_means):.1f}]")
        print(f"    NEW: mean={np.mean(new_base_means):.1f}, std={np.std(new_base_means):.1f}, range=[{np.min(new_base_means):.1f}, {np.max(new_base_means):.1f}]")

        # Depth comparison
        old_min_z = [ep.get("min_z") for ep in old_eps if ep.get("min_z") is not None]
        new_min_z = [ep.get("min_z") for ep in new_eps if ep.get("min_z") is not None]

        if old_min_z and new_min_z:
            print(f"\n  Min Z (grasp depth):")
            print(f"    OLD: mean={np.mean(old_min_z):.1f}mm, std={np.std(old_min_z):.1f}mm")
            print(f"    NEW: mean={np.mean(new_min_z):.1f}mm, std={np.std(new_min_z):.1f}mm")

        # Gripper comparison
        old_grip_max = [ep.get("gripper_max", 0) for ep in old_eps]
        new_grip_max = [ep.get("gripper_max", 0) for ep in new_eps]

        print(f"\n  Gripper max per episode:")
        print(f"    OLD: mean={np.mean(old_grip_max):.1f}, std={np.std(old_grip_max):.1f}")
        print(f"    NEW: mean={np.mean(new_grip_max):.1f}, std={np.std(new_grip_max):.1f}")

        # Duration comparison
        old_dur = [ep["num_frames"] for ep in old_eps]
        new_dur = [ep["num_frames"] for ep in new_eps]

        print(f"\n  Episode duration (frames):")
        print(f"    OLD: mean={np.mean(old_dur):.1f}, std={np.std(old_dur):.1f}")
        print(f"    NEW: mean={np.mean(new_dur):.1f}, std={np.std(new_dur):.1f}")

        # Zone distribution
        print(f"\n  Zone distribution:")
        old_zone_counts = defaultdict(int)
        new_zone_counts = defaultdict(int)
        for i, bm in enumerate(old_base_means):
            old_zone_counts[classify_position(bm)] += 1
        for i, bm in enumerate(new_base_means):
            new_zone_counts[classify_position(bm)] += 1

        for zone in zone_order:
            oc = old_zone_counts.get(zone, 0)
            nc = new_zone_counts.get(zone, 0)
            print(f"    {zone:12s}: OLD={oc:3d}  NEW={nc:3d}")

    # =========================================================================
    # 9. SHOULDER ANGLE DISTRIBUTION
    # =========================================================================
    print(f"\n{'='*80}")
    print("9. SHOULDER ANGLE DISTRIBUTION")
    print(f"{'='*80}")

    shoulder_all = all_angles_flat[:, 1]
    max_sh_list = [ep.get("max_shoulder", None) for ep in episodes]
    max_sh_list = [s for s in max_sh_list if s is not None]

    print(f"  Frame-level shoulder angle:")
    print(f"    Min:  {np.min(shoulder_all):.1f} deg")
    print(f"    Max:  {np.max(shoulder_all):.1f} deg")
    print(f"    Mean: {np.mean(shoulder_all):.1f} deg")
    print(f"    Std:  {np.std(shoulder_all):.1f} deg")

    if max_sh_list:
        print(f"\n  Episode-level max shoulder:")
        print(f"    Mean: {np.mean(max_sh_list):.1f} deg")
        print(f"    Min:  {np.min(max_sh_list):.1f} deg")
        print(f"    Max:  {np.max(max_sh_list):.1f} deg")
        sh_deep = sum(1 for s in max_sh_list if s > 60)
        sh_approach = sum(1 for s in max_sh_list if 40 <= s <= 60)
        sh_shallow = sum(1 for s in max_sh_list if s < 40)
        print(f"\n  Shoulder depth classification (max shoulder per episode):")
        print(f"    DEEP (>60 deg):     {sh_deep:3d} episodes ({100*sh_deep/len(max_sh_list):.1f}%)")
        print(f"    APPROACH (40-60):   {sh_approach:3d} episodes ({100*sh_approach/len(max_sh_list):.1f}%)")
        print(f"    SHALLOW (<40):      {sh_shallow:3d} episodes ({100*sh_shallow/len(max_sh_list):.1f}%)")

    # Shoulder distribution histogram
    print(f"\n  Shoulder angle histogram (frame-level):")
    sh_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110]
    for i in range(len(sh_bins)-1):
        count = np.sum((shoulder_all >= sh_bins[i]) & (shoulder_all < sh_bins[i+1]))
        pct = 100 * count / len(shoulder_all)
        bar = "#" * int(pct)
        print(f"    {sh_bins[i]:3d}-{sh_bins[i+1]:3d} deg: {count:6d} ({pct:5.1f}%) {bar}")

    # =========================================================================
    # 10. PER-EPISODE DETAIL TABLE
    # =========================================================================
    print(f"\n{'='*80}")
    print("10. PER-EPISODE SUMMARY TABLE")
    print(f"{'='*80}")

    header = f"  {'Ep':>4s} {'Frames':>6s} {'Dur(s)':>6s} {'BaseMn':>7s} {'ShMax':>6s} {'MinZ':>7s} {'GrpMax':>7s} {'GrpRng':>7s} {'Zone':>10s} {'Depth':>8s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        nf = ep["num_frames"]
        dur = nf / ep.get("fps", 30)

        # Base mean
        frames = ep.get("frames", [])
        base_mn = np.mean([f["angles"][0] for f in frames]) if frames else 0

        sh_max = ep.get("max_shoulder", 0)
        min_z = ep.get("min_z", None)
        gr_max = ep.get("gripper_max", 0)
        gr_rng = ep.get("gripper_range", 0)

        zone = classify_position(base_mn)

        if min_z is not None:
            if min_z < Z_DEEP:
                depth = "DEEP"
            elif min_z < Z_APPROACH:
                depth = "APPROACH"
            else:
                depth = "SHALLOW"
            min_z_str = f"{min_z:7.1f}"
        else:
            depth = "?"
            min_z_str = "    N/A"

        print(f"  {ep_id:4d} {nf:6d} {dur:6.1f} {base_mn:7.1f} {sh_max:6.1f} {min_z_str} {gr_max:7.1f} {gr_rng:7.1f} {zone:>10s} {depth:>8s}")

    # =========================================================================
    # 11. TRAINING READINESS ASSESSMENT
    # =========================================================================
    print(f"\n{'='*80}")
    print("11. TRAINING READINESS ASSESSMENT")
    print(f"{'='*80}")

    score = 0
    max_score = 10
    issues = []
    strengths = []

    # Check 1: Episode count (target 100+)
    if len(episodes) >= 100:
        score += 2
        strengths.append(f"Episode count: {len(episodes)} (target: 100+)")
    elif len(episodes) >= 50:
        score += 1
        issues.append(f"Episode count: {len(episodes)} (target: 100+, minimum 50 met)")
    else:
        issues.append(f"Episode count: {len(episodes)} (target: 100+, BELOW MINIMUM)")

    # Check 2: All DEEP grasps
    deep_pct = 100 * len(depth_classes["DEEP"]) / len(episodes) if episodes else 0
    if deep_pct >= 80:
        score += 2
        strengths.append(f"Deep grasp ratio: {deep_pct:.0f}% ({len(depth_classes['DEEP'])}/{len(episodes)})")
    elif deep_pct >= 50:
        score += 1
        issues.append(f"Deep grasp ratio only {deep_pct:.0f}%, want 80%+")
    else:
        issues.append(f"Deep grasp ratio very low: {deep_pct:.0f}%")

    # Check 3: Position diversity
    filled_zones = sum(1 for z in zone_order if len(ep_zones.get(z, [])) > 0)
    zone_with_3plus = sum(1 for z in zone_order if len(ep_zones.get(z, [])) >= 3)
    if filled_zones >= 5 and zone_with_3plus >= 4:
        score += 2
        strengths.append(f"Position diversity: {filled_zones}/5 zones filled, {zone_with_3plus} with 3+ episodes")
    elif filled_zones >= 3:
        score += 1
        issues.append(f"Position diversity: only {filled_zones}/5 zones filled")
    else:
        issues.append(f"Position diversity critically low: only {filled_zones}/5 zones")

    # Check 4: Gripper open ratio
    grip_open_pct = 100 * frames_open / len(gripper_all) if len(gripper_all) > 0 else 0
    if grip_open_pct >= 30:
        score += 1
        strengths.append(f"Gripper open ratio: {grip_open_pct:.1f}%")
    else:
        issues.append(f"Gripper open ratio low: {grip_open_pct:.1f}% (want 30%+)")

    # Check 5: Static frame ratio
    static_pct = 100 * total_static / total_transitions if total_transitions > 0 else 0
    if static_pct < 25:
        score += 1
        strengths.append(f"Static frame ratio: {static_pct:.1f}% (good, <25%)")
    elif static_pct < 40:
        issues.append(f"Static frame ratio: {static_pct:.1f}% (moderate, want <25%)")
    else:
        issues.append(f"Static frame ratio high: {static_pct:.1f}% (want <25%)")

    # Check 6: Episode duration consistency
    dur_cv = np.std(num_frames_list) / np.mean(num_frames_list)
    if dur_cv < 0.3 and len(short_eps) == 0 and len(long_eps) == 0:
        score += 1
        strengths.append(f"Duration consistency: CV={dur_cv:.2f}, no outliers")
    elif len(short_eps) == 0:
        score += 0.5
        issues.append(f"Duration has some variation: CV={dur_cv:.2f}")
    else:
        issues.append(f"Duration issues: {len(short_eps)} too short, {len(long_eps)} too long")

    # Check 7: Base angle std
    base_std = np.std(ep_base_means) if ep_base_means else 0
    if base_std >= 15:
        score += 1
        strengths.append(f"Base angle std: {base_std:.1f} deg (good diversity)")
    elif base_std >= 10:
        score += 0.5
        issues.append(f"Base angle std: {base_std:.1f} deg (moderate diversity)")
    else:
        issues.append(f"Base angle std very low: {base_std:.1f} deg")

    print(f"\n  SCORE: {score:.1f} / {max_score}")
    print(f"\n  STRENGTHS:")
    for s in strengths:
        print(f"    [+] {s}")
    print(f"\n  ISSUES:")
    for iss in issues:
        print(f"    [-] {iss}")

    # Overall verdict
    print(f"\n  VERDICT:", end=" ")
    if score >= 8:
        print("READY FOR TRAINING")
    elif score >= 6:
        print("PARTIALLY READY - address issues before training")
    elif score >= 4:
        print("NEEDS MORE WORK - significant gaps remain")
    else:
        print("NOT READY - major data quality issues")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if len(episodes) < 100:
        need = 100 - len(episodes)
        print(f"\n  1. COLLECT {need} MORE EPISODES to reach 100+ target")

    if missing_zones:
        print(f"\n  2. FILL MISSING POSITION ZONES: {', '.join(missing_zones)}")
        for zone in missing_zones:
            lo, hi = POSITION_ZONES[zone]
            print(f"     {zone}: place sponge so base angle is {lo} to {hi} deg")

    weak_zones = [z for z in zone_order if 0 < len(ep_zones.get(z, [])) < 5]
    if weak_zones:
        print(f"\n  3. STRENGTHEN WEAK ZONES (< 5 episodes):")
        for z in weak_zones:
            print(f"     {z}: {len(ep_zones[z])} episodes, need {5 - len(ep_zones[z])} more")

    if static_pct > 30:
        print(f"\n  4. REDUCE STATIC FRAMES: Consider frame deduplication or faster demonstrations")

    if grip_open_pct < 30:
        print(f"\n  5. IMPROVE GRIPPER COVERAGE: Hold gripper open longer during approach phase")

    print(f"\n{'='*80}")
    print("END OF ANALYSIS")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
