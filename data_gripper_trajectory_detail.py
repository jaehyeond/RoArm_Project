#!/usr/bin/env python3
"""
Detailed gripper trajectory analysis for a few representative episodes.
Check if the gripper does a proper open->close cycle for sponge grasping.
"""

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

def analyze_episode_gripper(ep_dir):
    """Print detailed gripper trajectory for one episode."""
    meta_path = ep_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    frames = meta["frames"]
    gripper = np.array([f["angles"][5] for f in frames])
    shoulder = np.array([f["angles"][1] for f in frames])
    poses = np.array([f["pose"] for f in frames]) if frames[0].get("pose") else None

    n = len(gripper)
    ep_id = meta["episode_id"]

    # Sample at 10% intervals
    print(f"\n--- Episode {ep_id} ({n} frames, {n/30:.1f}s) ---")
    print(f"  {'Pct':>4s} {'Frame':>5s} {'Gripper':>8s} {'Shoulder':>9s} {'Z(mm)':>7s}")

    for pct in range(0, 101, 5):
        idx = min(int(pct / 100 * n), n - 1)
        z_str = f"{poses[idx][2]:.1f}" if poses is not None else "N/A"
        print(f"  {pct:3d}% {idx:5d} {gripper[idx]:8.1f} {shoulder[idx]:9.1f} {z_str:>7s}")

    # Key moments
    max_grip_idx = np.argmax(gripper)
    print(f"\n  Gripper max: {gripper[max_grip_idx]:.1f} deg at frame {max_grip_idx} ({100*max_grip_idx/n:.0f}%)")
    print(f"  Gripper start: {gripper[0]:.1f} deg")
    print(f"  Gripper end:   {gripper[-1]:.1f} deg")

    # Check: does gripper drop below 20 after opening?
    after_max = gripper[max_grip_idx:]
    min_after_max = np.min(after_max)
    min_after_max_idx = max_grip_idx + np.argmin(after_max)
    print(f"  Min gripper after peak: {min_after_max:.1f} deg at frame {min_after_max_idx} ({100*min_after_max_idx/n:.0f}%)")

    # Z at key moments
    if poses is not None:
        z_at_max_grip = poses[max_grip_idx][2]
        z_min_idx = np.argmin(poses[:, 2])
        z_at_min = poses[z_min_idx][2]
        print(f"  Z at gripper max: {z_at_max_grip:.1f} mm")
        print(f"  Z minimum: {z_at_min:.1f} mm at frame {z_min_idx} ({100*z_min_idx/n:.0f}%)")

    return gripper


def main():
    # Analyze a sample of episodes
    sample_ids = [0, 5, 12, 20, 31, 36, 40, 45, 50]

    print("DETAILED GRIPPER TRAJECTORY ANALYSIS")
    print("=" * 80)
    print("\nChecking if gripper does proper open->close for sponge grasping")
    print("Sponge is soft -- full close may not be needed, but should go below ~20 deg")

    for ep_id in sample_ids:
        ep_dir = DATA_DIR / f"episode_{ep_id:04d}"
        if ep_dir.exists() and (ep_dir / "metadata.json").exists():
            analyze_episode_gripper(ep_dir)

    # Overall: check what the "resting" gripper position is
    print("\n\n" + "=" * 80)
    print("GRIPPER RESTING POSITION ANALYSIS")
    print("=" * 80)

    all_end_grips = []
    all_start_grips = []
    all_min_after_peak = []

    for ep_dir in sorted(DATA_DIR.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        gripper = np.array([f["angles"][5] for f in meta["frames"]])
        all_start_grips.append(gripper[0])
        all_end_grips.append(gripper[-1])

        max_idx = np.argmax(gripper)
        after_max = gripper[max_idx:]
        all_min_after_peak.append(np.min(after_max))

    print(f"\n  Start gripper: mean={np.mean(all_start_grips):.1f}, std={np.std(all_start_grips):.1f}")
    print(f"  End gripper:   mean={np.mean(all_end_grips):.1f}, std={np.std(all_end_grips):.1f}")
    print(f"  Min after peak: mean={np.mean(all_min_after_peak):.1f}, std={np.std(all_min_after_peak):.1f}")

    # Histogram of min-after-peak
    print(f"\n  Distribution of min gripper after peak (proxy for grip tightness):")
    bins = [0, 5, 10, 15, 20, 25, 30, 40]
    for i in range(len(bins)-1):
        count = sum(1 for v in all_min_after_peak if bins[i] <= v < bins[i+1])
        print(f"    {bins[i]:3d}-{bins[i+1]:3d} deg: {count:3d} {'#'*count}")
    count_over = sum(1 for v in all_min_after_peak if v >= 40)
    if count_over:
        print(f"    >{bins[-1]} deg:  {count_over:3d} {'#'*count_over}")


if __name__ == "__main__":
    main()
