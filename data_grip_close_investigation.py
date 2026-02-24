#!/usr/bin/env python3
"""
Investigate grip_close_frame: why only 19/51 episodes have it set.
Also check gripper trajectory patterns more closely.
"""

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

def main():
    ep_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    print("GRIP CLOSE INVESTIGATION")
    print("=" * 80)

    has_close = []
    no_close = []

    for ep_dir in ep_dirs:
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        ep_id = meta["episode_id"]
        frames = meta.get("frames", [])
        gripper_vals = np.array([f["angles"][5] for f in frames])
        gc = meta.get("grip_close_frame")
        go = meta.get("grip_open_frame")

        # Check gripper trajectory shape
        # Find where gripper is at its max
        max_idx = np.argmax(gripper_vals)
        max_val = gripper_vals[max_idx]

        # Check if gripper closes after opening (returns below 15 deg after being above 30)
        opened = False
        close_idx = None
        for i, g in enumerate(gripper_vals):
            if g > 30:
                opened = True
            if opened and g < 15:
                close_idx = i
                break

        # Last N frames gripper value
        last_5 = gripper_vals[-5:]
        first_5 = gripper_vals[:5]

        # Classify trajectory
        if not opened:
            traj_type = "NEVER_OPENS"
        elif close_idx is not None:
            traj_type = "OPEN_THEN_CLOSE"
        else:
            traj_type = "OPENS_STAYS_OPEN"

        info = {
            "ep_id": ep_id,
            "name": ep_dir.name,
            "gc_meta": gc,
            "go_meta": go,
            "detected_close": close_idx,
            "max_grip_idx": max_idx,
            "max_grip_val": max_val,
            "end_grip": gripper_vals[-1],
            "start_grip": gripper_vals[0],
            "traj_type": traj_type,
            "n_frames": len(gripper_vals),
        }

        if gc is not None:
            has_close.append(info)
        else:
            no_close.append(info)

    print(f"\nEpisodes WITH grip_close_frame: {len(has_close)}")
    print(f"Episodes WITHOUT grip_close_frame: {len(no_close)}")

    print(f"\n--- Episodes WITH grip_close_frame ---")
    print(f"  {'Ep':>4s} {'GripOpen':>8s} {'GripClose':>9s} {'DetClose':>8s} {'MaxGrip':>7s} {'EndGrip':>7s} {'Traj':>18s}")
    for info in has_close:
        print(f"  {info['ep_id']:4d} {info['go_meta'] or 'N/A':>8} {info['gc_meta'] or 'N/A':>9} {str(info['detected_close']) or 'N/A':>8} {info['max_grip_val']:7.1f} {info['end_grip']:7.1f} {info['traj_type']:>18s}")

    print(f"\n--- Episodes WITHOUT grip_close_frame ---")
    print(f"  {'Ep':>4s} {'GripOpen':>8s} {'DetClose':>8s} {'MaxGrip':>7s} {'EndGrip':>7s} {'Traj':>18s}")
    for info in no_close:
        print(f"  {info['ep_id']:4d} {info['go_meta'] or 'N/A':>8} {str(info['detected_close']) or 'N/A':>8} {info['max_grip_val']:7.1f} {info['end_grip']:7.1f} {info['traj_type']:>18s}")

    # Summary of trajectory types
    all_info = has_close + no_close
    types = {}
    for info in all_info:
        t = info['traj_type']
        types[t] = types.get(t, 0) + 1

    print(f"\n--- Trajectory Type Summary ---")
    for t, c in sorted(types.items()):
        print(f"  {t}: {c} episodes ({100*c/len(all_info):.1f}%)")

    # Check: does OPENS_STAYS_OPEN mean the episode was cut before gripper closes?
    stays_open = [info for info in all_info if info['traj_type'] == 'OPENS_STAYS_OPEN']
    if stays_open:
        print(f"\n--- OPENS_STAYS_OPEN episodes (gripper never closes after opening) ---")
        end_grips = [info['end_grip'] for info in stays_open]
        print(f"  End gripper values: min={np.min(end_grips):.1f}, max={np.max(end_grips):.1f}, mean={np.mean(end_grips):.1f}")
        print(f"  These episodes end with gripper OPEN -- episode may be cut before grasp completes")


if __name__ == "__main__":
    main()
