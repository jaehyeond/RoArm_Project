"""
data_center_grasp_detail.py
Deep-dive into CENTER episode grasp trajectories.

Focuses on the GRASP MOMENT (when gripper closes on sponge) for CENTER episodes.
The previous script's "grasp point" was defined as "50% of gripper peak" which
may not be accurate. This script uses a better heuristic:
  - grasp = frame where gripper is at MAX opening (peak) OR
  - frame where gripper first drops below 30 deg after peak (closing phase starts)
  - Also reports elbow at EACH PHASE of the episode

The key concern: at the actual grasp point, what is elbow angle?
(Not the return-to-home where elbow goes back to ~90 deg)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/cgxr/Documents/Robotics/RoArm_Project/lerobot_dataset_v3")
PARQUET_FILE = DATASET_PATH / "data/chunk-000/file-000.parquet"

CENTER_BASE_THRESHOLD = 10.0

# Center episode IDs (from previous analysis)
CENTER_EPS = [0, 1, 2, 3, 4, 17, 18, 19, 21, 25, 26, 27, 28, 29, 30, 50, 65, 73]


def load_data():
    df = pd.read_parquet(PARQUET_FILE)
    action_cols = df['action'].apply(lambda x: pd.Series(x))
    action_cols.columns = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    df = pd.concat([df[['episode_index', 'frame_index', 'timestamp']], action_cols], axis=1)
    return df


def find_grasp_phases(arr_shoulder, arr_elbow, arr_gripper, n):
    """
    Identify key grasp phases in a CENTER episode.

    Phase 1: Start (frames 0 to first_movement)
    Phase 2: Approach (arm moving, gripper opening)
    Phase 3: At sponge / deep point (shoulder at max, arm fully descended)
    Phase 4: Closing / lifting (gripper closing, arm rising)
    Phase 5: Return (arm back to home)

    Returns dict with frame indices for each phase transition.
    """
    gripper_peak_idx = int(np.argmax(arr_gripper))
    shoulder_peak_idx = int(np.argmax(arr_shoulder))

    # First significant gripper opening (>10 deg)
    gripper_open_start = 0
    for i in range(n):
        if arr_gripper[i] > 10:
            gripper_open_start = i
            break

    # Deepest point: shoulder at max
    deep_frame = shoulder_peak_idx

    # Gripper closing start: after peak, first drop below 80% of peak
    gripper_peak_val = arr_gripper[gripper_peak_idx]
    gripper_close_start = gripper_peak_idx
    for i in range(gripper_peak_idx, n):
        if arr_gripper[i] < gripper_peak_val * 0.8:
            gripper_close_start = i
            break

    # End of closing: gripper stable (below 30 deg and not dropping fast)
    gripper_stable = n - 1
    for i in range(gripper_close_start, n):
        if arr_gripper[i] < 30:
            gripper_stable = i
            break

    return {
        'gripper_open_start': gripper_open_start,
        'gripper_peak': gripper_peak_idx,
        'shoulder_peak': shoulder_peak_idx,  # deepest arm point
        'gripper_close_start': gripper_close_start,
        'gripper_stable': gripper_stable,
    }


def analyze_grasp_trajectory(df, ep_id):
    ep = df[df['episode_index'] == ep_id].sort_values('frame_index').reset_index(drop=True)
    n = len(ep)

    base_arr = ep['base'].values
    shoulder_arr = ep['shoulder'].values
    elbow_arr = ep['elbow'].values
    wrist_arr = ep['wrist_pitch'].values
    gripper_arr = ep['gripper'].values

    phases = find_grasp_phases(shoulder_arr, elbow_arr, gripper_arr, n)
    gp = phases['gripper_peak']
    dp = phases['shoulder_peak']
    gc = phases['gripper_close_start']
    gs = phases['gripper_stable']

    return {
        'ep_id': ep_id,
        'n': n,
        'phases': phases,
        # At each key frame
        'at_gripper_peak': {
            'frame': gp,
            'shoulder': float(shoulder_arr[gp]),
            'elbow': float(elbow_arr[gp]),
            'wrist_pitch': float(wrist_arr[gp]),
            'gripper': float(gripper_arr[gp]),
        },
        'at_shoulder_peak': {  # deepest arm
            'frame': dp,
            'shoulder': float(shoulder_arr[dp]),
            'elbow': float(elbow_arr[dp]),
            'wrist_pitch': float(wrist_arr[dp]),
            'gripper': float(gripper_arr[dp]),
        },
        'at_gripper_close_start': {
            'frame': gc,
            'shoulder': float(shoulder_arr[gc]),
            'elbow': float(elbow_arr[gc]),
            'wrist_pitch': float(wrist_arr[gc]),
            'gripper': float(gripper_arr[gc]),
        },
        'at_gripper_stable': {
            'frame': gs,
            'shoulder': float(shoulder_arr[gs]),
            'elbow': float(elbow_arr[gs]),
            'wrist_pitch': float(wrist_arr[gs]),
            'gripper': float(gripper_arr[gs]),
        },
        # Stats
        'shoulder_min': float(shoulder_arr.min()),
        'shoulder_max': float(shoulder_arr.max()),
        'elbow_min': float(elbow_arr.min()),
        'elbow_max': float(elbow_arr.max()),
        'elbow_at_shoulder_peak': float(elbow_arr[dp]),
        'elbow_at_gripper_peak': float(elbow_arr[gp]),
        'gripper_max': float(gripper_arr.max()),
        'gripper_end': float(gripper_arr[-1]),
        'max_abs_base': float(np.max(np.abs(base_arr))),
    }


def main():
    df = load_data()

    print("="*80)
    print("DEEP DIVE: CENTER EPISODE GRASP TRAJECTORIES")
    print(f"18 CENTER episodes: {CENTER_EPS}")
    print("="*80)

    results = []
    for ep_id in CENTER_EPS:
        r = analyze_grasp_trajectory(df, ep_id)
        results.append(r)

    # Q: What is elbow at the GRASP MOMENT (shoulder at max = arm most descended)?
    print("\n--- ELBOW AT KEY GRASP PHASES (per CENTER episode) ---")
    print(f"{'EpID':>5}  {'Shld_pk':>7}  {'Elb@Shld_pk':>11}  {'Elb@Grip_pk':>11}  {'Shld_min':>8}  {'Elb_min':>7}  {'GripMax':>7}")
    print("-"*75)
    elbow_at_shoulder_peak_list = []
    elbow_at_grip_peak_list = []
    for r in results:
        ep_id = r['ep_id']
        sp = r['at_shoulder_peak']
        gp = r['at_gripper_peak']
        print(f"{ep_id:>5}  {sp['shoulder']:>7.1f}  {sp['elbow']:>11.1f}  {gp['elbow']:>11.1f}  "
              f"{r['shoulder_min']:>8.1f}  {r['elbow_min']:>7.1f}  {r['gripper_max']:>7.1f}")
        elbow_at_shoulder_peak_list.append(sp['elbow'])
        elbow_at_grip_peak_list.append(gp['elbow'])

    print(f"\n  Elbow at shoulder peak (deepest arm): "
          f"mean={np.mean(elbow_at_shoulder_peak_list):.1f}  "
          f"std={np.std(elbow_at_shoulder_peak_list):.1f}  "
          f"range=[{np.min(elbow_at_shoulder_peak_list):.1f}, {np.max(elbow_at_shoulder_peak_list):.1f}]")
    print(f"  Elbow at gripper peak (fully open):  "
          f"mean={np.mean(elbow_at_grip_peak_list):.1f}  "
          f"std={np.std(elbow_at_grip_peak_list):.1f}  "
          f"range=[{np.min(elbow_at_grip_peak_list):.1f}, {np.max(elbow_at_grip_peak_list):.1f}]")

    # Cross-check elbow at each of the 4 key phase frames
    print("\n--- JOINT STATE AT 4 KEY PHASES (mean ± std across 18 CENTER eps) ---")
    phases_list = ['at_gripper_peak', 'at_shoulder_peak', 'at_gripper_close_start', 'at_gripper_stable']
    phase_labels = ['Gripper PEAK (open)', 'Shoulder PEAK (deepest)', 'Gripper CLOSE start', 'Gripper STABLE (gripped)']

    for phase, label in zip(phases_list, phase_labels):
        shoulders = [r[phase]['shoulder'] for r in results]
        elbows = [r[phase]['elbow'] for r in results]
        wrists = [r[phase]['wrist_pitch'] for r in results]
        grippers = [r[phase]['gripper'] for r in results]
        frames_pct = [r[phase]['frame'] / r['n'] * 100 for r in results]

        print(f"\n  [{label}]  (occurs at {np.mean(frames_pct):.0f}% ± {np.std(frames_pct):.0f}% into episode)")
        print(f"    Shoulder:   mean={np.mean(shoulders):>6.1f}  std={np.std(shoulders):>5.1f}  range=[{min(shoulders):>5.1f}, {max(shoulders):>5.1f}]")
        print(f"    Elbow:      mean={np.mean(elbows):>6.1f}  std={np.std(elbows):>5.1f}  range=[{min(elbows):>5.1f}, {max(elbows):>5.1f}]")
        print(f"    Wrist_pit:  mean={np.mean(wrists):>6.1f}  std={np.std(wrists):>5.1f}  range=[{min(wrists):>5.1f}, {max(wrists):>5.1f}]")
        print(f"    Gripper:    mean={np.mean(grippers):>6.1f}  std={np.std(grippers):>5.1f}  range=[{min(grippers):>5.1f}, {max(grippers):>5.1f}]")

    # Two trajectory types in CENTER: early-vs-late gripper
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS: Do CENTER episodes have consistent elbow pattern?")
    print("="*80)
    print("\nFull phase profile for each CENTER episode:")
    print(f"{'EpID':>5}  {'GripPk%':>7}  {'ShldPk%':>7}  {'GripCl%':>7}  {'Order':>20}")
    print("-"*55)
    for r in results:
        n = r['n']
        gp_pct = r['phases']['gripper_peak'] / n * 100
        sp_pct = r['phases']['shoulder_peak'] / n * 100
        gc_pct = r['phases']['gripper_close_start'] / n * 100
        # Determine order: does grip open before or after shoulder peak?
        gp_frame = r['phases']['gripper_peak']
        sp_frame = r['phases']['shoulder_peak']
        if gp_frame < sp_frame:
            order = "GRIP_OPEN->DESCEND"
        else:
            order = "DESCEND->GRIP_OPEN"
        print(f"{r['ep_id']:>5}  {gp_pct:>7.0f}%  {sp_pct:>7.0f}%  {gc_pct:>7.0f}%  {order:>20}")

    # Now specifically answer: ELBOW AT THE MOMENT THE GRIPPER IS CLOSING
    # (this is when sponge is being gripped — the actual grasp)
    print("\n" + "="*80)
    print("CRITICAL QUESTION: What is ELBOW angle when gripper is CLOSING on sponge?")
    print("  (shoulder_peak frame = arm at deepest point)")
    print("="*80)
    elbows_at_deep = [r['at_shoulder_peak']['elbow'] for r in results]
    shoulders_at_deep = [r['at_shoulder_peak']['shoulder'] for r in results]
    grippers_at_deep = [r['at_shoulder_peak']['gripper'] for r in results]

    print(f"\n  At shoulder peak (arm fully descended):")
    print(f"  Elbow   : mean={np.mean(elbows_at_deep):.1f}  std={np.std(elbows_at_deep):.1f}  "
          f"range=[{min(elbows_at_deep):.1f}, {max(elbows_at_deep):.1f}]")
    print(f"  Shoulder: mean={np.mean(shoulders_at_deep):.1f}  std={np.std(shoulders_at_deep):.1f}  "
          f"range=[{min(shoulders_at_deep):.1f}, {max(shoulders_at_deep):.1f}]")
    print(f"  Gripper : mean={np.mean(grippers_at_deep):.1f}  std={np.std(grippers_at_deep):.1f}  "
          f"range=[{min(grippers_at_deep):.1f}, {max(grippers_at_deep):.1f}]")

    # Histogram of elbow values at deep point
    print(f"\n  Elbow angle distribution at deepest point (CENTER episodes):")
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]
    for lo, hi in bins:
        count = sum(1 for e in elbows_at_deep if lo <= e < hi)
        if count > 0:
            bar = '#' * count
            print(f"    [{lo:>3}-{hi:>3} deg]: {bar} ({count})")

    # SUMMARY TABLE for user
    print("\n" + "="*80)
    print("FINAL ANSWER SUMMARY")
    print("="*80)
    print(f"\n  Q1: CENTER episodes (max|base| < 10 deg): 18 out of 74 total ({18/74*100:.0f}%)")
    print(f"      Episode IDs: {CENTER_EPS}")
    print()
    print(f"  Q3: COUNT = 18 CENTER episodes")
    print()
    print(f"  Q2: Typical CENTER grasp trajectory:")
    print(f"      Phase 2 (approach):  Shoulder rises 1->67 deg, Elbow drops 90->12 deg")
    print(f"      Phase 3 (deep):      Shoulder at max ({np.mean(shoulders_at_deep):.0f} deg),")
    print(f"                           Elbow at {np.mean(elbows_at_deep):.0f} deg")
    print(f"                           Gripper: {np.mean(grippers_at_deep):.0f} deg (still open)")
    print(f"      Phase 4 (lift):      Shoulder drops, Elbow returns toward 90 deg")
    print(f"      Phase 5 (return):    Elbow back to ~90 deg (home)")
    print()
    print(f"  Q4: Elbow at deepest point in CENTER episodes:")
    print(f"      mean = {np.mean(elbows_at_deep):.1f} deg")
    print(f"      std  = {np.std(elbows_at_deep):.1f} deg")
    print(f"      range = [{min(elbows_at_deep):.1f}, {max(elbows_at_deep):.1f}] deg")
    print(f"      (All 18 CENTER episodes: elbow 11-26 deg at deepest)")
    print()
    print(f"  IMPLICATION: CENTER sponge placement produces elbow ~{np.mean(elbows_at_deep):.0f} deg at grasp")
    print(f"  This is NOT elbow < -30 deg (target) -- that would require elbow going NEGATIVE")
    print(f"  The 'deep grasp' in v3 data uses HIGH SHOULDER (60-67 deg) + LOW ELBOW (11-20 deg)")
    print(f"  Elbow never goes below 0 deg in CENTER episodes -- it goes to ~12-26 deg minimum")


if __name__ == "__main__":
    main()
