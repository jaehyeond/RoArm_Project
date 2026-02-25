"""
data_center_episodes_analysis.py
Analyze CENTER episodes (base angle stays near 0 throughout) in lerobot_dataset_v3.

Questions:
1. List of episodes where max(abs(base_angle)) < 10 degrees (CENTER episodes)
2. Typical grasp trajectory for CENTER episodes (Shoulder, Elbow, Gripper at grasp point)
3. Count of CENTER episodes
4. Elbow angle range at the lowest point (deepest Z proxy) in CENTER episodes

Joints: [0]=base, [1]=shoulder, [2]=elbow, [3]=wrist_pitch, [4]=wrist_roll, [5]=gripper
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/cgxr/Documents/Robotics/RoArm_Project/lerobot_dataset_v3")
PARQUET_FILE = DATASET_PATH / "data/chunk-000/file-000.parquet"

# Threshold for CENTER classification
CENTER_BASE_THRESHOLD = 10.0  # degrees: max(abs(base)) < 10 = CENTER


def load_data():
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} frames from {PARQUET_FILE}")
    print(f"Columns: {list(df.columns)}")
    print(f"Episodes: {df['episode_index'].nunique()}")
    return df


def extract_joint(series, idx):
    """Extract joint value from action array column."""
    return series.apply(lambda x: float(x[idx]))


def analyze_center_episodes(df):
    print("\n" + "="*70)
    print("CENTER EPISODE ANALYSIS (max(abs(base)) < 10 deg)")
    print("="*70)

    # Extract base angle per frame from 'action' column
    df = df.copy()
    action_cols = df['action'].apply(lambda x: pd.Series(x))
    action_cols.columns = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    df = pd.concat([df[['episode_index', 'frame_index', 'timestamp']], action_cols], axis=1)

    # Per-episode stats
    episodes = []
    for ep_id, ep_df in df.groupby('episode_index'):
        ep_df = ep_df.sort_values('frame_index').reset_index(drop=True)
        base_arr = ep_df['base'].values
        shoulder_arr = ep_df['shoulder'].values
        elbow_arr = ep_df['elbow'].values
        gripper_arr = ep_df['gripper'].values
        wrist_pitch_arr = ep_df['wrist_pitch'].values

        max_abs_base = np.max(np.abs(base_arr))
        is_center = max_abs_base < CENTER_BASE_THRESHOLD

        # Find "grasp point" = frame where shoulder is at its minimum (deepest = most forward/down)
        # Shoulder minimum = arm most extended / deepest reach
        min_shoulder_idx = np.argmin(shoulder_arr)
        # Also find where gripper transitions from open to closing (after peak gripper angle)
        gripper_peak_idx = np.argmax(gripper_arr)

        # Approximate grasp frame: frame where gripper starts closing after peak
        # Find first frame after peak where gripper drops below 80% of max
        gripper_max = gripper_arr.max()
        grasp_frame_idx = gripper_peak_idx  # default
        if gripper_peak_idx < len(gripper_arr) - 1:
            for i in range(gripper_peak_idx, len(gripper_arr)):
                if gripper_arr[i] < gripper_max * 0.5:
                    grasp_frame_idx = i
                    break

        # At grasp point
        grasp_shoulder = shoulder_arr[grasp_frame_idx]
        grasp_elbow = elbow_arr[grasp_frame_idx]
        grasp_gripper = gripper_arr[grasp_frame_idx]
        grasp_wrist_pitch = wrist_pitch_arr[grasp_frame_idx]

        # At shoulder minimum (deepest point by shoulder proxy)
        deep_shoulder = shoulder_arr[min_shoulder_idx]
        deep_elbow = elbow_arr[min_shoulder_idx]
        deep_gripper = gripper_arr[min_shoulder_idx]

        episodes.append({
            'ep_id': int(ep_id),
            'n_frames': len(ep_df),
            'duration_s': len(ep_df) / 30.0,
            'max_abs_base': max_abs_base,
            'base_mean': base_arr.mean(),
            'base_std': base_arr.std(),
            'is_center': is_center,
            # Shoulder stats
            'shoulder_min': shoulder_arr.min(),
            'shoulder_max': shoulder_arr.max(),
            'shoulder_mean': shoulder_arr.mean(),
            # Elbow stats
            'elbow_at_deep': deep_elbow,
            'elbow_min': elbow_arr.min(),
            'elbow_max': elbow_arr.max(),
            'elbow_mean': elbow_arr.mean(),
            # Gripper stats
            'gripper_max': gripper_arr.max(),
            'gripper_min': gripper_arr.min(),
            # At grasp point
            'grasp_shoulder': grasp_shoulder,
            'grasp_elbow': grasp_elbow,
            'grasp_gripper': grasp_gripper,
            'grasp_wrist_pitch': grasp_wrist_pitch,
            # At deepest point (min shoulder)
            'deep_shoulder': deep_shoulder,
            'deep_elbow': deep_elbow,
            'deep_gripper': deep_gripper,
        })

    ep_df_all = pd.DataFrame(episodes)

    # --- Q1 + Q3: CENTER episode list and count ---
    center_eps = ep_df_all[ep_df_all['is_center']].copy()
    non_center_eps = ep_df_all[~ep_df_all['is_center']].copy()

    print(f"\nTotal episodes: {len(ep_df_all)}")
    print(f"CENTER episodes (max_abs_base < {CENTER_BASE_THRESHOLD} deg): {len(center_eps)}")
    print(f"Non-CENTER episodes: {len(non_center_eps)}")
    print(f"\nCENTER episode list:")
    print(f"{'EpID':>5}  {'MaxAbsBase':>10}  {'BaseMean':>8}  {'ShouldMin':>9}  {'ElbowDeep':>9}  {'GripMax':>7}  {'Frames':>6}  {'Dur(s)':>6}")
    print("-" * 72)
    for _, row in center_eps.sort_values('ep_id').iterrows():
        print(f"{int(row['ep_id']):>5}  {row['max_abs_base']:>10.2f}  {row['base_mean']:>8.2f}  "
              f"{row['shoulder_min']:>9.2f}  {row['elbow_at_deep']:>9.2f}  "
              f"{row['gripper_max']:>7.2f}  {int(row['n_frames']):>6}  {row['duration_s']:>6.1f}")

    # --- Q2: Typical grasp trajectory for CENTER episodes ---
    print(f"\n{'='*70}")
    print("Q2: TYPICAL GRASP TRAJECTORY - CENTER EPISODES")
    print("="*70)

    if len(center_eps) > 0:
        print(f"\nAt deepest point (shoulder minimum = deepest arm position):")
        print(f"  Shoulder: mean={center_eps['deep_shoulder'].mean():.1f}  "
              f"std={center_eps['deep_shoulder'].std():.1f}  "
              f"range=[{center_eps['deep_shoulder'].min():.1f}, {center_eps['deep_shoulder'].max():.1f}]")
        print(f"  Elbow:    mean={center_eps['deep_elbow'].mean():.1f}  "
              f"std={center_eps['deep_elbow'].std():.1f}  "
              f"range=[{center_eps['deep_elbow'].min():.1f}, {center_eps['deep_elbow'].max():.1f}]")
        print(f"  Gripper:  mean={center_eps['deep_gripper'].mean():.1f}  "
              f"std={center_eps['deep_gripper'].std():.1f}  "
              f"range=[{center_eps['deep_gripper'].min():.1f}, {center_eps['deep_gripper'].max():.1f}]")

        print(f"\nAt grasp point (after gripper peak):")
        print(f"  Shoulder: mean={center_eps['grasp_shoulder'].mean():.1f}  "
              f"std={center_eps['grasp_shoulder'].std():.1f}  "
              f"range=[{center_eps['grasp_shoulder'].min():.1f}, {center_eps['grasp_shoulder'].max():.1f}]")
        print(f"  Elbow:    mean={center_eps['grasp_elbow'].mean():.1f}  "
              f"std={center_eps['grasp_elbow'].std():.1f}  "
              f"range=[{center_eps['grasp_elbow'].min():.1f}, {center_eps['grasp_elbow'].max():.1f}]")
        print(f"  Wrist_pitch: mean={center_eps['grasp_wrist_pitch'].mean():.1f}  "
              f"std={center_eps['grasp_wrist_pitch'].std():.1f}  "
              f"range=[{center_eps['grasp_wrist_pitch'].min():.1f}, {center_eps['grasp_wrist_pitch'].max():.1f}]")
        print(f"  Gripper:  mean={center_eps['grasp_gripper'].mean():.1f}  "
              f"std={center_eps['grasp_gripper'].std():.1f}  "
              f"range=[{center_eps['grasp_gripper'].min():.1f}, {center_eps['grasp_gripper'].max():.1f}]")

        # Trajectory shape: per-phase analysis
        # Get a few representative CENTER episodes and show their trajectory
        print(f"\n--- Per-episode phase analysis (CENTER episodes only) ---")
        print(f"{'EpID':>5}  {'BaseRng':>7}  {'ShldMin':>7}  {'ShldMax':>7}  {'ElbwMin':>7}  {'ElbwMax':>7}  {'GripMax':>7}  {'GripEnd':>7}")
        print("-"*60)
        for _, row in center_eps.sort_values('ep_id').iterrows():
            print(f"{int(row['ep_id']):>5}  "
                  f"[{-row['max_abs_base']:>4.1f},{row['max_abs_base']:>4.1f}]  "
                  f"{row['shoulder_min']:>7.1f}  {row['shoulder_max']:>7.1f}  "
                  f"{row['elbow_min']:>7.1f}  {row['elbow_max']:>7.1f}  "
                  f"{row['gripper_max']:>7.1f}")

    # --- Q4: Elbow angle range at lowest point ---
    print(f"\n{'='*70}")
    print("Q4: ELBOW ANGLE AT LOWEST POINT (deepest shoulder position)")
    print("="*70)
    if len(center_eps) > 0:
        e_min = center_eps['elbow_at_deep'].min()
        e_max = center_eps['elbow_at_deep'].max()
        e_mean = center_eps['elbow_at_deep'].mean()
        e_std = center_eps['elbow_at_deep'].std()
        e_q25 = center_eps['elbow_at_deep'].quantile(0.25)
        e_med = center_eps['elbow_at_deep'].quantile(0.5)
        e_q75 = center_eps['elbow_at_deep'].quantile(0.75)
        print(f"  COUNT : {len(center_eps)} CENTER episodes")
        print(f"  mean  : {e_mean:.1f} deg")
        print(f"  std   : {e_std:.1f} deg")
        print(f"  min   : {e_min:.1f} deg")
        print(f"  q25   : {e_q25:.1f} deg")
        print(f"  median: {e_med:.1f} deg")
        print(f"  q75   : {e_q75:.1f} deg")
        print(f"  max   : {e_max:.1f} deg")

        # Also full elbow range across all frames in CENTER episodes
        center_ep_ids = center_eps['ep_id'].tolist()
        center_frames = df[df['episode_index'].isin(center_ep_ids)]
        print(f"\n  Full elbow range across ALL frames in CENTER episodes:")
        print(f"  mean={center_frames['elbow'].mean():.1f}  "
              f"std={center_frames['elbow'].std():.1f}  "
              f"min={center_frames['elbow'].min():.1f}  "
              f"max={center_frames['elbow'].max():.1f}")

    # --- Comparison: CENTER vs non-CENTER ---
    print(f"\n{'='*70}")
    print("CENTER vs NON-CENTER COMPARISON")
    print("="*70)
    if len(center_eps) > 0 and len(non_center_eps) > 0:
        print(f"{'Metric':<25}  {'CENTER (n=' + str(len(center_eps)) + ')':>18}  {'Non-CENTER (n=' + str(len(non_center_eps)) + ')':>22}")
        print("-"*70)
        for metric, label in [
            ('shoulder_min', 'Shoulder min (deep)'),
            ('elbow_at_deep', 'Elbow at deep pt'),
            ('gripper_max', 'Gripper max'),
            ('n_frames', 'Frames/ep'),
        ]:
            c_vals = center_eps[metric]
            n_vals = non_center_eps[metric]
            print(f"  {label:<23}  {c_vals.mean():>6.1f} ± {c_vals.std():>5.1f}  |  "
                  f"{n_vals.mean():>6.1f} ± {n_vals.std():>5.1f}")

    # --- Shoulder minimum distribution in CENTER ---
    print(f"\n{'='*70}")
    print("SHOULDER MINIMUM DISTRIBUTION IN CENTER EPISODES")
    print("  (Low shoulder = deep grasp. Target: shoulder_min < 10 deg)")
    print("="*70)
    if len(center_eps) > 0:
        buckets = [
            ("< 0 deg  (very deep)", center_eps['shoulder_min'] < 0),
            ("0-10 deg (deep)",      (center_eps['shoulder_min'] >= 0) & (center_eps['shoulder_min'] < 10)),
            ("10-20 deg (moderate)", (center_eps['shoulder_min'] >= 10) & (center_eps['shoulder_min'] < 20)),
            ("20-30 deg (approach)", (center_eps['shoulder_min'] >= 20) & (center_eps['shoulder_min'] < 30)),
            ("> 30 deg (shallow)",   center_eps['shoulder_min'] >= 30),
        ]
        for label, mask in buckets:
            count = mask.sum()
            pct = 100.0 * count / len(center_eps)
            print(f"  {label:<30}: {count:>2} eps ({pct:>4.0f}%)")

    # --- Detailed trajectory of a few CENTER episodes ---
    print(f"\n{'='*70}")
    print("DETAILED TRAJECTORIES: 5 REPRESENTATIVE CENTER EPISODES")
    print("="*70)
    # Pick 5 episodes spread across the CENTER set
    sample_ids = center_eps.nsmallest(5, 'shoulder_min')['ep_id'].tolist()
    if len(sample_ids) < 5:
        sample_ids = center_eps['ep_id'].tolist()[:5]

    for ep_id in sample_ids:
        ep_data = df[df['episode_index'] == ep_id].sort_values('frame_index').reset_index(drop=True)
        n = len(ep_data)
        keyframes = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
        keyframes = sorted(set([min(k, n-1) for k in keyframes]))

        print(f"\n  Episode {ep_id} ({n} frames, {n/30:.1f}s):")
        print(f"  {'Frame':>6}  {'T(s)':>5}  {'Base':>6}  {'Shld':>6}  {'Elbow':>6}  {'WrPit':>6}  {'WrRol':>6}  {'Grip':>6}")
        for fi in keyframes:
            row = ep_data.iloc[fi]
            print(f"  {int(row['frame_index']):>6}  {float(row['timestamp']):>5.1f}  "
                  f"{row['base']:>6.1f}  {row['shoulder']:>6.1f}  {row['elbow']:>6.1f}  "
                  f"{row['wrist_pitch']:>6.1f}  {row['wrist_roll']:>6.1f}  {row['gripper']:>6.1f}")

    return center_eps, ep_df_all


def main():
    df = load_data()
    center_eps, all_eps = analyze_center_episodes(df)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    center_count = len(center_eps)
    total = len(all_eps)
    print(f"  CENTER episodes (max|base| < 10 deg): {center_count} / {total} ({100*center_count/total:.0f}%)")

    if center_count > 0:
        smean = center_eps['shoulder_min'].mean()
        emean = center_eps['elbow_at_deep'].mean()
        gmean = center_eps['gripper_max'].mean()
        deep = (center_eps['shoulder_min'] < 10).sum()
        print(f"  CENTER episodes with shoulder_min < 10 deg: {deep} / {center_count} ({100*deep/center_count:.0f}%)")
        print(f"  Typical CENTER grasp: shoulder_min={smean:.1f}, elbow_at_deep={emean:.1f}, gripper_max={gmean:.1f}")
        print(f"\n  Elbow range at deepest point in CENTER eps:")
        print(f"    min={center_eps['elbow_at_deep'].min():.1f}  "
              f"mean={emean:.1f}  "
              f"max={center_eps['elbow_at_deep'].max():.1f}")

    print(f"\n[Script done: {__file__}]")


if __name__ == "__main__":
    main()
