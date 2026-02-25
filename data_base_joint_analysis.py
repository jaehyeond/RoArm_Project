#!/usr/bin/env python3
"""
data_base_joint_analysis.py
Analyze base joint (joint 0) distribution in lerobot_dataset_v3
to understand clustering that may cause model to predict Base->51 deg consistently.

Key questions:
1. What is the actual base angle distribution across all 13145 frames?
2. Are there clusters (bimodal, attractor at ~51 deg)?
3. Per-episode base angle spread and mean
4. Why does model output Base=51 deg when dataset mean=-0.47?
"""

import pandas as pd
import numpy as np
import json
import os

DATASET_PATH = "/home/cgxr/Documents/Robotics/RoArm_Project/lerobot_dataset_v3"
PARQUET_PATH = f"{DATASET_PATH}/data/chunk-000/file-000.parquet"
META_PATH = f"{DATASET_PATH}/meta"

def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    # Extract joint arrays into separate columns
    action_arr = np.array(df['action'].tolist())
    state_arr = np.array(df['observation.state'].tolist())

    joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    for i, name in enumerate(joint_names):
        df[f'action_{name}'] = action_arr[:, i]
        df[f'state_{name}'] = state_arr[:, i]
    return df

def analyze_base_distribution(df):
    print("=" * 70)
    print("BASE JOINT DISTRIBUTION ANALYSIS (lerobot_dataset_v3)")
    print("=" * 70)

    base = df['action_base']

    print("\n--- Overall Base Joint Stats (action, 13145 frames) ---")
    print(f"  Mean:    {base.mean():.2f} deg")
    print(f"  Std:     {base.std():.2f} deg")
    print(f"  Min:     {base.min():.2f} deg")
    print(f"  Max:     {base.max():.2f} deg")
    print(f"  Median:  {base.median():.2f} deg")
    print(f"  Q10:     {base.quantile(0.10):.2f} deg")
    print(f"  Q25:     {base.quantile(0.25):.2f} deg")
    print(f"  Q75:     {base.quantile(0.75):.2f} deg")
    print(f"  Q90:     {base.quantile(0.90):.2f} deg")
    print(f"  Q99:     {base.quantile(0.99):.2f} deg")

    # Histogram buckets to find clusters
    print("\n--- Base Angle Histogram (10 deg bins) ---")
    bins = np.arange(-70, 80, 10)
    counts, edges = np.histogram(base, bins=bins)
    total = len(base)
    for i, (count, lo) in enumerate(zip(counts, edges[:-1])):
        hi = edges[i+1]
        pct = 100 * count / total
        bar = '#' * int(pct / 0.5)
        print(f"  [{lo:+5.0f} to {hi:+5.0f}]: {count:5d} ({pct:5.1f}%) {bar}")

    # Check around 51 deg specifically
    print("\n--- Frames with base in 40-60 deg range ---")
    mask_50 = (base >= 40) & (base <= 60)
    n_50 = mask_50.sum()
    print(f"  Frames 40-60 deg: {n_50} ({100*n_50/total:.1f}% of all frames)")

    mask_positive = base > 10
    mask_negative = base < -10
    mask_center = (base >= -10) & (base <= 10)
    print(f"  Frames > +10 deg: {mask_positive.sum()} ({100*mask_positive.sum()/total:.1f}%)")
    print(f"  Frames -10 to +10: {mask_center.sum()} ({100*mask_center.sum()/total:.1f}%)")
    print(f"  Frames < -10 deg: {mask_negative.sum()} ({100*mask_negative.sum()/total:.1f}%)")

def analyze_per_episode_base(df):
    print("\n" + "=" * 70)
    print("PER-EPISODE BASE ANGLE ANALYSIS")
    print("=" * 70)

    episodes = df.groupby('episode_index')

    ep_stats = []
    for ep_id, ep_df in episodes:
        base = ep_df['action_base']
        ep_stats.append({
            'episode': ep_id,
            'n_frames': len(ep_df),
            'base_mean': base.mean(),
            'base_std': base.std(),
            'base_min': base.min(),
            'base_max': base.max(),
            'base_range': base.max() - base.min(),
        })

    ep_df2 = pd.DataFrame(ep_stats)

    print(f"\nTotal episodes: {len(ep_df2)}")
    print(f"\nEpisode-level base mean: {ep_df2['base_mean'].mean():.2f} +/- {ep_df2['base_mean'].std():.2f} deg")
    print(f"Episode-level base mean range: [{ep_df2['base_mean'].min():.2f}, {ep_df2['base_mean'].max():.2f}]")

    # Zone distribution
    zones = {
        'LEFT_FAR (< -30)': (ep_df2['base_mean'] < -30).sum(),
        'LEFT (-30 to -10)': ((ep_df2['base_mean'] >= -30) & (ep_df2['base_mean'] < -10)).sum(),
        'CENTER (-10 to +10)': ((ep_df2['base_mean'] >= -10) & (ep_df2['base_mean'] <= 10)).sum(),
        'RIGHT (+10 to +30)': ((ep_df2['base_mean'] > 10) & (ep_df2['base_mean'] <= 30)).sum(),
        'RIGHT_FAR (> +30)': (ep_df2['base_mean'] > 30).sum(),
    }
    print("\n--- Episode Zone Distribution (by episode mean base angle) ---")
    for zone, count in zones.items():
        print(f"  {zone}: {count} episodes ({100*count/len(ep_df2):.1f}%)")

    # Show episodes with high base angles that might be attractors
    print("\n--- Episodes with base mean > +20 deg (right-side attractors) ---")
    high_base = ep_df2[ep_df2['base_mean'] > 20].sort_values('base_mean', ascending=False)
    if len(high_base) > 0:
        print(high_base.to_string(index=False))
    else:
        print("  None found")

    # Frame-count weighted: do high-base episodes have more frames?
    print("\n--- Frame count vs base mean correlation ---")
    corr = ep_df2[['n_frames', 'base_mean']].corr().iloc[0, 1]
    print(f"  Correlation (n_frames vs base_mean): {corr:.3f}")

    # Weighted mean (frame-count weighted)
    weighted_mean = np.average(ep_df2['base_mean'], weights=ep_df2['n_frames'])
    print(f"  Frame-weighted episode base mean: {weighted_mean:.2f} deg")
    print(f"  Simple (frame-level) base mean: {df['action_base'].mean():.2f} deg")

    return ep_df2

def analyze_base_vs_shoulder(df):
    """Check if high-base frames have specific shoulder patterns (potential cluster)."""
    print("\n" + "=" * 70)
    print("BASE vs SHOULDER: DO HIGH-BASE FRAMES CLUSTER?")
    print("=" * 70)

    # Split by base angle quadrants
    base = df['action_base']
    shoulder = df['action_shoulder']

    segments = [
        ('base < -20', base < -20),
        ('-20 <= base < 0', (base >= -20) & (base < 0)),
        ('0 <= base < 20', (base >= 0) & (base < 20)),
        ('20 <= base < 40', (base >= 20) & (base < 40)),
        ('base >= 40', base >= 40),
    ]

    print(f"\n{'Base segment':<30} {'N frames':>8} {'Pct':>6} {'Shoulder mean':>14} {'Shoulder std':>12}")
    print("-" * 75)
    for label, mask in segments:
        n = mask.sum()
        pct = 100 * n / len(df)
        sh_mean = shoulder[mask].mean() if n > 0 else float('nan')
        sh_std = shoulder[mask].std() if n > 0 else float('nan')
        print(f"  {label:<28} {n:>8} {pct:>5.1f}% {sh_mean:>13.2f} {sh_std:>12.2f}")

    # The critical question: is there a cluster at base~51, shoulder~30?
    # That's where model converges
    mask_attractor = (base >= 40) & (base <= 60) & (shoulder >= 25) & (shoulder <= 40)
    n_attractor = mask_attractor.sum()
    print(f"\n--- Frames matching deployment attractor [base 40-60, shoulder 25-40] ---")
    print(f"  Count: {n_attractor} ({100*n_attractor/len(df):.1f}% of all frames)")

def analyze_temporal_base_pattern(df):
    """Check if base tends to go toward +51 at specific points in episodes."""
    print("\n" + "=" * 70)
    print("TEMPORAL PATTERN: BASE ANGLE ACROSS EPISODE PHASES")
    print("=" * 70)

    # Normalize frame position to [0, 1] within each episode
    df2 = df.copy()
    ep_lengths = df2.groupby('episode_index')['frame_index'].transform('max')
    df2['phase'] = df2['frame_index'] / ep_lengths.clip(lower=1)

    # Bin by phase
    phase_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

    print(f"\n{'Phase':<20} {'N frames':>8} {'Base mean':>10} {'Base std':>10}")
    print("-" * 55)
    for lo, hi in phase_bins:
        mask = (df2['phase'] >= lo) & (df2['phase'] < hi)
        n = mask.sum()
        b_mean = df2.loc[mask, 'action_base'].mean()
        b_std = df2.loc[mask, 'action_base'].std()
        print(f"  {lo:.0%} - {hi:.0%}           {n:>8} {b_mean:>10.2f} {b_std:>10.2f}")

def analyze_start_end_positions(df):
    """Look at start (frame 0) and end positions per episode."""
    print("\n" + "=" * 70)
    print("START / END BASE ANGLE PER EPISODE")
    print("=" * 70)

    start_frames = df[df['frame_index'] == 0].copy()

    print(f"\n--- Start (frame 0) base angle stats ({len(start_frames)} episodes) ---")
    base_start = start_frames['action_base']
    print(f"  Mean: {base_start.mean():.2f} deg")
    print(f"  Std:  {base_start.std():.2f} deg")
    print(f"  Range: [{base_start.min():.2f}, {base_start.max():.2f}]")

    # End frames per episode
    end_frames = df.loc[df.groupby('episode_index')['frame_index'].idxmax()]

    print(f"\n--- End (last frame) base angle stats ({len(end_frames)} episodes) ---")
    base_end = end_frames['action_base']
    print(f"  Mean: {base_end.mean():.2f} deg")
    print(f"  Std:  {base_end.std():.2f} deg")
    print(f"  Range: [{base_end.min():.2f}, {base_end.max():.2f}]")

    # Histogram of start positions
    print("\n--- Start position distribution (10 deg bins) ---")
    bins = np.arange(-70, 80, 10)
    counts, edges = np.histogram(base_start, bins=bins)
    for i, (count, lo) in enumerate(zip(counts, edges[:-1])):
        if count > 0:
            hi = edges[i+1]
            bar = '#' * count
            print(f"  [{lo:+5.0f} to {hi:+5.0f}]: {count:3d} eps  {bar}")

    # Episodes where start position > 40 deg base
    high_start = start_frames[start_frames['action_base'] > 40]
    print(f"\n--- Episodes starting with base > +40 deg: {len(high_start)} ---")
    if len(high_start) > 0:
        print(high_start[['episode_index', 'action_base', 'action_shoulder', 'action_elbow']].to_string(index=False))

def analyze_attractor_hypothesis(df):
    """
    Key question: Why does model output Base=51 consistently?
    Check if there's a sub-cluster of frames at ~51 deg that could be over-represented.
    """
    print("\n" + "=" * 70)
    print("ATTRACTOR HYPOTHESIS: WHY DOES MODEL PREDICT BASE=51?")
    print("=" * 70)

    base = df['action_base']

    # The model converges to [2.5, 30, 70, 14, -1.7, 25] (from memory)
    # Wait - user says model outputs Base=51. Let me analyze robustly.

    # Check density at specific angles
    test_angles = [-50, -30, -10, 0, 10, 20, 30, 40, 50, 60]
    print("\n--- Frame density at specific base angles (+-5 deg window) ---")
    for angle in test_angles:
        mask = (base >= angle - 5) & (base < angle + 5)
        n = mask.sum()
        pct = 100 * n / len(df)
        bar = '#' * int(pct / 0.3)
        print(f"  base ~= {angle:+4d} deg: {n:5d} frames ({pct:5.1f}%) {bar}")

    # How many frames have base exactly in [45-55] range?
    mask_51 = (base >= 45) & (base <= 55)
    print(f"\n--- Frames with base in [45, 55] deg: {mask_51.sum()} ({100*mask_51.sum()/len(df):.1f}%) ---")
    if mask_51.sum() > 0:
        sub = df[mask_51]
        print(f"  Shoulder mean in these frames: {sub['action_shoulder'].mean():.2f}")
        print(f"  Elbow mean in these frames: {sub['action_elbow'].mean():.2f}")
        print(f"  Which episodes have these frames?")
        ep_counts = sub.groupby('episode_index').size()
        print(f"  {ep_counts.to_dict()}")

    # The actual deployment attractor from memory was [2.5, 30, 70, 14, -1.7, 25]
    # base=2.5, not 51. Let me verify the user's statement.
    print("\n--- NOTE: V3 deployment converged to [2.5, 30, 70, 14, -1.7, 25] (from analysis) ---")
    print("  If user sees Base=51, this may be from a DIFFERENT deployment run.")
    print("  Let me check what frames cluster around base=2.5, shoulder=30 ---")

    mask_converged = (base >= -5) & (base <= 7) & (df['action_shoulder'] >= 25) & (df['action_shoulder'] <= 35)
    n_conv = mask_converged.sum()
    print(f"  Frames at [base~2.5, shoulder~30]: {n_conv} ({100*n_conv/len(df):.1f}%)")

def main():
    print("Loading lerobot_dataset_v3 parquet data...")
    df = load_data()
    print(f"Loaded {len(df)} frames, {df['episode_index'].nunique()} episodes")

    analyze_base_distribution(df)
    ep_df = analyze_per_episode_base(df)
    analyze_base_vs_shoulder(df)
    analyze_temporal_base_pattern(df)
    analyze_start_end_positions(df)
    analyze_attractor_hypothesis(df)

    print("\n" + "=" * 70)
    print("SUMMARY: BASE JOINT ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey stats for model behavior diagnosis:")
    print(f"  action.base mean: {df['action_base'].mean():.2f} deg (should match stats.json: -0.47)")
    print(f"  action.base std:  {df['action_base'].std():.2f} deg (should match stats.json: 25.81)")
    print(f"  action.base median: {df['action_base'].median():.2f} deg")
    print(f"  % frames base > 0: {100*(df['action_base'] > 0).mean():.1f}%")
    print(f"  % frames base > 20: {100*(df['action_base'] > 20).mean():.1f}%")
    print(f"  % frames base > 40: {100*(df['action_base'] > 40).mean():.1f}%")

if __name__ == '__main__':
    main()
