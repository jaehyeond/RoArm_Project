"""
Data Distribution Gap Analysis (No matplotlib version)
Analyzes action distributions and identifies underrepresented regions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Joint indices and names
JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist_pitch', 'Wrist_roll', 'Gripper']
BASE, SHOULDER, ELBOW, WRIST_PITCH, WRIST_ROLL, GRIPPER = 0, 1, 2, 3, 4, 5

def analyze_distribution():
    """Analyze action distribution and identify gaps."""

    # Load dataset
    parquet_path = Path("E:/RoArm_Project/lerobot_dataset_v3/data/chunk-000/file-000.parquet")
    print(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Extract actions
    if 'action' in df.columns:
        actions = np.stack(df['action'].values)
    else:
        action_cols = [col for col in df.columns if col.startswith('action')]
        actions = df[action_cols].values

    print(f"Action shape: {actions.shape}")
    print(f"Total frames: {len(actions)}")

    # Compute normalization stats
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)

    print("\n" + "="*80)
    print("NORMALIZATION STATISTICS")
    print("="*80)
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:12s}: mean={action_mean[i]:7.2f}, std={action_std[i]:6.2f}")

    # Detailed elbow analysis
    print("\n" + "="*80)
    print("ELBOW DISTRIBUTION ANALYSIS (Joint 2)")
    print("="*80)
    elbow_vals = actions[:, ELBOW]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(elbow_vals, percentiles)

    print("\nPercentile values:")
    for pct, val in zip(percentiles, pct_vals):
        z_score = (val - action_mean[ELBOW]) / action_std[ELBOW]
        print(f"  P{pct:2d}: {val:7.2f} deg  (z={z_score:+.2f})")

    # Underrepresented regions
    print("\nUnderrepresented regions:")
    regions = [
        ("Elbow < -60 deg (deep grasp)", elbow_vals < -60),
        ("Elbow < -40 deg (grasp zone)", elbow_vals < -40),
        ("Elbow < -30 deg (pre-grasp)", elbow_vals < -30),
        ("Elbow < -20 deg (approach)", elbow_vals < -20),
        ("Elbow < 0 deg (forward)", elbow_vals < 0),
    ]

    for region_name, mask in regions:
        count = mask.sum()
        pct = count / len(elbow_vals) * 100
        print(f"  {region_name:35s}: {count:5d} frames ({pct:5.2f}%)")

    # Z-score analysis
    print("\nZ-score distribution for critical angles:")
    critical_angles = [-64, -60, -40, -30, -20, 0, 25.19]  # 25.19 is mean
    for angle in critical_angles:
        z = (angle - action_mean[ELBOW]) / action_std[ELBOW]
        label = "(dataset mean)" if angle == 25.19 else ""
        print(f"  Elbow = {angle:6.1f} deg → z = {z:+.2f}  {label}")

    # Gripper analysis
    print("\n" + "="*80)
    print("GRIPPER DISTRIBUTION ANALYSIS (Joint 5)")
    print("="*80)
    gripper_vals = actions[:, GRIPPER]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(gripper_vals, percentiles)

    print("\nPercentile values:")
    for pct, val in zip(percentiles, pct_vals):
        z_score = (val - action_mean[GRIPPER]) / action_std[GRIPPER]
        print(f"  P{pct:2d}: {val:7.2f} deg  (z={z_score:+.2f})")

    print("\nGripper opening regions:")
    regions = [
        ("Gripper > 50 deg (wide open)", gripper_vals > 50),
        ("Gripper > 30 deg (open)", gripper_vals > 30),
        ("Gripper > 10 deg (slightly open)", gripper_vals > 10),
        ("Gripper < 10 deg (closed)", gripper_vals < 10),
    ]

    for region_name, mask in regions:
        count = mask.sum()
        pct = count / len(gripper_vals) * 100
        print(f"  {region_name:35s}: {count:5d} frames ({pct:5.2f}%)")

    # Calculate data augmentation needs
    print("\n" + "="*80)
    print("DATA AUGMENTATION ESTIMATE")
    print("="*80)

    current_frames = len(elbow_vals)
    current_neg30_frames = (elbow_vals < -30).sum()
    current_neg30_pct = current_neg30_frames / current_frames * 100

    # Target: 20% of data should be elbow < -30
    target_pct = 20.0

    # If we want 20% to be elbow < -30, assuming all new episodes have elbow < -30:
    # (current_frames + new_frames) * 0.20 = current_neg30 + new_frames
    # Solve for new_frames
    needed_frames = (current_neg30_frames - current_frames * target_pct / 100) / (1 - target_pct / 100)

    avg_frames_per_ep = current_frames / 51

    if needed_frames > 0:
        needed_episodes = int(np.ceil(needed_frames / avg_frames_per_ep))

        print(f"Current elbow < -30 deg coverage: {current_neg30_pct:.2f}% ({current_neg30_frames} frames)")
        print(f"Target coverage: {target_pct:.1f}%")
        print(f"Additional frames needed: {int(needed_frames)}")
        print(f"Estimated episodes to collect: {needed_episodes} (avg {avg_frames_per_ep:.0f} frames/ep)")
        print(f"Total episodes after: {51 + needed_episodes}")
    else:
        print(f"Current coverage ({current_neg30_pct:.2f}%) exceeds target ({target_pct:.1f}%)")

    # For elbow < -60 (deep grasp)
    current_neg60_frames = (elbow_vals < -60).sum()
    current_neg60_pct = current_neg60_frames / current_frames * 100
    target_neg60_pct = 10.0

    needed_frames_60 = (current_neg60_frames - current_frames * target_neg60_pct / 100) / (1 - target_neg60_pct / 100)

    if needed_frames_60 > 0:
        needed_episodes_60 = int(np.ceil(needed_frames_60 / avg_frames_per_ep))

        print(f"\nFor deep grasp zone (elbow < -60 deg):")
        print(f"Current coverage: {current_neg60_pct:.2f}% ({current_neg60_frames} frames)")
        print(f"Target coverage: {target_neg60_pct:.1f}%")
        print(f"Additional frames needed: {int(needed_frames_60)}")
        print(f"Estimated episodes to collect: {needed_episodes_60}")
        print(f"Total episodes after: {51 + needed_episodes_60}")

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(actions.T)

    print("\nJoint correlation matrix:")
    print("         ", "  ".join([f"{name:8s}" for name in JOINT_NAMES]))
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:8s}", "  ".join([f"{corr_matrix[i, j]:8.3f}" for j in range(6)]))

    # Key correlations
    print(f"\nKey correlations:")
    print(f"  Shoulder-Elbow: {corr_matrix[SHOULDER, ELBOW]:.3f}")
    print(f"  Elbow-Wrist_pitch: {corr_matrix[ELBOW, WRIST_PITCH]:.3f}")
    print(f"  Elbow-Gripper: {corr_matrix[ELBOW, GRIPPER]:.3f}")

    # Episode-level statistics
    print("\n" + "="*80)
    print("EPISODE-LEVEL STATISTICS")
    print("="*80)

    episodes = df['episode_index'].unique()
    elbow_mins_per_ep = []

    for ep_idx in episodes:
        ep_data = df[df['episode_index'] == ep_idx]
        if 'action' in df.columns:
            ep_actions = np.stack(ep_data['action'].values)
        else:
            action_cols = [col for col in df.columns if col.startswith('action')]
            ep_actions = ep_data[action_cols].values

        ep_elbow = ep_actions[:, ELBOW]
        elbow_mins_per_ep.append(ep_elbow.min())

    elbow_mins_per_ep = np.array(elbow_mins_per_ep)

    print(f"Episode count: {len(episodes)}")
    print(f"Episodes with min_elbow < 0 deg: {(elbow_mins_per_ep < 0).sum()} ({(elbow_mins_per_ep < 0).sum() / len(episodes) * 100:.1f}%)")
    print(f"Episodes with min_elbow < -20 deg: {(elbow_mins_per_ep < -20).sum()} ({(elbow_mins_per_ep < -20).sum() / len(episodes) * 100:.1f}%)")
    print(f"Episodes with min_elbow < -40 deg: {(elbow_mins_per_ep < -40).sum()} ({(elbow_mins_per_ep < -40).sum() / len(episodes) * 100:.1f}%)")
    print(f"Episodes with min_elbow < -60 deg: {(elbow_mins_per_ep < -60).sum()} ({(elbow_mins_per_ep < -60).sum() / len(episodes) * 100:.1f}%)")

    print("\n[OK] Distribution analysis complete")

if __name__ == "__main__":
    analyze_distribution()
