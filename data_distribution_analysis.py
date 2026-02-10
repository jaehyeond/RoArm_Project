"""
Data Distribution Gap Analysis for RoArm M3 SmolVLA Dataset
Creates histograms and identifies underrepresented regions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print("Action mean:", action_mean)
    print("Action std: ", action_std)

    # Create distribution plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('RoArm M3 Action Distribution Analysis (51 Episodes, 13,010 Frames)', fontsize=14, fontweight='bold')

    for i, (joint_name, ax) in enumerate(zip(JOINT_NAMES, axes.flatten())):
        joint_vals = actions[:, i]

        # Histogram
        ax.hist(joint_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.percentile(joint_vals, percentiles)

        # Mark key percentiles
        for pct, val in zip([5, 50, 95], [pct_vals[1], pct_vals[4], pct_vals[7]]):
            ax.axvline(val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P{pct}: {val:.1f}°')

        # Mean and std
        ax.axvline(action_mean[i], color='green', linestyle='-', linewidth=2, label=f'Mean: {action_mean[i]:.1f}°')
        ax.axvspan(action_mean[i] - action_std[i], action_mean[i] + action_std[i],
                   alpha=0.2, color='green', label=f'±1σ: {action_std[i]:.1f}°')

        ax.set_xlabel(f'{joint_name} Angle (degrees)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{joint_name} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("E:/RoArm_Project/data_distribution_histograms.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Histograms saved to {output_path}")
    plt.close()

    # Detailed elbow analysis
    print("\n" + "="*80)
    print("ELBOW DISTRIBUTION ANALYSIS (Joint 2)")
    print("="*80)
    elbow_vals = actions[:, ELBOW]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(elbow_vals, percentiles)

    print("\nPercentile values:")
    for pct, val in zip(percentiles, pct_vals):
        print(f"  P{pct:2d}: {val:6.1f}°")

    # Underrepresented regions
    print("\nUnderrepresented regions:")
    regions = [
        ("Elbow < -60° (deep grasp)", elbow_vals < -60),
        ("Elbow < -40° (grasp zone)", elbow_vals < -40),
        ("Elbow < -30° (pre-grasp)", elbow_vals < -30),
        ("Elbow < -20° (approach)", elbow_vals < -20),
        ("Elbow < 0° (forward)", elbow_vals < 0),
    ]

    for region_name, mask in regions:
        count = mask.sum()
        pct = count / len(elbow_vals) * 100
        print(f"  {region_name:30s}: {count:5d} frames ({pct:5.2f}%)")

    # Create detailed elbow plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(elbow_vals, bins=100, alpha=0.7, color='steelblue', edgecolor='black')

    # Color code bins
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -60:
            patch.set_facecolor('darkred')  # Critical grasp zone
        elif bin_center < -30:
            patch.set_facecolor('orange')  # Grasp approach
        elif bin_center < 0:
            patch.set_facecolor('yellow')  # Forward reach

    # Mark key thresholds
    ax.axvline(-60, color='darkred', linestyle='--', linewidth=2, label='Deep Grasp (-60°): 0.4% of data')
    ax.axvline(-30, color='orange', linestyle='--', linewidth=2, label='Pre-Grasp (-30°): 3.1% of data')
    ax.axvline(0, color='yellow', linestyle='--', linewidth=2, label='Forward (0°): 17.3% of data')
    ax.axvline(action_mean[ELBOW], color='green', linestyle='-', linewidth=2.5, label=f'Mean: {action_mean[ELBOW]:.1f}°')

    # Normalization z-score overlay
    ax2 = ax.twiny()
    z_ticks = np.array([-3, -2, -1, 0, 1, 2, 3])
    z_labels = [f"{z:.1f}σ" for z in z_ticks]
    z_positions = action_mean[ELBOW] + z_ticks * action_std[ELBOW]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(z_positions)
    ax2.set_xticklabels(z_labels)
    ax2.set_xlabel('Z-Score (normalized space)', fontsize=11, color='purple')
    ax2.tick_params(axis='x', colors='purple')

    ax.set_xlabel('Elbow Angle (degrees)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Elbow Distribution with Critical Zones\n(Model outputs z ∈ [-1.5, 1.5], but elbow=-64° requires z=-3.04)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("E:/RoArm_Project/data_elbow_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Elbow plot saved to {output_path}")
    plt.close()

    # Gripper analysis
    print("\n" + "="*80)
    print("GRIPPER DISTRIBUTION ANALYSIS (Joint 5)")
    print("="*80)
    gripper_vals = actions[:, GRIPPER]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(gripper_vals, percentiles)

    print("\nPercentile values:")
    for pct, val in zip(percentiles, pct_vals):
        print(f"  P{pct:2d}: {val:6.1f}°")

    print("\nGripper opening regions:")
    regions = [
        ("Gripper > 50° (wide open)", gripper_vals > 50),
        ("Gripper > 30° (open)", gripper_vals > 30),
        ("Gripper > 10° (slightly open)", gripper_vals > 10),
        ("Gripper < 10° (closed)", gripper_vals < 10),
    ]

    for region_name, mask in regions:
        count = mask.sum()
        pct = count / len(gripper_vals) * 100
        print(f"  {region_name:30s}: {count:5d} frames ({pct:5.2f}%)")

    # Calculate data augmentation needs
    print("\n" + "="*80)
    print("DATA AUGMENTATION ESTIMATE")
    print("="*80)

    # Current: elbow < -30 is 3.1% of data
    # Target: 20% of data (balanced with other regions)
    current_elbow_neg30_pct = (elbow_vals < -30).sum() / len(elbow_vals) * 100
    target_pct = 20.0

    current_frames = len(elbow_vals)
    current_neg30_frames = (elbow_vals < -30).sum()

    # If we want 20% to be elbow < -30:
    # new_total * 0.20 = current_neg30 + new_neg30
    # Assume all new episodes have elbow < -30:
    # (current_frames + new_frames) * 0.20 = current_neg30 + new_frames
    # current_frames * 0.20 + new_frames * 0.20 = current_neg30 + new_frames
    # new_frames * 0.80 = current_neg30 - current_frames * 0.20
    # new_frames = (current_neg30 - current_frames * 0.20) / 0.80

    needed_frames = (current_neg30_frames - current_frames * target_pct / 100) / (1 - target_pct / 100)

    if needed_frames > 0:
        # Frames per episode (average)
        avg_frames_per_ep = current_frames / 51
        needed_episodes = int(np.ceil(needed_frames / avg_frames_per_ep))

        print(f"Current elbow < -30° coverage: {current_elbow_neg30_pct:.2f}%")
        print(f"Target coverage: {target_pct:.1f}%")
        print(f"Additional frames needed: {int(needed_frames)}")
        print(f"Estimated episodes to collect: {needed_episodes} (avg {avg_frames_per_ep:.0f} frames/ep)")
        print(f"Total episodes after: {51 + needed_episodes}")
    else:
        # For elbow < -60 (currently 0.4%)
        current_neg60_frames = (elbow_vals < -60).sum()
        target_neg60_pct = 10.0

        needed_frames_60 = (current_neg60_frames - current_frames * target_neg60_pct / 100) / (1 - target_neg60_pct / 100)
        avg_frames_per_ep = current_frames / 51
        needed_episodes_60 = int(np.ceil(needed_frames_60 / avg_frames_per_ep))

        print(f"Current elbow < -60° coverage: {(elbow_vals < -60).sum() / len(elbow_vals) * 100:.2f}%")
        print(f"Target coverage: {target_neg60_pct:.1f}%")
        print(f"Additional frames needed: {int(needed_frames_60)}")
        print(f"Estimated episodes to collect: {needed_episodes_60} (avg {avg_frames_per_ep:.0f} frames/ep)")
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

    print("\n[OK] Distribution analysis complete")

if __name__ == "__main__":
    analyze_distribution()
