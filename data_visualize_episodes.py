"""
Episode Quality Visualization (requires matplotlib)
Run this after installing matplotlib to generate visual reports.

Installation:
    E:\RoArm_Project\.venv\Scripts\pip.exe install matplotlib seaborn

Usage:
    E:\RoArm_Project\.venv\Scripts\python.exe data_visualize_episodes.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_episodes():
    """Create comprehensive episode quality visualizations."""

    # Load episode quality results
    csv_path = Path("E:/RoArm_Project/data_episode_quality_results.csv")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} episodes from {csv_path}")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RoArm M3 Episode Quality Analysis (51 Episodes)', fontsize=16, fontweight='bold')

    # Plot 1: Episode quality scatter (elbow vs gripper)
    ax1 = axes[0, 0]
    good_eps = df[df['quality'] == 'GOOD']
    bad_eps = df[df['quality'] == 'BAD']

    ax1.scatter(good_eps['min_elbow'], good_eps['max_gripper'],
                c='green', s=100, alpha=0.6, label=f'GOOD ({len(good_eps)})', edgecolors='black')
    ax1.scatter(bad_eps['min_elbow'], bad_eps['max_gripper'],
                c='red', s=100, alpha=0.6, label=f'BAD ({len(bad_eps)})', marker='x', linewidths=2)

    # Mark gold standard episodes
    gold_episodes = [0, 4, 11, 19]
    gold_data = df[df['episode'].isin(gold_episodes)]
    ax1.scatter(gold_data['min_elbow'], gold_data['max_gripper'],
                c='gold', s=200, alpha=0.9, label='Gold Standard', edgecolors='black', linewidths=2, marker='*')

    # Add threshold lines
    ax1.axvline(-20, color='orange', linestyle='--', linewidth=2, label='Min Elbow Threshold (-20°)')
    ax1.axhline(30, color='blue', linestyle='--', linewidth=2, label='Gripper Threshold (30°)')

    ax1.set_xlabel('Min Elbow (degrees)', fontsize=11)
    ax1.set_ylabel('Max Gripper Opening (degrees)', fontsize=11)
    ax1.set_title('Episode Quality: Elbow Reach vs Gripper Opening', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Elbow range distribution
    ax2 = axes[0, 1]

    colors = ['green' if q == 'GOOD' else 'red' for q in df['quality']]
    bars = ax2.barh(df['episode'], df['elbow_range'], color=colors, alpha=0.6, edgecolor='black')

    ax2.axvline(40, color='orange', linestyle='--', linewidth=2, label='Range Threshold (40°)')
    ax2.set_xlabel('Elbow Range (degrees)', fontsize=11)
    ax2.set_ylabel('Episode Index', fontsize=11)
    ax2.set_title('Elbow Range per Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Min elbow distribution histogram
    ax3 = axes[1, 0]

    # Create histogram with color coding
    n, bins, patches = ax3.hist(df['min_elbow'], bins=30, alpha=0.7, edgecolor='black')

    # Color code bins
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -60:
            patch.set_facecolor('darkgreen')
        elif bin_center < -40:
            patch.set_facecolor('green')
        elif bin_center < -20:
            patch.set_facecolor('yellow')
        elif bin_center < 0:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')

    # Add threshold lines
    ax3.axvline(-60, color='darkgreen', linestyle='--', linewidth=2, label='Deep Grasp (-60°)')
    ax3.axvline(-40, color='green', linestyle='--', linewidth=2, label='Grasp Zone (-40°)')
    ax3.axvline(-20, color='yellow', linestyle='--', linewidth=2, label='Approach (-20°)')
    ax3.axvline(0, color='orange', linestyle='--', linewidth=2, label='Forward (0°)')

    # Mark mean
    mean_min_elbow = df['min_elbow'].mean()
    ax3.axvline(mean_min_elbow, color='black', linestyle='-', linewidth=2.5,
                label=f'Mean: {mean_min_elbow:.1f}°')

    ax3.set_xlabel('Min Elbow per Episode (degrees)', fontsize=11)
    ax3.set_ylabel('Episode Count', fontsize=11)
    ax3.set_title('Distribution of Minimum Elbow Angles', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Episode timeline (min elbow over episode index)
    ax4 = axes[1, 1]

    # Color points by quality
    good_mask = df['quality'] == 'GOOD'
    ax4.scatter(df[good_mask]['episode'], df[good_mask]['min_elbow'],
                c='green', s=80, alpha=0.7, label='GOOD', edgecolors='black')
    ax4.scatter(df[~good_mask]['episode'], df[~good_mask]['min_elbow'],
                c='red', s=80, alpha=0.7, label='BAD', marker='x', linewidths=2)

    # Mark gold episodes
    ax4.scatter(gold_data['episode'], gold_data['min_elbow'],
                c='gold', s=200, alpha=0.9, label='Gold Standard', edgecolors='black', linewidths=2, marker='*')

    # Add threshold lines
    ax4.axhline(-60, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7, label='Deep Grasp (-60°)')
    ax4.axhline(-40, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Grasp Zone (-40°)')
    ax4.axhline(-20, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='Approach (-20°)')
    ax4.axhline(0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Forward (0°)')

    # Mean line
    ax4.axhline(mean_min_elbow, color='black', linestyle='-', linewidth=2, label=f'Mean: {mean_min_elbow:.1f}°')

    ax4.set_xlabel('Episode Index', fontsize=11)
    ax4.set_ylabel('Min Elbow (degrees)', fontsize=11)
    ax4.set_title('Episode Quality Over Collection Timeline', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("E:/RoArm_Project/data_episode_quality_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to {output_path}")
    plt.close()

    # Create summary statistics figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create text summary
    summary_text = f"""
RoArm M3 Dataset Quality Summary
================================

Total Episodes: {len(df)}
Good Episodes: {(df['quality'] == 'GOOD').sum()} ({(df['quality'] == 'GOOD').sum() / len(df) * 100:.1f}%)
Bad Episodes: {(df['quality'] == 'BAD').sum()} ({(df['quality'] == 'BAD').sum() / len(df) * 100:.1f}%)

Elbow Statistics:
  Mean min elbow: {df['min_elbow'].mean():.1f}°
  Mean max elbow: {df['max_elbow'].mean():.1f}°
  Mean elbow range: {df['elbow_range'].mean():.1f}°

Episode Coverage by Elbow Zone:
  < -60° (deep grasp):     {(df['min_elbow'] < -60).sum():2d} episodes ({(df['min_elbow'] < -60).sum() / len(df) * 100:4.1f}%)
  -60° to -40° (grasp):    {((df['min_elbow'] >= -60) & (df['min_elbow'] < -40)).sum():2d} episodes ({((df['min_elbow'] >= -60) & (df['min_elbow'] < -40)).sum() / len(df) * 100:4.1f}%)
  -40° to -20° (approach): {((df['min_elbow'] >= -40) & (df['min_elbow'] < -20)).sum():2d} episodes ({((df['min_elbow'] >= -40) & (df['min_elbow'] < -20)).sum() / len(df) * 100:4.1f}%)
  -20° to 0° (forward):    {((df['min_elbow'] >= -20) & (df['min_elbow'] < 0)).sum():2d} episodes ({((df['min_elbow'] >= -20) & (df['min_elbow'] < 0)).sum() / len(df) * 100:4.1f}%)
  > 0° (never forward):    {(df['min_elbow'] >= 0).sum():2d} episodes ({(df['min_elbow'] >= 0).sum() / len(df) * 100:4.1f}%)

Gripper Statistics:
  Mean max gripper: {df['max_gripper'].mean():.1f}°
  Episodes with gripper > 30°: {(df['max_gripper'] > 30).sum()} ({(df['max_gripper'] > 30).sum() / len(df) * 100:.1f}%)
  Episodes with gripper > 50°: {(df['max_gripper'] > 50).sum()} ({(df['max_gripper'] > 50).sum() / len(df) * 100:.1f}%)

Gold Standard Episodes (elbow < -60°):
"""

    # Add gold episodes details
    for _, row in df[df['episode'].isin(gold_episodes)].iterrows():
        summary_text += f"  Episode {row['episode']:2d}: min_elbow={row['min_elbow']:6.1f}°, max_gripper={row['max_gripper']:4.1f}°\n"

    summary_text += f"""
Worst Episodes (never forward):
"""

    # Add worst episodes
    worst_eps = df.nlargest(5, 'min_elbow')
    for _, row in worst_eps.iterrows():
        summary_text += f"  Episode {row['episode']:2d}: min_elbow={row['min_elbow']:6.1f}°, range={row['elbow_range']:4.1f}°\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axis('off')
    plt.tight_layout()
    output_path = Path("E:/RoArm_Project/data_summary_text.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Summary text saved to {output_path}")
    plt.close()

    print("\n[OK] All visualizations complete")
    print("\nGenerated files:")
    print("  - data_episode_quality_visualization.png")
    print("  - data_summary_text.png")

if __name__ == "__main__":
    try:
        visualize_episodes()
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall matplotlib and seaborn:")
        print("  pip install matplotlib seaborn")
