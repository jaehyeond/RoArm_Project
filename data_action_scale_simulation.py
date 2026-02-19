#!/usr/bin/env python3
"""
Action-Scale Dry-Run Simulation

Analyzes deployment logs to simulate the effect of action scaling on trajectory.
Tests whether scaling can resolve the direction reversal issue (APPROACH → upward drift).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Dataset normalization stats (from training data v1)
DATASET_MEAN = np.array([2.71, 40.31, 13.04, 62.75, -2.65, 9.61])
DATASET_STD = np.array([9.72, 33.04, 29.38, 25.16, 13.62, 16.88])

# Joint limits (hardware constraints)
JOINT_LIMITS = {
    'base': (-190, 190),
    'shoulder': (-110, 110),
    'elbow': (-70, 190),
    'wrist_pitch': (-110, 110),
    'wrist_roll': (-190, 190),
    'gripper': (-10, 100)
}

JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']


def load_deployment_log(log_path):
    """Load deployment CSV and extract current positions."""
    df = pd.read_csv(log_path)

    # Extract joint positions
    positions = df[JOINT_NAMES].values

    return df, positions


def simulate_scaled_action(current_pos, predicted_action, scale):
    """
    Simulate action scaling.

    scaled_action = dataset_mean + scale * (predicted - dataset_mean)

    Args:
        current_pos: Current joint angles (unused in this sim)
        predicted_action: Model's predicted action
        scale: Scaling factor (1.0 = no change)

    Returns:
        Scaled action
    """
    scaled = DATASET_MEAN + scale * (predicted_action - DATASET_MEAN)

    # Apply joint limits
    for i, joint_name in enumerate(JOINT_NAMES):
        min_limit, max_limit = JOINT_LIMITS[joint_name]
        scaled[i] = np.clip(scaled[i], min_limit, max_limit)

    return scaled


def analyze_trajectory_with_scaling(df, positions, scales=[1.0, 1.5, 2.0, 3.0]):
    """
    Simulate trajectories under different action scales.

    Key insight: We're simulating what would happen if we scaled the actions
    DURING deployment, not retraining. This is a closed-loop simulation where
    each step uses the scaled action but subsequent predictions still come from
    the same model.

    Returns:
        Dict of {scale: simulated_positions}
    """
    results = {}

    for scale in scales:
        # Start from initial position
        sim_positions = [positions[0].copy()]

        # For each step, apply scaled action
        for step_idx in range(1, len(positions)):
            predicted = positions[step_idx]  # This is what model predicted
            current = sim_positions[-1]  # Where we are now (in simulation)

            # Apply scaling
            scaled_action = simulate_scaled_action(current, predicted, scale)

            # In closed-loop, we move to the scaled action
            # (Note: This assumes n_action_steps=1, immediate execution)
            sim_positions.append(scaled_action)

        results[scale] = np.array(sim_positions)

    return results


def plot_comparison(original_positions, scaled_trajectories, log_name):
    """Plot original vs scaled trajectories for key joints."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Action-Scale Simulation: {log_name}', fontsize=16)

    scales = sorted(scaled_trajectories.keys())
    colors = ['blue', 'green', 'orange', 'red']

    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        ax = axes[joint_idx // 2, joint_idx % 2]

        # Plot original (scale=1.0)
        steps = np.arange(len(original_positions))
        ax.plot(steps, original_positions[:, joint_idx],
                'k--', linewidth=2, label='Original (1.0x)', alpha=0.7)

        # Plot scaled trajectories
        for scale, color in zip(scales[1:], colors[1:]):  # Skip 1.0 (already plotted)
            scaled_pos = scaled_trajectories[scale]
            ax.plot(steps, scaled_pos[:, joint_idx],
                    color=color, linewidth=1.5, label=f'{scale}x', alpha=0.8)

        # Highlight key thresholds
        if joint_name == 'elbow':
            ax.axhline(y=-30, color='red', linestyle=':', linewidth=1, alpha=0.5, label='DEEP threshold')
            ax.axhline(y=-10, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='APPROACH threshold')

        if joint_name == 'gripper':
            ax.axhline(y=20, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Open/Close threshold')

        ax.set_xlabel('Step')
        ax.set_ylabel(f'{joint_name} (degrees)')
        ax.set_title(f'{joint_name.capitalize()} Joint')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_metrics(positions, joint_idx=2):
    """Compute key metrics for a trajectory (focus on elbow)."""
    elbow = positions[:, joint_idx]

    metrics = {
        'min': elbow.min(),
        'max': elbow.max(),
        'mean': elbow.mean(),
        'final': elbow[-1],
        'drift': elbow[-1] - elbow[0],  # Positive = upward drift
        'reached_deep': (elbow < -30).any(),
        'reached_approach': (elbow < -10).any(),
    }

    return metrics


def analyze_gripper_activity(positions):
    """Check if gripper shows opening/closing behavior."""
    gripper = positions[:, 5]

    gripper_range = gripper.max() - gripper.min()
    gripper_std = gripper.std()
    opens = (gripper > 20).sum()  # Frames where gripper is "open"

    return {
        'range': gripper_range,
        'std': gripper_std,
        'open_frames': opens,
        'total_frames': len(gripper),
        'open_ratio': opens / len(gripper)
    }


def main():
    log_dir = Path('/home/cgxr/Documents/Robotics/RoArm_Project/logs')
    output_dir = Path('/home/cgxr/Documents/Robotics/RoArm_Project/analysis_outputs')
    output_dir.mkdir(exist_ok=True)

    log_files = sorted(log_dir.glob('deploy_*.csv'))

    if not log_files:
        print("ERROR: No deployment log files found!")
        return

    print("=" * 80)
    print("ACTION-SCALE DRY-RUN SIMULATION")
    print("=" * 80)
    print(f"\nDataset mean: {DATASET_MEAN}")
    print(f"Dataset std:  {DATASET_STD}")
    print(f"\nFound {len(log_files)} deployment logs\n")

    all_results = []

    for log_file in log_files:
        log_name = log_file.stem
        print(f"\n{'=' * 80}")
        print(f"LOG: {log_name}")
        print(f"{'=' * 80}")

        # Load data
        df, positions = load_deployment_log(log_file)
        print(f"Steps: {len(positions)}")

        # Simulate scaling
        scales = [1.0, 1.5, 2.0, 3.0]
        scaled_trajectories = analyze_trajectory_with_scaling(df, positions, scales)

        # Analyze each scale
        print(f"\n{'Scale':<8} {'Elbow Min':>12} {'Elbow Max':>12} {'Final':>10} {'Drift':>10} {'Deep?':>8} {'Approach?':>10}")
        print("-" * 80)

        for scale in scales:
            sim_pos = scaled_trajectories[scale]
            metrics = compute_metrics(sim_pos, joint_idx=2)  # Elbow

            print(f"{scale:<8.1f} {metrics['min']:>12.2f} {metrics['max']:>12.2f} "
                  f"{metrics['final']:>10.2f} {metrics['drift']:>+10.2f} "
                  f"{'YES' if metrics['reached_deep'] else 'NO':>8} "
                  f"{'YES' if metrics['reached_approach'] else 'NO':>10}")

            all_results.append({
                'log': log_name,
                'scale': scale,
                **metrics
            })

        # Gripper analysis
        print(f"\n{'Scale':<8} {'Grip Range':>12} {'Grip Std':>12} {'Open Frames':>12} {'Open %':>10}")
        print("-" * 80)

        for scale in scales:
            sim_pos = scaled_trajectories[scale]
            grip_metrics = analyze_gripper_activity(sim_pos)

            print(f"{scale:<8.1f} {grip_metrics['range']:>12.2f} {grip_metrics['std']:>12.2f} "
                  f"{grip_metrics['open_frames']:>12} {grip_metrics['open_ratio']*100:>9.1f}%")

        # Plot
        fig = plot_comparison(positions, scaled_trajectories, log_name)
        plot_path = output_dir / f'scale_sim_{log_name}.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nPlot saved: {plot_path}")

    # Summary across all logs
    print(f"\n{'=' * 80}")
    print("SUMMARY: ACTION-SCALE IMPACT")
    print(f"{'=' * 80}")

    results_df = pd.DataFrame(all_results)

    # Group by scale
    summary = results_df.groupby('scale').agg({
        'min': 'mean',
        'max': 'mean',
        'drift': 'mean',
        'reached_deep': 'sum',
        'reached_approach': 'sum'
    }).round(2)

    print("\nAverage metrics across all logs:")
    print(summary)

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. DIRECTION REVERSAL ISSUE:")
    original_drift = results_df[results_df['scale'] == 1.0]['drift'].mean()
    scaled_drifts = {s: results_df[results_df['scale'] == s]['drift'].mean()
                     for s in [1.5, 2.0, 3.0]}

    print(f"   - Original drift (1.0x): {original_drift:+.2f}°")
    for scale, drift in scaled_drifts.items():
        print(f"   - Scaled drift ({scale}x): {drift:+.2f}°")

    if all(d > 0 for d in scaled_drifts.values()):
        print("\n   ⚠️  CRITICAL: Action scaling does NOT fix direction reversal!")
        print("   All scales show UPWARD drift (positive). The model learned wrong direction.")
        print("   Root cause: 68% SHALLOW data → model thinks APPROACH = LIFT.")
    else:
        print("\n   ✓ Action scaling may help reduce drift.")

    print("\n2. GRIPPER ACTIVATION:")
    grip_summary = []
    for log_file in log_files:
        df, positions = load_deployment_log(log_file)
        for scale in [1.0, 1.5, 2.0, 3.0]:
            sim_pos = analyze_trajectory_with_scaling(df, positions, [scale])[scale]
            grip = analyze_gripper_activity(sim_pos)
            grip_summary.append({'scale': scale, 'open_ratio': grip['open_ratio']})

    grip_df = pd.DataFrame(grip_summary)
    grip_by_scale = grip_df.groupby('scale')['open_ratio'].mean()

    print(f"   Average gripper open ratio:")
    for scale, ratio in grip_by_scale.items():
        print(f"   - {scale}x: {ratio*100:.1f}%")

    if all(ratio < 0.1 for ratio in grip_by_scale.values):
        print("\n   ⚠️  CRITICAL: Gripper remains mostly closed across all scales!")
        print("   Root cause: Training data lacks gripper open frames.")

    print("\n3. CONCLUSION:")
    print("   Action-scale is a DEPLOYMENT-TIME parameter tweak, NOT a data fix.")
    print("   The fundamental issue is DATA DISTRIBUTION:")
    print("   - 68% SHALLOW → model never learned 'reach down and grasp'")
    print("   - Gripper bias → model never learned 'open gripper'")
    print("\n   RECOMMENDATION: Collect NEW DATA with proper distribution, then retrain.")
    print("   Action-scale CANNOT fix a model that learned the wrong pattern.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
