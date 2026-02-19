"""
Analyze trajectory smoothness of collected data to compare collection methods.

Metrics:
- Frame-to-frame delta (velocity proxy)
- Jerk (acceleration change, 2nd derivative)
- Smoothness score (lower is smoother)
- Gripper timing quality
"""

import os
import json
import numpy as np
from pathlib import Path


def analyze_trajectory_smoothness(episode_dir: Path):
    """Analyze trajectory quality for one episode."""
    metadata_path = episode_dir / "metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if len(metadata['frames']) < 10:
        return None

    # Extract angle trajectories
    angles = np.array([frame['angles'] for frame in metadata['frames']])  # Shape: (T, 6)

    # 1. Frame-to-frame deltas (velocity proxy)
    deltas = np.diff(angles, axis=0)  # Shape: (T-1, 6)
    mean_delta = np.mean(np.abs(deltas), axis=0)
    max_delta = np.max(np.abs(deltas), axis=0)
    std_delta = np.std(deltas, axis=0)

    # 2. Jerk (2nd derivative - acceleration change)
    jerk = np.diff(deltas, axis=0)  # Shape: (T-2, 6)
    mean_jerk = np.mean(np.abs(jerk), axis=0)
    max_jerk = np.max(np.abs(jerk), axis=0)

    # 3. Overall smoothness score (lower is better)
    # Sum of std(delta) + mean(abs(jerk))
    smoothness_score = np.sum(std_delta) + np.sum(mean_jerk)

    # 4. Gripper analysis
    gripper_angles = angles[:, 5]
    gripper_delta = np.diff(gripper_angles)
    gripper_opens = np.sum(gripper_delta > 5)  # Opening events
    gripper_closes = np.sum(gripper_delta < -5)  # Closing events
    gripper_range = metadata.get('gripper_range', 0)

    # 5. Temporal consistency (how much time is spent in constant-velocity motion vs sharp turns)
    # Low jerk variance = more constant motion
    jerk_consistency = np.std(np.abs(jerk))

    return {
        'episode': episode_dir.name,
        'num_frames': len(metadata['frames']),
        'min_elbow': metadata.get('min_elbow', 0),

        # Delta metrics (velocity)
        'mean_delta': mean_delta,
        'max_delta': max_delta,
        'std_delta': std_delta,

        # Jerk metrics (acceleration change)
        'mean_jerk': mean_jerk,
        'max_jerk': max_jerk,

        # Overall quality
        'smoothness_score': smoothness_score,
        'jerk_consistency': jerk_consistency,

        # Gripper
        'gripper_range': gripper_range,
        'gripper_opens': gripper_opens,
        'gripper_closes': gripper_closes,

        # Joint correlation (are joints moving together?)
        'joint_correlation': np.corrcoef(angles.T),
    }


def main():
    data_dir = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

    results = []

    # Analyze all episodes
    for episode_dir in sorted(data_dir.glob("episode_*")):
        result = analyze_trajectory_smoothness(episode_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("No episodes found!")
        return

    # Aggregate statistics
    print("="*80)
    print("TRAJECTORY SMOOTHNESS ANALYSIS (Manual Collection - Torque OFF + Hand)")
    print("="*80)
    print(f"\nTotal episodes analyzed: {len(results)}")

    # Smoothness scores
    smoothness_scores = [r['smoothness_score'] for r in results]
    print(f"\nSmoothness Score (lower = better):")
    print(f"  Mean: {np.mean(smoothness_scores):.2f}")
    print(f"  Std:  {np.std(smoothness_scores):.2f}")
    print(f"  Min:  {np.min(smoothness_scores):.2f}")
    print(f"  Max:  {np.max(smoothness_scores):.2f}")

    # Per-joint delta analysis
    joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist_P', 'Wrist_R', 'Gripper']
    print(f"\n{'Joint':<10} {'Mean Δ':<10} {'Max Δ':<10} {'Std Δ':<10} {'Mean Jerk':<12}")
    print("-" * 60)
    for j, name in enumerate(joint_names):
        mean_delta = np.mean([r['mean_delta'][j] for r in results])
        max_delta = np.mean([r['max_delta'][j] for r in results])  # avg of max deltas
        std_delta = np.mean([r['std_delta'][j] for r in results])
        mean_jerk = np.mean([r['mean_jerk'][j] for r in results])
        print(f"{name:<10} {mean_delta:<10.2f} {max_delta:<10.2f} {std_delta:<10.2f} {mean_jerk:<12.2f}")

    # Jerk consistency
    jerk_consistency_vals = [r['jerk_consistency'] for r in results]
    print(f"\nJerk Consistency (lower = more constant motion):")
    print(f"  Mean: {np.mean(jerk_consistency_vals):.2f}")
    print(f"  Std:  {np.std(jerk_consistency_vals):.2f}")

    # Gripper usage
    gripper_ranges = [r['gripper_range'] for r in results]
    gripper_opens = [r['gripper_opens'] for r in results]
    gripper_closes = [r['gripper_closes'] for r in results]
    print(f"\nGripper Usage:")
    print(f"  Avg Range: {np.mean(gripper_ranges):.2f}°")
    print(f"  Avg Opens: {np.mean(gripper_opens):.2f}")
    print(f"  Avg Closes: {np.mean(gripper_closes):.2f}")

    # Find best and worst episodes
    best_idx = np.argmin(smoothness_scores)
    worst_idx = np.argmax(smoothness_scores)
    print(f"\nBest (smoothest) episode: {results[best_idx]['episode']} (score={smoothness_scores[best_idx]:.2f})")
    print(f"Worst (jerkiest) episode: {results[worst_idx]['episode']} (score={smoothness_scores[worst_idx]:.2f})")

    # Joint correlation analysis (sample from first 10 episodes)
    print(f"\nJoint Movement Correlation (first 10 episodes):")
    print("(1.0 = perfect positive correlation, 0.0 = no correlation, -1.0 = negative)")
    for i, result in enumerate(results[:10]):
        corr = result['joint_correlation']
        # Show correlation between elbow and shoulder (key for coordinated grasping)
        elbow_shoulder_corr = corr[2, 1]
        print(f"  {result['episode']}: Elbow-Shoulder corr = {elbow_shoulder_corr:.3f}")

    print("="*80)


if __name__ == "__main__":
    main()
