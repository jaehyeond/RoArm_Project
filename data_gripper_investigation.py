#!/usr/bin/env python3
"""
Deep investigation into gripper action detection.
Analyzes gripper trajectories to understand if detection thresholds are too strict.
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd

def analyze_gripper_trajectory(episode_dir: Path) -> dict:
    """Analyze gripper movement pattern in detail."""
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    episode_id = metadata['episode_id']
    gripper_angles = np.array([frame['angles'][5] for frame in metadata['frames']])

    # Statistics
    min_grip = np.min(gripper_angles)
    max_grip = np.max(gripper_angles)
    mean_grip = np.mean(gripper_angles)
    std_grip = np.std(gripper_angles)
    range_grip = max_grip - min_grip

    # Movement analysis
    grip_diff = np.abs(np.diff(gripper_angles))
    max_single_move = np.max(grip_diff) if len(grip_diff) > 0 else 0
    total_movement = np.sum(grip_diff)

    # Detect large movements (potential grasp actions)
    large_moves = np.where(grip_diff > 5.0)[0]  # Movements > 5 degrees
    num_large_moves = len(large_moves)

    # Check for opening/closing pattern
    opens = np.where(grip_diff > 10.0)[0]  # Opening movements
    closes = np.where(grip_diff > 10.0)[0]  # Closing movements (same, just large moves)

    # Different threshold levels
    criteria_strict = (max_grip > 50) and (min_grip < 20)  # Current
    criteria_relaxed = (max_grip > 30) and (min_grip < 30)  # Relaxed
    criteria_minimal = range_grip > 20  # Just needs 20° range

    return {
        'episode_id': episode_id,
        'min': min_grip,
        'max': max_grip,
        'mean': mean_grip,
        'std': std_grip,
        'range': range_grip,
        'max_single_move': max_single_move,
        'total_movement': total_movement,
        'num_large_moves': num_large_moves,
        'has_gripping_strict': criteria_strict,
        'has_gripping_relaxed': criteria_relaxed,
        'has_gripping_minimal': criteria_minimal,
        'trajectory': gripper_angles.tolist()
    }

def main():
    dataset_dir = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")
    episode_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])

    print("\n" + "="*100)
    print("GRIPPER ACTION INVESTIGATION: Analyzing Detection Thresholds")
    print("="*100 + "\n")

    results = []
    for ep_dir in episode_dirs:
        result = analyze_gripper_trajectory(ep_dir)
        results.append(result)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'trajectory'} for r in results])

    # Summary statistics
    print("GRIPPER MOVEMENT STATISTICS (All 50 Episodes):")
    print("-" * 100)
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 100)
    print(f"{'Gripper min angle':<30} {df['min'].mean():>10.2f} {df['min'].std():>10.2f} "
          f"{df['min'].min():>10.2f} {df['min'].max():>10.2f}")
    print(f"{'Gripper max angle':<30} {df['max'].mean():>10.2f} {df['max'].std():>10.2f} "
          f"{df['max'].min():>10.2f} {df['max'].max():>10.2f}")
    print(f"{'Gripper range':<30} {df['range'].mean():>10.2f} {df['range'].std():>10.2f} "
          f"{df['range'].min():>10.2f} {df['range'].max():>10.2f}")
    print(f"{'Total movement':<30} {df['total_movement'].mean():>10.2f} {df['total_movement'].std():>10.2f} "
          f"{df['total_movement'].min():>10.2f} {df['total_movement'].max():>10.2f}")
    print(f"{'Max single movement':<30} {df['max_single_move'].mean():>10.2f} {df['max_single_move'].std():>10.2f} "
          f"{df['max_single_move'].min():>10.2f} {df['max_single_move'].max():>10.2f}")
    print("="*100)
    print()

    # Detection rate comparison
    print("DETECTION RATE COMPARISON:")
    print("-" * 100)
    strict_count = df['has_gripping_strict'].sum()
    relaxed_count = df['has_gripping_relaxed'].sum()
    minimal_count = df['has_gripping_minimal'].sum()

    print(f"Strict criteria (max>50° AND min<20°):    {strict_count:2d} / 50 ({strict_count/50*100:5.1f}%)")
    print(f"Relaxed criteria (max>30° AND min<30°):   {relaxed_count:2d} / 50 ({relaxed_count/50*100:5.1f}%)")
    print(f"Minimal criteria (range > 20°):           {minimal_count:2d} / 50 ({minimal_count/50*100:5.1f}%)")
    print("="*100)
    print()

    # Identify episodes with minimal gripper movement
    static_gripper = df[df['range'] < 5]
    minimal_movement = df[(df['range'] >= 5) & (df['range'] < 20)]
    moderate_movement = df[(df['range'] >= 20) & (df['range'] < 40)]
    good_movement = df[df['range'] >= 40]

    print("GRIPPER MOVEMENT CATEGORIES:")
    print("-" * 100)
    print(f"Static gripper (range < 5°):           {len(static_gripper):2d} episodes "
          f"({len(static_gripper)/50*100:5.1f}%)")
    print(f"Minimal movement (5° <= range < 20°):  {len(minimal_movement):2d} episodes "
          f"({len(minimal_movement)/50*100:5.1f}%)")
    print(f"Moderate movement (20° <= range < 40°): {len(moderate_movement):2d} episodes "
          f"({len(moderate_movement)/50*100:5.1f}%)")
    print(f"Good movement (range >= 40°):          {len(good_movement):2d} episodes "
          f"({len(good_movement)/50*100:5.1f}%)")
    print()

    if len(static_gripper) > 0:
        print(f"Static gripper episodes: {static_gripper['episode_id'].tolist()}")
    if len(good_movement) > 0:
        print(f"Good movement episodes: {good_movement['episode_id'].tolist()}")
    print("="*100)
    print()

    # Detailed look at episodes with different detection results
    print("DETAILED ANALYSIS: Episodes detected differently by criteria")
    print("-" * 100)

    # Episodes that pass relaxed but not strict
    relaxed_only = df[(~df['has_gripping_strict']) & (df['has_gripping_relaxed'])]
    if len(relaxed_only) > 0:
        print(f"\nEpisodes passing relaxed criteria only ({len(relaxed_only)} episodes):")
        print(f"{'Ep':<4} {'Min':>8} {'Max':>8} {'Range':>8} {'Total_Move':>11} {'Large_Moves':>11}")
        print("-" * 100)
        for _, row in relaxed_only.iterrows():
            print(f"{int(row['episode_id']):<4} {row['min']:>8.2f} {row['max']:>8.2f} "
                  f"{row['range']:>8.2f} {row['total_movement']:>11.2f} {int(row['num_large_moves']):>11}")

    # Episodes that pass minimal but not relaxed
    minimal_only = df[(~df['has_gripping_relaxed']) & (df['has_gripping_minimal'])]
    if len(minimal_only) > 0:
        print(f"\nEpisodes passing minimal criteria only ({len(minimal_only)} episodes):")
        print(f"{'Ep':<4} {'Min':>8} {'Max':>8} {'Range':>8} {'Total_Move':>11} {'Large_Moves':>11}")
        print("-" * 100)
        for _, row in minimal_only.iterrows():
            print(f"{int(row['episode_id']):<4} {row['min']:>8.2f} {row['max']:>8.2f} "
                  f"{row['range']:>8.2f} {row['total_movement']:>11.2f} {int(row['num_large_moves']):>11}")

    print("="*100)
    print()

    # Inspect specific episodes that passed strict criteria
    strict_episodes = df[df['has_gripping_strict']]
    if len(strict_episodes) > 0:
        print(f"EPISODES PASSING STRICT CRITERIA ({len(strict_episodes)} episodes):")
        print("-" * 100)
        print(f"{'Ep':<4} {'Min':>8} {'Max':>8} {'Range':>8} {'Mean':>8} {'Std':>8} {'Total_Move':>11}")
        print("-" * 100)
        for _, row in strict_episodes.iterrows():
            print(f"{int(row['episode_id']):<4} {row['min']:>8.2f} {row['max']:>8.2f} "
                  f"{row['range']:>8.2f} {row['mean']:>8.2f} {row['std']:>8.2f} "
                  f"{row['total_movement']:>11.2f}")
        print()

        # Show trajectory for first strict episode
        first_strict_ep = int(strict_episodes.iloc[0]['episode_id'])
        trajectory = results[first_strict_ep]['trajectory']
        print(f"Gripper trajectory for Episode {first_strict_ep} (first strict pass):")
        print(f"  Frames: {len(trajectory)}")
        print(f"  Start: {trajectory[0]:.2f}°, End: {trajectory[-1]:.2f}°")
        print(f"  Min: {min(trajectory):.2f}°, Max: {max(trajectory):.2f}°")

        # Find closing action
        traj_arr = np.array(trajectory)
        diff = np.diff(traj_arr)
        large_closes = np.where(diff < -10)[0]  # Large negative changes
        large_opens = np.where(diff > 10)[0]   # Large positive changes

        if len(large_closes) > 0:
            print(f"  Closing actions at frames: {large_closes.tolist()}")
        if len(large_opens) > 0:
            print(f"  Opening actions at frames: {large_opens.tolist()}")

    print("="*100)
    print()

    # DIAGNOSIS
    print("="*100)
    print("DIAGNOSIS AND RECOMMENDATIONS:")
    print("="*100)
    print()

    avg_range = df['range'].mean()
    median_range = df['range'].median()

    if avg_range < 10:
        print("FINDING: Gripper is BARELY MOVING across most episodes")
        print("DIAGNOSIS: Data collection likely NOT including gripper actuation")
        print()
        print("CAUSE: The collection script (collect_data_manual.py) may not be:")
        print("  1. Recording gripper joint properly")
        print("  2. Prompting user to open/close gripper during recording")
        print("  3. User is not moving the gripper during manual teleoperation")
        print()
        print("REQUIRED ACTION: Re-collect data with explicit gripper actuation")

    elif avg_range < 30:
        print("FINDING: Gripper has SOME movement but insufficient for clear grasping")
        print("DIAGNOSIS: Gripper IS being recorded, but motion is too subtle")
        print()
        print("CAUSE:")
        print("  - User may not be fully opening/closing gripper during teleoperation")
        print("  - Gripper mechanical limits may be preventing full range")
        print()
        print("RECOMMENDED ACTION:")
        print("  1. Re-collect with EXPLICIT instruction to:")
        print("     - Start each episode with gripper FULLY OPEN (>50°)")
        print("     - Close gripper FULLY on object (<20°)")
        print("  2. Verify gripper mechanical range before collection")

    elif strict_count < 10:
        print("FINDING: Gripper HAS adequate movement, but detection is too strict")
        print(f"DIAGNOSIS: {relaxed_count} episodes would pass with relaxed criteria")
        print()
        print("RECOMMENDED ACTION:")
        print("  1. Use relaxed criteria (max>30° AND min<30°) for analysis")
        print(f"  2. This would increase detection rate to {relaxed_count/50*100:.1f}%")
        print("  3. Re-run comprehensive analysis with new thresholds")
        print()
        print("However, even with relaxed criteria:")
        if relaxed_count / 50 < 0.8:
            print(f"  WARNING: Only {relaxed_count/50*100:.1f}% would pass")
            print("  This suggests MOST episodes still lack proper gripping")
            print("  RECOMMENDATION: Still need to re-collect with better gripper actuation")

    else:
        print("FINDING: Gripper detection is working correctly")
        print(f"DIAGNOSIS: {strict_count} episodes have proper gripping action")
        print()
        print("RECOMMENDATION: Current episodes are good, just need more volume")

    print("="*100)
    print()

    # Save detailed results
    output_csv = dataset_dir / "gripper_analysis_detailed.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed gripper analysis saved to: {output_csv}")

if __name__ == "__main__":
    main()
