#!/usr/bin/env python3
"""
Comprehensive dataset quality analysis for 50 episodes.
Analyzes elbow depth, joint distributions, frame counts, gripper patterns, and RGB validity.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

# Joint names
JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist_pitch', 'Wrist_roll', 'Gripper']

def load_episode_metadata(episode_dir: Path) -> Dict:
    """Load metadata.json from episode directory."""
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)

def analyze_single_episode(episode_dir: Path) -> Dict:
    """Analyze a single episode comprehensively."""
    metadata = load_episode_metadata(episode_dir)
    episode_id = metadata['episode_id']
    num_frames = metadata['num_frames']

    # Extract all joint angles
    angles_array = np.array([frame['angles'] for frame in metadata['frames']])  # (num_frames, 6)

    # Elbow analysis (joint 2)
    elbow_angles = angles_array[:, 2]
    min_elbow = np.min(elbow_angles)
    max_elbow = np.max(elbow_angles)
    mean_elbow = np.mean(elbow_angles)
    elbow_range = max_elbow - min_elbow

    # Quality grade based on minimum elbow
    if min_elbow < -30:
        quality_grade = "DEEP"
    elif -30 <= min_elbow < -10:
        quality_grade = "APPROACH"
    else:
        quality_grade = "SHALLOW"

    # Gripper analysis (joint 5)
    gripper_angles = angles_array[:, 5]
    gripper_min = np.min(gripper_angles)
    gripper_max = np.max(gripper_angles)
    gripper_range = gripper_max - gripper_min
    gripper_opened = gripper_max > 50  # Gripper is considered opened if > 50 degrees
    gripper_closed = gripper_min < 20  # Gripper is considered closed if < 20 degrees
    has_gripping = gripper_opened and gripper_closed  # Both opening and closing

    # Joint movement analysis (detect static episodes)
    joint_ranges = np.max(angles_array, axis=0) - np.min(angles_array, axis=0)
    total_movement = np.sum(joint_ranges)
    is_static = total_movement < 10  # Less than 10 degrees total movement across all joints

    # Timestamp analysis
    timestamps = [frame['timestamp'] for frame in metadata['frames']]
    duration = timestamps[-1] - timestamps[0]
    frame_intervals = np.diff(timestamps)
    mean_fps = len(timestamps) / duration if duration > 0 else 0
    fps_std = np.std(frame_intervals) if len(frame_intervals) > 1 else 0

    # RGB validation (check first and last frame)
    first_rgb_path = episode_dir / metadata['frames'][0]['rgb_path']
    last_rgb_path = episode_dir / metadata['frames'][-1]['rgb_path']

    rgb_valid = True
    rgb_issues = []

    try:
        first_img = np.array(Image.open(first_rgb_path))
        last_img = np.array(Image.open(last_rgb_path))

        # Check shape
        if first_img.shape != last_img.shape:
            rgb_valid = False
            rgb_issues.append("shape_mismatch")

        # Check brightness (mean pixel value)
        first_brightness = np.mean(first_img)
        last_brightness = np.mean(last_img)

        if first_brightness < 10 or last_brightness < 10:
            rgb_valid = False
            rgb_issues.append("too_dark")

        if first_brightness > 245 or last_brightness > 245:
            rgb_valid = False
            rgb_issues.append("too_bright")

    except Exception as e:
        rgb_valid = False
        rgb_issues.append(f"load_error: {str(e)}")

    # Anomaly detection
    anomalies = []
    if num_frames < 10:
        anomalies.append("too_short")
    if num_frames > 500:
        anomalies.append("too_long")
    if is_static:
        anomalies.append("static")
    if not has_gripping:
        anomalies.append("no_gripping")
    if elbow_range < 5:
        anomalies.append("no_elbow_movement")

    return {
        'episode_id': episode_id,
        'num_frames': num_frames,
        'duration_sec': duration,
        'mean_fps': mean_fps,
        'fps_std': fps_std,
        'min_elbow': min_elbow,
        'max_elbow': max_elbow,
        'mean_elbow': mean_elbow,
        'elbow_range': elbow_range,
        'quality_grade': quality_grade,
        'gripper_min': gripper_min,
        'gripper_max': gripper_max,
        'gripper_range': gripper_range,
        'has_gripping': has_gripping,
        'total_movement': total_movement,
        'is_static': is_static,
        'rgb_valid': rgb_valid,
        'rgb_issues': ','.join(rgb_issues) if rgb_issues else 'none',
        'anomalies': ','.join(anomalies) if anomalies else 'none',
        'angles_array': angles_array,  # For later global analysis
    }

def analyze_all_episodes(dataset_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Analyze all episodes and return summary DataFrame and global angles array."""
    episode_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])

    print(f"Found {len(episode_dirs)} episodes\n")

    results = []
    all_angles = []

    for ep_dir in episode_dirs:
        result = analyze_single_episode(ep_dir)
        angles_array = result.pop('angles_array')  # Remove from dict for DataFrame
        results.append(result)
        all_angles.append(angles_array)

    df = pd.DataFrame(results)

    # Concatenate all angles for global analysis
    global_angles = np.vstack(all_angles)

    return df, global_angles

def print_episode_summary_table(df: pd.DataFrame):
    """Print per-episode summary table."""
    print("=" * 120)
    print("EPISODE SUMMARY TABLE (All 50 Episodes)")
    print("=" * 120)
    print(f"{'Ep':<4} {'Frames':<7} {'Duration':<9} {'Min_Elbow':<11} {'Max_Elbow':<11} {'Elbow_Range':<12} "
          f"{'Grade':<9} {'Gripping':<9} {'Anomalies':<20}")
    print("-" * 120)

    for _, row in df.iterrows():
        print(f"{row['episode_id']:<4} {row['num_frames']:<7} {row['duration_sec']:>8.2f}s "
              f"{row['min_elbow']:>10.2f}° {row['max_elbow']:>10.2f}° {row['elbow_range']:>11.2f}° "
              f"{row['quality_grade']:<9} {str(row['has_gripping']):<9} {row['anomalies']:<20}")

    print("=" * 120)
    print()

def print_elbow_distribution(df: pd.DataFrame):
    """Print elbow depth distribution analysis."""
    print("=" * 80)
    print("ELBOW DEPTH DISTRIBUTION ANALYSIS")
    print("=" * 80)

    deep = df[df['quality_grade'] == 'DEEP']
    approach = df[df['quality_grade'] == 'APPROACH']
    shallow = df[df['quality_grade'] == 'SHALLOW']

    total = len(df)

    print(f"DEEP (< -30°):        {len(deep):2d} episodes ({len(deep)/total*100:5.1f}%)")
    print(f"APPROACH (-30~-10°):  {len(approach):2d} episodes ({len(approach)/total*100:5.1f}%)")
    print(f"SHALLOW (> -10°):     {len(shallow):2d} episodes ({len(shallow)/total*100:5.1f}%)")
    print(f"Total:                {total:2d} episodes")
    print()

    print("Comparison with previous failed dataset:")
    print("  Previous: 51 episodes, DEEP = 2 (3.9%)")
    print(f"  Current:  {total} episodes, DEEP = {len(deep)} ({len(deep)/total*100:.1f}%)")

    if len(deep) / total < 0.5:
        print("  WARNING: DEEP episodes < 50%, may lead to similar failure!")
    else:
        print("  GOOD: DEEP episodes >= 50%, much better than previous dataset")

    print()
    print("DEEP episodes list:", deep['episode_id'].tolist())
    print("=" * 80)
    print()

def print_joint_distribution(global_angles: np.ndarray):
    """Print global joint distribution statistics."""
    print("=" * 80)
    print("GLOBAL JOINT DISTRIBUTION (All Frames)")
    print("=" * 80)

    print(f"{'Joint':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 80)

    for i, joint_name in enumerate(JOINT_NAMES):
        joint_angles = global_angles[:, i]
        mean = np.mean(joint_angles)
        std = np.std(joint_angles)
        min_val = np.min(joint_angles)
        max_val = np.max(joint_angles)
        range_val = max_val - min_val

        print(f"{joint_name:<15} {mean:>10.2f} {std:>10.2f} {min_val:>10.2f} {max_val:>10.2f} {range_val:>10.2f}")

    print("=" * 80)
    print()

def print_frame_count_analysis(df: pd.DataFrame):
    """Print frame count distribution analysis."""
    print("=" * 80)
    print("FRAME COUNT DISTRIBUTION")
    print("=" * 80)

    print(f"Mean frames per episode: {df['num_frames'].mean():.1f}")
    print(f"Std frames per episode:  {df['num_frames'].std():.1f}")
    print(f"Min frames:              {df['num_frames'].min()}")
    print(f"Max frames:              {df['num_frames'].max()}")
    print()

    too_short = df[df['num_frames'] < 10]
    too_long = df[df['num_frames'] > 500]

    if len(too_short) > 0:
        print(f"WARNING: {len(too_short)} episodes with < 10 frames:")
        print(f"  Episodes: {too_short['episode_id'].tolist()}")

    if len(too_long) > 0:
        print(f"WARNING: {len(too_long)} episodes with > 500 frames:")
        print(f"  Episodes: {too_long['episode_id'].tolist()}")

    print("=" * 80)
    print()

def print_time_analysis(df: pd.DataFrame):
    """Print temporal analysis."""
    print("=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    print(f"Mean episode duration:   {df['duration_sec'].mean():.2f}s")
    print(f"Std episode duration:    {df['duration_sec'].std():.2f}s")
    print(f"Min episode duration:    {df['duration_sec'].min():.2f}s")
    print(f"Max episode duration:    {df['duration_sec'].max():.2f}s")
    print()
    print(f"Mean FPS:                {df['mean_fps'].mean():.2f}")
    print(f"Mean FPS std deviation:  {df['fps_std'].mean():.4f}s")

    print("=" * 80)
    print()

def print_anomaly_detection(df: pd.DataFrame):
    """Print anomaly detection results."""
    print("=" * 80)
    print("ANOMALY DETECTION")
    print("=" * 80)

    anomaly_df = df[df['anomalies'] != 'none']

    if len(anomaly_df) == 0:
        print("No anomalies detected!")
    else:
        print(f"Found {len(anomaly_df)} episodes with anomalies:\n")

        for _, row in anomaly_df.iterrows():
            print(f"  Episode {row['episode_id']:4d}: {row['anomalies']}")

    print()

    # Static episodes
    static = df[df['is_static']]
    if len(static) > 0:
        print(f"Static episodes (< 10° total movement): {static['episode_id'].tolist()}")

    # No gripping action
    no_grip = df[~df['has_gripping']]
    if len(no_grip) > 0:
        print(f"Episodes without gripping action: {no_grip['episode_id'].tolist()}")

    print("=" * 80)
    print()

def print_gripper_analysis(df: pd.DataFrame):
    """Print gripper pattern analysis."""
    print("=" * 80)
    print("GRIPPER ANALYSIS")
    print("=" * 80)

    has_grip = df[df['has_gripping']]
    no_grip = df[~df['has_gripping']]

    print(f"Episodes with gripping action:     {len(has_grip)} ({len(has_grip)/len(df)*100:.1f}%)")
    print(f"Episodes without gripping action:  {len(no_grip)} ({len(no_grip)/len(df)*100:.1f}%)")
    print()

    print(f"Mean gripper range:                {df['gripper_range'].mean():.2f}°")
    print(f"Max gripper opening (across all):  {df['gripper_max'].max():.2f}°")
    print(f"Min gripper closing (across all):  {df['gripper_min'].min():.2f}°")

    if len(no_grip) > 0:
        print(f"\nWARNING: Episodes without proper gripping: {no_grip['episode_id'].tolist()}")

    print("=" * 80)
    print()

def print_rgb_validation(df: pd.DataFrame):
    """Print RGB image validation results."""
    print("=" * 80)
    print("RGB IMAGE VALIDATION")
    print("=" * 80)

    valid = df[df['rgb_valid']]
    invalid = df[~df['rgb_valid']]

    print(f"Valid RGB episodes:   {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
    print(f"Invalid RGB episodes: {len(invalid)} ({len(invalid)/len(df)*100:.1f}%)")

    if len(invalid) > 0:
        print("\nInvalid RGB episodes:")
        for _, row in invalid.iterrows():
            print(f"  Episode {row['episode_id']:4d}: {row['rgb_issues']}")

    print("=" * 80)
    print()

def print_training_readiness(df: pd.DataFrame):
    """Print comprehensive training readiness assessment."""
    print("\n")
    print("=" * 80)
    print("TRAINING READINESS ASSESSMENT")
    print("=" * 80)
    print()

    total_episodes = len(df)
    deep_episodes = len(df[df['quality_grade'] == 'DEEP'])
    approach_episodes = len(df[df['quality_grade'] == 'APPROACH'])
    shallow_episodes = len(df[df['quality_grade'] == 'SHALLOW'])

    anomalies = len(df[df['anomalies'] != 'none'])
    no_gripping = len(df[~df['has_gripping']])
    invalid_rgb = len(df[~df['rgb_valid']])

    # Scoring criteria
    issues = []
    warnings = []
    strengths = []

    # 1. Total episode count
    if total_episodes < 50:
        issues.append(f"Insufficient episodes ({total_episodes} < 50)")
    elif total_episodes < 100:
        warnings.append(f"Episode count adequate but not optimal ({total_episodes} < 100)")
    else:
        strengths.append(f"Sufficient episodes ({total_episodes} >= 100)")

    # 2. DEEP episode ratio
    deep_ratio = deep_episodes / total_episodes
    if deep_ratio < 0.3:
        issues.append(f"DEEP episodes too few ({deep_episodes}/{total_episodes} = {deep_ratio*100:.1f}% < 30%)")
    elif deep_ratio < 0.5:
        warnings.append(f"DEEP episodes acceptable but low ({deep_ratio*100:.1f}%)")
    else:
        strengths.append(f"Good DEEP episode ratio ({deep_ratio*100:.1f}%)")

    # 3. Diversity
    if deep_episodes > 0 and approach_episodes > 0 and shallow_episodes > 0:
        strengths.append("Good diversity across all depth categories")
    else:
        warnings.append("Missing episodes in some depth categories")

    # 4. Data quality
    if anomalies > total_episodes * 0.2:
        issues.append(f"Too many anomalies ({anomalies}/{total_episodes} = {anomalies/total_episodes*100:.1f}%)")
    elif anomalies > 0:
        warnings.append(f"{anomalies} episodes with anomalies")
    else:
        strengths.append("No anomalies detected")

    # 5. Gripping action
    if no_gripping > total_episodes * 0.3:
        issues.append(f"Many episodes lack gripping action ({no_gripping}/{total_episodes})")
    elif no_gripping > 0:
        warnings.append(f"{no_gripping} episodes without gripping action")
    else:
        strengths.append("All episodes have gripping action")

    # 6. RGB validity
    if invalid_rgb > 0:
        issues.append(f"{invalid_rgb} episodes with invalid RGB data")
    else:
        strengths.append("All RGB images valid")

    # Print assessment
    print("STRENGTHS:")
    if strengths:
        for s in strengths:
            print(f"  + {s}")
    else:
        print("  (none)")
    print()

    print("WARNINGS:")
    if warnings:
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("  (none)")
    print()

    print("CRITICAL ISSUES:")
    if issues:
        for i in issues:
            print(f"  X {i}")
    else:
        print("  (none)")
    print()

    # Final recommendation
    print("=" * 80)
    print("FINAL RECOMMENDATION:")
    print("=" * 80)

    if len(issues) > 0:
        print("STATUS: NOT READY FOR TRAINING")
        print()
        print("REASONING:")
        print("  This dataset has critical issues that will likely lead to training failure.")
        print("  Previous dataset (51 episodes, 3.9% DEEP) failed due to insufficient DEEP episodes.")
        print()
        print("REQUIRED ACTIONS:")

        if deep_ratio < 0.5:
            additional_deep_needed = int(total_episodes * 0.5 - deep_episodes)
            print(f"  1. Collect {additional_deep_needed}+ additional DEEP episodes (elbow < -30°)")

        if total_episodes < 100:
            additional_total_needed = 100 - total_episodes
            print(f"  2. Collect {additional_total_needed}+ more episodes total (target: 100+)")

        if no_gripping > 5:
            print(f"  3. Re-collect {no_gripping} episodes that lack gripping action")

        if invalid_rgb > 0:
            print(f"  4. Fix or re-collect {invalid_rgb} episodes with invalid RGB data")

    elif len(warnings) > 0:
        print("STATUS: PROCEED WITH CAUTION")
        print()
        print("REASONING:")
        print("  Dataset meets minimum requirements but has some weaknesses.")
        print("  Training may succeed but performance could be suboptimal.")
        print()
        print("RECOMMENDED ACTIONS:")

        if deep_ratio < 0.5:
            additional_deep_suggested = int(total_episodes * 0.6 - deep_episodes)
            print(f"  1. (Optional) Collect {additional_deep_suggested}+ more DEEP episodes for better performance")

        if total_episodes < 100:
            print(f"  2. (Recommended) Collect more episodes to reach 100+ for robustness")

        print()
        print("  You may proceed with training, but monitor validation metrics closely.")

    else:
        print("STATUS: READY FOR TRAINING")
        print()
        print("REASONING:")
        print("  Dataset meets all quality criteria:")
        print(f"    - Sufficient episodes: {total_episodes}")
        print(f"    - Good DEEP ratio: {deep_ratio*100:.1f}%")
        print(f"    - Low anomalies: {anomalies}")
        print(f"    - Valid RGB data: {total_episodes - invalid_rgb}/{total_episodes}")
        print()
        print("NEXT STEPS:")
        print("  1. Convert to LeRobot v3 format: python convert_to_lerobot_v3.py")
        print("  2. Train with official CLI: python run_official_train.py")
        print("  3. Monitor L2 error, z-score, and diversity during training")

    print("=" * 80)

def main():
    dataset_dir = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")

    print("\n" + "="*80)
    print("RoArm M3 Dataset Comprehensive Quality Analysis")
    print("="*80 + "\n")

    # Analyze all episodes
    df, global_angles = analyze_all_episodes(dataset_dir)

    # Print all analysis sections
    print_episode_summary_table(df)
    print_elbow_distribution(df)
    print_joint_distribution(global_angles)
    print_frame_count_analysis(df)
    print_time_analysis(df)
    print_gripper_analysis(df)
    print_anomaly_detection(df)
    print_rgb_validation(df)
    print_training_readiness(df)

    # Save detailed CSV
    output_csv = dataset_dir / "analysis_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
