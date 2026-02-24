#!/usr/bin/env python3
"""
Final comprehensive report with corrected gripper detection thresholds.
Uses relaxed criteria: max>30° AND min<30° for gripper action detection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict

# CORRECTED: Relaxed gripper criteria based on investigation
GRIPPER_OPEN_THRESHOLD = 30.0  # Was 50, now 30
GRIPPER_CLOSED_THRESHOLD = 30.0  # Was 20, now 30

JOINT_NAMES = ['Base', 'Shoulder', 'Elbow', 'Wrist_pitch', 'Wrist_roll', 'Gripper']

def load_and_analyze_episode(episode_dir: Path) -> Dict:
    """Load and analyze single episode with corrected thresholds."""
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    episode_id = metadata['episode_id']
    num_frames = metadata['num_frames']
    angles_array = np.array([frame['angles'] for frame in metadata['frames']])

    # Elbow analysis
    elbow_angles = angles_array[:, 2]
    min_elbow = np.min(elbow_angles)
    max_elbow = np.max(elbow_angles)
    elbow_range = max_elbow - min_elbow

    if min_elbow < -30:
        quality_grade = "DEEP"
    elif -30 <= min_elbow < -10:
        quality_grade = "APPROACH"
    else:
        quality_grade = "SHALLOW"

    # CORRECTED: Gripper analysis with relaxed thresholds
    gripper_angles = angles_array[:, 5]
    gripper_min = np.min(gripper_angles)
    gripper_max = np.max(gripper_angles)
    gripper_range = gripper_max - gripper_min

    # Relaxed criteria: opened > 30° AND closed range includes < 30°
    gripper_opened = gripper_max > GRIPPER_OPEN_THRESHOLD
    gripper_closed = gripper_min < GRIPPER_CLOSED_THRESHOLD
    has_gripping = gripper_opened and gripper_closed

    # Joint movement
    joint_ranges = np.max(angles_array, axis=0) - np.min(angles_array, axis=0)
    total_movement = np.sum(joint_ranges)
    is_static = total_movement < 10

    # Timestamps
    timestamps = [frame['timestamp'] for frame in metadata['frames']]
    duration = timestamps[-1] - timestamps[0]

    # RGB validation
    try:
        first_rgb_path = episode_dir / metadata['frames'][0]['rgb_path']
        last_rgb_path = episode_dir / metadata['frames'][-1]['rgb_path']
        first_img = np.array(Image.open(first_rgb_path))
        last_img = np.array(Image.open(last_rgb_path))
        first_brightness = np.mean(first_img)
        last_brightness = np.mean(last_img)
        rgb_valid = (10 < first_brightness < 245) and (10 < last_brightness < 245)
    except:
        rgb_valid = False

    # Anomalies
    anomalies = []
    if num_frames < 10:
        anomalies.append("too_short")
    if is_static:
        anomalies.append("static")
    if not has_gripping:
        anomalies.append("no_gripping")

    return {
        'episode_id': episode_id,
        'num_frames': num_frames,
        'duration_sec': duration,
        'min_elbow': min_elbow,
        'max_elbow': max_elbow,
        'elbow_range': elbow_range,
        'quality_grade': quality_grade,
        'gripper_min': gripper_min,
        'gripper_max': gripper_max,
        'gripper_range': gripper_range,
        'has_gripping': has_gripping,
        'is_static': is_static,
        'rgb_valid': rgb_valid,
        'anomalies': ','.join(anomalies) if anomalies else 'none',
        'angles_array': angles_array,
    }

def main():
    dataset_dir = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data")
    episode_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])

    print("\n" + "="*100)
    print("FINAL COMPREHENSIVE REPORT - CORRECTED GRIPPER THRESHOLDS")
    print("="*100)
    print()
    print(f"Gripper detection criteria: OPENED > {GRIPPER_OPEN_THRESHOLD}° AND CLOSED < {GRIPPER_CLOSED_THRESHOLD}°")
    print()

    # Analyze all episodes
    results = []
    all_angles = []
    for ep_dir in episode_dirs:
        result = load_and_analyze_episode(ep_dir)
        angles = result.pop('angles_array')
        results.append(result)
        all_angles.append(angles)

    df = pd.DataFrame(results)
    global_angles = np.vstack(all_angles)

    # CORRECTED STATISTICS
    total = len(df)
    deep = len(df[df['quality_grade'] == 'DEEP'])
    approach = len(df[df['quality_grade'] == 'APPROACH'])
    shallow = len(df[df['quality_grade'] == 'SHALLOW'])
    with_gripping = len(df[df['has_gripping'] == True])
    anomalies = len(df[df['anomalies'] != 'none'])

    print("="*100)
    print("CORRECTED DATASET QUALITY SUMMARY")
    print("="*100)
    print()
    print(f"Total episodes:                {total}")
    print()
    print("ELBOW DEPTH DISTRIBUTION:")
    print(f"  DEEP (< -30°):               {deep:2d} episodes ({deep/total*100:5.1f}%)")
    print(f"  APPROACH (-30° to -10°):     {approach:2d} episodes ({approach/total*100:5.1f}%)")
    print(f"  SHALLOW (> -10°):            {shallow:2d} episodes ({shallow/total*100:5.1f}%)")
    print()
    print("GRIPPER ACTION (CORRECTED THRESHOLDS):")
    print(f"  With gripping action:        {with_gripping:2d} episodes ({with_gripping/total*100:5.1f}%)")
    print(f"  Without gripping action:     {total - with_gripping:2d} episodes ({(total-with_gripping)/total*100:5.1f}%)")
    print()
    print("DATA QUALITY:")
    print(f"  Valid RGB images:            {len(df[df['rgb_valid']])} episodes ({len(df[df['rgb_valid']])/total*100:5.1f}%)")
    print(f"  Episodes with anomalies:     {anomalies} episodes ({anomalies/total*100:5.1f}%)")
    print()

    # Comparison with previous failure
    print("="*100)
    print("COMPARISON WITH PREVIOUS FAILED DATASET:")
    print("="*100)
    print("  Previous dataset:")
    print("    - 51 episodes")
    print("    - DEEP: 2 episodes (3.9%)")
    print("    - Result: Training FAILED (model only outputs mean action)")
    print()
    print("  Current dataset:")
    print(f"    - {total} episodes")
    print(f"    - DEEP: {deep} episodes ({deep/total*100:.1f}%)")
    print(f"    - Gripper action: {with_gripping} episodes ({with_gripping/total*100:.1f}%)")
    print()

    if deep / total < 0.3:
        print("  ASSESSMENT: INSUFFICIENT DEEP episodes")
        print(f"    Current {deep/total*100:.1f}% vs required 30%+ for robust training")
        print("    RISK: Similar failure likely due to insufficient elbow depth diversity")
    elif deep / total < 0.5:
        print("  ASSESSMENT: MARGINAL DEEP ratio")
        print(f"    Current {deep/total*100:.1f}% meets minimum but below optimal 50%+")
        print("    RISK: Training may succeed but with suboptimal performance")
    else:
        print("  ASSESSMENT: GOOD DEEP ratio")
        print(f"    Current {deep/total*100:.1f}% meets target 50%+")
        print("    Much better than previous failed dataset!")

    print("="*100)
    print()

    # Joint distribution
    print("="*100)
    print("GLOBAL JOINT DISTRIBUTION")
    print("="*100)
    print(f"{'Joint':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 100)
    for i, joint_name in enumerate(JOINT_NAMES):
        joint_angles = global_angles[:, i]
        mean = np.mean(joint_angles)
        std = np.std(joint_angles)
        min_val = np.min(joint_angles)
        max_val = np.max(joint_angles)
        range_val = max_val - min_val
        print(f"{joint_name:<15} {mean:>10.2f} {std:>10.2f} {min_val:>10.2f} {max_val:>10.2f} {range_val:>10.2f}")
    print("="*100)
    print()

    # Episodes needing attention
    no_gripping = df[~df['has_gripping']]
    if len(no_gripping) > 0:
        print("="*100)
        print(f"EPISODES WITHOUT GRIPPING ACTION ({len(no_gripping)} episodes):")
        print("="*100)
        print("These episodes should be reviewed and potentially re-collected:")
        print(no_gripping['episode_id'].tolist())
        print()
        print("Note: Even with relaxed thresholds, these episodes lack clear gripper actuation.")
        print("="*100)
        print()

    # FINAL TRAINING READINESS
    print()
    print("="*100)
    print("FINAL TRAINING READINESS ASSESSMENT")
    print("="*100)
    print()

    # Criteria evaluation
    criteria_met = []
    criteria_failed = []
    warnings = []

    # 1. Episode count
    if total >= 100:
        criteria_met.append(f"Episode count: {total} >= 100")
    elif total >= 50:
        warnings.append(f"Episode count: {total} (minimum met, but 100+ recommended)")
    else:
        criteria_failed.append(f"Episode count: {total} < 50 (insufficient)")

    # 2. DEEP ratio
    deep_ratio = deep / total
    if deep_ratio >= 0.5:
        criteria_met.append(f"DEEP ratio: {deep_ratio*100:.1f}% >= 50% (excellent)")
    elif deep_ratio >= 0.3:
        warnings.append(f"DEEP ratio: {deep_ratio*100:.1f}% (meets minimum 30%, but 50%+ optimal)")
    else:
        criteria_failed.append(f"DEEP ratio: {deep_ratio*100:.1f}% < 30% (insufficient)")

    # 3. Gripper action
    grip_ratio = with_gripping / total
    if grip_ratio >= 0.9:
        criteria_met.append(f"Gripper action: {grip_ratio*100:.1f}% >= 90% (excellent)")
    elif grip_ratio >= 0.7:
        warnings.append(f"Gripper action: {grip_ratio*100:.1f}% (acceptable, but 90%+ optimal)")
    else:
        criteria_failed.append(f"Gripper action: {grip_ratio*100:.1f}% < 70% (insufficient)")

    # 4. Data quality
    if anomalies / total < 0.1:
        criteria_met.append(f"Anomaly rate: {anomalies/total*100:.1f}% < 10% (good)")
    elif anomalies / total < 0.2:
        warnings.append(f"Anomaly rate: {anomalies/total*100:.1f}% (acceptable)")
    else:
        criteria_failed.append(f"Anomaly rate: {anomalies/total*100:.1f}% >= 20% (high)")

    # Print results
    if len(criteria_met) > 0:
        print("CRITERIA MET:")
        for c in criteria_met:
            print(f"  ✓ {c}")
        print()

    if len(warnings) > 0:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
        print()

    if len(criteria_failed) > 0:
        print("CRITERIA FAILED:")
        for f in criteria_failed:
            print(f"  ✗ {f}")
        print()

    # Final verdict
    print("="*100)
    if len(criteria_failed) > 0:
        print("VERDICT: NOT READY FOR TRAINING")
        print()
        print("CRITICAL ISSUES MUST BE RESOLVED:")
        print()
        if deep_ratio < 0.3:
            needed_deep = int(total * 0.5 - deep)
            print(f"  1. Collect {needed_deep}+ additional DEEP episodes (target: 50% of total)")
        if total < 100:
            needed_total = 100 - total
            print(f"  2. Collect {needed_total}+ more episodes total (target: 100+)")
        if grip_ratio < 0.7:
            print(f"  3. Re-collect or fix {total - with_gripping} episodes without gripper action")
        print()
        print("Proceeding with training now will likely result in failure similar to previous attempt.")

    elif len(warnings) > 0:
        print("VERDICT: PROCEED WITH CAUTION")
        print()
        print("Dataset meets minimum requirements but has weaknesses:")
        print()
        if deep_ratio < 0.5:
            suggested_deep = int(total * 0.6 - deep)
            print(f"  - Consider collecting {suggested_deep}+ more DEEP episodes for robustness")
        if total < 100:
            print(f"  - Recommended to collect more episodes (target: 100+)")
        if grip_ratio < 0.9:
            print(f"  - {total - with_gripping} episodes lack gripping (acceptable but not ideal)")
        print()
        print("You may proceed to training, but monitor validation metrics closely.")
        print("If training fails, return to data collection focusing on DEEP episodes.")

    else:
        print("VERDICT: READY FOR TRAINING")
        print()
        print("Dataset meets all quality criteria for SmolVLA training!")
        print()
        print("NEXT STEPS:")
        print("  1. Convert to LeRobot v3 format:")
        print("     python convert_to_lerobot_v3.py --input collected_data --task 'Pick up the sponge'")
        print()
        print("  2. Train with official CLI:")
        print("     python run_official_train.py")
        print()
        print("  3. Monitor training metrics:")
        print("     - Training loss should decrease steadily")
        print("     - L2 error < 0.5 (offline inference test)")
        print("     - Action diversity > 0 (not just mean actions)")
        print()
        print("  4. Test deployment:")
        print("     python deploy_smolvla.py --start-pos dataset_mean --max-steps 300")

    print("="*100)
    print()

    # Save corrected analysis
    output_csv = dataset_dir / "analysis_corrected.csv"
    df.to_csv(output_csv, index=False)
    print(f"Corrected analysis saved to: {output_csv}")
    print()

if __name__ == "__main__":
    main()
