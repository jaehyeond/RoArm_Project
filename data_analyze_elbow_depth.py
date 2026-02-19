#!/usr/bin/env python3
"""
Analyze whether elbow angle < -30° actually correlates with deep grasps.
Investigate the relationship between shoulder, elbow, and wrist_pitch angles.
"""

import json
import csv
from pathlib import Path

PROJECT_ROOT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
DATA_DIR = PROJECT_ROOT / "collected_data"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_episode(ep_num):
    """Analyze a single episode's joint trajectory."""
    metadata_path = DATA_DIR / f"episode_{ep_num:04d}" / "metadata.json"

    with open(metadata_path) as f:
        data = json.load(f)

    frames = data['frames']
    angles_list = [f['angles'] for f in frames]

    # Joint order: [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
    elbow_values = [angles[2] for angles in angles_list]
    shoulder_values = [angles[1] for angles in angles_list]
    wrist_pitch_values = [angles[3] for angles in angles_list]
    gripper_values = [angles[5] for angles in angles_list]

    # Find min elbow
    min_elbow_idx = elbow_values.index(min(elbow_values))
    angles_at_min_elbow = angles_list[min_elbow_idx]

    # Start and end
    start_angles = angles_list[0]
    end_angles = angles_list[-1]

    return {
        'episode': ep_num,
        'num_frames': len(frames),
        'min_elbow': min(elbow_values),
        'min_elbow_frame': min_elbow_idx,
        'shoulder_at_min_elbow': angles_at_min_elbow[1],
        'wrist_pitch_at_min_elbow': angles_at_min_elbow[3],
        'gripper_at_min_elbow': angles_at_min_elbow[5],
        'start_elbow': start_angles[2],
        'start_shoulder': start_angles[1],
        'start_gripper': start_angles[5],
        'end_elbow': end_angles[2],
        'end_shoulder': end_angles[1],
        'end_gripper': end_angles[5],
        'elbow_change': end_angles[2] - start_angles[2],
        'shoulder_change': end_angles[1] - start_angles[1],
    }

def main():
    # Read the analysis CSV to get episode classifications
    csv_path = DATA_DIR / "analysis_corrected.csv"

    episodes_by_grade = {'DEEP': [], 'APPROACH': [], 'SHALLOW': []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_id = int(row['episode_id'])
            grade = row['quality_grade']
            if grade in episodes_by_grade:
                episodes_by_grade[grade].append(ep_id)

    print("=== DEEP EPISODES ===")
    deep_results = []
    for ep in sorted(episodes_by_grade['DEEP']):
        result = analyze_episode(ep)
        deep_results.append(result)
        print(f"Episode {result['episode']:2d}: min_elbow={result['min_elbow']:6.1f}° at frame {result['min_elbow_frame']:3d}/{result['num_frames']:3d}")
        print(f"  At min_elbow: shoulder={result['shoulder_at_min_elbow']:6.1f}°, wrist_pitch={result['wrist_pitch_at_min_elbow']:6.1f}°, gripper={result['gripper_at_min_elbow']:5.1f}")
        print(f"  Trajectory: elbow {result['start_elbow']:6.1f}° → {result['end_elbow']:6.1f}° (Δ={result['elbow_change']:+6.1f}°)")
        print(f"              shoulder {result['start_shoulder']:6.1f}° → {result['end_shoulder']:6.1f}° (Δ={result['shoulder_change']:+6.1f}°)")
        print()

    print("\n=== SHALLOW EPISODES (sample) ===")
    shallow_results = []
    for ep in sorted(episodes_by_grade['SHALLOW'])[:10]:  # First 10
        result = analyze_episode(ep)
        shallow_results.append(result)
        print(f"Episode {result['episode']:2d}: min_elbow={result['min_elbow']:6.1f}° at frame {result['min_elbow_frame']:3d}/{result['num_frames']:3d}")
        print(f"  At min_elbow: shoulder={result['shoulder_at_min_elbow']:6.1f}°, wrist_pitch={result['wrist_pitch_at_min_elbow']:6.1f}°, gripper={result['gripper_at_min_elbow']:5.1f}")
        print(f"  Trajectory: elbow {result['start_elbow']:6.1f}° → {result['end_elbow']:6.1f}° (Δ={result['elbow_change']:+6.1f}°)")
        print(f"              shoulder {result['start_shoulder']:6.1f}° → {result['end_shoulder']:6.1f}° (Δ={result['shoulder_change']:+6.1f}°)")
        print()

    # Statistical analysis
    print("\n=== STATISTICAL COMPARISON ===")

    deep_shoulders = [r['shoulder_at_min_elbow'] for r in deep_results]
    shallow_shoulders = [r['shoulder_at_min_elbow'] for r in shallow_results]

    print(f"DEEP episodes ({len(deep_results)} total):")
    print(f"  Shoulder at min_elbow: mean={sum(deep_shoulders)/len(deep_shoulders):.1f}°, "
          f"min={min(deep_shoulders):.1f}°, max={max(deep_shoulders):.1f}°")

    print(f"\nSHALLOW episodes (sample {len(shallow_results)}):")
    print(f"  Shoulder at min_elbow: mean={sum(shallow_shoulders)/len(shallow_shoulders):.1f}°, "
          f"min={min(shallow_shoulders):.1f}°, max={max(shallow_shoulders):.1f}°")

    # Key insight check
    print("\n=== KEY INSIGHT ===")
    print("For a true DEEP grasp (reaching DOWN to table):")
    print("  - Elbow should be NEGATIVE (extended)")
    print("  - Shoulder should be LOW-to-MODERATE (forward lean, NOT upright)")
    print("  - Combined effect: end-effector Z is LOW")
    print("\nFor a horizontal extension (NOT useful for grasping):")
    print("  - Elbow NEGATIVE (extended)")
    print("  - Shoulder HIGH (upright or backward)")
    print("  - Combined effect: end-effector Z is MEDIUM-to-HIGH")

    # Write detailed report
    output_path = OUTPUT_DIR / "data_elbow_depth_analysis.md"
    with open(output_path, 'w') as f:
        f.write("# Elbow Depth Analysis: Is min_elbow < -30° a Valid Grasp Depth Metric?\n\n")
        f.write("## Investigation\n\n")
        f.write("**Question**: Does elbow angle < -30° actually indicate deep grasps, or could it be horizontal arm extension?\n\n")
        f.write("**Hypothesis**: For a 6-DOF arm, end-effector height depends on:\n")
        f.write("- Shoulder angle (lifts/lowers the upper arm)\n")
        f.write("- Elbow angle (extends/folds the forearm)\n")
        f.write("- Wrist pitch (tilts the gripper)\n\n")
        f.write("A negative elbow could mean:\n")
        f.write("- (a) Arm reaching DOWN to table (shoulder forward, elbow extending down) — GOOD for grasping\n")
        f.write("- (b) Arm extending HORIZONTALLY (shoulder up, elbow straightened) — NOT useful for grasping\n\n")

        f.write("## DEEP Episodes Analysis\n\n")
        f.write("| Episode | min_elbow | Shoulder @ min | Wrist_pitch @ min | Gripper @ min | Start→End Elbow | Start→End Shoulder |\n")
        f.write("|---------|-----------|----------------|-------------------|---------------|-----------------|--------------------|\n")
        for r in deep_results:
            f.write(f"| {r['episode']:2d} | {r['min_elbow']:6.1f}° | {r['shoulder_at_min_elbow']:6.1f}° | "
                   f"{r['wrist_pitch_at_min_elbow']:6.1f}° | {r['gripper_at_min_elbow']:5.1f} | "
                   f"{r['start_elbow']:5.1f}→{r['end_elbow']:5.1f} ({r['elbow_change']:+5.1f}) | "
                   f"{r['start_shoulder']:5.1f}→{r['end_shoulder']:5.1f} ({r['shoulder_change']:+5.1f}) |\n")

        f.write("\n## SHALLOW Episodes Analysis (sample)\n\n")
        f.write("| Episode | min_elbow | Shoulder @ min | Wrist_pitch @ min | Gripper @ min | Start→End Elbow | Start→End Shoulder |\n")
        f.write("|---------|-----------|----------------|-------------------|---------------|-----------------|--------------------|\n")
        for r in shallow_results:
            f.write(f"| {r['episode']:2d} | {r['min_elbow']:6.1f}° | {r['shoulder_at_min_elbow']:6.1f}° | "
                   f"{r['wrist_pitch_at_min_elbow']:6.1f}° | {r['gripper_at_min_elbow']:5.1f} | "
                   f"{r['start_elbow']:5.1f}→{r['end_elbow']:5.1f} ({r['elbow_change']:+5.1f}) | "
                   f"{r['start_shoulder']:5.1f}→{r['end_shoulder']:5.1f} ({r['shoulder_change']:+5.1f}) |\n")

        f.write("\n## Statistical Summary\n\n")
        f.write(f"**DEEP episodes** (n={len(deep_results)}):\n")
        f.write(f"- Shoulder at min_elbow: mean={sum(deep_shoulders)/len(deep_shoulders):.1f}°, "
               f"range=[{min(deep_shoulders):.1f}°, {max(deep_shoulders):.1f}°]\n\n")

        f.write(f"**SHALLOW episodes** (n={len(shallow_results)} sampled):\n")
        f.write(f"- Shoulder at min_elbow: mean={sum(shallow_shoulders)/len(shallow_shoulders):.1f}°, "
               f"range=[{min(shallow_shoulders):.1f}°, {max(shallow_shoulders):.1f}°]\n\n")

        f.write("## Findings\n\n")
        f.write("### Pattern Detection\n\n")

        # Detect if DEEP episodes show descend-grasp-lift pattern
        descend_lift_count = sum(1 for r in deep_results if r['shoulder_change'] > 10)
        f.write(f"- **Descend-grasp-lift pattern**: {descend_lift_count}/{len(deep_results)} DEEP episodes show shoulder rising >10° (lift after grasp)\n\n")

        # Check shoulder range at min_elbow
        deep_shoulder_mean = sum(deep_shoulders) / len(deep_shoulders)
        shallow_shoulder_mean = sum(shallow_shoulders) / len(shallow_shoulders)
        f.write(f"- **Shoulder positioning**: DEEP episodes have mean shoulder={deep_shoulder_mean:.1f}° vs SHALLOW mean={shallow_shoulder_mean:.1f}° at min_elbow\n\n")

        f.write("### Interpretation\n\n")

        if abs(deep_shoulder_mean - shallow_shoulder_mean) < 10:
            f.write("⚠️ **WARNING**: DEEP and SHALLOW episodes have similar shoulder angles at min_elbow!\n\n")
            f.write("This suggests that **elbow < -30° is NOT a reliable proxy for grasp depth**.\n\n")
            f.write("**Possible scenarios**:\n")
            f.write("1. Both groups extend the arm horizontally (high shoulder + negative elbow)\n")
            f.write("2. The camera view cannot distinguish vertical depth from horizontal extension\n")
            f.write("3. Manual teleoperation varied widely in approach strategy\n\n")
        else:
            f.write("✓ DEEP episodes show distinct shoulder positioning compared to SHALLOW.\n\n")
            f.write(f"DEEP episodes tend to have {'lower' if deep_shoulder_mean < shallow_shoulder_mean else 'higher'} "
                   f"shoulder angles, suggesting {'forward lean toward table' if deep_shoulder_mean < shallow_shoulder_mean else 'upright posture'}.\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. **Visual inspection**: Review video frames at min_elbow for DEEP episodes 2, 23, 24, 25, 31, 34, 41, 48, 49\n")
        f.write("   - Check if gripper is actually near the table surface\n")
        f.write("   - Verify if arm is reaching DOWN vs extending OUTWARD\n\n")
        f.write("2. **Better metric**: Consider using a combination of:\n")
        f.write("   - Shoulder angle (forward lean indicator)\n")
        f.write("   - Elbow angle (extension indicator)\n")
        f.write("   - Wrist pitch (downward tilt indicator)\n")
        f.write("   - Gripper trajectory (open→close→lift pattern)\n\n")
        f.write("3. **Data collection strategy**:\n")
        f.write("   - If current DEEP episodes are NOT actually deep grasps, the 77-episode collection plan needs revision\n")
        f.write("   - Focus on ensuring the gripper actually contacts the table before grasping\n")
        f.write("   - Record Z-height from Kinect depth to validate grasp depth\n\n")

    print(f"\nDetailed report written to: {output_path}")

if __name__ == "__main__":
    main()
