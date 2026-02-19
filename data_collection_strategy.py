#!/usr/bin/env python3
"""
Data Collection Strategy (Phase 1 Diagnosis - Step 4)

Analyzes existing 50 episodes and creates a concrete collection guide for the next round.
Addresses the root cause: 68% SHALLOW data → model learned wrong behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_existing_analysis():
    """Load the corrected analysis CSV."""
    csv_path = Path('/home/cgxr/Documents/Robotics/RoArm_Project/collected_data/analysis_corrected.csv')
    df = pd.read_csv(csv_path)
    return df


def analyze_current_distribution(df):
    """Analyze the distribution of existing 50 episodes."""
    total = len(df)

    # Count by quality grade
    grade_counts = df['quality_grade'].value_counts()
    grade_pct = (grade_counts / total * 100).round(1)

    # Identify problematic episodes
    static_episodes = df[df['is_static'] == True]
    no_gripping = df[df['has_gripping'] == False]
    both_issues = df[(df['is_static'] == True) | (df['has_gripping'] == False)]

    # Gripper analysis
    gripper_stats = {
        'mean_range': df['gripper_range'].mean(),
        'mean_min': df['gripper_min'].mean(),
        'mean_max': df['gripper_max'].mean(),
    }

    # Elbow depth distribution
    elbow_bins = [
        ('VERY_SHALLOW', df[df['min_elbow'] > 0]),
        ('SHALLOW', df[(df['min_elbow'] <= 0) & (df['min_elbow'] > -10)]),
        ('APPROACH', df[(df['min_elbow'] <= -10) & (df['min_elbow'] > -30)]),
        ('DEEP', df[df['min_elbow'] <= -30]),
    ]

    return {
        'total': total,
        'grade_counts': grade_counts,
        'grade_pct': grade_pct,
        'static_count': len(static_episodes),
        'no_gripping_count': len(no_gripping),
        'problematic_count': len(both_issues),
        'problematic_ids': both_issues['episode_id'].tolist(),
        'gripper_stats': gripper_stats,
        'elbow_bins': elbow_bins,
    }


def design_collection_targets():
    """
    Design target distribution for next collection round.

    Current problem: 68% SHALLOW → model thinks "APPROACH = LIFT"
    Solution: Rebalance to emphasize DEEP grasps.
    """
    # Target distribution (aggressive rebalancing)
    targets = {
        'DEEP': {
            'count': 60,  # 50% of 120 episodes
            'description': 'Elbow < -30°, full reach down',
            'requirements': [
                'Elbow min < -30°',
                'Gripper opens (>20°) then closes (<10°)',
                'Smooth descent trajectory',
                'No static frames (elbow range > 10°)',
            ]
        },
        'APPROACH': {
            'count': 36,  # 30% of 120 episodes
            'description': 'Elbow -30° to -10°, medium depth',
            'requirements': [
                '-30° < Elbow min < -10°',
                'Gripper activity (range > 15°)',
                'Clear reaching motion',
            ]
        },
        'SHALLOW': {
            'count': 24,  # 20% of 120 episodes
            'description': 'Elbow > -10°, surface operations',
            'requirements': [
                'Elbow min > -10°',
                'Gripper activity (range > 10°)',
                'Functional motion (not static)',
            ]
        },
        'TOTAL': 120,
    }

    return targets


def identify_episodes_to_remove(df):
    """Identify episodes that should be removed from dataset."""
    remove_criteria = [
        ('Static (no motion)', df['is_static'] == True),
        ('No gripping', df['has_gripping'] == False),
        ('Very short (<50 frames)', df['num_frames'] < 50),
        ('Gripper stuck (range < 5°)', df['gripper_range'] < 5),
    ]

    to_remove = set()
    reasons = {}

    for reason, mask in remove_criteria:
        episodes = df[mask]['episode_id'].tolist()
        for ep_id in episodes:
            if ep_id not in to_remove:
                reasons[ep_id] = reason
            to_remove.add(ep_id)

    return sorted(to_remove), reasons


def create_realtime_guidance_spec():
    """
    Design real-time guidance display for collect_data_manual.py.

    This will help the data collector know what type of episode to record next.
    """
    guidance = {
        'display_during_collection': {
            'current_stats': [
                'Total episodes collected',
                'DEEP count / 60 target',
                'APPROACH count / 36 target',
                'SHALLOW count / 24 target',
                'Progress bar for each category',
            ],
            'next_recommended': [
                'If DEEP < 60: "COLLECT DEEP grasp (elbow < -30°)"',
                'Else if APPROACH < 36: "COLLECT APPROACH grasp (-30° < elbow < -10°)"',
                'Else: "COLLECT SHALLOW grasp (elbow > -10°)"',
            ],
            'quality_checklist': [
                '✓ Elbow moved > 10° (not static)',
                '✓ Gripper opened AND closed',
                '✓ Smooth, controlled motion',
                '✓ Target elbow depth achieved',
            ]
        },
        'post_episode_validation': {
            'compute_immediately': [
                'min_elbow → classify as DEEP/APPROACH/SHALLOW',
                'gripper_range → check for activity',
                'elbow_range → check for motion',
            ],
            'accept_reject_prompt': [
                'ACCEPT: Episode meets target criteria',
                'RETRY: Episode quality insufficient (suggest why)',
                'SKIP: Intentionally skip this type for now',
            ]
        }
    }

    return guidance


def plot_current_vs_target_distribution(current_stats, targets):
    """Visualize current vs target distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Current distribution
    ax1 = axes[0]
    grades = ['DEEP', 'APPROACH', 'SHALLOW']
    current_counts = [current_stats['grade_counts'].get(g, 0) for g in grades]
    current_pct = [c / current_stats['total'] * 100 for c in current_counts]

    bars1 = ax1.bar(grades, current_pct, color=['red', 'orange', 'green'], alpha=0.7)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Current Distribution (50 episodes)')
    ax1.set_ylim(0, 80)

    # Add value labels
    for bar, pct, count in zip(bars1, current_pct, current_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n({count} ep)',
                ha='center', va='bottom', fontsize=10)

    # Target distribution
    ax2 = axes[1]
    target_counts = [targets[g]['count'] for g in grades]
    target_pct = [c / targets['TOTAL'] * 100 for c in target_counts]

    bars2 = ax2.bar(grades, target_pct, color=['red', 'orange', 'green'], alpha=0.7)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Target Distribution (120 episodes)')
    ax2.set_ylim(0, 80)

    for bar, pct, count in zip(bars2, target_pct, target_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n({count} ep)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def plot_gripper_distribution(df):
    """Analyze gripper behavior in existing data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Gripper range distribution
    ax1 = axes[0, 0]
    ax1.hist(df['gripper_range'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=15, color='red', linestyle='--', label='Desired min range')
    ax1.set_xlabel('Gripper Range (degrees)')
    ax1.set_ylabel('Episode Count')
    ax1.set_title('Gripper Range Distribution')
    ax1.legend()

    # Gripper min/max
    ax2 = axes[0, 1]
    ax2.scatter(df['gripper_min'], df['gripper_max'], alpha=0.6, c=df['gripper_range'], cmap='viridis')
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax2.axhline(y=20, color='green', linestyle=':', alpha=0.5, label='Open threshold')
    ax2.axhline(y=10, color='red', linestyle=':', alpha=0.5, label='Close threshold')
    ax2.set_xlabel('Gripper Min (degrees)')
    ax2.set_ylabel('Gripper Max (degrees)')
    ax2.set_title('Gripper Min vs Max')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Has gripping vs quality grade
    ax3 = axes[1, 0]
    grip_by_grade = df.groupby('quality_grade')['has_gripping'].apply(lambda x: (x == True).sum())
    no_grip_by_grade = df.groupby('quality_grade')['has_gripping'].apply(lambda x: (x == False).sum())

    x = np.arange(len(grip_by_grade))
    width = 0.35

    ax3.bar(x - width/2, grip_by_grade, width, label='Has gripping', color='green', alpha=0.7)
    ax3.bar(x + width/2, no_grip_by_grade, width, label='No gripping', color='red', alpha=0.7)
    ax3.set_xlabel('Quality Grade')
    ax3.set_ylabel('Episode Count')
    ax3.set_title('Gripping Behavior by Quality Grade')
    ax3.set_xticks(x)
    ax3.set_xticklabels(grip_by_grade.index)
    ax3.legend()

    # Elbow depth vs gripper range
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['min_elbow'], df['gripper_range'],
                         c=df['quality_grade'].map({'DEEP': 0, 'APPROACH': 1, 'SHALLOW': 2}),
                         cmap='RdYlGn', alpha=0.6, s=50)
    ax4.axvline(x=-30, color='red', linestyle='--', alpha=0.5, label='DEEP threshold')
    ax4.axvline(x=-10, color='orange', linestyle='--', alpha=0.5, label='APPROACH threshold')
    ax4.axhline(y=15, color='blue', linestyle=':', alpha=0.5, label='Min gripper range')
    ax4.set_xlabel('Min Elbow (degrees)')
    ax4.set_ylabel('Gripper Range (degrees)')
    ax4.set_title('Elbow Depth vs Gripper Activity')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_collection_guide_markdown(current_stats, targets, to_remove, guidance):
    """Generate a markdown guide for data collectors."""
    md = []

    md.append("# Data Collection Strategy - Phase 1 Diagnosis (Step 4)")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append("**Root Cause Identified:** 68% SHALLOW data → model learned *wrong* direction")
    md.append("- Model thinks: APPROACH position = LIFT (go up)")
    md.append("- Should be: APPROACH position = REACH DOWN (go down)")
    md.append("- Gripper bias: Most frames have gripper closed → model never learned to open")
    md.append("")
    md.append("**Solution:** Rebalance dataset to 50% DEEP, 30% APPROACH, 20% SHALLOW")
    md.append("")

    md.append("## Current Dataset Analysis (50 episodes)")
    md.append("")
    md.append("| Grade | Count | Percentage |")
    md.append("|-------|-------|------------|")
    for grade in ['DEEP', 'APPROACH', 'SHALLOW']:
        count = current_stats['grade_counts'].get(grade, 0)
        pct = current_stats['grade_pct'].get(grade, 0)
        md.append(f"| {grade} | {count} | {pct}% |")
    md.append("")

    md.append(f"**Problematic Episodes:** {current_stats['problematic_count']} episodes should be removed")
    md.append(f"- Static (no motion): {current_stats['static_count']}")
    md.append(f"- No gripping: {current_stats['no_gripping_count']}")
    md.append("")

    md.append("### Gripper Analysis")
    md.append("")
    md.append(f"- Average gripper range: {current_stats['gripper_stats']['mean_range']:.1f}°")
    md.append(f"- Average gripper min: {current_stats['gripper_stats']['mean_min']:.1f}°")
    md.append(f"- Average gripper max: {current_stats['gripper_stats']['mean_max']:.1f}°")
    md.append("")
    md.append("**Problem:** Gripper stays mostly closed (min ≈ 1-2°), rarely opens wide.")
    md.append("")

    md.append("## Target Distribution (120 total episodes)")
    md.append("")
    md.append("| Grade | Target Count | Percentage | Requirements |")
    md.append("|-------|--------------|------------|--------------|")
    for grade in ['DEEP', 'APPROACH', 'SHALLOW']:
        count = targets[grade]['count']
        pct = count / targets['TOTAL'] * 100
        md.append(f"| {grade} | {count} | {pct:.0f}% | See below |")
    md.append("")

    for grade in ['DEEP', 'APPROACH', 'SHALLOW']:
        md.append(f"### {grade} Requirements")
        md.append("")
        md.append(f"**Description:** {targets[grade]['description']}")
        md.append("")
        md.append("**Requirements:**")
        for req in targets[grade]['requirements']:
            md.append(f"- {req}")
        md.append("")

    md.append("## Episodes to Remove")
    md.append("")
    md.append(f"Remove {len(to_remove)} episodes from the dataset:")
    md.append("")
    md.append("```")
    for ep_id in to_remove:
        md.append(f"Episode {ep_id}")
    md.append("```")
    md.append("")

    md.append("## Collection Protocol")
    md.append("")
    md.append("### Before Starting")
    md.append("")
    md.append("1. **Camera Position:** Ensure camera is FIXED (clamp/tripod). Any movement = all data invalid!")
    md.append("2. **Workspace Setup:** Clear workspace, consistent object placement")
    md.append("3. **Robot State:** Power cycle, run `scan_servos.py` if needed")
    md.append("4. **Lighting:** Consistent lighting (no shadows/glare)")
    md.append("")

    md.append("### During Collection")
    md.append("")
    md.append("**Real-Time Display (to be added to collect_data_manual.py):**")
    md.append("")
    md.append("```")
    md.append("Current Progress:")
    md.append("  DEEP:     [####------] 25/60  (42%)")
    md.append("  APPROACH: [###-------] 12/36  (33%)")
    md.append("  SHALLOW:  [##--------]  8/24  (33%)")
    md.append("")
    md.append("NEXT: Collect DEEP grasp (elbow < -30°)")
    md.append("```")
    md.append("")

    md.append("### Per-Episode Checklist")
    md.append("")
    md.append("Before accepting an episode, verify:")
    md.append("")
    md.append("- [ ] Elbow moved > 10° (not static)")
    md.append("- [ ] Gripper opened (>20°) AND closed (<10°)")
    md.append("- [ ] Smooth, controlled motion (no jerks)")
    md.append("- [ ] Target elbow depth achieved:")
    md.append("  - DEEP: elbow < -30°")
    md.append("  - APPROACH: -30° < elbow < -10°")
    md.append("  - SHALLOW: elbow > -10°")
    md.append("- [ ] RGB camera captured all frames")
    md.append("")

    md.append("### Collection Tips")
    md.append("")
    md.append("**For DEEP grasps (elbow < -30°):**")
    md.append("- Start with arm raised (elbow ≈ 30-50°)")
    md.append("- Open gripper WIDE (>30°)")
    md.append("- Slowly lower arm (bend elbow DOWN to -40° ~ -60°)")
    md.append("- Close gripper around object (<5°)")
    md.append("- Lift back up")
    md.append("")
    md.append("**For APPROACH grasps (-30° < elbow < -10°):**")
    md.append("- Medium height start")
    md.append("- Open gripper (>20°)")
    md.append("- Lower to -15° ~ -25°")
    md.append("- Close gripper")
    md.append("- Slight lift")
    md.append("")
    md.append("**For SHALLOW grasps (elbow > -10°):**")
    md.append("- Surface-level operations")
    md.append("- Still need gripper activity!")
    md.append("- Avoid static hovering")
    md.append("")

    md.append("## Post-Collection Validation")
    md.append("")
    md.append("After collecting all episodes, run:")
    md.append("")
    md.append("```bash")
    md.append("python data_episode_quality.py")
    md.append("python data_distribution_simple.py")
    md.append("```")
    md.append("")
    md.append("Verify:")
    md.append("")
    md.append("- Distribution matches target (50% DEEP, 30% APPROACH, 20% SHALLOW)")
    md.append("- All episodes have gripper activity (range > 15°)")
    md.append("- No static episodes (elbow range > 10°)")
    md.append("- Gripper open/close cycle visible in most episodes")
    md.append("")

    md.append("## Implementation Notes")
    md.append("")
    md.append("To add real-time guidance to `collect_data_manual.py`:")
    md.append("")
    md.append("1. Load `analysis_corrected.csv` at startup")
    md.append("2. Count existing DEEP/APPROACH/SHALLOW episodes")
    md.append("3. Display progress bars after each episode")
    md.append("4. Recommend next episode type based on targets")
    md.append("5. Compute metrics immediately after recording (min_elbow, gripper_range)")
    md.append("6. Show ACCEPT/RETRY prompt with reasons")
    md.append("")

    return "\n".join(md)


def main():
    print("=" * 80)
    print("DATA COLLECTION STRATEGY - Phase 1 Diagnosis (Step 4)")
    print("=" * 80)

    # Load existing data
    print("\nLoading existing dataset analysis...")
    df = load_existing_analysis()

    # Analyze current state
    print("Analyzing current distribution...")
    current_stats = analyze_current_distribution(df)

    print(f"\nCurrent dataset: {current_stats['total']} episodes")
    print("\nDistribution:")
    for grade in ['DEEP', 'APPROACH', 'SHALLOW']:
        count = current_stats['grade_counts'].get(grade, 0)
        pct = current_stats['grade_pct'].get(grade, 0)
        print(f"  {grade:<10} {count:>3} episodes ({pct:>5.1f}%)")

    print(f"\nProblematic episodes: {current_stats['problematic_count']}")
    print(f"  - Static: {current_stats['static_count']}")
    print(f"  - No gripping: {current_stats['no_gripping_count']}")

    # Design targets
    print("\nDesigning target distribution...")
    targets = design_collection_targets()

    print(f"\nTarget: {targets['TOTAL']} episodes")
    print("\nTarget distribution:")
    for grade in ['DEEP', 'APPROACH', 'SHALLOW']:
        count = targets[grade]['count']
        pct = count / targets['TOTAL'] * 100
        print(f"  {grade:<10} {count:>3} episodes ({pct:>5.1f}%)")

    # Identify episodes to remove
    print("\nIdentifying episodes to remove...")
    to_remove, reasons = identify_episodes_to_remove(df)

    print(f"\nEpisodes to remove: {len(to_remove)}")
    for ep_id in to_remove[:10]:  # Show first 10
        print(f"  Episode {ep_id}: {reasons[ep_id]}")
    if len(to_remove) > 10:
        print(f"  ... and {len(to_remove) - 10} more")

    # Real-time guidance spec
    print("\nDesigning real-time collection guidance...")
    guidance = create_realtime_guidance_spec()

    # Generate plots
    print("\nGenerating visualizations...")
    output_dir = Path('/home/cgxr/Documents/Robotics/RoArm_Project/analysis_outputs')
    output_dir.mkdir(exist_ok=True)

    # Distribution comparison
    fig1 = plot_current_vs_target_distribution(current_stats, targets)
    fig1.savefig(output_dir / 'data_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {output_dir / 'data_distribution_comparison.png'}")

    # Gripper analysis
    fig2 = plot_gripper_distribution(df)
    fig2.savefig(output_dir / 'data_gripper_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {output_dir / 'data_gripper_analysis.png'}")

    # Generate markdown guide
    print("\nGenerating collection guide...")
    md_content = generate_collection_guide_markdown(current_stats, targets, to_remove, guidance)

    guide_path = output_dir / 'DATA_COLLECTION_GUIDE.md'
    with open(guide_path, 'w') as f:
        f.write(md_content)
    print(f"  Saved: {guide_path}")

    # Save structured data for integration with collect_data_manual.py
    print("\nSaving structured guidance for collect_data_manual.py...")
    guidance_json = {
        'targets': {grade: targets[grade]['count'] for grade in ['DEEP', 'APPROACH', 'SHALLOW']},
        'thresholds': {
            'deep_elbow': -30,
            'approach_elbow': -10,
            'min_gripper_range': 15,
            'min_elbow_range': 10,
        },
        'remove_episodes': to_remove,
        'current_counts': {
            grade: int(current_stats['grade_counts'].get(grade, 0))
            for grade in ['DEEP', 'APPROACH', 'SHALLOW']
        }
    }

    guidance_json_path = output_dir / 'collection_guidance.json'
    with open(guidance_json_path, 'w') as f:
        json.dump(guidance_json, f, indent=2)
    print(f"  Saved: {guidance_json_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nCurrent problem:")
    print("  - 68% SHALLOW data → model learned 'APPROACH = LIFT' (WRONG!)")
    print("  - Gripper bias → model never learned to open gripper")
    print("  - Result: Model goes UP instead of DOWN, gripper stays closed")

    print("\nSolution:")
    print("  - Collect 60 DEEP + 36 APPROACH + 24 SHALLOW = 120 episodes")
    print("  - Emphasize gripper open/close cycles")
    print("  - Remove 8 static/no-gripping episodes from existing data")

    print("\nNext steps:")
    print("  1. Review DATA_COLLECTION_GUIDE.md")
    print("  2. (Optional) Integrate real-time guidance into collect_data_manual.py")
    print("  3. Collect 70 new episodes (to reach 120 total after removing 8)")
    print("  4. Prioritize DEEP grasps (need 51 more DEEP episodes)")
    print("  5. Retrain with new balanced dataset")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
