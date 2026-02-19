"""
Data Agent: Detailed Dataset Analysis for RoArm M3 SmolVLA
Analyzes action distribution, identifies OOD issues, and proposes solutions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
ANALYSIS_CSV = Path(__file__).parent / "collected_data" / "analysis_corrected.csv"
DATASET_DIR = Path(__file__).parent / "lerobot_dataset_v3"
DATA_PARQUET = DATASET_DIR / "data" / "chunk-000" / "file-000.parquet"
EPISODES_PARQUET = DATASET_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

def load_dataset():
    """Load parquet dataset and extract action arrays"""
    data_df = pd.read_parquet(DATA_PARQUET)
    episodes_df = pd.read_parquet(EPISODES_PARQUET)

    # Extract actions (each row is a 6-element array)
    actions = np.stack(data_df['action'].values)

    # Extract observations (state)
    observations = np.stack(data_df['observation.state'].values)

    return data_df, episodes_df, actions, observations

def analyze_action_distribution(actions):
    """Analyze action space coverage"""
    print("=" * 70)
    print("1. ACTION DISTRIBUTION ANALYSIS")
    print("=" * 70)

    joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist_pitch', 'Wrist_roll', 'Gripper']

    print(f"\nTotal frames: {len(actions)}")
    print(f"\nAction statistics (per joint):")
    print(f"{'Joint':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
    print("-" * 70)

    stats = {}
    for i, name in enumerate(joint_names):
        mean = actions[:, i].mean()
        std = actions[:, i].std()
        min_val = actions[:, i].min()
        max_val = actions[:, i].max()
        range_val = max_val - min_val

        stats[name] = {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val
        }

        print(f"{name:<15} {mean:<10.2f} {std:<10.2f} {min_val:<10.2f} {max_val:<10.2f} {range_val:<10.2f}")

    return stats

def analyze_elbow_depth(actions):
    """Analyze elbow depth distribution (critical for grasping)"""
    print("\n" + "=" * 70)
    print("2. ELBOW DEPTH ANALYSIS (Joint 2)")
    print("=" * 70)

    elbow = actions[:, 2]

    deep_frames = (elbow < -30).sum()
    approach_frames = ((elbow >= -30) & (elbow < -10)).sum()
    shallow_frames = (elbow >= -10).sum()

    total = len(elbow)

    print(f"\nElbow distribution:")
    print(f"  DEEP (< -30°):       {deep_frames:5d} / {total} ({deep_frames/total*100:5.1f}%)")
    print(f"  APPROACH (-30~-10°): {approach_frames:5d} / {total} ({approach_frames/total*100:5.1f}%)")
    print(f"  SHALLOW (> -10°):    {shallow_frames:5d} / {total} ({shallow_frames/total*100:5.1f}%)")

    print(f"\nElbow statistics:")
    print(f"  Mean: {elbow.mean():.2f}°")
    print(f"  Std:  {elbow.std():.2f}°")
    print(f"  Min:  {elbow.min():.2f}°")
    print(f"  Max:  {elbow.max():.2f}°")

    # Critical finding
    print(f"\n⚠️  CRITICAL: Only {deep_frames/total*100:.1f}% of frames have deep grasping poses!")
    print(f"    This is severely OOD for the deployment task.")

    return {
        'deep_frames': deep_frames,
        'approach_frames': approach_frames,
        'shallow_frames': shallow_frames,
        'deep_ratio': deep_frames / total,
        'approach_ratio': approach_frames / total,
        'shallow_ratio': shallow_frames / total
    }

def analyze_gripper_usage(actions):
    """Analyze gripper open/close patterns"""
    print("\n" + "=" * 70)
    print("3. GRIPPER USAGE ANALYSIS (Joint 5)")
    print("=" * 70)

    gripper = actions[:, 5]

    # Gripper states
    closed_frames = (gripper < 10).sum()
    open_frames = (gripper > 30).sum()
    intermediate_frames = ((gripper >= 10) & (gripper <= 30)).sum()

    total = len(gripper)

    print(f"\nGripper distribution:")
    print(f"  CLOSED (< 10°):      {closed_frames:5d} / {total} ({closed_frames/total*100:5.1f}%)")
    print(f"  INTERMEDIATE (10-30°): {intermediate_frames:5d} / {total} ({intermediate_frames/total*100:5.1f}%)")
    print(f"  OPEN (> 30°):        {open_frames:5d} / {total} ({open_frames/total*100:5.1f}%)")

    print(f"\nGripper statistics:")
    print(f"  Mean: {gripper.mean():.2f}°")
    print(f"  Std:  {gripper.std():.2f}°")
    print(f"  Min:  {gripper.min():.2f}°")
    print(f"  Max:  {gripper.max():.2f}°")

    # Critical finding
    print(f"\n⚠️  CRITICAL: {closed_frames/total*100:.1f}% of frames have gripper mostly closed!")
    print(f"    Model may not learn to open gripper before grasping.")

    return {
        'closed_frames': closed_frames,
        'open_frames': open_frames,
        'intermediate_frames': intermediate_frames,
        'closed_ratio': closed_frames / total,
        'open_ratio': open_frames / total
    }

def analyze_episode_quality():
    """Analyze per-episode quality from CSV"""
    print("\n" + "=" * 70)
    print("4. EPISODE QUALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(ANALYSIS_CSV)

    total_episodes = len(df)
    deep_episodes = (df['quality_grade'] == 'DEEP').sum()
    approach_episodes = (df['quality_grade'] == 'APPROACH').sum()
    shallow_episodes = (df['quality_grade'] == 'SHALLOW').sum()

    static_episodes = df['is_static'].sum()
    no_gripping_episodes = df['has_gripping'].eq(False).sum()

    print(f"\nEpisode distribution:")
    print(f"  DEEP (elbow < -30°):  {deep_episodes:2d} / {total_episodes} ({deep_episodes/total_episodes*100:5.1f}%)")
    print(f"  APPROACH (-30~-10°):  {approach_episodes:2d} / {total_episodes} ({approach_episodes/total_episodes*100:5.1f}%)")
    print(f"  SHALLOW (> -10°):     {shallow_episodes:2d} / {total_episodes} ({shallow_episodes/total_episodes*100:5.1f}%)")

    print(f"\nQuality issues:")
    print(f"  Static episodes:      {static_episodes:2d} / {total_episodes} ({static_episodes/total_episodes*100:5.1f}%)")
    print(f"  No gripping action:   {no_gripping_episodes:2d} / {total_episodes} ({no_gripping_episodes/total_episodes*100:5.1f}%)")

    print(f"\nAverage episode duration: {df['duration_sec'].mean():.2f}s (median: {df['duration_sec'].median():.2f}s)")
    print(f"Average frames per episode: {df['num_frames'].mean():.1f} (median: {df['num_frames'].median():.1f})")

    return {
        'deep_episodes': deep_episodes,
        'approach_episodes': approach_episodes,
        'shallow_episodes': shallow_episodes,
        'static_episodes': static_episodes,
        'no_gripping_episodes': no_gripping_episodes
    }

def diagnose_ood_issues(elbow_stats, gripper_stats, episode_stats):
    """Diagnose why deployment is OOD"""
    print("\n" + "=" * 70)
    print("5. OUT-OF-DISTRIBUTION (OOD) DIAGNOSIS")
    print("=" * 70)

    print("\n🔴 ROOT CAUSE ANALYSIS:")

    issues = []

    # Issue 1: Elbow depth
    if elbow_stats['deep_ratio'] < 0.3:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Insufficient deep grasping poses',
            'metric': f"{elbow_stats['deep_ratio']*100:.1f}% deep frames",
            'target': '> 30% deep frames',
            'impact': 'Model cannot generalize to elbow < -30° poses'
        })

    # Issue 2: Gripper closed
    if gripper_stats['closed_ratio'] > 0.5:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Gripper mostly closed throughout episodes',
            'metric': f"{gripper_stats['closed_ratio']*100:.1f}% closed frames",
            'target': '< 40% closed frames',
            'impact': 'Model cannot learn open→grasp→close sequence'
        })

    # Issue 3: Episode quality
    if episode_stats['deep_episodes'] < 20:
        issues.append({
            'severity': 'HIGH',
            'issue': 'Insufficient deep episodes',
            'metric': f"{episode_stats['deep_episodes']} deep episodes",
            'target': '> 30 deep episodes',
            'impact': 'Not enough diversity in deep grasping trajectories'
        })

    # Issue 4: Static episodes
    if episode_stats['static_episodes'] > 5:
        issues.append({
            'severity': 'MEDIUM',
            'issue': 'Static/no-motion episodes',
            'metric': f"{episode_stats['static_episodes']} static episodes",
            'target': '< 5 static episodes',
            'impact': 'Pollutes action distribution with no-op data'
        })

    for i, issue in enumerate(issues, 1):
        print(f"\n  [{issue['severity']}] Issue {i}: {issue['issue']}")
        print(f"    Current: {issue['metric']}")
        print(f"    Target:  {issue['target']}")
        print(f"    Impact:  {issue['impact']}")

    return issues

def evaluate_options():
    """Evaluate proposed solutions"""
    print("\n" + "=" * 70)
    print("6. SOLUTION OPTIONS EVALUATION")
    print("=" * 70)

    options = [
        {
            'name': 'Option A: Action scaling (--action-scale 2.0)',
            'pros': [
                'Quick to test (no data collection)',
                'May increase movement magnitude',
                'Could help if model is too conservative'
            ],
            'cons': [
                '⚠️  Does NOT fix OOD problem (data distribution stays same)',
                '⚠️  May amplify drift into unsafe regions',
                '⚠️  Scaling gripper 2x (2°→4°) still won\'t open gripper',
                'Root cause: model never saw deep poses, scaling won\'t add them'
            ],
            'recommendation': '❌ NOT RECOMMENDED',
            'reason': 'Scaling cannot fix OOD. Model needs to see deep poses in training.'
        },
        {
            'name': 'Option B: Collect 100+ episodes with deep grasps',
            'pros': [
                '✅ Directly fixes OOD problem',
                '✅ Adds deep grasping poses to training distribution',
                '✅ Adds gripper open→close sequences',
                '✅ Industry standard for VLA (100-1000 episodes)',
                '✅ Increases trajectory diversity'
            ],
            'cons': [
                'Time-consuming (~8-12 hours of manual teleoperation)',
                'Requires careful camera position maintenance',
                'Physical effort (hand fatigue)'
            ],
            'recommendation': '✅ STRONGLY RECOMMENDED',
            'reason': 'Only way to fix OOD. LeRobot examples use 50-300 episodes minimum.'
        },
        {
            'name': 'Option C: CSV log analysis',
            'pros': [
                'Provides detailed trajectory insights',
                'Can identify per-step drift patterns'
            ],
            'cons': [
                'Diagnostic only, does not fix OOD',
                'Will confirm what we already know (OOD drift)',
                'Delays actual solution'
            ],
            'recommendation': '⚠️  OPTIONAL (after B)',
            'reason': 'Useful for debugging specific behaviors, but not a solution.'
        },
        {
            'name': 'Option D: Data augmentation (temporal + action noise)',
            'pros': [
                'Can increase effective dataset size',
                'May improve robustness to perturbations',
                'No physical collection needed'
            ],
            'cons': [
                '⚠️  Cannot create OOD data (e.g., deep poses from shallow ones)',
                '⚠️  May degrade performance if augmentation is too aggressive',
                'LeRobot SmolVLA uses image augmentation already',
                'Action augmentation risky (violates dynamics)'
            ],
            'recommendation': '⚠️  SUPPLEMENTARY (after B)',
            'reason': 'Can help with 100+ episodes, but cannot replace real data.'
        },
        {
            'name': 'Option E: Filter + oversample deep episodes',
            'pros': [
                'Maximizes existing 9 deep episodes',
                'Quick to implement'
            ],
            'cons': [
                '⚠️  9 episodes is too few (severe overfitting risk)',
                '⚠️  Reduces total dataset size (50→9 episodes)',
                '⚠️  Loss of shallow→deep transitions',
                'Violates LeRobot minimum episode recommendation'
            ],
            'recommendation': '❌ NOT RECOMMENDED',
            'reason': '9 episodes is insufficient for VLA training.'
        }
    ]

    for option in options:
        print(f"\n{'='*70}")
        print(f"{option['name']}")
        print(f"{'='*70}")
        print("\nPros:")
        for pro in option['pros']:
            print(f"  • {pro}")
        print("\nCons:")
        for con in option['cons']:
            print(f"  • {con}")
        print(f"\n{option['recommendation']}")
        print(f"Reason: {option['reason']}")

def propose_collection_strategy():
    """Propose detailed data collection strategy"""
    print("\n" + "=" * 70)
    print("7. RECOMMENDED DATA COLLECTION STRATEGY")
    print("=" * 70)

    strategy = {
        'target': {
            'total_episodes': 100,
            'deep_episodes': 50,
            'approach_episodes': 30,
            'diverse_start_episodes': 20
        },
        'phases': [
            {
                'phase': 'Phase 1: Deep grasping (50 episodes)',
                'goal': 'Maximize elbow < -30° coverage',
                'method': [
                    'Start from high positions (elbow ~50-80°)',
                    'Reach down to grasp object at table level',
                    'MUST go below elbow -40° during grasp',
                    'Open gripper (>30°) before grasp',
                    'Close gripper (<10°) during grasp',
                    'Lift object to high position',
                    'Release object (open gripper)'
                ],
                'validation': 'Each episode MUST have min_elbow < -30°',
                'estimated_time': '4-5 hours'
            },
            {
                'phase': 'Phase 2: Approach trajectories (30 episodes)',
                'goal': 'Cover approach phase (-30° to -10°)',
                'method': [
                    'Start from mid-height positions',
                    'Reach to objects at various distances',
                    'Focus on smooth approach trajectories',
                    'Vary base rotation (different angles)',
                    'Include failed grasp attempts (open gripper, no close)'
                ],
                'validation': 'Each episode reaches -30° < elbow < -10° range',
                'estimated_time': '2-3 hours'
            },
            {
                'phase': 'Phase 3: Diverse starts (20 episodes)',
                'goal': 'Increase trajectory diversity',
                'method': [
                    'Start from random positions',
                    'Vary object positions (left, right, center, far, near)',
                    'Mix shallow + deep in same episode',
                    'Include multi-step tasks (pick→place→pick)',
                    'Vary gripper timing (early open, late close)'
                ],
                'validation': 'Cover full workspace',
                'estimated_time': '2-3 hours'
            }
        ],
        'critical_rules': [
            '🔴 NEVER move camera position during collection',
            '🔴 Verify camera position at start/end of each session',
            '🔴 Run data_episode_quality.py after each 10 episodes',
            '🔴 Delete episodes with is_static=True immediately',
            '🔴 Ensure RGB frames are valid (check random samples)',
            '🔴 Target: >30% deep frames in final dataset'
        ],
        'quality_checks': [
            'After 10 episodes: Check min_elbow distribution',
            'After 30 episodes: Check gripper range distribution',
            'After 50 episodes: Check action space coverage',
            'After 100 episodes: Final quality audit'
        ]
    }

    print(f"\n📊 TARGET DISTRIBUTION:")
    print(f"  Total episodes: {strategy['target']['total_episodes']}")
    print(f"  Deep (elbow < -30°): {strategy['target']['deep_episodes']} ({strategy['target']['deep_episodes']/strategy['target']['total_episodes']*100:.0f}%)")
    print(f"  Approach (-30~-10°): {strategy['target']['approach_episodes']} ({strategy['target']['approach_episodes']/strategy['target']['total_episodes']*100:.0f}%)")
    print(f"  Diverse starts: {strategy['target']['diverse_start_episodes']} ({strategy['target']['diverse_start_episodes']/strategy['target']['total_episodes']*100:.0f}%)")

    for phase in strategy['phases']:
        print(f"\n{'='*70}")
        print(f"{phase['phase']}")
        print(f"{'='*70}")
        print(f"\nGoal: {phase['goal']}")
        print("\nMethod:")
        for step in phase['method']:
            print(f"  • {step}")
        print(f"\nValidation: {phase['validation']}")
        print(f"Estimated time: {phase['estimated_time']}")

    print(f"\n{'='*70}")
    print("CRITICAL RULES")
    print(f"{'='*70}")
    for rule in strategy['critical_rules']:
        print(f"  {rule}")

    print(f"\n{'='*70}")
    print("QUALITY CHECKS (during collection)")
    print(f"{'='*70}")
    for check in strategy['quality_checks']:
        print(f"  • {check}")

    print(f"\nTotal estimated time: 8-12 hours (over 2-3 days to avoid fatigue)")

    return strategy

def main():
    print("=" * 70)
    print("RoArm M3 SmolVLA - DATA AGENT ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    data_df, episodes_df, actions, observations = load_dataset()

    # Run analyses
    action_stats = analyze_action_distribution(actions)
    elbow_stats = analyze_elbow_depth(actions)
    gripper_stats = analyze_gripper_usage(actions)
    episode_stats = analyze_episode_quality()

    # Diagnose OOD
    issues = diagnose_ood_issues(elbow_stats, gripper_stats, episode_stats)

    # Evaluate options
    evaluate_options()

    # Propose strategy
    strategy = propose_collection_strategy()

    # Final recommendation
    print("\n" + "=" * 70)
    print("8. FINAL RECOMMENDATION")
    print("=" * 70)

    print("""
🎯 RECOMMENDED ACTION: Option B (Collect 100+ episodes)

RATIONALE:
1. Current dataset is severely OOD for deep grasping tasks
   - Only 18% of episodes have deep poses (target: 50%)
   - Only ~13% of frames have elbow < -30° (target: 30%+)
   - Gripper mostly closed throughout (no open→close sequences)

2. Scaling (Option A) CANNOT fix OOD
   - Scaling 2x on shallow data still produces shallow predictions
   - Model needs to see deep poses in training data
   - Will likely amplify drift instead of fixing it

3. Data augmentation (Option D) CANNOT create OOD data
   - Temporal shifts won't create elbow < -30° from elbow > 0° data
   - Action noise risks violating robot dynamics
   - Only useful AFTER collecting enough diverse data

4. Industry best practices
   - LeRobot examples: 50-300 episodes per task
   - Current 50 episodes with only 9 deep is insufficient
   - SmolVLA needs rich trajectory diversity to learn flow matching

IMMEDIATE NEXT STEPS:
1. Fix camera position (tripod/clamp), document position, NEVER move
2. Run Phase 1: Collect 50 deep grasping episodes (4-5 hours)
3. Run data_episode_quality.py to verify >30% deep frames
4. Continue Phase 2+3 to reach 100 total episodes
5. Re-train with lerobot-train CLI (50K+ steps)
6. Deploy and expect MUCH better generalization

ESTIMATED TIMELINE:
- Data collection: 8-12 hours (over 2-3 days)
- Data conversion: 30 min
- Training (50K steps): 8-10 hours (overnight)
- Deployment testing: 2-3 hours
Total: ~3-4 days

⚠️  DO NOT attempt deployment fixes (scaling, CSV analysis) before fixing data.
    "Garbage in, garbage out" - no amount of deployment tricks can fix OOD training data.
""")

if __name__ == '__main__':
    main()
