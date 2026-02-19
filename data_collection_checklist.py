"""
Data Collection Progress Tracker & Quality Validator
Run after every 10 episodes to ensure on-track progress
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ANALYSIS_CSV = Path(__file__).parent / "collected_data" / "analysis_corrected.csv"

def load_quality_data():
    """Load episode quality CSV"""
    if not ANALYSIS_CSV.exists():
        print(f"❌ Error: {ANALYSIS_CSV} not found!")
        print("   Run data_episode_quality.py first to generate analysis.")
        sys.exit(1)

    return pd.read_csv(ANALYSIS_CSV)

def check_progress(df):
    """Check collection progress against targets"""
    total = len(df)
    deep = (df['quality_grade'] == 'DEEP').sum()
    approach = (df['quality_grade'] == 'APPROACH').sum()
    shallow = (df['quality_grade'] == 'SHALLOW').sum()
    static = df['is_static'].sum()
    no_grip = df['has_gripping'].eq(False).sum()

    print("=" * 70)
    print("COLLECTION PROGRESS REPORT")
    print("=" * 70)

    print(f"\n📊 Current Status: {total} episodes collected")

    # Episode distribution
    print(f"\n{'='*70}")
    print("EPISODE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"  DEEP (< -30°):     {deep:3d} / {total:3d} ({deep/total*100:5.1f}%)  [Target: 50%]")
    print(f"  APPROACH (-30~-10°): {approach:3d} / {total:3d} ({approach/total*100:5.1f}%)  [Target: 30%]")
    print(f"  SHALLOW (> -10°):  {shallow:3d} / {total:3d} ({shallow/total*100:5.1f}%)  [Target: 20%]")

    # Quality issues
    print(f"\n{'='*70}")
    print("QUALITY ISSUES")
    print(f"{'='*70}")
    if static > 0:
        print(f"  ⚠️  Static episodes: {static} (DELETE IMMEDIATELY!)")
    if no_grip > 5:
        print(f"  ⚠️  No-gripping episodes: {no_grip} (> 5 is too many)")
    if static == 0 and no_grip <= 5:
        print(f"  ✅ No critical quality issues")

    # Progress toward 100 episodes
    print(f"\n{'='*70}")
    print("PROGRESS TOWARD 100 EPISODES")
    print(f"{'='*70}")

    targets = {
        'Deep': (50, deep),
        'Approach': (30, approach),
        'Diverse/Shallow': (20, shallow)
    }

    for name, (target, current) in targets.items():
        remaining = target - current
        progress = current / target * 100 if target > 0 else 0
        status = "✅" if current >= target else "🔄"
        print(f"  {status} {name:<18} {current:3d} / {target:3d} ({progress:5.1f}%)  [Remaining: {max(0, remaining):2d}]")

    total_remaining = 100 - total
    print(f"\n  Total episodes remaining: {total_remaining}")

def check_action_distribution():
    """Validate action space coverage from parquet"""
    dataset_dir = Path(__file__).parent / "lerobot_dataset_v3"
    data_parquet = dataset_dir / "data" / "chunk-000" / "file-000.parquet"

    if not data_parquet.exists():
        print("\n⚠️  Warning: Dataset not converted to LeRobot format yet")
        print("   Run convert_to_lerobot_v3.py to check frame-level distribution")
        return

    data_df = pd.read_parquet(data_parquet)
    actions = np.stack(data_df['action'].values)

    elbow = actions[:, 2]
    gripper = actions[:, 5]

    deep_frames = (elbow < -30).sum()
    closed_frames = (gripper < 10).sum()
    total_frames = len(elbow)

    print(f"\n{'='*70}")
    print("FRAME-LEVEL DISTRIBUTION (from parquet)")
    print(f"{'='*70}")
    print(f"  Total frames: {total_frames}")
    print(f"  Deep frames (elbow < -30°): {deep_frames} ({deep_frames/total_frames*100:.1f}%)  [Target: >30%]")
    print(f"  Closed gripper frames: {closed_frames} ({closed_frames/total_frames*100:.1f}%)  [Target: <40%]")

    if deep_frames / total_frames < 0.30:
        print(f"\n  ⚠️  WARNING: Deep frames < 30%. Need more deep grasping episodes!")
    else:
        print(f"\n  ✅ Deep frame ratio looks good!")

    if closed_frames / total_frames > 0.40:
        print(f"  ⚠️  WARNING: Too many closed gripper frames. Open gripper more at start!")
    else:
        print(f"  ✅ Gripper distribution looks good!")

def give_next_phase_guidance(df):
    """Suggest what to collect next"""
    total = len(df)
    deep = (df['quality_grade'] == 'DEEP').sum()
    approach = (df['quality_grade'] == 'APPROACH').sum()
    shallow = (df['quality_grade'] == 'SHALLOW').sum()

    print(f"\n{'='*70}")
    print("NEXT PHASE GUIDANCE")
    print(f"{'='*70}")

    if total < 10:
        print("\n🎯 PHASE 1a: Initial Deep Grasping (0-10 episodes)")
        print("   Focus: Get comfortable with deep grasps")
        print("   - Start high, reach down low (elbow < -40°)")
        print("   - Open gripper (>30°) before approach")
        print("   - Close gripper (<10°) during grasp")
        print("   - Verify with: min_elbow < -30° in CSV")
    elif total < 30:
        print("\n🎯 PHASE 1b: More Deep Grasping (10-30 episodes)")
        print("   Focus: Increase deep trajectory diversity")
        print("   - Vary object positions (left, right, center)")
        print("   - Vary starting heights")
        print("   - Vary base rotation angles")
        print("   - Target: 50% of episodes should be DEEP")
    elif total < 50:
        print("\n🎯 PHASE 1c: Complete Deep Set (30-50 episodes)")
        print("   Focus: Finish deep grasping coverage")
        print("   - Fill gaps in base rotation")
        print("   - Try objects at different distances")
        print("   - Ensure gripper opens WIDE (>40°) before grasp")
    elif total < 70:
        print("\n🎯 PHASE 2: Approach Trajectories (50-70 episodes)")
        print("   Focus: Mid-height approach sequences")
        print("   - Start from mid positions")
        print("   - Reach elbow -30° to -10° range")
        print("   - Smooth continuous motions")
        print("   - Include some grasp failures (open, but no close)")
    elif total < 100:
        print("\n🎯 PHASE 3: Diverse Starts & Complex Tasks (70-100 episodes)")
        print("   Focus: Full workspace coverage")
        print("   - Random starting positions")
        print("   - Multi-step: pick→place→pick")
        print("   - Mix shallow + deep in same episode")
        print("   - Edge cases (far reach, side angles)")
    else:
        print("\n🎉 PHASE COMPLETE: 100 episodes collected!")
        print("   Next steps:")
        print("   1. Run final quality audit: data_episode_quality.py")
        print("   2. Convert to LeRobot v3: convert_to_lerobot_v3.py")
        print("   3. Train model: run_official_train.py")
        print("   4. Deploy: deploy_smolvla.py")

    # Specific warnings
    if deep < total * 0.4 and total >= 20:
        print(f"\n  ⚠️  WARNING: Only {deep/total*100:.0f}% deep episodes (target: 50%)")
        print("      → Prioritize DEEP episodes in next collection session!")

    if approach < total * 0.2 and total >= 50:
        print(f"\n  ⚠️  WARNING: Only {approach/total*100:.0f}% approach episodes (target: 30%)")
        print("      → Add more mid-height approach trajectories!")

def main():
    print("\n" + "=" * 70)
    print("RoArm M3 Data Collection Checklist")
    print("=" * 70)

    df = load_quality_data()
    check_progress(df)
    check_action_distribution()
    give_next_phase_guidance(df)

    print("\n" + "=" * 70)
    print("✅ Checklist complete! Review warnings above.")
    print("=" * 70)
    print()

if __name__ == '__main__':
    main()
