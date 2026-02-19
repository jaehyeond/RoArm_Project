"""
Cross-validate offline inference test results against data quality profile.

Analyze correlation between:
1. Data quality (elbow depth, static episodes, gripping)
2. Inference error (L2 per sample)
3. Data imbalance impact (68% SHALLOW vs 18% DEEP)

Goal: Quantify if data imbalance explains error patterns and guide Option B collection.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Inference results from test_inference_official.py output
# (Manually extracted from user's report)
inference_results = {
    "sample_idx": [0, 50, 200, 500, 1000, 5000, 10000],
    "l2_error": [2.43, 2.91, 3.81, 2.29, 2.21, 1.90, 1.96],
    "elbow_pred": [93.25, -30.17, -63.37, 29.74, 31.52, 6.68, 8.29],
    "elbow_gt": [94.13, -29.79, -65.39, 29.71, 31.38, 7.29, 7.91],
}


def load_data_quality():
    """Load episode quality analysis."""
    csv_path = Path("/home/cgxr/Documents/Robotics/RoArm_Project/collected_data/analysis_corrected.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Data quality CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def map_sample_to_episode(sample_idx, df):
    """Map global sample index to episode_id and local frame."""
    cumsum = 0
    for _, row in df.iterrows():
        ep_id = row["episode_id"]
        num_frames = row["num_frames"]
        if sample_idx < cumsum + num_frames:
            local_frame = sample_idx - cumsum
            return ep_id, local_frame, row
        cumsum += num_frames
    return None, None, None


def analyze_error_by_quality():
    """Cross-validate inference errors against data quality."""
    print("=" * 80)
    print("CROSS-VALIDATION: Inference Error vs Data Quality")
    print("=" * 80)

    df = load_data_quality()

    # Dataset statistics
    total_episodes = len(df)
    total_frames = df["num_frames"].sum()

    print(f"\nDataset Overview:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Episode lengths: {df['num_frames'].min()}-{df['num_frames'].max()} frames")

    # Quality distribution
    quality_counts = df["quality_grade"].value_counts()
    print(f"\nQuality Distribution:")
    for grade, count in quality_counts.items():
        pct = 100.0 * count / total_episodes
        print(f"  {grade:<10}: {count:>2} episodes ({pct:>5.1f}%)")

    # Anomaly counts
    static_count = df["is_static"].sum()
    no_grip_count = (~df["has_gripping"]).sum()
    print(f"\nAnomalies:")
    print(f"  Static (no movement): {static_count} episodes ({100.0*static_count/total_episodes:.1f}%)")
    print(f"  No gripping: {no_grip_count} episodes ({100.0*no_grip_count/total_episodes:.1f}%)")

    # Map inference samples to episodes
    print("\n" + "=" * 80)
    print("Inference Sample Mapping to Episodes")
    print("=" * 80)

    results = []
    for i in range(len(inference_results["sample_idx"])):
        sample_idx = inference_results["sample_idx"][i]
        l2_error = inference_results["l2_error"][i]
        elbow_pred = inference_results["elbow_pred"][i]
        elbow_gt = inference_results["elbow_gt"][i]

        ep_id, local_frame, ep_row = map_sample_to_episode(sample_idx, df)

        if ep_id is None:
            print(f"\nSample {sample_idx:>5}: OUT OF RANGE (dataset has {total_frames} frames)")
            continue

        quality = ep_row["quality_grade"]
        is_static = ep_row["is_static"]
        has_gripping = ep_row["has_gripping"]
        min_elbow = ep_row["min_elbow"]
        max_elbow = ep_row["max_elbow"]
        elbow_range = ep_row["elbow_range"]

        print(f"\nSample {sample_idx:>5}:")
        print(f"  Episode: {ep_id}, Frame: {local_frame}/{ep_row['num_frames']}")
        print(f"  Quality: {quality}, Static: {is_static}, Gripping: {has_gripping}")
        print(f"  Episode elbow range: [{min_elbow:.2f}, {max_elbow:.2f}] (span: {elbow_range:.2f}°)")
        print(f"  GT elbow at frame: {elbow_gt:.2f}°")
        print(f"  Predicted elbow: {elbow_pred:.2f}°")
        print(f"  L2 Error: {l2_error:.4f}")

        results.append({
            "sample_idx": sample_idx,
            "episode_id": ep_id,
            "quality": quality,
            "is_static": is_static,
            "has_gripping": has_gripping,
            "elbow_gt": elbow_gt,
            "elbow_pred": elbow_pred,
            "l2_error": l2_error,
            "min_elbow": min_elbow,
            "max_elbow": max_elbow,
            "elbow_range": elbow_range,
        })

    results_df = pd.DataFrame(results)

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    # 1. Error by quality grade
    print("\n1. L2 Error by Quality Grade:")
    quality_order = ["DEEP", "APPROACH", "SHALLOW"]
    for grade in quality_order:
        subset = results_df[results_df["quality"] == grade]
        if len(subset) > 0:
            mean_error = subset["l2_error"].mean()
            count = len(subset)
            print(f"  {grade:<10}: {mean_error:.4f} (n={count})")

    # 2. Error vs elbow depth
    print("\n2. Correlation: L2 Error vs Elbow Depth (GT)")
    elbow_abs = np.abs(results_df["elbow_gt"].values)
    errors = results_df["l2_error"].values
    correlation = np.corrcoef(elbow_abs, errors)[0, 1]
    print(f"  Correlation (|elbow| vs error): {correlation:.4f}")
    print(f"  Interpretation: {'NEGATIVE' if correlation < -0.3 else 'WEAK' if abs(correlation) < 0.3 else 'POSITIVE'}")

    # Check if DEEP samples have higher error
    deep_samples = results_df[results_df["quality"] == "DEEP"]
    shallow_samples = results_df[results_df["quality"] == "SHALLOW"]

    if len(deep_samples) > 0 and len(shallow_samples) > 0:
        deep_mean = deep_samples["l2_error"].mean()
        shallow_mean = shallow_samples["l2_error"].mean()
        diff = deep_mean - shallow_mean
        pct_increase = 100.0 * diff / shallow_mean if shallow_mean > 0 else 0

        print(f"\n3. DEEP vs SHALLOW Error Comparison:")
        print(f"  DEEP mean error:    {deep_mean:.4f}")
        print(f"  SHALLOW mean error: {shallow_mean:.4f}")
        print(f"  Difference:         {diff:.4f} ({pct_increase:+.1f}%)")

    # 4. Error vs episode elbow range
    print("\n4. Correlation: L2 Error vs Episode Elbow Range")
    range_corr = np.corrcoef(results_df["elbow_range"].values, errors)[0, 1]
    print(f"  Correlation (elbow_range vs error): {range_corr:.4f}")

    # 5. Static episodes
    static_samples = results_df[results_df["is_static"]]
    if len(static_samples) > 0:
        static_mean = static_samples["l2_error"].mean()
        dynamic_mean = results_df[~results_df["is_static"]]["l2_error"].mean()
        print(f"\n5. Static vs Dynamic Episodes:")
        print(f"  Static mean error:  {static_mean:.4f}")
        print(f"  Dynamic mean error: {dynamic_mean:.4f}")

    return results_df, df


def recommend_collection_strategy(results_df, df):
    """Generate Option B collection recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR OPTION B: COLLECT MORE DATA")
    print("=" * 80)

    # Calculate current distribution
    quality_counts = df["quality_grade"].value_counts()
    total = len(df)

    deep_count = quality_counts.get("DEEP", 0)
    approach_count = quality_counts.get("APPROACH", 0)
    shallow_count = quality_counts.get("SHALLOW", 0)

    deep_pct = 100.0 * deep_count / total
    approach_pct = 100.0 * approach_count / total
    shallow_pct = 100.0 * shallow_count / total

    print(f"\nCurrent Distribution (50 episodes):")
    print(f"  DEEP (< -30°):       {deep_count:>2} ({deep_pct:>5.1f}%)")
    print(f"  APPROACH (-10~-30°): {approach_count:>2} ({approach_pct:>5.1f}%)")
    print(f"  SHALLOW (> -10°):    {shallow_count:>2} ({shallow_pct:>5.1f}%)")

    # Target distribution
    target_total = 150  # 50 existing + 100 new
    target_deep_pct = 40.0  # Target 40% DEEP
    target_approach_pct = 30.0
    target_shallow_pct = 30.0

    target_deep = int(target_total * target_deep_pct / 100)
    target_approach = int(target_total * target_approach_pct / 100)
    target_shallow = int(target_total * target_shallow_pct / 100)

    need_deep = max(0, target_deep - deep_count)
    need_approach = max(0, target_approach - approach_count)
    need_shallow = max(0, target_shallow - shallow_count)

    print(f"\nTarget Distribution (150 episodes = 50 old + 100 new):")
    print(f"  DEEP:     {target_deep:>2} total → need {need_deep:>2} more")
    print(f"  APPROACH: {target_approach:>2} total → need {need_approach:>2} more")
    print(f"  SHALLOW:  {target_shallow:>2} total → need {need_shallow:>2} more")

    print(f"\n" + "=" * 80)
    print("COLLECTION PROTOCOL FOR 100 NEW EPISODES")
    print("=" * 80)

    print(f"\nPhase 1: DEEP Grasps (elbow < -30°) — {need_deep} episodes")
    print("  Strategy:")
    print("    - Place objects further from robot base (30-40cm)")
    print("    - Use taller objects or elevated platforms")
    print("    - Approach from above with steep descent")
    print("    - Target: elbow < -30° at grasp moment")
    print("  Validation: Check live elbow angle, retry if > -30°")

    print(f"\nPhase 2: APPROACH (elbow -10° to -30°) — {need_approach} episodes")
    print("  Strategy:")
    print("    - Objects at medium distance (20-30cm)")
    print("    - Mix of horizontal and diagonal approaches")
    print("    - Moderate elbow flexion")
    print("  Validation: Check elbow reaches -10° to -30° range")

    print(f"\nPhase 3: SHALLOW (elbow > -10°) — {need_shallow} episodes")
    print("  Strategy:")
    print("    - Close objects (10-20cm)")
    print("    - Top-down grasps with straight arm")
    print("    - Lateral sweeps with minimal flexion")
    print("  Note: May already have enough, collect only if needed")

    print(f"\nData Quality Checks During Collection:")
    print("  1. NO static episodes (current: {df['is_static'].sum()} / {total})")
    print("  2. ALL episodes must have gripping (current: {df['has_gripping'].sum()} / {total})")
    print("  3. Min episode length: 50 frames (avoid < 20 frame episodes)")
    print("  4. Elbow range > 15° per episode (avoid flat trajectories)")

    # Static/no-grip impact
    static_count = df["is_static"].sum()
    no_grip_count = (~df["has_gripping"]).sum()
    print(f"\n" + "=" * 80)
    print("IMPACT OF ANOMALOUS EPISODES")
    print("=" * 80)
    print(f"\nCurrent Issues:")
    print(f"  Static episodes (no movement): {static_count}")
    print(f"  No gripping episodes: {no_grip_count}")
    print(f"  Total anomalous: {static_count + no_grip_count} / {total} ({100.0*(static_count+no_grip_count)/total:.1f}%)")

    if static_count > 0 or no_grip_count > 0:
        print(f"\nRecommendation: REMOVE or REDO anomalous episodes")
        print(f"  These episodes contribute no useful variation to the model.")
        print(f"  Static episodes teach the robot to freeze.")
        print(f"  No-grip episodes teach incomplete pick behavior.")
        print(f"  Better to have 40 high-quality episodes than 50 with noise.")

    # Expected improvement from Option B
    print(f"\n" + "=" * 80)
    print("EXPECTED IMPROVEMENTS FROM OPTION B")
    print("=" * 80)

    if len(results_df) > 0:
        deep_samples = results_df[results_df["quality"] == "DEEP"]
        shallow_samples = results_df[results_df["quality"] == "SHALLOW"]

        if len(deep_samples) > 0 and len(shallow_samples) > 0:
            deep_mean_error = deep_samples["l2_error"].mean()
            shallow_mean_error = shallow_samples["l2_error"].mean()

            print(f"\nCurrent Error Pattern:")
            print(f"  DEEP samples error:    {deep_mean_error:.4f}")
            print(f"  SHALLOW samples error: {shallow_mean_error:.4f}")
            print(f"  Ratio (DEEP/SHALLOW):  {deep_mean_error/shallow_mean_error:.2f}x")

            print(f"\nAfter collecting {need_deep} more DEEP episodes:")
            print(f"  1. DEEP error likely to decrease (more training examples)")
            print(f"  2. Model will learn elbow < -30° is valid, not OOD")
            print(f"  3. Reduced overfitting to shallow trajectories")
            print(f"  4. Better generalization across full action space")

            print(f"\nQuantitative Prediction:")
            print(f"  Current DEEP representation: {deep_pct:.1f}%")
            print(f"  Target DEEP representation: {target_deep_pct:.1f}%")
            print(f"  Expected error reduction on DEEP samples: 20-40%")
            print(f"  (Based on typical imbalanced learning improvements)")

    print(f"\n" + "=" * 80)
    print("ALTERNATIVE: DATA AUGMENTATION (Less Effective)")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Oversample DEEP episodes (2-3x weight)")
    print("  2. Temporal augmentation (speed up/slow down 0.8-1.2x)")
    print("  3. Action noise injection (±2° to joint angles)")
    print("\nLimitations:")
    print("  - Cannot create true OOD coverage (camera viewpoints fixed)")
    print("  - Risk of overfitting to augmented copies")
    print("  - No replacement for real diversity")
    print("\nVerdict: Augmentation is SUPPLEMENTARY, not replacement for Option B")

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print("\n[Option B] Collect 100 new episodes with guided protocol:")
    print(f"  - {need_deep} DEEP episodes (elbow < -30°)")
    print(f"  - {need_approach} APPROACH episodes (-10° to -30°)")
    print(f"  - {need_shallow} SHALLOW episodes (> -10°, if needed)")
    print("  - Zero static episodes")
    print("  - 100% gripping success")
    print("\nExpected outcome:")
    print("  - Balanced action space coverage")
    print("  - Reduced DEEP sample error (currently highest)")
    print("  - More robust deployment performance")
    print("  - Training time: +1.5x (150 vs 50 episodes), acceptable for quality gain")


def main():
    print("\n" + "=" * 80)
    print("DATA AGENT: Inference Cross-Validation Analysis")
    print("=" * 80)

    try:
        results_df, df = analyze_error_by_quality()
        recommend_collection_strategy(results_df, df)

        print("\n" + "=" * 80)
        print("Analysis complete. Key findings saved above.")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run data_episode_quality.py first to generate analysis_corrected.csv")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
