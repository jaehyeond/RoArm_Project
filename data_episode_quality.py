"""
Per-Episode Quality Analysis for RoArm M3 SmolVLA Dataset
Analyzes elbow range, gripper opening, and classifies episode quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Joint indices
BASE, SHOULDER, ELBOW, WRIST_PITCH, WRIST_ROLL, GRIPPER = 0, 1, 2, 3, 4, 5

def analyze_episodes():
    """Analyze per-episode quality metrics."""

    # Load dataset
    parquet_path = Path("E:/RoArm_Project/lerobot_dataset_v3/data/chunk-000/file-000.parquet")
    print(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total frames: {len(df)}")

    # Get episode info
    episodes = df['episode_index'].unique()
    print(f"Total episodes: {len(episodes)}")

    # Analysis results
    results = []

    for ep_idx in episodes:
        ep_data = df[df['episode_index'] == ep_idx]

        # Extract actions (should be shape (N, 6))
        # Check column structure
        if 'action' in df.columns:
            actions = np.stack(ep_data['action'].values)
        else:
            # Try individual action columns
            action_cols = [col for col in df.columns if col.startswith('action')]
            if action_cols:
                actions = ep_data[action_cols].values
            else:
                print(f"ERROR: Cannot find action data in columns: {df.columns.tolist()}")
                return

        elbow_vals = actions[:, ELBOW]
        gripper_vals = actions[:, GRIPPER]

        # Elbow metrics
        min_elbow = elbow_vals.min()
        max_elbow = elbow_vals.max()
        elbow_range = max_elbow - min_elbow
        mean_elbow = elbow_vals.mean()

        # Frame where elbow first goes negative
        negative_frames = np.where(elbow_vals < 0)[0]
        first_negative_frame = negative_frames[0] if len(negative_frames) > 0 else None
        pct_negative = (elbow_vals < 0).sum() / len(elbow_vals) * 100

        # Frame where elbow < -20 (good grasp zone)
        grasp_frames = np.where(elbow_vals < -20)[0]
        first_grasp_frame = grasp_frames[0] if len(grasp_frames) > 0 else None
        pct_grasp_zone = (elbow_vals < -20).sum() / len(elbow_vals) * 100

        # Gripper metrics
        max_gripper = gripper_vals.max()
        mean_gripper = gripper_vals.mean()

        # Frame where gripper opens > 30
        open_frames = np.where(gripper_vals > 30)[0]
        first_open_frame = open_frames[0] if len(open_frames) > 0 else None
        pct_open = (gripper_vals > 30).sum() / len(gripper_vals) * 100

        # Quality classification
        # Good: elbow reaches < -20, gripper opens > 30, sufficient range
        is_good = (min_elbow < -20) and (max_gripper > 30) and (elbow_range > 40)

        results.append({
            'episode': ep_idx,
            'frames': len(ep_data),
            'min_elbow': min_elbow,
            'max_elbow': max_elbow,
            'mean_elbow': mean_elbow,
            'elbow_range': elbow_range,
            'pct_negative': pct_negative,
            'first_negative_frame': first_negative_frame,
            'pct_grasp_zone': pct_grasp_zone,
            'first_grasp_frame': first_grasp_frame,
            'max_gripper': max_gripper,
            'mean_gripper': mean_gripper,
            'pct_gripper_open': pct_open,
            'first_open_frame': first_open_frame,
            'quality': 'GOOD' if is_good else 'BAD'
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total episodes: {len(results_df)}")
    print(f"Good episodes: {(results_df['quality'] == 'GOOD').sum()}")
    print(f"Bad episodes: {(results_df['quality'] == 'BAD').sum()}")
    print(f"Good episode rate: {(results_df['quality'] == 'GOOD').sum() / len(results_df) * 100:.1f}%")

    print("\n" + "="*80)
    print("ELBOW STATISTICS")
    print("="*80)
    print(f"Mean min elbow: {results_df['min_elbow'].mean():.1f}°")
    print(f"Mean max elbow: {results_df['max_elbow'].mean():.1f}°")
    print(f"Mean elbow range: {results_df['elbow_range'].mean():.1f}°")
    print(f"Episodes with elbow < 0°: {(results_df['min_elbow'] < 0).sum()} ({(results_df['min_elbow'] < 0).sum() / len(results_df) * 100:.1f}%)")
    print(f"Episodes with elbow < -20°: {(results_df['min_elbow'] < -20).sum()} ({(results_df['min_elbow'] < -20).sum() / len(results_df) * 100:.1f}%)")
    print(f"Episodes with elbow < -40°: {(results_df['min_elbow'] < -40).sum()} ({(results_df['min_elbow'] < -40).sum() / len(results_df) * 100:.1f}%)")
    print(f"Episodes with elbow < -60°: {(results_df['min_elbow'] < -60).sum()} ({(results_df['min_elbow'] < -60).sum() / len(results_df) * 100:.1f}%)")

    print("\n" + "="*80)
    print("GRIPPER STATISTICS")
    print("="*80)
    print(f"Mean max gripper: {results_df['max_gripper'].mean():.1f}°")
    print(f"Episodes with gripper > 30°: {(results_df['max_gripper'] > 30).sum()} ({(results_df['max_gripper'] > 30).sum() / len(results_df) * 100:.1f}%)")
    print(f"Episodes with gripper > 50°: {(results_df['max_gripper'] > 50).sum()} ({(results_df['max_gripper'] > 50).sum() / len(results_df) * 100:.1f}%)")

    # Top 10 good episodes
    print("\n" + "="*80)
    print("TOP 10 EPISODES BY ELBOW REACH (lowest min_elbow)")
    print("="*80)
    top_episodes = results_df.nsmallest(10, 'min_elbow')[['episode', 'min_elbow', 'elbow_range', 'max_gripper', 'quality']]
    print(top_episodes.to_string(index=False))

    # Top 10 bad episodes
    print("\n" + "="*80)
    print("TOP 10 WORST EPISODES (highest min_elbow)")
    print("="*80)
    worst_episodes = results_df.nlargest(10, 'min_elbow')[['episode', 'min_elbow', 'elbow_range', 'max_gripper', 'quality']]
    print(worst_episodes.to_string(index=False))

    # Save to CSV
    output_path = Path("E:/RoArm_Project/data_episode_quality_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to {output_path}")

    return results_df

if __name__ == "__main__":
    analyze_episodes()
