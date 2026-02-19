"""
Create Validation Split for LeRobot Dataset

Splits dataset into train/val by selecting diverse episodes for validation:
- Episodes with elbow < -30° (deep grasping)
- Episodes with diverse base rotations
- Episodes with rapid gripper transitions

Usage:
    python train_create_validation_split.py --val-episodes 10
    python train_create_validation_split.py --val-episodes 10 --strategy diverse_joints
    python train_create_validation_split.py --val-episodes 10 --strategy random
"""

import argparse
import numpy as np
from pathlib import Path
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def analyze_episode_diversity(dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Compute diversity metrics for a single episode."""
    # Get all frames in this episode
    episode_data = dataset.episodes[episode_idx]
    from_idx = episode_data["from"]
    to_idx = episode_data["to"]

    # Collect actions for this episode
    actions = []
    for frame_idx in range(from_idx, to_idx):
        sample = dataset[frame_idx]
        action = sample.get("action")
        if action is not None:
            actions.append(action.numpy()[:6])

    if not actions:
        return {}

    actions = np.array(actions)

    # Compute metrics
    metrics = {
        "base_std": actions[:, 0].std(),
        "base_range": actions[:, 0].max() - actions[:, 0].min(),
        "elbow_min": actions[:, 2].min(),
        "elbow_max": actions[:, 2].max(),
        "elbow_std": actions[:, 2].std(),
        "wrist_r_std": actions[:, 4].std(),
        "wrist_r_range": actions[:, 4].max() - actions[:, 4].min(),
        "gripper_transitions": np.abs(np.diff(actions[:, 5])).sum(),  # Total gripper motion
        "overall_diversity": actions.std(axis=0).mean(),
    }

    return metrics


def select_validation_episodes_diverse(
    dataset: LeRobotDataset, num_val_episodes: int
) -> list[int]:
    """Select validation episodes based on diversity criteria."""

    print("\nAnalyzing episode diversity...")
    episode_metrics = []

    for ep_idx in range(dataset.num_episodes):
        metrics = analyze_episode_diversity(dataset, ep_idx)
        if metrics:
            episode_metrics.append((ep_idx, metrics))

    if not episode_metrics:
        raise ValueError("No valid episodes found in dataset")

    # Prioritize episodes with:
    # 1. Deep elbow extension (< -30°)
    # 2. High wrist_R diversity
    # 3. Rapid gripper transitions

    # Score each episode
    scores = []
    for ep_idx, metrics in episode_metrics:
        score = 0.0

        # Deep elbow (high priority)
        if metrics["elbow_min"] < -30:
            score += 10.0

        # Wrist_R diversity (high priority)
        score += metrics["wrist_r_std"] * 0.5

        # Gripper transitions (medium priority)
        score += metrics["gripper_transitions"] * 0.1

        # Base rotation diversity (low priority)
        score += metrics["base_std"] * 0.2

        scores.append((ep_idx, score, metrics))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Select top N episodes
    val_episodes = [ep_idx for ep_idx, score, metrics in scores[:num_val_episodes]]

    print(f"\nSelected {len(val_episodes)} validation episodes:")
    for i, (ep_idx, score, metrics) in enumerate(scores[:num_val_episodes]):
        print(f"  Episode {ep_idx:>3d} (score={score:>6.2f}): "
              f"elbow_min={metrics['elbow_min']:>6.1f}°, "
              f"wrist_r_std={metrics['wrist_r_std']:>5.2f}°, "
              f"gripper_trans={metrics['gripper_transitions']:>6.1f}")

    return val_episodes


def select_validation_episodes_random(
    dataset: LeRobotDataset, num_val_episodes: int, seed: int = 42
) -> list[int]:
    """Select validation episodes randomly."""
    np.random.seed(seed)
    all_episodes = list(range(dataset.num_episodes))
    val_episodes = np.random.choice(all_episodes, num_val_episodes, replace=False).tolist()
    print(f"\nRandomly selected {len(val_episodes)} validation episodes: {val_episodes}")
    return val_episodes


def save_split_metadata(
    output_dir: Path,
    train_episodes: list[int],
    val_episodes: list[int],
    dataset_root: str,
    strategy: str,
):
    """Save train/val split metadata to JSON."""
    metadata = {
        "dataset_root": dataset_root,
        "strategy": strategy,
        "num_train": len(train_episodes),
        "num_val": len(val_episodes),
        "train_episodes": sorted(train_episodes),
        "val_episodes": sorted(val_episodes),
    }

    split_file = output_dir / "train_val_split.json"
    with open(split_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSplit metadata saved to: {split_file}")
    return split_file


def main():
    parser = argparse.ArgumentParser(description="Create validation split for LeRobot dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="lerobot_dataset_v3",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=10,
        help="Number of validation episodes",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["diverse_joints", "random"],
        default="diverse_joints",
        help="Episode selection strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for split metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for random strategy)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LeRobot Dataset Validation Split Creation")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_root}")
    dataset = LeRobotDataset(
        repo_id="roarm_m3_pick",
        root=Path(args.dataset_root),
    )
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")

    # Check if enough episodes
    if args.val_episodes >= dataset.num_episodes:
        raise ValueError(
            f"Not enough episodes: {dataset.num_episodes} total, "
            f"requested {args.val_episodes} for validation"
        )

    # Select validation episodes
    if args.strategy == "diverse_joints":
        val_episodes = select_validation_episodes_diverse(dataset, args.val_episodes)
    else:
        val_episodes = select_validation_episodes_random(dataset, args.val_episodes, args.seed)

    # Compute train episodes (all except validation)
    train_episodes = [i for i in range(dataset.num_episodes) if i not in val_episodes]

    print(f"\nFinal split:")
    print(f"  Train: {len(train_episodes)} episodes")
    print(f"  Val:   {len(val_episodes)} episodes")

    # Count frames
    train_frames = sum(
        dataset.episodes[ep]["to"] - dataset.episodes[ep]["from"]
        for ep in train_episodes
    )
    val_frames = sum(
        dataset.episodes[ep]["to"] - dataset.episodes[ep]["from"]
        for ep in val_episodes
    )
    print(f"  Train frames: {train_frames}")
    print(f"  Val frames:   {val_frames}")

    # Save metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_metadata(output_dir, train_episodes, val_episodes, args.dataset_root, args.strategy)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Modify run_official_train.py to use train_episodes:")
    print("   # Add episode filtering in dataset loading")
    print(f"   train_episodes = {train_episodes[:5]}...{train_episodes[-2:]}")
    print("\n2. Create validation evaluation script:")
    print(f"   python test_inference_official.py --episodes {val_episodes[:3]}...")
    print("\n3. Re-train model on train split only")
    print("\n4. Evaluate on validation split (unseen during training)")
    print("=" * 80)


if __name__ == "__main__":
    main()
