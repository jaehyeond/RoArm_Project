"""
SmolVLA Checkpoint Evaluation Script

Loads multiple checkpoints and evaluates them on dataset samples to:
1. Compute per-joint L2 error (especially elbow and gripper)
2. Analyze z-score distribution of model outputs
3. Compare checkpoints to find optimal stopping point
4. Detect "mean action" problem vs diverse predictions

Usage:
    python train_eval_checkpoints.py
    python train_eval_checkpoints.py --checkpoints 5000 10000 15000 20000
    python train_eval_checkpoints.py --num-samples 50
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, List
import json

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_P", "Wrist_R", "Gripper"]

# Critical joints that need precise control
CRITICAL_JOINTS = {
    "Elbow": 2,    # Needs z=-3.04 for grasping (-64°)
    "Gripper": 5,  # Needs wide range for open/close
}


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load policy and normalization stats from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    # Load normalization stats
    pre_stats = load_file(
        f"{checkpoint_path}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    post_stats = load_file(
        f"{checkpoint_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )

    stats = {
        "action_mean": post_stats["action.mean"].to(device),
        "action_std": post_stats["action.std"].to(device),
        "state_mean": pre_stats["observation.state.mean"].to(device),
        "state_std": pre_stats["observation.state.std"].to(device),
    }

    return policy, stats


def evaluate_checkpoint(
    policy,
    stats: Dict[str, torch.Tensor],
    dataset: LeRobotDataset,
    test_indices: List[int],
    device: torch.device,
) -> Dict:
    """Evaluate policy on test samples and return metrics."""

    # Get tokenizer
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer

    # Pre-tokenize task
    task_text = "Pick up the white box"
    tokenized = tokenizer(
        [task_text],
        max_length=48,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_mask = tokenized["attention_mask"].bool().to(device)

    all_predictions = []
    all_ground_truth = []
    all_raw_outputs = []  # Normalized z-scores

    for idx in test_indices:
        sample = dataset[idx]

        # Build batch
        batch = {}
        for key, val in sample.items():
            if key in ("action", "task", "episode_index", "frame_index",
                       "timestamp", "index", "task_index", "next.done", "next.reward"):
                continue
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device)

        # Normalize state
        if "observation.state" in batch:
            batch["observation.state"] = (
                batch["observation.state"] - stats["state_mean"]
            ) / (stats["state_std"] + 1e-8)

        # Add language
        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_mask

        # Predict
        policy.reset()
        with torch.inference_mode():
            raw_action = policy.select_action(batch)

        # Unnormalize
        action = raw_action * stats["action_std"] + stats["action_mean"]

        action_np = action.cpu().numpy().squeeze()[:6]
        raw_np = raw_action.cpu().numpy().squeeze()[:6]

        all_predictions.append(action_np)
        all_raw_outputs.append(raw_np)

        # Ground truth
        gt_action = sample.get("action", None)
        if gt_action is not None:
            all_ground_truth.append(gt_action.numpy()[:6])

    # Convert to arrays
    predictions = np.array(all_predictions)
    raw_outputs = np.array(all_raw_outputs)
    ground_truth = np.array(all_ground_truth) if all_ground_truth else None

    # Compute metrics
    metrics = {
        "prediction_mean": predictions.mean(axis=0),
        "prediction_std": predictions.std(axis=0),
        "prediction_min": predictions.min(axis=0),
        "prediction_max": predictions.max(axis=0),
        "raw_z_mean": raw_outputs.mean(axis=0),
        "raw_z_std": raw_outputs.std(axis=0),
        "raw_z_min": raw_outputs.min(axis=0),
        "raw_z_max": raw_outputs.max(axis=0),
    }

    if ground_truth is not None:
        # Per-joint L2 error
        per_joint_errors = np.abs(predictions - ground_truth)
        metrics["per_joint_l2_mean"] = per_joint_errors.mean(axis=0)
        metrics["per_joint_l2_std"] = per_joint_errors.std(axis=0)
        metrics["per_joint_l2_max"] = per_joint_errors.max(axis=0)

        # Overall L2 error
        l2_errors = np.linalg.norm(predictions - ground_truth, axis=1)
        metrics["overall_l2_mean"] = l2_errors.mean()
        metrics["overall_l2_std"] = l2_errors.std()
        metrics["overall_l2_min"] = l2_errors.min()
        metrics["overall_l2_max"] = l2_errors.max()

    return metrics


def print_comparison_table(checkpoint_metrics: Dict[str, Dict]):
    """Print comparison table across checkpoints."""

    checkpoints = sorted(checkpoint_metrics.keys())

    print("\n" + "=" * 120)
    print("CHECKPOINT COMPARISON")
    print("=" * 120)

    # Overall L2 error
    print("\n📊 Overall L2 Error:")
    print(f"{'Checkpoint':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 65)
    for ckpt in checkpoints:
        m = checkpoint_metrics[ckpt]
        if "overall_l2_mean" in m:
            print(f"{ckpt:<15} {m['overall_l2_mean']:>11.4f} {m['overall_l2_std']:>11.4f} "
                  f"{m['overall_l2_min']:>11.4f} {m['overall_l2_max']:>11.4f}")

    # Per-joint L2 error (critical joints only)
    print("\n🎯 Critical Joints L2 Error (Mean ± Std):")
    print(f"{'Checkpoint':<15}", end="")
    for joint_name, joint_idx in CRITICAL_JOINTS.items():
        print(f" {joint_name:<20}", end="")
    print()
    print("-" * (15 + 20 * len(CRITICAL_JOINTS)))

    for ckpt in checkpoints:
        m = checkpoint_metrics[ckpt]
        print(f"{ckpt:<15}", end="")
        if "per_joint_l2_mean" in m:
            for joint_name, joint_idx in CRITICAL_JOINTS.items():
                mean_err = m["per_joint_l2_mean"][joint_idx]
                std_err = m["per_joint_l2_std"][joint_idx]
                print(f" {mean_err:>8.2f} ± {std_err:<8.2f}", end="")
        print()

    # Z-score range (check for conservative outputs)
    print("\n📈 Z-Score Range (Normalized Output):")
    print(f"{'Checkpoint':<15} {'Elbow (z)':<20} {'Gripper (z)':<20} {'Overall Range':<20}")
    print("-" * 80)
    for ckpt in checkpoints:
        m = checkpoint_metrics[ckpt]
        elbow_range = f"[{m['raw_z_min'][2]:.2f}, {m['raw_z_max'][2]:.2f}]"
        gripper_range = f"[{m['raw_z_min'][5]:.2f}, {m['raw_z_max'][5]:.2f}]"
        overall_range = f"[{m['raw_z_min'].min():.2f}, {m['raw_z_max'].max():.2f}]"
        print(f"{ckpt:<15} {elbow_range:<20} {gripper_range:<20} {overall_range:<20}")

    # Diversity check (std of predictions)
    print("\n🌈 Prediction Diversity (Std across samples):")
    print(f"{'Checkpoint':<15}", end="")
    for name in JOINT_NAMES:
        print(f" {name:<10}", end="")
    print()
    print("-" * (15 + 10 * len(JOINT_NAMES)))

    for ckpt in checkpoints:
        m = checkpoint_metrics[ckpt]
        print(f"{ckpt:<15}", end="")
        for i in range(6):
            print(f" {m['prediction_std'][i]:>9.2f}", end="")
        print()

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000],
        help="Checkpoint steps to evaluate (default: 5K, 10K, 15K, 20K)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="E:/RoArm_Project/outputs/smolvla_official",
        help="Base output directory containing checkpoints",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="E:/RoArm_Project/lerobot_dataset_v3",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to test per checkpoint",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="train_checkpoint_eval_results.json",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        repo_id="roarm_m3_pick",
        root=Path(args.dataset_root),
    )
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")

    # Select test indices (stratified across dataset)
    test_indices = np.linspace(0, len(dataset) - 1, args.num_samples, dtype=int).tolist()
    print(f"  Test samples: {args.num_samples} (indices: {test_indices[0]}...{test_indices[-1]})")

    # Evaluate each checkpoint
    checkpoint_metrics = {}

    for step in args.checkpoints:
        ckpt_path = Path(args.output_dir) / "checkpoints" / f"{step:06d}" / "pretrained_model"

        if not ckpt_path.exists():
            print(f"\n⚠️ Checkpoint {step} not found at {ckpt_path}, skipping...")
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating Checkpoint: {step} steps")
        print(f"{'=' * 80}")

        try:
            policy, stats = load_checkpoint(str(ckpt_path), device)
            metrics = evaluate_checkpoint(policy, stats, dataset, test_indices, device)
            checkpoint_metrics[f"{step}K"] = metrics

            # Quick summary for this checkpoint
            if "overall_l2_mean" in metrics:
                print(f"✅ Overall L2: {metrics['overall_l2_mean']:.4f} ± {metrics['overall_l2_std']:.4f}")
                print(f"   Elbow L2: {metrics['per_joint_l2_mean'][2]:.2f}°")
                print(f"   Gripper L2: {metrics['per_joint_l2_mean'][5]:.2f}°")
            print(f"   Z-score range: [{metrics['raw_z_min'].min():.2f}, {metrics['raw_z_max'].max():.2f}]")
            print(f"   Pred diversity (mean std): {metrics['prediction_std'].mean():.2f}°")

            # Clear GPU memory
            del policy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error evaluating checkpoint {step}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if checkpoint_metrics:
        print_comparison_table(checkpoint_metrics)

        # Save to JSON
        save_path = Path(args.save_json)
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for ckpt, metrics in checkpoint_metrics.items():
            serializable_metrics[ckpt] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }

        with open(save_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"\n💾 Results saved to: {save_path.absolute()}")

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Find best checkpoint by overall L2
        best_ckpt = min(
            checkpoint_metrics.keys(),
            key=lambda k: checkpoint_metrics[k].get("overall_l2_mean", float("inf"))
        )
        print(f"✨ Best checkpoint (lowest L2): {best_ckpt}")

        # Check if any checkpoint has good z-score range
        needs_more_training = True
        for ckpt, m in checkpoint_metrics.items():
            z_range = m["raw_z_max"].max() - m["raw_z_min"].min()
            if z_range > 5.0:  # Need at least ±2.5 range
                needs_more_training = False
                print(f"✅ {ckpt} has good z-score range: {z_range:.2f}")

        if needs_more_training:
            print("\n⚠️ All checkpoints show conservative z-score range (< ±2.5)")
            print("   Recommendation: Continue training to 50K-100K steps")
            print("   Target: z-score range > ±3.0 for elbow control")

        print("=" * 80)

    else:
        print("\n❌ No checkpoints evaluated successfully.")


if __name__ == "__main__":
    main()
