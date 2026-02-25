"""
SmolVLA v3 Checkpoint Evaluation Script (50K steps, 74 episodes, sponge task)

Evaluates ALL 10 checkpoints (5K–50K) to find the optimal deployment checkpoint.
Key question: which checkpoint has the best balance of low L2 AND high diversity?

Metrics:
1. Per-joint L2 error (degrees) — lower is better, but watch for overfitting
2. Per-joint prediction diversity (std) — CRITICAL: low = mean regression = deployment fail
3. Z-score range — conservative (<±1.5) vs. expressive (>±2.5)
4. Gripper diversity — did v3 fix the "gripper never opens" problem from v1?

Usage:
    /home/cgxr/miniconda3/envs/roarm/bin/python train_eval_v3_checkpoints.py
    /home/cgxr/miniconda3/envs/roarm/bin/python train_eval_v3_checkpoints.py --num-samples 300
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
import time

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ── Joint metadata ──────────────────────────────────────────────────────────
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_P", "Wrist_R", "Gripper"]

# v3 dataset reference statistics (74 episodes, 13,145 frames)
# From: lerobot_dataset_v3/meta/stats.json
V3_DATASET_MEAN = np.array([-0.471, 30.177, 58.876, 40.721, -2.328, 26.479])
V3_DATASET_STD  = np.array([25.812, 18.807, 24.829, 30.069, 20.216, 24.153])

# v1 reference for comparison (50 episodes)
V1_DATASET_STD  = np.array([21.75, 26.08, 29.03, 26.00, 22.14, 13.65])

# Diversity threshold: fraction of dataset std that counts as "healthy"
DIVERSITY_WARN_THRESHOLD = 0.50   # < 50% of dataset std → conservative
DIVERSITY_FAIL_THRESHOLD = 0.25   # < 25%  → mean action problem


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load policy and normalization stats from a pretrained_model directory."""
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()

    pre_stats = load_file(
        str(checkpoint_path / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    )
    post_stats = load_file(
        str(checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    )

    stats = {
        "action_mean": post_stats["action.mean"].to(device),
        "action_std":  post_stats["action.std"].to(device),
        "state_mean":  pre_stats["observation.state.mean"].to(device),
        "state_std":   pre_stats["observation.state.std"].to(device),
    }
    return policy, stats


def build_lang_tokens(policy, task_text: str, device: torch.device):
    """Tokenize task text once and return reusable tensors."""
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer
    tokenized = tokenizer(
        [task_text],
        max_length=48,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return (
        tokenized["input_ids"].to(device),
        tokenized["attention_mask"].bool().to(device),
    )


def stratified_sample_indices(dataset: LeRobotDataset, num_samples: int) -> List[int]:
    """
    Return frame indices stratified across episodes.
    For each episode, pick frames uniformly within it.
    This ensures coverage of start/mid/end of episodes across all 74 episodes.
    """
    total_frames = len(dataset)
    num_episodes = dataset.num_episodes

    # Build episode frame ranges using hf_dataset episode_index column
    ep_idx_col = np.array(dataset.hf_dataset["episode_index"])
    ep_frame_ranges = []
    for ep_idx in range(num_episodes):
        frame_positions = np.where(ep_idx_col == ep_idx)[0]
        if len(frame_positions) == 0:
            continue
        from_idx = int(frame_positions[0])
        to_idx   = int(frame_positions[-1]) + 1   # exclusive
        ep_frame_ranges.append((from_idx, to_idx))

    # Distribute samples across episodes proportionally
    samples_per_ep = max(1, num_samples // num_episodes)
    indices = []
    for (from_idx, to_idx) in ep_frame_ranges:
        ep_len = to_idx - from_idx
        if ep_len <= 0:
            continue
        count = min(samples_per_ep, ep_len)
        chosen = np.linspace(from_idx, to_idx - 1, count, dtype=int)
        indices.extend(chosen.tolist())

    # If we under-sampled, add more from uniformly across full dataset
    if len(indices) < num_samples:
        extra_needed = num_samples - len(indices)
        extra = np.linspace(0, total_frames - 1, extra_needed + 2, dtype=int)[1:-1]
        indices.extend(extra.tolist())

    # Deduplicate and sort
    indices = sorted(set(indices))

    # Trim to num_samples (take every k-th if over)
    if len(indices) > num_samples:
        step = len(indices) // num_samples
        indices = indices[::step][:num_samples]

    return indices


def evaluate_checkpoint(
    policy,
    stats: Dict,
    dataset: LeRobotDataset,
    test_indices: List[int],
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Run inference on test_indices; return comprehensive metrics."""

    all_predictions = []
    all_ground_truth = []
    all_raw_z = []
    inference_times = []

    for idx in test_indices:
        sample = dataset[idx]

        # Build observation batch (exclude labels/metadata)
        batch = {}
        SKIP_KEYS = {"action", "task", "episode_index", "frame_index",
                     "timestamp", "index", "task_index", "next.done", "next.reward"}
        for key, val in sample.items():
            if key in SKIP_KEYS:
                continue
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device)

        # Normalize observation state
        if "observation.state" in batch:
            batch["observation.state"] = (
                batch["observation.state"] - stats["state_mean"]
            ) / (stats["state_std"] + 1e-8)

        # Inject language conditioning
        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_mask

        # Inference
        policy.reset()
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw_action = policy.select_action(batch)
        inference_times.append(time.perf_counter() - t0)

        # Unnormalize to degrees
        action = raw_action * stats["action_std"] + stats["action_mean"]

        pred_np = action.cpu().numpy().squeeze()[:6]
        raw_np  = raw_action.cpu().numpy().squeeze()[:6]

        all_predictions.append(pred_np)
        all_raw_z.append(raw_np)

        gt = sample.get("action", None)
        if gt is not None:
            all_ground_truth.append(gt.numpy()[:6])

    predictions = np.array(all_predictions)   # (N, 6)
    raw_z       = np.array(all_raw_z)         # (N, 6)
    ground_truth = np.array(all_ground_truth) if all_ground_truth else None

    # ── Diversity metrics ───────────────────────────────────────────────────
    metrics = {
        "n_samples":         len(predictions),
        "inference_ms_mean": np.mean(inference_times) * 1000,

        "pred_mean":  predictions.mean(axis=0),
        "pred_std":   predictions.std(axis=0),
        "pred_min":   predictions.min(axis=0),
        "pred_max":   predictions.max(axis=0),
        "pred_range": predictions.max(axis=0) - predictions.min(axis=0),

        "raw_z_mean": raw_z.mean(axis=0),
        "raw_z_std":  raw_z.std(axis=0),
        "raw_z_min":  raw_z.min(axis=0),
        "raw_z_max":  raw_z.max(axis=0),
        "raw_z_range":raw_z.max(axis=0) - raw_z.min(axis=0),

        # Diversity ratio: pred_std / dataset_std (> 0.5 is healthy)
        "diversity_ratio": predictions.std(axis=0) / (V3_DATASET_STD + 1e-8),
        "mean_diversity_ratio": float((predictions.std(axis=0) / (V3_DATASET_STD + 1e-8)).mean()),
    }

    # ── Accuracy metrics (if ground truth available) ────────────────────────
    if ground_truth is not None:
        per_joint_err = np.abs(predictions - ground_truth)
        metrics["per_joint_l2_mean"]    = per_joint_err.mean(axis=0)
        metrics["per_joint_l2_std"]     = per_joint_err.std(axis=0)
        metrics["per_joint_l2_max"]     = per_joint_err.max(axis=0)

        l2_errors = np.linalg.norm(predictions - ground_truth, axis=1)
        metrics["overall_l2_mean"] = float(l2_errors.mean())
        metrics["overall_l2_std"]  = float(l2_errors.std())
        metrics["overall_l2_min"]  = float(l2_errors.min())
        metrics["overall_l2_max"]  = float(l2_errors.max())

    return metrics


# ── Printing helpers ─────────────────────────────────────────────────────────

def flag(val, warn, fail):
    """Return a text flag for a ratio value."""
    if val < fail:
        return " [FAIL]"
    if val < warn:
        return " [WARN]"
    return " [OK]  "


def print_full_report(checkpoint_metrics: Dict[str, Dict]):
    steps_sorted = sorted(checkpoint_metrics.keys())

    W = 132
    print("\n" + "=" * W)
    print("V3 CHECKPOINT EVALUATION REPORT  (batch_size=64, 74 episodes, 13,145 frames)")
    print("=" * W)

    # ── Table 1: Overall L2 Error ────────────────────────────────────────────
    print("\n  TABLE 1 — Overall L2 Error (degrees)  [lower is better, but plateau = overfitting]")
    print(f"  {'Checkpoint':<12} {'N_samples':>9} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  {'Inference_ms':>13}")
    print("  " + "-" * 70)
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        if "overall_l2_mean" in m:
            print(f"  {ckpt:<12} {m['n_samples']:>9d} "
                  f"{m['overall_l2_mean']:>8.3f} {m['overall_l2_std']:>8.3f} "
                  f"{m['overall_l2_min']:>8.3f} {m['overall_l2_max']:>8.3f}  "
                  f"{m['inference_ms_mean']:>12.1f}")

    # ── Table 2: Per-joint L2 Error ──────────────────────────────────────────
    print("\n  TABLE 2 — Per-joint L2 Error (degrees)  [critical: Elbow, Gripper]")
    header = f"  {'Checkpoint':<12}"
    for name in JOINT_NAMES:
        header += f" {name:>11}"
    print(header)
    print("  " + "-" * (12 + 12 * 6))
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        row = f"  {ckpt:<12}"
        if "per_joint_l2_mean" in m:
            for i in range(6):
                row += f" {m['per_joint_l2_mean'][i]:>11.2f}"
        print(row)

    # ── Table 3: Prediction Diversity ────────────────────────────────────────
    print("\n  TABLE 3 — Prediction Diversity (Std, degrees)  [CRITICAL: must be >50% of dataset std]")
    print(f"  Dataset std:   ", end="")
    for v in V3_DATASET_STD:
        print(f" {v:>11.2f}", end="")
    print()
    print(f"  50% threshold: ", end="")
    for v in V3_DATASET_STD * 0.5:
        print(f" {v:>11.2f}", end="")
    print()
    print("  " + "-" * (12 + 12 * 6 + 20))
    header = f"  {'Checkpoint':<12}"
    for name in JOINT_NAMES:
        header += f" {name:>11}"
    header += f"  {'AvgRatio':>9}  {'Status':>10}"
    print(header)
    print("  " + "-" * (12 + 12 * 6 + 25))
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        row = f"  {ckpt:<12}"
        for i in range(6):
            row += f" {m['pred_std'][i]:>11.2f}"
        ratio = m["mean_diversity_ratio"]
        status = flag(ratio, DIVERSITY_WARN_THRESHOLD, DIVERSITY_FAIL_THRESHOLD)
        row += f"  {ratio:>9.3f}{status}"
        print(row)

    # ── Table 4: Diversity Ratio (pred_std / dataset_std) ───────────────────
    print("\n  TABLE 4 — Diversity Ratio per joint (pred_std / dataset_std)  [>0.5 = healthy]")
    header = f"  {'Checkpoint':<12}"
    for name in JOINT_NAMES:
        header += f" {name:>11}"
    print(header)
    print("  " + "-" * (12 + 12 * 6))
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        row = f"  {ckpt:<12}"
        for i in range(6):
            ratio = m["diversity_ratio"][i]
            tag = "" if ratio >= 0.5 else ("*" if ratio >= 0.25 else "**")
            row += f" {ratio:>9.3f}{tag:>2}"
        print(row)
    print("  (* = below 50% threshold, ** = below 25% = mean action failure)")

    # ── Table 5: Z-score Range ───────────────────────────────────────────────
    print("\n  TABLE 5 — Z-score Range (normalized space)  [range>3.0 = expressive, <1.5 = conservative]")
    print(f"  {'Checkpoint':<12}", end="")
    for name in JOINT_NAMES:
        print(f" {name:>11}", end="")
    print(f"  {'GlobalRange':>12}")
    print("  " + "-" * (12 + 12 * 7))
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        row = f"  {ckpt:<12}"
        for i in range(6):
            r = m["raw_z_range"][i]
            row += f" {r:>11.3f}"
        global_range = m["raw_z_range"].max()
        row += f"  {global_range:>12.3f}"
        print(row)

    # ── Table 6: Gripper Focus ───────────────────────────────────────────────
    print("\n  TABLE 6 — Gripper Deep-dive (index 5)  [v1 failed: never opened]")
    print(f"  {'Checkpoint':<12} {'PredMean':>10} {'PredStd':>9} {'PredMin':>9} {'PredMax':>9} "
          f"{'PredRange':>10} {'ZRange':>7} {'DivRatio':>9}")
    print("  " + "-" * 80)
    for ckpt in steps_sorted:
        m = checkpoint_metrics[ckpt]
        i = 5  # Gripper index
        pred_range = m["pred_range"][i]
        z_range    = m["raw_z_range"][i]
        div_ratio  = m["diversity_ratio"][i]
        tag = flag(div_ratio, DIVERSITY_WARN_THRESHOLD, DIVERSITY_FAIL_THRESHOLD)
        print(f"  {ckpt:<12} {m['pred_mean'][i]:>10.2f} {m['pred_std'][i]:>9.2f} "
              f"{m['pred_min'][i]:>9.2f} {m['pred_max'][i]:>9.2f} "
              f"{pred_range:>10.2f} {z_range:>7.3f} {div_ratio:>9.3f}{tag}")
    print(f"  Dataset:       mean={V3_DATASET_MEAN[5]:.2f}  std={V3_DATASET_STD[5]:.2f}")
    print(f"  v1 std={V1_DATASET_STD[5]:.2f} (old reference)")

    # ── Recommendation ───────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  RECOMMENDATION")
    print("=" * W)

    # Score each checkpoint: low L2 + high diversity ratio
    scored = {}
    for ckpt, m in checkpoint_metrics.items():
        if "overall_l2_mean" not in m:
            continue
        l2 = m["overall_l2_mean"]
        div = m["mean_diversity_ratio"]
        # Normalize L2 (lower is better → higher score)
        all_l2 = [checkpoint_metrics[k]["overall_l2_mean"]
                  for k in checkpoint_metrics if "overall_l2_mean" in checkpoint_metrics[k]]
        l2_score = 1.0 - (l2 - min(all_l2)) / (max(all_l2) - min(all_l2) + 1e-9)
        # Combined: 40% L2 score + 60% diversity ratio
        combined = 0.40 * l2_score + 0.60 * min(div, 1.0)
        scored[ckpt] = {"combined": combined, "l2": l2, "div": div}

    if scored:
        best_l2 = min(scored, key=lambda k: scored[k]["l2"])
        best_div = max(scored, key=lambda k: scored[k]["div"])
        best_combined = max(scored, key=lambda k: scored[k]["combined"])

        print(f"\n  Best by L2 error alone:          {best_l2}  "
              f"(L2={scored[best_l2]['l2']:.3f}, div={scored[best_l2]['div']:.3f})")
        print(f"  Best by diversity alone:         {best_div}  "
              f"(L2={scored[best_div]['l2']:.3f}, div={scored[best_div]['div']:.3f})")
        print(f"  Best combined (40% L2 + 60% div): {best_combined}  "
              f"(L2={scored[best_combined]['l2']:.3f}, div={scored[best_combined]['div']:.3f})")
        print()

        # Overfitting warning: check if L2 goes down but diversity also drops
        steps_with_l2 = [(k, scored[k]["l2"], scored[k]["div"]) for k in steps_sorted if k in scored]
        if len(steps_with_l2) >= 3:
            # Check if late checkpoints have lower diversity than early ones
            early_div = np.mean([d for _, _, d in steps_with_l2[:3]])
            late_div  = np.mean([d for _, _, d in steps_with_l2[-3:]])
            if late_div < early_div * 0.9:
                print(f"  OVERFITTING SIGNAL: diversity drops from early ({early_div:.3f}) "
                      f"to late ({late_div:.3f}) checkpoints.")
                print(f"  Recommendation: use {best_combined} not the final checkpoint.")
            else:
                print(f"  No clear overfitting signal in diversity trend.")

        print()
        print(f"  DEPLOY RECOMMENDATION: {best_combined}")
        m = checkpoint_metrics[best_combined]
        gripper_range = m["pred_range"][5]
        gripper_div   = m["diversity_ratio"][5]
        elbow_div     = m["diversity_ratio"][2]
        print(f"    - Overall L2: {m['overall_l2_mean']:.3f} degrees")
        print(f"    - Mean diversity ratio: {m['mean_diversity_ratio']:.3f}")
        print(f"    - Gripper pred range: {gripper_range:.1f} degrees "
              f"({'GOOD' if gripper_range > 30 else 'LOW - gripper may not open!'})")
        print(f"    - Gripper diversity ratio: {gripper_div:.3f} "
              f"({'OK' if gripper_div >= 0.5 else 'WARN' if gripper_div >= 0.25 else 'FAIL'})")
        print(f"    - Elbow diversity ratio: {elbow_div:.3f} "
              f"({'OK' if elbow_div >= 0.5 else 'WARN' if elbow_div >= 0.25 else 'FAIL'})")
        ckpt_path = f"outputs/smolvla_v3_sponge/checkpoints/"
        step_num = best_combined.replace("K", "000")
        print(f"    - Checkpoint path: {ckpt_path}{int(step_num):06d}/pretrained_model")

    print("=" * W)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA v3 checkpoints (5K-50K)")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
        help="Checkpoint steps (default: all 10, 5K-50K)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/smolvla_v3_sponge",
        help="Base output directory",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="lerobot_dataset_v3",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=222,
        help="Samples per checkpoint (default: 222 = 3 per episode across 74 episodes)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="train_v3_checkpoint_eval_results.json",
        help="JSON output file for results",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default="Pick up the sponge",
        help="Task text (must match training)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SmolVLA v3 Checkpoint Evaluation")
    print("=" * 80)
    print(f"Device:      {device}")
    print(f"Task text:   '{args.task_text}'")
    print(f"Num samples: {args.num_samples}")
    print(f"Checkpoints: {args.checkpoints}")

    # Load dataset once (shared across checkpoints)
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        repo_id="roarm_m3_pick",
        root=Path(args.dataset_root),
    )
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes:     {dataset.num_episodes}")

    # Stratified sample indices (fixed across all checkpoint evaluations)
    test_indices = stratified_sample_indices(dataset, args.num_samples)
    print(f"  Test frames:  {len(test_indices)} "
          f"(indices {test_indices[0]}..{test_indices[-1]})")

    # Check task text matches dataset
    sample0 = dataset[0]
    actual_task = sample0.get("task", "UNKNOWN")
    if actual_task != args.task_text:
        print(f"\n  WARNING: dataset task='{actual_task}' but --task-text='{args.task_text}'")
        print(f"  Using dataset task text: '{actual_task}'")
        args.task_text = actual_task

    # Evaluate each checkpoint
    checkpoint_metrics = {}

    for step in args.checkpoints:
        ckpt_dir = f"{step:06d}"
        ckpt_path = Path(args.output_dir) / "checkpoints" / ckpt_dir / "pretrained_model"

        if not ckpt_path.exists():
            print(f"\n  [SKIP] Checkpoint {step} not found: {ckpt_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"  Checkpoint {step:,} steps  ({ckpt_path})")
        print(f"{'=' * 80}")

        t_start = time.perf_counter()
        try:
            policy, stats = load_checkpoint(ckpt_path, device)

            # Build language tokens (use same policy/tokenizer)
            lang_tokens, lang_mask = build_lang_tokens(policy, args.task_text, device)

            metrics = evaluate_checkpoint(
                policy, stats, dataset, test_indices,
                lang_tokens, lang_mask, device
            )
            t_eval = time.perf_counter() - t_start

            # Label key: "5K", "10K", etc.
            label = f"{step // 1000}K"
            checkpoint_metrics[label] = metrics

            # Quick per-checkpoint summary
            l2 = metrics.get("overall_l2_mean", float("nan"))
            div = metrics.get("mean_diversity_ratio", float("nan"))
            gripper_range = metrics["pred_range"][5]
            elbow_range   = metrics["pred_range"][2]
            z_global_max  = metrics["raw_z_range"].max()

            print(f"  Overall L2:          {l2:.4f} degrees")
            print(f"  Mean diversity ratio:{div:.4f}  "
                  f"({'HEALTHY' if div >= 0.5 else 'WARN: conservative' if div >= 0.25 else 'FAIL: mean action'})")
            print(f"  Gripper pred range:  {gripper_range:.1f} deg  "
                  f"{'(GOOD)' if gripper_range > 30 else '(LOW)'}")
            print(f"  Elbow pred range:    {elbow_range:.1f} deg")
            print(f"  Global z-score range:{z_global_max:.3f}  "
                  f"{'(expressive)' if z_global_max > 3.0 else '(moderate)' if z_global_max > 1.5 else '(conservative)'}")
            print(f"  Eval time:           {t_eval:.1f}s  ({metrics['inference_ms_mean']:.1f}ms/inference)")

            # Free GPU memory before next checkpoint
            del policy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Print full comparison report
    if checkpoint_metrics:
        print_full_report(checkpoint_metrics)

        # Serialize and save results
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        serializable = {}
        for ckpt, m in checkpoint_metrics.items():
            serializable[ckpt] = {k: to_serializable(v) for k, v in m.items()}

        save_path = Path(args.save_json)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved: {save_path.absolute()}")
    else:
        print("\n  No checkpoints evaluated successfully.")


if __name__ == "__main__":
    main()
