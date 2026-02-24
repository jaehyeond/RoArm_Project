"""
Open-Loop Action Chunk Analysis for SmolVLA Deployment

Analyzes model behavior by generating 50-step action chunks from single observations
at different scenarios (DEEP, APPROACH, SHALLOW grasp episodes).

Key Analysis:
1. Temporal patterns: Does the model show purposeful joint trajectories?
2. Gripper behavior: Does the gripper open/close within the chunk?
3. Convergence: Do actions converge in later steps?
4. Scenario sensitivity: Different outputs for different starting conditions?

Usage:
    python deploy_openloop_analysis.py --checkpoint outputs/smolvla_official/checkpoints/035000/pretrained_model
    python deploy_openloop_analysis.py --checkpoint outputs/smolvla_official/checkpoints/last/pretrained_model
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Joint indices
BASE, SHOULDER, ELBOW, WRIST_PITCH, WRIST_ROLL, GRIPPER = 0, 1, 2, 3, 4, 5
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_P", "Wrist_R", "Gripper"]


def load_model(checkpoint_path, device):
    """Load model and normalization statistics."""
    print("=" * 60)
    print("Loading SmolVLA Model")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    # Load policy
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total params: {total_params:,}")

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

    print(f"Action mean: {stats['action_mean'].cpu().numpy()}")
    print(f"Action std:  {stats['action_std'].cpu().numpy()}")

    # Get tokenizer
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer

    print("Model loaded successfully!")
    return policy, tokenizer, stats


def tokenize_task(tokenizer, task_text, device):
    """Tokenize task description."""
    tokenized = tokenizer(
        [task_text],
        max_length=48,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "tokens": tokenized["input_ids"].to(device),
        "mask": tokenized["attention_mask"].bool().to(device),
    }


def build_observation_from_sample(sample, lang, stats, device):
    """Build model observation from dataset sample."""
    batch = {}

    # Copy observation tensors
    for key, val in sample.items():
        if key in ("action", "task", "episode_index", "frame_index",
                   "timestamp", "index", "task_index", "next.done", "next.reward"):
            continue
        if isinstance(val, torch.Tensor):
            batch[key] = val.unsqueeze(0).to(device)

    # Normalize state (MEAN_STD)
    if "observation.state" in batch:
        batch["observation.state"] = (batch["observation.state"] - stats["state_mean"]) / (stats["state_std"] + 1e-8)

    # Add language tokens
    batch["observation.language.tokens"] = lang["tokens"]
    batch["observation.language.attention_mask"] = lang["mask"]

    return batch


def unnormalize_action(raw_action, stats):
    """Convert normalized action (z-scores) to real joint angles."""
    return raw_action * stats["action_std"] + stats["action_mean"]


def generate_action_chunk(policy, obs, stats, chunk_size=50):
    """Generate full action chunk from single observation (open-loop).

    Returns:
        actions: (chunk_size, 6) unnormalized actions
        z_scores: (chunk_size, 6) z-scores (before unnormalization)
    """
    policy.reset()

    with torch.inference_mode():
        # Get raw action chunk
        batch = policy._prepare_batch(obs)
        raw_chunk = policy._get_action_chunk(batch, noise=None)
        # raw_chunk shape: (1, chunk_size, action_dim)

        # Unnormalize
        chunk = unnormalize_action(raw_chunk, stats)

        # Extract first 6 joints (gripper + 5 others)
        actions = chunk.cpu().numpy().squeeze()[:chunk_size, :6]
        z_scores = raw_chunk.cpu().numpy().squeeze()[:chunk_size, :6]

    return actions, z_scores


def analyze_temporal_pattern(actions, joint_idx, joint_name):
    """Analyze temporal pattern in action trajectory."""
    trajectory = actions[:, joint_idx]

    # Compute deltas (frame-to-frame changes)
    deltas = np.diff(trajectory)

    # Metrics
    total_range = trajectory.max() - trajectory.min()
    mean_delta = np.mean(np.abs(deltas))
    monotonic_ratio = np.sum(deltas > 0) / len(deltas) if len(deltas) > 0 else 0

    # Check for convergence (last 10 steps)
    if len(trajectory) >= 10:
        late_std = np.std(trajectory[-10:])
        early_std = np.std(trajectory[:10])
        convergence = late_std < early_std and late_std < 1.0
    else:
        convergence = False

    return {
        "joint": joint_name,
        "range": total_range,
        "mean_delta": mean_delta,
        "monotonic_ratio": monotonic_ratio,
        "convergence": convergence,
        "start": trajectory[0],
        "end": trajectory[-1],
        "total_change": trajectory[-1] - trajectory[0],
    }


def analyze_gripper_behavior(actions):
    """Analyze gripper opening/closing behavior."""
    gripper_traj = actions[:, GRIPPER]

    # Find open/close events
    open_threshold = 30  # gripper > 30 = open
    close_threshold = 10  # gripper < 10 = closed

    open_frames = np.where(gripper_traj > open_threshold)[0]
    close_frames = np.where(gripper_traj < close_threshold)[0]

    # Check for open->close pattern
    has_open = len(open_frames) > 0
    has_close = len(close_frames) > 0

    if has_open and has_close:
        first_open = open_frames[0]
        first_close = close_frames[close_frames > first_open]
        has_grasp_pattern = len(first_close) > 0
    else:
        has_grasp_pattern = False

    return {
        "max_opening": gripper_traj.max(),
        "min_opening": gripper_traj.min(),
        "mean_opening": gripper_traj.mean(),
        "range": gripper_traj.max() - gripper_traj.min(),
        "has_open": has_open,
        "has_close": has_close,
        "has_grasp_pattern": has_grasp_pattern,
        "start": gripper_traj[0],
        "end": gripper_traj[-1],
    }


def find_scenario_samples(dataset):
    """Find representative samples for DEEP, APPROACH, SHALLOW scenarios.

    Returns dict with keys: 'DEEP', 'APPROACH', 'SHALLOW'
    Each value is (episode_idx, frame_idx, sample_idx, elbow_angle)
    """
    print("\nSearching for scenario samples...")

    # Load parquet to analyze all frames
    parquet_path = Path("lerobot_dataset_v3/data/chunk-000/file-000.parquet")
    df = pd.read_parquet(parquet_path)

    # Extract elbow angles from action column (assuming it's shape (N, 6))
    if 'action' in df.columns:
        actions = np.stack(df['action'].values)
    else:
        action_cols = [col for col in df.columns if col.startswith('action')]
        actions = df[action_cols].values

    elbow_angles = actions[:, ELBOW]
    episode_indices = df['episode_index'].values

    # Find samples
    scenarios = {}

    # DEEP: elbow < -30
    deep_mask = elbow_angles < -30
    if deep_mask.sum() > 0:
        # Find middle frame in first DEEP episode
        deep_episodes = np.unique(episode_indices[deep_mask])
        first_deep_ep = deep_episodes[0]
        deep_frames = np.where((episode_indices == first_deep_ep) & deep_mask)[0]
        mid_idx = deep_frames[len(deep_frames) // 2]
        scenarios['DEEP'] = (first_deep_ep, mid_idx, elbow_angles[mid_idx])

    # APPROACH: -30 < elbow < -10
    approach_mask = (elbow_angles > -30) & (elbow_angles < -10)
    if approach_mask.sum() > 0:
        approach_episodes = np.unique(episode_indices[approach_mask])
        first_approach_ep = approach_episodes[0]
        approach_frames = np.where((episode_indices == first_approach_ep) & approach_mask)[0]
        mid_idx = approach_frames[len(approach_frames) // 2]
        scenarios['APPROACH'] = (first_approach_ep, mid_idx, elbow_angles[mid_idx])

    # SHALLOW: elbow > -10
    shallow_mask = elbow_angles > -10
    if shallow_mask.sum() > 0:
        shallow_episodes = np.unique(episode_indices[shallow_mask])
        first_shallow_ep = shallow_episodes[len(shallow_episodes) // 2]  # Pick middle episode
        shallow_frames = np.where((episode_indices == first_shallow_ep) & shallow_mask)[0]
        mid_idx = shallow_frames[len(shallow_frames) // 2]
        scenarios['SHALLOW'] = (first_shallow_ep, mid_idx, elbow_angles[mid_idx])

    return scenarios


def plot_action_chunk(actions, z_scores, scenario_name, state_at_t0, output_dir):
    """Plot action chunk trajectory."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{scenario_name} - Action Chunk Trajectory\nInitial State: {state_at_t0}',
                 fontsize=14, fontweight='bold')

    steps = np.arange(len(actions))

    for i, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        # Plot action trajectory
        ax.plot(steps, actions[:, i], 'b-', linewidth=2, label='Action')
        ax.axhline(actions[0, i], color='g', linestyle='--', alpha=0.5, label='Start')
        ax.axhline(actions[-1, i], color='r', linestyle='--', alpha=0.5, label='End')

        ax.set_title(f'{name} (Δ={actions[-1, i] - actions[0, i]:.1f}°)', fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (deg)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / f'{scenario_name.lower()}_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_z_scores(z_scores, scenario_name, output_dir):
    """Plot z-scores over time."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'{scenario_name} - Z-Scores (Normalized Actions)',
                 fontsize=14, fontweight='bold')

    steps = np.arange(len(z_scores))

    for i, (ax, name) in enumerate(zip(axes.flat, JOINT_NAMES)):
        ax.plot(steps, z_scores[:, i], 'r-', linewidth=2)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(1, color='gray', linestyle='--', alpha=0.3, label='±1σ')
        ax.axhline(-1, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(2, color='orange', linestyle='--', alpha=0.3, label='±2σ')
        ax.axhline(-2, color='orange', linestyle='--', alpha=0.3)

        ax.set_title(f'{name} (range: {z_scores[:, i].min():.2f} to {z_scores[:, i].max():.2f})',
                     fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Z-Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / f'{scenario_name.lower()}_zscores.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Open-loop action chunk analysis")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/smolvla_official/checkpoints/last/pretrained_model",
                        help="Path to SmolVLA checkpoint")
    parser.add_argument("--dataset-root", type=str,
                        default="lerobot_dataset_v3",
                        help="Path to LeRobot dataset")
    parser.add_argument("--task", type=str,
                        default="Pick up the sponge",
                        help="Task description")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Number of action steps to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="analysis_openloop",
                        help="Output directory for plots and logs")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    policy, tokenizer, stats = load_model(args.checkpoint, device)
    lang = tokenize_task(tokenizer, args.task, device)

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    dataset = LeRobotDataset(
        repo_id="roarm_m3_pick",
        root=Path(args.dataset_root),
    )
    print(f"Total frames: {len(dataset)}")
    print(f"Episodes: {dataset.num_episodes}")

    # Find scenario samples
    scenarios = find_scenario_samples(dataset)

    print("\nScenario samples found:")
    for scenario, info in scenarios.items():
        ep_idx, sample_idx, elbow = info
        print(f"  {scenario:10s}: Episode {ep_idx:2d}, Sample {sample_idx:5d}, Elbow={elbow:6.1f}°")

    # Analyze each scenario
    print("\n" + "=" * 60)
    print("Generating and Analyzing Action Chunks")
    print("=" * 60)

    all_results = []

    for scenario_name, (ep_idx, sample_idx, elbow) in scenarios.items():
        print(f"\n{scenario_name} Scenario")
        print("-" * 60)

        # Get sample
        sample = dataset[sample_idx]

        # Get initial state
        state_t0 = sample['observation.state'].numpy()
        print(f"Initial state: {state_t0}")

        # Build observation
        obs = build_observation_from_sample(sample, lang, stats, device)

        # Generate action chunk
        print(f"Generating {args.chunk_size}-step action chunk...")
        actions, z_scores = generate_action_chunk(policy, obs, stats, chunk_size=args.chunk_size)

        print(f"Chunk shape: {actions.shape}")
        print(f"Action range: [{actions.min():.1f}, {actions.max():.1f}]")
        print(f"Z-score range: [{z_scores.min():.2f}, {z_scores.max():.2f}]")

        # Temporal pattern analysis
        print("\nTemporal Pattern Analysis:")
        print(f"{'Joint':<12} {'Range':>8} {'Δ Mean':>8} {'Start':>8} {'End':>8} {'Total Δ':>8} {'Conv':>6}")
        print("-" * 65)

        temporal_results = []
        for i, name in enumerate(JOINT_NAMES):
            result = analyze_temporal_pattern(actions, i, name)
            temporal_results.append(result)
            print(f"{result['joint']:<12} {result['range']:>8.1f} {result['mean_delta']:>8.2f} "
                  f"{result['start']:>8.1f} {result['end']:>8.1f} {result['total_change']:>8.1f} "
                  f"{'YES' if result['convergence'] else 'NO':>6}")

        # Gripper analysis
        print("\nGripper Behavior Analysis:")
        gripper_result = analyze_gripper_behavior(actions)
        print(f"  Max opening: {gripper_result['max_opening']:.1f}°")
        print(f"  Min opening: {gripper_result['min_opening']:.1f}°")
        print(f"  Range: {gripper_result['range']:.1f}°")
        print(f"  Has open (>30°): {gripper_result['has_open']}")
        print(f"  Has close (<10°): {gripper_result['has_close']}")
        print(f"  Has grasp pattern (open→close): {gripper_result['has_grasp_pattern']}")

        # Plot
        print("\nGenerating plots...")
        plot_action_chunk(actions, z_scores, scenario_name, state_t0, output_dir)
        plot_z_scores(z_scores, scenario_name, output_dir)

        # Save CSV
        csv_path = output_dir / f'{scenario_name.lower()}_chunk.csv'
        chunk_df = pd.DataFrame(actions, columns=JOINT_NAMES)
        chunk_df['step'] = np.arange(len(actions))
        chunk_df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

        # Store results
        result_summary = {
            'scenario': scenario_name,
            'episode': ep_idx,
            'sample_idx': sample_idx,
            'initial_elbow': elbow,
            **{f'{k}_{v}': temporal_results[k][v]
               for k in range(6)
               for v in ['range', 'total_change', 'convergence']},
            **{f'gripper_{k}': v for k, v in gripper_result.items()},
        }
        all_results.append(result_summary)

    # Summary comparison
    print("\n" + "=" * 60)
    print("Cross-Scenario Comparison")
    print("=" * 60)

    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / 'scenario_comparison.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Key findings
    print("\nKey Findings:")
    print("-" * 60)

    for i, name in enumerate(JOINT_NAMES):
        ranges = [r[i]['range'] for r in [
            [analyze_temporal_pattern(actions, i, name)
             for i in range(6)]
        ]]
        print(f"\n{name}:")
        for scenario in scenarios.keys():
            idx = list(scenarios.keys()).index(scenario)
            result = all_results[idx]
            range_key = f'{i}_range'
            change_key = f'{i}_total_change'
            if range_key in result:
                print(f"  {scenario:10s}: Range={result[range_key]:6.1f}°, "
                      f"Total Δ={result[change_key]:6.1f}°")

    print("\nGripper:")
    for scenario in scenarios.keys():
        idx = list(scenarios.keys()).index(scenario)
        result = all_results[idx]
        print(f"  {scenario:10s}: Range={result['gripper_range']:6.1f}°, "
              f"Grasp pattern={result['gripper_has_grasp_pattern']}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Check for purposeful behavior
    has_diversity = len(set([r['scenario'] for r in all_results])) > 1
    elbow_changes = [r['2_total_change'] for r in all_results]  # Elbow is index 2
    has_elbow_movement = any(abs(c) > 10 for c in elbow_changes)
    has_gripper_action = any(r['gripper_has_grasp_pattern'] for r in all_results)

    print(f"Scenario diversity: {'YES' if has_diversity else 'NO'}")
    print(f"Meaningful elbow movement (>10°): {'YES' if has_elbow_movement else 'NO'}")
    print(f"Gripper grasp pattern observed: {'YES' if has_gripper_action else 'NO'}")

    if has_elbow_movement and has_gripper_action:
        print("\nPASS: Model shows purposeful, task-relevant behavior")
    elif has_elbow_movement:
        print("\nPARTIAL: Elbow movement present, but no gripper action")
    else:
        print("\nFAIL: Minimal joint movement, likely stuck in mean action")

    print("\nAll analysis complete!")
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
