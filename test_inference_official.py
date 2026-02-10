"""
SmolVLA 공식 학습 모델 추론 테스트

20K steps 학습 완료 모델 (outputs/smolvla_official/checkpoints/020000)
mean action 문제 해결 여부 검증

수동으로 normalization 통계 로드하여 unnormalize
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("=" * 60)
    print("SmolVLA Inference Test (Official Training - 20K steps)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset for test samples
    print("\nLoading dataset...")
    dataset = LeRobotDataset(
        repo_id="roarm_m3_pick",
        root=Path("E:/RoArm_Project/lerobot_dataset_v3"),
    )
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")

    # Load trained model
    checkpoint_path = "E:/RoArm_Project/outputs/smolvla_official/checkpoints/020000/pretrained_model"
    print(f"\nLoading model from: {checkpoint_path}")

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total params: {total_params:,}")

    # Load normalization statistics from preprocessor
    pre_stats = load_file(
        f"{checkpoint_path}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    post_stats = load_file(
        f"{checkpoint_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )

    print("\nNormalization stats (from preprocessor):")
    for k, v in pre_stats.items():
        print(f"  {k}: {v.numpy()}")

    print("\nUnnormalization stats (from postprocessor):")
    for k, v in post_stats.items():
        print(f"  {k}: {v.numpy()}")

    # Extract action mean/std for unnormalization
    action_mean = post_stats["action.mean"].to(device)
    action_std = post_stats["action.std"].to(device)
    state_mean = pre_stats["observation.state.mean"].to(device)
    state_std = pre_stats["observation.state.std"].to(device)

    # Get tokenizer for language input
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

    # Test on multiple samples
    print("\n" + "=" * 60)
    print("Inference Results (manual normalize/unnormalize)")
    print("=" * 60)

    test_indices = [0, 50, 100, 150, 200, 500, 1000, 5000, 10000, 13000]
    test_indices = [i for i in test_indices if i < len(dataset)]

    all_actions = []
    all_gt = []

    for idx in test_indices:
        sample = dataset[idx]

        # Build batch with proper normalization
        batch = {}
        for key, val in sample.items():
            if key in ("action", "task", "episode_index", "frame_index",
                       "timestamp", "index", "task_index", "next.done", "next.reward"):
                continue
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device)

        # Normalize state manually (MEAN_STD normalization)
        if "observation.state" in batch:
            batch["observation.state"] = (batch["observation.state"] - state_mean) / (state_std + 1e-8)

        # Add language tokens
        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_mask

        # Reset and predict
        policy.reset()
        with torch.inference_mode():
            raw_action = policy.select_action(batch)

        # Unnormalize action (MEAN_STD unnormalization)
        action = raw_action * action_std + action_mean

        action_np = action.cpu().numpy().squeeze()
        raw_np = raw_action.cpu().numpy().squeeze()
        all_actions.append(action_np)

        # Get ground truth
        gt_action = sample.get("action", None)
        if gt_action is not None:
            gt_np = gt_action.numpy()
            all_gt.append(gt_np)
            l2_error = np.linalg.norm(action_np[:6] - gt_np[:6])
            print(f"\nSample {idx:>5d}:")
            print(f"  Raw (norm): [{', '.join(f'{v:>7.3f}' for v in raw_np[:6])}]")
            print(f"  Predicted:  [{', '.join(f'{v:>7.2f}' for v in action_np[:6])}]")
            print(f"  Ground-T:   [{', '.join(f'{v:>7.2f}' for v in gt_np[:6])}]")
            print(f"  L2 Error:   {l2_error:.4f}")

    # Diversity analysis
    all_actions_arr = np.array([a[:6] for a in all_actions])

    print("\n" + "=" * 60)
    print("Diversity Analysis (Mean Action Problem Check)")
    print("=" * 60)

    joint_names = ["Base", "Shoulder", "Elbow", "Wrist_P", "Wrist_R", "Gripper"]
    stds = np.std(all_actions_arr, axis=0)
    means = np.mean(all_actions_arr, axis=0)

    print(f"\n{'Joint':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for i, name in enumerate(joint_names):
        col = all_actions_arr[:, i]
        print(f"{name:<12} {means[i]:>10.4f} {stds[i]:>10.4f} {np.min(col):>10.4f} {np.max(col):>10.4f}")

    total_std = np.mean(stds)
    print(f"\n{'Overall Std':>12}: {total_std:.4f}")

    # L2 error summary
    if all_gt:
        all_gt_arr = np.array([g[:6] for g in all_gt])
        l2_errors = np.linalg.norm(all_actions_arr - all_gt_arr, axis=1)
        print(f"\nL2 Error Summary:")
        print(f"  Mean L2: {np.mean(l2_errors):.4f}")
        print(f"  Std L2:  {np.std(l2_errors):.4f}")
        print(f"  Min L2:  {np.min(l2_errors):.4f}")
        print(f"  Max L2:  {np.max(l2_errors):.4f}")

    # Verdict
    print("\n" + "=" * 60)

    stats = dataset.meta.stats
    if "action" in stats:
        ds_std = np.array(stats["action"]["std"][:6])
        ds_mean = np.array(stats["action"]["mean"][:6])
        print(f"Dataset action mean: [{', '.join(f'{v:.2f}' for v in ds_mean)}]")
        print(f"Dataset action std:  [{', '.join(f'{v:.2f}' for v in ds_std)}]")
        print(f"Predicted mean:      [{', '.join(f'{v:.2f}' for v in means)}]")
        print(f"Predicted std:       [{', '.join(f'{v:.2f}' for v in stds)}]")

        mean_distance = np.linalg.norm(means - ds_mean)
        print(f"\nDistance (predicted mean vs dataset mean): {mean_distance:.4f}")

    if total_std < 1.0:
        print("\nVERDICT: FAIL - Mean action problem detected")
        print(f"  Predicted std: {total_std:.4f}, Expected: ~{np.mean(ds_std):.2f}")
    elif total_std < 5.0:
        print("\nVERDICT: MARGINAL - Some diversity, needs real robot testing")
        print(f"  Predicted std: {total_std:.4f}")
    else:
        print("\nVERDICT: PASS - Diverse, sample-dependent actions!")
        print(f"  Predicted std: {total_std:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
