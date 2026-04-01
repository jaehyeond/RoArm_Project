"""
V5 Multi-Checkpoint Evaluation
Tests priority checkpoints: 50K, 80K, 120K, 200K
Metrics: L2 error, per-joint std, gripper coverage, z-score outliers, diversity
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

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]
CHECKPOINTS = [50000, 80000, 120000, 200000]
DATASET_ROOT = "lerobot_dataset_v5"
DATASET_REPO = "roarm_m3_pick"
OUTPUT_DIR = "outputs/smolvla_v5_multipos/checkpoints"
TASK_TEXT = "Pick up the sponge"

# Minimum thresholds (from VLA Scientist recommendations)
MIN_JOINT_STD = [5.0, 3.0, 5.0, 5.0, 5.0, 5.0]  # per joint
MIN_GRIPPER_OPEN_RATIO = 0.20  # 20% of predictions should have gripper > 40°
MAX_ZSCORE_OUTLIER_RATIO = 0.05  # <5% of predictions with z-score > 3


def evaluate_checkpoint(step, dataset, device):
    ckpt_path = f"{OUTPUT_DIR}/{step:06d}/pretrained_model"
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: {ckpt_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"Checkpoint: {step:,} steps")
    print(f"{'='*60}")

    # Load model
    policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()

    # Load normalization stats
    post_stats = load_file(
        f"{ckpt_path}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    pre_stats = load_file(
        f"{ckpt_path}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )

    action_mean = post_stats["action.mean"].to(device)
    action_std = post_stats["action.std"].to(device)
    state_mean = pre_stats["observation.state.mean"].to(device)
    state_std = pre_stats["observation.state.std"].to(device)

    # Tokenize task
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer
    tokenized = tokenizer(
        [TASK_TEXT], max_length=48, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_mask = tokenized["attention_mask"].bool().to(device)

    # Sample across dataset — every 100 frames, spread across all zones
    n_frames = len(dataset)
    test_indices = list(range(0, n_frames, max(1, n_frames // 50)))[:50]

    all_actions = []
    all_gt = []
    l2_errors = []

    for idx in test_indices:
        sample = dataset[idx]
        batch = {}
        for key, val in sample.items():
            if key in ("action", "task", "episode_index", "frame_index",
                       "timestamp", "index", "task_index", "next.done", "next.reward"):
                continue
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device)

        if "observation.state" in batch:
            batch["observation.state"] = (batch["observation.state"] - state_mean) / (state_std + 1e-8)

        batch["observation.language.tokens"] = lang_tokens
        batch["observation.language.attention_mask"] = lang_mask

        policy.reset()
        with torch.inference_mode():
            raw_action = policy.select_action(batch)

        action = raw_action * action_std + action_mean
        action_np = action.cpu().numpy().squeeze()[:6]
        all_actions.append(action_np)

        gt = sample.get("action", None)
        if gt is not None:
            gt_np = gt.numpy()[:6]
            all_gt.append(gt_np)
            l2 = np.linalg.norm(action_np - gt_np)
            l2_errors.append(l2)

    all_actions_arr = np.array(all_actions)
    all_gt_arr = np.array(all_gt)

    # === Metric 1: Per-joint std ===
    stds = np.std(all_actions_arr, axis=0)
    means = np.mean(all_actions_arr, axis=0)

    print(f"\n{'Joint':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Thresh':>8} {'Pass':>6}")
    print("-" * 62)
    std_pass = True
    for i, name in enumerate(JOINT_NAMES):
        col = all_actions_arr[:, i]
        passed = stds[i] >= MIN_JOINT_STD[i]
        if not passed:
            std_pass = False
        print(f"{name:<10} {means[i]:>8.2f} {stds[i]:>8.2f} {np.min(col):>8.2f} {np.max(col):>8.2f} {MIN_JOINT_STD[i]:>8.1f} {'OK' if passed else 'FAIL':>6}")

    # === Metric 2: Gripper open coverage ===
    gripper_preds = all_actions_arr[:, 5]
    gripper_open_ratio = np.mean(gripper_preds > 40)
    gripper_pass = gripper_open_ratio >= MIN_GRIPPER_OPEN_RATIO

    print(f"\nGripper open ratio: {gripper_open_ratio:.1%} (threshold: {MIN_GRIPPER_OPEN_RATIO:.0%}) {'OK' if gripper_pass else 'FAIL'}")

    # === Metric 3: Z-score outliers ===
    ds_stats = dataset.meta.stats
    ds_mean = np.array(ds_stats["action"]["mean"][:6])
    ds_std = np.array(ds_stats["action"]["std"][:6])

    z_scores = np.abs((all_actions_arr - ds_mean) / (ds_std + 1e-8))
    outlier_ratio = np.mean(z_scores > 3.0)
    zscore_pass = outlier_ratio <= MAX_ZSCORE_OUTLIER_RATIO

    print(f"Z-score outlier ratio: {outlier_ratio:.1%} (threshold: {MAX_ZSCORE_OUTLIER_RATIO:.0%}) {'OK' if zscore_pass else 'FAIL'}")

    # === Metric 4: L2 error ===
    l2_arr = np.array(l2_errors)
    mean_l2 = np.mean(l2_arr)
    print(f"Mean L2 error: {mean_l2:.2f}° (std={np.std(l2_arr):.2f}, min={np.min(l2_arr):.2f}, max={np.max(l2_arr):.2f})")

    # === Metric 5: Overall diversity (mean action check) ===
    overall_std = np.mean(stds)
    mean_action_check = "FAIL" if overall_std < 1.0 else ("MARGINAL" if overall_std < 5.0 else "PASS")
    print(f"Overall std: {overall_std:.2f}° — {mean_action_check}")

    # === Metric 6: Zone-conditioned L2 (approximate) ===
    # Split samples into quartiles by base angle to approximate zones
    base_angles = all_gt_arr[:, 0]  # ground truth base
    left_mask = base_angles < -15
    center_mask = (base_angles >= -15) & (base_angles <= 15)
    right_mask = base_angles > 15

    zone_l2s = {}
    for name, mask in [("LEFT", left_mask), ("CENTER", center_mask), ("RIGHT", right_mask)]:
        if mask.sum() > 0:
            zone_l2 = np.mean(np.linalg.norm(all_actions_arr[mask] - all_gt_arr[mask], axis=1))
            zone_l2s[name] = zone_l2
            print(f"  Zone {name}: L2={zone_l2:.2f}° (n={mask.sum()})")

    if len(zone_l2s) >= 2:
        max_z = max(zone_l2s.values())
        min_z = min(zone_l2s.values())
        ratio = max_z / max(min_z, 0.01)
        print(f"  Zone L2 ratio (max/min): {ratio:.2f} (threshold: <2.0) {'OK' if ratio < 2.0 else 'WARN'}")

    # === VERDICT ===
    all_pass = std_pass and gripper_pass and zscore_pass and mean_action_check == "PASS"
    verdict = "PASS" if all_pass else "MARGINAL" if mean_action_check != "FAIL" else "FAIL"

    result = {
        "step": step,
        "loss": None,  # from training log
        "mean_l2": mean_l2,
        "overall_std": overall_std,
        "gripper_open_ratio": gripper_open_ratio,
        "zscore_outlier_ratio": outlier_ratio,
        "per_joint_std": stds.tolist(),
        "per_joint_mean": means.tolist(),
        "verdict": verdict,
        "zone_l2s": zone_l2s,
    }

    print(f"\n>>> VERDICT: {verdict} <<<")

    # Free GPU memory
    del policy
    torch.cuda.empty_cache()

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id=DATASET_REPO, root=Path(DATASET_ROOT))
    print(f"  Frames: {len(dataset)}, Episodes: {dataset.num_episodes}")

    results = []
    for step in CHECKPOINTS:
        r = evaluate_checkpoint(step, dataset, device)
        if r:
            results.append(r)

    # === COMPARISON TABLE ===
    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON")
    print("=" * 80)
    print(f"{'Step':>8} {'L2':>8} {'Std':>8} {'Grip%':>8} {'Zout%':>8} {'Verdict':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r['step']:>8,} {r['mean_l2']:>8.2f} {r['overall_std']:>8.2f} {r['gripper_open_ratio']:>7.1%} {r['zscore_outlier_ratio']:>7.1%} {r['verdict']:>10}")

    # Best checkpoint
    passing = [r for r in results if r["verdict"] == "PASS"]
    if passing:
        best = min(passing, key=lambda r: r["mean_l2"])
        print(f"\n>>> BEST CHECKPOINT: {best['step']:,} steps (L2={best['mean_l2']:.2f}°, Std={best['overall_std']:.2f}°)")
    else:
        marginal = [r for r in results if r["verdict"] == "MARGINAL"]
        if marginal:
            best = min(marginal, key=lambda r: r["mean_l2"])
            print(f"\n>>> BEST MARGINAL: {best['step']:,} steps (L2={best['mean_l2']:.2f}°)")
        else:
            print("\n>>> ALL CHECKPOINTS FAILED — mean action problem or collapse")


if __name__ == "__main__":
    main()
