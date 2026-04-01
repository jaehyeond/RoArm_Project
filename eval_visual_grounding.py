"""
Visual Grounding Sensitivity Test (VGST)
========================================
v5 배포 실패 원인 진단: 모델이 이미지에서 sponge 위치를 base 각도로 매핑하는지 검증.

기존 eval_v5_checkpoints.py의 L2/std 메트릭은 visual grounding을 측정하지 않음.
이 스크립트는 5가지 메트릭으로 visual grounding을 직접 테스트:

  M1. Per-joint Pearson correlation (pred vs GT)
  M2. Image permutation ablation (이미지 셔플 시 base 변화 측정)
  M3. Zone directional accuracy (좌/우 방향 맞추는지)
  M4. Per-zone BASE-only L2 (6D L2가 아닌 base만)
  M5. Constant-predictor baseline comparison

Usage:
    python eval_visual_grounding.py
    python eval_visual_grounding.py --checkpoint outputs/smolvla_v5_multipos/checkpoints/120000/pretrained_model
    python eval_visual_grounding.py --n-samples 100
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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]
DATASET_ROOT = "lerobot_dataset_v5"
DATASET_REPO = "roarm_m3_pick"
DEFAULT_CHECKPOINT = "outputs/smolvla_v5_multipos/checkpoints/120000/pretrained_model"
TASK_TEXT = "Pick up the sponge"

# Zone thresholds (base angle)
ZONE_LEFT_THRESHOLD = -15    # base < -15 = LEFT
ZONE_RIGHT_THRESHOLD = 15    # base > 15 = RIGHT

# VGST pass/fail thresholds
PEARSON_PASS = 0.50           # base correlation with GT
PEARSON_FAIL = 0.30           # below this = deployment blocked
IMAGE_SENSITIVITY_PASS = 5.0  # degrees: base change when image shuffled
IMAGE_SENSITIVITY_FAIL = 2.0
DIRECTIONAL_ACC_PASS = 0.70   # fraction of correct left/right predictions
BASELINE_IMPROVEMENT = 0.30   # model must beat constant predictor by 30%
MIN_ZONE_SAMPLES = 10         # minimum samples per zone for valid analysis


def build_zone_balanced_indices(dataset, target_per_zone=30):
    """Zone-balanced 샘플링: LEFT/CENTER/RIGHT 각각 최소 target_per_zone개.

    기존 eval의 uniform 샘플링(50개 균등간격)과 달리,
    각 zone에서 충분한 샘플을 확보하여 통계적 유의성 보장.
    """
    n_frames = len(dataset)

    # 모든 프레임의 episode_index와 base angle 수집 (샘플링으로)
    # 전체 스캔은 너무 느리므로 episode 단위로 대표 프레임 선택
    ep_count = dataset.num_episodes

    left_indices = []
    center_indices = []
    right_indices = []

    # Episode별로 중간 프레임의 base angle 확인하여 zone 분류
    ep_start = 0
    for ep_idx in range(ep_count):
        # episode_data_index에서 시작/끝 프레임 찾기
        ep_len_info = dataset.meta.episodes[ep_idx]
        ep_length = ep_len_info.get("length", 100)

        # 에피소드 중간 프레임 3개 선택 (시작, 중간, 끝 근처)
        mid_frames = [
            ep_start + ep_length // 4,
            ep_start + ep_length // 2,
            ep_start + 3 * ep_length // 4,
        ]
        mid_frames = [min(f, n_frames - 1) for f in mid_frames]

        # 첫 번째 유효 프레임의 GT base angle로 zone 판정
        sample = dataset[mid_frames[1]]
        gt_action = sample.get("action", None)
        if gt_action is None:
            ep_start += ep_length
            continue

        gt_base = gt_action[0].item()

        if gt_base < ZONE_LEFT_THRESHOLD:
            left_indices.extend(mid_frames)
        elif gt_base > ZONE_RIGHT_THRESHOLD:
            right_indices.extend(mid_frames)
        else:
            center_indices.extend(mid_frames)

        ep_start += ep_length

    # 각 zone에서 target_per_zone개 선택 (부족하면 전부 사용)
    rng = np.random.RandomState(42)

    def sample_zone(indices, target):
        indices = list(set(indices))  # deduplicate
        if len(indices) <= target:
            return indices
        return list(rng.choice(indices, size=target, replace=False))

    selected_left = sample_zone(left_indices, target_per_zone)
    selected_center = sample_zone(center_indices, target_per_zone)
    selected_right = sample_zone(right_indices, target_per_zone)

    print(f"\n  Zone-balanced sampling:")
    print(f"    LEFT:   {len(selected_left)} / {len(left_indices)} available")
    print(f"    CENTER: {len(selected_center)} / {len(center_indices)} available")
    print(f"    RIGHT:  {len(selected_right)} / {len(right_indices)} available")

    all_indices = selected_left + selected_center + selected_right
    zones = (["LEFT"] * len(selected_left) +
             ["CENTER"] * len(selected_center) +
             ["RIGHT"] * len(selected_right))

    return all_indices, zones


def run_inference(policy, dataset, indices, state_mean, state_std,
                  action_mean, action_std, lang_tokens, lang_mask, device,
                  permute_images=False):
    """주어진 인덱스들에 대해 추론 실행.

    permute_images=True: 이미지를 셔플하여 state는 유지하되 이미지만 다른 프레임의 것 사용.
    """
    all_actions = []
    all_gt = []

    # 이미지 셔플용 인덱스
    if permute_images:
        rng = np.random.RandomState(123)
        perm = rng.permutation(len(indices))
        image_indices = [indices[p] for p in perm]
    else:
        image_indices = indices

    for i, idx in enumerate(indices):
        img_idx = image_indices[i]

        sample = dataset[idx]
        batch = {}

        # State는 원본 프레임에서
        for key, val in sample.items():
            if key in ("action", "task", "episode_index", "frame_index",
                       "timestamp", "index", "task_index", "next.done", "next.reward"):
                continue
            if isinstance(val, torch.Tensor):
                batch[key] = val.unsqueeze(0).to(device)

        # 이미지 퍼뮤테이션: 이미지만 다른 프레임에서 가져오기
        if permute_images and img_idx != idx:
            img_sample = dataset[img_idx]
            for key, val in img_sample.items():
                if "image" in key and isinstance(val, torch.Tensor):
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
            all_gt.append(gt.numpy()[:6])

    return np.array(all_actions), np.array(all_gt)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_verdict(name, passed, detail=""):
    symbol = "PASS" if passed else "FAIL"
    color_hint = "OK" if passed else "BLOCKED"
    print(f"  >>> {name}: {symbol} {detail} [{color_hint}]")


def main():
    parser = argparse.ArgumentParser(description="Visual Grounding Sensitivity Test")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DATASET_ROOT, help="Dataset root directory (default: lerobot_dataset_v5)")
    parser.add_argument("--n-samples", type=int, default=30, help="Target samples per zone")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_root}...")
    dataset = LeRobotDataset(repo_id=DATASET_REPO, root=Path(args.dataset_root))
    print(f"  Frames: {len(dataset)}, Episodes: {dataset.num_episodes}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Load normalization stats
    post_stats = load_file(
        f"{args.checkpoint}/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    pre_stats = load_file(
        f"{args.checkpoint}/policy_preprocessor_step_5_normalizer_processor.safetensors"
    )
    action_mean = post_stats["action.mean"].to(device)
    action_std = post_stats["action.std"].to(device)
    state_mean = pre_stats["observation.state.mean"].to(device)
    state_std = pre_stats["observation.state.std"].to(device)

    # Dataset mean (for constant-predictor baseline)
    ds_action_mean = np.array(dataset.meta.stats["action"]["mean"][:6])

    # Tokenize task
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer
    tokenized = tokenizer(
        [TASK_TEXT], max_length=48, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_mask = tokenized["attention_mask"].bool().to(device)

    # Zone-balanced sampling
    print("\nBuilding zone-balanced test set...")
    indices, zones = build_zone_balanced_indices(dataset, target_per_zone=args.n_samples)
    zones_arr = np.array(zones)

    # ================================================================
    # Phase 1: Normal inference
    # ================================================================
    print_header("Phase 1: Normal Inference")
    print(f"  Running inference on {len(indices)} samples...")

    pred_actions, gt_actions = run_inference(
        policy, dataset, indices, state_mean, state_std,
        action_mean, action_std, lang_tokens, lang_mask, device,
        permute_images=False
    )

    print(f"  Done. pred shape={pred_actions.shape}, gt shape={gt_actions.shape}")

    # ================================================================
    # M1: Per-joint Pearson Correlation
    # ================================================================
    print_header("M1: Per-joint Pearson Correlation (pred vs GT)")
    print(f"  {'Joint':<12} {'r':>8} {'Threshold':>12} {'Verdict':>8}")
    print(f"  {'-'*44}")

    correlations = {}
    m1_pass = True
    for i, name in enumerate(JOINT_NAMES):
        r = np.corrcoef(pred_actions[:, i], gt_actions[:, i])[0, 1]
        correlations[name] = r

        if name == "Base":
            if r >= PEARSON_PASS:
                verdict = "PASS"
            elif r >= PEARSON_FAIL:
                verdict = "WARN"
                m1_pass = False
            else:
                verdict = "FAIL"
                m1_pass = False
            threshold_str = f">={PEARSON_PASS:.2f}"
        else:
            verdict = "OK" if r > 0.3 else "LOW"
            threshold_str = ">=0.30"

        print(f"  {name:<12} {r:>8.3f} {threshold_str:>12} {verdict:>8}")

    print_verdict("M1 Base Correlation", m1_pass,
                  f"(r={correlations['Base']:.3f}, need >={PEARSON_PASS})")

    # ================================================================
    # M2: Image Permutation Ablation
    # ================================================================
    print_header("M2: Image Permutation Ablation")
    print(f"  Running inference with SHUFFLED images (same states)...")

    pred_shuffled, _ = run_inference(
        policy, dataset, indices, state_mean, state_std,
        action_mean, action_std, lang_tokens, lang_mask, device,
        permute_images=True
    )

    # Per-joint image sensitivity
    print(f"\n  {'Joint':<12} {'Sensitivity':>12} {'Threshold':>12} {'Verdict':>8}")
    print(f"  {'-'*48}")

    m2_pass = True
    for i, name in enumerate(JOINT_NAMES):
        sensitivity = np.mean(np.abs(pred_actions[:, i] - pred_shuffled[:, i]))

        if name == "Base":
            if sensitivity >= IMAGE_SENSITIVITY_PASS:
                verdict = "PASS"
            elif sensitivity >= IMAGE_SENSITIVITY_FAIL:
                verdict = "WARN"
                m2_pass = False
            else:
                verdict = "FAIL"
                m2_pass = False
            threshold_str = f">={IMAGE_SENSITIVITY_PASS:.1f}°"
        else:
            verdict = f"{sensitivity:.2f}°"
            threshold_str = "info"

        print(f"  {name:<12} {sensitivity:>10.2f}° {threshold_str:>12} {verdict:>8}")

    base_sensitivity = np.mean(np.abs(pred_actions[:, 0] - pred_shuffled[:, 0]))
    print_verdict("M2 Image Sensitivity (Base)", m2_pass,
                  f"({base_sensitivity:.2f}°, need >={IMAGE_SENSITIVITY_PASS}°)")

    # ================================================================
    # M3: Zone Directional Accuracy
    # ================================================================
    print_header("M3: Zone Directional Accuracy")

    left_mask = zones_arr == "LEFT"
    center_mask = zones_arr == "CENTER"
    right_mask = zones_arr == "RIGHT"

    m3_pass = True
    for zone_name, mask in [("LEFT", left_mask), ("RIGHT", right_mask)]:
        n = mask.sum()
        if n < MIN_ZONE_SAMPLES:
            print(f"  {zone_name}: INSUFFICIENT DATA (n={n}, need >={MIN_ZONE_SAMPLES})")
            m3_pass = False
            continue

        pred_base = pred_actions[mask, 0]
        gt_base = gt_actions[mask, 0]

        # Directional: does pred_base have same sign as gt_base?
        if zone_name == "LEFT":
            correct = pred_base < 0  # LEFT에서는 pred가 음수여야 함
        else:
            correct = pred_base > 0  # RIGHT에서는 pred가 양수여야 함

        acc = np.mean(correct)
        passed = acc >= DIRECTIONAL_ACC_PASS
        if not passed:
            m3_pass = False

        print(f"  {zone_name} (n={n}): accuracy={acc:.1%} "
              f"(need >={DIRECTIONAL_ACC_PASS:.0%}) "
              f"{'PASS' if passed else 'FAIL'}")
        print(f"    pred_base: mean={np.mean(pred_base):.1f}°, "
              f"std={np.std(pred_base):.1f}°, "
              f"range=[{np.min(pred_base):.1f}, {np.max(pred_base):.1f}]")
        print(f"    gt_base:   mean={np.mean(gt_base):.1f}°, "
              f"std={np.std(gt_base):.1f}°, "
              f"range=[{np.min(gt_base):.1f}, {np.max(gt_base):.1f}]")

    print_verdict("M3 Directional Accuracy", m3_pass)

    # ================================================================
    # M4: Per-zone BASE-only L2 (not 6D L2!)
    # ================================================================
    print_header("M4: Per-zone BASE-only L2 Error")

    m4_results = {}
    for zone_name, mask in [("LEFT", left_mask), ("CENTER", center_mask), ("RIGHT", right_mask)]:
        n = mask.sum()
        if n < 3:
            print(f"  {zone_name}: INSUFFICIENT (n={n})")
            continue

        base_l2 = np.mean(np.abs(pred_actions[mask, 0] - gt_actions[mask, 0]))
        total_l2 = np.mean(np.linalg.norm(pred_actions[mask] - gt_actions[mask], axis=1))
        m4_results[zone_name] = base_l2

        print(f"  {zone_name} (n={n}): base_L2={base_l2:.2f}°, total_L2={total_l2:.2f}°")

    if "LEFT" in m4_results and "CENTER" in m4_results:
        ratio = m4_results["LEFT"] / max(m4_results["CENTER"], 0.01)
        print(f"\n  LEFT/CENTER base_L2 ratio: {ratio:.2f}")
        if ratio > 3.0:
            print(f"  WARNING: Model is {ratio:.1f}x worse on LEFT base angle")

    # ================================================================
    # M5: Constant-predictor Baseline Comparison
    # ================================================================
    print_header("M5: Constant-predictor Baseline Comparison")
    print(f"  Baseline: always predict dataset_mean = [{', '.join(f'{v:.1f}' for v in ds_action_mean)}]")

    m5_pass = True
    for zone_name, mask in [("LEFT", left_mask), ("RIGHT", right_mask)]:
        n = mask.sum()
        if n < MIN_ZONE_SAMPLES:
            print(f"  {zone_name}: INSUFFICIENT (n={n})")
            m5_pass = False
            continue

        # Model base L2
        model_base_l2 = np.mean(np.abs(pred_actions[mask, 0] - gt_actions[mask, 0]))
        # Constant predictor base L2
        baseline_base_l2 = np.mean(np.abs(ds_action_mean[0] - gt_actions[mask, 0]))

        improvement = 1.0 - (model_base_l2 / max(baseline_base_l2, 0.01))
        passed = improvement >= BASELINE_IMPROVEMENT
        if not passed:
            m5_pass = False

        print(f"  {zone_name} (n={n}):")
        print(f"    Model base_L2:    {model_base_l2:.2f}°")
        print(f"    Baseline base_L2: {baseline_base_l2:.2f}°")
        print(f"    Improvement:      {improvement:.1%} (need >={BASELINE_IMPROVEMENT:.0%}) "
              f"{'PASS' if passed else 'FAIL'}")

    print_verdict("M5 Beats Constant Baseline", m5_pass)

    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    print_header("OVERALL VISUAL GROUNDING VERDICT")

    all_pass = m1_pass and m2_pass and m3_pass and m5_pass

    results = [
        ("M1 Pearson Correlation (Base)", m1_pass),
        ("M2 Image Sensitivity (Base)", m2_pass),
        ("M3 Directional Accuracy", m3_pass),
        ("M5 Beats Baseline", m5_pass),
    ]

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {'='*50}")
    if all_pass:
        print(f"  VISUAL GROUNDING: PASS — safe to deploy")
    else:
        failed = [name for name, p in results if not p]
        print(f"  VISUAL GROUNDING: FAIL — DO NOT DEPLOY")
        print(f"  Failed: {', '.join(failed)}")
        print(f"\n  Root cause: Model does not use image for base angle prediction.")
        print(f"  Fix: Collect balanced data across base angles, retrain.")
    print(f"  {'='*50}")

    # ================================================================
    # Supplementary: Prediction Distribution
    # ================================================================
    print_header("Supplementary: Prediction Distribution by Zone")

    for zone_name, mask in [("LEFT", left_mask), ("CENTER", center_mask), ("RIGHT", right_mask)]:
        n = mask.sum()
        if n < 3:
            continue

        pred_base_zone = pred_actions[mask, 0]
        gt_base_zone = gt_actions[mask, 0]

        print(f"\n  {zone_name} (n={n}):")
        print(f"    GT base:   mean={np.mean(gt_base_zone):>7.1f}° "
              f"std={np.std(gt_base_zone):>5.1f}° "
              f"[{np.min(gt_base_zone):>6.1f}, {np.max(gt_base_zone):>6.1f}]")
        print(f"    Pred base: mean={np.mean(pred_base_zone):>7.1f}° "
              f"std={np.std(pred_base_zone):>5.1f}° "
              f"[{np.min(pred_base_zone):>6.1f}, {np.max(pred_base_zone):>6.1f}]")

        # Per-sample comparison (first 5)
        if n <= 10:
            print(f"    Sample details:")
            for j in range(n):
                diff = pred_base_zone[j] - gt_base_zone[j]
                print(f"      GT={gt_base_zone[j]:>7.1f}° → Pred={pred_base_zone[j]:>7.1f}° "
                      f"(diff={diff:>+6.1f}°)")

    # Cleanup
    del policy
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
