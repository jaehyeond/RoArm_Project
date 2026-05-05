"""Phase ST-C v3 진단 #2 — Vision conditioning probe.

목적
----
Sim demo 설계가 vision-blind 학습을 유발했는지 정량 검증.

5/05 deploy에서 모델이 sponge 배치 무관하게 ~동일 trajectory(default S1 area)로 직진.
원인 가설: sim_demos_v3 50ep 모두 first-grasp = S1 (50/50 fixed) + spread ±25mm.
→ 모델이 "image → action" 매핑 학습 안 함, "default action 평균" 출력.

방법
----
SmolVLA flow matching은 noise 인자 fix 시 deterministic. 이 사실로 분리 측정:
  σ_det     = 같은 image, 같은 noise, 3 forwards → 0이어야 (sanity)
  σ_noise   = 같은 image, 다른 5 noise → 모델 내부 stochasticity (baseline)
  σ_vision  = 다른 50 image, 같은 noise → image가 action에 미치는 영향

핵심 ratio = σ_vision / σ_noise
  ≈ 1.0   → vision-blind (image 영향 = noise 정도)
  > 5.0   → vision conditioning 작동

Run
---
1) v3 5K (현 deploy ckpt):
   python vision_conditioning_diagnostic.py \
       --ckpt outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model

2) v6 base 비교 (선택):
   python vision_conditioning_diagnostic.py \
       --ckpt outputs/smolvla_v6_b200/checkpoints/last/pretrained_model \
       --task "Pick up the sponge"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from deploy_smolvla import load_model, tokenize_task, build_observation  # noqa: E402

DEFAULT_CKPT = REPO / "outputs/smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model"
DEFAULT_RENDERS = REPO / "sim_renders_v5"
HOME_STATE = [0.0, 0.0, 90.0, 0.0, 0.0, 5.0]
DEFAULT_TASK = "Stack four pink sponges into a # pattern"
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--renders-dir", type=str, default=str(DEFAULT_RENDERS),
                   help="sim_renders_v5 dir; uses episode_XXX/frame_0000.png from each")
    p.add_argument("--n-images", type=int, default=50,
                   help="Number of layout images (max 50 = one per ep)")
    p.add_argument("--task", type=str, default=DEFAULT_TASK)
    p.add_argument("--n-noise-samples", type=int, default=5,
                   help="Different noise tensors for σ_noise estimation")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame-name", type=str, default="frame_0000.png",
                   help="Which frame to take from each episode")
    return p.parse_args()


def load_image_bgr(png_path):
    """cv2.imread returns BGR (matches deploy_smolvla.py expected input)."""
    img = cv2.imread(str(png_path))
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None: {png_path}")
    return img


def fixed_noise(seed, chunk_size=50, max_action_dim=32, device="cuda"):
    """SmolVLA pads actions to max_action_dim=32 internally; noise must match."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, chunk_size, max_action_dim, generator=g, device=device)


def get_chunk(policy, obs_image_bgr, state, lang, stats, noise, device):
    """Run predict_action_chunk and return un-normalized 6-DoF action chunk (T, 6)."""
    obs = build_observation(obs_image_bgr, state, lang, stats, device)
    raw_chunk = policy.predict_action_chunk(obs, noise=noise)  # (1, T, 6) normalized
    # Un-normalize: chunk * std + mean
    a_mean = stats["action_mean"]
    a_std = stats["action_std"]
    chunk = raw_chunk[0] * a_std + a_mean  # (T, 6) deg
    return chunk.detach().cpu().numpy()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    out_json = log_dir / f"vision_diag_{ts}.json"
    out_csv = log_dir / f"vision_diag_{ts}.csv"
    out_png = log_dir / f"vision_diag_{ts}.png"

    # ============================================================
    # [1/5] Load model + state + tokenized task.
    # ============================================================
    print(f"[1/5] Loading ckpt: {args.ckpt}")
    if not Path(args.ckpt).exists():
        sys.exit(f"ERROR: ckpt path not found: {args.ckpt}")
    policy, tokenizer, stats = load_model(args.ckpt, args.device)
    lang = tokenize_task(tokenizer, args.task, args.device)
    state = HOME_STATE
    print(f"  State (HOME) = {state}")
    print(f"  Task = {args.task!r}")

    # Get chunk size + max_action_dim from policy config
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    print(f"  Chunk size = {chunk_size}, max_action_dim = {max_action_dim}, real action_dim = 6")
    # Patch fixed_noise default to use actual config
    global _CHUNK, _MAX_AD
    _CHUNK = chunk_size
    _MAX_AD = max_action_dim

    # ============================================================
    # [2/5] Locate N images.
    # ============================================================
    rdir = Path(args.renders_dir)
    image_paths = []
    for i in range(args.n_images):
        cand = rdir / f"episode_{i:03d}" / args.frame_name
        if cand.exists():
            image_paths.append(cand)
        else:
            print(f"  WARN: missing {cand}")
    if len(image_paths) < 5:
        sys.exit(f"ERROR: too few images found ({len(image_paths)})")
    print(f"\n[2/5] Found {len(image_paths)} images. First={image_paths[0]}, last={image_paths[-1]}")

    # ============================================================
    # [3/5] DETERMINISM CHECK + NOISE SENSITIVITY (σ_det, σ_noise).
    # ============================================================
    print("\n[3/5] Sanity: determinism + noise-only variance using image 0")
    img0 = load_image_bgr(image_paths[0])

    # Same image, same noise, 3 forwards → should match exactly
    n_fix = fixed_noise(42, chunk_size=chunk_size, max_action_dim=max_action_dim, device=args.device)
    chunks_det = [get_chunk(policy, img0, state, lang, stats, n_fix, args.device) for _ in range(3)]
    det_diff = np.max(np.abs(chunks_det[0] - chunks_det[1]))
    print(f"  σ_det (max|chunk_a - chunk_b| same noise): {det_diff:.6f}deg "
          f"({'PASS — deterministic' if det_diff < 0.01 else '!!! FAIL — non-deterministic'})")

    # Same image, N different noises → σ_noise per (frame, joint)
    chunks_noise = []
    for k in range(args.n_noise_samples):
        n_k = fixed_noise(1000 + k, chunk_size=chunk_size, max_action_dim=max_action_dim, device=args.device)
        chunks_noise.append(get_chunk(policy, img0, state, lang, stats, n_k, args.device))
    arr_noise = np.stack(chunks_noise, axis=0)  # (K, T, 6)
    sigma_noise_per_step = arr_noise.std(axis=0)  # (T, 6)
    sigma_noise_first = sigma_noise_per_step[0]
    sigma_noise_chunk_mean = sigma_noise_per_step.mean(axis=0)
    print(f"  σ_noise (first action, K={args.n_noise_samples} noise samples):")
    for i, n in enumerate(JOINT_NAMES):
        print(f"    {n:9s}: {sigma_noise_first[i]:.3f}deg")
    print(f"  σ_noise mean across chunk:")
    for i, n in enumerate(JOINT_NAMES):
        print(f"    {n:9s}: {sigma_noise_chunk_mean[i]:.3f}deg")

    # ============================================================
    # [4/5] VISION SENSITIVITY (σ_vision).
    # ============================================================
    print(f"\n[4/5] Vision sensitivity: {len(image_paths)} different images, same noise={42}")
    n_fix = fixed_noise(42, chunk_size=chunk_size, max_action_dim=max_action_dim, device=args.device)
    chunks_vision = []
    first_actions = []
    t0 = time.time()
    for i, p in enumerate(image_paths):
        img = load_image_bgr(p)
        ch = get_chunk(policy, img, state, lang, stats, n_fix, args.device)
        chunks_vision.append(ch)
        first_actions.append(ch[0])
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(image_paths)} images "
                  f"({(time.time()-t0)/(i+1)*1000:.0f} ms/forward)")
    arr_vis = np.stack(chunks_vision, axis=0)  # (N, T, 6)
    arr_first = np.stack(first_actions, axis=0)  # (N, 6)

    sigma_vision_per_step = arr_vis.std(axis=0)  # (T, 6)
    sigma_vision_first = sigma_vision_per_step[0]
    sigma_vision_chunk_mean = sigma_vision_per_step.mean(axis=0)

    print(f"\n  σ_vision (first action, N={len(image_paths)} images, fixed noise):")
    for i, n in enumerate(JOINT_NAMES):
        print(f"    {n:9s}: {sigma_vision_first[i]:.3f}deg")

    # ============================================================
    # [5/5] Verdict.
    # ============================================================
    print("\n========== VERDICT ==========")
    ratio_first = sigma_vision_first / (sigma_noise_first + 1e-6)
    ratio_chunk = sigma_vision_chunk_mean / (sigma_noise_chunk_mean + 1e-6)
    print("σ_vision / σ_noise (FIRST action):")
    for i, n in enumerate(JOINT_NAMES):
        verdict = (
            "VISION-BLIND" if ratio_first[i] < 1.5 else
            "weak"         if ratio_first[i] < 3.0 else
            "moderate"     if ratio_first[i] < 6.0 else
            "STRONG"
        )
        print(f"  {n:9s}: σ_vision={sigma_vision_first[i]:6.3f}  σ_noise={sigma_noise_first[i]:6.3f}  "
              f"ratio={ratio_first[i]:5.2f}  [{verdict}]")
    print(f"\nσ_vision / σ_noise (chunk-mean):")
    for i, n in enumerate(JOINT_NAMES):
        print(f"  {n:9s}: ratio={ratio_chunk[i]:5.2f}")

    # First-action distribution across images (does it cluster around one trajectory?)
    print(f"\nFirst-action joint distributions (across {len(image_paths)} different layouts):")
    for i, n in enumerate(JOINT_NAMES):
        col = arr_first[:, i]
        print(f"  {n:9s}: mean={col.mean():+7.2f}  std={col.std():5.2f}  "
              f"range=[{col.min():+7.2f}, {col.max():+7.2f}]")

    # Aggregate verdict
    base_ratio_first = ratio_first[0]  # base joint = primary direction signal
    if base_ratio_first < 1.5:
        agg = "VISION-BLIND (strong) — Plan A (sim diversity↑) needed first"
    elif base_ratio_first < 3.0:
        agg = "VISION-BLIND (weak)  — Plan A or Plan C"
    elif base_ratio_first < 6.0:
        agg = "PARTIAL vision conditioning — Plan B (real data) likely sufficient"
    else:
        agg = "STRONG vision conditioning — z-dive must have other root cause"
    print(f"\n>>> AGGREGATE VERDICT (base joint): {agg}")

    # Save artifacts
    summary = {
        "ckpt": str(args.ckpt),
        "task": args.task,
        "n_images": len(image_paths),
        "n_noise_samples": args.n_noise_samples,
        "sigma_det_max_abs": float(det_diff),
        "sigma_noise_first_per_joint": sigma_noise_first.tolist(),
        "sigma_vision_first_per_joint": sigma_vision_first.tolist(),
        "ratio_first_per_joint": ratio_first.tolist(),
        "ratio_chunk_per_joint": ratio_chunk.tolist(),
        "first_action_mean_per_joint": arr_first.mean(axis=0).tolist(),
        "first_action_std_per_joint": arr_first.std(axis=0).tolist(),
        "first_action_range_per_joint": [
            [float(arr_first[:, i].min()), float(arr_first[:, i].max())] for i in range(6)
        ],
        "verdict": agg,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON saved → {out_json}")

    # CSV — per image first action
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_idx", "image_path"] + [f"first_act_{n}" for n in JOINT_NAMES])
        for i, (p, a) in enumerate(zip(image_paths, arr_first)):
            w.writerow([i, str(p)] + [f"{v:+.4f}" for v in a])
    print(f"CSV saved  → {out_csv}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for i, (n, ax) in enumerate(zip(JOINT_NAMES, axes.flatten())):
            # Histogram of first action across N images (vision)
            ax.hist(arr_first[:, i], bins=15, alpha=0.7, label=f"vision (N={len(image_paths)})",
                    color="#1f77b4")
            # Overlay noise samples: take K noise on image 0
            noise_first_arr = arr_noise[:, 0, i]  # K samples of first action joint i
            for v in noise_first_arr:
                ax.axvline(v, color="#d62728", alpha=0.6, linewidth=0.8)
            ax.set_title(f"{n}  σ_vis={sigma_vision_first[i]:.2f}  σ_noise={sigma_noise_first[i]:.2f}  "
                         f"ratio={ratio_first[i]:.2f}")
            ax.set_xlabel("first action (deg)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Vision conditioning probe — {Path(args.ckpt).parent.parent.name}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f"PNG saved  → {out_png}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
