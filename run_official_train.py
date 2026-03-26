"""
공식 lerobot-train 파이프라인으로 SmolVLA 학습

smolvla_base 사전학습 모델 사용 (Action Expert + VLM 모두 사전학습됨)
공식 파이프라인이 정규화, LR 스케줄러, gradient clipping 등 자동 처리

=== V5 Config: 200K steps (5-zone multi-position grasping) ===
- Dataset: 150 episodes (lerobot_dataset_v5), ~26,700 frames
- batch_size=64 (공식 권장, RTX 4090 Laptop에서 9.85GB/16.7GB = 59%)
- steps=200,000 (OOD 로봇은 150K-200K 필요, 공식 상한)
- scheduler_decay_steps=200,000 (전체 학습 구간에 걸쳐 cosine decay)
- scheduler_warmup_steps=2,000 (전체의 1%, OOD 초반 불안정 방지)
- save_freq=10,000 (200K/10K = 20 체크포인트, 평가용)
- eval_freq=20,000 (중간 점검 10회)

=== Epoch 계산 (150 eps, 178 frames/ep, bs=64) ===
  frames      = 150 * 178 = 26,700
  steps/epoch = 26,700 / 64 ≈ 417
  epochs      = 200,000 / 417 ≈ 480 (적절: 과적합 방지에 충분한 다양성)

=== Scheduler 설명 ===
  CosineDecayWithWarmupScheduler:
  - peak_lr=1e-4 (step 0 → warmup → step 2,000)
  - cosine decay: step 2,000 → 200,000
  - decay_lr=2.5e-6 (최저 LR, peak의 2.5%)
  - num_decay_steps=200,000 (전체 steps와 일치 → smooth decay)
  Note: num_training_steps < num_decay_steps이면 auto-scale됨
        → decay_steps를 steps와 정확히 맞추는 게 가장 안전

=== Auto-scale 동작 (schedulers.py line 99-111) ===
  if num_training_steps < num_decay_steps:
      scale_factor = num_training_steps / num_decay_steps
      actual_warmup = int(warmup * scale_factor)
      actual_decay  = num_training_steps
  → decay_steps > steps이면 warmup이 줄어들어 LR 불안정 가능성!
  → decay_steps = steps로 설정하면 auto-scale 없이 의도한 schedule 실행

=== CLI 인자 전달 방식 ===
  scheduler 관련 설정은 policy 필드를 통해 전달:
  --policy.scheduler_warmup_steps=N
  --policy.scheduler_decay_steps=N
  --policy.scheduler_decay_lr=F
  (use_policy_training_preset=True가 기본값이므로 policy config에서 scheduler 생성)
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unbuffered output for piped output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# === V5 Dataset (150 episodes, 5-zone multi-position sponge pick) ===
DATASET_ROOT = "lerobot_dataset_v5"
DATASET_REPO = "roarm_m3_pick"
OUTPUT_DIR = "outputs/smolvla_v5_multipos"

# === Training hyperparameters ===
STEPS = 200_000
BATCH_SIZE = 64
SCHEDULER_WARMUP_STEPS = 2_000    # 1% of total steps (OOD 초반 불안정 방지)
SCHEDULER_DECAY_STEPS = 200_000   # = STEPS (전체 구간 cosine decay, auto-scale 방지)
SCHEDULER_DECAY_LR = 2.5e-6      # peak_lr의 2.5% (official SmolVLA 기본값 유지)
SAVE_FREQ = 10_000                # 20 checkpoints total (200K / 10K)
EVAL_FREQ = 20_000                # 10 eval points (200K / 20K)
LOG_FREQ = 100

# Resume or fresh start
last_ckpt_file = Path(f"{OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json")

if last_ckpt_file.exists():
    print(f"Resuming from checkpoint: {last_ckpt_file.parent}")
    sys.argv = [
        "lerobot-train",
        f"--config_path={last_ckpt_file}",
        "--resume=true",
    ]
else:
    # Fresh start from smolvla_base pretrained
    print("Starting fresh training (SmolVLA v5 5-zone multipos, 200K steps)...")
    print(f"  Dataset: {DATASET_ROOT}/{DATASET_REPO}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  batch_size={BATCH_SIZE}, steps={STEPS:,}")
    print(f"  Scheduler: warmup={SCHEDULER_WARMUP_STEPS}, decay={SCHEDULER_DECAY_STEPS}, decay_lr={SCHEDULER_DECAY_LR}")
    print(f"  Checkpoints: every {SAVE_FREQ:,} steps ({STEPS // SAVE_FREQ} total)")
    print()
    print("  Estimated: ~150ep * 178fr / 64bs = 417 steps/epoch → ~480 epochs")
    print("  VRAM: batch_size=64 → ~9.85 GB (59% of 16.72 GB)")
    sys.argv = [
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.pretrained_path=lerobot/smolvla_base",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={DATASET_ROOT}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--eval_freq={EVAL_FREQ}",
        f"--save_freq={SAVE_FREQ}",
        f"--log_freq={LOG_FREQ}",
        f"--output_dir={OUTPUT_DIR}",
        "--num_workers=4",
        "--policy.device=cuda",
        # Scheduler: cosine decay spanning the full training run
        f"--policy.scheduler_warmup_steps={SCHEDULER_WARMUP_STEPS}",
        f"--policy.scheduler_decay_steps={SCHEDULER_DECAY_STEPS}",
        f"--policy.scheduler_decay_lr={SCHEDULER_DECAY_LR}",
    ]

from lerobot.scripts.lerobot_train import main

main()
