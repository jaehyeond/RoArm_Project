"""
공식 lerobot-train 파이프라인으로 SmolVLA 학습

smolvla_base 사전학습 모델 사용 (Action Expert + VLM 모두 사전학습됨)
공식 파이프라인이 정규화, LR 스케줄러, gradient clipping 등 자동 처리

=== V6 Config: 20K steps (공식 바닐라 레시피 정확 재현) ===
- Dataset: 50 episodes (lerobot_dataset_v6), 5 zones × 10 reps
- batch_size=64 (공식 권장)
- steps=20,000 (공식 기본값 — SO100 pickplace 50ep 기준)
- scheduler: SmolVLA config defaults (warmup=1000, decay=30000)
- save_freq=5,000 (4 체크포인트)

=== 과거 실패 기록 (절대 반복 금지) ===
  v1: bs=8 비표준 → 실패
  v3: 50K/74ep → false positive (1곳만 테스트, 궤적 암기, M2=1.73°)
  v5: 200K/136ep → 실패 (HOME 미시작 → echo → 제자리)
  → 200K는 공식 20K의 10배 과다. 에피소드 길이(3.3초)도 공식(13초)의 1/4

=== Epoch 계산 (50 eps, ~200 frames/ep, bs=64) ===
  frames      = 50 * 200 = 10,000
  steps/epoch = 10,000 / 64 ≈ 156
  epochs      = 20,000 / 156 ≈ 128

=== Scheduler (공식 SmolVLA config defaults) ===
  warmup_steps = 1,000 (SmolVLAConfig default)
  decay_steps  = 30,000 (SmolVLAConfig default)
  decay_lr     = 2.5e-6 (SmolVLAConfig default)
  Note: decay_steps(30K) > steps(20K) → auto-scale 발생
        → 공식이 이렇게 설계함. 이 동작이 정상.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unbuffered output for piped output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# === V6 Dataset (50 episodes, 5-zone × 10 reps, HOME start, 공식 레시피 재현) ===
DATASET_ROOT = "lerobot_dataset_v6"
DATASET_REPO = "roarm_m3_pick"
OUTPUT_DIR = "outputs/smolvla_v6"

# === Training hyperparameters (공식 SmolVLA defaults) ===
STEPS = 20_000                    # 공식 기본값 (50ep 기준). Phase 2에서 50K으로 증가 가능
BATCH_SIZE = 64
SCHEDULER_WARMUP_STEPS = 1_000    # SmolVLAConfig default
SCHEDULER_DECAY_STEPS = 30_000    # SmolVLAConfig default (> STEPS → auto-scale 정상 동작)
SCHEDULER_DECAY_LR = 2.5e-6      # SmolVLAConfig default
SAVE_FREQ = 5_000                 # 4 checkpoints (5K, 10K, 15K, 20K)
EVAL_FREQ = 10_000                # 2 eval points
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
    print("Starting fresh training (SmolVLA v6, 공식 레시피 재현, 20K steps)...")
    print(f"  Dataset: {DATASET_ROOT}/{DATASET_REPO}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  batch_size={BATCH_SIZE}, steps={STEPS:,}")
    print(f"  Scheduler: warmup={SCHEDULER_WARMUP_STEPS}, decay={SCHEDULER_DECAY_STEPS}, decay_lr={SCHEDULER_DECAY_LR}")
    print(f"  Checkpoints: every {SAVE_FREQ:,} steps ({STEPS // SAVE_FREQ} total)")
    print()
    print("  Estimated: ~50ep * 200fr / 64bs = 156 steps/epoch → ~128 epochs")
    print("  VRAM: batch_size=64 → ~9.85 GB (59% of 16.72 GB)")
    print("  NOTE: 공식 SmolVLA 레시피 정확 재현. v5(200K)는 10x 과다였음.")
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
