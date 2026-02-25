"""
공식 lerobot-train 파이프라인으로 SmolVLA 학습

smolvla_base 사전학습 모델 사용 (Action Expert + VLM 모두 사전학습됨)
공식 파이프라인이 정규화, LR 스케줄러, gradient clipping 등 자동 처리

SmolVLA SOTA settings (2026-02-24 research):
- batch_size=64 (공식 권장, RTX 4090 Laptop에서 9.85GB/16.7GB = 59%)
- steps=50000 (공식 quick-start 20K, paper real-world 200K, 우리는 중간)
- pretrained_path=lerobot/smolvla_base (필수! Action Expert 사전학습)
- save_freq=5000 (체크포인트 평가용)
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unbuffered output for piped output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# === V3 Dataset (74 episodes, sponge pick task) ===
DATASET_ROOT = "lerobot_dataset_v3"
DATASET_REPO = "roarm_m3_pick"
OUTPUT_DIR = "outputs/smolvla_v3_sponge"

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
    print("Starting fresh training (SmolVLA v3 sponge)...")
    print(f"  Dataset: {DATASET_ROOT}/{DATASET_REPO}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  batch_size=64, steps=50000")
    sys.argv = [
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.pretrained_path=lerobot/smolvla_base",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={DATASET_ROOT}",
        "--batch_size=64",              # 공식 SmolVLA 권장 (VRAM 검증 완료)
        "--steps=50000",                # 50K steps (bs=64 → effective 3.2M samples)
        "--eval_freq=10000",
        "--save_freq=5000",             # 5K 간격 체크포인트 (best 선택용)
        "--log_freq=100",
        f"--output_dir={OUTPUT_DIR}",
        "--num_workers=4",
        "--policy.device=cuda",
    ]

from lerobot.scripts.lerobot_train import main

main()
