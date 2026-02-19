"""
공식 lerobot-train 파이프라인으로 SmolVLA 학습

smolvla_base 사전학습 모델 사용 (Action Expert + VLM 모두 사전학습됨)
공식 파이프라인이 정규화, LR 스케줄러, gradient clipping 등 자동 처리
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Unbuffered output for piped output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Resume or fresh start
# last 체크포인트가 있으면 거기서 재개, 없으면 fresh start
last_ckpt_file = Path("outputs/smolvla_v2_cleaned/checkpoints/last/pretrained_model/train_config.json")

if last_ckpt_file.exists():
    print(f"Resuming from checkpoint: {last_ckpt_file.parent}")
    sys.argv = [
        "lerobot-train",
        f"--config_path={last_ckpt_file}",
        "--resume=true",
    ]
else:
    # Fresh start
    print("Starting fresh training...")
    sys.argv = [
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.pretrained_path=lerobot/smolvla_base",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=roarm_m3_pick",
        "--dataset.root=lerobot_dataset_v4",
        "--batch_size=8",
        "--steps=50000",
        "--eval_freq=10000",
        "--save_freq=5000",
        "--log_freq=100",
        "--output_dir=outputs/smolvla_v2_cleaned",
        "--num_workers=4",
        "--policy.device=cuda",
    ]

from lerobot.scripts.lerobot_train import main

main()
