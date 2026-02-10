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
checkpoint_config = Path("E:/RoArm_Project/outputs/smolvla_official/checkpoints/005000/pretrained_model/train_config.json")

if checkpoint_config.exists():
    # Resume from step 5000 checkpoint
    print(f"Resuming from checkpoint: {checkpoint_config.parent}")
    sys.argv = [
        "lerobot-train",
        f"--config_path={checkpoint_config}",
        "--resume=true",
    ]
else:
    # Fresh start
    print("Starting fresh training...")
    sys.argv = [
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.pretrained_path=E:/RoArm_Project/models/smolvla_base",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=roarm_m3_pick",
        "--dataset.root=E:/RoArm_Project/lerobot_dataset_v3",
        "--batch_size=8",
        "--steps=20000",
        "--eval_freq=-1",
        "--save_freq=5000",
        "--log_freq=100",
        "--output_dir=outputs/smolvla_official",
        "--num_workers=0",
        "--policy.device=cuda",
    ]

from lerobot.scripts.lerobot_train import main

main()
