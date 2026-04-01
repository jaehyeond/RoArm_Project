#!/bin/bash
# SmolVLA v6 본 학습 — RunPod A100 40GB
# tmux 안에서 실행 권장: tmux new -s train && bash train_runpod.sh
#
# 바닐라 SmolVLA 레시피 (공식 SO100 pickplace 50ep 기준):
#   - batch_size=64, steps=20K, smolvla_base pretrained
#   - scheduler: warmup=1000, decay=30000, decay_lr=2.5e-6
#   - save: 5K마다 (5K, 10K, 15K, 20K)
#
# 예상 소요: A100 40GB에서 ~1-2시간
# 예상 비용: ~$3-4

set -e

echo "============================================"
echo "  SmolVLA v6 Training (20K steps, bs=64)"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "  Dataset: /workspace/lerobot_dataset_v6"
echo "  Output: /workspace/outputs/smolvla_v6"
echo "============================================"
echo ""
echo "  학습 시작... (Ctrl+B, D 로 tmux detach 가능)"
echo ""

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=roarm_m3_pick \
  --dataset.root=/workspace/lerobot_dataset_v6 \
  --batch_size=64 \
  --steps=20000 \
  --eval_freq=10000 \
  --save_freq=5000 \
  --log_freq=100 \
  --output_dir=/workspace/outputs/smolvla_v6 \
  --num_workers=4 \
  --policy.device=cuda \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=2.5e-6

echo ""
echo "============================================"
echo "  학습 완료!"
echo ""
echo "  체크포인트 확인:"
echo "    ls /workspace/outputs/smolvla_v6/checkpoints/"
echo ""
echo "  로컬로 다운로드 (로컬 터미널에서):"
echo "    scp -P {PORT} -r root@{POD_IP}:/workspace/outputs/smolvla_v6/checkpoints/020000/ outputs/smolvla_v6/checkpoints/020000/"
echo ""
echo "  비용 절약: Pod 중지하세요!"
echo "============================================"
