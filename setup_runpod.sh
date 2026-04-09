#!/bin/bash
# RunPod 초기화 스크립트 — SmolVLA v6 학습용
# Pod 내부(/workspace)에서 실행: bash setup_runpod.sh
#
# 전제조건:
#   - lerobot_dataset_v6/ 가 /workspace/에 업로드되어 있어야 함
#   - GPU pod (RTX 4090 $0.34/hr 권장. L4/A100도 OK — VRAM 24GB+면 충분)
#
# 사용법:
#   1. 로컬에서 데이터셋 전송: scp -P {PORT} -r lerobot_dataset_v6/ root@{POD_IP}:/workspace/
#   2. 이 스크립트 전송: scp -P {PORT} setup_runpod.sh root@{POD_IP}:/workspace/
#   3. Pod에서 실행: cd /workspace && bash setup_runpod.sh

set -e

echo "============================================"
echo "  SmolVLA v6 RunPod Setup"
echo "============================================"

# 1. GPU 확인
echo ""
echo "[1/5] GPU 확인..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available!')
    exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {name}')
print(f'  VRAM: {vram:.1f} GB')
if vram < 20:
    print('  WARNING: VRAM < 20GB. SmolVLA bs=64 needs ~10GB, but margin is small.')
print('  OK')
"

# 2. LeRobot + SmolVLA 설치
echo ""
echo "[2/5] LeRobot + SmolVLA 설치 (~5분)..."
pip install "lerobot[smolvla]==0.4.4" --quiet
LEROBOT_VER=$(pip show lerobot 2>/dev/null | grep Version | awk '{print $2}')
echo "  LeRobot version: $LEROBOT_VER"
if [ "$LEROBOT_VER" != "0.4.4" ]; then
    echo "  WARNING: Expected 0.4.4, got $LEROBOT_VER. 데이터 포맷 호환 깨질 수 있음!"
fi

# 3. 데이터셋 확인
echo ""
echo "[3/5] 데이터셋 확인..."
DATASET_DIR="/workspace/lerobot_dataset_v6"
if [ ! -d "$DATASET_DIR" ]; then
    echo "  ERROR: $DATASET_DIR not found!"
    echo "  로컬에서 전송 필요: scp -P {PORT} -r lerobot_dataset_v6/ root@{POD_IP}:/workspace/"
    exit 1
fi

for subdir in data meta videos; do
    if [ ! -d "$DATASET_DIR/$subdir" ]; then
        echo "  ERROR: $DATASET_DIR/$subdir not found!"
        exit 1
    fi
done

echo "  Dataset path: $DATASET_DIR"
echo "  Size: $(du -sh $DATASET_DIR | awk '{print $1}')"

python3 -c "
import json
with open('$DATASET_DIR/meta/info.json') as f:
    info = json.load(f)
print(f'  Episodes: {info[\"total_episodes\"]}')
print(f'  Frames: {info[\"total_frames\"]}')
print(f'  FPS: {info[\"fps\"]}')
print(f'  Format: {info[\"codebase_version\"]}')
"
echo "  OK"

# 4. smolvla_base 모델 사전 다운로드
echo ""
echo "[4/5] smolvla_base 모델 다운로드 (~865MB)..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('lerobot/smolvla_base')
print(f'  Model cached at: {path}')
print('  OK')
"

# 5. 테스트 실행 (10 steps)
echo ""
echo "[5/5] 테스트 실행 (10 steps)..."
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=roarm_m3_pick \
  --dataset.root=/workspace/lerobot_dataset_v6 \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=10 \
  --log_freq=1 \
  --save_freq=10000 \
  --output_dir=/workspace/outputs/smolvla_v6_test \
  --num_workers=4 \
  --policy.device=cuda \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=2.5e-6

echo ""
echo "============================================"
echo "  Setup 완료! 테스트 10 steps 성공."
echo ""
echo "  본 학습 시작:"
echo "    tmux new -s train"
echo "    bash train_runpod.sh"
echo "============================================"
