#!/bin/bash
# Offline eval of 12 OpenVLA-OFT v6 LoRA checkpoints on B200.
# Run inside the same env used for training.
set -e

[[ "$(whoami)" != "sogang_jhki" ]] && { echo "GUARD FAIL: user=$(whoami)"; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo "GUARD FAIL: host=$(hostname)"; exit 1; }
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo "GUARD FAIL: ROARM_B200_ROOT unset"; exit 1; }

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
export PYTHONUNBUFFERED=1

cd "$ROARM_B200_ROOT/code/openvla_oft_roarm"

# openvla-oft repo is on PYTHONPATH already via env.sh, but ensure prismatic
# package is importable from the cloned fork (D072).
export PYTHONPATH="$ROARM_B200_ROOT/code/openvla-oft:${PYTHONPATH:-}"

OUT_DIR="$ROARM_B200_ROOT/outputs/openvla_oft_v6_eval"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/openvla_oft_v6_eval_$(date +%Y%m%d_%H%M%S).json"

python -u eval_offline_v6.py \
  --base_model openvla/openvla-7b \
  --checkpoint_root "$ROARM_B200_ROOT/outputs/openvla_oft_v6_b200" \
  --dataset_repo_id roarm_v6_pick \
  --dataset_root "$ROARM_B200_ROOT/data/lerobot_dataset_v6" \
  --holdout_episodes 45 46 47 48 49 \
  --train_sanity_episodes 0 1 \
  --output "$OUT_JSON" \
  --dtype bfloat16 \
  2>&1 | tee /tmp/openvla_oft_v6_eval.out

echo
echo "[done] result json: $OUT_JSON"
