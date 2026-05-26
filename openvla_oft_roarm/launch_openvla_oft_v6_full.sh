#!/bin/bash
# OpenVLA-OFT 30K LoRA on v6 LeRobot — FULL finetune
set -e
[[ "$(whoami)" != "sogang_jhki" ]] && { echo "guard fail user"; exit 1; }
[[ "$(hostname)" != "JHPark-container" ]] && { echo "guard fail host"; exit 1; }
source /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/env.sh
[[ -z "$ROARM_B200_ROOT" ]] && { echo "guard fail root"; exit 1; }

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

cd "$ROARM_B200_ROOT/code/openvla_oft_roarm"

torchrun --standalone --nproc_per_node=1 train_roarm_v6.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$ROARM_B200_ROOT/data/lerobot_dataset_v6" \
  --dataset_name roarm_v6_pick \
  --run_root_dir "$ROARM_B200_ROOT/outputs/openvla_oft_v6_b200" \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_freq 2500 \
  --use_l1_regression True \
  --use_lora True \
  --lora_rank 32 \
  --merge_lora_during_training False \
  --num_images_in_input 1 \
  --use_proprio False \
  --image_aug False \
  --wandb_entity local \
  --wandb_project openvla-oft-roarm \
  --run_id_note v6_30k 2>&1 | tee /tmp/openvla_oft_v6_full.out
