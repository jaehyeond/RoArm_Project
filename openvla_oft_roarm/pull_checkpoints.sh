#!/bin/bash
# Pull latest checkpoints from B200 openvla_oft_v6_b200 outputs to Lenovo
# Run periodically while training.
set -e

REMOTE_BASE="JHPark:/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/outputs/openvla_oft_v6_b200/"
LOCAL_BASE="/home/cgxr/Documents/Robotics/RoArm_Project/openvla_oft_b200_pulls/"

mkdir -p "$LOCAL_BASE"

echo "=== pulling $REMOTE_BASE -> $LOCAL_BASE ==="
rsync -av --partial --info=progress2 \
  --include="*/" \
  --include="*.pt" \
  --include="*.json" \
  --include="*.safetensors" \
  --include="adapter_*" \
  --include="dataset_statistics.json" \
  --include="*--lora_adapter--*" \
  --include="run_id*" \
  --exclude="optimizer*" \
  --exclude="*.tmp" \
  "$REMOTE_BASE" "$LOCAL_BASE"

echo ""
echo "=== local pull state ==="
du -sh "$LOCAL_BASE"
find "$LOCAL_BASE" -maxdepth 4 -type d 2>/dev/null | head -30
