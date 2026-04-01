---
name: Cloud GPU Strategy — RunPod (2026-04-01 updated)
description: RunPod A100 40GB for SmolVLA v6 training. Pipeline: local convert → scp → RunPod train → scp download. Scripts ready (setup_runpod.sh, train_runpod.sh).
type: project
---

## Context
User chose **RunPod** (A100 40GB) for external GPU training. 2026-04-01 결정.
Previously: VAST.ai/GCP 검토 (2026-03-24). 실제 사용은 RunPod이 최종.
SmolVLA(450M)은 로컬 RTX 4090에서도 가능하지만, 속도+GPU해방 목적으로 RunPod 선택.

## RunPod SmolVLA v6 파이프라인 (2026-04-01 구축)
- 로컬 변환: `convert_to_lerobot_v3.py` → `lerobot_dataset_v6/` (75MB)
- 업로드: `scp -P {PORT} -r lerobot_dataset_v6/ root@{POD_IP}:/workspace/`
- RunPod 셋업: `bash setup_runpod.sh` (pip install + 데이터확인 + 10step 테스트)
- 본 학습: `bash train_runpod.sh` (20K steps, bs=64, A100 ~1-2hr, ~$3-4)
- 다운로드: `scp -P {PORT} -r root@{POD_IP}:/workspace/outputs/smolvla_v6/checkpoints/020000/ outputs/smolvla_v6/checkpoints/020000/`
- 로컬 평가+배포: `eval_visual_grounding.py` + `deploy_smolvla.py`

**Why this matters:** Paper claim changes from "SmolVLA works on OOD robot"
to "method works across VLA architectures." Stronger generalizability.

## Cloud GPU Economics (VAST.ai, 2026 estimates)

| Task | GPU | Est. time | Cost |
|------|-----|-----------|------|
| pi0-fast compat test | A100 40GB | 0.5h | ~$1 |
| pi0-fast 200K steps | A100 40GB | 10h | $15-20 |
| pi0 data sweep (5 sizes) | A100 40GB | 50h | $75-100 |
| OpenVLA-OFT 100K steps | A100 80GB | 15h | $40-55 |
| Octo comparison | A100 40GB | 4h | $6-8 |
| Total budget estimate | | ~80h | ~$140-190 |

## Gating Tests (Do Before Committing Budget)

### Gate 1: pi0-fast compatibility (2 hours, <$2)
- Run: `lerobot-train --policy.path=lerobot/pi0fast --dataset.repo_id=<our_data>`
- Pass condition: training step runs without error, loss decreases
- If fails: need data adapter (1-2 days of code) before pi0 experiments
- **Critical**: pi0 expects specific normalization + camera config. Verify first.

### Gate 2: OpenVLA-OFT compatibility (2 hours, <$2)
- OpenVLA-OFT uses custom training script, not lerobot-train
- May need data format adapter
- Lower priority than pi0

## Deployment Strategy (Cloud vs Local)

- Training: cloud GPU (A100)
- **Inference for experiments**: cloud OR local RTX 4090
  - 4-chunk open-loop = 50 steps pre-computed = no real-time requirement
  - Network latency acceptable for experimental deployment
  - For final robot deployment: local inference still preferred
- SmolVLA inference stays local (108ms/step, RTX 4090 sufficient)
- pi0 inference: may need quantization for local RTX 4090 (3B vs 16GB)

## What Cloud Does NOT Fix

1. SigLIP frozen in SmolVLA — unfreeze requires training from scratch on massive data
2. Camera shift / existing 74ep data OOD problem — hardware issue
3. Data collection bottleneck — still ~3-4h human labor per 100ep
4. Isaac Lab sim-to-real visual domain gap — SigLIP cosine 0.6-0.8 unchanged
5. RoArm-M3 deployment latency constraint for future closed-loop work

## Strategic Impact on Paper

Before cloud: "SmolVLA works on $130 robot" — single data point
After cloud: "Method reduces data requirements across SmolVLA/pi0/OpenVLA-OFT" — generalizable claim

New contribution 3 (added): Method is model-agnostic.
Validated across 3 VLA sizes (450M, 3B, 7B) on same robot/task/data.

## Hardware Decisions Enabled by Cloud

3-camera setup (matching smolvla_base pretraining):
- smolvla_base pretrained with 3 cameras (camera1/2/3 on SO-100)
- Single-cam fine-tuning discards 2/3 of visual pretraining
- Azure Kinect ×3 + ZED Mini ×1 now available
- **Decision must happen BEFORE data recollection** (camera count change = all data invalid)
- Suggested setup: external Kinect + overhead Kinect + ZED Mini wrist
- ZED Mini wrist viability: check hw_wrist_camera_feasibility.py result

## How to Apply in Future Conversations

- When discussing paper scope: include multi-VLA comparison as core experiment
- When estimating costs: use $15-20 per pi0 training run as baseline
- When evaluating new ideas: ask "does this need cloud GPU or is RTX 4090 sufficient?"
- For pi0 experiments: always run Gate 1 first before committing
- "SmolVLA only" answer is no longer complete — add pi0 comparison context
