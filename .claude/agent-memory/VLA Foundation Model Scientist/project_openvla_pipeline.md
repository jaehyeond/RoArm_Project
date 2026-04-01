---
name: OpenVLA and OpenVLA-OFT pipeline analysis
description: Practical implementation details for OpenVLA (7B) and OpenVLA-OFT fine-tuning — data format, GPU requirements, custom robot adaptation, LeRobot v3 conversion path
type: project
---

## OpenVLA Vanilla (7B) — Key Facts

Architecture: DINOv2 + SigLIP (224px, fused) + Llama-2 7B
Action rep: Discrete, 256 uniform bins per dim in [-1, 1] → mapped to last 256 vocab tokens
Action chunking: NONE — single-step autoregressive prediction
Control freq: 5-10 Hz recommended (not 30 Hz)
Resolution: ~1.4° at 180° range (2/256 bins)
Pretraining: OXE 970K trajectories, diverse embodiments (much broader than SmolVLA SO-100)
Paper: arXiv 2406.09246

**Why:** Useful mostly as historical baseline. OFT supersedes it entirely.

## OpenVLA-OFT — Key Facts

Paper: arXiv 2502.19645 (RSS 2025)
Same backbone as vanilla (DINOv2+SigLIP+Llama-2)
Key changes vs vanilla:
  1. Continuous L1 regression head (MLPResNet, replaces discrete tokens)
  2. Parallel action chunking: 8-step (LIBERO), 25-step (ALOHA), 5-step (BridgeV2)
  3. Multi-image: 1-3 cameras supported
  4. FiLM language conditioning (OFT+): critical for language-variation tasks (33% → 60%+)
  5. MultiStepLR: 10x decay at num_steps_before_decay, better convergence
Speed: 25-50x faster than vanilla (109.7 Hz reported for 8-step chunks)
LIBERO results: 95.3% avg (OFT) vs 76.5% (vanilla) vs 94.2% (pi0)

Action normalization per robot type:
  - BOUNDS_Q99 (LIBERO, BridgeV2): relative EEF poses → clip outliers OK
  - BOUNDS (ALOHA, RoArm-M3): absolute joint angles → NO clipping (would block extreme positions)

**Why:** For RoArm-M3, use BOUNDS normalization always.

## Data Format — BOTH OpenVLA and OFT

Required format: RLDS (TensorFlow Datasets / TFRecord)
NOT supported natively: LeRobot v3 (parquet + mp4), HDF5, raw numpy

RLDS schema per step:
  observation: {image: uint8 (H,W,3), wrist_image: optional, state: float32[N]}
  action: float32[N], normalized to [-1, 1]
  language_instruction: string
  language_embedding: float32[512] (Universal Sentence Encoder, from TF Hub)
  is_first, is_last, is_terminal: bool scalars
  reward, discount: float32 scalars

RLDS conversion from LeRobot v3: 2-4 days engineering
  Steps: decode mp4 → extract frames → load parquet → compute stats → USE embeddings
         → write via rlds_dataset_builder → tfds build → register in 4 files (configs/transforms/mixtures/constants)

Custom PyTorch Dataset path: documented in comments but NOT officially tested. Risky.

## GPU Requirements

OpenVLA-OFT LoRA fine-tune (r=32, bfloat16):
  batch=8: ~62GB → A100 80GB or H100
  batch=1: ~25GB → A100 40GB feasible
  RTX 4090 24GB: ~25-27GB needed → NOT feasible without QLoRA
  RTX 4090 Laptop 16.7GB: NOT FEASIBLE (even batch=1 too large)
  QLoRA for OFT: not officially supported (issue #134 open, no answer)

OpenVLA-OFT inference:
  bfloat16: ~15-16GB → RTX 4090 24GB YES, laptop 16.7GB MARGINAL
  4-bit quant: ~9-10GB → laptop YES

Cloud recommendation:
  Training: A100 40GB on VAST.ai (~$1.5-2/hr) → batch_size=2-4 → ~$20-30 per 100K step run
  Inference: deploy on A100 or serve via REST API

## Custom Robot Reports

WidowX 250s (6-DOF, Trossen): issue #312 — fine-tuned 5K steps, model produced tiny constant actions
  → Root cause: unnorm_key mismatch + insufficient steps
7-DOF arm (issue #149, OFT): successfully adapted constants.py + ActionEncoding.JOINT_POS
  → Required: adding JOINT_POS encoding to materialize.py validation list
UR3: issue #292 — feasible, use single primary camera
PyBullet sim (210 demos, 5Hz, issue #304): fine-tuned but 0% success — sim image OOD for DINOv2/SigLIP

## RoArm-M3 Adaptation Notes

constants.py entry needed:
  ROARM_CONSTANTS = {"NUM_ACTIONS_CHUNK": 8, "ACTION_DIM": 6, "PROPRIO_DIM": 6, "ACTION_PROPRIO_NORMALIZATION_TYPE": BOUNDS}
  → 8-step chunks at 30fps = 267ms
  → Or 15-step chunks for 500ms (smoother)

Action encoding: ActionEncoding.JOINT_POS (absolute angles, not delta)
Normalization: BOUNDS using hardware joint limits (not Q99)

**How to apply:** When planning OFT fine-tuning on RoArm-M3, budget 2-4 days for RLDS conversion,
use A100 40GB cloud GPU for training, plan for constants.py + materialize.py modifications.
Budget: ~$20-30/run on VAST.ai.

## SmolVLA vs OpenVLA Comparison (for CoRL 2026 paper)

SmolVLA advantage: native LeRobot v3 format, consumer GPU full FT, no LoRA confound
OFT advantage: 25-50x faster inference, continuous actions, multi-image, FiLM language grounding
Key CoRL comparison: same 136ep dataset on SmolVLA (local) vs OFT (cloud) → method-agnostic claim

Detail doc: claudedocs/OPENVLA_PIPELINE_ANALYSIS.md
