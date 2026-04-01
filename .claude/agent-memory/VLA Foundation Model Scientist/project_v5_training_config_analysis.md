---
name: V5 Training Config Critical Analysis vs Published Papers (2026-03-26)
description: 6-question comparison of our 200K-step training config vs SmolVLA/ACT/pi0/OpenVLA-OFT. Key: scheduler fix confirmed correct, 952 epochs MEDIUM risk, checkpoint selection protocol.
type: project
---

## Context
Training running NOW (2026-03-26). 136ep, 13,470 frames, batch=64, 200K steps, 5-zone sponge pick.
Full analysis: `/home/cgxr/Documents/Robotics/RoArm_Project/model_training_comparison.md`

**Why:** Verify our config against published numbers before training completes.

## Q1: Batch Size = 64 — APPROPRIATE
- LeRobot train.py default = 8 (verified, train.py line 55)
- SmolVLA paper pretraining used 64+ (large-scale only)
- ACT/OpenVLA-OFT use batch=8-16 for fine-tuning
- Our batch=64 is justified: VRAM headroom (59%), OOD stability, 5-zone diversity
- Steps/epoch = 13,470 / 64 = 210.5

## Q2: 200K Steps — JUSTIFIED for 5-zone OOD
- V3 success at 50K steps (244 epochs, 1 zone)
- SmolVLA paper recommends 150K-200K for OOD embodiments
- Per-zone training: 27 ep/zone < V3's 74ep → needs more steps per zone
- 952 epochs vs V3's 244 — higher but justified by lower per-zone data density

## Q3: Scheduler Fix — CRITICAL AND CONFIRMED CORRECT
- SmolVLA DEFAULT: decay_steps=30,000 → LR decays to 2.5e-6 by step 30K
- For 200K training: 85% of steps at near-minimum LR = almost no learning
- Our fix: decay_steps=200,000 = spans full training
- The auto-scale corruption in schedulers.py (lines 99-111): avoided by setting decay_steps=steps exactly
- **This is the most important config decision for 200K training**

## Q4: 952 Epochs — MEDIUM Overfitting Risk
- Published: ACT 100 epochs, Diffusion Policy ~200 epochs, V3 244 epochs (all success)
- Our 952 epochs is 3.9x V3's successful 244 epochs
- Mitigators: Beta(1.5,1.0) noise injection (strong regularizer), MEAN_STD normalization, 5-zone diversity
- Expected optimal checkpoint: 80K-120K steps (not final 200K)
- FAR_CENTER bias risk (39/136 = 29%) — zone imbalance amplifies over 952 epochs

## Q5: Frozen VLM — CORRECT for Kinematic OOD
- OOD is robot kinematics, NOT visual recognition
- SigLIP recognizes sponge (billion-scale pretraining)
- Unfreezing 350M VLM on 13,470 frames = catastrophic forgetting risk
- V3 empirical evidence: frozen VLM → 100% success
- Exception: multi-object Stage 2+ → consider OpenVLA-OFT parallel LoRA

## Q6: Checkpoint Selection Protocol (6 metrics)
1. Per-joint action std > 5°/3°/10° (base/wrist/gripper) — collapse detection
2. Gripper open coverage > 20% — prevents V1 failure (gripper never opened)
3. Z-score ratio < 5% outliers — prevents V1 Wrist_R polarity runaway
4. Zone-conditioned L2: max/min ratio < 2.0 — zone overfitting detection
5. Inter-sample trajectory diversity (3 noise seeds) — collapse detection

Priority checkpoints to test on robot: step 50K (V3 baseline), 80K (dense zones), 120K (sparse zones)

## How to Apply
- Before any deployment: run 5-metric offline evaluation on each checkpoint
- Do NOT just deploy final 200K checkpoint automatically
- If 120K+ shows zone L2 imbalance > 2x, use 80K checkpoint for OVERHEAD zone tests
