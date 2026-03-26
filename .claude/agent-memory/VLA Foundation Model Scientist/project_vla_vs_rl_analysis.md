---
name: VLA vs RL Comparison Analysis for RoArm-M3 (2026-03-26)
description: Critical comparison of VLA vs RL vs Hybrid for robot manipulation, grounded in 2024-2026 papers. Decision: VLA is the right choice for RoArm-M3.
type: project
---

## Core Decision

VLA (SmolVLA fine-tuning) is the correct choice for RoArm-M3 multi-object grasping.
Pure RL is not viable given: uncalibrated URDF, consumer control latency, sim-to-real gap for contact manipulation.
Hybrid (reward-weighted BC) is a low-effort research contribution worth adding.

**Why:** RL sim-to-real for tabletop manipulation is NOT solved (unlike locomotion). SmolVLA already works at 100% on 1 object. The academic contribution is democratization (consumer hardware + no sim required).

**How to apply:** When RL is proposed as an alternative to VLA for this project, point to this analysis. Reserve RL use case for "reward-weighted BC augmentation" only.

## Object Understanding

- SigLIP (frozen, 768-dim, 4B training pairs): semantic object understanding from pretraining
- RL reward: no semantic understanding, implicit pattern memorization only
- Quantitative: OpenVLA 68.7% vs RT-2 FT 32.1% on zero-shot novel object tasks (CoRL 2024)
- SigLIP compression bottleneck: 64 tokens/camera after pixel-shuffle — spatial precision partly lost

## New Object Generalization

- VLA generalizes IF new object is semantically in SigLIP distribution (common household objects)
- pi0: 67% novel vs 89% training — strong but NOT zero-shot perfect
- OpenVLA-OFT: ~20-30% relative drop on novel objects vs training
- RL: must retrain per new object class — no semantic transfer
- OUR GAP: 74ep RoArm-M3, novel object test NOT yet done

## Data Efficiency

| System | Demos | Result |
|--------|-------|--------|
| SmolVLA RoArm-M3 | 74ep | 100% (1 object) |
| RL PPO Isaac | 0 demos + 50M sim steps | unverified on RoArm-M3 |
| ACT baseline | 50 demos | ~80% |

Hidden cost of RL: requires calibrated URDF + sim physics — neither done for RoArm-M3.

## Sim-to-Real Gap Status

- Locomotion: LARGELY SOLVED (ANYmal, quadruped, bipedal via DR)
- Manipulation tabletop: NOT SOLVED — contact physics precision, <3mm error matters
- VLA sim images: BROKEN (SigLIP cosine 0.6-0.8 for rasterized sim) — 0 papers verified
- 3DGS alternative: cosine 0.1-0.2, potential research gap (no verified paper as of Aug 2025)

## Hybrid VLA+RL Papers (Ranked by SmolVLA Applicability)

| Paper | Compatibility | Effort |
|-------|--------------|--------|
| SimpleVLA-RL / Reward-weighted BC | HIGH | 2 days |
| Beyond Imitation (ICLR 2025 Oral) | MEDIUM | 30 days |
| HIL-SERL (RSS 2024) | LOW (SAC, not VLA) | 20 days |
| VLA-RFT / World model RL | LOW (token-level, not flow-matching) | 90 days |

## 4-Object Multi-Task Capacity

- SigLIP separability for 4 objects: HIGH (distinct semantic features in pretrained space)
- Action Expert capacity (100M): MEDIUM — sufficient for 4 simple grasp trajectories
- Required: 200ep (50 per object) + 200K steps
- Basis: OOD robot needs 150+ ep baseline × 1.3x multi-task multiplier

## Recommendation for pipeline-agent

Training config for multi-object grasping:
- episodes: 200 total (50 per object × 4 objects)
- steps: 200_000
- batch_size: 64 (verified fits in 16GB VRAM)
- pretrained: lerobot/smolvla_base (mandatory)
- language tasks: "Pick up the [color] [object]" — distinct per object
- evaluation: per-object success rate + language-conditioned selectivity test
