---
name: VLA demo requirements — novel robot fine-tuning analysis
description: Per-model demo counts for novel (OOD) robot fine-tuning, with honest caveats. Addresses the "why still 100+ demos?" question.
type: project
---

## Two-Component Insight (key framing)

Component 1: Object/scene understanding — solved by VL pretraining (SigLIP, OXE).
Component 2: Motor policy binding — robot-specific, camera-specific, workspace-specific. NOT in internet data. Demos teach this.

The 100+ demos are NOT teaching the model what objects look like. They are teaching what joint angles
to execute in THIS robot's coordinate frame, from THIS camera's viewpoint, in THIS workspace.

## Per-Model Demo Counts (novel robot, as of 2026-03-26)

| Model | Demos | Confidence | Source |
|-------|-------|------------|--------|
| SmolVLA (450M) | 100-200 | MEDIUM-HIGH | RoArm-M3 empirical: 50ep fail, 74ep = 100% (1 task) |
| Octo (93M) | ~100 | HIGH | Octo paper, fine-tuned 72% avg |
| OpenVLA (7B) | ~100 | HIGH | GitHub docs |
| OpenVLA-OFT (7B) | 20-300 task-dep | HIGH | RSS 2025 ALOHA results |
| pi0 (3B) | ~100+ (estimate) | MEDIUM | "hours of data" per paper, not precise |
| GR00T N1.5 (3B) | 20-40 | MEDIUM | On own hardware platforms, not strictly novel |
| RT-2 (55B) | N/A (closed) | N/A | Not fine-tunable |

## VL Pretraining Benefit (honest)

Pretrained vs scratch (SmolVLA, 74ep, RoArm-M3): 78.3% vs 51.7%.
= +26.6% success rate improvement
= saves ~20-50 demos at the margin
NOT: reduction from 100 demos to 10 demos. Still need 100+.

Why benefit is modest:
1. VLM backbone FROZEN in SmolVLA — 350M params fixed, only 100M Action Expert trained
2. Action representation is robot-specific — SO-100 biases must be overwritten for RoArm-M3
3. Spatial grounding (joint-to-Cartesian mapping) not in internet data regardless of model size
4. SmolVLA pretrained on SO-100 ONLY — maximum OOD case for RoArm-M3

## Open Research Question

Can FK-depth coverage (collection-time quality metric) reduce the demo count needed?
"Quality over quantity" hypothesis: 75ep with full FK-depth coverage > 150ep without.
This is the research gap this project is positioned to answer.
Not yet proven — need quantitative correlation experiment.

## 4-Object Multi-Task Estimate (RoArm-M3)

Estimate: 200 total episodes (50 per object × 4), 200K steps.
Confidence: MEDIUM (extrapolation, not measured).
Basis: OOD baseline 150ep × 1.3x multi-task multiplier.
