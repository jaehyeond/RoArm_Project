---
name: CoRL 2026 research direction
description: Primary paper positioning and contribution structure for CoRL 2026 submission (deadline 2026-05-28)
type: project
---

Strongest CoRL 2026 angle: "Data-Efficient OOD Adaptation of Small VLAs"

Core claim: systematic characterization of episode count x data quality x training steps trade-offs
for fine-tuning a pretrained VLA (SmolVLA 450M) to out-of-distribution robot hardware ($130 arm, no sim data).

**Why:** "Accessible Physical AI" (2512.11921) already claimed "VLA on cheap hardware."
We must differentiate on the *data-efficiency characterization* angle, not the deployment fact.

**How to apply:** Frame every experiment as answering "how much data, of what quality, for how many steps?"
Avoid framing as "we built a system" — frame as "we characterized a phenomenon."

## What is NOT novel (do not claim)
- "First VLA on consumer hardware" — 2512.11921 is prior art
- "RGBD-VLA" or "depth fusion" — DepthVLA, SpatialVLA, etc. exist
- "Self-improving VLA" — 7+ papers exist (SOAR, SimpleVLA-RL, RISE, CRL-VLA, etc.)
- "New VLA architecture" — insufficient pretraining data to validate
- "3DGS + robot" for CoRL — too risky in 65 days

## Verified gaps (MEDIUM-HIGH confidence)
1. OOD embodiment scaling laws (episode count x quality x steps) — Data Scaling Laws (ICLR 2025) studies environment diversity, NOT embodiment OOD
2. Collection-time data quality metrics (FK-depth, gripper phase, static frame ratio) — most quality work is post-hoc
3. Open-loop vs closed-loop chunk trade-off — 4-chunk open-loop beats 1-step closed-loop (empirical finding to characterize)

## Claims that need pre-submission verification
- "First OOD embodiment scaling laws" — search arXiv Nov 2025 - May 2026
- "FK-depth metric predicts success" — need quantitative correlation, not just qualitative
- "SigLIP separates novel objects" — need feature extraction experiment

## Timeline
- CoRL 2026: 2026-05-28 (65 days from 2026-03-23)
- Master's proposal: 2026-08
- Master's thesis: 2026-12

## Minimum experiments for CoRL
1. Scaling curve: 25ep / 50ep / 74ep / 100ep (same task, fixed quality)
2. Quality ablation: FK-depth coverage vs success rate correlation
3. Step ablation: 25K / 50K / 100K checkpoints on real robot
4. Chunk ablation: 1-step vs 4-chunk vs 8-chunk (no retraining)
5. SigLIP object feature separability analysis (2 hours, no robot needed)

## Comparison baseline
Must include OpenVLA-OFT comparison or written justification.
CoRL reviewers will ask "why SmolVLA not OpenVLA?"
Answer: SmolVLA is the only open VLA trainable full-FT on consumer GPU without LoRA/quantization.
