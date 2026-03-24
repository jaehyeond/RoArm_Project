---
name: CoRL 2026 research direction
description: Primary paper positioning and contribution structure for CoRL 2026 submission (deadline 2026-05-28)
type: project
---

Strongest CoRL 2026 angle: "Data-Efficient OOD Adaptation of VLAs — Method Validated Across Model Scales"

Core claim: AR-guided collection + real-time quality filtering reduces demo requirements for
VLA adaptation to OOD embodiment hardware. Validated on SmolVLA (450M), pi0 (3B), and
OpenVLA-OFT (7B) on same robot/task/data — method is model-agnostic.

**UPDATE 2026-03-24**: Cloud GPU (VAST.ai/GCP) now available. No longer limited to SmolVLA.
pi0-fast (3B) full fine-tune on A100 40GB: feasible, ~$15-20/run.
OpenVLA-OFT (7B) on A100 80GB: feasible, ~$40-55/run.
This changes the paper from single-VLA study to multi-VLA comparison.

**Why:** "Accessible Physical AI" (2512.11921) already claimed "VLA on cheap hardware."
We must differentiate on the *data-efficiency characterization* angle, not the deployment fact.
Multi-VLA comparison makes method generalizable claim stronger.

**How to apply:** Frame every experiment as answering "how much data, of what quality, for how many steps?"
Avoid framing as "we built a system" — frame as "we characterized a phenomenon."
SmolVLA is now "efficient baseline", pi0 is "rich pretraining upper bound."

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

## Minimum experiments for CoRL (UPDATED 2026-03-24 for cloud GPU)
1. Scaling curve: 25ep / 50ep / 74ep / 100ep — on SmolVLA AND pi0-fast
2. Quality ablation: FK-depth coverage vs success rate correlation (SmolVLA)
3. Step ablation: 25K / 50K / 100K checkpoints on real robot (SmolVLA)
4. Chunk ablation: 1-step vs 4-chunk vs 8-chunk (no retraining)
5. SigLIP object feature separability analysis (2 hours, no robot needed)
6. Multi-model comparison table: SmolVLA vs pi0-fast vs OpenVLA-OFT on same 74ep data

## NEW: Gating test (do before committing cloud budget)
- pi0-fast lerobot-train compatibility test: 2 hours, A100 spot instance (<$2)
  Command: lerobot-train with policy.path=lerobot/pi0fast + our dataset
  If fails: need data adapter code (1-2 days before pi0 experiments)

## Comparison baseline
SmolVLA is "efficient baseline" (full fine-tune without LoRA, 450M).
pi0-fast is "strong baseline" (rich pretraining, 3B, cloud only).
OpenVLA-OFT is "SOTA VLA fine-tuning method baseline" (7B, LoRA).
If all three improve with AR-guided+Oracle, method is model-agnostic → strong paper claim.

## Hardware additions (2026-03-24)
- RoArm-M3 ×3 (3 setups for parallel data collection or multi-task)
- Azure Kinect ×3 (enables 3-camera setup matching smolvla_base pretraining)
- ZED Mini ×1 (wrist-mountable stereo RGBD)
- 3-camera decision must be made BEFORE data recollection (changes everything)
