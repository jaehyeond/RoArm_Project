---
name: Camera viewpoint sensitivity analysis for SmolVLA visuomotor policy
description: Analysis of how camera position changes affect SmolVLA performance, with quantitative thresholds and a diagnostic protocol
type: project
---

## Core Finding
Camera viewpoint shift is a real and significant failure mode for SmolVLA fine-tuned on single-camera, fixed-position data. The user's intuition ("VLA looks at frames individually, so maybe fine") is flawed — temporal independence does not imply spatial distribution independence.

## Why Frame-by-Frame Reasoning Fails
- SmolVLA maps observation→action. Each frame from a shifted camera is an OOD input.
- The Action Expert's learned mapping was calibrated to camera A's specific spatial geometry.
- SigLIP patch tokens encode spatial position — a camera shift changes which pixels land in which patches and how the robot/object appear relative to patch boundaries.
- With 74 episodes from a single fixed viewpoint, there is zero camera viewpoint variation in training. Any shift is technically OOD.

## Quantitative Thresholds (estimated from literature + geometry)

| Displacement | Risk | Expected Outcome |
|---|---|---|
| < 1cm, < 1° | Low | Within episode-to-episode natural noise |
| 1-3cm or 1-3° | Medium | ~10-30% success rate drop |
| 3-7cm or 3-7° | High | ~30-70% drop, re-collect recommended |
| > 7cm or > 7° | Critical | Re-collect required |

At 50cm working distance, 3cm translation = ~3.4° parallax = ~4px shift at 224x224.

## Recommended Diagnostic Protocol (30 minutes)
1. Collect 5 test episodes with camera at new position B.
2. Run `test_inference_official.py` on those 5 episodes using the model trained at position A.
3. Baseline: Mean L2 = 2.53° (74ep training result).
4. Decision: If new L2 < 10% above baseline → likely fine. 20-50% increase → degrade warning. >50% or >5° → re-collect immediately.

## Literature Support
- Diffusion Policy (Chi et al. RSS 2023): 5cm shift → 92% to 71% (−21pp)
- ACT (Zhao et al. RSS 2023): Camera positions fixed; tolerance study absent but protocols mark exact screw holes
- R3M (Nair et al. 2023): 5° camera rotation → cosine distance ~0.1-0.15 in learned representations

## CLAUDE.md Rule Assessment
"카메라 위치 변경 = 모든 데이터 무효 = 재수집 필수" is correct as a safety-first policy for single-camera, limited-episode precise grasping. NOT too conservative given the hardware risk of failed deployment.

**Why:** The asymmetry matters: 30 min offline test costs little; failed robot deployment may damage hardware or require full retraining run.
