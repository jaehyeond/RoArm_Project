---
name: Data collection automation research findings
description: Critical analysis of autonomous/augmented data collection approaches for reducing manual episode collection burden
type: project
---

## Research Question
Can LLM/VLM autonomously generate or augment robot demonstration data to reduce manual collection burden (currently 2-4 days for 200 episodes by hand)?

**Why:** Current bottleneck is the "200 episodes by hand" problem — labor intensive and the #1 practical barrier for small labs.

**How to apply:** Use these findings when advising on thesis chapter for data efficiency, and when pipeline-agent asks about data scaling.

## Approaches and Feasibility (verified 2026-03-23)

### 1. LLM-guided autonomous exploration
**Closest real paper: SOAR (arXiv:2404.11617, CoRL 2024)**
- WidowX robot, autonomous practice with success detection via CLIP image comparison
- Requires a partially working seed policy first (cannot bootstrap from zero)
- Consumer hardware? NOT demonstrated in SOAR
- Buildable with SmolVLA: YES — outer loop via GPT-4V API (~$10 for 200 episodes)
- SmolVLA limitation: No built-in success signal. Need external VLM (GPT-4V) for success detection.

**RT-2, SayCan, Code-as-Policies**: These are task EXECUTION papers, NOT data collection papers. Do not conflate.

### 2. Synthetic augmentation
**MimicGen (arXiv:2310.17596, CoRL/RSS 2024 Best Paper)**
- Sim only. Requires ground truth 3D poses. Cannot be directly used with real Azure Kinect without object pose estimation.
- Would need Isaac Lab integration (3+ months effort).

**GenAug (arXiv:2302.06671, ICRA 2023)**
- Appearance-only augmentation. Real robot (WidowX). +14% improvement.
- CRITICAL PROBLEM for VLA: appearance augmentation with wrong object position = wrong action labels.
- SmolVLA's frozen SigLIP: augmented images map to old action labels if object positions change.

**RoboSplat (arXiv:2504.13175, RSS 2025)**
- 3DGS-based, multi-view cameras required. Significant improvement (2-3x).
- Cannot directly apply: Azure Kinect is single-view.
- Depth-GS-Aug (our idea from RESEARCH_IDEAS.md): single-view RGB-D version — advisor expertise match.

**Image augmentation for SmolVLA: FUNDAMENTAL PROBLEM**
SmolVLA VLM is FROZEN. Augmented images must have SAME action labels to be valid training data. Only valid augmentations: appearance-only (color/lighting/background) with FIXED object position. Object position changes require action retargeting.

### 3. Human-in-the-loop assistance
**Most practical, available NOW: Leader-follower teleoperation** (user already has 2nd arm)
- Reduces effort 30-40% vs. hand-guiding, already configured in the pipeline.

**DAgger-style intervention reduction (2-3 months to implement)**
- Use SmolVLA denoising variance (across 10 denoising steps) as uncertainty signal
- Human only corrects high-uncertainty steps
- Gap: No paper uses SmolVLA flow-matching variance as active learning signal

## Verified Gaps (Cross-referenced with existing papers)

1. **Seed dataset threshold for autonomous practice on consumer hardware**
   - SOAR (CoRL 2024) did not characterize minimum human demos needed
   - For SmolVLA OOD embodiment, this threshold is completely unknown
   - Hypothesis: N=20-30 quality episodes may be sufficient seed

2. **Single-view RGB-D augmentation for VLA fine-tuning**
   - RoboSplat used multi-view. Azure Kinect is single-view.
   - Depth-guided single-view 3DGS augmentation for VLA: no paper found (search Nov 2024 - Mar 2026)
   - Advisor alignment: PERFECT (3DGS is advisor's expertise)

3. **Cross-object transfer scaling law**
   - How many extra episodes needed per new object, given existing trained policy?
   - Most tractable for CoRL 2026 (no new infrastructure needed)

## Recommended Action Path

**For CoRL 2026 (65 days):**
- 20-30 episodes × 3 new objects = cross-object transfer scaling law
- No new infrastructure. Buildable in 2-3 weeks.

**For thesis chapter (3-6 months):**
- Depth-GS-Aug: Azure Kinect depth → single-view 3DGS reconstruction → novel view synthesis → VLA training
- Validate in 2 weeks first: can Azure Kinect depth reconstruct manipulation scene well enough?

**NOT recommended:**
- MimicGen (sim-only, 3+ months Isaac Lab setup)
- Full autonomous SOAR-equivalent (3-4 months infrastructure)
- GenAug/image-only augmentation (action label misalignment problem)
