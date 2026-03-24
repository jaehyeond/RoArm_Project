---
name: Reaching/Grasping Decomposition Phase-Selective Augmentation Novelty Check
description: 2026-03-24 novelty verification — decompose demos into reaching/grasping, augment reaching only. MimicGen is the critical prior work. SigLIP asymmetry is the strongest novel angle.
type: project
---

## Core Idea Checked
Decompose manipulation demos into reaching phase (approach to object) and grasping phase (contact + grasp + lift). Augment ONLY the reaching phase with synthetic/virtual trajectories. Keep real grasping data intact.

## Verdict: [PARTIALLY FOUND]

**MimicGen (NeurIPS 2024 oral, arXiv:2310.17407)** is the closest prior work.
- It decomposes demonstrations into subtask segments
- Approach/transition segments are synthetically generated (IK-based)
- Manipulation/grasping segments are kept from real demonstrations
- REQUIRES physics simulator (IsaacGym/MuJoCo) for feasibility checking
- REQUIRES exact 3D object poses for each target configuration
- Evaluated in simulation only — no real robot transfer demonstrated

## What IS novel compared to MimicGen

1. **Sim-free**: MimicGen requires simulator. Proposed idea targets real-robot-only augmentation.
2. **No 3D pose ground truth**: MimicGen needs exact poses. Visual/FK-based detection is accessible.
3. **VLA + frozen SigLIP encoder**: MimicGen is for Diffusion Policy/BC. Domain gap constraint in frozen encoders is NOT studied.
4. **SigLIP asymmetry claim** (strongest): Reaching phase is more SigLIP-tolerant than grasping phase. This has NOT been published. Frame this as the central scientific contribution.

## Recommended Claim Framing
"We show SigLIP domain gap is asymmetric across manipulation phases (reaching: cosine dist ≤ 0.15, grasping: ≥ 0.45). This motivates phase-selective augmentation for frozen-encoder VLAs — a contribution distinct from MimicGen which targets simulation-only diffusion policies."

## DO NOT CLAIM
- "First to decompose reaching vs grasping" — multiple prior works
- "MimicGen is unrelated" — must cite and differentiate
- "First to keep grasping real" — MimicGen does this in sim

## Full Report
`paper/REACHING_GRASPING_DECOMPOSITION_NOVELTY_REPORT.md`

**Why**: Prevent 2026-03-10-style overclaim. MimicGen match found on search term 7. The idea has a real novel angle but it is narrow and specific — must be framed precisely.
**How to apply**: If this idea appears in any paper draft, the related work MUST include MimicGen with explicit differentiation. The SigLIP asymmetry framing is the safest and most defensible angle.
