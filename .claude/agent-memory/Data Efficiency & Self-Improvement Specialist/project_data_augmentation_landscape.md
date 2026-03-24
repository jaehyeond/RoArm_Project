---
name: Data Augmentation for VLA — What Exists and What's Missing
description: Survey of augmentation methods for VLA/imitation learning as of 2026-03. MimicGen, RoCoDA, TGM-VLA all exist. Real gap is simulator-free trajectory augmentation.
type: project
---

# Data Augmentation for VLA Training

**Verified: 2026-03-23**

## What Exists (sorted by relevance)

### Trajectory-Level (requires simulator)
- **MimicGen** (2310.17596, CoRL 2023, NeurIPS 2024 oral): 10-200 source demos → 1000s via trajectory retargeting to randomized object positions. Works with Diffusion Policy + ACT. 10x-100x amplification. Requires simulator for replay.
- **RoCoDA** (2411.13031): Counterfactual data augmentation specifically targeting VLA training. Randomizes object positions, relabels actions. Requires scene reconstruction + simulator.
- **TGM-VLA** (2603.00615, March 2026): Task-guided mixup. 3x data efficiency. Does NOT require simulator — mixes existing demo data at feature level.

### Image-Level (no simulator needed)
- **GenAug** (2302.11550): Diffusion model augments demo backgrounds + object appearances. 3-5x generalization improvement.
- **Rosie** (2309.11386): Text-conditioned image augmentation using diffusion. Few real demos → diverse training images.
- **CACTI** (2212.05711): Camera + context augmentation for imitation learning.
- **RoboSplat** (2504.13175): 3DGS novel viewpoints from real captured scene. RSS 2025.

## SmolVLA Official Stance
Official SmolVLA documentation + our project CLAUDE.md: "No image augmentation — use real diversity instead." Image-level augmentation (color jitter, random crop) alone does NOT help VLA generalization (confirmed in SmolVLA ablations). This matches the literature finding that spatial/trajectory diversity matters more than pixel augmentation.

## The Genuine Gap for Us
**Trajectory-level augmentation without a simulator for unstructured grasping.**

MimicGen retargets trajectories to new object positions — but only if the task is geometrically parameterizable (same grasp, just translated). Sponge → cup transfer does NOT satisfy this (different grasp geometry, different approach angle).

**What we can actually implement:**
1. **TGM-VLA style mixup**: Mix existing demos at feature level — no simulator needed, could work
2. **Multi-zone collection as natural augmentation**: Our 5-zone (LEFT_FAR...RIGHT_FAR) strategy IS trajectory-level augmentation via real collection — not post-hoc but collection-time
3. **VLM success detection + quality filter**: SOAR-style, applied to autonomous rollouts as the model improves

## Key Insight for Paper
v1 (50ep, SHALLOW) → 0% vs v3 (74ep, all DEEP) → 100% is the strongest empirical argument we have. This shows that **collection-time quality > post-hoc augmentation**. Frame this as the paper's central finding.

**How to apply:** When designing augment_*.py scripts, prioritize: (1) TGM-VLA style mixup (implementable without simulator), (2) data quality filtering using existing FK-depth tools, (3) NOT MimicGen (requires simulator we can't reliably use with RoArm).
