---
name: sim2real_research_direction
description: Sim-to-real and digital twin research direction for RoArm M3 + SmolVLA CoRL 2026
type: project
---

The core research question investigated (2026-03-23): "Can a Unity/3DGS digital twin replace manual demonstrations for VLA training?"

Answer: FEASIBLE WITH CAVEATS. Not plug-and-play.

**Why:** Plain sim renders will NOT fool SigLIP. 3DGS approach (scan real scene with Azure Kinect) is the best path. Minimum ~20-50 real demos still needed for fine-tuning.

**How to apply:** Frame paper as GS-XR-Demo: Azure Kinect → 3DGS → Unity XR editing → demo generation → SmolVLA training. Student's Unity expertise is the key differentiator.

## Key Quantified Gaps

### Physics Gaps (Isaac Lab vs. Real)
- Actuator lag: 20-50ms real vs <1ms sim (SEVERITY: HIGH)
- Stiction dead-band: 1-3° real vs 0 sim (SEVERITY: HIGH, NOT domain-randomizable)
- Joint backlash: 1-2° per reversal (SEVERITY: MEDIUM, DR-able)
- Gravity sag: 2-5° at full extension (SEVERITY: MEDIUM, DR-able)
- Contact dynamics: CRITICAL — sponge deformation not in Isaac Lab default

### SigLIP Visual Gap (SmolVLA's vision encoder)
- Isaac Lab rasterizer: ~0.6-0.8 cosine distance (will NOT transfer)
- Isaac Lab RTX renderer: ~0.3-0.5 cosine distance (~50-60% transfer)
- 3DGS from real scene: ~0.1-0.2 cosine distance (~80-85% transfer)

### Stats.json Incompatibility
- Real shoulder mean=30.2°, sim estimated ~20°
- Real elbow mean=58.9°, sim estimated ~80°
- ALWAYS retrain from smolvla_base when mixing sim + real data
- NEVER resume from real-only checkpoint with sim-mixed data

## Recommended Thesis Direction
GS-XR-Demo: Azure Kinect RGBD → few-shot 3DGS → Unity XR interactive editing → demonstration generation → SmolVLA training.
Timeline: 10-14 weeks total. Feasible for CoRL 2026 (5/28 deadline).

## Key Papers
- SplatSim (arXiv:2409.10161): GS→82% of real, plain raster→45%
- RoboSplat (arXiv:2504.13175): 25 real + GS aug > 100 real demos
- Real2Render2Real (arXiv:2505.09601): scan real → render variations
- RoboTwin (arXiv:2504.13059): generative digital twins, CVPR 2025 Highlight
- TRANSIC (arXiv:2405.14523): 72% transfer sim→real manipulation
