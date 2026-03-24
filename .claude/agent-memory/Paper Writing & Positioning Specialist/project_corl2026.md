---
name: project_corl2026
description: CoRL 2026 paper direction, timeline, and four contributions
type: project
---

Direction confirmed 2026-03-19: Data-Centric Multi-Object VLA (dropped Bimanual track).

**Why**: 70 days to CoRL 2026 (2026-05-28). Bimanual required 4-6 weeks for SO-101 hardware. Data-Centric builds on existing 74-episode pipeline immediately.

**Paper title**: "Data-Efficient VLA Adaptation on Consumer Hardware"

**Four contributions**:
1. OOD scaling laws: episodes(25-150) x quality(filtered/not) x steps(25K-200K) -> success rate
2. Data quality methodology: FK-based depth, gripper phase analysis, static frame detection
3. Multi-object transfer: 4 objects (sponge/cup/box/tool) x 50ep, cross-task transfer on RTX4090 + $130 robot
4. Self-improving loop: deploy → VLM judge → reuse successful rollouts → retrain (Seed2Scale-lite)

**Backup venue**: IROS 2026 LBR (deadline 2026-07-31)

**Timeline**:
- D-68~D-56: Multi-object data collection (cup/box/tool each 50ep)
- D-56~D-46: Scaling experiment matrix (40 runs)
- D-46~D-38: Deployment evaluation (20 trials per object per checkpoint)
- D-38~D-30: Self-improving loop implementation + experiments
- D-30~D-24: Multi-task joint training + transfer experiments
- D-24~D-10: Paper writing + figures
- D-10~D-0: Finalize + submit (5/28)

**How to apply**: When drafting sections, check against the four contributions. Do not add new contributions without confirming experiment data exists.
