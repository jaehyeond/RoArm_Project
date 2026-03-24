---
name: AR/Projector/Laser x Robot x VLA — Exhaustive Landscape Survey
description: 2026-03-24 comprehensive literature search across 5 categories, 25 papers, AR/projector/laser intersection with robot manipulation and VLA/foundation models
type: project
---

Exhaustive landscape survey conducted 2026-03-24. Full report: `paper/AR_PROJECTOR_LASER_LANDSCAPE_SURVEY.md`

## Key Findings

**The intersection of AR/projector/laser with VLA training data collection is genuinely sparse.**

### Category Verdicts

| Category | Crowdedness | Notes |
|----------|------------|-------|
| Post-hoc visual augmentation | HIGH (5+ canonical papers) | Concept B — not a threat to our Concept A |
| Sim-based domain randomization | HIGH | Not relevant (requires sim) |
| AR for HRI/teleoperation | HIGH | Wrong goal — not data distribution coverage |
| AR for ML data collection (our claim) | SPARSE | No confirmed paper except AR2-D2 (unknown) |
| VLA + physical AR hardware | VERY SPARSE | Structural barrier: frozen encoder OOD problem |
| Projector + robot learning | NEAR EMPTY | No canonical paper found |
| Laser + VLA | EMPTY | Not realistic; laser pointer = 2010-era HRI |

### Closest Prior Work (ranked by threat)

1. **AR2-D2** — HIGHEST RISK. Unknown paper. Must verify immediately.
2. **GreenAug (CoRL 2024)** — Collection-time green screen backdrop. Only background diversity, not spatial coverage. LOW-MEDIUM threat.
3. **SOAR (CoRL 2024, 2404.11617)** — Same spirit (coverage enforcement) but different method (autonomous practice, no AR). LOW-MEDIUM.
4. **GenAug, Rosie, CACTI, RoboSplat, RoCoDA** — All post-hoc Concept B. LOW threat to Concept A.

### Structural Analysis: Why VLA + AR is Sparse

1. Frozen visual encoder problem: AR overlays in training images = OOD at deployment (SigLIP not trained on AR images)
2. Benchmark non-compatibility: LIBERO/CALVIN don't have AR setups
3. Train-test gap: projected light visible in training images won't be there at deployment

**Concept A elegantly avoids all of these**: AR shown to human on monitor, NOT projected into physical scene. Robot camera sees clean real-world images. No distribution shift.

### Mandatory Verification Before Any AR Claim

1. **AR2-D2 robot** — arXiv + Semantic Scholar (BLOCKING)
2. **"augmented reality demonstration collection robot learning"** — arXiv 2023-2026 (BLOCKING)
3. **HRI 2025 proceedings** — most likely venue for AR+robot paper (BLOCKING)
4. **IROS 2024/2025 + CoRL 2025 + ICRA 2025** — HIGH importance
5. **ISMAR 2024/2025** — AR-specific conference (MEDIUM importance)

**Why**: 2026-03-10 incident showed 4/5 "gaps" were false. Must verify before claiming novelty.
**How to apply**: Do not include any Concept A novelty claim in paper until all Priority 1 items are verified. If AR2-D2 does Concept A, fall back to data quality + scaling laws contributions.
