---
name: Research gap verification protocol
description: Protocol for verifying research claims before stating them — from 2026-03-10 incident where 4/5 "gaps" were false
type: feedback
---

Never state "X does not exist" or "we are the first to do X" without running the full verification protocol.

**Why:** On 2026-03-10, five "research gaps" were proposed, four were wrong (RGBD-VLA, adaptive chunking,
self-improving VLA, deployment monitoring all existed). Root causes: confirmation bias, insufficient search,
term misinterpretation.

**How to apply:**
1. Before claiming a gap: search at minimum 3 different keyword combinations across 2 sources (arXiv + Google Scholar)
2. Search specifically for papers that would DISPROVE the gap (anti-confirmation search)
3. For any "first" claim: find 5 existing related papers and explicitly explain how we differ from each
4. Confidence levels: HIGH (>5 searches, no counter-evidence) / MEDIUM (3-5 searches, minor overlap found) / LOW (1-2 searches)
5. Never use absolute language ("no papers", "zero work"). Use qualified language ("to our knowledge", "within our search scope")
6. If an arXiv ID is cited, verify the paper's actual content matches the described finding

## Confirmed false claims (do not repeat)
- "RGBD-VLA does not exist" → DepthVLA, SpatialVLA, RD-VLA (8+ papers)
- "Adaptive chunking has no prior work" → Mixture of Horizons (2025-11)
- "Self-improving VLA has no prior work" → SOAR CoRL'24, SimpleVLA-RL ICLR'26, RISE, CRL-VLA (7+ papers)
- "Deployment monitoring VLA has no prior work" → DeeR-VLA NeurIPS'24, Diff-DAgger ICRA'25
- "RD-VLA's Depth means depth camera" → actually means network depth

## Verified true gaps (as of 2026-03-23)
- SmolVLA bimanual: 0 papers confirmed
- OOD embodiment adaptation scaling laws: no direct prior work found (MEDIUM confidence)
- Collection-time data quality metrics for VLA: no direct prior work (MEDIUM confidence)
- Consumer hardware self-improving loop without simulation: narrowly true (MEDIUM confidence)
