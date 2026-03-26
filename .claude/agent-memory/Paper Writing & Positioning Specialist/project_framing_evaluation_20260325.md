---
name: CoRL 2026 Framing Evaluation (2026-03-25)
description: Brutal evaluation of 4 research framings (A-D). Recommended unified framing + 3 go/no-go gate experiments.
type: project
---

## Framing Verdicts

| Framing | Score | Verdict |
|---------|-------|---------|
| A: Autonomous Competence Expansion | 8/10 | STRONG — run VLM judge test first |
| B: Fleet Self-Improvement | 3/10 | KILL — no scientific question |
| C: Data-Efficient Adaptation | 6/10 | CONDITIONAL — control mode confound must be ablated |
| D: Environmental Drift | 4/10 | KILL as standalone — use as motivation example in A |

## Critical Risk in Framing C
v1→v3 improvement was primarily caused by control mode (open-loop→closed-loop) and batch_size (64), NOT data quality per main memory. If true, the "smart demo collection" narrative is undermined. Must run ablation: same data, different control mode. If closed-loop alone explains 100%, the data quality contribution disappears.

## Recommended Unified Framing
**Problem**: A VLA policy trained in small labs has unknown competence boundaries, fails silently outside them, and requires painful manual re-collection to expand them.

Three user experiences → three contributions:
1. v1 failure (bad data detected too late) → data quality oracle
2. v3 zone-limited success (unknown failure zones) → spatial competence map
3. Camera bump (drift erases competence invisibly) → drift detector

Together: "competence boundary problem in small-lab VLA deployment"

## Technically Novel Core
SmolVLA flow matching generates 10 denoising steps → intermediate sample variance = uncertainty proxy → use as competence signal. Per research memory: "VLA denoising variance as active learning signal = 없음 (HIGH confidence)." This is the strongest technically original piece.

## 3 Go/No-Go Gate Experiments (run before any writing commitment)
1. VLM judge accuracy — Qwen2.5-VL 3B on 74 episodes — must be >85% or self-improvement loop is unreliable (half day)
2. Control mode ablation — same data, open-loop vs closed-loop — resolves v1→v3 causality question (one day)
3. 5-zone competence map — deploy in 5 zones, record success/failure per zone — quantifies the problem (half day)

## Competitor Positioning
- SOAR (CoRL 2024): WidowX (~$2K), CLIP, >30-40% seed needed. Ours: $130 arm, local 3B VLM, seed at 100% in zone 1
- Data Scaling Laws (ICLR 2025): quantity axis. Ours: quality + spatial coverage axis
- arXiv:2512.11921 "Towards Accessible Physical AI": consumer hardware already claimed. Must differentiate on DATA efficiency, not hardware cost

**Why**: Framing determines reviewer reception. Wrong framing → reject even with good results.
**How to apply**: Do not start writing Introduction until gate experiments 1-3 are complete. Framing A unified is the working hypothesis. Update this memory once gate results are in.
