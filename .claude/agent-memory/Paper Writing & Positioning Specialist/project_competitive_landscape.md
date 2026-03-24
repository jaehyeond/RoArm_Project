---
name: project_competitive_landscape
description: Key competitors, positioning, and overclaim risks for CoRL 2026 paper
type: project
---

Competitive landscape as of 2026-03-24. Full analysis: paper/GEMINI_ROBOTICS_LANDSCAPE_ANALYSIS.md

**Critical competitor**: "Towards Accessible Physical AI" (arXiv:2512.11921) — consumer-grade VLA on RTX 4060 with LoRA/quantization on a 3.1B model. We CANNOT claim "first consumer hardware VLA." Differentiate on DATA efficiency and quality, not model compression.

**Data scaling**: Data Scaling Laws (ICLR 2025, Hu et al.) — environment diversity scaling. We scale episode count + quality for OOD embodiment. Different axis.

**Self-improving VLA**: 7+ papers exist (SOAR, SimpleVLA-RL, RISE, CRL-VLA, etc.). None without simulation on consumer hardware. Narrow claim: "no sim, no fleet, data-quality-driven improvement."

**ICLR 2026 analysis (Moritz Reuss)**: 164 VLA submissions. Acknowledged gaps: consumer hardware scaling, data quality/curation, pretraining recipe ablations. These align with our contributions.

**Safe positioning**: "First systematic scaling laws for OOD VLA adaptation to an unseen embodiment" — still needs verification before submission.

**Gemini Robotics (March 2025)**: Gemini 2.0 Flash backbone → action decoder. Closed, no API, no open weights. Demonstrated on ALOHA 2 + humanoids. NOT a competitor — use as CONTRAST point. Framing: "frontier closed systems demonstrate what is possible at scale; we study the long tail." Gemini Edge (on-device variant) announced but specs undisclosed. Do NOT cite Gemini as a baseline.

**GR00T N1.5/N1.6** (NVIDIA): Open weights, ~2B, humanoid-focused. Runs on RTX 4090 for inference only. Not directly comparable to our 6-DOF arm work. Cite as related work in the "open large VLA" category.

**pi0/pi0.5** (Physical Intelligence): 3B, open-source, flow matching. VRAM-intensive for training (24-48 GB). Only commercially deployed VLA. Cite as sota frontier open model.

**Recommended narrative**: "Enabling the long tail" — frontier systems (Gemini, pi0.5) work at scale; we show what works for the majority who cannot access frontier infrastructure.

**Why: 2026-03-10 incident showed 4/5 "research gaps" were false. Always verify claims with 10+ search terms.**
**How to apply: All "first/novel" claims in paper drafts must go through overclaim review checklist before inclusion.**
