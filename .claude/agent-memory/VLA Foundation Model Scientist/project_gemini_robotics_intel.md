---
name: Gemini Robotics competitive intelligence
description: Factual analysis of Google DeepMind's Gemini Robotics stack for CoRL 2026 paper positioning
type: project
---

## Core Facts (HIGH confidence)

- Announced: March 12, 2025. arXiv: 2503.07218 ("Gemini Robotics: Bringing AI into the Physical World")
- Built on Gemini 2.0 base model. Parameter count NOT disclosed. Presumed >70B.
- Family members: Gemini Robotics (base) + Gemini Robotics-ER (Embodied Reasoning)
- ER variant focuses on spatial/physical reasoning BEFORE acting (e.g., "which is heavier? pick it up")
- "Edge" variant claimed for on-device deployment — LOW confidence on specs, no paper ID confirmed
- Demonstrated on Franka Research 3 arms in controlled lab settings

## Open Source Status (HIGH confidence)

- NOT open source. No weights, no public API, no code. "Trusted testers" only as of March 2026.
- Student with RTX 4090 CANNOT use this. Cannot compare directly in CoRL paper.
- Contrast: SmolVLA (450M, Apache 2.0, LeRobot), Octo (93M, open), OpenVLA (7B, open)

## Historical Context (HIGH confidence)

- Lineage: RT-1 (2022) → RT-2 (2307.15818, 2023) → RT-X (2310.08864) → Gemini Robotics (2025)
- Everyday Robots (EDR) team shut down January 2023. Google has NO in-house robot hardware team now.
- All current demos use Franka arms (external hardware), NOT Google-owned robots.
- Octo (arXiv:2405.12213) is a SEPARATE, open-source academic line — NOT a Gemini Robotics variant.

## Honest Assessment (MEDIUM confidence — no independent replication)

- Cherry-picking risk: demos show folding/origami in controlled settings; no systematic real-world eval
- SIMPLER benchmark showed RT-2 performs significantly worse than claimed on held-out tasks
- pi0 is ACTUALLY deployed to customers (Weave, Ultra). Gemini Robotics is NOT confirmed deployed.
- Evaluates primarily on LIBERO (sim) and their own GROOT benchmark — no independent verification

## CoRL Paper Positioning (CRITICAL)

**Why:** Gemini Robotics helps our narrative WITHOUT needing direct comparison.

**How to apply:**
- DO: "Gemini Robotics [2503.07218] shows what is possible at scale; we ask what is possible at minimum resource"
- DO: Cite as closed/inaccessible approach in related work contrast
- DO NOT: Claim we outperform Gemini Robotics
- DO NOT: Claim it's "not deployable" (may be by review time)
- PRIMARY comparison question from reviewers: "Why not OpenVLA-OFT?" (NOT "Why not Gemini Robotics?")

## Items to Verify Before CoRL Submission

- Has 2503.07218 been published at a venue (CoRL 2025 Seoul? NeurIPS 2025)?
- Are there Gemini Robotics follow-up papers with independent benchmarks?
- What is the Edge variant's actual specification and target hardware?
- Has any trusted tester published independent results?
