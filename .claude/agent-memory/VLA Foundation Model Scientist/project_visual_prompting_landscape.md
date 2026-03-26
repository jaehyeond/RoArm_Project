---
name: Visual prompting for VLA landscape analysis
description: Digital vs physical visual prompting landscape — 21 arXiv searches, 9 digital papers confirmed, 0 physical+learned-policy papers found, gap verified 2026-03-25
type: project
---

Systematic search (21 queries, 2026-03-25) to verify: "nobody has compared digital vs physical visual prompting for VLA."

**Why:** Research direction proposed for CoRL 2026. Before investing data collection, verified the gap is real.

**How to apply:** The gap is confirmed TRUE. Key facts below inform related-work writing, experiment design, and CoRL positioning.

## Confirmed counts (HIGH confidence)
- Digital visual prompting for learned robot policy: 9 papers confirmed
- Physical visual prompting AS learned policy input: 0 papers
- Comparison of digital vs physical: 0 papers

## 9 confirmed digital visual prompting papers
| Paper | Venue | What it does |
|-------|-------|-------------|
| TraceVLA (2412.10345) | ICLR 2025 | Trajectory trace overlay on image → VLA input |
| AimBot (2508.08113) | CoRL 2025 | Scope reticle/crosshair overlay → visuomotor policy |
| GENIMA (2407.07875) | IROS 2024 | Diffusion draws joint-action dots on RGB image |
| PIVOT (2402.07872) | ICML 2024 | Arrows+circles on image → zero-shot VLM control |
| MOKA (2403.03174) | RSS 2024 | Mark-based prompting → open-world manipulation |
| RT-Trajectory (2311.01977) | Google 2023 | Trajectory sketch overlay → task generalization |
| RT-Sketch (2403.02709) | Google 2024 | Hand-drawn sketch as goal specification |
| RoboPoint (2406.10721) | arXiv 2024 | Affordance point prediction on image |
| RoVI (2505.00693) | arXiv 2025 | Visual annotation as new modality replacing language |

## Physical laser/projector papers — all hardcoded control, no learned policy
- Torielli (2503.15987): laser pointer → wheelchair arm grasp (hardcoded)
- Kaiser 2009: laser pointer → industrial robot grasp (rule-based)
- LARS (2411.00007): projector for swarm stigmergy (no manipulation)

## SigLIP frozen encoder implication
- PIVOT (2402.07872) confirms: frozen VLMs CAN recognize arrows/circles from internet pretraining
- SigLIP has seen laser dots and projected circles in pretraining data (internet images)
- Key unknown: does SigLIP encode digital circle ≈ physical projected circle?
- Gate test: `model_siglip_marker_test.py` — 2-hour test, run before investing in data collection

## Gap framing for CoRL 2026 paper
"While digital visual prompting (9+ papers) has received growing attention, no prior work has used
physical visual cues (projected light, laser markers, colored stickers) as direct camera inputs to
a learned manipulation policy, nor has any work empirically investigated whether frozen vision
encoders respond equivalently to digital vs. physical instantiations of the same visual prompt."

## Residual verification risk
- IEEE Xplore workshop papers not in arXiv (recommended: manual search before submission)
- Industry demos without publications (unverifiable)
- Confidence: HIGH for arXiv scope, MEDIUM for full literature including workshop papers

## Files created
- `model_visual_prompting_landscape.md` — full 12-section landscape document
- `model_siglip_marker_test.py` — SigLIP gate test script (3 test modes)
