---
name: Self-Improving VLA Landscape (as of 2026-03)
description: 7+ competing papers for self-improving VLA. Defensible niche is no-simulator + consumer hardware + data-quality-driven (not RL reward).
type: project
---

# Self-Improving VLA Landscape

**Verified: 2026-03-23**

## Competing Papers (7+)
| Paper | Venue | Requires Sim? | Consumer HW? | Method |
|-------|-------|---------------|--------------|--------|
| SOAR | CoRL 2024 | No | Not demonstrated | Autonomous practice + VLM success detection |
| SimpleVLA-RL | ICLR 2026 | Yes | No | Residual RL fine-tuning |
| RISE (2602.11075) | Feb 2026 | No | No (large GPU) | World model + imagination RL |
| Reflection-Based (2510.12710) | Oct 2025 | No | Not demonstrated | VLM reflection + PPO + prioritized SFT |
| CRL-VLA (2602.03445) | Feb 2026 | Yes | No | Continual RL with bounds |
| Simple Recipe (2603.11653) | Mar 2026 | Yes | No | Sequential fine-tuning + RL |
| Self-Improving + Residual RL | ICLR 2026 WS | Yes | No | Data gen via residual RL |

## Defensible Niche
"Self-improving loop that operates entirely on a single consumer-grade setup without simulation or fleet infrastructure, using **data quality curation** (not RL reward) as the primary improvement signal."

Three requirements that no existing paper satisfies simultaneously:
1. No simulator
2. Consumer hardware ($130 robot + RTX 4090)
3. Improvement driven by data quality metrics, not reward function

## SOAR is the closest competitor
SOAR (CoRL 2024): autonomous practice + VLM success detection. Key differences:
- SOAR requires some baseline task performance to start practicing
- We start from zero autonomous capability, bootstrap with hand-guided demos
- SOAR doesn't address data quality filtering, just binary success detection

**How to apply:** When designing self_improve_*.py scripts, frame the loop as quality-curation-driven, not reward-driven. This is the differentiating angle.
