---
name: CoRL 2026 Paper Synthesis (2026-03-25)
description: Final synthesized paper direction from 4 agent reports — title, abstract, 3 contributions, competitor, reviewer criticisms, gate experiments
type: project
---

## Final Paper Direction

**Title (working)**: "Denoising Uncertainty as a Competence Signal: Self-Improving VLA on a $130 Robot"

**Alternative (conservative)**: "Competence-Aware Adaptation for Consumer-Grade VLA Robots"

**Why**: Title must name the technical contribution (denoising uncertainty) AND the problem (competence boundaries). Not a system paper title.

## 3-Sentence Abstract (template — X/Y/Z = placeholders)

A robot arm that succeeds in training zones but fails silently outside them is not deployable — yet today's small-lab VLA practitioners have no principled way to know where their policy is competent. We present a method that uses the variance across denoising steps in a flow-matching VLA as a spatially-grounded uncertainty signal, requiring no modifications to the model architecture, no simulation, and no fleet infrastructure. On a $130 6-DOF arm with an RTX 4090, this signal drives a fully autonomous overnight loop: the robot maps its own competence boundaries, solicits new demonstrations in uncertain zones, and retrains — achieving X% success rate across Y workspace positions after Z nights without human intervention.

## 3 Contributions

**C1 (Technical — Novel):** Per-step denoising variance in flow-matching VLA = calibrated spatial uncertainty signal, no architectural changes, no labels.
- Overclaim status: MEDIUM. Verify "flow matching uncertainty robot" on arXiv 2025-2026 with 10+ searches before submission.

**C2 (System — Reproducible):** Fully autonomous self-improving loop: deploy → 3B VLM judge (FK+gripper hard gate, VLM soft label) → retrain → redeploy. ~6 hr cycle, ~$4.50/cycle, $130 hardware, zero simulation.
- Overclaim status: MEDIUM. Differentiate from SOAR on: (1) no sim, (2) consumer hardware, (3) uncertainty-guided zone selection. Verify SimpleVLA-RL hardware requirements.

**C3 (Empirical):** Systematic competence boundary characterization for fine-tuned open-source VLA on unseen embodiment — success rate vs workspace position × object × episode count.
- Overclaim status: LOW risk. Use "methodology for characterizing" not "first characterization."

## Primary Competitor
SOAR (CoRL 2024). WidowX (~$2K), CLIP reward, fleet assumptions, no spatial uncertainty.
Position as: "SOAR at fleet scale; we ask if the same is achievable with one robot + principled uncertainty signal."

## Top 4 Reviewer Criticisms

1. (HIGH) "Denoising variance is not principled uncertainty — could be multi-modal."
   Defense: calibration experiment (variance vs failure rate, Pearson r > 0.7). Acknowledge multi-modality as confounder, argue single-task setting mitigates it.

2. (HIGH) "Just SOAR on cheap hardware — no scientific contribution."
   Defense: Frame the paper as uncertainty estimation paper VALIDATED in self-improving loop, not system paper. Contribution 1 is the science; loop is the evaluation vehicle.

3. (MEDIUM) "Single robot, anecdotal."
   Defense: Acknowledge in limitations. Architecture-agnostic uncertainty signal + task-agnostic VLM judge are the generalization arguments. Compare to SOAR/SERL which are also single-robot.

4. (MEDIUM) "VLM judge accuracy not validated."
   Defense: Gate 1 must be Table 1 in the paper, not supplementary. If <70%: paper is significantly weaker.

## 3 Non-Negotiable Gate Experiments (before writing Introduction)

1. Gate 1: VLM judge accuracy on 74 episodes — must be >85% (half day)
2. Gate 2: Denoising variance vs empirical failure rate across 5 zones — Pearson r must be >0.6 (one day)
3. Gate 3: Control mode ablation (same data, open-loop vs closed-loop) — resolves v1→v3 causality (one day)

**Do not write Introduction until all 3 gates pass.**
Method section can be drafted in parallel (no result dependency).

## Thesis Mapping (December 2026)
- Ch1: Competence boundaries problem
- Ch2: Related Work
- Ch3: Data quality tools (Framing C material — thesis only, not CoRL)
- Ch4: Denoising uncertainty (C1 expanded — EE background, information-theoretic interpretation)
- Ch5: Autonomous loop + boundary characterization (C2+C3)
CoRL paper = Ch4+Ch5 compressed to 8 pages.

**Why**: Synthesized from B1 (uncertainty signal), C3 (framing), B3 (loop feasibility), C1 (gate experiments) reports 2026-03-25.
**How to apply**: All paper drafting decisions must trace back to the 3 gate experiment results. Do not commit to framing before gate data exists.
