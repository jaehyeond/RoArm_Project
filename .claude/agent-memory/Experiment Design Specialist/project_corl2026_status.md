---
name: project_corl2026_status
description: CoRL 2026 research direction, contributions, and competitive risks as of March 2026
type: project
---

Current framing (C3 agent, 2026-03-23): "Data-efficiency frontier for OOD VLA adaptation."
Four claimed contributions:
1. Scaling laws (episodes x quality x steps) for OOD VLA fine-tuning on SmolVLA 450M
2. Collection-time data quality methodology (FK-depth, gripper phase, static frame detection)
3. Multi-object transfer characterization on consumer hardware
4. Self-improving loop (no simulator, no fleet, data-quality-driven)

Key risks already identified:
- "Accessible Physical AI" (arXiv:2512.11921) — closest competitor on consumer VLA hardware
- Data Scaling Laws (ICLR 2025) — scaling laws angle already partially covered
- 7+ self-improving VLA papers exist (SOAR, SimpleVLA-RL, RISE, etc.)
- Must NOT claim "first consumer hardware VLA" — already done

Defensible niche: OOD *embodiment* (RoArm-M3 was never in SmolVLA pretraining), collection-time quality metrics, no-simulation self-improvement on single consumer GPU.

Gemini Robotics analysis (2026-03-24):
- Gemini Robotics API: NOT publicly available. Research-only, requires Google DeepMind collaboration.
- Standard Gemini 2.5 Pro API: usable TODAY for image-based quality judgment ($0.0013/call).
- "Gemini as Oracle" ablation APPROVED: add to Contribution 2 as Section 4.3 VLM filtering comparison.
  - Conditions: no_filter / rule_filter / Qwen2.5-VL-3B (local) / Gemini 2.5 Pro (cloud)
  - N=30 trials per condition, same 200-episode source pool, equalized filter acceptance rate
  - Additional time cost: ~2-3 days. Fits within existing D-56 to D-44 window.
- All other Gemini directions rejected (guidance/annotation/distillation) — see experiment_gemini_oracle_design.py.
- Critical confound: filter acceptance rate must be equalized; human quality labels required as ground truth.

**Why:** CoRL 2026 deadline May 28, 2026. C3 agent landscape analysis shows crowded space but identifiable gap around practitioner-oriented OOD adaptation methodology.
**How to apply:** Every experiment must be designed to fill specifically the OOD embodiment + data quality gap, not generic "consumer VLA" framing. Gemini usage is ONLY as an ablation baseline within Contribution 2 — not a standalone direction.

---

## AR Augmentation Comparison Analysis (2026-03-24)

**Decision: sim-to-real is NOT a primary condition in the comparison experiment.**
- Drop sim-to-real from primary experiment; include only as 2-paragraph negative result in appendix using A2 agent's SigLIP cosine distance evidence (Isaac rasterizer: 0.6-0.8, transfer-blocking).
- The comparison is fundamentally about "which data strategy works within 1-person/65-day budget," not "which is globally optimal."

**Key finding: "AR visual augmentation at collection time" is NOT the right framing.**
- "AR augmentation during collection vs. post-hoc (GenAug)" = incremental, "augmentation with extra steps."
- Defensible framing: "AR guidance enforces spatial/task diversity as a structural property of collection, not post-processing."
- Post-hoc augmentation cannot retroactively add physical diversity. AR guidance changes operator behavior → changes demonstration quality.

**Recommended 4-condition structure:**
- A: 50ep baseline, no guidance, no augmentation
- B: Same 50ep as A + offline GenAug-style augmentation (shares data with A, eliminates collection confound)
- C: 50ep with AR target-circle spatial guidance, no offline augmentation
- D (optional): Same as C + AR visual overlay active at collection time

**Minimum N for statistical validity: N=50 trials per condition (5 positions x 10 trials).**
- N=20 is insufficient: 80% vs 90% CI overlap even at N=20. N=50 gives 80% power for 15 percentage point difference.

**Revolutionary scenario (worth designing toward):**
- Interactive AR for task-conditioned demonstrations: AR overlays different colors/textures on same physical object → same human, same physical trajectory, different task labels. Post-hoc augmentation structurally cannot do this. This is the only scenario where real-time AR is strictly superior to post-hoc.
- Requires multi-task evaluation; may be out of scope for 65-day timeline but worth flagging for thesis Chapter 4.

**Innovation test verdict:**
- "AR visual augmentation timing" = FAILS (incremental)
- "AR spatial guidance for operator behavior" = PASSES ("huh, that's clever" response expected from reviewers)

---

## 3DGS+VLA Feasibility Analysis (2026-03-24)

**Verdict: NO-GO (standalone main paper). CONDITIONAL (as AR+Oracle ablation).**
**Confidence: HIGH**

Key finding: 65-day budget cannot support both AR+Oracle (main, ~45 days) and 3DGS standalone (~40 days realistic).

**Go/No-Go Gate: SigLIP cosine distance measurement**
- < 0.30: GO (3DGS augmentation viable)
- 0.30–0.50: CAUTION (pilot required)
- > 0.50: NO-GO (same failure mode as Isaac rasterizer at 0.65)
- Reference: SplatSim (multi-view) = 0.15, Isaac rasterizer = 0.65

**3 scenarios evaluated:**
1. Standalone CoRL paper → REJECTED (time insufficient)
2. AR+Oracle Section 4.4 ablation → CONDITIONAL (gate pass + AR+Oracle done by 4/20)
3. Negative result appendix → RECOMMENDED if gate fails (cost: 1 day)

**Structural risks that cannot be mitigated:**
- Single Azure Kinect = sparse views → 3DGS quality uncertain
- Dynamic scene (robot arm + objects) requires foreground/background separation pipeline (1-2 weeks extra)
- GeoPredict (arXiv:2512.16811) already did 3DGS+VLA with multi-view setup

**Immediate action: SigLIP gate test this week (2026-03-28 deadline, cost: 1 day)**
- File: experiment_3dgs_vla_feasibility.py
- Run: python experiment_3dgs_vla_feasibility.py --mode gate_check --cosine_dist X
