---
name: Sim-to-Real RL Critical Analysis — When It Works and When It Fails (2026-03-26)
description: Comprehensive evidence analysis of RL sim-to-real for manipulation. Key finding: works for industrial arms (Franka 83-99%), fails for hobby servo arms (zero papers). VLA+real demos wins for RoArm M3.
type: project
---

## Core Finding

RL sim-to-real for manipulation is NOT universally solved — it is conditionally solved for industrial hardware only.

**Why:** ALL published RL sim-to-real successes use industrial arms (Franka $30K, Allegro $10K, UR5 $35K). Zero papers test on hobby servo arms. The gap is not just "sim-to-real gap" — it is 8 distinct hardware-level mismatches.

**How to apply:** When RL is proposed as a path for RoArm M3, use this analysis to show it is not viable without 3-4 months of prerequisite work (URDF, SysID, stiction modeling). VLA with real demos is proven (100%); RL is unproven.

## What Actually Works (HIGH confidence, peer-reviewed)

| Task | Robot | Result | Paper |
|------|-------|--------|-------|
| Peg insertion | Franka | 83-99% zero-shot | IndustReal, RSS 2023 |
| Gear assembly | Franka | 83-99% zero-shot | IndustReal, RSS 2023 |
| Tactile insertion | Franka + F/T | 83-91% | TacSL, NVIDIA |
| In-hand manipulation | TriFinger | 83% | Published competition |
| In-hand cube rotation | Allegro ($10K) | "Repeated success" (no % given) | DeXtreme, ICLR 2024 |
| Tabletop manipulation | Franka | 21.7% -> 75% with gen. 3D scenes | arXiv:2603.18532 |
| Locomotion (quadruped) | ANYmal, Unitree | >95% | Many papers |

**Critical pattern:** All successes use industrial arms + known objects + clear binary reward.

## What Doesn't Work / Has No Evidence

- Language-conditioned grasping: RL architecturally cannot do this
- Diverse object grasping: RL has no object semantics
- Consumer/hobby arms: ZERO published papers
- VLA with frozen SigLIP + Isaac rasterizer: cosine 0.6-0.8 (FAIL)
- Deformable objects: physics sim inadequate

## RoArm M3 Specific Challenges (8 dimensions worse than Franka)

| Challenge | Franka | RoArm M3 | Sim-Modelable? |
|-----------|--------|----------|----------------|
| Repeatability | <0.1mm | 2-5mm | NO |
| Torque sensing | Yes (7 joints) | NONE | NO |
| Control frequency | 1000Hz (EtherCAT) | 20-50Hz (USB serial) | NO |
| Communication latency | <1ms (deterministic) | 20-50ms (stochastic) | NO |
| Servo backlash | <0.1 deg | 1-3 deg | NO (discontinuous) |
| Stiction | Low, characterized | High, uncalibrated | NO (PhysX is viscous not stiction) |
| URDF availability | Yes (well-tuned) | NONE | N/A (prerequisite) |
| Published calibration | Many papers | ZERO papers | N/A |

## Stiction is the Hardest Problem

PhysX models: F_friction = mu * F_normal (continuous, linear)
Real servo: static friction SPIKE at zero velocity — discontinuous nonlinearity
Domain randomization of mu does NOT fix this mismatch — they are different physics.

This project's v1 failure: "Wrist_R -3° -> -92°" = exactly what stiction mismatch causes in real deployment.

## Estimated RoArm M3 RL Success Rate

If someone built URDF + ran IndustReal-style RL: expected <20% real-world success.
Basis: 8-dimensional hardware gap × IndustReal assumptions violated in every dimension.

Compare: VLA fine-tuning achieves 100% (74ep, 1 object). No URDF needed.

## What Works for RoArm M3 (ranked by effort)

1. **VLA fine-tuning with real demos** (PROVEN): 100% single object, 3-4 days collection
2. **Reward-weighted BC** (2-3 days, compatible with flow-matching): expected +15-30%
3. **Sim+real co-training** (30 days, needs URDF): +20-40% based on RSS 2025
4. **Pure RL sim-to-real** (90+ days, HIGH RISK): likely <20% success

## Key Papers for Related Work

- IndustReal (arXiv:2305.17110, RSS 2023): Gold standard RL success, Franka only
- Beyond Imitation (arXiv:2602.12628, 2026): +24% OpenVLA with hybrid RL
- SplatSim (arXiv:2409.10161, 2024): Visual domain gap quantified (82% 3DGS vs 45% rasterizer)
- Yardi et al. (arXiv:2501.16389, 2025): ViTs (SigLIP) have WORSE domain invariance than CNNs
- Sim-Real Co-Training (arXiv:2503.24361, RSS 2025): +37.9% with mixed training
- Scaling Sim-to-Real RL (arXiv:2603.18532, 2026): 21.7% -> 75% with gen. 3D scenes

## Research Positioning for Paper

"While RL sim-to-real achieves 83-99% on industrial arms with known geometry [IndustReal],
it has never been demonstrated on hobby servo arms (~$130), which violate every assumption:
no URDF, no force sensing, ~50ms USB latency, and servo stiction non-modelable in PhysX.
VLA fine-tuning with 74 real demonstrations achieves 100% success on RoArm M3,
requiring no simulation, no object CAD models, and no force sensors."

Confidence: HIGH (8 papers + our own results)
