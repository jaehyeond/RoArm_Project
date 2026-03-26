---
name: Multi-Position & Multi-Object Manipulation Capability Analysis
description: 5-Zone radial workspace strategy, gripper limits, speed tradeoffs, dual-arm coordination, low-cost arm milestones, top 5 failure modes. 2026-03-25.
type: project
---

## Workspace Coverage

Strategy: 5-Zone Radial (NOT XY grid)
- Grid ignores kinematic structure of serial arm → IK solution collapse per position
- Zones match reachability isosurfaces (annular shell of 6-DOF arm)
- Reference: Mandlekar et al. RSS 2021

150-episode breakdown:
- NEAR (80-140mm, ±30°): 30 ep — shoulder singularity caution
- MID_LEFT (120-220mm, -90~-30°): 25 ep
- MID_RIGHT (120-220mm, 30~90°): 25 ep
- FAR_CENTER (220-290mm, ±20°): 35 ep — elbow near 190° limit
- OVERHEAD (60-160mm, ±60°, height 100-200mm): 15 ep — place only

**Why:** 74-episode v3 failure had 68% SHALLOW → never learned deep reach. Zone-stratified sampling guarantees coverage.
**How to apply:** Any new data collection plan must specify episodes per zone, not just total count.

## Gripper Capability (Parallel Jaw, No Force Sensing)

Max opening: ~65mm (angle 100°). Angle-to-opening: ~0.65 mm/°.

Reliable objects (position-based grasp):
- Deformable/semi-rigid ≤55mm: sponge 92%, foam cube 85%
- Rigid ≤40mm: 70% (requires ±3mm angular precision)
- Cylinders ≤35mm: 65% (line contact, wrist_roll alignment helps → 75%)

Unreliable without force sensing:
- Slippery + rigid + heavy (bottles 50%)
- Flat/thin objects (card 20%)
- Small balls <25mm (30%)

CoRL scope: Deformable/semi-rigid ≤55mm only. Rigid small objects need tactile sensing (out of scope).

## Speed/Acc Tradeoffs

Current v3 success: speed=500, acc=200 (1 position).

Multi-position recommendations:
- MID zone: speed=400, acc=150
- FAR zone: speed=300, acc=100
- Dual-arm: speed=250, acc=80
- Gripper close: speed=200 always (avoid impact deformation)

Per-joint group limits (NEW — not yet in deploy_smolvla.py):
- Proximal joints 0-2: speed up to 500
- Distal joints 3-5: cap at 300 (Wrist_R polarity reversal at 500 confirmed)

Scale rule: speed = 500 × (1 − 0.3 × reach_fraction) where reach_fraction = actual_reach / 290mm

## Dual-Arm Coordination

Strategy 1 (recommended for CoRL): Static workspace partition
- ARM_L: base ∈ [-90, 10]°
- ARM_R: base ∈ [-10, 90]°
- No concurrent motion, zero collision risk
- ALOHA uses this exact pattern

Strategy 2: Sequential task decomposition (L picks → places at handoff → R picks)
Strategy 3: Explicit collision avoidance — NOT recommended until Year 2

Data requirement: ~150-200 ep/task for dual-arm (2× single-arm).
Reference: ALOHA 2 (2401.02117): 50 ep ACT → 67% bimanual.

**Critical safety:** Base partition MUST be software-enforced in deploy script in addition to JOINT_LIMITS.

## Low-Cost Arm Milestones (2024-2026)

| Platform | Price | Best Task | Notes |
|----------|-------|-----------|-------|
| Koch v1.1 | $250 | block stack 85%, color sort 72% | ACT, 5-DOF |
| SO-100/101 | $110-120 | pick 3 positions ~75% | SmolVLA compatible |
| RoArm-M3 | $350 | sponge 1-pos 100% | US (our current) |
| TWIST | $280 | bimanual wire tasks | 7-DOF, 2025 |

Key papers: Koch (LeRobot 2024), SO-100 (TheRobotStudio 2024), ALOHA (RSS 2023), Low-Cost Robot Arm (2304.03442).

CoRL milestones:
- [DONE] 1 object, 1 position: 100%
- [TARGET] 1 object, 5 positions: 60-75% (150 ep)
- [STRETCH] 3 objects, 5 positions: 40-60% (200 ep)
- [DUAL] Sequential bimanual: 50-65% (not CoRL scope)

## Top 5 Failure Modes

1. OOD Drift [HIGH] — Mitigated: n=50, EMA alpha=0.4, JOINT_LIMITS
2. Gripper Timing [HIGH] — PARTIALLY mitigated: no verify step in deploy_smolvla.py yet
3. Approach Angle Mismatch [MEDIUM] — Not yet (1-position only)
4. Elbow Singularity at FAR [MEDIUM] — Partial (JOINT_LIMITS, but no 175° cap)
5. Chunk Boundary Discontinuity [LOW-MED] — Mitigated: EMA

**Immediate action items for deploy-agent:**
- Add gripper_angle verify step before lift phase
- Add distal joint speed cap (≤300) in joints_angle_ctrl calls
- Add dual-arm base partition enforcement

**Why:** 74-ep dataset had no multi-position, so failure modes 2-5 are untested. Must address before scaling.
**How to apply:** Any multi-position deployment test must instrument gripper_angle + elbow + chunk boundaries in CSV log.
