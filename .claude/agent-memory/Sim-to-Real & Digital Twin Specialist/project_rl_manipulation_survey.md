---
name: Isaac Lab Manipulation RL Survey (2026-03-26)
description: Comprehensive survey of RL manipulation tasks in Isaac Lab — success rates, sim-to-real methods, object generalization, simulator comparison, real-world examples, and RoArm M3 applicability
type: project
---

## Survey Summary

Full survey in: `/home/cgxr/Documents/Robotics/RoArm_Project/sim_rl_manipulation_survey.py`

## Task Solvability (sim vs real)

| Task | Sim | Real | Transfer Gap | RoArm M3 |
|------|-----|------|--------------|----------|
| Reach EE | >95% | 85-92% | 3-10% | IMPLEMENTED |
| Pick-and-place (rigid) | 70-90% | 40-75% | 15-30% | NOT IMPLEMENTED |
| Block stacking | 60-80% | 30-55% | 25-40% | NOT APPLICABLE |
| Peg insertion | 85-95% | 50-70% | 20-35% | NOT FEASIBLE (tolerance) |
| In-hand manip | 80-95% | 55-75% | 15-25% | NOT APPLICABLE (1-DOF jaw) |

**Why:** RoArm M3 has ±2-3mm repeatability vs Franka's ±0.1mm. Sub-mm tasks infeasible.
**How to apply:** Limit RL tasks to pick-and-place with large (>3cm) objects.

## Sim-to-Real Methods Ranked

1. **Sys-ID** (FIRST): measure real dynamics, calibrate sim params. 30-50% gap reduction.
2. **Domain Randomization**: essential, +10-25%. NOT enough alone for contact tasks.
3. **Teacher-Student**: +15-40% for contact-rich. Adds 2-4 weeks dev.
4. **Residual RL**: useful if have VLA base policy. Hard with flow-matching.
5. **Visual Domain Adaptation**: INEFFECTIVE for SmolVLA (frozen SigLIP).

**What fails:** DR alone for contact tasks. Online RL on real robot. No-DR direct transfer.

## Consumer Arm Reality Check

Almost ZERO peer-reviewed sim-to-real RL papers use $100-500 arms.
Franka ($20k+), UR5 ($30k+) dominate.
Nearest example: Trossen PX100 (~$400) with DQN — 60% on training objects, 35% novel.

## Isaac Lab vs Competitors

- **Throughput**: Isaac Lab >> SAPIEN > MuJoCo MJX > PyBullet
- **Contact quality**: MuJoCo > Isaac Lab > SAPIEN > PyBullet
- **Rendering**: Isaac Lab (RTX) >> others (key for visual sim-to-real)
- **Setup**: MuJoCo easiest, Isaac Lab hardest

**For RoArm M3**: Isaac Lab for RL (already set up). MuJoCo for IL baselines.

## Real-World Deployment Numbers

| Paper | Robot | Real Success | Method |
|-------|-------|-------------|--------|
| TRANSIC (2405.14523) | Franka | 72% (6 tasks avg) | DR + interactive correction |
| Factory (2205.03532) | Franka | 50-70% (assembly) | Contact-tuned sim + DR |
| IndustReal (2310.03490) | Franka | 48-58% (insertion) | Sim-aware policy |
| GR00T N1 (2503.14734) | Humanoid | "40% improvement" | Isaac+fine-tune |

**GR00T N1 numbers: LOW confidence** — 40% is relative, no absolute rate, humanoid only.

## RoArm M3 Pick-and-Place Estimate

Expected: 65-80% [SIM], 35-55% [REAL], 50-65% with teacher-student.
SmolVLA IL comparison: 100% on demonstrated scenarios.

**RL advantage**: generalizes to new positions without re-demonstration.
**VLA advantage**: 100% on demonstrated, language conditioned.

## Blockers for Pick-and-Place RL

1. `activate_contact_sensors=False` in roarm_m3.py — must enable
2. No objects in scene (reach task only)
3. No gripper force control (parallel jaw, no feedback)
4. Isaac→LeRobot converter not built (1-2 weeks)
5. Stats.json mismatch when mixing sim+real (joint means differ 10-21 deg)

## Recommended Sequence

1. Sys-ID (3-5 days) — calibrate stiffness/damping to real hardware
2. Enable contact sensors (1 day)
3. Build pick-and-place task (1-2 weeks)
4. Build Isaac→LeRobot converter (1-2 weeks)
5. Sim-to-real ablation for CoRL (2-3 weeks)

**Why:** Novel contribution — no paper has done SmolVLA + RL sim augmentation on consumer arm.
**How to apply:** This is ablation contribution, not primary. Primary = AR+Oracle.
