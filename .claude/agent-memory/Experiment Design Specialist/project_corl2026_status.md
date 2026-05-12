---
name: project_corl2026_status
description: CoRL 2026 research direction — pivoted to Pure RL stacking task (May 2026). No deadline pressure.
type: project
---

## Current Direction (2026-05-13, major pivot from BC/VLA framing)

**Research task**: 4-sponge # tower stacking via Pure PPO RL in Isaac Lab (state-only, 28-dim obs).
**Goal**: sim RL mastery of sequential manipulation → characterize what Pure RL, BC, and BC+RL hybrid each achieve.
**3-way comparison (HARD RULE #21)**: Pure VLA (BC) / Pure RL (PPO from scratch) / Real-to-sim hybrid (BC warmstart + RL).

## Current RL State (Phase 1.B-alpha, P6v14a)

- Env: 1 sponge → L1.spot1 single fixed target
- Phase 0a DONE: stage4_success 0.778, gripper_open 0.578, jackpot 0.0044 (first ever)
- 7 reward farming patterns across P6v6-v14 → key lesson: exploration infeasibility (P(success|random pi)≈0) is root cause, not reward shape
- Phase 0a fix = structural init state (pregrasp), NOT reward shaping
- Next: Phase 0b (remove pregrasp, annulus close spawn, full grasp-transport-release chain)

## Phase Ladder (0b → 5, deadline-free)

```
0b: full chain, annulus r=0.08-0.15m, thresh 0.05/0.04 (~7min)
0c: annulus r=0.08-0.22m (~3.5min)
1:  R1-R4 WS + yaw ±30°, thresh 0.04/0.03 (~7min)
2:  production thresh 0.030/0.025, 3x robustness (~10min)
3:  goal-conditioned L1 both spots (~7min)
4:  L2 + wrist_roll 90° + static base sponge (~10-14min)
5:  full 4-sponge # tower, hierarchical RL (~20-40min)
```

Total estimated B200 wall: ~65-90min phases 0b-5

## Key Geometry (HARD RULES #19/#20)

- Sponge: edge-stand 47mm tall × 22mm wide × 125mm long
- L1: Y c2c = 87mm, spot1 Y = -0.0435m, spot2 Y = +0.0435m
- L2: X c2c = 67mm, ON TOP (z +47mm), wrist_roll 90°
- Table z = -0.012117, sponge_center_z = +0.011383

## Farming Pattern History (7 patterns, P6v6-v14)

1. P6v6/7/8: stage3 close-hover at z=88mm
2. P6v9/10: same with z-gating
3. P6v11: stage2 near-zone hold
4. P6v12: stage3 transient farm
5. P6v13: stage2 outside-zone avoidance hover at d=167mm
6. P6v14: stage2 boundary hover at xy_thresh=0.053m
7. (P6v14a broke the pattern via structural init fix)

**Why:** Each reward fix created a new local opt. PPO never observed stage4 success in 1B+ steps across all 7 attempts until Phase 0a.

## Statistical Notes

- Sim: N=4096 envs per iter, no power concern for phase gates
- Real robot: N=20 for go/no-go, N=50 for paper claims
- Multi-seed: run 3 seeds for Phase 2 production only (ablation verification)

**Why:** Deadline removed. No time pressure on sample size decisions.
**How to apply:** Every new phase launch requires: (1) pre-launch reward math at ALL threshold boundaries (margin > 20%), (2) rolling window diagnostic at iter 100/200/500, (3) hold-out spawn eval every 500 iter.
