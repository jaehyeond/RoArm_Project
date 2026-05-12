# Phase 1.B-α P6v14 Launch + Sanity Gate FAIL (2026-05-12 evening)

## Status: 6th reward farming pattern (boundary hover) — sanity gate FAIL

P6v14 Phase 0 Curriculum (Option B, Plan from P6v13 result.md) launched on B200 at 23:43.
Sanity check at iter 421/1000 (time 3:06, ETA 4:15) detected 6th farming local opt
exactly as user warned in P6v13 session ("매번 fix가 새 local opt 만듦").

## Patch Summary

**Code** (md5 envs B200/local matched):
- env  `453acf68eac6a778c16eebb81c4131ef` (roarm_stack_env.py)
- train `2a2b7e93e4932f49f2d0b73e439096f6` (train_ppo.py)

**Three traps addressed**:
1. spawn-at-target (iter-0 trivial jackpot) — min_r=0.08 > xy_thresh=0.05 margin 30mm ✓
2. near-zone cap conflict (short-transport blocked) — cap disabled in Phase 0 ✓
3. tight thresholds (random π release infeasible) — xy 0.030→0.050, z 0.025→0.040 ✓

**Phase 0 launch params**:
```
--curriculum_spawn_min_r 0.08
--curriculum_spawn_max_r 0.15
--curriculum_xy_thresh 0.05
--curriculum_z_thresh 0.04
--curriculum_disable_nearzone_cap
--resume p6v13/model_999 --reset_std 1.5 --entropy_coef 0.001
--episode_length_s 2.0
```

## Sanity Gate Result (iter 422 snapshot)

| Metric | P6v13 final | P6v14 i422 | Δ |
|---|---|---|---|
| Mean reward | 894 | 1053 | +18% (Path A 강화) |
| grasped_frac | 0.865 | 0.872 | flat |
| gripper_open_rate | 0.061 | **0.061** | **identical, NO release learning** |
| sponge_target_dist | 0.167m | **0.087m** | **−48% transport 학습 ✓** |
| xy_offset_mean | n/a | **0.053** | **boundary 정확 stick (thresh 0.05)** |
| z_offset_mean | n/a | **0.058** | **above z_thresh 0.04, sponge hovers 8cm above table** |
| sponge_height_m | n/a | 0.081 | hover 8cm |
| ungrasp_signal | n/a | 0.19 | gripper hard-closed |
| sponge_stable_rate | n/a | 0.149 | |
| jackpot_fire | 0 | **0** | **FAIL (gate: >0.001 by iter 20)** |
| stage4_success | 0 | **0** | release 0회 |

**Verdict**: jackpot_fire stayed at 0 throughout iter 1-422. Sanity gate FAILED.

## Root Cause — 6th farming pattern (boundary hover)

### Numerical analysis at policy convergence d=0.053

Stage 2 (grasped, cap removed): reward = 4 + 3·(1−tanh(5·d))
- d=0.053: 4 + 3·(1−tanh(0.265)) = 4 + 3·0.741 = **6.22/step**
- 200 step × 6.22 = **1244** total reward

Stage 4 release path: 5 jackpot + 8·150 sustained = **1205** total reward

**Close-hover @ xy_thresh boundary wins by 1244−1205 = 39 reward (+3.2%)** → PPO sits exactly at boundary to maximize stage 2 without entering stage 3 cap (3.0/step close).

### My design error

In numerical sanity I checked d=0.08 (spawn distance) only:
- d=0.08: stage 2 = 5.86 → 1172 < release 1205 ✓ (I declared safe)

I did NOT check d=xy_thresh+ε=0.052 where PPO would converge after learning to
move sponge from spawn toward target. With cap removed, stage 2 reward is
monotonically increasing as d→0 until on_target boundary. PPO finds exactly
that boundary (cap re-enters at on_target=True for closed gripper at 3.0).

**Lesson**: Curriculum that removes a guard creates a new local opt at the
guard's removed boundary. Verified empirically (xy_offset_mean = 0.053 ≈ thresh 0.050).

## What worked

- **Transport learning**: sponge_target_dist 0.167 → 0.087m (−48%). Annulus spawn
  giving agent close-distance starting points DID teach transport gradient.
- **No zone avoidance regression**: P6v13's d=0.167m hover (zone avoidance, stage 2
  outside cap > inside cap) eliminated by cap removal. Confirmed agent now enters
  near-target region.
- **Spawn-at-target avoidance**: iter 1-5 no trivial jackpot fire from spawn (xy_offset
  started at ~0.10-0.12, decreased to 0.053). min_r=0.08 margin correct.

## What didn't work

- **Release path still infeasible**: gripper_open_rate IDENTICAL to P6v13 (0.061).
  Curriculum lowered transport difficulty but did not address fundamental issue:
  gripper bias is hard-closed (P6v13 actor.6.bias[5] likely strongly positive).
- **Cap removal created boundary hover**: 6th farming pattern.

## Critical Re-evaluation — Pure RL inherent limitation

This is the **6th farming local opt across 7 reward iterations** (P6v6→v14).
Each fix creates a new local opt:
1. P6v6/7/8 stage 3 close-hover farm at z=88mm
2. P6v9/10 same (z-gating tried)
3. P6v11 stage 2 near-zone hold farm
4. P6v12 stage 3 transient farm (close)
5. P6v13 stage 2 outside-zone avoidance hover at d=167mm
6. **P6v14 stage 2 boundary hover at xy_thresh=0.053** ← here

Pure reward shaping cannot escape this without addressing **the underlying
exploration problem**: PPO has never observed a single stage-4 success in 1+ billion
steps, so it has no gradient toward the release path. No reward shape alone can
force exploration of an unobserved action.

## Recommended Next Actions — sorted by escalation

User decision needed. Phase 0 attempt clear NOT enough. Options:

### Option α — pre-grasp init (more aggressive curriculum) ⭐⭐⭐⭐⭐
Force gripper closed + sponge attached at episode start, sponge spawn directly above
target. Agent ONLY needs to release (lower gripper to table + open). Only release
path possible. ~50 LOC in `_reset_idx`. Bootstrap signal guaranteed within iter 1-5.

### Option β — bigger jackpot (5 → 100) + smaller release threshold (50mm xy → 40mm) ⭐⭐
Math: close-hover 1244 stays, release path 100+8·150 = 1300 wins by +4.5%. Risk:
7th farming pattern at NEW boundary. Faster (no code).

### Option γ — additive structural penalty: "must release" episode-end penalty ⭐⭐⭐
If episode ends with `_was_open=False` (never opened gripper), apply -200 penalty.
Forces exploration of release. May cause premature release at random spots.

### Option δ — HER (Hindsight Experience Replay) ⭐⭐⭐⭐
Re-label failed episodes as "succeeded at wherever sponge ended up". Provides
gradient signal even when never reaching target. Requires off-policy buffer
(PPO is on-policy → architectural change).

### Option ε — BC warm-start from v6 50-ep real demos ⭐⭐⭐
1-2 day setup. Initializes π near-grasp+lift trajectory, breaks gripper bias.
HARD RULE #26 deadline 5/19 → 7 days left → feasible but eats half deadline.

### Option ζ — STOP. Report "Pure PPO + sparse manipulation reward is intractable" ⭐
Honest negative result. HARD RULE #26 explicit abort criteria allow this. Pivot
to BC+RL hybrid (HARD RULE #21 #3) after 5/19.

## Recommendation — Option α (pre-grasp init)

**Rationale**:
- 5-9 lines in `_reset_idx`. ~30 min wall to launch.
- GUARANTEED release signal observation in iter 1-5 (no exploration required).
- Once release learned (Phase 0a, sponge above target only), expand to need-grasp
  (Phase 0b → 0c). Curriculum escalation.
- Other options either shape-tweak (β/γ — risk 7th farming) or architectural
  (δ/ε — eats deadline).

**Phase 0a design**:
- Sponge spawn = target xy ± 20mm + z = target_z + 0.05 (5cm above table = above target)
- Gripper joint q init = 0.8 (closed, gripping sponge)
- `_grasped=True`, `_was_grasped=True` at episode start
- Sponge attached to TCP at episode start (via `_update_grasp_attach`)
- Agent must: lower TCP + open gripper
- Stage 4 success_now = is_on_target & gripper_open & stable
- Random π: sometimes opens gripper → sponge falls → if xy<0.05 and z drops to target_z → success_now fires
- Expected first-fire iter < 5

**Phase 0b** (after Phase 0a converges): remove pre-grasp init, sponge on table near
target. Agent must grasp+lift+place+release.

**Phase 0c → Phase 1 → Phase 2**: progressive expansion as designed.

## Decision Pending — user to confirm

α / β / γ / δ / ε / ζ ?

Recommendation: **α (pre-grasp init Phase 0a)** — fastest bootstrap, structural fix
to exploration not shaping.
