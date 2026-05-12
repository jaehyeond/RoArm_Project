# Phase 1.B-α P6v12 Result (η fix: stage 2 near-cap + stage 3 transient bonus)

**Date**: 2026-05-12
**Status**: 🔴 **PARTIAL FAIL — design flaw confirmed, η-v2 escalate**
**Wall**: 6:40 (1000 iter @ 0.40s/iter, 243K steps/s)
**B200 PID**: 2179863
**Ckpt**: `logs/roarm_rl/p6v12_eta_stage2cap_stage3transient_resumeP6v11/model_999.pt`
**Env md5**: `b2027d9f7ad5caa0c0194e62738441fe`

## Patch (η-v1)

```python
# Stage 2 near-cap (d_sponge_target < 0.1 → 2.0)
stage2_r = torch.where(d_sponge_target < 0.1, torch.full_like(stage2_r, 2.0), stage2_r)

# Stage 3 rising-edge transient +10
just_on_target = is_on_target & ~self._stage3_fired
self._stage3_fired = self._stage3_fired | is_on_target
stage3_r = 6.0 + 0.5 * ungrasp_signal + 0.5 * static_signal + 10.0 * just_on_target.float()
```

## Launch Config

- Resume: `p6v11/model_999.pt` (bias[5]=+0.014 zero-ish, weight[5,:] mean=0.033 signed)
- `--reset_std 1.0` (was 1.5 in P6v11, final std 1.44 → contract)
- `--entropy_coef 0.001`, `--episode_length_s 2.0`, `--num_envs 4096`
- Removed (vs P6v11): `--reset_actor_bias_idx 5` (δ zero-cost, isolate η)

## Iter Trend

| iter | reward | gripper_open | is_on_target | stage4 | stage2_grasp | stage3_near | xy_off | z_off | std |
|---|---|---|---|---|---|---|---|---|---|
| 0 (P6v11 baseline) | (no opt) | 0.070 | 0.019 | 0.0003 | 0.82 | 0.019 | 0.166 | 0.054 | 1.00 |
| 65 | 637 | 0.072 | 0.185 | 0 | 0.67 | 0.185 | 0.069 | 0.044 | 1.00 |
| 247 | 755 | 0.069 | 0.320 | 0 | 0.51 | 0.320 | 0.072 | 0.046 | 0.97 |
| 351 | 787 | 0.069 | 0.346 | 0 | 0.51 | 0.346 | 0.072 | 0.044 | 0.96 |
| 532 | 794 | 0.062 | 0.397 | 0 | (n/a) | 0.397 | (n/a) | (n/a) | (n/a) |
| 750 | 822 | 0.064 | 0.394 | 0 | (n/a) | 0.394 | (n/a) | (n/a) | (n/a) |
| 966 | 824 | 0.064 | 0.397 | **0.0001** | (n/a) | 0.397 | (n/a) | (n/a) | (n/a) |
| **999** | **854** | **0.064** | **0.406** | **0.0002** | **0.45** | **0.406** | **0.081** | **0.048** | **0.88** |

## P6v11 vs P6v12 Comparison

| metric | P6v11 999 | P6v12 999 | Δ | verdict |
|---|---|---|---|---|
| Mean reward | 1059 | 854 | -19% | stage 2 cap penalty |
| gripper_open_rate | 0.063 | 0.064 | **+0.001 FLAT** | 🔴 release 학습 0 |
| is_on_target_rate | 0.016 | **0.406** | **25× ↑** | ✅ η transient + γ transport |
| stage4_success_frac | 0 | 0.0002 | first sporadic | 🟡 ~1/4096 fire |
| jackpot_fire_rate | 0 | 0 | 0 | 🔴 stable 조건 부족 |
| stage2_grasp_frac | 0.85 | 0.45 | -47% | reward farm shift |
| stage3_neartgt_frac | ~0 | **0.41** | new | reward farm shift |
| xy_offset (mm) | 67 | 81 | +21% worse | precision regression |
| z_offset (mm) | 48 | 48 | flat | |
| action_std | 1.44 | 0.88 | reset effect | |

## Root Cause (2-layer problem)

### Layer 1: η-v1 design flaw (transient gate)

```python
just_on_target = is_on_target & ~self._stage3_fired   # missing gripper_open
```

`is_on_target = (xy < 30mm) & (z < 25mm)` — **gripper state independent**.
→ Policy enters zone CLOSED ~93% of time → transient +10 fires in close state → no release incentive.

Expected first-fire reward by state (random exploration ~7% open):
- Close + zone entry: +17.0 × 0.93 = 15.81 (gradient mass on close path)
- Open + zone entry: +17.0 × 0.07 = 1.19 (gradient mass on open path)

→ PPO learns **close path** dominantly. Stage 3 reward farming shifts from stage 2 (P6v11) to stage 3 (P6v12).

### Layer 2: Stage 3 close-hover farming (newly exposed)

Stage 3 reward differentiation:
- Close + grasped: `6 + 0.5·~0 + 0.5·~1 = 6.5`
- Open + grasped: `6 + 0.5·~0.5 + 0.5·~0.5 = 7.0`
- Open + released: `6 + 0.5·1.0 (force-set) + 0.5·static = 6.5~7.0`

**1-step margin AT zone (after transient burnt): +0.5 only** (close vs open). Weak release incentive.

stage3_neartgt_frac 0.41 with gripper_open 0.064 = **41% of time policy is in close-hover at zone**.

## Forecast vs Actual (P6v12)

Forecast (5/12 session start):
- gripper_open 0.05~0.15 (slow rise) — **Actual 0.064 (flat, lower than min forecast)**
- stage4_success 0~0.05 — **Actual 0.0002 (within forecast lower bound)**
- on_target 0.20~0.30 — **Actual 0.41 (above forecast)**
- Mean reward ~1000 — **Actual 854 (lower, stage 2 cap penalty stronger than est)**

## Cross-Check Calculations

**Reward decomposition (iter 999, ~854)**:
- Stage 1 reach: ~10 step × 1.81 ≈ 18
- Stage 2 grasped (45% of time, mixed near-cap 2.0 + far 5.5): 199 × 0.45 × 3.75 ≈ 336
- Stage 3 (41% time, base ~6.5): 199 × 0.41 × 6.5 ≈ 530
- Stage 3 transient fires (~0.41 zone entries × 17 ≈ 7)
- Stage 4 latched (0.0002 × 199 × 8): 0.3
- Action penalty: -27 × 199 ≈ -5.4
- **Total estimate ≈ 886** vs actual 854 (-4% close enough)

## Path A vs Path B (η-v1)

- Path A (close, hover stage 3): 199 × 0.86 × 6.5 ≈ **1112**
- Path B (open at zone → stage 4 latched): 17 (transient) + 13 (jackpot) + 8 × 180 ≈ **1470**
- Advantage Path B = **+358** (32% relative)

PPO didn't find Path B because:
1. Transient was equally rewarded in close (+17.0 × 0.93 mass)
2. Persistent stage 3 close vs open margin only +0.5
3. Sponge_stable requirement for stage 4 fire = high variance after release

## η-v2 Design (next attempt)

### Plan A (minimal isolation)

```python
just_on_target = is_on_target & gripper_open & ~self._stage3_fired
self._stage3_fired = self._stage3_fired | (is_on_target & gripper_open)
```

- Transient +10 fires ONLY when open + on_target
- 1-step margin AT zone (transient unfired): close 6.5 vs open 16.5 = **+10.0** (first time)
- After fired: close 6.5 vs open 7.0 = **+0.5** (insufficient persistent)

### Plan B (η-v2 + stage 3 close-cap)

```python
just_on_target = is_on_target & gripper_open & ~self._stage3_fired
self._stage3_fired = self._stage3_fired | (is_on_target & gripper_open)
# Stage 3 close-cap 3.0 (analog to stage 2 near-cap)
stage3_r_open = 6.0 + 0.5 * ungrasp_signal + 0.5 * static_signal + 10.0 * just_on_target.float()
stage3_r_close = torch.full_like(stage3_r_open, 3.0)
stage3_r = torch.where(gripper_open, stage3_r_open, stage3_r_close)
```

- 1-step margin AT zone (transient unfired): close 3.0 vs open 16.5 = **+13.5**
- After fired: close 3.0 vs open 7.0 = **+4.0** (strong persistent)
- Path A (close hold zone): 199 × 0.86 × 3.0 ≈ 513
- Path B (open + stage 4): ~1500+
- Advantage: **+1000** (~3×)

**Recommendation: Plan B** (more aggressive, P6v12 close-hover farming explicitly broken).

## Next Launch (p6v13)

- Resume: P6v12 model_999 (on_target 0.41 learning retained)
- `--reset_std 1.5` (exploration boost, P6v12 std 0.88 → 1.5)
- `--entropy_coef 0.001`, `--episode_length_s 2.0`

## Lessons (for HARD RULES / memory)

1. **Reward-farm phenomenon**: caps in one stage shift farming to next adjacent stage. Need to break all "farm-able" stages, not just close.
2. **Rising-edge transient bonus design rule**: gating condition must INCLUDE the target behavior (gripper_open here), not just the spatial condition (is_on_target).
3. **Multi-condition gates**: stage 4 (on_target & gripper_open & sponge_stable) joint AND is harder than expected — sponge_stable after release transient is variance issue.
4. **Iter time 0.40s achievable**: P6v12 single GPU achieved 243K steps/s. P6v11 wall 6:50h was anomaly (possibly different episode_length or epoch settings).

## Wall Time Note

P6v12: 6:40 min wall for 1000 iter. Earlier P6v11 reported "6:50h wall". Difference suggests possible old episode_length_s=4.0 (200 step → 400 step) or different epoch count. P6v8α changed episode_length_s 4.0→2.0 (line 88). Need verification — out of scope for P6v12 result, but consider re-running P6v11 to verify 7-min wall consistency.
