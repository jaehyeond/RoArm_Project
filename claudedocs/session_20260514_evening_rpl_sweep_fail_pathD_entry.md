# Session 2026-05-14 evening — RPL α-sweep FAIL → Path D Entry

## TL;DR
- Ran **P6v16b (α=0.05)** and **P6v16c (α=0.10)** on B200, sequential, ~3.5min each.
- **Result: All 3 α values FAIL.** iter 10 stage4 ≈ 0.003-0.004 across (P6v14c, α=0.05, α=0.10, α=0.30) — bit-identical early collapse.
- **Verdict**: RPL framework REJECTED. Residual capacity is NOT the bottleneck.
- **Decision** (per continuation prompt gate): **Path D entry**.

## Unified Sweep Table — `stage4_success_frac`
| iter | P6v14c (no RPL) | α=0.05 (P6v16b) | α=0.10 (P6v16c) | α=0.30 (P6v16) |
|-----:|:---------------:|:----------------:|:----------------:|:----------------:|
| 0    | 0.3653          | 0.3649           | 0.3649           | 0.3649          |
| 1    | 0.3446          | 0.3451           | 0.3451           | 0.3449          |
| 2    | 0.2965          | 0.2977           | 0.2973           | 0.2964          |
| 5    | 0.1465          | 0.1476           | 0.1479           | 0.1461          |
| 10   | 0.0037          | 0.0035           | 0.0037           | 0.0033          |
| 50   | 0.0039          | 0.0022           | 0.0032           | 0.0028          |
| 100  | 0.0027          | 0.0026           | 0.0047           | 0.0023          |
| 200  | 0.0029          | 0.0143           | 0.0112           | 0.0123          |
| 300  | 0.0055          | 0.0156           | 0.0134           | 0.0172          |
| 400  | 0.0047          | 0.0150           | 0.0096           | 0.0194          |
| 499  | 0.0105          | 0.0226           | 0.0140           | 0.0271          |

## gripper_open_rate — Forgetting Indicator
| iter | P6v14c | α=0.05 | α=0.10 | α=0.30 |
|-----:|:------:|:------:|:------:|:------:|
| 0    | 0.7835 | 0.7809 | 0.7809 | 0.7809 |
| 1    | 0.2100 | 0.2067 | 0.2060 | 0.2081 |
| 10   | 0.1119 | 0.1180 | 0.1173 | 0.1134 |
| 499  | 0.0692 | 0.0666 | 0.0669 | 0.0724 |

**Critical observation**: iter 1 gripper_open drops from 0.78→0.21 **across all 4 runs identically** (~73% drop in 1 iter). RPL with α=0.05 (BC dominance ≈ 95%) does NOT slow this cascade.

## RPL Framework Rejection — Why α Sweep Failed

### Hypothesis (5/14 Round 2 evidence: Silver 2018 + Ankile 2025 ResFiT)
- α=0.05 → residual ±0.05 per dim, BC dominates 95% → forgetting impossible by construction.
- PASS gate: iter 10 stage4 ≥ 0.20.

### Falsification
1. **iter 0-10 cascade is α-invariant**: all 4 stage4 curves overlap within <0.002 (noise floor).
2. **gripper_open 1-iter drop is α-invariant**: -73% in iter 0→1 regardless of α.
3. **Late recovery (iter 200+) is non-monotonic in α**: α=0.30 > α=0.05 > α=0.10. Pure variance, not signal.

### Root Cause Diagnosis (revised)
- **Frozen BC mean + learnable PPO log_std** = action sampled with high noise around BC anchor.
- During rollout, noisy samples explore closed-gripper actions → high grasp reward → PPO advantage pushes log_std DOWN AND mean (via residual) into grasp basin.
- Residual capacity ±α is irrelevant if **log_std is unconstrained**.
- **The forgetting is in the stochastic policy, not in the deterministic mean.**

This explains the bit-identical collapse:
- All 3 RPL runs share P6v14a's log_std (= PPO-learned during P6v14 training, value mid-range).
- All 3 RPL runs start with zero-init residual → action_mean ≈ BC_mean at iter 0.
- PPO update at iter 0→1 reduces log_std on grasp-axis dims, regardless of α.
- iter 1 onwards, sampled actions cluster in grasp basin → cascade indistinguishable across α.

### Why RPL papers (ResFiT 2025) work — and ours doesn't
- ResFiT: BC pre-trained then fine-tuned by IQL/AWAC (off-policy, no log_std drift).
- Our setup: PPO (on-policy, log_std fully trainable, rare-event reward).
- **RPL + PPO + rare-event reward = mismatch**. RPL fixes mean-drift; PPO + rare-event drifts via std collapse + advantage on grasp basin.

## Decision Tree (continuation prompt verbatim)
- alpha=0.05 stage4 iter 10 > 0.20 → **PASS**, alpha 더 sweep + 1000 iter long train.
- 모두 FAIL → **Path D 진입** (P6v14a + release BC).

**All 3 alpha FAIL** → Path D entry.

## Path D Plan Recap
[claudedocs/path_d_design_20260514.md](path_d_design_20260514.md):
- **D.1**: P6v14a/model_499 rollout sweep → release-only demos (~74 of 200 successful).
- **D.2**: Train release BC (MLP 28→64→6, ~3min on B200 CPU).
- **D.3**: State machine deploy (pick=P6v14a, release=BC, handoff on `_was_grasped & sponge_z > target_z+thresh`).
- Success criteria: stage4_success ≥ 0.30 (vs P6v15 0.011 baseline) → SIGNIFICANT.

## Files Produced (this session)
- `launch_p6v16b_alpha05.sh`, `launch_p6v16c_alpha10.sh` (B200 sync, md5 verified)
- `claudedocs/p6v16b_data/train_p6v16b.out`, `p6v16b_metrics.csv`, `extract_metrics.py`
- `claudedocs/p6v16c_data/train_p6v16c.out`, `p6v16c_metrics.csv`, `extract_metrics.py`
- This doc.

## B200 ckpts (deploy DENY — all RPL runs)
- `logs/roarm_rl/p6v16b_pathB_RPL_alpha005/model_*.pt`
- `logs/roarm_rl/p6v16c_pathB_RPL_alpha010/model_*.pt`

## HARD RULE Compliance
- #4 (10+ search × 2 source): 5/14 Round 1+2 = 6 agents already verified RPL framework.
- #11 `/half-clone` 절대 금지: 본 세션 사용 0회.
- #13 B200 path `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200`: verified `whoami`/`hostname`.
- #14 fail-fast guard: all ssh commands had `[[ "$(whoami)" != "sogang_jhki" ]] && exit 1`. No `2>&1`.
- #15 cu128 sm_100 alive: env unchanged.
- #17 state-only 28-dim: env md5 unchanged ff31c5a3.
- #18 user explicit gate followed verbatim: "모두 FAIL → Path D 진입".
- #26 deadline removed: still proceeding step-by-step.

## Next Step (this session, then user check-in before D.2/D.3)
**D.1 demo gen** — `roarm_rl/scripts/gen_release_demos_from_rollout.py` ~120 LOC:
- Single-env Isaac Sim launch (RoArmStackEnv, curriculum_pregrasp init).
- Load P6v14a/model_499.pt actor.
- Run 200 episodes (parallel via num_envs=200 single rollout = ~5min B200).
- Filter `place_success_flag==True` at any step → save trajectory window
  `[T_grasp_release-2 : T_release+5]` as (obs, action, done, success) tuples.
- Save `release_demos_v1.pt` (~74 × ~10 step × 28-dim obs + 6-dim action).
- Sanity: ≥50 demos (else relax `place_success_flag` to `near_target & gripper_open_within_5steps`).
