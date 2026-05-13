# Path D Phase D.3 v2 — CLEAN rate measurement → PATH D FAIL

**Date**: 2026-05-17 (continuation of 5/15 evening)
**Owner**: pjhyun9711@gmail.com
**Outcome**: PATH D FAIL at the user-specified <10% CLEAN gate. Pivot recommended.

---

## What we did

| Step | Change | File | New md5 |
|---|---|---|---|
| 1 | Capture `gripper_q` at the exact step `_place_success_flag` rises (per env) | `roarm_rl/eval_release_bc.py` | `1c9c7c0c` |
| 2 | Save `gripper_q_at_success`, `n_clean`, `clean_thresh_rad`, `grasp_thresh_rad` to output `.pt` | same | same |
| 3 | Rebuild PASS gate on CLEAN rate (gq@s < 0.4 rad) | same | same |
| 4 | Bump launch script md5 + output dir to `pathD_eval_bc_v2` | `launch_pathD_eval_bc.sh` | `82b8fa6f` |
| 5 | Rsync to B200, md5 verify, tmux re-launch | — | — |
| 6 | Pull `eval_metrics.pt` → `claudedocs/pathD_data/eval_metrics_v2.pt` (`fe16b447`) | — | — |
| 7 | Extended analyzer with CLEAN/DIRTY split + histogram | `claudedocs/pathD_data/analyze_eval_v2.py` | — |

LOC change vs v1: +21 lines net (~5 LOC capture + ~16 LOC summary/save), HARD RULE #14 fail-fast guards preserved, no `2>&1` used.

## Headline numbers

| Metric | v1 (5/15) | v2 (5/17) | Δ |
|---|---:|---:|---:|
| Nominal success | 175/256 = 68.36% | 175/256 = 68.36% | bit-identical (reproducible) |
| CLEAN success (gq@s < 0.4 rad) | unknown | **24/256 = 9.38%** | — |
| DIRTY (counter-path) | unknown | 151/256 = 58.98% | — |
| Counter-path inflation factor | — | **7.3×** | — |
| Δ CLEAN vs P6v14a baseline (7.81%) | — | **+1.56pp** | marginal |

**Decision (user matrix)**: CLEAN 9.38% < 10% → **PATH D FAIL** → procedural release pivot.

## Sub-evidence supporting the verdict

1. **gq@s distribution clearly bimodal** — 24 envs in [0, 0.4) rad (CLEAN bucket), 151 envs in [0.6, 1.6) rad (gripper still ≥34° closed). Only 4 envs in transition band [0.4, 0.6). Threshold 0.4 rad is well-placed.
2. **CLEAN successes are early, DIRTY are late**:
   - CLEAN-only success_step: mean=49, median=25, range [17, 156]
   - DIRTY-only success_step: mean=90, median=88, range [70, 181]
   - **All 19 Early (s<50) successes are CLEAN. All 5 mid-range CLEAN (s>=50, s<157) exist, but rare.**
3. **GOR (episode mean gripper_open_rate) correlates with success type**:
   - CLEAN success: 0.228
   - DIRTY success: 0.140
   - Failure: 0.095
   - CLEAN envs do open gripper for sustained periods (~46 steps avg), consistent with true release. DIRTY envs barely open at all.
4. **What BC learned**: "hover near target with gripper closed" → triggers `_place_counter ≥ 50` (env line 793-794) → DIRTY success fires. The 9.4% CLEAN rate is the actual release skill BC absorbed from the contaminated D.1 demos.

## Why D.1 demo source was contaminated

D.1 generated demos by rolling out P6v14a model on the same env. P6v14a iter 0 stage4=0.37 reflects **release skill transfer from P6v14a's `curriculum_pregrasp` (release-only) task** — but D.1 only filtered on `gripper_open@s` (Filter1). It did not exclude `_place_counter ≥ 50` counter-path demos. So 4 of the 20 D.1 demos (success_step 158/135/115/197) are themselves counter-path artifacts.

The BC then learned a mix: 16 real-release demos + 4 hover-with-closed-gripper demos. With only 80% clean teacher data and limited BC capacity (28→64→6 Tanh), the policy collapsed toward the safer "hover" mode (smaller MSE on average).

## What v2 settles vs what remains open

| Question | Answer |
|---|---|
| Is 68% real release? | **No** — 86% are counter-path. |
| Is BC learning anything? | Yes, but the actual release skill rate (9.4%) is only marginally better than P6v14a baseline (7.8%). |
| Does BC need more capacity? | **Unlikely the bottleneck** — gripper_corr 0.9645 on train data, val=0.115. The data itself is contaminated. |
| Does BC need more demos? | **Not from the same P6v14a rollout source** — they'll carry the same bias. |
| Real fix? | Procedural (IK + scripted) demos: IK +5cm above target → scripted gripper-open over 10 steps → gravity-driven sponge drop → place success. |

## HARD RULE compliance this session

- **#4**: No new external citations introduced (carried other-agent SkillGen/MimicGen reference from 5/15 evening — still unverified, treated as direction-of-inquiry only).
- **#8**: Will archive 5/15 evening MEMORY entry → `MEMORY_archive_20260517.md` to keep Recent Sessions ≤ 5.
- **#11**: `/half-clone` rejected; this session continues via MEMORY + continuation prompt.
- **#14**: fail-fast guards (`set -e`, root unset, whoami, hostname, md5) preserved in `launch_pathD_eval_bc.sh`. No `2>&1`.
- **#15**: cu128 env (`isaacsim_5_1`) used as before — eval ran fine on B200 sm_100 in ~5s wall (256 envs × 200 step inference).
- **#17**: State-only RL boundary respected — eval uses 28-dim obs only.
- **#18**: User's "비판적/분석적/의심 step-by-step" mandate followed. Headline 68% **not accepted at face value** — CLEAN rate becomes the truth.
- **#26**: 5/19 deadline still binding for #2 Pure RL infrastructure. PATH D was the auxiliary BC channel under #21 hybrid path; FAIL here does not affect #26 Pure RL.

## Files inventory (5/17)

- `roarm_rl/eval_release_bc.py` (md5 `1c9c7c0c`, 204 LOC) — v2 with gq@s capture
- `launch_pathD_eval_bc.sh` (md5 `82b8fa6f`, 56 LOC)
- `claudedocs/pathD_data/eval_metrics_v2.pt` (md5 `fe16b447`, 6478 bytes) — adds `gripper_q_at_success`, `n_clean`, `clean_thresh_rad`, `grasp_thresh_rad`
- `claudedocs/pathD_data/eval_v2_run.out` — full B200 stdout
- `claudedocs/pathD_data/analyze_eval_v2.py` (143 LOC) — CLEAN/DIRTY split analyzer
- B200: `logs/roarm_rl/pathD_eval_bc_v2/{run.out, run.err, eval_metrics.pt}`

## Next session — recommended branches (user picks)

The user's 5/15 evening continuation prompt specified the FAIL branch as **"SkillGen/MimicGen procedural release demo pivot"**. Three reasonable starting points:

**A. Procedural release-only demos (smallest scope, fastest test of hypothesis)**
- Single-env Isaac Sim launch with same `curriculum_pregrasp` init (sponge attached, near target).
- Scripted action sequence: hold pose for 5 steps → open gripper command (action[5] → +1.0) for 10 steps → release attachment, let gravity settle → 5 step settle.
- Generate ~50 clean demos at varied init poses (perturb base/sponge pose ±5cm).
- Train BC on these, eval same gate. Hypothesis: CLEAN rate jumps to 80%+. ETA: ~half day script + ~3h B200.

**B. Extend D.1 filter to exclude counter-path demos and re-train**
- Re-run D.1 with stricter filter: `success_step < 50` AND `gripper_q@success < 0.4 rad`. Likely yields 4-8 demos from 256 trials.
- May be too few demos for BC. Useful as a control to isolate "demo contamination" vs "demo count" as root cause. ETA: ~1h.

**C. Skip BC entirely, go to T2 Pure RL (#26)**
- Reframe Path D as a falsified hypothesis ("BC from RL rollouts in rare-event tasks does not work"), document in paper. Move B200 to #26 Pure RL infrastructure setup (5/19 deadline).

**Recommendation**: A first (test procedural-demo hypothesis cleanly), then C if A FAILs. B is informative but lower-impact.
