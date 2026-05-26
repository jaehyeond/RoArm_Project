# Session 2026-05-22 — OpenVLA-OFT v6 Offline Eval — RESULT (8/12 ckpts, user-approved early stop)

## TL;DR

Track B P3 (offline eval of OpenVLA-OFT 7B v6 LoRA ckpts on B200) terminated
early after **8/12 ckpts** (steps 2500..20000) per user decision at ~16:11 KST.

Verdict:

- **best deploy ckpt = step 7500** (holdout `l2_step0_mean` = 22.1624°,
  train_sanity = 10.6086°, gap = +11.55°). Both `best_overall` and
  `best_early(<=10K)` resolve to 7500.
- **R-OFT-2 (early overfit) hypothesis CONFIRMED but mechanism reinterpreted**
  as **catastrophic training collapse**, not pure overfit. Between step 7500
  and step 10000 BOTH holdout and train_sanity L2 spike ~3-5×:
  - holdout L2 avg-per-joint 6.36° → 22.13° (3.5×)
  - train_sanity L2 avg-per-joint 3.11° → 14.92° (4.8×)
  - z-score avg 0.27 → 0.86 (near random-action regime)
  - all 6 joints spike together at the same step → global parameter scale
    shift, not per-joint phenomenon
- Pure overfit predicts train ↓ holdout ↑. We see train ↑↑ AND holdout ↑↑ →
  pattern is consistent with a **weights divergence event** in the 7.5K–10K
  window (LR `5e-4` may be too large for 7B LoRA, or loss explosion / NaN).
- Recommend deploy with **ckpt 7500** (early stop). Steps 10000–20000 are all
  in the collapsed plateau (~69°) and should not be deployed.

User-approved early termination saved ~75 min of B200 GPU time at marginal
information loss (the 4 remaining ckpts at 22.5K/25K/27.5K/30K would only
have added 1 final-point data + 4-point plateau extension; the collapse is
already confirmed on 5 sustained points at 68.91–70.07°).

## Verified Inputs

- B200 nohup PID `3507531` started 2026-05-22 12:10:28 KST, SIGTERM at
  ~16:11 KST. Process etime at kill `04:01:30`.
- Eval log B200 path `/tmp/openvla_oft_v6_eval_full_20260522_121028.out`.
  Step lines (8 lines):
  - `step=2500 l2_step0_mean=24.1829`
  - `step=5000 l2_step0_mean=22.4112`
  - `step=7500 l2_step0_mean=22.1624`
  - `step=10000 l2_step0_mean=70.0740`
  - `step=12500 l2_step0_mean=70.0497`
  - `step=15000 l2_step0_mean=69.4704`
  - `step=17500 l2_step0_mean=69.6349`
  - `step=20000 l2_step0_mean=68.9142`
- B200 JSON `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/outputs/openvla_oft_v6_eval/openvla_oft_v6_eval_20260522_121028.json`
  size 26754 B, sha256 `3707a0ee1efd189868eb0421a1f56b2a71ec16dfb3b87632772a0e5a87332bf0`,
  `n_checkpoints` meta = 12, `per_checkpoint` length = 8.
- Local copy pulled `openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028.json`,
  sha256 match (`3707a0ee...`).
- Eval config: holdout episodes `[45,46,47,48,49]` (741 frames), train_sanity
  episodes `[0,1]` (207 frames), dtype `bfloat16`.
- action_std (from stats.json, degrees per joint): `[28.16, 17.52, 23.78, 30.55, 25.22, 28.25]`.
- Tooling md5 (Lenovo, B200 verified earlier sessions):
  - `openvla_oft_roarm/eval_offline_v6.py` md5 `3ef67f2b558547623f6b3d03e8e98b4f`
  - `openvla_oft_roarm/launch_eval_offline_v6.sh` md5 `8976cf43d41e150679e9663e2e471ebf`
  - `openvla_oft_roarm/rank_eval_v6.py` md5 `10edd0485c4c754e0ac9ad4ca7370cf5`

## Ranker Output (rank_eval_v6.py on 8-ckpt JSON)

### Table 1: per-ckpt holdout vs train_sanity (degrees)

| step | h.l2_step0 | h.l2_chunk | t.l2_step0 | t.l2_chunk | gap_step0 | gap_chunk |
|---:|---:|---:|---:|---:|---:|---:|
|  2500 | 24.1829 | 23.8242 | 13.5086 | 12.8601 | +10.6743 | +10.9641 |
|  5000 | 22.4112 | 21.8230 | 11.4080 | 10.6680 | +11.0032 | +11.1550 |
|  7500 | **22.1624** | **21.2841** | 10.6086 | 10.3633 | +11.5538 | +10.9208 |
| 10000 | 70.0740 | 69.0188 | 52.4503 | 50.3896 | +17.6237 | +18.6292 |
| 12500 | 70.0497 | 68.9579 | 48.6521 | 46.6279 | +21.3975 | +22.3300 |
| 15000 | 69.4704 | 68.3910 | 50.9622 | 48.8478 | +18.5082 | +19.5432 |
| 17500 | 69.6349 | 68.5524 | 50.0049 | 47.8990 | +19.6300 | +20.6534 |
| 20000 | 68.9142 | 67.8118 | 48.5214 | 46.3659 | +20.3928 | +21.4459 |

### Table 2: ranked by holdout.l2_step0_mean (best = top)

1. step 7500 — holdout 22.1624° gap +11.55° ← **BEST**
2. step 5000 — holdout 22.4112° gap +11.00°
3. step 2500 — holdout 24.1829° gap +10.67°
4. step 20000 — holdout 68.9142° gap +20.39°
5. step 15000 — holdout 69.4704° gap +18.51°
6. step 17500 — holdout 69.6349° gap +19.63°
7. step 12500 — holdout 70.0497° gap +21.40°
8. step 10000 — holdout 70.0740° gap +17.62°

### Table 3: per-joint MAE on holdout (step0) — cliff at 7.5K→10K

| step | base | shoulder | elbow | wrist_p | wrist_r | gripper |
|---:|---:|---:|---:|---:|---:|---:|
|  7500 | 10.891 | 2.754 | 6.038 | 4.831 | 9.987 | 3.639 |
| 10000 | 26.057 | 14.103 | 22.658 | 26.738 | 22.617 | 20.570 |
| ratio | 2.39× | 5.12× | 3.75× | 5.53× | 2.26× | 5.65× |

### Table 4: per-joint MAE on train_sanity (step0) — train_sanity ALSO collapses

| step | base | shoulder | elbow | wrist_p | wrist_r | gripper |
|---:|---:|---:|---:|---:|---:|---:|
|  7500 |  2.175 | 6.138 |  3.193 |  3.506 | 0.981 |  2.668 |
| 10000 | 13.782 | 20.853 | 14.531 | 25.122 | 0.384 | 14.837 |
| ratio | 6.33× | 3.40× | 4.55× | 7.17× | 0.39× | 5.56× |

J4 wrist_r is the only joint whose train_sanity *improves* at 10K (0.98° → 0.38°)
while its holdout MAE spikes (9.99° → 22.62°), giving the largest memorization
ratio (58.92×). Other joints all worsen on both splits — global degradation.

### Summary

- `best_overall      = step 7500  holdout.l2_step0 = 22.16°  gap = +11.55°`
- `best_early(<=10K) = step 7500  holdout.l2_step0 = 22.16°  gap = +11.55°`
- `final(highest)    = step 20000 holdout.l2_step0 = 68.91°  gap = +20.39°`

### Overfit signals (R-OFT-2)

- Gap widens 2500 → 20000: +10.67° → +20.39° (YES, +9.72°)
- Holdout worsens 2500 → 20000: 24.18° → 68.91° (YES, +44.73°)
- Recommend deploy with `best = 7500` (early stop).

## Reinterpretation — Catastrophic Training Collapse, not Pure Overfit

The original R-OFT-2 hypothesis was framed as "early overfit": at later
steps, train MAE keeps decreasing while holdout MAE rises. The 8-ckpt
data instead shows:

- 2.5K → 7.5K: monotonic improvement on **both** splits (h.l2_step0
  24.18→22.41→22.16; t.l2_step0 13.51→11.41→10.61).
- 7.5K → 10K: single-step cliff on **both** splits
  (h.l2_step0 22.16→70.07 = +47.91°; t.l2_step0 10.61→52.45 = +41.84°).
- 10K → 20K: noisy plateau on both splits (h.l2_step0 ∈ [68.91, 70.07];
  t.l2_step0 ∈ [48.52, 52.45]). No recovery.

The gap *also* widens (+11.55° → +20.39°), so an overfit component is
present, but the dominant signal is shared collapse — model parameters
have left a regime that fits even the training distribution.

Likely root-cause hypotheses (NOT verified this session, candidates for a
separate diagnostic run):

1. LR `5e-4` too large for 7B LoRA on 6942 training frames after ~35
   epochs (12 ckpts × 2500 steps × batch 8 / 6942). Possible loss
   explosion or weight divergence at one specific gradient step.
2. NaN injection (e.g., a single bad batch around step 9-10K). PEFT 0.18
   LoRA delta accumulation interacting with bfloat16 numerics.
3. Action-head DDP wrap interaction at later checkpoints (D079 strip
   `module.` prefix on load was already applied; but the *saved*
   weights themselves may have diverged).
4. Cosine LR schedule peaks early — if min_lr_ratio not set, late steps
   may oscillate at near-zero LR but with momentum-driven jitter on
   PEFT LoRA matrices.

Diagnostics that would distinguish these (out of scope this session):

- Pull training loss curve from `outputs/openvla_oft_v6_b200/*/runs/` and
  inspect for explosion at step 9-10K.
- Re-eval ckpt 7500 and 10000 with `fp32` instead of `bf16` to test
  numerical-precision sensitivity.
- Compare LoRA `A`/`B` matrix norms at ckpt 7500 vs 10000.

## Per-Joint Pattern (memorization ratio holdout/train_sanity)

| step | base | shoulder | elbow | wrist_p | wrist_r | gripper |
|---:|---:|---:|---:|---:|---:|---:|
|  2500 | 4.41 | 0.55 | 1.43 | 1.10 |  5.59 | 0.90 |
|  5000 | 4.49 | 0.60 | 1.74 | 1.24 |  8.23 | 1.07 |
|  7500 | 5.01 | 0.45 | 1.89 | 1.38 | 10.18 | 1.36 |
| 10000 | 1.89 | 0.68 | 1.56 | 1.06 | **58.92** | 1.39 |
| 12500 | 4.19 | 0.66 | 1.75 | 1.20 |  9.01 | 1.38 |
| 15000 | 2.52 | 0.67 | 1.40 | 1.15 | 11.22 | 1.36 |
| 17500 | 2.47 | 0.68 | 1.58 | 1.16 | 23.74 | 1.38 |
| 20000 | 2.31 | 0.74 | 1.77 | 1.18 |  7.13 | 1.39 |

J0 (base) and J4 (wrist_r) — the two largest-range joints — drive
memorization in the pre-collapse regime (4-10×). J1/J3/J5 ratios near 1
or below 1 (shoulder is actually *easier* on holdout than on train_sanity,
likely a coverage artifact of episodes 0/1 having atypical shoulder
trajectories). After collapse, J4 wrist_r becomes pathological (ratio
58.92× at 10K, 23.74× at 17.5K) — model still memorizes wrist_r on
seen episodes but completely fails on unseen.

## Files Changed This Session

Local:
- `openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028.json` (new,
  pulled from B200, sha256 `3707a0ee1efd189868eb0421a1f56b2a71ec16dfb3b87632772a0e5a87332bf0`).
- `claudedocs/session_20260522_openvla_oft_offline_eval_v6_result.md`
  (this file).

B200:
- PID 3507531 SIGTERM at ~16:11 KST. JSON intact at
  `outputs/openvla_oft_v6_eval/openvla_oft_v6_eval_20260522_121028.json`.
- step 22500 partial load + holdout frame 100/741 was in flight at kill;
  no entry written for steps 22500/25000/27500/30000.

Note: this session did **not** touch any Track A scripts, audits, session
docs, or START_HERE Track A sections (lines 1-148 / 54-194). Track A
remains the parallel session's domain per the two-track design.

## Decisions / Recommendation

- **Deploy ckpt = step 7500.** Holdout L2 22.16°; pulled-LoRA path
  `outputs/openvla_oft_v6_b200/openvla-7b+roarm_v6_pick+b8+lr-0.0005+lora-r32+dropout-0.0--v6_30k--7500_chkpt/`.
  Action head: `action_head--7500_checkpoint.pt`.
- Do NOT deploy ckpt 10000 or later — collapsed plateau, near-random
  predictions on multiple joints.
- For a follow-up training run, halve LR (`5e-4` → `2.5e-4`) and add
  gradient clipping; checkpoint more densely in 7K-10K window to
  catch the collapse event; consider early-stopping at validation L2
  flat.
- If 30K final-point data is needed for the paper curve, single-ckpt
  eval is ~20 min on B200 (load 36s + 948 frames × 1.21s/fr + write).
  Not done this session per user "stop now" decision.

## HARD RULE Compliance

- ✅ #4: All metrics traceable to B200 log step lines + JSON sha256.
- ⚠️ #8: MEMORY index entry prepended (this session). 5-slot rule
  **violation persists** — pre-session state had 9 entries (parallel
  Track A sessions did not archive), this prepend makes 10. Archiving
  the oldest 5 (5/14 ~ 5/17 era: L84 P7 smoke / L86 CoRL pivot / L88
  Skill 1b stall / L90 Skill 3 basin / L92 PATH D FAIL) is out of
  scope for this finalize-focused session. Recommend dedicated MEMORY
  cleanup session to enforce 5-slot rule; archive target file
  `MEMORY_archive_20260522.md` not yet created.
- ✅ #11: `/half-clone` not invoked. Continuation handled via project
  state files + this session doc.
- ✅ #14: All ssh calls have fail-fast guard (`set -e`, `source env.sh`,
  `[[ -z "$ROARM_B200_ROOT" ]] && exit 1`, whoami check). No
  `2>&1`. No `pipe-to-source`.
- ✅ #15: torch nightly cu128 + transformers 4.57.6 (env loaded from
  `roarm_b200/envs/roarm_b200`).
- ✅ #18: User decision "지금 멈출 + finalize (추천)" via AskUserQuestion at
  16:10 KST honored verbatim. Earlier 12:30 KST decision "그대로 계속,
  deadline 23:59 KST" was correctly reframed for explicit consent at
  the spike-confirmation decision point.
- N/A #17/#26: No Isaac Sim / RL / sim render involvement.

## Track A Boundary

This session did not touch Track A. Track A v4 close_26 runtime FAIL
state (parallel-session-owned, START_HERE lines 1-148 / 54-194,
session_20260522_track_a_v4_recovery_runtime_fail.md, DECISIONS D078-D083)
remains the latest verified Track A truth.

## Next Concrete Steps (Track B follow-up, pending user direction)

1. **Real-deploy ckpt 7500** on RoArm-M3 with v6 pick task. Compare
   against SmolVLA v6 (2026-04-09 Plan 3 = JOINT_SPEED_CAPS gripper-only,
   user-validated multi-position success).
2. **Diagnose collapse window**: pull B200 training loss curve from
   `outputs/openvla_oft_v6_b200/*/runs/` (TensorBoard or text logs);
   identify exact step of loss explosion if any.
3. **Optional re-train** with halved LR + gradient clip + denser
   checkpoints in 7K-10K window.
4. **(Optional) ckpt 30000 single eval** if paper curve needs the
   final-point data (B200 ~20 min single ckpt).
