# Session 2026-06-29 - Cube10cm top-view D305 closed-loop recovery repair

## Scope

- Continue from D304 without running PPO.
- Goal: repair D304 failed collection envs through supervised closed-loop
  recovery/action diagnostics.
- No long PPO, tiny PPO trace gate, PPO ladder, partial actor preservation, real
  actor update, render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or
  RoArm deployment was performed.

## Inputs

- D304 failed episodes: `561,265,341,991,536,29`
- Source actor:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/model_0.pt`
- D304 failed6 D256 replay dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/d304_failed6_d256_replay_dataset.pt`
- D304 failed6 closed-loop recovery dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_dataset_d304_failed6.pt`

## Candidate 1

- Training: non-PPO supervised actor fit.
- Datasets: D304 failed6 D256 replay + D304 failed6 recovery.
- Hyperparameters: `lr=1e-4`, `epochs=80`, `batch_size=512`.
- Checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- Checkpoint sha256:
  `07043ec3d75f70f08dbd827d029578d1c5a1d3be2d1a208035672fcd17b43b1d`
- Train summary:
  - final val MSE: `0.10983244329690933`
  - final val cosine: `0.8255895972251892`
  - verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_WARN_NEEDS_ROLLOUT_EVAL`
- Offline diagnostic:
  - verdict: `D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS`
  - aggregate MSE/cosine: `0.10729696601629257` /
    `0.8226829171180725`

## Candidate 1 Fresh Probes

Default probe contract still stops after useful seen. It is valid for contact
recovery but not for displacement promotion.

| episode | useful max | overshoot max | cap max | safe-bin |
| ---: | ---: | ---: | ---: | --- |
| 561 | 1.0 | 0.0 | 0.333333 | false |
| 265 | 1.0 | 0.0 | 0.166667 | true |
| 341 | 1.0 | 0.0 | 0.166667 | true |
| 991 | 1.0 | 0.0 | 0.333333 | false |
| 536 | 1.0 | 0.0 | 0.333333 | false |
| 29 | 1.0 | 0.0 | 0.166667 | true |

D304-like contract: `--no-tap_stop_after_useful_seen`,
`--tap_stop_after_disp_m 0.003`, `--no-tap_useful_terminate`,
`--no-tap_overshoot_terminate`.

| episode | useful max | overshoot max | cap max | max XY mean trace |
| ---: | ---: | ---: | ---: | ---: |
| 561 | 1.0 | 0.0 | 0.333333 | 0.0000129262 |
| 265 | 1.0 | 0.0 | 0.566667 | 0.0000127921 |
| 991 | 1.0 | 0.0 | 0.666667 | 0.0000156090 |

Interpretation: contact/useful recovered, but displacement is too small and cap
pressure is too high.

## Candidate 1 Closed-Loop Recovery

- Verdict:
  `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- Useful/overshoot: `1.0` / `0.0`
- Mean/max XY: `0.0011089661857113242m` /
  `0.0035506743006408215m`
- Actor-vs-recovery MSE/cosine: `0.890877366065979` /
  `-0.08254306018352509`
- Recovery clip rate mean/max: `0.6602873916962537` /
  `0.9333333969116211`
- Joint cap mean/max: `0.09865900926805776` /
  `0.2777777910232544`

This is a partial improvement over D304 recovery, but still not a PPO gate.

## Candidate 2

- Training: candidate-1 source actor, D304 failed6 D256 replay plus candidate-1
  closed-loop recovery dataset.
- Hyperparameters: `lr=5e-5`, `epochs=80`, `batch_size=512`.
- Checkpoint sha256:
  `fb1fc7137574face1c0f4c55beb16fdaf71012ca98a6c059e39901e32e1fd880`
- Train summary:
  - final val MSE: `0.095654`
  - final val cosine: `0.768090`
- Closed-loop recovery after candidate-2:
  - verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
  - useful/overshoot: `0.833333` / `0.0`
  - mean/max XY: `0.0000167370m` / `0.0000381783m`
  - actor-vs-recovery MSE: `0.70577472448349`
  - cap max: `0.111111119389534`

Candidate-2 reduced MSE and cap, but lost useful coverage and displacement.
It is not better for the task.

## Action Clip Control

Candidate-1 with `exec_action_clip_abs=0.75` under the D304-like contract:

| episode | useful max | overshoot max | cap max | action max |
| ---: | ---: | ---: | ---: | ---: |
| 561 | 1.0 | 0.0 | 0.333333 | 0.75 |
| 265 | 1.0 | 0.0 | 0.500000 | 0.75 |
| 991 | 1.0 | 0.0 | 0.666667 | 0.75 |

Simple magnitude clipping does not solve cap pressure.

## Decision

- D305 partially repaired the D304 no-contact failure mode.
- It did not meet the displacement/cap contract needed before another PPO gate.
- Candidate-1 is the best D305 behavioral candidate, but it is not promoted.
- Candidate-2 is rejected for task behavior despite better MSE/cap.
- Do not run long PPO, tiny PPO trace gate, PPO ladder, partial actor
  preservation, or real actor update from D305.
- Next work should be phase/displacement-aware non-PPO action repair:
  - separate approach/contact/push labels or phases;
  - add cap/smoothness penalties to supervised actor repair;
  - preserve minimum 1-3mm displacement while keeping overshoot zero;
  - then re-run fresh one-bin/direct-reset diagnostics.

## Verdict

`D305_CLOSED_LOOP_RECOVERY_REPAIR_PARTIAL_CONTACT_RESTORED_NO_PPO_PROMOTION`
