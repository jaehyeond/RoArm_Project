# Session 2026-06-30 - Cube10cm top-view D306 phase-aware action repair

## Scope

- Continue from D305 without PPO.
- Goal: repair the D304/D305 failed-state actor behavior with
  phase/displacement-aware supervised targets and action projection checks.
- No long PPO, tiny PPO trace gate, PPO ladder, partial actor preservation, real
  actor update, render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or
  RoArm deployment was performed.

## Inputs

- D305 best behavioral actor:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- D304 failed6 D256 replay dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/d304_failed6_d256_replay_dataset.pt`
- D305 candidate-1 closed-loop recovery dataset:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/closed_loop_recovery_after_repair/closed_loop_recovery_dataset_d305_after_repair.pt`

## Candidate 1

- Built a phase-aware target dataset from D305 candidate-1 recovery data:
  early phase favors recovery action, late push phase favors recorded D256
  action.
- Target rewrite:
  - recovery weight `0.65 -> 0.10`
  - transition steps `40..260`
  - target clip `0.85`
  - smooth alpha `0.45`
- Dataset result:
  - target abs mean/max: `0.2988685668` / `0.8500000238`
  - target clip >=0.99: `0.0`
  - late push target-vs-recorded cosine: `0.9524435997`
- Supervised actor training:
  - output checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
  - sha256:
    `a407729e342197dffd2b6395dafff1a7b6c7cb55d252c013efbfb9817530427c`
  - val MSE/cosine: `0.0334460661` / `0.8521609306`
  - offline actor-vs-target diagnostic: PASS, MSE/cosine
    `0.0311942883` / `0.8587358594`

## Candidate 1 Rollout

D304 runtime contract was rechecked explicitly:

- `max_joint_delta_per_step_rad=0.04`
- `contact_joint_delta_scale=0.35`
- `fast_cube_joint_delta_scale=0.2`
- `action_smoothing_alpha=0.25`
- `tap_stop_after_disp_m=0.003`
- no useful/overshoot terminate

ep561 result:

- useful/overshoot: `1.0` / `0.0`
- cap max: `0.0`
- max XY/along: `0.000037225m` / `0.000027448m`

Interpretation: candidate-1 repaired no-contact but still produced only about
`0.037mm` displacement, far below the 1mm Tier-1 displacement gate.

## Per-Step Trace Finding

ep561 per-step trace showed the late push mismatch:

- in steps `300..579`, recorded D256 actions are large:
  - elbow abs mean `0.9646`
  - wrist_pitch abs mean `0.7303`
- candidate-1 actor is much weaker:
  - elbow abs mean `0.3992`
  - wrist_pitch abs mean `0.0960`

This explains why the actor contacts the cube but does not meaningfully push it.

## Candidate 2

- Collected D306 failed6 closed-loop states with candidate-1 under the D304
  runtime contract.
- Built a stronger late-push target:
  - recovery weight `0.50 -> 0.00`
  - transition steps `40..180`
  - target clip `1.0`
  - smooth alpha `0.80`
- The target dataset intentionally preserved D256 recorded-action clip pressure:
  target clip >=0.99 was `0.128352`, close to the original recorded-action clip
  rate.
- Supervised actor training:
  - output checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
  - sha256:
    `8f5d154f9ba76bc467e96f73ed3017e21dd6b8ead265d547c3cadc4ff30844b5`
  - val MSE/cosine: `0.0345205478` / `0.8549892306`

## Candidate 2 Action Projection Checks

ep561 under D304 runtime contract:

| variant | useful | overshoot | cap | max XY |
| --- | ---: | ---: | ---: | ---: |
| no projection | 1.0 | 1.0 | 0.0 | 0.041465m |
| exec clip 0.50 | 1.0 | 0.0 | 0.0 | 0.0000428m |
| exec clip 0.75 | 1.0 | 0.0 | 0.0 | 0.0000450m |
| contact slowdown proxy | 1.0 | 0.0 | 0.0 | 0.0000364m |

Interpretation:

- Candidate-2 can create displacement, but the unprojected action overshoots to
  about `41.5mm`.
- Simple global action clipping and contact slowdown prevent overshoot but also
  collapse displacement back below `0.05mm`.
- This is a threshold/impulse problem, not just an offline actor-fit problem.

## Decision

- D306 is not a PPO gate.
- Do not run long PPO, tiny PPO trace gate, PPO ladder, partial actor
  preservation, or real actor update from D306.
- Do not promote D306 as learned-policy success or RoArm readiness.
- Next work should be a non-PPO displacement-aware action governor:
  - use current displacement, cube velocity, and contact state;
  - shape a short push pulse instead of relying on scalar actor output alone;
  - brake before overshoot rather than only after `tap_stop_after_disp_m`;
  - require 1mm/3mm displacement tiers plus overshoot <=5% before any PPO gate.

## Verdict

`D306_PHASE_ACTION_REPAIR_BRACKETED_TINY_VS_OVERSHOOT_NO_PPO_PROMOTION`
