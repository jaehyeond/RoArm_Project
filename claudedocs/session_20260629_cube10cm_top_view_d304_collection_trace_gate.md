# Session 2026-06-29 - Cube10cm top-view D304 collection trace gate

## Scope

- Run one tiny no-success-terminate actor-preserved PPO collection trace gate.
- Purpose was diagnostic only: capture true PPO collection-path failed envs in a
  fresh JSONL file.
- No long PPO, PPO ladder, partial actor preservation, real actor update,
  render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm deployment
  was performed.

## PPO Runtime

- Env: `tap10cm`
- Actor checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- `num_envs=32`, `max_iterations=1`, `num_steps_per_env=580`, seed `29801`
- D256 reset: random frame-0 reset from
  `ppo_actor_prior_teacher_rows_d256.csv`
- Action contract:
  - `action_scale=0.04`
  - `max_joint_delta_per_step_rad=0.04`
  - `joint_target_lead_limit_rad=0.06`
  - `joint_delta_reference=joint_pos`
  - `tap_stop_after_disp_m=0.003`
  - `tap_success_terminate=False`
  - `bc_teacher_blend=0.0`
  - `bc_teacher_imitation_reward_scale=0.0`
  - `init_noise_std=0.005`
  - `actor_preserve_blend=1.0`

## PPO Result

- PPO clean exit.
- Actor preservation:
  - `max_pre_restore_delta=0.228326231`
  - `max_post_restore_delta=0.000000000`
- `model_0.pt` sha256:
  `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`
- `collection_final_env_trace_iter_0.jsonl` sha256:
  `38bc56857210d25cf46dc17db55f8843b2504afff15a49dcf275c98fe0848291`

## TensorBoard Gate

- Gate verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- Collection-final metrics:
  - contact/reaction: `0.84375`
  - useful: `0.8125`
  - success: `0.84375`
  - overshoot: `0.03125`
  - XY >=1mm: `0.625`
  - XY >=3mm: `0.5625`
  - mean/max XY: `0.0037104846m` / `0.0537344441m`
- Issues:
  - collection-final contact/reaction below `0.90`
  - collection-final useful below `0.90`
- Dashboard command:
  `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it --host 127.0.0.1 --port 6006`

## Collection Trace

The JSONL trace contains 32 rows and directly identifies the failed final envs.

| env | D256 episode | contact | useful | overshoot | max XY m | note |
| --- | ---: | --- | --- | --- | ---: | --- |
| 4 | 561 | false | false | false | 0.0042661247 | stop-after-displacement held despite no useful/contact |
| 5 | 265 | true | false | true | 0.0537344441 | true overshoot case |
| 14 | 341 | false | false | false | 0.0000117311 | no-contact tiny-motion case |
| 15 | 991 | false | false | false | 0.0000115872 | no-contact tiny-motion case |
| 22 | 536 | false | false | false | 0.0000115828 | no-contact tiny-motion case |
| 31 | 29 | false | false | false | 0.0000114695 | no-contact tiny-motion case |

## Failed-Episode Probes

Fresh one-bin probes with `action_noise_std=0.005`:

| episode | useful | overshoot | interpretation |
| ---: | ---: | ---: | --- |
| 561 | 0.0 | 0.0 | robust no-contact failure |
| 265 | 0.8 | 0.2 | stochastic overshoot-sensitive case |
| 341 | 0.8 | 0.0 | borderline partial case |
| 991 | 0.0 | 0.0 | robust no-contact failure |
| 536 | 1.0 | 0.0 | passes fresh one-bin; collection-path/stochastic miss |
| 29 | 1.0 | 0.0 | passes fresh one-bin; collection-path/stochastic miss |

Deterministic fresh one-bin probes:

| episode | useful | overshoot | cap max | interpretation |
| ---: | ---: | ---: | ---: | --- |
| 561 | 0.0 | 0.0 | 0.0 | no-contact persists |
| 265 | 0.0 | 0.0 | 0.833333 | cap-pressure no-contact when noise is removed |
| 991 | 0.0 | 0.0 | 0.833333 | cap-pressure no-contact persists |

## D256 Replay And Actor Matching

- D256 recorded-action replay for failed6 passed:
  - contact/useful: `1.0`
  - overshoot: `0.0`
  - mean/max XY: `0.0097861877m` / `0.0161318127m`
  - target action abs mean/max: `0.22222358` / `1.0`
- Offline actor-vs-D256 comparison passed:
  - MSE: `0.0061870781`
  - MAE: `0.0377806723`
  - cosine: `0.9509615898`
  - pred abs mean/max: `0.21952645` / `1.38647449`

## Closed-Loop Recovery

- Closed-loop failed6 recovery verdict:
  `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- Actor execution:
  - contact/useful: `0.833333`
  - overshoot: `0.0`
  - mean/max XY: `0.0023242724m` / `0.0070422743m`
- Actor-vs-recorded:
  - MSE/MAE/cosine: `0.29715565` / `0.32562143` / `0.41405916`
- Actor-vs-recovery:
  - MSE/MAE/cosine: `1.08494043` / `0.80379063` / `-0.18368588`
- Recovery action clip rate mean/max:
  `0.71080464` / `0.96666670`

## Interpretation

- D304 achieved the diagnostic goal: failed envs are now captured from the real
  PPO collection path, not reconstructed from stale saved-checkpoint eval.
- D304 does not promote the policy. Collection-final contact/useful are below
  `0.90`, and actor preservation was full.
- D256 replay success proves safe actions exist for the failed episodes.
- Offline actor-vs-D256 success proves static imitation on recorded rows is not
  the main missing piece.
- Closed-loop recovery failure shows the actual blocker: actor recovery/stability
  once the rollout leaves the exact recorded state-action path.

## Decision

- No long PPO.
- No PPO ladder.
- No partial actor preservation or real actor update.
- Do not lower the `0.90` collection-final contact/useful threshold as a
  promotion standard based on D304.
- Next work is non-PPO closed-loop recovery/action repair:
  - aggregate failed-state recovery data, especially ep561/ep265/ep991;
  - or add a pre-contact action projection/constraint;
  - then re-run fresh one-bin/direct-reset diagnostics;
  - only after that consider one new tiny PPO trace gate.

## Verdict

`D304_COLLECTION_TRACE_GATE_FAIL_NO_PROMOTION_CLOSED_LOOP_RECOVERY_REPAIR_NEXT`
