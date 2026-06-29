# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_lr5e5_ep80/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `1.541455276310444e-05` / `2.2765229005017318e-05`
- actor-vs-recovery MSE/MAE/cosine: `0.48597173486868367` / `0.5086350249190783` / `-0.023821402987433148`
- actor-vs-recorded MSE/MAE/cosine: `0.09393950551748276` / `0.18195313215255737` / `0.7715947031974792`
- actor action abs mean/max: `0.16118563138719263` / `1.0`
- recovery action abs mean/max: `0.5061081199858595` / `1.0`
- recovery clip rate mean/max: `0.4094827817046437` / `0.7666667103767395`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_failed6_predict_stop_h020_v200/closed_loop_dataset_d307.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_failed6_predict_stop_h020_v200/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_failed6_predict_stop_h020_v200/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
