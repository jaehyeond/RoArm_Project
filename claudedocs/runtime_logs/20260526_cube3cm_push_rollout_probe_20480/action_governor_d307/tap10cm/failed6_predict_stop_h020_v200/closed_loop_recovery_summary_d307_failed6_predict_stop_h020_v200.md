# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.002726597711443901` / `0.00716980779543519`
- actor-vs-recovery MSE/MAE/cosine: `0.6222878609816062` / `0.531497665154266` / `-0.006627978775891122`
- actor-vs-recorded MSE/MAE/cosine: `0.11461852490901947` / `0.1816042810678482` / `0.743841826915741`
- actor action abs mean/max: `0.3056486247171616` / `1.0`
- recovery action abs mean/max: `0.5646198751549397` / `1.0`
- recovery clip rate mean/max: `0.5101724396363414` / `0.8333333730697632`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/closed_loop_dataset_d307.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
