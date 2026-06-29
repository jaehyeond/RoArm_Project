# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `0.8333333730697632` / `0.8333333730697632` / `0.8333333730697632`
- actor rollout overshoot: `0.0`
- max XY mean/max: `1.6737019905121997e-05` / `3.8178251998033375e-05`
- actor-vs-recovery MSE/MAE/cosine: `0.7057747038466663` / `0.6840335594169025` / `-0.2501281668180158`
- actor-vs-recorded MSE/MAE/cosine: `0.16091778874397278` / `0.2481393963098526` / `0.5012983083724976`
- actor action abs mean/max: `0.12315632409319796` / `1.0`
- recovery action abs mean/max: `0.6488834828009893` / `1.0`
- recovery clip rate mean/max: `0.6259770478154051` / `0.9333333969116211`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/closed_loop_recovery_after_iter2/closed_loop_recovery_dataset_d305_iter2.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/closed_loop_recovery_after_iter2/closed_loop_recovery_envs_d305_iter2.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/closed_loop_recovery_after_iter2/closed_loop_recovery_step_envs_d305_iter2.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
