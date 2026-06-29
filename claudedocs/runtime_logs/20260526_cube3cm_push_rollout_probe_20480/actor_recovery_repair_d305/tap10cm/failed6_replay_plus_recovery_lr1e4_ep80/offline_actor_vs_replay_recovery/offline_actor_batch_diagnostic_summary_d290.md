# D290 Offline Actor Batch Diagnostic

- verdict: `D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- dataset count: `2`
- aggregate samples: `6960`
- aggregate MSE/MAE/cosine: `0.10729696601629257` / `0.18814946711063385` / `0.8226829171180725`

## Batch Rows

- claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/d304_failed6_d256_replay_dataset.pt: samples `3480`, mse `0.04861883446574211`, cosine `0.790066659450531`, pred_abs_max `2.202512264251709`, target_clip `0.11465517431497574`
- claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_dataset_d304_failed6.pt: samples `3480`, mse `0.16597507894039154`, cosine `0.855299174785614`, pred_abs_max `1.776991844177246`, target_clip `0.5923371911048889`

## Issues

- none
