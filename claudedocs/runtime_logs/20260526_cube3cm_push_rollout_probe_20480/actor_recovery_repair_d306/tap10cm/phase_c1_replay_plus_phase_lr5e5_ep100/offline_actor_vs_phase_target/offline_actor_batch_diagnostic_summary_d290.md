# D290 Offline Actor Batch Diagnostic

- verdict: `D290_OFFLINE_ACTOR_BATCH_DIAGNOSTIC_PASS`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- dataset count: `2`
- aggregate samples: `6960`
- aggregate MSE/MAE/cosine: `0.031194288283586502` / `0.10164138674736023` / `0.8587358593940735`

## Batch Rows

- claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/d304_failed6_d256_replay_dataset.pt: samples `3480`, mse `0.016482677310705185`, cosine `0.8983221650123596`, pred_abs_max `1.6340113878250122`, target_clip `0.11465517431497574`
- claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_target_dataset/phase_action_repair_dataset_d306_phase_c1_recovery065to010_clip085_smooth045.pt: samples `3480`, mse `0.04590589553117752`, cosine `0.8191496133804321`, pred_abs_max `1.3900212049484253`, target_clip `0.0`

## Issues

- none
