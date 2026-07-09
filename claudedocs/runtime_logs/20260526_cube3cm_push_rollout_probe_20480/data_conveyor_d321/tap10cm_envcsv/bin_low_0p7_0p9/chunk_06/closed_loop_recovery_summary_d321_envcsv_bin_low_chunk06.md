# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `9..999` / `94`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.949999988079071` / `1.0`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.009519503451883793` / `0.060902006924152374`
- actor-vs-recovery MSE/MAE/cosine: `0.06355801939900066` / `0.20317287748229915` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.06355801969766617` / `0.20317289233207703` / `0.0`
- actor action abs mean/max: `0.20317287748229915` / `0.8916105628013611`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.48765864290280975` / `0.7100000381469727`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_06/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk06.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_06/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk06.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
