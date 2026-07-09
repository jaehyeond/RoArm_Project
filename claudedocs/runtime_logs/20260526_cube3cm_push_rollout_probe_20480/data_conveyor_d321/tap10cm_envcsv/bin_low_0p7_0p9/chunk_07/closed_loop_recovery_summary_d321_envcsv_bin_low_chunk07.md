# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `3..981` / `94`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.8999999761581421` / `1.0`
- actor rollout overshoot: `0.09999999403953552`
- max XY mean/max: `0.011141885071992874` / `0.06092732399702072`
- actor-vs-recovery MSE/MAE/cosine: `0.08357376609768333` / `0.2243274961566103` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.08357376605272293` / `0.22432750463485718` / `0.0`
- actor action abs mean/max: `0.2243274961566103` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5058483005276528` / `0.6920000314712524`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_07/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk07.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_07/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk07.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
