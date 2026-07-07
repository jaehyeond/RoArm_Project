# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `2..960` / `90`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.949999988079071` / `1.0`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.008866867981851101` / `0.05705149844288826`
- actor-vs-recovery MSE/MAE/cosine: `0.060498812084953335` / `0.19872365524542743` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.060498811304569244` / `0.1987236738204956` / `0.0`
- actor action abs mean/max: `0.19872365524542743` / `0.824271559715271`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.48483794920371265` / `0.7080000042915344`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_low_0p7_0p9/chunk_00/closed_loop_recovery_dataset_d319_envcsv_bin_low_chunk00.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_low_0p7_0p9/chunk_00/closed_loop_recovery_envs_d319_envcsv_bin_low_chunk00.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
