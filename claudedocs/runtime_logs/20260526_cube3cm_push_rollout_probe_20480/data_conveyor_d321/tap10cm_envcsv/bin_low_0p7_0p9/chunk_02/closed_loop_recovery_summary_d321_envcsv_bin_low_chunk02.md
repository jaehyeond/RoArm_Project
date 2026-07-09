# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `3..942` / `95`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9899999499320984` / `1.0`
- actor rollout overshoot: `0.009999999776482582`
- max XY mean/max: `0.007767918985337019` / `0.03715104982256889`
- actor-vs-recovery MSE/MAE/cosine: `0.0615119917607256` / `0.20101549471246785` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.061511993408203125` / `0.20101548731327057` / `0.0`
- actor action abs mean/max: `0.20101549471246785` / `0.969380795955658`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4998138145736322` / `0.7260000109672546`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_02/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk02.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_02/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk02.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
