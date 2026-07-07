# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `29..961` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9799999594688416` / `1.0`
- actor rollout overshoot: `0.019999999552965164`
- max XY mean/max: `0.00819053128361702` / `0.02571866661310196`
- actor-vs-recovery MSE/MAE/cosine: `0.059886158389392596` / `0.20071372788014083` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.059886153787374496` / `0.20071373879909515` / `0.0`
- actor action abs mean/max: `0.20071372788014083` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4811551887750754` / `0.7080000042915344`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_low_0p7_0p9/chunk_01/closed_loop_recovery_dataset_d319_envcsv_bin_low_chunk01.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_low_0p7_0p9/chunk_01/closed_loop_recovery_envs_d319_envcsv_bin_low_chunk01.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
