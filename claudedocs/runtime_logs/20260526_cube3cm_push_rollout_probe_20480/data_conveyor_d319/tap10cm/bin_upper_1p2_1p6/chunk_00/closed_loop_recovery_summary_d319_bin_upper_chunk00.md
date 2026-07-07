# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `22..997` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `0.9699999690055847` / `0.1899999976158142` / `0.9699999690055847`
- actor rollout overshoot: `0.8100000023841858`
- max XY mean/max: `0.33313900232315063` / `11.135595321655273`
- actor-vs-recovery MSE/MAE/cosine: `0.13275872194689922` / `0.2738970438468045` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.1327587217092514` / `0.27389705181121826` / `0.0`
- actor action abs mean/max: `0.2738970438468045` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.6715758946916923` / `0.796000063419342`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm/bin_upper_1p2_1p6/chunk_00/closed_loop_recovery_dataset_d319_bin_upper_chunk00.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
