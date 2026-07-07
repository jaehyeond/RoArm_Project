# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `3..999` / `96`
- samples: `58000`
- actor rollout contact/useful/reaction: `0.9799999594688416` / `0.1899999976158142` / `0.9799999594688416`
- actor rollout overshoot: `0.8100000023841858`
- max XY mean/max: `0.25061267614364624` / `11.140390396118164`
- actor-vs-recovery MSE/MAE/cosine: `0.11805885115574147` / `0.2570328443214811` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.11805884540081024` / `0.2570328414440155` / `0.0`
- actor action abs mean/max: `0.2570328443214811` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.6518138243029986` / `0.7820000648498535`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm/bin_upper_1p2_1p6/chunk_01/closed_loop_recovery_dataset_d319_bin_upper_chunk01.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
