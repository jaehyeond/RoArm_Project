# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `41..993` / `92`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9699999690055847` / `1.0`
- actor rollout overshoot: `0.029999999329447746`
- max XY mean/max: `0.013530425727367401` / `0.2763620913028717`
- actor-vs-recovery MSE/MAE/cosine: `0.0715178045187274` / `0.21402710917694814` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.07151780277490616` / `0.21402710676193237` / `0.0`
- actor action abs mean/max: `0.21402710917694814` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.49406553994350394` / `0.6780000329017639`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_01/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk01.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_01/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk01.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
