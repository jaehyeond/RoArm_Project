# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `11..997` / `95`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9799999594688416` / `1.0`
- actor rollout overshoot: `0.019999999552965164`
- max XY mean/max: `0.012106786482036114` / `0.10925743728876114`
- actor-vs-recovery MSE/MAE/cosine: `0.06355962933137499` / `0.20323630668993653` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0635596290230751` / `0.20323629677295685` / `0.0`
- actor action abs mean/max: `0.20323630668993653` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4824551913156656` / `0.7120000123977661`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_03/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk03.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_03/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk03.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
