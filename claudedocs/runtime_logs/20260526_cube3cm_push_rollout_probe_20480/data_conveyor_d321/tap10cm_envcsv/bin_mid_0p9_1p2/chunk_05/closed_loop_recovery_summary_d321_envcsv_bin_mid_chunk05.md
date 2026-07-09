# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `11..995` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9899999499320984` / `1.0`
- actor rollout overshoot: `0.009999999776482582`
- max XY mean/max: `0.012555818073451519` / `0.04747290164232254`
- actor-vs-recovery MSE/MAE/cosine: `0.06640776530357784` / `0.20751755008923597` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.06640776246786118` / `0.2075175642967224` / `0.0`
- actor action abs mean/max: `0.20751755008923597` / `0.9178016185760498`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.48713105519960537` / `0.7160000205039978`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_05/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk05.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_05/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk05.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
