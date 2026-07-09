# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `55..993` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.949999988079071` / `1.0`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.010106092318892479` / `0.06629106402397156`
- actor-vs-recovery MSE/MAE/cosine: `0.07170401946480932` / `0.2119350517618245` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.07170401513576508` / `0.21193504333496094` / `0.0`
- actor action abs mean/max: `0.2119350517618245` / `0.9891343712806702`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4762310526654895` / `0.7080000042915344`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_08/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk08.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_08/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk08.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
