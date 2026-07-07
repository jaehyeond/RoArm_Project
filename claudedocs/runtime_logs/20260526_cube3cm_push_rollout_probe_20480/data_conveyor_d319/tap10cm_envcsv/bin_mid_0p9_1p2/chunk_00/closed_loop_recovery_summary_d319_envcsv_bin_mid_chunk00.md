# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `11..981` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.949999988079071` / `1.0`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.01780831813812256` / `0.2959577143192291`
- actor-vs-recovery MSE/MAE/cosine: `0.0834875988966689` / `0.2211803317840757` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0834876000881195` / `0.22118033468723297` / `0.0`
- actor action abs mean/max: `0.2211803317840757` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5362931319191281` / `0.7420000433921814`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_00/closed_loop_recovery_dataset_d319_envcsv_bin_mid_chunk00.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_00/closed_loop_recovery_envs_d319_envcsv_bin_mid_chunk00.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
