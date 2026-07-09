# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `15..993` / `92`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.949999988079071` / `1.0`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.010314232669770718` / `0.058166537433862686`
- actor-vs-recovery MSE/MAE/cosine: `0.059428142579593536` / `0.1946674104394584` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0594281367957592` / `0.19466739892959595` / `0.0`
- actor action abs mean/max: `0.1946674104394584` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4962448510212888` / `0.7320000529289246`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_09/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk09.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_09/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk09.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
