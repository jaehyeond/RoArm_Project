# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `51..991` / `95`
- samples: `58000`
- actor rollout contact/useful/reaction: `0.9899999499320984` / `0.949999988079071` / `0.9899999499320984`
- actor rollout overshoot: `0.04999999701976776`
- max XY mean/max: `0.03774702548980713` / `2.125263214111328`
- actor-vs-recovery MSE/MAE/cosine: `0.08041491172244322` / `0.22060884869304195` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.08041491359472275` / `0.22060883045196533` / `0.0`
- actor action abs mean/max: `0.22060884869304195` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5020758880310339` / `0.7400000095367432`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_08/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk08.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_08/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk08.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
