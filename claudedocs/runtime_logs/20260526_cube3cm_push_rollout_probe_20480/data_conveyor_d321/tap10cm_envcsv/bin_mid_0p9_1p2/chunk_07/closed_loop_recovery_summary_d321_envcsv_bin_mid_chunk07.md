# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `2..972` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9599999785423279` / `1.0`
- actor rollout overshoot: `0.03999999910593033`
- max XY mean/max: `0.01630036160349846` / `0.14059022068977356`
- actor-vs-recovery MSE/MAE/cosine: `0.0734753758485975` / `0.2088487991742019` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.07347537577152252` / `0.20884880423545837` / `0.0`
- actor action abs mean/max: `0.2088487991742019` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5059276075383391` / `0.7300000190734863`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_07/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk07.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_07/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk07.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
