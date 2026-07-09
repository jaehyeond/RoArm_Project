# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `9..978` / `96`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9699999690055847` / `1.0`
- actor rollout overshoot: `0.029999999329447746`
- max XY mean/max: `0.013429004698991776` / `0.05284982547163963`
- actor-vs-recovery MSE/MAE/cosine: `0.06539603131341523` / `0.20264260827981193` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0653960332274437` / `0.20264260470867157` / `0.0`
- actor action abs mean/max: `0.20264260827981193` / `0.8667848706245422`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.49275174975025887` / `0.7280000448226929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_06/closed_loop_recovery_dataset_d321_envcsv_bin_mid_chunk06.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_mid_0p9_1p2/chunk_06/closed_loop_recovery_envs_d321_envcsv_bin_mid_chunk06.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
