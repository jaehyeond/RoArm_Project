# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `1..978` / `90`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9399999976158142` / `1.0`
- actor rollout overshoot: `0.05999999865889549`
- max XY mean/max: `0.008954375982284546` / `0.054284367710351944`
- actor-vs-recovery MSE/MAE/cosine: `0.06588126416478692` / `0.20624564749927357` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.06588126718997955` / `0.20624566078186035` / `0.0`
- actor action abs mean/max: `0.20624564749927357` / `0.9366870522499084`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.4746103646073105` / `0.6980000138282776`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_04/closed_loop_recovery_dataset_d321_envcsv_bin_low_chunk04.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv/bin_low_0p7_0p9/chunk_04/closed_loop_recovery_envs_d321_envcsv_bin_low_chunk04.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
