# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.004013972356915474` / `0.006260571535676718`
- actor-vs-recovery MSE/MAE/cosine: `0.548915588508906` / `0.5551631696522236` / `0.30242301677032535`
- actor-vs-recorded MSE/MAE/cosine: `0.2861156165599823` / `0.3661239743232727` / `0.15197701752185822`
- actor action abs mean/max: `0.3203237529972504` / `1.0`
- recovery action abs mean/max: `0.6599682654928545` / `1.0`
- recovery clip rate mean/max: `0.6075215657651103` / `0.668749988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_02_size_009/closed_loop_recovery_dataset_d314_matrix_size009.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_02_size_009/closed_loop_recovery_envs_d314_matrix_size009.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_02_size_009/closed_loop_recovery_steps_d314_matrix_size009.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
