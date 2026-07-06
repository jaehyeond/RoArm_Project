# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0039956895634531975` / `0.006443373393267393`
- actor-vs-recovery MSE/MAE/cosine: `0.5153314662776116` / `0.5637923151768487` / `0.29756398668255785`
- actor-vs-recorded MSE/MAE/cosine: `0.14449596405029297` / `0.26523348689079285` / `0.5545495748519897`
- actor action abs mean/max: `0.22119878129712467` / `1.0`
- recovery action abs mean/max: `0.6138572707515338` / `1.0`
- recovery clip rate mean/max: `0.538103454823381` / `0.625`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_04_mass_050/closed_loop_recovery_dataset_d314_matrix_mass050.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_04_mass_050/closed_loop_recovery_envs_d314_matrix_mass050.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_04_mass_050/closed_loop_recovery_steps_d314_matrix_mass050.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
