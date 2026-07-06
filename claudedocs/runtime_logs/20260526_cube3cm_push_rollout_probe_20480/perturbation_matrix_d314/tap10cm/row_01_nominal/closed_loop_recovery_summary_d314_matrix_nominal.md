# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0037114338483661413` / `0.00575611786916852`
- actor-vs-recovery MSE/MAE/cosine: `0.49547951451406397` / `0.5339426565016138` / `0.41859195068125327`
- actor-vs-recorded MSE/MAE/cosine: `0.24731779098510742` / `0.33392754197120667` / `0.33848074078559875`
- actor action abs mean/max: `0.2896744529234952` / `1.0`
- recovery action abs mean/max: `0.6566250455662095` / `1.0`
- recovery clip rate mean/max: `0.6097737162679049` / `0.65625`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_01_nominal/closed_loop_recovery_dataset_d314_matrix_nominal.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_01_nominal/closed_loop_recovery_envs_d314_matrix_nominal.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_01_nominal/closed_loop_recovery_steps_d314_matrix_nominal.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
