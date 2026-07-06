# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.25` / `0.0` / `0.25`
- actor rollout overshoot: `1.0`
- max XY mean/max: `3.7688376903533936` / `11.989521980285645`
- actor-vs-recovery MSE/MAE/cosine: `1.088470640971229` / `0.7185194991785905` / `0.04183167178697627`
- actor-vs-recorded MSE/MAE/cosine: `0.7039543390274048` / `0.6423366069793701` / `0.1266978234052658`
- actor action abs mean/max: `0.6511809188744118` / `1.0`
- recovery action abs mean/max: `0.6027672714328971` / `1.0`
- recovery clip rate mean/max: `0.5585775967823855` / `0.7562500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_07_friction_high/closed_loop_recovery_dataset_d314_matrix_friction_high.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_07_friction_high/closed_loop_recovery_envs_d314_matrix_friction_high.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_07_friction_high/closed_loop_recovery_steps_d314_matrix_friction_high.csv`

## Issues

- actor-vs-recovery MSE too high: 1.088470640971229

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
