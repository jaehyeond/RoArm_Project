# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.34375` / `0.3125` / `0.34375`
- actor rollout overshoot: `0.03125`
- max XY mean/max: `0.009177335537970066` / `0.053462278097867966`
- actor-vs-recovery MSE/MAE/cosine: `0.4422978868129952` / `0.4799657862248092` / `0.46285211957734207`
- actor-vs-recorded MSE/MAE/cosine: `0.18541663885116577` / `0.29123467206954956` / `0.5803486108779907`
- actor action abs mean/max: `0.24898703822801854` / `1.0`
- recovery action abs mean/max: `0.5985746886581182` / `1.0`
- recovery clip rate mean/max: `0.5557435452376075` / `0.7562500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_06_friction_low/closed_loop_recovery_dataset_d314_matrix_friction_low.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_06_friction_low/closed_loop_recovery_envs_d314_matrix_friction_low.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_06_friction_low/closed_loop_recovery_steps_d314_matrix_friction_low.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
