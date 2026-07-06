# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.96875` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0037658787332475185` / `0.007346120662987232`
- actor-vs-recovery MSE/MAE/cosine: `0.5568813460782684` / `0.5447030133728323` / `0.43242780862954155`
- actor-vs-recorded MSE/MAE/cosine: `0.30173689126968384` / `0.37073877453804016` / `0.3291885256767273`
- actor action abs mean/max: `0.3578696287397681` / `1.0`
- recovery action abs mean/max: `0.7081642193655515` / `1.0`
- recovery clip rate mean/max: `0.7295797549589569` / `0.7875000238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_05_mass_100/closed_loop_recovery_dataset_d314_matrix_mass100.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_05_mass_100/closed_loop_recovery_envs_d314_matrix_mass100.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_05_mass_100/closed_loop_recovery_steps_d314_matrix_mass100.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
