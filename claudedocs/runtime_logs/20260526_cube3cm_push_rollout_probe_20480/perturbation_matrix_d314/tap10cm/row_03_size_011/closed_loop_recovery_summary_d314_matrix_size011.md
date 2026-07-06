# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `1..988` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.96875` / `1.0`
- actor rollout overshoot: `0.03125`
- max XY mean/max: `0.004119234159588814` / `0.026199331507086754`
- actor-vs-recovery MSE/MAE/cosine: `0.4770670017934051` / `0.5255700071310175` / `0.476969972387727`
- actor-vs-recorded MSE/MAE/cosine: `0.23699656128883362` / `0.3396185040473938` / `0.31343981623649597`
- actor action abs mean/max: `0.28560381421241265` / `1.0`
- recovery action abs mean/max: `0.6557291174500153` / `1.0`
- recovery clip rate mean/max: `0.6227370811883232` / `0.706250011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_03_size_011/closed_loop_recovery_dataset_d314_matrix_size011.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_03_size_011/closed_loop_recovery_envs_d314_matrix_size011.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_03_size_011/closed_loop_recovery_steps_d314_matrix_size011.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
