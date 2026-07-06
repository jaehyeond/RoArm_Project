# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `80..931` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0037014579866081476` / `0.005943730939179659`
- actor-vs-recovery MSE/MAE/cosine: `0.5316295141900151` / `0.5609567031392764` / `0.38935700525882944`
- actor-vs-recorded MSE/MAE/cosine: `0.26567938923835754` / `0.3494895100593567` / `0.17671209573745728`
- actor action abs mean/max: `0.26819150975809014` / `1.0`
- recovery action abs mean/max: `0.6564422814753549` / `1.0`
- recovery clip rate mean/max: `0.6103556060200107` / `0.699999988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_08_obs_noise_005/closed_loop_recovery_dataset_d314_matrix_obs_noise_005.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_08_obs_noise_005/closed_loop_recovery_envs_d314_matrix_obs_noise_005.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_08_obs_noise_005/closed_loop_recovery_steps_d314_matrix_obs_noise_005.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
