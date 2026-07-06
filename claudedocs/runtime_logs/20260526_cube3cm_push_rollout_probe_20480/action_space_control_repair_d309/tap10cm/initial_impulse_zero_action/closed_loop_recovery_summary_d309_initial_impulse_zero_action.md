# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `35..993` / `32`
- samples: `160`
- actor rollout contact/useful/reaction: `0.5625` / `0.40625` / `0.5625`
- actor rollout overshoot: `0.1875`
- max XY mean/max: `0.004548152908682823` / `0.026208428665995598`
- actor-vs-recovery MSE/MAE/cosine: `0.14834947437047957` / `0.24049243330955505` / `0.2695200443267822`
- actor-vs-recorded MSE/MAE/cosine: `0.13862080872058868` / `0.23591943085193634` / `0.089238241314888`
- actor action abs mean/max: `0.22741520702838897` / `1.0`
- recovery action abs mean/max: `0.12944600731134415` / `1.0`
- recovery clip rate mean/max: `0.05` / `0.0625`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action/closed_loop_recovery_dataset_d309_initial_impulse_zero_action.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
