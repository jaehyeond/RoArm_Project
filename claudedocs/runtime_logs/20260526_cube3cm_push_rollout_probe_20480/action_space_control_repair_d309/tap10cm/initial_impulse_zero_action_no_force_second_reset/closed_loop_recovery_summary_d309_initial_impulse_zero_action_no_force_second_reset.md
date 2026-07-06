# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `87..916` / `31`
- samples: `160`
- actor rollout contact/useful/reaction: `0.53125` / `0.53125` / `0.53125`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.00041354604763910174` / `0.011983133852481842`
- actor-vs-recovery MSE/MAE/cosine: `0.10025770887732506` / `0.20364776849746705` / `0.4005880832672119`
- actor-vs-recorded MSE/MAE/cosine: `0.0872633159160614` / `0.1899031549692154` / `0.16648569703102112`
- actor action abs mean/max: `0.17771630883216857` / `1.0`
- recovery action abs mean/max: `0.13106319308280945` / `1.0`
- recovery clip rate mean/max: `0.04249999970197678` / `0.0625`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action_no_force_second_reset/closed_loop_recovery_dataset_d309_initial_impulse_zero_action_no_force_second_reset.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action_no_force_second_reset/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_zero_action_no_force_second_reset/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
