# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `87..916` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0009764182614162564` / `0.008340612053871155`
- actor-vs-recovery MSE/MAE/cosine: `0.5969019138157882` / `0.59952348550332` / `0.3033591813086693`
- actor-vs-recorded MSE/MAE/cosine: `0.23487578332424164` / `0.32602134346961975` / `0.2263946235179901`
- actor action abs mean/max: `0.27822196229264656` / `1.0`
- recovery action abs mean/max: `0.6781585781481759` / `1.0`
- recovery clip rate mean/max: `0.6293319062233484` / `0.75`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220_no_force_second_reset/closed_loop_recovery_dataset_d309_tap_push_primitive_fresh32_goal003_steps220_no_force_second_reset.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220_no_force_second_reset/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220_no_force_second_reset/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
