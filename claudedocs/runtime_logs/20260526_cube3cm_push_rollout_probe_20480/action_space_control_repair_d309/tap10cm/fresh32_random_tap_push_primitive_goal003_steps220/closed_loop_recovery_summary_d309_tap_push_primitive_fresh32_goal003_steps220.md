# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `35..993` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.96875` / `0.8125` / `0.96875`
- actor rollout overshoot: `0.1875`
- max XY mean/max: `0.004970538429915905` / `0.025996508076786995`
- actor-vs-recovery MSE/MAE/cosine: `0.6151353610101445` / `0.5875798008051412` / `0.261323785186135`
- actor-vs-recorded MSE/MAE/cosine: `0.30982881784439087` / `0.3680802583694458` / `0.12306933104991913`
- actor action abs mean/max: `0.293738519862808` / `1.0`
- recovery action abs mean/max: `0.647032137472054` / `1.0`
- recovery clip rate mean/max: `0.6243319085317439` / `0.768750011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220/closed_loop_recovery_dataset_d309_tap_push_primitive_fresh32_goal003_steps220.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_goal003_steps220/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
