# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `29..999` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.90625` / `1.0`
- actor rollout overshoot: `0.09375`
- max XY mean/max: `0.010159309953451157` / `0.02502155862748623`
- actor-vs-recovery MSE/MAE/cosine: `0.8652942900770697` / `0.6940357456433361` / `0.07810723246717505`
- actor-vs-recorded MSE/MAE/cosine: `0.26405689120292664` / `0.36896514892578125` / `0.5510848760604858`
- actor action abs mean/max: `0.4059475920837501` / `1.0`
- recovery action abs mean/max: `0.6552845259486088` / `1.0`
- recovery clip rate mean/max: `0.6063254390987728` / `0.675000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_goal003_steps220_no_force_second_reset_seed30702/closed_loop_recovery_dataset_d309_tap_push_primitive_legacy_far_prevtarget_fresh32_goal003_steps220_no_force_second_reset_seed30702.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_goal003_steps220_no_force_second_reset_seed30702/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_goal003_steps220_no_force_second_reset_seed30702/step_env.csv`

## Issues

- actor-vs-recovery MSE too high: 0.8652942900770697

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
