# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `87..916` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0038021435029804707` / `0.010024919174611568`
- actor-vs-recovery MSE/MAE/cosine: `0.5305353644143405` / `0.572946263644202` / `0.41693621345892035`
- actor-vs-recorded MSE/MAE/cosine: `0.2355116754770279` / `0.3356180191040039` / `0.2160256952047348`
- actor action abs mean/max: `0.2642661955079128` / `1.0`
- recovery action abs mean/max: `0.6688665128730494` / `1.0`
- recovery clip rate mean/max: `0.595010787659678` / `0.6500000357627869`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30701/closed_loop_recovery_dataset_d309_tap_push_primitive_legacy_far_prevtarget_stopterm_fresh32_goal003_steps220_no_force_second_reset_seed30701.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30701/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30701/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
