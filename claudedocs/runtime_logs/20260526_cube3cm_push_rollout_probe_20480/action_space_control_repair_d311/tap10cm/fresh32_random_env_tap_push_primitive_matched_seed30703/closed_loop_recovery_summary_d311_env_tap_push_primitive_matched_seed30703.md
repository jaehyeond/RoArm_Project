# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `13..950` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.003819497302174568` / `0.006012641359120607`
- actor-vs-recovery MSE/MAE/cosine: `0.5232678945079959` / `0.5668464478233765` / `0.3626544280202481`
- actor-vs-recorded MSE/MAE/cosine: `0.240557461977005` / `0.3385956883430481` / `0.21512867510318756`
- actor action abs mean/max: `0.2450478816083793` / `1.0`
- recovery action abs mean/max: `0.6228415325990525` / `1.0`
- recovery clip rate mean/max: `0.555258636639036` / `0.6312500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d311/tap10cm/fresh32_random_env_tap_push_primitive_matched_seed30703/closed_loop_recovery_dataset_d311_env_tap_push_primitive_matched_seed30703.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d311/tap10cm/fresh32_random_env_tap_push_primitive_matched_seed30703/closed_loop_recovery_envs_d311_env_tap_push_primitive_matched_seed30703.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
