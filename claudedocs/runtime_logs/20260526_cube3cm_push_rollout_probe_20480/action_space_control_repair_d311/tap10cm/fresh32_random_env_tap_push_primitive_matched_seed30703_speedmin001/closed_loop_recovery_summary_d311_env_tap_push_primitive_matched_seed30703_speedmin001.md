# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `13..950` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0038173857610672712` / `0.0059450725093483925`
- actor-vs-recovery MSE/MAE/cosine: `0.5226914090200745` / `0.5675372862867241` / `0.3637013676383629`
- actor-vs-recorded MSE/MAE/cosine: `0.24024464190006256` / `0.33859023451805115` / `0.2160828709602356`
- actor action abs mean/max: `0.24542756895052975` / `1.0`
- recovery action abs mean/max: `0.6225983650327243` / `1.0`
- recovery clip rate mean/max: `0.555258636639036` / `0.6312500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d311/tap10cm/fresh32_random_env_tap_push_primitive_matched_seed30703_speedmin001/closed_loop_recovery_dataset_d311_env_tap_push_primitive_matched_seed30703_speedmin001.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d311/tap10cm/fresh32_random_env_tap_push_primitive_matched_seed30703_speedmin001/closed_loop_recovery_envs_d311_env_tap_push_primitive_matched_seed30703_speedmin001.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
