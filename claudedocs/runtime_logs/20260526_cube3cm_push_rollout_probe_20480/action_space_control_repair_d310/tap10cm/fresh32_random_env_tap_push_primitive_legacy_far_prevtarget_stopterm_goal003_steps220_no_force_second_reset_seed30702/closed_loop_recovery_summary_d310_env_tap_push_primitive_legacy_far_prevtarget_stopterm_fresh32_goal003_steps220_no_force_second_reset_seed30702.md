# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `29..999` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.003755867714062333` / `0.005437684245407581`
- actor-vs-recovery MSE/MAE/cosine: `0.571021490672539` / `0.5815851464353758` / `0.29121788643653795`
- actor-vs-recorded MSE/MAE/cosine: `0.25322890281677246` / `0.3495902121067047` / `0.3138168454170227`
- actor action abs mean/max: `0.2911713442925749` / `1.0`
- recovery action abs mean/max: `0.6506721347120815` / `1.0`
- recovery clip rate mean/max: `0.5893642400535916` / `0.668749988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d310/tap10cm/fresh32_random_env_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30702/closed_loop_recovery_dataset_d310_env_tap_push_primitive_legacy_far_prevtarget_stopterm_fresh32_goal003_steps220_no_force_second_reset_seed30702.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d310/tap10cm/fresh32_random_env_tap_push_primitive_legacy_far_prevtarget_stopterm_goal003_steps220_no_force_second_reset_seed30702/closed_loop_recovery_envs_d310_env_tap_push_primitive_legacy_far_prevtarget_stopterm_fresh32_goal003_steps220_no_force_second_reset_seed30702.csv`
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
