# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `35..993` / `32`
- samples: `160`
- actor rollout contact/useful/reaction: `0.625` / `0.46875` / `0.625`
- actor rollout overshoot: `0.1875`
- max XY mean/max: `0.004507537465542555` / `0.025839945301413536`
- actor-vs-recovery MSE/MAE/cosine: `0.19064656049013137` / `0.27539390325546265` / `-0.018395352363586425`
- actor-vs-recorded MSE/MAE/cosine: `0.16145281493663788` / `0.25346559286117554` / `0.16255943477153778`
- actor action abs mean/max: `0.26037302017211916` / `1.0`
- recovery action abs mean/max: `0.1423176884651184` / `1.0`
- recovery clip rate mean/max: `0.052500000596046446` / `0.06875000149011612`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_tap_push_primitive_goal003_steps220/closed_loop_recovery_dataset_d309_initial_impulse_tap_push_primitive_goal003_steps220.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_tap_push_primitive_goal003_steps220/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/initial_impulse_tap_push_primitive_goal003_steps220/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
