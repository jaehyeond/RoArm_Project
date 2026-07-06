# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `87..916` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0021548650693148375` / `0.01644902676343918`
- actor-vs-recovery MSE/MAE/cosine: `0.8464430445291359` / `0.7043629230353339` / `-0.13976975398555655`
- actor-vs-recorded MSE/MAE/cosine: `0.19537314772605896` / `0.2707095444202423` / `0.3663065433502197`
- actor action abs mean/max: `0.26066929006884837` / `1.0`
- recovery action abs mean/max: `0.6687994833393344` / `1.0`
- recovery clip rate mean/max: `0.6650431137275079` / `0.875`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_actor_no_force_second_reset/closed_loop_recovery_dataset_d309_actor_fresh32_no_force_second_reset.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_actor_no_force_second_reset/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_space_control_repair_d309/tap10cm/fresh32_random_actor_no_force_second_reset/step_env.csv`

## Issues

- actor-vs-recovery MSE too high: 0.8464430445291359

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
