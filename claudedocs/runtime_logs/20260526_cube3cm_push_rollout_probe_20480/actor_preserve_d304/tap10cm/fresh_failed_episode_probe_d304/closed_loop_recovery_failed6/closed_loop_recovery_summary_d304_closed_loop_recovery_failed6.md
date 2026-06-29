# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/model_0.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `0.8333333730697632` / `0.8333333730697632` / `0.8333333730697632`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0023242724128067493` / `0.007042274344712496`
- actor-vs-recovery MSE/MAE/cosine: `1.08494046284769` / `0.803790689818561` / `-0.18368589867514573`
- actor-vs-recorded MSE/MAE/cosine: `0.2971556484699249` / `0.32562142610549927` / `0.4140591621398926`
- actor action abs mean/max: `0.32602654134790443` / `1.0`
- recovery action abs mean/max: `0.7031284950577237` / `1.0`
- recovery clip rate mean/max: `0.7108046368920597` / `0.9666666984558105`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_dataset_d304_failed6.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_envs_d304_failed6.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_step_envs_d304_failed6.csv`

## Issues

- actor-vs-recovery MSE too high: 1.08494046284769

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
