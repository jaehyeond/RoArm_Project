# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `manual`
- selected episodes: `87..260` / `5`
- samples: `2900`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0016546513652428985` / `0.003466173307970166`
- actor-vs-recovery MSE/MAE/cosine: `0.7969673933310771` / `0.7028968249608216` / `0.09449000608222001`
- actor-vs-recorded MSE/MAE/cosine: `0.12961597740650177` / `0.22529271245002747` / `0.6097518801689148`
- actor action abs mean/max: `0.22308129404896293` / `1.0`
- recovery action abs mean/max: `0.7057953592743083` / `1.0`
- recovery clip rate mean/max: `0.733517222234915` / `0.9599999785423279`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_randomfail5/closed_loop_action_dataset_manual_randomfail5_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_randomfail5/closed_loop_action_envs_manual_randomfail5_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_randomfail5/closed_loop_action_steps_manual_randomfail5_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
