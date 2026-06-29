# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `manual`
- selected episodes: `42..862` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.96875` / `0.96875` / `0.96875`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0020878061186522245` / `0.016737202182412148`
- actor-vs-recovery MSE/MAE/cosine: `0.7622268412573311` / `0.6642149004250251` / `0.09043509764436247`
- actor-vs-recorded MSE/MAE/cosine: `0.15239687263965607` / `0.22928592562675476` / `0.5978657603263855`
- actor action abs mean/max: `0.25047087821595626` / `1.0`
- recovery action abs mean/max: `0.653466286664379` / `1.0`
- recovery clip rate mean/max: `0.6524569086312991` / `0.793749988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_envhook_selected32/closed_loop_action_dataset_manual_envhook_selected32_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_envhook_selected32/closed_loop_action_envs_manual_envhook_selected32_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/manual_envhook_selected32/closed_loop_action_steps_manual_envhook_selected32_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
