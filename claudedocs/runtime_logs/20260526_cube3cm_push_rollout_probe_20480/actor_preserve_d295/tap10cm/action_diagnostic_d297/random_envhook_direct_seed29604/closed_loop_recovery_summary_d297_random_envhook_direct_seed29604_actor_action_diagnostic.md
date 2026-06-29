# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `env_hook`
- selected episodes: `13..935` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.001000076299533248` / `0.00349514395929873`
- actor-vs-recovery MSE/MAE/cosine: `0.6838679458970075` / `0.634866558657638` / `0.1021301406515955`
- actor-vs-recorded MSE/MAE/cosine: `0.12106165289878845` / `0.20830674469470978` / `0.6979314088821411`
- actor action abs mean/max: `0.2114034340299409` / `1.0`
- recovery action abs mean/max: `0.6334390185002623` / `1.0`
- recovery clip rate mean/max: `0.6100215628373854` / `0.8062500357627869`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_direct_seed29604/closed_loop_action_dataset_random_envhook_direct_seed29604_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_direct_seed29604/closed_loop_action_envs_random_envhook_direct_seed29604_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_direct_seed29604/closed_loop_action_steps_random_envhook_direct_seed29604_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
