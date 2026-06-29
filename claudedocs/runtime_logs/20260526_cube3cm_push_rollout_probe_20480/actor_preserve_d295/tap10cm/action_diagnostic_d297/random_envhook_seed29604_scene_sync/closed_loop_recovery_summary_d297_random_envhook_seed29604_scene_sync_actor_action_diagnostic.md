# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `env_hook`
- selected episodes: `42..862` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.96875` / `0.8125` / `0.96875`
- actor rollout overshoot: `0.15625`
- max XY mean/max: `0.00674250815063715` / `0.0531604140996933`
- actor-vs-recovery MSE/MAE/cosine: `0.8474138520326835` / `0.6981397283038703` / `0.03862997549907144`
- actor-vs-recorded MSE/MAE/cosine: `0.17520633339881897` / `0.24701449275016785` / `0.5782856941223145`
- actor action abs mean/max: `0.28306489805851515` / `1.0`
- recovery action abs mean/max: `0.6632877843261793` / `1.0`
- recovery clip rate mean/max: `0.6740086333505039` / `0.8125`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_seed29604_scene_sync/closed_loop_action_dataset_random_envhook_seed29604_scene_sync_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_seed29604_scene_sync/closed_loop_action_envs_random_envhook_seed29604_scene_sync_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_seed29604_scene_sync/closed_loop_action_steps_random_envhook_seed29604_scene_sync_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
