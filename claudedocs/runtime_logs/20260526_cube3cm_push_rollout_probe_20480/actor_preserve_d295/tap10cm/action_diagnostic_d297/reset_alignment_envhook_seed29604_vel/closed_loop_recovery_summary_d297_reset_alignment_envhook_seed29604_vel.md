# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `env_hook`
- selected episodes: `42..862` / `31`
- samples: `32`
- actor rollout contact/useful/reaction: `0.5` / `0.375` / `0.5`
- actor rollout overshoot: `0.125`
- max XY mean/max: `0.0035313679836690426` / `0.02510492503643036`
- actor-vs-recovery MSE/MAE/cosine: `0.0013725863536819816` / `0.025026995688676834` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0013725862372666597` / `0.025026997551321983` / `0.0`
- actor action abs mean/max: `0.025026995688676834` / `0.1465715765953064`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.0` / `0.0`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/reset_alignment_envhook_seed29604_vel/closed_loop_action_dataset_reset_alignment_envhook_seed29604_vel_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/reset_alignment_envhook_seed29604_vel/closed_loop_action_envs_reset_alignment_envhook_seed29604_vel_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/reset_alignment_envhook_seed29604_vel/closed_loop_action_steps_reset_alignment_envhook_seed29604_vel_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
