# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- reset pose source: `manual`
- selected episodes: `154..736` / `8`
- samples: `4640`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0012648596893996` / `0.003277832642197609`
- actor-vs-recovery MSE/MAE/cosine: `0.6157773454639467` / `0.601127570414723` / `0.2823666429853645`
- actor-vs-recorded MSE/MAE/cosine: `0.10553111881017685` / `0.19538317620754242` / `0.6818239092826843`
- actor action abs mean/max: `0.20322808127143774` / `1.0`
- recovery action abs mean/max: `0.6438802675247706` / `1.0`
- recovery clip rate mean/max: `0.6011207052089017` / `0.800000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/closed_loop_action_dataset_fail8_d297.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/closed_loop_action_envs_fail8_d297.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/closed_loop_action_steps_fail8_d297.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
