# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/model_0.pt`
- reset pose source: `manual`
- selected episodes: `13..935` / `5`
- samples: `2900`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `7.656301750103012e-05` / `0.00030884623993188143`
- actor-vs-recovery MSE/MAE/cosine: `0.6908976563465268` / `0.6131636484943587` / `0.0008526466251451298`
- actor-vs-recorded MSE/MAE/cosine: `0.1194179430603981` / `0.20330607891082764` / `0.7233831882476807`
- actor action abs mean/max: `0.24015341833683437` / `1.0`
- recovery action abs mean/max: `0.5918118008749623` / `1.0`
- recovery clip rate mean/max: `0.5468965386008394` / `0.8399999737739563`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_closed_loop_recovery_hard_episodes/closed_loop_recovery_dataset_d303.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_closed_loop_recovery_hard_episodes/closed_loop_recovery_envs_d303.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_closed_loop_recovery_hard_episodes/closed_loop_recovery_step_envs_d303.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
