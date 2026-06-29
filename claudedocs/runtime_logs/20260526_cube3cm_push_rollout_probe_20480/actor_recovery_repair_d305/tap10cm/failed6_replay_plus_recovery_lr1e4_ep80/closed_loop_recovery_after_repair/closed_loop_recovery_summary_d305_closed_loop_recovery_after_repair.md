# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0011089661857113242` / `0.0035506743006408215`
- actor-vs-recovery MSE/MAE/cosine: `0.8908774285320321` / `0.7222268879413605` / `-0.08254305558905893`
- actor-vs-recorded MSE/MAE/cosine: `0.24620631337165833` / `0.32677310705184937` / `0.5558361411094666`
- actor action abs mean/max: `0.32486629845767184` / `1.0`
- recovery action abs mean/max: `0.6616541367152642` / `1.0`
- recovery clip rate mean/max: `0.6602873916962537` / `0.9333333969116211`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/closed_loop_recovery_after_repair/closed_loop_recovery_dataset_d305_after_repair.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/closed_loop_recovery_after_repair/closed_loop_recovery_envs_d305_after_repair.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/closed_loop_recovery_after_repair/closed_loop_recovery_step_envs_d305_after_repair.csv`

## Issues

- actor-vs-recovery MSE too high: 0.8908774285320321

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
