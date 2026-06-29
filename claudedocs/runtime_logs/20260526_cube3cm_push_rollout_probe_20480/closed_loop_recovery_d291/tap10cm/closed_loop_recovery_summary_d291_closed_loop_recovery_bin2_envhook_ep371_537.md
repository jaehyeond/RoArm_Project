# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `371..537` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0005855984054505825` / `0.011980446986854076`
- actor-vs-recovery MSE/MAE/cosine: `0.44061636433029416` / `0.5041938124866835` / `0.39812056485178143`
- actor action abs mean/max: `0.1254259844214238` / `1.0`
- recovery action abs mean/max: `0.5566316668329567` / `1.0`
- recovery clip rate mean/max: `0.5096551830013251` / `0.675000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/closed_loop_recovery_d291/tap10cm/closed_loop_recovery_bin2_envhook_ep371_537.pt`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
