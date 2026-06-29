# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `716..999` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `0.90625` / `0.90625` / `0.90625`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0002634782576933503` / `0.008024415001273155`
- actor-vs-recovery MSE/MAE/cosine: `0.634303804270618` / `0.617542855313112` / `0.17199780319753136`
- actor action abs mean/max: `0.14026873634550077` / `1.0`
- recovery action abs mean/max: `0.6042698280371982` / `1.0`
- recovery clip rate mean/max: `0.5802586293169136` / `0.887499988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/closed_loop_recovery_d291/tap10cm/closed_loop_recovery_bin4_envhook_ep716_999.pt`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
