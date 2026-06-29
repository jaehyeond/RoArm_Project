# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `538..715` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `4.970424924977124e-05` / `0.0009295984636992216`
- actor-vs-recovery MSE/MAE/cosine: `0.4541968695871564` / `0.518001187278022` / `0.393645487179787`
- actor action abs mean/max: `0.11954327848065516` / `1.0`
- recovery action abs mean/max: `0.5701126591013423` / `1.0`
- recovery clip rate mean/max: `0.5175646602722077` / `0.731249988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/closed_loop_recovery_d291/tap10cm/closed_loop_recovery_bin3_envhook_ep538_715.pt`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
