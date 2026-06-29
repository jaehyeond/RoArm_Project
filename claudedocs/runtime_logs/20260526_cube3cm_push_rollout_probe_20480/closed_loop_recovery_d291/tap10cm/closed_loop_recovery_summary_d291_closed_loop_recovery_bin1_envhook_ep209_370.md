# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `209..370` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.00017283893248531967` / `0.0020340927876532078`
- actor-vs-recovery MSE/MAE/cosine: `0.5957295808314892` / `0.6278639827308984` / `0.15520472455095372`
- actor action abs mean/max: `0.103545839467953` / `1.0`
- recovery action abs mean/max: `0.6374615843183008` / `1.0`
- recovery clip rate mean/max: `0.6378017369914671` / `0.8500000238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/closed_loop_recovery_d291/tap10cm/closed_loop_recovery_bin1_envhook_ep209_370.pt`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
