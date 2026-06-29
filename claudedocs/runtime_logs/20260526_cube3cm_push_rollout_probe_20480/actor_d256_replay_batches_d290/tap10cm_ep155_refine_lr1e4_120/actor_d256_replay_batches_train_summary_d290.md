# D290 Actor Training From D256 Replay Batches

- verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_PASS_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- output checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155_refine_lr1e4_120/model_actor_d256_replay_batches_d290.pt`
- dataset count: `5`
- samples train/val: `83520` / `9280`
- selected episode range/count: `1..999` / `160`
- aggregate oracle contact/useful/reaction: `1.0` / `1.0` / `1.0`
- aggregate oracle overshoot: `0.0`
- final val MSE/MAE/cosine: `0.00429362989962101` / `0.02726930007338524` / `0.9730817079544067`

## Issues

- none

## Interpretation

This trains only the actor from clean separately collected replay-action batches. It is not PPO.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics.
