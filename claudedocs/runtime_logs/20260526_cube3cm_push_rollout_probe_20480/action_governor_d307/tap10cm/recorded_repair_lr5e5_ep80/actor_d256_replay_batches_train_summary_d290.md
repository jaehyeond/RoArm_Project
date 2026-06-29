# D290 Actor Training From D256 Replay Batches

- verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_PASS_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- output checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_lr5e5_ep80/model_actor_d256_replay_batches_d290.pt`
- dataset count: `2`
- samples train/val: `6264` / `696`
- selected episode range/count: `29..991` / `6`
- aggregate oracle contact/useful/reaction: `1.0` / `1.0` / `1.0`
- aggregate oracle overshoot: `0.0`
- final val MSE/MAE/cosine: `0.030511973425745964` / `0.0931229442358017` / `0.8834095001220703`

## Issues

- none

## Interpretation

This trains only the actor from clean separately collected replay-action batches. It is not PPO.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics.
