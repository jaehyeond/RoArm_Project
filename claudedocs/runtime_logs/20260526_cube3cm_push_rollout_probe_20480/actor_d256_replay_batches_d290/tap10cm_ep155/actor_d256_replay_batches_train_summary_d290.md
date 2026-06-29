# D290 Actor Training From D256 Replay Batches

- verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_PASS_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- output checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- dataset count: `5`
- samples train/val: `83520` / `9280`
- selected episode range/count: `1..999` / `160`
- aggregate oracle contact/useful/reaction: `1.0` / `1.0` / `1.0`
- aggregate oracle overshoot: `0.0`
- final val MSE/MAE/cosine: `0.004835817962884903` / `0.031057896092534065` / `0.9703168272972107`

## Issues

- none

## Interpretation

This trains only the actor from clean separately collected replay-action batches. It is not PPO.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics.
