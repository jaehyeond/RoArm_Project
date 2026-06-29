# D290 Actor Training From D256 Replay Batches

- verdict: `D290_D256_REPLAY_BATCH_ACTOR_TRAIN_WARN_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/model_0.pt`
- output checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- dataset count: `2`
- samples train/val: `6264` / `696`
- selected episode range/count: `29..991` / `6`
- aggregate oracle contact/useful/reaction: `0.9166666865348816` / `0.9166666865348816` / `0.9166666865348816`
- aggregate oracle overshoot: `0.0`
- final val MSE/MAE/cosine: `0.10983244329690933` / `0.1917351484298706` / `0.8255895972251892`

## Issues

- aggregate oracle useful rate below 0.99: 0.9166666865348816

## Interpretation

This trains only the actor from clean separately collected replay-action batches. It is not PPO.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics.
