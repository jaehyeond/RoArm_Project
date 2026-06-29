# D289 Actor Distillation From D256 Replay

- verdict: `D289_D256_REPLAY_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- distilled checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_distill_d289_160eps_manualreset_smoke/tap10cm/model_actor_d256_replay_d289.pt`
- samples train/val: `83520` / `9280`
- oracle replay contact/useful/reaction: `0.9875` / `0.54375` / `0.9875`
- oracle replay overshoot: `0.45625`
- oracle max XY mean/max: `0.08392387628555298` / `10.880388259887695`
- target action abs mean/max: `0.2084288028900608` / `1.0`
- target action clip rate mean/max: `0.12464870930777798` / `0.35625001788139343`
- initial val MSE/MAE/cosine: `0.6558146476745605` / `0.438286155462265` / `-0.3129933178424835`
- final val MSE/MAE/cosine: `0.04516212269663811` / `0.13073980808258057` / `0.7887061834335327`

## Issues

- oracle replay useful rate below 0.99: 0.54375
- oracle replay overshoot high: 0.45625
- final val MSE above threshold: 0.04516212269663811
- final val cosine below threshold: 0.7887061834335327

## Interpretation

This is supervised actor warm-start from D256 recorded-action oracle replay. It does not train PPO and does not use the D257 MLP teacher as an action target.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics with the same action contract.
