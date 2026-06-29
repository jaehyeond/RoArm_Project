# D289 Actor Distillation From D256 Replay

- verdict: `D289_D256_REPLAY_ACTOR_DISTILL_SUPERVISED_FIT_PASS_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- distilled checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_distill_d289/tap10cm/model_actor_d256_replay_d289.pt`
- samples train/val: `16704` / `1856`
- oracle replay contact/useful/reaction: `1.0` / `1.0` / `1.0`
- oracle replay overshoot: `0.0`
- oracle max XY mean/max: `0.005660976283252239` / `0.015575507655739784`
- target action abs mean/max: `0.2175050291521796` / `1.0`
- target action clip rate mean/max: `0.12980603701756174` / `0.35625001788139343`
- initial val MSE/MAE/cosine: `0.27974340319633484` / `0.3240380883216858` / `-0.19687886536121368`
- final val MSE/MAE/cosine: `0.007201300468295813` / `0.04466653987765312` / `0.9516013264656067`

## Issues

- none

## Interpretation

This is supervised actor warm-start from D256 recorded-action oracle replay. It does not train PPO and does not use the D257 MLP teacher as an action target.
Promotion still requires teacher-off frozen eval and D256 reset-bin diagnostics with the same action contract.
