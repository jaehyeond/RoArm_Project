# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_30it/model_29.pt`
- reset pose source: `env_hook`
- selected episodes: `31..935` / `29`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.125` / `1.0`
- actor rollout overshoot: `0.875`
- max XY mean/max: `0.030706288293004036` / `0.04188673570752144`
- actor-vs-recovery MSE/MAE/cosine: `0.04646368690059874` / `0.17535286121841134` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.04646368697285652` / `0.17535287141799927` / `0.0`
- actor action abs mean/max: `0.17535286121841134` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.7914116466177049` / `0.875`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/friction_0p8_0p6/closed_loop_recovery_dataset_d317_cross_eval_model29_friction_0p8_0p6.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
