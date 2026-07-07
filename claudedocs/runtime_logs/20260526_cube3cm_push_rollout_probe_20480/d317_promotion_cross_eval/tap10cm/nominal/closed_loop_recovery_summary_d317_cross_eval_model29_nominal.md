# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_30it/model_29.pt`
- reset pose source: `env_hook`
- selected episodes: `90..972` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.004240675829350948` / `0.007709149736911058`
- actor-vs-recovery MSE/MAE/cosine: `0.017432297769420107` / `0.1018765556015845` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.01743229664862156` / `0.10187654942274094` / `0.0`
- actor action abs mean/max: `0.1018765556015845` / `0.8525850176811218`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5917133705690503` / `0.675000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/nominal/closed_loop_recovery_dataset_d317_cross_eval_model29_nominal.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
