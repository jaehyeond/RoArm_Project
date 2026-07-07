# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_WARN_REVIEW_BEFORE_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31713/model_225.pt`
- reset pose source: `env_hook`
- selected episodes: `9..981` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.25` / `1.0`
- actor rollout overshoot: `0.75`
- max XY mean/max: `0.02707330696284771` / `0.041937075555324554`
- actor-vs-recovery MSE/MAE/cosine: `0.1721746616065502` / `0.3330072117776706` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.17217466235160828` / `0.33300721645355225` / `0.0`
- actor action abs mean/max: `0.3330072117776706` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.8135775975184515` / `0.887499988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/d290_seed31713_model225_fixed_friction_580/closed_loop_recovery_dataset_d318_d290_seed31713_model225_fixed_friction_580.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- recovery clip rate too high: 0.8135775975184515

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
