# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31713/model_225.pt`
- reset pose source: `env_hook`
- selected episodes: `9..981` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.96875` / `1.0`
- actor rollout overshoot: `0.03125`
- max XY mean/max: `0.009067488834261894` / `0.041936952620744705`
- actor-vs-recovery MSE/MAE/cosine: `0.06467618076451893` / `0.21142344729139886` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0646761804819107` / `0.21142342686653137` / `0.0`
- actor action abs mean/max: `0.21142344729139886` / `0.9749246835708618`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5199353549934538` / `0.699999988079071`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/contract_diag_d318/tap10cm/d290_seed31713_model225_fixed_friction_580_hybrid/closed_loop_recovery_dataset_d318_d290_seed31713_model225_fixed_friction_580_hybrid.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
