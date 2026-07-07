# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d317/tap10cm/d317_reward_v2_friction_uniform_seed31711/model_25.pt`
- reset pose source: `env_hook`
- selected episodes: `9..968` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.125` / `1.0`
- actor rollout overshoot: `0.875`
- max XY mean/max: `0.032598961144685745` / `0.04163865000009537`
- actor-vs-recovery MSE/MAE/cosine: `0.10978746615221788` / `0.27022463555993703` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.1097874641418457` / `0.2702246606349945` / `0.0`
- actor action abs mean/max: `0.27022463555993703` / `1.0`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.7332435445097307` / `0.8062500357627869`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_reward_v2_peak_cross_eval/tap10cm/seed31711_model25_friction_0p8_0p6/closed_loop_recovery_dataset_d317_v2_seed31711_model25_friction_0p8_0p6.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
