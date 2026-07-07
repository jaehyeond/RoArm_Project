# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_125.pt`
- reset pose source: `env_hook`
- selected episodes: `43..929` / `29`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.001370967598631978` / `0.0026177868712693453`
- actor-vs-recovery MSE/MAE/cosine: `0.06018358019405398` / `0.21258094187440543` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.0601835772395134` / `0.21258093416690826` / `0.0`
- actor action abs mean/max: `0.21258094187440543` / `0.6351051330566406`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.6012931182723621` / `0.6937500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d318_hybrid_checkpoint_eval/tap10cm/model_125_nominal/closed_loop_recovery_dataset_d318_hybrid_seed31813_model125_nominal.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
