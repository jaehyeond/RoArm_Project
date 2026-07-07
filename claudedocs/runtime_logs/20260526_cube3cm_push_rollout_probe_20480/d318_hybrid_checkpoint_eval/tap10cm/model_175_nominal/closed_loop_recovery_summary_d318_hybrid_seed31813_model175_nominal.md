# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_175.pt`
- reset pose source: `env_hook`
- selected episodes: `43..929` / `29`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0013681479031220078` / `0.0026123214047402143`
- actor-vs-recovery MSE/MAE/cosine: `0.059948371498492255` / `0.21297000492441243` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.05994837358593941` / `0.21297000348567963` / `0.0`
- actor action abs mean/max: `0.21297000492441243` / `0.6431599259376526`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.6003771702534166` / `0.6937500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d318_hybrid_checkpoint_eval/tap10cm/model_175_nominal/closed_loop_recovery_dataset_d318_hybrid_seed31813_model175_nominal.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
