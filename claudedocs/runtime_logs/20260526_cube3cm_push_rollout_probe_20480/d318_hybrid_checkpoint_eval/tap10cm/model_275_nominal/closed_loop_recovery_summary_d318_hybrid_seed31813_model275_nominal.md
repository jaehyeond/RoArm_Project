# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_275.pt`
- reset pose source: `env_hook`
- selected episodes: `43..929` / `29`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.00136288208886981` / `0.0026105318684130907`
- actor-vs-recovery MSE/MAE/cosine: `0.0589875975238352` / `0.21012817169057912` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.05898759886622429` / `0.21012815833091736` / `0.0`
- actor action abs mean/max: `0.21012817169057912` / `0.667774498462677`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.6008944107797639` / `0.6937500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d318_hybrid_checkpoint_eval/tap10cm/model_275_nominal/closed_loop_recovery_dataset_d318_hybrid_seed31813_model275_nominal.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
