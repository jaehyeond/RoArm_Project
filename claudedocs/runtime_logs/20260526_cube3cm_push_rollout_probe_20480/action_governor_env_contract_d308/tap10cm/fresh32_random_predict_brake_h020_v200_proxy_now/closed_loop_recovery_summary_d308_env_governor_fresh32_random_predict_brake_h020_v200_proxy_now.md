# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `env_hook`
- selected episodes: `35..993` / `32`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `0.8125` / `1.0`
- actor rollout overshoot: `0.1875`
- max XY mean/max: `0.006538099609315395` / `0.03394607454538345`
- actor-vs-recovery MSE/MAE/cosine: `0.735396786046953` / `0.6385537086375829` / `0.02642731518195621`
- actor-vs-recorded MSE/MAE/cosine: `0.20206665992736816` / `0.27649930119514465` / `0.45376360416412354`
- actor action abs mean/max: `0.2839211431299818` / `1.0`
- recovery action abs mean/max: `0.6527372930841199` / `1.0`
- recovery clip rate mean/max: `0.6303987192953455` / `0.8187500238418579`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_env_contract_d308/tap10cm/fresh32_random_predict_brake_h020_v200_proxy_now/closed_loop_recovery_dataset_d308_env_governor_fresh32_random_predict_brake_h020_v200_proxy_now.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_env_contract_d308/tap10cm/fresh32_random_predict_brake_h020_v200_proxy_now/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_env_contract_d308/tap10cm/fresh32_random_predict_brake_h020_v200_proxy_now/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
