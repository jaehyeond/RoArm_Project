# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `561..561` / `1`
- samples: `580`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.0049958922900259495` / `0.0049958922900259495`
- actor-vs-recovery MSE/MAE/cosine: `0.6834165122464363` / `0.5472016032656719` / `-0.09768692444817259`
- actor-vs-recorded MSE/MAE/cosine: `0.09344366192817688` / `0.1678268164396286` / `0.6990780830383301`
- actor action abs mean/max: `0.29031096345841373` / `1.0`
- recovery action abs mean/max: `0.5595774482333519` / `1.0`
- recovery clip rate mean/max: `0.5306896656751633` / `0.800000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/ep561_predict_stop_h020_v200/closed_loop_dataset_d307.pt`
- per-env CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/ep561_predict_stop_h020_v200/env.csv`
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/ep561_predict_stop_h020_v200/step_env.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
