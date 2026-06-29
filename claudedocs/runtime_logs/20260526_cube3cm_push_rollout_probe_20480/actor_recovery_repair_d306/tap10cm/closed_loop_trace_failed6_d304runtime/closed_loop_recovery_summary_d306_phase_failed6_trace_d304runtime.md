# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `29..991` / `6`
- samples: `3480`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `5.3272597142495215e-05` / `0.00025380132137797773`
- actor-vs-recovery MSE/MAE/cosine: `0.5297244423157375` / `0.5593369818838506` / `-0.18400375488884588`
- actor-vs-recorded MSE/MAE/cosine: `0.11808811873197556` / `0.2046857625246048` / `0.7185251116752625`
- actor action abs mean/max: `0.1109763480144842` / `1.0`
- recovery action abs mean/max: `0.5500857683027099` / `1.0`
- recovery clip rate mean/max: `0.4548850843747114` / `0.8333333730697632`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/closed_loop_trace_failed6_d304runtime/closed_loop_recovery_dataset_d306_phase_failed6_trace_d304runtime.pt`
- per-env CSV: ``
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/closed_loop_trace_failed6_d304runtime/closed_loop_step_envs_d306_failed6.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
