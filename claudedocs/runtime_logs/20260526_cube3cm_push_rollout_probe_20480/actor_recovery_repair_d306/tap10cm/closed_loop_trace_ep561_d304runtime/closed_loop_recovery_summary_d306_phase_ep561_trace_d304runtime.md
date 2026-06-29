# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`
- reset pose source: `manual`
- selected episodes: `561..561` / `1`
- samples: `580`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `3.7224977859295905e-05` / `3.7224977859295905e-05`
- actor-vs-recovery MSE/MAE/cosine: `0.4816393563419516` / `0.5157158843081059` / `-0.28778397442756926`
- actor-vs-recorded MSE/MAE/cosine: `0.10142343491315842` / `0.19006530940532684` / `0.7069126963615417`
- actor action abs mean/max: `0.09926557626960607` / `0.7390739917755127`
- recovery action abs mean/max: `0.48084907149494593` / `1.0`
- recovery clip rate mean/max: `0.3434482852446622` / `0.800000011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/closed_loop_trace_ep561_d304runtime/closed_loop_recovery_dataset_d306_phase_ep561_trace_d304runtime.pt`
- per-env CSV: ``
- per-step/env action CSV: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/closed_loop_trace_ep561_d304runtime/closed_loop_step_envs_d306_ep561.csv`

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
