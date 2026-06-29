# D306 Phase-Aware Action Repair Dataset

- verdict: `D306_PHASE_ACTION_REPAIR_DATASET_WARN_REVIEW`
- source dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/closed_loop_trace_failed6_d304runtime/closed_loop_recovery_dataset_d306_phase_failed6_trace_d304runtime.pt`
- output dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_target_dataset_iter2/phase_action_repair_dataset_d306_iter2_failed6_recovery050to000_clip100_smooth080.pt`
- samples: `3480`
- recovery weight start/end: `0.5` / `0.0`
- transition steps: `40` / `180`
- target clip abs: `1.0`
- smooth alpha: `0.8`
- target abs mean/max: `0.25557631254196167` / `1.0`
- target clip >=0.99/0.90/0.75: `0.12835249304771423` / `0.14123563468456268` / `0.14717432856559753`
- target-vs-recorded MSE/cosine: `0.005670062731951475` / `0.7611364126205444`
- target-vs-recovery MSE/cosine: `0.5445536375045776` / `0.35691002011299133`

## Issues

- target clip >=0.99 still high: 0.12835249304771423

## Interpretation

This is a supervised target-rewrite dataset only. It does not prove policy success.
The next required checks are offline actor-vs-target diagnostics and fresh one-bin/direct-reset rollouts.
