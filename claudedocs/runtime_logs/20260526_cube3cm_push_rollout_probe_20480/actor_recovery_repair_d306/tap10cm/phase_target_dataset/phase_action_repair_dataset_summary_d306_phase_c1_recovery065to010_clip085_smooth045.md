# D306 Phase-Aware Action Repair Dataset

- verdict: `D306_PHASE_ACTION_REPAIR_DATASET_READY`
- source dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/closed_loop_recovery_after_repair/closed_loop_recovery_dataset_d305_after_repair.pt`
- output dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_target_dataset/phase_action_repair_dataset_d306_phase_c1_recovery065to010_clip085_smooth045.pt`
- samples: `3480`
- recovery weight start/end: `0.65` / `0.1`
- transition steps: `40` / `260`
- target clip abs: `0.85`
- smooth alpha: `0.45`
- target abs mean/max: `0.2988685667514801` / `0.8500000238418579`
- target clip >=0.99/0.90/0.75: `0.0` / `0.0` / `0.13314175605773926`
- target-vs-recorded MSE/cosine: `0.040475670248270035` / `0.5850999355316162`
- target-vs-recovery MSE/cosine: `0.5093693733215332` / `0.5821784138679504`

## Issues

- none

## Interpretation

This is a supervised target-rewrite dataset only. It does not prove policy success.
The next required checks are offline actor-vs-target diagnostics and fresh one-bin/direct-reset rollouts.
