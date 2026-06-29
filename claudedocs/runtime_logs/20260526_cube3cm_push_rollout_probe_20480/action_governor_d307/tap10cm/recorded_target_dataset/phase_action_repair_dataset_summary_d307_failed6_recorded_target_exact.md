# D306 Phase-Aware Action Repair Dataset

- verdict: `D306_PHASE_ACTION_REPAIR_DATASET_WARN_REVIEW`
- source dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/closed_loop_dataset_d307.pt`
- output dataset: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_target_dataset/phase_action_repair_dataset_d307_failed6_recorded_target_exact.pt`
- samples: `3480`
- recovery weight start/end: `0.0` / `0.0`
- transition steps: `1` / `2`
- target clip abs: `1.0`
- smooth alpha: `1.0`
- target abs mean/max: `0.236792653799057` / `1.0`
- target clip >=0.99/0.90/0.75: `0.12907087802886963` / `0.14066092669963837` / `0.1451149433851242`
- target-vs-recorded MSE/cosine: `0.0` / `0.9948275685310364`
- target-vs-recovery MSE/cosine: `0.5202174782752991` / `0.11430244892835617`

## Issues

- target clip >=0.99 still high: 0.12907087802886963

## Interpretation

This is a supervised target-rewrite dataset only. It does not prove policy success.
The next required checks are offline actor-vs-target diagnostics and fresh one-bin/direct-reset rollouts.
