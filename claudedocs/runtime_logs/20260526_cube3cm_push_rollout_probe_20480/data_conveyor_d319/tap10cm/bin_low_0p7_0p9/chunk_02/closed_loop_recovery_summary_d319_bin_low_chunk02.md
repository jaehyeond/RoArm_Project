# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d318/tap10cm/d318_hybrid_stop_reward_v2_seed31813/model_299.pt`
- reset pose source: `env_hook`
- selected episodes: `3..991` / `93`
- samples: `58000`
- actor rollout contact/useful/reaction: `1.0` / `0.9599999785423279` / `1.0`
- actor rollout overshoot: `0.03999999910593033`
- max XY mean/max: `0.008854761719703674` / `0.04709460586309433`
- actor-vs-recovery MSE/MAE/cosine: `0.061325293763315886` / `0.1956629658824411` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.06132529303431511` / `0.1956629455089569` / `0.0`
- actor action abs mean/max: `0.1956629658824411` / `0.8821466565132141`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5014103688829546` / `0.7280000448226929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm/bin_low_0p7_0p9/chunk_02/closed_loop_recovery_dataset_d319_bin_low_chunk02.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
