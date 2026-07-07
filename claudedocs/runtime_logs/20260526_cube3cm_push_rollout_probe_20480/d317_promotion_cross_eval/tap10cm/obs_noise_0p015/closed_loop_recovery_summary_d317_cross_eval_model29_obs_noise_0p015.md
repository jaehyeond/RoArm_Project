# D290 Closed-Loop Recovery Probe

- verdict: `D290_CLOSED_LOOP_RECOVERY_DATASET_PASS_FOR_AGGREGATION`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/primitive_parameter_ppo_d316/tap10cm/d316_candidate8_friction_low_reward_v1_30it/model_29.pt`
- reset pose source: `env_hook`
- selected episodes: `29..997` / `31`
- samples: `18560`
- actor rollout contact/useful/reaction: `1.0` / `1.0` / `1.0`
- actor rollout overshoot: `0.0`
- max XY mean/max: `0.004151546396315098` / `0.006141963414847851`
- actor-vs-recovery MSE/MAE/cosine: `0.02007779412870777` / `0.11181710677157189` / `0.0`
- actor-vs-recorded MSE/MAE/cosine: `0.020077792927622795` / `0.11181710660457611` / `0.0`
- actor action abs mean/max: `0.11181710677157189` / `0.8128553032875061`
- recovery action abs mean/max: `0.0` / `0.0`
- recovery clip rate mean/max: `0.5938038872314039` / `0.706250011920929`
- dataset path: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d317_promotion_cross_eval/tap10cm/obs_noise_0p015/closed_loop_recovery_dataset_d317_cross_eval_model29_obs_noise_0p015.pt`
- per-env CSV: ``
- per-step/env action CSV: ``

## Issues

- none

## Interpretation

This probe executes the frozen actor, but labels the visited states with D256 recorded-time recovery actions.
A pass here does not mean the actor is safe; it only means the collected closed-loop states look useful for the next supervised aggregation step.
