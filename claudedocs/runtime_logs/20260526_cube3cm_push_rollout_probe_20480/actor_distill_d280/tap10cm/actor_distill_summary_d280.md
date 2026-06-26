# D280 Actor Distillation

- verdict: `D280_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL`
- source actor: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`
- distilled checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- samples train/val: `16704` / `1856`
- initial val MSE/MAE/cosine: `0.38865897059440613` / `0.5184221863746643` / `0.32961708307266235`
- final val MSE/MAE/cosine: `0.01078740879893303` / `0.0625312477350235` / `0.9815400838851929`
- teacher rollout contact/useful/reaction: `0.75` / `0.53125` / `0.75`
- teacher rollout overshoot: `0.21875`
- D256 reset active: `1.0`

## Issues

- teacher rollout overshoot high during data collection: 0.21875

## Interpretation

This is supervised actor warm-start, not PPO training. It only attempts to make the rsl_rl actor match the D257 teacher sidecar under the D256 reset/AABB contract.
Promotion still requires teacher-off eval and actor-vs-teacher trace after loading the saved actor checkpoint.
