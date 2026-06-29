# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `actor`
- exec teacher blend: `0.5`
- exec action clip abs: `1.0`
- warmup action source: `zero`
- joint delta reference: `joint_pos`
- bc teacher delta scale: `1.0`
- tap stop after disp m: `0.003`
- tap contact slowdown use proxy: `False`
- bins/envs/steps: `5` / `5` / `580`
- action noise std: `0.005`
- cap action threshold abs: `1.0`
- safe bins: `[]`

## Bin Rows

- 265-265: cap max `0.0`, action max `1.0`, useful max `1.0`, contact max `1.0`, overshoot max `0.20000000298023224`, mse `0.7431076526192242`, cube_y `-0.05999999865889549`

## Issues

- no bin met cap/overshoot/useful thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
