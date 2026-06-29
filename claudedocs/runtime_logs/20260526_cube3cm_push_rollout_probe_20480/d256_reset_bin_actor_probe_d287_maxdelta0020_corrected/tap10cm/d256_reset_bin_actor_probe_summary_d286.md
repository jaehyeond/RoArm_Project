# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `reset_episode_bin_dependent_action_cap_pressure`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- bins/envs/steps: `5` / `32` / `580`
- action noise std: `0.02`
- cap action threshold abs: `0.5`
- safe bins: `[]`

## Bin Rows

- 1-208: cap max `0.4375`, action max `1.0`, useful max `0.5625`, contact max `0.6875`, overshoot max `0.125`, mse `0.10670966105937059`, cube_y `-0.06911564684238564`
- 209-370: cap max `0.2864583432674408`, action max `1.0`, useful max `0.28125`, contact max `0.46875`, overshoot max `0.625`, mse `0.11092897437448646`, cube_y `-0.0543918915948755`
- 371-537: cap max `0.8020833730697632`, action max `1.0`, useful max `0.25`, contact max `0.3125`, overshoot max `0.65625`, mse `0.04200600304416027`, cube_y `-0.02238095218480444`
- 538-715: cap max `0.4947916865348816`, action max `1.0`, useful max `0.28125`, contact max `0.28125`, overshoot max `0.59375`, mse `0.051136146552041815`, cube_y `0.012263513459647829`
- 716-999: cap max `0.703125`, action max `1.0`, useful max `0.21875`, contact max `0.40625`, overshoot max `0.5625`, mse `0.057380680569270946`, cube_y `0.0936602445788124`

## Issues

- no bin met cap/overshoot/useful thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
