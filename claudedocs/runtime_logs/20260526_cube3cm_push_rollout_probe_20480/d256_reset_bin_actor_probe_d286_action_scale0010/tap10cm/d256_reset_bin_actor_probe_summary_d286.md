# D256 Reset Bin Actor Probe D286

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_PASS_HAS_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- bins/envs/steps: `5` / `32` / `72`
- action noise std: `0.02`
- cap action threshold abs: `1.0`
- safe bins: `[[1, 208], [209, 370], [371, 537], [538, 715], [716, 999]]`

## Bin Rows

- 1-208: cap max `0.0`, action max `0.5292695760726929`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.364762726589106`, cube_y `-0.06911564684238564`
- 209-370: cap max `0.0`, action max `0.6470056176185608`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.38923003477328977`, cube_y `-0.0543918915948755`
- 371-537: cap max `0.0`, action max `0.6831187009811401`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.37822910612966454`, cube_y `-0.02238095218480444`
- 538-715: cap max `0.0`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.38278130468097515`, cube_y `0.012263513459647829`
- 716-999: cap max `0.0`, action max `0.8501474261283875`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.4088074672488599`, cube_y `0.0936602445788124`

## Issues

- none

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
