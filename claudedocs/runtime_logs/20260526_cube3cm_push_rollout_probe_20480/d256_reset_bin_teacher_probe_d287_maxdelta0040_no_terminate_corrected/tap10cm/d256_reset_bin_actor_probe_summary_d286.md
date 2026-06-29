# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `teacher`
- exec teacher blend: `0.5`
- warmup action source: `zero`
- bins/envs/steps: `5` / `32` / `580`
- action noise std: `0.0`
- cap action threshold abs: `1.0`
- safe bins: `[]`

## Bin Rows

- 1-208: cap max `0.0`, action max `1.0`, useful max `0.84375`, contact max `1.0`, overshoot max `0.15625`, mse `0.08196937883320939`, cube_y `-0.06911564684238564`
- 209-370: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `1.0`, overshoot max `0.875`, mse `0.23405083753650302`, cube_y `-0.0543918915948755`
- 371-537: cap max `0.0`, action max `1.0`, useful max `0.40625`, contact max `0.875`, overshoot max `0.78125`, mse `0.4033344389951049`, cube_y `-0.02238095218480444`
- 538-715: cap max `0.0`, action max `1.0`, useful max `0.40625`, contact max `0.875`, overshoot max `0.71875`, mse `0.3312584528815933`, cube_y `0.012263513459647829`
- 716-999: cap max `0.0`, action max `1.0`, useful max `0.28125`, contact max `0.9375`, overshoot max `0.84375`, mse `0.2128128995167509`, cube_y `0.0936602445788124`

## Issues

- no bin met cap/overshoot/useful thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
