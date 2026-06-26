# D256 Reset Bin Actor Probe D286

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `reset_episode_bin_dependent_action_cap_pressure`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- bins/envs/steps: `5` / `32` / `72`
- action noise std: `0.02`
- cap action threshold abs: `0.25`
- safe bins: `[]`

## Bin Rows

- 1-208: cap max `0.390625`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.07524083745536497`, cube_y `-0.06911564684238564`
- 209-370: cap max `0.359375`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.08713604297695889`, cube_y `-0.0543918915948755`
- 371-537: cap max `0.4583333730697632`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.054966785725102656`, cube_y `-0.02238095218480444`
- 538-715: cap max `0.46875`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.02834511639149342`, cube_y `0.012263513459647829`
- 716-999: cap max `0.53125`, action max `1.0`, useful `0.0`, contact `0.0`, overshoot `0.0`, mse `0.049837260671968885`, cube_y `0.0936602445788124`

## Issues

- no bin met cap/overshoot safety thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
