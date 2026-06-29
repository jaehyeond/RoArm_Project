# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `actor`
- exec teacher blend: `0.5`
- exec action clip abs: `0.5`
- warmup action source: `zero`
- joint delta reference: `joint_pos`
- bc teacher delta scale: `1.0`
- tap stop after disp m: `0.0`
- tap contact slowdown use proxy: `False`
- bins/envs/steps: `5` / `32` / `580`
- action noise std: `0.0`
- cap action threshold abs: `1.0`
- safe bins: `[]`

## Bin Rows

- 1-208: cap max `0.0`, action max `0.5`, useful max `0.8125`, contact max `0.96875`, overshoot max `0.15625`, mse `0.10971376879927541`, cube_y `-0.06911564684238564`
- 209-370: cap max `0.0`, action max `0.5`, useful max `0.21875`, contact max `0.8125`, overshoot max `0.8125`, mse `0.5332040563229761`, cube_y `-0.0543918915948755`
- 371-537: cap max `0.0`, action max `0.5`, useful max `0.375`, contact max `0.84375`, overshoot max `0.78125`, mse `0.5132491309512889`, cube_y `-0.02238095218480444`
- 538-715: cap max `0.0`, action max `0.5`, useful max `0.375`, contact max `0.84375`, overshoot max `0.71875`, mse `0.3701864627723839`, cube_y `0.012263513459647829`
- 716-999: cap max `0.0`, action max `0.5`, useful max `0.25`, contact max `0.65625`, overshoot max `0.8125`, mse `0.5281913412333434`, cube_y `0.0936602445788124`

## Issues

- no bin met cap/overshoot/useful thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
