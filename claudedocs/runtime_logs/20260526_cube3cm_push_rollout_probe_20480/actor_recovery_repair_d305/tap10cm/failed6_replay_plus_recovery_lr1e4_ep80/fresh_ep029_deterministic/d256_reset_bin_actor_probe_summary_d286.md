# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_PASS_HAS_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/model_actor_d256_replay_batches_d290.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `actor`
- exec teacher blend: `0.5`
- exec action clip abs: `1.0`
- warmup action source: `zero`
- joint delta reference: `joint_pos`
- bc teacher delta scale: `1.0`
- tap stop after disp m: `0.0`
- tap contact slowdown use proxy: `False`
- bins/envs/steps: `1` / `5` / `580`
- action noise std: `0.0`
- cap action threshold abs: `0.25`
- safe bins: `[[29, 29]]`

## Bin Rows

- 29-29: cap max `0.1666666716337204`, action max `0.44932442903518677`, useful max `1.0`, contact max `1.0`, overshoot max `0.0`, mse `0.019093171676725213`, cube_y `0.15000000596046448`

## Issues

- none

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
