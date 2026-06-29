# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_PASS_HAS_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `teacher`
- exec teacher blend: `0.5`
- exec action clip abs: `1.0`
- warmup action source: `zero`
- joint delta reference: `joint_pos`
- bc teacher delta scale: `1.0`
- tap stop after disp m: `0.003`
- tap contact slowdown use proxy: `False`
- bins/envs/steps: `5` / `8` / `580`
- action noise std: `0.0`
- cap action threshold abs: `1.0`
- safe bins: `[[221, 221], [198, 198]]`

## Bin Rows

- 221-221: cap max `0.0`, action max `0.800000011920929`, useful max `1.0`, contact max `1.0`, overshoot max `0.0`, mse `0.5576095527769462`, cube_y `-0.07000000029802322`
- 198-198: cap max `0.0`, action max `0.800000011920929`, useful max `1.0`, contact max `1.0`, overshoot max `0.0`, mse `0.1540315839333524`, cube_y `-0.07500000298023224`
- 13-13: cap max `0.0`, action max `0.800000011920929`, useful max `1.0`, contact max `1.0`, overshoot max `1.0`, mse `1.0234325904431272`, cube_y `-0.10000000149011612`
- 322-322: cap max `0.0`, action max `0.800000011920929`, useful max `0.0`, contact max `1.0`, overshoot max `1.0`, mse `1.0839873200503243`, cube_y `-0.05000000074505806`
- 935-935: cap max `0.0`, action max `0.800000011920929`, useful max `0.0`, contact max `1.0`, overshoot max `1.0`, mse `1.0838221468689755`, cube_y `0.13124999403953552`

## Issues

- none

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
