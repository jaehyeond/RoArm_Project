# D256 Reset Bin Actor Probe

- verdict: `D286_D256_RESET_BIN_ACTOR_PROBE_FAIL_NO_SAFE_BIN`
- diagnostic class: `cap_pressure_not_strongly_episode_bin_dependent`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- exec source: `teacher`
- exec teacher blend: `0.5`
- warmup action source: `zero`
- bc teacher delta scale: `1.0`
- tap stop after disp m: `0.0`
- tap contact slowdown use proxy: `False`
- bins/envs/steps: `20` / `8` / `580`
- action noise std: `0.0`
- cap action threshold abs: `1.0`
- safe bins: `[]`

## Bin Rows

- 1-67: cap max `0.0`, action max `0.800000011920929`, useful max `0.875`, contact max `0.875`, overshoot max `0.125`, mse `0.008633036143146455`, cube_y `-0.02432432388131683`
- 69-119: cap max `0.0`, action max `0.9999967813491821`, useful max `0.375`, contact max `0.625`, overshoot max `0.375`, mse `0.06690922729943975`, cube_y `-0.09378378475840027`
- 120-166: cap max `0.0`, action max `0.9999923706054688`, useful max `0.5`, contact max `0.5`, overshoot max `0.75`, mse `0.03395295443759974`, cube_y `-0.08364864881779696`
- 167-208: cap max `0.0`, action max `1.0`, useful max `0.5`, contact max `0.5`, overshoot max `0.75`, mse `0.04551712802956523`, cube_y `-0.07486111277507411`
- 209-250: cap max `0.0`, action max `1.0`, useful max `0.5`, contact max `0.625`, overshoot max `0.625`, mse `0.024828925180040172`, cube_y `-0.06675675541565225`
- 251-290: cap max `0.0`, action max `1.0`, useful max `0.5`, contact max `0.625`, overshoot max `0.875`, mse `0.03527976292675233`, cube_y `-0.058108107161683006`
- 291-331: cap max `0.0`, action max `1.0`, useful max `0.375`, contact max `0.75`, overshoot max `0.75`, mse `0.04628580587089126`, cube_y `-0.05027027095894556`
- 332-370: cap max `0.0`, action max `1.0`, useful max `0.375`, contact max `0.5`, overshoot max `0.75`, mse `0.023599163144987462`, cube_y `-0.0424324328432212`
- 371-407: cap max `0.0`, action max `1.0`, useful max `0.5`, contact max `0.5`, overshoot max `0.75`, mse `0.02582451558776264`, cube_y `-0.03445945931850253`
- 408-447: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `0.25`, overshoot max `0.75`, mse `0.014830440692683874`, cube_y `-0.027083333271245163`
- 448-491: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `0.25`, overshoot max `0.75`, mse `0.021208443523962693`, cube_y `-0.018513513150046002`
- 492-537: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `0.25`, overshoot max `0.625`, mse `0.0174907222117201`, cube_y `-0.009594594380138693`
- 538-582: cap max `0.0`, action max `1.0`, useful max `0.375`, contact max `0.5`, overshoot max `0.75`, mse `0.012391137580210664`, cube_y `-0.0006756756605731474`
- 583-623: cap max `0.0`, action max `1.0`, useful max `0.375`, contact max `0.5`, overshoot max `0.875`, mse `0.04768300307031464`, cube_y `0.007837837662648511`
- 624-672: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `0.375`, overshoot max `0.625`, mse `0.023912871016414258`, cube_y `0.017162161778557946`
- 673-715: cap max `0.0`, action max `1.0`, useful max `0.25`, contact max `0.25`, overshoot max `0.625`, mse `0.032744091395931`, cube_y `0.024729730057958012`
- 716-775: cap max `0.0`, action max `1.0`, useful max `0.125`, contact max `0.125`, overshoot max `0.625`, mse `0.047935315030867814`, cube_y `0.04537016629344887`
- 776-835: cap max `0.0`, action max `0.9812252521514893`, useful max `0.25`, contact max `0.25`, overshoot max `0.75`, mse `0.021872368929626677`, cube_y `0.0778529442645408`
- 836-901: cap max `0.0`, action max `0.8343466520309448`, useful max `0.375`, contact max `0.375`, overshoot max `0.875`, mse `0.027248612284291023`, cube_y `0.11328840618197983`
- 902-999: cap max `0.0`, action max `1.0`, useful max `0.125`, contact max `0.25`, overshoot max `0.5`, mse `0.045511089420700764`, cube_y `0.13682432432432431`

## Issues

- no bin met cap/overshoot/useful thresholds

## Interpretation

This probe does not train PPO. It checks whether reset episode ranges make the frozen actor produce large actions and frequent joint-delta caps.
A high cap rate means PPO collection is dominated by saturated target deltas rather than fine contact control.
Contact/useful/overshoot gates use the maximum of post-step buffers and env log scalars, because terminate-on-useful can reset buffers before the diagnostic reads them.
