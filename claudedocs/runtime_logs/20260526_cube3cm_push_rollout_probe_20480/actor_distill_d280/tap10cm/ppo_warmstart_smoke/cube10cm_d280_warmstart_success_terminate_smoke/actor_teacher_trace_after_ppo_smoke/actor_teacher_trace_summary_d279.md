# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `teacher_like_action_but_unsafe_physics`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/ppo_warmstart_smoke/cube10cm_d280_warmstart_success_terminate_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.086099773645401` / `0.17566514015197754` / `0.8869514465332031`
- actor clipped abs mean/max trace: `0.5055050533393334` / `1.0`
- teacher abs mean/max trace: `0.5650238461888812` / `1.0`
- actor raw clip exceed rate/max: `0.07002514581163896` / `0.1614583432674408`
- contact/useful/reaction seen: `0.6875` / `0.5` / `0.6875`
- success/overshoot seen: `0.6875` / `0.1875`
- max disp along mean/max: `0.003012566827237606` / `0.024276018142700195`
- max disp xy mean/max: `0.011208204552531242` / `0.09943809360265732`
- max vertical offset mean/max: `0.13498258590698242` / `0.25403064489364624`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.65625` / `0.78125`

## Issues

- tap overshoot seen rate too high: 0.1875
- tap vertical offset too high: max=0.25403064489364624
- joint delta cap rate too high: max_trace=0.78125

## Groups

- all: count `32`, mse `0.086099773645401`, actor abs `0.5055050849914551`, teacher abs `0.5650237202644348`, max disp xy `0.011208204552531242`, max vertical `0.13498258590698242`
- overshoot: count `6`, mse `0.141568124294281`, actor abs `0.5162705779075623`, teacher abs `0.5609105825424194`, max disp xy `0.052025385200977325`, max vertical `0.08863203227519989`
- no_overshoot: count `26`, mse `0.07329938560724258`, actor abs `0.503020703792572`, teacher abs `0.5659729242324829`, max disp xy `0.0017888557631522417`, max vertical `0.14567889273166656`
- useful: count `16`, mse `0.08252769708633423`, actor abs `0.4854898154735565`, teacher abs `0.5586209297180176`, max disp xy `0.002895111683756113`, max vertical `0.12971997261047363`
- not_useful: count `16`, mse `0.08967185020446777`, actor abs `0.5255202651023865`, teacher abs `0.5714265704154968`, max disp xy `0.01952129788696766`, max vertical `0.1402452290058136`
- vertical_over_threshold: count `27`, mse `0.09076593816280365`, actor abs `0.505719006061554`, teacher abs `0.5696601867675781`, max disp xy `0.009156755171716213`, max vertical `0.15141049027442932`
- vertical_ok: count `5`, mse `0.06090248376131058`, actor abs `0.5043496489524841`, teacher abs `0.5399870872497559`, max disp xy `0.022286027669906616`, max vertical `0.04627196118235588`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
