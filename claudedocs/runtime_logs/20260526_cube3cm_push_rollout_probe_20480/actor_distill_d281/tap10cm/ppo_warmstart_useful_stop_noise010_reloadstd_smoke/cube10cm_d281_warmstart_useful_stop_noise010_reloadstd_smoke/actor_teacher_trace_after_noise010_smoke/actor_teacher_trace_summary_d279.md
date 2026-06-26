# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `actor_teacher_mismatch_plus_unsafe_physics`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `True`
- env useful hold rate last/max: `0.0` / `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.2621913254261017` / `0.31786149740219116` / `0.6388046145439148`
- actor clipped abs mean/max trace: `0.4593748463381981` / `1.0`
- teacher abs mean/max trace: `0.5778417526423161` / `1.0`
- actor raw clip exceed rate/max: `0.11970187202818564` / `0.1822916865348816`
- contact/useful/reaction seen: `0.75` / `0.0` / `0.75`
- success/overshoot seen: `0.0` / `0.90625`
- max disp along mean/max: `0.006107145920395851` / `0.0398554801940918`
- max disp xy mean/max: `0.03543534874916077` / `0.1134743019938469`
- max vertical offset mean/max: `0.07815609872341156` / `0.14660897850990295`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.6354166865348816` / `0.7135416865348816`

## Issues

- actor-teacher action MSE above diagnostic threshold: 0.2621913254261017
- tap overshoot seen rate too high: 0.90625
- joint delta cap rate too high: max_trace=0.7135416865348816

## Groups

- all: count `32`, mse `0.2621913254261017`, actor abs `0.4593748450279236`, teacher abs `0.5778416395187378`, max disp xy `0.03543534874916077`, max vertical `0.07815609872341156`
- overshoot: count `29`, mse `0.2837602198123932`, actor abs `0.48100292682647705`, teacher abs `0.6098315119743347`, max disp xy `0.03845314309000969`, max vertical `0.081246517598629`
- no_overshoot: count `3`, mse `0.05369172990322113`, actor abs `0.25030362606048584`, teacher abs `0.2686062455177307`, max disp xy `0.006263337098062038`, max vertical `0.04828205332159996`
- useful: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- not_useful: count `32`, mse `0.2621913254261017`, actor abs `0.4593748450279236`, teacher abs `0.5778416395187378`, max disp xy `0.03543534874916077`, max vertical `0.07815609872341156`
- vertical_over_threshold: count `15`, mse `0.21805131435394287`, actor abs `0.49048611521720886`, teacher abs `0.5833057761192322`, max disp xy `0.04166663810610771`, max vertical `0.13290934264659882`
- vertical_ok: count `17`, mse `0.30113837122917175`, actor abs `0.4319237768650055`, teacher abs `0.5730202794075012`, max disp xy `0.029937151819467545`, max vertical `0.02984442375600338`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
