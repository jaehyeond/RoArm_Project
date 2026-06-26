# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `teacher_like_action_but_unsafe_physics`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.0765833854675293` / `0.1536317765712738` / `0.8944697976112366`
- actor clipped abs mean/max trace: `0.5059758505302256` / `1.0`
- teacher abs mean/max trace: `0.543213422451939` / `1.0`
- actor raw clip exceed rate/max: `0.06666666853546711` / `0.1354166716337204`
- contact/useful/reaction seen: `0.71875` / `0.59375` / `0.71875`
- success/overshoot seen: `0.71875` / `0.125`
- max disp along mean/max: `0.003212651237845421` / `0.017521381378173828`
- max disp xy mean/max: `0.007455273997038603` / `0.046959709376096725`
- max vertical offset mean/max: `0.10082323849201202` / `0.22511835396289825`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.609375` / `0.7604166865348816`

## Issues

- tap overshoot seen rate too high: 0.125
- tap vertical offset too high: max=0.22511835396289825
- joint delta cap rate too high: max_trace=0.7604166865348816

## Groups

- all: count `32`, mse `0.0765833854675293`, actor abs `0.5059758424758911`, teacher abs `0.5432136058807373`, max disp xy `0.007455273997038603`, max vertical `0.10082323849201202`
- overshoot: count `4`, mse `0.08727531135082245`, actor abs `0.5476028919219971`, teacher abs `0.5799036026000977`, max disp xy `0.03506717085838318`, max vertical `0.08663922548294067`
- no_overshoot: count `28`, mse `0.07505597174167633`, actor abs `0.5000290870666504`, teacher abs `0.5379722714424133`, max disp xy `0.003510716836899519`, max vertical `0.10284952819347382`
- useful: count `19`, mse `0.0654120072722435`, actor abs `0.49499621987342834`, teacher abs `0.5359299182891846`, max disp xy `0.005164364352822304`, max vertical `0.09308463335037231`
- not_useful: count `13`, mse `0.0929107666015625`, actor abs `0.5220229029655457`, teacher abs `0.5538591742515564`, max disp xy `0.01080352533608675`, max vertical `0.11213351041078568`
- vertical_over_threshold: count `20`, mse `0.060397107154130936`, actor abs `0.5326260924339294`, teacher abs `0.5697445273399353`, max disp xy `0.008534936234354973`, max vertical `0.13553038239479065`
- vertical_ok: count `12`, mse `0.10356050729751587`, actor abs `0.46155881881713867`, teacher abs `0.49899551272392273`, max disp xy `0.005655836313962936`, max vertical `0.042977988719940186`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
