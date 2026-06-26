# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `actor_teacher_mismatch_plus_unsafe_physics`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `True`
- env useful hold rate last/max: `0.0` / `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.15061450004577637` / `0.20462650060653687` / `0.7776092290878296`
- actor clipped abs mean/max trace: `0.4815749166726038` / `1.0`
- teacher abs mean/max trace: `0.5443651565222134` / `1.0`
- actor raw clip exceed rate/max: `0.10025143965291951` / `0.1666666567325592`
- contact/useful/reaction seen: `0.3125` / `0.0` / `0.3125`
- success/overshoot seen: `0.0` / `0.34375`
- max disp along mean/max: `0.0028562918305397034` / `0.05042266845703125`
- max disp xy mean/max: `0.010894259437918663` / `0.05186805501580238`
- max vertical offset mean/max: `0.12020806968212128` / `0.20163214206695557`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.5677083730697632` / `0.7447916865348816`

## Issues

- actor-teacher action MSE above diagnostic threshold: 0.15061450004577637
- tap overshoot seen rate too high: 0.34375
- joint delta cap rate too high: max_trace=0.7447916865348816

## Groups

- all: count `32`, mse `0.15061450004577637`, actor abs `0.4815748929977417`, teacher abs `0.5443654656410217`, max disp xy `0.010894259437918663`, max vertical `0.12020806968212128`
- overshoot: count `11`, mse `0.3598141372203827`, actor abs `0.4808928370475769`, teacher abs `0.6008321046829224`, max disp xy `0.0316624790430069`, max vertical `0.07423609495162964`
- no_overshoot: count `21`, mse `0.04103373363614082`, actor abs `0.48193225264549255`, teacher abs `0.5147877931594849`, max disp xy `1.566808350617066e-05`, max vertical `0.14428864419460297`
- useful: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- not_useful: count `32`, mse `0.15061450004577637`, actor abs `0.4815748929977417`, teacher abs `0.5443654656410217`, max disp xy `0.010894259437918663`, max vertical `0.12020806968212128`
- vertical_over_threshold: count `26`, mse `0.1367741823196411`, actor abs `0.501194953918457`, teacher abs `0.5566214919090271`, max disp xy `0.007791810669004917`, max vertical `0.14614993333816528`
- vertical_ok: count `6`, mse `0.21058917045593262`, actor abs `0.3965548276901245`, teacher abs `0.4912562370300293`, max disp xy `0.02433820068836212`, max vertical `0.007793386932462454`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
