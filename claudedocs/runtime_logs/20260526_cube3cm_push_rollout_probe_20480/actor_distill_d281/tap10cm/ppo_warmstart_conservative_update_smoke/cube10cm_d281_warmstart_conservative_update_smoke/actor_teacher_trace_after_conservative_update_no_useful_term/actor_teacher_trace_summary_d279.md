# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
- diagnostic class: `no_major_trace_blocker`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.8125` / `0.8125`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.04292111471295357` / `0.13137522339820862` / `0.6536584496498108`
- actor clipped abs mean/max trace: `0.2666637097196332` / `1.0`
- teacher abs mean/max trace: `0.26386851928807026` / `1.0`
- actor raw clip exceed rate/max: `0.07107579303616336` / `0.0989583358168602`
- contact/useful/reaction seen: `0.8125` / `0.8125` / `0.8125`
- success/overshoot seen: `0.8125` / `0.0`
- max disp along mean/max: `0.00010607298463582993` / `0.0028111934661865234`
- max disp xy mean/max: `0.00011117621033918113` / `0.002858859021216631`
- max vertical offset mean/max: `0.02931986376643181` / `0.18713559210300446`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.125` / `0.1666666716337204`

## Issues

- none

## Groups

- all: count `32`, mse `0.04292111471295357`, actor abs `0.26666364073753357`, teacher abs `0.26386862993240356`, max disp xy `0.00011117621033918113`, max vertical `0.02931986376643181`
- overshoot: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- no_overshoot: count `32`, mse `0.04292111471295357`, actor abs `0.26666364073753357`, teacher abs `0.26386862993240356`, max disp xy `0.00011117621033918113`, max vertical `0.02931986376643181`
- useful: count `26`, mse `0.041819602251052856`, actor abs `0.20322994887828827`, teacher abs `0.18969231843948364`, max disp xy `0.00013229783507995307`, max vertical `0.0`
- not_useful: count `6`, mse `0.047694336622953415`, actor abs `0.5415429472923279`, teacher abs `0.5852994322776794`, max disp xy `1.9649150999612175e-05`, max vertical `0.15637260675430298`
- vertical_over_threshold: count `6`, mse `0.047694336622953415`, actor abs `0.5415429472923279`, teacher abs `0.5852994322776794`, max disp xy `1.9649150999612175e-05`, max vertical `0.15637260675430298`
- vertical_ok: count `26`, mse `0.041819602251052856`, actor abs `0.20322994887828827`, teacher abs `0.18969231843948364`, max disp xy `0.00013229783507995307`, max vertical `0.0`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
