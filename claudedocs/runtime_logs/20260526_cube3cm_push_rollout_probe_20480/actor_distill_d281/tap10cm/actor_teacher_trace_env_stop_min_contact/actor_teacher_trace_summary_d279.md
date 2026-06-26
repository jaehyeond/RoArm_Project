# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
- diagnostic class: `no_major_trace_blocker`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.71875` / `0.71875`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.05346343293786049` / `0.13944056630134583` / `0.6641471982002258`
- actor clipped abs mean/max trace: `0.27170097621093536` / `1.0`
- teacher abs mean/max trace: `0.2712310585193336` / `1.0`
- actor raw clip exceed rate/max: `0.0592403040179212` / `0.109375`
- contact/useful/reaction seen: `0.71875` / `0.71875` / `0.71875`
- success/overshoot seen: `0.71875` / `0.0`
- max disp along mean/max: `0.00010608416050672531` / `0.002812623977661133`
- max disp xy mean/max: `0.00011114588414784521` / `0.0028603090904653072`
- max vertical offset mean/max: `0.034724317491054535` / `0.22511784732341766`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.203125` / `0.2135416716337204`

## Issues

- none

## Groups

- all: count `32`, mse `0.05346343293786049`, actor abs `0.271700918674469`, teacher abs `0.2712312340736389`, max disp xy `0.00011114588414784521`, max vertical `0.034724317491054535`
- overshoot: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- no_overshoot: count `32`, mse `0.05346343293786049`, actor abs `0.271700918674469`, teacher abs `0.2712312340736389`, max disp xy `0.00011114588414784521`, max vertical `0.034724317491054535`
- useful: count `23`, mse `0.03811090812087059`, actor abs `0.17854778468608856`, teacher abs `0.16525337100028992`, max disp xy `0.00014693582488689572`, max vertical `0.0`
- not_useful: count `9`, mse `0.09269767254590988`, actor abs `0.5097589492797852`, teacher abs `0.5420635342597961`, max disp xy `1.9682722268044017e-05`, max vertical `0.12346424162387848`
- vertical_over_threshold: count `6`, mse `0.05094677209854126`, actor abs `0.5435609817504883`, teacher abs `0.5819053649902344`, max disp xy `1.9643430277938023e-05`, max vertical `0.17171329259872437`
- vertical_ok: count `26`, mse `0.0540442019701004`, actor abs `0.20896399021148682`, teacher abs `0.19953717291355133`, max disp xy `0.0001322618336416781`, max vertical `0.0031114749144762754`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
