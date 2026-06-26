# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
- diagnostic class: `no_major_trace_blocker`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.71875` / `0.71875`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.05328662693500519` / `0.13893452286720276` / `0.6633936166763306`
- actor clipped abs mean/max trace: `0.27245025384914257` / `1.0`
- teacher abs mean/max trace: `0.27105723700068635` / `1.0`
- actor raw clip exceed rate/max: `0.060281970882627725` / `0.109375`
- contact/useful/reaction seen: `0.71875` / `0.71875` / `0.71875`
- success/overshoot seen: `0.71875` / `0.0`
- max disp along mean/max: `0.00010607670992612839` / `0.0028123855590820312`
- max disp xy mean/max: `0.00011114159133285284` / `0.002860074630007148`
- max vertical offset mean/max: `0.0351683646440506` / `0.23437753319740295`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.203125` / `0.21875`

## Issues

- none

## Groups

- all: count `32`, mse `0.05328662693500519`, actor abs `0.2724502980709076`, teacher abs `0.2710574269294739`, max disp xy `0.00011114159133285284`, max vertical `0.0351683646440506`
- overshoot: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- no_overshoot: count `32`, mse `0.05328662693500519`, actor abs `0.2724502980709076`, teacher abs `0.2710574269294739`, max disp xy `0.00011114159133285284`, max vertical `0.0351683646440506`
- useful: count `23`, mse `0.037135589867830276`, actor abs `0.17858661711215973`, teacher abs `0.16518332064151764`, max disp xy `0.00014692984404973686`, max vertical `0.0`
- not_useful: count `9`, mse `0.09456150978803635`, actor abs `0.5123242139816284`, teacher abs `0.5416246056556702`, max disp xy `1.9682722268044017e-05`, max vertical `0.1250430792570114`
- vertical_over_threshold: count `6`, mse `0.05019821226596832`, actor abs `0.5455025434494019`, teacher abs `0.580610454082489`, max disp xy `1.9643430277938023e-05`, max vertical `0.17464491724967957`
- vertical_ok: count `26`, mse `0.053999342024326324`, actor abs `0.2094382643699646`, teacher abs `0.19962210953235626`, max disp xy `0.0001322565512964502`, max vertical `0.002981470199301839`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
