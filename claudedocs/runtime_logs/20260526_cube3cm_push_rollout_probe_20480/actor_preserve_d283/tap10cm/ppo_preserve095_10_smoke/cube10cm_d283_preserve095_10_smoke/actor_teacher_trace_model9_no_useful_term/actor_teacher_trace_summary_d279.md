# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_PASS_FOR_SHORT_PPO_REVIEW`
- diagnostic class: `no_major_trace_blocker`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.71875` / `0.71875`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.05202638357877731` / `0.13698193430900574` / `0.6651849150657654`
- actor clipped abs mean/max trace: `0.27275253452103715` / `1.0`
- teacher abs mean/max trace: `0.2688439975749573` / `1.0`
- actor raw clip exceed rate/max: `0.05924928416337433` / `0.109375`
- contact/useful/reaction seen: `0.71875` / `0.71875` / `0.71875`
- success/overshoot seen: `0.71875` / `0.0`
- max disp along mean/max: `0.00010607670992612839` / `0.0028123855590820312`
- max disp xy mean/max: `0.00011114222434116527` / `0.0028599663637578487`
- max vertical offset mean/max: `0.03629881516098976` / `0.24010327458381653`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.2083333283662796` / `0.2135416716337204`

## Issues

- none

## Groups

- all: count `32`, mse `0.05202638357877731`, actor abs `0.27275240421295166`, teacher abs `0.26884421706199646`, max disp xy `0.00011114222434116527`, max vertical `0.03629881516098976`
- overshoot: count `0`, mse `None`, actor abs `None`, teacher abs `None`, max disp xy `None`, max vertical `None`
- no_overshoot: count `32`, mse `0.05202638357877731`, actor abs `0.27275240421295166`, teacher abs `0.26884421706199646`, max disp xy `0.00011114222434116527`, max vertical `0.03629881516098976`
- useful: count `23`, mse `0.0358458086848259`, actor abs `0.18076153099536896`, teacher abs `0.16288360953330994`, max disp xy `0.0001469307317165658`, max vertical `0.0`
- not_useful: count `9`, mse `0.09337674081325531`, actor abs `0.507840096950531`, teacher abs `0.5396324992179871`, max disp xy `1.9682722268044017e-05`, max vertical `0.1290624737739563`
- vertical_over_threshold: count `6`, mse `0.04712029919028282`, actor abs `0.5427166223526001`, teacher abs `0.5810665488243103`, max disp xy `1.9643430277938023e-05`, max vertical `0.16955700516700745`
- vertical_ok: count `26`, mse `0.053158558905124664`, actor abs `0.21045294404029846`, teacher abs `0.19679291546344757`, max disp xy `0.00013225733709987253`, max vertical `0.005546928849071264`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
