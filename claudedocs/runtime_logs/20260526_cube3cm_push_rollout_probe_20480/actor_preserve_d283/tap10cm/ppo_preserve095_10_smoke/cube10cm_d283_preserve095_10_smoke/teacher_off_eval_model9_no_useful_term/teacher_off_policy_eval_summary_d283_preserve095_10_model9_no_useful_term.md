# D283_PRESERVE095_10_MODEL9_NO_USEFUL_TERM_TEACHER_OFF Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/model_9.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.71875` / `0.71875`
- zero actions after useful seen: `False`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.71875` / `0.71875` / `0.71875`
- success rate: `0.71875`
- overshoot seen rate: `0.0`
- max disp along mean/max: `0.00010607670992612839` / `0.0028123855590820312`
- max disp xy mean/max: `0.00011114222434116527` / `0.0028599663637578487`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.02604622393846512` / `0.15769723057746887`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.2083333283662796` / `0.2135416716337204`
- policy action abs mean/max trace: `0.2857095104353181` / `3.6983871459960938`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- none

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
