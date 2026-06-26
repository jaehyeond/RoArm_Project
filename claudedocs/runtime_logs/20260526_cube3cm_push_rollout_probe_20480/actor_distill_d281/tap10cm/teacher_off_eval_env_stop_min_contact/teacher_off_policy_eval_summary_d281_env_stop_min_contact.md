# D281_TEACHER_OFF_ENV_STOP_MIN_CONTACT Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
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
- max disp along mean/max: `0.00010608416050672531` / `0.002812623977661133`
- max disp xy mean/max: `0.00011114588414784521` / `0.0028603090904653072`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.02301400899887085` / `0.15143468976020813`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.203125` / `0.2135416716337204`
- policy action abs mean/max trace: `0.2855779259754666` / `3.4911892414093018`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- none

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
