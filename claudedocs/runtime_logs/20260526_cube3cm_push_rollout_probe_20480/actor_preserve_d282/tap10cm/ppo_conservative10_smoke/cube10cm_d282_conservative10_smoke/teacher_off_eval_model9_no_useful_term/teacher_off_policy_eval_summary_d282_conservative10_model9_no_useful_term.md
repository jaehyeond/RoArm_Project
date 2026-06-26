# D282_CONSERVATIVE10_MODEL9_NO_USEFUL_TERM_TEACHER_OFF Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/model_9.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.65625` / `0.65625`
- zero actions after useful seen: `False`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.65625` / `0.65625` / `0.65625`
- success rate: `0.65625`
- overshoot seen rate: `0.03125`
- max disp along mean/max: `0.00011611264199018478` / `0.002808094024658203`
- max disp xy mean/max: `0.0009035203838720918` / `0.025373222306370735`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.03782736882567406` / `0.14641129970550537`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.234375` / `0.2760416567325592`
- policy action abs mean/max trace: `0.28218275945762106` / `2.2301270961761475`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- joint delta cap rate too high: max_trace=0.2760416567325592

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
