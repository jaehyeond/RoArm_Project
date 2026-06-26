# D282_AFTER_ACTOR_FREEZE_TEACHER_OFF Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `True` / `True`
- env useful hold rate last/max: `0.0` / `0.0`
- zero actions after useful seen: `False`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.21875` / `0.0` / `0.21875`
- success rate: `0.0`
- overshoot seen rate: `0.21875`
- max disp along mean/max: `0.0016120737418532372` / `0.026876449584960938`
- max disp xy mean/max: `0.008075850084424019` / `0.07192132622003555`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.0974622517824173` / `0.15143540501594543`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.6458333730697632` / `0.7239583730697632`
- policy action abs mean/max trace: `0.49288914299987513` / `3.2941884994506836`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap useful seen rate below threshold: 0.0
- tap overshoot seen rate too high: 0.21875
- joint delta cap rate too high: max_trace=0.7239583730697632

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
