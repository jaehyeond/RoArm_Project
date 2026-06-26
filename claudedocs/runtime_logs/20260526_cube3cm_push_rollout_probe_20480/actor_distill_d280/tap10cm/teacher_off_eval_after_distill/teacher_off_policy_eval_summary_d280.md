# D280_TEACHER_OFF_AFTER_ACTOR_DISTILL Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- contact/useful/reaction seen: `0.71875` / `0.59375` / `0.71875`
- success rate: `0.71875`
- overshoot seen rate: `0.125`
- max disp along mean/max: `0.003212651237845421` / `0.017521381378173828`
- max disp xy mean/max: `0.007455273997038603` / `0.046959709376096725`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.050115544348955154` / `0.15143540501594543`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.609375` / `0.7604166865348816`
- policy action abs mean/max trace: `0.5180111268342569` / `3.2941884994506836`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap overshoot seen rate too high: 0.125
- joint delta cap rate too high: max_trace=0.7604166865348816
- tap contact vertical offset too high: max=0.15143540501594543

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
