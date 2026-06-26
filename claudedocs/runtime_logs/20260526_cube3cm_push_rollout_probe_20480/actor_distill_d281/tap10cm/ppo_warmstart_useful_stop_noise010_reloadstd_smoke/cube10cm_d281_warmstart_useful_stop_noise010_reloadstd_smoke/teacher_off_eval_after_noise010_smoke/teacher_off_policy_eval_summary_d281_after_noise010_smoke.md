# D281_AFTER_NOISE010_SMOKE_TEACHER_OFF Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke/model_0.pt`
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
- contact/useful/reaction seen: `0.75` / `0.0` / `0.75`
- success rate: `0.0`
- overshoot seen rate: `0.90625`
- max disp along mean/max: `0.006107145920395851` / `0.0398554801940918`
- max disp xy mean/max: `0.03543534874916077` / `0.1134743019938469`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.05638359487056732` / `0.145905002951622`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.6354166865348816` / `0.7135416865348816`
- policy action abs mean/max trace: `0.5008288720558429` / `3.056727886199951`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap useful seen rate below threshold: 0.0
- tap overshoot seen rate too high: 0.90625
- joint delta cap rate too high: max_trace=0.7135416865348816

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
