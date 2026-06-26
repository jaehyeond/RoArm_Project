# D280_TEACHER_OFF_ACTION_SCALE_0010 Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.01` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- contact/useful/reaction seen: `0.625` / `0.46875` / `0.625`
- success rate: `0.625`
- overshoot seen rate: `0.15625`
- max disp along mean/max: `0.007729633711278439` / `0.03227519989013672`
- max disp xy mean/max: `0.018082385882735252` / `0.10892839729785919`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.06067018583416939` / `0.21656420826911926`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0364583358168602` / `0.046875`
- policy action abs mean/max trace: `0.36322255540510706` / `2.594240427017212`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap overshoot seen rate too high: 0.15625
- tap contact vertical offset too high: max=0.21656420826911926

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
