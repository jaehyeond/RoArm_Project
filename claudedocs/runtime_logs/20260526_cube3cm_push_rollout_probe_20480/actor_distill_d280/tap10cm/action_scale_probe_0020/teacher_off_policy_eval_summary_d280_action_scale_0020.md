# D280_TEACHER_OFF_ACTION_SCALE_0020 Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.02` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- contact/useful/reaction seen: `0.65625` / `0.5625` / `0.65625`
- success rate: `0.65625`
- overshoot seen rate: `0.125`
- max disp along mean/max: `0.004751141648739576` / `0.017197608947753906`
- max disp xy mean/max: `0.012525304220616817` / `0.09939990937709808`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.046395815908908844` / `0.16351671516895294`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.59375` / `0.7239583730697632`
- policy action abs mean/max trace: `0.5049614672516954` / `3.232712507247925`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap overshoot seen rate too high: 0.125
- joint delta cap rate too high: max_trace=0.7239583730697632
- tap contact vertical offset too high: max=0.16351671516895294

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
