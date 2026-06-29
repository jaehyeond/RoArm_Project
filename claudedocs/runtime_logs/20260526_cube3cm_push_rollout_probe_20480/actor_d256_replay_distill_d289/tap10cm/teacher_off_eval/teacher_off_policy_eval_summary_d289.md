# D289_D256_REPLAY_ACTOR_TEACHER_OFF_EVAL Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_distill_d289/tap10cm/model_actor_d256_replay_d289.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.04`
- action smoothing/contact scales: `1.0` / `1.0` / `1.0`
- joint delta reference: `joint_pos`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.96875` / `0.96875`
- zero actions after useful seen: `True`
- useful action hold rate last/max: `0.96875` / `0.96875`
- contact/useful/reaction seen: `0.96875` / `0.96875` / `0.96875`
- success rate: `0.96875`
- overshoot seen rate: `0.0`
- max disp along mean/max: `0.00011815037578344345` / `0.0031893253326416016`
- max disp xy mean/max: `0.0001242639118572697` / `0.00327298603951931`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.0072176288813352585` / `0.23096412420272827`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.010416666977107525` / `0.0260416679084301`
- policy action abs mean/max trace: `0.01797055769511017` / `3.7364823818206787`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- none

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
