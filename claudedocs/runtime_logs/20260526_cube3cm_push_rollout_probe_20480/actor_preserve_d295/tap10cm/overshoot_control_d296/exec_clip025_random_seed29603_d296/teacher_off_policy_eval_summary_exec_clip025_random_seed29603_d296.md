# EXEC_CLIP025_RANDOM_SEED29603_D296 Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.04`
- action smoothing/contact scales: `0.25` / `0.35` / `0.2`
- joint delta reference: `joint_pos`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `False` / `False`
- env stop after displacement m: `0.0`
- env contact slowdown uses proxy: `False`
- env useful hold rate last/max: `0.0` / `0.0`
- env displacement hold rate last/max: `0.0` / `0.0`
- zero actions after useful seen: `False`
- exec action clip abs: `0.25`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.75` / `0.5` / `0.75`
- success rate: `0.5`
- overshoot seen rate: `0.25`
- max disp along mean/max: `0.0007157600484788418` / `0.00684453547000885`
- max disp xy mean/max: `0.007333547342568636` / `0.03446927294135094`
- max disp along >=1mm/>=3mm rate: `0.15625` / `0.125`
- max disp xy >=1mm/>=3mm rate: `0.4375` / `0.375`
- displacement gate mean/max along: `0.0` / `0.0`
- displacement gate mean/max xy: `0.0005` / `0.001`
- displacement gate >=1mm rate along/xy: `0.0` / `0.25`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.00022624246776103973` / `0.0072397589683532715`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0` / `0.0`
- policy action abs mean/max trace: `0.14724237424610503` / `0.25`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap useful seen rate below threshold: 0.5
- tap overshoot seen rate too high: 0.25

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
