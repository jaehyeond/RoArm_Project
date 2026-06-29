# PROXY_SLOWDOWN035_D296 Teacher-Off Frozen Policy Eval

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
- env contact slowdown uses proxy: `True`
- env useful hold rate last/max: `0.0` / `0.0`
- env displacement hold rate last/max: `0.0` / `0.0`
- zero actions after useful seen: `False`
- exec action clip abs: `1.0`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `1.0` / `1.0` / `1.0`
- success rate: `1.0`
- overshoot seen rate: `0.0`
- max disp along mean/max: `0.00011727958917617798` / `0.0031642913818359375`
- max disp xy mean/max: `0.00012359549873508513` / `0.003246211912482977`
- max disp along >=1mm/>=3mm rate: `0.03125` / `0.03125`
- max disp xy >=1mm/>=3mm rate: `0.03125` / `0.03125`
- displacement gate mean/max along: `0.0` / `0.0`
- displacement gate mean/max xy: `0.0005` / `0.001`
- displacement gate >=1mm rate along/xy: `0.0` / `0.25`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.0` / `0.0`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0` / `0.0`
- policy action abs mean/max trace: `0.1478229000425801` / `1.0`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap mean max xy displacement below threshold: 0.00012359549873508513
- tap XY displacement >=1mm rate below threshold: 0.03125

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
