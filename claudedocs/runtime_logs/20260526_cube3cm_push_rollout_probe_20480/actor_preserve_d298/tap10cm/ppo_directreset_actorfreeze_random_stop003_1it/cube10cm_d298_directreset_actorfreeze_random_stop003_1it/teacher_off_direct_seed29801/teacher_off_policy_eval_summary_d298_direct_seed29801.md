# D298_DIRECT_SEED29801 Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.04`
- action smoothing/contact scales: `0.25` / `0.35` / `0.2`
- joint delta reference: `joint_pos`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `False` / `False`
- D256 reset warmup mode: `direct_reset`
- env stop after displacement m: `0.003`
- env contact slowdown uses proxy: `False`
- env useful hold rate last/max: `0.0` / `0.0`
- env displacement hold rate last/max: `0.4375` / `0.4375`
- zero actions after useful seen: `False`
- exec action clip abs: `1.0`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `1.0` / `0.96875` / `1.0`
- success rate: `1.0`
- overshoot seen rate: `0.03125`
- max disp along mean/max: `0.0036056730896234512` / `0.06229519844055176`
- max disp xy mean/max: `0.004003090318292379` / `0.06263629347085953`
- max disp along >=1mm/>=3mm rate: `0.5` / `0.375`
- max disp xy >=1mm/>=3mm rate: `0.5625` / `0.46875`
- displacement gate mean/max along: `0.0` / `0.0`
- displacement gate mean/max xy: `0.0005` / `0.001`
- displacement gate >=1mm rate along/xy: `0.25` / `0.25`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.00863186176866293` / `0.09966869652271271`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0` / `0.0`
- policy action abs mean/max trace: `0.2552498017726787` / `1.0`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- none

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
