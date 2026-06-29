# D299_PPO_LIKE_STOCHASTIC_NO_TERM_SEED29801 Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.04`
- action smoothing/contact scales: `0.25` / `0.35` / `0.2`
- joint delta reference: `joint_pos`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- action mode: `ppo_stochastic`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/success/useful terminate: `False` / `False` / `False`
- D256 reset warmup mode: `direct_reset`
- env stop after displacement m: `0.003`
- env contact slowdown uses proxy: `False`
- done rate mean/max/total: `0.0` / `0.0` / `0`
- RSL-like log contact/useful/success/overshoot mean: `0.8638469827586207` / `0.8638469827586207` / `0.8638469827586207` / `0.0`
- RSL-like log max disp along/xy mean: `0.0011969568400547422` / `0.0015055149030652908`
- env useful hold rate last/max: `0.0` / `0.0`
- env displacement hold rate last/max: `0.46875` / `0.46875`
- zero actions after useful seen: `False`
- exec action clip abs: `1.0`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `1.0` / `1.0` / `1.0`
- success rate: `1.0`
- overshoot seen rate: `0.0`
- max disp along mean/max: `0.0015951255336403847` / `0.005297183990478516`
- max disp xy mean/max: `0.0020347752142697573` / `0.007077273912727833`
- max disp along >=1mm/>=3mm rate: `0.4375` / `0.375`
- max disp xy >=1mm/>=3mm rate: `0.53125` / `0.46875`
- displacement gate mean/max along: `0.0` / `0.0`
- displacement gate mean/max xy: `0.0` / `0.0`
- displacement gate >=1mm rate along/xy: `0.0` / `0.25`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.0016043446958065033` / `0.051339030265808105`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0` / `0.0`
- policy action abs mean/max trace: `0.2941691809558663` / `1.0`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- RSL-like useful seen rate below threshold: 0.8638469827586207

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
