# D299_PPO_LIKE_STOCHASTIC_SUCCESS_TERM_SEED29801 Teacher-Off Frozen Policy Eval

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
- env stop/success/useful terminate: `False` / `True` / `False`
- D256 reset warmup mode: `direct_reset`
- env stop after displacement m: `0.003`
- env contact slowdown uses proxy: `False`
- done rate mean/max/total: `0.0020474137931034485` / `0.46875` / `38`
- RSL-like log contact/useful/success/overshoot mean: `0.6160021551724137` / `0.0027478448275862067` / `0.0020474137931034485` / `0.7538793103448276`
- RSL-like log max disp along/xy mean: `0.23124684133438458` / `0.2987509671237635`
- env useful hold rate last/max: `0.0` / `0.0`
- env displacement hold rate last/max: `0.03125` / `0.0625`
- zero actions after useful seen: `False`
- exec action clip abs: `1.0`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.75` / `0.0` / `0.75`
- success rate: `0.0`
- overshoot seen rate: `0.84375`
- max disp along mean/max: `0.32669803500175476` / `10.092741012573242`
- max disp xy mean/max: `0.45712873339653015` / `13.797537803649902`
- max disp along >=1mm/>=3mm rate: `0.53125` / `0.5`
- max disp xy >=1mm/>=3mm rate: `0.875` / `0.875`
- displacement gate mean/max along: `0.0` / `0.0`
- displacement gate mean/max xy: `0.0` / `0.0`
- displacement gate >=1mm rate along/xy: `0.0` / `0.25`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.007245023734867573` / `0.05146211385726929`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.0` / `0.0`
- policy action abs mean/max trace: `0.3390345779472384` / `1.0`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap useful seen rate below threshold: 0.0
- tap overshoot seen rate too high: 0.84375
- RSL-like useful seen rate below threshold: 0.0027478448275862067
- RSL-like overshoot seen rate too high: 0.7538793103448276

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
