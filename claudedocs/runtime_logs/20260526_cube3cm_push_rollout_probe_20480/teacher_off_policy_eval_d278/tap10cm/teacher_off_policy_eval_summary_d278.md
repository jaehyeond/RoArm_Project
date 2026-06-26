# D278_TEACHER_OFF_FROZEN_EVAL Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- contact/useful/reaction seen: `0.875` / `0.5625` / `0.875`
- success rate: `0.875`
- overshoot seen rate: `0.3125`
- max disp along mean/max: `0.0024283849634230137` / `0.018782615661621094`
- max disp xy mean/max: `0.020250540226697922` / `0.10077980160713196`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.02129734866321087` / `0.24940747022628784`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.1145833432674408` / `0.15625`
- policy action abs mean/max trace: `0.1184795308344323` / `1.3003933429718018`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- tap overshoot seen rate too high: 0.3125
- tap contact vertical offset too high: max=0.24940747022628784

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
