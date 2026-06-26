# D281_AFTER_CONSERVATIVE_UPDATE_NO_USEFUL_TERM_TEACHER_OFF Teacher-Off Frozen Policy Eval

- verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`
- checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/model_0.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- d256 reset active rate: `1.0`
- bc teacher blend mean last: `0.0`
- vertical gate mode/value: `min_contact` / `0.0`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.8125` / `0.8125`
- zero actions after useful seen: `False`
- useful action hold rate last/max: `0.0` / `0.0`
- contact/useful/reaction seen: `0.8125` / `0.8125` / `0.8125`
- success rate: `0.8125`
- overshoot seen rate: `0.0`
- max disp along mean/max: `0.00010607298463582993` / `0.0028111934661865234`
- max disp xy mean/max: `0.00011117621033918113` / `0.002858859021216631`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max: `0.025583874434232712` / `0.14650186896324158`
- raw TCP-threshold contact seen rate: `0.0`
- joint delta cap rate mean/max trace: `0.125` / `0.1666666716337204`
- policy action abs mean/max trace: `0.27907493517830456` / `1.7776037454605103`
- reward finite all: `True`
- obs finite all: `True`
- action finite all: `True`

## Issues

- none

## Interpretation

This is a teacher-off frozen policy evaluation. The BC teacher is not allowed to blend actions.
For tap10cm, AABB/tool-surface contact is the primary contact contract; raw TCP threshold is diagnostic.
