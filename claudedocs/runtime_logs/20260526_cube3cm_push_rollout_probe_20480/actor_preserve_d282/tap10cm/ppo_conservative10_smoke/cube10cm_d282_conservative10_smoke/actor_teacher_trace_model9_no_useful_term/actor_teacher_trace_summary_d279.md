# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `no_major_trace_blocker`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/model_9.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- action scale/max joint delta: `0.04` / `0.01`
- env stop/useful terminate: `True` / `False`
- env useful hold rate last/max: `0.65625` / `0.65625`
- vertical gate mode/value: `min_contact` / `0.0`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.05501702427864075` / `0.13700318336486816` / `0.6031392812728882`
- actor clipped abs mean/max trace: `0.2714850156728564` / `1.0`
- teacher abs mean/max trace: `0.295561692436579` / `1.0`
- actor raw clip exceed rate/max: `0.06315553315738537` / `0.1614583283662796`
- contact/useful/reaction seen: `0.65625` / `0.65625` / `0.65625`
- success/overshoot seen: `0.65625` / `0.03125`
- max disp along mean/max: `0.00011611264199018478` / `0.002808094024658203`
- max disp xy mean/max: `0.0009035203838720918` / `0.025373222306370735`
- max vertical offset mean/max: `0.0475609265267849` / `0.21064907312393188`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.234375` / `0.2760416567325592`

## Issues

- joint delta cap rate too high: max_trace=0.2760416567325592

## Groups

- all: count `32`, mse `0.05501702427864075`, actor abs `0.27148497104644775`, teacher abs `0.29556187987327576`, max disp xy `0.0009035203838720918`, max vertical `0.0475609265267849`
- overshoot: count `1`, mse `0.1565590500831604`, actor abs `0.4908398389816284`, teacher abs `0.5532764196395874`, max disp xy `0.025373222306370735`, max vertical `0.04065413773059845`
- no_overshoot: count `31`, mse `0.05174146592617035`, actor abs `0.26440900564193726`, teacher abs `0.28724852204322815`, max disp xy `0.00011417519272072241`, max vertical `0.04778372496366501`
- useful: count `21`, mse `0.027705201879143715`, actor abs `0.12947532534599304`, teacher abs `0.13047820329666138`, max disp xy `0.00015956997231114656`, max vertical `0.0`
- not_useful: count `11`, mse `0.10715777426958084`, actor abs `0.5425944328308105`, teacher abs `0.6107217073440552`, max disp xy `0.0023237895220518112`, max vertical `0.13835905492305756`
- vertical_over_threshold: count `10`, mse `0.1022176519036293`, actor abs `0.5477698445320129`, teacher abs `0.616466224193573`, max disp xy `1.8846156308427453e-05`, max vertical `0.14812953770160675`
- vertical_ok: count `22`, mse `0.03356219455599785`, actor abs `0.14590099453926086`, teacher abs `0.14969630539417267`, max disp xy `0.0013056449824944139`, max vertical `0.0018479153513908386`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
