# D326 G0a Execution Contract Probe

이번 case의 신규 변수: `[]` — D326 is G0a execution-contract repair only.

- verdict: `D326_G0A_TELEPORT_STATIC_FAIL_STOP`
- teleport pass: `False`
- selected repair: ``
- final pass_all: `0/10`

## Diagnostic Questions


## Artifacts

- teleport_check snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_check.png`
- teleport_check rrd: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2.rrd`
- rerun headless open screenshot: `claudedocs/runtime_logs/grasp_track/g0a_d326/d326_teleport_static_v2_rerun_open.png`

## Stop Reason

Teleport static check reached the target TCP (`0.349mm`) and jaw tangent
(`9.174deg`), but the fixed-jaw proxy penetrated the cube side by `0.151mm`.
Per the D326 prompt, this means the IK/criterion geometry must be rechecked and
execution-contract repair is prohibited in this session.
