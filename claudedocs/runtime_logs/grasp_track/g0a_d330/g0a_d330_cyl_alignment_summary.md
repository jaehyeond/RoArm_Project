# D330 G0a Cylinder Alignment Probe

이번 case의 신규 변수: `[]` -- D330 executes the D329-approved object redefinition; no extra variable.

- verdict: `D330_G0A_CYL_ALIGNMENT_FAIL`
- pass_all: `0/10`
- output json: `claudedocs/runtime_logs/grasp_track/g0a_d330/g0a_d330_cyl_alignment_summary.json`
- trial csv: `claudedocs/runtime_logs/grasp_track/g0a_d330/g0a_d330_cyl_alignment_trials.csv`
- rrd: `claudedocs/runtime_logs/grasp_track/g0a_d330/d330_cyl_alignment_trace_v2.rrd`

## Trial Table

| trial | pos mm | cmd pos mm | tangent deg | plane gap mm | radius gap mm | top clearance mm | disp mm | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 27.233 | 0.587 | 1.402 | 2.991 | 6.475 | 25.950 | 39.456 | False |
| 2 | 3.044 | 0.588 | 1.387 | 2.625 | 5.036 | 45.870 | 10.015 | False |
| 3 | 7.686 | 0.251 | 1.989 | 2.451 | 6.978 | 48.641 | 8.962 | False |
| 4 | 1.884 | 0.252 | 2.685 | 2.652 | 4.664 | 45.299 | 10.910 | False |
| 5 | 75.775 | 0.945 | 2.856 | 5.198 | 31.004 | -11.822 | 35.451 | False |
| 6 | 6.004 | 0.252 | 3.206 | 2.424 | 6.208 | 47.933 | 9.514 | False |
| 7 | 70.303 | 0.252 | 3.093 | 4.248 | 12.243 | -19.566 | 9.710 | False |
| 8 | 7.881 | 0.250 | 1.256 | 2.506 | 7.097 | 48.570 | 8.872 | False |
| 9 | 79.989 | 0.252 | 1.495 | 5.065 | 27.451 | -20.622 | 28.615 | False |
| 10 | 80.530 | 0.413 | 2.828 | 5.060 | 27.954 | -20.574 | 29.194 | False |

## Failure Counts

- tcp_pose: `8`
- jaw_tangent: `0`
- fixed_jaw_gap: `3`
- fixed_jaw_penetration: `0`
- contact_height: `4`
- object_displacement: `10`

## Contact Trace

- robot_net_max_force_n: `0.0`
- robot_net_first_contact_step: `-1`
- robot_net_argmax_body_name: ``

## Contact Sensor Status

- robot_net ok: `False`
- robot_net error: `RuntimeError('Failed to initialize contact reporter for specified bodies.\n\tInput prim path    : /World/envs/env_.*/Robot/.*\n\tResolved prim paths: /World/envs/env_.*/Robot/(world|link1|link2|link3|link4|link5|gripper_link)')`
- gripper_link ok: `False`
  error: `RuntimeError('Failed to initialize contact reporter for specified bodies.\n\tInput prim path    : /World/envs/env_.*/Robot/gripper_link\n\tResolved prim paths: /World/envs/env_.*/Robot/(gripper_link)')`
- link4 ok: `False`
  error: `RuntimeError('Failed to initialize contact reporter for specified bodies.\n\tInput prim path    : /World/envs/env_.*/Robot/link4\n\tResolved prim paths: /World/envs/env_.*/Robot/(link4)')`
- link5 ok: `False`
  error: `RuntimeError('Failed to initialize contact reporter for specified bodies.\n\tInput prim path    : /World/envs/env_.*/Robot/link5\n\tResolved prim paths: /World/envs/env_.*/Robot/(link5)')`

## Snapshots

- trial 1: `claudedocs/runtime_logs/grasp_track/g0a_d330/d330_cyl_alignment_trial_01_snapshot.png`
- trial 5: `claudedocs/runtime_logs/grasp_track/g0a_d330/d330_cyl_alignment_trial_05_snapshot.png`
- trial 10: `claudedocs/runtime_logs/grasp_track/g0a_d330/d330_cyl_alignment_trial_10_snapshot.png`
