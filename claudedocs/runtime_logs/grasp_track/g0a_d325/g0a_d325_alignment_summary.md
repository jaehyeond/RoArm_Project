# D325 G0a Redefined Alignment Probe

이번 case의 신규 변수: `[]` — D325 is criterion repair inside G0a.

- verdict: `D325_G0A_REDEFINED_ALIGNMENT_FAIL`
- pass_all: `0/10`
- hard_failure: `True`
- trial CSV: `claudedocs/runtime_logs/grasp_track/g0a_d325/g0a_d325_alignment_trials.csv`

## Trial Table

| trial | pos mm | tangent deg | gap mm | penetration mm | top clearance mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 60.673 | 10.489 | 10.087 | 0.000 | -1.591 | 0.044 | False |
| 2 | 58.984 | 11.448 | 12.742 | 0.000 | 1.839 | 0.039 | False |
| 3 | 57.520 | 10.538 | 11.868 | 0.000 | 1.002 | 0.044 | False |
| 4 | 56.492 | 10.250 | 11.605 | 0.000 | 0.961 | 0.022 | False |
| 5 | 57.616 | 10.764 | 12.231 | 0.000 | 1.816 | 0.017 | False |
| 6 | 58.130 | 10.400 | 12.408 | 0.000 | 1.790 | 0.020 | False |
| 7 | 56.895 | 9.941 | 11.797 | 0.000 | 1.297 | 0.020 | False |
| 8 | 58.066 | 10.952 | 12.375 | 0.000 | 1.472 | 0.019 | False |
| 9 | 58.086 | 11.460 | 12.223 | 0.000 | 1.240 | 0.020 | False |
| 10 | 58.498 | 11.411 | 12.626 | 0.000 | 1.921 | 0.019 | False |

## Failure Counts

- tcp_pose: `10`
- jaw_tangent: `0`
- fixed_jaw_gap: `10`
- fixed_jaw_penetration: `0`
- contact_height: `10`
- cube_displacement: `0`

## Snapshots

- trial 1: `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_snapshot.png`
- trial 5: `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_05_snapshot.png`
- trial 10: `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_10_snapshot.png`

## Rerun

- `claudedocs/runtime_logs/grasp_track/g0a_d325/d325_trial_01_frames.rrd`
