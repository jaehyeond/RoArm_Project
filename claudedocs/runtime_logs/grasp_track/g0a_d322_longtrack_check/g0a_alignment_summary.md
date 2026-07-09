# D322 G0a Alignment Probe

이번 case의 신규 변수: [grasp pose geometry: base yaw alignment + asymmetric TCP offset]

- verdict: `D322_G0A_ALIGNMENT_FAIL`
- trials: `10`
- pass_all: `0/10`
- hard_failure: `True`
- output CSV: `claudedocs/runtime_logs/grasp_track/g0a_d322_longtrack_check/g0a_alignment_trials.csv`

## Criteria

- TCP pose error <= 5mm and base-yaw error <= 3deg.
- Fixed-jaw face gap to cube face <= 3mm and no penetration.
- Cube XY displacement < 5mm.
- Strict pass requires all 10 trials to satisfy all criteria.

## Trial Table

| trial | pose err mm | yaw err deg | face gap mm | penetration mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 95.983 | 0.081 | -55.130 | 55.130 | 0.037 | False |
| 1 | 98.711 | 0.149 | -59.455 | 59.455 | 0.038 | False |
| 2 | 96.900 | 0.157 | -56.358 | 56.358 | 0.037 | False |
| 3 | 94.677 | 0.141 | -52.604 | 52.604 | 0.012 | False |
| 4 | 98.421 | 0.065 | -58.853 | 58.853 | 0.012 | False |
| 5 | 97.875 | 0.070 | -57.903 | 57.903 | 0.011 | False |
| 6 | 94.451 | 0.156 | -51.871 | 51.871 | 0.013 | False |
| 7 | 96.127 | 0.082 | -55.287 | 55.287 | 0.010 | False |
| 8 | 96.101 | 0.086 | -55.568 | 55.568 | 0.013 | False |
| 9 | 95.674 | 0.052 | -54.815 | 54.815 | 0.013 | False |

## Failure Counts

- tcp_pose: `10`
- base_yaw: `0`
- fixed_jaw_gap: `10`
- fixed_jaw_penetration: `10`
- cube_displacement: `0`
