# D322 G0a Alignment Probe

이번 case의 신규 변수: [grasp pose geometry: base yaw alignment + asymmetric TCP offset]

- verdict: `D322_G0A_ALIGNMENT_FAIL`
- trials: `10`
- pass_all: `0/10`
- hard_failure: `True`
- output CSV: `claudedocs/runtime_logs/grasp_track/g0a_d322/g0a_alignment_trials.csv`

## Criteria

- TCP pose error <= 5mm and base-yaw error <= 3deg.
- Fixed-jaw face gap to cube face <= 3mm and no penetration.
- Cube XY displacement < 5mm.
- Strict pass requires all 10 trials to satisfy all criteria.

## Trial Table

| trial | pose err mm | yaw err deg | face gap mm | penetration mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 96.079 | 0.061 | -55.276 | 55.276 | 0.036 | False |
| 1 | 98.644 | 0.133 | -59.338 | 59.338 | 0.036 | False |
| 2 | 97.001 | 0.155 | -56.498 | 56.498 | 0.036 | False |
| 3 | 94.832 | 0.145 | -52.885 | 52.885 | 0.011 | False |
| 4 | 98.480 | 0.056 | -58.954 | 58.954 | 0.011 | False |
| 5 | 98.183 | 0.063 | -58.296 | 58.296 | 0.010 | False |
| 6 | 94.716 | 0.158 | -52.365 | 52.365 | 0.015 | False |
| 7 | 96.307 | 0.071 | -55.618 | 55.618 | 0.011 | False |
| 8 | 96.260 | 0.088 | -55.847 | 55.847 | 0.012 | False |
| 9 | 95.811 | 0.044 | -54.879 | 54.879 | 0.013 | False |

## Failure Counts

- tcp_pose: `10`
- base_yaw: `0`
- fixed_jaw_gap: `10`
- fixed_jaw_penetration: `10`
- cube_displacement: `0`
