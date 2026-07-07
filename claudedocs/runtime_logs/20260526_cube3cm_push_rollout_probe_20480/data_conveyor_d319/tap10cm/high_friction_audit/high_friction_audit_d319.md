# D319 High-Friction Audit

Source: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/perturbation_matrix_d314/tap10cm/row_07_friction_high/closed_loop_recovery_summary_d314_matrix_friction_high.json`

- row: `D314 friction high static=2.2 dynamic=1.8`
- useful/overshoot/contact/reaction: `0/32` / `32/32` / `8/32` / `8/32`
- max XY mean/max/min: `3.768838` / `11.989522` / `0.025654` m
- envs >=0.1m / >=1m / >=5m: `17` / `13` / `12`
- max speed mean/max: `4.247753` / `10.000000` m/s
- global max-speed event: `{'env_id': 2, 'step': 0, 'speed_mps': 10.0, 'disp_xy_m': 0.10866019874811172}`
- primitive stop step unique: `[1]`, latched `32/32`
- interpretation: `solver_or_runaway_artifact_suspect_not_valid_training_target`

## Top Max-XY Outliers

| env | episode | max XY m | max speed m/s | max speed step | first overshoot step | primitive stop step |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 742 | 11.989522 | 10.000000 | 0 | 0 | 1 |
| 19 | 80 | 9.988884 | 10.000000 | 0 | 0 | 1 |
| 15 | 889 | 9.837431 | 10.000000 | 0 | 0 | 1 |
| 21 | 256 | 9.616860 | 9.999999 | 0 | 0 | 1 |
| 29 | 157 | 9.480302 | 9.999999 | 0 | 0 | 1 |
| 28 | 939 | 9.456218 | 10.000000 | 0 | 0 | 1 |
| 18 | 828 | 9.337893 | 10.000000 | 0 | 0 | 1 |
| 12 | 132 | 9.204433 | 10.000000 | 0 | 0 | 1 |
