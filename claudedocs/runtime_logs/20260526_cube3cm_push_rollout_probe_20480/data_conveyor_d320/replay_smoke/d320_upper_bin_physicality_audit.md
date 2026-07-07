# D320 upper-bin physicality audit

- Source: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_all_env_filter_rows.csv`
- Upper overshoot rows: 242
- Below 300mm: 236/242 (0.975)
- Meter-scale rows: 6/242 (0.025)
- Decision: `MIXED_PHYSICAL_FAILURE_WITH_SOLVER_OUTLIERS`

| quantile | max XY (mm) |
|---:|---:|
| 0.0 | 20.068 |
| 0.1 | 22.603 |
| 0.25 | 26.185 |
| 0.5 | 30.943 |
| 0.75 | 36.314 |
| 0.9 | 40.463 |
| 0.95 | 45.575 |
| 0.99 | 11124.628 |
| 1.0 | 11140.390 |

| bin | count | rate |
|---|---:|---:|
| 0-20mm | 0 | 0.000 |
| 20-30mm | 111 | 0.459 |
| 30-50mm | 122 | 0.504 |
| 50-100mm | 3 | 0.012 |
| 100-300mm | 0 | 0.000 |
| 300-1000mm | 0 | 0.000 |
| >=1000mm | 6 | 0.025 |
