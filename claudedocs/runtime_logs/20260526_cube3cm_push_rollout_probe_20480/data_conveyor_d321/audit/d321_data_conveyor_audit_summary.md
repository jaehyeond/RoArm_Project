# D321 data conveyor audit

Offline audit only: no Isaac runtime, no PPO, no render.

Filter rule: contact=1, reaction=1, useful=1, overshoot=0, max XY >= 1mm, max XY < 300mm.

## Bin pass rates

| bin | generated | accepted | contact | reaction | useful | overshoot | solver_outlier | reject_reasons | delta vs D319 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bin_low_0p7_0p9 | 1000 | 954 (95.4%) | 1000 | 1000 | 954 | 46 | 0 | {"accepted": 954, "not_useful": 46} | -0.93pp |
| bin_mid_0p9_1p2 | 1000 | 966 (96.6%) | 999 | 999 | 966 | 34 | 1 | {"accepted": 966, "not_useful": 33, "solver_outlier": 1} | +0.10pp |

## Script-only vs D321 diversity

| corpus | accepted | mean accepted XY | accepted XY variance | direction histogram |
| --- | --- | --- | --- | --- |
| script_0_999 accepted | 812 | 7.12mm | 14.21mm^2 | {"+x": 496, "+x/+y": 139, "+x/-y": 167, "+y": 6, "-x": 4} |
| d321 accepted | 1920 | 9.83mm | 8.14mm^2 | {"+x_object_frame_commanded": 1920} |

## Gate interpretation

- Any bin below 90% pass rate triggers the D321 failable-experiment failure condition.
- `solver_outlier` is a physicality gate, not controller tuning.
- D321 production remains +x only; direction diversification is reserved for D322+ goal-conditioned learning.

JSON: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_data_conveyor_audit_summary.json`
Accepted rows: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_accepted_env_rows.csv`
All rows: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_all_env_filter_rows.csv`
