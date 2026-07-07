# D319 data conveyor audit

Offline audit only: no Isaac runtime, no PPO, no render.

Filter rule: contact=1, reaction=1, useful=1, overshoot=0, max XY >= 1mm.

## Bin pass rates

| bin | generated | accepted | contact | reaction | useful | overshoot | mean XY | max XY | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bin_low_0p7_0p9 | 300 | 289 (96.3%) | 300 | 300 | 289 | 11 | 8.64mm | 57.05mm | producer_bin |
| bin_mid_0p9_1p2 | 200 | 193 (96.5%) | 200 | 200 | 193 | 7 | 15.70mm | 295.96mm | producer_bin |
| bin_upper_1p2_1p6 | 300 | 58 (19.3%) | 294 | 294 | 58 | 242 | 232.40mm | 11140.39mm | rl_contribution_candidate_freeze |

## Script-only vs D319 diversity

| corpus | accepted | mean accepted XY | accepted XY variance | direction histogram |
| --- | --- | --- | --- | --- |
| script_0_999 accepted | 812 | 7.12mm | 14.21mm^2 | {"+x": 496, "+x/+y": 139, "+x/-y": 167, "+y": 6, "-x": 4} |
| d319 accepted | 540 | 10.33mm | 11.37mm^2 | {"+x_object_frame_commanded": 540} |

## Critical findings

- `bin_low_0p7_0p9` and `bin_mid_0p9_1p2` clear the >=30% generator gate.
- `bin_upper_1p2_1p6` is below the generator gate and should be frozen as an RL contribution candidate instead of hand-patching the controller.
- D319 accepted trajectories remain directionally narrow: the commanded primitive direction is fixed +x in object frame. This is acceptable for a fixture pilot but not sufficient for POSCO-style generalization.
- The 200-row replay manifest is a selection manifest only. Existing render tooling does not yet replay D319 D290 env rows into LeRobot v3 without an additional replay renderer.

JSON: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_data_conveyor_audit_summary.json`
Accepted rows: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_accepted_env_rows.csv`
Replay selection: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_selected_200_for_replay_manifest.csv`
