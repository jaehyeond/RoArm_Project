# D328 G0a Collision-vs-Drive Probe

이번 case의 ladder 신규 변수: `[]` — D328 is a G0a runtime-stall diagnosis and one branch repair.

- verdict: `D328_G0A_COLLISION_DRIVE_REPAIR_FAIL`
- branch: `A_collision_path`
- selected repair: `waypoint_path_repair`
- final pass_all: `0/10`

## Step 1 Decision Experiment

- prediction A: `cube removed -> TCP error <5mm if path collision is the blocker`
- prediction B: `cube removed -> ~70mm stall if drive/override semantics are the blocker`
- final TCP error: `1.512mm`
- judgement: `cube_removed_reaches_tcp_under_5mm_collision_path_confirmed`

## Evidence Trial

- max contact force: `0.000N`
- torque saturation max: `1.0`
- contact sensor status: `{'mode': 'robot_net_forces_w', 'ok': True, 'prim_path': '/World/envs/env_.*/Robot/.*'}`

## Final 10-Trial Table

| trial | pos mm | cmd pos mm | tangent deg | gap mm | top clearance mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 59.250 | 0.442 | 12.783 | 21.191 | -3.747 | 0.048 | False |
| 2 | 59.142 | 0.442 | 12.730 | 20.809 | -3.909 | 0.048 | False |
| 3 | 58.713 | 0.440 | 13.127 | 21.249 | -3.452 | 0.048 | False |
| 4 | 59.017 | 0.660 | 12.909 | 21.354 | -3.427 | 0.023 | False |
| 5 | 59.379 | 0.441 | 12.626 | 21.232 | -3.608 | 0.023 | False |
| 6 | 59.369 | 0.661 | 12.519 | 21.366 | -3.657 | 0.023 | False |
| 7 | 59.130 | 0.442 | 12.784 | 21.406 | -3.609 | 0.023 | False |
| 8 | 59.323 | 0.440 | 13.104 | 21.118 | -3.529 | 0.023 | False |
| 9 | 58.656 | 0.659 | 12.399 | 21.200 | -3.431 | 0.023 | False |
| 10 | 58.876 | 0.441 | 12.805 | 21.346 | -3.487 | 0.024 | False |

## Artifacts

- step1_cube_removed trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_removed_decision_trial_01_snapshot.png`
- step1_cube_removed rrd: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_removed_decision_trace_v2.rrd`
- cube_present_evidence trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_present_evidence_trial_01_snapshot.png`
- cube_present_evidence rrd: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_present_evidence_trace_v2.rrd`
- final_retest trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_01_snapshot.png`
- final_retest trial 5 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_05_snapshot.png`
- final_retest trial 10 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_10_snapshot.png`
- final_retest rrd: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trace_v2.rrd`
