# D327 G0a Standoff Execution Probe

이번 case의 ladder 신규 변수: `[]` — D327 is a G0a reactive standoff repair plus one execution-contract repair.

- verdict: `D327_G0A_STANDOFF_EFFORT_REPAIR_FAIL`
- alignment standoff: `2.000mm`
- teleport pass: `True`
- selected repair: `arm_effort_limit_2p5_to_8p0`
- final pass_all: `0/10`

## Diagnostic Questions

- joint_or_drive_saturation: prediction=possible if applied torque sits at effort limit or final commanded-actual joint error remains high; judgement=`True`; evidence={'final_joint_err_rad': 0.1431235373020172, 'torque_saturation_rate_max': 1.0}
- lead_limit: prediction=unlikely: D325 external override bypasses env joint_target_lead_limit_rad; judgement=`False`; evidence=roarm_cube_push_env._pre_physics_step override path directly writes robot_dof_targets and zeroes lead-limit rate
- step_clip_budget: prediction=unlikely as a 0.010rad env step-clip issue because D325 uses a custom IK-to-external-target loop; judgement=`False`; evidence={'max_step_command_delta_rad': 0.6313783490564182, 'note': 'D325 loop does not call env candidate6_diffik_step_clip_rad; IK solver has its own 4deg clip.'}
- time_shortage: prediction=likely if baseline error is still decreasing or x3 reduces TCP error materially; judgement=`False`; evidence={'baseline_final_minus_mid_mm': 0.541982187052767, 'baseline_final_tcp_error_mm': 71.00439545019697, 'improvement_mm': -1.7147884103404465, 'x3_final_tcp_error_mm': 72.71918386053741}

## Final 10-Trial Table

| trial | pos mm | cmd pos mm | tangent deg | gap mm | top clearance mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 69.810 | 0.238 | 9.594 | 11.848 | -8.203 | 0.046 | False |
| 2 | 67.828 | 0.928 | 9.758 | 13.529 | -7.040 | 0.048 | False |
| 3 | 69.851 | 0.237 | 10.939 | 13.374 | -7.539 | 0.046 | False |
| 4 | 70.849 | 0.927 | 10.568 | 12.719 | -8.000 | 0.022 | False |
| 5 | 69.527 | 0.926 | 9.894 | 13.237 | -7.190 | 0.021 | False |
| 6 | 69.002 | 0.236 | 10.427 | 13.013 | -7.490 | 0.021 | False |
| 7 | 68.590 | 0.236 | 10.940 | 13.692 | -7.024 | 0.020 | False |
| 8 | 67.556 | 0.236 | 9.950 | 13.450 | -7.043 | 0.023 | False |
| 9 | 69.037 | 0.397 | 11.210 | 13.355 | -7.284 | 0.023 | False |
| 10 | 69.260 | 0.234 | 11.306 | 13.145 | -7.303 | 0.022 | False |

## Artifacts

- teleport_check snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_teleport_static_check.png`
- teleport_check rrd: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_teleport_static_v2.rrd`
- baseline_diagnostic trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_baseline_trial_01_snapshot.png`
- baseline_diagnostic trial 5 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_baseline_trial_05_snapshot.png`
- baseline_diagnostic trial 10 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_baseline_trial_10_snapshot.png`
- baseline_diagnostic rrd: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_baseline_trace_v2.rrd`
- x3_diagnostic trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_x3_diagnostic_trial_01_snapshot.png`
- x3_diagnostic trial 5 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_x3_diagnostic_trial_05_snapshot.png`
- x3_diagnostic trial 10 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_x3_diagnostic_trial_10_snapshot.png`
- final_retest trial 1 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trial_01_snapshot.png`
- final_retest trial 5 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trial_05_snapshot.png`
- final_retest trial 10 snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trial_10_snapshot.png`
- final_retest rrd: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trace_v2.rrd`
