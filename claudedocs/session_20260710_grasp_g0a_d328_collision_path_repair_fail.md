# D328 G0a Collision-vs-Drive Diagnosis

Date: 2026-07-10 KST

이번 case의 ladder 신규 변수: `[]` -- D328 is a G0a runtime-stall diagnosis and one branch repair. It does not advance to G0b and does not change the pose family, standoff, object, friction, gates, gripper state, RL/PPO, VLA, or robot deployment.

Verdict: `D328_G0A_COLLISION_DRIVE_REPAIR_FAIL`

## Objective

Distinguish the D327 runtime stall cause:

- Hypothesis A, collision/path: if the cube is removed, the same runtime approach reaches the D325/D327 TCP target under `5mm`.
- Hypothesis B, drive/override semantics: if the cube is removed, the same runtime approach still stalls around the prior `~70mm` TCP error.

After the branch is decided, apply exactly one branch repair and rerun the D325 four-condition 10-trial G0a gate.

## Execution Note

The first all-in-one Isaac run completed the cube-removed stage but then hung during subsequent environment lifecycle. The same experiment was rerun as separate registered stages:

- `--stage cube_removed`
- `--stage evidence`
- `--stage candidate_paths`
- `--stage final_collision --final_collision_waypoint_mode far_side_slide`
- `--stage summarize`

This changed execution packaging only; it did not change the registered variables or gates.

## Step 1 - Cube Removed Decision

Result: branch A confirmed.

| Condition | TCP error | Commanded TCP error | Joint error | Judgement |
|---|---:|---:|---:|---|
| Cube removed | `1.512mm` | `0.927mm` | `0.00193rad` | reaches target under `5mm` |
| Cube present evidence | `72.178mm` | `0.927mm` | `0.132rad` | stalls near prior D327 error |

Interpretation:

- The same commanded target is valid and reachable when the cube is moved out of the workspace.
- The D327/D328 stall is therefore not primarily "external target override cannot move this pose" or "the target is unreachable in free space."
- The proximate blocker is the cube-present path/contact geometry. Drive saturation is still observed, but D328 makes it look like an effect of pushing into a blocked path, not the root cause to tune blindly.

## Step 2 - Evidence

Cube-present evidence run:

- TCP error: `72.178mm`.
- Commanded TCP error: `0.927mm`.
- Fixed-jaw gap: `13.184mm`.
- Contact point below cube top: `-9.199mm` (above the allowed side-contact height gate).
- Torque saturation max: `1.0`.
- Torque saturation final: `0.8`.
- ContactSensor status: `ok=True`, mode `robot_net_forces_w`, prim path `/World/envs/env_.*/Robot/.*`.
- Logged max contact force was `0.000N`.

Critical note:

The contact force channel is not reliable evidence in this D328 run: it reports `0.000N` even though the cube-removal contrast decisively changes the outcome. Treat the Step 1 contrast and Rerun/snapshot geometry as the decision evidence; do not infer "no contact" from the force trace alone.

Drive audit in the staged runs reported arm joint values:

- stiffness: `[80, 80, 80, 80, 80]`
- damping: `[4, 4, 4, 4, 4]`
- effort limits: `[2.5, 2.5, 2.5, 2.5, 2.5]`
- velocity limits: `[3.14, 3.14, 3.14, 3.14, 3.14]`

This audit is recorded, but D328 does not apply a drive repair because branch A was selected.

## Step 3 - Collision Branch Repair

Three waypoint candidates were checked by geometry/IK before runtime repair:

| Candidate | IK converged | Max IK error | Min approach TCP-over-top clearance | Waypoints |
|---|---:|---:|---:|---:|
| `d327_radial` | true | `0.968mm` | `70.000mm` | 2 |
| `far_side_slide` | true | `0.910mm` | `70.000mm` | 3 |
| `high_corridor_drop` | true | `0.953mm` | `20.264mm` | 3 |

Selected repair: `far_side_slide`.

Selection reason: among IK-feasible candidates, it tied the best approach clearance and had the lower max IK error than `d327_radial`.

Limitation:

The clearance metric is an approach-corridor proxy. It excludes the final waypoint because D327 teleport-static already showed the final standoff pose is geometrically valid. It is not yet a full moving-jaw collision sweep.

## Step 4 - Final 10-Trial Rejudgement

Result: `0/10` pass-all.

| Gate | Failures |
|---|---:|
| TCP pose | `10/10` |
| Jaw tangent | `0/10` |
| Fixed-jaw gap | `10/10` |
| No penetration | `0/10` |
| Contact height | `10/10` |
| Cube displacement | `0/10` |

Final 10-trial table:

| trial | TCP error mm | commanded TCP error mm | tangent deg | gap mm | contact below top mm | cube disp mm | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 59.250 | 0.442 | 12.783 | 21.191 | -3.747 | 0.048 | false |
| 2 | 59.142 | 0.442 | 12.730 | 20.809 | -3.909 | 0.048 | false |
| 3 | 58.713 | 0.440 | 13.127 | 21.249 | -3.452 | 0.048 | false |
| 4 | 59.017 | 0.660 | 12.909 | 21.354 | -3.427 | 0.023 | false |
| 5 | 59.379 | 0.441 | 12.626 | 21.232 | -3.608 | 0.023 | false |
| 6 | 59.369 | 0.661 | 12.519 | 21.366 | -3.657 | 0.023 | false |
| 7 | 59.130 | 0.442 | 12.784 | 21.406 | -3.609 | 0.023 | false |
| 8 | 59.323 | 0.440 | 13.104 | 21.118 | -3.529 | 0.023 | false |
| 9 | 58.656 | 0.659 | 12.399 | 21.200 | -3.431 | 0.023 | false |
| 10 | 58.876 | 0.441 | 12.805 | 21.346 | -3.487 | 0.024 | false |

The path repair improved the cube-present TCP error from the evidence-stage `72.178mm` to roughly `58.7-59.4mm`, but it did not reach the D325 gate. The persistent failures are still TCP height/side gap/contact-height, while tangent, no-penetration, and cube displacement remain controlled.

## Artifacts

- Script: `sim_scripts/cube10cm_top_view_d328_grasp_g0a_collision_vs_drive_probe.py`
- Summary: `claudedocs/runtime_logs/grasp_track/g0a_d328/g0a_d328_collision_vs_drive_summary.md`
- Summary JSON: `claudedocs/runtime_logs/grasp_track/g0a_d328/g0a_d328_collision_vs_drive_summary.json`
- Final CSV: `claudedocs/runtime_logs/grasp_track/g0a_d328/g0a_d328_final_retest_trials.csv`
- Cube-removed snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_removed_decision_trial_01_snapshot.png`
- Cube-removed RRD: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_removed_decision_trace_v2.rrd`
- Cube-present evidence snapshot: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_present_evidence_trial_01_snapshot.png`
- Cube-present evidence RRD: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_cube_present_evidence_trace_v2.rrd`
- Final retest snapshots:
  - `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_01_snapshot.png`
  - `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_05_snapshot.png`
  - `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trial_10_snapshot.png`
- Final retest RRD: `claudedocs/runtime_logs/grasp_track/g0a_d328/d328_final_collision_path_retest_trace_v2.rrd`

## Implication

D328 narrows the blocker:

- Static target geometry is valid with the D327 `2mm` standoff.
- Free-space runtime reaches the target under `5mm`.
- Cube-present runtime stalls high/sideward.
- A simple waypoint repair is insufficient.

Next valid G0a work is not G0b, not drive/effort tuning, not standoff/gate tuning, and not another blind path tweak. The next decision-changing repair should audit the true open-gripper collision/sweep geometry and contact instrumentation: fixed jaw, moving jaw swing, link5, cube, and table. The ContactSensor path used in D328 is not adequate as the only contact witness because it returned zero force in a case where cube removal changed the outcome.
