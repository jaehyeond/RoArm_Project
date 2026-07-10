# Session 2026-07-10 — D327 G0a Standoff + Execution Contract Diagnosis

Verdict: `D327_G0A_STANDOFF_EFFORT_REPAIR_FAIL`

이번 case의 ladder 신규 변수: `[]`. D327 stayed inside active case G0a. It
introduced one approved reactive alignment-target repair, `alignment_standoff_m =
0.002`, and one execution-contract repair, `arm_effort_limit_sim = 8.0`. It did
not advance the ladder, spawn a cylinder, close the gripper, grasp, lift, train
RL/PPO, render trajectories, use RoArm, B200, SSH, or VLA.

## Step 0 - Standoff Repair

D326 showed the D325/D324 position-only tangent-minus target was almost valid
under teleport-static evaluation, but failed the fixed-jaw no-penetration gate:

- D326 fixed-jaw gap: `-0.151mm`.
- D326 penetration: `0.151mm`.
- all other static gates passed.

D327 therefore separates the alignment target from the future grasp flush target:

- alignment target tangent offset: `D/2 - 8mm + 2mm`.
- future grasp flush formula remains: `D/2 - 8mm`.
- the `2mm` standoff is fixed for this repair; it was not tuned.

Environment manifest:

- `claudedocs/env_manifest_isaaclab_d327.txt`

Key verified pins:

- `numpy 1.26.0`
- `psutil 5.9.8`
- `rerun 0.34.1`

## Step 1 - Teleport Recheck

Result: teleport-static check passed.

| metric | value |
|---|---:|
| TCP error | `0.349mm` |
| commanded TCP error | `0.349mm` |
| jaw tangent error | `9.602deg` |
| fixed-jaw gap | `1.837mm` |
| fixed-jaw penetration | `0.000mm` |
| contact point below top | `49.733mm` |
| cube displacement | `0.000mm` |
| max arm joint error | `0.000000047rad` |

Interpretation: the D326 static blocker is repaired. G0a is now allowed to move
back to runtime execution diagnosis, but the pose family and gates remain
unchanged.

## Step 2 - Execution Diagnosis

Baseline runtime with the original approach/hold contract still failed:

- pass all: `0/10`.
- TCP pose failures: `10/10`.
- fixed-jaw gap failures: `10/10`.
- contact-height failures: `10/10`.
- jaw tangent, no-penetration, and cube displacement passed.
- baseline final TCP error for trial 1: `71.004mm`.
- commanded TCP error for trial 1: `0.926mm`.
- max torque saturation rate: `1.0`.
- final commanded-actual joint error: `0.143rad`.

The x3 step diagnostic did not improve the miss:

- trial 1 baseline final TCP error: `71.004mm`.
- trial 1 x3 TCP error: `72.719mm`.
- improvement: `-1.715mm`.

Registered diagnostic judgements:

| question | judgement | evidence |
|---|---:|---|
| time shortage | `False` | x3 did not reduce TCP error |
| lead limit | `False` | external override path bypasses env lead-limit rate |
| step clip budget | `False` | custom IK-to-external-target loop, not env step clip |
| joint/drive saturation | `True` | torque saturation `1.0`, final joint error `0.143rad` |

## Step 3 - One Repair Retest

Applied exactly one execution-contract repair:

- `arm_effort_limit_sim: 2.5 -> 8.0`

Final 10-trial result:

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

Failure counts after the single repair:

- TCP pose: `10/10`.
- fixed-jaw gap: `10/10`.
- contact height: `10/10`.
- jaw tangent: `0/10`.
- fixed-jaw penetration: `0/10`.
- cube displacement: `0/10`.

Interpretation: effort alone is not the missing contract. The command-side IK
target is valid (`cmd pos` stays below about `1mm`), but the runtime arm remains
roughly `68-71mm` high/back from the target. The next G0a work should inspect
the position actuator/drive semantics, external target override path, stiffness
or drive mode, and commanded-vs-actual joint evolution in the `.rrd`; do not
tune the G0a target, epsilon, gates, or advance to G0b.

## Visualization

Visualization DoD was satisfied:

- teleport static RRD: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_teleport_static_v2.rrd`
- baseline motion RRD: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_baseline_trace_v2.rrd`
- final repaired motion RRD: `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_trace_v2.rrd`
- headless final Rerun screenshot:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/d327_final_effort_retest_v2_rerun_open.png`
- decision snapshots for teleport, baseline, x3, and final trials are in:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/`

## Artifacts

- Probe: `sim_scripts/cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe.py`
- Summary JSON:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/g0a_d327_standoff_execution_summary.json`
- Summary MD:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/g0a_d327_standoff_execution_summary.md`
- Final trial CSV:
  `claudedocs/runtime_logs/grasp_track/g0a_d327/g0a_d327_final_retest_trials.csv`

## Verification

- `python -m py_compile roarm_rl/viz_debug.py sim_scripts/cube10cm_top_view_d327_grasp_g0a_standoff_execution_probe.py` PASS.
- `conda run -n isaaclab rerun ... --headless --screenshot-to ...` PASS for the final repaired trace.
- No B200/JHPark/SSH/pull/.ssh path was used.
