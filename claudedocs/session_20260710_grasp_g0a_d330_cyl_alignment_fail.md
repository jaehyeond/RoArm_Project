# D330 G0a cylinder alignment probe — FAIL

Date: 2026-07-10 KST

Verdict: `D330_G0A_CYL_ALIGNMENT_FAIL`

이번 case의 신규 변수: `[]` — D330 executes the D329-approved object redefinition to cylinder D34 x H90 and adds no extra ladder variable.

## Boot / Scope

- Current active case is G0a on the redefined cylinder D34 x H90. The D329 design requires a new probe file, no existing probe/env edits, probe-local cylinder override, fixed mass/friction, simple D327 radial 2-waypoint approach, and no waypoint search (`claudedocs/session_20260710_grasp_g0a_d329_object_mismatch_audit.md:105-144`).
- Professor direction says start with graspable shapes such as cylinders, while G0b is the first D34 x H90 grasp/lift case (`claudedocs/direction_20260708_grasp_pivot.md:5-18`).
- The base env still spawns the robot with contact sensors off by default (`roarm_rl/roarm_stack_env.py:146-150`), so D330 applies the requested probe-local `env_cfg.robot.spawn.activate_contact_sensors=True` override (`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:327-331`).
- Gripper convention remains LOW=OPEN, HIGH=CLOSED (`roarm_rl/roarm_stack_env.py:698-710`). D330 does not close the gripper.

## Implementation

- Added `sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py`.
- Cylinder override is probe-local: `CylinderCfg(radius=D/2, height=0.090, axis="Z")`, mass `0.72kg`, friction `1.5/1.2`, and object z at `TABLE_Z + 0.045` (`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:321-362`).
- Target formula is the pre-registered D329 formula: `TCP = cyl_center - radial*(D/2-10mm) - tangent*(D/2-8mm+2mm)`, TCP z = cylinder center (`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:401-430`).
- Motion remains the D327 radial 2-waypoint style: pre-target at `pre_clearance=0.040m`, then linear approach to final target; no waypoint search (`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:663-682`).
- Visualization DoD outputs were produced: snapshots for trials 1/5/10 and one Rerun v2 trace.

## Runtime Result

Output root: `claudedocs/runtime_logs/grasp_track/g0a_d330/`

Artifacts:

- `g0a_d330_cyl_alignment_summary.json`
- `g0a_d330_cyl_alignment_summary.md`
- `g0a_d330_cyl_alignment_trials.csv`
- `d330_cyl_alignment_trial_01_snapshot.png`
- `d330_cyl_alignment_trial_05_snapshot.png`
- `d330_cyl_alignment_trial_10_snapshot.png`
- `d330_cyl_alignment_trace_v2.rrd`

10-trial gate:

| metric | result |
|---|---:|
| pass_all | `0/10` |
| TCP pose failures | `8/10` |
| jaw tangent failures | `0/10` |
| fixed-jaw gap failures | `3/10` |
| penetration failures | `0/10` |
| contact-height failures | `4/10` |
| object-displacement failures | `10/10` |
| mean TCP error | `36.033mm` |
| min/max TCP error | `1.884mm / 80.530mm` |
| mean commanded TCP error | `0.404mm` |
| mean object XY displacement | `19.070mm` |

Trial-level table is in `claudedocs/runtime_logs/grasp_track/g0a_d330/g0a_d330_cyl_alignment_trials.csv`.

## Interpretation

- The D329 prediction was not supported. Cylinder geometry improved some trials compared with the cube stall, but did not produce the expected one-digit TCP error and `10/10` pass.
- This is not the same clean `~72mm` cube-stall signature in every env: trials 2 and 4 pass TCP position, while trials 5/7/9/10 still show `70-80mm` actual TCP error. The failure is mixed: execution/contact dynamics plus object disturbance, not only wrong object.
- The commanded FK is close to the target (`0.25-0.95mm`), so the offline/command target formula is not the primary failure. Runtime execution and contact/drive interaction remain suspect.
- The hard blocker for G0a is now object disturbance: displacement gate fails `10/10` (`8.872-39.456mm`) even when pose/gap/tangent gates pass in some trials. Alignment-only on the cylinder still moves the object too much.

## Contact Witness Audit

- Isaac Lab ContactSensor requires PhysX ContactReporter activation on the sensor body (`contact_sensor.py:35-46`) and then resolves body names into a PhysX rigid body/contact view (`contact_sensor.py:255-297`).
- D330 did set `activate_contact_sensors=True` on the robot spawn (`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:327-331`), but all robot-link ContactSensor witnesses failed to initialize:
  - robot net: `Failed to initialize contact reporter ... /World/envs/env_.*/Robot/(world|link1|link2|link3|link4|link5|gripper_link)`
  - `link4`, `link5`, `gripper_link`: same failure for exact link paths.
- I also tried two probe-local alternatives during the session and rejected them before finalizing:
  - scene-time per-link sensors failed at the same PhysX view initialization step.
  - an env0-only cylinder sensor initialized but broke `InteractiveScene.reset(env_ids)` because scene sensors must match the scene env index domain.
- Therefore the D330 force channel is not a valid zero-force result. It is a sensor-contract failure. The geometric/kinematic gate results above are valid, but contact-force attribution is still missing.

## Decision / Next

- G0a is not complete. Do not advance to G0b, gripper close, grasp/lift, RL/PPO, VLA, RoArm, B200, randomization, or cube reintroduction.
- The next valid G0a step is on the **correct cylinder object**, not the cube: repair the contact witness contract and diagnose why alignment-only motion displaces the cylinder. A minimal next discriminator should separate free-space execution, table/cylinder contact, and gripper-link contact on the cylinder without changing target offsets or gate thresholds.
- The D329 wrong-object audit remains useful because it eliminated the impossible 100mm cube grasp proxy, but D330 shows wrong-object was not a sufficient explanation for runtime failure.
