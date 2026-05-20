# Session 2026-05-20 — P7 Branch B cube contact state repair

## Scope

Track A P7/Branch B only. Track B CoRL paper remains separate.

This session repaired stale project-state docs, cross-checked the current cube
grasp state against B200 logs, ran one B200 URDF-to-USD conversion-only retry
for v7 using the D024 userspace override path, and then ran the separately
approved v7 close_26 runtime jaw telemetry. It did not run training, generate
datasets, run hold-lift, integrate constraints/defaults, attach SurfaceGripper,
transport, release, tune P7 scalar/gates, or continue the old 2cm sweep.

## Why The State Docs Were Wrong

The prior rolling state was stale relative to later B200 work:

- `START_HERE.md` line 60 in the previous version still claimed v4 USD
  conversion had not been run.
- `claudedocs/EXPERIMENT_LEDGER.md` row 73 stopped at v4 static-only /
  physics-unvalidated.
- `claudedocs/session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md`
  lines 193-204 also said v4 USD conversion did not exist yet.

The process failure was not that B200 logs were absent. The logs existed under
`/tmp`. The failure was that the latest B200 results were not promoted into the
rolling state system before the session ended. `CLAUDE.md` requires
`START_HERE.md` overwrite, ledger append, optional durable decision append, new
session doc, and cross-verification at session close. That closure step had not
been completed for the later v4/v5/v6/v7 work.

`START_HERE.md` had also grown to 782 lines, despite `CLAUDE.md` asking for a
short dashboard around 120 lines. That made stale history look like current
truth.

## Worktree At Repair Start

Expected dirty/untracked state:

- Modified:
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `8a7b987d5777f8ea2473270d1058246d`.
- Untracked:
  `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
  md5 `598c7ac68f0844143ac9589c18c2b7e6`.
- Untracked:
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
  md5 `dd1e4723b2930fc7795c65cd104e4587`.

These were not reverted.

## Verified Local Code Facts

`sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`:

- Lines 2-8: diagnostic/static-only scope; no Isaac, training, dataset,
  defaults, constraints, SurfaceGripper, transport, or release.
- Lines 50-55: v7 writes to
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_urdf`.
- Lines 108-113: fixed counter jaw is mounted to `link5`, not `gripper_link`.
- Lines 165-191: default v7 candidate is 3cm cube, yaw 0, close_26,
  fixed-counter clearance `0.0021m`, fixed slop `0.0010m`.
- Lines 225-245: moving and fixed-counter centers are computed in object frame
  then transformed to gripper/link5 local frames.
- Lines 296-328: static open-waypoint and close_26 contact checks.
- Lines 350-361: gates report open descent clearance, moving contact, fixed
  slop contact, authoring-offset and dynamic-push intent, and require separate
  approval for Isaac runtime.

`sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`:

- Lines 54-56: v4/v5/v6 runtime telemetry variants use versioned USD paths.
- Lines 204-224: variant defaults select v5/v6 geometry and USD path.
- Line 289: runtime sets `ROARM_M3_USD_PATH` from the selected USD.

`roarm_rl/roarm_stack_env.py`:

- Lines 97-100: env loads robot USD from `ROARM_M3_USD_PATH`, with the default
  asset path only as fallback.

Implication: URDF is the editable/prep source for each candidate. USD is the
converted Isaac runtime asset. Candidate versions can and should have separate
URDF prep outputs and separate USD output directories.

## Verified B200 Facts

### v4

- Conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_convert_b200.out:82` reports
  `cube2cm_counter_jaw_v4_link` merged into `gripper_link`.
- Root USD md5:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/roarm_m3.usd`
  `4497024d25abab11de5c50e144124553`.
- Payload md5s:
  - `config.yaml` `9488710ffa036ff1ec19bd33c2bfdf88`
  - `configuration/roarm_m3_base.usd` `f8fe06c48b64b994341a5a6ac8565f69`
  - `configuration/roarm_m3_physics.usd` `1636dd14c96caad828a39e5656766399`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- Close-hold-lift failed:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out:390-391`
  reports `reached=NO`, `final_target_error_m=0.018023`, and
  `verdict=LATCH_FAIL`.
- Runtime telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v4_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, `reached=NO`, and `success_claim=NO`.

### v5

- Static prep succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_urdf_prep_b200.out:25,28`
  reports moving/counter contact at close_26 and
  `static_fixture_pair_plausible=YES`.
- Conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_convert_b200.out:82,84,86`.
- Runtime telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v5_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, and `success_claim=NO`.

### v6

- Static prep succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_urdf_prep_b200.out:1-2,5,23-26`.
- Conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_convert_b200.out:82`.
- v6 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `56e13505eec52e1f17552421d40525ca`
  - `configuration/roarm_m3_base.usd` `e9976dbdce4cca0be2daf5630381210b`
  - `configuration/roarm_m3_physics.usd` `649b02bdeee3af67baf74b19b4501ae2`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- First v6 telemetry without LD_PRELOAD crashed and is not physics evidence.
- Valid LD_PRELOAD runtime telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v6_ldpreload_b200.out:417`
  reports `target_error_m=0.024584`, moving jaw contact, counter gap
  `[+0.000260,+0.006757,+0]`, `counter_contact=NO`,
  `one_sided_push=YES`, and `reached=NO`.
- `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v6_ldpreload_b200.out:418`
  reports `success_claim=NO`.

### v7

- B200 static prep succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:1-2,5-7,23-28`.
- v7 prep md5s:
  - output URDF `307384a6e18b3c70ef8c74a6c0524148`
  - moving mesh `b8cc44b72cf1d89404f9df3adf079b17`
  - counter mesh `7e56e1c6c3d35a8e3af1fe98d5f8e0cc`
- v7 prep line 27 reports:
  `open_descent_clearance=YES`,
  `moving_contact_at_design_close=YES`,
  `fixed_counter_slop_contact_at_design_close=YES`,
  and `v7_candidate_static_plausible=YES`.
- Normal conversion crashed:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_b200.err:1-7,64-66,87-90`
  shows NVML driver/library mismatch and GLX/NVIDIA segfault.
- LD_PRELOAD conversion also crashed:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_ldpreload_b200.err:1-7,64-66,87-90`
  shows the same class of failure.
- D024 conversion-only retry succeeded using
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- D024 stdout:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out:84-89`
  reports `cube2cm_fixed_counter_jaw_v7_link` merged into `link5`,
  `hand_tcp` merged into `link5`, and `base_link` merged into `world`.
- D024 stderr:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err:1-6`
  contains cpufreq and `NVML_ERROR_UNINITIALIZED` messages only; grep found no
  traceback, exception, fatal, segfault, or driver/library mismatch.
- v7 D024 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `f2777880ff2c90182484d82b7f49e5a6`
  - `configuration/roarm_m3_base.usd` `d7aae34ddca6a4d4f1ce092bda28d1a2`
  - `configuration/roarm_m3_physics.usd` `75f7b1e6da1f5f14019a53f091ec2076`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`

### v7 Approved Runtime Jaw Telemetry

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py` was patched
  for v7 diagnostic telemetry:
  - v7 D024 USD path;
  - `counter_parent=link5`;
  - actual runtime link5 transform for the fixed counter jaw;
  - strict contact and 1mm slop contact logged separately.
- Patched telemetry md5:
  `0b4d3f579d3bb56f994983a876198d65`.
- Local and B200 `py_compile` passed; B200 md5 matched.
- B200 runtime command was close_26-only and used the D024 userspace overrides.
- B200 scope:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38`
  confirms diagnostic-only, `variant=v7`, D024 USD path, no training, no
  constraints, no SurfaceGripper, no transport/release, no gate tuning,
  close_26-only, and `claim_p7_success=NO`.
- B200 selected pose:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:39`
  reports 3cm cube, yaw 0, `ik_ok=YES`, and
  `max_fk_error_m=0.000518`.
- B200 authored geometry:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:68`
  reports `counter_parent=link5`,
  `design_moving_center_ref=([+0.000000,+0.014250,+0.002000])`,
  `design_counter_center_ref=([+0.000000,-0.019600,+0.002000])`,
  and `counter_contact_slop_m=0.001000`.
- B200 final close step:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419`
  reports `target_error_m=0.023422`, `moving_contact=YES`,
  `counter_contact=NO`, `moving_slop_contact=YES`,
  `counter_slop_contact=NO`, `one_sided_push=YES`, and `reached=NO`.
- B200 aggregate:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:420`
  reports `approach_ok=YES`, `descend_ok=YES`, `close_reached=NO`,
  `grasped_seen=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- B200 completion:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:421`.
- B200 stderr:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.err:1-4`
  contains cpufreq, NVML-uninitialized, and Fabric messages only; grep found no
  traceback, exception, fatal, segfault, or driver/library mismatch.
- Post-run process check found no matching P7/Isaac/conversion/training process.
- Log md5s:
  - stdout `3939f08ea684c34f76669293b96610ba`
  - stderr `a0cb0d2eb0dca684599e693fcd1e7af7`

## Interpretation

- This is not physics success.
- v4/v5/v6 conversion and static plausibility did not produce stable runtime
  pinch. Runtime telemetry shows one-sided pushing.
- TCP-only IK is not the blocker; approach/descent IK succeeded in the telemetry
  logs. The failure is close-time contact/dynamics.
- Correct framing: the current Isaac rigid-cube/jaw collision/contact proxy is
  not reproducing real foam grasp. Do not say the real robot cannot grasp the
  cube.
- v7 addresses both suspected authoring offset and dynamic-push issues in a
  static object-frame way. It is static/prep-valid and USD-converted under D024,
  but approved close_26 runtime telemetry still failed.
- The v7 fixed-counter/backstop did not resolve the rigid-cube one-sided-push
  failure. At final close the moving jaw contacted the cube, but the counter jaw
  had neither strict contact nor 1mm slop contact.

## State Docs Updated

- Rewrote `START_HERE.md` into a shorter current dashboard.
- Appended latest v4/v5/v6 correction and v7 static/conversion-blocked rows to
  `claudedocs/EXPERIMENT_LEDGER.md`.
- Added `claudedocs/DECISIONS.md` D054: conversion, static contact, and runtime
  physics are separate gates.
- Added `claudedocs/DECISIONS.md` D055: v7 conversion requires the D024 B200
  override path and still does not validate grasp physics.
- Added `claudedocs/DECISIONS.md` D056: v7 close_26 runtime telemetry failed,
  so hold-lift is not justified.
- Wrote this session doc as the current detailed reference.

## Next Direction

Recommended next action is to **stop before hold-lift and new runtime gates**:

1. Keep v7 static/prep-valid, D024 USD-converted, and runtime-telemetry failed.
2. Do not interpret USD conversion or telemetry exit code as grasp success.
3. Do not run hold-lift from this state; close-time contact/dynamics failed.
4. Next technical work should be analytical/modeling: decide whether more rigid
   proxy probing is informative, or whether to explicitly model foam/contact
   compliance before any dataset generation.
5. Any further runtime gate still requires separate explicit approval.

Not approved:

- Training or VLA/SmolVLA/LeRobot training.
- Cube sim dataset generation.
- Runtime telemetry or hold-lift.
- Transport, transport target, release, or scripted release.
- Constraint/default integration.
- SurfaceGripper.
- P7 scalar/gate/release guidance tuning.
