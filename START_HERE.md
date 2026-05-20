# START_HERE.md

Last updated: 2026-05-20 KST (Track A P7/Branch B cube grasp state repair)

This is the rolling current-state dashboard. It is not full history.
Durable rules live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## Current Truth

The project is two-track:

- **Track A**: existing sim/lab stacking work. Current active line is P7/Branch B
  normalized cube grasp, currently focused on rigid-cube/jaw contact proxy
  diagnosis and static object-frame asset candidates.
- **Track B**: CoRL 2026 paper sprint. Keep separate unless explicitly asked.

Important correction: earlier `START_HERE.md`, ledger row 73, and
`session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md` are stale
relative to later B200 logs. They still say v4 USD conversion was not run and
v4 physics was unvalidated. That is no longer the current evidence.

## Latest Verified B200 Evidence

v4:

- Conversion succeeded. B200
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_convert_b200.out:82` reports
  `cube2cm_counter_jaw_v4_link` merged into `gripper_link`.
- v4 root USD md5:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/roarm_m3.usd`
  `4497024d25abab11de5c50e144124553`.
- Physics/telemetry failed. B200 close-hold-lift
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out:390-391`
  reports `reached=NO` and `verdict=LATCH_FAIL`.
- Runtime jaw telemetry
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v4_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, and `success_claim=NO`.

v5:

- Static prep and conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_urdf_prep_b200.out:25,28`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_convert_b200.out:82,84,86`.
- Runtime jaw telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v5_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, and `success_claim=NO`.

v6:

- Static prep succeeded for the 3cm candidate:
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
- Valid LD_PRELOAD telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v6_ldpreload_b200.out:417-418`
  reports `target_error_m=0.024584`, `moving_contact=YES`,
  `counter_contact=NO`, `one_sided_push=YES`, `reached=NO`, and
  `success_claim=NO`.

v7:

- Local/static v7 object-frame analysis exists as an untracked script:
  `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
  md5 `598c7ac68f0844143ac9589c18c2b7e6`.
- v7 static/prep script exists as an untracked script:
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
  md5 `dd1e4723b2930fc7795c65cd104e4587`.
- v7 prep is diagnostic/static only. It mounts a fixed counter jaw on `link5`
  and keeps the moving jaw on `gripper_link`; see
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`.
- B200 v7 prep succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:23-28`
  reports open descent clearance, moving contact, fixed-counter slop contact,
  and `v7_candidate_static_plausible=YES`.
- v7 prep md5s:
  - URDF `307384a6e18b3c70ef8c74a6c0524148`
  - moving mesh `b8cc44b72cf1d89404f9df3adf079b17`
  - counter mesh `7e56e1c6c3d35a8e3af1fe98d5f8e0cc`
- Initial v7 conversion attempts did **not** succeed. Normal and wrong-library
  LD_PRELOAD conversion attempts both crashed with B200 NVIDIA/NVML/GLX
  driver-library mismatch:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_b200.err:1-7,64-66,87-90`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_ldpreload_b200.err:1-7,64-66,87-90`.
- D024 conversion-only retry succeeded with the matching B200 userspace
  overrides:
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- D024 conversion stdout:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out:84-89`
  reports `cube2cm_fixed_counter_jaw_v7_link` merged into `link5`,
  `hand_tcp` merged into `link5`, and `base_link` merged into `world`.
- D024 conversion stderr:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err:1-6`
  contains cpufreq/NVML-uninitialized messages only; grep found no traceback,
  exception, fatal, segfault, or driver/library mismatch.
- v7 D024 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `f2777880ff2c90182484d82b7f49e5a6`
  - `configuration/roarm_m3_base.usd` `d7aae34ddca6a4d4f1ce092bda28d1a2`
  - `configuration/roarm_m3_physics.usd` `75f7b1e6da1f5f14019a53f091ec2076`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- Therefore v7 is static/prep-valid and USD-converted under D024, but still
  was physics-unvalidated before the approved runtime telemetry below.
- Approved v7 D024 runtime jaw telemetry was run close_26-only:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out`.
  Scope line 38 confirms telemetry-only, no training, no constraints,
  no SurfaceGripper, no transport/release, no gate tuning, and no success claim.
- v7 telemetry selected the 3cm cube and IK was OK:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:39`.
- Authored v7 runtime geometry used `counter_parent=link5`:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:68`.
- Final close_26 telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419`
  reports `target_error_m=0.023422`, `moving_contact=YES`,
  `counter_contact=NO`, `moving_slop_contact=YES`,
  `counter_slop_contact=NO`, `one_sided_push=YES`, and `reached=NO`.
- Aggregate line 420 reports `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- v7 runtime stderr
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.err:1-4`
  contains cpufreq/NVML-uninitialized/Fabric messages; grep found no traceback,
  exception, fatal, segfault, or driver/library mismatch. Post-run process check
  was empty.
- v7 runtime log md5s:
  - stdout `3939f08ea684c34f76669293b96610ba`
  - stderr `a0cb0d2eb0dca684599e693fcd1e7af7`

## Current Interpretation

- This is **not** physics success.
- TCP-only IK is not the current blocker: approach/descent IK succeeded in the
  v4/v5/v6/v7 telemetry logs.
- The current Isaac rigid-cube/jaw collision/contact proxy is not reproducing
  real foam grasp. The rigid cube is pushed by the moving jaw before counter
  contact closes, producing one-sided push.
- Do not say or imply that the real robot cannot grasp the cube.
- Conversion success is only asset import success. It does not validate grasp
  physics.
- v7 link5-fixed counter/backstop did not resolve the runtime rigid-contact
  failure. It remains useful evidence for the proxy mismatch, not a solved
  grasp primitive.

## Current Worktree

Expected local state:

- Modified and expected:
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `0b4d3f579d3bb56f994983a876198d65`.
- Untracked and expected:
  `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
  md5 `598c7ac68f0844143ac9589c18c2b7e6`.
- Untracked and expected:
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
  md5 `dd1e4723b2930fc7795c65cd104e4587`.

Do not revert these without explicit user approval.

## Current Direction

Immediate priority: keep the state docs synchronized with verified B200 logs so
new sessions do not restart from stale v4 assumptions.

Next technical action after state repair:

1. Do not run training, dataset generation, runtime telemetry, hold-lift,
   transport/release, constraints, or SurfaceGripper.
2. Treat v7 as static/prep-valid, D024 USD-converted, and runtime-telemetry
   failed at close_26.
3. Do not interpret v7 conversion or telemetry exit code as grasp success.
4. Stop here unless the user gives separate explicit approval for a new gate.
5. Hold-lift remains not approved and is not justified by this failed telemetry.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` latest P7/Branch B decisions
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260520_p7_branch_b_cube_contact_state_repair.md`
6. `claudedocs/session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md`
7. `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
8. `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
9. `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
10. The B200 `/tmp` logs cited above, with line numbers.

## Explicitly Not Approved

- Training or VLA/SmolVLA/LeRobot training.
- Cube sim dataset generation.
- Old handoff diagnostic.
- Constraint/default integration.
- SurfaceGripper attachment.
- Transport, transport target, release, or scripted release variants.
- P7 scalar/threshold/release guidance tuning.
- Diagnostic gate tuning.
- Old 2cm sweep continuation.
- Claiming physics success from v4/v5/v6 conversion.
- Hold-lift or runtime telemetry unless separately approved.
