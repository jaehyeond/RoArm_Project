# Session 2026-05-20 - P7 Branch B normalized 2cm cube grasp feedback

## Scope

- Continued Track A P7/Branch B only.
- Recorded the user's corrected report of the professor's latest feedback.
- Did not train.
- Did not run Isaac.
- Did not execute the old handoff diagnostic.
- Did not integrate constraints into RoArm chain defaults.
- Did not attach SurfaceGripper.
- Did not execute transport, transport target, release, or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not revert existing dirty/untracked state.
- Did not use `HANDOFF.md` or `TASKS.md`.

## Correction

The earlier assistant response incorrectly summarized an old presentation/render
feedback item as the professor's "1/2/3" feedback. The current important
feedback is instead about normalizing the grasp problem around a real
`2cm x 2cm x 2cm` cube sponge and using that normalized primitive to scale sim
demonstration generation.

This correction is now recorded as current Track A context.

## Professor Feedback - Step-by-Step Interpretation

1. Normalize the object before scaling the task.

   - The long sponge should not remain the only target geometry.
   - The user physically cut the sponge into a cube: `2cm x 2cm x 2cm`.
   - This cube should be treated as a canonical object for grasp primitive design.
   - The reason is not cosmetic. A normalized object frame makes future object
     variations and stacking heights easier to parameterize.

2. Do not command the gripper TCP blindly to the cube center.

   - A 2cm cube has its center 1cm from each face.
   - If the gripper is not opened and the end/TCP is driven toward the cube center,
     the gripper geometry can press into or crush the sponge before a valid pinch.
   - Therefore the target should be an object-frame grasp pose, not simply the cube
     center.
   - The fixed-side gripper geometry must be accounted for: one side of the gripper
     is effectively fixed, so the grasp pose should be laterally offset from the
     cube center by the known jaw geometry/position.

3. Build a canonical open-descend-close-lift primitive.

   The intended primitive is:

   - define cube pose and size;
   - compute fixed-jaw and moving-jaw positions relative to the cube frame;
   - open the gripper before descent;
   - descend with the gripper already offset so the fixed jaw is near one cube side;
   - sweep gripper opening/closing angles and contact heights;
   - close around the cube;
   - lift and verify physical follow.

   `_grasped_marker=YES` is not sufficient. Success still requires:

   - target reached;
   - stable hold;
   - lift follow;
   - bounded object drift/speed/tilt;
   - `posewrite_calls=0`;
   - no hidden kinematic artifact.

4. Treat height and yaw as dataset-generation variables.

   - `x/y` are planar translations.
   - `z` is a variable because stacking needs grasps at multiple heights/layers.
   - yaw/rotation should also be varied.
   - The same normalized grasp primitive should work for table height and stacked
     layers if the object-frame pose and gripper offset are computed correctly.

5. Use the primitive to scale sim demonstrations on B200.

   The professor's direction is to avoid spending months collecting VLA imitation
   data only on the real robot. Instead:

   - validate the normalized cube grasp primitive in sim;
   - generate many procedural demonstrations over `x/y/z/yaw`;
   - convert demonstrations to the learning format;
   - use the sim corpus for VLA imitation/co-training or later sim-to-real work.

   This is why normalization matters: it turns the task from "this exact sponge
   at this exact pose" into an object-frame procedure that can scale to many
   objects and stack heights.

## Current Evidence Checked This Session

Existing v2/v3 B200 evidence was checked from remote logs, not rerun:

- v2 close log:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v2_latchstop26_continue_close_b200.out`
  lines 390 and 399-401 show close 23/26 both `reached=NO` with target errors
  `0.014404m` and `0.017537m`, aggregate `LATCH_FAIL`, `posewrite_calls=0`.
- v2 conversion log:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v2_convert_b200.out` line 82 shows
  `cube2cm_counter_jaw_v2_link` merged into `link5`.
- v3 static prep log:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v3_urdf_prep_local.out` lines 1-2
  confirm static-only/no-Isaac scope and `counter_mount=gripper_link`;
  line 29 shows close_26 counter AABB contact; line 34 marks static plausibility.
- v3 conversion log:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v3_convert_b200.out` line 351 shows
  `cube2cm_counter_jaw_v3_link` merged into `gripper_link`.
- v3 close log:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v3_latchstop26_continue_close_b200.out`
  lines 390 and 399-401 show close 23/26 both `reached=NO` with target errors
  `0.014733m` and `0.018066m`, aggregate `LATCH_FAIL`, `posewrite_calls=0`.

Interpretation:

- v2 improved open descent but did not solve grasp.
- v3 made the counter close-dependent and statically plausible, but physics still
  failed close/latch and was slightly worse than v2.
- Neither v2 nor v3 solved grasp.

## What Was Done Locally

Static analysis was extended and a v4 static candidate was added.

Files:

- `sim_scripts/p7_branch_b_cube2cm_close_equilibrium_static_analysis.py`
  md5 `3f6897dec7af2595508c4adddea3e8c9`
- `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v4_urdf.py`
  md5 `bb951300cdc38f87ca0f21e3e04cf0bd`

Local static sweep log:

- `/tmp/p7_branch_b_cube2cm_close_equilibrium_static_analysis_v3_v4_sweep_local.out`

Key static lines:

- Lines 1-2 confirm static-only/no-Isaac/no-training/no-constraint/no-SurfaceGripper/
  no-transport/no-release scope.
- Lines 32-33 show v3's design close imbalance at 26deg:
  moving y-overlap `0.004011m`, counter y-overlap `0.000261m`.
- Line 41 shows the best v4 static candidate:
  `moving_close_overlap_mm=-1.500`,
  `counter_open_clearance_mm=+0.750`,
  design 26deg moving/counter y-overlap `2.011mm / 2.011mm`,
  open descent clearance `YES`.
- Line 41 also exposes the caveat: at 30deg, moving contact is `NO` and counter
  dominates. Therefore v4 is a latch-stop26 candidate only, not a close_30-general
  solution.

Local v4 prep log:

- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_urdf_prep_local.out`

Key static lines:

- Lines 1-2 confirm static-only/no-Isaac/no-training/no-default-edit/
  no-constraint/no-SurfaceGripper/no-transport/no-release scope.
- Line 5 records the v4 parameters:
  `moving_close_overlap_m=-0.001500`,
  `counter_open_clearance_m=0.000750`,
  `design_balance_gate_m=0.000250`,
  `design_min_overlap_y_m=0.001000`.
- Lines 8-23 show the open descent static check with no contact.
- Lines 27-29 show close_26 moving and counter both contact, with y-overlap
  `0.002011m / 0.002011m`.
- Lines 30-32 show the close_30 caveat: moving contact `NO`, counter contact `YES`.
- Lines 33-34 show local v4 generated asset md5s and
  `static_opposing_pair_plausible=YES`.

Local generated v4 asset md5s:

- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_urdf/roarm_m3.urdf`
  `8ed4f0170770d2bc6f4b6380781e21e9`
- moving mesh `177324438c660d0a2f77f1589fb9116e`
- counter mesh `7c531e15150e7955bbdf8959ea5b7d79`

## USD / B200 Status

USD status:

- v3 USD conversion on B200 was previously completed and verified.
- Verified B200 v3 USD md5:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v3_collision_usd/roarm_m3.usd`
  `4497024d25abab11de5c50e144124553`.
- Verified B200 v3 payload md5s:
  - `config.yaml` `7bac0354b78586f22aec8139479bbce6`
  - `configuration/roarm_m3_base.usd` `832aba8d2b4be779f25dd155c8c20a72`
  - `configuration/roarm_m3_physics.usd` `72afa7aa949e13a9fa8066c576766c8e`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- v4 USD conversion has not been run yet. Only local v4 URDF/mesh static prep has
  been generated.

B200 status:

- No new B200 Isaac run was launched in this session.
- No v4 B200 conversion was launched in this session.
- B200 was used only to read existing v2/v3 logs and verify existing v3 USD md5s.
- A B200 process check returned no matching P7/Isaac/training process.

Therefore: v3 USD is good/verified; v4 USD does not exist yet and must not be
claimed as converted or validated.

## Critical Questions / Doubts To Carry Forward

1. Is the current v4 opposing-jaw geometry still too artifact-specific?

   v4 balances AABB overlap at close_26, but the professor's feedback points to a
   broader object-frame grasp primitive. v4 may be a useful diagnostic point, but
   it should not replace a principled fixed-jaw/moving-jaw offset model.

2. Is AABB overlap enough?

   No. AABB overlap is only static plausibility. It does not prove contact normal,
   compliance, friction, stable hold, or lift follow.

3. Should close_30 be included?

   Probably not for the first v4 physics check. v4 is explicitly latch-stop26
   balanced; close_30 loses moving contact and becomes counter-dominant. A first
   physics test, if approved, should be close_26-focused.

4. Is the cube center the right reference?

   The cube center is the normalization reference, not necessarily the TCP target.
   The grasp target must be an object-frame offset derived from fixed-jaw geometry,
   moving-jaw opening, and desired contact height.

5. Does normalization require z/yaw variation now?

   Not before a single canonical primitive is falsifiably valid. But the design
   must be written so `x/y/z/yaw` variation is the next step, not a rewrite.

## Recommended Next Work

Do not jump straight to large data generation.

Next narrow sequence:

1. Static geometry audit:
   - compute fixed-jaw and moving-jaw world frames explicitly;
   - express the grasp target as cube-object-frame offsets;
   - compare that principled offset to the current v4 AABB-balanced candidate.

2. If the object-frame model agrees with v4 or refines it, ask for explicit
   approval for B200 conversion only:
   - sync v4 prep script;
   - run static prep on B200;
   - convert v4 URDF to USD;
   - verify conversion logs and md5s.

3. Only after a second explicit approval, run a close/lift physics diagnostic:
   - no transport;
   - no release;
   - no SurfaceGripper;
   - no constraints/default integration;
   - close_26-focused first;
   - success requires reached, stable hold, lift follow, low drift/speed/tilt, and
     `posewrite_calls=0`.

4. If the canonical cube primitive passes, then design the dataset generator:
   - sample `x/y` translations;
   - sample `z` layer heights;
   - sample yaw/rotation;
   - produce procedural demonstrations for VLA imitation/co-training.

## Continuation Prompt

Use the continuation prompt in the final response of this session. It must include:

- read `CLAUDE.md` and Current-State Protocol;
- forbid training/Isaac unless approved;
- verify dirty state and md5s;
- verify the v2/v3/v4 log lines above;
- emphasize professor feedback: normalized `2cm^3` cube, object-frame fixed-jaw
  offset, z/yaw variation, B200-scale sim demo generation;
- next work is object-frame static audit before any B200 conversion/run.
