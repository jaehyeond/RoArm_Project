# START_HERE.md

Last updated: 2026-07-27 KST (5th update). D398 remains frozen FAIL_STOP.
D400 preregistration V2 is reviewed but not executed. Documentation only: no code,
derivative asset, USD/Isaac/PhysX, physics, commit, or push.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a. `q5=0` CLOSED; frozen OPEN `1.5413rad`.
- D362 (2026-07-17) is the last physics run: A64 collider pushed over the
  34x90mm/0.72kg cylinder (final XY 60.619mm / tilt 89.998deg). Zero physics since.
- Actual product nominal `29x50mm`; mass/tolerance/COM/inertia/friction
  unmeasured; NO code constant implements 29x50 (docs only).
- A64 (`64+64=128`) = reference candidate, not optimum/NVIDIA limit.
  P34 (`16+18`) = manual concept, live/cooked identity FAIL.
- D384-D398 low-count convex-partition existence under project gates ended in a
  localized greedy dead end (D398); global impossibility unproven.
- D385/D397 `12 vertices/child` is a project-authored budget for eight manually
  split source parents, not a PhysX/NVIDIA default, optimum, or hard limit. It
  does not apply to SDF.
- The historical A64 reference is materialized, but no complete/materializable
  **replacement** collider from P34/D397/SDF exists. `g0a_pass=false`.

## Active Case — D400 preregistered; implementation/static attestation PENDING

User chose nominal `29x50mm`, D400 for SDF, and D399 reservation for the D398
label repair. D400 preregistration was written under:

`claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/d400_preregistration.json`

Current result is `PREREGISTERED_NOT_EXECUTED`; three independent reviews
removed the cook `0->0` false-pass, weak property-query, ambiguous A64-disable,
cleanup-order, technical-fail/Rerun branch, non-exact Rerun entity, and
unreviewed-script hash errors. The next approval is no-Isaac implementation and
static hash attestation only; actual Isaac/PhysX worker requires a further
separate approval after those hashes are reviewed. Latest detail:
`claudedocs/session_20260727_grasp_g0a_d400_gripper_sdf_live_cook_articulation_preregistration.md`

## SDF and cylinder facts

- Installed omni.physx 107.3.26 / Isaac Sim 5.1.0 / Isaac Lab 2.3.0 /
  PhysX 5.6.1 `[INFERENCE]` provides `PhysxSDFMeshCollisionAPI`; schema default
  resolution is 256. SDF is not GPU-only or yet validated on this articulation.
- Isaac Lab `CylinderCfg` authors a real `UsdGeom.Cylinder`; the child path name
  `/geometry/mesh` does not make it a mesh prim.
- Installed 107.3 uses `/physics/collisionApproximateCylinders`; default-off
  documentation strongly supports analytic collision. D362 did not log the
  runtime value/type, so authority is `HIGH_CONFIDENCE_INFERENCE`, not exact
  confirmation; working contact reports cannot distinguish analytic/convex.
- A future cylinder case must log carb current/default values and PhysX geometry
  type; legacy custom-geometry vs SDF may fall back to TriangleMesh collision.

## Corrected forward-only ladder

### D399 — reserved presentation repair

`d398_rerun_label_deconfliction_observability_repair` remains optional. It is
not a prerequisite for collider or physics work.

### D400 — preregistered SDF configuration/load/owner preflight only

One proposed variable: `gripper_link A64 -> SDF`, resolution `256`.

- keep `link5=A64`; direct pxr API + exact `approximation="sdf"` readback
- record active owner/path, exact property-query rows, stage-wide cook task
  deltas, parser/fallback warnings, source/live USD inputs, and robot property
  invariance
- one worker/no retry/bounded watchdog; no product cylinder, target/IK/path
  change, q5/contact/controlled physics
- no `link5` SDF, resolution 512, remeshing, mesh repair, or sweep

Zero-contact D400 can prove exact SDF configuration, stage load admission,
nonzero global cook-queue drain, and rigid-link owner enumeration. It cannot
prove a per-prim internal SDF object or articulation collision participation.
Cooked-SDF surface/void and cross-body OPEN clearance remain `null`; the latter
belongs to D402 because D400 reads no q5/common world pose.

### D401 — proposed SDF articulation collision-response positive control

Use one bounded non-product box probe to compare A64 versus frozen D400 SDF
contact/response on the articulation. Keep q5 and target/IK/path frozen, record
the exact owner pair, and do not use a cylinder. This is a representation
plumbing check, not product-grasp evidence.

### D402 — proposed nominal-product zero-step pose localization

Two variables: nominal `29x50mm` geometry and deterministic height/radial pose;
wrist frozen. Use one common pose for A64/SDF, log cylinder settings and actual
geometry type, and run no closure physics/contact.
Historical D362 contact samples were z `78.010/49.408mm`; the new nominal top is
about `37.883mm`, strongly motivating but not proving a miss. Full-surface
zero-step clearance is the decision authority.

### Mass pin, then D403/D404 physics

- Measure product mass before dynamics; record any uniform COM/inertia
  assumption and keep placeholder friction explicit.
- D403: new `29x50` A64 physics baseline on the frozen D402 pose. Re-derive
  contact capacity immediately before this first contact-generating case.
- D404: inherit D403 bit-for-bit and change only
  `gripper_link A64 -> frozen D400 SDF`.

## Next authorization boundary

Separately approve D400 no-Isaac implementation/static attestation: create only
the two registered controller/worker scripts and
`d400_reviewed_script_attestation.json` plus
`d400_proposed_runtime_hash_tuple.json`, statically verify them, and do not
materialize a derivative USD or import/launch Isaac/Kit/PhysX. After that report,
the actual one-worker approval must explicitly cite the proposed tuple-file SHA.
D401 remains a later separate approval even if D400 passes.

## D400 runtime boundary; physics remains separately prohibited

- Before separate D400 implementation approval: no D400 script or attestation
  creation is authorized.
- The implementation/static-attestation approval, if granted, still authorizes
  no derivative USD/collider write or Isaac/Kit/PhysX/Warp launch.
- Actual worker needs a further approval after reviewed script hashes are fixed.
- Even if D400 runtime is approved: physics step, q5 sample, contact query, and
  cylinder creation remain prohibited; those require their later named cases.
- Do not treat default mass, COM, inertia, or friction as product evidence.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260727_grasp_g0a_d400_gripper_sdf_live_cook_articulation_preregistration.md`
4. `claudedocs/session_20260727_grasp_g0a_vertex12_cylinder_d400_scope_review.md`
5. `claudedocs/session_20260727_grasp_g0a_sdf_case_recon_option_a_briefing.md`
6. `claudedocs/session_20260727_grasp_g0a_d398_resume_verification_and_sdf_reevaluation.md`
7. `claudedocs/DECISIONS.md` (read fully; focus D385, D397-D398, D398-F1,
   D400-P0 and its authority corrections D400-P1/D400-P2)
8. `claudedocs/EXPERIMENT_LEDGER.md` (read fully; verify the latest rows)

## Authorization and Do-Not-Repeat

- D389-D397 paths and D398 attempt1 are frozen; never rerun or overwrite them.
- Do not materialize or physically test D397/D398 diagnostic geometry.
- Do not change max-12, plane family, and greedy search together.
- Do not carry max-12 into SDF or bundle product dimensions, pose, mass, and
  collider representation into one variable.
- Any future timeline PLAY case must inherit D367's explicit commit bridge and
  supervisor/pre-close authority.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale; `/half-clone` forbidden (HARD RULE #11).
- No hardware, process signal, dependency install, commit, or push authorized.

## Git

- `HEAD == origin/master == 4c88865bdd4ac82f034253320cb3e46f9770a46d`
  (`현재 D398분석`).
- Prior SDF-recon changes were already uncommitted; this update adds the latest
  D400 preregistration/session/state continuity changes.
- Commit/push is not authorized.
