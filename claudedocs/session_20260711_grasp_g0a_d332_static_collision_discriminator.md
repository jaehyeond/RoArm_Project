# Session 2026-07-11 - D332 G0a canonical static collision discriminator

Pre-run status: `D332_IN_PROGRESS` (the contract below was frozen before the
official offline/runtime artifacts were generated).

Final status:
`D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED`.

이번 case의 신규 변수: `[]` - D330 cylinder, target formula, open gripper,
mass/friction, and G0a gates stay fixed. The thresholds below are diagnostic
event definitions, not G0a gate changes.

## Research question

At one deterministic HOME-seeded canonical G0a pose, does the actual
PhysX-cooked link5 `convexHull` collision overlap the D34 x H90 cylinder, and does a direct
teleport to that same pose produce a link-attributed contact and object
disturbance during controlled physics settling?

This is a final-pose static discriminator. It does not test the D330 swept
approach path and cannot promote G0a or reopen G0b by itself.

## Frozen inputs

- Object: upright cylinder, radius `0.017m`, height `0.090m`, center
  `(0.300, 0.000, TABLE_Z + 0.045)` with `TABLE_Z=-0.012117m`.
- Object mass/friction stay at the D330 placeholders: `0.72kg`, static `1.5`,
  dynamic `1.2`; no material, mass, or gate tuning.
- Target formula stays D330:
  `TCP = center - radial*0.007 - tangent*0.011`, TCP z at cylinder center.
- Canonical joint vector is deterministic position-only IK from exact
  `HOME=[0,0,90,0,0,0]deg`, `max_iter=120`, `pos_tol=1mm`; gripper stays open
  at `q=0`.
- Runtime uses one env. Reset jitter is removed by explicit post-reset writes
  of exact HOME/target joint state and exact object pose/zero velocity.

## Geometry representation contract

- Source mesh: `local_assets/roarm_m3/urdf/meshes/link5.stl`, URDF scale
  `0.001`, link5 collision origin identity.
- Stage 1 (`offline`) computes both the raw, non-convex STL negative control and
  the unrestricted mathematical Qhull of the source vertices. This is a
  precheck only: the mathematical hull has 200 vertices while PhysX convex
  cooking defaults to a 64-vertex limit, so it is not called the runtime hull.
- Stage 2 (`runtime`) synchronously requests a default PhysX-cooked convex
  representation of an exact source-mesh mirror. The live asset places
  `MeshCollisionAPI` on an
  instance-proxy Xform while the `UsdGeomMesh` is its child, which the public
  cooking request cannot parse directly. Therefore the probe extracts the live
  mesh through the USD hierarchy into link5-local meters, creates an exact
  temporary non-stepped Mesh mirror with the same `convexHull` approximation,
  synchronously cooks it with default PhysX properties, and removes it before
  any physics step. The returned mirror-cooked vertices are queried against the same
  analytic cylinder with `hppfcl` GJK/EPA. Source topology, transforms, mirror
  removal, and cooked counts are recorded. Direct extraction of the live
  articulation collider cook is unsupported. The mirror matches source
  topology, transform, and `convexHull` approximation, but it does not prove
  rigid-body ownership or parity of every live cooking attribute.
- AABB is not a decision input.
- Cooked-hull signed distance verdict: `<= -0.1mm` = `OVERLAP`,
  `>= +0.1mm` = `CLEAR`,
  otherwise `BORDERLINE`. GJK distance and collision/EPA depth must agree in
  sign; all transforms, closest/contact points, mesh hashes, versions, and
  solver residual are recorded. Full-hull and cooked-hull results remain
  separately named in every artifact.

## Runtime witness and settle contract

- Exactly one witness method: a scene-owned, pre-PLAY ContactSensor whose body
  is the cylinder and whose ordered filters are the actual support plane,
  link4, link5, and
  gripper_link.
- The cylinder ContactReporter is authored at spawn with threshold `0.0N`.
  This also sets the rigid-body sleep threshold to zero; that instrumentation
  side effect is recorded.
- Hard sensor contract: `num_instances=1`, `num_bodies=1`, four resolved
  filters with verified path-to-index mapping, expected force/contact tensor
  shapes, and an authored reporter threshold of zero.
- Robot-free baseline: exact HOME + exact object pose, `200` physics steps
  (`1.0s` at `dt=1/200`). Positive control requires the last-50-step median
  support normal force `>1N`; robot-filter baseline max must be `<0.1N`.
- Target settle: reset object pose/velocity, teleport the robot to the frozen
  canonical vector, then run `200` physics steps (`1.0s`) with no approach,
  close, lift, action, or target change.
- Robot witness onset: the first of two consecutive physics steps with any
  robot-filter normal force `>=0.1N`.
- Disturbance onset: the first of two consecutive physics steps with object XY
  displacement `>=0.5mm` or tilt `>=1deg`. These are diagnostic onset
  thresholds; the unchanged G0a displacement gate remains `<5mm`.
- Timing agrees only when link5 witness onset is no later than one physics step
  after the disturbance onset; otherwise the outcome is mixed.
- Every target-settle physics step records full object xyz/quaternion,
  linear/angular velocity, tilt, displacement, actual/commanded joints,
  target/actual TCP, and filter force/vector/contact-point data.

### D330 witness failure classification

The D330 robot-link failure is not classified as a missing reporter or a prim
path typo. D330 sets `activate_contact_sensors=True`
(`sim_scripts/cyl34_top_view_d330_grasp_g0a_alignment_probe.py:327-331`), and
the exception records resolved link4/link5/gripper_link body paths. Failure
occurs later when the PhysX contact view body count does not match the resolved
reporter bodies (`ContactSensor._initialize_impl`, installed Isaac Lab
`contact_sensor.py:255-297`). D330 also creates these sensors after PLAY and
manually calls the private initializer (`d330 probe:482-510`), while its
env0-only cylinder experiment violates the scene sensor env-index domain.

The most defensible classification is therefore an **articulation-body/view-
domain lifecycle contract failure**. D330 does not isolate post-PLAY lifecycle
from articulation-link tensor-view support, so neither sub-cause is claimed as
proven alone. A separate helper defect is also present: Isaac Lab forwards the
boolean `activate_contact_sensors` value as the numeric reporter threshold;
`True` becomes `1.0N`. That threshold bug cannot explain view initialization
failure, but it makes the D330 configuration unsuitable for a `>0N` witness.
D332 applies one replacement strategy rather than mixing sensors: one
cylinder-owned, scene-registered pre-PLAY sensor with an explicit `0.0N`
reporter threshold and robot-link filters (`d332 probe:504-559`).

### Reactive witness repair after invalid attempt 0

The first runtime implementation reached physics but cannot be a D332 verdict.
The PhysX view returned filter paths as a nested `[[four paths]]` array; the
probe incorrectly stringified the outer list and assigned all labels to filter
index 0. It also pre-registered TapTable as the support filter. Runtime pose
showed the cylinder center settling at about `z=0.045m`: the global terrain
plane at `z=0` sits `12.117mm` above the TapTable top (`TABLE_Z=-0.012117m`), so
terrain, not TapTable, is the active support. TapTable `0N` was therefore a
filter-domain error, not a failed ContactReporter. The one permitted witness
repair keeps the same cylinder-owned sensor, flattens and one-to-one validates
the four resolved indices, and replaces only the support filter with
`/World/ground`. No physics or target value changes. Invalid attempt-0
JSON/CSV are preserved under
`g0a_d332/attempt0_invalid_table_filter/`; its PNG/RRD are not duplicated so
the Visualization DoD remains at three snapshots and one final RRD.

The repaired run resolved the robot filter indices correctly, but the root
filter `/World/ground` returned no usable support channel. The raw warning was
not retained, and the actual collision prim
`/World/ground/terrain/GroundPlane/CollisionPlane` was not tested. Therefore
the evidence does **not** establish a general GPU-filter limitation. No further
physics run or sensor method was added. The already-recorded unfiltered net
force matching cylinder weight is a useful posthoc reporter diagnostic, but it
cannot retroactively pass the frozen filtered-support `>1N` gate. The
pre-reanalysis summary is preserved under
`g0a_d332/attempt1_ground_filter_gpu_unsupported/`; its own
`decision_evidence=false` remains correct.

## Pre-registered decision matrix

| Actual cooked hull | Runtime contact/disturbance | D332 interpretation |
|---|---|---|
| overlap | link5-attributed contact + disturbance | collision-geometry blocker supported |
| overlap | no corresponding runtime event | offline/runtime collider or transform mismatch; no repair |
| clear | contact or disturbance | other link, transform, or non-final-pose cause; no drive conclusion |
| clear | no event | final-pose overlap refuted only; swept-path audit is next |

`BORDERLINE`, invalid contact positive control, or contact/disturbance timing
that disagrees are mixed outcomes and cannot promote a geometry repair.

## Visualization DoD

Maximum three PNGs and one RRD:

1. `d332_offline_hull_overlap.png`
2. `d332_teleport_first_event.png` (or first post-teleport step if no event)
3. `d332_teleport_final.png`
4. `d332_contact_disturbance_trace_v2.rrd`

Runtime frames include target TCP, actual TCP, separate link5 and gripper_link
frames, live cylinder frame, and a sensor or offline witness point.

## Stop rule

The D331 wrist null-space scan is conditional, not automatic. Skip it when the
canonical signed distance is decisively outside the `+/-0.1mm` borderline and
the runtime result is classifiable. It may run only after a borderline or
offline/runtime-mismatch result because otherwise it cannot change this
session's decision.

## Non-goals

No collision mesh re-authoring, target/gate/offset/standoff changes, waypoint
search, G0b, gripper close, grasp/lift, RL/PPO, randomization, render beyond the
three diagnostic PNGs, material/friction/mass changes, VLA, RoArm, B200/SSH,
cube reintroduction, or controller hand-condition additions.

## Executed stages

1. `offline`: solved the frozen HOME-seeded position-only IK once, wrote the
   exact commanded joint vector, and compared the raw STL and unrestricted
   mathematical hull against the analytic D34 x H90 cylinder with hpp-fcl.
2. `runtime`: created one deterministic env, synchronously default-recooked an
   exact mirror of the live-stage link5 source mesh, removed that mirror before
   physics, ran a 200-step robot-free baseline, reset exact state, and ran a
   200-step canonical target settle.
3. `reanalyze`: recomputed the baseline positive control and first-contact
   direction checks from the two committed CSV traces. This stage did not run
   physics again. The trace hashes are stored in
   `d332_contact_witness_reanalysis.json` and the final summary.

Attempt 0 is excluded because its nested PhysX filter-path array was mapped
incorrectly and its support path named the inactive TapTable. Attempt 1's
robot filters are valid, but root filter `/World/ground` returned no usable
channel in this run; the exact collision-plane path was not tested. Both
invalid/intermediate summaries are retained under their named attempt
directories and are not decision evidence.

## Final result

Verdict:
`D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED`.

| Metric | Result |
|---|---:|
| Canonical commanded TCP error | `0.817812mm` |
| Raw non-convex STL control | `+4.273819mm` (`CLEAR`) |
| Unrestricted mathematical hull | `-6.363467mm` (`OVERLAP`) |
| Default PhysX mirror recook | `35` vertices, `48` polygons |
| Pre-step mirror-recook signed distance | `-6.236272mm` (`OVERLAP`) |
| Direct live-collider cook extracted | `False` |
| GJK/EPA sign/depth cross-check | PASS (`<0.1mm` disagreement) |
| Robot-free baseline | `200` steps / `1.0s` |
| Frozen filtered-support positive control | FAIL (`0N`, exact collider path untested) |
| Posthoc net reporter diagnostic | `7.0632007N` (expected `7.0632N`) PASS |
| Baseline robot-filter maxima | link4/link5/gripper_link = `0/0/0N` |
| Baseline max object XY / tilt | `0.459483mm` / `0.672643deg` |
| Initial cylinder-ground penetration | `12.117000mm` |
| Baseline first post-step z correction | `+12.256849mm` |
| Canonical target settle | `200` steps / `1.0s` |
| First observed robot witness / disturbance | post-step `0` / post-step `0` |
| Target first post-step z correction / net force | `+12.707490mm` / `125.033206N` |
| Runtime attributed body | `gripper_link` |
| gripper_link peak normal force | `66.866266N` at step `0` |
| link5/link4 sampled peak force | `0/0N` |
| Force vs XY displacement / velocity cosine | `0.972725` / `0.969457` |
| Peak object linear / angular speed | `0.315708m/s` / `8.004216rad/s`, step `0` |
| Final/max object XY displacement | `10.282285mm` / `10.452925mm` |
| Final/max object tilt | `9.235161deg` / `9.439981deg` |
| Final actual TCP / joint tracking error | `3.413499mm` / `0.009325rad` |

The cylinder-owned sensor hard contract passed: one instance, one body, four
resolved filters, expected tensor shapes, and reporter threshold `0.0N`.
Initialization alone was not accepted. The unfiltered HOME net force matching
`m*g` validates net reporting posthoc, and the positive gripper event validates
that filter channel. It does not independently positive-control link4/link5;
their `0N` means only no sampled filtered force in this run. The frozen
filtered-support gate failed. The root `/World/ground` filter was the wrong
granularity and the exact `CollisionPlane` path was never tested.

The wrist null-space scan was not run. The mirror distance is far outside the
borderline, but the runtime discriminator is invalidated by the support-domain
confound. A family scan cannot repair that contract and would add an
unnecessary search variable before a clean static retest.

## Critical interpretation

1. D331's gap-fill hypothesis is strongly supported at the frozen pre-step
   command: the raw STL is clear while the 35-vertex default PhysX mirror
   recook overlaps by `6.236mm`. This is not AABB evidence, but neither is it a
   direct extraction of the live articulation collider cook.
2. The first recorded row is after one physics step. At that point the cylinder
   has moved upward `12.707mm` while the net force is `125.033N`; ground
   depenetration and the `66.866N` gripper_link lateral event are coupled. The
   data support **a gripper filtered lateral event was observed**, not a clean
   contact onset or robot-only disturbance cause.
3. The pre-step mirror query and the post-step contact witness are different
   poses. Link5 has no sampled filtered force after the step, and link4/link5
   negative channels lack independent positive controls. Body-specific repair
   is not justified.
4. D332 teleports directly to the final target and then settles, so every
   runtime sample is a final-pose/hold condition. The first-step event cannot
   distinguish early approach from final-hold contact and does not refute
   target-depth/final-pose interaction or drive coupling.
5. The frozen support positive control failed and the reporter threshold also
   changes rigid-body sleep behavior relative to D330. Posthoc reanalysis is
   useful diagnostics, not promotion evidence.
6. This is not a G0a PASS. No 10-trial gate was run, G0b stays blocked, and no
   target, gate, material, mass, drive, waypoint, or collision mesh changed.

## D330 hypotheses versus D332

| D330 preregistered possibility | D332 observation | Judgment |
|---|---|---|
| Early approach/link contact | No approach was executed; first record is post-step | Not tested |
| Final hold/depth/settling | Entire runtime is final-target settle; coupled event appears in first record | Compatible, not isolated or refuted |
| Drive alone | Low commanded error and later joint error coexist with coupled ground/robot impulses | Cannot isolate from this run |
| Contact witness remains invalid | Hard tensor contract, net reporter, and gripper positive channel work; frozen support gate fails | Partially repaired, not decision-valid |
| link5 coarse hull is the sole disturbing body | Pre-step mirror overlaps; post-step sampled event is gripper_link | Strong geometry hint, sole-body claim unsupported |

## Visualization and artifacts

- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_offline_hull_overlap.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_teleport_first_event.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_teleport_final.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_contact_disturbance_trace_v2.rrd`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/g0a_d332_static_collision_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_contact_baseline_trace.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d332/d332_teleport_settle_trace.csv`

All three PNGs were inspected at original resolution. Each runtime snapshot
contains the target TCP, actual TCP, link5, gripper_link, cylinder, and contact
witness frames. The single RRD contains 200 dynamic target-settle steps plus
actual/commanded URDF entities and a valid blueprint. Runtime PNGs were
regenerated from CSV object pose and actual joints using repo FK; the original
live frame stream remains in the RRD.

## One next action

D333 should apply one scene-domain contract repair and repeat only the same
one-env static discriminator: disable the redundant global-ground collision so
the existing TapTable top at `TABLE_Z` is the sole support, restore a valid
TapTable filtered-force positive control, and keep the object pose, relative
target formula, joint reset, mass/friction, thresholds, and 200+200 steps
unchanged. Stop after deciding whether a gripper/link event and object motion
remain without the `12.117mm` depenetration. Collision ownership/cook-attribute
audit is deferred until that clean runtime result.
