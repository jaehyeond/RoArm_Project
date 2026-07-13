# Session 2026-07-13 - D338: collision-representation repair

Final status: `D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP`

Pre-runtime status was `D338_PRE_REGISTERED_RUNTIME_PENDING`; all registration
text below predates the first invocation unless explicitly labeled post-run.

이번 case의 신규 변수: `[collision_representation]` (정확히 1개)

The one changed physical variable is the collision representation of the two
audited tool bodies, `link5` and `gripper_link`.  The canonical USD/URDF/STL
files are never overwritten.  D338 creates one forward-only derivative asset
from the D337-audited **full** link5 and gripper meshes, decomposes each source
with one frozen configuration below, and materializes the returned hulls
as explicit convex collision meshes.  The old single `convexHull` collider is
disabled only inside that derivative.  All robot transforms, visuals, joints,
drives, masses, inertias, limits, collision groups, non-tool colliders, target,
q5, scene, solver, seed, and gates remain frozen.  The licensed claim is
deliberately **task-local**: fidelity at the frozen `(7,11)` G0a command and
all 201 readings of its conditional settle.  D338 does not claim a global
Hausdorff proof that the convex union equals either raw triangle soup at every
possible robot/object pose.

## Why this case exists

D337 recovered a raw-clear open-jaw target at the original D325 `(r,t)=(7,11)
mm`, `q5=1.5413rad`: raw link5 `+4.2726455mm`, raw gripper
`+11.1750884mm`.  Its conditional settle nevertheless received a link5
`38.861N` step-0 impulse because the live physics used D334's certified single
cooked link5 hull (`-6.2367mm` overlap), not the raw surface.  This is the sole
remaining G0a blocker.

The current URDF is not converted as-is: it points the moving-jaw collision at
the later 4mm `gripper_link_collision_g2a.stl`, whereas the audited USD contains
the full 41,094-vertex `gripper_link.stl`.  Blind conversion would change the
physical source and applying converter-level convex decomposition would also
change every robot collider.  D338 therefore derives only the two tool-body
colliders from the current audited USD source.

## Frozen inputs and single build candidate

- Source root: `local_assets/roarm_m3/usd/roarm_m3.usd` and all four composed
  configuration layers, separately hash-pinned (the root hash alone is not a
  collision-asset pin): root
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`,
  base `ea0ee8f258e935799cf927b8c67e871f935c09b3c9be4f971006937334a11841`,
  physics `1df07d387da76dcde4cd700ee1f9546cba25965776a9700897314ef884c37ed2`,
  robot `2227536fcb8c9dae1aa9cc1cf422350fcf85e662eed97fe9ea48535c6b4aa65d`,
  sensor `3f44081f42b452bc5f9791a8df1c37e00ba5a6dc98a9e49e065c7acacdda0d0f`.
- Raw source meshes: full `link5.stl`
  `1d63f374a78c1419b21eec63fa8efeef40d0d42ca89c5de3ceb0d86476d9c7eb`
  and full `gripper_link.stl`
  `7946a374e24a2f467a0581b4946e0ec41b1b86a92f070bc00aa9bced1bf65a56`;
  g2a `bd34df3187305c3a18d572ce5c4a37e3144684cce45ee1d03ee3435b37a6d40a`
  is pinned only as the excluded negative identity.
- D337 summary sha256:
  `80df2f0b3765faee5bbeb190ded03bc326d54602fe16bf5c8fd73513fe5500d4`.
- Seed `33201`; cylinder/sole-support scene and sensors exactly D333/D337.
- Frozen target: `position_only_tangent_minus1`, `(7,11)mm`, HOME-seeded
  position-only IK, `q5=1.5413rad` (open; q5=0 remains CLOSED).
- Controlled steps: zero through the representation gate; then, conditionally,
  200 sole-support baseline + 200 frozen-target settle.
- One registered decomposition configuration, no post-result tuning:
  `hullVertexLimit=64`, `maxConvexHulls=64`,
  `voxelResolution=1_000_000`, `errorPercentage=1.0`,
  `minThickness=0.0001m`, `shrinkWrap=true`.
- Every frozen piece is subsequently authored with
  `MeshCollisionAPI(convexHull)` plus explicit
  `PhysxConvexHullCollisionAPI(hullVertexLimit=64,
  minThickness=0.0001m)`.  Live readback and direct-cook surface parity guard
  against silent defaulting or clamping.
- Output/asset root:
  `claudedocs/runtime_logs/grasp_track/g0a_d338/`.
- First immutable asset attempt:
  `collision_asset/attempt1/`; its manifest and any invocation abort are kept
  inside that attempt directory. A mechanical retry, if ever required, must
  use a new forward-only `attemptN` and may not overwrite prior evidence.

Parameter/tolerance basis: 64 vertices is the PhysX convex-piece limit already
used by the schema; 64 hulls raises only the registered part-count ceiling for
the non-watertight link5 while remaining a one-body finite compound; 1M voxels,
1% error, shrink-wrap, and 0.1mm minimum thickness are the single high-fidelity
candidate, not an adaptive sweep.  The task delta limit `0.5mm` preserves at
least `3.7726mm` of D337's `+4.2726mm` limiting link5 raw margin, and the live
per-piece `0.1mm` surface limit equals the existing clearance-border scale.

Direct triangle mesh (`physics:approximation=none`) is forbidden on these
dynamic articulation links because PhysX can fall back to a single convex hull.
SDF, hand boxes/capsules, mesh repair/rewrite, and a second decomposition
configuration are not part of D338.

## Phase A - deterministic derivative build (no simulation steps)

1. Read the two full source meshes from the composed original USD in each body
   frame and prove D334 topology/bounds identity.
2. Cold-cook each source twice on separate temporary non-instance
   `UsdGeom.Mesh` stages/StageCache IDs, each with an inert `UsdPhysics.Scene`.
   Release both the PhysX local and private runtime mesh caches before **each**
   cook, author/read the exact registered
   `convexDecomposition` parameters, and canonically sort and
   normalize hull vertex order because PhysX does not promise returned part
   order. Preserve outward face winding using Qhull facet normals; never sort
   the three indices within a face. Record before/after cooking statistics;
   each isolated cook must show positive scheduled-task and cache-miss deltas,
   finish every newly scheduled task, and show zero cache hits, polygon-limit
   warnings, or GPU-compatibility warnings. UJITSO and local-cache settings are
   disabled only around each synchronous cook and then restored.
3. Hard reproducibility gate: both cooks must have the same hull count and
   canonical topology/hash; paired coordinates must agree within `1e-9m`.
   Every hull must be finite, positive-volume, convex, contain at most 64
   vertices, and each body must return `1..64` hulls.
4. Copy the original USD bundle forward-only.  Open the copied physics source
   layer directly (never author on a composed instance proxy), disable exactly
   `/colliders/link5/link5/node_STL_BINARY_` and
   `/colliders/gripper_link/gripper_link/node_STL_BINARY_`, then author identity,
   body-local-meter pieces under `/colliders/link5/d338_convex_parts/` and
   `/colliders/gripper_link/d338_convex_parts/`.  They are direct siblings of
   the existing 0.001-scaled mesh-reference subtree, not its children.
5. Record all input, build, hull, layer, and final-asset hashes.  Root/base/
   robot/sensor copied layers must be byte-identical; the physics layer is the
   only changed source file.  A composed semantic allowlist diff must prove the
   only changed old properties are those two `collisionEnabled=false` values
   and the only new specs are the two piece subtrees; every other transform,
   joint, drive, mass/COM/inertia, limit, reference, relationship, collision
   group, and property remains equal. In addition to composed inventory, a
   source-layer-exact comparison after deleting only the two allowed new
   subtrees and two allowed `collisionEnabled` opinions must match.

Any Phase-A failure is a STOP.  It is not permission to tune the decomposition
after seeing the result.

## Phase B - live zero-step representation gate

After loading only the D338 derivative:

1. D333 support/stage/sensor contracts and Isaac pins
   `numpy==1.26.0`, `psutil==5.9.8` must pass.
2. The disabled original source prims must still reproduce D334's full-mesh
   topology and body-local bounds.  The enabled collider set must be exactly
   the authored D338 pieces; the original single hull must not be live.
3. Stage `metersPerUnit` must be `1.0`; each piece-to-rigid-body composed
   transform matrix (not merely its AABB) must be identity within `1e-9`. USD
   and PhysX property-query collider paths
   must match exactly under their
   intended rigid-body owner, with no cross-body attachment and no state change
   during the query.
4. Direct convex-cook retrieval from every live piece (instance or its prototype
   source) must return exactly one valid convex.  Authored-vs-live convex-solid
   bidirectional surface distance must be `<=0.1mm`; volume parity `<=0.5%` is
   corroboration only and AABB is informational only. A separate broad
   property-query-vs-direct-cook volume binding sanity gate is `<=5%`: this is
   not the fidelity metric, but prevents an arbitrarily different attached
   PhysX shape from being licensed. Per-piece hull parameter
   readback, the PhysX asset validator's live per-piece GPU-convex compatibility,
   live mass/COM/inertia/principal-axis parity, and property-query mass parity
   are hard gates. The validator must not change simulation state.
5. Re-run the four **raw/source/kinematic** D337 controls against the retained
   raw full meshes: closed-
   jaw D334 bit parity, D336 exact layer, two D336 grid keys, and open-jaw
   scoping/link5-invariance.  All original tolerances remain unchanged.
6. Re-materialize the frozen open target with zero physics steps.  For each
   body, compute raw exact signed distance and the exact signed distance of the
   **union of all live cooked pieces** (minimum separation when clear; maximum
   contact-level EPA depth when colliding).  Hard gate per body:
   - raw and cooked judgments are internally consistent;
   - raw and cooked signed-distance delta `<=0.5mm`;
   - raw and cooked signed distance both `>=+0.1mm`;
   - all frozen alignment gates pass;
   - simulation counter remains unchanged.

No colliding BVH scalar is ever interpreted as proximity.  A Phase-B failure
forbids baseline/settle physics.

## Pre-runtime implementation audit amendments (before first invocation)

The first Isaac invocation has not occurred. Adversarial source/API review
found and repaired implementation-contract mismatches before any outcome was
visible:

1. Convex face indices now retain outward winding instead of sorting each
   triangle's vertices.
2. Both local and runtime PhysX cook caches are released, and cook statistics
   plus distinct StageCache IDs are hard-audited.
3. Volume parity remains corroboration-only exactly as registered.
4. Direct-cook vertices are explicitly mapped from the live instance prim into
   the owning rigid-body frame, and the actual relative matrix is gated.
5. An uncertified empty/partial part set produces the registered structured
   pre-physics STOP and is never passed to `min()` as a fake union.
6. Attempt manifests/aborts are immutable and attempt-local, so a mechanical
   retry cannot overwrite evidence.
7. Source-layer composition arcs/specs receive an exact sanitized-layer audit.
8. Every live piece receives NVIDIA's zero-step GPU-convex compatibility check.
9. A global Phase-B simulation-counter start/end check covers property-query
   pumps, validator/direct cooks, controls, and final raw/cooked queries.
10. Visualization calls are exception-contained: their failure yields the
    registered artifact verdict while retaining the scientific classification.
11. Raw-source or live-audit exceptions/partial sets now produce structured
    pre-physics STOP payloads; D337 controls and union distance are skipped when
    their two-body prerequisites are absent.
12. A failing target pre-step writes its one-row raw+cooked distance trace
    immediately and embeds the complete evidence in the summary.
13. Every failed Phase-A attempt receives an immutable attempt-local failure
    manifest plus monotonically numbered abort; the same attempt cannot be
    reused after partial/failed evidence exists.

Local schema documentation permits `minThickness` on `[0, inf)` while the
property UI exposes `0.001m` as its lowest slider value. This is recorded as a
clamp risk, not proof of invalidity: D338 keeps the already registered
`0.0001m` candidate and relies on live direct-surface parity plus GPU validation
to detect any silent clamp or incompatibility. No parameter was selected from
runtime results.

The `5%` property-volume binding sanity threshold was fixed before runtime. It
is over three times the already known D334 full-gripper reporting discrepancy
(`1.46%`) while still rejecting gross API/path attachment mismatch; the
registered `0.5%` value remains explicitly corroborative and cannot decide
task fidelity. Cold-cook UJITSO/local caching is disabled during each request,
and positive scheduled-task/cache-miss deltas are required, so `0==0` cannot
vacuously certify an independent cook.

## Phase C - conditional sole-support settle

Only a complete Phase-B PASS licenses:

1. Exact HOME write with `q5=1.5413`, unchanged D333 200-step sole-support
   baseline and complete baseline hard gate.
2. Exact frozen `(7,11)` target write followed by one 200-step command hold.
   Record contact, object pose/velocity, TCP, joints, support, root, and both
   raw and live-cooked distances at the pre-step reading and every post-step.
3. Static support requires all of:
   - raw and live-cooked tool clear `>=+0.1mm` at all 201 readings;
   - per-body `abs(raw exact - live cooked exact) <=0.5mm` at all 201
     readings, with non-saturated contact-level EPA when collision is queried;
   - link4/link5/gripper filtered-force maxima `<0.1N`;
   - object max XY `<0.5mm`, tilt `<1deg`, disturbance onset `-1`;
   - support and root contracts PASS;
   - final G0a alignment PASS and final displacement `<5mm`.

The D337 sustained-onset metric is not changed proactively.  The maximum-force
gate already detects a one-row impulse.  An impulse-row onset field may be
added reactively only if D338 again observes a force event that the sustained
onset field misses; it may not relax the static PASS conditions.

## Pre-registered outcomes

1. `D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP` - deterministic build,
   compatibility, or forward-only mutation contract failed.
2. `D338_G0A_PREPHYSICS_CONTRACT_FAIL_STOP` - pins, source, live owner/path,
   direct-cook, D337-control, or zero-step contract failed.
3. `D338_G0A_REPAIRED_COOKED_TARGET_NOT_CLEAR_STOP` - representation is
   auditable but the frozen command fails the `+0.1mm` clear or `0.5mm` fidelity
   gate; no physics.
4. `D338_G0A_COLLISION_REPRESENTATION_STATIC_SUPPORTED_STOP` - the task-local
   frozen-pose/settle representation, baseline, and static-clean gates all
   PASS.  This licenses the next frozen 10-trial G0a case; it does not itself
   set `g0a_pass=true` or certify global mesh equivalence.
5. `D338_G0A_STATIC_RUNTIME_MIXED_STOP` - baseline or target dynamics fail any
   registered static gate.
6. `D338_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` - scientific evidence
   is retained but the visualization/artifact contract failed.

Every branch stops D338 before a 10-trial run.  If and only if outcome 4 is
obtained, the already user-selected critical path advances to a separately
pre-registered frozen-target 10-trial case.

## Post-run result - attempt1 (2026-07-13 11:44 KST)

Command:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output \
  python sim_scripts/cyl34_top_view_d338_grasp_g0a_collision_representation_repair.py
```

Observed branch: `D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP`.

The call reached the first `link5_cold1` synchronous
`request_convex_collision_representation` and returned control, with all three
isolation settings read back as applied (update-to-USD off, UJITSO cook off,
local mesh-cache setting off) and both local/runtime caches explicitly
released. The registered positive statistics witness nevertheless read:

| Counter delta around request | Value |
|---|---:|
| scheduled tasks | `0` |
| finished tasks | `0` |
| cache hits | `0` |
| cache misses | `0` |
| polygon-limit warnings | `0` |
| GPU-compatibility warnings | `0` |

Thus `positive_scheduled_task_delta=false` and
`positive_cache_miss_delta=false`; the remaining four statistics checks passed.
The script deliberately checked this contract before consuming/recording the
callback result, so **neither `RESULT_VALID` nor a convex count is licensed or
claimed** from attempt1. The zero deltas show that these global counters do not
positively instrument this explicit synchronous request path; they cannot
distinguish a real cook from a non-counted direct request here.

Consequences:

- no derivative USD directory or hull manifest was created;
- the IsaacLab task environment was never constructed;
- controlled physics steps `0` (baseline `0`, target `0`);
- no live owner/direct-cook/GPU-validator/target-fidelity gate ran;
- no PNG/RRD was attempted because the registered Phase-A prerequisite failed;
- canonical USD/physics-layer/URDF hashes were rechecked unchanged after stop:
  `a4be58...e46fff`, `1df07d...7ed2`, `64dc8d...9dae2`;
- `g0a_pass=false`; no 10-trial, G0b, RL, or ladder promotion.

Immutable evidence:

- failure manifest: `collision_asset/attempt1/d338_asset_build_manifest.json`,
  sha256 `8fec513a4e344132f4e445061bbc383da2d6347f5e5883b4c53b2695da1acdda`;
- invocation abort: `collision_asset/attempt1/d338_invocation_abort_001.json`,
  sha256 `f168087ac672e2bffdd9fdf29b6200afcd69e6551e4fd9e891b1e8f7695c8d42`;
- preserved Kit log: `collision_asset/attempt1/kit_20260713_114440.log`,
  `381,911` bytes, sha256
  `075ce099543ae952e362c47e44894e92abb63e72610c274441e6a82690a87a6b`;
- post-run consolidation summary:
  `g0a_d338_collision_representation_repair_summary.json`, sha256
  `0bda3990751253a7c50408b0106cdc9e3504a35e6ac4f72a9504de0b90aa9a1e`.

This is a registered build-contract STOP, not evidence that the decomposition
geometry is good or bad. Attempt1 remains immutable and no attempt2 was run.
The recommended next choice is a **separately pre-registered** cook-witness
contract repair that leaves every physical/decomposition parameter frozen,
records callback result/count before the witness decision, makes unsupported
global statistics informational, and proves independence using cache-disabled
distinct stages plus two-cook canonical geometry equality. That is a case
change and was not authorized or executed in this session.

## Visualization and artifacts

- Pre-physics decision PNG: cylinder + retained raw surfaces + every live
  cooked piece + nearest witnesses + target/commanded/actual frames.
- Conditional final PNG with the same representation overlay.
- `draw_frames` target-vs-actual markers and exactly one non-empty RRD; the RRD
  contains all 200 target-settle rows when physics runs.
- JSON/CSV artifacts include frozen contract, build manifest, hull manifest,
  live collider inventory, D337 controls, representation gate, baseline/target
  traces, raw+cooked distance trace, summary, and explicit snapshot paths.

## Session progress rule

The live representation gate is a failable perturbation evaluation: the one
new collision representation may fail build reproducibility, live parity,
task-pose fidelity, or clearance.  The conditional settle is a second failable
evaluation.  If Phase A itself cannot safely materialize the intervention,
physics/Phase B cannot run; that branch explicitly satisfies the session rule
by documenting that the registered failable perturbation was blocked before a
valid intervention existed.  No training is authorized.

## Non-goals

No target/q5/IK/waypoint/solver/seed change; no 10-trial in D338; no close,
grasp, lift, G0b, RL/PPO, randomization, VLA, real RoArm, cube, B200, large
render/video, cleanup, canonical asset rewrite, JOINT_LIMITS removal, or
commit/push.
