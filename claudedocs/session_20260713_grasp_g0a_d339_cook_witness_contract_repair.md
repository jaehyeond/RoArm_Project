# Session 2026-07-13 - D339: cook-witness contract repair

Final status: `D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`

Historical pre-runtime registration status: `D339_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수: `[cook_witness_contract]` (정확히 1개,
measurement-contract variable)

D339 does not change a physical or decomposition setting. It carries forward
the one D338 full-mesh link5/gripper collision-representation candidate exactly
and changes only the positive witness used to decide whether two direct cooks
are valid and independent. The first D338 attempt remains immutable; D339 writes
only to its forward-only `collision_asset/attempt2/` path.

## 1. What and why

D338 reached the synchronous `link5_cold1`
`request_convex_collision_representation` call with UJITSO/local caching
disabled and both local/runtime caches released. The request returned control,
but all global `get_cooking_statistics()` deltas were zero. Its registered
positive scheduled-task/cache-miss gate therefore stopped before the callback
result or convex count was consumed. D338 correctly licenses no geometry claim.

D338 established a durable measurement lesson: those global counters do not
positively instrument this explicit synchronous request path. D339 repairs that
measurement contract. It records the callback itself first, uses its exact
result and returned convex count as the direct validity witness, treats global
statistics as informational, and proves repeatability with two cache-disabled
cooks on distinct stages whose canonical geometries must match.

This is reactive contract hardening caused directly by D338's registered
failable build failure. It does not widen the collision-representation search.

## 2. Frozen inputs and intervention

- Canonical root USD sha256:
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`.
- Canonical physics layer sha256:
  `1df07d387da76dcde4cd700ee1f9546cba25965776a9700897314ef884c37ed2`.
- Canonical URDF sha256:
  `64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2`.
- D338 summary sha256:
  `0bda3990751253a7c50408b0106cdc9e3504a35e6ac4f72a9504de0b90aa9a1e`.
- Immutable D338 attempt1 manifest sha256:
  `8fec513a4e344132f4e445061bbc383da2d6347f5e5883b4c53b2695da1acdda`.
- Immutable D338 attempt1 abort sha256:
  `f168087ac672e2bffdd9fdf29b6200afcd69e6551e4fd9e891b1e8f7695c8d42`.
- Immutable D338 attempt1 Kit log sha256:
  `075ce099543ae952e362c47e44894e92abb63e72610c274441e6a82690a87a6b`.
  The exact attempt1 inventory is those three files and no others.
- Reused D338 helper implementation sha256:
  `f3d330a9a5ca6f886728d0e5dc8037baa68d83a2b911aa105904d7d369ead426`.
  D339 hard-pins this source before reusing its authoring/live helper surface.
- Source meshes remain the audited full `link5.stl` and full
  `gripper_link.stl`; the g2a 4mm proxy remains excluded.
- Decomposition remains exactly:
  `hullVertexLimit=64`, `maxConvexHulls=64`,
  `voxelResolution=1_000_000`, `errorPercentage=1.0`,
  `minThickness=0.0001m`, `shrinkWrap=true`.
- Target/physics remain exactly D338: `(r,t)=(7,11)mm`, `q5=1.5413rad`,
  seed `33201`, cylinder/sole-support scene, solver, sensors, masses, friction,
  alignment gates, `0.1mm` clear gate, `0.5mm` raw-vs-cooked task-fidelity
  tolerance, and conditional `200+200` steps.
- Isaac pins remain `numpy==1.26.0`, `psutil==5.9.8`.
- D339 output root:
  `claudedocs/runtime_logs/grasp_track/g0a_d339/`.
- Only allowed asset attempt:
  `claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/`.
- The derivative's internal authored subtrees remain exactly D338's registered
  `/colliders/{link5,gripper_link}/d338_convex_parts`; a D339 path rename is
  forbidden because it is not part of the witness-contract variable.

No parameter sweep, fallback configuration, post-result tuning, or attempt1
reuse is allowed.

## 3. Phase A - callback-first independent-cook gate

For each body (`link5`, then `gripper_link`), D339 performs two synchronous
cooks, `cold1` and `cold2`, from the same pinned source arrays.

1. Before each request, construct a fresh in-memory USD stage with a unique
   identifier, an inert physics scene, and the exact registered decomposition
   parameter opinions/readback.
2. Insert that live stage in `UsdUtils.StageCache`, record its valid numeric ID
   and identifier, disable UJITSO/local mesh caching, and invoke both local and
   private runtime cache-release APIs.
3. The synchronous callback deep-copies an event immediately: callback ordinal,
   enum name/value/repr, returned convex count, and each convex's complete
   vertex/index/polygon/plane payload. The cook-specific witness JSON is
   persisted in attempt2 **before any result, count, statistics, or geometry
   gate**. The synchronous request's return value is informational and is not a
   validity witness.
   The base event (enum/count) is appended before per-convex serialization;
   serialization errors are recorded per part and are a hard failure without
   erasing the already observed enum/count.
4. Direct callback hard gates:
   - callback invoked exactly once;
   - result enum exactly `RESULT_VALID` (`value=0`);
   - returned convex count in `1..64` and equal to the retained object count;
   - every returned convex has `4..64` finite vertices and non-empty
     index/polygon data; polygon spans and indices are in range, polygon planes
     are finite with nonzero normals, polygon spans cover the index buffer once
     without gaps/overlap, and the canonical solid has positive finite volume.
5. Isolation hard gates:
   - all three cache/update settings read back disabled during the request;
   - local and runtime cache-release calls returned without exception;
   - all four body/cook requests have distinct valid StageCache IDs and stage
     identifiers while all four stage objects remain alive through comparison;
   - the final four-cook checkpoint re-queries every retained stage, requires
     the same live ID, and resolves each recorded ID back to the same layer
     identifier through `StageCache.Find`;
   - source vertex/triangle stream hashes and parameter readbacks match.
6. Global cooking-statistics before/after/deltas are persisted verbatim as
   informational observations only. Positive deltas, zero deltas, hits, misses,
   and warning counters do not decide D339 pass/fail.
7. Canonicalize every returned convex's **vertex set** with outward Qhull
   winding and stable part sorting. Here canonical topology means the
   reconstructed convex-solid topology actually authored by D339, not the
   nondeterministically ordered callback polygon stream. Persist the callback
   payload plus separate canonical vertex-stream, topology, and combined
   geometry hashes so the comparison is independently re-auditable. For every
   body, the two cooks must have:
   - equal hull count;
   - equal canonical vertex/triangle counts and triangle topology;
   - bit-exact equal canonical combined geometry sha256 for every paired part
     (the primary, stricter equality gate);
   - maximum paired coordinate difference `<=1e-9m` (a simultaneous hard check
     and diagnostic; combined-hash equality already implies exact bytes);
   - `1..64` positive-volume finite convex parts, each with `<=64` vertices.

Any Phase-A failure is an immutable attempt2 STOP. Callback evidence already
written remains evidence of what occurred; failure never licenses derivative
authoring or physics.
An invocation that begins with any existing attempt2 path is refused before the
Isaac app launches and writes no new abort or other file inside that attempt.

The PhysX callback has no cache-provenance field. Therefore D339 claims two
independent API requests on fresh retained stages under cache-disabled/released
conditions; it does **not** claim that the API proves an internal cache miss or
process-level recomputation. The global counters cannot supply that stronger
claim.

## 4. Conditional derivative, live gate, and physics

Only complete Phase-A PASS may reuse D338's remaining registered pipeline:

1. Copy the canonical USD bundle forward-only into attempt2 and author the
   first cook's frozen explicit convex pieces. Only the physics layer may
   change; the exact D338 semantic allowlist, source-layer audit, mass/inertia
   parity, and hash contract remain hard.
2. Load only that derivative and pass D333 support/sensor/source contracts,
   exact live owner/path binding, per-piece direct-cook surface `<=0.1mm`,
   broad property-volume binding `<=5%`, GPU compatibility, and zero-step
   simulation-counter gates.
3. Re-run the unchanged D337 controls and frozen `(7,11), q5=1.5413` target.
   Raw and cooked link5/gripper must both be `>=+0.1mm` clear and differ by at
   most `0.5mm`. A failure forbids physics.
4. Only a complete live-gate PASS licenses the unchanged 200-step sole-support
   baseline and 200-step frozen-target settle. All D338 static gates remain.
5. D339 stops after this static outcome. It never runs or licenses the G0a
   10-trial gate.

## 5. Pre-registered outcomes

1. `D339_G0A_COOK_WITNESS_CONTRACT_FAIL_STOP` - callback-first validity,
   isolation, or two-cook geometry equality failed; no valid derivative.
2. `D339_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP` - witness passed but derivative
   mutation/copy/semantic/hash contract failed.
3. `D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP` - derivative exists but live
   source/owner/path/direct-cook/control/zero-step contract failed.
4. `D339_G0A_REPAIRED_COOKED_TARGET_NOT_CLEAR_STOP` - live representation is
   auditable but fails the frozen `+0.1mm` clear or `0.5mm` task-fidelity gate.
5. `D339_G0A_COLLISION_REPRESENTATION_STATIC_SUPPORTED_STOP` - callback,
   build, live target, baseline, and static-clean gates all pass. This licenses
   only a separately pre-registered frozen 10-trial case; it does not set
   `g0a_pass=true`.
6. `D339_G0A_STATIC_RUNTIME_MIXED_STOP` - baseline or target dynamics fail a
   registered static gate.
7. `D339_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` - scientific evidence
   is retained but the visualization/artifact contract fails.

## 6. Session progress rule and non-goals

The fresh attempt2 cook is a failable perturbation evaluation: callback
validity, independent geometry equality, derivative fidelity, and conditional
settle can each fail and change the decision. If Phase A fails, the invalid
intervention cannot safely advance to physics; the session records that
explicitly. No training is authorized.

Non-goals: no target/q5/IK/waypoint/solver/seed/decomposition change; no mesh
repair/rewrite, SDF, proxy boxes/capsules, second candidate, 10-trial, G0b,
close/lift, RL/PPO, randomization, VLA, real RoArm, cube, B200, cleanup,
JOINT_LIMITS removal, commit, or push.

## 7. Pre-runtime implementation audit

The first D339 Isaac invocation has not occurred. The finalized harness is
`sim_scripts/cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair.py`,
sha256
`fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6`.
Machine-readable registration is
`claudedocs/runtime_logs/grasp_track/g0a_d339/d339_preregistration.json`.

Three independent read-only audits reached READY after pre-runtime blockers
were repaired:

1. callback evidence is written inside the request boundary before stats,
   settings restoration, or classification and is never overwritten;
2. local PhysX 107.3.26 enum/Float3/index/polygon/Float4 APIs, inline sync
   callback, serialization error handling, and final StageCache `GetId`/`Find`
   lifecycle checks are compatible;
3. D338 decomposition/prim/tolerance identity, attempt1 hashes, witness->hull->
   asset manifest binding, physics-layer-only mutation, semantic/mass parity,
   and the Phase-A-to-B/C bridge are hard-gated.

Static checks passed: Python byte compilation, `git diff --check`, exact D338
parameter/path equality, attempt1 integrity, synthetic callback payload
structure, and Isaac pins `numpy==1.26.0`, `psutil==5.9.8`. No package was
installed. `attempt2` and the D339 outcome summary are absent immediately before
runtime.

## 8. Runtime execution

Final status: `D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`

The pre-registered Isaac command was invoked exactly once. It exited normally
after the scientific STOP; no retry, fallback, parameter change, or attempt2
overwrite occurred. The harness and machine-readable preregistration retained
their pre-run hashes:

- script:
  `fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6`;
- preregistration:
  `e3413268794f5741cd9114a3a27b747af6eb90472b0cb0b73c4bd83cda5e243a`.

The result separates cleanly into a passing cook/build phase and a failing live
realization phase.

### 8.1 Callback-first cook witness: PASS

All four requests (`link5_cold1`, `link5_cold2`,
`gripper_link_cold1`, `gripper_link_cold2`) produced exactly one inline
callback with `RESULT_VALID` (`value=0`), exactly `64` returned and serialized
convexes, zero serialization errors, and structurally valid payloads. The four
retained StageCache IDs were distinct and still resolved to their original
stage identifiers at the final lifecycle check:

- link5 cold1/cold2: `9223002`, `9223003`;
- gripper cold1/cold2: `9223004`, `9223005`.

Every cook read back the frozen parameters. The `minThickness` float readback
was `0.00009999999747378752m`, within the pre-registered tolerance for
`0.0001m`; every other registered value was exact. The three cache/update
settings were disabled during each request and restored to their saved values
without error.

The global statistics were persisted only after the callback evidence. All six
after-minus-before deltas remained zero on all four cooks: scheduled, finished,
cache hit, cache miss, polygon warning, and GPU warning. This independently
confirms D338's measurement lesson: these counters do not decide whether this
synchronous callback path returned a valid cook.

### 8.2 Two independent cook geometry equality: PASS

For both bodies, cold1 and cold2 returned `64/64` parts. The canonical
outward-Qhull solids matched part-by-part in vertex count, triangle count,
topology, vertex-stream hash, topology hash, and combined geometry hash. The
largest paired coordinate difference over either body was exactly `0.0m`
against the `1e-9m` limit.

An independent post-run recomputation read the four raw callback witnesses and
canonical arrays rather than trusting the summary booleans. It reconstructed
the little-endian vertex/topology digests for all paired parts; all digest and
array comparisons passed. Thus D339 licenses the precise claim that two fresh,
cache-disabled API requests on distinct retained stages returned identical
canonical geometry. It does not claim an internal process-level cache miss,
which the API does not expose.

Both bodies reached the frozen `maxConvexHulls=64` ceiling. That is not a Phase-A
failure, but it means live surface fidelity must carry the adequacy decision;
the zero warning counters do not discharge cap pressure.

### 8.3 Derivative asset build: PASS

The first cook's `64 + 64` convexes were authored into the forward-only
attempt2 derivative. Every registered build check passed:

- source layers/STLs and full link5/gripper source meshes matched;
- D338 helper implementation stayed hash-pinned;
- non-physics layers were copied bit-exactly;
- only the physics layer changed, from
  `1df07d387da76dcde4cd700ee1f9546cba25965776a9700897314ef884c37ed2`
  to
  `9261986d363327e33beb0b555d0ffce320416e827e0b1a8532c8e938d25b8e44`;
- the semantic mutation allowlist and mass/COM/inertia parity passed;
- witness -> hull -> asset manifest hash bindings passed.

D338 attempt1 remained an exact three-file inventory with its three registered
hashes unchanged both after the cooks and after asset build.

### 8.4 Live collider realization: FAIL

Stage, sensor, retained raw-source, frozen candidate, D337-control, mass, owner,
transform, bounds, authored-hash, parameter-readback, GPU-compatibility, and
zero-step guards passed. The failure was narrower: the authored explicit convex
parts did not all survive the next live PhysX convex-hull cook within the
registered surface tolerance.

| Body | Manifest | Enabled USD parts | Property rows | One-convex direct cooks | Fully certified |
|---|---:|---:|---:|---:|---:|
| `link5` | 64 | 64 | 65 | 64/64 | 56/64 |
| `gripper_link` | 64 | 64 | 65 | 64/64 | 59/64 |

The property query had no error or missing expected path. Its sole extra row
per body was the known old `node_STL_BINARY_` path even though USD inventory
marked that source collider disabled. This is evidence about property-query
enumeration only; it is not evidence that the disabled legacy shape actively
participated in collision.

Every new part's live-instance and prototype request returned
`RESULT_VALID`, one convex, without exception. Nevertheless the hard
authored-vs-live surface gate (`<=0.1mm`) failed on exactly 13 parts:

- gripper: `part_000 0.245397mm`, `part_035 0.557594mm`,
  `part_036 0.699067mm`, `part_048 0.597893mm`,
  `part_057 0.484295mm`;
- link5: `part_011 0.471243mm`, `part_018 0.188496mm`,
  `part_023 0.470118mm`, `part_024 0.687223mm`,
  `part_040 0.642557mm`, `part_041 0.241435mm`,
  `part_045 4.894877mm`, `part_054 0.410824mm`.

In all 13, the live-to-authored directed surface distance was zero while the
authored-to-live distance equaled the reported error. The live cook discarded
one extreme vertex in 12 parts and two vertices in link5 `part_041`.
`link5/part_045` also failed property-vs-direct volume binding:
`27.331672% > 5%`. All gripper volume bindings and the other link5 volume
bindings passed.

Consequently, `property_paths_exact`, `all_parts_direct_certified`, and
`live_part_count_exact` failed for both bodies. The `56/64` and `59/64` counts
are certification-filter counts, not missing authored parts: all 64 parts exist
and all 64 direct-cook successfully for each body.

### 8.5 Pre-physics stop and artifacts

Because the live audit failed, the cooked union was intentionally not queried.
The representation gate contains no per-body target distance; no statement
about frozen-target raw-vs-cooked fidelity is licensed. The simulation counter
stayed `0 -> 0`; no 200-step baseline, 200-step target settle, or 10-trial ran.
Controlled physics steps were `0`.

The artifact contract passed: six frame markers, one decision snapshot, and one
non-empty RRD with one diagnostic step. There is intentionally no distance,
baseline, target, or final trace because the preceding live gate stopped them.

- summary sha256:
  `727fe02f36cf6ae84260985bccf5324b9716e5338623a28003c92589b99f8418`;
- cook witness manifest sha256:
  `7d0a82842af141c1e194ffcb5f9947777b8087c8fd56c72e13f684cf61481e81`;
- asset manifest sha256:
  `3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589`;
- live audit sha256:
  `6148252b654a6250faf78a1ebcde4caa57870e800fa1d3c45b93c803fdf882cb`;
- decision PNG sha256:
  `818533e1808654b96284c8cc53a8eacfef46411ce8a4f2ef0962aabb25adafec`;
- RRD sha256:
  `1b2cb570f3e78fdf2c231777f8baba048bdfa71e621c31b286145acff762f8d1`.

## 9. Verdict and next decision

Verdict: `D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`.

D339 repaired the D338 witness contract and proved repeatable decomposition,
but repeatable callback geometry was not sufficient to certify the authored
live collision representation. Re-authoring callback convexes as USD
`convexHull` meshes triggers another PhysX cook that can discard extreme
vertices and change the live surface. Per-piece live surface/property binding
therefore remains mandatory.

The narrowest next candidate is a separately approved, prephysics-only D340
fixed-point live-authoring case. It should retain the 115 already passing parts,
capture both live-instance and prototype geometry for the 13 failures, and
re-author only stabilized failing pieces forward-only without changing the
source, target, physics, decomposition parameters, or thresholds. Its property
contract should distinguish the exact 64 enabled expected parts from the one
known USD-disabled legacy enumeration row, without treating enumeration as
active collision. It must reach `64/64` surface certification per body and
`128/128` property/direct volume binding before any cooked-union query or
physics. This is a recommendation only; no D340 change is authorized here.

`g0a_pass=false`; G0b, RL/PPO, 10-trial, and ladder promotion remain blocked.
attempt1 and attempt2 are immutable. No package was installed and no commit or
push was performed.
