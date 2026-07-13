# Session 2026-07-13 - D340: fixed-point live-authoring repair

Final status: `D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`

Historical pre-runtime status: `D340_PRE_REGISTERED_CAPTURE_PENDING`

이번 case의 신규 변수:
`[failing_part_fixed_point_geometry, enabled_shape_property_binding_contract]`
(정확히 2개)

- `failing_part_fixed_point_geometry`: physical variable, but the mutation
  allowlist is exactly the 13 D339-failing derivative convex parts.
- `enabled_shape_property_binding_contract`: measurement-only variable that
  distinguishes the 64 enabled expected parts from the one known USD-disabled
  legacy row returned by the property query.

No existing physical, decomposition, cache, target, controller, solver, or
tolerance scalar is increased or changed. D340 is prephysics-only and cannot
promote G0a.

## 1. What and why

D339 repaired the D338 callback witness and produced a valid forward-only
attempt2 derivative. Four source-decomposition callbacks each returned exactly
once and inline with `RESULT_VALID`, 64 serialized hulls, and repeatable
geometry. The derivative build preserved nonphysics layers and tool mass
semantics.

The remaining D339 failure is downstream: authoring each returned convex as a
USD `convexHull` mesh invoked another PhysX cook. Thirteen authored pieces did
not reproduce their own live surface within the frozen `0.1mm` gate:

- gripper: `part_000`, `part_035`, `part_036`, `part_048`, `part_057`;
- link5: `part_011`, `part_018`, `part_023`, `part_024`, `part_040`,
  `part_041`, `part_045`, `part_054`.

The other 115 pieces already passed and are not search variables. D340 applies
the live cook once to only those 13 pieces, authors the shared
instance/prototype result through an explicit float32 round trip, then uses a
fresh process to ask whether the authored result is a true fixed point.

The second D339 measurement issue is deterministic rather than physical:
authored USD has exactly 64 enabled new parts plus one disabled legacy collider
per body, while the PhysX property query enumerates all 65 paths. D340 therefore
requires the exact 65-row inventory but certifies only the 64 USD-enabled
parts. A disabled row's enumeration is not evidence that it actively collides.

## 2. Parameter-increase audit

Machine-readable source:
`claudedocs/runtime_logs/grasp_track/g0a_d340/d340_parameter_freeze_audit.json`
(pre-runtime sha256
`9e7f5dae62f0aa4f3a0f7d936ab818bce54e9d896e555475c115bd6a3ed5526b`).

The exact comparison is D338 preregistration -> D339 execution -> D340
registration. Results:

- existing parameter increases: `0`;
- existing parameter changes: `0`;
- physical scalar changes: `0`;
- decomposition changes: `0`;
- threshold/tolerance relaxations: `0`.

In particular, D339's observed `64 hulls/body` is not a parameter increase.
`maxConvexHulls=64` was already frozen in D338; the result saturated that
existing cap.

Frozen decomposition values remain:

- `hullVertexLimit=64`;
- `maxConvexHulls=64`;
- `voxelResolution=1_000_000`;
- `errorPercentage=1.0`;
- `minThickness=0.0001m`;
- `shrinkWrap=true`.

During every explicit request D340 retains D339's cache contract:
`updateToUsd=false`, `UJITSO=false`, `useLocalMeshCache=false`, with local and
runtime caches released before the request. It then restores the exact values
saved before that request. The D339-observed saved values were respectively
`false/true/true`; the contract is save -> isolate -> exact restore, not a new
hard-coded environment setting.

Frozen representation gates remain:

- independent-channel coordinate equality `<=1e-9m`;
- authored/live surface `<=0.1mm`;
- property/direct volume difference `<=5%` hard gate;
- raw anchor `<=0.05mm`;
- raw/cooked task-fidelity difference `<=0.5mm`;
- target clearance `>=+0.1mm`.

Frozen task values remain `(radial,tangent)=(7,11)mm`, tangent sign `-1`,
`q5=1.5413rad` (OPEN), seed `33201`, HOME-seeded position-only IK with
`max_iter=120` and `1mm` position tolerance. Object, table, actuator, solver,
mass/COM/inertia, friction, and sensor values are also exact in the parameter
audit. D340 requires the simulation counter to remain unchanged and runs zero
controlled physics steps.

## 3. Immutable inputs and forward-only output

- Boot HEAD: `a51369554625b3a4bf31142193a4ae5f726dbf89`.
- D338 helper sha256:
  `f3d330a9a5ca6f886728d0e5dc8037baa68d83a2b911aa105904d7d369ead426`.
- D339 helper sha256:
  `fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6`.
- D339 canonical summary sha256:
  `727fe02f36cf6ae84260985bccf5324b9716e5338623a28003c92589b99f8418`.
- D339 live audit sha256:
  `6148252b654a6250faf78a1ebcde4caa57870e800fa1d3c45b93c803fdf882cb`.
- D339 attempt2 asset manifest sha256:
  `3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589`.
- D339 attempt2 cook-witness manifest sha256:
  `7d0a82842af141c1e194ffcb5f9947777b8087c8fd56c72e13f684cf61481e81`.
- D339 attempt2 hull manifest sha256:
  `d70a13edbb8500cde97ad23779811475e1c8bb2d0f6045b4183e704d2157bedd`.
- D338 attempt1 exact three-file inventory and all 28 D339 files are pinned in
  `d340_preregistration.json`; a pre-runtime reconstruction matched all 28
  paths and hashes exactly.

D338 attempt1 and D339 attempt2 are immutable. D340 writes only under:
`claudedocs/runtime_logs/grasp_track/g0a_d340/`.
The only allowed derivative is:
`collision_asset/attempt3/`.

Harness:
`sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py`
(pre-runtime sha256
`1bdea659fadd7801b0f5749cca6286c7eb90c95a857ed516e09e34ceb12a023a`).

Machine preregistration:
`claudedocs/runtime_logs/grasp_track/g0a_d340/d340_preregistration.json`
(pre-runtime sha256
`e4317d638fc48a6c591eed5a748935f299a32c584a5488f463d37acac179a038`).

## 4. Registered sequential procedure

### Stage A - capture on immutable D339 attempt2

1. Refuse before app launch unless the output root contains exactly the
   preregistration and parameter-audit files and all script/source hashes pass.
2. Reconstruct the exact D339 failure set; it must equal the registered 13.
3. Load D339 attempt2 at the frozen zero-step scene.
4. For each registered part, request `live instance` then dynamically derived
   `prototype`. Before each synchronous request, disable the same three cook
   settings and release both caches.
5. Persist callback enum/count/full convex payload before classification.
   Require exactly one inline callback, `RESULT_VALID`, one convex, and a valid
   payload for both channels.
6. Canonical instance and prototype geometry must be bit-exact, including
   hashes/topology, with coordinate delta `<=1e-9m`. Divergence is a STOP; one
   channel may not be selected.
7. Require the shared output to be contained in the authored input, to have
   strictly fewer vertices, not to cycle to the input, and to survive the
   explicit float32 authoring round trip within `1e-9m`.
8. Emit the 13 candidates, 26 callback witnesses, a diagnostic PNG, frame
   markers, and a non-empty RRD. Attempt3 must still be absent.

### Stage B - one authoring application and fresh validation

1. Run only if Stage A's summary and exact artifact inventory pass.
2. Copy the pristine D339 derivative forward to attempt3. Change only
   `points`, `faceVertexCounts`, and `faceVertexIndices` on the exact 13 paths.
3. Require the other 115 parts to be exact. Remove only the 39 registered
   geometry properties from source/variant layer clones and require the entire
   remaining physics-layer prim/path/type/schema/metadata/attribute/
   relationship inventory to be equal. Require all nonphysics files and tool
   mass semantics to remain exact.
4. Load attempt3 in a fresh process. For all 128 parts, request `prototype`
   then `live instance`, deliberately reversing Stage A's order.
5. Require both channels to agree exactly. For the 13 changed parts, require
   both `F(x1)=x1`; if a second change is available, record it and STOP. No
   iterative retry is registered.
6. Require exactly 64 enabled USD parts plus one exact disabled legacy prim per
   body; require the property query to return those exact 65 paths. Certify only
   the 64 enabled parts.
7. Require surface `64/64` per body and property/direct volume `128/128`, plus
   owner, identity transform, parameter readback, mass, and GPU checks. In
   particular, link5 `part_045` must fall from D339's `27.331672%` mismatch to
   the unchanged `<=5%` gate.
8. Only after all prior checks pass, query the certified cooked union at the
   frozen target. Require both bodies `>=+0.1mm` clear and raw/cooked delta
   `<=0.5mm`.
9. Emit the final JSON/Markdown summary, attempt3 manifest, 256 callback
   witnesses, live audit, decision PNG, frame markers, and non-empty RRD.
   Physics remains forbidden.

## 5. Registered commands

Capture:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py --stage capture
```

Validate, only after a clean capture:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python sim_scripts/cyl34_top_view_d340_grasp_g0a_fixed_point_live_authoring_repair.py --stage validate
```

## 6. Pre-runtime static checks

- Python compile: PASS.
- `git diff --check`: PASS.
- D339 attempt2 exact 18-file inventory/hash: PASS.
- D339 full exact 28-file inventory/hash against preregistration: PASS.
- D339 failure set exact 13: PASS.
- registered counts `13 changed / 115 preserved`: PASS.
- D338/D339 decomposition equality: PASS.
- physics-step/settle call search: zero.
- independent parameter audit: existing scalar increases `0`, changes `0`.

The bounded runtime uncertainty is explicit: D339 separately proved per-part
instance/prototype requests and cache-isolated cook requests, but D340 is the
first to combine both for every per-part channel. If either channel rejects
that combination, D340 preserves its callback/precall witness and stops without
fallback, authoring, or parameter change.

## 7. Runtime result

The exact registered `--stage capture` command ran once. Its canonical result
is `D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`; therefore validation did
not run and attempt3 was not created.

### 7.1 Contracts that passed

- Preflight: all seed/URDF/helper/evidence/package/hash/inventory pins passed.
- D339 failure set: exact registered 13.
- Stage, sensor, retained raw source, and USD inventory contracts: PASS.
- Simulation counter: `0 -> 0`; controlled physics steps: `0`.
- D338 attempt1 and D339 attempt2 integrity before/after: PASS.
- Callback witnesses: `26/26` exactly once and inline, `26/26`
  `RESULT_VALID` with enum value `0`, `26/26` exactly one convex, zero
  serialization-error files.
- Cache/settings: both cache releases `26/26`; isolated settings restored
  `26/26`.
- Instance/prototype consensus: `13/13`; maximum coordinate delta `0.0m`.
- Output-contained-in-input: `13/13`, maximum violation `0.0m`.
- Strict vertex decrease: `13/13`. Total vertices changed `114 -> 100`:
  link5 `73 -> 64`, gripper `41 -> 36`; link5 `part_041` removed two and every
  other part removed one.
- Float32 round trip: `13/13`, maximum surface delta `0.0m`.
- Visualization artifact contract: PASS. The inspected PNG correctly shows all
  13 vertex decreases and zero-valued containment/round-trip traces. PNG
  sha256 `5d311f047113906287e352a14a2b9acf7de4498c3e731cca0a51098d5fe1bd66`
  (`110,208` bytes); RRD sha256
  `8eb3d613033034b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47`
  (`2,480,049` bytes).

Capture summary sha256:
`ba62c879e78bfa7db47b003e1bfdd4ee2bd4ff250b083e423f94ebec67992163`.
Candidate manifest sha256:
`f288d5232f039e58ccd209f332ebfabbf9fec137e746e97d9a3c58688420ef86`.

### 7.2 Sole failing gate

Every one of the 13 part rows failed only
`authored_hash_matches_d339_manifest`. The fixed-point capture sub-contract was
otherwise true for every part. The D340 verdict remains FAIL because the gate
was pre-registered and cannot be waived after seeing the result.

Post-run discrimination showed that this is a coordinate-stream measurement
false negative rather than a source-asset or cook mismatch:

- D339's direct authored-point hash check is true for all 13 parts on the same
  immutable attempt2.
- D340 hashed points after applying the identity-gated prim-local -> body-local
  float64 transform. The transform's maximum identity delta was only
  `2.220446049250313e-16`.
- Relative to the D339 manifest, maximum bounds/centroid changes were only
  `2.220446049250313e-16m` / `1.1102230246251565e-16m`; maximum volume relative
  change was `6.772066266696707e-14`; vertex counts were equal for all 13.
- Those machine-epsilon changes nevertheless changed all 13 exact float64
  geometry hashes and changed Qhull topology hashes on 10/13 parts. A
  transformed bit hash is therefore not the same contract as the authored
  Vec3f point-stream bit hash.

Post-run root-cause audit:
`claudedocs/runtime_logs/grasp_track/g0a_d340/d340_capture_postrun_root_cause_audit.json`
(sha256
`2d6bc90a71d9ec407206ade7c32069ad209f8a0ba40cb44a40f42281318a1207`).
This audit explicitly does not reclassify D340 as PASS.

### 7.3 Outcome and everyday-language translation

D340 proved that both live cook channels agree exactly on the 13 proposed
one-step fixed-point candidates, but the case stopped before authoring because
the harness compared two different bit streams: original authored points versus
the same points after a numerically near-identity transform. In ordinary
language: the geometry measurement looked good, but the proof form was wrong,
so no collision asset was changed and no later test is licensed.

The precise next candidate is a separately approved D341 reactive
`authored_geometry_frame_contract` repair. It must preserve all D340 evidence,
compare the direct authored Vec3f stream to the D339 manifest before any
transform, retain body-mapped geometry only for proximity/containment, and then
separately preregister attempt3 authoring plus fresh validation. D341 is not
authorized in this session.

## 8. Scope guards

- `g0a_pass=false` regardless of a D340 prephysics support verdict.
- No physics, settle, or 10-trial in D340.
- No G0b, RL, PPO, or ladder promotion.
- No raw STL, canonical USD, or URDF rewrite.
- No target, q5, IK, seed, solver, actuator, object, table, mass,
  decomposition, cache-policy, or tolerance increase.
- D338 attempt1 and D339 attempt2 remain immutable.
- Git commit/push only on explicit user request.
