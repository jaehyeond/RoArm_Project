# D380 P34 failed-part cook provenance and semantic-impact audit

Date: 2026-07-25 KST

## 1. What and why

`D379 [p34_full_live_identity_classifier_resume]` proved that the P34 candidate
was correctly bound and readable, but only `17/34` authored collider parts
retained the frozen authored-to-cooked identity contract. D380 was approved to
explain the other `17` parts without changing the acceptance tolerances and
without launching Isaac/Kit/PhysX.

This case answers only:

1. whether the cooked callback geometry retained, omitted, moved, or introduced
   authored vertices;
2. whether the resulting shape change is inward or outward in the same body-local
   frame;
3. what can be concluded about jaw material, authored voids, and geometric
   clearance from that one-sided relation.

It does not answer actual OPEN jaw clearance, cylinder contact, q5 closure,
physics, tipping, grasp feasibility, or target/IK/path repair.

## 2. Preregistered scope

이번 case의 신규 변수:

- `failed_part_cook_provenance_classifier_v1`
- `body_local_semantic_monotonic_impact_contract_v1`

Sole measurement-authority input:

- `claudedocs/runtime_logs/grasp_track/g0a_d379/attempt2_d372_measurement_field_repair/d379_p34_full_live_identity_evidence.json`
- SHA-256:
  `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5`

Forward-only output:

- `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/`

Frozen limits inherited from D379:

- symmetric surface distance: `0.1mm`
- bounds difference: `0.1mm`
- authored-to-callback original-topology volume relative difference: `0.5%`
- polygon-plane residual: `1e-5m`

The exact internal PhysX cook heuristic/tolerance was not inferred. The
`hullVertexLimit=64` value was retained only as a causal boundary check.

Forbidden and held at zero:

- tolerance changes;
- asset/USD reads or writes;
- Isaac, Kit, or PhysX imports, launches, or calls;
- collider materialization, regeneration, or automatic-decomposition sweep;
- cylinder creation/write, physics/public-forward, q5 command/sample, contact;
- target/IK/path/pose or material/mass/actuator/physics-setting changes.

The offline audit was failure-capable: prepare perturbations passed `4/4`, and
the actual evidence negative controls passed `7/7`. No runtime physics
experiment was allowed because this reactive case was explicitly approved as an
immutable-D379 provenance audit after the D379 identity failure.

## 3. Execution in observable order

1. The source script, START_HERE authorization state, D379 measurement input,
   dirty-worktree baseline, interpreter, and package pins were frozen by hash.
2. Preregistration passed `24/24`.
3. The supervisor launched exactly one offline worker with the frozen
   `/home/cgxr/miniconda3/envs/isaaclab/bin/python` interpreter. Automatic retry
   count was zero.
4. The worker read only the embedded D379 JSON arrays. For every P34 part it
   compared authored Float32 JSON vertices/faces with cooked callback
   vertices/triangles in the proven same body-local binding.
5. It computed retained, omitted, and introduced-or-moved vertex sets,
   containment certificates, original-topology part-volume diagnostics, failed
   gate signatures, and role-scoped inward-distance bounds.
6. It generated one exact `1920x1080` board plus save-only RRD/RBL and ran the
   strict Rerun validation.
7. Original-resolution manual inspection was performed separately.
8. Completion finalization preserved the numeric audit verdict but failed the
   presentation contract because the manual inspection failed.

Supervisor result:

- worker/retry: `1/0`
- return code: `0`
- elapsed: `3.5377263869158924s`
- timeout/TERM/KILL/process-group residue: all false
- supervisor SHA-256:
  `e14f7bee95e118c67d8194ba6d46abc13166f5382b90be3ca67f3beefe1a13a7`

The only stderr was Matplotlib selecting a writable `/tmp` configuration
directory. It was not an Isaac or PhysX error.

## 4. Canonical numeric result

Canonical evidence:

- `d380_p34_failed_part_cook_provenance_evidence.json`
- SHA-256:
  `4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1`
- `audit_pass=true`
- verdict:
  `D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`

### 4.1 Exact failed set and gate signatures

- failed parts: `17/34`
  - link5: `4/16`
  - gripper_link: `13/18`
- failed-gate signatures:
  - surface only: `2`
  - surface + volume: `12`
  - plane + surface + volume: `1`
  - bounds + surface + volume: `2`

This exactly reproduces the D379 failed set.

### 4.2 Vertex provenance

Across the 17 failed parts:

- authored unique vertices: `401`
- cooked retained vertices: `178`
- omitted authored vertices: `223`
- omitted vertices beyond the inherited `0.1mm` surface limit: `181`
- introduced or moved JSON-coordinate vertices: `0`

Across all `34/34` parts, every cooked callback vertex set is a JSON-numeric
exact subset of its authored vertex set, and every cooked shape passes its
authored convex containment certificate in the same body-local frame.

This is JSON numeric equality after parsing the embedded evidence, not a claim
of raw-memory byte identity.

The classification for all 17 failed parts is:

`AUTHORED_VERTEX_ELISION_WITH_INWARD_COOK`

In plain terms, the cooked collision shape did not grow outward. It discarded
some authored corner/support points and became a smaller subset of the authored
convex part.

### 4.3 Volume diagnostic

For the failed parts only, sums of per-part original-topology volumes are:

- authored sum: `8708.834857803575mm^3`
- cooked sum: `8367.592932676003mm^3`
- signed loss: `341.24192512757054mm^3`
- signed loss: `3.91834189876502%`

These are sums of individual part volumes, not the boolean union volume of the
compound. P34 parts can overlap, so these values must not be presented as actual
void volume or whole-collider material loss.

Role diagnostics:

| Role | Failed / parts | Max inward surface (mm) | Part-volume sum loss (mm^3) | Loss (%) |
|---|---:|---:|---:|---:|
| fixed jaw | 2 / 10 | 0.657364511940 | 14.156195806668 | 1.401972256978 |
| fixed-jaw backbone | 2 / 2 | 0.684166832185 | 106.240540626035 | 4.649275441991 |
| moving jaw | 7 / 12 | 0.441658680073 | 11.718767219378 | 0.837662440852 |
| moving-jaw backbone | 2 / 2 | 0.269509640144 | 6.762390527489 | 0.804274943819 |
| moving support | 4 / 4 | 0.683359316380 | 202.363996175972 | 4.965120882387 |

The main named window core remains exact at the D379 surface gate: moving-jaw
center bridge and upper/lower rails plus the five fixed-jaw lower/upper
legs and middle bridge all have `surface_symmetric_mm=0.0`.

### 4.4 What the geometry proves and does not prove

At the same rigid transforms:

- cooked union is a subset of authored union;
- cooked geometry cannot newly fill an authored void;
- pure set-to-set geometric clearance cannot decrease; it stays equal or
  increases;
- cooked geometry cannot create an earlier outward protruding contact;
- contact surface can be removed, so a contact can occur later or disappear.

Role-scoped one-sided separation-increase upper bound:

- fixed-jaw system: `0.684166832184637mm`
- moving-jaw system: `0.4416586800734206mm`
- summed bound: `1.1258255122580576mm`

This is a conservative geometry bound, not an observed OPEN-pose jaw-gap change.
The actual OPEN jaw clearance and actual cylinder-facing patch remain `null`
because D379 does not contain both jaws and the cylinder in one authoritative
world frame.

The statement also does not cover `contactOffset`, `restOffset`, CCD, solver
timing, penetration depth, or contact-report timing. Those require a separately
approved live/physics case.

### 4.5 Why the 64-vertex setting is not the observed cause

- maximum authored unique vertices in any P34 part: `44`
- inherited `hullVertexLimit`: `64`
- no P34 part exceeded the limit

Therefore the 17 failures were not forced by an input part exceeding the
64-vertex cap. The exact internal cooking heuristic or tolerance responsible
for the deletion remains `null`.

This matches the NVIDIA terminology boundary: `convexHull` requests a convex
hull collision approximation, while cooking is the generation of collision
approximations from mesh data. It does not mean the cooked output must preserve
every authored input vertex.

### 4.6 Independent D379-only cross-check

An independent read-only recalculation reproduced the failed set, gate
signatures, vertex counts, subset/containment result, role bounds, and stored
volume sums. It found no decision-changing mismatch or unsupported semantic
claim.

Direct integration from the embedded authored Float32 points gives authored
volume `8708.833874651791mm^3` and loss `341.240941975788mm^3`, differing from
the registered D379 stored D372 original-topology authority by only
`0.000983152mm^3`. D380 intentionally uses the preregistered D379 stored
authority, not this new roundoff-sensitive reintegration. The difference does
not affect any gate or verdict.

Official references:

- NVIDIA Omni Physics 107.3, “Colliders”:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html
- NVIDIA Omni Physics 107.3, “Query The Mass and Volume”:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/mass_inertia_queries.html

## 5. Visualization result

Artifacts:

- exact board:
  `d380_p34_failed_part_cook_provenance_1920x1080.png`,
  `1920x1080`, SHA-256
  `61317db2dd22e94f35ea37e8b9258fe02eba29e57ec92b96d014356d14a4d9ca`
- RRD:
  `d380_p34_failed_part_cook_provenance.rrd`, SHA-256
  `7ae91348bc6cc64b583c1e92ff2ea8776647a660042471a075d9216b9fadcaff`
- RBL:
  `d380_p34_failed_part_cook_provenance.rbl`, SHA-256
  `a2b8eed159ecb48b4c816a5e0b0565bc36796f7f1fb05dc92923a73b1115683f`
- Rerun inspection screenshot:
  `d380_rerun_inspection.png`, `3840x2160` physical pixels for the requested
  `1920x1080` logical window, SHA-256
  `730374f419654e829177a426418ca2756fc503e59f563ace01f770b3c9bcb8c6`

Automated Rerun validation passed: the exact required entities, components, and
timelines were present, and no `Unknown timeline` appeared.

Manual visual inspection failed:

- the board subtitle overlaps both upper 3D panel titles;
- several left-side bar labels are clipped;
- the Rerun startup notifications cover part of the right-hand summary.

Therefore the presentation/completion verdict is separately:

`D380_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`

The attempt1 artifacts are frozen. The presentation failure does not change the
canonical JSON geometry result, but the flawed board/screenshot must not be
called presentation-complete.

## 6. Final interpretation and boundaries

D380 establishes that the D379 P34 identity failures are one-sided inward
erosion caused by authored-vertex omission, not outward expansion and not a
64-vertex-cap overflow. That is enough to reject the current P34 cooked shapes
as authored-identity-equivalent and to require representation repair before
P34 cylinder physics.

It does not prove that the physical jaw opening increased by
`1.1258255122580576mm`, that an authored mouth/window kept the same measured
volume, or that cylinder contact will be delayed. Those remain hypotheses or
bounds until measured in a common live frame.

Frozen state:

- P34 authored-to-cooked identity: false
- D380 canonical numeric audit: PASS, repair required
- D380 presentation completion: FAIL
- current-pose closure, cylinder contact/tipping, grasp feasibility,
  target/IK/path justification: null
- `g0a_pass=false`

No representation repair, live verification, 29x50 target rebase,
physics/q5/contact, or presentation-repair rerun is approved by this case.

Recommended separate next boundaries:

1. a forward-only observability-only repair that reads immutable D380 artifacts
   and fixes only the board/Rerun overlap and clipping;
2. after separate approval, an offline P34 representation-repair design that
   reaches an authored-to-cooked fixed point without relaxing D379 tolerances;
3. a separate live-identity PASS before any P34 cylinder physics;
4. only then rebase the nominal `29x50mm` primitive cylinder and proceed through
   pose and physics as separately approved cases.

## 7. Evidence index

- script:
  `sim_scripts/cyl34_top_view_d380_p34_failed_part_cook_provenance_and_semantic_impact_audit.py`
- preregistration:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_preregistration.json`
- supervisor:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_offline_worker_supervisor.json`
- canonical evidence:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_p34_failed_part_cook_provenance_evidence.json`
- per-part CSV:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_failed_part_metrics.csv`
- manual inspection:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_manual_visual_inspection.json`
- completion summary:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/d380_completion_summary.json`
