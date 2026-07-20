# D368 — current 64-cap semantic allocation audit (offline-only)

Date: 2026-07-20 KST  
Case: `g0a_d368`  
Status: one audit completed; measurement PASS, observability completion FAIL

이번 case의 신규 변수:

1. `semantic_contact_patch_authority`
2. `current_64cap_part_to_patch_allocation`

## 1. What and why

D338 created one high-fidelity-first decomposition candidate with
`maxConvexHulls=64` and `hullVertexLimit=64`; it was not an adaptive sweep and did not prove that
64 hulls are optimal. D339/D348 later showed that both `link5` and `gripper_link` saturated this
authored cap at `64/64`. D362 then observed moving-jaw contact followed by cylinder motion and a
toppled endpoint, but it did not test whether hull count or hull allocation caused that outcome.

D368 therefore asks one narrower offline question: **which current 64+64 callback parts contain
faces certified to the frozen contact patches, what whole-hull budget do those carriers consume,
and what non-authoritative nearest-region profile do the remaining surfaces show?** It does not
create or compare another collider candidate.

Successful completion means only that the current candidate's allocation was measured with intact
provenance. It cannot establish optimal hull count, physical equivalence, a tipping cause, actual
GPU narrowphase execution, grasp feasibility, or G0a success.

## 2. Frozen Git and input authority

- Git base before D368 edits:
  `HEAD == origin/master == 7c4819632bb193c8fd552372c919f8a107675b41`, subject
  `D367 status`; the worktree was clean.
- The subsequent `AGENTS.md` change adds only the NVIDIA official-source verification rule.
- Frozen authoring USD:
  `local_assets/roarm_m3/usd/roarm_m3.usd`, SHA-256
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`.
- Frozen D348 callback-topology evidence:
  `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json`,
  SHA-256 `83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6`.
- Frozen D350 binding/measurement SHA-256:
  `1ec1c309461357eeae89204fa55a498b64d2d216708ab6e6c7dfdd3d0b878c12` /
  `4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446`.
- Frozen D354 moving-surface binding SHA-256:
  `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15`.
- Frozen D359 provenance evidence SHA-256:
  `9a4c2aa38bfc8e26722852a328d5f228aeccba17e372b017767f4da7c281f822`.
- D339 link5/gripper witness files are raw-source hash pins only; D344 changed 13 current convex
  parts, so D339 convex part geometry is never reused as the current candidate.
- The D334 collision-table sidecar is user-owned and read/hash-only.

Numerical collider authority is the original D347/D348 callback polygon stream and D348's
corrected topology. `qhull_triangles`, newly regenerated hulls, and Rerun Float32 copies are not
authority.

## 3. Installed/version-scoped NVIDIA basis

Installed Isaac Sim is 5.1.0 with PhysX schema extension `107.3.26+107.3.3`.

- Installed `PhysxSchema/resources/schema.usda:886-901` defines schema defaults
  `hullVertexLimit=64` and `maxConvexHulls=32`.
- Installed `omni.kit.property.physx/.../database.py:954-958` exposes UI ranges
  `hullVertexLimit 8..64` and `maxConvexHulls 1..2048`. A UI range is not an optimality claim.
- NVIDIA Omni Physics 107.3, *Colliders*:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html>
  explains the primitive/convex complexity order and GPU convex aspect-ratio guidance.
- NVIDIA Omni Physics 107.3, *Current Limitations*:
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/current_limitations.html>
  states the GPU convex vertex/face limits and warns that multiple convex pieces perform contact
  detection independently, so decompositions can change physical behavior.
- NVIDIA PhysX 5.6.1, *GPU Rigid Bodies*:
  <https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html>
  gives the 64-vertex/64-polygon GPU compatibility limit and 32 vertices per polygon. This is
  supplementary official corroboration: the installed schema extension is pinned, but no local
  artifact identified the embedded PhysX SDK semantic version as exactly 5.6.1, so version equality
  is `null` rather than asserted.
- Isaac Sim 5.1.0 Core API:
  <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html>
  describes convex decomposition as a detail-versus-computation trade-off.

Offline D368 can verify callback geometry compatibility fields only. It cannot prove that D362
contacts actually executed on the GPU narrowphase; that field remains `null`.

## 4. Preregistered semantic authority

### 4.1 Fixed side (`link5`)

D350 proved a D349-witness-containing connected component with `7,250` faces and `3,519` welded
vertices. That broad connected shell is **not** renamed as the fixed jaw pad.

D368 derives a narrower `D350_seed_contact_plane_patch` from frozen seed face `1984`:

1. weld raw vertices by exact Float64 coordinate identity;
2. take faces whose every vertex lies within `1e-9m` of the seed support plane;
3. require oriented normal dot seed normal `>= 1 - 1e-12`;
4. retain only the welded-edge-connected component containing seed face `1984`;
5. require it to be a subset of the frozen D350 connected component and owned by `link5`;
6. require BFS and union-find traversals over the same frozen candidate/weld/edge inputs to produce
   the same face set and canonical digest. This is traversal agreement, not two fully independent
   pipelines.

A pre-audit feasibility check, which did not inspect live-hull allocation, registered the expected
binding as 267 faces, 255 welded vertices, face IDs `1740..2006`, on local plane
`x=-0.010025849818548...m`. If the sole audit does not reproduce this exactly, fixed-side
allocation is `null` and the case stops.

The remaining link5 labels are `D350_witness_component_remainder` and
`other_raw_connected_components`; neither is renamed housing/neck/tip.

### 4.2 Moving side (`gripper_link`)

- closing-facing inner source patch: face IDs `672..1164`, 493 faces, normal `-local-Y`;
- paired outer negative patch: face IDs `13205..13697`, 493 faces, normal `+local-Y`;
- all other raw faces: structural remainder.

The historical identity recipe is D359's ascending original USD point-ID remap. Coordinate-row
lexicographic remap and reverse point-ID order are negative controls. The D354 frozen live-inner
partition must reproduce 40 callback triangles across these 17 parts:
`030,035,042,045,046,047,048,050,051,053,056,058,059,060,061,062,063`, with face-key SHA-256
`5bb7ad8a21826cb0709da55f85b0e3772114a782e1263483c180963aa9eccab5`.
This inherits only the D354 live-partition subresult; D354's overall `pass=false` remains immutable.

## 5. Preregistered allocation measurements

For every one of the 128 D348 parts, D368 records the source witness path/hash, callback vertex,
original polygon-index/descriptor, and corrected-topology hashes, plus vertex/polygon/triangle
counts, maximum polygon size, and property-query volume. The large authoritative arrays remain in
the frozen D348 evidence/witness files and are hash-pinned rather than duplicated into D368 JSON.

It separates two concepts:

- **certified patch surface**: callback triangles satisfying the frozen/new plane, normal, owner,
  and source-patch projection rules;
- **whole-hull budget carrier**: the full convex part containing such a face. The carrier's full
  volume is not called pad volume, unique occupied volume, or mass because convex parts can overlap.

For the newly measured link5 fixed patch and moving outer patch, zero certified callback faces is a
valid allocation result, not an integrity failure. If a nonzero certified set is found, its frozen
source-projection contract must pass. Only the inherited D354 moving-inner result is required to
reproduce the exact registered `40 faces / 17 parts / hash` lineage.

Raw patch samples use deterministic vertices, unique edge midpoints, triangle centroids, and fixed
barycentric refined points. Exact triangle-surface nearest queries produce per-method nearest-part
attribution sets. Their intersection is `recurrent_nearest_sample`; their union is
`union_nearest_sample`. These are finite-sample distance diagnostics, not proof that a hull supports
contact. Ties within `1e-9m` remain multi-assigned, and `certified_surface_carrier` remains the only
authoritative contact-patch allocation label.

Full live vertices/centroids/refined samples also get a tie-inclusive nearest-raw-region profile with
explicit counts, denominator, and incidence fractions. That profile is diagnostic only. A part is
classified `certified:<region>` or `mixed_certified:<regions>` only from the registered callback-face
contracts; a part with no certified contact face stays `no_certified_contact_face`. This is a
neutral scope label, not an error. It is never forced into a raw semantic region merely because
some region is the closest one.

Reported metrics include:

- certified carrier IDs plus recurrent/union nearest-sample attribution IDs and count out of 64;
- vertices, original polygons, max vertices per polygon, corrected topology triangles;
- carrier part-volume sum explicitly marked overlap-prone diagnostic;
- patch-to-live and live-support-to-raw sampled max/P95/RMS distances;
- normal disagreement and D350 seed witness nearest-part distance;
- per-part tie-inclusive nearest-region sample incidence fractions plus mixed-certified and
  no-certified-contact-face inventory;
- offline NVIDIA compatibility observations (`vertices<=64`, `polygons<=64`,
  `vertices_per_polygon<=32`, finite aspect-ratio diagnostic).

The measured distance distributions are metrics, not a new hard fidelity threshold. In particular,
D338's historical `0.5mm` task-distance gate is not repurposed as a whole-patch Hausdorff gate.

## 6. Failure-capable perturbation controls

The sole audit must reject or distinguish all registered perturbations. For rejection controls, the
same pure predicate must accept the frozen baseline and reject the perturbed candidate:

1. swap `link5` and `gripper_link` ownership;
2. substitute the moving outer patch as inner and reject it by closing-motion normal sign;
3. substitute vertex-only Qhull topology for callback polygon authority;
4. add the disabled legacy collider as a 65th part;
5. multiply metre geometry by 1000 as if already millimetres;
6. permute part order: ordered stream changes, canonical aggregate remains invariant;
7. use D359 coordinate-row and reverse-point-ID remaps instead of original point-ID order;
8. remove the D350 seed-bearing face set and demonstrate loss/worsening of the seed anchor.

These perturbations satisfy the research-session requirement that the evaluation can fail without
running Isaac or physics. Mere constants, length arithmetic, or "hash changed" assertions do not
count unless the perturbed candidate is passed through the same registered predicate.

## 7. Visualization and completion contract

Because allocation is spatial, Rerun is mandatory. The numerical decision remains in original
Float64/callback JSON; Rerun receives only display copies.

Expected visual evidence:

- fixed 2x2 Rerun layout: link5 full/seed-patch zoom, gripper full/inner-patch zoom;
- raw contact patches cyan, certified contact carriers green, inner+outer dual carriers yellow,
  outer negative carriers purple, and no-certified-contact-face/noncritical carriers translucent
  blue/gray;
- zoom views query only the critical raw patch and certified/dual carrier hierarchy so the other 64
  hulls cannot visually bury the decision surface;
- one `2400x1400` logical headless screenshot, allowing the environment's 2x DPR raster;
- one exact `1920x1080` professor-facing summary PNG with short ASCII labels;
- exact SDK/CLI `0.34.1`, footer, entity, timeline, required-component and RBL checks;
- separate actual original-resolution visual inspection record.

`prepare` is not an audit. Exactly one `audit` invocation computes allocation and writes the
authority evidence. Manual inspection is followed by a control-only `finalize` stage; finalize does
not recompute allocation and is not a retry.

## 8. Verdict and immutable boundary

- Complete measurement:
  `D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS`.
- Any input, semantic binding, allocation, or perturbation integrity failure:
  `D368_SEMANTIC_AUTHORITY_OR_ALLOCATION_INTEGRITY_FAIL_STOP`.
- A failure after authoritative measurement evidence is written, or a final visualization/hash-chain
  failure:
  `D368_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`, while preserving the measurement verdict.

In both outcomes:

```text
current_64cap_optimal = null
physics_equivalence = null
collider_count_tipping_causality = null
actual_gpu_contact_execution = null
grasp_feasibility = null
g0a_pass = false
```

Isaac/Kit/PhysX/Warp/CUDA compute execution, recook/decomposition, USD/asset write, q5, physics
step, contact query, target/IK/path change, and material/mass/actuator/physics-setting change are
all zero. Offline Rerun display rendering is the explicitly allowed visualization exception.
D351-D367 and the user-owned D334 sidecar remain immutable. No commit/push is authorized.

## 9. Executed sequence

1. Three read-only static reviews found and repaired pre-run false gates, misleading nearest-hull
   semantics, incomplete artifact hash chaining, and zoom/readability risks.
2. `prepare` ran once and passed `7/7`. It froze Git
   `7c4819632bb193c8fd552372c919f8a107675b41`, all input/dynamic hashes, the installed NVIDIA
   schema facts, the two new variables, the forward-only inventory, and all zero-runtime guards.
3. `audit` ran exactly once with no retry. It finished in `42.26971553196199s`, wrote the
   authoritative Float64 evidence before visualization, and recorded invocation count `1` with
   Isaac/PhysX/q5/physics count `0`.
4. The RRD/RBL/headless PNG and exact 1920x1080 summary were generated after the evidence JSON.
5. Both PNGs were opened at original decoded resolution. The manual visual gate failed for an
   `Unknown timeline` metric panel and remaining label/layout overlap.
6. Control-only `finalize` recomputed the entire evidence/RRD/RBL/PNG/manual hash chain. It did not
   rerun allocation.

## 10. Measured current 64-cap allocation

All `128/128` callback parts loaded with exact witness/topology/prototype lineage: `64 link5 + 64
gripper_link`. All 128 observed parts were within the offline geometry fields `vertices<=64`,
`polygons<=64`, and `vertices_per_polygon<=32`; this does not attest that D362 contact generation
actually ran on GPU.

Certified callback-face carriers were:

| Raw semantic patch | Certified callback faces | Certified carrier parts | Carrier share of body | Max certified-vertex residual |
|---|---:|---:|---:|---:|
| link5 D350 seed-plane patch | 12 | 4 (`027,029,030,031`) | 4/64 = 6.25% | 0.0241181509mm |
| moving inner patch | 40 | 17 | 17/64 = 26.5625% | 0.0789906424mm |
| moving outer negative patch | 36 | 16 | 16/64 = 25% | 0.0789906424mm |

The moving-side classification is especially important as an allocation fact: `16/64` parts carry
both certified inner and outer faces, `1/64` (`part_035`) carries a certified inner face without a
certified outer face, and `47/64` have no certified contact-patch face. This is not, by itself, a
defect, an optimality result, or a cause of the D362 toppling; a convex piece spanning the jaw
thickness can legitimately expose both opposing faces.

Finite-sample nearest-surface diagnostics, which are not contact-support authority, were:

| Raw patch | recurrent/union nearest parts | P95 nearest residual | Max nearest residual |
|---|---:|---:|---:|
| link5 seed-plane | 10/10 | 0.1309419611mm | 0.2803960136mm |
| moving inner | 18/23 | 0.2006238851mm | 0.6693417955mm |
| moving outer | 19/22 | 0.3081875088mm | 0.8007207893mm |

These distance distributions remain diagnostics; no new whole-patch fidelity gate was introduced.
The D350 seed was nearest to `link5 part_031` at `2.5268486e-7mm`. The inherited D354 moving-inner
lineage reproduced exactly `40 faces / 17 parts / 5bb7ad8a...eccab5`. All 8 registered perturbation
controls rejected or distinguished the corrupted input; Qhull substitution differed on `128/128`
parts, and removing the D350 witness component changed the seed distance from effectively zero to
`20.4859540342mm`.

Measurement verdict:
`D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS`.

## 11. Visualization result

Automated Rerun validation passed: SDK/CLI `0.34.1`, footer, exact entity/timeline/component
inventory, external RBL verification, and headless rendering. The RRD/RBL/Rerun-PNG/summary-PNG
SHA-256 values were respectively `f66a9fe4...5af0`, `9b4db461...b8e`,
`fdc88cf2...c9e2`, and `3ef14771...1796`.

Human inspection nevertheless failed two preregistered checks:

- the Rerun `allocation counts and geometry budget` panel displayed `Unknown timeline` and no
  values; the screenshot also contained a message-proxy `Operation not permitted` notification;
- moving-jaw marker labels overlapped, and the 1920x1080 sheet retained upper-axis/lower-title
  crowding.

Therefore `visualization_pass=false` and overall `pass=false`, with completion verdict
`D368_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`. This does not override the earlier
measurement verdict. Manual-inspection and completion SHA-256 are
`21068831...ab17` and `d56a5c20...4dd6`.

## 12. Interpretation and next authorization boundary

- D368 establishes a reproducible semantic/budget inventory for the **current 64-cap reference
  candidate**. It does not show that 64 is best, that fewer hulls are inadequate, or that hull count
  caused toppling.
- The present candidate uses 4 fixed-contact carriers and 17 moving-inner carriers; 16 of those
  moving carriers also expose the paired outer plane. A future candidate comparison must report the
  same face-carrier, nearest-distance, whole-carrier-budget, GPU-geometry, and physical-contact
  metrics rather than compare hull count alone.
- D368 output is frozen. No audit rerun, overwrite, recook, candidate generation, Isaac/physics, q5,
  target/IK/path, or settings change is authorized.
- The narrow next recommendation is a forward-only D369 observability-only repair that requires a
  separate approval, reads immutable D368 evidence, and fixes the Rerun metric timeline plus text
  layout. Collider Pareto candidate generation/comparison and any pose/physics test remain later,
  separately approval-required cases.
