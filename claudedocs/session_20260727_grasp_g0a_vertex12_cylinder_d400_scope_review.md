# Session 2026-07-27 (3rd) — max-12 provenance, cylinder representation, and D400 scope review

> Append-only documentation/audit session. The user authorized state-document
> correction, not a D400 experiment. No code, asset, USD, Isaac/Kit/PhysX/Warp
> launch, cylinder creation, controlled physics step, q5 sample, contact query,
> hardware action, process signal, dependency change, commit, or push occurred.
> The session-progress rule is satisfied by explicit justification: this turn
> corrected the authorization boundary and causal variable count before an
> unapproved case; running a perturbation would have exceeded the user's scope.

## 1. Boot and repository state

- Read `AGENTS.md`, `START_HERE.md`, all of `DECISIONS.md` and
  `EXPERIMENT_LEDGER.md` (with focused audits of D362 and D385/D397/D398),
  the referenced D385/D397/D398/SDF-recon session documents, and the original
  D362 code/trace.
- Read the installed Isaac Lab 2.3.0 spawner and SimulationContext source,
  PhysX 107.3.26 bindings/schema/demo, and NVIDIA PhysX 5.6.1 documentation.
  Treat the installed plugin's exact PhysX 5.6.1 engine-version match as
  `[INFERENCE]`, not package-metadata proof.
- At audit start:
  `HEAD == origin/master == 4c88865bdd4ac82f034253320cb3e46f9770a46d`
  (`현재 D398분석`).
- The worktree already contained the prior SDF-recon state changes:
  `START_HERE.md`, `claudedocs/EXPERIMENT_LEDGER.md`, and the untracked
  `claudedocs/session_20260727_grasp_g0a_sdf_case_recon_option_a_briefing.md`.
  They were preserved and updated only where this user-authorized state
  correction required it.

## 2. User decisions now recorded

1. The product direction is nominal `29x50mm`, not another historical
   `34x90mm` product-physics campaign. Actual mass will be measured on arrival;
   friction remains a placeholder until a later dedicated measurement or
   sensitivity case.
2. Keep `D399` reserved for the already-defined D398 Rerun label-deconfliction
   repair. Use `D400` for the SDF direction.
3. Include cylinder-representation provenance, but do not overstate the
   offline result as runtime-exact authority.

These decisions select a direction. They do not approve a D400 preregistration
or runtime execution.

## 3. What `max-12` actually means

### 3.1 Provenance

- D385 introduced two project variables:
  `source_hull_semantic_thin_layer_profile_cell_partition_v1` and
  `source_child_max12_vertex_budget_v1`
  (`session_20260725_...d385...md:16-20`; D385 script `:123-131`).
- The `12` counts unique Float32-authored vertices in each manually designed
  source child. It applied only to the eight failed general 3-D P34 source
  parents, not every RoArm collider.
- D385 explicitly registered child vertices `<=12`, source children `<=64`,
  total parts `<128`, surface error `<=0.1mm`, topology-volume relative error
  `<=0.5%`, and zero positive-volume sibling overlap as project design gates
  (`D385 session:46-63`).
- D397 inherited `12` as a frozen leaf-termination and geometry gate. D398 did
  not prove that `12` is necessary, optimal, or impossible; it proved only
  that the selected greedy history was not locally forced and that completion
  feasibility remained `null`.

### 3.2 Category separation

| Category | Value | Meaning |
|---|---:|---|
| D385/D397 project budget | `12` | authored vertices per manually designed source child |
| installed PhysxSchema default | `hullVertexLimit=64` | cook target for one convex hull |
| installed PhysxSchema default | `maxConvexHulls=32` | automatic decomposition output cap |
| Kit property UI ranges | `8..64`, `1..2048` | authoring ranges, not engine hard limits |
| generic PhysX convex descriptor | up to `255` | CPU/general SDK validation range |
| GPU contact compatibility | `64` vertices, `64` polygons, `32` vertices/face | cooked convex GPU-contact count conditions; GPU data and other checks still required |

Installed evidence:

- PhysxSchema 107.3.26 `schema.usda:852-865,880-902`
- Kit property database `database.py:954-958`
- PhysX 5.6.1 `PxConvexMeshDesc.h:124-190`
- PhysX 5.6.1 *GPU Simulation*, lines 74-89

Therefore `12` may be relaxed only as a separately registered convex/BSP
variable; changing it is not an NVIDIA violation. It must not be inherited by
the SDF path. SDF uses a triangle mesh plus a distance grid rather than a set
of max-12 convex children.

## 4. D362 cylinder-representation audit

### 4.1 Confirmed authored/runtime-input facts

- Isaac Lab `CylinderCfg` dispatches to `spawn_cylinder`, which calls
  `_spawn_geom_from_prim_type(..., "Cylinder", ...)`
  (`shapes_cfg.py:74-88`; `shapes.py:108-144`).
- The helper path is named `{root}/geometry/mesh`, but the actual prim type is
  the supplied `Cylinder`, not `UsdGeom.Mesh` (`shapes.py:249-280`).
- Isaac Lab 2.3.0 writes the legacy setting
  `/physics/collisionCylinderCustomGeometry=False`
  (`simulation_context.py:672-676`).
- Installed omni.physx 107.3.26 exposes the current setting
  `/physics/collisionApproximateCylinders` and its defaults path
  (`_physx.pyi:4069-4070`). The bundled `AnalyticCylinderDemo.py:6-11`
  states that this setting is off by default and that off uses an analytic
  shape cylinder while on uses a convex-mesh approximation.
- Static text/binary inspection found the legacy key only in the Isaac Lab
  setter and asset-validator compatibility material, while the current key is
  present in the PhysX binding/plugin. This strongly supports, but does not
  runtime-prove, the legacy key being unconsumed by the current engine.

### 4.2 Authority boundary

The safe verdict is:

`D362_ANALYTIC_CYLINDER_HIGH_CONFIDENCE_INFERENCE_RUNTIME_GEOMETRY_TYPE_PENDING`

It is not:

`D362_EXACT_ANALYTIC_CYLINDER_CONFIRMED`

Reasons:

1. D362 did not record the actual runtime value of
   `/physics/collisionApproximateCylinders` or its defaults path.
2. D362 did not record the underlying PhysX geometry enum/implementation.
   The installed plugin contains evidence for both legacy custom-cylinder and
   newer convex-core-cylinder paths.
3. D362's working 500-step contact report proves that the contact-report
   pipeline worked. It does not distinguish analytic cylinder from a convex
   approximation; both can report contacts.
4. PhysX 5.6.1 warns that custom geometry interacting with SDF falls back to
   TriangleMesh collision, potentially degrading quality and performance.

A future cylinder-bearing case must log the current/default carb settings and
the actual PhysX geometry type before using cylinder representation as a
causal invariant.

## 5. Nominal 29x50 geometry and current-pose implication

- D362 first positive moving-jaw contact:
  global/closure step `232/31`, point z `78.0104175mm`.
- D362 first positive link5 contact:
  global/closure step `246/45`, point z `49.4077131mm`.
- A nominal 50mm cylinder on the historical table
  `z=-12.117mm` has a nominal top near `37.883mm`.
- The historical contact samples are therefore about `40.13mm` and `11.53mm`
  above the new nominal top.

This strongly motivates a new height/radial pose derivation. It does not prove
that both entire jaw surfaces miss the product: two old contact samples are not
the minimum-z bounds of the full jaw surfaces. The decision authority must be
a new zero-step surface/clearance scan on the nominal geometry.

## 6. Corrected variable accounting

Resolving cylinder representation does not collapse the remaining changes into
two variables. Relative to D362, the direct-product program potentially changes:

1. product diameter/height (`34x90 -> 29x50mm`);
2. target/arm height and radial placement;
3. mass when the real value is measured;
4. moving-jaw collision representation (`A64 -> SDF`).

Friction may remain numerically frozen for an initial relative comparison, but
it remains a placeholder and cannot support a real-product grasp/hold claim.
Bundling dimensions, mass, and pose under one label such as `object_rebase`
would hide causal variables and violate the Variable Ladder's purpose.

## 7. Corrected forward-only ladder

### D399 — reserved, optional presentation repair

Keep the existing name
`d398_rerun_label_deconfliction_observability_repair`. It is not a prerequisite
for collider or physics work unless an immediately clean interactive D398 Rerun
presentation is needed.

### D400 — SDF live/cook/articulation preflight only

Proposed single new variable:

`gripper_link_collision_representation_a64_to_sdf_res256_v1`

Frozen:

- `link5=A64`
- `gripper_link` source mesh and robot articulation/actuator/mass properties
- SDF resolution `256`, remeshing off, no sweep
- no product cylinder, target/IK/path change, q5 sample, contact query, or
  controlled physics step

Required observations:

- direct pxr SDF API application and exact `approximation="sdf"` readback
- active owner/path and articulation-link attachment
- cook/parser/fallback warnings
- live bounds, source-surface/void/open-jaw clearance diagnostics
- robot mass/COM/inertia unchanged
- one worker, no retry, bounded watchdog

This can establish materialization/cook feasibility. With zero contact and zero
physics it cannot establish SDF collision response on the articulation.
`link5` SDF, resolution `512`, remeshing, or mesh repair are not D400 variables.

### D401 — SDF articulation collision-response positive control

Use one separately preregistered, bounded, non-product, known-box
contact-positive probe to verify that the frozen D400 `gripper_link` SDF
actually participates in contact and produces physical response while attached
to the articulation. Compare it against the same A64 probe contract, identify
the exact contacting owner pair, and keep q5 and target/IK/path frozen. Do not
use a cylinder in D401, so cylinder representation remains isolated to D402.
This is a representation plumbing gate, not product-cylinder or grasp evidence.

D400's zero-contact live/cook PASS alone must not authorize product physics.
If D401 or a later physics case uses timeline PLAY, it must inherit D367's
explicit commit bridge and supervisor/pre-close authority; a post-close marker
must not be promoted into the sole PASS authority.

### D402 — nominal 29x50 zero-step geometry and pose localization

Proposed two variables:

1. nominal product geometry `29x50mm`;
2. deterministic height/radial pose derivation, with wrist orientation frozen.

Use the same proposed pose for A64 and the frozen D400 SDF representation.
Record cylinder current/default carb settings and actual geometry type. Run no
closure physics or contact verdict. Mass is not required for this zero-step
geometry decision.

### Mass pin before dynamics

Measure actual product mass before product-representative physics. Record the
uniform-density COM/inertia assumption separately if direct measurements are
unavailable. Keep placeholder friction explicit.

### D403 — new-product A64 physics baseline

Use the frozen D402 geometry/pose, measured mass, A64, and one registered
physics/contact contract. Re-derive contact capacity here, immediately before
the first contact-generating run. Its verdict is valid only under the frozen
placeholder friction.

### D404 — collider representation causal comparison

Inherit D403 bit-for-bit and change only
`gripper_link A64 -> frozen D400 SDF`. This restores a clean A64-versus-SDF
comparison on the new product even though D362's historical numbers do not
transfer.

### Superseded wording

- The earlier BACKLOG/ledger phrase that SDF is the globally “only
  dynamic+concave option” is too broad. The version-matched Omni Physics
  compatibility table establishes SDF as one documented dynamic-rigid-body
  concavity-preserving path in that table; it does not prove global uniqueness
  across every PhysX representation or compound strategy.
- The earlier phrase that the convex-partition gates are “likely
  unsatisfiable” is also stronger than the evidence. D398 localized a dead end
  of the selected greedy, max-12 construction history. It did not prove global
  infeasibility for convex decomposition, another branch search, or a different
  registered budget.

## 8. Documentation decision

- `START_HERE.md`, `EXPERIMENT_LEDGER.md`, and the existing SDF `BACKLOG.md`
  entry are updated/clarified to this corrected scope.
- `DECISIONS.md` receives append-only `D398-F1`. It defers only D398's old
  “label repair first” ordering after the user's SDF/D400 direction choice;
  D398's numeric/operational verdict remains frozen and D399 remains an
  unexecuted reserved presentation case. No D399 or D400 decision/result is
  fabricated.

## 9. Authorization boundary

- No D399-D404 preregistration/execution has been approved by this
  documentation update.
- Next concrete action requires explicit approval to write the D400
  preregistration. Runtime execution then requires a separate approval.
- D389-D398 outputs remain frozen. Do not overwrite or materialize D397/D398
  partial diagnostic geometry.
- Do not modify
  `claudedocs/lab_meeting/20260715/d334_collision_table/`.

## Official sources

- NVIDIA PhysX 5.6.1, *GPU Simulation*:
  https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html
- NVIDIA PhysX 5.6.1, `PxConvexMeshDesc.h`:
  https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/program_listing_file_include_cooking_PxConvexMeshDesc.h.html
- NVIDIA Omni Physics 107.3, *Colliders*:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html
- NVIDIA Omni Physics 107.3, *Collision Behavior Guide*:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/collision_guide.html
