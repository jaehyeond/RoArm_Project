# Session 2026-07-27 — D398 resume verification + SDF re-evaluation (read-only, no experiment)

> Append-only session log. This session ran NO experiment: START_HERE said no
> approved next case, and the user asked for a current-state / what-changed /
> critical-assessment briefing while consulting NVIDIA official docs. No code,
> USD, Isaac/PhysX, physics, commit, or push was performed.

## Boot / verification performed

- Read `START_HERE.md` (D398 frozen), `DECISIONS.md` D372–D398 in full,
  `EXPERIMENT_LEDGER.md` tail, `git log`, and grepped the installed
  `PhysxSchema 107.3.26` `schema.usda` directly.
- Cross-checked the version-matched NVIDIA doc *Omni Physics 107.3 — Colliders*
  (collision.html) via WebFetch.

## What the user is trying to do (current state)

- Fix "A64 gripper collider tips the object instead of grasping (D362)" by
  redesigning the gripper (link5 + gripper_link) collider into a more accurate,
  lower-count representation. The new collider is neither materialized nor
  physics-tested yet.
- Object model changed: historical `34x90mm/0.72kg` (D362) → actual nominal
  `29x50mm`, mass/friction unmeasured (`START_HERE.md:11-14`, commit `b880bc8`).
- Candidates: A64 (`64+64=128`, reference only); P34 (`16+18=34`, manual, live/
  cooked-identity NOT passed). Physics entry blocked; `g0a_pass=false`.

## What changed since D371 (D372→D398)

- (a) P34 actually tried → live identity FAIL (D373). Fixed instancing/
  StageCache/typed-scalar plumbing (D374–D378).
- (b) authored↔cooked shape identity FAIL: D379 surface max `0.684mm` > `0.1mm`
  gate; D380 localized cause = cook inward vertex erosion (cooked ⊂ authored,
  loss `341mm^3/3.9%`), NOT `hullVertexLimit=64` overflow (max authored `44<64`).
- (c) D384–D398 low-count convex-partition existence rabbit hole under the
  project gates `<=12 vertices/child, source<=64, total<128, identity 0.1mm/0.5%`:
  D384 authored-only split `268/558` >> 128; D385–D388 partial covers/graph
  disconnection; D389–D395 affine-rank/Float32/canonical-order/micro-volume
  numeric analysis; D396 rejected D388 (overlap `6404x/2413x` over gate);
  D397 shared-seam BSP completed only `2/8` parents; D398 localized the greedy
  dead-end and proved `14/14` selected ancestors had an unselected admissible
  option (greedy dead-end ≠ proven global impossibility; completion feasibility
  still `null`). Rerun presentation FAILed repeatedly (label overlap etc.).

## NVIDIA verification (installed primary source + version-matched doc)

Category separation (schema default ≠ GPU hard limit ≠ project gate):

- `physxConvexHullCollision:hullVertexLimit = 64` — schema default
  (`schema.usda:858`); convexDecomposition variant `:886`.
- `physxConvexDecompositionCollision:maxConvexHulls = 32` — schema default
  (`schema.usda:895`).
- UI ranges `8..64` / `1..2048` = property-editor authoring range, not engine
  limit.
- GPU-compatible convex limit `64` vertices/polygons per hull = GPU hard limit.
- Project gates `<=12 vertices/child, source<=64, total<128, 0.1mm/0.5% identity`
  = PROJECT-AUTHORED, not NVIDIA defaults/optimum/GPU limits (D385 impl, D397 §8).

KEY FINDING — the installed schema already provides the concave-capable path:

- `class "PhysxSDFMeshCollisionAPI"` (`schema.usda:1043`), `sdfResolution = 256`
  (`:1049`), `sdf` token = "SDF triangle mesh approximation".
- `physxCollisionCustomGeometry` token = exact cone/cylinder without convex
  approximation → applies directly to the `29x50mm` cylinder TARGET.
- `class "PhysxMeshMergeCollisionAPI"` (`:1188`).
- Version-matched *Omni Physics 107.3 Colliders* approximation table:
  none/meshSimplification = concave but STATIC only; convexHull/
  convexDecomposition/boundingSphere/boundingCube = dynamic but CONVEX only;
  **SDF = dynamic AND concave** ("dynamic and kinematic rigid bodies with
  high-detail mesh based colliders"). SDF is the ONLY option that is both
  dynamic-capable and concave-preserving.

## Critical assessment

Good: extremely rigorous provenance/no-overclaim discipline; correctly separates
geometry-FAIL vs presentation-FAIL vs physics-null; no silent gate relaxation;
correctly distinguishes schema defaults from project gates.

Concerns:
1. Zero physics since D362 — 27 offline decisions (D373–D398) never re-tested
   whether the collider difference changes the grasp/tip outcome.
2. Self-imposed gates may make the problem unsatisfiable: `<=12 vertices/child`
   is ~5x stricter than the schema default 64, and "authored↔cooked 0.1mm
   identity" is arguably impossible for convex decomposition of a concave mesh
   (D380 already measured 0.68mm inward erosion).
3. Rerun presentation contract (HiDPI, label overlap, glyph, JSON serialization,
   rrd print verbosity) has consumed a large fraction of attempts/decisions.
4. SDF was mentioned once (D398 §8) but effectively down-weighted ("no reason to
   default"). That judgment predates the D384–D398 cost blow-up; SDF structurally
   bypasses the exact wall (few-convex-pieces vs concave gripper) that has
   blocked 15 decisions.

## Recommendation (user decision pending — nothing executed)

- Preferred: approve a NEW case to re-evaluate the collider REPRESENTATION via
  SDF mesh collider (gripper concavity preserved) + custom-geometry exact
  cylinder target, whose goal is to FIRST re-measure physics (does D362 tipping
  improve?) rather than to perfect authored↔cooked identity. SDF needs its own
  validation (memory/perf, thin-feature contact, articulation-link caveats) — not
  an uncritical swap.
- Alternative: stay on START_HERE's D399 (label deconfliction only) → later
  bounded backtracking search, but first justify that the convex-partition
  problem is worth solving physically.

## Authorization boundary (unchanged)

No experiment auto-run. D389–D398 paths frozen; D334 sidecar untouched; no USD/
Isaac/PhysX/cylinder/physics/q5/contact/commit/push. Any SDF direction requires a
new preregistration (new variable, gates, output paths) shown for approval first.
Did NOT run `/half-clone` (HARD RULE #11).
