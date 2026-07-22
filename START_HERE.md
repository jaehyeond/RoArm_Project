# START_HERE.md

Last updated: 2026-07-22 KST. D374 completed an offline-only provenance audit and
visualization of the frozen D373 fail-stop. It did not rerun Isaac/PhysX or promote P34 live
identity, physics, tipping, or grasp conclusions.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED and frozen OPEN is `1.5413rad`.
- D362 remains the physical authority: the current A64 path pushed the cylinder over rather
  than holding it. D372-D374 did not rerun or supersede that physical result.
- D368 established that `64 link5 + 64 gripper_link` is a current 64-cap reference candidate,
  not an optimum. `maxConvexHulls=32` is an automatic-decomposition schema default, not a
  manual compound target count or engine hard limit.
- D372 built the professor's task-local semantic P34 candidate offline:
  - link5 `16`: body box-shaped convex Mesh `1`, connector/pivot support `3`, fixed-jaw contact
    pieces `10`, fixed-jaw backbones `2`.
  - gripper_link `18`: moving support `4`, moving-jaw contact pieces `12`, moving-jaw
    backbones `2`.
  - total `34`, a design-count reduction from A64 total `128`, not a speed/physics/optimality
    result.
- D373 materialized P34 once and obtained exact limited raw evidence:
  - active path/owner/count `16+18`, active A64 `0`, disabled known legacy `2`;
  - direct Float32 geometry payload `34/34`, live instance/prototype authored streams
    `34/34`, callback protocol `68/68`;
  - authored MassAPI delta `0.0` for link5/gripper_link.
- D373 still stopped at `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP` because its certification
  pipeline had four contract defects: decimal-vs-Float32 comparator drift, unsupported dynamic
  rigid-body instance proxies in property query, default traversal omitting instance proxies,
  and returncode-only supervisor PASS.
- D374 proved those four causes from immutable evidence and created the previously missing
  failure/collider visualizations. Final D374 verdict:
  `D374_D373_FAIL_STOP_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS`.
- D374 PASS means provenance/observability completion only. D373 remains FAIL_STOP.
- Full P34 live identity, authored↔callback surface/bounds/original-polygon topology-volume,
  live property mass/COM/inertia, physical equivalence/speed, tipping causality, and grasp
  feasibility remain `null`; `g0a_pass=false`.

## D374 Verified Results

- Preregistration and frozen input/env checks PASS; failure-capable controls `4/4` PASS.
- Actual offline audit/retry `1/0`; headless Rerun capture `1`.
- D343 typed value `0.00009999999747378752m` matches D373 direct/live numeric values
  `34/34`. Exact bits `0x38d1b717` are inherited D343 authority; D373 did not persist
  `typeName` or bits, and D374 typed-scalar retest count is `0`.
- D373 property query results remain link5/gripper_link `ERROR_PARSING(5)`. Their empty path,
  zero mass and zero volume are error sentinels, not measurements.
- D373 default comparator P34 population `0` is a traversal blind spot: proxy-aware live
  inventory and callback parts are both `34`.
- Old supervisor `pass=true`; raw/preclose worker protocol `false/false`; preclose↔raw hash
  exact. The hash-bound repaired effective PASS is `false`.
- Callback witness protocol/hash `68/68`; instance↔prototype original-polygon payload
  `34/34 exact`; instance vertices/original polygons `314/262`, maxima per part `13/17`.
- Exact 1920×1080 boards `4`, save-only RRD/RBL, strict validation, and original-resolution
  inspection PASS. Logical Viewer request was 1920×1080; DPR 2 physical PNG is 3840×2160.
- Scope counters Isaac/PhysX/USD write/physics/q5/contact/cylinder/target-IK-path/collider
  regeneration/decomposition sweep are all `0`.
- Immutable D373 inventory and the user-owned D334 sidecar are unchanged.

## Active Case / Authorization Boundary

- No case is currently approved. D374 is complete and frozen; do not retry or overwrite it.
- Recommended next minimum is a separate forward-only live repair case, not yet approved:
  `D375 [p34_live_asset_identity_contract_repair]`.
  - Inherit D343 typed-Float32 authority without decimal comparator regression.
  - Inherit D374 proxy-aware/direct-layer traversal and hash-bound supervisor authority.
  - Keep dynamic articulation rigid-body owners non-instance while validating P34 collider
    geometry/property identity in one actual worker with no retry.
  - Maintain physics step/q5/contact/cylinder/target-IK-path changes at `0`.
- A repaired live-identity PASS is required before any A64, link5-only P34,
  gripper-only P34, or both-P34 cylinder physics comparison.
- Physical comparison, center-height/wrist pose repair, target/IK/path, material/mass/actuator/
  physics changes, settle/hold/lift, ten-trial, G0b, RL/PPO/VLA each require separate approval.

## Frozen Boundaries / Do Not Repeat

- Do not call P34's 34 parts a mathematical, global, or performance optimum.
- Do not claim D373 directly observed Float32 bits; its raw JSON recorded numeric values only.
- Do not repeat decimal `0.0001` versus typed Float32 with `1e-12m` after D343.
- Do not query dynamic articulation bodies through whole-robot instance proxies.
- Default stage traversal zero is not collider-absence evidence when proxies are omitted.
- Nonzero property-query error rows contain sentinels, not mass/volume measurements.
- Process return `0` proves orderly cleanup only; it cannot override worker/preclose verdicts.
- Callback protocol PASS alone is not full surface/property identity PASS.
- Display fan triangles and Rerun Float32 copies are inspection-only; original callback polygon
  JSON/hash remains authority.
- Do not combine link5 and gripper_link owner-local shapes into a fake world/q5 pose.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install, commit,
  push, signal, new live worker, or physical comparison is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D373-D374; ledger tail
2. `claudedocs/session_20260722_grasp_g0a_d374_d373_fail_stop_provenance_and_failure_visualization.md`
3. D374 preregistration, failure provenance evidence, live repair contract, manual inspection,
   and completion summary
4. `claudedocs/session_20260722_grasp_g0a_d373_p34_live_asset_identity_preflight.md`
   only when tracing upstream failure/raw callback evidence
5. D372 completion/geometry only when tracing the frozen P34 design source
6. D339 valid property-query evidence and version-matched Omni Physics 107.3 docs only when
   designing a later approved live repair
7. D362 physical trace only after a physical comparison is separately approved

## Git

- Session boot verified the user's D373 push:
  `HEAD == origin/master == 548d3517f5a7936529646c5d8b0009427eb936ab`, subject
  `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`, with a clean worktree before D374 edits.
- Current worktree contains only the uncommitted D374 implementation, evidence, visualization,
  and state-doc updates described here.
- The five D374 PNGs exist at their exact completion-summary paths but are hidden from normal
  `git status` by `.gitignore:110` (`*.png`); force-add needs separate user authorization.
- Commit/push was not authorized and was not performed.
