# START_HERE.md

Last updated: 2026-07-22 KST. D373 attempted the first live USD/PhysX identity preflight
for the D372 professor-style P34 collider, then stopped fail-closed before normal analysis and
visualization. The P34 geometry itself was not classified PASS or FAIL, and no physical grasp
test ran.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius/diameter/height are
  `0.017/0.034/0.090m = 17/34/90mm`; `q5=0` CLOSED and frozen OPEN is `1.5413rad`.
- D362 remains the physical authority: the current A64 path pushed the cylinder over rather
  than holding it. D372-D373 did not rerun or supersede that physical result.
- D368 established that `64 link5 + 64 gripper_link` is a current 64-cap reference candidate,
  not an optimum. `maxConvexHulls=32` is an automatic-decomposition schema default, not a
  manual compound target count or engine hard limit.
- D372 built the professor's task-local semantic P34 candidate offline:
  - link5 `16`: body box-shaped Mesh `1`, connector/pivot support `3`, fixed-jaw contact pieces
    `10`, fixed-jaw backbones `2`.
  - gripper_link `18`: moving support `4`, moving-jaw contact pieces `12`, moving-jaw backbones
    `2`.
  - total `34`, a design count reduction from A64 total `128`, not a speed/physics/optimality
    result.
- D373 materialized that frozen D372 Float64 geometry once into a forward-only derivative USD
  and launched exactly one Isaac/PhysX worker with no retry.
- D373 raw acquisition succeeded in these limited senses:
  - active collider paths/owners/counts were exactly link5 `16` + gripper_link `18`; active A64
    `0`; only the two known legacy mesh colliders remained disabled.
  - authored Float32 point/count/index/aggregate payload was exact `34/34` in direct readback,
    and live instance/prototype authored streams were exact `34/34`.
  - PhysX callback protocol was valid `68/68` channels (`34` prototype + `34` instance), with
    one convex result per channel and no worker exception.
  - authored MassAPI mass/COM/inertia/principal-axes deltas were `0.0` for both bodies.
- D373 did **not** complete full live identity:
  - all 34 direct and 34 live rows tripped only `min_thickness_frozen`: schema Float32 readback
    was `0.00009999999747378752m`, differing from decimal `0.0001m` by
    `2.5262124884e-12m`, beyond the incorrectly tight `1e-12m` comparison. This is a regression
    of the same D342 failure already repaired by D343's exact typed-Float32 contract.
  - both property queries returned `PhysxPropertyQueryResult.ERROR_PARSING`. D373 made the
    whole `/World/Robot` instanceable, while installed PhysX explicitly logged that dynamic
    RigidBodyAPI on the resulting instance proxies was unsupported for link5/gripper_link.
    Version-matched Omni Physics 107.3 also states articulation links may not be instanced.
  - the D345 canonical population check used default stage traversal and omitted instance
    proxies, so it falsely saw A64/P34 path counts `0/0`; the exact 34-part live traversal and
    68 callbacks disprove asset absence, but the outside-subtree whole-stage proof is incomplete.
  - the supervisor recorded `pass=true` from process return `0`, while the authoritative worker
    summary recorded `worker_protocol_pass=false`; orderly shutdown was mistaken for contract
    success by the supervisor layer.
- Frozen D373 verdict: `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`.
- Full authored↔callback surface/bounds/original-polygon topology-volume, callback↔property
  volume, live property mass/COM/inertia, full P34 identity, physical equivalence, tipping cause,
  and grasp feasibility remain `null`; `g0a_pass=false`.

## D373 Execution and Stop Boundary

- Preregistration: `13/13` PASS; new variable
  `[p34_live_asset_materialization_and_binding_v1]`.
- Worker/supervisor: actual worker/retry `1/0`, elapsed `7.6713462870s`, watchdog `900s`,
  timeout/signal `false/false`, process return `0`.
- Scope counters: derivative asset/materialized collider authors/callback/property query/stage
  attach-detach were `1/34/68/2/1-1` as approved.
- SimulationContext/reset/timeline play/commit/physics step/public forward/q5/contact/cylinder/
  target-IK-path/automatic decomposition sweep/inherited physical-setting change/Isaac Hydra
  render/app update pump were all `0`.
- The frozen controller requires both supervisor and worker protocol PASS before offline identity
  classification. Therefore normal analyze/finalize were not invoked.
- Exact 1920x1080 board, save-only RRD/RBL and original-resolution inspection are
  `not_run/null_due_upstream_fail_stop`; no unregistered bypass visualization was synthesized.
- The D373 derivative and all attempt1 evidence are frozen. Do not retry, overwrite, run its
  normal analyze/finalize, or treat callback acquisition alone as full identity PASS.

## Current Authorization Boundary

- No next case is approved.
- Recommended next minimum: separate forward-only offline-only
  `D374 [d373_fail_stop_provenance_and_failure_visualization]`.
  - Read only immutable D373 raw/USD evidence; no Isaac/PhysX worker.
  - Formalize typed Float32 scalar authority, instance-proxy/canonical traversal scope, property
    query incompatibility, and worker-vs-supervisor authority.
  - Produce a failure-focused exact 1920x1080 board plus save-only RRD/RBL, clearly labeling
    full identity and all physics conclusions as `null`.
- Only after D374 may a separate one-worker/no-retry live repair case be proposed. A repaired
  live-identity PASS is required before any A64/link5-only P34/gripper-only P34/both-P34 physical
  comparison.
- Target/IK/path, center-height/wrist pose repair, materials/mass/actuator/physics settings,
  settle/hold/lift, ten-trial, G0b, RL/PPO/VLA remain separate approval boundaries.

## Frozen Boundaries / Do Not Repeat

- Do not interpret P34's 34 parts as a mathematical or global optimum.
- Do not reinterpret the Float32 thickness gate false failure as geometry corruption, and do not
  repeat D342/D373 decimal-vs-Float32 tolerance drift after D343 established exact typed bits.
- Do not query a dynamic articulation link through a whole-robot instance proxy; collision
  geometry instancing and rigid-body-owner instancing are different contracts.
- Instanceable-stage population audits must explicitly traverse instance proxies or audit the
  direct authored layer; a default traversal count of zero is not absence evidence.
- A worker process return code is operational evidence only; it cannot override an internal
  worker verdict.
- Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar; D373 confirmed all three files
  unchanged.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install, commit,
  push, signal, physical comparison, or new worker is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D372-D373; ledger tail
2. `claudedocs/session_20260722_grasp_g0a_d373_p34_live_asset_identity_preflight.md`
3. D373 preregistration, worker supervisor/raw summary/preclose sentinel, fail-stop attestation,
   stdout/stderr, and callback witnesses
4. D372 attempt2 geometry/evidence/completion for the frozen P34 source
5. D339 valid property-query audit and Omni Physics 107.3 rigid-body/property-query documents
   only when designing a later approved repair
6. D362 physical trace only when a later physical comparison is explicitly approved

## Git

- D373 boot verified the user's D371-D372 push:
  `HEAD == origin/master == 5214721e91bd23b224998cba2b13a1f76294edad`, subject
  `D371-372변경`, with a clean worktree before D373 edits.
- Current worktree contains only the uncommitted D373 implementation, evidence, and state-doc
  updates described here. Commit/push is not authorized and was not performed.
