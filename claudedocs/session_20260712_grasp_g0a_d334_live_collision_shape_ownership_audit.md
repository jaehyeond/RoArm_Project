# Session 2026-07-12 - D334: frozen-pose live collision shape / ownership audit

Status before runtime: `D334_PRE_REGISTERED_RUNTIME_PENDING`

이번 case의 신규 변수: `[]` (물리 변수 0개 — audit only)
- 신규 방법 변수(물리 아님): live rigid-body collider enumeration
  (`PhysxPropertyQuery.QUERY_RIGID_BODY_WITH_COLLIDERS`) + per-collider cook
  parity gate + D333 step-0 exact replay parity gate.

This section is written before the D334 run. Runtime results must be appended
below without changing these gates.

## Research question

D333 confirmed that at the frozen clean final pose the cylinder is disturbed
from step 0 with a `gripper_link` rigid-body force attribution, while the D332
link5 mirror-recooked hull overlap remains an unresolved offline artifact.
D334 asks: **which live collision shape (owned by which live rigid body)
geometrically explains the recorded contact — a cooked-proxy artifact, the
actual tool geometry, or neither?**

This run does not decide G0a PASS, ladder promotion, collision repair, target
repair, or the D330 swept reattribution. It only routes the next repair choice.

## Frozen comparison contract

- Sources: `claudedocs/runtime_logs/grasp_track/g0a_d333/g0a_d333_sole_support_static_summary.json`
  (+ its D332 source chain), `d333_teleport_settle_trace.csv` row 0,
  `d333_contact_baseline_trace.csv`.
- Seed `33201`, 1 env, `dt=0.005s`, cylinder D34xH90 at `[0.300,0.000,0.032883]`,
  mass placeholder `0.72kg`, friction `1.5/1.2`, canonical D325 command,
  HOME reset, open gripper — all byte-identical to D333 (module reuse).
- Robot USD and URDF SHA-256 must equal the D333 artifact values; mismatch
  fails before scene creation.
- Scene = D333 sole-support scene verbatim: exact global-ground collider
  disabled pre-PLAY, TapTable sole support, cylinder-owned ContactSensor with
  the same four filters, reporter/sleep thresholds 0.

## Audit protocol (pre-registered)

Physics steps: **200 + 1 — every step is a replay of a D333-recorded step.**
The D333 CSVs record no per-joint state, so the recorded post-step-0 pose can
only be reproduced by replaying D333's exact procedure: the 200-step robot-free
HOME baseline (its hard gate re-applied), the exact-state teleport, and one
target physics step. No new command, no new pose family, no settle beyond the
replayed step 0.

1. **Contracts** (reuse D333 code): frozen invariant contract, stage contract,
   sensor contract. Any failure -> `D334_G0A_AUDIT_CONTRACT_FAIL_STOP`.
2. **Live collider inventory** (pose-independent, done before teleport):
   - `PhysxPropertyQuery.query_prim(QUERY_RIGID_BODY_WITH_COLLIDERS)` on the
     live `/World/envs/env_0/Robot/link5` and `.../Robot/gripper_link` rigid
     bodies only (ownership scan beyond these two bodies is forbidden).
     Record per collider: USD path (via `intToSdfPath`), local pos/rot,
     local AABB, **volume**.
   - USD instance-proxy traversal of the same two subtrees: every
     `UsdPhysics.CollisionAPI` prim, its enabled state, `MeshCollisionAPI`
     approximation, source mesh topology counts, nearest ancestor with
     `RigidBodyAPI`.
   - **Ownership parity gate**: the PhysX-side collider path set must equal the
     USD-side enabled-collision path set per body. A mismatch is not a failure
     — it is recorded evidence of the D332 owner-mismatch hypothesis.
   - Property-query wait must not pump untracked physics: joint/object state
     is snapshotted before and re-verified after every query
     (`max |delta| <= 1e-12`); a violated guard re-teleports and is recorded.
3. **Cook evidence per collider** (both bodies):
   - Attempt `request_convex_collision_representation` **directly on the live
     prim ids** (both the API-carrying Xform proxy and the child mesh),
     synchronously. Record result codes — this converts D332's
     "not supported" assumption into recorded evidence.
   - If direct live cook fails: mirror-cook the exact stage-extracted mesh
     (D332 method, per collider, in owner-body local frame).
   - **Cook parity gate** (non-AABB primary): `|scipy ConvexHull(cooked).volume
     - live PhysX collider volume| / live volume <= 0.005`; corroboration:
     per-axis local AABB extent difference `<= 0.5mm` (AABB is never the sole
     evidence). Parity FAIL for a collider marks its cooked shape UNCERTIFIED.
4. **Baseline replay**: D333's 200-step robot-free HOME sole-support baseline,
   verbatim, with the D333 baseline hard gate re-applied. Gate failure ->
   `D334_G0A_AUDIT_CONTRACT_FAIL_STOP` (pose-independent inventory/cook
   artifacts are still written).
5. **Pose A — frozen pre-step command pose**: `_write_exact_state` teleport to
   canonical joints + object center (no step). Live body poses read from
   articulation data (PhysX side), FK parity vs canonical contract recorded.
6. **Step-0 exact replay -> Pose B**: one `_physics_step` with the canonical
   command held (identical to D333 step 0). **Replay parity gate** vs
   `d333_teleport_settle_trace.csv` row 0 using the same `_state_row` code
   path: object pos delta `<= 0.05mm`, actual TCP delta `<= 0.05mm`,
   gripper_link force norm relative delta `<= 1%`, gripper contact point
   delta `<= 1mm`. Failure -> `D334_G0A_REPLAY_PARITY_FAIL_STOP`
   (pre-step-pose audit results are still written).
7. **Signed-distance matrix** (hppfcl GJK/EPA, `SIGNED_DISTANCE_BORDER_M=0.1mm`,
   AABB-only reasoning forbidden): for each pose in {A, B}, for each body in
   {link5, gripper_link}, for each representation in {certified cooked hull,
   raw stage mesh BVH}: signed distance / nearest points / EPA depth vs the
   analytic cylinder at that pose's live object pose. GJK/EPA sign
   consistency is a hard check per query.
8. **Contact-point mapping**: recorded D333 step-0 gripper aggregate contact
   point `[0.291517, 0.003320, 0.066635]m` (and the replayed contact point)
   mapped to each candidate shape: point-to-surface distance `<= 1mm` marks
   the shape as an on-surface candidate; also record distance to the actual
   tool raw meshes and to the cylinder surface.

## Pre-registered outcomes

Let "overlap" = signed distance `<= -0.1mm`, "clear" = `>= +0.1mm`.

1. `D334_G0A_COOK_ARTIFACT_SUPPORTED`: at pose B (or A), at least one
   **certified** cooked shape overlaps the cylinder while its corresponding
   raw mesh is clear, and the contact-point mapping + recorded gripper_link
   attribution are consistent with that shape's live owner body.
   -> collision-representation repair becomes the candidate (user choice).
2. `D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED`: a raw tool mesh itself overlaps
   the cylinder at the frozen pose(s), consistent with the recorded
   attribution. Raw overlap takes precedence over cook overlap.
   -> target-family repair becomes the candidate (user choice).
3. `D334_G0A_SHAPE_OWNERSHIP_PARITY_UNRESOLVED_MIXED_STOP`: cook/ownership
   parity failures, borderline distances, or overlap evidence inconsistent
   with the recorded gripper_link attribution.
4. `D334_G0A_REPLAY_PARITY_FAIL_STOP` / `D334_G0A_AUDIT_CONTRACT_FAIL_STOP`:
   contract failures as defined above.

Every branch stops after D334: `g0a_pass=false`, no ladder promotion, no mesh
rewrite, no target change in this session.

## Artifacts and Visualization DoD

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d334/` (forward-only path).
- JSONs: frozen invariant contract, prebaseline contract, live collider
  inventory, cook parity, signed-distance matrix, step-0 replay parity,
  summary (+ markdown summary).
- CSVs: 200-row baseline replay trace; two-row audit trace (pose A + replayed
  step 0) via the same `_state_row`/`_flatten_trace_row` code path as D333.
- PNGs `1..3`: pose-B 3D shape figure (cylinder + cooked hulls + raw meshes +
  recorded/replayed contact points), pose-A counterpart, optional contact
  close-up. Isaac frame markers via `draw_frames`; exactly one non-empty RRD.
- Snapshot/marker/RRD failure ->
  `D334_G0A_VISUALIZATION_ARTIFACT_CONTRACT_FAIL_STOP` (physics/geometry JSONs
  preserved).

## Non-goals

No mesh rewrite, no collision-approximation change, no target/gate/offset/
standoff tuning, no ownership scan beyond link5/gripper_link, no waypoint or
approach run, no second settle run, no 10-trial gate, no gripper close/grasp/
lift, no G0b, no RL/PPO, no randomization, no render beyond three diagnostic
PNGs, no video, no VLA, no RoArm real, no B200, no cube, no `/half-clone`.
Isaac package pins remain binding: `numpy==1.26.0`, `psutil==5.9.8` (no
installs planned; versions recorded at runtime).

## Amendments (pre-run, after an adversarial 3-lens design review — no physics executed yet)

A 3-agent review (contract parity / installed-API correctness / replay-logic)
found 8 distinct defects in the first probe implementation. All fixes tighten
gates and were applied before any run:

1. Cook certification = **volume parity only**, with **no direct-live-cook
   exemption**; the AABB extent comparison is demoted to recorded
   corroboration (property-query AABB frame/scale conventions are unverified,
   so it must not gate).
2. Raw-mesh "overlap" = collision **with EPA penetration depth >= 0.1mm**
   (bare `is_collision` no longer qualifies; sub-border grazing is borderline).
3. Contact-point "on-surface" = strictly `|surface distance| <= 1mm`
   (an interior aggregate point no longer qualifies via `is_collision`).
4. GJK/EPA consistency is enforced as the registered hard check: an
   inconsistent query is unusable evidence and can only be `borderline`
   (raw-mesh consistency = collision implies an EPA contact record exists).
5. The pose-A PNG is rendered at pose A (before the replay step), giving
   PNGs: pose A, pose B, zoomed contact map.
6. Property-query wait pumps run with `/app/player/playSimulations` disabled
   plus a before/after state guard, so waits cannot step physics.
7. Early D326 pin gate before scene creation; contract-failure aborts also
   write a minimal `D334_G0A_AUDIT_CONTRACT_FAIL_STOP` summary artifact.
8. Non-finite replayed contact points skip the point probe with a recorded
   null entry instead of feeding NaN into GJK.

## Runtime result (appended after the run)

Verdict: `D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED` (exit 0, artifact contract PASS)

### 1. Replay fidelity — bit-exact

Baseline replay hard gate PASS. Step-0 replay vs `d333_teleport_settle_trace.csv`
row 0: object pos delta `0.000000mm`, TCP delta `0.000000mm`, gripper force
`76.4128N` relative delta `0.00e+00`, contact point delta `0.000000mm`.
Every frozen-pose conclusion below is licensed by exact reproduction.
Pose-A FK parity `0.001255mm`.

### 2. Live ownership — clean, D332 owner-mismatch hypothesis refuted

PhysX property query (QUERY_RIGID_BODY_WITH_COLLIDERS): link5 owns exactly
`.../link5/collisions/link5/node_STL_BINARY_`, gripper_link owns exactly
`.../gripper_link/collisions/gripper_link/node_STL_BINARY_`; USD/PhysX parity
PASS, cross-body attachments `[]`. Direct live cook failed on both prim ids
(`RESULT_ERROR_COOKING_FAILED` on the API Xform, `RESULT_ERROR_INVALID_PARSING`
on the mesh child) — D332's "not supported" assumption is now recorded
evidence; mirror cook remains the only cook route.

### 3. Cook parity

- link5: volume parity `0.0498%` <= 0.5% -> **certified**.
- gripper_link: volume parity `1.4600%` > 0.5% -> **uncertified** (cooked
  gripper evidence cannot gate, recorded only).

### 4. Signed distances (border 0.1mm; all GJK/EPA consistency checks PASS)

| Pose | Body | cooked | raw |
|---|---|---:|---:|
| A (pre-step) | link5 | `-6.2367mm` overlap | `+4.2726mm` clear |
| A (pre-step) | gripper_link | `-15.3867mm` overlap | **`-5.9567mm` overlap** (EPA 5.863mm) |
| B (post-step-0) | link5 | `+3.0438mm` clear | `+7.3557mm` clear |
| B (post-step-0) | gripper_link | `-5.2737mm` overlap | **`-1.7216mm` overlap** (EPA 1.722mm) |

Contact-point mapping (recorded == replayed, both finite): point sits
`-5.3834mm` inside the cylinder surface, `0.549mm` from the gripper cooked
hull (on-surface), `1.289mm` from gripper raw, `9.69mm` from link5 cooked.

### 5. Interpretation

- The D333 gripper_link attribution is geometrically explained: **the actual
  tool geometry (gripper g2a collision STL, raw mesh) penetrates the cylinder
  at the frozen canonical pose** — ~5.96mm at the commanded pose, ~1.72mm
  after the step-0 settle. This is a target-family placement error, not a
  cook/proxy artifact. Pre-registered branch 2 applies; raw overlap takes
  precedence over cook overlap.
- The D332 link5 mirror overlap is reconciled: it is a real cook artifact of
  the link5 hull at the commanded pose (reproduced to `0.4um`), but link5 is
  clear at the settled pose and recorded `0N` — it was never the runtime cause.
- Secondary recorded finding: the gripper cooked hull inflates `~3.5-9.4mm`
  beyond raw with a failed volume parity; any future repair should keep this
  inflation in view, but it is not the primary cause.

### 6. Stop and user choice (pre-registered)

`g0a_pass=false`; no ladder promotion; no mesh rewrite; no target change in
this session. **The candidate next repair is a target-family repair** (the
canonical D325 pose family must stop placing the physical tool inside the
cylinder). User decision required before any repair session.

### Runtime evidence

- `sim_scripts/cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_live_collider_inventory.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_cook_parity.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_signed_distance_matrix.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_step0_replay_parity.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_baseline_replay_trace.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d334/d334_audit_trace.csv`
- PNGs: `d334_pose_a_shapes.png`, `d334_pose_b_shapes.png`, `d334_contact_map.png`
- RRD: `d334_live_collision_audit_trace.rrd`

### 7. User decision (recorded post-verdict, 2026-07-12)

At the pre-registered stop the user chose **option 1: target-family repair**,
to be executed as **D335 in a fresh session** (new variable
`[target_family_geometry]`; D334 signed-distance harness as the pre-physics
gate). The gripper hull-representation case stays deferred until after D335.
