# START_HERE.md

Last updated: 2026-07-13 KST (D340 complete:
`D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`; all 26 cook callbacks and
13 fixed-point subcontracts PASS, authored-stream hash gate FAIL. No attempt3,
no validate, physics `0` steps. D338 attempt1/D339 attempt2 are immutable.)

## Current Truth

- Active pivot remains **grasp track G0a on cylinder D34 x H90**. Cube repair,
  G0b close/lift, PPO/RL, VLA, randomization, real RoArm, and B200 remain out
  of scope.
- D337 verdict: `D337_G0A_STATIC_RUNTIME_MIXED_STOP`. Durable q5 convention:
  `q5=0` = CLOSED, `1.571rad` = OPEN. At `q5=1.5413`, the open-jaw scan passed
  `2,560/2,629`; original `(7,11)mm` was raw-clear (`+4.2726/+11.1751mm`).
  The 200-step settle kept raw meshes clear and final displacement `2.754mm`,
  but a link5 `38.861N` step-0 impulse disturbed the object (max XY `5.418mm`,
  tilt `4.208deg`). D334's cooked link5 hull overlaps `-6.2367mm` there, so
  collision representation remains the blocker.
- **USD/URDF divergence (durable fact)**: the robot USD (5/13) embeds full
  `gripper_link.stl` as the moving-jaw collision mesh; the URDF was changed
  5/14 to a 4mm-box proxy (`g2a`). The USD is stale vs the URDF but more
  physical; it remains the audited truth. Any regeneration must decide the
  moving-jaw representation explicitly.
- D338 verdict: `D338_G0A_ASSET_BUILD_CONTRACT_FAIL_STOP`. attempt1's explicit
  cook returned while every global cooking-statistics delta stayed zero; its
  callback result/count was not recorded before the gate, so it licenses no
  geometry claim. No derivative or physics was created; attempt1 is immutable.
- D339 verdict: `D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`.
  - The repaired callback-first witness passed. All four fresh-stage requests
    called back exactly once with `RESULT_VALID` and `64` serialized hulls.
    For both link5 and gripper, cold1/cold2 had equal `64/64` part topology and
    hashes with maximum coordinate delta `0.0m`. All six global counter deltas
    again stayed zero and are informational only. D338 attempt1 stayed exact.
  - attempt2 derivative build PASS: only the physics layer changed under the
    registered allowlist; non-physics layers and tool mass/COM/inertia remained
    equal. Both bodies saturated the frozen `maxConvexHulls=64` cap.
  - Live audit FAIL. USD inventory correctly had 64 enabled new parts and one
    disabled legacy hull per body, but PhysX property query still enumerated
    `65` colliders by including that legacy path. Direct re-cook returned one
    convex for every new part, yet surface fidelity `<=0.1mm` failed on link5
    `8/64` (worst `4.894877mm`) and gripper `5/64` (worst `0.699067mm`), leaving
    only `56/64` and `59/64` directly certified parts.
  - Therefore cooked-union target distance was intentionally not queried;
    baseline/settle did not run; controlled physics steps `0`. One PNG and one
    non-empty 1-step RRD passed the artifact contract. No claim is licensed
    about target fidelity or static physics.
- D340 verdict: `D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`.
  - All 26 instance/prototype requests called back exactly once/inline with
    `RESULT_VALID(0)`, one convex, zero serialization errors, full cache release
    and settings restore. All 13 channel pairs were bit-exact (`0.0m` max
    delta); containment and float32 round trip were `0.0m`; every part strictly
    reduced vertices (`114 -> 100`, link5 part_041 removed two, others one).
  - The sole failed check on all 13 was `authored_hash_matches_d339_manifest`.
    D340 hashed the points after a near-identity body transform (max delta
    `2.22e-16`), while D339's manifest hashes the direct authored Vec3f stream.
    Bounds changed at most `2.22e-16m`, but all float64 hashes and 10/13 Qhull
    topology hashes changed. This is a proof-frame false negative, not a cook
    divergence or parameter increase; the registered FAIL verdict stands.
  - Attempt3 is absent; validate/cooked-union/physics did not run; sim counter
    `0->0`. PNG/RRD passed and were inspected. Existing scalar increases and
    changes remain `0/0`; `g0a_pass=false`.

## Active Case: G0a / D340

- Object: cylinder r `0.017m`, h `0.090m`, fixed `(0.300,0.000)`; mass
  placeholder `0.72kg` (real mass required before G0b); friction `1.5/1.2`.
- Feasible open-jaw target family exists: `position_only_tangent_minus1` +
  `q5=1.5413rad`; canonical candidate `(7,11)mm`.
- G0a gates remain: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- **D340 complete / stopped at capture**. 이번 case의 신규 변수:
  `[failing_part_fixed_point_geometry, enabled_shape_property_binding_contract]`
  (exactly 2; one physical 13-part allowlist + one measurement-only contract).
- Existing physical/decomposition/target/control/solver/cache/tolerance scalar
  increases and changes are both `0`; `64 hull/body` is saturation of the
  already-frozen D338 `maxConvexHulls=64`, not an increase.
- Output: `claudedocs/runtime_logs/grasp_track/g0a_d340/`; the only allowed
  derivative is forward-only `collision_asset/attempt3/`. D338 attempt1 and
  D339 attempt2 are immutable.
- Detailed registration and result:
  `claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md`.
- `g0a_pass=false`; G0b/RL/ladder remain blocked.

## Next Concrete Action

Stop for user choice. Recommended next case is a reactive, measurement-only
**D341 authored-coordinate-stream contract repair**. Preserve D340 capture
evidence; compare direct authored Vec3f points against the D339 manifest before
any transform, and use body-mapped coordinates only for containment/proximity.
Then preregister the still-uncreated attempt3 authoring plus fresh validation.
Do not overwrite/rerun D340, do not relax the `1e-9m/0.1mm/5%/0.5mm` gates,
and do not run physics. Collision authoring remains user-approval gated.

Reserve choices (not active):

1. Onset-metric hardening (record impulse-row onsets) — REACTIVE fix allowed
   only as part of the next settle case, not standalone.
2. `r>17mm` grasp-depth redefinition — unchanged, not needed now.

All future ideas go to `claudedocs/BACKLOG.md`. `g0a_pass=false`;
G0b/RL/ladder promotion remain blocked.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` only for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D340)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d340/d340_capture_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d340/d340_capture_postrun_root_cause_audit.json`
9. `claudedocs/runtime_logs/grasp_track/g0a_d340/d340_preregistration.json`
10. `claudedocs/session_20260713_grasp_g0a_d339_cook_witness_contract_repair.md`
11. `claudedocs/runtime_logs/grasp_track/g0a_d339/d339_live_collider_audit.json`

## Durable Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- Git commit/push only on explicit user request.
- B200/JHPark/SSH/pull/.ssh and `/half-clone` remain forbidden.
- Variable Ladder: one or two new variables per case; grasp outputs only
  under `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`; forward-only.
- **q5 convention: `q5=0` = CLOSED; sim "open" = `~1.541-1.571rad`** (D337).
- BVH distance scalars of colliding meshes are ranking-invalid (D336); use
  contact-level EPA enumeration. AABB-only reasoning is forbidden.
  Distinguish raw mesh, mathematical hull, mirror cook, live cook.
- PhysX `get_cooking_statistics()` all-zero deltas do not positively witness
  synchronous explicit `request_convex_collision_representation` (D338).
- Callback-first two-cook geometry equality proves deterministic cook output,
  not live shape binding/fidelity. USD `collisionEnabled=false` alone did not
  remove the legacy shape from D339's PhysX property query. Never infer live
  collider cardinality from authored USD inventory alone (D339).
- D338 attempt1 and D339 attempt2 may never be reused or overwritten.
- D340 may author only 13 registered derivative parts once; no iterative retry,
  channel fallback, scalar increase, tolerance relaxation, or physics is allowed.
- Bit-exact geometry hashes are meaningful only for the same coordinate stream.
  Never compare direct authored Vec3f hashes to post-transform float64 hashes;
  prove authored identity before mapping, then gate mapped geometry numerically.
- Visualization DoD and Isaac pins binding: `numpy==1.26.0`, `psutil==5.9.8`.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union` (now folded into the D338 representation decision).
