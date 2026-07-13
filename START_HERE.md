# START_HERE.md

Last updated: 2026-07-13 KST (D339 complete:
`D339_G0A_PREPHYSICS_CONTRACT_FAIL_STOP`; cook/build PASS, live-collider audit
FAIL, physics `0` steps. D338 attempt1 and D339 attempt2 are immutable.)

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

## Active Case: G0a / D339

- Object: cylinder r `0.017m`, h `0.090m`, fixed `(0.300,0.000)`; mass
  placeholder `0.72kg` (real mass required before G0b); friction `1.5/1.2`.
- Feasible open-jaw target family exists: `position_only_tangent_minus1` +
  `q5=1.5413rad`; canonical candidate `(7,11)mm`.
- G0a gates remain: TCP `<=5mm`, tangent `<=15deg`, jaw gap `[0,5mm]`, no
  penetration, contact `>=15mm` below top, displacement `<5mm`, `10/10`.
- **D339 complete / stopped pre-physics**. 이번 case의 신규 변수:
  `[cook_witness_contract]` (1개, measurement contract only).
- Frozen intervention: D338's full-mesh link5/gripper decomposition candidate
  and every physical/decomposition parameter remain unchanged. D339 changes
  only how an independent cook is positively witnessed.
- Output: `claudedocs/runtime_logs/grasp_track/g0a_d339/`; asset build writes
  only to `collision_asset/attempt2/`. D338 `g0a_d338/.../attempt1/` is immutable.
- Detailed registration and result:
  `claudedocs/session_20260713_grasp_g0a_d339_cook_witness_contract_repair.md`.
- `g0a_pass=false`; G0b/RL/ladder remain blocked.

## Next Concrete Action

Stop for user choice. Recommended next case is a separately pre-registered
**D340 fixed-point live-authoring repair**: keep the D339 source, target,
physics, decomposition parameters, and thresholds frozen; retain the 115
passing parts and stabilize only the 13 failing parts after measuring both
live-instance and prototype cook geometry. The property contract must separate
the exact 64 enabled shapes from the one known disabled legacy enumeration row;
enumeration alone is not active-collision evidence. D339 attempt2 is immutable.
Collision-asset changes require explicit approval. Only a clean `64/64`
per-body surface audit plus `128/128` property/direct volume binding may query
the frozen cooked-union target distance; physics and 10-trial remain blocked.

Reserve choices (not active):

1. Onset-metric hardening (record impulse-row onsets) — REACTIVE fix allowed
   only as part of the next settle case, not standalone.
2. `r>17mm` grasp-depth redefinition — unchanged, not needed now.

All future ideas go to `claudedocs/BACKLOG.md`. `g0a_pass=false`;
G0b/RL/ladder promotion remain blocked.

## Must Read First

1. `AGENTS.md` (then `CLAUDE.md` only for Claude-specific workflow)
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D339)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/direction_20260708_grasp_pivot.md`
6. `claudedocs/session_20260713_grasp_g0a_d339_cook_witness_contract_repair.md`
7. `claudedocs/runtime_logs/grasp_track/g0a_d339/g0a_d339_cook_witness_contract_repair_summary.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_cook_witness_manifest.json`
9. `claudedocs/runtime_logs/grasp_track/g0a_d339/d339_live_collider_audit.json`
10. `claudedocs/session_20260713_grasp_g0a_d338_collision_representation_repair.md`

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
- Visualization DoD and Isaac pins binding: `numpy==1.26.0`, `psutil==5.9.8`.

## Frozen Background

- Professor direction: finish one graspable cylinder case first, then
  G0b -> G1a grid -> G1b standalone PPO
  (`claudedocs/direction_20260708_grasp_pivot.md:3-32`).
- Tap track frozen at D321: `1920/2000` accepted (`96.0%`).
- G0b prerequisites: real cylinder mass measurement and BACKLOG
  `tool_surface_union` (now folded into the D338 representation decision).
