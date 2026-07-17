# START_HERE.md

Last updated: 2026-07-16 KST. D361 completed the offline total-contact-capacity
and durable per-step/body/value prefix repair. No actual q5/PhysX science or new
contact video ran. There is now no active approved case; an actual rerun still
requires a new explicit user approval.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius is `0.017m = 17mm = 1.7cm`;
  diameter is `0.034m = 34mm = 3.4cm`; height is `0.090m = 90mm = 9cm`.
- q5 convention: `q5=0` is CLOSED; frozen sim OPEN is `q5=1.5413rad`.
  D347 measured HOME-near + q5=0 CLOSED, not exact HOME.
- D348 proved PhysX volume must use callback polygon topology, not a new
  vertex-only Qhull. Corrected gate: `256/256` channels, `128/128` parts.
- D349 frozen-OPEN raw/live distances: link5
  `4.2726455336/4.2727365803mm`; moving gripper
  `11.1750883746/11.3402623263mm`. These are not contact/grasp proof.
- D350 completed connected fixed-jaw surface measurement, real Isaac Viewer,
  and 64+64 collider visualization, but `aligned_pass=null`, `g0a_pass=false`.
- D351 never reached q5 science. D352 localized deferred Timeline PAUSE; D353
  proved exactly one conditional `Timeline.commit()` applies PAUSE with zero
  registered world advance.
- Latest completed zero-step geometry science remains D354:
  `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`, controlled physics steps `0`,
  `g0a_pass=false`.
- Latest completed control/evidence case is D361:
  `D361_CONTACT_CAPACITY_AND_PREFIX_TRACE_REPAIR_PASS_NO_PHYSICS`. It does not
  change D354/D360 physical nulls or `g0a_pass=false`.

## D354 Scientific Boundary

- Raw/live shared q5 last-clear/first-overlap bracket
  `1.0269782543182373/1.0269775390625rad`, width
  `7.152557373046875e-7rad`.
- Both clear endpoints were exactly cylinder-local top `z=+0.045m`, classified
  `cap_or_rim_boundary`; adjacent overlap endpoints alone were
  `barrel_interior`. Barrel-first was therefore not established.
- This neither proves current-pose grasp nor proves it impossible, and does not
  justify target/IK repair.

## D359 Completed Evidence-Lineage Recovery

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d359/`.
- Verdict: `D359_D351_HASH_PROVENANCE_RECOVERED`.
- Actual historical generator used ascending original USD point IDs. Later
  narration/validator used coordinate-row lexicographic unique.
- Original-ID replay reproduced frozen D351 `8/8` for both D344/local streams;
  coordinate replay reproduced historical `2/8` but D358 current-authored
  `8/8`.
- Root cause was vertex-remap/serialization provenance, not geometry, unit,
  Isaac, PhysX, GPU, Warp, or SM efficiency. D351/D354/D358 remain immutable;
  no expected hash or gate was changed.

## D360 Sole Actual-Physics Invocation

- Output `g0a_d360/` is immutable: no add/overwrite/rename/retry/finalize.
- Verdict:
  `D360_SINGLE_INVOCATION_CUDA_DEVICE_ASSERT_AFTER_243_CONTROLLED_STEPS_PRE_TRACE_FAIL_STOP`;
  all physical/body/contact/motion fields `null`, `g0a_pass=false`.
- One real GUI/RTX run passed prepare `16/16`, worker preflight `14/14`, corrected
  live binding `128/128 (64+64)`, OPEN baseline `200/200`, then changed only q5
  target once to `0.0rad`. Exception evidence records 243 completed steps, so 43
  closure rows existed only in RAM.
- Counters 233/243 preserve only that the registered any-robot contact and object
  motion triggers fired in program order. Lost body/value rows mean no moving-jaw,
  force, q5-response, displacement, or contact-location claim is allowed.
- Root operational failure: inherited detailed-contact total capacity was
  `16 × 1 body × 1 env = 16`; four filters did not multiply it. Log warned
  `>16`, then ATen index bounds and PhysX CUDA device asserts occurred.
- This was not a geometry verdict or recorded OOM/Warp/SM failure. GPU used/free
  extrema were `7703/8241MiB`, utilization max `43%`; exit `-9` sender unknown.
- Eight actual failure-state PNGs exist but do not expose the interface/body/value;
  trace/summary/final/RRD/RBL never existed.

## D361 Completed Capacity / Prefix Repair

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d361/` (immutable; do not
  rerun, overwrite, add, rename, or resume).
- Exact installed-version/shape envelope:
  `33,280 = 1 sensor cylinder × (1 table + 1 link4 + 64 link5 + 64 gripper)
  × 256 PhysX-5.6.1 contacts/geometry-pair`.
- Direct/per-filter capacity and visible-memory arithmetic passed; checks
  `13/13`, negative controls `9/9`. Visible detail arrays are
  `1,064,960B = 1.015625MiB` plus count/start 32B. Actual runtime sufficiency
  remains `null` because no PhysX run occurred.
- Durable protocol: exclusive `header/(step_begin,step_observation)*/seal`, each
  record fsync + fresh exact reread, previous/self SHA-256, full inherited D360
  state, exact body/filter path-index, force vector/norm, contact-count range and
  high-water, independently recomputed event body/value, registered seal count.
- Offline failure injection passed `17/17`. Normal reference was 8 records/3
  observations/seal; abrupt exits recovered `0/1/1` completed observations and
  isolated a 147-byte partial tail. Four ordinary tamper cases and four
  hash-valid semantic tamper cases were rejected. Per-test journal 17 rows was
  durable before aggregate verdict.
- Exact artifacts `23/23`; D360 tree and D334 sidecar unchanged; exception,
  image/video/RRD/RBL absent. Isaac/PhysX/physics-step/q5/change counts all `0`.
- Final nulls: contacting body, force, object motion, current-pose support,
  grasp feasibility all `null`; `g0a_pass=false`.

## No Active Approved Case

- Narrow next candidate: a new forward-only bounded actual q5/PhysX
  contact-motion rerun that inherits the exact frozen D360 scene/target/physics
  variables and integrates D361 capacity + durable prefix before invocation.
- It may include a newly approved interface-visible contact video, but capacity
  or trace PASS alone is never contact/grasp/G0a proof.
- This candidate is **not approved**. Do not prepare, implement, or execute it
  until the user explicitly approves a new case and its variable boundary.

## Frozen Boundaries

- No target/IK/path or initial q0-q5/object-state change.
- No asset, cook, decomposition, gate/tolerance, material, mass, actuator,
  renderer, solver, physics, or dependency change without a new scoped case.
- No exact cap/rim/barrel discriminator, force-closure claim, grasp/hold/lift,
  settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion.
- Do not substitute Rerun Float32 display data or vertex-only Qhull for canonical
  callback/Float64 evidence.
- D351-D360 evidence is immutable. Do not modify user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, unapproved
  signal, commit, or push.

## Operational Residue

- No D360 Isaac/Kit/Rerun worker remains.
- Historical D342 worker PID `1729639` still survives/reparented after the
  previously approved SIGTERM. SIGKILL or another signal is not approved.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D360-D361; ledger tail
2. `claudedocs/session_20260716_grasp_g0a_d361_contact_point_capacity_and_prefix_trace_repair.md`
3. D361 preregistration, capacity, protocol, failure results/journal, completion
4. `claudedocs/session_20260716_grasp_g0a_d360_current_pose_bounded_physx_contact_motion.md`
5. D360 prerequisites, phase, exception, raw log, supervisor JSON
6. D359 session/evidence and D354 measurement/binding/attestation/completion
7. D333 sensor/trace, D348 topology, D353/D352 bridge, D351 original/repair

## Git

- Current base verified at D361 boot: `HEAD == origin/master ==
  e7ed71ca80768df9037c16e53a12d3c032af3d5d`.
- Worktree was clean at D361 boot and is intentionally dirty only with completed
  forward-only D361 code, state docs, and evidence.
- No commit or push is authorized.
