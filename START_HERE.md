# START_HERE.md

Last updated: 2026-07-16 KST. D359 recovered the historical D351 hash-generator
lineage. The sole D360 actual-physics invocation then stopped after 243 completed
controlled steps before its numeric trace was durably written. No active case is
approved now.

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

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d360/` (immutable failure
  evidence; do not add, overwrite, rename, retry, or finalize).
- Verdict:
  `D360_SINGLE_INVOCATION_CUDA_DEVICE_ASSERT_AFTER_243_CONTROLLED_STEPS_PRE_TRACE_FAIL_STOP`.
- Final fields: `physical_verdict=null`, `body_identity=null`,
  `moving_gripper_contact_supported=null`,
  `object_motion_after_moving_gripper_contact_supported=null`,
  `g0a_pass=false`, retry/finalize unauthorized.

### What Completed

- Prepare `16/16`, worker preflight `14/14`, D348 corrected audit `128/128`,
  live collider binding `64+64`, frozen stage/actuator/object and sole-support
  sensor prerequisites PASS.
- One real `headless=false`, `cuda:0`, `DISPLAY=:1` Isaac GUI invocation ran.
- OPEN baseline completed `200/200` controlled steps.
- q5 target changed exactly once to CLOSED `0.0rad`; q0-q4 target bits were
  unchanged.
- Worker exception recorded `243` completed controlled steps: 200 baseline plus
  43 closure rows computed in memory.

### What Survived Only Provisionally

- Contact capture counter `233`: program order proves only that at least one of
  link4/link5/gripper met the registered `>=0.1N` two-row trigger (closure
  onset/confirmation steps `31/32`).
- Motion capture counter `243`: program order proves only that the registered
  cylinder XY `>=0.5mm` or tilt `>=1deg` two-row trigger fired (steps `41/42`).
- Body label and numeric rows remained in RAM and were lost before trace write.
  Therefore do not claim moving-jaw contact, force value, q5 actual response,
  cylinder displacement, or contact location.

### Why It Stopped

- Inherited D333 ContactSensor used `track_contact_points=True` and
  `max_contact_data_count_per_prim=16`.
- IsaacLab defines this as total capacity across sensor bodies/environments;
  here `16 * 1 body * 1 env = 16`. Four filters do not multiply it.
- Worker log first warned that actual contact points exceeded 16, then ATen
  `indexSelectLargeIndex` failed its bounds assertion, followed by PhysX CUDA
  device-side asserts.
- This is a contact-point buffer/indexing runtime-observability failure, not a
  geometry FAIL or a physical verdict.
- The recorded direct failure was not VRAM exhaustion or a Warp/SM-occupancy
  failure. Across 102 telemetry samples: GPU used max/free min
  `7703/8241MiB`, utilization max `43%`, RAM available min
  `11063463936 bytes`; no OOM/allocation-failed record. GPU utilization is not
  itself a Warp-occupancy measurement.
- Worker exit was `-9`; watchdog was false. SIGKILL sender is unknown and must
  not be attributed to user, watchdog, or OOM killer.

### Visualization and Missing Evidence

- Eight actual `1280x720 RGBA` Isaac PNGs decode and were inspected at original
  resolution: initial, precommand, provisional contact, provisional motion,
  each in two views.
- The moving-jaw/cylinder interface and numeric/body overlays are not clear;
  event frames look nearly unchanged. They preserve failure-state images but
  cannot identify the contacting body or quantify motion.
- Physics trace JSON/CSV, worker summary, final PNG, RRD/RBL, Rerun validation
  and screenshot, beginner sheet, manual/completion summary do not exist.
  Finalize was not run and is not authorized.

## No Active Approved Case / Next Candidate

- Narrowest candidate, requiring new explicit approval: D361
  `[contact_point_capacity_and_prefix_trace_repair]`.
- Before any q5/PhysX rerun, preregister a justified total contact-point capacity
  and append-only per-step prefix/event body/value writes so a crash cannot erase
  completed decision rows.
- Exact capacity is not chosen yet. Do not silently use an arbitrary large value.
- Any actual q5/PhysX science rerun requires explicit approval after D361 scope
  briefing. Target/IK repair remains blocked because D360 physical verdict is null.

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

1. `AGENTS.md`; this file; DECISIONS D359-D360; ledger tail
2. `claudedocs/session_20260716_grasp_g0a_d360_current_pose_bounded_physx_contact_motion.md`
3. D360 prepare, prerequisites, phase, exception, raw log, supervisor JSON
4. D359 session, evidence, clarification, and completion JSON
5. D354 session, measurement, moving binding, attestation, and completion JSON
6. D333 sensor/trace and D348 corrected topology evidence
7. D353/D352 pause/commit evidence; D351 harness and original/repair sessions

## Git

- Current base: `HEAD == origin/master ==
  d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5`.
- Worktree was clean at D359 boot and is intentionally dirty only with approved
  forward-only D359/D360 code, state docs, and evidence.
- No commit or push is authorized.
