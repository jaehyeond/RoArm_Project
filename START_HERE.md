# START_HERE.md

Last updated: 2026-07-16 KST. D354 remains the latest q5 geometry science.
D357 completed the beginner-readable visualization and D358 completed the approved
offline moving-jaw hash-provenance retry. No later case is currently approved.

## Current Truth

- The pivot remains cylinder grasp-track G0a (`radius=0.017m`, `height=0.090m`).
  G0b close/lift, settle, ten-trial, PPO/RL, VLA, ladder, real hardware, and B200
  are out of scope.
- q5 convention is fixed: URDF `q5=0` is CLOSED; frozen sim OPEN is
  `q5=1.5413rad`. D347 measured HOME-near + q5=0 CLOSED, not exact HOME.
- D348 proved PhysX volume must use callback polygon topology, not a new
  vertex-only Qhull. Corrected gate: `256/256` channels, `128/128` parts.
- D349 frozen-OPEN raw/live distances were link5
  `4.2726455336/4.2727365803mm` and moving gripper
  `11.1750883746/11.3402623263mm`; these were not contact/grasp proof.
- D350 measured the actual connected fixed-jaw surface and completed real Isaac
  Viewer plus 64+64 collider visualization, but `aligned_pass=null`,
  `g0a_pass=false`.
- D351 never reached q5 science. D352 localized pending Timeline PAUSE; D353
  proved one conditional main-thread `Timeline.commit()` applies it with zero
  registered world advance.

## Latest Scientific Case: D354 current-pose q5 closure geometry

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d354/`.
- Verdict: `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`; completion PASS,
  scientific contract FAIL, controlled physics steps `0`, `g0a_pass=false`.
- Raw/live shared Float32 q5 clear/overlap bracket
  `1.0269782543182373/1.0269775390625rad`, width
  `7.152557373046875e-7rad`.
- Raw signed distances: `+0.0010050812803802547/-0.000988475720559677mm`;
  live: `+0.0010049780471806762/-0.0009864198978583663mm`.
- Both clear endpoints were exactly cylinder-local `z=+0.045m`, classified
  `cap_or_rim_boundary`; adjacent overlap endpoints alone were
  `barrel_interior`. Barrel-feature consensus and cap-competitor exclusion failed.
- Moving contact patch identity was unambiguous, but full surface binding failed
  derived-hash/runtime-roundtrip exactness. Immutable authored streams and face
  order were exact; paired-XZ SHA authored `917b7154...bcaf9` versus raw-derived
  `98ef77e6...18bbae`.
- This neither certifies barrel-first/current-pose grasp nor proves grasp
  impossible or target/IK repair necessary. Do not invent a cap/rim tolerance.

## D355/D356 runtime-cause status

- D355 registered one offline audit invocation, then plain `isaaclab` Python
  stopped at the first `from pxr import ...`. USD loads, recipes, hashes,
  perturbations, q5, physics, and Isaac launch were all `0`; provenance is null.
- D356 corrected the cause without rerun: D343/D345 had already proven the
  installed bundled `omni.usd.libs` standalone-core-PXR route with exact
  `PYTHONPATH`/`LD_LIBRARY_PATH`, OpenUSD `0.24.5`, and no Kit/GPU/physics.
  D355 omitted that environment and an import/version preflight. Isaac Sim,
  RTX, Warp/SM, and PhysX did not fail.
- D355 remains immutable `D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP` evidence;
  only its overbroad Kit-only cause inference is superseded by D356.

## Latest Completed Observability Case: D357

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d357/`.
- Verdict: `D357_D354_BEGINNER_VISUALIZATION_REPAIR_COMPLETE`.
- One actual `headless=false` Isaac GUI invocation completed normally: worker
  exit `0`, watchdog false, elapsed `84.53322536998894s`, Viewer hold
  `60.02044868003577s`, UI updates `565`.
- Engine log was not clean: it contains a non-fatal Fabric clone error and two
  missing-`d338_convex_parts` warnings before successful scene/capture/close.
  The no-advance result means unchanged from reset baseline (timeline `0.03s`,
  sim `0.01s/index2`), not absolute clock zero.
- Frozen OPEN/last-clear/first-overlap display writes were `3/3`; q5 science,
  distance/contact queries, new classification, and controlled physics steps
  were all `0`.
- Three 1280x720 Isaac PNGs, Korean sheet, actual-scale and Z×50000 display-only
  diagrams, exact 62-entity RRD/RBL and 4800x2800 screenshot passed.
- Original-resolution inspection found the moving jaw occluded behind the
  cylinder in the same side camera. Therefore those three Isaac PNGs alone do
  not show contact. A forward-only 2400x1500 Korean addendum separates:
  visible placement, occluded jaw interface, and unexecuted PhysX/grasp tests.
- GPU 82 samples: VRAM `2676..8321MiB`; utilization `0..36%`, mean
  `9.963414634146341%`. Static Viewer hold is not a GPU-saturation workload and
  used no arbitrary Warp/SM tuning.
- Completion / addendum PNG SHA-256:
  - `89a20139c12d6936ae052d0069829f0381e6935ba5dcb1b3dcbf581fc3581e71`
  - `567aab0e719c3cef52470c8b275b46b3f3b492b8eaadb9213c5b1e726309294f`
- D354 science is unchanged. Isaac reached/displayed the cylinder pose, but no
  physical jaw closure, force/friction, object motion, hold, or lift test ran.

## Operational Residue

- The persistent inspection-only Rerun 0.34.1 GUI from D355 may remain open on
  `DISPLAY=:1`; it is not a scientific process.
- D342 cleanup is incomplete: approved SIGTERM removed wrapper PID `1729610`,
  but worker `1729639` survived/reparented with about `977284KiB` RSS and
  `320MiB` GPU at the recorded audit. No SIGKILL or extra signal is approved.

## Latest Completed Offline Provenance Case: D358

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d358/`.
- Verdict: `D358_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`; operational execution
  PASS, provenance localization FAIL_STOP, `g0a_pass=false`.
- Registered bundled standalone core-PXR/OpenUSD `0.24.5` preflight passed. One
  audit invocation completed exit `0` without retry: recipe search
  `90.19101224502083s`, completion marker `90.21373272209894s`.
- Current D354 authored 8-field bundle, authored point/count/index streams, raw
  full stream, and raw inner paired-XZ all reproduced. Independent current
  authored/raw calculations were `17/17`; perturbation controls were `7/7`.
- Only inner/outer paired-XZ of the D351 frozen expected bundle reproduced
  (`2/8`). Inner/outer vertex, triangle, and patch fields each had zero matches
  across `20,736` registered recipes; no coherent eight-field recipe existed.
- Authored Float32-mm ↔ runtime Float64-m roundtrip differed in
  `58,506/123,282` components and `36,519/41,094` vertices, with maximum absolute
  delta `0.0000031862526839177008mm`. The registered roundtrip recipes still did
  not produce the six historical hashes, so a simple dtype/unit cause is not proven.
- D334 sidecar was unchanged; D358 case-local Isaac/Kit/GPU/q5/physics/contact/
  cap-rim/asset/target-IK/path/dependency contract counts were all `0`, and the
  process imported no forbidden modules. These are not system-wide telemetry for
  a separately open persistent GUI. Rerun was correctly omitted because this was
  a pure file/hash/schema audit.
- Do not call this an Isaac or geometry failure. Do not replace the D351 constants
  with D354 values or relax the binding gate without source/generator evidence.
- D358 is immutable and must not be rerun or overwritten.

## Next Authorization Boundary

- There is no active approved case. D358 cannot decide cap-versus-rim contact
  order or physical grasp.
- Narrow evidence-recovery candidate: a separate forward-only, read-only
  historical provenance case tracing the first generator/source/commit for the
  six unreproduced D351 expected hashes. It must not change a gate or hash.
- Alternative physical candidate: an actual PhysX jaw-close/contact-force/object-
  motion test with a camera that exposes the jaw-cylinder interface rather than
  repeating D357 occlusion. It also requires fresh approval and preregistration.
- These candidates are not interchangeable: the first repairs evidence lineage;
  the second asks a new physical question. Do not run either without the user
  choosing and approving it.
- Target/IK/path repair remains blocked until the required evidence actually
  rejects the frozen current pose.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, q0-q5/object initial state,
  gates/tolerances, material, mass, actuator, renderer, solver, or physics.
- Do not run settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion.
- Do not substitute vertex-only Qhull or Rerun Float32 display data for canonical
  callback/Float64 evidence.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D358 evidence is immutable.
- Do not modify user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- Hardware, B200/SSH, `/half-clone`, unapproved signal, commit, and push are forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D358; ledger tail
2. `claudedocs/session_20260716_grasp_g0a_d358_moving_jaw_patch_hash_provenance_retry.md`
3. D358 preregistration, evidence, phase markers, report, and completion JSON
4. `claudedocs/session_20260716_grasp_g0a_d357_d354_beginner_result_visualization_repair.md`
5. D357 completion, worker/supervisor, Rerun/manual, and occlusion addendum
6. D356/D355 plus D343/D345 bundled standalone-core-PXR evidence
7. D354 session, measurement, moving binding, attestation, and completion JSON
8. D353/D352; D351 original/repair/harness; D348-D350 referenced evidence

## Git

- Current base `HEAD == origin/master ==
  161f6d9d185bb41eb29259349ee0fd897a3c6de8`.
- D354 base `b7beb91997859a5ddb2b0407388e80aed45898dc` and D355 base
  `64aa5b2c9552a053a3a9a34551fbfd168ce644ba` are historical.
- Worktree was clean before D356 and is intentionally dirty with completed
  D356/D357/D358 work. No commit or push is authorized.
