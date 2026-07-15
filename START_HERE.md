# START_HERE.md

Last updated: 2026-07-16 KST. D354 completed the frozen current-pose q5
closure-science measurement and full observability contract in one forward-only
run. The evidence pipeline completed, but the scientific contract did not:
`D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`, controlled physics steps `0`,
`g0a_pass=false`. No next case is approved.

## Current Truth

- The research pivot remains cylinder grasp-track G0a (`radius=0.017m`,
  `height=0.090m`). G0b close/lift, settle, ten-trial, PPO/RL, VLA, ladder,
  real hardware, and B200 are out of scope.
- q5 convention is fixed: URDF `q5=0` is CLOSED; frozen sim OPEN is
  `q5=1.5413rad`. D347's measured reset was HOME-near q5=0 CLOSED, not exact HOME.
- D348 proved PhysX property volume must be compared with callback polygon
  topology, not a new vertex-only Qhull. Corrected gate: `256/256` channels,
  `128/128` parts.
- D349 frozen-OPEN raw/live distances were link5
  `4.2726455336/4.2727365803mm` and moving gripper
  `11.1750883746/11.3402623263mm`; these were not contact/grasp proof.
- D350 measured the actual connected fixed-jaw surface and completed the real
  Isaac Viewer plus 64+64 collider visualization, but `aligned_pass=null` and
  `g0a_pass=false`.
- D351 never reached q5 science. D352 localized the current blocker to deferred
  Timeline PAUSE state; D353 proved one conditional main-thread
  `Timeline.commit()` applies it without advancing registered world state.

## Latest Completed Scientific + Observability Case: D354

- Forward-only output:
  `claudedocs/runtime_logs/grasp_track/g0a_d354/`.
- Base `HEAD == origin/master` was
  `b7beb91997859a5ddb2b0407388e80aed45898dc`. Prepare and exactly one actual
  `DISPLAY=:1`, `headless=false`, `cuda:0` validate ran; no retry occurred.
- New scientific/physical variables were `[]/[]`. The only new operational
  variable repaired D351's negative `asset_write=false` aggregation polarity;
  target/IK/path and all scientific/physics inputs remained frozen.
- The D353 bridge reproduced with exactly one commit. Evaluator invocation /
  cache-miss state write / primary unique row was `377/72/70`; repeat rows `2`,
  auxiliary writes `13/13`, Viewer updates `11161`, guard failures `0`.
- Separate summary fsync+reread attestation authoritatively recorded controlled
  physics steps `0`; earlier intermediate `null` fields remain historical.

### Measured first-contact bracket

- Raw and live used the same Float32 bracket:
  - clear q5 `1.0269782543182373rad`
  - overlap q5 `1.0269775390625rad`
  - width `7.152557373046875e-7rad`, adaptive depth `16`
- Raw clear/overlap distance:
  `+0.0010050812803802547/-0.000988475720559677mm`.
- Live clear/overlap distance:
  `+0.0010049780471806762/-0.0009864198978583663mm`.
- Raw/live contact q delta was `0rad`; contact surface-travel delta was
  `0.00004817170331236983mm`; maximum contact-endpoint distance delta was
  `0.000002055822701310661mm`. Both contact-order certificates passed.
- The precontact table corridor was continuously clear. Minimum classified
  clearance was `65.42070265676648mm`; minimum conservative strict margin was
  `63.22081483325994mm`.

### Why the scientific contract failed

- Raw/live clear witnesses were exactly cylinder-local `z=+0.045m`, classified
  `cap_or_rim_boundary`. Their immediately adjacent overlap witnesses were
  `barrel_interior` at raw/live z
  `0.044999618601561694/0.044999619394590046m`.
- The frozen strict classifier therefore had no clear/overlap barrel-feature
  consensus and could not exclude the cap competitor over the full bracket.
  This is not `D354_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE` and not
  `D354_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED`.
- The moving contact patch itself was the frozen distal inner patch and raw/live
  surface identity was unambiguous. Separately, full binding failed derived
  patch-hash/runtime-roundtrip exactness: immutable authored streams and face
  order were exact, but raw-derived paired-XZ SHA `98ef77e6...18bbae` differed
  from frozen authored SHA `917b7154...bcaf9`. Do not call this asset mutation;
  do not waive the failed binding gate without a separate provenance audit.
- A legacy D354 summary key named `d352_q5_evaluation_count=377` is mislabeled;
  it duplicates D354 invocation count. D352's immutable q5 count remains `0`.

### Operational and visualization completion

- Worker reached `SimulationApp.close` at `153.70643517s`, exited `0`, and was
  reaped. Markers `1242/1242` were valid; watchdog, retry, runtime exception,
  termination signal, and residual process group were absent.
- Actual Viewer hold was `120.007414064s` with `11072` UI updates and zero
  timeline interventions. Four `1280x720 RGBA` Isaac captures passed.
- RRD/RBL footer verify passed with Rerun `0.34.1`: dynamic samples `70`, meshes
  `131`, point/arrow/scalar/event rows `350/280/350/70`, exact entity/component
  paths `279/279`. The `4800x2800 RGBA` screenshot and all five original-resolution
  images passed manual inspection; manual PASS does not override science.
- GPU hardware contract: RTX 4090 Laptop, compute capability `8.9`, `76` SM,
  warp `32`. Telemetry `157/157` valid; active-time utilization mean/max
  `12.2357/42%`, VRAM max `7601MiB`, SM clock max `2385MHz`. Warp occupancy was
  not measured, and no clock/power/persistence/kernel/physics tuning occurred.
- One nonfatal startup log line said `Failed to clone in Fabric`; the subsequent
  environment, 70-row science, Viewer/RRD, attestation, and clean close completed.
  Preserve the line, but do not classify it as this run's failure cause.

### Final authority

- Final verdict: `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`.
- `completion_pass=true`, `scientific_contract_pass=false`, controlled physics
  steps `0`, `g0a_pass=false`; target/IK/path and physics configuration unchanged.
- Completion / measurement / moving-binding / attestation SHA-256:
  - `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`
  - `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed`
  - `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15`
  - `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`

## Active Case: none — separate approval required

- D354 is immutable and must not be rerun or overwritten.
- The result does not certify barrel-first/current-pose grasp, but also does not
  prove grasp impossible, force closure failure, or target/IK repair necessity.
- Narrowest proposed next action is a new forward-only offline/no-Isaac audit of
  derived moving-jaw patch hash provenance using only frozen authored/raw streams.
  It has no assigned case ID and is not approved.
- A cap/rim boundary discriminator and any target/IK/path change are later,
  separate authorization boundaries. Do not bundle or implement them now.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, q0-q5/object initial state,
  gates/tolerances, material, mass, actuator, renderer, solver, or physics settings.
- Do not run settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion.
- Do not introduce a post-hoc tolerance to relabel the cap/rim endpoint.
- Do not substitute vertex-only Qhull for callback topology or Rerun Float32
  display copies for canonical Float64/callback evidence.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D354 evidence is immutable.
  Hardware control, B200/SSH, `/half-clone`, unapproved commit/push are forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D354; ledger tail
2. `claudedocs/session_20260716_grasp_g0a_d354_current_pose_q5_closure_science_resume.md`
3. D354 `d354_completion_summary.json`,
   `d354_zero_step_closure_geometry_measurement.json`,
   `d354_moving_jaw_surface_binding.json`, `d354_zero_step_science_attestation.json`,
   and `d354_supervisor_audit.json` in the D354 output folder
4. `claudedocs/session_20260715_grasp_g0a_d353_timeline_pause_pending_state_commit_bridge.md`
5. `claudedocs/session_20260715_grasp_g0a_d352_d351_validate_phase_localization_watchdog.md`
6. D351 original/repair sessions and external termination audit
7. D348-D350 sessions referenced by DECISIONS D348-D350

## Operational Storage and Git

- User-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` remained
  read-only exact; do not modify it or use it as scientific authority.
- `HEAD == origin/master == b7beb91997859a5ddb2b0407388e80aed45898dc` remains the
  base. The worktree is intentionally dirty only for D354 code, forward-only
  evidence, and current-state updates. No D354 commit or push was performed.
