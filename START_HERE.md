# START_HERE.md

Last updated: 2026-07-15 KST. D350 remains the last completed scientific and
observability case. D351 is an operational pre-science STOP, D352 localized its
pending Timeline state, and D353 proved the explicit-commit zero-step control
repair in one run. D353 did not execute q5 science. No next executable case is
authorized; `g0a_pass=false`, `settle_authorized=false`, and target/IK is frozen.

## Current Truth

- The active research pivot remains cylinder grasp-track G0a
  (`radius=0.017m`, `height=0.090m`). G0b close/lift, settle, ten-trial,
  PPO/RL, VLA, ladder, real hardware, and B200 are out of scope.
- q5 convention is fixed: URDF `q5=0` is CLOSED; frozen sim OPEN is
  `q5=1.5413rad`. Nominal HOME is `[0,0,90,0,0,0]deg`, but D347 measured a
  HOME-near q5=0 CLOSED state, not exact HOME.
- D348 proved the PhysX property-volume comparator must use callback polygon
  topology, not a new vertex-only Qhull. The corrected gate is `256/256`
  channels and `128/128` parts.
- D349 frozen-OPEN raw/live distances were link5
  `4.2726455336/4.2727365803mm` and moving gripper
  `11.1750883746/11.3402623263mm`. They do not prove contact or grasp.
- D350 measured the actual connected fixed-jaw `link5` surface and completed
  the real Isaac Viewer plus 64+64 collider visualization. Its verdict is
  `D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED`, with
  `aligned_pass=null`, not an alignment/grasp PASS.
- D352 showed that repeated `timeline.pause()` followed by immediate Boolean
  checks only read the old committed PLAY state. Time, clock, joint, object,
  and custom-step state stayed exact; this was a deferred-state control defect.
- D353 applied that one pending state with one main-thread `Timeline.commit()`.
  The state changed `PLAY -> pending PLAY -> PAUSE`, while all registered
  time/world/state sentinels stayed exact and controlled physics steps were `0`.
  This repairs the zero-step control bridge only; it is not q5 science.

## Last Scientific + Observability Case: D350

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d350/`. The connected fixed
  jaw was `7,250` faces / `3,519` vertices; centerline tangent/height offsets
  were `-24.055688/-20.360856mm`, with `aligned_pass=null`.
- The real Viewer ran `180.007254s` / `8,222` updates at zero controlled steps;
  six Isaac captures and repaired RRD/RBL/manual inspection passed. Completion
  SHA-256: `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`.

## D351 Closed Outcome

- attempt1 reached corrected live `128/128`, observed PLAY, and stopped before
  q5. Forward-only attempt2 passed exact parameters and preflight `20/20`, then
  emitted no live/bridge/science artifact before user-approved SIGTERM at Kit
  `3693.302s`; it was not retried.
- Verdict `D351_ATTEMPT2_PRE_SCIENCE_RUNTIME_LONG_RUN_STOP`; q5 count `0`,
  controlled steps `null`, science/geometry/current-pose/target-IK `null`.
  External-audit SHA-256:
  `af17995b40d5818055388f97e38cbb50f0895f3a2aa4d2cb7f5cf1df3b6166fe`.

## D352 Closed Outcome

- One real `headless=false`, `DISPLAY=:1`, `cuda:0` localization run completed
  at worker elapsed `27.093245s`; live was `128/128`, body `64+64`, raw PASS,
  watchdog false, and no retry occurred. D351's long run did not reproduce.
- All five bridge snapshots remained PLAY even after pause requests, while
  counter/time/SimulationContext/joint/object bits stayed exact. Corrected
  verdict: `D352_LOCALIZATION_COMPLETE_TIMELINE_PAUSE_PENDING_STATE_STOP`.
- q5 count `0`; controlled steps `null`; scientific/geometry/current-pose/
  target-IK verdicts `null`; `g0a_pass=false`. GPU `31/31` active-time samples
  were device activity, not warp occupancy or a causal bottleneck.
- Postrun audit SHA-256:
  `92c186a7a4175101e7a3890f6bedf4cb6125bc5a78f13f38b79004a9b6035594`.

## D353 Closed Outcome

- Forward-only output: `claudedocs/runtime_logs/grasp_track/g0a_d353/`.
  Prepare and exactly one real `headless=false`, `DISPLAY=:1`, `cuda:0`
  validate ran; no retry occurred. The only new variable was
  `explicit_timeline_commit_after_pause`.
- Supervisor/worker preflight passed `25/25` and `35/35`. The worker reached
  bridge authority at elapsed `21.847929s`, reached `SimulationApp.close` at
  `22.440569s`, and was reaped with exit `0`; watchdog false, runtime exception
  absent, and the 17-file success inventory was exact.
- Exactly one commit attempt/call caused one discriminating
  `PLAY -> pending PLAY -> PAUSE` transition and one PAUSE callback inside the
  commit window on the caller MainThread. Live/raw boundaries were already
  PAUSE and correctly made no extra commit.
- All 9 detailed + 5 canonical snapshots kept timeline time
  `0.029999999329447746`, SimulationContext time/index
  `0.009999999776482582/2`, custom counter `0`, joint/object bits, geometry
  counter `0`, and q5 evaluation/state-write/trap counts `0` exact. Live was
  `128/128` (`64+64`), and raw payload reproduced D352 exactly.
- After summary fsync and exact reread, the separate attestation authoritatively
  recorded D353 controlled physics steps `0`. D351/D352 historical controlled
  values remain `null`.
- Final verdict: `D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`.
  scientific/geometry/current-pose/grasp/target-IK verdicts remain `null`,
  `g0a_pass=false`; no moving-surface measurement, q5 sweep, Viewer, RRD, or RBL
  was produced.
- GPU telemetry was `26/26` valid: utilization min/mean/max `0/2.6923/21%`,
  VRAM `2052/4013/7437MiB`, worker CPU `9.3/153.644/740.2%`. This does not
  measure warp occupancy; a one-env zero-step control case has no batched
  workload with which to fill 76 SM.
- Commit attestation / final supervisor SHA-256:
  `4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5` /
  `65c57e69f017d7d7afbb5fd03b10b56e87bb1bbc442b1351a25c18a0a55a31a5`.

## Current Authorization Boundary

- No D354 code, output path, prepare, or validate is authorized yet.
- The narrow next candidate is a new forward-only current-pose q5 closure-science
  case that inherits the D353 attested PAUSE bridge. Its exact science variables,
  q5 sampling/sweep, moving-surface measurement, visualization, watchdog, and
  artifact gates must be preregistered before one run.
- D353 itself is immutable and must not be rerun or overwritten. q5 science can
  start only after a new explicit user approval issued after the D353 briefing.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, q0-q5/object initial
  state, gates/tolerances, material, actuator, mass, renderer, solver, or physics
  settings without a separately approved case.
- Do not run settle, ten-trial, G0b, RL/PPO, VLA, or ladder promotion. Do not
  call D353 a geometry/contact/grasp PASS.
- Deferred timeline commands require committed-state evidence. In a zero-step
  case, any apply mechanism must also prove exact timeline/SimulationContext/
  joint/object/state non-advancement.
- Do not substitute vertex-only Qhull volume for callback topology or Rerun
  Float32 display copies for canonical Float64/callback evidence.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D353 evidence is immutable.
  Hardware control, `JOINT_LIMITS` removal, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D353; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d348_physx_property_query_volume_semantics.md`
3. `claudedocs/session_20260714_grasp_g0a_d349_frozen_open_jaw_target_live_distance_gate.md`
4. `claudedocs/session_20260714_grasp_g0a_d350_fixed_jaw_geometry_viewer.md`
5. `claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md`
6. `claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md`
7. `claudedocs/session_20260715_grasp_g0a_d351_timeline_pause_repair.md`
8. `claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json`
9. `claudedocs/session_20260715_grasp_g0a_d352_d351_validate_phase_localization_watchdog.md`
10. `claudedocs/runtime_logs/grasp_track/g0a_d352/d352_postrun_classification_audit.json`
11. `claudedocs/session_20260715_grasp_g0a_d353_timeline_pause_pending_state_commit_bridge.md`
12. D353 `d353_timeline_commit_bridge_attestation.json` and
    `d353_supervisor_audit.json` in `claudedocs/runtime_logs/grasp_track/g0a_d353/`

## Operational Storage Sidecar

- The D242 raw PNG set was externally archived and verified before approved
  local raw-only cleanup. Restore only through
  `raw_predelete_manifest_20260714.tsv` verification.
- User-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` is a
  non-scientific sidecar. Do not modify it or infer D353 science from it.

Base HEAD and local `origin/master` were cross-verified clean at
`1f235b8a310afeb9f4f6734d69aba2a5430b7602` (`D352수정`) before D353 edits.
The prior `c2cfa5f...` wording was stale after the user's D352 push. D353
commit/push remains user-request-only and is not authorized.
