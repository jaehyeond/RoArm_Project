# START_HERE.md

Last updated: 2026-07-15 KST. D350 remains the last completed scientific and
observability case. D351 remains an operational pre-science STOP. D352 completed
its single localization run and found a deferred timeline-state verification
defect, but did not reproduce D351's 3693.302s long run or enter q5 science.
`g0a_pass=false`, `settle_authorized=false`, and target/IK remains frozen.

## Current Truth

- The active research pivot remains cylinder grasp-track G0a
  (`radius=0.017m`, `height=0.090m`). G0b close/lift, settle, ten-trial,
  PPO/RL, VLA, ladder, real hardware, and B200 are out of scope.
- q5 convention is fixed: URDF `q5=0` is CLOSED; frozen sim OPEN is
  `q5=1.5413rad`. Nominal HOME is `[0,0,90,0,0,0]deg`, but D347 measured a
  HOME-near q5=0 CLOSED state, not exact HOME.
- D348 proved the PhysX property volume comparator must use the callback's
  original polygon topology, not a newly generated vertex-only Qhull envelope.
  The corrected representation gate is `256/256` channels and `128/128` parts.
- D349 measured the frozen `(radial,tangent)=(7,11)mm`, sign `-1`, seed `33201`,
  HOME-seeded position-only IK target before physics. Raw/live gaps were
  link5 `4.2726455336/4.2727365803mm` and moving gripper
  `11.1750883746/11.3402623263mm`. These do not prove contact or grasp.
- D350 bound the actual connected fixed-jaw `link5` surface and measured it
  without inventing an alignment tolerance. Its verdict is
  `D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED`, with
  `aligned_pass=null`, not an alignment/grasp PASS.
- D352 proved D351 attempt2's repeated `timeline.pause()` checks were synchronous
  checks of a deferred state change: all five snapshots remained PLAY while
  step/time/clock/joint/object state stayed exact. This is a pre-q5 control STOP,
  not evidence that physics advanced or that closure geometry passed/failed.

## Last Scientific + Observability Case: D350

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d350/`. Actual fixed-jaw
  connected surface was `7,250` faces / `3,519` vertices; centerline tangent/height
  offsets were `-24.055688/-20.360856mm`, with `aligned_pass=null`.
- Real Viewer ran `180.007254s` / `8,222` updates at zero controlled steps; six
  Isaac captures and repaired RRD/RBL/manual inspection passed. Completion SHA:
  `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`.

## D351 Closed Outcome

- Immutable attempt1 root: `claudedocs/runtime_logs/grasp_track/g0a_d351/`.
  Forward-only attempt2 root: `claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/`.
- attempt1 reached corrected live `128/128`, saw PLAY, and stopped before q5.
  attempt2 parameter exact + preflight `20/20` + real GUI passed, but emitted no
  live/bridge/raw/science artifact before user-approved SIGTERM at Kit
  `3693.302s`; it was not retried.
- Verdict `D351_ATTEMPT2_PRE_SCIENCE_RUNTIME_LONG_RUN_STOP`; q5=`0` by program
  order, controlled steps=`null`, science/geometry/current-pose/target-IK=`null`.
  External audit SHA:
  `af17995b40d5818055388f97e38cbb50f0895f3a2aa4d2cb7f5cf1df3b6166fe`.

## D352 Closed Outcome

- Forward-only output:
  `claudedocs/runtime_logs/grasp_track/g0a_d352/`. Prepare and one real
  `headless=false`, `DISPLAY=:1`, `cuda:0` validate were run; no retry occurred.
- AppLauncher took `18.496625s`, `_make_runtime_env` `3.228809s`, reset
  `0.038351s`, corrected audit `0.096625s`, live builder `1.882320s`, payload
  write `0.024480s`, and raw binding `1.937505s`. Localization completed at worker
  elapsed `27.093245s`; live trace was `128/128`, body subchecks `64+64`, raw PASS.
- Thus D351's `3693.302s` long run did not reproduce. The deterministic-block
  hypothesis for the registered phases is unsupported, but D351's historical
  function-level cause remains `null` because it has no marker or stack dump.
- The exact five bridge snapshots all had `timeline_playing=true` even after
  three pause calls at each repair boundary. Meanwhile custom counter stayed `0`,
  timeline time, SimulationContext clock, and joint/object Float32 bits were exact,
  and `/app/player/playSimulations=false` was readable.
- Installed `omni.timeline 1.0.14` documents that state changes apply in the next
  frame; `Timeline.commit()` applies pending state. D351 attempt2 called neither a
  frame update nor commit before immediate `is_playing()`. The corrected operational
  verdict is `D352_LOCALIZATION_COMPLETE_TIMELINE_PAUSE_PENDING_STATE_STOP`.
- The raw supervisor string `D352_LOCALIZATION_EXCEPTION_STOP` is a catch-all
  classifier defect: worker exit was `0`, watchdog false, runtime exception null.
  The raw case PASS remains false.
- D352 q5 count is `0`; controlled physics steps are `null` because the PAUSE
  bridge failed. Science/geometry/current-pose/target-IK verdicts remain `null`,
  `g0a_pass=false`. No moving-surface measurement, sweep, Viewer, RRD, or RBL exists.
- GPU telemetry was 31/31 valid: device active-time min/mean/max
  `0/3.870968/15%`, VRAM `2052/3863.968/7430MiB`, worker CPU
  `4/127.935/724.7%`. This is device-level activity, not warp occupancy or a
  causal bottleneck. A zero-step single-env audit has no batched GPU workload to
  fill 76 SM, and GPU tuning is not a repair for pending timeline state.
- Postrun classification audit SHA-256:
  `92c186a7a4175101e7a3890f6bedf4cb6125bc5a78f13f38b79004a9b6035594`.

## Next Authorization Boundary

- There is no approved executable case now. The narrow candidate is D353
  `[timeline_pause_pending_state_commit_bridge]`, with exactly one new operational
  variable: `explicit_timeline_commit_after_pause`.
- D353 would prove PAUSE plus unchanged timeline time, SimulationContext clock,
  custom step counter, joint/object Float32 bits, and q5 count zero. It must not
  call `simulation_app.update()` or `forward_one_frame()` and must not run q5
  science, geometry, target/IK, Viewer/Rerun, settle, or promotion.
- Only after that zero-step bridge PASS may closure science be proposed in another
  forward-only case. The user's q5-science intent is recorded, but the failed
  bridge prevents using that authorization before a separately approved repair.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, tolerances, material,
  actuator, mass, or physics settings. Do not run settle, ten-trial, G0b,
  RL/PPO, VLA, or ladder promotion.
- Do not treat GUI launch, preflight PASS, CPU/GPU activity, an open Viewer, or
  signal-followed exit `0` as proof that geometry or visualization completed.
- Do not treat repeated `timeline.pause()` plus an immediate `is_playing()` query
  as committed PAUSE. Timeline state is deferred; any commit-based repair must
  separately prove zero timeline/SimulationContext/state advancement.
- Do not call D347's vertex-only Qhull value callback-topology volume, and do not
  substitute Rerun Float32 display copies for canonical Float64/callback data.
- Do not use the legacy 8mm proxy or a near-radial PCA axis alone as actual
  connected-jaw alignment authority.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D352 run evidence is immutable.
  Hardware control, `JOINT_LIMITS` removal, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D352; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d349_frozen_open_jaw_target_live_distance_gate.md`
3. `claudedocs/session_20260714_grasp_g0a_d348_physx_property_query_volume_semantics.md`
4. `claudedocs/session_20260714_grasp_g0a_d350_fixed_jaw_geometry_viewer.md`
5. `claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json`
   and `d350_manual_visual_inspection.json` in the same folder
7. `claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md`
8. `claudedocs/session_20260715_grasp_g0a_d351_timeline_pause_repair.md`
9. `claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json`
10. `claudedocs/session_20260715_grasp_g0a_d352_d351_validate_phase_localization_watchdog.md`
11. `claudedocs/runtime_logs/grasp_track/g0a_d352/d352_postrun_classification_audit.json`

## Operational Storage Sidecar

- The D242 raw PNG set was externally archived and verified before approved
  local raw-only cleanup. Restore only through
  `raw_predelete_manifest_20260714.tsv` verification.
- User-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` is a
  non-scientific sidecar. Do not modify or infer D351 science from it.

Base HEAD and local `origin/master` are
`c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61` (D351 operational STOP state).
The worktree was clean before D352 implementation and now contains only the
forward-only D352/state changes plus preexisting evidence. Commit/push remains
user-request-only and was not performed.
