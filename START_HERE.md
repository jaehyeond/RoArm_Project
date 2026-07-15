# START_HERE.md

Last updated: 2026-07-15 KST. D350 remains the last completed scientific and
observability case. D351 is closed as an operational pre-science STOP, not as a
closure-geometry PASS or FAIL. There is no approved active execution case.
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

## Last Completed Case: D350

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d350/`.
- The actual fixed-jaw component contained `7,250` faces and `3,519` unique
  vertices. At the cylinder radial station its centerline offset was tangent
  `-24.055688mm`, height `-20.360856mm`, radial residual `-0.471744mm`.
- The actual nearest fixed-jaw witness was `+15.894261mm` above cylinder center;
  the legacy 8mm proxy differed from it by `17.027401mm`.
- The real Isaac Viewer ran `180.007254s` over `8,222` UI/render updates with
  timeline paused, counter `0->0`, state drift `0`, and physics steps `0`.
- Six `1280x720 RGBA` Isaac captures showed the assembled robot, actual PhysX
  collider display, and colored link5 `64` + gripper `64` parts. Repaired
  RRD/RBL, exact entity/component checks, and manual visual inspection passed.
- Completion summary SHA-256:
  `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`.

## D351 Closed Outcome

- Immutable attempt1 root: `claudedocs/runtime_logs/grasp_track/g0a_d351/`.
  Forward-only attempt2 root: `claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/`.
- Approved question: before changing target/IK, zero-step measure whether the
  actual moving-jaw inside surface first contacts the cylinder barrel while q5
  closes from OPEN to CLOSED in the frozen D350 pose.
- New scientific variables were limited to
  `[moving_jaw_actual_contact_surface_binding, frozen_pose_q5_closure_sweep]`;
  new physical variables were `0`. Assets, `64+64` decomposition, q0-q4/object,
  target/IK/path, gates, material, actuator, and physics settings stayed frozen.
- attempt1 launched the real `headless=False` Isaac GUI and wrote corrected live
  binding `128/128`, then observed timeline `PLAY` and stopped before q5 sample 1.
  Its q5 evaluation count is `0`; measurement, Viewer/RRD, and scientific verdict
  are absent. Its controlled-physics-step field remains the recorded `null`.
- Reactive attempt2 reproduced the exact attempt1 parameter SHA-256
  `98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2`.
  Fresh validate preflight was `20/20 PASS`; a real RTX 4090 GUI launched with
  `headless=false` and `DISPLAY=:1`.
- Kit logged app-ready `13.360s` and startup-complete `15.953s`, but no attempt2
  live-binding, five-snapshot bridge, raw binding, measurement, D351 Viewer
  capture, RRD, or runtime-exception artifact appeared before shutdown elapsed
  `3693.302s`.
- After more than 60 minutes, the user approved `SIGTERM`; Kit closed gracefully
  and the process/GPU context disappeared. Shell exit `0` from graceful close is
  not a scientific PASS. No automatic retry was made.
- attempt2 q5 evaluation count is `0` by exact program order: the first q5 call
  occurs only after the missing live-binding write and raw prerequisite.
  Controlled physics steps remain `null` because the zero-step bridge never
  completed; they must not be backfilled as `0`.
- Final operational verdict:
  `D351_ATTEMPT2_PRE_SCIENCE_RUNTIME_LONG_RUN_STOP`. Closure geometry,
  current-pose grasp feasibility, and target/IK-repair justification are all
  `null`. D351 has no completed Rerun/Viewer/collider inspection contract.
- External termination audit SHA-256:
  `af17995b40d5818055388f97e38cbb50f0895f3a2aa4d2cb7f5cf1df3b6166fe`.
  Kit log SHA-256:
  `b4eb319c2b19638f6e263e6b654fb517f494a847e073b573c24d4563e7f72e20`.

## Active Case / Next Authorization Boundary

- No case is currently approved for implementation or execution. D351 attempt1
  and attempt2 are immutable STOP evidence and must not be rerun or overwritten.
- The narrow next candidate, requiring separate user approval, is D352
  `[d351_validate_phase_localization_watchdog]`. It would add only forward-only
  phase markers and a bounded wall-clock watchdog around `_make_runtime_env`,
  reset, corrected audit, live parts `0..127`, and the zero-step bridge.
- D352 is localization-only: no q5 science sample, geometry verdict, target/IK
  change, physics step, gate change, or promotion is authorized by this entry.
  Resuming the q5 closure experiment after D352 requires another explicit user
  approval.
- A later target/IK geometry-repair case may be considered only if a completed
  closure discriminator supplies evidence. D351 supplies no such evidence.

## Frozen Boundaries

- Do not change assets, decomposition, target/IK/path, tolerances, material,
  actuator, mass, or physics settings. Do not run settle, ten-trial, G0b,
  RL/PPO, VLA, or ladder promotion.
- Do not treat GUI launch, preflight PASS, CPU/GPU activity, an open Viewer, or
  signal-followed exit `0` as proof that geometry or visualization completed.
- Do not call D347's vertex-only Qhull value callback-topology volume, and do not
  substitute Rerun Float32 display copies for canonical Float64/callback data.
- Do not use the legacy 8mm proxy or a near-radial PCA axis alone as actual
  connected-jaw alignment authority.
- `HANDOFF.md` and `TASKS.md` are stale. D338-D351 evidence is immutable.
  Hardware control, `JOINT_LIMITS` removal, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D348-D351; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d349_frozen_open_jaw_target_live_distance_gate.md`
3. `claudedocs/session_20260714_grasp_g0a_d348_physx_property_query_volume_semantics.md`
4. `claudedocs/session_20260714_grasp_g0a_d350_fixed_jaw_geometry_viewer.md`
5. `claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json`
   and `d350_manual_visual_inspection.json` in the same folder
7. `claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md`
8. `claudedocs/session_20260715_grasp_g0a_d351_timeline_pause_repair.md`
9. `claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json`

## Operational Storage Sidecar

- The D242 raw PNG set was externally archived and verified before approved
  local raw-only cleanup. Restore only through
  `raw_predelete_manifest_20260714.tsv` verification.
- User-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` is a
  non-scientific sidecar. Do not modify or infer D351 science from it.

Base HEAD and local `origin/master` are
`cfd9e7501df89724c3cc2b1038fda05ce0d88e2f` (`D350`). The worktree was clean
before D351 preregistration. Commit/push remains user-request-only.
