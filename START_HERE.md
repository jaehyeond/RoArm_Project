# START_HERE.md

Last updated: 2026-07-13 KST. D341 complete:
`D341_RERUN_OBSERVABILITY_COMPLETION_CONTRACT_PASS`. This is an observability
contract PASS only. D340 remains
`D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`; `g0a_pass=false`, attempt3
absent, and G0b/RL/ladder blocked.

## Current Truth

- Active pivot remains grasp-track G0a on the cylinder (r `0.017m`, h
  `0.090m`). Cube repair, G0b close/lift, PPO/RL, VLA, randomization, real
  RoArm, and B200 are out of scope.
- D337 repaired the q5 convention: URDF `q5=0` = CLOSED and sim OPEN is
  `~1.541-1.571rad`. At `q5=1.5413`, `2,560/2,629` targets passed the raw-clear
  grid; frozen `(7,11)mm` had link5/gripper clearances
  `+4.2726/+11.1751mm`. Its settle still produced a link5 `38.861N` step-0
  impulse and object disturbance because physics uses the inflated cooked hull.
- D338 attempt1 proved global cook statistics do not witness the synchronous
  explicit request path. D339 repaired the callback witness and produced the
  immutable attempt2, but live recook fidelity failed on 13 of 128 parts.
- D340 captured one fixed-point candidate through both live channels for those
  13 parts. All 26 callbacks and every fixed-point subcheck passed except the
  authored hash predicate: it compared direct authored Vec3f bits with a
  post-transform float64 stream. The transform delta was only `2.22e-16m`, but
  exact hashes changed on `13/13`; therefore D340 correctly stopped before
  attempt3, validation, cooked-union query, or physics.
- D341 repaired the project-wide Rerun completion contract without rerunning
  D340 or changing geometry. The good RRD/RBL are footer-complete
  (`742,647/96,376` bytes) and passed exact `254` non-system entities, four
  timelines, and required component schemas. Subject: `52` meshes, `143`
  scalars, `67` events.
- The finalized-copy negative control removed `4,096` bytes and was rejected
  with `footer_manifest_present=false` plus the explicit footer error.
- The `2400x1400` logical (`4800x2800` raster) screenshot was actually opened:
  all eight independent link5/gripper source/instance/prototype/candidate
  panels, the metric Dataframe, and INFO/WARN events were visible. Pixels were
  not used as bit-exact evidence.
- D340 remained exact across `33 -> 33` files with digest
  `ce77a75e9ee8ba559e57bf443e4eee587352498bbb154f91f06bb81b4462c8ab`.
  Existing/decomposition/tolerance parameter changes, collision writes, and
  physics steps in D341 were all `0`.

## Active Case: G0a / D341 Complete, D342 Approval Pending

- 이번 case의 신규 변수 was exactly one measurement-only variable:
  `[rerun_observability_completion_contract]`.
- Final D341 authority:
  `claudedocs/runtime_logs/grasp_track/g0a_d341/d341_rerun_observability_completion_summary.json`.
- The automated summary intentionally remains
  `AUTOMATED_PASS_MANUAL_INSPECTION_PENDING`; it was not overwritten. The
  separate manual report and final summary close the completion gate.
- D340's actual RRD sha256 is
  `8eb3d6130330334b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47`.
  The preserved D340 session contains a 63-character typo. Its PNG was
  inspected; its RRD was generated but lacked a footer/scientific subject and
  was not visually completion-certified. Do not reuse the old broad
  “PNG/RRD inspected” wording.
- Output: `claudedocs/runtime_logs/grasp_track/g0a_d341/`.
- HEAD remains `2c8a25f689bd7c7f3927a956755c8642764d81`; worktree changes are
  intentionally uncommitted. Commit/push only on explicit user request.

## Next Concrete Action

Stop for user approval. Recommended physical next case is D342, a separately
pre-registered `authored_geometry_frame_contract` repair:

1. Reuse and pin immutable D340 callback/candidate evidence; do not rerun D340.
2. Compare the direct authored Vec3f point stream with the D339 manifest before
   any coordinate transform.
3. Use body-mapped coordinates only for registered numerical containment and
   proximity gates.
4. Only after that proof passes, separately authorize the still-absent attempt3
   authoring and fresh validation. Collision-asset mutation requires explicit
   user approval.

No tolerance, decomposition, target, controller, solver, or physics change is
justified. Do not run settle or 10-trial before representation certification.

Reserve only:

1. Onset-metric hardening for the step-0 impulse row, reactive within a future
   settle case rather than standalone.
2. `r>17mm` grasp-depth redefinition; currently unnecessary.

## Rerun Completion Contract

- Required for verdicts involving geometry, pose/frames, collision/contact,
  trajectory, or synchronized sensor time. Pure file/hash/schema audit may
  omit it only with written justification.
- Order is mandatory: exact version pins -> pre-log footer-enabled sink ->
  actual scientific subject and named coordinate frames/full timeline -> flush
  and finalize -> footer/exact entity/timeline/component validation -> fixed
  embedded Blueprint plus verified RBL export -> headless decision screenshot
  -> separate actual inspection report.
- Non-empty, decodable, loadable, or screenshot-created does not mean
  “inspected” or “complete”. Automated evidence must remain pending until the
  separate manual gate closes.
- Rerun is observability, not numerical authority. Original callback arrays and
  canonical JSON/hashes decide bit equality; Float32 display geometry must not
  be hashed back into a scientific gate.
- Cook cases expose source/instance/prototype/candidate independently. Physics
  cases log every executed step plus object/tool state, decision scalars, and
  contact points/force arrows. Training trackers own optimizer-scale history;
  Rerun owns sampled spatial rollout evidence.
- Directly invoked `sim_scripts/*.py` files importing project packages must
  bootstrap the resolved repo root before those imports.

## Must Read First

1. `AGENTS.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D334-D341)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d341/d341_rerun_observability_completion_summary.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d341/d341_manual_visual_inspection.json`
8. `claudedocs/HOWTO_viz_debug.md`
9. `claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md`
10. `claudedocs/runtime_logs/grasp_track/g0a_d340/d340_capture_summary.json`
11. `claudedocs/runtime_logs/grasp_track/g0a_d340/d340_capture_postrun_root_cause_audit.json`

## Durable Do-Not-Repeat Rules

- `HANDOFF.md` and `TASKS.md` are stale. Memory is an index, not evidence.
- q5 convention: `q5=0` = CLOSED; sim OPEN = `~1.541-1.571rad`.
- BVH scalar on colliding meshes is not penetration depth; use certified
  contact-level EPA. Distinguish raw mesh, mathematical hull, mirror cook, and
  live cook.
- Global zero cooking-stat deltas do not witness synchronous explicit cooks.
  Callback-first independent cooks and canonical geometry equality are needed.
- Callback-repeatable decomposition does not prove live shape binding/fidelity.
  USD disabled inventory does not by itself prove runtime collider exclusion.
- Bit-exact geometry hashes require the same coordinate/value stream. Prove
  authored identity before mapping, then use explicit mapped-geometry gates.
- D338 attempt1, D339 attempt2, and D340 evidence are immutable. No overwrite,
  D340 rerun, attempt3, threshold relaxation, or parameter increase.
- `JOINT_LIMITS` removal, B200/SSH/pull, `/half-clone`, hardware control, and
  unapproved commit/push remain forbidden.
