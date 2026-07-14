# START_HERE.md

Last updated: 2026-07-14 KST. D346 is complete with verdict
`D346_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`. This is an instrumentation
activation-order STOP before callback 1/256, not a collision-geometry rejection.
The immutable D344 attempt3 remains unclassified in fresh PhysX runtime.
`g0a_pass=false`; settle, G0b, RL, and ladder remain blocked.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out of
  scope.
- q5 convention: URDF `q5=0` = CLOSED; sim OPEN is `~1.541-1.571rad`.
  Frozen target is q5 `1.5413rad`, `(radial,tangent)=(7,11)mm`, tangent sign
  `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw target family: `2,560/2,629` raw-clear candidates;
  frozen-target raw clearances link5/gripper `+4.2726/+11.1751mm`. The stale
  cooked link5 hull caused the historical `38.861N` step-0 impulse.
- D339 attempt2 proved independent cold-cook equality but found 13/128 live
  fidelity failures. D340 captured fixed-point candidates; D342 repaired the
  coordinate-stream proof; D343 proved typed float32 bits `0x38d1b717` on
  128/128 parts.
- D344 authored immutable attempt3 with 13 changed and 115 preserved parts, but
  remains historical FAIL because its metadata comparator used address-bearing
  PXR `repr`. D345 repaired that comparator across two independent processes;
  D345 PASS did not reclassify D344 or certify live geometry.
- D341 Rerun lifecycle remains mandatory for geometry/pose/contact/runtime:
  finalized RRD/RBL, exact entities/timelines/components, headless screenshot,
  and actual inspection. Rerun is observability, never numerical authority.

## D346 Verified Result

- New variable: `[attempt3_fresh_live_representation_validation]` (measurement
  only). New physical variables `0`; parameter increases/changes/threshold
  relaxations/decomposition changes `0/0/0/0`; asset writes `0`.
- A first managed execution hid CUDA and stopped while constructing
  `SimulationContext`. It produced callback `0`, physics `0`, no scientific
  outputs, and was preserved as non-scientific evidence. A hash-chained reactive
  amendment allowed one effective GPU execution without changing the experiment.
- Effective preflight PASS: RTX 4090 visible; frozen source hashes/inventories,
  numpy `1.26.0`, psutil `5.9.8`, Rerun `0.34.1`, AppLauncher, stage/sensor,
  `metersPerUnit=1.0`, and retained raw source were exact.
- The effective run stopped before the first callback with
  `ModuleNotFoundError: No module named 'omni.physxassetvalidator'`.
  Callback witness `0/256`; classified live parts `0/128`; D337 controls and
  frozen raw/live distance query were correctly skipped; body distances are
  `null`. Simulation counter `0->0`; controlled physics steps `0`.
- This is not “128 parts failed.” The extension is installed as
  `omni.physx.asset_validator` v`107.3.26` and declares Python module
  `omni.physxassetvalidator`. D340 imports the module before enabling the
  extension; D339's working sequence enables/verifies it before import.
- D344 attempt3 and all earlier inputs remained immutable. D344 stays
  `D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP`; D345 stays PASS.

## D346 Rerun / Visual Result

- Footer-complete RRD/RBL and screenshot exist, but the exact machine contract
  failed: frames `6/6`, body frames `2/2`, meshes `266/522`, scalars
  `1,040/1,040`, events `132/132`, non-system entities `1,588/2,100`.
- Exactly 256 callback-derived instance/prototype meshes (and 512 related
  entities) are absent. Placeholder scalar/event rows cannot replace geometry.
- Original-resolution inspection found eight configured panels, but both bodies'
  live-instance/prototype panels lacked callback part geometry; the table showed
  unavailable values/WARN rows and viewer notifications obscured part of one
  panel. Manual visual completion is FAIL.
- Decision PNG correctly says pre-physics STOP and confirms that no partial/empty
  live union was queried. Its TCP `0.817895mm` and jaw-tangent `2.148675deg`
  values are IK alignment errors, not collision clearances.
- Completion summary SHA-256:
  `98a0c126824a27e7651ea2fe352394eb8829a4bf1137532e180ed7ae5629bece`.
- Postrun root-cause JSON is a pre-manual-inspection diagnosis snapshot. Final
  manual/completion authority is `d346_manual_visual_inspection.json` plus
  `d346_completion_summary.json`.

## Active Case — D346 Closed / D347 User Approval Pending

- Do not edit or rerun D346. The next recommended case is D347 measurement-only
  `[physx_asset_validator_activation_order]`.
- D347's only allowed repair is: in a new wrapper/fresh process, record initial
  extension state -> enable exact ID `omni.physx.asset_validator` -> verify true
  -> import `omni.physxassetvalidator` and record module origin/API -> then call
  the frozen D346 callback/part/target/Rerun contract once.
- D347 must keep D344/D346 immutable and preserve callback `256`, parts `128`,
  q5 `1.5413`, target `(7,11)mm`, all decomposition settings/tolerances, and
  controlled physics `0`.
- Forbidden shortcuts: manual `PYTHONPATH`, private `.so` import, whole PhysX
  bundle/custom experience, `simulation_app.update()`, fallback/retry, asset
  recook/rewrite, tolerance relaxation, or parameter tuning.
- Even a future D347 PASS only makes a separately approved fresh settle case
  eligible. It does not set `g0a_pass=true`.
- Completion summary's `next_case_requires_separate_approval=null` refers only
  to the unopened post-PASS settle gate; it does not negate the new D347 repair
  recommendation from the postrun diagnosis.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D344-D346; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d346_fresh_live_attempt3_validation.md`
3. `claudedocs/runtime_logs/grasp_track/g0a_d346/d346_completion_summary.json`
4. `claudedocs/runtime_logs/grasp_track/g0a_d346/d346_postrun_root_cause_audit.json`
5. `claudedocs/runtime_logs/grasp_track/g0a_d346/d346_raw_live_measurement.json`
6. `claudedocs/runtime_logs/grasp_track/g0a_d346/d346_rerun_validation.json`
7. `claudedocs/session_20260714_grasp_g0a_d345_deterministic_usd_metadata_comparator.md`
8. `claudedocs/session_20260714_grasp_g0a_d344_attempt3_fixed_point_collision_geometry.md`
9. `claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md`

## Do Not Trust As Current / Durable Boundaries

- `HANDOFF.md` and `TASKS.md` are stale. q5 `0` means CLOSED.
- D346 `d346_rerun_manual_visual_inspection.json/.md` are preserved supplemental
  files created under a guessed filename. Registered authority is
  `d346_manual_visual_inspection.json/.md` plus `d346_completion_summary.json`.
- Never hash default PXR `repr(...)`; canonicalize typed content and prove
  cross-process determinism.
- Extension-owned Python modules in a minimal fresh Kit process must be imported
  only after the exact extension is enabled and verified. Installed files alone
  do not prove that the namespace is active.
- Rerun placeholders, empty panels, or nonzero files never substitute for missing
  callback geometry or an exact completion contract.
- D338 attempt1, D339 attempt2, D340, D342, D343, D344, D345, and D346 are
  immutable: no overwrite, silent rerun, retroactive PASS, or promotion.
- `JOINT_LIMITS` removal, hardware control, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

HEAD remains `b09b62e0ffad919b9bdc1bb6155de2f662f2ab5c` (`D345 및 roarm cube10cm
render 및 fair데이터 이전`). D346 state/code/runtime outputs are uncommitted;
commit/push is user-request-only.
