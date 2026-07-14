# START_HERE.md

Last updated: 2026-07-14 KST. D347 is complete with verdict
`D347_G0A_FRESH_LIVE_REPRESENTATION_FAIL_STOP`. The D346 import-order defect is
repaired and callback evidence is complete, but one of 128 collision pieces failed
the frozen PhysX-volume cross-check. The target-distance query was therefore not
run. `g0a_pass=false`; settle, G0b, RL, and ladder remain blocked.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out of
  scope.
- q5 convention: URDF `q5=0` = CLOSED; sim OPEN is `~1.541-1.571rad`.
  Frozen target is q5 `1.5413rad`, `(radial,tangent)=(7,11)mm`, tangent sign
  `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw target family: `2,560/2,629` raw-clear candidates;
  its frozen-target raw clearances are link5/gripper `+4.2726/+11.1751mm`.
  These are D337 anchors, not D347 measurements.
- D339 attempt2 proved two-cook equality but found 13 surface failures and the
  link5 `part_045` volume discrepancy. D340/D342/D343 captured and repaired the
  fixed-point, coordinate-stream, and typed-float proof contracts.
- D344 authored immutable attempt3 with 13 changed and 115 preserved parts but
  retained historical FAIL because its metadata comparator used address-bearing
  PXR `repr`. D345 repaired that comparator; it did not retroactively pass D344.
- D341 Rerun lifecycle is mandatory for geometry/pose/contact/runtime: finalized
  RRD/RBL, exact entities/timelines/components, headless screenshot, and actual
  inspection. Rerun is observability, never numerical authority.

## D347 Verified Result

- New variable: measurement-only `[physx_asset_validator_activation_order]`.
  New physical variables `0`; parameters added/removed/changed/increased/decreased,
  decomposition/target/scene/callback/Rerun changes, and threshold relaxations are
  all `0`. Asset writes/recooks and controlled physics are `0`.
- Fresh RTX 4090 validate ran exactly once. The exact extension
  `omni.physx.asset_validator` v`107.3.26` began disabled; D347 enabled it once,
  verified its ID/root/five pinned files, then imported
  `omni.physxassetvalidator` and acquired its public interface. Retry, fallback,
  manual `PYTHONPATH`, private `.so`, bundle/custom experience, and app-update
  pump counts are all `0`.
- The activation JSON was persisted before callback 1. Callback witnesses are
  `256/256`: 128 pieces x prototype then instance, each callback once/inline,
  `RESULT_VALID`, one convex, serialization error `0`, and settings/cache restore
  PASS.
- Corrected live audit is `127/128`: gripper `64/64`, link5 `63/64`. Surface,
  fixed-point/preserved, typed float bits, owner, and GPU-compatible checks are
  each `128/128`.
- The sole failure is link5 `part_045`. Prototype and instance callback geometry
  are bit-exact, surface error `0m`, and callback volume is
  `5.171636397368745e-7m^3`. PhysX property-query reports
  `4.061547542733024e-7m^3`; relative difference `27.331672%` exceeds the frozen
  `5%` cross-check. Independent triangle-volume recomputation matched the callback
  volume. This establishes a volume-channel discrepancy, not yet which channel's
  semantics are appropriate.
- Because the 128/128 prerequisite failed, D337 controls and raw/live target-union
  distances were deliberately not queried. D347 body distances are `null`; this
  run did not observe collision or clearance at the target.
- Simulation counter stayed `0->0`; controlled physics `0`; settle, ten-trial,
  G0b, RL, and ladder were not run. Inputs and D344 attempt3 stayed immutable.
- Final completion summary SHA-256:
  `93ae7a6daea4d8ba9af6fa09d01deb6c72017925375195a53804b0d55286d65e`.

## D347 Rerun / Visual Result

- Machine contract PASS: frames `6/6`, body frames `2/2`, meshes `522/522`,
  Float64 scalars `1,040/1,040`, events `132/132`, non-system entities
  `2,100/2,100`; RRD/RBL footer, entity, component, and registered timeline-name
  checks passed.
- Original-resolution inspection PASS. The `4800x2800` screenshot shows eight
  nonempty independent link5/gripper source/instance/prototype/candidate panels,
  target cylinder, frame markers, metric table, and event table. The `1076x665`
  decision PNG is legible and reports the pre-physics stop.
- Viewer notifications overlap a panel edge but do not hide required title or
  decision geometry. CLI reports `/events/d347` `part_idx` as unsorted; registered
  timeline names, `event_idx`/`log_time`, footer, entities, and components still
  pass. Preserve this as an observability caveat, not the scientific failure.

## Active Case / Next User Choice

- D347 is complete and immutable: no edit, overwrite, or silent rerun.
- Recommended next case is separately approved D348 measurement-only
  `[physx_property_query_volume_semantics]`. Its question is narrow: why does
  `part_045` have identical callback surfaces but a `27.331672%` property-query
  volume difference, and are those two API outputs meant to describe the same
  geometric quantity?
- D348 must begin from immutable D339/D347 evidence and exact API/source semantics,
  use matched passing controls, and keep the asset, decomposition settings, target,
  all tolerances, and physics unchanged. It must not “fix” the result by raising
  the `5%` threshold or dropping the per-part check.
- Only if a new case proves that the volume cross-check is semantically valid and
  obtains 128/128 may the frozen target-distance query become eligible. Settle is
  a still-later, separately approved case; no automatic promotion follows.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D344-D347; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d347_asset_validator_activation_order_repair.md`
3. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_completion_summary.json`
4. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_asset_validator_activation_order.json`
5. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_validate_cook_witness_manifest.json`
6. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_fresh_live_representation_audit.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_zero_step_representation_gate.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_rerun_validation.json`
9. `claudedocs/runtime_logs/grasp_track/g0a_d347/d347_manual_visual_inspection.json`
10. D346, D345, D344, and D341 session documents when tracing provenance

## Do Not Trust As Current / Durable Boundaries

- `HANDOFF.md` and `TASKS.md` are stale. q5 `0` means CLOSED.
- Do not report D337 anchor clearances as D347 measurements. D347 target queries
  are empty/null by contract.
- Do not summarize D347 as “target collision” or “128 parts failed.” It is one
  volume cross-check failure after `256/256` callbacks and `127/128` part PASS.
- `d347_raw_live_measurement.json` contains inherited D340 pre-correction fields.
  Corrected authority is `d347_fresh_live_representation_audit.json`.
- Callback hull volume and PhysX property-query volume are not interchangeable
  until their semantics are independently established. Surface equality alone
  does not waive the frozen volume gate.
- Never hash default PXR `repr(...)`; canonicalize typed values and list operations.
  Extension-owned modules must be imported only after exact extension enable and
  verification.
- D338-D347 evidence is forward-only: no overwrite, silent rerun, retroactive PASS,
  or promotion. Rerun never substitutes for callback/Float64 evidence.
- `JOINT_LIMITS` removal, hardware control, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

Base HEAD is `d9d224be7793c02754992401a06c3b5eb94826fa` (`D346`), pushed by the
user. D347 code, state documents, and outputs are uncommitted; commit/push remains
user-request-only.
