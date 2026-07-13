# START_HERE.md

Last updated: 2026-07-13 KST. D343 completed with verdict
`D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS`. D342 remains FAIL;
`g0a_pass=false`; attempt3 is absent; G0b/RL/ladder remain blocked.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out
  of scope.
- q5 convention is repaired: URDF `q5=0` = CLOSED; sim OPEN is
  `~1.541-1.571rad`. Frozen target/control uses `q5=1.5413rad`, `(7,11)mm`,
  tangent sign `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw family (`2,560/2,629` raw-clear). The frozen target
  had link5/gripper raw clearance `+4.2726/+11.1751mm`, but settle produced a
  `38.861N` link5 step-0 impulse from cooked-hull inflation.
- D338 attempt1 failed because global cooking statistics did not witness the
  synchronous cook. D339 attempt2 repaired callback witnessing and proved two
  independent cold cooks bit-exact, but fresh live fidelity failed on 13/128
  pieces. D338 attempt1 and D339 attempt2 are immutable.
- D340 captured one fixed-point candidate for each of those 13 pieces through
  both live channels, but compared a transformed float64 stream with D339's
  direct authored stream and stopped. D342 proved the intended direct authored
  coordinate/hash contract `13/13`, but its overall registered gate remained
  FAIL because it used an unregistered `1e-12m` minThickness comparator.
- D341 installed the Rerun completion lifecycle. Spatial/temporal probes require
  finalized footer, exact entity/timeline/component validation, embedded and
  verified Blueprint, headless screenshot, and separate actual inspection.
  Rerun is observability, never numerical/hash authority.

## D343 Verified Result

- Sole new variable: measurement-only
  `[usd_float_parameter_readback_contract]`; effective runs: `1`.
- Preregistration passed `35/35` under standalone OpenUSD `0.24.5` with
  `numpy==1.26.0`, `psutil==5.9.8`; Isaac Kit/GPU was not started.
- All immutable D339 attempt2 parts passed: `128/128`, `32` predicates each,
  total `4,096`, false `0`.
- Direct Sdf spec/default/type/API authorship and composed Usd attr agreed.
  Direct, composed, and D339-live unique bits were exactly `0x38d1b717`;
  typed value was `9.999999747378752e-05m` for requested `0.0001m`.
- Authored value/opinion passed `128/128`; resolve source was authored Default
  `128/128`; schema fallback, blocked value, time-varying value, and nonzero
  samples were each `0/128`; property stack was exactly one for all 128.
- Direct and composed `metersPerUnit=1.0`; PhysX schema fallback is typed
  `0.0010000000474974513m` (`0x3a83126f`), not the authored value.
- D342 failure-subset anchors passed `13/13`. D342 remains
  `D342_AUTHORED_COORDINATE_STREAM_HARNESS_TOLERANCE_DRIFT_FAIL_STOP`.
- Adjacent negatives passed: `0x38d1b716` and `0x38d1b718` were rejected by
  exact identity although frozen `1e-10m` accepted both. Exact bits are the
  identity gate; `1e-10m` is compatibility diagnostic only.
- Correct typed representation delta `2.526212488436659e-12m` passed frozen
  `1e-10m` and reproduced D342 failure under executed `1e-12m`.

## Scope / Parameter / Artifact Audit

- 13→128 is coverage of the same scalar, not a new variable or parameter
  change. It certifies the 115 parts a future attempt3 would retain.
- Physical variables, existing parameter increases/changes, decomposition
  changes, threshold relaxations, target/controller/solver changes:
  `0/0/0/0/0/0`.
- Collision asset writes, recooks, SimulationContext, physics steps, attempt3:
  `0/0/false/0/absent`.
- D339/D340/D342 remained exact at `18/33/13` files with digests
  `0dae41fd3937a0a8aea18488019c74f097d32f7b8de916943ff31334e30464a1`,
  `def37cc3c4d10cad8919ce71175211cc34fe2e8b567dbc107f13de151a92940d`,
  `7c205d7f6222a2a091a70bb1cf784b339512efbfe8d50bbb3b5ee8c2fed35232`.
- New Rerun was correctly omitted under the preregistered non-spatial,
  non-temporal scalar/schema/bit exception. D342 RRD was context-only and not
  reused as D343 completion evidence.

## Active Case / Next Concrete Action

No further experiment or asset mutation is authorized. D343 stops after proof
repair. Recommended next user choice is a separate D344 collision-asset case:

1. Reuse immutable D340 candidate evidence and preserve D338 attempt1/D339
   attempt2; create only forward-only `collision_asset/attempt3`.
2. Author only the 13 registered fixed-point pieces; retain the other 115
   pieces and all physical/decomposition/target/solver values unchanged.
3. Apply the D342 direct-authored coordinate/hash contract and D343 exact typed
   scalar contract before accepting the derivative.
4. Run fresh live 128-piece/owner/enabled-enumeration and raw-vs-live fidelity
   validation. No stale D339 live result may certify attempt3.
5. Because D344 makes geometry/live-representation decisions, the D341 Rerun
   lifecycle is mandatory at the decision point; D343's omission does not carry.
6. Stop D344 before settle/10-trial/G0b/RL. Physics requires a later separately
   approved case after fresh live PASS.

Reserve only: reactive step-0 onset-metric hardening inside a future settle.
`r>17mm` grasp-depth redefinition remains unnecessary.

## Must Read First

1. `AGENTS.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D340-D343)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/session_20260713_grasp_g0a_d343_usd_typed_float_readback_contract_repair.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_summary.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_evidence.json`
8. `claudedocs/session_20260713_grasp_g0a_d342_authored_coordinate_stream_repair.md`
9. `claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md`
10. `claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md`

## Durable Do-Not-Repeat Rules

- `HANDOFF.md`/`TASKS.md` are stale. q5 `0` means CLOSED.
- Exact hashes require the same coordinate/value/type stream. Prove direct
  authored identity before mapping; mapped geometry uses numeric/solid gates.
- Runtime comparators must be mechanically bound to the freeze source. A tighter
  hardcoded tolerance is a parameter change, not conservative validation.
- USD floats must preserve requested value, type, authored opinion/default,
  resolve source, typed readback, bits, and comparator source. Exact bits are
  typed identity authority; tolerance is compatibility evidence only.
- In core-only PXR, prove unregistered API authorship from direct Sdf metadata;
  do not infer asset absence from composed schema-registry omission.
- Rerun omission is limited to preregistered scalar/schema/bit audits. Geometry,
  pose, contact, frame, trajectory, event-time, Kit, cooking, or physics restores
  the full D341 lifecycle.
- D338 attempt1, D339 attempt2, D340, D342, and D343 effective outputs are
  immutable. No overwrite, silent rerun, tolerance relaxation, or promotion.
- `JOINT_LIMITS` removal, hardware control, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

Actual HEAD remains `b1476d1acc681f392eb3478da5192f3b3898085e` (`rerun 강제 및
셋팅 완료`). D342/D343 work is intentionally uncommitted. Commit/push only on
an explicit user request.
