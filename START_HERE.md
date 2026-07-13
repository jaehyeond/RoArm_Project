# START_HERE.md

Last updated: 2026-07-14 KST. D344 ended with registered verdict
`D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP`. A forward-only attempt3
asset exists, but fresh Isaac/live validation did not run. `g0a_pass=false`;
G0b/RL/ladder remain blocked.

## Current Truth

- Active pivot remains cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`).
  Cube, G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are
  out of scope.
- q5 convention: URDF `q5=0` = CLOSED; sim OPEN is `~1.541-1.571rad`.
  Frozen target/control is `q5=1.5413rad`, `(radial,tangent)=(7,11)mm`, tangent
  sign `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw family (`2,560/2,629` raw-clear). At the frozen
  target, raw link5/gripper clearances were `+4.2726/+11.1751mm`, but the old
  cooked link5 hull caused a `38.861N` step-0 impulse.
- D339 attempt2 proved two cold cooks bit-exact, then found 13/128 live-fidelity
  failures. D340 captured matching fixed-point candidates for those 13 pieces.
  D342 repaired the coordinate-stream proof; D343 repaired the typed float32
  readback proof (`128/128`, bits `0x38d1b717`). Historical FAIL verdicts were
  not rewritten.
- D341 Rerun lifecycle remains mandatory for geometry/pose/contact/runtime
  decisions: finalized footer, exact entities/timelines/components, embedded
  Blueprint plus RBL, headless screenshot, and separate actual inspection.
  Rerun is observability, not numerical/hash authority.

## D344 Verified Result

- Sole new variable: `[attempt3_fixed_point_collision_geometry]`; effective
  asset-build runs: `1`.
- Build preflight passed all `26` checks. D338-D343 source inventories/hashes,
  environment pins, exact 13-part set, user authorization, and stop rules were
  valid before mutation.
- New forward-only path:
  `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/`.
- Asset authoring applied exactly `13` registered pieces and preserved `115`.
  Nonphysics files were bit-exact; mass/center-of-mass/inertia were unchanged;
  D338 attempt1 and D339 attempt2 stayed immutable.
- Authored `minThickness` type/bits passed `128/128`; decomposition settings
  stayed hull vertices `64`, max hulls `64`, voxel resolution `1,000,000`,
  error `1.0`, min thickness `0.0001m`, shrink-wrap enabled.
- The only failed core check was the composed-scene semantic hash after masking
  the registered 39 geometry properties. The registered verdict therefore
  remains FAIL and the fresh validation process was correctly not started.

## D344 Postrun Root Cause

- The whole physics layer became exact after removing the 39 allowed property
  specs; all 310 composed paths, layer header, and non-geometry semantics were
  otherwise structurally unchanged.
- Exact row decomposition found `194/310` raw differences, all and only
  `metadata.apiSchemas`. The comparator used `repr(Sdf.TokenListOp)`, which
  embeds a process-local memory address. It therefore hashed runtime addresses
  as if they were authored asset meaning.
- Read-only diagnosis in two independent processes found non-address semantic
  differences `0`. Raw hashes changed between processes, while every normalized
  source/variant hash was exactly
  `1e458982f356a6d546b73631abf133d302e0371f9a898eae674f76e65f82f9fe`.
- This supports “comparator false difference,” not “live collider success.”
  D344 is not retroactively reclassified, and the attempt3 build must not be
  rerun or overwritten.

## Scope / Parameter / Rerun Audit

- New variables `1`; existing parameter increases `0`; parameter changes `0`;
  threshold relaxations `0`; decomposition changes `0`; target/control/solver
  changes `0`; physics parameter changes `0`.
- Isaac runtime was not created; controlled physics steps `0`; settle and
  10-trial absent.
- D344 Rerun was preregistered for the fresh live-validation process. The build
  hard-stop prohibited that process, so no RRD/RBL/screenshot/manual inspection
  exists. Do not call D344 Rerun-complete; a future live case must execute the
  full D341 lifecycle.

## Active Case / Next Concrete Action

No new case is authorized yet. Recommended next choice is D345, a proof-only
`deterministic_usd_metadata_comparator` repair. If approved, its sole new
measurement variable is `[deterministic_usd_metadata_comparator]`.

1. Seal immutable D344 attempt3 and both D344 diagnosis files before execution.
2. Read source and attempt3 only; do not write/copy/recook any collision asset.
3. Serialize USD metadata by actual type and content. For `Sdf.TokenListOp`,
   record the list-operation mode and token items, never `repr(...)` or an
   object address.
4. In two independent standalone-PXR processes, require identical canonical
   rows and hashes after masking exactly the registered 39 geometry values.
5. Include a negative control showing the old address-bearing representation
   changes between processes and is rejected as nondeterministic.
6. Stop without Isaac, GPU, Rerun, physics, settle, or promotion. This is a
   preregistered non-spatial/non-temporal comparator audit.
7. Only if D345 passes, request a separate D346 live-validation case for the
   immutable D344 attempt3: 256 callback witnesses, 128-part fidelity, frozen
   target-distance gate, and full D341 Rerun completion lifecycle.

Reserve only: reactive step-0 onset-metric hardening inside a future settle;
long-term purpose-built simple production colliders after this forensic chain.
`r>17mm` grasp-depth redefinition remains unnecessary.

## Must Read First

1. `AGENTS.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` tail (D341-D344)
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. `claudedocs/session_20260714_grasp_g0a_d344_attempt3_fixed_point_collision_geometry.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d344/d344_attempt3_build_summary.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d344/d344_postrun_root_cause_audit.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/d344_attempt3_asset_manifest.json`
9. `claudedocs/session_20260713_grasp_g0a_d343_usd_typed_float_readback_contract_repair.md`
10. `claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md`

## Durable Do-Not-Repeat Rules

- `HANDOFF.md`/`TASKS.md` are stale. q5 `0` means CLOSED.
- Never hash default `repr(...)` of PXR objects as stable asset semantics.
  Canonicalize actual typed content and prove cross-process determinism.
- Exact geometry hashes require the same coordinate/value/type stream. Compare
  direct authored identity before transforms; use numeric/solid gates after
  mapping.
- Comparator thresholds and serialization rules are registered parameters; do
  not silently tighten, relax, or replace them after results.
- Rerun omission is limited to preregistered scalar/schema/hash audits with no
  spatial or temporal verdict. Geometry/live/Kit/cooking/physics restores D341.
- D338 attempt1, D339 attempt2, D340, D342, D343, and D344 outputs are
  immutable. No overwrite, silent rerun, retroactive PASS, or promotion.
- `JOINT_LIMITS` removal, hardware control, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

Actual HEAD is `7868abdf2f5c042d6757575a296b3c4881a52425` (`D343`), equal
to `origin/master` before D344. D344 code, state docs, and artifacts are
intentionally uncommitted. Commit/push only on explicit user request.
