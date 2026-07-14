# START_HERE.md

Last updated: 2026-07-14 KST. D345 proof-only deterministic USD metadata
comparator completed PASS in one registered run. It proves D344's sole semantic
failure was an address-bearing comparator false difference; it does not certify
attempt3 in Isaac/PhysX. D344 remains FAIL, `g0a_pass=false`, and G0b/RL/ladder
remain blocked. A separate D346 live case is not authorized.

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

## D345 Verified Result

- Verdict: `D345_DETERMINISTIC_USD_METADATA_COMPARATOR_PASS`; sole new variable
  `[deterministic_usd_metadata_comparator]`; registered scientific runs `1`.
- Two standalone-PXR workers each read immutable D339 attempt2 and D344 attempt3.
  All four 310-row canonical streams were `164,675,173` bytes with SHA-256
  `3f85d121439060ef5c6deb49cab7860dbc72eb94e23e54617c4ac2b1f7cdcd09`.
  Unknown types, runtime-address leaks, and time samples were all `0`.
- Exactly 39 registered geometry values were masked. Direct authored
  `apiSchemas` `149` rows and composed metadata `194` rows were four-way exact.
- The old `repr(TokenListOp)` negative control contained addresses in all 194
  rows and changed hash across workers. Removing one PhysX token (`3->2`) and
  changing `prepend` to `explicit` with the same final tokens were both rejected.
- D339 attempt2 `18->18`, D344 attempt3 `9->9`, and D344 output `19->19` stayed
  exact. Existing parameter increases/changes, threshold relaxations,
  decomposition changes, asset operations, Isaac runtime, and physics were `0`.
- Rerun was correctly omitted under the preregistered non-spatial/non-temporal
  file/type/schema/hash exception. No RRD/RBL/PNG or collision asset was created.

## Active Boundary — D345 Complete / D346 Not Authorized

- D344 stays `D344_G0A_ATTEMPT3_AUTHORING_CONTRACT_FAIL_STOP`; no retroactive
  PASS, attempt3 overwrite, or same-path D345 rerun is allowed.
- D345 certifies comparator behavior and non-geometry semantic equality only.
  It does not prove callback count, live 128-part fidelity, target clearance,
  contact behavior, settle, or grasp success.
- Recommended next choice is separately preregistered D346 fresh live validation
  of immutable attempt3: callback `256`, actual pieces `128`, frozen open-jaw
  `(7,11)mm` target gate, and full D341 Rerun completion. User approval is required
  before any Isaac/Kit/GPU execution.

Reserve only: reactive step-0 onset-metric hardening inside a future settle;
long-term purpose-built simple production colliders after this forensic chain.
`r>17mm` grasp-depth redefinition remains unnecessary.

## Operational Storage Sidecar

- User-authorized read-only audit completed for the pre-RL D242/D247 0-999
  script corpus. It is not a current D322-D345 runtime input, but compact D247,
  labels, D256, and D257 remain the frozen script-control lineage.
- Keep the D242 parent and all non-raw/control files local. External archive
  plan is one non-raw core copy plus five 200-episode raw-PNG batches; no USB
  write or local deletion has started.
- Plan: `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/ARCHIVE_PLAN.md`.
- This storage sidecar does not change the completed D345 boundary or authorize D346.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D341-D345; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d345_deterministic_usd_metadata_comparator.md`
3. `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_summary.json`
4. `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_deterministic_usd_metadata_evidence.json`
5. `claudedocs/session_20260714_grasp_g0a_d344_attempt3_fixed_point_collision_geometry.md`
6. `claudedocs/runtime_logs/grasp_track/g0a_d344/d344_postrun_root_cause_audit.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/d344_attempt3_asset_manifest.json`
8. `claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md`

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
- D338 attempt1, D339 attempt2, D340, D342, D343, D344, and D345 outputs are
  immutable. No overwrite, silent rerun, retroactive PASS, or promotion.
- `JOINT_LIMITS` removal, hardware control, B200/SSH/pull, `/half-clone`, and
  unapproved commit/push remain forbidden.

At preflight start, HEAD/origin were `c90b671f479e019f8582775dc0e041a8bb7ba2e0`
(`D335 시작전`; tracked content through D344), already user-pushed. Dataset
payloads and ignored D344 USD assets are not in that push. Current archive/state
edits plus D345 work are uncommitted; commit/push remains user-approval-only.
