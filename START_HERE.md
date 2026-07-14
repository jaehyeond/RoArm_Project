# START_HERE.md

Last updated: 2026-07-14 KST. D348 is complete with verdict
`D348_PHYSX_PROPERTY_QUERY_TOPOLOGY_SEMANTICS_SUPPORTED`. The only D347
part failure was caused by comparing PhysX collider volume with a newly
re-Qhulled vertex envelope instead of the callback's own polygon topology.
The corrected representation gate is `128/128`, but the frozen target-distance
query was not run. `g0a_pass=false`; settle, G0b, RL, and ladder remain blocked.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out of
  scope.
- q5 convention: URDF `q5=0` = CLOSED; sim OPEN is `~1.541-1.571rad`.
  Frozen target remains q5 `1.5413rad`, `(radial,tangent)=(7,11)mm`, tangent
  sign `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw target family: `2,560/2,629` raw-clear candidates;
  selected target raw clearances were link5/gripper `+4.2726/+11.1751mm`.
  These are D337 anchors, not D347/D348 target measurements.
- D344 attempt3 is the current collision derivative. D345 repaired its USD
  metadata comparator; D347 repaired validator extension activation order and
  captured all `256/256` callbacks.
- D341 Rerun lifecycle remains mandatory for geometry/pose/contact/runtime:
  finalized RRD/RBL, exact entities/timelines/components, headless screenshot,
  and actual original-resolution inspection. Rerun is observability, never
  numerical authority.

## D348 Verified Result

- New variables:
  `[physx_property_query_volume_semantics, rerun_static_summary_and_hidpi_contract]`.
  New physical variables `0`.
- D348 was offline. It read immutable D347/D339 JSON and 256 callback files;
  Isaac/PhysX startup, callback/cook requests, asset writes, target queries,
  and physics steps were all `0`.
- Asset, decomposition, 128-part count, `(7,11)mm/q5=1.5413rad` target, and
  frozen property-volume tolerance `5%` were unchanged. No physical or verdict
  parameter was increased, decreased, relaxed, added, or removed.
- Raw instance/prototype callback payloads were exact `128/128`. All `256/256`
  callback channels were closed and consistently oriented.
- Callback polygon-topology volume versus PhysX property-query volume passed
  `256/256` at the unchanged 5% gate. Maximum/median relative errors were
  `1.362105296456897e-7` / `2.6262618696070446e-8`.
- For link5 `part_045`, PhysX property volume was
  `4.061547542733024e-7m^3`; callback-face volume was
  `4.061547420257619e-7m^3`; relative error was
  `3.015486183560612e-8` (`0.000003015486%`).
- Re-Qhulling only the vertices produced a different envelope:
  `5.171636397369118e-7m^3`, `27.3316720525%` from the property value.
  The callback polygon's maximum Float32 plane residual was `0.3147465198mm`;
  re-Qhulling therefore replaced, rather than preserved, the reported faces.
- Center-translation independent volume recomputation changed at most
  `2.117582368135751e-21m^3`. Historical passing controls, registered nearest
  controls, and same-topology controls passed; a removed-face negative control
  failed as required.
- Scope is PhysX `107.3.26` plus the frozen D347 256 callbacks. Do not generalize
  this to every PhysX version or internal implementation.
- Final completion summary SHA-256:
  `bc93b77fbfbeee074b1241b8f48c0317745b62ff5bca5e2196da00d25eb28697`.

## HOME / Runtime Answer

- Nominal project HOME is `[0,0,90,0,0,0]deg`.
- D347 did not measure at exact HOME. Reset added frozen-seed joint jitter
  `±0.02rad`, then forced q5 to `0rad` CLOSED. Callback/property APIs were read
  at that HOME-near closed pose with simulation counter `0->0` and physics
  steps `0`.
- The q5 `1.5413rad` open target was teleported only for post-failure
  visualization with `sim.forward`/zero-time update; the robot did not
  physically move from HOME to the target.
- D348 created no Isaac/PhysX runtime, so it had no reset or start pose. It
  reinterpreted the immutable D347 measurements offline.

## D348 Rerun / Visual Result

- Scientific attempt2 passed numerically but its actual Rerun screenshot failed:
  timeline-dependent panels were blank and logical `2400x1400` was confused
  with the HiDPI `4800x2800` raster. This failure is preserved.
- Attempt3 caught escaped/truncated HOME text. Attempt4 preserved UTF-8 but
  exposed missing Korean glyphs in Rerun 0.34.1. Both manual failures are
  preserved and did not change science.
- Attempt5 passed the registered short-ASCII viewer contract: coordinate frames
  `2`, meshes `512`, Float64 scalars `1,280`, events `133`, non-system entities
  `2,309`, exact timelines `4`; logical `2400x1400`, raster `4800x2800`, DPR 2.
- Original-resolution inspection confirmed eight geometry panels, `5%`,
  `256/256`, `128/128`, `D347 HOME-near; q5=0 CLOSED`, `0 steps`,
  `D348 OFFLINE`, and `G0A=false`. Korean translation belongs in state docs and
  user briefings because the viewer's bundled font lacks Korean glyphs.

## Active Case / Next User Choice

- D348 and all attempts are forward-only: no edit, overwrite, silent rerun, or
  retroactive change to D347's historical verdict.
- Recommended next case is separately approved D349 measurement-only
  `[frozen_open_jaw_target_live_distance_gate]`.
- D349 should reuse immutable D344 attempt3, D337 `(7,11)mm/q5=1.5413rad`,
  D337 controls, and D348's correct 128/128 topology-volume contract. It should
  query raw-mesh and live-collider target distances before any physics step.
- D349 must not alter assets, decomposition, target, tolerances, material,
  actuator, or physics settings. Settle and ten-trial remain out of scope.
- Only a separately completed target-distance PASS may make a later settle case
  eligible. No automatic G0a/G0b/RL/ladder promotion follows.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D347-D348; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d348_physx_property_query_volume_semantics.md`
3. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt5_ascii_contract/d348_completion_summary.json`
4. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json`
5. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_matched_controls.json`
6. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_home_start_contract.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt5_ascii_contract/d348_ascii_rerun_validation.json`
8. `claudedocs/runtime_logs/grasp_track/g0a_d348/attempt5_ascii_contract/d348_ascii_manual_visual_inspection.json`
9. D347 session/completion/audit when tracing live callback provenance

## Do Not Trust As Current / Durable Boundaries

- `HANDOFF.md` and `TASKS.md` are stale. q5 `0` means CLOSED.
- Do not call D347's `5.171636397...e-7m^3` value the callback-topology volume;
  it is the vertex-only re-Qhull envelope. The topology volume is
  `4.061547420...e-7m^3`.
- D348 did not measure target clearance, collision, settle, grasp, or motion.
  D337 target distances remain anchors only.
- A nominal HOME default is not proof that a runtime measurement used exact
  HOME. Record reset jitter, q5 override, simulation-step count, and whether a
  pose was physical or visualization-only.
- Do not raise 5%, drop per-part checks, or substitute a vertex-only Qhull for
  callback face topology. Rerun display copies never replace original evidence.
- D338-D348 evidence is immutable. `JOINT_LIMITS` removal, hardware control,
  B200/SSH/pull, `/half-clone`, and unapproved commit/push remain forbidden.

Base HEAD is `d452921e04b7d5082c20d4edcfcc44bcefc7c34d` (`D347`), pushed by the
user. D348 code, state documents, and outputs are uncommitted; commit/push remains
user-request-only.
