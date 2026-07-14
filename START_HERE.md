# START_HERE.md

Last updated: 2026-07-14 KST. D349 is complete with verdict
`D349_FROZEN_OPEN_JAW_TARGET_LIVE_DISTANCE_SUPPORTED`. At the frozen OPEN
target `(radial,tangent)=(7,11)mm`, q5 `1.5413rad`, both raw mesh and the
D348-corrected live callback-topology surface proxy were clear and faithful
before any physics step. `g0a_pass=false`; settle, G0b, RL, and ladder remain
blocked unless the user separately approves a later case.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out of
  scope.
- q5 convention: URDF `q5=0` = CLOSED; sim OPEN is `~1.541-1.571rad`.
  Frozen target remains q5 `1.5413rad`, `(radial,tangent)=(7,11)mm`, tangent
  sign `-1`, seed `33201`, HOME-seeded position-only IK.
- D337 restored the open-jaw target family: `2,560/2,629` raw-clear candidates;
  selected target raw clearances were link5/gripper `+4.2726/+11.1751mm`.
  D349 reproduced these raw anchors and measured the corresponding live proxy.
- D348 proved the correct PhysX volume comparator: preserve callback polygon
  topology instead of making a new vertex-only Qhull. The corrected gate is
  `256/256` channels and `128/128` parts.
- D344 attempt3 is the current collision derivative. D345 repaired its USD
  metadata comparator; D347 repaired validator extension activation order and
  captured all `256/256` callbacks.
- D341 Rerun lifecycle remains mandatory for geometry/pose/contact/runtime:
  finalized RRD/RBL, exact entities/timelines/components, headless screenshot,
  and actual original-resolution inspection. Rerun is observability, never
  numerical authority.

## D349 Verified Result

- New variable: `[frozen_open_jaw_target_live_distance_gate]`; new physical
  variables `0`.
- Frozen assets, bodywise `64+64` decomposition, target, `0.1mm` clear gate,
  `0.5mm` raw/live agreement gate, material, actuator, and physics settings were
  unchanged. Asset write, cook callback, property query, and physics step were
  all `0`.
- D337 controls, D348 corrected audit `128/128`, active-part/owner/source
  binding, stage/sensor/unit checks, exact target state, and all authoritative
  pose-stream checks passed.
- link5: raw `4.2726455336106985mm`, live proxy
  `4.272736580324082mm`, absolute difference
  `0.00009104671338366899mm` — PASS.
- gripper_link: raw `11.175088374613944mm`, live proxy
  `11.340262326338637mm`, absolute difference
  `0.16517395172469307mm` — PASS.
- Raw witness repeat was exact for both bodies. Both raw/live distances were
  finite, non-colliding, `>=0.1mm`, and within `0.5mm` of one another.
- Convex-support and vertex-only-Qhull distances are diagnostic only and had no
  PASS/STOP authority.
- Final completion summary SHA-256:
  `6ec883c4ebf4dd25aa2795006699b1d09e3b554412e2dcfa86277de541bd677e`.

## HOME / Runtime Answer

- Nominal project HOME is `[0,0,90,0,0,0]deg`.
- D349 reset was HOME-near, not exact HOME. Actual Float32 joint radians were
  `[0.0189636499,0.0193511546,1.5649892092,-0.0134565402,-0.0147889536,0]`;
  q5 `0` means CLOSED.
- The frozen target was exact Float32
  `[0.0375023820,0.5429451466,1.9687392712,0.1829932779,0,1.5413000584]`;
  q5 was OPEN and the object pose was exact.
- Exact-write used `sim.forward` plus zero-time update, not physical motion.
  All eight recorded phases held the global simulation counter at `0`; controlled
  physics steps were `0`.

## D349 Rerun / Visual Result

- Registered `MEASURED_AUTHORITY` archive passed: frames `6`, coordinate frames
  `2`, meshes `522`, points `4`, arrows `4`, Float64 scalars `1,040`, events
  `136`, exact non-system entities `2,112`, and exact timelines `4`.
- Finalized RRD/RBL footer, embedded blueprint/export, required components,
  counts, and headless render all passed. Main screenshot is logical
  `2400x1400`, raster `4800x2800`.
- Original-resolution inspection confirmed eight nonempty spatial panels,
  raw/live separation witnesses for both bodies, target cylinder, and target /
  commanded / actual frames. Viewer notices partly covered one corner but not
  a decision subject.
- The embedded event viewport did not visibly show the four static summary
  rows. Two failed supplementary display attempts are preserved. A separate
  non-authoritative text-only RRD, bound to the main evidence hashes, made the
  exact four strings legible; it never replaces the main RRD or Float64 JSON.

## Active Case / Next User Choice

- D349 and all diagnostic/display attempts are forward-only and immutable; do
  not edit, overwrite, silently rerun, or retroactively change D347-D349.
- D349 completed the pre-physics distance question only. The live authority is
  a bodywise 64-part BVH union reconstructed from D347 callback faces validated
  by D348. It is an active-collider surface proxy, not a direct PhysX narrowphase
  distance API result.
- A separately approved settle evaluation is now eligible, but not authorized.
  Do not start it without explicit user approval and a new case/path.
- `g0a_pass=false`. No automatic settle, ten-trial, G0b, RL, PPO, or ladder
  promotion follows from D349.

## Operational Storage Sidecar

- The pre-RL D242 0-999 raw PNG set is archived on another Windows PC per the
  user's explicit final-copy confirmation. The intermediate 195,000-file copy
  was machine-verified byte-for-byte before transfer.
- The user explicitly approved local raw-only cleanup. All 195,000 PNGs were
  removed; the canonical raw directory remains empty, while compact/control
  `267` files / `1627858015` bytes remain local and D249 hashes PASS.
- Disk free is now `80836767744` bytes (`87%` used). Restore raw pixels only to
  the canonical path and verify `raw_predelete_manifest_20260714.tsv` first.
- Receipt: `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_local_cleanup_receipt_20260714.json`.
- This storage operation does not change D349 or authorize settle, RL, or rendering.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D348-D349; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d349_frozen_open_jaw_target_live_distance_gate.md`
3. `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_completion_summary.json`
4. `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json`
5. `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_home_start_contract.json`
6. `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_rerun_validation.json`
7. `claudedocs/runtime_logs/grasp_track/g0a_d349/d349_manual_visual_inspection.json`
8. D348 session/evidence for callback-topology volume semantics
9. D347 session/callback evidence when tracing live-proxy provenance

## Do Not Trust As Current / Durable Boundaries

- `HANDOFF.md` and `TASKS.md` are stale. q5 `0` means CLOSED.
- Do not call D347's `5.171636397...e-7m^3` value the callback-topology volume;
  it is the vertex-only re-Qhull envelope. The topology volume is
  `4.061547420...e-7m^3`.
- D349 measured only zero-step raw/live-proxy target clearance. It did not run
  PhysX narrowphase distance, settle, grasp, contact trajectory, or motion.
- A nominal HOME default is not proof that a runtime measurement used exact
  HOME. Record reset jitter, q5 override, simulation-step count, and whether a
  pose was physical or visualization-only.
- Do not raise 5%, drop per-part checks, or substitute a vertex-only Qhull for
  callback face topology. Rerun display copies never replace original evidence.
- D338-D349 evidence is immutable. `JOINT_LIMITS` removal, hardware control,
  B200/SSH/pull, `/half-clone`, and unapproved commit/push remain forbidden.

Base HEAD is `25f085a388a29c18baffa5789cc0d47f713a4728` (`D348완료`), matching
`origin/master` when D349 was approved. The storage-cleanup manifest, receipt,
plan, and session remained the unchanged external dirty baseline. D349
state/code/output also remains uncommitted; commit/push remains user-request-only.
