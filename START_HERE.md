# START_HERE.md

Last updated: 2026-07-14 KST. D350 completed the approved zero-step actual
fixed-jaw geometry measurement, real `headless=False` Isaac Viewer inspection,
and exact `64+64` collider visualization. Final verdict is
`D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED`.
This is a measurement/observability PASS, not an alignment or grasp PASS:
`aligned_pass=null`, `g0a_pass=false`, and `settle_authorized=false`.

## Current Truth

- Active pivot is cylinder grasp-track G0a (`r=0.017m`, `h=0.090m`). Cube,
  G0b close/lift, PPO/RL, VLA, randomization, real hardware, and B200 are out of
  scope.
- q5 convention is fixed: URDF `q5=0` = CLOSED; frozen sim OPEN is
  `q5=1.5413rad`. Nominal project HOME is `[0,0,90,0,0,0]deg`, but D347's
  measured source was HOME-near with q5=0 CLOSED, not exact HOME.
- D348 proved that PhysX property volume must be compared with the callback's
  original polygon topology, not a new vertex-only Qhull envelope. Corrected
  representation gate is `256/256` channels and `128/128` parts.
- D349 then measured the frozen `(radial,tangent)=(7,11)mm`, sign `-1`, seed
  `33201`, HOME-seeded position-only IK target before physics. Its raw and live
  callback-face proxy distances passed, but it did not prove narrowphase,
  settle, contact, grasp, or G0a.
- D350 now replaces the legacy 8mm-point interpretation with a deterministic
  binding to the actual connected `link5` surface containing D349's raw nearest
  witness. No new alignment tolerance was introduced.

## D350 Verified Result

- Approved output: `claudedocs/runtime_logs/grasp_track/g0a_d350/`.
- New variables:
  `[fixed_jaw_semantic_surface_binding,
  frozen_target_fixed_jaw_centerline_measurement]`; new physical variables `0`.
- Frozen target joints and object pose remained Float32 bit-exact. Assets,
  `64+64` decomposition, target, `0.1mm` clear gate, `0.5mm` agreement gate,
  material, actuator, and physics settings were unchanged.
- Actual fixed-jaw component binding PASS: `7,250` faces, `3,519` unique
  vertices, digest
  `8f64ddb03308521ce905d0714def9b72e1e69871d2f9f13ea3bd2a3f07559a4d`.
  The q5-moving `gripper_link` negative control was rejected.
- Fixed-jaw principal axis is `1.123455deg` from the cylinder radial direction,
  `3.158094deg` from link5 `+z`, with world pitch `-67.523879deg`.
- At the cylinder radial station, the measured jaw centerline is
  `-24.055688mm` tangent-offset and `-20.360856mm` in height relative to the
  cylinder center; radial residual is `-0.471744mm`.
- The actual nearest jaw witness is `+15.894261mm` above the cylinder center.
  The legacy `TCP - link5 local x * 8mm` proxy differs from that actual witness
  by `17.027401mm`.
- Actual surface normal is `0.020327deg` from the measured gap direction but
  `36.765887deg` from the surface-to-cylinder-center direction.
- D349 anchors reproduced exactly: link5 raw/live
  `4.2726455336/4.2727365803mm`; gripper raw/live
  `11.1750883746/11.3402623263mm`.
- These are measurements, not a thresholded alignment verdict. D350 therefore
  records `MEASURED`, keeps `aligned_pass=null`, and does not retroactively
  alter D325 or D349.

## Real Viewer and Collider Evidence

- D350 launched a real Isaac GUI with `headless=False`, `livestream=0`,
  `xr=false`, and `cuda:0`. It stayed visible for `180.007254s` through `8,222`
  UI/render updates. This was the actual Isaac Viewer, not a Rerun PNG.
- Timeline stayed paused; counter was `0->0`; target/object state drift was `0`;
  controlled physics steps were `0`.
- Six post-close-verified `1280x720 RGBA` Isaac viewport captures show the full
  assembled robot, actual PhysX collider display, and display-only colored
  link5 `64` + gripper `64` parts from whole/tool/top/side/oblique views.
- The repaired Rerun archive keeps the original five timelines exactly,
  preserves all `297/297` original recording and `12/12` embedded-blueprint
  payloads, keeps `130` static Mesh3D chunks, and adds exactly `130` temporal
  metadata rows for `part_idx=0..129`.
- Repaired RRD/RBL footer, exact `296/296` entities/components, screenshot, and
  original-resolution seven-image inspection all passed. Rerun is display
  evidence only; Float64 JSON remains the numerical authority.
- Attempt1's three async-PNG early-stat failures, missing static `part_idx`
  timeline, and false-valued `asset_write` aggregation bug remain immutable.
  Attempt2 repaired observability without launching Isaac, stepping physics,
  reconstructing Mesh3D, remeasuring geometry, or changing a gate.
- Final completion summary:
  `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json`
  (SHA-256
  `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`).

## Active Case / Next Authorization Boundary

- There is no active execution case after D350. D350 is complete and immutable.
- The measured axis points almost radially, but the actual jaw centerline does
  not pass through the cylinder center in tangent/height. The narrow next
  candidate is a separately approved target/IK geometry-repair case that uses
  the actual connected jaw surface as authority and designs a collision-safe
  OPEN placement through the intended cylinder centerline.
- That future case may need to change target/IK/path variables, so it must be
  preregistered and explicitly approved. No case ID, code, output directory, or
  target change is authorized yet.
- Settle, 10-trial, G0b, RL/PPO, and ladder remain blocked. D350 does not license
  them, and a geometry-repair result would need its own later physics gate.

## Operational Storage Sidecar

- The D242 raw PNG set was externally archived and machine-verified before the
  user-approved local raw-only cleanup. Canonical raw remains empty; restore
  only through `raw_predelete_manifest_20260714.tsv` verification.
- Receipt:
  `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_local_cleanup_receipt_20260714.json`.
  This sidecar does not change or authorize grasp-track work.

## Must Read First

1. `AGENTS.md`; `START_HERE.md`; DECISIONS D348-D350; ledger tail
2. `claudedocs/session_20260714_grasp_g0a_d350_fixed_jaw_geometry_viewer.md`
3. `claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md`
4. D350 completion summary and manual visual inspection under the attempt2 path
5. `claudedocs/session_20260714_grasp_g0a_d349_frozen_open_jaw_target_live_distance_gate.md`
6. D348 session/evidence for callback-topology volume semantics

## Do Not Trust As Current / Durable Boundaries

- `HANDOFF.md` and `TASKS.md` are stale. q5 `0` means CLOSED.
- Do not call D347's vertex-only Qhull value callback-topology volume. Do not
  substitute Rerun Float32 display copies for original callback/Float64 data.
- Do not use the legacy 8mm proxy or near-radial axis angle alone as the actual
  fixed-jaw centerline/alignment authority. Use the bound connected surface.
- Do not convert D350 `MEASURED` into `ALIGNED_PASS`, settle eligibility, grasp
  success, or G0a success. No alignment tolerance exists in D350.
- D338-D350 evidence is immutable. `JOINT_LIMITS` removal, hardware control,
  B200/SSH/pull, `/half-clone`, and unapproved commit/push remain forbidden.

Base HEAD remains `647dfe6ba8e13c781b39850bf7228010fd1683b4`
(`D349완료`) and matches local `origin/master`. D350 is uncommitted; external
user-owned lab-meeting files remain untouched. Commit/push remains
user-request-only.
