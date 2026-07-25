# START_HERE.md

Last updated: 2026-07-25 KST. D384 is complete and frozen; no new experiment is
approved. D380 numeric provenance and D379 P34 identity remain unchanged.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a.
- `q5=0` is CLOSED; frozen OPEN is `1.5413rad`.
- Historical D362 target only: radius/diameter/height `17/34/90mm`, mass
  `0.72kg`. A64 moved and tipped it but did not grasp it.
- Actual product: nominal diameter/height `29/50mm`, zelkova or walnut; mass,
  tolerance, COM, inertia, friction, bottom flatness, and roundness unmeasured.
- No `29x50mm` target has been authored, loaded, rendered, measured, or simulated.
  Existing cylinder visuals are old `34x90mm`; D362 metrics do not transfer.
- D368 A64 (`64 link5 + 64 gripper_link = 128`) is the current 64-cap reference
  candidate, not an optimum or NVIDIA limit.
- D372 P34 is manual: link5 `16` = body `1` + connector `3` + fixed jaw `10` +
  backbone `2`; gripper_link `18` = support `4` + moving jaw `12` + backbone `2`.
  Total `34` is a design choice, not a measured optimum.

## Preserved D379-D380 Result

- D379 full authored-to-cooked identity passed only `17/34`; therefore
  `p34_authored_to_cooked_identity_pass=false`.
- D380 classified the failed `17/34` as link5 `4` plus gripper_link `13`.
- Failed authored/retained/omitted vertices were `401/178/223`; `181` omissions
  exceeded `0.1mm`; introduced or moved JSON-coordinate vertices were `0`.
- All `34/34` cooked vertex sets were authored subsets inside the authored
  convex shape; the 17 changes were inward vertex elision.
- Original-topology part-volume sum loss was `341.24192512757054mm^3`
  (`3.91834189876502%`), not compound boolean-union or void volume.
- Max authored vertices was `44 < hullVertexLimit 64`; overflow did not force
  this. Exact internal cook cause remains `null`.
- Verdict: `D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`.

## Latest Completed Case — D384

- Case: `D384 [p34_failed_part_representation_repair_design]`.
- 신규 변수: `failed_profile_prism_authored_subpartition_v1`,
  `failed_source_hull_authored_recursive_partition_v1`.
- Immutable D379/D380 evidence only; ask whether 17 failures can be repaired
  exactly below the A64 reference total `128`.
- Classification: profile prisms `9`; general 3-D convex hulls `8`.
- R1: profile children `46` + merged source cells `205` + unchanged `17` =
  total `268`; project count gate `<128` FAIL.
- R2 upper bound: profile `46` + positive tetrahedra `495` + unchanged `17` =
  total `558`; zero/sliver `40` rejected; count gate FAIL.
- R0 direct points+polygons: preconditions `17/17`, theoretical total `34`, but
  installed public USD has no selector. A C++/opaque bridge is a new pipeline;
  R0 is reserve-only and was not materialized.
- Frozen gates remained surface `0.1mm` and topology-volume `0.5%`; negative
  controls passed `8/8`.
- Isaac/Kit/PhysX, asset/USD, collider, cylinder, physics, q5, contact, and
  target/IK/path counters were all `0`.
- Canonical design verdict:
  `D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP`.
- Meaning: the audit worked, but the exact-split methods found no low-count
  repair. It is not an Isaac/physics failure or grasp test.
- `repair_materialized=false`, `live_identity_pass=null`,
  `p34_authored_to_cooked_identity_pass=false`, `g0a_pass=false`.

## D384 Evidence and Presentation

- Canonical calculation:
  `claudedocs/runtime_logs/grasp_track/g0a_d384/attempt2_callback_vertex_count_field_preflight_repair/`.
- Attempt1 stopped on a stale field name; attempt2 repaired only the in-memory
  alias and completed calculation. Attempts3-6 repaired presentation only.
- Final presentation:
  `claudedocs/runtime_logs/grasp_track/g0a_d384/attempt6_rerun_ascii_glyph_compatibility_repair/`.
- Attempt6 worker/retry `1/0`, return `0`, elapsed `2.321078158915043s`;
  Viewer/retry `1/0`, return `0`, elapsed `1.0074184099212289s`; no timeout,
  signal, or process residue.
- Board exact `1920x1080`; logical `1920x1080` Viewer output native HiDPI
  `3840x2160`; manual checks `9/9`.
- Presentation verdict: `D384_PRESENTATION_CONTRACT_REPAIRED_PASS`.
- Postcompletion label audit: attempt6 JSON `10/13` retain inherited `ATTEMPT3`
  schema labels and `3/13` say `ATTEMPT6`; paths/explicit attempt fields/hashes
  identify the run. Label-clean=false, content/verdict unchanged, no mutation.
- Freeze attempts1-6. Do not rerun or overwrite their paths.

## Remaining Nulls

- Low-count repaired P34 and its live authored/callback identity.
- Actual `29x50mm` geometry/readback/render and measured physical properties.
- OPEN gap, void volume, contact-patch identity, middle-height pose,
  closure/contact/tipping, force closure, hold/lift, grasp, and target/IK/path
  justification.
- `g0a_pass=false`.

## Next Direction — Not Approved

- Recommended next collider case:
  `D385 [p34_source_hull_semantic_low_count_redesign]`, offline only.
- Replace only 8 failed source hulls with semantic low-count parts; freeze the
  profile repair, 17 passing parts, D372 semantic gates, and D379/D380 tolerances.
- A `<128` offline PASS is required before separately approved materialization
  and live identity readback.
- Separate target case: radius/height `14.5/50mm`, Z axis, exact readback,
  cylinder-approximation setting/actual representation, table bottom/center/top,
  actual Isaac views, RRD/RBL; physical properties remain unknown until measured.

## Authorization Boundary

- D384 is complete. D385, direct-polygon bridge work, asset/USD
  materialization, Isaac/live identity, `29x50mm` target rebase, mass/pose,
  A64/P34 physics, q5/contact, hold/lift, G0b, RL/PPO/VLA are not approved.
- Do not call P34 `34`, D384 `268/558`, or A64 `128` an optimum.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, signal, dependency
  install, commit, or push is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D379-D384; ledger tail
2. `claudedocs/session_20260725_grasp_g0a_d384_p34_failed_part_representation_repair_design.md`
3. D384 attempt2 evidence, attempt6 completion/manual inspection, label audit
4. D380 session/evidence, D379 identity result, D372 P34 design
5. D362 only as historical `34x90mm` cylinder evidence

## Git

- D384 cross-check:
  `HEAD == origin/master == b880bc8f28c269f56f05a757dc725619d88c77b1`,
  subject `모델 change(grap당하는 원기둥)`.
- That subject does not prove a new target asset or render; repository evidence
  confirms the actual `29x50mm` target has not been materialized.
- Worktree contains approved uncommitted D382-D384 code, evidence, session, and
  state-document changes. Commit/push was not authorized and was not performed.
