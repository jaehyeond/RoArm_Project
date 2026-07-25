# START_HERE.md

Last updated: 2026-07-25 KST. D381 is complete and frozen as a pre-Rerun
presentation-serialization FAIL_STOP. D380's numeric result is unchanged.
No active case is approved.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a.
- Historical D362 target only: radius/diameter/height `17/34/90mm`, mass
  `0.72kg`. A64 moved and tipped it but did not grasp it.
- Actual product target: vendor nominal diameter/height `29/50mm`, zelkova or
  walnut. Mass, tolerance, COM, inertia, and friction remain unmeasured.
- D362 contact/tipping numbers do not transfer to the 29x50 target.
- `q5=0` is CLOSED; frozen OPEN is `1.5413rad`.
- D368 A64 (`64 link5 + 64 gripper_link`) is the current 64-cap reference
  candidate, not an optimum.
- D372 P34 is a manually partitioned candidate:
  link5 `16` = body `1` + connector support `3` + fixed jaw `10` + backbone `2`;
  gripper_link `18` = moving support `4` + moving jaw `12` + backbone `2`.
  Total `34` is a semantic design, not a measured optimum.

## Preserved D380 Geometry Result

- Frozen output:
  `claudedocs/runtime_logs/grasp_track/g0a_d380/attempt1_failed_part_cook_provenance_semantic_impact_audit/`
- Canonical evidence SHA-256:
  `4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1`.
- Numeric verdict:
  `D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`.
- Exact failed set `17/34`: link5 `4`, gripper_link `13`.
- Failed-part authored/retained/omitted vertices `401/178/223`; `181` omitted
  vertices exceed `0.1mm`; introduced/moved JSON-coordinate vertices `0`.
- `34/34` cooked vertex sets are authored-vertex subsets with authored-convex
  containment. The 17 failures are inward vertex elision, not outward growth.
- Failed-part original-topology part-volume sum loss is
  `341.24192512757054mm^3` (`3.91834189876502%`). It is not compound
  boolean-union or void volume.
- Role-scoped jaw-separation increase upper bound is
  `1.1258255122580576mm`; it is not an observed OPEN jaw gap.
- Max authored vertices per P34 part `44 < hullVertexLimit 64`; cap overflow
  did not force the failures. Exact internal cook cause remains `null`.
- P34 authored-to-cooked identity remains false. Representation repair and a
  separate live identity PASS are required before P34 physics.
- D380 visual completion separately failed from overlap/clipping. That failure
  did not change the numeric result.

## Latest Completed Case — D381

- Case: `D381 [d380_visual_contract_repair]`.
- 이번 case의 신규 변수:
  - `d380_board_pixel_layout_repair_v1`
  - `d380_rerun_notification_buffer_layout_v1`
- Frozen output:
  `claudedocs/runtime_logs/grasp_track/g0a_d381/attempt1_d380_visual_contract_repair/`
- Input: immutable D380 artifacts only. D379 read, numeric/geometry audit,
  Isaac/Kit/PhysX/USD/collider/cylinder/physics/q5/contact/target-IK-path and
  settings work were all `0`.
- Preregistration `15/15`, negative controls `10/10`; D380 input hashes
  `11/11` and registered sources `3/3` exact.
- Actual worker/retry `1/0`, return `1`, elapsed
  `0.7154044299386442s`; timeout/TERM/KILL/process residue all false.
- The worker saved one exact `1920x1080`, `230110B` static board, SHA-256
  `19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d`.
- It then failed while writing layout-validation JSON:
  `TypeError: Object of type bool_ is not JSON serializable`.
- Root cause: Matplotlib Bbox uses `numpy.float64`; its comparison result
  remained `numpy.bool_` instead of Python `bool`.
- The validation JSON is a truncated `144B` file and has no authority. Do not
  parse, repair, or overwrite it.
- `_render_board()` did not return. Recording-only RRD, new RBL, merge,
  Rerun validation, Viewer receipt/screenshot were not reached. Actual Viewer
  invocation is `0`; the registered value `1` was only the upper plan.
- Operational verdict:
  `D381_BOARD_VALIDATION_JSON_SERIALIZATION_FAIL_STOP`.
- The partial board is visually improved but cannot be promoted to D381
  completion PASS because automatic layout and Rerun contracts are incomplete.

## Remaining Nulls

- Complete D381 presentation contract.
- Repaired P34 representation and live identity.
- Actual OPEN jaw-gap change, mouth/window void-volume change, and
  cylinder-facing contact-patch identity.
- 29x50 mass/COM/inertia/friction, mid-height pose, closure/contact/tipping,
  force closure, hold/lift, grasp and target/IK/path justification.
- `g0a_pass=false`.

## Authorization Boundary

- D381 attempt1 is immutable. Do not rerun it, overwrite its truncated JSON,
  or reinterpret the partial board as completion PASS.
- No active case is approved.
- Recommended next minimum is unapproved
  `D382 [d381_layout_validation_native_scalar_serialization_repair]`,
  observability-only:
  1. normalize NumPy/Matplotlib scalars recursively to JSON-native bool/float;
  2. fully serialize in memory before exclusive-create so a failure cannot
     leave a misleading partial JSON.
- D382 must use a new forward-only path, inherit frozen D380 facts and the
  D381 worker `1`/retry `0`/Viewer max `1`/retry `0` contract, and keep
  representation, target, Isaac/PhysX and all physics/q5/contact work at `0`.
- P34 representation repair/live identity, 29x50 target rebase, mass/pose,
  A64/P34 physics, hold/lift, G0b, RL/PPO/VLA each require separate approval.
- Do not call P34 `34` or A64 `128` an optimum.
- Rerun is inspection, not numeric authority; preserve callback arrays/hashes.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale.
- No hardware, signal, dependency install, commit, or push is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D379-D381; ledger tail
2. `claudedocs/session_20260725_grasp_g0a_d381_d380_visual_contract_repair.md`
3. D381 preregistration, stderr, supervisor, partial inspection, fail attestation
4. D380 session/evidence/completion; D379 session/evidence
5. D372 P34 design; D362 only as historical 34x90 physics evidence

## Git

- D381 run-time cross-check:
  `HEAD == origin/master == 2acb5b99567946d343e95e61087357193da0826c`,
  subject `D377(376case)`.
- Worktree contains approved uncommitted D378-D381 code/evidence/state changes.
- Commit/push was not authorized and was not performed.
