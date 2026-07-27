# START_HERE.md

Last updated: 2026-07-27 KST. D397 is complete and frozen. Its offline
shared-boundary construction failed to produce a complete collider candidate;
its failure visualization is complete. No experiment is currently approved.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a.
- `q5=0` is CLOSED; frozen OPEN is `1.5413rad`.
- D362's `34x90mm`, `0.72kg` cylinder is historical; A64 tipped it and did
  not grasp it.
- Actual product nominal is `29x50mm`; actual mass, tolerance, COM, inertia,
  and jaw/table friction are unmeasured.
- A64 (`64+64=128`) is a reference candidate, not an optimum/NVIDIA limit.
- P34 (`link5 16 + gripper 18`) is a manual concept, not an optimum or a
  live/cooked-identity PASS.
- D396 rejected D388 because two completed pre-Float32 overlaps exceeded the
  frozen `1e-18m^3` gate by about `6404x` and `2413x`.
- D397's new Float32 shared-seam BSP completed only `2/8` source parents.
- No complete/materializable collider exists. Final part count, void,
  clearance, raw surface, seed, live identity, physics, contact, and grasp
  are `null`.
- D397 performed no USD/collider write, Isaac/Kit/PhysX/Warp-CUDA launch,
  cylinder creation, physics step, q5 sample, or contact query.
- `materializable_candidate=false`; `g0a_pass=false`.

## Latest Case — D397 Shared-Boundary Construction FAIL

`D397 [shared_boundary_zero_volume_construction_design]`

Science variable:
`float32_canonical_shared_plane_balanced_bsp_v1`

Attempt1:

- preflight `10/10` PASS
- stopped before geometry on
  `_phase() got multiple values for argument 'name'`
- source-parent start/end and geometry evaluation `0/0/0`
- science verdict `null`

Attempt2, sole science authority:

- worker/retry/signal `1/0/0`; elapsed `3.019634233787656s`
- source constructions `8`
- complete: `proximal_upper_arm_hull_b`,
  `proximal_lower_arm_hull_b`
- each children/splits/max vertices/max polygons/max face vertices
  `8/7/12/8/6`
- each 28 child pairs, certified positive-volume overlap `0`
- failed: PUA, PLA, moving upper/lower backbone, fixed left/right
- all six stopped at `no_admissible_shared_plane_split`
- diagnostic source parents/leaves/seams `8/46/38`
- verdict:
  `D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP`

Failed terminal-leaf vertex ranges:
PUA `26-31`, PLA `19-28`, MUB/MLB `16/16`, FBL `14-21`,
FBR `15-22`; completed PUB/PLB `10-12`.
This does not justify a global `12 -> 13` budget change.

Attempt3 presentation-only:

- automated board/RRD PASS; science worker `0`
- manual FAIL: duplicate failure text overlapped axis text and Rerun geometry
  was too small
- `D397_ATTEMPT3_COMPLETION_INTEGRITY_FAIL_STOP`

Attempt4 presentation-only:

- science/Isaac/physics `0`
- clean board exact `1920x1080`
- strict RRD/RBL PASS; Viewer/retry `1/0`; manual `8/8` PASS
- operational:
  `D397_FAILURE_PRESENTATION_REPAIRED_COMPLETE`
- scientific FAIL unchanged

Final output:
`claudedocs/runtime_logs/grasp_track/g0a_d397/attempt4_manual_visual_clarity_repair/`

Science evidence:
`claudedocs/runtime_logs/grasp_track/g0a_d397/attempt2_phase_marker_payload_key_repair/d397_shared_boundary_design_evidence.json`

Session:
`claudedocs/session_20260727_grasp_g0a_d397_shared_boundary_zero_volume_construction_design.md`

## Interpretation Boundary

- Shared seams worked for two completed trees; D397 does not prove the whole
  method impossible.
- Greedy choice, axis/midpoint planes, immediate vertex reduction, and max-12
  jointly reached six dead ends. D397 did not isolate which caused each.
- This is not an Isaac timeout, GPU failure, PhysX cook failure, or physical
  grasp result; those stages did not run.
- Automatic visualization PASS cannot replace manual inspection.

## Recommended Next Candidate — Not Approved

`D398 [d397_six_failed_parent_greedy_bsp_dead_end_provenance_localization]`

Proposed diagnostic variable:
`six_failed_parent_axis_midpoint_option_rejection_provenance_v1`

Proposed scope:

- read immutable D397 attempt2 source/partial-tree geometry only
- at each first stuck leaf, enumerate the frozen axis/midpoint candidates
- count rejection at paired split creation, seam/volume validity, or strict
  vertex-reduction filtering
- record whether an ancestor had an unselected admissible option
- do not select a branch or construct/adopt a candidate

Frozen:

- max-12, plane family, tolerance, count, overlap, and geometry gates
- no backtracking/depth-2 search or non-axis plane
- no USD/asset/collider, Isaac/Kit/PhysX/Warp-CUDA
- no `29x50mm` cylinder, physics, q5, contact, target/IK/path, or settings

Only afterward choose one separate repair: greedy branch, backtracking,
plane-family expansion, or vertex-budget review.

## Physics-Entry Boundary

- D397 failed, so the approved sequence stops before USD/PhysX cook/readback.
- Nominal-cylinder zero-step, measured product properties, center-height pose,
  wrist, closure/contact, and tipping remain blocked behind a complete offline
  collider and separate live-identity PASS.
- Do not treat default mass, COM, inertia, or friction as product evidence.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/DECISIONS.md` D397
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. D397 session, attempt2 evidence, attempt4 completion/manual inspection

## Authorization and Do-Not-Repeat

- D389-D397 paths are frozen; never rerun or overwrite them.
- D398 is not approved.
- Do not materialize or physically test D397's incomplete diagnostic forest.
- Do not change max-12, plane family, and greedy search together.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale/current-state forbidden.
- No hardware, process signal, dependency install, commit, or push authorized.

## Git

- `HEAD == origin/master ==
  d354d46134fe002073642441a7d24c99fe579edd`, subject D388.
- Worktree: user's frozen uncommitted D389-D397 code/evidence/state changes.
