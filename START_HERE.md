# START_HERE.md

Last updated: 2026-07-25 KST. D386 is complete and frozen. The same D385
partition has finite minimum vertex budgets for three first-observed failed
layers, but one layer has no admissible path through 64 under the frozen
polygon gate. No new budget was selected or applied.

## Current Truth

- Pivot: RoArm cylinder grasp-track G0a.
- `q5=0` is CLOSED; frozen OPEN is `1.5413rad`.
- Historical D362 target only: radius/diameter/height `17/34/90mm`, mass
  `0.72kg`. A64 moved and tipped it but did not grasp it.
- Actual product: nominal diameter/height `29/50mm`, zelkova or walnut; mass,
  tolerance, COM, inertia, friction, bottom flatness, and roundness unmeasured.
- No `29x50mm` target has been authored, loaded, rendered, measured, or
  simulated. Existing cylinder visuals are old `34x90mm`; D362 metrics do not
  transfer.
- D368 A64 (`64 link5 + 64 gripper_link = 128`) is the current 64-cap reference
  candidate, not an optimum or NVIDIA limit.
- D372 P34 is manual: link5 `16` = body `1` + connector `3` + fixed jaw `10` +
  backbone `2`; gripper_link `18` = support `4` + moving jaw `12` + backbone
  `2`. Total `34` is a design choice, not a measured optimum.
- D379 authored-to-cooked identity passed only `17/34`;
  `p34_authored_to_cooked_identity_pass=false`.
- D380 proved inward vertex elision in the failed `17/34`; exact internal cook
  cause remains `null`.

## Preserved D385 Baseline

- D385 froze the passing `17` parts and exact profile-repair `46` children,
  then applied one registered thin-layer/profile-cell construction to the
  failed source 3-D hulls.
- Complete partitions existed for `4/8` source parents, with partial child
  counts `4+4+6+6=20`.
- Four parents stopped at their first no-cover layer under the project-authored
  `12 vertices/child` gate. Therefore complete source-child and total-part
  counts were `null`, not `83`.
- D385 verdict:
  `D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_NO_ADMISSIBLE_CANDIDATE_FAIL_STOP`.
- `12`, source `<=64`, and total `<128` are project design gates. Installed
  PhysX schema defaults remain `hullVertexLimit=64`,
  `maxConvexHulls=32`; UI ranges are `8..64`, `1..2048`.

## Latest Completed Case — D386

- Case:
  `D386 [d385_minimum_admissible_vertex_budget_localization]`, offline only.
- 신규 변수:
  `observed_no_cover_layer_exact_minimax_vertex_budget_localizer_v1`.
- Exact scope was D385's four first-observed failed layers only. Seven
  later/shadowed layers were inventoried but evaluated `0` times.
- Each target used one complete fixed candidate graph plus one frozen-D385
  `B=12` helper replay. Dynamic-programming minimax and independent exhaustive
  path enumeration both stayed within `12..64`.
- Frozen gates: fan group size `1..4`, polygon count `<=64`, face width `<=32`,
  positive volume, surface `<=0.1mm`, topology-volume relative error `<=0.5%`,
  and positive-volume child overlap `0`.

Results:

- `fixed_backbone_left/y_layer_01`: `29` no-cover, minimum `30` cover,
  `5` children.
- `fixed_backbone_right/y_layer_00`: `12` no-cover, minimum `13` cover,
  `6` children.
- `proximal_upper_arm_hull_a/z_layer_00`: `27` no-cover, minimum `28` cover,
  `6` children.
- `proximal_lower_arm_hull_a/z_layer_01`: no cover at `12` or `64`;
  minimum `null`. All `82` candidate geometries were constructed, `42` failed
  the frozen `polygon_count<=64` gate, and the remaining `40` did not form a
  complete path.
- D385 `B=12` no-cover replay was exact `4/4`. Primary and independent
  algorithms agreed for all four layers. All finite threshold witnesses passed
  surface, volume, polygon, face-width, positive-volume, and overlap gates.
- Three-layer finite maximum `30` is diagnostic only. Because the fourth value
  is `null`, observed-four-layer maximum, selected budget, parent-wide budget,
  complete-P34 budget, complete counts, and global semantic verdict are all
  `null`; materializable candidate is false.
- Scientific verdict:
  `D386_OBSERVED_LAYER_VERTEX_BUDGET_NOT_LOCALIZABLE_FAIL_STOP`.
- Worker/retry `1/0`, return `0`, elapsed `4.933896491071209s`; no timeout,
  signal, or process residue. Completion/observability PASS records the
  scientific FAIL correctly; it is not a collider-design PASS.
- Exact `1920x1080` board, save-only RRD/RBL, one headless Viewer, strict
  entity/component/timeline/footer checks, and manual checks `7/7` completed.
  The Viewer native HiDPI PNG is `3840x2160`; its sandbox message-proxy warning
  was recorded and did not override RRD/RBL validation.
- Asset/USD/Isaac/Kit/PhysX/live callback/collider materialization,
  alternate partition, overlap/tolerance relaxation, Warp/CUDA, `29x50mm`
  cylinder, physics/q5/contact/grasp, target/IK/path, and physical settings were
  all `0`.
- `live_identity_pass=null`, `live_gpu_compatibility_pass=null`,
  `physics_or_grasp_result=null`, `p34_authored_to_cooked_identity_pass=false`,
  `g0a_pass=false`.
- Canonical output:
  `claudedocs/runtime_logs/grasp_track/g0a_d386/attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/`.
- Freeze D386 attempt1. Do not rerun or overwrite.

## Remaining Nulls

- A complete low-count repaired P34 and live authored/callback identity.
- Full fixed-graph status of the seven later/shadowed D385 layers.
- A representation/partition repair for the polygon-gated lower support layer.
- Actual `29x50mm` geometry/readback/render and measured physical properties.
- OPEN gap, void/contact-patch identity, middle-height pose, closure/contact/
  tipping, force closure, hold/lift, grasp, and target/IK/path justification.
- `g0a_pass=false`.

## Next Direction — Not Approved

- Recommended minimum:
  `D387 [d386_shadowed_layer_fixed_graph_completion_localization]`, offline
  only. Apply the already frozen D386 graph/localizer to the seven inventoried
  later layers, without changing a partition or gate, so repair design is not
  based on an incomplete four-layer map.
- Only after that result should a separate one-variable partition/
  representation repair for polygon-gated layers be proposed.
- Separate target case: radius/height `14.5/50mm`, Z axis, exact readback,
  collision-cylinder approximation setting/actual representation, table
  bottom/center/top, actual Isaac views, RRD/RBL; physical properties remain
  unknown until measured.

## Authorization Boundary

- D386 is complete. D387, budget selection/application, further-layer
  evaluation, alternate partition, gate relaxation, asset/USD materialization,
  Isaac/live identity, `29x50mm` target rebase, mass/pose, A64/P34 physics,
  q5/contact, hold/lift, G0b, RL/PPO/VLA are not approved.
- Do not call `30`, P34 `34`, D384 `268/558`, or A64 `128` an optimum.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, signal, dependency
  install, commit, or push is authorized.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D385-D386; ledger tail
2. `claudedocs/session_20260725_grasp_g0a_d386_minimum_admissible_vertex_budget_localization.md`
3. D386 evidence, completion, CSV, board, Rerun validation, manual inspection
4. D385 session/evidence; D384 repair design; D380 provenance; D379 identity
5. D362 only as historical `34x90mm` cylinder evidence

## Git

- D386 approval and completion:
  `HEAD == origin/master == 35f10e3079b19e51209ba4cf1dd66391a431b053`,
  subject `D384`.
- D385-D386 forward-only code/evidence/state files make the worktree dirty.
  Commit/push was not authorized and was not performed.
