# START_HERE.md

Last updated: 2026-07-27 KST. D398 ran once and is frozen. Its numeric
localization passed, but final completion stopped because the Rerun labels
overlapped. No D399 or branch search is approved.

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
- D398 classified all `181` raw axis/midpoint candidates at the six first
  stuck leaves. Every candidate first failed strict vertex reduction; none
  failed split creation or seam/volume validity.
- All `14/14` selected-lineage ancestors had at least one unselected
  admissible option. This proves the frozen greedy choice was not locally
  forced, but alternative-branch completion feasibility remains `null`.
- No complete/materializable collider exists. Final part count, void,
  clearance, raw surface, seed, live identity, physics, contact, and grasp
  are `null`.
- D398 performed no USD/collider write, Isaac/Kit/PhysX/Warp-CUDA launch,
  cylinder creation, physics step, q5 sample, or contact query.
- `materializable_candidate=false`; `g0a_pass=false`.

## Latest Case — D398 Dead-End Provenance Localized; Presentation FAIL

`D398 [d397_six_failed_parent_greedy_bsp_dead_end_provenance_localization]`

Diagnostic variable:
`six_failed_parent_axis_midpoint_option_rejection_provenance_v1`

Forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d398/attempt1_six_failed_parent_greedy_bsp_dead_end_provenance_localization/`

Execution:

- input hashes exact; preregistration PASS
- worker/retry/signal `1/0/0`
- calculation/supervisor elapsed
  `5.820609834045172/6.199380008969456s`
- six frozen final forests, first-stuck identities, selected lineage: exact
- negative controls `5/5` rejected

Candidate localization:

- raw candidates by parent: `38/45/18/18/30/32`, total `181`
- midpoint generation failure `0`
- paired split creation failure `0`
- seam/volume validity failure `0`
- strict vertex-reduction failure `181`
- admissible at first stuck leaf `0`
- ancestor lineage `14`; unselected admissible option existed `14/14`
- ephemeral trace/parity/total split evaluations `991/991/1982`
- new branch, backtracking, serialized/adopted geometry `0/0/0`

Numeric verdict:

`D398_SIX_FAILED_PARENT_GREEDY_BSP_DEAD_END_PROVENANCE_LOCALIZED`

Diagnostic conclusion:

`AT_LEAST_ONE_FROZEN_GREEDY_ANCESTOR_HAD_AN_UNSELECTED_ADMISSIBLE_OPTION_COMPLETION_FEASIBILITY_NULL`

Observability:

- exact `1920x1080` board: readable, no overlap
- strict RRD/RBL automated validation: PASS
- Viewer/retry `1/0`
- manual inspection: Rerun red labels overlap; required
  `no_text_overlap_or_clipping=false`
- final operational verdict:
  `D398_COMPLETION_INTEGRITY_FAIL_STOP`

## Interpretation Boundary

- The first stuck leaves failed the immediate strict-reduction rule, but the
  selected histories were not forced.
- An unselected admissible ancestor option does not prove that option completes
  the whole tree. D398 intentionally did not follow it.
- Do not infer that max-12 must be relaxed or the plane family must change.
- D398 is not an Isaac timeout, GPU, PhysX cook, contact, or grasp result.
- The label-overlap failure does not erase the numeric localization; the
  Rerun presentation is separately incomplete.

## Active Case — None; Awaiting Explicit Approval

2026-07-27 read-only resume verified D372-D398 and the installed
`PhysxSchema 107.3.26`. It surfaced an UNAPPROVED alternative direction: an SDF
mesh collider (concave-preserving; per version-matched Omni Physics 107.3 the
only dynamic+concave option; `schema.usda:1043`) plus a custom-geometry exact
`29x50` cylinder target, aimed at re-measuring D362 tipping in physics rather
than perfecting authored-to-cooked identity. Not a scope change; requires
explicit approval. See
`claudedocs/session_20260727_grasp_g0a_d398_resume_verification_and_sdf_reevaluation.md`
and `claudedocs/BACKLOG.md`.

Recommended minimum:

`D399 [d398_rerun_label_deconfliction_observability_repair]`

- immutable D398 evidence/display only
- label placement and presentation output only
- numeric worker, branch search, geometry generation/adoption `0`
- USD/Isaac/PhysX/cylinder/physics/q5/contact `0`

Only after D399, a separately approved bounded alternative-branch completion
case may keep max-12, axis/midpoint plane family, and all gates frozen while
changing only the no-backtracking search policy.

## Physics-Entry Boundary

- D397 still has no complete collider; D398 did not create one. The sequence
  remains stopped before USD/PhysX cook/readback.
- Nominal-cylinder zero-step, measured product properties, center-height pose,
  wrist, closure/contact, and tipping remain blocked behind a complete offline
  collider and separate live-identity PASS.
- Do not treat default mass, COM, inertia, or friction as product evidence.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/DECISIONS.md` D397-D398
4. `claudedocs/EXPERIMENT_LEDGER.md` tail
5. D398 session, evidence, manual inspection, completion summary

## Authorization and Do-Not-Repeat

- D389-D397 paths are frozen; never rerun or overwrite them.
- D398 attempt1 is now also frozen; never rerun or overwrite it.
- No D399, branch search, geometry repair, or physics is approved.
- Do not materialize or physically test D397/D398 diagnostic geometry.
- Do not change max-12, plane family, and greedy search together.
- Do not modify `claudedocs/lab_meeting/20260715/d334_collision_table/`.
- `HANDOFF.md` and `TASKS.md` are stale/current-state forbidden.
- No hardware, process signal, dependency install, commit, or push authorized.

## Git

- Before D398 authoring, `HEAD == origin/master ==
  7736c73910aa5756ef1560ee55640ba005faa012`, subject `~D397까지의 작업`,
  and the worktree was clean.
- D398 code, evidence, and state-doc changes remain local; commit/push is not
  authorized.
