# D390 — D389 directional epsilon-zero boundary-collapse semantics localization

Date: 2026-07-26 KST

## 1. What and why

User-approved case:

`D390 [d389_directional_epsilon0_boundary_collapse_semantics_localization]`

이번 case의 신규 변수:

`directional_epsilon0_terminal_affine_rank_boundary_classification_v1`

D389 preserved 41 failed directional epsilon-zero clipping calls but did not
have a registered meaning for a terminal candidate that could not be completed
as a three-dimensional convex hull. D390 was therefore an offline perturbation
and localization case that could fail. It asked:

1. Can every one of the 41 failures be reconstructed from immutable D389
   artifacts with the same error family, clip count, skipped-plane count, and
   fallback branch?
2. At the first observed collapse, are the terminal unique points empty,
   point-like, line-like, face-like, or full-dimensional under the registered
   Float64 affine-rank contract?
3. Does that terminal diagnosis remain inside D389's already stored strict
   halfspace authority, without changing epsilon, the frozen 5nm replay,
   tolerance, gate, partition, budget, or geometry?

D390 did not import, execute, repair, recompute, or finalize D389. It did not
invoke an asset, USD, Isaac/Kit/PhysX, Warp/CUDA, a cylinder, physics, q5,
contact, grasp, target/IK/path, or any material/mass/actuator/physics-setting
change.

## 2. Frozen inputs, code, and Git baseline

At execution:

- `HEAD == origin/master`
  `d354d46134fe002073642441a7d24c99fe579edd`
- the worktree already contained the frozen uncommitted D389 state
- D390 did not commit or push

Immutable D389 inputs:

- evidence:
  `9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd`
- geometry:
  `66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be`
- seam CSV:
  `1fdbaac1c756983c8bd2d2d8e8eabed36a4530393b5f3e3491678335d778f66f`

Execution code and authorization:

- script:
  `sim_scripts/cyl34_top_view_d390_d389_directional_epsilon0_boundary_collapse_semantics_localization.py`
- executed script SHA:
  `1ac467ee63d9847fc18ba9ca289a9bc257e0dafa64c3087622d65e79f0d5d936`
- preregistration SHA:
  `0a05584d2020a6da0053696a24c212a3147f6817fb2b9585971bfe176c7ec117`
- registered `START_HERE.md` SHA:
  `90cf1f95619f4be21a04e1b11f1a38b4b7ca9e2f9724c419eb6c3b51e644d1a3`

The preregistration, invocation, authorization, start sentinel, worker claim,
failure attestation, and supervisor all bind the same preregistration SHA.

## 3. Observable execution order

Forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/`

Observed phases:

1. `prepare_start`
2. `prepare_end(pass=true)`
3. `supervisor_before_worker`
4. `worker_start`
5. `canonical_numeric_evidence_committed`
6. `worker_end(worker_claim_pass=false)`
7. `worker_fail_stop(deadline_exceeded=false)`
8. `supervisor_after_worker(returncode=1, pass=false)`

There is no `finalize_start`, `finalize_end`, or completion summary.

Execution facts:

- offline worker/retry: `1/0`
- worker return: `1`
- worker process exited: `true`
- process signals sent: `0`
- numeric algorithm elapsed:
  `0.11435000598430634s`
- worker elapsed:
  `2.542525211116299s`
- failure attestation elapsed:
  `2.6323085092008114s`
- supervisor elapsed:
  `2.9062803348060697s`
- cooperative deadline:
  `300s`
- cooperative deadline exceeded:
  `false`
- hard wall-clock watchdog:
  `null`

The supervisor had no process-signal authority, so the 300-second value was a
cooperative check rather than a hard watchdog. The child exited on its own in
under three seconds. This was not a timeout, external termination, Isaac, GPU,
or crash failure.

## 4. D389 failure-manifest reconstruction

D390 rechecked the immutable D389 denominator:

| Measure | Result |
|---|---:|
| Pair count | `36` |
| Directional calls | `144` |
| Successful calls | `103` |
| Failed calls | `41` |
| Affected pairs | `26` |
| Pre-/post-Float32 failures | `24 / 17` |
| Left-by-right / right-by-left failures | `16 / 25` |
| Adjacent / nonadjacent failures | `13 / 28` |
| Fewer than four unique points | `12` |
| Affine rank below three | `17` |
| QH6154 flat/coplanar | `12` |

For all 41 failed calls, D390 matched:

- error family: `41/41`
- active clip count: `41/41`
- skipped-plane count: `41/41`
- recorded fallback branch: `41/41`
- finite terminal candidate coordinates: `41/41`
- stored strict-halfspace authority relation: `41/41`

All 41 stored strict results remained
`STRICT_NONPOSITIVE_OR_SUBTHRESHOLD_WITHIN_SOLVER_BAND`. D390 did not recompute
the strict solver or the frozen 5nm procedure.

The canonical Float64 terminal-geometry artifact contains all 41 calls:

- evidence SHA:
  `3014610b7b2fd953740d239b91d9d9dce8aa917be67c6a1be15cf6ac052d9975`
- geometry SHA:
  `73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7`
- trace CSV SHA:
  `5a8295c85b62552459806c8a51a2ec485c20248365e8d1382857c19ebb527591`

Thus the D389 failed-call lineage and reconstruction are preserved. The failure
is narrower: the registered affine-rank agreement did not hold for six calls.

## 5. Affine-rank result and six disputed calls

The registered affine-rank checks agreed for `35/41` calls and disagreed for
`6/41`.

The stable-under-the-registered-check subset of 35 had provisional classes:

| Terminal class | Stable-call count |
|---|---:|
| `FACE_LIKE` | `23` |
| `FULL_DIMENSIONAL` | `10` |
| `LINE` | `2` |

This 35-call subset is useful localization, but it is not a complete 41-call
scientific verdict.

The six disputed calls were:

| Call | Unique points | mean-centered / NumPy / first-anchor rank |
|---|---:|---:|
| `upper_00_03_post_float32_rbl` | `3` | `3 / 3 / 2` |
| `upper_01_03_pre_float32_rbl` | `3` | `3 / 3 / 2` |
| `upper_01_03_post_float32_rbl` | `3` | `3 / 3 / 2` |
| `upper_01_04_post_float32_rbl` | `6` | `3 / 3 / 2` |
| `lower_00_05_pre_float32_rbl` | `2` | `2 / 2 / 1` |
| `lower_01_02_pre_float32_lbr` | `6` | `3 / 3 / 2` |

The first three-point calls cannot have mathematical affine rank greater than
`2`; the two-point call cannot have affine rank greater than `1`. Their
mean-centered smallest singular values were approximately `0.73e-18` to
`0.87e-18m`, while the derived thresholds were approximately `1.4e-33m`.
Mean subtraction at this tiny scale left Float64 cancellation residue, and the
relative threshold counted it as a dimension.

For `lower_01_02_pre_float32_lbr`, the mean-centered third singular value was
`2.8067339907644104e-18m` against a threshold of
`2.4932093518772976e-18m`, while the first-anchor value was
`1.833422441012149e-18m` against `5.599467974513542e-18m`. It is therefore a
centering/threshold-sensitive boundary case, not an authoritative
full-dimensional result.

The stored aggregate
`FACE_LIKE 24 / FULL_DIMENSIONAL 15 / LINE 2` includes the disputed six and is
not authoritative. A different anchor cannot simply be substituted either,
because a single first anchor is point-order dependent. A separate,
translation- and ordering-stable affine-rank authority is required.

## 6. Synthetic controls

D390 recorded 12 synthetic controls:

- five base affine classes: `5/5` PASS
- two rank-threshold straddles: `2/2` PASS
- five single-plane clipping controls: `3/5` PASS

The `LINE` and `POINT` clipping controls produced the expected class, matching
rank checks, and finite candidate points. Their only false check was
`plane_equation_binary_exact_fixture`.

The direct code cause is input mutation:

1. the fixtures are authored with binary-exact integer coefficients
   `[1,1,0,0]` and `[1,1,1,0]`;
2. `_clip_candidate` takes `equation[:3]` through `np.asarray` without a copy;
3. `unit /= length` normalizes that view in place;
4. the later fixture-integrity check sees the mutated
   `1/sqrt(2)` or `1/sqrt(3)` coefficients and fails.

This is a control-instrumentation/input-immutability defect. It is not evidence
that the `LINE` or `POINT` geometry classification failed.

A read-only post-fail hash audit then recomputed `_array_sha` with the registered
`(1,4)` Float64 shape for every stored real trace plane. All `351/351` stored
plane arrays matched their stored pre-call hashes. Therefore the aliasing defect
is directly observed in the two non-unit synthetic fixtures, but no stored
real-trace plane/hash divergence was observed in D390 attempt1. This does not
make the helper safe; a forward repair must still enforce input immutability.
The post-fail/read-only audit SHA is
`1baf7754de5cc9fb48356608fa4b103c4ff12b6ebdbf519c42cf150b0b5c9b96`.

## 7. Visualization result

Canonical numeric JSON and CSV were committed before presentation.

Professor board:

- actual dimensions: `1920x1080`
- all 41 rows visible in a `21/20` split
- automated layout report: PASS
- board SHA:
  `5248ad176bfb3892e506c4483a3b8e0af8bfb76c7ca65453101f303c008f1b99`

Manual inspection nevertheless failed because panel 2's last black dimension
row and red nonclaim sentence overlap.

Rerun:

- SDK/CLI: `0.34.1`
- source geometry (teal), clipping geometry (orange), terminal points (red),
  and collapse plane (purple) are visible
- save-only RRD/RBL and base structural entity/component/footer checks passed
- headless screenshot was created at native HiDPI `3840x2160`
- screenshot SHA:
  `9d567472c52490a0630f6cbd31159662c5cd4f80ca42251d304d5e97a6514ad6`
- RRD SHA:
  `d6a078db496c92484d7234134832a1b272a7797063ec50ac72e6d5715000bdea`
- RBL SHA:
  `839b234e9b88409d355dffe4472d321da22ed117c4da118a6d67ab6cef413fb8`

The D390-specific strict timeline validator failed for an instrumentation
reason. The RRD table actually contains all 41 `failed_call_index` values
`0..40`, but the parser assumed a fixed printed column position and tried to
parse `log_time` as the index. It must locate the column by its header name.

Manual Rerun inspection also failed:

- the visible timeline cursor is `#0`, not the registered `#40`;
- selected-call and nonclaim text is horizontally truncated;
- the teal/orange/red/purple geometry legend is not fully readable;
- warning/loading notices stayed inside their reserved buffer and did not cover
  the decision geometry.

The post-fail manual audit is `5/10` PASS overall. It is inspection evidence
only and does not authorize finalize or override the failure verdict.

## 8. Exact verdict and nonclaims

Numeric verdict:

`D390_TERMINAL_CLASSIFICATION_OR_TRACE_IDENTITY_FAIL_STOP`

Operational verdict:

`D390_OFFLINE_WORKER_OR_OBSERVABILITY_INTEGRITY_FAIL_STOP`

The first verdict means that the full registered rank/trace contract did not
pass. It does not mean all 41 reconstructions were wrong: the failure is six
rank disagreements plus the two mutated control fixtures.

The second verdict additionally preserves the Rerun parser and manual
presentation failures. D390 attempt1 is frozen. It must not be rerun,
overwritten, repaired in place, or finalized.

Frozen nonclaims:

- no D389 or D390 retroactive PASS
- no authoritative 41-call affine-class aggregate
- no epsilon/5nm/tolerance/gate repair or relaxation
- no QJ or random jitter
- selected/adopted vertex budget: `null / null`
- budget application count: `0`
- materializable candidate: `false`
- live identity: `null`
- physics/contact/grasp result: `null`
- all asset/USD/Isaac/Kit/PhysX/Warp/CUDA/cylinder/physics/q5/contact/grasp/
  target-IK-path/settings counters: `0`
- `g0a_pass=false`

## 9. Next authorization boundary

The next numerical minimum is an unapproved forward-only offline case over the
immutable D390 terminal geometry:

`D391 [d390_affine_rank_basis_and_clip_input_immutability_repair]`

Proposed new variables:

1. `translation_and_order_stable_terminal_affine_rank_authority_v1`
2. `clip_plane_fixture_input_immutability_v1`

It should re-evaluate only the six disputed terminal point sets and the frozen
controls. It must enforce the mathematical bound
`rank <= unique_point_count - 1`, use a preregistered translation/order-stable
difference basis, and leave a boundary case explicitly ambiguous rather than
forcing a class. It must copy fixture inputs before normalization and verify
their pre/post SHA identity. The observed D390 real-trace plane/hash baseline is
`351/351` matches and must remain exact.

The known Rerun header-parser, cursor, text-layout, and board-overlap defects
also require a separately bounded reactive observability repair or an explicit
presentation sub-contract before any repaired scientific result can complete.
They must not be hidden inside a scientific result.

D391 is not approved by the D390 authorization. Before separate approval, do
not run it, change a rank authority, repair presentation, select a budget,
materialize a collider/USD, invoke Isaac/PhysX, or run the `29x50mm` cylinder,
physics, q5, contact, or grasp.

## Sources

- `sim_scripts/cyl34_top_view_d390_d389_directional_epsilon0_boundary_collapse_semantics_localization.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_boundary_collapse_localization_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_terminal_candidate_geometry.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_failed_directional_call_trace.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_post_fail_plane_input_aliasing_audit.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_failure_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_board_layout_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_manual_visual_inspection.json`
- `claudedocs/session_20260726_grasp_g0a_d389_d388_overlap_gate_numeric_provenance_and_canonical_tie_audit.md`
- `START_HERE.md`
