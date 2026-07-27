# D389 — D388 overlap-gate numeric provenance and canonical-tie audit

Date: 2026-07-26 KST

## 1. What and why

User-approved case:

`D389 [d388_overlap_gate_numeric_provenance_and_canonical_tie_audit]`

이번 case의 신규 변수:

1. `lower_b35_global_canonical_path_order_and_dp_pruning_provenance_v1`
2. `adjacent_seam_prepost_float32_epsilon0_vs_frozen5nm_provenance_v1`

D388 had two separate unresolved questions:

1. The lower-layer dynamic-programming result and exhaustive result agreed on
   minimum budget `35` and child count `7`, but disagreed on the canonical cut
   sequence.
2. All 11 adjacent child seams were positive under D388's frozen 5nm clipping
   procedure, but it was unknown whether that volume existed before Float32
   registration, appeared only after Float32 registration, or was only a
   numerical-band effect.

D389 was an offline perturbation/numeric evaluation that could fail. It read
only the immutable D388 evidence, geometry, and CSV. It did not rerun/import or
modify D388 and did not use an asset, USD, Isaac/Kit/PhysX, Warp/CUDA, cylinder,
physics, q5, contact, grasp, target/IK/path, or settings.

## 2. Frozen inputs and Git baseline

At approval:

- `HEAD == origin/master`
  `d354d46134fe002073642441a7d24c99fe579edd`
- subject: D388
- worktree: clean

Immutable D388 inputs:

- evidence:
  `582368f093ba08fec0207967e8e24ac24f0a44774dfa1a7b8c82ae2b6781caba`
- geometry:
  `c119ededf4400efbef55de4d89ccd6c1c8b4e33d4d3795710b6882d369f5e882`
- candidate CSV:
  `a4640cfb09e9a0b4b72a08fa6401f16492c63c9b4089b8e83540a52c7c355505`

Frozen D388 verdict:

`D388_REANCHOR_PARTITION_CONTRACT_FAIL_STOP`

## 3. Attempt1 pre-worker failure and forward-only repair

The first D389 `prepare` stopped before invocation or worker creation.

Exact operational verdict:

`D389_ATTEMPT1_PRE_WORKER_GIT_PORCELAIN_LEADING_SPACE_GATE_FAIL_STOP`

Cause:

- Git porcelain reported the tracked modification as
  `" M START_HERE.md"`.
- `_git(...).stdout.strip()` removed the meaningful first-column space and
  produced `"M START_HERE.md"`.
- The exact-state gate therefore rejected the permitted state.

Attempt1 contains exactly two files:

- preregistration SHA:
  `a6cf72e4527ce40c7bc5cce6334608fa6a697b649bb48e5e31de2ceaa2aa8fc3`
- phase markers SHA:
  `a4deb402e09ee3afae15395dd6ea6a8797f61458e146ea39f4fe6a5b82a610ee`

Its only false checks are the before/after Git-status checks. Invocation,
stdout/stderr, supervisor, worker claim, numeric evidence, RRD/RBL, Viewer, and
completion are absent. Therefore attempt1 actual worker/retry/Viewer is
`0/0/0`.

The original attempt1 script was restored and frozen bit-exact:

`c8f1e07c628ecbefe2dcf49e1a94231b9cbbf51f0b6727e3d5fb4a8083d74b6e`

Reactive operational repair:

- new script:
  `sim_scripts/cyl34_top_view_d389_attempt2_prereg_status_whitespace_repair.py`
- SHA:
  `105b2403b8d49d5baf80390bfc319c446786055e395e141c6918dc55a4d983cb`
- `_git()` removes only `CR/LF`, preserving porcelain status columns.
- `git status --porcelain=v1 --untracked-files=all` is checked as an exact,
  ordered path list.
- attempt1's two hashes, false-check set, phase sequence, old script hash, and
  downstream-artifact absence are all rechecked.
- attempt1 worker `0` plus attempt2 planned worker `1` is gated as total worker
  maximum `1`, retry `0`, Viewer maximum `1`.

Three independent static reviews found that the 27 numeric/path functions and
all scientific constants were unchanged from the frozen attempt1 script.
Attempt2 preregistration passed every check.

## 4. Observable execution order

Attempt2 forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d389/attempt2_prereg_status_whitespace_repair/`

Observed phases:

1. `prepare_start`
2. `prepare_end(pass=true)`
3. `supervisor_before_worker`
4. `worker_start(signal_authority=false)`
5. `canonical_numeric_evidence_committed`
6. `worker_end(worker_claim_pass=false)`
7. `worker_fail_stop(deadline_exceeded=false)`
8. `supervisor_after_worker(returncode=1, pass=false)`

There is no `finalize_start` or `finalize_end`.

Execution facts:

- case aggregate actual worker/retry: `1/0`
- supervisor return: `1`
- supervisor error: `null`
- worker exited: `true`
- supervisor elapsed: `3.6342517430894077s`
- algorithm elapsed: `1.0665364551823586s / 300s`
- deadline exceeded: `false`
- process signals: `0`
- Viewer/retry: `1/0`
- process timeout/kill path: absent

The numeric evidence was committed before board, RRD, and screenshot creation.
This is an intentional numeric-contract FAIL_STOP, not timeout, crash, Isaac,
GPU, or presentation causation.

## 5. Canonical-path result — PASS subresult

The complete lower graph contained:

- all complete paths through B64: `151,664`
- B35 complete paths: `22,464`
- B35 minimum-child paths: `10`
- minimum child count: `7`

Whole-path order:

`(maximum vertex count, child count, full cut sequence lexicographic)`

Global canonical path:

`[0,1,5,9,10,14,18,22]`

D388 local-DP path:

`[0,2,5,9,10,14,18,22]`

The D388 path is global rank `2`.

The first incorrect local pruning occurred at state `5`:

- discarded global prefix `[0,1,5]`, current maximum `16`
- retained local prefix `[0,2,5]`, current maximum `15`
- both later encounter the common `10→14` edge with `35` vertices

That later bottleneck makes the earlier `16` versus `15` difference irrelevant;
the full-path lexical tie breaker should therefore retain `[0,1,5,...]`.
Every registered tie-audit check passed.

Source:

- evidence lines `6-119`
- ranking CSV SHA:
  `4ede2a9867a349b2ffbc476cbf6e4f7a56e80ef8fb6ef26b5549b99727f20e96`

This result does not substitute the new global path into D388 geometry and does
not select or adopt budget `35`.

## 6. Seam numeric-provenance result — FAIL_STOP

The geometry subject remained D388's selected paths:

- upper cuts `[0,3,7,11,12,16,20]`
- lower cuts `[0,2,5,9,10,14,18,22]`

Two independent epsilon-zero methods were compared:

1. a strict linear-program/halfspace method that can classify a full-dimensional
   overlap or a zero-volume boundary;
2. a directional convex-clipping method that tries to construct a 3-D hull from
   each direction.

Summary from the original evidence:

| Measure | Result |
|---|---:|
| Adjacent seams | `11` |
| Nonadjacent negative controls | `25` |
| Pre-Float32 strict-method positive | `2/11` |
| Post-Float32 strict-method positive | `11/11` |
| Frozen D388 5nm replay positive | `11/11` |
| Nonadjacent strict/frozen positive | `0/25` |
| Float32 child roundtrip bit-exact | `13/13` |
| Determinate adjacent provenance | `2/11` |
| Indeterminate adjacent provenance | `9/11` |

The two determinate pre-Float32-positive seams are:

- `UPPER 1-2`: pre `6.4038856253626914e-15m^3`, post
  `5.778450901106452e-15m^3`
- `LOWER 2-3`: pre `2.4130456372851684e-15m^3`, post
  `2.867702544543871e-15m^3`

Thus those two overlaps already exist in the reconstructed geometry produced by
D388's 5nm construction before Float32 registration.

For the other nine adjacent seams:

- the strict method completed and classified pre-Float32 as zero-volume
  touch/boundary;
- post-Float32 strict and directional methods both completed and found positive
  volume;
- the pre-Float32 directional method could not turn the terminal point/line/
  face-like intersection into a 3-D hull.

Adjacent pre-directional failures comprise 12
`ValueError: points are not three-dimensional` directions and one Qhull
`QH6154 Initial simplex is flat/coplanar` direction. The latter still failed
under the registered deterministic `Q12 Pp` fallback.

Across all 36 adjacent/nonadjacent pairs and pre/post directions, 41 of 144
directional calls failed; 26 pairs were affected. Those broader failures do not
change the strict nonadjacent negative count, but they prevent the required
all-pair independent agreement.

Exact seam classification:

`INDETERMINATE_PROVENANCE_OR_SOLVER_DISAGREEMENT`

This is not two solvers giving opposite positive/negative answers. It is one
method completing the boundary classification while the other method has no
registered terminal semantics for a lower-dimensional result.

What is supported:

- the stored post-Float32 child geometry has positive mathematical intersection
  volume on all 11 adjacent seams under both completed methods;
- `UPPER 1-2` and `LOWER 2-3` are positive before Float32 registration;
- the frozen D388 5nm values replay exactly for all adjacent and nonadjacent
  pairs;
- the other nine are Float32-induced candidates.

What is not supported:

- that Float32 definitely introduced the other nine overlaps;
- that all 11 are pre-existing or all 11 are Float32-induced;
- that a failed directional 3-D hull means positive or negative volume;
- that D388 or D389 retroactively passes;
- any partition, tolerance, gate, collider, or vertex-budget adoption.

Primary source:

- evidence lines `168-191`, pair details `1066-7899`
- seam CSV SHA:
  `1fdbaac1c756983c8bd2d2d8e8eabed36a4530393b5f3e3491678335d778f66f`
- geometry SHA:
  `66042a93389cb8d0e6c867be87382566c753cd965ceda619e947e73de4a607be`

## 7. Visualization and manual inspection

Exact board:

- path:
  `d389_numeric_provenance_and_tie_audit_1920x1080.png`
- dimensions: `1920x1080`
- SHA:
  `623f5d8499e9c6bde3f2296553a18044d05771376c454cea64cde44b9d1fb3fa`
- layout checks: `6/6` PASS
- manual: global/local path, state-5 pruning, all 11 rows, and the three numeric
  representations are readable without text overlap.

Rerun:

- RRD SHA:
  `d32802bdaf1b2ff74b49448c3174fc378336642e6c02c1e8efb80bcbd0f92330`
- RBL SHA:
  `d8e86e50debd9b8171f1528f307c515b05d63968268e3419973c2670029ffc91`
- validation SHA:
  `85d9d30691211742c79654d8267d7b6e8795caaaa03c5c6c5f6709da397e4df7`
- archive/footer/entity/timeline/component validation: PASS
- Viewer/retry: `1/0`
- screenshot: HiDPI `3840x2160`, SHA
  `68bfae93222bdb929ee00a69d521a3817a5f50ff8816be47cef485210c80f53e`

Manual visual inspection is `5/6` FAIL:

- the bottom decision summary is readable;
- the actual meshes are too small;
- seam labels overlap heavily;
- the notification panel shows a message-proxy crash and loading warning.

Therefore strict RRD validation does not imply spatial readability. The
post-run manual result was recorded at
`d389_manual_visual_inspection.json`, SHA
`ec4fd92e00b55c2c155674486ad6dbaa6599b57daada0e045e183d47a928808b`.
It does not change the worker FAIL or authorize finalize.

## 8. Verdict and frozen nonclaims

Numeric verdict:

`D389_AUDIT_CONTRACT_FAIL_STOP`

Operational verdict:

`D389_ATTEMPT2_OFFLINE_WORKER_CLAIM_FAIL_STOP_NO_FINALIZE`

Diagnostic cause:

`D389_DIRECTIONAL_EPSILON0_BOUNDARY_COLLAPSE_PROVENANCE_INDETERMINATE`

Frozen state:

- D389 attempt1 and attempt2 are consumed; no rerun, retry, overwrite, or
  finalize.
- completion summary is absent.
- selected/adopted vertex budgets: `null/null`
- budget application: `0`
- partition/tolerance/gate changes: `0`
- materializable candidate: `false`
- live identity: `null`
- physics/grasp: `null`
- `g0a_pass=false`
- all asset/USD/Isaac/Kit/PhysX/Warp/CUDA/cylinder/physics/q5/contact/grasp/
  target-IK-path/settings counters: `0`

Core hashes:

- evidence:
  `9423e870c0a218606781943abd2f5c48cb1e5d53cbbf9fb1212294b4ef5bb5dd`
- supervisor:
  `a56f14a590f0fcfc7467146eb0020af7784c9d1c36bd7fa722acb87785e1643e`
- worker claim:
  `9f85e73e71b66da5b79c8329222847b3032cf0ce682dd7327fdffc8278ceabfe`
- failure attestation:
  `080f5d36e106a660f93cc7b328d0f60223b88c1b7df1e52570747d73a2ced32f`

## 9. Next authorization boundary

Recommended but unapproved:

`D390 [d389_directional_epsilon0_boundary_collapse_semantics_localization]`

Proposed single new variable:

`directional_epsilon0_terminal_affine_rank_boundary_classification_v1`

Offline-only scope:

- read immutable D389 evidence/geometry/CSV;
- localize each failed directional call's first collapse plane;
- record terminal unique-point count, affine rank, singular values, and
  empty/point/line/face/full-dimensional class;
- compare that class with strict halfspace feasibility and registered synthetic
  controls.

D390 must not use QJ/random jitter, change epsilon/5nm/tolerance/gate, change a
partition or budget, repair/recompute/finalize D389, materialize an asset, or
run Isaac/PhysX/cylinder/physics/q5/contact/grasp. Even a D390 PASS would only
localize semantics; a later separately approved case would be needed to repair
the directional zero-boundary contract and re-audit all 36 pairs.

No commit or push was performed.
