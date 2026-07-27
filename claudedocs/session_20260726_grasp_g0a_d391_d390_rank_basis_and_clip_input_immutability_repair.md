# D391 — D390 rank-basis and clip-input immutability repair

Date: 2026-07-26 KST

## 1. What and why

User-approved case:

`D391 [d390_rank_basis_and_clip_input_immutability_repair]`

이번 case의 신규 변수:

1. `translation_and_order_stable_terminal_affine_rank_authority_v1`
2. `clip_plane_fixture_input_immutability_v1`

D390 reconstructed 41 failed directional clipping calls, but six terminal point
sets changed rank when the centering or reference point changed. Its clipping
helper also normalized a NumPy view in place and mutated two synthetic
caller-owned plane equations.

D391 was a failure-capable offline numerical perturbation case. It asked:

1. Can the six disputed point sets be classified with a rank authority that
   respects the mathematical bound `rank <= unique points - 1`, tests every
   reference point and every point pair, and remains stable under point order,
   exact translation, and power-of-two scale changes?
2. If the numerical bases still disagree, is the disagreement preserved as an
   explicit ambiguity rather than forced into a line, face, or volume label?
3. Does copy-before-normalization keep all caller-owned plane inputs bit-exact?

Here affine rank means the number of independent spatial directions in a point
set: rank 0 is a point, rank 1 a line, rank 2 a flat face-like set, and rank 3 a
full three-dimensional set.

D391 read only the immutable D390 terminal-geometry JSON as scientific input.
It did not rerun D389/D390 clipping or invoke Isaac Sim, Kit, PhysX, Warp,
CUDA, a collider/USD, the cylinder, physics, q5, contact, grasp, target/IK, or
path logic. The NVIDIA official-source rule was therefore not applicable to
this numerical NumPy/SciPy/Rerun-only case.

## 2. Frozen authority and inputs

At execution:

- `HEAD == origin/master`
  `d354d46134fe002073642441a7d24c99fe579edd`
- the worktree already contained the frozen uncommitted D389/D390 state
- no commit or push was performed

Frozen D390 input:

- path:
  `claudedocs/runtime_logs/grasp_track/g0a_d390/attempt1_d389_directional_epsilon0_boundary_collapse_semantics_localization/d390_terminal_candidate_geometry.json`
- SHA-256:
  `73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7`
- parent directory manifest:
  `22` files,
  `8ceb1aa2b3d8ec6f543d4f9bccadb363164c2422a06285dfffdb3932d535209e`

D391 code and external execution authority:

- script:
  `sim_scripts/cyl34_top_view_d391_d390_rank_basis_and_clip_input_immutability_repair.py`
- script SHA-256:
  `2e92e03bd7174622c3e010a88c23d84cb70159bcb7265876a526ec81e99b37b1`
- execution-authority SHA-256:
  `af148b1a392399f9918ad01b62fd7bf2839056ab8fea57c2442985738a15958f`
- preregistration SHA-256:
  `0617fd65d747fbd9695b076e5323b5c15283d8e5b0826b7b6861f150cde6e618`
- registered `START_HERE.md` SHA-256:
  `315c8af4f9fe5e7a97c4746dd0a6700641436d3e3038128044bb82205e23b90b`

The external authority captured the exact self-inclusive 43-line Git porcelain
state before preregistration. Its status-manifest SHA-256 was
`dbddc9d0a62a961c927283b0f0b61bb190bbd88701d273d9aa9b78144701fc41`.

Preflight passed:

- D390 artifact/schema and exact six-call manifest
- 41 source records
- all 351 stored plane hashes
- NumPy `1.26.0`
- Rerun SDK/CLI `0.34.1`
- no direct Isaac/PhysX/Omniverse/Warp import root

## 3. Observable execution order

Forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/`

Observed phases, exactly in order:

1. `prepare_start`
2. `prepare_end`
3. `supervisor_before_worker`
4. `worker_start`
5. `canonical_numeric_evidence_committed`
6. `worker_end`
7. `supervisor_after_worker`
8. `finalize_start`
9. `finalize_end`

Execution facts:

- actual offline worker/retry/signal: `1/0/0`
- worker return: `0`
- worker process exited: `true`
- algorithm elapsed: `1.8120580650866032s`
- worker elapsed: `4.2507297589909285s`
- supervisor elapsed: `4.562493240926415s`
- cooperative deadline: `300s`, not exceeded
- hard watchdog: `null`
- headless Rerun Viewer actual/retry: `1/0`
- failure attestation: absent

The completion summary linked execution authority, preregistration, invocation,
authorization, worker-start sentinel, worker claim, supervisor, evidence,
visuals, and manual inspection by exact path and SHA. All linkage checks passed.

## 4. Registered rank method

For each frozen point set D391:

1. sorted unique Float64 point rows lexicographically;
2. converted each Float64 coordinate to its exact binary fraction;
3. formed exact point differences from every possible anchor and from all
   unordered point pairs;
4. converted each exact difference matrix once to Float64 for SVD;
5. used
   `tau = sigma_max * max(unique_point_count, 3) * Float64 epsilon`;
6. tested threshold multipliers `0.5`, `1.0`, and `2.0`;
7. enforced the hard affine-rank cap
   `min(3, max(0, unique_point_count - 1))`;
8. exhaustively tested all point permutations;
9. tested one exact dyadic translation and power-of-two scale exponents
   `-20, -10, +10, +20`.

Permutation tests were `6, 6, 6, 720, 2, 720` for the six sets, or `1,460`
total. Every set produced one order-invariant rank signature. Translation and
all four scale controls passed for all six.

The registered policy was conservative: agreement across bases and threshold
bands yields a stable class; disagreement remains explicitly
`NUMERICALLY_AMBIGUOUS_*`.

## 5. Six disputed results

| Call index and ID | Points | D390 historical | D391 result |
|---|---:|---|---|
| `3 upper_00_03_post_float32_rbl` | `3` | rank3 / full | rank2 / face-like / stable |
| `7 upper_01_03_pre_float32_rbl` | `3` | rank3 / full | rank2 / face-like / stable |
| `9 upper_01_03_post_float32_rbl` | `3` | rank3 / full | rank2 / face-like / stable |
| `12 upper_01_04_post_float32_rbl` | `6` | rank3 / full | rank2 / face-like / stable |
| `27 lower_00_05_pre_float32_rbl` | `2` | rank2 / face-like | rank1 / line / stable |
| `29 lower_01_02_pre_float32_lbr` | `6` | rank3 / full | numerically ambiguous across bases |

Aggregate over these six:

- stable: `5/6`
- explicit ambiguity: `1/6`
- stable face-like: `4`
- stable line: `1`
- stable full-dimensional: `0`
- hard-cap-corrected status: `0`

For call 29 the exact binary-coordinate rank is 3, but bounded SVD ranks at the
nominal threshold differ between 2 and 3 depending on the registered difference
basis. At doubled threshold all bases report 2. D391 therefore correctly leaves
its authoritative rank and class `null`.

The D390 provisional 35-call subset and D391's five stable repairs give
mixed-contract provisional coverage of 40 of 41 calls only. D391's stronger
all-anchor/all-pair/order/translation/scale authority was not applied to those
35 calls, so this is not an authoritative 40-call result. Call29 must not be
silently counted as either a face or a volume, and D391 deliberately does not
publish an authoritative all-41 aggregate.

## 6. Plane-input immutability

Copy-before-normalization passed:

- synthetic plane fixtures: `5/5`
- frozen real-trace plane copies: `351/351`
- non-unit/nonzero-offset regression:
  `[2,0,0,-1] -> [1,0,0,-0.5]`
- caller pre/post SHA for that regression:
  `74995978fedb1ae88321e419b7a639a4001d03d22d20247fdf2f820059e22c81`
  before and after
- working copy shared memory with caller: false

The 351 real planes were copied and normalized only to test input immutability.
D389/D390 clipping was not replayed.

## 7. Visualization and actual inspection

Professor board:

- exact dimensions: `1920x1080`
- six cards: `6/6`
- automated text bounds and owner bounds: PASS
- text overlaps: `0`
- SHA-256:
  `a921a87412cc18915c9baf6bdb65a79f53541fadb51017cf8397ed6ce1d42e4a`

Rerun:

- save-only RRD/RBL produced and verified
- exact six point entities and six all-pair chord entities plus metadata
- timeline names exact: `blueprint`, `log_time`
- decision timeline/time panel hidden; no decision cursor
- Viewer actual/retry: `1/0`
- native HiDPI screenshot: `3840x2160` for logical `1920x1080`
- screenshot SHA-256:
  `c91f9587cd7c2a3f1e332bcebe827595da8dd7c551fe034a5e257e9108ec6ff2`
- RRD SHA-256:
  `d552e88792177b3e6b430a6283ec4b29170139314751077faf6cce388694cb3f`
- RBL SHA-256:
  `f533e9e1211e8334114a8239cafcc30f4535706c32d6a80a8518efe3bc213c69`

Actual original-resolution inspection passed all `9/9` registered checks. The
right-side notification-only column showed a sandbox
`message proxy server ... Operation not permitted` warning and a loading
notice. They did not obscure the six decision views, titles, chords, or
numeric/nonclaim metadata. This observation is recorded rather than hidden.

## 8. Verdict and nonclaims

Numeric verdict:

`D391_RANK_CONTRACT_PASS_WITH_EXPLICIT_AMBIGUITY_CLIP_INPUT_IMMUTABILITY_PASS`

Operational verdict:

`D391_RANK_BASIS_AND_CLIP_INPUT_IMMUTABILITY_PASS_NO_D390_REPAIR`

Plain meaning:

- the five mathematically impossible or stable D390 rank labels are repaired
  in a new D391 evidence lineage;
- the sixth boundary-sensitive case remains honestly ambiguous;
- the plane-input mutation defect is repaired and verified in D391 controls;
- D390 itself remains frozen and failed.

Frozen nonclaims:

- D389/D390 clipping reexecution: `0`
- D390 repaired or retroactively passed: false
- authoritative all-41 class aggregate: `null`
- epsilon/5nm/tolerance changes: `0`
- QJ/random jitter: `0`
- partition/budget/geometry changes: `0`
- selected/adopted vertex budget: `null/null`
- collider/asset/USD materialization: `0`
- Isaac/Kit/PhysX/Warp/CUDA: `0`
- cylinder/physics/q5/contact/grasp: `0`
- physics or grasp result: `null`
- target/IK/path/settings changes: `0`
- `g0a_pass=false`

Completion-summary SHA-256:

`9d09e55e6cf7cb7e6d60b1fb1e7722dff09a4da3487e299f3304ad4310d085ea`

## 9. Next authorization boundary

D391 attempt1 is consumed and frozen. Do not rerun, overwrite, or edit its
evidence in place.

The next recommended numerical minimum is an unapproved offline one-variable
evaluation-set case that applies the already frozen D391 rank authority to
D390's other 35 provisional calls. It must not rerun clipping or change a
threshold.

If those 35 remain stable, a later separate case may localize call29,
`lower_01_02_pre_float32_lbr`, by determining the provenance and dimensionless
magnitude of its basis-sensitive third direction without forcing a class.
Only after same-contract coverage is established should a separately approved
propagation case map terminal boundary classes back to D389's nine
indeterminate seams. No current result authorizes a partition, vertex budget,
collider/USD, Isaac/PhysX, `29x50mm` cylinder, physics, q5, contact, grasp,
target/IK, or path step.

## Sources

- `sim_scripts/cyl34_top_view_d391_d390_rank_basis_and_clip_input_immutability_repair.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_execution_authority.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_rank_and_plane_immutability_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_disputed_terminal_geometry.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_disputed_rank_catalog.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d391/attempt1_d390_rank_basis_and_clip_input_immutability_repair/d391_completion_summary.json`
- `START_HERE.md`
