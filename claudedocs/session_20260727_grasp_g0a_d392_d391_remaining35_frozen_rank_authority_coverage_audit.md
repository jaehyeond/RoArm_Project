# D392 — D391 frozen rank authority coverage audit over the remaining 35 calls

Date: 2026-07-27 KST

## 1. What and why

User-approved case:

`D392 [d391_remaining35_frozen_rank_authority_coverage_audit]`

이번 case의 신규 변수:

1. `d390_remaining35_frozen_d391_rank_authority_evaluation_set_v1`
2. `factorial_free_canonical_order_invariance_proof_v1`

D391 repaired the numerical rank authority for six disputed D390 terminal point
sets, but it intentionally tested only those six. Five were stable and call29
remained explicitly ambiguous. D392 asked whether the same frozen exact-
difference rank core also gives stable answers for the other 35 D390 terminal
point sets.

For small point sets D391 could enumerate every input permutation. The remaining
35 include sets with up to 25 unique points, so literal `25!` enumeration is not
tractable. D392 therefore separated:

- the mathematical proof: the frozen rank core first factors every input
  through the exact lexicographic `sorted(set(Float64 row tuples))` canonical
  point set; and
- finite smoke controls: independent canonical reconstruction, reversal,
  rotations, and adjacent swaps.

It retained exhaustive permutations only for point sets with at most six unique
points. It did not claim exhaustive `n!` coverage for the 12 larger sets.

This was a failure-capable offline numerical perturbation case. It read only
immutable D390/D391 evidence. It did not rerun clipping, repair D389/D390,
force call29, change a threshold, or invoke collider/USD/Isaac/PhysX, the
cylinder, physics, q5, contact, grasp, target/IK, or path logic. The NVIDIA
official-source rule was therefore not applicable.

## 2. Frozen authority and inputs

At execution:

- `HEAD == origin/master`
  `d354d46134fe002073642441a7d24c99fe579edd`
- the worktree already contained the frozen uncommitted D389-D391 lineage
- no commit or push was performed

Frozen scientific inputs:

- D390 terminal geometry SHA-256:
  `73fc986043b976bec26e1cc92643b8aab281a529f1c71c2918163ba7b98475c7`
- D390 directory manifest SHA-256:
  `8ceb1aa2b3d8ec6f543d4f9bccadb363164c2422a06285dfffdb3932d535209e`
- D391 script SHA-256:
  `2e92e03bd7174622c3e010a88c23d84cb70159bcb7265876a526ec81e99b37b1`
- D391 rank/plane evidence SHA-256:
  `d76bbd88b2a6f188f9c46c382adc7292e56267e6e4a3beb8b9004b70e34b80cc`
- D391 directory manifest SHA-256:
  `b39d874c39e6a4303c2ee4013855e378016a3e9154403061192c163fca83080e`
- exact remaining-35 manifest SHA-256:
  `a77a6e200f63f4ff0a58e576370ef1d29c4e8de32d2f8955e717b46aeb955870`

D392 execution lineage:

- script:
  `sim_scripts/cyl34_top_view_d392_d391_remaining35_same_authority_coverage_audit.py`
- script SHA-256:
  `dd6c3642884935476bc737c8950a33b8e8bafa469b20dbd43df0691644f0303b`
- execution-authority SHA-256:
  `d77e1589b04fbdb63bb9e85f659b30cf5cd22f0a71d5f20e727f1741e5c066a4`
- preregistration SHA-256:
  `7822d1b5f81a5b7db9ec89af7a3ada8c0b62286a1b748336628ee0043283046e`
- registered `START_HERE.md` SHA-256:
  `ed9545eee3d01180b07df2b7822ceb156990ff71a1884590cb9a0fbad43be48d`

Preflight bound exact schemas, call identities, point hashes, the five-stable/
one-ambiguous D391 vector, `numpy==1.26.0`, Rerun SDK/CLI `0.34.1`, and the
absence of direct NVIDIA-stack imports.

## 3. Observable execution order

Forward-only output:

`claudedocs/runtime_logs/grasp_track/g0a_d392/attempt1_d391_remaining35_same_authority_coverage_audit/`

Observed phases, exactly in order:

1. `prepare_start`
2. `prepare_end`
3. `supervisor_before_worker`
4. `worker_start`
5. `canonical_numeric_evidence_committed`
6. `worker_end`
7. `supervisor_after_worker`
8. `observability_start`
9. `observability_end`
10. `finalize_start`
11. `finalize_end`

Execution facts:

- actual offline worker/retry/signal: `1/0/0`
- worker return: `0`
- worker process exited: `true`
- algorithm elapsed: `10.412405928131193s`
- worker elapsed: `10.756219690898433s`
- supervisor elapsed: `11.13938453909941s`
- hard watchdog: `null`
- headless Rerun Viewer actual/retry: `1/0`
- observability stage elapsed: `2.372476889984682s`
- failure attestation: absent

The call-progress journal contained an exact forward prefix of 35 committed
records. Its SHA-256 was
`e2a66afb4af3dc93c95c4ca705eb5de777af9774f933f3e643c9570c50dac2cf`.

## 4. Rank and order controls

D392 imported the frozen D391 numerical core rather than creating a second rank
formula:

1. exact binary-fraction point differences;
2. every point anchor and every unordered point-pair basis;
3. threshold multipliers `0.5`, `1.0`, and `2.0`;
4. hard cap `rank <= min(3, unique_points - 1)`;
5. exact translation controls;
6. power-of-two scale controls.

Order controls:

- exhaustive `n!` mode: `23` calls with at most six unique points;
- source-bound structural proof plus finite generator smoke: `12` larger calls;
- registered transformed input orders: `3,636`;
- every canonical reconstruction and rank signature matched.

A signed-zero negative control also passed: numerically equal `+0.0` and `-0.0`
rows with different Float64 bits were rejected by the bit-exact alias gate.

## 5. Quantified result

The remaining 35 calls were all stable:

| Class | Count |
|---|---:|
| `FACE_LIKE` / face-like rank2 | 23 |
| `FULL_DIMENSIONAL` / rank3 | 10 |
| `LINE` / rank1 | 2 |
| ambiguous | 0 |
| total | 35 |

The D390 historical class happened to match `35/35`, but D392 treats that only
as a diagnostic comparison, not as an answer key.

Combining exact D391 and D392 authority sources gives:

- stable resolved calls: `40/41`
- explicit ambiguity: `1/41`
- resolved classes: `FACE_LIKE 27 / FULL_DIMENSIONAL 10 / LINE 3`
- call29:
  `lower_01_02_pre_float32_lbr`
- call29 point SHA-256:
  `dcd4590e77d929d5abd4edb15f594d5956a9472f9ee099724b39544a7fdfddc6`
- combined 41-entry authority-vector SHA-256:
  `6e2218e8e8d6e8599217137bb37e55bea8f6ae494c4a332874a8ed064949074d`

Because call29 remains explicitly ambiguous, the authoritative aggregate over
all 41 calls remains `null`. D392 did not force it into either rank2 or rank3.

Canonical evidence SHA-256:

`9ced175925c6c528d47bf94e5ae224e65bae1d4d6c88fc236952343de0c72102`

## 6. Visualization and actual inspection

Professor board:

- exact dimensions: `1920x1080`
- rows: left `18`, right `17`, total `35`
- full call IDs visible: `35/35`
- automated text overlaps: `0`
- board SHA-256:
  `6f52952f4fe01c792885f9bf8ca93684aca09a4d97f6656f1348e74ad05c512d`

Rerun:

- save-only RRD/RBL strict validation: PASS
- exact 35 terminal point entities and 35 principal-axis entities
- exact timelines: `blueprint`, `log_time`
- time panel hidden
- RRD SHA-256:
  `147f99385a5a17d6c6b0e33739350b12af9ac07486dabf84a4128f6fe097999e`
- RBL SHA-256:
  `91ec8cd43b944b3d3b429756dad8c454337fcb9e20caa82cc5f5e4709b40af96`
- native HiDPI Viewer screenshot: `3840x2160`
- screenshot SHA-256:
  `9c9e7a121ce877387a4ddfa8bc8d050072899368843039f619e946b925a546b5`

Actual original-resolution inspection passed all ten registered checks. The 35
groups, class colors, labels, and principal axes were visible. The right-side
notification column retained a sandbox `message proxy server crashed` warning,
and some Korean UI text rendered as square glyphs. Neither obscured the 35
decision subjects or the readable English authority/nonclaim metadata. These
defects are recorded rather than hidden.

Manual inspection SHA-256:

`e672c35811601b27a94216183a641a40d338996b39fd1ce542b4b080318a31a9`

## 7. Verdict and boundaries

Numeric verdict:

`D392_REMAINING35_FROZEN_RANK_AUTHORITY_WITH_SCALABLE_ORDER_PROOF_PASS`

Operational verdict:

`D392_REMAINING35_FROZEN_RANK_AUTHORITY_SCALABLE_ORDER_PROOF_COVERAGE_COMPLETE`

Plain meaning:

- D391's stronger rank method now covers the other 35 point sets;
- those 35 introduce no new ambiguity;
- call29 is the only unresolved numerical-rank case;
- D389/D390 remain frozen FAIL_STOP and are not retroactively repaired.

Frozen nonclaims:

- D389/D390 clipping reexecution or repair: `0`
- call29 forced classification: `0`
- authoritative all-41 aggregate: `null`
- seam propagation: `0`
- tolerance/epsilon/jitter/gate/partition/budget/geometry changes: `0`
- selected/adopted budget: `null`
- collider/asset/USD materialization: `0`
- Isaac/Kit/PhysX/Warp/CUDA: `0`
- cylinder/physics/q5/contact/grasp: `0`
- physics or grasp result: `null`
- target/IK/path/settings changes: `0`
- `g0a_pass=false`

The next conditionally approved case may inspect only call29's frozen provenance
to locate the origin and physical scale of its third direction. It must not
change the D391 threshold, inject jitter, update seams, or enter
collider/USD/Isaac/PhysX/cylinder physics in the same case.

