# W13R contract addendum 01 — source semantics and HOLD

2026-09-13 14:10 KST. 原 contract r1 remains preserved. This addendum records root corrections already sent to both active dispatches; it does not grant a new launch.

1. **GPU HOLD:** readiness attempts 01/02/03 already occurred against A4's one-attempt allowance. They remain recorded as a contract deviation. All further renderer/physics launches are on HOLD until root reviews an exact new command, fresh path, bounds and corrected evidence. CPU implementation/audit may continue. Production technical GO has never been issued.
2. **Door displacement metadata:** A3's `2.057714892 mm/deg` is the corrected W12 reference, not a geometry-independent constant to force on W13. For W13 require its actual local hinge radius, frame/units and small-angle calculation. A radius `0.125163049 m` gives approximately `2.184507313 mm/deg`; verify the radius from the actual mesh. Preserve the W12 reference separately. Old `7.7822 mm/deg` remains invalid. This changes no door control or geometry.
3. **Engine time:** new `sync_t_s` must preserve the unrounded binary64 engine reading. Old rounded-to-nine-decimal values may be recorded separately as legacy display values. A source-reconstructed internal step count is DERIVED, not an observed engine counter. Existing old times are immutable.
4. **Numerical position inputs:** initialized lattice `l`, voxel size and domain coordinate bounds require pinned source/old-log evidence. `TEST_ONLY_L_M=1e-7` is a synthetic fixture, never a production fallback. Put derived numerical evidence in separate metadata/config; do not alter the frozen physical parameters to imply a new physics setting.
5. **Raw geometry/forces:** polygon apothem vs circumradius, world/local force frame, mesh role IDs and scalar force reduction must be explicit and source-matched (`audit/RAW_SCHEMA_REQUIRED_ERRATUM_02.md`). This prevents inconsistent reconstruction; it is not a new threshold.
6. **Hard duration:** 32400 seconds is the total production physics process bound including shutdown allowance, not 32400 plus another 1200 seconds. Timeout remains non-success even when a child handles the signal and exits 0; do not continue to the next stage automatically after it.

Source-bound arithmetic corrections already transmitted: `SOURCE_EVIDENCE.md` with errata 01/02/03 (latest wins); actual fixed/door bodies independently bounded; strict `>20 m/s` retains its old operator. Old uncalibrated wall values remain descriptive. No new scientific acceptance threshold is introduced here.

## 14:23 source correction — erratum04

Audit confirmed that the previous four-term derivation omitted effective float32 command conversion and voxel-boundary arithmetic. Before any production outcome, root selected erratum04's source-only conservative position representation term `sqrt(3)*N*(voxelSize+l+8*u64*B)` (old-domain N4001 gives0.0025228147373688847m), with exact binary32 effective velocity/angular-velocity accounting. Do not choose a smaller alternative after seeing clearance. This replaces the strict `<l`-only proof, not a physical threshold. Pin finite/active-domain/range preconditions and all constituent inputs. No open-ended quotient-boundary proof or new physics is authorized by this selection. Bridge01/02's older numeric formulas remain unaccepted.

## 14:26 frame-identity clarification

One-to-one production replay means every **saved particle-frame row** is displayed once. It does not by itself mean every particle frame has a different dense sync index. Old raw contains PFsync `[0,0,0,25,25,25]` from forced decision snapshots. New producer/audit must agree on either explicit PF-row identity with nondecreasing sync mapping, or deduplication of only new observation copies with decision references preserved. Never change physical stepping/time, or alter old raw, to satisfy an invented uniqueness rule. Positive same-sync decision and negative duplicate-PF-row controls are required for the chosen semantics.

## 14:31 chosen frame mapping / 14:38 bin observation

Root selected the auditor's erratum03 option: retain every saved PF row, explicit `particle_frame_row = arange(F)`, nondecreasing sync mapping, exact raw time/phase/sync correspondence per row, and a separate Rerun particle-frame timeline. Do not deduplicate the current collector or change its physical stepping. Oracle04's33 CPU controls were actually re-executed by root; this is not a production coverage result.

Existing sim tracks fixed/door/tray only. `bin:null` therefore means not observed, not zero force. Root's earlier assumption that bin3 rows already existed is withdrawn in CODE_REVIEW_01.md's appended erratum. Add only observation of the already-existing bin mesh (tracker/getters, raw point/force arrays, scalar count/vector-sum norm, explicit role3 distinct from engine ownerID), and match Rerun/audit. Preserve old raw and unchanged physical geometry/parameters/controller. No execution HOLD is lifted.
