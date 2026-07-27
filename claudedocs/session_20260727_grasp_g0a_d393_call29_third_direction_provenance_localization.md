# 2026-07-27 — Grasp G0a D393 call29 third-direction provenance localization

## What and why

`D393 [call29_third_direction_provenance_localization]` asked one bounded
offline question: is the registered microscopic third-direction witness for
`lower_01_02_pre_float32_lbr` already present in the frozen D389 source child,
or is that registered witness first created by a later D390 clip?

이번 case의 신규 변수:

1. `call29_d389_shared_fan_seam_exact_rational_point_lineage_v1`
2. `call29_d390_clip_carry_and_near_duplicate_decomposition_v1`

This was provenance localization, not seam repair. It did not adopt a rank,
class, overlap result, collider, or physics result.

## Frozen inputs and execution

- `HEAD == origin/master ==
  d354d46134fe002073642441a7d24c99fe579edd`.
- START_HERE SHA-256:
  `8d1c36a8e7145b0ed9cff7614a3561bb982a6aca3c78f5efd32bece6ccd9237a`.
- Script SHA-256:
  `1984c00b39d21af66ddb056a368b16cebc3be30e8138e979b20139a194e5da70`.
- External authority SHA-256:
  `457e884ae3cce6bddb7c04d710c9d4df2b47017ebb6be30a121c49075651f56a`.
- Preregistration SHA-256:
  `3d01d756013246bdcd7fc564ce33cc2d49311ccfde546be2d259fefbb433d5e2`.
- Actual worker/retry/signal: `1/0/0`.
- Worker return/runtime: `0 / 0.7505593399982899s`.
- Viewer/retry: `1/0`.
- Hard watchdog: `null`; cooperative numeric deadline: `300s`, not exceeded.
- Phase contract: exact 15/15 in forward order.

The deterministic replay uses
`numpy.einsum("ij,j->i", optimize=False) + offset`. BLAS-backed matmul is
stored only as a diagnostic because allocation/summation order can change a
boundary bit around `1e-18m`. Frozen branch choice was never re-decided from
that diagnostic.

## Observable procedure

1. Reconstructed only the frozen LOWER source child (`17` points) and clipping
   child (`24` points).
2. Bound the D390 trace to exactly 21 frozen plane rows.
3. Replayed only active planes `3`, `15`, and `20`; candidate counts were
   `18`, `18`, and `6`.
4. Matched every active candidate and terminal array to the frozen SHA-256.
5. Recursively mapped the terminal maximum-tetra lineage back to D389
   source-sorted vertices.
6. Compared a registered stored-Float64 quartet with an exact-rational
   raw-semantic-plane shadow.
7. Committed numeric JSON/CSV before generating the board or Rerun.
8. Generated and actually inspected the exact `1920x1080` board plus one
   save-only RRD/RBL Viewer capture.

## Numeric result

Canonical evidence SHA-256:
`537537d89a2204987eebfa9bf668968801247e7cef70b7b694434b16b98883a9`.

- Terminal maximum tetra indices: `[0,1,4,5]`.
- Its D389 source ancestry union: `[0,1,2,3,4,6,16]`.
- Final-fan intersection vertices in that ancestry: `[3,4,6]`.
- Exact-seam carried candidates: `[0,16]`.
- Registered rule: all three final-fan vertices plus canonical-first
  exact-seam carried candidate `0` -> quartet `[0,3,4,6]`.
- Stored quartet determinant:
  `-2.208548787108009e-25 m^3`.
- Stored quartet tetra volume:
  `3.6809146451800146e-26 m^3`.
- Maximum normalized raw-seam residual:
  `1.7772201399509673e-18m`.
- Exact-rational raw-plane shadow determinant/volume: `0/0`.
- Terminal near-pair distances `(1,2)` and `(3,4)`:
  `1.4846841437362876e-17m` each.
- Each near pair is exactly one D390 intersection plus one carried point.
- Terminal maximum tetra volume:
  `5.63726747190885e-26 m^3`.
- Source child ConvexHull volume:
  `2.221400826255014e-9 m^3`.
- Terminal micro-tetra/source-volume ratio:
  `2.5377083708987954e-17`.

Numeric verdict:

`D393_CALL29_REGISTERED_MICRO_THIRD_DIRECTION_WITNESS_ALREADY_PRESENT_AFTER_D389_FINAL_FAN_CLIP_PASS`

This means the registered terminal-linked witness already exists after D389's
final fan clip. It supports residue from the full Float64 fan-clip pipeline.
It does not prove that this quartet is unique/maximal, that it is the earliest
micro-rank3 event among every possible quartet, or that a manufactured object
has physical thickness at that scale.

## Visualization result

- Board: exact `1920x1080`, automated text overlap `0`, manual checks PASS.
- RRD/RBL/footer/entities/components/timelines: PASS.
- Viewer screenshot: native HiDPI `3840x2160`.
- Manual inspection: `10/10` registered checks PASS.
- Rerun per-point labels are crowded for nearly coincident points. The clear
  board and canonical JSON remain the reading/numeric authority; Rerun is a
  rotatable inspection atlas only.

## Frozen nonclaims and next boundary

- D391/D392 call29 rank/class remain `null/null`.
- Authoritative all-41 aggregate remains `null`.
- D389/D390 remain frozen FAIL_STOP and are not retroactively repaired.
- Seam/overlap verdict updates: `0`.
- Other calls/pair sweeps/volume solver: `0/0/0`.
- Collider/asset/USD/Isaac/Kit/PhysX/Warp/CUDA: `0`.
- Cylinder/physics/q5/contact/grasp and target/IK/path changes: `0`.
- `g0a_pass=false`.

The next conditionally approved separate case must quantify the ten stable
FULL_DIMENSIONAL terminal failures and their volume/subthreshold semantics.
Only after that may a full `36 pairs / 144 directions` propagation case be
preregistered. No collider or physics follows directly from D393.

## Sources

- `sim_scripts/cyl34_top_view_d393_call29_third_direction_provenance_localization.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_call29_provenance_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_call29_lineage_geometry.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_board_layout_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d393/attempt1_call29_third_direction_provenance_localization/d393_completion_summary.json`
