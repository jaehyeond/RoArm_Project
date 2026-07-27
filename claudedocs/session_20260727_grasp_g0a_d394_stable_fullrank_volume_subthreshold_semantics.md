# 2026-07-27 — Grasp G0a D394 stable full-rank volume/subthreshold semantics

## What and why

`D394 [stable_fullrank_terminal_volume_subthreshold_semantics]` asked one
offline question about the ten D392 calls classified as stable
`FULL_DIMENSIONAL`: do their terminal points contain exact positive volume, and
is that volume nevertheless too small to pass the frozen D389 positive-volume
gate?

이번 case의 신규 변수:

1. `stable_fullrank_terminal_exact_dyadic_volume_sandwich_v1`
2. `frozen_volume_gate_monotone_early_stop_semantics_v1`

This case did not reclassify call29, rerun all 41 failed calls, update any seam,
or materialize a collider.

## Frozen inputs and execution

- `HEAD == origin/master ==
  d354d46134fe002073642441a7d24c99fe579edd`.
- Frozen call indices:
  `[1,6,13,14,16,21,23,26,34,39]`.
- Base script SHA-256:
  `4595bee98e87a192c2e55a57f9f545e5a12548a40301ac0b897e3f2b2bbeeac7`.
- Attempt2 wrapper SHA-256:
  `13b28be7b94063daff3398561aacf7123afb03bb1258d40fdc075ed3fd8b9907`.
- Attempt3 visual-repair script SHA-256:
  `e236210387038e7838514655a7cccdf28cdbb6188016b67656e14846943d5e80`.
- Aggregate numeric worker/retry/signal: `1/0/0`.
- Aggregate Viewer/retry: `1/0`.
- Numeric worker return/runtime: `0 / 1.2048863130621612s`.
- Hard watchdog: `null`; post-exit elapsed budget audit: `300s`, not exceeded.

The numeric evidence SHA-256 is
`7672f208cc704bd9c3a51bc0b60040e2a121335cf12dcfaa5fd851484dd089a1`.

## Observable procedure

1. Attempt1 preregistered the exact-volume method but stopped before the worker.
   Static review found that writing `derived_gate_volume_m3=0.0` would falsely
   convert a Boolean gate result into an exact numeric-volume claim.
2. Attempt2 changed only that representation to
   `derived_gate_volume_m3=null`, retained
   `original_calculation_pass=false`, and ran the one authorized numeric worker.
3. For each of the ten terminal point sets, exact dyadic arithmetic established
   a positive tetrahedron witness, an exact convex-hull volume, and an exact
   axis-aligned bounding-box upper bound.
4. The script checked the sandwich
   `0 < tetra witness <= exact hull <= exact AABB <= 1e-18m^3`.
5. It also proved the early-stop rule: every remaining frozen half-space clip
   can only take a subset, so its volume cannot increase above the gate.
6. Attempt2 finalized the save-only RRD/RBL and one Viewer capture, but manual
   inspection rejected the board because Unicode superscript minus glyphs were
   missing.
7. Attempt3 recomputed no science and launched no worker or Viewer. It only
   regenerated the board with ASCII `10^-18` and `10^-13`, then inspected the
   repaired board and the bit-exact frozen Rerun capture.

## Numeric result

- Passing records: `10/10`.
- Upper/lower: `5/5`.
- Pre-/post-Float32: `4/6`.
- Left-by-right/right-by-left: `3/7`.
- Adjacent/nonadjacent: `0/10`.
- Unique pair contexts: `9`.
- Maximum exact convex-hull volume:
  `7.636172300630593e-50m^3`.
- Maximum exact AABB upper bound:
  `1.2462809509519742e-48m^3`.
- Maximum diameter-cube diagnostic:
  `3.2811000933883083e-47m^3`.
- Minimum frozen volume-gate/AABB-upper-bound ratio:
  `8.023872941620009e29`.
- Minimum strict-radius-threshold/coordinate-width-upper-bound ratio:
  `5012.702193942813`.

All ten records retain:

- `original_calculation_pass=false`;
- `propagated_gate_decision_available=true`;
- `propagated_positive_volume=false`;
- `derived_gate_volume_m3=null`; and
- exact final intersection volume `null`.

Numeric verdict:

`D394_FULL10_EXACT_POSITIVE_BUT_FROZEN_SUBTHRESHOLD_MONOTONIC_EARLY_STOP_PASS`

In plain language, each failed terminal point cloud has a mathematically
positive microscopic 3-D volume, but even a rigorous upper bound is vastly
below the frozen solver's `1e-18m^3` acceptance threshold. The safe propagated
claim is therefore only the Boolean gate result “not positive enough”; the
unknown final numeric volume is not zero.

## Visualization result

- Repaired board: exact `1920x1080`, SHA-256
  `3880232e698a14936d5a0f386e00defe3d55eaee2aa1a2e0d93fb7b7ffc7e802`.
- Frozen Rerun screenshot: `3840x2160`, SHA-256
  `221aae6acaa87f6c3db74cd2bd82fa5e59ad2368510256b31a5784e9528ad42d`.
- RRD/RBL strict validation: PASS.
- Manual inspection: `10/10` PASS.
- Completion SHA-256:
  `0b04651f1f984f71d880eb52b3fa968154dd9970bbbeffcf410ae17eb4d21e30`.

Operational verdict:

`D394_FULL10_EXACT_VOLUME_SEMANTICS_AND_ASCII_VISUALIZATION_COMPLETE_NO_PAIR_OR_SEAM_ADOPTION`

## Frozen nonclaims and next boundary

- Attempt1 is a pre-worker semantic stop; attempt2 is the only numeric run;
  attempt3 is an observability-only repair.
- Call29 rank/class remain `null/null`.
- The all-41 class aggregate remains `null`.
- D389/D390 remain frozen FAIL_STOP.
- Pair/seam verdict updates: `0`.
- Collider/asset/USD/Isaac/Kit/PhysX/Warp/CUDA: `0`.
- Cylinder/physics/q5/contact/grasp and target/IK/path changes: `0`.
- Materializable candidate: `false`.
- `g0a_pass=false`.

The next conditionally approved separate case must apply the now-frozen gate
semantics to all `36 pairs / 144 directions`, preserving call29's null
rank/class and the original D389 records. Only after that aggregate may a new
seam/candidate decision be made.

## Sources

- `sim_scripts/cyl34_top_view_d394_stable_fullrank_terminal_volume_subthreshold_semantics.py`
- `sim_scripts/cyl34_top_view_d394_attempt2_gate_numeric_null_semantics_repair.py`
- `sim_scripts/cyl34_top_view_d394_attempt3_ascii_exponent_visual_repair.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt1_stable_fullrank_terminal_volume_subthreshold_semantics/d394_pre_worker_semantic_review_stop.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt2_gate_numeric_null_semantics_repair/d394_full10_volume_semantics_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt2_gate_numeric_null_semantics_repair/d394_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt2_gate_numeric_null_semantics_repair/d394_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt2_gate_numeric_null_semantics_repair/d394_visual_contract_stop.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt3_ascii_exponent_visual_repair/d394_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d394/attempt3_ascii_exponent_visual_repair/d394_completion_summary.json`
