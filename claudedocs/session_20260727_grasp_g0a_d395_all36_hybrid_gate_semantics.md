# 2026-07-27 — Grasp G0a D395 all-36-pair hybrid gate semantics

## What and why

`D395 [all36_pair_144direction_gate_semantics_propagation]` audited the complete
`36 pairs x 4 directional contexts = 144` denominator without pretending that
D389's 41 failed Float64/Qhull calculations had succeeded.

이번 case의 신규 변수:

1. `all36_pair_144direction_hybrid_registered_gate_semantics_v1`
2. `failed41_rank_agnostic_exact_tetra_sum_upper_bound_v1`

The resulting table has deliberately mixed authority:

- 103 rows copy immutable D389 actual Float64/Qhull solver outputs.
- 41 rows keep the actual solver result and numeric volume `null`, while adding
  a separate ideal exact-halfspace subset certificate.

The table is diagnostic and not adopted into D389.

## Forward-only attempts

### Attempt1 — prepare-stage schema stop

Attempt1 stopped during prepare before any worker. It looked for the registered
volume threshold at
`frozen_contract.positive_volume_epsilon_m3`, but the immutable D389
preregistration stores it at
`frozen_constants.positive_volume_epsilon_m3`.

- worker/viewer/signal: `0/0/0`
- failure stage: `prepare`
- error: `KeyError('frozen_contract')`
- attempt1 authority SHA-256:
  `452cc6f529351e779207d946a4e93d041a96deb61ec118798edc512b80df3bb0`
- failure SHA-256:
  `3ee158a06c53ab40640c28c83bc269e08fc3dc546620b57befaa2037a9fe30b0`

Attempt1 was not overwritten or rerun.

### Attempt2 — threshold-path-only repair

Attempt2 changed only the immutable JSON lookup to
`frozen_constants.positive_volume_epsilon_m3`. It also bound the frozen
attempt1 base script, authority, and failure as inputs.

- wrapper SHA-256:
  `36f0f6a177b6ad2c03519e681c07eda0c1b57b5e837b9407d0ee9bc48999c21b`
- START_HERE SHA-256 at execution:
  `3f4df2d38a35aa634a2120c31cc13065efd59f83123ddf1bbb4dd31a790afa15`
- attempt2 authority SHA-256:
  `3a16e2e953026a3c7d4eb1922b3f2ad52ff6f73ef64ce5c9421708e5b715bcd2`
- preregistration SHA-256:
  `1e6e58eb6983d065034727c326347c002a032622b5450b4e86274aad64f4d912`
- worker/retry/signal: `1/0/0`
- worker return/runtime: `0 / 1.8573395209386945s`
- Viewer/retry: `1/0`

## Observable numeric procedure

1. Reconstructed the immutable D389 36-pair order and all 144 directional IDs.
2. Copied the 103 completed D389 directional Boolean results without rerunning
   clipping.
3. For each of the 41 failed D390 terminal point sets, enumerated every 4-point
   subset and summed the exact absolute tetrahedron volumes:
   `U(P) = sum |det| / 6`.
4. Used only the ideal mathematical rule
   `K subset conv(P) -> volume(K) <= U(P)`.
5. Kept all failed actual solver results and derived numeric volumes `null`.
6. Recombined the 103 actual outputs and 41 ideal certificates into an explicit
   mixed-authority diagnostic table.
7. Committed canonical JSON/CSV before creating visualization.

The proof does not replay remaining Float64 clips and does not bound
roundoff-induced point motion outside `conv(P)`.

## Numeric result

Canonical evidence SHA-256:
`e44b250b6177aed1089dda3627fc7719f7f8d3b43f0377e8e2109f09e25b7dae`.

- pairs/directions: `36/144`
- adjacent/nonadjacent pairs: `11/25`
- inherited actual solver successes: `103`
- ideal exact-subset certificates: `41`
- failed actual solver results still null: `41/41`
- total exact tetrahedron subsets: `42,928`
- positive/zero exact upper bounds: `23/18`
- all failed upper bounds `<= 1e-18m^3`: `41/41`
- hybrid table gate-positive/nonpositive: `26/118`
- hybrid registered pair patterns:
  - pre+post positive: `2`
  - post-only positive: `9`
  - pre+post nonpositive: `25`
- maximum upper bound: call35
  `lower_03_04_pre_float32_lbr`,
  `1.4721648449531712e-20m^3`
- frozen gate/maximum-bound ratio: `67.9271756439619`
- call29 exact upper bound:
  `1.8633477135066152e-25m^3`
- call29 rank/class: `null/null`
- actual Float64 solver pair classification: all `null`
- ideal-exact-only pair classification: all `null`

Numeric verdict:

`D395_HYBRID_103_ACTUAL_41_IDEAL_CERTIFICATE_TABLE_PASS_NO_SOLVER_REPAIR_NO_ADOPTION`

## Visualization result

- exact board: `1920x1080`, SHA-256
  `6e1c1a02137784d2d34e530ac3867adaae66d3cac760e1e3f02b79437c096f85`
- terminal-cloud RRD/RBL: strict validation PASS
- RRD decision subject: `41` normalized terminal clouds, `307` points
- auxiliary subject: `36` pair-pattern markers
- Viewer screenshot: `3840x2160`, SHA-256
  `a899d92ad5c4d5b27cb7bba270663bf2d3c30d5cf42101f9ff8e3d35bce151c1`
- manual checks: `14/14` PASS

The static Rerun screenshot has crowded auxiliary pair labels. The exact board
is the clear pair-table presentation, while the RRD remains rotatable for
terminal-cloud inspection. The message-proxy warning stays in the reserved
notification column and covers no decision subject.

## Completion and nonclaims

Completion SHA-256:
`3aa48cebadaf9922ef32aacc3c29f74ad6f3735a9ba0f986c5fdd66634fc3396`.

Operational verdict:

`D395_HYBRID_103_ACTUAL_41_IDEAL_CERTIFICATE_TABLE_COMPLETE_NO_FLOAT64_SOLVER_REPAIR_NO_ADOPTION`

- D389/D390 remain frozen FAIL_STOP.
- D389 was not modified or retroactively passed.
- The hybrid table is not adopted.
- Remaining Float64 clipping was not replayed.
- Roundoff outside `conv(P)` was not bounded.
- Failed direction volume and result remain `null`.
- call29 rank/class remain `null`.
- Collider/USD/Isaac/PhysX/Warp/CUDA: `0`.
- Cylinder/physics/q5/contact/grasp and target/IK/path changes: `0`.
- `g0a_pass=false`.

The next approved separate case must decide D388 candidate non-overlap
admissibility from D389's two direct, fully successful pre-Float32 positive
pairs. It must not use the non-adopted D395 hybrid table as decision authority.

## Sources

- `sim_scripts/cyl34_top_view_d395_all36_pair_144direction_gate_semantics_propagation.py`
- `sim_scripts/cyl34_top_view_d395_attempt2_d389_prereg_threshold_path_repair.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt1_all36_pair_144direction_gate_semantics_propagation/d395_failure_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_all36_gate_semantics_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_all144_direction_semantics.csv`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_offline_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d395/attempt2_d389_prereg_threshold_path_repair/d395_completion_summary.json`
