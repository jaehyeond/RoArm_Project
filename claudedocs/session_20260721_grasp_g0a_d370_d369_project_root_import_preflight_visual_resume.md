# D370 — D369 project-root import preflight and visual resume (observability only)

Date: 2026-07-21 KST  
Case: `g0a_d370`  
Status: one-shot run frozen at phase 9/14; import and Viewer launch PASS, visual completion FAIL

이번 case의 신규 변수: [`production_command_repo_root_import_preflight`]

## 1. What and why

D369 already serialized a display-only presentation RRD and RBL from immutable D368 evidence, but
its one approved host worker stopped before the Rerun Viewer. A late direct-script import of
`roarm_rl.rerun_contract` failed because the repo root was not in Python's module search path.

D370 changes only that launch contract. It must prove, before the one host worker, that the same
absolute Python executable + `-B` + direct harness path used by production can import the exact
repo-local helper with ambient `PYTHONPATH` removed. It must also deliberately disable the bootstrap
in a no-output negative subprocess and reproduce the D369 `ModuleNotFoundError`. This is a
failure-capable control repair directly responding to D369; it is not new geometry or physics work.

## 2. Frozen input authority

Only these immutable domain/display inputs may be read:

- D368 evidence JSON: `be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5`
- D368 RRD: `f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0`
- D369 preregistration: `f991aaeefd88d5066773b573d2a540b8c7d5658ef4252a792d9f422378338642`
- D369 phase stream: `5ddae9f25f33fab446954e197b478af937fd3ded458cdf59aa8526ec01457efa`
- D369 runtime exception: `0b338694f77d34f910764ed99f5be6e61d1a554cd347c53be978d0496e47d1bb`
- D369 D368-bitexact copy: `f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0`
- D369 recording-only display copy: `ce00df2fbb95630e58439e9d7fd13afd56e27d3386581da8c71edac902f2403e`
- D369 static text overlay: `1df88ad1a0aad052d3ba879e49aa10164cc878a8793fc61a07dea8dda79d9cc2`
- D369 RBL: `429407b11120167655c059085e8f3f4ef81191d49f4b5728dc20cfdfda45e216`
- D369 presentation RRD: `0f394dec88ad1d253d5c4e0996e80a01752b272b3a4e04ff0a0f0de439302aab`

The repo helper `roarm_rl/rerun_contract.py` is executable contract code, not scientific input; its
expected SHA-256 is `aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e`.

## 3. Exact import preflight contract

`prepare` may invoke the D370 harness in an import-only stage. Both child commands use the exact
production executable/direct-script prefix and remove ambient `PYTHONPATH`:

1. bootstrap enabled: return `0`, repo root present exactly once, helper path/hash and required
   symbols exact;
2. bootstrap disabled: return a preregistered nonzero code with `ModuleNotFoundError`, repo root
   absent, and no D370 output mutation.

The accepted baseline plus targeted perturbations must reject a disabled/missing bootstrap, wrong
helper path/hash, ambient-path dependence, changed D369 source hash, a second host worker, or a
second Viewer invocation. These controls may fail without consuming the approved host worker or
Viewer because they do not render and write only the final preregistration artifact.

## 4. One-shot display resume

After preregistration and static review, the actual host worker may run exactly once/no retry with
`D370_HOST_RENDER_APPROVED=1`. It must:

1. recheck the in-process import attestation, Git base, dirty scope, and frozen source hashes;
2. copy the frozen D369 presentation RRD/RBL bit-for-bit into `g0a_d370/`;
3. verify footer, exact entity/timeline/component sets, `4 Spatial3DView + 2 TextDocumentView`, zero
   `DataframeView/TimeSeriesView`, and exactly one matching blueprint activation;
4. run one local-loopback bind/listen capability check;
5. write the invocation artifact before calling Rerun Viewer at most once;
6. preserve the original Viewer PNG and derive one exact `1920x1080` professor board;
7. write automated validation, then stop for original-resolution human inspection;
8. run `finalize` only after the two PNGs are manually inspected and hash-bound.

The D369 presentation is not rebuilt or merged again. Rerun pixels remain inspection evidence; D368
JSON/RRD remain numerical authority.

The professor board must also preserve D369's preregistered 21-item fact-card plan, title,
subtitle, and footer rather than introducing new D370 display content. D370 lineage belongs in the
new output path, validation artifacts, manual inspection record, and this session document; it is
not stamped into the frozen D369 board pixels.

## 5. Registered phase sequence

The host render worker must stop after the first 12 forward-only phases. The later read-only
`finalize` stage may append only phases 13-14 after both PNGs have been inspected at original
resolution and the inspection records have been hash-bound.

```text
render_worker_started
production_command_import_attestation_rechecked
frozen_d369_manifest_verified
frozen_d368_authority_verified
active_rrd_rbl_bitexact_copies_written
pre_render_artifact_contract_pass
host_loopback_bind_capability_pass
one_shot_viewer_invocation_recorded
one_shot_viewer_returned
raw_png_automated_gate_pass
professor_board_written
automated_validation_finalized
manual_visual_inspection_contract_pass
completion_gate_pass
```

## 6. Frozen boundary

Host worker `1`; automatic retry `0`; Rerun Viewer `<=1`. Collider generation/recook/decomposition,
Isaac/Kit/PhysX/SimulationApp, q5, physics step, contact query, target/IK/path, USD/assets,
material/mass/actuator/physics settings, Warp/CUDA compute, and `nvidia-smi` remain exactly zero.

`current_64cap_optimal`, `physics_equivalence`, `collider_count_tipping_causality`,
`actual_gpu_contact_execution`, and `grasp_feasibility` remain `null`; `g0a_pass=false`.
D368-D369 paths and the user-owned D334 collision-table sidecar are immutable. No commit/push is
authorized. Collider candidate comparison and all physical science require separate approval after
D370 reports.

## 7. Prepare result

`prepare` ran once with ambient `PYTHONPATH` removed and returned `0`.

- Preregistration: all `18/18` checks PASS.
- Production-command import baseline: return `0`, repo root at `sys.path[0]`, repo-root occurrence
  count `1`, exact helper path and SHA-256
  `aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e`, all required symbols
  callable, exact Python/direct-script/cwd/`-B` contract PASS.
- Bootstrap-disabled control: return `86`, repo-root occurrence count `0`, exact
  `ModuleNotFoundError`, `exc.name=roarm_rl`, message `No module named 'roarm_rl'`.
- Failure-capable baseline/perturbations: `9/9` PASS.
- Frozen D369 professor-board plan: `21` items, `y_end=548`, canonical SHA-256
  `adf0973b417ccb54ac6c8e086ab5a713a87555e13a95f48d203332fe95353dea`, PASS.
- Preregistration SHA-256:
  `8f8ecd0f1fd697c124cf6cfdc8b687c206b3979c1c1da23ae2ba0c5a3c475a09`.

Source: `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_preregistration.json`.

## 8. One-shot execution in observable order

The approved host worker was invoked exactly once with the same direct-script prefix and ambient
`PYTHONPATH` removed. No retry was attempted.

1. `render_worker_started`.
2. The in-process import attestation passed and was serialized.
3. The exact frozen D369 manifest and D368 authority chain passed.
4. The D369 presentation RRD and RBL were copied bit-for-bit into the D370 path.
5. The inherited footer/entity/timeline/component/view/activation contract passed before Viewer.
6. The loopback bind/listen probe passed.
7. The invocation artifact was durably written before Viewer.
8. Rerun Viewer returned `0` after `0.7552606039680541s` and emitted a `3840x2160` PNG.
9. The run then stopped at the raw-PNG semantic-color gate. Registered phase 10
   `raw_png_automated_gate_pass` was not written.

The phase stream therefore contains the exact forward-only prefix `9/14`. The long outer tool wall
time included interactive privilege-approval waiting; the Viewer receipt, not that wait, is the
runtime authority.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_phase_markers.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_import_preflight_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_render_invocation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_render_receipt.json`

## 9. Quantified raw-PNG gate result

The allowed resolution and all four nonblank checks passed. Three of the four registered semantic
color signatures passed. The only failed scalar was purple in the moving-jaw full view:

| View | variance | sampled semantic pixels | Result |
|---|---:|---|---|
| link5 full | `520.563553026326` | cyan `309`, green `542`, blue `3709` | PASS |
| link5 fixed-patch zoom | `488.00306086696725` | cyan `1204`, green `1715` | PASS |
| moving jaw full | `453.73627986063883` | cyan `729`, yellow `684`, purple `23`, blue `853` | **FAIL** (`23 < 25`) |
| moving contact-patch zoom | `890.7773565800289` | cyan `2287`, yellow `2367`, purple `98` | PASS |

This is an inspection heuristic failure, not evidence that the purple outer patch is absent. The
same PNG visibly contains purple in both moving-jaw views, and the registered zoom signature has
`98` sampled purple pixels. The fixed absolute `25`-sample requirement is not yet justified as a
resolution-invariant visual gate and must not be relaxed post hoc in D370.

Raw PNG:

- dimensions `3840x2160`
- bytes `6,557,083`
- SHA-256 `7df0231a6be5e4c98cf8ce5c70896e7a7a84ee5de3b939ed632b60193b2e4a32`

Source: `claudedocs/runtime_logs/grasp_track/g0a_d370/d370_runtime_exception.json`.

## 10. Original-resolution visual inspection

The raw PNG was opened at original resolution after the stop. This inspection is diagnostic only;
it is not a completion manual artifact because the registered board was never produced.

- All four named spatial panels are present and nonblank: `link5 | all 64`,
  `link5 | fixed patch`, `moving jaw | all 64`, and `moving jaw | contact patches`.
- Both static TextDocument panels are present. The bottom status timeline reads `log_time`; there is
  no `Unknown timeline` or empty metric panel.
- In-scene anchor/normal text labels are absent, so the old geometry-label overlap is not visible.
- Purple outer-patch pixels are visibly present in the moving-jaw views; the `23/25` miss is
  consistent with a brittle strided-sample threshold rather than missing display geometry.
- The upper-right `Frozen D368 allocation` card is partially obscured by three informational Rerun
  notifications: `Listening for gRPC connections`, `Loading ...`, and
  `Headless viewer running at 1920x1080`. They are not errors, but they cover text and therefore
  independently fail a professor-readable screenshot contract.
- The lower `What 64 means / does not mean` card is visible and contains the separate
  `maxConvexHulls`, `hullVertexLimit`, five `NULL` fields, and `G0a: false` statements.

## 11. Artifact disposition

The D370 path is frozen with exactly nine files. Important hashes are:

- active RBL copy: `429407b11120167655c059085e8f3f4ef81191d49f4b5728dc20cfdfda45e216`
- active presentation RRD copy:
  `0f394dec88ad1d253d5c4e0996e80a01752b272b3a4e04ff0a0f0de439302aab`
- import attestation: `2a6b1f20ea429ba1b1a405f545e4e1893bb8f5e1b2871c19bc249cbb1e6f1417`
- phase stream: `35db1d827e1fafb7e41f68e633bcb8906f8675bb6e0540811a45b3c628963317`
- invocation: `013fb1894dcab9cf2218eb4353499073ae1cf1b709745b008d43bdf5e26a3962`
- receipt: `8d073ce0e79322cc7b3a7fd948645997a6f79a467660d24423aacfb9f7b57f80`
- runtime exception: `bbfe7602720148a7f3cde733f45cd89d26ffc45bc43c89686fab4d7c8cb005a9`

Because phase 10 failed, there is no professor board, automated summary/report, final validation,
manual JSON/Markdown, or completion summary. None will be synthesized after the fact.

## 12. Verdict and causal interpretation

Subresults:

- `D370_PRODUCTION_COMMAND_REPO_ROOT_IMPORT_PREFLIGHT_PASS`
- `D370_ONE_SHOT_RERUN_VIEWER_CAPTURE_PASS`

Overall verdict:

`D370_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`

The D369 import cause is closed: explicit repo-root bootstrap worked under the real production
command. The intended timeline-free/static-card Rerun layout also loaded. D370 nevertheless did not
complete its visual contract because (a) the absolute purple-pixel smoke gate rejected `23` against
`25`, and (b) original-resolution inspection found transient informational notifications obscuring
the upper fact card. These are two observability issues, not Isaac/PhysX/GPU-contact failures and
not collider or grasp results.

Host worker/Viewer/retry are `1/1/0`. Collider regeneration, Isaac/Kit/PhysX/SimulationApp, q5,
physics step, contact query, target/IK/path, USD/assets, material/mass/actuator/physics changes,
Warp/CUDA compute, and `nvidia-smi` are all `0`. The five withheld science fields remain `null` and
`g0a_pass=false`.

## 13. Next authorization boundary

D370 must not be retried, overwritten, salvaged into a board, or finalized. A next case is not yet
approved. The narrow candidate should first preregister two observability variables at most:

1. a bounded post-load screenshot-stability contract that prevents Rerun informational
   notifications from covering the frozen cards; and
2. a resolution-normalized semantic-presence gate justified from the immutable D370 PNG rather
   than lowering `25` post hoc.

That candidate must continue to use the frozen D369 presentation and must not run Isaac/PhysX/q5,
physics/contact, collider regeneration, target/IK/path, or physical science. Collider Pareto and
actual grasp work remain later, separately approved cases.
