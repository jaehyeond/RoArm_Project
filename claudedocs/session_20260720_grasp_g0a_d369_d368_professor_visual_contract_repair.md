# D369 — D368 professor visual-contract repair (observability only)

Date: 2026-07-20 KST  
Case: `g0a_d369`  
Status: final pre-Viewer stop; see §7 (`D369_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`)

이번 case의 신규 변수:

1. `timeline_free_static_metric_overlay`
2. `label_suppressed_professor_layout`

## 1. What and why

D368 already completed one authoritative offline allocation measurement. Its measurement verdict
was `D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS`, but the human-facing visual
completion failed for two presentation defects: a static scalar `DataframeView` showed
`Unknown timeline`, and labels/2x2 summary text overlapped.

D369 repairs only those display defects. It does not reopen D368's allocation calculation and does
not answer whether 64 hulls are optimal, whether the collider is physically equivalent, why the
cylinder tipped, or whether the cylinder can be grasped.

## 2. Frozen input authority

The only scientific/display inputs read by the D369 worker are:

- `claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json`
  - bytes: `953696`
  - SHA-256: `be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5`
- `claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation.rrd`
  - bytes: `1339534`
  - SHA-256: `f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0`

The JSON supplies already-committed counts and boundary statements. The RRD supplies the already
logged Float32 display geometry. D369 does not read USD, raw meshes, witnesses, callback arrays,
D368 RBL/PNG, or any Isaac/PhysX state. The RRD is copied bit-for-bit into the new output folder;
neither the original nor that preservation copy is rewritten or decoded into a new collider.

The one Viewer input is a separate display-only `d369_professor_visual_contract.rrd`. The installed
Rerun 0.34.1 `RrdReader` selects the sole recording store from the bit-exact preservation copy and
writes a recording-only display projection, thereby excluding the inherited blueprint store. That
projection must pass `rerun rrd compare --unordered` against the bit-exact D368 copy. Only then does
`rerun rrd merge` combine it with the static D369 text overlay and new D369 RBL. This packaging does
not calculate new geometry: the final archive must contain exactly one blueprint activation, and
that identifier must equal the new RBL activation identifier.

This method was selected after a pre-preregistration `/tmp` compatibility probe rejected the first
`rrd route` design: installed Rerun 0.34.1 exited before any Viewer render with an internal footer/
chunk-order assertion on the frozen D368 file. D369 output was still empty, so no approved render or
forward-only evidence path was consumed. The replacement `RrdReader` probe returned unordered
recording-data comparison `0`, merge `0`, recording/overlay activation count `0/0`, and final
presentation/RBL activation count `1/1` with the same identifier.

## 3. Root cause and registered repair

D368 logged ten scalar rows with `static=True`, so those recording rows have no time. Its embedded
blueprint nevertheless selected a timeline-dependent `DataframeView` without an explicit valid
recording timeline. The D368 validator saw `log_time` in the combined blueprint/recording inventory
and therefore did not detect the view-level mismatch. This strongly localizes `Unknown timeline`
to the display contract, not Isaac, PhysX, or geometry.

D369 replaces the visible metric table with two static `TextDocumentView` cards. It does not invent
a time axis. The four spatial views query the frozen source/collider geometry only; D368 anchor and
normal entities carrying in-scene labels are deliberately excluded. The serialized contract is
therefore exactly four `Spatial3DView` + two `TextDocumentView`, with zero `DataframeView` or
`TimeSeriesView`.

The exact 1920x1080 professor board is derived only from that one Rerun screenshot plus the frozen
JSON fields. It uses fixed non-overlapping zones, a separate fact card, and explicit NULL/not-tested
boundaries.

## 4. Frozen numbers copied for presentation

- current callback output: `64 link5 + 64 gripper_link`;
- certified fixed patch: `12 faces / 4 parts`;
- certified moving inner: `40 faces / 17 parts`;
- certified moving outer: `36 faces / 16 parts`;
- moving classification: `16` dual inner+outer, `1` inner-only, `47` no certified contact face;
- negative controls: `8/8`;
- project-authored `maxConvexHulls=64`; installed schema default `32`; UI authoring range
  `1..2048`;
- separate `hullVertexLimit=64`; installed schema default `64`; UI range `8..64`.

The moving inner and outer counts overlap: the outer 16 are the same 16 dual carriers, so
`17 + 16` must not be presented as 33 independent parts.

## 5. Failure-capable controls

Before the one render, the same registered predicates must accept the baseline and reject:

1. a tampered D368 evidence hash;
2. a tampered D368 RRD hash;
3. a flipped `g0a_pass` boundary;
4. a flipped NULL science boundary;
5. a timeline-dependent `DataframeView` substitution;
6. inclusion of D368 anchor/normal label paths;
7. a second render invocation;
8. a compressed professor fact-card layout that cannot fit the same text;
9. a deliberately intersecting text bounding box.

Together with the accepted baseline these are ten failure-capable checks. The exact preregistered
expectation is `10/10`.

The final visual gate additionally requires no `Unknown timeline`, empty metric card, label overlap,
or error notification in the original-resolution screenshot. Numerical evidence remains D368 JSON;
Rerun and PNG pixels are inspection evidence only.

## 6. One-shot execution and immutable boundary

Execution order is `prepare` once, static review, `render` once/no retry, original-resolution manual
inspection, then control-only `finalize`. `prepare` writes preregistration only. The actual worker
must be invoked host-side on its first attempt, carry `D369_HOST_RENDER_APPROVED=1`, and pass one
non-render local-loopback ephemeral socket bind+listen probe before consuming the single Rerun Viewer
render invocation. Serialization, compare, merge, verify, stats, and print are pre-render file
checks, not additional Viewer renders.
`render` creates a bit-exact RRD copy, semantically compared recording-only display projection,
static text overlay, new RBL, one display-only merged presentation RRD, one headless screenshot,
and one derived exact 1920x1080 board. The raw Viewer
PNG may be `1920x1080`, `3840x2160`, or the previously observed platform-native `4800x2800`;
the professor board itself must be exactly `1920x1080`. `finalize` rechecks the complete hash chain
and never renders or recomputes allocation.

All of the following remain exactly zero:

```text
collider regeneration / recook / decomposition
Isaac / Kit / PhysX / SimulationApp
q5 sample or target
physics step or contact query
target / IK / path change
USD / asset / material / mass / actuator / physics-setting change
```

`current_64cap_optimal`, `physics_equivalence`, `collider_count_tipping_causality`,
`actual_gpu_contact_execution`, `grasp_feasibility` remain `null`; `g0a_pass=false`.
D368 and D351-D367 outputs plus the user-owned D334 collision-table sidecar remain immutable.
No commit or push is authorized.

## 7. Actual execution result — pre-Viewer import stop

Final status: `D369_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`  
Approved host worker attempts: `1`; Rerun Viewer invocations: `0`; automatic retries: `0`.

Execution proceeded in this observable order:

1. `prepare` ran once. `d369_preregistration.json` bound the exact D368 JSON/RRD hashes, the two
   display variables, the exact harness/helper hashes, scope guards, and the intended single-render
   contract. All registered prerequisites were `14/14` PASS and the accepted baseline plus nine
   targeted perturbations were `10/10` PASS. Preregistration SHA-256 is
   `f991aaeefd88d5066773b573d2a540b8c7d5658ef4252a792d9f422378338642`.
2. Independent static review found no blocker in the frozen harness and confirmed the preregistration
   binding. No Viewer was opened by that review.
3. The approved host-side `--stage render` worker was launched exactly once with
   `D369_HOST_RENDER_APPROVED=1`. The local loopback bind/listen capability probe passed.
4. The append-only phase stream reached exactly the first `7/12` registered phases:
   `render_started`, `host_loopback_bind_capability_pass`, `frozen_evidence_fields_copied`,
   `d368_rrd_bitexact_copy_complete`, `static_text_overlay_and_blueprint_finalized`,
   `d368_recording_only_display_copy_finalized`, and `single_presentation_archive_finalized`.
5. The next operation tried to import `roarm_rl.rerun_contract`. It raised
   `ModuleNotFoundError("No module named 'roarm_rl'")`. The Viewer invocation artifact would only be
   written later, after the missing pre-render validation phase, so the exception artifact records
   `render_invocation_exists=false` and `render_retry_forbidden=true`.

The immediate cause is a Python launch-contract defect. The harness calculates the repository root
but does not add it to the module search path. When a file under `sim_scripts/` is executed directly,
that script directory is the import root; the sibling repo package `roarm_rl` was therefore not
resolvable. This is not evidence of a GPU, Isaac, PhysX, RRD, blueprint, or Viewer-renderer failure.

## 8. Durable partial artifacts and exact hashes

The following eight files are preserved in the new D369 path and must not be overwritten:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `d369_preregistration.json` | 73,385 | `f991aaeefd88d5066773b573d2a540b8c7d5658ef4252a792d9f422378338642` |
| `d369_phase_markers.jsonl` | 860 | `5ddae9f25f33fab446954e197b478af937fd3ded458cdf59aa8526ec01457efa` |
| `d369_runtime_exception.json` | 911 | `0b338694f77d34f910764ed99f5be6e61d1a554cd347c53be978d0496e47d1bb` |
| `d369_d368_base_bitexact_copy.rrd` | 1,339,534 | `f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0` |
| `d369_d368_recording_only_display_copy.rrd` | 1,237,161 | `ce00df2fbb95630e58439e9d7fd13afd56e27d3386581da8c71edac902f2403e` |
| `d369_static_text_overlay.rrd` | 9,229 | `1df88ad1a0aad052d3ba879e49aa10164cc878a8793fc61a07dea8dda79d9cc2` |
| `d369_professor_visual_contract.rbl` | 105,470 | `429407b11120167655c059085e8f3f4ef81191d49f4b5728dc20cfdfda45e216` |
| `d369_professor_visual_contract.rrd` | 1,347,605 | `0f394dec88ad1d253d5c4e0996e80a01752b272b3a4e04ff0a0f0de439302aab` |

The final presentation RRD/RBL being serialized is only a partial operational milestone. Because the
pre-render artifact gate did not run, it is not reported as a completed visual contract. No render
invocation/receipt, raw PNG, 1920x1080 professor board, automated report, validation report, manual
inspection, or completion summary exists. `finalize` was deliberately not run because all of those
are mandatory inputs and it cannot change this decision.

## 9. Boundary and next authorization

All registered forbidden counters stayed zero: collider regeneration/recook, Isaac/Kit/PhysX,
SimulationApp, q5, physics steps, contact queries, target/IK/path changes, USD/asset writes,
material/mass/actuator/physics-setting changes, Warp/CUDA compute, and `nvidia-smi`. D368's scientific
measurement remains unchanged. The original D368 `Unknown timeline` and overlap defects have not yet
been shown repaired; `visualization_repair_pass=null`, all withheld science fields remain `null`, and
`g0a_pass=false`.

Freeze `g0a_d369`: no retry, attempt2, overwrite, post-hoc harness repair, synthetic PNG/manual
inspection, or finalize. The narrow next candidate requires separate approval and a new path:
`D370 [d369_project_root_import_preflight_visual_resume]`. It should introduce only an exact
production-command repo-root import preflight/bootstrap, hash-bind the frozen inputs, and then resume
the display-only gate and at most one Rerun Viewer invocation. It must still exclude collider
generation, Isaac/PhysX/q5/physics/contact, and target/IK/path work.
