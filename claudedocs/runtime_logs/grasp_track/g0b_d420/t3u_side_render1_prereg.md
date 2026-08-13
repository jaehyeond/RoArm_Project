# T3U side-preflight13 RTX render-only repair 1 preregistration

Status: **FROZEN BEFORE EXECUTION / INDEPENDENT GO STILL REQUIRED**.
This is a forward-only repair of the
posthoc RTX visualization path.  It is not a physics retry and cannot change
the P13 measurements, classifications, or scientific authority.

Render executable:
`sim_scripts/p16_g0b_t3u_side_preflight13_rtx_render_repair_v1.py`.

Host lifecycle supervisor:
`sim_scripts/p16_g0b_t3u_side_preflight13_rtx_render_repair_supervisor_v1.py`.

## Why this repair exists

The original P13 physics child completed all 2,340 task steps and durably wrote
its JSON/NPZ/RRD/RBL/PNG evidence.  Its separate RTX render child then stopped
making observable progress inside the Replicator capture path and produced no
PNG or MP4.  This repair replays only the already-frozen P13 body/object
transforms with the installed Replicator API's explicit zero-time step call.

## Frozen P13 inputs

All paths are relative to the repository root.  Every byte is checked before
the first new output is created and again before the manifest is committed.

| Input | SHA-256 |
|---|---|
| `sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v12.py` | `f0c88e73a2ebf0c20e48ecbfa5bf672bc11d5adab523ca1c9817d96e1a511999` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_script.py.txt` | `f0c88e73a2ebf0c20e48ecbfa5bf672bc11d5adab523ca1c9817d96e1a511999` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_prereg.md` | `ed3d3f5afd9a2ff5f01f341322367a6f68b168ca3760d67a7465b9e95d8864da` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_results.json` | `8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_plan.json` | `d7fcfb47c26c38f4817ce7630671d915e0d77a4b3bcc1f2d7df40fd816f94f66` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_trace.npz` | `ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_timeline.rrd` | `3235ae954121a9218252785e254e037460176c3699b1e86d9c044ea2187a5601` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_timeline.rbl` | `2c0ae6c1672720486105924fa55f60ec820596051a480208cd4d7a4ff363f34d` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_rerun_validation.json` | `526bd29020577b483e21a3b8686ba07eb170a5aff1ba0169ad4b7e52f41ec1e4` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_decision_snapshot.png` | `f0ffd1607061c4dfdc60388f22760299771225fa0950bf8c24d36b29999542bf` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_inspection.png` | `2f0e164424dea7b5d28382675736129b0c0840032865c7f68c4530ee0b9d0aa2` |
| `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight13_preclose_sentinel.json` | `228ae61d6d2df02fdac10a065729caa30cc2bf5a54c33642da755ebb77f2d401` |

The repair also loads the frozen V12 module under a path-and-content-unique
module name and requires its complete `RESULT_SEMANTIC_CHECK_KEYS` map to be
exactly present and all `True`.  Independent direct checks additionally require
P13 `runtime_instrumentation_pass=true`, Rerun technical PASS, five valid
measurements, zero successes, the exact representative binding
`c05_o00 / candidate 5 / offset 0 / slot 0`, and the frozen trace hash recorded
inside `results.json`.

The full original P13 render-decision closure is also mandatory: frozen V12
must recompute all 245 keys returned by
`render_dependency_snapshot(side_preflight13)`, and that exact path/hash map
must equal both `results.provenance.dependency_hashes_at_start` and
`results.provenance.dependency_hashes_at_finalize`.  The same 245-key map is
rehash-compared after MP4 decoding.  This transitively binds the P15 candidate
artifact, the complete Attempt3 composed USD asset set, robot/jaw/URDF inputs,
and every already-frozen historical dependency used by the P13 decision.

## Frozen replay and renderer contract

- P13 physics is never launched.  No Isaac Lab environment, `SimulationContext`,
  physics scene, articulation controller, reset, or physics-step API is created.
- The timeline is stopped before stage construction.  A PhysX step-event
  subscription, timeline time/state, and `SimulationManager` counters are
  sampled before and after every capture.  Any observed change is fatal.
- Moving-body and cylinder transforms come directly from P13 `trace.npz`.
  Every authored transform matrix is compared with its source at an absolute
  tolerance of `1e-12` before and after capture.
- Scene, materials, Attempt3 robot reference, camera pose/intrinsics, and
  overlay match the frozen P13 posthoc renderer.
- Attempt3 stage composition has a 120-second bounded load wait using audited
  stopped-timeline app updates; capture cannot start while unresolved stage
  assets remain.
- Installed renderer contract: Isaac Sim 5.1 / Kit 107.3 and
  `omni.replicator.core 1.12.27`.  Capture-on-play is explicitly disabled.
  Every capture uses exactly `rep.orchestrator.step(rt_subframes=1,
  pause_timeline=True, delta_time=0.0, wait_for_render=True)`, followed
  immediately by annotator `get_data()`.  There is no extra per-frame
  `simulation_app.update()`.  This is the installed static/stopped-timeline
  pattern: zero delta prevents time advancement and `wait_for_render=True`
  binds annotator data to the just-authored state.
- Physics cadence is 200 Hz.  Frames map to task steps 10, 20, ..., 2340:
  exactly 234 RGB PNGs at 1280x720, encoded at 20 fps (11.7 seconds).
- Exactly six warmup captures plus exactly 234 written captures are performed.
  Warmups are never output PNGs but must satisfy the same zero-clock,
  zero-physics, capture-on-play-false, and transform-fidelity gates.  Progress
  is flushed to stdout and the phase ledger after every warmup and at least
  every 10 written frames.
- Each PNG uses exclusive creation and is decoded immediately.  MP4 creation
  uses ffmpeg no-clobber mode and must fully decode to exactly 234 frames at
  1280x720 and 20 fps before the final manifest is written.

## Forward-only output boundary

All generated artifacts use only the new prefix
`claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_render1_*`.
The repair refuses to start if any owned output or frame directory already
exists.  It never creates, deletes, replaces, or appends to any
`t3u_side_preflight13_*` file.  A failed attempt remains durable and this tag is
retired; any later attempt requires a new forward-only tag.

Expected main outputs are:

- `t3u_side_render1_input_gate.json`
- `t3u_side_render1_phase.jsonl`
- `t3u_side_render1_rgb_frames/frame_0000.png` ... `frame_0233.png`
- `t3u_side_render1_rgb_frames_manifest.json`
- `t3u_side_render1_side_grasp.mp4`
- `t3u_side_render1_failure.json` only on failure

The host supervisor additionally owns only the new-prefix PID/PGID, stdout,
GPU-before/after, outcome, exit-status, and supervisor-failure evidence files.
It creates no P13-prefixed output.

## Scientific meaning and launch boundary

This video is explicitly `scientific_authoritative=false` and
`render_is_posthoc_observability_only=true`.  P13 currently reports valid
instrumentation but `success=0/5`: one row is classified
`premature_jaw_contact` and four rows are `no_bilateral_close`, with population
selection `NO_BILATERAL_SIDE_CONTACT`.  The video must not be described as a
successful grasp.  This standalone attachment cannot satisfy, replace, repair,
or complete the original P13 terminal-attestation contract, and it cannot raise
P13's scientific authority or authorize canonical execution.

Execution is forbidden until the original host P13 supervisor/render PIDs have
actually exited according to host-visible evidence and an independent frozen
source/prereg audit returns GO.  The repair then runs once under a host-visible,
bounded supervisor.  Its exact lifecycle limits are: capture-start/launch
deadline 120 seconds; once capture has started, maximum phase-ledger no-progress
interval 90 seconds; total child wall time 900 seconds; SIGTERM to the child
process group, then SIGKILL after 20 seconds if necessary.  The single child PID
and PGID must be reaped, and GPU PID sets sampled before/after must show no fresh
survivor.  A synchronous Replicator stall is therefore killed and recorded
under the new prefix rather than waiting indefinitely.  Exactly one attempt,
zero automatic retries, and no output reuse are permitted.

The final freeze receipt publishes the preregistration, render source, and
supervisor SHA-256 values after all three become immutable.  They use one-way
pinning (render source pins this preregistration; supervisor pins both) rather
than an impossible self-referential hash cycle.
