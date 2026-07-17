# START_HERE.md

Last updated: 2026-07-17 KST. D362 consumed its one approved Isaac/PhysX invocation.
Its canonical 500-row physics trace is complete, but the exact-video and actual-viewport
observability contract failed. D362 is concluded/frozen without finalize or retry.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius is `0.017m = 17mm = 1.7cm`,
  diameter `0.034m = 34mm = 3.4cm`, height `0.090m = 90mm = 9cm`.
- q5 convention: `q5=0` CLOSED; frozen sim OPEN `q5=1.5413rad`. D347 measured
  HOME-near + q5=0 CLOSED, not exact HOME.
- D348 corrected callback-topology gate is `256/256` channels, `128/128` parts;
  live connected link5/gripper binding is `64+64`.
- D350 proved a real static Isaac Viewer/collider display, not dynamic PhysX-tensor to
  renderer synchronization. It remained `aligned_pass=null`, `g0a_pass=false`.
- D354 zero-step contact order remained unresolved at the cylinder top cap/rim boundary.
- D359 recovered the historical hash generator as original-point-ID ordering; no
  geometry/unit/GPU/Warp cause and no old gate rewrite.
- D360 actual physics failed after 243 controlled rows because total detailed-contact
  capacity was 16; body/value rows were lost and all physical claims stayed null.
- D361 registered capacity `33,280` and proved the durable prefix protocol offline, but
  did not run physics.
- D362 now supplies the first durable body/value actual-physics answer. It does not
  prove grasp or complete G0a.

## D362 Completed Physical Sub-result

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d362/` — exactly 33 files,
  immutable; do not add, overwrite, rename, resume, retry, or finalize.
- One `headless=false` RTX/GUI Isaac worker completed OPEN baseline `200/200` and
  q5-close `300/300`; controlled physics `500`, q5 target update `1`, worker exit `0`.
- OPEN baseline had no registered robot contact or threshold cylinder motion.
- Moving `gripper_link` contact onset/confirmation: closure step `31/32`.
- Cylinder motion onset/confirmation: step `41/42`, ten physics steps (`0.05s`) after
  moving-jaw onset.
- Fixed `link5` contact onset/confirmation: step `45/46`; link4 registered positive
  count/force event: `0` in all 500 rows. This does not prove absolute no-contact.
- Peak moving-gripper/link5/link4 filtered force:
  `43.8583399/23.2278653/0.0N`.
- Endpoint cylinder XY/tilt/z change:
  `60.6189978mm/89.9977746deg/-28.0005205mm` — it was pushed over, not held.
- Physical sub-verdict:
  `D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`.
- Exact face/manifold, cap/rim/barrel order, force closure, stable grasp, hold/lift,
  target/IK repair justification remain `null`; `g0a_pass=false`.

## D362 Capacity / Evidence Completion

- Runtime sensor cfg, derived envelope, and PhysX backend max were all `33,280`.
- Actual collider inventory was cylinder/table/link4/link5/gripper `1/1/1/64/64`;
  six backend detailed-contact buffer shapes matched the preregistration.
- Observed contact-point high-water was `22/33,280`; registered overflow warnings `0`.
  This closes D361 runtime sufficiency only for this exact inventory/version/run.
- Durable prefix SHA-256 `aa7f7419516f4dda723290d89389df680ef3336f2b16984d4f467b76eee41a8e`:
  500 observations, 1,002 records, sealed, tail 0, terminal inflight null.
- Supervisor recovery classified it `COMPLETE_SEALED_PREFIX`; no watchdog, worker
  exception, retry, target/IK/path, or physical-setting change occurred.

## D362 Overall FAIL_STOP — Observability, Not Physics Crash

- Overall verdict:
  `D362_SINGLE_INVOCATION_PHYSX_TRACE_COMPLETE_OBSERVABILITY_FAIL_STOP`.
- Trace-replay MP4 was H.264/yuv420p, 250 frames, 20fps, full-decode PASS, but
  imageio `macro_block_size=16` resized registered `1920x1080` to `1920x1088`.
  Video report/phase contract therefore failed; supervisor outer exit was `2`.
- Manual original-resolution MP4 quadrants, storyboard, and beginner sheet were
  legible; the exact-resolution gate is still binding and was not relaxed.
- More importantly, precommand/contact/motion/final actual Isaac PNGs all retained
  the same upright cylinder bbox `(628,299,90,209)` with centroids within 0.1px,
  while the canonical final trace recorded 60.619mm motion and ~90deg toppling.
- Therefore actual viewport temporal/state synchronization was not established.
  Rerun/video correctly displaying the canonical trace does not repair stale Isaac PNGs.
- Confirmed implementation gap: controlled steps used `sim.step(render=False)` with
  default `update_fabric=False`, then capture used only `app.update()` without the
  `forward()` Fabric flush used by IsaacLab render and D350's static state-write path.
  The separate `Failed to clone in Fabric` log is not proven as another single cause.
- No automated summary, manual PASS artifact, or completion/finalize artifact exists.

## Active Case / Next Approval Boundary

- No case is currently approved for execution.
- Recommended separate candidate: D363
  `[d362_trace_replay_1080_and_isaac_render_sync_repair]`, observability-only.
- Candidate variable 1: regenerate the immutable D362 trace in a new forward-only
  output as exact 1920x1080 H.264/yuv420p; no physics/q5 recomputation.
- Candidate variable 2: apply one preregistered zero-step Fabric flush at the four
  recorded decision states, then compare rendered cylinder pose with canonical trace.
- D363 would allow no controlled physics step, q5 science sample/update, cap/rim
  discriminator, target/IK/path or asset/decomposition/gate/material/mass/actuator/
  solver/physics change. It requires a new explicit user approval.
- Any later cap/rim science, target/IK repair, settle/grasp/G0a, ten-trial, G0b,
  RL/PPO/VLA, or ladder promotion requires another separate approval.

## Frozen Boundaries

- Freeze D351-D362 evidence and paths. Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- Do not substitute Rerun Float32 display values or vertex-only Qhull for canonical
  callback/Float64/sensor evidence.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, unapproved signal,
  dependency install, commit, or push.

## Operational Residue

- Post-D362 check found no active Isaac/Kit/D342/D362 worker.
- Isaac telemetry transmitter PID `719072` remains; it is not the simulator worker and
  was not signaled. The older recorded D342 PID `1729639` no longer exists.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D360-D362; ledger tail
2. `claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md`
3. D362 runtime prerequisites, prefix/audit, physics trace, worker/video/Rerun/supervisor
4. `claudedocs/session_20260716_grasp_g0a_d361_contact_point_capacity_and_prefix_trace_repair.md`
5. D360 session/prerequisites/phase/exception/raw log/supervisor
6. D359 provenance and D354 measurement/binding/attestation/completion evidence
7. D350 static Viewer, D348 topology, D353/D352 bridge, D351 original/repair

## Git

- Verified `HEAD == origin/master ==
  68f2ff040831c13b0198fe68ef88fe84a76a9df3`, commit `D361완결 및 동결`.
- Worktree was clean before D362. Current uncommitted changes are the forward-only
  D362 harness/output/session plus START_HERE/DECISIONS/ledger updates.
- No commit or push is authorized or performed in D362.
