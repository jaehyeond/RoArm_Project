# START_HERE.md

Last updated: 2026-07-19 KST. D367 actual worker `1/1` is complete and frozen. Its PLAY
zero-step bridge subresult passed, but preregistered overall completion failed only at an
unreachable post-`SimulationApp.close()` marker. No D366 measurement resume is active.

## Current Truth

- Pivot: cylinder grasp-track G0a. Cylinder radius is `0.017m = 17mm = 1.7cm`,
  diameter `0.034m = 34mm = 3.4cm`, height `0.090m = 90mm = 9cm`.
- q5 convention: `q5=0` CLOSED; frozen sim OPEN `q5=1.5413rad`. D347 measured
  HOME-near + q5=0 CLOSED, not exact HOME.
- D348 corrected callback-topology gate is `256/256` channels, `128/128` parts;
  live connected link5/gripper binding is `64+64`.
- D350 proved a real static Isaac Viewer/collider display, not dynamic PhysX-tensor to
  renderer synchronization. It remained `aligned_pass=null`, `g0a_pass=false`.
- D354 zero-step contact order remains unresolved at the cylinder top cap/rim boundary.
- D359 recovered the historical hash generator as original-point-ID ordering; no
  geometry/unit/GPU/Warp cause and no old gate rewrite.
- D360 failed before durable physical evidence because total contact-point capacity was 16.
  D361 registered capacity `33,280` and proved the durable prefix protocol offline.
- D362 is the current physical authority. D363-D367 changed only observability/control evidence.
- D367 directly proved raw PLAY pending-state commit behavior with zero controlled physics, but
  its overall completion is not PASS because the installed terminal close cannot return to write
  the preregistered post-close marker.

## Latest D367 — PLAY Bridge PASS, Overall Completion FAIL

- Output `claudedocs/runtime_logs/grasp_track/g0a_d367/` is frozen; no rerun/retry/overwrite.
- Prepare `18/18`, failure-capable negative controls `34/34`; actual worker/retry `1/0`.
- Raw `(playing,stopped)` was directly
  `(false,false) → (false,false) → (true,false) → (true,false)`: raw PLAY stayed pending and
  the one new MainThread commit applied it.
- Inherited PAUSE commit `1`, PLAY attempt/call/return `1/1/1`, total runtime commit `2`.
  Exact one PLAY callback(type `0`) occurred inside the `7,010,253ns` commit window.
- Timeline/SimulationContext/custom counter/physics callback/joint/cylinder bits were exact
  invariant. Hash-bound zero-step attestation `7/7` makes controlled physics steps `0`.
- Cylinder write/step/public forward/q5 science/q5 target/contact query were all `0`.
  Bridge subresult: `D367_TIMELINE_PLAY_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`.
- Cleanup reached observer/callback release, `inner.close()` end, then
  `SimulationApp.close():start` and `Simulation App Shutting Down`. The installed close calls
  terminal `shutdown_and_release_framework()`; Python cannot write a post-return marker.
- Because that marker was nevertheless preregistered as an overall gate, original completion is
  preserved as `pass=false`, `D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP`. This is a cleanup-contract
  mismatch, not a failed PLAY bridge or physics result.
- PhysX-Fabric-Hydra, cap/rim, grasp, target/IK/path science remains `null`; `g0a_pass=false`.

## Latest Completed D366 Safe Stop

- Output `claudedocs/runtime_logs/grasp_track/g0a_d366/` is frozen/no rerun/overwrite.
- Prepare/worker/runtime prerequisites passed `22/22`, `20/20`, `17/17`; one actual worker
  reached baseline, then raw PLAY guard recorded `playing_not_stopped=false` and stopped before
  write/step/forward. Exact post-request tuple was not serialized.
- Program-order write/step/forward/callback/q5/target/contact counts were all zero, but the
  registered bridge was incomplete so `controlled_physics_steps=null`.
- Cleanup phase-inactivity watchdog occurred at `204.97441041003913s`; final verdict
  `D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP`, science `null`, `g0a_pass=false`.

## D362 Frozen Physical Sub-result

- Output `claudedocs/runtime_logs/grasp_track/g0a_d362/` is an immutable 33-file tree.
- One actual `headless=false` Isaac/PhysX worker completed OPEN `200/200` and q5-close
  `300/300`: controlled physics `500`, q5 target update `1`, worker exit `0`.
- Moving `gripper_link` contact onset/confirmation: closure step `31/32`.
- Cylinder motion onset/confirmation: step `41/42`, ten steps (`0.05s`) after moving-jaw
  contact onset. Fixed `link5` contact onset/confirmation: `45/46`.
- Link4 registered positive count/force event was `0` in this 500-row filter; this is not
  an absolute proof that link4 never touched.
- Peak moving-gripper/link5/link4 force: `43.8583399/23.2278653/0.0N`.
- Endpoint cylinder XY/tilt/z change:
  `60.6189978mm/89.9977746deg/-28.0005205mm`; it was pushed over, not held.
- Physical sub-verdict: `D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`.
- Exact face/manifold, cap/rim/barrel order, force closure, stable grasp, hold/lift,
  target/IK repair justification remain `null`; `g0a_pass=false`.

## D363-D365 Frozen Observability Lineage

- D363 regenerated the immutable D362 trace at exact `1920x1080`, but its actual Isaac views
  stayed upright; cache self-read was not independent PhysX proof. Completion failed.
- D364 stopped before mutation because absent compatibility `_world*` attrs had been modeled as
  required. This did not prove Fabric failure; write/forward/physics were `0/0/0`.
- D365 removed only that invalid prerequisite. One pose write made asset cache and independent
  PhysX tensor TARGET, while Fabric hierarchy/mesh/cache and Hydra stayed BASELINE through one
  public forward. It localized the measured break at PhysX tensor→Fabric hierarchy under the
  paused zero-step route; it did not prove the missing update mechanism or any grasp science.
- All D363-D365 outputs and visual evidence remain frozen. Generic clone warnings, CPU/PCIe,
  VRAM, Warp or SM efficiency were never proven causal.

## Current Authorization Boundary

- D367 control-only authorization was consumed by one actual worker/no retry; no active runtime
  case remains.
- The user's conditional statement about resuming D366 after a D367 overall PASS was not
  consumed: bridge subresult PASS와 overall completion FAIL을 먼저 보고하고 새 forward-only
  경계를 다시 승인받는다.
- Next narrow choices are a separately approved offline terminal-close attestation contract
  repair, or an explicit user acceptance of the bridge/overall split followed by a newly named
  one-write/one-step/one-forward measurement case.
- Cylinder write, physics step, public forward, q5/contact or science remain unauthorized here.
- Any q5/physics science, cap/rim discriminator, target/IK/path repair,
  asset/physics setting change, grasp/settle/hold/lift, ten-trial, G0b, RL/PPO/VLA needs a new
  explicit approval and forward-only preregistration.

## Frozen Boundaries / Operational Residue

- Freeze D351-D367 paths. Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- Do not substitute Rerun Float32 display values or vertex-only Qhull for canonical
  callback/Float64/sensor evidence.
- Historical D342 worker PID `1729639` remained under user-systemd PID `1123`. One previously
  approved SIGTERM was sent after lineage recheck, but the process remained `Sl`; no SIGKILL or
  unapproved signal was used. Its observed GPU allocation was 320MiB.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install,
  commit, push, or unapproved signal was performed.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D353 and D362-D367; ledger tail
2. `claudedocs/session_20260719_grasp_g0a_d367_timeline_play_pending_state_commit_localization.md`
3. `claudedocs/session_20260719_grasp_g0a_d366_tensor_step_fabric_visibility_commit.md`;
   D366 worker exception/phases, runtime prerequisites, supervisor, completion, baseline manual
   inspection, and process cleanup audit
4. `claudedocs/session_20260718_grasp_g0a_d365_hierarchy_current_render_cache_propagation_localization.md`
5. D365 completion, localization report, layer journal/audit, supervisor, Rerun validation,
   manual inspection, and six original Isaac PNGs
6. `claudedocs/session_20260718_grasp_g0a_d364_paused_render_state_layer_localization.md`;
   D364 runtime attestation, worker exception/phase markers, baseline manual inspection,
   and pre-write failure completion
7. `claudedocs/session_20260718_grasp_g0a_d363_trace_replay_1080_and_isaac_render_sync_repair.md`
8. D363 completion, sync report, exact-video report, manual inspection, Rerun validation,
   worker log and supervisor summary
9. `claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md`
10. D362 runtime prerequisites, durable prefix/audit, physics trace, worker and supervisor;
   then D361/D360/D359/D354/D350/D348 lineage as needed

## Git

- Verified at D367 boot:
  `HEAD == origin/master == 9f956a42db1bb43c817ffe435a4e9698707049f1`, subject
  `D366`; worktree was clean before D367 preregistration edits.
- Current authorized uncommitted scope is limited to D367 harness/session/output plus
  `START_HERE.md` and post-result append-only state docs. No commit/push is authorized.
