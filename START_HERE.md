# START_HERE.md

Last updated: 2026-07-18 KST. D365 observability-only localization completed and is frozen.
No further runtime, repair, or q5/physics science case is currently approved.

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
- D362 is the current physical authority. D363-D365 changed only observability/control evidence.

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

## D363 Completed Observability Case

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d363/`; completion verdict
  `D363_OBSERVABILITY_OR_INTEGRITY_FAIL_STOP`, `pass=false`. Freeze/no rerun.
- Exact frozen-trace replay passed at `1920x1080`, 250 frames, 20fps, 12.5s,
  H.264/yuv420p/full decode. D362's existing 1088-height file remains immutable.
- Four AssetData writes plus four explicit `SimulationContext.forward()` calls used zero
  physics/q5/contact, but all 16 actual Isaac views stayed upright. Primary
  precommand→final centroid/axis/IoU was `0.049105px/0.049759deg/0.996594`.
- Rerun/footer/RBL/screenshot and all-frame video inspection passed but did not repair Hydra.
- D363 bit-exact readback was AssetData cache self-read, not independent PhysX evidence.
  Generic `Failed to clone in Fabric`, CPU/PCIe/VRAM/Warp/SM are not proven causes.

## D364 Completed Pre-write Localization Attempt

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d364/`; freeze/no rerun/overwrite.
- Operational verdict:
  `D364_PREWRITE_OPTIONAL_FABRIC_COMPATIBILITY_ATTRIBUTE_MISMODELED_FAIL_STOP`.
  `localization_verdict=null`, `g0a_pass=false`.
- Prepare passed. One actual `headless=false` worker reached launcher, env, reset, paused
  timeline, two baseline layer records, and two guarded 1280x720 captures; no retry.
- Root compatibility `_worldPosition/_worldOrientation/_worldScale` attrs were absent, so the
  preregistered required-layer gate stopped before mutation. This does not mean Fabric failed:
  root and render-mesh hierarchy local/current/cached matrices were valid.
- Independent PhysX and Fabric root-hierarchy baseline matched exactly at position
  `[0.3000000119,0,0.0328829996]m`, quaternion wxyz `[1,0,0,0]`. Independent static
  root→mesh reconstruction max-abs error was `0.0`.
- Actual counts: cylinder pose write `0`, explicit forward `0`, controlled physics/q5
  science/q5 target/contact query `0`, invocation/retry `1/0`.
- Both manually inspected baseline views were upright. Primary/opposite yellow area
  `17003/17010px`; no post-write/final captures, report, or RRD exist by program order.
- Outer process exit `0` was misleading: worker exception + terminal stop + missing worker
  summary are completion authority. Never use worker exit code alone as PASS.
- GPU used max/free min `7760/8185MiB`, utilization max `22%`, worker RSS max
  `7,055,708,160B`; watchdog null.

## D365 Completed Hierarchy Localization

- Output: `claudedocs/runtime_logs/grasp_track/g0a_d365/`; freeze both the sandbox-access
  attempt1 and completed `attempt2_host_access_prepare_repair/`; no rerun/overwrite.
- Attempt2 prepare `19/19` PASS. One actual `headless=false` worker/no retry completed in
  `28.8189958s`, exit `0`, watchdog null. Pose write/forward were `1/1`; controlled physics,
  q5 science, q5 target, contact query were all `0`.
- AssetData cache and independent PhysX tensor view became `TARGET` immediately after write.
  IFabricHierarchy root current, mesh current, Boundable mesh cached worldMatrix, and Hydra
  pixels remained `BASELINE` even after public `SimulationContext.forward()`.
- Optional compatibility `_world*` remained `UNAVAILABLE` and optional root cached worldMatrix
  remained `BASELINE`; neither gated the verdict. Selected Fabric callable was the actual bound
  `force_update`; Fabric/FSD/Hydra-transform attestations were true.
- Primary baseline→post-forward centroid/axis/IoU was
  `0.0144265px/0.0191790deg/0.999118`; opposite was
  `0.0134108px/0.00654886deg/0.997652`. All six 1280x720 captures stayed upright.
- Exact 8-path original visual inspection, RRD/RBL/footer/entity/timeline/component checks,
  manual checks `14/14`, integrity checks `19/19` PASS.
- Final verdict: `D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED`, completion `pass=true`.
  This localizes the display-state break; it does not identify the deeper missing update/commit
  event and is not a grasp-science PASS/FAIL. D362 physical sub-result and all nulls remain.

## Current Authorization Boundary

- No active approved case. Before a new runtime repair, the next candidate should first trace the
  installed public setter→PhysX scene update→Fabric hierarchy update ordering/commit contract.
- Any bridge experiment, q5/physics science, cap/rim discriminator, target/IK/path repair,
  asset/physics setting change, grasp/settle/hold/lift, ten-trial, G0b, RL/PPO/VLA needs a new
  explicit approval and forward-only preregistration.

## Frozen Boundaries / Operational Residue

- Freeze D351-D365 paths. Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- Do not substitute Rerun Float32 display values or vertex-only Qhull for canonical
  callback/Float64/sensor evidence.
- Historical D342 worker PID `1729639` remained under user-systemd PID `1123`. One previously
  approved SIGTERM was sent after lineage recheck, but the process remained `Sl`; no SIGKILL or
  unapproved signal was used. Its observed GPU allocation was 320MiB.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install,
  commit, push, or unapproved signal was performed.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D362-D365; ledger tail
2. `claudedocs/session_20260718_grasp_g0a_d365_hierarchy_current_render_cache_propagation_localization.md`
3. D365 completion, localization report, layer journal/audit, supervisor, Rerun validation,
   manual inspection, and six original Isaac PNGs
4. `claudedocs/session_20260718_grasp_g0a_d364_paused_render_state_layer_localization.md`;
   D364 runtime attestation, worker exception/phase markers, baseline manual inspection,
   and pre-write failure completion
5. `claudedocs/session_20260718_grasp_g0a_d363_trace_replay_1080_and_isaac_render_sync_repair.md`
6. D363 completion, sync report, exact-video report, manual inspection, Rerun validation,
   worker log and supervisor summary
7. `claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md`
8. D362 runtime prerequisites, durable prefix/audit, physics trace, worker and supervisor;
   then D361/D360/D359/D354/D350/D348 lineage as needed

## Git

- Verified before D364/D365 edits: `HEAD == origin/master ==
  94c0644ef3d4e69278bc864f0f8c2f3a40908dc8`, commit `D363test`; worktree was clean.
- Current uncommitted changes contain frozen D364 and completed D365 evidence/state/harness.
  No commit or push is authorized or performed.
