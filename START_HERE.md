# START_HERE.md

Last updated: 2026-07-18 KST. D363 observability-only case is concluded and frozen.
Exact 1080 trace replay passed; actual Isaac viewport state synchronization failed.
No later case is approved.

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
- D362 is the current physical authority. D363 changed only its observability layer.

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
- Prepare `25/25` and worker preflight `15/15` passed. One worker exited `0` after
  `54.454797s`; watchdog/retry/postprocess exception were absent.
- Controlled physics step, q5 science sample, q5 target update, contact query were all `0`.
  Direct display-state writes and explicit `SimulationContext.forward()` were `4/4`.
- D362 33-file manifest, D334 sidecar, inputs/harness, D363-D362 inode separation,
  37 bound artifacts and precompletion inventory/hash map stayed exact.

### Exact 1080 Replay — PASS

- Immutable D362 trace was regenerated as exact `1920x1080`, 250 frames, 20fps,
  12.5s, H.264/yuv420p. Full decode and exact source-index digest passed.
- MP4 SHA-256:
  `2385fc89094acb03a7e8c3aa0b203e73a1e9a110dc9eb63c1116601c51a951e4`.
- Manual all-frame playback saw upright row 0, contact/tilt transition, and toppled/moved
  row 499. Every frame says physics was not recomputed.
- D362's original 1920x1088 failure file remains immutable.

### Actual Isaac Render Synchronization — FAIL

- All 16 actual primary/opposite, before/after PNGs decoded, but the cylinder stayed
  upright at essentially the same location in all four recorded states.
- Primary precommand→final centroid/axis/IoU:
  `0.049105px/0.049759deg/0.996594`; opposite:
  `0.161524px/0.118677deg/0.994711`.
- Primary final-before→after was only `0.011875px/0.003691deg/0.999471`.
  D362 stale-final→D363 final-after was `0.111188px/0.084595deg/0.996066`.
  Both actual final views were `toppled=false`.
- Rerun 0.34.1 validation passed with exact timelines `blueprint/log_time/sync_step`,
  footer/RBL/headless screenshot. It honestly shows actual upright images beside a
  canonical toppled trace panel; Rerun PASS does not repair Hydra FAIL.
- Manual required paths/hashes `22/22` exact; final moved/toppled seen `false`, manual
  `pass=false`.

## D363 Root-cause Boundary

- `all_write_readback_bits_exact=true` was an IsaacLab AssetData cache self-read, not an
  independent PhysX backend getter. The writer updates its cache before the backend setter,
  and D363 did not advance the data timestamp that would trigger a backend re-read.
- The explicit `forward()` call count is exact, but D363 did not attest run-level
  `cfg.use_fabric`, `_fabric_iface`, selected `force_update/update`, independent
  `root_physx_view.get_transforms()`, or USDRT/Fabric world matrices.
- Therefore the break is localized only to somewhere in
  `AssetData cache → PhysX backend → Fabric/USDRT → Hydra`; the exact arrow is unknown.
- Worker log contains generic `Failed to clone in Fabric`, but the same line appears in
  D352/D353/D354/D357/D360/D362. It is not proven as the D363 root cause. CPU powersave,
  PCIe Gen1, VRAM, Warp, or SM efficiency are also not demonstrated causes.
- Telemetry: GPU used max/free min `7760/8185MiB`, utilization max `42%`, worker RSS max
  `7,166,943,232B`. Utilization is not Warp occupancy or SM-efficiency evidence.

## Active Case / Next Approval Boundary

- There is no active approved case.
- Narrowest candidate: D364 `[paused_render_state_layer_localization]`, observability-only.
  With frozen D362 states and zero physics steps, read before/after at:
  1. AssetData cache,
  2. independent PhysX getter,
  3. USD plus USDRT/Fabric world matrix,
  4. Hydra yellow-mask centroid/axis.
- D364 must also attest Fabric enable/interface and the actual `force_update/update` callable.
  It is not approved and must not be implemented or run without explicit user authorization.
- Any q5/physics science, cap/rim discriminator, target/IK/path repair, asset/physics setting
  change, grasp/settle/hold/lift, ten-trial, G0b, RL/PPO/VLA needs separate approval.

## Frozen Boundaries / Operational Residue

- Freeze D351-D363 paths. Do not modify the user-owned
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar.
- Do not substitute Rerun Float32 display values or vertex-only Qhull for canonical
  callback/Float64/sensor evidence.
- Historical D342 worker PID `1729639` remains under user-systemd PID `1123`, RSS about
  954MiB and GPU 320MiB. D363 did not grant signal authority; it was not terminated.
- `HANDOFF.md` and `TASKS.md` are stale. No hardware, B200/SSH, dependency install,
  commit, push, or signal was performed.

## Must Read First

1. `AGENTS.md`; this file; DECISIONS D362-D363; ledger tail
2. `claudedocs/session_20260718_grasp_g0a_d363_trace_replay_1080_and_isaac_render_sync_repair.md`
3. D363 completion, sync report, exact-video report, manual inspection, Rerun validation,
   worker log and supervisor summary
4. `claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md`
5. D362 runtime prerequisites, durable prefix/audit, physics trace, worker and supervisor
6. D361 capacity/prefix session; D360 exception/session when tracing the failure lineage
7. D359 provenance; D354 measurement/binding; D350 static Viewer; D348 topology

## Git

- Verified before D363 edits: `HEAD == origin/master ==
  f085463d2e994a633cd1bcefe0c98c0b6c19e18e`, commit
  `D363 observability-only 전 저장`; worktree was clean.
- Current uncommitted changes are D363-only harness/output/state documentation. No commit or
  push is authorized or performed.
