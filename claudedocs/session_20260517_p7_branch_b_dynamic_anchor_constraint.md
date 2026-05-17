# Session 2026-05-17 — P7 Branch B dynamic-anchor constraint actuation

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or constraints.
- Did not tune P7 reward/scalars/thresholds/release guidance.
- Did not run structured A training.
- Did not add scripted release variants.
- Added only an isolated pre-chain constraint actuation probe.

## Question

The prior fixed-constraint micro-move failed because pose-writing a kinematic
anchor did not pull the sponge. This session tests whether that failure was
specific to kinematic pose-write actuation rather than the fixed joint itself.

## Script

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_probe.py`
- md5 `082f20f84eac10b76b3d678845321243`

Design:

- CPU-only unit probe.
- Dynamic anchor body with `disable_gravity=True`, mass `100.0`.
- RoArm sponge attached to anchor by USD `FixedJoint`.
- Anchor is driven by `write_root_velocity_to_sim`, not by pose-writing a
  kinematic body.
- Gates require:
  - close and hold rel <= `0.005m`
  - move and post-move rel <= `0.005m`
  - both anchor and sponge move at least 75% of the commanded move norm
  - release separates/falls after joint removal + wake velocity
- No RoArm chain, no SurfaceGripper, no P7 training.

## B200 Runs

Runtime used the known per-run B200 Isaac fix:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Full-command smoke

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_smoke.err`

Command used `move_dx=0.020`, `move_dz=0.010`, `move_steps=80`,
`anchor_mass=100.0`.

Evidence:

- Lines 40-41: CPU/no chain/no transport/no SurfaceGripper/no P7 training.
- Line 48: requested `move_delta=([0.020, 0.0, 0.010])`,
  `move_velocity=([0.050, 0.0, 0.025])`, `anchor_mass=100.000`.
- Line 49: joint closes at `rel=0.000000`.
- Lines 50-58: initial hold stays at `rel=0.000000`; drift only reaches
  `0.000019`.
- Lines 59-71: during velocity-driven move, anchor and sponge positions are
  identical at every logged step; line 71 reaches
  `anchor_pos=([0.0399997, 0.0, 0.369969])`,
  `sponge_pos=([0.0399997, 0.0, 0.369969])`, `rel=0.000000`.
- Lines 72-84: post-move hold remains coupled with `rel=0.000000`.
- Lines 85-102: release works; sponge falls from `z=0.369931` to `z=0.023501`.
- Line 103: aggregate reports `max_move_rel=0.000000`,
  `max_post_move_rel=0.000000`, `move_norm=0.022361`,
  `anchor_moved=0.044707`, `sponge_moved=0.044707`,
  `release_drop=0.346430`.
- Lines 104-105: all gates pass and `FIXED_DYNAMIC_ANCHOR_SUCCESS=YES`.

### Half-command cross-check

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_halfcmd_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_halfcmd_smoke.err`

Command used `move_dx=0.010`, `move_dz=0.005`, `move_steps=80`,
`anchor_mass=100.0`.

Evidence:

- Lines 40-41: same CPU/no chain/no transport/no SurfaceGripper/no P7 training
  scope.
- Line 48: requested `move_delta=([0.010, 0.0, 0.005])`,
  `move_velocity=([0.025, 0.0, 0.0125])`.
- Lines 59-71: anchor and sponge again move together with `rel=0.000000`; line
  71 reaches `anchor_pos=([0.01999985, 0.0, 0.35997447])` and matching sponge
  position.
- Lines 72-84: post-move hold remains coupled with `rel=0.000000`.
- Lines 85-102: release works; sponge falls from `z=0.359937` to `z=0.023500`.
- Line 103: aggregate reports `max_move_rel=0.000000`,
  `max_post_move_rel=0.000000`, `move_norm=0.011180`,
  `anchor_moved=0.022349`, `sponge_moved=0.022349`,
  `release_drop=0.336436`.
- Lines 104-105: all gates pass and `FIXED_DYNAMIC_ANCHOR_SUCCESS=YES`.

stderr:

- Both runs have only cpufreq/NVML warnings in stderr lines 1-3; no Python
  traceback.

## Interpretation

- The kinematic pose-write failure was not proof that fixed joints cannot move
  the sponge. A dynamic, gravity-disabled, velocity-driven anchor does move the
  attached sponge while keeping `rel=0`.
- The result is still not chain-ready. Both runs moved about 2x the requested
  displacement (`move_norm=0.022361` -> moved `0.044707`; `move_norm=0.011180`
  -> moved `0.022349`). The coupling is good, but target tracking/calibration is
  not yet controlled.
- This is a Branch B isolated actuation PASS only. It does not solve P7 and does
  not validate RoArm chain transport.

## Next Step

Before any chain integration, add a target-tracking isolated unit around this
dynamic-anchor semantics:

- drive to a requested target displacement using measured physics dt or a
  closed-loop body velocity controller;
- require both low attachment rel and final anchor/sponge displacement error
  below a small threshold;
- then release.
