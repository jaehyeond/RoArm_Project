# Session 2026-05-17 — P7 Branch B dynamic-anchor target tracking

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or constraints.
- Did not tune P7 reward/scalars/thresholds/release guidance.
- Did not run structured A training.
- Did not add scripted release variants.
- Added only an isolated pre-chain target-tracking constraint probe.

## Question

The prior dynamic-anchor fixed-joint probe proved attached coupling and release,
but both full and half commands moved about 2x farther than requested. This
session tests whether closed-loop velocity control can hit a requested target
displacement while preserving attached relative stability.

## Script

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_target_probe.py`
- md5 `4706cdd555de659833df6756f95a4cb0`

Design:

- CPU-only unit probe.
- Dynamic anchor body with `disable_gravity=True`, mass `100.0`.
- RoArm sponge attached to anchor by USD `FixedJoint`.
- Anchor is driven by a closed-loop velocity servo:
  `velocity = clamp(target_kp * (target_pos - anchor_pos), max_cmd_speed)`.
- Target is `close_anchor_pos + move_delta`.
- Target is checked after motion and again after post-move hold before release.
- No RoArm chain, no SurfaceGripper, no P7 training.

Falsifiable gates:

- `close_rel <= 0.005m`
- `max_move_rel <= 0.005m`
- `max_post_move_rel <= 0.005m`
- `final_anchor_target_error <= 0.003m`
- `final_sponge_target_error <= 0.003m`
- `release_drop >= 0.050m` after joint removal + wake velocity

## B200 Runs

Runtime used the known per-run B200 Isaac fix:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Full-command smoke

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.err`

Command used `move_dx=0.020`, `move_dz=0.010`, `max_move_steps=160`,
`target_settle_steps=5`, `anchor_mass=100.0`.

Evidence:

- Lines 40-41: CPU/no chain/no transport/no SurfaceGripper/no P7 training.
- Line 48: target delta `([0.020, 0.0, 0.010])`, `target_kp=8.000`,
  `max_cmd_speed=0.080`, `stop_target_error=0.001500`.
- Line 49: joint closes at `rel=0.000000`, target
  `[0.020000, 0.0, 0.3599995]`.
- Lines 59-68: closed-loop motion keeps anchor and sponge positions identical
  with `rel=0.000000`; line 68 reaches `settled_steps=5`.
- Line 69: target stop errors are `0.001409` for both anchor and sponge.
- Lines 70-82: post-move hold remains coupled with `rel=0.000000`.
- Line 83: post-hold final target errors are
  `final_anchor_target_error=0.001426`,
  `final_sponge_target_error=0.001426`.
- Lines 84-101: release works; sponge falls to table after joint removal + wake
  velocity.
- Line 102: aggregate reports `max_move_rel=0.000000`,
  `max_post_move_rel=0.000000`, `move_steps_used=43`,
  `final_anchor_target_error=0.001426`,
  `final_sponge_target_error=0.001426`,
  `target_error_threshold=0.003000`, `release_drop=0.335825`.
- Lines 103-104: all gates pass and
  `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.

### Half-command cross-check

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_halfcmd_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_halfcmd_smoke.err`

Command used `move_dx=0.010`, `move_dz=0.005`, `max_move_steps=160`,
`target_settle_steps=5`, `anchor_mass=100.0`.

Evidence:

- Line 48: target delta `([0.010, 0.0, 0.005])`.
- Line 67: target stop reached in `move_steps_used=29`, with target-stop errors
  `0.001411` for both anchor and sponge.
- Line 81: post-hold final target errors are
  `final_anchor_target_error=0.001429`,
  `final_sponge_target_error=0.001429`.
- Line 100: aggregate reports `max_move_rel=0.000000`,
  `max_post_move_rel=0.000000`, `final_anchor_target_error=0.001429`,
  `final_sponge_target_error=0.001429`,
  `target_error_threshold=0.003000`, `release_drop=0.330823`.
- Lines 101-102: all gates pass and
  `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.

stderr:

- Both runs have only cpufreq/NVML warnings in stderr lines 1-3; no Python
  traceback.

## Interpretation

- The closed-loop target servo fixes the prior open-loop displacement caveat in
  this isolated unit: full and half target commands both finish within the
  `0.003m` target-error gate while keeping `rel=0`.
- This remains isolated Branch B mechanics only. It is not P7 success, not
  SurfaceGripper evidence, and not RoArm chain-ready.
- The next valid step is still pre-chain: define the smallest interface probe
  that maps this target-tracked anchor semantics toward a future TCP/anchor
  controller without inserting it into the RoArm chain.

## Verification

- Local `python -m py_compile` passed.
- B200 synced md5 matched local md5:
  `4706cdd555de659833df6756f95a4cb0`.
