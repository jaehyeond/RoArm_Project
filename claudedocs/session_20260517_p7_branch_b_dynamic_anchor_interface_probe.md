# Session 2026-05-17 — P7 Branch B dynamic-anchor mock-TCP interface probe

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or constraints.
- Did not tune P7 reward/scalars/thresholds/release guidance.
- Did not run structured A training.
- Did not add scripted release variants.
- Added only an isolated pre-chain mock-TCP interface probe.

## Question

The prior target-tracking unit proved that a dynamic, gravity-disabled anchor can
hit a requested target while carrying the sponge through a fixed joint. This
session tests the next pre-chain risk: can that mechanism be wrapped in a small
TCP-like command interface with waypoint targets and a TCP-to-anchor offset,
without attaching anything to the RoArm chain?

## Script

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_interface_probe.py`
- md5 `eb81372d78828730e63879a996911bbd`

Design:

- CPU-only unit probe.
- Dynamic anchor body with `disable_gravity=True`, mass `100.0`.
- RoArm sponge attached to anchor by USD `FixedJoint`.
- Mock TCP command surface only; no articulation and no IK.
- Interface mapping: `anchor_target = tcp_target + tcp_to_anchor_offset`.
- Three default waypoints:
  `(0.010, 0.000, 0.005)`,
  `(0.020, 0.006, 0.010)`,
  `(0.012, -0.004, 0.012)`.
- Anchor is driven by the same closed-loop velocity servo as the target-tracking
  unit.
- No RoArm chain, no SurfaceGripper, no P7 training.

Falsifiable gates:

- `close_rel <= 0.005m`
- waypoint and hold `rel <= 0.005m`
- per-waypoint final anchor/sponge target error `<= 0.003m`
- release drop `>= 0.050m` after joint removal + wake velocity

## B200 Runs

Runtime used the known per-run B200 Isaac fix:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Default-offset smoke

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.err`

Evidence:

- Lines 40-41: CPU/no chain/no transport/no SurfaceGripper/no P7 training.
- Line 48: close at `rel=0.000000`, `tcp_to_anchor_offset=([0,0,0])`,
  `waypoints=3`.
- Lines 58, 75, 93: three waypoint transforms all report
  `transform_error=0.000000`.
- Lines 67, 85, 102: waypoint target-stop errors are `0.001411`, `0.001464`,
  and `0.001394` for both anchor and sponge.
- Line 128: aggregate reports `max_move_rel=0.000000`,
  `max_hold_rel=0.000000`, `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`,
  `target_error_threshold=0.003000`, `release_drop=0.338178`.
- Lines 129-130: all gates pass and `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.

### Nonzero-offset cross-check

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.err`

Command added `--tcp_to_anchor_offset 0.015,0.000,-0.010`.

Evidence:

- Line 48: nonzero offset is active:
  `tcp_to_anchor_offset=([0.015, 0.0, -0.010])`.
- Lines 58, 75, 93: mock TCP targets differ from anchor targets, while
  `transform_error=0.000000` confirms the mapping.
- Lines 67, 85, 102: waypoint target-stop errors remain `0.001411`,
  `0.001464`, and `0.001394`.
- Line 128: aggregate again reports `max_move_rel=0.000000`,
  `max_hold_rel=0.000000`, `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`,
  `target_error_threshold=0.003000`, `release_drop=0.338178`.
- Lines 129-130: all gates pass and `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.

stderr:

- Both runs have only cpufreq/NVML warnings in stderr lines 1-3; no Python
  traceback.

## Interpretation

- The dynamic-anchor target-tracking mechanism survives a thin mock-TCP command
  wrapper, including multi-waypoint direction changes and nonzero TCP-to-anchor
  offset mapping.
- This is still isolated Branch B mechanics. It is not P7 success, not
  SurfaceGripper evidence, and not RoArm chain-ready.
- This does not validate RoArm kinematics, IK, articulation dynamics, controller
  latency, or chain contact. Those remain the real integration risks.

## Verification

- Local `python -m py_compile` passed.
- B200 synced md5 matched local md5:
  `eb81372d78828730e63879a996911bbd`.
