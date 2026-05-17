# Session 2026-05-17 — P7 Branch B dynamic-anchor mock chain-command contract

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or constraints.
- Did not tune P7 reward/scalars/thresholds/release guidance.
- Did not run structured A training.
- Did not add scripted release variants.
- Added only an isolated pre-chain mock chain-command contract probe.

## Question

The previous mock-TCP interface probe proved coordinate mapping and multi-waypoint
tracking around the target-tracked dynamic anchor. This session tests the next
pre-chain risk: whether a minimal chain-facing command contract can reject unsafe
call order and still execute a valid close/move/hold/release sequence.

## Script

- `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
- md5 `6af24284baef540f190b762e5da164a5`

Design:

- CPU-only unit probe.
- Dynamic anchor body with `disable_gravity=True`, mass `100.0`.
- RoArm sponge attached to anchor by USD `FixedJoint`.
- Mock chain command surface only; no RoArm articulation and no IK.
- Contract commands:
  - `CLOSE`: allowed only when not attached and not released.
  - `MOVE` / `HOLD`: allowed only while attached and not released.
  - `RELEASE`: allowed only while attached, not released, and target reached.
- Negative checks reject move-before-close, release-before-close, double-close,
  early-release, and move-after-release.
- Valid path uses three waypoints with nonzero
  `tcp_to_anchor_offset=(0.015, 0.000, -0.010)`.

Falsifiable gates:

- all negative contract checks must pass
- `close_rel <= 0.005m`
- attached `rel <= 0.005m`
- max final anchor/sponge target error `<= 0.003m`
- release drop `>= 0.050m` after joint removal + wake velocity

## B200 Run

Runtime used the known per-run B200 Isaac fix:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

Logs:

- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.err`

Evidence:

- Lines 40-41: CPU/no chain/no transport/no SurfaceGripper/no P7 training.
- Line 42: negative contract checks all pass:
  `move_before_close_rejected=YES`, `release_before_close_rejected=YES`,
  `double_close_rejected=YES`, `early_release_rejected=YES`,
  `move_after_release_rejected=YES`.
- Line 49: `CLOSE` accepted, `rel=0.000000`, joint exists, nonzero
  `tcp_to_anchor_offset=([0.015, 0.0, -0.010])`, `waypoints=3`.
- Lines 50-58: after-close hold remains attached with `rel=0.000000`.
- Lines 59, 76, 94: `MOVE` commands accepted and all transform checks report
  `transform_error=0.000000`.
- Lines 68, 86, 103: waypoint target stops with anchor/sponge target errors
  `0.001411`, `0.001464`, and `0.001394`.
- Lines 69-75, 87-93, 104-110: waypoint holds remain attached with
  `rel=0.000000` and target errors below `0.003m`.
- Line 111: `RELEASE` accepted only after target-reached state; joint removed.
- Lines 112-128: release/fall after joint removal + wake velocity.
- Line 129: aggregate reports `contract_negative_ok=YES`, `close_rel=0.000000`,
  `max_attached_rel=0.000000`,
  `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`,
  `target_error_threshold=0.003000`, `release_drop=0.338178`.
- Lines 130-131: all gates pass and
  `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.

stderr:

- Lines 1-3 are cpufreq/NVML warnings only; no Python traceback.

## Interpretation

- The isolated dynamic-anchor path now has a minimal chain-facing command
  contract around close/move/hold/release ordering.
- The immediate problem is now sharper: the remaining unknown is whether the real
  RoArm chain can supply TCP/IK/timing signals that satisfy this contract.
- This is not P7 success, not SurfaceGripper evidence, and not RoArm chain-ready.
  It does not validate IK convergence, articulation dynamics, controller latency,
  TCP estimation, real contact, or attach/release timing in the chain.

## Verification

- Local `python -m py_compile` passed.
- B200 synced md5 matched local md5:
  `6af24284baef540f190b762e5da164a5`.
