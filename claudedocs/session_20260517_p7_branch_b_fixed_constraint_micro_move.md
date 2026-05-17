# Session 2026-05-17 — P7 Branch B fixed-constraint micro-move

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or fixed constraints.
- Did not tune P7 reward/scalars/thresholds/release guidance.
- Did not run structured A training.
- Did not add scripted release variants.
- Added only a pre-chain unit probe.

## Script

- `sim_scripts/p7_branch_b_fixed_constraint_micro_move_probe.py`
- md5 `fd0d11908cac2fff82b0ec1da3934606`

Design:

- CPU-only unit probe.
- Kinematic anchor body + RoArm sponge.
- Creates a USD `FixedJoint` for close.
- Runs an initial static hold.
- Moves the anchor by a small scripted displacement:
  `dx=0.020`, `dy=0.000`, `dz=0.010` over 80 steps.
- Holds after the move, then removes the joint and applies the same downward
  wake velocity used by the previous fixed-unit release test.
- No RoArm chain, no transport controller, no SurfaceGripper, no P7 training.

Falsifiable gate:

- `close_rel <= 0.005m`
- `max_initial_hold_rel <= 0.005m`
- `max_move_rel <= 0.005m`
- `max_post_move_rel <= 0.005m`
- release must separate/fall by at least `0.050m`

## B200 Run

Runtime used the known per-run B200 Isaac fix:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

Logs:

- `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.out`
- `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.err`

Command:

```bash
./IsaacLab/isaaclab.sh -p sim_scripts/p7_branch_b_fixed_constraint_micro_move_probe.py \
  --initial_hold_steps 40 \
  --move_steps 80 \
  --post_move_hold_steps 80 \
  --release_steps 120 \
  --move_dx 0.020 \
  --move_dz 0.010
```

Exit code: `2`.

## Evidence

- Lines 40-41: probe header confirms CPU/no chain/no transport/no
  SurfaceGripper/no P7 training.
- Line 48: reset starts with no joint and `move_delta=([0.020, 0.0, 0.010])`.
- Line 49: close creates the joint and starts at `rel=0.000000`.
- Lines 50-58: initial static hold is perfect for 40 steps:
  `rel=0.000000`, `drift=0.000000`, `speed_norm=0.000000`.
- Lines 59-71: during the micro-move, the anchor moves but sponge remains at the
  original position. `rel` grows from `0.000280` to `0.022361`; sponge speed
  remains `0.000000`.
- Lines 72-84: post-move hold remains separated at `rel=0.022361`, while sponge
  still has zero drift/speed.
- Lines 85-102: release still works after joint removal + wake velocity; sponge
  falls from `z=0.350000` to `z=0.023501`.
- Line 103: aggregate reports `max_move_rel=0.022361`,
  `max_post_move_rel=0.022361`, `release_drop=0.326499`.
- Lines 104-105: final gates are `close_ok=YES`, `initial_hold_ok=YES`,
  `move_ok=NO`, `post_move_ok=NO`, `release_ok=YES`,
  `FIXED_MICRO_MOVE_SUCCESS=NO`.

stderr:

- `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.err` lines 1-3 are
  cpufreq/NVML warnings only; no Python traceback.

## Verdict

- This kills the current fixed-constraint API as an attached transport mechanism.
- The previous static fixed-unit PASS remains useful only as a close/release API
  smoke; it is not evidence that the constraint can move the sponge.
- The failure is clean: a fixed joint exists and static hold is perfect, but
  direct pose-writing the kinematic anchor does not pull the dynamic sponge.
- Do not chain-integrate this fixed-constraint path.
- Next valid Branch B step is not chain integration. It must either redesign the
  authored constraint actuation semantics in another isolated unit test
  (for example a drive/target/body-pose method that actually moves the attached
  object) or stop this fixed-joint path.
