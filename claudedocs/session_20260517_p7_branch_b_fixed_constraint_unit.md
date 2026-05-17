# Session 2026-05-17 — P7 Branch B fixed-constraint unit

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper or the fixed constraint.
- Did not tune P7 scalar/threshold/release-guidance.
- Did not run structured A long training.
- Did not add scripted release variants.
- Did not edit `roarm_stack_env.py`, `train_ppo.py`, `chain_skills.py`, or launch
  defaults.

## B200 Runtime Recovery

Initial B200 Isaac runs crashed before probe logic because container userspace
NVIDIA libraries did not match the kernel module:

- `nvidia-smi` failed with `NVML library version: 580.159`.
- `/proc/driver/nvidia/version` reported kernel module `580.95.05`.
- `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` pointed to `580.159.03`, while
  `libnvidia-ml.so.580.95.05` was also present.
- `/etc/vulkan/icd.d/nvidia_icd.json` pointed through `libGLX_nvidia.so.0`
  (`580.159.03`), while `/usr/share/vulkan/icd.d/nvidia_icd.json` directly
  pointed to `libGLX_nvidia.so.580.95.05`.

Non-destructive runtime fix used for all successful B200 runs:

```bash
CONDA_PREFIX=/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 \
OMNI_KIT_ACCEPT_EULA=YES \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
./IsaacLab/isaaclab.sh -p ...
```

Validation:

- With `LD_PRELOAD`, `nvidia-smi` reported `NVIDIA-SMI 580.95.05`,
  `Driver Version: 580.95.05`.
- With both `LD_PRELOAD` and `VK_ICD_FILENAMES`, the previous tiny unit recheck
  reached script output: `/tmp/p7_branch_b_surface_gripper_unit_recheck_vkicd.out`
  line 16 reported driver `580.95.05`; lines 40-41 printed the unit probe header;
  lines 78-84 printed SurfaceGripper metrics rather than crashing.

## SurfaceGripper Axis/Object Diagnostic

Script:

- `sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py`
- md5 `9f2d877115d9d06465dcc7dfb33a5113`

B200 run:

- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out`
- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.err`

Command used the runtime fix above with:

```bash
./IsaacLab/isaaclab.sh -p sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py \
  --close_steps 80 \
  --hold_steps 120 \
  --max_grip_distance 0.120
```

Key stdout evidence:

- lines 40-41: probe header, CPU/no chain/no transport.
- lines 78-79: canonical SurfaceGripper rig and comparison cases verified.
- canonical cuboid:
  - lines 80-93: close never reached `Closed`; state stayed `0.0` or `-1.0`.
  - lines 94-110: hold stayed non-Closed and drifted.
  - line 111: `closed_detect_step=-1`, `closed_frac=0.0000`,
    `gripped_positive_frac=1.0000`, `max_drift=0.11145`.
  - lines 112-113: final `state=+0.0`, `success=NO`.
- RoArm sponge:
  - lines 114-127: close never reached `Closed`.
  - lines 128-144: hold stayed non-Closed and drifted.
  - line 145: `closed_detect_step=-1`, `closed_frac=0.0000`,
    `gripped_positive_frac=1.0000`, `max_drift=0.34692`.
  - lines 146-147: final `state=+0.0`, `success=NO`.
- lines 148-149: `canonical_cuboid=FAIL`, `roarm_sponge=FAIL`,
  `diagnosis=COMMON_SURFACE_GRIPPER_FAIL`, `SURFACE_AXIS_OBJECT_SUCCESS=NO`.

Interpretation:

- This falsifies the narrow hypothesis that the first failure was only RoArm
  sponge geometry/material/scale.
- Positive `gripped_count` again did not mean stable attach.
- SurfaceGripper remains not chain-ready.

## Fixed Constraint Unit

Script:

- `sim_scripts/p7_branch_b_fixed_constraint_unit_probe.py`
- md5 `ff004e3bd4cdf92a6a9b648c3e42986f`

Design:

- CPU-only unit probe.
- Kinematic anchor body + RoArm sponge.
- `close_constraint()` creates a USD `FixedJoint`.
- `release_constraint()` removes the joint prim; the final passing version also
  writes a small downward root velocity to wake the dynamic sponge after removal.
- No transport and no chain integration.

### v1/v2: hold passes, release fails

B200 run:

- `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v2.out`
- `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v2.err`

Key stdout evidence:

- line 49: close creates joint, `rel=0.000000`, `joint_exists=True`.
- lines 50-66: hold is perfectly stable for 120 steps:
  `rel=0.000000`, `drift=0.000000`, `speed_norm=0.000000`.
- line 67: release removed the joint prim (`joint_exists=False`).
- lines 68-84: sponge did not move after release (`z=0.350000`,
  `speed_norm=0.000000`).
- lines 85-87: `hold_ok=YES`, `release_ok=NO`, `FIXED_UNIT_SUCCESS=NO`.

Interpretation:

- Explicit fixed joint can provide stable attached hold.
- Removing the USD joint prim alone does not wake/detach the dynamic body in this
  runtime path.

### v3: close/hold/release passes

B200 run:

- `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.out`
- `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.err`

Key stdout evidence:

- line 49: close creates joint, `rel=0.000000`, `joint_exists=True`.
- lines 50-66: attached hold before any transport is stable for 120 steps:
  `rel=0.000000`, `drift=0.000000`, `speed_norm=0.000000`.
- line 67: release removed the joint prim (`joint_exists=False`).
- lines 68-75: after wake velocity, sponge separates and falls:
  `rel=0.002736 -> 0.326501`, `z=0.347264 -> 0.023499`.
- lines 76-84: object remains settled on the table near `z=0.023501`.
- line 85: `max_hold_rel=0.000000`, `max_hold_drift=0.000000`,
  `release_drop=0.326499`, `max_release_rel=0.326501`.
- lines 86-87: `hold_ok=YES`, `release_ok=YES`, `FIXED_UNIT_SUCCESS=YES`.

stderr:

- `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.err` lines 1-3 are cpufreq/NVML
  warnings only; no Python traceback.

## Verdict

- B200 runtime is usable for these probes when the 580.95.05 NVML preload and
  matching Vulkan ICD override are set.
- Controlled SurfaceGripper axis/object diagnostic FAILS for both canonical cuboid
  and RoArm sponge. It is not a sponge-only failure.
- Explicit fixed-constraint unit PASS: stable attached hold before transport and
  release via joint removal + wake velocity are both demonstrated.
- Do not chain-integrate yet. Next Branch B step should be a controlled
  micro-move/hold/release unit using the same fixed-constraint API, still outside
  the RoArm chain.
