# Session 2026-05-17 — P7 Branch B SurfaceGripper axis/object probe

## Scope

- Continued Track A P7/Branch B only.
- Did not chain-integrate SurfaceGripper.
- Did not tune P7 scalar/threshold/release-guidance.
- Did not run structured A long training.
- Did not add scripted release variants.
- Did not edit RoArm parent/offset SurfaceGripper placement.

## Boot Verification

Read `CLAUDE.md`, then followed the Current-State Protocol with the user-specified
extra reads.

Pre-code local md5s matched the requested baseline:

- `roarm_rl/chain_skills.py` = `c6e610216197994c6b7d2b6625d87560`
- `launch_chain_topdown.sh` = `b34ef3853ac993a1e2adbaddb420adab`
- `roarm_rl/roarm_stack_env.py` = `e2748144034d5a09d6c7a0f6c0da6906`
- `roarm_rl/train_ppo.py` = `795ee48b1bfdd83e8c9735efd01f6920`
- `launch_p6v17_transport_release.sh` = `2acd462042d0997610fca25ff7a41e21`
- `sim_scripts/p7_action_tcp_quat_trace.py` = `e6c9424cfe7ffafdf00fe0625f0553f7`
- `sim_scripts/p7_attach_semantics_env_probe.py` = `4997a3ec058773004441b74419da114f`
- `sim_scripts/p7_attach_quat_constraint_probe.py` = `a2e16f7683856ead1a9a9eef1da8ea69`
- `sim_scripts/p7_rollout_failure_diag.py` = `a9743d74886c454b1c161a1bade3df93`
- `sim_scripts/p7_structured_release_curriculum_probe.py` = `41e6b48bfaa46b82f2add262903a2a5e`
- `sim_scripts/p7_branch_b_surface_gripper_unit_probe.py` = `1d093ebbd39d2c64252545574e74ad34`

Requested B200 logs existed on B200 `/tmp`; key rechecked lines:

- `/tmp/p7v7_structured_release_smoke.out` lines 68-81: structured A mechanism
  active, release `64/64`, no attached tip before release, close/upright release,
  but final `sz=0.2484`, `success_rate=0.2344`, `EARLY_KILL=YES`.
- `/tmp/p7_branch_b_surface_gripper_unit_smoke.out` lines 89-123: canonical asset
  and sponge verified, close never reached `Closed`, `closed_detect_step=-1`,
  `closed_frac=0.0000`, `max_drift=0.37595`, `SURFACE_UNIT_SUCCESS=NO`.
- `/tmp/p7_attach_semantics_identity_keep.out` lines 64-66: identity+keep active,
  tipped attached sponge reset to `sz_mean=1.0000`, velocity kept.
- `/tmp/p7_attach_semantics_preserve_zero.out` lines 64-66: default
  preserve+zero preserved tipped `sz_mean=0.5000`, velocity zeroed.
- `/tmp/p7v4_attach_identity_keep_model19_trace.out` lines 338-355:
  no release/open (`0/256`), final `d_xy=0.1488`, `sz=0.9036`.
- `/tmp/p7v5_identity_keep_release_guidance_model19_trace.out` lines 239-256:
  release/open `256/256`, but release `d_xy=0.1522`, final `sz=0.4126`.
- `/tmp/p7v6_identity_keep_release_guidance_xy08_model19_trace.out` lines 340-355:
  release/open `256/256`, attached tip before open `118/256`, final `sz=0.2840`.

## Pre-Code Answer

Chose Branch B option A: a controlled canonical-rig SurfaceGripper diagnostic.
The falsifiable comparison is canonical cuboid vs RoArm sponge at the same
canonical rig pose. If cuboid cannot reach stable `Closed`, this points to
SurfaceGripper rig/API/axis usage rather than sponge-specific geometry. If cuboid
passes and sponge fails, the sponge geometry/material/scale hypothesis is killed.

The early-kill gate for each object is:

- `closed_detect_step >= 0`
- hold `closed_frac >= 0.95`
- `max_drift <= 0.020m`
- final `state=+1.0`

`gripped_count > 0` is logged only as a weak auxiliary signal and is not accepted
as attach evidence.

## Code Change

Added:

- `sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py`

Properties:

- CPU-only SurfaceGripper unit probe.
- Uses Isaac Lab canonical `Tests/SurfaceGripper/test_gripper.usd`.
- Compares two objects in the same rig:
  - `canonical_cuboid`: Isaac Lab style `1.0m` cuboid at `(0,0,0.5)`.
  - `roarm_sponge`: project RoArm sponge at the same pose.
- No env/reward/train/chain/launcher changes.
- No RoArm parent/offset search.
- No transport.
- No chain integration.

Post-change local and B200 md5:

- `sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py` =
  `9f2d877115d9d06465dcc7dfb33a5113`

Local check:

- `python -m py_compile sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py`
  passed.

## B200 Smoke Attempt

Target logs:

- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out`
- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.err`

Command path used after confirming `isaaclab.sh` needed `CONDA_PREFIX`:

```bash
CONDA_PREFIX=/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 \
OMNI_KIT_ACCEPT_EULA=YES \
./IsaacLab/isaaclab.sh -p \
  sim_scripts/p7_branch_b_surface_gripper_axis_object_probe.py \
  --close_steps 80 \
  --hold_steps 120 \
  --max_grip_distance 0.120
```

Result:

- Process exited `139` before the script reached its diagnostic prints.
- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 1-8 show Isaac
  startup only.
- `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.err` lines 1-2 show
  `NVML_ERROR_LIB_RM_VERSION_MISMATCH`.
- Same stderr lines 3-7 show Isaac crash reporter and `Crash detected`.
- Same stderr line 42 shows the attempted script command line.
- Same stderr lines 64-66 show `nvidia-smi` failed with driver/library mismatch.
- Same stderr lines 81-89 show GLX/NVIDIA backtrace ending in
  `vk_icdNegotiateLoaderICDInterfaceVersion`.
- Same stderr line 90 shows `Segmentation fault`.

To separate script bug from current B200 Isaac runtime, the previous successful
unit probe was re-run with tiny step counts:

```bash
CONDA_PREFIX=/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 \
OMNI_KIT_ACCEPT_EULA=YES \
./IsaacLab/isaaclab.sh -p \
  sim_scripts/p7_branch_b_surface_gripper_unit_probe.py \
  --close_steps 1 \
  --hold_steps 1 \
  --max_grip_distance 0.120
```

It also exited `139` before script diagnostics:

- `/tmp/p7_branch_b_surface_gripper_unit_recheck_tmp.out` lines 1-9 show Isaac
  startup only.
- `/tmp/p7_branch_b_surface_gripper_unit_recheck_tmp.err` lines 64-66 show the
  same driver/library mismatch.
- Same stderr lines 81-90 show the same GLX/NVIDIA crash and segfault.

## Interpretation

No SurfaceGripper axis/object verdict was produced. The B200 smoke is blocked by
an Isaac/driver runtime issue that reproduces on the prior known script before
any probe-specific logic runs.

Therefore:

- Do not count this as `canonical_cuboid` PASS/FAIL.
- Do not count this as `roarm_sponge` PASS/FAIL.
- Do not chain-integrate SurfaceGripper.
- Keep the next Track A action as: re-run the same axis/object probe after B200
  Isaac runtime is healthy, or switch to the fixed/D6 constraint unit only if the
  SurfaceGripper diagnostic remains operationally blocked.

## Docs

- Updated `START_HERE.md` with the new blocked latest state.
- Did not append `claudedocs/EXPERIMENT_LEDGER.md`: the major diagnostic did not
  complete.
- Did not append `claudedocs/DECISIONS.md`: no durable technical lesson changed;
  only the current B200 runtime is blocked.
