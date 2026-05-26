# 2026-05-26 Track A CUDA Reboot And Codex Sandbox Readiness

## Scope

- User asked why local CUDA was considered unavailable, then rebooted the local
  Ubuntu PC and asked for step-by-step verification plus a next-session prompt.
- No B200 SSH/reconnect/pull was used. No `.ssh` material was copied.
- No Isaac physics runtime, PPO/training, rollout, dataset generation, hold-lift,
  transport/release, constraints, SurfaceGripper, object attach, posewrite, or
  gate tuning was run in this update.
- This is an operational readiness update before the next Track A v7 close_26
  runtime.

## Pre-Reboot Cause Reverified

The earlier local CUDA block was not an IsaacLab logic failure. It was a local
NVIDIA driver/userspace mismatch after an apt driver update:

- Before reboot, `nvidia-smi` failed.
- Before reboot, `/proc/driver/nvidia/version` and `/sys/module/nvidia/version`
  showed loaded kernel module `580.126.09`.
- Before reboot, `libnvidia-ml.so.1` pointed to
  `libnvidia-ml.so.580.159.03`.
- apt/dpkg logs showed the local NVIDIA stack upgraded from `580.126.09` to
  `580.159.03` on 2026-05-21 06:04 KST, while the system had last booted at
  2026-05-20 22:53 KST.
- Kernel journal showed repeated NVRM API mismatch messages: clients using
  `580.159.03` talked to a loaded kernel module `580.126.09`.

Interpretation: the correct fix was a local Ubuntu reboot, not code changes.

## Post-Reboot Verification

The user rebooted the local PC.

Step 1 - boot time:

- `who -b` reported `system boot 2026-05-26 14:08`.
- `uptime -s` reported `2026-05-26 14:08:34`.

Step 2 - NVIDIA kernel/userspace versions:

- `/proc/driver/nvidia/version` now reports NVIDIA open kernel module
  `580.159.03`.
- `/sys/module/nvidia/version` now reports `580.159.03`.
- `readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` reports
  `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03`.

This confirms the previous `580.126.09` vs `580.159.03` mismatch is resolved on
the host OS.

Step 3 - Codex sandbox behavior:

- Default Codex sandbox command `nvidia-smi` still failed with:
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`.
- Default Codex sandbox `ls -l /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm`
  reported all three device nodes missing.
- The same `nvidia-smi` command run with `sandbox_permissions=require_escalated`
  succeeded and reported:
  - `NVIDIA-SMI 580.159.03`
  - `Driver Version: 580.159.03`
  - `CUDA Version: 13.0`
  - GPU `NVIDIA GeForce RTX 4090 Laptop`
  - memory `842MiB / 16376MiB`

Interpretation: local CUDA is fixed on the host. The remaining default failure
is a Codex sandbox device-visibility issue, not a PC/NVIDIA failure.

Step 4 - IsaacLab Python CUDA:

Run outside the default sandbox:

```bash
conda run -n isaaclab python -c "import torch; print('torch_cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Result:

- `torch_cuda_available True`
- `device_count 1`
- `device_name NVIDIA GeForce RTX 4090 Laptop GPU`

Step 5 - IsaacLab import layer:

Run outside the default sandbox:

```bash
conda run -n isaaclab python -c "import torch; import gymnasium; import isaaclab; import roarm_rl; print('imports_ok', True); print('torch_cuda_available', torch.cuda.is_available())"
```

Result:

- `imports_ok True`
- `torch_cuda_available True`

## Track A v7 Readiness Recheck

Local static checks after reboot:

- `python3 -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` passed.
- `python3 sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  exited 0 and printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- Readiness also rechecked:
  - runtime default-off v7 wiring present;
  - criteria audit metadata guard present;
  - archived v6 runtime is rejected as v7;
  - old v7 reference, virtual no-damping, v3 zero-backlog, v4 hard-freeze, and
    v7 no-active-recovery controls are rejected;
  - synthetic v7 pass reference is accepted.

The local backup top USD remains present and verified:

- `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`
- md5 `4497024d25abab11de5c50e144124553`

The earlier prepared RunPod overlay at
`/tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz` is no longer
present after reboot. This is expected because `/tmp` is volatile. If RunPod is
needed again, recreate the overlay from local files; do not assume that tarball
still exists.

## Current Interpretation

- Local host CUDA is now usable.
- Codex default sandbox does not expose GPU device nodes, so GPU/Isaac commands
  that need `/dev/nvidia*` must be run with `sandbox_permissions=require_escalated`.
- This is not a reason to use B200. B200 remains expired/disconnected.
- This is not a v7 physics result. No close steps were run after reboot.
- The next valid Track A action remains exactly one close_26-only v7 active
  recovery runtime on local CUDA, followed immediately by the v7 posthoc audit.

## Next Concrete Step

Run the local v7 runtime outside the default sandbox:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
  --variant v7 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --object_size_m 0.030 0.030 0.030 \
  --close_deg 26.0 \
  --log_every_close_step 1 \
  --target_guarded_micro_close_v7_active_recovery_diagnostic
```

Capture stdout/stderr to a new local log path, then immediately audit stdout:

```bash
python3 sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py \
  --log <runtime_stdout> \
  --expected_mechanism target_guarded_micro_close_v7_active_recovery_diagnostic
```

If audit fails, analyze the first failing runtime/audit lines before any rerun.
If audit passes, do not start dataset/training; the next gate is hold-lift PASS.
