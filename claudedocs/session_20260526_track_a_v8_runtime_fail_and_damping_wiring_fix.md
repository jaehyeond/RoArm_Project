# Track A v8 Runtime Fail And Damping Wiring Fix - 2026-05-26

## Scope

This session executed one user-approved local Stage 0 close_26-only v8 runtime
attempt, audited it immediately, stopped on failure, and performed static
failure analysis plus a code-only wiring fix. No B200 SSH/reconnect/pull was
used. No `.ssh` material was copied. No dataset, PPO/training, rollout,
hold-lift, transport/release, constraints, SurfaceGripper, object attach, or
object posewrite path was used.

RunPod was not used for runtime. Current-session tool discovery exposed
`mcp__runpod__`, and `list_pods(computeType=GPU)` returned `[]`, so there was no
active GPU pod to use.

## Inputs And Preflight

- Local backup source is valid by backup manifest:
  `b200_backup_20260522_final/README_BACKUP.md:8-12` records `tmp_p7/` as
  B200 `/tmp/p7_branch_b_*`, verified 494 local files, key v6 runtime md5
  `9a4f8825a88ee3c9d93d83e5b9a28b41`, and key v6 audit md5
  `480a3355864937763eb665e086aadbb0`.
- Runtime used preserved local USD
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`
  md5 `4497024d25abab11de5c50e144124553`.
- Escalated host GPU check was healthy:
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/nvidia_smi.txt`
  md5 `d0123ba06f2d167c5b9c5b8fc956a650`; lines 1-3 show
  2026-05-26 15:57:08, driver `580.159.03`, CUDA `13.0`; lines 9-10 show the
  RTX 4090 Laptop GPU.
- Escalated IsaacLab CUDA check was healthy:
  `isaaclab_cuda_check.txt` md5 `39579d017469bf3fc10334aa427faade`; lines 7-9
  are `True`, `1`, and `NVIDIA GeForce RTX 4090 Laptop GPU`.
- Pre-run static readiness:
  `readiness.out` md5 `6a2a62808451175b65e5d522b695b8b6`; lines 1-2 confirm
  local/static only and no forbidden mechanisms, line 16 prints the future v8
  command with the local backup USD, and line 19 reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

## Approved Runtime

Command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
  --variant v8 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --object_size_m 0.030 0.030 0.030 \
  --close_deg 26.0 \
  --log_every_close_step 1 \
  --target_guarded_micro_close_v8_observed_recovery_diagnostic
```

Preserved outputs:

- Runtime stdout
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out`
  md5 `74095570c2d6a60abdf522c2413735db`.
- Runtime stderr
  `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.err`
  md5 `0d62dc460bd00423202f6f41ede98de3`.
- Runtime stderr lines 1-6 are requests warnings; line 7 logs
  `Failed to clone in Fabric`. The process still produced close-step rows and
  aggregate stdout, so this is recorded as a runtime stderr error/warning, not as
  an infrastructure block.

Runtime stdout evidence:

- Line 4 confirms `variant=v8`, close_26-only, no env/default edits, no training,
  no constraints, no SurfaceGripper, no attached transport, no release, no gate
  tuning, no hidden posewrite, and no success claim.
- Line 6 confirms object physics mode
  `target_guarded_micro_close_v8_observed_recovery_diagnostic` and separate
  runtime approval marker.
- Line 8 confirms v8 observed recovery enabled, projected reserve trigger enabled,
  posthoc observed-response audit, counter-contact requirement, object posewrite
  NO, and robot-joint-target-only recovery writes.
- Line 423 reports `future_close26_posthoc_pass=NO`, virtual damping writes `0`,
  4 close advances, 41 holds, 39 hard safety freezes, and zero v5/v7 recovery
  writes.
- Line 424 aggregate reports `close_reached=NO`, attach/posewrite zero,
  telemetry-only YES, and success claim NO.

## Immediate Audit

Audit command:

```bash
python3 sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py \
  --log claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/runtime.out \
  --expected_mechanism target_guarded_micro_close_v8_observed_recovery_diagnostic
```

Preserved outputs:

- Audit stdout md5 `7cd38eddb1dc9c925b01948cbc5cb416`.
- Audit stderr was empty.

Audit evidence:

- Line 20 is the first failing criterion: `close_reached pass=NO`.
- Line 26 fails `virtual_velocity_damping_writes_positive` with value `0`.
- Lines 35-36 fail because step3 virtual damping was inactive and no damping
  write was seen by step3.
- Lines 45-49 show no v5/v7 recovery present or triggered.
- Line 53 fails `target_guarded_v8_projected_reserve_trigger_seen`.
- Line 60 reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Failure Analysis

Generated static analysis:

- `claudedocs/runtime_logs/20260526_track_a_v8_observed_recovery_close26_local_approved/v8_runtime_failure_static_analysis.out`
  md5 `7e81773b91d39658a3ec5c6eaf878f0c`.

Key lines:

- Line 1 records local static-only analysis and `rerun=NO`.
- Line 4 shows runtime line 384 / close step 7 as the first hard-safety freeze:
  target error still small (`0.000504m`), but object speed was `0.008672m/s`,
  one-sided push YES, virtual damping inactive, hard safety NO, v8 trigger NO,
  and recovery NO.
- Line 6 shows runtime line 392 / close step 15 with target error `0.003045m`,
  target/support margins both negative, one-sided push YES, virtual support NO,
  virtual damping inactive, hard freeze YES, and no v8 trigger/recovery.
- Lines 8-10 summarize the first hard freeze and first target/support breaches.
- Line 11 reports `seen_trigger=NO seen_needed=NO seen_recovery=NO`.
- Line 12 concludes the failure happened before recovery and that pre-fix v8
  lacked virtual damping activation.

Interpretation:

- This was a real Stage 0 physics/audit failure, not an infrastructure block.
- v8 metadata being present did not mean v8 recovery operated. The pre-fix v8
  path entered target-guarded close logic, but virtual damping stayed inactive.
- Because the v8 projected-reserve trigger is also gated by hard-safety OK, the
  system reached hard freeze before the trigger/recovery window appeared.

## Static Code Fix

Post-fail inspection found a wiring mismatch:

- Runtime lines 1174-1184 include v8 in `target_guarded_close_active`.
- Runtime lines 1255-1265 now include v8 in `virtual_damping_active`. This was
  the missing inheritance in the failed run.
- Runtime lines 1426-1434 show the v8 projected-reserve trigger is gated by
  `target_guarded_v4_hard_safety_ok`, which explains why early hard freeze can
  suppress the trigger.

Fixes made:

- Added v8 observed-recovery diagnostic to the virtual damping activation block.
- Added readiness helper `_contains_in_block`.
- Added readiness check `runtime_probe_v8_inherits_virtual_damping_active`.

Post-fix md5s:

- Runtime probe `acae0ca2e85a522dd4ac8fb583cb8fb8`.
- Audit script unchanged `8dbf621c983ec03f46e5d52843781fda`.
- Readiness script `dc2bdaa8d882f12b5cc901a677caccc0`.
- Post-fix readiness output
  `readiness_after_v8_damping_fix.out` md5
  `b652520a81792bf12373ff742cdba6b5`.

Post-fix readiness evidence:

- Line 1 confirms local/static only, no runtime execution.
- Line 5 confirms `runtime_probe_v8_inherits_virtual_damping_active pass=YES`.
- Lines 6-14 confirm negative controls still reject stale/invalid references.
- Line 15 accepts synthetic PASS.
- Line 17 prints the future local-backup-USD command.
- Line 20 reports `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

## Verification

- `python3 -m py_compile` passed for runtime, audit, and readiness scripts.
- `git diff --check` passed.

## Decision

The first approved v8 runtime is FAIL, not grasp success. The only code action
after failure was a static wiring fix and static readiness re-check. No post-fix
v8 runtime has run. The next valid Track A action is either further static review
or exactly one separately approved post-fix close_26-only v8 runtime with
escalated GPU/Isaac execution and immediate audit.

Dataset/training remains blocked until close_26 PASS, then hold-lift PASS, then
small pilot dataset/replay PASS.
