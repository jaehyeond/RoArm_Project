# 2026-05-26 Track A v7 Active Recovery Static Readiness

## Scope

Local/static/code-first only. No Isaac runtime, no PPO/training, no rollout, no dataset generation, no hold-lift, no transport/release, no constraints, no SurfaceGripper, no B200 SSH/reconnect/pull, and no success claim.

## Evidence Reverified

- Runtime backup md5:
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  = `9a4f8825a88ee3c9d93d83e5b9a28b41`.
- Audit backup md5:
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  = `480a3355864937763eb665e086aadbb0`.
- Runtime lines 43 and 45 confirm strict v6 diagnostic metadata, close_26-only, no attach/posewrite, separate approval marker.
- Runtime lines 393-397 show v6 recovery holds with 12 total recovery writes by line 397 and 0 IK failures, but the target/support margins keep shrinking instead of recovering.
- Runtime line 398 is the first support hard-freeze: target error is still inside the fixed 3mm gate (`0.002914m`), but counter gap is over budget (`0.002075m > 0.002m`).
- Runtime line 399 is the first target+support breach: target error `0.003052m`, counter gap `0.002146m`.
- Runtime lines 427-428 report `future_close26_posthoc_pass=NO`, close advances `4`, holds `41`, zero zero-backlog holds, zero safety rollbacks, `29` hard freezes, `close_reached=NO`, attach/posewrite zero, telemetry-only, no success claim.
- Audit lines 18, 31, 51-53, and 58 confirm FAIL: close not reached, hard freezes nonzero, fixed target/support criteria fail, and overall criteria pass is NO.

## Failure Mechanism

v6 fixed the v5 unsafe-advance decision by blocking projected target/support margin breaches, but its recovery action remained effectively target-only/passive. Once projected/blocking holds began, the writes did not reduce the support gap or restore target margin fast enough. The first decisive failure is support budget loss at runtime line 398; line 399 then crosses both fixed gates.

## Implemented v7 Design

Updated `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`:

- Added default-off `--target_guarded_micro_close_v7_active_recovery_diagnostic`.
- Added finite-difference TCP sweep helper `_v7_active_recovery_decision`.
- The sweep evaluates candidate TCP offsets against current object pose and current jaw geometry.
- Candidate objective maximizes the minimum fixed target/support margin, rejects negative fixed margins, and requires counter-gap reduction before issuing recovery.
- Recovery writes only robot joint targets via IK. It does not attach objects, write object/root pose, add constraints, add SurfaceGripper, tune gates, or claim success.
- v7 active recovery runs only when the v5/v6 preemptive recovery trigger is active; when v7 is enabled it does not fall back to the old v5 target-only recovery path.
- Added per-step/aggregate telemetry for v7 active recovery needed/selected, candidate count, selected margins, counter-gap delta, recovery step, write count, and IK failures.

Updated `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`:

- Added expected mechanism `target_guarded_micro_close_v7_active_recovery_diagnostic`.
- Added v7 metadata, step, and aggregate parsing.
- Added v7 criteria requiring positive active-recovery writes, zero v7 IK failures, observed trigger, actual active recovery, counter-gap reduction, and nonnegative selected target/support margins.
- Added synthetic v7 no-active-recovery negative control.

Updated `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`:

- Converted readiness to v7 active recovery.
- Future command is a local/RunPod-shaped command template, not a B200 SSH/micromamba command.
- It explicitly rejects the archived v6 runtime log when audited as v7.
- It requires old negative controls plus the new v7 no-active-recovery negative control.

## Static Verification

All checks were local/static:

- `python3 -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` -> PASS.
- `git diff --check -- sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` -> PASS.
- v7 synthetic pass audit -> exit 0, `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=YES`.
- v7 synthetic no-active-recovery audit -> expected exit 1, `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- archived v6 runtime log audited as v7 -> expected exit 1, `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- readiness -> exit 0, `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`, and future status `V7_REQUIRES_SEPARATE_RUNTIME_APPROVAL`.

Current md5s after this static patch:

- runtime probe: `1e3610c45b8a75f374b4d762c91dc0ac`
- criteria audit: `dc3d8b146276c421bc8d5139de7cda1e`
- readiness: `85e6c4485f5503658542d2c369f0a02b`

## Next Gate

The next possible Track A action is not PPO, dataset generation, rollout, or hold-lift. It is a separately approved close_26-only v7 active-recovery runtime on local/RunPod, followed immediately by posthoc audit as v7. Dataset/training remains blocked until close_26 PASS, then hold-lift PASS, then small pilot dataset/replay PASS.
