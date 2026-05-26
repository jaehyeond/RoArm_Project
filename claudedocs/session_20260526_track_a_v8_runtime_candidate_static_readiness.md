# Session 2026-05-26 - Track A v8 Runtime Candidate Static Readiness

## Scope

- Local/static/code-only. No Isaac runtime, no GPU simulation, no hold-lift, no
  dataset, no PPO/training, no constraints, no SurfaceGripper, no attach, and no
  object posewrite.
- Goal: implement the default-off v8 runtime candidate and matching posthoc audit
  checks after the v8 observed-recovery static design.

## Code Artifacts

- Runtime probe:
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py` md5
  `7e6dfc35bbfeacb5d1689f2f175e5120`.
- Posthoc audit:
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py` md5
  `8dbf621c983ec03f46e5d52843781fda`.
- Static readiness:
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py` md5
  `a31ced20b754a4a42058349525d1a435`.
- Local backup USD for the future runtime command:
  `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`
  md5 `4497024d25abab11de5c50e144124553`.

## Implemented Contract

- Added default-off flag
  `--target_guarded_micro_close_v8_observed_recovery_diagnostic`.
- Added `--variant v8`, reusing the preserved v7 collision USD geometry path unless
  an explicit `--robot_usd_path` is supplied.
- v8 triggers recovery from projected reserve depletion:
  target reserve `<= 0.000800m` or support reserve `<= 0.000400m`.
- v8 recovery candidates must model counter contact or counter slop contact; a
  pure counter-gap-delta candidate is not enough.
- v8 posthoc audit checks active recovery observed response by comparing the next
  close row, and rejects active rows whose next row worsens both target error and
  support gap.
- v8 posthoc audit checks TCP follow by comparing current TCP, selected recovery
  TCP, and next-row TCP.

## Verification

- `py_compile` PASS for runtime probe, criteria audit, and readiness script.
- `git diff --check` PASS.
- Readiness output:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_runtime_candidate_readiness.out`
  md5 `6a2a62808451175b65e5d522b695b8b6`.
- Readiness lines 1-2 confirm local/static only and no training/dataset/constraints/
  SurfaceGripper/attach/posewrite/gate tuning.
- Readiness lines 3-4 confirm runtime wiring and audit metadata guard.
- Readiness lines 5-13 confirm negative controls reject archived v6-as-v8, v7
  reference, no damping, v3 zero-backlog, v4 hard-freeze, v7 no-active-recovery,
  v8 worsening response, v8 no TCP follow, and v8 no counter contact.
- Readiness line 14 confirms synthetic v8 PASS is accepted.
- Readiness line 16 prints the future runtime command with the preserved local
  backup USD path, not volatile `/tmp`.
- Readiness line 19 reports `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

## Unchanged v7 Rejection

- Saved v8 audit of the preserved post-reboot v7 runtime:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_rejects_post_reboot_v7_audit.out`
  md5 `cb082918d92a0f95b585ade432c34730`; audit exit code was 1 as expected.
- Lines 5, 14, and 15 reject the log by metadata: the log is v7, not v8.
- Lines 20 and 30 still reject close success and hard-freeze-free success.
- Lines 53, 55, 57, and 58 reject the v7 log under the new v8 contract: no v8
  reserve trigger, observed response worsens, TCP follow is not positive, and
  candidate counter contact/slop-contact was not modeled.
- Line 60 reports `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Next

- This is code/static readiness, not physics validation.
- The next valid Track A action, only after explicit user approval, is exactly one
  local close_26-only v8 runtime with escalated Codex execution, captured to a new
  preserved log directory, followed immediately by the v8 posthoc audit.
- If that audit fails, stop and analyze the first failing runtime/audit lines
  before any rerun. If it passes, the next gate is hold-lift, not dataset/training.
