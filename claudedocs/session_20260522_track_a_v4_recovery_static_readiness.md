# 2026-05-22 Track A v4 Recovery Static Readiness

## Scope

Track A only. No Isaac runtime, no v2/v3 rerun, no hold-lift, no transport or
release, no constraints, no SurfaceGripper, no dataset generation, no training,
no gate tuning, and no success claim.

The goal was to design the next code/static mechanism after the approved v3
close_26 runtime failed posthoc.

## Reverified B200 Evidence

v3 runtime stdout:

- `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out`
- md5 `5f2d1a626edcdccce8086fafd321c9af`

v3 final audit stdout:

- `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out`
- md5 `ca60c09b03a156c85197e34ec7b28bb5`

Key runtime lines rechecked:

- line 37: diagnostic-only, close_26-only, v3 flag YES, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim.
- line 39: mode `target_guarded_micro_close_v3_progress_diagnostic`, separate
  approval marker YES.
- line 41: zero-backlog hold NO, backlog preserve YES, hard support uses fixed
  budget, rollback on safety degradation YES.
- line 385 step 9: v3 advanced while `target_guarded_target_nonworsening=NO`.
  Target error increased from line 384 `0.000703m` to `0.001249m`, growth
  `0.000546m`, above the existing `0.000250m` tolerance.
- line 387 step 11: first v3 safety rollback. `target_error_m=0.002769`, still
  inside fixed `0.003m` target gate; counter gap `0.001909m`, still inside
  fixed `0.002m` support budget; one-sided push NO.
- line 392 step 16: fixed target/support breach. `target_error_m=0.003070`
  and counter gap `0.002074m`, so gate relaxation remains rejected.
- line 421: final plateau at gripper actual `7.144deg`, command `7.147deg`,
  remaining close `18.856deg`.
- line 422: posthoc summary `future_close26_posthoc_pass=NO`, advances 6,
  holds 39, zero-backlog holds 0, backlog-preserved holds 5, safety rollbacks
  34.
- line 423: `close_reached=NO`, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

Audit lines rechecked:

- line 15: `close_reached pass=NO`.
- line 25: `target_guarded_v3_safety_rollbacks_zero pass=NO value=34`.
- lines 42-43: no zero-backlog holds and every nonrollback hold preserves
  backlog PASS.
- line 46: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Static Attribution

Added:

- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_static_design.py`
- md5 `265391a9a421bb7535925a77ef3e5b37`

Static output:

- verifies v3 stdout/audit md5s;
- identifies line 385 step 9 as the first v4 intervention point;
- reclassifies line 387 first rollback as recoverable recovery-hold, not
  rollback;
- treats line 392 target/support breach as hard audit fail if reached;
- records v4 contract: recovery target error `0.002400m`, fixed target gate
  `0.003000m`, fixed support budget `0.002000m`, zero-backlog holds forbidden,
  safety rollbacks forbidden.

## Code Changes

Runtime probe:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- md5 `2326b68cf5fc7098182b574b4f7a1eb1`

Added default-off:

- `--target_guarded_micro_close_v4_recovery_diagnostic`
- `--target_guarded_v4_recovery_target_error_m` default `0.0024`

v4 behavior:

- advances only when hard fixed safety is OK, target error is below recovery
  threshold, target error is non-worsening, progress gate is OK, and backlog
  room is OK;
- if hard fixed safety is still OK but recovery is not ready, it holds the close
  command and preserves backlog;
- if fixed hard safety is violated, it logs a hard safety freeze that the audit
  rejects;
- it does not rollback command to actual and does not zero backlog.

Audit:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- md5 `7f3b368460d26acb3da549ace3e4b25f`

New expected mechanism:

- `target_guarded_micro_close_v4_recovery_diagnostic`

v4 audit requires:

- v4 metadata match and other candidate modes disabled;
- fixed close_26 gates unchanged;
- positive close advances;
- positive backlog-preserved holds;
- zero zero-backlog holds;
- zero safety rollbacks;
- positive v4 recovery holds;
- zero v4 hard safety freezes;
- every close step within fixed target gate and fixed support budget.

Readiness:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- md5 `db3a8a48ba17cea7570d8e9c45d028e7`

Readiness now targets v4 and prints the future close_26-only command plus the
v4 audit command. It remains local/static; it does not run Isaac.

## Verification

Local/static verification run:

- `python -m py_compile` for runtime, audit, readiness, and v4 static design:
  PASS.
- `python sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_static_design.py`:
  PASS.
- v4 synthetic pass audit: PASS.
- v4 synthetic hard-freeze audit: FAIL as intended.
- v4 synthetic no-damping audit: FAIL as intended.
- v3 B200 stdout audited as v4: FAIL as intended.
- v3 synthetic pass audit: PASS.
- v2 synthetic pass audit: PASS.
- readiness: `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- `git diff --check`: PASS.

## Conclusion

v4 is not a success claim and no runtime has been run. It is the next
close_26-only runtime candidate if separately approved.

The structural change is deliberately narrow: v4 addresses v3's specific
scheduler mistake by blocking the line-385 style target-error-worsening advance
and using a backlog-preserving recovery hold. It does not relax fixed target or
support gates, and if the line-392 style fixed-gate breach occurs again, the v4
audit must fail.

## Next Step

If separately approved, run exactly one B200 close_26-only v4 runtime with:

```bash
env OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --target_guarded_micro_close_v4_recovery_diagnostic
```

Then immediately audit:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out --expected_mechanism target_guarded_micro_close_v4_recovery_diagnostic
```

Still forbidden until close_26 audit PASS: hold-lift, transport/release,
constraints, SurfaceGripper, dataset generation, training, and Track B/OpenVLA
work as Track A evidence.
