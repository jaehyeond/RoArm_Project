# 2026-05-22 Track A v4 Recovery Runtime FAIL

## Scope

Track A only: P7/Branch B normalized 3cm cube close_26 proxy. Track B/OpenVLA
work is not evidence for this result.

Forbidden work was not performed: no dataset generation, no training, no
hold-lift, no transport/release, no constraints, no SurfaceGripper, no gate
tuning, and no success claim.

## Preflight

The user separately approved the next executable Track A gate: exactly one
close_26-only B200 runtime with
`--target_guarded_micro_close_v4_recovery_diagnostic`, followed immediately by
the v4 posthoc audit.

Local code before sync:

- runtime probe md5 `2326b68cf5fc7098182b574b4f7a1eb1`
- criteria audit md5 `7f3b368460d26acb3da549ace3e4b25f`
- readiness md5 `db3a8a48ba17cea7570d8e9c45d028e7`
- static v4 design md5 `265391a9a421bb7535925a77ef3e5b37`

B200 sync and static checks:

- The four md5s matched exactly on B200 after rsync.
- B200 `py_compile` returned `py_compile_exit:0`.
- B200 readiness printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- B200 process preflight showed no active relevant Isaac/P7/training process.
- The future v4 stdout/audit files were missing before the run.
- The D024 v7 USD path existed:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd`.

## Runtime

Approved command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --target_guarded_micro_close_v4_recovery_diagnostic
```

Runtime exit: `runtime_exit:0`.

Logs:

- stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out`
  md5 `fe6a733727a6eeb288c6c6464c178af1`, 425 lines.
- stderr
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.err`
  md5 `4dc0d3c542e38524807f8fe75a82f841`, 4 lines.
- stderr lines 1-4 contain the known cpufreq/NVML/Fabric messages only.

Important stdout lines:

- line 37: strict diagnostic-only, close_26-only, v4 flag YES; no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim.
- line 39: object physics mode
  `target_guarded_micro_close_v4_recovery_diagnostic`, separate approval YES.
- line 41: v4 recovery enabled, recovery target error `0.002400m`,
  zero-backlog hold NO, recovery holds preserve backlog YES, rollback on safety
  degradation NO, hard safety violation fails candidate YES.
- line 379 step 3: advance still OK; speed `0.000126m/s`, target error
  `0.001095m`, support budget YES, support horizon YES, recovery ready YES.
- line 380 step 4: first recovery hold due backlog room; gripper `1.918deg`,
  command `6.000deg`, backlog `4.082deg`, target error `0.000943m`.
- line 385 step 9: the v3 line-385 mistake is corrected. v4 holds instead of
  advancing while `target_guarded_target_nonworsening=NO`; recovery hold YES,
  hard freeze NO, target error `0.001249m`.
- line 390 step 14: last recovery hold before fixed-gate breach; target error
  `0.002891m`, counter gap `0.001969m`, command `8.000deg`, gripper
  `7.020deg`.
- line 391 step 15: first hard safety freeze. Target error `0.003035m` exceeds
  fixed `0.003m`; counter gap `0.002050m` exceeds fixed `0.002m`; speed
  `0.001148m/s`, one-sided push NO, support horizon YES.
- line 421 step 45: final plateau at gripper `7.977deg`, command `8.000deg`,
  remaining close `18.023deg`, target error `0.003826m`, counter gap
  `0.002496m`, hard safety freeze YES.
- line 422: posthoc summary FAIL, advances 4, holds 41, zero-backlog holds 0,
  backlog-preserved holds 41, safety rollbacks 0, v4 recovery holds 10, hard
  safety freezes 31.
- line 423: aggregate `close_reached=NO`, attach/posewrite zero,
  telemetry-only YES, success_claim NO.

## Audit

Immediate posthoc audit:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out --expected_mechanism target_guarded_micro_close_v4_recovery_diagnostic
```

Audit result:

- audit stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out`
  md5 `47f4ec7b78298fde0a46ac57105a6e6c`, 54 lines.
- audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e`, empty.
- audit exit code: 1.

Important audit lines:

- lines 4-12: v4 metadata matches and other candidates are disabled.
- line 16: `close_reached pass=NO`.
- lines 23-27: positive advances, positive backlog-preserved holds,
  zero zero-backlog holds, zero safety rollbacks, positive v4 recovery holds all
  PASS.
- line 28: `target_guarded_v4_hard_safety_freezes_zero pass=NO value=31`.
- line 38: `target_guarded_v4_recovery_hold_seen_by_step3 pass=NO`. This is an
  over-specific readiness-style criterion for this real runtime; it is not the
  primary physical blocker because line 49 confirms recovery holds exist.
- lines 50-52: hard freezes, fixed target gate, and fixed support budget fail
  from stdout lines 391-421.
- line 54: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Static Attribution

Added:

- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v4_recovery_runtime_static_analysis.py`
- md5 `e381cbbe65ff899c479e3aad3c399d4a`

Verification:

- local `python -m py_compile`: PASS.
- rsynced to B200; B200 md5 matched local.
- B200 `py_compile_exit:0`.
- B200 static run verified runtime/audit md5s above.

Static output:

- 45 close steps, 4 advances, 41 holds, zero zero-backlog holds, 41
  backlog-preserved holds, zero safety rollbacks, 10 v4 recovery holds, 31 hard
  safety freezes.
- first recovery hold: line 380 step 4.
- v3 line-385 target-worsening advance was successfully converted into a v4
  recovery hold.
- first hard freeze: line 391 step 15, caused by fixed target gate plus fixed
  support budget breach, not speed or one-sided push.
- final gripper improved from v3's `7.144deg` to v4's `7.977deg`, but still
  left `18.023deg` to close_26.
- primary attribution:
  `target_support_hard_gate_freeze_after_recovery_hold`.

## Interpretation

v4 is a useful negative, not grasp success and not close_26 success.

It fixed two prior scheduler failures:

- v2-style zero-backlog starvation stayed fixed: zero zero-backlog holds.
- v3-style safety rollback stayed fixed: zero safety rollbacks.

But recovery hold alone did not make the contact primitive compatible with the
fixed target/support contract. By line 391, the primitive had drifted beyond both
fixed gates while speed and one-sided push remained acceptable. Relaxing the
fixed target/support gates would hide the failure rather than solve it.

## Next Step

Do not rerun v4 as the next experiment. Do not run hold-lift, transport/release,
constraints, SurfaceGripper, dataset generation, training, or gate tuning.

Next Track A work should stay local/static/code-first: design a structural
target/support recovery or contact-compatible close mechanism that preserves:

- fixed close_26 target/support gates;
- zero zero-backlog holds;
- zero safety rollbacks;
- no attach/posewrite;
- no hard safety freezes before close_26 can be reached.
