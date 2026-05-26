# 2026-05-26 Track A v7 Active Recovery Runtime FAIL

## Scope

- Followed current-state boot: read `CLAUDE.md`, `START_HERE.md`, DECISIONS D083-D091, latest ledger rows, required session docs, backup README, and the v7 runtime/audit/readiness scripts. After the new FAIL, appended D092 and re-read the updated state files.
- Did not use `HANDOFF.md` or `TASKS.md`.
- Did not use B200 SSH/reconnect/pull, `ssh JHPark`, or `.ssh` material.
- Ran exactly one local post-reboot close_26-only v7 active-recovery IsaacLab runtime with escalated Codex execution, then immediately ran the v7 posthoc audit.
- Did not run PPO/training, rollout, dataset generation, hold-lift, transport/release, constraints, SurfaceGripper, object attach, posewrite, or gate tuning.

## Preflight Evidence

- Dirty/untracked state was present before work and was not reverted.
- Local B200 backup evidence reverified:
  - v6 runtime stdout md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`.
  - v6 audit stdout md5 `480a3355864937763eb665e086aadbb0`.
  - v7 top USD md5 `4497024d25abab11de5c50e144124553`.
- v6 remains FAIL from local backup:
  - runtime line 398 first support hard-freeze: target `0.002914m`, support gap `0.002075m`.
  - runtime line 399 first target+support breach: target `0.003052m`, support gap `0.002146m`.
  - runtime line 428 aggregate: close_reached NO, hard freezes 29, attach/posewrite 0, success_claim NO.
  - audit line 58 `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Escalated `nvidia-smi` passed: Driver `580.159.03`, CUDA `13.0`, RTX 4090 Laptop.
- Escalated `conda run -n isaaclab` torch CUDA check passed:
  - `True`
  - `1`
  - `NVIDIA GeForce RTX 4090 Laptop GPU`
- Readiness recheck passed:
  - `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`
  - v6 archived runtime rejected as v7, synthetic v7 pass accepted, v7 no-active-recovery negative control rejected.

## Runtime And Audit Logs

Log directory:

`claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/`

Artifacts:

- `runtime.out`: md5 `621d00b9d157b4e70178c28f94ca4c7f`, 426 lines.
- `runtime.err`: md5 `20f7c0cb603b1774eb9752bf8e1547af`, 7 lines.
- `audit.out`: md5 `406b96557d94418f16273e517ec4d69b`, 66 lines.
- `audit.err`: md5 `d41d8cd98f00b204e9800998ecf8427e`, empty.

Runtime command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
  --variant v7 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --object_size_m 0.030 0.030 0.030 \
  --close_deg 26.0 \
  --log_every_close_step 1 \
  --target_guarded_micro_close_v7_active_recovery_diagnostic
```

Runtime exit code was 0. This is not success.

Audit command:

```bash
python3 sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py \
  --log claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out \
  --expected_mechanism target_guarded_micro_close_v7_active_recovery_diagnostic
```

Audit exit code was 1.

## Result

FAIL. This is a real post-reboot physics/audit result, not a CUDA infrastructure block.

Key audit lines:

- audit line 19: `close_reached pass=NO`, source runtime line 424.
- audit line 32: hard-freezes-zero FAIL, value `31`, source runtime line 424.
- audit line 54: hard-freeze criterion FAIL, first source runtime line 392.
- audit line 55: fixed target criterion FAIL, first source runtime line 393.
- audit line 56: fixed support criterion FAIL, first source runtime line 392.
- audit lines 60-64: v7 active recovery present, trigger seen, IK OK, counter-gap reduction, and selected margins all PASS.
- audit line 66: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Key runtime lines:

- runtime line 6: mechanism is `target_guarded_micro_close_v7_active_recovery_diagnostic`, separate-approval marker YES.
- runtime line 8: v7 active recovery enabled, finite-difference TCP sweep, current object pose, object posewrite NO, robot joint target writes only, constraints NO, SurfaceGripper NO.
- runtime lines 389-391: v7 active recovery did trigger for steps 12-14:
  - line 389: target `0.002401m`, support gap `0.001688m`, v7 gap delta `-0.000684m`, writes total 1.
  - line 390: target `0.002650m`, support gap `0.001847m`, v7 gap delta `-0.000703m`, writes total 2.
  - line 391: target `0.002827m`, support gap `0.001963m`, v7 gap delta `-0.000716m`, writes total 3.
- runtime line 392: first hard freeze and first fixed-support failure:
  - target `0.002962m` remains inside 3mm gate;
  - support gap `0.002048m > 0.002m`;
  - v7 active recovery no longer triggers because the row is already beyond the fixed support budget.
- runtime line 393: first fixed target breach:
  - target `0.003059m > 0.003m`;
  - support gap `0.002104m > 0.002m`.
- runtime line 423: posthoc summary `future_close26_posthoc_pass=NO`, 4 advances, 41 holds, 31 hard freezes, 3 v7 recovery writes, 0 v7 IK failures.
- runtime line 424: aggregate close_reached NO, attach_calls 0, posewrite_calls 0, telemetry_only YES, success_claim NO.

Runtime stderr:

- Lines 1-6 are conda/requests warnings.
- Line 7 reports `Failed to clone in Fabric`.
- These are not the audit cause; the runtime reached close rows and aggregate.

## Interpretation

v7 active recovery was wired and exercised. The failure is not metadata, CUDA, missing close rows, missing aggregate, attach/posewrite, safety rollback, zero-backlog hold, or v7 IK failure.

The first runtime failure is support budget loss at runtime line 392. The first audit failure is `close_reached` at audit line 19, with the more diagnostic hard-freeze/fixed-gate failures at audit lines 32 and 54-56.

Compared with v6, v7 reduced the number of active recovery writes from 12 to 3 and moved the first support hard-freeze from v6 runtime line 398 / step 17 to v7 runtime line 392 / step 15. It did prove the active-recovery telemetry path and audit criteria, but it did not preserve the fixed support/target gates through close_26.

Do not rerun v7 unchanged. The next Track A step should be static analysis/redesign of the post-v7 failure mechanism before any new runtime approval. Dataset/training remain blocked until close_26 PASS + hold-lift PASS + small pilot dataset/replay PASS.
