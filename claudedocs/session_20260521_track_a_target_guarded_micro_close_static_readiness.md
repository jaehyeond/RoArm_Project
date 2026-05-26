# Session 2026-05-21 — Track A Target-Guarded Micro-Close Static Readiness

Date: 2026-05-21 KST

Scope: Track A P7/Branch B only. This session did not run OpenVLA/Track B,
training, dataset generation, Isaac runtime, hold-lift, transport/release,
constraints, SurfaceGripper, gate tuning, or success claims.

## Starting Evidence

The active failure remains the approved B200 virtual compression+damping
close_26 diagnostic:

- stdout `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`
  md5 `7097b2c2eb70ba77d363dcfade601952`;
- stderr `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.err`
  md5 `35dc65de1f7982e1a7b1115784cff075`;
- stdout line 37: diagnostic-only, close_26-only,
  `virtual_compression_damping_diagnostic=YES`, no disallowed mechanisms;
- lines 39-40: virtual mode, separate-approval marker, 2mm compression budget,
  3mm max plausible compression, residual `0.08`, velocity damping YES,
  posewrite NO;
- line 378: step 3 partial win, speed `0.004955`, virtual support/damping YES,
  one-sided push NO;
- line 379: step 4 target fail, `target_error_m=0.003130 > 0.003`;
- line 380: step 5 support loss, counter y-gap `0.002738`, virtual support NO,
  damping NO, speed `0.050912`, one-sided push YES;
- lines 421-422: posthoc FAIL, `close_reached=NO`, attach/posewrite zero,
  telemetry-only, success claim NO.

This is not grasp success and not evidence that the real robot cannot grasp the
cube. The correct frame remains Isaac rigid-cube/jaw proxy mismatch/failure.

## User Clarification

The user clarified that the current work is not the OpenVLA/Track B plan. The
active Track A goal is still to build the sim/Isaac Lab grasp primitive that can
later support broad sim/lab dataset generation and learning. Dataset generation
and training remain blocked until the close_26 primitive passes fixed audit.

## Implemented Static/Code Work

Updated `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`:

- new default-off flag:
  `--target_guarded_micro_close_support_horizon_diagnostic`;
- new mechanism mode:
  `target_guarded_micro_close_support_horizon_diagnostic`;
- mutually exclusive with the older soft-contact and virtual compression flags;
- target-guarded close command only advances gripper close when current target
  error is below design limit `0.0027m`;
- micro-close step size default `2deg`;
- support-horizon damping can remain active until max plausible compression
  `0.003m`, while the audit still uses fixed gates;
- logs target-guarded advance/hold counters and support-horizon state;
- still no attach, posewrite, constraints, SurfaceGripper, transport/release,
  env default edits, gate tuning, or success claim.

Updated `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`:

- added expected mechanism
  `target_guarded_micro_close_support_horizon_diagnostic`;
- parses new metadata and per-step target-guard/support-horizon fields;
- requires metadata match, positive virtual damping writes, positive
  target-guarded close advances, step-3 support/damping/write, no one-sided push
  steps 2-4, fixed step-4 2mm support and 3mm target gates, and step-5 support
  horizon within `0.003m`;
- rejects older logs as wrong mechanism rather than relabeling them.

Updated `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`:

- readiness now targets the new mechanism;
- prints the future B200 runtime command as separate-approval-only;
- does not launch Isaac.

New local md5s:

- runtime probe `2c8926a8862d549989cf52b2f77e80e0`;
- criteria audit `34effe957ffb3ea387adfd786d8f65d3`;
- readiness `85d930d0ecd3d29fd5f721b2df69c76d`.

## Verification

Local verification:

- `python -m py_compile` for the three modified scripts: PASS;
- `git diff --check`: PASS;
- synthetic pass reference with expected target-guarded mechanism: PASS;
- synthetic no-damping reference with expected target-guarded mechanism: FAIL as
  intended;
- v7 reference with expected target-guarded mechanism: FAIL as intended;
- existing approved virtual compression+damping B200 log still FAILs as old
  mechanism on `close_reached` and `target_step4_within_gate`;
- the same existing B200 log also FAILs as the new mechanism due metadata,
  target-guard advance, and support-horizon field mismatch.

B200 sync and verification:

- copied only the three Track A scripts to
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`;
- did not use rsync while backup rsync temp was active; used small `scp`;
- B200 md5s matched local:
  - `2c8926a8862d549989cf52b2f77e80e0`
  - `34effe957ffb3ea387adfd786d8f65d3`
  - `85d930d0ecd3d29fd5f721b2df69c76d`
- B200 `python -m py_compile ...` returned `py_compile_exit:0`;
- B200 static readiness returned `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`;
- B200 audit of the existing virtual log with old expected mechanism returned
  FAIL, `audit_virtual_exit:1`;
- B200 audit of the existing virtual log with new expected mechanism returned
  FAIL, `audit_target_guarded_exit:1`.

## Backup State Observed During This Session

Local backup path: `b200_backup_20260521/`.

At this session's check:

- Phase2 `outputs/smolvla_v6_b200` completed at `_backup.log:579`;
- Phase3 `outputs/smolvla_v6_stacking_v3_b200` had started at `_backup.log:580`;
- active temp file:
  `b200_backup_20260521/outputs_smolvla_v6_stacking_v3_b200/checkpoints/005000/pretrained_model/.model.safetensors.8rXR0l`;
- `b200_backup_20260521/` size about `7.7G`;
- `outputs_smolvla_v6_b200` size `6.0G`;
- `outputs_smolvla_v6_stacking_v3_b200` size about `1.7G`;
- `code/` backup still had 0 files at check time.

Do not infer final backup completion from the presence of Phase2 outputs. The
Phase3 rsync temp was still active when Track A static readiness was synced to
B200.

## Approved Runtime Result

After separate user approval, one Track A close_26-only B200 runtime was run
with the target-guarded mechanism.

B200 logs:

- stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out`
  md5 `c9ae7f3af650a87c3f38ba2d8e41d5b1`;
- stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.err`
  md5 `5cec3e9234de5a95e02692492b276d57`.

Direct B200 stdout line checks:

- line 37: strict diagnostic-only scope; no training, constraints,
  SurfaceGripper, transport/release, gate tuning, or success claim;
- line 39: mode
  `target_guarded_micro_close_support_horizon_diagnostic` with
  `runtime_candidate_requires_separate_approval=YES`;
- line 41: target-guarded support-horizon diagnostic enabled; design limit
  `0.002700m`, micro-close step `2.000000deg`, close-command writes YES,
  posewrite/constraints/SurfaceGripper NO;
- line 379 step3: speed `0.000126m/s`, support horizon active, damping active,
  writes total 1, target-guard advances total 3, one-sided push NO;
- line 380 step4: target error `0.000943m`, counter gap max `0.000001m`,
  support horizon active, one-sided push NO;
- line 381 step5: support horizon active, compression gap max `0.000330m`,
  damping active, target-guard advances total 5;
- line 422: fixed early criteria pass, but `future_close26_posthoc_pass=NO`;
- line 423: `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, `success_claim=NO`.

Posthoc audit was run on B200:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out --expected_mechanism target_guarded_micro_close_support_horizon_diagnostic
```

Audit result: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`, `audit_exit:1`.
The single failing criterion was `close_reached`. Metadata, step3 speed/support,
step4 target/counter support, step5 support horizon, target-guard advances,
telemetry-only, zero attach calls, zero posewrite calls, and no success claim all
passed.

Failure onset from B200 stdout:

- line 384 step8: target error rose to `0.003108m`, target guard held the close
  command at `14deg`; support horizon still active, but virtual support was
  already NO;
- line 385 step9: counter gap reached `0.002825m`, still inside the max
  plausible `0.003m` horizon but outside the old support label;
- line 386 step10: counter gap reached `0.003427m`, support horizon/damping
  turned OFF, speed jumped to `0.033058m/s`, and one-sided push became YES;
- lines 387-392: target error and counter gap continued to grow while the
  command remained held at `14deg`.

Interpretation: this is not grasp success. It is an informative Track A proxy
failure. The target-guarded micro-close fixed the early step3-step5 audit
conditions and the prior step4 target gate failure, but the mechanism plateaued
after step7/8 and still did not reach close_26. The next Track A work should be
static/code-first analysis of the guard plateau and post-step8 support-horizon
loss. Do not rerun the same parameters, tune gates, start dataset generation, or
start training from this result.

## Next Step

The readiness-printed runtime command below has already been used once after
separate approval and FAILED posthoc. Keep it here only for provenance:

```bash
env OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --target_guarded_micro_close_support_horizon_diagnostic
```

The posthoc audit command used for the approved runtime:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out --expected_mechanism target_guarded_micro_close_support_horizon_diagnostic
```

No further Track A runtime is approved by this document. Future work should
begin with static failure analysis of the step8 guard plateau and step10 horizon
loss, then require separate approval before any new runtime.
