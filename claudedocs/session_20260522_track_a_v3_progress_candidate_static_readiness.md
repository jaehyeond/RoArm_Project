# 2026-05-22 Track A v3 Progress Candidate Static Readiness

## Scope

Track A only. This was code-first/static work. No Isaac runtime was launched. No
dataset generation, training, hold-lift, transport/release, constraints,
SurfaceGripper, gate tuning, or success claim was performed.

The work responds to the verified v2 failure:

- B200 stdout
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out`
  md5 `52fa5cf2cc0cc5dbdc2f55f0d099611f`.
- B200 line 41: v2 used 2deg micro close, 0.75deg command gate, 0.0015m
  support margin, target non-worsening, and zero-backlog holds.
- B200 lines 377-379: the first 2deg command yielded only `0.361deg` actual
  gripper motion, then v2 reset command to actual and discarded `1.639deg` of
  backlog.
- B200 line 410: first support-margin block was only `0.000083m` over the
  0.0015m warning margin while still inside fixed 0.002m support budget and
  0.003m horizon.
- B200 line 421: final gripper actual `6.087deg`, command `6.089deg`, remaining
  close `19.913deg`; support horizon YES, virtual support YES, support margin NO.
- B200 lines 422-423: posthoc FAIL, 17 advances, 28 holds, 28 zero-backlog
  holds, close_reached NO, attach/posewrite zero, telemetry-only, success_claim
  NO.

## Code Changes

Updated runtime probe:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- md5 `9cdfd04876078110186435bb15ba34ab`

New default-off flag:

- `--target_guarded_micro_close_v3_progress_diagnostic`

v3 design:

- preserves fixed close_26 audit gates;
- keeps v2/v1 mutually exclusive with v3;
- keeps damping active under support horizon;
- separates the old 0.0015m support margin as warning telemetry from the hard
  support budget;
- hard support for v3 uses the fixed 0.002m close_26 support budget and 0.003m
  horizon, not the 0.0015m warning margin;
- never zeroes backlog during normal holds;
- preserves backlog on safe holds;
- rolls back command to actual only on safety degradation;
- requires actual progress before permitting repeated ratchet advances;
- caps projected command backlog at 5deg.

Updated audit:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- md5 `8b75c9d7d678419d0d1c96bf61115aed`

Audit now supports expected mechanism:

- `target_guarded_micro_close_v3_progress_diagnostic`

v3 audit requires:

- v3 metadata match and v1/v2/soft/virtual-only metadata disabled;
- existing fixed gates unchanged: close_reached, step3 speed, no one-sided push
  steps 2-4, step4 support, step4 target, step5 support horizon, attach/posewrite
  zero, telemetry-only, no success claim;
- positive target-guarded advances;
- positive backlog-preserved holds;
- zero zero-backlog holds;
- zero safety rollbacks;
- step3 v3 safety OK, support-budget OK, actual progress >= 0.25deg, backlog
  preserved, and backlog room OK.

Updated readiness:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- md5 `b853e0f2198d1d005ac72bc1e83dafcd`

Readiness is static only and prints the future runtime command, but does not run
Isaac.

## Verification

Local/static only:

- `python -m py_compile` for runtime, audit, readiness, and v2 static progress
  analysis: PASS.
- v3 synthetic pass audit: PASS.
- v3 synthetic no-damping audit: FAIL as intended.
- v3 synthetic zero-backlog audit: FAIL as intended.
- old v2 B200 stdout audited as v3: FAIL as intended.
- old v2 synthetic pass audit: PASS.
- old target-guarded v1 synthetic pass audit: PASS.
- v2 progress static analysis still reproduces md5
  `52fa5cf2cc0cc5dbdc2f55f0d099611f` and reports primary
  `zero_backlog_pulse_progress_starvation`.
- readiness: `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- `git diff --check`: PASS.

Readiness future runtime command:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
/NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run \
-p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 \
python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
--variant v7 \
--robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
--object_size_m 0.030 0.030 0.030 \
--close_deg 26.0 \
--log_every_close_step 1 \
--target_guarded_micro_close_v3_progress_diagnostic
```

Expected first posthoc audit after any future approved runtime:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py \
--log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out \
--expected_mechanism target_guarded_micro_close_v3_progress_diagnostic
```

## Conclusion

v3 is not a success claim. It is a default-off close_26 runtime candidate ready
for separate approval.

The intended falsification is now crisp: if v3 still fails, the audit should say
whether progress still starved, safety rollback happened, damping/support failed,
one-sided push returned, or close_reached remained NO. Do not proceed to
hold-lift, dataset generation, training, constraints, SurfaceGripper, transport,
release, or gate tuning before a v3 close_26 audit PASS.
