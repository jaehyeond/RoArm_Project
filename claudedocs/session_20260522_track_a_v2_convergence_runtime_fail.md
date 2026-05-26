# Session 2026-05-22 — Track A Target-Guarded v2 Convergence Runtime FAIL

Date: 2026-05-22 KST

Scope: Track A P7/Branch B only. This session ran one separately approved
close_26-only B200 runtime for the v2 convergence diagnostic. It did not run
OpenVLA/Track B work, training, dataset generation, hold-lift, transport/release,
constraints, SurfaceGripper, gate tuning, or success claims.

## Starting State

The previous approved target-guarded v1 runtime failed posthoc:

- stdout `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_support_horizon_v7_close26_b200.out`
  md5 `c9ae7f3af650a87c3f38ba2d8e41d5b1`;
- stderr md5 `5cec3e9234de5a95e02692492b276d57`;
- audit recheck stdout md5 `263eb869ada006be9bc5c5b4de6924cd`;
- B200 stdout lines 37/39/41 confirmed strict target-guarded diagnostic scope;
- lines 379-381 improved early step3-step5 criteria;
- line 384 step8 target error rose to `0.003108m`, close command held at
  `14deg`, and command backlog was `6.842deg`;
- line 386 step10 had counter gap `0.003427m`, support horizon/damping OFF,
  speed `0.033058m/s`, and one-sided push YES;
- lines 422-423 reported posthoc FAIL and `close_reached=NO`.

Static attribution script
`sim_scripts/p7_branch_b_cube2cm_target_guarded_failure_static_analysis.py`
identified the primary failure as `advance_scheduling_and_hold_backlog` and
recommended zero-backlog holds plus convergence/support/non-worsening gates.

## Code Synced To B200

Local md5s before sync:

- runtime probe `5446716a908d0869c0c308d22af0eb75`;
- criteria audit `baf1cbec4f8a837458e3695a158a129c`;
- readiness `ca34226d94db9ff09231a84fee8ab1bf`;
- static attribution `7114699126c3f24f5ba4523ba0439e7f`.

B200 initially still had v1 hashes. The four files above were copied to
`/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`, and B200
md5s then matched the local values exactly.

## Static Verification

B200 static checks before runtime:

- `python -m py_compile ...` returned `py_compile_exit:0`;
- readiness printed `local_static_only=YES`, `isaac_run=NO`,
  `runtime_probe_executed=NO`;
- readiness checks all passed and printed
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`;
- readiness future command used `--variant v7`, `--close_deg 26.0`, and
  `--target_guarded_micro_close_v2_convergence_diagnostic`;
- static attribution rerun printed the known v1 failure mechanism:
  first hold line 384 step 8, command backlog `6.842deg`, horizon loss line 386,
  and final close remaining `12.053deg`.

## Runtime

Process check before runtime found no active Isaac/P7/training process relevant
to this gate; only `tail -F /tmp/openvla_oft_v6_smoke.out` was present.

The v2 output files did not exist before the run. One runtime was executed:

```bash
env OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --target_guarded_micro_close_v2_convergence_diagnostic
```

Runtime exit: `runtime_exit:0`.

Logs:

- stdout `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out`
  md5 `52fa5cf2cc0cc5dbdc2f55f0d099611f`, 425 lines;
- stderr `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.err`
  md5 `9061693c9914e735b53a19417cdebb9c`, 4 lines.

Stderr lines 1-4 contain the known cpufreq/NVML/Fabric messages only.

## Verified Runtime Lines

- line 37: diagnostic-only, close_26-only,
  `target_guarded_micro_close_v2_convergence_diagnostic=YES`, no forbidden
  mechanisms and no success claim.
- line 39: `mode=target_guarded_micro_close_v2_convergence_diagnostic` and
  `runtime_candidate_requires_separate_approval=YES`.
- line 41: `v2_convergence_enabled=YES`, command gate `0.750000deg`, advance
  support margin `0.001500m`, growth tolerance `0.000250m`, zero-backlog hold
  YES, command convergence/support margin/non-worsening target required.
- line 379 step 3: speed `0.000128m/s`, command backlog `0.000deg`, command
  converged YES, support margin YES, target non-worsening YES, support horizon
  YES, damping active YES, one-sided push NO. However `virtual_support=NO` and
  `virtual_compression_gap_max_m=0.002262`.
- line 380 step 4: target error `0.000944m`, counter y-gap `0.000001m`,
  one-sided push NO, virtual support YES.
- line 381 step 5: support horizon YES, one-sided push NO, virtual support YES.
- line 402 step 26: first target non-worsening block appears with
  `target_guarded_target_nonworsening=NO`, but support margin remains YES.
- line 409 step 33: last observed advance, `target_guarded_close_advances_total=17`.
- line 410 step 34: first support-margin block,
  `target_guarded_support_margin_ok=NO`, counter y-gap `0.001583m`.
- line 421 step 45: final `target_error_m=0.001921`, speed `0.000527m/s`,
  one-sided push NO, support horizon YES, command backlog `0.002deg`, but actual
  gripper only `6.087deg` and command `6.089deg`.
- line 422: `future_close26_posthoc_pass=NO`, `virtual_velocity_damping_writes=43`,
  `target_guarded_close_advances=17`, `target_guarded_close_holds=28`,
  `target_guarded_zero_backlog_holds=28`.
- line 423: `approach_ok=YES`, `descend_ok=YES`, `close_reached=NO`,
  `attach_calls=0`, `posewrite_calls=0`, `telemetry_only=YES`,
  `success_claim=NO`.

## Posthoc Audit

Command:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out --expected_mechanism target_guarded_micro_close_v2_convergence_diagnostic
```

Audit output:

- stdout `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_audit_b200.out`
  md5 `563a9194dfc1cbe611aa38b9bee45dd3`;
- stderr `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_audit_b200.err`
  md5 `d41d8cd98f00b204e9800998ecf8427e`.

Audit result: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Failing criteria:

- audit line 14: `close_reached pass=NO`;
- audit line 24: `virtual_support_step3 pass=NO`.

Passing criteria included metadata, approach, descend, close early-kill NO,
attach/posewrite zero, telemetry-only, no success claim, positive damping writes,
positive target-guarded advances, zero-backlog hold reporting, step3 speed,
step3 damping activation/write, step3 support horizon, step3 command convergence,
step3 support margin, step3 target non-worsening, no one-sided push steps 2-4,
step4 counter support, step4 target gate, step5 support horizon, every hold being
zero-backlog, and finite close metrics.

## Interpretation

This is not close_26 pass and not grasp success.

v2 corrected the v1 failure in a real way: command backlog is no longer allowed
to grow unchecked, one-sided push did not occur, target error stayed below the
3mm fixed gate, and support horizon stayed active through the final logged close
step.

But v2 is too conservative under the current 45-step close horizon. It reaches
only `6.087deg` actual gripper angle, leaving `19.913deg` to the 26deg close
target. The mechanism starves progress rather than losing the object violently.

The first durable next hypothesis is not "relax success gates". It is to design
a bounded progress mechanism that preserves zero-backlog stability and no
one-sided push while allowing enough close advancement, or to add true compliant
counter behavior so support margin is not a label-only blocker.

## Next Step

- Do not run the same v2 runtime again.
- Do not run hold-lift, dataset generation, training, constraints,
  SurfaceGripper, transport/release, or gate tuning from this result.
- Next Track A work should be static/code-first failure analysis of v2 progress
  starvation: progress budget, max close steps/horizon assumptions, support
  margin threshold vs fixed support budget, and whether a true compliant counter
  model is needed before another runtime is justified.
