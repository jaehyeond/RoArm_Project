# P7 Branch B Cube2cm Failure-Mode Register

Date: 2026-05-21

Scope: Track A P7/Branch B normalized 3cm foam-cube grasp proxy work only.
Track B CoRL paper work remains separate.

This register records what failed, why it failed, what not to repeat, and what a
future success must change. It is intentionally grounded in file/log lines, not
memory.

## Hard Boundaries

- Do not use this register as approval to run runtime, hold-lift, dataset
  generation, training, constraints, SurfaceGripper, transport/release, scalar
  tuning, or new runtime gates.
- Do not claim the real robot cannot grasp the cube. The current failure is that
  the Isaac rigid-cube/jaw collision/contact proxy does not reproduce the real
  foam grasp.
- Do not treat USD conversion, static/prep contact, slop-contact labels, or a
  synthetic pass reference as physics success.

## Current v7 Failure Chain

1. The v7 runtime run was diagnostic-only and did not use training, constraints,
   SurfaceGripper, transport/release, gate tuning, or success claims:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:38`.
2. TCP-only reachability is not the blocker. The selected 3cm cube close_26 plan
   had `ik_ok=YES` and `max_fk_error_m=0.000518`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:39`.
3. Runtime geometry was the intended v7 link5-mounted counter/backstop geometry:
   `counter_parent=link5`, moving ref `+0.014250`, counter ref `-0.019600`, and
   `counter_contact_slop_m=0.001000`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:68`.
4. Close step 2 was only a near-support/slop event, not strict two-sided contact:
   `moving_contact=YES`, `counter_contact=NO`, `counter_slop_contact=YES`, and
   `one_sided_push=NO`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:376`.
5. Close step 3 is where the dynamic failure starts: object speed jumps to
   `0.061935m/s`, above the fixed `0.005m/s` push-speed criterion, while
   `one_sided_push=YES`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377`.
6. Close step 4 shows the support basin is already being lost:
   `counter_slop_contact=NO`, `target_error_m=0.003151` exceeds the `0.003m`
   criterion, and `one_sided_push=YES`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:378`.
7. By final close step 45, the candidate is decisively moving-only: final
   `target_error_m=0.023422`, `counter_contact=NO`, `counter_slop_contact=NO`,
   `counter_gap_obj_m` y-gap `0.014319`, and `reached=NO`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419`.
8. Aggregate result confirms the failure without hidden attachment:
   `approach_ok=YES`, `descend_ok=YES`, `close_reached=NO`, `attach_calls=0`,
   `posewrite_calls=0`, `telemetry_only=YES`, `success_claim=NO`:
   `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:420`.

## Failure Modes To Avoid Repeating

| ID | Tried / tempting method | What happened | Why it failed | Do not repeat as |
| --- | --- | --- | --- | --- |
| FM-01 | Treat conversion/prep success as grasp evidence | v7 D024 conversion and prep were valid, but close_26 runtime failed | Asset import and authored geometry do not prove dynamic contact | "USD converted, therefore graspable" |
| FM-02 | Rigid offset/backstop variants | v4/v5/v6/v7 improved static/prep stories but still produced moving-only or one-sided runtime contact | The failure mode is close-time dynamics, not only object-frame placement | More small rigid offsets without a new mechanism |
| FM-03 | Contact-label/slop expansion | 2mm-ish envelope can relabel early support through step 4, but step 3/4 speeds still violate the push criterion | Label changes do not absorb impulse or keep the object in the counter basin | Slop-contact pass as physics pass |
| FM-04 | Mass-only inertia | Static estimate needs about `0.247740kg` worst-case mass from a `0.020kg` object to suppress steps 3-5 below `0.005m/s` | It is implausible and does not model foam contact | Heavier cube as default next branch |
| FM-05 | Runtime posthoc audit with only a fail case | v7 FAIL was correctly rejected, but that alone did not prove the audit could ever pass | A valid judge needs both negative and positive self-checks | A one-sided validator |
| FM-06 | Passing a baseline or wrong-mode log through the soft-contact audit | Numeric criteria alone could be satisfied by a log that was not generated with the intended soft-contact candidate | The hypothesis under test must match the metadata, not only the outcome numbers | Audits that ignore `soft_contact_material_diagnostic=YES` and object physics mode |
| FM-07 | Running Isaac runtime with system `python` | The first approved soft-contact execution attempt produced `ModuleNotFoundError: No module named 'isaaclab'` | B200 runtime needs the Isaac Sim micromamba env, not `/usr/bin/python` | Direct `python sim_scripts/...` for Isaac runtime |
| FM-08 | Running IsaacLab wrapper without the correct Isaac Sim python path | The second attempt with `./IsaacLab/isaaclab.sh -p` failed because `_isaac_sim/python.sh` was missing | This B200 checkout uses the `isaacsim_5_1` micromamba env rather than the wrapper default path | Assuming `isaaclab.sh -p` works without checking env |
| FM-09 | Soft-contact/material preset alone | The approved soft-contact run executed correctly but failed the posthoc criteria | Material/contact parameters reduced some drift/error but did not suppress the step-3 one-sided impulse enough | Treating softer material settings as sufficient foam compliance |
| FM-10 | Compression+damping without metadata/readiness/audit | A future virtual damping run could otherwise look numerically good while testing the wrong mechanism or hiding velocity writes | The mechanism must be named and falsified explicitly, with attach/posewrite separate from velocity damping | Unlabeled runtime, wrong expected mechanism, or success claim from static readiness |
| FM-11 | Virtual metadata without actual damping activation | A future log could carry `virtual_compression_damping_diagnostic=YES` and good numeric gates while never activating the damping write path | The hypothesis is not just a label; damping must be active by close step 3 and must write velocity damping | PASS from virtual metadata with `virtual_damping_active=NO` or zero `virtual_velocity_damping_writes` |
| FM-12 | Speed-only damping success | Approved virtual damping passed step-3 speed and removed one-sided push in steps 2-4, but still failed step-4 target error and final close | Object velocity damping alone does not keep the TCP/close target within gate or maintain support after step 4 | Treating speed gate pass as close_26 grasp pass |
| FM-13 | B200 endgame without Track A preservation | Track B large-model runs can consume the remaining B200 window and leave `/tmp` logs/artifacts unbacked | Track A conclusions depend on B200 `/tmp` stdout/stderr lines, md5s, USD artifacts, and code snapshots | Starting long Track B training before Track A log/code backup |

## What Actually Worked So Far

No physical/runtime grasp success has been achieved in this branch yet.

Useful successes so far are narrower:

- v7 is static/prep-valid and D024 USD-converted, so the asset path is usable as
  a diagnostic platform. This is not contact success.
- The static compliance audit identified the smallest useful early support
  scale: about `0.001813m` through close steps 2-4. This is a design constraint,
  not a runtime pass.
- The dynamics audit selected the right next falsifiable mechanism: a
  soft-contact/material diagnostic must suppress early speed by about `91.9%`
  across steps 3-5 while keeping counter support through step 4.
- The posthoc criteria audit now has negative and positive self-checks:
  `--use_v7_reference` returns FAIL, the virtual no-damping synthetic returns
  FAIL, and `--use_synthetic_pass_reference` returns PASS. This proves the judge
  is neither metadata-only nor hardwired to reject all inputs.
- The audit now also checks metadata. The old material-only candidate required
  `soft_contact_material_diagnostic=YES` and `mode=soft_contact_material_diagnostic`.
  The next virtual candidate must instead identify
  `virtual_compression_damping_diagnostic=YES`,
  `mode=virtual_compression_damping_diagnostic`, and
  `runtime_candidate_requires_separate_approval=YES`; otherwise it fails even if
  the numeric telemetry happens to look good.
- The approved soft-contact/material runtime executed with correct metadata, but
  still failed the fixed telemetry criteria. It is now a useful negative result,
  not a success.
- The approved virtual compression+damping runtime executed with correct metadata
  and actual step-3 damping activation. It is a partial mechanism result, not
  success: step-3 speed passed and steps 2-4 avoided one-sided push, but step-4
  target error and final close still failed.

## Approved Soft-Contact Runtime Result

The approved B200 soft-contact/material close_26 run was executed on 2026-05-21:

- stdout:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`
  md5 `c3c81c1e6d481f23fdbb35411987ea8a`;
- stderr:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err`
  md5 `c0d91f52cb47b553b3d7746ac08995f8`.

Execution pitfalls before the valid run:

- direct system-python attempt failed with `ModuleNotFoundError: No module named
  'isaaclab'`; preserved as
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_python_direct_fail_b200.err`
  md5 `4261bcab144070602917ac4e1ab228e1`;
- `./IsaacLab/isaaclab.sh -p` attempt failed because
  `IsaacLab/_isaac_sim/python.sh` was missing; preserved as
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_isaaclab_launcher_fail_b200.err`
  md5 `88e033670a9853c9b4c045a1e6d048d1`.

The valid command path used the B200 micromamba env:

```bash
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --soft_contact_material_diagnostic
```

Result:

- line 37 confirms the correct scope and `soft_contact_material_diagnostic=YES`;
- line 39 confirms the soft-contact material parameters and
  `runtime_candidate_requires_separate_approval=YES`;
- line 377 fails the step-3 speed criterion:
  `object_speed_mps=0.049059` vs required `<=0.005`, with
  `one_sided_push=YES`;
- line 378 fails one-sided push and target error:
  `target_error_m=0.003492` vs required `<=0.003`, with
  `one_sided_push=YES`;
- line 420 reports `future_close26_posthoc_pass=NO`;
- line 421 reports `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.

Interpretation:

- Soft-contact/material settings alone did not reproduce foam grasp behavior in
  the current Isaac rigid-cube/jaw proxy.
- The step-3 speed improved relative to rigid v7 (`0.049059m/s` vs
  `0.061935m/s`), but this is only about 20.8% suppression and far short of the
  about 91.9% suppression required by the static dynamics audit.
- Step-4 counter gap stayed barely within the `0.002m` support budget
  (`0.001989m`), but target error worsened to `0.003492m`, and one-sided push
  remained active.
- This falsifies the minimal material-only explanation. The next mechanism must
  explicitly model bounded compression plus damping, not just material friction,
  solver iterations, or depenetration caps.

## Virtual Compression Plus Damping Runtime Result

The next candidate was approved for one close_26-only B200 runtime after
default-off static/code preparation:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9e5292f176d9b90df30cfd23bdb36028` adds
  `--virtual_compression_damping_diagnostic`.
- The candidate declares `virtual_compression_damping_diagnostic=YES` and
  `mode=virtual_compression_damping_diagnostic` only when that flag is enabled.
- It uses a bounded compression budget `0.002m`, max plausible compression
  `0.003m`, damping start close step `3`, and velocity residual ratio `0.08`.
- It logs `virtual_velocity_damping_writes` separately. This is not attach and
  not posewrite; future audit must still require `attach_calls=0` and
  `posewrite_calls=0`.
- `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73` projects damped speeds
  `0.003925/0.003016/0.004148m/s` for close steps 3/4/5 from the soft-contact
  result, but this projection does not prove runtime success. Step-4 target
  error and one-sided push remain required falsifiers.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `065110aa514e49c62747fe4ab6ceecf4` now requires the expected mechanism
  metadata via `--expected_mechanism virtual_compression_damping_diagnostic`,
  positive aggregate damping writes, and step-3 `virtual_support=YES`,
  `virtual_damping_active=YES`, plus at least one write by step 3.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `04934025ecf5a4793002c2d9fed20b36` printed
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for command shape only. It did not
  run Isaac.
- Approved runtime stdout
  `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`
  md5 `7097b2c2eb70ba77d363dcfade601952`; stderr md5
  `35dc65de1f7982e1a7b1115784cff075`.
- Runtime line 378 confirms actual step-3 damping:
  `virtual_damping_active=YES`, `virtual_velocity_damping_writes_total=1`,
  speed `0.004955m/s`, and `one_sided_push=NO`.
- Runtime line 379 fails step-4 target error:
  `target_error_m=0.003130 > 0.003`, despite counter y-gap `0.001794m` and
  speed `0.003203m/s`.
- Runtime line 380 loses support after step 4:
  counter y-gap `0.002738m`, `virtual_support=NO`,
  `virtual_damping_active=NO`, speed `0.050912m/s`, and `one_sided_push=YES`.
- Runtime lines 421-422 and the posthoc audit report FAIL, `close_reached=NO`,
  attach/posewrite zero, telemetry-only, and no success claim.
- Static attribution script
  `sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
  md5 `0cccd8d9f3e5aaf7dc27fc3eb034967c` confirms the remaining blockers:
  step-4 target excess `0.130mm`, step-5 support excess `0.738mm`, and final
  counter y-gap `0.013828m`, which is `0.010828m` beyond the 3mm max plausible
  compression.
- Target/support horizon static design script
  `sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
  md5 `dca5322e654f3b0d415822f0972d383e` rejects stronger damping alone and
  support-label-only. The next mechanism shape is target-guarded micro-close plus
  support-horizon damping, default-off and still audited by fixed gates.

## Direction For A Real Success

A future close_26-only runtime is worth approving only if it tests a mechanism
that can change the step-3/step-4 dynamics, not just contact labels.

The minimal material-only candidate has now failed. The next candidate should be
an explicit virtual compression plus damping proxy that can both keep bounded
counter support and suppress the asymmetric close impulse.

Success must show all of these, in the future stdout and posthoc audit:

- `approach_ok=YES`;
- `descend_ok=YES`;
- `close_reached=YES`;
- close step 3 `object_speed_mps <= 0.005`;
- `one_sided_push=NO` for close steps 2-4;
- close step 4 max counter gap <= `0.002`;
- close step 4 `target_error_m <= 0.003`;
- `attach_calls=0`;
- `posewrite_calls=0`;
- `telemetry_only=YES`;
- `success_claim=NO`.

If the soft-contact/material diagnostic still fails step-3 speed or one-sided
push, record that failure and pivot to an explicit virtual compression plus
damping proxy. Do not jump to hold-lift, dataset generation, training,
constraints, SurfaceGripper, transport/release, or gate tuning from a failed
close-time contact result.

## Future Approval Checklist

The checklist below was used for the now-failed material-only soft-contact
candidate. Do not reuse it as approval for another material-only runtime.

For any future approved runtime, the proposed command must be close_26-only and
must name the active mechanism explicitly in stdout. If it still uses the
soft-contact/material path, stdout must contain:

- diagnostic line with `soft_contact_material_diagnostic=YES`;
- object physics line with `mode=soft_contact_material_diagnostic`;
- object physics line with `runtime_candidate_requires_separate_approval=YES`;
- step logs for close steps 2, 3, and 4;
- aggregate line with `telemetry_only=YES` and `success_claim=NO`.

For the next virtual compression+damping candidate, stdout must instead contain:

- diagnostic line with `virtual_compression_damping_diagnostic=YES`;
- object physics/metadata line with
  `mode=virtual_compression_damping_diagnostic`;
- object physics/metadata line with
  `runtime_candidate_requires_separate_approval=YES`;
- virtual proxy line with bounded compression budget and damping start step;
- step logs for close steps 2, 3, and 4;
- close step 3 must show `virtual_support=YES`,
  `virtual_damping_active=YES`, and
  `virtual_velocity_damping_writes_total>=1`;
- aggregate line with `attach_calls=0`, `posewrite_calls=0`,
  positive `virtual_velocity_damping_writes`, `telemetry_only=YES`, and
  `success_claim=NO`.

After the approved virtual runtime failed, do not rerun the same parameters as a
new experiment. Any future virtual-family runtime must first add a static/code
explanation for target-error control and support/damping retention after close
step 4.

After such a run, the first analysis command should be:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log <future_stdout_log> --expected_mechanism virtual_compression_damping_diagnostic
```

If that audit fails, the failure should be recorded here before any new mechanism
is tried.

The static readiness command before asking for runtime approval is:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py
```

On 2026-05-21, this readiness check first reported
`READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for the material-only candidate, and
that candidate was subsequently run and failed. It now reports
`READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for the virtual compression+damping
command shape only. A runtime still needs separate approval.
