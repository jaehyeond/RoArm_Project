# Session 2026-05-21 - P7 Branch B compliance direction analysis

## Scope

Track A P7/Branch B only. Track B CoRL paper remains separate.

This was an analytical/modeling pass after the approved v7 close_26 runtime
telemetry failure. It did not run training, generate cube datasets, run new
runtime telemetry, run hold-lift, integrate constraints/defaults, attach
SurfaceGripper, transport, release, tune P7 scalar/gates, or continue the old
2cm sweep.

## Verified Starting State

- `git status --short --untracked-files=all` was empty.
- HEAD was `f4404e9` (`v7`).
- The three v7/telemetry scripts are now tracked at HEAD, not dirty/untracked:
  - `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
    md5 `0b4d3f579d3bb56f994983a876198d65`;
  - `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
    md5 `598c7ac68f0844143ac9589c18c2b7e6`;
  - `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
    md5 `dd1e4723b2930fc7795c65cd104e4587`.

## Rechecked Evidence

- v7 D024 conversion stdout
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out:84-89`
  confirms `cube2cm_fixed_counter_jaw_v7_link` and `hand_tcp` merged into
  `link5`, and `base_link` merged into `world`.
- v7 D024 conversion stderr
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err:1-6`
  contains cpufreq/NVML-uninitialized messages only.
- v7 prep
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:23-28`
  shows the key asymmetry: moving jaw strict contact is YES, fixed counter strict
  contact is NO, but fixed counter 1mm slop contact is YES.
- v7 runtime line 38 confirms strict scope: diagnostic-only, close_26-only, no
  training, no constraints, no SurfaceGripper, no transport/release, no gate
  tuning, and no success claim.
- v7 runtime line 39 confirms the 3cm cube and IK OK:
  `ik_err_mm=(0.477,0.316)`, `max_fk_error_m=0.000518`.
- v7 runtime line 67 replanned after settling from the requested low center to a
  settled center at z `+0.015000m`, updating descend TCP to z `+0.030500m`.
- v7 runtime line 68 confirms authored runtime geometry:
  `counter_parent=link5`, moving center `[0,+0.014250,+0.002000]`, counter
  center `[0,-0.019600,+0.002000]`, slop `0.001000m`.
- Close step 2 already had moving strict contact and counter 1mm slop contact,
  but strict counter contact was still NO.
- Close step 3 turned on `one_sided_push=YES` while object speed jumped to about
  `0.061935m/s`.
- By close step 4, counter slop contact was also NO.
- Final close step 45, line 419, reports `target_error_m=0.023422`,
  `moving_contact=YES`, `counter_contact=NO`, `moving_slop_contact=YES`,
  `counter_slop_contact=NO`, `one_sided_push=YES`, and `reached=NO`.
- Aggregate line 420 reports `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- Runtime stderr lines 1-4 contain only cpufreq/NVML-uninitialized/Fabric
  messages; grep for traceback/exception/fatal/segfault/driver-library mismatch
  was empty. Runtime log md5s remained stdout
  `3939f08ea684c34f76669293b96610ba`, stderr
  `a0cb0d2eb0dca684599e693fcd1e7af7`.

## Code Facts

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py` lines 392-417
  spawn the test object as a rigid `CuboidCfg` with rigid-body properties,
  mass, collision props, and rigid material friction/restitution. There is no
  foam/deformable/compliant body model in this telemetry path.
- The same script lines 536-635 computes contact from runtime body transforms,
  logs strict contact and slop contact separately, and declares one-sided push
  only when object drift/speed begins while exactly one jaw is in strict contact.
- The same script lines 667-693 only runs approach, descend, and close_26, then
  reports `success_claim=NO`.
- `roarm_rl/roarm_stack_env.py` lines 192-208 define the default sponge as a
  rigid `CuboidCfg` with rigid material. Lines 174-188 use implicit actuators
  with stiffness `80.0`, damping `4.0`, and effort limit `2.5`.
- `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py` lines
  317-338 make v7 static success depend on moving strict contact plus fixed
  counter slop contact, not fixed counter strict contact.
- `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py` lines
  49-88 define strict vs slop contact, and the local run found 9/168 slop
  contact hits but 0/168 strict two-sided hits.
- `sim_scripts/p7_branch_b_cube2cm_v6_static_runtime_contact_audit.py` was also
  rerun locally. It reproduced the same pattern for prior runtime endpoints:
  authored static designs can show two-sided contact, but v4/v5 runtime endpoints
  remain moving-only even with simple contact-patch margins up to 5mm.
- Existing older analysis independently says the same physical gap:
  `sim_gap_analysis.py:190-210` marks grasp contact dynamics as CRITICAL because
  the real sponge is deformable while Isaac Lab default is a rigid approximation.
  `data_v5_crossvalidation_v2.py:485-487,628-634,852-858` treats about
  18-20deg as realistic sponge-held gripper state due to compliance. 
  `trajectory_manipulation_capability_analysis.py:195-200,223-229,247-256`
  says the current parallel gripper is best matched to deformable/semi-rigid
  objects and that rigid small objects require tighter width sensing/control.

## Interpretation

The current blocker is not TCP-only IK. Approach/descent succeeded and the close
target was kinematically reachable before contact. The blocker is the contact
model at close time.

The strongest causal chain is:

1. v7 static geometry is already a slop/tolerance candidate, not a strict
   two-sided rigid pinch candidate.
2. At runtime, the first close steps briefly enter the intended neighborhood:
   moving contact appears and counter slop contact appears.
3. As soon as the rigid cube receives the asymmetric moving-jaw contact impulse,
   object speed rises and one-sided push begins.
4. The arm/TCP then settles about 23mm away from the target, so the link5 fixed
   counter no longer sits in the static object-frame pose. Final counter gap is
   far outside the 1mm slop band.

Therefore more small rigid-jaw offsets are unlikely to be the highest-value next
work. They can still produce static-prep passes, but the runtime failure is the
absence of a foam/contact-compliance mechanism that can absorb early asymmetric
contact and keep the object inside a two-sided pinch basin.

Correct framing remains: the current Isaac rigid-cube/jaw proxy is not
reproducing real foam grasp. This does not imply the real robot cannot grasp the
cube.

## Next Falsifiable Direction

Do not jump to dataset generation or training. The next technical branch should
be compliance-first and still separated into gates:

1. Static/prep gate: define a compliance abstraction explicitly, such as a
   contact-patch/slop model with bounded virtual compression, or a true
   deformable/soft-body route if Isaac Lab support is viable.
2. Conversion gate: if an asset change is required, keep USD conversion separate
   from physics validation.
3. Runtime close-contact gate: close_26-only first, with no hold-lift. Required
   pass criteria: approach/descent YES, close_reached YES, no one-sided push,
   moving and counter contact under the declared compliance model, attach/posewrite
   zero, no success overclaim.
4. Hold/lift gate: only after close-time contact/dynamics pass.
5. Dataset/training gate: only after canonical close and hold/lift gates pass.

Recommended next action is not another v8 rigid offset by default. The best next
step is to design the smallest diagnostic compliance proxy and predict what it
must change in the close-step telemetry before any new runtime approval.

## Static Compliance Proxy Audit Follow-Up

Added `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
md5 `bd1f26da1d371e27b559528a6210a941`.

Scope:

- Static/local analysis only.
- No Isaac run, no runtime telemetry, no hold-lift, no training, no dataset
  generation, no defaults edit, no constraint insertion, no SurfaceGripper, no
  transport/release, no scalar/gate tuning, and no success claim.

Inputs:

- Existing B200 v7 close_26 samples:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:376-379,419`.
- Existing runtime gates from
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`:
  `push_drift_gate_m=0.00020`, `push_speed_gate_mps=0.005`, and
  `target_error_gate_m=0.003`.

Local run result:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
  reported:
  - `required_budget_close_steps_2_to_4_m=0.001813`;
  - `required_budget_close_steps_2_to_5_m=0.002911`;
  - `required_budget_final_step_45_m=0.014319`;
  - `close_2_to_4_budget_plausible=YES` under a 5mm diagnostic budget;
  - `final_budget_plausible=NO`;
  - `close_2_to_4_dynamic_ok_without_impulse_absorption=NO`;
  - `contact_label_only_sufficient=NO`.

Interpretation:

- A bounded virtual-compression/contact envelope around 2mm can explain counter
  support through close step 4 as a label/support abstraction.
- That does not explain the physics failure. Step 3 still violates the existing
  push-speed gate by a large margin (`0.061935m/s` vs `0.005m/s`), and step 4 also
  remains a dynamic failure.
- A 15mm envelope would be needed to relabel final step 45 counter support; that
  is outside the declared 5mm diagnostic budget and would be overclaiming.

Next implication:

- The future close_26-only runtime candidate must change close-time dynamics, not
  just contact classification. The predicted necessary telemetry change is:
  counter support remains through step 4 and the step-3 speed impulse no longer
  crosses the push gate.

## Static Dynamics Design Follow-Up

Added `sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
md5 `d43c93d2810dd56468e5d8b885013146`.

Scope:

- Static/local design calculation only.
- No Isaac run, no runtime telemetry, no hold-lift, no training, no dataset
  generation, no defaults edit, no constraint insertion, no SurfaceGripper, no
  transport/release, no scalar/gate tuning, and no success claim.

Code-review notes:

- The script imports the previously encoded v7 B200 close samples from
  `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`, so the
  source of truth for close-step constants remains one local file rather than
  duplicated again.
- Candidate verdicts are intentionally mechanism-level, not success claims:
  `label_only_contact_patch` and `mass_only_inertia` are rejected; the selected
  future mechanism is `soft_contact_material_diagnostic`, but runtime remains
  separately unapproved.

Local run:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
  reported:
  - step 3 speed `0.061935m/s`, allowed residual ratio `0.080730`;
  - step 4 speed `0.043783m/s`, allowed residual ratio `0.114200`;
  - step 5 speed `0.054294m/s`, allowed residual ratio `0.092091`;
  - required speed suppression across steps 3-5 `0.919270`;
  - mass-only worst-case required mass `0.247740kg` from current `0.020kg`;
  - `mass_only_plausible=NO` under a `0.050kg` diagnostic cap;
  - `support_step4_ok=YES`, `support_step5_ok=NO`, `final_support_ok=NO`;
  - `target_step4_ok_if_unchanged=NO` because step 4 target error is `0.003151m`.

Interpretation:

- Mass-only inertia is not a good next proxy. It would need an implausibly heavy
  diagnostic cube relative to the current 20g sim object, while still not proving
  foam-like contact.
- The selected minimal future runtime mechanism is a soft-contact/material
  diagnostic: it is the least artificial candidate that directly targets the
  observed failure surface, namely step-3 impulse/speed and step-4 counter support.
- Reserve virtual compression plus damping only if the softer/contact-parameter
  route cannot produce the required telemetry changes.

Future close_26 pass criteria:

- `approach_ok=YES`;
- `descend_ok=YES`;
- `close_reached=YES`;
- step-3 `object_speed_mps <= 0.005`;
- `one_sided_push=NO` through close steps 2-4;
- counter support still true at close step 4 under the declared compliance model;
- `attach_calls=0`;
- `posewrite_calls=0`;
- `success_claim=NO`.

## Soft-Contact Candidate Wiring Follow-Up

Added default-off runtime candidate support in
`sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
md5 `7a261b72386ee549cb0ce162916597f7`.

Scope:

- Code/static preparation only in this session.
- No Isaac runtime, no hold-lift, no training, no dataset generation, no env
  default edit, no chain default edit, no constraint insertion, no SurfaceGripper,
  no transport/release, no scalar/gate tuning, and no success claim.
- The new runtime candidate is gated behind
  `--soft_contact_material_diagnostic`; it is not approved to run yet.

Code-review notes:

- Baseline physics is preserved when the flag is absent:
  `static_friction=1.5`, `dynamic_friction=1.2`, `restitution=0.0`, solver
  iterations `8/1`, `max_linear_velocity=10.0`, `max_angular_velocity=10.0`, and
  `max_depenetration_velocity=5.0`.
- The candidate changes only object material/contact-response diagnostics:
  higher friction, more solver iterations, lower max linear/angular velocities,
  and lower max depenetration velocity.
- The runtime probe now prints a `future_close26_posthoc_criteria` summary, but
  that summary is explicitly `runtime_gate=NO`; it does not alter early-kill or
  success semantics.

Added posthoc audit script
`sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
md5 `a28c2fa8d8d58c617720f96417707677`.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_v7_reference`
  intentionally returned FAIL. This is expected and confirms the audit can kill
  the known failed v7 case.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_synthetic_pass_reference`
  returned PASS. This confirms the audit is not hardwired to fail; it accepts a
  sample that meets the fixed close_26 criteria.
- The audit now also requires matching metadata:
  `soft_contact_material_diagnostic=YES`,
  `mode=soft_contact_material_diagnostic`, and
  `runtime_candidate_requires_separate_approval=YES`. This prevents a baseline
  or wrong-mode log from passing as a soft-contact result.

The v7 reference fails the fixed criteria at:

- aggregate close reached:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:420`
  has `close_reached=NO`;
- step-3 speed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377`
  has `object_speed_mps=0.061935`, above the `0.005m/s` criterion;
- one-sided push:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:377-378`
  has one-sided push at close steps 3-4;
- step-4 target error:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:378`
  has `target_error_m=0.003151`, above the `0.003m` criterion.

Future close_26-only runtime pass/fail criteria are now fixed before approval:

- `approach_ok=YES`;
- `descend_ok=YES`;
- `close_reached=YES`;
- close step 3 `object_speed_mps <= 0.005`;
- `one_sided_push=NO` for close steps 2-4;
- close step 4 `counter_gap_obj_m` max <= `0.002`;
- close step 4 `target_error_m <= 0.003`;
- `attach_calls=0`;
- `posewrite_calls=0`;
- `telemetry_only=YES`;
- `success_claim=NO`.

Interpretation:

- This is not proof that the material preset will work. It is only a minimal,
  falsifiable runtime candidate scaffold.
- If a future approved run still fails step-3 speed or one-sided push, the
  soft-contact/material explanation is falsified for this proxy and the branch
  should move to a more explicit virtual compression plus damping model rather
  than hold-lift or dataset/training.

## Failure-Mode Register Follow-Up

Added `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`.

Purpose:

- Preserve why the approved v7 close_26 telemetry failed.
- Record which tempting interpretations are now known traps.
- Separate narrow static/code successes from actual physics success.
- State the exact telemetry transition required before future hold-lift,
  dataset/training, constraints, SurfaceGripper, transport/release, or gate
  tuning can be considered.

The register records the v7 chain as a dynamics failure:

- line 38: diagnostic-only, no training/constraints/SurfaceGripper/transport/
  release/gate tuning/success claim;
- line 39: `ik_ok=YES`, `max_fk_error_m=0.000518`, so this is not a TCP-only IK
  blocker;
- line 68: intended link5 counter geometry and 1mm slop were active;
- line 376: close step 2 had only near/slop counter support, not strict
  two-sided contact;
- line 377: close step 3 created the decisive speed failure
  (`object_speed_mps=0.061935`) with `one_sided_push=YES`;
- line 378: close step 4 lost counter slop support and exceeded the 3mm target
  error criterion;
- line 419: final close step stayed moving-only with counter y-gap `0.014319m`;
- line 420: aggregate close failed with zero attach/posewrite and no success
  claim.

Methods recorded as traps:

- conversion/prep success as grasp evidence;
- more small rigid offsets without a new mechanism;
- slop/contact-label-only passes;
- mass-only inertia;
- validators checked only against a known failure.

Current success status:

- No physical/runtime grasp success has been achieved in this Branch B cube path.
- What has succeeded is narrower: v7 asset/prep/conversion as a diagnostic
  platform; static identification of the early support budget; selection of the
  soft-contact/material mechanism; and a posthoc audit that rejects v7 but accepts
  a synthetic sample satisfying the fixed criteria.
- The posthoc audit now also rejects wrong-mode logs, so a future result must be
  both numerically good and actually generated by the soft-contact diagnostic
  candidate.

## Soft-Contact Runtime Readiness Follow-Up

Added `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
md5 `1d022dbbcd57481d1fbf6763663c5041`.

Purpose:

- Check static readiness for a future approved close_26-only runtime.
- Do not launch Isaac.
- Do not execute the runtime probe.
- Do not train, generate datasets, insert constraints, attach SurfaceGripper,
  transport/release, tune gates, or claim success.

Local verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  reported:
  - `runtime_probe_soft_contact_default_off_wiring pass=YES`;
  - `criteria_audit_metadata_guard pass=YES`;
  - `criteria_audit_rejects_v7_reference pass=YES`;
  - `criteria_audit_accepts_synthetic_pass_reference pass=YES`;
  - `future_candidate_command_has_required_flags pass=YES`;
  - `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

The future command was corrected after B200 environment checks. Direct system
Python failed because `isaaclab` was not installed there, and
`./IsaacLab/isaaclab.sh -p` failed because `IsaacLab/_isaac_sim/python.sh` was
missing. The correct B200 execution path is the `isaacsim_5_1` micromamba env.

The command printed by the readiness script after correction is:

```bash
env OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --soft_contact_material_diagnostic
```

The first posthoc command after any future approved run must be:

```bash
python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out
```

Interpretation:

- Static readiness is now complete for the selected soft-contact/material
  candidate.
- This does not approve or execute runtime.
- This does not prove the candidate will work.
- If a future approved runtime fails the posthoc audit, record the exact failed
  criteria before trying another mechanism.

## Approved Soft-Contact Runtime Follow-Up

User approved the close_26-only soft-contact/material runtime.

Execution:

- First attempt with system `python` failed before Isaac runtime:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_python_direct_fail_b200.err`
  md5 `4261bcab144070602917ac4e1ab228e1`, missing `isaaclab`.
- Second attempt with `./IsaacLab/isaaclab.sh -p` failed before Isaac runtime:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_isaaclab_launcher_fail_b200.err`
  md5 `88e033670a9853c9b4c045a1e6d048d1`, missing `_isaac_sim/python.sh`.
- Valid attempt used the B200 `isaacsim_5_1` micromamba env with
  `OMNI_KIT_ACCEPT_EULA=YES` and D024 NVML/Vulkan overrides.

Valid run logs:

- stdout `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`,
  423 lines, md5 `c3c81c1e6d481f23fdbb35411987ea8a`;
- stderr `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err`,
  4 lines, md5 `c0d91f52cb47b553b3d7746ac08995f8`.

Verified stdout:

- line 37: strict diagnostic scope and `soft_contact_material_diagnostic=YES`;
- line 38: 3cm cube, `ik_ok=YES`, `max_fk_error_m=0.000518`;
- line 39: `mode=soft_contact_material_diagnostic`,
  `runtime_candidate_requires_separate_approval=YES`, friction `2.5/2.0`,
  solver iterations `16/4`, max velocities `2.0/5.0`, and max depenetration
  velocity `0.25`;
- line 67: settled pose replan to center z `+0.015000` and descend TCP z
  `+0.030500`;
- line 68: v7 link5 counter geometry and 1mm slop active;
- line 376: step 2 has moving contact and counter slop support, no one-sided
  push yet;
- line 377: step 3 fails speed and one-sided-push criteria:
  `object_speed_mps=0.049059`, `one_sided_push=YES`;
- line 378: step 4 still has one-sided push and target error fail:
  counter gap `0.001989m`, `target_error_m=0.003492`, `one_sided_push=YES`;
- line 419: final close remains moving-only with counter y-gap `0.010935m`;
- line 420: `future_close26_posthoc_pass=NO`;
- line 421: aggregate `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, `success_claim=NO`.

Posthoc audit:

- Updated audit md5 `a28c2fa8d8d58c617720f96417707677`.
- Audit rejects the valid soft-contact runtime log.
- Metadata criteria pass, so the failure is not wrong-mode execution.
- Failed criteria are `close_reached`, step-3 speed, one-sided push through
  steps 2-4, and step-4 target error.

Interpretation:

- Material-only soft contact is falsified as the next mechanism.
- It reduced step-3 speed relative to rigid v7 (`0.049059m/s` vs
  `0.061935m/s`) but only by about 20.8%, far short of the required `<=0.005m/s`
  criterion.
- Step-4 counter support stayed barely within the 2mm budget, but target error
  worsened and one-sided push remained.
- Next work should be static-first explicit virtual compression plus damping,
  not more material-only tuning.

## Virtual Compression Plus Damping Static Design Follow-Up

Added `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
md5 `aab11fb5ecaec645e49f4a9e34d9c185`.

Scope:

- Static/local calculation only.
- No Isaac run, no runtime, no training, no dataset generation, no constraints,
  no SurfaceGripper, no transport/release, no gate tuning, and no success claim.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  reported:
  - step 3 material-only suppression vs rigid v7 `0.207895`;
  - step 4 material-only suppression vs rigid v7 `0.138890`;
  - step 5 material-only suppression vs rigid v7 `0.044959`;
  - worst required extra speed suppression from the soft-contact result
    `0.903574` (`90.4%`);
  - step 4 compression room to 3mm max budget `0.001011m`;
  - step 5 over 2mm support budget `0.001205m`.

Interpretation:

- The next proxy must apply damping before or at close step 3. Waiting until the
  object has already left the counter-support basin is too late.
- Bounded compression remains useful only as support accounting; it must be
  coupled to explicit damping or impulse suppression.
- A future runtime should be rejected if it still shows step-3 speed above gate,
  one-sided push through close steps 2-4, or step-4 target error above gate.

## Virtual Compression Plus Damping Candidate Wiring Follow-Up

This follow-up stayed static/code-first. It did not launch Isaac, run runtime
telemetry, run hold-lift, train, generate datasets, add constraints, attach a
SurfaceGripper, transport/release, tune gates, or claim success.

Re-verified B200 evidence:

- Soft-contact stdout md5 is `c3c81c1e6d481f23fdbb35411987ea8a`; stderr md5 is
  `c0d91f52cb47b553b3d7746ac08995f8`.
- `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out:37`
  confirms strict scope and `soft_contact_material_diagnostic=YES`.
- Line 39 confirms `mode=soft_contact_material_diagnostic`,
  `runtime_candidate_requires_separate_approval=YES`, friction `2.5/2.0`, solver
  iterations `16/4`, max velocities `2.0/5.0`, and max depenetration `0.25`.
- Line 377 fails step-3 speed with `object_speed_mps=0.049059` and
  `one_sided_push=YES`.
- Line 378 has step-4 counter y-gap `0.001989m`, but still
  `target_error_m=0.003492` and `one_sided_push=YES`.
- Line 420 reports `future_close26_posthoc_pass=NO`.
- Line 421 reports `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.

Code changes:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9e5292f176d9b90df30cfd23bdb36028`.
- Added default-off `--virtual_compression_damping_diagnostic`, mutually
  exclusive with `--soft_contact_material_diagnostic`.
- The candidate metadata is explicit:
  `virtual_compression_damping_diagnostic=YES`,
  `mode=virtual_compression_damping_diagnostic`, and
  `runtime_candidate_requires_separate_approval=YES` when enabled.
- The mechanism is bounded by `virtual_compression_budget_m=0.002`,
  `virtual_max_plausible_compression_m=0.003`, and
  `virtual_damping_start_close_step=3`.
- The only runtime intervention designed here is velocity damping via
  `write_root_velocity_to_sim`, with `virtual_velocity_damping_writes` logged.
  It does not call attach, posewrite, constraints, SurfaceGripper, transport,
  release, or env default changes.

Static design:

- `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73`.
- Local run now prints the proposed proxy:
  compression budget `0.002m`, max plausible compression `0.003m`, damping start
  close step `3`, residual velocity ratio `0.08`, and no attach/posewrite,
  constraints, SurfaceGripper, transport/release, or env default edits.
- It still reports material-only suppression vs rigid as:
  step 3 `0.207895`, step 4 `0.138890`, step 5 `0.044959`.
- It projects damped speeds from the soft-contact result as:
  step 3 `0.003925m/s`, step 4 `0.003016m/s`, step 5 `0.004148m/s`.
- This is not a success claim: step 4 target error still fails if unchanged, and
  step 5 is outside the 2mm support budget.

Audit/readiness:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `fba03491e25bdd637c73dc90ca6a0836`.
- The audit now supports
  `--expected_mechanism virtual_compression_damping_diagnostic`, rejects wrong
  metadata, rejects the encoded v7 reference, and accepts a synthetic pass.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `dcec12b0b0063fb34115e3467d435a51`.
- Local readiness printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`, but only for
  command shape and static checks. It did not execute runtime.

Verification run locally:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  reported `CUBE2CM_VIRTUAL_COMPRESSION_DAMPING_STATIC_DONE=YES`.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_v7_reference --expected_mechanism virtual_compression_damping_diagnostic`
  intentionally returned FAIL: wrong metadata, `close_reached=NO`, step-3 speed
  above gate, one-sided push in steps 3-4, and step-4 target error above gate.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_synthetic_pass_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned PASS.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  returned `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

Future close_26-only runtime kill criteria:

- wrong metadata: missing
  `virtual_compression_damping_diagnostic=YES`,
  `mode=virtual_compression_damping_diagnostic`, or
  `runtime_candidate_requires_separate_approval=YES`;
- close step 3 `object_speed_mps > 0.005`;
- `one_sided_push=YES` in close steps 2-4;
- close step 4 max counter gap > `0.002`;
- close step 4 `target_error_m > 0.003`;
- `close_reached=NO`;
- nonzero `attach_calls` or `posewrite_calls`;
- `success_claim=YES`.

## Pre-Compact Checkpoint

This checkpoint was written before user-triggered session compaction.

Current active truth:

- Track A P7/Branch B only. Track B CoRL paper remains separate.
- Material-only soft-contact has been falsified by the approved B200 run.
- The active next mechanism is default-off virtual compression plus damping.
- Runtime is still separately unapproved. Do not run runtime, hold-lift,
  training, dataset generation, constraints, SurfaceGripper, transport/release,
  gate tuning, or old handoff diagnostics.
- Correct framing remains: the current Isaac rigid-cube/jaw plus material-only
  proxy is not reproducing real foam grasp. Do not say the real robot cannot
  grasp the cube.

Files/md5s after this session's code/static updates:

- `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9e5292f176d9b90df30cfd23bdb36028`;
- `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73`;
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `fba03491e25bdd637c73dc90ca6a0836`;
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `dcec12b0b0063fb34115e3467d435a51`.

Commands verified in this session:

- `git diff --check` passed.
- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  reported projected damped speeds `0.003925/0.003016/0.004148m/s` for steps
  3/4/5 from the soft-contact result, with step-4 target error and one-sided
  push still retained as future runtime falsifiers.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_v7_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned FAIL as intended.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_synthetic_pass_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned PASS.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for command shape only.

Current dirty/untracked set is expected and must not be reverted:

- `M START_HERE.md`
- `M claudedocs/DECISIONS.md`
- `M claudedocs/EXPERIMENT_LEDGER.md`
- `M sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
- `?? claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`
- `?? claudedocs/session_20260521_p7_branch_b_compliance_direction_analysis.md`
- `?? sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
- `?? sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
- `?? sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
- `?? sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
- `?? sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`

## Post-Compact Direction Follow-Up: Virtual Damping Audit Hardening

After compaction, the next useful action was not another runtime. The better
static/code step was to close a falsifiability gap in the virtual
compression+damping audit:

- runtime probe already logs `virtual_support`, `virtual_damping_active`,
  per-step `virtual_velocity_damping_writes_total`, and aggregate
  `virtual_velocity_damping_writes`;
- before this follow-up, the audit could reject wrong metadata and bad numeric
  outcomes, but did not yet require the virtual damping path to have actually
  activated.

Code/static updates:

- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `065110aa514e49c62747fe4ab6ceecf4` now parses the virtual support and
  damping fields.
- For `--expected_mechanism virtual_compression_damping_diagnostic`, the audit
  now requires:
  - positive aggregate `virtual_velocity_damping_writes`;
  - close step 3 `virtual_support=YES`;
  - close step 3 `virtual_damping_active=YES`;
  - close step 3 `virtual_velocity_damping_writes_total>=1`.
- Added a synthetic negative control,
  `--use_synthetic_virtual_no_damping_reference`, which carries correct virtual
  metadata and passing numeric gates but zero damping writes. It must FAIL.
- `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `04934025ecf5a4793002c2d9fed20b36` now checks for the new audit criteria
  and verifies the no-damping synthetic rejection.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  still reports projected damped speeds `0.003925/0.003016/0.004148m/s` for
  close steps 3/4/5 and keeps step-4 target error plus one-sided push as future
  runtime falsifiers.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_v7_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned FAIL as intended.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_synthetic_virtual_no_damping_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned FAIL as intended.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --use_synthetic_pass_reference --expected_mechanism virtual_compression_damping_diagnostic`
  returned PASS.
- `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  returned `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- `git diff --check` passed.

No Isaac runtime, training, dataset generation, hold-lift, constraints,
SurfaceGripper, transport/release, gate tuning, or success claim was run.

## Approved Virtual Compression Plus Damping Runtime Result

User then approved the next close_26-only runtime. Scope remained Track A
P7/Branch B only:

- no training or cube sim dataset generation;
- no hold-lift;
- no constraints/default integration;
- no SurfaceGripper;
- no transport, transport target, release, or scripted release;
- no P7 scalar/gate tuning;
- no success claim.

Remote code sync:

- B200 `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9e5292f176d9b90df30cfd23bdb36028`;
- B200 `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `065110aa514e49c62747fe4ab6ceecf4`;
- B200 `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `04934025ecf5a4793002c2d9fed20b36`;
- B200 `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73`.

Runtime command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json /NHNHOME/WORKSPACE/0526040060_A/JHPark/opt/micromamba/bin/micromamba run -p /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/envs/isaacsim_5_1 python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path /tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --virtual_compression_damping_diagnostic
```

Logs:

- stdout `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`,
  md5 `7097b2c2eb70ba77d363dcfade601952`;
- stderr `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.err`,
  md5 `35dc65de1f7982e1a7b1115784cff075`.

Verified B200 stdout lines:

- line 37: strict diagnostic-only scope, close_26 only,
  `soft_contact_material_diagnostic=NO`,
  `virtual_compression_damping_diagnostic=YES`, no disallowed mechanisms, and no
  success claim.
- line 39: `mode=virtual_compression_damping_diagnostic` and
  `runtime_candidate_requires_separate_approval=YES`.
- line 40: compression budget `0.002000`, max plausible compression `0.003000`,
  velocity residual ratio `0.080000`, damping start close step `3`,
  `damping_writes_pose=NO`, and `damping_writes_velocity=YES`.
- line 378: close step 3 passes the speed/damping checks:
  `object_speed_mps=0.004955`, `virtual_speed_pre_damping_mps=0.061935`,
  `virtual_support=YES`, `virtual_damping_active=YES`,
  `virtual_velocity_damping_writes_total=1`, and `one_sided_push=NO`.
- line 379: close step 4 still fails target error:
  `target_error_m=0.003130`, while speed is `0.003203`, counter y-gap is
  `0.001794`, `virtual_damping_active=YES`, and `one_sided_push=NO`.
- line 380: close step 5 leaves the bounded support/damping window:
  counter y-gap `0.002738`, `virtual_support=NO`,
  `virtual_damping_active=NO`, speed `0.050912`, and `one_sided_push=YES`.
- lines 421-422: posthoc summary `future_close26_posthoc_pass=NO`,
  `close_reached=NO`, `virtual_velocity_damping_writes=2`, attach/posewrite
  zero, telemetry-only, and `success_claim=NO`.

B200 posthoc audit result:

- command:
  `python sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log /tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out --expected_mechanism virtual_compression_damping_diagnostic`;
- returned FAIL as intended for the captured result;
- passing checks included metadata, separate-approval marker, positive virtual
  damping writes, step-3 speed below gate, step-3 support, step-3 damping active,
  at least one write by step 3, no one-sided push in steps 2-4, and step-4
  counter support;
- failing checks were `close_reached` and `target_step4_within_gate`
  (`0.003130 > 0.003`).

Stderr:

- lines 1-4 contained the known cpufreq/NVML/Fabric messages;
- grep for traceback, exception, fatal, segfault, driver mismatch, missing module,
  and missing python returned no matches.

Interpretation:

- This runtime is not a close_26 pass and not grasp success.
- It is still useful: explicit damping actually activated by close step 3,
  suppressed the step-3 speed below `0.005m/s`, and removed one-sided push in the
  required steps 2-4.
- The next blocker is target-error control plus support/damping horizon. Step 4
  misses the 3mm target gate by `0.000130m`; step 5 exits the 2mm support budget
  and the high-speed one-sided push returns.
- Do not rescue this by gate tuning or jumping to hold-lift, constraints,
  SurfaceGripper, transport/release, or training. Next work should be
  static/code-first failure attribution.

## Virtual Runtime Failure Static Attribution

Added
`sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
md5 `0cccd8d9f3e5aaf7dc27fc3eb034967c`.

Scope:

- local/static only;
- no Isaac runtime;
- no training or dataset generation;
- no hold-lift;
- no constraints, SurfaceGripper, transport/release, gate tuning, or success
  claim.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
  ran successfully.

Key output:

- step 3 from B200 line 378:
  `damping_speed_suppression=0.919997`, speed `0.004955`,
  support YES, damping active YES, one-sided push NO.
- step 4 from B200 line 379:
  `damping_speed_suppression=0.919989`, speed `0.003203`, support YES,
  damping active YES, one-sided push NO, but target excess `0.000130m`
  (`0.130mm`).
- step 5 from B200 line 380:
  support excess `0.000738m` (`0.738mm`) beyond the 2mm support budget, only
  `0.000262m` margin to the 3mm max plausible compression, damping inactive,
  speed `0.050912`, and one-sided push YES.
- final from B200 line 420:
  final target error `0.022778m`, final counter y-gap `0.013828m`, which is
  `0.010828m` beyond the 3mm max plausible compression.

Conclusion:

- Speed damping alone is not the missing mechanism; it already works while active.
- The next code-first design must address both target-error control under the
  fixed 3mm gate and support/damping horizon after step 4.
- Do not rerun the same parameters; do not tune gates.

## B200 Endgame + Target/Support Horizon Static Design

User stated a separate Track B plan for the remaining B200 window:

- Phase 1 backup pipeline test: target confirmation, rsync 1GB speed, Track A
  `/tmp` log preservation plan.
- Phase 2 B200 env setup: `openvla-oft` conda, `flash-attn==2.5.5`, and HARD
  RULE #15 nightly cu128 recovery after dependency install.
- Phase 2 smoke: 1K smoke with `action_dim=6`, image `top`, loss curve, and
  time/step.
- Phase 3 OpenVLA-OFT main finetune: choose 30K-50K from smoke time/step and
  save 5K/10K/15K/20K/30K/50K checkpoints.
- Phase 4 offline eval and final backup: L2, z-score, diversity by checkpoint,
  plus codebase, best checkpoint, train config, and Track A `/tmp` logs.
- Phase 5 pi0 RunPod handoff after B200 release around 2026-05-22 23:59.

Track B remains separate from Track A. The relevant Track A consequence is
preservation pressure: B200 `/tmp` logs and Track A artifacts must be backed up
before heavy Track B training or B200 release.

Added
`claudedocs/b200_endgame_track_a_preservation_track_b_plan_20260521.md` to
capture this boundary and the Track A preservation list.

Observed local backup state:

- `b200_backup_20260521/env.sh`, 2054 bytes;
- `b200_backup_20260521/._speedtest_model.safetensors.MIJ5aq`, rsync-style temp
  file whose observed size changed during inspection.
- Final check showed only `env.sh` remaining. Do not treat the transient temp
  file as a completed speed-test or backup artifact until target path, final
  file, elapsed time, and md5 manifest are recorded.

Added
`sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
md5 `dca5322e654f3b0d415822f0972d383e`.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
  passed.
- `python sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
  ran successfully.

Key output:

- fixed gates stay unchanged:
  target-error gate `0.003000`, support budget `0.002000`, max plausible
  compression `0.003000`, speed gate `0.005000`.
- design target-error limit is `0.002700`; this is a controller design margin,
  not audit gate tuning.
- step 4 still has target excess over the fixed gate of `0.000130m`
  (`0.130mm`) and excess over the design limit of `0.000430m`.
- step 5 has target excess `0.001843m` and support excess `0.000738m`; if
  damping had remained active, projected speed would be `0.004073m/s`, but
  target and support would still need control.
- stronger damping alone is rejected; support-label-only is rejected because
  final counter gap `0.013828m` exceeds the 3mm max plausible compression by
  `0.010828m`.

Proposed next Track A mechanism shape:

- default-off target-guarded micro-close;
- only advance close when target error is below a design limit;
- support-horizon damping may remain active until max plausible compression, but
  future audit still uses fixed 3mm target gate and fixed 2mm step-4 support
  budget;
- no attach, posewrite, constraints, SurfaceGripper, transport/release, env
  default edits, gate tuning, or success claim.

Future falsifiers:

- step-4 target error > `0.003`;
- step-5 support > `0.003`;
- `close_reached=NO`;
- nonzero attach/posewrite.
