# START_HERE.md

Last updated: 2026-05-21 KST (Track A P7/Branch B virtual compression+damping runtime result)

This is the rolling current-state dashboard. It is not full history.
Durable rules live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## Current Truth

The project is two-track:

- **Track A**: existing sim/lab stacking work. Current active line is P7/Branch B
  normalized cube grasp, now focused on compliance-first contact proxy design
  after v4/v5/v6/v7 rigid runtime failures.
- **Track B**: CoRL 2026 paper sprint. Keep separate unless explicitly asked.

Important correction: earlier `START_HERE.md`, ledger row 73, and
`session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md` are stale
relative to later B200 logs. They still say v4 USD conversion was not run and
v4 physics was unvalidated. That is no longer the current evidence.

## Latest Verified B200 Evidence

v4:

- Conversion succeeded. B200
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_convert_b200.out:82` reports
  `cube2cm_counter_jaw_v4_link` merged into `gripper_link`.
- v4 root USD md5:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_collision_usd/roarm_m3.usd`
  `4497024d25abab11de5c50e144124553`.
- Physics/telemetry failed. B200 close-hold-lift
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v4_close26_hold_lift_b200.out:390-391`
  reports `reached=NO` and `verdict=LATCH_FAIL`.
- Runtime jaw telemetry
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v4_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, and `success_claim=NO`.

v5:

- Static prep and conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_urdf_prep_b200.out:25,28`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v5_convert_b200.out:82,84,86`.
- Runtime jaw telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v5_b200.out:422-423`
  reports `moving_contact=YES`, `counter_contact=NO`,
  `one_sided_push=YES`, and `success_claim=NO`.

v6:

- Static prep succeeded for the 3cm candidate:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_urdf_prep_b200.out:1-2,5,23-26`.
- Conversion succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v6_convert_b200.out:82`.
- v6 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `56e13505eec52e1f17552421d40525ca`
  - `configuration/roarm_m3_base.usd` `e9976dbdce4cca0be2daf5630381210b`
  - `configuration/roarm_m3_physics.usd` `649b02bdeee3af67baf74b19b4501ae2`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- First v6 telemetry without LD_PRELOAD crashed and is not physics evidence.
- Valid LD_PRELOAD telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v6_ldpreload_b200.out:417-418`
  reports `target_error_m=0.024584`, `moving_contact=YES`,
  `counter_contact=NO`, `one_sided_push=YES`, `reached=NO`, and
  `success_claim=NO`.

v7:

- Local/static v7 object-frame analysis exists as a tracked script:
  `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
  md5 `598c7ac68f0844143ac9589c18c2b7e6`.
- v7 static/prep script exists as a tracked script:
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
  md5 `dd1e4723b2930fc7795c65cd104e4587`.
- v7 prep is diagnostic/static only. It mounts a fixed counter jaw on `link5`
  and keeps the moving jaw on `gripper_link`; see
  `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`.
- B200 v7 prep succeeded:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out:23-28`
  reports open descent clearance, moving contact, fixed-counter slop contact,
  and `v7_candidate_static_plausible=YES`.
- v7 prep md5s:
  - URDF `307384a6e18b3c70ef8c74a6c0524148`
  - moving mesh `b8cc44b72cf1d89404f9df3adf079b17`
  - counter mesh `7e56e1c6c3d35a8e3af1fe98d5f8e0cc`
- Initial v7 conversion attempts did **not** succeed. Normal and wrong-library
  LD_PRELOAD conversion attempts both crashed with B200 NVIDIA/NVML/GLX
  driver-library mismatch:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_b200.err:1-7,64-66,87-90`;
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_ldpreload_b200.err:1-7,64-66,87-90`.
- D024 conversion-only retry succeeded with the matching B200 userspace
  overrides:
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- D024 conversion stdout:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out:84-89`
  reports `cube2cm_fixed_counter_jaw_v7_link` merged into `link5`,
  `hand_tcp` merged into `link5`, and `base_link` merged into `world`.
- D024 conversion stderr:
  `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err:1-6`
  contains cpufreq/NVML-uninitialized messages only; grep found no traceback,
  exception, fatal, segfault, or driver/library mismatch.
- v7 D024 USD md5s:
  - `roarm_m3.usd` `4497024d25abab11de5c50e144124553`
  - `config.yaml` `f2777880ff2c90182484d82b7f49e5a6`
  - `configuration/roarm_m3_base.usd` `d7aae34ddca6a4d4f1ce092bda28d1a2`
  - `configuration/roarm_m3_physics.usd` `75f7b1e6da1f5f14019a53f091ec2076`
  - `configuration/roarm_m3_robot.usd` `5452694ecb266c48d9d333e98fda4e78`
  - `configuration/roarm_m3_sensor.usd` `656c6832b091e467c0af6f292c403e11`
- Therefore v7 is static/prep-valid and USD-converted under D024, but still
  was physics-unvalidated before the approved runtime telemetry below.
- Approved v7 D024 runtime jaw telemetry was run close_26-only:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out`.
  Scope line 38 confirms telemetry-only, no training, no constraints,
  no SurfaceGripper, no transport/release, no gate tuning, and no success claim.
- v7 telemetry selected the 3cm cube and IK was OK:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:39`.
- Authored v7 runtime geometry used `counter_parent=link5`:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:68`.
- Final close_26 telemetry failed:
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out:419`
  reports `target_error_m=0.023422`, `moving_contact=YES`,
  `counter_contact=NO`, `moving_slop_contact=YES`,
  `counter_slop_contact=NO`, `one_sided_push=YES`, and `reached=NO`.
- Aggregate line 420 reports `approach_ok=YES`, `descend_ok=YES`,
  `close_reached=NO`, `attach_calls=0`, `posewrite_calls=0`,
  `telemetry_only=YES`, and `success_claim=NO`.
- v7 runtime stderr
  `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.err:1-4`
  contains cpufreq/NVML-uninitialized/Fabric messages; grep found no traceback,
  exception, fatal, segfault, or driver/library mismatch. Post-run process check
  was empty.
- v7 runtime log md5s:
  - stdout `3939f08ea684c34f76669293b96610ba`
  - stderr `a0cb0d2eb0dca684599e693fcd1e7af7`

## Current Interpretation

- This is **not** physics success.
- TCP-only IK is not the current blocker: approach/descent IK succeeded in the
  v4/v5/v6/v7 telemetry logs.
- The current Isaac rigid-cube/jaw collision/contact proxy is not reproducing
  real foam grasp. The rigid cube is pushed by the moving jaw before counter
  contact closes, producing one-sided push.
- Do not say or imply that the real robot cannot grasp the cube.
- Conversion success is only asset import success. It does not validate grasp
  physics.
- v7 link5-fixed counter/backstop did not resolve the runtime rigid-contact
  failure. It remains useful evidence for the proxy mismatch, not a solved
  grasp primitive.
- 2026-05-21 analytical/modeling review narrowed the next branch: rigid offset
  variants are no longer the primary path unless they test a new mechanism.
  v7 briefly entered moving-contact + counter-slop-contact near close step 2,
  then one-sided push began by close step 3 and counter slop contact disappeared.
  The next primary branch should explicitly model foam/contact compliance.
- Static compliance-proxy audit now exists:
  `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
  md5 `bd1f26da1d371e27b559528a6210a941`.
- That audit predicts that a bounded virtual-compression/contact envelope of
  about `0.001813m` is enough to relabel counter support through close steps 2-4,
  but this is **not sufficient**: step 3 and step 4 still violate the existing
  push-speed gate, so contact-label/slop expansion alone would overclaim.
- Future close_26 runtime must reduce the step-3 speed impulse and keep counter
  support through step 4; merely increasing slop/contact labels is not a physics
  solution.
- Static dynamics design audit now exists:
  `sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
  md5 `d43c93d2810dd56468e5d8b885013146`.
- Dynamics audit rejects mass-only inertia: a constant-impulse estimate would
  require about `0.248kg` object mass to keep close steps 3-5 below the existing
  `0.005m/s` push-speed gate from the current `0.020kg` diagnostic cube.
- Selected next design **on paper/static only**: a future
  soft-contact/material diagnostic, because it directly targets the required
  `91.9%` speed suppression across close steps 3-5 while preserving counter
  support through step 4. Runtime remains not approved.
- Soft-contact/material runtime candidate is now code-designed but still
  default-off and separately unapproved:
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `7a261b72386ee549cb0ce162916597f7`.
- The new probe option is only `--soft_contact_material_diagnostic`; when absent,
  the baseline object physics constants remain the prior values
  (`static_friction=1.5`, `dynamic_friction=1.2`, `restitution=0.0`,
  solver iterations `8/1`, max depenetration velocity `5.0`).
- Added a local posthoc criteria audit:
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `a28c2fa8d8d58c617720f96417707677`. It does not launch Isaac and can reject
  future stdout logs against fixed criteria: step-3 speed <= `0.005m/s`, no
  one-sided push through steps 2-4, step-4 counter gap <= `0.002m`, step-4 target
  error <= `0.003m`, close reached, zero attach/posewrite, and no success claim.
- The audit also requires matching future-run metadata:
  `soft_contact_material_diagnostic=YES`,
  `mode=soft_contact_material_diagnostic`, and
  `runtime_candidate_requires_separate_approval=YES`.
- Static self-check against the encoded v7 reference intentionally returns FAIL:
  v7 fails on close reached, step-3 speed, one-sided push at steps 3-4, and
  step-4 target error. This confirms the future candidate is falsifiable rather
  than a looser success label.
- Static self-check against `--use_synthetic_pass_reference` returns PASS, so the
  audit is not hardwired to fail; it accepts only logs meeting the fixed
  close_26 criteria.
- Added failure-mode register:
  `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`. It records the v7
  failure chain, methods already tried, what not to repeat, and the exact
  telemetry changes required before any future hold-lift/dataset/training gate.
- Added static runtime-readiness preflight:
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `1d022dbbcd57481d1fbf6763663c5041`. It does not launch Isaac or execute the
  runtime probe. Local run reported `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`
  after verifying default-off wiring, metadata-guarded audit behavior, v7
  reference rejection, synthetic pass acceptance, and the future close_26 command
  shape.
- After explicit user approval, the B200 soft-contact/material close_26 runtime
  was executed with the correct micromamba Isaac Sim env. It **FAILED** the fixed
  posthoc criteria:
  `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`
  md5 `c3c81c1e6d481f23fdbb35411987ea8a`, stderr md5
  `c0d91f52cb47b553b3d7746ac08995f8`.
- Valid soft-contact run facts:
  line 37 `soft_contact_material_diagnostic=YES`, line 39 soft-contact physics
  params, line 377 step-3 speed `0.049059m/s` and `one_sided_push=YES`, line 378
  step-4 target error `0.003492m` and `one_sided_push=YES`, line 420
  `future_close26_posthoc_pass=NO`, line 421 `close_reached=NO`,
  `attach_calls=0`, `posewrite_calls=0`, `success_claim=NO`.
- Two execution-command pitfalls are also recorded: direct system Python failed
  with missing `isaaclab`, and `./IsaacLab/isaaclab.sh -p` failed because
  `_isaac_sim/python.sh` was missing. Correct B200 runtime path is the
  `isaacsim_5_1` micromamba env with `OMNI_KIT_ACCEPT_EULA=YES`.
- Added static virtual compression+damping design script:
  `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
  md5 `c45fb69a4cef556deaa87cb5247b4c73`. Local run says material-only produced
  only 20.8% step-3 suppression vs rigid v7, 13.9% at step 4, and 4.5% at step 5.
  The next proxy still needs up to `90.4%` extra speed suppression from the
  soft-contact result. The proposed static/code proxy uses a bounded `0.002m`
  compression budget, max plausible compression `0.003m`, velocity damping active
  by close step 3, and residual velocity ratio `0.08`; it still must prove
  step-4 target error and no one-sided push in any future runtime.
- Runtime probe now has a default-off
  `--virtual_compression_damping_diagnostic` candidate:
  `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
  md5 `9e5292f176d9b90df30cfd23bdb36028`. It is a runtime candidate only,
  separately unapproved, and logs velocity damping writes separately from
  `attach_calls`/`posewrite_calls`.
- Posthoc audit/readiness were updated for the virtual mechanism:
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
  md5 `065110aa514e49c62747fe4ab6ceecf4`;
  `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
  md5 `04934025ecf5a4793002c2d9fed20b36`. The audit now kills a virtual
  candidate unless close step 3 logs `virtual_support=YES`,
  `virtual_damping_active=YES`, and at least one
  `virtual_velocity_damping_writes_total` by that step. It also requires positive
  aggregate `virtual_velocity_damping_writes`. The readiness check is still local
  static only; it printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES` for the
  candidate command shape but did not run Isaac.

Approved virtual compression+damping runtime result:

- User approved one close_26-only B200 runtime for the default-off virtual
  compression+damping candidate. No training, dataset generation, hold-lift,
  constraints, SurfaceGripper, transport/release, gate tuning, or success claim
  was run.
- B200 stdout:
  `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`
  md5 `7097b2c2eb70ba77d363dcfade601952`; stderr md5
  `35dc65de1f7982e1a7b1115784cff075`.
- stdout line 37 confirms strict diagnostic scope,
  `virtual_compression_damping_diagnostic=YES`, close_26 only, and no
  disallowed mechanisms. Lines 39-40 confirm
  `mode=virtual_compression_damping_diagnostic`,
  `runtime_candidate_requires_separate_approval=YES`, compression budget
  `0.002m`, max plausible compression `0.003m`, residual ratio `0.08`, damping
  start close step `3`, velocity writes YES, pose writes NO.
- Runtime partially worked but still failed: line 378 step 3 has
  `object_speed_mps=0.004955`, `virtual_damping_active=YES`,
  `virtual_velocity_damping_writes_total=1`, and `one_sided_push=NO`. Line 379
  step 4 has support OK (`counter_gap_obj_m` y `0.001794`) and
  `object_speed_mps=0.003203`, but target error fails:
  `target_error_m=0.003130 > 0.003`.
- Line 380 step 5 loses the support/damping window:
  `counter_gap_obj_m` y `0.002738 > 0.002`,
  `virtual_support=NO`, `virtual_damping_active=NO`,
  `object_speed_mps=0.050912`, and `one_sided_push=YES`.
- Line 421 reports `future_close26_posthoc_pass=NO`; line 422 reports
  `close_reached=NO`, attach/posewrite zero, telemetry-only, and no success
  claim. The posthoc audit also returns FAIL, with failing criteria
  `close_reached` and `target_step4_within_gate`.
- Added static-only failure analysis:
  `sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
  md5 `0cccd8d9f3e5aaf7dc27fc3eb034967c`. It reports step-3 and step-4 damping
  suppression about `92.0%`, step-4 target excess `0.130mm`, step-5 support
  excess `0.738mm`, and final counter gap `0.013828m`, which is
  `0.010828m` beyond the `0.003m` max plausible compression. Its next
  requirement is target-error control below 3mm plus support/damping horizon
  beyond step 4; speed gate alone is not success.
- Added the next static design script:
  `sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
  md5 `dca5322e654f3b0d415822f0972d383e`. It rejects stronger damping alone
  because step-4 target error still exceeds the fixed gate by `0.130mm` and
  step-5 target excess is `1.843mm`; it also rejects support-label-only because
  final counter gap is `0.013828m`, `0.010828m` beyond the 3mm max plausible
  compression. The proposed next mechanism shape is default-off target-guarded
  micro-close plus support-horizon damping, with unchanged fixed audit gates.

B200 endgame / Track B boundary:

- User stated the Track B plan separately: backup pipeline test, B200
  OpenVLA-OFT env setup with HARD RULE #15 nightly cu128 recovery,
  1K smoke, 30K-50K OpenVLA-OFT finetune, offline eval/final backup, then pi0
  RunPod handoff after B200 release around 2026-05-22 23:59.
- Track B stays separate from Track A P7/Branch B verdicts. Track A priority
  before B200 release is preservation of B200 `/tmp` logs, code, docs, and USD
  artifacts.
- Added preservation/Track B boundary doc:
  `claudedocs/b200_endgame_track_a_preservation_track_b_plan_20260521.md`.
- Local untracked `b200_backup_20260521/` exists. During inspection, an
  rsync-style temp file `._speedtest_model.safetensors.MIJ5aq` was observed
  growing, but the final check showed only `env.sh` remains. Do not treat this
  as a completed backup; target confirmation and a clean rsync speed test are
  still needed before Track B heavy runs.

## Current Worktree

2026-05-21 compact checkpoint:
`git status --short --untracked-files=all` is expected dirty/untracked. Do not
revert those changes unless explicitly requested. HEAD remains `f4404e9` (`v7`).
The dirty/untracked set is:

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

Current-session verification passed:

- `git diff --check`
- `python -m py_compile` for runtime probe, criteria audit, readiness, and
  virtual compression+damping static design
- v7 reference audit with expected virtual mechanism returned FAIL as intended
- synthetic virtual metadata with no damping writes returned FAIL as intended
- synthetic pass audit with expected virtual mechanism returned PASS
- virtual readiness printed `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`
- approved virtual compression+damping B200 runtime executed once and returned
  posthoc FAIL, with B200 stdout lines 378-382 and 421-422 as the active
  evidence
- virtual runtime failure static analysis passed
- target/support-horizon static design passed

The latest runtime is a useful negative/partial result, not grasp success.

## Current Direction

Immediate priority: material-only soft-contact is falsified, and the approved
virtual compression+damping runtime is also not a close_26 pass. It did prove
that explicit damping can suppress step-3 speed and remove one-sided push through
steps 2-4, but it did not keep target error under gate or maintain support beyond
step 4. Do not run another runtime without separate approval.

Next technical action after state repair:

1. Do not run training, dataset generation, runtime telemetry, hold-lift,
   transport/release, constraints, or SurfaceGripper.
2. Treat v7 as static/prep-valid, D024 USD-converted, and runtime-telemetry
   failed at close_26.
3. Do not interpret v7 conversion or telemetry exit code as grasp success.
4. More rigid offset probing is not the default next path; it must justify a new
   falsifiable mechanism if proposed.
5. Next branch should be static/code-first failure attribution for the virtual
   result: target-guarded micro-close and support/damping horizon, not gate
   tuning. Hold-lift remains not approved and is not justified by this failed
   telemetry.
6. Before Track B heavy runs or B200 release, preserve Track A `/tmp` logs and
   code/doc artifacts. Keep Track B OpenVLA-OFT/pi0 work separate.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` latest P7/Branch B decisions
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260520_p7_branch_b_cube_contact_state_repair.md`
6. `claudedocs/session_20260521_p7_branch_b_compliance_direction_analysis.md`
7. `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`
8. `claudedocs/session_20260520_p7_branch_b_normalized_cube_grasp_feedback.md`
9. `sim_scripts/p7_branch_b_cube2cm_v7_object_frame_static_analysis.py`
10. `sim_scripts/p7_branch_b_prepare_roarm_cube2cm_opposing_jaw_v7_urdf.py`
11. `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
12. `sim_scripts/p7_branch_b_cube2cm_compliance_proxy_static_analysis.py`
13. `sim_scripts/p7_branch_b_cube2cm_compliance_dynamics_static_design.py`
14. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
15. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
16. `sim_scripts/p7_branch_b_cube2cm_virtual_compression_damping_static_design.py`
17. `sim_scripts/p7_branch_b_cube2cm_virtual_runtime_failure_static_analysis.py`
18. `sim_scripts/p7_branch_b_cube2cm_target_support_horizon_static_design.py`
19. `claudedocs/b200_endgame_track_a_preservation_track_b_plan_20260521.md`
20. The B200 `/tmp` logs cited above, with line numbers.

## Explicitly Not Approved

- Training or VLA/SmolVLA/LeRobot training.
- Cube sim dataset generation.
- Old handoff diagnostic.
- Constraint/default integration.
- SurfaceGripper attachment.
- Transport, transport target, release, or scripted release variants.
- P7 scalar/threshold/release guidance tuning.
- Diagnostic gate tuning.
- Old 2cm sweep continuation.
- Claiming physics success from v4/v5/v6 conversion.
- Hold-lift or runtime telemetry unless separately approved.
