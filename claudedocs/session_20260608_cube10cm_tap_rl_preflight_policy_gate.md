# 2026-06-08 cube10cm tap RL preflight/policy gate

## Scope

- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier branch.
- Task: consolidate link5-corner event-label metadata, noisy Tier-B teacher policy, visual contact risk, and the default-off 10cm tap RL wrapper sanity before any dataset/RL/RoArm step.
- Added local-only script: `sim_scripts/cube10cm_tap_rl_preflight_policy_gate.py`.
- Not run: new IsaacLab runtime, new GPU physics, dataset generation, PPO/RL training, RoArm-M3-Pro control, B200/SSH/pull, Track A.

## Correction

- Isaac Lab is not treated as broken. Local GPU evidence already exists for the wrapper sanity.
- The confusion was layer mismatch:
  - trace-derived visual inspection explains the existing link5 DiffIK runtime contact geometry;
  - `RoArm-CubeTap10cm-Direct-v0` is a new default-off RL wrapper to avoid the old 3cm relocation objective.
- CPU/sandbox diagnostic failure is not promotion evidence and must not override the local RTX 4090/cuda:0 logs.

## Inputs

- Event-label manifest:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_event_label_metadata_manifest.json`
- Noisy Tier-B teacher policy:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_noisy_tierb_teacher_policy_gate.json`
- Visual proxy-contact inspection:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_link5corner_visual_proxy_contact_inspection.json`
- Runtime gate audit:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_env_runtime_gate_audit.json`

## Results

- Preflight/policy gate line 1: local consolidation only; no GPU, dataset, training, robot, SSH, B200, or Track A.
- Line 2: Isaac Lab status is OK under local GPU wrapper sanity; CPU/sandbox diagnostic failure is not promotion evidence.
- Line 3: event-label metadata is `READY_LOCAL_ONLY`; env wrapper is `READY_LOCAL_PREFLIGHT_ONLY`; weak 1mm is the only verified objective evidence; strong 2-3mm is not required by current evidence.
- Line 4: strict clean action teacher remains blocked; noisy Tier-B action teacher requires explicit user/professor exception; tiny action dataset is not allowed by default; Tier B count is 16 and clip mean is `0.666666667`.
- Line 5: visual risk remains preserved: clean tap not visually verified, grazing/outside contact and early freeze supported.
- Line 6: positive-control tap sanity in the new wrapper has not been run, so PPO, large dataset, and RoArm remain blocked.
- Line 7: next allowed local work is reward/done/log contract freeze plus scripted positive-control design; no new GPU runtime without explicit approval.

## Verdict

- Unblocked:
  - link5 event-label/quality-tier metadata for local bookkeeping;
  - default-off 10cm tap env wrapper for local preflight/design only.
- Still blocked:
  - strict action-teacher dataset;
  - noisy Tier-B action-teacher exception;
  - tiny action dataset dry run;
  - PPO/RL training;
  - large dataset generation;
  - RoArm deployment.

## Next

- Freeze the 10cm tap reward/done/log contract locally.
- Design one scripted positive-control tap sanity for the new wrapper without launching it yet.
- Keep strong 2-3mm tap as out of scope unless explicitly required by professor/user.

## Follow-Up: Contract Freeze And Positive-Control Design

- Added local-only script:
  `sim_scripts/cube10cm_tap_rl_contract_positive_control_design.py`.
- Not run: new IsaacLab runtime, GPU physics, dataset generation, PPO/RL, RoArm,
  B200/SSH/pull, Track A.
- Summary line 2 freezes the objective/reaction contract:
  final 1cm is default-off, tap target is `0.001m`, overshoot is `0.020m`, and
  reaction remains contact-gated.
- Summary line 3 freezes reward/done:
  final-success leak count is `0`, overshoot terminates, success termination is
  default-off.
- Summary line 4 freezes logs:
  raw reaction, contact context, reaction seen, overshoot, and success are
  separate.
- Summary lines 5-6 design the next possible positive-control sanity, but do not
  launch it:
  `RoArm-CubeTap10cm-Direct-v0`, `cuda:0`, `num_envs=2`, `max_steps=120`,
  scripted TCP DifferentialIK-to-joint-delta actions, pass only if contact,
  contact-gated reaction, reaction seen, tap success are `>0` while overshoot is
  `0` and final flag is `0`.
- Summary line 7 verdict:
  contract design is ready only for considering one explicitly approved tiny
  positive-control runtime. PPO/RL, large dataset, RoArm, and action teacher
  remain blocked.

## Follow-Up: Approved Positive-Control Runtime

- Added runtime harness:
  `roarm_rl/test_positive_control_cube_tap10cm.py`.
- First launch result:
  `BLOCKED` before env creation because the harness passed
  `TerrainImporterCfg(use_terrain_origins=False)`, which current local IsaacLab
  does not accept. This was a harness-only compatibility issue and not physics
  evidence.
- Fixed the harness by removing the unsupported terrain argument from both
  `test_positive_control_cube_tap10cm.py` and `test_sanity_cube_tap10cm.py`.
- Actual tiny local GPU run:
  local RTX4090/cuda:0, local USD, `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`,
  `max_steps=120`, fixed cube `(0.250, 0.000)`, push dir `(1, 0)`,
  side-center TCP target via `tcp_top_margin_m=-0.050`, no dataset/PPO/robot/B200/SSH/Track A.
- Result: `FAIL`.
  Reset IK and teacher goal were both OK (`1.0/1.0`), but contact never registered
  (`contact_seen=0.0`). Raw reaction speed fired (`reaction_signal=1.0`) without
  contact context (`reaction_contact_context=0.0`), so `reaction_seen=0.0` and
  `tap_success=0.0`.
- Motion evidence:
  max displacement `0.000824004m`, max z delta `0.000043014m`, max speed
  `0.077039227m/s`, overshoot `0.0`.
- Added and ran local-only failure audit:
  `sim_scripts/cube10cm_tap_rl_positive_control_failure_audit.py`.
  It shows final face gap remained outside the contact band:
  `final_face_gap_m=-0.021077018`, shortfall `0.011077018m`.
- Interpretation:
  the wrapper false-positive guard worked correctly. It did not convert a raw
  speed reaction into tap success without contact context.
- Added revised candidate support/design but did not run it:
  `external_closed_loop` controller mode in the harness, plus
  `sim_scripts/cube10cm_tap_rl_revised_positive_control_candidate_design.py`.
  The single revised candidate is code-ready and local-design-ready only;
  any new GPU runtime requires explicit approval.
- Still blocked:
  PPO/RL, large dataset, action teacher/tiny action dataset, and RoArm.

## Follow-Up: Positive-Control Visual Contact Audit

- Added local existing-log-only visual audit:
  `sim_scripts/cube10cm_tap_rl_positive_control_visual_contact_audit.py`.
- Generated:
  `cube10cm_tap_rl_positive_control_visual_contact_audit.png`,
  `cube10cm_tap_rl_positive_control_visual_contact_audit.svg`,
  `cube10cm_tap_rl_positive_control_visual_contact_audit.html`,
  `cube10cm_tap_rl_positive_control_visual_contact_audit.json`, and summary.
- Important limitation:
  this is reset/final scalar reconstruction from saved JSON, not a per-step
  video/trace. It does not launch IsaacLab/GPU, data generation, training,
  robot control, SSH/B200, or Track A.
- Summary line 3 reconstructs the contact frame:
  initial/final along `-0.070252299m/-0.071077018m`, initial/final face gap
  `-0.020252299m/-0.021077018m`, final shortfall to the `[-0.010,+0.010]m`
  contact band `0.011077018m`.
- Summary line 4 gives the axis diagnosis:
  lateral and vertical are OK (`0.000003905m` and `0.000538668m`), while the
  along gap remains outside the band. Gap delta `-0.000824720m` nearly cancels
  cube displacement `0.000824004m` (`abs=0.000000715m`).
- Summary line 5 confirms the guard:
  contact `0.0`, raw reaction signal `1.0`, contact context `0.0`, tap success
  `0.0`, wrapper false-positive blocking `True`.
- Visual inspection of the generated PNG confirms the red final TCP marker stays
  left/outside the green contact band; the failure is not a lateral or height
  placement issue.
- Still blocked:
  PPO/RL, large dataset, action teacher/tiny action dataset, and RoArm.
- Next possible runtime remains exactly one revised `external_closed_loop`
  positive-control candidate only after explicit approval.

## Follow-Up: Strict Revised Candidate Correction

- Rechecked the revised candidate design because the old candidate was not truly
  one-knob:
  it combined `controller_mode=external_closed_loop` with
  `action_smoothing_alpha=1.0` and `contact_joint_delta_scale=1.0`.
- Updated and reran:
  `sim_scripts/cube10cm_tap_rl_revised_positive_control_candidate_design.py`.
- New summary line 3:
  `strict_selected=True`, `changed_knobs=1`,
  `controller_mode=builtin_teacher->external_closed_loop`,
  `action_smoothing_alpha=env_default_0.25`,
  `contact_joint_delta_scale=env_default_0.35`,
  `closed_loop_push_steps=default_72`, status `DESIGNED_NOT_RUN`.
- New summary line 4:
  the smoothing/scale `1.0/1.0` strength variant is
  `NOT_SELECTED_NOT_RUN` because it would mix controller mode with action
  smoothing and contact delta scale.
- The selected next candidate, if explicitly approved, is therefore strict
  controller-mode-only. It still does not unblock PPO/RL, large dataset, action
  teacher/tiny action dataset, or RoArm.

## Follow-Up: Approved Strict External-Closed-Loop Runtime

- User explicitly approved the next tiny GPU runtime.
- Ran exactly one local RTX4090/cuda:0 IsaacLab positive-control runtime:
  `RoArm-CubeTap10cm-Direct-v0`, local USD, `num_envs=2`, `max_steps=120`,
  `seed=962`, fixed cube `(0.250, 0.000)`, push dir `(1, 0)`,
  `controller_mode=external_closed_loop`, action smoothing default `0.25`,
  contact delta scale default `0.35`, closed-loop push steps default `72`.
- Not run:
  PPO/RL, dataset generation, action-teacher dataset, RoArm control, SSH/B200,
  Track A, or another GPU runtime after this one.
- Runtime result: `FAIL`.
  Strict external controller target solving worked
  (`closed_loop_ik_ok_rate=1.0`, mean closed-loop IK error `0.617469787mm`), but
  contact stayed `0.0`.
- Tap logs:
  raw reaction signal `1.0`, reaction contact context `0.0`, reaction seen
  `0.0`, tap success `0.0`, overshoot `0.0`.
- Motion:
  max displacement `0.000824124m`, max speed `0.077135637m/s`, max z delta
  `0.000043118m`.
- Failure/visual audits:
  final face gap `-0.020518493m`, contact-band shortfall `0.010518493m`.
  Lateral and vertical were OK; along gap remains the blocker.
- Comparison with builtin teacher positive-control:
  strict external improved final face gap by only `0.000558525m`, still far
  outside the `[-0.010,+0.010]m` contact band.
- Corrected a local harness gate issue:
  external mode now uses `closed_loop_ik_ok_rate` as the controller-goal gate
  instead of stale `teacher_goal_ok_rate`. This does not change the runtime
  verdict because contact/reaction context/tap success were all still zero.
- Added future diagnostic log keys to the tap wrapper:
  `cube_push_tcp_cube_dist_m`, `cube_push_joint_delta_abs_mean`,
  `cube_push_contact_slowdown_mean`, and `cube_push_teacher_blend_mean`.
  These are code-ready but not runtime-verified because no second GPU run was
  launched after instrumentation.
- Current interpretation:
  the strict controller layer is not enough. The next blocker is likely in the
  action application path, such as smoothing, per-step delta cap, contact
  slowdown, target reference/lead limit, or some combination. This is not yet
  proven because the just-finished runtime did not include those diagnostic log
  keys.
- Still blocked:
  PPO/RL, large dataset, action teacher/tiny action dataset, and RoArm.
- Next:
  local-only design/instrumentation of one action-path/slowdown/cap diagnostic
  candidate. Any new GPU runtime requires explicit approval.

## Follow-Up: Instrumented Action-Path Sanity

- User asked to proceed with the action-path diagnostic.
- Ran one additional local RTX4090/cuda:0 tiny strict external sanity with the
  newly added action-path logs:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=external_closed_loop`.
- Not run:
  DiffIK dataset generation, PPO/RL, action-teacher dataset, RoArm control,
  SSH/B200, Track A.
- Runtime result: `FAIL`.
  `controller_goal_ok_rate=1.0`, `closed_loop_ik_ok_rate=1.0`, but contact,
  reaction context, reaction seen, and tap success stayed `0.0`.
- New action-path logs:
  TCP-cube distance `0.070519388m`, joint delta abs mean `0.005000000`,
  contact slowdown mean `1.0`, teacher blend `0.0`, action penalty `-0.015`.
- Interpretation:
  contact slowdown is inactive, and the mean joint delta is below the `0.010rad`
  per-step cap. So current final-scalar evidence does not support direct
  slowdown/cap blame.
- Failure/visual audits still show the same blocker:
  final face gap `-0.020518493m`, shortfall `0.010518493m`, lateral/vertical OK.
- Added per-step aggregate trace instrumentation to the harness after this run:
  face gap min/max/final, contact-band shortfall min/final, TCP distance min,
  joint delta max, and controller trace stats.
- This new per-step aggregate code is compile-checked but not yet runtime-verified,
  because no further GPU run was launched.
- Current status:
  the DiffIK action dataset has not been built. We are still in wrapper and
  positive-control debugging. Dataset/RL/RoArm remain blocked.

## Follow-Up: TCP-Progress Instrumented Sanity

- User asked to verify whether the face gap closes mid-run.
- Ran one local RTX4090/cuda:0 tiny strict external sanity with per-step
  aggregate trace enabled:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=external_closed_loop`.
- Not run:
  DiffIK dataset generation, PPO/RL, action-teacher dataset, RoArm control,
  SSH/B200, Track A.
- Runtime result: `FAIL`.
- Key trace diagnostics:
  initial face gap `-0.020252299m`, best/closest face gap `-0.019507330m`,
  worst face gap `-0.024245869m`, final face gap `-0.020518493m`.
- Best improvement from initial:
  `0.000744969m`. The TCP/contact proxy does move slightly toward the band, so
  the controller/action mapping is not a complete no-op.
- But best contact-band shortfall is still `0.009507330m`.
  It never gets near the `[-0.010,+0.010]m` band.
- Action path:
  contact slowdown `1.0`, joint delta max/mean `0.005000000`, TCP distance min
  `0.069517583m`.
- Interpretation:
  this is not primarily a timing/contact-band miss. The evidence points to
  insufficient action progress/gain/target application through the sim action
  path.
- Current status:
  DiffIK action dataset is still not built. PPO/RL, large dataset, action
  teacher/tiny action dataset, and RoArm remain blocked.
- Next:
  local-only design of exactly one action-progress/gain/target-application
  candidate and its pass/fail audit. Any new GPU runtime requires explicit
  approval.

## Follow-Up: Action-Progress Candidate Design

- User asked to proceed from the TCP-progress conclusion.
- Added and ran `sim_scripts/cube10cm_tap_rl_action_progress_candidate_design.py`.
- This is local design/static audit only:
  no GPU runtime, no IsaacLab physics launch, no dataset generation, no training,
  no robot control, no SSH/B200, no Track A.
- Inputs:
  latest TCP-progress result audit, latest TCP-progress runtime JSON, the
  positive-control harness, and the 10cm tap wrapper source.
- Result:
  `code_ready=True`, `basis_ok=True`.
- Failure basis preserved:
  contact `0.0`, reaction context `0.0`, tap success `0.0`;
  face gap moved slightly toward the band but did not get near it;
  best improvement `0.000744969m`, best shortfall `0.009507330m`.
- Action-path basis:
  previous smoothing `0.25`, closed-loop alpha final `1.0`,
  contact slowdown `1.0`, joint-delta abs max `0.005000000`,
  TCP distance min `0.069517583m`.
- Selected next candidate:
  exactly one runtime knob, `action_smoothing_alpha 0.25 -> 1.0`.
- Fixed for that candidate:
  `controller_mode=external_closed_loop`, `closed_loop_push_steps=72`,
  `contact_joint_delta_scale=0.35`, fixed cube/push direction, side-center TCP
  height, precontact clearance, through distance, contact/reaction gates, and
  final-1cm default-off contract.
- Not selected first:
  `contact_joint_delta_scale` because slowdown was inactive;
  `closed_loop_push_steps` because alpha already reached `1.0`;
  goal push/contact band because that changes contact geometry;
  `joint_delta_reference` because current logs do not yet prove target-vs-joint
  lead is the dominant cause.
- Verdict:
  ready only for one explicitly approved tiny cuda:0 positive-control runtime.
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked.

## Follow-Up: Action Smoothing 1.0 Positive-Control

- User explicitly approved the next tiny positive-control runtime.
- Ran one local RTX4090/cuda:0 Isaac Lab sanity:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=external_closed_loop`, single changed knob
  `action_smoothing_alpha 0.25 -> 1.0`.
- Not run:
  DiffIK dataset generation, tiny action dataset dry run, PPO/RL, large dataset,
  RoArm control, SSH/B200, Track A.
- Runtime result: `FAIL`.
- Runtime summary:
  contact `0.0`, raw reaction signal `1.0`, reaction context `0.0`, reaction seen
  `0.0`, tap success `0.0`, overshoot `0.0`, max displacement `0.000820309m`,
  final face gap `-0.020514678m`, best shortfall `0.009533616m`.
- Visual audit:
  final TCP remains outside the contact band; lateral and height are OK
  (`final_lateral=0.000023734m`, `final_vertical_offset=0.000356138m`).
- Result comparison:
  smoothing did not improve contact progress. Best improvement from initial went
  from `0.000744969m` to `0.000718683m`, a delta of `-0.000026286m`;
  best shortfall worsened from `0.009507330m` to `0.009533616m`;
  final gap changed by only `0.000003815m`.
- Interpretation:
  smoothing is not the root cause. The next unresolved layer is action command
  magnitude, per-joint delta cap, target reference, or target lead/lead-limit.
- Critical correction:
  the previous `joint_delta_abs_max=0.005000000` in summaries was max-over-time
  of `cube_push_joint_delta_abs_mean`, not a per-joint delta max. It cannot prove
  the per-joint cap is inactive.
- Implemented follow-up instrumentation for future approved diagnostics:
  per-joint `cube_push_joint_delta_abs_max`, `cube_push_joint_delta_cap_rate`,
  `cube_push_action_abs_mean/max`, `cube_push_target_lead_abs_mean/max`, and
  `cube_push_target_lead_limit_rate`.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked.

## Follow-Up: Cap / Target-Lead Diagnostic

- User asked to proceed with the next gate:
  per-joint cap, action command magnitude, and target lead.
- Ran one local RTX4090/cuda:0 tiny diagnostic:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=external_closed_loop`, default `action_smoothing_alpha=0.25`,
  default `contact_joint_delta_scale=0.35`, default `closed_loop_push_steps=72`.
- Not run:
  DiffIK dataset generation, tiny action dataset dry run, PPO/RL, large dataset,
  RoArm control, SSH/B200, Track A.
- Runtime result:
  still `FAIL`, with contact `0.0`, reaction context `0.0`, tap success `0.0`,
  overshoot `0.0`, best shortfall `0.009507330m`, final shortfall
  `0.010518493m`.
- New action-path evidence:
  `action_abs_mean=0.5`, `action_abs_max=1.0`,
  `joint_delta_abs_mean=0.005000000`,
  `joint_delta_abs_max=0.010000000`, `joint_delta_cap_rate=0.5`.
- Lead/slowdown evidence:
  contact slowdown remains inactive (`1.0`);
  target lead-limit is seen in trace (`target_lead_limit_rate_trace=0.333333343`)
  but final lead-limit rate is `0.0`, so it is secondary to cap/action saturation.
- Visual audit:
  final TCP remains outside the contact band; lateral and height are OK.
- Result audit verdict:
  `CAP_ACTION_SATURATION_PRIMARY_HYPOTHESIS`.
- Implemented default-off harness override:
  `--max_joint_delta_per_step_rad`.
- Added local-only design audit:
  `sim_scripts/cube10cm_tap_rl_cap_only_candidate_design.py`.
- Selected next candidate, not run:
  exactly one knob, `max_joint_delta_per_step_rad 0.010 -> 0.040`.
- Fixed for that candidate:
  action scale, action smoothing, controller mode, closed-loop steps, contact
  slowdown, cube geometry, side-center height, precontact, through distance,
  target reference/lead limit, and pass gates.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked. Any cap040 runtime requires explicit approval.

## Follow-Up: Cap040 Positive-Control

- User explicitly approved the next runtime:
  cap040 tiny cuda:0 positive-control, one run only.
- Ran one local RTX4090/cuda:0 Isaac Lab sanity:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=external_closed_loop`, single changed knob
  `max_joint_delta_per_step_rad 0.010 -> 0.040`.
- Not run:
  DiffIK dataset generation, tiny action dataset dry run, PPO/RL, large dataset,
  RoArm control, SSH/B200, Track A.
- Runtime result:
  still `FAIL`.
- Runtime summary:
  contact `0.0`, raw reaction signal `1.0`, reaction context `0.0`,
  reaction seen `0.0`, tap success `0.0`, overshoot `0.0`,
  max displacement `0.000824124m`, final face gap `-0.020518493m`,
  best shortfall `0.009507330m`, final shortfall `0.010518493m`.
- Cap effect:
  the cap override did apply. Compared with the previous cap/target-lead
  diagnostic, max trace joint delta went `0.010000000 -> 0.039999995`, and cap
  rate went `0.5 -> 0.0`.
- Contact progress:
  there was no improvement. Best shortfall remained `0.009507330m`, best face
  gap remained `-0.019507330m`, and both comparison deltas were `0.0`.
- Visual audit:
  final TCP remains outside the contact band. Lateral and vertical are OK;
  the blocker is along-axis live face gap.
- Interpretation:
  cap-only is falsified as the primary positive-control blocker. The next
  unresolved layer is target application, especially target lead-limit and/or
  target-vs-joint reference.
- 3cm DiffIK comparison:
  prior 3cm DiffIK was indeed run first, but its own audit only said the
  mechanism runtime ran. It did not mark dataset ready. The 3cm dataset build
  later produced a candidate dataset, and BC smoke was only a pipeline smoke,
  not a validated rollout. For 10cm, creating a DiffIK action dataset before
  contact-gated positive-control would risk storing a failed/no-contact teacher.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked.
- Next allowed work:
  local-only design of exactly one target-application candidate, likely choosing
  between `joint_target_lead_limit` and `joint_delta_reference`. Any new GPU
  runtime still requires explicit approval.

## Follow-Up: Target-Application Candidate Design

- After cap040, added a default-off harness override:
  `--joint_target_lead_limit_rad`.
- Added and ran `sim_scripts/cube10cm_tap_rl_target_application_candidate_design.py`.
- This is local design/static audit only:
  no GPU runtime, no Isaac Lab physics launch, no dataset generation, no
  training, no robot control, no SSH/B200, no Track A.
- Basis:
  `code_ready=True`, `basis_ok=True`, cap-only falsified as primary, cap no
  longer active, lead-limit observed, `target_lead_abs_max_trace=0.069168568`,
  and `target_lead_limit_rate_trace=0.5`.
- Selected next candidate:
  baseline is cap040, changed knobs vs cap040 is exactly 1:
  `joint_target_lead_limit_rad 0.060 -> 0.120`.
- Fixed for that candidate:
  `max_joint_delta_per_step_rad=0.040`, `joint_delta_reference=target`,
  `action_scale=0.04`, `action_smoothing_alpha=0.25`,
  `controller_mode=external_closed_loop`, cube geometry, side-center height,
  precontact, through distance, and pass gates.
- Not selected:
  `joint_delta_reference` because it changes target-base semantics and likely
  requires matching harness action-base review;
  action scale because it changes command normalization;
  goal push/contact band because it changes geometry;
  smoothing because it was already tested;
  cap-only because cap040 falsified it as primary.
- Verdict:
  `cap040_lead120` is ready only for explicit tiny cuda:0 runtime approval.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked.

## Follow-Up: Isaac Lab Direct IK Apply Source Cross-Check

- User challenged the contradiction:
  if TCP target and IK exist, why does contact progress not improve?
- Checked local installed Isaac Lab source instead of relying on memory.
- Isaac Lab `DifferentialInverseKinematicsAction` pattern:
  process task-space command, compute `joint_pos_des`, then call
  `set_joint_position_target(joint_pos_des, joint_ids)`.
- Local Isaac Lab differential IK test uses the same direct joint-target apply
  pattern.
- Current harness pattern:
  compute TCP target and IK, convert `target_rad - target_base_rad` back into
  normalized RL action, then let env smoothing/action scale/cap/reference/lead
  limit produce `robot_dof_targets`.
- Interpretation:
  current failure mixes two questions:
  whether the TCP target/IK is correct, and whether the RL action-wrapper path
  applies the resulting joint target strongly enough.
- Added default-off `external_closed_loop_direct_apply` support in the tap
  harness/env.
- Added and ran `sim_scripts/cube10cm_tap_rl_direct_ik_apply_candidate_design.py`.
- This is local design/static audit only:
  no GPU runtime, no Isaac Lab physics launch, no dataset generation, no
  training, no robot control, no SSH/B200, no Track A.
- Result:
  `code_ready=True`, `basis_ok=True`, `isaac_pattern_supported=True`.
- Selected next diagnostic:
  `direct_ik_apply_positive_control`.
- Implementation status:
  `HARNESS_AND_ENV_DEFAULT_OFF_MODE_READY`; designed but not run.
- Purpose:
  apply the IK joint target directly in an Isaac Lab-style positive-control loop
  to separate target geometry/IK failure from RL action target-application
  failure.
- Reserve:
  `cap040_lead120` moves behind direct-IK-apply; otherwise we may tune the
  wrapper before proving the IK target itself works.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked. Any direct-IK-apply cuda:0 runtime requires explicit approval.
