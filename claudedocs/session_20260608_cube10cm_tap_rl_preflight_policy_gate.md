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

## Follow-Up: Direct IK Apply Runtime Result

- User approved the next tiny local cuda:0 direct-IK-apply positive-control.
- Ran:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`,
  `seed=962`, `controller_mode=external_closed_loop_direct_apply`.
- This was a local RTX4090/Isaac Lab runtime only:
  no dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime contract:
  10cm cube, 0.72kg mass, final 1cm relocation disabled.
- Direct apply was active:
  summary line 3 reports `direct_ik_joint_target_apply=True`.
- Closed-loop IK was numerically OK:
  `closed_loop_ik_ok_rate=1.0`, mean IK error `0.617469787mm`.
- But the contact-gated tap still failed:
  contact `0.0`, reaction context `0.0`, reaction seen `0.0`,
  tap success `0.0`, overshoot `0.0`.
- Direct action path evidence:
  action abs max trace `0.0`, cap rate `0.0`, lead-limit rate `0.0`;
  therefore the normalized RL action wrapper was bypassed.
- Contact-frame trace:
  initial face gap `-0.020252299m`, best `-0.019533616m`,
  final `-0.019564968m`, best shortfall `0.009533616m`,
  final shortfall `0.009564968m`.
- Motion/reaction:
  max displacement `0.000922590m`, max speed `0.008078601m/s`;
  this is below the 1mm reaction signal.
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_direct_ik_apply_result_audit.py`.
- Visual audit:
  generated PNG/SVG/HTML and viewed the PNG locally. It shows initial, best,
  and final TCP all outside the contact band while lateral/vertical are OK.
- Interpretation:
  the old wrapper-only explanation is falsified for this target/path. The next
  question is target geometry / FK-frame agreement / actuator-follow, not
  lead/cap/action-scale tuning.
- Current status:
  DiffIK action dataset, tiny action dry run, PPO/RL, large dataset, and RoArm
  remain blocked.

## Follow-Up: Direct IK Telemetry Candidate

- Added default-preserving telemetry to
  `roarm_rl/test_positive_control_cube_tap10cm.py`.
- New telemetry keys:
  target face gap, target inside-contact-band rate, target lateral/vertical,
  target FK error, actual-FK-vs-Isaac-TCP frame error, target delta from actual
  joint, direct joint follow error, and actual joint step magnitude.
- Added summary line 10 for controller telemetry.
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_direct_ik_telemetry_candidate_design.py`.
- This was local design/static audit only:
  no GPU runtime, no dataset generation, no training, no robot control,
  no SSH/B200, no Track A.
- Design summary:
  `basis_ok=True`, `telemetry_ready=True`,
  previous contact `0.0`, previous best shortfall `0.009533616m`,
  wrapper-only explanation falsified.
- Selected next candidate:
  `direct_ik_apply_telemetry_repeat`.
- Candidate constraints:
  zero control-knob changes vs the direct-IK-apply runtime,
  same controller mode, `num_envs=2`, `max_steps=120`, `seed=962`, `cuda:0`.
- Purpose:
  distinguish target geometry failure, FK frame mismatch, and actuator/joint
  follow lag before any more tuning.
- Current status:
  candidate is `READY_FOR_EXPLICIT_RUNTIME_APPROVAL_ONLY`.
  Dataset/RL/RoArm remain blocked.

## Follow-Up: Professor Physical-Reaction Gate Separation

- User clarified that weak physical object movement/reaction is acceptable for
  the professor 10cm/0.72kg push/tap objective unless a stronger 2-3mm transient
  target is explicitly required.
- Corrected the gate design:
  professor physical-reaction evidence and RL contact-gated positive-control are
  now separate statuses.
- Code changes:
  - `roarm_rl/roarm_cube_push_env.py` now has
    `professor_physical_reaction_disp_m=0.0005`,
    `professor_physical_reaction_speed_mps=0.005`, and
    `professor_physical_reaction_z_delta_m=0.0005`.
  - The env logs `cube_tap_professor_physical_reaction_*` separately from
    `cube_tap_reaction_seen_rate` and `cube_tap_success_rate`.
  - `roarm_rl/test_positive_control_cube_tap10cm.py` now reports both
    `rl_contact_gated_positive_control` and
    `professor_physical_reaction_evidence`.
  - `sim_scripts/cube10cm_tap_rl_direct_ik_apply_result_audit.py` now classifies
    the direct-IK run as professor physical evidence PASS when weak motion/speed
    evidence exists without overshoot, even if contact-gated tap fails.
  - `sim_scripts/cube10cm_tap_rl_preflight_policy_gate.py` now has a first-class
    `professor_physical_reaction_evidence` gate.
- Re-ran local-only audits:
  `sim_scripts/cube10cm_tap_rl_direct_ik_apply_result_audit.py` and
  `sim_scripts/cube10cm_tap_rl_preflight_policy_gate.py`.
- Direct-IK audit result:
  max displacement `0.000922590m`, max speed `0.008078601m/s`, overshoot `0.0`,
  `professor_physical_reaction_evidence=PASS`.
- Direct-IK RL/contact result remains failed:
  contact `0.0`, reaction context `0.0`, reaction seen `0.0`, tap success `0.0`.
- Preflight policy result:
  `professor_physical_reaction_evidence=READY_PROFESSOR_EVIDENCE_ONLY`,
  event-label metadata `READY_LOCAL_ONLY`, env wrapper
  `READY_LOCAL_PREFLIGHT_ONLY`.
- Still blocked:
  strict clean action teacher, noisy Tier-B action teacher without explicit
  exception, tiny action dataset dry run, PPO/RL, large dataset, and RoArm.
- Interpretation:
  the earlier failure was real for RL contact-gated positive-control, but it
  should not block professor-facing weak physical reaction evidence. The next
  safe output is a professor evidence/metadata package or local RL blocker debug,
  not training or robot deployment.

## Follow-Up: Direct IK Apply slow240 Runtime Result

- User selected the next actual unblock step:
  `direct_ik_apply_slow240` tiny runtime.
- Purpose:
  test the actuator-follow timing hypothesis without changing geometry or
  action-wrapper knobs.
- Ran one local RTX4090/cuda:0 tiny Isaac Lab runtime only:
  `RoArm-CubeTap10cm-Direct-v0`, `num_envs=2`, `max_steps=120`,
  `seed=962`, `controller_mode=external_closed_loop_direct_apply`,
  `closed_loop_push_steps=240`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime result:
  still `FAIL` for RL contact-gated positive-control.
- Contact/tap evidence:
  contact `0.0`, reaction context `0.0`, reaction seen `0.0`,
  tap success `0.0`, overshoot `0.0`.
- Professor weak physical-reaction evidence stayed positive:
  professor physical reaction seen `1.0`, max displacement `0.000942111m`,
  max speed `0.024404075m/s`.
- Actual contact-gap result:
  best face gap `-0.019191336m`, best shortfall `0.009191336m`;
  this is still outside the `[-0.010,+0.010]m` contact band.
- Controller telemetry:
  target final face gap `0.043000001m`,
  target FK error `0.579830546mm`,
  actual-FK-vs-Isaac-TCP error `0.000880511mm`,
  direct joint-follow final `0.170210123rad`.
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_slow240_result_audit.py`.
- Result audit comparison:
  baseline follow final `0.362854958rad`,
  slow240 follow final `0.170210123rad`,
  follow ratio `0.469085841`.
- Contact-gap comparison:
  baseline best shortfall `0.009533616m`,
  slow240 best shortfall `0.009191336m`,
  improvement only `0.000342280m`.
- Exclusions:
  target enters the contact band in both runs, FK frame remains OK, and wrapper
  path is bypassed in both runs.
- Verdict:
  `FAIL_SLOW240_IMPROVES_FOLLOW_BUT_NOT_CONTACT`.
- Interpretation:
  slow240 supports the actuator-follow/timing hypothesis, but fixed-horizon
  slowdown is insufficient for contact-gated success.
- Still blocked:
  DiffIK action dataset, tiny action dataset dry run, PPO/RL, large dataset, and
  RoArm.
- Next local-only candidate noted by the audit:
  design-only `direct_ik_apply_slow360_candidate_design`; this was not run and
  is not approved as runtime by this audit.

## Follow-Up: Built-In DiffIK Parity and Step-Clipped Target Application

- User challenged the controller mismatch before any contact-gate relaxation.
- Re-checked the intended 10cm controller history:
  the original 10cm transition was supposed to preserve IsaacLab built-in
  `DifferentialIKController` behavior from the 3cm probe, while later
  `external_closed_loop_direct_apply` existed as a wrapper-isolation diagnostic.
- Added local-only code-review/design artifacts:
  - `sim_scripts/cube10cm_tap_rl_builtin_diffik_parity_code_review.py`
  - `sim_scripts/cube10cm_tap_rl_builtin_diffik_parity_candidate_design.py`
- Added default-off built-in DiffIK mode in
  `roarm_rl/test_positive_control_cube_tap10cm.py`:
  `isaac_builtin_diffik_direct_apply`.
- This mode uses IsaacLab `DifferentialIKController` with position command,
  DLS, live PhysX Jacobian, base-frame transform, and TCP tool-proxy offset.
- Ran one local RTX4090/cuda:0 tiny runtime:
  `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_direct_apply`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime result:
  still `FAIL` for strict contact-gated positive-control.
- Full-target built-in DiffIK evidence:
  - contact `0.0`, tap success `0.0`
  - professor seen `1.0`, but professor evidence `FAIL` because
    `terminated_count=2`
  - target path OK: target inside-band max `1.0`, target final face gap
    `0.105999991m`, FK-vs-Isaac TCP error `0.000000000mm`
  - actual face gap max `-0.019483667m`, best shortfall `0.009483667m`
  - direct follow final `0.447358370rad`, worse than external baseline
    `0.362854958rad` and slow240 `0.170210123rad`
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_builtin_diffik_parity_result_audit.py`.
- Audit verdict:
  `BUILTIN_DIFFIK_FULL_TARGET_APPLICATION_STILL_HAS_ACTUATOR_FOLLOW_LAG`.
- Critical interpretation:
  full `joint_pos_des` direct apply is not 3cm parity; 3cm used step-clipped
  target application.

## Follow-Up: Built-In DiffIK Step-Clipped Positive Control

- Added default-off 3cm-style target application mode:
  `isaac_builtin_diffik_step_clipped_direct_apply`.
- The helper computes:
  `raw_delta_arm = joint_pos_des - joint_pos_arm`,
  clips it to `builtin_diffik_step_clip_rad`, then targets
  `joint_pos_arm + clipped_delta_arm`.
- Added telemetry:
  `builtin_diffik_step_clipped_target_apply`,
  `builtin_diffik_step_clip_rate`,
  `builtin_diffik_raw_delta_abs_max_rad`,
  `builtin_diffik_clipped_delta_abs_max_rad`.
- Added and ran local-only candidate design:
  `sim_scripts/cube10cm_tap_rl_builtin_diffik_step_clipped_candidate_design.py`.
- Ran one local RTX4090/cuda:0 tiny runtime:
  `num_envs=2`, `max_steps=120`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `builtin_diffik_step_clip_rad=0.010`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime result:
  still `FAIL` for strict contact-gated positive-control, but professor weak
  evidence is `PASS`.
- Step-clipped metrics:
  - contact `0.0`, tap success `0.0`
  - professor evidence `PASS`
  - closed-loop IK OK `1.0`, target inside-band max `1.0`
  - raw delta final `0.416009426rad`
  - clipped delta final `0.010000000rad`
  - clip-rate final `0.800000012`
  - actual face gap max `-0.019376528m`
  - best shortfall `0.009376528m`
  - follow final `0.008570671rad`
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_builtin_diffik_step_clipped_result_audit.py`.
- Audit verdict:
  `STEP_CLIPPED_DIFFIK_TARGET_APPLICATION_HORIZON_OR_PROGRESS_TOO_SHORT`.
- Interpretation:
  built-in DiffIK compute mismatch and full-target actuator follow lag are
  separated; strict contact/tap success is still `0.0`, so dataset/RL/RoArm
  remain blocked.

## Follow-Up: 3cm Horizon/Cadence Comparison and h580 Runtime

- User asked to compare the 3cm long horizon/step-clipped cadence with 10cm and
  then run one tiny runtime.
- Re-read existing controller-contract audit:
  3cm used built-in `DifferentialIKController`, env action loop bypass,
  580 steps / `6.080s`; 10cm current positive-control harness was 120 steps /
  `1.200s`.
- Added and ran local-only candidate design:
  `sim_scripts/cube10cm_tap_rl_step_clipped_horizon_progress_candidate_design.py`.
- Candidate selected exactly one horizon/progress contract:
  `steps 120 -> 580` and `closed_loop_push_steps 72 -> 580`.
- Preserved:
  controller `isaac_builtin_diffik_step_clipped_direct_apply`,
  `builtin_diffik_step_clip_rad=0.010`, geometry, strict contact gate,
  no dataset/RL/RoArm.
- Design predicted target inside-band dwell would increase from 12 steps
  to 92 steps, runtime from `1.200s` to `5.800s`.
- Ran one local RTX4090/cuda:0 tiny runtime:
  `num_envs=2`, `max_steps=580`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `closed_loop_push_steps=580`,
  `builtin_diffik_step_clip_rad=0.010`.
- Runtime result:
  still `FAIL` for strict contact-gated positive-control.
- h580 metrics:
  - contact `0.0`, tap success `0.0`
  - professor evidence `PASS`
  - max steps `580`, steps executed `580`
  - actual face gap max `-0.019437712m`
  - best shortfall `0.009437712m`
  - final shortfall `0.010676415m`
  - target inside-band max `1.0`
  - follow final `0.010833263rad`
  - `terminated_count=0`, `truncated_count=8`
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_step_clipped_h580_result_audit.py`.
- Critical audit correction:
  h580 did not create one continuous `5.8s` episode. The env still truncated at
  its 10cm `1.2s` episode contract, as shown by `truncated_count=8`.
- Audit verdict:
  `ENV_EPISODE_LENGTH_1P2S_TRUNCATES_H580_HORIZON_TEST`.
- Interpretation:
  do not repeat `--steps 580` alone and do not claim continuous horizon is
  falsified. Next local-only unblock is a default-off episode-length override
  design/patch, then a repeat step-clipped horizon runtime if explicitly
  approved.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.

## Follow-Up: Default-Off Per-Step Reach Trace Patch

- User requested the D193 next unblock:
  patch default-off per-step reach trace before any further tiny runtime.
- Added `--reach_trace_json` to
  `roarm_rl/test_positive_control_cube_tap10cm.py`.
- Default behavior:
  no trace JSON is written unless `--reach_trace_json` is explicitly provided.
- Trace artifact:
  `cube10cm_tap_rl_per_step_reach_trace_v1`, separate from the positive-control
  result JSON.
- Trace is diagnostic telemetry only:
  `dataset_generation=false`, `training=false`, `robot_control=false`,
  `ssh=false`, `b200=false`, `track_a=false`,
  `action_teacher_dataset=false`.
- Trace rows include:
  command target face/lateral/vertical gap and inside-band flag; applied
  joint-target FK face/lateral/vertical gap, inside-band flag, and FK error;
  actual TCP face/lateral/vertical gap and contact proxy; joint target delta,
  direct joint follow, actual joint step; cube displacement/speed; professor
  reaction flags; tap success flags; terminated/truncated flags.
- Added and ran local-only static contract audit:
  `sim_scripts/cube10cm_tap_rl_per_step_reach_trace_patch_contract_audit.py`.
- Audit line 3:
  code ready, `reach_trace_arg=700`, trace writer `591`, applied FK metric `462`,
  row-count metadata `1127`.
- Audit line 4:
  schema contains command target gap, applied joint-target FK gap, actual TCP
  gap, joint follow, cube reaction, done flags; default-off and separate JSON are
  true; `action_teacher_dataset=False`.
- Audit line 5:
  the only designed runtime is the h580 ep608 step-clipped repeat with
  `reach_trace_json` added as the only change.
- Audit line 6:
  patch contract `READY_LOCAL_ONLY`; runtime approval is required; all
  dataset/RL/RoArm gates remain blocked.
- No GPU/runtime was launched in this step.

## Follow-Up: Default-Off Episode-Length Override and Continuous h580

- User explicitly rejected contact-gate relaxation as the next unblock.
- Implemented default-off `--episode_length_s` in
  `roarm_rl/test_positive_control_cube_tap10cm.py`.
- Scope of the patch:
  only the positive-control harness changes `cfg.episode_length_s`, and only
  when the CLI value is positive.
- The env default remains `episode_length_s = 1.2` in
  `roarm_rl/roarm_cube_push_env.py`.
- Added and ran local-only design audit:
  `sim_scripts/cube10cm_tap_rl_episode_length_override_candidate_design.py`.
- Design selected exactly one repeat:
  keep `steps=580`, `closed_loop_push_steps=580`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `builtin_diffik_step_clip_rad=0.010`, geometry unchanged, strict contact gate
  unchanged; change only `episode_length_s 1.2 -> 6.08`.
- Ran one local RTX4090/cuda:0 tiny runtime:
  `num_envs=2`, `max_steps=580`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `closed_loop_push_steps=580`,
  `builtin_diffik_step_clip_rad=0.010`,
  `episode_length_s=6.08`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime contract result:
  `episode_length_s=6.08`, `env_max_episode_length=608`,
  `steps_executed=580`, `terminated_count=0`, `truncated_count=0`.
- This resolves the previous h580 episode-cap blocker.
- Runtime task result still FAILs strict contact-gated positive-control:
  contact `0.0`, tap success `0.0`, professor weak evidence PASS.
- Added and ran:
  `sim_scripts/cube10cm_tap_rl_episode_length_override_result_audit.py`.
- Audit line 3:
  `continuous_horizon_valid=True`, `episode_cap_blocker_resolved=True`, previous
  h580 `truncated_count=8`.
- Audit line 4:
  actual best shortfall `0.009535881m`, no better than step120
  `0.009376528m` or prior h580 `0.009437712m`.
- Audit line 5-6:
  target path still reached the contact band at least once
  (`target_inside_max=1.0`), FK-vs-Isaac TCP error stayed `0.000000000mm`,
  clipped delta was `0.010000000rad`, and follow final was
  `0.010640383rad`.
- Audit verdict:
  `CONTINUOUS_STEP_CLIPPED_DIFFIK_H580_STILL_OUTSIDE_STRICT_CONTACT_BAND`.
- Interpretation:
  continuous horizon is now tested and does not unblock the gate. The next
  local-only work is target/actual contact trajectory and reach-contract audit
  design using existing logs/code first.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.

## Follow-Up: Target/Actual Contact Trajectory Reach-Contract Audit

- User requested the D192 next unblock:
  target/actual contact trajectory and reach-contract audit/design from existing
  logs/code first.
- Added and ran local-only audit:
  `sim_scripts/cube10cm_tap_rl_target_actual_contact_trajectory_reach_contract_audit.py`.
- No GPU runtime, no dataset generation, no PPO/RL, no robot control, no
  SSH/B200, no Track A.
- Current basis:
  continuous h580 ep608 is valid, but strict contact/tap success remains `0.0`
  and professor weak physical evidence remains PASS.
- Contact-gate contract:
  face band `0.010m`, lateral limit `0.065m`, vertical limit `0.070m`.
- ep608 actual lateral/vertical:
  lateral max `0.000231256m`, vertical max `0.020352287m`; both are inside the
  gate.
- ep608 target-vs-actual split:
  command target reaches the contact band at least once
  (`command_target_inside_max=1.0`, command target face-gap min
  `-0.019782793m`) and finishes beyond the face (`0.105999991m`).
- Actual TCP stays outside the along face band:
  actual face-gap max `-0.019535881m`, best shortfall `0.009535881m`, final
  shortfall `0.015875252m`.
- Cross-run stability:
  step120 best shortfall `0.009376528m`, h580 `0.009437712m`, direct telemetry
  `0.009533616m`, slow240 `0.009191336m`; all remain strict contact/tap fail.
- Critical data-contract limitation:
  existing JSON has only min/max/final trace stats, not a full per-step
  timeline.
- Additional built-in DiffIK limitation:
  the current step-clipped built-in path does not record applied joint-target FK
  face-gap; `closed_loop_target_fk_err_mm_mean` is `nan`.
- Verdict:
  `REACH_TRACE_CONTRACT_GAP_IDENTIFIED`.
- Interpretation:
  do not relax the contact gate yet. The current evidence narrows the failure to
  along-axis target/application/actual reach mismatch, but cannot localize the
  time step or stage where the divergence appears.
- Next local unblock:
  patch a default-off per-step reach trace that records command target gap,
  applied joint-target FK gap, actual TCP gap, joint follow, cube reaction, and
  done flags. Run one tiny repeat only after explicit approval.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.

## Follow-Up: Per-Step Reach Trace Repeat Result

- User explicitly approved the D194 tiny reach-trace repeat.
- Ran exactly one local RTX4090/cuda:0 runtime:
  `num_envs=2`, `steps=580`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `closed_loop_push_steps=580`,
  `builtin_diffik_step_clip_rad=0.010`,
  `episode_length_s=6.08`.
- The only added output/change relative to the previous h580 ep608 contract was:
  `--reach_trace_json`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Runtime summary line 1:
  status `FAIL`, local tiny GPU positive-control only.
- Runtime summary line 5:
  contact/tap remain `0.0`, professor weak reaction evidence remains seen,
  overshoot remains `0.0`.
- Runtime summary line 6:
  `terminated_count=0`, `truncated_count=0`, so the continuous h580 episode did
  not regress.
- Runtime summary line 9:
  professor physical reaction evidence `PASS`, RL contact-gated positive-control
  `FAIL`, downstream gates blocked.
- Runtime summary line 10:
  command target final face gap is `0.105999991m`, final target FK error is
  `127.704326062mm`, actual FK-vs-Isaac TCP error is `0.0mm`, final direct
  joint follow is `0.010640383rad`.
- Added and ran local-only posthoc audit:
  `sim_scripts/cube10cm_tap_rl_per_step_reach_trace_result_audit.py`.
- Audit line 2:
  trace artifact is `cube10cm_tap_rl_per_step_reach_trace_v1`,
  `action_teacher_dataset=False`, row count `1160/1160`, steps `0..579`, envs
  `[0,1]`.
- Audit line 4:
  command target entered the contact band for 184 rows / 92 unique steps,
  from step `46` through `137`.
- Audit line 5:
  applied joint-target FK entered 0 rows; best face shortfall is
  `0.004059910m`; final target FK error mean is `127.704326062mm`.
- Audit line 6:
  actual TCP entered 0 rows; best face shortfall is `0.009534182m`; lateral max
  `0.000234434m`, vertical max `0.020371564m`.
- Audit line 7:
  direct joint follow max `0.010850668rad`, actual joint step max
  `0.001486838rad`, cube displacement max `0.000899076m`, professor seen rate
  `1.0`, tap success seen rate `0.0`.
- Audit line 8 verdict:
  `APPLIED_AND_ACTUAL_REACH_NEVER_ENTER_CONTACT_BAND`.
- Interpretation:
  contact gate relaxation is still the wrong unblock. The command target crosses
  the gate, but the applied joint-target FK and actual TCP never do. This points
  to the applied target/reach contract before any Tier-B/noisy exception,
  dataset, PPO/RL, or RoArm path.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.

## Follow-Up: Same-Center vs Same-Face Fixed Pose Audit and x240 Repeat

- User challenged whether the current 10cm fixed pose still inherited the 3cm
  center convention, making the near face too close/far for contact. This was a
  valid suspicion, so I did a local design audit before another runtime.
- Added and ran local-only design audit:
  `sim_scripts/cube10cm_tap_rl_same_center_vs_same_face_pose_audit.py`.
- No dataset generation, no PPO/RL, no robot control, no SSH/B200, no Track A.
- Design audit line 2:
  current same-center pose is 10cm center `x=0.250`, near face `x=0.200`, with
  prior actual best face gap `-0.019534182m` and shortfall `0.009534182m`.
- Design audit line 3:
  preserving the near face of a 3cm cube centered at `x=0.250` would require a
  10cm center `x=0.285`, face `x=0.235`; for the current +x push this is rejected
  because it moves the target face farther from the observed reachable range.
- Design audit line 3 also identifies the direction-aware alternative:
  preserve the 3cm low-x workspace near face, giving 10cm center `x=0.240`, face
  `x=0.190`.
- Design audit line 4:
  observed reach boundary from the previous trace is `touch_center_max_x=0.240465818`;
  selected fixed pose is `fixed_cube_x_m=0.240`, `fixed_cube_y_m=0.000`.
- Ran exactly one approved local RTX4090/cuda:0 x240 tiny runtime:
  `num_envs=2`, `steps=580`, `seed=962`,
  `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`,
  `closed_loop_push_steps=580`, `builtin_diffik_step_clip_rad=0.010`,
  `episode_length_s=6.08`, `fixed_cube_x_m=0.240`, `fixed_cube_y_m=0.000`,
  with `--reach_trace_json`.
- Runtime summary line 3 confirms the only intended runtime geometry change:
  `cube_xy=(0.24,0.0)` while preserving the same step-clipped built-in DiffIK
  h580 ep608 contract.
- Runtime summary line 5:
  contact/tap remain `0.0`, professor weak physical evidence remains seen,
  overshoot remains `0.0`.
- Runtime summary line 8:
  best actual face-gap shortfall is `0.008962901m`, only slightly better than the
  previous x250 shortfall.
- Runtime summary line 9:
  professor physical reaction evidence `PASS`, RL contact-gated positive-control
  `FAIL`, action teacher/PPO/RL/large dataset/RoArm still `BLOCKED`.
- Added and ran local-only result audit:
  `sim_scripts/cube10cm_tap_rl_same_face_pose_result_audit.py`.
- Result audit line 6:
  applied joint-target FK still entered 0 contact rows; shortfall improved only
  `0.004059910 -> 0.003457759m`.
- Result audit line 7:
  actual TCP still entered 0 contact rows; shortfall improved only
  `0.009534182 -> 0.008961143m`.
- Result audit line 8:
  direct joint follow max `0.010857821rad`, actual joint step max
  `0.001492023rad`, cube displacement max `0.000899076m`, professor seen rate
  `1.0`, tap success seen rate `0.0`.
- Result audit line 9 verdict:
  `X240_POSE_IMPROVES_FACE_SHORTFALL_BUT_STILL_NO_CONTACT`.
- Interpretation:
  fixed pose was part of the problem and x240 is the defensible +x pose, but the
  improvement is only about `0.57-0.60mm` and does not reach contact. Do not use
  `x=0.285` for this +x case, because it moves the face farther. Do not relax
  the contact gate from this result.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.
- Next local-only unblock:
  applied joint-target/TCP reach-contract diagnosis: why the command target
  crosses the face band while applied joint-target FK and actual TCP still never
  enter it.

## Follow-Up: Applied Joint-Target/TCP Reach-Contract Diagnosis

- User asked what "locking" meant. Interpretation clarified:
  it means documenting the current evidence and next-step guardrails so a later
  session does not jump to `x=0.285` or contact-gate relaxation before the
  controller/application reach-contract is diagnosed. It does not mean a code
  lock, permanent ban, or refusal to test alternatives after evidence changes.
- Proceeded with the requested next unblock:
  applied joint-target/TCP reach-contract diagnosis.
- Added and ran local-only posthoc/code audit:
  `sim_scripts/cube10cm_tap_rl_applied_target_tcp_reach_contract_diagnosis.py`.
- Inputs:
  existing x250/x240 per-step reach traces, x240 runtime/result summaries, and
  local harness/env code only.
- No GPU runtime, no dataset generation, no PPO/RL, no robot control, no
  SSH/B200, no Track A.
- Diagnosis line 2 code contract:
  target path `test_positive_control_cube_tap10cm.py:327-332`, built-in DiffIK
  compute `:362`, step clip `:366`, `target_full` assignment `:379`, applied FK
  trace `:413`, env direct override `roarm_cube_push_env.py:633-638`,
  `set_joint_position_target` `:753`, post-step actual trace
  `test_positive_control_cube_tap10cm.py:895`, tap contact proxy
  `roarm_cube_push_env.py:1103-1105`.
- Diagnosis line 3:
  x240 command target enters the contact band for 184 rows / 92 unique steps,
  from step `46` through `137`.
- Diagnosis line 4:
  applied joint-target FK enters 0 rows; best face gap is `-0.013457759m`, best
  shortfall `0.003457759m`, and final FK error `127.058100165mm`.
- Diagnosis line 5:
  actual TCP enters 0 rows; best face gap is `-0.018961143m`, best shortfall
  `0.008961143m`.
- Diagnosis line 6:
  at the first command-band step, command-applied miss is `0.003801962m` and
  applied-actual miss is `0.005616973m`; across the full command-inside window,
  the mean command-applied miss grows to `0.014288675m` and mean applied-actual
  miss is `0.005629399m`.
- Diagnosis line 7:
  direct follow stays near the `0.010rad` step clip while actual joint motion is
  only about `0.0014-0.0015rad` per step.
- Diagnosis line 8:
  x240 improves x250 by only `0.000602150m` in applied shortfall and
  `0.000573039m` in actual shortfall.
- Diagnosis line 9 verdict:
  `TARGET_FULL_FK_NEVER_REACHES_FACE_BAND_AND_ACTUAL_TCP_LAGS_TARGET_FULL`.
- Interpretation:
  the command target side is not the immediate blocker; it enters the band. The
  first hard blocker is that the step-clipped `target_full` FK never reaches the
  face band. The second blocker is physical/sim tracking lag from that
  already-insufficient joint target to actual TCP.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.
- Next local-only unblock:
  code-level design for the step-clipped built-in DiffIK target-generation
  contract: raw delta clipping, `target_full` FK progression, Jacobian/tool-proxy
  frame, and whether the Cartesian command schedule outruns the applied
  joint-target FK. Do not relax contact gate or jump to x285 from this evidence.

## Follow-Up: Reach-Contract Root Cause Audit

- User asked for the actual cause after the applied-target/TCP diagnosis.
- Added and ran local-only root-cause audit:
  `sim_scripts/cube10cm_tap_rl_reach_contract_root_cause_audit.py`.
- Inputs:
  existing x240 sanity JSON, applied-target/TCP diagnosis JSON, local harness/env
  code, and local installed IsaacLab source. No web-only or memory-only metric was
  used for the audit values.
- No GPU runtime, no dataset generation, no PPO/RL, no robot control, no
  SSH/B200, no Track A.
- Root-cause summary line 2:
  primary cause is `STEP_CLIPPED_CURRENT_JOINT_BASED_TARGET_GENERATION`.
  Built-in DiffIK raw delta max is `0.427774668rad`, but the harness clips to
  `0.010000000rad`; target delta from actual is capped at `0.010000005rad`;
  final target FK error grows to `127.058100165mm`; final target TCP error before
  command is `0.131981999m`.
- Root-cause summary line 3 code basis:
  actual joint source `test_positive_control_cube_tap10cm.py:358`,
  DiffIK compute `:362`, raw delta `:364`, step clip `:366`,
  clipped arm target `:367`, `target_full` seeded from actual joint position
  `:378`, target assignment `:379`; installed IsaacLab DiffIK returns
  `joint_pos + delta_joint_pos` at source line `174`.
- Root-cause summary line 4 contact effect:
  command target crosses (`184` rows / `92` unique steps), but applied FK and
  actual TCP both enter 0 rows; first command-applied miss is `0.003801962m`,
  and command-inside-window mean command-applied miss is `0.014288675m`.
- Root-cause summary line 5:
  secondary cause is `POSITION_DRIVE_ACTUAL_TCP_LAG`. Direct follow max is
  `0.010857821rad`, actual joint step max `0.001492023rad`, actual/target step
  ratio `0.149202267`, control dt `0.010000000s`.
- Root-cause summary line 6:
  env direct override path is `roarm_cube_push_env.py:633-638`,
  `set_joint_position_target` is `:753`, control cadence is `decimation=2` and
  `dt=1/200`, arm actuator settings are stiffness/damping/effort/velocity
  `80/4/2.5/3.14`, and IsaacLab implicit actuator PD is handled by simulation.
- Root-cause summary line 7:
  command target geometry, x285, contact gate, and cube mass are not primary
  causes for the current failure because the command crosses but clipped target
  FK and actual TCP do not.
- Root-cause summary line 8:
  exact effort/stiffness/damping split needs torque or drive telemetry; worst
  joint contribution needs per-joint trace.
- Interpretation:
  the immediate reason is not "TCP target was wrong" and not "cube is too heavy."
  The command is asking for contact, but the one-step clipped joint target is
  generated from the current actual joint position and never puts the FK tool into
  the contact band. The physical/sim TCP then trails that insufficient target by
  another actuator-follow gap.
- Next local-only unblock:
  design a default-off target-generation contract candidate that separates
  Cartesian schedule, raw-delta clip, target base (`actual joint_pos` versus prior
  target), and actuator-follow/per-joint telemetry before any new tiny runtime.
- Still blocked:
  strict contact-gated positive-control, DiffIK action dataset, tiny action
  dataset dry run, PPO/RL, large dataset, and RoArm.
