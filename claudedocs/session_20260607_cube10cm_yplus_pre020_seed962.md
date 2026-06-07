# 2026-06-07 cube10cm y+ pre020 seed962

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window +
  quality-tier branch.
- Not Track A, not grasp/dataset/training, not B200/SSH.
- Objective order: reaction/contact/no-posewrite/no-overshoot first; quality tier
  second; final 1cm relocation secondary only if explicitly requested.

## Pre-Runtime Verification

- Read `CLAUDE.md` and followed Current-State Protocol.
- Verified `START_HERE.md:9-13,1238-1255`, DECISIONS D153-D155,
  EXPERIMENT_LEDGER row 197, session log `697-763`,
  `sim_scripts/cube10cm_yplus_precontact_candidate_audit.py:1-8,58-88,152-225`,
  candidate summary `1-8`, and MEMORY recent-session line 74.
- `git status --short --untracked-files=all --branch` was clean at session start:
  `## master...origin/master`.

## Guards

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS:
  primary objective `reaction_contact_no_posewrite_no_overshoot`, final 1cm
  default `NO`.
- `python sim_scripts/cube10cm_next_research_step_audit.py` exit 0 but critical:
  `teacher_quality_ready=False`, clip `1.0`, final TCP error `0.062821m`.
- `python sim_scripts/cube10cm_yplus_precontact_candidate_audit.py` PASS:
  only `precontact_clearance_m 0.010 -> 0.020`; no GPU/data in the audit.
- `python -m py_compile ...` PASS for the requested script set.
- `git diff --check` PASS.

## Runtime

Ran exactly one approved local IsaacLab tiny runtime:

- seed `962`
- `num_envs=16`, `episodes=1`
- fixed cube `x=0.295`, `y=-0.044`
- fixed push dir y+
- lateral `-0.020`
- `xneg_tcp_center_height_offset_m=0.050`
- only changed runtime variable: `precontact_clearance_m=0.020`

No B200/SSH/pull, no dataset generation, no PPO/RL/VLA, no Track A.

## Runtime Results

Source: `diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json`.

- Reaction event rate: `1.0`.
- Measured contact seen rate: `1.0`.
- Contact stop seen rate: `1.0`.
- Posewrite calls: `0`.
- Contact overshoot rate: `0.0`.
- Controlled push rate: `0.5625`.
- `disp_ge_gate_rate`: `0.5625`.
- `max_disp_along_push_mean_m`: `0.002923812717`.
- `max_cube_z_delta_mean_m`: `0.007974284701`.
- `max_tip_angle_mean_deg`: `9.205449760`.
- Low-motion rate: `1.0`.
- DiffIK clip mean: `1.0`.
- Final TCP target error mean: `0.051811996m`.

## Post-Runtime Audits

- Trace diagnostic PASSed mechanism/row checks but reports
  `dataset_ready=NO`, `JOINT_STEP_CLIPPING_DOMINANT`, and
  `ACTUATOR_TARGET_TRACKING_LAG`.
- Reaction gate PASSed reaction/contact/no-posewrite/no-overshoot:
  reaction/contact `1.0/1.0`, overshoot `0.0`, tap gate `1.0`,
  `teacher_quality_ready=false`.
- Reaction-window contract accepted 16/16 windows, zero rejected:
  2 Tier B + 14 Tier C, follow p95/cap p95 `1.160505840`, clean teacher false.

## Comparison To y+ Seed958/960/961

Same early-contact audit definition, seed962 included:

| seed | changed variable | y+ tiers | anchor minus push start mean | pre24 disp mean m | pre24 tip mean deg | max/pre-all disp mean m |
|---|---|---:|---:|---:|---:|---:|
| 958 | baseline pre010 | 16 C | `-5.625000` | `0.010906445` | `11.900443137` | `0.025062508` |
| 960 | cap050 | 16 C | `-65.937500` | `0.011423650` | `12.545439422` | `0.025335398` |
| 961 | stiffness600 | 16 C | `-55.625000` | `0.014568937` | `15.546741426` | `0.022021953` |
| 962 | pre020 | 2 B + 14 C | `27.187500` | `0.005104796` | `5.079945311` | `0.008559460` |

Interpretation:

- `precontact=0.020` did reduce y+ pre-anchor/pre24 reaction and delayed the
  anchor to after push start.
- It did not solve y+ quality: clip remains `1.0`, follow p95/cap remains above
  the Tier B threshold for most windows, and clean teacher is false.
- It also weakened reaction strength: controlled push fell to `0.5625`, max
  displacement and tip are much lower than seed958/960/961.

## Seed962-Inclusive Matrix

`cube10cm_reaction_window_tier_matrix_with_seed962.json`:

- 144 candidate windows
- 128 accepted
- acceptance `0.888888889`
- 55 Tier B
- 73 Tier C
- 16 Rejected
- zero Tier A
- y+ direction: 67/67 accepted, 2 Tier B + 65 Tier C
- `ready_for_1024_or_data=false`

## Decision

- Primary contract PASS: reaction/contact/no-posewrite/no-overshoot.
- Quality/data readiness FAIL: no Tier A, clip/follow blockers remain.
- Precontact 0.020 is a useful diagnostic, not a fix. It reduced early/pre-anchor
  y+ reaction but moved the failure toward weak late contact.

## Next Research Step

Do local-only timing/contact-strength separation before any further runtime:

1. Compare seed958/960/961/962 by anchor timing, phase alpha, pre24 reaction,
   max reaction strength, follow p95/cap, and contact-stop phase.
2. Decide whether the next tiny variable should target timing/path shape rather
   than simply increasing precontact.
3. Do not run another GPU screen until that audit proposes exactly one variable
   and the user explicitly approves it.

Blocked:

- 1024/10240
- dataset generation
- PPO/RL
- VLA
- Track A
- B200/SSH
- blind precontact/lateral/height/actuator/DLS/cap sweeps

## Follow-Up Local Audit

Added and ran `sim_scripts/cube10cm_yplus_pre020_failure_shift_audit.py`.

Purpose:

- Compare seed958/960/961/962 under the same per-window timing/pre24 definitions.
- Decide whether seed962 should lead to another precontact runtime or a different
  local research question.

Verification:

- `python -m py_compile sim_scripts/cube10cm_yplus_pre020_failure_shift_audit.py`
  PASS.
- `python sim_scripts/cube10cm_yplus_pre020_failure_shift_audit.py` PASS.
- The audit is local/posthoc only: no GPU, no IsaacLab runtime, no dataset, no
  training, no SSH, no trace mutation.

Key output:

- `pre020_reduces_preanchor_reaction=True`.
- `pre020_weakens_reaction_strength=True`.
- `quality_still_blocked=True`.
- seed962 pre24 displacement/tip vs seed958/960/961 mean:
  `0.415034926` / `0.381066167`.
- seed962 max displacement/tip vs seed958/960/961 mean:
  `0.661469914` / `0.376103186`.

Updated conclusion:

- The next research task was performed locally and confirms that seed962 is a
  failure-shift, not a fix.
- Do not request another GPU yet. First design one path/timing/contact-strength
  candidate with an explicit predicted tradeoff.

## Follow-Up Contact-Strength Candidate Audit

Added and ran `sim_scripts/cube10cm_yplus_contact_strength_candidate_audit.py`.

Purpose:

- Use the existing seed958/960/961/962 y+ CSV/JSON artifacts only.
- Separate path/timing alternatives from a direct contact-strength retention
  hypothesis.
- Select one candidate before any further GPU approval request.

Verification:

- `python -m py_compile sim_scripts/cube10cm_yplus_contact_strength_candidate_audit.py`
  PASS.
- `python sim_scripts/cube10cm_yplus_contact_strength_candidate_audit.py` PASS.
- The audit is local/posthoc/config only: no GPU, no IsaacLab runtime, no dataset,
  no training, no SSH, no trace mutation.

Key output:

- seed962 max 1mm gate: `1.000000000`.
- seed962 final 1mm gate: `0.562500000`.
- seed962 retention mean: `0.462406074`.
- seed958/960/961 retention mean: `0.737681608`.
- retention ratio: `0.626836930`.
- seed962 max displacement mean: `0.002923813m`.
- seed958/960/961 max displacement mean: `0.004420175m`.
- max displacement ratio: `0.661469914`.
- contact-stop step-rate mean: seed962 `0.366185905`, previous y+ mean
  `0.544604709`.

Corrected interpretation:

- User correctly challenged the premise that final 1mm retention matters for the
  professor tap/reaction objective.
- Re-checked the contract: final retention is not primary. The primary objective
  remains reaction/contact/no-posewrite/no-overshoot.
- Updated and reran the audit. It now records `final_retention_primary=NO`,
  `final_retention_primary_objective=False`, and
  `selected_next_candidate=NONE_FROM_FINAL_RETENTION_ALONE`.
- `contact_stop_disp_m 0.001 -> 0.002` is downgraded to an optional diagnostic
  only if stronger transient 2-3mm push is explicitly requested. It is not a
  next-GPU target from final retention alone.
- GPU runtime was not run.

Rejected first for this step:

- `approach_steps 220 -> 200`: too weak for the observed stop-retention failure.
- `push_steps 90 -> 70`: raises per-step target demand while clip/follow is already
  blocked.
- `push_through_m 0.010 -> 0.020`: plausible path-strength candidate, but less
  direct than the max-vs-final retention evidence.
- `contact_stop_joint_step_scale`: actuator/step-scale mixing.
- precontact/lateral/height/DLS/cap changes: breaks current one-variable separation.

Updated conclusion after correction:

- Do not treat post-push final 1mm retention as a task failure.
- Keep logging cube displacement because it proves physical contact/reaction,
  catches no-motion misses, catches overshoot/knockdown, and anchors
  reaction-window quality labels.
- If later approved for a stronger tap, judge by reaction/contact/no-posewrite/
  no-overshoot first, optional max transient 1/2/3mm second, then quality tier.
- Do not claim data/1024 readiness from this local audit.

## Follow-Up Transient Tap-Strength Audit

Added and ran `sim_scripts/cube10cm_yplus_transient_tap_strength_audit.py`.

Purpose:

- Reframe the next y+ decision around transient tap tiers.
- Exclude final cube position as a success gate.
- Compare fixed y+ seed958/960/961/962 with contact, reaction, overshoot, max
  displacement thresholds, tip/z/speed, and quality-tier metadata.

Verification:

- `python -m py_compile sim_scripts/cube10cm_yplus_transient_tap_strength_audit.py`
  PASS.
- `python sim_scripts/cube10cm_yplus_transient_tap_strength_audit.py` PASS.
- The audit is local/posthoc only: no GPU, no IsaacLab runtime, no dataset, no
  training, no SSH, no trace mutation.

Key output:

- Summary line 1: `final_position_gate=NO`.
- Summary line 2: seed962 primary event PASS evidence:
  contact `1.000000000`, reaction `1.000000000`, overshoot `0.000000000`, max
  1mm `1.000000000`.
- Summary line 3: seed962 max 2mm `0.812500000`, max 3mm `0.500000000`, max 5mm
  `0.000000000`, max displacement mean `0.002923813m`.
- Summary line 4: previous y+ seed958/960/961 mean max 2mm `1.000000000`, max
  3mm `0.979166667`, max displacement mean `0.004420175m`.
- Summary line 5: seed962 is less aggressive than the previous y+ pocket:
  max displacement ratio `0.661469914`, tip ratio `0.376103186`, z ratio
  `0.492015176`.
- Summary line 6: `primary_1mm_tap_event_pass=True`,
  `two_mm_transient_majority=True`, `three_mm_transient_not_reliable=True`,
  `quality_still_blocks_data_readiness=True`.

Next order:

1. Do not use final 1cm or final retention.
2. If the intended professor task is a 1-2mm tap/reaction, stop y+ contact-geometry
   tuning; seed962 is acceptable at the event level.
3. Keep quality-tier metadata separate: seed962 is still not data-ready because
   quality remains blocked.
4. If a 3mm transient tap is explicitly required, define that target first and
   then propose exactly one local candidate without mixing knobs.

## Dataset/RL/Robot Readiness Gate

Added and ran `sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py`.

Purpose:

- Answer the requested progression: dataset -> IsaacLab dataset/RL -> RoArm-M3-Pro.
- Separate event-label readiness from action-teacher dataset readiness.
- Keep final 1cm/final retention out of the primary gate.

Verification:

- `python -m py_compile sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py`
  PASS.
- `python sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py` PASS.
- The audit is local/readiness only: no GPU, no dataset generation, no training,
  no robot control, no SSH, no trace mutation.

Key output:

- Event gate ready:
  `primary_event_ready=True`, `one_two_mm_objective_ready=True`,
  contact/reaction/overshoot/max1mm/max2mm = `1.0/1.0/0.0/1.0/0.8125`.
- Quality gate blocks action-teacher dataset:
  `action_teacher_dataset_ready=False`, `clean_teacher=False`, 2 Tier B + 14
  Tier C, clip mean `1.0`, follow p95/cap `1.160505840`.
- Pipeline gates:
  `event_label_dataset_ready=True`,
  `large_isaaclab_dataset_ready=False`,
  `isaaclab_rl_ready=False`,
  `roarm_m3_pro_deploy_ready=False`.
- The existing `cube3cm_push_diffik_build_dataset.py` is not valid as-is for this
  10cm tap branch because it filters final controlled/success markers.
- Existing `roarm_rl/roarm_cube_push_env.py` is explicitly a 3cm cube push task,
  so 10cm/0.72kg tap RL env/random sanity is not validated.

## Event-Label Manifest

Added and ran `sim_scripts/cube10cm_event_label_dataset_manifest.py`.

Allowed artifact:

- Local schema/label manifest only.
- Not action-teacher dataset.
- Not LeRobot/RLDS.
- Not training data.
- Not robot control.

Manifest summary:

- 16 reaction-window events.
- contact `16`.
- reaction `16`.
- overshoot `0`.
- window-level transient counts: 1mm `16`, 2mm `13`, 3mm `7`.
- quality tier counts: 2 Tier B + 14 Tier C.
- schema explicitly excludes final 1cm relocation, final 1mm retention, and
  post-push final position.

Updated conclusion:

- The only completed dataset step is a local event-label manifest.
- Do not generate a large IsaacLab dataset, do not run PPO/RL, and do not deploy to
  RoArm-M3-Pro from this state.
- Next safe implementation work is either:
  1. design a 10cm tap-specific dataset builder using reaction-window labels, or
  2. design a 10cm/0.72kg tap RL env preflight/random sanity gate.

## DiffIK Action-Dataset Blocker Audit

Added and ran `sim_scripts/cube10cm_diffik_action_dataset_blocker_audit.py`.

Purpose:

- Answer the direct question: whether IsaacLab built-in Differential IK data can
  now become an action-teacher dataset.
- Keep the professor branch sequence visible: dataset -> IsaacLab RL ->
  RoArm-M3-Pro.
- Separate the ready local event-label manifest from blocked action-teacher
  dataset, large dataset, RL training, and robot deployment.

Verification:

- `python -m py_compile sim_scripts/cube10cm_diffik_action_dataset_blocker_audit.py sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py sim_scripts/cube10cm_event_label_dataset_manifest.py`
  PASS.
- `python sim_scripts/cube10cm_dataset_rl_robot_readiness_audit.py` PASS.
- `python sim_scripts/cube10cm_event_label_dataset_manifest.py` PASS.
- `python sim_scripts/cube10cm_diffik_action_dataset_blocker_audit.py` PASS.
- `git diff --check` PASS.
- No GPU runtime, dataset generation, training, robot control, SSH, B200, Track A,
  or VLA work was run.

Key blocker audit output:

- `event_label_dataset=READY_LOCAL_ONLY`: 16 events, contact/reaction/overshoot
  `1.0/1.0/0.0`, window-level 1/2/3mm counts `16/13/7`.
- `diffik_action_teacher_dataset=BLOCKED`: clean teacher false, 2 Tier B + 14
  Tier C, clip mean `1.0`, follow p95/cap `1.160505840`, final TCP error
  `0.051811996m`.
- Trace quality modes remain `LINK5_BODY_TARGET_NOT_REACHED`,
  `JOINT_STEP_CLIPPING_DOMINANT`, and `ACTUATOR_TARGET_TRACKING_LAG`.
- Code conflicts are now explicitly recorded:
  - old dataset builder final controlled/low-motion/success filters at
    `sim_scripts/cube3cm_push_diffik_build_dataset.py:190,196,199,429`.
  - existing RL env 3cm/20g relocation assumptions at
    `roarm_rl/roarm_cube_push_env.py:1,31,72,100,817`.
- Pipeline remains blocked:
  `large_isaaclab_dataset=BLOCKED`, `isaaclab_rl=BLOCKED`,
  `roarm_m3_pro=BLOCKED`.

Resolution order:

1. Keep 1-2mm tap/reaction objective; do not drift back to final 1cm/final
   retention.
2. Use the event-label manifest as local-only evidence, not action data.
3. Next local branch work is a 10cm tap-specific dataset-builder preflight with
   no final-success filter.
4. Resolve or explicitly gate noisy DiffIK teacher quality before any action
   teacher training dataset.
5. Validate a 10cm/0.72kg tap RL env random-sanity gate before RL training.
6. Only after a validated policy and safety/replay gate should RoArm-M3-Pro be
   considered.

## 2026-06-08 Tap Dataset-Builder Preflight And Teacher Policy Gate

User requested step-by-step unblocking along the professor branch sequence:
dataset -> IsaacLab RL -> RoArm-M3-Pro.

Implemented local unblock step:

- Added `sim_scripts/cube10cm_tap_reaction_dataset_builder_preflight.py`.
- This is local preflight only. It writes a tiny preview artifact, not a large
  dataset, not an action-teacher dataset, not LeRobot/RLDS, and not training data.
- It uses the existing event-label manifest and blocker audit.

Verification:

- `python -m py_compile sim_scripts/cube10cm_tap_reaction_dataset_builder_preflight.py sim_scripts/cube10cm_diffik_action_dataset_blocker_audit.py sim_scripts/cube10cm_event_label_dataset_manifest.py`
  PASS.
- `python sim_scripts/cube10cm_tap_reaction_dataset_builder_preflight.py` PASS.
- `wc -l .../cube10cm_tap_reaction_dataset_builder_preflight_preview.jsonl`
  returned `16`.
- `rg -n "final_1cm|final_1mm|post_push_final|success_marker|controlled_push|low_motion|final_disp|target_xy_dist|cube_success_disp" .../cube10cm_tap_reaction_dataset_builder_preflight_preview.jsonl`
  returned no matches.

Preflight result:

- Local event-label builder preflight is `READY_LOCAL_ONLY`.
- Preview rows: 16.
- contact `16`, reaction `16`, overshoot `0`.
- transient 1/2/3mm counts: `16/13/7`.
- quality tiers: 2 Tier B + 14 Tier C.
- forbidden gate check passed: `forbidden_present=[]`,
  `uses_final_success_filter=NO`, `uses_final_1cm_or_retention=NO`.
- Legacy final-success filter is locally bypassed for this event-label path.

Implemented next policy gate:

- Added `sim_scripts/cube10cm_diffik_teacher_quality_policy_gate.py`.
- This local audit separates event-label readiness from action-teacher policy.

Verification:

- `python -m py_compile sim_scripts/cube10cm_diffik_teacher_quality_policy_gate.py sim_scripts/cube10cm_tap_reaction_dataset_builder_preflight.py`
  PASS.
- `python sim_scripts/cube10cm_diffik_teacher_quality_policy_gate.py` PASS.

Policy gate result:

- Event-label path: `READY_LOCAL_ONLY`, 16 rows.
- Strict clean DiffIK action teacher: `BLOCKED`.
  Evidence: clean teacher false, Tier A/B/C = `0/2/14`, clip mean `1.0`,
  follow p95/cap `1.160505840`, final TCP error `0.051811996m`.
- Tier-B-only action teacher: `BLOCKED_INSUFFICIENT_ROWS`, only 2 usable rows out
  of 16 accepted windows.
- Tier-B/C noisy action teacher: `REQUIRES_EXPLICIT_POLICY_EXCEPTION`.
- Default action-teacher dataset policy: `BLOCKED_DEFAULT_POLICY`.
- Quality policy is not resolved for training.

Updated conclusion:

1. The local event-label dataset-builder preflight is now unblocked.
2. DifferentialIK action-teacher dataset is still blocked.
3. Large IsaacLab dataset, IsaacLab RL, and RoArm-M3-Pro are still blocked.
4. Next branch step must be one of:
   - improve/retest teacher quality before action dataset, or
   - explicitly record a noisy-teacher policy exception, then do only a tiny
     audited action-dataset dry run.
5. No GPU runtime, IsaacLab data generation, training, robot control, SSH, B200,
   Track A, VLA, 1024, or 10240 work was run.

## Teacher Quality Improvement/Revalidation Path

User selected priority: improve/retest teacher quality first, not noisy Tier B/C
exception first.

Implemented local revalidation:

- Added `sim_scripts/cube10cm_teacher_quality_revalidation_audit.py`.
- It reads existing seed962 reaction-window audit, trace CSV, and summary JSON.
- It sweeps anchor-relative action-row policies:
  `[-24,+48]`, `[-24,0]`, `[-8,+8]`, `[0,+16]`, `[0,+24]`,
  `[-8,+16]`, `[-4,+12]`.
- It performs no GPU, no IsaacLab runtime, no dataset generation, no training, no
  robot control, no SSH, and no trace mutation.

Verification:

- `python -m py_compile sim_scripts/cube10cm_teacher_quality_revalidation_audit.py`
  PASS.
- `python sim_scripts/cube10cm_teacher_quality_revalidation_audit.py` PASS.

Revalidation result:

- Official reaction window `[-24,+48]`:
  accepted `16/16`, tiers `2B+14C`, clip mean `1.0`, follow p95/cap
  `1.140652384`.
- Best trimmed action-row policy:
  `contact_to_p16` `[0,+16]`.
- `contact_to_p16` result:
  accepted `16/16`, tiers `16B+0C`, clip mean `1.0`, follow p95/cap
  `0.251552037`.
- Interpretation:
  - Tier-C follow-lag quality is partly a row-window definition issue.
  - Strict clean teacher is still not solved because strict clean count is `0`
    and clip remains `1.0`.
  - Remaining blocker is likely command clipping/control tracking, not only
    window definition.

Implemented tiny Tier-B action dry-run preview:

- Added `sim_scripts/cube10cm_tierb_action_dryrun_preview.py`.
- It selects `contact_to_p16` rows from the existing sparse trace.
- It writes only a local JSONL preview, not a training dataset and not a large
  dataset.

Verification:

- `python -m py_compile sim_scripts/cube10cm_tierb_action_dryrun_preview.py sim_scripts/cube10cm_teacher_quality_revalidation_audit.py`
  PASS.
- First execution revealed the readiness criterion incorrectly expected dense
  17 rows/env; trace uses stride 4, so correct sparse expectation is 4-5 rows/env.
- Updated the dry-run readiness criterion to require all envs present, at least 4
  sparse rows/env, and no forbidden fields.
- `python sim_scripts/cube10cm_tierb_action_dryrun_preview.py` PASS.
- `wc -l .../cube10cm_tierb_action_dryrun_preview_rows.jsonl` returned `66`.
- `rg -n "final_1cm|final_1mm|post_push_final|success_marker|controlled_push|low_motion|final_disp|target_xy_dist|cube_success_disp" .../cube10cm_tierb_action_dryrun_preview_rows.jsonl`
  returned no matches.

Dry-run preview result:

- Selected policy: `contact_to_p16` `[0,+16]`.
- Quality tier: Tier B (`B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH`).
- Strict clean teacher: NO.
- Clip high: YES.
- Events: 16.
- Rows: 66 sparse trace rows.
- Rows/env: min 4, max 5.
- Forbidden final/success fields: none.
- Action abs mean/p95/max: `0.005356158` / `0.007000000` / `0.007000000` rad.
- Clip-any rows: 66.
- Status: `tierb_action_dryrun_preview=READY_LOCAL_ONLY`.
- Actual action-teacher dataset: `NOT_BUILT`.
- Large IsaacLab dataset/RL/RoArm remain `BLOCKED`.

Updated conclusion:

1. We did not need a noisy Tier B/C exception to remove the Tier-C follow-lag
   problem; contact-row trimming revalidates seed962 as Tier B.
2. Clean teacher is still not achieved because every dry-run row is clipped.
3. This supports only a tiny Tier-B dry-run preview, not actual training data.
4. The next default research step is a single, explicit, local GPU candidate for
   clipping reduction while preserving `contact_to_p16`, or stop at metadata-only
   evidence. No GPU was run in this step.
