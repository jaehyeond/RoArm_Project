# 2026-06-06 - Professor Cube10cm Reaction-Window Contract

## Scope

- Active branch: professor 10cm / 0.72kg cube push/tap DiffIK reaction branch.
- Not Track A, not grasp, not PPO/RL/VLA, not 1024/10240 scale-up.
- B200 was not used. No SSH, no pull, no `.ssh` copy.
- Initial contract/tier work was local/posthoc only on existing logs; later seed958
  and seed959 were explicitly approved tiny local IsaacLab direction-coverage
  screens.

## Why This Step

The clarified objective is tap/reaction/contact, not final 1cm relocation. Whole
rollouts mix approach, contact, stop/freeze, and post-stop actuator behavior.
The new data-unit contract therefore uses short reaction windows around contact,
not final displacement.

## Code Changes

- Added `sim_scripts/cube10cm_reaction_window_contract_audit.py`.
  - Lines 1-5 define it as local posthoc only.
  - It reads existing trace/summary logs, may write a window CSV audit artifact,
    and does not run IsaacLab, train, generate new rollouts, or create a final
    training dataset.
  - It anchors each env on `first_contact_step` first, then fallback contact
    markers, and cuts `pre_contact_steps=24`, `post_contact_steps=48` by default.
  - It requires contact evidence, reaction signal, no posewrite/training/attach,
    and no overshoot.
  - Reaction signal is any of max displacement, z delta, speed, or contact-gated
    tip angle. Tilt alone is not enough without contact evidence.
  - Clip/follow are metadata plus a separate clean-DiffIK teacher diagnostic.
- Updated `sim_scripts/cube10cm_tap_objective_contract_audit.py` wording:
  `explicit_1cm_override_allowed` became
  `explicit_final_relocation_override_available`.

## Cross-Checks

All cross-checks used existing trace logs in
`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/`.

- seed957:
  - Output: `cube10cm_reaction_window_seed957_audit.json`
  - Accepted windows: `16/16`
  - Accepted rows: `294`
  - `reaction_window_contract_pass=true`
  - `clean_diffik_teacher_window_ready=false`
  - env0 anchor: step `203`
  - env0 signals: contact evidence true, displacement reaction true, speed
    reaction true, contact-gated tip reaction true, no overshoot.
- seed949:
  - Output: `cube10cm_reaction_window_seed949_audit.json`
  - Accepted windows: `16/16`
  - Accepted rows: `290`
  - Reaction-window contract PASS.
- seed950:
  - Output: `cube10cm_reaction_window_seed950_audit.json`
  - Accepted windows: `16/16`
  - Accepted rows: `292`
  - Reaction-window contract PASS.
  - Clean-DiffIK teacher diagnostic remains false due high window clip/follow.
- seed948 negative control:
  - Output: `cube10cm_reaction_window_seed948_audit.json`
  - Accepted windows: `0/16`
  - Accepted rows: `0`
  - Rejected because `missing_contact_anchor`.
  - This is the important false-positive check: large tip/z/speed without contact
    should not become a success label.

## Guards

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit, still
  blocks dataset/PPO/VLA/TrackA/1024_10k and reports
  `NARROW_ACTUATOR_IK_TRACKING_CLEANUP_INSIDE_WORKING_TAP_GEOMETRY`.
- `python -m py_compile ... cube10cm_reaction_window_contract_audit.py` PASS.
- `git diff --check` PASS before documentation updates.

## Decision

Reaction-window labeling is the right next local path. It translates the professor
tap/reaction objective into data units without reverting to final 1cm relocation.
It does not mean full data generation is ready. The open decision is whether to
accept reaction-window traces with quality metadata, or keep requiring clean
DiffIK teacher windows.

## Quality-Tier Follow-Up

User selected the middle path: clean DiffIK teacher should be a quality tier, not
the absolute tap/reaction filter. The script was updated to v2 quality tiers:

- `A_CLEAN_DIFFIK_TEACHER`: valid reaction window and clip/follow pass clean
  teacher thresholds.
- `B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH`: valid reaction window with follow under
  threshold but clip above threshold.
- `C_REACTION_VALID_FOLLOW_LAG`: valid reaction window with follow p95/cap above
  threshold.
- `REJECTED`: not a valid reaction window.

Existing-log tier distribution:

- seed957: 16/16 valid windows, all Tier B. Clip mean `0.673245614`, follow
  p95/cap p95 `0.776854157`.
- seed949: 16/16 valid windows, all Tier B.
- seed950: 16/16 valid windows, 10 Tier B + 6 Tier C. Follow p95/cap p95
  `1.142495019`.
- seed948: 0/16 valid windows, 16 Rejected due missing contact anchor.

Interpretation: the current branch can continue as reaction-window + quality
metadata. Tier A can become clean BC teacher data. Tier B/C should not silently
be treated as clean teacher, but they preserve useful contact/reaction evidence
for analysis, ablation, and later filtered dataset variants.

## Next Step

Use the reaction-window contract on existing traces and any future explicitly
approved tiny traces. Do not start 1024/10240/data/RL/VLA/Track A from the current
seed set. The next local research task is tier-distribution reporting per
direction/workspace.

## Tier Matrix Follow-Up

Added `sim_scripts/cube10cm_reaction_window_tier_matrix.py` as the next local
research report. It reads existing reaction-window audit JSONs and their trace
CSVs only; it does not run IsaacLab, generate rollouts, train, or build a final
dataset.

Why it was needed:

- Direction-only tier counts can be misleading when different contact geometry or
  actuator/IK settings are mixed.
- The matrix joins each per-window tier back to trace-level `push_dx/push_dy`,
  local cube start position, workspace bin, and audit/config name.
- This keeps "x- geometry failed" separate from "x- direction is impossible".

Generated outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_reaction_window_tier_matrix_existing_seeds.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_reaction_window_tier_matrix_existing_seeds.csv`

Existing seed948/949/950/957 matrix:

- Overall: 4 audits, 64 candidate windows, 48 accepted, acceptance `0.75`.
- Quality tiers: 42 Tier B, 6 Tier C, 16 Rejected, 0 Tier A.
- x+: 20/20 accepted, all Tier B.
- x- aggregate: 21/37 accepted, but this is config-mixed.
  - seed948 x-: 0/16, all Rejected due missing contact anchor.
  - seed949 x-: 16/16, all Tier B.
  - seed950 x-: 5/5, all Tier B.
- y+: 3/3 accepted, all Tier C.
- y-: 4/4 accepted, 1 Tier B + 3 Tier C.

Interpretation:

- This is not ready for 1024/10240/data. The matrix explicitly reports
  `ready_for_1024_or_data=false`.
- No Tier A exists in the existing matrix, but that is not an absolute rejection
  of reaction-valid data because Tier A is a quality tier, not the primary label.
- y+/y- accepted windows are too few and mostly Tier C; they are not robust
  direction evidence yet.
- x- is not globally dead; the failed seed948 geometry is rejected, while the
  height050 x- evidence is valid Tier B.

Verification:

- `python sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS.
- `python -m py_compile sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS.
- `git diff --check` PASS after the script change.

## Fixed Y+ Direction-Coverage Screen

User explicitly approved one tiny local IsaacLab direction-coverage runtime,
relaxing the previous "actuator/IK parameter only" restriction for this objective.
The run was not data generation, not 1024/10240, not PPO/RL/VLA, and not Track A.

Command intent:

- Fixed direction: y+ only.
- Geometry: seed950-like goodxy `x=0.295,y=-0.044`, lateral `-0.020`.
- Config: wrapper tap defaults, baseline cap/stiffness/effort, trace all envs.
- Seed: 958, `num_envs=16`, one episode.

Runtime summary:

- controlled/contact/reaction `1.0`.
- max displacement mean `0.004254133m`.
- max z delta mean `0.016543288m`.
- max speed mean `0.130731522m/s`.
- no posewrite, no overshoot.
- low-motion remains `1.0`.

Reaction gate:

- PASS on reaction/contact/no-posewrite/no-overshoot.
- tap gate `1.0`.
- final relocation not used (`final_relocation_gate_rate=None`).
- teacher false: DiffIK clip `1.0`, final TCP error `0.047422359m`.

Trace diagnostic:

- clip_any `1.0`.
- dominant likely modes: link5 target not reached, joint-step clipping, actuator
  target tracking lag.
- pre-stop clip_any `1.0`.
- pre-stop worst follow joint 2 p95 `0.041796684rad`.

Reaction-window result:

- accepted windows `16/16`, row count `291`.
- clean teacher false.
- quality tiers: 16 Tier C, 0 Tier A, 0 Tier B, 0 Rejected.
- follow p95/cap p95 `1.151057652`.

Updated tier matrix:

- Inputs now include seed948/949/950/957/958.
- Overall: 80 candidate windows, 64 accepted, acceptance `0.8`.
- Quality tiers: 42 Tier B, 22 Tier C, 16 Rejected, 0 Tier A.
- y+: 19/19 accepted, all Tier C.
- y- remains under-sampled: 4/4 accepted, 1 Tier B + 3 Tier C.
- readiness remains `ready_for_1024_or_data=false`.

Interpretation:

- y+ is no longer just a 3-sample lucky-contact observation. In this goodxy/lateral
  pocket, fixed y+ repeatedly produces real contact/reaction windows.
- y+ is still not clean-teacher/data-ready. It is repeated Tier C follow-lag
  evidence.
- The next tiny direction-coverage question is y-. The next quality question is
  actuator/follow cleanup. Neither authorizes 1024/10240/data/RL/VLA.

Verification after seed958:

- `python sim_scripts/cube10cm_reaction_event_gate_audit.py ...seed958...` PASS.
- `python sim_scripts/cube10cm_reaction_window_contract_audit.py ...seed958...`
  PASS.
- `python sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS.

## Fixed Y- Direction-Coverage Screen

User explicitly approved the remaining y- coverage screen as the next tiny local
runtime. The run was not data generation, not 1024/10240, not PPO/RL/VLA, and not
Track A. B200 was not used.

Command intent:

- Fixed direction: y- only.
- Geometry: same seed950-like goodxy `x=0.295,y=-0.044`, lateral `-0.020`.
- Config: wrapper tap defaults, baseline cap/stiffness/effort, trace all envs.
- Seed: 959, `num_envs=16`, one episode.

Runtime/reaction gate:

- Runtime reported no training, no dataset generation, no posewrite, and local
  cuda execution.
- Reaction gate PASSed reaction/contact/no-posewrite/no-overshoot.
- Tap gate `1.0`; final relocation not used (`final_relocation_gate_rate=None`).
- max displacement mean `0.001532856m`.
- final displacement mean `0.001279060m`.
- max speed mean `0.049428599m/s`.
- teacher false: DiffIK clip `1.0`, final TCP error `0.037017073m`, actuator
  tracking lag.

Trace diagnostic:

- `clip_any_rate=1.0`.
- pre-stop rows `1168`, pre-stop `clip_any_rate=1.0`.
- worst pre-stop follow joint 1: mean `0.032676051rad`, p95
  `0.035429358rad`.
- worst pre-stop raw delta joint 2: mean `0.139460339rad`, p95
  `0.914806247rad`.
- likely modes remain link5 target not reached, joint-step clipping, actuator
  target tracking lag.

Reaction-window result:

- accepted windows `16/16`, row count `292`.
- clean teacher false.
- quality tiers: 11 Tier B + 5 Tier C, 0 Tier A, 0 Rejected.
- follow p95/cap p95 `1.006378446`.
- clip mean `1.0`.

Updated tier matrix after seed959:

- Inputs now include seed948/949/950/957/958/959.
- Overall: 96 candidate windows, 80 accepted, acceptance `0.833333333`.
- Quality tiers: 53 Tier B, 27 Tier C, 16 Rejected, 0 Tier A.
- x+: 20/20 accepted, all Tier B.
- x- aggregate: 21/37 accepted, still config-mixed because seed948 old x-
  geometry is rejected while seed949/950 x- are valid Tier B.
- y+: 19/19 accepted, all Tier C.
- y-: 20/20 accepted, 12 Tier B + 8 Tier C.
- readiness remains `ready_for_1024_or_data=false`.

Interpretation:

- y- is no longer under-sampled in this goodxy/lateral pocket.
- y- is better than y+ on quality tier distribution, because seed959 produced 11
  Tier B + 5 Tier C while seed958 y+ produced 16 Tier C.
- This still does not authorize scale-up. There is no Tier A, y+ is still all
  Tier C, and x- must be interpreted by matched config rather than direction-only
  aggregation.
- The next local research step is actuator/follow cleanup on one explicitly
  approved tiny screen. The narrowest quality screen is fixed y+ seed958-like
  geometry changing only `max_diffik_joint_step_rad 0.035 -> 0.050`; a
  config-separated x- cleanup is the alternative if the immediate question is
  direction balance.

Verification after seed959:

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS before runtime.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit before
  runtime and again for seed959.
- `python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py ...seed959...`
  PASS.
- `python sim_scripts/cube10cm_reaction_event_gate_audit.py ...seed959...` PASS
  after sequential rerun.
- `python sim_scripts/cube10cm_reaction_window_contract_audit.py ...seed959...`
  PASS.
- `python sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS.
- A first parallel postprocessing attempt asked the reaction gate/next-step audits
  to read the trace diagnostic before it existed; this was a local ordering race,
  not a runtime/data failure. Sequential reruns produced the PASS artifacts cited
  above.

## Fixed Y+ Cap050 Quality Screen

User explicitly approved the next tiny quality screen. The run was not data
generation, not 1024/10240, not PPO/RL/VLA, and not Track A. B200 was not used.

Question:

- seed958 proved y+ was not a lucky-contact one-off, but all 16 windows were Tier C.
- The narrow hypothesis was whether increasing only `max_diffik_joint_step_rad`
  from `0.035` to `0.050` could move y+ from Tier C to Tier B by reducing the
  follow p95/cap ratio.

Command intent:

- Fixed direction: y+ only.
- Geometry: same seed950/seed958-like goodxy `x=0.295,y=-0.044`, lateral
  `-0.020`.
- Config: wrapper tap defaults, baseline stiffness/damping/effort, trace all envs.
- Changed variable: only `max_diffik_joint_step_rad=0.050`.
- Seed: 960, `num_envs=16`, one episode.

Runtime/reaction gate:

- Reaction gate PASSed reaction/contact/no-posewrite/no-overshoot.
- Tap gate `1.0`; final relocation not used (`final_relocation_gate_rate=None`).
- max displacement mean `0.004593883m`.
- final displacement mean `0.003125247m`.
- max speed mean `0.169194604m/s`.
- teacher false: DiffIK clip `1.0`, final TCP error `0.045421781m`, actuator
  tracking lag.

Trace diagnostic:

- `clip_any_rate=1.0`.
- pre-stop rows `621`, pre-stop `clip_any_rate=1.0`.
- worst pre-stop follow joint 2: mean `0.048595302rad`, p95
  `0.059611082rad`.
- worst pre-stop raw delta joint 2: mean `0.211398292rad`, p95
  `0.783860564rad`.
- likely modes remain link5 target not reached, joint-step clipping, actuator
  target tracking lag.

Reaction-window result:

- accepted windows `16/16`, row count `293`.
- clean teacher false.
- quality tiers: 16 Tier C, 0 Tier A, 0 Tier B, 0 Rejected.
- follow p95/cap p95 `1.141746044`.
- clip mean `1.0`.

Comparison to seed958:

- seed958 y+ baseline: 16/16 accepted, all Tier C, follow p95/cap p95
  `1.151057652`.
- seed960 y+ cap050: 16/16 accepted, all Tier C, follow p95/cap p95
  `1.141746044`.
- The ratio improvement is too small and still above the Tier B follow threshold.
  Absolute pre-stop follow also worsened (`0.041796684rad` p95 in seed958 versus
  `0.059611082rad` p95 in seed960), so pushing cap alone is not the y+ quality fix.

Updated tier matrix after seed960:

- Inputs now include seed948/949/950/957/958/959/960.
- Overall: 112 candidate windows, 96 accepted, acceptance `0.857142857`.
- Quality tiers: 53 Tier B, 43 Tier C, 16 Rejected, 0 Tier A.
- y+: 35/35 accepted, all Tier C.
- y-: 20/20 accepted, 12 Tier B + 8 Tier C.
- x- aggregate remains config-mixed, not a direction-only verdict.

Matrix code cleanup:

- Updated `sim_scripts/cube10cm_reaction_window_tier_matrix.py` so readiness
  reasons distinguish config-mixed direction aggregation.
- The readiness reason is now
  `direction_x-_config_mixed_acceptance_rate=0.567568_inspect_audit_direction`
  instead of a plain direction-only x- acceptance failure.

Interpretation:

- y+ is now robust contact/reaction evidence in this pocket: 35/35 accepted windows.
- y+ is not clean teacher data and not even Tier B yet: all 35 accepted y+ windows
  remain Tier C.
- cap-only cleanup is rejected as the next default route. The next decision is
  whether to spend the next approved tiny runtime on y+ actuator follow itself, or
  on config-separated x- balance.

Verification after seed960:

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS before runtime.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit before
  runtime.
- `python -m py_compile ...` PASS before runtime.
- `git diff --check` PASS before runtime.
- `python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py ...seed960...`
  PASS.
- `python sim_scripts/cube10cm_reaction_event_gate_audit.py ...seed960...` PASS.
- `python sim_scripts/cube10cm_reaction_window_contract_audit.py ...seed960...`
  PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py ...seed960...` PASS as
  audit, still blocking dataset/RL/VLA/TrackA/1024_10k.
- `python sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS after seed960
  and after the config-mixed readiness reason cleanup.

## Fixed Y+ Stiffness600 Quality Screen

User asked to continue the y+ actuator-follow direction and requested that English
terms be explained inline.

Terminology used here:

- `y+`: fixed push/tap direction `[0, 1]` in the simulation/world coordinate frame.
- `actuator follow`: how closely the simulated joint motors follow the joint target
  commands produced by DiffIK. This is motor target-tracking quality, not a
  follower robot.
- `DiffIK`: Differential Inverse Kinematics, the controller that turns desired
  TCP/end-effector motion into small joint commands.
- `reaction gate PASS`: the object-level tap/reaction contract passed: contact
  evidence, reaction evidence, no posewrite, and no overshoot. It does not mean
  clean teacher/data readiness.
- `Tier C`: a valid reaction window whose follow p95/cap ratio is above the quality
  threshold, meaning contact/reaction happened but actuator-follow quality is low.

Question:

- seed960 showed cap-only cleanup was not enough. The next direct actuator-follow
  hypothesis was whether increasing actuator stiffness from `400` to `600` could
  make the same y+ DiffIK commands track better without changing contact geometry.

Command intent:

- Fixed direction: y+ only.
- Geometry: same seed950/seed958-like goodxy `x=0.295,y=-0.044`, lateral
  `-0.020`.
- Config: wrapper tap defaults, default cap `0.035`, default damping/effort,
  trace all envs.
- Changed variable: only `arm_stiffness_override=600`.
- Seed: 961, `num_envs=16`, one episode.

Runtime/reaction gate:

- Reaction gate PASSed reaction/contact/no-posewrite/no-overshoot.
- Tap gate `1.0`; final relocation not used (`final_relocation_gate_rate=None`).
- max displacement mean `0.004412510m`.
- final displacement mean `0.003470089m`.
- max speed mean `0.174832085m/s`.
- teacher false: DiffIK clip `1.0`, final TCP error `0.046149086m`, actuator
  tracking lag.

Trace diagnostic:

- `clip_any_rate=1.0`.
- pre-stop rows `665`, pre-stop `clip_any_rate=1.0`.
- worst pre-stop follow joint 2: mean `0.035403880rad`, p95
  `0.044479728rad`.
- worst pre-stop raw delta joint 2: mean `0.212813382rad`, p95
  `0.734717607rad`.
- likely modes remain link5 target not reached, joint-step clipping, actuator
  target tracking lag.

Reaction-window result:

- accepted windows `16/16`, row count `291`.
- clean teacher false.
- quality tiers: 16 Tier C, 0 Tier A, 0 Tier B, 0 Rejected.
- follow p95/cap p95 `1.200965473`.
- clip mean `1.0`.

Comparison to y+ seed958/960:

- seed958 y+ baseline: 16/16 accepted, all Tier C, follow p95/cap p95
  `1.151057652`.
- seed960 y+ cap050: 16/16 accepted, all Tier C, follow p95/cap p95
  `1.141746044`.
- seed961 y+ stiffness600: 16/16 accepted, all Tier C, follow p95/cap p95
  `1.200965473`.
- Stiffness lowered some aggregate follow numbers but did not fix the window
  quality criterion. It made the reaction-window follow ratio worse than baseline.

Updated tier matrix after seed961:

- Inputs now include seed948/949/950/957/958/959/960/961.
- Overall: 128 candidate windows, 112 accepted, acceptance `0.875`.
- Quality tiers: 53 Tier B, 59 Tier C, 16 Rejected, 0 Tier A.
- y+: 51/51 accepted, all Tier C.
- y-: 20/20 accepted, 12 Tier B + 8 Tier C.
- x- aggregate remains config-mixed, not a direction-only verdict.

Interpretation:

- y+ contact/reaction is robust: 51/51 windows accepted.
- y+ quality is still blocked: 51/51 accepted y+ windows are Tier C.
- cap-only and stiffness-only y+ actuator cleanup both failed. The next move should
  not be a bigger cap or another blind actuator knob. The next local step should
  compare y+ C windows against x-/x+/y- Tier B windows by raw IK delta, clipped
  delta, follow ratio, contact timing, and target/TCP error to decide whether the
  real cause is target/IK demand geometry rather than actuator strength.

Verification after seed961:

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS before runtime.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit before
  runtime.
- `python -m py_compile ...` PASS before runtime.
- `git diff --check` PASS before runtime.
- `python sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py ...seed961...`
  PASS.
- `python sim_scripts/cube10cm_reaction_event_gate_audit.py ...seed961...` PASS.
- `python sim_scripts/cube10cm_reaction_window_contract_audit.py ...seed961...`
  PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py ...seed961...` PASS as
  audit, still blocking dataset/RL/VLA/TrackA/1024_10k.
- `python sim_scripts/cube10cm_reaction_window_tier_matrix.py` PASS after seed961.

## 2026-06-07 local y+ Tier C per-window failure diagnosis

User direction:

- Stop bigger cap, stiffness, and blind actuator knob experiments.
- Do not run GPU/IsaacLab. Use existing traces only.
- Compare y+ Tier C windows against x-/x+/y- Tier B windows by raw IK delta,
  clipped delta, follow ratio, contact timing, and target/TCP error.

Implemented local diagnostic:

- Added `sim_scripts/cube10cm_yplus_tierc_failure_diagnostic.py`.
- This is local/posthoc only. It reads existing seed949/950/957/958/959/960/961
  reaction-window JSON/CSV files and writes:
  - `cube10cm_yplus_tierc_failure_diagnostic_existing_seeds.json`
  - `cube10cm_yplus_tierc_failure_diagnostic_existing_seeds.csv`
  - `cube10cm_yplus_tierc_failure_diagnostic_existing_seeds_windows.csv`
  - `cube10cm_yplus_tierc_failure_diagnostic_existing_seeds_summary.out`
- It performs no GPU runtime, no IsaacLab app launch, no training, no dataset
  generation, no SSH/B200/JHPark, no Track A, and no trace mutation.

Key summary-log evidence:

- Summary line 1: local/posthoc only, `gpu_runtime=NO`, `dataset_generation=NO`.
- Summary line 2: target y+ Tier C windows `51`, follow p95/cap p95
  `1.223191874`, raw delta p95 `0.174502504`, TCP target error p95
  `0.063764448m`, max XY displacement mean `0.012257356m`, anchor step mean
  `179.764706`, phase alpha at anchor mean `0.001307190`.
- Summary line 3: Tier B non-y+ baseline windows `53`, follow p95/cap p95
  `1.030052730`, raw delta p95 `0.280128609`, TCP target error p95
  `0.059400027m`, max XY displacement mean `0.001198941m`, anchor step mean
  `260.641509`, phase alpha at anchor mean `0.072746330`.
- Summary line 4: direct ratios y+ vs Tier B non-y+: raw delta `0.622168298`,
  follow `1.187431643`, TCP error `1.072293444`, max XY displacement
  `10.223485837`, anchor step delta `-80.876804`, phase alpha delta
  `-0.071439141`.
- Summary line 5: x+ Tier B counterexample has raw delta p95 `0.285478155`
  with follow p95/cap p95 `0.996711138`; therefore raw delta size alone does
  not explain y+ Tier C.
- Summary line 8 verdict:
  `supports_simple_raw_ik_demand=False`,
  `supports_yplus_geometry_follow_coupling=True`.

Interpretation:

- The simple "y+ target asks for bigger raw IK delta" hypothesis is rejected.
- y+ is also not weak contact: it moves the cube about `10.223485837x` more than
  the Tier B non-y+ baseline in the reaction window.
- The tighter hypothesis is contact timing / target-geometry coupling: y+ contacts
  around `80.876804` steps earlier and near zero phase alpha, then produces a
  much stronger object reaction while actuator follow exceeds Tier B.
- Next local work should inspect why y+ contacts so early and whether
  precontact/target side-center/lateral/timing geometry is the actual quality
  cause. Do not go back to bigger cap, stiffness, blind actuator sweep, GPU,
  1024/10240/data, PPO/RL, VLA, or Track A.

Verification:

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit,
  still blocking dataset/PPO/RL/VLA/TrackA/1024_10k.
- `python -m py_compile sim_scripts/cube10cm_yplus_tierc_failure_diagnostic.py
  sim_scripts/cube10cm_reaction_window_contract_audit.py
  sim_scripts/cube10cm_reaction_window_tier_matrix.py
  sim_scripts/cube10cm_next_research_step_audit.py
  sim_scripts/cube10cm_tap_objective_contract_audit.py
  sim_scripts/cube10cm_push_diffik_probe.py
  sim_scripts/cube3cm_push_diffik_probe.py
  sim_scripts/cube10cm_reaction_event_gate_audit.py
  sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py` PASS.
- `python sim_scripts/cube10cm_yplus_tierc_failure_diagnostic.py` PASS and wrote
  the JSON/CSV/window CSV/summary artifacts above.

## 2026-06-07 local y+ early-contact geometry audit

User direction:

- Continue from the y+ Tier C diagnosis.
- Do not use GPU. Inspect why y+ contacts/reacts at near-zero phase alpha.
- Focus on precontact, side-center target, lateral offset, and timing geometry.
- Also answer whether this style of validation is normal in robotics research.

Implemented local audit:

- Added `sim_scripts/cube10cm_yplus_early_contact_geometry_audit.py`.
- The script reads existing seed949/950/957/958/959/960/961 reaction-window
  audit JSONs and their source trace CSVs.
- It computes, per accepted window:
  - first push-phase step,
  - first near-TCP step,
  - first object-reaction step,
  - first measured-contact step,
  - pre-anchor max displacement/tip/speed,
  - initial target along/lateral/z geometry,
  - anchor target/TCP geometry.
- It performs no GPU runtime, no IsaacLab app launch, no training, no dataset
  generation, no SSH/B200/JHPark, no Track A, and no trace mutation.

Key evidence:

- Summary line 1: local/posthoc only, `gpu_runtime=NO`,
  `dataset_generation=NO`.
- Summary line 2: y+ Tier C windows `51`, first reaction step mean `46.039216`,
  measured contact step mean `181.176471`, reaction lead `135.137255`, anchor
  `40.235294` steps before push start, first reaction phase alpha `0.0`.
- Summary line 3: Tier B non-y+ baseline windows `53`, first reaction step mean
  `110.792453`, measured contact step mean `249.150943`, reaction lead
  `144.679245`, anchor `40.641509` steps after push start, first reaction phase
  alpha `0.037735850`.
- Summary line 4: in the 24 steps before anchor, y+ max XY displacement mean is
  `0.012257356m` versus baseline `0.000895049m` (`13.694612400x`), and y+ tip
  mean is `13.261459369deg` versus baseline `1.004456226deg`
  (`13.202625486x`).
- Summary line 5: nominal initial target along/lateral offsets match:
  y+ `-0.059999944m/-0.019999995m`, baseline
  `-0.059999983m/-0.019999983m`. y+ target z is side-center near cube z
  (`-0.000000022m`), while mixed non-y+ Tier B baseline averages
  `0.034905637m` above cube z.
- Summary line 7: y- has low pre-anchor displacement `0.000900645m`, so low
  side-center height alone is not sufficient.
- Summary line 8 verdict:
  `supports_yplus_preanchor_reaction_accumulation=True`,
  `supports_unique_measured_contact_lead=False`,
  `supports_yplus_approach_phase_geometry_hypothesis=True`.

Interpretation:

- y+ is not merely an actuator-strength problem and not simply a larger raw IK
  demand problem.
- y+ also is not uniquely explained by "first object reaction occurs earlier than
  measured contact"; the baseline has long reaction-to-contact lead too.
- The differentiator is the 24-step pre-anchor window: y+ has about 13x larger
  displacement and tip before/around contact anchor while still in the approach
  phase.
- The next local step should audit/propose one tiny y+ config screen around
  precontact/lateral/height/timing. Do not return to bigger cap/stiffness/blind
  actuator sweeps, GPU, 1024/10240/data, PPO/RL, VLA, or Track A.

Robotics-method note:

- Yes, this style is normal when a robot task passes at the outcome level but is
  not yet dataset-quality: isolate by event phase, contact geometry, and command
  quality before scaling data.
- The guardrail is that the loop must remain bounded: each diagnostic should kill
  or support exactly one hypothesis and produce a concrete next variable. This
  session killed "bigger raw IK demand", killed "actuator knob first", and
  narrowed the next variable to y+ precontact/lateral/height/timing geometry.

Verification:

- `python -m py_compile sim_scripts/cube10cm_yplus_early_contact_geometry_audit.py`
  PASS.
- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit,
  still blocking dataset/PPO/RL/VLA/TrackA/1024_10k.
- `python sim_scripts/cube10cm_yplus_early_contact_geometry_audit.py` PASS and
  wrote the JSON/CSV/summary artifacts above.

## 2026-06-07 local y+ precontact candidate audit

User direction:

- The next GPU should not be run blindly.
- The narrowest candidate is y+ `precontact_clearance_m` only, because the y+
  issue is approach/pre-anchor reaction, not push-phase displacement.
- Height is risky because seed944 height050 killed contact; lateral should wait
  because it introduces direction-side asymmetry.

Implemented local audit:

- Added `sim_scripts/cube10cm_yplus_precontact_candidate_audit.py`.
- The script reads existing seed958 summary plus the y+ Tier C and early-contact
  diagnostic JSONs.
- It computes the nominal target geometry for baseline `precontact=0.010` and
  candidate `precontact=0.020`.
- It writes JSON and summary artifacts only. It performs no GPU runtime, no
  IsaacLab app launch, no training, no dataset generation, no SSH/B200/JHPark, no
  Track A, and no trace mutation.

Key evidence:

- Summary line 1: local/config only, `gpu_runtime=NO`, `dataset_generation=NO`.
- Summary line 2: y+ windows `51`, pre-anchor displacement ratio
  `13.694612400`, pre-anchor tip ratio `13.202625486`, anchor
  `40.235294` steps before push start, raw delta ratio `0.622168298`, follow
  ratio `1.187431643`.
- Summary line 3: the one changed variable is `precontact_clearance_m`, from
  `0.010000` to `0.020000`; nominal pre-target along changes from `-0.060000m`
  to `-0.070000m`.
- Summary line 4: through target stays `-0.040000m`; push path length increases
  from `0.020000m` to `0.030000m`, which is the known risk.
- Summary line 5: `supports_precontact_first=True` and
  `candidate_is_tiny_one_variable_change=True`; height-first and lateral-first
  are rejected for now.
- Summary line 6: runtime was NOT run; next seed would be seed962 and requires
  explicit GPU approval.

Interpretation:

- This audit does not prove the candidate will fix y+ Tier C. It only makes the
  next hypothesis precise: delay/reduce approach-phase early contact by increasing
  initial standoff, while leaving lateral, height, actuator, cap, DLS, and data
  scale untouched.
- The go/no-go order remains reaction/contact/no-posewrite/no-overshoot first;
  then quality tier metadata. Final 1cm relocation is not part of this objective.

Verification:

- `python sim_scripts/cube10cm_yplus_precontact_candidate_audit.py` PASS and
  wrote the JSON/summary artifacts.
- `python -m py_compile sim_scripts/cube10cm_yplus_precontact_candidate_audit.py
  sim_scripts/cube10cm_yplus_early_contact_geometry_audit.py
  sim_scripts/cube10cm_yplus_tierc_failure_diagnostic.py
  sim_scripts/cube10cm_reaction_window_contract_audit.py
  sim_scripts/cube10cm_reaction_window_tier_matrix.py
  sim_scripts/cube10cm_next_research_step_audit.py
  sim_scripts/cube10cm_tap_objective_contract_audit.py
  sim_scripts/cube10cm_push_diffik_probe.py
  sim_scripts/cube3cm_push_diffik_probe.py
  sim_scripts/cube10cm_reaction_event_gate_audit.py
  sim_scripts/cube3cm_push_diffik_trace_diagnostic_audit.py` PASS.
- `python sim_scripts/cube10cm_tap_objective_contract_audit.py` PASS.
- `python sim_scripts/cube10cm_next_research_step_audit.py` PASS as audit,
  still blocking dataset/PPO/RL/VLA/TrackA/1024_10k.
- `git diff --check` PASS.
