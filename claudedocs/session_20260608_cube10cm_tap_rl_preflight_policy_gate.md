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
