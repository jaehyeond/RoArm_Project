# 2026-06-08 cube10cm teacher contact-frame design

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier work.
- Not Track A, not grasp/dataset/training, not final 1cm relocation work.
- No B200/SSH/pull/.ssh work. B200 remains expired/disconnected.
- One local IsaacLab runtime was run only after explicit approval and only as a negative control.

## User Question

The user rejected asking them to choose among:

1. `true_side_center_tcp`
2. `upper_edge_contact_proxy`
3. `tool_oriented_side_contact_proxy`

The task was to test all three, explain why existing seed962 evidence alone was or was not enough, and keep dataset/RL/RoArm blocked unless evidence actually unblocks it.

## Why Existing seed962 Evidence Was Not Enough

Existing seed962 side-center evidence was enough to prove the current execution is not clean side-center tap:

- `cube10cm_contact_frame_geometry_mismatch_audit_summary.out` shows side-center target contract but first-contact `tcp_minus_target_z_mean=0.052857013m`, z-error fraction `0.983196354`, and top-near contact rate `1.0`.
- `cube10cm_visual_sim_sanity_audit_summary.out` shows visual replay contact is present but clean tap is not verified because env0 contact frame has `tcp_z=0.100452900`, `target_z=0.049999580`, delta `0.050453320m`, and `clip_any=1`.

But existing seed962 alone could not distinguish whether an upper-edge/top proxy is a valid teacher criterion or merely an accidental contact shortcut. That required one controlled negative comparison: keep seed962 geometry and change only `--tcp_height_mode top_margin`.

## Local Design Audit

Added:

- `sim_scripts/cube10cm_teacher_contact_frame_design_audit.py`

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_teacher_contact_frame_design_audit.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_teacher_contact_frame_design_audit_summary.out`

Summary:

- Line 1: local audit only; no GPU, no dataset generation, no training, no robot control, no SSH.
- Line 4: `true_side_center_tcp` score `0.467792681`, `side_center_z_reached_10mm_rate=0.0`, z err mean `0.052857013`; semantically correct but tracking-failed.
- Line 5: `upper_edge_contact_proxy` score `0.662870807`, upper z/total reach rates `1.0/1.0`, upper z err mean `0.001206460`; best explains current visual contact but teaches top contact.
- Line 6: `tool_oriented_side_contact_proxy` score `0.654000000`, current DiffIK command type `position`; position-only cannot validate orientation path from trace alone.
- Line 7: selected teacher criterion is `tool_oriented_side_contact_proxy`.

## Guard Checks Before Runtime

Ran local guards before the approved runtime:

- `python sim_scripts/cube10cm_tap_objective_contract_audit.py`
- `python sim_scripts/cube10cm_next_research_step_audit.py`
- `python sim_scripts/cube10cm_yplus_precontact_candidate_audit.py`
- `python -m py_compile ...`
- `git diff --check`

Explicit seed962 next-step audit stayed blocked for teacher quality:

- Reaction gate true.
- Teacher quality false.
- Contact `1.0`, overshoot `0.0`.
- DiffIK clip `1.0`.
- Final TCP error `0.051811996m`.
- Next direction remained narrow IK/tracking cleanup, not dataset/RL/RoArm.

## Approved Negative-Control Runtime

Ran exactly one local IsaacLab 16-env runtime after explicit approval, changing only:

- `--tcp_height_mode top_margin`

The command kept:

- `--num_envs 16`
- `--episodes 1`
- `--seed 962`
- fixed y+ geometry `x=0.295`, `y=-0.044`, `push_dir 0 1`
- `--base_lateral_offset_m -0.020`
- `--xneg_tcp_center_height_offset_m 0.050`
- `--precontact_clearance_m 0.020`
- trace diagnostics/all envs

Runtime stdout confirmed local professor-wrapper mode:

- 10cm/0.72kg cube.
- `tcp_height_mode=top_margin`.
- `training=NO`
- `dataset_generation=NO`
- `grasp=NO`
- `attach_posewrite=NO`
- `rollout_object_posewrite=NO`

## Post-Runtime Audits

Top-margin reaction gate:

- `diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_topmargin_seed962_reaction_gate_audit.json`
- Line 1: PASS, reaction event `1.0`, contact evidence `1.0`, no posewrite, no overshoot.
- Line 2: max displacement mean `0.001112372m`, final displacement mean `0.000045329m`.
- Line 3: summary teacher quality `READY`, final TCP err mean `0.011275044m`, clip mean `0.495833354`.

Top-margin reaction-window audit:

- `cube10cm_reaction_window_seed962_topmargin_audit.json`
- Line 1: PASS, envs `16`, accepted windows `16`.
- Line 3: clean DiffIK teacher window `NOT_READY`, clip mean `0.669956140`.
- Line 4: tiers `{'B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH': 16}`.

Top-margin contact-frame mismatch audit:

- `cube10cm_contact_frame_geometry_mismatch_topmargin_seed962_summary.out`
- Line 3: `tcp_height_mode=top_margin`, `diffik_clip_rate_mean=0.495833354`.
- Line 4: first-contact z mismatch reduced to `0.005681654m`.
- Line 5: contact remains near/above cube top: `tcp_above_live_cube_center_z_mean=0.057616640`, `tcp_below_live_cube_top_z_mean=-0.007616640`, top-near rate `1.0`.
- Line 7: first-contact clip is still `1.0`, mode `link2_to_link3`.

Runtime comparison audit:

- `cube10cm_teacher_contact_frame_runtime_comparison_audit_summary.out`
- Line 2: side-center baseline had reaction gate true, teacher quality false, clip `1.0`, final TCP err `0.051811996m`, max displacement `0.002923813m`, controlled push `0.5625`, tiers 2B+14C.
- Line 3: top-margin had reaction gate true, teacher quality true, clip `0.495833354`, final TCP err `0.011275044m`, max displacement `0.001112372m`, final displacement `0.000045329m`, controlled push `0.0`, tiers 16B, but clean window false.
- Line 4: top-margin versus side-center ratios: clip `0.495833354x`, final TCP err `0.217614551x`, max displacement `0.380452440x`, final displacement `0.028655940x`, tip `0.157045606x`.
- Line 6: upper-edge proxy tracking improved, upper-edge proxy tap strength weakened, upper-edge proxy not selected as teacher.
- Line 7: dataset/RL/RoArm still not unblocked.

## Critical Verdict

All three criteria were tested, but not all by the same mechanism:

- `true_side_center_tcp`: tested by the existing seed962 side-center runtime and mismatch/visual audits. Verdict: correct semantic goal, failed execution under clipping.
- `upper_edge_contact_proxy`: tested by the local design audit and the approved top-margin runtime. Verdict: improves tracking and summary teacher quality, but weakens tap and encodes upper/top contact. Reject as teacher.
- `tool_oriented_side_contact_proxy`: tested by code/design feasibility. Verdict: selected criterion, but current DiffIK probe is position-only, so it needs a local tool/contact-proxy plus orientation-path preflight before any next runtime.

The important conclusion is negative: top-margin proves we can make numbers look cleaner by moving the target upward, but that is not the professor branch teacher. It changes the physical contact semantics and weakens the tap.

## Dataset/RL/RoArm Status

- Event/contact evidence remains useful.
- Clean tap teacher is still not verified.
- Action-teacher dataset remains blocked.
- Large IsaacLab dataset remains blocked.
- IsaacLab RL remains blocked.
- RoArm-M3-Pro deployment remains blocked.

## Next Step

Local-only next step:

1. Define the physical tool/contact proxy that should touch the 10cm cube side.
2. Check whether the current position-only DiffIK target can express that proxy without top contact.
3. If not, design an orientation-aware or proxy-aware local preflight.
4. Only after that, request at most one tiny local runtime.

Do not run another top-margin/height sweep. Do not start dataset, RL, RoArm deployment, Track A, 1024/10240, B200, SSH, pull, or .ssh copying.
