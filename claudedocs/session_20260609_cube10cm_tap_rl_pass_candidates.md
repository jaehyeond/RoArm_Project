# 2026-06-09 cube10cm tap RL pass candidates

Scope: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier / RL unblock diagnosis branch only. No Track A, dataset generation, PPO/RL training, robot control, SSH, or B200 use.

## Candidate Ladder

1. Previous-target-base controller candidate:
   - Added default-off `--builtin_diffik_target_base_mode previous_joint_target`.
   - Result: partial controller fix. Applied FK enters the band, actual point contact proxy remains zero.

2. Previous-target-base plus drive boost:
   - Added default-off arm actuator knobs and tested `160/8/8/6.28`.
   - Result: cube motion increased, strict point contact still zero.

3. Link5 single-corner proxy retarget:
   - Added default-off `--tool_contact_proxy_mode link5_collision_corner_011`.
   - Result: applied FK enters, actual single-point contact still zero; single-point proxy is insufficient.

4. Larger target lead:
   - Tested `--joint_target_lead_limit_rad 0.120`.
   - Result: more motion and more terminations, strict point contact still zero.

5. Link5 collision AABB contact proxy with default precontact:
   - Added default-off env/harness `tap_contact_proxy_mode=link5_collision_aabb`.
   - Result: PASS but degenerate, because initial AABB contact is already true.

6. Link5 collision AABB contact proxy with `precontact_clearance_m=0.040`:
   - Result: non-degenerate PASS.
   - Summary line 3: `steps_executed=334`, `tap_contact_proxy_mode=link5_collision_aabb`, `precontact_clearance_m=0.04`.
   - Summary line 4: initial contact is not already true: `initial_face_gap_m=-0.019569765776395798`.
   - Summary line 5: `contact_seen=1.0`, `reaction_seen=0.5`, `tap_success=0.5`, `overshoot_seen=0.0`.
   - Summary line 6: `terminated_count=0`, `truncated_count=0`.
   - Detail trace posthoc: rows `668`, `contains_action_fields=false`, `action_teacher_dataset=false`, first AABB contact step `162`, first success step `333`, actual contact rows `343`.

## Interpretation

The exact broken behavior was not simply "DiffIK cannot push the cube." The earlier controller command and applied-target path could move the object, but the strict contact gate used a single `_tcp_pos_w` point that did not represent the actual collision geometry. Blind drive boost and larger target lead increased cube motion or terminations without producing strict point-contact success.

Candidate6 is the first non-degenerate strict positive-control pass for this branch because it uses a geometry-aware contact proxy and a reset/precontact offset that starts outside contact, then reaches contact during rollout. This unblocks Stage-0 IsaacLab validation planning for this branch.

Large dataset generation, PPO/RL scale-up, and RoArm deployment are still blocked until Candidate6 is promoted beyond a 2-env single-seed tiny positive-control. The detail traces are diagnosis telemetry and still not action-teacher datasets.

## Sources

- `roarm_rl/roarm_cube_push_env.py`
- `roarm_rl/test_positive_control_cube_tap10cm.py`
- `sim_scripts/cube10cm_tap_rl_pass_candidate_compare_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_prevtarget_pass_candidates_audit_summary.out:1-12`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_pre040_candidate6_sanity_summary.out:1-10`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_nearface_prevtargetbase_link5aabb_pre040_candidate6_detail_trace.json`
