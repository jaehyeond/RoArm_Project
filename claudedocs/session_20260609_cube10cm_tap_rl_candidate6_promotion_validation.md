# 2026-06-09 cube10cm tap RL Candidate6 promotion validation

Scope: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier / RL unblock diagnosis branch only. No Track A, dataset generation, PPO/RL training, action-teacher dataset generation, robot control, SSH, or B200 use.

## Fixed Contract

Candidate6 was frozen as the Stage-0 positive-control contract:

- `cube=(0.240,0.000)`, push dir `(+1,0)`.
- `controller_mode=isaac_builtin_diffik_step_clipped_direct_apply`.
- `target_path_mode=near_face_goal`.
- `builtin_diffik_target_base_mode=previous_joint_target`.
- `tap_contact_proxy_mode=link5_collision_aabb`.
- `tool_contact_proxy_mode=hand_tcp`.
- `precontact_clearance_m=0.040`.
- `episode_length_s=6.08`, `steps=580`.
- `builtin_diffik_step_clip_rad=0.010`.
- `joint_target_lead_limit_rad=0.060`.
- Default arm drive remains `arm_stiffness=80.0`, `arm_damping=4.0`, `arm_effort_limit=2.5`, `arm_velocity_limit=3.14`.

## Launch Hygiene

The first direct base-Python seed963 launch was BLOCKED before Isaac startup:

- blocker: `ModuleNotFoundError`.
- error: `No module named 'isaaclab'`.
- status: `BLOCKED`.
- steps executed: `0`.

That artifact is launch hygiene only, not physics evidence. It was overwritten by the valid seed963 runtime.

Valid runtime command family:

```bash
conda run -n isaaclab --no-capture-output python -u -m roarm_rl.test_positive_control_cube_tap10cm \
  --num_envs <2-or-8> --steps 580 --seed <seed> --device cuda:0 \
  --fixed_cube_x_m 0.240 --fixed_cube_y_m 0.000 \
  --fixed_push_dir_x 1.0 --fixed_push_dir_y 0.0 \
  --precontact_clearance_m 0.040 --tcp_top_margin_m -0.050 --goal_push_m 0.006 \
  --target_path_mode near_face_goal --episode_length_s 6.08 \
  --controller_mode isaac_builtin_diffik_step_clipped_direct_apply \
  --closed_loop_push_steps 580 \
  --builtin_diffik_lambda 0.010 --builtin_diffik_step_clip_rad 0.010 \
  --builtin_diffik_target_base_mode previous_joint_target \
  --tool_contact_proxy_mode hand_tcp --tap_contact_proxy_mode link5_collision_aabb
```

## Stage0A Multi-Seed Fixed Geometry

Runs:

- Existing baseline seed962, `num_envs=2`.
- New seed963, `num_envs=2`.
- New seed964, `num_envs=2`.
- New seed965, `num_envs=2`.

All four pass the promotion audit:

- status `PASS`.
- initial contact `0.0`.
- first contact step `162`.
- first success step `333`.
- actual contact rows `343`.
- `tap_success=0.5`.
- `contact_seen=1.0`.
- `overshoot_seen=0.0`.
- terminated/truncated `0/0`.
- `contains_action_fields=false`.
- contract violations `0`.

## Stage0B Small Env-Scale

Run:

- seed962, `num_envs=8`.

Promotion audit result:

- status `PASS`.
- initial contact `0.0`.
- first contact step `162`.
- first success step `331`.
- actual contact rows `1358`.
- `tap_success=0.375`.
- `contact_seen=1.0`.
- `overshoot_seen=0.0`.
- terminated/truncated `0/0`.
- `contains_action_fields=false`.
- contract violations `0`.

## Verdict

Candidate6 promotion validation PASSed:

- `baseline_pass=True`.
- `stage0a_complete=True`.
- `stage0a_pass=True`.
- `stage0b_complete=True`.
- `stage0b_pass=True`.
- `candidate6_promotion_validation_pass=True`.

This promotes Candidate6 from a single 2-env/seed962 pass into a small Stage-0 positive-control contract. It unblocks pilot RL smoke/design using the fixed AABB-contact env contract.

Large dataset generation, PPO/RL scale-up, action-teacher dataset claims, and RoArm deployment remain blocked. The detail traces are diagnosis/promotion telemetry only and are not action-teacher datasets.

## Files

- `sim_scripts/cube10cm_tap_rl_candidate6_promotion_audit.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_validation_audit_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_validation_audit.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_stage0a_seed963_n2_sanity_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_stage0a_seed964_n2_sanity_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_stage0a_seed965_n2_sanity_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tap_rl_candidate6_promotion_stage0b_seed962_n8_sanity_summary.out`

## Next

Design a tiny pilot RL smoke path around the fixed Candidate6 AABB-contact env contract. The next step should be a local design/preflight or very small smoke, not a large dataset, PPO scale-up, action-teacher dataset, or RoArm deployment.
