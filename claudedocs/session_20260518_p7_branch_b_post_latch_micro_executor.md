# Session 2026-05-18 - P7 Branch B post-latch micro-command executor probe

## Scope Guard

- Stayed on Track A P7/Branch B, local CLOSE handoff diagnostics only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not go to the transport target.
- Did not execute release or scripted release variants.
- Did not run P7 training or tune scalar/threshold/release guidance.
- Did not edit env/train/chain defaults.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D036,
  `claudedocs/EXPERIMENT_LEDGER.md`, and the requested Branch B session docs.
- Ran `git status --short` before coding: only this new diagnostic became
  untracked after the edit.
- Verified requested md5s before coding; all matched:
  - `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
    `a7ed4387e0ab1ce5b95de08f59c2eb52`
  - `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
    `938a94b3b856dcc5a48527991a87c1e9`
  - `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
    `58b628682a536535d3d9a6790c51974d`
  - `sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
    `6cb899ca124ff588fcc011d2805fa605`
  - `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
    `339bdfd2ced7cf05b4ce87d2cd92128a`
  - `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
    `d9a07b43bed44f6061144234d7f6ec36`
  - `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
    `fe2b227d2a111bf1c7acfe82e8f43133`
  - `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
    `88b4b8b33cd7aeecd6a18f78bf144283`
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
    `6af24284baef540f190b762e5da164a5`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
  - `launch_chain_topdown.sh` `b34ef3853ac993a1e2adbaddb420adab`
  - `launch_p6v17_transport_release.sh` `2acd462042d0997610fca25ff7a41e21`
- New diagnostic md5 after coding:
  - `sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`
    `c74d92816df12953c26fed577656840e`

## Prior Evidence Rechecked

- D034/D035 code boundary:
  - `roarm_rl/roarm_stack_env.py` lines 484-498: `_pre_physics_step`
    updates `robot_dof_targets` from action, `_apply_action` sends targets and
    calls `_update_grasp_attach` if `_grasped`.
  - `roarm_rl/roarm_stack_env.py` lines 1184-1195: `_grasped` latches from
    distance plus gripper threshold.
  - `roarm_rl/roarm_stack_env.py` lines 1216-1236: current attach pose-writes
    sponge root pose to TCP and optionally zeroes velocity.
- D034 B200 default:
  `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_default_b200.out`
  line 41 confirms no constraints/SurfaceGripper/transport/release/P7/default edits;
  line 43 confirms `move_cmds_executed=0`, `raw_max_gap_m=0.211271`,
  `raw_gap_ok=NO`; line 83 fails first stationary hold with
  `target_error_m=0.015684`, `tcp_step_m=0.016131`, `quat_angle_deg=21.267`;
  lines 85-86 report `post_latch_hold_ok=NO` and success `NO`.
- D035 marker-only control:
  `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_no_posewrite_b200.out`
  lines 83-88 pass sampled hold steps, line 89 reports
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_pose_drift_m=0.000000`, and lines 90-91 report
  `post_latch_hold_ok=YES` and success `YES`.
- D036 model matrix:
  - `posewrite_tcp` v2 line 83 reproduces `target_error_m=0.015684`,
    `tcp_step_m=0.016131`, `quat_angle_deg=21.267`, success `NO`.
  - `delayed_posewrite` v2 line 86 fails when center-snap begins:
    `target_error_m=0.015686`, `tcp_step_m=0.016133`.
  - `oneshot_align` v2 line 83 fails first hold:
    `target_error_m=0.005097`, `quat_angle_deg=7.080`.
  - `offset_preserve_posewrite` v2 lines 83-91 pass stationary hold with
    line 89 `hold_max_target_error_m=0.000817`,
    `hold_max_offset_error_m=0.000001`, `hold_max_speed_mps=0.000869`,
    `posewrite_calls=40`, but lines 90-91 are still only local kinematic
    stationary pass with `attach_physics_validated=NO`.
- Previous micro-motion failure:
  - `posewrite_tcp` micro line 83 still fails before micro-motion.
  - `marker_only` micro lines 88-94 and `offset_preserve_posewrite` micro
    lines 88-94 both fail to reach the 4mm `plus_x` target with
    `micro_max_target_error_m=0.004764`, `micro_motion_ok=NO`.
  - `offset_preserve_posewrite_d8mm` line 42 changes only
    `micro_delta_m=0.008000`; lines 88-94 fail with
    `micro_max_target_error_m=0.008699`, `micro_motion_ok=NO`.
  - All relevant stderr files have only known cpufreq/NVML/Fabric warnings on
    lines 1-4 and no Python traceback.

## Code Added

- Added `sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`.
- The probe reuses the conservative `PRE_MOVE* -> CLOSE` stream, then performs
  a short marker-only or offset-preserve local hold and instruments one bounded
  post-latch micro command.
- It prints:
  - `target_q_deg`, close command q, `delta_q_deg`
  - commanded `robot_dof_targets` before/after `env.step(null_action)`
  - current joint positions and joint error in rad/deg
  - FK expected TCP from target_q versus realized fresh TCP
  - overwrite detection for `robot_dof_targets`
  - realized TCP delta from close pose
- It does not change `roarm_rl/roarm_stack_env.py` or any train/chain defaults.

Local checks:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`
- `python sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py --help`

## B200 Runs

### Marker-only 4mm TCP micro

Command wrote:

- `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_marker_only_b200.out`
- `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_marker_only_b200.err`

Evidence:

- stdout line 41 confirms strict executor-only scope:
  no constraint insertion, no fixed/dynamic integration, no SurfaceGripper, no
  attached transport, no transport target, no release, no P7 training, no
  default edits, and `claim_attach_success=NO`.
- stdout line 42 reports `micro_mode=tcp_micro`, `micro_delta_m=0.004000`,
  `executor_steps=20`, `handoff_model=marker_only`.
- stdout line 43 confirms source stream truncation before MOVE:
  `source_events_total=44`, `executed_events=39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- stdout lines 82-86 pass the marker-only post-latch stationary hold.
- stdout line 87 proves the 4mm micro target is not a zero command:
  `ik_converged=YES`, `delta_q_norm_deg=0.790232`,
  `delta_q_max_abs_deg=0.581820`, and `expected_tcp_delta_m=0.003511`.
- stdout lines 88-93 show `robot_dof_targets` were not overwritten:
  `target_overwrite_max_rad=0.00000007`, `overwrite_after_step=NO`.
- stdout lines 88-93 also show the robot did not realize the command:
  `target_tcp_error_m` stays around `0.004764-0.004765`,
  `realized_tcp_delta_from_close_m` reaches only `0.000080`, and
  `reached=NO`.
- stdout line 94 aggregates the separation:
  `target_q_distinct=YES`, `expected_motion_nonzero=YES`,
  `targets_not_overwritten=YES`, but `realized_motion_seen=NO`,
  `executor_reached=NO`, `max_realized_tcp_delta_m=0.000080`,
  `min_joint_error_max_deg=0.748257`, `action_scale=0.100000`,
  `null_action_max_abs=0.000000`.
- stdout lines 95-96 report success `NO`.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings.

### Marker-only 5deg joint nudge cross-check

Command wrote:

- `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_joint_nudge_b200.out`
- `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_joint_nudge_b200.err`

Evidence:

- stdout line 41 confirms the same strict scope.
- stdout line 42 reports `micro_mode=joint_nudge`, `joint_nudge_index=1`,
  `joint_nudge_deg=5.000`, `executor_steps=80`, `handoff_model=marker_only`.
- stdout lines 82-86 again pass marker-only stationary hold.
- stdout line 87 proves this was a large nonzero target:
  `delta_q_norm_deg=5.000000`, `delta_q_max_abs_deg=5.000000`,
  `expected_tcp_delta_m=0.024271`.
- stdout lines 88-93 again show targets were not overwritten:
  `target_overwrite_max_rad=0.00000004`, `overwrite_after_step=NO`.
- stdout lines 88-93 show joint 1 stayed near 25deg instead of moving to the
  30deg target; `joint_error_max_deg` remains about `5.0`, and
  `target_tcp_error_m` remains about `0.0239`.
- stdout line 94 aggregates:
  `target_q_distinct=YES`, `expected_motion_nonzero=YES`,
  `targets_not_overwritten=YES`, but `realized_motion_seen=NO`,
  `executor_reached=NO`, `max_realized_tcp_delta_m=0.000206`,
  `min_joint_error_max_deg=4.992061`, `posewrite_calls=0`,
  `action_scale=0.100000`, `null_action_max_abs=0.000000`.
- stdout lines 95-96 report success `NO`.
- stderr lines 1-4 contain only known cpufreq/NVML/Fabric warnings.

## Interpretation

- Do not interpret the failed previous micro-motion probe as offset-preserve
  moving failure. The micro target was not realized by the robot.
- The immediate executor hypotheses split cleanly:
  - The micro targets were nonzero and FK-predicted motion was nonzero.
  - `robot_dof_targets` were not overwritten by `env.step(null_action)` or
    action scaling/clamping in these runs.
  - Even a 5deg post-CLOSE joint target nudge did not move the corresponding
    actual joint or fresh TCP materially.
- The live blocker is now narrower: after local CLOSE/latch, the articulation
  is not realizing newly commanded post-latch joint targets in this diagnostic,
  despite the env target buffer preserving them.
- This is not P7 success, not constraint integration, not object attachment
  physics, not attached transport, and not release physics.

## Next Step

- Continue pre-integration executor debugging only.
- Instrument whether `_robot.set_joint_position_target()` is actually called
  with the post-latch target and whether the Articulation/controller internal
  target buffer changes after `_apply_action`.
- If needed, test the same 5deg joint nudge before CLOSE vs immediately after
  CLOSE to isolate whether the latch state, gripper-closed contact, or command
  application path changes controller behavior.
- Only after realized TCP micro movement works, re-run marker-only vs
  offset-preserve local micro-motion. Still do not proceed to transport target,
  release, SurfaceGripper, or RoArm chain constraint insertion.
