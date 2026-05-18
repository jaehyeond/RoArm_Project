# Session 2026-05-18 - P7 Branch B post-latch target-delivery probe

## Scope Guard

- Continued Track A P7/Branch B only.
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
- Ran `git status --short` before coding. Existing dirty state was preserved:
  `START_HERE.md`, `claudedocs/EXPERIMENT_LEDGER.md`, the prior
  post-latch micro-executor session doc, and the prior post-latch executor probe
  were already modified/untracked.
- Verified requested md5s before coding; all matched the prompt, including:
  `sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`
  `c74d92816df12953c26fed577656840e`,
  `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
  `a7ed4387e0ab1ce5b95de08f59c2eb52`,
  `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
  `938a94b3b856dcc5a48527991a87c1e9`,
  `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
  `58b628682a536535d3d9a6790c51974d`,
  `sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
  `6cb899ca124ff588fcc011d2805fa605`,
  `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
  `339bdfd2ced7cf05b4ce87d2cd92128a`,
  `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
  `d9a07b43bed44f6061144234d7f6ec36`,
  `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
  `fe2b227d2a111bf1c7acfe82e8f43133`,
  `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
  `88b4b8b33cd7aeecd6a18f78bf144283`,
  `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
  `6af24284baef540f190b762e5da164a5`,
  `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`,
  `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`,
  `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`,
  `launch_chain_topdown.sh` `b34ef3853ac993a1e2adbaddb420adab`, and
  `launch_p6v17_transport_release.sh` `2acd462042d0997610fca25ff7a41e21`.

## Prior Evidence Rechecked

- D034/D035 still stand: current TCP-center pose-write attach fails first
  stationary post-latch hold, while marker-only/no-posewrite passes only as a
  negative control.
- D036 still stands: center-snap geometry is the local stationary handoff
  trigger; offset-preserve is only a stationary kinematic candidate.
- Previous post-latch micro-executor result still stands:
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_marker_only_b200.out`
  line 87 and
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_joint_nudge_b200.out`
  line 87 prove nonzero targets; lines 88-93 in both logs show target buffers
  were not overwritten; line 94 in both logs shows realized motion failed.

## Code Added

- Added `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
  md5 `aad6398a9d47fef5c80efbd212e619d8`.
- The script wraps `_robot.set_joint_position_target()` inside the diagnostic,
  snapshots Articulation data target fields, and compares the same 5deg shoulder
  nudge:
  - before CLOSE, at the grasp pose with gripper open and `_grasped=NO`;
  - after CLOSE/latch through normal `env.step(null_action)`;
  - after CLOSE/latch through direct `_robot.set_joint_position_target()` plus
    `scene.write_data_to_sim()`, `sim.step()`, and `scene.update()`.
- It keeps `_update_grasp_attach` marker-only. No pose-write attach, constraint,
  SurfaceGripper, transport target, release, training, or default edit is used.

Local checks:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
- `python sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py --help`

## B200 Run

Authoritative final run:

- `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out`
- `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.err`

Earlier v1/v2 runs were useful for debugging the script but are not the final
interpretation. v1/v2 used a too-loose incidental TCP-motion metric; v1 also
accidentally closed the gripper during the intended before-CLOSE comparison.
v3 fixes both.

Evidence from v3 stdout:

- Line 41 confirms strict scope: no constraint prim insertion, no fixed/dynamic
  integration, no SurfaceGripper, no attached transport, no transport target, no
  release, no P7 training, no default edits, and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`,
  `claim_attach_success=NO`.
- Line 43 confirms execution remains pre-transport:
  source `events_total=44`, executed pre-moves `38`, `move_cmds_executed=0`,
  `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 44 proves the 5deg shoulder nudge is nonzero:
  `delta_q_deg=[+0.000, +5.000, ...]`, `target_q_open_deg` keeps gripper open
  for before-CLOSE, and `expected_tcp_delta_m=0.024271`.
- Lines 82-94 compare before CLOSE with gripper open and `grasped=NO`. Lines
  83-85 show `_robot.set_joint_position_target()` and Articulation target fields
  receive the target (`joint_pos_target` diff `0.00000004rad`). But line 94
  reports `final_joint_error_max_deg=5.046748`, `final_target_tcp_error_m=0.023923`,
  `tcp_target_reduced=NO`, `joint_error_reduced=NO`, and `target_realized=NO`.
- Lines 95-107 restore/close the gripper and reach the close pose; line 107 is a
  sanity check that normal close-pose restoration can be commanded and reaches
  `target_realized=YES` for the close target.
- Line 108 reports CLOSE/latch reached with `grasped=YES`, still no transport.
- Lines 109-121 compare after CLOSE/latch through normal `env_step`. Lines
  110-112 again show the target is passed into `set_joint_position_target()` and
  Articulation target fields (`joint_pos_target` diff `0.00000004rad`). Line 121
  reports `final_joint_error_max_deg=5.044317`,
  `final_target_tcp_error_m=0.023842`, `tcp_target_reduced=NO`,
  `joint_error_reduced=NO`, and `target_realized=NO`.
- Lines 122-134 compare after CLOSE/latch through direct set+sim-step. Line 134
  reports `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
  but `max_realized_tcp_delta_m=0.000098`,
  `final_joint_error_max_deg=5.046912`, and `target_realized=NO`.
- Line 135 aggregates:
  `before_target_realized=NO`, `after_env_target_realized=NO`,
  `after_direct_target_realized=NO`, `before_vs_after_split=NO`,
  `direct_rescues=NO`, `post_latch_target_delivery_blocker=YES`, and
  `general_grasp_pose_target_delivery_blocker=YES`.
- Lines 136-137 report the diagnostic completed its gates. This is diagnostic
  success only, not attach/transport/release/P7 success.

stderr:

- Lines 1-4 are the known cpufreq/NVML/Fabric warnings. There is no Python
  traceback.

## Interpretation

- The post-latch target buffer preservation result is confirmed at a lower level:
  `_robot.set_joint_position_target()` is called with the watched target, and
  Articulation data target fields including `joint_pos_target` match the target
  within `4e-8rad`.
- The failed micro-motion probe is not offset-preserve moving failure and not
  attach physics evidence.
- The blocker is broader than "post-latch only": the same 5deg shoulder nudge
  also fails before CLOSE at the grasp pose with gripper open and `_grasped=NO`.
- Direct set+sim-step after latch does not rescue the target, so the blocker is
  not just `env.step(null_action)` overwriting or skipping `_apply_action`.
- Current next diagnostic should isolate whether this is a grasp-pose/local-drive
  issue, a joint/drive stiffness/limit issue, or a state-cache/targeting issue by
  testing the same 5deg shoulder nudge earlier in the approach or at HOME/high
  with the same target-buffer instrumentation.

## Verification

- Local `py_compile` passed.
- B200 `py_compile` passed.
- Local and B200 md5 for the new script matched:
  `aad6398a9d47fef5c80efbd212e619d8`.
- B200 v3 run exit code was 0 and logs were line-checked above.
