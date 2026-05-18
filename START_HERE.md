# START_HERE.md

Last updated: 2026-05-18 KST (Track A Branch B approach-stage target-delivery probe; no constraint integration)

This is the rolling current-state dashboard. Do not treat it as full history.
Durable lessons live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

## Current Truth

The project is two-track:

- **Track A**: existing sim/lab stacking work. Current active line is P7/Branch B
  authored constraint mechanics, isolated/pre-chain units only.
- **Track B**: CoRL 2026 paper sprint. Keep separate unless the user explicitly
  asks to switch tracks.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## Track A Latest

Latest session:

- `claudedocs/session_20260518_p7_branch_b_approach_target_delivery.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
  md5 `ebe8eddafd4c6f35c28e5b79a82511b3`.
- This approach-stage target-delivery probe is diagnostic-local only: no
  constraint prim insertion, no fixed/dynamic integration, no SurfaceGripper, no
  attached transport, no transport target, no release, no P7 training, and no
  env/train/chain default edits.
- B200 v2 log:
  `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out`.
- Lines 41-43 confirm strict scope and that execution remains pre-transport:
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 72 records the controller context:
  `action_scale=0.100000`, `null_action_max_abs=0.000000`, and soft limits.
- The same +5deg shoulder nudge reaches `_robot.set_joint_position_target()` and
  Articulation target fields at every tested stage (`set_target_seen=YES`,
  `best_data_target_attr_diff_rad` `0.00000004-0.00000009rad`).
- HOME/early/high/hover realize the nudge under `env.step(null_action)`:
  line 87 HOME `target_realized=YES`, final nudge error `0.109396deg`;
  line 115 early PRE_MOVE `target_realized=YES`, final nudge error `0.106804deg`;
  line 143 high `target_realized=YES`, final nudge error `0.084780deg`;
  line 171 hover `target_realized=YES`, final nudge error `0.105476deg`.
- The grasp-before-CLOSE/open-gripper stage still fails despite target delivery:
  line 187 proves the target is nonzero and within soft/analytic limits
  (`expected_tcp_delta_m=0.024271`, limits OK); line 199 reports env-step
  `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
  `final_target_tcp_error_m=0.023947`, `final_nudge_joint_error_deg=5.042476`,
  `tcp_target_reduced=NO`, `nudge_joint_error_reduced=NO`,
  `target_realized=NO`, `grasped=NO`.
- Direct set+sim-step at the same grasp-before-CLOSE stage does not rescue:
  line 211 reports `set_target_seen=YES`, `max_realized_tcp_delta_m=0.000108`,
  `final_target_tcp_error_m=0.023927`, `final_nudge_joint_error_deg=5.044027`,
  and `target_realized=NO`.
- Line 213 aggregates `env_realized_stages=['settled_home', 'early_pre_move',
  'high', 'hover']`, `env_failed_stages=['grasp_before_close_open']`,
  `direct_rescue_stages=[]`, `home_high_realize_grasp_fails=YES`, and
  `latch_seen=NO`; line 214 reports `broader_command_realization_blocker=NO` and
  `local_grasp_pose_only_blocker=YES`.
- Therefore D037 is refined: the current blocker is not broad articulation target
  delivery/realization and not post-latch-only. It is a local grasp-pose command
  realization failure before CLOSE, with gripper open and `_grasped=NO`.
  Offset-preserve moving behavior remains untested.
- Previous target-delivery probe added
  `sim_scripts/p7_branch_b_roarm_chain_post_latch_target_delivery_probe.py`
  md5 `aad6398a9d47fef5c80efbd212e619d8`.
- Its B200 v3 log
  `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out` proved
  the same grasp-pose 5deg target was delivered before CLOSE (lines 83-85), after
  CLOSE/latch (lines 110-112), and by direct set (line 134), but was not realized
  in any of those grasp-pose comparisons. The new approach-stage probe refines
  that D037 result to a grasp-pose-local realization blocker.
- Previous executor probe added
  `sim_scripts/p7_branch_b_roarm_chain_post_latch_micro_executor_probe.py`
  md5 `c74d92816df12953c26fed577656840e`.
- B200 marker-only 4mm TCP micro target was nonzero but not realized:
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_marker_only_b200.out`
  line 87 reports `delta_q_norm_deg=0.790232` and
  `expected_tcp_delta_m=0.003511`; lines 88-93 show
  `robot_dof_targets` were not overwritten, but line 94 reports
  `realized_motion_seen=NO`, `executor_reached=NO`,
  `max_realized_tcp_delta_m=0.000080`, and success `NO` on lines 95-96.
- B200 marker-only 5deg joint-nudge cross-check also did not realize motion:
  `/tmp/p7_branch_b_roarm_chain_post_latch_micro_executor_joint_nudge_b200.out`
  line 87 reports `delta_q_max_abs_deg=5.000000` and
  `expected_tcp_delta_m=0.024271`; line 94 reports
  `targets_not_overwritten=YES`, but `realized_motion_seen=NO`,
  `executor_reached=NO`, `max_realized_tcp_delta_m=0.000206`,
  `min_joint_error_max_deg=4.992061`, and success `NO` on lines 95-96.
- Therefore the failed micro-motion result remains uninterpretable as
  offset-preserve moving behavior; the robot did not realize the commanded target.
- Previous micro-motion probe added
  `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
  md5 `a7ed4387e0ab1ce5b95de08f59c2eb52`.
- The probe reuses the conservative stream and gated scheduling, executes only
  `PRE_MOVE* -> CLOSE`, holds the grasp pose briefly, then attempts tiny TCP
  perturbations around the grasp pose. It compares current TCP-center
  pose-write, marker-only, and TCP-offset-preserving pose-write. It does not
  insert constraint prims, integrate fixed/dynamic constraints, attach
  SurfaceGripper, go to the transport target, run release, run P7 training, or
  edit env/train/chain defaults.

B200 evidence:

- Logs: B200
  `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_{posewrite_tcp,marker_only,offset_preserve_posewrite}_b200.{out,err}` and
  `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_d8mm_b200.{out,err}`.
- Line 41 in each stdout confirms scope: no constraint prim insertion, no
  fixed/dynamic integration, no SurfaceGripper, no attached transport, no
  release marker, no P7 training, no default edits, `transport_target=NO`,
  `micro_motion_not_transport=YES`, and `claim_attach_success=NO`.
- Line 43 confirms source stream truncation before MOVE:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Current TCP-center pose-write baseline still fails before micro-motion:
  `posewrite_tcp` line 83 reports `target_error_m=0.015684`,
  `tcp_step_m=0.016131`, `pose_drift_m=0.017552`,
  `sponge_speed_mps=1.696947`, `quat_angle_deg=21.267`; lines 85-87 report
  `post_latch_hold_ok=NO`, `micro_motion_ok=NO`, success `NO`.
- Marker-only passes the short stationary hold but does not reach the first 4mm
  micro target: lines 88-91 keep `target_error_m` around `0.004764-0.004765`,
  and lines 92-94 report `micro_events_done=1`, `micro_events_planned=4`,
  `micro_motion_ok=NO`, success `NO`. This remains a negative control, not
  attach evidence.
- Offset-preserving pose-write also passes the short stationary hold but does
  not reach the first 4mm micro target: lines 88-94 report
  `micro_max_target_error_m=0.004764`, `micro_motion_ok=NO`, success `NO`.
- An 8mm offset-preserve cross-check still does not reach the first micro target:
  d8mm lines 88-94 report `micro_max_target_error_m=0.008699`,
  `micro_motion_ok=NO`, success `NO`.

Interpretation:

- Current planner kinematics can provide a contract-compatible TCP event stream
  only with explicit conservative resampling.
- The existing raw planner waypoints/targets are too coarse, and exact 10mm
  resampling can still fail due to FK/IK realized-step error; use a safety
  margin before any approved integration design.
- The real Isaac/RoArm articulation can execute the conservative stream under
  a realized-TCP gated scheduler; a one-sim-step-per-command assumption remains
  false from the previous dynamics probe.
- Nominal passive approach/close timing did not show pre-close sponge push or
  pre-close env latch in this one-sponge diagnostic. CLOSE produced only an env
  kinematic latch marker at the gripper threshold step.
- The immediate latch marker step itself did not jump, but the first stationary
  post-close hold step under the current env kinematic attach boundary produced
  a large target/TCP violation, sponge pose drift, velocity spike, and quaternion
  change. Therefore the current env `_grasped` attach boundary is not a stable
  post-close handoff surface for chain transport.
- The attribution matrix points to `_update_grasp_attach` pose-write as the
  proximate trigger. Velocity mode and quaternion mode did not rescue the
  failure; disabling only pose-write while keeping the latch marker allowed the
  stationary hold to pass. This is marker-only evidence, not attach physics.
- The handoff-model matrix narrows the trigger further: snapping the sponge
  center to TCP is the bad local handoff geometry. Waiting before the same snap
  only delays failure, and one-shot center align still fails. Preserving the
  latch-time TCP-to-sponge offset avoids the stationary post-close hold failure.
- The micro-motion probe did not validate moving offset-preserve behavior:
  marker-only and offset-preserve both survived short stationary hold, but 4mm
  and 8mm post-close `plus_x` perturbation targets were not reached.
- The micro-executor probe showed target buffers were not overwritten by null
  action/action scaling. The follow-up target-delivery probe goes deeper:
  `_robot.set_joint_position_target()` and Articulation `joint_pos_target` receive
  the watched 5deg target, but the target is not realized before CLOSE, after
  CLOSE/latch, or with direct set+sim-step. Treat this as a grasp-pose target
  realization blocker, not attach physics evidence and not moving offset-preserve
  evidence.
- This is **not P7 success** and **not constraint integration**. It does not
  validate object attachment physics, release physics, attached transport, or
  constraint insertion inside the chain. Offset-preserving stationary PASS is
  only a local kinematic handoff diagnostic, and offset-preserving micro-motion
  is still unvalidated.
- Any actual RoArm chain integration still needs explicit user approval and a
  new falsifiable gate.

## Previous Track A Evidence To Preserve

- Previous mock chain-command contract passed:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`;
  B200 lines 129-131 report target errors `0.001468`, release drop `0.338178`,
  and all gates YES.
- Previous timing dry-run remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_timing_resample.md`.
- Mock-TCP interface and dynamic-anchor target tracking passed in isolation:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`,
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_target_tracking.md`.
- SurfaceGripper still must not be attached to the RoArm chain:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`; B200
  `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 111-113 and
  145-149 show canonical cuboid and RoArm sponge both fail Closed gates.
- Kinematic pose-write fixed-joint micro-move is killed:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_micro_move.md`; B200
  lines 59-71 show anchor motion while sponge stays, and lines 103-105 fail.
- Open-loop dynamic velocity anchor coupled but overshot about 2x:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_constraint.md`.
- Previous RoArm chain-side contract dry-run remains useful evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_contract_dryrun.md`.
- Previous command-stream dry-run remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`.
- Previous RoArm articulation timing remains core evidence:
  `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`.
- Previous passive close timing remains useful but is superseded by the
  post-close boundary failure:
  `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`.
- Previous stationary handoff-model matrix remains the source for D036:
  `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`.

## Track B Status

- Must-read: `claudedocs/session_20260517_corl2026_paper_track_pivot.md`.
- CoRL 2026 full paper deadline was estimated as 2026-05-28 AoE; user must verify
  on corl.org directly.
- Candidate paper pipeline remains separate from Track A unless explicitly merged.

## Do-Not-Repeat Rules

- Do not claim P7 success.
- Do not tune P7 scalar/threshold/release-guidance blindly.
- Do not run structured A curriculum long training from the killed smoke.
- Do not resume random SurfaceGripper parent/offset search.
- Do not add scripted release variants.
- Do not attach SurfaceGripper to the RoArm chain.
- Do not integrate fixed/dynamic constraints into the RoArm chain yet.
- Do not proceed from CLOSE into attached transport using the current env
  `_grasped` kinematic attach boundary as a valid handoff surface.
- Do not treat marker-only or offset-preserving stationary hold pass as attach
  physics, release physics, attached transport, or constraint validation.
- Do not treat the failed post-close micro-motion probe as evidence that
  offset-preserving attached MOVE is valid; the micro target was not reached.
- Do not treat the failed 5deg target-delivery probe as offset-preserve failure;
  the same grasp-pose nudge fails before CLOSE with `_grasped=NO`.
- Do not describe the current command-realization blocker as broad articulation
  targeting failure: HOME/early/high/hover realize the same +5deg shoulder nudge;
  the surviving failure is local to the grasp-before-CLOSE/open-gripper pose.
- Do not change B200 system NVIDIA symlinks; use per-run
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

## Current Direction

Active pivot: Track A P7/Branch B, isolated/pre-integration mechanics and chain-side timing.

Next concrete action: do not integrate constraints yet and do not proceed to
transport/release claims. If continuing, inspect why the same +5deg shoulder
nudge realizes at HOME/early/high/hover but fails at the grasp-before-CLOSE pose
with gripper open and `_grasped=NO`; likely candidates are local grasp-pose
drive/limit/contact/controller configuration around that pose. Only after grasp
pose command realization works, re-test marker-only vs offset-preserve locally.
Still no SurfaceGripper, no RoArm chain constraint insertion, and no
attached-transport or release physics claims unless explicitly approved.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D024-D038
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
6. `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`
7. `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`
8. `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`
9. `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`
10. `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`
11. `claudedocs/session_20260518_p7_branch_b_handoff_micro_motion_probe.md`
12. `claudedocs/session_20260518_p7_branch_b_post_latch_micro_executor.md`
13. `claudedocs/session_20260518_p7_branch_b_post_latch_target_delivery.md`
14. `claudedocs/session_20260518_p7_branch_b_approach_target_delivery.md`
15. The B200/local logs cited above, with line numbers
