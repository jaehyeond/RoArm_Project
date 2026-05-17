# START_HERE.md

Last updated: 2026-05-18 KST (Track A Branch B post-close handoff-model probe; no constraint integration)

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

- `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
  md5 `938a94b3b856dcc5a48527991a87c1e9`.
- The probe reuses the conservative stream and gated scheduling, executes only
  `PRE_MOVE* -> CLOSE`, then holds the same grasp pose. It compares diagnostic-
  local handoff models around CLOSE: current TCP-center pose-write, marker-only,
  delayed pose-write after 3 stationary env steps, one-shot align, and continuous
  TCP-offset-preserving pose-write. It does not insert constraint prims,
  integrate fixed/dynamic constraints, attach SurfaceGripper, execute attached
  transport, run release, run P7 training, or edit env/train/chain defaults.

B200 evidence:

- Logs: B200
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_{posewrite_tcp,marker_only,delayed_posewrite,oneshot_align,offset_preserve_posewrite}_v2_b200.{out,err}`.
- B200 stdout line 41 confirms scope: post-close handoff-model only, no
  constraint prim insertion, no fixed/dynamic integration, no SurfaceGripper, no
  attached transport, no release marker, no P7 training, no default edits, and
  `attach_physics_validated=NO`, `release_physics_validated=NO`,
  `claim_attach_success=NO`.
- Line 43 confirms the source stream shape and truncation:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 71 confirms HOME FK and settled sponge baseline:
  `home_fk_error_m=0.001894`, settled sponge
  `(+0.266020, -0.034486, +0.023500)`, and
  `settled_upright_z=1.000000`.
- Line 81 reports CLOSE reached in 15 steps:
  `gripper_q_deg=+23.02`, `d_tcp_sponge_m=0.023599`,
  `sponge_xy_drift_m=0.000005`, `min_upright_z=1.000000`,
  `latch_seen=YES`, `latch_step=15`.
- Line 82 shows the latch step itself was quiet:
  `pose_jump_m=0.000000`, `d_tcp_sponge_jump_m=0.000000`,
  `quat_angle_deg=0.000`, `latch_global_step=275`, threshold step `275`.
- Current TCP-center pose-write baseline still fails on first hold step
  (`posewrite_tcp` line 83):
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `xy_drift_m=0.006564`,
  `sponge_speed_mps=1.696947`, `sponge_ang_speed_rps=17.195574`,
  `quat_angle_deg=21.267`, `early_kill=YES`.
- Marker-only still passes 20 hold steps (`marker_only` lines 83-91) with
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_pose_drift_m=0.000000`, and success `YES`; this remains a negative
  control, not attach evidence.
- Delayed TCP-center pose-write passes the first 3 stationary env steps but fails
  when pose-write starts (`delayed_posewrite` line 86):
  `target_error_m=0.015686`, `tcp_step_m=0.016133`,
  `pose_drift_m=0.017553`, `quat_angle_deg=21.266`; lines 87-89 report
  `post_latch_hold_ok=NO`.
- One-shot TCP-center align still fails first hold step (`oneshot_align` line
  83): `target_error_m=0.005097`, `pose_drift_m=0.005682`,
  `sponge_speed_mps=0.823018`, `quat_angle_deg=7.080`; lines 84-86 report
  `post_latch_hold_ok=NO`.
- Continuous TCP-offset-preserving pose-write passes the stationary hold
  (`offset_preserve_posewrite` lines 83-91): `hold_max_target_error_m=0.000817`,
  `max_sim_tcp_step_m=0.001947`, `hold_max_pose_drift_m=0.000000`,
  `hold_max_offset_error_m=0.000001`, `hold_max_speed_mps=0.000869`,
  `posewrite_calls=40`, `post_latch_hold_ok=YES`, success `YES`.

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
- This is **not P7 success** and **not constraint integration**. It does not
  validate object attachment physics, release physics, attached transport, or
  constraint insertion inside the chain. Offset-preserving stationary PASS is
  only a local kinematic handoff diagnostic.
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
- Do not change B200 system NVIDIA symlinks; use per-run
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

## Current Direction

Active pivot: Track A P7/Branch B, isolated/pre-integration mechanics and chain-side timing.

Next concrete action: do not integrate constraints yet and do not proceed to
transport/release claims. If continuing, test only the local CLOSE handoff model
around the offset-preserving candidate in another isolated diagnostic, still no
SurfaceGripper, no RoArm chain constraint insertion, and no attached-transport
or release physics claims unless explicitly approved.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D024-D036
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
6. `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`
7. `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`
8. `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`
9. `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`
10. `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`
11. The B200/local logs cited above, with line numbers
