# START_HERE.md

Last updated: 2026-05-18 KST (Track A Branch B post-close handoff micro-motion probe; no constraint integration)

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

- `claudedocs/session_20260518_p7_branch_b_handoff_micro_motion_probe.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
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
  and 8mm post-close `plus_x` perturbation targets were not reached. Treat this
  as a post-latch micro-command execution blocker, not as attach physics
  evidence and not as a successful moving handoff.
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
- Do not change B200 system NVIDIA symlinks; use per-run
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

## Current Direction

Active pivot: Track A P7/Branch B, isolated/pre-integration mechanics and chain-side timing.

Next concrete action: do not integrate constraints yet and do not proceed to
transport/release claims. If continuing, instrument or redesign the post-latch
micro-command executor so a tiny bounded TCP perturbation is actually realized,
then re-test marker-only vs offset-preserve locally. Still no SurfaceGripper, no
RoArm chain constraint insertion, and no attached-transport or release physics
claims unless explicitly approved.

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
11. `claudedocs/session_20260518_p7_branch_b_handoff_micro_motion_probe.md`
12. The B200/local logs cited above, with line numbers
