# START_HERE.md

Last updated: 2026-05-18 KST (Track A Branch B passive-contact/close-timing probe; no constraint integration)

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

- `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
  md5 `6cb899ca124ff588fcc011d2805fa605`.
- The probe reuses the conservative `PRE_MOVE* -> CLOSE -> MOVE* -> HOLD ->
  RELEASE` stream but executes only `PRE_MOVE* -> CLOSE` on the real Isaac/RoArm
  articulation with the sponge at nominal pick. It measures passive sponge drift,
  close timing, env `_grasped` marker timing, realized TCP steps, target error,
  and sponge uprightness. It does not insert constraint prims, integrate
  fixed/dynamic constraints, attach SurfaceGripper, execute attached transport,
  run release, run P7 training, or edit env/train/chain defaults.

B200 evidence:

- Logs: B200
  `/tmp/p7_branch_b_roarm_chain_passive_contact_close_timing_probe_b200.{out,err}`.
- B200 stdout line 41 confirms scope: passive-contact/close timing only, no
  constraint prim insertion, no fixed/dynamic integration, no SurfaceGripper,
  no attached transport, no release marker, no P7 training, no default edits,
  and env kinematic latch is marker-only.
- Line 43 confirms the source stream shape and truncation:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 71 confirms HOME FK and settled sponge baseline:
  `home_fk_error_m=0.001894`, settled sponge
  `(+0.266020, -0.034486, +0.023500)`, `settled_upright_z=1.000000`.
- Lines 72-80 show sampled PRE_MOVE events all reached under gated execution,
  with zero measurable sponge XY drift and no latch before close.
- Line 81 reports CLOSE reached in 15 steps:
  `gripper_q_deg=+23.02`, `d_tcp_sponge_m=0.023599`,
  `sponge_xy_drift_m=0.000005`, `min_upright_z=1.000000`,
  `latch_seen=YES`, `latch_step=15`.
- Line 82 aggregate: executed events `39`, `total_sim_steps=275`,
  `max_final_target_error_m=0.002399`, `max_sim_tcp_step_m=0.001947`,
  `max_preclose_sponge_xy_drift_m=0.000000`,
  `max_close_sponge_xy_drift_m=0.000005`, `latch_event_index=39`,
  `gripper_threshold_global_step=275`, `preclose_latch_seen=NO`,
  `kinematic_env_latch_is_marker_only=YES`.
- Lines 83-84 report scoped gates YES and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`.

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
- This is **not P7 success** and **not constraint integration**. It does not
  validate object attachment, release physics, attached transport, or constraint
  insertion inside the chain.
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
- Do not change B200 system NVIDIA symlinks; use per-run
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05` and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

## Current Direction

Active pivot: Track A P7/Branch B, isolated/pre-integration mechanics and chain-side timing.

Next concrete action: do not integrate constraints yet. If continuing, design the
next falsifiable pre-integration check after close: still no SurfaceGripper, no
RoArm chain constraint insertion, and no attached transport unless explicitly
approved.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D024-D033
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
6. `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`
7. `claudedocs/session_20260517_p7_branch_b_roarm_chain_dynamics_timing.md`
8. `claudedocs/session_20260518_p7_branch_b_passive_contact_close_timing.md`
9. The B200/local logs cited above, with line numbers
