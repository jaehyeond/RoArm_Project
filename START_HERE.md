# START_HERE.md

Last updated: 2026-05-17 KST (Track A Branch B RoArm chain-side command-stream dry-run; no integration)

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

- `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`

What changed:

- Added `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
  md5 `d9a07b43bed44f6061144234d7f6ec36`.
- The probe is local/numpy-only and chain-side only. It converts existing
  `TrajectoryPlanner` waypoints into explicit
  `PRE_MOVE* -> CLOSE -> MOVE* -> HOLD -> RELEASE` events and validates the
  event stream. It does not run Isaac, insert constraint prims, use
  SurfaceGripper, change env/train/chain defaults, or integrate constraints.

B200/local dry-run evidence:

- Logs: local `/tmp/p7_branch_b_roarm_chain_command_stream_probe.{out,err}`;
  B200 `/tmp/p7_branch_b_roarm_chain_command_stream_probe_b200.{out,err}`.
- Line 2: `command_stream_only=YES`, `chain_side_only=YES`,
  `isaac_chain_integration=NO`, `constraint_prim_insertion=NO`,
  `surface_gripper=NO`, `p7_training=NO`, and no env/chain default edits.
- Line 3: `resample_fraction=0.900`; line 4 schema:
  `PRE_MOVE* CLOSE MOVE* HOLD RELEASE`.
- Lines 19-24: raw planner gaps still fail the `0.010m` step gate; max raw gap
  is `0.211271m` from HOME to high.
- No-margin cross-check failed:
  `/tmp/p7_branch_b_roarm_chain_command_stream_probe_nomargin_fail*.out`
  line 39 showed one `PRE_MOVE` step at `0.010351m`; lines 77-79 failed.
- With `resample_fraction=0.900`, line 73 accepts `CLOSE`, lines 79-80 accept
  `HOLD` and `RELEASE` only after target reached.
- Line 81 reports `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
  max steps `0.009525/0.007691m`, max FK errors `0.000997/0.000655m`,
  final error `0.000655m`, and zero IK failures.
- Lines 82-83: all stream/order/release gates YES and
  `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=YES`.

Interpretation:

- Current planner kinematics can provide a contract-compatible dry-run TCP
  event stream only with explicit conservative resampling.
- The existing raw planner waypoints/targets are too coarse, and exact 10mm
  resampling can still fail due to FK/IK realized-step error; use a safety
  margin before any approved integration design.
- This is **not P7 success** and **not constraint integration**. It does not
  validate articulation dynamics, controller latency,
  TCP estimation in sim, contact, or attach/release timing inside the chain.
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

Active pivot: Track A P7/Branch B, isolated constraint mechanics.

Next concrete action: do not integrate constraints yet. If continuing, design the
next falsifiable pre-integration check for real Isaac/RoArm chain dynamics,
controller latency, TCP-estimation timing, and attach/release timing, still
without constraint prim insertion unless explicitly approved.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D024-D032
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`
6. `claudedocs/session_20260517_p7_branch_b_roarm_chain_command_stream.md`
7. The B200/local logs cited above, with line numbers
