# START_HERE.md

Last updated: 2026-05-17 KST (Track A Branch B mock-TCP interface PASS; still isolated/pre-chain)

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

- `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`

What changed:

- Added `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_interface_probe.py`
  md5 `eb81372d78828730e63879a996911bbd`.
- The probe is CPU-only and pre-chain. It wraps the target-tracked dynamic anchor
  in a mock TCP command interface: `anchor_target = tcp_target + tcp_to_anchor_offset`.
- It tests multi-waypoint target tracking and nonzero TCP-to-anchor offset mapping.
- It does not use RoArm chain integration, SurfaceGripper, P7 training, reward
  tuning, release guidance, scripted release variants, or launch-default changes.

B200 default-offset smoke:

- Logs: `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.{out,err}`.
- Lines 40-41: CPU/no chain/no transport/no SurfaceGripper/no P7 training.
- Line 48: close at `rel=0.000000`, `waypoints=3`.
- Lines 58, 75, 93: waypoint transforms have `transform_error=0.000000`.
- Lines 67, 85, 102: waypoint target-stop errors are `0.001411`, `0.001464`,
  and `0.001394`.
- Line 128: aggregate `max_move_rel=0.000000`, `max_hold_rel=0.000000`,
  `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`,
  `target_error_threshold=0.003000`, `release_drop=0.338178`.
- Lines 129-130: all gates YES and `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.

B200 nonzero-offset cross-check:

- Logs:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.{out,err}`.
- Line 48: nonzero `tcp_to_anchor_offset=([0.015, 0.0, -0.010])`.
- Lines 58, 75, 93: transformed mock TCP targets still map to anchor targets with
  `transform_error=0.000000`.
- Lines 67, 85, 102: waypoint target-stop errors again `0.001411`, `0.001464`,
  and `0.001394`.
- Line 128: aggregate again reports `max_move_rel=0.000000`,
  `max_hold_rel=0.000000`, max target errors `0.001468`, and
  `release_drop=0.338178`.
- Lines 129-130: all gates YES and `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.

Interpretation:

- Isolated dynamic-anchor target tracking now has a thin mock-TCP command surface.
- This is **not P7 success** and **not chain-ready**. It does not validate RoArm
  kinematics, IK, articulation dynamics, controller latency, or real chain contact.
- Any RoArm chain integration still needs explicit user approval and a new
  falsifiable gate.

## Previous Track A Evidence To Preserve

- Dynamic-anchor target-tracking unit passed before this interface wrapper:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_target_tracking.md`.
  B200 target logs line 83/102 and halfcmd line 81/100 show final target errors
  about `0.00143m`, `max_move_rel=0`, `max_post_move_rel=0`, and release success.
- SurfaceGripper is still not chain-ready:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`; B200
  `/tmp/p7_branch_b_surface_gripper_axis_object_smoke.out` lines 111-113 and
  145-149 show canonical cuboid and RoArm sponge both fail Closed gates.
- Kinematic pose-write fixed-joint micro-move is killed:
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_micro_move.md`; B200
  `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.out` lines 59-71 show anchor
  motion while sponge stays; line 103 reports `max_move_rel=0.022361`; lines
  104-105 report `FIXED_MICRO_MOVE_SUCCESS=NO`.
- Open-loop dynamic velocity anchor coupling passed but overshot:
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_constraint.md`; full and
  half B200 logs line 103 showed `rel=0` but actual displacement about 2x requested.

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

Next concrete action: decide whether to stop here for review or design one more
pre-chain falsification probe that includes a **mock chain-command contract**
without any RoArm articulation. Do not attach the constraint to the real RoArm
chain unless the user explicitly approves that transition.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D024-D029
4. `claudedocs/EXPERIMENT_LEDGER.md` latest Branch B rows
5. `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`
6. The B200 logs cited above, with line numbers
