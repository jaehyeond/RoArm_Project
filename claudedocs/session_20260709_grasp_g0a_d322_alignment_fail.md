# D322 Grasp G0a Alignment Fail

이번 case의 신규 변수: [grasp pose geometry: base yaw alignment + asymmetric TCP offset]

## Scope

- Active case: G0a alignment only.
- Invariants: existing 10cm cube, fixed position `(0.30, 0.00)`, friction `1.5/1.2`, state-only, no render, no gripper close, no grasp, no lift, no RL/PPO.
- Non-goals respected: no cylinder spawn, no position randomization, no friction/material change, no VLA/RoArm/B200.

## Plan Installed

- Created `claudedocs/direction_20260708_grasp_pivot.md`.
- Added `Variable Ladder Protocol (D322~)` to `CLAUDE.md`.
- Created `claudedocs/BACKLOG.md`.
- Added G0a active-case truth to `START_HERE.md`.

## Implementation

- Added `sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`.
- The probe computes:
  - `base_yaw = atan2(cube_y, cube_x)`.
  - object-center offset from TCP toward moving jaw: `(D/2 - 8mm) = 42mm`.
  - side pre-approach 40mm outside the final alignment pose.
- The probe drives the live Isaac env with external joint targets, using kinematic IK for the target sequence, while keeping the gripper open.
- G0b gripper stall contract is present only as comments. It is not active.

## Runtime Notes

- First sandbox run failed because Isaac could not see CUDA/NVML.
- Host GPU reruns completed.
- No PPO/training/render was run.

## Primary Result

Source: `claudedocs/runtime_logs/grasp_track/g0a_d322/g0a_alignment_summary.json`

- verdict: `D322_G0A_ALIGNMENT_FAIL`
- pass all criteria: `0/10`
- hard failure: `true`
- failure counts:
  - TCP pose: `10/10`
  - base yaw: `0/10`
  - fixed-jaw gap: `10/10`
  - fixed-jaw penetration: `10/10`
  - cube displacement: `0/10`
- mean TCP pose error: `96.63mm`
- mean fixed-jaw signed face gap: `-56.00mm`
- mean cube XY displacement: `0.019mm`
- mean max arm joint tracking error: `0.174rad`

## Long-Hold Diagnostic

Source: `claudedocs/runtime_logs/grasp_track/g0a_d322_longtrack_check/g0a_alignment_summary.json`

This was a sidecar diagnosis only, run with `500` approach steps + `500` hold steps + `12s` episode length.

- verdict: `D322_G0A_ALIGNMENT_FAIL`
- pass all criteria: `0/10`
- failure counts match the primary run.
- mean TCP pose error: `96.49mm`
- mean fixed-jaw signed face gap: `-55.78mm`
- mean cube XY displacement: `0.019mm`
- mean max arm joint tracking error: `0.174rad`

## Interpretation

This is a useful failure. Base yaw alignment passes, and cube displacement is effectively zero, so the failure is not the old tap overshoot/contact-rich displacement problem.

The failure is in the side-alignment pose contract:

- the commanded low side-alignment TCP target is not reached in live Isaac;
- longer hold does not materially improve the error;
- the current fixed-jaw face proxy reports large penetration, which may mean the TCP/fixed-jaw/tool-surface proxy does not match the USD tool geometry or that the chosen TCP target is infeasible under the current actuator/pose contract.

Do not respond by changing friction, adding cylinder spawn, closing the gripper, starting PPO, or adding broad controller conditions. The next task is G0a repair only: verify the actual USD jaw/link frames and choose the correct alignment target representation.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_d322_grasp_g0a_alignment_probe.py`: pass.
- `git diff --check`: pass before and after runtime edits.
- No Isaac/PPO/TensorBoard/torchrun residual process was found by `pgrep`.
- GPU was 0% before runtime. After runtime, `nvidia-smi` still reported 19%
  utilization, but the compute-app table was empty and only Xorg graphics
  processes were listed. This is not a residual Isaac/PPO/TensorBoard/torchrun
  process.

## Verdict

`D322_G0A_ALIGNMENT_FAIL`
