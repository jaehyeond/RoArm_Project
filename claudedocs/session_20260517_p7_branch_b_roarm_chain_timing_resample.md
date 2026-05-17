# Session 2026-05-17 — P7 Branch B RoArm chain-side timing resample

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not use SurfaceGripper.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Did not add scripted release variants.
- Added only a local/numpy chain-side TCP resampling/timing diagnostic.

## Why This Follow-Up Was Needed

The prior RoArm chain-side dry-run proved a resampled attached
`grasp -> transport` command stream could satisfy the D030-style contract, but
it left two holes:

- It did not validate the full HOME→grasp pre-close path.
- Its aggregate transport target error used the planner target instead of
  separately stressing the final stream FK endpoint.

This session closes those dry-run holes without any Isaac or constraint
integration.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
- md5 `fe2b227d2a111bf1c7acfe82e8f43133`

Design:

- Directly loads `roarm_rl/chain_skills.py` by file path, so it does not import
  Isaac/gym through `roarm_rl/__init__.py`.
- Uses existing `TrajectoryPlanner` and `sim_scripts/roarm_kinematics.py`.
- Emits proposed `PRE_MOVE`, `CLOSE`, attached `MOVE`, `HOLD`, and `RELEASE`
  events only.
- Checks:
  - raw planner waypoint gaps against `0.010m`;
  - HOME→grasp pre-close resampling;
  - grasp→transport attached resampling;
  - realized FK TCP step size, not just requested target spacing;
  - final transport endpoint error from the actual resampled stream;
  - command order, release-after-target, and no-move-after-release.

## Runs

Local pass:

- `/tmp/p7_branch_b_roarm_chain_timing_resample_probe.out`
- `/tmp/p7_branch_b_roarm_chain_timing_resample_probe.err`

B200 pass:

- `/tmp/p7_branch_b_roarm_chain_timing_resample_probe_b200.out`
- `/tmp/p7_branch_b_roarm_chain_timing_resample_probe_b200.err`

No-margin failure cross-check:

- Local `/tmp/p7_branch_b_roarm_chain_timing_resample_probe_nomargin_fail.out`
- B200 `/tmp/p7_branch_b_roarm_chain_timing_resample_probe_nomargin_fail_b200.out`

Cross-machine verification:

- Local and B200 pass `.out` sha256 matched:
  `5518152feecd8e97c3132191048a4057e946e98c192ba6206c2d9bd6b491c240`.
- Local and B200 no-margin failure `.out` sha256 matched:
  `aa1b2b2d0539c29017c1ae640ae2040fcf7dd168de4d8a69fa72326bdec1a760`.
- All stderr files were empty.

## Evidence

Pass run, local and B200 identical:

- Line 2: `chain_side_only=YES`, `isaac_chain_integration=NO`,
  `constraint_prim_insertion=NO`, `surface_gripper=NO`, `p7_training=NO`,
  `env_default_edits=NO`, `chain_defaults_edits=NO`.
- Line 3: gates are `fk_error_gate_m=0.003000`,
  `endpoint_gate_m=0.003000`, `max_tcp_step_m=0.010000`,
  `resample_fraction=0.900`.
- Lines 11-16: raw planner gaps fail the 10mm step gate; max raw gap is
  `0.211271m` from HOME to high.
- Lines 17-64: HOME→grasp pre-close path is resampled into 38 `PRE_MOVE`
  commands with IK convergence YES and endpoint errors below `0.003m`.
- Line 65: `CLOSE` accepted with `target_reached=YES`.
- Lines 67-69: attached `MOVE` commands from grasp to transport all have
  `ik_converged=YES`; realized TCP steps are `0.007089`, `0.007648`, and
  `0.007691`.
- Line 70: final stream endpoint error to transport target is `0.000655m`.
- Lines 71-72: `HOLD` and `RELEASE` are accepted after target reached.
- Line 73: aggregate reports `preclose_cmds=38`, `attached_cmds=3`,
  `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`,
  `max_preclose_tcp_step_m=0.009525`,
  `max_attached_tcp_step_m=0.007691`,
  `max_preclose_fk_error_m=0.000997`,
  `max_attached_fk_error_m=0.000655`,
  `transport_final_error_m=0.000655`,
  `preclose_ik_failures=0`, `attached_ik_failures=0`.
- Lines 74-75: all stream/order/release gates pass and
  `ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS=YES`.

No-margin failure:

- Line 3: same external `max_tcp_step_m=0.010000`, but
  `resample_fraction=1.000`.
- Line 31: one HOME→high pre-close command has realized
  `tcp_step_m=0.010351` and `ok=NO`.
- Line 69: aggregate `max_preclose_tcp_step_m=0.010351`.
- Lines 70-71: `preclose_stream_ok=NO`, `command_order_ok=NO`, and
  `ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS=NO`.

## Interpretation

- The current planner/kinematics can produce a full chain-side dry-run TCP event
  stream only if raw planner gaps are replaced with conservative TCP
  resampling.
- Exact 10mm target spacing is not enough for a 10mm realized-step gate because
  FK/IK realization can slightly overshoot the intended TCP interval.
- This remains pre-integration evidence. It does not validate articulation
  dynamics, controller latency, TCP estimation in Isaac, contact, or
  attach/release timing.
- This is not P7 success and not chain-ready.

## Verification

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
  passed locally.
- Local pass run exit code was 0.
- B200 pass run exit code was 0.
- Local and B200 no-margin cross-checks both exited 2 as intended.
