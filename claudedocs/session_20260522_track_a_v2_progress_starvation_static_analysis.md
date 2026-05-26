# 2026-05-22 Track A v2 Progress Starvation Static Analysis

## Scope

Track A only. No Isaac runtime was launched. No dataset generation, training,
hold-lift, transport/release, constraints, SurfaceGripper, gate tuning, or
success claim was performed.

The analysis uses the already-produced B200 target-guarded v2 close_26 stdout:

- `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v2_convergence_v7_close26_b200.out`
- md5 `52fa5cf2cc0cc5dbdc2f55f0d099611f`

## Code Added

Added:

- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v2_progress_static_analysis.py`
- md5 `7269b126b9aa1b6ce2da75e67f78702c`

The script is static-only. It reads the B200 log, verifies md5 by default, and
prints attribution for the v2 close starvation.

Verification:

- `python -m py_compile sim_scripts/p7_branch_b_cube2cm_target_guarded_v2_progress_static_analysis.py` PASS
- `python sim_scripts/p7_branch_b_cube2cm_target_guarded_v2_progress_static_analysis.py` PASS

## B200 Line Evidence

Runtime metadata:

- stdout line 37: diagnostic-only close_26-only run; no training, constraints,
  SurfaceGripper, transport/release, gate tuning, posewrite, or success claim.
- stdout line 39: mode
  `target_guarded_micro_close_v2_convergence_diagnostic`.
- stdout line 41: v2 uses a 2deg micro close step, 0.75deg command-error gate,
  0.0015m advance support margin, non-worsening target-error requirement, and
  zero-backlog holds.

Progress starvation:

- stdout line 377 step 1: first advance is issued at gripper actual `0.000deg`
  and command `0.000deg`.
- stdout line 378 step 2: the 2deg command produces only `0.361deg` actual
  gripper motion; command backlog is `1.639deg`, so v2 holds and zeroes backlog.
- stdout line 379 step 3: command is reset to actual (`0.361deg`), backlog is
  `0.000deg`, so v2 can advance again.
- The same pulse/reset pattern repeats until line 409 step 33, the last advance.
- Static extraction reports 17 advances and 28 holds, all holds zero-backlog.
- Average actual motion after an advance is `0.360deg`; average next-step
  backlog before zeroing is `1.641deg`, i.e. about `82.0%` of the 2deg micro
  command is discarded.

Blockers:

- stdout line 402 step 26: first target non-worsening block appears, but support
  margin is still YES and target error is only `0.000427m`.
- stdout line 410 step 34: first support-margin block; counter gap is
  `0.001583m`, only `0.000083m` above the v2 advance margin `0.0015m`.
  Fixed audit support budget is still `0.002m`, max plausible horizon is
  `0.003m`, target error is `0.001897m`, speed is `0.000898m/s`, support horizon
  YES, virtual support YES, one-sided push NO.
- stdout line 411 step 35: support margin becomes the only blocker with command
  backlog `0.002deg`, support horizon YES, virtual support YES.
- stdout line 421 step 45: final actual gripper is only `6.087deg`, command is
  `6.089deg`, close remaining is `19.913deg`, target error is `0.001921m`,
  speed is `0.000527m/s`, support horizon YES, virtual support YES, support
  margin NO.
- stdout lines 422-423: runtime posthoc remains FAIL; advances 17, holds 28,
  zero-backlog holds 28, close_reached NO, attach/posewrite zero, telemetry-only,
  success_claim NO.

Audit evidence:

- audit stdout md5 `563a9194dfc1cbe611aa38b9bee45dd3`.
- audit line 14: `close_reached pass=NO`.
- audit line 24: `virtual_support_step3 pass=NO`.
- audit line 41: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Static Output

The new static script reports:

- `advances=17`, `holds=28`, `zero_backlog_holds=28`;
- `advance_steps=[1, 3, 5, ..., 33]`;
- `avg_actual_motion_after_advance_deg=0.360`;
- `avg_next_step_backlog_before_zero_deg=1.641`;
- `discarded_fraction_of_micro_step=0.820`;
- even with the maximum alternating advance count in 45 steps (`23`), projected
  close would be only `8.279deg`, leaving `17.721deg` to the 26deg target;
- first support-margin block line 410 has only `0.000083m` margin excess while
  remaining inside the fixed support budget/horizon;
- attribution:
  `primary=zero_backlog_pulse_progress_starvation`,
  `support_margin_0p0015m=STRICT_AND_SECONDARY`,
  `support_margin_relaxation_alone=INSUFFICIENT`,
  `fixed_gate_relaxation=REJECT`,
  `structural_progress_guarantee_required=YES`.

## Conclusion

The primary failure is not that close_26 fixed audit gates are too strict. It is
that v2 turns each 2deg micro-close into a one-step pulse, then zeroes the
remaining backlog before the joint can converge. That makes 17 advances worth
only about 6deg of actual close motion.

The 0.0015m support margin is also too strict as an advance blocker because it
stops progress while the fixed 0.002m support budget and 0.003m support horizon
are still satisfied. However, relaxing that margin alone would not solve
close_26: with the current 45-step pulse/reset schedule, even the theoretical
no-margin-freeze projection is only about 8.3deg.

Next code-first direction: design a new default-off mechanism that preserves the
fixed close_26 audit gates but does not discard micro-close backlog after a
single physics step. It needs an explicit actual-progress guarantee: advance or
settle until the gripper makes bounded actual progress, and rollback only on
real safety degradation.
