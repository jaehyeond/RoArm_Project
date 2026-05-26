# Track A v3 Progress Runtime FAIL - 2026-05-22

## Scope

Track A only: P7/Branch B normalized 3cm cube close_26 proxy. No Track B/OpenVLA
claim is evidence for this result.

Forbidden work was not performed: no dataset generation, no training, no
hold-lift, no transport/release, no constraints, no SurfaceGripper, no gate
tuning, and no success claim.

## Preflight

The first B200 command failed before Isaac because the B200 code workspace had
not yet received the v3 flag.

- preflight stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.preflight_argparse.err`
- md5 `acbbdfe97f41fe0a2130816a4c281d63`
- stderr line 59: unrecognized argument
  `--target_guarded_micro_close_v3_progress_diagnostic`
- preflight stdout was empty:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.preflight_argparse.out`
- md5 `d41d8cd98f00b204e9800998ecf8427e`

After rsync to B200:

- runtime probe md5 `9cdfd04876078110186435bb15ba34ab`
- audit md5 `22db78e81d25804cc6ed26ccbe608579`
- readiness md5 `b853e0f2198d1d005ac72bc1e83dafcd`
- B200 `py_compile`: PASS

After the v3 runtime FAIL, readiness was updated to block rerunning v3:

- readiness md5 `5675db108ac15de6f333caf2d2e9ce9d`
- B200 synced md5 `5675db108ac15de6f333caf2d2e9ce9d`
- readiness now prints `READY_FOR_SEPARATE_RUNTIME_APPROVAL=NO` and
  `future_runtime_command_status=HISTORICAL_DO_NOT_RERUN_V3`

## Runtime

Approved close_26-only v3 command ran once after the preflight code-sync fix.

- stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.out`
- stdout md5 `5f2d1a626edcdccce8086fafd321c9af`
- stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_v7_close26_b200.err`
- stderr md5 `13671d0ae55c7faee9ae90a4e8c242c6`
- runtime exit code: 0

Important stdout lines:

- line 37: diagnostic-only, Isaac run YES, v7, close_26-only, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim; v3 flag YES.
- line 39: object physics mode
  `target_guarded_micro_close_v3_progress_diagnostic`; separate approval YES.
- line 41: zero-backlog hold NO, v3 progress enabled YES, min actual progress
  `0.25deg`, max command backlog `5deg`, backlog preserve YES, support margin
  warning-only, hard support uses fixed budget, rollback only on safety
  degradation.
- line 379 step 3: early actual progress improved. `gripper_q_deg=1.019`,
  `target_guarded_close_advances_total=3`, zero-backlog holds total 0,
  `virtual_support=YES`, `virtual_damping_active=YES`,
  `target_guarded_v3_actual_progress_deg=0.657`.
- line 380 step 4 and lines 381, 383, 384: backlog-preserved holds occur instead
  of zero-backlog holds when projected backlog exceeds the 5deg cap.
- line 387 step 11: first safety rollback. `gripper_q_deg=6.890`,
  `gripper_command_deg=10.000`, command backlog `3.111deg`,
  `target_error_m=0.002769` > design limit `0.0027`; object speed
  `0.001647m/s`, support budget YES, support horizon YES, one-sided push NO.
- line 391 step 15: last advance. `gripper_q_deg=6.878`,
  `target_error_m=0.002698`, safety rollbacks total already 4.
- line 392 step 16: peak target/support excursion. `gripper_q_deg=7.235`,
  `target_error_m=0.003070` > fixed target gate `0.003`,
  counter gap max `0.002074m` > fixed support budget `0.002m`.
- line 421 step 45: final plateau. `gripper_q_deg=7.144`,
  `gripper_command_deg=7.147`, remaining close `18.856deg`,
  `target_error_m=0.002872`, support budget YES, support horizon YES,
  safety rollback YES.
- line 422: `future_close26_posthoc_pass=NO`, advances 6, holds 39,
  zero-backlog holds 0, backlog-preserved holds 5, safety rollbacks 34.
- line 423: `close_reached=NO`, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

## Audit

Immediate posthoc audit was run on the runtime stdout.

- final audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v3_progress_audit_b200.out`
- md5 `ca60c09b03a156c85197e34ec7b28bb5`
- final audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e`
- audit exit code: 1

Important audit lines:

- lines 4-11: metadata matches v3 and other candidate modes are disabled.
- line 15: `close_reached pass=NO`.
- line 23: backlog-preserved holds positive PASS, value 5.
- line 24: zero-backlog holds zero PASS, value 0.
- line 25: safety rollbacks zero FAIL, value 34.
- lines 26-41: step3 speed/support/damping and step4/step5 support gates PASS.
- lines 42-43: v3 no-zero-backlog and every nonrollback hold preserves backlog PASS.
- line 44: v3 no safety rollbacks FAIL; sources are runtime lines 387-390 and
  392-421.
- line 46: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

An initial audit output had one noisy criterion requiring step3 specifically to
be a backlog-preserved hold. The actual runtime safely advanced at step3, so the
audit was corrected to remove that step-position assumption. This was not a
physical gate relaxation: the final audit still fails on close_reached and safety
rollbacks.

## Static Attribution

Added:

- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v3_progress_runtime_static_analysis.py`
- md5 `b3c446c1872127b19b49af929ded95ce`

Verification:

- `python -m py_compile`: PASS.
- Static run verifies stdout md5 `5f2d1a626edcdccce8086fafd321c9af` and audit md5
  `ca60c09b03a156c85197e34ec7b28bb5`.
- Static run reports: 45 close steps, 6 advances, 39 holds, zero zero-backlog
  holds, 5 backlog-preserved holds, 34 safety rollbacks.
- Primary attribution:
  `target_pose_error_safety_rollback_after_progress`.
- Secondary: support-budget breach after target-error overshoot.

## Interpretation

v3 fixed the specific v2 zero-backlog starvation bug. That is real progress, but
not close_26 success.

The new blocker is target-pose safety rollback after actual close progress. Once
the jaw reaches roughly 6.9-7.2deg, the fixed TCP target/contact geometry becomes
incompatible with further close under the current primitive: target error crosses
the v3 design limit and briefly crosses the fixed 0.003m target gate and 0.002m
support budget.

Do not relax target/support gates. Line 392 already shows why: the failure is not
a harmless threshold artifact; it is the contact primitive violating the fixed
proxy contract.

## Next Step

No hold-lift, dataset generation, training, transport/release, constraints,
SurfaceGripper, or gate tuning.

Next Track A work should be local/static/code-first: design a v4
contact-compatible close or target-error recovery mechanism that preserves fixed
close_26 gates, keeps zero-backlog holds at zero, and proves no safety rollback
before any future close_26-only runtime request.
