# Session 2026-05-26 - Track A v7 failure static analysis

## Scope

- Local/static analysis only after the post-reboot v7 close_26 runtime FAIL.
- No Isaac runtime, GPU command, PPO/training, rollout, dataset generation,
  hold-lift, transport/release, constraints, SurfaceGripper, gate tuning, B200
  SSH, B200 reconnect, extra pull, or `.ssh` copy was attempted.
- The goal was to separate the v7 failure into audit-contract, trigger-timing,
  candidate-model, actuator/TCP-follow, and contact-geometry domains before any
  v8 design or runtime approval.

## Inputs

- Runtime log:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out`
  md5 `621d00b9d157b4e70178c28f94ca4c7f`.
- Audit log:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out`
  md5 `406b96557d94418f16273e517ec4d69b`.
- Runtime line 8 confirms the v7 contract: current-object-pose finite-difference
  TCP sweep, object posewrite NO, robot joint target writes only, constraints NO,
  and SurfaceGripper NO.
- Audit lines 60-64 pass the v7-specific active-recovery checks, while audit
  line 19 fails close_reached, line 32 fails hard-freezes-zero, lines 54-56 fail
  hard freeze / fixed target / fixed support, and line 66 reports final FAIL.

## Artifact

Added:

- `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py`
  md5 `e13605f058cd1908ff3d863e8239fbc4`.

Verification:

- `python3 -m py_compile sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py`
  PASS.
- Analyzer output:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out`
  md5 `0fbf57f32473fa253ee1082b888bdcb1`.
- Analyzer final line 23: `ANALYZER_RESULT=PASS`.

## Findings

- Analyzer line 2 parsed 45 close rows from runtime lines 378-422.
- Analyzer line 3 found 3 v7 active rows.
- Analyzer line 5: first v7 active row is runtime line 389 / step 12, with
  target margin `+0.000599m` and support margin `+0.000312m`.
- Analyzer line 4: the first low-margin row is runtime line 391 / step 14, with
  target margin `+0.000173m` and support margin only `+0.000037m`.
- Analyzer lines 7 and 9: runtime line 392 / step 15 is the first support breach
  and first hard freeze. Target is still inside the fixed 3mm gate
  (`0.002962m`), but counter gap is `0.002048m > 0.002m`.
- Analyzer line 8: runtime line 393 / step 16 is the first fixed target breach
  (`0.003059m`) and still a support breach (`0.002104m`).
- Analyzer lines 11-13: every active row predicted a `-0.001500m` target-error
  improvement and a negative counter-gap delta, but the observed next close row
  worsened both target error and counter gap:
  - line 389 -> 390: observed target delta `+0.000249m`, observed gap delta
    `+0.000159m`, TCP follow ratio `-0.164`;
  - line 390 -> 391: observed target delta `+0.000177m`, observed gap delta
    `+0.000116m`, TCP follow ratio `-0.117`;
  - line 391 -> 392: observed target delta `+0.000135m`, observed gap delta
    `+0.000085m`, TCP follow ratio `-0.089`.
- Analyzer lines 15-20 classify the domains:
  - audit contract mismatch: NO;
  - trigger timing late: YES, only 3 steps from first active recovery to support
    breach;
  - candidate prediction mismatch: YES, worsening followups `3/3`;
  - weak TCP follow: YES, weak follow rows `3/3`;
  - contact geometry suspect: YES, moving contact active YES, counter contact
    active NO, max object drift only `0.000029m`;
  - hard safety lockout after active: YES, first hard step 15 after last active
    step 14.

## Interpretation

The failure is not that v7 failed to activate, nor that the audit missed v7
telemetry. The failure is that the selected v7 candidates were optimistic at the
candidate level but did not produce observed recovery in the next close rows.

The strongest current diagnosis is:

1. Trigger timing is late: v7 first activates at step 12 and support breaches at
   step 15.
2. The candidate model is insufficient: predicted target/support improvement is
   contradicted by the next observed rows in all 3 active followups.
3. The robot does not follow the selected recovery TCP in the intended direction
   within one close row; follow ratios are negative.
4. Contact geometry is suspect: the moving jaw remains in contact while counter
   jaw contact is absent and object drift is tiny, so support loss is dominated
   by jaw/TCP geometry response rather than object translation alone.
5. Once support budget is breached, the runtime hard-safety path disables v7
   recovery and the failure is locked into hard freezes.

## Next Valid Step

- Do not rerun v7 unchanged.
- Do not start hold-lift, PPO/training, rollout, dataset generation,
  constraints, SurfaceGripper, transport/release, or gate tuning.
- The next valid Track A work is static v8 design only. It must account for
  observed multi-step response, earlier trigger timing, actual TCP follow, and
  counter-contact geometry while preserving fixed target/support gates, no
  attach/posewrite, zero zero-backlog holds, zero safety rollbacks, and
  robot-joint-target-only writes.

## Sources

- `sim_scripts/p7_branch_b_cube2cm_v7_failure_analyzer.py`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out:1-23`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out:8,389-393,423-424`
- `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out:19,32,54-56,60-66`
