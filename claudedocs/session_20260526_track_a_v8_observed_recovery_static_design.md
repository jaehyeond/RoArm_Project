# Session 2026-05-26 - Track A v8 Observed-Recovery Static Design

## Scope

- Local/static only. No Isaac runtime, no GPU simulation, no hold-lift, no
  dataset, no PPO/training, no constraints, no SurfaceGripper, no attach, and no
  object posewrite.
- Goal: turn the post-reboot v7 close_26 failure into a v8 design contract before
  any new runtime approval.

## Inputs

- Runtime stdout:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out`
  md5 `621d00b9d157b4e70178c28f94ca4c7f`.
- v7 failure analysis:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v7_failure_static_analysis.out`
  md5 `0fbf57f32473fa253ee1082b888bdcb1`.

## Artifact

- Added
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v8_observed_recovery_static_design.py`
  md5 `56a382377b7fb0f0c6391bf59163af0d`.
- `py_compile` PASS.
- Saved output:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/v8_observed_recovery_static_design.out`
  md5 `c14e80ec5fc69c6e6e17925d61f81d0b`.

## Cross-Validation

- Output lines 1-2 verify both input md5s.
- Output line 3 finds the first projected reserve trigger at runtime line 386 /
  close step 9: target margin `+0.001750m`, support margin `+0.000990m`,
  projected target margin `+0.000784m`, projected support margin `+0.000373m`.
- Output line 4 identifies v7 first active recovery at runtime line 389 / step
  12, three close rows later than the projected reserve trigger.
- Output lines 5-6 identify runtime line 392 / step 15 as first support breach
  and first hard freeze.
- Output line 7 cross-checks that the projected reserve trigger is 3 steps before
  v7 first active and 6 steps before first support breach.
- Output lines 8-11 reject unchanged v7 by observed response: all three v7 active
  followups worsen target and support gap, with negative TCP follow ratios
  `-0.164`, `-0.117`, and `-0.089`.
- Output lines 12-20 report all static v8 checks as YES: input md5s verified,
  earlier trigger, reserve horizon, unchanged-v7 rejection by observed response,
  unchanged-v7 rejection by TCP follow, counter-contact geometry requirement,
  fixed gates preserved, and forbidden mechanisms forbidden.
- Output lines 21-26 define the v8 design contract. Output line 27 explicitly
  reports `RUNTIME_READY=NO`; line 28 reports `STATIC_V8_DESIGN_DONE=YES`.

## Interpretation

- v8 must not inherit the v7 mistake of treating candidate-level selected margins
  as observed recovery.
- v8 must trigger earlier from projected reserve depletion, track multi-step
  observed response, require actual TCP follow, and model counter-contact or
  counter-slop-contact restoration instead of only maximizing counter-gap
  reduction.
- This is not a physics result and not runtime approval. It is a static design
  gate for the next code step.

## Next

1. Implement a default-off v8 runtime candidate plus matching audit/readiness
   support, still preserving fixed target/support gates and forbidding
   attach/posewrite/constraints/SurfaceGripper/gate tuning.
2. Add static negative controls proving unchanged v7-style active rows are
   rejected by observed-response/TCP-follow checks.
3. Only after readiness passes and the user explicitly approves, run exactly one
   close_26-only runtime and immediately audit it.
