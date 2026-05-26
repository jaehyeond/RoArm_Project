# Session 2026-05-22 — Track A v5 runtime FAIL and v6 projected-guard readiness

## Scope

- Korean/user-requested Track A work after B200 GPU0 became free.
- Followed current-state constraints: no `HANDOFF.md` / `TASKS.md`, no dirty
  state rollback, no Track B evidence as Track A evidence.
- Ran exactly one approved Isaac close_26-only v5 runtime and immediate audit.
- Did not run v6 runtime, PPO/training, rollout, dataset generation, hold-lift,
  transport/release, constraints, SurfaceGripper, or gate tuning.

## B200 v5 Runtime

- Runtime stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_v7_close26_b200.out`
  md5 `f93ddaa75920a560777f8f9c8fae26f0`, 430 lines.
- Runtime stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_v7_close26_b200.err`
  md5 `e492a73cfd22c900a9c79510db75d9e8`, 379 bytes.
- Audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v5_preemptive_recovery_audit_b200.out`
  md5 `7709c2bc37424bc7c3874e978b34d104`, 59 lines.
- Audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e` (empty).

Key runtime lines:

- line 43: strict diagnostic-only, close_26-only, v5 flag YES, no forbidden
  Track A mechanisms, no posewrite, no success claim.
- line 45: `mode=target_guarded_micro_close_v5_preemptive_recovery_diagnostic`,
  separate approval marker YES.
- line 393: recovery writes total 8, IK failures 0, support margin `0.000335m`.
- line 394: v5 still allowed an advance while support margin was `0.000243m`.
- line 395: first hard freeze: `target_error_m=0.003008 > 0.003`, counter gap
  `0.002146m > 0.002`.
- lines 427-428: posthoc FAIL, 5 advances, 40 holds, zero zero-backlog holds,
  zero safety rollbacks, 8 v5 recovery writes, 0 IK failures, 32 hard freezes,
  close_reached NO, attach/posewrite zero, telemetry-only YES, success_claim NO.

Key audit lines:

- line 17: `close_reached pass=NO`.
- line 30: `target_guarded_v5_hard_safety_freezes_zero pass=NO value=32`.
- lines 52-54: hard freeze / fixed target / fixed support FAIL from runtime
  lines 395-426.
- line 56: `target_guarded_v5_preemptive_trigger_seen pass=NO`.
- line 59: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## v6 Static Readiness

v5 was useful but insufficient: recovery writes happened and IK was OK, yet the
next micro-close advance was issued too late. v6 adds a default-off projected
advance guard. Before issuing the next advance it estimates target/support
margin after the next close step using recent target-error and support-gap
degradation scaled by command backlog. If the projected margin would go
negative, v6 holds backlog and applies recovery instead of advancing.

Static recomputation over the v5 B200 log:

- line 390: projected support margin `0.000412m`, v6 advance OK.
- line 391: projected support margin `0.000253m`, v6 advance OK.
- line 392: projected support margin `0.000016m`, v6 advance OK.
- line 393: projected support margin `-0.000062m`, v6 advance blocked.
- line 394: projected support margin `-0.000003m`, v6 advance blocked.

Code md5s synced and verified local/B200:

- Runtime probe `e4d72390150a6660ce624d9ba1b4425d`.
- Criteria audit `d30c4583c2efd20a9449885e58a5dd80`.
- Readiness `821f523cf99bec4eedfb11016d977aa1`.

Verification:

- Local `py_compile` PASS.
- Local synthetic v6 audit PASS.
- Local readiness `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- B200 md5s match local.
- B200 `py_compile` PASS.
- B200 readiness `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- B200 GPU0/GPU1 idle after work; no Track A runtime process left.

## Next

- Next runtime candidate is v6 close_26-only on B200 GPU0, followed immediately
  by v6 audit.
- This requires separate approval. Until then, v6 is only static/B200 readiness,
  not runtime evidence and not grasp success.
