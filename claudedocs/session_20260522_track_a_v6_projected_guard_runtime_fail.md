# Session 2026-05-22 - Track A v6 projected-guard runtime FAIL

## Scope

- Korean/user-requested Track A execution after v5 runtime FAIL and v6
  static/B200 readiness.
- Followed current-state constraints: no `HANDOFF.md` / `TASKS.md`, no dirty
  state rollback, no Track B evidence as Track A evidence.
- Ran exactly one approved B200 GPU0 close_26-only v6 projected-guard Isaac
  runtime and immediate v6 posthoc audit.
- Did not run v2/v3/v4/v5 reruns, PPO/training, rollout, dataset generation,
  hold-lift, transport/release, constraints, SurfaceGripper, or gate tuning.

## Preflight

- `git status --short --untracked-files=all` showed expected dirty/untracked
  state, including the three v6 scripts and rolling state docs. Nothing was
  reverted.
- B200 GPU check with NVML preload reported both B200 GPUs idle:
  GPU0 util 0, mem 0 / 183359 MiB; GPU1 util 0, mem 0 / 183359 MiB.
- B200 process check found no Track A Isaac/P7/PPO/training process.
- Local and B200 md5s matched:
  - runtime probe `e4d72390150a6660ce624d9ba1b4425d`
  - criteria audit `d30c4583c2efd20a9449885e58a5dd80`
  - readiness `821f523cf99bec4eedfb11016d977aa1`
- Local `py_compile` PASS and local readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.
- B200 `py_compile` PASS and B200 readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`.

## B200 v6 Runtime

Command shape came from readiness: GPU0, `variant=v7`, `close_deg=26.0`,
`--target_guarded_micro_close_v6_projected_guard_diagnostic`, no forbidden
Track A mechanisms.

- Runtime stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`, 430 lines, 957475 bytes.
- Runtime stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.err`
  md5 `947cab475a1eff6ad2f3ccea6505d8c4`, 3 lines, 377 bytes.
- Runtime command exit code was 0, but this is not success.

Key runtime lines:

- line 43: strict diagnostic-only, close_26-only, v6 flag YES, no forbidden
  mechanisms, no posewrite, no success claim.
- line 45: `mode=target_guarded_micro_close_v6_projected_guard_diagnostic`,
  separate approval marker YES.
- line 382 step 1 through line 384 step 3: first three micro-close advances
  happened with fixed gates inside budget.
- line 385 step 4: first recovery hold/recovery write, IK OK.
- line 393 step 12: v6 blocked advance when projection went unsafe:
  `projOK=NO`, projected support margin `-0.000063m`, recovery write YES,
  IK OK.
- line 394 step 13: still blocked advance, projected support margin
  `-0.000005m`, recovery write YES, IK OK.
- line 395 step 14: recovery still active but projected target/support margins
  were negative; fixed gates were still not yet breached (`target=0.002602m`,
  support gap `0.001908m`).
- line 398 step 17: first hard freeze/support-gate breach. Target error was
  `0.002914m` (still within 0.003m), but counter support gap was
  `0.002075m > 0.002m`; hard freeze YES.
- line 399 step 18: both fixed gates breached: target error `0.003052m`,
  support gap `0.002146m`.
- lines 427-428: posthoc FAIL summary: 4 advances, 41 holds, zero zero-backlog
  holds, zero safety rollbacks, 12 recovery writes, 0 IK failures, 29 hard
  freezes, close_reached NO, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

Runtime stderr was non-empty but not the audit cause:

- `cat: '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor': No such file or directory`
- `Could not get NVML device handle: ... NVML_ERROR_UNINITIALIZED`
- `Failed to clone in Fabric`

## Immediate Posthoc Audit

- Audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  md5 `480a3355864937763eb665e086aadbb0`, 58 lines, 22418 bytes.
- Audit stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.err`
  md5 `d41d8cd98f00b204e9800998ecf8427e`, empty.
- Audit command exit code was 1.

Key audit lines:

- line 18: `close_reached pass=NO`.
- lines 27-30: zero zero-backlog holds PASS, zero safety rollbacks PASS,
  recovery writes positive PASS, recovery IK failures zero PASS.
- line 31: hard safety freezes zero FAIL (`value=29`).
- lines 41-46: step3/step4/step5 early fixed gates and support horizon PASS.
- lines 47-49: no zero-backlog holds, every nonrollback hold preserves backlog,
  and no safety rollbacks all PASS.
- line 50: recovery holds present PASS.
- line 51: hard safety freeze criterion FAIL, first source runtime line 398.
- line 52: fixed target gate criterion FAIL, first source runtime line 399.
- line 53: fixed support budget criterion FAIL, first source runtime line 398.
- lines 54-56: preemptive recovery present PASS, preemptive trigger seen PASS,
  recovery IK OK all PASS.
- line 58: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## v5 Context Reverified

Before citing v5, the old B200 logs were recopied and rechecked:

- v5 runtime stdout md5 `f93ddaa75920a560777f8f9c8fae26f0`, 430 lines.
- v5 runtime stderr md5 `e492a73cfd22c900a9c79510db75d9e8`, 3 lines.
- v5 audit stdout md5 `7709c2bc37424bc7c3874e978b34d104`, 59 lines.
- v5 audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e`, empty.
- v5 line 394 advanced while support margin was too small, and line 395
  immediately breached both fixed gates. v6 fixed that specific unsafe advance.

## Verdict

FAIL. v6 is not grasp success.

The first failing runtime line is line 398. Category: support gate / hard freeze.
Line 399 then becomes target gate + support gate. This is not close_reached,
not recovery write failure, not IK failure, not zero-backlog, not safety rollback,
not attach/posewrite, and not metadata.

## Next

- Do not rerun v6 unchanged.
- Do not relax fixed target/support gates.
- Do not start PPO/training, rollout, dataset generation, hold-lift,
  transport/release, constraints, SurfaceGripper, or gate tuning from this
  result.
- Next Track A work is local/static/code-first: design active target/support
  recovery after a projected block, while preserving fixed gates, no
  attach/posewrite, zero zero-backlog holds, and zero safety rollbacks.
