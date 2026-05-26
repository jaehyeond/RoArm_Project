# Session 2026-05-26 - Track A Stage 0 to dataset step plan

## Scope

- Korean user asked to preserve the step-by-step path from current Track A truth
  to a valid large dataset and learning pipeline after compaction.
- No Isaac runtime, PPO/training, rollout collection, dataset generation,
  hold-lift, transport/release, constraints, SurfaceGripper, gate tuning, B200
  SSH, B200 reconnect, extra pull, or `.ssh` copying was attempted.
- This is a local/static planning and state-recording session.

## Reverified Truth

- `START_HERE.md` says Track A dataset generation and training are blocked until
  close_26 proxy audit PASS, then hold-lift PASS, then small pilot dataset/replay
  PASS. It also says existing default Pick/Stack PPO envs use kinematic attach /
  posewrite and must not be used as Track A no-attach expert evidence.
- v6 close_26 is FAIL, not grasp success. Local backup md5s were reverified:
  - runtime stdout
    `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
    md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`
  - audit stdout
    `b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
    md5 `480a3355864937763eb665e086aadbb0`
- Runtime line 393 is the first pre-freeze projected block with recovery write
  and IK OK. Runtime line 398 is the first support-gate hard freeze while target
  is still inside the 3mm fixed gate. Runtime line 399 breaches both fixed
  target and support gates. Runtime lines 427-428 report close_reached NO,
  zero zero-backlog holds, zero safety rollbacks, 12 recovery writes, 0 IK
  failures, 29 hard freezes, attach/posewrite zero, telemetry-only YES,
  success_claim NO.
- Audit line 18 fails close_reached; line 31 fails hard-freezes-zero with value
  29; lines 51-53 fail hard freeze / fixed target / fixed support; line 58 says
  `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Stage 0 preflight remains valid: the professor/user pipeline is correct in
  principle - RL learns, policy becomes expert, expert rollouts record demos,
  demos become LeRobot/RLDS - but only after a Track A-valid no-attach contact
  primitive exists.
- `roarm_rl/train_ppo.py` still targets `RoArm-Pick-Direct-v0` /
  `RoArm-Stack-Direct-v0`; `roarm_rl/roarm_pick_env.py` and
  `roarm_rl/roarm_stack_env.py` still use kinematic attach /
  `write_root_pose_to_sim`, so direct PPO there is not Track A-valid no-attach
  expert evidence.

## New Static Design Artifact

Added:

- `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  md5 `14a462526945f3c5bca1c5e8c3e13525`

Verification:

- `python3 -m py_compile sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  PASS.
- `python3 sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  PASS and prints `RUNTIME_READY=NO STATIC_DESIGN_DONE=YES`.
- `git diff --check -- sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  PASS.

Key static output:

- First projected block: runtime line 393, step 12,
  `projected_support_margin_m=-0.000063`, target margin `0.000836`,
  support margin `0.000335`, recovery write YES, IK OK.
- Last recovery before freeze: runtime line 397, step 16,
  projected advance OK but support margin only `0.000016`, target not recovered,
  recovery step `0.001500m`.
- First hard freeze: runtime line 398, step 17, target error `0.002914m`
  still within the 3mm gate, but support gap `0.002075m > 0.002m`.
- First target violation: runtime line 399, step 18, target error `0.003052m`.
- During pre-freeze recovery rows, target error grew `0.000626m` and support gap
  grew `0.000319m`. Therefore v6 had recovery writes, but they did not actively
  reduce target/support error.

## Valid Step-By-Step Path

1. Stage 0-A - static active recovery design:
   v6 target-only overshoot recovery is rejected. The next candidate should enter
   active recovery after projected block and use a finite-difference TCP sweep
   with the current object pose. The selector objective should maximize the
   minimum of fixed target margin and fixed support margin.

2. Stage 0-B - runtime probe plus audit contract:
   Add a default-off v7 active-recovery diagnostic flag. Preserve no attach,
   no object posewrite, no constraints, no SurfaceGripper, fixed target/support
   gates unchanged, zero zero-backlog holds, zero safety rollbacks, close_26-only
   first. Add audit criteria so hard freeze, fixed target/support breach,
   attach/posewrite, zero-backlog, rollback, or missing recovery all fail.

3. Stage 0-C - static/readiness only:
   Run py_compile, synthetic pass/fail, old v6 negative-control reject, and
   readiness. Do not launch Isaac yet. If readiness is not clean, fix the
   contract before asking for runtime approval.

4. Stage 0-D - one close_26 runtime plus immediate posthoc audit:
   Only after explicit user approval and only on local/RunPod, not B200. Success
   requires close_reached YES, hard freezes 0, fixed target/support violations 0,
   attach/posewrite 0, zero zero-backlog holds, zero safety rollbacks, positive
   recovery where needed, and audit final PASS.

5. Stage 1 - hold-lift gate:
   A close_26 PASS is not yet a dataset gate. Run a no-attach hold-lift check:
   object remains supported, no slip/posewrite/attach, target/support gates stay
   within budget. If this fails, return to Stage 0 mechanism design.

6. Stage 2 - no-attach RL env:
   Build or adapt an RL environment whose success does not depend on
   `_grasped` posewrite attach. The agent may write robot joint targets only.
   Privileged object state may be used for reward/audit, but not as final VLA
   observation evidence.

7. Stage 3 - random sanity before PPO:
   Run random or scripted-action sanity in the no-attach env. Confirm reward
   cannot be gamed by attach, posewrite, hidden latches, or gate relaxation.
   Only then run a tiny PPO smoke.

8. Stage 4 - PPO expert:
   Train small first, with explicit metrics: close_26 pass rate, hold-lift pass
   rate, hard-freeze count, attach/posewrite count, rollout replay pass rate,
   failure modes. Scale PPO only after smoke metrics are physically valid.

9. Stage 5 - expert rollout to pilot dataset:
   Collect a small pilot dataset first, not mass data. Include observations,
   actions, state, audit summaries, success/failure labels, and replay checks.
   Pilot replay must pass before large-scale generation.

10. Stage 6 - large dataset:
    Scale across pose/yaw/object/friction/mass/camera domains only after pilot
    replay PASS. Each shard needs audit summaries and failure accounting.

11. Stage 7 - learning:
    Train BC/VLA/IL after data validity is established. Do not treat Track B
    OpenVLA status as Track A contact evidence. Use camera+proprio style inputs
    that can exist on the real robot; keep privileged sim state out of final VLA
    observations unless explicitly studying oracle baselines.

## Critical Interpretation

The professor's method is correct as a pipeline, but only after Stage 0.
Starting PPO/training/dataset now would generate expert evidence from an invalid
contact model. The next concrete implementation task is v7 active target/support
recovery in the runtime probe, plus audit/readiness hardening, still local/static
first.

## Continuation Prompt

Paste this in the next compacted/new session:

```
Read CLAUDE.md first, then follow Current-State Protocol exactly.

한국어로 브리핑하고, 비판적/분석적으로 진행.
기억만으로 말하지 말고 반드시 파일/라인과 로컬 백업 로그 라인을 확인.
HANDOFF.md / TASKS.md 사용 금지.
기존 dirty/untracked 상태를 임의로 되돌리지 말 것.
B200은 만료/disconnect 상태다. 절대 ssh JHPark / B200 재접속 / 추가 pull / .ssh 복사 시도 금지.
local + RunPod + 로컬 백업만 사용.

Start by running:
git status --short --untracked-files=all

Must read:
1. CLAUDE.md
2. START_HERE.md
3. claudedocs/DECISIONS.md D083-D089
4. claudedocs/EXPERIMENT_LEDGER.md latest rows
5. claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md
6. claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md
7. claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md
8. b200_backup_20260522_final/README_BACKUP.md
9. sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py
10. sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py
11. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py
12. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py

Current Track A truth:
- v6 close_26 audit FAIL, not grasp success.
- Local backup v6 runtime:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out
  md5 9a4f8825a88ee3c9d93d83e5b9a28b41
- Local backup v6 audit:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out
  md5 480a3355864937763eb665e086aadbb0
- Reverify runtime lines 43,45,393-399,427-428 and audit lines 18,27-31,51-58 before citing.
- Interpretation: v6 blocked unsafe projected advance but failed because recovery writes did not reduce target/support margins; first support failure line 398, first target+support failure line 399.

Professor pipeline decision:
- RL->expert->rollout->dataset->learning is valid only after Stage 0 no-attach contact primitive.
- Do NOT use existing RoArm-Pick/Stack PPO envs as clean Track A expert sources because they use kinematic attach / write_root_pose_to_sim.
- Do NOT start PPO/training/dataset/rollout first.

Next concrete task:
1. Implement default-off v7 active target/support recovery candidate in runtime probe.
2. Use finite-difference TCP candidate sweep at projected block, current object pose, robot joint target writes only.
3. Objective: maximize minimum fixed target/support margin; reduce counter gap before next close advance while keeping target error inside fixed 3mm gate.
4. Add matching audit/readiness support and negative controls.
5. Run only local static checks first: py_compile, synthetic audit pass/fail, old v6 log reject, readiness.
6. No runtime unless explicitly approved; future runtime is close_26-only first and immediate posthoc audit.
7. Dataset/training remains blocked until close_26 PASS + hold-lift PASS + small pilot dataset/replay PASS.
```
