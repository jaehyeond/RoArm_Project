# 2026-05-26 Track A v7 Local Runtime CUDA Blocked

## Scope

User explicitly approved the next Track A gate: close_26-only v7 active recovery runtime, followed immediately by v7 posthoc audit. B200 SSH/reconnect/pull was not used. No PPO, rollout, dataset generation, hold-lift, transport/release, constraints, SurfaceGripper, object attach, posewrite, or gate tuning was run.

## Commands Attempted

Direct local Python first failed before Isaac because the active base Python could not import `isaaclab`:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
  --variant v7 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --object_size_m 0.030 0.030 0.030 \
  --close_deg 26.0 \
  --log_every_close_step 1 \
  --target_guarded_micro_close_v7_active_recovery_diagnostic
```

The valid local IsaacLab entrypoint was then used:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py \
  --variant v7 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --object_size_m 0.030 0.030 0.030 \
  --close_deg 26.0 \
  --log_every_close_step 1 \
  --target_guarded_micro_close_v7_active_recovery_diagnostic
```

## Preserved Logs

Logs were copied into:

`claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/`

md5s:

- `runtime.out`: `7a965678e5e4a6eefa5867efb0f1f029`
- `runtime.err`: `d0a0ee1890efabeb1136fed63dd6b3aa`
- `audit.out`: `59249b5d77c28e2115fd367bc143ebef`
- `audit.err`: `d41d8cd98f00b204e9800998ecf8427e`
- `cuda_check.txt`: `f269e67af6fa04d8952c0dd65509f840`

## Result

This was not a valid physics runtime result. It reached IsaacLab metadata emission but failed before environment creation completed because local CUDA/NVIDIA driver access is unavailable.

Evidence:

- Runtime stdout line 28 selected `cuda:0`.
- Runtime stdout line 31 confirms strict v7 diagnostic scope: close_26-only, v7 active recovery enabled, no training, no constraints, no SurfaceGripper, no attach/transport/release, no gate tuning, no hidden posewrite, no success claim.
- Runtime stdout line 33 confirms expected mechanism `target_guarded_micro_close_v7_active_recovery_diagnostic`.
- Runtime stdout line 35 confirms v7 finite-difference TCP sweep contract, current object pose use, no object posewrite, and robot joint target writes only.
- Runtime stderr lines 9-16 report no CUDA-capable device and NVML driver not loaded.
- Runtime stderr lines 32-52 show the traceback from `gym.make("RoArm-Stack-Direct-v0")` to `RuntimeError: No CUDA GPUs are available`.
- CUDA check lines 1-3 show `nvidia-smi` failed with exit 9.
- CUDA check lines 12-13 show `torch_cuda_available False` and `device_count 0`.

The immediate posthoc audit was run against the produced stdout. It correctly failed:

- Audit line 3: required close steps `[2, 3, 4]` missing.
- Audit line 16: aggregate line missing.
- Audit line 18: step5 support-horizon evidence missing.
- Audit lines 29-30: no v7 active recovery present or triggered.
- Audit line 35: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

## Interpretation

Do not treat this as v7 contact success or v7 contact failure. It is an infrastructure block on the local machine. The v7 mechanism was not exercised because the simulation never reached close steps.

The next valid action is to run the exact close_26-only v7 active recovery command on a CUDA-valid local machine or RunPod IsaacLab/Isaac Sim environment, then immediately run the v7 posthoc audit. Dataset/training remain blocked.

## Continuation Prompt

```text
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
8. claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md
9. claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md
10. b200_backup_20260522_final/README_BACKUP.md
11. sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py
12. sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py
13. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py
14. sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py

Current Track A state:
- v6 close_26 audit FAIL, not grasp success.
- v6 runtime stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out
  md5 9a4f8825a88ee3c9d93d83e5b9a28b41
- v6 audit stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out
  md5 480a3355864937763eb665e086aadbb0
- Reverify runtime lines 43,45,393-399,427-428 and audit lines 18,27-31,51-58 before citing.
- Interpretation: v6 blocked unsafe projected advance but recovery writes did not reduce target/support margins. First support failure line 398; first target+support breach line 399.

Current v7 status:
- Default-off v7 active target/support recovery candidate is implemented and static-ready.
- It uses finite-difference TCP candidate sweep with current object pose and robot joint target writes only.
- Matching audit/readiness support and negative controls exist.
- Local static checks passed.
- Approved local runtime attempt on 2026-05-26 did NOT produce a physics result: local IsaacLab entered metadata, then CUDA failed before env creation.
- Preserved logs:
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.err
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/audit.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/cuda_check.txt
- Reverify runtime.out lines 28,31,33,35; runtime.err lines 9-16,32-52; audit.out lines 3,16,18,29-30,35; cuda_check.txt lines 1-3,12-13 before citing.

Professor pipeline decision:
- RL→expert→rollout→dataset→learning is valid only after Stage 0 no-attach contact primitive.
- Do NOT use existing RoArm-Pick/Stack PPO envs as clean Track A expert sources because they use kinematic attach / write_root_pose_to_sim.
- Do NOT start PPO/training/dataset/rollout first.

Next concrete step:
1. Run the exact close_26-only v7 active recovery runtime on CUDA-valid local/RunPod IsaacLab.
2. Use the local backup USD:
   b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd
3. Runtime command shape:
   OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py --variant v7 --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd --object_size_m 0.030 0.030 0.030 --close_deg 26.0 --log_every_close_step 1 --target_guarded_micro_close_v7_active_recovery_diagnostic
4. Immediately audit:
   python3 sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py --log <runtime_stdout> --expected_mechanism target_guarded_micro_close_v7_active_recovery_diagnostic
5. If close_26 audit fails, analyze the new line-level failure before any rerun.
6. If close_26 audit passes, do not jump to dataset/training; next gate is hold-lift PASS, then small pilot dataset/replay PASS.
```
