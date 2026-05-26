# START_HERE.md

Last updated: 2026-05-26 KST (B200 disconnected; Track A v6 FAIL reverified from local backup; v7 active recovery implemented and static-ready; first local v7 runtime attempt was pre-reboot CUDA-blocked; local reboot fixed host CUDA; post-reboot close_26-only v7 active-recovery runtime ran locally and audit FAILED; do not rerun v7 unchanged; dataset/training still blocked)

This is the rolling current-state dashboard. It is not full history. Durable
rules live in `claudedocs/DECISIONS.md`; experiment history lives in
`claudedocs/EXPERIMENT_LEDGER.md`; detailed logs live in `claudedocs/session_*.md`.

Do **not** use `HANDOFF.md` or `TASKS.md` as current state.

## B200 Retired / Backup Truth

- NHN/Sogang B200 access expired on 2026-05-22 at 23:59 KST, and the user
  reported B200 now shows disconnect on 2026-05-23 KST. Future research must
  not depend on entering B200 through SSH or on B200-only paths.
- Do not copy, request, or depend on `.ssh` private material. We preserved
  research artifacts, logs, code snapshots, checkpoints, env specs, and wandb
  cache; not login secrets.
- Track A B200 evidence is locally preserved and verified: B200 `/tmp/p7_branch_b_*`
  ↔ `b200_backup_20260522_final/tmp_p7` has 494 files, path+size hash
  `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
  and file-content aggregate hash
  `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
- Track A B200 `sim_scripts` snapshot is locally preserved and verified:
  53 non-pycache files, path+size hash
  `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
  file-content aggregate hash
  `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
- Track B B200 outputs are locally preserved, but split across
  `b200_backup_20260522_final/outputs`, `b200_backup_20260521`, and
  `openvla_oft_b200_pulls`. Do not assume
  `b200_backup_20260522_final/outputs/openvla_oft_v6_b200` is complete; the
  complete OpenVLA full checkpoints live in `openvla_oft_b200_pulls`.
- Full verification details:
  `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
  and `b200_backup_20260522_final/README_BACKUP.md`.

## Current Truth

**Track A active line**: P7/Branch B normalized 3cm cube grasp primitive.

- Track A goal: first make the sim/Isaac Lab contact primitive reliable, then
  move toward broad sim/lab dataset collection and learning.
- Dataset generation and training are blocked until close_26 proxy audit PASS,
  then hold-lift PASS, then small pilot dataset/replay PASS.
- v4/v5/v6/v7 rigid/offset variants, soft-contact material-only, virtual
  compression+damping, and target-guarded v1 through v7 have all failed
  close_26 posthoc audit. None is grasp success.
- This is an Isaac proxy/contact primitive failure, not proof that the real robot
  cannot grasp the foam cube.
- The RL-to-expert-to-rollout-to-demo pipeline is valid only after a Track A
  Stage 0 no-attach contact gate exists. Existing default Pick/Stack PPO envs use
  kinematic attach / posewrite and must not be used as Track A no-attach expert
  evidence.
- Latest Track A B200 v6 `close_26` runtime is FAIL, not grasp success.
- 2026-05-26 step-plan update: the professor-style
  RL→expert→rollout→dataset→learning pipeline is still the right high-level
  path, but only after Stage 0 no-attach contact primitive PASS. Added local
  static design artifact
  `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
  md5 `14a462526945f3c5bca1c5e8c3e13525`; it reports that v6 pre-freeze recovery
  rows increased target error by `0.000626m` and support gap by `0.000319m`.
- 2026-05-26 v7 active-recovery code/readiness update: added default-off v7
  finite-difference TCP recovery, matching audit support, and readiness negative
  controls. Local static checks pass, including synthetic v7 PASS, v7 no-active
  recovery rejection, archived v6 log rejection as v7, and readiness
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. No runtime was run.
- 2026-05-26 approved local v7 runtime attempt was blocked by infrastructure, not
  physics: IsaacLab metadata emitted, but local CUDA/NVIDIA access failed before
  environment creation and no close step/aggregate lines were produced. The
  immediate audit correctly failed. Preserved logs live in
  `claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/`.
- 2026-05-26 RunPod/Codex continuation setup: Claude had RunPod MCP configured,
  but Codex did not. Added `[mcp_servers.runpod]` to
  `/home/cgxr/.codex/config.toml` from Claude's RunPod MCP config, with the
  `RUNPOD_API_KEY` value not printed. Backup:
  `/home/cgxr/.codex/config.toml.bak_runpod_20260526` md5
  `1ef4acf6f1c92a64b9bbd79a2e35b7e7`. Same-session `tool_search` still did not
  expose `mcp__runpod__...`, so each new Codex session must verify loaded tools
  before using RunPod MCP. A later Codex session did expose `mcp__runpod__...`
  and `list_pods` returned no GPU pods.
- 2026-05-26 post-reboot local CUDA update: user rebooted the local Ubuntu PC.
  Boot time now `2026-05-26 14:08`; host NVIDIA kernel/userspace now match at
  `580.159.03`. Host `nvidia-smi` and `conda run -n isaaclab` CUDA checks pass
  only when run outside the default Codex sandbox. The default Codex sandbox
  hides `/dev/nvidia*`, so sandboxed `nvidia-smi` still fails; this is not a host
  CUDA failure. v7 readiness still reports
  `READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES`. The old `/tmp` RunPod overlay is
  gone after reboot; recreate it if RunPod is needed. Top local backup USD md5
  remains `4497024d25abab11de5c50e144124553`.
- 2026-05-26 post-reboot v7 runtime/audit result: exactly one local
  close_26-only v7 active-recovery runtime ran with escalated Codex execution and
  immediate audit. This is a real physics/audit FAIL, not a CUDA infrastructure
  block. Logs:
  `claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/`.
  Runtime stdout md5 `621d00b9d157b4e70178c28f94ca4c7f`; audit stdout md5
  `406b96557d94418f16273e517ec4d69b`. Runtime lines 389-391 show v7 active
  recovery did trigger (3 writes, 0 IK failures, negative counter-gap deltas),
  but runtime line 392 is first support hard-freeze (`counter_gap=0.002048m >
  0.002m`, target `0.002962m` still inside gate), line 393 is first target+
  support breach (`target=0.003059m`, gap `0.002104m`), and line 424 aggregate
  has close_reached NO, 31 hard freezes, attach/posewrite 0, telemetry-only YES,
  success_claim NO. Audit line 19 fails close_reached; line 32 fails
  hard-freezes-zero; lines 54-56 fail hard-freeze/fixed-target/fixed-support;
  line 66 `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

**Track B/OpenVLA** is separate. Do not use Track B training or eval status as
Track A contact success evidence. Latest Track B P3 result remains: best deploy
ckpt = step 7500; steps 10000+ are collapsed and must not be deployed.
Track B data/continuation assets are backed up locally; P5 real robot deploy is
still pending local reboot/CUDA verification and user approval for robot motion.

**Track B Cube Task Pivot (2026-05-26, user-confirmed)**: sponge → **cube 3×3×3cm
× 5개 → 3+2 pyramid stacking** (L1=3, L2=2) 신규 task. Camera = Azure Kinect 고정
v6 동일 viewpoint. Sponge HARD RULES #19/#20/#24 자동 SUPERSEDED (HARD RULE #18
사용자 명시 정정 우선). Track A 직접 비교 → **sim demo 증강으로 재포지셔닝**
(Track A close_26 PASS 후 cube stacking sim demos co-training).

Hyperparam 갱신 (P3 7500→10000 collapse 회피): per_gpu_batch=8 + grad_accum=4 →
**effective batch=32** (vanilla OpenVLA-OFT LoRA 최소 권장치, 우리 P2 effective
8은 1/4였음). LR `5e-4` → **`2.5e-4`** (½, linear scaling 보수적). grad_clip_norm
=1.0, warmup 1K step, cosine. RunPod **A100 80GB**, 30K step ~8h ~$13.

데이터 신규 수집: **250ep (200 cube stacking + 50 cube pick), 일 50ep × 5일**,
ep당 ~400fr → 80K frames (v6 6942fr 대비 11.5×, task horizon 10× 근거).

7-phase plan: P0 cube+gripper calib (0.5일, sponge anchor 무효, cube 30mm 신규
측정) → P1 데이터 수집 (4일, mid γ-gate) → P2 LeRobot 변환 (0.5일) → P3 RunPod
학습 (1일) → P4 12-ckpt offline eval rank (0.5일) → P5 real multi-position
deploy (0.5일) → P6 Track A close_26 PASS 후 sim demo co-train (별개 trace) →
P7 비교 paper (1일). 상세:
`claudedocs/session_20260526_track_b_cube_task_pivot_plan.md`, ledger row 123.
v6 sponge ckpt 7500 deploy (P5 pending CUDA reboot)는 별개 보존, cube pivot과 무관.

**Track B P4 result — 2026-05-22 ~17:00 KST (deploy prep + offline + hw sanity
all PASS, real deploy pending CUDA reboot + openvla-7b 14GB download)**:

- Built `deploy_openvla_oft.py` 561 lines mirroring `deploy_smolvla.py` 4/9 Plan 3
  SUCCESS setup (INIT_POS [0,0,90,0,0,5] HOME, JOINT_SPEED_CAPS
  [500,500,500,300,300,300], gripper-only unlock pattern `arm.gripper_angle_ctrl(
  angle, speed=1000, acc=0)` directly after `joints_angle_ctrl`, Z_FLOOR=-130mm,
  DIST_MAX=420mm, Follower-only `--port /dev/ttyUSB0` blocked). Inference path
  replaces SmolVLA with OpenVLA-OFT (224×224 PIL RGB + language prompt, no state
  input, chunk (8,6) BOUNDS_Q99-denorm via `vla.predict_action`).
- Inline `L1RegressionActionHead` (deploy_openvla_oft.py:78-138) bypasses
  `prismatic.models.__init__` → `vlas` → `vla.materialize` → `dlimp` chain.
  See `claudedocs/DECISIONS.md` D086 for full rationale.
- Offline sanity 1+2+3 PASS (CPU only):
  1. Inline L1 head strict-load from B200 ckpt 7500 `action_head--7500_checkpoint.pt`
     after `module.` prefix strip: missing=0, unexpected=0, 134,328,326 params,
     forward (1,48,4096)→(1,8,6) OK.
  2. `dataset_statistics.json` key `roarm_v6_pick` q01/q99 for all 6 joints inside
     JOINT_LIMITS even at ±1.0 saturation.
  3. Script `ast.parse` PASS + 3 critical sub-imports OK.
- Hardware sanity 4+5 PASS:
  4. Kinect 720P NFOV_UNBINNED 1-frame capture (1280×720×3 BGR) →
     `logs/hw_sanity_20260522/kinect_sanity_frame.png`.
  5. Follower `/dev/ttyUSB1` (serial `ee7a06468e98ef1194edca63a8793231`, Leader
     USB0 serial `7842202ff8d9ef11b33f513dc8728757` per
     `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/tech_leader_follower_setup.md`)
     → torque ON → INIT_POS reached in 0.5s, max_diff=1.93°,
     FK pose x=353 y=2 z=204 mm (Z_FLOOR/DIST_MAX safe).
- Blockers for Step 6 real deploy:
  - CUDA driver mismatch `Failed to initialize NVML / NVML lib 580.159 / Error 804
    forward compatibility` → `torch.cuda.is_available()=False`. Fix = `sudo reboot`
    (no PC power-cycle needed).
  - `openvla/openvla-7b` HF cache 14 GB download in background at pinned revision
    `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, ~2 GB / 14 GB at session end.
- `roarm` conda env additions: `peft 0.18.0` (`--no-deps`), `rich 15.0.0`,
  `timm 0.9.16` (HARD RULE #15 pin), prismatic editable from
  `/home/cgxr/Documents/Robotics/openvla-oft/`.
- Full detail: `claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md`.
- DECISIONS: D086 OpenVLA-OFT local inference deps + inline action head pattern.
- Ledger row: 2026-05-22 (Track B P4 deploy prep ...) at line 118.
- Next session: `sudo reboot` → verify CUDA → resume snapshot_download → GPU
  dry-run sanity (1-chunk inference) → Kinect dry-run → real deploy
  (multi-position, head-to-head vs SmolVLA v6 4/9 Plan 3 SUCCESS baseline).
  Verbatim continuation prompt in P4 session doc.

**Track B P4.5 — 2026-05-22 ~19:00 KST (post-P4 verification, real deploy
aborted at Step 0, reboot still pending)**:

- Session entered under premise "reboot done after P4, proceed to GPU dry-run
  + real deploy". Premise verified **FALSE**: `uptime`=1d 20:02,
  `who -b`=2026-05-20 22:53. No reboot between P4 prep (today 2026-05-22) and
  this session.
- `nvidia-smi` still returns `Failed to initialize NVML: Driver/library
  version mismatch / NVML library version: 580.159`. Kernel module
  `580.126.09`, userspace `libnvidia-ml.so.580.159.03`. Same P4 Blocker (a).
- `conda run -n roarm python -c "import torch; print(torch.cuda.is_available())"`
  → `False` (Error 804 forward compatibility).
- `openvla/openvla-7b` HF cache 14 GB on disk (17 blobs) but
  `snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/` only shows
  `model-00003-of-00003.safetensors` symlink; 00001/00002 symlinks not
  finalized. Likely byte-complete in blobs; next session must re-run
  `snapshot_download` for idempotent fixup.
- No `deploy_openvla_oft.py` change, no env change, no Isaac, no RL, no robot
  command, no Track A file touched. User explicitly chose "Reboot 후 새 세션
  (권장)" via AskUserQuestion.
- Full detail: `claudedocs/session_20260522_track_b_p4_5_reboot_blocked.md`
  (includes verbatim continuation prompt for the post-reboot Track B P5 real
  deploy session).
- Ledger row: 2026-05-22 (Track B P4.5 post-P4 verification ...) at line 119.
- No new DECISIONS entry (no durable lesson — operational reboot omission,
  not a new rule).
- Next: user runs `sudo reboot` from terminal (assistant cannot run sudo
  autonomously). After ~1 min, new Claude Code session, paste P4.5 doc's
  continuation prompt to start Track B P5.

## Latest Verified Track A B200 Evidence

User approved exactly one close_26-only v6 projected-guard runtime on B200 GPU0,
followed immediately by v6 posthoc audit. It failed.

- Runtime stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out`
  md5 `9a4f8825a88ee3c9d93d83e5b9a28b41`, 430 lines.
- Runtime stderr:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.err`
  md5 `947cab475a1eff6ad2f3ccea6505d8c4`, 3 lines / 377 bytes.
- Audit stdout:
  `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out`
  md5 `480a3355864937763eb665e086aadbb0`, 58 lines.
- Audit stderr md5 `d41d8cd98f00b204e9800998ecf8427e` (empty).

Key v6 runtime lines:

- line 43: strict diagnostic-only, close_26-only, v6 flag YES, no training,
  constraints, SurfaceGripper, transport/release, gate tuning, posewrite, or
  success claim.
- line 45: `mode=target_guarded_micro_close_v6_projected_guard_diagnostic`
  and separate-approval marker YES.
- lines 393-397: v6 correctly blocked advance when projected support/target
  margins went unsafe; recovery writes continued with IK OK.
- line 398: first hard freeze/support-gate breach. `target_error=0.002914m`
  was still inside the fixed 0.003m target gate, but counter support gap was
  `0.002075m > 0.002m`; hard freeze YES.
- line 399: both fixed gates were breached: target error `0.003052m > 0.003m`
  and support gap `0.002146m > 0.002m`.
- lines 427-428: posthoc FAIL, 4 advances, 41 holds, zero zero-backlog holds,
  zero safety rollbacks, 12 recovery writes, 0 IK failures, 29 hard freezes,
  `close_reached=NO`, attach/posewrite zero, telemetry-only YES,
  success_claim NO.

Key v6 audit lines:

- line 18: `close_reached pass=NO`.
- lines 27-30: zero zero-backlog holds, zero safety rollbacks, positive recovery
  writes, and zero IK failures all PASS.
- line 31: hard safety freezes zero FAIL (`value=29`).
- lines 51-53: hard freeze / fixed target / fixed support criteria FAIL from
  runtime lines 398-426.
- lines 54-56: recovery present, preemptive trigger seen, and IK OK all PASS.
- line 58: `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.

Interpretation:

- v6 fixed the specific v5 mistake at old line 394: it did not advance once the
  projection went unsafe.
- v6 still failed because the recovery/hold behavior did not reduce target and
  support error fast enough after advance was blocked. The first failure is a
  support-gate hard freeze at runtime line 398, followed by target+support breach
  at line 399.
- Runtime exit 0 is not success. Success requires audit line 58 to be YES.

## Previous Track A Evidence To Keep In Mind

- v5 runtime/audit remain FAIL: stdout md5
  `f93ddaa75920a560777f8f9c8fae26f0`, audit md5
  `7709c2bc37424bc7c3874e978b34d104`. v5 line 394 advanced while support margin
  was too small; v6 corrected that specific advance decision but not the overall
  close_26 outcome.
- D083: RL-to-expert-to-rollout-to-demo is valid only after a no-attach Stage 0
  contact gate. Existing attach-based Pick/Stack PPO envs are not Track A
  evidence.
- D084: v5 recovery writes alone were insufficient; next advance needed projected
  fixed target/support margin checks.
- D085: v6 projection alone is also insufficient; once projection blocks advance,
  the mechanism must actively recover target/support before hard freeze.

## Current Direction

1. Do not rerun v2, v3, v4, v5, v6, or v7 unchanged.
2. Do not run dataset generation, PPO/training, rollout, hold-lift,
   transport/release, constraints, SurfaceGripper, or gate tuning from this
   result.
3. v7 active recovery is implemented and diagnostic telemetry works, but the
   post-reboot close_26 audit FAILED. It is not grasp success.
4. The next valid Track A work is static failure analysis/redesign of the v7
   support/target gate failure before any new runtime approval. In Codex, any
   future GPU/Isaac command still needs `sandbox_permissions=require_escalated`
   because the default sandbox hides `/dev/nvidia*` even though host CUDA is
   healthy.
5. Runtime PASS is not enough for data; next gate is hold-lift.
6. Dataset/training remain blocked until close_26 PASS + hold-lift PASS + small
   pilot dataset/replay PASS. Then proceed: no-attach RL env → random sanity →
   PPO smoke → expert rollout → pilot dataset → replay/audit → large dataset →
   BC/VLA/IL training.
7. Do not plan future work around B200 SSH. Use local backups plus local/RunPod
   GPUs. Any remote compute should start by rebuilding/verifying env and smoke
   tests from backed-up artifacts.

## Must Read First

1. `CLAUDE.md`
2. `START_HERE.md`
3. `claudedocs/DECISIONS.md` D083-D092
4. `claudedocs/EXPERIMENT_LEDGER.md` latest rows
5. `claudedocs/session_20260522_track_a_v6_projected_guard_runtime_fail.md`
6. `claudedocs/session_20260522_track_a_contact_rl_stage0_preflight.md`
7. `claudedocs/session_20260526_track_a_stage0_to_dataset_step_plan.md`
8. `claudedocs/session_20260526_track_a_v7_active_recovery_static_readiness.md`
9. `claudedocs/session_20260526_track_a_v7_local_runtime_cuda_blocked.md`
10. `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
11. `claudedocs/session_20260523_b200_disconnected_next_steps.md`
12. `b200_backup_20260522_final/README_BACKUP.md`
13. `sim_scripts/p7_branch_b_cube2cm_target_guarded_v7_active_recovery_static_design.py`
14. `sim_scripts/p7_branch_b_cube2cm_runtime_jaw_telemetry_probe.py`
15. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_criteria_audit.py`
16. `sim_scripts/p7_branch_b_cube2cm_soft_contact_runtime_readiness.py`
17. `claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md`
18. `claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md`
19. `claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md`

## Do Not Trust As Current

- `HANDOFF.md`
- `TASKS.md`
- Any claim that runtime exit code 0 means grasp success
- Any claim that target-guarded v1 through v6 passed close_26
- Any claim that target-guarded v7 passed close_26
- Any claim that v5, v6, or v7 should be rerun unchanged
- Any Track B/OpenVLA training status as evidence for Track A contact success
- Any claim that existing default Pick/Stack PPO envs produce Track A-valid
  no-attach contact experts
- Any plan that requires new B200 SSH access after 2026-05-22 23:59 KST
- Any assumption that `.ssh` secrets were or should be copied as research data
- Any claim that Codex RunPod MCP is available or unavailable without checking
  both `/home/cgxr/.codex/config.toml` and the currently loaded tool namespace
- Any assumption that all complete Track B outputs live under
  `b200_backup_20260522_final/outputs` alone
- Any use of stale RunPod pod `az53n8t8alp8pz` from 2026-05-06 unless the user
  explicitly confirms it is current and active
- Any claim that default Codex sandbox `nvidia-smi` failure means host CUDA is
  still broken. Post-reboot host CUDA is healthy; default sandbox hides
  `/dev/nvidia*`.
- Any assumption that `/tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz`
  still exists after reboot. `/tmp` is volatile; recreate the overlay if RunPod
  is needed.

## Current Dirty/Untracked Note

Dirty/untracked state is expected. Do not revert it unless explicitly requested.
Track B/OpenVLA files may be present; they are separate from Track A verdicts.

## Continuation Prompt For Next Session

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
3. claudedocs/DECISIONS.md D083-D092
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
15. claudedocs/session_20260526_runpod_mcp_codex_registration_and_next_prompt.md
16. claudedocs/session_20260526_track_a_cuda_reboot_codex_sandbox_ready.md
17. claudedocs/session_20260526_track_a_v7_active_recovery_runtime_fail.md

Current Track A state:
- v6 close_26 audit FAIL, not grasp success.
- Runtime stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_v7_close26_b200.out
  md5 9a4f8825a88ee3c9d93d83e5b9a28b41
- Audit stdout:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_target_guarded_micro_close_v6_projected_guard_audit_b200.out
  md5 480a3355864937763eb665e086aadbb0
- Reverify runtime lines 43,45,393-399,427-428 and audit lines 18,27-31,51-58 before citing.
- Interpretation: v6 blocked unsafe projected advance but recovery writes did not reduce target/support margins. First support failure line 398; first target+support breach line 399.

Professor pipeline decision:
- RL→expert→rollout→dataset→learning is valid only after Stage 0 no-attach contact primitive.
- Do NOT use existing RoArm-Pick/Stack PPO envs as clean Track A expert sources because they use kinematic attach / write_root_pose_to_sim.
- Do NOT start PPO/training/dataset/rollout first.

Current v7 status:
- Default-off v7 active target/support recovery candidate is implemented and has
  now been physics-tested locally after reboot.
- It uses finite-difference TCP candidate sweep with current object pose and robot joint target writes only.
- Objective: maximize minimum fixed target/support margin; reduce counter gap before next close advance while keeping target error inside fixed 3mm gate.
- Matching audit/readiness support and negative controls exist.
- Local static checks already passed: py_compile, git diff --check, synthetic v7 pass, v7 no-active-recovery reject, archived v6 log reject as v7, readiness.
- Approved local runtime attempt on 2026-05-26 did not produce a physics result: local IsaacLab emitted v7 metadata, then CUDA failed before env creation.
- Preserved local-block logs:
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/runtime.err
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/audit.out
  claudedocs/runtime_logs/20260526_track_a_v7_local_cuda_blocked/cuda_check.txt
- Reverify runtime.out lines 28,31,33,35; runtime.err lines 9-16,32-52; audit.out lines 3,16,18,29-30,35; cuda_check.txt lines 1-3,12-13 before citing.
- User rebooted the local Ubuntu PC after that blocked attempt. Host CUDA is now healthy: boot time 2026-05-26 14:08, NVIDIA kernel/userspace 580.159.03, host nvidia-smi OK, isaaclab torch CUDA True/device_count 1 when run outside the default Codex sandbox.
- Default Codex sandbox still hides /dev/nvidia*, so sandboxed nvidia-smi fails. This is a sandbox device visibility issue, not host CUDA failure. Run GPU/Isaac commands in Codex with sandbox_permissions=require_escalated.
- Post-reboot readiness still reports READY_FOR_SEPARATE_RUNTIME_APPROVAL=YES.
- Post-reboot close_26-only v7 runtime/audit FAILED:
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/runtime.out
  md5 621d00b9d157b4e70178c28f94ca4c7f
  claudedocs/runtime_logs/20260526_track_a_v7_active_recovery_close26_local_post_reboot/audit.out
  md5 406b96557d94418f16273e517ec4d69b
- Reverify runtime.out lines 389-393,423-424 and audit.out lines 19,32,54-56,60-66 before citing.
- Interpretation: v7 active recovery did trigger and passed its v7-specific audit checks, but fixed support failed first at runtime line 392 and fixed target failed at line 393. This is not close_26 success and v7 must not be rerun unchanged.

Current RunPod/Codex state:
- Claude has RunPod MCP configured, but Codex initially did not.
- Codex config was updated at /home/cgxr/.codex/config.toml with [mcp_servers.runpod], command npx, args ["-y", "@runpod/mcp-server@latest"], and RUNPOD_API_KEY copied from Claude config without printing the value.
- Backup exists: /home/cgxr/.codex/config.toml.bak_runpod_20260526 md5 1ef4acf6f1c92a64b9bbd79a2e35b7e7.
- Same-session tool_search after config edit initially did not expose mcp__runpod__..., but a later Codex session did expose mcp__runpod__ and list_pods returned no GPU pods. Still verify loaded tools before claiming RunPod MCP can be used.
- Do not use stale RunPod pod az53n8t8alp8pz from 2026-05-06 unless the user explicitly confirms it is current and active.
- The old minimal RunPod overlay at /tmp/track_a_v7_active_recovery_runpod_overlay_20260526.tar.gz was lost after reboot because /tmp is volatile. Recreate it if RunPod is needed.
- Local backup top USD path:
  b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd
  md5 4497024d25abab11de5c50e144124553.

Next concrete step:
1. Do not rerun v7 unchanged and do not start hold-lift/dataset/training.
2. Do static failure analysis/redesign from the v7 fail logs first.
3. Any future runtime needs separate approval and must preserve fixed target/support gates, no attach/posewrite, zero zero-backlog holds, zero safety rollbacks, and robot-joint-target-only writes.
4. Dataset/training remains blocked until close_26 PASS + hold-lift PASS + small pilot dataset/replay PASS.
```
