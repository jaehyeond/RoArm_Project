# Session 2026-05-21 — B200 Endgame OpenVLA-OFT P2 Plan + Backup Pipeline + .gitignore Fix

## TL;DR

B200 종료 5/22 23:59 KST (학습용 실효 5/22 15:00 KST, **23h**). 사용자 명시:
"VLA 여기서 smolvla 돌려본거 + 다른 트랙 pure sim 밖에 없음, 큰 모델 학습해야 하는데
데이터셋 부족, sim으로 데이터셋 모으는 게 표준 아님?"

분석 후 lock-in:

- **Plan = P2** (OpenVLA-OFT 7B 30K LoRA, ~14h, 5.5h 여유)
- **Backup = S1** (Lenovo `/` 104GB free, MUST ~20GB, 학습 후 ~81GB free)
- **π0는 후속** (RunPod RTX A6000, B200 release 후 12-15h, paper deadline 5/28 fit)
- **사용자 가정 검증**: OpenVLA visual+language asset HIGH (L1/L2), action head LOW
  (L3 6-DOF mapping은 50ep fine-tune으로 재학습). RoArm-M3 6-DOF mapping은
  `prismatic/vla/constants.py`에 `ROARM_M3_CONSTANTS` 이미 작성됨 (5/06 local).

## Two-Track State (5/21 21:30 KST 시점)

- **Track A** (P7/Branch B compliance-first contact proxy): 5/21 static design 단계,
  runtime unapproved. B200 active process 0 (검증). 본 Track B 작업 영향 0.
- **Track B** (CoRL 2026 paper sprint): 본 세션 = endgame plan + Phase 1 backup +
  Phase 2A openvla-oft upload + Phase 2B torch 재install 단계 진입.

## Verified Facts (cross-checked)

### Time budget
- Now: 5/21 ~21:30 KST (세션 중)
- B200 effective deadline: 5/22 15:00 KST → **~17.5h 남음** (수면 시간 포함)
- Actual hard deadline: 5/22 23:59 KST (이후 backup/migration 시간 사용자 추정)

### Data inventory (B200 + Lenovo)
- B200 `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/data/`:
  - `lerobot_dataset_v6/` 75MB ✓ (real PICK 50ep × 6942fr)
  - `lerobot_dataset_v6_stacking_v1/` 98MB (4/29, lying-flat 폐기)
  - `lerobot_dataset_v6_stacking_v2/` 116MB (5/01, lying-flat 폐기)
  - `lerobot_dataset_v6_stacking_v3/` 115MB (5/04, **5/05 vision-blind 진단 — 사용 금지**)
- Lenovo `lerobot_dataset_v6/` 75MB ✓
- Lenovo `sim_v1/` 87MB (4/24 v6 trajectory replay, sim image rendering)
- Local critical: `lerobot_dataset_v6_stacking_v3/` 5/05 진단:
  - σ_vision/σ_noise = 0.89~1.32 (base/gripper joints) = vision-blind
  - Cause #1: 50/50 ep S1 fixed first-grasp + spread ±25mm only
  - HARD RULE #23 위반 (identical-pattern, 5/07 evening)
  - 본 Track B 학습 사용 금지

### Backup state (S1)
- Lenovo `/`: 590GB total, 104GB free (82% used)
- rsync speed (B200 → Lenovo, 1.2GB safetensors test): **avg 2.67MB/s, 7m08s**
- MUST backup ~20GB → ETA **~2.2h**
- Background script: `/tmp/b200_backup_runner.sh` (4 phases, 시작됨)
  - PHASE1 `/tmp/p7_branch_b_*` 86MB ✅ done (or near done at session end)
  - PHASE2 `outputs/smolvla_v6_b200/` 6GB 진행 중
  - PHASE3 `outputs/smolvla_v6_stacking_v3_b200/` 12GB 대기
  - PHASE4 `code/launch_*.sh, chain_skills.py` 작은 파일 대기
- Log: `b200_backup_20260521/_backup.log`
- nohup PID: 698855

### .gitignore fix (이 세션 작업)
- 이전: `b200_backup_*/` ignored 안 됨 → GitHub Desktop 326 changed files 표시
- 수정: line 25-30에 `b200_backup_*/`, `*_backup_20*/`, `wandb/` 추가
- 검증: `git check-ignore -v b200_backup_20260521/` → `.gitignore:27:*_backup_20*/` 매치
- 결과: `git status` 에서 b200_backup_20260521/ 사라짐. push 위험 0.

### B200 env state (Phase 2B 진입 직전)
- `roarm_b200` env 활성 OK (env.sh + micromamba)
- **🔴 `python -c "import torch"` → `ModuleNotFoundError: No module named 'torch'`**
- 추정 원인: 5/15~5/17 P7 작업 중 lerobot deps re-install이 torch 다운그레이드 →
  누군가 uninstall, 또는 새 env. HARD RULE #15 관련.
- 다음 세션 1번 task: torch nightly cu128 재install (~30분-1h)

### NVML 충돌 (HARD RULE #15 관련, 새 발견)
- Kernel driver (NVRM): 580.95.05 (Sep 23 2025)
- Userspace lib default: libnvidia-ml.so.1 → 580.159.03 (May 6 2026)
- nvidia-smi 호출 시 NVML mismatch 에러
- **Fix**: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`
  (V7 D024 conversion에서 검증된 패턴)

### openvla-oft local repo (5/06 plan 산물)
- Path: `/home/cgxr/Documents/Robotics/openvla-oft/`
- Size: 2.2MB
- Last commit: `e4287e9` (Update pyproject.toml: Pin diffusers version)
- **Local modification**: `prismatic/vla/constants.py` —
  RoArm-M3 6-DOF mapping 이미 작성:
  ```python
  ROARM_M3_CONSTANTS = {
      "NUM_ACTIONS_CHUNK": 8,
      "ACTION_DIM": 6,
      "PROPRIO_DIM": 6,
      "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
  }
  ```
  + `detect_robot_platform()` 에 `roarm` keyword 인식 추가
- B200 upload 명령은 발동됐으나 본 세션 마지막 확인 단계 미완 (다음 세션 1번 task)

## Decisions Made (HARD RULE #18 사용자 명시)

1. **B200 plan = P2 (30K LoRA)** — 50K 대비 risk-adjusted 우수, paper value 거의 동일
2. **Backup target = S1 (Lenovo only)** — USB 외장 미보유, RunPod HOLD 상태
3. **π0는 RunPod 후속** — B200 비교 우위는 7B (OpenVLA-OFT), π0 3.3B는 다른 platform 가능
4. **stacking_v3 학습 금지** — 5/05 vision-blind 진단 + HARD RULE #23 위반
5. **OpenVLA-OFT 학습 data = v6 real 50ep + sim_v1 co-train (Sim-and-Real Co-Training, RSS 2025)**

## Track A Impact Assessment (cross-verified)

- C1 GPU 0 contention: LOW (active compute process 0, 검증됨)
- C2 Codebase git conflict: LOW (Track B는 sim_scripts/ 안 건드림)
- C3 sim_scripts/ touch: NONE
- C4 Track A `/tmp` 로그 backup loss: LOW-MED → 본 세션 backup PHASE1으로 회수 중
- C5 B200 release 자체 영향: HIGH but Track B 무관 (Track A는 어차피 5/22 release 후 B200 사용 불가)
- C6 START_HERE.md 학습 후 update 충돌: MED → Track A current direction 보존하고 Track B 결과 별도 섹션 append
- C7 HARD RULE #14 fail-fast guard 위반 destruction: LOW (모든 ssh 명령에 guard 적용)
- C8 HARD RULE #15 torch 다운그레이드: MED-HIGH (이미 발생, 다음 세션 fix)
- C9 paper_v1 branch 미존재: LOW (학습 자체는 git에 영향 없음, server-side)

→ **Net: Track A 측정 가능한 손해 LOW**

## Continuation Prompt (다음 세션 paste-verbatim)

```
Read CLAUDE.md first, then follow the Current-State Protocol exactly.

Step-by-step:
1. Read START_HERE.md.
2. Read claudedocs/DECISIONS.md.
3. Read claudedocs/EXPERIMENT_LEDGER.md.
4. Read claudedocs/session_20260521_b200_endgame_openvla_oft_pivot.md (이 세션의
   detailed plan + verified facts + Phase 1~5 step + Track A impact).
5. Run `git status --short`.
6. Check backup background progress:
   tail -30 /home/cgxr/Documents/Robotics/RoArm_Project/b200_backup_20260521/_backup.log
   ls -la /home/cgxr/Documents/Robotics/RoArm_Project/b200_backup_20260521/
7. Brief me on:
   - B200 남은 시간 (5/22 15:00 KST까지)
   - Backup 진행 상황 (PHASE2/3 완료 여부)
   - 다음 즉시 task = Phase 2B (B200 torch nightly cu128 재install)

Rules:
- HARD RULE #11: /half-clone 절대 사용/제안 금지.
- HARD RULE #14: 모든 ssh 명령에 fail-fast guard
  (set -e; [[ -z "$ROARM_B200_ROOT" ]] && exit 1; [[ "$(whoami)" != "sogang_jhki" ]] && exit 1).
- HARD RULE #15: torch nightly cu128 강제 회복 (sm_100). lerobot install 후 별도
  `pip install --upgrade --pre torch torchvision torchaudio --index-url
  https://download.pytorch.org/whl/nightly/cu128` 강제.
- HARD RULE #18: 사용자 명시 정정 우선.
- HARD RULE #19/#20: Edge-stand 47mm sponge, # tower 87/67mm c2c (현재 PICK 학습이라 무관, 단 deploy 시 적용).

Plan = P2 OpenVLA-OFT 30K LoRA on v6 real + sim_v1 co-train (~14h).
backup background 진행 중 → 학습 시작 전 PHASE1-PHASE4 완료 확인.

다음 즉시 명령 sequence:
(a) Phase 2A 검증: ssh JHPark 'ls -la /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/openvla-oft/'
    → ROARM_M3_CONSTANTS 포함 prismatic/vla/constants.py 존재 확인
(b) Phase 2B torch 재install: ssh JHPark에서 micromamba activate + nightly cu128
(c) Phase 2C openvla-oft pip install -e . --no-deps + flash-attn + peft + accelerate
(d) Phase 2D 1K smoke test (action_dim=6, image=top, LoRA rank=32, save_lora_only=True)
(e) Phase 3 30K LoRA finetune (smoke 결과 기반 final 결정)

세션 시작 시 즉시 backup 진행도 확인 (PHASE2/3 자동 진행됐어야).
```

## Open Risks (다음 세션 우선)

1. **R1 torch 재install 실패** (HARD RULE #15 lerobot deps 재충돌): MED-HIGH
   - 완화: `pip install --upgrade --pre torch ... --index-url .../nightly/cu128`
     **lerobot 다운로드/install 직후 강제 회복**
2. **R2 NVML mismatch 학습 중 영향**: MED
   - 완화: 모든 GPU 명령에 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`
3. **R3 backup 학습과 SSH 채널 경합**: LOW
   - 완화: backup rsync는 PHASE2 (6GB ~40min) 완료 후 PHASE3 (12GB ~80min) — 학습 시작 시점에도 진행
4. **R4 30K convergence 불충분 가능**: MED
   - 완화: 5K/10K/15K/20K/25K/30K checkpoint sweep + offline eval로 best 추출
5. **R5 OpenVLA 50ep underdata overfit**: MED-HIGH
   - 완화: paper에서 "G4 gap negative finding"으로 reframe (CoRL 2025 <$1k arm 0/221편)

## Files Created / Modified (이 세션)

- `b200_backup_20260521/` (untracked, ignored): backup target folder
- `b200_backup_20260521/env.sh` (B200 env script copy)
- `b200_backup_20260521/tmp_logs/p7_branch_b_*` (PHASE1 진행 중)
- `b200_backup_20260521/outputs_smolvla_v6_b200/` (PHASE2 진행 중)
- `b200_backup_20260521/_backup.log`
- `/tmp/b200_backup_runner.sh` (background script)
- `.gitignore` (M, line 25-30 추가)
- `claudedocs/session_20260521_b200_endgame_openvla_oft_pivot.md` (이 파일)

## HARD RULE Compliance (이 세션)

- #11 `/half-clone` 거부 1회 (115% context Stop hook) — 본 doc 작성 + continuation prompt로 대체
- #14 B200 ssh 명령에 fail-fast guard 적용 (whoami, hostname, ROARM_B200_ROOT 검증)
- #15 torch nightly cu128 issue 인식 (다음 세션 fix)
- #18 사용자 명시 정정 우선 (P2 채택, S1 backup, π0 RunPod 후속)
