# AGENTS.md — RoArm_Project (single source of project instructions)

이 파일은 이 repo에서 작업하는 **모든 AI 도구 공통**(도구 중립) 지침의 단일 소스다.

- Claude Code: `CLAUDE.md` 첫 줄의 `@AGENTS.md` 임포트로 자동 인라인 로드.
- Codex CLI / Cursor: cwd의 `AGENTS.md`로 네이티브 자동 로드.
- Claude 전용 내용(12-agent 팀, 스킬, auto-memory 워크플로우)은 `CLAUDE.md`에 있다.

**상태 ≠ 규칙**: 이 파일에는 규칙만 둔다. 진행 상태·실험 결과의 인계는
`START_HERE.md` → `claudedocs/DECISIONS.md` → `claudedocs/session_*.md`로만 한다.
**같은 프로젝트에서 Claude와 Codex(또는 Cursor) 편집 세션 동시 실행 금지** — 한 도구의
세션이 state doc 갱신으로 끝난 뒤 다른 도구를 시작한다.

## Current-State Protocol

This repo uses rolling state docs so Claude Code, Cursor Codex, and CLI Codex can
resume without relying on memory alone.

### File roles (do not confuse)

| File | Role | Mutation |
|---|---|---|
| `START_HERE.md` | Rolling current-state dashboard. New sessions boot from here. | Overwrite each major update. |
| `claudedocs/DECISIONS.md` | Durable lessons / do-not-repeat rules with evidence. | Append-only (`D001`, `D002`, ...). Mark superseded; never delete. |
| `claudedocs/EXPERIMENT_LEDGER.md` | Append-only table of major experiments + verdicts. | Append-only. |
| `claudedocs/session_YYYYMMDD_*.md` | Detailed session logs with metrics, code paths, evidence. | Append-only new file per session. |
| `HANDOFF.md`, `TASKS.md` | Historical (stale). | Do not trust unless `START_HERE.md` explicitly points to them. |
| `~/.claude/projects/.../memory/MEMORY.md` | Per-user auto-memory across all conversations (Claude 전용 HARD RULES, recent sessions index). | Prepend recent sessions; HARD RULES never deleted. Complementary, NOT a replacement for `START_HERE.md`. |

### Session boot procedure

Before answering current project-state questions or making edits:

1. Read `START_HERE.md`.
2. Read `claudedocs/DECISIONS.md`.
3. Read `claudedocs/EXPERIMENT_LEDGER.md`.
4. Read only the `claudedocs/session_*.md` files referenced by `START_HERE.md`
   unless more evidence is needed.
5. Run `git status --short`.
6. Verify any metric from the referenced log/data file before citing it.

## Session progress rule

- Every research session must run at least one experiment that can fail
  (RL training with real updates, or perturbation evaluation), or explicitly
  justify why not in the session doc.
- Control-contract hardening is REACTIVE only: it is permitted solely in
  response to a failure observed during training or perturbation evaluation.
- A verdict ending in `NO_PPO_PROMOTION` without a training attempt or
  perturbation evaluation in the same session requires explicit justification
  against this rule.
- Validation that cannot change a decision must not be run.

## Research briefing language and teaching rule

- User-facing research briefings must be written primarily in Korean. Keep exact
  code, schema, field, file, and verdict identifiers when needed, but explain each
  unfamiliar English term in plain Korean at first use.
- Never use only a `Dxxx`, attempt number, acronym, or internal check name as the
  explanation. State what it is, why it was checked, what PASS/FAIL means, and
  how it changes the next decision.
- A step-by-step request means reporting auditable actions, observations, and
  evidence in execution order. Do not replace that report with opaque labels.
- The final experiment briefing follows this order: (1) what/why, (2) procedure
  in observable steps, (3) quantified result with source paths, and (4) an
  everyday-language verdict plus the next authorization boundary. Put this
  briefing at the end of the turn so later tool chatter does not bury it.

## Variable Ladder Protocol (D322~)

- Each active case may introduce only one or two new variables. The session doc
  must state near the top: `이번 case의 신규 변수: [...]`.
- Future-looking ideas must not be implemented immediately. Append them to
  `claudedocs/BACKLOG.md`, then return to the current critical path.
- The `START_HERE.md` `Active Case` section is the single source of truth for
  what is in scope. Everything outside it is a non-goal unless the user
  explicitly approves a case change.
- Folders are forward-only. Do not move or rename existing files/folders, so
  old evidence paths remain valid. New grasp outputs must be created only under
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`, and the path must be
  listed in `START_HERE.md` `Active Case`.

## Visualization Definition of Done (D324~)

- Any probe/evaluation that reasons about geometry, pose, contact, jaw faces, or
  tool frames must emit visual diagnostics through `roarm_rl.viz_debug` when
  practical.
- Required artifacts are: target-vs-actual frame markers, at least one
  decision-time diagnostic snapshot in the run output folder, and explicit
  snapshot paths in the session document.
- This rule is for single-frame debugging only. It does not relax the existing
  ban on large renders, trajectory videos, new data generation, or variable
  ladder advancement without explicit user approval.

## Rerun Observability Completion Contract (D341~)

- A replayable RRD is mandatory when a verdict depends on geometry, pose,
  coordinate frames, collision/contact, a trajectory, or synchronized sensor
  time. Rerun may be omitted only for a pure file/hash/schema audit with no
  spatial or temporal judgment; the session doc must state that justification.
- Deterministic Isaac/batch work uses save-only recording by default. A live
  Viewer is optional for exploration, but the file sink must be attached before
  the first user log in either mode. The recording must be finalized by a
  `RecordingStream` context exit or disconnect before any artifact gate runs.
- The RRD must contain the actual decision subject, not only generic robot/frame
  markers. Cook/representation cases log source, instance, prototype, and
  candidate geometry as separate entities. Physics/settle cases log the full
  executed step timeline plus decision scalars and contact points/force arrows;
  a final or trial-1-only row is insufficient for a trajectory verdict.
- Rerun is an observability/replay layer, not the bit-exact authority. Original
  callback arrays and canonical JSON/hashes decide equality. Float64 metrics may
  be plotted in Rerun, while its Float32 spatial copies are inspection evidence
  only and must never be hashed back into a scientific gate.
- Rerun completion requires all of the following: the exact SDK/CLI version pin;
  footer-enabled `rrd verify` PASS after finalization; exact non-system entity,
  timeline, and required-component contracts PASS; a fixed embedded blueprint
  plus its verified `.rbl` export; a headless decision
  screenshot; and an actual visual inspection whose path and observations are
  recorded in the session doc. Non-empty generation, loadability, or screenshot
  creation alone must never be reported as "inspected".
- RRD, RBL, validation report, and inspection screenshot belong in the active
  run output folder. If any required item fails, the visualization contract
  fails without overriding the scientific verdict or relaxing a gate.

## IsaacLab Environment Package Rule (D326~)

- Any package install into the `isaaclab` conda environment must record the
  dependency impact and verify the known Isaac-compatible pins afterward:
  `numpy==1.26.0` and `psutil==5.9.8`.
- If an install upgrades either package, immediately restore those pins and
  verify imports before running Isaac. This rule comes from D325, where
  installing `rerun-sdk` pulled incompatible `numpy 2.4.6` and `psutil 7.2.2`.

### Session boot prompt (paste this verbatim at new-session start)

```
Read AGENTS.md first, then follow the Current-State Protocol exactly.

Step-by-step:
1. Read START_HERE.md.
2. Read claudedocs/DECISIONS.md.
3. Read claudedocs/EXPERIMENT_LEDGER.md.
4. Read only the claudedocs/session_*.md files referenced by START_HERE.md
   unless missing evidence requires more.
5. Run `git status --short`.
6. Brief me on:
   - Current verified state (with file:line citations)
   - Active pivot vs reserve pivots
   - Open risks / do-not-repeat rules from DECISIONS.md
   - Next concrete action

Rules:
- Be critical, analytical, and skeptical. Cross-verify before claiming.
- Do not rely on memory-only claims. Cite the referenced file/line.
- Verify metrics from logs/data files; flag any mismatch.
- Do not treat HANDOFF.md or TASKS.md as current state.
- If context approaches 95%, stop new work and run the end-of-session update.
- We are continuing the RoArm Isaac Lab hierarchical chain skills work
  (or whatever START_HERE.md says is the active pivot — do not assume).
```

### End-of-session update prompt (paste before closing session)

```
Before ending this session, update the project state system.

Step-by-step:
1. Update START_HERE.md (overwrite) with:
   - Current truth (latest session_*.md link)
   - Current status (key metrics, latest run results)
   - Current direction (active pivot + next concrete step)
   - Must-read first list
   - Do-not-trust-as-current list
2. Append to claudedocs/EXPERIMENT_LEDGER.md any major run/result row
   (Date | Run | Goal | Key Result | Verdict | Source).
3. Append to claudedocs/DECISIONS.md ONLY if a durable lesson, failure rule,
   or do-not-repeat rule changed. Use Dxxx numbered sections with Evidence /
   Implication / Source.
4. Write a new claudedocs/session_YYYYMMDD_short_title.md (append-only) with
   detailed metrics, code paths, evidence, decisions, next steps.
5. Do not overwrite previous session logs.
6. Keep START_HERE.md short (~120 lines). Put history in the ledger and
   detail in the session doc.
7. Cross-verify: re-read all 4 files (START_HERE, DECISIONS, EXPERIMENT_LEDGER,
   new session doc). Check numbers match across files.
```

### Context 95% emergency protocol

If active chat context approaches 95%:

1. Stop new implementation work immediately.
2. Run the end-of-session update prompt above (state files only — no new code).
3. Output a concise continuation prompt for the next session (≤80 lines, lists
   active pivot, next concrete step, files to read, current md5s).
4. Do NOT use `/half-clone` or `/handoff` skills (project rule, see auto-memory
   HARD RULE #11).
5. User starts a new session and pastes the boot prompt above.

### Project rules for state files

- `START_HERE.md` is the current dashboard and is overwritten as work progresses.
- `claudedocs/DECISIONS.md` is append-only durable lessons / do-not-repeat rules.
- `claudedocs/EXPERIMENT_LEDGER.md` is append-only major experiment history.
- `claudedocs/session_*.md` files are detailed append-only session logs.
- `HANDOFF.md` and `TASKS.md` are historical and stale unless `START_HERE.md`
  explicitly points to them.
- Auto-memory `MEMORY.md` is per-user across conversations; it is complementary
  to (not a replacement for) the project-state docs above. Update both when a
  session closes: project-state for repo continuity, MEMORY.md for user-level
  habits/preferences and HARD RULES.

## HARD RULES — 절대 위반 금지 (도구 중립)

> 원 출처: Claude auto-memory
> `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/MEMORY.md`.
> 도구 중립 규칙을 2026-07-12에 **verbatim 이전** — Claude는 `CLAUDE.md`의 `@AGENTS.md`
> 임포트로, Codex/Cursor는 네이티브로 모두 자동 로드한다. **번호는 원본 그대로 보존** —
> 다른 문서의 "HARD RULE #n" 참조가 계속 유효하다. 삭제 금지, 새 실패 발생 시에만 append.
> 잔류분: #8(MEMORY.md 운영)·#11(/half-clone 금지)은 Claude 전용이라 auto-memory에 잔류.
> #13-#17/#21-#26은 B200-era 비활성 pointer로 auto-memory에 잔류
> (원문: auto-memory의 `hard_rules_b200_era_archive_20260711.md`).

1. **데이터 수집 = HOME [0,0,90,0,0,0] 시작** — v5 136ep 전량 폐기 사유. 유저 직관을 데이터 없이 기각 금지. (상세: feedback_v5_data_collection_failure.md)
2. **학습 = `lerobot-train` CLI + `lerobot/smolvla_base` pretrained 필수** — 커스텀 학습 3회 실패. Action Expert 랜덤 초기화 금지. (상세: tech_critical_lessons.md)
3. **VGST 자동 verdict FAIL ≠ 실제 실패** — 5° 임계값 too conservative. v3(M2=1.73° FAIL→5/5 성공), v6(M2=4.62° FAIL→reach 성공) 2회 전례. open-loop 직접 테스트 필수. (상세: experiment_log.md 2026-04-07)
4. **"없다/최초" 주장 = 10개+ 검색어 × 2개 소스 검증** — 3/10, 3/23에 거짓 갭 주장 3회 반복. 반증 검색 먼저. (상세: research_verification_failures.md)
5. **JOINT_LIMITS 제거 금지** — 하드웨어 보호.
6. **Camera 위치 변경 = 전체 데이터 무효** — 단, "카메라 절대 고정"은 과적합 원인이기도 함 (대형 VLA는 다양 각도 OK). 수집 중에는 고정, 데이터셋 설계 시에는 다양성 확보.
7. **HANDOFF.md 절대 생성/건드림 금지** — MEMORY + continuation prompt 방식만 사용.
8. **[Claude 전용 — auto-memory 잔류]** MEMORY.md 운영 규칙 (오버라이드 금지, Recent Sessions prepend).
9. **VLA 모델은 SmolVLA에 한정 안 됨** — 클라우드 GPU 사용 가능, OpenVLA/Octo/pi0 비교 가능.
10. **문제-중심 연구만 수용** — "X% 향상" 메트릭 논문 구조 거부. 시간 제약으로 아이디어 축소 금지 (유저 20h/day 투입 의지).
11. **[Claude 전용 — auto-memory 잔류]** `/half-clone` 절대 사용/제안 금지.
12. **배포 디버깅 = 데이터셋 이미지/상태 먼저 확인** — 모터/속도/대역폭 같은 HW 추측 전에 학습 parquet의 state 분포, ep 끝 프레임 gripper/pose, Azure Kinect 이미지를 직접 본다. L-F 수집의 action(leader)과 state(follower) gap은 설계 기능이지 버그 아님. (4/9 gripper 실패 오진 → 규칙화)
13. -17. **[B200-era 비활성 — auto-memory 잔류]** 원문: auto-memory `hard_rules_b200_era_archive_20260711.md`.
18. **사용자 명시 정정 > Claude 추론 (절대 우선)** — 사용자가 "X = Y"로 명시 정정한 사항은 후속 분석/PNG/parquet 재해석으로 절대 무효화 금지. 재해석 필요 시 반드시 사용자에게 confirm 후 진행. **근거**: 4/30 evening 사용자 "sponge=세움 (이미지2처럼)" 정정 → 4/30 late-evening Claude가 "v6 PNG 재분석"으로 "vertical pillar + 우물정자=lying-flat" 결론 도출, 사용자 정정 무효화 → 5/01 sim_demos_v2 lying-flat 50ep × 146fr + 5/03 B200 10K finetune (~42min GPU) + ST-C 1차 deploy 모두 폐기. 약 4 세션 손실. **적용**: sponge orientation, # pattern geometry, grasp 방향 등 사용자 한 번이라도 명시한 design 결정은 단독 변경 금지. (5/03 evening 규칙화)
19. -20. **[Track A sponge/stacking 확정 design — 변경 절대 금지 유지, 본문 그대로 보존]** Sponge = edge-stand 47mm tall(#19, lying-flat/vertical-pillar 금지, TCP grasp z +33mm world) / # tower = 2-layer cross, L1 Y c2c=87mm·L2 X c2c=67mm(#20). **sponge/stacking 작업 재개 시 원문 read 필수**: auto-memory `hard_rules_b200_era_archive_20260711.md` + `project_well_pattern_design_v3.md`.
21. -26. **[B200-era 비활성 — auto-memory 잔류]** 원문: auto-memory `hard_rules_b200_era_archive_20260711.md`.
27. **B200 lease retired after 2026-05-22 23:59 KST — no future work may require B200 SSH or `.ssh` secrets.** `JHPark/roarm_b200` 재진입, B200 Isaac 재실행, 추가 파일 pull, `.ssh` 키 복사/요청 전부 금지. 백업 안 된 B200-only 파일이 필요한 경로 = blocked, 로컬 증거로 재설계. Track A/B 백업 검증 해시 + 위치(Track B 완전 checkpoint는 `openvla_oft_b200_pulls`) 원문: auto-memory `hard_rules_b200_era_archive_20260711.md` + `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`, DECISIONS D087-D088.
28. **D232 storage rule — SmolVLA `outputs/` 기본 보존; `collected_data*`/`b200_backup_*`/`openvla_oft_b200_pulls` 삭제 금지 (archive/move-only, 명시 승인 필요).** 디스크 압박 시 무작위 재스캔 금지. 1차 승인 경로 = `outputs/*/checkpoints/*/training_state` cleanup (~25.6GB, manifest+명시 승인 후, pretrained_model 보존). 2차 = run별 keep-one pruning (~90.15GB 총, **무승인 실행 절대 금지**). Run별 keep-one 목록 원문: auto-memory `hard_rules_b200_era_archive_20260711.md` + D232 docs/logs.

## Safety constraints (모든 도구 공통)

- 로봇 하드웨어 직접 제어(`serial` `/dev/ttyUSB*`, `torque_set`, `joints_angle_ctrl`,
  `move_init`, `T:106`)는 사용자 명시 승인 없이 실행 금지.
- `lerobot-train` 실행은 사용자 승인 후에만 (config 설계/검토는 자유).
- `rm -rf` 금지. `JOINT_LIMITS` 코드 제거 금지 (HARD RULE #5).
- git commit/push는 사용자가 요청할 때만.

## File ownership (요약 — 상세 표는 CLAUDE.md Agent Team)

파일 prefix별 소유 규칙: `data_*`/`collect_data_manual.py`(data), `train_*`/`run_official_train.py`(pipeline),
`deploy_*`(deploy), `trajectory_*`(manipulation), `sim_*`(sim2real), `hw_*`/`calibrate_*`(hardware),
`model_*`(vla-model), `augment_*`/`self_improve_*`(data-efficiency), `monitor_*`/`safety_*`(deployment-safety),
`experiment_*`/`eval_*`(experiment), `analysis_*`/`figure_*`(analysis), `paper/`(writing).
어떤 도구든 이 경계를 존중한다.

## Project Overview

RoArm-M3-Pro + SmolVLA (Vision-Language-Action) 파이프라인.
Azure Kinect 카메라 → SmolVLA(450M) 모델 → RoArm M3 (6-DOF) 실시간 제어.

```
[Azure Kinect] → [SmolVLA] → [RoArm M3 Pro]
     │              │              │
  RGB 720P    Flow Matching    6-DOF joints
              10 denoise steps   ~10ms/step
```

## Environment

| Component | Details |
|-----------|---------|
| OS | Ubuntu 22.04 (Linux) |
| GPU | RTX 4090 Laptop (15.6 GB VRAM), Driver 580, CUDA 12.6 |
| Python | 3.11.14 (conda env `roarm`) |
| PyTorch | 2.7.1+cu126 |
| LeRobot | 0.4.4 (source install at `lerobot/`, .gitignored) |
| Follower | RoArm-M3-Pro via `/dev/ttyUSB1` (배포/추론 대상, 카메라가 촬영, 팔 #3) |
| Leader | RoArm-M3-Pro via `/dev/ttyUSB0` (L-F 수집 시 손으로 조작, 그리퍼 클램프, 팔 #1) |
| Camera | Azure Kinect DK (pyk4a 1.5.0 + libk4a 1.4.2) |
| Framework | LeRobot + SmolVLA (HuggingFace) |

## Key Commands

```bash
# conda 환경 활성화
conda activate roarm

# 데이터 수집 (토크 OFF 수동 모드)
python collect_data_manual.py

# LeRobot v3 포맷 변환
python convert_to_lerobot_v3.py --input collected_data --task "Pick up the white box"

# 학습 (공식 CLI 사용 — 커스텀 학습 스크립트 절대 금지!)
python run_official_train.py

# 오프라인 추론 테스트
python test_inference_official.py

# 실제 로봇 배포
python deploy_smolvla.py --start-pos dataset_mean --max-steps 300

# 데이터 품질 검증
python data_episode_quality.py
python data_distribution_simple.py

# 로봇 복구 (모터 버스 문제) — 포트는 복구 대상에 맞게: Leader=/dev/ttyUSB0, Follower=/dev/ttyUSB1
python scan_servos.py /dev/ttyUSB0   # 예시: Leader. Follower 복구 시 /dev/ttyUSB1
python reset_robot.py

# 하드웨어 테스트 (Leader=USB0, Follower=USB1 — 양쪽 다 확인 권장)
python -c "from pyk4a import PyK4A; k4a = PyK4A(); k4a.start(); print('Kinect OK'); k4a.stop()"
python -c "from roarm_sdk.roarm import roarm; arm = roarm('roarm_m3', '/dev/ttyUSB0', 115200); print('Leader OK (USB0)'); arm.disconnect()"
python -c "from roarm_sdk.roarm import roarm; arm = roarm('roarm_m3', '/dev/ttyUSB1', 115200); print('Follower OK (USB1)'); arm.disconnect()"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Pipeline Architecture

### Core Pipeline (5단계)

```
collect_data_manual.py     [1] 토크 OFF + Azure Kinect로 데이터 수집
        ↓
convert_to_lerobot_v3.py   [2] LeRobot v3 포맷 변환 (parquet + video)
        ↓
run_official_train.py      [3] lerobot-train CLI 래퍼 (smolvla_base 사전학습)
        ↓
test_inference_official.py [4] 오프라인 추론 테스트 (L2 error, z-score, diversity)
        ↓
deploy_smolvla.py          [5] 실제 로봇 배포 (dataset_mean 시작, closed-loop)
```

### Key Files

| 파일 | 역할 |
|------|------|
| `collect_data_manual.py` | 데이터 수집 (Azure Kinect + 토크 OFF) |
| `collect_data.py` | 데이터 수집 (대체 스크립트) |
| `convert_to_lerobot_v3.py` | LeRobot v3 포맷 변환 |
| `run_official_train.py` | lerobot-train CLI 래퍼 |
| `test_inference_official.py` | 오프라인 추론 테스트 |
| `deploy_smolvla.py` | 실시간 로봇 배포 |
| `scan_servos.py` | T:106 명령으로 모터 버스 리셋 |
| `reset_robot.py` | 로봇 리셋 유틸리티 |
| `calibrate_azure_kinect.py` | 카메라 캘리브레이션 |
| `data_episode_quality.py` | 에피소드 품질 분석 |
| `data_distribution_simple.py` | 액션 분포 시각화 |
| `train_eval_checkpoints.py` | 체크포인트 평가 |
| `train_config_50k.py` | 50K 학습 설정 |
| `lerobot_backup/roarm_m3.py` | LeRobot RoArm M3 통합 (백업) |
| `lerobot_backup/configs.py` | RoarmRobotConfig (백업) |

### YAML Configs (Leader-Follower)

| 파일 | 설명 |
|------|------|
| `lf_teleop_config.yaml` | L-F 텔레옵 (카메라 없음) |
| `lf_teleop_nocam_config.yaml` | L-F 텔레옵 (카메라 없음, 주석 포함) |
| `lf_teleop_camera_config.yaml` | L-F 텔레옵 + OpenCV 카메라 |

## RoArm M3 Hardware

### Joint Specs

| Joint | Name | Range (deg) | Note |
|-------|------|-------------|------|
| 0 | Base rotation | -190 ~ 190 | 좌우 회전 |
| 1 | Shoulder | -110 ~ 110 | 어깨 |
| 2 | Elbow | -70 ~ 190 | 비대칭! |
| 3 | Wrist pitch | -110 ~ 110 | 손목 상하 |
| 4 | Wrist roll | -190 ~ 190 | 손목 회전 |
| 5 | Gripper | -10 ~ 100 | 그리퍼 개폐 |

### SDK API

```python
from roarm_sdk.roarm import roarm

arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB1", baudrate=115200)  # Follower 예시. Leader는 /dev/ttyUSB0

angles = arm.joints_angle_get()           # → list[6] (degrees)
arm.joints_angle_ctrl(angles=[0]*6, speed=500, acc=200)
arm.torque_set(cmd=1)                     # 1=on, 0=off (keyword arg cmd 필수!)
arm.move_init()                           # 초기 위치
arm.disconnect()
```

### SDK Bugs & Workarounds
- **print(data) 스팸**: `roarm_sdk.common.DataProcessor._process_received` 몽키패치로 억제 (모듈명 주의: `sdk_common` 아님)
- **BaseController 로거**: CRITICAL 레벨로 설정 (백그라운드 스레드 디코드 에러)
- **safe_joints_angle_get()**: 5회 재시도 (간헐적 None/KeyError 대응)

#### 올바른 `_silent_process` 패턴

⚠️ **`lambda *a, **k: None` 사용 절대 금지**: `_process_received`는 단순 print만 하는 게 아니라 `data['x'/'y'/'z']` 추출 + `handle_m3_feedback()` 호출 등 **데이터 파싱 핵심 로직**을 담당. `lambda: None`으로 치환하면 `joints_angle_get()` 등 모든 read API가 `None` 반환 → `subscript` 에러. 반드시 아래 패턴 사용 (출처: `collect_data_manual.py:44-60`):

```python
import logging
logging.getLogger().setLevel(logging.CRITICAL)
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback

def _silent_process(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data = [data['x'], data['y'], data['z']]
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res
DataProcessor._process_received = _silent_process
```

### USB Configuration

```
Laptop ──USB──→ [USB Hub]
                    │
        ┌───────────┴───────────┐
        ↓           ↓           ↓
  Azure Kinect    Leader     Follower
     (DK)     (/dev/ttyUSB0) (/dev/ttyUSB1)
```

## Motor Recovery (모터 응답 없음)

> 포트는 복구 대상에 맞게: **Leader=/dev/ttyUSB0, Follower=/dev/ttyUSB1**. 아래 예시는 단일 로봇 시나리오라 USB0을 사용 — 실제 사용 시 대상 포트로 교체.

### 증상
- 전원 ON해도 팔이 초기 위치로 안 감
- `joints_angle_get()` → `[180, -180, -90, -180, 180, 180]` (에러 기본값)

### 해결 방법 1: T:106 ESP32 리셋

```bash
python scan_servos.py /dev/ttyUSB0
```

```python
import serial, time
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
time.sleep(1)
ser.write(b'{"T":106}\n')  # ESP32 크래시 → 자동 리셋 → 모터 버스 재초기화
time.sleep(1)
ser.close()
```

### 해결 방법 2: 토크 ON + 초기 위치

```python
from roarm_sdk.roarm import roarm
arm = roarm(roarm_type='roarm_m3', port='/dev/ttyUSB0', baudrate=115200)
arm.torque_set(cmd=1)
arm.move_init()
arm.disconnect()
```

## Camera Setup

| Item | Value |
|------|-------|
| Model | Azure Kinect DK |
| RGB | 1280x720 (720P) |
| Depth | NFOV_UNBINNED |
| Library | `pyk4a` |
| Connection | USB 3.0 |

```python
import pyk4a
from pyk4a import Config, PyK4A

k4a = PyK4A(Config(
    color_resolution=pyk4a.ColorResolution.RES_720P,
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
    synchronized_images_only=True,
))
k4a.start()
capture = k4a.get_capture()
rgb = capture.color[:, :, :3]  # BGRA → BGR
```

**카메라 nuance** (HARD RULE #6 반영, 2026-04-28):
- **수집 단일 세션 내**: 카메라 절대 고정 (삼각대/클램프) — 위치 변경 시 그 데이터셋 무효
- **데이터셋 설계**: 다양한 viewpoint 사용 가능 (대형 VLA는 다양 각도 OK, 카메라 절대 고정은 과적합 원인)
- **Sim env (4/24)**: Kinect 빨간 마커 calibration RMSE 10.13mm — sim 내 동일 viewpoint 1:1 매핑

## Critical Rules (절대 지켜야 할 것)

### 학습

| Rule | Why |
|------|-----|
| **커스텀 학습 스크립트 작성 금지** | 공식 파이프라인의 정규화/스케줄러/전처리가 누락됨 |
| **`lerobot-train` CLI만 사용** | `run_official_train.py`가 래핑 |
| **`lerobot/smolvla_base` 사전학습 필수** | Action Expert가 사전학습 안 되면 평균 액션만 출력 |
| **Loss ↓ ≠ 좋은 모델** | L2 error + z-score + diversity 함께 확인 |

### 배포

| Rule | Why |
|------|-----|
| **dataset_mean 시작 위치** | [0,0,0,0,0,0] 시작은 OOD → 소심한 동작 |
| **`n_action_steps=1`** | Closed-loop: 매 스텝 새 추론 |
| **JOINT_LIMITS 절대 제거 금지** | 하드웨어 보호 |

### 데이터

| Rule | Why |
|------|-----|
| **수집 세션 내 카메라 고정 (삼각대/클램프)** | 단일 세션 위치 변경 = 그 데이터 무효. 데이터셋 설계 시 다양 viewpoint는 OK |
| **Azure Kinect 메인 사용** | 본 프로젝트 v6 = pyk4a single Kinect (IMX335/ZED Mini 보유 미장착) |
| **v6 50ep + sim demos co-training** | 4/24 결정. 단순 "100+ ep 수집"보다 sim demos (Mimic 500+) co-training이 stacking에 효과적 |

## LeRobot Integration

### 데이터 수집 방식
**현재 (4/01부터): Leader-Follower (L-F) 텔레옵**:
- Leader (USB0, 팔 #1, 그리퍼 클램프) → 손으로 조작
- Follower (USB1, 팔 #3, 카메라 촬영 대상) → 동기 추종
- `collect_data_manual.py` (L-F 모드) → Azure Kinect (pyk4a)
- v6 50ep는 모두 L-F로 수집 (4/01 전환 후)

**Legacy (참고만)**: 팔 1개 + 토크 OFF 수동 모드 (v1~v5에서 사용, v6부터 L-F)

### LeRobot 백업 파일
`lerobot_backup/` 폴더에 RoArm M3 통합 코드 백업:
- `roarm_m3.py` → `lerobot/lerobot/common/robot_devices/robots/` 에 복사
- `configs.py` → 동일 경로에 복사 (RoarmRobotConfig 추가)

### Strategy Pattern Architecture

```
RoarmRobot
├── connect() → strategy.initialize()
├── teleop_step() → strategy.generate_goal_positions()
├── capture_observation() → follower 읽기 + 카메라
├── send_action() → policy 추론용
└── disconnect() → strategy.cleanup()

Strategies:
├── KeyboardTeleopStrategy   (leader_arms={})
└── LeaderFollowerTeleopStrategy (leader_arms 설정 시)
```

## Training Lessons (실패에서 배운 것)

### 커스텀 학습 3회 실패 (Windows 환경)
| Attempt | Config | Result |
|---------|--------|--------|
| 1 | batch_size=1, vlm=False | 평균 액션 |
| 2 | batch_size=8, vlm=False | 평균 액션 |
| 3 | batch_size=8, vlm=True | 평균 액션 |

### Root Causes
1. **Action Expert 랜덤 초기화**: `SmolVLAConfig()` 대신 `from_pretrained("lerobot/smolvla_base")` 사용 필수
2. **정규화 누락**: 공식은 MEAN_STD preprocessor 적용
3. **LR 스케줄러 없음**: cosine decay + warmup 필요

### 해결
```bash
# 공식 CLI (이것만 사용!)
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_pick \
  --dataset.root=lerobot_dataset_v4 \
  --batch_size=8 \
  --steps=50000 \
  --output_dir=outputs/smolvla_official
```

### 배포 실패 (2026-02-11, Linux 환경)
| Attempt | Data | Steps | Result |
|---------|------|-------|--------|
| 1 | 50ep (68% SHALLOW) | 50K | 그리퍼 미작동, Wrist_R -92° 폭주 |
| 2 | 동일 | 50K | 한 방향 드리프트, 파지 동작 없음 |

### 배포 실패 Root Causes
1. **데이터 부족**: 50 에피소드, DEEP 9개뿐 → 모델이 "내려가서 잡기" 안 배움
2. **Gripper 편향**: 대부분 프레임 gripper closed → open 동작 미학습
3. **Closed-loop drift**: 작은 오차 누적 → OOD → 한 방향 드리프트
4. **오프라인 ≠ 온라인**: 오프라인 L2=2.53° 양호해도 실제 배포 실패 가능

## Current Status (2026-04-28)

> Historical snapshot — 현재 상태의 진실은 `START_HERE.md` (상태 ≠ 규칙 원칙).

### Completed
- **v6 데이터 수집 완료 (4/01)**: 50 ep, 6942 frames, L-F 텔레옵, single Azure Kinect (`lerobot_dataset_v6/`)
- **v6 학습 완료 (4/05)**: 50K steps, smolvla_base pretrained, batch=8 (4090 5.2h)
- **v6 배포 SUCCESS (4/9)**: Plan 3 = JOINT_SPEED_CAPS gripper-only unlock (`speed=1000`). 유저 물리 검증: **다양한 위치/방향 sponge 전부 파지 성공**. git commit `2e840e4`
- **Kinect↔RoArm calibration 완료 (4/15)**: 빨간 스티커 마커, RMSE 10.13mm. git commit `a217cd3`
- **현실 측정 완료 (4/24)**: Hand-eye solve, table plane (z=-12.12mm RMSE 1.24mm), sponge poses 50ep. git commit `1f0d52e`
- **Sim env 구축 완료 (4/24)**: Isaac Sim (`isaaclab` env) + URDF + Kinect calib pose + table USD. SigLIP 0.7222 (48/50 GO ≥0.70). Joint replay RMSE 0.43°. `sim_v1/` (87MB lerobot v3)
- **Stacking scene 시각 검증 (4/28)**: [sim_renders_v2/stacking_initial.png](sim_renders_v2/stacking_initial.png) — A/B/Temp Layout 정확
- **B200 학습 reproducibility 검증 (4/28)**: 4090 동등 (loss bit-exact, weight diff frozen 378/500 bit-exact, max\|diff\|=0.0319 saturate). 1.4h vs 5.2h (3.7×). git commit `18abcef`
- **Stacking task pivot (4/24)**: 교수님 target = N=2 sponge stacking (3-step pick-place). Layout A(+280,0)/B(+280,+130)/Temp(+280,-110)
- **Sim2real gap 정량화**: SigLIP 0.7222, sim 70% brighter than real (dome light), LEFT zone weakest

### v6 Stacking Feasibility Analysis (4/28)
- Pick z 분포 ✅ in-distribution (elbow > 90: 22.3%, elbow < 50: 33.5%)
- Place 동작 ❌ v6에 없음 (sim demos 필수)
- Tower context image ❌ OOD (single sponge만 학습)
- v6 trajectory ~50% 재사용 가능, place + tower visual은 sim에서 새로 학습

### Active Blockers / Pending Decisions
1. Stacking 3-step vs 4-step 순서
2. Curriculum 도입 (Phase A 단독 pick → B 1-stack → C 2-stack)?
3. Safety limit hard-code (`z_world > +148+3mm`)?
4. Sim demo 생성 방식 (Procedural IK vs Leader teleop in sim vs Isaac Lab Mimic)
5. B200 SERVER 5K/10K/15K cleanup (1개월 대여)
6. 단톡방 발송 (Vulkan ICD 정책 미답)

### Next Steps (Phase ST-A → ST-B → ST-C, 2-3주)
1. **ST-A (1-2일)**: stacking_scene.py 2-sponge 패치 (현재 1 sponge spawn) + procedural pick-place script 설계
2. **ST-B (1-1.5주)**: Sim에서 50-100 stacking demos 생성 → `sim_to_lerobot.py` 변환 → Co-training (v6 real + sim) → **B200 finetune 1.5h**
3. **ST-C (3-7일)**: Real deploy A→Temp → A→B → Temp→B 3-step, dataset_mean 시작

## Research Verification Rules (연구 검증 — 2026-03-10 실수에서 배운 것)

> **배경**: 2026-03-10에 "연구 갭" 5가지를 제시했으나 4/5가 거짓이었음.
> 원인: 충분한 검색 없이 "없다"고 단정. 논문 제목의 단어를 잘못 해석.

### 절대 규칙

| Rule | Why | 위반 사례 |
|------|-----|----------|
| **"없다/최초"는 반드시 10개+ 검색어로 검증** | 한두 번 검색으로 "없다"고 단정하면 거짓 positive | "RGBD-VLA 없음" → 실제 8개+ 존재 |
| **논문 제목의 단어를 문맥 없이 해석 금지** | "Depth"가 depth 카메라인지 network depth인지 확인 필수 | RD-VLA의 "Depth" = 네트워크 깊이 |
| **"갭 발견" 시 반증 검색 먼저** | 갭을 주장하기 전에 그 갭을 채운 논문을 적극 검색 | "adaptive chunking 없음" → MoH 존재 |
| **arXiv ID 있으면 반드시 확인** | 논문 실존 여부 + 내용 일치 검증 | pi0.6 → 실제 π\*₀.₆ (5B, RECAP) |
| **"X가 유일/최초" 주장 전에 경쟁자 최소 5개 검색** | 주장의 강도에 비례하는 검증 필요 | "SmolVLA가 유일한 로컬 학습 VLA" 등 |
| **분야별 최신 서베이/메타분석 먼저 확인** | 개별 검색보다 서베이가 전체 그림 제공 | ICLR 2026 VLA 메타분석 활용 |

### 검증 프로세스 (연구 갭 주장 시)

```
1. "X가 없다" 주장하려면:
   ├── 최소 3가지 다른 검색어로 검색
   ├── 최소 2개 소스 (arXiv, Google Scholar, Semantic Scholar)
   ├── 2024-2026 논문 중심으로 확인
   └── 반증 논문 1개라도 발견 시 → 주장 철회

2. "세계 최초" 주장하려면:
   ├── 위 1번 + 관련 학회 proceedings 확인
   ├── 유사 논문의 Related Work 섹션 확인
   └── 확신도를 명시: HIGH/MEDIUM/LOW

3. 검증 실패 시:
   ├── 즉시 정정 (정정 경위 + 올바른 정보)
   ├── ResearchPlan.md에 ⚠️ 정정 마크 추가
   └── 이전 주장을 삭제하지 말고 정정 기록 유지
```

### 근본 원인 분석 (2026-03-10 실수)

| 실수 유형 | 원인 | 방지책 |
|-----------|------|--------|
| 확증 편향 | "갭을 찾고 싶다" → 갭이 아닌 증거 무시 | 반증 검색을 먼저 수행 |
| 검색 부족 | 1-2개 키워드만 검색 | 최소 3개 검색어 × 2개 소스 |
| 용어 오해 | "Depth" = depth camera라고 가정 | 논문 abstract/method 반드시 확인 |
| 시간 지연 | 2025 중반 기준 지식으로 2026 주장 | 최신 arXiv (최근 6개월) 필수 확인 |
| 과대 주장 | "zero papers" 같은 절대적 표현 | "우리 검색 범위 내에서" 등 한정어 사용 |

## Reference

### External
- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA: https://huggingface.co/docs/lerobot/en/smolvla
- RoArm M3 PR: https://github.com/huggingface/lerobot/pull/820

### Sim env (4/24, 4/28)
- [sim_scripts/stacking_scene.py](sim_scripts/stacking_scene.py) — Stacking 씬 spawn (Layout A/B/Temp)
- [sim_scripts/replay_v6_sim.py](sim_scripts/replay_v6_sim.py) — V6 trajectory sim replay (50ep ✓)
- [sim_scripts/sim_to_lerobot.py](sim_scripts/sim_to_lerobot.py) — Sim → LeRobot v3 변환기
- [sim_scripts/kinect_calib.yaml](sim_scripts/kinect_calib.yaml) — Kinect intrinsic + extrinsic
- [sim_scripts/table_plane.json](sim_scripts/table_plane.json) — Table plane fit (-12.12mm)
- [sim_scripts/sponge_poses.json](sim_scripts/sponge_poses.json) — 50 ep 별 sponge 위치
- [sim_v1/](sim_v1/) — Sim replay LeRobot v3 dataset (87MB)
- [sim_renders_v2/](sim_renders_v2/) — 50ep frame PNGs + tracking RMSE

### Calibration (4/15, 4/24)
- [claudedocs/marker_real_photo.png](claudedocs/marker_real_photo.png) — 실제 빨간 마커 사진
- [claudedocs/marker_urdf_truth.png](claudedocs/marker_urdf_truth.png) — URDF 정답 비교
- [claudedocs/stepDE_siglip50_sim_v1_20260424.md](claudedocs/stepDE_siglip50_sim_v1_20260424.md) — SigLIP 0.7222 GO 분석
- [claudedocs/session_20260424_stacking_design_pivot.md](claudedocs/session_20260424_stacking_design_pivot.md) — N=2 stacking design
