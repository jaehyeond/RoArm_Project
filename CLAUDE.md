@AGENTS.md

# CLAUDE.md — Claude Code 전용

도구 중립 프로젝트 규칙 전부(부트 절차, HARD RULES, 안전 제약)는 `AGENTS.md`가 단일 소스이며,
위 `@AGENTS.md` 임포트로 자동 인라인 로드된다. 하드웨어/파이프라인/명령어/세션 프롬프트
레퍼런스는 2026-08-25에 `docs/reference/`로 분리됐다 (자동 로드 아님 — `AGENTS.md`의 참조 표 참고).
이 파일에는 Claude Code 전용 내용만 남긴다: 12-agent 팀, 스킬/auto-memory 워크플로우.
규칙 추가·수정은 `AGENTS.md`에서 한다 (상태는 `START_HERE.md` 계열 — 상태 ≠ 규칙).

## Session Workflow (Claude 전용 세션 운영 규칙)

> Authoritative end-of-session procedure lives in
> `docs/reference/session_protocol.md` (2026-08-25 `AGENTS.md`에서 분리, 원문 그대로).
> `AGENTS.md ## Current-State Protocol`은 부트 6단계와 파일 역할을 계속 보유한다.
> This section retains the Claude-specific rules and clarifies
> the relationship between project-state docs and auto-memory.

| Rule | Why |
|------|-----|
| **HANDOFF.md / `/handoff` 자동 생성/발동 금지** | /handoff 스킬은 이 프로젝트 워크플로우가 아님. `START_HERE.md` overwrite + new `session_*.md`로 대체. |
| **/half-clone 절대 사용/제안 금지** | 유저 반복 지시 (auto-memory HARD RULE #11). Context 95% → end-of-session update + new session boot, NOT /half-clone. |
| **context 차면 project-state + MEMORY.md 둘 다 업데이트** | Project-state (`START_HERE.md`, `EXPERIMENT_LEDGER.md`, new `session_*.md`, `DECISIONS.md` if durable) = repo continuity. Auto-memory (`MEMORY.md`) = user-level habits, HARD RULES, recent session index. 둘은 보완 관계. |
| **다음 세션용 continuation prompt 제공** | Project-state file-based boot이 1차. Continuation prompt는 유저가 즉시 새 세션 시작 시 보조 진입점. `docs/reference/session_protocol.md`의 "Session boot prompt" 텍스트 사용. |
| **중요 결과는 claudedocs/ 파일로 저장** | 파일 기반 상태 보존. Detail은 `session_*.md`, summary table은 `EXPERIMENT_LEDGER.md`, durable lesson은 `DECISIONS.md`. |

```
세션 종료 프로세스 (docs/reference/session_protocol.md "End-of-session update prompt"의 단축형):
1. START_HERE.md → 현재 truth + active pivot + next step으로 overwrite
2. EXPERIMENT_LEDGER.md → 주요 run/result row append
3. DECISIONS.md → durable lesson만 append (Dxxx 번호)
4. claudedocs/session_YYYYMMDD_*.md → detailed append-only 새 파일
5. ~/.claude/.../memory/MEMORY.md → recent sessions index prepend (auto-memory HARD RULE #8)
6. (option) continuation prompt → 유저에게 텍스트 출력 (95% emergency 또는 명시 요청 시)
7. HANDOFF.md → 절대 건드리지 않음
```

## Agent Team (16 agents = 12 + Execution Roles 4)

### Engineering Workers (3개 — 코드 실행)

| Agent | Role | File Ownership |
|-------|------|----------------|
| **data-agent** | 데이터 분석, 수집 전략 | `data_*.py`, `collect_data_manual.py` |
| **pipeline-agent** | 학습 설정, 체크포인트 평가 | `train_*.py`, `run_official_train.py` |
| **deploy-agent** | 추론 루프, 배포 개선 | `deploy_*.py` |

### Research Agents (9개 — 분석, 실험, 논문)

| Team | Agent | Role | File Ownership |
|------|-------|------|----------------|
| **A. Robotics** | A1 robotics-manipulation | 궤적 분석, 관절 검증 | `trajectory_*.py` |
| | A2 robotics-sim2real | 시뮬레이션 연결 | `sim_*.py` |
| | A3 robotics-hardware | 하드웨어 테스트/캘리브 | `hw_*.py`, `calibrate_*.py` |
| **B. Physical AI** | B1 pai-vla-model | 모델 아키텍처 분석 | `model_*.py` |
| | B2 pai-data-efficiency | 증강, 자기개선 루프 | `augment_*.py`, `self_improve_*.py` |
| | B3 pai-deployment | 안전 모니터링, OOD 감지 | `monitor_*.py`, `safety_*.py` |
| **C. Research** | C1 research-experiment | 실험 매트릭스, 평가 | `experiment_*.py`, `eval_*.py` |
| | C2 research-analysis | 통계, 시각화 | `analysis_*.py`, `figure_*.py` |
| | C3 research-writing | 논문 LaTeX | `paper/*` |

### Execution Roles (4개 — 시뮬 실행·회계·재생·감사, 2026-09-19 채택 D492)

W14~W19 에서 worktree 를 뗄 때마다 같은 계약을 TASK_SPEC 에 다시 적던 것을 역할 정의로 묶은 것이다.
규칙의 단일 소스는 `AGENTS.md` 이며 역할 파일은 규칙을 복제하지 않고 `파일:줄` 로 참조만 한다.
계약 정본은 언제나 배정 시 받는 `TASK_SPEC_*.md` 이고, 역할 파일과 어긋나면 TASK_SPEC 이 이긴다.
초안·대응표 원본 = `claudedocs/research/role_agents_20260918/README.md`(worktree `w19-role-agents`).

| Agent | Role | 산출 경계 | 안 하는 것 |
|-------|------|-----------|-----------|
| **deme-runner** | 동결 revision 사본 → preflight → smoke → 승인된 GPU 실행 1회 → 원자료·영수증 | `claudedocs/runtime_logs/**` | 물리 파라미터 설계, 판정 승격 |
| **raw-accountant** | 저장된 원자료를 규약대로 파생 재계산 + 항목별 PASS/FAIL (CPU, 물리 0) | 같음 | 새 실행, 성공 선언 |
| **replay-renderer** | 같은 원자료를 Rerun RRD/RBL·결정 화면·Isaac 관절로 재생 + 육안 검수 | 같음 | 새 물리, 과학 판정 변경 |
| **independent-auditor** | 워커 산출을 NumPy-only 독자 구현으로 재계산해 사전 등록 항목 감사 | 같음 (`Edit` 미부여) | 코드 수정, 수정 대행 |

넷 다 `model: claude-opus-5`, `disallowedTools: Task`(재귀 소환 차단),
PreToolUse `Bash → safety-check.sh` + `Write|Edit → file-ownership-check.sh <이름>`.
기존 12-agent 의 `permissionMode: plan` 은 실행 역할과 맞지 않아 넣지 않았다.

### 소환 규칙 (상황별 2-3개)

| 상황 | 소환 에이전트 |
|------|-------------|
| 데이터 수집 | data-agent + A3(Hardware) + B2(Data Efficiency) |
| 학습 설정 | pipeline-agent + B1(VLA Model) + C1(Experiment) |
| 배포 테스트 | deploy-agent + A1(Manipulation) + B3(Deployment) |
| 논문 작성 | C3(Writing) + C2(Analysis) + B1(VLA Model) |
| Sim-to-Real | A2(Sim2Real) + B1(VLA Model) |

### 교차 검증 프로세스
```
1. Worker가 코드/결과 생성
2. Research agent가 critical questions로 검증
3. 문제 발견 → worker에게 수정 권장
4. 실험 필요 → C1이 설계, worker가 실행
```

Safety hooks (전 에이전트 공통):
- `safety-check.sh`: git, 로봇 직접 제어(`serial`·`/dev/ttyUSB`·`torque_set`·`joints_angle_ctrl`·`move_init`·`T:106`),
  `rm -rf`, `lerobot-train` 차단 — 2026-09-19 실측 확인, **확대 불필요**
- `file-ownership-check.sh`: agent별 파일 소유권 강제 (**16개 전부 등록** = 12 + Execution Roles 4).
  미등록 이름은 fail-closed 로 전면 차단. 2026-09-19 추가: **상태 원장 배타 차단** —
  어떤 agent 이름이든 `START_HERE.md`·`claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,LEDGER_RECENT}.md`·
  `claudedocs/relay/` 쓰기는 exit 2 (`AGENTS.md` 원장 배타 소유 규칙의 기계적 강제).
- 상세 페르소나: `claudedocs/AGENT_PERSONAS.md`

## Memory (Topic Files) — `~/.claude/projects/.../memory/`

- `tech_b200_server_setup.md` — B200 NHN/Sogang env 셋업 + S0-S16 + reproducibility 검증 (4/28)
- `tech_lerobot_camera_keys.md` — single Kinect vs SmolVLA 3-camera default 매핑 fix (4/28)
- `tech_leader_follower_setup.md` — L-F 포트매핑 (USB0=Leader, USB1=Follower), 물리배치
- `tech_critical_lessons.md` — 실패 교훈 33+개
- `tech_gripper_grasp_anchors.md` — Gripper jaw stroke 측정 + Option A/B/C grasp 시퀀스 (4/27)
- `tech_servo_observer_effect.md` — Read 빈도가 servo motion 속도 결정 (4/27)
- `experiment_log_v6_deployment.md` — v6 4/9 Plan 3 SUCCESS 분석
- `project_hardware_inventory.md` — 스펀지/로봇/카메라/3D프린터/VR 인벤토리
- `project_corl2026_direction.md` — CoRL+석사논문 전략 (3-VLA 비교, Phase-Selective)
