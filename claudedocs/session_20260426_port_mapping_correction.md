# Session 2026-04-26 — 포트 매핑 + SDK 몽키패치 정정

## 발단 (4/24 → 4/26 계승된 오진)

4/24 entry "(7) Follower HW 실측 실패 ⚠️"는 **오진**:
- 실제로는 USB0(=Leader)에 `torque_set(1)` + gripper sweep 명령 보냄
- 원인: CLAUDE.md `Robot=USB0 (follower)` 표기 → 4/24 세션이 그대로 신뢰
- 진실 (4/01 물리검증, tech_leader_follower_setup.md): **USB0=Leader, USB1=Follower**

## 4/26 검증 결과

### HW 정상
- Raw T:106 reset 후 양쪽 포트 실제 서보 응답 확인
- SDK joint_read 양쪽 8/8 good (`_silent_process` 적용 후):
  - USB0(Leader): `[0.35, 1.76, 91.32, 0.35, 0.0, 1.32]` — home 근처
  - USB1(Follower): `[0.44, 0.26, 91.32, 0.09, -0.09, 0.35]` — home 근처
- 4/24 "서보 물리 응답 없음" 결론 **틀림**, 실제로는 SDK 몽키패치 버그

### 근본 원인 1 — SDK 몽키패치 잘못된 패턴
`gripper_stroke_probe.py` (4/24 작성) + `sim_scripts/kinect_handeye_capture.py:35` (4/18 작성):
```python
sdk_common.DataProcessor._process_received = lambda *a, **k: None  # ❌
```

`_process_received` 실체 (`roarm_sdk/common.py:317`):
- `print(data)` 한 줄만 있는 게 아님
- `data['x'/'y'/'z']` 추출 + `handle_m3_feedback()` 호출 = **데이터 파싱 핵심**
- `lambda: None` → `joints_angle_get()` 등 모든 read API → `None` → subscript 에러

정답 (`collect_data_manual.py:44-60`):
```python
from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback
def _silent_process(self, data, genre):
    if not data: return None
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

### 근본 원인 2 — CLAUDE.md 포트 매핑 오류
- 2026-03 중순까지: 로봇 1대 = USB0 (단일팔 시기)
- 2026-04-01: 2대 체제 전환, 새 팔 #3 = USB1 = Follower, 기존 팔 #1 = USB0이 **Leader로 역할 변경**
- CLAUDE.md + 대부분 default ports + lf_teleop_*.yaml = **업데이트 안 됨** → 모순 영구화

### Gripper Probe 1차 결과 = 무효
`cmd 0/30/60/90 → state 1.32/5.63/31.55/57.48°` 는 **Leader 팔 데이터** (그리퍼 클램프 장착, OOD).
Follower 재측정은 robot 전원 복구 후.

## 모순 분포 (전수 조사)

### 두 세계의 충돌

**올바른 매핑 (USB0=Leader, USB1=Follower) — 11개 소스**:
- MEMORY.md 4/9~4/24 entries
- tech_leader_follower_setup.md (4/01 물리검증)
- tech_roarm_m2_vs_m3_verification.md:12-14
- project_current_state.md:23-25
- experiment_log.md:229-253 (4/01)
- session_20260418_kinect_calib.md:43-44
- session_20260424_step3_5a_execution.md:69
- test_leader_follower.py:52-53
- collect_data_manual.py (--leader-port help)
- sim_scripts/kinect_handeye_capture.py:39
- 4/14 Stage 1 deploy 명령 `--port /dev/ttyUSB1`

**잘못된 매핑 (USB0=Follower, USB1=Leader) — 15+ 소스**:
- **CLAUDE.md** (이번 세션에 정정됨)
- `lf_teleop_*.yaml` 3개 (전부)
- Python defaults: deploy_smolvla.py, collect_data.py, eval_deployment.py, calibrate_azure_kinect.py, test_phase0_*.py (6개), scan_servos.py, reset_robot.py, test_gripper_check.py, test_read_only.py, test_phase0_dual.py
- Markdown 레거시: RoArm_Spc1.md, LINUX_SETUP_GUIDE.md, LINUX_MIGRATION_GUIDE.md, TASKS.md, DEPENDENCIES.md, ResearchPlan.md:85-86, train_50ep_strategy_analysis.md
- Claudedocs: PROJECT_AUDIT.md, AGENT_PERSONAS.md:141, deploy_data_collection_comparison.md, DATA_COLLECTION_STRATEGY.md
- Agent: .claude/agents/deploy-agent.md:31, robotics-hardware.md:38
- Agent memory: .claude/agent-memory/Hardware & Sensing Specialist/project_hardware_state.md, .claude/agent-memory/deploy-agent/MEMORY.md:38
- Backup: lerobot_backup/configs.py:512, lerobot_backup/test_lerobot_roarm.py

## A 단계 — 정정 완료 (이번 세션)

### CLAUDE.md
| 위치 | 정정 |
|---|---|
| line 26-27 | Follower=USB1, Leader=USB0 표 (팔 #3/#1, 그리퍼 클램프, 카메라 명시) |
| line 56-57 | Key Commands 로봇 복구에 포트 코멘트 |
| line 62-63 | HW 테스트 — Leader+Follower 양쪽 검사 |
| line 129 | SDK API 예시 → USB1 + 코멘트 |
| line 139 | 모듈명 `sdk_common` → `roarm_sdk.common` |
| line 143-164 | `_silent_process` 패턴 코드 + `lambda: None` 금지 경고 |
| line 174-175 | USB diagram 스왑 |
| line 180 | Motor Recovery 헤더 코멘트 |

### MEMORY.md
- 4/24 entry (line 48)에 ⚠️ 4/26 정정 inline append (HARD RULE #8 오버라이드 금지 준수, 원문 보존)

### 코드
- `gripper_stroke_probe.py`: USB0→USB1, `lambda: None`→`_silent_process`
- Leader(USB0) torque OFF 복구 (이전 세션이 잘못 ON 시킨 것)

## B/C 단계 — 미진행 (유저 승인 대기)

### B. 코드 default 정정
- `deploy_smolvla.py:394` `default="/dev/ttyUSB0"` → `"/dev/ttyUSB1"` (deploy target = follower)
- `eval_deployment.py:88,191,218,366,374` 동일
- `calibrate_azure_kinect.py:23,291` 동일
- `collect_data.py:38,405` 동일 (단일팔, 사실상 deprecated)
- `collect_data_manual.py:206,1113` 단일팔 fallback default만
- `lf_teleop_*.yaml` 3개 — leader/follower 스왑 (단, 사용 여부 grep 먼저)
- `scan_servos.py`, `reset_robot.py` — recovery는 양쪽 가능, default 그대로 + docstring 명시
- `test_phase0_*.py` 6개 — phase0=초기 단독팔, 보존 가치, 코멘트만

### C. SDK 마이너 버그
- `sim_scripts/kinect_handeye_capture.py:35` — `lambda: None` → `_silent_process`

### D. 추가 확인 (B/C 후)
- `lf_teleop_*.yaml` 실제 사용 여부 grep (안 쓰면 archive)
- `.claude/agents/*.md`, `.claude/agent-memory/**/*.md` 포트 매핑 스왑
- 레거시 markdown에 "⚠️ ARCHIVED 2026-03 이전 기준" 배너

### E. MEMORY.md 위생 (별도 승인 필요)
- 현재 40.3KB / 24.4KB 한계 초과 → index 후반부 잘림 (Topic Files 목차 등)
- 4/24 entries (3개)가 각 1500-2500자 → entry 200자로 축소, 상세는 session topic file로 이동

## 현재 상태 (4/26 종료 시점)
- Robot 전원 OFF (유저 확인)
- Leader(USB0) torque 직전 ON 상태였으나 OFF 복구 완료
- gripper_stroke_probe.py = 정정된 상태 (USB1, _silent_process)
- CLAUDE.md + MEMORY.md 4/24 entry 정정 완료
- A 단계 ✅, B/C/D/E 대기

## 의심 리스트 (다음 세션 검증)
1. CLAUDE.md `## Hardware Specs` 표 wrist_pitch range — 4/14 entry에 "URDF ±90° vs CLAUDE.md ±110° → URDF 틀림" 기록. CLAUDE.md는 정확. URDF는 정정 완료(±1.92rad). 이 부분 변경 불필요.
2. lf_teleop_*.yaml — `lerobot-teleop` CLI가 사용? 또는 archive? grep 필요.
3. agent-memory 파일들 정정 시 서브에이전트 영향. 정정 후 첫 소환에서 검증.
