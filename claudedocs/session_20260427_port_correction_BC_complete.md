# Session 2026-04-27 — B/C 단계 정정 완료

## 목적
4/26 세션이 시작한 포트 매핑 + SDK 몽키패치 정정 작업의 B/C 단계 완료. 4/24 (7) "Follower HW 실패" 오진 회복.

## C 단계 — 1 파일

`sim_scripts/kinect_handeye_capture.py:31-54`:
- `_sdk_common.DataProcessor._process_received = lambda self, data, genre=None: None` (데이터 파싱 죽임)
- → `from roarm_sdk.common import DataProcessor, JsonCmd, handle_m3_feedback` + `_silent_process` pattern
- WARNING 주석 추가 ("do NOT replace with `lambda *a, **k: None`")

## B 단계 — 21 파일

### Critical (deploy/eval target)
- `deploy_smolvla.py:12, 394`: USB0 → USB1
- `eval_deployment.py:88, 191, 218, 366, 374`: USB0 → USB1 (5곳)
- `calibrate_azure_kinect.py:23, 291`: USB0 → USB1

### Single-arm legacy/fallback
- `collect_data.py:38, 405`: USB0 → USB1 (deprecated 키보드 조작 단일팔)
- `collect_data_manual.py:206, 1113`: USB1 (단일팔 fallback)
  - `:13, 14, 1121` Leader docstring 3곳 USB0 보존 (정답)

### Config files
- `lf_teleop_config.yaml`, `lf_teleop_nocam_config.yaml`, `lf_teleop_camera_config.yaml`: leader/follower swap
  - leader=USB0 (팔 #1, 클램프), follower=USB1 (팔 #3, 카메라 촬영)
  - yaml.safe_load 파싱 검증 PASS

### Recovery scripts (default 보존, docstring 추가)
- `scan_servos.py`, `reset_robot.py`: 양 포트 사용법 docstring 명시

### Legacy diagnostic (LEGACY 주석만, default 보존)
- `test_phase0_*.py` 6개 + `test_gripper_check.py` + `test_read_only.py` (8개)
- 각 파일에 `# LEGACY pre-L-F (2026-04-01): USB0=Leader (gripper clamp). Edit to /dev/ttyUSB1 for Follower.` 추가
- 이유: Mar 26 작성 단일팔 시기 스크립트, 하드코드 + argparse 없음, 보존 가치

### LeRobot 백업
- `lerobot_backup/configs.py:512`: USB0 → USB1 (`follower_arms.main`)
- `lerobot_backup/test_lerobot_roarm.py:7-8`: USB0 → USB1 (deploy test)

## HW + SDK 검증 통과

```
Leader USB0:   [0.53, 1.76, 91.49, 0.26, 0.0, 1.32]   — home, 4/26 baseline 일치
Follower USB1: [0.53, 0.70, 92.55, 0.18, -0.09, 0.18] — home, 4/26 baseline 일치
Azure Kinect:  Depth + 4K + Mic 모두 USB 인식 OK
```

## 잔존 USB0 (전수 재검증, 모두 정답)

| 분류 | 개수 | 위치 |
|---|---|---|
| Leader 정답 | 5 | collect_data_manual.py:13,1121 / test_leader_follower.py:52 / yaml 3개 leader_arms |
| Recovery 양쪽 가능 | 4 | scan_servos.py:5,13 / reset_robot.py:5,50 |
| B8 LEGACY 마킹 | 8 | test_phase0_*.py 6개 + test_gripper_check + test_read_only |
| 외부 라이브러리 docstring | 1 | lerobot/src/lerobot/motors/motors_bus.py:547 (우리 코드 아님) |
| 양쪽 표기 코멘트 | 1 | lerobot_backup/test_lerobot_roarm.py:7 (양쪽 명시) |

## 모든 파일 py_compile + yaml.safe_load PASS

## 별도 발견 (B 범위 밖)
- `calibrate_azure_kinect.py`: SDK 몽키패치 미적용 (line 19에서 `from roarm_sdk.roarm import roarm`만 직접) → print spam 발생하지만 default `_process_received`는 데이터 파싱 정상이라 동작은 함

## 미완료 (다음 세션)
- **D 단계**: `.claude/agents/*.md` + `.claude/agent-memory/**/*.md` 포트 매핑 스왑 (서브에이전트 오염 방지)
  - 식별된 파일: `.claude/agents/deploy-agent.md:31`, `.claude/agents/robotics-hardware.md:38`, `.claude/agent-memory/Hardware & Sensing Specialist/project_hardware_state.md`, `.claude/agent-memory/deploy-agent/MEMORY.md:38`
- **E 단계**: MEMORY.md 크기 정리 (40.3KB > 24.4KB 한계)
- **원래 작업**: Stacking 4 결정, Follower(USB1) gripper stroke 실측, finger 캘리퍼

## 신규 메모리 파일
- `project_hardware_inventory.md` — 스펀지 125×47×20mm 다수 보유, 양 팔 포트, Kinect/IMX335/ZED Mini, 3D프린터, Quest VR, GPU
- MEMORY.md Topic Files index에 1줄 추가
