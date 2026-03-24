# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

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
| Robot | RoArm-M3-Pro via `/dev/ttyUSB0` (follower) |
| Leader | RoArm-M3-Pro via `/dev/ttyUSB1` (leader, L-F 모드 시) |
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

# 로봇 복구 (모터 버스 문제)
python scan_servos.py /dev/ttyUSB0
python reset_robot.py

# 하드웨어 테스트
python -c "from pyk4a import PyK4A; k4a = PyK4A(); k4a.start(); print('Kinect OK'); k4a.stop()"
python -c "from roarm_sdk.roarm import roarm; arm = roarm('roarm_m3', '/dev/ttyUSB0', 115200); print('Robot OK'); arm.disconnect()"
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

arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB0", baudrate=115200)

angles = arm.joints_angle_get()           # → list[6] (degrees)
arm.joints_angle_ctrl(angles=[0]*6, speed=500, acc=200)
arm.torque_set(cmd=1)                     # 1=on, 0=off (keyword arg cmd 필수!)
arm.move_init()                           # 초기 위치
arm.disconnect()
```

### SDK Bugs & Workarounds
- **print(data) 스팸**: `sdk_common.DataProcessor._process_received` 몽키패치로 억제
- **BaseController 로거**: CRITICAL 레벨로 설정 (백그라운드 스레드 디코드 에러)
- **safe_joints_angle_get()**: 5회 재시도 (간헐적 None/KeyError 대응)

### USB Configuration

```
Laptop ──USB──→ [USB Hub]
                    │
        ┌───────────┴───────────┐
        ↓           ↓           ↓
  Azure Kinect  Follower     Leader
     (DK)     (/dev/ttyUSB0) (/dev/ttyUSB1)
```

## Motor Recovery (모터 응답 없음)

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

**카메라 위치 변경 = 모든 데이터 무효 = 재수집 필수!**

## Session Workflow (세션 운영 규칙)

| Rule | Why |
|------|-----|
| **HANDOFF.md 자동 생성/발동 금지** | /handoff 스킬은 이 프로젝트 워크플로우가 아님 |
| **context 차면 MEMORY.md 업데이트** | 날짜별로 세션 결과 기록 (간결하게) |
| **다음 세션용 continuation prompt 제공** | 유저가 새 대화에 붙여넣을 구체적 프롬프트 출력 |
| **중요 결과는 claudedocs/ 파일로 저장** | 파일 기반 상태 보존 |

```
세션 종료 프로세스:
1. 중요 결과 → claudedocs/에 파일 저장
2. MEMORY.md → 날짜별 세션 결과 인덱스 업데이트
3. continuation prompt → 유저에게 텍스트로 출력
4. HANDOFF.md → 절대 건드리지 않음
```

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
| **카메라 고정 (삼각대/클램프)** | 위치 변경 시 전체 데이터 무효 |
| **Azure Kinect만 사용** | VLA 데이터는 반드시 pyk4a |
| **100+ 에피소드 목표** | 51개는 부족했음 (elbow 깊이 다양성 부족) |

## LeRobot Integration

### 데이터 수집 방식
**팔 1개 + 토크 OFF 수동 모드**:
- `collect_data_manual.py` → Azure Kinect (pyk4a)
- 토크 OFF → 손으로 로봇 직접 움직임
- Leader-Follower: 구현됨, 듀얼 팔 보유 시 사용 가능

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

## Agent Team (12 agents)

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
- `safety-check.sh`: git, 로봇 직접 제어, rm -rf, lerobot-train 차단
- `file-ownership-check.sh`: agent별 파일 소유권 강제 (12개 전부 등록)
- 상세 페르소나: `claudedocs/AGENT_PERSONAS.md`

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

## Current Status (2026-02-11)

### Completed
- Git repo 정리: Isaac Sim/RL 제거, GitHub push 완료 (SSH)
- Windows → Linux 완전 이관: COM 포트, 패치, 경로 모두 정리
- Agent team hooks: .ps1 → .sh, fail-closed 보안 강화, Linux 검증 완료
- **환경 구축 완료**: conda `roarm` env (Python 3.11 + PyTorch 2.7.1+cu126 + LeRobot 0.4.4 + SmolVLA + roarm_sdk)
- **Azure Kinect SDK 설치 완료**: libk4a 1.4.2 + pyk4a 1.5.0 + udev rules
- **데이터 수집 (v1)**: 50 에피소드, 10,803 프레임 (DEEP 18%, APPROACH 14%, SHALLOW 68%)
- **학습 완료 (v1)**: 50K steps, loss 0.007, smolvla_base pretrained
- **오프라인 테스트 통과**: Overall Std 21.55°, Mean L2 2.53°, 50K checkpoint 최적
- **배포 2회 실패 (2026-02-11)**: 그리퍼 미작동, 한 방향 드리프트, 파지 실패

### Deployment Failure Analysis (2026-02-11)

| 문제 | 증상 | 원인 |
|------|------|------|
| 그리퍼 미작동 | 한번도 열리지 않음 (2-4°) | 학습 데이터에 gripper open 프레임 부족 |
| 단방향 드리프트 | 모든 관절이 한 방향으로만 천천히 이동 | Closed-loop 오차 누적 → OOD |
| Wrist_R 폭주 (Run 1) | -3° → -92° | 4σ OOD drift |
| Elbow 상승 | 13° → 36° (위로만) | DEEP 에피소드 부족 (9/50) |

### Next Steps
1. **데이터 추가 수집**: 100+ 에피소드 (DEEP 50개+, gripper open/close 다양성)
2. **재학습**: 새 데이터셋으로 50K+ steps
3. **재배포 테스트**: dataset_mean 시작, closed-loop

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

- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA: https://huggingface.co/docs/lerobot/en/smolvla
- RoArm M3 PR: https://github.com/huggingface/lerobot/pull/820
- VLA 기술 총정리: `2026_Physical_AI.md`
- 프로젝트 감사: `claudedocs/PROJECT_AUDIT.md`
- 연구 아이디어: `claudedocs/RESEARCH_IDEAS.md`
