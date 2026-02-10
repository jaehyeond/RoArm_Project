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
| Camera | Azure Kinect DK (pyk4a — libk4a 설치 필요) |
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

## Agent Team

3개 전문 agent가 협업:

| Agent | Role | File Ownership |
|-------|------|----------------|
| **data-agent** | 데이터 분석, 수집 전략 | `data_*.py`, `collect_data_manual.py` |
| **pipeline-agent** | 학습 설정, 체크포인트 평가 | `train_*.py`, `run_official_train.py` |
| **deploy-agent** | 추론 루프, 배포 개선 | `deploy_*.py` |

Safety hooks:
- `safety-check.sh`: git, 로봇 직접 제어, rm -rf, lerobot-train 차단
- `file-ownership-check.sh`: agent별 파일 소유권 강제

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

## Current Status (2026-02-10)

### Completed
- Git repo 정리: 49개 파일 (Isaac Sim/RL 제거), GitHub push 완료 (SSH)
- Windows → Linux 완전 이관: COM 포트, 패치, 경로 모두 정리
- Agent team hooks: .ps1 → .sh, fail-closed 보안 강화
- **환경 구축 완료**: conda `roarm` env (Python 3.11 + PyTorch 2.7.1+cu126 + LeRobot 0.4.4 + SmolVLA + roarm_sdk)
- LeRobot 0.4.4 구조 변경 확인 (lerobot_backup/ 파일은 구버전 API)

### Pending
- **Azure Kinect SDK**: sudo로 libk4a 설치 + pip install pyk4a 필요
- **LeRobot 로봇 통합 포팅**: lerobot_backup/ → LeRobot 0.4.4 구조 (필요시)

### Next Steps
1. **Azure Kinect SDK 설치** (sudo 필요: libk4a1.4-dev + pyk4a)
2. **USB 하드웨어 연결 테스트** (로봇 + Kinect)
3. **카메라 고정**: 삼각대/클램프로 고정, 위치 문서화
4. **데이터 수집**: 100+ 에피소드 (elbow < -30° 50개, approach 30개, 다양한 시작 20개)
5. **학습**: `lerobot-train` CLI, smolvla_base, 50K+ steps
6. **배포**: dataset_mean 시작, n_action_steps=1, closed-loop

## Reference

- LeRobot: https://github.com/huggingface/lerobot
- SmolVLA: https://huggingface.co/docs/lerobot/en/smolvla
- RoArm M3 PR: https://github.com/huggingface/lerobot/pull/820
- VLA 기술 총정리: `2026_Physical_AI.md`
- 프로젝트 감사: `claudedocs/PROJECT_AUDIT.md`
- 연구 아이디어: `claudedocs/RESEARCH_IDEAS.md`
