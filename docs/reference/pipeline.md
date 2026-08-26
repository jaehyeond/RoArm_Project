# Reference — Project Overview · Environment · Key Commands · Pipeline

> 출처: 분리 전 `AGENTS.md` **291-395행**. 원본 전체는 `docs/archive/AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25). 규칙이 아니라 레퍼런스다 —
> 명령을 실행하거나 파이프라인 파일을 건드리기 전에 읽는다.
> ⚠️ 여기 적힌 환경/버전/경로는 2026-04 시점 기준이다. 실행 전 실물 확인.

---

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
