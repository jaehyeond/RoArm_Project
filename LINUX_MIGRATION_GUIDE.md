# Linux Migration Guide: RoArm M3 SmolVLA Pipeline

**Date**: 2026-02-07
**From**: Windows 11 Desktop (RTX 4070 Ti SUPER)
**To**: Lenovo Laptop (32GB VRAM, Linux dual-boot)
**Purpose**: Move SmolVLA VLA pipeline to Linux for new data collection

---

## 1. Overview

### What We're Moving
```
Git repo (code only, ~19K lines)
├── Core pipeline: collect → train → deploy
├── Analysis tools: dataset quality, distribution, evaluation
├── Agent team definitions & safety hooks
├── Documentation & lessons learned
└── Reference: Isaac Lab configs, LeRobot integration backup
```

### What We're NOT Moving (recollect/reinstall on Linux)
- `.venv/` (5.6 GB) → reinstall from `requirements_core.txt`
- `lerobot/` (0.4 GB) → `git clone` fresh
- `lerobot_dataset_v3/` (0.4 GB) → **OBSOLETE** (camera moved)
- `collected_data/` (25.1 GB) → **OBSOLETE** (camera moved)
- `outputs/` (11.4 GB) → copy 20K checkpoint only (optional)
- `models/` (0.85 GB) → `smolvla_base` auto-downloads from HuggingFace

### Why Recollect Data?
카메라 위치가 크게 바뀌어서 복원 불가 → 기존 51 에피소드는 전부 OOD(out-of-distribution).
SmolVLA는 Vision-Language-Action 모델이라 이미지 분포가 바뀌면 학습 데이터 재수집 필수.

---

## 2. Step 1: Git Clone

### On Linux Laptop

```bash
# GitHub에서 clone
git clone https://github.com/<YOUR_USERNAME>/roarm-smolvla.git
cd roarm-smolvla

# 또는 USB로 bare repo 복사
# Windows: git clone --bare E:\RoArm_Project roarm-smolvla.git
# USB에 복사 후 Linux: git clone roarm-smolvla.git roarm-smolvla
```

### GitHub Desktop 연동
1. Windows에서 GitHub Desktop 설치
2. `E:\RoArm_Project` → "Add Existing Repository"
3. "Publish repository" → GitHub에 push
4. Linux에서 `git clone` (HTTPS or SSH)

### GitHub Remote 설정 (Windows에서)
```powershell
cd E:\RoArm_Project
git remote add origin https://github.com/<YOUR_USERNAME>/roarm-smolvla.git
git branch -M main
git push -u origin main
```

---

## 3. Step 2: Linux Environment Setup

### 3.1 System Dependencies

```bash
# Ubuntu 22.04 기본 패키지
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    git curl wget build-essential \
    libusb-1.0-0-dev  # Azure Kinect USB

# Azure Kinect SDK (Ubuntu 22.04)
# Microsoft 공식 repo 추가
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc
sudo apt-add-repository https://packages.microsoft.com/ubuntu/22.04/prod
sudo apt update
sudo apt install -y libk4a1.4 libk4a1.4-dev k4a-tools

# Azure Kinect 권한 설정
sudo cp /lib/udev/rules.d/99-k4a.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 3.2 Python Virtual Environment

```bash
cd ~/roarm-smolvla

# venv 생성
python3.11 -m venv .venv
source .venv/bin/activate

# PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install -r requirements_core.txt

# LeRobot (소스에서 설치)
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e . && cd ..

# RoArm SDK
pip install roarm_sdk
# 또는 로컬: pip install -e /path/to/roarm_sdk/
```

### 3.3 USB 장치 권한

```bash
# RoArm M3 시리얼 포트 권한
sudo usermod -aG dialout $USER
# 로그아웃 후 다시 로그인 필요

# 포트 확인
ls /dev/ttyUSB*
# 보통 /dev/ttyUSB0 (로봇)
```

### 3.4 Hardware Test

```bash
# Azure Kinect 테스트
python -c "from pyk4a import PyK4A; k4a = PyK4A(); k4a.start(); print('Kinect OK'); k4a.stop()"

# RoArm M3 테스트
python -c "from roarm_sdk.roarm import roarm; arm = roarm('roarm_m3', '/dev/ttyUSB0', 115200); print('Robot OK'); arm.disconnect()"

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 4. Step 3: Code Patches to Remove

Linux에서는 Windows 전용 패치를 **제거**해야 합니다.

### 4.1 Path→POSIX 변환 패치 (제거)

**파일**: `run_official_train.py`, `deploy_smolvla.py`, `test_inference_official.py`, `train_eval_checkpoints.py`, `train_config_50k.py`

**제거할 코드** (각 파일에서 찾아서 삭제):
```python
# ===== 이 블록 전체 삭제 =====
# Windows Path 문제 해결
import lerobot.policies.pretrained as _pretrained
_original_from_pretrained = _pretrained.PreTrainedPolicy.from_pretrained.__func__

@classmethod
def _patched_from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
    if isinstance(pretrained_name_or_path, Path):
        pretrained_name_or_path = pretrained_name_or_path.as_posix()
    return _original_from_pretrained(cls, pretrained_name_or_path, *args, **kwargs)

_pretrained.PreTrainedPolicy.from_pretrained = _patched_from_pretrained
# ===== 여기까지 삭제 =====
```

### 4.2 Symlink→Text File 패치 (제거)

**파일**: `run_official_train.py`, `train_config_50k.py`

**제거할 코드**:
```python
# ===== 이 블록 전체 삭제 =====
# Windows symlink 문제 해결
import shutil
import lerobot.utils.train_utils as _train_utils

def _patched_update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Windows-compatible: write text file pointer instead of symlink."""
    ...

_train_utils.update_last_checkpoint = _patched_update_last_checkpoint
# ===== 여기까지 삭제 =====
```

### 4.3 Encoding 수정 (간소화)

**변경 전** (Windows):
```python
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

**변경 후** (Linux):
```python
# Linux는 UTF-8 기본. 버퍼링만 해제
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
```

### 4.4 HuggingFace Symlink 경고 (제거)

```python
# 이 줄 삭제
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

### 4.5 COM 포트 변경

모든 파일에서:
```python
# 변경 전
port = "COM3"  # 또는 "COM8"

# 변경 후
port = "/dev/ttyUSB0"  # 또는 auto-detect
```

**자동 감지 패턴** (권장):
```python
import serial.tools.list_ports

def find_roarm_port():
    """RoArm M3 자동 감지."""
    for port in serial.tools.list_ports.comports():
        if "USB" in port.description or "CH340" in port.description:
            return port.device
    raise RuntimeError("RoArm M3 not found. Check USB connection.")
```

---

## 5. Step 4: Full Workflow on Linux

### 5.1 Camera Setup (중요!)

```
카메라를 새 위치에 고정하고 절대 움직이지 마세요.
카메라가 움직이면 = 데이터 전부 무효 = 처음부터 다시

권장 setup:
- 카메라를 삼각대/클램프로 고정
- 로봇 workspace 전체가 보이는 각도
- 조명 일정하게 유지
- 카메라 위치 사진 찍어서 문서화
```

### 5.2 Data Collection (100+ episodes)

```bash
source .venv/bin/activate

# 데이터 수집 (토크 OFF 수동 모드)
python collect_data_manual.py

# 수집 후 품질 검증
python data_episode_quality.py
python data_distribution_simple.py
```

**목표**: 최소 100 에피소드
- 50개: 깊은 grasp (elbow < -30°)
- 30개: approach 동작 (다양한 경로)
- 20개: 다양한 시작 위치

### 5.3 Data Conversion

```bash
# LeRobot v3.0 format으로 변환
python convert_to_lerobot_v3.py \
    --input collected_data \
    --task "Pick up the white box"

# 변환 결과 확인
python data_distribution_simple.py
```

### 5.4 Training

```bash
# 공식 lerobot-train CLI (smolvla_base 사전학습 모델 사용)
python run_official_train.py

# GPU 모니터링 (별도 터미널)
watch -n 1 nvidia-smi

# 체크포인트 평가
python train_eval_checkpoints.py
```

**핵심 설정**:
- `pretrained_path=lerobot/smolvla_base` (필수!)
- `batch_size=8`
- `steps=20000` (최소), 50000-100000 (권장)
- `save_freq=2500`

### 5.5 Testing & Deployment

```bash
# 오프라인 추론 테스트
python test_inference_official.py

# 실제 로봇 배포 (dataset mean 시작 위치)
python deploy_smolvla.py --start-pos dataset_mean --max-steps 300

# CSV 로깅
python deploy_smolvla.py --start-pos dataset_mean --max-steps 300 --log-csv
```

---

## 6. Lessons Learned (반복하면 안 되는 실수들)

### 절대 하면 안 되는 것

| 실수 | 결과 | 올바른 방법 |
|------|------|------------|
| 커스텀 학습 스크립트 작성 | "평균 액션"만 출력 | `lerobot-train` CLI 사용 |
| `SmolVLAConfig()` 새로 생성 | Action Expert 랜덤 | `from_pretrained("lerobot/smolvla_base")` |
| `n_action_steps=50` | 50스텝마다 1회 추론 | `n_action_steps=1` (closed-loop) |
| `[0,0,0,0,0,0]` 시작 | OOD → 소심한 동작 | dataset_mean 시작 |
| Loss만 보고 판단 | Loss↓ but 평균 액션 | L2 error + z-score + diversity 확인 |
| 카메라 위치 변경 | 데이터 전부 무효 | 카메라 고정 후 수집 |

### 항상 해야 하는 것

1. **사전학습 모델 사용**: `lerobot/smolvla_base` (Action Expert + VLM 둘 다 사전학습됨)
2. **정규화 확인**: state normalize + action unnormalize 필수
3. **Dataset mean 시작**: `--start-pos dataset_mean` 옵션 사용
4. **Closed-loop**: `n_action_steps=1` 로 매 스텝 새 추론
5. **품질 검증**: 수집 후 `data_episode_quality.py` 실행
6. **모터 문제 시**: `scan_servos.py` (T:106 리셋) 또는 `torque_set(cmd=1)` + `move_init()`

---

## 7. File Structure on Linux

```
~/roarm-smolvla/
├── .gitignore
├── CLAUDE.md                      # 프로젝트 설명서
├── requirements_core.txt          # pip 의존성
│
├── collect_data_manual.py         # [1] 데이터 수집
├── convert_to_lerobot_v3.py       # [2] 포맷 변환
├── run_official_train.py          # [3] 학습
├── test_inference_official.py     # [4] 오프라인 테스트
├── deploy_smolvla.py              # [5] 실제 배포
│
├── data_*.py                      # 분석 도구들
├── train_*.py                     # 학습 보조 도구들
├── analyze_action_dist.py         # 액션 분포 분석
│
├── scan_servos.py                 # 모터 리셋 유틸리티
├── roarm_demo.py                  # SDK 래퍼 참고
├── reset_robot.py                 # 로봇 리셋
│
├── lerobot/                       # (git clone, .gitignore)
├── .venv/                         # (python venv, .gitignore)
├── lerobot_dataset_v4/            # (새 데이터, .gitignore)
├── outputs/                       # (체크포인트, .gitignore)
│
├── claudedocs/
│   ├── PROJECT_AUDIT.md           # 전체 프로젝트 감사
│   ├── RESEARCH_IDEAS.md          # 연구 아이디어
│   └── VLA_PAPERS.md              # 관련 논문 목록
│
├── lerobot_backup/                # LeRobot RoArm M3 통합 코드 백업
│   ├── roarm_m3.py
│   └── configs.py
│
└── isaaclab_roarm/                # Isaac Lab 환경 (아카이브)
    ├── roarm_cfg.py
    └── roarm_*.py
```

---

## 8. Checkpoint Transfer (Optional)

20K 체크포인트를 가져가서 이어서 학습하고 싶으면:

```bash
# Windows에서 USB로 복사
# 복사할 폴더: outputs/smolvla_official/checkpoints/020000/
# 크기: ~2-3 GB

# Linux에서:
mkdir -p outputs/smolvla_official/checkpoints/
cp -r /media/usb/020000 outputs/smolvla_official/checkpoints/

# 이어서 학습 (resume)
python train_config_50k.py
```

**주의**: 새 카메라 위치로 데이터를 다시 수집하면 기존 체크포인트는 의미 없음.
새 데이터로 처음부터 학습하는 것이 더 나음.

---

## 9. Quick Start Checklist

```
□ Git clone 완료
□ Python 3.11 + venv 설치
□ PyTorch CUDA 설치 (torch.cuda.is_available() == True)
□ requirements_core.txt 설치
□ LeRobot git clone + pip install -e
□ roarm_sdk 설치
□ Azure Kinect SDK 설치 + 권한 설정
□ USB 시리얼 권한 (dialout 그룹)
□ Azure Kinect 테스트 통과
□ RoArm M3 테스트 통과
□ 카메라 고정 (삼각대/클램프)
□ 카메라 위치 사진 촬영 + 문서화
□ Windows 패치 코드 제거 (4개 파일)
□ COM 포트 → /dev/ttyUSB0 변경
□ collect_data_manual.py 테스트 실행
□ 본격 데이터 수집 시작 (100+ episodes)
```
