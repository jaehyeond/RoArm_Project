# Ubuntu 22.04 네이티브 환경 설치 가이드

RoArm-M3-Pro 프로젝트를 Ubuntu 22.04 네이티브 환경에서 구축하는 방법

**최종 업데이트: 2026-01-19**

---

## ⚠️ 중요 경고: 구버전 가이드 주의!

인터넷의 많은 가이드가 **outdated** 되어 있습니다:

| 항목 | ❌ 구버전 (무시해야 함) | ✅ 현재 올바른 방법 |
|------|----------------------|-------------------|
| **Omniverse Launcher** | "Launcher 설치하세요" | **2025.10.01 deprecated됨!** |
| **Isaac Sim 버전** | "4.2.0 권장" | **5.1.0 사용** (4.2 지원 중단) |
| **설치 방법** | Launcher에서 다운로드 | **pip install** 또는 직접 다운로드 |

> **Omniverse Launcher, Nucleus Workstation, Nucleus Cache는 2025년 10월 1일부로 deprecated되었습니다.**
> Isaac Sim 4.5.0이 Launcher에서 설치 가능한 마지막 버전이었습니다.
>
> — [NVIDIA 공식 문서](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)

---

## 현재 환경 vs 새 환경 비교

| 항목 | Windows + WSL (현재) | Ubuntu 22.04 네이티브 (새로운) |
|------|---------------------|-------------------------------|
| Isaac Sim | Windows에서 실행 | Linux에서 직접 실행 |
| ROS2 | WSL 내부 | 네이티브 설치 |
| USB 장치 | usbipd로 패스스루 필요 | 직접 접근 가능 (/dev/ttyUSB*) |
| 성능 | WSL 오버헤드 있음 | 네이티브 성능 |
| GPU | 직접 사용 | 직접 사용 |
| 복잡도 | 높음 (2개 OS) | 낮음 (1개 OS) |

**결론: Ubuntu 네이티브가 더 간단하고 성능도 좋음**

---

## 시스템 요구사항

### 하드웨어
| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | NVIDIA RTX 30xx+ | RTX 40xx |
| VRAM | 8GB | 12GB+ |
| RAM | 16GB | 32GB |
| 디스크 | 50GB | 100GB (SSD) |
| CPU | 8코어 | 12코어+ |

### 소프트웨어 버전
| 컴포넌트 | 버전 |
|----------|------|
| Ubuntu | 22.04 LTS |
| NVIDIA 드라이버 | 535.216+ (권장: 580+) |
| CUDA | 12.x |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | main (latest) |
| ROS2 | Humble |

---

## 설치 순서 (총 5단계)

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: NVIDIA 드라이버 + CUDA 설치                             │
│     ↓                                                           │
│  Step 2: Python 3.11 + 가상환경 설정                             │
│     ↓                                                           │
│  Step 3: Isaac Sim 5.1.0 설치                                    │
│     ↓                                                           │
│  Step 4: Isaac Lab 설치                                          │
│     ↓                                                           │
│  Step 5: ROS2 Humble 설치 + 연동                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: NVIDIA 드라이버 + CUDA 설치

### 1.1 기존 드라이버 제거 (필요시)
```bash
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo reboot
```

### 1.2 필수 패키지 설치
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget unzip curl
```

### 1.3 NVIDIA 드라이버 설치 (방법 1: apt - 권장)
```bash
# 사용 가능한 드라이버 확인
ubuntu-drivers devices

# 권장 드라이버 자동 설치
sudo ubuntu-drivers autoinstall

# 또는 특정 버전 설치 (580 권장)
sudo apt install nvidia-driver-580

sudo reboot
```

### 1.4 드라이버 설치 확인
```bash
nvidia-smi
```

예상 출력:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.xx.xx    Driver Version: 580.xx.xx    CUDA Version: 12.x    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
+-------------------------------+----------------------+----------------------+
```

### 1.5 CUDA Toolkit 설치
```bash
# NVIDIA CUDA 저장소 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA 12 설치
sudo apt install cuda-toolkit-12-4

# 환경변수 설정 (~/.bashrc에 추가)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.6 CUDA 설치 확인
```bash
nvcc --version
```

---

## Step 2: Python 3.11 환경 설정

**두 가지 방법 중 선택:**

### 방법 A: Conda 사용 (공식 문서 권장)

```bash
# 2A.1 Miniconda 설치
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# 터미널 재시작 (또는 source ~/.bashrc)
# (base)가 프롬프트에 보이면 성공

# 2A.2 Conda 환경 생성
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# 2A.3 pip 업그레이드
pip install --upgrade pip
```

### 방법 B: venv 사용 (더 간단, 가볍게)

```bash
# 2B.1 Python 3.11 설치 (Ubuntu 22.04는 기본 3.10)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# 2B.2 작업 디렉토리 생성
mkdir -p ~/workspace
cd ~/workspace

# 2B.3 venv 가상환경 생성
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate

# 2B.4 pip 업그레이드
pip install --upgrade pip
```

### 어떤 방법을 선택해야 하나?

| 상황 | 권장 |
|------|------|
| 처음 설치, 간단하게 | venv (방법 B) |
| 여러 프로젝트 관리, Python 버전 전환 필요 | Conda (방법 A) |
| 공식 문서 따라하기 | Conda (방법 A) |

---

## Step 3: Isaac Sim 5.1.0 설치

### 방법 A: pip 설치 (권장 - 간단함)

```bash
# 가상환경 활성화 상태에서
source ~/workspace/env_isaaclab/bin/activate

# Isaac Sim 설치 (약 8-10GB, 30분+ 소요)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 방법 B: 바이너리 설치 (전통적 방법)

```bash
# 1. 다운로드 (NVIDIA 계정 필요)
# https://developer.nvidia.com/isaac-sim 에서 다운로드

# 2. 압축 해제
mkdir ~/workspace/isaacsim
cd ~/Downloads
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -d ~/workspace/isaacsim

# 3. 설치 스크립트 실행
cd ~/workspace/isaacsim
./post_install.sh
```

### 3.1 Isaac Sim 설치 확인
```bash
# pip 설치의 경우
isaacsim

# 바이너리 설치의 경우
cd ~/workspace/isaacsim
./isaac-sim.selector.sh
```

**첫 실행 시 10분+ 소요됨 (확장 프로그램 다운로드)**

---

## Step 4: Isaac Lab 설치

### 4.1 저장소 클론
```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 4.2 시스템 의존성 설치
```bash
sudo apt install cmake build-essential
```

### 4.3 Isaac Lab 설치
```bash
# 가상환경 활성화 확인
source ~/workspace/env_isaaclab/bin/activate

# Isaac Lab 설치 (rsl_rl 포함)
./isaaclab.sh --install
```

### 4.4 PyTorch 설치/업데이트
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4.5 설치 검증
```bash
# 빈 시뮬레이션 테스트 (검은 화면 뜨면 성공)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# RL 학습 테스트
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

---

## Step 5: ROS2 Humble 설치

> ⚠️ **Isaac Sim + ROS2 주의사항**
> - Isaac Sim은 Python 3.11 사용, ROS2 Humble은 Python 3.10 사용
> - **같은 터미널에서 동시에 source 하면 안 됨!**
> - Isaac Sim은 내장 ROS2 라이브러리 사용 (Cyclone DDS, Python 3.11 빌드)
> - 외부 ROS 노드는 별도 터미널에서 Python 3.10으로 실행
>
> — [NVIDIA Isaac Sim ROS2 문서](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)

### 5.1 시스템 패키지 업그레이드 (중요!)
```bash
# Ubuntu 22.04 초기 업데이트 문제 방지
sudo apt update
sudo apt upgrade -y
```

### 5.2 Locale 설정 (UTF-8 필수)
```bash
locale  # 현재 locale 확인
sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 5.3 ROS2 저장소 추가
```bash
# Universe 저장소 활성화
sudo apt install software-properties-common
sudo add-apt-repository universe

# ROS2 GPG 키 추가
sudo apt update && sudo apt install curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# ROS2 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 5.4 ROS2 Humble 설치
```bash
sudo apt update

# Desktop 설치 (RViz, demos, tutorials 포함) - 권장
sudo apt install ros-humble-desktop -y

# 또는 Base 설치 (GUI 없음, 최소 설치)
# sudo apt install ros-humble-ros-base -y
```

### 5.5 Isaac Sim용 추가 메시지 타입 설치
```bash
sudo apt install ros-humble-vision-msgs ros-humble-ackermann-msgs -y
```

### 5.6 ROS2 개발 도구 설치
```bash
sudo apt install python3-colcon-common-extensions python3-rosdep -y
sudo rosdep init
rosdep update
```

### 5.7 ROS2 환경 설정

**⚠️ 중요: bashrc에 추가하지 마세요!**

Isaac Sim과 ROS2를 함께 사용할 때, 자동 source는 충돌을 일으킬 수 있습니다.
대신 필요할 때만 수동으로 source하는 것을 권장합니다.

```bash
# 필요할 때만 실행 (bashrc에 추가하지 말 것)
source /opt/ros/humble/setup.bash
```

만약 ROS2만 사용하는 터미널이 필요하다면:
```bash
# 별도의 alias로 관리 (선택사항)
echo 'alias ros2env="source /opt/ros/humble/setup.bash"' >> ~/.bashrc
source ~/.bashrc

# 사용할 때
ros2env
```

### 5.8 ROS2 설치 확인
```bash
# ROS2 source 후 테스트
source /opt/ros/humble/setup.bash

# 버전 확인
ros2 --version

# talker-listener 테스트 (터미널 2개 필요)
# 터미널 1:
ros2 run demo_nodes_cpp talker
# 터미널 2:
ros2 run demo_nodes_py listener
```

### 5.9 Isaac Sim ROS2 Bridge 설정

Isaac Sim에서 ROS2 Bridge를 사용할 때:

**방법 1: 내장 라이브러리 사용 (권장)**
```bash
# conda 환경 활성화 (Isaac Sim용)
conda activate env_isaaclab

# ROS2를 source하지 않은 상태에서 Isaac Sim 실행
# Isaac Sim이 자동으로 내장 ROS2 Humble 라이브러리 로드
isaacsim
```

**방법 2: 시스템 ROS2와 연결**
```bash
# 외부 ROS 노드는 별도 터미널에서 실행
# 터미널 1 (Isaac Sim):
conda activate env_isaaclab
isaacsim

# 터미널 2 (ROS2 노드):
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic echo /clock
```

### DDS 설정 (선택사항)

Isaac Sim 5.1.0은 Cyclone DDS를 내장 사용합니다.
멀티 머신 통신이 필요한 경우:

```bash
# FastDDS 사용 시
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Cyclone DDS 사용 시
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

---

## USB 장치 연결 (로봇 + 카메라)

### 네이티브 Linux의 장점: USB 직접 접근

```bash
# USB 장치 확인
ls /dev/ttyUSB*
ls /dev/video*

# 권한 설정 (재부팅 후에도 유지)
sudo usermod -aG dialout $USER  # 시리얼 포트
sudo usermod -aG video $USER    # 카메라

# 로그아웃 후 다시 로그인 필요
```

### 로봇 연결 테스트
```bash
# Python 환경에서
source ~/workspace/env_isaaclab/bin/activate
pip install roarm-sdk

python3 -c "
from roarm_sdk import RoArm
arm = RoArm(port='/dev/ttyUSB0')  # 자동 검색: port='auto'
print(arm.get_joints_angle())
"
```

### 카메라 연결 테스트
```bash
pip install opencv-python

python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
cap.release()
"
```

---

## 프로젝트 파일 이동

### Windows에서 Linux로 파일 복사

**방법 1: USB 드라이브**
```bash
# Windows에서 USB에 복사 후 Linux에서
cp -r /media/$USER/USB_DRIVE/RoArm_Project ~/workspace/
```

**방법 2: 네트워크 공유 (같은 네트워크)**
```bash
# Windows에서 폴더 공유 설정 후
sudo apt install cifs-utils
sudo mount -t cifs //WINDOWS_IP/RoArm_Project /mnt/windows -o user=USERNAME
cp -r /mnt/windows ~/workspace/RoArm_Project
```

**방법 3: Git (권장)**
```bash
# Windows에서 GitHub에 push 후
cd ~/workspace
git clone https://github.com/YOUR_USERNAME/RoArm_Project.git
```

---

## 환경 설정 스크립트 (편의용)

`~/workspace/setup_env.sh` 파일 생성:

```bash
#!/bin/bash
# RoArm-M3-Pro 개발 환경 활성화 스크립트

echo "=== RoArm-M3-Pro Development Environment ==="

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Python 가상환경
source ~/workspace/env_isaaclab/bin/activate

# ROS2
source /opt/ros/humble/setup.bash

# 작업 디렉토리
cd ~/workspace/RoArm_Project

echo "Environment ready!"
echo "  - Python: $(python --version)"
echo "  - CUDA: $(nvcc --version | grep release)"
echo "  - ROS2: $ROS_DISTRO"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

사용법:
```bash
chmod +x ~/workspace/setup_env.sh
source ~/workspace/setup_env.sh
```

---

## Windows vs Linux 명령어 차이

| 작업 | Windows (현재) | Linux (새로운) |
|------|---------------|----------------|
| Isaac Sim 실행 | `C:\isaac-sim\python.bat script.py` | `isaacsim` 또는 `./isaac-sim.sh` |
| Isaac Lab 스크립트 | N/A | `./isaaclab.sh -p script.py` |
| 가상환경 활성화 | `.venv\Scripts\activate` | `source env/bin/activate` |
| 로봇 포트 | `COM8` | `/dev/ttyUSB0` |
| 경로 구분자 | `\` | `/` |

---

## 트러블슈팅

### 문제 1: nvidia-smi 안 됨
```bash
# 드라이버 재설치
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 문제 2: Isaac Sim 실행 안 됨
```bash
# 호환성 체크
# https://developer.nvidia.com/isaac-sim 에서 Compatibility Checker 다운로드
./omni.isaac.sim.compatibility_check.sh
```

### 문제 3: Permission denied on /dev/ttyUSB0
```bash
sudo usermod -aG dialout $USER
# 로그아웃 후 다시 로그인
```

### 문제 4: Isaac Lab import 에러
```bash
# 가상환경 확인
which python  # ~/workspace/env_isaaclab/bin/python 이어야 함

# 재설치
cd ~/workspace/IsaacLab
./isaaclab.sh --install
```

---

## 설치 진행 상황 (2026-01-19)

| 단계 | 항목 | 상태 | 비고 |
|------|------|------|------|
| Step 1 | NVIDIA 드라이버 580.95.05 | ✅ 완료 | RTX 4090 Laptop |
| Step 1 | CUDA 13.0 | ✅ 완료 | nvidia-smi 확인 |
| Step 2 | Miniconda | ✅ 완료 | ~/miniconda3 |
| Step 2 | Python 3.11 (env_isaaclab) | ✅ 완료 | conda 환경 |
| Step 3 | Isaac Sim 5.1.0.0 | ✅ 완료 | pip 설치 |
| Step 4 | Isaac Lab 2.3.0 | ✅ 완료 | pip 설치 |
| Step 5 | ROS2 Humble | ✅ 완료 | Desktop 설치 완료 |
| 추가 | USB 장치 테스트 | ⏳ 대기 | 로봇+카메라 연결 후 |
| 추가 | 프로젝트 파일 이동 | ⏳ 대기 | Windows → Linux |

### 확인된 정보
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **드라이버**: 580.95.05
- **CUDA**: 13.0
- **Python**: 3.11.14 (conda env_isaaclab)
- **Isaac Sim**: 5.1.0.0 (GUI 정상 실행 확인)
- **Isaac Lab**: 2.3.0
- **ROS2**: Humble (ros2 topic list 정상)

### 알려진 경고 (무시 가능)
- PYTHONPATH 경고: pip 설치 시 정상
- ROS2 Bridge 실패: ROS2 미설치 상태에서 정상
- 폴더 잠금 아이콘: omniverse:// 클라우드 에셋 (읽기 전용)

---

## 체크리스트

설치 완료 후 확인사항:

- [x] `nvidia-smi` 출력 정상 ✅
- [x] `nvcc --version` 출력 정상 ✅
- [x] `python --version` = 3.11.x ✅ (3.11.14)
- [x] `isaacsim` 실행 → GUI 창 열림 ✅
- [ ] `./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py` → 검은 화면
- [x] `ros2 topic list` 출력 정상 ✅
- [ ] `/dev/ttyUSB0` 접근 가능 (로봇 연결 시)
- [ ] `/dev/video0` 접근 가능 (카메라 연결 시)

---

## 참고 링크

- [Isaac Sim 공식 문서](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/)
- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/main/)
- [Isaac Lab pip 설치 가이드](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [ROS2 Humble 설치](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [NVIDIA 드라이버 Ubuntu](https://ubuntuhandbook.org/index.php/2025/09/ubuntu-added-nvidia-580-driver/)
- [자동 설치 스크립트 (비공식)](https://github.com/robosmiths/isaac-sim-lab-installer)
