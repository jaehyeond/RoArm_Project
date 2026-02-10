# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoArm-M3-Pro 로봇 강화학습 파이프라인 프로젝트. Isaac Sim + Isaac Lab + RL 조합으로 시뮬레이션 학습 후 ROS2를 통해 실제 로봇으로 전이(Sim2Real).

```
[Isaac Sim] → [Isaac Lab] → [RL Training] → [ROS2 Bridge] → [Real Robot]
     │             │              │               │              │
  USD Model    RL Env        PPO/SAC         Joint Cmd      RoArm-M3-Pro
```

## Environment

| Component | Details |
|-----------|---------|
| Isaac Sim | 5.1.0 @ `C:\isaac-sim` |
| GPU | RTX 4070 Ti SUPER |
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| ROS2 | Humble @ `/opt/ros/humble` (WSL) |
| Robot | RoArm-M3-Pro via COM8 (USB-UART) |
| Camera | **Azure Kinect DK** (VLA 데이터 수집용, pyk4a) |
| USB Hub | Waveshare USB3.2-Gen1-HUB-2IN-4OUT |

## Key Commands

### Isaac Sim Scripts
```powershell
# Run simulation scripts (MUST use Isaac Sim's Python)
cd C:\isaac-sim
.\python.bat "E:\RoArm_Project\step3_move.py"

# Analysis script (check USD structure)
.\python.bat "E:\RoArm_Project\step1_analyze.py"

# Fix base test (standalone base fixing)
.\python.bat "E:\RoArm_Project\step2_fix_base.py"
```

### ROS2 (WSL)
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# USB passthrough to WSL (requires usbipd on Windows)
# Windows (Admin): usbipd bind --busid=<BUSID> && usbipd attach --wsl --busid=<BUSID>
# WSL: ls /dev/ttyUSB*
```

### Real Robot (Windows .venv)
```powershell
# Activate project venv
E:\RoArm_Project\.venv\Scripts\activate

# Run robot control (requires roarm_sdk)
python roarm_demo.py
```

## Architecture

### Pipeline Stages
1. **step1_analyze.py** - USD 구조 분석, Joint/RigidBody 목록 출력
2. **step2_fix_base.py** - Base 고정 테스트 (kinematic=True)
3. **step3_move.py** - 전체 데모: 로드 → 고정 → Drive 설정 → 동작 시퀀스
4. **roarm_demo.py** - 실제 로봇 제어 래퍼 클래스

### USD Robot Structure
```
/World/RoArm/roarm_description/
├── base_link [RigidBody, ArticulationRoot] → kinematic=True (fixed)
├── link1 [RigidBody]
├── link2 [RigidBody]
├── link3 [RigidBody]
├── gripper_link [RigidBody]
└── joints/
    ├── base_link_to_link1 [PhysicsRevoluteJoint]
    ├── link1_to_link2 [PhysicsRevoluteJoint]
    ├── link2_to_link3 [PhysicsRevoluteJoint]
    └── link3_to_gripper_link [PhysicsRevoluteJoint]
```

### Isaac Sim Script Template
```python
# Required boilerplate - SimulationApp MUST be first import
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, UsdLux, PhysicsSchemaTools

# Load stage and USD
stage = omni.usd.get_context().get_stage()
roarm_prim = stage.DefinePrim("/World/RoArm", "Xform")
roarm_prim.GetReferences().AddReference(r"E:\RoArm_Project\RoARM PRO_M3.usd")

# Physics scene (required)
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# Ground plane
PhysicsSchemaTools.addGroundPlane(stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0,0,0), Gf.Vec3f(0.5))

# Start simulation
omni.timeline.get_timeline_interface().play()
while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
```

### Joint Control Pattern (Isaac Sim 5.1.0)
```python
from pxr import UsdPhysics

# Fix base to ground (CRITICAL - robot falls without this)
rigid_body = UsdPhysics.RigidBodyAPI(base_link_prim)
rigid_body.CreateKinematicEnabledAttr(True)

# Configure joint drive
drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
drive.CreateTypeAttr("force")
drive.CreateStiffnessAttr(1000.0)   # 강성
drive.CreateDampingAttr(100.0)      # 감쇠
drive.CreateMaxForceAttr(500.0)     # 최대 토크
drive.CreateTargetPositionAttr(target_degrees)  # degrees, NOT radians
```

### Real Robot Control (roarm_sdk)
```python
from roarm_sdk import RoArm
arm = RoArm(port="COM8")  # or "auto"
arm.joints_angle_ctrl(angles=[0,0,0,0,0,0], acc=300, speed=900)  # 6 joints
arm.torque_set(1)  # 1=on, 0=off (NOT bool)
```

## Task Tracking

**항상 작업 전후로 `TASKS.md` 확인 및 업데이트할 것.**

현재 진행 상황:
- ✅ Phase 1: Isaac Sim 시뮬레이션 (완료)
- ✅ Phase 3: ROS2 연동 (완료)
- ✅ Phase 4: Isaac Lab RL 환경 (Grasping 학습 완료)
- ✅ Phase 4.5: 카메라 연동 (Azure Kinect DK 연결 완료)
- ✅ Phase 5: Sim2Real 테스트 (정책 극단값 문제 분석 완료)
- ✅ Phase 6: VLA/Diffusion Policy 방향 전환 (LeRobot + SmolVLA)
- ✅ Phase 6.1: LeRobot RoArm M3 통합 (roarm_m3.py 구현 완료)
- ✅ Phase 6.2: 키보드 텔레오퍼레이션 (QWERTY 6-DOF 제어)
- ✅ Phase 6.3: Leader-Follower 모드 (듀얼 팔 미러링)
- ✅ Phase 6.4: Strategy Pattern 리팩토링 (if/else → 4클래스 분리)
- ✅ **Phase 7: 데이터 수집 및 VLA 학습**
- ✅ Phase 7.1: 토크 OFF 수동 모드로 51 에피소드 수집 (`collect_data_manual.py`)
- ❌ Phase 7.2a: 커스텀 학습 실패 (batch_size=1, batch_size=8+vlm 모두 "평균 액션")
- ✅ Phase 7.2b: 공식 lerobot-train CLI + smolvla_base 20K steps 학습 완료 (loss 0.009)
- ✅ Phase 7.3: 오프라인 추론 테스트 PASS (Mean L2 Error: 4.39, 다양한 액션 출력)
- ✅ Phase 7.4: 실제 로봇 배포 성공 (5스텝 + 50스텝, ~10ms/step)

## Critical Notes

- **Isaac Sim 5.1.0 API**: Uses `from isaacsim import SimulationApp`, not old `omni.isaac.kit`
- **SimulationApp first**: 반드시 다른 import보다 먼저 SimulationApp 생성
- **Base must be fixed**: Set `kinematic=True` on base_link or robot falls over
- **Joint angles in degrees**: DriveAPI TargetPosition uses degrees, not radians
- **Sim joint count**: 4 joints (simulation) vs 6 joints (real robot)
- **USB to WSL**: Requires usbipd-win for COM port passthrough to ROS2
- **Robot standing fix**: BBox 계산 후 회전 필요 (Y축이 높이인 경우 -90° X축 회전)

## ⚠️ RoArm M3 모터 응답 없음 문제 해결 (중요!)

### 증상
- 전원 켜도 팔이 초기 위치로 안 움직임 (정상: 자동으로 중간 위치로 이동)
- `joints_angle_get()` 반환값이 `[180, -180, -90, -180, 180, 180]` (SDK 에러 기본값)
- 모터 온도 피드백 전부 0 (`tB:0, tS:0, tE:0, tT:0, tR:0`)
- ESP32 보드는 살아있음 (LCD 켜짐, 시리얼 통신 됨)

### 원인
모터 버스 통신이 초기화 안 됨 (ESP32 ↔ 서보 모터 간 통신 문제)

### 해결 방법 1: 서보 스캔으로 ESP32 리셋 (T:106)

```powershell
# 스크립트 실행 (권장)
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\scan_servos.py COM3
```

```python
import serial
import time

# 문제 있는 로봇의 COM 포트로 연결
ser = serial.Serial('COM3', 115200, timeout=2)
time.sleep(1)

# T:106 서보 스캔 명령 전송 - 이게 핵심!
ser.write(b'{"T":106}\n')
time.sleep(1)
response = ser.read_all()
print(response)  # ESP32가 리셋되면서 모터 버스 재초기화

ser.close()
```

### 해결 방법 2: 토크 ON + 초기 위치 이동 (팔이 고개 숙일 때)

팔이 힘이 빠지면서 고개를 숙이는 경우, 토크가 꺼진 상태일 수 있음:

```python
from roarm_sdk.roarm import roarm
import time

arm = roarm(roarm_type='roarm_m3', port='COM3', baudrate=115200)
time.sleep(0.5)

# 토크 ON
arm.torque_set(cmd=1)
time.sleep(0.5)

# 초기 위치로 이동
arm.move_init()
time.sleep(2)

arm.disconnect()
```

### 왜 동작하는가
- `T:106` 명령은 서보 스캔을 시도함
- 모터 버스에 문제가 있으면 ESP32가 크래시하고 자동 리셋됨
- 리셋 과정에서 모터 버스가 재초기화되면서 정상화됨
- `torque_set(cmd=1)` + `move_init()`은 토크가 꺼진 상태에서 복구

### 확인 방법
1. 명령 전송 후 로봇 전원 OFF → ON
2. 팔이 초기 위치로 자동 이동하면 성공
3. 팔이 뻣뻣해지면 (토크 ON) 정상

### 관련 파일
- `E:\RoArm_Project\scan_servos.py` - 서보 스캔 스크립트
- `E:\RoArm_Project\reset_robot.py` - 리셋 명령 스크립트

## Camera Setup

### ⚠️ 카메라 혼동 주의 (중요!)

| 카메라 | 용도 | 라이브러리 | 현재 사용 |
|--------|------|-----------|----------|
| **Azure Kinect DK** | VLA 데이터 수집 | `pyk4a` | ✅ **사용 중** |
| IMX335 USB | Sim2Real 테스트용 (미사용) | `cv2.VideoCapture` | ❌ 미사용 |

**VLA 데이터 수집에는 반드시 Azure Kinect를 사용!**
- `collect_data_manual.py` → Azure Kinect (pyk4a)
- `collect_data.py` → Azure Kinect (pyk4a)
- 1 에피소드 데이터 → Azure Kinect로 수집됨

### Azure Kinect 사양
| 항목 | 값 |
|------|-----|
| 모델 | Azure Kinect DK |
| RGB 해상도 | 1280×720 (720P) |
| Depth 모드 | NFOV_UNBINNED |
| 라이브러리 | `pyk4a` (Python Azure Kinect API) |
| 연결 | USB 3.0 |

### Azure Kinect 테스트
```powershell
# Azure Kinect 연결 확인
E:\RoArm_Project\.venv\Scripts\python.exe -c "from pyk4a import PyK4A; k4a = PyK4A(); k4a.start(); print('OK'); k4a.stop()"

# 데이터 수집 미리보기
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\collect_data_manual.py
```

### 데이터 수집 코드 패턴
```python
# Azure Kinect 사용 (올바름)
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
depth = capture.transformed_depth

# IMX335 USB 사용 (VLA 데이터 수집에 사용하지 말 것)
# cap = cv2.VideoCapture(0)  # ← 이건 IMX335용, VLA에 쓰면 안됨
```

### USB 허브 구성 (Waveshare USB3.2-Gen1-HUB-2IN-4OUT)
```
PC ──USB케이블──→ [IN1]
                    │
        ┌───────────┴───────────┐
        ↓           ↓           ↓
     [USB1]      [USB2]      [USB3]
        │           │
  Azure Kinect    로봇
     (DK)        (COM8)
```

## Sim2Real 현황

### 정책 분석 결과 (2026-01-11)
| 체크포인트 | 액션 출력 | 상태 |
|-----------|----------|------|
| model_0.pt (초기) | [+0.1, -0.03, +0.01, +0.03] | 합리적 |
| model_100.pt (초기 학습) | [+1.9, +1.7, +2.6, -1.4] | 적절 |
| model_500.pt+ (후기) | [+36, +37, +36, -36] | 포화됨 |

### Sim2Real 스크립트
```powershell
# 로봇 동작 테스트
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\test_robot_motion.py

# 초기 체크포인트 정책 (권장)
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\sim2real_grasp_v3.py
```

## Next Steps

### 현재 파이프라인 (State-based)
```
[고정 좌표] → [학습된 정책] → [로봇 이동]
```

### 목표 파이프라인 (Vision-based)
```
[카메라] → [물체 감지] → [좌표 변환] → [정책] → [로봇 이동] → [그리퍼 제어]
```

### 남은 작업 (VLA 방향)
1. ~~**LeRobot 설치**: RoArm M3 통합~~ ✅ 완료 (roarm_m3.py + configs.py)
2. ~~**키보드 텔레옵 구현**~~ ✅ 완료 (KeyboardTeleopStrategy)
3. ~~**Leader-Follower 구현**~~ ✅ 완료 (LeaderFollowerTeleopStrategy) - 미사용
4. ~~**코드 리팩토링**~~ ✅ 완료 (Strategy Pattern, 4클래스 분리)
5. ✅ **데이터 수집**: 토크 OFF 수동 모드로 51 에피소드 수집 완료 (`collect_data_manual.py`)
6. ✅ **SmolVLA Fine-tuning**: 공식 CLI + smolvla_base 20K steps 완료 (loss 0.009)
7. ✅ **오프라인 추론 테스트**: PASS (Mean L2 Error: 4.39)
8. ✅ **실제 로봇 배포**: 50스텝 실행 성공 (`deploy_smolvla.py`, ~10ms/step)

## LeRobot Integration (roarm_m3.py)

### 개요

LeRobot 프레임워크에 RoArm M3 Pro를 통합.

### ⚠️ 현재 데이터 수집 방식 (중요!)

**팔 1개 + 토크 OFF 수동 모드** 사용 중:
- 스크립트: `collect_data_manual.py` (Azure Kinect + pyk4a)
- 방식: 토크 OFF → 손으로 로봇 직접 움직여서 데이터 수집
- Leader-Follower 모드: 구현됨 but **미사용** (팔로워 팔 없음)

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `lerobot/lerobot/common/robot_devices/robots/roarm_m3.py` | 메인 로봇 제어 (Strategy Pattern) |
| `lerobot/lerobot/common/robot_devices/robots/configs.py` | `RoarmRobotConfig` 데이터클래스 |
| `test_leader_follower.py` | L-F 독립 테스트 (SDK 직접 사용) |
| `test_lerobot_roarm.py` | LeRobot 통합 테스트 |

### Strategy Pattern 아키텍처

```
TeleopStrategy (ABC)                   ← 추상 인터페이스
├── initialize(robot)                  ← connect() 끝에 호출
├── generate_goal_positions(robot)     ← teleop_step()에서 호출
├── cleanup(robot)                     ← disconnect() 시작에 호출
└── is_active → bool                   ← 세션 계속 여부

KeyboardTeleopStrategy                 ← leader_arms={} 일 때
├── pynput.keyboard.Listener           ← 키 입력 감지
├── Q/A, W/S, E/D, R/F, T/G, Y/H      ← 6관절 +/- 제어
├── -/= (속도 조절), P/ESC (종료)
├── step_size: [2, 5, 10] deg/step
├── motor_speed=500, motor_acc=200     ← 텔레옵 전용 속도
└── joint limits 클램핑

LeaderFollowerTeleopStrategy           ← leader_arms 설정시
├── initialize: torque_set(cmd=0)      ← 리더 자유 이동
├── generate: leader 읽기 → follower 미러링
│   └── speed=0, acc=0                 ← 즉시 최대속도 추적
└── cleanup: torque_set(cmd=1) → disconnect leaders

RoarmRobot (깔끔한 메인 클래스)
├── __init__: _create_teleop_strategy() 팩토리
├── connect(): 공통 초기화 → strategy.initialize()
├── teleop_step(): strategy.generate_goal_positions() → 데이터 기록
├── disconnect(): strategy.cleanup() → 공통 정리
├── capture_observation(): 팔로워 읽기 + 카메라 캡처
└── send_action(): 정책 추론용 (motor_speed=500, acc=200)
```

### 모드 선택 로직

```python
# RoarmRobotConfig에서:
leader_arms: dict[str, str] = {}           # 비어있으면 키보드 모드
follower_arms: dict[str, str] = {"main": "COM8"}

# _create_teleop_strategy()에서:
if len(self.leader_arms) == 0:
    return KeyboardTeleopStrategy()        # 키보드 모드
else:
    return LeaderFollowerTeleopStrategy()  # L-F 모드
```

### Leader-Follower 구현 Step-by-Step

#### Step 1: 하드웨어 준비

```
USB 허브 구성 (현재 - 팔 1개):
PC ──USB──→ [HUB IN1]
              │
    ┌─────────┴──────────┐
    ↓         ↓
  [USB1]    [USB2]
    │         │
Azure Kinect  로봇
   (DK)      (COM8)
```

- 로봇 1개만 사용 (COM8)
- 토크 OFF 모드에서 손으로 직접 움직여서 데이터 수집

#### Step 2: SDK 연결 및 초기화

```python
from roarm_sdk.roarm import roarm

# 팔로워 (제어 대상)
follower = roarm(roarm_type="roarm_m3", port="COM8", baudrate=115200)

# 리더 (사람이 움직이는 팔)
leader = roarm(roarm_type="roarm_m3", port="COM9", baudrate=115200)
```

**SDK 버그 패치 (필수):**
- `sdk_common.DataProcessor._process_received` 몽키패치
- SDK line 320의 `print(data)` 호출 제거
- `BaseController` 로거 CRITICAL 레벨로 설정 (백그라운드 스레드 디코드 에러 억제)

#### Step 3: 리더 토크 비활성화

```python
# 리더 팔의 토크를 끄면 손으로 자유롭게 움직일 수 있음
leader.torque_set(cmd=0)   # cmd: 0=off, 1=on (NOT bool)
time.sleep(0.3)
```

**주의:** `torque_set(cmd=0)` 이지 `torque_set(0)` 아님. SDK는 keyword arg `cmd` 필요.

#### Step 4: 팔로워 홈 포지션 이동

```python
# 팔로워를 안전한 시작 위치로
follower.joints_angle_ctrl(
    angles=[0, 0, 0, 0, 0, 0],  # 6-DOF 홈 포지션
    speed=500,                    # 중간 속도
    acc=200                       # 중간 가속도
)
time.sleep(2.5)  # 이동 완료 대기
```

#### Step 5: 실시간 미러링 루프

```python
while True:
    # 1. 리더 위치 읽기 (재시도 로직 포함)
    leader_angles = safe_joints_angle_get(leader, max_retries=5)
    # → [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]

    # 2. 팔로워에 즉시 전송
    follower.joints_angle_ctrl(
        angles=leader_angles,
        speed=0,    # 0 = 최대 속도 즉시 추적 (Waveshare 기본값)
        acc=0       # 0 = 최대 가속도
    )

    time.sleep(0.02)  # ~50Hz 제어 루프
```

**핵심:** `speed=0, acc=0`은 Waveshare SDK에서 "최대 속도로 즉시 이동"을 의미.

#### Step 6: safe_joints_angle_get() 재시도 로직

```python
def safe_joints_angle_get(arm, max_retries=5, delay=0.1):
    """SDK 백그라운드 스레드 디코드 에러 대응."""
    for attempt in range(max_retries):
        try:
            angles = arm.joints_angle_get()
            if angles is not None and len(angles) >= 6:
                return list(angles)
        except (KeyError, TypeError, AttributeError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts")
    raise RuntimeError("Got None or invalid data")
```

**왜 필요한가:**
- SDK의 백그라운드 스레드가 시리얼 데이터를 `readline()`으로 읽음
- 타임아웃 시 `None` 반환 → `json.loads(None)` → KeyError/TypeError
- 최대 5회 재시도로 간헐적 에러 극복

#### Step 7: 안전한 종료

```python
# 1. 리더 토크 복원 (안전)
leader.torque_set(cmd=1)
time.sleep(0.2)

# 2. 팔로워 홈 복귀
follower.joints_angle_ctrl(angles=[0,0,0,0,0,0], speed=500, acc=200)
time.sleep(2)

# 3. 연결 해제
leader.disconnect()
follower.disconnect()
```

#### Step 8: LeRobot에서 Leader-Follower 실행

```powershell
# 키보드 모드 (기본값: leader_arms 비어있음)
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --control.type=teleoperate

# Leader-Follower 모드 (leader_arms 지정)
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --robot.leader_arms='{"main": "COM9"}' \
  --control.type=teleoperate

# L-F 모드 + 데이터 수집 (50 에피소드)
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --robot.leader_arms='{"main": "COM9"}' \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the object and place it in the box." \
  --control.repo_id=${HF_USER}/roarm_m3_grasping \
  --control.num_episodes=50
```

### 독립 테스트 (test_leader_follower.py)

LeRobot 없이 SDK만으로 L-F 동작 검증:

```powershell
# COM 포트 확인 후 실행
python test_leader_follower.py
```

테스트 단계:
1. 양팔 USB 연결 확인
2. 리더 토크 OFF → 손으로 자유 이동 확인
3. 실시간 미러링 (Ctrl+C로 종료)
4. 리더 토크 복원 및 정리

### RoArm M3 관절 사양

| 관절 | 이름 | 범위 (도) | 비고 |
|------|------|-----------|------|
| 0 | Base rotation | -190 ~ 190 | 좌우 회전 |
| 1 | Shoulder | -110 ~ 110 | 어깨 |
| 2 | Elbow | -70 ~ 190 | 비대칭! |
| 3 | Wrist pitch | -110 ~ 110 | 손목 상하 |
| 4 | Wrist roll | -190 ~ 190 | 손목 회전 |
| 5 | Gripper | -10 ~ 100 | 그리퍼 개폐 |

### SDK 주요 API

```python
from roarm_sdk.roarm import roarm

arm = roarm(roarm_type="roarm_m3", port="COM8", baudrate=115200)

# 관절 각도 읽기 → list[6] (degrees)
angles = arm.joints_angle_get()

# 관절 각도 제어
arm.joints_angle_ctrl(
    angles=[0, 0, 0, 0, 0, 0],  # 6개 관절 (degrees)
    speed=500,                    # 0=최대속도, 1-1000
    acc=200                       # 0=최대가속, 1-500
)

# 토크 설정
arm.torque_set(cmd=1)  # 1=on, 0=off (keyword arg 필수)

# 연결 해제
arm.disconnect()
```

### 리팩토링 히스토리

| 버전 | 구조 | 줄 수 | 특징 |
|------|------|-------|------|
| v1 | 키보드 전용 | ~400 | 초기 구현 |
| v2 | 듀얼 모드 (if/else) | 569 | L-F 추가, 스파게티 코드 |
| v3 | Strategy Pattern | 645 | 4개 클래스 분리, 깔끔한 위임 |

## Reference Documents

### Physical AI 참고 문서
- **[2026_Physical_AI.md](2026_Physical_AI.md)**: 2024-2025 VLA/Physical AI 기술 총정리
  - RT-1/RT-2 → Octo → OpenVLA → π₀ → CogACT 발전 흐름
  - Flow Matching vs Diffusion 비교
  - BitVLA, PD-VLA, RTC 온디바이스 최적화
  - Google Gemini Robotics vs NVIDIA GR00T 비교

### LeRobot 참고 자료
- RoArm M3 LeRobot PR: https://github.com/huggingface/lerobot/pull/820
- SmolVLA 문서: https://huggingface.co/docs/lerobot/en/smolvla
- LeRobot GitHub: https://github.com/huggingface/lerobot

## ⚠️ SmolVLA 학습 설정 교훈 (중요!)

### 커스텀 학습 스크립트 실패 기록 (2026-02-05 ~ 02-06)

51 에피소드(13,010 프레임)로 3차례 학습했지만 모두 **"평균 액션"만 출력**하는 문제 발생.

#### 실패한 시도들
| 시도 | 설정 | 결과 |
|------|------|------|
| 1 | batch_size=1, vlm=False | 평균 액션 (그래디언트 노이즈) |
| 2 | batch_size=8, vlm=False | 평균 액션 (하이퍼파라미터 문제) |
| 3 | batch_size=8, vlm=True | 평균 액션 (Action Expert 랜덤) |

#### 근본 원인 3가지 (2026-02-06 분석)
| 원인 | 설명 | 심각도 |
|------|------|--------|
| **Action Expert 랜덤 초기화** | `SmolVLAConfig()`로 새 모델 생성 → Expert 랜덤. 공식은 `lerobot/smolvla_base`(487개 데이터셋, 1000만 프레임 사전학습) 사용 | 치명적 |
| **정규화 누락** | 공식: `preprocessor(batch)` MEAN_STD 정규화. 우리: `v.to(device)` 만 | 심각 |
| **LR 스케줄러 없음** | 공식: cosine decay + warmup. 우리: 고정 lr | 보통 |

#### 해결책: 공식 lerobot-train CLI 사용
```bash
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_pick \
  --dataset.root=E:/RoArm_Project/lerobot_dataset_v3 \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/smolvla_official
```

#### 핵심 교훈
1. **커스텀 학습 스크립트 작성 금지**: 공식 파이프라인에 빠진 요소가 너무 많음
2. **사전학습 모델(smolvla_base) 필수**: VLM만으로는 부족, Action Expert도 사전학습 필요
3. **Loss 감소 ≠ 좋은 모델**: Loss 0.57까지 떨어져도 "평균 액션" 출력 가능
4. **정규화 필수**: state/action 정규화 없이는 학습 효율 극히 낮음

### 관련 파일
| 파일 | 설명 |
|------|------|
| `train_smolvla.py` | 커스텀 학습 스크립트 (실패, 참고용) |
| `outputs/smolvla_roarm_51ep_vlm/` | load_vlm_weights=True 체크포인트 (실패) |
| `outputs/smolvla_roarm_51ep_bs8/` | batch_size=8 체크포인트 (실패) |
| `SMOLVLA_TRAINING_RESULTS.md` | 상세 학습 결과 기록 |
