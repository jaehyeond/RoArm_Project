# Reference — RoArm M3 Hardware · Motor Recovery · Camera Setup

> 출처: 분리 전 `AGENTS.md` **397-524행**. 원본 전체는 `docs/archive/AGENTS_full_20260825_pre_split.md`.
> 아래 본문은 원본에서 **바이트 동일**하게 이동했다 (2026-08-25).
> 하드웨어 직접 제어는 `AGENTS.md ## Safety constraints`에 따라 **사용자 명시 승인 후에만** 실행한다.
> 관절 범위(JOINT_LIMITS)·포트 매핑(Leader=USB0 / Follower=USB1)을 건드리기 전에 이 파일을 먼저 읽는다.

---

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
