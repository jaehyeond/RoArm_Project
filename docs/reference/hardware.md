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
| 3 | Wrist pitch | -110 ~ 110 (표기) — 🔴 **실측 2026-09-07: 펌웨어가 +90 에서 클램프**(SDK 는 110 까지 통과시키나 명령 103.5° → 읽기 89.8°). 계획은 ±90 안에서 | 손목 상하 |
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

### ~~해결 방법 1: T:106 ESP32 리셋~~ — 🔴 **틀렸다. 사용 금지 (2026-09-03, D479)**

`T:106` 은 리셋이 아니라 **그리퍼 서보 구동 명령**(`CMD_EOAT_HAND_CTRL`, 펌웨어 `json_cmd.h:65`, `uart_ctrl.h:66-71`)이다.
`cmd` 없이 `{"T":106}` 만 보내면 cmd=0 → `handJointCtrlRad` 가 [700, 2596] 스텝으로 클램프(`RoArm-M3_module.h:346-347`) → 서보가
**700 스텝 = 1.074 rad = 조 118.5° 개방** 위치로 최대 토크로 움직인다. 순정 조 단독이면 기계 스토퍼에 걸릴 뿐이지만,
**그랩 v1 이 장착돼 있으면 4절 링크 범위(0~89°) 밖으로 밀어 링크·PLA 를 부순다.** ESP32 재부팅은 `T:600`(D473 ①).
아래 "해결 방법 2" 만 쓴다. (`scan_servos.py` 의 T:106 주석도 같은 오류 — D473 ①)

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

## 그랩 v1 장착 시 그리퍼 서보 규약 (2026-09-03, D479 — 펌웨어 `EffectsMachine/roarm-m3` 원문 근거)

커스텀 그랩은 순정 그리퍼를 떼지 않고 얹는다: 브래킷 = 순정 고정 조, 크랭크판 = 순정 **가동 조**(D462·D476). 가동 조는 집게가 아니라
**서보 레버**이고, 그 회전(0~89°)이 4절 링크로 셸을 0~44.5°(입 0~58 mm) 연다. 즉 순정 서보 명령이 곧 그랩 명령이다.

| 항목 | 값 | 근거 |
|---|---|---|
| 부팅 동작 | `setup()` → `RoArmM3_moveInit()` → 그리퍼 서보를 **중앙 2047 스텝 = π rad** 로 이동(속도 600·가속 20), 이어 전 관절 토크 상한 = ST_TORQUE_MAX(1000) | `roarm-m3.ino:113,117`, `RoArm-M3_module.h:238-239` |
| 각도 규약 | 펌웨어 EOAT rad = 스텝·2π/4096(오프셋 0). **π = 닫힘**, 1.57 = 열림, 4.0 = 과폐합(`json_cmd.h:60-64` 예시 "grab = 3.14") | `RoArm-M3_module.h:59-61` |
| SDK 규약 | `joints_angle_ctrl` 의 그리퍼 각도 h(deg): 펌웨어 = 180 − h / rad: π − h → **h = 조 개방각(0 닫힘, 90 열림)** = `grab_v1_meta.json` 의 `servo_deg` 와 동일 | `roarm_sdk/common.py:147,180` |
| 클램프 | 서보 위치 [700, 2596] = EOAT [1.074, 3.982] rad = 조 개방 **+118.5° ~ −48°(과폐합)**. 링크 범위(0~89°) 밖도 명령 가능 → 상위에서 막아야 함 | `RoArm-M3_module.h:346-347` |
| 토크 제한 | `{"T":107,"tor":N}` (50~1000) 이 그리퍼 서보 토크 상한을 쓴다. 🔴 **2026-09-09 정정(80th)**: 대상은 EPROM 이 아니라 **SRAM 48 번(토크 리밋, 휘발)** 이고, 소스 0.84 기준 부팅 완료 시 그리퍼 상한은 1000 이 아니라 **300**(`ino:146 handTorqueCtrl(300)`). "부팅 시 1000" 은 `ino:117` 만 읽은 추론이었다(실기 미확인). 매 세션 재전송 규칙은 그대로. 상세 `servo_pid_st3215.md` §4 | `json_cmd.h:67`, `RoArm-M3_module.h:398-402`, `ino:118,146` |
| 그랩 상태 | 설계 서보 0° = 셸 닫힘 = 펌웨어 π = **부팅 위치**. 부팅은 "닫힘" 명령이라 그랩 장착 상태에서 무해(단 팔 전체가 HOME 으로 이동하니 주변을 비울 것) | D476·D478 |

**운용 규칙 (그랩 장착 시)**
1. 그리퍼 명령은 **SDK 각도 0~89°** 또는 **T:106 cmd = π − θ(rad), θ ∈ [0, 1.553]** 만. 맨 `{"T":106}`·cmd<1.59·cmd>π+0.1(과폐합) 금지 — 링크가 범위 밖으로 밀린다.
2. 폐합은 **T:107 토크 상한을 먼저 낮춘 뒤**(초기 200 권장, D452 전류-제한 stall 정책) 명령한다. 상한은 휘발(SRAM 48)이라 매 세션 재설정(부팅 후 값은 소스상 300, 실기 미확인 — 2026-09-09 정정). ⚠️ 서보 과부하 보호(출력 80 % 초과 2 s → 20 % 강하)가 상한 900 스톨에서 걸릴 수 있다 — `servo_pid_st3215.md` §4.
3. **개구 > 44 mm 이면 손목 롤 |r| 제한**(완전개방 |r| ≤ 14°, D473 ⑥·`grab_v1_meta.json wrist_roll_constraint`). 열기 전에 롤을 0 근처로.
4. 부팅·`move_init()` 전에 그랩 주변(순정 조가 89° 로 벌어질 공간 + 팔 HOME 경로)을 비운다.
5. 서보→셸 매핑은 비선형(`servo_shell_mouth_nonlinear` 표). 개구 mm 가 필요하면 표로 환산, 각도 비례로 추정 금지(D463).
