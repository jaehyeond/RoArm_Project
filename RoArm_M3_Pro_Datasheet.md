# RoArm-M3-Pro Technical Datasheet

> 작성일: 2025-12-02
> 출처: Waveshare 공식 문서, GitHub 리포지토리, 기술 커뮤니티 크로스 체크
> 목적: Isaac Sim 시뮬레이션 및 ROS2 실기 제어 구현을 위한 기술 데이터 정리

---

## 1. 하드웨어 스펙 (Physics & Simulation Data)

디지털 트윈 구축 시 PhysX 엔진에 입력해야 할 물리적 수치들입니다.

| 항목 | 값 | 비고 |
|------|-----|------|
| **모델명** | RoArm-M3-Pro | SKU: 24603 / 25304 등 옵션 존재 |
| **자유도 (DOF)** | 5 + 1 | 5축 암 + 1축 그리퍼 |
| **축 구성** | Base(Yaw) → Shoulder(Pitch) → Elbow(Pitch) → Wrist(Pitch/Roll 복합) → Gripper | |
| **중량** | 1,020.8g ±15g | 테이블 클램프 제외, RoArm-M3-S(973g)보다 무거움 |
| **작업 반경 (수평)** | 최대 직경 1120mm | 360° 전방향 |
| **작업 반경 (수직)** | 최대 높이 798mm | |
| **가반 하중** | 0.5kg | 최대 신장 시 0.2kg @ 0.5m 안정권 |

> **Tip**: 시뮬레이션에서는 보수적으로 0.2~0.3kg로 페이로드 세팅 권장

---

## 2. 액추에이터 (ST3235 Serial Bus Servo)

'Pro' 모델의 핵심. 시뮬레이션의 조인트 토크/속도 한계 설정에 사용.

| 항목 | 값 | 비고 |
|------|-----|------|
| **서보 모델** | ST3235 | ST3215 아님 (주의) |
| **통신 방식** | 시리얼 버스 (UART) | ID 기반 데이지 체인 연결 |
| **토크** | 30 kg.cm @ 12V | 매우 강력함 (일반 취미용 15~20kg.cm 대비) |
| **엔코더** | 12-bit 360° 마그네틱 | 기계적 마모 없음 |

### 피드백 데이터 (Realtime Readable)

ST3235 서보는 다음 데이터를 실시간으로 읽을 수 있음:

- **Position** (위치)
- **Speed** (속도)
- **Load** (하중/토크)
- **Voltage** (전압)
- **Current** (전류)
- **Temperature** (온도)

> **활용**: 힘 제어(Force Control) 흉내, 충돌 감지 로직, 과부하 보호, 물체 파지 확인 등에 활용 가능

---

## 3. 제어 인터페이스 & 프로토콜

### 메인 컨트롤러

| 항목 | 값 |
|------|-----|
| **MCU** | ESP32-WROOM-32 |
| **코어** | 듀얼 코어, 240MHz |
| **통신 포트** | USB-C (PC), UART 헤더, WiFi (ESP-NOW 지원) |

### JSON 명령 프로토콜

모든 명령은 JSON 문자열로 전송. **cmd 필드는 Boolean이 아닌 Integer 0/1 사용!**

#### Torque Control (T=210)

```json
{"T": 210, "cmd": 0}  // OFF - 토크 해제 (Free Move)
{"T": 210, "cmd": 1}  // ON  - 토크 활성화 (Lock)
```

#### Joint Angle Control (T=122)

```json
{
    "T": 122,
    "b": <base_angle>,
    "s": <shoulder_angle>,
    "e": <elbow_angle>,
    "t": <wrist_tilt_angle>,
    "r": <wrist_roll_angle>,
    "h": <gripper_angle>,
    "spd": <speed>,
    "acc": <acceleration>
}
```

#### Single Joint Radian Control (T=101)

```json
{"T": 101, "joint": 1, "rad": 3.14, "spd": 100, "acc": 10}
```

#### Inverse Kinematics / Position Control (T=1041)

```json
{"T": 1041, "x": 100, "y": 0, "z": 200, "t": 0, "r": 0, "g": 90}
```

#### Feedback Request (T=105)

```json
{"T": 105}
```

Response (M3 format):
```json
{
    "T": 1051,
    "x": <ee_x>, "y": <ee_y>, "z": <ee_z>,
    "tit": <tilt>,
    "b": <base_rad>, "s": <shoulder_rad>, "e": <elbow_rad>,
    "t": <wrist_rad>, "r": <roll_rad>, "g": <gripper_rad>,
    "tB": <torque_base>, "tS": <torque_shoulder>, "tE": <torque_elbow>,
    "tT": <torque_wrist>, "tR": <torque_roll>, "tG": <torque_gripper>
}
```

---

## 4. 명령 코드 레퍼런스 (JsonCmd)

| T 코드 | 명령 | 설명 |
|--------|------|------|
| 101 | JOINT_RADIAN_CTRL | 단일 조인트 라디안 제어 |
| 102 | JOINTS_RADIAN_CTRL | 전체 조인트 라디안 제어 |
| 105 | FEEDBACK_GET | 피드백 데이터 요청 |
| 112 | DYNAMIC_ADAPTATION_SET | 동적 적응 설정 |
| 114 | LED_CTRL | LED 밝기 제어 (0-255) |
| 121 | JOINT_ANGLE_CTRL | 단일 조인트 각도 제어 |
| 122 | JOINTS_ANGLE_CTRL | 전체 조인트 각도 제어 |
| 210 | TORQUE_SET | 토크 ON/OFF |
| 222 | GRIPPER_MODE_SET | 그리퍼 모드 설정 |
| 502 | MIDDLE_SET | 중립 위치 캘리브레이션 |
| 605 | ECHO_SET | 에코 모드 설정 |
| 1041 | POSE_CTRL | IK 기반 위치 제어 |

---

## 5. 속도/가속도 변환 공식

SDK 내부에서 사용하는 변환 공식:

```python
# 속도 변환 (SDK → 실제)
실제_속도(deg/s) = spd * 180 / 2048

# 가속도 변환 (SDK → 실제)
실제_가속도(deg/s²) = acc * 180 / (254 * 100)

# 역변환 (실제 → SDK)
spd = 실제_속도 * 2048 / 180
acc = 실제_가속도 * 254 * 100 / 180
```

### 속도 예시

| spd 값 | 실제 속도 (deg/s) | 설명 |
|--------|------------------|------|
| 50 | 4.4 | 매우 느림 |
| 500 | 43.9 | 보통 |
| 900 | 79.1 | 빠름 (권장) |
| 2048 | 180 | 최대 |

---

## 6. 조인트 매핑 (6 DOF)

| Joint # | SDK Key | 이름 | 타입 | 범위 (추정) |
|---------|---------|------|------|-------------|
| 1 | `b` | Base | Yaw (회전) | -180° ~ +180° |
| 2 | `s` | Shoulder | Pitch | -90° ~ +90° |
| 3 | `e` | Elbow | Pitch | -135° ~ +135° |
| 4 | `t` | Wrist Tilt | Pitch | -90° ~ +90° |
| 5 | `r` | Wrist Roll | Roll | -180° ~ +180° |
| 6 | `h` | Gripper/Hand | Grip | 0° ~ 90° |

> **주의**: Gripper(h)는 SDK 내부에서 `180 - angle` 변환 적용됨

---

## 7. 소프트웨어 리소스

### 공식 GitHub 리포지토리

| 리소스 | URL | 설명 |
|--------|-----|------|
| **Python SDK** | `waveshareteam/waveshare_roarm_sdk` | 시리얼/HTTP 통신 래퍼 |
| **ROS2 패키지** | `waveshareteam/roarm_ws_em0` | URDF, 런치 파일 포함 |

### URDF 위치

```
roarm_ws_em0/
└── src/
    └── roarm_description/
        ├── urdf/
        │   ├── roarm.urdf
        │   └── roarm.xacro
        └── meshes/
            └── *.stl
```

> **Isaac Sim 연동**: URDF를 Import 후 Inertia 값 실측 또는 근사치 수정 필요

### AI/LeRobot 지원

HuggingFace의 LeRobot 프레임워크 지원 추가됨:
- Leader-Follower 데이터셋 수집 모드 예제 존재
- 모방 학습(Imitation Learning) 파이프라인 구축 가능

---

## 8. M3 vs M3-Pro 비교

| 구분 | RoArm-M3-S (Standard) | RoArm-M3-Pro (Professional) |
|------|----------------------|----------------------------|
| **서보** | 일반 시리얼 서보 | ST3235 (30kg.cm, 메탈 기어) |
| **무게** | 973g | 1020g (더 묵직함) |
| **내구성** | 보통 | 높음 (산업용/연구용 적합) |
| **용도** | 교육용, 가벼운 작업 | 알고리즘 검증, 장시간 구동, 정밀 제어 |

---

## 9. Sim-to-Real 권장 파라미터

Isaac Sim 디지털 트윈 구축 시 권장 설정:

```python
# Physics Parameters
joint_stiffness = 1000.0      # N.m/rad
joint_damping = 100.0         # N.m.s/rad
max_force = 500.0             # N.m (토크 한계)

# Payload (Conservative)
max_payload = 0.25            # kg (안전 마진 포함)

# Speed Limits
max_joint_velocity = 3.14     # rad/s (~180 deg/s)
default_speed = 1.38          # rad/s (~79 deg/s, spd=900)
```

---

## 10. 참고 자료

- [Waveshare RoArm-M3 Wiki](https://www.waveshare.com/wiki/RoArm-M3)
- [ST3235 Servo Datasheet](https://www.waveshare.com/wiki/ST3235_Servo)
- [YouTube: How to Control Waveshare RoArm Easily](https://www.youtube.com/watch?v=...) - 초기 세팅 튜토리얼

---

## 분석가 의견

RoArm-M3-Pro는 연구용으로 매우 적합한 선택입니다. 특히:

1. **ST3235 서보의 피드백 기능(전류/부하)** 활용 가능
2. Isaac Sim에서 학습한 RL 정책을 실기 적용 시:
   - 과부하 보호 로직
   - 물체 파지 확인 로직
   - 충돌 감지 로직
3. **30 kg.cm 토크**로 실용적인 작업 가능

이 데이터들을 바탕으로 **Sim-to-Real의 물리 파라미터(마찰, 댐핑, 토크 한계)**를 정밀하게 세팅하면 됩니다.
