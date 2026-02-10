# RoArm-M3-Pro 모델 분석: Isaac Sim vs 실제 로봇

> 작성일: 2025-12-02
> 목적: Isaac Sim 시뮬레이션과 실제 로봇 간 움직임 차이 원인 분석

---

## 개요

Isaac Sim에서 로봇이 빠르게 움직이는 반면, 실제 로봇은 매우 느리게(소극적으로) 움직이는 문제가 발생. 이 문서는 양쪽 시스템의 구조를 분석하고 차이점을 정리함.

---

## Step 1: Isaac Sim USD 모델 구조

### 시뮬레이션 Joint 개수: 4개

| # | Joint Name | 연결 | 용도 |
|---|------------|------|------|
| 1 | `base_link_to_link1` | base → link1 | Base Rotation |
| 2 | `link1_to_link2` | link1 → link2 | Shoulder |
| 3 | `link2_to_link3` | link2 → link3 | Elbow |
| 4 | `link3_to_gripper_link` | link3 → gripper | Gripper |

### USD 경로 구조

```
/World/RoArm/roarm_description/
├── base_link [RigidBody, ArticulationRoot] → kinematic=True (고정)
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

### Drive 설정 (step4_ros_bridge.py)

```python
drive.CreateStiffnessAttr(1000.0)  # 강성
drive.CreateDampingAttr(100.0)     # 감쇠
drive.CreateMaxForceAttr(500.0)    # 최대 토크
# 각도: degrees 단위
```

---

## Step 2: 실제 로봇 SDK JSON 포맷

### 실제 로봇 Joint 개수: 6개

| # | SDK Key | 이름 | 용도 | Sim 대응 |
|---|---------|------|------|----------|
| 1 | `b` | base | Base Rotation | ✅ Joint 1 |
| 2 | `s` | shoulder | Shoulder | ✅ Joint 2 |
| 3 | `e` | elbow | Elbow | ✅ Joint 3 |
| 4 | `t` | wrist tilt | 손목 Pitch | ❌ 없음 |
| 5 | `r` | wrist roll | 손목 Roll | ❌ 없음 |
| 6 | `h` | hand/gripper | Gripper | ✅ Joint 4 |

### SDK JSON 명령 포맷 (T=122: JOINTS_ANGLE_CTRL)

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

### SDK 속도/가속도 변환 공식

```python
# roarm_sdk/common.py에서 정의
실제_속도(deg/s) = spd * 180 / 2048
실제_가속도(deg/s²) = acc * 180 / (254 * 100)

# 예시:
spd=900  → 79.1 deg/s
spd=50   → 4.4 deg/s

acc=300  → 2.1 deg/s²
acc=10   → 0.07 deg/s²
```

### Gripper 특수 처리

```python
# SDK에서 gripper는 180 - angle 변환 필요
command_data[5] = 180 - command_data[5]
```

---

## Step 3: 현재 데이터 흐름 분석

### 3.1 Isaac Sim → TCP Bridge (step4_ros_bridge.py)

```python
# Sim 4 joints → Real 6 joints 매핑
real_angles = [
    sim_angles[0],  # J1: base
    sim_angles[1],  # J2: shoulder
    sim_angles[2],  # J3: elbow
    0.0,            # J4: wrist tilt (고정 0)
    0.0,            # J5: wrist roll (고정 0)
    sim_angles[3],  # J6: gripper
]

# TCP로 전송하는 JSON
data = {
    "type": "joint_cmd",
    "angles": real_angles,  # 6개 각도
    "timestamp": time.time()
}
```

**문제점**: 속도/가속도 정보 없이 각도만 전송

### 3.2 ROS2 Bridge Node (ros2_bridge_node.py)

```python
# TCP에서 받은 데이터를 /joint_cmd 토픽으로 발행
scaled_angles = [a * self.scale for a in angles]
msg = Float64MultiArray()
msg.data = scaled_angles
self.joint_pub.publish(msg)
```

**역할**: 단순 중계 (angles만 전달)

### 3.3 RoArm Driver Node (roarm_node.py)

```python
cmd = {
    "T": 122,  # JOINTS_ANGLE_CTRL
    "b": angles[0],
    "s": angles[1],
    "e": angles[2],
    "t": angles[3],
    "r": angles[4],
    "h": 180 - angles[5],  # gripper 변환
    "spd": 50,   # ⚠️ 하드코딩된 낮은 속도!
    "acc": 10    # ⚠️ 하드코딩된 낮은 가속도!
}
```

**🔴 핵심 문제 발견!**

---

## Step 4: 핵심 차이점 및 문제점

### 비교표

| 항목 | Isaac Sim | 실제 로봇 | 상태 |
|------|-----------|----------|------|
| **Joint 개수** | 4개 | 6개 | ⚠️ 매핑 필요 |
| **Wrist Tilt (J4)** | 없음 | 있음 | ⚠️ 0으로 고정 |
| **Wrist Roll (J5)** | 없음 | 있음 | ⚠️ 0으로 고정 |
| **속도 제어** | 물리 시뮬레이션 | spd 파라미터 | 🔴 전달 안됨 |
| **가속도 제어** | 물리 시뮬레이션 | acc 파라미터 | 🔴 전달 안됨 |
| **속도 값** | 즉각 반응 | `spd: 50` | 🔴 **18배 느림** |
| **가속도 값** | 물리 기반 | `acc: 10` | 🔴 **30배 느림** |
| **Gripper** | 직접 각도 | `180 - angle` | ✅ 처리됨 |

### 속도 비교

| 설정 위치 | spd 값 | acc 값 | 실제 속도 |
|-----------|--------|--------|-----------|
| roarm_demo.py | 900 | 300 | 79.1 deg/s |
| roarm_node.py | 50 | 10 | 4.4 deg/s |
| **차이** | **18배** | **30배** | **매우 느림** |

---

## 🔴 "소극적으로 움직이는" 원인

### 원인 1: 하드코딩된 낮은 속도/가속도

`roarm_node.py`에서 속도와 가속도가 매우 낮은 값으로 고정되어 있음:

```python
"spd": 50,   # SDK 기준 max 4096의 1.2%만 사용
"acc": 10    # SDK 기준 max 254의 4%만 사용
```

### 원인 2: 속도 정보 미전달

Isaac Sim에서 각도만 전송하고 속도 정보는 전송하지 않음:
- TCP 데이터: `{"type": "joint_cmd", "angles": [...], "timestamp": ...}`
- 속도/가속도 필드 없음

### 원인 3: Joint 개수 불일치

시뮬레이션 4 joints vs 실제 6 joints로 인해:
- Wrist tilt (J4)와 Wrist roll (J5)가 항상 0으로 고정
- 실제 로봇의 일부 자유도를 사용하지 못함

---

## 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│  Isaac Sim (4 joints)                                       │
│  - base, shoulder, elbow, gripper                           │
│  - 물리 시뮬레이션으로 빠르게 움직임                          │
│  - Stiffness: 1000, Damping: 100                            │
│  - 속도 정보: 물리 엔진이 자동 계산                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ TCP:5555
                      │ {"angles": [4개→6개 매핑], "timestamp": ...}
                      │ ⚠️ 속도 정보 없음!
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  ros2_bridge_node (WSL)                                     │
│  - TCP 수신 → /joint_cmd 토픽 발행                          │
│  - 4 joints → 6 joints 매핑 (J4, J5 = 0)                    │
│  - scale_factor 적용 가능                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ /joint_cmd (Float64MultiArray)
                      │ [6개 각도값]
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  roarm_node (WSL)                                           │
│  - /joint_cmd 구독 → Serial JSON 명령 생성                  │
│  - 🔴 spd: 50  (하드코딩, 너무 느림!)                        │
│  - 🔴 acc: 10  (하드코딩, 너무 느림!)                        │
│  - JSON: {"T":122, "b":, "s":, "e":, "t":, "r":, "h":}      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Serial (/dev/ttyUSB0, 115200)
                      │ JSON 명령
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  실제 RoArm-M3-Pro (6 joints)                               │
│  - 🐢 느리게 움직임 (spd=50, acc=10 때문)                    │
│  - J4, J5 항상 0 (시뮬레이션에 없는 joint)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 해결 방안

### 방안 1: 빠른 해결 (속도 값 수정)

`roarm_node.py`에서 속도/가속도 값을 높임:

```python
# 변경 전
"spd": 50,
"acc": 10

# 변경 후
"spd": 900,   # roarm_demo.py와 동일
"acc": 300    # roarm_demo.py와 동일
```

### 방안 2: 동적 속도 제어

Isaac Sim에서 속도 정보도 함께 전송:

```python
# step4_ros_bridge.py 수정
data = {
    "type": "joint_cmd",
    "angles": real_angles,
    "speed": calculated_speed,      # 추가
    "acceleration": calculated_acc,  # 추가
    "timestamp": time.time()
}
```

### 방안 3: ROS2 파라미터화

`roarm_node.py`에서 속도/가속도를 ROS2 파라미터로 설정:

```python
self.declare_parameter("speed", 900)
self.declare_parameter("acceleration", 300)

spd = self.get_parameter("speed").value
acc = self.get_parameter("acceleration").value
```

실행 시:
```bash
ros2 run roarm_driver roarm_node --ros-args -p speed:=900 -p acceleration:=300
```

---

## 참고 파일

| 파일 | 위치 | 역할 |
|------|------|------|
| step4_ros_bridge.py | E:\RoArm_Project\ | Isaac Sim TCP 브릿지 |
| ros2_bridge_node.py | ~/ros2_ws/src/roarm_driver/ | TCP→ROS2 변환 |
| roarm_node.py | ~/ros2_ws/src/roarm_driver/ | ROS2→Serial 명령 |
| roarm_sdk/common.py | .venv/Lib/site-packages/ | SDK JSON 포맷 정의 |
| roarm_sdk/generate.py | .venv/Lib/site-packages/ | SDK 명령 생성기 |
