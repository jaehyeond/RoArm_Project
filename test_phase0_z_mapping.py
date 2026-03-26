#!/usr/bin/env python3
"""Phase 0: Z축 매핑 — FK z좌표의 의미 파악 + 안전 하한 결정

안전한 범위에서만 움직임:
- shoulder: 위로만 (0~60, 아래로 안 감)
- elbow: 적당한 범위 (30~90)
- base: 고정 0
- 속도 느리게 (300)

목적:
1. FK z=0이 어디인지 (base? 모터 축?)
2. z 양수 = 위? 아래?
3. 책상 높이를 추정할 수 있는 관절 각도 한계
"""
import time
import math
import logging

logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

def _silent(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data.extend([data['x'], data['y'], data['z']])
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res
roarm._process_received = _silent

arm = roarm(roarm_type='roarm_m3', port='/dev/ttyUSB0', baudrate=115200)
time.sleep(0.5)
arm.torque_set(cmd=1)
time.sleep(0.5)

# 1단계: 초기 위치에서 FK 확인
arm.move_init()
time.sleep(2)

def read_fk():
    for _ in range(5):
        try:
            pose = arm.pose_get()
            if pose and len(pose) >= 3:
                return pose
        except Exception:
            time.sleep(0.1)
    return None

def read_joints():
    for _ in range(5):
        angles = arm.joints_angle_get()
        if angles and len(angles) == 6 and angles[0] != 180:
            return angles
        time.sleep(0.1)
    return None

print("=== Z-axis Mapping (safe movements only) ===\n")

# 초기 위치
pose = read_fk()
joints = read_joints()
print(f"[INIT] Joints: [{', '.join(f'{a:.1f}' for a in joints)}]")
print(f"[INIT] FK: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}")
print()

# 2단계: Shoulder만 변경 (base=0, elbow=0 고정)
# shoulder 올리면(양수) 팔이 위로 가는지 아래로 가는지 확인
print("--- Shoulder sweep (base=0, elbow=0, wrist=0) ---")
print("팔이 위를 향할수록 z가 어떻게 변하는지 확인")
for sh in [0, 15, 30, 45, 60, 75, 90]:
    arm.joints_angle_ctrl(angles=[0, sh, 0, 0, 0, 50], speed=300, acc=100)
    time.sleep(2)
    pose = read_fk()
    joints = read_joints()
    if pose:
        fk_dist = math.sqrt(pose[0]**2 + pose[1]**2)
        print(f"  shoulder={sh:3d}° → z={pose[2]:+8.1f}mm  dist={fk_dist:.0f}mm  x={pose[0]:.0f}")

print()

# 3단계: 초기 위치 복귀 후, Elbow만 변경
arm.joints_angle_ctrl(angles=[0, 30, 0, 0, 0, 50], speed=300, acc=100)
time.sleep(2)

print("--- Elbow sweep (base=0, shoulder=30, wrist=0) ---")
print("팔을 더 펴면 z가 어떻게 변하는지 확인")
for el in [0, 20, 40, 60, 80, 100]:
    arm.joints_angle_ctrl(angles=[0, 30, el, 0, 0, 50], speed=300, acc=100)
    time.sleep(2)
    pose = read_fk()
    if pose:
        fk_dist = math.sqrt(pose[0]**2 + pose[1]**2)
        print(f"  elbow={el:3d}° → z={pose[2]:+8.1f}mm  dist={fk_dist:.0f}mm  x={pose[0]:.0f}")

print()

# 4단계: 가장 높은 위치 vs 가장 낮은 위치 (안전 범위 내)
print("--- Extreme positions (safe) ---")
test_cases = [
    ("HIGHEST", [0, 0, 0, 0, 0, 50]),       # 팔 완전 위
    ("UP_MID",  [0, 30, 30, 0, 0, 50]),     # 중간 위
    ("LEVEL",   [0, 45, 90, 0, 0, 50]),      # 수평 근처
    ("LOW_MID", [0, 60, 100, 0, 0, 50]),     # 중간 아래
    ("LOWEST_SAFE", [0, 70, 110, 0, 0, 50]), # 낮지만 안전할 듯
]

for name, angles_cmd in test_cases:
    arm.joints_angle_ctrl(angles=angles_cmd, speed=300, acc=100)
    time.sleep(2.5)
    pose = read_fk()
    joints = read_joints()
    if pose and joints:
        fk_dist = math.sqrt(pose[0]**2 + pose[1]**2)
        print(f"  {name:14s} sh={joints[1]:+6.1f} el={joints[2]:+6.1f} → "
              f"z={pose[2]:+8.1f}mm dist={fk_dist:.0f}mm")

# 복귀
arm.move_init()
time.sleep(1)
arm.disconnect()

print("\n=== ANALYSIS ===")
print("z 양수 = 위 (base 위) / z 음수 = 아래 (base 아래) 인지 확인")
print("책상 높이 = 로봇 base에서 책상 표면까지의 z offset")
print("→ 이 값을 JOINT_LIMITS나 FK z 하한으로 설정")
