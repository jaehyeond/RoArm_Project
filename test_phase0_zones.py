#!/usr/bin/env python3
"""Phase 0-5: classify_zone() live test — 5 positions"""
import time
import math
import logging

logging.getLogger('BaseController').setLevel(logging.CRITICAL)

from roarm_sdk.roarm import roarm
from roarm_sdk.common import handle_m3_feedback, JsonCmd

# SDK print spam 억제
_orig = roarm._process_received
def _silent(self, data, genre):
    if not data:
        return None
    res, valid_data = [], []
    if genre == JsonCmd.FEEDBACK_GET:
        valid_data.append(data['x'])
        valid_data.append(data['y'])
        valid_data.append(data['z'])
        if self.type == "roarm_m3":
            valid_data = handle_m3_feedback(valid_data, data)
    else:
        valid_data = data
    res.append(valid_data)
    return res
roarm._process_received = _silent

# classify_zone 함수 복사
def classify_zone(base_angle, fk_dist, fk_z):
    if fk_z is not None and fk_z > 150:
        return "OVERHEAD"
    if fk_dist is None:
        return "UNKNOWN"
    if fk_dist < 140 and abs(base_angle) <= 30:
        return "NEAR"
    if 120 <= fk_dist <= 220:
        if base_angle < -30:
            return "MID_LEFT"
        elif base_angle > 30:
            return "MID_RIGHT"
    if fk_dist > 200 and abs(base_angle) <= 30:
        return "FAR_CENTER"
    if base_angle < -15:
        return "MID_LEFT"
    elif base_angle > 15:
        return "MID_RIGHT"
    elif fk_dist < 160:
        return "NEAR"
    else:
        return "FAR_CENTER"

# 테스트 위치 (5-zone 대표)
# [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
TEST_POSITIONS = {
    "NEAR":       [0, 40, 60, 10, 0, 50],      # 가까운 중앙
    "MID_LEFT":   [-60, 40, 80, 10, 0, 50],     # 왼쪽 중간
    "MID_RIGHT":  [60, 40, 80, 10, 0, 50],      # 오른쪽 중간
    "FAR_CENTER": [0, 60, 100, -10, 0, 50],     # 먼 중앙
    "OVERHEAD":   [0, 20, 40, 40, 0, 50],       # 높은 위치
}

# LEGACY pre-L-F (2026-04-01): USB0=Leader (gripper clamp). Edit to /dev/ttyUSB1 for Follower.
arm = roarm(roarm_type='roarm_m3', port='/dev/ttyUSB0', baudrate=115200)
time.sleep(0.5)

# 토크 ON + 초기 위치
arm.torque_set(cmd=1)
time.sleep(0.5)
arm.move_init()
time.sleep(2)

print("=== Zone Classification Test ===\n")

results = []
for expected_zone, angles in TEST_POSITIONS.items():
    print(f"Moving to {expected_zone} position: {angles}")
    arm.joints_angle_ctrl(angles=angles, speed=500, acc=200)
    time.sleep(2.5)

    # FK 읽기
    pose = None
    for _ in range(5):
        try:
            pose = arm.pose_get()
            if pose is not None and len(pose) >= 3:
                break
        except Exception:
            time.sleep(0.1)

    if pose is not None:
        base_angle = angles[0]
        fk_x, fk_y, fk_z = pose[0], pose[1], pose[2]
        fk_dist = math.sqrt(fk_x**2 + fk_y**2)
        detected = classify_zone(base_angle, fk_dist, fk_z)
        match = "OK" if detected == expected_zone else "MISMATCH"
        results.append((expected_zone, detected, match))
        print(f"  FK: x={fk_x:.1f} y={fk_y:.1f} z={fk_z:.1f} dist={fk_dist:.1f}mm")
        print(f"  Base angle: {base_angle}°")
        print(f"  Detected: {detected} | Expected: {expected_zone} | {match}")
    else:
        results.append((expected_zone, "NO_FK", "FAIL"))
        print(f"  FK read failed!")
    print()

# 초기 위치 복귀
arm.move_init()
time.sleep(1)
arm.disconnect()

# 결과 요약
print("=== RESULTS ===")
ok = sum(1 for _, _, m in results if m == "OK")
for exp, det, m in results:
    print(f"  {exp:12s} → {det:12s} [{m}]")
print(f"\n{ok}/{len(results)} zones correctly classified")
