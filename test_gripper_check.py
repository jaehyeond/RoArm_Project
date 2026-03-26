#!/usr/bin/env python3
"""그리퍼 기능 체크: 현재 위치에서 그리퍼만 열고 닫기"""
import time, logging
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

arm = roarm(roarm_type="roarm_m3", port="/dev/ttyUSB0", baudrate=115200)
time.sleep(1)

# 현재 위치 읽기
angles = None
for _ in range(5):
    angles = arm.joints_angle_get()
    if angles and len(angles) == 6 and angles[0] != 180:
        break
    time.sleep(0.1)

if not angles:
    print("ERROR: Cannot read joints")
    arm.disconnect()
    exit()

print("Current joints: [%s]" % ", ".join("%.1f" % a for a in angles))
print("Current gripper: %.1f deg" % angles[5])
print()

# 그리퍼만 움직이기 (다른 관절 유지)
arm.torque_set(cmd=1)
time.sleep(0.5)

# 열기
print("Opening gripper to 80 deg...")
test_angles = list(angles)
test_angles[5] = 80
arm.joints_angle_ctrl(angles=test_angles, speed=200, acc=100)
time.sleep(2)

angles_after = arm.joints_angle_get()
if angles_after:
    print("  Gripper after open: %.1f deg" % angles_after[5])

# 닫기
print("Closing gripper to 5 deg...")
test_angles[5] = 5
arm.joints_angle_ctrl(angles=test_angles, speed=200, acc=100)
time.sleep(2)

angles_after = arm.joints_angle_get()
if angles_after:
    print("  Gripper after close: %.1f deg" % angles_after[5])

print()
if angles_after and abs(angles_after[5] - 5) < 10:
    print("Gripper OK - opens and closes normally")
else:
    print("WARNING: Gripper may be damaged or stuck")

arm.disconnect()
