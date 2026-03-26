#!/usr/bin/env python3
"""Read-only: 로봇 현재 위치만 읽기. 움직임 명령 없음."""
import time, math, logging
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

# 읽기만! 움직임 없음!
angles = None
for _ in range(5):
    angles = arm.joints_angle_get()
    if angles and len(angles) == 6 and angles[0] != 180:
        break
    time.sleep(0.1)

pose = None
for _ in range(5):
    try:
        pose = arm.pose_get()
        if pose and len(pose) >= 3:
            break
    except Exception:
        time.sleep(0.1)

if angles:
    names = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]
    print("=== Current Robot State (READ ONLY) ===")
    for name, val in zip(names, angles):
        print("  %s: %.1f deg" % (name, val))
if pose:
    dist = math.sqrt(pose[0]**2 + pose[1]**2)
    print("FK: x=%.1f y=%.1f z=%.1f  dist=%.0fmm" % (pose[0], pose[1], pose[2], dist))

print("\nNOTE: No movement commands sent.")
arm.disconnect()
